"""Current semantics retained from the earlier journal prototypes."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from brain_pixels import journal as m


class MemoryContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def store(self, name="store"):
        return m.GuardedMemoryStore.create(self.root / name, "text")

    def record(self, store, op="SET", relation="location", value="blue equipment locker"):
        entry = store.append(op, "rescue scanner", relation, value, 23)
        store.mark_ready(entry["id"])
        return entry

    def test_uncertainty_and_forgetting_do_not_change_other_slots(self):
        store = self.store()
        self.record(store)
        self.record(store, relation="owner", value="Mira Chen")
        self.record(store, "MAYBE", value="padded transport crate")
        self.assertEqual(store.recall("Where is the evening scanner?")["answer"], "UNKNOWN")
        self.assertEqual(store.recall("Who owns the evening scanner?")["answer"], "Mira Chen")
        self.record(store, value="padded transport crate")
        self.record(store, "FORGET", value=None)
        self.assertEqual(store.recall("Where is the rescue scanner?")["answer"], "UNKNOWN")
        self.assertEqual(store.recall("Who owns the rescue scanner?")["answer"], "Mira Chen")

    def test_missing_old_record_cannot_be_skipped(self):
        store = self.store()
        first = self.record(store)
        self.record(store, value="padded transport crate")
        marker = store.folder / "blocks" / first["block"] / "ready.json"
        held = marker.with_suffix(".held")
        os.replace(marker, held)
        try:
            result = store.recall("Where is the rescue scanner?")
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["context"]["sources"], [])
            self.assertEqual(result["context"]["facts"], [])
        finally:
            os.replace(held, marker)

    def test_invalid_input_cannot_commit(self):
        store = self.store()
        head = (store.folder / "HEAD.json").read_bytes()
        for args in [
            ("SET", "unknown", "location", "blue equipment locker"),
            ("SET", "rescue scanner", "owner", "blue equipment locker"),
            ("SET", "rescue scanner", "location", None),
        ]:
            with self.assertRaises(ValueError):
                store.append(*args)
        self.assertEqual((store.folder / "HEAD.json").read_bytes(), head)

    def test_payload_corruption_and_vocabulary_tampering_block_recall(self):
        store = self.store()
        entry = self.record(store)
        (store.folder / "blocks" / entry["block"] / "packet.hex").write_text("0" * 8)
        self.assertEqual(store.recall("Where is the rescue scanner?")["status"], "unavailable")
        other = self.store("other")
        self.record(other)
        book = m.read(other.folder / "codebook.json")
        book["subjects"]["2"] = "different object"
        (other.folder / "codebook.json").write_text(json.dumps(book))
        self.assertEqual(other.recall("Where is the rescue scanner?")["status"], "unavailable")

    def test_sequence_binding_and_packet_limits(self):
        for sequence in (1, 255):
            values = m.encode_record("SET", "location", 2, 27, sequence, 23)
            self.assertEqual(
                m.parse_record(values, sequence, m.DEFAULT_BOOK)["value"], "blue equipment locker"
            )
            with self.assertRaises(ValueError):
                m.parse_record(values, 2 if sequence == 1 else 254, m.DEFAULT_BOOK)
        for sequence in (0, 256, True):
            with self.assertRaises(ValueError):
                m.encode_record("SET", "location", 2, 27, sequence, 23)

    def test_manifest_cannot_smuggle_expected_values(self):
        profile = m.read(m.BUNDLE / "profile.json")
        values = [0, 1, 2, 3] * 4
        manifest = m.guard.make_manifest(values, profile)
        for bad in (
            None,
            {},
            dict(manifest, expected=values),
            dict(manifest, version=True),
            dict(manifest, payload_sha256="x" * 64),
        ):
            result = m.guard.verify_readout(values, bad, profile)
            self.assertEqual(result["status"], "unavailable")
            self.assertIsNone(result["values"])


if __name__ == "__main__":
    unittest.main()
