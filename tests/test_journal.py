import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from brain_pixels import journal as m


class RecoveryJournalV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def store(self, policy="one-cell"):
        store = m.GuardedMemoryStore.create(self.root / policy, policy=policy)
        entry = store.append("SET", "rescue scanner", "location", "blue equipment locker", 23)
        folder = store.folder / "blocks" / entry["block"]
        features = np.zeros((49, 2560), dtype="<f4")
        (folder / "features.pending.sve").write_bytes(
            struct.pack("<5i", 0x31455653, 49, 7, 7, 2560) + features.tobytes()
        )
        store.mark_ready(1)
        return store, m.encode_record("SET", "location", 2, 27, 1, 23)

    def test_persisted_policy_recovers_and_labels(self):
        store, values = self.store()
        bad = list(values)
        bad[0] = (bad[0] + 1) % 4
        original = Path.read_bytes

        def protected(path):
            if path.suffix == ".png":
                raise AssertionError("Source image forbidden")
            return original(path)

        with (
            patch.object(Path, "read_bytes", protected),
            patch.object(m.guard, "source_values", side_effect=AssertionError("Labels forbidden")),
            patch.object(m.guard, "predict_values", return_value=bad),
        ):
            result = m.GuardedMemoryStore(store.folder).recall("Where is the evening scanner?")
        self.assertEqual(result["answer"], "blue equipment locker")
        self.assertTrue(result["readout"]["records"][0]["recovered"])
        self.assertEqual(result["readout"]["records"][0]["candidate_checks"], 48)

    def test_verify_does_not_repair(self):
        store, values = self.store("verify")
        bad = list(values)
        bad[0] = (bad[0] + 1) % 4
        with patch.object(m.guard, "predict_values", return_value=bad):
            result = store.recall("Where is the rescue scanner?")
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["readout"]["records"][0]["recovered"])

    def test_two_cell_error_blocks_all_evidence(self):
        store, values = self.store()
        values[0] = (values[0] + 1) % 4
        values[1] = (values[1] + 1) % 4
        with patch.object(m.guard, "predict_values", return_value=values):
            result = store.recall("Where is the rescue scanner?")
        self.assertEqual(result["answer"], "UNKNOWN")
        self.assertEqual(result["context"]["sources"], [])
        self.assertEqual(result["context"]["facts"], [])

    def test_policy_tamper_fails_closed(self):
        store, _ = self.store()
        (store.folder / "policy.json").write_text(json.dumps(m.POLICIES["verify"]))
        self.assertEqual(store.recall("Where is the rescue scanner?")["status"], "unavailable")

    def test_pending_update_still_blocks(self):
        store, values = self.store()
        store.append("SET", "rescue scanner", "location", "padded transport crate", 25)
        with patch.object(m.guard, "predict_values", return_value=values):
            result = store.recall("Where is the rescue scanner?")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["context"]["facts"], [])

    def test_text_cannot_claim_recovery(self):
        with self.assertRaises(ValueError):
            m.GuardedMemoryStore.create(self.root / "text", backend="text", policy="one-cell")

    def test_two_cell_policy_recovers(self):
        store, values = self.store("two-cell")
        bad = list(values)
        for i in (0, 1):
            bad[i] = (bad[i] + 1) % 4
        with patch.object(m.guard, "predict_values", return_value=bad):
            result = m.GuardedMemoryStore(store.folder).recall("Where is the evening scanner?")
        self.assertEqual(result["answer"], "blue equipment locker")
        self.assertEqual(result["readout"]["records"][0]["distance"], 2)
        self.assertEqual(result["readout"]["records"][0]["candidate_checks"], 1128)

    def test_compact64_policy_recovers(self):
        store, values = self.store("compact-64")
        index, _, _ = store._read()
        metadata = m.read(store.folder / "blocks" / index["records"][0]["block"] / "readout.json")
        self.assertEqual(metadata["bits"], 64)
        bad = list(values)
        bad[0] = (bad[0] + 1) % 4
        bad[1] = (bad[1] + 1) % 4
        with patch.object(m.guard, "predict_values", return_value=bad):
            result = m.GuardedMemoryStore(store.folder).recall("Where is the evening scanner?")
        self.assertEqual(result["answer"], "blue equipment locker")
        self.assertTrue(result["readout"]["records"][0]["recovered"])


if __name__ == "__main__":
    unittest.main()
