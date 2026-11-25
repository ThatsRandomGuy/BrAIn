import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from brain_pixels import recovery as r
from brain_pixels import journal as m


class RecoveryV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = r.g.make_profile(r.g.ASSETS / "decoder.npz")
        cls.values = [i % 4 for i in range(16)]
        cls.manifest = r.g.make_manifest(cls.values, cls.profile)

    def test_candidate_bounds_and_distances(self):
        for radius, count in [(0, 0), (1, 48), (2, 1128)]:
            rows = list(r.candidates(self.values, radius))
            self.assertEqual(len(rows), count)
            self.assertEqual(len({p for p, _ in rows}), count)
            for packet, positions in rows:
                self.assertEqual(
                    sum(a != b for a, b in zip(r.g.unpack_values(packet), self.values)),
                    len(positions),
                )

    def test_all_two_cell_positive_controls(self):
        # All120position pairs,one changed value per pair;full alternatives tested in evaluation.
        for a in range(16):
            for b in range(a + 1, 16):
                bad = list(self.values)
                bad[a] = (bad[a] + 1) % 4
                bad[b] = (bad[b] + 1) % 4
                receipt = r.recover(bad, self.manifest, self.profile)
                self.assertEqual(receipt["values"], self.values)
                self.assertEqual(receipt["distance"], 2)
                self.assertEqual(receipt["candidate_checks"], 1128)

    def test_three_cell_outside_radius(self):
        bad = list(self.values)
        for i in (0, 4, 8):
            bad[i] = (bad[i] + 1) % 4
        receipt = r.recover(bad, self.manifest, self.profile)
        self.assertEqual(receipt["status"], "unavailable")
        self.assertIsNone(receipt["values"])

    def test_invalid_inputs_fail_closed(self):
        for metadata in (None, {}, dict(self.manifest, version=True)):
            self.assertEqual(r.recover(self.values, metadata, self.profile)["candidate_checks"], 0)
        self.assertEqual(
            r.recover(self.values, self.manifest, self.profile, radius=True)["status"],
            "unavailable",
        )

    def test_compact_tags_and_validation(self):
        for bits in r.TAG_BITS:
            tag = r.make_tag(self.values, self.profile, bits)
            self.assertEqual(len(bytes.fromhex(tag["tag"])), bits // 8)
            self.assertEqual(r.recover_tag(self.values, tag, self.profile)["values"], self.values)
        for tag in (None, {}, dict(r.make_tag(self.values, self.profile, 64), bits=True)):
            self.assertEqual(r.recover_tag(self.values, tag, self.profile)["status"], "unavailable")

    def test_no_source_access(self):
        with (
            patch.object(r.g, "source_values", side_effect=AssertionError("forbidden")),
            patch.object(Path, "read_bytes", side_effect=AssertionError("forbidden")),
        ):
            bad = list(self.values)
            bad[0] = (bad[0] + 1) % 4
            bad[1] = (bad[1] + 1) % 4
            self.assertEqual(r.recover(bad, self.manifest, self.profile)["values"], self.values)

    def test_policy_saved_and_inspected(self):
        with tempfile.TemporaryDirectory() as folder:
            store = m.GuardedMemoryStore.create(Path(folder) / "store", policy="two-cell")
            self.assertEqual(
                m.GuardedMemoryStore(store.folder).inspect()["readout_policy"],
                m.POLICIES["two-cell"],
            )

    def test_one_cell_recovery_for_every_single_cell_fault(self):
        for packet, _ in r.candidates(self.values, 1):
            result = r.recover(r.g.unpack_values(packet), self.manifest, self.profile, 1)
            self.assertEqual(result["values"], self.values)
            self.assertEqual(result["candidate_checks"], 48)


if __name__ == "__main__":
    unittest.main()
