import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research import run_spatial_readout as study
from brain_pixels import integrity as g


class SpatialStudyProtocolTests(unittest.TestCase):
    def test_exact_paired_test_and_no_difference(self):
        expected = np.zeros((8, 16), dtype=int)
        bad = np.ones((8, 16), dtype=int)
        result = study.paired_metrics(expected, bad, expected)
        self.assertEqual(result["improved_images"], 8)
        self.assertEqual(result["mcnemar_exact_two_sided_p"], 0.0078125)
        self.assertEqual(study.paired_metrics(expected, bad, bad)["mcnemar_exact_two_sided_p"], 1.0)

    def test_recovery_envelope_is_strict_and_bounded(self):
        plan = "1" * 64
        values = [0, 1, 2, 3] * 4
        manifest = study.make_tag(values, plan)
        bad = list(values)
        bad[0] = 1
        bad[2] = 0
        receipt = study.recover(bad, manifest, plan)
        self.assertEqual(receipt["values"], values)
        self.assertEqual(receipt["candidate_checks"], 1128)
        for metadata in (
            None,
            {},
            dict(manifest, version=True),
            dict(manifest, expected=values),
            dict(manifest, plan_sha256="2" * 64),
        ):
            self.assertEqual(study.recover(values, metadata, plan)["status"], "unavailable")
        bad[5] = 2
        self.assertEqual(study.recover(bad, manifest, plan)["status"], "unavailable")

    def test_failed_checkpoint_replace_preserves_previous_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            study.save(path, {"completed": 1})
            with patch(
                "brain_pixels.persistence.os.replace", side_effect=OSError("synthetic failure")
            ):
                with self.assertRaises(OSError):
                    study.save(path, {"completed": 2})
            self.assertEqual(study.read(path), {"completed": 1})
            self.assertEqual(list(Path(temporary).iterdir()), [path])

    def test_sources_are_deterministic_unique_and_excluded(self):
        excluded = {g.sha(g.image_bytes([0] * 16))}
        a = study.generate_cases(2026112501, excluded)
        b = study.generate_cases(2026112501, excluded)
        self.assertEqual(a, b)
        digests = {case["image_sha256"] for case in a}
        self.assertEqual(len(digests), 576)
        self.assertFalse(digests.intersection(excluded))
        self.assertEqual(sum(case["split"] == "test" for case in a), 512)


if __name__ == "__main__":
    unittest.main()
