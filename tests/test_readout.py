"""Regression fixtures from real native vision exports,not mocked predictions."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch
from brain_pixels import integrity as g, recovery

FIXTURES = Path(__file__).parent / "fixtures"


class FrozenReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads((g.ASSETS / "profile.json").read_text())
        cls.model = g.load_model(g.ASSETS / "decoder.npz", cls.profile)
        cls.cases = json.loads((FIXTURES / "readout.json").read_text())

    def test_real_predictions_and_recovery_boundaries(self):
        for case in self.cases:
            with self.subTest(case=case["name"]):
                features, _ = g.read_features(FIXTURES / (case["name"] + ".sve"))
                prediction = g.predict_values(features, self.model, self.profile)
                self.assertEqual(prediction, case["prediction"])
                receipt = recovery.recover(
                    prediction, g.make_manifest(case["values"], self.profile), self.profile, 2
                )
                self.assertEqual(receipt["status"], case["expected_status"])
                if receipt["status"] == "accepted":
                    self.assertEqual(receipt["values"], case["values"])
                else:
                    self.assertIsNone(receipt["values"])

    def test_feature_and_image_provenance(self):
        for case in self.cases:
            self.assertEqual(
                g.sha((FIXTURES / (case["name"] + ".sve")).read_bytes()), case["features_sha256"]
            )
            self.assertEqual(
                g.sha((FIXTURES / (case["name"] + ".png")).read_bytes()), case["png_sha256"]
            )
            self.assertEqual(
                g.source_values(FIXTURES / (case["name"] + ".png"), self.profile), case["values"]
            )

    def test_recall_does_not_read_source_image(self):
        case = self.cases[1]
        original = Path.read_bytes

        def protected(path):
            if path.suffix == ".png":
                raise AssertionError("source image read during recall")
            return original(path)

        with (
            patch.object(Path, "read_bytes", protected),
            patch.object(g, "source_values", side_effect=AssertionError("source labels forbidden")),
        ):
            features, _ = g.read_features(FIXTURES / (case["name"] + ".sve"))
            receipt = recovery.recover(
                g.predict_values(features, self.model, self.profile),
                g.make_manifest(case["values"], self.profile),
                self.profile,
                2,
            )
        self.assertEqual(receipt["status"], "accepted")

    def test_short_tag_false_accept_is_preserved(self):
        example = json.loads(
            (FIXTURES / "short-tag-collision.json").read_text()
        )  # Source is the first ordinary test case.
        rows = json.loads((FIXTURES / "readout.json").read_text())
        values = rows[0]["values"]
        tag = recovery.make_tag(values, self.profile, 16)
        receipt = recovery.recover_tag(example["bad"], tag, self.profile, 2)
        self.assertEqual(receipt["status"], "accepted")
        self.assertNotEqual(receipt["values"], values)


if __name__ == "__main__":
    unittest.main()
