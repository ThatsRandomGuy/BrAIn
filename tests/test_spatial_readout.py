import tempfile
import unittest
from pathlib import Path
import numpy as np
from brain_pixels import spatial_readout as s


class SpatialReadoutTests(unittest.TestCase):
    def test_fixed_bounded_neighborhoods(self):
        for name, count in [("concat4", 4), ("concat9", 9)]:
            indices = s.token_indices(name)
            self.assertEqual(indices.shape, (16, count))
            self.assertTrue(np.all((indices >= 0) & (indices < 49)))
            for row in indices:
                self.assertEqual(len(set(row.tolist())), count)
                self.assertEqual(row.tolist(), sorted(row.tolist()))
        with self.assertRaises(ValueError):
            s.token_indices("unknown")

    def test_learns_independent_heldout_color_signal(self):
        rng = np.random.default_rng(1125)
        training = rng.integers(0, 4, (48, 16))
        heldout = rng.integers(0, 4, (20, 16))

        def features(labels):
            x = rng.normal(0, 0.001, (len(labels), 49, 2560)).astype("float32")
            # Independent channels at all49tokens expose each cell's color.
            # No test labels enter fit_candidates.
            for p in range(16):
                for color in range(4):
                    x[:, :, p * 4 + color] += (labels[:, p] == color)[:, None]
            return x

        train_x = features(training)
        test_x = features(heldout)
        model = s.fit_candidates(train_x, training, "concat4", alphas=(0.001,))[0.001]
        prediction = s.predict(test_x, model, "concat4")
        self.assertGreaterEqual(s.scores(prediction, heldout)["correct_cells"], 310)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decoder.npz"
            digest = s.save_model(path, model)
            loaded = s.load_model(path, "concat4", digest)
            np.testing.assert_array_equal(prediction, s.predict(test_x, loaded, "concat4"))
            path.write_bytes(path.read_bytes()[:-1] + b"x")
            with self.assertRaises(ValueError):
                s.load_model(path, "concat4", digest)

    def test_invalid_shapes_and_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            s.validate_features(np.zeros((1, 48, 2560)))
        with self.assertRaises(ValueError):
            s.fit_candidates(np.zeros((4, 49, 2560)), np.zeros((4, 16), dtype=int), "concat4")
        with self.assertRaises(ValueError):
            s.fit_candidates(
                np.zeros((4, 49, 2560)), np.zeros((4, 16), dtype=int), "concat4", alphas=(0,)
            )

    def test_selection_uses_whole_images_then_cells_and_cost(self):
        rows = [
            {"correct_images": 9, "correct_cells": 159, "width": 23040, "alpha": 0.001},
            {"correct_images": 10, "correct_cells": 160, "width": 23040, "alpha": 0.1},
            {"correct_images": 10, "correct_cells": 160, "width": 10240, "alpha": 0.01},
        ]
        self.assertEqual(s.choose(rows), rows[2])


if __name__ == "__main__":
    unittest.main()
