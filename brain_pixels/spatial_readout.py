"""Experimental spatially concatenated ridge readout.

This module is deliberately separate from the journal's frozen default decoder.
It preserves local projected-token arrangement instead of averaging tokens.
"""

from pathlib import Path
import numpy as np
from . import integrity as g

REPRESENTATIONS = ("concat4", "concat9")
ALPHAS = (0.001, 0.01, 0.1)


def token_indices(representation):
    """Fixed row-major neighborhoods;selection never depends on image values."""
    if representation not in REPRESENTATIONS:
        raise ValueError("unsupported_representation")
    result = []
    for y in range(4):
        for x in range(4):
            if representation == "concat4":
                rows = sorted(
                    sorted(range(7), key=lambda r: (abs((r + 0.5) * 32 - (y + 0.5) * 56), r))[:2]
                )
                cols = sorted(
                    sorted(range(7), key=lambda c: (abs((c + 0.5) * 32 - (x + 0.5) * 56), c))[:2]
                )
            else:
                center_y = min(6, int((y + 0.5) * 7 / 4))
                center_x = min(6, int((x + 0.5) * 7 / 4))
                start_y = max(0, min(4, center_y - 1))
                start_x = max(0, min(4, center_x - 1))
                rows = range(start_y, start_y + 3)
                cols = range(start_x, start_x + 3)
            result.append([row * 7 + col for row in rows for col in cols])
    return np.asarray(result, dtype=np.int64)


def validate_features(features):
    values = np.asarray(features)
    if values.ndim != 3 or values.shape[1:] != (49, 2560) or not np.isfinite(values).all():
        raise ValueError("expected_finite_batch_n_49_2560")
    return values


def position_samples(features, indices, position):
    rows = np.asarray(
        features[:, indices[position], :].reshape(len(features), -1), dtype=np.float64
    )
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


def fit_candidates(features, labels, representation, alphas=ALPHAS, progress=None):
    """Fit on training data only,using one shared Gram matrix per position."""
    features = validate_features(features)
    labels = np.asarray(labels)
    if (
        labels.shape != (len(features), 16)
        or labels.dtype.kind not in "iu"
        or np.any((labels < 0) | (labels > 3))
        or not len(features)
    ):
        raise ValueError("invalid_training_labels")
    alphas = tuple(alphas)
    if (
        not alphas
        or len(set(alphas)) != len(alphas)
        or any(
            not isinstance(a, (int, float)) or isinstance(a, bool) or not np.isfinite(a) or a <= 0
            for a in alphas
        )
    ):
        raise ValueError("invalid_alphas")
    indices = token_indices(representation)
    width = indices.shape[1] * 2560
    models = {
        alpha: {
            "mean": np.empty((16, width)),
            "bias": np.empty((16, 4)),
            "weights": np.empty((16, width, 4)),
        }
        for alpha in alphas
    }
    for position in range(16):
        if set(labels[:, position].tolist()) != set(range(4)):
            raise ValueError("training_class_missing_at_position")
        rows = position_samples(features, indices, position)
        mean = rows.mean(axis=0)
        target = np.eye(4)[labels[:, position]]
        bias = target.mean(axis=0)
        centered = rows - mean
        gram = centered @ centered.T
        for alpha, model in models.items():
            coefficient = np.linalg.solve(gram + alpha * np.eye(len(rows)), target - bias)
            model["mean"][position] = mean
            model["bias"][position] = bias
            model["weights"][position] = centered.T @ coefficient
        if progress is not None:
            progress(representation, position + 1)
    return models


def predict(features, model, representation):
    features = validate_features(features)
    indices = token_indices(representation)
    width = indices.shape[1] * 2560
    validate_model(model, width)
    values = []
    for position in range(16):
        rows = position_samples(features, indices, position)
        scores = (rows - model["mean"][position]) @ model["weights"][position] + model["bias"][
            position
        ]
        values.append(scores.argmax(axis=1))
    return np.stack(values, axis=1).astype(int)


def validate_model(model, width):
    shapes = {"mean": (16, width), "bias": (16, 4), "weights": (16, width, 4)}
    if set(model) != set(shapes) or any(
        model[key].shape != shape or not np.isfinite(model[key]).all()
        for key, shape in shapes.items()
    ):
        raise ValueError("invalid_spatial_readout")


def save_model(path, model):
    path = Path(path)
    with path.open("xb") as stream:
        np.savez(stream, **model)
    return g.sha(path.read_bytes())


def load_model(path, representation, expected_sha256):
    path = Path(path)
    if not g.valid_hash(expected_sha256) or g.sha(path.read_bytes()) != expected_sha256:
        raise ValueError("spatial_model_checksum_mismatch")
    with np.load(path, allow_pickle=False) as saved:
        model = {key: saved[key] for key in saved.files}
    validate_model(model, token_indices(representation).shape[1] * 2560)
    return model


def scores(prediction, expected):
    prediction = np.asarray(prediction)
    expected = np.asarray(expected)
    if prediction.shape != expected.shape or prediction.ndim != 2 or prediction.shape[1] != 16:
        raise ValueError("invalid_metric_shape")
    same = prediction == expected
    return {
        "images": len(expected),
        "correct_images": int(same.all(axis=1).sum()),
        "correct_cells": int(same.sum()),
        "total_cells": int(same.size),
    }


def choose(validation_rows):
    """Whole images,cells,fewer parameters,then smaller alpha;test is absent."""
    if not validation_rows:
        raise ValueError("no_validation_candidates")
    return min(
        validation_rows,
        key=lambda row: (-row["correct_images"], -row["correct_cells"], row["width"], row["alpha"]),
    )
