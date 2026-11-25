"""Frozen four-color readout and consistency envelopes.

Image ingestion may inspect source pixels. Recall consumes projected features,
the fixed decoder,and ingestion metadata;it never consults source images.
"""

import hashlib
import hmac
import struct
from pathlib import Path
from . import png_codec

VERSION = 1
ENCODING = "four-color-row-major-2bit-v1"
DOMAIN = b"BrAIn-visual-readout-v1\x00"
ASSETS = Path(__file__).resolve().parent / "assets"
COLORS = ((230, 25, 25), (20, 170, 65), (25, 80, 230), (245, 210, 20))
SPEC = {"id": "ridge_overlap_0.001", "method": "ridge", "representation": "overlap", "alpha": 0.001}


def canonical(value):
    return png_codec.canonical_bytes(value)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def valid_hash(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def validate_profile(profile):
    required = {
        "version",
        "encoding",
        "image_size",
        "grid",
        "palette",
        "feature_tokens",
        "feature_width",
        "model_sha256",
        "decoder_spec",
    }
    if (
        not isinstance(profile, dict)
        or set(profile) != required
        or type(profile["version"]) is not int
        or profile["version"] != VERSION
    ):
        raise ValueError("unsupported_profile")
    if (
        profile["encoding"] != ENCODING
        or profile["image_size"] != [224, 224]
        or profile["grid"] != [4, 4]
        or profile["palette"] != [list(c) for c in COLORS]
        or profile["feature_tokens"] != 49
        or profile["feature_width"] != 2560
        or profile["decoder_spec"] != SPEC
        or not valid_hash(profile["model_sha256"])
    ):
        raise ValueError("profile_mismatch")


def profile_id(profile):
    validate_profile(profile)
    return sha(canonical(profile))


def make_profile(modelpath):
    profile = {
        "version": VERSION,
        "encoding": ENCODING,
        "image_size": [224, 224],
        "grid": [4, 4],
        "palette": [list(c) for c in COLORS],
        "feature_tokens": 49,
        "feature_width": 2560,
        "model_sha256": sha(Path(modelpath).read_bytes()),
        "decoder_spec": dict(SPEC),
    }
    validate_profile(profile)
    return profile


def pack_values(values):
    if (
        not isinstance(values, list)
        or len(values) != 16
        or any(type(v) is not int or not 0 <= v < 4 for v in values)
    ):
        raise ValueError("invalid_values")
    return bytes(
        sum(values[start + i] << (6 - 2 * i) for i in range(4)) for start in range(0, 16, 4)
    )


def unpack_values(payload):
    if len(payload) != 4:
        raise ValueError("invalid_payload")
    return [(byte >> shift) & 3 for byte in payload for shift in (6, 4, 2, 0)]


def make_manifest(values, profile):
    pid = profile_id(profile)
    return {
        "version": VERSION,
        "profile_sha256": pid,
        "payload_sha256": sha(DOMAIN + bytes.fromhex(pid) + pack_values(values)),
    }


def unavailable(reason):
    return {"status": "unavailable", "values": None, "reason": reason}


def verify_readout(values, manifest, profile):
    try:
        pid = profile_id(profile)
        payload = pack_values(values)
    except (ValueError, TypeError, KeyError):
        return unavailable("invalid_profile_or_prediction")
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"version", "profile_sha256", "payload_sha256"}
        or type(manifest.get("version")) is not int
        or manifest.get("version") != VERSION
    ):
        return unavailable("invalid_manifest")
    if manifest["profile_sha256"] != pid or not valid_hash(manifest["payload_sha256"]):
        return unavailable("manifest_profile_mismatch")
    if not hmac.compare_digest(
        sha(DOMAIN + bytes.fromhex(pid) + payload), manifest["payload_sha256"]
    ):
        return unavailable("checksum_mismatch")
    return {"status": "accepted", "values": list(values), "profile_sha256": pid}


def image_bytes(values):
    pack_values(values)
    return png_codec.rgb_png(
        224,
        224,
        b"".join(
            b"".join(bytes(COLORS[v]) * 56 for v in values[row * 4 : row * 4 + 4]) * 56
            for row in range(4)
        ),
    )


def source_values(image, profile):
    validate_profile(profile)
    width, height, rgb = png_codec.read_rgb_png(Path(image).read_bytes())
    if (width, height) != (224, 224):
        raise ValueError("unsupported_image_shape")
    inverse = {bytes(color): i for i, color in enumerate(COLORS)}
    values = []
    for row in range(4):
        for col in range(4):
            offset = (row * 56 * 224 + col * 56) * 3
            color = rgb[offset : offset + 3]
            if color not in inverse:
                raise ValueError("image_is_not_exact_color_grid")
            expected = color * 56
            for y in range(row * 56, (row + 1) * 56):
                start = (y * 224 + col * 56) * 3
                if rgb[start : start + 56 * 3] != expected:
                    raise ValueError("image_is_not_exact_color_grid")
            values.append(inverse[color])
    return values


def read_features(path):
    import numpy as np

    path = Path(path)
    if path.stat().st_size != 20 + 49 * 2560 * 4:
        raise ValueError("unexpected_feature_size")
    raw = path.read_bytes()
    magic, n, nx, ny, width = struct.unpack("<5i", raw[:20])
    if (magic, n, nx, ny, width) != (0x31455653, 49, 7, 7, 2560):
        raise ValueError("unexpected_feature_header")
    features = np.frombuffer(raw[20:], dtype="<f4").reshape(n, width).copy()
    if not np.isfinite(features).all():
        raise ValueError("nonfinite_features")
    return features, {"tokens": n, "nx": nx, "ny": ny, "width": width}


def load_model(modelpath, profile):
    import numpy as np

    validate_profile(profile)
    if sha(Path(modelpath).read_bytes()) != profile["model_sha256"]:
        raise ValueError("model_checksum_mismatch")
    with np.load(modelpath, allow_pickle=False) as loaded:
        model = {key: loaded[key] for key in loaded.files}
    shapes = {"mean": (16, 2560), "bias": (16, 4), "weights": (16, 2560, 4)}
    if set(model) != set(shapes) or any(
        model[k].shape != shape or not np.isfinite(model[k]).all() for k, shape in shapes.items()
    ):
        raise ValueError("invalid_decoder_model")
    return model


def overlap_weights():
    import numpy as np

    weights = np.zeros((16, 49), dtype=np.float64)
    for y in range(4):
        for x in range(4):
            for row in range(7):
                height = max(0, min((y + 1) * 56, (row + 1) * 32) - max(y * 56, row * 32))
                for col in range(7):
                    width = max(0, min((x + 1) * 56, (col + 1) * 32) - max(x * 56, col * 32))
                    weights[y * 4 + x, row * 7 + col] = width * height / (56 * 56)
    return weights


def predict_values(features, model, profile):
    import numpy as np

    validate_profile(profile)
    if features.shape != (49, 2560) or not np.isfinite(features).all():
        raise ValueError("invalid_features")
    inputs = (overlap_weights() @ features)[None]
    rows = np.asarray(inputs, dtype=np.float64)
    rows = rows / np.maximum(np.linalg.norm(rows, axis=-1, keepdims=True), 1e-12)
    scores = (
        np.einsum("spd,pdc->spc", rows - model["mean"], model["weights"], optimize=True)
        + model["bias"]
    )
    return scores.argmax(axis=2)[0].tolist()
