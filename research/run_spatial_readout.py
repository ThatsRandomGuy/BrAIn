"""Preregister and run a local spatial-readout comparison with untouched test data.

Training features are explicitly supplied. Only the CPU vision encoder is started.
Neither the packaged production readout nor journal profile is replaced.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
from brain_pixels import integrity as g, recovery, spatial_readout as spatial
from brain_pixels.persistence import atomic
from brain_pixels.encoder import NativeEncoder

DOMAIN = b"BrAIn-spatial-readout-study-v1\x00"
SEED = 2026112502
VALIDATION = 64
TEST = 512


def save(path, value):
    atomic(path, g.canonical(value))


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def make_tag(values, plan_id):
    tag = hashlib.sha256(DOMAIN + bytes.fromhex(plan_id) + g.pack_values(values)).digest()[:8].hex()
    return {"version": 1, "plan_sha256": plan_id, "tag_bits": 64, "tag": tag}


def recover(values, metadata, plan_id):
    """Study-scoped consistency,not a receipt under the old production model ID."""
    try:
        packet = g.pack_values(values)
        if (
            not g.valid_hash(plan_id)
            or not isinstance(metadata, dict)
            or set(metadata) != {"version", "plan_sha256", "tag_bits", "tag"}
            or type(metadata["version"]) is not int
            or metadata["version"] != 1
            or metadata["plan_sha256"] != plan_id
            or type(metadata["tag_bits"]) is not int
            or metadata["tag_bits"] != 64
            or not isinstance(metadata["tag"], str)
            or len(metadata["tag"]) != 16
            or any(c not in "0123456789abcdef" for c in metadata["tag"])
        ):
            raise ValueError("invalid_study_metadata")
        context = hashlib.sha256(DOMAIN + bytes.fromhex(plan_id))
        digest = context.copy()
        digest.update(packet)
        if digest.digest()[:8].hex() == metadata["tag"]:
            return {
                "status": "accepted",
                "values": list(values),
                "recovered": False,
                "distance": 0,
                "candidate_checks": 0,
            }
        return recovery.search(values, 2, context, bytes.fromhex(metadata["tag"]))
    except (ValueError, KeyError, TypeError):
        return {
            **g.unavailable("invalid_study_input"),
            "recovered": False,
            "distance": None,
            "candidate_checks": 0,
        }


def generate_cases(seed, excluded):
    rng = random.Random(seed)
    seen = set(excluded)
    result = []
    for split, count in [("validation", VALIDATION), ("test", TEST)]:
        for index in range(count):
            while True:
                values = [rng.randrange(4) for _ in range(16)]
                matrix = np.array(values).reshape(4, 4)
                if all(
                    len(set(matrix[y : y + 2, x : x + 2].flat)) == 1
                    for y, x in ((0, 0), (0, 2), (2, 0), (2, 2))
                ):
                    continue
                digest = g.sha(g.image_bytes(values))
                if digest not in seen:
                    seen.add(digest)
                    break
            result.append(
                {
                    "id": f"{split}-{index:04d}",
                    "split": split,
                    "values": values,
                    "image_sha256": digest,
                }
            )
    return result


def paired_metrics(expected, baseline, selected):
    a = (np.asarray(baseline) == expected).all(axis=1)
    b = (np.asarray(selected) == expected).all(axis=1)
    improved = int((~a & b).sum())
    regressed = int((a & ~b).sum())
    discordant = improved + regressed
    p_value = (
        min(
            1.0,
            2
            * sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1))
            / (2**discordant),
        )
        if discordant
        else 1.0
    )
    return {
        "improved_images": improved,
        "regressed_images": regressed,
        "both_correct": int((a & b).sum()),
        "both_wrong": int((~a & ~b).sum()),
        "mcnemar_exact_two_sided_p": p_value,
        "accuracy_delta": float(b.mean() - a.mean()),
    }


def load_training(path, profile):
    doc = read(path)
    rows = doc["records"]
    if doc.get("version") != 1 or len(rows) != 512:
        raise ValueError("expected512_frozen_training_records")
    x = []
    labels = []
    public = []
    seen = set()
    for row in rows:
        features = Path(row["features"])
        image = Path(row["image"])
        if not features.is_absolute():
            features = path.parent / features
        if not image.is_absolute():
            image = path.parent / image
        digest = g.sha(image.read_bytes())
        feature_digest = g.sha(features.read_bytes())
        if (
            digest != row["image_sha256"]
            or feature_digest != row["feature_sha256"]
            or digest in seen
            or g.source_values(image, profile) != row["values"]
        ):
            raise ValueError("training_provenance_mismatch")
        seen.add(digest)
        x.append(g.read_features(features)[0])
        labels.append(row["values"])
        public.append({key: row[key] for key in ("id", "values", "image_sha256", "feature_sha256")})
    return np.stack(x), np.array(labels, dtype=int), public


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    configured = NativeEncoder(args.output.resolve() / "runtime", args.config)
    fingerprints = {
        "exe_sha256": file_sha256(configured.config["exe"]),
        "mmproj_sha256": file_sha256(configured.config["mmproj"]),
        "config_sha256": g.sha(configured.config_raw),
    }
    reference = read(g.ASSETS / "provenance.json")
    if any(fingerprints[k] != reference[k] for k in ("exe_sha256", "mmproj_sha256")):
        raise ValueError("encoder_changed_since_frozen_training;new training exports required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = read(g.ASSETS / "profile.json")
    baseline_model = g.load_model(g.ASSETS / "decoder.npz", profile)
    training, labels, training_rows = load_training(args.training_manifest.resolve(), profile)
    excluded = sorted(set(read(args.exclude)) | {row["image_sha256"] for row in training_rows})
    cases = generate_cases(args.seed, excluded)
    plan = {
        "version": 1,
        "encoder_fingerprints": fingerprints,
        "seed": args.seed,
        "training_images": 512,
        "training_metadata_sha256": g.sha(g.canonical(training_rows)),
        "baseline_model_sha256": profile["model_sha256"],
        "reference_feature_profile": profile,
        "representations": list(spatial.REPRESENTATIONS),
        "alphas": list(spatial.ALPHAS),
        "selection": "validation whole images,cells,fewer features,smaller alpha;no test features before selection",
        "validation_images": VALIDATION,
        "test_images": TEST,
        "max_recovery_candidates": 1128,
        "tag_bits": 64,
        "integrity": "Study-scoped envelope shared across decoders;not a production-profile receipt",
        "primary_gate": "Raw complete-image accuracy improves>=1 percentage point,exact two-sided McNemar p<=.05,cell accuracy not lower,correct accepted recovery coverage not lower,and zero wrong accepted streams. Report decoder byte cost.",
        "scope": "No production model replacement,language-model inference,context-saving or smaller-model claim.",
    }
    source_files = [
        "brain_pixels/spatial_readout.py",
        "research/run_spatial_readout.py",
        "research/audit_spatial_readout.py",
        "tests/test_spatial_readout.py",
    ]
    repo = Path(__file__).resolve().parents[1]
    sources = []
    for name in source_files:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / name).read_bytes())
        sources.append({"path": name, "sha256": g.sha(target.read_bytes())})
    plan["source_sha256"] = g.sha(g.canonical(sources))
    save(output / "source-manifest.json", sources)
    save(output / "training.json", training_rows)
    save(output / "excluded.json", excluded)
    save(output / "cases.json", cases)
    save(output / "plan.json", plan)
    plan_id = g.sha(g.canonical(plan))
    save(output / "plan-id.json", {"sha256": plan_id})
    for case in cases:
        folder = output / "inputs" / case["id"]
        folder.mkdir(parents=True)
        image = folder / "memory.png"
        image.write_bytes(g.image_bytes(case["values"]))
        save(folder / "readout.json", make_tag(g.source_values(image, profile), plan_id))
    print("PLAN_FROZEN " + str(output), flush=True)
    selection = None
    test_rows = []
    with NativeEncoder(output / "runtime", args.config) as encoder:
        validation = [case for case in cases if case["split"] == "validation"]
        val_features = []
        for index, case in enumerate(validation):
            folder = output / "inputs" / case["id"]
            features, _ = encoder.encode(folder / "memory.png", folder / "features.sve")
            val_features.append(features)
            if (index + 1) % 16 == 0:
                print(f"VALIDATION_ENCODED {index + 1}/{VALIDATION}", flush=True)
        val_features = np.stack(val_features)
        val_labels = np.array([case["values"] for case in validation])
        candidates = []
        parameters = {}

        def progress(name, position):
            if position % 4 == 0:
                print(f"TRAINED {name} positions={position}/16", flush=True)

        for representation in spatial.REPRESENTATIONS:
            models = spatial.fit_candidates(training, labels, representation, progress=progress)
            for alpha, model in models.items():
                prediction = spatial.predict(val_features, model, representation)
                row = {
                    "id": f"{representation}-{alpha:g}",
                    "representation": representation,
                    "alpha": alpha,
                    "width": spatial.token_indices(representation).shape[1] * 2560,
                    **spatial.scores(prediction, val_labels),
                }
                candidates.append(row)
                parameters[row["id"]] = model
        chosen = spatial.choose(candidates)
        assert not any(
            (output / "inputs" / case["id"] / "features.sve").exists()
            for case in cases
            if case["split"] == "test"
        )
        model = parameters[chosen["id"]]
        digest = spatial.save_model(output / "selected.npz", model)
        selection = {
            **chosen,
            "model_sha256": digest,
            "model_bytes": (output / "selected.npz").stat().st_size,
            "plan_sha256": plan_id,
        }
        save(output / "validation.json", candidates)
        save(output / "selection.json", selection)
        selection_raw = (output / "selection.json").read_bytes()
        print("SELECTION_FROZEN " + json.dumps(selection), flush=True)
        del parameters, models, training
        tests = [case for case in cases if case["split"] == "test"]
        for index, case in enumerate(tests):
            folder = output / "inputs" / case["id"]
            features, metadata = encoder.encode(folder / "memory.png", folder / "features.sve")
            baseline = g.predict_values(features, baseline_model, profile)
            candidate = spatial.predict(features[None], model, chosen["representation"])[0].tolist()
            manifest = read(folder / "readout.json")
            receipts = {
                key: recover(prediction, manifest, plan_id)
                for key, prediction in [("baseline", baseline), ("spatial", candidate)]
            }
            test_rows.append(
                {
                    "id": case["id"],
                    "baseline": baseline,
                    "spatial": candidate,
                    "receipts": receipts,
                    "feature_sha256": metadata["feature_sha256"],
                }
            )
            save(output / "test-results.json", test_rows)
            if (index + 1) % 32 == 0:
                print(f"TESTED {index + 1}/{TEST}", flush=True)
        assert (output / "selection.json").read_bytes() == selection_raw
    expected = np.array([case["values"] for case in tests])
    baseline = np.array([row["baseline"] for row in test_rows])
    candidate = np.array([row["spatial"] for row in test_rows])
    metrics = {
        "baseline": spatial.scores(baseline, expected),
        "spatial": spatial.scores(candidate, expected),
    }
    paired = paired_metrics(expected, baseline, candidate)
    repair = {}
    for name in metrics:
        accepted = [
            (row, case)
            for row, case in zip(test_rows, tests)
            if row["receipts"][name]["status"] == "accepted"
        ]
        wrong = sum(row["receipts"][name]["values"] != case["values"] for row, case in accepted)
        repair[name] = {
            "accepted": len(accepted),
            "correct": len(accepted) - wrong,
            "wrong_accepted": wrong,
            "unavailable": TEST - len(accepted),
            "recovered": sum(row["receipts"][name]["recovered"] for row in test_rows),
            "candidate_checks": sum(row["receipts"][name]["candidate_checks"] for row in test_rows),
        }
    gate = (
        paired["accuracy_delta"] >= 0.01
        and paired["mcnemar_exact_two_sided_p"] <= 0.05
        and metrics["spatial"]["correct_cells"] >= metrics["baseline"]["correct_cells"]
        and repair["spatial"]["correct"] >= repair["baseline"]["correct"]
        and not repair["baseline"]["wrong_accepted"]
        and not repair["spatial"]["wrong_accepted"]
    )
    summary = {
        "plan_sha256": plan_id,
        "selection": selection,
        "raw": metrics,
        "paired": paired,
        "recovery": repair,
        "primary_gate_passed": bool(gate),
        "overhead": {
            "baseline_decoder_bytes": (g.ASSETS / "decoder.npz").stat().st_size,
            "spatial_decoder_bytes": selection["model_bytes"],
            "vision_positions_per_image": 49,
            "source_payload_bytes": 4,
            "checksum_bytes": 8,
            "manifest_bytes": len(
                g.canonical(read(output / "inputs" / tests[0]["id"] / "readout.json"))
            ),
        },
        "encoder": encoder.state,
        "scope": plan["scope"],
    }
    save(output / "summary.json", summary)
    print("SUMMARY " + json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
