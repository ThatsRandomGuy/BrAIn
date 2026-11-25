"""Audit frozen generation,validation choice and fresh spatial-readout results."""

import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from brain_pixels import integrity as g, spatial_readout as spatial
from . import run_spatial_readout as study


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def independently_recover(values, manifest, plan_id):
    context = hashlib.sha256(study.DOMAIN + bytes.fromhex(plan_id))
    expected = bytes.fromhex(manifest["tag"])
    base = int.from_bytes(g.pack_values(values), "big")

    def match(packet):
        digest = context.copy()
        digest.update(packet)
        return digest.digest()[:8] == expected

    if match(base.to_bytes(4, "big")):
        return values
    masks = [
        [(values[p] ^ color) << (30 - 2 * p) for color in range(4) if color != values[p]]
        for p in range(16)
    ]
    found = []
    for options in masks:
        for mask in options:
            packet = (base ^ mask).to_bytes(4, "big")
            if match(packet):
                found.append(packet)
    for a in range(16):
        for b in range(a + 1, 16):
            for x in masks[a]:
                for y in masks[b]:
                    packet = (base ^ x ^ y).to_bytes(4, "big")
                    if match(packet):
                        found.append(packet)
    return g.unpack_values(found[0]) if len(found) == 1 else None


def audit(output):
    output = output.resolve()
    plan = read(output / "plan.json")
    plan_id = g.sha(g.canonical(plan))
    assert plan_id == read(output / "plan-id.json")["sha256"]
    cases = read(output / "cases.json")
    assert cases == study.generate_cases(plan["seed"], read(output / "excluded.json"))
    training = read(output / "training.json")
    assert len(training) == 512 and g.sha(g.canonical(training)) == plan["training_metadata_sha256"]
    assert {c["image_sha256"] for c in cases}.isdisjoint(row["image_sha256"] for row in training)
    for source in read(output / "source-manifest.json"):
        assert g.sha((output / "source" / source["path"]).read_bytes()) == source["sha256"]
    choice = spatial.choose(read(output / "validation.json"))
    selection = read(output / "selection.json")
    assert all(selection[k] == v for k, v in choice.items())
    assert selection["plan_sha256"] == plan_id
    model = spatial.load_model(
        output / "selected.npz", selection["representation"], selection["model_sha256"]
    )
    profile = plan["reference_feature_profile"]
    baseline_model = g.load_model(g.ASSETS / "decoder.npz", profile)
    val = [c for c in cases if c["split"] == "validation"]
    x = np.stack([g.read_features(output / "inputs" / c["id"] / "features.sve")[0] for c in val])
    pred = spatial.predict(x, model, selection["representation"])
    assert all(
        selection[k] == v
        for k, v in spatial.scores(pred, np.array([c["values"] for c in val])).items()
    )
    tests = [c for c in cases if c["split"] == "test"]
    rows = read(output / "test-results.json")
    assert len(rows) == study.TEST
    summary = read(output / "summary.json")
    predictions = {name: [] for name in ("baseline", "spatial")}
    for case, row in zip(tests, rows, strict=True):
        folder = output / "inputs" / case["id"]
        assert row["id"] == case["id"]
        assert g.sha((folder / "memory.png").read_bytes()) == case["image_sha256"]
        assert g.source_values(folder / "memory.png", profile) == case["values"]
        features, _ = g.read_features(folder / "features.sve")
        assert g.sha((folder / "features.sve").read_bytes()) == row["feature_sha256"]
        assert row["baseline"] == g.predict_values(features, baseline_model, profile)
        assert (
            row["spatial"]
            == spatial.predict(features[None], model, selection["representation"])[0].tolist()
        )
        manifest = read(folder / "readout.json")
        assert manifest == study.make_tag(case["values"], plan_id)
        for name in predictions:
            predictions[name].append(row[name])
            assert row["receipts"][name] == study.recover(row[name], manifest, plan_id)
            assert row["receipts"][name]["values"] == independently_recover(
                row[name], manifest, plan_id
            )
    expected = np.array([c["values"] for c in tests])
    for name, values in predictions.items():
        assert summary["raw"][name] == spatial.scores(np.array(values), expected)
    assert summary["paired"] == study.paired_metrics(
        expected, np.array(predictions["baseline"]), np.array(predictions["spatial"])
    )
    for name in predictions:
        accepted = [
            (row, case)
            for row, case in zip(rows, tests)
            if row["receipts"][name]["status"] == "accepted"
        ]
        wrong = sum(row["receipts"][name]["values"] != case["values"] for row, case in accepted)
        assert summary["recovery"][name] == {
            "accepted": len(accepted),
            "correct": len(accepted) - wrong,
            "wrong_accepted": wrong,
            "unavailable": study.TEST - len(accepted),
            "recovered": sum(row["receipts"][name]["recovered"] for row in rows),
            "candidate_checks": sum(row["receipts"][name]["candidate_checks"] for row in rows),
        }
    paired = summary["paired"]
    metrics = summary["raw"]
    repair = summary["recovery"]
    expected_gate = (
        paired["accuracy_delta"] >= 0.01
        and paired["mcnemar_exact_two_sided_p"] <= 0.05
        and metrics["spatial"]["correct_cells"] >= metrics["baseline"]["correct_cells"]
        and repair["spatial"]["correct"] >= repair["baseline"]["correct"]
        and not repair["baseline"]["wrong_accepted"]
        and not repair["spatial"]["wrong_accepted"]
    )
    assert summary["primary_gate_passed"] == expected_gate
    assert summary["overhead"]["spatial_decoder_bytes"] == (output / "selected.npz").stat().st_size
    assert (
        summary["overhead"]["baseline_decoder_bytes"] == (g.ASSETS / "decoder.npz").stat().st_size
    )
    assert (
        summary["encoder"]["stopped"]
        and summary["encoder"]["exit_code"] == 0
        and summary["encoder"]["config_unchanged"]
    )
    assert read(output / "runtime/lifecycle.json") == summary["encoder"]
    result = {
        "passed": True,
        "training_images": 512,
        "fresh_validation_images": study.VALIDATION,
        "fresh_test_images": study.TEST,
        "selection_replayed": True,
        "predictions_replayed": True,
        "encoder_stopped": True,
        "primary_gate_passed": summary["primary_gate_passed"],
    }
    study.save(output / "audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    audit(Path(sys.argv[1]))
