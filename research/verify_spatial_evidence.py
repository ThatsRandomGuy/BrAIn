"""Replay public spatial-study records without claiming a new native-model run."""

import argparse
import json
from pathlib import Path
import numpy as np
from brain_pixels import integrity as g, spatial_readout as spatial
from . import run_spatial_readout as study


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify(folder):
    plan = read(folder / "plan.json")
    plan_id = g.sha(g.canonical(plan))
    summary = read(folder / "summary.json")
    assert summary["plan_sha256"] == plan_id
    cases = read(folder / "cases.json")
    excluded = read(folder / "excluded.json")
    assert cases == study.generate_cases(plan["seed"], excluded)
    training = read(folder / "training.json")
    assert g.sha(g.canonical(training)) == plan["training_metadata_sha256"] and len(training) == 512
    old = read(folder / "interrupted-attempt.json")
    old_hashes = set(old["source_image_hashes"])
    assert len(old_hashes) == 576 and old_hashes.issubset(excluded)
    assert old_hashes.isdisjoint(case["image_sha256"] for case in cases)
    choice = spatial.choose(read(folder / "validation.json"))
    selection = read(folder / "selection.json")
    assert all(selection[k] == v for k, v in choice.items())
    assert selection["plan_sha256"] == plan_id
    tests = [case for case in cases if case["split"] == "test"]
    rows = read(folder / "predictions.json")
    assert len(rows) == len(tests) == 512
    expected = np.array([case["values"] for case in tests])
    predictions = {name: np.array([row[name] for row in rows]) for name in ("baseline", "spatial")}
    for case, row in zip(tests, rows, strict=True):
        assert (
            case["id"] == row["id"] and g.sha(g.image_bytes(case["values"])) == case["image_sha256"]
        )
    repair = {}
    for name, predicted in predictions.items():
        assert summary["raw"][name] == spatial.scores(predicted, expected)
        counts = {
            "accepted": 0,
            "correct": 0,
            "wrong_accepted": 0,
            "unavailable": 0,
            "recovered": 0,
            "candidate_checks": 0,
        }
        for case, row in zip(tests, rows):
            receipt = study.recover(row[name], study.make_tag(case["values"], plan_id), plan_id)
            accepted = receipt["status"] == "accepted"
            counts["accepted"] += accepted
            counts["correct"] += accepted and receipt["values"] == case["values"]
            counts["wrong_accepted"] += accepted and receipt["values"] != case["values"]
            counts["unavailable"] += not accepted
            counts["recovered"] += receipt["recovered"]
            counts["candidate_checks"] += receipt["candidate_checks"]
        assert counts == summary["recovery"][name]
        repair[name] = counts
    paired = study.paired_metrics(expected, predictions["baseline"], predictions["spatial"])
    assert paired == summary["paired"]
    metrics = summary["raw"]
    gate = (
        paired["accuracy_delta"] >= 0.01
        and paired["mcnemar_exact_two_sided_p"] <= 0.05
        and metrics["spatial"]["correct_cells"] >= metrics["baseline"]["correct_cells"]
        and repair["spatial"]["correct"] >= repair["baseline"]["correct"]
        and not repair["baseline"]["wrong_accepted"]
        and not repair["spatial"]["wrong_accepted"]
    )
    assert bool(gate) == summary["primary_gate_passed"]
    return {
        "passed": True,
        "scope": "Recorded evidence replay,not a new encoder export,training run,or selected-model prediction audit.",
        "fresh_validation_images": 64,
        "fresh_test_images": 512,
        "interrupted_cases_excluded": 576,
        "primary_gate_passed": bool(gate),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.folder), indent=2))
