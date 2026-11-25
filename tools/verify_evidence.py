"""Replay the published result aggregates from recorded predictions,without a model.

This checks the compact public evidence,not the full native feature exports.
Three real exports separately exercise the learned decoder in tests/test_readout.py.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain_pixels import integrity as g, recovery  # noqa: E402

DATA = ROOT / "docs/evidence/2026-11-05"


def read(name):
    return json.loads((DATA / name).read_text())


def verify():
    profile = json.loads((g.ASSETS / "profile.json").read_text())
    cases = read("images-evaluation-cases.json")
    rows = read("image-predictions.json")
    summary = read("images-summary.json")
    conditions = {"verify": 0, "one": 1, "two": 2, "compact64": 2}
    counts = {key: {"accepted": 0, "correct": 0, "wrong": 0} for key in conditions}
    for case, row in zip(cases, rows, strict=True):
        assert case["id"] == row["id"]
        assert g.sha(g.image_bytes(case["colors"])) == case["image_sha256"]
        for condition, radius in conditions.items():
            manifest = (
                recovery.make_tag(case["colors"], profile, 64)
                if condition == "compact64"
                else g.make_manifest(case["colors"], profile)
            )
            receipt = (
                recovery.recover_tag(row["prediction"], manifest, profile, radius)
                if condition == "compact64"
                else recovery.recover(row["prediction"], manifest, profile, radius)
            )
            if receipt["status"] == "accepted":
                counts[condition]["accepted"] += 1
                counts[condition]["correct"] += receipt["values"] == case["colors"]
                counts[condition]["wrong"] += receipt["values"] != case["colors"]
    assert len(cases) == 512
    for condition, c in counts.items():
        expected = summary["conditions"][condition]
        assert (c["accepted"], c["correct"], c["wrong"]) == (
            expected["accepted"],
            expected["correct_accepted"],
            expected["wrong_accepted"],
        )
    histories = read("histories-evaluation-cases.json")
    frames = read("history-predictions.json")
    queries = read("history-query-results.json")
    hsummary = read("histories-summary.json")
    actions = {(case["id"], a["sequence"]): a for case in histories for a in case["actions"]}
    book = read("histories-settings.json")["codebook"]
    by_depth = {}
    from brain_pixels.journal import encode_record, subject_id, value_id

    for f in frames:
        a = actions[f["case"], f["depth"]]
        symbols = encode_record(
            a["op"],
            a["relation"],
            subject_id(book, a["subject"]),
            0 if a["op"] == "FORGET" else value_id(book, a["value"], a["relation"]),
            a["sequence"],
            a["nonce"],
        )
        assert symbols == f["symbols"]
        assert g.sha(g.image_bytes(symbols)) == a["image_sha256"]
        by_depth[f["case"], f["depth"]] = f
    assert len(frames) == 416 and len(queries) == 810
    text_contexts = {
        (q["case"], q["depth"], q["question"]): q["context_sha256"]
        for q in queries
        if q["condition"] == "text" and q["phase"] == "ready"
    }
    for condition, expected in hsummary["conditions"].items():
        normal = [q for q in queries if q["condition"] == condition and q["phase"] == "ready"]
        safety = [q for q in queries if q["condition"] == condition and q["phase"] != "ready"]
        assert len(normal) == expected["queries"]
        assert (
            sum(q["expected"] != "UNKNOWN" and q["answer"] == q["expected"] for q in normal)
            == expected["known_correct"]
        )
        assert sum(q["status"] != "unavailable" for q in normal) == expected["available_queries"]
        assert (
            sum(q["answer"] != "UNKNOWN" and q["answer"] != q["expected"] for q in normal)
            == expected["wrong_assertions"]
        )
        for q in safety:
            assert (
                q["answer"] == "UNKNOWN"
                and q["status"] == "unavailable"
                and q["facts_count"] == q["sources_count"] == 0
            )
        for q in normal:
            assert q["expected"] == actions[q["case"], q["depth"]]["expected"][q["question"]]
            if q["status"] != "unavailable":
                assert q["context_sha256"] == text_contexts[q["case"], q["depth"], q["question"]]
        assert len(safety) == expected["safety_passed"] == 54
    image_hashes = {c["image_sha256"] for c in cases}
    history_hashes = {a["image_sha256"] for a in actions.values()}
    assert len(image_hashes | history_hashes) == 928
    return {
        "passed": True,
        "fresh_source_images": 928,
        "recorded_semantic_process_queries": 810,
        "image_conditions": counts,
        "scope": "Offline replay of recorded predictions and query aggregates;not a new native encoder/model benchmark.",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
