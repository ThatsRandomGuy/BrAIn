"""Run inspectable demos and isolated typed memory journals."""

import argparse
import json
import queue
import sys
import time
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from . import journal as m
from .encoder import encoder_factory as configured_encoder


def add(store, op, subject, relation, value=None, nonce=None, encoder_factory=None, config=None):
    index, _, _ = store._read()
    if index["backend"] == "visual" and encoder_factory is None:
        encoder_factory = configured_encoder(config)
    entry = store.append(op, subject, relation, value, nonce)
    index, _, _ = store._read()
    folder = store.folder / "blocks" / entry["block"]
    runtime = None
    if index["backend"] == "visual":
        runtime = (
            store.folder
            / "runtime"
            / ("ingest-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
        )
        runtime.mkdir(parents=True)
        with redirect_stdout(sys.stderr):
            with encoder_factory(runtime) as encoder:
                encoder.encode(folder / "memory.png", folder / "features.pending.sve")
    store.mark_ready(entry["id"])
    return {
        "status": "ready_for_verified_recall",
        "record": entry,
        "policy": store._policy,
        "runtime": str(runtime) if runtime else None,
        "note": "Ready is not proof of accepted readout;query checks every active record.",
    }


def demo(output):
    """A deterministic protocol demonstration,not a visual-model benchmark."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    store = m.GuardedMemoryStore.create(output / "journal", backend="text")
    answers = []
    for op, value in [
        ("SET", "blue equipment locker"),
        ("MAYBE", "padded transport crate"),
        ("SET", "padded transport crate"),
    ]:
        add(store, op, "rescue scanner", "location", value, nonce=23)
        answers.append(store.recall("Where is the evening scanner?")["answer"])
    profile = m.read(m.BUNDLE / "profile.json")
    values = m.encode_record("SET", "location", 2, 27, 1, 23)
    (output / "memory.png").write_bytes(m.guard.image_bytes(values))
    manifest = m.recovery.make_tag(values, profile, 64)
    corrupted = list(values)
    corrupted[0] = (corrupted[0] + 1) % 4
    corrupted[9] = (corrupted[9] + 1) % 4
    receipt = m.recovery.recover_tag(corrupted, manifest, profile, 2)
    return {
        "scope": "Model-free typed-memory and error-recovery demonstration;visual model inference is not performed.",
        "answers": answers,
        "pixel_image": str((output / "memory.png").resolve()),
        "recovery": receipt,
        "journal": str((output / "journal").resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--backend", choices=("visual", "text"), default="text")
    init.add_argument("--policy", choices=tuple(m.POLICIES), default="verify")
    append = commands.add_parser("add")
    append.add_argument("--op", choices=m.OPS, required=True)
    append.add_argument("--subject", required=True)
    append.add_argument("--relation", choices=m.RELATIONS, required=True)
    append.add_argument("--value")
    append.add_argument("--config", type=Path)
    query = commands.add_parser("query")
    query.add_argument("question")
    commands.add_parser("inspect")
    commands.add_parser("clear")
    trial = commands.add_parser("demo")
    trial.add_argument(
        "--output", type=Path, default=Path("artifacts") / ("demo-" + uuid.uuid4().hex[:8])
    )
    replay = commands.add_parser("replay")
    replay.add_argument("--features", type=Path, required=True)
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--profile", type=Path, default=m.BUNDLE / "profile.json")
    replay.add_argument("--model", type=Path, default=m.BUNDLE / "decoder.npz")
    replay.add_argument("--policy", choices=tuple(m.POLICIES), default="two-cell")
    args = parser.parse_args()
    try:
        if args.command == "demo":
            result = demo(args.output)
        elif args.command == "replay":
            profile = m.read(args.profile)
            model = m.guard.load_model(args.model, profile)
            features, _ = m.guard.read_features(args.features)
            prediction = m.guard.predict_values(features, model, profile)
            manifest = m.read(args.manifest)
            if args.policy == "compact-64":
                result = m.recovery.recover_tag(prediction, manifest, profile, 2)
            else:
                result = m.recovery.recover(
                    prediction,
                    manifest,
                    profile,
                    {"verify": 0, "one-cell": 1, "two-cell": 2}[args.policy],
                )
            result = {
                "scope": "Frozen learned visual readout of cached native features;no language model inference.",
                "raw_prediction": prediction,
                **result,
            }
        else:
            if args.store is None:
                parser.error("--store is required for journal commands")
            store = (
                m.GuardedMemoryStore.create(args.store, args.backend, policy=args.policy)
                if args.command == "init"
                else m.GuardedMemoryStore(args.store)
            )
            if args.command in ("init", "inspect"):
                result = store.inspect()
            elif args.command == "add":
                result = add(
                    store, args.op, args.subject, args.relation, args.value, config=args.config
                )
            elif args.command == "clear":
                store.clear()
                result = {"status": "cleared", "history_retained": True}
            else:
                result = store.recall(args.question)
        print(json.dumps(result, indent=2))
        return 0
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        TimeoutError,
        queue.Empty,
        ImportError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": "operation_failed",
                    "reason": str(exc),
                    "note": "Committed pending records remain required. Visual replay needs the vision extra;see docs/quickstart.md.",
                },
                indent=2,
            )
        )
        return 2
