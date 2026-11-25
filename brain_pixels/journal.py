"""Version2 recovery-policy journal (separate from frozen earlier prototypes);image recall uses cached projected features.

The word codebook and typed replay are software,not learned English semantics.
Any missing/unverified active record blocks the WHOLE evidence bundle.
"""

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import sys
import uuid
from pathlib import Path
from . import integrity as guard
from . import recovery
from . import persistence

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "assets"
SCHEMA = {
    "version": 1,
    "fields": "op2|relation2|subject4;value8;sequence8;nonce8",
    "ops": ["SET", "MAYBE", "FORGET"],
    "relations": ["location", "owner"],
    "whitening": "SHA256(BrAIn-record-mask-v1 + sequence-byte),first4bytes XOR",
    "policy": "MAYBE andFORGET make their slot UNKNOWN;later SET resolves it. All active records required.",
}
DEFAULT_BOOK = {
    "version": 1,
    "subjects": {"2": "rescue scanner", "7": "field camera", "12": "survey radio"},
    "values": {
        "27": {"label": "blue equipment locker", "kind": "location"},
        "78": {"label": "cedar supply cupboard", "kind": "location"},
        "150": {"label": "padded transport crate", "kind": "location"},
        "201": {"label": "east service cabinet", "kind": "location"},
        "109": {"label": "Mira Chen", "kind": "person"},
        "182": {"label": "Owen Patel", "kind": "person"},
    },
    "aliases": {"evening scanner": 2, "night camera": 7},
}
OPS = ("SET", "MAYBE", "FORGET")
RELATIONS = ("location", "owner")
POLICIES = {
    "verify": {"version": 2, "mode": "verify", "max_candidates": 0},
    "one-cell": {"version": 2, "mode": "one-cell", "max_candidates": 48},
    "two-cell": {"version": 2, "mode": "two-cell", "max_candidates": 1128},
    "compact-64": {"version": 2, "mode": "compact-64", "max_candidates": 1128},
}


def normalize(text):
    if not isinstance(text, str):
        raise ValueError("Expected a name")
    return " ".join(text.strip().casefold().removeprefix("the ").split())


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_book(book):
    if (
        not isinstance(book, dict)
        or set(book) != {"version", "subjects", "values", "aliases"}
        or type(book["version"]) is not int
        or book["version"] != 1
    ):
        raise ValueError("Invalid codebook")
    if any(not isinstance(book[k], dict) for k in ("subjects", "values", "aliases")):
        raise ValueError("Invalid codebook tables")
    if not book["subjects"] or any(
        not k.isdigit()
        or not 0 <= int(k) < 16
        or str(int(k)) != k
        or not isinstance(v, str)
        or not normalize(v)
        for k, v in book["subjects"].items()
    ):
        raise ValueError("Invalid subject IDs")
    names = [normalize(v) for v in book["subjects"].values()]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate subjects")
    for k, v in book["values"].items():
        if (
            not k.isdigit()
            or not 0 <= int(k) < 256
            or str(int(k)) != k
            or not isinstance(v, dict)
            or set(v) != {"label", "kind"}
            or v["kind"] not in ("location", "person")
            or not isinstance(v["label"], str)
            or not v["label"]
        ):
            raise ValueError("Invalid value IDs")
    for alias, subject in book["aliases"].items():
        if (
            not isinstance(alias, str)
            or not normalize(alias)
            or type(subject) is not int
            or str(subject) not in book["subjects"]
        ):
            raise ValueError("Invalid alias")
        for sid, name in book["subjects"].items():
            if normalize(alias) == normalize(name) and int(sid) != subject:
                raise ValueError("Alias collision")


def subject_id(book, name):
    matches = {int(k) for k, v in book["subjects"].items() if normalize(v) == normalize(name)} | {
        sid for alias, sid in book["aliases"].items() if normalize(alias) == normalize(name)
    }
    if len(matches) != 1:
        raise ValueError("Unknown or ambiguous subject")
    return next(iter(matches))


def value_id(book, label, relation):
    kind = "person" if relation == "owner" else "location"
    matches = [
        int(k)
        for k, v in book["values"].items()
        if normalize(v["label"]) == normalize(label) and v["kind"] == kind
    ]
    if len(matches) != 1:
        raise ValueError("Unknown or ambiguous value")
    return matches[0]


def mask(sequence):
    if type(sequence) is not int or not 1 <= sequence <= 255:
        raise ValueError("Record sequence out of range")
    return hashlib.sha256(b"BrAIn-record-mask-v1" + bytes([sequence])).digest()[:4]


def encode_record(op, relation, subject, value, sequence, nonce):
    if (
        op not in OPS
        or relation not in RELATIONS
        or type(subject) is not int
        or not 0 <= subject < 16
        or type(value) is not int
        or not 0 <= value < 256
        or type(nonce) is not int
        or not 0 <= nonce < 256
    ):
        raise ValueError("Invalid record fields")
    if op == "FORGET" and value != 0:
        raise ValueError("FORGET value must be zero")
    raw = bytes(
        [(OPS.index(op) << 6) | (RELATIONS.index(relation) << 4) | subject, value, sequence, nonce]
    )
    payload = bytes(a ^ b for a, b in zip(raw, mask(sequence)))
    return guard.unpack_values(payload)


def parse_record(values, sequence, book):
    raw = bytes(a ^ b for a, b in zip(guard.pack_values(values), mask(sequence)))
    header, value, stored_sequence, nonce = raw
    op = header >> 6
    relation = (header >> 4) & 3
    subject = header & 15
    if (
        op >= len(OPS)
        or relation >= len(RELATIONS)
        or stored_sequence != sequence
        or str(subject) not in book["subjects"]
    ):
        raise ValueError("Invalid decoded record")
    operation = OPS[op]
    rel = RELATIONS[relation]
    if operation == "FORGET":
        if value != 0:
            raise ValueError("Invalid forgetting record")
        label = None
    else:
        entry = book["values"].get(str(value))
        kind = "person" if rel == "owner" else "location"
        if entry is None or entry["kind"] != kind:
            raise ValueError("Value type mismatch")
        label = entry["label"]
    return {
        "sequence": sequence,
        "op": operation,
        "relation": rel,
        "subject_id": subject,
        "subject": book["subjects"][str(subject)],
        "value_id": value,
        "value": label,
        "nonce": nonce,
    }


def unavailable(reason, revision=None, generation=None, blocked=None):
    return {
        "answer": "UNKNOWN",
        "status": "unavailable",
        "reason": reason,
        "revision": revision,
        "generation": generation,
        "context": {"available": False, "sources": [], "facts": [], "blocked_ids": blocked or []},
    }


class GuardedMemoryStore:
    def __init__(self, folder):
        self.folder = Path(folder)
        self._policy = None
        self._readouts = []

    @classmethod
    def create(cls, folder, backend="visual", book=None, bundle=BUNDLE, policy="verify"):
        if backend not in ("visual", "text"):
            raise ValueError("Unsupported backend")
        if policy not in POLICIES or (backend == "text" and policy != "verify"):
            raise ValueError("Unsupported readout policy")
        book = json.loads(json.dumps(DEFAULT_BOOK if book is None else book))
        validate_book(book)
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "revisions").mkdir()
        (folder / "blocks").mkdir()
        (folder / "bundle").mkdir()
        bundle = Path(bundle)
        profile = read(bundle / "profile.json")
        guard.validate_profile(profile)
        (folder / "bundle/profile.json").write_bytes(guard.canonical(profile))
        if backend == "visual":
            shutil.copy2(bundle / "decoder.npz", folder / "bundle/decoder.npz")
            guard.load_model(folder / "bundle/decoder.npz", profile)
        (folder / "codebook.json").write_bytes(guard.canonical(book))
        (folder / "schema.json").write_bytes(guard.canonical(SCHEMA))
        (folder / "policy.json").write_bytes(guard.canonical(POLICIES[policy]))
        store = cls(folder)
        index = {
            "policy_sha256": guard.sha(guard.canonical(POLICIES[policy])),
            "version": 1,
            "backend": backend,
            "revision": 1,
            "generation": 1,
            "codebook_sha256": guard.sha(guard.canonical(book)),
            "schema_sha256": guard.sha(guard.canonical(SCHEMA)),
            "profile_sha256": guard.profile_id(profile),
            "records": [],
        }
        with store.writer():
            store._publish(index)
        return store

    def writer(self):
        return persistence.writer(self.folder)

    def _publish(self, index):
        name = f"r{index['revision']:06d}-" + uuid.uuid4().hex[:12] + ".json"
        raw = guard.canonical(index)
        (self.folder / "revisions" / name).write_bytes(raw)
        persistence.atomic(
            self.folder / "HEAD.json",
            guard.canonical({"version": 1, "index": name, "sha256": guard.sha(raw)}),
        )

    def _read(self):
        head = read(self.folder / "HEAD.json")
        if (
            set(head) != {"version", "index", "sha256"}
            or head["version"] != 1
            or not re.fullmatch(r"r\d{6,}-[0-9a-f]{12}\.json", head["index"])
        ):
            raise ValueError("Invalid head")
        raw = (self.folder / "revisions" / head["index"]).read_bytes()
        if guard.sha(raw) != head["sha256"]:
            raise ValueError("Index checksum mismatch")
        index = json.loads(raw)
        if (
            set(index)
            != {
                "policy_sha256",
                "version",
                "backend",
                "revision",
                "generation",
                "codebook_sha256",
                "schema_sha256",
                "profile_sha256",
                "records",
            }
            or index["version"] != 1
            or index["backend"] not in ("visual", "text")
            or type(index["revision"]) is not int
            or index["revision"] < 1
            or type(index["generation"]) is not int
            or index["generation"] < 1
            or not isinstance(index["records"], list)
            or len(index["records"]) > 255
        ):
            raise ValueError("Invalid index")
        policyraw = (self.folder / "policy.json").read_bytes()
        policy = json.loads(policyraw)
        if (
            guard.sha(policyraw) != index["policy_sha256"]
            or not isinstance(policy, dict)
            or policy not in list(POLICIES.values())
            or type(policy.get("version")) is not int
            or type(policy.get("max_candidates")) is not int
            or (index["backend"] == "text" and policy != POLICIES["verify"])
        ):
            raise ValueError("Readout policy mismatch")
        self._policy = policy
        bookraw = (self.folder / "codebook.json").read_bytes()
        schemaraw = (self.folder / "schema.json").read_bytes()
        profile = read(self.folder / "bundle/profile.json")
        if (
            guard.sha(bookraw) != index["codebook_sha256"]
            or guard.sha(schemaraw) != index["schema_sha256"]
            or json.loads(schemaraw) != SCHEMA
            or guard.profile_id(profile) != index["profile_sha256"]
        ):
            raise ValueError("Shared metadata mismatch")
        book = json.loads(bookraw)
        validate_book(book)
        for i, entry in enumerate(index["records"], 1):
            if (
                set(entry) != {"id", "block", "manifest_sha256", "source_sha256"}
                or entry["id"] != i
                or not re.fullmatch(r"b-[0-9a-f]{12}", entry["block"])
            ):
                raise ValueError("Invalid record index")
        return index, book, profile

    def append(self, op, subject, relation, value=None, nonce=None):
        with self.writer():
            index, book, profile = self._read()
            sid = subject_id(book, subject)
            if relation not in RELATIONS:
                raise ValueError("Invalid relation")
            vid = 0 if op == "FORGET" else value_id(book, value, relation)
            sequence = len(index["records"]) + 1
            values = encode_record(
                op, relation, sid, vid, sequence, secrets.randbelow(256) if nonce is None else nonce
            )
            parse_record(values, sequence, book)
            block = "b-" + uuid.uuid4().hex[:12]
            folder = self.folder / "blocks" / block
            folder.mkdir()
            manifest = (
                recovery.make_tag(values, profile, 64)
                if self._policy["mode"] == "compact-64"
                else guard.make_manifest(values, profile)
            )
            raw = guard.canonical(manifest)
            (folder / "readout.json").write_bytes(raw)
            source = (
                guard.image_bytes(values)
                if index["backend"] == "visual"
                else guard.pack_values(values).hex().encode("ascii")
            )
            (folder / ("memory.png" if index["backend"] == "visual" else "packet.hex")).write_bytes(
                source
            )
            entry = {
                "id": sequence,
                "block": block,
                "manifest_sha256": guard.sha(raw),
                "source_sha256": guard.sha(source),
            }
            index["records"].append(entry)
            index["revision"] += 1
            self._publish(index)
            # Commit pending update BEFORE feature export;never fall back to old state.
            return entry

    def mark_ready(self, sequence):
        with self.writer():
            index, _, profile = self._read()
            entry = index["records"][sequence - 1]
            if entry["id"] != sequence:
                raise ValueError("Record identity mismatch")
            folder = self.folder / "blocks" / entry["block"]
            if index["backend"] == "visual":
                pending = folder / "features.pending.sve"
                _, shape = guard.read_features(pending)
                if shape["width"] != profile["feature_width"]:
                    raise ValueError("Feature width mismatch")
                os.replace(pending, folder / "features.sve")
                digest = guard.sha((folder / "features.sve").read_bytes())
            else:
                payload = (folder / "packet.hex").read_bytes()
                if guard.sha(payload) != entry["source_sha256"]:
                    raise ValueError("Packet integrity mismatch")
                digest = guard.sha(payload)
            persistence.atomic(
                folder / "ready.json",
                guard.canonical(
                    {
                        "version": 1,
                        "manifest_sha256": entry["manifest_sha256"],
                        "data_sha256": digest,
                    }
                ),
            )

    def clear(self):
        with self.writer():
            index, _, _ = self._read()
            index["generation"] += 1
            index["revision"] += 1
            index["records"] = []
            self._publish(index)

    def recall(self, question):
        self._policy = None
        self._readouts = []
        result = self._recall(question)
        result["readout"] = {"policy": self._policy, "records": self._readouts}
        return result

    def _recall(self, question):
        try:
            index, book, profile = self._read()
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return unavailable(str(exc))
        records = []
        blocked = []
        try:
            model = (
                guard.load_model(self.folder / "bundle/decoder.npz", profile)
                if index["backend"] == "visual" and index["records"]
                else None
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return unavailable(str(exc), index["revision"], index["generation"])
        for entry in index["records"]:
            try:
                folder = self.folder / "blocks" / entry["block"]
                raw = (folder / "readout.json").read_bytes()
                if guard.sha(raw) != entry["manifest_sha256"]:
                    raise ValueError("Manifest mismatch")
                metadata = json.loads(raw)
                ready = read(folder / "ready.json")
                if (
                    set(ready) != {"version", "manifest_sha256", "data_sha256"}
                    or ready["version"] != 1
                    or ready["manifest_sha256"] != entry["manifest_sha256"]
                ):
                    raise ValueError("Not ready")
                if index["backend"] == "visual":
                    featurefile = folder / "features.sve"
                    if guard.sha(featurefile.read_bytes()) != ready["data_sha256"]:
                        raise ValueError("Cached features changed")
                    features, _ = guard.read_features(featurefile)
                    prediction = guard.predict_values(features, model, profile)
                    receipt = (
                        recovery.recover_tag(prediction, metadata, profile, 2)
                        if self._policy["mode"] == "compact-64"
                        else (
                            recovery.recover(
                                prediction,
                                metadata,
                                profile,
                                radius=1 if self._policy["mode"] == "one-cell" else 2,
                            )
                            if self._policy["mode"] != "verify"
                            else guard.verify_readout(prediction, metadata, profile)
                        )
                    )
                else:
                    payload = (folder / "packet.hex").read_bytes()
                    if (
                        guard.sha(payload) != ready["data_sha256"]
                        or guard.sha(payload) != entry["source_sha256"]
                    ):
                        raise ValueError("Packet mismatch")
                    receipt = guard.verify_readout(
                        guard.unpack_values(bytes.fromhex(payload.decode("ascii"))),
                        metadata,
                        profile,
                    )
                self._readouts.append(
                    {
                        "id": entry["id"],
                        "status": receipt["status"],
                        "recovered": receipt.get("recovered", False),
                        "candidate_checks": receipt.get("candidate_checks", 0),
                        "distance": receipt.get(
                            "distance", 0 if receipt["status"] == "accepted" else None
                        ),
                    }
                )
                if receipt["status"] != "accepted":
                    raise ValueError(receipt["reason"])
                records.append(parse_record(receipt["values"], entry["id"], book))
            except (OSError, ValueError, TypeError, KeyError, IndexError):
                blocked.append(entry["id"])
        if blocked:
            return unavailable(
                "Required record unavailable", index["revision"], index["generation"], blocked
            )
        state = {}
        sources = []
        for record in records:
            source_id = f"g{index['generation']}:r{record['sequence']}"
            state[record["subject_id"], record["relation"]] = {
                "op": record["op"],
                "value": record["value"] if record["op"] == "SET" else None,
                "source_id": source_id,
            }
            sources.append(
                {
                    "id": source_id,
                    **{k: v for k, v in record.items() if k not in ("nonce", "sequence")},
                }
            )
        facts = [
            {
                "subject": book["subjects"][str(sid)],
                "relation": rel,
                "value": value["value"],
                "source_id": value["source_id"],
            }
            for (sid, rel), value in sorted(state.items())
            if value["value"] is not None
        ]
        context = {"available": True, "sources": sources, "facts": facts, "blocked_ids": []}
        match = re.fullmatch(r"Where is (.+)\?", question, re.I)
        relation = "location"
        if not match:
            match = re.fullmatch(r"Who owns (.+)\?", question, re.I)
            relation = "owner"
        if not match:
            return {
                "answer": "UNKNOWN",
                "status": "unsupported_question",
                "revision": index["revision"],
                "generation": index["generation"],
                "context": context,
            }
        try:
            sid = subject_id(book, match.group(1))
        except ValueError:
            sid = None
        current = state.get((sid, relation))
        answer = current["value"] if current and current["value"] is not None else "UNKNOWN"
        return {
            "answer": answer,
            "status": "supported" if answer != "UNKNOWN" else "unknown",
            "revision": index["revision"],
            "generation": index["generation"],
            "context": context,
        }

    def inspect(self):
        index, _, _ = self._read()
        return {
            "readout_policy": self._policy,
            "backend": index["backend"],
            "revision": index["revision"],
            "generation": index["generation"],
            "active_records": len(index["records"]),
            "retained_bytes": sum(p.stat().st_size for p in self.folder.rglob("*") if p.is_file()),
            "codebook_bytes": (self.folder / "codebook.json").stat().st_size,
            "schema_bytes": (self.folder / "schema.json").stat().st_size,
            "note": "Visual recall uses cached projected features;source PNGs are not read by recall. Logical reset retains history.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--backend", choices=("visual", "text"), default="visual")
    init.add_argument("--policy", choices=tuple(POLICIES), default="verify")
    append = sub.add_parser("append")
    append.add_argument("--op", choices=OPS, required=True)
    append.add_argument("--subject", required=True)
    append.add_argument("--relation", choices=RELATIONS, required=True)
    append.add_argument("--value")
    append.add_argument("--nonce", type=int)
    ready = sub.add_parser("ready")
    ready.add_argument("sequence", type=int)
    query = sub.add_parser("query")
    query.add_argument("question")
    sub.add_parser("clear")
    sub.add_parser("inspect")
    args = parser.parse_args()
    try:
        store = (
            GuardedMemoryStore.create(args.store, args.backend, policy=args.policy)
            if args.command == "init"
            else GuardedMemoryStore(args.store)
        )
        if args.command == "init" or args.command == "inspect":
            result = store.inspect()
        elif args.command == "append":
            result = store.append(args.op, args.subject, args.relation, args.value, args.nonce)
        elif args.command == "ready":
            store.mark_ready(args.sequence)
            result = {"status": "ready"}
        elif args.command == "clear":
            store.clear()
            result = {"status": "cleared", "history_retained": True}
        else:
            result = store.recall(args.question)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        print(json.dumps(unavailable(str(exc)), indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
