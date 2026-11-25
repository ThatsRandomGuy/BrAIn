from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from brain_codec import BrainCodec
from brain_parser import BrainParser, ParseResult
from brain_policy import BrainPolicy
from brain_evidence import BrainEvidence
from brain_merge import BrainMerge
from brain_similarity import BrainSimilarity
from brain_queue import BrainQueue
from brain_history import BrainHistory


OUTPUT_DIR = Path("brain_output")


class BrainPipeline:
    def __init__(self) -> None:
        self.parser = BrainParser()
        self.codec = BrainCodec()
        self.policy = BrainPolicy()
        self.evidence = BrainEvidence()
        self.merge = BrainMerge()
        self.similarity = BrainSimilarity()
        self.queue = BrainQueue()
        self.history = BrainHistory()

    def run(self, text: str, actor: str = "user") -> dict:
        parse_result: ParseResult = self.parser.parse(text, actor=actor)
        memory = parse_result.memory

        policy_result = self.policy.apply(self.codec, memory)

        merged_registered, merge_decisions_registered = self.merge.canonicalize_policy_items(
            policy_result["registered"]
        )
        merged_provisional, merge_decisions_provisional = self.merge.canonicalize_policy_items(
            policy_result["provisional"]
        )
        merged_ignored, merge_decisions_ignored = self.merge.canonicalize_policy_items(
            policy_result["ignored"]
        )

        policy_result = {
            "registered": merged_registered,
            "provisional": merged_provisional,
            "ignored": merged_ignored,
        }

        merge_decisions = (
            merge_decisions_registered
            + merge_decisions_provisional
            + merge_decisions_ignored
        )

        evidence_result = self.evidence.observe(policy_result["provisional"])
        evidence_registered = self.policy.register_promoted(
            self.codec,
            evidence_result["promoted_now"],
        )

        evidence_registered, merge_decisions_evidence = self.merge.canonicalize_policy_items(
            evidence_registered
        )
        merge_decisions += merge_decisions_evidence

        self.similarity.observe_memory(memory.to_dict())
        similarity_suggestions = self.similarity.suggest_merges()

        queue_added = self.queue.add_suggestions(similarity_suggestions[:10])

        bundle = self.codec.encode_memory_bundle(memory)
        decoded_memory = self.codec.decode_memory_bundle(bundle)
        decoded_memory.confidence = memory.confidence

        reconstructed_text = self.codec.reconstruct_text(decoded_memory)

        history_event = self.history.log(
            event_type="ingest",
            summary=f"Ingested text from {actor}: {text[:80]}",
            payload={
                "actor": actor,
                "input_text": text,
                "action": memory.action,
                "object": memory.object,
                "topic": memory.topic,
                "registered_count": len(policy_result["registered"]),
                "provisional_count": len(policy_result["provisional"]),
                "queue_added_count": len(queue_added),
            },
        )

        return {
            "input_text": text,
            "parsed_memory": memory.to_dict(),
            "parser_debug": parse_result.debug,
            "policy_result": policy_result,
            "merge_decisions": merge_decisions,
            "evidence_result": evidence_result,
            "evidence_registered": evidence_registered,
            "similarity_suggestions": similarity_suggestions[:15],
            "queue_added": queue_added,
            "queue_pending": self.queue.get_pending(),
            "queue_approved": self.queue.get_approved(),
            "queue_rejected": self.queue.get_rejected(),
            "core_grid": bundle["core_grid"],
            "relation_grid": bundle["relation_grid"],
            "decoded_memory": decoded_memory.to_dict(),
            "reconstructed_text": reconstructed_text,
            "history_event": history_event,
            "registry_path": str(self.codec.registry_path),
            "evidence_path": str(self.evidence.evidence_path),
            "merge_path": str(self.merge.merge_path),
            "similarity_path": str(self.similarity.similarity_path),
            "queue_path": str(self.queue.queue_path),
            "history_path": str(self.history.history_path),
        }

    def apply_approved_merges(self, prefer: str = "right") -> dict:
        approved = self.queue.get_approved()
        applied = self.merge.apply_approved_queue_items(approved, prefer=prefer)
        cleared = self.queue.clear_approved()

        history_event = self.history.log(
            event_type="apply_approved_merges",
            summary=f"Applied {len(applied)} approved merge(s)",
            payload={
                "prefer": prefer,
                "approved_count": len(approved),
                "applied_aliases": applied,
            },
        )

        return {
            "approved_items": approved,
            "applied_aliases": applied,
            "cleared_items": cleared,
            "history_event": history_event,
            "merge_path": str(self.merge.merge_path),
            "queue_path": str(self.queue.queue_path),
            "history_path": str(self.history.history_path),
        }

    def save_artifacts(
        self,
        result: dict,
        base_name: str,
        export_images: bool = False,
    ) -> dict:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        safe_name = self._sanitize_filename(base_name)
        result_json_path = OUTPUT_DIR / f"{safe_name}.json"
        core_grid_json_path = OUTPUT_DIR / f"{safe_name}.core.grid.json"
        relation_grid_json_path = OUTPUT_DIR / f"{safe_name}.relation.grid.json"
        core_image_path = OUTPUT_DIR / f"{safe_name}.core.png"
        relation_image_path = OUTPUT_DIR / f"{safe_name}.relation.png"

        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        core_grid = self.codec.jsonable_to_grid(result["core_grid"])
        relation_grid = self.codec.jsonable_to_grid(result["relation_grid"])

        self.codec.save_grid_json(core_grid, core_grid_json_path)
        self.codec.save_grid_json(relation_grid, relation_grid_json_path)

        exported_core_image = None
        exported_relation_image = None

        if export_images:
            try:
                self.codec.export_grid_image(core_grid, core_image_path)
                exported_core_image = str(core_image_path)
            except Exception as exc:
                exported_core_image = f"core image export failed: {exc}"

            try:
                self.codec.export_grid_image(relation_grid, relation_image_path)
                exported_relation_image = str(relation_image_path)
            except Exception as exc:
                exported_relation_image = f"relation image export failed: {exc}"

        return {
            "result_json": str(result_json_path),
            "core_grid_json": str(core_grid_json_path),
            "relation_grid_json": str(relation_grid_json_path),
            "core_grid_image": exported_core_image,
            "relation_grid_image": exported_relation_image,
            "registry_path": result["registry_path"],
            "evidence_path": result["evidence_path"],
            "merge_path": result["merge_path"],
            "similarity_path": result["similarity_path"],
            "queue_path": result["queue_path"],
            "history_path": result["history_path"],
        }

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
        cleaned = "_".join(part for part in cleaned.split("_") if part)
        return cleaned[:60] or "brain_memory"


def print_result(result: dict) -> None:
    print("=" * 72)
    print("INPUT")
    print(result["input_text"])

    print("\nPARSED MEMORY")
    print(json.dumps(result["parsed_memory"], indent=2))

    print("\nPARSER DEBUG")
    print(json.dumps(result["parser_debug"], indent=2))

    print("\nPOLICY RESULT")
    print(json.dumps(result["policy_result"], indent=2))

    print("\nMERGE DECISIONS")
    print(json.dumps(result["merge_decisions"], indent=2))

    print("\nEVIDENCE RESULT")
    print(json.dumps(result["evidence_result"], indent=2))

    print("\nEVIDENCE REGISTERED")
    print(json.dumps(result["evidence_registered"], indent=2))

    print("\nSIMILARITY SUGGESTIONS")
    print(json.dumps(result["similarity_suggestions"], indent=2))

    print("\nQUEUE ADDED")
    print(json.dumps(result["queue_added"], indent=2))

    print("\nQUEUE PENDING")
    print(json.dumps(result["queue_pending"], indent=2))

    print("\nQUEUE APPROVED")
    print(json.dumps(result["queue_approved"], indent=2))

    print("\nQUEUE REJECTED")
    print(json.dumps(result["queue_rejected"], indent=2))

    print("\nCORE GRID")
    print(json.dumps(result["core_grid"], indent=2))

    print("\nRELATION GRID")
    print(json.dumps(result["relation_grid"], indent=2))

    print("\nDECODED MEMORY")
    print(json.dumps(result["decoded_memory"], indent=2))

    print("\nRECONSTRUCTED TEXT")
    print(result["reconstructed_text"])

    print("\nHISTORY EVENT")
    print(json.dumps(result["history_event"], indent=2))

    print("\nREGISTRY PATH")
    print(result["registry_path"])

    print("\nEVIDENCE PATH")
    print(result["evidence_path"])

    print("\nMERGE PATH")
    print(result["merge_path"])

    print("\nSIMILARITY PATH")
    print(result["similarity_path"])

    print("\nQUEUE PATH")
    print(result["queue_path"])

    print("\nHISTORY PATH")
    print(result["history_path"])


def demo(samples: Iterable[str], export_images: bool = False) -> None:
    pipeline = BrainPipeline()

    for idx, sample in enumerate(samples, start=1):
        result = pipeline.run(sample)
        print_result(result)

        paths = pipeline.save_artifacts(
            result,
            base_name=f"sample_{idx}",
            export_images=export_images,
        )

        print("\nSAVED ARTIFACTS")
        print(json.dumps(paths, indent=2))
        print()

    print("=" * 72)
    print("PENDING MERGE QUEUE")
    print(json.dumps(pipeline.queue.get_pending(), indent=2))


if __name__ == "__main__":
    samples = [
        "Create a scary game about apples.",
        "Create a horror game about apples.",
        "Build a webapp about compression.",
        "Build a website about compression.",
    ]

    demo(samples, export_images=False)