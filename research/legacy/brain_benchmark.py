from __future__ import annotations

# Historical harnesses run with the maintained core on the import path.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]/"core"))

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from brain_bitpacked_codec import BrainBitPackedCodec
from brain_binary_codec import BrainBinaryCodec
from brain_codec import (
    CORE_GRID_COLS,
    CORE_GRID_ROWS,
    EMPTY,
    RELATION_GRID_COLS,
    RELATION_GRID_ROWS,
    BrainCodec,
    BrainMemory,
    Grid,
)
from brain_evidence import BrainEvidence
from brain_history import BrainHistory
from brain_merge import BrainMerge
from brain_parser import BrainParser
from brain_hierarchical_pattern_codec import (
    BrainHierarchicalPatternCodec,
    BrainHierarchicalPatternMemoryStore,
    BrainHierarchicalPatternRegistry,
)
from brain_hybrid_codec import BrainHybridSelector, BrainHybridStore
from brain_input_v1 import BrainInputCodec, BrainInputV1, BrainSemanticTokenRegistry
from brain_pattern_evolution import BrainPatternEvolution
from brain_pattern_codec import BrainPatternCodec, BrainPatternMemoryStore, BrainPatternStore
from brain_policy import BrainPolicy
from brain_queue import BrainQueue
from brain_delta_codec import BrainDeltaCodec, BrainDeltaStore
from brain_sparse_codec import BrainSparseCodec
from brain_similarity import BrainSimilarity


DEFAULT_OUTPUT_DIR = Path("brain_output") / "benchmarks"
ROOT_DIR = Path(".")
CANONICAL_REGISTRY_PATH = ROOT_DIR / "brain_registry.json"
CANONICAL_MERGE_PATH = ROOT_DIR / "brain_merge_map.json"

STATE_FILE_NAMES = (
    "brain_registry.json",
    "brain_evidence.json",
    "brain_merge_map.json",
    "brain_merge_queue.json",
    "brain_similarity.json",
    "brain_history.json",
    "brain_patterns.json",
    "brain_hierarchical_patterns.json",
    "brain_semantic_tokens.json",
)

SINGLE_FIELDS = ("actor", "intent", "action", "object", "topic", "time", "priority")
LIST_FIELDS = ("modifiers", "negations", "constraints", "conditions")


@dataclass(frozen=True)
class BenchmarkPrompt:
    text: str
    actor: str = "user"
    label: str = ""

    def resolved_label(self, index: int) -> str:
        if self.label.strip():
            return _slugify(self.label)
        return f"prompt_{index:02d}"


DEFAULT_BENCHMARK_PROMPTS: List[BenchmarkPrompt] = [
    BenchmarkPrompt(
        label="known_create",
        text="Create a funny game about apples.",
    ),
    BenchmarkPrompt(
        label="known_create_variant",
        text="Create a funny game about oranges.",
    ),
    BenchmarkPrompt(
        label="unknown_topic",
        text="Build a website about semantic compression.",
    ),
    BenchmarkPrompt(
        label="unknown_object_and_modifier",
        text="Create a cozy farming simulator about strawberries.",
    ),
    BenchmarkPrompt(
        label="constraint_negation",
        text="Make a horror app about brain memory, not educational, with low hardware requirements.",
    ),
    BenchmarkPrompt(
        label="compare_request",
        text="Compare web app versus website right now.",
    ),
]

HIGH_SIMILARITY_BENCHMARK_PROMPTS: List[BenchmarkPrompt] = [
    BenchmarkPrompt(
        label="hs_create_apples",
        text="Create a funny game about apples.",
    ),
    BenchmarkPrompt(
        label="hs_create_oranges",
        text="Create a funny game about oranges.",
    ),
    BenchmarkPrompt(
        label="hs_create_pears",
        text="Create a funny game about pears.",
    ),
    BenchmarkPrompt(
        label="hs_create_casual_apples",
        text="Create a casual game about apples.",
    ),
    BenchmarkPrompt(
        label="hs_create_casual_oranges",
        text="Create a casual game about oranges.",
    ),
    BenchmarkPrompt(
        label="hs_website_compression",
        text="Build a website about semantic compression.",
    ),
    BenchmarkPrompt(
        label="hs_website_memory",
        text="Build a website about semantic memory.",
    ),
    BenchmarkPrompt(
        label="hs_website_context",
        text="Build a website about semantic context.",
    ),
]

BUILTIN_PROMPT_SETS: Dict[str, List[BenchmarkPrompt]] = {
    "default": list(DEFAULT_BENCHMARK_PROMPTS),
    "high-similarity": list(HIGH_SIMILARITY_BENCHMARK_PROMPTS),
    "combined": list(DEFAULT_BENCHMARK_PROMPTS) + list(HIGH_SIMILARITY_BENCHMARK_PROMPTS),
}


class BrainBenchmark:
    """
    Deterministic benchmark harness for BrAIn compression experiments.

    The harness reuses the existing parser/policy/evidence/merge/similarity/queue/codec
    components while keeping benchmark state isolated by default so experiments do not
    mutate BrAIn's working memory.
    """

    def __init__(
        self,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        use_live_state: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_live_state = use_live_state

    def run(
        self,
        prompts: Sequence[BenchmarkPrompt],
        report_name: str = "brain_benchmark",
        state_dir: str | Path | None = None,
        prompt_set_name: str | None = None,
    ) -> Dict[str, object]:
        slug = _slugify(report_name)
        resolved_state_dir = self._prepare_state_dir(slug, state_dir)
        runtime = self._build_runtime(resolved_state_dir)

        baseline_registry = set(runtime.codec.primitive_to_color.keys())
        prompt_results: List[Dict[str, object]] = []

        for index, prompt in enumerate(prompts, start=1):
            prompt_results.append(
                self._run_prompt(
                    runtime=runtime,
                    prompt=prompt,
                    index=index,
                )
            )

        evolution_summary = self._apply_pattern_evolution(
            runtime=runtime,
            prompt_results=prompt_results,
        )
        hybrid_summary = self._apply_hybrid_selection(
            runtime=runtime,
            prompt_results=prompt_results,
        )
        brain_input_summary = self._apply_brain_input_v1(
            runtime=runtime,
            prompt_results=prompt_results,
        )

        final_registry = set(runtime.codec.primitive_to_color.keys())
        report = {
            "report_name": report_name,
            "report_slug": slug,
            "prompt_set_name": prompt_set_name or "custom",
            "state_mode": "live" if self.use_live_state else "isolated",
            "state_dir": str(resolved_state_dir),
            "output_dir": str(self.output_dir),
            "registry_baseline_count": len(baseline_registry),
            "registry_final_count": len(final_registry),
            "pattern_registry_count": len(runtime.pattern_store.patterns),
            "hierarchical_pattern_registry_count": len(runtime.hierarchical_pattern_registry.patterns),
            "hierarchical_subpattern_registry_count": len(runtime.hierarchical_pattern_registry.subpatterns),
            "hierarchical_active_pattern_registry_count": len(runtime.hierarchical_pattern_registry.all_active_patterns()),
            "hierarchical_active_subpattern_registry_count": len(runtime.hierarchical_pattern_registry.all_active_subpatterns()),
            "pattern_evolution": evolution_summary,
            "hybrid_selection": hybrid_summary,
            "brain_input_v1": brain_input_summary,
            "aggregate": self._build_aggregate(
                prompt_results=prompt_results,
                baseline_registry=baseline_registry,
                final_registry=final_registry,
            ),
            "results": prompt_results,
        }

        json_path = self.output_dir / f"{slug}.json"
        markdown_path = self.output_dir / f"{slug}.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)

        markdown = self._build_markdown_summary(report)
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        return {
            "report": report,
            "json_path": str(json_path),
            "markdown_path": str(markdown_path),
            "state_dir": str(resolved_state_dir),
        }

    def _prepare_state_dir(self, slug: str, state_dir: str | Path | None) -> Path:
        if self.use_live_state:
            return ROOT_DIR

        resolved = Path(state_dir) if state_dir is not None else self.output_dir / f"{slug}.state"
        resolved.mkdir(parents=True, exist_ok=True)

        for file_name in STATE_FILE_NAMES:
            path = resolved / file_name
            if path.exists():
                path.unlink()

        if CANONICAL_REGISTRY_PATH.exists():
            shutil.copyfile(CANONICAL_REGISTRY_PATH, resolved / "brain_registry.json")

        if CANONICAL_MERGE_PATH.exists():
            shutil.copyfile(CANONICAL_MERGE_PATH, resolved / "brain_merge_map.json")

        return resolved

    @staticmethod
    def _build_runtime(state_dir: Path) -> "_BenchmarkRuntime":
        codec = BrainCodec(registry_path=state_dir / "brain_registry.json")
        binary_codec = BrainBinaryCodec(codec=codec)
        delta_codec = BrainDeltaCodec(
            codec=codec,
            binary_codec=binary_codec,
        )
        pattern_codec = BrainPatternCodec(
            codec=codec,
            binary_codec=binary_codec,
        )
        pattern_store = BrainPatternStore(
            pattern_codec=pattern_codec,
            pattern_path=state_dir / "brain_patterns.json",
        )
        hierarchical_pattern_codec = BrainHierarchicalPatternCodec(
            codec=codec,
            binary_codec=binary_codec,
        )
        hierarchical_pattern_registry = BrainHierarchicalPatternRegistry(
            codec=hierarchical_pattern_codec,
            registry_path=state_dir / "brain_hierarchical_patterns.json",
        )
        pattern_evolution = BrainPatternEvolution(
            codec=hierarchical_pattern_codec,
            registry=hierarchical_pattern_registry,
        )
        semantic_token_registry = BrainSemanticTokenRegistry(
            registry_path=state_dir / "brain_semantic_tokens.json",
        )
        brain_input_codec = BrainInputCodec(
            codec=codec,
            binary_codec=binary_codec,
            token_registry=semantic_token_registry,
            pattern_codec=pattern_codec,
            pattern_store=pattern_store,
            hierarchical_codec=hierarchical_pattern_codec,
            hierarchical_registry=hierarchical_pattern_registry,
            pattern_evolution=pattern_evolution,
        )
        hybrid_selector = BrainHybridSelector(
            bitpacked_codec=BrainBitPackedCodec(binary_codec=binary_codec),
            delta_codec=delta_codec,
            pattern_codec=pattern_codec,
            pattern_store=pattern_store,
            hierarchical_codec=hierarchical_pattern_codec,
            hierarchical_registry=hierarchical_pattern_registry,
            pattern_evolution=pattern_evolution,
        )
        return _BenchmarkRuntime(
            parser=BrainParser(),
            codec=codec,
            binary_codec=binary_codec,
            sparse_codec=BrainSparseCodec(binary_codec=binary_codec),
            bitpacked_codec=BrainBitPackedCodec(binary_codec=binary_codec),
            delta_codec=delta_codec,
            delta_store=BrainDeltaStore(delta_codec=delta_codec),
            pattern_codec=pattern_codec,
            pattern_store=pattern_store,
            pattern_memory_store=BrainPatternMemoryStore(
                pattern_codec=pattern_codec,
                pattern_store=pattern_store,
            ),
            hierarchical_pattern_codec=hierarchical_pattern_codec,
            hierarchical_pattern_registry=hierarchical_pattern_registry,
            hierarchical_pattern_memory_store=BrainHierarchicalPatternMemoryStore(
                codec=hierarchical_pattern_codec,
                registry=hierarchical_pattern_registry,
            ),
            pattern_evolution=pattern_evolution,
            hybrid_selector=hybrid_selector,
            hybrid_store=BrainHybridStore(selector=hybrid_selector),
            semantic_token_registry=semantic_token_registry,
            brain_input_codec=brain_input_codec,
            policy=BrainPolicy(),
            evidence=BrainEvidence(evidence_path=state_dir / "brain_evidence.json"),
            merge=BrainMerge(merge_path=state_dir / "brain_merge_map.json"),
            similarity=BrainSimilarity(similarity_path=state_dir / "brain_similarity.json"),
            queue=BrainQueue(queue_path=state_dir / "brain_merge_queue.json"),
            history=BrainHistory(history_path=state_dir / "brain_history.json"),
        )

    def _run_prompt(
        self,
        runtime: "_BenchmarkRuntime",
        prompt: BenchmarkPrompt,
        index: int,
    ) -> Dict[str, object]:
        label = prompt.resolved_label(index)
        registry_before = set(runtime.codec.primitive_to_color.keys())

        parse_result = runtime.parser.parse(prompt.text, actor=prompt.actor)
        memory = parse_result.memory

        policy_result = runtime.policy.apply(runtime.codec, memory)

        merged_registered, merge_decisions_registered = runtime.merge.canonicalize_policy_items(
            policy_result["registered"]
        )
        merged_provisional, merge_decisions_provisional = runtime.merge.canonicalize_policy_items(
            policy_result["provisional"]
        )
        merged_ignored, merge_decisions_ignored = runtime.merge.canonicalize_policy_items(
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

        evidence_result = runtime.evidence.observe(policy_result["provisional"])
        evidence_registered = runtime.policy.register_promoted(
            runtime.codec,
            evidence_result["promoted_now"],
        )
        evidence_registered, merge_decisions_evidence = runtime.merge.canonicalize_policy_items(
            evidence_registered
        )
        merge_decisions += merge_decisions_evidence

        runtime.similarity.observe_memory(memory.to_dict())
        similarity_suggestions = runtime.similarity.suggest_merges()
        queue_added = runtime.queue.add_suggestions(similarity_suggestions[:10])

        bundle = runtime.codec.encode_memory_bundle(memory)
        decoded_memory = runtime.codec.decode_memory_bundle(bundle)
        decoded_memory.confidence = memory.confidence
        reconstructed_text = runtime.codec.reconstruct_text(decoded_memory)

        registry_after = set(runtime.codec.primitive_to_color.keys())
        new_primitives = sorted(registry_after.difference(registry_before))

        parsed_memory = memory.to_dict()
        parsed_json = _canonical_json(parsed_memory)
        bundle_json = _canonical_json(bundle)
        raw_text_bytes = prompt.text.encode("utf-8")
        parsed_json_bytes = parsed_json.encode("utf-8")
        bundle_json_bytes = bundle_json.encode("utf-8")
        reconstructed_bytes = reconstructed_text.encode("utf-8")
        core_grid = runtime.codec.jsonable_to_grid(bundle["core_grid"])
        relation_grid = runtime.codec.jsonable_to_grid(bundle["relation_grid"])
        runtime.binary_codec.sync_from_codec()
        binary_bundle = runtime.binary_codec.encode_jsonable_bundle(bundle)
        binary_decoded_memory = runtime.binary_codec.decode_memory_bundle(
            binary_bundle.core_ids,
            binary_bundle.relation_ids,
        )
        binary_decoded_memory.confidence = memory.confidence
        runtime.sparse_codec.binary_codec.sync_from_codec()
        sparse_bundle = runtime.sparse_codec.encode_jsonable_bundle_sparse(bundle)
        sparse_decoded_bundle = runtime.sparse_codec.decode_memory_bundle_sparse(sparse_bundle.payload)
        if sparse_decoded_bundle != bundle:
            raise ValueError(f"Sparse codec roundtrip mismatch for benchmark prompt: {label}")

        sparse_decoded_memory = runtime.codec.decode_memory_bundle(sparse_decoded_bundle)
        sparse_decoded_memory.confidence = memory.confidence
        bitpacked_bundle = runtime.bitpacked_codec.encode_jsonable_bundle_bitpacked(bundle)
        bitpacked_decoded_bundle = runtime.bitpacked_codec.decode_memory_bundle_bitpacked(bitpacked_bundle.payload)
        if bitpacked_decoded_bundle != bundle:
            raise ValueError(f"Bit-packed codec roundtrip mismatch for benchmark prompt: {label}")

        bitpacked_decoded_memory = runtime.codec.decode_memory_bundle(bitpacked_decoded_bundle)
        bitpacked_decoded_memory.confidence = memory.confidence
        delta_entry = runtime.delta_store.add_memory(
            memory_id=index,
            memory_bundle=bundle,
        )
        delta_resolved_bundle = runtime.delta_store.resolve_bundle(index)
        if delta_resolved_bundle != bundle:
            raise ValueError(f"Delta codec roundtrip mismatch for benchmark prompt: {label}")

        delta_decoded_memory = runtime.codec.decode_memory_bundle(delta_resolved_bundle)
        delta_decoded_memory.confidence = memory.confidence
        delta_bundle = delta_entry.delta_bundle
        pattern_entry = runtime.pattern_memory_store.add_memory(
            memory_id=index,
            memory_bundle=bundle,
        )
        pattern_resolved_bundle = runtime.pattern_memory_store.resolve_bundle(index)
        if pattern_resolved_bundle != bundle:
            raise ValueError(f"Pattern codec roundtrip mismatch for benchmark prompt: {label}")

        pattern_decoded_memory = runtime.codec.decode_memory_bundle(pattern_resolved_bundle)
        pattern_decoded_memory.confidence = memory.confidence
        pattern_encoded_bundle = pattern_entry.encoded_bundle
        hierarchical_entry = runtime.hierarchical_pattern_memory_store.add_memory(
            memory_id=index,
            memory_bundle=bundle,
        )
        hierarchical_resolved_bundle = runtime.hierarchical_pattern_memory_store.resolve_bundle(index)
        if hierarchical_resolved_bundle != bundle:
            raise ValueError(f"Hierarchical pattern codec roundtrip mismatch for benchmark prompt: {label}")

        hierarchical_decoded_memory = runtime.codec.decode_memory_bundle(hierarchical_resolved_bundle)
        hierarchical_decoded_memory.confidence = memory.confidence
        hierarchical_encoded_bundle = hierarchical_entry.encoded_bundle

        result = {
            "index": index,
            "label": label,
            "actor": prompt.actor,
            "input_text": prompt.text,
            "parsed_memory": parsed_memory,
            "parser_debug": parse_result.debug,
            "governance": {
                "policy_result": policy_result,
                "merge_decisions": merge_decisions,
                "evidence_result": evidence_result,
                "evidence_registered": evidence_registered,
                "queue_added": queue_added,
                "pending_queue_size": len(runtime.queue.get_pending()),
            },
            "representation": {
                "core_grid": bundle["core_grid"],
                "relation_grid": bundle["relation_grid"],
                "core_binary_ids": binary_bundle.core_ids,
                "relation_binary_ids": binary_bundle.relation_ids,
                "sparse_cells": [
                    {
                        "grid_type": cell.grid_type,
                        "cell_index": cell.cell_index,
                        "primitive_id": cell.primitive_id,
                    }
                    for cell in sparse_bundle.cells
                ],
                "decoded_memory": decoded_memory.to_dict(),
                "binary_decoded_memory": binary_decoded_memory.to_dict(),
                "sparse_decoded_memory": sparse_decoded_memory.to_dict(),
                "bitpacked_payload_hex": bitpacked_bundle.payload.hex(),
                "bitpacked_decoded_memory": bitpacked_decoded_memory.to_dict(),
                "delta_payload_hex": delta_entry.payload.hex(),
                "delta_decoded_memory": delta_decoded_memory.to_dict(),
                "pattern_payload_hex": pattern_entry.payload.hex(),
                "pattern_decoded_memory": pattern_decoded_memory.to_dict(),
                "hierarchical_payload_hex": hierarchical_entry.payload.hex(),
                "hierarchical_decoded_memory": hierarchical_decoded_memory.to_dict(),
                "reconstructed_text": reconstructed_text,
            },
            "metrics": {
                "raw_text_length": len(prompt.text),
                "raw_text_size_bytes": len(raw_text_bytes),
                "approximate_token_count": _approximate_token_count(prompt.text),
                "parsed_memory_json_length": len(parsed_json),
                "parsed_memory_json_size_bytes": len(parsed_json_bytes),
                "encoded_bundle_json_length": len(bundle_json),
                "encoded_bundle_json_size_bytes": len(bundle_json_bytes),
                "encoded_bundle_fixed_binary_size_bytes": binary_bundle.total_bytes,
                "encoded_core_binary_size_bytes": len(binary_bundle.core_bytes),
                "encoded_relation_binary_size_bytes": len(binary_bundle.relation_bytes),
                "encoded_bundle_binary_size_bytes": binary_bundle.total_bytes,
                "encoded_bundle_sparse_binary_size_bytes": sparse_bundle.total_bytes,
                "encoded_bundle_bitpacked_binary_size_bytes": bitpacked_bundle.total_bytes,
                "encoded_sparse_core_cells": sparse_bundle.core_populated_cells,
                "encoded_sparse_relation_cells": sparse_bundle.relation_populated_cells,
                "encoded_sparse_populated_cells": sparse_bundle.populated_cells,
                "sparse_index_bytes": sparse_bundle.index_bytes,
                "sparse_primitive_id_bytes": sparse_bundle.primitive_id_bytes,
                "bitpacked_core_cells": bitpacked_bundle.core_populated_cells,
                "bitpacked_relation_cells": bitpacked_bundle.relation_populated_cells,
                "bitpacked_populated_cells": bitpacked_bundle.populated_cells,
                "bitpacked_primitive_id_bits": bitpacked_bundle.primitive_id_bits,
                "encoded_bundle_delta_size_bytes": delta_entry.stored_size_bytes,
                "delta_reference_reused": delta_entry.encoding_type == "delta",
                "delta_encoding_type": delta_entry.encoding_type,
                "delta_reference_id": delta_entry.reference_id,
                "delta_similarity": delta_entry.similarity,
                "delta_core_structure_similarity": (
                    delta_entry.match.core_structure_similarity if delta_entry.match is not None else 0.0
                ),
                "delta_reuse_basis": delta_entry.match.reuse_basis if delta_entry.match is not None else "none",
                "delta_full_size_bytes": delta_entry.full_size_bytes,
                "delta_changed_field_count": len(delta_bundle.changed_fields) if delta_bundle is not None else 0,
                "delta_added_relation_count": len(delta_bundle.added_relations) if delta_bundle is not None else 0,
                "delta_removed_relation_count": len(delta_bundle.removed_relations) if delta_bundle is not None else 0,
                "delta_new_primitive_count": len(delta_bundle.new_primitives) if delta_bundle is not None else 0,
                "encoded_bundle_pattern_size_bytes": pattern_entry.stored_size_bytes,
                "pattern_reference_reused": pattern_entry.encoding_type == "pattern",
                "pattern_encoding_type": pattern_entry.encoding_type,
                "pattern_match_found": pattern_entry.match is not None,
                "pattern_id": pattern_entry.pattern_id,
                "pattern_variable_slot_count": pattern_entry.variable_slot_count,
                "pattern_new_primitive_count": len(pattern_entry.new_primitives),
                "pattern_full_size_bytes": pattern_entry.full_size_bytes,
                "pattern_registry_size": len(runtime.pattern_store.patterns),
                "encoded_bundle_hierarchical_pattern_size_bytes": hierarchical_entry.stored_size_bytes,
                "hierarchical_pattern_reference_reused": hierarchical_entry.encoding_type == "pattern",
                "hierarchical_subpattern_reference_reused": hierarchical_entry.encoding_type == "subpatterns",
                "hierarchical_encoding_type": hierarchical_entry.encoding_type,
                "hierarchical_pattern_match_found": hierarchical_entry.match is not None,
                "hierarchical_pattern_id": hierarchical_entry.pattern_id,
                "hierarchical_subpattern_count": len(hierarchical_entry.subpattern_ids),
                "hierarchical_matched_subpattern_count": (
                    hierarchical_entry.match.matched_subpattern_count if hierarchical_entry.match is not None else 0
                ),
                "hierarchical_variable_slot_count": hierarchical_entry.variable_slot_count,
                "hierarchical_new_primitive_count": len(hierarchical_entry.new_primitives),
                "hierarchical_full_size_bytes": hierarchical_entry.full_size_bytes,
                "hierarchical_pattern_registry_size": len(runtime.hierarchical_pattern_registry.patterns),
                "hierarchical_subpattern_registry_size": len(runtime.hierarchical_pattern_registry.subpatterns),
                "binary_bytes_per_cell": runtime.binary_codec.bytes_per_cell,
                "reconstructed_text_length": len(reconstructed_text),
                "reconstructed_text_size_bytes": len(reconstructed_bytes),
                "reconstructed_token_count": _approximate_token_count(reconstructed_text),
                "parsed_field_counts": _summarize_field_counts(parsed_memory),
                "core_grid": _summarize_grid(core_grid, CORE_GRID_ROWS, CORE_GRID_COLS),
                "relation_grid": _summarize_grid(relation_grid, RELATION_GRID_ROWS, RELATION_GRID_COLS),
                "new_primitives_registered": len(new_primitives),
                "new_primitives": new_primitives,
                "primitive_id_space_size": len(runtime.binary_codec.primitive_to_id),
                "max_primitive_id": max(runtime.binary_codec.id_to_primitive.keys()),
                "confidence_breakdown": _summarize_confidence(memory.confidence),
                "size_ratios": {
                    "parsed_json_vs_raw_bytes": _safe_ratio(len(parsed_json_bytes), len(raw_text_bytes)),
                    "encoded_bundle_json_vs_raw_bytes": _safe_ratio(len(bundle_json_bytes), len(raw_text_bytes)),
                    "encoded_bundle_fixed_binary_vs_raw_bytes": _safe_ratio(binary_bundle.total_bytes, len(raw_text_bytes)),
                    "encoded_bundle_binary_vs_raw_bytes": _safe_ratio(binary_bundle.total_bytes, len(raw_text_bytes)),
                    "encoded_bundle_binary_vs_json_bytes": _safe_ratio(binary_bundle.total_bytes, len(bundle_json_bytes)),
                    "encoded_bundle_sparse_binary_vs_raw_bytes": _safe_ratio(sparse_bundle.total_bytes, len(raw_text_bytes)),
                    "encoded_bundle_sparse_binary_vs_json_bytes": _safe_ratio(sparse_bundle.total_bytes, len(bundle_json_bytes)),
                    "encoded_bundle_sparse_binary_vs_fixed_binary_bytes": _safe_ratio(sparse_bundle.total_bytes, binary_bundle.total_bytes),
                    "encoded_bundle_bitpacked_binary_vs_raw_bytes": _safe_ratio(bitpacked_bundle.total_bytes, len(raw_text_bytes)),
                    "encoded_bundle_bitpacked_binary_vs_json_bytes": _safe_ratio(bitpacked_bundle.total_bytes, len(bundle_json_bytes)),
                    "encoded_bundle_bitpacked_binary_vs_sparse_binary_bytes": _safe_ratio(bitpacked_bundle.total_bytes, sparse_bundle.total_bytes),
                    "encoded_bundle_bitpacked_binary_vs_fixed_binary_bytes": _safe_ratio(bitpacked_bundle.total_bytes, binary_bundle.total_bytes),
                    "encoded_bundle_delta_vs_raw_bytes": _safe_ratio(delta_entry.stored_size_bytes, len(raw_text_bytes)),
                    "encoded_bundle_delta_vs_bitpacked_bytes": _safe_ratio(delta_entry.stored_size_bytes, bitpacked_bundle.total_bytes),
                    "encoded_bundle_delta_vs_sparse_bytes": _safe_ratio(delta_entry.stored_size_bytes, sparse_bundle.total_bytes),
                    "encoded_bundle_pattern_vs_raw_bytes": _safe_ratio(pattern_entry.stored_size_bytes, len(raw_text_bytes)),
                    "encoded_bundle_pattern_vs_bitpacked_bytes": _safe_ratio(pattern_entry.stored_size_bytes, bitpacked_bundle.total_bytes),
                    "encoded_bundle_pattern_vs_delta_bytes": _safe_ratio(pattern_entry.stored_size_bytes, delta_entry.stored_size_bytes),
                    "encoded_bundle_hierarchical_pattern_vs_raw_bytes": _safe_ratio(hierarchical_entry.stored_size_bytes, len(raw_text_bytes)),
                    "encoded_bundle_hierarchical_pattern_vs_bitpacked_bytes": _safe_ratio(hierarchical_entry.stored_size_bytes, bitpacked_bundle.total_bytes),
                    "encoded_bundle_hierarchical_pattern_vs_delta_bytes": _safe_ratio(hierarchical_entry.stored_size_bytes, delta_entry.stored_size_bytes),
                    "encoded_bundle_hierarchical_pattern_vs_pattern_bytes": _safe_ratio(hierarchical_entry.stored_size_bytes, pattern_entry.stored_size_bytes),
                    "reconstructed_vs_raw_bytes": _safe_ratio(len(reconstructed_bytes), len(raw_text_bytes)),
                },
            },
            "delta": {
                "encoding_type": delta_entry.encoding_type,
                "reference_id": delta_entry.reference_id,
                "similarity": delta_entry.similarity,
                "core_structure_similarity": (
                    delta_entry.match.core_structure_similarity if delta_entry.match is not None else 0.0
                ),
                "reuse_basis": delta_entry.match.reuse_basis if delta_entry.match is not None else "none",
                "match_reasons": delta_entry.match.reasons if delta_entry.match is not None else [],
                "matched_scalar_fields": delta_entry.match.matched_scalar_fields if delta_entry.match is not None else [],
                "changed_scalar_fields": delta_entry.match.changed_scalar_fields if delta_entry.match is not None else [],
                "component_scores": delta_entry.match.component_scores if delta_entry.match is not None else {},
                "list_field_scores": delta_entry.match.list_field_scores if delta_entry.match is not None else {},
                "changed_fields": delta_bundle.changed_fields if delta_bundle is not None else {},
                "added_relations": delta_bundle.added_relations if delta_bundle is not None else [],
                "removed_relations": delta_bundle.removed_relations if delta_bundle is not None else [],
                "new_primitives": delta_bundle.new_primitives if delta_bundle is not None else [],
            },
            "pattern": {
                "encoding_type": pattern_entry.encoding_type,
                "pattern_id": pattern_entry.pattern_id,
                "pattern_name": pattern_encoded_bundle.pattern_name if pattern_encoded_bundle is not None else "",
                "match_found": pattern_entry.match is not None,
                "score": pattern_entry.match.score if pattern_entry.match is not None else 0.0,
                "fingerprint": pattern_entry.match.fingerprint if pattern_entry.match is not None else "",
                "stable_fields": pattern_entry.match.stable_fields if pattern_entry.match is not None else [],
                "variable_slots": pattern_entry.match.variable_slots if pattern_entry.match is not None else {},
                "reasons": pattern_entry.match.reasons if pattern_entry.match is not None else [],
                "stored_slot_values": pattern_encoded_bundle.slot_values if pattern_encoded_bundle is not None else {},
                "new_primitives": pattern_entry.new_primitives,
            },
            "hierarchical_pattern": {
                "encoding_type": hierarchical_entry.encoding_type,
                "pattern_id": hierarchical_entry.pattern_id,
                "subpattern_ids": hierarchical_entry.subpattern_ids,
                "match_found": hierarchical_entry.match is not None,
                "reuse_mode": hierarchical_entry.match.reuse_mode if hierarchical_entry.match is not None else "none",
                "score": hierarchical_entry.match.score if hierarchical_entry.match is not None else 0.0,
                "fingerprint": hierarchical_entry.match.fingerprint if hierarchical_entry.match is not None else "",
                "matched_subpattern_ids": hierarchical_entry.match.matched_subpattern_ids if hierarchical_entry.match is not None else [],
                "missing_subpattern_fingerprints": (
                    hierarchical_entry.match.missing_subpattern_fingerprints if hierarchical_entry.match is not None else []
                ),
                "variable_slots": hierarchical_entry.match.variable_slots if hierarchical_entry.match is not None else {},
                "reasons": hierarchical_entry.match.reasons if hierarchical_entry.match is not None else [],
                "stored_slot_values": hierarchical_encoded_bundle.slot_values if hierarchical_encoded_bundle is not None else {},
                "new_primitives": hierarchical_entry.new_primitives,
            },
        }

        return result

    def _apply_pattern_evolution(
        self,
        runtime: "_BenchmarkRuntime",
        prompt_results: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        summary = runtime.pattern_evolution.evolve_registry()
        initial_active_patterns = runtime.hierarchical_pattern_registry.all_active_patterns()
        initial_score_sequence = max(1, len(prompt_results))
        average_pattern_score_before = round(
            sum(
                runtime.pattern_evolution.score_pattern(pattern, current_sequence=initial_score_sequence).score
                for pattern in initial_active_patterns
            ) / max(1, len(initial_active_patterns)),
            4,
        ) if initial_active_patterns else 0.0

        selected_score_total = 0.0
        selected_pattern_count = 0
        selected_mode_counts = {"pattern": 0, "subpatterns": 0, "full": 0}
        pruned_pattern_count = 0
        pruned_subpattern_count = 0

        for sequence, result in enumerate(prompt_results, start=1):
            if not isinstance(result, dict):
                continue
            representation = result.get("representation", {})
            metrics = result.get("metrics", {})
            if not isinstance(representation, dict) or not isinstance(metrics, dict):
                continue

            bundle = {
                "core_grid": representation.get("core_grid", []),
                "relation_grid": representation.get("relation_grid", []),
            }
            baseline_size = int(metrics.get("encoded_bundle_hierarchical_pattern_size_bytes", 0))
            active_pattern_count_before = len(runtime.hierarchical_pattern_registry.all_active_patterns())
            active_subpattern_count_before = len(runtime.hierarchical_pattern_registry.all_active_subpatterns())
            decision = runtime.pattern_evolution.select_adaptive_encoding(
                memory_bundle=bundle,
                current_sequence=sequence,
                include_subpatterns=True,
                update_usage=True,
            )
            prune_info = runtime.pattern_evolution.prune_registry(current_sequence=sequence)
            pruned_pattern_count += int(prune_info.get("pruned_patterns", 0))
            pruned_subpattern_count += int(prune_info.get("pruned_subpatterns", 0))
            active_pattern_count_after = len(runtime.hierarchical_pattern_registry.all_active_patterns())
            active_subpattern_count_after = len(runtime.hierarchical_pattern_registry.all_active_subpatterns())

            temporal_selected_mode = "full"
            temporal_selected_size = baseline_size
            temporal_pattern_id: Optional[int] = None
            temporal_generalization_level = 0
            temporal_generalized_slot_count = 0
            temporal_is_generalized = False
            temporal_score = 0.0
            temporal_reasons: List[str] = []
            evolved_payload_hex = ""
            evolved_decoded_memory: Dict[str, object] = {}

            if decision is not None:
                temporal_selected_mode = decision.mode
                temporal_selected_size = decision.encoded_size_bytes
                temporal_pattern_id = decision.pattern_id
                temporal_generalization_level = decision.generalization_level
                temporal_is_generalized = decision.generalized_pattern
                temporal_score = decision.reinforcement_score
                temporal_reasons = list(decision.reasons)
                selected_mode_counts[decision.mode] = selected_mode_counts.get(decision.mode, 0) + 1
                selected_score_total += decision.reinforcement_score
                selected_pattern_count += 1

                if decision.mode == "pattern" and decision.pattern_id is not None:
                    pattern = runtime.hierarchical_pattern_registry.get_pattern(decision.pattern_id)
                    encoded = runtime.hierarchical_pattern_codec.encode_memory_bundle_hierarchical(
                        current_bundle=bundle,
                        registry=runtime.hierarchical_pattern_registry,
                        pattern=pattern,
                    )
                    decoded_bundle = runtime.hierarchical_pattern_codec.decode_memory_bundle_hierarchical(
                        encoded.payload,
                        runtime.hierarchical_pattern_registry,
                    )
                    if decoded_bundle != bundle:
                        raise ValueError(f"Pattern evolution roundtrip mismatch for benchmark prompt: {result.get('label', '')}")
                    temporal_generalized_slot_count = pattern.generalized_slot_count
                    evolved_payload_hex = encoded.payload.hex()
                    evolved_decoded_memory = runtime.codec.decode_memory_bundle(decoded_bundle).to_dict()
                elif decision.mode == "subpatterns":
                    encoded = runtime.hierarchical_pattern_codec.encode_memory_bundle_hierarchical(
                        current_bundle=bundle,
                        registry=runtime.hierarchical_pattern_registry,
                        subpattern_ids=decision.subpattern_ids,
                    )
                    decoded_bundle = runtime.hierarchical_pattern_codec.decode_memory_bundle_hierarchical(
                        encoded.payload,
                        runtime.hierarchical_pattern_registry,
                    )
                    if decoded_bundle != bundle:
                        raise ValueError(f"Temporal subpattern roundtrip mismatch for benchmark prompt: {result.get('label', '')}")
                    evolved_payload_hex = encoded.payload.hex()
                    evolved_decoded_memory = runtime.codec.decode_memory_bundle(decoded_bundle).to_dict()

            metrics["encoded_bundle_evolved_pattern_candidate_size_bytes"] = temporal_selected_size if temporal_selected_mode != "full" else 0
            metrics["encoded_bundle_evolved_pattern_size_bytes"] = temporal_selected_size
            metrics["evolved_pattern_match_found"] = temporal_selected_mode == "pattern"
            metrics["evolved_pattern_selected"] = temporal_selected_mode != "full"
            metrics["evolved_pattern_id"] = temporal_pattern_id
            metrics["evolved_generalization_level"] = temporal_generalization_level
            metrics["evolved_generalized_slot_count"] = temporal_generalized_slot_count
            metrics["evolved_generalized_pattern_reused"] = temporal_selected_mode == "pattern" and temporal_is_generalized
            metrics["evolved_pattern_vs_raw_bytes"] = _safe_ratio(
                temporal_selected_size,
                int(metrics.get("raw_text_size_bytes", 0)),
            )
            metrics["evolved_pattern_vs_hierarchical_bytes"] = _safe_ratio(
                temporal_selected_size,
                baseline_size,
            )
            metrics["temporal_selection_mode"] = temporal_selected_mode
            metrics["temporal_pattern_score"] = temporal_score
            metrics["temporal_active_pattern_count_before"] = active_pattern_count_before
            metrics["temporal_active_pattern_count_after"] = active_pattern_count_after
            metrics["temporal_active_subpattern_count_before"] = active_subpattern_count_before
            metrics["temporal_active_subpattern_count_after"] = active_subpattern_count_after
            metrics["temporal_pruned_pattern_count"] = int(prune_info.get("pruned_patterns", 0))
            metrics["temporal_pruned_subpattern_count"] = int(prune_info.get("pruned_subpatterns", 0))
            size_ratios = metrics.get("size_ratios", {})
            if isinstance(size_ratios, dict):
                size_ratios["encoded_bundle_evolved_pattern_vs_raw_bytes"] = metrics["evolved_pattern_vs_raw_bytes"]
                size_ratios["encoded_bundle_evolved_pattern_vs_hierarchical_bytes"] = metrics["evolved_pattern_vs_hierarchical_bytes"]
            representation["evolved_pattern_payload_hex"] = evolved_payload_hex
            representation["evolved_pattern_decoded_memory"] = evolved_decoded_memory
            result["pattern_evolution"] = {
                "match_found": temporal_selected_mode == "pattern",
                "selected": temporal_selected_mode != "full",
                "pattern_id": temporal_pattern_id,
                "generalization_level": temporal_generalization_level,
                "generalized_slot_count": temporal_generalized_slot_count,
                "is_generalized_pattern": temporal_is_generalized,
                "candidate_size_bytes": temporal_selected_size if temporal_selected_mode != "full" else 0,
                "selected_size_bytes": temporal_selected_size,
                "reasons": temporal_reasons,
            }
            result["temporal_memory"] = {
                "selection_mode": temporal_selected_mode,
                "pattern_id": temporal_pattern_id,
                "reinforcement_score": temporal_score,
                "generalization_level": temporal_generalization_level,
                "generalized_pattern": temporal_is_generalized,
                "active_pattern_count_before": active_pattern_count_before,
                "active_pattern_count_after": active_pattern_count_after,
                "active_subpattern_count_before": active_subpattern_count_before,
                "active_subpattern_count_after": active_subpattern_count_after,
                "prune_info": prune_info,
                "reasons": temporal_reasons,
            }

        final_active_patterns = runtime.hierarchical_pattern_registry.all_active_patterns()
        average_pattern_score_after = round(
            sum(
                runtime.pattern_evolution.score_pattern(pattern, current_sequence=max(1, len(prompt_results))).score
                for pattern in final_active_patterns
            ) / max(1, len(final_active_patterns)),
            4,
        ) if final_active_patterns else 0.0
        return {
            "before_active_pattern_count": summary.before_active_pattern_count,
            "after_active_pattern_count": summary.after_active_pattern_count,
            "before_active_subpattern_count": summary.before_active_subpattern_count,
            "after_active_subpattern_count": summary.after_active_subpattern_count,
            "merged_pattern_count": summary.merged_pattern_count,
            "generalized_pattern_count": summary.generalized_pattern_count,
            "total_generalized_slots": summary.total_generalized_slots,
            "average_generalized_slots": summary.average_generalized_slots,
            "average_pattern_score_before_temporal": average_pattern_score_before,
            "average_pattern_score_after_temporal": average_pattern_score_after,
            "selected_pattern_score_average": round(selected_score_total / max(1, selected_pattern_count), 4),
            "temporal_selection_counts": selected_mode_counts,
            "temporal_pruned_pattern_count": pruned_pattern_count,
            "temporal_pruned_subpattern_count": pruned_subpattern_count,
            "temporal_pattern_survival_rate": _safe_ratio(
                len(final_active_patterns),
                max(1, summary.after_active_pattern_count),
            ),
            "temporal_subpattern_survival_rate": _safe_ratio(
                len(runtime.hierarchical_pattern_registry.all_active_subpatterns()),
                max(1, summary.after_active_subpattern_count),
            ),
            "temporal_final_active_pattern_count": len(final_active_patterns),
            "temporal_final_active_subpattern_count": len(runtime.hierarchical_pattern_registry.all_active_subpatterns()),
            "events": [
                {
                    "source_pattern_ids": event.source_pattern_ids,
                    "merged_pattern_id": event.merged_pattern_id,
                    "score": event.score,
                    "generalized_fields": event.generalized_fields,
                    "generalized_slot_names": event.generalized_slot_names,
                    "optional_slot_names": event.optional_slot_names,
                    "generalization_level": event.generalization_level,
                    "reasons": event.reasons,
                }
                for event in summary.events
            ],
        }

    def _apply_hybrid_selection(
        self,
        runtime: "_BenchmarkRuntime",
        prompt_results: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        selection_counts: Dict[str, int] = {}
        size_totals: Dict[str, int] = {}

        for sequence, result in enumerate(prompt_results, start=1):
            if not isinstance(result, dict):
                continue
            representation = result.get("representation", {})
            metrics = result.get("metrics", {})
            if not isinstance(representation, dict) or not isinstance(metrics, dict):
                continue

            bundle = {
                "core_grid": representation.get("core_grid", []),
                "relation_grid": representation.get("relation_grid", []),
            }
            entry = runtime.hybrid_store.add_memory(
                memory_id=sequence,
                memory_bundle=bundle,
                current_sequence=sequence,
            )
            selection_counts[entry.encoding_type] = selection_counts.get(entry.encoding_type, 0) + 1
            size_totals[entry.encoding_type] = size_totals.get(entry.encoding_type, 0) + entry.stored_size_bytes

            metrics["encoded_bundle_hybrid_size_bytes"] = entry.stored_size_bytes
            metrics["hybrid_encoding_type"] = entry.encoding_type
            metrics["hybrid_reference_id"] = entry.reference_id
            metrics["hybrid_pattern_id"] = entry.pattern_id
            metrics["hybrid_subpattern_count"] = len(entry.subpattern_ids)
            metrics["hybrid_temporal_score"] = round(entry.temporal_score, 4)
            metrics["hybrid_reuse_confidence"] = round(entry.reuse_confidence, 4)
            metrics["hybrid_vs_raw_bytes"] = _safe_ratio(
                entry.stored_size_bytes,
                int(metrics.get("raw_text_size_bytes", 0)),
            )
            metrics["hybrid_vs_hierarchical_bytes"] = _safe_ratio(
                entry.stored_size_bytes,
                int(metrics.get("encoded_bundle_hierarchical_pattern_size_bytes", 0)),
            )
            metrics["hybrid_vs_evolved_bytes"] = _safe_ratio(
                entry.stored_size_bytes,
                int(metrics.get("encoded_bundle_evolved_pattern_size_bytes", 0)),
            )
            size_ratios = metrics.get("size_ratios", {})
            if isinstance(size_ratios, dict):
                size_ratios["encoded_bundle_hybrid_vs_raw_bytes"] = metrics["hybrid_vs_raw_bytes"]
                size_ratios["encoded_bundle_hybrid_vs_hierarchical_bytes"] = metrics["hybrid_vs_hierarchical_bytes"]
                size_ratios["encoded_bundle_hybrid_vs_evolved_bytes"] = metrics["hybrid_vs_evolved_bytes"]

            result["hybrid"] = {
                "encoding_type": entry.encoding_type,
                "reference_id": entry.reference_id,
                "pattern_id": entry.pattern_id,
                "subpattern_ids": entry.subpattern_ids,
                "stored_size_bytes": entry.stored_size_bytes,
                "temporal_score": round(entry.temporal_score, 4),
                "reuse_confidence": round(entry.reuse_confidence, 4),
                "reasons": entry.reasons,
            }

        average_size_by_type = {
            encoding_type: round(size_totals[encoding_type] / selection_counts[encoding_type], 3)
            for encoding_type in sorted(selection_counts)
            if selection_counts[encoding_type] > 0
        }
        return {
            "selection_counts": {key: selection_counts[key] for key in sorted(selection_counts)},
            "average_size_by_type": average_size_by_type,
            "total_hybrid_size_bytes": sum(size_totals.values()),
        }

    def _apply_brain_input_v1(
        self,
        runtime: "_BenchmarkRuntime",
        prompt_results: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        total_baseline_sequence_tokens = 0
        total_baseline_sequence_bytes = 0
        total_sequence_tokens = 0
        total_sequence_bytes = 0
        total_pattern_tokens = 0
        total_subpattern_tokens = 0
        total_variable_slot_tokens = 0
        total_primitive_ids = 0
        max_sequence_tokens = 0
        inputs_with_pattern_refs = 0
        inputs_with_subpattern_refs = 0
        compact_match_count = 0
        pattern_usage_frequency: Dict[str, int] = {}
        unique_tokens: set[str] = set()

        for sequence_index, result in enumerate(prompt_results, start=1):
            if not isinstance(result, dict):
                continue
            metrics = result.get("metrics", {})
            representation = result.get("representation", {})
            hybrid = result.get("hybrid", {})
            decoded_memory = representation.get("decoded_memory", {})
            if not isinstance(metrics, dict) or not isinstance(representation, dict) or not isinstance(hybrid, dict) or not isinstance(decoded_memory, dict):
                continue

            memory = BrainMemory.from_dict(decoded_memory)
            core_ids = representation.get("core_binary_ids", [])
            relation_ids = representation.get("relation_binary_ids", [])
            primitive_ids = [
                int(item)
                for item in list(core_ids) + list(relation_ids)
                if isinstance(item, int) and item > 0
            ]
            semantic_brain_input = runtime.brain_input_codec.build_brain_input(
                memory=memory,
                encoding_type="bitpacked",
                primitive_ids=[],
                temporal_score=(
                    float(hybrid["temporal_score"])
                    if hybrid.get("temporal_score") is not None
                    else None
                ),
            )
            baseline_sequence = runtime.brain_input_codec.brain_input_to_sequence(
                semantic_brain_input,
                strategy="hybrid",
            )
            baseline_roundtrip = runtime.brain_input_codec.sequence_to_brain_input(
                baseline_sequence,
                strategy="hybrid",
            )
            if baseline_roundtrip.to_dict() != semantic_brain_input.to_dict():
                raise ValueError(f"BrainInputV1 baseline sequence roundtrip mismatch for prompt: {result.get('label', '')}")

            pattern_first_input = runtime.brain_input_codec.brain_input_to_pattern_first(
                semantic_brain_input,
                current_sequence=sequence_index,
            )
            sequence = runtime.brain_input_codec.brain_input_to_sequence(
                semantic_brain_input,
                strategy="pattern_first",
                current_sequence=sequence_index,
            )
            roundtrip = runtime.brain_input_codec.sequence_to_brain_input(
                sequence,
                strategy="pattern_first",
            )
            if (
                roundtrip.encoding_type != pattern_first_input.encoding_type
                or roundtrip.pattern_id != pattern_first_input.pattern_id
                or roundtrip.reference_id != pattern_first_input.reference_id
                or roundtrip.subpattern_refs != pattern_first_input.subpattern_refs
                or roundtrip.temporal_score != pattern_first_input.temporal_score
            ):
                raise ValueError(f"BrainInputV1 pattern-first sequence roundtrip mismatch for prompt: {result.get('label', '')}")

            source_bundle = runtime.codec.encode_memory_bundle(memory)
            roundtrip_bundle = runtime.codec.encode_memory_bundle(
                runtime.brain_input_codec.brain_input_to_memory(roundtrip)
            )
            if roundtrip_bundle != source_bundle:
                raise ValueError(f"BrainInputV1 pattern-first semantic mismatch for prompt: {result.get('label', '')}")

            grid_payload = runtime.brain_input_codec.brain_input_to_grid(
                semantic_brain_input,
                strategy="pattern_first",
                current_sequence=sequence_index,
            )
            grid_roundtrip = runtime.brain_input_codec.grid_to_brain_input(grid_payload)
            if (
                grid_roundtrip.encoding_type != pattern_first_input.encoding_type
                or grid_roundtrip.pattern_id != pattern_first_input.pattern_id
                or grid_roundtrip.reference_id != pattern_first_input.reference_id
                or grid_roundtrip.subpattern_refs != pattern_first_input.subpattern_refs
                or grid_roundtrip.temporal_score != pattern_first_input.temporal_score
            ):
                raise ValueError(f"BrainInputV1 pattern-first grid roundtrip mismatch for prompt: {result.get('label', '')}")
            grid_roundtrip_bundle = runtime.codec.encode_memory_bundle(
                runtime.brain_input_codec.brain_input_to_memory(grid_roundtrip)
            )
            if grid_roundtrip_bundle != source_bundle:
                raise ValueError(f"BrainInputV1 pattern-first grid semantic mismatch for prompt: {result.get('label', '')}")

            runtime.semantic_token_registry.save()

            baseline_sequence_bytes = " ".join(baseline_sequence).encode("utf-8")
            sequence_bytes = " ".join(sequence).encode("utf-8")
            pattern_tokens = sum(1 for token in sequence if token.startswith("PATTERN_"))
            subpattern_tokens = sum(1 for token in sequence if token.startswith("SUBP_"))
            variable_slot_tokens = sum(1 for token in sequence if token.startswith("SLOT_"))
            pattern_usage = (
                pattern_first_input.encoding_type
                if pattern_first_input.pattern_id is not None or pattern_first_input.subpattern_refs
                else "full"
            )
            unique_tokens.update(sequence)
            total_baseline_sequence_tokens += len(baseline_sequence)
            total_baseline_sequence_bytes += len(baseline_sequence_bytes)
            total_sequence_tokens += len(sequence)
            total_sequence_bytes += len(sequence_bytes)
            total_pattern_tokens += pattern_tokens
            total_subpattern_tokens += subpattern_tokens
            total_variable_slot_tokens += variable_slot_tokens
            total_primitive_ids += len(primitive_ids)
            max_sequence_tokens = max(max_sequence_tokens, len(sequence))
            if pattern_tokens > 0:
                inputs_with_pattern_refs += 1
            if subpattern_tokens > 0:
                inputs_with_subpattern_refs += 1
            if pattern_usage != "full":
                compact_match_count += 1
            pattern_usage_frequency[pattern_usage] = pattern_usage_frequency.get(pattern_usage, 0) + 1

            metrics["brain_input_baseline_sequence_token_count"] = len(baseline_sequence)
            metrics["brain_input_baseline_sequence_size_bytes"] = len(baseline_sequence_bytes)
            metrics["brain_input_sequence_token_count"] = len(sequence)
            metrics["brain_input_sequence_size_bytes"] = len(sequence_bytes)
            metrics["brain_input_pattern_token_count"] = pattern_tokens
            metrics["brain_input_subpattern_token_count"] = subpattern_tokens
            metrics["brain_input_variable_slot_token_count"] = variable_slot_tokens
            metrics["brain_input_primitive_id_count"] = len(primitive_ids)
            metrics["brain_input_grid_cell_count"] = int(grid_payload.get("length", 0))
            metrics["brain_input_unit_reduction_count"] = len(baseline_sequence) - len(sequence)
            metrics["brain_input_unit_reduction_percent"] = round(
                ((len(baseline_sequence) - len(sequence)) / len(baseline_sequence)) * 100.0,
                3,
            ) if baseline_sequence else 0.0
            metrics["brain_input_pattern_first_usage"] = pattern_usage
            metrics["brain_input_baseline_vs_raw_tokens"] = _safe_ratio(
                len(baseline_sequence),
                int(metrics.get("approximate_token_count", 0)),
            )
            metrics["brain_input_vs_raw_tokens"] = _safe_ratio(
                len(sequence),
                int(metrics.get("approximate_token_count", 0)),
            )

            size_ratios = metrics.get("size_ratios", {})
            if isinstance(size_ratios, dict):
                size_ratios["brain_input_baseline_sequence_vs_raw_tokens"] = metrics["brain_input_baseline_vs_raw_tokens"]
                size_ratios["brain_input_sequence_vs_raw_tokens"] = metrics["brain_input_vs_raw_tokens"]

            result["brain_input_v1"] = {
                "baseline_schema": semantic_brain_input.to_dict(),
                "schema": pattern_first_input.to_dict(),
                "baseline_sequence_strategy": "hybrid",
                "sequence_strategy": "pattern_first",
                "baseline_sequence": baseline_sequence,
                "sequence": sequence,
                "baseline_sequence_roundtrip": baseline_roundtrip.to_dict(),
                "sequence_roundtrip": roundtrip.to_dict(),
                "grid_payload": grid_payload,
                "grid_roundtrip": grid_roundtrip.to_dict(),
                "pattern_usage": pattern_usage,
            }

        prompt_count = len(prompt_results)
        return {
            "baseline_strategy": "hybrid",
            "strategy": "pattern_first",
            "total_baseline_sequence_tokens": total_baseline_sequence_tokens,
            "total_baseline_sequence_size_bytes": total_baseline_sequence_bytes,
            "total_sequence_tokens": total_sequence_tokens,
            "total_sequence_size_bytes": total_sequence_bytes,
            "average_baseline_sequence_tokens": round(total_baseline_sequence_tokens / prompt_count, 3) if prompt_count else 0.0,
            "average_baseline_sequence_size_bytes": round(total_baseline_sequence_bytes / prompt_count, 3) if prompt_count else 0.0,
            "average_sequence_tokens": round(total_sequence_tokens / prompt_count, 3) if prompt_count else 0.0,
            "average_sequence_size_bytes": round(total_sequence_bytes / prompt_count, 3) if prompt_count else 0.0,
            "total_unit_reduction": total_baseline_sequence_tokens - total_sequence_tokens,
            "average_unit_reduction": round((total_baseline_sequence_tokens - total_sequence_tokens) / prompt_count, 3) if prompt_count else 0.0,
            "unit_reduction_percent": round(
                ((total_baseline_sequence_tokens - total_sequence_tokens) / total_baseline_sequence_tokens) * 100.0,
                3,
            ) if total_baseline_sequence_tokens else 0.0,
            "pattern_token_count": total_pattern_tokens,
            "subpattern_token_count": total_subpattern_tokens,
            "variable_slot_token_count": total_variable_slot_tokens,
            "inputs_with_pattern_refs": inputs_with_pattern_refs,
            "inputs_with_subpattern_refs": inputs_with_subpattern_refs,
            "compact_match_count": compact_match_count,
            "pattern_usage_frequency": {key: pattern_usage_frequency[key] for key in sorted(pattern_usage_frequency)},
            "average_primitive_id_count": round(total_primitive_ids / prompt_count, 3) if prompt_count else 0.0,
            "max_sequence_tokens": max_sequence_tokens,
            "unique_semantic_token_count": len(unique_tokens),
            "token_registry_size": len(runtime.semantic_token_registry.token_to_id),
        }

    def _build_aggregate(
        self,
        prompt_results: Sequence[Dict[str, object]],
        baseline_registry: set[str],
        final_registry: set[str],
    ) -> Dict[str, object]:
        total_raw_chars = 0
        total_raw_bytes = 0
        total_raw_tokens = 0
        total_parsed_json_chars = 0
        total_parsed_json_bytes = 0
        total_bundle_json_chars = 0
        total_bundle_json_bytes = 0
        total_bundle_binary_bytes = 0
        total_bundle_sparse_binary_bytes = 0
        total_bundle_bitpacked_binary_bytes = 0
        total_bundle_delta_bytes = 0
        total_bundle_pattern_bytes = 0
        total_bundle_hierarchical_pattern_bytes = 0
        total_bundle_evolved_pattern_bytes = 0
        total_bundle_hybrid_bytes = 0
        total_brain_input_baseline_sequence_tokens = 0
        total_brain_input_baseline_sequence_bytes = 0
        total_brain_input_sequence_tokens = 0
        total_brain_input_sequence_bytes = 0
        total_brain_input_pattern_tokens = 0
        total_brain_input_subpattern_tokens = 0
        total_brain_input_primitive_ids = 0
        max_brain_input_sequence_tokens = 0
        brain_input_pattern_ref_inputs = 0
        brain_input_subpattern_ref_inputs = 0
        total_reconstructed_chars = 0
        total_reconstructed_bytes = 0
        total_core_cells = 0
        total_relation_cells = 0
        total_sparse_cells = 0
        total_bitpacked_cells = 0
        reused_prompt_count = 0
        delta_entry_count = 0
        total_delta_only_bytes = 0
        pattern_reuse_count = 0
        pattern_match_count = 0
        pattern_fallback_count = 0
        pattern_entry_count = 0
        total_pattern_only_bytes = 0
        total_pattern_variable_slots = 0
        hierarchical_pattern_reuse_count = 0
        hierarchical_subpattern_reuse_prompt_count = 0
        hierarchical_match_count = 0
        hierarchical_entry_count = 0
        total_hierarchical_only_bytes = 0
        total_hierarchical_variable_slots = 0
        total_hierarchical_reused_subpatterns = 0
        evolved_match_count = 0
        evolved_selected_count = 0
        evolved_generalized_reuse_count = 0
        total_evolved_generalized_slots = 0
        total_temporal_pattern_score = 0.0
        temporal_pattern_score_count = 0
        temporal_pattern_selection_count = 0
        temporal_subpattern_selection_count = 0
        temporal_full_selection_count = 0
        temporal_pruned_pattern_count = 0
        temporal_pruned_subpattern_count = 0
        first_temporal_pattern_count_before: Optional[int] = None
        last_temporal_pattern_count_after: Optional[int] = None
        hybrid_selection_counts: Dict[str, int] = {}
        hybrid_size_totals: Dict[str, int] = {}
        confidence_sources: Dict[str, Dict[str, float]] = {}

        unique_new_primitives = sorted(final_registry.difference(baseline_registry))

        for result in prompt_results:
            metrics = result["metrics"]
            if not isinstance(metrics, dict):
                continue

            total_raw_chars += int(metrics.get("raw_text_length", 0))
            total_raw_bytes += int(metrics.get("raw_text_size_bytes", 0))
            total_raw_tokens += int(metrics.get("approximate_token_count", 0))
            total_parsed_json_chars += int(metrics.get("parsed_memory_json_length", 0))
            total_parsed_json_bytes += int(metrics.get("parsed_memory_json_size_bytes", 0))
            total_bundle_json_chars += int(metrics.get("encoded_bundle_json_length", 0))
            total_bundle_json_bytes += int(metrics.get("encoded_bundle_json_size_bytes", 0))
            total_bundle_binary_bytes += int(metrics.get("encoded_bundle_binary_size_bytes", 0))
            total_bundle_sparse_binary_bytes += int(metrics.get("encoded_bundle_sparse_binary_size_bytes", 0))
            total_bundle_bitpacked_binary_bytes += int(metrics.get("encoded_bundle_bitpacked_binary_size_bytes", 0))
            total_bundle_delta_bytes += int(metrics.get("encoded_bundle_delta_size_bytes", 0))
            total_bundle_pattern_bytes += int(metrics.get("encoded_bundle_pattern_size_bytes", 0))
            total_bundle_hierarchical_pattern_bytes += int(metrics.get("encoded_bundle_hierarchical_pattern_size_bytes", 0))
            total_bundle_evolved_pattern_bytes += int(metrics.get("encoded_bundle_evolved_pattern_size_bytes", 0))
            total_bundle_hybrid_bytes += int(metrics.get("encoded_bundle_hybrid_size_bytes", 0))
            total_brain_input_baseline_sequence_tokens += int(metrics.get("brain_input_baseline_sequence_token_count", 0))
            total_brain_input_baseline_sequence_bytes += int(metrics.get("brain_input_baseline_sequence_size_bytes", 0))
            total_brain_input_sequence_tokens += int(metrics.get("brain_input_sequence_token_count", 0))
            total_brain_input_sequence_bytes += int(metrics.get("brain_input_sequence_size_bytes", 0))
            total_brain_input_pattern_tokens += int(metrics.get("brain_input_pattern_token_count", 0))
            total_brain_input_subpattern_tokens += int(metrics.get("brain_input_subpattern_token_count", 0))
            total_brain_input_primitive_ids += int(metrics.get("brain_input_primitive_id_count", 0))
            max_brain_input_sequence_tokens = max(
                max_brain_input_sequence_tokens,
                int(metrics.get("brain_input_sequence_token_count", 0)),
            )
            if int(metrics.get("brain_input_pattern_token_count", 0)) > 0:
                brain_input_pattern_ref_inputs += 1
            if int(metrics.get("brain_input_subpattern_token_count", 0)) > 0:
                brain_input_subpattern_ref_inputs += 1
            total_reconstructed_chars += int(metrics.get("reconstructed_text_length", 0))
            total_reconstructed_bytes += int(metrics.get("reconstructed_text_size_bytes", 0))
            total_sparse_cells += int(metrics.get("encoded_sparse_populated_cells", 0))
            total_bitpacked_cells += int(metrics.get("bitpacked_populated_cells", 0))
            if bool(metrics.get("delta_reference_reused", False)):
                reused_prompt_count += 1
                delta_entry_count += 1
                total_delta_only_bytes += int(metrics.get("encoded_bundle_delta_size_bytes", 0))
            if bool(metrics.get("pattern_match_found", False)):
                pattern_match_count += 1
            if bool(metrics.get("pattern_reference_reused", False)):
                pattern_reuse_count += 1
                pattern_entry_count += 1
                total_pattern_only_bytes += int(metrics.get("encoded_bundle_pattern_size_bytes", 0))
                total_pattern_variable_slots += int(metrics.get("pattern_variable_slot_count", 0))
            else:
                pattern_fallback_count += 1
            if bool(metrics.get("hierarchical_pattern_match_found", False)):
                hierarchical_match_count += 1
                total_hierarchical_reused_subpatterns += int(metrics.get("hierarchical_matched_subpattern_count", 0))
            if bool(metrics.get("hierarchical_pattern_reference_reused", False)):
                hierarchical_pattern_reuse_count += 1
                hierarchical_entry_count += 1
                total_hierarchical_only_bytes += int(metrics.get("encoded_bundle_hierarchical_pattern_size_bytes", 0))
                total_hierarchical_variable_slots += int(metrics.get("hierarchical_variable_slot_count", 0))
            if bool(metrics.get("evolved_pattern_match_found", False)):
                evolved_match_count += 1
            if bool(metrics.get("evolved_pattern_selected", False)):
                evolved_selected_count += 1
                total_evolved_generalized_slots += int(metrics.get("evolved_generalized_slot_count", 0))
            if bool(metrics.get("evolved_generalized_pattern_reused", False)):
                evolved_generalized_reuse_count += 1
            temporal_mode = str(metrics.get("temporal_selection_mode", ""))
            if temporal_mode == "pattern":
                temporal_pattern_selection_count += 1
            elif temporal_mode == "subpatterns":
                temporal_subpattern_selection_count += 1
            elif temporal_mode == "full":
                temporal_full_selection_count += 1
            if "temporal_pattern_score" in metrics:
                total_temporal_pattern_score += float(metrics.get("temporal_pattern_score", 0.0))
                temporal_pattern_score_count += 1
            temporal_pruned_pattern_count += int(metrics.get("temporal_pruned_pattern_count", 0))
            temporal_pruned_subpattern_count += int(metrics.get("temporal_pruned_subpattern_count", 0))
            before_count = metrics.get("temporal_active_pattern_count_before")
            after_count = metrics.get("temporal_active_pattern_count_after")
            if isinstance(before_count, int) and first_temporal_pattern_count_before is None:
                first_temporal_pattern_count_before = before_count
            if isinstance(after_count, int):
                last_temporal_pattern_count_after = after_count
            hybrid_type = str(metrics.get("hybrid_encoding_type", "") or "")
            if hybrid_type:
                hybrid_selection_counts[hybrid_type] = hybrid_selection_counts.get(hybrid_type, 0) + 1
                hybrid_size_totals[hybrid_type] = hybrid_size_totals.get(hybrid_type, 0) + int(metrics.get("encoded_bundle_hybrid_size_bytes", 0))
            elif bool(metrics.get("hierarchical_subpattern_reference_reused", False)):
                hierarchical_subpattern_reuse_prompt_count += 1
                hierarchical_entry_count += 1
                total_hierarchical_only_bytes += int(metrics.get("encoded_bundle_hierarchical_pattern_size_bytes", 0))
                total_hierarchical_variable_slots += int(metrics.get("hierarchical_variable_slot_count", 0))

            core_grid = metrics.get("core_grid", {})
            relation_grid = metrics.get("relation_grid", {})
            if isinstance(core_grid, dict):
                total_core_cells += int(core_grid.get("populated_cells", 0))
            if isinstance(relation_grid, dict):
                total_relation_cells += int(relation_grid.get("populated_cells", 0))

            confidence_breakdown = metrics.get("confidence_breakdown", {})
            if not isinstance(confidence_breakdown, dict):
                continue
            by_source = confidence_breakdown.get("by_source", {})
            if not isinstance(by_source, dict):
                continue

            for source, entry in by_source.items():
                if not isinstance(entry, dict):
                    continue
                bucket = confidence_sources.setdefault(
                    source,
                    {"count": 0.0, "total_score": 0.0},
                )
                bucket["count"] += float(entry.get("count", 0))
                bucket["total_score"] += float(entry.get("average_score", 0.0)) * float(entry.get("count", 0))

        prompt_count = len(prompt_results)
        average_core_cells = round(total_core_cells / prompt_count, 3) if prompt_count else 0.0
        average_relation_cells = round(total_relation_cells / prompt_count, 3) if prompt_count else 0.0
        average_sparse_cells = round(total_sparse_cells / prompt_count, 3) if prompt_count else 0.0
        average_bitpacked_cells = round(total_bitpacked_cells / prompt_count, 3) if prompt_count else 0.0
        average_delta_size = round(total_delta_only_bytes / delta_entry_count, 3) if delta_entry_count else 0.0
        average_pattern_size = round(total_pattern_only_bytes / pattern_entry_count, 3) if pattern_entry_count else 0.0
        average_pattern_variable_slots = (
            round(total_pattern_variable_slots / pattern_entry_count, 3)
            if pattern_entry_count
            else 0.0
        )
        average_hierarchical_size = (
            round(total_hierarchical_only_bytes / hierarchical_entry_count, 3)
            if hierarchical_entry_count
            else 0.0
        )
        average_hierarchical_variable_slots = (
            round(total_hierarchical_variable_slots / hierarchical_entry_count, 3)
            if hierarchical_entry_count
            else 0.0
        )
        average_evolved_size = round(total_bundle_evolved_pattern_bytes / prompt_count, 3) if prompt_count else 0.0
        average_evolved_generalized_slots = (
            round(total_evolved_generalized_slots / evolved_selected_count, 3)
            if evolved_selected_count
            else 0.0
        )
        average_temporal_pattern_score = (
            round(total_temporal_pattern_score / temporal_pattern_score_count, 4)
            if temporal_pattern_score_count
            else 0.0
        )
        average_hybrid_size = round(total_bundle_hybrid_bytes / prompt_count, 3) if prompt_count else 0.0
        average_hybrid_size_by_type = {
            encoding_type: round(hybrid_size_totals[encoding_type] / hybrid_selection_counts[encoding_type], 3)
            for encoding_type in sorted(hybrid_selection_counts)
            if hybrid_selection_counts[encoding_type] > 0
        }
        average_brain_input_sequence_tokens = (
            round(total_brain_input_sequence_tokens / prompt_count, 3)
            if prompt_count
            else 0.0
        )
        average_brain_input_baseline_sequence_tokens = (
            round(total_brain_input_baseline_sequence_tokens / prompt_count, 3)
            if prompt_count
            else 0.0
        )
        average_brain_input_baseline_sequence_bytes = (
            round(total_brain_input_baseline_sequence_bytes / prompt_count, 3)
            if prompt_count
            else 0.0
        )
        average_brain_input_sequence_bytes = (
            round(total_brain_input_sequence_bytes / prompt_count, 3)
            if prompt_count
            else 0.0
        )
        average_brain_input_primitive_ids = (
            round(total_brain_input_primitive_ids / prompt_count, 3)
            if prompt_count
            else 0.0
        )

        confidence_source_summary: Dict[str, Dict[str, float]] = {}
        for source in sorted(confidence_sources):
            count = confidence_sources[source]["count"]
            total_score = confidence_sources[source]["total_score"]
            confidence_source_summary[source] = {
                "count": int(count),
                "average_score": round(total_score / count, 3) if count else 0.0,
            }

        return {
            "prompt_count": prompt_count,
            "total_raw_text_length": total_raw_chars,
            "total_raw_text_size_bytes": total_raw_bytes,
            "total_approximate_tokens": total_raw_tokens,
            "total_parsed_memory_json_length": total_parsed_json_chars,
            "total_parsed_memory_json_size_bytes": total_parsed_json_bytes,
            "total_encoded_bundle_json_length": total_bundle_json_chars,
            "total_encoded_bundle_json_size_bytes": total_bundle_json_bytes,
            "total_encoded_bundle_fixed_binary_size_bytes": total_bundle_binary_bytes,
            "total_encoded_bundle_binary_size_bytes": total_bundle_binary_bytes,
            "total_encoded_bundle_sparse_binary_size_bytes": total_bundle_sparse_binary_bytes,
            "total_encoded_bundle_bitpacked_binary_size_bytes": total_bundle_bitpacked_binary_bytes,
            "total_encoded_bundle_delta_size_bytes": total_bundle_delta_bytes,
            "total_encoded_bundle_pattern_size_bytes": total_bundle_pattern_bytes,
            "total_encoded_bundle_hierarchical_pattern_size_bytes": total_bundle_hierarchical_pattern_bytes,
            "total_encoded_bundle_evolved_pattern_size_bytes": total_bundle_evolved_pattern_bytes,
            "total_encoded_bundle_hybrid_size_bytes": total_bundle_hybrid_bytes,
            "total_brain_input_baseline_sequence_tokens": total_brain_input_baseline_sequence_tokens,
            "total_brain_input_baseline_sequence_size_bytes": total_brain_input_baseline_sequence_bytes,
            "total_brain_input_sequence_tokens": total_brain_input_sequence_tokens,
            "total_brain_input_sequence_size_bytes": total_brain_input_sequence_bytes,
            "total_reconstructed_text_length": total_reconstructed_chars,
            "total_reconstructed_text_size_bytes": total_reconstructed_bytes,
            "total_new_primitives_registered": len(unique_new_primitives),
            "unique_new_primitives": unique_new_primitives,
            "average_populated_core_grid_cells": average_core_cells,
            "average_populated_relation_grid_cells": average_relation_cells,
            "average_sparse_populated_cells": average_sparse_cells,
            "average_bitpacked_populated_cells": average_bitpacked_cells,
            "reused_prompt_count": reused_prompt_count,
            "delta_entry_count": delta_entry_count,
            "average_delta_size_bytes": average_delta_size,
            "delta_improvement_vs_bitpacked_bytes": total_bundle_bitpacked_binary_bytes - total_bundle_delta_bytes,
            "pattern_reuse_count": pattern_reuse_count,
            "pattern_match_count": pattern_match_count,
            "pattern_full_fallback_count": pattern_fallback_count,
            "pattern_entry_count": pattern_entry_count,
            "average_pattern_size_bytes": average_pattern_size,
            "average_pattern_variable_slot_count": average_pattern_variable_slots,
            "pattern_improvement_vs_bitpacked_bytes": total_bundle_bitpacked_binary_bytes - total_bundle_pattern_bytes,
            "pattern_improvement_vs_delta_bytes": total_bundle_delta_bytes - total_bundle_pattern_bytes,
            "hierarchical_pattern_reuse_count": hierarchical_pattern_reuse_count,
            "hierarchical_subpattern_reuse_prompt_count": hierarchical_subpattern_reuse_prompt_count,
            "hierarchical_match_count": hierarchical_match_count,
            "hierarchical_entry_count": hierarchical_entry_count,
            "average_hierarchical_size_bytes": average_hierarchical_size,
            "average_hierarchical_variable_slot_count": average_hierarchical_variable_slots,
            "hierarchical_reused_subpattern_count": total_hierarchical_reused_subpatterns,
            "hierarchical_improvement_vs_bitpacked_bytes": (
                total_bundle_bitpacked_binary_bytes - total_bundle_hierarchical_pattern_bytes
            ),
            "hierarchical_improvement_vs_delta_bytes": (
                total_bundle_delta_bytes - total_bundle_hierarchical_pattern_bytes
            ),
            "hierarchical_improvement_vs_pattern_bytes": (
                total_bundle_pattern_bytes - total_bundle_hierarchical_pattern_bytes
            ),
            "evolved_pattern_match_count": evolved_match_count,
            "evolved_pattern_selected_count": evolved_selected_count,
            "evolved_generalized_pattern_reuse_count": evolved_generalized_reuse_count,
            "average_evolved_pattern_size_bytes": average_evolved_size,
            "average_evolved_generalized_slot_count": average_evolved_generalized_slots,
            "evolved_improvement_vs_hierarchical_bytes": (
                total_bundle_hierarchical_pattern_bytes - total_bundle_evolved_pattern_bytes
            ),
            "average_temporal_pattern_score": average_temporal_pattern_score,
            "average_hybrid_size_bytes": average_hybrid_size,
            "average_brain_input_baseline_sequence_tokens": average_brain_input_baseline_sequence_tokens,
            "average_brain_input_baseline_sequence_size_bytes": average_brain_input_baseline_sequence_bytes,
            "average_brain_input_sequence_tokens": average_brain_input_sequence_tokens,
            "average_brain_input_sequence_size_bytes": average_brain_input_sequence_bytes,
            "average_brain_input_primitive_id_count": average_brain_input_primitive_ids,
            "brain_input_pattern_token_count": total_brain_input_pattern_tokens,
            "brain_input_subpattern_token_count": total_brain_input_subpattern_tokens,
            "brain_input_inputs_with_pattern_refs": brain_input_pattern_ref_inputs,
            "brain_input_inputs_with_subpattern_refs": brain_input_subpattern_ref_inputs,
            "max_brain_input_sequence_tokens": max_brain_input_sequence_tokens,
            "hybrid_selection_counts": {key: hybrid_selection_counts[key] for key in sorted(hybrid_selection_counts)},
            "average_hybrid_size_by_type": average_hybrid_size_by_type,
            "hybrid_improvement_vs_hierarchical_bytes": total_bundle_hierarchical_pattern_bytes - total_bundle_hybrid_bytes,
            "hybrid_improvement_vs_evolved_bytes": total_bundle_evolved_pattern_bytes - total_bundle_hybrid_bytes,
            "temporal_pattern_selection_count": temporal_pattern_selection_count,
            "temporal_subpattern_selection_count": temporal_subpattern_selection_count,
            "temporal_full_selection_count": temporal_full_selection_count,
            "temporal_pruned_pattern_count": temporal_pruned_pattern_count,
            "temporal_pruned_subpattern_count": temporal_pruned_subpattern_count,
            "temporal_pattern_survival_rate": _safe_ratio(
                last_temporal_pattern_count_after or 0,
                first_temporal_pattern_count_before or 0,
            ),
            "aggregate_size_ratios": {
                "parsed_json_vs_raw_bytes": _safe_ratio(total_parsed_json_bytes, total_raw_bytes),
                "encoded_bundle_json_vs_raw_bytes": _safe_ratio(total_bundle_json_bytes, total_raw_bytes),
                "encoded_bundle_fixed_binary_vs_raw_bytes": _safe_ratio(total_bundle_binary_bytes, total_raw_bytes),
                "encoded_bundle_binary_vs_raw_bytes": _safe_ratio(total_bundle_binary_bytes, total_raw_bytes),
                "encoded_bundle_binary_vs_json_bytes": _safe_ratio(total_bundle_binary_bytes, total_bundle_json_bytes),
                "encoded_bundle_sparse_binary_vs_raw_bytes": _safe_ratio(total_bundle_sparse_binary_bytes, total_raw_bytes),
                "encoded_bundle_sparse_binary_vs_json_bytes": _safe_ratio(total_bundle_sparse_binary_bytes, total_bundle_json_bytes),
                "encoded_bundle_sparse_binary_vs_fixed_binary_bytes": _safe_ratio(total_bundle_sparse_binary_bytes, total_bundle_binary_bytes),
                "encoded_bundle_bitpacked_binary_vs_raw_bytes": _safe_ratio(total_bundle_bitpacked_binary_bytes, total_raw_bytes),
                "encoded_bundle_bitpacked_binary_vs_json_bytes": _safe_ratio(total_bundle_bitpacked_binary_bytes, total_bundle_json_bytes),
                "encoded_bundle_bitpacked_binary_vs_sparse_binary_bytes": _safe_ratio(total_bundle_bitpacked_binary_bytes, total_bundle_sparse_binary_bytes),
                "encoded_bundle_bitpacked_binary_vs_fixed_binary_bytes": _safe_ratio(total_bundle_bitpacked_binary_bytes, total_bundle_binary_bytes),
                "encoded_bundle_delta_vs_raw_bytes": _safe_ratio(total_bundle_delta_bytes, total_raw_bytes),
                "encoded_bundle_delta_vs_bitpacked_bytes": _safe_ratio(total_bundle_delta_bytes, total_bundle_bitpacked_binary_bytes),
                "encoded_bundle_delta_vs_sparse_bytes": _safe_ratio(total_bundle_delta_bytes, total_bundle_sparse_binary_bytes),
                "encoded_bundle_pattern_vs_raw_bytes": _safe_ratio(total_bundle_pattern_bytes, total_raw_bytes),
                "encoded_bundle_pattern_vs_bitpacked_bytes": _safe_ratio(total_bundle_pattern_bytes, total_bundle_bitpacked_binary_bytes),
                "encoded_bundle_pattern_vs_delta_bytes": _safe_ratio(total_bundle_pattern_bytes, total_bundle_delta_bytes),
                "encoded_bundle_hierarchical_pattern_vs_raw_bytes": _safe_ratio(total_bundle_hierarchical_pattern_bytes, total_raw_bytes),
                "encoded_bundle_hierarchical_pattern_vs_bitpacked_bytes": _safe_ratio(total_bundle_hierarchical_pattern_bytes, total_bundle_bitpacked_binary_bytes),
                "encoded_bundle_hierarchical_pattern_vs_delta_bytes": _safe_ratio(total_bundle_hierarchical_pattern_bytes, total_bundle_delta_bytes),
                "encoded_bundle_hierarchical_pattern_vs_pattern_bytes": _safe_ratio(total_bundle_hierarchical_pattern_bytes, total_bundle_pattern_bytes),
                "encoded_bundle_evolved_pattern_vs_raw_bytes": _safe_ratio(total_bundle_evolved_pattern_bytes, total_raw_bytes),
                "encoded_bundle_evolved_pattern_vs_hierarchical_bytes": _safe_ratio(total_bundle_evolved_pattern_bytes, total_bundle_hierarchical_pattern_bytes),
                "encoded_bundle_hybrid_vs_raw_bytes": _safe_ratio(total_bundle_hybrid_bytes, total_raw_bytes),
                "encoded_bundle_hybrid_vs_hierarchical_bytes": _safe_ratio(total_bundle_hybrid_bytes, total_bundle_hierarchical_pattern_bytes),
                "encoded_bundle_hybrid_vs_evolved_bytes": _safe_ratio(total_bundle_hybrid_bytes, total_bundle_evolved_pattern_bytes),
                "brain_input_baseline_sequence_vs_raw_tokens": _safe_ratio(total_brain_input_baseline_sequence_tokens, total_raw_tokens),
                "brain_input_sequence_vs_raw_tokens": _safe_ratio(total_brain_input_sequence_tokens, total_raw_tokens),
                "reconstructed_vs_raw_bytes": _safe_ratio(total_reconstructed_bytes, total_raw_bytes),
            },
            "confidence_by_source": confidence_source_summary,
        }

    def _build_markdown_summary(self, report: Dict[str, object]) -> str:
        aggregate = report.get("aggregate", {})
        results = report.get("results", [])

        if not isinstance(aggregate, dict):
            aggregate = {}
        if not isinstance(results, list):
            results = []

        lines = [
            "# BrAIn Compression Benchmark",
            "",
            "This report compares raw text against BrAIn's structured semantic memory and current grid-based internal representation.",
            "",
            "## Overview",
            "",
            f"- Report name: `{report.get('report_name', 'brain_benchmark')}`",
            f"- Prompt set: `{report.get('prompt_set_name', 'custom')}`",
            f"- State mode: `{report.get('state_mode', 'isolated')}`",
            f"- Learned pattern count: `{report.get('pattern_registry_count', 0)}`",
            f"- Learned hierarchical pattern count: `{report.get('hierarchical_pattern_registry_count', 0)}`",
            f"- Learned hierarchical subpattern count: `{report.get('hierarchical_subpattern_registry_count', 0)}`",
            f"- Active hierarchical pattern count: `{report.get('hierarchical_active_pattern_registry_count', 0)}`",
            f"- Active hierarchical subpattern count: `{report.get('hierarchical_active_subpattern_registry_count', 0)}`",
            f"- Prompt count: `{aggregate.get('prompt_count', 0)}`",
            f"- Total raw text length: `{aggregate.get('total_raw_text_length', 0)}` chars",
            f"- Total raw text size: `{aggregate.get('total_raw_text_size_bytes', 0)}` bytes",
            f"- Total approximate tokens: `{aggregate.get('total_approximate_tokens', 0)}`",
            f"- Total parsed memory JSON size: `{aggregate.get('total_parsed_memory_json_size_bytes', 0)}` bytes",
            f"- Total encoded bundle JSON size: `{aggregate.get('total_encoded_bundle_json_size_bytes', 0)}` bytes",
            f"- Total encoded bundle fixed binary size: `{aggregate.get('total_encoded_bundle_fixed_binary_size_bytes', 0)}` bytes",
            f"- Total encoded bundle sparse binary size: `{aggregate.get('total_encoded_bundle_sparse_binary_size_bytes', 0)}` bytes",
            f"- Total encoded bundle bit-packed binary size: `{aggregate.get('total_encoded_bundle_bitpacked_binary_size_bytes', 0)}` bytes",
            f"- Total encoded bundle delta size: `{aggregate.get('total_encoded_bundle_delta_size_bytes', 0)}` bytes",
            f"- Total encoded bundle pattern size: `{aggregate.get('total_encoded_bundle_pattern_size_bytes', 0)}` bytes",
            f"- Total encoded bundle hierarchical pattern size: `{aggregate.get('total_encoded_bundle_hierarchical_pattern_size_bytes', 0)}` bytes",
            f"- Total encoded bundle evolved pattern size: `{aggregate.get('total_encoded_bundle_evolved_pattern_size_bytes', 0)}` bytes",
            f"- Total encoded bundle hybrid size: `{aggregate.get('total_encoded_bundle_hybrid_size_bytes', 0)}` bytes",
            f"- Total BrainInputV1 sequence tokens: `{aggregate.get('total_brain_input_sequence_tokens', 0)}`",
            f"- Average BrainInputV1 sequence tokens: `{aggregate.get('average_brain_input_sequence_tokens', 0.0)}`",
            f"- Reused prior memory count: `{aggregate.get('reused_prompt_count', 0)}`",
            f"- Average delta size: `{aggregate.get('average_delta_size_bytes', 0.0)}` bytes",
            f"- Pattern matches: `{aggregate.get('pattern_match_count', 0)}`",
            f"- Pattern reuses stored compactly: `{aggregate.get('pattern_reuse_count', 0)}`",
            f"- Pattern full fallbacks: `{aggregate.get('pattern_full_fallback_count', 0)}`",
            f"- Average pattern variable slots: `{aggregate.get('average_pattern_variable_slot_count', 0.0)}`",
            f"- Hierarchical matches: `{aggregate.get('hierarchical_match_count', 0)}`",
            f"- Hierarchical exact pattern reuses: `{aggregate.get('hierarchical_pattern_reuse_count', 0)}`",
            f"- Hierarchical subpattern-list reuses: `{aggregate.get('hierarchical_subpattern_reuse_prompt_count', 0)}`",
            f"- Reused hierarchical subpatterns: `{aggregate.get('hierarchical_reused_subpattern_count', 0)}`",
            f"- Average hierarchical variable slots: `{aggregate.get('average_hierarchical_variable_slot_count', 0.0)}`",
            f"- Evolved pattern matches: `{aggregate.get('evolved_pattern_match_count', 0)}`",
            f"- Evolved pattern selections: `{aggregate.get('evolved_pattern_selected_count', 0)}`",
            f"- Evolved generalized reuses: `{aggregate.get('evolved_generalized_pattern_reuse_count', 0)}`",
            f"- Average evolved generalized slots: `{aggregate.get('average_evolved_generalized_slot_count', 0.0)}`",
            f"- Average temporal pattern score: `{aggregate.get('average_temporal_pattern_score', 0.0)}`",
            f"- Temporal pattern selections: `{aggregate.get('temporal_pattern_selection_count', 0)}`",
            f"- Temporal subpattern selections: `{aggregate.get('temporal_subpattern_selection_count', 0)}`",
            f"- Temporal full fallbacks: `{aggregate.get('temporal_full_selection_count', 0)}`",
            f"- Average hybrid size: `{aggregate.get('average_hybrid_size_bytes', 0.0)}` bytes",
            f"- Total new primitives registered: `{aggregate.get('total_new_primitives_registered', 0)}`",
            "",
            "## Aggregate Ratios",
            "",
        ]

        hybrid_selection = report.get("hybrid_selection", {})
        if isinstance(hybrid_selection, dict):
            lines.extend(
                [
                    "",
                    "## Hybrid Selection",
                    "",
                    f"- Selection counts: `{hybrid_selection.get('selection_counts', {})}`",
                    f"- Average size by type: `{hybrid_selection.get('average_size_by_type', {})}`",
                    f"- Total hybrid size: `{hybrid_selection.get('total_hybrid_size_bytes', 0)}` bytes",
                    "",
                ]
            )

        brain_input_v1 = report.get("brain_input_v1", {})
        if isinstance(brain_input_v1, dict):
            lines.extend(
                [
                    "",
                    "## BrainInputV1",
                    "",
                    f"- Baseline strategy: `{brain_input_v1.get('baseline_strategy', 'hybrid')}`",
                    f"- Pattern-first strategy: `{brain_input_v1.get('strategy', 'pattern_first')}`",
                    f"- Total baseline sequence tokens: `{brain_input_v1.get('total_baseline_sequence_tokens', 0)}`",
                    f"- Total pattern-first sequence tokens: `{brain_input_v1.get('total_sequence_tokens', 0)}`",
                    f"- Average baseline sequence tokens: `{brain_input_v1.get('average_baseline_sequence_tokens', 0.0)}`",
                    f"- Average pattern-first sequence tokens: `{brain_input_v1.get('average_sequence_tokens', 0.0)}`",
                    f"- Total baseline sequence size: `{brain_input_v1.get('total_baseline_sequence_size_bytes', 0)}` bytes",
                    f"- Total pattern-first sequence size: `{brain_input_v1.get('total_sequence_size_bytes', 0)}` bytes",
                    f"- Total semantic unit reduction: `{brain_input_v1.get('total_unit_reduction', 0)}`",
                    f"- Average semantic unit reduction: `{brain_input_v1.get('average_unit_reduction', 0.0)}`",
                    f"- Semantic unit reduction percent: `{brain_input_v1.get('unit_reduction_percent', 0.0)}`%",
                    f"- Pattern reference tokens: `{brain_input_v1.get('pattern_token_count', 0)}`",
                    f"- Subpattern reference tokens: `{brain_input_v1.get('subpattern_token_count', 0)}`",
                    f"- Variable slot tokens: `{brain_input_v1.get('variable_slot_token_count', 0)}`",
                    f"- Inputs with pattern refs: `{brain_input_v1.get('inputs_with_pattern_refs', 0)}`",
                    f"- Inputs with subpattern refs: `{brain_input_v1.get('inputs_with_subpattern_refs', 0)}`",
                    f"- Inputs compressed by pattern-first matching: `{brain_input_v1.get('compact_match_count', 0)}`",
                    f"- Pattern usage frequency: `{brain_input_v1.get('pattern_usage_frequency', {})}`",
                    f"- Unique semantic tokens seen: `{brain_input_v1.get('unique_semantic_token_count', 0)}`",
                    f"- Token registry size: `{brain_input_v1.get('token_registry_size', 0)}`",
                    "",
                ]
            )

        pattern_evolution = report.get("pattern_evolution", {})
        if isinstance(pattern_evolution, dict):
            lines.extend(
                [
                    "",
                    "## Pattern Evolution",
                    "",
                    f"- Active patterns before evolution: `{pattern_evolution.get('before_active_pattern_count', 0)}`",
                    f"- Active patterns after evolution: `{pattern_evolution.get('after_active_pattern_count', 0)}`",
                    f"- Active subpatterns before evolution: `{pattern_evolution.get('before_active_subpattern_count', 0)}`",
                    f"- Active subpatterns after evolution: `{pattern_evolution.get('after_active_subpattern_count', 0)}`",
                    f"- Merged pattern events: `{pattern_evolution.get('merged_pattern_count', 0)}`",
                    f"- Generalized pattern count: `{pattern_evolution.get('generalized_pattern_count', 0)}`",
                    f"- Average generalized slots per active generalized pattern: `{pattern_evolution.get('average_generalized_slots', 0.0)}`",
                    f"- Average active pattern score before temporal replay: `{pattern_evolution.get('average_pattern_score_before_temporal', 0.0)}`",
                    f"- Average active pattern score after temporal replay: `{pattern_evolution.get('average_pattern_score_after_temporal', 0.0)}`",
                    f"- Selected pattern score average: `{pattern_evolution.get('selected_pattern_score_average', 0.0)}`",
                    f"- Temporal pattern survival rate: `{pattern_evolution.get('temporal_pattern_survival_rate', 0.0)}`",
                    f"- Temporal subpattern survival rate: `{pattern_evolution.get('temporal_subpattern_survival_rate', 0.0)}`",
                    f"- Pruned patterns: `{pattern_evolution.get('temporal_pruned_pattern_count', 0)}`",
                    f"- Pruned subpatterns: `{pattern_evolution.get('temporal_pruned_subpattern_count', 0)}`",
                    "",
                ]
            )

        aggregate_ratios = aggregate.get("aggregate_size_ratios", {})
        if isinstance(aggregate_ratios, dict):
            lines.extend(
                [
                    f"- Parsed memory JSON vs raw text: `{aggregate_ratios.get('parsed_json_vs_raw_bytes', 0.0)}`",
                    f"- Encoded bundle JSON vs raw text: `{aggregate_ratios.get('encoded_bundle_json_vs_raw_bytes', 0.0)}`",
                    f"- Encoded bundle fixed binary vs raw text: `{aggregate_ratios.get('encoded_bundle_fixed_binary_vs_raw_bytes', 0.0)}`",
                    f"- Encoded bundle sparse binary vs raw text: `{aggregate_ratios.get('encoded_bundle_sparse_binary_vs_raw_bytes', 0.0)}`",
                    f"- Encoded bundle bit-packed binary vs raw text: `{aggregate_ratios.get('encoded_bundle_bitpacked_binary_vs_raw_bytes', 0.0)}`",
                    f"- Encoded bundle delta vs raw text: `{aggregate_ratios.get('encoded_bundle_delta_vs_raw_bytes', 0.0)}`",
                    f"- Encoded bundle pattern vs raw text: `{aggregate_ratios.get('encoded_bundle_pattern_vs_raw_bytes', 0.0)}`",
                    f"- Encoded bundle hierarchical pattern vs raw text: `{aggregate_ratios.get('encoded_bundle_hierarchical_pattern_vs_raw_bytes', 0.0)}`",
                    f"- Encoded bundle fixed binary vs encoded bundle JSON: `{aggregate_ratios.get('encoded_bundle_binary_vs_json_bytes', 0.0)}`",
                    f"- Encoded bundle sparse binary vs encoded bundle JSON: `{aggregate_ratios.get('encoded_bundle_sparse_binary_vs_json_bytes', 0.0)}`",
                    f"- Encoded bundle bit-packed binary vs encoded bundle JSON: `{aggregate_ratios.get('encoded_bundle_bitpacked_binary_vs_json_bytes', 0.0)}`",
                    f"- Encoded bundle sparse binary vs fixed binary: `{aggregate_ratios.get('encoded_bundle_sparse_binary_vs_fixed_binary_bytes', 0.0)}`",
                    f"- Encoded bundle bit-packed binary vs sparse binary: `{aggregate_ratios.get('encoded_bundle_bitpacked_binary_vs_sparse_binary_bytes', 0.0)}`",
                    f"- Encoded bundle bit-packed binary vs fixed binary: `{aggregate_ratios.get('encoded_bundle_bitpacked_binary_vs_fixed_binary_bytes', 0.0)}`",
                    f"- Encoded bundle delta vs bit-packed binary: `{aggregate_ratios.get('encoded_bundle_delta_vs_bitpacked_bytes', 0.0)}`",
                    f"- Encoded bundle delta vs sparse binary: `{aggregate_ratios.get('encoded_bundle_delta_vs_sparse_bytes', 0.0)}`",
                    f"- Encoded bundle pattern vs bit-packed binary: `{aggregate_ratios.get('encoded_bundle_pattern_vs_bitpacked_bytes', 0.0)}`",
                    f"- Encoded bundle pattern vs delta: `{aggregate_ratios.get('encoded_bundle_pattern_vs_delta_bytes', 0.0)}`",
                    f"- Encoded bundle hierarchical pattern vs bit-packed binary: `{aggregate_ratios.get('encoded_bundle_hierarchical_pattern_vs_bitpacked_bytes', 0.0)}`",
                    f"- Encoded bundle hierarchical pattern vs delta: `{aggregate_ratios.get('encoded_bundle_hierarchical_pattern_vs_delta_bytes', 0.0)}`",
                    f"- Encoded bundle hierarchical pattern vs flat pattern: `{aggregate_ratios.get('encoded_bundle_hierarchical_pattern_vs_pattern_bytes', 0.0)}`",
                    f"- Encoded bundle evolved pattern vs raw text: `{aggregate_ratios.get('encoded_bundle_evolved_pattern_vs_raw_bytes', 0.0)}`",
                    f"- Encoded bundle evolved pattern vs hierarchical pattern: `{aggregate_ratios.get('encoded_bundle_evolved_pattern_vs_hierarchical_bytes', 0.0)}`",
                    f"- Encoded bundle hybrid vs raw text: `{aggregate_ratios.get('encoded_bundle_hybrid_vs_raw_bytes', 0.0)}`",
                    f"- Encoded bundle hybrid vs hierarchical pattern: `{aggregate_ratios.get('encoded_bundle_hybrid_vs_hierarchical_bytes', 0.0)}`",
                    f"- Encoded bundle hybrid vs evolved pattern: `{aggregate_ratios.get('encoded_bundle_hybrid_vs_evolved_bytes', 0.0)}`",
                    f"- BrainInputV1 baseline sequence vs approximate raw tokens: `{aggregate_ratios.get('brain_input_baseline_sequence_vs_raw_tokens', 0.0)}`",
                    f"- BrainInputV1 sequence vs approximate raw tokens: `{aggregate_ratios.get('brain_input_sequence_vs_raw_tokens', 0.0)}`",
                    f"- Reconstructed text vs raw text: `{aggregate_ratios.get('reconstructed_vs_raw_bytes', 0.0)}`",
                    f"- Average populated core grid cells: `{aggregate.get('average_populated_core_grid_cells', 0.0)}` of `{CORE_GRID_ROWS * CORE_GRID_COLS}`",
                    f"- Average populated relation grid cells: `{aggregate.get('average_populated_relation_grid_cells', 0.0)}` of `{RELATION_GRID_ROWS * RELATION_GRID_COLS}`",
                    f"- Average sparse populated cells stored: `{aggregate.get('average_sparse_populated_cells', 0.0)}`",
                    f"- Average bit-packed populated cells stored: `{aggregate.get('average_bitpacked_populated_cells', 0.0)}`",
                    f"- Bit-packed to delta byte improvement: `{aggregate.get('delta_improvement_vs_bitpacked_bytes', 0)}` bytes",
                    f"- Bit-packed to pattern byte improvement: `{aggregate.get('pattern_improvement_vs_bitpacked_bytes', 0)}` bytes",
                    f"- Delta to pattern byte improvement: `{aggregate.get('pattern_improvement_vs_delta_bytes', 0)}` bytes",
                    f"- Bit-packed to hierarchical byte improvement: `{aggregate.get('hierarchical_improvement_vs_bitpacked_bytes', 0)}` bytes",
                    f"- Delta to hierarchical byte improvement: `{aggregate.get('hierarchical_improvement_vs_delta_bytes', 0)}` bytes",
                    f"- Flat pattern to hierarchical byte improvement: `{aggregate.get('hierarchical_improvement_vs_pattern_bytes', 0)}` bytes",
                    f"- Hierarchical to evolved byte improvement: `{aggregate.get('evolved_improvement_vs_hierarchical_bytes', 0)}` bytes",
                    f"- Hierarchical to hybrid byte improvement: `{aggregate.get('hybrid_improvement_vs_hierarchical_bytes', 0)}` bytes",
                    f"- Evolved to hybrid byte improvement: `{aggregate.get('hybrid_improvement_vs_evolved_bytes', 0)}` bytes",
                ]
            )

        unique_new_primitives = aggregate.get("unique_new_primitives", [])
        if isinstance(unique_new_primitives, list) and unique_new_primitives:
            lines.extend(
                [
                    "",
                    "## New Primitives",
                    "",
                    f"`{', '.join(str(item) for item in unique_new_primitives)}`",
                ]
            )

        lines.extend(
            [
                "",
                "## Prompt Results",
                "",
                "| # | Label | Raw bytes | Tokens | BrainInput tokens | BrainInput/raw tokens | Parsed JSON bytes | Grid JSON bytes | Fixed binary bytes | Sparse binary bytes | Bit-packed bytes | Delta bytes | Pattern bytes | Hier bytes | Evolved bytes | Hybrid bytes | Hybrid type | Hybrid/hier | Hybrid/evolved | Temporal mode | Temporal score | Hier/raw | Hier mode | Hier subpatterns | Variable slots | Evolved gen slots | Populated cells | Core cells | Relation cells | New primitives | Reconstructed |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )

        for result in results:
            if not isinstance(result, dict):
                continue
            metrics = result.get("metrics", {})
            representation = result.get("representation", {})
            if not isinstance(metrics, dict):
                metrics = {}
            if not isinstance(representation, dict):
                representation = {}
            size_ratios = metrics.get("size_ratios", {})
            if not isinstance(size_ratios, dict):
                size_ratios = {}

            core_grid = metrics.get("core_grid", {})
            relation_grid = metrics.get("relation_grid", {})
            if not isinstance(core_grid, dict):
                core_grid = {}
            if not isinstance(relation_grid, dict):
                relation_grid = {}

            reconstructed = _escape_markdown_cell(str(representation.get("reconstructed_text", "")))
            lines.append(
                "| {index} | {label} | {raw} | {tokens} | {brain_input_tokens} | {brain_input_vs_raw} | {parsed} | {bundle} | {fixed_binary} | {sparse_binary} | {bitpacked_binary} | {delta_bytes} | {pattern_bytes} | {hier_bytes} | {evolved_bytes} | {hybrid_bytes} | {hybrid_type} | {hybrid_vs_hier} | {hybrid_vs_evolved} | {temporal_mode} | {temporal_score} | {hier_vs_raw} | {hier_mode} | {hier_subpatterns} | {variable_slots} | {evolved_generalized_slots} | {stored_cells} | {core} | {relation} | {new_primitive_count} | {reconstructed} |".format(
                    index=result.get("index", ""),
                    label=_escape_markdown_cell(str(result.get("label", ""))),
                    raw=metrics.get("raw_text_size_bytes", 0),
                    tokens=metrics.get("approximate_token_count", 0),
                    brain_input_tokens=metrics.get("brain_input_sequence_token_count", 0),
                    brain_input_vs_raw=size_ratios.get("brain_input_sequence_vs_raw_tokens", 0.0),
                    parsed=metrics.get("parsed_memory_json_size_bytes", 0),
                    bundle=metrics.get("encoded_bundle_json_size_bytes", 0),
                    fixed_binary=metrics.get("encoded_bundle_fixed_binary_size_bytes", 0),
                    sparse_binary=metrics.get("encoded_bundle_sparse_binary_size_bytes", 0),
                    bitpacked_binary=metrics.get("encoded_bundle_bitpacked_binary_size_bytes", 0),
                    delta_bytes=metrics.get("encoded_bundle_delta_size_bytes", 0),
                    pattern_bytes=metrics.get("encoded_bundle_pattern_size_bytes", 0),
                    hier_bytes=metrics.get("encoded_bundle_hierarchical_pattern_size_bytes", 0),
                    evolved_bytes=metrics.get("encoded_bundle_evolved_pattern_size_bytes", 0),
                    hybrid_bytes=metrics.get("encoded_bundle_hybrid_size_bytes", 0),
                    hybrid_type=_escape_markdown_cell(str(metrics.get("hybrid_encoding_type", ""))),
                    hybrid_vs_hier=size_ratios.get("encoded_bundle_hybrid_vs_hierarchical_bytes", 0.0),
                    hybrid_vs_evolved=size_ratios.get("encoded_bundle_hybrid_vs_evolved_bytes", 0.0),
                    temporal_mode=_escape_markdown_cell(str(metrics.get("temporal_selection_mode", "full"))),
                    temporal_score=metrics.get("temporal_pattern_score", 0.0),
                    hier_vs_raw=size_ratios.get("encoded_bundle_hierarchical_pattern_vs_raw_bytes", 0.0),
                    hier_mode=_escape_markdown_cell(str(metrics.get("hierarchical_encoding_type", "full"))),
                    hier_subpatterns=metrics.get("hierarchical_subpattern_count", 0),
                    variable_slots=metrics.get("hierarchical_variable_slot_count", 0),
                    evolved_generalized_slots=metrics.get("evolved_generalized_slot_count", 0),
                    stored_cells=metrics.get("bitpacked_populated_cells", 0),
                    core=core_grid.get("populated_cells", 0),
                    relation=relation_grid.get("populated_cells", 0),
                    new_primitive_count=metrics.get("new_primitives_registered", 0),
                    reconstructed=reconstructed,
                )
            )

        lines.extend(
            [
                "",
                "## Notes",
                "",
                "- Raw text size is measured as UTF-8 bytes.",
                "- Parsed memory JSON and encoded bundle JSON are measured from canonical UTF-8 JSON bytes for deterministic comparison.",
                "- Fixed binary grid size is measured from packed uint32 arrays over the full core and relation grids.",
                "- Sparse binary size stores only populated cells as `(grid_type, cell_index, primitive_id)` entries with compact integer widths.",
                "- Bit-packed binary size groups core and relation cells separately, then stores 4-bit indices plus packed primitive IDs without per-entry grid metadata.",
                "- Delta size stores either a full bit-packed bundle or a delta against the best reusable prior memory, whichever is smaller.",
                "- Pattern size stores either a full bit-packed bundle or a pattern ID plus variable slot primitive IDs when BrAIn recognizes a reusable semantic skeleton.",
                "- Hierarchical size stores either a full bit-packed bundle, an exact hierarchical pattern ID, or a compact list of reusable subpattern IDs plus variable slot primitive IDs.",
                "- Evolved size applies deterministic pattern generalization on top of the hierarchical registry, then selects the smaller of the evolved-pattern payload and the original hierarchical payload.",
                "- Hybrid size evaluates bit-packed, delta, flat pattern, hierarchical, and evolved options, then selects the best deterministic tradeoff between size and reinforced reuse.",
                "- Cell occupancy highlights how much of BrAIn's fixed internal grid was used for each prompt.",
                "- Isolated benchmark state copies the current primitive registry and approved merge map when present, while keeping evidence, similarity, queue, and history fresh for the run.",
                "",
            ]
        )

        return "\n".join(lines)


@dataclass
class _BenchmarkRuntime:
    parser: BrainParser
    codec: BrainCodec
    binary_codec: BrainBinaryCodec
    sparse_codec: BrainSparseCodec
    bitpacked_codec: BrainBitPackedCodec
    delta_codec: BrainDeltaCodec
    delta_store: BrainDeltaStore
    pattern_codec: BrainPatternCodec
    pattern_store: BrainPatternStore
    pattern_memory_store: BrainPatternMemoryStore
    hierarchical_pattern_codec: BrainHierarchicalPatternCodec
    hierarchical_pattern_registry: BrainHierarchicalPatternRegistry
    hierarchical_pattern_memory_store: BrainHierarchicalPatternMemoryStore
    pattern_evolution: BrainPatternEvolution
    hybrid_selector: BrainHybridSelector
    hybrid_store: BrainHybridStore
    semantic_token_registry: BrainSemanticTokenRegistry
    brain_input_codec: BrainInputCodec
    policy: BrainPolicy
    evidence: BrainEvidence
    merge: BrainMerge
    similarity: BrainSimilarity
    queue: BrainQueue
    history: BrainHistory


def _summarize_field_counts(parsed_memory: Dict[str, object]) -> Dict[str, object]:
    by_field: Dict[str, int] = {}

    for field in SINGLE_FIELDS:
        value = parsed_memory.get(field)
        by_field[field] = 1 if isinstance(value, str) and value else 0

    for field in LIST_FIELDS + ("relations",):
        value = parsed_memory.get(field, [])
        by_field[field] = len(value) if isinstance(value, list) else 0

    single_fields_populated = sum(by_field[field] for field in SINGLE_FIELDS)
    list_entries_populated = sum(by_field[field] for field in LIST_FIELDS)
    relation_count = by_field["relations"]

    return {
        "by_field": by_field,
        "single_fields_populated": single_fields_populated,
        "list_entries_populated": list_entries_populated,
        "relation_count": relation_count,
        "total_populated_entries": single_fields_populated + list_entries_populated + relation_count,
    }


def _summarize_grid(grid: Grid, expected_rows: int, expected_cols: int) -> Dict[str, object]:
    populated_cells = sum(1 for row in grid for cell in row if cell != EMPTY)
    populated_rows = sum(1 for row in grid if any(cell != EMPTY for cell in row))
    total_cells = expected_rows * expected_cols

    return {
        "rows": expected_rows,
        "cols": expected_cols,
        "total_cells": total_cells,
        "populated_cells": populated_cells,
        "populated_rows": populated_rows,
        "utilization_ratio": _safe_ratio(populated_cells, total_cells),
    }


def _summarize_confidence(confidence: Dict[str, object]) -> Dict[str, object]:
    entries = _flatten_confidence(confidence)
    by_source: Dict[str, Dict[str, float]] = {}

    for entry in entries:
        source = entry["source"]
        bucket = by_source.setdefault(
            source,
            {
                "count": 0.0,
                "total_score": 0.0,
                "min_score": entry["score"],
                "max_score": entry["score"],
            },
        )
        bucket["count"] += 1.0
        bucket["total_score"] += entry["score"]
        bucket["min_score"] = min(bucket["min_score"], entry["score"])
        bucket["max_score"] = max(bucket["max_score"], entry["score"])

    summarized_sources: Dict[str, Dict[str, float]] = {}
    for source in sorted(by_source):
        bucket = by_source[source]
        count = bucket["count"]
        summarized_sources[source] = {
            "count": int(count),
            "average_score": round(bucket["total_score"] / count, 3) if count else 0.0,
            "min_score": round(bucket["min_score"], 3),
            "max_score": round(bucket["max_score"], 3),
        }

    return {
        "entry_count": len(entries),
        "by_source": summarized_sources,
        "entries": entries,
    }


def _flatten_confidence(confidence: Dict[str, object]) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []

    for field in SINGLE_FIELDS:
        record = confidence.get(field)
        if isinstance(record, dict):
            entries.append(
                {
                    "field": field,
                    "value": str(record.get("value", "")),
                    "source": str(record.get("source", "unknown")),
                    "score": float(record.get("score", 0.0)),
                }
            )

    for field in LIST_FIELDS:
        record = confidence.get(field)
        if not isinstance(record, list):
            continue
        for index, item in enumerate(record):
            if not isinstance(item, dict):
                continue
            entries.append(
                {
                    "field": field,
                    "index": index,
                    "value": str(item.get("value", "")),
                    "source": str(item.get("source", "unknown")),
                    "score": float(item.get("score", 0.0)),
                }
            )

    entries.sort(key=lambda item: (str(item["field"]), int(item.get("index", -1)), str(item["value"])))
    return entries


def _approximate_token_count(text: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text)
    return len(tokens)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:80] or "brain_benchmark"


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _load_prompts_from_path(path: str | Path, default_actor: str = "user") -> List[BenchmarkPrompt]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Prompt file not found: {resolved}")

    suffix = resolved.suffix.lower()
    if suffix == ".json":
        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _parse_prompt_items(data, default_actor=default_actor)

    prompts: List[BenchmarkPrompt] = []
    with open(resolved, "r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            prompts.append(BenchmarkPrompt(text=text, actor=default_actor, label=f"line_{index:02d}"))
    return prompts


def _parse_prompt_items(data: object, default_actor: str) -> List[BenchmarkPrompt]:
    if not isinstance(data, list):
        raise ValueError("Prompt JSON must be a list of strings or prompt objects.")

    prompts: List[BenchmarkPrompt] = []
    for index, item in enumerate(data, start=1):
        if isinstance(item, str):
            prompts.append(BenchmarkPrompt(text=item, actor=default_actor, label=f"item_{index:02d}"))
            continue
        if not isinstance(item, dict):
            raise ValueError("Each prompt entry must be a string or object.")

        text = str(item.get("text", "")).strip()
        if not text:
            raise ValueError("Prompt objects must include non-empty 'text'.")

        actor = str(item.get("actor", default_actor)).strip() or default_actor
        label = str(item.get("label", f"item_{index:02d}")).strip()
        prompts.append(BenchmarkPrompt(text=text, actor=actor, label=label))

    return prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brain_benchmark",
        description="Run a deterministic BrAIn compression benchmark batch.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Add a prompt directly on the command line. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="Path to a .json prompt list or a text file with one prompt per line.",
    )
    parser.add_argument(
        "--actor",
        type=str,
        default="user",
        help="Default actor label for prompts that do not specify one.",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="brain_benchmark",
        help="Base name for the saved benchmark report files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the JSON report and markdown summary should be written.",
    )
    parser.add_argument(
        "--state-dir",
        type=str,
        help="Optional directory for isolated benchmark state files.",
    )
    parser.add_argument(
        "--use-live-state",
        action="store_true",
        help="Use live BrAIn state files in the project root instead of isolated benchmark state.",
    )
    parser.add_argument(
        "--use-default-prompts",
        action="store_true",
        help="Use the selected built-in prompt set even if no prompt file or direct prompts are provided.",
    )
    parser.add_argument(
        "--prompt-set",
        type=str,
        choices=sorted(BUILTIN_PROMPT_SETS.keys()),
        default="default",
        help="Built-in prompt set to use when benchmarking without custom prompt input.",
    )
    return parser


def _resolve_prompts(args: argparse.Namespace) -> tuple[List[BenchmarkPrompt], str]:
    prompts: List[BenchmarkPrompt] = []

    if args.prompts_file:
        prompts.extend(_load_prompts_from_path(args.prompts_file, default_actor=args.actor))

    if args.prompts:
        for index, text in enumerate(args.prompts, start=1):
            prompts.append(BenchmarkPrompt(text=text, actor=args.actor, label=f"cli_{index:02d}"))

    if prompts:
        return prompts, "custom"

    if args.use_default_prompts or (not args.prompts_file and not args.prompts):
        return list(BUILTIN_PROMPT_SETS[args.prompt_set]), args.prompt_set

    return prompts, "custom"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    prompts, prompt_set_name = _resolve_prompts(args)
    if not prompts:
        parser.error("No prompts were provided.")

    benchmark = BrainBenchmark(
        output_dir=args.output_dir,
        use_live_state=args.use_live_state,
    )
    result = benchmark.run(
        prompts=prompts,
        report_name=args.report_name,
        state_dir=args.state_dir,
        prompt_set_name=prompt_set_name,
    )

    print("=" * 72)
    print("BRAIN BENCHMARK")
    print(f"Prompt count: {len(prompts)}")
    print(f"JSON report: {result['json_path']}")
    print(f"Markdown summary: {result['markdown_path']}")
    print(f"State dir: {result['state_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
