from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from brain_bitpacked_codec import BrainBitPackedCodec, BitPackedBundle
from brain_codec import BrainMemory
from brain_delta_codec import BrainDeltaCodec, DeltaBundle, ReferenceMatch
from brain_hierarchical_pattern_codec import (
    BrainHierarchicalPatternCodec,
    BrainHierarchicalPatternRegistry,
    HierarchicalPatternDefinition,
)
from brain_pattern_codec import BrainPatternCodec, BrainPatternStore, PatternDefinition
from brain_pattern_evolution import AdaptiveSelectionDecision, BrainPatternEvolution


STRATEGY_PRIORITY = {
    "hierarchical_pattern": 0,
    "hierarchical_subpatterns": 1,
    "evolved_pattern": 2,
    "evolved_subpatterns": 3,
    "pattern": 4,
    "delta": 5,
    "bitpacked": 6,
}


@dataclass(frozen=True)
class HybridEncodingOption:
    encoding_type: str
    payload: bytes
    size_bytes: int
    reference_id: Optional[int] = None
    pattern_id: Optional[int] = None
    subpattern_ids: List[int] = field(default_factory=list)
    temporal_score: float = 0.0
    reuse_confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    adaptive_decision: Optional[AdaptiveSelectionDecision] = None


@dataclass
class StoredHybridMemory:
    memory_id: int
    encoding_type: str
    payload: bytes
    reference_id: Optional[int]
    pattern_id: Optional[int]
    subpattern_ids: List[int]
    resolved_bundle: Dict[str, List[List[List[int]]]]
    stored_size_bytes: int
    temporal_score: float
    reuse_confidence: float
    reasons: List[str]


class BrainHybridSelector:
    """
    Deterministic hybrid selection across BrAIn encoding strategies.
    """

    def __init__(
        self,
        bitpacked_codec: BrainBitPackedCodec,
        delta_codec: BrainDeltaCodec,
        pattern_codec: BrainPatternCodec,
        pattern_store: BrainPatternStore,
        hierarchical_codec: BrainHierarchicalPatternCodec,
        hierarchical_registry: BrainHierarchicalPatternRegistry,
        pattern_evolution: BrainPatternEvolution,
    ) -> None:
        self.bitpacked_codec = bitpacked_codec
        self.delta_codec = delta_codec
        self.pattern_codec = pattern_codec
        self.pattern_store = pattern_store
        self.hierarchical_codec = hierarchical_codec
        self.hierarchical_registry = hierarchical_registry
        self.pattern_evolution = pattern_evolution

    def evaluate_encoding_options(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
        existing_memories: Sequence[StoredHybridMemory],
        current_sequence: int,
    ) -> List[HybridEncodingOption]:
        full_payload = self.bitpacked_codec.encode_jsonable_bundle_bitpacked(memory_bundle).payload
        options: List[HybridEncodingOption] = [
            HybridEncodingOption(
                encoding_type="bitpacked",
                payload=full_payload,
                size_bytes=len(full_payload),
                reasons=["bitpacked_fallback"],
            )
        ]

        delta_option = self._evaluate_delta(memory_bundle, existing_memories)
        if delta_option is not None:
            options.append(delta_option)

        pattern_option = self._evaluate_pattern(memory_bundle)
        if pattern_option is not None:
            options.append(pattern_option)

        hierarchical_option = self._evaluate_hierarchical(memory_bundle)
        if hierarchical_option is not None:
            options.append(hierarchical_option)

        evolved_option = self._evaluate_evolved(memory_bundle, current_sequence=current_sequence)
        if evolved_option is not None:
            options.append(evolved_option)

        verified: List[HybridEncodingOption] = []
        for option in options:
            try:
                resolved = self.decode_option(option, existing_memories)
            except Exception:
                continue
            if resolved != memory_bundle:
                continue
            verified.append(option)
        return verified

    def select_best_encoding(
        self,
        options: Sequence[HybridEncodingOption],
    ) -> HybridEncodingOption:
        if not options:
            raise ValueError("Hybrid selection requires at least one encoding option.")

        smallest_size = min(option.size_bytes for option in options)
        eligible: List[HybridEncodingOption] = []
        for option in options:
            tolerance = 1
            if option.temporal_score >= 0.85:
                tolerance = 2
            if option.size_bytes <= smallest_size + tolerance:
                eligible.append(option)

        candidate_pool = eligible or list(options)
        return min(
            candidate_pool,
            key=lambda option: (
                self._effective_cost(option),
                option.size_bytes,
                -option.reuse_confidence,
                -option.temporal_score,
                STRATEGY_PRIORITY.get(option.encoding_type, 99),
                option.reference_id if option.reference_id is not None else 10**9,
                option.pattern_id if option.pattern_id is not None else 10**9,
            ),
        )

    def decode_option(
        self,
        option: HybridEncodingOption,
        existing_memories: Sequence[StoredHybridMemory],
    ) -> Dict[str, List[List[List[int]]]]:
        if option.encoding_type == "bitpacked":
            return self.bitpacked_codec.decode_memory_bundle_bitpacked(option.payload)

        if option.encoding_type == "delta":
            if option.reference_id is None:
                raise ValueError("Delta option requires a reference ID.")
            reference_bundle = self._resolve_existing_bundle(option.reference_id, existing_memories)
            return self.delta_codec.decode_memory_bundle_delta(option.payload, reference_bundle)

        if option.encoding_type == "pattern":
            if option.pattern_id is None:
                raise ValueError("Pattern option requires a pattern ID.")
            pattern = self.pattern_store.get(option.pattern_id)
            return self.pattern_codec.decode_memory_bundle_pattern(option.payload, pattern)

        if option.encoding_type in {"hierarchical_pattern", "evolved_pattern", "hierarchical_subpatterns", "evolved_subpatterns"}:
            return self.hierarchical_codec.decode_memory_bundle_hierarchical(
                option.payload,
                self.hierarchical_registry,
            )

        raise ValueError(f"Unsupported hybrid encoding type: {option.encoding_type}")

    def _evaluate_delta(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
        existing_memories: Sequence[StoredHybridMemory],
    ) -> Optional[HybridEncodingOption]:
        if not existing_memories:
            return None
        existing_items = [
            {"id": entry.memory_id, "bundle": entry.resolved_bundle}
            for entry in existing_memories
        ]
        match = self.delta_codec.find_best_reference(memory_bundle, existing_items)
        if match is None:
            return None
        reference_bundle = self._resolve_existing_bundle(match.reference_id, existing_memories)
        delta_bundle = self.delta_codec.encode_memory_bundle_delta(
            current_bundle=memory_bundle,
            reference_bundle=reference_bundle,
            reference_id=match.reference_id,
            similarity=match.similarity,
        )
        return HybridEncodingOption(
            encoding_type="delta",
            payload=delta_bundle.payload,
            size_bytes=delta_bundle.total_bytes,
            reference_id=match.reference_id,
            temporal_score=0.0,
            reuse_confidence=match.similarity,
            reasons=list(match.reasons),
        )

    def _evaluate_pattern(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
    ) -> Optional[HybridEncodingOption]:
        match = self.pattern_codec.find_best_pattern(
            memory_bundle=memory_bundle,
            known_patterns=self.pattern_store.all_patterns(),
        )
        if match is None:
            return None
        pattern = self.pattern_store.get(match.pattern_id)
        encoded = self.pattern_codec.encode_memory_bundle_pattern(memory_bundle, pattern)
        return HybridEncodingOption(
            encoding_type="pattern",
            payload=encoded.payload,
            size_bytes=encoded.total_bytes,
            pattern_id=pattern.pattern_id,
            temporal_score=0.0,
            reuse_confidence=match.score,
            reasons=list(match.reasons),
        )

    def _evaluate_hierarchical(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
    ) -> Optional[HybridEncodingOption]:
        match = self.hierarchical_codec.find_best_pattern(memory_bundle, self.hierarchical_registry)
        if match.pattern_id is not None:
            pattern = self.hierarchical_registry.get_pattern(match.pattern_id)
            encoded = self.hierarchical_codec.encode_memory_bundle_hierarchical(
                current_bundle=memory_bundle,
                registry=self.hierarchical_registry,
                pattern=pattern,
            )
            temporal_score = self.pattern_evolution.score_pattern(pattern).score
            return HybridEncodingOption(
                encoding_type="hierarchical_pattern",
                payload=encoded.payload,
                size_bytes=encoded.total_bytes,
                pattern_id=pattern.pattern_id,
                subpattern_ids=list(pattern.subpattern_ids),
                temporal_score=temporal_score,
                reuse_confidence=match.score,
                reasons=list(match.reasons) + [f"temporal_score={temporal_score}"],
            )

        if match.matched_subpattern_ids:
            candidate = self.hierarchical_codec.extract_subpatterns(memory_bundle)
            subpattern_ids: List[int] = []
            for subpattern in candidate.subpatterns:
                existing = self.hierarchical_registry.find_subpattern_by_fingerprint(subpattern.fingerprint, active_only=True)
                if existing is None:
                    return None
                subpattern_ids.append(existing.subpattern_id)
            encoded = self.hierarchical_codec.encode_memory_bundle_hierarchical(
                current_bundle=memory_bundle,
                registry=self.hierarchical_registry,
                subpattern_ids=subpattern_ids,
            )
            subpattern_score = 0.0
            if subpattern_ids:
                subpattern_score = round(
                    sum(self.pattern_evolution._score_subpattern(self.hierarchical_registry.get_subpattern(item), current_sequence=max(1, len(subpattern_ids))) for item in subpattern_ids)
                    / len(subpattern_ids),
                    4,
                )
            return HybridEncodingOption(
                encoding_type="hierarchical_subpatterns",
                payload=encoded.payload,
                size_bytes=encoded.total_bytes,
                subpattern_ids=subpattern_ids,
                temporal_score=subpattern_score,
                reuse_confidence=match.score,
                reasons=list(match.reasons) + [f"temporal_score={subpattern_score}"],
            )

        return None

    def _evaluate_evolved(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
        current_sequence: int,
    ) -> Optional[HybridEncodingOption]:
        decision = self.pattern_evolution.select_adaptive_encoding(
            memory_bundle=memory_bundle,
            current_sequence=current_sequence,
            include_subpatterns=True,
            update_usage=False,
        )
        if decision is None or decision.mode == "full":
            return None

        if decision.mode == "pattern" and decision.pattern_id is not None:
            pattern = self.hierarchical_registry.get_pattern(decision.pattern_id)
            encoded = self.hierarchical_codec.encode_memory_bundle_hierarchical(
                current_bundle=memory_bundle,
                registry=self.hierarchical_registry,
                pattern=pattern,
            )
            return HybridEncodingOption(
                encoding_type="evolved_pattern",
                payload=encoded.payload,
                size_bytes=encoded.total_bytes,
                pattern_id=pattern.pattern_id,
                subpattern_ids=list(pattern.subpattern_ids),
                temporal_score=decision.reinforcement_score,
                reuse_confidence=decision.reinforcement_score,
                reasons=list(decision.reasons),
                adaptive_decision=decision,
            )

        encoded = self.hierarchical_codec.encode_memory_bundle_hierarchical(
            current_bundle=memory_bundle,
            registry=self.hierarchical_registry,
            subpattern_ids=decision.subpattern_ids,
        )
        return HybridEncodingOption(
            encoding_type="evolved_subpatterns",
            payload=encoded.payload,
            size_bytes=encoded.total_bytes,
            subpattern_ids=list(decision.subpattern_ids),
            temporal_score=decision.reinforcement_score,
            reuse_confidence=decision.reinforcement_score,
            reasons=list(decision.reasons),
            adaptive_decision=decision,
        )

    @staticmethod
    def _effective_cost(option: HybridEncodingOption) -> float:
        bonus = min(1.0, (option.temporal_score * 0.6) + (option.reuse_confidence * 0.4))
        return round(option.size_bytes - bonus, 4)

    @staticmethod
    def _resolve_existing_bundle(
        memory_id: int,
        existing_memories: Sequence[StoredHybridMemory],
    ) -> Dict[str, List[List[List[int]]]]:
        for entry in existing_memories:
            if entry.memory_id == memory_id:
                return entry.resolved_bundle
        raise ValueError(f"Unknown hybrid reference memory ID: {memory_id}")


class BrainHybridStore:
    """
    In-memory storage for the selected hybrid encoding per BrAIn memory.
    """

    def __init__(
        self,
        selector: BrainHybridSelector,
    ) -> None:
        self.selector = selector
        self.entries: Dict[int, StoredHybridMemory] = {}
        self.order: List[int] = []

    def add_memory(
        self,
        memory_id: int,
        memory_bundle: Dict[str, List[List[List[int]]]],
        current_sequence: int,
    ) -> StoredHybridMemory:
        if memory_id in self.entries:
            raise ValueError(f"Memory ID already exists in hybrid store: {memory_id}")
        existing = [self.entries[item_id] for item_id in self.order]
        options = self.selector.evaluate_encoding_options(
            memory_bundle=memory_bundle,
            existing_memories=existing,
            current_sequence=current_sequence,
        )
        selected = self.selector.select_best_encoding(options)
        entry = StoredHybridMemory(
            memory_id=memory_id,
            encoding_type=selected.encoding_type,
            payload=selected.payload,
            reference_id=selected.reference_id,
            pattern_id=selected.pattern_id,
            subpattern_ids=list(selected.subpattern_ids),
            resolved_bundle=memory_bundle,
            stored_size_bytes=selected.size_bytes,
            temporal_score=selected.temporal_score,
            reuse_confidence=selected.reuse_confidence,
            reasons=list(selected.reasons),
        )
        self.entries[memory_id] = entry
        self.order.append(memory_id)
        self._apply_selection_updates(entry, selected, current_sequence=current_sequence)
        return entry

    def resolve_bundle(self, memory_id: int) -> Dict[str, List[List[List[int]]]]:
        entry = self.entries.get(memory_id)
        if entry is None:
            raise ValueError(f"Unknown hybrid memory ID: {memory_id}")
        return entry.resolved_bundle

    def _apply_selection_updates(
        self,
        entry: StoredHybridMemory,
        option: HybridEncodingOption,
        current_sequence: int,
    ) -> None:
        full_size = len(self.selector.bitpacked_codec.encode_jsonable_bundle_bitpacked(entry.resolved_bundle).payload)
        if option.encoding_type == "pattern" and option.pattern_id is not None:
            self.selector.pattern_store.record_usage(option.pattern_id)
            return

        if option.encoding_type == "hierarchical_pattern" and option.pattern_id is not None:
            self.selector.hierarchical_registry.record_pattern_usage(
                option.pattern_id,
                increment=1,
                sequence=current_sequence,
                encoded_size_bytes=option.size_bytes,
                full_size_bytes=full_size,
            )
            self.selector.hierarchical_registry.record_subpattern_usage(
                option.subpattern_ids,
                increment=1,
                sequence=current_sequence,
            )
            return

        if option.encoding_type == "hierarchical_subpatterns":
            self.selector.hierarchical_registry.record_subpattern_usage(
                option.subpattern_ids,
                increment=1,
                sequence=current_sequence,
            )
            return

        if option.encoding_type in {"evolved_pattern", "evolved_subpatterns"} and option.adaptive_decision is not None:
            self.selector.pattern_evolution._record_selection_usage(
                option.adaptive_decision,
                memory_bundle=entry.resolved_bundle,
                current_sequence=current_sequence,
            )
            self.selector.pattern_evolution.prune_registry(current_sequence=current_sequence)


def evaluate_encoding_options(
    memory_bundle: Dict[str, List[List[List[int]]]],
    selector: BrainHybridSelector,
    existing_memories: Sequence[StoredHybridMemory],
    current_sequence: int,
) -> List[HybridEncodingOption]:
    return selector.evaluate_encoding_options(memory_bundle, existing_memories, current_sequence)


def select_best_encoding(
    options: Sequence[HybridEncodingOption],
    selector: BrainHybridSelector,
) -> HybridEncodingOption:
    return selector.select_best_encoding(options)
