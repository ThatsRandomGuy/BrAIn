from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from brain_hierarchical_pattern_codec import (
    FIELD_GROUPS,
    BrainHierarchicalPatternCodec,
    BrainHierarchicalPatternRegistry,
    HierarchicalPatternCandidate,
    HierarchicalPatternDefinition,
    HierarchicalSubpatternDefinition,
    _ComposedHierarchicalStructure,
)
from brain_pattern_codec import PatternRelationTemplate, PatternSlot, SINGLE_FIELDS


GENERALIZABLE_FIELD_TYPES: Dict[str, str] = {
    "actor": "actor",
    "intent": "intent",
    "action": "action",
    "object": "object",
    "topic": "topic",
    "time": "time",
    "priority": "priority",
}

FIELD_WEIGHTS: Dict[str, float] = {
    "action": 3.0,
    "object": 3.0,
    "actor": 2.0,
    "intent": 2.0,
    "topic": 1.75,
    "priority": 1.25,
    "time": 1.0,
}

LIST_SLOT_WEIGHTS: Dict[str, float] = {
    "modifier": 1.0,
    "negation": 0.8,
    "constraint": 0.8,
    "condition": 0.8,
    "time": 0.8,
    "compare_entity": 1.0,
}


@dataclass(frozen=True)
class PatternSimilarityResult:
    score: float
    field_similarity: float
    relation_similarity: float
    subpattern_similarity: float
    slot_similarity: float
    shared_subpattern_count: int
    shared_relation_count: int
    generalized_fields: List[str]
    reasons: List[str]


@dataclass(frozen=True)
class PatternMergeResult:
    candidate: HierarchicalPatternCandidate
    subpatterns: List[HierarchicalSubpatternDefinition]
    generalized_fields: List[str]
    generalized_slot_names: List[str]
    optional_slot_names: List[str]
    generalization_level: int
    reasons: List[str]

    @property
    def generalized_slot_count(self) -> int:
        return len(self.generalized_slot_names)


@dataclass(frozen=True)
class PatternEvolutionEvent:
    source_pattern_ids: List[int]
    merged_pattern_id: int
    score: float
    generalized_fields: List[str]
    generalized_slot_names: List[str]
    optional_slot_names: List[str]
    generalization_level: int
    reasons: List[str]


@dataclass(frozen=True)
class PatternEvolutionSummary:
    before_active_pattern_count: int
    after_active_pattern_count: int
    before_active_subpattern_count: int
    after_active_subpattern_count: int
    merged_pattern_count: int
    generalized_pattern_count: int
    total_generalized_slots: int
    average_generalized_slots: float
    events: List[PatternEvolutionEvent]


@dataclass(frozen=True)
class EvolvedPatternMatch:
    pattern_id: int
    pattern_name: str
    score: float
    encoded_size_bytes: int
    slot_values: Dict[str, str]
    generalized_slot_count: int
    generalization_level: int
    is_generalized_pattern: bool
    reasons: List[str]


@dataclass(frozen=True)
class PatternReinforcementScore:
    pattern_id: int
    score: float
    frequency_signal: float
    recency_signal: float
    efficiency_signal: float
    decay_penalty: float
    reuse_frequency: float
    reasons: List[str]
    success_signal: float = 0.0
    compression_efficiency: float = 0.0
    temporal_score: float = 0.0


@dataclass(frozen=True)
class AdaptiveSelectionDecision:
    mode: str
    pattern_id: Optional[int]
    pattern_name: str
    encoded_size_bytes: int
    full_size_bytes: int
    reinforcement_score: float
    effective_cost: float
    generalization_level: int
    subpattern_ids: List[int]
    matched_subpattern_ids: List[int]
    generalized_pattern: bool
    reasons: List[str]


class BrainPatternEvolution:
    """
    Deterministic hierarchical pattern evolution for BrAIn.

    The evolution layer generalizes structurally similar hierarchical patterns
    into broader reusable patterns while keeping original pattern IDs available
    for lossless decode of existing payloads.
    """

    def __init__(
        self,
        codec: BrainHierarchicalPatternCodec,
        registry: BrainHierarchicalPatternRegistry,
        similarity_threshold: float = 0.64,
    ) -> None:
        self.codec = codec
        self.registry = registry
        self.similarity_threshold = similarity_threshold

    def compare_pattern_similarity(
        self,
        pattern_a: HierarchicalPatternDefinition | int,
        pattern_b: HierarchicalPatternDefinition | int,
    ) -> PatternSimilarityResult:
        resolved_a = self._resolve_pattern(pattern_a)
        resolved_b = self._resolve_pattern(pattern_b)
        composed_a = self.codec.compose_pattern(resolved_a, self.registry)
        composed_b = self.codec.compose_pattern(resolved_b, self.registry)

        field_similarity, generalized_fields = self._score_field_similarity(composed_a, composed_b)
        relation_similarity, shared_relation_count = self._score_relation_similarity(composed_a, composed_b)
        subpattern_similarity, shared_subpattern_count = self._score_subpattern_similarity(resolved_a, resolved_b)
        slot_similarity = self._score_slot_similarity(composed_a, composed_b)

        score = round(
            (field_similarity * 0.45)
            + (relation_similarity * 0.3)
            + (subpattern_similarity * 0.15)
            + (slot_similarity * 0.1),
            4,
        )
        reasons = [
            f"field_similarity={field_similarity}",
            f"relation_similarity={relation_similarity}",
            f"subpattern_similarity={subpattern_similarity}",
            f"slot_similarity={slot_similarity}",
            f"generalized_fields={','.join(generalized_fields) or 'none'}",
            f"shared_subpatterns={shared_subpattern_count}",
            f"shared_relations={shared_relation_count}",
        ]
        return PatternSimilarityResult(
            score=score,
            field_similarity=field_similarity,
            relation_similarity=relation_similarity,
            subpattern_similarity=subpattern_similarity,
            slot_similarity=slot_similarity,
            shared_subpattern_count=shared_subpattern_count,
            shared_relation_count=shared_relation_count,
            generalized_fields=generalized_fields,
            reasons=reasons,
        )

    def merge_patterns(
        self,
        pattern_a: HierarchicalPatternDefinition | int,
        pattern_b: HierarchicalPatternDefinition | int,
    ) -> PatternMergeResult:
        resolved_a = self._resolve_pattern(pattern_a)
        resolved_b = self._resolve_pattern(pattern_b)
        similarity = self.compare_pattern_similarity(resolved_a, resolved_b)
        composed_a = self.codec.compose_pattern(resolved_a, self.registry)
        composed_b = self.codec.compose_pattern(resolved_b, self.registry)
        merged = self._merge_composed_structures(composed_a, composed_b)
        subpatterns, candidate = self._build_candidate_from_composed(
            stable_fields=merged.stable_fields,
            variable_slots=merged.variable_slots,
            relation_templates=merged.relation_templates,
            notes=[
                f"merged_from={resolved_a.pattern_id},{resolved_b.pattern_id}",
                f"generalized_fields={','.join(similarity.generalized_fields) or 'none'}",
            ],
        )
        generalization_level = max(resolved_a.generalization_level, resolved_b.generalization_level) + 1
        generalized_slot_names = [
            slot.slot_name
            for slot in candidate.variable_slots
            if slot.path in similarity.generalized_fields
        ]
        optional_slot_names = [slot.slot_name for slot in candidate.variable_slots if slot.optional]
        reasons = list(similarity.reasons) + [
            f"subpattern_count={len(subpatterns)}",
            f"generalization_level={generalization_level}",
        ]
        return PatternMergeResult(
            candidate=candidate,
            subpatterns=[
                HierarchicalSubpatternDefinition(
                    subpattern_id=0,
                    kind=subpattern.kind,
                    name=subpattern.name,
                    fingerprint=subpattern.fingerprint,
                    created_sequence=0,
                    usage_count=0,
                    stable_fields=subpattern.stable_fields,
                    variable_slots=subpattern.variable_slots,
                    relation_templates=subpattern.relation_templates,
                    notes=subpattern.notes,
                    active=True,
                    generalization_level=generalization_level,
                    merged_from_subpattern_ids=[],
                    last_used_sequence=0,
                    reuse_frequency=0.0,
                    cooccurrence_count=0,
                    temporal_score=0.0,
                )
                for subpattern in subpatterns
            ],
            generalized_fields=similarity.generalized_fields,
            generalized_slot_names=sorted(set(generalized_slot_names)),
            optional_slot_names=sorted(set(optional_slot_names)),
            generalization_level=generalization_level,
            reasons=reasons,
        )

    def evolve_registry(self, threshold: Optional[float] = None) -> PatternEvolutionSummary:
        active_threshold = self.similarity_threshold if threshold is None else threshold
        before_active_pattern_count = len(self.registry.all_active_patterns())
        before_active_subpattern_count = len(self.registry.all_active_subpatterns())
        events: List[PatternEvolutionEvent] = []

        while True:
            pair = self._find_best_merge_pair(active_threshold)
            if pair is None:
                break

            pattern_a, pattern_b, similarity, merge_result = pair
            merged_pattern = self._materialize_merge(pattern_a, pattern_b, merge_result)
            events.append(
                PatternEvolutionEvent(
                    source_pattern_ids=[pattern_a.pattern_id, pattern_b.pattern_id],
                    merged_pattern_id=merged_pattern.pattern_id,
                    score=similarity.score,
                    generalized_fields=list(merge_result.generalized_fields),
                    generalized_slot_names=list(merge_result.generalized_slot_names),
                    optional_slot_names=list(merge_result.optional_slot_names),
                    generalization_level=merge_result.generalization_level,
                    reasons=list(merge_result.reasons),
                )
            )

        after_active_patterns = self.registry.all_active_patterns()
        generalized_patterns = [pattern for pattern in after_active_patterns if pattern.generalization_level > 0]
        total_generalized_slots = sum(pattern.generalized_slot_count for pattern in generalized_patterns)
        average_generalized_slots = round(
            total_generalized_slots / len(generalized_patterns),
            3,
        ) if generalized_patterns else 0.0
        return PatternEvolutionSummary(
            before_active_pattern_count=before_active_pattern_count,
            after_active_pattern_count=len(after_active_patterns),
            before_active_subpattern_count=before_active_subpattern_count,
            after_active_subpattern_count=len(self.registry.all_active_subpatterns()),
            merged_pattern_count=len(events),
            generalized_pattern_count=len(generalized_patterns),
            total_generalized_slots=total_generalized_slots,
            average_generalized_slots=average_generalized_slots,
            events=events,
        )

    def find_best_pattern(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
        current_sequence: Optional[int] = None,
    ) -> Optional[EvolvedPatternMatch]:
        decision = self.recommend_pattern(
            memory_bundle=memory_bundle,
            current_sequence=current_sequence,
            include_subpatterns=False,
            update_usage=False,
        )
        if decision is None or decision.mode != "pattern" or decision.pattern_id is None:
            return None
        pattern = self.registry.get_pattern(decision.pattern_id)
        memory = self.codec.codec.decode_memory_bundle(memory_bundle)
        composed = self.codec.compose_pattern(pattern, self.registry)
        slot_values = self._try_extract_slot_values(memory, composed.variable_slots) or {}
        return EvolvedPatternMatch(
            pattern_id=pattern.pattern_id,
            pattern_name=pattern.name,
            score=decision.reinforcement_score,
            encoded_size_bytes=decision.encoded_size_bytes,
            slot_values=slot_values,
            generalized_slot_count=pattern.generalized_slot_count,
            generalization_level=pattern.generalization_level,
            is_generalized_pattern=pattern.generalization_level > 0,
            reasons=list(decision.reasons),
        )

    def recommend_pattern(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
        current_sequence: Optional[int] = None,
        include_subpatterns: bool = True,
        update_usage: bool = False,
    ) -> Optional[AdaptiveSelectionDecision]:
        return self.select_adaptive_encoding(
            memory_bundle=memory_bundle,
            current_sequence=current_sequence,
            include_subpatterns=include_subpatterns,
            update_usage=update_usage,
        )

    def score_pattern(
        self,
        pattern: HierarchicalPatternDefinition | int,
        current_sequence: Optional[int] = None,
    ) -> PatternReinforcementScore:
        resolved = self._resolve_pattern(pattern)
        sequence = max(
            current_sequence or 0,
            resolved.last_used_sequence,
            resolved.created_sequence,
            1,
        )
        effective_last_used = resolved.last_used_sequence or resolved.created_sequence or 1
        age = max(0, sequence - effective_last_used)
        frequency_signal = round(min(1.0, resolved.usage_count / max(1, sequence)), 4)
        recency_signal = round(1.0 / (1.0 + age), 4)
        reuse_frequency = round(
            resolved.reuse_frequency if resolved.reuse_frequency > 0.0 else resolved.usage_count / max(1, sequence),
            4,
        )
        if resolved.compression_efficiency > 0.0:
            compression_efficiency = resolved.compression_efficiency
        elif resolved.average_compression_ratio > 0.0:
            compression_efficiency = round(max(0.0, min(1.0, 1.0 - resolved.average_compression_ratio)), 4)
        elif resolved.total_full_size_bytes > 0:
            compression_efficiency = round(
                max(0.0, min(1.0, 1.0 - (resolved.total_encoded_size_bytes / resolved.total_full_size_bytes))),
                4,
            )
        else:
            compression_efficiency = 0.0
        efficiency_signal = compression_efficiency
        total_outcomes = resolved.success_count + resolved.failure_count
        success_signal = round(resolved.success_rate if total_outcomes > 0 else 0.5, 4)
        decay_penalty = round(min(0.75, age / max(4, sequence)), 4)
        score = round(
            max(
                0.0,
                min(
                    1.0,
                    (frequency_signal * 0.22)
                    + (recency_signal * 0.18)
                    + (efficiency_signal * 0.18)
                    + (min(1.0, reuse_frequency) * 0.12)
                    + (success_signal * 0.25)
                    - (decay_penalty * 0.15),
                ),
            ),
            4,
        )
        return PatternReinforcementScore(
            pattern_id=resolved.pattern_id,
            score=score,
            frequency_signal=frequency_signal,
            recency_signal=recency_signal,
            efficiency_signal=efficiency_signal,
            decay_penalty=decay_penalty,
            reuse_frequency=reuse_frequency,
            reasons=[
                f"frequency_signal={frequency_signal}",
                f"recency_signal={recency_signal}",
                f"efficiency_signal={efficiency_signal}",
                f"reuse_frequency={reuse_frequency}",
                f"success_signal={success_signal}",
                f"compression_efficiency={compression_efficiency}",
                f"decay_penalty={decay_penalty}",
            ],
            success_signal=success_signal,
            compression_efficiency=compression_efficiency,
            temporal_score=score,
        )

    def select_adaptive_encoding(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
        current_sequence: Optional[int] = None,
        include_subpatterns: bool = True,
        update_usage: bool = False,
    ) -> Optional[AdaptiveSelectionDecision]:
        memory = self.codec.codec.decode_memory_bundle(memory_bundle)
        full_size_bytes = len(self.codec.full_bundle_bytes(memory_bundle))
        sequence = max(
            current_sequence or 0,
            max([pattern.last_used_sequence for pattern in self.registry.all_patterns()] + [0]),
            max([pattern.created_sequence for pattern in self.registry.all_patterns()] + [0]),
            1,
        )

        decisions: List[AdaptiveSelectionDecision] = []

        for pattern in self.registry.all_active_patterns():
            composed = self.codec.compose_pattern(pattern, self.registry)
            slot_values = self._try_extract_slot_values(memory, composed.variable_slots)
            if slot_values is None:
                continue
            reconstructed = self.codec.codec.encode_memory_bundle(
                self.codec._reconstruct_from_composed(composed, slot_values)
            )
            if reconstructed != memory_bundle:
                continue

            encoded = self.codec.encode_memory_bundle_hierarchical(
                current_bundle=memory_bundle,
                registry=self.registry,
                pattern=pattern,
            )
            reinforcement = self.score_pattern(pattern, current_sequence=sequence)
            decisions.append(
                AdaptiveSelectionDecision(
                    mode="pattern",
                    pattern_id=pattern.pattern_id,
                    pattern_name=pattern.name,
                    encoded_size_bytes=encoded.total_bytes,
                    full_size_bytes=full_size_bytes,
                    reinforcement_score=reinforcement.score,
                    effective_cost=self._effective_cost(encoded.total_bytes, reinforcement.score),
                    generalization_level=pattern.generalization_level,
                    subpattern_ids=list(pattern.subpattern_ids),
                    matched_subpattern_ids=list(pattern.subpattern_ids),
                    generalized_pattern=pattern.generalization_level > 0,
                    reasons=[
                        "lossless_pattern_match",
                        *reinforcement.reasons,
                        f"encoded_size_bytes={encoded.total_bytes}",
                        f"effective_cost={self._effective_cost(encoded.total_bytes, reinforcement.score)}",
                    ],
                )
            )

        if include_subpatterns:
            subpattern_decision = self._build_subpattern_decision(
                memory_bundle=memory_bundle,
                current_sequence=sequence,
                full_size_bytes=full_size_bytes,
            )
            if subpattern_decision is not None:
                decisions.append(subpattern_decision)

        decisions.append(
            AdaptiveSelectionDecision(
                mode="full",
                pattern_id=None,
                pattern_name="",
                encoded_size_bytes=full_size_bytes,
                full_size_bytes=full_size_bytes,
                reinforcement_score=0.0,
                effective_cost=float(full_size_bytes),
                generalization_level=0,
                subpattern_ids=[],
                matched_subpattern_ids=[],
                generalized_pattern=False,
                reasons=["fallback_full_bundle"],
            )
        )

        if not decisions:
            return None

        selected = min(
            decisions,
            key=lambda item: (
                item.effective_cost,
                item.encoded_size_bytes,
                -item.reinforcement_score,
                -item.generalization_level,
                item.pattern_id if item.pattern_id is not None else 10**9,
            ),
        )

        if update_usage:
            self._record_selection_usage(selected, memory_bundle=memory_bundle, current_sequence=sequence)

        return selected

    def record_reasoning_outcome(
        self,
        decision: Optional[AdaptiveSelectionDecision],
        success: bool,
        current_sequence: Optional[int] = None,
    ) -> None:
        if decision is None:
            return
        sequence = max(current_sequence or 0, 1)
        if decision.mode == "pattern" and decision.pattern_id is not None:
            self.registry.record_pattern_outcome(
                pattern_id=decision.pattern_id,
                success=success,
            )
            self._refresh_pattern_temporal_state(decision.pattern_id, sequence)
            for subpattern_id in decision.subpattern_ids:
                self._refresh_subpattern_temporal_state(subpattern_id, sequence)
            return
        if decision.mode == "subpatterns":
            for subpattern_id in decision.subpattern_ids:
                self._refresh_subpattern_temporal_state(subpattern_id, sequence)

    def refresh_registry_temporal_scores(
        self,
        current_sequence: Optional[int] = None,
    ) -> None:
        sequence = max(current_sequence or 0, 1)
        for pattern in self.registry.all_patterns():
            self._refresh_pattern_temporal_state(pattern.pattern_id, sequence)
        for subpattern in self.registry.all_subpatterns():
            self._refresh_subpattern_temporal_state(subpattern.subpattern_id, sequence)

    def promote_frequent_subpatterns(
        self,
        current_sequence: Optional[int] = None,
        min_cooccurrence: int = 3,
        max_promotions: int = 8,
    ) -> List[Dict[str, object]]:
        sequence = max(current_sequence or 0, 1)
        promotions: List[Dict[str, object]] = []
        ranked_pairs = sorted(
            self.registry.subpattern_cooccurrence.items(),
            key=lambda item: (-int(item[1]), item[0]),
        )
        for key, count in ranked_pairs:
            if int(count) < min_cooccurrence:
                continue
            subpattern_ids = [int(item) for item in str(key).split(":") if str(item).isdigit()]
            if len(subpattern_ids) < 2:
                continue
            try:
                subpatterns = [self.registry.get_subpattern(subpattern_id) for subpattern_id in subpattern_ids]
            except Exception:
                continue
            if not all(subpattern.active for subpattern in subpatterns):
                continue
            candidate = HierarchicalPatternCandidate(
                fingerprint=self.codec._fingerprint_hierarchical_candidate(subpatterns),
                subpatterns=list(subpatterns),
                variable_slots=self.codec._compose_subpatterns(subpatterns).variable_slots,
                notes=[
                    f"promoted_from_cooccurrence={','.join(str(item) for item in subpattern_ids)}",
                    f"cooccurrence_count={int(count)}",
                ],
            )
            ensured_pattern, created = self.registry.ensure_pattern(candidate, subpattern_ids)
            updated_pattern = HierarchicalPatternDefinition(
                pattern_id=ensured_pattern.pattern_id,
                name=ensured_pattern.name,
                fingerprint=ensured_pattern.fingerprint,
                created_sequence=ensured_pattern.created_sequence,
                usage_count=max(ensured_pattern.usage_count, int(count)),
                subpattern_ids=list(subpattern_ids),
                variable_slots=list(candidate.variable_slots),
                notes=list(dict.fromkeys(list(ensured_pattern.notes) + list(candidate.notes))),
                active=True,
                generalization_level=max(1, ensured_pattern.generalization_level),
                generalized_slot_count=max(ensured_pattern.generalized_slot_count, len(candidate.variable_slots)),
                merged_from_pattern_ids=list(ensured_pattern.merged_from_pattern_ids),
                superseded_by_pattern_id=ensured_pattern.superseded_by_pattern_id,
                last_used_sequence=max(ensured_pattern.last_used_sequence, sequence),
                reuse_frequency=max(ensured_pattern.reuse_frequency, round(int(count) / max(1, sequence), 4)),
                total_encoded_size_bytes=ensured_pattern.total_encoded_size_bytes,
                total_full_size_bytes=ensured_pattern.total_full_size_bytes,
                average_compression_ratio=ensured_pattern.average_compression_ratio,
                success_count=ensured_pattern.success_count,
                failure_count=ensured_pattern.failure_count,
                success_rate=ensured_pattern.success_rate,
                compression_efficiency=ensured_pattern.compression_efficiency,
                temporal_score=ensured_pattern.temporal_score,
            )
            self.registry.update_pattern(updated_pattern)
            self._refresh_pattern_temporal_state(updated_pattern.pattern_id, sequence)
            promotions.append(
                {
                    "event_type": "promote_subpatterns",
                    "pattern_id": updated_pattern.pattern_id,
                    "subpattern_ids": list(subpattern_ids),
                    "cooccurrence_count": int(count),
                    "created": created,
                }
            )
            if len(promotions) >= max_promotions:
                break
        return promotions

    def split_broad_patterns(
        self,
        current_sequence: Optional[int] = None,
        min_usage: int = 6,
        max_success_rate: float = 0.75,
        min_variable_slots: int = 3,
    ) -> List[Dict[str, object]]:
        sequence = max(current_sequence or 0, 1)
        splits: List[Dict[str, object]] = []
        for pattern in self.registry.all_active_patterns():
            total_outcomes = pattern.success_count + pattern.failure_count
            if pattern.usage_count < min_usage:
                continue
            if len(pattern.variable_slots) < min_variable_slots and pattern.generalized_slot_count < 2:
                continue
            if total_outcomes > 0 and pattern.success_rate > max_success_rate:
                continue
            groups: Dict[str, List[int]] = {}
            for subpattern_id in pattern.subpattern_ids:
                try:
                    subpattern = self.registry.get_subpattern(subpattern_id)
                except Exception:
                    continue
                groups.setdefault(subpattern.kind, []).append(subpattern_id)
            if len(groups) < 2:
                continue
            split_pattern_ids: List[int] = []
            for group_name in sorted(groups):
                group_ids = groups[group_name]
                try:
                    subpatterns = [self.registry.get_subpattern(subpattern_id) for subpattern_id in group_ids]
                except Exception:
                    continue
                candidate = HierarchicalPatternCandidate(
                    fingerprint=self.codec._fingerprint_hierarchical_candidate(subpatterns),
                    subpatterns=list(subpatterns),
                    variable_slots=self.codec._compose_subpatterns(subpatterns).variable_slots,
                    notes=[
                        f"split_from_pattern={pattern.pattern_id}",
                        f"split_group={group_name}",
                    ],
                )
                ensured_pattern, _ = self.registry.ensure_pattern(candidate, group_ids)
                if ensured_pattern.pattern_id == pattern.pattern_id:
                    continue
                updated_child = HierarchicalPatternDefinition(
                    pattern_id=ensured_pattern.pattern_id,
                    name=ensured_pattern.name,
                    fingerprint=ensured_pattern.fingerprint,
                    created_sequence=ensured_pattern.created_sequence,
                    usage_count=max(ensured_pattern.usage_count, max(1, pattern.usage_count // max(1, len(groups)))),
                    subpattern_ids=list(group_ids),
                    variable_slots=list(candidate.variable_slots),
                    notes=list(dict.fromkeys(list(ensured_pattern.notes) + list(candidate.notes))),
                    active=True,
                    generalization_level=max(0, pattern.generalization_level - 1),
                    generalized_slot_count=max(ensured_pattern.generalized_slot_count, len(candidate.variable_slots)),
                    merged_from_pattern_ids=sorted(set(list(ensured_pattern.merged_from_pattern_ids) + [pattern.pattern_id])),
                    superseded_by_pattern_id=ensured_pattern.superseded_by_pattern_id,
                    last_used_sequence=max(ensured_pattern.last_used_sequence, sequence),
                    reuse_frequency=max(ensured_pattern.reuse_frequency, pattern.reuse_frequency),
                    total_encoded_size_bytes=ensured_pattern.total_encoded_size_bytes,
                    total_full_size_bytes=ensured_pattern.total_full_size_bytes,
                    average_compression_ratio=ensured_pattern.average_compression_ratio,
                    success_count=ensured_pattern.success_count,
                    failure_count=ensured_pattern.failure_count,
                    success_rate=ensured_pattern.success_rate,
                    compression_efficiency=ensured_pattern.compression_efficiency,
                    temporal_score=ensured_pattern.temporal_score,
                )
                self.registry.update_pattern(updated_child)
                self._refresh_pattern_temporal_state(updated_child.pattern_id, sequence)
                split_pattern_ids.append(updated_child.pattern_id)
            if not split_pattern_ids:
                continue
            self.registry.update_pattern(
                HierarchicalPatternDefinition(
                    pattern_id=pattern.pattern_id,
                    name=pattern.name,
                    fingerprint=pattern.fingerprint,
                    created_sequence=pattern.created_sequence,
                    usage_count=pattern.usage_count,
                    subpattern_ids=pattern.subpattern_ids,
                    variable_slots=pattern.variable_slots,
                    notes=list(dict.fromkeys(list(pattern.notes) + [f"split_into={','.join(str(item) for item in split_pattern_ids)}"])),
                    active=False,
                    generalization_level=pattern.generalization_level,
                    generalized_slot_count=pattern.generalized_slot_count,
                    merged_from_pattern_ids=list(pattern.merged_from_pattern_ids),
                    superseded_by_pattern_id=pattern.superseded_by_pattern_id,
                    last_used_sequence=pattern.last_used_sequence,
                    reuse_frequency=pattern.reuse_frequency,
                    total_encoded_size_bytes=pattern.total_encoded_size_bytes,
                    total_full_size_bytes=pattern.total_full_size_bytes,
                    average_compression_ratio=pattern.average_compression_ratio,
                    success_count=pattern.success_count,
                    failure_count=pattern.failure_count,
                    success_rate=pattern.success_rate,
                    compression_efficiency=pattern.compression_efficiency,
                    temporal_score=pattern.temporal_score,
                )
            )
            splits.append(
                {
                    "event_type": "split_pattern",
                    "source_pattern_id": pattern.pattern_id,
                    "split_pattern_ids": split_pattern_ids,
                    "usage_count": pattern.usage_count,
                    "success_rate": pattern.success_rate,
                }
            )
        return splits

    def run_adaptive_cycle(
        self,
        current_sequence: Optional[int] = None,
        threshold: Optional[float] = None,
        min_cooccurrence: int = 3,
    ) -> Dict[str, object]:
        sequence = max(current_sequence or 0, 1)
        merge_summary = self.evolve_registry(threshold=threshold)
        promotions = self.promote_frequent_subpatterns(
            current_sequence=sequence,
            min_cooccurrence=min_cooccurrence,
        )
        splits = self.split_broad_patterns(current_sequence=sequence)
        prune_info = self.prune_registry(current_sequence=sequence)
        self.refresh_registry_temporal_scores(current_sequence=sequence)
        return {
            "merge_summary": {
                "before_active_pattern_count": merge_summary.before_active_pattern_count,
                "after_active_pattern_count": merge_summary.after_active_pattern_count,
                "before_active_subpattern_count": merge_summary.before_active_subpattern_count,
                "after_active_subpattern_count": merge_summary.after_active_subpattern_count,
                "merged_pattern_count": merge_summary.merged_pattern_count,
                "generalized_pattern_count": merge_summary.generalized_pattern_count,
                "total_generalized_slots": merge_summary.total_generalized_slots,
                "average_generalized_slots": merge_summary.average_generalized_slots,
                "events": [
                    {
                        "event_type": "merge_patterns",
                        "source_pattern_ids": event.source_pattern_ids,
                        "merged_pattern_id": event.merged_pattern_id,
                        "score": event.score,
                        "generalized_fields": event.generalized_fields,
                        "generalized_slot_names": event.generalized_slot_names,
                        "optional_slot_names": event.optional_slot_names,
                        "generalization_level": event.generalization_level,
                        "reasons": event.reasons,
                    }
                    for event in merge_summary.events
                ],
            },
            "promotions": promotions,
            "splits": splits,
            "prune_info": prune_info,
        }

    def prune_registry(
        self,
        current_sequence: Optional[int] = None,
        min_pattern_score: float = 0.18,
        min_subpattern_score: float = 0.15,
        min_idle: int = 4,
        max_usage_for_prune: int = 2,
    ) -> Dict[str, int]:
        sequence = max(current_sequence or 0, 1)
        pruned_patterns = 0
        pruned_subpatterns = 0

        referenced_subpattern_ids = {
            subpattern_id
            for pattern in self.registry.all_active_patterns()
            for subpattern_id in pattern.subpattern_ids
        }

        for pattern in self.registry.all_active_patterns():
            score = self.score_pattern(pattern, current_sequence=sequence)
            age = max(0, sequence - (pattern.last_used_sequence or pattern.created_sequence or sequence))
            if (
                score.score < min_pattern_score
                and pattern.usage_count <= max_usage_for_prune
                and age >= min_idle
            ):
                self.registry.update_pattern(
                    HierarchicalPatternDefinition(
                        pattern_id=pattern.pattern_id,
                        name=pattern.name,
                        fingerprint=pattern.fingerprint,
                        created_sequence=pattern.created_sequence,
                        usage_count=pattern.usage_count,
                        subpattern_ids=pattern.subpattern_ids,
                        variable_slots=pattern.variable_slots,
                        notes=list(dict.fromkeys(list(pattern.notes) + [f"pruned_at_sequence={sequence}"])),
                        active=False,
                        generalization_level=pattern.generalization_level,
                        generalized_slot_count=pattern.generalized_slot_count,
                        merged_from_pattern_ids=list(pattern.merged_from_pattern_ids),
                        superseded_by_pattern_id=pattern.superseded_by_pattern_id,
                        last_used_sequence=pattern.last_used_sequence,
                        reuse_frequency=pattern.reuse_frequency,
                        total_encoded_size_bytes=pattern.total_encoded_size_bytes,
                        total_full_size_bytes=pattern.total_full_size_bytes,
                        average_compression_ratio=pattern.average_compression_ratio,
                        success_count=pattern.success_count,
                        failure_count=pattern.failure_count,
                        success_rate=pattern.success_rate,
                        compression_efficiency=pattern.compression_efficiency,
                        temporal_score=pattern.temporal_score,
                    )
                )
                pruned_patterns += 1

        for subpattern in self.registry.all_active_subpatterns():
            if subpattern.subpattern_id in referenced_subpattern_ids:
                continue
            subpattern_score = self._score_subpattern(subpattern, current_sequence=sequence)
            age = max(0, sequence - (subpattern.last_used_sequence or subpattern.created_sequence or sequence))
            if (
                subpattern_score < min_subpattern_score
                and subpattern.usage_count <= max_usage_for_prune
                and age >= min_idle
            ):
                self.registry.update_subpattern(
                    HierarchicalSubpatternDefinition(
                        subpattern_id=subpattern.subpattern_id,
                        kind=subpattern.kind,
                        name=subpattern.name,
                        fingerprint=subpattern.fingerprint,
                        created_sequence=subpattern.created_sequence,
                        usage_count=subpattern.usage_count,
                        stable_fields=subpattern.stable_fields,
                        variable_slots=subpattern.variable_slots,
                        relation_templates=subpattern.relation_templates,
                        notes=list(dict.fromkeys(list(subpattern.notes) + [f"pruned_at_sequence={sequence}"])),
                        active=False,
                        generalization_level=subpattern.generalization_level,
                        merged_from_subpattern_ids=list(subpattern.merged_from_subpattern_ids),
                        last_used_sequence=subpattern.last_used_sequence,
                        reuse_frequency=subpattern.reuse_frequency,
                        cooccurrence_count=subpattern.cooccurrence_count,
                        temporal_score=subpattern.temporal_score,
                    )
                )
                pruned_subpatterns += 1

        return {
            "pruned_patterns": pruned_patterns,
            "pruned_subpatterns": pruned_subpatterns,
        }

    def _find_best_merge_pair(
        self,
        threshold: float,
    ) -> Optional[Tuple[HierarchicalPatternDefinition, HierarchicalPatternDefinition, PatternSimilarityResult, PatternMergeResult]]:
        active_patterns = self.registry.all_active_patterns()
        best: Optional[Tuple[HierarchicalPatternDefinition, HierarchicalPatternDefinition, PatternSimilarityResult, PatternMergeResult]] = None
        best_key: Optional[Tuple[float, int, int, int, int]] = None

        for index, pattern_a in enumerate(active_patterns):
            for pattern_b in active_patterns[index + 1:]:
                similarity = self.compare_pattern_similarity(pattern_a, pattern_b)
                if similarity.score < threshold:
                    continue
                merge_result = self.merge_patterns(pattern_a, pattern_b)
                candidate_key = (
                    similarity.score,
                    similarity.shared_subpattern_count,
                    similarity.shared_relation_count,
                    -(pattern_a.pattern_id + pattern_b.pattern_id),
                    -max(pattern_a.generalization_level, pattern_b.generalization_level),
                )
                if best_key is not None and candidate_key <= best_key:
                    continue
                best = (pattern_a, pattern_b, similarity, merge_result)
                best_key = candidate_key

        return best

    def _materialize_merge(
        self,
        pattern_a: HierarchicalPatternDefinition,
        pattern_b: HierarchicalPatternDefinition,
        merge_result: PatternMergeResult,
    ) -> HierarchicalPatternDefinition:
        merged_from_pattern_ids = sorted(
            set(pattern_a.merged_from_pattern_ids + pattern_b.merged_from_pattern_ids + [pattern_a.pattern_id, pattern_b.pattern_id])
        )
        merged_from_subpattern_ids = sorted(set(pattern_a.subpattern_ids + pattern_b.subpattern_ids))

        ensured_subpattern_ids: List[int] = []
        for subpattern in merge_result.subpatterns:
            candidate = HierarchicalSubpatternDefinition(
                subpattern_id=0,
                kind=subpattern.kind,
                name=subpattern.name,
                fingerprint=subpattern.fingerprint,
                created_sequence=0,
                usage_count=0,
                stable_fields=subpattern.stable_fields,
                variable_slots=subpattern.variable_slots,
                relation_templates=subpattern.relation_templates,
                notes=list(subpattern.notes),
                active=True,
                generalization_level=merge_result.generalization_level,
                merged_from_subpattern_ids=list(merged_from_subpattern_ids),
                last_used_sequence=max(pattern_a.last_used_sequence, pattern_b.last_used_sequence),
                reuse_frequency=0.0,
                cooccurrence_count=0,
                temporal_score=0.0,
            )
            ensured, _ = self.registry.ensure_subpattern(candidate)
            if (
                ensured.generalization_level < merge_result.generalization_level
                or not ensured.merged_from_subpattern_ids
            ):
                self.registry.update_subpattern(
                    HierarchicalSubpatternDefinition(
                        subpattern_id=ensured.subpattern_id,
                        kind=ensured.kind,
                        name=ensured.name,
                        fingerprint=ensured.fingerprint,
                        created_sequence=ensured.created_sequence,
                        usage_count=max(ensured.usage_count, pattern_a.usage_count + pattern_b.usage_count),
                        stable_fields=ensured.stable_fields,
                        variable_slots=ensured.variable_slots,
                        relation_templates=ensured.relation_templates,
                        notes=list(dict.fromkeys(list(ensured.notes) + list(candidate.notes))),
                        active=True,
                        generalization_level=max(ensured.generalization_level, merge_result.generalization_level),
                        merged_from_subpattern_ids=list(merged_from_subpattern_ids),
                        last_used_sequence=max(ensured.last_used_sequence, pattern_a.last_used_sequence, pattern_b.last_used_sequence),
                        reuse_frequency=max(ensured.reuse_frequency, candidate.reuse_frequency),
                        cooccurrence_count=max(ensured.cooccurrence_count, candidate.cooccurrence_count),
                        temporal_score=max(ensured.temporal_score, candidate.temporal_score),
                    )
                )
                ensured = self.registry.get_subpattern(ensured.subpattern_id)
            ensured_subpattern_ids.append(ensured.subpattern_id)

        ensured_pattern, _ = self.registry.ensure_pattern(
            candidate=merge_result.candidate,
            subpattern_ids=ensured_subpattern_ids,
        )
        updated_pattern = HierarchicalPatternDefinition(
            pattern_id=ensured_pattern.pattern_id,
            name=ensured_pattern.name,
            fingerprint=ensured_pattern.fingerprint,
            created_sequence=ensured_pattern.created_sequence,
            usage_count=max(ensured_pattern.usage_count, pattern_a.usage_count + pattern_b.usage_count),
            subpattern_ids=list(ensured_subpattern_ids),
            variable_slots=list(merge_result.candidate.variable_slots),
            notes=list(dict.fromkeys(list(ensured_pattern.notes) + list(merge_result.candidate.notes) + list(merge_result.reasons))),
            active=True,
            generalization_level=max(ensured_pattern.generalization_level, merge_result.generalization_level),
            generalized_slot_count=max(ensured_pattern.generalized_slot_count, merge_result.generalized_slot_count),
            merged_from_pattern_ids=merged_from_pattern_ids,
            superseded_by_pattern_id=None,
            last_used_sequence=max(pattern_a.last_used_sequence, pattern_b.last_used_sequence),
            reuse_frequency=max(pattern_a.reuse_frequency, pattern_b.reuse_frequency),
            total_encoded_size_bytes=pattern_a.total_encoded_size_bytes + pattern_b.total_encoded_size_bytes,
            total_full_size_bytes=pattern_a.total_full_size_bytes + pattern_b.total_full_size_bytes,
            average_compression_ratio=round(
                (pattern_a.total_encoded_size_bytes + pattern_b.total_encoded_size_bytes)
                / max(1, pattern_a.total_full_size_bytes + pattern_b.total_full_size_bytes),
                4,
            ) if (pattern_a.total_full_size_bytes + pattern_b.total_full_size_bytes) > 0 else 0.0,
            success_count=pattern_a.success_count + pattern_b.success_count,
            failure_count=pattern_a.failure_count + pattern_b.failure_count,
            success_rate=round(
                (pattern_a.success_count + pattern_b.success_count)
                / max(1, pattern_a.success_count + pattern_b.success_count + pattern_a.failure_count + pattern_b.failure_count),
                4,
            ),
            compression_efficiency=round(
                max(
                    0.0,
                    min(
                        1.0,
                        1.0
                        - (
                            (pattern_a.total_encoded_size_bytes + pattern_b.total_encoded_size_bytes)
                            / max(1, pattern_a.total_full_size_bytes + pattern_b.total_full_size_bytes)
                        ),
                    ),
                ),
                4,
            ) if (pattern_a.total_full_size_bytes + pattern_b.total_full_size_bytes) > 0 else max(pattern_a.compression_efficiency, pattern_b.compression_efficiency),
            temporal_score=max(pattern_a.temporal_score, pattern_b.temporal_score),
        )
        self.registry.update_pattern(updated_pattern)

        for source_pattern in (pattern_a, pattern_b):
            self.registry.update_pattern(
                HierarchicalPatternDefinition(
                    pattern_id=source_pattern.pattern_id,
                    name=source_pattern.name,
                    fingerprint=source_pattern.fingerprint,
                    created_sequence=source_pattern.created_sequence,
                    usage_count=source_pattern.usage_count,
                    subpattern_ids=source_pattern.subpattern_ids,
                    variable_slots=source_pattern.variable_slots,
                    notes=list(dict.fromkeys(list(source_pattern.notes) + [f"superseded_by={updated_pattern.pattern_id}"])),
                    active=False,
                    generalization_level=source_pattern.generalization_level,
                    generalized_slot_count=source_pattern.generalized_slot_count,
                    merged_from_pattern_ids=list(source_pattern.merged_from_pattern_ids),
                    superseded_by_pattern_id=updated_pattern.pattern_id,
                    last_used_sequence=source_pattern.last_used_sequence,
                    reuse_frequency=source_pattern.reuse_frequency,
                    total_encoded_size_bytes=source_pattern.total_encoded_size_bytes,
                    total_full_size_bytes=source_pattern.total_full_size_bytes,
                    average_compression_ratio=source_pattern.average_compression_ratio,
                    success_count=source_pattern.success_count,
                    failure_count=source_pattern.failure_count,
                    success_rate=source_pattern.success_rate,
                    compression_efficiency=source_pattern.compression_efficiency,
                    temporal_score=source_pattern.temporal_score,
                )
            )

        return updated_pattern

    def _build_candidate_from_composed(
        self,
        stable_fields: Dict[str, str],
        variable_slots: Sequence[PatternSlot],
        relation_templates: Sequence[PatternRelationTemplate],
        notes: Sequence[str],
    ) -> Tuple[List[HierarchicalSubpatternDefinition], HierarchicalPatternCandidate]:
        subpatterns: List[HierarchicalSubpatternDefinition] = []
        remaining_stable_fields = dict(stable_fields)
        normalized_slots = self.codec._normalize_slots(variable_slots)
        assigned_relation_indexes: set[int] = set()

        for group in FIELD_GROUPS:
            grouped_fields = {
                field: remaining_stable_fields[field]
                for field in group
                if field in remaining_stable_fields
            }
            if not grouped_fields:
                continue
            for field in grouped_fields:
                remaining_stable_fields.pop(field, None)
            subpatterns.append(
                self.codec._make_subpattern_candidate(
                    kind="field_group",
                    name=f"fields_{'_'.join(group)}",
                    stable_fields=grouped_fields,
                    variable_slots=[],
                    relation_templates=[],
                    notes=[f"stable_fields={','.join(sorted(grouped_fields))}"],
                )
            )

        for field in sorted(remaining_stable_fields):
            subpatterns.append(
                self.codec._make_subpattern_candidate(
                    kind="field_group",
                    name=f"field_{field}",
                    stable_fields={field: remaining_stable_fields[field]},
                    variable_slots=[],
                    relation_templates=[],
                    notes=[f"stable_field={field}"],
                )
            )

        for slot in normalized_slots:
            related_relations = [
                template
                for template in relation_templates
                if template.relation_index not in assigned_relation_indexes
                and self.codec._relation_references_slot(template, slot.slot_name)
            ]
            for template in related_relations:
                assigned_relation_indexes.add(template.relation_index)
            subpatterns.append(
                self.codec._make_subpattern_candidate(
                    kind="attachment",
                    name=f"attachment_{slot.slot_name}",
                    stable_fields={},
                    variable_slots=[slot],
                    relation_templates=related_relations,
                    notes=[f"slot_path={slot.path}"],
                )
            )

        for template in relation_templates:
            if template.relation_index in assigned_relation_indexes:
                continue
            subpatterns.append(
                self.codec._make_subpattern_candidate(
                    kind="relation",
                    name=f"relation_{template.relation_type}_{template.relation_index:02d}",
                    stable_fields={},
                    variable_slots=[],
                    relation_templates=[template],
                    notes=[f"relation_type={template.relation_type}"],
                )
            )

        candidate = HierarchicalPatternCandidate(
            fingerprint=self.codec._fingerprint_hierarchical_candidate(subpatterns),
            subpatterns=subpatterns,
            variable_slots=self.codec._compose_subpatterns(subpatterns).variable_slots,
            notes=list(notes),
        )
        return subpatterns, candidate

    def _merge_composed_structures(
        self,
        composed_a: _ComposedHierarchicalStructure,
        composed_b: _ComposedHierarchicalStructure,
    ) -> _ComposedHierarchicalStructure:
        descriptors_a = self._field_descriptors(composed_a)
        descriptors_b = self._field_descriptors(composed_b)

        stable_fields: Dict[str, str] = {}
        merged_slots_by_key: Dict[Tuple[str, str, str], PatternSlot] = {}

        for field in SINGLE_FIELDS:
            descriptor_a = descriptors_a.get(field)
            descriptor_b = descriptors_b.get(field)
            if descriptor_a is None and descriptor_b is None:
                continue
            if (
                descriptor_a is not None
                and descriptor_b is not None
                and descriptor_a["mode"] == "stable"
                and descriptor_b["mode"] == "stable"
                and descriptor_a["value"] == descriptor_b["value"]
            ):
                stable_fields[field] = str(descriptor_a["value"])
                continue

            primitive_type = self._generalized_field_primitive_type(field, descriptor_a, descriptor_b)
            optional = descriptor_a is None or descriptor_b is None
            slot = self.codec._build_slot(
                slot_index=0,
                source_kind="field",
                path=field,
                primitive_type=primitive_type,
                description=f"{field} generalized slot",
                optional=optional,
            )
            merged_slots_by_key[self._slot_key(slot)] = slot

        slots_a = {self._slot_key(slot): slot for slot in composed_a.variable_slots if not self._is_single_field_slot(slot)}
        slots_b = {self._slot_key(slot): slot for slot in composed_b.variable_slots if not self._is_single_field_slot(slot)}
        for key in sorted(set(slots_a) | set(slots_b)):
            slot_a = slots_a.get(key)
            slot_b = slots_b.get(key)
            base = slot_a or slot_b
            if base is None:
                continue
            merged_slots_by_key[key] = PatternSlot(
                slot_name=base.slot_name,
                slot_index=base.slot_index,
                source_kind=base.source_kind,
                path=base.path,
                primitive_type=base.primitive_type,
                description=base.description,
                optional=(slot_a is None or slot_b is None or bool((slot_a and slot_a.optional) or (slot_b and slot_b.optional))),
            )

        merged_slots = self.codec._normalize_slots(list(merged_slots_by_key.values()))
        slot_key_to_merged = {self._slot_key(slot): slot for slot in merged_slots}
        slot_names_a = {slot.slot_name: self._slot_key(slot) for slot in composed_a.variable_slots}
        slot_names_b = {slot.slot_name: self._slot_key(slot) for slot in composed_b.variable_slots}
        generalized_field_slot_names = {
            slot.path: slot.slot_name
            for slot in merged_slots
            if slot.source_kind == "field" and slot.path in SINGLE_FIELDS
        }

        transformed_a = self._transform_relations(
            composed_a.relation_templates,
            slot_name_to_key=slot_names_a,
            slot_key_to_merged=slot_key_to_merged,
            generalized_field_slot_names=generalized_field_slot_names,
        )
        transformed_b = self._transform_relations(
            composed_b.relation_templates,
            slot_name_to_key=slot_names_b,
            slot_key_to_merged=slot_key_to_merged,
            generalized_field_slot_names=generalized_field_slot_names,
        )

        counter_a = Counter(self._relation_key(template) for template in transformed_a)
        counter_b = Counter(self._relation_key(template) for template in transformed_b)
        relation_templates: List[PatternRelationTemplate] = []
        relation_index = 0
        for key in sorted(set(counter_a) | set(counter_b)):
            required_count = min(counter_a.get(key, 0), counter_b.get(key, 0))
            total_count = max(counter_a.get(key, 0), counter_b.get(key, 0))
            for occurrence in range(total_count):
                relation_index += 1
                relation_templates.append(
                    self._relation_from_key(
                        key=key,
                        relation_index=relation_index,
                        optional=occurrence >= required_count,
                    )
                )

        return _ComposedHierarchicalStructure(
            stable_fields=stable_fields,
            variable_slots=merged_slots,
            relation_templates=relation_templates,
        )

    def _score_field_similarity(
        self,
        composed_a: _ComposedHierarchicalStructure,
        composed_b: _ComposedHierarchicalStructure,
    ) -> Tuple[float, List[str]]:
        descriptors_a = self._field_descriptors(composed_a)
        descriptors_b = self._field_descriptors(composed_b)
        numerator = 0.0
        denominator = 0.0
        generalized_fields: List[str] = []

        for field in SINGLE_FIELDS:
            descriptor_a = descriptors_a.get(field)
            descriptor_b = descriptors_b.get(field)
            if descriptor_a is None and descriptor_b is None:
                continue
            weight = FIELD_WEIGHTS.get(field, 1.0)
            denominator += weight

            if descriptor_a is not None and descriptor_b is not None:
                if descriptor_a == descriptor_b:
                    numerator += weight
                    continue
                if descriptor_a["mode"] == descriptor_b["mode"] == "stable":
                    numerator += weight * 0.7
                    generalized_fields.append(field)
                    continue
                if descriptor_a["path"] == descriptor_b["path"]:
                    numerator += weight * 0.8
                    generalized_fields.append(field)
                    continue
                numerator += weight * 0.35
                generalized_fields.append(field)
                continue

            numerator += weight * 0.15

        return round(numerator / denominator, 4) if denominator else 0.0, sorted(set(generalized_fields))

    def _score_relation_similarity(
        self,
        composed_a: _ComposedHierarchicalStructure,
        composed_b: _ComposedHierarchicalStructure,
    ) -> Tuple[float, int]:
        slot_names_a = {slot.slot_name: self._slot_key(slot) for slot in composed_a.variable_slots}
        slot_names_b = {slot.slot_name: self._slot_key(slot) for slot in composed_b.variable_slots}
        counter_a = Counter(self._relation_key(template, slot_name_to_key=slot_names_a) for template in composed_a.relation_templates)
        counter_b = Counter(self._relation_key(template, slot_name_to_key=slot_names_b) for template in composed_b.relation_templates)
        shared = sum(min(counter_a[key], counter_b[key]) for key in set(counter_a) | set(counter_b))
        total = sum(max(counter_a[key], counter_b[key]) for key in set(counter_a) | set(counter_b))
        return (round(shared / total, 4) if total else 1.0, shared)

    def _score_subpattern_similarity(
        self,
        pattern_a: HierarchicalPatternDefinition,
        pattern_b: HierarchicalPatternDefinition,
    ) -> Tuple[float, int]:
        set_a = set(pattern_a.subpattern_ids)
        set_b = set(pattern_b.subpattern_ids)
        shared = len(set_a & set_b)
        total = len(set_a | set_b)
        return (round(shared / total, 4) if total else 1.0, shared)

    def _score_slot_similarity(
        self,
        composed_a: _ComposedHierarchicalStructure,
        composed_b: _ComposedHierarchicalStructure,
    ) -> float:
        slots_a = {self._slot_key(slot) for slot in composed_a.variable_slots if not self._is_single_field_slot(slot)}
        slots_b = {self._slot_key(slot) for slot in composed_b.variable_slots if not self._is_single_field_slot(slot)}
        total_weight = 0.0
        matched_weight = 0.0

        for key in sorted(slots_a | slots_b):
            primitive_type = key[2]
            weight = LIST_SLOT_WEIGHTS.get(primitive_type, 0.75)
            total_weight += weight
            if key in slots_a and key in slots_b:
                matched_weight += weight

        return round(matched_weight / total_weight, 4) if total_weight else 1.0

    def _field_descriptors(
        self,
        composed: _ComposedHierarchicalStructure,
    ) -> Dict[str, Dict[str, object]]:
        descriptors: Dict[str, Dict[str, object]] = {}
        for field, value in composed.stable_fields.items():
            descriptors[field] = {
                "mode": "stable",
                "value": value,
                "path": field,
            }
        for slot in composed.variable_slots:
            if slot.source_kind != "field" or slot.path not in SINGLE_FIELDS:
                continue
            descriptors[slot.path] = {
                "mode": "slot",
                "value": slot.primitive_type,
                "path": slot.path,
                "primitive_type": slot.primitive_type,
                "optional": slot.optional,
            }
        return descriptors

    def _generalized_field_primitive_type(
        self,
        field: str,
        descriptor_a: Optional[Dict[str, object]],
        descriptor_b: Optional[Dict[str, object]],
    ) -> str:
        for descriptor in (descriptor_a, descriptor_b):
            if descriptor is None:
                continue
            primitive_type = str(descriptor.get("primitive_type", "") or "")
            if primitive_type:
                return primitive_type
        return GENERALIZABLE_FIELD_TYPES.get(field, field)

    def _is_single_field_slot(self, slot: PatternSlot) -> bool:
        return slot.source_kind == "field" and slot.path in SINGLE_FIELDS

    def _slot_key(self, slot: PatternSlot) -> Tuple[str, str, str]:
        return (slot.source_kind, slot.path, slot.primitive_type)

    def _relation_key(
        self,
        template: PatternRelationTemplate,
        slot_name_to_key: Optional[Dict[str, Tuple[str, str, str]]] = None,
    ) -> Tuple[object, ...]:
        return (
            template.relation_type,
            self._relation_endpoint_key(template.source_mode, template.source_value, slot_name_to_key),
            self._relation_endpoint_key(template.target_mode, template.target_value, slot_name_to_key),
        )

    def _relation_endpoint_key(
        self,
        mode: str,
        value: str,
        slot_name_to_key: Optional[Dict[str, Tuple[str, str, str]]] = None,
    ) -> Tuple[object, ...]:
        if mode != "slot" or slot_name_to_key is None:
            return (mode, value)
        slot_key = slot_name_to_key.get(value)
        if slot_key is None:
            return (mode, value)
        return (mode,) + slot_key

    def _transform_relations(
        self,
        relation_templates: Sequence[PatternRelationTemplate],
        slot_name_to_key: Dict[str, Tuple[str, str, str]],
        slot_key_to_merged: Dict[Tuple[str, str, str], PatternSlot],
        generalized_field_slot_names: Dict[str, str],
    ) -> List[PatternRelationTemplate]:
        transformed: List[PatternRelationTemplate] = []
        for template in relation_templates:
            source_mode, source_value = self._transform_relation_endpoint(
                mode=template.source_mode,
                value=template.source_value,
                slot_name_to_key=slot_name_to_key,
                slot_key_to_merged=slot_key_to_merged,
                generalized_field_slot_names=generalized_field_slot_names,
            )
            target_mode, target_value = self._transform_relation_endpoint(
                mode=template.target_mode,
                value=template.target_value,
                slot_name_to_key=slot_name_to_key,
                slot_key_to_merged=slot_key_to_merged,
                generalized_field_slot_names=generalized_field_slot_names,
            )
            transformed.append(
                PatternRelationTemplate(
                    relation_index=template.relation_index,
                    relation_type=template.relation_type,
                    source_mode=source_mode,
                    source_value=source_value,
                    target_mode=target_mode,
                    target_value=target_value,
                    optional=template.optional,
                )
            )
        return transformed

    def _transform_relation_endpoint(
        self,
        mode: str,
        value: str,
        slot_name_to_key: Dict[str, Tuple[str, str, str]],
        slot_key_to_merged: Dict[Tuple[str, str, str], PatternSlot],
        generalized_field_slot_names: Dict[str, str],
    ) -> Tuple[str, str]:
        if mode == "field_ref" and value in generalized_field_slot_names:
            return "slot", generalized_field_slot_names[value]
        if mode != "slot":
            return mode, value
        slot_key = slot_name_to_key.get(value)
        if slot_key is None:
            return mode, value
        merged_slot = slot_key_to_merged.get(slot_key)
        if merged_slot is None:
            return mode, value
        return "slot", merged_slot.slot_name

    def _build_subpattern_decision(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
        current_sequence: int,
        full_size_bytes: int,
    ) -> Optional[AdaptiveSelectionDecision]:
        candidate = self.codec.extract_subpatterns(memory_bundle)
        candidate_subpattern_ids: List[int] = []
        matched_subpattern_ids: List[int] = []
        total_subpattern_score = 0.0

        for subpattern in candidate.subpatterns:
            existing = self.registry.find_subpattern_by_fingerprint(subpattern.fingerprint, active_only=True)
            if existing is None:
                existing = self.registry.find_subpattern_by_fingerprint(subpattern.fingerprint, active_only=False)
            if existing is None:
                continue
            candidate_subpattern_ids.append(existing.subpattern_id)
            if existing.active:
                matched_subpattern_ids.append(existing.subpattern_id)
                total_subpattern_score += self._score_subpattern(existing, current_sequence=current_sequence)

        if not candidate_subpattern_ids or not matched_subpattern_ids:
            return None

        encoded = self.codec.encode_memory_bundle_hierarchical(
            current_bundle=memory_bundle,
            registry=self.registry,
            subpattern_ids=candidate_subpattern_ids,
        )
        average_subpattern_score = round(total_subpattern_score / max(1, len(matched_subpattern_ids)), 4)
        return AdaptiveSelectionDecision(
            mode="subpatterns",
            pattern_id=None,
            pattern_name="",
            encoded_size_bytes=encoded.total_bytes,
            full_size_bytes=full_size_bytes,
            reinforcement_score=average_subpattern_score,
            effective_cost=self._effective_cost(encoded.total_bytes, average_subpattern_score),
            generalization_level=0,
            subpattern_ids=candidate_subpattern_ids,
            matched_subpattern_ids=matched_subpattern_ids,
            generalized_pattern=False,
            reasons=[
                "subpattern_composition",
                f"matched_subpatterns={len(matched_subpattern_ids)}",
                f"encoded_size_bytes={encoded.total_bytes}",
                f"average_subpattern_score={average_subpattern_score}",
                f"effective_cost={self._effective_cost(encoded.total_bytes, average_subpattern_score)}",
            ],
        )

    def _relation_from_key(
        self,
        key: Tuple[object, ...],
        relation_index: int,
        optional: bool,
    ) -> PatternRelationTemplate:
        relation_type = str(key[0])
        source_mode, source_value = self._endpoint_from_key(key[1])
        target_mode, target_value = self._endpoint_from_key(key[2])
        return PatternRelationTemplate(
            relation_index=relation_index,
            relation_type=relation_type,
            source_mode=source_mode,
            source_value=source_value,
            target_mode=target_mode,
            target_value=target_value,
            optional=optional,
        )

    def _endpoint_from_key(self, endpoint_key: Tuple[object, ...]) -> Tuple[str, str]:
        mode = str(endpoint_key[0])
        if mode != "slot":
            return mode, str(endpoint_key[1])
        if len(endpoint_key) < 4:
            return mode, str(endpoint_key[1])
        source_kind = str(endpoint_key[1])
        path = str(endpoint_key[2])
        primitive_type = str(endpoint_key[3])
        slot = self.codec._build_slot(
            slot_index=0,
            source_kind=source_kind,
            path=path,
            primitive_type=primitive_type,
            description=f"{path} generalized slot",
        )
        return "slot", slot.slot_name

    def _try_extract_slot_values(
        self,
        memory: object,
        variable_slots: Sequence[PatternSlot],
    ) -> Optional[Dict[str, str]]:
        slot_values: Dict[str, str] = {}
        for slot in sorted(variable_slots, key=lambda item: item.slot_index):
            value = self.codec._read_path(memory, slot.path)
            if value == "" and not slot.optional:
                return None
            slot_values[slot.slot_name] = value
        return slot_values

    def _score_subpattern(
        self,
        subpattern: HierarchicalSubpatternDefinition,
        current_sequence: int,
    ) -> float:
        sequence = max(current_sequence, subpattern.last_used_sequence, subpattern.created_sequence, 1)
        effective_last_used = subpattern.last_used_sequence or subpattern.created_sequence or 1
        age = max(0, sequence - effective_last_used)
        frequency_signal = min(1.0, subpattern.usage_count / max(1, sequence))
        recency_signal = 1.0 / (1.0 + age)
        reuse_frequency = subpattern.reuse_frequency if subpattern.reuse_frequency > 0.0 else subpattern.usage_count / max(1, sequence)
        cooccurrence_signal = min(1.0, subpattern.cooccurrence_count / max(1, sequence))
        decay_penalty = min(0.75, age / max(4, sequence))
        return round(
            max(
                0.0,
                min(
                    1.0,
                    (frequency_signal * 0.35)
                    + (recency_signal * 0.25)
                    + (min(1.0, reuse_frequency) * 0.2)
                    + (cooccurrence_signal * 0.15)
                    - (decay_penalty * 0.2),
                ),
            ),
            4,
        )

    def _effective_cost(self, size_bytes: int, reinforcement_score: float) -> float:
        return round(float(size_bytes) - min(2.0, reinforcement_score * 2.0), 4)

    def _record_selection_usage(
        self,
        decision: AdaptiveSelectionDecision,
        memory_bundle: Dict[str, List[List[List[int]]]],
        current_sequence: int,
    ) -> None:
        full_size_bytes = len(self.codec.full_bundle_bytes(memory_bundle))
        if decision.mode == "pattern" and decision.pattern_id is not None:
            pattern = self.registry.get_pattern(decision.pattern_id)
            self.registry.record_pattern_usage(
                pattern.pattern_id,
                increment=1,
                sequence=current_sequence,
                encoded_size_bytes=decision.encoded_size_bytes,
                full_size_bytes=full_size_bytes,
            )
            self.registry.record_subpattern_usage(
                pattern.subpattern_ids,
                increment=1,
                sequence=current_sequence,
            )
            self.registry.record_subpattern_cooccurrence(
                pattern.subpattern_ids,
                increment=1,
                sequence=current_sequence,
            )
            self._refresh_pattern_temporal_state(pattern.pattern_id, current_sequence)
            for subpattern_id in pattern.subpattern_ids:
                self._refresh_subpattern_temporal_state(subpattern_id, current_sequence)
            return

        if decision.mode == "subpatterns":
            self.registry.record_subpattern_usage(
                decision.subpattern_ids,
                increment=1,
                sequence=current_sequence,
            )
            self.registry.record_subpattern_cooccurrence(
                decision.subpattern_ids,
                increment=1,
                sequence=current_sequence,
            )
            for subpattern_id in decision.subpattern_ids:
                self._refresh_subpattern_temporal_state(subpattern_id, current_sequence)

    def _refresh_pattern_temporal_state(
        self,
        pattern_id: int,
        current_sequence: int,
    ) -> None:
        try:
            pattern = self.registry.get_pattern(pattern_id)
        except Exception:
            return
        score = self.score_pattern(pattern, current_sequence=current_sequence)
        refreshed = HierarchicalPatternDefinition(
            pattern_id=pattern.pattern_id,
            name=pattern.name,
            fingerprint=pattern.fingerprint,
            created_sequence=pattern.created_sequence,
            usage_count=pattern.usage_count,
            subpattern_ids=pattern.subpattern_ids,
            variable_slots=pattern.variable_slots,
            notes=pattern.notes,
            active=pattern.active,
            generalization_level=pattern.generalization_level,
            generalized_slot_count=pattern.generalized_slot_count,
            merged_from_pattern_ids=list(pattern.merged_from_pattern_ids),
            superseded_by_pattern_id=pattern.superseded_by_pattern_id,
            last_used_sequence=pattern.last_used_sequence,
            reuse_frequency=pattern.reuse_frequency,
            total_encoded_size_bytes=pattern.total_encoded_size_bytes,
            total_full_size_bytes=pattern.total_full_size_bytes,
            average_compression_ratio=pattern.average_compression_ratio,
            success_count=pattern.success_count,
            failure_count=pattern.failure_count,
            success_rate=pattern.success_rate,
            compression_efficiency=score.compression_efficiency,
            temporal_score=score.temporal_score,
        )
        self.registry.update_pattern(refreshed)

    def _refresh_subpattern_temporal_state(
        self,
        subpattern_id: int,
        current_sequence: int,
    ) -> None:
        try:
            subpattern = self.registry.get_subpattern(subpattern_id)
        except Exception:
            return
        score = self._score_subpattern(subpattern, current_sequence=current_sequence)
        refreshed = HierarchicalSubpatternDefinition(
            subpattern_id=subpattern.subpattern_id,
            kind=subpattern.kind,
            name=subpattern.name,
            fingerprint=subpattern.fingerprint,
            created_sequence=subpattern.created_sequence,
            usage_count=subpattern.usage_count,
            stable_fields=subpattern.stable_fields,
            variable_slots=subpattern.variable_slots,
            relation_templates=subpattern.relation_templates,
            notes=subpattern.notes,
            active=subpattern.active,
            generalization_level=subpattern.generalization_level,
            merged_from_subpattern_ids=list(subpattern.merged_from_subpattern_ids),
            last_used_sequence=subpattern.last_used_sequence,
            reuse_frequency=subpattern.reuse_frequency,
            cooccurrence_count=subpattern.cooccurrence_count,
            temporal_score=score,
        )
        self.registry.update_subpattern(refreshed)

    def _resolve_pattern(
        self,
        pattern: HierarchicalPatternDefinition | int,
    ) -> HierarchicalPatternDefinition:
        if isinstance(pattern, HierarchicalPatternDefinition):
            return pattern
        return self.registry.get_pattern(int(pattern))


def compare_pattern_similarity(
    pattern_a: HierarchicalPatternDefinition | int,
    pattern_b: HierarchicalPatternDefinition | int,
    codec: BrainHierarchicalPatternCodec,
    registry: BrainHierarchicalPatternRegistry,
) -> PatternSimilarityResult:
    engine = BrainPatternEvolution(codec=codec, registry=registry)
    return engine.compare_pattern_similarity(pattern_a, pattern_b)


def merge_patterns(
    pattern_a: HierarchicalPatternDefinition | int,
    pattern_b: HierarchicalPatternDefinition | int,
    codec: BrainHierarchicalPatternCodec,
    registry: BrainHierarchicalPatternRegistry,
) -> PatternMergeResult:
    engine = BrainPatternEvolution(codec=codec, registry=registry)
    return engine.merge_patterns(pattern_a, pattern_b)
