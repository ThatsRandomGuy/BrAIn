from __future__ import annotations

import math
import re
import json
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "does",
    "do",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "to",
    "with",
}
_QUERY_HELPER_TERMS = {
    "how",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}
_NEGATION_TERMS = {"cant", "cannot", "isnt", "never", "no", "not", "without", "wont"}
_BOOLEAN_TERMS = {"false", "no", "true", "yes"}
_COLOR_TERMS = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}
_CAUSAL_TERMS = {
    "because",
    "cause",
    "caused",
    "causes",
    "causing",
    "due",
    "lead",
    "leads",
    "led",
    "result",
    "results",
    "resulting",
    "since",
    "therefore",
    "trigger",
    "triggered",
    "triggers",
}
_DIRECT_FACT_VERBS = {"appears", "are", "is", "looks", "means", "refers"}


def _normalize_text(value: object) -> str:
    lowered = str(value or "").lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _tokenize(value: object, *, drop_query_helpers: bool = False) -> List[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return []
    tokens = [token for token in normalized.split() if token and token not in _STOPWORDS]
    if not drop_query_helpers:
        return tokens
    return [token for token in tokens if token not in _QUERY_HELPER_TERMS]


def _jaccard_score(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = {item for item in left if item}
    right_set = {item for item in right if item}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / float(len(left_set | right_set))


def _contains_causal_language(text: str) -> bool:
    tokens = set(_tokenize(text))
    return bool(tokens & _CAUSAL_TERMS)


def _important_terms(text: str) -> set[str]:
    tokens = set(_tokenize(text))
    return {
        token
        for token in tokens
        if token in _COLOR_TERMS or token in _BOOLEAN_TERMS or token in _NEGATION_TERMS or token.isdigit()
    }


def _clamp(value: float, *, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class RelationshipType(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    EXTENDS = "EXTENDS"
    CAUSES = "CAUSES"


class QueryIntent(str, Enum):
    FACT = "FACT"
    EXPLANATION = "EXPLANATION"
    PROCESS = "PROCESS"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ReasoningGraphScoringConfig:
    support_weight_fact: float = 1.0
    support_weight_explanation: float = 1.05
    support_weight_process: float = 0.95
    support_weight_unknown: float = 0.9
    causal_weight_fact: float = 0.25
    causal_weight_explanation: float = 1.1
    causal_weight_process: float = 0.65
    causal_weight_unknown: float = 0.4
    extension_weight: float = 0.3
    contradiction_weight: float = 1.5
    contradiction_exponent: float = 1.2
    directness_weight_fact: float = 1.2
    directness_weight_explanation: float = 0.45
    directness_weight_process: float = 0.55
    directness_weight_unknown: float = 0.35
    directness_exponent: float = 1.2
    directness_bonus_cap: float = 1.35
    coherence_weight_fact: float = 0.9
    coherence_weight_explanation: float = 0.7
    coherence_weight_process: float = 0.65
    coherence_weight_unknown: float = 0.5
    fact_extension_multiplier: float = 0.1
    fact_coherence_multiplier: float = 0.3
    fact_concise_bonus_weight: float = 0.45
    explanation_richness_bonus_weight: float = 0.35
    process_richness_bonus_weight: float = 0.3
    support_neighbor_bonus: float = 0.32
    support_strength_bonus: float = 0.2
    mutual_support_bonus: float = 0.35
    dense_cluster_bonus_weight: float = 0.12
    dense_cluster_min_size: int = 2
    cluster_score_density_weight: float = 0.45
    cluster_score_support_edge_weight: float = 0.18
    cluster_influence_weight: float = 0.22
    cluster_support_weight: float = 0.4
    cluster_causal_support_weight: float = 0.55
    cluster_extend_support_weight: float = 0.18
    cluster_contradiction_weight: float = 0.85
    cluster_contradiction_exponent: float = 1.1
    cluster_activation_query_match_weight: float = 0.55
    cluster_activation_score_weight: float = 0.35
    cluster_activation_density_weight: float = 0.25
    node_activation_weight: float = 1.0
    path_node_weight: float = 0.12
    path_edge_weight_causes: float = 0.95
    path_edge_weight_supports: float = 0.55
    path_edge_weight_extends: float = 0.18
    path_reversed_edge_penalty: float = 0.45
    path_activation_weight: float = 0.28
    path_competition_weight: float = 0.85
    activation_decay_factor: float = 0.65
    inhibition_factor: float = 0.38
    spatial_grid_width: int = 32
    spatial_grid_height: int = 32
    spatial_node_region_radius: int = 1
    spatial_cluster_region_radius: int = 2
    spatial_cluster_activation_weight: float = 0.32
    spatial_decay_factor: float = 0.28
    spatial_propagation_steps: int = 2
    spatial_inhibition_factor: float = 0.24
    spatial_peak_threshold: float = 0.18
    spatial_peak_limit: int = 4
    spatial_force_iterations: int = 8
    spatial_attraction_rate: float = 0.18
    spatial_repulsion_rate: float = 0.28
    spatial_cluster_anchor_rate: float = 0.16
    spatial_support_target_distance: float = 1.4
    spatial_cause_target_distance: float = 1.8
    spatial_extend_target_distance: float = 2.3
    spatial_weak_target_distance: float = 3.2
    spatial_contradiction_target_distance: float = 6.0
    spatial_base_pair_distance: float = 3.2
    spatial_support_distance_weight: float = 1.5
    spatial_contradiction_distance_weight: float = 2.4
    spatial_causal_distance_weight: float = 0.7
    spatial_extend_distance_weight: float = 0.45
    spatial_cluster_size_radius_weight: float = 0.6
    spatial_region_competition_weight: float = 0.32
    spatial_region_competition_radius: float = 16.0
    spatial_region_dominance_threshold: float = 0.72
    spatial_region_dominance_weight: float = 0.6
    spatial_region_merge_distance: float = 5.0
    spatial_region_merge_support_threshold: float = 0.35
    spatial_region_expansion_radius: int = 2
    spatial_temporal_steps: int = 3
    spatial_temporal_reinforcement_weight: float = 0.22
    spatial_temporal_suppression_weight: float = 0.18
    spatial_directional_expansion_weight: float = 0.65
    concept_memory_influence_weight: float = 0.18
    concept_memory_similarity_threshold: float = 0.22
    concept_memory_merge_distance: float = 4.0
    concept_memory_merge_overlap_threshold: float = 0.25
    concept_memory_limit: int = 12
    concept_memory_activation_decay: float = 0.82
    concept_memory_reinforcement_weight: float = 0.28
    concept_competition_weight: float = 0.24
    concept_competition_radius: float = 10.0
    concept_selection_weight: float = 0.35
    concept_importance_reuse_weight: float = 0.08
    concept_importance_dominance_weight: float = 0.55
    concept_importance_stability_weight: float = 0.37
    concept_importance_influence_weight: float = 0.22
    concept_pre_activation_limit: int = 4
    concept_pre_activation_node_weight: float = 0.16
    concept_meta_min_coactivation_count: int = 1
    concept_meta_overlap_threshold: float = 0.3
    concept_meta_activation_weight: float = 0.55
    concept_cluster_similarity_threshold: float = 0.28
    concept_meta_child_influence_weight: float = 0.34
    concept_cluster_competition_weight: float = 0.2
    concept_cluster_competition_radius: float = 12.0
    concept_importance_success_reinforcement: float = 0.06
    concept_importance_decay_factor: float = 0.015
    concept_anticipation_limit: int = 3
    concept_prediction_success_threshold: float = 0.32
    concept_exploration_rate: float = 0.12
    concept_exploration_boost: float = 0.08
    concept_exploration_importance_threshold: float = 0.62
    concept_uncertainty_exploration_weight: float = 0.06
    concept_low_confidence_exploration_weight: float = 0.05
    concept_path_exploration_rate: float = 0.08
    concept_exploration_seed: int = 1701
    temporal_path_max_concepts: int = 5
    temporal_plan_candidate_limit: int = 8
    temporal_transition_success_reinforcement: float = 0.12
    temporal_transition_failure_decay: float = 0.08
    goal_priority_weight: float = 0.4
    goal_plan_importance_weight: float = 0.25
    goal_plan_confidence_weight: float = 0.35
    goal_plan_reliability_weight: float = 0.4
    goal_plan_semantic_edge_weight: float = 0.22
    goal_plan_temporal_likelihood_weight: float = 0.32
    goal_plan_past_success_weight: float = 0.28
    goal_plan_simulation_success_weight: float = 0.45
    goal_plan_simulation_stability_weight: float = 0.25
    goal_plan_simulation_uncertainty_weight: float = 0.35
    goal_simulation_accuracy_reinforcement: float = 0.1
    goal_simulation_accuracy_decay: float = 0.08
    goal_simulation_weight_adjustment_rate: float = 0.08
    goal_simulation_weight_adjustment_cap: float = 0.18
    goal_simulation_accuracy_exploration_threshold: float = 0.68
    goal_simulation_exploration_boost_cap: float = 0.08
    concept_cross_domain_overlap_threshold: float = 0.16
    concept_cross_domain_link_weight: float = 0.12
    concept_generalized_cross_domain_floor: float = 0.08
    concept_generalized_child_influence_weight: float = 0.18
    concept_generalization_overlap_threshold: float = 0.28
    concept_generalization_activation_weight: float = 0.42
    concept_importance_cap: float = 0.92
    reasoning_path_max_nodes: int = 4
    direct_pattern_bonus: float = 0.3
    fact_color_bonus: float = 0.35
    fact_causal_penalty: float = 0.22


def detect_query_intent(query_text: str) -> QueryIntent:
    normalized = _normalize_text(query_text)
    if not normalized:
        return QueryIntent.UNKNOWN
    if normalized.startswith("why ") or " why " in f" {normalized} ":
        return QueryIntent.EXPLANATION
    if normalized.startswith("how ") or " how " in f" {normalized} ":
        return QueryIntent.PROCESS
    if any(
        phrase in normalized
        for phrase in (
            "what color",
            "what is",
            "who is",
            "where is",
            "when is",
            "which is",
        )
    ):
        return QueryIntent.FACT
    if normalized.startswith(("what ", "who ", "where ", "when ", "which ")):
        return QueryIntent.FACT
    return QueryIntent.UNKNOWN


@dataclass(frozen=True)
class ReasoningGraphCandidate:
    entry_id: int
    label: str
    base_score: float
    topic_key: str = ""
    answer_text: str = ""
    answer_signature: str = ""
    terms: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.facts) if self.facts else self.answer_text

    @property
    def normalized_terms(self) -> List[str]:
        merged = list(self.terms)
        merged.extend(_tokenize(self.topic_key))
        merged.extend(_tokenize(self.answer_text))
        for fact in self.facts:
            merged.extend(_tokenize(fact))
        deduped: List[str] = []
        seen = set()
        for token in merged:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
        return deduped

    @property
    def detail_score(self) -> float:
        return float(len(_tokenize(self.text)))

    @property
    def concise_score(self) -> float:
        detail = self.detail_score
        if detail <= 8.0:
            return 1.0
        return _clamp(1.0 - ((detail - 8.0) * 0.05), low=0.2, high=1.0)

    @property
    def richness_score(self) -> float:
        return _clamp((self.detail_score - 5.0) * 0.12, low=0.0, high=1.0)

    @property
    def causal(self) -> bool:
        return _contains_causal_language(self.text)

    @property
    def important_terms(self) -> set[str]:
        return _important_terms(self.text)


@dataclass(frozen=True)
class ReasoningGraphEdge:
    source_entry_id: int
    target_entry_id: int
    relationship_type: RelationshipType
    strength: float
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_entry_id": self.source_entry_id,
            "target_entry_id": self.target_entry_id,
            "relationship_type": self.relationship_type.value,
            "strength": round(self.strength, 4),
            "evidence": list(self.evidence),
        }


@dataclass
class ReasoningGraphNode:
    entry_id: int
    label: str
    base_score: float
    raw_support_score: float = 0.0
    raw_causal_score: float = 0.0
    raw_contradiction_score: float = 0.0
    raw_extension_score: float = 0.0
    support_bonus: float = 0.0
    contradiction_penalty: float = 0.0
    extension_bonus: float = 0.0
    directness_score: float = 0.0
    directness_bonus: float = 0.0
    cluster_strength: float = 0.0
    coherence_bonus: float = 0.0
    dense_cluster_bonus: float = 0.0
    cluster_influence_bonus: float = 0.0
    intent_bonus: float = 0.0
    concept_predictive_bias: float = 0.0
    cluster_id: int = -1
    cluster_score: float = 0.0
    cluster_rank: int = 0
    cluster_size: int = 1
    cluster_density: float = 0.0
    base_activation_value: float = 0.0
    propagated_activation_bonus: float = 0.0
    inhibition_penalty: float = 0.0
    activation_value: float = 0.0
    relationships: List[Dict[str, object]] = field(default_factory=list)

    @property
    def final_score(self) -> float:
        return (
            self.base_score
            + self.support_bonus
            - self.contradiction_penalty
            + self.extension_bonus
            + self.directness_bonus
            + self.coherence_bonus
            + self.dense_cluster_bonus
            + self.cluster_influence_bonus
            + self.intent_bonus
            + self.concept_predictive_bias
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "label": self.label,
            "base_score": round(self.base_score, 4),
            "raw_support_score": round(self.raw_support_score, 4),
            "raw_causal_score": round(self.raw_causal_score, 4),
            "raw_contradiction_score": round(self.raw_contradiction_score, 4),
            "raw_extension_score": round(self.raw_extension_score, 4),
            "support_bonus": round(self.support_bonus, 4),
            "contradiction_penalty": round(self.contradiction_penalty, 4),
            "extension_bonus": round(self.extension_bonus, 4),
            "directness_score": round(self.directness_score, 4),
            "directness_bonus": round(self.directness_bonus, 4),
            "cluster_strength": round(self.cluster_strength, 4),
            "coherence_bonus": round(self.coherence_bonus, 4),
            "dense_cluster_bonus": round(self.dense_cluster_bonus, 4),
            "cluster_influence_bonus": round(self.cluster_influence_bonus, 4),
            "intent_bonus": round(self.intent_bonus, 4),
            "concept_predictive_bias": round(self.concept_predictive_bias, 4),
            "cluster_id": self.cluster_id,
            "cluster_score": round(self.cluster_score, 4),
            "cluster_rank": self.cluster_rank,
            "cluster_size": self.cluster_size,
            "cluster_density": round(self.cluster_density, 4),
            "base_activation_value": round(self.base_activation_value, 4),
            "propagated_activation_bonus": round(self.propagated_activation_bonus, 4),
            "inhibition_penalty": round(self.inhibition_penalty, 4),
            "activation_value": round(self.activation_value, 4),
            "final_score": round(self.final_score, 4),
            "relationships": [dict(item) for item in self.relationships],
        }


@dataclass(frozen=True)
class ReasoningGraphCluster:
    cluster_id: int
    nodes: List[int]
    cluster_score: float
    support_edges: int
    contradiction_edges: int
    density: float
    node_score_sum: float = 0.0
    structural_bonus: float = 0.0
    cluster_support_bonus: float = 0.0
    cluster_contradiction_penalty: float = 0.0
    cluster_final_score: float = 0.0
    base_activation_value: float = 0.0
    propagated_activation_bonus: float = 0.0
    inhibition_penalty: float = 0.0
    activation_value: float = 0.0

    @property
    def entry_ids(self) -> List[int]:
        return list(self.nodes)

    def to_dict(self) -> Dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "nodes": list(self.nodes),
            "entry_ids": list(self.nodes),
            "cluster_score": round(self.cluster_score, 4),
            "support_edges": self.support_edges,
            "support_edge_count": self.support_edges,
            "contradiction_edges": self.contradiction_edges,
            "density": round(self.density, 4),
            "node_score_sum": round(self.node_score_sum, 4),
            "structural_bonus": round(self.structural_bonus, 4),
            "cluster_support_bonus": round(self.cluster_support_bonus, 4),
            "cluster_contradiction_penalty": round(self.cluster_contradiction_penalty, 4),
            "cluster_final_score": round(self.cluster_final_score, 4),
            "base_activation_value": round(self.base_activation_value, 4),
            "propagated_activation_bonus": round(self.propagated_activation_bonus, 4),
            "inhibition_penalty": round(self.inhibition_penalty, 4),
            "activation_value": round(self.activation_value, 4),
        }


@dataclass(frozen=True)
class ReasoningGraphClusterRelationship:
    source_cluster_id: int
    target_cluster_id: int
    relationship_type: RelationshipType
    strength: float
    node_edge_count: int
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_cluster_id": self.source_cluster_id,
            "target_cluster_id": self.target_cluster_id,
            "relationship_type": self.relationship_type.value,
            "strength": round(self.strength, 4),
            "node_edge_count": self.node_edge_count,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ReasoningPixelCell:
    x: int
    y: int
    activation_value: float
    source_nodes: List[int] = field(default_factory=list)
    source_clusters: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "x": self.x,
            "y": self.y,
            "activation_value": round(self.activation_value, 4),
            "source_nodes": list(self.source_nodes),
            "source_clusters": list(self.source_clusters),
        }


@dataclass(frozen=True)
class ReasoningSpatialNodeMapping:
    entry_id: int
    cluster_id: int
    center_x: int
    center_y: int
    cells: List[Tuple[int, int]] = field(default_factory=list)
    initial_region_activation: float = 0.0
    propagated_region_activation: float = 0.0
    final_region_activation: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "cluster_id": self.cluster_id,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "cells": [{"x": x, "y": y} for x, y in self.cells],
            "initial_region_activation": round(self.initial_region_activation, 4),
            "propagated_region_activation": round(self.propagated_region_activation, 4),
            "final_region_activation": round(self.final_region_activation, 4),
        }


@dataclass(frozen=True)
class ReasoningSpatialClusterMapping:
    cluster_id: int
    center_x: int
    center_y: int
    radius: int
    entry_ids: List[int] = field(default_factory=list)
    activation_value: float = 0.0
    final_region_activation: float = 0.0
    boundary_cells: List[Tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "radius": self.radius,
            "entry_ids": list(self.entry_ids),
            "activation_value": round(self.activation_value, 4),
            "final_region_activation": round(self.final_region_activation, 4),
            "boundary_cells": [{"x": x, "y": y} for x, y in self.boundary_cells],
        }


@dataclass(frozen=True)
class ReasoningSpatialRegion:
    region_id: int
    cells: List[Tuple[int, int]] = field(default_factory=list)
    total_activation: float = 0.0
    peak_activation: float = 0.0
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    source_nodes: List[int] = field(default_factory=list)
    source_clusters: List[int] = field(default_factory=list)
    competition_penalty: float = 0.0
    post_competition_rank: int = 0
    dominance_value: float = 0.0
    merged_region_ids: List[int] = field(default_factory=list)
    dominant_node_id: int | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "region_id": self.region_id,
            "cells": [{"x": x, "y": y} for x, y in self.cells],
            "total_activation": round(self.total_activation, 4),
            "peak_activation": round(self.peak_activation, 4),
            "centroid_x": round(self.centroid_x, 2),
            "centroid_y": round(self.centroid_y, 2),
            "source_nodes": list(self.source_nodes),
            "source_clusters": list(self.source_clusters),
            "competition_penalty": round(self.competition_penalty, 4),
            "post_competition_rank": self.post_competition_rank,
            "dominance_value": round(self.dominance_value, 4),
            "merged_region_ids": list(self.merged_region_ids),
            "dominant_node_id": self.dominant_node_id,
        }


@dataclass
class ReasoningSpatialConcept:
    concept_id: int
    centroid_x: float
    centroid_y: float
    activation_value: float
    associated_nodes: List[int] = field(default_factory=list)
    associated_clusters: List[int] = field(default_factory=list)
    relationship_types: List[str] = field(default_factory=list)
    associated_terms: List[str] = field(default_factory=list)
    activation_pattern: List[Tuple[int, int, float]] = field(default_factory=list)
    reuse_count: int = 0
    last_query_intent: str = ""
    average_dominance: float = 0.0
    stability_score: float = 0.0
    importance: float = 0.0
    importance_memory_adjustment: float = 0.0
    is_meta_concept: bool = False
    child_concept_ids: List[int] = field(default_factory=list)
    parent_concept_ids: List[int] = field(default_factory=list)
    concept_cluster_id: int = -1
    coactivation_counts: Dict[int, int] = field(default_factory=dict)
    prediction_total_count: int = 0
    prediction_correct_count: int = 0
    prediction_accuracy: float = 0.0
    prediction_confidence: float = 0.0
    uncertainty: float = 1.0
    cross_domain_links: Dict[int, float] = field(default_factory=dict)
    is_generalized_concept: bool = False
    abstraction_level: int = 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "concept_id": self.concept_id,
            "centroid_x": round(self.centroid_x, 2),
            "centroid_y": round(self.centroid_y, 2),
            "activation_value": round(self.activation_value, 4),
            "associated_nodes": list(self.associated_nodes),
            "associated_clusters": list(self.associated_clusters),
            "relationship_types": list(self.relationship_types),
            "associated_terms": list(self.associated_terms),
            "activation_pattern": [
                {"x": x, "y": y, "activation_value": round(value, 4)}
                for x, y, value in self.activation_pattern
            ],
            "reuse_count": self.reuse_count,
            "last_query_intent": self.last_query_intent,
            "average_dominance": round(self.average_dominance, 4),
            "stability_score": round(self.stability_score, 4),
            "importance": round(self.importance, 4),
            "importance_memory_adjustment": round(self.importance_memory_adjustment, 4),
            "is_meta_concept": self.is_meta_concept,
            "child_concept_ids": list(self.child_concept_ids),
            "parent_concept_ids": list(self.parent_concept_ids),
            "concept_cluster_id": self.concept_cluster_id,
            "coactivation_counts": {
                str(concept_id): int(count)
                for concept_id, count in sorted(self.coactivation_counts.items())
            },
            "prediction_total_count": self.prediction_total_count,
            "prediction_correct_count": self.prediction_correct_count,
            "prediction_accuracy": round(self.prediction_accuracy, 4),
            "prediction_confidence": round(self.prediction_confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "cross_domain_links": {
                str(concept_id): round(float(strength), 4)
                for concept_id, strength in sorted(self.cross_domain_links.items())
            },
            "is_generalized_concept": self.is_generalized_concept,
            "abstraction_level": int(self.abstraction_level),
        }


@dataclass(frozen=True)
class ReasoningSpatialConceptActivation:
    concept_id: int
    similarity: float
    influence: float
    importance: float
    pre_activation_influence: float
    pre_reuse_count: int
    post_reuse_count: int
    centroid_x: float
    centroid_y: float
    associated_nodes: List[int] = field(default_factory=list)
    associated_clusters: List[int] = field(default_factory=list)
    is_meta_concept: bool = False
    is_generalized_concept: bool = False
    abstraction_level: int = 1
    concept_cluster_id: int = -1
    predicted: bool = False
    cluster_adjustment: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "concept_id": self.concept_id,
            "similarity": round(self.similarity, 4),
            "influence": round(self.influence, 4),
            "importance": round(self.importance, 4),
            "pre_activation_influence": round(self.pre_activation_influence, 4),
            "pre_reuse_count": self.pre_reuse_count,
            "post_reuse_count": self.post_reuse_count,
            "centroid_x": round(self.centroid_x, 2),
            "centroid_y": round(self.centroid_y, 2),
            "associated_nodes": list(self.associated_nodes),
            "associated_clusters": list(self.associated_clusters),
            "is_meta_concept": self.is_meta_concept,
            "is_generalized_concept": self.is_generalized_concept,
            "abstraction_level": int(self.abstraction_level),
            "concept_cluster_id": self.concept_cluster_id,
            "predicted": self.predicted,
            "cluster_adjustment": round(self.cluster_adjustment, 4),
        }


@dataclass(frozen=True)
class ReasoningConceptPrediction:
    concept_id: int
    predicted_score: float
    similarity: float
    importance: float
    prediction_confidence: float
    is_meta_concept: bool = False
    is_generalized_concept: bool = False
    abstraction_level: int = 1
    concept_cluster_id: int = -1
    child_concept_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "concept_id": self.concept_id,
            "predicted_score": round(self.predicted_score, 4),
            "similarity": round(self.similarity, 4),
            "importance": round(self.importance, 4),
            "prediction_confidence": round(self.prediction_confidence, 4),
            "is_meta_concept": self.is_meta_concept,
            "is_generalized_concept": self.is_generalized_concept,
            "abstraction_level": int(self.abstraction_level),
            "concept_cluster_id": self.concept_cluster_id,
            "child_concept_ids": list(self.child_concept_ids),
        }


@dataclass(frozen=True)
class ReasoningSpatialConceptCluster:
    concept_cluster_id: int
    concept_ids: List[int]
    cluster_score: float
    centroid_x: float
    centroid_y: float
    importance: float
    shared_nodes: List[int] = field(default_factory=list)
    shared_relationships: List[str] = field(default_factory=list)
    competition_penalty: float = 0.0
    final_score: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "concept_cluster_id": self.concept_cluster_id,
            "concept_ids": list(self.concept_ids),
            "cluster_score": round(self.cluster_score, 4),
            "centroid_x": round(self.centroid_x, 2),
            "centroid_y": round(self.centroid_y, 2),
            "importance": round(self.importance, 4),
            "shared_nodes": list(self.shared_nodes),
            "shared_relationships": list(self.shared_relationships),
            "competition_penalty": round(self.competition_penalty, 4),
            "final_score": round(self.final_score, 4),
        }


@dataclass(frozen=True)
class ReasoningSpatialGrid:
    width: int
    height: int
    decay_factor: float
    propagation_steps: int
    active_cells: List[ReasoningPixelCell] = field(default_factory=list)
    node_mappings: List[ReasoningSpatialNodeMapping] = field(default_factory=list)
    cluster_mappings: List[ReasoningSpatialClusterMapping] = field(default_factory=list)
    pre_competition_regions: List[ReasoningSpatialRegion] = field(default_factory=list)
    top_regions: List[ReasoningSpatialRegion] = field(default_factory=list)
    heatmap_before_lines: List[str] = field(default_factory=list)
    heatmap_after_lines: List[str] = field(default_factory=list)
    propagation_log: List[str] = field(default_factory=list)
    inhibition_log: List[str] = field(default_factory=list)
    region_competition_log: List[str] = field(default_factory=list)
    region_dominance_log: List[str] = field(default_factory=list)
    region_merge_log: List[str] = field(default_factory=list)
    region_expansion_log: List[str] = field(default_factory=list)
    temporal_update_log: List[str] = field(default_factory=list)
    dominance_evolution_log: List[str] = field(default_factory=list)
    concept_creation_log: List[str] = field(default_factory=list)
    concept_merge_log: List[str] = field(default_factory=list)
    concept_reuse_log: List[str] = field(default_factory=list)
    concept_activation_log: List[str] = field(default_factory=list)
    concept_evolution_log: List[str] = field(default_factory=list)
    concept_competition_log: List[str] = field(default_factory=list)
    concept_pre_activation_log: List[str] = field(default_factory=list)
    concept_importance_log: List[str] = field(default_factory=list)
    concept_hierarchy_log: List[str] = field(default_factory=list)
    concept_cluster_log: List[str] = field(default_factory=list)
    meta_concept_influence_log: List[str] = field(default_factory=list)
    concept_cluster_competition_log: List[str] = field(default_factory=list)
    anticipation_log: List[str] = field(default_factory=list)
    prediction_accuracy_log: List[str] = field(default_factory=list)
    concept_importance_evolution_log: List[str] = field(default_factory=list)
    concept_exploration_log: List[str] = field(default_factory=list)
    concept_uncertainty_log: List[str] = field(default_factory=list)
    concept_cross_domain_link_log: List[str] = field(default_factory=list)
    concept_generalization_log: List[str] = field(default_factory=list)
    concept_stability_log: List[str] = field(default_factory=list)
    concept_abstraction_log: List[str] = field(default_factory=list)
    temporal_sequence_log: List[str] = field(default_factory=list)
    temporal_prediction_log: List[str] = field(default_factory=list)
    goal_selection_log: List[str] = field(default_factory=list)
    goal_plan_log: List[str] = field(default_factory=list)
    goal_feedback_log: List[str] = field(default_factory=list)
    activated_concepts: List[ReasoningSpatialConceptActivation] = field(default_factory=list)
    anticipated_concepts: List[ReasoningConceptPrediction] = field(default_factory=list)
    concept_clusters: List[ReasoningSpatialConceptCluster] = field(default_factory=list)
    concept_memory_snapshot: List[ReasoningSpatialConcept] = field(default_factory=list)
    temporal_transitions: List[ReasoningTemporalTransition] = field(default_factory=list)
    active_goal: ReasoningGoal | None = None
    goal_plans: List[ReasoningGoalPlan] = field(default_factory=list)
    placement_map_lines: List[str] = field(default_factory=list)
    cluster_boundary_lines: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "decay_factor": round(self.decay_factor, 4),
            "propagation_steps": self.propagation_steps,
            "active_cells": [cell.to_dict() for cell in self.active_cells],
            "node_mappings": [item.to_dict() for item in self.node_mappings],
            "cluster_mappings": [item.to_dict() for item in self.cluster_mappings],
            "pre_competition_regions": [item.to_dict() for item in self.pre_competition_regions],
            "top_regions": [item.to_dict() for item in self.top_regions],
            "heatmap_before_lines": list(self.heatmap_before_lines),
            "heatmap_after_lines": list(self.heatmap_after_lines),
            "propagation_log": list(self.propagation_log),
            "inhibition_log": list(self.inhibition_log),
            "region_competition_log": list(self.region_competition_log),
            "region_dominance_log": list(self.region_dominance_log),
            "region_merge_log": list(self.region_merge_log),
            "region_expansion_log": list(self.region_expansion_log),
            "temporal_update_log": list(self.temporal_update_log),
            "dominance_evolution_log": list(self.dominance_evolution_log),
            "concept_creation_log": list(self.concept_creation_log),
            "concept_merge_log": list(self.concept_merge_log),
            "concept_reuse_log": list(self.concept_reuse_log),
            "concept_activation_log": list(self.concept_activation_log),
            "concept_evolution_log": list(self.concept_evolution_log),
            "concept_competition_log": list(self.concept_competition_log),
            "concept_pre_activation_log": list(self.concept_pre_activation_log),
            "concept_importance_log": list(self.concept_importance_log),
            "concept_hierarchy_log": list(self.concept_hierarchy_log),
            "concept_cluster_log": list(self.concept_cluster_log),
            "meta_concept_influence_log": list(self.meta_concept_influence_log),
            "concept_cluster_competition_log": list(self.concept_cluster_competition_log),
            "anticipation_log": list(self.anticipation_log),
            "prediction_accuracy_log": list(self.prediction_accuracy_log),
            "concept_importance_evolution_log": list(self.concept_importance_evolution_log),
            "concept_exploration_log": list(self.concept_exploration_log),
            "concept_uncertainty_log": list(self.concept_uncertainty_log),
            "concept_cross_domain_link_log": list(self.concept_cross_domain_link_log),
            "concept_generalization_log": list(self.concept_generalization_log),
            "concept_stability_log": list(self.concept_stability_log),
            "concept_abstraction_log": list(self.concept_abstraction_log),
            "temporal_sequence_log": list(self.temporal_sequence_log),
            "temporal_prediction_log": list(self.temporal_prediction_log),
            "goal_selection_log": list(self.goal_selection_log),
            "goal_plan_log": list(self.goal_plan_log),
            "goal_feedback_log": list(self.goal_feedback_log),
            "activated_concepts": [concept.to_dict() for concept in self.activated_concepts],
            "anticipated_concepts": [prediction.to_dict() for prediction in self.anticipated_concepts],
            "concept_clusters": [cluster.to_dict() for cluster in self.concept_clusters],
            "concept_memory_snapshot": [concept.to_dict() for concept in self.concept_memory_snapshot],
            "temporal_transitions": [transition.to_dict() for transition in self.temporal_transitions],
            "active_goal": None if self.active_goal is None else self.active_goal.to_dict(),
            "goal_plans": [plan.to_dict() for plan in self.goal_plans],
            "placement_map_lines": list(self.placement_map_lines),
            "cluster_boundary_lines": list(self.cluster_boundary_lines),
        }


@dataclass(frozen=True)
class ReasoningGraph:
    query_intent: QueryIntent
    nodes: List[ReasoningGraphNode]
    edges: List[ReasoningGraphEdge]
    clusters: List[ReasoningGraphCluster] = field(default_factory=list)
    cluster_relationships: List[ReasoningGraphClusterRelationship] = field(default_factory=list)
    spatial_grid: ReasoningSpatialGrid | None = None

    def node_map(self) -> Dict[int, ReasoningGraphNode]:
        return {node.entry_id: node for node in self.nodes}

    def to_dict(self) -> Dict[str, object]:
        return {
            "query_intent": self.query_intent.value,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "cluster_relationships": [edge.to_dict() for edge in self.cluster_relationships],
            "spatial_grid": None if self.spatial_grid is None else self.spatial_grid.to_dict(),
        }


@dataclass(frozen=True)
class ReasoningGraphSelection:
    selection_mode: str
    selected_cluster_id: int | None
    selected_cluster_score: float
    selected_node_entry_id: int | None
    ranked_entry_ids: List[int]
    selected_reasoning_path: List[int] = field(default_factory=list)
    selected_path_score: float = 0.0
    selected_path_activation: float = 0.0
    selected_path_final_score: float = 0.0
    selected_path_breakdown: str = ""
    candidate_paths: List[Dict[str, object]] = field(default_factory=list)
    selected_spatial_region_id: int | None = None
    selected_spatial_nodes: List[int] = field(default_factory=list)
    selected_spatial_clusters: List[int] = field(default_factory=list)
    selected_spatial_activation: float = 0.0
    spatial_selection_reason: str = ""
    spatial_consistency_warning: str = ""
    cluster_selection_reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "selection_mode": self.selection_mode,
            "selected_cluster_id": self.selected_cluster_id,
            "selected_cluster_score": round(self.selected_cluster_score, 4),
            "selected_node_entry_id": self.selected_node_entry_id,
            "ranked_entry_ids": list(self.ranked_entry_ids),
            "selected_reasoning_path": list(self.selected_reasoning_path),
            "selected_path_score": round(self.selected_path_score, 4),
            "selected_path_activation": round(self.selected_path_activation, 4),
            "selected_path_final_score": round(self.selected_path_final_score, 4),
            "selected_path_breakdown": self.selected_path_breakdown,
            "candidate_paths": [dict(item) for item in self.candidate_paths],
            "selected_spatial_region_id": self.selected_spatial_region_id,
            "selected_spatial_nodes": list(self.selected_spatial_nodes),
            "selected_spatial_clusters": list(self.selected_spatial_clusters),
            "selected_spatial_activation": round(self.selected_spatial_activation, 4),
            "spatial_selection_reason": self.spatial_selection_reason,
            "spatial_consistency_warning": self.spatial_consistency_warning,
            "cluster_selection_reason": self.cluster_selection_reason,
        }


@dataclass(frozen=True)
class ReasoningGraphPath:
    node_ids: List[int]
    path_score: float
    relative_strength: float
    adjusted_score: float
    path_activation: float
    path_final_score: float
    breakdown: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "node_ids": list(self.node_ids),
            "path_score": round(self.path_score, 4),
            "relative_strength": round(self.relative_strength, 4),
            "adjusted_score": round(self.adjusted_score, 4),
            "path_activation": round(self.path_activation, 4),
            "path_final_score": round(self.path_final_score, 4),
            "breakdown": self.breakdown,
        }


@dataclass
class ReasoningTemporalTransition:
    source_concept_id: int
    target_concept_id: int
    frequency: int = 0
    confidence: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    reliability: float = 0.5
    simulation_total_count: int = 0
    simulation_correct_count: int = 0
    simulation_accuracy: float = 0.0
    last_prediction_error: float = 0.0
    average_prediction_error: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_concept_id": int(self.source_concept_id),
            "target_concept_id": int(self.target_concept_id),
            "frequency": int(self.frequency),
            "confidence": round(float(self.confidence), 4),
            "success_count": int(self.success_count),
            "failure_count": int(self.failure_count),
            "reliability": round(float(self.reliability), 4),
            "simulation_total_count": int(self.simulation_total_count),
            "simulation_correct_count": int(self.simulation_correct_count),
            "simulation_accuracy": round(float(self.simulation_accuracy), 4),
            "last_prediction_error": round(float(self.last_prediction_error), 4),
            "average_prediction_error": round(float(self.average_prediction_error), 4),
        }


@dataclass(frozen=True)
class ReasoningGoal:
    target_concept_id: int
    priority: float
    success_condition: str
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "target_concept_id": int(self.target_concept_id),
            "priority": round(float(self.priority), 4),
            "success_condition": self.success_condition,
            "reason": self.reason,
        }


@dataclass
class ReasoningSimulationModelState:
    total_predictions: int = 0
    correct_predictions: int = 0
    simulation_accuracy: float = 0.0
    last_prediction_error: float = 0.0
    last_prediction_direction: str = "unknown"
    average_prediction_error: float = 0.0
    calibrated_confidence: float = 0.0
    adaptive_exploration_boost: float = 0.0
    weight_adjustments: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_predictions": int(self.total_predictions),
            "correct_predictions": int(self.correct_predictions),
            "simulation_accuracy": round(float(self.simulation_accuracy), 4),
            "last_prediction_error": round(float(self.last_prediction_error), 4),
            "last_prediction_direction": self.last_prediction_direction,
            "average_prediction_error": round(float(self.average_prediction_error), 4),
            "calibrated_confidence": round(float(self.calibrated_confidence), 4),
            "adaptive_exploration_boost": round(float(self.adaptive_exploration_boost), 4),
            "weight_adjustments": {
                str(key): round(float(value), 4)
                for key, value in sorted(self.weight_adjustments.items())
            },
        }


@dataclass(frozen=True)
class ReasoningGoalPlan:
    target_concept_id: int
    path: List[int]
    plan_score: float
    prediction_confidence: float
    path_reliability: float
    cumulative_importance: float = 0.0
    semantic_edge_strength: float = 0.0
    temporal_likelihood: float = 0.0
    past_success_rate: float = 0.0
    simulated_success_probability: float = 0.0
    simulated_stability: float = 0.0
    simulated_uncertainty: float = 1.0
    predicted_outcome: str = "unknown"
    selection_reason: str = ""
    success_score: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "target_concept_id": int(self.target_concept_id),
            "path": list(self.path),
            "plan_score": round(float(self.plan_score), 4),
            "prediction_confidence": round(float(self.prediction_confidence), 4),
            "path_reliability": round(float(self.path_reliability), 4),
            "cumulative_importance": round(float(self.cumulative_importance), 4),
            "semantic_edge_strength": round(float(self.semantic_edge_strength), 4),
            "temporal_likelihood": round(float(self.temporal_likelihood), 4),
            "past_success_rate": round(float(self.past_success_rate), 4),
            "simulated_success_probability": round(float(self.simulated_success_probability), 4),
            "simulated_stability": round(float(self.simulated_stability), 4),
            "simulated_uncertainty": round(float(self.simulated_uncertainty), 4),
            "predicted_outcome": self.predicted_outcome,
            "selection_reason": self.selection_reason,
            "success_score": round(float(self.success_score), 4),
        }


class ReasoningGraphEvaluator:
    def __init__(
        self,
        *,
        min_candidates: int = 5,
        max_candidates: int = 10,
        scoring: ReasoningGraphScoringConfig | None = None,
        concept_memory_path: str | Path | None = None,
    ) -> None:
        self.min_candidates = max(2, int(min_candidates))
        self.max_candidates = max(self.min_candidates, int(max_candidates))
        self.scoring = scoring or ReasoningGraphScoringConfig()
        self.spatial_concept_memory: List[ReasoningSpatialConcept] = []
        self.temporal_transition_memory: Dict[Tuple[int, int], ReasoningTemporalTransition] = {}
        self.simulation_model_state = ReasoningSimulationModelState()
        self._next_spatial_concept_id = 1
        self.concept_memory_path = Path(concept_memory_path) if concept_memory_path is not None else None
        self._load_spatial_concept_memory()

    def _load_spatial_concept_memory(self) -> None:
        if self.concept_memory_path is None or not self.concept_memory_path.exists():
            return
        try:
            payload = json.loads(self.concept_memory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        concepts_payload = payload.get("concepts", []) if isinstance(payload, dict) else []
        concepts: List[ReasoningSpatialConcept] = []
        for item in concepts_payload:
            if not isinstance(item, dict):
                continue
            pattern: List[Tuple[int, int, float]] = []
            for cell in item.get("activation_pattern", []):
                if not isinstance(cell, dict):
                    continue
                pattern.append((
                    int(cell.get("x", 0) or 0),
                    int(cell.get("y", 0) or 0),
                    float(cell.get("activation_value", 0.0) or 0.0),
                ))
            raw_coactivation_counts = item.get("coactivation_counts", {})
            coactivation_counts: Dict[int, int] = {}
            if isinstance(raw_coactivation_counts, dict):
                for concept_id, count in raw_coactivation_counts.items():
                    try:
                        coactivation_counts[int(concept_id)] = int(count)
                    except (TypeError, ValueError):
                        continue
            raw_cross_domain_links = item.get("cross_domain_links", {})
            cross_domain_links: Dict[int, float] = {}
            if isinstance(raw_cross_domain_links, dict):
                for concept_id, strength in raw_cross_domain_links.items():
                    try:
                        cross_domain_links[int(concept_id)] = float(strength)
                    except (TypeError, ValueError):
                        continue
            is_meta_concept = bool(item.get("is_meta_concept", False))
            is_generalized_concept = bool(item.get("is_generalized_concept", False))
            concepts.append(
                ReasoningSpatialConcept(
                    concept_id=int(item.get("concept_id", len(concepts) + 1) or len(concepts) + 1),
                    centroid_x=float(item.get("centroid_x", 0.0) or 0.0),
                    centroid_y=float(item.get("centroid_y", 0.0) or 0.0),
                    activation_value=float(item.get("activation_value", 0.0) or 0.0),
                    associated_nodes=[int(value) for value in item.get("associated_nodes", [])],
                    associated_clusters=[int(value) for value in item.get("associated_clusters", [])],
                    relationship_types=[str(value) for value in item.get("relationship_types", [])],
                    associated_terms=[str(value) for value in item.get("associated_terms", [])],
                    activation_pattern=pattern,
                    reuse_count=int(item.get("reuse_count", 0) or 0),
                    last_query_intent=str(item.get("last_query_intent", "") or ""),
                    average_dominance=float(item.get("average_dominance", 0.0) or 0.0),
                    stability_score=float(item.get("stability_score", 0.0) or 0.0),
                    importance=float(item.get("importance", 0.0) or 0.0),
                    importance_memory_adjustment=float(item.get("importance_memory_adjustment", 0.0) or 0.0),
                    is_meta_concept=is_meta_concept,
                    child_concept_ids=[int(value) for value in item.get("child_concept_ids", [])],
                    parent_concept_ids=[int(value) for value in item.get("parent_concept_ids", [])],
                    concept_cluster_id=int(item.get("concept_cluster_id", -1) or -1),
                    coactivation_counts=coactivation_counts,
                    prediction_total_count=int(item.get("prediction_total_count", 0) or 0),
                    prediction_correct_count=int(item.get("prediction_correct_count", 0) or 0),
                    prediction_accuracy=float(item.get("prediction_accuracy", 0.0) or 0.0),
                    prediction_confidence=float(item.get("prediction_confidence", 0.0) or 0.0),
                    uncertainty=float(item.get("uncertainty", 1.0) or 0.0),
                    cross_domain_links=cross_domain_links,
                    is_generalized_concept=is_generalized_concept,
                    abstraction_level=int(
                        item.get(
                            "abstraction_level",
                            3 if is_generalized_concept else (2 if is_meta_concept else 1),
                        )
                    ),
                )
            )
        for concept in concepts:
            self._recalculate_concept_importance(concept)
        self.spatial_concept_memory = concepts[: max(1, int(self.scoring.concept_memory_limit))]
        self.temporal_transition_memory = {}
        transitions_payload = payload.get("temporal_transitions", []) if isinstance(payload, dict) else []
        if isinstance(transitions_payload, list):
            for item in transitions_payload:
                if not isinstance(item, dict):
                    continue
                try:
                    transition = ReasoningTemporalTransition(
                        source_concept_id=int(item.get("source_concept_id", 0) or 0),
                        target_concept_id=int(item.get("target_concept_id", 0) or 0),
                        frequency=int(item.get("frequency", 0) or 0),
                        confidence=float(item.get("confidence", 0.0) or 0.0),
                        success_count=int(item.get("success_count", 0) or 0),
                        failure_count=int(item.get("failure_count", 0) or 0),
                        reliability=float(item.get("reliability", 0.5) or 0.5),
                        simulation_total_count=int(item.get("simulation_total_count", 0) or 0),
                        simulation_correct_count=int(item.get("simulation_correct_count", 0) or 0),
                        simulation_accuracy=float(item.get("simulation_accuracy", 0.0) or 0.0),
                        last_prediction_error=float(item.get("last_prediction_error", 0.0) or 0.0),
                        average_prediction_error=float(item.get("average_prediction_error", 0.0) or 0.0),
                    )
                except (TypeError, ValueError):
                    continue
                if transition.source_concept_id <= 0 or transition.target_concept_id <= 0:
                    continue
                self.temporal_transition_memory[(transition.source_concept_id, transition.target_concept_id)] = transition
        simulation_payload = payload.get("simulation_model_state", {}) if isinstance(payload, dict) else {}
        if isinstance(simulation_payload, dict):
            raw_adjustments = simulation_payload.get("weight_adjustments", {})
            weight_adjustments: Dict[str, float] = {}
            if isinstance(raw_adjustments, dict):
                for key, value in raw_adjustments.items():
                    try:
                        weight_adjustments[str(key)] = float(value)
                    except (TypeError, ValueError):
                        continue
            self.simulation_model_state = ReasoningSimulationModelState(
                total_predictions=int(simulation_payload.get("total_predictions", 0) or 0),
                correct_predictions=int(simulation_payload.get("correct_predictions", 0) or 0),
                simulation_accuracy=float(simulation_payload.get("simulation_accuracy", 0.0) or 0.0),
                last_prediction_error=float(simulation_payload.get("last_prediction_error", 0.0) or 0.0),
                last_prediction_direction=str(simulation_payload.get("last_prediction_direction", "unknown") or "unknown"),
                average_prediction_error=float(simulation_payload.get("average_prediction_error", 0.0) or 0.0),
                calibrated_confidence=float(simulation_payload.get("calibrated_confidence", 0.0) or 0.0),
                adaptive_exploration_boost=float(simulation_payload.get("adaptive_exploration_boost", 0.0) or 0.0),
                weight_adjustments=weight_adjustments,
            )
        self._next_spatial_concept_id = 1 + max((concept.concept_id for concept in self.spatial_concept_memory), default=0)

    def _save_spatial_concept_memory(self) -> None:
        if self.concept_memory_path is None:
            return
        payload = {
            "version": 1,
            "concepts": [concept.to_dict() for concept in self.spatial_concept_memory],
            "temporal_transitions": [
                transition.to_dict()
                for transition in sorted(
                    self.temporal_transition_memory.values(),
                    key=lambda item: (item.source_concept_id, item.target_concept_id),
                )
            ],
            "simulation_model_state": self.simulation_model_state.to_dict(),
        }
        try:
            self.concept_memory_path.parent.mkdir(parents=True, exist_ok=True)
            self.concept_memory_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            return

    def _allocate_spatial_concept_id(self) -> int:
        self._next_spatial_concept_id = max(
            int(self._next_spatial_concept_id),
            1 + max((concept.concept_id for concept in self.spatial_concept_memory), default=0),
        )
        concept_id = self._next_spatial_concept_id
        self._next_spatial_concept_id += 1
        return concept_id

    def build_graph(
        self,
        *,
        candidates: Sequence[ReasoningGraphCandidate],
        query_text: str = "",
    ) -> ReasoningGraph:
        if not candidates:
            return ReasoningGraph(
                query_intent=detect_query_intent(query_text),
                nodes=[],
                edges=[],
                clusters=[],
                cluster_relationships=[],
            )
        query_intent = detect_query_intent(query_text)
        sorted_candidates = sorted(candidates, key=lambda item: (-item.base_score, item.entry_id))
        candidate_count = min(len(sorted_candidates), self.max_candidates)
        working_set = list(sorted_candidates[:candidate_count])
        nodes = {
            candidate.entry_id: ReasoningGraphNode(
                entry_id=candidate.entry_id,
                label=candidate.label,
                base_score=float(candidate.base_score),
            )
            for candidate in working_set
        }
        edges: List[ReasoningGraphEdge] = []
        for index, left in enumerate(working_set):
            for right in working_set[index + 1:]:
                edge = self._classify_pair(left, right)
                if edge is None:
                    continue
                edges.append(edge)
                self._register_edge(nodes, edge)
        (
            concept_predictive_biases,
            concept_pre_activation_log,
            anticipated_concepts,
            pre_activation_exploration_log,
        ) = self._concept_pre_activation_bias(
            query_text=query_text,
            nodes=nodes,
        )
        for candidate in working_set:
            node = nodes.get(candidate.entry_id)
            if node is None:
                continue
            self._finalize_node(
                node=node,
                candidate=candidate,
                edges=edges,
                query_text=query_text,
                query_intent=query_intent,
                concept_predictive_bias=concept_predictive_biases.get(candidate.entry_id, 0.0),
            )
        clusters = self._assign_clusters(nodes=nodes, edges=edges, query_intent=query_intent)
        self._apply_cluster_influence(nodes=nodes, clusters=clusters, query_intent=query_intent)
        clusters = self._assign_clusters(nodes=nodes, edges=edges, query_intent=query_intent)
        cluster_relationships = self._build_cluster_relationships(clusters=clusters, nodes=nodes, edges=edges)
        clusters = self._score_cluster_graph(
            clusters=clusters,
            cluster_relationships=cluster_relationships,
        )
        clusters = self._apply_cluster_activation(
            clusters=clusters,
            nodes=nodes,
            query_intent=query_intent,
            cluster_relationships=cluster_relationships,
            edges=edges,
        )
        spatial_grid = self._build_spatial_grid(
            query_text=query_text,
            query_intent=query_intent,
            nodes=nodes,
            clusters=clusters,
            edges=edges,
            cluster_relationships=cluster_relationships,
            concept_pre_activation_log=concept_pre_activation_log,
            concept_exploration_log=pre_activation_exploration_log,
            anticipated_concepts=anticipated_concepts,
        )
        ordered_nodes = sorted(nodes.values(), key=lambda item: (-item.final_score, item.entry_id))
        ordered_clusters = sorted(
            clusters,
            key=lambda item: (-item.cluster_final_score, -item.cluster_score, item.cluster_id),
        )
        ordered_cluster_relationships = sorted(
            cluster_relationships,
            key=lambda item: (
                item.source_cluster_id,
                item.target_cluster_id,
                item.relationship_type.value,
            ),
        )
        return ReasoningGraph(
            query_intent=query_intent,
            nodes=ordered_nodes,
            edges=edges,
            clusters=ordered_clusters,
            cluster_relationships=ordered_cluster_relationships,
            spatial_grid=spatial_grid,
        )

    def select_nodes(self, graph: ReasoningGraph) -> ReasoningGraphSelection:
        if not graph.nodes:
            return ReasoningGraphSelection(
                selection_mode="node_first",
                selected_cluster_id=None,
                selected_cluster_score=0.0,
                selected_node_entry_id=None,
                ranked_entry_ids=[],
                selected_reasoning_path=[],
                selected_path_score=0.0,
                selected_path_activation=0.0,
                selected_path_final_score=0.0,
                selected_path_breakdown="",
                candidate_paths=[],
                selected_spatial_region_id=None,
                selected_spatial_nodes=[],
                selected_spatial_clusters=[],
                selected_spatial_activation=0.0,
                spatial_selection_reason="",
                cluster_selection_reason="",
            )
        spatial_region, spatial_reason, spatial_warning = self._select_spatial_region(
            spatial_grid=graph.spatial_grid,
            selected_node_entry_id=None,
            selected_cluster_id=None,
        )
        node_lookup = graph.node_map()
        if graph.query_intent == QueryIntent.EXPLANATION and graph.clusters:
            best_cluster = max(
                graph.clusters,
                key=lambda item: (item.cluster_final_score, item.cluster_score, -item.cluster_id),
            )
            selected_nodes = [
                node_lookup[entry_id]
                for entry_id in best_cluster.nodes
                if entry_id in node_lookup
            ]
            selected_nodes.sort(key=lambda item: (-item.final_score, item.entry_id))
            best_path, candidate_paths = self._select_best_path(graph=graph, cluster_id=best_cluster.cluster_id)
            path_node_ids = list(best_path.node_ids) if best_path is not None else []
            path_node_lookup = {entry_id: index for index, entry_id in enumerate(path_node_ids)}
            if path_node_ids:
                selected_nodes.sort(
                    key=lambda item: (
                        path_node_lookup.get(item.entry_id, len(path_node_lookup)),
                        -item.final_score,
                        item.entry_id,
                    )
                )
            remaining_nodes = [
                node
                for node in graph.nodes
                if node.cluster_id != best_cluster.cluster_id
            ]
            remaining_nodes.sort(
                key=lambda item: (-item.cluster_score, -item.final_score, item.entry_id)
            )
            ranked_nodes = selected_nodes + remaining_nodes
            selected_node_entry_id = None
            if best_path is not None and best_path.node_ids:
                selected_node_entry_id = best_path.node_ids[0]
            elif ranked_nodes:
                selected_node_entry_id = ranked_nodes[0].entry_id
            selected_reasoning_path = list(best_path.node_ids) if best_path is not None else (
                [selected_node_entry_id] if selected_node_entry_id is not None else []
            )
            spatial_region, spatial_reason, spatial_warning = self._select_spatial_region(
                spatial_grid=graph.spatial_grid,
                selected_node_entry_id=selected_node_entry_id,
                selected_cluster_id=best_cluster.cluster_id,
            )
            spatial_node_entry_id = self._select_node_from_spatial_region(
                graph=graph,
                spatial_region=spatial_region,
                preferred_cluster_id=best_cluster.cluster_id,
                fallback_node_entry_id=selected_node_entry_id,
            )
            if spatial_node_entry_id is not None:
                selected_node_entry_id = spatial_node_entry_id
            spatial_region, spatial_reason, spatial_warning = self._select_spatial_region(
                spatial_grid=graph.spatial_grid,
                selected_node_entry_id=selected_node_entry_id,
                selected_cluster_id=best_cluster.cluster_id,
            )
            ranked_nodes = self._promote_selected_node(
                ranked_nodes=ranked_nodes,
                selected_node_entry_id=selected_node_entry_id,
            )
            return ReasoningGraphSelection(
                selection_mode="cluster_graph_first",
                selected_cluster_id=best_cluster.cluster_id,
                selected_cluster_score=best_cluster.cluster_final_score,
                selected_node_entry_id=selected_node_entry_id,
                ranked_entry_ids=[node.entry_id for node in ranked_nodes],
                selected_reasoning_path=selected_reasoning_path,
                selected_path_score=0.0 if best_path is None else best_path.path_score,
                selected_path_activation=0.0 if best_path is None else best_path.path_activation,
                selected_path_final_score=0.0 if best_path is None else best_path.path_final_score,
                selected_path_breakdown="" if best_path is None else best_path.breakdown,
                candidate_paths=[path.to_dict() for path in candidate_paths],
                selected_spatial_region_id=None if spatial_region is None else spatial_region.region_id,
                selected_spatial_nodes=[] if spatial_region is None else list(spatial_region.source_nodes),
                selected_spatial_clusters=[] if spatial_region is None else list(spatial_region.source_clusters),
                selected_spatial_activation=0.0 if spatial_region is None else spatial_region.total_activation,
                spatial_selection_reason=spatial_reason,
                spatial_consistency_warning=spatial_warning,
                cluster_selection_reason=(
                    f"Cluster {best_cluster.cluster_id} won with final score {round(best_cluster.cluster_final_score, 4)} "
                    + f"(base={round(best_cluster.cluster_score, 4)}, support={round(best_cluster.cluster_support_bonus, 4)}, "
                    + f"contradiction={round(best_cluster.cluster_contradiction_penalty, 4)})."
                ),
            )
        ranked_nodes = sorted(graph.nodes, key=lambda item: (-item.final_score, item.entry_id))
        selected_node_entry_id = ranked_nodes[0].entry_id if ranked_nodes else None
        selected_cluster_id = ranked_nodes[0].cluster_id if ranked_nodes else None
        selected_cluster_score = ranked_nodes[0].cluster_score if ranked_nodes else 0.0
        spatial_region, spatial_reason, spatial_warning = self._select_spatial_region(
            spatial_grid=graph.spatial_grid,
            selected_node_entry_id=selected_node_entry_id,
            selected_cluster_id=selected_cluster_id,
        )
        spatial_node_entry_id = self._select_node_from_spatial_region(
            graph=graph,
            spatial_region=spatial_region,
            preferred_cluster_id=selected_cluster_id,
            fallback_node_entry_id=selected_node_entry_id,
        )
        if spatial_node_entry_id is not None:
            selected_node_entry_id = spatial_node_entry_id
        spatial_region, spatial_reason, spatial_warning = self._select_spatial_region(
            spatial_grid=graph.spatial_grid,
            selected_node_entry_id=selected_node_entry_id,
            selected_cluster_id=selected_cluster_id,
        )
        ranked_nodes = self._promote_selected_node(
            ranked_nodes=ranked_nodes,
            selected_node_entry_id=selected_node_entry_id,
        )
        selected_cluster_id = ranked_nodes[0].cluster_id if ranked_nodes else selected_cluster_id
        selected_cluster_score = ranked_nodes[0].cluster_score if ranked_nodes else selected_cluster_score
        return ReasoningGraphSelection(
            selection_mode="node_first",
            selected_cluster_id=selected_cluster_id,
            selected_cluster_score=selected_cluster_score,
            selected_node_entry_id=selected_node_entry_id,
            ranked_entry_ids=[node.entry_id for node in ranked_nodes],
            selected_reasoning_path=[selected_node_entry_id] if selected_node_entry_id is not None else [],
            selected_path_score=float(ranked_nodes[0].final_score) if ranked_nodes else 0.0,
            selected_path_activation=float(ranked_nodes[0].activation_value) if ranked_nodes else 0.0,
            selected_path_final_score=(
                float(ranked_nodes[0].final_score)
                + (float(ranked_nodes[0].final_score) * self.scoring.path_competition_weight)
                + (float(ranked_nodes[0].activation_value) * self.scoring.path_activation_weight)
            ) if ranked_nodes else 0.0,
            selected_path_breakdown=(
                f"single_node_path score={round(float(ranked_nodes[0].final_score), 4)} "
                + f"+ adjusted_score={round(float(ranked_nodes[0].final_score), 4)} "
                + f"+ activation={round(float(ranked_nodes[0].activation_value), 4)}"
            ) if ranked_nodes else "",
            candidate_paths=[
                ReasoningGraphPath(
                    node_ids=[ranked_nodes[0].entry_id],
                    path_score=round(float(ranked_nodes[0].final_score), 4),
                    relative_strength=1.0,
                    adjusted_score=round(float(ranked_nodes[0].final_score), 4),
                    path_activation=round(float(ranked_nodes[0].activation_value), 4),
                    path_final_score=round(
                        float(ranked_nodes[0].final_score)
                        + (float(ranked_nodes[0].final_score) * self.scoring.path_competition_weight)
                        + (float(ranked_nodes[0].activation_value) * self.scoring.path_activation_weight),
                        4,
                    ),
                    breakdown=(
                        f"single_node_path score={round(float(ranked_nodes[0].final_score), 4)} "
                        + f"+ adjusted_score={round(float(ranked_nodes[0].final_score), 4)} "
                        + f"+ activation={round(float(ranked_nodes[0].activation_value), 4)}"
                    ),
                ).to_dict()
            ] if ranked_nodes else [],
            selected_spatial_region_id=None if spatial_region is None else spatial_region.region_id,
            selected_spatial_nodes=[] if spatial_region is None else list(spatial_region.source_nodes),
            selected_spatial_clusters=[] if spatial_region is None else list(spatial_region.source_clusters),
            selected_spatial_activation=0.0 if spatial_region is None else spatial_region.total_activation,
            spatial_selection_reason=spatial_reason,
            spatial_consistency_warning=spatial_warning,
            cluster_selection_reason="Selected the top node directly from node-level scoring.",
        )

    def _promote_selected_node(
        self,
        *,
        ranked_nodes: Sequence[ReasoningGraphNode],
        selected_node_entry_id: int | None,
    ) -> List[ReasoningGraphNode]:
        if selected_node_entry_id is None:
            return list(ranked_nodes)
        selected_node = next((node for node in ranked_nodes if node.entry_id == selected_node_entry_id), None)
        if selected_node is None:
            return list(ranked_nodes)
        remaining_nodes = [node for node in ranked_nodes if node.entry_id != selected_node_entry_id]
        return [selected_node] + remaining_nodes

    def _select_node_from_spatial_region(
        self,
        *,
        graph: ReasoningGraph,
        spatial_region: ReasoningSpatialRegion | None,
        preferred_cluster_id: int | None,
        fallback_node_entry_id: int | None,
    ) -> int | None:
        if spatial_region is None:
            return fallback_node_entry_id
        region_node_ids = list(spatial_region.source_nodes)
        if preferred_cluster_id is not None:
            region_node_ids = [
                node_id
                for node_id in region_node_ids
                if node_id in graph.node_map() and graph.node_map()[node_id].cluster_id == preferred_cluster_id
            ]
        dominant_node_id = self._dominant_node_for_region_node_ids(
            node_ids=region_node_ids or spatial_region.source_nodes,
            centroid_x=spatial_region.centroid_x,
            centroid_y=spatial_region.centroid_y,
            nodes=graph.node_map(),
            spatial_grid=graph.spatial_grid,
        )
        return dominant_node_id if dominant_node_id is not None else fallback_node_entry_id

    def _classify_pair(
        self,
        left: ReasoningGraphCandidate,
        right: ReasoningGraphCandidate,
    ) -> ReasoningGraphEdge | None:
        left_terms = set(left.normalized_terms)
        right_terms = set(right.normalized_terms)
        shared_terms = sorted(left_terms & right_terms)
        same_topic = bool(left.topic_key and right.topic_key and left.topic_key == right.topic_key)
        if not shared_terms and not same_topic:
            return None

        left_fact_tokens = _tokenize(left.text)
        right_fact_tokens = _tokenize(right.text)
        fact_overlap = _jaccard_score(left_fact_tokens, right_fact_tokens)
        important_overlap = left.important_terms & right.important_terms
        left_only_important = left.important_terms - right.important_terms
        right_only_important = right.important_terms - left.important_terms
        shared_ratio = len(shared_terms) / float(max(1, min(len(left_terms), len(right_terms))))

        if left.answer_signature and left.answer_signature == right.answer_signature:
            return ReasoningGraphEdge(
                source_entry_id=min(left.entry_id, right.entry_id),
                target_entry_id=max(left.entry_id, right.entry_id),
                relationship_type=RelationshipType.SUPPORTS,
                strength=round(min(1.2, 0.6 + (fact_overlap * 0.35) + (shared_ratio * 0.25)), 4),
                evidence=shared_terms[:6],
            )

        contradiction_signal = False
        if same_topic and left_only_important and right_only_important and not important_overlap:
            contradiction_signal = True
        if left.important_terms & _NEGATION_TERMS and not right.important_terms & _NEGATION_TERMS:
            contradiction_signal = contradiction_signal or (shared_ratio >= 0.25)
        if right.important_terms & _NEGATION_TERMS and not left.important_terms & _NEGATION_TERMS:
            contradiction_signal = contradiction_signal or (shared_ratio >= 0.25)
        if contradiction_signal:
            return ReasoningGraphEdge(
                source_entry_id=min(left.entry_id, right.entry_id),
                target_entry_id=max(left.entry_id, right.entry_id),
                relationship_type=RelationshipType.CONTRADICTS,
                strength=round(min(1.25, 0.5 + (shared_ratio * 0.45) + (0.2 if same_topic else 0.0)), 4),
                evidence=sorted((left_only_important | right_only_important))[:6] or shared_terms[:6],
            )

        if left.causal and not right.causal:
            return ReasoningGraphEdge(
                source_entry_id=left.entry_id,
                target_entry_id=right.entry_id,
                relationship_type=RelationshipType.CAUSES,
                strength=round(min(1.05, 0.35 + (shared_ratio * 0.4) + (0.15 if same_topic else 0.0)), 4),
                evidence=shared_terms[:6],
            )
        if right.causal and not left.causal:
            return ReasoningGraphEdge(
                source_entry_id=right.entry_id,
                target_entry_id=left.entry_id,
                relationship_type=RelationshipType.CAUSES,
                strength=round(min(1.05, 0.35 + (shared_ratio * 0.4) + (0.15 if same_topic else 0.0)), 4),
                evidence=shared_terms[:6],
            )

        detail_gap = left.detail_score - right.detail_score
        if abs(detail_gap) >= 3.0 and (same_topic or shared_ratio >= 0.35):
            extender, base = (left, right) if detail_gap > 0 else (right, left)
            return ReasoningGraphEdge(
                source_entry_id=extender.entry_id,
                target_entry_id=base.entry_id,
                relationship_type=RelationshipType.EXTENDS,
                strength=round(
                    min(
                        1.0,
                        0.22 + (abs(detail_gap) * 0.03) + (shared_ratio * 0.18) + (0.08 if same_topic else 0.0),
                    ),
                    4,
                ),
                evidence=shared_terms[:6],
            )

        if same_topic or shared_ratio >= 0.3 or fact_overlap >= 0.25:
            return ReasoningGraphEdge(
                source_entry_id=min(left.entry_id, right.entry_id),
                target_entry_id=max(left.entry_id, right.entry_id),
                relationship_type=RelationshipType.SUPPORTS,
                strength=round(min(1.0, 0.22 + (shared_ratio * 0.45) + (fact_overlap * 0.35) + (0.1 if same_topic else 0.0)), 4),
                evidence=shared_terms[:6],
            )
        return None

    def _register_edge(
        self,
        nodes: Dict[int, ReasoningGraphNode],
        edge: ReasoningGraphEdge,
    ) -> None:
        source = nodes.get(edge.source_entry_id)
        target = nodes.get(edge.target_entry_id)
        if source is None or target is None:
            return
        relationship = edge.relationship_type
        if relationship == RelationshipType.SUPPORTS:
            source.raw_support_score += edge.strength
            target.raw_support_score += edge.strength
        elif relationship == RelationshipType.CONTRADICTS:
            source.raw_contradiction_score += edge.strength
            target.raw_contradiction_score += edge.strength
        elif relationship == RelationshipType.EXTENDS:
            source.raw_extension_score += edge.strength
        elif relationship == RelationshipType.CAUSES:
            source.raw_causal_score += edge.strength
            target.raw_causal_score += edge.strength * 0.45
        source.relationships.append(
            {
                "related_entry_id": edge.target_entry_id,
                "relationship_type": relationship.value,
                "strength": round(edge.strength, 4),
                "evidence": list(edge.evidence),
            }
        )
        if relationship in {RelationshipType.SUPPORTS, RelationshipType.CONTRADICTS}:
            target.relationships.append(
                {
                    "related_entry_id": edge.source_entry_id,
                    "relationship_type": relationship.value,
                    "strength": round(edge.strength, 4),
                    "evidence": list(edge.evidence),
                }
            )
        else:
            target.relationships.append(
                {
                    "related_entry_id": edge.source_entry_id,
                    "relationship_type": f"RELATED_{relationship.value}",
                    "strength": round(edge.strength, 4),
                    "evidence": list(edge.evidence),
                }
            )

    def _finalize_node(
        self,
        *,
        node: ReasoningGraphNode,
        candidate: ReasoningGraphCandidate,
        edges: Sequence[ReasoningGraphEdge],
        query_text: str,
        query_intent: QueryIntent,
        concept_predictive_bias: float = 0.0,
    ) -> None:
        node.support_bonus = round(self._weighted_support_bonus(node=node, query_intent=query_intent), 4)
        node.extension_bonus = round(
            self._weighted_extension_bonus(raw_extension_score=node.raw_extension_score, query_intent=query_intent),
            4,
        )
        node.contradiction_penalty = round(
            self._weighted_contradiction_penalty(raw_contradiction_score=node.raw_contradiction_score),
            4,
        )
        node.directness_score = round(self._directness_score(candidate=candidate, query_text=query_text, query_intent=query_intent), 4)
        node.directness_bonus = round(
            self._weighted_directness_bonus(raw_directness_score=node.directness_score, query_intent=query_intent),
            4,
        )
        node.cluster_strength = round(self._cluster_strength(node_id=node.entry_id, edges=edges, query_intent=query_intent), 4)
        node.coherence_bonus = round(
            self._weighted_coherence_bonus(cluster_strength=node.cluster_strength, query_intent=query_intent),
            4,
        )
        node.intent_bonus = round(self._intent_bonus(candidate=candidate, query_intent=query_intent), 4)
        node.concept_predictive_bias = round(max(0.0, float(concept_predictive_bias)), 4)

    def _weighted_support_bonus(
        self,
        *,
        node: ReasoningGraphNode,
        query_intent: QueryIntent,
    ) -> float:
        return (
            (node.raw_support_score * self._support_weight(query_intent))
            + (node.raw_causal_score * self._causal_weight(query_intent))
        )

    def _weighted_contradiction_penalty(
        self,
        *,
        raw_contradiction_score: float,
    ) -> float:
        if raw_contradiction_score <= 0.0:
            return 0.0
        return (
            (raw_contradiction_score ** self.scoring.contradiction_exponent)
            * self.scoring.contradiction_weight
        )

    def _weighted_extension_bonus(
        self,
        *,
        raw_extension_score: float,
        query_intent: QueryIntent,
    ) -> float:
        extension_bonus = raw_extension_score * self.scoring.extension_weight
        if query_intent == QueryIntent.FACT:
            extension_bonus *= self.scoring.fact_extension_multiplier
        return extension_bonus

    def _weighted_directness_bonus(
        self,
        *,
        raw_directness_score: float,
        query_intent: QueryIntent,
    ) -> float:
        if raw_directness_score <= 0.0:
            return 0.0
        scaled_directness = raw_directness_score ** self.scoring.directness_exponent
        capped_directness = min(self.scoring.directness_bonus_cap, scaled_directness)
        return capped_directness * self._directness_weight(query_intent)

    def _directness_score(
        self,
        *,
        candidate: ReasoningGraphCandidate,
        query_text: str,
        query_intent: QueryIntent,
    ) -> float:
        query_focus_terms = _tokenize(query_text, drop_query_helpers=True)
        overlap = _jaccard_score(query_focus_terms, candidate.normalized_terms)
        answer_text = _normalize_text(candidate.answer_text or candidate.text)
        answer_tokens = answer_text.split()
        leading_tokens = answer_tokens[:8]
        direct_pattern = bool(set(leading_tokens) & set(query_focus_terms)) and bool(set(leading_tokens) & _DIRECT_FACT_VERBS)
        if query_intent == QueryIntent.FACT:
            score = overlap * 0.7
            if direct_pattern:
                score += self.scoring.direct_pattern_bonus
            if "color" in query_focus_terms and candidate.important_terms & _COLOR_TERMS:
                score += self.scoring.fact_color_bonus
            score += candidate.concise_score * 0.4
            if candidate.causal:
                score -= self.scoring.fact_causal_penalty
            score -= max(0.0, candidate.richness_score - 0.4) * 0.2
            return _clamp(score, low=0.0, high=1.6)
        if query_intent == QueryIntent.EXPLANATION:
            score = (overlap * 0.5) + (0.55 if candidate.causal else 0.0) + (candidate.richness_score * 0.35)
            return _clamp(score, low=0.0, high=1.6)
        if query_intent == QueryIntent.PROCESS:
            score = (overlap * 0.55) + (0.2 if candidate.causal else 0.0) + (candidate.richness_score * 0.25)
            return _clamp(score, low=0.0, high=1.4)
        score = (overlap * 0.45) + (candidate.concise_score * 0.2)
        return _clamp(score, low=0.0, high=1.1)

    def _cluster_strength(
        self,
        *,
        node_id: int,
        edges: Sequence[ReasoningGraphEdge],
        query_intent: QueryIntent,
    ) -> float:
        if not edges:
            return 0.0
        relevant_support_types = {RelationshipType.SUPPORTS}
        if query_intent in {QueryIntent.EXPLANATION, QueryIntent.PROCESS}:
            relevant_support_types.add(RelationshipType.CAUSES)
        supporters: set[int] = set()
        support_strength = 0.0
        for edge in edges:
            if edge.relationship_type not in relevant_support_types:
                continue
            if edge.source_entry_id == node_id:
                supporters.add(edge.target_entry_id)
                support_strength += edge.strength
            elif edge.target_entry_id == node_id:
                supporters.add(edge.source_entry_id)
                support_strength += edge.strength
        if not supporters:
            return 0.0
        mutual_links = 0
        for edge in edges:
            if edge.relationship_type not in relevant_support_types:
                continue
            if edge.source_entry_id in supporters and edge.target_entry_id in supporters:
                mutual_links += 1
        raw_strength = (
            (max(0, len(supporters) - 1) * self.scoring.support_neighbor_bonus)
            + (support_strength * self.scoring.support_strength_bonus)
            + (mutual_links * self.scoring.mutual_support_bonus)
        )
        return raw_strength

    def _weighted_coherence_bonus(
        self,
        *,
        cluster_strength: float,
        query_intent: QueryIntent,
    ) -> float:
        coherence_bonus = cluster_strength * self._coherence_weight(query_intent)
        if query_intent == QueryIntent.FACT:
            coherence_bonus *= self.scoring.fact_coherence_multiplier
        return coherence_bonus

    def _assign_clusters(
        self,
        *,
        nodes: Dict[int, ReasoningGraphNode],
        edges: Sequence[ReasoningGraphEdge],
        query_intent: QueryIntent,
    ) -> List[ReasoningGraphCluster]:
        if not nodes:
            return []
        edge_types = self._cluster_edge_types(query_intent)
        adjacency: Dict[int, set[int]] = {entry_id: set() for entry_id in nodes}
        support_pairs: Dict[tuple[int, int], int] = {}
        for edge in edges:
            if edge.relationship_type not in edge_types:
                continue
            left = int(edge.source_entry_id)
            right = int(edge.target_entry_id)
            if left == right:
                continue
            if left not in adjacency or right not in adjacency:
                continue
            adjacency[left].add(right)
            adjacency[right].add(left)
            support_pairs[(min(left, right), max(left, right))] = 1
        visited: set[int] = set()
        components: List[List[int]] = []
        for entry_id in sorted(nodes):
            if entry_id in visited:
                continue
            stack = [entry_id]
            visited.add(entry_id)
            component: List[int] = []
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in sorted(adjacency.get(current, set()), reverse=True):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    stack.append(neighbor)
            components.append(sorted(component))

        cluster_infos: List[Dict[str, object]] = []
        for component in components:
            support_edge_count = 0
            contradiction_edge_count = 0
            for index, left in enumerate(component):
                for right in component[index + 1:]:
                    support_edge_count += support_pairs.get((min(left, right), max(left, right)), 0)
            for edge in edges:
                if edge.relationship_type != RelationshipType.CONTRADICTS:
                    continue
                if edge.source_entry_id in component and edge.target_entry_id in component:
                    contradiction_edge_count += 1
            possible_edges = (len(component) * (len(component) - 1)) / 2.0
            density = support_edge_count / possible_edges if possible_edges else 0.0
            cluster_infos.append(
                {
                    "nodes": component,
                    "support_edges": int(support_edge_count),
                    "contradiction_edges": int(contradiction_edge_count),
                    "density": float(density),
                }
            )
        cluster_infos.sort(
            key=lambda item: (
                -float(item["density"]),
                -len(item["nodes"]),
                min(int(entry_id) for entry_id in item["nodes"]),
            )
        )

        clusters: List[ReasoningGraphCluster] = []
        for cluster_index, cluster_info in enumerate(cluster_infos, start=1):
            entry_ids = [int(entry_id) for entry_id in cluster_info["nodes"]]
            density = float(cluster_info["density"])
            support_edge_count = int(cluster_info["support_edges"])
            contradiction_edge_count = int(cluster_info["contradiction_edges"])
            cluster_size = len(entry_ids)
            dense_cluster_bonus = 0.0
            if cluster_size >= self.scoring.dense_cluster_min_size:
                dense_cluster_bonus = density * max(0, cluster_size - 1) * self.scoring.dense_cluster_bonus_weight
            for entry_id in entry_ids:
                node = nodes[entry_id]
                node.cluster_id = cluster_index
                node.cluster_size = cluster_size
                node.cluster_density = round(density, 4)
                node.dense_cluster_bonus = round(dense_cluster_bonus, 4)
            ordered_entry_ids = sorted(entry_ids, key=lambda item: (-nodes[item].final_score, item))
            node_score_sum = round(sum(nodes[entry_id].final_score for entry_id in ordered_entry_ids), 4)
            structural_bonus = round(
                (density * self.scoring.cluster_score_density_weight)
                + (support_edge_count * self.scoring.cluster_score_support_edge_weight),
                4,
            )
            cluster_score = round(node_score_sum + structural_bonus, 4)
            for rank, entry_id in enumerate(ordered_entry_ids, start=1):
                node = nodes[entry_id]
                node.cluster_rank = rank
                node.cluster_score = cluster_score
            clusters.append(
                ReasoningGraphCluster(
                    cluster_id=cluster_index,
                    nodes=ordered_entry_ids,
                    cluster_score=cluster_score,
                    support_edges=support_edge_count,
                    contradiction_edges=contradiction_edge_count,
                    density=round(density, 4),
                    node_score_sum=node_score_sum,
                    structural_bonus=structural_bonus,
                    cluster_support_bonus=0.0,
                    cluster_contradiction_penalty=0.0,
                    cluster_final_score=cluster_score,
                )
            )
        return clusters

    def _build_cluster_relationships(
        self,
        *,
        clusters: Sequence[ReasoningGraphCluster],
        nodes: Dict[int, ReasoningGraphNode],
        edges: Sequence[ReasoningGraphEdge],
    ) -> List[ReasoningGraphClusterRelationship]:
        if not clusters or not edges:
            return []
        node_to_cluster = {node.entry_id: node.cluster_id for node in nodes.values()}
        aggregates: Dict[tuple[int, int, str], Dict[str, object]] = {}
        for edge in edges:
            source_cluster_id = node_to_cluster.get(edge.source_entry_id)
            target_cluster_id = node_to_cluster.get(edge.target_entry_id)
            if source_cluster_id is None or target_cluster_id is None or source_cluster_id == target_cluster_id:
                continue
            if edge.relationship_type == RelationshipType.CONTRADICTS:
                relationship_strength = edge.strength
                for left_cluster_id, right_cluster_id in (
                    (source_cluster_id, target_cluster_id),
                    (target_cluster_id, source_cluster_id),
                ):
                    key = (left_cluster_id, right_cluster_id, RelationshipType.CONTRADICTS.value)
                    payload = aggregates.setdefault(
                        key,
                        {"strength": 0.0, "node_edge_count": 0, "evidence": set()},
                    )
                    payload["strength"] = float(payload["strength"]) + relationship_strength
                    payload["node_edge_count"] = int(payload["node_edge_count"]) + 1
                    evidence = payload["evidence"]
                    if isinstance(evidence, set):
                        evidence.update(str(item) for item in edge.evidence if str(item).strip())
                continue
            normalized_support = self._cluster_relationship_support_strength(edge)
            if normalized_support <= 0.0:
                continue
            if edge.relationship_type == RelationshipType.SUPPORTS:
                directions = (
                    (source_cluster_id, target_cluster_id),
                    (target_cluster_id, source_cluster_id),
                )
            else:
                directions = ((source_cluster_id, target_cluster_id),)
            for left_cluster_id, right_cluster_id in directions:
                key = (left_cluster_id, right_cluster_id, RelationshipType.SUPPORTS.value)
                payload = aggregates.setdefault(
                    key,
                    {"strength": 0.0, "node_edge_count": 0, "evidence": set()},
                )
                payload["strength"] = float(payload["strength"]) + normalized_support
                payload["node_edge_count"] = int(payload["node_edge_count"]) + 1
                evidence = payload["evidence"]
                if isinstance(evidence, set):
                    evidence.update(str(item) for item in edge.evidence if str(item).strip())
        relationships: List[ReasoningGraphClusterRelationship] = []
        for key, payload in aggregates.items():
            source_cluster_id, target_cluster_id, relationship_name = key
            evidence = sorted(str(item) for item in payload["evidence"] if str(item).strip())[:8]
            relationships.append(
                ReasoningGraphClusterRelationship(
                    source_cluster_id=int(source_cluster_id),
                    target_cluster_id=int(target_cluster_id),
                    relationship_type=RelationshipType(relationship_name),
                    strength=round(float(payload["strength"]), 4),
                    node_edge_count=int(payload["node_edge_count"]),
                    evidence=evidence,
                )
            )
        return relationships

    def _score_cluster_graph(
        self,
        *,
        clusters: Sequence[ReasoningGraphCluster],
        cluster_relationships: Sequence[ReasoningGraphClusterRelationship],
    ) -> List[ReasoningGraphCluster]:
        if not clusters:
            return []
        support_lookup: Dict[int, float] = {}
        contradiction_lookup: Dict[int, float] = {}
        for relationship in cluster_relationships:
            if relationship.relationship_type == RelationshipType.SUPPORTS:
                support_lookup[relationship.target_cluster_id] = support_lookup.get(relationship.target_cluster_id, 0.0) + relationship.strength
            elif relationship.relationship_type == RelationshipType.CONTRADICTS:
                contradiction_lookup[relationship.target_cluster_id] = contradiction_lookup.get(relationship.target_cluster_id, 0.0) + relationship.strength
        scored_clusters: List[ReasoningGraphCluster] = []
        for cluster in clusters:
            raw_support = support_lookup.get(cluster.cluster_id, 0.0)
            raw_contradiction = contradiction_lookup.get(cluster.cluster_id, 0.0)
            support_bonus = raw_support * self.scoring.cluster_support_weight
            contradiction_penalty = 0.0
            if raw_contradiction > 0.0:
                contradiction_penalty = (
                    raw_contradiction ** self.scoring.cluster_contradiction_exponent
                ) * self.scoring.cluster_contradiction_weight
            cluster_final_score = cluster.cluster_score + support_bonus - contradiction_penalty
            scored_clusters.append(
                ReasoningGraphCluster(
                    cluster_id=cluster.cluster_id,
                    nodes=list(cluster.nodes),
                    cluster_score=cluster.cluster_score,
                    support_edges=cluster.support_edges,
                    contradiction_edges=cluster.contradiction_edges,
                    density=cluster.density,
                    node_score_sum=cluster.node_score_sum,
                    structural_bonus=cluster.structural_bonus,
                    cluster_support_bonus=round(support_bonus, 4),
                    cluster_contradiction_penalty=round(contradiction_penalty, 4),
                    cluster_final_score=round(cluster_final_score, 4),
                    activation_value=cluster.activation_value,
                )
            )
        return scored_clusters

    def _apply_cluster_activation(
        self,
        *,
        clusters: Sequence[ReasoningGraphCluster],
        nodes: Dict[int, ReasoningGraphNode],
        query_intent: QueryIntent,
        cluster_relationships: Sequence[ReasoningGraphClusterRelationship] = (),
        edges: Sequence[ReasoningGraphEdge] = (),
    ) -> List[ReasoningGraphCluster]:
        if not clusters:
            for node in nodes.values():
                node.base_activation_value = 0.0
                node.propagated_activation_bonus = 0.0
                node.inhibition_penalty = 0.0
                node.activation_value = 0.0
            return []
        max_cluster_score = max((cluster.cluster_score for cluster in clusters), default=0.0)
        for node in nodes.values():
            node.base_activation_value = 0.0
            node.propagated_activation_bonus = 0.0
            node.inhibition_penalty = 0.0
            node.activation_value = 0.0
        cluster_activation_map: Dict[int, Dict[str, float]] = {}
        for cluster in clusters:
            cluster_nodes = [nodes[entry_id] for entry_id in cluster.nodes if entry_id in nodes]
            if cluster_nodes:
                query_match_strength = sum(node.directness_score for node in cluster_nodes) / float(len(cluster_nodes))
                max_node_score = max((node.final_score for node in cluster_nodes), default=0.0)
            else:
                query_match_strength = 0.0
                max_node_score = 0.0
            normalized_cluster_score = 0.0 if max_cluster_score <= 0.0 else (cluster.cluster_score / max_cluster_score)
            activation_value = (
                (query_match_strength * self.scoring.cluster_activation_query_match_weight)
                + (normalized_cluster_score * self.scoring.cluster_activation_score_weight)
                + (cluster.density * self.scoring.cluster_activation_density_weight)
            )
            activation_value = round(_clamp(activation_value, low=0.0, high=2.5), 4)
            cluster_activation_map[cluster.cluster_id] = {
                "base": activation_value,
                "propagated": 0.0,
                "inhibition": 0.0,
                "final": activation_value,
            }
            for node in cluster_nodes:
                node_ratio = 0.0 if max_node_score <= 0.0 else (node.final_score / max_node_score)
                node.base_activation_value = round(
                    activation_value * node_ratio * self.scoring.node_activation_weight,
                    4,
                )
                node.propagated_activation_bonus = 0.0
                node.inhibition_penalty = 0.0
                node.activation_value = node.base_activation_value

        for relationship in cluster_relationships:
            source_cluster = cluster_activation_map.get(relationship.source_cluster_id)
            target_cluster = cluster_activation_map.get(relationship.target_cluster_id)
            if source_cluster is None or target_cluster is None:
                continue
            source_final = float(source_cluster["final"])
            if relationship.relationship_type == RelationshipType.SUPPORTS:
                propagated = (
                    source_final
                    * float(relationship.strength)
                    * self.scoring.activation_decay_factor
                )
                target_cluster["propagated"] = float(target_cluster["propagated"]) + propagated
            elif relationship.relationship_type == RelationshipType.CONTRADICTS:
                inhibition = (
                    source_final
                    * float(relationship.strength)
                    * self.scoring.inhibition_factor
                )
                target_cluster["inhibition"] = float(target_cluster["inhibition"]) + inhibition

        activated_clusters: List[ReasoningGraphCluster] = []
        for cluster in clusters:
            activation_payload = cluster_activation_map.get(
                cluster.cluster_id,
                {"base": 0.0, "propagated": 0.0, "inhibition": 0.0, "final": 0.0},
            )
            cluster_final_activation = max(
                0.0,
                float(activation_payload["base"])
                + float(activation_payload["propagated"])
                - float(activation_payload["inhibition"]),
            )
            cluster_activation_map[cluster.cluster_id]["final"] = cluster_final_activation
            cluster_nodes = [nodes[entry_id] for entry_id in cluster.nodes if entry_id in nodes]
            max_node_score = max((node.final_score for node in cluster_nodes), default=0.0)
            for node in cluster_nodes:
                node_ratio = 0.0 if max_node_score <= 0.0 else (node.final_score / max_node_score)
                node.base_activation_value = round(
                    cluster_final_activation * node_ratio * self.scoring.node_activation_weight,
                    4,
                )
                node.propagated_activation_bonus = 0.0
                node.inhibition_penalty = 0.0
                node.activation_value = node.base_activation_value

            activated_clusters.append(
                ReasoningGraphCluster(
                    cluster_id=cluster.cluster_id,
                    nodes=list(cluster.nodes),
                    cluster_score=cluster.cluster_score,
                    support_edges=cluster.support_edges,
                    contradiction_edges=cluster.contradiction_edges,
                    density=cluster.density,
                    node_score_sum=cluster.node_score_sum,
                    structural_bonus=cluster.structural_bonus,
                    cluster_support_bonus=cluster.cluster_support_bonus,
                    cluster_contradiction_penalty=cluster.cluster_contradiction_penalty,
                    cluster_final_score=cluster.cluster_final_score,
                    base_activation_value=round(float(activation_payload["base"]), 4),
                    propagated_activation_bonus=round(float(activation_payload["propagated"]), 4),
                    inhibition_penalty=round(float(activation_payload["inhibition"]), 4),
                    activation_value=round(cluster_final_activation, 4),
                )
            )

        base_node_activation = {entry_id: float(node.base_activation_value) for entry_id, node in nodes.items()}
        for edge in edges:
            if edge.relationship_type == RelationshipType.CONTRADICTS:
                continue
            source_base = base_node_activation.get(edge.source_entry_id, 0.0)
            target_base = base_node_activation.get(edge.target_entry_id, 0.0)
            edge_weight = self._path_edge_weight(edge.relationship_type)
            if edge.relationship_type == RelationshipType.SUPPORTS:
                propagated_to_target = source_base * edge_weight * self.scoring.activation_decay_factor
                propagated_to_source = target_base * edge_weight * self.scoring.activation_decay_factor
                nodes[edge.target_entry_id].propagated_activation_bonus += propagated_to_target
                nodes[edge.source_entry_id].propagated_activation_bonus += propagated_to_source
            else:
                propagated = source_base * edge_weight * self.scoring.activation_decay_factor
                nodes[edge.target_entry_id].propagated_activation_bonus += propagated

        pre_inhibition_activation: Dict[int, float] = {}
        for entry_id, node in nodes.items():
            pre_inhibition = max(0.0, float(node.base_activation_value) + float(node.propagated_activation_bonus))
            pre_inhibition_activation[entry_id] = pre_inhibition
        for edge in edges:
            if edge.relationship_type != RelationshipType.CONTRADICTS:
                continue
            source_pre = pre_inhibition_activation.get(edge.source_entry_id, 0.0)
            target_pre = pre_inhibition_activation.get(edge.target_entry_id, 0.0)
            source_inhibition = target_pre * float(edge.strength) * self.scoring.inhibition_factor
            target_inhibition = source_pre * float(edge.strength) * self.scoring.inhibition_factor
            nodes[edge.source_entry_id].inhibition_penalty += source_inhibition
            nodes[edge.target_entry_id].inhibition_penalty += target_inhibition
        for node in nodes.values():
            node.activation_value = round(
                max(
                    0.0,
                    float(node.base_activation_value)
                    + float(node.propagated_activation_bonus)
                    - float(node.inhibition_penalty),
                ),
                4,
            )
        return activated_clusters

    def _build_spatial_grid(
        self,
        *,
        query_text: str,
        query_intent: QueryIntent,
        nodes: Dict[int, ReasoningGraphNode],
        clusters: Sequence[ReasoningGraphCluster],
        edges: Sequence[ReasoningGraphEdge],
        cluster_relationships: Sequence[ReasoningGraphClusterRelationship],
        concept_pre_activation_log: Sequence[str] = (),
        concept_exploration_log: Sequence[str] = (),
        anticipated_concepts: Sequence[ReasoningConceptPrediction] = (),
    ) -> ReasoningSpatialGrid | None:
        if not nodes:
            return None
        width = max(8, int(self.scoring.spatial_grid_width))
        height = max(8, int(self.scoring.spatial_grid_height))
        node_radius = max(0, int(self.scoring.spatial_node_region_radius))
        activation_grid = [[0.0 for _ in range(width)] for _ in range(height)]
        source_nodes = [[set() for _ in range(width)] for _ in range(height)]
        source_clusters = [[set() for _ in range(width)] for _ in range(height)]
        cluster_radii = self._cluster_radii(clusters=clusters, node_radius=node_radius)
        max_cluster_radius = max(cluster_radii.values(), default=max(node_radius + 1, int(self.scoring.spatial_cluster_region_radius)))
        cluster_centers = self._cluster_centers(clusters=clusters, width=width, height=height, radius=max_cluster_radius)
        node_centers = self._node_centers(
            nodes=nodes,
            clusters=clusters,
            edges=edges,
            cluster_centers=cluster_centers,
            cluster_radii=cluster_radii,
            width=width,
            height=height,
            node_radius=node_radius,
        )
        placement_map_lines = self._placement_map_lines(
            width=width,
            height=height,
            node_centers=node_centers,
            cluster_centers=cluster_centers,
        )

        node_regions: Dict[int, List[Tuple[int, int]]] = {}
        cluster_regions: Dict[int, List[Tuple[int, int]]] = {}
        for entry_id, (center_x, center_y) in node_centers.items():
            region_cells = self._region_cells(
                center_x=center_x,
                center_y=center_y,
                radius=node_radius,
                width=width,
                height=height,
            )
            node_regions[entry_id] = region_cells
            node = nodes.get(entry_id)
            if node is None:
                continue
            weights = self._region_weights(center_x=center_x, center_y=center_y, cells=region_cells, scale=1.0)
            self._seed_region(
                grid=activation_grid,
                source_nodes=source_nodes,
                source_clusters=source_clusters,
                cells=region_cells,
                weights=weights,
                total_activation=float(node.activation_value),
                entry_id=entry_id,
                cluster_id=node.cluster_id,
            )

        for cluster in clusters:
            center_x, center_y = cluster_centers.get(cluster.cluster_id, (width // 2, height // 2))
            cluster_radius = cluster_radii.get(cluster.cluster_id, max_cluster_radius)
            region_cells = self._region_cells(
                center_x=center_x,
                center_y=center_y,
                radius=cluster_radius,
                width=width,
                height=height,
            )
            cluster_regions[cluster.cluster_id] = region_cells
            weights = self._region_weights(center_x=center_x, center_y=center_y, cells=region_cells, scale=0.75)
            self._seed_region(
                grid=activation_grid,
                source_nodes=source_nodes,
                source_clusters=source_clusters,
                cells=region_cells,
                weights=weights,
                total_activation=float(cluster.activation_value) * self.scoring.spatial_cluster_activation_weight,
                entry_id=None,
                cluster_id=cluster.cluster_id,
            )

        initial_concept_clusters, initial_concept_cluster_log = self._cluster_spatial_concepts()
        (
            concept_reuse_log,
            concept_activation_log,
            activated_concepts,
            memory_exploration_log,
            cross_domain_influence_log,
        ) = self._apply_concept_memory_bias(
            grid=activation_grid,
            source_nodes=source_nodes,
            source_clusters=source_clusters,
            query_text=query_text,
            nodes=nodes,
            clusters=clusters,
            width=width,
            height=height,
        )
        meta_concept_influence_log, activated_concepts = self._apply_meta_concept_influence(
            grid=activation_grid,
            source_nodes=source_nodes,
            source_clusters=source_clusters,
            activated_concepts=activated_concepts,
            width=width,
            height=height,
        )
        concept_cluster_competition_log, activated_concepts, initial_concept_clusters = self._apply_concept_cluster_competition(
            grid=activation_grid,
            activated_concepts=activated_concepts,
            concept_clusters=initial_concept_clusters,
            width=width,
            height=height,
        )
        activated_concepts = self._mark_predicted_concept_activations(
            activated_concepts=activated_concepts,
            anticipated_concepts=anticipated_concepts,
        )
        concept_competition_log = self._apply_concept_competition(
            grid=activation_grid,
            activated_concepts=activated_concepts,
            width=width,
            height=height,
        )
        initial_grid = self._copy_grid(activation_grid)
        propagation_log = [f"initialized_total_activation={round(self._grid_total(activation_grid), 4)}"]
        current_grid = self._copy_grid(activation_grid)
        current_node_sources = self._copy_source_grid(source_nodes)
        current_cluster_sources = self._copy_source_grid(source_clusters)
        inhibition_log: List[str] = []
        region_merge_log: List[str] = []
        region_dominance_log: List[str] = []
        region_expansion_log: List[str] = []
        region_competition_log: List[str] = list(concept_cluster_competition_log) + list(concept_competition_log)
        temporal_update_log: List[str] = []
        dominance_evolution_log: List[str] = []
        dominance_state: Dict[str, float] = {}
        dominant_regions: List[ReasoningSpatialRegion] = []
        propagated_grid = self._copy_grid(current_grid)
        temporal_steps = max(1, int(self.scoring.spatial_temporal_steps))
        propagation_steps_per_tick = max(1, int(self.scoring.spatial_propagation_steps))
        for tick in range(1, temporal_steps + 1):
            for step in range(1, propagation_steps_per_tick + 1):
                current_grid, current_node_sources, current_cluster_sources = self._propagate_spatial_activation(
                    grid=current_grid,
                    node_sources=current_node_sources,
                    cluster_sources=current_cluster_sources,
                )
                propagation_log.append(
                    f"tick_{tick}_step_{step}_total_activation={round(self._grid_total(current_grid), 4)}"
                )
            if tick == 1:
                propagated_grid = self._copy_grid(current_grid)
            tick_inhibition_log = self._apply_spatial_inhibition(
                grid=current_grid,
                node_regions=node_regions,
                cluster_regions=cluster_regions,
                nodes=nodes,
                edges=edges,
                clusters=clusters,
                cluster_relationships=cluster_relationships,
                width=width,
                height=height,
            )
            inhibition_log.extend(f"t{tick}: {item}" for item in tick_inhibition_log)
            temporal_regions = self._extract_spatial_regions(
                grid=current_grid,
                node_sources=current_node_sources,
                cluster_sources=current_cluster_sources,
            )
            merged_regions, tick_merge_log = self._merge_spatial_regions(
                regions=temporal_regions,
                edges=edges,
                nodes=nodes,
            )
            region_merge_log.extend(f"t{tick}: {item}" for item in tick_merge_log)
            dominant_regions, tick_dominance_log, tick_expansion_log, dominance_state = self._score_region_dominance(
                regions=merged_regions,
                grid=current_grid,
                width=width,
                height=height,
                edges=edges,
                previous_dominance=dominance_state,
                tick=tick,
            )
            region_dominance_log.extend(tick_dominance_log)
            region_expansion_log.extend(tick_expansion_log)
            dominance_evolution_log.extend(tick_dominance_log)
            tick_competition_log = self._apply_region_competition(
                grid=current_grid,
                regions=dominant_regions,
                width=width,
                height=height,
            )
            region_competition_log.extend(f"t{tick}: {item}" for item in tick_competition_log)
            temporal_update_log.append(
                "t"
                + str(tick)
                + f": total_activation={round(self._grid_total(current_grid), 4)} "
                + f"regions={len(dominant_regions)} "
                + f"inhibition_events={len(tick_inhibition_log)} "
                + f"competition_events={len(tick_competition_log)}"
            )
        active_cells = self._active_cells_from_grid(
            grid=current_grid,
            node_sources=current_node_sources,
            cluster_sources=current_cluster_sources,
        )
        node_mappings = self._build_spatial_node_mappings(
            nodes=nodes,
            node_centers=node_centers,
            node_regions=node_regions,
            initial_grid=initial_grid,
            propagated_grid=propagated_grid,
            final_grid=current_grid,
        )
        cluster_mappings = self._build_spatial_cluster_mappings(
            clusters=clusters,
            cluster_centers=cluster_centers,
            cluster_regions=cluster_regions,
            final_grid=current_grid,
            cluster_radii=cluster_radii,
        )
        top_regions = self._extract_spatial_regions(
            grid=current_grid,
            node_sources=current_node_sources,
            cluster_sources=current_cluster_sources,
        )
        top_regions = self._reconcile_region_metadata(
            regions=top_regions,
            prior_regions=dominant_regions,
            nodes=nodes,
        )
        top_regions = [
            ReasoningSpatialRegion(
                region_id=region.region_id,
                cells=list(region.cells),
                total_activation=region.total_activation,
                peak_activation=region.peak_activation,
                centroid_x=region.centroid_x,
                centroid_y=region.centroid_y,
                source_nodes=list(region.source_nodes),
                source_clusters=list(region.source_clusters),
                competition_penalty=region.competition_penalty,
                post_competition_rank=index,
                dominance_value=region.dominance_value,
                merged_region_ids=list(region.merged_region_ids),
                dominant_node_id=region.dominant_node_id,
            )
            for index, region in enumerate(top_regions, start=1)
        ]
        top_regions = self._rank_regions_with_concept_influence(
            regions=top_regions,
            activated_concepts=activated_concepts,
        )
        (
            concept_creation_log,
            concept_merge_log,
            concept_evolution_log,
            concept_hierarchy_log,
            concept_cluster_log,
            concept_clusters,
            prediction_accuracy_log,
            concept_importance_evolution_log,
            concept_cross_domain_link_log,
            concept_generalization_log,
            concept_stability_log,
            concept_uncertainty_log,
            concept_abstraction_log,
            temporal_sequence_log,
            temporal_prediction_log,
            goal_selection_log,
            goal_plan_log,
            goal_feedback_log,
            temporal_transitions,
            active_goal,
            goal_plans,
        ) = self._remember_spatial_concepts(
            query_text=query_text,
            query_intent=query_intent,
            regions=top_regions,
            grid=current_grid,
            edges=edges,
            activated_concepts=activated_concepts,
            anticipated_concepts=anticipated_concepts,
        )
        if initial_concept_cluster_log:
            concept_cluster_log = list(initial_concept_cluster_log) + list(concept_cluster_log)
        concept_clusters = self._carry_forward_concept_cluster_competition(
            concept_clusters=concept_clusters,
            competed_clusters=initial_concept_clusters,
        )
        concept_importance_log = self._concept_importance_log()
        return ReasoningSpatialGrid(
            width=width,
            height=height,
            decay_factor=self.scoring.spatial_decay_factor,
            propagation_steps=temporal_steps * propagation_steps_per_tick,
            active_cells=active_cells,
            node_mappings=node_mappings,
            cluster_mappings=cluster_mappings,
            pre_competition_regions=dominant_regions,
            top_regions=top_regions,
            heatmap_before_lines=self._heatmap_lines(initial_grid),
            heatmap_after_lines=self._heatmap_lines(current_grid),
            propagation_log=propagation_log,
            inhibition_log=inhibition_log,
            region_competition_log=region_competition_log,
            region_dominance_log=region_dominance_log,
            region_merge_log=region_merge_log,
            region_expansion_log=region_expansion_log,
            temporal_update_log=temporal_update_log,
            dominance_evolution_log=dominance_evolution_log,
            concept_creation_log=concept_creation_log,
            concept_merge_log=concept_merge_log,
            concept_reuse_log=concept_reuse_log,
            concept_activation_log=concept_activation_log,
            concept_evolution_log=concept_evolution_log,
            concept_competition_log=concept_competition_log,
            concept_pre_activation_log=list(concept_pre_activation_log),
            concept_importance_log=concept_importance_log,
            concept_hierarchy_log=concept_hierarchy_log,
            concept_cluster_log=concept_cluster_log,
            meta_concept_influence_log=meta_concept_influence_log,
            concept_cluster_competition_log=concept_cluster_competition_log,
            anticipation_log=[
                f"predicted concept {prediction.concept_id}: score={round(prediction.predicted_score, 4)} similarity={round(prediction.similarity, 4)} importance={round(prediction.importance, 4)} confidence={round(prediction.prediction_confidence, 4)} meta={prediction.is_meta_concept} generalized={prediction.is_generalized_concept} abstraction_level={prediction.abstraction_level} cluster={prediction.concept_cluster_id}"
                for prediction in anticipated_concepts
            ],
            prediction_accuracy_log=prediction_accuracy_log,
            concept_importance_evolution_log=concept_importance_evolution_log,
            concept_exploration_log=list(concept_exploration_log) + list(memory_exploration_log),
            concept_uncertainty_log=concept_uncertainty_log,
            concept_cross_domain_link_log=list(cross_domain_influence_log) + list(concept_cross_domain_link_log),
            concept_generalization_log=concept_generalization_log,
            concept_stability_log=concept_stability_log,
            concept_abstraction_log=concept_abstraction_log,
            temporal_sequence_log=temporal_sequence_log,
            temporal_prediction_log=temporal_prediction_log,
            goal_selection_log=goal_selection_log,
            goal_plan_log=goal_plan_log,
            goal_feedback_log=goal_feedback_log,
            activated_concepts=activated_concepts,
            anticipated_concepts=list(anticipated_concepts),
            concept_clusters=concept_clusters,
            concept_memory_snapshot=[concept for concept in self.spatial_concept_memory],
            temporal_transitions=temporal_transitions,
            active_goal=active_goal,
            goal_plans=goal_plans,
            placement_map_lines=placement_map_lines,
            cluster_boundary_lines=self._cluster_boundary_lines(cluster_mappings=cluster_mappings),
        )

    def _cluster_centers(
        self,
        *,
        clusters: Sequence[ReasoningGraphCluster],
        width: int,
        height: int,
        radius: int,
    ) -> Dict[int, Tuple[int, int]]:
        ordered_clusters = sorted(
            clusters,
            key=lambda item: (-item.cluster_final_score, -item.cluster_score, item.cluster_id),
        )
        if not ordered_clusters:
            return {}
        center_x = width // 2
        center_y = height // 2
        if len(ordered_clusters) == 1:
            return {ordered_clusters[0].cluster_id: (center_x, center_y)}
        orbit_radius = max(3.0, min(width, height) * 0.25)
        usable_x_min = radius + 1
        usable_x_max = max(usable_x_min, width - radius - 2)
        usable_y_min = radius + 1
        usable_y_max = max(usable_y_min, height - radius - 2)
        centers: Dict[int, Tuple[int, int]] = {}
        for index, cluster in enumerate(ordered_clusters):
            angle = (-math.pi / 2.0) + ((2.0 * math.pi * index) / float(len(ordered_clusters)))
            x = int(round(center_x + (math.cos(angle) * orbit_radius)))
            y = int(round(center_y + (math.sin(angle) * orbit_radius)))
            centers[cluster.cluster_id] = (
                max(usable_x_min, min(usable_x_max, x)),
                max(usable_y_min, min(usable_y_max, y)),
            )
        return centers

    def _cluster_radii(
        self,
        *,
        clusters: Sequence[ReasoningGraphCluster],
        node_radius: int,
    ) -> Dict[int, int]:
        radii: Dict[int, int] = {}
        base_radius = max(node_radius + 1, int(self.scoring.spatial_cluster_region_radius))
        for cluster in clusters:
            cluster_size = max(1, len(cluster.nodes))
            scaled_radius = base_radius + int(round(math.sqrt(float(cluster_size - 1)) * self.scoring.spatial_cluster_size_radius_weight))
            radii[cluster.cluster_id] = max(base_radius, scaled_radius)
        return radii

    def _node_centers(
        self,
        *,
        nodes: Dict[int, ReasoningGraphNode],
        clusters: Sequence[ReasoningGraphCluster],
        edges: Sequence[ReasoningGraphEdge],
        cluster_centers: Dict[int, Tuple[int, int]],
        cluster_radii: Dict[int, int],
        width: int,
        height: int,
        node_radius: int,
    ) -> Dict[int, Tuple[int, int]]:
        positions = self._initial_node_positions(
            nodes=nodes,
            clusters=clusters,
            cluster_centers=cluster_centers,
            cluster_radii=cluster_radii,
        )
        relation_lookup = self._spatial_relation_lookup(edges=edges)
        node_ids = sorted(nodes.keys())
        for _ in range(max(1, int(self.scoring.spatial_force_iterations))):
            for left_index, left_id in enumerate(node_ids):
                for right_id in node_ids[left_index + 1:]:
                    left_pos = positions[left_id]
                    right_pos = positions[right_id]
                    dx = right_pos[0] - left_pos[0]
                    dy = right_pos[1] - left_pos[1]
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance < 0.001:
                        dx = 0.23 if left_id < right_id else -0.23
                        dy = 0.17 if ((left_id + right_id) % 2) == 0 else -0.17
                        distance = math.sqrt((dx * dx) + (dy * dy))
                    unit_x = dx / distance
                    unit_y = dy / distance
                    desired_distance, rate = self._spatial_pair_rule(
                        left_id=left_id,
                        right_id=right_id,
                        relation_lookup=relation_lookup,
                        nodes=nodes,
                    )
                    adjustment = (distance - desired_distance) * rate
                    positions[left_id] = (left_pos[0] + (unit_x * adjustment * 0.5), left_pos[1] + (unit_y * adjustment * 0.5))
                    positions[right_id] = (right_pos[0] - (unit_x * adjustment * 0.5), right_pos[1] - (unit_y * adjustment * 0.5))
            for node_id in node_ids:
                node = nodes[node_id]
                cluster_center_x, cluster_center_y = cluster_centers.get(node.cluster_id, (width / 2.0, height / 2.0))
                cluster_radius = cluster_radii.get(node.cluster_id, max(node_radius + 1, int(self.scoring.spatial_cluster_region_radius)))
                current_x, current_y = positions[node_id]
                offset_x = current_x - cluster_center_x
                offset_y = current_y - cluster_center_y
                distance_from_center = math.sqrt((offset_x * offset_x) + (offset_y * offset_y))
                if distance_from_center > float(cluster_radius):
                    pull = min(1.0, (distance_from_center - float(cluster_radius)) * self.scoring.spatial_cluster_anchor_rate)
                    current_x -= offset_x * pull
                    current_y -= offset_y * pull
                else:
                    current_x += (cluster_center_x - current_x) * self.scoring.spatial_cluster_anchor_rate * 0.25
                    current_y += (cluster_center_y - current_y) * self.scoring.spatial_cluster_anchor_rate * 0.25
                min_x = float(node_radius)
                max_x = float(max(node_radius, width - node_radius - 1))
                min_y = float(node_radius)
                max_y = float(max(node_radius, height - node_radius - 1))
                positions[node_id] = (
                    max(min_x, min(max_x, current_x)),
                    max(min_y, min(max_y, current_y)),
                )

        centers: Dict[int, Tuple[int, int]] = {}
        used_centers: Set[Tuple[int, int]] = set()
        for cluster in sorted(clusters, key=lambda item: item.cluster_id):
            cluster_nodes = [nodes[entry_id] for entry_id in cluster.nodes if entry_id in nodes]
            cluster_nodes.sort(key=lambda item: (-item.final_score, item.entry_id))
            candidate_offsets = self._candidate_offsets(max_distance=max(4, cluster_radii.get(cluster.cluster_id, 4) + 2))
            for node in cluster_nodes:
                target_x, target_y = positions[node.entry_id]
                center = self._resolve_node_center_near_target(
                    target=(target_x, target_y),
                    candidate_offsets=candidate_offsets,
                    used_centers=used_centers,
                    width=width,
                    height=height,
                    node_radius=node_radius,
                )
                centers[node.entry_id] = center
                used_centers.add(center)
        return centers

    def _initial_node_positions(
        self,
        *,
        nodes: Dict[int, ReasoningGraphNode],
        clusters: Sequence[ReasoningGraphCluster],
        cluster_centers: Dict[int, Tuple[int, int]],
        cluster_radii: Dict[int, int],
    ) -> Dict[int, Tuple[float, float]]:
        positions: Dict[int, Tuple[float, float]] = {}
        for cluster in sorted(clusters, key=lambda item: item.cluster_id):
            cluster_center_x, cluster_center_y = cluster_centers.get(cluster.cluster_id, (0, 0))
            cluster_radius = max(1, cluster_radii.get(cluster.cluster_id, 1) - 1)
            cluster_nodes = [nodes[entry_id] for entry_id in cluster.nodes if entry_id in nodes]
            cluster_nodes.sort(key=lambda item: (-item.final_score, item.entry_id))
            total_nodes = max(1, len(cluster_nodes))
            for index, node in enumerate(cluster_nodes):
                if total_nodes == 1:
                    positions[node.entry_id] = (float(cluster_center_x), float(cluster_center_y))
                    continue
                angle = (2.0 * math.pi * index) / float(total_nodes)
                distance = min(float(cluster_radius), 1.0 + (index * 0.6))
                positions[node.entry_id] = (
                    float(cluster_center_x) + (math.cos(angle) * distance),
                    float(cluster_center_y) + (math.sin(angle) * distance),
                )
        return positions

    def _spatial_relation_lookup(
        self,
        *,
        edges: Sequence[ReasoningGraphEdge],
    ) -> Dict[Tuple[int, int], ReasoningGraphEdge]:
        lookup: Dict[Tuple[int, int], ReasoningGraphEdge] = {}
        for edge in edges:
            lookup[(edge.source_entry_id, edge.target_entry_id)] = edge
            if edge.relationship_type in {RelationshipType.SUPPORTS, RelationshipType.CONTRADICTS}:
                lookup[(edge.target_entry_id, edge.source_entry_id)] = edge
        return lookup

    def _spatial_pair_rule(
        self,
        *,
        left_id: int,
        right_id: int,
        relation_lookup: Dict[Tuple[int, int], ReasoningGraphEdge],
        nodes: Dict[int, ReasoningGraphNode],
    ) -> Tuple[float, float]:
        support_strength = 0.0
        contradiction_strength = 0.0
        causal_strength = 0.0
        extend_strength = 0.0
        for candidate_pair in ((left_id, right_id), (right_id, left_id)):
            edge = relation_lookup.get(candidate_pair)
            if edge is None:
                continue
            strength = float(edge.strength)
            if edge.relationship_type == RelationshipType.SUPPORTS:
                support_strength = max(support_strength, strength)
            elif edge.relationship_type == RelationshipType.CONTRADICTS:
                contradiction_strength = max(contradiction_strength, strength)
            elif edge.relationship_type == RelationshipType.CAUSES:
                causal_strength = max(causal_strength, strength)
            elif edge.relationship_type == RelationshipType.EXTENDS:
                extend_strength = max(extend_strength, strength)
        if support_strength > 0.0 or contradiction_strength > 0.0 or causal_strength > 0.0 or extend_strength > 0.0:
            desired_distance = (
                self.scoring.spatial_base_pair_distance
                - (support_strength * self.scoring.spatial_support_distance_weight)
                - (causal_strength * self.scoring.spatial_causal_distance_weight)
                - (extend_strength * self.scoring.spatial_extend_distance_weight)
                + (contradiction_strength * self.scoring.spatial_contradiction_distance_weight)
            )
            desired_distance = _clamp(
                desired_distance,
                low=0.8,
                high=self.scoring.spatial_contradiction_target_distance + 2.0,
            )
            if contradiction_strength > max(support_strength, causal_strength, extend_strength):
                return desired_distance, self.scoring.spatial_repulsion_rate
            return desired_distance, self.scoring.spatial_attraction_rate
        same_cluster = nodes[left_id].cluster_id == nodes[right_id].cluster_id
        if same_cluster:
            return self.scoring.spatial_weak_target_distance, self.scoring.spatial_attraction_rate * 0.4
        return self.scoring.spatial_contradiction_target_distance * 0.75, self.scoring.spatial_repulsion_rate * 0.25

    def _candidate_offsets(self, *, max_distance: int) -> List[Tuple[int, int]]:
        offsets: List[Tuple[int, int]] = [(0, 0)]
        for dy in range(-max_distance, max_distance + 1):
            for dx in range(-max_distance, max_distance + 1):
                if dx == 0 and dy == 0:
                    continue
                offsets.append((dx, dy))
        return offsets

    def _resolve_node_center_near_target(
        self,
        *,
        target: Tuple[float, float],
        candidate_offsets: Sequence[Tuple[int, int]],
        used_centers: Set[Tuple[int, int]],
        width: int,
        height: int,
        node_radius: int,
    ) -> Tuple[int, int]:
        target_x, target_y = target
        scored_offsets: List[Tuple[float, int, int]] = []
        for dx, dy in candidate_offsets:
            candidate_x = int(round(target_x + dx))
            candidate_y = int(round(target_y + dy))
            score = abs(candidate_x - target_x) + abs(candidate_y - target_y)
            scored_offsets.append((score, dx, dy))
        scored_offsets.sort(key=lambda item: (item[0], abs(item[1]) + abs(item[2]), item[2], item[1]))
        min_x = node_radius
        max_x = max(min_x, width - node_radius - 1)
        min_y = node_radius
        max_y = max(min_y, height - node_radius - 1)
        for _, dx, dy in scored_offsets:
            x = max(min_x, min(max_x, int(round(target_x + dx))))
            y = max(min_y, min(max_y, int(round(target_y + dy))))
            if (x, y) not in used_centers:
                return (x, y)
        return (
            max(min_x, min(max_x, int(round(target_x)))),
            max(min_y, min(max_y, int(round(target_y)))),
        )

    def _region_cells(
        self,
        *,
        center_x: int,
        center_y: int,
        radius: int,
        width: int,
        height: int,
    ) -> List[Tuple[int, int]]:
        cells: List[Tuple[int, int]] = []
        for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
            for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
                cells.append((x, y))
        return cells

    def _region_weights(
        self,
        *,
        center_x: int,
        center_y: int,
        cells: Sequence[Tuple[int, int]],
        scale: float,
    ) -> Dict[Tuple[int, int], float]:
        raw_weights: Dict[Tuple[int, int], float] = {}
        total = 0.0
        for x, y in cells:
            distance = math.sqrt(float(((x - center_x) ** 2) + ((y - center_y) ** 2)))
            weight = max(0.1, 1.0 / (1.0 + distance))
            raw_weights[(x, y)] = weight
            total += weight
        if total <= 0.0:
            return {(x, y): 0.0 for x, y in cells}
        return {(x, y): (weight / total) * scale for (x, y), weight in raw_weights.items()}

    def _seed_region(
        self,
        *,
        grid: List[List[float]],
        source_nodes: List[List[Set[int]]],
        source_clusters: List[List[Set[int]]],
        cells: Sequence[Tuple[int, int]],
        weights: Dict[Tuple[int, int], float],
        total_activation: float,
        entry_id: int | None,
        cluster_id: int | None,
    ) -> None:
        if total_activation <= 0.0:
            return
        for x, y in cells:
            contribution = total_activation * float(weights.get((x, y), 0.0))
            if contribution <= 0.0:
                continue
            grid[y][x] += contribution
            if entry_id is not None:
                source_nodes[y][x].add(int(entry_id))
            if cluster_id is not None and cluster_id >= 0:
                source_clusters[y][x].add(int(cluster_id))

    def _concept_pre_activation_bias(
        self,
        *,
        query_text: str,
        nodes: Dict[int, ReasoningGraphNode],
    ) -> tuple[Dict[int, float], List[str], List[ReasoningConceptPrediction], List[str]]:
        if not self.spatial_concept_memory or not nodes:
            return {}, [], [], []
        query_terms = set(_tokenize(query_text))
        current_node_ids = set(nodes.keys())
        current_relationship_types = {
            relationship_type
            for node in nodes.values()
            for relationship in node.relationships
            for relationship_type in [str(relationship.get("relationship_type", ""))]
            if relationship_type
        }
        matches: List[tuple[float, float, ReasoningSpatialConcept, float, float, int | None]] = []
        exploration_logs: List[str] = []
        for concept in self.spatial_concept_memory:
            importance = self._recalculate_concept_importance(concept)
            similarity = self._concept_similarity(
                concept=concept,
                node_ids=current_node_ids,
                cluster_ids=set(),
                relationship_types=current_relationship_types,
                query_terms=query_terms,
            )
            cross_domain_similarity, cross_domain_source_id = self._cross_domain_query_similarity(
                concept=concept,
                query_terms=query_terms,
            )
            effective_similarity = max(similarity, cross_domain_similarity)
            if concept.is_generalized_concept and cross_domain_source_id is not None:
                effective_similarity = max(effective_similarity, self.scoring.concept_generalized_cross_domain_floor)
            exploration_bonus, exploration_roll = self._concept_exploration_bonus(
                query_text=query_text,
                concept=concept,
                similarity=effective_similarity,
            )
            effective_threshold = float(self.scoring.concept_memory_similarity_threshold)
            if concept.is_generalized_concept:
                effective_threshold = max(0.02, effective_threshold * 0.55)
            if effective_similarity < effective_threshold and exploration_bonus <= 0.0:
                continue
            if effective_similarity < effective_threshold and exploration_bonus > 0.0:
                effective_similarity = max(effective_similarity, 0.35)
            relevance = effective_similarity * (1.0 + (importance * self.scoring.concept_importance_influence_weight))
            if exploration_bonus > 0.0:
                relevance += exploration_bonus
                exploration_logs.append(
                    f"concept {concept.concept_id} exploration boost: roll={round(exploration_roll, 4)} rate={round(self._concept_exploration_threshold(concept), 4)} uncertainty={round(concept.uncertainty, 4)} confidence={round(concept.prediction_confidence, 4)} importance={round(importance, 4)} similarity={round(effective_similarity, 4)} boost={round(exploration_bonus, 4)} abstraction_level={self._concept_abstraction_level(concept)}"
                )
            pre_activation_influence = relevance * self.scoring.concept_pre_activation_node_weight
            matches.append((relevance, effective_similarity, concept, pre_activation_influence, cross_domain_similarity, cross_domain_source_id))
        matches.sort(key=lambda item: (-item[0], -item[1], item[2].concept_id))
        limit = max(1, int(self.scoring.concept_pre_activation_limit))
        predictive_biases: Dict[int, float] = {}
        logs: List[str] = []
        anticipated_concepts: List[ReasoningConceptPrediction] = []
        for relevance, similarity, concept, pre_activation_influence, cross_domain_similarity, cross_domain_source_id in matches[:limit]:
            matched_nodes = sorted(current_node_ids & set(concept.associated_nodes))
            for entry_id in matched_nodes:
                predictive_biases[entry_id] = predictive_biases.get(entry_id, 0.0) + pre_activation_influence
            child_node_ids: Set[int] = set()
            if concept.is_meta_concept:
                for child in self._child_concepts(concept):
                    child_node_ids.update(set(child.associated_nodes) & current_node_ids)
                child_weight = self.scoring.concept_meta_child_influence_weight
                if concept.is_generalized_concept:
                    child_weight += self.scoring.concept_generalized_child_influence_weight
                for entry_id in sorted(child_node_ids):
                    predictive_biases[entry_id] = (
                        predictive_biases.get(entry_id, 0.0)
                        + (pre_activation_influence * child_weight)
                    )
            prediction = ReasoningConceptPrediction(
                concept_id=concept.concept_id,
                predicted_score=round(relevance, 4),
                similarity=round(similarity, 4),
                importance=round(concept.importance, 4),
                prediction_confidence=round(concept.prediction_confidence, 4),
                is_meta_concept=concept.is_meta_concept,
                is_generalized_concept=concept.is_generalized_concept,
                abstraction_level=self._concept_abstraction_level(concept),
                concept_cluster_id=concept.concept_cluster_id,
                child_concept_ids=list(concept.child_concept_ids),
            )
            anticipated_concepts.append(prediction)
            logs.append(
                f"concept {concept.concept_id} pre-activated: similarity={round(similarity, 4)} importance={round(concept.importance, 4)} relevance={round(relevance, 4)} prediction_confidence={round(concept.prediction_confidence, 4)} uncertainty={round(concept.uncertainty, 4)} node_bias={round(pre_activation_influence, 4)} matched_nodes={matched_nodes} child_nodes={sorted(child_node_ids)} cross_domain_source={cross_domain_source_id} cross_domain_similarity={round(cross_domain_similarity, 4)} meta={concept.is_meta_concept} generalized={concept.is_generalized_concept} abstraction_level={self._concept_abstraction_level(concept)}"
            )
        return {
            entry_id: round(value, 4)
            for entry_id, value in predictive_biases.items()
        }, logs, anticipated_concepts, exploration_logs

    def _apply_concept_memory_bias(
        self,
        *,
        grid: List[List[float]],
        source_nodes: List[List[Set[int]]],
        source_clusters: List[List[Set[int]]],
        query_text: str,
        nodes: Dict[int, ReasoningGraphNode],
        clusters: Sequence[ReasoningGraphCluster],
        width: int,
        height: int,
    ) -> tuple[List[str], List[str], List[ReasoningSpatialConceptActivation], List[str], List[str]]:
        if not self.spatial_concept_memory:
            return [], [], [], [], []
        logs: List[str] = []
        activation_logs: List[str] = []
        exploration_logs: List[str] = []
        cross_domain_logs: List[str] = []
        activated_concepts: List[ReasoningSpatialConceptActivation] = []
        query_terms = set(_tokenize(query_text))
        current_node_ids = set(nodes.keys())
        current_cluster_ids = {cluster.cluster_id for cluster in clusters}
        current_relationship_types = {
            relationship_type
            for node in nodes.values()
            for relationship in node.relationships
            for relationship_type in [str(relationship.get("relationship_type", ""))]
            if relationship_type
        }
        for concept in self.spatial_concept_memory:
            importance = self._recalculate_concept_importance(concept)
            similarity = self._concept_similarity(
                concept=concept,
                node_ids=current_node_ids,
                cluster_ids=current_cluster_ids,
                relationship_types=current_relationship_types,
                query_terms=query_terms,
            )
            cross_domain_similarity, cross_domain_source_id = self._cross_domain_query_similarity(
                concept=concept,
                query_terms=query_terms,
            )
            effective_similarity = max(similarity, cross_domain_similarity)
            if concept.is_generalized_concept and cross_domain_source_id is not None:
                effective_similarity = max(effective_similarity, self.scoring.concept_generalized_cross_domain_floor)
            if cross_domain_source_id is not None and cross_domain_similarity >= similarity:
                cross_domain_logs.append(
                    f"concept {concept.concept_id} received cross-domain activation from concept {cross_domain_source_id}: base_similarity={round(similarity, 4)} link_similarity={round(cross_domain_similarity, 4)} generalized={concept.is_generalized_concept} abstraction_level={self._concept_abstraction_level(concept)}"
                )
            exploration_bonus, exploration_roll = self._concept_exploration_bonus(
                query_text=query_text,
                concept=concept,
                similarity=effective_similarity,
            )
            effective_threshold = float(self.scoring.concept_memory_similarity_threshold)
            if concept.is_generalized_concept:
                effective_threshold = max(0.02, effective_threshold * 0.55)
            if effective_similarity < effective_threshold and exploration_bonus <= 0.0:
                continue
            if effective_similarity < effective_threshold and exploration_bonus > 0.0:
                effective_similarity = max(effective_similarity, 0.35)
            if exploration_bonus > 0.0:
                exploration_logs.append(
                    f"concept {concept.concept_id} memory exploration boost: roll={round(exploration_roll, 4)} rate={round(self._concept_exploration_threshold(concept), 4)} uncertainty={round(concept.uncertainty, 4)} confidence={round(concept.prediction_confidence, 4)} importance={round(importance, 4)} similarity={round(effective_similarity, 4)} boost={round(exploration_bonus, 4)} abstraction_level={self._concept_abstraction_level(concept)}"
                )
            importance_multiplier = 1.0 + (importance * self.scoring.concept_importance_influence_weight)
            influence = (
                float(concept.activation_value)
                * effective_similarity
                * self.scoring.concept_memory_influence_weight
                * importance_multiplier
                + exploration_bonus
            )
            pre_activation_influence = (
                effective_similarity
                * importance_multiplier
                * self.scoring.concept_pre_activation_node_weight
                + (exploration_bonus * 0.25)
            )
            pre_reuse_count = int(concept.reuse_count)
            cells = self._region_cells(
                center_x=int(round(concept.centroid_x)),
                center_y=int(round(concept.centroid_y)),
                radius=max(1, int(self.scoring.spatial_node_region_radius) + 1),
                width=width,
                height=height,
            )
            weights = self._region_weights(
                center_x=int(round(concept.centroid_x)),
                center_y=int(round(concept.centroid_y)),
                cells=cells,
                scale=1.0,
            )
            matched_nodes = sorted(current_node_ids & set(concept.associated_nodes))
            matched_clusters = sorted(current_cluster_ids & set(concept.associated_clusters))
            concept_query_overlap = sorted(query_terms & set(concept.associated_terms))
            self._seed_region(
                grid=grid,
                source_nodes=source_nodes,
                source_clusters=source_clusters,
                cells=cells,
                weights=weights,
                total_activation=influence,
                entry_id=matched_nodes[0] if matched_nodes else None,
                cluster_id=matched_clusters[0] if matched_clusters else None,
            )
            concept.reuse_count += 1
            activation = ReasoningSpatialConceptActivation(
                concept_id=concept.concept_id,
                similarity=round(effective_similarity, 4),
                influence=round(influence, 4),
                importance=round(importance, 4),
                pre_activation_influence=round(pre_activation_influence, 4),
                pre_reuse_count=pre_reuse_count,
                post_reuse_count=int(concept.reuse_count),
                centroid_x=concept.centroid_x,
                centroid_y=concept.centroid_y,
                associated_nodes=list(concept.associated_nodes),
                associated_clusters=list(concept.associated_clusters),
                is_meta_concept=concept.is_meta_concept,
                is_generalized_concept=concept.is_generalized_concept,
                abstraction_level=self._concept_abstraction_level(concept),
                concept_cluster_id=concept.concept_cluster_id,
            )
            activated_concepts.append(activation)
            logs.append(
                f"concept {concept.concept_id} reused: similarity={round(effective_similarity, 4)} importance={round(importance, 4)} uncertainty={round(concept.uncertainty, 4)} confidence={round(concept.prediction_confidence, 4)} influence={round(influence, 4)} reuse_count={pre_reuse_count}->{concept.reuse_count} matched_nodes={matched_nodes} matched_clusters={matched_clusters} query_overlap={concept_query_overlap} cross_domain_source={cross_domain_source_id} meta={concept.is_meta_concept} generalized={concept.is_generalized_concept} abstraction_level={self._concept_abstraction_level(concept)}"
            )
            activation_logs.append(
                f"concept {concept.concept_id} activation influence={round(influence, 4)} pre_activation={round(pre_activation_influence, 4)} similarity={round(effective_similarity, 4)} importance={round(importance, 4)} uncertainty={round(concept.uncertainty, 4)} abstraction_level={self._concept_abstraction_level(concept)} centroid=({round(concept.centroid_x, 2)}, {round(concept.centroid_y, 2)}) cluster={concept.concept_cluster_id}"
            )
        activated_concepts.sort(key=lambda item: (-item.influence, -item.similarity, item.concept_id))
        return logs, activation_logs, activated_concepts, exploration_logs, cross_domain_logs

    def _concept_similarity(
        self,
        *,
        concept: ReasoningSpatialConcept,
        node_ids: Set[int],
        cluster_ids: Set[int],
        relationship_types: Set[str],
        query_terms: Set[str] | None = None,
    ) -> float:
        node_overlap = len(node_ids & set(concept.associated_nodes))
        cluster_overlap = len(cluster_ids & set(concept.associated_clusters))
        relation_overlap = len(relationship_types & set(concept.relationship_types))
        query_terms = query_terms or set()
        # query term scoring is optional for backward-compatible concepts loaded from disk.
        node_score = 0.0 if not concept.associated_nodes else node_overlap / float(max(1, len(set(concept.associated_nodes))))
        cluster_score = 0.0 if not concept.associated_clusters else cluster_overlap / float(max(1, len(set(concept.associated_clusters))))
        relation_score = 0.0 if not concept.relationship_types else relation_overlap / float(max(1, len(set(concept.relationship_types))))
        term_score = 0.0 if not concept.associated_terms else len(query_terms & set(concept.associated_terms)) / float(max(1, len(set(concept.associated_terms))))
        return _clamp((node_score * 0.45) + (cluster_score * 0.25) + (relation_score * 0.15) + (term_score * 0.15), low=0.0, high=1.0)

    def _cross_domain_query_similarity(
        self,
        *,
        concept: ReasoningSpatialConcept,
        query_terms: Set[str],
    ) -> tuple[float, int | None]:
        if not concept.cross_domain_links or not query_terms:
            return 0.0, None
        concept_lookup = {
            item.concept_id: item
            for item in self.spatial_concept_memory
        }
        best_similarity = 0.0
        best_source_id: int | None = None
        for linked_id, strength in sorted(concept.cross_domain_links.items()):
            linked = concept_lookup.get(int(linked_id))
            if linked is None or not linked.associated_terms:
                continue
            linked_terms = set(linked.associated_terms)
            term_overlap = len(query_terms & linked_terms) / float(max(1, min(len(query_terms), len(linked_terms))))
            similarity = float(strength) * term_overlap
            if similarity > best_similarity:
                best_similarity = similarity
                best_source_id = int(linked_id)
        return round(_clamp(best_similarity, low=0.0, high=1.0), 4), best_source_id

    def _concept_exploration_threshold(self, concept: ReasoningSpatialConcept) -> float:
        uncertainty = self._update_concept_uncertainty(concept)
        low_confidence = 1.0 - _clamp(float(concept.prediction_confidence), low=0.0, high=1.0)
        return round(
            _clamp(
                self.scoring.concept_exploration_rate
                + float(self.simulation_model_state.adaptive_exploration_boost)
                + (uncertainty * self.scoring.concept_uncertainty_exploration_weight)
                + (low_confidence * self.scoring.concept_low_confidence_exploration_weight),
                low=0.0,
                high=1.0,
            ),
            4,
        )

    def _concept_exploration_bonus(
        self,
        *,
        query_text: str,
        concept: ReasoningSpatialConcept,
        similarity: float,
    ) -> tuple[float, float]:
        importance = float(concept.importance)
        uncertainty = self._update_concept_uncertainty(concept)
        low_confidence = 1.0 - _clamp(float(concept.prediction_confidence), low=0.0, high=1.0)
        if (
            importance > self.scoring.concept_exploration_importance_threshold
            and uncertainty < 0.5
            and low_confidence < 0.45
        ):
            return 0.0, self._controlled_exploration_roll(query_text=query_text, concept_id=concept.concept_id)
        roll = self._controlled_exploration_roll(query_text=query_text, concept_id=concept.concept_id)
        if roll >= self._concept_exploration_threshold(concept):
            return 0.0, roll
        importance_gap = max(0.0, self.scoring.concept_exploration_importance_threshold - importance)
        uncertainty_drive = 0.35 + (uncertainty * 0.55) + (low_confidence * 0.35)
        bonus = self.scoring.concept_exploration_boost * uncertainty_drive * (0.5 + importance_gap) * max(0.35, similarity)
        return round(_clamp(bonus, low=0.0, high=self.scoring.concept_exploration_boost), 4), roll

    def _controlled_exploration_roll(self, *, query_text: str, concept_id: int) -> float:
        payload = f"{self.scoring.concept_exploration_seed}:{_normalize_text(query_text)}:{concept_id}"
        return round(random.Random(payload).random(), 4)

    def _deterministic_exploration_roll(self, *, query_text: str, concept_id: int) -> float:
        return self._controlled_exploration_roll(query_text=query_text, concept_id=concept_id)

    def _update_concept_uncertainty(self, concept: ReasoningSpatialConcept) -> float:
        concept.uncertainty = round(_clamp(1.0 - float(concept.prediction_accuracy), low=0.0, high=1.0), 4)
        return concept.uncertainty

    def _concept_abstraction_level(self, concept: ReasoningSpatialConcept) -> int:
        level = 3 if concept.is_generalized_concept else (2 if concept.is_meta_concept else 1)
        concept.abstraction_level = level
        return level

    def _child_concepts(self, concept: ReasoningSpatialConcept) -> List[ReasoningSpatialConcept]:
        child_ids = set(concept.child_concept_ids)
        if not child_ids:
            return []
        return [
            child
            for child in self.spatial_concept_memory
            if child.concept_id in child_ids
        ]

    def _concept_activation_with(
        self,
        activation: ReasoningSpatialConceptActivation,
        *,
        influence: float | None = None,
        cluster_adjustment: float | None = None,
        predicted: bool | None = None,
    ) -> ReasoningSpatialConceptActivation:
        return ReasoningSpatialConceptActivation(
            concept_id=activation.concept_id,
            similarity=activation.similarity,
            influence=round(float(activation.influence if influence is None else influence), 4),
            importance=activation.importance,
            pre_activation_influence=activation.pre_activation_influence,
            pre_reuse_count=activation.pre_reuse_count,
            post_reuse_count=activation.post_reuse_count,
            centroid_x=activation.centroid_x,
            centroid_y=activation.centroid_y,
            associated_nodes=list(activation.associated_nodes),
            associated_clusters=list(activation.associated_clusters),
            is_meta_concept=activation.is_meta_concept,
            is_generalized_concept=activation.is_generalized_concept,
            abstraction_level=activation.abstraction_level,
            concept_cluster_id=activation.concept_cluster_id,
            predicted=activation.predicted if predicted is None else predicted,
            cluster_adjustment=round(float(activation.cluster_adjustment if cluster_adjustment is None else cluster_adjustment), 4),
        )

    def _apply_meta_concept_influence(
        self,
        *,
        grid: List[List[float]],
        source_nodes: List[List[Set[int]]],
        source_clusters: List[List[Set[int]]],
        activated_concepts: Sequence[ReasoningSpatialConceptActivation],
        width: int,
        height: int,
    ) -> tuple[List[str], List[ReasoningSpatialConceptActivation]]:
        if not activated_concepts:
            return [], list(activated_concepts)
        logs: List[str] = []
        concept_lookup = {
            concept.concept_id: concept
            for concept in self.spatial_concept_memory
        }
        activation_by_id: Dict[int, ReasoningSpatialConceptActivation] = {
            activation.concept_id: activation
            for activation in activated_concepts
        }
        for activation in activated_concepts:
            if not activation.is_meta_concept:
                continue
            meta_concept = concept_lookup.get(activation.concept_id)
            if meta_concept is None:
                continue
            children = self._child_concepts(meta_concept)
            if not children:
                continue
            child_influence_total = 0.0
            for child in children:
                abstraction_multiplier = (
                    1.0 + self.scoring.concept_generalized_child_influence_weight
                    if meta_concept.is_generalized_concept
                    else 1.0
                )
                child_influence = (
                    float(activation.influence)
                    * self.scoring.concept_meta_child_influence_weight
                    * abstraction_multiplier
                    * (1.0 + (float(child.importance) * self.scoring.concept_importance_influence_weight))
                )
                if child_influence <= 0.0:
                    continue
                cells = self._region_cells(
                    center_x=int(round(child.centroid_x)),
                    center_y=int(round(child.centroid_y)),
                    radius=max(1, int(self.scoring.spatial_node_region_radius) + 1),
                    width=width,
                    height=height,
                )
                weights = self._region_weights(
                    center_x=int(round(child.centroid_x)),
                    center_y=int(round(child.centroid_y)),
                    cells=cells,
                    scale=0.85,
                )
                self._seed_region(
                    grid=grid,
                    source_nodes=source_nodes,
                    source_clusters=source_clusters,
                    cells=cells,
                    weights=weights,
                    total_activation=child_influence,
                    entry_id=child.associated_nodes[0] if child.associated_nodes else None,
                    cluster_id=child.associated_clusters[0] if child.associated_clusters else None,
                )
                child_influence_total += child_influence
                existing_activation = activation_by_id.get(child.concept_id)
                if existing_activation is not None:
                    activation_by_id[child.concept_id] = self._concept_activation_with(
                        existing_activation,
                        influence=float(existing_activation.influence) + child_influence,
                    )
                else:
                    activation_by_id[child.concept_id] = ReasoningSpatialConceptActivation(
                        concept_id=child.concept_id,
                        similarity=activation.similarity,
                        influence=round(child_influence, 4),
                        importance=round(child.importance, 4),
                        pre_activation_influence=0.0,
                        pre_reuse_count=int(child.reuse_count),
                        post_reuse_count=int(child.reuse_count),
                        centroid_x=child.centroid_x,
                        centroid_y=child.centroid_y,
                        associated_nodes=list(child.associated_nodes),
                        associated_clusters=list(child.associated_clusters),
                        is_meta_concept=child.is_meta_concept,
                        is_generalized_concept=child.is_generalized_concept,
                        abstraction_level=self._concept_abstraction_level(child),
                        concept_cluster_id=child.concept_cluster_id,
                    )
                logs.append(
                    f"meta-concept {meta_concept.concept_id} injected child concept {child.concept_id}: influence={round(child_influence, 4)} child_importance={round(child.importance, 4)} generalized={meta_concept.is_generalized_concept} parent_level={self._concept_abstraction_level(meta_concept)} child_level={self._concept_abstraction_level(child)} nodes={child.associated_nodes}"
                )
            if child_influence_total > 0.0:
                logs.append(
                    f"meta-concept {meta_concept.concept_id} child influence total={round(child_influence_total, 4)} children={meta_concept.child_concept_ids}"
                )
        adjusted_activations = sorted(
            activation_by_id.values(),
            key=lambda item: (-float(item.influence), -float(item.similarity), item.concept_id),
        )
        return logs, adjusted_activations

    def _apply_concept_cluster_competition(
        self,
        *,
        grid: List[List[float]],
        activated_concepts: Sequence[ReasoningSpatialConceptActivation],
        concept_clusters: Sequence[ReasoningSpatialConceptCluster],
        width: int,
        height: int,
    ) -> tuple[List[str], List[ReasoningSpatialConceptActivation], List[ReasoningSpatialConceptCluster]]:
        if len(concept_clusters) < 2 or not activated_concepts:
            return [], list(activated_concepts), list(concept_clusters)
        active_by_cluster: Dict[int, List[ReasoningSpatialConceptActivation]] = {}
        for activation in activated_concepts:
            if activation.concept_cluster_id < 0:
                continue
            active_by_cluster.setdefault(activation.concept_cluster_id, []).append(activation)
        if len(active_by_cluster) < 2:
            return [], list(activated_concepts), list(concept_clusters)
        cluster_lookup = {
            cluster.concept_cluster_id: cluster
            for cluster in concept_clusters
        }
        cluster_scores = {
            cluster_id: sum(float(item.influence) for item in activations)
            + float(cluster_lookup.get(cluster_id, ReasoningSpatialConceptCluster(cluster_id, [], 0.0, 0.0, 0.0, 0.0)).importance)
            for cluster_id, activations in active_by_cluster.items()
        }
        penalties: Dict[int, float] = {cluster_id: 0.0 for cluster_id in active_by_cluster}
        logs: List[str] = []
        ordered_cluster_ids = sorted(
            active_by_cluster,
            key=lambda cluster_id: (-cluster_scores.get(cluster_id, 0.0), cluster_id),
        )
        for index, stronger_id in enumerate(ordered_cluster_ids):
            stronger_cluster = cluster_lookup.get(stronger_id)
            if stronger_cluster is None:
                continue
            for weaker_id in ordered_cluster_ids[index + 1:]:
                weaker_cluster = cluster_lookup.get(weaker_id)
                if weaker_cluster is None:
                    continue
                distance = math.sqrt(
                    ((float(stronger_cluster.centroid_x) - float(weaker_cluster.centroid_x)) ** 2)
                    + ((float(stronger_cluster.centroid_y) - float(weaker_cluster.centroid_y)) ** 2)
                )
                if distance > self.scoring.concept_cluster_competition_radius:
                    continue
                score_gap = max(0.0, cluster_scores.get(stronger_id, 0.0) - cluster_scores.get(weaker_id, 0.0))
                if score_gap <= 0.0:
                    continue
                proximity_scale = 1.0 - (distance / self.scoring.concept_cluster_competition_radius)
                suppression = score_gap * proximity_scale * self.scoring.concept_cluster_competition_weight
                if suppression <= 0.0:
                    continue
                penalties[weaker_id] = penalties.get(weaker_id, 0.0) + suppression
                cells = self._region_cells(
                    center_x=int(round(weaker_cluster.centroid_x)),
                    center_y=int(round(weaker_cluster.centroid_y)),
                    radius=max(1, int(self.scoring.spatial_node_region_radius) + 2),
                    width=width,
                    height=height,
                )
                removed = self._apply_region_reduction(grid=grid, cells=cells, reduction_value=suppression)
                logs.append(
                    f"concept-cluster {stronger_id} suppressed concept-cluster {weaker_id}: distance={round(distance, 4)} score_gap={round(score_gap, 4)} suppression={round(suppression, 4)} removed={round(removed, 4)}"
                )
        adjusted_activations: List[ReasoningSpatialConceptActivation] = []
        for activation in activated_concepts:
            penalty = penalties.get(activation.concept_cluster_id, 0.0)
            cluster_size = max(1, len(active_by_cluster.get(activation.concept_cluster_id, [])))
            adjustment = -(penalty / cluster_size)
            adjusted_activations.append(
                self._concept_activation_with(
                    activation,
                    influence=max(0.0, float(activation.influence) + adjustment),
                    cluster_adjustment=adjustment,
                )
            )
        adjusted_clusters: List[ReasoningSpatialConceptCluster] = []
        for cluster in concept_clusters:
            penalty = penalties.get(cluster.concept_cluster_id, 0.0)
            adjusted_clusters.append(
                ReasoningSpatialConceptCluster(
                    concept_cluster_id=cluster.concept_cluster_id,
                    concept_ids=list(cluster.concept_ids),
                    cluster_score=cluster.cluster_score,
                    centroid_x=cluster.centroid_x,
                    centroid_y=cluster.centroid_y,
                    importance=cluster.importance,
                    shared_nodes=list(cluster.shared_nodes),
                    shared_relationships=list(cluster.shared_relationships),
                    competition_penalty=round(penalty, 4),
                    final_score=round(max(0.0, float(cluster.cluster_score) - penalty), 4),
                )
            )
        adjusted_activations.sort(key=lambda item: (-float(item.influence), -float(item.similarity), item.concept_id))
        adjusted_clusters.sort(key=lambda item: (-float(item.final_score), item.concept_cluster_id))
        return logs, adjusted_activations, adjusted_clusters

    def _mark_predicted_concept_activations(
        self,
        *,
        activated_concepts: Sequence[ReasoningSpatialConceptActivation],
        anticipated_concepts: Sequence[ReasoningConceptPrediction],
    ) -> List[ReasoningSpatialConceptActivation]:
        if not activated_concepts or not anticipated_concepts:
            return list(activated_concepts)
        predicted_ids = {prediction.concept_id for prediction in anticipated_concepts}
        for prediction in anticipated_concepts:
            predicted_ids.update(prediction.child_concept_ids)
        return [
            self._concept_activation_with(
                activation,
                predicted=activation.concept_id in predicted_ids,
            )
            for activation in activated_concepts
        ]

    def _carry_forward_concept_cluster_competition(
        self,
        *,
        concept_clusters: Sequence[ReasoningSpatialConceptCluster],
        competed_clusters: Sequence[ReasoningSpatialConceptCluster],
    ) -> List[ReasoningSpatialConceptCluster]:
        if not concept_clusters or not competed_clusters:
            return list(concept_clusters)
        competed_with_penalties = [
            cluster
            for cluster in competed_clusters
            if float(cluster.competition_penalty) > 0.0
        ]
        if not competed_with_penalties:
            return list(concept_clusters)
        carried: List[ReasoningSpatialConceptCluster] = []
        for cluster in concept_clusters:
            cluster_concepts = set(cluster.concept_ids)
            cluster_nodes = set(cluster.shared_nodes)
            competed = max(
                (
                    item
                    for item in competed_with_penalties
                    if (cluster_concepts & set(item.concept_ids))
                    or (cluster_nodes & set(item.shared_nodes))
                ),
                key=lambda item: (float(item.competition_penalty), float(item.final_score)),
                default=None,
            )
            if competed is None:
                carried.append(cluster)
                continue
            carried_penalty = float(competed.competition_penalty)
            carried_final_score = max(0.0, float(cluster.cluster_score) - carried_penalty)
            carried.append(
                ReasoningSpatialConceptCluster(
                    concept_cluster_id=cluster.concept_cluster_id,
                    concept_ids=list(cluster.concept_ids),
                    cluster_score=cluster.cluster_score,
                    centroid_x=cluster.centroid_x,
                    centroid_y=cluster.centroid_y,
                    importance=cluster.importance,
                    shared_nodes=list(cluster.shared_nodes),
                    shared_relationships=list(cluster.shared_relationships),
                    competition_penalty=carried_penalty,
                    final_score=carried_final_score,
                )
            )
        return carried

    def _apply_concept_competition(
        self,
        *,
        grid: List[List[float]],
        activated_concepts: Sequence[ReasoningSpatialConceptActivation],
        width: int,
        height: int,
    ) -> List[str]:
        if len(activated_concepts) < 2:
            return []
        logs: List[str] = []
        ordered_concepts = sorted(
            activated_concepts,
            key=lambda item: (-float(item.influence), -float(item.similarity), item.concept_id),
        )
        for index, stronger in enumerate(ordered_concepts):
            for weaker in ordered_concepts[index + 1:]:
                distance = math.sqrt(
                    ((float(stronger.centroid_x) - float(weaker.centroid_x)) ** 2)
                    + ((float(stronger.centroid_y) - float(weaker.centroid_y)) ** 2)
                )
                if distance > self.scoring.concept_competition_radius:
                    continue
                strength_gap = max(0.0, float(stronger.influence) - float(weaker.influence))
                if strength_gap <= 0.0:
                    continue
                proximity_scale = 1.0 - (distance / self.scoring.concept_competition_radius)
                reuse_scale = 1.0 + max(0, int(stronger.post_reuse_count) - int(weaker.post_reuse_count)) * 0.05
                reduction_value = strength_gap * proximity_scale * reuse_scale * self.scoring.concept_competition_weight
                cells = self._region_cells(
                    center_x=int(round(weaker.centroid_x)),
                    center_y=int(round(weaker.centroid_y)),
                    radius=max(1, int(self.scoring.spatial_node_region_radius) + 2),
                    width=width,
                    height=height,
                )
                removed = self._apply_region_reduction(grid=grid, cells=cells, reduction_value=reduction_value)
                if removed <= 0.0:
                    continue
                logs.append(
                    f"concept {stronger.concept_id} suppressed concept {weaker.concept_id}: distance={round(distance, 4)} influence_gap={round(strength_gap, 4)} removed={round(removed, 4)}"
                )
        return logs

    def _rank_regions_with_concept_influence(
        self,
        *,
        regions: Sequence[ReasoningSpatialRegion],
        activated_concepts: Sequence[ReasoningSpatialConceptActivation],
    ) -> List[ReasoningSpatialRegion]:
        if not regions or not activated_concepts:
            return list(regions)
        max_influence = max((float(concept.influence) for concept in activated_concepts), default=0.0)
        if max_influence <= 0.0:
            return list(regions)

        def region_score(region: ReasoningSpatialRegion) -> tuple[float, float, float, int]:
            concept_influence = self._region_concept_influence(
                region=region,
                activated_concepts=activated_concepts,
            )
            normalized_influence = concept_influence / max_influence
            field_score = float(region.total_activation) * (1.0 + (normalized_influence * self.scoring.concept_selection_weight))
            return (field_score, normalized_influence, float(region.dominance_value), -region.region_id)

        ranked_regions = sorted(regions, key=region_score, reverse=True)
        return [
            ReasoningSpatialRegion(
                region_id=region.region_id,
                cells=list(region.cells),
                total_activation=region.total_activation,
                peak_activation=region.peak_activation,
                centroid_x=region.centroid_x,
                centroid_y=region.centroid_y,
                source_nodes=list(region.source_nodes),
                source_clusters=list(region.source_clusters),
                competition_penalty=region.competition_penalty,
                post_competition_rank=index,
                dominance_value=region.dominance_value,
                merged_region_ids=list(region.merged_region_ids),
                dominant_node_id=region.dominant_node_id,
            )
            for index, region in enumerate(ranked_regions, start=1)
        ]

    def _region_concept_influence(
        self,
        *,
        region: ReasoningSpatialRegion,
        activated_concepts: Sequence[ReasoningSpatialConceptActivation],
    ) -> float:
        influence = 0.0
        region_nodes = set(region.source_nodes)
        region_clusters = set(region.source_clusters)
        for concept in activated_concepts:
            distance = math.sqrt(
                ((float(region.centroid_x) - float(concept.centroid_x)) ** 2)
                + ((float(region.centroid_y) - float(concept.centroid_y)) ** 2)
            )
            distance_weight = 1.0 / (1.0 + distance)
            node_overlap = len(region_nodes & set(concept.associated_nodes))
            cluster_overlap = len(region_clusters & set(concept.associated_clusters))
            overlap_weight = 1.0 + (node_overlap * 0.35) + (cluster_overlap * 0.2)
            influence += float(concept.influence) * distance_weight * overlap_weight
        return influence

    def _remember_spatial_concepts(
        self,
        *,
        query_text: str,
        query_intent: QueryIntent,
        regions: Sequence[ReasoningSpatialRegion],
        grid: Sequence[Sequence[float]],
        edges: Sequence[ReasoningGraphEdge],
        activated_concepts: Sequence[ReasoningSpatialConceptActivation],
        anticipated_concepts: Sequence[ReasoningConceptPrediction],
    ) -> tuple[
        List[str],
        List[str],
        List[str],
        List[str],
        List[str],
        List[ReasoningSpatialConceptCluster],
        List[str],
        List[str],
        List[str],
        List[str],
        List[str],
        List[str],
        List[str],
        List[str],
        List[str],
        List[str],
        List[str],
        List[str],
        List[ReasoningTemporalTransition],
        ReasoningGoal | None,
        List[ReasoningGoalPlan],
    ]:
        creation_log: List[str] = []
        evolution_log: List[str] = []
        if not regions:
            concept_clusters, concept_cluster_log = self._cluster_spatial_concepts()
            prediction_accuracy_log, importance_evolution_log = self._update_prediction_feedback(
                anticipated_concepts=anticipated_concepts,
                regions=[],
                activated_concepts=activated_concepts,
            )
            stability_log = self._stabilize_concept_memory()
            uncertainty_log = self._concept_uncertainty_log()
            abstraction_log = self._concept_abstraction_log()
            temporal_sequence_log, temporal_prediction_log, sequence = self._update_temporal_sequence(activated_concepts=activated_concepts)
            active_goal, goal_plans, goal_selection_log, goal_plan_log = self._select_goal_and_plan(
                activated_concepts=activated_concepts,
                anticipated_concepts=anticipated_concepts,
                regions=[],
            )
            goal_feedback_log = self._evaluate_goal_outcome(
                goal=active_goal,
                goal_plans=goal_plans,
                regions=[],
                activated_concepts=activated_concepts,
            )
            return creation_log, [], evolution_log, [], concept_cluster_log, concept_clusters, prediction_accuracy_log, importance_evolution_log, [], [], stability_log, uncertainty_log, abstraction_log, temporal_sequence_log, temporal_prediction_log, goal_selection_log, goal_plan_log, goal_feedback_log, self._temporal_transition_snapshot(), active_goal, goal_plans
        relationship_types = self._region_relationship_types(regions=regions, edges=edges)
        query_terms = _tokenize(query_text)
        for activation in activated_concepts:
            matched_concept = next(
                (concept for concept in self.spatial_concept_memory if concept.concept_id == activation.concept_id),
                None,
            )
            if matched_concept is None:
                continue
            matched_region = self._best_region_for_concept(
                concept_activation=activation,
                regions=regions,
            )
            if matched_region is None:
                continue
            old_activation = float(matched_concept.activation_value)
            old_centroid = (float(matched_concept.centroid_x), float(matched_concept.centroid_y))
            new_activation = (
                old_activation * self.scoring.concept_memory_activation_decay
                + float(matched_region.total_activation) * float(activation.similarity) * self.scoring.concept_memory_reinforcement_weight
            )
            matched_concept.activation_value = round(new_activation, 4)
            matched_concept.average_dominance = round(
                (float(matched_concept.average_dominance) * 0.75)
                + (float(matched_region.dominance_value) * 0.25),
                4,
            )
            matched_concept.stability_score = round(
                _clamp(
                    (float(matched_concept.stability_score) * 0.7)
                    + (((float(activation.similarity) + float(matched_region.dominance_value)) / 2.0) * 0.3),
                    low=0.0,
                    high=1.0,
                ),
                4,
            )
            matched_concept.centroid_x = round(
                (float(matched_concept.centroid_x) * 0.72) + (float(matched_region.centroid_x) * 0.28),
                2,
            )
            matched_concept.centroid_y = round(
                (float(matched_concept.centroid_y) * 0.72) + (float(matched_region.centroid_y) * 0.28),
                2,
            )
            matched_concept.associated_nodes = sorted(set(matched_concept.associated_nodes) | set(matched_region.source_nodes))
            matched_concept.associated_clusters = sorted(set(matched_concept.associated_clusters) | set(matched_region.source_clusters))
            matched_concept.relationship_types = sorted(
                set(matched_concept.relationship_types) | set(relationship_types.get(matched_region.region_id, []))
            )
            matched_concept.associated_terms = sorted(set(matched_concept.associated_terms) | set(query_terms))
            matched_concept.activation_pattern = self._merge_activation_patterns(
                left=matched_concept.activation_pattern,
                right=self._activation_pattern_for_region(region=matched_region, grid=grid),
            )
            matched_concept.last_query_intent = query_intent.value
            self._recalculate_concept_importance(matched_concept)
            evolution_log.append(
                f"concept {matched_concept.concept_id} evolved: activation={round(old_activation, 4)}->{round(matched_concept.activation_value, 4)} centroid=({round(old_centroid[0], 2)}, {round(old_centroid[1], 2)})->({round(matched_concept.centroid_x, 2)}, {round(matched_concept.centroid_y, 2)}) dominance={round(matched_concept.average_dominance, 4)} stability={round(matched_concept.stability_score, 4)} importance={round(matched_concept.importance, 4)} nodes={matched_concept.associated_nodes} relationships={matched_concept.relationship_types}"
            )
        remembered_regions = [
            region
            for region in regions
            if region.dominance_value >= self.scoring.spatial_region_dominance_threshold
        ] or list(regions[:1])
        for region in remembered_regions[:2]:
            if self._region_already_represented(region=region):
                continue
            concept = ReasoningSpatialConcept(
                concept_id=self._allocate_spatial_concept_id(),
                centroid_x=float(region.centroid_x),
                centroid_y=float(region.centroid_y),
                activation_value=float(region.total_activation),
                associated_nodes=list(region.source_nodes),
                associated_clusters=list(region.source_clusters),
                relationship_types=relationship_types.get(region.region_id, []),
                associated_terms=list(query_terms),
                activation_pattern=self._activation_pattern_for_region(region=region, grid=grid),
                reuse_count=0,
                last_query_intent=query_intent.value,
                average_dominance=round(float(region.dominance_value), 4),
                stability_score=round(
                    _clamp(
                        (float(region.dominance_value) * 0.65)
                        + (min(1.0, len(region.cells) / 24.0) * 0.35),
                        low=0.0,
                        high=1.0,
                    ),
                    4,
                ),
                importance_memory_adjustment=0.0,
                abstraction_level=1,
            )
            self._recalculate_concept_importance(concept)
            self.spatial_concept_memory.append(concept)
            creation_log.append(
                f"created concept {concept.concept_id}: centroid=({round(concept.centroid_x, 2)}, {round(concept.centroid_y, 2)}) activation={round(concept.activation_value, 4)} dominance={round(concept.average_dominance, 4)} stability={round(concept.stability_score, 4)} importance={round(concept.importance, 4)} nodes={concept.associated_nodes} clusters={concept.associated_clusters}"
            )
        hierarchy_log = self._create_or_update_meta_concepts(
            activated_concepts=activated_concepts,
            regions=regions,
            grid=grid,
            query_terms=query_terms,
            query_intent=query_intent,
        )
        merge_log = self._merge_spatial_concepts()
        cross_domain_link_log = self._update_cross_domain_links()
        generalization_log = self._create_generalized_meta_concepts(
            query_terms=query_terms,
            query_intent=query_intent,
        )
        concept_clusters, concept_cluster_log = self._cluster_spatial_concepts()
        for concept in self.spatial_concept_memory:
            self._recalculate_concept_importance(concept)
        prediction_accuracy_log, importance_evolution_log = self._update_prediction_feedback(
            anticipated_concepts=anticipated_concepts,
            regions=regions,
            activated_concepts=activated_concepts,
        )
        stability_log = self._stabilize_concept_memory()
        uncertainty_log = self._concept_uncertainty_log()
        abstraction_log = self._concept_abstraction_log()
        temporal_sequence_log, temporal_prediction_log, sequence = self._update_temporal_sequence(activated_concepts=activated_concepts)
        active_goal, goal_plans, goal_selection_log, goal_plan_log = self._select_goal_and_plan(
            activated_concepts=activated_concepts,
            anticipated_concepts=anticipated_concepts,
            regions=regions,
        )
        goal_feedback_log = self._evaluate_goal_outcome(
            goal=active_goal,
            goal_plans=goal_plans,
            regions=regions,
            activated_concepts=activated_concepts,
        )
        self.spatial_concept_memory.sort(
            key=lambda item: (-float(item.importance), -float(item.activation_value), -int(item.reuse_count), item.concept_id)
        )
        limit = max(1, int(self.scoring.concept_memory_limit))
        if len(self.spatial_concept_memory) > limit:
            del self.spatial_concept_memory[limit:]
            concept_clusters, concept_cluster_log = self._cluster_spatial_concepts()
        self._save_spatial_concept_memory()
        return creation_log, merge_log, evolution_log, hierarchy_log, concept_cluster_log, concept_clusters, prediction_accuracy_log, importance_evolution_log, cross_domain_link_log, generalization_log, stability_log, uncertainty_log, abstraction_log, temporal_sequence_log, temporal_prediction_log, goal_selection_log, goal_plan_log, goal_feedback_log, self._temporal_transition_snapshot(), active_goal, goal_plans

    def _recalculate_concept_importance(self, concept: ReasoningSpatialConcept) -> float:
        self._update_concept_uncertainty(concept)
        self._concept_abstraction_level(concept)
        reuse_score = min(1.0, max(0, int(concept.reuse_count)) / 8.0)
        importance = (
            reuse_score * self.scoring.concept_importance_reuse_weight
            + float(concept.average_dominance) * self.scoring.concept_importance_dominance_weight
            + float(concept.stability_score) * self.scoring.concept_importance_stability_weight
            + float(concept.importance_memory_adjustment)
        )
        concept.importance = round(_clamp(importance, low=0.0, high=self.scoring.concept_importance_cap), 4)
        return concept.importance

    def _update_prediction_feedback(
        self,
        *,
        anticipated_concepts: Sequence[ReasoningConceptPrediction],
        regions: Sequence[ReasoningSpatialRegion],
        activated_concepts: Sequence[ReasoningSpatialConceptActivation],
    ) -> tuple[List[str], List[str]]:
        if not self.spatial_concept_memory:
            return [], []
        concept_lookup = {
            concept.concept_id: concept
            for concept in self.spatial_concept_memory
        }
        predicted_ids = {prediction.concept_id for prediction in anticipated_concepts}
        activated_lookup = {
            activation.concept_id: activation
            for activation in activated_concepts
        }
        prediction_logs: List[str] = []
        importance_logs: List[str] = []
        dominant_region = regions[0] if regions else None
        for concept in self.spatial_concept_memory:
            was_predicted = concept.concept_id in predicted_ids
            prediction_success = False
            if was_predicted:
                prediction_success = self._concept_prediction_matches_result(
                    concept=concept,
                    dominant_region=dominant_region,
                    activated_lookup=activated_lookup,
                )
                concept.prediction_total_count += 1
                if prediction_success:
                    concept.prediction_correct_count += 1
                concept.prediction_accuracy = round(
                    concept.prediction_correct_count / float(max(1, concept.prediction_total_count)),
                    4,
                )
                self._update_concept_uncertainty(concept)
                target_confidence = 1.0 if prediction_success else 0.0
                concept.prediction_confidence = round(
                    (float(concept.prediction_confidence) * 0.72) + (target_confidence * 0.28),
                    4,
                )
                prediction_logs.append(
                    f"concept {concept.concept_id} prediction {'correct' if prediction_success else 'missed'}: total={concept.prediction_total_count} correct={concept.prediction_correct_count} accuracy={round(concept.prediction_accuracy, 4)} uncertainty={round(concept.uncertainty, 4)} confidence={round(concept.prediction_confidence, 4)}"
                )
            old_adjustment = float(concept.importance_memory_adjustment)
            old_importance = float(concept.importance)
            reinforcement = self.scoring.concept_importance_success_reinforcement if prediction_success else 0.0
            decay = self.scoring.concept_importance_decay_factor
            if was_predicted or concept.concept_id in activated_lookup:
                concept.importance_memory_adjustment = round(
                    _clamp(old_adjustment + reinforcement - decay, low=-0.25, high=0.35),
                    4,
                )
                self._recalculate_concept_importance(concept)
                importance_logs.append(
                    f"concept {concept.concept_id} importance evolved: {round(old_importance, 4)}->{round(concept.importance, 4)} adjustment={round(old_adjustment, 4)}->{round(concept.importance_memory_adjustment, 4)} reinforcement={round(reinforcement, 4)} decay={round(decay, 4)} predicted={was_predicted}"
                )
        return prediction_logs, importance_logs

    def _concept_prediction_matches_result(
        self,
        *,
        concept: ReasoningSpatialConcept,
        dominant_region: ReasoningSpatialRegion | None,
        activated_lookup: Dict[int, ReasoningSpatialConceptActivation],
    ) -> bool:
        activation = activated_lookup.get(concept.concept_id)
        if activation is not None and float(activation.influence) >= self.scoring.concept_prediction_success_threshold:
            return True
        if dominant_region is None:
            return False
        region_nodes = set(dominant_region.source_nodes)
        region_clusters = set(dominant_region.source_clusters)
        concept_nodes = set(concept.associated_nodes)
        concept_clusters = set(concept.associated_clusters)
        if region_nodes & concept_nodes:
            return True
        if region_clusters & concept_clusters:
            return True
        for child in self._child_concepts(concept):
            if region_nodes & set(child.associated_nodes):
                return True
            if region_clusters & set(child.associated_clusters):
                return True
        distance = math.sqrt(
            ((float(dominant_region.centroid_x) - float(concept.centroid_x)) ** 2)
            + ((float(dominant_region.centroid_y) - float(concept.centroid_y)) ** 2)
        )
        return distance <= self.scoring.concept_memory_merge_distance

    def _stabilize_concept_memory(self) -> List[str]:
        logs: List[str] = []
        cap = float(self.scoring.concept_importance_cap)
        for concept in self.spatial_concept_memory:
            raw_importance = self._raw_concept_importance(concept)
            old_importance = float(concept.importance)
            self._recalculate_concept_importance(concept)
            if raw_importance > cap:
                logs.append(
                    f"concept {concept.concept_id} importance capped: raw={round(raw_importance, 4)} cap={round(cap, 4)} final={round(concept.importance, 4)} previous={round(old_importance, 4)}"
                )
        return logs

    def _raw_concept_importance(self, concept: ReasoningSpatialConcept) -> float:
        reuse_score = min(1.0, max(0, int(concept.reuse_count)) / 8.0)
        return (
            reuse_score * self.scoring.concept_importance_reuse_weight
            + float(concept.average_dominance) * self.scoring.concept_importance_dominance_weight
            + float(concept.stability_score) * self.scoring.concept_importance_stability_weight
            + float(concept.importance_memory_adjustment)
        )

    def _concept_uncertainty_log(self) -> List[str]:
        logs: List[str] = []
        for concept in sorted(self.spatial_concept_memory, key=lambda item: item.concept_id):
            uncertainty = self._update_concept_uncertainty(concept)
            logs.append(
                f"concept {concept.concept_id} uncertainty={round(uncertainty, 4)} prediction_accuracy={round(concept.prediction_accuracy, 4)} total_predictions={concept.prediction_total_count}"
            )
        return logs

    def _concept_abstraction_log(self) -> List[str]:
        logs = [
            "level 0: graph nodes",
            "level 1: spatial concepts",
            "level 2: meta-concepts",
            "level 3: generalized concepts",
        ]
        for concept in sorted(self.spatial_concept_memory, key=lambda item: (self._concept_abstraction_level(item), item.concept_id)):
            logs.append(
                f"concept {concept.concept_id}: level={self._concept_abstraction_level(concept)} meta={concept.is_meta_concept} generalized={concept.is_generalized_concept} children={concept.child_concept_ids} parents={concept.parent_concept_ids}"
            )
        return logs

    def _update_temporal_sequence(
        self,
        *,
        activated_concepts: Sequence[ReasoningSpatialConceptActivation],
    ) -> tuple[List[str], List[str], List[int]]:
        sequence: List[int] = []
        seen: Set[int] = set()
        ordered = sorted(
            activated_concepts,
            key=lambda item: (-float(item.influence), -float(item.similarity), -float(item.importance), item.concept_id),
        )
        for activation in ordered:
            if activation.concept_id in seen:
                continue
            seen.add(activation.concept_id)
            sequence.append(int(activation.concept_id))
        sequence_log: List[str] = []
        prediction_log: List[str] = []
        if not sequence:
            return ["temporal sequence: []"], ["no active concept available for temporal prediction"], []
        sequence_log.append(
            "temporal sequence: "
            + " -> ".join(f"concept {concept_id}" for concept_id in sequence)
        )
        for source_id, target_id in zip(sequence, sequence[1:]):
            transition = self._transition_for(source_id, target_id)
            transition.frequency += 1
            sequence_log.append(
                f"transition {source_id}->{target_id} frequency={transition.frequency}"
            )
        self._refresh_temporal_transition_confidence()
        current_concept_id = sequence[0]
        prediction = self._predict_next_temporal_concept(current_concept_id)
        if prediction is None:
            prediction_log.append(f"concept {current_concept_id} has no learned next-concept transition yet")
        else:
            prediction_log.append(
                f"concept {current_concept_id} predicts next concept {prediction.target_concept_id}: confidence={round(prediction.confidence, 4)} reliability={round(prediction.reliability, 4)} frequency={prediction.frequency}"
            )
        return sequence_log, prediction_log, sequence

    def _transition_for(self, source_id: int, target_id: int) -> ReasoningTemporalTransition:
        key = (int(source_id), int(target_id))
        transition = self.temporal_transition_memory.get(key)
        if transition is None:
            transition = ReasoningTemporalTransition(source_concept_id=key[0], target_concept_id=key[1])
            self.temporal_transition_memory[key] = transition
        return transition

    def _refresh_temporal_transition_confidence(self) -> None:
        outgoing_totals: Dict[int, int] = {}
        for transition in self.temporal_transition_memory.values():
            outgoing_totals[transition.source_concept_id] = outgoing_totals.get(transition.source_concept_id, 0) + max(0, int(transition.frequency))
        for transition in self.temporal_transition_memory.values():
            outgoing_total = max(1, outgoing_totals.get(transition.source_concept_id, 0))
            transition.confidence = round(_clamp(float(transition.frequency) / float(outgoing_total), low=0.0, high=1.0), 4)
            outcomes = int(transition.success_count) + int(transition.failure_count)
            if outcomes:
                transition.reliability = round(_clamp(float(transition.success_count) / float(outcomes), low=0.0, high=1.0), 4)

    def _predict_next_temporal_concept(self, concept_id: int) -> ReasoningTemporalTransition | None:
        candidates = [
            transition
            for transition in self.temporal_transition_memory.values()
            if transition.source_concept_id == int(concept_id)
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (float(item.confidence), float(item.reliability), int(item.frequency), -int(item.target_concept_id)),
        )

    def _select_goal_and_plan(
        self,
        *,
        activated_concepts: Sequence[ReasoningSpatialConceptActivation],
        anticipated_concepts: Sequence[ReasoningConceptPrediction],
        regions: Sequence[ReasoningSpatialRegion],
    ) -> tuple[ReasoningGoal | None, List[ReasoningGoalPlan], List[str], List[str]]:
        target_id, priority, reason = self._select_goal_target(
            activated_concepts=activated_concepts,
            anticipated_concepts=anticipated_concepts,
        )
        if target_id is None:
            return None, [], ["no goal selected: no active or anticipated concept"], []
        goal = ReasoningGoal(
            target_concept_id=target_id,
            priority=round(priority, 4),
            success_condition="dominant_region_matches_target_concept",
            reason=reason,
        )
        selection_log = [
            f"selected goal target concept {goal.target_concept_id}: priority={round(goal.priority, 4)} condition={goal.success_condition} reason={goal.reason}"
        ]
        plans = self._generate_goal_plans(goal=goal, activated_concepts=activated_concepts)
        plan_log = [
            (
                f"plan target={plan.target_concept_id} path={plan.path} score={round(plan.plan_score, 4)} "
                f"importance={round(plan.cumulative_importance, 4)} edge_strength={round(plan.semantic_edge_strength, 4)} "
                f"temporal_likelihood={round(plan.temporal_likelihood, 4)} "
                f"confidence={round(plan.prediction_confidence, 4)} reliability={round(plan.path_reliability, 4)} "
                f"past_success={round(plan.past_success_rate, 4)} simulated_success={round(plan.simulated_success_probability, 4)} "
                f"stability={round(plan.simulated_stability, 4)} uncertainty={round(plan.simulated_uncertainty, 4)} "
                f"predicted_outcome={plan.predicted_outcome}"
            )
            for plan in plans
        ]
        if plans:
            selected_plan = plans[0]
            plan_log.append(
                f"selected simulated plan path={selected_plan.path}: {selected_plan.selection_reason}"
            )
        if not plan_log:
            plan_log.append(f"no temporal plan available for goal concept {goal.target_concept_id}")
        return goal, plans, selection_log, plan_log

    def _select_goal_target(
        self,
        *,
        activated_concepts: Sequence[ReasoningSpatialConceptActivation],
        anticipated_concepts: Sequence[ReasoningConceptPrediction],
    ) -> tuple[int | None, float, str]:
        if anticipated_concepts:
            prediction = max(
                anticipated_concepts,
                key=lambda item: (
                    float(item.predicted_score) * (1.0 + float(item.prediction_confidence)),
                    float(item.importance),
                    -int(item.concept_id),
                ),
            )
            priority = _clamp(
                (float(prediction.predicted_score) * 0.65)
                + (float(prediction.prediction_confidence) * 0.25)
                + (float(prediction.importance) * 0.1),
                low=0.0,
                high=1.0,
            )
            return int(prediction.concept_id), round(priority, 4), "highest anticipatory prediction"
        if activated_concepts:
            activation = max(
                activated_concepts,
                key=lambda item: (float(item.influence), float(item.importance), -int(item.concept_id)),
            )
            priority = _clamp(
                (min(1.0, float(activation.influence)) * 0.55)
                + (float(activation.importance) * 0.25)
                + (float(activation.similarity) * 0.2),
                low=0.0,
                high=1.0,
            )
            return int(activation.concept_id), round(priority, 4), "strongest activated concept"
        return None, 0.0, ""

    def _generate_goal_plans(
        self,
        *,
        goal: ReasoningGoal,
        activated_concepts: Sequence[ReasoningSpatialConceptActivation],
    ) -> List[ReasoningGoalPlan]:
        concept_lookup = {concept.concept_id: concept for concept in self.spatial_concept_memory}
        activated_lookup = {int(activation.concept_id): activation for activation in activated_concepts}
        sources = [
            int(activation.concept_id)
            for activation in sorted(activated_concepts, key=lambda item: (-float(item.influence), item.concept_id))
        ]
        if goal.target_concept_id not in sources:
            sources.append(goal.target_concept_id)
        plans: Dict[tuple[int, ...], ReasoningGoalPlan] = {}
        for source_id in sources[:5]:
            candidate_paths = self._temporal_candidate_paths_to_goal(source_id=source_id, target_id=goal.target_concept_id)
            if not candidate_paths:
                candidate_paths = [[source_id]] if source_id == goal.target_concept_id else [[source_id, goal.target_concept_id]]
            for path in candidate_paths:
                if not path:
                    continue
                path_key = tuple(path)
                if path_key in plans:
                    continue
                confidence, reliability, temporal_likelihood, past_success = self._temporal_path_metrics(path)
                importance_score = (
                    sum(
                        float(concept_lookup.get(concept_id, ReasoningSpatialConcept(concept_id, 0.0, 0.0, 0.0)).importance)
                        for concept_id in path
                    )
                    / float(max(1, len(path)))
                )
                simulated_success, simulated_stability, simulated_uncertainty, predicted_outcome = self._simulate_goal_plan(
                    path=path,
                    goal=goal,
                    concept_lookup=concept_lookup,
                    activated_lookup=activated_lookup,
                )
                semantic_edge_strength = self._temporal_path_semantic_edge_strength(path=path, concept_lookup=concept_lookup)
                simulation_trust = self._simulation_trust()
                importance_weight = self._simulation_effective_factor_weight(
                    "concept_importance",
                    self.scoring.goal_plan_importance_weight,
                )
                edge_weight = self._simulation_effective_factor_weight(
                    "edge_strength",
                    self.scoring.goal_plan_semantic_edge_weight,
                )
                temporal_likelihood_weight = self._simulation_effective_factor_weight(
                    "temporal_likelihood",
                    self.scoring.goal_plan_temporal_likelihood_weight,
                )
                past_success_weight = self._simulation_effective_factor_weight(
                    "past_success",
                    self.scoring.goal_plan_past_success_weight,
                )
                base_score = (
                    float(goal.priority) * self.scoring.goal_priority_weight
                    + importance_score * importance_weight
                    + semantic_edge_strength * edge_weight
                    + confidence * self.scoring.goal_plan_confidence_weight
                    + reliability * self.scoring.goal_plan_reliability_weight
                    + temporal_likelihood * temporal_likelihood_weight
                    + past_success * past_success_weight
                )
                simulation_score = (
                    simulated_success * self.scoring.goal_plan_simulation_success_weight
                    + simulated_stability * self.scoring.goal_plan_simulation_stability_weight
                    - simulated_uncertainty * self.scoring.goal_plan_simulation_uncertainty_weight
                )
                plan_score = (
                    base_score
                    + simulation_score * simulation_trust
                )
                plans[path_key] = ReasoningGoalPlan(
                    target_concept_id=goal.target_concept_id,
                    path=list(path),
                    plan_score=round(max(0.0, plan_score), 4),
                    prediction_confidence=round(confidence, 4),
                    path_reliability=round(reliability, 4),
                    cumulative_importance=round(importance_score, 4),
                    semantic_edge_strength=round(semantic_edge_strength, 4),
                    temporal_likelihood=round(temporal_likelihood, 4),
                    past_success_rate=round(past_success, 4),
                    simulated_success_probability=round(simulated_success, 4),
                    simulated_stability=round(simulated_stability, 4),
                    simulated_uncertainty=round(simulated_uncertainty, 4),
                    predicted_outcome=predicted_outcome,
                    selection_reason=(
                        f"predicted_success={round(simulated_success, 4)} "
                        f"uncertainty={round(simulated_uncertainty, 4)} "
                        f"historical_reliability={round(reliability, 4)} "
                        f"edge_strength={round(semantic_edge_strength, 4)} "
                        f"simulation_trust={round(simulation_trust, 4)} "
                        f"temporal_likelihood={round(temporal_likelihood, 4)}"
                    ),
                )
        ordered_plans = sorted(
            plans.values(),
            key=lambda item: (
                -float(item.plan_score),
                -float(item.simulated_success_probability),
                float(item.simulated_uncertainty),
                -float(item.path_reliability),
                item.path,
            ),
        )
        return ordered_plans[: max(1, int(self.scoring.temporal_plan_candidate_limit))]

    def _temporal_path_to_goal(self, *, source_id: int, target_id: int) -> List[int]:
        candidate_paths = self._temporal_candidate_paths_to_goal(source_id=source_id, target_id=target_id)
        return candidate_paths[0] if candidate_paths else []

    def _temporal_candidate_paths_to_goal(self, *, source_id: int, target_id: int) -> List[List[int]]:
        if source_id == target_id:
            return [[int(source_id)]]
        max_nodes = max(3, int(self.scoring.temporal_path_max_concepts))
        queue: deque[List[int]] = deque([[int(source_id)]])
        candidate_paths: List[List[int]] = []
        candidate_keys: Set[tuple[int, ...]] = set()
        while queue:
            path = queue.popleft()
            if len(path) >= max_nodes:
                continue
            outgoing = sorted(
                [
                    transition
                    for transition in self.temporal_transition_memory.values()
                    if transition.source_concept_id == path[-1]
                ],
                key=lambda item: (-float(item.confidence), -float(item.reliability), -int(item.frequency), item.target_concept_id),
            )
            for transition in outgoing:
                next_id = int(transition.target_concept_id)
                if next_id in path:
                    continue
                next_path = list(path) + [next_id]
                if next_id == int(target_id):
                    path_key = tuple(next_path)
                    if path_key not in candidate_keys:
                        candidate_keys.add(path_key)
                        candidate_paths.append(next_path)
                    continue
                if len(next_path) < max_nodes:
                    queue.append(next_path)
        if not candidate_paths:
            direct_transition = self.temporal_transition_memory.get((int(source_id), int(target_id)))
            if direct_transition is not None:
                candidate_paths.append([int(source_id), int(target_id)])
        return sorted(
            candidate_paths,
            key=lambda item: (
                -self._temporal_path_metrics(item)[0],
                -self._temporal_path_metrics(item)[1],
                len(item),
                item,
            ),
        )[: max(1, int(self.scoring.temporal_plan_candidate_limit))]

    def _temporal_path_stats(self, path: Sequence[int]) -> tuple[float, float]:
        confidence, reliability, _, _ = self._temporal_path_metrics(path)
        return confidence, reliability

    def _temporal_path_metrics(self, path: Sequence[int]) -> tuple[float, float, float, float]:
        if len(path) < 2:
            return 1.0, 1.0, 1.0, 1.0
        transitions = [
            self.temporal_transition_memory.get((int(source_id), int(target_id)))
            for source_id, target_id in zip(path, path[1:])
        ]
        confidence_values = [float(transition.confidence) for transition in transitions if transition is not None]
        reliability_values = [float(transition.reliability) for transition in transitions if transition is not None]
        success_values = [
            float(transition.success_count) / float(max(1, int(transition.success_count) + int(transition.failure_count)))
            for transition in transitions
            if transition is not None and (int(transition.success_count) + int(transition.failure_count)) > 0
        ]
        confidence = sum(confidence_values) / float(len(confidence_values)) if confidence_values else 0.0
        reliability = sum(reliability_values) / float(len(reliability_values)) if reliability_values else 0.35
        temporal_likelihood = 1.0
        if confidence_values:
            for value in confidence_values:
                temporal_likelihood *= _clamp(value, low=0.0, high=1.0)
            temporal_likelihood = temporal_likelihood ** (1.0 / float(len(confidence_values)))
        else:
            temporal_likelihood = 0.0
        past_success = sum(success_values) / float(len(success_values)) if success_values else reliability
        return round(confidence, 4), round(reliability, 4), round(temporal_likelihood, 4), round(past_success, 4)

    def _simulation_trust(self) -> float:
        if int(self.simulation_model_state.total_predictions) <= 0:
            return 0.65
        return round(
            _clamp(
                0.35
                + (float(self.simulation_model_state.simulation_accuracy) * 0.55)
                + (float(self.simulation_model_state.calibrated_confidence) * 0.1)
                - (float(self.simulation_model_state.average_prediction_error) * 0.15),
                low=0.2,
                high=1.0,
            ),
            4,
        )

    def _simulation_effective_factor_weight(self, factor_name: str, base_weight: float) -> float:
        adjustment = float(self.simulation_model_state.weight_adjustments.get(factor_name, 0.0))
        return round(
            _clamp(
                float(base_weight) + adjustment,
                low=0.0,
                high=max(1.0, float(base_weight) + self.scoring.goal_simulation_weight_adjustment_cap),
            ),
            4,
        )

    def _temporal_path_semantic_edge_strength(
        self,
        *,
        path: Sequence[int],
        concept_lookup: Dict[int, ReasoningSpatialConcept],
    ) -> float:
        if len(path) < 2:
            return 1.0
        strengths = [
            self._concept_semantic_edge_strength(source_id=int(source_id), target_id=int(target_id), concept_lookup=concept_lookup)
            for source_id, target_id in zip(path, path[1:])
        ]
        return round(sum(strengths) / float(max(1, len(strengths))), 4)

    def _concept_semantic_edge_strength(
        self,
        *,
        source_id: int,
        target_id: int,
        concept_lookup: Dict[int, ReasoningSpatialConcept],
    ) -> float:
        source = concept_lookup.get(int(source_id))
        target = concept_lookup.get(int(target_id))
        transition = self.temporal_transition_memory.get((int(source_id), int(target_id)))
        transition_strength = 0.0
        if transition is not None:
            transition_strength = (float(transition.confidence) * 0.55) + (float(transition.reliability) * 0.45)
        if source is None or target is None:
            return round(_clamp(transition_strength, low=0.0, high=1.0), 4)
        source_terms = set(source.associated_terms)
        target_terms = set(target.associated_terms)
        term_overlap = len(source_terms & target_terms) / float(max(1, len(source_terms | target_terms)))
        node_overlap = len(set(source.associated_nodes) & set(target.associated_nodes)) / float(
            max(1, len(set(source.associated_nodes) | set(target.associated_nodes)))
        )
        hierarchy_link = (
            1.0
            if int(target_id) in source.child_concept_ids
            or int(target_id) in source.parent_concept_ids
            or int(source_id) in target.child_concept_ids
            or int(source_id) in target.parent_concept_ids
            else 0.0
        )
        cross_domain_strength = max(
            float(source.cross_domain_links.get(int(target_id), 0.0)),
            float(target.cross_domain_links.get(int(source_id), 0.0)),
        )
        coactivation_strength = min(
            1.0,
            max(
                float(source.coactivation_counts.get(int(target_id), 0)),
                float(target.coactivation_counts.get(int(source_id), 0)),
            )
            / 4.0,
        )
        semantic_strength = (
            transition_strength * 0.45
            + term_overlap * 0.18
            + node_overlap * 0.14
            + hierarchy_link * 0.1
            + cross_domain_strength * 0.08
            + coactivation_strength * 0.05
        )
        return round(_clamp(semantic_strength, low=0.0, high=1.0), 4)

    def _simulate_goal_plan(
        self,
        *,
        path: Sequence[int],
        goal: ReasoningGoal,
        concept_lookup: Dict[int, ReasoningSpatialConcept],
        activated_lookup: Dict[int, ReasoningSpatialConceptActivation],
    ) -> tuple[float, float, float, str]:
        if not path:
            return 0.0, 0.0, 1.0, "miss"
        first_id = int(path[0])
        first_concept = concept_lookup.get(first_id)
        first_activation = activated_lookup.get(first_id)
        activation_value = _clamp(
            (float(first_activation.influence) if first_activation is not None else 0.0) * 0.55
            + (float(first_concept.importance) if first_concept is not None else 0.0) * 0.45,
            low=0.0,
            high=1.0,
        )
        stability_values: List[float] = []
        uncertainty_values: List[float] = []
        for source_id, target_id in zip(path, path[1:]):
            transition = self.temporal_transition_memory.get((int(source_id), int(target_id)))
            target_concept = concept_lookup.get(int(target_id))
            transition_confidence = float(transition.confidence) if transition is not None else 0.18
            transition_reliability = float(transition.reliability) if transition is not None else 0.25
            simulation_accuracy = float(transition.simulation_accuracy) if transition is not None else 0.0
            target_importance = float(target_concept.importance) if target_concept is not None else 0.0
            edge_drive = (
                transition_confidence * 0.34
                + transition_reliability * 0.34
                + simulation_accuracy * 0.12
                + target_importance * 0.2
            )
            activation_value = _clamp((activation_value * 0.6) + (edge_drive * 0.4), low=0.0, high=1.0)
            stability_values.append(_clamp((transition_reliability * 0.7) + (simulation_accuracy * 0.3), low=0.0, high=1.0))
            uncertainty_values.append(_clamp(1.0 - ((transition_confidence * 0.45) + (transition_reliability * 0.45) + (simulation_accuracy * 0.1)), low=0.0, high=1.0))
        target_concept = concept_lookup.get(int(goal.target_concept_id))
        target_activation = activated_lookup.get(int(goal.target_concept_id))
        target_signal = (
            (float(target_concept.importance) if target_concept is not None else 0.0) * 0.45
            + (float(target_activation.influence) if target_activation is not None else 0.0) * 0.35
            + float(goal.priority) * 0.2
        )
        simulated_success = _clamp((activation_value * 0.65) + (target_signal * 0.35), low=0.0, high=1.0)
        simulated_stability = (
            sum(stability_values) / float(len(stability_values))
            if stability_values
            else _clamp((float(goal.priority) + (float(target_concept.importance) if target_concept is not None else 0.0)) / 2.0, low=0.0, high=1.0)
        )
        simulated_uncertainty = (
            sum(uncertainty_values) / float(len(uncertainty_values))
            if uncertainty_values
            else _clamp(1.0 - simulated_success, low=0.0, high=1.0)
        )
        predicted_outcome = "success" if simulated_success >= self.scoring.concept_prediction_success_threshold else "miss"
        return round(simulated_success, 4), round(simulated_stability, 4), round(simulated_uncertainty, 4), predicted_outcome

    def _evaluate_goal_outcome(
        self,
        *,
        goal: ReasoningGoal | None,
        goal_plans: Sequence[ReasoningGoalPlan],
        regions: Sequence[ReasoningSpatialRegion],
        activated_concepts: Sequence[ReasoningSpatialConceptActivation],
    ) -> List[str]:
        if goal is None:
            return ["goal feedback skipped: no active goal"]
        target = next((concept for concept in self.spatial_concept_memory if concept.concept_id == goal.target_concept_id), None)
        dominant_region = regions[0] if regions else None
        target_activation = next((activation for activation in activated_concepts if activation.concept_id == goal.target_concept_id), None)
        success = False
        if target_activation is not None and float(target_activation.influence) >= self.scoring.concept_prediction_success_threshold:
            success = True
        if target is not None and dominant_region is not None:
            success = success or self._concept_prediction_matches_result(
                concept=target,
                dominant_region=dominant_region,
                activated_lookup={activation.concept_id: activation for activation in activated_concepts},
            )
        success_score = 1.0 if success else 0.0
        logs = [
            f"goal target concept {goal.target_concept_id} {'succeeded' if success else 'missed'}: success_score={round(success_score, 4)} condition={goal.success_condition}"
        ]
        if goal_plans:
            best_plan = max(
                goal_plans,
                key=lambda item: (
                    float(item.plan_score),
                    float(item.simulated_success_probability),
                    -float(item.simulated_uncertainty),
                    float(item.path_reliability),
                    [-concept_id for concept_id in item.path],
                ),
            )
            predicted_success = best_plan.predicted_outcome == "success"
            prediction_matched = predicted_success == success
            prediction_error = abs(float(best_plan.simulated_success_probability) - success_score)
            logs.append(
                f"evaluated best plan path={best_plan.path}: score={round(best_plan.plan_score, 4)} reliability={round(best_plan.path_reliability, 4)} predicted={best_plan.predicted_outcome} actual={'success' if success else 'miss'} simulation_match={prediction_matched} prediction_error={round(prediction_error, 4)}"
            )
            logs.extend(
                self._update_simulation_model_from_outcome(
                    plan=best_plan,
                    actual_success=success_score,
                    prediction_matched=prediction_matched,
                )
            )
            for source_id, target_id in zip(best_plan.path, best_plan.path[1:]):
                transition = self._transition_for(source_id, target_id)
                if success:
                    transition.success_count += 1
                    transition.reliability = round(_clamp(float(transition.reliability) + self.scoring.temporal_transition_success_reinforcement, low=0.0, high=1.0), 4)
                else:
                    transition.failure_count += 1
                    transition.reliability = round(_clamp(float(transition.reliability) - self.scoring.temporal_transition_failure_decay, low=0.0, high=1.0), 4)
                transition.simulation_total_count += 1
                if prediction_matched:
                    transition.simulation_correct_count += 1
                transition.simulation_accuracy = round(
                    float(transition.simulation_correct_count) / float(max(1, int(transition.simulation_total_count))),
                    4,
                )
                previous_average_error = float(transition.average_prediction_error)
                transition.last_prediction_error = round(prediction_error, 4)
                transition.average_prediction_error = round(
                    ((previous_average_error * max(0, int(transition.simulation_total_count) - 1)) + prediction_error)
                    / float(max(1, int(transition.simulation_total_count))),
                    4,
                )
                logs.append(
                    f"transition {source_id}->{target_id} {'reinforced' if success else 'decayed'}: reliability={round(transition.reliability, 4)} simulation_accuracy={round(transition.simulation_accuracy, 4)} prediction_error={round(transition.last_prediction_error, 4)} average_prediction_error={round(transition.average_prediction_error, 4)} successes={transition.success_count} failures={transition.failure_count} simulation_correct={transition.simulation_correct_count}/{transition.simulation_total_count}"
                )
        self._refresh_temporal_transition_confidence()
        return logs

    def _update_simulation_model_from_outcome(
        self,
        *,
        plan: ReasoningGoalPlan,
        actual_success: float,
        prediction_matched: bool,
    ) -> List[str]:
        predicted_success = _clamp(float(plan.simulated_success_probability), low=0.0, high=1.0)
        actual_success = _clamp(float(actual_success), low=0.0, high=1.0)
        prediction_error = abs(predicted_success - actual_success)
        direction = "calibrated"
        if predicted_success > actual_success:
            direction = "optimistic"
        elif predicted_success < actual_success:
            direction = "pessimistic"

        state = self.simulation_model_state
        previous_total = int(state.total_predictions)
        state.total_predictions = previous_total + 1
        state.last_prediction_error = round(prediction_error, 4)
        state.last_prediction_direction = direction
        if prediction_matched:
            state.correct_predictions += 1
        state.simulation_accuracy = round(
            float(state.correct_predictions) / float(max(1, int(state.total_predictions))),
            4,
        )
        state.average_prediction_error = round(
            ((float(state.average_prediction_error) * previous_total) + prediction_error)
            / float(max(1, int(state.total_predictions))),
            4,
        )
        state.calibrated_confidence = round(_clamp(1.0 - float(state.average_prediction_error), low=0.0, high=1.0), 4)

        factor_values = {
            "concept_importance": float(plan.cumulative_importance),
            "edge_strength": float(plan.semantic_edge_strength),
            "temporal_likelihood": float(plan.temporal_likelihood),
            "past_success": float(plan.past_success_rate),
        }
        factor_weights = {
            "concept_importance": self._simulation_effective_factor_weight(
                "concept_importance",
                self.scoring.goal_plan_importance_weight,
            ),
            "edge_strength": self._simulation_effective_factor_weight(
                "edge_strength",
                self.scoring.goal_plan_semantic_edge_weight,
            ),
            "temporal_likelihood": self._simulation_effective_factor_weight(
                "temporal_likelihood",
                self.scoring.goal_plan_temporal_likelihood_weight,
            ),
            "past_success": self._simulation_effective_factor_weight(
                "past_success",
                self.scoring.goal_plan_past_success_weight,
            ),
        }
        raw_contributions = {
            key: max(0.0, float(factor_values[key]) * float(factor_weights[key]))
            for key in factor_values
        }
        total_contribution = sum(raw_contributions.values()) or 1.0
        normalized_contributions = {
            key: float(value) / float(total_contribution)
            for key, value in raw_contributions.items()
        }

        adjustment_sign = 0.0
        if direction == "optimistic":
            adjustment_sign = -1.0
        elif direction == "pessimistic":
            adjustment_sign = 1.0
        adjustment_log_parts: List[str] = []
        for factor_name, contribution in sorted(normalized_contributions.items()):
            old_adjustment = float(state.weight_adjustments.get(factor_name, 0.0))
            delta = adjustment_sign * float(self.scoring.goal_simulation_weight_adjustment_rate) * prediction_error * contribution
            new_adjustment = round(
                _clamp(
                    old_adjustment + delta,
                    low=-float(self.scoring.goal_simulation_weight_adjustment_cap),
                    high=float(self.scoring.goal_simulation_weight_adjustment_cap),
                ),
                4,
            )
            state.weight_adjustments[factor_name] = new_adjustment
            adjustment_log_parts.append(
                f"{factor_name}: contribution={round(contribution, 4)} adjustment={round(old_adjustment, 4)}->{new_adjustment}"
            )

        if state.simulation_accuracy < self.scoring.goal_simulation_accuracy_exploration_threshold:
            state.adaptive_exploration_boost = round(
                _clamp(
                    (float(self.scoring.goal_simulation_accuracy_exploration_threshold) - float(state.simulation_accuracy)) * 0.25,
                    low=0.0,
                    high=float(self.scoring.goal_simulation_exploration_boost_cap),
                ),
                4,
            )
        else:
            state.adaptive_exploration_boost = 0.0

        return [
            (
                f"simulation calibration: predicted_success={round(predicted_success, 4)} "
                f"actual_success={round(actual_success, 4)} prediction_error={round(prediction_error, 4)} "
                f"direction={direction} simulation_accuracy={round(state.simulation_accuracy, 4)} "
                f"average_prediction_error={round(state.average_prediction_error, 4)} "
                f"calibrated_confidence={round(state.calibrated_confidence, 4)}"
            ),
            "simulation error attribution: " + "; ".join(adjustment_log_parts),
            (
                f"adaptive exploration: boost={round(state.adaptive_exploration_boost, 4)} "
                f"threshold={round(self.scoring.goal_simulation_accuracy_exploration_threshold, 4)} "
                f"effective_exploration_rate={round(_clamp(self.scoring.concept_exploration_rate + state.adaptive_exploration_boost, low=0.0, high=1.0), 4)}"
            ),
        ]

    def _temporal_transition_snapshot(self) -> List[ReasoningTemporalTransition]:
        return sorted(
            self.temporal_transition_memory.values(),
            key=lambda item: (-int(item.frequency), -float(item.confidence), item.source_concept_id, item.target_concept_id),
        )

    def _concept_importance_log(self) -> List[str]:
        logs: List[str] = []
        for concept in sorted(
            self.spatial_concept_memory,
            key=lambda item: (-float(item.importance), item.concept_id),
        ):
            logs.append(
                f"concept {concept.concept_id}: importance={round(concept.importance, 4)} adjustment={round(concept.importance_memory_adjustment, 4)} reuse={concept.reuse_count} average_dominance={round(concept.average_dominance, 4)} stability={round(concept.stability_score, 4)} prediction_accuracy={round(concept.prediction_accuracy, 4)} uncertainty={round(concept.uncertainty, 4)} prediction_confidence={round(concept.prediction_confidence, 4)} meta={concept.is_meta_concept} generalized={concept.is_generalized_concept} abstraction_level={self._concept_abstraction_level(concept)} cluster={concept.concept_cluster_id}"
            )
        return logs

    def _create_or_update_meta_concepts(
        self,
        *,
        activated_concepts: Sequence[ReasoningSpatialConceptActivation],
        regions: Sequence[ReasoningSpatialRegion],
        grid: Sequence[Sequence[float]],
        query_terms: Sequence[str],
        query_intent: QueryIntent,
    ) -> List[str]:
        if len(activated_concepts) < 2:
            return []
        logs: List[str] = []
        concept_lookup = {
            concept.concept_id: concept
            for concept in self.spatial_concept_memory
        }
        active_ids = [
            activation.concept_id
            for activation in sorted(activated_concepts, key=lambda item: item.concept_id)
            if activation.concept_id in concept_lookup and not concept_lookup[activation.concept_id].is_meta_concept
        ]
        for index, left_id in enumerate(active_ids):
            left = concept_lookup[left_id]
            for right_id in active_ids[index + 1:]:
                right = concept_lookup[right_id]
                left.coactivation_counts[right_id] = int(left.coactivation_counts.get(right_id, 0)) + 1
                right.coactivation_counts[left_id] = int(right.coactivation_counts.get(left_id, 0)) + 1
                coactivation_count = min(
                    int(left.coactivation_counts.get(right_id, 0)),
                    int(right.coactivation_counts.get(left_id, 0)),
                )
                overlap = self._concept_overlap(left, right)
                distance = math.sqrt(
                    ((float(left.centroid_x) - float(right.centroid_x)) ** 2)
                    + ((float(left.centroid_y) - float(right.centroid_y)) ** 2)
                )
                if coactivation_count < max(1, int(self.scoring.concept_meta_min_coactivation_count)):
                    continue
                if overlap < self.scoring.concept_meta_overlap_threshold and distance > (self.scoring.concept_memory_merge_distance * 2.0):
                    continue
                child_ids = sorted([left_id, right_id])
                existing_meta = next(
                    (
                        concept
                        for concept in self.spatial_concept_memory
                        if concept.is_meta_concept and sorted(concept.child_concept_ids) == child_ids
                    ),
                    None,
                )
                if existing_meta is not None:
                    old_importance = float(existing_meta.importance)
                    existing_meta.activation_value = round(
                        (float(existing_meta.activation_value) * 0.84)
                        + ((float(left.activation_value) + float(right.activation_value)) * 0.08),
                        4,
                    )
                    existing_meta.associated_terms = sorted(set(existing_meta.associated_terms) | set(query_terms))
                    existing_meta.last_query_intent = query_intent.value
                    self._recalculate_concept_importance(existing_meta)
                    logs.append(
                        f"meta-concept {existing_meta.concept_id} reinforced children={child_ids} coactivation_count={coactivation_count} importance={round(old_importance, 4)}->{round(existing_meta.importance, 4)}"
                    )
                    continue
                total_activation = max(0.0001, float(left.activation_value) + float(right.activation_value))
                centroid_x = (
                    (float(left.centroid_x) * float(left.activation_value))
                    + (float(right.centroid_x) * float(right.activation_value))
                ) / total_activation
                centroid_y = (
                    (float(left.centroid_y) * float(left.activation_value))
                    + (float(right.centroid_y) * float(right.activation_value))
                ) / total_activation
                dominant_region = regions[0] if regions else None
                activation_pattern = self._merge_activation_patterns(
                    left=left.activation_pattern,
                    right=right.activation_pattern,
                )
                if dominant_region is not None:
                    activation_pattern = self._merge_activation_patterns(
                        left=activation_pattern,
                        right=self._activation_pattern_for_region(region=dominant_region, grid=grid),
                    )
                meta_concept = ReasoningSpatialConcept(
                    concept_id=self._allocate_spatial_concept_id(),
                    centroid_x=round(centroid_x, 2),
                    centroid_y=round(centroid_y, 2),
                    activation_value=round(total_activation * self.scoring.concept_meta_activation_weight, 4),
                    associated_nodes=sorted(set(left.associated_nodes) | set(right.associated_nodes)),
                    associated_clusters=sorted(set(left.associated_clusters) | set(right.associated_clusters)),
                    relationship_types=sorted(set(left.relationship_types) | set(right.relationship_types)),
                    associated_terms=sorted(set(left.associated_terms) | set(right.associated_terms) | set(query_terms)),
                    activation_pattern=activation_pattern,
                    reuse_count=0,
                    last_query_intent=query_intent.value,
                    average_dominance=round((float(left.average_dominance) + float(right.average_dominance)) / 2.0, 4),
                    stability_score=round((float(left.stability_score) + float(right.stability_score)) / 2.0, 4),
                    importance_memory_adjustment=round(
                        (float(left.importance_memory_adjustment) + float(right.importance_memory_adjustment)) / 2.0,
                        4,
                    ),
                    is_meta_concept=True,
                    child_concept_ids=child_ids,
                    abstraction_level=2,
                )
                self._recalculate_concept_importance(meta_concept)
                self.spatial_concept_memory.append(meta_concept)
                left.parent_concept_ids = sorted(set(left.parent_concept_ids) | {meta_concept.concept_id})
                right.parent_concept_ids = sorted(set(right.parent_concept_ids) | {meta_concept.concept_id})
                logs.append(
                    f"created meta-concept {meta_concept.concept_id}: children={child_ids} coactivation_count={coactivation_count} overlap={round(overlap, 4)} distance={round(distance, 4)} importance={round(meta_concept.importance, 4)}"
                )
        return logs

    def _update_cross_domain_links(self) -> List[str]:
        if len(self.spatial_concept_memory) < 2:
            return []
        logs: List[str] = []
        concepts = sorted(self.spatial_concept_memory, key=lambda item: item.concept_id)
        for index, left in enumerate(concepts):
            left_terms = set(left.associated_terms)
            if not left_terms:
                continue
            for right in concepts[index + 1:]:
                right_terms = set(right.associated_terms)
                if not right_terms:
                    continue
                shared_terms = sorted(left_terms & right_terms)
                overlap = len(shared_terms) / float(max(1, min(len(left_terms), len(right_terms))))
                if overlap < self.scoring.concept_cross_domain_overlap_threshold:
                    continue
                strength = round(_clamp(overlap, low=0.0, high=0.35), 4)
                old_left = float(left.cross_domain_links.get(right.concept_id, 0.0))
                old_right = float(right.cross_domain_links.get(left.concept_id, 0.0))
                if strength <= old_left and strength <= old_right:
                    continue
                left.cross_domain_links[right.concept_id] = max(old_left, strength)
                right.cross_domain_links[left.concept_id] = max(old_right, strength)
                logs.append(
                    f"concept {left.concept_id} linked concept {right.concept_id}: strength={strength} shared_terms={shared_terms}"
                )
        return logs

    def _create_generalized_meta_concepts(
        self,
        *,
        query_terms: Sequence[str],
        query_intent: QueryIntent,
    ) -> List[str]:
        if len(self.spatial_concept_memory) < 2:
            return []
        logs: List[str] = []
        concept_lookup = {
            concept.concept_id: concept
            for concept in self.spatial_concept_memory
        }
        candidates: List[tuple[float, int, int, List[str]]] = []
        seen_pairs: Set[tuple[int, int]] = set()
        for left in self.spatial_concept_memory:
            if left.is_generalized_concept:
                continue
            for right_id, strength in sorted(left.cross_domain_links.items()):
                if left.concept_id >= int(right_id):
                    continue
                right = concept_lookup.get(int(right_id))
                if right is None or right.is_generalized_concept:
                    continue
                shared_terms = sorted(set(left.associated_terms) & set(right.associated_terms))
                if float(strength) < self.scoring.concept_generalization_overlap_threshold and len(shared_terms) < 2:
                    continue
                seen_pairs.add((left.concept_id, int(right_id)))
                candidates.append((float(strength), left.concept_id, int(right_id), shared_terms))
        concepts = sorted(
            [concept for concept in self.spatial_concept_memory if not concept.is_generalized_concept],
            key=lambda item: item.concept_id,
        )
        for index, left in enumerate(concepts):
            for right in concepts[index + 1:]:
                pair_key = (left.concept_id, right.concept_id)
                if pair_key in seen_pairs:
                    continue
                shared_terms = sorted(set(left.associated_terms) & set(right.associated_terms))
                shared_relationships = sorted(set(left.relationship_types) & set(right.relationship_types))
                shared_nodes = sorted(set(left.associated_nodes) & set(right.associated_nodes))
                term_score = len(shared_terms) / float(max(1, min(len(left.associated_terms), len(right.associated_terms))))
                relationship_score = len(shared_relationships) / float(max(1, min(len(left.relationship_types), len(right.relationship_types))))
                node_score = min(0.25, len(shared_nodes) * 0.08)
                structural_score = round(_clamp((term_score * 0.55) + (relationship_score * 0.35) + node_score, low=0.0, high=1.0), 4)
                if (
                    structural_score < self.scoring.concept_generalization_overlap_threshold
                    and len(shared_relationships) < 2
                    and len(shared_terms) < 3
                ):
                    continue
                candidates.append((structural_score, left.concept_id, right.concept_id, sorted(set(shared_terms) | set(shared_relationships))))
                seen_pairs.add(pair_key)
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        for strength, left_id, right_id, shared_terms in candidates[:2]:
            left = concept_lookup.get(left_id)
            right = concept_lookup.get(right_id)
            if left is None or right is None:
                continue
            child_ids = sorted([left_id, right_id])
            existing = next(
                (
                    concept
                    for concept in self.spatial_concept_memory
                    if concept.is_meta_concept and sorted(concept.child_concept_ids) == child_ids
                ),
                None,
            )
            if existing is not None:
                if not existing.is_generalized_concept:
                    existing.is_generalized_concept = True
                    existing.abstraction_level = 3
                    existing.associated_terms = sorted(set(existing.associated_terms) | set(self._generalized_terms(left, right, query_terms)))
                    self._recalculate_concept_importance(existing)
                    logs.append(
                        f"meta-concept {existing.concept_id} generalized: children={child_ids} strength={round(strength, 4)} terms={existing.associated_terms}"
                    )
                continue
            total_activation = max(0.0001, float(left.activation_value) + float(right.activation_value))
            generalized_terms = self._generalized_terms(left, right, query_terms)
            generalized = ReasoningSpatialConcept(
                concept_id=self._allocate_spatial_concept_id(),
                centroid_x=round(((float(left.centroid_x) + float(right.centroid_x)) / 2.0), 2),
                centroid_y=round(((float(left.centroid_y) + float(right.centroid_y)) / 2.0), 2),
                activation_value=round(total_activation * self.scoring.concept_generalization_activation_weight, 4),
                associated_nodes=sorted(set(left.associated_nodes) | set(right.associated_nodes)),
                associated_clusters=sorted(set(left.associated_clusters) | set(right.associated_clusters)),
                relationship_types=sorted(set(left.relationship_types) | set(right.relationship_types)),
                associated_terms=generalized_terms,
                activation_pattern=self._merge_activation_patterns(left=left.activation_pattern, right=right.activation_pattern),
                reuse_count=0,
                last_query_intent=query_intent.value,
                average_dominance=round((float(left.average_dominance) + float(right.average_dominance)) / 2.0, 4),
                stability_score=round((float(left.stability_score) + float(right.stability_score)) / 2.0, 4),
                importance_memory_adjustment=round(
                    (float(left.importance_memory_adjustment) + float(right.importance_memory_adjustment)) / 2.0,
                    4,
                ),
                is_meta_concept=True,
                child_concept_ids=child_ids,
                is_generalized_concept=True,
                abstraction_level=3,
            )
            self._recalculate_concept_importance(generalized)
            self.spatial_concept_memory.append(generalized)
            left.parent_concept_ids = sorted(set(left.parent_concept_ids) | {generalized.concept_id})
            right.parent_concept_ids = sorted(set(right.parent_concept_ids) | {generalized.concept_id})
            logs.append(
                f"created generalized meta-concept {generalized.concept_id}: children={child_ids} strength={round(strength, 4)} terms={generalized_terms}"
            )
        return logs

    def _generalized_terms(
        self,
        left: ReasoningSpatialConcept,
        right: ReasoningSpatialConcept,
        query_terms: Sequence[str],
    ) -> List[str]:
        terms = set(left.associated_terms) | set(right.associated_terms) | set(query_terms)
        shared_terms = set(left.associated_terms) & set(right.associated_terms)
        generalized_terms = set(shared_terms) | set(query_terms)
        if terms & {"sky", "atmosphere", "atmospheric", "scatter", "scattering", "light", "wavelengths"}:
            generalized_terms.update({"atmospheric", "optics"})
        if terms & {"color", "blue", "green", "red"}:
            generalized_terms.update({"color", "appearance"})
        if not generalized_terms:
            generalized_terms = set(sorted(terms)[:6])
        return sorted(generalized_terms)

    def _cluster_spatial_concepts(self) -> tuple[List[ReasoningSpatialConceptCluster], List[str]]:
        if not self.spatial_concept_memory:
            return [], []
        concepts = sorted(self.spatial_concept_memory, key=lambda item: item.concept_id)
        adjacency: Dict[int, Set[int]] = {concept.concept_id: set() for concept in concepts}
        concept_lookup = {concept.concept_id: concept for concept in concepts}
        for index, left in enumerate(concepts):
            for right in concepts[index + 1:]:
                relationship_score = self._concept_relation_score(left, right)
                if relationship_score < self.scoring.concept_cluster_similarity_threshold:
                    continue
                adjacency[left.concept_id].add(right.concept_id)
                adjacency[right.concept_id].add(left.concept_id)
        visited: Set[int] = set()
        concept_clusters: List[ReasoningSpatialConceptCluster] = []
        logs: List[str] = []
        cluster_id = 1
        for concept in concepts:
            if concept.concept_id in visited:
                continue
            stack = [concept.concept_id]
            visited.add(concept.concept_id)
            component_ids: List[int] = []
            while stack:
                current_id = stack.pop()
                component_ids.append(current_id)
                for neighbor_id in sorted(adjacency.get(current_id, set()), reverse=True):
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)
                    stack.append(neighbor_id)
            component = [concept_lookup[concept_id] for concept_id in sorted(component_ids)]
            total_weight = sum(max(0.0001, float(item.importance) + (float(item.activation_value) * 0.02)) for item in component)
            centroid_x = sum(float(item.centroid_x) * max(0.0001, float(item.importance) + (float(item.activation_value) * 0.02)) for item in component) / total_weight
            centroid_y = sum(float(item.centroid_y) * max(0.0001, float(item.importance) + (float(item.activation_value) * 0.02)) for item in component) / total_weight
            shared_nodes = sorted({node_id for item in component for node_id in item.associated_nodes})
            shared_relationships = sorted({relationship for item in component for relationship in item.relationship_types})
            cluster_importance = round(sum(float(item.importance) for item in component), 4)
            cluster_score = round(cluster_importance + (len(component) * 0.05), 4)
            for item in component:
                item.concept_cluster_id = cluster_id
            concept_cluster = ReasoningSpatialConceptCluster(
                concept_cluster_id=cluster_id,
                concept_ids=[item.concept_id for item in component],
                cluster_score=cluster_score,
                centroid_x=round(centroid_x, 2),
                centroid_y=round(centroid_y, 2),
                importance=cluster_importance,
                shared_nodes=shared_nodes,
                shared_relationships=shared_relationships,
                competition_penalty=0.0,
                final_score=cluster_score,
            )
            concept_clusters.append(concept_cluster)
            logs.append(
                f"concept-cluster {cluster_id}: concepts={concept_cluster.concept_ids} score={round(cluster_score, 4)} importance={round(cluster_importance, 4)} nodes={shared_nodes} relationships={shared_relationships}"
            )
            cluster_id += 1
        concept_clusters.sort(key=lambda item: (-item.cluster_score, item.concept_cluster_id))
        return concept_clusters, logs

    def _concept_relation_score(self, left: ReasoningSpatialConcept, right: ReasoningSpatialConcept) -> float:
        overlap = self._concept_overlap(left, right)
        hierarchy_score = 0.0
        if left.concept_id in right.child_concept_ids or right.concept_id in left.child_concept_ids:
            hierarchy_score = 1.0
        if set(left.parent_concept_ids) & set(right.parent_concept_ids):
            hierarchy_score = max(hierarchy_score, 0.6)
        cross_domain_score = max(
            float(left.cross_domain_links.get(right.concept_id, 0.0)),
            float(right.cross_domain_links.get(left.concept_id, 0.0)),
        ) * 0.5
        return max(overlap, hierarchy_score, cross_domain_score)

    def _region_relationship_types(
        self,
        *,
        regions: Sequence[ReasoningSpatialRegion],
        edges: Sequence[ReasoningGraphEdge],
    ) -> Dict[int, List[str]]:
        relationship_map: Dict[int, Set[str]] = {region.region_id: set() for region in regions}
        for region in regions:
            region_nodes = set(region.source_nodes)
            if not region_nodes:
                continue
            for edge in edges:
                if edge.source_entry_id in region_nodes or edge.target_entry_id in region_nodes:
                    relationship_map[region.region_id].add(edge.relationship_type.value)
        return {
            region_id: sorted(relationship_types)
            for region_id, relationship_types in relationship_map.items()
        }

    def _activation_pattern_for_region(
        self,
        *,
        region: ReasoningSpatialRegion,
        grid: Sequence[Sequence[float]],
    ) -> List[Tuple[int, int, float]]:
        pattern: List[Tuple[int, int, float]] = []
        for x, y in region.cells:
            if y < 0 or y >= len(grid) or x < 0 or (grid and x >= len(grid[0])):
                continue
            value = float(grid[y][x])
            if value <= 0.0:
                continue
            pattern.append((x, y, round(value, 4)))
        pattern.sort(key=lambda item: (-item[2], item[1], item[0]))
        return pattern[:24]

    def _merge_activation_patterns(
        self,
        *,
        left: Sequence[Tuple[int, int, float]],
        right: Sequence[Tuple[int, int, float]],
    ) -> List[Tuple[int, int, float]]:
        merged: Dict[Tuple[int, int], float] = {}
        for x, y, value in list(left) + list(right):
            key = (int(x), int(y))
            merged[key] = max(float(merged.get(key, 0.0)), float(value))
        pattern = [(x, y, round(value, 4)) for (x, y), value in merged.items()]
        pattern.sort(key=lambda item: (-item[2], item[1], item[0]))
        return pattern[:24]

    def _best_region_for_concept(
        self,
        *,
        concept_activation: ReasoningSpatialConceptActivation,
        regions: Sequence[ReasoningSpatialRegion],
    ) -> ReasoningSpatialRegion | None:
        if not regions:
            return None
        concept_nodes = set(concept_activation.associated_nodes)
        concept_clusters = set(concept_activation.associated_clusters)

        def score(region: ReasoningSpatialRegion) -> tuple[float, float, float, int]:
            distance = math.sqrt(
                ((float(region.centroid_x) - float(concept_activation.centroid_x)) ** 2)
                + ((float(region.centroid_y) - float(concept_activation.centroid_y)) ** 2)
            )
            overlap = len(set(region.source_nodes) & concept_nodes) + (len(set(region.source_clusters) & concept_clusters) * 0.5)
            return (
                overlap + (1.0 / (1.0 + distance)),
                float(region.dominance_value),
                float(region.total_activation),
                -region.region_id,
            )

        return max(regions, key=score)

    def _region_already_represented(self, *, region: ReasoningSpatialRegion) -> bool:
        region_nodes = set(region.source_nodes)
        region_clusters = set(region.source_clusters)
        for concept in self.spatial_concept_memory:
            node_overlap = 0.0
            if region_nodes or concept.associated_nodes:
                node_overlap = len(region_nodes & set(concept.associated_nodes)) / float(max(1, len(region_nodes | set(concept.associated_nodes))))
            cluster_overlap = 0.0
            if region_clusters or concept.associated_clusters:
                cluster_overlap = len(region_clusters & set(concept.associated_clusters)) / float(max(1, len(region_clusters | set(concept.associated_clusters))))
            distance = math.sqrt(
                ((float(region.centroid_x) - float(concept.centroid_x)) ** 2)
                + ((float(region.centroid_y) - float(concept.centroid_y)) ** 2)
            )
            if distance <= self.scoring.concept_memory_merge_distance and ((node_overlap * 0.65) + (cluster_overlap * 0.35)) >= self.scoring.concept_memory_merge_overlap_threshold:
                return True
        return False

    def _merge_spatial_concepts(self) -> List[str]:
        logs: List[str] = []
        concepts = sorted(self.spatial_concept_memory, key=lambda item: item.concept_id)
        merged_ids: Set[int] = set()
        merged_concepts: List[ReasoningSpatialConcept] = []
        for concept in concepts:
            if concept.concept_id in merged_ids:
                continue
            base = concept
            for candidate in concepts:
                if candidate.concept_id <= base.concept_id or candidate.concept_id in merged_ids:
                    continue
                if base.is_meta_concept != candidate.is_meta_concept:
                    continue
                if base.concept_id in candidate.child_concept_ids or candidate.concept_id in base.child_concept_ids:
                    continue
                if set(base.parent_concept_ids) & {candidate.concept_id} or set(candidate.parent_concept_ids) & {base.concept_id}:
                    continue
                distance = math.sqrt(
                    ((float(base.centroid_x) - float(candidate.centroid_x)) ** 2)
                    + ((float(base.centroid_y) - float(candidate.centroid_y)) ** 2)
                )
                if distance > self.scoring.concept_memory_merge_distance:
                    continue
                overlap = self._concept_overlap(base, candidate)
                if overlap < self.scoring.concept_memory_merge_overlap_threshold:
                    continue
                logs.append(
                    f"merged concept {candidate.concept_id} into concept {base.concept_id}: distance={round(distance, 4)} overlap={round(overlap, 4)}"
                )
                base = self._combine_concepts(left=base, right=candidate)
                merged_ids.add(candidate.concept_id)
            merged_concepts.append(base)
            merged_ids.add(base.concept_id)
        self.spatial_concept_memory = merged_concepts
        return logs

    def _concept_overlap(self, left: ReasoningSpatialConcept, right: ReasoningSpatialConcept) -> float:
        left_nodes = set(left.associated_nodes)
        right_nodes = set(right.associated_nodes)
        left_clusters = set(left.associated_clusters)
        right_clusters = set(right.associated_clusters)
        left_relationships = set(left.relationship_types)
        right_relationships = set(right.relationship_types)
        left_terms = set(left.associated_terms)
        right_terms = set(right.associated_terms)
        node_score = 0.0 if not (left_nodes or right_nodes) else len(left_nodes & right_nodes) / float(len(left_nodes | right_nodes))
        cluster_score = 0.0 if not (left_clusters or right_clusters) else len(left_clusters & right_clusters) / float(len(left_clusters | right_clusters))
        relationship_score = 0.0 if not (left_relationships or right_relationships) else len(left_relationships & right_relationships) / float(len(left_relationships | right_relationships))
        term_score = 0.0 if not (left_terms or right_terms) else len(left_terms & right_terms) / float(len(left_terms | right_terms))
        return _clamp((node_score * 0.45) + (cluster_score * 0.25) + (relationship_score * 0.15) + (term_score * 0.15), low=0.0, high=1.0)

    def _combine_concepts(
        self,
        *,
        left: ReasoningSpatialConcept,
        right: ReasoningSpatialConcept,
    ) -> ReasoningSpatialConcept:
        total_activation = max(0.0001, float(left.activation_value) + float(right.activation_value))
        centroid_x = (
            (float(left.centroid_x) * float(left.activation_value))
            + (float(right.centroid_x) * float(right.activation_value))
        ) / total_activation
        centroid_y = (
            (float(left.centroid_y) * float(left.activation_value))
            + (float(right.centroid_y) * float(right.activation_value))
        ) / total_activation
        combined_pattern = list(left.activation_pattern) + list(right.activation_pattern)
        combined_pattern.sort(key=lambda item: (-float(item[2]), item[1], item[0]))
        coactivation_counts: Dict[int, int] = dict(left.coactivation_counts)
        for concept_id, count in right.coactivation_counts.items():
            coactivation_counts[concept_id] = max(int(coactivation_counts.get(concept_id, 0)), int(count))
        cross_domain_links: Dict[int, float] = dict(left.cross_domain_links)
        for concept_id, strength in right.cross_domain_links.items():
            if concept_id in {left.concept_id, right.concept_id}:
                continue
            cross_domain_links[concept_id] = max(float(cross_domain_links.get(concept_id, 0.0)), float(strength))
        combined = ReasoningSpatialConcept(
            concept_id=min(left.concept_id, right.concept_id),
            centroid_x=round(centroid_x, 2),
            centroid_y=round(centroid_y, 2),
            activation_value=round(total_activation, 4),
            associated_nodes=sorted(set(left.associated_nodes) | set(right.associated_nodes)),
            associated_clusters=sorted(set(left.associated_clusters) | set(right.associated_clusters)),
            relationship_types=sorted(set(left.relationship_types) | set(right.relationship_types)),
            associated_terms=sorted(set(left.associated_terms) | set(right.associated_terms)),
            activation_pattern=combined_pattern[:24],
            reuse_count=int(left.reuse_count) + int(right.reuse_count),
            last_query_intent=left.last_query_intent or right.last_query_intent,
            average_dominance=round((float(left.average_dominance) + float(right.average_dominance)) / 2.0, 4),
            stability_score=round((float(left.stability_score) + float(right.stability_score)) / 2.0, 4),
            importance_memory_adjustment=round(
                (float(left.importance_memory_adjustment) + float(right.importance_memory_adjustment)) / 2.0,
                4,
            ),
            is_meta_concept=left.is_meta_concept and right.is_meta_concept,
            child_concept_ids=sorted(set(left.child_concept_ids) | set(right.child_concept_ids)),
            parent_concept_ids=sorted(set(left.parent_concept_ids) | set(right.parent_concept_ids)),
            concept_cluster_id=min(left.concept_cluster_id, right.concept_cluster_id),
            coactivation_counts=coactivation_counts,
            prediction_total_count=int(left.prediction_total_count) + int(right.prediction_total_count),
            prediction_correct_count=int(left.prediction_correct_count) + int(right.prediction_correct_count),
            cross_domain_links=cross_domain_links,
            is_generalized_concept=left.is_generalized_concept or right.is_generalized_concept,
            abstraction_level=3 if (left.is_generalized_concept or right.is_generalized_concept) else (2 if (left.is_meta_concept or right.is_meta_concept) else 1),
        )
        combined.prediction_accuracy = round(
            combined.prediction_correct_count / float(max(1, combined.prediction_total_count)),
            4,
        )
        combined.prediction_confidence = round(
            (float(left.prediction_confidence) + float(right.prediction_confidence)) / 2.0,
            4,
        )
        self._update_concept_uncertainty(combined)
        self._recalculate_concept_importance(combined)
        return combined

    def _copy_grid(self, grid: Sequence[Sequence[float]]) -> List[List[float]]:
        return [[float(cell) for cell in row] for row in grid]

    def _copy_source_grid(self, grid: Sequence[Sequence[Set[int]]]) -> List[List[Set[int]]]:
        return [[set(cell) for cell in row] for row in grid]

    def _grid_total(self, grid: Sequence[Sequence[float]]) -> float:
        return sum(float(cell) for row in grid for cell in row)

    def _propagate_spatial_activation(
        self,
        *,
        grid: Sequence[Sequence[float]],
        node_sources: Sequence[Sequence[Set[int]]],
        cluster_sources: Sequence[Sequence[Set[int]]],
    ) -> tuple[List[List[float]], List[List[Set[int]]], List[List[Set[int]]]]:
        height = len(grid)
        width = len(grid[0]) if height else 0
        new_grid = self._copy_grid(grid)
        new_node_sources = self._copy_source_grid(node_sources)
        new_cluster_sources = self._copy_source_grid(cluster_sources)
        for y in range(height):
            for x in range(width):
                neighbor_total = 0.0
                neighbor_count = 0
                influential_node_sources: Set[int] = set()
                influential_cluster_sources: Set[int] = set()
                for delta_y in (-1, 0, 1):
                    for delta_x in (-1, 0, 1):
                        if delta_x == 0 and delta_y == 0:
                            continue
                        neighbor_x = x + delta_x
                        neighbor_y = y + delta_y
                        if neighbor_x < 0 or neighbor_y < 0 or neighbor_x >= width or neighbor_y >= height:
                            continue
                        neighbor_value = float(grid[neighbor_y][neighbor_x])
                        if neighbor_value <= 0.0:
                            continue
                        weight = 0.7 if delta_x != 0 and delta_y != 0 else 1.0
                        neighbor_total += neighbor_value * weight
                        neighbor_count += 1
                        if neighbor_value >= float(grid[y][x]) * 0.5:
                            influential_node_sources.update(node_sources[neighbor_y][neighbor_x])
                            influential_cluster_sources.update(cluster_sources[neighbor_y][neighbor_x])
                if neighbor_count <= 0:
                    continue
                propagated = (neighbor_total / float(neighbor_count)) * self.scoring.spatial_decay_factor
                new_grid[y][x] = float(grid[y][x]) + propagated
                new_node_sources[y][x].update(influential_node_sources)
                new_cluster_sources[y][x].update(influential_cluster_sources)
        return new_grid, new_node_sources, new_cluster_sources

    def _expanded_region_cells(
        self,
        *,
        cells: Sequence[Tuple[int, int]],
        width: int,
        height: int,
        extra_radius: int,
    ) -> List[Tuple[int, int]]:
        expanded: Set[Tuple[int, int]] = set()
        for x, y in cells:
            for delta_y in range(-extra_radius, extra_radius + 1):
                for delta_x in range(-extra_radius, extra_radius + 1):
                    neighbor_x = x + delta_x
                    neighbor_y = y + delta_y
                    if neighbor_x < 0 or neighbor_y < 0 or neighbor_x >= width or neighbor_y >= height:
                        continue
                    expanded.add((neighbor_x, neighbor_y))
        return sorted(expanded, key=lambda item: (item[1], item[0]))

    def _apply_spatial_inhibition(
        self,
        *,
        grid: List[List[float]],
        node_regions: Dict[int, List[Tuple[int, int]]],
        cluster_regions: Dict[int, List[Tuple[int, int]]],
        nodes: Dict[int, ReasoningGraphNode],
        edges: Sequence[ReasoningGraphEdge],
        clusters: Sequence[ReasoningGraphCluster],
        cluster_relationships: Sequence[ReasoningGraphClusterRelationship],
        width: int,
        height: int,
    ) -> List[str]:
        inhibition_log: List[str] = []
        for edge in edges:
            if edge.relationship_type != RelationshipType.CONTRADICTS:
                continue
            source_node = nodes.get(edge.source_entry_id)
            target_node = nodes.get(edge.target_entry_id)
            if source_node is None or target_node is None:
                continue
            source_strength = float(source_node.activation_value) * float(edge.strength) * self.scoring.spatial_inhibition_factor
            target_strength = float(target_node.activation_value) * float(edge.strength) * self.scoring.spatial_inhibition_factor
            target_cells = self._expanded_region_cells(
                cells=node_regions.get(edge.target_entry_id, []),
                width=width,
                height=height,
                extra_radius=1,
            )
            source_cells = self._expanded_region_cells(
                cells=node_regions.get(edge.source_entry_id, []),
                width=width,
                height=height,
                extra_radius=1,
            )
            removed_from_target = self._apply_region_reduction(grid=grid, cells=target_cells, reduction_value=source_strength)
            removed_from_source = self._apply_region_reduction(grid=grid, cells=source_cells, reduction_value=target_strength)
            inhibition_log.append(
                f"node {edge.source_entry_id} contradicted node {edge.target_entry_id}: removed={round(removed_from_target, 4)}"
            )
            inhibition_log.append(
                f"node {edge.target_entry_id} contradicted node {edge.source_entry_id}: removed={round(removed_from_source, 4)}"
            )
        cluster_lookup = {cluster.cluster_id: cluster for cluster in clusters}
        for relationship in cluster_relationships:
            if relationship.relationship_type != RelationshipType.CONTRADICTS:
                continue
            source_cluster = cluster_lookup.get(relationship.source_cluster_id)
            if source_cluster is None:
                continue
            target_cells = self._expanded_region_cells(
                cells=cluster_regions.get(relationship.target_cluster_id, []),
                width=width,
                height=height,
                extra_radius=1,
            )
            cluster_reduction = (
                float(source_cluster.activation_value)
                * float(relationship.strength)
                * self.scoring.spatial_inhibition_factor
                * 0.6
            )
            removed = self._apply_region_reduction(grid=grid, cells=target_cells, reduction_value=cluster_reduction)
            inhibition_log.append(
                f"cluster {relationship.source_cluster_id} contradicted cluster {relationship.target_cluster_id}: removed={round(removed, 4)}"
            )
        return inhibition_log

    def _apply_region_reduction(
        self,
        *,
        grid: List[List[float]],
        cells: Sequence[Tuple[int, int]],
        reduction_value: float,
    ) -> float:
        if reduction_value <= 0.0 or not cells:
            return 0.0
        per_cell = reduction_value / float(len(cells))
        removed_total = 0.0
        for x, y in cells:
            current = float(grid[y][x])
            removed = min(current, per_cell)
            grid[y][x] = current - removed
            removed_total += removed
        return removed_total

    def _active_cells_from_grid(
        self,
        *,
        grid: Sequence[Sequence[float]],
        node_sources: Sequence[Sequence[Set[int]]],
        cluster_sources: Sequence[Sequence[Set[int]]],
    ) -> List[ReasoningPixelCell]:
        cells: List[ReasoningPixelCell] = []
        for y, row in enumerate(grid):
            for x, activation_value in enumerate(row):
                if activation_value <= 0.01:
                    continue
                cells.append(
                    ReasoningPixelCell(
                        x=x,
                        y=y,
                        activation_value=round(float(activation_value), 4),
                        source_nodes=sorted(node_sources[y][x]),
                        source_clusters=sorted(cluster_sources[y][x]),
                    )
                )
        cells.sort(key=lambda item: (-item.activation_value, item.y, item.x))
        return cells[: min(128, len(cells))]

    def _sum_region_activation(
        self,
        *,
        grid: Sequence[Sequence[float]],
        cells: Sequence[Tuple[int, int]],
    ) -> float:
        return sum(float(grid[y][x]) for x, y in cells)

    def _build_spatial_node_mappings(
        self,
        *,
        nodes: Dict[int, ReasoningGraphNode],
        node_centers: Dict[int, Tuple[int, int]],
        node_regions: Dict[int, List[Tuple[int, int]]],
        initial_grid: Sequence[Sequence[float]],
        propagated_grid: Sequence[Sequence[float]],
        final_grid: Sequence[Sequence[float]],
    ) -> List[ReasoningSpatialNodeMapping]:
        mappings: List[ReasoningSpatialNodeMapping] = []
        for node in sorted(nodes.values(), key=lambda item: item.entry_id):
            center_x, center_y = node_centers.get(node.entry_id, (0, 0))
            cells = node_regions.get(node.entry_id, [])
            mappings.append(
                ReasoningSpatialNodeMapping(
                    entry_id=node.entry_id,
                    cluster_id=node.cluster_id,
                    center_x=center_x,
                    center_y=center_y,
                    cells=list(cells),
                    initial_region_activation=round(self._sum_region_activation(grid=initial_grid, cells=cells), 4),
                    propagated_region_activation=round(self._sum_region_activation(grid=propagated_grid, cells=cells), 4),
                    final_region_activation=round(self._sum_region_activation(grid=final_grid, cells=cells), 4),
                )
            )
        return mappings

    def _build_spatial_cluster_mappings(
        self,
        *,
        clusters: Sequence[ReasoningGraphCluster],
        cluster_centers: Dict[int, Tuple[int, int]],
        cluster_regions: Dict[int, List[Tuple[int, int]]],
        final_grid: Sequence[Sequence[float]],
        cluster_radii: Dict[int, int],
    ) -> List[ReasoningSpatialClusterMapping]:
        mappings: List[ReasoningSpatialClusterMapping] = []
        for cluster in sorted(clusters, key=lambda item: item.cluster_id):
            center_x, center_y = cluster_centers.get(cluster.cluster_id, (0, 0))
            cells = cluster_regions.get(cluster.cluster_id, [])
            radius = cluster_radii.get(cluster.cluster_id, 0)
            mappings.append(
                ReasoningSpatialClusterMapping(
                    cluster_id=cluster.cluster_id,
                    center_x=center_x,
                    center_y=center_y,
                    radius=radius,
                    entry_ids=list(cluster.nodes),
                    activation_value=round(float(cluster.activation_value), 4),
                    final_region_activation=round(self._sum_region_activation(grid=final_grid, cells=cells), 4),
                    boundary_cells=self._boundary_cells(cells=cells, center_x=center_x, center_y=center_y, radius=radius),
                )
            )
        return mappings

    def _boundary_cells(
        self,
        *,
        cells: Sequence[Tuple[int, int]],
        center_x: int,
        center_y: int,
        radius: int,
    ) -> List[Tuple[int, int]]:
        if not cells or radius <= 0:
            return []
        boundary: List[Tuple[int, int]] = []
        for x, y in cells:
            if abs(x - center_x) == radius or abs(y - center_y) == radius:
                boundary.append((x, y))
        boundary.sort(key=lambda item: (item[1], item[0]))
        return boundary[:24]

    def _placement_map_lines(
        self,
        *,
        width: int,
        height: int,
        node_centers: Dict[int, Tuple[int, int]],
        cluster_centers: Dict[int, Tuple[int, int]],
    ) -> List[str]:
        if width <= 0 or height <= 0:
            return []
        canvas = [[" " for _ in range(width)] for _ in range(height)]
        for cluster_id, (center_x, center_y) in cluster_centers.items():
            if 0 <= center_x < width and 0 <= center_y < height:
                canvas[center_y][center_x] = "C"
        for entry_id, (center_x, center_y) in node_centers.items():
            marker = str(entry_id)[-1]
            if 0 <= center_x < width and 0 <= center_y < height:
                canvas[center_y][center_x] = marker
        bucket_size = 2 if width >= 24 else 1
        lines: List[str] = []
        for start_y in range(0, height, bucket_size):
            row_chars: List[str] = []
            for start_x in range(0, width, bucket_size):
                bucket_chars: List[str] = []
                for inner_y in range(start_y, min(height, start_y + bucket_size)):
                    for inner_x in range(start_x, min(width, start_x + bucket_size)):
                        char = canvas[inner_y][inner_x]
                        if char != " ":
                            bucket_chars.append(char)
                row_chars.append(bucket_chars[0] if bucket_chars else " ")
            lines.append("".join(row_chars))
        return lines

    def _cluster_boundary_lines(
        self,
        *,
        cluster_mappings: Sequence[ReasoningSpatialClusterMapping],
    ) -> List[str]:
        lines: List[str] = []
        for mapping in cluster_mappings:
            lines.append(
                f"cluster {mapping.cluster_id} boundary={mapping.boundary_cells} radius={mapping.radius} center=({mapping.center_x},{mapping.center_y})"
            )
        return lines

    def _extract_spatial_regions(
        self,
        *,
        grid: Sequence[Sequence[float]],
        node_sources: Sequence[Sequence[Set[int]]],
        cluster_sources: Sequence[Sequence[Set[int]]],
    ) -> List[ReasoningSpatialRegion]:
        height = len(grid)
        width = len(grid[0]) if height else 0
        peak_activation = max((float(cell) for row in grid for cell in row), default=0.0)
        if peak_activation <= 0.0:
            return []
        threshold = max(peak_activation * self.scoring.spatial_peak_threshold, 0.05)
        visited: Set[Tuple[int, int]] = set()
        regions: List[ReasoningSpatialRegion] = []
        region_id = 1
        for y in range(height):
            for x in range(width):
                if (x, y) in visited or float(grid[y][x]) < threshold:
                    continue
                queue = deque([(x, y)])
                visited.add((x, y))
                cells: List[Tuple[int, int]] = []
                total_activation = 0.0
                max_region_activation = 0.0
                weighted_x = 0.0
                weighted_y = 0.0
                region_nodes: Set[int] = set()
                region_clusters: Set[int] = set()
                while queue:
                    current_x, current_y = queue.popleft()
                    cell_activation = float(grid[current_y][current_x])
                    cells.append((current_x, current_y))
                    total_activation += cell_activation
                    max_region_activation = max(max_region_activation, cell_activation)
                    weighted_x += current_x * cell_activation
                    weighted_y += current_y * cell_activation
                    region_nodes.update(node_sources[current_y][current_x])
                    region_clusters.update(cluster_sources[current_y][current_x])
                    for delta_y in (-1, 0, 1):
                        for delta_x in (-1, 0, 1):
                            if delta_x == 0 and delta_y == 0:
                                continue
                            neighbor_x = current_x + delta_x
                            neighbor_y = current_y + delta_y
                            if neighbor_x < 0 or neighbor_y < 0 or neighbor_x >= width or neighbor_y >= height:
                                continue
                            if (neighbor_x, neighbor_y) in visited:
                                continue
                            if float(grid[neighbor_y][neighbor_x]) < threshold:
                                continue
                            visited.add((neighbor_x, neighbor_y))
                            queue.append((neighbor_x, neighbor_y))
                centroid_x = (weighted_x / total_activation) if total_activation > 0.0 else float(x)
                centroid_y = (weighted_y / total_activation) if total_activation > 0.0 else float(y)
                regions.append(
                    ReasoningSpatialRegion(
                        region_id=region_id,
                        cells=sorted(cells, key=lambda item: (item[1], item[0])),
                        total_activation=round(total_activation, 4),
                        peak_activation=round(max_region_activation, 4),
                        centroid_x=round(centroid_x, 2),
                        centroid_y=round(centroid_y, 2),
                        source_nodes=sorted(region_nodes),
                        source_clusters=sorted(region_clusters),
                    )
                )
                region_id += 1
        regions.sort(key=lambda item: (-item.total_activation, -item.peak_activation, item.region_id))
        return regions[: max(1, int(self.scoring.spatial_peak_limit))]

    def _merge_spatial_regions(
        self,
        *,
        regions: Sequence[ReasoningSpatialRegion],
        edges: Sequence[ReasoningGraphEdge],
        nodes: Dict[int, ReasoningGraphNode],
    ) -> tuple[List[ReasoningSpatialRegion], List[str]]:
        if not regions:
            return [], []
        merge_log: List[str] = []
        remaining = list(regions)
        merged_any = True
        while merged_any:
            merged_any = False
            next_regions: List[ReasoningSpatialRegion] = []
            used_indices: Set[int] = set()
            for left_index, left_region in enumerate(remaining):
                if left_index in used_indices:
                    continue
                merged_region = left_region
                for right_index in range(left_index + 1, len(remaining)):
                    if right_index in used_indices:
                        continue
                    right_region = remaining[right_index]
                    distance = math.sqrt(
                        ((float(merged_region.centroid_x) - float(right_region.centroid_x)) ** 2)
                        + ((float(merged_region.centroid_y) - float(right_region.centroid_y)) ** 2)
                    )
                    if distance > self.scoring.spatial_region_merge_distance:
                        continue
                    support_strength = self._support_between_regions(left_region=merged_region, right_region=right_region, edges=edges)
                    if support_strength < self.scoring.spatial_region_merge_support_threshold:
                        continue
                    merged_any = True
                    used_indices.add(right_index)
                    merge_log.append(
                        f"merged region {merged_region.region_id} with region {right_region.region_id}: distance={round(distance, 4)} support={round(support_strength, 4)}"
                    )
                    merged_region = self._combine_regions(
                        left_region=merged_region,
                        right_region=right_region,
                        nodes=nodes,
                    )
                used_indices.add(left_index)
                next_regions.append(merged_region)
            remaining = next_regions
        normalized: List[ReasoningSpatialRegion] = []
        for index, region in enumerate(sorted(remaining, key=lambda item: (-item.total_activation, -item.peak_activation, item.region_id)), start=1):
            normalized.append(
                ReasoningSpatialRegion(
                    region_id=index,
                    cells=list(region.cells),
                    total_activation=region.total_activation,
                    peak_activation=region.peak_activation,
                    centroid_x=region.centroid_x,
                    centroid_y=region.centroid_y,
                    source_nodes=list(region.source_nodes),
                    source_clusters=list(region.source_clusters),
                    competition_penalty=region.competition_penalty,
                    post_competition_rank=region.post_competition_rank,
                    dominance_value=region.dominance_value,
                    merged_region_ids=list(region.merged_region_ids or [region.region_id]),
                    dominant_node_id=region.dominant_node_id,
                )
            )
        return normalized[: max(1, int(self.scoring.spatial_peak_limit))], merge_log

    def _support_between_regions(
        self,
        *,
        left_region: ReasoningSpatialRegion,
        right_region: ReasoningSpatialRegion,
        edges: Sequence[ReasoningGraphEdge],
    ) -> float:
        left_nodes = set(left_region.source_nodes)
        right_nodes = set(right_region.source_nodes)
        if not left_nodes or not right_nodes:
            return 0.0
        total_support = 0.0
        for edge in edges:
            if edge.source_entry_id in left_nodes and edge.target_entry_id in right_nodes:
                if edge.relationship_type == RelationshipType.SUPPORTS:
                    total_support += float(edge.strength)
                elif edge.relationship_type == RelationshipType.CAUSES:
                    total_support += float(edge.strength) * 0.7
            elif edge.source_entry_id in right_nodes and edge.target_entry_id in left_nodes:
                if edge.relationship_type == RelationshipType.SUPPORTS:
                    total_support += float(edge.strength)
                elif edge.relationship_type == RelationshipType.CAUSES:
                    total_support += float(edge.strength) * 0.7
        return total_support

    def _contradiction_between_regions(
        self,
        *,
        left_region: ReasoningSpatialRegion,
        right_region: ReasoningSpatialRegion,
        edges: Sequence[ReasoningGraphEdge],
    ) -> float:
        left_nodes = set(left_region.source_nodes)
        right_nodes = set(right_region.source_nodes)
        if not left_nodes or not right_nodes:
            return 0.0
        total_contradiction = 0.0
        for edge in edges:
            if edge.relationship_type != RelationshipType.CONTRADICTS:
                continue
            if edge.source_entry_id in left_nodes and edge.target_entry_id in right_nodes:
                total_contradiction += float(edge.strength)
            elif edge.source_entry_id in right_nodes and edge.target_entry_id in left_nodes:
                total_contradiction += float(edge.strength)
        return total_contradiction

    def _region_memory_key(self, region: ReasoningSpatialRegion) -> str:
        if region.source_nodes:
            return "nodes:" + ",".join(str(node_id) for node_id in sorted(region.source_nodes))
        if region.source_clusters:
            return "clusters:" + ",".join(str(cluster_id) for cluster_id in sorted(region.source_clusters))
        return f"region:{region.region_id}"

    def _directional_expansion_cells(
        self,
        *,
        region: ReasoningSpatialRegion,
        regions: Sequence[ReasoningSpatialRegion],
        edges: Sequence[ReasoningGraphEdge],
        width: int,
        height: int,
        extra_radius: int,
    ) -> List[Tuple[int, int]]:
        expanded_cells = self._expanded_region_cells(
            cells=region.cells,
            width=width,
            height=height,
            extra_radius=extra_radius,
        )
        if not expanded_cells:
            return []
        direction_x = 0.0
        direction_y = 0.0
        for other_region in regions:
            if other_region.region_id == region.region_id:
                continue
            delta_x = float(other_region.centroid_x) - float(region.centroid_x)
            delta_y = float(other_region.centroid_y) - float(region.centroid_y)
            distance = math.sqrt((delta_x * delta_x) + (delta_y * delta_y))
            if distance <= 0.001:
                continue
            support_strength = self._support_between_regions(left_region=region, right_region=other_region, edges=edges)
            contradiction_strength = self._contradiction_between_regions(left_region=region, right_region=other_region, edges=edges)
            relation_force = support_strength - contradiction_strength
            if relation_force == 0.0:
                continue
            direction_x += (delta_x / distance) * relation_force
            direction_y += (delta_y / distance) * relation_force
        direction_magnitude = math.sqrt((direction_x * direction_x) + (direction_y * direction_y))
        if direction_magnitude <= 0.001:
            return expanded_cells
        unit_x = direction_x / direction_magnitude
        unit_y = direction_y / direction_magnitude
        selected_cells: List[Tuple[int, int]] = []
        base_cells = set(region.cells)
        for x, y in expanded_cells:
            if (x, y) in base_cells:
                selected_cells.append((x, y))
                continue
            outward_x = float(x) - float(region.centroid_x)
            outward_y = float(y) - float(region.centroid_y)
            outward_magnitude = math.sqrt((outward_x * outward_x) + (outward_y * outward_y))
            if outward_magnitude <= 0.001:
                selected_cells.append((x, y))
                continue
            alignment = ((outward_x / outward_magnitude) * unit_x) + ((outward_y / outward_magnitude) * unit_y)
            if alignment >= -0.15:
                selected_cells.append((x, y))
        return selected_cells or list(region.cells)

    def _combine_regions(
        self,
        *,
        left_region: ReasoningSpatialRegion,
        right_region: ReasoningSpatialRegion,
        nodes: Dict[int, ReasoningGraphNode],
    ) -> ReasoningSpatialRegion:
        combined_cells = sorted(set(left_region.cells) | set(right_region.cells), key=lambda item: (item[1], item[0]))
        total_activation = float(left_region.total_activation) + float(right_region.total_activation)
        weighted_x = (
            (float(left_region.centroid_x) * float(left_region.total_activation))
            + (float(right_region.centroid_x) * float(right_region.total_activation))
        )
        weighted_y = (
            (float(left_region.centroid_y) * float(left_region.total_activation))
            + (float(right_region.centroid_y) * float(right_region.total_activation))
        )
        centroid_x = (weighted_x / total_activation) if total_activation > 0.0 else float(left_region.centroid_x)
        centroid_y = (weighted_y / total_activation) if total_activation > 0.0 else float(left_region.centroid_y)
        source_nodes = sorted(set(left_region.source_nodes) | set(right_region.source_nodes))
        source_clusters = sorted(set(left_region.source_clusters) | set(right_region.source_clusters))
        merged_ids = sorted(set(left_region.merged_region_ids or [left_region.region_id]) | set(right_region.merged_region_ids or [right_region.region_id]))
        peak_activation = max(float(left_region.peak_activation), float(right_region.peak_activation))
        dominant_node_id = self._dominant_node_for_region_node_ids(
            node_ids=source_nodes,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            nodes=nodes,
        )
        return ReasoningSpatialRegion(
            region_id=min(left_region.region_id, right_region.region_id),
            cells=combined_cells,
            total_activation=round(total_activation, 4),
            peak_activation=round(peak_activation, 4),
            centroid_x=round(centroid_x, 2),
            centroid_y=round(centroid_y, 2),
            source_nodes=source_nodes,
            source_clusters=source_clusters,
            merged_region_ids=merged_ids,
            dominant_node_id=dominant_node_id,
        )

    def _score_region_dominance(
        self,
        *,
        regions: Sequence[ReasoningSpatialRegion],
        grid: List[List[float]],
        width: int,
        height: int,
        edges: Sequence[ReasoningGraphEdge] = (),
        previous_dominance: Dict[str, float] | None = None,
        tick: int = 1,
    ) -> tuple[List[ReasoningSpatialRegion], List[str], List[str], Dict[str, float]]:
        if not regions:
            return [], [], [], {}
        max_total = max(float(region.total_activation) for region in regions)
        max_peak = max(float(region.peak_activation) for region in regions)
        scored_regions: List[ReasoningSpatialRegion] = []
        dominance_log: List[str] = []
        expansion_log: List[str] = []
        next_dominance: Dict[str, float] = {}
        previous_dominance = previous_dominance or {}
        raw_dominance_by_region: Dict[int, float] = {}
        for region in regions:
            normalized_total = 0.0 if max_total <= 0.0 else float(region.total_activation) / max_total
            normalized_peak = 0.0 if max_peak <= 0.0 else float(region.peak_activation) / max_peak
            raw_dominance_by_region[region.region_id] = _clamp(
                (normalized_total * 0.7) + (normalized_peak * 0.3),
                low=0.0,
                high=1.5,
            )
        for region in regions:
            raw_dominance = raw_dominance_by_region.get(region.region_id, 0.0)
            region_key = self._region_memory_key(region)
            previous_value = float(previous_dominance.get(region_key, float(region.dominance_value)))
            stronger_neighbor_pressure = sum(
                max(0.0, other_value - raw_dominance)
                for other_region_id, other_value in raw_dominance_by_region.items()
                if other_region_id != region.region_id
            )
            reinforcement = raw_dominance * self.scoring.spatial_temporal_reinforcement_weight
            suppression = stronger_neighbor_pressure * self.scoring.spatial_temporal_suppression_weight
            dominance_value = _clamp(previous_value + reinforcement - suppression, low=0.0, high=1.0)
            if tick <= 1 and previous_value <= 0.0:
                dominance_value = _clamp(raw_dominance, low=0.0, high=1.0)
            next_dominance[region_key] = dominance_value
            dominance_log.append(
                f"t{tick}: region {region.region_id} dominance={round(dominance_value, 4)} previous={round(previous_value, 4)} reinforcement={round(reinforcement, 4)} suppression={round(suppression, 4)} total={round(float(region.total_activation), 4)} peak={round(float(region.peak_activation), 4)}"
            )
            if dominance_value >= self.scoring.spatial_region_dominance_threshold:
                expansion_cells = self._directional_expansion_cells(
                    region=region,
                    regions=regions,
                    edges=edges,
                    width=width,
                    height=height,
                    extra_radius=max(1, int(self.scoring.spatial_region_expansion_radius)),
                )
                expansion_strength = float(region.total_activation) * dominance_value * self.scoring.spatial_region_dominance_weight * 0.08
                per_cell = 0.0 if not expansion_cells else expansion_strength / float(len(expansion_cells))
                for x, y in expansion_cells:
                    grid[y][x] += per_cell
                expansion_log.append(
                    f"t{tick}: region {region.region_id} directionally expanded: dominance={round(dominance_value, 4)} cells={len(expansion_cells)} added={round(expansion_strength, 4)}"
                )
            scored_regions.append(
                ReasoningSpatialRegion(
                    region_id=region.region_id,
                    cells=list(region.cells),
                    total_activation=region.total_activation,
                    peak_activation=region.peak_activation,
                    centroid_x=region.centroid_x,
                    centroid_y=region.centroid_y,
                    source_nodes=list(region.source_nodes),
                    source_clusters=list(region.source_clusters),
                    competition_penalty=region.competition_penalty,
                    post_competition_rank=region.post_competition_rank,
                    dominance_value=round(dominance_value, 4),
                    merged_region_ids=list(region.merged_region_ids),
                    dominant_node_id=region.dominant_node_id,
                )
            )
        return scored_regions, dominance_log, expansion_log, next_dominance

    def _apply_region_competition(
        self,
        *,
        grid: List[List[float]],
        regions: Sequence[ReasoningSpatialRegion],
        width: int,
        height: int,
    ) -> List[str]:
        if not regions:
            return []
        competition_log: List[str] = []
        ordered_regions = sorted(
            regions,
            key=lambda item: (-item.dominance_value, -item.total_activation, item.region_id),
        )
        for index, stronger in enumerate(ordered_regions):
            for weaker in ordered_regions[index + 1:]:
                distance = math.sqrt(
                    ((float(stronger.centroid_x) - float(weaker.centroid_x)) ** 2)
                    + ((float(stronger.centroid_y) - float(weaker.centroid_y)) ** 2)
                )
                if distance > self.scoring.spatial_region_competition_radius:
                    continue
                strength_gap = max(
                    0.0,
                    (float(stronger.total_activation) * (1.0 + float(stronger.dominance_value)))
                    - (float(weaker.total_activation) * (1.0 + float(weaker.dominance_value) * 0.5)),
                )
                if strength_gap <= 0.0:
                    continue
                proximity_scale = 1.0 - (distance / self.scoring.spatial_region_competition_radius)
                dominance_scale = 1.0 + max(0.0, float(stronger.dominance_value) - float(weaker.dominance_value))
                reduction_value = strength_gap * proximity_scale * dominance_scale * self.scoring.spatial_region_competition_weight
                expansion_radius = 1
                if float(stronger.dominance_value) >= self.scoring.spatial_region_dominance_threshold:
                    expansion_radius = max(expansion_radius, int(self.scoring.spatial_region_expansion_radius))
                removed = self._apply_region_reduction(
                    grid=grid,
                    cells=self._expanded_region_cells(
                        cells=weaker.cells,
                        width=width,
                        height=height,
                        extra_radius=expansion_radius,
                    ),
                    reduction_value=reduction_value,
                )
                if removed <= 0.0:
                    continue
                competition_log.append(
                    f"region {stronger.region_id} suppressed region {weaker.region_id}: distance={round(distance, 4)} dominance_scale={round(dominance_scale, 4)} removed={round(removed, 4)}"
                )
        return competition_log

    def _reconcile_region_metadata(
        self,
        *,
        regions: Sequence[ReasoningSpatialRegion],
        prior_regions: Sequence[ReasoningSpatialRegion],
        nodes: Dict[int, ReasoningGraphNode],
    ) -> List[ReasoningSpatialRegion]:
        reconciled: List[ReasoningSpatialRegion] = []
        for region in regions:
            best_match = None
            best_overlap = -1
            region_nodes = set(region.source_nodes)
            for prior_region in prior_regions:
                overlap = len(region_nodes & set(prior_region.source_nodes))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = prior_region
            dominance_value = 0.0 if best_match is None else float(best_match.dominance_value)
            merged_region_ids = [] if best_match is None else list(best_match.merged_region_ids)
            dominant_node_id = self._dominant_node_for_region_node_ids(
                node_ids=region.source_nodes,
                centroid_x=region.centroid_x,
                centroid_y=region.centroid_y,
                nodes=nodes,
            )
            reconciled.append(
                ReasoningSpatialRegion(
                    region_id=region.region_id,
                    cells=list(region.cells),
                    total_activation=region.total_activation,
                    peak_activation=region.peak_activation,
                    centroid_x=region.centroid_x,
                    centroid_y=region.centroid_y,
                    source_nodes=list(region.source_nodes),
                    source_clusters=list(region.source_clusters),
                    competition_penalty=0.0 if best_match is None else float(best_match.competition_penalty),
                    post_competition_rank=region.post_competition_rank,
                    dominance_value=round(dominance_value, 4),
                    merged_region_ids=merged_region_ids,
                    dominant_node_id=dominant_node_id,
                )
            )
        return reconciled

    def _dominant_node_for_region_node_ids(
        self,
        *,
        node_ids: Sequence[int],
        centroid_x: float,
        centroid_y: float,
        nodes: Dict[int, ReasoningGraphNode],
        spatial_grid: ReasoningSpatialGrid | None = None,
    ) -> int | None:
        if not node_ids:
            return None
        mapping_lookup: Dict[int, ReasoningSpatialNodeMapping] = {}
        if spatial_grid is not None:
            mapping_lookup = {mapping.entry_id: mapping for mapping in spatial_grid.node_mappings}
        best_node_id = None
        best_score = None
        for node_id in node_ids:
            node = nodes.get(int(node_id))
            if node is None:
                continue
            mapping = mapping_lookup.get(int(node_id))
            distance_penalty = 0.0
            if mapping is not None:
                distance_penalty = math.sqrt(
                    ((float(mapping.center_x) - float(centroid_x)) ** 2)
                    + ((float(mapping.center_y) - float(centroid_y)) ** 2)
                ) * 0.08
            candidate_score = float(node.activation_value) + (float(node.final_score) * 0.12) - distance_penalty
            candidate_key = (candidate_score, float(node.activation_value), -int(node.entry_id))
            if best_score is None or candidate_key > best_score:
                best_score = candidate_key
                best_node_id = int(node.entry_id)
        return best_node_id

    def _heatmap_lines(self, grid: Sequence[Sequence[float]]) -> List[str]:
        if not grid:
            return []
        levels = " .:-=+*#%@"
        height = len(grid)
        width = len(grid[0]) if height else 0
        bucket_size = 2 if width >= 24 else 1
        max_activation = max((float(cell) for row in grid for cell in row), default=0.0)
        if max_activation <= 0.0:
            compressed_height = max(1, height // bucket_size)
            compressed_width = max(1, width // bucket_size)
            return [" " * compressed_width for _ in range(compressed_height)]
        lines: List[str] = []
        for start_y in range(0, height, bucket_size):
            row_chars: List[str] = []
            for start_x in range(0, width, bucket_size):
                bucket_values: List[float] = []
                for inner_y in range(start_y, min(height, start_y + bucket_size)):
                    for inner_x in range(start_x, min(width, start_x + bucket_size)):
                        bucket_values.append(float(grid[inner_y][inner_x]))
                average_value = sum(bucket_values) / float(len(bucket_values)) if bucket_values else 0.0
                normalized = average_value / max_activation
                index = min(len(levels) - 1, int(round(normalized * (len(levels) - 1))))
                row_chars.append(levels[index])
            lines.append("".join(row_chars))
        return lines

    def _select_spatial_region(
        self,
        *,
        spatial_grid: ReasoningSpatialGrid | None,
        selected_node_entry_id: int | None,
        selected_cluster_id: int | None,
    ) -> tuple[ReasoningSpatialRegion | None, str, str]:
        if spatial_grid is None or not spatial_grid.top_regions:
            return None, "", ""
        strongest_region = spatial_grid.top_regions[0]
        if selected_node_entry_id is not None:
            node_match = next(
                (region for region in spatial_grid.top_regions if selected_node_entry_id in region.source_nodes),
                None,
            )
            if node_match is not None:
                warning = ""
                if node_match.region_id != strongest_region.region_id:
                    warning = (
                        f"Selected node {selected_node_entry_id} aligns with spatial region {node_match.region_id}, "
                        f"but region {strongest_region.region_id} is strongest overall."
                    )
                return (
                    node_match,
                    f"Spatial peak region {node_match.region_id} aligned with selected node {selected_node_entry_id}.",
                    warning,
                )
        if selected_cluster_id is not None:
            cluster_match = next(
                (region for region in spatial_grid.top_regions if selected_cluster_id in region.source_clusters),
                None,
            )
            if cluster_match is not None:
                warning = ""
                if cluster_match.region_id != strongest_region.region_id:
                    warning = (
                        f"Selected cluster {selected_cluster_id} aligns with spatial region {cluster_match.region_id}, "
                        f"but region {strongest_region.region_id} is strongest overall."
                    )
                return (
                    cluster_match,
                    f"Spatial peak region {cluster_match.region_id} aligned with selected cluster {selected_cluster_id}.",
                    warning,
                )
        return (
            strongest_region,
            f"Spatial peak region {strongest_region.region_id} provided the strongest activation focus.",
            "Selected graph outcome did not align with any dominant spatial region.",
        )

    def _apply_cluster_influence(
        self,
        *,
        nodes: Dict[int, ReasoningGraphNode],
        clusters: Sequence[ReasoningGraphCluster],
        query_intent: QueryIntent,
    ) -> None:
        for node in nodes.values():
            node.cluster_influence_bonus = 0.0
        if query_intent != QueryIntent.EXPLANATION or not clusters:
            return
        max_cluster_score = max((cluster.cluster_score for cluster in clusters), default=0.0)
        if max_cluster_score <= 0.0:
            return
        for cluster in clusters:
            cluster_ratio = cluster.cluster_score / max_cluster_score
            cluster_influence = round(cluster_ratio * self.scoring.cluster_influence_weight, 4)
            for entry_id in cluster.nodes:
                node = nodes.get(entry_id)
                if node is None:
                    continue
                node.cluster_influence_bonus = cluster_influence

    def _cluster_relationship_support_strength(self, edge: ReasoningGraphEdge) -> float:
        if edge.relationship_type == RelationshipType.SUPPORTS:
            return edge.strength
        if edge.relationship_type == RelationshipType.CAUSES:
            return edge.strength * self.scoring.cluster_causal_support_weight
        if edge.relationship_type == RelationshipType.EXTENDS:
            return edge.strength * self.scoring.cluster_extend_support_weight
        return 0.0

    def _select_best_path(
        self,
        *,
        graph: ReasoningGraph,
        cluster_id: int,
    ) -> tuple[ReasoningGraphPath | None, List[ReasoningGraphPath]]:
        candidate_paths = self._enumerate_cluster_paths(graph=graph, cluster_id=cluster_id)
        if not candidate_paths:
            return None, []
        candidate_paths = self._apply_path_competition(candidate_paths)
        best_path = max(
            candidate_paths,
            key=lambda item: (item.path_final_score, item.path_score, item.path_activation, [-node_id for node_id in item.node_ids]),
        )
        ordered_paths = sorted(
            candidate_paths,
            key=lambda item: (-item.path_final_score, -item.adjusted_score, -item.path_activation, item.node_ids),
        )
        best_path = self._select_exploratory_path(
            cluster_id=cluster_id,
            ordered_paths=ordered_paths,
            default_path=best_path,
        )
        return best_path, ordered_paths

    def _select_exploratory_path(
        self,
        *,
        cluster_id: int,
        ordered_paths: Sequence[ReasoningGraphPath],
        default_path: ReasoningGraphPath,
    ) -> ReasoningGraphPath:
        if len(ordered_paths) < 2 or self.scoring.concept_path_exploration_rate <= 0.0:
            return default_path
        payload = (
            f"{self.scoring.concept_exploration_seed}:path:{cluster_id}:"
            + "|".join(",".join(str(node_id) for node_id in path.node_ids) for path in ordered_paths[:4])
        )
        roll = round(random.Random(payload).random(), 4)
        if roll >= self.scoring.concept_path_exploration_rate:
            return default_path
        viable_paths = [
            path
            for path in ordered_paths[1:4]
            if float(path.path_final_score) >= (float(default_path.path_final_score) * 0.72)
        ]
        if not viable_paths:
            return default_path
        exploratory_path = max(
            viable_paths,
            key=lambda item: (item.path_activation, item.adjusted_score, item.path_score, [-node_id for node_id in item.node_ids]),
        )
        return ReasoningGraphPath(
            node_ids=list(exploratory_path.node_ids),
            path_score=exploratory_path.path_score,
            relative_strength=exploratory_path.relative_strength,
            adjusted_score=exploratory_path.adjusted_score,
            path_activation=exploratory_path.path_activation,
            path_final_score=exploratory_path.path_final_score,
            breakdown=(
                f"{exploratory_path.breakdown} + exploratory_path_selection roll={roll} "
                + f"rate={round(self.scoring.concept_path_exploration_rate, 4)} default_path={default_path.node_ids}"
            ),
        )

    def _enumerate_cluster_paths(
        self,
        *,
        graph: ReasoningGraph,
        cluster_id: int,
    ) -> List[ReasoningGraphPath]:
        cluster = next((item for item in graph.clusters if item.cluster_id == cluster_id), None)
        if cluster is None:
            return []
        node_lookup = graph.node_map()
        cluster_nodes = [node_lookup[entry_id] for entry_id in cluster.nodes if entry_id in node_lookup]
        if not cluster_nodes:
            return []
        candidate_paths: Dict[tuple[int, ...], ReasoningGraphPath] = {}
        adjacency = self._path_adjacency(graph=graph, cluster_nodes={node.entry_id for node in cluster_nodes})
        for node in sorted(cluster_nodes, key=lambda item: (-item.final_score, item.entry_id)):
            self._collect_paths_from_node(
                start_node_id=node.entry_id,
                current_path=[node.entry_id],
                adjacency=adjacency,
                node_lookup=node_lookup,
                candidate_paths=candidate_paths,
            )
        if not candidate_paths:
            for node in cluster_nodes:
                path = self._score_path(node_ids=[node.entry_id], traversals=[], node_lookup=node_lookup)
                candidate_paths[tuple(path.node_ids)] = path
        return list(candidate_paths.values())

    def _apply_path_competition(
        self,
        candidate_paths: Sequence[ReasoningGraphPath],
    ) -> List[ReasoningGraphPath]:
        if not candidate_paths:
            return []
        total_path_score = sum(max(0.0, float(path.path_score)) for path in candidate_paths)
        if total_path_score <= 0.0:
            total_path_score = float(len(candidate_paths))
        adjusted_paths: List[ReasoningGraphPath] = []
        for path in candidate_paths:
            relative_strength = max(0.0, float(path.path_score)) / total_path_score if total_path_score > 0.0 else 0.0
            adjusted_score = float(path.path_score) * relative_strength
            path_final_score = (
                float(path.path_score)
                + (adjusted_score * self.scoring.path_competition_weight)
                + (float(path.path_activation) * self.scoring.path_activation_weight)
            )
            adjusted_paths.append(
                ReasoningGraphPath(
                    node_ids=list(path.node_ids),
                    path_score=round(float(path.path_score), 4),
                    relative_strength=round(relative_strength, 4),
                    adjusted_score=round(adjusted_score, 4),
                    path_activation=round(float(path.path_activation), 4),
                    path_final_score=round(path_final_score, 4),
                    breakdown=(
                        f"{path.breakdown} "
                        + f"+ relative_strength={round(relative_strength, 4)} "
                        + f"+ adjusted_score={round(adjusted_score, 4)}"
                    ),
                )
            )
        return adjusted_paths

    def _collect_paths_from_node(
        self,
        *,
        start_node_id: int,
        current_path: List[int],
        adjacency: Dict[int, List[Dict[str, object]]],
        node_lookup: Dict[int, ReasoningGraphNode],
        candidate_paths: Dict[tuple[int, ...], ReasoningGraphPath],
    ) -> None:
        current_key = tuple(current_path)
        if current_key not in candidate_paths:
            traversals = self._path_traversals_for_nodes(current_path=current_path, adjacency=adjacency)
            candidate_paths[current_key] = self._score_path(
                node_ids=current_path,
                traversals=traversals,
                node_lookup=node_lookup,
            )
        if len(current_path) >= self.scoring.reasoning_path_max_nodes:
            return
        tail = current_path[-1]
        options = adjacency.get(tail, [])
        for option in options:
            next_node_id = int(option["target"])
            if next_node_id in current_path:
                continue
            next_path = list(current_path)
            next_path.append(next_node_id)
            self._collect_paths_from_node(
                start_node_id=start_node_id,
                current_path=next_path,
                adjacency=adjacency,
                node_lookup=node_lookup,
                candidate_paths=candidate_paths,
            )

    def _path_adjacency(
        self,
        *,
        graph: ReasoningGraph,
        cluster_nodes: set[int],
    ) -> Dict[int, List[Dict[str, object]]]:
        adjacency: Dict[int, List[Dict[str, object]]] = {entry_id: [] for entry_id in cluster_nodes}
        for edge in graph.edges:
            if edge.source_entry_id not in cluster_nodes or edge.target_entry_id not in cluster_nodes:
                continue
            if edge.relationship_type == RelationshipType.CAUSES:
                adjacency[edge.source_entry_id].append(
                    {"target": edge.target_entry_id, "edge": edge, "reversed": False}
                )
                adjacency[edge.target_entry_id].append(
                    {"target": edge.source_entry_id, "edge": edge, "reversed": True}
                )
            elif edge.relationship_type == RelationshipType.SUPPORTS:
                adjacency[edge.source_entry_id].append(
                    {"target": edge.target_entry_id, "edge": edge, "reversed": False}
                )
                adjacency[edge.target_entry_id].append(
                    {"target": edge.source_entry_id, "edge": edge, "reversed": False}
                )
            elif edge.relationship_type == RelationshipType.EXTENDS:
                adjacency[edge.source_entry_id].append(
                    {"target": edge.target_entry_id, "edge": edge, "reversed": False}
                )
                adjacency[edge.target_entry_id].append(
                    {"target": edge.source_entry_id, "edge": edge, "reversed": True}
                )
        for entry_id, options in adjacency.items():
            options.sort(
                key=lambda item: (
                    item["edge"].relationship_type.value,
                    -float(item["edge"].strength),
                    int(item["target"]),
                    bool(item["reversed"]),
                )
            )
        return adjacency

    def _path_traversals_for_nodes(
        self,
        *,
        current_path: Sequence[int],
        adjacency: Dict[int, List[Dict[str, object]]],
    ) -> List[Dict[str, object]]:
        traversals: List[Dict[str, object]] = []
        for left_node_id, right_node_id in zip(current_path, current_path[1:]):
            match = next(
                (
                    option
                    for option in adjacency.get(left_node_id, [])
                    if int(option["target"]) == int(right_node_id)
                ),
                None,
            )
            if match is None:
                continue
            traversals.append(match)
        return traversals

    def _score_path(
        self,
        *,
        node_ids: Sequence[int],
        traversals: Sequence[Dict[str, object]],
        node_lookup: Dict[int, ReasoningGraphNode],
    ) -> ReasoningGraphPath:
        node_score_sum = sum(
            float(node_lookup[node_id].final_score)
            for node_id in node_ids
            if node_id in node_lookup
        ) * self.scoring.path_node_weight
        edge_score_sum = 0.0
        reversed_penalty = 0.0
        edge_terms: List[str] = []
        for traversal in traversals:
            edge = traversal["edge"]
            relationship_type = edge.relationship_type
            edge_weight = self._path_edge_weight(relationship_type)
            contribution = float(edge.strength) * edge_weight
            edge_score_sum += contribution
            edge_term = f"{relationship_type.value.lower()}={round(contribution, 4)}"
            if bool(traversal.get("reversed", False)):
                reversed_penalty += self.scoring.path_reversed_edge_penalty
                edge_term += f"(reversed_penalty={round(self.scoring.path_reversed_edge_penalty, 4)})"
            edge_terms.append(edge_term)
        path_score = max(0.0, node_score_sum + edge_score_sum - reversed_penalty)
        flow_breakdown: List[str] = []
        if node_ids:
            current_flow = float(node_lookup[node_ids[0]].activation_value) if node_ids[0] in node_lookup else 0.0
            path_activation = current_flow
            flow_breakdown.append(f"entry {node_ids[0]} start={round(current_flow, 4)}")
        else:
            current_flow = 0.0
            path_activation = 0.0
        for index, traversal in enumerate(traversals, start=1):
            edge = traversal["edge"]
            next_node_id = int(node_ids[index]) if index < len(node_ids) else int(traversal["target"])
            next_base_activation = float(node_lookup[next_node_id].activation_value) if next_node_id in node_lookup else 0.0
            edge_flow = current_flow * self._path_edge_weight(edge.relationship_type) * self.scoring.activation_decay_factor
            if bool(traversal.get("reversed", False)):
                edge_flow = max(0.0, edge_flow - self.scoring.path_reversed_edge_penalty)
            next_flow = next_base_activation + edge_flow
            path_activation += next_flow
            flow_breakdown.append(
                f"entry {next_node_id} base={round(next_base_activation, 4)} flow={round(edge_flow, 4)} total={round(next_flow, 4)}"
            )
            current_flow = next_flow
        path_final_score = path_score + (path_activation * self.scoring.path_activation_weight)
        breakdown = (
            f"node_sum={round(node_score_sum, 4)} "
            + f"+ edge_sum={round(edge_score_sum, 4)} "
            + f"- reversed_penalty={round(reversed_penalty, 4)} "
            + f"+ activation={round(path_activation, 4)} "
            + f"edges={edge_terms} "
            + f"flow={flow_breakdown}"
        )
        return ReasoningGraphPath(
            node_ids=[int(node_id) for node_id in node_ids],
            path_score=round(path_score, 4),
            relative_strength=0.0,
            adjusted_score=0.0,
            path_activation=round(path_activation, 4),
            path_final_score=round(path_final_score, 4),
            breakdown=breakdown,
        )

    def _path_edge_weight(self, relationship_type: RelationshipType) -> float:
        if relationship_type == RelationshipType.CAUSES:
            return self.scoring.path_edge_weight_causes
        if relationship_type == RelationshipType.SUPPORTS:
            return self.scoring.path_edge_weight_supports
        if relationship_type == RelationshipType.EXTENDS:
            return self.scoring.path_edge_weight_extends
        return 0.0

    def _extract_reasoning_path(
        self,
        *,
        graph: ReasoningGraph,
        cluster_id: int,
        selected_node_entry_id: int | None,
    ) -> List[int]:
        if selected_node_entry_id is None:
            return []
        cluster = next((item for item in graph.clusters if item.cluster_id == cluster_id), None)
        if cluster is None:
            return [selected_node_entry_id]
        cluster_nodes = set(cluster.nodes)
        adjacency: Dict[int, List[tuple[int, float, int]]] = {entry_id: [] for entry_id in cluster.nodes}
        for edge in graph.edges:
            if edge.source_entry_id not in cluster_nodes or edge.target_entry_id not in cluster_nodes:
                continue
            if edge.relationship_type == RelationshipType.CAUSES:
                adjacency.setdefault(edge.source_entry_id, []).append((edge.target_entry_id, edge.strength, 2))
            elif edge.relationship_type == RelationshipType.SUPPORTS:
                adjacency.setdefault(edge.source_entry_id, []).append((edge.target_entry_id, edge.strength, 1))
                adjacency.setdefault(edge.target_entry_id, []).append((edge.source_entry_id, edge.strength, 1))
        path = [selected_node_entry_id]
        visited = {selected_node_entry_id}
        while len(path) < self.scoring.reasoning_path_max_nodes:
            tail = path[-1]
            options = [
                item
                for item in adjacency.get(tail, [])
                if item[0] not in visited
            ]
            if not options:
                break
            options.sort(key=lambda item: (-item[1], -item[2], item[0]))
            next_node_id = options[0][0]
            path.append(next_node_id)
            visited.add(next_node_id)
        if len(path) == 1:
            incoming_options: List[tuple[int, float, int]] = []
            for source_entry_id, targets in adjacency.items():
                if source_entry_id in visited:
                    continue
                for target_entry_id, strength, priority in targets:
                    if target_entry_id == selected_node_entry_id:
                        incoming_options.append((source_entry_id, strength, priority))
            if incoming_options:
                incoming_options.sort(key=lambda item: (-item[1], -item[2], item[0]))
                path.insert(0, incoming_options[0][0])
        return path

    def _intent_bonus(
        self,
        *,
        candidate: ReasoningGraphCandidate,
        query_intent: QueryIntent,
    ) -> float:
        if query_intent == QueryIntent.FACT:
            return candidate.concise_score * self.scoring.fact_concise_bonus_weight
        if query_intent == QueryIntent.EXPLANATION:
            return candidate.richness_score * self.scoring.explanation_richness_bonus_weight
        if query_intent == QueryIntent.PROCESS:
            return candidate.richness_score * self.scoring.process_richness_bonus_weight
        return 0.0

    def _support_weight(self, query_intent: QueryIntent) -> float:
        if query_intent == QueryIntent.FACT:
            return self.scoring.support_weight_fact
        if query_intent == QueryIntent.EXPLANATION:
            return self.scoring.support_weight_explanation
        if query_intent == QueryIntent.PROCESS:
            return self.scoring.support_weight_process
        return self.scoring.support_weight_unknown

    def _causal_weight(self, query_intent: QueryIntent) -> float:
        if query_intent == QueryIntent.FACT:
            return self.scoring.causal_weight_fact
        if query_intent == QueryIntent.EXPLANATION:
            return self.scoring.causal_weight_explanation
        if query_intent == QueryIntent.PROCESS:
            return self.scoring.causal_weight_process
        return self.scoring.causal_weight_unknown

    def _directness_weight(self, query_intent: QueryIntent) -> float:
        if query_intent == QueryIntent.FACT:
            return self.scoring.directness_weight_fact
        if query_intent == QueryIntent.EXPLANATION:
            return self.scoring.directness_weight_explanation
        if query_intent == QueryIntent.PROCESS:
            return self.scoring.directness_weight_process
        return self.scoring.directness_weight_unknown

    def _coherence_weight(self, query_intent: QueryIntent) -> float:
        if query_intent == QueryIntent.FACT:
            return self.scoring.coherence_weight_fact
        if query_intent == QueryIntent.EXPLANATION:
            return self.scoring.coherence_weight_explanation
        if query_intent == QueryIntent.PROCESS:
            return self.scoring.coherence_weight_process
        return self.scoring.coherence_weight_unknown

    def _cluster_edge_types(self, query_intent: QueryIntent) -> set[RelationshipType]:
        edge_types = {RelationshipType.SUPPORTS}
        if query_intent in {QueryIntent.EXPLANATION, QueryIntent.PROCESS}:
            edge_types.add(RelationshipType.CAUSES)
        return edge_types
