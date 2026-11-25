from __future__ import annotations

import hashlib
import math
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


PIXEL_MEMORY_DEFAULT_WIDTH = 64
PIXEL_MEMORY_DEFAULT_HEIGHT = 64
PIXEL_MEMORY_DECAY_RATE = 0.08
PIXEL_MEMORY_DIFFUSION_RATE = 0.16
PIXEL_MEMORY_REINFORCEMENT_RATE = 0.22
PIXEL_MEMORY_INHIBITION_RATE = 0.35
PIXEL_MEMORY_REGION_THRESHOLD = 0.08
PIXEL_MEMORY_LAYOUT_ITERATIONS = 16
PIXEL_REASONING_DOMINANCE_HIGH = 0.90
PIXEL_REASONING_DOMINANCE_LOW = 0.10
PIXEL_REASONING_STRONG_REGION_SHARE = 0.10
PIXEL_REASONING_STRONG_REGION_STRENGTH = 0.65
PIXEL_REASONING_STRONG_TEMPORAL_GRADIENT = 0.08


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _tokenize(value: object) -> List[str]:
    normalized = re.sub(r"[^a-z0-9\s]", " ", str(value or "").lower())
    return [
        token
        for token in re.sub(r"\s+", " ", normalized).strip().split()
        if token and token not in _STOPWORDS
    ]


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / float(len(left_set | right_set))


def _stable_unit_pair(value: str) -> Tuple[float, float]:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    x_raw = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    y_raw = int.from_bytes(digest[8:16], "big") / float(2**64 - 1)
    return x_raw, y_raw


@dataclass
class PixelMemoryCell:
    activation_value: float = 0.0
    decay_value: float = PIXEL_MEMORY_DECAY_RATE
    importance_trace: float = 0.0
    recency_trace: float = 0.0
    inhibition_value: float = 0.0
    contributing_concepts: List[str] = field(default_factory=list)
    contributing_weights: Dict[str, float] = field(default_factory=dict)

    def reinforce(self, concept_id: str, weight: float, importance: float, recency: float) -> None:
        self.activation_value = _clamp(self.activation_value + weight)
        self.importance_trace = _clamp(max(self.importance_trace, importance * weight))
        self.recency_trace = _clamp(max(self.recency_trace, recency * weight))
        if concept_id not in self.contributing_concepts:
            self.contributing_concepts.append(concept_id)
        self.contributing_weights[concept_id] = round(
            self.contributing_weights.get(concept_id, 0.0) + weight,
            6,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "activation_value": round(self.activation_value, 4),
            "decay_value": round(self.decay_value, 4),
            "importance_trace": round(self.importance_trace, 4),
            "recency_trace": round(self.recency_trace, 4),
            "inhibition_value": round(self.inhibition_value, 4),
            "contributing_concepts": list(self.contributing_concepts),
            "contributing_weights": {
                key: round(float(value), 4)
                for key, value in sorted(self.contributing_weights.items())
            },
        }


@dataclass(frozen=True)
class PixelProjectionConcept:
    concept_id: str
    label: str
    terms: List[str] = field(default_factory=list)
    abstraction_level: int = 1
    importance: float = 0.5
    prediction_accuracy: float = 0.0
    uncertainty: float = 1.0
    temporal_links: List[str] = field(default_factory=list)
    hierarchy_links: List[str] = field(default_factory=list)
    contradiction_links: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def all_terms(self) -> List[str]:
        terms = list(self.terms) if self.terms else _tokenize(self.label)
        for item in self.metadata.values():
            if isinstance(item, str):
                terms.extend(_tokenize(item))
        return sorted(set(terms))

    def to_dict(self) -> Dict[str, object]:
        return {
            "concept_id": self.concept_id,
            "label": self.label,
            "terms": list(self.terms),
            "abstraction_level": int(self.abstraction_level),
            "importance": round(float(self.importance), 4),
            "prediction_accuracy": round(float(self.prediction_accuracy), 4),
            "uncertainty": round(float(self.uncertainty), 4),
            "temporal_links": list(self.temporal_links),
            "hierarchy_links": list(self.hierarchy_links),
            "contradiction_links": list(self.contradiction_links),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PixelRelationship:
    source_concept_id: str
    target_concept_id: str
    relationship_type: str
    strength: float = 1.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_concept_id": self.source_concept_id,
            "target_concept_id": self.target_concept_id,
            "relationship_type": self.relationship_type,
            "strength": round(float(self.strength), 4),
        }


@dataclass(frozen=True)
class PixelConceptPlacement:
    concept_id: str
    x: int
    y: int
    footprint_radius: int
    abstraction_level: int
    importance: float
    prediction_accuracy: float
    uncertainty: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "concept_id": self.concept_id,
            "x": self.x,
            "y": self.y,
            "footprint_radius": self.footprint_radius,
            "abstraction_level": self.abstraction_level,
            "importance": round(float(self.importance), 4),
            "prediction_accuracy": round(float(self.prediction_accuracy), 4),
            "uncertainty": round(float(self.uncertainty), 4),
        }


@dataclass(frozen=True)
class PixelActiveRegion:
    region_id: int
    cells: List[Tuple[int, int]]
    total_activation: float
    peak_activation: float
    centroid_x: float
    centroid_y: float
    dominant_contributing_concepts: List[Tuple[str, float]]
    abstraction_mix: Dict[int, float]
    importance_mix: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "region_id": self.region_id,
            "cells": [{"x": x, "y": y} for x, y in self.cells],
            "total_activation": round(float(self.total_activation), 4),
            "peak_activation": round(float(self.peak_activation), 4),
            "centroid_x": round(float(self.centroid_x), 2),
            "centroid_y": round(float(self.centroid_y), 2),
            "dominant_contributing_concepts": [
                {"concept_id": concept_id, "weight": round(float(weight), 4)}
                for concept_id, weight in self.dominant_contributing_concepts
            ],
            "abstraction_mix": {
                str(level): round(float(weight), 4)
                for level, weight in sorted(self.abstraction_mix.items())
            },
            "importance_mix": {
                key: round(float(value), 4)
                for key, value in sorted(self.importance_mix.items())
            },
        }


@dataclass(frozen=True)
class PixelReadback:
    active_regions: List[PixelActiveRegion]
    dominant_regions: List[Dict[str, object]]
    region_strengths: List[Dict[str, object]]
    dominant_concepts: List[Dict[str, object]]
    multi_constraint_concepts: List[Dict[str, object]]
    supporting_concepts: List[Dict[str, object]]
    contradictory_concepts: List[Dict[str, object]]
    temporal_next_concepts: List[Dict[str, object]]
    spatial_confidence: float
    reconstructed_summary: str
    confidence: float
    dominance_gap: float = 0.0
    region_coherence: float = 0.0
    debug_log: List[str] = field(default_factory=list)
    heatmap_lines: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "active_regions": [region.to_dict() for region in self.active_regions],
            "dominant_regions": list(self.dominant_regions),
            "region_strengths": list(self.region_strengths),
            "dominant_concepts": list(self.dominant_concepts),
            "multi_constraint_concepts": list(self.multi_constraint_concepts),
            "supporting_concepts": list(self.supporting_concepts),
            "contradictory_concepts": list(self.contradictory_concepts),
            "temporal_next_concepts": list(self.temporal_next_concepts),
            "spatial_confidence": round(float(self.spatial_confidence), 4),
            "reconstructed_summary": self.reconstructed_summary,
            "confidence": round(float(self.confidence), 4),
            "dominance_gap": round(float(self.dominance_gap), 4),
            "region_coherence": round(float(self.region_coherence), 4),
            "debug_log": list(self.debug_log),
            "heatmap_lines": list(self.heatmap_lines),
        }


def detect_reasoning_mode(pixel_output: PixelReadback | Dict[str, object]) -> str:
    if isinstance(pixel_output, PixelReadback):
        dominance_gap = float(pixel_output.dominance_gap)
        dominant_region_count = len(pixel_output.dominant_regions)
        temporal_next = pixel_output.temporal_next_concepts
    else:
        dominance_gap = float(pixel_output.get("dominance_gap", 0.0) or 0.0)
        dominant_regions = pixel_output.get("dominant_regions", [])
        temporal_next = pixel_output.get("temporal_next_concepts", [])
        dominant_region_count = len(dominant_regions) if isinstance(dominant_regions, list) else 0
    if dominance_gap > PIXEL_REASONING_DOMINANCE_HIGH:
        return "single_answer"
    if dominance_gap < PIXEL_REASONING_DOMINANCE_LOW and dominant_region_count >= 2:
        return "multi_constraint"
    if temporal_next:
        return "temporal"
    return "hybrid"


@dataclass(frozen=True)
class PixelReasoningTrace:
    reasoning_mode: str
    selected_output_type: str
    selected_concepts: List[Dict[str, object]]
    constraints: List[Dict[str, object]]
    temporal_sequence: List[Dict[str, object]]
    suppressed_concepts: List[Dict[str, object]]
    conflict_report: List[Dict[str, object]]
    explanation: str
    dominance_gap: float
    active_region_count: int
    strong_region_count: int
    debug_log: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "reasoning_mode": self.reasoning_mode,
            "selected_output_type": self.selected_output_type,
            "selected_concepts": list(self.selected_concepts),
            "constraints": list(self.constraints),
            "temporal_sequence": list(self.temporal_sequence),
            "suppressed_concepts": list(self.suppressed_concepts),
            "conflict_report": list(self.conflict_report),
            "explanation": self.explanation,
            "dominance_gap": round(float(self.dominance_gap), 4),
            "active_region_count": int(self.active_region_count),
            "strong_region_count": int(self.strong_region_count),
            "debug_log": list(self.debug_log),
        }


class PixelReasoningEngine:
    """Interprets Pixel Memory readback into reasoning behavior."""

    def __init__(
        self,
        *,
        dominance_high: float = PIXEL_REASONING_DOMINANCE_HIGH,
        dominance_low: float = PIXEL_REASONING_DOMINANCE_LOW,
        strong_region_share: float = PIXEL_REASONING_STRONG_REGION_SHARE,
        strong_region_strength: float = PIXEL_REASONING_STRONG_REGION_STRENGTH,
        strong_temporal_gradient: float = PIXEL_REASONING_STRONG_TEMPORAL_GRADIENT,
    ) -> None:
        self.dominance_high = float(dominance_high)
        self.dominance_low = float(dominance_low)
        self.strong_region_share = float(strong_region_share)
        self.strong_region_strength = float(strong_region_strength)
        self.strong_temporal_gradient = float(strong_temporal_gradient)

    def reason(self, readback: PixelReadback) -> PixelReasoningTrace:
        strong_regions = self._strong_regions(readback)
        strong_temporal = [
            dict(item)
            for item in readback.temporal_next_concepts
            if float(item.get("score", item.get("gradient", 0.0)) or 0.0) >= self.strong_temporal_gradient
        ]
        conflict_report, suppressed = self._resolve_conflicts(readback)
        mode = detect_reasoning_mode(readback)
        if mode == "multi_constraint" and not readback.multi_constraint_concepts:
            mode = "temporal" if readback.temporal_next_concepts else "hybrid"
        if mode == "multi_constraint":
            selected = self._rank_constraints(readback, strong_regions)
            constraints = selected
            selected_output_type = "constraint_list"
        elif mode == "temporal":
            selected = strong_temporal or list(readback.temporal_next_concepts[:3])
            constraints = []
            selected_output_type = "next_step_concepts"
        elif mode == "single_answer":
            selected = list(readback.dominant_concepts[:1])
            constraints = []
            selected_output_type = "single_concept"
        else:
            selected = list(readback.dominant_concepts[:3])
            constraints = list(readback.multi_constraint_concepts)
            selected_output_type = "hybrid_context"
        explanation = self._explain(readback, mode, selected_output_type, strong_regions)
        return PixelReasoningTrace(
            reasoning_mode=mode,
            selected_output_type=selected_output_type,
            selected_concepts=selected,
            constraints=constraints,
            temporal_sequence=strong_temporal or list(readback.temporal_next_concepts[:3]),
            suppressed_concepts=suppressed,
            conflict_report=conflict_report,
            explanation=explanation,
            dominance_gap=readback.dominance_gap,
            active_region_count=len(readback.active_regions),
            strong_region_count=len(strong_regions),
            debug_log=[
                f"reasoning_mode={mode}",
                f"dominance_gap={readback.dominance_gap:.4f}",
                f"active_regions={len(readback.active_regions)}",
                f"strong_regions={len(strong_regions)}",
                f"selected_output_type={selected_output_type}",
            ],
        )

    def _strong_regions(self, readback: PixelReadback) -> List[Dict[str, object]]:
        strong: List[Dict[str, object]] = []
        for region in readback.region_strengths:
            activation_share = float(region.get("activation_share", 0.0) or 0.0)
            activation_strength = float(region.get("activation_strength", 0.0) or 0.0)
            if activation_share >= self.strong_region_share or activation_strength >= self.strong_region_strength:
                strong.append(dict(region))
        return strong

    def _rank_constraints(self, readback: PixelReadback, strong_regions: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        region_strengths = [float(region.get("total_activation", 0.0) or 0.0) for region in strong_regions]
        fallback_strength = max(region_strengths, default=0.0)
        ranked: List[Dict[str, object]] = []
        for index, concept in enumerate(readback.multi_constraint_concepts):
            item = dict(concept)
            item["constraint_rank"] = index + 1
            item["region_strength"] = round(region_strengths[index] if index < len(region_strengths) else fallback_strength, 4)
            ranked.append(item)
        return sorted(ranked, key=lambda item: (-float(item.get("region_strength", 0.0) or 0.0), int(item.get("constraint_rank", 999))))

    def _resolve_conflicts(self, readback: PixelReadback) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        if not readback.contradictory_concepts:
            return [], []
        winner = dict(readback.dominant_concepts[0]) if readback.dominant_concepts else {}
        winner_score = float(winner.get("score", 0.0) or 0.0)
        suppressed: List[Dict[str, object]] = []
        reports: List[Dict[str, object]] = []
        for concept in readback.contradictory_concepts:
            candidate = dict(concept)
            candidate_score = float(candidate.get("score", 0.0) or 0.0)
            if candidate.get("concept_id") == winner.get("concept_id"):
                continue
            suppressed.append(candidate)
            reports.append(
                {
                    "winner_concept_id": winner.get("concept_id", ""),
                    "suppressed_concept_id": candidate.get("concept_id", ""),
                    "winner_score": round(winner_score, 4),
                    "suppressed_score": round(candidate_score, 4),
                    "dominance_gap": round(float(readback.dominance_gap), 4),
                    "resolution": "pixel_dominance",
                }
            )
        return reports, suppressed

    def _explain(
        self,
        readback: PixelReadback,
        mode: str,
        selected_output_type: str,
        strong_regions: Sequence[Dict[str, object]],
    ) -> str:
        top_region = readback.region_strengths[0] if readback.region_strengths else {}
        region_text = (
            f"top region {top_region.get('region_id')} strength={top_region.get('activation_strength')} share={top_region.get('activation_share')}"
            if top_region
            else "no active top region"
        )
        return (
            f"Pixel reasoning selected {mode} ({selected_output_type}) from "
            f"{len(readback.active_regions)} active regions and {len(strong_regions)} strong regions; "
            f"dominance_gap={readback.dominance_gap:.4f}; {region_text}."
        )


class PixelMemoryCanvas:
    """
    Pixel Memory is an inspectable projection layer over BrAIn's canonical
    graph/concept stores. It never owns truth: concepts and relationships are
    projected into this 2D field, activated, read back, and compared with graph
    retrieval as an auxiliary reconstruction signal.
    """

    def __init__(
        self,
        *,
        width: int = PIXEL_MEMORY_DEFAULT_WIDTH,
        height: int = PIXEL_MEMORY_DEFAULT_HEIGHT,
    ) -> None:
        self.width = max(8, int(width))
        self.height = max(8, int(height))
        self.cells: List[List[PixelMemoryCell]] = [
            [PixelMemoryCell() for _ in range(self.width)]
            for _ in range(self.height)
        ]
        self.concepts: Dict[str, PixelProjectionConcept] = {}
        self.relationships: List[PixelRelationship] = []
        self.placements: Dict[str, PixelConceptPlacement] = {}
        self.debug_log: List[str] = []
        self.inhibition_events: List[str] = []

    def reset_field(self) -> None:
        self.cells = [
            [PixelMemoryCell() for _ in range(self.width)]
            for _ in range(self.height)
        ]
        self.inhibition_events = []

    def project_concepts(
        self,
        concepts: Sequence[PixelProjectionConcept],
        relationships: Sequence[PixelRelationship] = (),
    ) -> None:
        self.reset_field()
        self.concepts = {
            str(concept.concept_id): concept
            for concept in concepts
            if str(concept.concept_id).strip()
        }
        self.relationships = [
            relationship
            for relationship in relationships
            if relationship.source_concept_id in self.concepts
            and relationship.target_concept_id in self.concepts
        ]
        self.debug_log = [
            f"project_concepts count={len(self.concepts)} relationships={len(self.relationships)} canvas={self.width}x{self.height}"
        ]
        self.placements = self._layout_concepts()
        for placement in sorted(self.placements.values(), key=lambda item: item.concept_id):
            concept = self.concepts[placement.concept_id]
            self._seed_concept_region(concept, placement)
            self.debug_log.append(
                "placement "
                + f"concept_id={placement.concept_id} center=({placement.x},{placement.y}) "
                + f"radius={placement.footprint_radius} importance={round(placement.importance, 4)} "
                + f"abstraction={placement.abstraction_level}"
            )
        self.apply_inhibition()

    def update_concept(self, concept: PixelProjectionConcept) -> None:
        concepts = list(self.concepts.values())
        replaced = False
        for index, existing in enumerate(concepts):
            if existing.concept_id == concept.concept_id:
                concepts[index] = concept
                replaced = True
                break
        if not replaced:
            concepts.append(concept)
        self.project_concepts(concepts, self.relationships)

    def reinforce_concepts(self, concept_ids: Iterable[str], amount: float = PIXEL_MEMORY_REINFORCEMENT_RATE) -> None:
        for concept_id in concept_ids:
            placement = self.placements.get(str(concept_id))
            concept = self.concepts.get(str(concept_id))
            if placement is None or concept is None:
                continue
            for x, y, falloff in self._footprint_cells(placement):
                self.cells[y][x].reinforce(
                    concept.concept_id,
                    _clamp(amount * falloff),
                    _clamp(concept.importance),
                    1.0,
                )

    def activate_query(self, query_text: str) -> List[str]:
        query_terms = _tokenize(query_text)
        activated: List[str] = []
        for concept in self.concepts.values():
            similarity = _jaccard(query_terms, concept.all_terms())
            if similarity <= 0.0:
                continue
            amount = _clamp(0.18 + similarity + (concept.importance * 0.2))
            self.reinforce_concepts([concept.concept_id], amount=amount)
            activated.append(concept.concept_id)
            self.debug_log.append(
                f"query_activation concept_id={concept.concept_id} similarity={round(similarity, 4)} amount={round(amount, 4)}"
            )
        self.diffuse_activation()
        self.decay()
        self.apply_inhibition()
        return activated

    def diffuse_activation(self, rate: float = PIXEL_MEMORY_DIFFUSION_RATE) -> None:
        next_values = [[self.cells[y][x].activation_value for x in range(self.width)] for y in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                value = self.cells[y][x].activation_value
                if value <= 0.0:
                    continue
                neighbors = list(self._neighbors(x, y))
                spread_each = (value * rate) / float(max(1, len(neighbors)))
                next_values[y][x] = max(0.0, next_values[y][x] - (spread_each * len(neighbors)))
                for nx, ny in neighbors:
                    next_values[ny][nx] = _clamp(next_values[ny][nx] + spread_each)
                    self._merge_cell_sources(source=self.cells[y][x], target=self.cells[ny][nx], scale=0.4)
        for y in range(self.height):
            for x in range(self.width):
                self.cells[y][x].activation_value = _clamp(next_values[y][x])
        self.debug_log.append(f"diffuse rate={round(rate, 4)}")

    def decay(self, rate: float = PIXEL_MEMORY_DECAY_RATE) -> None:
        for row in self.cells:
            for cell in row:
                local_rate = max(0.0, min(0.95, rate + cell.decay_value * 0.1))
                cell.activation_value = _clamp(cell.activation_value * (1.0 - local_rate))
                cell.recency_trace = _clamp(cell.recency_trace * (1.0 - (local_rate * 0.5)))
                cell.inhibition_value = _clamp(cell.inhibition_value * 0.92)
        self.debug_log.append(f"decay rate={round(rate, 4)}")

    def apply_inhibition(self, rate: float = PIXEL_MEMORY_INHIBITION_RATE) -> List[str]:
        events: List[str] = []
        for relationship in self.relationships:
            if relationship.relationship_type.lower() not in {"contradicts", "contradict", "contradiction"}:
                continue
            left = self.placements.get(relationship.source_concept_id)
            right = self.placements.get(relationship.target_concept_id)
            if left is None or right is None:
                continue
            distance = math.hypot(left.x - right.x, left.y - right.y)
            target_distance = max(8.0, min(self.width, self.height) * 0.35)
            if distance >= target_distance:
                continue
            pressure = _clamp((target_distance - distance) / target_distance) * _clamp(relationship.strength) * rate
            for placement in (left, right):
                for x, y, falloff in self._footprint_cells(placement):
                    cell = self.cells[y][x]
                    cell.activation_value = _clamp(cell.activation_value * (1.0 - (pressure * falloff)))
                    cell.inhibition_value = _clamp(cell.inhibition_value + pressure * falloff)
            event = (
                f"inhibit {relationship.source_concept_id}<->{relationship.target_concept_id} "
                + f"distance={round(distance, 2)} pressure={round(pressure, 4)}"
            )
            events.append(event)
        self.inhibition_events.extend(events)
        self.debug_log.extend(events)
        return events

    def detect_active_regions(self, threshold: float = PIXEL_MEMORY_REGION_THRESHOLD) -> List[PixelActiveRegion]:
        visited = [[False for _ in range(self.width)] for _ in range(self.height)]
        regions: List[PixelActiveRegion] = []
        for y in range(self.height):
            for x in range(self.width):
                if visited[y][x] or self.cells[y][x].activation_value < threshold:
                    continue
                cells: List[Tuple[int, int]] = []
                queue = deque([(x, y)])
                visited[y][x] = True
                while queue:
                    cx, cy = queue.popleft()
                    cells.append((cx, cy))
                    for nx, ny in self._neighbors(cx, cy):
                        if visited[ny][nx] or self.cells[ny][nx].activation_value < threshold:
                            continue
                        visited[ny][nx] = True
                        queue.append((nx, ny))
                regions.append(self._build_active_region(len(regions) + 1, cells))
        regions.sort(key=lambda item: (-item.total_activation, -item.peak_activation, item.region_id))
        return [
            PixelActiveRegion(
                region_id=index,
                cells=region.cells,
                total_activation=region.total_activation,
                peak_activation=region.peak_activation,
                centroid_x=region.centroid_x,
                centroid_y=region.centroid_y,
                dominant_contributing_concepts=region.dominant_contributing_concepts,
                abstraction_mix=region.abstraction_mix,
                importance_mix=region.importance_mix,
            )
            for index, region in enumerate(regions, start=1)
        ]

    def readback(self, max_regions: int = 6) -> PixelReadback:
        regions = self.detect_active_regions()[:max_regions]
        concept_scores: Dict[str, float] = {}
        for region in regions:
            for concept_id, weight in region.dominant_contributing_concepts:
                concept_scores[concept_id] = concept_scores.get(concept_id, 0.0) + weight * region.total_activation
        ranked = sorted(concept_scores.items(), key=lambda item: (-item[1], item[0]))
        dominant = [self._concept_readback_dict(concept_id, score) for concept_id, score in ranked[:5]]
        supporting = [self._concept_readback_dict(concept_id, score) for concept_id, score in ranked[5:10]]
        dominant_ids = {str(item.get("concept_id", "")) for item in dominant}
        contradictory_ids = {
            relationship.target_concept_id
            for relationship in self.relationships
            if relationship.source_concept_id in dominant_ids
            and relationship.relationship_type.lower() in {"contradicts", "contradict", "contradiction"}
        } | {
            relationship.source_concept_id
            for relationship in self.relationships
            if relationship.target_concept_id in dominant_ids
            and relationship.relationship_type.lower() in {"contradicts", "contradict", "contradiction"}
        }
        contradictory = [
            self._concept_readback_dict(concept_id, concept_scores.get(concept_id, 0.0))
            for concept_id in sorted(contradictory_ids)
            if concept_id in self.concepts
        ][:5]
        total_score = sum(score for _, score in ranked)
        top_score = ranked[0][1] if ranked else 0.0
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        dominance_gap = _clamp((top_score - second_score) / max(1.0, top_score + second_score))
        total_region_activation = sum(region.total_activation for region in regions)
        top_region = regions[0] if regions else None
        activation_share = (
            _clamp(top_region.total_activation / max(0.001, total_region_activation))
            if top_region is not None
            else 0.0
        )
        activation_strength = (
            _clamp(top_region.total_activation / max(1.0, float(len(top_region.cells)) * 0.3))
            if top_region is not None
            else 0.0
        )
        region_weight_total = (
            sum(float(weight) for _, weight in top_region.dominant_contributing_concepts)
            if top_region is not None
            else 0.0
        )
        region_coherence = (
            _clamp(float(top_region.dominant_contributing_concepts[0][1]) / max(0.001, region_weight_total))
            if top_region is not None and top_region.dominant_contributing_concepts
            else 0.0
        )
        spatial_confidence = _clamp(
            (activation_strength * 0.35)
            + (activation_share * 0.25)
            + (region_coherence * 0.25)
            + (dominance_gap * 0.15)
        )
        confidence = _clamp(sum(score for _, score in ranked[:3]) / max(1.0, total_score or 1.0))
        labels = [str(item.get("label", "")).strip() for item in dominant if str(item.get("label", "")).strip()]
        summary = "Pixel Memory readback: " + "; ".join(labels[:3]) if labels else "Pixel Memory readback: no stable dominant concept."
        region_strengths = [
            {
                "region_id": region.region_id,
                "total_activation": round(float(region.total_activation), 4),
                "peak_activation": round(float(region.peak_activation), 4),
                "activation_share": round(float(region.total_activation / max(0.001, total_region_activation)), 4),
                "activation_strength": round(float(region.total_activation / max(1.0, float(len(region.cells)) * 0.3)), 4),
                "centroid_x": round(float(region.centroid_x), 2),
                "centroid_y": round(float(region.centroid_y), 2),
            }
            for region in regions
        ]
        multi_constraint_ids: set[str] = set()
        constraint_like_ids = {
            concept_id
            for concept_id, concept in self.concepts.items()
            if "constraint" in str(concept.metadata.get("state_type", "")).lower()
            or "constraint" in str(concept.metadata.get("entry_type", "")).lower()
            or "constraint" in concept.label.lower()
            or "constraint" in {term.lower() for term in concept.terms}
        }
        strong_region_count = sum(
            1
            for region in regions
            if (region.total_activation / max(0.001, total_region_activation)) >= 0.1
            and (region.total_activation / max(1.0, float(len(region.cells)) * 0.3)) >= 0.65
        )
        preserve_multi_region_constraints = len(constraint_like_ids) >= 2 and strong_region_count >= 2
        if not contradictory_ids and len(regions) > 1 and preserve_multi_region_constraints:
            top_activation = max(0.001, regions[0].total_activation)
            for region in regions[:4]:
                activation_share_for_region = region.total_activation / max(0.001, total_region_activation)
                activation_strength_for_region = region.total_activation / max(1.0, float(len(region.cells)) * 0.3)
                if (
                    region.total_activation < top_activation * 0.55
                    and activation_share_for_region < 0.1
                    and activation_strength_for_region < 0.65
                ):
                    continue
                for concept_id, _weight in region.dominant_contributing_concepts[:2]:
                    if not preserve_multi_region_constraints or concept_id in constraint_like_ids:
                        multi_constraint_ids.add(concept_id)
        multi_constraint_concepts = [
            self._concept_readback_dict(concept_id, concept_scores.get(concept_id, 0.0))
            for concept_id in sorted(multi_constraint_ids, key=lambda concept_id: (-concept_scores.get(concept_id, 0.0), concept_id))
            if concept_id in self.concepts
        ][:8]
        if multi_constraint_concepts:
            constraint_labels = [
                str(item.get("label", "")).strip()
                for item in multi_constraint_concepts
                if str(item.get("label", "")).strip()
            ]
            summary = "Pixel Memory multi-constraint readback: " + "; ".join(constraint_labels[:4])
            top_region_strength = (
                float(region_strengths[0].get("activation_strength", 0.0) or 0.0)
                if region_strengths
                else 0.0
            )
            second_region_strength = (
                float(region_strengths[1].get("activation_strength", 0.0) or 0.0)
                if len(region_strengths) > 1
                else 0.0
            )
            dominance_gap = _clamp(
                abs(top_region_strength - second_region_strength)
                / max(1.0, top_region_strength + second_region_strength)
            )
        return PixelReadback(
            active_regions=regions,
            dominant_regions=[region.to_dict() for region in regions[:3]],
            region_strengths=region_strengths,
            dominant_concepts=dominant,
            multi_constraint_concepts=multi_constraint_concepts,
            supporting_concepts=supporting,
            contradictory_concepts=contradictory,
            temporal_next_concepts=self._temporal_next_concepts(dominant_ids, concept_scores),
            spatial_confidence=round(spatial_confidence, 4),
            reconstructed_summary=summary,
            confidence=round(confidence, 4),
            dominance_gap=round(dominance_gap, 4),
            region_coherence=round(region_coherence, 4),
            debug_log=list(self.debug_log[-80:]),
            heatmap_lines=self.heatmap_lines(),
        )

    def heatmap_lines(self, rows: int = 16, cols: int = 32) -> List[str]:
        chars = " .:-=+*#%@"
        y_step = max(1, math.ceil(self.height / max(1, rows)))
        x_step = max(1, math.ceil(self.width / max(1, cols)))
        lines: List[str] = []
        for y0 in range(0, self.height, y_step):
            parts: List[str] = []
            for x0 in range(0, self.width, x_step):
                values = [
                    self.cells[y][x].activation_value
                    for y in range(y0, min(self.height, y0 + y_step))
                    for x in range(x0, min(self.width, x0 + x_step))
                ]
                average = sum(values) / float(max(1, len(values)))
                parts.append(chars[min(len(chars) - 1, int(round(average * (len(chars) - 1))))])
            lines.append("".join(parts))
        return lines[:rows]

    def to_debug_dict(self) -> Dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "concept_placements": [item.to_dict() for item in sorted(self.placements.values(), key=lambda p: p.concept_id)],
            "relationships": [item.to_dict() for item in self.relationships],
            "inhibition_events": list(self.inhibition_events),
            "heatmap_lines": self.heatmap_lines(),
            "debug_log": list(self.debug_log[-120:]),
        }

    def _layout_concepts(self) -> Dict[str, PixelConceptPlacement]:
        ids = sorted(self.concepts)
        if not ids:
            return {}
        positions: Dict[str, Tuple[float, float]] = {}
        for concept_id in ids:
            ux, uy = _stable_unit_pair(concept_id + ":" + self.concepts[concept_id].label)
            positions[concept_id] = (2.0 + ux * (self.width - 5.0), 2.0 + uy * (self.height - 5.0))
        term_cache = {concept_id: self.concepts[concept_id].all_terms() for concept_id in ids}
        relationship_lookup: Dict[Tuple[str, str], Tuple[str, float]] = {}
        for relationship in self.relationships:
            key = tuple(sorted((relationship.source_concept_id, relationship.target_concept_id)))
            relationship_lookup[key] = (relationship.relationship_type.lower(), _clamp(relationship.strength))
        for _ in range(PIXEL_MEMORY_LAYOUT_ITERATIONS):
            deltas = {concept_id: [0.0, 0.0] for concept_id in ids}
            for index, left_id in enumerate(ids):
                for right_id in ids[index + 1 :]:
                    lx, ly = positions[left_id]
                    rx, ry = positions[right_id]
                    dx = rx - lx
                    dy = ry - ly
                    distance = max(0.001, math.hypot(dx, dy))
                    semantic = _jaccard(term_cache[left_id], term_cache[right_id])
                    relationship_type, strength = relationship_lookup.get(tuple(sorted((left_id, right_id))), ("", 0.0))
                    if relationship_type in {"contradicts", "contradict", "contradiction"}:
                        target = min(self.width, self.height) * 0.42
                        force = (target - distance) * -0.018 * max(0.8, strength)
                    else:
                        linked = strength if relationship_type else 0.0
                        hierarchy = 1.0 if right_id in self.concepts[left_id].hierarchy_links or left_id in self.concepts[right_id].hierarchy_links else 0.0
                        temporal = 1.0 if right_id in self.concepts[left_id].temporal_links or left_id in self.concepts[right_id].temporal_links else 0.0
                        target = 4.0 + (1.0 - semantic) * 18.0
                        if linked or hierarchy or temporal:
                            target = min(target, 5.0 + (1.0 - max(linked, hierarchy, temporal)) * 8.0)
                        force = (distance - target) * 0.012 * max(semantic, linked, hierarchy * 0.8, temporal * 0.65, 0.1)
                    ux = dx / distance
                    uy = dy / distance
                    deltas[left_id][0] += ux * force
                    deltas[left_id][1] += uy * force
                    deltas[right_id][0] -= ux * force
                    deltas[right_id][1] -= uy * force
            for concept_id in ids:
                x, y = positions[concept_id]
                positions[concept_id] = (
                    min(self.width - 2.0, max(1.0, x + deltas[concept_id][0])),
                    min(self.height - 2.0, max(1.0, y + deltas[concept_id][1])),
                )
        placements: Dict[str, PixelConceptPlacement] = {}
        occupied: set[Tuple[int, int]] = set()
        for concept_id in ids:
            concept = self.concepts[concept_id]
            x, y = positions[concept_id]
            ix = int(round(x))
            iy = int(round(y))
            while (ix, iy) in occupied and ix < self.width - 2:
                ix += 1
            occupied.add((ix, iy))
            radius = 2 if concept.importance >= 0.72 else 1
            placements[concept_id] = PixelConceptPlacement(
                concept_id=concept_id,
                x=ix,
                y=iy,
                footprint_radius=radius,
                abstraction_level=max(1, int(concept.abstraction_level)),
                importance=_clamp(concept.importance),
                prediction_accuracy=_clamp(concept.prediction_accuracy),
                uncertainty=_clamp(concept.uncertainty),
            )
        return placements

    def _seed_concept_region(self, concept: PixelProjectionConcept, placement: PixelConceptPlacement) -> None:
        base = _clamp(0.18 + (concept.importance * 0.52) + (concept.prediction_accuracy * 0.18) - (concept.uncertainty * 0.08))
        for x, y, falloff in self._footprint_cells(placement):
            self.cells[y][x].reinforce(
                concept.concept_id,
                base * falloff,
                _clamp(concept.importance),
                float(concept.metadata.get("recency", 0.5) or 0.5),
            )

    def _footprint_cells(self, placement: PixelConceptPlacement) -> List[Tuple[int, int, float]]:
        cells: List[Tuple[int, int, float]] = []
        radius = max(1, int(placement.footprint_radius))
        for y in range(max(0, placement.y - radius), min(self.height, placement.y + radius + 1)):
            for x in range(max(0, placement.x - radius), min(self.width, placement.x + radius + 1)):
                distance = math.hypot(x - placement.x, y - placement.y)
                if distance > radius + 0.25:
                    continue
                cells.append((x, y, _clamp(1.0 - (distance / (radius + 1.0)), 0.12, 1.0)))
        return cells

    def _neighbors(self, x: int, y: int) -> Iterable[Tuple[int, int]]:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    yield nx, ny

    @staticmethod
    def _merge_cell_sources(source: PixelMemoryCell, target: PixelMemoryCell, scale: float) -> None:
        for concept_id in source.contributing_concepts:
            if concept_id not in target.contributing_concepts:
                target.contributing_concepts.append(concept_id)
            target.contributing_weights[concept_id] = round(
                target.contributing_weights.get(concept_id, 0.0)
                + (source.contributing_weights.get(concept_id, 0.0) * scale),
                6,
            )

    def _build_active_region(self, region_id: int, cells: Sequence[Tuple[int, int]]) -> PixelActiveRegion:
        total_activation = sum(self.cells[y][x].activation_value for x, y in cells)
        peak_activation = max((self.cells[y][x].activation_value for x, y in cells), default=0.0)
        if total_activation <= 0.0:
            centroid_x = sum(x for x, _ in cells) / float(max(1, len(cells)))
            centroid_y = sum(y for _, y in cells) / float(max(1, len(cells)))
        else:
            centroid_x = sum(x * self.cells[y][x].activation_value for x, y in cells) / total_activation
            centroid_y = sum(y * self.cells[y][x].activation_value for x, y in cells) / total_activation
        concept_weights: Dict[str, float] = {}
        abstraction_mix: Dict[int, float] = {}
        importance_mix = {"low": 0.0, "medium": 0.0, "high": 0.0}
        for x, y in cells:
            cell = self.cells[y][x]
            for concept_id, weight in cell.contributing_weights.items():
                weighted = weight * max(0.01, cell.activation_value)
                concept_weights[concept_id] = concept_weights.get(concept_id, 0.0) + weighted
                concept = self.concepts.get(concept_id)
                if concept is None:
                    continue
                level = max(1, int(concept.abstraction_level))
                abstraction_mix[level] = abstraction_mix.get(level, 0.0) + weighted
                bucket = "high" if concept.importance >= 0.72 else "medium" if concept.importance >= 0.4 else "low"
                importance_mix[bucket] += weighted
        dominant = sorted(concept_weights.items(), key=lambda item: (-item[1], item[0]))[:5]
        return PixelActiveRegion(
            region_id=region_id,
            cells=list(cells),
            total_activation=total_activation,
            peak_activation=peak_activation,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            dominant_contributing_concepts=dominant,
            abstraction_mix=abstraction_mix,
            importance_mix=importance_mix,
        )

    def _concept_readback_dict(self, concept_id: str, score: float) -> Dict[str, object]:
        concept = self.concepts.get(concept_id)
        placement = self.placements.get(concept_id)
        if concept is None:
            return {"concept_id": concept_id, "score": round(float(score), 4)}
        return {
            "concept_id": concept_id,
            "label": concept.label,
            "score": round(float(score), 4),
            "abstraction_level": int(concept.abstraction_level),
            "importance": round(float(concept.importance), 4),
            "prediction_accuracy": round(float(concept.prediction_accuracy), 4),
            "uncertainty": round(float(concept.uncertainty), 4),
            "placement": None if placement is None else {"x": placement.x, "y": placement.y},
            "metadata": dict(concept.metadata),
        }

    def _temporal_next_concepts(self, dominant_ids: set[str], concept_scores: Dict[str, float]) -> List[Dict[str, object]]:
        candidates: Dict[str, float] = {}
        for relationship in self.relationships:
            relationship_type = relationship.relationship_type.lower()
            if relationship_type not in {"temporal", "causes", "next"}:
                continue
            if relationship.source_concept_id not in dominant_ids:
                continue
            source = self.placements.get(relationship.source_concept_id)
            target = self.placements.get(relationship.target_concept_id)
            if source is None or target is None:
                continue
            distance = max(1.0, math.hypot(target.x - source.x, target.y - source.y))
            gradient = concept_scores.get(relationship.source_concept_id, 0.0) * _clamp(relationship.strength) / distance
            candidates[relationship.target_concept_id] = max(candidates.get(relationship.target_concept_id, 0.0), gradient)
        return [
            self._concept_readback_dict(concept_id, score)
            for concept_id, score in sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[:5]
        ]
