from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from brain_codec import BrainCodec
from brain_input_v1 import BrainInputV1
from brain_reasoning_engine import (
    DEFAULT_OUTPUT_DIR,
    BrainNativeReasoningEngine,
    BrainReasoningRuntime,
    build_reasoning_runtime,
)


@dataclass(frozen=True)
class ReasoningLink:
    link_id: int
    relation_type: str
    antecedent: BrainInputV1
    consequent: BrainInputV1
    created_sequence: int
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    reliability: float = 0.5
    temporal_score: float = 0.0
    last_used_sequence: int = 0
    active: bool = True
    split_from_link_id: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "link_id": self.link_id,
            "relation_type": self.relation_type,
            "antecedent": self.antecedent.to_dict(),
            "consequent": self.consequent.to_dict(),
            "created_sequence": self.created_sequence,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "reliability": round(self.reliability, 4),
            "temporal_score": round(self.temporal_score, 4),
            "last_used_sequence": self.last_used_sequence,
            "active": self.active,
            "notes": list(self.notes),
        }
        if self.split_from_link_id is not None:
            payload["split_from_link_id"] = self.split_from_link_id
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ReasoningLink":
        return cls(
            link_id=int(data.get("link_id", 0)),
            relation_type=str(data.get("relation_type", "")),
            antecedent=BrainInputV1.from_dict(data.get("antecedent", {}) if isinstance(data.get("antecedent", {}), dict) else {}),
            consequent=BrainInputV1.from_dict(data.get("consequent", {}) if isinstance(data.get("consequent", {}), dict) else {}),
            created_sequence=int(data.get("created_sequence", 0)),
            usage_count=int(data.get("usage_count", 0)),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            reliability=float(data.get("reliability", 0.5)),
            temporal_score=float(data.get("temporal_score", 0.0)),
            last_used_sequence=int(data.get("last_used_sequence", 0)),
            active=bool(data.get("active", True)),
            split_from_link_id=int(data["split_from_link_id"]) if data.get("split_from_link_id") is not None else None,
            notes=[str(item) for item in data.get("notes", [])] if isinstance(data.get("notes", []), list) else [],
        )


@dataclass(frozen=True)
class ChainStep:
    link_id: int
    relation_type: str
    antecedent: BrainInputV1
    consequent: BrainInputV1
    match_score: float
    reliability: float
    confidence: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "link_id": self.link_id,
            "relation_type": self.relation_type,
            "antecedent": self.antecedent.to_dict(),
            "consequent": self.consequent.to_dict(),
            "match_score": round(self.match_score, 4),
            "reliability": round(self.reliability, 4),
            "confidence": round(self.confidence, 4),
        }


@dataclass(frozen=True)
class ReasoningPath:
    steps: List[ChainStep]
    final_outcome: BrainInputV1
    confidence: float

    @property
    def depth(self) -> int:
        return len(self.steps)

    def to_dict(self) -> Dict[str, object]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "final_outcome": self.final_outcome.to_dict(),
            "confidence": round(self.confidence, 4),
            "depth": self.depth,
        }


@dataclass(frozen=True)
class ReasoningPrediction:
    input_brain: BrainInputV1
    candidate_paths: List[ReasoningPath]
    selected_path: Optional[ReasoningPath]
    confidence_score: float
    final_predicted_outcome: Optional[BrainInputV1]

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "input_brain": self.input_brain.to_dict(),
            "candidate_paths": [path.to_dict() for path in self.candidate_paths],
            "confidence_score": round(self.confidence_score, 4),
        }
        if self.selected_path is not None:
            payload["selected_path"] = self.selected_path.to_dict()
        if self.final_predicted_outcome is not None:
            payload["final_predicted_outcome"] = self.final_predicted_outcome.to_dict()
        return payload


class BrainReasoningLinkRegistry:
    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)
        self.links: Dict[int, ReasoningLink] = {}
        self.fingerprint_to_link_id: Dict[str, int] = {}
        self.next_link_id = 1
        self._load()

    def _load(self) -> None:
        if not self.registry_path.exists():
            self.links = {}
            self.fingerprint_to_link_id = {}
            self.next_link_id = 1
            return
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded_links: Dict[int, ReasoningLink] = {}
        loaded_fingerprints: Dict[str, int] = {}
        if isinstance(data, dict):
            self.next_link_id = int(data.get("next_link_id", 1))
            for item in data.get("links", []):
                if not isinstance(item, dict):
                    continue
                link = ReasoningLink.from_dict(item)
                if link.link_id <= 0:
                    continue
                loaded_links[link.link_id] = link
                loaded_fingerprints[self._fingerprint(link.relation_type, link.antecedent, link.consequent)] = link.link_id
        self.links = loaded_links
        self.fingerprint_to_link_id = loaded_fingerprints
        self.next_link_id = max([self.next_link_id] + [link_id + 1 for link_id in self.links], default=1)

    def save(self) -> None:
        payload = {
            "next_link_id": self.next_link_id,
            "links": [self.links[link_id].to_dict() for link_id in sorted(self.links)],
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def all_links(self) -> List[ReasoningLink]:
        return [self.links[link_id] for link_id in sorted(self.links)]

    def all_active_links(self) -> List[ReasoningLink]:
        return [self.links[link_id] for link_id in sorted(self.links) if self.links[link_id].active]

    def get(self, link_id: int) -> ReasoningLink:
        link = self.links.get(link_id)
        if link is None:
            raise ValueError(f"Unknown reasoning link ID: {link_id}")
        return link

    def ensure_link(
        self,
        relation_type: str,
        antecedent: BrainInputV1,
        consequent: BrainInputV1,
        current_sequence: int,
        split_from_link_id: Optional[int] = None,
        notes: Optional[Sequence[str]] = None,
    ) -> Tuple[ReasoningLink, bool]:
        fingerprint = self._fingerprint(relation_type, antecedent, consequent)
        existing_id = self.fingerprint_to_link_id.get(fingerprint)
        if existing_id is not None and existing_id in self.links:
            return self.links[existing_id], False
        link_id = self.next_link_id
        created = ReasoningLink(
            link_id=link_id,
            relation_type=relation_type,
            antecedent=antecedent,
            consequent=consequent,
            created_sequence=current_sequence,
            usage_count=0,
            success_count=0,
            failure_count=0,
            reliability=0.5,
            temporal_score=0.0,
            last_used_sequence=current_sequence,
            active=True,
            split_from_link_id=split_from_link_id,
            notes=[str(item) for item in (notes or [])],
        )
        self.links[link_id] = created
        self.fingerprint_to_link_id[fingerprint] = link_id
        self.next_link_id += 1
        self.save()
        return created, True

    def update_usage(
        self,
        link_id: int,
        current_sequence: int,
        success: Optional[bool] = None,
    ) -> None:
        link = self.get(link_id)
        usage_count = link.usage_count + 1
        success_count = link.success_count + (1 if success is True else 0)
        failure_count = link.failure_count + (1 if success is False else 0)
        reliability = self._compute_reliability(usage_count, success_count, failure_count)
        temporal_score = self._compute_temporal_score(
            usage_count=usage_count,
            reliability=reliability,
            current_sequence=current_sequence,
            last_used_sequence=current_sequence,
            created_sequence=link.created_sequence,
        )
        self.links[link_id] = ReasoningLink(
            link_id=link.link_id,
            relation_type=link.relation_type,
            antecedent=link.antecedent,
            consequent=link.consequent,
            created_sequence=link.created_sequence,
            usage_count=usage_count,
            success_count=success_count,
            failure_count=failure_count,
            reliability=reliability,
            temporal_score=temporal_score,
            last_used_sequence=current_sequence,
            active=link.active,
            split_from_link_id=link.split_from_link_id,
            notes=link.notes,
        )
        self.save()

    def set_active(self, link_id: int, active: bool, note: Optional[str] = None) -> None:
        link = self.get(link_id)
        notes = list(link.notes)
        if note:
            notes.append(str(note))
        self.links[link_id] = ReasoningLink(
            link_id=link.link_id,
            relation_type=link.relation_type,
            antecedent=link.antecedent,
            consequent=link.consequent,
            created_sequence=link.created_sequence,
            usage_count=link.usage_count,
            success_count=link.success_count,
            failure_count=link.failure_count,
            reliability=link.reliability,
            temporal_score=link.temporal_score,
            last_used_sequence=link.last_used_sequence,
            active=active,
            split_from_link_id=link.split_from_link_id,
            notes=notes,
        )
        self.save()

    @staticmethod
    def _fingerprint(relation_type: str, antecedent: BrainInputV1, consequent: BrainInputV1) -> str:
        payload = {
            "relation_type": relation_type,
            "antecedent": antecedent.to_dict(),
            "consequent": consequent.to_dict(),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_reliability(usage_count: int, success_count: int, failure_count: int) -> float:
        total_outcomes = success_count + failure_count
        if total_outcomes == 0:
            return round(min(0.75, 0.45 + (0.05 * min(usage_count, 5))), 4)
        return round(success_count / max(1, total_outcomes), 4)

    @staticmethod
    def _compute_temporal_score(
        usage_count: int,
        reliability: float,
        current_sequence: int,
        last_used_sequence: int,
        created_sequence: int,
    ) -> float:
        sequence = max(current_sequence, last_used_sequence, created_sequence, 1)
        age = max(0, sequence - max(last_used_sequence, created_sequence))
        frequency_signal = min(1.0, usage_count / max(1, sequence))
        recency_signal = 1.0 / (1.0 + age)
        return round(
            max(
                0.0,
                min(
                    1.0,
                    (reliability * 0.55) + (frequency_signal * 0.25) + (recency_signal * 0.2),
                ),
            ),
            4,
        )


class BrainReasoningEngineV1:
    def __init__(
        self,
        state_dir: str | Path = DEFAULT_OUTPUT_DIR / "brain_reasoning_v1.state",
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.runtime: BrainReasoningRuntime = build_reasoning_runtime(self.state_dir / "runtime")
        self.native_engine = BrainNativeReasoningEngine(self.runtime)
        self.link_registry = BrainReasoningLinkRegistry(self.state_dir / "brain_reasoning_links.json")

    def learn_link(
        self,
        antecedent: BrainInputV1,
        consequent: BrainInputV1,
        relation_type: str,
        current_sequence: int,
        split_from_link_id: Optional[int] = None,
        notes: Optional[Sequence[str]] = None,
    ) -> ReasoningLink:
        normalized_antecedent = self.native_engine.normalize(antecedent, current_sequence=current_sequence)
        normalized_consequent = self.native_engine.normalize(consequent, current_sequence=current_sequence)
        link, created = self.link_registry.ensure_link(
            relation_type=relation_type,
            antecedent=normalized_antecedent,
            consequent=normalized_consequent,
            current_sequence=current_sequence,
            split_from_link_id=split_from_link_id,
            notes=notes,
        )
        if created:
            self._touch_patterns(normalized_antecedent, current_sequence)
            self._touch_patterns(normalized_consequent, current_sequence)
        return link

    def simulate(
        self,
        brain_input: BrainInputV1,
        max_depth: int = 3,
        beam_width: int = 3,
        current_sequence: int = 1,
    ) -> ReasoningPrediction:
        root = self.native_engine.normalize(brain_input, current_sequence=current_sequence)
        frontier: List[ReasoningPath] = [
            ReasoningPath(
                steps=[],
                final_outcome=root,
                confidence=1.0,
            )
        ]
        completed: List[ReasoningPath] = []

        for depth in range(1, max_depth + 1):
            next_frontier: List[ReasoningPath] = []
            for path in frontier:
                matches = self._find_matching_links(path.final_outcome, current_sequence=current_sequence)
                if not matches:
                    completed.append(path)
                    continue
                for link, match_score in matches[:beam_width]:
                    step_confidence = round(path.confidence * self._step_confidence(link, match_score), 4)
                    step = ChainStep(
                        link_id=link.link_id,
                        relation_type=link.relation_type,
                        antecedent=link.antecedent,
                        consequent=link.consequent,
                        match_score=match_score,
                        reliability=link.reliability,
                        confidence=step_confidence,
                    )
                    next_frontier.append(
                        ReasoningPath(
                            steps=list(path.steps) + [step],
                            final_outcome=link.consequent,
                            confidence=step_confidence,
                        )
                    )
            if not next_frontier:
                break
            frontier = sorted(
                next_frontier,
                key=lambda item: (-item.confidence, -item.depth, self._path_sort_key(item)),
            )[:beam_width]

        completed.extend(frontier)
        candidate_paths = sorted(
            [path for path in completed if path.steps],
            key=lambda item: (-item.confidence, -item.depth, self._path_sort_key(item)),
        )
        selected_path = candidate_paths[0] if candidate_paths else None
        return ReasoningPrediction(
            input_brain=root,
            candidate_paths=candidate_paths,
            selected_path=selected_path,
            confidence_score=selected_path.confidence if selected_path is not None else 0.0,
            final_predicted_outcome=selected_path.final_outcome if selected_path is not None else None,
        )

    def validate_prediction(
        self,
        input_brain: BrainInputV1,
        prediction: ReasoningPrediction,
        expected_chain: Sequence[BrainInputV1],
        relation_types: Sequence[str],
        current_sequence: int,
    ) -> Dict[str, object]:
        normalized_input = self.native_engine.normalize(input_brain, current_sequence=current_sequence)
        normalized_expected = [
            self.native_engine.normalize(item, current_sequence=current_sequence)
            for item in expected_chain
        ]
        predicted_final = prediction.final_predicted_outcome
        expected_final = normalized_expected[-1] if normalized_expected else None
        success = bool(
            predicted_final is not None
            and expected_final is not None
            and self._same_structure(predicted_final, expected_final)
        )

        touched_link_ids: List[int] = []
        if prediction.selected_path is not None:
            for step in prediction.selected_path.steps:
                self.link_registry.update_usage(
                    link_id=step.link_id,
                    current_sequence=current_sequence,
                    success=success,
                )
                touched_link_ids.append(step.link_id)

        created_links: List[int] = []
        split_links: List[int] = []
        if success:
            for state in [normalized_input] + list(normalized_expected):
                self._touch_patterns(state, current_sequence)
        else:
            states = [normalized_input] + list(normalized_expected)
            for index in range(len(states) - 1):
                split_from_link_id = None
                if prediction.selected_path is not None and index < len(prediction.selected_path.steps):
                    prior_step = prediction.selected_path.steps[index]
                    if not self._same_structure(prior_step.antecedent, states[index]):
                        split_from_link_id = prior_step.link_id
                link = self.learn_link(
                    antecedent=states[index],
                    consequent=states[index + 1],
                    relation_type=self._relation_type_at(relation_types, index),
                    current_sequence=current_sequence,
                    split_from_link_id=split_from_link_id,
                    notes=["created_from_validation_failure"],
                )
                self.link_registry.update_usage(
                    link_id=link.link_id,
                    current_sequence=current_sequence,
                    success=True,
                )
                created_links.append(link.link_id)
                if split_from_link_id is not None:
                    split_links.append(link.link_id)
            self.runtime.pattern_evolution.run_adaptive_cycle(current_sequence=current_sequence)

        return {
            "success": success,
            "predicted_final_outcome": predicted_final.to_dict() if predicted_final is not None else None,
            "expected_final_outcome": expected_final.to_dict() if expected_final is not None else None,
            "used_link_ids": touched_link_ids,
            "created_link_ids": created_links,
            "split_link_ids": split_links,
            "confidence_score": round(prediction.confidence_score, 4),
        }

    def registry_summary(self) -> Dict[str, object]:
        active_links = self.link_registry.all_active_links()
        average_reliability = round(
            sum(link.reliability for link in active_links) / max(1, len(active_links)),
            4,
        ) if active_links else 0.0
        average_temporal_score = round(
            sum(link.temporal_score for link in active_links) / max(1, len(active_links)),
            4,
        ) if active_links else 0.0
        return {
            "link_count": len(self.link_registry.links),
            "active_link_count": len(active_links),
            "average_reliability": average_reliability,
            "average_temporal_score": average_temporal_score,
        }

    def _find_matching_links(
        self,
        current: BrainInputV1,
        current_sequence: int,
    ) -> List[Tuple[ReasoningLink, float]]:
        matches: List[Tuple[ReasoningLink, float]] = []
        for link in self.link_registry.all_active_links():
            match_score = self._match_score(current, link.antecedent)
            if match_score < 0.55:
                continue
            matches.append((link, match_score))
        return sorted(
            matches,
            key=lambda item: (
                -item[1],
                -item[0].reliability,
                -item[0].temporal_score,
                -self._specificity_score(item[0].antecedent),
                item[0].link_id,
            ),
        )

    def _match_score(
        self,
        current: BrainInputV1,
        antecedent: BrainInputV1,
    ) -> float:
        current_memory = self.runtime.brain_input_codec.brain_input_to_memory(current)
        antecedent_memory = self.runtime.brain_input_codec.brain_input_to_memory(antecedent)
        numerator = 0.0
        denominator = 0.0

        weights = {
            "actor": 1.0,
            "intent": 1.5,
            "action": 2.5,
            "object": 2.5,
            "topic": 2.0,
            "time": 1.0,
            "priority": 0.5,
        }
        for field_name, weight in weights.items():
            expected = str(getattr(antecedent_memory, field_name) or "")
            if not expected:
                continue
            denominator += weight
            actual = str(getattr(current_memory, field_name) or "")
            if actual == expected:
                numerator += weight

        for field_name, weight in {
            "modifiers": 1.0,
            "constraints": 0.8,
            "negations": 0.8,
            "conditions": 0.8,
        }.items():
            expected_values = [str(item) for item in getattr(antecedent_memory, field_name)]
            if not expected_values:
                continue
            actual_values = {str(item) for item in getattr(current_memory, field_name)}
            denominator += weight * len(expected_values)
            numerator += weight * sum(1 for item in expected_values if item in actual_values)

        expected_relations = {
            (
                str(item.get("type", "")),
                str(item.get("source", "")),
                str(item.get("target", "")),
            )
            for item in antecedent_memory.relations
        }
        if expected_relations:
            actual_relations = {
                (
                    str(item.get("type", "")),
                    str(item.get("source", "")),
                    str(item.get("target", "")),
                )
                for item in current_memory.relations
            }
            denominator += 0.75 * len(expected_relations)
            numerator += 0.75 * len(expected_relations.intersection(actual_relations))

        if antecedent.pattern_id is not None:
            denominator += 1.0
            if antecedent.pattern_id == current.pattern_id:
                numerator += 1.0

        if denominator <= 0.0:
            return 0.0
        return round(numerator / denominator, 4)

    def _step_confidence(self, link: ReasoningLink, match_score: float) -> float:
        return round(
            max(
                0.0,
                min(
                    1.0,
                    (match_score * 0.45) + (link.reliability * 0.4) + (link.temporal_score * 0.15),
                ),
            ),
            4,
        )

    def _specificity_score(self, brain_input: BrainInputV1) -> int:
        memory = self.runtime.brain_input_codec.brain_input_to_memory(brain_input)
        score = 0
        for field_name in ("actor", "intent", "action", "object", "topic", "time", "priority"):
            if str(getattr(memory, field_name) or ""):
                score += 1
        for field_name in ("modifiers", "constraints", "negations", "conditions"):
            score += len(list(getattr(memory, field_name)))
        score += len(memory.relations)
        return score

    def _same_structure(self, left: BrainInputV1, right: BrainInputV1) -> bool:
        left_bundle = self.runtime.codec.encode_memory_bundle(self.runtime.brain_input_codec.brain_input_to_memory(left))
        right_bundle = self.runtime.codec.encode_memory_bundle(self.runtime.brain_input_codec.brain_input_to_memory(right))
        return left_bundle == right_bundle

    def _touch_patterns(self, brain_input: BrainInputV1, current_sequence: int) -> None:
        bundle = self.runtime.codec.encode_memory_bundle(self.runtime.brain_input_codec.brain_input_to_memory(brain_input))
        self.runtime.pattern_evolution.recommend_pattern(
            memory_bundle=bundle,
            current_sequence=current_sequence,
            include_subpatterns=True,
            update_usage=True,
        )

    @staticmethod
    def _relation_type_at(relation_types: Sequence[str], index: int) -> str:
        if index < len(relation_types):
            return str(relation_types[index])
        return "cause_effect"

    @staticmethod
    def _path_sort_key(path: ReasoningPath) -> Tuple[int, ...]:
        return tuple(step.link_id for step in path.steps)
