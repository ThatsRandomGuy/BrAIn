from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from brain_codec import BrainMemory
from brain_input_v1 import BrainInputV1
from brain_reasoning_engine import DEFAULT_OUTPUT_DIR
from brain_reasoning_engine_v1 import BrainReasoningEngineV1, ReasoningLink, ReasoningPrediction


GENERALIZABLE_SLOT_PATHS: Tuple[str, ...] = (
    "actor",
    "intent",
    "action",
    "object",
    "topic",
    "modifiers[0]",
    "constraints[0]",
    "conditions[0]",
    "negations[0]",
)


@dataclass(frozen=True)
class ConceptNode:
    concept_id: int
    slot_path: str
    members: List[str]
    source_link_ids: List[int]
    support_count: int
    confidence: float
    active: bool = True
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "concept_id": self.concept_id,
            "slot_path": self.slot_path,
            "members": list(self.members),
            "source_link_ids": list(self.source_link_ids),
            "support_count": self.support_count,
            "confidence": round(self.confidence, 4),
            "active": self.active,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ConceptNode":
        return cls(
            concept_id=int(data.get("concept_id", 0)),
            slot_path=str(data.get("slot_path", "")),
            members=[str(item) for item in data.get("members", [])] if isinstance(data.get("members", []), list) else [],
            source_link_ids=[int(item) for item in data.get("source_link_ids", []) if isinstance(item, int)] if isinstance(data.get("source_link_ids", []), list) else [],
            support_count=int(data.get("support_count", 0)),
            confidence=float(data.get("confidence", 0.0)),
            active=bool(data.get("active", True)),
            notes=[str(item) for item in data.get("notes", [])] if isinstance(data.get("notes", []), list) else [],
        )


@dataclass(frozen=True)
class ConceptLink:
    concept_link_id: int
    relation_type: str
    antecedent_template: BrainInputV1
    consequent: BrainInputV1
    concept_bindings: Dict[str, int]
    source_link_ids: List[int]
    support_count: int
    reliability: float
    active: bool = True
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "concept_link_id": self.concept_link_id,
            "relation_type": self.relation_type,
            "antecedent_template": self.antecedent_template.to_dict(),
            "consequent": self.consequent.to_dict(),
            "concept_bindings": {str(key): int(value) for key, value in self.concept_bindings.items()},
            "source_link_ids": list(self.source_link_ids),
            "support_count": self.support_count,
            "reliability": round(self.reliability, 4),
            "active": self.active,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ConceptLink":
        raw_bindings = data.get("concept_bindings", {})
        return cls(
            concept_link_id=int(data.get("concept_link_id", 0)),
            relation_type=str(data.get("relation_type", "")),
            antecedent_template=BrainInputV1.from_dict(data.get("antecedent_template", {}) if isinstance(data.get("antecedent_template", {}), dict) else {}),
            consequent=BrainInputV1.from_dict(data.get("consequent", {}) if isinstance(data.get("consequent", {}), dict) else {}),
            concept_bindings={str(key): int(value) for key, value in raw_bindings.items()} if isinstance(raw_bindings, dict) else {},
            source_link_ids=[int(item) for item in data.get("source_link_ids", []) if isinstance(item, int)] if isinstance(data.get("source_link_ids", []), list) else [],
            support_count=int(data.get("support_count", 0)),
            reliability=float(data.get("reliability", 0.0)),
            active=bool(data.get("active", True)),
            notes=[str(item) for item in data.get("notes", [])] if isinstance(data.get("notes", []), list) else [],
        )


@dataclass(frozen=True)
class ConceptChainStep:
    source_level: str
    relation_type: str
    antecedent: BrainInputV1
    consequent: BrainInputV1
    confidence: float
    match_score: float
    link_id: Optional[int] = None
    concept_link_id: Optional[int] = None
    concept_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "source_level": self.source_level,
            "relation_type": self.relation_type,
            "antecedent": self.antecedent.to_dict(),
            "consequent": self.consequent.to_dict(),
            "confidence": round(self.confidence, 4),
            "match_score": round(self.match_score, 4),
            "concept_ids": list(self.concept_ids),
        }
        if self.link_id is not None:
            payload["link_id"] = self.link_id
        if self.concept_link_id is not None:
            payload["concept_link_id"] = self.concept_link_id
        return payload


@dataclass(frozen=True)
class ConceptReasoningPath:
    steps: List[ConceptChainStep]
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
class ConceptPrediction:
    input_brain: BrainInputV1
    candidate_paths: List[ConceptReasoningPath]
    selected_path: Optional[ConceptReasoningPath]
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


@dataclass(frozen=True)
class _ConceptCandidate:
    source_level: str
    relation_type: str
    antecedent: BrainInputV1
    consequent: BrainInputV1
    match_score: float
    confidence_factor: float
    link_id: Optional[int] = None
    concept_link_id: Optional[int] = None
    concept_ids: List[int] = field(default_factory=list)


class BrainConceptRegistry:
    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)
        self.concepts: Dict[int, ConceptNode] = {}
        self.concept_links: Dict[int, ConceptLink] = {}
        self.next_concept_id = 1
        self.next_concept_link_id = 1
        self._load()

    def _load(self) -> None:
        if not self.registry_path.exists():
            self.concepts = {}
            self.concept_links = {}
            self.next_concept_id = 1
            self.next_concept_link_id = 1
            return
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        loaded_concepts: Dict[int, ConceptNode] = {}
        loaded_links: Dict[int, ConceptLink] = {}
        self.next_concept_id = int(data.get("next_concept_id", 1))
        self.next_concept_link_id = int(data.get("next_concept_link_id", 1))
        for item in data.get("concepts", []):
            if not isinstance(item, dict):
                continue
            concept = ConceptNode.from_dict(item)
            if concept.concept_id > 0:
                loaded_concepts[concept.concept_id] = concept
        for item in data.get("concept_links", []):
            if not isinstance(item, dict):
                continue
            concept_link = ConceptLink.from_dict(item)
            if concept_link.concept_link_id > 0:
                loaded_links[concept_link.concept_link_id] = concept_link
        self.concepts = loaded_concepts
        self.concept_links = loaded_links
        self.next_concept_id = max([self.next_concept_id] + [concept_id + 1 for concept_id in self.concepts], default=1)
        self.next_concept_link_id = max([self.next_concept_link_id] + [link_id + 1 for link_id in self.concept_links], default=1)

    def save(self) -> None:
        payload = {
            "next_concept_id": self.next_concept_id,
            "next_concept_link_id": self.next_concept_link_id,
            "concepts": [self.concepts[concept_id].to_dict() for concept_id in sorted(self.concepts)],
            "concept_links": [self.concept_links[link_id].to_dict() for link_id in sorted(self.concept_links)],
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def replace_all(
        self,
        concepts: Sequence[ConceptNode],
        concept_links: Sequence[ConceptLink],
    ) -> None:
        self.concepts = {concept.concept_id: concept for concept in concepts}
        self.concept_links = {link.concept_link_id: link for link in concept_links}
        self.next_concept_id = max([1] + [concept.concept_id + 1 for concept in concepts], default=1)
        self.next_concept_link_id = max([1] + [link.concept_link_id + 1 for link in concept_links], default=1)
        self.save()

    def all_active_concepts(self) -> List[ConceptNode]:
        return [self.concepts[concept_id] for concept_id in sorted(self.concepts) if self.concepts[concept_id].active]

    def all_active_concept_links(self) -> List[ConceptLink]:
        return [self.concept_links[link_id] for link_id in sorted(self.concept_links) if self.concept_links[link_id].active]

    def get_concept(self, concept_id: int) -> ConceptNode:
        return self.concepts[int(concept_id)]

    def get_concept_link(self, concept_link_id: int) -> ConceptLink:
        return self.concept_links[int(concept_link_id)]


class BrainConceptAbstractionV2:
    def __init__(
        self,
        state_dir: str | Path = DEFAULT_OUTPUT_DIR / "brain_reasoning_v2.state",
        reliability_threshold: float = 0.75,
        generalization_support_threshold: int = 3,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.v1_engine = BrainReasoningEngineV1(state_dir=self.state_dir / "v1")
        self.concept_registry = BrainConceptRegistry(self.state_dir / "brain_concepts.json")
        self.reliability_threshold = reliability_threshold
        self.generalization_support_threshold = generalization_support_threshold

    def learn_link(
        self,
        antecedent: BrainInputV1,
        consequent: BrainInputV1,
        relation_type: str,
        current_sequence: int,
        split_from_link_id: Optional[int] = None,
        notes: Optional[Sequence[str]] = None,
    ) -> ReasoningLink:
        link = self.v1_engine.learn_link(
            antecedent=antecedent,
            consequent=consequent,
            relation_type=relation_type,
            current_sequence=current_sequence,
            split_from_link_id=split_from_link_id,
            notes=notes,
        )
        self.rebuild_concepts(current_sequence=current_sequence)
        return link

    def rebuild_concepts(self, current_sequence: int = 1) -> Dict[str, int]:
        active_links = [
            link
            for link in self.v1_engine.link_registry.all_active_links()
            if link.reliability >= self.reliability_threshold
        ]
        grouped: Dict[Tuple[str, str, str, str], List[ReasoningLink]] = {}
        templates: Dict[Tuple[str, str, str, str], BrainInputV1] = {}
        consequents: Dict[Tuple[str, str, str, str], BrainInputV1] = {}
        members_by_group: Dict[Tuple[str, str, str, str], Dict[str, int]] = {}

        for link in active_links:
            antecedent_memory = self._memory_from_brain(link.antecedent)
            consequent_key = self._brain_fingerprint(link.consequent)
            for slot_path in GENERALIZABLE_SLOT_PATHS:
                slot_value = self._get_slot_path_value(antecedent_memory, slot_path)
                if not slot_value:
                    continue
                template_memory = self._copy_memory(antecedent_memory)
                self._clear_slot_path(template_memory, slot_path)
                template_brain = self._brain_from_memory(template_memory, current_sequence=current_sequence)
                template_key = self._brain_fingerprint(template_brain)
                group_key = (link.relation_type, slot_path, template_key, consequent_key)
                grouped.setdefault(group_key, []).append(link)
                templates[group_key] = template_brain
                consequents[group_key] = link.consequent
                member_counts = members_by_group.setdefault(group_key, {})
                member_counts[slot_value] = member_counts.get(slot_value, 0) + 1

        concepts: List[ConceptNode] = []
        concept_links: List[ConceptLink] = []

        for group_key in sorted(grouped):
            source_links = grouped[group_key]
            members = sorted(members_by_group.get(group_key, {}))
            if len(members) < 2:
                continue
            unique_link_ids = sorted({link.link_id for link in source_links})
            support_count = len(unique_link_ids)
            average_reliability = round(
                sum(link.reliability for link in source_links) / max(1, len(source_links)),
                4,
            )
            concept_confidence = round(
                min(1.0, (average_reliability * 0.7) + (min(1.0, len(members) / 4.0) * 0.3)),
                4,
            )
            relation_type, slot_path, _, _ = group_key
            concept = ConceptNode(
                concept_id=len(concepts) + 1,
                slot_path=slot_path,
                members=members,
                source_link_ids=unique_link_ids,
                support_count=support_count,
                confidence=concept_confidence,
                active=True,
                notes=["emergent_concept_from_reliable_links"],
            )
            concepts.append(concept)
            concept_links.append(
                ConceptLink(
                    concept_link_id=len(concept_links) + 1,
                    relation_type=relation_type,
                    antecedent_template=templates[group_key],
                    consequent=consequents[group_key],
                    concept_bindings={slot_path: concept.concept_id},
                    source_link_ids=unique_link_ids,
                    support_count=support_count,
                    reliability=average_reliability,
                    active=True,
                    notes=["generalized_link_from_concept_cluster"],
                )
            )

        self.concept_registry.replace_all(concepts=concepts, concept_links=concept_links)
        return {
            "concept_count": len(concepts),
            "concept_link_count": len(concept_links),
        }

    def simulate(
        self,
        brain_input: BrainInputV1,
        max_depth: int = 4,
        beam_width: int = 4,
        current_sequence: int = 1,
    ) -> ConceptPrediction:
        self.rebuild_concepts(current_sequence=current_sequence)
        root = self.v1_engine.native_engine.normalize(brain_input, current_sequence=current_sequence)
        frontier: List[ConceptReasoningPath] = [
            ConceptReasoningPath(
                steps=[],
                final_outcome=root,
                confidence=1.0,
            )
        ]
        completed: List[ConceptReasoningPath] = []

        for _ in range(max_depth):
            next_frontier: List[ConceptReasoningPath] = []
            for path in frontier:
                matches = self._find_matching_candidates(path.final_outcome, current_sequence=current_sequence)
                if not matches:
                    completed.append(path)
                    continue
                for candidate in matches[:beam_width]:
                    if self._would_cycle(path, candidate.consequent):
                        continue
                    step_confidence = round(path.confidence * candidate.confidence_factor, 4)
                    step = ConceptChainStep(
                        source_level=candidate.source_level,
                        relation_type=candidate.relation_type,
                        antecedent=candidate.antecedent,
                        consequent=candidate.consequent,
                        confidence=step_confidence,
                        match_score=candidate.match_score,
                        link_id=candidate.link_id,
                        concept_link_id=candidate.concept_link_id,
                        concept_ids=list(candidate.concept_ids),
                    )
                    next_frontier.append(
                        ConceptReasoningPath(
                            steps=list(path.steps) + [step],
                            final_outcome=candidate.consequent,
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
        candidate_paths = [path for path in completed if path.steps]
        candidate_paths = self._apply_support_bonus(candidate_paths)
        selected_path = candidate_paths[0] if candidate_paths else None
        return ConceptPrediction(
            input_brain=root,
            candidate_paths=candidate_paths,
            selected_path=selected_path,
            confidence_score=selected_path.confidence if selected_path is not None else 0.0,
            final_predicted_outcome=selected_path.final_outcome if selected_path is not None else None,
        )

    def validate_prediction(
        self,
        input_brain: BrainInputV1,
        prediction: ConceptPrediction,
        expected_chain: Sequence[BrainInputV1],
        relation_types: Sequence[str],
        current_sequence: int,
    ) -> Dict[str, object]:
        normalized_input = self.v1_engine.native_engine.normalize(input_brain, current_sequence=current_sequence)
        normalized_expected = [
            self.v1_engine.native_engine.normalize(item, current_sequence=current_sequence)
            for item in expected_chain
        ]
        expected_final = normalized_expected[-1] if normalized_expected else None
        predicted_final = prediction.final_predicted_outcome
        success = bool(
            expected_final is not None
            and predicted_final is not None
            and self.v1_engine._same_structure(predicted_final, expected_final)
        )

        v1_prediction = self.v1_engine.simulate(
            normalized_input,
            max_depth=max(3, len(expected_chain) + 1),
            beam_width=4,
            current_sequence=current_sequence,
        )
        v1_validation = self.v1_engine.validate_prediction(
            input_brain=normalized_input,
            prediction=v1_prediction,
            expected_chain=normalized_expected,
            relation_types=relation_types,
            current_sequence=current_sequence,
        )
        observed_link_ids: List[int] = []
        observed_states = [normalized_input] + list(normalized_expected)
        for index in range(len(observed_states) - 1):
            observed_link = self.v1_engine.learn_link(
                antecedent=observed_states[index],
                consequent=observed_states[index + 1],
                relation_type=self.v1_engine._relation_type_at(relation_types, index),
                current_sequence=current_sequence,
                notes=["validated_observed_transition"],
            )
            self.v1_engine.link_registry.update_usage(
                link_id=observed_link.link_id,
                current_sequence=current_sequence,
                success=True,
            )
            observed_link_ids.append(observed_link.link_id)
        rebuild_summary = self.rebuild_concepts(current_sequence=current_sequence)
        selected_source_levels = (
            [step.source_level for step in prediction.selected_path.steps]
            if prediction.selected_path is not None
            else []
        )
        return {
            "success": success,
            "selected_depth": prediction.selected_path.depth if prediction.selected_path is not None else 0,
            "selected_source_levels": selected_source_levels,
            "used_concept_abstraction": any(level != "specific" for level in selected_source_levels),
            "concept_link_ids": (
                [step.concept_link_id for step in prediction.selected_path.steps if step.concept_link_id is not None]
                if prediction.selected_path is not None
                else []
            ),
            "concept_ids": (
                sorted({concept_id for step in prediction.selected_path.steps for concept_id in step.concept_ids})
                if prediction.selected_path is not None
                else []
            ),
            "observed_link_ids": observed_link_ids,
            "v1_validation": v1_validation,
            "concept_rebuild": rebuild_summary,
        }

    def registry_summary(self) -> Dict[str, object]:
        active_concepts = self.concept_registry.all_active_concepts()
        active_links = self.concept_registry.all_active_concept_links()
        average_concept_confidence = round(
            sum(concept.confidence for concept in active_concepts) / max(1, len(active_concepts)),
            4,
        ) if active_concepts else 0.0
        average_link_reliability = round(
            sum(link.reliability for link in active_links) / max(1, len(active_links)),
            4,
        ) if active_links else 0.0
        average_support = round(
            sum(link.support_count for link in active_links) / max(1, len(active_links)),
            4,
        ) if active_links else 0.0
        return {
            **self.v1_engine.registry_summary(),
            "concept_count": len(active_concepts),
            "concept_link_count": len(active_links),
            "average_concept_confidence": average_concept_confidence,
            "average_concept_link_reliability": average_link_reliability,
            "average_concept_support": average_support,
        }

    def _find_matching_candidates(
        self,
        current: BrainInputV1,
        current_sequence: int,
    ) -> List[_ConceptCandidate]:
        candidates: List[_ConceptCandidate] = []
        for link, match_score in self.v1_engine._find_matching_links(current, current_sequence=current_sequence):
            if not self._is_exact_match(current, link.antecedent):
                continue
            candidates.append(
                _ConceptCandidate(
                    source_level="specific",
                    relation_type=link.relation_type,
                    antecedent=link.antecedent,
                    consequent=link.consequent,
                    match_score=match_score,
                    confidence_factor=self.v1_engine._step_confidence(link, match_score),
                    link_id=link.link_id,
                )
            )

        current_memory = self._memory_from_brain(current)
        for concept_link in self.concept_registry.all_active_concept_links():
            template_match = self.v1_engine._match_score(current, concept_link.antecedent_template)
            if template_match < 0.55:
                continue
            matched_concept_ids: List[int] = []
            matched_modes: List[str] = []
            concept_confidences: List[float] = []
            member_scores: List[float] = []
            valid = True
            for slot_path, concept_id in sorted(concept_link.concept_bindings.items()):
                slot_value = self._get_slot_path_value(current_memory, slot_path)
                if not slot_value:
                    valid = False
                    break
                concept = self.concept_registry.get_concept(concept_id)
                matched_concept_ids.append(concept_id)
                concept_confidences.append(concept.confidence)
                if slot_value in concept.members:
                    matched_modes.append("concept")
                    member_scores.append(1.0)
                elif concept.support_count >= self.generalization_support_threshold:
                    matched_modes.append("generalized")
                    member_scores.append(0.72)
                else:
                    valid = False
                    break
            if not valid or not member_scores:
                continue
            source_level = "generalized" if "generalized" in matched_modes else "concept"
            match_score = round((template_match * 0.75) + ((sum(member_scores) / len(member_scores)) * 0.25), 4)
            support_signal = min(1.0, concept_link.support_count / 5.0)
            concept_signal = sum(concept_confidences) / max(1, len(concept_confidences))
            confidence_factor = (
                (match_score * 0.38)
                + (concept_link.reliability * 0.32)
                + (concept_signal * 0.18)
                + (support_signal * 0.12)
            )
            if source_level == "concept":
                confidence_factor += 0.06
            else:
                confidence_factor -= 0.04
            confidence_factor = round(max(0.0, min(1.0, confidence_factor)), 4)
            if confidence_factor <= 0.0:
                continue
            candidates.append(
                _ConceptCandidate(
                    source_level=source_level,
                    relation_type=concept_link.relation_type,
                    antecedent=concept_link.antecedent_template,
                    consequent=concept_link.consequent,
                    match_score=match_score,
                    confidence_factor=confidence_factor,
                    concept_link_id=concept_link.concept_link_id,
                    concept_ids=matched_concept_ids,
                )
            )

        return sorted(
            candidates,
            key=lambda item: (
                -item.confidence_factor,
                self._source_level_rank(item.source_level),
                -item.match_score,
                -self.v1_engine._specificity_score(item.antecedent),
                item.link_id if item.link_id is not None else 10_000 + (item.concept_link_id or 0),
            ),
        )

    def _apply_support_bonus(self, paths: Sequence[ConceptReasoningPath]) -> List[ConceptReasoningPath]:
        grouped: Dict[str, List[ConceptReasoningPath]] = {}
        for path in paths:
            grouped.setdefault(self._brain_fingerprint(path.final_outcome), []).append(path)

        boosted_paths: List[ConceptReasoningPath] = []
        for group_paths in grouped.values():
            support_count = len(group_paths)
            confidence_sum = sum(path.confidence for path in group_paths)
            for path in group_paths:
                bonus = 0.0
                if support_count > 1:
                    bonus = min(0.18, (0.04 * (support_count - 1)) + (0.02 * max(0.0, confidence_sum - path.confidence)))
                boosted_paths.append(
                    ConceptReasoningPath(
                        steps=path.steps,
                        final_outcome=path.final_outcome,
                        confidence=round(min(1.0, path.confidence + bonus), 4),
                    )
                )
        return sorted(
            boosted_paths,
            key=lambda item: (-item.confidence, -item.depth, self._path_sort_key(item)),
        )

    def _would_cycle(self, path: ConceptReasoningPath, next_outcome: BrainInputV1) -> bool:
        next_key = self._brain_fingerprint(next_outcome)
        seen = {self._brain_fingerprint(path.final_outcome)}
        for step in path.steps:
            seen.add(self._brain_fingerprint(step.antecedent))
            seen.add(self._brain_fingerprint(step.consequent))
        return next_key in seen

    def _brain_from_memory(self, memory: BrainMemory, current_sequence: int) -> BrainInputV1:
        built = self.v1_engine.runtime.brain_input_codec.build_brain_input(
            memory=memory,
            encoding_type="bitpacked",
            primitive_ids=[],
        )
        return self.v1_engine.native_engine.normalize(built, current_sequence=current_sequence)

    def _memory_from_brain(self, brain_input: BrainInputV1) -> BrainMemory:
        return self.v1_engine.runtime.brain_input_codec.brain_input_to_memory(brain_input)

    def _brain_fingerprint(self, brain_input: BrainInputV1) -> str:
        bundle = self.v1_engine.runtime.codec.encode_memory_bundle(self._memory_from_brain(brain_input))
        return json.dumps(bundle, sort_keys=True)

    def _is_exact_match(self, current: BrainInputV1, antecedent: BrainInputV1) -> bool:
        current_memory = self._memory_from_brain(current)
        antecedent_memory = self._memory_from_brain(antecedent)
        for field_name in ("actor", "intent", "action", "object", "topic", "time", "priority"):
            expected = str(getattr(antecedent_memory, field_name) or "")
            if expected and str(getattr(current_memory, field_name) or "") != expected:
                return False
        for field_name in ("modifiers", "constraints", "conditions", "negations"):
            expected_values = [str(item) for item in getattr(antecedent_memory, field_name)]
            actual_values = {str(item) for item in getattr(current_memory, field_name)}
            for expected in expected_values:
                if expected not in actual_values:
                    return False
        expected_relations = {
            (
                str(item.get("type", "")),
                str(item.get("source", "")),
                str(item.get("target", "")),
            )
            for item in antecedent_memory.relations
        }
        actual_relations = {
            (
                str(item.get("type", "")),
                str(item.get("source", "")),
                str(item.get("target", "")),
            )
            for item in current_memory.relations
        }
        return expected_relations.issubset(actual_relations)

    @staticmethod
    def _copy_memory(memory: BrainMemory) -> BrainMemory:
        return BrainMemory.from_dict(memory.to_dict())

    @staticmethod
    def _get_slot_path_value(memory: BrainMemory, slot_path: str) -> str:
        if "[" in slot_path and slot_path.endswith("]"):
            field_name, index_part = slot_path[:-1].split("[", 1)
            values = list(getattr(memory, field_name, []))
            try:
                index = int(index_part)
            except ValueError:
                return ""
            if 0 <= index < len(values):
                return str(values[index])
            return ""
        return str(getattr(memory, slot_path, "") or "")

    @staticmethod
    def _clear_slot_path(memory: BrainMemory, slot_path: str) -> None:
        cleared_value = BrainConceptAbstractionV2._get_slot_path_value(memory, slot_path)
        if "[" in slot_path and slot_path.endswith("]"):
            field_name, index_part = slot_path[:-1].split("[", 1)
            values = list(getattr(memory, field_name, []))
            try:
                index = int(index_part)
            except ValueError:
                return
            if 0 <= index < len(values):
                del values[index]
                setattr(memory, field_name, values)
        else:
            setattr(memory, slot_path, "")
        if cleared_value:
            memory.relations = [
                relation
                for relation in memory.relations
                if str(relation.get("source", "")) != cleared_value and str(relation.get("target", "")) != cleared_value
            ]

    @staticmethod
    def _source_level_rank(source_level: str) -> int:
        ranks = {
            "specific": 0,
            "concept": 1,
            "generalized": 2,
        }
        return ranks.get(source_level, 9)

    @staticmethod
    def _path_sort_key(path: ConceptReasoningPath) -> Tuple[int, ...]:
        values: List[int] = []
        for step in path.steps:
            if step.link_id is not None:
                values.append(step.link_id)
            elif step.concept_link_id is not None:
                values.append(10_000 + step.concept_link_id)
            else:
                values.append(99_999)
        return tuple(values)


def prediction_to_v1_prediction(prediction: ConceptPrediction) -> ReasoningPrediction:
    return ReasoningPrediction(
        input_brain=prediction.input_brain,
        candidate_paths=[],
        selected_path=None,
        confidence_score=prediction.confidence_score,
        final_predicted_outcome=prediction.final_predicted_outcome,
    )
