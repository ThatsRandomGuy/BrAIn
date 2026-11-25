from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from brain_codec import BrainMemory
from brain_concept_abstraction_v2 import (
    BrainConceptAbstractionV2,
    ConceptChainStep,
    ConceptPrediction,
    ConceptReasoningPath,
)
from brain_input_v1 import BrainInputV1
from brain_reasoning_engine import DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class OutcomeProfile:
    profile_id: str
    outcome: BrainInputV1
    usage_count: int
    desirability_score: float
    risk_level: float
    temporal_score: float
    active: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "outcome": self.outcome.to_dict(),
            "usage_count": self.usage_count,
            "desirability_score": round(self.desirability_score, 4),
            "risk_level": round(self.risk_level, 4),
            "temporal_score": round(self.temporal_score, 4),
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "OutcomeProfile":
        return cls(
            profile_id=str(data.get("profile_id", "")),
            outcome=BrainInputV1.from_dict(data.get("outcome", {}) if isinstance(data.get("outcome", {}), dict) else {}),
            usage_count=int(data.get("usage_count", 0)),
            desirability_score=float(data.get("desirability_score", 0.0)),
            risk_level=float(data.get("risk_level", 0.5)),
            temporal_score=float(data.get("temporal_score", 0.0)),
            active=bool(data.get("active", True)),
        )


@dataclass(frozen=True)
class PlanningPath:
    steps: List[ConceptChainStep]
    final_outcome: BrainInputV1
    confidence: float
    desirability: float
    risk_level: float
    decision_score: float
    outcome_profile_confidence: float

    @property
    def depth(self) -> int:
        return len(self.steps)

    def to_concept_path(self) -> ConceptReasoningPath:
        return ConceptReasoningPath(
            steps=self.steps,
            final_outcome=self.final_outcome,
            confidence=self.confidence,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "final_outcome": self.final_outcome.to_dict(),
            "confidence": round(self.confidence, 4),
            "desirability": round(self.desirability, 4),
            "risk_level": round(self.risk_level, 4),
            "decision_score": round(self.decision_score, 4),
            "outcome_profile_confidence": round(self.outcome_profile_confidence, 4),
            "depth": self.depth,
        }


@dataclass(frozen=True)
class PlanningDecision:
    input_brain: BrainInputV1
    candidate_paths: List[PlanningPath]
    selected_path: Optional[PlanningPath]
    highest_confidence_path: Optional[PlanningPath]
    decision_score: float
    final_predicted_outcome: Optional[BrainInputV1]

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "input_brain": self.input_brain.to_dict(),
            "candidate_paths": [path.to_dict() for path in self.candidate_paths],
            "decision_score": round(self.decision_score, 4),
        }
        if self.selected_path is not None:
            payload["selected_path"] = self.selected_path.to_dict()
        if self.highest_confidence_path is not None:
            payload["highest_confidence_path"] = self.highest_confidence_path.to_dict()
        if self.final_predicted_outcome is not None:
            payload["final_predicted_outcome"] = self.final_predicted_outcome.to_dict()
        return payload


class BrainOutcomeRegistry:
    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)
        self.profiles: Dict[str, OutcomeProfile] = {}
        self._load()

    def _load(self) -> None:
        if not self.registry_path.exists():
            self.profiles = {}
            return
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded: Dict[str, OutcomeProfile] = {}
        if isinstance(data, dict):
            for item in data.get("profiles", []):
                if not isinstance(item, dict):
                    continue
                profile = OutcomeProfile.from_dict(item)
                if profile.profile_id:
                    loaded[profile.profile_id] = profile
        self.profiles = loaded

    def save(self) -> None:
        payload = {
            "profiles": [self.profiles[key].to_dict() for key in sorted(self.profiles)],
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def all_active_profiles(self) -> List[OutcomeProfile]:
        return [self.profiles[key] for key in sorted(self.profiles) if self.profiles[key].active]

    def get(self, profile_id: str) -> Optional[OutcomeProfile]:
        return self.profiles.get(str(profile_id))

    def update(
        self,
        profile_id: str,
        outcome: BrainInputV1,
        desirability: float,
        risk_level: float,
        current_sequence: int,
    ) -> OutcomeProfile:
        existing = self.profiles.get(profile_id)
        if existing is None:
            usage_count = 1
            desirability_score = round(max(-1.0, min(1.0, desirability)), 4)
            risk_score = round(max(0.0, min(1.0, risk_level)), 4)
        else:
            usage_count = existing.usage_count + 1
            desirability_score = round(
                ((existing.desirability_score * existing.usage_count) + desirability) / usage_count,
                4,
            )
            risk_score = round(
                ((existing.risk_level * existing.usage_count) + risk_level) / usage_count,
                4,
            )
        temporal_score = round(
            max(
                0.0,
                min(1.0, (min(1.0, usage_count / max(1, current_sequence)) * 0.4) + ((1.0 - risk_score) * 0.2) + (((desirability_score + 1.0) / 2.0) * 0.4)),
            ),
            4,
        )
        profile = OutcomeProfile(
            profile_id=profile_id,
            outcome=outcome,
            usage_count=usage_count,
            desirability_score=desirability_score,
            risk_level=risk_score,
            temporal_score=temporal_score,
            active=True,
        )
        self.profiles[profile_id] = profile
        self.save()
        return profile


class BrainPlanningSimulationV3:
    def __init__(
        self,
        state_dir: str | Path = DEFAULT_OUTPUT_DIR / "brain_planning_v3.state",
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.v2_engine = BrainConceptAbstractionV2(state_dir=self.state_dir / "v2")
        self.outcome_registry = BrainOutcomeRegistry(self.state_dir / "brain_outcome_profiles.json")

    def plan(
        self,
        brain_input: BrainInputV1,
        max_depth: int = 6,
        beam_width: int = 8,
        current_sequence: int = 1,
    ) -> PlanningDecision:
        concept_prediction = self.v2_engine.simulate(
            brain_input,
            max_depth=max_depth,
            beam_width=beam_width,
            current_sequence=current_sequence,
        )
        planning_paths = [
            self._score_concept_path(path, current_sequence=current_sequence)
            for path in concept_prediction.candidate_paths
        ]
        planning_paths = sorted(
            planning_paths,
            key=lambda item: (-item.decision_score, -item.confidence, -item.depth, self._planning_sort_key(item)),
        )
        highest_confidence_path = max(
            planning_paths,
            key=lambda item: (item.confidence, item.decision_score, item.depth, tuple(self._planning_sort_key(item))),
            default=None,
        )
        selected_path = planning_paths[0] if planning_paths else None
        return PlanningDecision(
            input_brain=concept_prediction.input_brain,
            candidate_paths=planning_paths,
            selected_path=selected_path,
            highest_confidence_path=highest_confidence_path,
            decision_score=selected_path.decision_score if selected_path is not None else 0.0,
            final_predicted_outcome=selected_path.final_outcome if selected_path is not None else None,
        )

    def validate_decision(
        self,
        input_brain: BrainInputV1,
        decision: PlanningDecision,
        preferred_chain: Sequence[BrainInputV1],
        preferred_relation_types: Sequence[str],
        alternative_chains: Sequence[Tuple[Sequence[BrainInputV1], Sequence[str], float, float]],
        current_sequence: int,
    ) -> Dict[str, object]:
        normalized_input = self.v2_engine.v1_engine.native_engine.normalize(input_brain, current_sequence=current_sequence)
        normalized_preferred = [
            self.v2_engine.v1_engine.native_engine.normalize(item, current_sequence=current_sequence)
            for item in preferred_chain
        ]
        expected_final = normalized_preferred[-1] if normalized_preferred else None
        selected = decision.selected_path
        success = bool(
            selected is not None
            and expected_final is not None
            and self.v2_engine.v1_engine._same_structure(selected.final_outcome, expected_final)
        )
        candidate_quality = 1.0 if any(
            expected_final is not None and self.v2_engine.v1_engine._same_structure(path.final_outcome, expected_final)
            for path in decision.candidate_paths
        ) else 0.0
        reinforced_link_ids: List[int] = []
        if selected is not None:
            for step in selected.steps:
                if step.link_id is not None:
                    self.v2_engine.v1_engine.link_registry.update_usage(
                        link_id=step.link_id,
                        current_sequence=current_sequence,
                        success=success,
                    )
                    reinforced_link_ids.append(step.link_id)

        observed_feedback = [
            self._observe_chain_feedback(
                input_brain=normalized_input,
                chain=normalized_preferred,
                relation_types=preferred_relation_types,
                desirability=1.0,
                risk_level=0.05,
                current_sequence=current_sequence,
                note="preferred_path",
            )
        ]
        for chain, relation_types, desirability, risk_level in alternative_chains:
            normalized_chain = [
                self.v2_engine.v1_engine.native_engine.normalize(item, current_sequence=current_sequence)
                for item in chain
            ]
            observed_feedback.append(
                self._observe_chain_feedback(
                    input_brain=normalized_input,
                    chain=normalized_chain,
                    relation_types=relation_types,
                    desirability=desirability,
                    risk_level=risk_level,
                    current_sequence=current_sequence,
                    note="alternative_path",
                )
            )

        self.v2_engine.rebuild_concepts(current_sequence=current_sequence)
        return {
            "success": success,
            "selected_depth": selected.depth if selected is not None else 0,
            "decision_score": round(decision.decision_score, 4),
            "outcome_prediction_quality": round(candidate_quality, 4),
            "used_concept_abstraction": bool(
                selected is not None and any(step.source_level != "specific" for step in selected.steps)
            ),
            "selected_desirability": round(selected.desirability, 4) if selected is not None else 0.0,
            "selected_risk_level": round(selected.risk_level, 4) if selected is not None else 0.5,
            "selected_source_levels": [step.source_level for step in selected.steps] if selected is not None else [],
            "reinforced_link_ids": reinforced_link_ids,
            "observed_feedback": observed_feedback,
        }

    def registry_summary(self) -> Dict[str, object]:
        active_profiles = self.outcome_registry.all_active_profiles()
        average_desirability = round(
            sum(profile.desirability_score for profile in active_profiles) / max(1, len(active_profiles)),
            4,
        ) if active_profiles else 0.0
        average_risk = round(
            sum(profile.risk_level for profile in active_profiles) / max(1, len(active_profiles)),
            4,
        ) if active_profiles else 0.0
        return {
            **self.v2_engine.registry_summary(),
            "outcome_profile_count": len(active_profiles),
            "average_outcome_desirability": average_desirability,
            "average_outcome_risk": average_risk,
        }

    def _score_concept_path(
        self,
        path: ConceptReasoningPath,
        current_sequence: int,
    ) -> PlanningPath:
        outcome_desirability = 0.0
        outcome_risk = 0.5
        profile_confidence = 0.0
        states = [step.consequent for step in path.steps]
        if states:
            weights = [index + 1 for index in range(len(states))]
            total_weight = float(sum(weights))
            desirability_total = 0.0
            risk_total = 0.0
            confidence_total = 0.0
            for weight, state in zip(weights, states):
                desirability, risk_level, confidence = self._assess_outcome(state, current_sequence=current_sequence)
                normalized_weight = float(weight) / total_weight
                desirability_total += desirability * normalized_weight
                risk_total += risk_level * normalized_weight
                confidence_total += confidence * normalized_weight
            outcome_desirability = round(desirability_total, 4)
            outcome_risk = round(risk_total, 4)
            profile_confidence = round(confidence_total, 4)

        concept_support = self._concept_support_signal(path.steps)
        scaled_confidence = round(
            min(1.0, path.confidence + (concept_support * 0.06) + (profile_confidence * 0.08)),
            4,
        )
        decision_score = round(
            max(
                0.0,
                min(
                    1.0,
                    (scaled_confidence * 0.4)
                    + (((outcome_desirability + 1.0) / 2.0) * 0.45)
                    + ((1.0 - outcome_risk) * 0.15),
                ),
            ),
            4,
        )
        return PlanningPath(
            steps=path.steps,
            final_outcome=path.final_outcome,
            confidence=scaled_confidence,
            desirability=outcome_desirability,
            risk_level=outcome_risk,
            decision_score=decision_score,
            outcome_profile_confidence=profile_confidence,
        )

    def _observe_chain_feedback(
        self,
        input_brain: BrainInputV1,
        chain: Sequence[BrainInputV1],
        relation_types: Sequence[str],
        desirability: float,
        risk_level: float,
        current_sequence: int,
        note: str,
    ) -> Dict[str, object]:
        states = [input_brain] + list(chain)
        learned_link_ids: List[int] = []
        for index in range(len(states) - 1):
            link = self.v2_engine.learn_link(
                antecedent=states[index],
                consequent=states[index + 1],
                relation_type=self.v2_engine.v1_engine._relation_type_at(relation_types, index),
                current_sequence=current_sequence,
                notes=[f"observed_{note}"],
            )
            self.v2_engine.v1_engine.link_registry.update_usage(
                link_id=link.link_id,
                current_sequence=current_sequence,
                success=True,
            )
            learned_link_ids.append(link.link_id)

        updated_profile_ids: List[str] = []
        chain_length = max(1, len(chain))
        for index, state in enumerate(chain, start=1):
            weight = float(index) / float(chain_length)
            profile_id = self._brain_fingerprint(state)
            self.outcome_registry.update(
                profile_id=profile_id,
                outcome=state,
                desirability=desirability * weight,
                risk_level=min(1.0, max(0.0, risk_level * weight + (0.1 * (1.0 - weight)))),
                current_sequence=current_sequence,
            )
            updated_profile_ids.append(profile_id)

        return {
            "note": note,
            "learned_link_ids": learned_link_ids,
            "updated_profile_ids": updated_profile_ids,
        }

    def _assess_outcome(
        self,
        outcome: BrainInputV1,
        current_sequence: int,
    ) -> Tuple[float, float, float]:
        profile_id = self._brain_fingerprint(outcome)
        exact = self.outcome_registry.get(profile_id)
        if exact is not None:
            confidence = round(
                min(1.0, (min(1.0, exact.usage_count / max(1, current_sequence)) * 0.5) + (exact.temporal_score * 0.5)),
                4,
            )
            return exact.desirability_score, exact.risk_level, confidence

        similar_profiles: List[Tuple[OutcomeProfile, float]] = []
        for profile in self.outcome_registry.all_active_profiles():
            match_score = self.v2_engine.v1_engine._match_score(outcome, profile.outcome)
            if match_score >= 0.55:
                similar_profiles.append((profile, match_score))
        if not similar_profiles:
            return 0.0, 0.5, 0.0

        total_weight = sum(match_score for _, match_score in similar_profiles)
        desirability = sum(profile.desirability_score * match_score for profile, match_score in similar_profiles) / max(0.0001, total_weight)
        risk_level = sum(profile.risk_level * match_score for profile, match_score in similar_profiles) / max(0.0001, total_weight)
        confidence = min(1.0, total_weight / max(1.0, float(len(similar_profiles))))
        return round(desirability, 4), round(risk_level, 4), round(confidence, 4)

    def _concept_support_signal(self, steps: Sequence[ConceptChainStep]) -> float:
        concept_confidences: List[float] = []
        for step in steps:
            for concept_id in step.concept_ids:
                concept = self.v2_engine.concept_registry.get_concept(concept_id)
                concept_confidences.append(concept.confidence)
        if not concept_confidences:
            return 0.0
        return round(sum(concept_confidences) / len(concept_confidences), 4)

    def _brain_fingerprint(self, brain_input: BrainInputV1) -> str:
        bundle = self.v2_engine.v1_engine.runtime.codec.encode_memory_bundle(
            self.v2_engine.v1_engine.runtime.brain_input_codec.brain_input_to_memory(brain_input)
        )
        return json.dumps(bundle, sort_keys=True)

    @staticmethod
    def _planning_sort_key(path: PlanningPath) -> Tuple[int, ...]:
        values: List[int] = []
        for step in path.steps:
            if step.link_id is not None:
                values.append(step.link_id)
            elif step.concept_link_id is not None:
                values.append(10_000 + step.concept_link_id)
            else:
                values.append(99_999)
        return tuple(values)


def memory_to_brain_input(
    engine: BrainPlanningSimulationV3,
    memory: BrainMemory,
    current_sequence: int,
) -> BrainInputV1:
    base = engine.v2_engine.v1_engine.runtime.brain_input_codec.build_brain_input(
        memory=memory,
        encoding_type="bitpacked",
        primitive_ids=[],
    )
    return engine.v2_engine.v1_engine.native_engine.normalize(base, current_sequence=current_sequence)
