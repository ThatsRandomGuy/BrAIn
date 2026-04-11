from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from brain_concept_abstraction_v2 import ConceptChainStep, ConceptLink
from brain_generative_expansion_v35 import BrainGenerativeExpansionV35
from brain_input_v1 import BrainInputV1
from brain_planning_simulation_v3 import PlanningDecision, PlanningPath
from brain_reasoning_engine import DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class GroundingSignal:
    signal_id: str
    signal_type: str
    usage_count: int
    success_count: int
    failure_count: int
    reliability: float
    plausibility_score: float
    temporal_score: float
    flagged_unreliable: bool = False
    active: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "reliability": round(self.reliability, 4),
            "plausibility_score": round(self.plausibility_score, 4),
            "temporal_score": round(self.temporal_score, 4),
            "flagged_unreliable": self.flagged_unreliable,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "GroundingSignal":
        return cls(
            signal_id=str(data.get("signal_id", "")),
            signal_type=str(data.get("signal_type", "")),
            usage_count=int(data.get("usage_count", 0)),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            reliability=float(data.get("reliability", 0.5)),
            plausibility_score=float(data.get("plausibility_score", 0.5)),
            temporal_score=float(data.get("temporal_score", 0.0)),
            flagged_unreliable=bool(data.get("flagged_unreliable", False)),
            active=bool(data.get("active", True)),
        )


class GroundingSignalRegistry:
    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)
        self.signals: Dict[str, GroundingSignal] = {}
        self._load()

    def _load(self) -> None:
        if not self.registry_path.exists():
            self.signals = {}
            return
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded: Dict[str, GroundingSignal] = {}
        if isinstance(data, dict):
            for item in data.get("signals", []):
                if not isinstance(item, dict):
                    continue
                signal = GroundingSignal.from_dict(item)
                if signal.signal_id:
                    loaded[signal.signal_id] = signal
        self.signals = loaded

    def save(self) -> None:
        payload = {
            "signals": [self.signals[key].to_dict() for key in sorted(self.signals)],
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def get(self, signal_id: str) -> Optional[GroundingSignal]:
        return self.signals.get(str(signal_id))

    def all_active_signals(self) -> List[GroundingSignal]:
        return [self.signals[key] for key in sorted(self.signals) if self.signals[key].active]

    def update(
        self,
        signal_id: str,
        signal_type: str,
        success: bool,
        plausibility_score: float,
        current_sequence: int,
    ) -> GroundingSignal:
        existing = self.signals.get(signal_id)
        if existing is None:
            usage_count = 1
            success_count = 1 if success else 0
            failure_count = 0 if success else 1
            reliability = 1.0 if success else 0.0
            average_plausibility = max(0.0, min(1.0, plausibility_score))
        else:
            usage_count = existing.usage_count + 1
            success_count = existing.success_count + (1 if success else 0)
            failure_count = existing.failure_count + (0 if success else 1)
            reliability = success_count / max(1, success_count + failure_count)
            average_plausibility = (
                (existing.plausibility_score * existing.usage_count) + max(0.0, min(1.0, plausibility_score))
            ) / max(1, usage_count)
        recency = min(1.0, usage_count / max(1, current_sequence))
        temporal_score = round(
            max(0.0, min(1.0, (reliability * 0.55) + (average_plausibility * 0.3) + (recency * 0.15))),
            4,
        )
        flagged_unreliable = failure_count >= 2 and failure_count > success_count
        active = not (failure_count >= 4 and reliability < 0.35)
        updated = GroundingSignal(
            signal_id=signal_id,
            signal_type=signal_type,
            usage_count=usage_count,
            success_count=success_count,
            failure_count=failure_count,
            reliability=round(reliability, 4),
            plausibility_score=round(average_plausibility, 4),
            temporal_score=temporal_score,
            flagged_unreliable=flagged_unreliable,
            active=active,
        )
        self.signals[signal_id] = updated
        self.save()
        return updated


@dataclass(frozen=True)
class GroundedPlanningPath:
    steps: List[ConceptChainStep]
    final_outcome: BrainInputV1
    confidence: float
    desirability: float
    risk_level: float
    decision_score: float
    outcome_profile_confidence: float
    plausibility_score: float
    frequency_score: float
    consistency_score: float
    concept_alignment_score: float
    cross_path_support: float
    grounded_score: float
    filtered_out: bool = False
    unreliable_signal_count: int = 0

    @property
    def depth(self) -> int:
        return len(self.steps)

    def to_planning_path(self) -> PlanningPath:
        return PlanningPath(
            steps=self.steps,
            final_outcome=self.final_outcome,
            confidence=self.confidence,
            desirability=self.desirability,
            risk_level=self.risk_level,
            decision_score=self.decision_score,
            outcome_profile_confidence=self.outcome_profile_confidence,
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
            "plausibility_score": round(self.plausibility_score, 4),
            "frequency_score": round(self.frequency_score, 4),
            "consistency_score": round(self.consistency_score, 4),
            "concept_alignment_score": round(self.concept_alignment_score, 4),
            "cross_path_support": round(self.cross_path_support, 4),
            "grounded_score": round(self.grounded_score, 4),
            "filtered_out": self.filtered_out,
            "unreliable_signal_count": self.unreliable_signal_count,
            "depth": self.depth,
        }


@dataclass(frozen=True)
class GroundedPlanningDecision:
    input_brain: BrainInputV1
    candidate_paths: List[GroundedPlanningPath]
    retained_paths: List[GroundedPlanningPath]
    selected_path: Optional[GroundedPlanningPath]
    highest_confidence_path: Optional[GroundedPlanningPath]
    highest_plausibility_path: Optional[GroundedPlanningPath]
    decision_score: float
    final_predicted_outcome: Optional[BrainInputV1]
    filtered_path_count: int

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "input_brain": self.input_brain.to_dict(),
            "candidate_paths": [path.to_dict() for path in self.candidate_paths],
            "retained_paths": [path.to_dict() for path in self.retained_paths],
            "decision_score": round(self.decision_score, 4),
            "filtered_path_count": self.filtered_path_count,
        }
        if self.selected_path is not None:
            payload["selected_path"] = self.selected_path.to_dict()
        if self.highest_confidence_path is not None:
            payload["highest_confidence_path"] = self.highest_confidence_path.to_dict()
        if self.highest_plausibility_path is not None:
            payload["highest_plausibility_path"] = self.highest_plausibility_path.to_dict()
        if self.final_predicted_outcome is not None:
            payload["final_predicted_outcome"] = self.final_predicted_outcome.to_dict()
        return payload


class BrainRealityGroundingV4:
    def __init__(
        self,
        state_dir: str | Path = DEFAULT_OUTPUT_DIR / "brain_reality_grounding_v4.state",
        minimum_retained_paths: int = 2,
        plausibility_threshold: float = 0.34,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.v35_engine = BrainGenerativeExpansionV35(state_dir=self.state_dir / "v35")
        self.grounding_registry = GroundingSignalRegistry(self.state_dir / "brain_grounding_signals.json")
        self.minimum_retained_paths = minimum_retained_paths
        self.plausibility_threshold = plausibility_threshold

    def plan(
        self,
        brain_input: BrainInputV1,
        max_depth: int = 5,
        beam_width: int = 4,
        current_sequence: int = 1,
    ) -> GroundedPlanningDecision:
        base_decision = self.v35_engine.plan(
            brain_input=brain_input,
            max_depth=max_depth,
            beam_width=beam_width,
            current_sequence=current_sequence,
        )
        grounded_paths = self._ground_candidate_paths(
            root=base_decision.input_brain,
            candidate_paths=base_decision.candidate_paths,
            current_sequence=current_sequence,
        )
        retained_paths = [path for path in grounded_paths if not path.filtered_out]
        retained_paths = sorted(
            retained_paths,
            key=lambda item: (
                -item.grounded_score,
                -item.plausibility_score,
                -item.confidence,
                -item.depth,
                self._planning_sort_key(item.steps),
            ),
        )
        selected_path = retained_paths[0] if retained_paths else None
        highest_confidence_path = max(
            grounded_paths,
            key=lambda item: (item.confidence, item.plausibility_score, item.grounded_score, item.depth, tuple(self._planning_sort_key(item.steps))),
            default=None,
        )
        highest_plausibility_path = max(
            grounded_paths,
            key=lambda item: (item.plausibility_score, item.grounded_score, item.confidence, item.depth, tuple(self._planning_sort_key(item.steps))),
            default=None,
        )
        return GroundedPlanningDecision(
            input_brain=base_decision.input_brain,
            candidate_paths=grounded_paths,
            retained_paths=retained_paths,
            selected_path=selected_path,
            highest_confidence_path=highest_confidence_path,
            highest_plausibility_path=highest_plausibility_path,
            decision_score=selected_path.grounded_score if selected_path is not None else 0.0,
            final_predicted_outcome=selected_path.final_outcome if selected_path is not None else None,
            filtered_path_count=sum(1 for path in grounded_paths if path.filtered_out),
        )

    def validate_decision(
        self,
        input_brain: BrainInputV1,
        decision: GroundedPlanningDecision,
        preferred_chain: Sequence[BrainInputV1],
        preferred_relation_types: Sequence[str],
        alternative_chains: Sequence[Tuple[Sequence[BrainInputV1], Sequence[str], float, float]],
        current_sequence: int,
    ) -> Dict[str, object]:
        base_decision = PlanningDecision(
            input_brain=decision.input_brain,
            candidate_paths=[path.to_planning_path() for path in decision.retained_paths],
            selected_path=decision.selected_path.to_planning_path() if decision.selected_path is not None else None,
            highest_confidence_path=decision.highest_confidence_path.to_planning_path() if decision.highest_confidence_path is not None else None,
            decision_score=decision.selected_path.decision_score if decision.selected_path is not None else 0.0,
            final_predicted_outcome=decision.selected_path.final_outcome if decision.selected_path is not None else None,
        )
        validation = self.v35_engine.validate_decision(
            input_brain=input_brain,
            decision=base_decision,
            preferred_chain=preferred_chain,
            preferred_relation_types=preferred_relation_types,
            alternative_chains=alternative_chains,
            current_sequence=current_sequence,
        )
        selected = decision.selected_path
        updated_signal_ids: List[str] = []
        if selected is not None:
            success = bool(validation.get("success", False))
            for signal_id, signal_type in self._signal_targets_for_path(selected):
                self.grounding_registry.update(
                    signal_id=signal_id,
                    signal_type=signal_type,
                    success=success,
                    plausibility_score=selected.plausibility_score,
                    current_sequence=current_sequence,
                )
                updated_signal_ids.append(signal_id)
            root_outcome_signal = self._root_outcome_signal_id(input_brain, selected.final_outcome)
            self.grounding_registry.update(
                signal_id=root_outcome_signal,
                signal_type="root_outcome",
                success=success,
                plausibility_score=selected.plausibility_score,
                current_sequence=current_sequence,
            )
            updated_signal_ids.append(root_outcome_signal)
        return {
            **validation,
            "selected_plausibility_score": round(selected.plausibility_score, 4) if selected is not None else 0.0,
            "selected_grounded_score": round(selected.grounded_score, 4) if selected is not None else 0.0,
            "selected_cross_path_support": round(selected.cross_path_support, 4) if selected is not None else 0.0,
            "selected_consistency_score": round(selected.consistency_score, 4) if selected is not None else 0.0,
            "selected_concept_alignment_score": round(selected.concept_alignment_score, 4) if selected is not None else 0.0,
            "selected_frequency_score": round(selected.frequency_score, 4) if selected is not None else 0.0,
            "filtered_path_count": decision.filtered_path_count,
            "retained_path_count": len(decision.retained_paths),
            "used_reality_grounding": bool(decision.filtered_path_count > 0),
            "grounding_signal_ids": updated_signal_ids,
        }

    def registry_summary(self) -> Dict[str, object]:
        active_signals = self.grounding_registry.all_active_signals()
        average_grounding_reliability = round(
            sum(signal.reliability for signal in active_signals) / max(1, len(active_signals)),
            4,
        ) if active_signals else 0.0
        average_grounding_plausibility = round(
            sum(signal.plausibility_score for signal in active_signals) / max(1, len(active_signals)),
            4,
        ) if active_signals else 0.0
        unreliable_signal_count = sum(1 for signal in active_signals if signal.flagged_unreliable)
        return {
            **self.v35_engine.registry_summary(),
            "grounding_signal_count": len(active_signals),
            "average_grounding_reliability": average_grounding_reliability,
            "average_grounding_plausibility": average_grounding_plausibility,
            "unreliable_signal_count": unreliable_signal_count,
        }

    def _ground_candidate_paths(
        self,
        root: BrainInputV1,
        candidate_paths: Sequence[PlanningPath],
        current_sequence: int,
    ) -> List[GroundedPlanningPath]:
        raw_metrics: List[Dict[str, float]] = [
            self._grounding_metrics_for_path(root=root, path=path, current_sequence=current_sequence)
            for path in candidate_paths
        ]
        support_signals = self._cross_path_support(candidate_paths)
        retained_indexes = self._retained_indexes(raw_metrics, support_signals)
        grounded_paths: List[GroundedPlanningPath] = []
        for index, path in enumerate(candidate_paths):
            metrics = raw_metrics[index]
            cross_support = support_signals[index]
            plausibility = round(
                max(
                    0.0,
                    min(
                        1.0,
                        (metrics["frequency"] * 0.24)
                        + (metrics["consistency"] * 0.34)
                        + (metrics["concept_alignment"] * 0.22)
                        + (cross_support * 0.20)
                        - (metrics["unreliable_penalty"] * 0.18),
                    ),
                ),
                4,
            )
            desirability_signal = (path.desirability + 1.0) / 2.0
            grounded_score = round(
                max(
                    0.0,
                    min(
                        1.0,
                        (path.decision_score * 0.3)
                        + (plausibility * 0.28)
                        + (path.confidence * 0.18)
                        + (desirability_signal * 0.12)
                        + ((1.0 - path.risk_level) * 0.06)
                        + (path.outcome_profile_confidence * 0.06),
                    ),
                ),
                4,
            )
            grounded_paths.append(
                GroundedPlanningPath(
                    steps=path.steps,
                    final_outcome=path.final_outcome,
                    confidence=path.confidence,
                    desirability=path.desirability,
                    risk_level=path.risk_level,
                    decision_score=path.decision_score,
                    outcome_profile_confidence=path.outcome_profile_confidence,
                    plausibility_score=plausibility,
                    frequency_score=metrics["frequency"],
                    consistency_score=metrics["consistency"],
                    concept_alignment_score=metrics["concept_alignment"],
                    cross_path_support=cross_support,
                    grounded_score=grounded_score,
                    filtered_out=index not in retained_indexes,
                    unreliable_signal_count=int(metrics["unreliable_count"]),
                )
            )
        return sorted(
            grounded_paths,
            key=lambda item: (
                item.filtered_out,
                -item.grounded_score,
                -item.plausibility_score,
                -item.confidence,
                -item.depth,
                self._planning_sort_key(item.steps),
            ),
        )

    def _grounding_metrics_for_path(
        self,
        root: BrainInputV1,
        path: PlanningPath,
        current_sequence: int,
    ) -> Dict[str, float]:
        frequency_signals: List[float] = []
        consistency_signals: List[float] = []
        concept_alignment_signals: List[float] = []
        unreliable_count = 0

        for step in path.steps:
            step_frequency, step_consistency, step_alignment, step_unreliable = self._score_step_grounding(
                step=step,
                current_sequence=current_sequence,
            )
            frequency_signals.append(step_frequency)
            consistency_signals.append(step_consistency)
            concept_alignment_signals.append(step_alignment)
            unreliable_count += step_unreliable

        outcome_frequency, outcome_consistency, outcome_unreliable = self._score_outcome_grounding(
            root=root,
            outcome=path.final_outcome,
            current_sequence=current_sequence,
        )
        frequency_signals.append(outcome_frequency)
        consistency_signals.append(outcome_consistency)
        concept_alignment_signals.append(max(path.outcome_profile_confidence, self._concept_support_signal(path.steps)))
        unreliable_penalty = 0.0
        if outcome_unreliable:
            unreliable_count += 1
            unreliable_penalty += 0.4
        unreliable_penalty += min(0.6, float(unreliable_count) * 0.15)

        return {
            "frequency": round(sum(frequency_signals) / max(1, len(frequency_signals)), 4),
            "consistency": round(sum(consistency_signals) / max(1, len(consistency_signals)), 4),
            "concept_alignment": round(sum(concept_alignment_signals) / max(1, len(concept_alignment_signals)), 4),
            "unreliable_penalty": round(unreliable_penalty, 4),
            "unreliable_count": float(unreliable_count),
        }

    def _score_step_grounding(
        self,
        step: ConceptChainStep,
        current_sequence: int,
    ) -> Tuple[float, float, float, int]:
        v2_engine = self.v35_engine.v3_engine.v2_engine
        frequency = 0.08
        consistency = max(0.08, min(1.0, step.match_score))
        concept_alignment = max(0.12, min(1.0, step.match_score))
        unreliable_count = 0

        if step.link_id is not None:
            link = v2_engine.v1_engine.link_registry.get(step.link_id)
            frequency = min(1.0, link.usage_count / max(1, current_sequence))
            consistency = min(1.0, (link.reliability * 0.75) + (link.temporal_score * 0.25))
            concept_alignment = min(1.0, 0.7 + (link.reliability * 0.3))
            signal = self.grounding_registry.get(f"link:{step.link_id}")
            if signal is not None:
                consistency = min(1.0, (consistency * 0.65) + (signal.temporal_score * 0.35))
                if signal.flagged_unreliable:
                    consistency *= 0.5
                    concept_alignment *= 0.7
                    unreliable_count += 1
        elif step.concept_link_id is not None:
            concept_link = v2_engine.concept_registry.get_concept_link(step.concept_link_id)
            support_signal = min(1.0, concept_link.support_count / 4.0)
            concept_alignment = self._concept_consistency_score(step, concept_link)
            if step.source_level == "generated":
                rule = self.v35_engine.generated_registry.get(
                    self._generated_rule_id(
                        relation_type=step.relation_type,
                        consequent=step.consequent,
                        concept_link_id=step.concept_link_id,
                    )
                )
                rule_reliability = rule.reliability if rule is not None else 0.5
                rule_frequency = min(1.0, (rule.usage_count / max(1, current_sequence))) if rule is not None else 0.0
                frequency = min(1.0, (support_signal * 0.55) + (rule_frequency * 0.45))
                consistency = min(1.0, (concept_link.reliability * 0.45) + (rule_reliability * 0.35) + (step.match_score * 0.2))
                signal = self.grounding_registry.get(
                    self._generated_signal_id(step.relation_type, step.concept_link_id, step.consequent)
                )
                if signal is not None:
                    consistency = min(1.0, (consistency * 0.6) + (signal.temporal_score * 0.4))
                    if signal.flagged_unreliable:
                        consistency *= 0.45
                        concept_alignment *= 0.65
                        unreliable_count += 1
            else:
                frequency = support_signal
                consistency = min(1.0, (concept_link.reliability * 0.7) + (step.match_score * 0.3))
                signal = self.grounding_registry.get(f"concept:{step.concept_link_id}")
                if signal is not None:
                    consistency = min(1.0, (consistency * 0.7) + (signal.temporal_score * 0.3))
                    if signal.flagged_unreliable:
                        consistency *= 0.55
                        concept_alignment *= 0.7
                        unreliable_count += 1

        return round(frequency, 4), round(consistency, 4), round(concept_alignment, 4), unreliable_count

    def _score_outcome_grounding(
        self,
        root: BrainInputV1,
        outcome: BrainInputV1,
        current_sequence: int,
    ) -> Tuple[float, float, bool]:
        v3_engine = self.v35_engine.v3_engine
        _, _, profile_confidence = v3_engine._assess_outcome(outcome, current_sequence=current_sequence)
        signal = self.grounding_registry.get(self._outcome_signal_id(outcome))
        root_signal = self.grounding_registry.get(self._root_outcome_signal_id(root, outcome))
        if signal is None:
            if root_signal is None:
                return round(profile_confidence, 4), round(profile_confidence, 4), False
            frequency = min(1.0, (profile_confidence * 0.25) + (min(1.0, root_signal.usage_count / max(1, current_sequence)) * 0.75))
            consistency = min(1.0, (profile_confidence * 0.2) + (root_signal.reliability * 0.45) + (root_signal.temporal_score * 0.35))
            return round(frequency, 4), round(consistency, 4), root_signal.flagged_unreliable
        frequency = min(1.0, (profile_confidence * 0.45) + (min(1.0, signal.usage_count / max(1, current_sequence)) * 0.55))
        consistency = min(1.0, (profile_confidence * 0.35) + (signal.reliability * 0.35) + (signal.temporal_score * 0.3))
        flagged_unreliable = signal.flagged_unreliable
        if root_signal is not None:
            root_frequency = min(1.0, root_signal.usage_count / max(1, current_sequence))
            frequency = min(1.0, (frequency * 0.55) + (root_frequency * 0.45))
            consistency = min(1.0, (consistency * 0.45) + (root_signal.reliability * 0.3) + (root_signal.temporal_score * 0.25))
            flagged_unreliable = flagged_unreliable or root_signal.flagged_unreliable
        return round(frequency, 4), round(consistency, 4), flagged_unreliable

    def _concept_consistency_score(
        self,
        step: ConceptChainStep,
        concept_link: ConceptLink,
    ) -> float:
        v2_engine = self.v35_engine.v3_engine.v2_engine
        antecedent_memory = v2_engine._memory_from_brain(step.antecedent)
        scores: List[float] = []
        for slot_path, concept_id in sorted(concept_link.concept_bindings.items()):
            concept = v2_engine.concept_registry.get_concept(concept_id)
            slot_value = v2_engine._get_slot_path_value(antecedent_memory, slot_path)
            if slot_value in concept.members:
                membership = 1.0
            elif slot_value:
                membership = 0.35 if concept.support_count >= 2 else 0.0
            else:
                membership = 0.0
            scores.append(min(1.0, (membership * 0.6) + (concept.confidence * 0.4)))
        if not scores:
            return 0.15
        return round(sum(scores) / len(scores), 4)

    def _cross_path_support(self, candidate_paths: Sequence[PlanningPath]) -> List[float]:
        match_score = self.v35_engine.v3_engine.v2_engine.v1_engine._match_score
        supports: List[float] = []
        for index, path in enumerate(candidate_paths):
            final_supports: List[float] = []
            intermediate_supports: List[float] = []
            current_step_states = [step.consequent for step in path.steps]
            for other_index, other in enumerate(candidate_paths):
                if index == other_index:
                    continue
                outcome_match = match_score(path.final_outcome, other.final_outcome)
                if outcome_match >= 0.55:
                    final_supports.append(
                        min(1.0, (outcome_match * 0.7) + (other.confidence * 0.3))
                    )
                other_states = [step.consequent for step in other.steps]
                best_intermediate = 0.0
                for state in current_step_states:
                    for other_state in other_states:
                        best_intermediate = max(best_intermediate, match_score(state, other_state))
                if best_intermediate >= 0.6:
                    intermediate_supports.append(best_intermediate)
            if not final_supports and not intermediate_supports:
                supports.append(0.0)
                continue
            final_average = sum(final_supports) / max(1, len(final_supports))
            intermediate_average = sum(intermediate_supports) / max(1, len(intermediate_supports))
            supports.append(round(min(1.0, (final_average * 0.75) + (intermediate_average * 0.25)), 4))
        return supports

    def _retained_indexes(
        self,
        metrics: Sequence[Dict[str, float]],
        support_signals: Sequence[float],
    ) -> set[int]:
        scored: List[Tuple[int, float]] = []
        for index, metric in enumerate(metrics):
            score = (
                (metric["frequency"] * 0.24)
                + (metric["consistency"] * 0.34)
                + (metric["concept_alignment"] * 0.22)
                + (support_signals[index] * 0.20)
                - (metric["unreliable_penalty"] * 0.18)
            )
            scored.append((index, score))
        scored = sorted(scored, key=lambda item: (-item[1], item[0]))
        retained = {
            index
            for index, score in scored
            if score >= self.plausibility_threshold
        }
        minimum_to_keep = min(len(scored), max(1, self.minimum_retained_paths))
        if len(retained) < minimum_to_keep:
            retained = {index for index, _ in scored[:minimum_to_keep]}
        return retained

    def _signal_targets_for_path(self, path: GroundedPlanningPath) -> List[Tuple[str, str]]:
        targets: List[Tuple[str, str]] = [(self._outcome_signal_id(path.final_outcome), "outcome")]
        for step in path.steps:
            if step.link_id is not None:
                targets.append((f"link:{step.link_id}", "specific_link"))
            elif step.concept_link_id is not None and step.source_level == "generated":
                targets.append((
                    self._generated_signal_id(step.relation_type, step.concept_link_id, step.consequent),
                    "generated_pattern",
                ))
            elif step.concept_link_id is not None:
                targets.append((f"concept:{step.concept_link_id}", "concept_link"))
        unique_targets: List[Tuple[str, str]] = []
        seen: set[str] = set()
        for signal_id, signal_type in targets:
            if signal_id in seen:
                continue
            seen.add(signal_id)
            unique_targets.append((signal_id, signal_type))
        return unique_targets

    def _generated_rule_id(
        self,
        relation_type: str,
        consequent: BrainInputV1,
        concept_link_id: Optional[int],
    ) -> str:
        raw = f"{relation_type}|{concept_link_id or 0}|{self._brain_fingerprint(consequent)}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _generated_signal_id(
        self,
        relation_type: str,
        concept_link_id: Optional[int],
        consequent: BrainInputV1,
    ) -> str:
        return f"generated:{self._generated_rule_id(relation_type, consequent, concept_link_id)}"

    def _outcome_signal_id(self, outcome: BrainInputV1) -> str:
        return f"outcome:{self._brain_fingerprint(outcome)}"

    def _root_outcome_signal_id(self, root: BrainInputV1, outcome: BrainInputV1) -> str:
        return f"root_outcome:{self._brain_fingerprint(root)}->{self._brain_fingerprint(outcome)}"

    def _brain_fingerprint(self, brain_input: BrainInputV1) -> str:
        v1_engine = self.v35_engine.v3_engine.v2_engine.v1_engine
        bundle = v1_engine.runtime.codec.encode_memory_bundle(
            v1_engine.runtime.brain_input_codec.brain_input_to_memory(brain_input)
        )
        return json.dumps(bundle, sort_keys=True)

    def _concept_support_signal(self, steps: Sequence[ConceptChainStep]) -> float:
        concept_confidences: List[float] = []
        concept_registry = self.v35_engine.v3_engine.v2_engine.concept_registry
        for step in steps:
            for concept_id in step.concept_ids:
                concept_confidences.append(concept_registry.get_concept(concept_id).confidence)
        if not concept_confidences:
            return 0.0
        return round(sum(concept_confidences) / len(concept_confidences), 4)

    @staticmethod
    def _planning_sort_key(steps: Sequence[ConceptChainStep]) -> Tuple[int, ...]:
        values: List[int] = []
        for step in steps:
            if step.link_id is not None:
                values.append(step.link_id)
            elif step.concept_link_id is not None:
                values.append(10_000 + step.concept_link_id)
            else:
                values.append(99_999)
        return tuple(values)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="brain_reality_grounding_v4",
        description="Run a simple deterministic BrAIn V4 grounded planning pass.",
    )
    parser.add_argument(
        "--state-dir",
        default=str(DEFAULT_OUTPUT_DIR / "brain_reality_grounding_v4.state"),
        help="State directory for the BrAIn V4 reality grounding engine.",
    )
    args = parser.parse_args()
    engine = BrainRealityGroundingV4(state_dir=args.state_dir)
    summary_path = Path(args.state_dir) / "brain_reality_grounding_v4.summary.json"
    summary_path.write_text(json.dumps(engine.registry_summary(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"Reality grounding registry summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
