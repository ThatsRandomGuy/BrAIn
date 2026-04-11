from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from brain_input_v1 import BrainInputV1
from brain_reality_grounding_v4 import (
    BrainRealityGroundingV4,
    GroundedPlanningDecision,
    GroundedPlanningPath,
)
from brain_reasoning_engine import DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class GoalTarget:
    target_id: str
    target_brain: BrainInputV1
    polarity: str
    weight: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "target_id": self.target_id,
            "target_brain": self.target_brain.to_dict(),
            "polarity": self.polarity,
            "weight": round(self.weight, 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "GoalTarget":
        return cls(
            target_id=str(data.get("target_id", "")),
            target_brain=BrainInputV1.from_dict(
                data.get("target_brain", {}) if isinstance(data.get("target_brain", {}), dict) else {}
            ),
            polarity=str(data.get("polarity", "seek")),
            weight=float(data.get("weight", 1.0)),
        )


@dataclass(frozen=True)
class GoalNode:
    goal_id: str
    base_weight: float
    targets: List[GoalTarget]
    usage_count: int = 0
    success_count: int = 0
    average_reward: float = 0.0
    average_alignment: float = 0.0
    temporal_score: float = 0.0
    active: bool = True
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "goal_id": self.goal_id,
            "base_weight": round(self.base_weight, 4),
            "targets": [target.to_dict() for target in self.targets],
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "average_reward": round(self.average_reward, 4),
            "average_alignment": round(self.average_alignment, 4),
            "temporal_score": round(self.temporal_score, 4),
            "active": self.active,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "GoalNode":
        targets = data.get("targets", [])
        return cls(
            goal_id=str(data.get("goal_id", "")),
            base_weight=float(data.get("base_weight", 1.0)),
            targets=[GoalTarget.from_dict(item) for item in targets if isinstance(item, dict)] if isinstance(targets, list) else [],
            usage_count=int(data.get("usage_count", 0)),
            success_count=int(data.get("success_count", 0)),
            average_reward=float(data.get("average_reward", 0.0)),
            average_alignment=float(data.get("average_alignment", 0.0)),
            temporal_score=float(data.get("temporal_score", 0.0)),
            active=bool(data.get("active", True)),
            notes=[str(item) for item in data.get("notes", [])] if isinstance(data.get("notes", []), list) else [],
        )


class GoalRegistry:
    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)
        self.goals: Dict[str, GoalNode] = {}
        self._load()

    def _load(self) -> None:
        if not self.registry_path.exists():
            self.goals = {}
            return
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded: Dict[str, GoalNode] = {}
        if isinstance(data, dict):
            for item in data.get("goals", []):
                if not isinstance(item, dict):
                    continue
                goal = GoalNode.from_dict(item)
                if goal.goal_id:
                    loaded[goal.goal_id] = goal
        self.goals = loaded

    def save(self) -> None:
        payload = {
            "goals": [self.goals[key].to_dict() for key in sorted(self.goals)],
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def get(self, goal_id: str) -> Optional[GoalNode]:
        return self.goals.get(str(goal_id))

    def all_active_goals(self) -> List[GoalNode]:
        return [self.goals[key] for key in sorted(self.goals) if self.goals[key].active]

    def ensure_goal(
        self,
        goal_id: str,
        targets: Sequence[GoalTarget],
        base_weight: float = 1.0,
        notes: Optional[Sequence[str]] = None,
    ) -> GoalNode:
        existing = self.goals.get(goal_id)
        goal = GoalNode(
            goal_id=goal_id,
            base_weight=base_weight,
            targets=list(targets),
            usage_count=existing.usage_count if existing is not None else 0,
            success_count=existing.success_count if existing is not None else 0,
            average_reward=existing.average_reward if existing is not None else 0.0,
            average_alignment=existing.average_alignment if existing is not None else 0.0,
            temporal_score=existing.temporal_score if existing is not None else 0.0,
            active=True,
            notes=[str(item) for item in (notes or [])],
        )
        self.goals[goal_id] = goal
        self.save()
        return goal

    def update_feedback(
        self,
        goal_id: str,
        success: bool,
        reward: float,
        alignment: float,
        current_sequence: int,
    ) -> GoalNode:
        goal = self.goals[goal_id]
        usage_count = goal.usage_count + 1
        success_count = goal.success_count + (1 if success else 0)
        average_reward = round(((goal.average_reward * goal.usage_count) + reward) / max(1, usage_count), 4)
        average_alignment = round(((goal.average_alignment * goal.usage_count) + alignment) / max(1, usage_count), 4)
        success_rate = success_count / max(1, usage_count)
        recency = min(1.0, usage_count / max(1, current_sequence))
        temporal_score = round(
            max(0.0, min(1.0, (success_rate * 0.45) + (((average_reward + 1.0) / 2.0) * 0.3) + (recency * 0.25))),
            4,
        )
        updated = GoalNode(
            goal_id=goal.goal_id,
            base_weight=goal.base_weight,
            targets=goal.targets,
            usage_count=usage_count,
            success_count=success_count,
            average_reward=average_reward,
            average_alignment=average_alignment,
            temporal_score=temporal_score,
            active=True,
            notes=goal.notes,
        )
        self.goals[goal_id] = updated
        self.save()
        return updated


@dataclass(frozen=True)
class GoalOutcomeStat:
    stat_id: str
    goal_id: str
    outcome: BrainInputV1
    usage_count: int
    success_count: int
    average_reward: float
    average_alignment: float
    reliability: float
    temporal_score: float
    active: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "stat_id": self.stat_id,
            "goal_id": self.goal_id,
            "outcome": self.outcome.to_dict(),
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "average_reward": round(self.average_reward, 4),
            "average_alignment": round(self.average_alignment, 4),
            "reliability": round(self.reliability, 4),
            "temporal_score": round(self.temporal_score, 4),
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "GoalOutcomeStat":
        return cls(
            stat_id=str(data.get("stat_id", "")),
            goal_id=str(data.get("goal_id", "")),
            outcome=BrainInputV1.from_dict(data.get("outcome", {}) if isinstance(data.get("outcome", {}), dict) else {}),
            usage_count=int(data.get("usage_count", 0)),
            success_count=int(data.get("success_count", 0)),
            average_reward=float(data.get("average_reward", 0.0)),
            average_alignment=float(data.get("average_alignment", 0.0)),
            reliability=float(data.get("reliability", 0.0)),
            temporal_score=float(data.get("temporal_score", 0.0)),
            active=bool(data.get("active", True)),
        )


class GoalOutcomeRegistry:
    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)
        self.stats: Dict[str, GoalOutcomeStat] = {}
        self._load()

    def _load(self) -> None:
        if not self.registry_path.exists():
            self.stats = {}
            return
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded: Dict[str, GoalOutcomeStat] = {}
        if isinstance(data, dict):
            for item in data.get("stats", []):
                if not isinstance(item, dict):
                    continue
                stat = GoalOutcomeStat.from_dict(item)
                if stat.stat_id:
                    loaded[stat.stat_id] = stat
        self.stats = loaded

    def save(self) -> None:
        payload = {
            "stats": [self.stats[key].to_dict() for key in sorted(self.stats)],
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def get(self, stat_id: str) -> Optional[GoalOutcomeStat]:
        return self.stats.get(str(stat_id))

    def all_active_stats(self) -> List[GoalOutcomeStat]:
        return [self.stats[key] for key in sorted(self.stats) if self.stats[key].active]

    def update(
        self,
        stat_id: str,
        goal_id: str,
        outcome: BrainInputV1,
        success: bool,
        reward: float,
        alignment: float,
        current_sequence: int,
    ) -> GoalOutcomeStat:
        existing = self.stats.get(stat_id)
        usage_count = (existing.usage_count if existing is not None else 0) + 1
        success_count = (existing.success_count if existing is not None else 0) + (1 if success else 0)
        average_reward = round(
            (((existing.average_reward if existing is not None else 0.0) * (existing.usage_count if existing is not None else 0)) + reward) / max(1, usage_count),
            4,
        )
        average_alignment = round(
            (((existing.average_alignment if existing is not None else 0.0) * (existing.usage_count if existing is not None else 0)) + alignment) / max(1, usage_count),
            4,
        )
        reliability = round(success_count / max(1, usage_count), 4)
        recency = min(1.0, usage_count / max(1, current_sequence))
        temporal_score = round(
            max(0.0, min(1.0, (reliability * 0.45) + (((average_reward + 1.0) / 2.0) * 0.25) + (average_alignment * 0.15) + (recency * 0.15))),
            4,
        )
        updated = GoalOutcomeStat(
            stat_id=stat_id,
            goal_id=goal_id,
            outcome=outcome,
            usage_count=usage_count,
            success_count=success_count,
            average_reward=average_reward,
            average_alignment=average_alignment,
            reliability=reliability,
            temporal_score=temporal_score,
            active=True,
        )
        self.stats[stat_id] = updated
        self.save()
        return updated


@dataclass(frozen=True)
class DecisionEpisode:
    episode_id: str
    sequence: int
    input_brain: BrainInputV1
    goal_weights: Dict[str, float]
    selected_outcome: Optional[BrainInputV1]
    success: bool
    reward: float
    goal_alignment_score: float
    goal_weighted_score: float

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "episode_id": self.episode_id,
            "sequence": self.sequence,
            "input_brain": self.input_brain.to_dict(),
            "goal_weights": {str(key): round(float(value), 4) for key, value in sorted(self.goal_weights.items())},
            "success": self.success,
            "reward": round(self.reward, 4),
            "goal_alignment_score": round(self.goal_alignment_score, 4),
            "goal_weighted_score": round(self.goal_weighted_score, 4),
        }
        if self.selected_outcome is not None:
            payload["selected_outcome"] = self.selected_outcome.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "DecisionEpisode":
        raw_weights = data.get("goal_weights", {})
        return cls(
            episode_id=str(data.get("episode_id", "")),
            sequence=int(data.get("sequence", 0)),
            input_brain=BrainInputV1.from_dict(data.get("input_brain", {}) if isinstance(data.get("input_brain", {}), dict) else {}),
            goal_weights={str(key): float(value) for key, value in raw_weights.items()} if isinstance(raw_weights, dict) else {},
            selected_outcome=BrainInputV1.from_dict(data.get("selected_outcome", {}) if isinstance(data.get("selected_outcome", {}), dict) else {}) if isinstance(data.get("selected_outcome", {}), dict) else None,
            success=bool(data.get("success", False)),
            reward=float(data.get("reward", 0.0)),
            goal_alignment_score=float(data.get("goal_alignment_score", 0.0)),
            goal_weighted_score=float(data.get("goal_weighted_score", 0.0)),
        )


class DecisionHistoryRegistry:
    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)
        self.episodes: List[DecisionEpisode] = []
        self._load()

    def _load(self) -> None:
        if not self.registry_path.exists():
            self.episodes = []
            return
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded: List[DecisionEpisode] = []
        if isinstance(data, dict):
            for item in data.get("episodes", []):
                if isinstance(item, dict):
                    loaded.append(DecisionEpisode.from_dict(item))
        self.episodes = loaded

    def save(self) -> None:
        payload = {"episodes": [episode.to_dict() for episode in self.episodes]}
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def record(self, episode: DecisionEpisode) -> None:
        self.episodes.append(episode)
        self.save()


@dataclass(frozen=True)
class GoalScoreComponent:
    goal_id: str
    runtime_weight: float
    effective_weight: float
    alignment_score: float
    reward_prior: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "goal_id": self.goal_id,
            "runtime_weight": round(self.runtime_weight, 4),
            "effective_weight": round(self.effective_weight, 4),
            "alignment_score": round(self.alignment_score, 4),
            "reward_prior": round(self.reward_prior, 4),
        }


@dataclass(frozen=True)
class GoalDrivenPlanningPath:
    base_path: GroundedPlanningPath
    goal_alignment_score: float
    reward_prior: float
    goal_weighted_score: float
    goal_breakdown: List[GoalScoreComponent]

    @property
    def steps(self) -> List[object]:
        return self.base_path.steps

    @property
    def final_outcome(self) -> BrainInputV1:
        return self.base_path.final_outcome

    @property
    def confidence(self) -> float:
        return self.base_path.confidence

    @property
    def grounded_score(self) -> float:
        return self.base_path.grounded_score

    @property
    def depth(self) -> int:
        return self.base_path.depth

    def to_dict(self) -> Dict[str, object]:
        payload = self.base_path.to_dict()
        payload.update(
            {
                "goal_alignment_score": round(self.goal_alignment_score, 4),
                "reward_prior": round(self.reward_prior, 4),
                "goal_weighted_score": round(self.goal_weighted_score, 4),
                "goal_breakdown": [component.to_dict() for component in self.goal_breakdown],
            }
        )
        return payload


@dataclass(frozen=True)
class GoalDrivenDecision:
    input_brain: BrainInputV1
    goal_weights: Dict[str, float]
    candidate_paths: List[GoalDrivenPlanningPath]
    selected_path: Optional[GoalDrivenPlanningPath]
    highest_goal_alignment_path: Optional[GoalDrivenPlanningPath]
    decision_score: float
    final_predicted_outcome: Optional[BrainInputV1]

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "input_brain": self.input_brain.to_dict(),
            "goal_weights": {str(key): round(float(value), 4) for key, value in sorted(self.goal_weights.items())},
            "candidate_paths": [path.to_dict() for path in self.candidate_paths],
            "decision_score": round(self.decision_score, 4),
        }
        if self.selected_path is not None:
            payload["selected_path"] = self.selected_path.to_dict()
        if self.highest_goal_alignment_path is not None:
            payload["highest_goal_alignment_path"] = self.highest_goal_alignment_path.to_dict()
        if self.final_predicted_outcome is not None:
            payload["final_predicted_outcome"] = self.final_predicted_outcome.to_dict()
        return payload


class BrainGoalRewardV5:
    def __init__(
        self,
        state_dir: str | Path = DEFAULT_OUTPUT_DIR / "brain_goal_reward_v5.state",
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.v4_engine = BrainRealityGroundingV4(state_dir=self.state_dir / "v4")
        self.goal_registry = GoalRegistry(self.state_dir / "brain_goals.json")
        self.goal_outcome_registry = GoalOutcomeRegistry(self.state_dir / "brain_goal_outcomes.json")
        self.decision_history = DecisionHistoryRegistry(self.state_dir / "brain_goal_history.json")

    def ensure_goal(
        self,
        goal_id: str,
        targets: Sequence[GoalTarget],
        base_weight: float = 1.0,
        notes: Optional[Sequence[str]] = None,
    ) -> GoalNode:
        return self.goal_registry.ensure_goal(
            goal_id=goal_id,
            targets=targets,
            base_weight=base_weight,
            notes=notes,
        )

    def plan(
        self,
        brain_input: BrainInputV1,
        goal_weights: Dict[str, float],
        max_depth: int = 5,
        beam_width: int = 4,
        current_sequence: int = 1,
    ) -> GoalDrivenDecision:
        base_decision = self.v4_engine.plan(
            brain_input=brain_input,
            max_depth=max_depth,
            beam_width=beam_width,
            current_sequence=current_sequence,
        )
        rescored = [
            self._score_goal_path(path, goal_weights=goal_weights, current_sequence=current_sequence)
            for path in base_decision.retained_paths
        ]
        rescored = sorted(
            rescored,
            key=lambda item: (
                -item.goal_weighted_score,
                -item.goal_alignment_score,
                -item.grounded_score,
                -item.confidence,
                -item.depth,
            ),
        )
        selected_path = rescored[0] if rescored else None
        highest_goal_alignment_path = max(
            rescored,
            key=lambda item: (item.goal_alignment_score, item.goal_weighted_score, item.grounded_score, item.confidence),
            default=None,
        )
        return GoalDrivenDecision(
            input_brain=base_decision.input_brain,
            goal_weights=dict(goal_weights),
            candidate_paths=rescored,
            selected_path=selected_path,
            highest_goal_alignment_path=highest_goal_alignment_path,
            decision_score=selected_path.goal_weighted_score if selected_path is not None else 0.0,
            final_predicted_outcome=selected_path.final_outcome if selected_path is not None else None,
        )

    def validate_decision(
        self,
        input_brain: BrainInputV1,
        decision: GoalDrivenDecision,
        goal_weights: Dict[str, float],
        preferred_chain: Sequence[BrainInputV1],
        preferred_relation_types: Sequence[str],
        alternative_chains: Sequence[Tuple[Sequence[BrainInputV1], Sequence[str], float, float]],
        current_sequence: int,
    ) -> Dict[str, object]:
        selected = decision.selected_path
        base_decision = self._to_v4_decision(input_brain=input_brain, decision=decision)
        validation = self.v4_engine.validate_decision(
            input_brain=input_brain,
            decision=base_decision,
            preferred_chain=preferred_chain,
            preferred_relation_types=preferred_relation_types,
            alternative_chains=alternative_chains,
            current_sequence=current_sequence,
        )
        reward = 0.0
        updated_goal_ids: List[str] = []
        updated_goal_outcome_ids: List[str] = []
        if selected is not None:
            success = bool(validation.get("success", False))
            reward = self._reward_signal(selected=selected, success=success)
            for component in selected.goal_breakdown:
                self.goal_registry.update_feedback(
                    goal_id=component.goal_id,
                    success=success,
                    reward=reward,
                    alignment=component.alignment_score,
                    current_sequence=current_sequence,
                )
                updated_goal_ids.append(component.goal_id)
                stat_id = self._goal_outcome_stat_id(component.goal_id, selected.final_outcome)
                self.goal_outcome_registry.update(
                    stat_id=stat_id,
                    goal_id=component.goal_id,
                    outcome=selected.final_outcome,
                    success=success,
                    reward=reward,
                    alignment=component.alignment_score,
                    current_sequence=current_sequence,
                )
                updated_goal_outcome_ids.append(stat_id)
            self.decision_history.record(
                DecisionEpisode(
                    episode_id=f"episode_{current_sequence:05d}",
                    sequence=current_sequence,
                    input_brain=input_brain,
                    goal_weights=dict(goal_weights),
                    selected_outcome=selected.final_outcome,
                    success=success,
                    reward=reward,
                    goal_alignment_score=selected.goal_alignment_score,
                    goal_weighted_score=selected.goal_weighted_score,
                )
            )
        return {
            **validation,
            "reward": round(reward, 4),
            "selected_goal_alignment_score": round(selected.goal_alignment_score, 4) if selected is not None else 0.0,
            "selected_goal_weighted_score": round(selected.goal_weighted_score, 4) if selected is not None else 0.0,
            "selected_reward_prior": round(selected.reward_prior, 4) if selected is not None else 0.0,
            "updated_goal_ids": updated_goal_ids,
            "updated_goal_outcome_ids": updated_goal_outcome_ids,
            "used_goal_reward_layer": bool(selected is not None and bool(selected.goal_breakdown)),
        }

    def evaluate_goal_alignment(
        self,
        states: Sequence[BrainInputV1],
        goal_weights: Dict[str, float],
        current_sequence: int,
    ) -> Tuple[float, float, List[GoalScoreComponent]]:
        normalized_states = list(states)
        components: List[GoalScoreComponent] = []
        total_weight = 0.0
        alignment_total = 0.0
        prior_total = 0.0
        for goal_id, runtime_weight in sorted(goal_weights.items()):
            goal = self.goal_registry.get(goal_id)
            if goal is None or not goal.active or runtime_weight <= 0.0:
                continue
            effective_weight = max(0.0, runtime_weight * goal.base_weight)
            if effective_weight <= 0.0:
                continue
            alignment = self._goal_alignment_for_states(normalized_states, goal)
            reward_prior = self._goal_reward_prior(goal_id, normalized_states[-1], current_sequence=current_sequence)
            components.append(
                GoalScoreComponent(
                    goal_id=goal_id,
                    runtime_weight=runtime_weight,
                    effective_weight=effective_weight,
                    alignment_score=alignment,
                    reward_prior=reward_prior,
                )
            )
            total_weight += effective_weight
            alignment_total += alignment * effective_weight
            prior_total += reward_prior * effective_weight
        if total_weight <= 0.0:
            return 0.0, 0.0, []
        return (
            round(alignment_total / total_weight, 4),
            round(prior_total / total_weight, 4),
            components,
        )

    def registry_summary(self) -> Dict[str, object]:
        active_goals = self.goal_registry.all_active_goals()
        active_stats = self.goal_outcome_registry.all_active_stats()
        average_goal_reward = round(
            sum(goal.average_reward for goal in active_goals) / max(1, len(active_goals)),
            4,
        ) if active_goals else 0.0
        average_goal_alignment = round(
            sum(goal.average_alignment for goal in active_goals) / max(1, len(active_goals)),
            4,
        ) if active_goals else 0.0
        average_goal_success = round(
            sum(goal.success_count / max(1, goal.usage_count) for goal in active_goals) / max(1, len(active_goals)),
            4,
        ) if active_goals else 0.0
        average_goal_outcome_reliability = round(
            sum(stat.reliability for stat in active_stats) / max(1, len(active_stats)),
            4,
        ) if active_stats else 0.0
        average_decision_reward = round(
            sum(episode.reward for episode in self.decision_history.episodes) / max(1, len(self.decision_history.episodes)),
            4,
        ) if self.decision_history.episodes else 0.0
        return {
            **self.v4_engine.registry_summary(),
            "goal_count": len(active_goals),
            "goal_outcome_stat_count": len(active_stats),
            "decision_history_count": len(self.decision_history.episodes),
            "average_goal_reward": average_goal_reward,
            "average_goal_alignment": average_goal_alignment,
            "average_goal_success_rate": average_goal_success,
            "average_goal_outcome_reliability": average_goal_outcome_reliability,
            "average_decision_reward": average_decision_reward,
        }

    def _score_goal_path(
        self,
        path: GroundedPlanningPath,
        goal_weights: Dict[str, float],
        current_sequence: int,
    ) -> GoalDrivenPlanningPath:
        states = [step.consequent for step in path.steps] or [path.final_outcome]
        goal_alignment, reward_prior, components = self.evaluate_goal_alignment(
            states=states,
            goal_weights=goal_weights,
            current_sequence=current_sequence,
        )
        goal_weighted_score = round(
            max(
                0.0,
                min(
                    1.0,
                    (path.grounded_score * 0.45)
                    + (goal_alignment * 0.4)
                    + (reward_prior * 0.15),
                ),
            ),
            4,
        )
        return GoalDrivenPlanningPath(
            base_path=path,
            goal_alignment_score=goal_alignment,
            reward_prior=reward_prior,
            goal_weighted_score=goal_weighted_score,
            goal_breakdown=components,
        )

    def _goal_alignment_for_states(
        self,
        states: Sequence[BrainInputV1],
        goal: GoalNode,
    ) -> float:
        match_score = self.v4_engine.v35_engine.v3_engine.v2_engine.v1_engine._match_score
        if not goal.targets:
            return 0.0
        target_total = 0.0
        weight_total = 0.0
        for target in goal.targets:
            weight = max(0.0, target.weight)
            if weight <= 0.0:
                continue
            best_match = 0.0
            for index, state in enumerate(states, start=1):
                state_weight = float(index) / float(max(1, len(states)))
                best_match = max(best_match, match_score(state, target.target_brain) * state_weight)
            if target.polarity == "avoid":
                target_score = 1.0 - min(1.0, best_match)
            else:
                target_score = min(1.0, best_match)
            target_total += target_score * weight
            weight_total += weight
        if weight_total <= 0.0:
            return 0.0
        return round(target_total / weight_total, 4)

    def _goal_reward_prior(
        self,
        goal_id: str,
        outcome: BrainInputV1,
        current_sequence: int,
    ) -> float:
        stat = self.goal_outcome_registry.get(self._goal_outcome_stat_id(goal_id, outcome))
        goal = self.goal_registry.get(goal_id)
        if stat is None and goal is None:
            return 0.0
        if stat is None:
            return round(
                max(0.0, min(1.0, (((goal.average_reward + 1.0) / 2.0) * 0.6) + (goal.temporal_score * 0.4))),
                4,
            ) if goal is not None else 0.0
        usage_signal = min(1.0, stat.usage_count / max(1, current_sequence))
        reward_signal = (stat.average_reward + 1.0) / 2.0
        base = (stat.reliability * 0.4) + (reward_signal * 0.35) + (stat.average_alignment * 0.15) + (usage_signal * 0.1)
        if goal is not None:
            base = (base * 0.8) + ((((goal.average_reward + 1.0) / 2.0) * 0.1) + (goal.temporal_score * 0.1))
        return round(max(0.0, min(1.0, base)), 4)

    def _reward_signal(
        self,
        selected: GoalDrivenPlanningPath,
        success: bool,
    ) -> float:
        success_signal = 1.0 if success else -1.0
        reward = (
            (success_signal * 0.45)
            + (selected.goal_alignment_score * 0.35)
            + (selected.grounded_score * 0.15)
            + (selected.reward_prior * 0.05)
        )
        return round(max(-1.0, min(1.0, reward)), 4)

    def _to_v4_decision(
        self,
        input_brain: BrainInputV1,
        decision: GoalDrivenDecision,
    ) -> GroundedPlanningDecision:
        selected_base = decision.selected_path.base_path if decision.selected_path is not None else None
        highest_base = (
            decision.highest_goal_alignment_path.base_path
            if decision.highest_goal_alignment_path is not None
            else None
        )
        return GroundedPlanningDecision(
            input_brain=input_brain,
            candidate_paths=[path.base_path for path in decision.candidate_paths],
            retained_paths=[path.base_path for path in decision.candidate_paths],
            selected_path=selected_base,
            highest_confidence_path=highest_base,
            highest_plausibility_path=highest_base,
            decision_score=selected_base.grounded_score if selected_base is not None else 0.0,
            final_predicted_outcome=selected_base.final_outcome if selected_base is not None else None,
            filtered_path_count=0,
        )

    def _brain_fingerprint(self, brain_input: BrainInputV1) -> str:
        v1_engine = self.v4_engine.v35_engine.v3_engine.v2_engine.v1_engine
        bundle = v1_engine.runtime.codec.encode_memory_bundle(
            v1_engine.runtime.brain_input_codec.brain_input_to_memory(brain_input)
        )
        return json.dumps(bundle, sort_keys=True)

    def _goal_outcome_stat_id(self, goal_id: str, outcome: BrainInputV1) -> str:
        return f"{goal_id}|{self._brain_fingerprint(outcome)}"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="brain_goal_reward_v5",
        description="Write a summary for the BrAIn goal and reward layer.",
    )
    parser.add_argument(
        "--state-dir",
        default=str(DEFAULT_OUTPUT_DIR / "brain_goal_reward_v5.state"),
        help="State directory for the BrAIn V5 goal and reward engine.",
    )
    args = parser.parse_args()
    engine = BrainGoalRewardV5(state_dir=args.state_dir)
    summary_path = Path(args.state_dir) / "brain_goal_reward_v5.summary.json"
    summary_path.write_text(json.dumps(engine.registry_summary(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"Goal and reward registry summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
