from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from brain_codec import BrainCodec, BrainMemory
from brain_input_v1 import BrainInputV1


def _has_text(value: object) -> bool:
    return bool(str(value or "").strip())


def _memory_from_brain_input(brain_input: BrainInputV1) -> BrainMemory:
    return BrainMemory(
        actor=brain_input.actor,
        intent=brain_input.intent,
        action=brain_input.action,
        object=str(brain_input.slots.get("object", "")),
        topic=str(brain_input.slots.get("topic", "")),
        modifiers=[str(item) for item in brain_input.slots.get("modifiers", [])],
        constraints=[str(item) for item in brain_input.slots.get("constraints", [])],
        negations=[str(item) for item in brain_input.slots.get("negations", [])],
        conditions=[str(item) for item in brain_input.slots.get("conditions", [])],
        relations=[relation.to_dict() for relation in brain_input.relations],
        time=brain_input.time or None,
        priority=brain_input.priority,
    )


@dataclass(frozen=True)
class GoalProgressEvent:
    sequence: int
    route_type: str
    status: str
    summary: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "sequence": self.sequence,
            "route_type": self.route_type,
            "status": self.status,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "GoalProgressEvent":
        return cls(
            sequence=int(data.get("sequence", 0)),
            route_type=str(data.get("route_type", "unknown")),
            status=str(data.get("status", "unknown")),
            summary=str(data.get("summary", "")),
        )


@dataclass(frozen=True)
class GoalRecordV6:
    goal_id: str
    goal_text: str
    route_type: str
    status: str
    progress_score: float
    inferred: bool
    success_count: int
    failure_count: int
    helpful_suggestion_count: int
    redundant_suggestion_count: int
    last_sequence: int
    related_task: str = ""
    last_suggested_action: str = ""
    events: List[GoalProgressEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "goal_id": self.goal_id,
            "goal_text": self.goal_text,
            "route_type": self.route_type,
            "status": self.status,
            "progress_score": round(self.progress_score, 4),
            "inferred": self.inferred,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "helpful_suggestion_count": self.helpful_suggestion_count,
            "redundant_suggestion_count": self.redundant_suggestion_count,
            "last_sequence": self.last_sequence,
            "related_task": self.related_task,
            "last_suggested_action": self.last_suggested_action,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "GoalRecordV6":
        return cls(
            goal_id=str(data.get("goal_id", "")),
            goal_text=str(data.get("goal_text", "")),
            route_type=str(data.get("route_type", "unknown")),
            status=str(data.get("status", "active")),
            progress_score=float(data.get("progress_score", 0.0)),
            inferred=bool(data.get("inferred", True)),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            helpful_suggestion_count=int(data.get("helpful_suggestion_count", 0)),
            redundant_suggestion_count=int(data.get("redundant_suggestion_count", 0)),
            last_sequence=int(data.get("last_sequence", 0)),
            related_task=str(data.get("related_task", "")),
            last_suggested_action=str(data.get("last_suggested_action", "")),
            events=[GoalProgressEvent.from_dict(item) for item in data.get("events", []) if isinstance(item, dict)] if isinstance(data.get("events", []), list) else [],
        )


@dataclass(frozen=True)
class GoalTrackingResult:
    active_goal_ids: List[str]
    goal_status: List[Dict[str, object]]
    progress_tracking: Dict[str, object]
    next_step_suggestion: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "active_goal_ids": list(self.active_goal_ids),
            "goal_status": list(self.goal_status),
            "progress_tracking": dict(self.progress_tracking),
            "next_step_suggestion": dict(self.next_step_suggestion),
        }


class BrainGoalTrackingV6:
    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.codec = BrainCodec()
        self.goals: Dict[str, GoalRecordV6] = {}
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            self.goals = {}
            return
        data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        loaded = data.get("goals", []) if isinstance(data, dict) else []
        self.goals = {
            record.goal_id: record
            for record in (
                GoalRecordV6.from_dict(item)
                for item in loaded
                if isinstance(item, dict)
            )
            if _has_text(record.goal_id)
        }

    def save(self) -> None:
        payload = {"goals": [record.to_dict() for record in sorted(self.goals.values(), key=lambda item: item.goal_id)]}
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def process_turn(
        self,
        *,
        sequence: int,
        user_input_text: str,
        effective_prompt_text: str,
        route_type: str,
        parsed_brain_input: BrainInputV1,
        final_result: BrainInputV1,
        candidate_path_count: int,
        clarification_state: Dict[str, object],
        recalled_turn_count: int,
        goal_weights: Dict[str, float],
        selected_path: Dict[str, object],
    ) -> GoalTrackingResult:
        goal_ids = self._infer_goal_ids(route_type, parsed_brain_input, effective_prompt_text)
        if not goal_ids and route_type in {"planning_request", "goal_request", "hybrid_request", "transform_request"}:
            goal_ids = [self._fallback_goal_id(route_type, effective_prompt_text, parsed_brain_input)]

        progress_status = self._progress_status(candidate_path_count, clarification_state, recalled_turn_count, route_type)
        progress_summary = self._progress_summary(progress_status, effective_prompt_text, final_result, candidate_path_count)

        updated_records: List[GoalRecordV6] = []
        for goal_id in goal_ids:
            previous = self.goals.get(goal_id)
            goal_text = self._goal_text(route_type, effective_prompt_text, parsed_brain_input)
            related_task = self._related_task(parsed_brain_input, effective_prompt_text)
            suggestion = self._suggest_next_step(
                route_type=route_type,
                parsed_brain_input=parsed_brain_input,
                final_result=final_result,
                candidate_path_count=candidate_path_count,
                clarification_state=clarification_state,
                selected_path=selected_path,
                goal_weights=goal_weights,
                previous_goal=previous,
            )
            updated = GoalRecordV6(
                goal_id=goal_id,
                goal_text=goal_text,
                route_type=route_type,
                status=self._goal_status(progress_status),
                progress_score=self._progress_score(progress_status, previous.progress_score if previous else 0.0),
                inferred=True,
                success_count=(previous.success_count if previous else 0) + (1 if progress_status in {"decision_ready", "memory_recalled"} else 0),
                failure_count=(previous.failure_count if previous else 0) + (1 if progress_status == "stalled" else 0),
                helpful_suggestion_count=(previous.helpful_suggestion_count if previous else 0) + (1 if progress_status == "decision_ready" and previous and _has_text(previous.last_suggested_action) else 0),
                redundant_suggestion_count=(previous.redundant_suggestion_count if previous else 0) + (1 if progress_status == "clarification_pending" and previous and previous.last_suggested_action == suggestion.get("text", "") else 0),
                last_sequence=sequence,
                related_task=related_task,
                last_suggested_action=str(suggestion.get("text", "")),
                events=((previous.events if previous else []) + [GoalProgressEvent(sequence=sequence, route_type=route_type, status=progress_status, summary=progress_summary)])[-8:],
            )
            self.goals[goal_id] = updated
            updated_records.append(updated)

        self.save()
        next_step = self._suggest_next_step(
            route_type=route_type,
            parsed_brain_input=parsed_brain_input,
            final_result=final_result,
            candidate_path_count=candidate_path_count,
            clarification_state=clarification_state,
            selected_path=selected_path,
            goal_weights=goal_weights,
            previous_goal=updated_records[0] if updated_records else None,
        )
        return GoalTrackingResult(
            active_goal_ids=goal_ids,
            goal_status=[record.to_dict() for record in updated_records],
            progress_tracking={
                "status": progress_status,
                "summary": progress_summary,
                "candidate_path_count": candidate_path_count,
                "recalled_turn_count": recalled_turn_count,
                "relevance_to_goals": self._relevance_to_goals(route_type, goal_ids, candidate_path_count, recalled_turn_count),
            },
            next_step_suggestion=next_step,
        )

    def tracked_goals_summary(self) -> List[Dict[str, object]]:
        return [
            record.to_dict()
            for record in sorted(self.goals.values(), key=lambda item: (-item.last_sequence, item.goal_id))
        ]

    def _infer_goal_ids(
        self,
        route_type: str,
        parsed_brain_input: BrainInputV1,
        effective_prompt_text: str,
    ) -> List[str]:
        if route_type not in {"planning_request", "goal_request", "hybrid_request", "transform_request"}:
            return []
        action = str(parsed_brain_input.action or "").strip().lower()
        object_value = str(parsed_brain_input.slots.get("object", "")).strip().lower()
        topic_value = str(parsed_brain_input.slots.get("topic", "")).strip().lower()
        fragments = [route_type]
        if action:
            fragments.append(action)
        if object_value:
            fragments.append(object_value)
        elif topic_value:
            fragments.append(topic_value)
        if len(fragments) == 1 and _has_text(effective_prompt_text):
            fragments.append(self._slug_fragment(effective_prompt_text))
        return ["goal:" + ":".join(self._slug_fragment(item) for item in fragments if _has_text(item))]

    def _fallback_goal_id(
        self,
        route_type: str,
        effective_prompt_text: str,
        parsed_brain_input: BrainInputV1,
    ) -> str:
        return "goal:" + ":".join(
            [
                self._slug_fragment(route_type),
                self._slug_fragment(str(parsed_brain_input.action or "") or effective_prompt_text),
            ]
        )

    def _goal_text(self, route_type: str, effective_prompt_text: str, parsed_brain_input: BrainInputV1) -> str:
        memory = _memory_from_brain_input(parsed_brain_input)
        natural = self.codec.reconstruct_text(memory)
        if _has_text(natural) and natural != "No recoverable semantic content.":
            return natural
        return f"{route_type}: {effective_prompt_text}".strip()

    def _related_task(self, parsed_brain_input: BrainInputV1, effective_prompt_text: str) -> str:
        action = str(parsed_brain_input.action or "").strip()
        object_value = str(parsed_brain_input.slots.get("object", "")).strip()
        if action and object_value:
            return f"{action} {object_value}".strip()
        if action:
            return action
        return effective_prompt_text[:120]

    @staticmethod
    def _progress_status(
        candidate_path_count: int,
        clarification_state: Dict[str, object],
        recalled_turn_count: int,
        route_type: str,
    ) -> str:
        if clarification_state.get("active"):
            return "clarification_pending"
        if route_type == "memory_question" and recalled_turn_count > 0:
            return "memory_recalled"
        if candidate_path_count > 0:
            return "decision_ready"
        return "stalled"

    def _progress_summary(
        self,
        progress_status: str,
        effective_prompt_text: str,
        final_result: BrainInputV1,
        candidate_path_count: int,
    ) -> str:
        if progress_status == "clarification_pending":
            return "BrAIn is waiting for the missing detail before continuing."
        if progress_status == "memory_recalled":
            return "BrAIn recovered prior goal-relevant context from episodic memory."
        if progress_status == "decision_ready":
            description = self.codec.reconstruct_text(_memory_from_brain_input(final_result))
            return f"BrAIn produced a decision-ready step after evaluating {candidate_path_count} path(s): {description}"
        return f"BrAIn did not find enough progress structure for: {effective_prompt_text}"

    @staticmethod
    def _goal_status(progress_status: str) -> str:
        if progress_status == "decision_ready":
            return "in_progress"
        if progress_status == "memory_recalled":
            return "contextualized"
        if progress_status == "clarification_pending":
            return "waiting_for_user"
        return "stalled"

    @staticmethod
    def _progress_score(progress_status: str, previous_score: float) -> float:
        if progress_status == "decision_ready":
            return min(1.0, max(previous_score, 0.45) + 0.25)
        if progress_status == "memory_recalled":
            return min(1.0, max(previous_score, 0.2) + 0.1)
        if progress_status == "clarification_pending":
            return max(previous_score, 0.15)
        return max(0.0, previous_score - 0.05)

    def _suggest_next_step(
        self,
        *,
        route_type: str,
        parsed_brain_input: BrainInputV1,
        final_result: BrainInputV1,
        candidate_path_count: int,
        clarification_state: Dict[str, object],
        selected_path: Dict[str, object],
        goal_weights: Dict[str, float],
        previous_goal: Optional[GoalRecordV6],
    ) -> Dict[str, object]:
        if clarification_state.get("active"):
            return {
                "type": "ask_user",
                "text": str(clarification_state.get("clarification_question", "")),
                "reason": "Missing structure blocks goal progression.",
            }
        if candidate_path_count <= 0:
            return {
                "type": "suggest_improvement",
                "text": "Provide a more concrete next action or object so BrAIn can continue goal tracking.",
                "reason": "No decision-ready path was available.",
            }
        grounded_score = float(selected_path.get("grounded_score", 0.0) or 0.0)
        plausibility_score = float(selected_path.get("plausibility_score", 0.0) or 0.0)
        if grounded_score < 0.45 or plausibility_score < 0.45:
            return {
                "type": "warn_risk",
                "text": "Proceed carefully and verify the next action, because the selected path has weak grounding.",
                "reason": "Low grounded/plausibility score suggests higher risk.",
            }
        final_text = self.codec.reconstruct_text(_memory_from_brain_input(final_result))
        if route_type == "memory_question":
            return {
                "type": "recommend_continuation",
                "text": "Use the recalled decision as the next reference point for the current goal.",
                "reason": "Recovered prior goal-relevant context can guide the next step.",
            }
        suggestion_text = f"Suggested next step: {final_text}".strip()
        if previous_goal is not None and previous_goal.last_suggested_action == suggestion_text:
            suggestion_text = f"Continue with: {final_text}"
        if goal_weights:
            strongest_goal = max(goal_weights.items(), key=lambda item: item[1])[0]
            suggestion_text = f"{suggestion_text} Focus on {strongest_goal}."
        return {
            "type": "recommend_continuation",
            "text": suggestion_text,
            "reason": "Derived from the selected structured outcome and current goal weighting.",
        }

    @staticmethod
    def _relevance_to_goals(route_type: str, goal_ids: Sequence[str], candidate_path_count: int, recalled_turn_count: int) -> float:
        score = 0.1
        if route_type in {"planning_request", "goal_request", "hybrid_request", "transform_request"}:
            score += 0.4
        if goal_ids:
            score += 0.2
        if candidate_path_count > 0:
            score += 0.2
        if recalled_turn_count > 0:
            score += 0.1
        return round(min(score, 1.0), 4)

    @staticmethod
    def _slug_fragment(value: str) -> str:
        compact = "".join(char.lower() if char.isalnum() else "_" for char in str(value or ""))
        while "__" in compact:
            compact = compact.replace("__", "_")
        return compact.strip("_") or "unknown"
