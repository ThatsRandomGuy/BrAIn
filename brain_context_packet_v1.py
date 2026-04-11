from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from brain_input_v1 import BrainInputV1


def _has_text(value: object) -> bool:
    return bool(str(value or "").strip())


def _compact(value: object) -> object:
    if isinstance(value, dict):
        compacted: Dict[str, object] = {}
        for key, item in value.items():
            reduced = _compact(item)
            if reduced in ("", None, [], {}):
                continue
            compacted[str(key)] = reduced
        return compacted
    if isinstance(value, list):
        compacted_list = [_compact(item) for item in value]
        return [item for item in compacted_list if item not in ("", None, [], {})]
    return value


@dataclass(frozen=True)
class BrAInContextPacketV1:
    version: str
    route_type: str
    task_type: str
    user_query: str
    current_state: Dict[str, object]
    relevant_history: List[Dict[str, object]]
    active_constraints: List[str]
    active_goals: List[Dict[str, object]]
    progress_state: Dict[str, object]
    reasoning_summary: Dict[str, object]
    answering_instructions: List[str]
    route_context: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return _compact(
            {
                "version": self.version,
                "route_type": self.route_type,
                "task_type": self.task_type,
                "user_query": self.user_query,
                "current_state": dict(self.current_state),
                "relevant_history": list(self.relevant_history),
                "active_constraints": list(self.active_constraints),
                "active_goals": list(self.active_goals),
                "progress_state": dict(self.progress_state),
                "reasoning_summary": dict(self.reasoning_summary),
                "answering_instructions": list(self.answering_instructions),
                "route_context": dict(self.route_context),
            }
        )


def build_context_packet_v1(
    *,
    user_query: str,
    route_decision: Dict[str, object],
    parsed_brain_input: BrainInputV1,
    final_result: BrainInputV1,
    merged_memory_state: Optional[Dict[str, object]] = None,
    goal_tracking: Optional[Dict[str, object]] = None,
    strategy_plan: Optional[Dict[str, object]] = None,
    selected_path: Optional[Dict[str, object]] = None,
    final_predicted_outcome: Optional[Dict[str, object]] = None,
    clarification_state: Optional[Dict[str, object]] = None,
    retrieved_memories: Optional[Sequence[Dict[str, object]]] = None,
    recalled_turns: Optional[Sequence[Dict[str, object]]] = None,
) -> Dict[str, object]:
    route = str(route_decision.get("route_type", "unknown")) if isinstance(route_decision, dict) else "unknown"
    merged = merged_memory_state if isinstance(merged_memory_state, dict) else {}
    goals = goal_tracking if isinstance(goal_tracking, dict) else {}
    strategy = strategy_plan if isinstance(strategy_plan, dict) else {}
    selected = selected_path if isinstance(selected_path, dict) else {}
    predicted = final_predicted_outcome if isinstance(final_predicted_outcome, dict) else {}
    clarification = clarification_state if isinstance(clarification_state, dict) else {}
    retrieved = [dict(item) for item in (retrieved_memories or []) if isinstance(item, dict)]
    recalled = [dict(item) for item in (recalled_turns or []) if isinstance(item, dict)]

    packet = BrAInContextPacketV1(
        version="BrAInContextPacketV1",
        route_type=route,
        task_type=_task_type_for_route(route),
        user_query=str(user_query or ""),
        current_state=_build_current_state(
            route=route,
            parsed_brain_input=parsed_brain_input,
            final_result=final_result,
            merged_memory_state=merged,
            goal_tracking=goals,
            selected_path=selected,
            final_predicted_outcome=predicted,
            clarification_state=clarification,
        ),
        relevant_history=_build_relevant_history(
            route=route,
            merged_memory_state=merged,
            recalled_turns=recalled,
        ),
        active_constraints=_build_active_constraints(
            route=route,
            parsed_brain_input=parsed_brain_input,
            final_result=final_result,
            merged_memory_state=merged,
            strategy_plan=strategy,
        ),
        active_goals=_build_active_goals(
            route=route,
            goal_tracking=goals,
        ),
        progress_state=_build_progress_state(
            route=route,
            merged_memory_state=merged,
            goal_tracking=goals,
            strategy_plan=strategy,
            clarification_state=clarification,
        ),
        reasoning_summary=_build_reasoning_summary(
            route=route,
            selected_path=selected,
            final_predicted_outcome=predicted,
            strategy_plan=strategy,
            goal_tracking=goals,
            retrieved_memories=retrieved,
        ),
        answering_instructions=_build_answering_instructions(
            route=route,
            merged_memory_state=merged,
            clarification_state=clarification,
        ),
        route_context=_build_route_context(
            route_decision=route_decision if isinstance(route_decision, dict) else {},
            retrieved_memories=retrieved,
            recalled_turns=recalled,
        ),
    )
    return packet.to_dict()


def _task_type_for_route(route: str) -> str:
    mapping = {
        "factual_question": "factual_answer",
        "memory_question": "memory_recall",
        "state_update": "state_write",
        "planning_request": "planning",
        "goal_request": "goal_reasoning",
        "transform_request": "transformation",
        "hybrid_request": "hybrid_reasoning",
    }
    return mapping.get(route, "unknown")


def _build_current_state(
    *,
    route: str,
    parsed_brain_input: BrainInputV1,
    final_result: BrainInputV1,
    merged_memory_state: Dict[str, object],
    goal_tracking: Dict[str, object],
    selected_path: Dict[str, object],
    final_predicted_outcome: Dict[str, object],
    clarification_state: Dict[str, object],
) -> Dict[str, object]:
    if route == "memory_question":
        return {
            "source_of_truth": "merged_memory_state",
            "merged_memory_state": merged_memory_state,
        }
    if route == "state_update":
        return {
            "source_of_truth": "final_result",
            "parsed_intent": parsed_brain_input.to_dict(),
            "updated_state": final_result.to_dict(),
        }
    if route in {"planning_request", "goal_request", "transform_request", "hybrid_request"}:
        return {
            "source_of_truth": "reasoning_result",
            "final_result": final_result.to_dict(),
            "selected_path_outcome": final_predicted_outcome or final_result.to_dict(),
            "clarification_state": clarification_state,
        }
    return {
        "source_of_truth": "final_result",
        "final_result": final_result.to_dict(),
    }


def _build_relevant_history(
    *,
    route: str,
    merged_memory_state: Dict[str, object],
    recalled_turns: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    history: List[Dict[str, object]] = []
    if route == "memory_question":
        for item in merged_memory_state.get("items", []) if isinstance(merged_memory_state.get("items", []), list) else []:
            if not isinstance(item, dict):
                continue
            if item.get("original_value") not in (None, "", []) and item.get("original_value") != item.get("current_value"):
                history.append(
                    {
                        "type": "state_correction",
                        "state_key": item.get("state_key", ""),
                        "original_value": item.get("original_value"),
                        "current_value": item.get("current_value"),
                        "history_count": item.get("history_count", 0),
                    }
                )
        if not history:
            for recall in recalled_turns[:2]:
                record = recall.get("record", {}) if isinstance(recall, dict) else {}
                if not isinstance(record, dict):
                    continue
                history.append(
                    {
                        "type": "prior_turn",
                        "prompt": record.get("raw_user_prompt", ""),
                        "route_type": record.get("route_type", ""),
                        "final_answer": record.get("final_answer", ""),
                    }
                )
    return history


def _build_active_constraints(
    *,
    route: str,
    parsed_brain_input: BrainInputV1,
    final_result: BrainInputV1,
    merged_memory_state: Dict[str, object],
    strategy_plan: Dict[str, object],
) -> List[str]:
    constraints: List[str] = []
    for source in (parsed_brain_input, final_result):
        constraints.extend(str(item) for item in source.slots.get("constraints", []) if _has_text(item))
        constraints.extend(str(item) for item in source.slots.get("negations", []) if _has_text(item))
    if route == "memory_question":
        for item in merged_memory_state.get("items", []) if isinstance(merged_memory_state.get("items", []), list) else []:
            if isinstance(item, dict) and str(item.get("state_type", "")) == "dietary_constraint":
                constraints.append(str(item.get("current_value", "")))
    selected_strategy = strategy_plan.get("selected_strategy", {}) if isinstance(strategy_plan, dict) else {}
    if isinstance(selected_strategy, dict):
        constraints.extend(str(item) for item in selected_strategy.get("tradeoffs", []) if _has_text(item))
    return list(dict.fromkeys(item for item in constraints if _has_text(item)))


def _build_active_goals(
    *,
    route: str,
    goal_tracking: Dict[str, object],
) -> List[Dict[str, object]]:
    if route not in {"planning_request", "goal_request", "hybrid_request", "transform_request"}:
        return []
    status_items = goal_tracking.get("goal_status", []) if isinstance(goal_tracking.get("goal_status", []), list) else []
    active_goal_ids = goal_tracking.get("active_goal_ids", []) if isinstance(goal_tracking.get("active_goal_ids", []), list) else []
    goals: List[Dict[str, object]] = []
    for item in status_items[:3]:
        if isinstance(item, dict):
            goals.append(
                {
                    "goal_id": item.get("goal_id", ""),
                    "goal_text": item.get("goal_text", ""),
                    "status": item.get("status", ""),
                    "progress_score": item.get("progress_score", 0.0),
                }
            )
    if not goals:
        goals = [{"goal_id": str(goal_id), "status": "active"} for goal_id in active_goal_ids[:3] if _has_text(goal_id)]
    return goals


def _build_progress_state(
    *,
    route: str,
    merged_memory_state: Dict[str, object],
    goal_tracking: Dict[str, object],
    strategy_plan: Dict[str, object],
    clarification_state: Dict[str, object],
) -> Dict[str, object]:
    progress: Dict[str, object] = {}
    if route == "memory_question":
        progress = {
            "memory_resolution": {
                "correction_chain_detected": merged_memory_state.get("correction_chain_detected", False),
                "latest_state_resolution_applied": merged_memory_state.get("latest_state_resolution_applied", False),
                "item_count": merged_memory_state.get("item_count", 0),
            }
        }
    elif route in {"planning_request", "goal_request", "transform_request", "hybrid_request"}:
        progress = {
            "goal_tracking": goal_tracking.get("progress_tracking", {}),
            "next_step_suggestion": goal_tracking.get("next_step_suggestion", {}),
            "strategy_feedback": strategy_plan.get("strategy_feedback", {}),
        }
    if clarification_state.get("active"):
        progress["clarification_required"] = True
        progress["clarification_question"] = clarification_state.get("clarification_question", "")
    return progress


def _build_reasoning_summary(
    *,
    route: str,
    selected_path: Dict[str, object],
    final_predicted_outcome: Dict[str, object],
    strategy_plan: Dict[str, object],
    goal_tracking: Dict[str, object],
    retrieved_memories: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    if route == "factual_question":
        return {"mode": "direct_answer", "selected_structure": final_predicted_outcome}
    if route == "memory_question":
        return {
            "mode": "memory_recall",
            "retrieved_memory_count": len(retrieved_memories),
            "selected_structure": final_predicted_outcome,
        }
    if route == "state_update":
        return {"mode": "state_write", "selected_structure": final_predicted_outcome}
    return {
        "mode": "reasoned_decision",
        "selected_path": {
            "depth": selected_path.get("depth"),
            "goal_alignment_score": selected_path.get("goal_alignment_score"),
            "goal_weighted_score": selected_path.get("goal_weighted_score"),
            "grounded_score": selected_path.get("grounded_score"),
            "plausibility_score": selected_path.get("plausibility_score"),
        },
        "selected_strategy": _compact_strategy(strategy_plan.get("selected_strategy", {})),
        "goal_progress": goal_tracking.get("progress_tracking", {}),
        "final_predicted_outcome": final_predicted_outcome,
    }


def _build_answering_instructions(
    *,
    route: str,
    merged_memory_state: Dict[str, object],
    clarification_state: Dict[str, object],
) -> List[str]:
    instructions: List[str] = []
    if route == "memory_question":
        instructions.append("Use current_state as source of truth.")
        if merged_memory_state.get("wants_original_state"):
            instructions.append("Answer both current and original state directly when available.")
        else:
            instructions.append("Prefer the latest resolved state over earlier mentions.")
        instructions.append("Keep the answer concise and direct.")
    elif route in {"planning_request", "goal_request", "transform_request", "hybrid_request"}:
        instructions.append("Use reasoning_summary and active_constraints as the decision frame.")
        instructions.append("Prefer the selected reasoning result over raw retrieved memory.")
    elif route == "factual_question":
        instructions.append("Answer the user query directly with minimal extra framing.")
    elif route == "state_update":
        instructions.append("Acknowledge the state update without unnecessary explanation.")
    if clarification_state.get("active"):
        instructions = ["Ask only for the missing detail shown in progress_state."]
    return instructions


def _build_route_context(
    *,
    route_decision: Dict[str, object],
    retrieved_memories: Sequence[Dict[str, object]],
    recalled_turns: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "confidence": route_decision.get("confidence", 0.0),
        "reasons": route_decision.get("reasons", []),
        "retrieved_memory_count": len(retrieved_memories),
        "recalled_turn_count": len(recalled_turns),
    }


def _compact_strategy(selected_strategy: object) -> Dict[str, object]:
    if not isinstance(selected_strategy, dict):
        return {}
    steps = selected_strategy.get("steps", []) if isinstance(selected_strategy.get("steps", []), list) else []
    compact_steps: List[Dict[str, object]] = []
    for step in steps[:3]:
        if not isinstance(step, dict):
            continue
        compact_steps.append(
            {
                "step_id": step.get("step_id", ""),
                "title": step.get("title", ""),
                "action": step.get("action", ""),
                "detail": step.get("detail", ""),
            }
        )
    return _compact(
        {
            "strategy_id": selected_strategy.get("strategy_id", ""),
            "name": selected_strategy.get("name", ""),
            "focus": selected_strategy.get("focus", ""),
            "target": selected_strategy.get("target", ""),
            "tradeoffs": selected_strategy.get("tradeoffs", []),
            "scores": selected_strategy.get("scores", {}),
            "steps": compact_steps,
        }
    )
