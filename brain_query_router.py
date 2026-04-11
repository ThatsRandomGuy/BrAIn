from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from brain_input_v1 import BrainInputV1


ROUTE_TYPES = (
    "factual_question",
    "memory_question",
    "state_update",
    "planning_request",
    "goal_request",
    "transform_request",
    "hybrid_request",
    "unknown",
)


@dataclass(frozen=True)
class QueryRouteDecision:
    route_type: str
    confidence: float
    reasons: List[str]
    run_brain_reasoning: bool
    run_direct_llm: bool
    run_hybrid_mode: bool
    score_breakdown: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "route_type": self.route_type,
            "confidence": round(self.confidence, 4),
            "reasons": list(self.reasons),
            "run_brain_reasoning": self.run_brain_reasoning,
            "run_direct_llm": self.run_direct_llm,
            "run_hybrid_mode": self.run_hybrid_mode,
            "score_breakdown": {key: round(value, 4) for key, value in sorted(self.score_breakdown.items())},
        }


def route_user_prompt(
    prompt_text: str,
    parsed_brain_input: Optional[BrainInputV1] = None,
) -> QueryRouteDecision:
    text = str(prompt_text or "").strip()
    normalized = _normalize(text)
    scores = {route_type: 0.0 for route_type in ROUTE_TYPES}
    reasons: List[str] = []

    if not normalized:
        reasons.append("Prompt was blank after normalization.")
        return _finalize(scores=scores, reasons=reasons, route_type="unknown")

    factual_cues = _matching_patterns(
        normalized,
        (
            r"\?$",
            r"^(what|who|when|where|why|which|is|are|can|could|does|do|did|will|would)\b",
        ),
    )
    memory_cues = _matching_patterns(
        normalized,
        (
            r"\bremember\b",
            r"\brecall\b",
            r"\bwhat did i ask\b",
            r"\bwhat did i say\b",
            r"\bwhat did you say\b",
            r"\bwhat did you tell me\b",
            r"\bwhat was my last\b",
            r"\bmy last planning request\b",
            r"\bearlier\b",
            r"\bpreviously\b",
            r"\blast time\b",
            r"\blatest\b",
            r"\bfrom our conversation\b",
            r"\bfrom memory\b",
            r"\bwhat do you know about me\b",
        ),
    )
    planning_cues = _matching_patterns(
        normalized,
        (
            r"\bhelp me\b",
            r"\bplan\b",
            r"\bsafest way\b",
            r"\bbest way\b",
            r"\bhow should i\b",
            r"\bstep by step\b",
            r"\bfinish my task\b",
            r"\bcomplete my task\b",
            r"\bavoid risk\b",
            r"\bsafely\b",
            r"\bwithout risk\b",
        ),
    )
    state_update_cues = _matching_patterns(
        normalized,
        (
            r"\bmy favorite\b",
            r"\bmy meeting\b",
            r"\bstore the\b",
            r"\bput the\b",
            r"\bi already\b",
            r"\bi also\b",
            r"^update:",
            r"\bactually\b",
            r"\bsorry\b",
            r"\bfinal update\b",
            r"\bfor my shipping checklist\b",
            r"\bfor the .* project\b",
            r"\bfor the .* issue\b",
            r"\bat the start of the sprint\b",
            r"\bthis week i am free\b",
            r"\bchange that\b",
            r"\bgoal is to\b",
            r"\bblocked\b",
            r"\bdone\b",
            r"\bin progress\b",
            r"\bnot started\b",
        ),
    )
    goal_cues = _matching_patterns(
        normalized,
        (
            r"\bmy goals\b",
            r"\bgoal[s]?\b",
            r"\bbest fits my goals\b",
            r"\baligns? with my goals\b",
            r"\bwhich option\b",
            r"\bwhich choice\b",
            r"\bwhat fits\b",
            r"\btrade-?off\b",
        ),
    )
    transform_cues = _matching_patterns(
        normalized,
        (
            r"\brewrite\b",
            r"\brephrase\b",
            r"\btransform\b",
            r"\bedit this\b",
            r"\bimprove this\b",
            r"\bsummarize\b",
            r"\bcondense\b",
            r"\bturn this into\b",
            r"\bmake this\b",
            r"\breduce risk\b",
        ),
    )
    hybrid_cues = _matching_patterns(
        normalized,
        (
            r"\buse what you know\b",
            r"\bbased on what you know\b",
            r"\busing my goals\b",
            r"\bwhat you know about my goals\b",
            r"\bcombine\b",
            r"\band tell me\b",
        ),
    )

    if factual_cues:
        scores["factual_question"] += 1.8
        reasons.append("Detected question-form phrasing.")
    if memory_cues:
        scores["memory_question"] += 4.2
        reasons.append(f"Matched memory cues: {', '.join(memory_cues[:3])}.")
    if planning_cues:
        scores["planning_request"] += 3.5
        reasons.append(f"Matched planning cues: {', '.join(planning_cues[:3])}.")
        if normalized.startswith("plan ") or re.search(r"\bmake\b.*\bplan\b", normalized):
            scores["planning_request"] += 1.4
            scores["state_update"] -= 0.8
            reasons.append("Prompt explicitly asks to create a plan, so planning outranks state-update cues.")
    if state_update_cues and not factual_cues:
        scores["state_update"] += 3.2
        reasons.append(f"Matched declarative state cues: {', '.join(state_update_cues[:3])}.")
    if goal_cues:
        scores["goal_request"] += 3.0
        reasons.append(f"Matched goal cues: {', '.join(goal_cues[:3])}.")
    if transform_cues:
        scores["transform_request"] += 3.8
        reasons.append(f"Matched transform cues: {', '.join(transform_cues[:3])}.")
    if hybrid_cues:
        scores["hybrid_request"] += 3.6
        reasons.append(f"Matched hybrid cues: {', '.join(hybrid_cues[:3])}.")

    token_count = len([token for token in normalized.split(" ") if token])
    if token_count <= 3 and not factual_cues:
        scores["unknown"] += 1.6
        reasons.append("Prompt is very short and does not strongly signal a route.")

    if factual_cues and not any((memory_cues, planning_cues, goal_cues, transform_cues, hybrid_cues)):
        scores["factual_question"] += 2.4
        reasons.append("No memory/planning/goal/transform cues were detected, so direct factual answering is preferred.")

    if memory_cues and factual_cues:
        scores["memory_question"] += 1.0
        reasons.append("Question asks about prior interaction state, which should prefer memory retrieval.")

    if state_update_cues and any(token in normalized for token in (" is ", " are ", " now ", " already ", " checklist", " project", " meeting", " store ", " put ")):
        scores["state_update"] += 1.1
        reasons.append("Prompt looks like a declarative state write rather than a question.")

    if goal_cues and planning_cues:
        scores["goal_request"] += 0.9
        reasons.append("Prompt asks for an option or recommendation in goal terms.")

    if transform_cues and re.search(r"\b(plan|draft|text|message|rewrite this|this plan)\b", normalized):
        scores["transform_request"] += 1.1
        reasons.append("Prompt refers to rewriting or reshaping existing content.")

    if _is_hybrid_request(normalized, factual_cues, memory_cues, goal_cues, planning_cues, hybrid_cues):
        scores["hybrid_request"] += 2.5
        reasons.append("Prompt combines stored-user-context cues with a direct answer request.")

    if parsed_brain_input is not None:
        action = str(parsed_brain_input.action or "").strip().lower()
        object_value = str(parsed_brain_input.slots.get("object", "")).strip().lower()
        topic_value = str(parsed_brain_input.slots.get("topic", "")).strip().lower()
        populated_slot_count = len(
            [
                value
                for value in (
                    parsed_brain_input.actor,
                    parsed_brain_input.intent,
                    action,
                    object_value,
                    topic_value,
                )
                if str(value).strip()
            ]
        )
        if action in {"charge", "repair", "study", "clean", "backup", "practice", "cook", "eat", "drink", "water"}:
            scores["planning_request"] += 0.8
            reasons.append(f"Parsed action '{action}' is a task-like action that BrAIn planning can reason over.")
        if (
            not factual_cues
            and populated_slot_count >= 2
            and any(token in normalized for token in ("my ", "i ", "update", "actually", "sorry", "final update"))
            and any(token in normalized for token in (" is ", " are ", " in the ", " at ", " done", "blocked", "in progress", "not started"))
        ):
            scores["state_update"] += 1.0
            reasons.append("Parsed structure and phrasing indicate a stored user-state update.")
        if populated_slot_count <= 2 and factual_cues and not any((memory_cues, goal_cues, planning_cues, transform_cues, hybrid_cues)):
            scores["factual_question"] += 0.5
            reasons.append("Parsed structure is sparse, which is typical for external factual questions.")

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_route, best_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if best_score < 1.5:
        scores["unknown"] += 2.0
        reasons.append("No route accumulated enough evidence, so the prompt falls back to unknown.")
        best_route = "unknown"
        best_score = scores["unknown"]
        runner_up_score = max(value for key, value in scores.items() if key != "unknown")

    return _finalize(
        scores=scores,
        reasons=reasons,
        route_type=best_route,
        best_score=best_score,
        runner_up_score=runner_up_score,
    )


def _finalize(
    *,
    scores: Dict[str, float],
    reasons: Sequence[str],
    route_type: str,
    best_score: Optional[float] = None,
    runner_up_score: Optional[float] = None,
) -> QueryRouteDecision:
    best = float(scores.get(route_type, 0.0) if best_score is None else best_score)
    alternate = 0.0 if runner_up_score is None else float(runner_up_score)
    margin = max(0.0, best - alternate)
    confidence = min(0.99, 0.5 + (min(best, 6.0) / 10.0) + (min(margin, 4.0) / 10.0))
    return QueryRouteDecision(
        route_type=route_type,
        confidence=confidence,
        reasons=list(dict.fromkeys(reason for reason in reasons if str(reason).strip())),
        run_brain_reasoning=route_type in {"memory_question", "planning_request", "goal_request", "transform_request", "hybrid_request"},
        run_direct_llm=route_type in {"factual_question", "hybrid_request", "state_update"},
        run_hybrid_mode=route_type == "hybrid_request",
        score_breakdown=scores,
    )


def _matching_patterns(text: str, patterns: Sequence[str]) -> List[str]:
    matches: List[str] = []
    for pattern in patterns:
        if re.search(pattern, text):
            matches.append(pattern.replace(r"\b", "").replace(r"\?", "?"))
    return matches


def _is_hybrid_request(
    text: str,
    factual_cues: Sequence[str],
    memory_cues: Sequence[str],
    goal_cues: Sequence[str],
    planning_cues: Sequence[str],
    hybrid_cues: Sequence[str],
) -> bool:
    if hybrid_cues and (goal_cues or memory_cues):
        return True
    if factual_cues and goal_cues and re.search(r"\b(safest|best|tell me|which)\b", text):
        return True
    if factual_cues and memory_cues and re.search(r"\bwhat should i\b|\btell me\b", text):
        return True
    if planning_cues and goal_cues and re.search(r"\buse what you know\b|\bbased on\b", text):
        return True
    return False


def _normalize(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered
