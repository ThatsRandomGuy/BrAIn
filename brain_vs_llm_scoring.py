from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence


BENCHMARK_CATEGORIES = ("factual_qa", "structured_planning", "memory_recall")
# Keep category boundaries stable so future grid/pixel-style internal semantic
# representations can be evaluated against the same task families.


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _keyword_overlap(keywords: Sequence[str], text: str) -> float:
    normalized = [str(item).strip().lower() for item in keywords if str(item).strip()]
    if not normalized:
        return 0.0
    lowered = str(text or "").lower()
    return sum(1 for item in normalized if item in lowered) / float(len(normalized))


def _forbidden_error_penalty(forbidden_errors: Sequence[str], text: str) -> float:
    normalized = [str(item).strip().lower() for item in forbidden_errors if str(item).strip()]
    if not normalized:
        return 0.0
    lowered = str(text or "").lower()
    matches = sum(1 for item in normalized if item in lowered)
    if matches <= 0:
        return 0.0
    return _clamp(matches / float(len(normalized)))


def _count_step_signals(text: str) -> int:
    lowered = str(text or "").lower()
    count = 0
    count += len(re.findall(r"(^|\n)\s*(?:\d+\.|-|\*)\s+", lowered))
    count += len(re.findall(r"\b(first|second|third|then|next|finally|step)\b", lowered))
    return count


def classify_benchmark_category(case: Dict[str, object]) -> str:
    explicit = str(case.get("benchmark_category", "")).strip().lower()
    if explicit in BENCHMARK_CATEGORIES:
        return explicit
    category = str(case.get("category", "")).strip().lower()
    if category == "memory_question":
        return "memory_recall"
    if category == "factual_question":
        return "factual_qa"
    return "structured_planning"


def is_meaningful_response(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    invalid_markers = (
        "configure `brain_llm_provider",
        "configure brain_llm_provider",
        "configure `brain_llm_command",
        "configure brain_llm_command",
        "configure `brain_llm_model",
        "configure brain_llm_model",
        "no live model-generated wording was available",
        "this prompt was routed to direct answering",
        "provider_unavailable",
        "model_unavailable",
        "placeholder",
    )
    if any(marker in normalized for marker in invalid_markers):
        return False
    if normalized in {"n/a", "none", "unknown", "no answer"}:
        return False
    return True


def detect_invalid_reason(
    *,
    answer_text: str,
    metadata: Dict[str, object] | None,
    error_text: str = "",
    require_live_model: bool = True,
    provider_name: str = "",
) -> Optional[str]:
    if str(error_text or "").strip():
        return f"runtime_error:{str(error_text).strip()}"
    if require_live_model and provider_name == "deterministic":
        return "provider_unavailable:deterministic_provider"
    payload = metadata or {}
    mode = str(payload.get("mode", "")).strip().lower()
    safe_fallback_modes = {
        "unknown_route_action_fallback",
        "clarification_bypassed_action_answer",
    }
    if mode in safe_fallback_modes and is_meaningful_response(answer_text):
        return None
    if "fallback" in mode:
        return f"provider_fallback:{mode}"
    if payload.get("error"):
        return f"provider_error:{payload.get('error')}"
    if not str(answer_text or "").strip():
        return "empty_answer"
    if not is_meaningful_response(answer_text):
        return "non_meaningful_placeholder"
    return None


def score_structure(*, response_text: str, selected_strategy: Dict[str, object] | None = None) -> float:
    score = 0.15 if str(response_text or "").strip() else 0.0
    step_count = _count_step_signals(response_text)
    if step_count >= 4:
        score += 0.45
    elif step_count >= 2:
        score += 0.3
    elif "\n" in str(response_text or ""):
        score += 0.1
    if isinstance(selected_strategy, dict) and selected_strategy.get("steps"):
        score += 0.3
    return round(_clamp(score), 4)


def score_consistency(*, response_text: str) -> float:
    lowered = str(response_text or "").lower()
    penalties = 0.0
    contradiction_pairs = [
        ("always", "never"),
        ("safe", "unsafe"),
        ("can", "cannot"),
    ]
    for left, right in contradiction_pairs:
        if left in lowered and right in lowered:
            penalties += 0.2
    if lowered.count("?") > 1 and any(word in lowered for word in ("definitely", "certainly")):
        penalties += 0.15
    return round(_clamp(1.0 - penalties), 4)


def score_completeness(
    *,
    response_text: str,
    expected_keywords: Sequence[str],
    clarification_active: bool,
) -> float:
    if clarification_active:
        return round(0.8 if "?" in str(response_text or "") else 0.55, 4)
    coverage = _keyword_overlap(expected_keywords, response_text)
    length_bonus = 0.15 if len(str(response_text or "").split()) >= 10 else 0.0
    return round(_clamp(0.15 + (coverage * 0.7) + length_bonus), 4)


def score_memory_usage(*, retrieved_memory_count: int, recalled_turn_count: int, benchmark_category: str) -> Optional[float]:
    if benchmark_category != "memory_recall":
        return None
    if recalled_turn_count > 0:
        return round(_clamp(0.65 + (0.15 * min(recalled_turn_count, 2)) + (0.05 * min(retrieved_memory_count, 2))), 4)
    if retrieved_memory_count > 0:
        return round(_clamp(0.35 + (0.1 * min(retrieved_memory_count, 3))), 4)
    return 0.0


def score_meaningful_response(*, response_text: str) -> float:
    if not str(response_text or "").strip():
        return 0.0
    score = 0.85 if is_meaningful_response(response_text) else 0.0
    if len(str(response_text or "").split()) >= 8:
        score += 0.1
    return round(_clamp(score), 4)


def score_correctness(
    *,
    benchmark_category: str,
    response_text: str,
    expected_keywords: Sequence[str],
    memory_fidelity_keywords: Sequence[str],
    clarification_active: bool,
) -> Optional[float]:
    if clarification_active and benchmark_category == "structured_planning":
        return round(0.8 if "?" in str(response_text or "") else 0.5, 4)
    if benchmark_category == "factual_qa":
        return round(_clamp(_keyword_overlap(expected_keywords, response_text)), 4)
    if benchmark_category == "memory_recall":
        keywords = list(memory_fidelity_keywords) or list(expected_keywords)
        return round(_clamp(_keyword_overlap(keywords, response_text)), 4)
    if benchmark_category == "structured_planning":
        return round(_clamp((_keyword_overlap(expected_keywords, response_text) * 0.7) + (0.3 if _count_step_signals(response_text) >= 2 else 0.0)), 4)
    return None


def score_constraint_satisfaction(
    *,
    benchmark_category: str,
    response_text: str,
    constraint_keywords: Sequence[str],
    clarification_active: bool,
) -> Optional[float]:
    if benchmark_category != "structured_planning":
        return None
    if clarification_active:
        return round(0.75 if "?" in str(response_text or "") else 0.45, 4)
    if not constraint_keywords:
        return round(0.55 if is_meaningful_response(response_text) else 0.0, 4)
    return round(_clamp(_keyword_overlap(constraint_keywords, response_text)), 4)


def score_memory_fidelity(
    *,
    benchmark_category: str,
    response_text: str,
    memory_fidelity_keywords: Sequence[str],
    recalled_turn_count: int,
    corrected_facts: Sequence[str] = (),
    forbidden_errors: Sequence[str] = (),
) -> Optional[float]:
    # This remains useful later if episodic recall is backed by non-token
    # semantic storage, including pixel/grid-oriented internal representations.
    if benchmark_category != "memory_recall":
        return None
    primary_keywords = list(memory_fidelity_keywords) + [str(item) for item in corrected_facts if str(item).strip()]
    coverage = _keyword_overlap(primary_keywords, response_text) if primary_keywords else 0.0
    support_bonus = 0.2 if recalled_turn_count > 0 else 0.0
    penalty = _forbidden_error_penalty(forbidden_errors, response_text)
    return round(_clamp((coverage * 0.75) + support_bonus - (penalty * 0.35)), 4)


def score_hallucination_penalty(
    *,
    response_text: str,
    benchmark_category: str,
    memory_fidelity_keywords: Sequence[str],
    recalled_turn_count: int = 0,
    forbidden_errors: Sequence[str] = (),
) -> float:
    lowered = str(response_text or "").lower()
    penalty = 0.0
    if any(word in lowered for word in ("definitely", "guaranteed", "always", "never")):
        penalty += 0.18
    if benchmark_category == "memory_recall" and any(phrase in lowered for phrase in ("you asked", "i told you", "previously")):
        if recalled_turn_count == 0:
            penalty += 0.4
        elif memory_fidelity_keywords and _keyword_overlap(memory_fidelity_keywords, response_text) < 0.34:
            penalty += 0.22
    penalty += 0.35 * _forbidden_error_penalty(forbidden_errors, response_text)
    return round(_clamp(penalty), 4)


def score_brain_output(
    *,
    benchmark_category: str,
    response_text: str,
    selected_strategy: Dict[str, object],
    clarification_state: Dict[str, object],
    retrieved_memory_count: int,
    recalled_turn_count: int,
    expected_keywords: Sequence[str],
    constraint_keywords: Sequence[str],
    memory_fidelity_keywords: Sequence[str],
    expected_recall_facts: Sequence[str] = (),
    corrected_facts: Sequence[str] = (),
    forbidden_errors: Sequence[str] = (),
) -> Dict[str, Optional[float]]:
    clarification_active = bool(clarification_state.get("active")) and not bool(clarification_state.get("resolved"))
    recall_keywords = list(expected_recall_facts) or list(memory_fidelity_keywords)
    return {
        "structure_score": score_structure(response_text=response_text, selected_strategy=selected_strategy),
        "consistency_score": score_consistency(response_text=response_text),
        "completeness_score": score_completeness(
            response_text=response_text,
            expected_keywords=expected_keywords,
            clarification_active=clarification_active,
        ),
        "memory_usage_score": score_memory_usage(
            retrieved_memory_count=retrieved_memory_count,
            recalled_turn_count=recalled_turn_count,
            benchmark_category=benchmark_category,
        ),
        "correctness_score": score_correctness(
            benchmark_category=benchmark_category,
            response_text=response_text,
            expected_keywords=expected_keywords,
            memory_fidelity_keywords=recall_keywords,
            clarification_active=clarification_active,
        ),
        "constraint_satisfaction_score": score_constraint_satisfaction(
            benchmark_category=benchmark_category,
            response_text=response_text,
            constraint_keywords=constraint_keywords,
            clarification_active=clarification_active,
        ),
        "memory_fidelity_score": score_memory_fidelity(
            benchmark_category=benchmark_category,
            response_text=response_text,
            memory_fidelity_keywords=recall_keywords,
            recalled_turn_count=recalled_turn_count,
            corrected_facts=corrected_facts,
            forbidden_errors=forbidden_errors,
        ),
        "hallucination_penalty": score_hallucination_penalty(
            response_text=response_text,
            benchmark_category=benchmark_category,
            memory_fidelity_keywords=recall_keywords,
            recalled_turn_count=recalled_turn_count,
            forbidden_errors=forbidden_errors,
        ),
        "meaningful_response_score": score_meaningful_response(response_text=response_text),
    }


def score_llm_output(
    *,
    benchmark_category: str,
    response_text: str,
    expected_keywords: Sequence[str],
    constraint_keywords: Sequence[str],
    memory_fidelity_keywords: Sequence[str],
    expected_recall_facts: Sequence[str] = (),
    corrected_facts: Sequence[str] = (),
    forbidden_errors: Sequence[str] = (),
) -> Dict[str, Optional[float]]:
    recall_keywords = list(expected_recall_facts) or list(memory_fidelity_keywords)
    return {
        "structure_score": score_structure(response_text=response_text),
        "consistency_score": score_consistency(response_text=response_text),
        "completeness_score": score_completeness(
            response_text=response_text,
            expected_keywords=expected_keywords,
            clarification_active=False,
        ),
        "memory_usage_score": None,
        "correctness_score": score_correctness(
            benchmark_category=benchmark_category,
            response_text=response_text,
            expected_keywords=expected_keywords,
            memory_fidelity_keywords=recall_keywords,
            clarification_active=False,
        ),
        "constraint_satisfaction_score": score_constraint_satisfaction(
            benchmark_category=benchmark_category,
            response_text=response_text,
            constraint_keywords=constraint_keywords,
            clarification_active=False,
        ),
        "memory_fidelity_score": score_memory_fidelity(
            benchmark_category=benchmark_category,
            response_text=response_text,
            memory_fidelity_keywords=recall_keywords,
            recalled_turn_count=0,
            corrected_facts=corrected_facts,
            forbidden_errors=forbidden_errors,
        ),
        "hallucination_penalty": score_hallucination_penalty(
            response_text=response_text,
            benchmark_category=benchmark_category,
            memory_fidelity_keywords=recall_keywords,
            recalled_turn_count=0,
            forbidden_errors=forbidden_errors,
        ),
        "meaningful_response_score": score_meaningful_response(response_text=response_text),
    }


def comparison_score(scores: Dict[str, Optional[float]]) -> Optional[float]:
    if not isinstance(scores, dict):
        return None
    positive_metrics = [
        "meaningful_response_score",
        "correctness_score",
        "constraint_satisfaction_score",
        "memory_fidelity_score",
        "structure_score",
        "completeness_score",
        "consistency_score",
    ]
    values = [float(scores[key]) for key in positive_metrics if scores.get(key) is not None]
    if not values:
        return None
    penalty = float(scores.get("hallucination_penalty") or 0.0)
    return round(_clamp((sum(values) / float(len(values))) - (penalty * 0.5)), 4)


def aggregate_scores(case_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    metrics = [
        "structure_score",
        "consistency_score",
        "completeness_score",
        "memory_usage_score",
        "correctness_score",
        "constraint_satisfaction_score",
        "memory_fidelity_score",
        "hallucination_penalty",
        "meaningful_response_score",
    ]
    summary: Dict[str, object] = {"overall": {}, "by_category": {}}

    eligible_cases = [
        item for item in case_results
        if isinstance(item, dict) and item.get("comparison_eligible", False)
    ]
    summary["overall"] = _aggregate_case_list(eligible_cases, metrics)
    by_category: Dict[str, object] = {}
    for category in BENCHMARK_CATEGORIES:
        category_cases = [item for item in eligible_cases if item.get("benchmark_category") == category]
        by_category[category] = _aggregate_case_list(category_cases, metrics)
    summary["by_category"] = by_category
    return summary


def _aggregate_case_list(case_results: Sequence[Dict[str, object]], metrics: Sequence[str]) -> Dict[str, object]:
    side_summary: Dict[str, Dict[str, Optional[float]]] = {"brain": {}, "llm": {}}
    wins = {"brain": 0, "llm": 0, "tie": 0}
    for side in ("brain", "llm"):
        for metric in metrics:
            values = [
                float(item.get("scores", {}).get(side, {}).get(metric))
                for item in case_results
                if item.get("scores", {}).get(side, {}).get(metric) is not None
            ]
            side_summary[side][metric] = round(sum(values) / float(len(values)), 4) if values else None
        comp_values = [
            float(item.get("comparison_scores", {}).get(side))
            for item in case_results
            if item.get("comparison_scores", {}).get(side) is not None
        ]
        side_summary[side]["comparison_score"] = round(sum(comp_values) / float(len(comp_values)), 4) if comp_values else None
    for item in case_results:
        brain_value = item.get("comparison_scores", {}).get("brain")
        llm_value = item.get("comparison_scores", {}).get("llm")
        if brain_value is None or llm_value is None:
            continue
        if abs(float(brain_value) - float(llm_value)) <= 0.02:
            wins["tie"] += 1
        elif float(brain_value) > float(llm_value):
            wins["brain"] += 1
        else:
            wins["llm"] += 1
    return {
        "eligible_case_count": len(case_results),
        "average_scores": side_summary,
        "wins": wins,
    }
