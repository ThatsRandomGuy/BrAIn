from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence


def heading(title: str) -> str:
    return f"\n{'=' * 72}\n{title}\n{'=' * 72}"


def subheading(title: str) -> str:
    return f"\n{'-' * 72}\n{title}\n{'-' * 72}"


def pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def short_json(value: Any, max_length: int = 220) -> str:
    text = json.dumps(value, sort_keys=True)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def summarize_memory_entries(entries: Sequence[Dict[str, Any]], *, full: bool = False) -> str:
    if not entries:
        return "No stored memory entries."
    if full:
        sanitized: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_copy = dict(entry)
            if not str(entry_copy.get("source_text", "")).strip():
                entry_copy.pop("source_text", None)
            sanitized.append(entry_copy)
        return pretty_json(sanitized)
    lines: List[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        parsed = entry.get("parsed_memory", {})
        summary = {
            "entry_id": entry.get("entry_id"),
            "entry_type": entry.get("entry_type"),
            "timestamp": entry.get("timestamp"),
            "action": parsed.get("action") if isinstance(parsed, dict) else None,
            "object": parsed.get("object") if isinstance(parsed, dict) else None,
            "topic": parsed.get("topic") if isinstance(parsed, dict) else None,
        }
        if str(entry.get("source_text", "")).strip():
            summary["source_text"] = entry.get("source_text", "")
        lines.append(short_json(summary))
    return "\n".join(lines)


def summarize_paths(paths: Sequence[Dict[str, Any]], *, full: bool = False) -> str:
    if not paths:
        return "No candidate paths were generated."
    if full:
        return pretty_json(list(paths))
    lines: List[str] = []
    for index, path in enumerate(paths, start=1):
        if not isinstance(path, dict):
            continue
        final_outcome = path.get("final_outcome", {})
        summary = {
            "index": index,
            "depth": path.get("depth"),
            "goal_weighted_score": path.get("goal_weighted_score"),
            "goal_alignment_score": path.get("goal_alignment_score"),
            "grounded_score": path.get("grounded_score"),
            "plausibility_score": path.get("plausibility_score"),
            "source_levels": [step.get("source_level") for step in path.get("steps", []) if isinstance(step, dict)],
            "final_outcome": {
                "action": final_outcome.get("action") if isinstance(final_outcome, dict) else None,
                "object": final_outcome.get("slots", {}).get("object") if isinstance(final_outcome, dict) and isinstance(final_outcome.get("slots", {}), dict) else None,
                "topic": final_outcome.get("slots", {}).get("topic") if isinstance(final_outcome, dict) and isinstance(final_outcome.get("slots", {}), dict) else None,
            },
        }
        lines.append(short_json(summary, max_length=320))
    return "\n".join(lines)


def summarize_recalled_turns(turns: Sequence[Dict[str, Any]], *, full: bool = False) -> str:
    if not turns:
        return "No episodic turns were recalled."
    if full:
        return pretty_json(list(turns))
    lines: List[str] = []
    for index, item in enumerate(turns, start=1):
        if not isinstance(item, dict):
            continue
        record = item.get("record", {})
        summary = {
            "index": index,
            "score": item.get("score"),
            "route_type": record.get("route_type") if isinstance(record, dict) else None,
            "prompt": record.get("raw_user_prompt") if isinstance(record, dict) else None,
            "answer": record.get("final_answer") if isinstance(record, dict) else None,
            "candidate_path_count": record.get("candidate_path_count") if isinstance(record, dict) else None,
            "structured_result": (
                {
                    "action": record.get("final_selected_result", {}).get("action"),
                    "object": record.get("final_selected_result", {}).get("slots", {}).get("object"),
                    "topic": record.get("final_selected_result", {}).get("slots", {}).get("topic"),
                }
                if isinstance(record, dict) and isinstance(record.get("final_selected_result", {}), dict)
                else {}
            ),
        }
        lines.append(short_json(summary, max_length=420))
    return "\n".join(lines)


def summarize_strategy_steps(strategy: Dict[str, Any], *, full: bool = False) -> str:
    if not isinstance(strategy, dict) or not strategy:
        return "No strategy plan was generated."
    steps = strategy.get("steps", [])
    if full:
        return pretty_json(strategy)
    if not isinstance(steps, list) or not steps:
        return short_json(strategy, 500)
    lines: List[str] = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        dependency_text = ",".join(str(item) for item in step.get("depends_on", []) if str(item).strip())
        lines.append(
            short_json(
                {
                    "index": index,
                    "title": step.get("title"),
                    "action": step.get("action"),
                    "risk_level": step.get("risk_level"),
                    "depends_on": dependency_text,
                },
                320,
            )
        )
    return "\n".join(lines) if lines else "No strategy steps were generated."


def render_turn_sections(result: Dict[str, Any], *, debug: bool, raw: bool) -> str:
    trace = result.get("trace", {})
    sections = [
        heading("USER PROMPT"),
        str(result.get("user_input_text", "")),
        heading("EFFECTIVE PROMPT"),
        str(result.get("effective_prompt_text", "")),
        heading("ROUTE DECISION"),
        pretty_json(result.get("route_decision", {})) if (raw or debug) else short_json({
            "route_type": result.get("route_decision", {}).get("route_type"),
            "confidence": result.get("route_decision", {}).get("confidence"),
            "reasons": result.get("route_decision", {}).get("reasons", []),
        }, 500),
        heading("CLARIFICATION STATE"),
        pretty_json(result.get("clarification_state", {})) if (raw or debug) else short_json({
            "active": result.get("clarification_state", {}).get("active"),
            "resolved": result.get("clarification_state", {}).get("resolved"),
            "clarification_question": result.get("clarification_state", {}).get("clarification_question"),
            "missing_fields": result.get("clarification_state", {}).get("missing_fields", []),
            "clarification_response": result.get("clarification_state", {}).get("clarification_response"),
            "merged_prompt": result.get("clarification_state", {}).get("merged_prompt"),
        }, 600),
        heading("GOAL STATUS"),
        pretty_json(result.get("goal_tracking", {}).get("goal_status", [])) if (raw or debug) else short_json(result.get("goal_tracking", {}).get("goal_status", []), 650),
        heading("PROGRESS TRACKING"),
        pretty_json(result.get("goal_tracking", {}).get("progress_tracking", {})) if (raw or debug) else short_json(result.get("goal_tracking", {}).get("progress_tracking", {}), 650),
        heading("NEXT STEP SUGGESTION"),
        pretty_json(result.get("goal_tracking", {}).get("next_step_suggestion", {})) if (raw or debug) else short_json(result.get("goal_tracking", {}).get("next_step_suggestion", {}), 650),
        heading("MERGED MEMORY STATE"),
        pretty_json(result.get("trace", {}).get("merged_memory_state", {})) if (raw or debug) else short_json(result.get("trace", {}).get("merged_memory_state", {}), 800),
        heading("STRATEGY PLAN"),
        pretty_json(result.get("strategy_plan", {}).get("selected_strategy", {})) if (raw or debug) else short_json({
            "name": result.get("strategy_plan", {}).get("selected_strategy", {}).get("name"),
            "focus": result.get("strategy_plan", {}).get("selected_strategy", {}).get("focus"),
            "goal": result.get("strategy_plan", {}).get("selected_strategy", {}).get("goal"),
            "scores": result.get("strategy_plan", {}).get("selected_strategy", {}).get("scores", {}),
            "tradeoffs": result.get("strategy_plan", {}).get("selected_strategy", {}).get("tradeoffs", []),
            "explanation": result.get("strategy_plan", {}).get("explanation"),
        }, 800),
        heading("STEP SEQUENCE"),
        summarize_strategy_steps(result.get("strategy_plan", {}).get("selected_strategy", {}), full=raw or debug),
        heading("ALTERNATIVE OPTIONS"),
        pretty_json(result.get("strategy_plan", {}).get("alternative_strategies", [])) if (raw or debug) else short_json(result.get("strategy_plan", {}).get("alternative_strategies", []), 900),
        heading("PARSED BRAIN INPUT"),
        pretty_json(result.get("parsed_brain_input", {})) if raw else short_json(result.get("parse_result", {}).get("memory", {}), 600),
        heading("ENRICHMENT RESULT"),
        pretty_json(trace.get("enrichment_result", {})) if (raw or debug) else short_json({
            "enrichment_applied": trace.get("enrichment_result", {}).get("enrichment_applied"),
            "needs_clarification": trace.get("enrichment_result", {}).get("needs_clarification"),
            "next_action": trace.get("enrichment_result", {}).get("next_action"),
            "clarification_question": trace.get("enrichment_result", {}).get("clarification_question"),
            "missing_fields": trace.get("enrichment_result", {}).get("missing_fields", []),
            "clarification_type": trace.get("enrichment_result", {}).get("clarification_type"),
            "clarification_reasons": trace.get("enrichment_result", {}).get("clarification_reasons", []),
            "inferred_fields": trace.get("enrichment_result", {}).get("inferred_fields", []),
        }, 550),
        heading("RECALLED TURNS"),
        summarize_recalled_turns(result.get("recalled_turns", []), full=raw or debug),
        heading("RETRIEVED MEMORY"),
        summarize_memory_entries(result.get("trace", {}).get("retrieved_memories", []), full=raw or debug),
        heading("CANDIDATE PATHS"),
        summarize_paths(trace.get("goal_scored_paths", []), full=raw),
        heading("GOAL / GROUNDING SCORES"),
        pretty_json({
            "goal_weights": trace.get("goal_weights", {}),
            "concept_matches": trace.get("concept_matches", []),
            "selected_scores": trace.get("selected_path", {}),
        }) if (raw or debug) else short_json({
            "goal_weights": trace.get("goal_weights", {}),
            "concept_match_count": len(trace.get("concept_matches", [])),
            "selected_scores": {
                "goal_alignment_score": trace.get("selected_path", {}).get("goal_alignment_score"),
                "grounded_score": trace.get("selected_path", {}).get("grounded_score"),
                "plausibility_score": trace.get("selected_path", {}).get("plausibility_score"),
            },
        }, 500),
        heading("SELECTED PATH"),
        pretty_json(trace.get("selected_path", {})) if (raw or debug) else short_json(trace.get("selected_path", {}), 700),
        heading("FINAL STRUCTURED RESULT"),
        pretty_json(result.get("final_result", {})) if raw else short_json(result.get("final_result", {}), 500),
        heading("LLM EXPLANATION"),
        str(result.get("llm_explanation", {}).get("user_facing_text", "")),
        heading("MEMORY WRITE RESULT"),
        pretty_json({
            "memory_written": trace.get("memory_written", False),
            "stored_entries": trace.get("stored_entries", []),
        }) if (raw or debug) else short_json({
            "memory_written": trace.get("memory_written", False),
            "stored_entry_count": len(trace.get("stored_entries", [])),
            "stored_entry_types": [item.get("entry_type") for item in trace.get("stored_entries", []) if isinstance(item, dict)],
        }, 300),
    ]
    return "\n".join(sections)
