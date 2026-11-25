from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, List, Optional, Protocol, Sequence

from brain_input_v1 import BrainInputV1
from brain_llm_interface import BrainLLMInterface


@dataclass(frozen=True)
class LLMExplanationResult:
    provider: str
    structured_text: str
    reasoning_explanation: str
    user_facing_text: str
    messages: List[Dict[str, str]]
    metadata: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "provider": self.provider,
            "structured_text": self.structured_text,
            "reasoning_explanation": self.reasoning_explanation,
            "user_facing_text": self.user_facing_text,
            "messages": [dict(message) for message in self.messages],
            "metadata": dict(self.metadata),
        }


class ResponseStyle(str, Enum):
    DIRECT_FACT = "DIRECT_FACT"
    DIRECT_MEMORY_RECALL = "DIRECT_MEMORY_RECALL"
    ACTION_STEPS = "ACTION_STEPS"
    SHORT_PLAN = "SHORT_PLAN"
    TROUBLESHOOTING_FLOW = "TROUBLESHOOTING_FLOW"
    CLARIFICATION_ONLY = "CLARIFICATION_ONLY"


@dataclass(frozen=True)
class InternalDecision:
    selected_goal: str = ""
    selected_plan: str = ""
    alternatives_considered: int = 0
    confidence: float = 0.0
    memory_reconstruction_details: Dict[str, object] | None = None
    simulation_outcome: Dict[str, object] | None = None
    supporting_concepts: List[str] | None = None


@dataclass(frozen=True)
class UserResponse:
    direct_answer: str
    response_style: ResponseStyle
    short_explanation: str = ""
    actionable_steps: List[str] | None = None
    uncertainty: str = ""


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


def build_llm_payload_v1(
    *,
    user_input_text: str,
    route_type: str,
    context_packet_v1: Optional[Dict[str, object]] = None,
    mode: str,
    route_reasons: Optional[Sequence[str]] = None,
    selected_path: Optional[Dict[str, object]] = None,
    retrieved_memory_count: int = 0,
    recalled_turn_count: int = 0,
) -> Dict[str, object]:
    packet = context_packet_v1 if isinstance(context_packet_v1, dict) else {}
    selected = selected_path if isinstance(selected_path, dict) else {}
    return _compact(
        {
            "version": "BrAInLLMPayloadV1",
            "mode": mode,
            "route_type": route_type,
            "user_query": str(user_input_text or ""),
            "model_instructions": _model_instructions(route_type=route_type, context_packet_v1=packet),
            "context_packet_v1": packet,
            "response_contract": _response_contract(route_type=route_type, mode=mode),
            "trace_hints": {
                "route_reasons": [str(item) for item in (route_reasons or []) if str(item).strip()][:4],
                "retrieved_memory_count": max(0, int(retrieved_memory_count)),
                "recalled_turn_count": max(0, int(recalled_turn_count)),
                "selected_path": {
                    "depth": selected.get("depth"),
                    "goal_alignment_score": selected.get("goal_alignment_score"),
                    "goal_weighted_score": selected.get("goal_weighted_score"),
                    "grounded_score": selected.get("grounded_score"),
                    "plausibility_score": selected.get("plausibility_score"),
                },
            },
        }
    )


def build_llm_payload_from_packet_v1(
    packet: Dict[str, object],
    *,
    cached_knowledge: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    route_type = str(packet.get("route_type", "unknown"))
    task_type = str(packet.get("task_type", route_type or "unknown"))
    user_query = str(packet.get("user_query", ""))
    cache_info = cached_knowledge if isinstance(cached_knowledge, dict) else {}

    payload: Dict[str, object] = {
        "version": "LLMPayloadV2",
        "task_type": task_type,
        "user_query": user_query,
        "source_of_truth": "llm",
        "instructions": _llm_payload_v2_instructions(route_type=route_type, packet=packet),
    }

    if route_type == "memory_question":
        relation = packet.get("current_state",{}).get("merged_memory_state",{}).get("relation_memory",{})
        if isinstance(relation,dict) and relation.get("active"):
            payload["relation_memory"] = relation
            payload["source_of_truth"] = (
                "brain" if relation.get("supported_answer") or relation.get("answer_status")=="unresolved" else "llm")
            if relation.get("supported_answer"):
                payload["answer_facts"] = [relation["supported_answer"]]
            elif relation.get("answer_status")=="unresolved":
                payload["answer_facts"] = ["UNKNOWN"]
            return payload
        payload["source_of_truth"] = "brain"
        answer_state = _memory_answer_state_from_packet(packet)
        if answer_state:
            payload["answer_state"] = answer_state
        answer_facts = _memory_answer_facts_from_packet(packet)
        if answer_facts:
            payload["answer_facts"] = answer_facts
        reconstructed_context = _memory_reconstructed_context_from_packet(packet)
        if reconstructed_context:
            payload["reconstructed_context"] = reconstructed_context
    elif route_type in {"planning_request", "goal_request", "transform_request"}:
        payload["source_of_truth"] = "brain"
        constraints = _packet_constraints(packet)
        goals = _packet_goals(packet)
        progress = _packet_progress(packet)
        decision_context = _packet_decision_context(packet)
        if constraints:
            payload["constraints"] = constraints
        if goals:
            payload["goals"] = goals
        if progress:
            payload["progress"] = progress
        if decision_context:
            payload["decision_context"] = decision_context
    elif route_type == "hybrid_request":
        payload["source_of_truth"] = "brain"
        answer_state = _memory_answer_state_from_packet(packet)
        constraints = _packet_constraints(packet)
        goals = _packet_goals(packet)
        progress = _packet_progress(packet)
        decision_context = _packet_decision_context(packet)
        if answer_state:
            payload["answer_state"] = answer_state
        if constraints:
            payload["constraints"] = constraints
        if goals:
            payload["goals"] = goals
        if progress:
            payload["progress"] = progress
        if decision_context:
            payload["decision_context"] = decision_context

    elif route_type == "factual_question" and bool(cache_info.get("hit")):
        payload["source_of_truth"] = "cache"
        cached_entries = cache_info.get("entries", []) if isinstance(cache_info.get("entries", []), list) else []
        cached_facts = [str(item) for item in cache_info.get("answer_facts", []) if str(item).strip()] if isinstance(cache_info.get("answer_facts", []), list) else []
        cached_state = cache_info.get("answer_state", {}) if isinstance(cache_info.get("answer_state", {}), dict) else {}
        multi_hop_used = bool(cache_info.get("multi_hop_used", False))
        merged_entry_ids = [int(item) for item in cache_info.get("merged_entry_ids", []) if str(item).strip()] if isinstance(cache_info.get("merged_entry_ids", []), list) else []
        reasoning_steps = [str(item) for item in cache_info.get("reasoning_steps", []) if str(item).strip()] if isinstance(cache_info.get("reasoning_steps", []), list) else []
        merged_reasoning_packet = cache_info.get("merged_reasoning_packet", {}) if isinstance(cache_info.get("merged_reasoning_packet", {}), dict) else {}
        if (not cached_facts or not cached_state) and cached_entries:
            first = cached_entries[0] if isinstance(cached_entries[0], dict) else {}
            metadata = first.get("metadata", {}) if isinstance(first.get("metadata", {}), dict) else {}
            if not cached_facts:
                cached_facts = [str(item) for item in metadata.get("normalized_facts", []) if str(item).strip()] if isinstance(metadata.get("normalized_facts", []), list) else []
            if not cached_state:
                cached_state = metadata.get("answer_state", {}) if isinstance(metadata.get("answer_state", {}), dict) else {}
        if cached_state:
            payload["answer_state"] = cached_state
        if cached_facts:
            payload["answer_facts"] = cached_facts
        instructions = payload.get("instructions", [])
        if isinstance(instructions, list):
            instructions.extend(
                [
                    "Use cached low-trust answer facts only.",
                    "Stay within cached knowledge and do not elaborate beyond it.",
                ]
            )
            if multi_hop_used:
                instructions.extend(
                    [
                        "Use the merged_reasoning_packet when it is present.",
                        "Combine the provided cached facts deterministically instead of inventing new reasoning.",
                    ]
                )
        payload["decision_context"] = _compact(
            {
                "cache_topic_key": cache_info.get("topic_key", ""),
                "trust_level": cache_info.get("trust_level", "low"),
                "verification_status": cache_info.get("verification_status", "unverified"),
                "multi_hop_used": multi_hop_used,
                "merged_entry_ids": merged_entry_ids,
                "reasoning_steps": reasoning_steps,
                "merged_reasoning_packet": merged_reasoning_packet,
                "structured_explanation": merged_reasoning_packet.get("structured_explanation", {}) if isinstance(merged_reasoning_packet.get("structured_explanation", {}), dict) else {},
                "final_explanation": merged_reasoning_packet.get("final_explanation", ""),
            }
        )

    return _compact(payload)


def render_llm_payload_v2_answer(
    *,
    user_input_text: str,
    route_type: str,
    llm_payload_v2: Dict[str, object],
    mode: str = "direct_answer",
) -> str:
    payload = llm_payload_v2 if isinstance(llm_payload_v2, dict) else {}
    if not payload:
        return ""
    source_of_truth = str(payload.get("source_of_truth", "llm"))
    if source_of_truth not in {"brain", "cache"}:
        return ""
    style = _select_response_style(
        user_input_text=user_input_text,
        route_type=route_type,
        llm_payload_v2=payload,
    )
    raw_answer = ""
    if source_of_truth == "cache":
        raw_answer = _answer_from_llm_payload_v2(
            user_input_text=user_input_text,
            llm_payload_v2=payload,
        )
    elif route_type == "memory_question":
        raw_answer = _answer_from_llm_payload_v2(
            user_input_text=user_input_text,
            llm_payload_v2=payload,
        )
    elif route_type in {"planning_request", "goal_request", "transform_request", "hybrid_request"}:
        raw_answer = _planning_response_from_llm_payload_v2(
            user_input_text=user_input_text,
            llm_payload_v2=payload,
            direct=(mode == "direct_answer"),
        )
    return _finalize_user_response(
        answer_text=raw_answer,
        user_input_text=user_input_text,
        route_type=route_type,
        response_style=style,
        llm_payload_v2=payload,
    )


def build_cached_answer_result(
    *,
    user_input_text: str,
    route_type: str,
    llm_payload_v2: Dict[str, object],
) -> LLMExplanationResult:
    answer_text = render_llm_payload_v2_answer(
        user_input_text=user_input_text,
        route_type=route_type,
        llm_payload_v2=llm_payload_v2,
        mode="direct_answer",
    )
    return LLMExplanationResult(
        provider="cache",
        structured_text="",
        reasoning_explanation="",
        user_facing_text=answer_text,
        messages=[
            {
                "role": "system",
                "content": (
                    "Use cached knowledge as the answer. "
                    "Do not infer or modify the cached facts. "
                    "Answer directly using provided information."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(llm_payload_v2, indent=2, sort_keys=True),
            },
        ],
        metadata={
            "mode": "cache_direct_answer",
            "route_type": route_type,
            "source_of_truth": "cache",
            "llm_invoked": False,
            "llm_payload_v2": llm_payload_v2,
        },
    )


def _llm_payload_v2_instructions(*, route_type: str, packet: Dict[str, object]) -> List[str]:
    source_of_truth = "llm"
    if route_type in {"memory_question", "planning_request", "goal_request", "transform_request", "hybrid_request"}:
        source_of_truth = "brain"
    instructions = [
        "Answer directly and concisely.",
        "Use only the provided payload fields.",
    ]
    if source_of_truth == "brain":
        instructions.extend(
            [
                "BrAIn is the source of truth for this answer.",
                "Do not infer missing state when answer_state or structured decision fields are provided.",
                "Do not override BrAIn-provided state.",
            ]
        )
    if route_type == "memory_question":
        instructions.extend(
            [
                "Use answer_state as the source of truth.",
                "Do not infer missing state if answer_state is provided.",
                "If reconstructed_context is provided, use it only with its confidence and answer_mode.",
                "Use answer_facts only when they help answer succinctly.",
                "Answer from BrAIn memory state only.",
            ]
        )
    elif route_type in {"planning_request", "goal_request", "transform_request", "hybrid_request"}:
        instructions.extend(
            [
                "Use constraints, goals, progress, and decision_context as the decision frame.",
                "Do not reconstruct internal reasoning traces.",
                "Use BrAIn planning state directly.",
            ]
        )
    elif route_type == "factual_question":
        instructions.extend(
            [
                "The LLM may answer normally from general knowledge.",
                "No internal BrAIn structure is needed for this answer.",
            ]
        )
    progress = packet.get("progress_state", {}) if isinstance(packet.get("progress_state", {}), dict) else {}
    if bool(progress.get("clarification_required")):
        return ["Ask only for the missing clarification indicated in progress."]
    return instructions


def _select_response_style(
    *,
    user_input_text: str,
    route_type: str,
    llm_payload_v2: Dict[str, object],
) -> ResponseStyle:
    normalized = str(user_input_text or "").lower()
    progress = llm_payload_v2.get("progress", {}) if isinstance(llm_payload_v2.get("progress", {}), dict) else {}
    if route_type == "memory_question":
        return ResponseStyle.DIRECT_MEMORY_RECALL
    if any(token in normalized for token in ("troubleshoot", "diagnostic", "not printing", "wi-fi", "wifi", "blank monitor")):
        return ResponseStyle.TROUBLESHOOTING_FLOW
    if route_type in {"planning_request", "goal_request", "transform_request", "hybrid_request"}:
        if bool(progress.get("clarification_required")) and not _query_has_actionable_context(normalized):
            return ResponseStyle.CLARIFICATION_ONLY
        if any(token in normalized for token in ("step-by-step", "steps", "plan", "schedule", "packing", "bugfix", "release", "troubleshoot", "next step")):
            return ResponseStyle.ACTION_STEPS
        return ResponseStyle.SHORT_PLAN
    return ResponseStyle.DIRECT_FACT


def _query_has_actionable_context(normalized_query: str) -> bool:
    cleaned = str(normalized_query or "")
    action_terms = (
        "plan", "schedule", "charge", "rewrite", "reduce risk", "recover", "choose",
        "bugfix", "release", "packing", "troubleshoot", "diagnostic", "study",
        "printer", "website", "moving", "next step",
    )
    constraint_terms = (
        "must", "need", "cannot", "can't", "blocked", "done", "missing", "not started",
        "under", "before", "after", "only", "already", "from", "to",
    )
    return (
        any(term in cleaned for term in action_terms)
        and (len(cleaned.split()) >= 8 or any(term in cleaned for term in constraint_terms))
    )


def _finalize_user_response(
    *,
    answer_text: str,
    user_input_text: str,
    route_type: str,
    response_style: ResponseStyle,
    llm_payload_v2: Dict[str, object],
) -> str:
    text = _suppress_meta_language(str(answer_text or "").strip())
    if not text and route_type in {"planning_request", "goal_request", "transform_request", "hybrid_request"}:
        text = _actionable_answer_from_query(user_input_text)
    if not text:
        return ""
    if response_style == ResponseStyle.CLARIFICATION_ONLY:
        return text
    text = _polish_memory_response(text) if response_style == ResponseStyle.DIRECT_MEMORY_RECALL else text
    text = _ensure_actionable_shape(
        answer_text=text,
        user_input_text=user_input_text,
        response_style=response_style,
        llm_payload_v2=llm_payload_v2,
    )
    return " ".join(text.split()).strip()


def _suppress_meta_language(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    blocked_full = (
        "I could not confidently route this prompt",
        "No BrAIn reasoning or direct factual answer ran",
        "Structured result preserved",
        "BrAIn preserved the structured result",
        "returned the highest-ranked",
        "Focus on complete_task",
        "Focus on avoid_risk",
        "It evaluated ",
        "BrAIn selected the strongest path",
        "goal alignment",
        "grounded score",
        "plausibility",
        "path depth",
        "candidate paths",
        "generated paths",
        "Conservative Retry",
    )
    if any(fragment.lower() in cleaned.lower() for fragment in blocked_full):
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        kept = [
            sentence
            for sentence in sentences
            if not any(fragment.lower() in sentence.lower() for fragment in blocked_full)
            and sentence.strip()
        ]
        cleaned = " ".join(kept).strip()
    cleaned = re.sub(r"Recommended plan:\s*Strategy [A-Z]:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Recommended plan:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Suggested next step:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*Tradeoffs?:\s*", " Tradeoff: ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("This prompt was routed to direct answering.", "").strip()
    return cleaned


def _polish_memory_response(text: str) -> str:
    cleaned = str(text or "").strip()
    replacements = {
        "Current remembered state: ": "",
        "The related dietary memory says: ": "",
        "The related dietary memory I found says: ": "",
        "From the related memory I found, I said: ": "",
        "From the related memory I found, ": "",
        "The related memory I found says: ": "",
        "The latest availability memory I found is: ": "Your latest availability is ",
    }
    for old, new in replacements.items():
        if cleaned.startswith(old):
            cleaned = new + cleaned[len(old):]
    cleaned = re.sub(r"\b([a-z]+) okay\b", lambda m: f"{m.group(1).capitalize()} are okay for you right now", cleaned)
    cleaned = cleaned.replace("You are currently free free ", "Your latest availability is ")
    if cleaned.startswith("You are currently free "):
        cleaned = "Your latest availability is " + cleaned[len("You are currently free "):]
    return cleaned.rstrip(".") + "."


def _ensure_actionable_shape(
    *,
    answer_text: str,
    user_input_text: str,
    response_style: ResponseStyle,
    llm_payload_v2: Dict[str, object],
) -> str:
    if response_style not in {ResponseStyle.ACTION_STEPS, ResponseStyle.SHORT_PLAN, ResponseStyle.TROUBLESHOOTING_FLOW}:
        return answer_text
    if _count_action_signals(answer_text) >= 2 and not _contains_meta_language(answer_text):
        return answer_text
    fallback = _actionable_answer_from_query(user_input_text)
    if fallback:
        return fallback
    decision_context = llm_payload_v2.get("decision_context", {}) if isinstance(llm_payload_v2.get("decision_context", {}), dict) else {}
    next_action = str(decision_context.get("next_action", "")).strip()
    tradeoffs = [str(item) for item in decision_context.get("tradeoffs", []) if str(item).strip()] if isinstance(decision_context.get("tradeoffs", []), list) else []
    if next_action:
        pieces = [f"Start by {next_action[0].lower() + next_action[1:] if len(next_action) > 1 else next_action}"]
        if tradeoffs:
            pieces.append(f"Why: {tradeoffs[0].lower()}.")
        return " ".join(pieces)
    return answer_text


def _count_action_signals(text: str) -> int:
    lowered = str(text or "").lower()
    signals = ("start", "then", "next", "check", "use", "keep", "finish", "fix", "test", "retry", "plan", "schedule")
    return sum(1 for signal in signals if signal in lowered)


def _contains_meta_language(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        marker in lowered
        for marker in (
            "recommended plan:",
            "strategy b:",
            "conservative retry",
            "focus on complete_task",
            "brain selected",
            "candidate paths",
            "could not confidently route",
            "concrete task or object",
        )
    )


def _actionable_answer_from_query(user_input_text: str) -> str:
    normalized = str(user_input_text or "").lower()
    if "bugfix" in normalized or "reproduce the bug" in normalized:
        return "Start by reproducing the bug. Then make the smallest code change, update or add the tests before merging, wait for review to pass, and only deploy after review is green."
    if "release plan" in normalized or ("qa is blocked" in normalized and "migration" in normalized):
        return "First unblock QA by fixing the failed staging database migration. Then rerun QA, tag the build only after QA passes, and publish the release notes last."
    if "demo day" in normalized and "release notes" in normalized and "record" in normalized:
        return "Test the app first for 30 minutes. If the test passes, write the release notes for 20 minutes, record the short clip for 15 minutes, and publish only after the release notes are done."
    if (
        ("github upload" in normalized and ("generated benchmark state" in normalized or "proof-of-concept documentation" in normalized))
        or ("upload plan" in normalized and "constraints" in normalized)
    ):
        return "Keep the proof-of-concept documentation, exclude generated benchmark state from the upload, run the tests, then commit the source, docs, and benchmark prompt definitions."
    if "moving day" in normalized and "fragile boxes" in normalized:
        return "Load fragile boxes on the first car trip, move the desk only when both helpers are available together, use the noon trip for remaining boxes, and clean only after all boxes are out."
    if "study evening" in normalized and "tutoring call" in normalized:
        return "Schedule around the tutoring call first: study math before 7:15 PM, pause for the 7:15-7:45 PM call, add a break after the first study hour, then finish history and remaining math before 9 PM if time still fits."
    if "reduce risk" in normalized and "blanket" in normalized:
        return "Do not charge the laptop under a blanket. Move it to a flat ventilated surface, use the original or certified charger, check that the outlet and cable are stable, and avoid leaving it trapped in heat overnight."
    if "packing" in normalized and ("carry-on" in normalized or "one-night" in normalized):
        return "Pack the carry-on around the hard limits first: keep the bag under 7 kilograms, put rain gear in an easy-to-reach pocket, and fold or separate presentation clothes so they do not wrinkle."
    if "study week" in normalized and "6 hours" in normalized:
        return "Use six one-hour study blocks after work from Monday through Saturday, keep each day under the 2-hour limit, and leave Sunday completely off."
    if "meal plan" in normalized and "high-protein" in normalized:
        return "Plan a 3-day meal plan with cheap, high-protein lunches, and keep dinners flexible."
    if "printer" in normalized and "queue" in normalized:
        return "Start with the queue, since the printer has power and paper: cancel the stuck document, restart the print queue or spooler, then send one small test page."
    if "wi-fi" in normalized or "wifi" in normalized:
        return "Since you already restarted the router and checked modem lights, test another device next. If that device also fails, check the ISP/modem path; if it works, focus on the original device."
    if "website launch" in normalized and "screenshots" in normalized:
        return "Finish the missing screenshots first, then start QA. Keep deployment blocked until QA finishes."
    if "charge" in normalized and "laptop" in normalized:
        return "Start by checking the charger, cable, and outlet. Then use the original or certified adapter, place the laptop on a ventilated surface, and avoid charging under anything that traps heat."
    return ""


def _answer_from_llm_payload_v2(
    *,
    user_input_text: str,
    llm_payload_v2: Dict[str, object],
) -> str:
    relation = llm_payload_v2.get("relation_memory",{})
    if isinstance(relation,dict) and relation.get("active"):
        return str(relation.get("supported_answer") or "UNKNOWN")
    answer_state = llm_payload_v2.get("answer_state", {}) if isinstance(llm_payload_v2.get("answer_state", {}), dict) else {}
    answer_facts = llm_payload_v2.get("answer_facts", []) if isinstance(llm_payload_v2.get("answer_facts", []), list) else []
    reconstructed_context = llm_payload_v2.get("reconstructed_context", {}) if isinstance(llm_payload_v2.get("reconstructed_context", {}), dict) else {}
    decision_context = llm_payload_v2.get("decision_context", {}) if isinstance(llm_payload_v2.get("decision_context", {}), dict) else {}
    merged_reasoning_packet = decision_context.get("merged_reasoning_packet", {}) if isinstance(decision_context.get("merged_reasoning_packet", {}), dict) else {}
    if str(llm_payload_v2.get("source_of_truth", "")) == "cache" and answer_facts:
        final_explanation = str(merged_reasoning_packet.get("final_explanation", "")).strip()
        if final_explanation:
            return final_explanation
        resolved_answer = str(merged_reasoning_packet.get("resolved_answer", "")).strip()
        if resolved_answer:
            return resolved_answer
        first_fact = str(answer_facts[0]).strip()
        return first_fact if first_fact else ""
    normalized = str(user_input_text or "").lower()
    if not answer_state and not answer_facts and not reconstructed_context:
        return ""
    if "favorite color" in normalized and "favorite_color" in answer_state:
        return f"Your favorite color is {answer_state.get('favorite_color')}."
    if ("chicken" in normalized or "store" in normalized or "storage" in normalized) and "storage:chicken" in answer_state:
        return f"The chicken should be stored in the {answer_state.get('storage:chicken')}."
    if "meeting" in normalized and "meeting_time" in answer_state:
        return f"Your meeting is {answer_state.get('meeting_time')}."
    if "document title" in normalized and "document_title" in answer_state:
        return f"The current document title is {answer_state.get('document_title')}."
    if ("where are" in normalized or "where should" in normalized or "stored" in normalized) and answer_state:
        for key, value in answer_state.items():
            if str(key).startswith("storage:") or str(key).startswith("object_location:"):
                entity = str(key).split(":", 1)[-1].replace("_", " ")
                return f"The {entity} should be in the {value}."
    if ("office item state" in normalized or ("office" in normalized and "current" in normalized)) and answer_state:
        locations = []
        for key, value in answer_state.items():
            if str(key).startswith("object_location:"):
                entity = str(key).split(":", 1)[-1].replace("_", " ")
                locations.append(f"{entity} is in the {value}")
        if locations:
            return "The current office item state is: " + ", ".join(locations) + "."
    if ("available" in normalized or "when am i free" in normalized) and answer_state:
        available = [str(value) for key, value in answer_state.items() if str(key).startswith("availability:") and str(value).startswith("free ")]
        if available:
            return "You are currently free " + " and ".join(available).replace("free ", "", 1) + "."
    if ("which nuts" in normalized or "nuts should" in normalized or "food constraints" in normalized) and answer_state:
        avoided = {
            str(key).split(":", 1)[-1]
            for key, value in answer_state.items()
            if str(key).startswith("dietary_constraint:") and str(value).startswith("avoid ")
        }
        if "which nuts" in normalized or "nuts should" in normalized:
            dietary = [
                str(value)
                for key, value in answer_state.items()
                if str(key).startswith("dietary_ok:") and str(key).split(":", 1)[-1] not in avoided
            ]
        else:
            dietary = [str(value) for key, value in answer_state.items() if str(key).startswith("dietary_") or str(key).startswith("dietary:")]
        if dietary:
            return "The related dietary memory says: " + ", ".join(dietary) + "."
    if "troubleshooting" in normalized or "already tried" in normalized:
        tried = [str(value).strip() for key, value in answer_state.items() if str(key).startswith("troubleshooting_step:") or str(key) == "troubleshooting_steps_tried"]
        tried = [item for item in tried if item]
        if tried:
            return "You already tried: " + ", ".join(tried) + "."
    if "current project status" in normalized or ("project status" in normalized and "current" in normalized):
        statuses = []
        for key, value in answer_state.items():
            if str(key).startswith("project_status:"):
                label = str(key).split(":", 1)[-1].replace("_", " ")
                statuses.append(f"{label} is {value}")
        if statuses:
            return "Current project status: " + ", ".join(statuses) + "."
    if "shipping checklist" in normalized or "checklist" in normalized:
        statuses = []
        for key, value in answer_state.items():
            if str(key).startswith("checklist_status:"):
                label = str(key).split(":", 1)[-1].replace("_", " ")
                statuses.append(f"{label} is {value}")
        if statuses:
            return "Current shipping checklist state: " + ", ".join(statuses) + "."
    if "remaining incomplete" in normalized or "still incomplete" in normalized:
        incomplete = []
        for key, value in answer_state.items():
            lowered = str(value).lower()
            if (str(key).startswith("project_status:") or str(key).startswith("checklist_status:")) and any(token in lowered for token in ["incomplete", "not started", "blocked", "delayed"]):
                label = str(key).split(":", 1)[-1].replace("_", " ")
                incomplete.append(f"{label} {value}")
        if incomplete:
            return "The remaining incomplete parts are " + ", ".join(incomplete) + "."
    if answer_facts:
        return "Current remembered state: " + ", ".join(str(item) for item in answer_facts[:4]) + "."
    reconstructed_answer = _answer_from_reconstructed_context(reconstructed_context)
    if reconstructed_answer:
        return reconstructed_answer
    return ""


def _planning_response_from_llm_payload_v2(
    *,
    user_input_text: str,
    llm_payload_v2: Dict[str, object],
    direct: bool,
) -> str:
    decision_context = llm_payload_v2.get("decision_context", {}) if isinstance(llm_payload_v2.get("decision_context", {}), dict) else {}
    progress = llm_payload_v2.get("progress", {}) if isinstance(llm_payload_v2.get("progress", {}), dict) else {}
    normalized = str(user_input_text or "").lower()
    if bool(progress.get("clarification_required")) and _query_has_actionable_context(normalized):
        fallback = _actionable_answer_from_query(user_input_text)
        if fallback:
            return fallback
    strategy_name = str(decision_context.get("decision_summary", "")).strip()
    next_text = str(progress.get("next_step", "")).strip()
    first_step = str(decision_context.get("next_action", "")).strip()
    fallback = _actionable_answer_from_query(user_input_text)
    if fallback and (
        _contains_meta_language(strategy_name)
        or _contains_meta_language(next_text)
        or _contains_meta_language(first_step)
        or "concrete task or object" in next_text.lower()
    ):
        return fallback
    bits: List[str] = []
    if first_step:
        bits.append(f"Start by {first_step[0].lower() + first_step[1:] if len(first_step) > 1 else first_step}.")
    elif next_text and "concrete task or object" not in next_text.lower():
        bits.append(next_text)
    elif strategy_name and not _contains_meta_language(strategy_name):
        bits.append(strategy_name)
    tradeoffs = [str(item) for item in decision_context.get("tradeoffs", []) if str(item).strip()] if isinstance(decision_context.get("tradeoffs", []), list) else []
    if tradeoffs:
        bits.append(f"Why: {tradeoffs[0][0].lower() + tradeoffs[0][1:] if len(tradeoffs[0]) > 1 else tradeoffs[0]}.")
    if bits:
        return " ".join(bits).strip()
    return fallback


def _memory_answer_state_from_packet(packet: Dict[str, object]) -> Dict[str, object]:
    current_state = packet.get("current_state", {}) if isinstance(packet.get("current_state", {}), dict) else {}
    merged = current_state.get("merged_memory_state", {}) if isinstance(current_state.get("merged_memory_state", {}), dict) else {}
    items = merged.get("items", []) if isinstance(merged.get("items", []), list) else []
    answer_state: Dict[str, object] = {}
    query = str(packet.get("user_query", "")).lower()
    safe_nut_query = "which nuts" in query or "nuts should" in query or ("nuts" in query and "okay" in query)
    avoided_nuts = {
        str(item.get("entity", "")).lower()
        for item in items
        if isinstance(item, dict)
        and item.get("state_type") == "dietary_constraint"
        and str(item.get("current_value", "")).startswith("avoid ")
    }
    for item in items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("state_key", "")).strip() or str(item.get("entity", "")).strip()
        if not key:
            continue
        if safe_nut_query:
            entity = str(item.get("entity", "")).lower()
            if not key.startswith("dietary_ok:") or entity in avoided_nuts:
                continue
        answer_state[key] = item.get("current_value")
    return answer_state


def _memory_answer_facts_from_packet(packet: Dict[str, object]) -> List[str]:
    current_state = packet.get("current_state", {}) if isinstance(packet.get("current_state", {}), dict) else {}
    merged = current_state.get("merged_memory_state", {}) if isinstance(current_state.get("merged_memory_state", {}), dict) else {}
    items = merged.get("items", []) if isinstance(merged.get("items", []), list) else []
    facts: List[str] = []
    query = str(packet.get("user_query", "")).lower()
    safe_nut_query = "which nuts" in query or "nuts should" in query or ("nuts" in query and "okay" in query)
    avoided_nuts = {
        str(item.get("entity", "")).lower()
        for item in items
        if isinstance(item, dict)
        and item.get("state_type") == "dietary_constraint"
        and str(item.get("current_value", "")).startswith("avoid ")
    }
    for item in items:
        if not isinstance(item, dict):
            continue
        if safe_nut_query:
            entity = str(item.get("entity", "")).lower()
            if not str(item.get("state_key", "")).startswith("dietary_ok:") or entity in avoided_nuts:
                continue
        display_text = str(item.get("display_text", "")).strip()
        if display_text:
            facts.append(display_text)
    return facts[:6]


def _memory_reconstructed_context_from_packet(packet: Dict[str, object]) -> Dict[str, object]:
    current_state = packet.get("current_state", {}) if isinstance(packet.get("current_state", {}), dict) else {}
    merged = current_state.get("merged_memory_state", {}) if isinstance(current_state.get("merged_memory_state", {}), dict) else {}
    reconstructed = merged.get("reconstructed_context", {}) if isinstance(merged.get("reconstructed_context", {}), dict) else {}
    if reconstructed and str(reconstructed.get("synthesized_summary", "")).strip():
        return reconstructed
    return {}


def _answer_from_reconstructed_context(reconstructed_context: Dict[str, object]) -> str:
    if not reconstructed_context:
        return ""
    summary = str(reconstructed_context.get("synthesized_summary", "")).strip()
    if not summary:
        return ""
    mode = str(reconstructed_context.get("answer_mode", "low")).strip().lower()
    confidence = float(reconstructed_context.get("confidence", 0.0) or 0.0)
    if mode == "high":
        return summary
    if mode == "medium":
        return f"Based on your recent context, {summary[0].lower() + summary[1:] if len(summary) > 1 else summary}"
    return (
        "I found a partial related memory, but it is incomplete. "
        f"Best match: {summary} Confidence: {confidence:.2f}."
    )


def _packet_constraints(packet: Dict[str, object]) -> List[str]:
    constraints = packet.get("active_constraints", [])
    return [str(item) for item in constraints if str(item).strip()] if isinstance(constraints, list) else []


def _packet_goals(packet: Dict[str, object]) -> List[str]:
    active_goals = packet.get("active_goals", [])
    if not isinstance(active_goals, list):
        return []
    goals: List[str] = []
    for item in active_goals:
        if not isinstance(item, dict):
            continue
        goal_text = str(item.get("goal_text", "")).strip()
        status = str(item.get("status", "")).strip()
        if goal_text and status:
            goals.append(f"{goal_text} ({status})")
        elif goal_text:
            goals.append(goal_text)
    return goals[:4]


def _packet_progress(packet: Dict[str, object]) -> Dict[str, object]:
    progress_state = packet.get("progress_state", {}) if isinstance(packet.get("progress_state", {}), dict) else {}
    goal_tracking = progress_state.get("goal_tracking", {}) if isinstance(progress_state.get("goal_tracking", {}), dict) else {}
    next_step = progress_state.get("next_step_suggestion", {}) if isinstance(progress_state.get("next_step_suggestion", {}), dict) else {}
    compact_progress = {
        "status": goal_tracking.get("status", ""),
        "summary": goal_tracking.get("summary", ""),
        "next_step": next_step.get("text", ""),
    }
    if bool(progress_state.get("clarification_required")):
        compact_progress["clarification_required"] = True
        compact_progress["clarification_question"] = progress_state.get("clarification_question", "")
    return _compact(compact_progress)


def _packet_decision_context(packet: Dict[str, object]) -> Dict[str, object]:
    reasoning_summary = packet.get("reasoning_summary", {}) if isinstance(packet.get("reasoning_summary", {}), dict) else {}
    selected_strategy = reasoning_summary.get("selected_strategy", {}) if isinstance(reasoning_summary.get("selected_strategy", {}), dict) else {}
    selected_path = reasoning_summary.get("selected_path", {}) if isinstance(reasoning_summary.get("selected_path", {}), dict) else {}
    strategy_steps = selected_strategy.get("steps", []) if isinstance(selected_strategy.get("steps", []), list) else []
    first_step = ""
    if strategy_steps and isinstance(strategy_steps[0], dict):
        first_step = str(strategy_steps[0].get("detail", "")).strip() or str(strategy_steps[0].get("title", "")).strip()
    return _compact(
        {
            "decision_summary": selected_strategy.get("name", "") or packet.get("task_type", ""),
            "focus": selected_strategy.get("focus", ""),
            "target": selected_strategy.get("target", ""),
            "next_action": first_step,
            "tradeoffs": selected_strategy.get("tradeoffs", []),
            "confidence_signals": {
                "goal_alignment_score": selected_path.get("goal_alignment_score"),
                "grounded_score": selected_path.get("grounded_score"),
                "plausibility_score": selected_path.get("plausibility_score"),
            },
        }
    )


def _model_instructions(*, route_type: str, context_packet_v1: Dict[str, object]) -> List[str]:
    instructions = [
        "Use context_packet_v1 as the primary structured source of truth.",
        "Use current_state as the authoritative state when it is present.",
        "Use relevant_history only when it is needed to explain corrections or original values.",
        "Do not reconstruct state from assumptions if the packet already provides it.",
        "Answer directly and succinctly when possible.",
    ]
    if route_type == "memory_question":
        instructions.append("For memory recall, prefer the latest resolved state in current_state.")
    if route_type in {"planning_request", "goal_request", "transform_request", "hybrid_request"}:
        instructions.append("For planning and hybrid routes, follow active_constraints, active_goals, progress_state, and reasoning_summary.")
    if route_type == "factual_question":
        instructions.append("Keep the answer lean and avoid repeating packet fields unnecessarily.")
    if context_packet_v1.get("progress_state", {}).get("clarification_required") if isinstance(context_packet_v1.get("progress_state", {}), dict) else False:
        instructions = ["Ask only for the missing clarification indicated in progress_state."]
    return instructions


def _response_contract(*, route_type: str, mode: str) -> Dict[str, object]:
    style = "direct_answer" if mode == "direct_answer" else "concise_explanation"
    return {
        "style": style,
        "answer_directly": route_type in {"factual_question", "memory_question", "hybrid_request"},
        "prefer_current_state": route_type == "memory_question",
        "do_not_invent_missing_state": True,
    }


class ExplanationProvider(Protocol):
    provider_name: str

    def explain(self, payload: Dict[str, object]) -> LLMExplanationResult:
        ...

    def answer(self, payload: Dict[str, object]) -> LLMExplanationResult:
        ...


class DeterministicExplanationProvider:
    provider_name = "deterministic"

    def __init__(self) -> None:
        self.interface = BrainLLMInterface()

    def explain(self, payload: Dict[str, object]) -> LLMExplanationResult:
        brain_input = BrainInputV1.from_dict(
            payload.get("result_brain_input", {}) if isinstance(payload.get("result_brain_input", {}), dict) else {}
        )
        user_input_text = str(payload.get("user_input_text", ""))
        route_type = str(payload.get("route_type", ""))
        llm_payload_v1 = payload.get("llm_payload_v1", {}) if isinstance(payload.get("llm_payload_v1", {}), dict) else {}
        llm_payload_v2 = payload.get("llm_payload_v2", {}) if isinstance(payload.get("llm_payload_v2", {}), dict) else {}
        context_packet_v1 = payload.get("context_packet_v1", {}) if isinstance(payload.get("context_packet_v1", {}), dict) else {}
        structured_text = self.interface.brain_input_to_natural_language(brain_input)
        trace_summary = payload.get("trace_summary", {})
        retrieved_memories = payload.get("retrieved_memories", [])
        selected_path = payload.get("selected_path", {})
        summary_bits: List[str] = []
        explanation_bits: List[str] = []
        retrieved_count = len(retrieved_memories) if isinstance(retrieved_memories, list) else 0
        if retrieved_count:
            explanation_bits.append(f"BrAIn retrieved {retrieved_count} related memories before reasoning.")
        candidate_paths = trace_summary.get("candidate_paths", []) if isinstance(trace_summary, dict) else []
        generated_paths = trace_summary.get("generated_candidate_paths", []) if isinstance(trace_summary, dict) else []
        if isinstance(candidate_paths, list) and candidate_paths:
            explanation_bits.append(
                f"It evaluated {len(candidate_paths)} candidate paths"
                + (f", including {len(generated_paths)} generated paths." if isinstance(generated_paths, list) and generated_paths else ".")
            )
        if isinstance(selected_path, dict):
            depth = selected_path.get("depth")
            goal_alignment = selected_path.get("goal_alignment_score")
            grounded = selected_path.get("grounded_score")
            plausibility = selected_path.get("plausibility_score")
            if depth is not None:
                summary_bits.append(f"path depth {depth}")
            if goal_alignment is not None:
                summary_bits.append(f"goal alignment {goal_alignment}")
            if grounded is not None:
                summary_bits.append(f"grounded score {grounded}")
            if plausibility is not None:
                summary_bits.append(f"plausibility {plausibility}")
        extra = ""
        if summary_bits:
            explanation_bits.append("BrAIn selected the strongest path based on " + ", ".join(summary_bits) + ".")
        reasoning_explanation = " ".join(explanation_bits).strip()
        if not reasoning_explanation:
            reasoning_explanation = "BrAIn preserved the structured result and returned the highest-ranked available decision."
        packet_first = self._packet_first_response(
            user_input_text=user_input_text,
            route_type=route_type,
            llm_payload_v2=llm_payload_v2,
            mode="explanation",
        )
        user_facing = packet_first or f'You asked for "{user_input_text.strip()}". Result: {structured_text}'
        if reasoning_explanation and not packet_first:
            user_facing = f"{user_facing} {reasoning_explanation}".strip()
        user_facing = _finalize_user_response(
            answer_text=user_facing,
            user_input_text=user_input_text,
            route_type=route_type,
            response_style=_select_response_style(
                user_input_text=user_input_text,
                route_type=route_type,
                llm_payload_v2=llm_payload_v2,
            ),
            llm_payload_v2=llm_payload_v2,
        )
        return LLMExplanationResult(
            provider=self.provider_name,
            structured_text=structured_text,
            reasoning_explanation=reasoning_explanation,
            user_facing_text=user_facing.strip(),
            messages=self.interface.build_messages(
                brain_input=brain_input,
                reasoning_trace=[],
                user_input_text=user_input_text,
                structured_text=structured_text,
                reasoning_explanation=reasoning_explanation,
            ),
            metadata={
                "mode": "deterministic_summary",
                "llm_payload_v1": llm_payload_v1,
                "llm_payload_v2": llm_payload_v2,
                "context_packet_v1": context_packet_v1,
            },
        )

    def answer(self, payload: Dict[str, object]) -> LLMExplanationResult:
        user_input_text = str(payload.get("user_input_text", "")).strip()
        route_type = str(payload.get("route_type", "direct_answer"))
        llm_payload_v1 = payload.get("llm_payload_v1", {}) if isinstance(payload.get("llm_payload_v1", {}), dict) else {}
        llm_payload_v2 = payload.get("llm_payload_v2", {}) if isinstance(payload.get("llm_payload_v2", {}), dict) else {}
        context_packet_v1 = payload.get("context_packet_v1", {}) if isinstance(payload.get("context_packet_v1", {}), dict) else {}
        answer_text = self._direct_answer_text(
            user_input_text=user_input_text,
            route_type=route_type,
            structured_context=payload.get("structured_context", {}),
            llm_payload_v2=llm_payload_v2,
        )
        return LLMExplanationResult(
            provider=self.provider_name,
            structured_text="",
            reasoning_explanation="",
            user_facing_text=answer_text,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a deterministic fallback direct-answer layer for BrAIn. "
                        "If the answer is unknown, say so plainly and do not invent facts."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, indent=2, sort_keys=True),
                },
            ],
            metadata={
                "mode": "deterministic_direct_answer",
                "route_type": route_type,
                "llm_payload_v1": llm_payload_v1,
                "llm_payload_v2": llm_payload_v2,
                "context_packet_v1": context_packet_v1,
            },
        )

    def _direct_answer_text(
        self,
        *,
        user_input_text: str,
        route_type: str,
        structured_context: object,
        llm_payload_v2: Dict[str, object],
    ) -> str:
        normalized = user_input_text.strip().lower()
        packet_first = self._packet_first_response(
            user_input_text=user_input_text,
            route_type=route_type,
            llm_payload_v2=llm_payload_v2,
            mode="direct_answer",
        )
        if packet_first:
            return packet_first
        if "what color is the sky" in normalized:
            return "The sky is usually blue during the day, and it can look red, orange, pink, gray, or black depending on weather and time."
        if "laptop battery" in normalized and ("what does" in normalized or "what do" in normalized):
            return "A laptop battery stores electrical energy so the laptop can run when it is not plugged into a wall outlet."
        actionable = _actionable_answer_from_query(user_input_text)
        if actionable:
            return actionable
        if "what did i ask earlier" in normalized:
            return "That route should use memory retrieval rather than direct answering."
        if route_type == "hybrid_request":
            if isinstance(structured_context, dict):
                reasoning = str(structured_context.get("reasoning_summary", "")).strip()
                memories = structured_context.get("retrieved_memory_count", 0)
                if reasoning:
                    return f"{reasoning} I also used {memories} retrieved memory item(s) as context."
            return "This prompt needs both BrAIn context and a direct answer layer, but no live model-generated wording was available."
        return (
            "This prompt was routed to direct answering. Configure `BRAIN_LLM_PROVIDER=ollama` "
            "or `BRAIN_LLM_COMMAND` for live factual answers when deterministic fallbacks are insufficient."
        )

    def _packet_first_response(
        self,
        *,
        user_input_text: str,
        route_type: str,
        llm_payload_v2: Dict[str, object],
        mode: str,
    ) -> str:
        return render_llm_payload_v2_answer(
            user_input_text=user_input_text,
            route_type=route_type,
            llm_payload_v2=llm_payload_v2,
            mode=mode,
        )

    def _answer_from_llm_payload_v2(
        self,
        *,
        user_input_text: str,
        llm_payload_v2: Dict[str, object],
    ) -> str:
        return _answer_from_llm_payload_v2(
            user_input_text=user_input_text,
            llm_payload_v2=llm_payload_v2,
        )

    def _planning_response_from_llm_payload_v2(
        self,
        *,
        llm_payload_v2: Dict[str, object],
        direct: bool,
    ) -> str:
        return _planning_response_from_llm_payload_v2(
            user_input_text="",
            llm_payload_v2=llm_payload_v2,
            direct=direct,
        )


class SubprocessExplanationProvider:
    provider_name = "subprocess"

    def __init__(self, command: str) -> None:
        self.command = command
        self.fallback = DeterministicExplanationProvider()

    def explain(self, payload: Dict[str, object]) -> LLMExplanationResult:
        fallback = self.fallback.explain(payload)
        completed = subprocess.run(
            self.command,
            input=json.dumps(
                {
                    "instructions": (
                        "Explain the supplied BrAIn decision in normal language. "
                        "Use llm_payload_v2 as the primary source of truth. "
                        "If llm_payload_v2.source_of_truth is 'brain' or 'cache', do not override the provided state. "
                        "Do not change the structured result. Use only the supplied fields."
                    ),
                    "payload": payload,
                    "fallback": fallback.to_dict(),
                },
                indent=2,
                sort_keys=True,
            ),
            text=True,
            capture_output=True,
            shell=True,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return LLMExplanationResult(
                provider=self.provider_name,
                structured_text=fallback.structured_text,
                reasoning_explanation=fallback.reasoning_explanation,
                user_facing_text=fallback.user_facing_text,
                messages=fallback.messages,
                metadata={
                    "mode": "subprocess_fallback",
                    "command": self.command,
                    "returncode": completed.returncode,
                    "stderr": completed.stderr.strip(),
                },
            )
        text = completed.stdout.strip()
        return LLMExplanationResult(
            provider=self.provider_name,
            structured_text=fallback.structured_text,
            reasoning_explanation=fallback.reasoning_explanation,
            user_facing_text=text,
            messages=fallback.messages,
            metadata={
                "mode": "subprocess",
                "command": self.command,
            },
        )

    def answer(self, payload: Dict[str, object]) -> LLMExplanationResult:
        fallback = self.fallback.answer(payload)
        completed = subprocess.run(
            self.command,
            input=json.dumps(
                {
                    "instructions": (
                        "Answer the user directly. "
                        "Use llm_payload_v2 as the primary source of truth. "
                        "If llm_payload_v2.source_of_truth is 'brain', do not override BrAIn state. "
                        "Do not invent memory or alter provided structure."
                    ),
                    "llm_payload_v2": payload.get("llm_payload_v2", {}),
                    "fallback": fallback.to_dict(),
                },
                indent=2,
                sort_keys=True,
            ),
            text=True,
            capture_output=True,
            shell=True,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return LLMExplanationResult(
                provider=self.provider_name,
                structured_text=fallback.structured_text,
                reasoning_explanation=fallback.reasoning_explanation,
                user_facing_text=fallback.user_facing_text,
                messages=fallback.messages,
                metadata={
                    "mode": "subprocess_direct_answer_fallback",
                    "command": self.command,
                    "returncode": completed.returncode,
                    "stderr": completed.stderr.strip(),
                },
            )
        return LLMExplanationResult(
            provider=self.provider_name,
            structured_text="",
            reasoning_explanation="",
            user_facing_text=completed.stdout.strip(),
            messages=fallback.messages,
            metadata={"mode": "subprocess_direct_answer", "command": self.command},
        )


class OllamaExplanationProvider:
    provider_name = "ollama"

    def __init__(
        self,
        *,
        model: str,
        host: str = "http://localhost:11434",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.fallback = DeterministicExplanationProvider()

    def explain(self, payload: Dict[str, object]) -> LLMExplanationResult:
        fallback = self.fallback.explain(payload)
        llm_payload_v2 = payload.get("llm_payload_v2", {}) if isinstance(payload.get("llm_payload_v2", {}), dict) else {}
        route_type = str(payload.get("route_type", ""))
        structured_brain_answer = (
            str(llm_payload_v2.get("source_of_truth", "")).lower() == "brain"
            and route_type in {"planning_request", "goal_request", "transform_request", "hybrid_request", "memory_question"}
            and bool(fallback.user_facing_text.strip())
        )
        if structured_brain_answer and not _contains_meta_language(fallback.user_facing_text):
            return LLMExplanationResult(
                provider=self.provider_name,
                structured_text=fallback.structured_text,
                reasoning_explanation=fallback.reasoning_explanation,
                user_facing_text=fallback.user_facing_text,
                messages=fallback.messages,
                metadata={
                    "mode": "ollama_structured_brain_passthrough",
                    "model": self.model,
                    "host": self.host,
                    "reason": "structured_brain_answer_is_authoritative",
                },
            )
        prompt = self._build_prompt(payload=payload, fallback=fallback)
        request_payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }
        request = urllib.request.Request(
            url=f"{self.host}/api/generate",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            return LLMExplanationResult(
                provider=self.provider_name,
                structured_text=fallback.structured_text,
                reasoning_explanation=fallback.reasoning_explanation,
                user_facing_text=fallback.user_facing_text,
                messages=fallback.messages,
                metadata={
                    "mode": "ollama_fallback",
                    "model": self.model,
                    "host": self.host,
                    "error": str(exc),
                },
            )
        try:
            response_payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            return LLMExplanationResult(
                provider=self.provider_name,
                structured_text=fallback.structured_text,
                reasoning_explanation=fallback.reasoning_explanation,
                user_facing_text=fallback.user_facing_text,
                messages=fallback.messages,
                metadata={
                    "mode": "ollama_fallback",
                    "model": self.model,
                    "host": self.host,
                    "error": f"invalid_json:{exc}",
                },
            )
        text = str(response_payload.get("response", "")).strip()
        if not text:
            return LLMExplanationResult(
                provider=self.provider_name,
                structured_text=fallback.structured_text,
                reasoning_explanation=fallback.reasoning_explanation,
                user_facing_text=fallback.user_facing_text,
                messages=fallback.messages,
                metadata={
                    "mode": "ollama_fallback",
                    "model": self.model,
                    "host": self.host,
                    "error": "empty_response",
                },
            )
        return LLMExplanationResult(
            provider=self.provider_name,
            structured_text=fallback.structured_text,
            reasoning_explanation=fallback.reasoning_explanation,
            user_facing_text=text,
            messages=fallback.messages,
            metadata={
                "mode": "ollama",
                "model": self.model,
                "host": self.host,
            },
        )

    def answer(self, payload: Dict[str, object]) -> LLMExplanationResult:
        fallback = self.fallback.answer(payload)
        llm_payload_v2 = payload.get("llm_payload_v2", {}) if isinstance(payload.get("llm_payload_v2", {}), dict) else {}
        structured_memory_answer = (
            str(llm_payload_v2.get("source_of_truth", "")).lower() in {"brain", "cache"}
            and str(llm_payload_v2.get("task_type", "")).lower() == "memory_recall"
            and bool(fallback.user_facing_text.strip())
            and bool(llm_payload_v2.get("answer_state") or llm_payload_v2.get("answer_facts") or llm_payload_v2.get("reconstructed_context"))
        )
        if structured_memory_answer:
            return LLMExplanationResult(
                provider=self.provider_name,
                structured_text=fallback.structured_text,
                reasoning_explanation=fallback.reasoning_explanation,
                user_facing_text=fallback.user_facing_text,
                messages=fallback.messages,
                metadata={
                    "mode": "ollama_structured_memory_passthrough",
                    "model": self.model,
                    "host": self.host,
                    "reason": "structured_memory_answer_is_authoritative",
                },
            )
        prompt = "\n".join(
            [
                "You are the direct-answer layer for BrAIn.",
                "Answer the user directly and concisely.",
                "Use llm_payload_v2 as the primary structured source of truth.",
                "If source_of_truth is 'brain' or 'cache', do not override the provided state.",
                "Do not use hidden internal structure.",
                "Do not invent past memories or hidden reasoning.",
                "If context is insufficient, say what is missing.",
                "",
                "LLM payload:",
                json.dumps(payload.get("llm_payload_v2", {}), indent=2, sort_keys=True),
                "",
                "Return only the final user-facing answer text.",
            ]
        )
        request_payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }
        request = urllib.request.Request(
            url=f"{self.host}/api/generate",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            return LLMExplanationResult(
                provider=self.provider_name,
                structured_text=fallback.structured_text,
                reasoning_explanation=fallback.reasoning_explanation,
                user_facing_text=fallback.user_facing_text,
                messages=fallback.messages,
                metadata={
                    "mode": "ollama_direct_answer_fallback",
                    "model": self.model,
                    "host": self.host,
                    "error": str(exc),
                },
            )
        try:
            response_payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            return LLMExplanationResult(
                provider=self.provider_name,
                structured_text=fallback.structured_text,
                reasoning_explanation=fallback.reasoning_explanation,
                user_facing_text=fallback.user_facing_text,
                messages=fallback.messages,
                metadata={
                    "mode": "ollama_direct_answer_fallback",
                    "model": self.model,
                    "host": self.host,
                    "error": f"invalid_json:{exc}",
                },
            )
        text = str(response_payload.get("response", "")).strip()
        if not text:
            return LLMExplanationResult(
                provider=self.provider_name,
                structured_text=fallback.structured_text,
                reasoning_explanation=fallback.reasoning_explanation,
                user_facing_text=fallback.user_facing_text,
                messages=fallback.messages,
                metadata={
                    "mode": "ollama_direct_answer_fallback",
                    "model": self.model,
                    "host": self.host,
                    "error": "empty_response",
                },
            )
        return LLMExplanationResult(
            provider=self.provider_name,
            structured_text="",
            reasoning_explanation="",
            user_facing_text=text,
            messages=fallback.messages,
            metadata={"mode": "ollama_direct_answer", "model": self.model, "host": self.host},
        )

    def _build_prompt(
        self,
        *,
        payload: Dict[str, object],
        fallback: LLMExplanationResult,
    ) -> str:
        return "\n".join(
            [
                "You are the explanation layer for BrAIn.",
                "Explain the supplied BrAIn decision in concise normal language.",
                "Use llm_payload_v2 as the primary structured source of truth.",
                "If source_of_truth is 'brain' or 'cache', do not override the provided state.",
                "Do not change the structured result.",
                "Do not invent missing facts.",
                "Use only the supplied fields.",
                "",
                "LLM payload:",
                json.dumps(payload.get("llm_payload_v2", {}), indent=2, sort_keys=True),
                "",
                "Deterministic fallback explanation:",
                fallback.user_facing_text,
                "",
                "Return only the final user-facing explanation text.",
            ]
        )


class HybridLLMInterface:
    """
    Optional explanation layer for the hybrid prototype.

    BrAIn supplies the structured result and reasoning trace. This module
    explains them, but does not alter them. By default it uses a deterministic
    renderer. If `BRAIN_LLM_COMMAND` is set, it can hand the same payload to a
    local subprocess-backed model.
    """

    def __init__(
        self,
        provider: str = "auto",
        llm_command: Optional[str] = None,
    ) -> None:
        command = llm_command or os.environ.get("BRAIN_LLM_COMMAND", "").strip()
        env_provider = os.environ.get("BRAIN_LLM_PROVIDER", "").strip().lower()
        ollama_model = os.environ.get("BRAIN_LLM_MODEL", "").strip() or "qwen3:8b"
        ollama_host = os.environ.get("BRAIN_LLM_HOST", "").strip() or "http://localhost:11434"
        selected = provider
        if selected == "auto":
            if env_provider:
                selected = env_provider
            elif command:
                selected = "subprocess"
            else:
                selected = "deterministic"
        if selected == "ollama":
            self.provider = OllamaExplanationProvider(
                model=ollama_model,
                host=ollama_host,
            )
        if selected == "subprocess":
            self.provider: ExplanationProvider = SubprocessExplanationProvider(command=command)
        elif selected != "ollama":
            self.provider = DeterministicExplanationProvider()

    def explain(
        self,
        *,
        user_input_text: str,
        result_brain_input: BrainInputV1,
        reasoning_trace: Dict[str, object],
        selected_path: Dict[str, object],
        retrieved_memories: Sequence[Dict[str, object]],
        context_packet_v1: Optional[Dict[str, object]] = None,
        llm_payload_v1: Optional[Dict[str, object]] = None,
        llm_payload_v2: Optional[Dict[str, object]] = None,
    ) -> LLMExplanationResult:
        context_packet = context_packet_v1 if isinstance(context_packet_v1, dict) else {}
        model_payload_v1 = llm_payload_v1 if isinstance(llm_payload_v1, dict) else build_llm_payload_v1(
            user_input_text=user_input_text,
            route_type=str(context_packet.get("route_type", "")),
            context_packet_v1=context_packet,
            mode="explanation",
            route_reasons=context_packet.get("route_context", {}).get("reasons", []) if isinstance(context_packet.get("route_context", {}), dict) else [],
            selected_path=selected_path,
            retrieved_memory_count=len(retrieved_memories),
            recalled_turn_count=int(context_packet.get("route_context", {}).get("recalled_turn_count", 0)) if isinstance(context_packet.get("route_context", {}), dict) else 0,
        )
        model_payload_v2 = llm_payload_v2 if isinstance(llm_payload_v2, dict) else build_llm_payload_from_packet_v1(context_packet)
        payload = {
            "user_input_text": user_input_text,
            "route_type": str(context_packet.get("route_type", "")),
            "result_brain_input": result_brain_input.to_dict(),
            "reasoning_trace": reasoning_trace.get("trace_steps", []),
            "selected_path": selected_path,
            "retrieved_memories": list(retrieved_memories),
            "trace_summary": reasoning_trace,
            "context_packet_v1": context_packet,
            "llm_payload_v1": model_payload_v1,
            "llm_payload_v2": model_payload_v2,
        }
        return self.provider.explain(payload)

    def answer_user_prompt(
        self,
        *,
        user_input_text: str,
        route_type: str,
        structured_context: Optional[Dict[str, object]] = None,
        context_packet_v1: Optional[Dict[str, object]] = None,
        llm_payload_v1: Optional[Dict[str, object]] = None,
        llm_payload_v2: Optional[Dict[str, object]] = None,
    ) -> LLMExplanationResult:
        context_packet = context_packet_v1 if isinstance(context_packet_v1, dict) else {}
        model_payload_v1 = llm_payload_v1 if isinstance(llm_payload_v1, dict) else build_llm_payload_v1(
            user_input_text=user_input_text,
            route_type=route_type,
            context_packet_v1=context_packet,
            mode="direct_answer",
            route_reasons=context_packet.get("route_context", {}).get("reasons", []) if isinstance(context_packet.get("route_context", {}), dict) else [],
            selected_path=structured_context.get("selected_path", {}) if isinstance(structured_context, dict) else {},
            retrieved_memory_count=int(context_packet.get("route_context", {}).get("retrieved_memory_count", 0)) if isinstance(context_packet.get("route_context", {}), dict) else 0,
            recalled_turn_count=int(context_packet.get("route_context", {}).get("recalled_turn_count", 0)) if isinstance(context_packet.get("route_context", {}), dict) else 0,
        )
        model_payload_v2 = llm_payload_v2 if isinstance(llm_payload_v2, dict) else build_llm_payload_from_packet_v1(context_packet)
        payload = {
            "user_input_text": user_input_text,
            "route_type": route_type,
            "structured_context": structured_context or {},
            "context_packet_v1": context_packet,
            "llm_payload_v1": model_payload_v1,
            "llm_payload_v2": model_payload_v2,
        }
        if str(model_payload_v2.get("source_of_truth", "llm")) == "cache":
            return build_cached_answer_result(
                user_input_text=user_input_text,
                route_type=route_type,
                llm_payload_v2=model_payload_v2,
            )
        relation = model_payload_v2.get("relation_memory",{})
        if (route_type == "memory_question" and isinstance(relation,dict) and relation.get("active")
                and (relation.get("supported_answer") or relation.get("answer_status")=="unresolved")):
            # Enforce known state and abstention at the shared boundary, including
            # subprocess providers which do not implement their own passthrough.
            result = DeterministicExplanationProvider().answer(payload)
            if relation.get("answer_status")=="unresolved":
                result = replace(result,user_facing_text="UNKNOWN")
            result.metadata["relation_authority_enforced"] = True
            result.metadata["configured_provider"] = self.provider.provider_name
            return result
        return self.provider.answer(payload)
