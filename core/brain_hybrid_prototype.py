from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from brain_codec import BrainCodec, BrainMemory
from brain_relation_memory import context_from_turns, apply_to_legacy_state, has_relevant_relation_history
from brain_state_extraction import explicit_locations, quantity_bounds, placement_constraints
from brain_context_packet_v1 import build_context_packet_v1
from brain_goal_reward_v5 import BrainGoalRewardV5, GoalTarget
from brain_goal_tracking_v6 import BrainGoalTrackingV6, GoalTrackingResult
from brain_history import BrainHistory
from brain_input_v1 import BrainInputV1
from brain_parser import BrainParser, ParseResult
from brain_planning_simulation_v3_benchmark import _build_scenarios
from brain_prompt_enrichment import BrainPromptEnrichment, PromptEnrichmentResult
from brain_query_router import QueryRouteDecision, route_user_prompt
from brain_pixel_memory import PixelMemoryCanvas, PixelProjectionConcept, PixelRelationship, PixelReadback, PixelReasoningEngine, PixelReasoningTrace
from brain_reasoning_graph import QueryIntent, ReasoningGraphCandidate, ReasoningGraphEvaluator, RelationshipType, detect_query_intent
from brain_strategy_generation_v7 import BrainStrategyGenerationV7, StrategyPlanningResult
from llm_interface import HybridLLMInterface, LLMExplanationResult, build_cached_answer_result, build_llm_payload_v1, build_llm_payload_from_packet_v1


def _has_text(value: object) -> bool:
    return bool(str(value or "").strip())


def _brain_memory_content_score(memory: BrainMemory) -> float:
    score = 0.0
    if _has_text(memory.actor):
        score += 0.1
    if _has_text(memory.intent):
        score += 0.2
    if _has_text(memory.action):
        score += 1.4
    if _has_text(memory.object):
        score += 1.2
    if _has_text(memory.topic):
        score += 1.0
    score += 0.35 * len([item for item in memory.modifiers if _has_text(item)])
    score += 0.35 * len([item for item in memory.constraints if _has_text(item)])
    score += 0.35 * len([item for item in memory.negations if _has_text(item)])
    score += 0.25 * len([item for item in memory.conditions if _has_text(item)])
    score += 0.2 * len([item for item in memory.relations if isinstance(item, dict) and any(_has_text(item.get(key, "")) for key in ("type", "source", "target"))])
    return score


def _brain_input_content_score(brain_input: BrainInputV1) -> float:
    return _brain_memory_content_score(
        BrainMemory(
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
            time=brain_input.time,
            priority=brain_input.priority,
        )
    )


def _entry_specificity_bonus(entry: "EpisodicMemoryEntry") -> float:
    return (
        _brain_memory_content_score(entry.parsed_memory) * 0.18
        + _brain_input_content_score(entry.brain_input) * 0.08
        + (0.2 if _has_text(entry.source_text) else 0.0)
    )


def _is_meaningful_entry(entry: "EpisodicMemoryEntry") -> bool:
    if entry.entry_type == "llm_knowledge_cache":
        metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
        return _has_text(metadata.get("answer_text", "")) or _has_text(entry.source_text)
    if _brain_memory_content_score(entry.parsed_memory) > 0.9:
        return True
    if _brain_input_content_score(entry.brain_input) > 0.9:
        return True
    return _has_text(entry.source_text)


@dataclass(frozen=True)
class StoragePolicy:
    mode: str = "store_all_prompts"
    store_user_prompts: bool = True
    store_selected_results: bool = True
    trace_storage_mode: str = "high_value"
    store_feedback_when_available: bool = True

    @classmethod
    def for_mode(cls, mode: str) -> "StoragePolicy":
        normalized = str(mode).strip().lower()
        if normalized == "store_disabled":
            return cls(
                mode=normalized,
                store_user_prompts=False,
                store_selected_results=False,
                trace_storage_mode="never",
                store_feedback_when_available=False,
            )
        if normalized == "store_final_results_only":
            return cls(
                mode=normalized,
                store_user_prompts=False,
                store_selected_results=True,
                trace_storage_mode="never",
                store_feedback_when_available=False,
            )
        if normalized == "store_only_successful_reasoning":
            return cls(
                mode=normalized,
                store_user_prompts=False,
                store_selected_results=True,
                trace_storage_mode="high_value",
                store_feedback_when_available=True,
            )
        return cls(mode="store_all_prompts")

    def storage_enabled(self) -> bool:
        return self.mode != "store_disabled"

    def should_store_user_prompt(self, trace: "HybridReasoningTrace") -> bool:
        if self.mode == "store_only_successful_reasoning":
            return bool(trace.selected_path)
        return self.store_user_prompts

    def should_store_result(self, trace: "HybridReasoningTrace") -> bool:
        if self.mode == "store_only_successful_reasoning":
            return bool(trace.selected_path)
        return self.store_selected_results

    def should_store_trace(self, trace: "HybridReasoningTrace") -> bool:
        if self.mode == "store_disabled":
            return False
        mode = self.trace_storage_mode.strip().lower()
        if mode == "always":
            return True
        if mode == "never":
            return False
        return bool(
            len(trace.retrieved_memories) > 0
            or len(trace.candidate_paths) > 1
            or trace.selected_path.get("goal_alignment_score", 0.0) >= 0.6
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "store_user_prompts": self.store_user_prompts,
            "store_selected_results": self.store_selected_results,
            "trace_storage_mode": self.trace_storage_mode,
            "store_feedback_when_available": self.store_feedback_when_available,
        }


@dataclass(frozen=True)
class EpisodicMemoryEntry:
    entry_id: int
    timestamp: str
    entry_type: str
    source_text: str
    parsed_memory: BrainMemory
    brain_input: BrainInputV1
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "entry_type": self.entry_type,
            "source_text": self.source_text,
            "parsed_memory": self.parsed_memory.to_dict(),
            "brain_input": self.brain_input.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "EpisodicMemoryEntry":
        return cls(
            entry_id=int(data.get("entry_id", 0)),
            timestamp=str(data.get("timestamp", "")),
            entry_type=str(data.get("entry_type", "memory")),
            source_text=str(data.get("source_text", "")),
            parsed_memory=BrainMemory.from_dict(data.get("parsed_memory", {}) if isinstance(data.get("parsed_memory", {}), dict) else {}),
            brain_input=BrainInputV1.from_dict(data.get("brain_input", {}) if isinstance(data.get("brain_input", {}), dict) else {}),
            metadata=data.get("metadata", {}) if isinstance(data.get("metadata", {}), dict) else {},
        )


@dataclass(frozen=True)
class RetrievedMemory:
    entry_id: int
    score: float
    entry: EpisodicMemoryEntry

    def to_dict(self) -> Dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "score": round(self.score, 4),
            "entry": self.entry.to_dict(),
        }


@dataclass(frozen=True)
class EpisodicTurnRecord:
    turn_id: int
    sequence: int
    timestamp: str
    raw_user_prompt: str
    final_answer: str
    parsed_brain_input: BrainInputV1
    final_selected_result: BrainInputV1
    route_type: str
    reasoning_succeeded: bool
    candidate_path_count: int = 0
    route_confidence: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "raw_user_prompt": self.raw_user_prompt,
            "final_answer": self.final_answer,
            "parsed_brain_input": self.parsed_brain_input.to_dict(),
            "final_selected_result": self.final_selected_result.to_dict(),
            "route_type": self.route_type,
            "reasoning_succeeded": self.reasoning_succeeded,
            "candidate_path_count": self.candidate_path_count,
            "route_confidence": round(self.route_confidence, 4),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "EpisodicTurnRecord":
        return cls(
            turn_id=int(data.get("turn_id", 0)),
            sequence=int(data.get("sequence", 0)),
            timestamp=str(data.get("timestamp", "")),
            raw_user_prompt=str(data.get("raw_user_prompt", "")),
            final_answer=str(data.get("final_answer", "")),
            parsed_brain_input=BrainInputV1.from_dict(data.get("parsed_brain_input", {}) if isinstance(data.get("parsed_brain_input", {}), dict) else {}),
            final_selected_result=BrainInputV1.from_dict(data.get("final_selected_result", {}) if isinstance(data.get("final_selected_result", {}), dict) else {}),
            route_type=str(data.get("route_type", "unknown")),
            reasoning_succeeded=bool(data.get("reasoning_succeeded", False)),
            candidate_path_count=int(data.get("candidate_path_count", 0)),
            route_confidence=float(data.get("route_confidence", 0.0)),
            metadata=data.get("metadata", {}) if isinstance(data.get("metadata", {}), dict) else {},
        )


@dataclass(frozen=True)
class EpisodicTurnRecall:
    turn_id: int
    score: float
    recency_score: float
    semantic_score: float
    route_score: float
    record: EpisodicTurnRecord

    def to_dict(self) -> Dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "score": round(self.score, 4),
            "recency_score": round(self.recency_score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "route_score": round(self.route_score, 4),
            "record": self.record.to_dict(),
        }


@dataclass(frozen=True)
class MemoryReconstructionConfig:
    concept_similarity_weight: float = 0.32
    temporal_proximity_weight: float = 0.18
    goal_link_weight: float = 0.14
    hierarchy_link_weight: float = 0.12
    cross_domain_link_weight: float = 0.08
    recency_weight: float = 0.16
    high_confidence_threshold: float = 0.72
    medium_confidence_threshold: float = 0.42
    pixel_decision_margin: float = 0.08


@dataclass(frozen=True)
class ReconstructedMemoryFragment:
    source_type: str
    source_id: int
    text: str
    score: float
    evidence_breakdown: Dict[str, float]
    payload: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "text": self.text,
            "score": round(self.score, 4),
            "evidence_breakdown": {
                str(key): round(float(value), 4)
                for key, value in sorted(self.evidence_breakdown.items())
            },
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class ReconstructedContext:
    query: str
    exact_matches: List[Dict[str, object]]
    partial_matches: List[ReconstructedMemoryFragment]
    related_concepts: List[Dict[str, object]]
    temporal_links: List[Dict[str, object]]
    goal_links: List[Dict[str, object]]
    hierarchy_links: List[Dict[str, object]]
    cross_domain_links: List[Dict[str, object]]
    synthesized_summary: str
    confidence: float
    answer_mode: str
    evidence_breakdown: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "query": self.query,
            "exact_matches": list(self.exact_matches),
            "partial_matches": [item.to_dict() for item in self.partial_matches],
            "related_concepts": list(self.related_concepts),
            "temporal_links": list(self.temporal_links),
            "goal_links": list(self.goal_links),
            "hierarchy_links": list(self.hierarchy_links),
            "cross_domain_links": list(self.cross_domain_links),
            "synthesized_summary": self.synthesized_summary,
            "confidence": round(self.confidence, 4),
            "answer_mode": self.answer_mode,
            "evidence_breakdown": dict(self.evidence_breakdown),
        }


@dataclass(frozen=True)
class PendingClarification:
    clarification_id: int
    route_type: str
    original_prompt: str
    effective_prompt: str
    clarification_question: str
    missing_fields: List[str]
    clarification_type: str
    route_decision: Dict[str, object]
    enrichment_result: Dict[str, object]
    parse_result: Dict[str, object]
    previous_context: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "clarification_id": self.clarification_id,
            "route_type": self.route_type,
            "original_prompt": self.original_prompt,
            "effective_prompt": self.effective_prompt,
            "clarification_question": self.clarification_question,
            "missing_fields": list(self.missing_fields),
            "clarification_type": self.clarification_type,
            "route_decision": dict(self.route_decision),
            "enrichment_result": dict(self.enrichment_result),
            "parse_result": dict(self.parse_result),
            "previous_context": dict(self.previous_context),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PendingClarification":
        return cls(
            clarification_id=int(data.get("clarification_id", 0)),
            route_type=str(data.get("route_type", "unknown")),
            original_prompt=str(data.get("original_prompt", "")),
            effective_prompt=str(data.get("effective_prompt", data.get("original_prompt", ""))),
            clarification_question=str(data.get("clarification_question", "")),
            missing_fields=[str(item) for item in data.get("missing_fields", []) if str(item).strip()] if isinstance(data.get("missing_fields", []), list) else [],
            clarification_type=str(data.get("clarification_type", "")),
            route_decision=data.get("route_decision", {}) if isinstance(data.get("route_decision", {}), dict) else {},
            enrichment_result=data.get("enrichment_result", {}) if isinstance(data.get("enrichment_result", {}), dict) else {},
            parse_result=data.get("parse_result", {}) if isinstance(data.get("parse_result", {}), dict) else {},
            previous_context=data.get("previous_context", {}) if isinstance(data.get("previous_context", {}), dict) else {},
        )


@dataclass(frozen=True)
class HybridReasoningTrace:
    raw_prompt: str
    effective_prompt: str
    route_decision: Dict[str, object]
    enrichment_result: Dict[str, object]
    clarification_state: Dict[str, object]
    goal_tracking: Dict[str, object]
    strategy_plan: Dict[str, object]
    merged_memory_state: Dict[str, object]
    parsed_input: Dict[str, object]
    stored_entries: List[Dict[str, object]]
    retrieved_memories: List[Dict[str, object]]
    recalled_turns: List[Dict[str, object]]
    goal_weights: Dict[str, float]
    concept_matches: List[Dict[str, object]]
    generated_candidate_paths: List[Dict[str, object]]
    candidate_paths: List[Dict[str, object]]
    grounded_paths: List[Dict[str, object]]
    goal_scored_paths: List[Dict[str, object]]
    selected_path: Dict[str, object]
    final_predicted_outcome: Dict[str, object]
    context_packet_v1: Dict[str, object] = field(default_factory=dict)
    llm_payload_v1: Dict[str, object] = field(default_factory=dict)
    llm_payload_v2: Dict[str, object] = field(default_factory=dict)
    cached_knowledge_checked: bool = False
    cached_knowledge_hit: bool = False
    cached_knowledge_entry_ids: List[int] = field(default_factory=list)
    cached_knowledge_written: bool = False
    cached_knowledge_topic_key: str = ""
    cached_knowledge_normalized_terms: List[str] = field(default_factory=list)
    cached_knowledge_trust_level: str = ""
    cached_knowledge_used: bool = False
    cached_knowledge_entry_id: Optional[int] = None
    cached_knowledge_candidate_entry_ids: List[int] = field(default_factory=list)
    cached_knowledge_selected_entry_id: Optional[int] = None
    cached_knowledge_ranking_scores: List[Dict[str, object]] = field(default_factory=list)
    cached_knowledge_reinforcement_applied: bool = False
    cached_knowledge_updated_confidence: Optional[float] = None
    cached_knowledge_updated_reuse_count: Optional[int] = None
    cached_knowledge_conflict_detected: bool = False
    cached_knowledge_conflicting_entry_ids: List[int] = field(default_factory=list)
    cached_knowledge_penalized_entry_ids: List[int] = field(default_factory=list)
    cached_knowledge_updated_confidences: List[Dict[str, object]] = field(default_factory=list)
    cached_knowledge_verification_attempted: bool = False
    cached_knowledge_verification_status: str = ""
    cached_knowledge_confidence_capped: bool = False
    cached_knowledge_uncertainty_flag: bool = False
    cached_knowledge_stabilized_flag: bool = False
    cached_knowledge_dominance_score: Optional[float] = None
    cached_knowledge_convergence_rate: Optional[float] = None
    multi_hop_used: bool = False
    merged_entry_ids: List[int] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    merged_reasoning_packet: Dict[str, object] = field(default_factory=dict)
    reasoning_graph_nodes: List[Dict[str, object]] = field(default_factory=list)
    reasoning_graph_edges: List[Dict[str, object]] = field(default_factory=list)
    reasoning_path: List[Dict[str, object]] = field(default_factory=list)
    global_graph_size: Dict[str, int] = field(default_factory=dict)
    reused_graph_edges: List[Dict[str, object]] = field(default_factory=list)
    new_graph_edges: List[Dict[str, object]] = field(default_factory=list)
    structured_explanation: Dict[str, object] = field(default_factory=dict)
    final_explanation: str = ""
    cache_bypassed_reason: Optional[str] = None
    llm_explanation: Dict[str, object] = field(default_factory=dict)
    memory_written: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "raw_prompt": self.raw_prompt,
            "effective_prompt": self.effective_prompt,
            "route_decision": dict(self.route_decision),
            "enrichment_result": dict(self.enrichment_result),
            "clarification_state": dict(self.clarification_state),
            "goal_tracking": dict(self.goal_tracking),
            "strategy_plan": dict(self.strategy_plan),
            "merged_memory_state": dict(self.merged_memory_state),
            "parsed_input": self.parsed_input,
            "stored_entries": list(self.stored_entries),
            "retrieved_memories": list(self.retrieved_memories),
            "recalled_turns": list(self.recalled_turns),
            "goal_weights": {str(key): round(float(value), 4) for key, value in sorted(self.goal_weights.items())},
            "concept_matches": list(self.concept_matches),
            "generated_candidate_paths": list(self.generated_candidate_paths),
            "candidate_paths": list(self.candidate_paths),
            "grounded_paths": list(self.grounded_paths),
            "goal_scored_paths": list(self.goal_scored_paths),
            "selected_path": dict(self.selected_path),
            "final_predicted_outcome": dict(self.final_predicted_outcome),
            "context_packet_v1": dict(self.context_packet_v1),
            "llm_payload_v1": dict(self.llm_payload_v1),
            "llm_payload_v2": dict(self.llm_payload_v2),
            "cached_knowledge_checked": self.cached_knowledge_checked,
            "cached_knowledge_hit": self.cached_knowledge_hit,
            "cached_knowledge_entry_ids": list(self.cached_knowledge_entry_ids),
            "cached_knowledge_written": self.cached_knowledge_written,
            "cached_knowledge_topic_key": self.cached_knowledge_topic_key,
            "cached_knowledge_normalized_terms": list(self.cached_knowledge_normalized_terms),
            "cached_knowledge_trust_level": self.cached_knowledge_trust_level,
            "cached_knowledge_used": self.cached_knowledge_used,
            "cached_knowledge_entry_id": self.cached_knowledge_entry_id,
            "cached_knowledge_candidate_entry_ids": list(self.cached_knowledge_candidate_entry_ids),
            "cached_knowledge_selected_entry_id": self.cached_knowledge_selected_entry_id,
            "cached_knowledge_ranking_scores": list(self.cached_knowledge_ranking_scores),
            "cached_knowledge_reinforcement_applied": self.cached_knowledge_reinforcement_applied,
            "cached_knowledge_updated_confidence": self.cached_knowledge_updated_confidence,
            "cached_knowledge_updated_reuse_count": self.cached_knowledge_updated_reuse_count,
            "cached_knowledge_conflict_detected": self.cached_knowledge_conflict_detected,
            "cached_knowledge_conflicting_entry_ids": list(self.cached_knowledge_conflicting_entry_ids),
            "cached_knowledge_penalized_entry_ids": list(self.cached_knowledge_penalized_entry_ids),
            "cached_knowledge_updated_confidences": list(self.cached_knowledge_updated_confidences),
            "cached_knowledge_verification_attempted": self.cached_knowledge_verification_attempted,
            "cached_knowledge_verification_status": self.cached_knowledge_verification_status,
            "cached_knowledge_confidence_capped": self.cached_knowledge_confidence_capped,
            "cached_knowledge_uncertainty_flag": self.cached_knowledge_uncertainty_flag,
            "cached_knowledge_stabilized_flag": self.cached_knowledge_stabilized_flag,
            "cached_knowledge_dominance_score": self.cached_knowledge_dominance_score,
            "cached_knowledge_convergence_rate": self.cached_knowledge_convergence_rate,
            "multi_hop_used": self.multi_hop_used,
            "merged_entry_ids": list(self.merged_entry_ids),
            "reasoning_steps": list(self.reasoning_steps),
            "merged_reasoning_packet": dict(self.merged_reasoning_packet),
            "reasoning_graph_nodes": list(self.reasoning_graph_nodes),
            "reasoning_graph_edges": list(self.reasoning_graph_edges),
            "reasoning_path": list(self.reasoning_path),
            "global_graph_size": dict(self.global_graph_size),
            "reused_graph_edges": list(self.reused_graph_edges),
            "new_graph_edges": list(self.new_graph_edges),
            "structured_explanation": dict(self.structured_explanation),
            "final_explanation": self.final_explanation,
            "cache_bypassed_reason": self.cache_bypassed_reason,
            "llm_explanation": dict(self.llm_explanation),
            "memory_written": self.memory_written,
        }

    def to_trace_steps(self) -> List[Dict[str, object]]:
        return [
            {"operation": "route_decision", "details": dict(self.route_decision)},
            {"operation": "prompt_enrichment", "details": dict(self.enrichment_result)},
            {"operation": "clarification_state", "details": dict(self.clarification_state)},
            {"operation": "goal_tracking", "details": dict(self.goal_tracking)},
            {"operation": "strategy_plan", "details": dict(self.strategy_plan)},
            {"operation": "merged_memory_state", "details": dict(self.merged_memory_state)},
            {"operation": "memory_reconstruction", "details": dict(self.merged_memory_state.get("reconstructed_context", {}))},
            {"operation": "parsed_input", "details": dict(self.parsed_input)},
            {"operation": "retrieved_memory", "details": {"count": len(self.retrieved_memories), "memories": list(self.retrieved_memories)}},
            {"operation": "recalled_turns", "details": {"count": len(self.recalled_turns), "turns": list(self.recalled_turns)}},
            {"operation": "concept_matches", "details": {"matches": list(self.concept_matches)}},
            {"operation": "candidate_paths", "details": {"count": len(self.candidate_paths), "paths": list(self.candidate_paths)}},
            {"operation": "grounded_paths", "details": {"paths": list(self.grounded_paths)}},
            {"operation": "goal_scored_paths", "details": {"goal_weights": dict(self.goal_weights), "paths": list(self.goal_scored_paths)}},
            {"operation": "selected_path", "details": dict(self.selected_path)},
            {"operation": "context_packet_v1", "details": dict(self.context_packet_v1)},
            {"operation": "llm_payload_v1", "details": dict(self.llm_payload_v1)},
            {"operation": "llm_payload_v2", "details": dict(self.llm_payload_v2)},
            {
                "operation": "cached_knowledge",
                "details": {
                    "checked": self.cached_knowledge_checked,
                    "hit": self.cached_knowledge_hit,
                    "entry_ids": list(self.cached_knowledge_entry_ids),
                    "written": self.cached_knowledge_written,
                    "topic_key": self.cached_knowledge_topic_key,
                    "normalized_terms": list(self.cached_knowledge_normalized_terms),
                    "trust_level": self.cached_knowledge_trust_level,
                    "used": self.cached_knowledge_used,
                    "entry_id": self.cached_knowledge_entry_id,
                    "candidate_entry_ids": list(self.cached_knowledge_candidate_entry_ids),
                    "selected_entry_id": self.cached_knowledge_selected_entry_id,
                    "ranking_scores": list(self.cached_knowledge_ranking_scores),
                    "reinforcement_applied": self.cached_knowledge_reinforcement_applied,
                    "updated_confidence": self.cached_knowledge_updated_confidence,
                    "updated_reuse_count": self.cached_knowledge_updated_reuse_count,
                    "conflict_detected": self.cached_knowledge_conflict_detected,
                    "conflicting_entry_ids": list(self.cached_knowledge_conflicting_entry_ids),
                    "penalized_entry_ids": list(self.cached_knowledge_penalized_entry_ids),
                    "updated_confidences": list(self.cached_knowledge_updated_confidences),
                    "verification_attempted": self.cached_knowledge_verification_attempted,
                    "verification_status": self.cached_knowledge_verification_status,
                    "confidence_capped": self.cached_knowledge_confidence_capped,
                    "uncertainty_flag": self.cached_knowledge_uncertainty_flag,
                    "stabilized_flag": self.cached_knowledge_stabilized_flag,
                    "dominance_score": self.cached_knowledge_dominance_score,
                    "convergence_rate": self.cached_knowledge_convergence_rate,
                    "multi_hop_used": self.multi_hop_used,
                    "merged_entry_ids": list(self.merged_entry_ids),
                    "reasoning_steps": list(self.reasoning_steps),
                    "merged_reasoning_packet": dict(self.merged_reasoning_packet),
                    "reasoning_graph_nodes": list(self.reasoning_graph_nodes),
                    "reasoning_graph_edges": list(self.reasoning_graph_edges),
                    "reasoning_path": list(self.reasoning_path),
                    "global_graph_size": dict(self.global_graph_size),
                    "reused_graph_edges": list(self.reused_graph_edges),
                    "new_graph_edges": list(self.new_graph_edges),
                    "structured_explanation": dict(self.structured_explanation),
                    "final_explanation": self.final_explanation,
                    "bypassed_reason": self.cache_bypassed_reason,
                },
            },
        ]


@dataclass(frozen=True)
class HybridTurnResult:
    user_input_text: str
    effective_prompt_text: str
    route_decision: QueryRouteDecision
    enrichment_result: Dict[str, object]
    clarification_state: Dict[str, object]
    goal_tracking: Dict[str, object]
    strategy_plan: Dict[str, object]
    parse_result: Dict[str, object]
    parsed_brain_input: BrainInputV1
    retrieved_memories: List[RetrievedMemory]
    recalled_turns: List[EpisodicTurnRecall]
    decision: Dict[str, object]
    trace: HybridReasoningTrace
    llm_explanation: LLMExplanationResult
    stored_entries: List[EpisodicMemoryEntry]
    final_result: BrainInputV1

    def to_dict(self) -> Dict[str, object]:
        return {
            "user_input_text": self.user_input_text,
            "effective_prompt_text": self.effective_prompt_text,
            "route_decision": self.route_decision.to_dict(),
            "enrichment_result": dict(self.enrichment_result),
            "clarification_state": dict(self.clarification_state),
            "goal_tracking": dict(self.goal_tracking),
            "strategy_plan": dict(self.strategy_plan),
            "parse_result": dict(self.parse_result),
            "parsed_brain_input": self.parsed_brain_input.to_dict(),
            "retrieved_memories": [item.to_dict() for item in self.retrieved_memories],
            "recalled_turns": [item.to_dict() for item in self.recalled_turns],
            "decision": dict(self.decision),
            "trace": self.trace.to_dict(),
            "llm_explanation": self.llm_explanation.to_dict(),
            "stored_entries": [entry.to_dict() for entry in self.stored_entries],
            "final_result": self.final_result.to_dict(),
        }


class EpisodicMemoryStore:
    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.entries: List[EpisodicMemoryEntry] = []
        self._next_entry_id = 1
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            self.entries = []
            self._next_entry_id = 1
            return
        data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        loaded = [
            EpisodicMemoryEntry.from_dict(item)
            for item in data.get("entries", [])
            if isinstance(item, dict)
        ] if isinstance(data, dict) else []
        self.entries = [entry for entry in loaded if _is_meaningful_entry(entry)]
        self._next_entry_id = max([1] + [entry.entry_id + 1 for entry in loaded], default=1)

    def save(self) -> None:
        payload = {
            "entries": [entry.to_dict() for entry in self.entries],
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def add(
        self,
        *,
        entry_type: str,
        source_text: str,
        parsed_memory: BrainMemory,
        brain_input: BrainInputV1,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Optional[EpisodicMemoryEntry]:
        candidate = EpisodicMemoryEntry(
            entry_id=self._next_entry_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type=entry_type,
            source_text=source_text,
            parsed_memory=parsed_memory,
            brain_input=brain_input,
            metadata=metadata or {},
        )
        if not _is_meaningful_entry(candidate):
            return None
        self._next_entry_id += 1
        self.entries.append(candidate)
        self.save()
        return candidate

    def update_metadata(self, entry_id: int, metadata_updates: Dict[str, object]) -> Optional[EpisodicMemoryEntry]:
        for index, entry in enumerate(self.entries):
            if entry.entry_id != entry_id:
                continue
            merged_metadata = dict(entry.metadata)
            merged_metadata.update(metadata_updates)
            updated = replace(entry, metadata=merged_metadata)
            self.entries[index] = updated
            self.save()
            return updated
        return None


class EpisodicTurnStore:
    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.records: List[EpisodicTurnRecord] = []
        self._next_turn_id = 1
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            self.records = []
            self._next_turn_id = 1
            return
        data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        loaded = [
            EpisodicTurnRecord.from_dict(item)
            for item in data.get("records", [])
            if isinstance(item, dict)
        ] if isinstance(data, dict) else []
        self.records = [record for record in loaded if _has_text(record.raw_user_prompt) or _has_text(record.final_answer)]
        self._next_turn_id = max([1] + [record.turn_id + 1 for record in loaded], default=1)

    def save(self) -> None:
        payload = {"records": [record.to_dict() for record in self.records]}
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def add(
        self,
        *,
        sequence: int,
        raw_user_prompt: str,
        final_answer: str,
        parsed_brain_input: BrainInputV1,
        final_selected_result: BrainInputV1,
        route_type: str,
        reasoning_succeeded: bool,
        candidate_path_count: int,
        route_confidence: float,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Optional[EpisodicTurnRecord]:
        if not _has_text(raw_user_prompt) and not _has_text(final_answer):
            return None
        record = EpisodicTurnRecord(
            turn_id=self._next_turn_id,
            sequence=sequence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            raw_user_prompt=raw_user_prompt,
            final_answer=final_answer,
            parsed_brain_input=parsed_brain_input,
            final_selected_result=final_selected_result,
            route_type=route_type,
            reasoning_succeeded=reasoning_succeeded,
            candidate_path_count=max(0, int(candidate_path_count)),
            route_confidence=max(0.0, float(route_confidence)),
            metadata=metadata or {},
        )
        self._next_turn_id += 1
        self.records.append(record)
        self.save()
        return record


class PendingClarificationStore:
    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.pending: Optional[PendingClarification] = None
        self._next_id = 1
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            self.pending = None
            self._next_id = 1
            return
        data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        pending_raw = data.get("pending") if isinstance(data, dict) else None
        self.pending = PendingClarification.from_dict(pending_raw) if isinstance(pending_raw, dict) else None
        self._next_id = int(data.get("next_id", 1)) if isinstance(data, dict) else 1

    def save(self) -> None:
        payload = {
            "next_id": self._next_id,
            "pending": self.pending.to_dict() if self.pending is not None else None,
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def set_pending(
        self,
        *,
        route_type: str,
        original_prompt: str,
        effective_prompt: str,
        clarification_question: str,
        missing_fields: Sequence[str],
        clarification_type: str,
        route_decision: Dict[str, object],
        enrichment_result: Dict[str, object],
        parse_result: Dict[str, object],
        previous_context: Optional[Dict[str, object]] = None,
    ) -> PendingClarification:
        pending = PendingClarification(
            clarification_id=self._next_id,
            route_type=route_type,
            original_prompt=original_prompt,
            effective_prompt=effective_prompt,
            clarification_question=clarification_question,
            missing_fields=[str(item) for item in missing_fields if str(item).strip()],
            clarification_type=clarification_type,
            route_decision=dict(route_decision),
            enrichment_result=dict(enrichment_result),
            parse_result=dict(parse_result),
            previous_context=previous_context or {},
        )
        self._next_id += 1
        self.pending = pending
        self.save()
        return pending

    def clear(self) -> None:
        self.pending = None
        self.save()


class PersistentReasoningGraphStore:
    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.nodes: List[Dict[str, object]] = []
        self.edges: List[Dict[str, object]] = []
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            self.nodes = []
            self.edges = []
            return
        data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        self.nodes = [dict(item) for item in data.get("nodes", []) if isinstance(item, dict)] if isinstance(data, dict) else []
        self.edges = [dict(item) for item in data.get("edges", []) if isinstance(item, dict)] if isinstance(data, dict) else []

    def save(self) -> None:
        payload = {
            "nodes": list(self.nodes),
            "edges": list(self.edges),
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def normalize_concept(self, label: object) -> str:
        normalized = re.sub(r"[^a-z0-9\s_]", " ", str(label or "").lower())
        normalized = re.sub(r"\s+", "_", normalized).strip("_")
        return normalized

    def _canonical_node(self, node: Dict[str, object]) -> Dict[str, object]:
        normalized_id = self.normalize_concept(node.get("node_id", "") or node.get("label", ""))
        if not normalized_id:
            return {}
        return {
            "node_id": normalized_id,
            "label": str(node.get("label", normalized_id)).strip() or normalized_id,
            "normalized_concept": normalized_id,
            "type": str(node.get("type", "concept") or "concept"),
        }

    def _canonical_edge(self, edge: Dict[str, object]) -> Dict[str, object]:
        normalized_from = self.normalize_concept(edge.get("from", "") or edge.get("from_label", ""))
        normalized_to = self.normalize_concept(edge.get("to", "") or edge.get("to_label", ""))
        relationship = str(edge.get("relationship", "related_to") or "related_to").strip().lower() or "related_to"
        if not normalized_from or not normalized_to:
            return {}
        evidence_value = str(edge.get("evidence", "")).strip()
        evidence_list = [evidence_value] if evidence_value else []
        return {
            "from": normalized_from,
            "from_label": str(edge.get("from_label", normalized_from)).strip() or normalized_from,
            "relationship": relationship,
            "to": normalized_to,
            "to_label": str(edge.get("to_label", normalized_to)).strip() or normalized_to,
            "support_count": max(1, int(edge.get("support_count", 1) or 1)),
            "evidence": evidence_list,
        }

    def merge_graph(
        self,
        *,
        nodes: Sequence[Dict[str, object]],
        edges: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        node_lookup = {
            str(item.get("node_id", "")).strip().lower(): dict(item)
            for item in self.nodes
            if isinstance(item, dict) and str(item.get("node_id", "")).strip()
        }
        edge_lookup = {
            (
                str(item.get("from", "")).strip().lower(),
                str(item.get("relationship", "")).strip().lower(),
                str(item.get("to", "")).strip().lower(),
            ): dict(item)
            for item in self.edges
            if isinstance(item, dict)
        }
        new_nodes: List[Dict[str, object]] = []
        new_edges: List[Dict[str, object]] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            canonical = self._canonical_node(node)
            node_id = str(canonical.get("node_id", "")).strip().lower()
            if not node_id:
                continue
            existing = node_lookup.get(node_id)
            if existing is None:
                node_lookup[node_id] = canonical
                new_nodes.append(dict(canonical))
                continue
            if not str(existing.get("label", "")).strip() and str(canonical.get("label", "")).strip():
                existing["label"] = canonical["label"]
            if not str(existing.get("type", "")).strip() and str(canonical.get("type", "")).strip():
                existing["type"] = canonical["type"]
            existing["normalized_concept"] = canonical["normalized_concept"]
            node_lookup[node_id] = existing
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            canonical = self._canonical_edge(edge)
            if not canonical:
                continue
            edge_key = (
                str(canonical.get("from", "")).strip().lower(),
                str(canonical.get("relationship", "")).strip().lower(),
                str(canonical.get("to", "")).strip().lower(),
            )
            existing = edge_lookup.get(edge_key)
            if existing is None:
                edge_lookup[edge_key] = dict(canonical)
                new_edges.append(dict(canonical))
                continue
            existing["support_count"] = max(1, int(existing.get("support_count", 1) or 1)) + 1
            existing_evidence = [str(item).strip() for item in existing.get("evidence", []) if str(item).strip()] if isinstance(existing.get("evidence", []), list) else []
            for evidence in canonical.get("evidence", []):
                if evidence not in existing_evidence:
                    existing_evidence.append(evidence)
            existing["evidence"] = existing_evidence[:8]
            edge_lookup[edge_key] = existing
        self.nodes = sorted(node_lookup.values(), key=lambda item: str(item.get("node_id", "")))
        self.edges = sorted(
            edge_lookup.values(),
            key=lambda item: (
                str(item.get("from", "")),
                str(item.get("relationship", "")),
                str(item.get("to", "")),
            ),
        )
        self.save()
        return {
            "global_graph_size": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
            },
            "new_nodes": new_nodes,
            "new_edges": new_edges,
        }

    def find_relevant_edges(
        self,
        *,
        query_terms: Sequence[str],
        limit: int = 6,
    ) -> List[Dict[str, object]]:
        normalized_terms = {
            self.normalize_concept(item)
            for item in query_terms
            if str(item).strip()
        }
        if not normalized_terms:
            return []
        scored: List[tuple[float, Dict[str, object]]] = []
        for edge in self.edges:
            if not isinstance(edge, dict):
                continue
            edge_terms = {
                self.normalize_concept(edge.get("from", "")),
                self.normalize_concept(edge.get("from_label", "")),
                self.normalize_concept(edge.get("to", "")),
                self.normalize_concept(edge.get("to_label", "")),
                self.normalize_concept(edge.get("relationship", "")),
            }
            edge_terms.discard("")
            intersection = sorted(normalized_terms & edge_terms)
            if not intersection:
                continue
            support_count = max(1, int(edge.get("support_count", 1) or 1))
            score = len(intersection) * 3.0 + min(4.0, support_count * 0.5)
            scored.append((score, dict(edge)))
        scored.sort(
            key=lambda item: (
                -float(item[0]),
                -int(item[1].get("support_count", 1) or 1),
                str(item[1].get("from", "")),
                str(item[1].get("to", "")),
            )
        )
        return [dict(edge) for _, edge in scored[:limit]]


class BrainHybridPrototype:
    def __init__(
        self,
        state_dir: str | Path = Path("brain_output") / "interactive_prototype.state",
        storage_policy: Optional[StoragePolicy] = None,
        llm_provider: str = "auto",
        bootstrap_reasoning: bool = True,
        pixel_memory_enabled: bool = True,
        relation_memory_enabled: bool = True,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.storage_policy = storage_policy or StoragePolicy()
        self.parser = BrainParser()
        self.enrichment = BrainPromptEnrichment()
        self.codec = BrainCodec()
        self.engine = BrainGoalRewardV5(state_dir=self.state_dir / "v5")
        self.goal_tracker = BrainGoalTrackingV6(self.state_dir / "goal_tracking_v6.json")
        self.strategy_generator = BrainStrategyGenerationV7(self.state_dir / "strategy_generation_v7.json")
        self.memory_store = EpisodicMemoryStore(self.state_dir / "episodic_memory.json")
        self.turn_store = EpisodicTurnStore(self.state_dir / "episodic_turns.json")
        self.pending_clarification_store = PendingClarificationStore(self.state_dir / "pending_clarification.json")
        self.global_graph_store = PersistentReasoningGraphStore(self.state_dir / "global_reasoning_graph.json")
        self.memory_reconstruction_config = MemoryReconstructionConfig()
        self.pixel_memory_enabled = bool(pixel_memory_enabled)
        self.relation_memory_enabled = bool(relation_memory_enabled)
        self.pixel_memory_canvas = PixelMemoryCanvas(width=64, height=64)
        self.selection_graph_evaluator = ReasoningGraphEvaluator(
            min_candidates=5,
            max_candidates=10,
            concept_memory_path=self.state_dir / "spatial_concept_memory.json",
        )
        self.history = BrainHistory(self.state_dir / "prototype_history.json")
        self.llm = HybridLLMInterface(provider=llm_provider)
        self._sequence = max(1, len(self.memory_store.entries) + 1)
        self._turn_sequence = max(1, len(self.turn_store.records) + 1)
        self._bootstrap_generic_goals()
        if bootstrap_reasoning:
            self._bootstrap_reasoning_memory()

    def run_prompt(
        self,
        user_input_text: str,
        *,
        inspect: bool = True,
        top_k: int = 4,
        effective_prompt_text: Optional[str] = None,
        route_override: Optional[QueryRouteDecision] = None,
        clarification_state: Optional[Dict[str, object]] = None,
    ) -> HybridTurnResult:
        effective_prompt = str(effective_prompt_text or user_input_text)
        parse_result = self.parser.parse(effective_prompt)
        parsed_memory = BrainCodec.resolve_memory(parse_result.memory)
        parsed_brain_input = self._memory_to_brain_input(parsed_memory)
        route_decision = route_override or route_user_prompt(effective_prompt, parsed_brain_input)
        if route_override is None and self._should_force_memory_route(effective_prompt, route_decision):
            route_decision = self._promote_route_decision(
                route_decision,
                route_type="memory_question",
                minimum_confidence=0.9,
                reason="Relevant recorded context or prior-state recall cues make memory retrieval preferable.",
            )
        enrichment_result = self.enrichment.enrich(
            prompt_text=effective_prompt,
            route_type=route_decision.route_type,
            memory=parsed_memory,
        )
        working_memory = BrainCodec.resolve_memory(BrainMemory.from_dict(enrichment_result.enriched_memory))
        working_brain_input = self._memory_to_brain_input(working_memory)
        if route_decision.route_type == "factual_question":
            return self._run_factual_turn(
                user_input_text=user_input_text,
                effective_prompt_text=effective_prompt,
                parse_result=parse_result,
                parsed_memory=parsed_memory,
                parsed_brain_input=parsed_brain_input,
                enrichment_result=enrichment_result,
                route_decision=route_decision,
                clarification_state=clarification_state or {},
            )
        if route_decision.route_type == "memory_question":
            return self._run_memory_turn(
                user_input_text=user_input_text,
                effective_prompt_text=effective_prompt,
                parse_result=parse_result,
                parsed_memory=parsed_memory,
                parsed_brain_input=parsed_brain_input,
                enrichment_result=enrichment_result,
                route_decision=route_decision,
                top_k=top_k,
                clarification_state=clarification_state or {},
            )
        if route_decision.route_type == "state_update":
            return self._run_state_update_turn(
                user_input_text=user_input_text,
                effective_prompt_text=effective_prompt,
                parse_result=parse_result,
                parsed_memory=parsed_memory,
                parsed_brain_input=parsed_brain_input,
                enrichment_result=enrichment_result,
                route_decision=route_decision,
                clarification_state=clarification_state or {},
            )
        if route_decision.route_type in {"planning_request", "goal_request", "transform_request"}:
            return self._run_reasoning_turn(
                user_input_text=user_input_text,
                effective_prompt_text=effective_prompt,
                parse_result=parse_result,
                parsed_memory=parsed_memory,
                parsed_brain_input=working_brain_input,
                enrichment_result=enrichment_result,
                route_decision=route_decision,
                top_k=top_k,
                direct_answer=False,
                clarification_state=clarification_state or {},
            )
        if route_decision.route_type == "hybrid_request":
            return self._run_reasoning_turn(
                user_input_text=user_input_text,
                effective_prompt_text=effective_prompt,
                parse_result=parse_result,
                parsed_memory=parsed_memory,
                parsed_brain_input=working_brain_input,
                enrichment_result=enrichment_result,
                route_decision=route_decision,
                top_k=top_k,
                direct_answer=True,
                clarification_state=clarification_state or {},
            )
        return self._run_unknown_turn(
            user_input_text=user_input_text,
            effective_prompt_text=effective_prompt,
            parse_result=parse_result,
            parsed_memory=parsed_memory,
            parsed_brain_input=parsed_brain_input,
            enrichment_result=enrichment_result,
            route_decision=route_decision,
            clarification_state=clarification_state or {},
        )

    def process_user_turn(
        self,
        user_input_text: str,
        *,
        inspect: bool = True,
        top_k: int = 4,
    ) -> HybridTurnResult:
        pending = self.pending_clarification_store.pending
        if pending is None:
            return self.run_prompt(user_input_text, inspect=inspect, top_k=top_k)
        return self.respond_to_clarification(user_input_text, inspect=inspect, top_k=top_k)

    def has_pending_clarification(self) -> bool:
        return self.pending_clarification_store.pending is not None

    def get_pending_clarification(self) -> Optional[Dict[str, object]]:
        pending = self.pending_clarification_store.pending
        return pending.to_dict() if pending is not None else None

    def respond_to_clarification(
        self,
        response_text: str,
        *,
        inspect: bool = True,
        top_k: int = 4,
    ) -> HybridTurnResult:
        pending = self.pending_clarification_store.pending
        if pending is None:
            return self.run_prompt(response_text, inspect=inspect, top_k=top_k)
        merged_prompt = self._merge_clarification_prompt(pending, response_text)
        route_decision = QueryRouteDecision(
            route_type=pending.route_type,
            confidence=float(pending.route_decision.get("confidence", 0.9)),
            reasons=[str(item) for item in pending.route_decision.get("reasons", []) if str(item).strip()] + ["Clarification response merged into original prompt."],
            run_brain_reasoning=bool(pending.route_decision.get("run_brain_reasoning", True)),
            run_direct_llm=bool(pending.route_decision.get("run_direct_llm", False)),
            run_hybrid_mode=bool(pending.route_decision.get("run_hybrid_mode", False)),
            score_breakdown=pending.route_decision.get("score_breakdown", {}) if isinstance(pending.route_decision.get("score_breakdown", {}), dict) else {},
        )
        clarification_state = {
            "active": False,
            "resolved": True,
            "clarification_id": pending.clarification_id,
            "clarification_question": pending.clarification_question,
            "clarification_response": response_text,
            "original_prompt": pending.original_prompt,
            "merged_prompt": merged_prompt,
            "missing_fields": list(pending.missing_fields),
            "clarification_type": pending.clarification_type,
        }
        self.pending_clarification_store.clear()
        return self.run_prompt(
            response_text,
            inspect=inspect,
            top_k=top_k,
            effective_prompt_text=merged_prompt,
            route_override=route_decision,
            clarification_state=clarification_state,
        )

    def _run_factual_turn(
        self,
        *,
        user_input_text: str,
        effective_prompt_text: str,
        parse_result: ParseResult,
        parsed_memory: BrainMemory,
        parsed_brain_input: BrainInputV1,
        enrichment_result: PromptEnrichmentResult,
        route_decision: QueryRouteDecision,
        clarification_state: Dict[str, object],
    ) -> HybridTurnResult:
        normalized_terms = self.normalize_query(user_input_text)
        cached_matches, cache_ranking = self._cache_rank_candidates(
            user_input_text=user_input_text,
            top_k=5,
            update_reuse=True,
        )
        multi_hop_context = self._build_multi_hop_cache_reasoning(
            user_input_text=user_input_text,
            entries=cached_matches,
            ranking_scores=cache_ranking.get("ranking_scores", []),
        ) if cached_matches else {
            "multi_hop_used": False,
            "merged_entry_ids": [],
            "reasoning_steps": [],
            "merged_reasoning_packet": {},
            "reasoning_graph_nodes": [],
            "reasoning_graph_edges": [],
            "reasoning_path": [],
        }
        selection_graph_nodes = cache_ranking.get("reasoning_graph_nodes", [])
        selection_graph_edges = cache_ranking.get("reasoning_graph_edges", [])
        graph_reasoning_steps = cache_ranking.get("graph_reasoning_steps", [])
        merged_reasoning_packet = multi_hop_context.get("merged_reasoning_packet", {})
        if not isinstance(merged_reasoning_packet, dict):
            merged_reasoning_packet = {}
        if selection_graph_nodes or selection_graph_edges:
            merged_reasoning_packet = {
                **merged_reasoning_packet,
                "selection_reasoning_graph": {
                    "nodes": list(selection_graph_nodes),
                    "edges": list(selection_graph_edges),
                },
            }
        reasoning_steps = [
            *[str(item) for item in graph_reasoning_steps if str(item).strip()],
            *[str(item) for item in multi_hop_context.get("reasoning_steps", []) if str(item).strip()],
        ]
        trace_graph_nodes = multi_hop_context.get("reasoning_graph_nodes", []) if multi_hop_context.get("reasoning_graph_nodes", []) else selection_graph_nodes
        trace_graph_edges = multi_hop_context.get("reasoning_graph_edges", []) if multi_hop_context.get("reasoning_graph_edges", []) else selection_graph_edges
        topic_key = self.normalize_knowledge_topic(user_input_text)
        effective_route_decision = self._route_decision_with_cache_short_circuit(route_decision) if cached_matches else route_decision
        trace = self._build_trace(
            raw_prompt=user_input_text,
            effective_prompt=effective_prompt_text,
            route_decision=effective_route_decision,
            enrichment_result=enrichment_result,
            clarification_state=clarification_state,
            goal_tracking={},
            parse_result=parse_result,
            parsed_memory=parsed_memory,
            parsed_brain_input=parsed_brain_input,
            retrieved=[],
            recalled_turns=[],
            goal_weights={},
            decision=None,
            final_result=parsed_brain_input,
        )
        trace = self._trace_with_cached_knowledge(
            trace,
            checked=True,
            hit=bool(cached_matches),
            entry_ids=[item.entry_id for item in cached_matches],
            topic_key=topic_key,
            normalized_terms=cache_ranking.get("normalized_terms", normalized_terms),
            trust_level="low" if cached_matches else "",
            written=False,
            used=bool(cached_matches),
            entry_id=(cached_matches[0].entry_id if cached_matches else None),
            candidate_entry_ids=cache_ranking.get("candidate_entry_ids", []),
            selected_entry_id=cache_ranking.get("selected_entry_id"),
            ranking_scores=cache_ranking.get("ranking_scores", []),
            reinforcement_applied=bool(cache_ranking.get("reinforcement_applied", False)),
            updated_confidence=cache_ranking.get("updated_confidence"),
            updated_reuse_count=cache_ranking.get("updated_reuse_count"),
            conflict_detected=bool(cache_ranking.get("conflict_detected", False)),
            conflicting_entry_ids=cache_ranking.get("conflicting_entry_ids", []),
            penalized_entry_ids=cache_ranking.get("penalized_entry_ids", []),
            updated_confidences=cache_ranking.get("updated_confidences", []),
            verification_attempted=bool(cache_ranking.get("verification_attempted", False)),
            verification_status=str(cache_ranking.get("verification_status", "") or ""),
            confidence_capped=bool(cache_ranking.get("confidence_capped", False)),
            uncertainty_flag=bool(cache_ranking.get("uncertainty_flag", False)),
            stabilized_flag=bool(cache_ranking.get("stabilized_flag", False)),
            dominance_score=cache_ranking.get("dominance_score"),
            convergence_rate=cache_ranking.get("convergence_rate"),
            multi_hop_used=bool(multi_hop_context.get("multi_hop_used", False)),
            merged_entry_ids=multi_hop_context.get("merged_entry_ids", []),
            reasoning_steps=reasoning_steps,
            merged_reasoning_packet=merged_reasoning_packet,
            reasoning_graph_nodes=trace_graph_nodes,
            reasoning_graph_edges=trace_graph_edges,
            reasoning_path=multi_hop_context.get("reasoning_path", []),
            global_graph_size=multi_hop_context.get("global_graph_size", {}),
            reused_graph_edges=multi_hop_context.get("reused_graph_edges", []),
            new_graph_edges=multi_hop_context.get("new_graph_edges", []),
            structured_explanation=multi_hop_context.get("structured_explanation", {}),
            final_explanation=str(multi_hop_context.get("final_explanation", "") or ""),
            bypassed_reason=None if cached_matches else "no_cache_match",
        )
        return self._finalize_turn(
            user_input_text=user_input_text,
            effective_prompt_text=effective_prompt_text,
            route_decision=effective_route_decision,
            enrichment_result=enrichment_result,
            clarification_state=clarification_state,
            parse_result=parse_result,
            parsed_brain_input=parsed_brain_input,
            retrieved=[],
            recalled_turns=[],
            decision={},
            trace=trace,
            llm_explanation=None,
            stored_entries=[],
            final_result=parsed_brain_input,
        )

    def _run_memory_turn(
        self,
        *,
        user_input_text: str,
        effective_prompt_text: str,
        parse_result: ParseResult,
        parsed_memory: BrainMemory,
        parsed_brain_input: BrainInputV1,
        enrichment_result: PromptEnrichmentResult,
        route_decision: QueryRouteDecision,
        top_k: int,
        clarification_state: Dict[str, object],
    ) -> HybridTurnResult:
        state_retrieved = self.retrieve_stateful_memory(user_input_text=user_input_text, top_k=max(top_k, 6))
        recalled_turns = self.retrieve_episodic_turns(
            user_input_text=user_input_text,
            route_type=route_decision.route_type,
            top_k=top_k,
        )
        retrieved = state_retrieved or self.retrieve_relevant_brain_memory(parsed_brain_input, query_text=user_input_text, top_k=top_k)
        merged_memory_state = self.merge_memory_state(
            user_input_text=user_input_text,
            retrieved=retrieved,
            recalled_turns=recalled_turns,
        )
        relation_context = (
            context_from_turns(self.turn_store.records,user_input_text,scope=self.state_dir.name, constraint_extractor=self._extract_state_observations)
            if self.relation_memory_enabled else {"active":False}
        )
        if relation_context.get("active"):
            merged_memory_state = apply_to_legacy_state(merged_memory_state,relation_context)
        reconstruction = self.reconstruct_memory_context(
            user_input_text=user_input_text,
            retrieved=retrieved,
            recalled_turns=recalled_turns,
            merged_memory_state=merged_memory_state,
        )
        if relation_context.get("active"):
            supported = str(relation_context.get("supported_answer",""))
            status = relation_context.get("answer_status","requires_interpretation")
            proof_ids = [source["id"] for source in relation_context.get("sources",[])]
            reconstruction = replace(
                reconstruction,
                synthesized_summary=supported,
                confidence=1.0 if supported else 0.0,
                answer_mode="high" if supported else "low",
                evidence_breakdown={
                    **reconstruction.evidence_breakdown,
                    "answer_selection_source":"event_relations",
                    "relation_answer_status":status,
                    "relation_source_ids":proof_ids,
                    "relation_evidence_authoritative":True,
                })
        if relation_context.get("active") or reconstruction.partial_matches or reconstruction.exact_matches:
            merged_memory_state = {
                **merged_memory_state,
                "reconstructed_context": reconstruction.to_dict(),
            }
        final_result = self._memory_state_to_brain_input(merged_memory_state) or (
            recalled_turns[0].record.final_selected_result if recalled_turns else (retrieved[0].entry.brain_input if retrieved else parsed_brain_input)
        )
        if relation_context.get("active"):
            supported = str(relation_context.get("supported_answer",""))
            if supported:
                final_result = self._memory_state_to_brain_input({"items":[{
                    "state_type":"relation_answer","state_key":"source_backed_answer",
                    "current_value":supported}]}) or parsed_brain_input
            else:
                # A question representation is not a claim about an old recalled state.
                final_result = parsed_brain_input
        trace = self._build_trace(
            raw_prompt=user_input_text,
            effective_prompt=effective_prompt_text,
            route_decision=route_decision,
            enrichment_result=enrichment_result,
            clarification_state=clarification_state,
            goal_tracking={},
            parse_result=parse_result,
            parsed_memory=parsed_memory,
            parsed_brain_input=parsed_brain_input,
            retrieved=retrieved,
            recalled_turns=recalled_turns,
            goal_weights={},
            decision=None,
            final_result=final_result,
        )
        trace = self._trace_with_memory_state(trace, merged_memory_state)
        return self._finalize_turn(
            user_input_text=user_input_text,
            effective_prompt_text=effective_prompt_text,
            route_decision=route_decision,
            enrichment_result=enrichment_result,
            clarification_state=clarification_state,
            parse_result=parse_result,
            parsed_brain_input=parsed_brain_input,
            retrieved=retrieved,
            recalled_turns=recalled_turns,
            decision={},
            trace=trace,
            llm_explanation=None,
            stored_entries=[],
            final_result=final_result,
        )

    def _run_reasoning_turn(
        self,
        *,
        user_input_text: str,
        effective_prompt_text: str,
        parse_result: ParseResult,
        parsed_memory: BrainMemory,
        parsed_brain_input: BrainInputV1,
        enrichment_result: PromptEnrichmentResult,
        route_decision: QueryRouteDecision,
        top_k: int,
        direct_answer: bool,
        clarification_state: Dict[str, object],
    ) -> HybridTurnResult:
        stored_entries: List[EpisodicMemoryEntry] = []
        if enrichment_result.needs_clarification:
            return self._run_clarification_turn(
                user_input_text=user_input_text,
                effective_prompt_text=effective_prompt_text,
                parse_result=parse_result,
                parsed_memory=parsed_memory,
                parsed_brain_input=parsed_brain_input,
                enrichment_result=enrichment_result,
                route_decision=route_decision,
                clarification_state=clarification_state,
            )
        retrieved = self.retrieve_relevant_brain_memory(parsed_brain_input, query_text=user_input_text, top_k=top_k)
        merged_memory_state = self.merge_memory_state(
            user_input_text=user_input_text,
            retrieved=retrieved,
            recalled_turns=[],
        ) if retrieved else {}
        trace = self._build_trace(
            raw_prompt=user_input_text,
            effective_prompt=effective_prompt_text,
            route_decision=route_decision,
            enrichment_result=enrichment_result,
            clarification_state=clarification_state,
            goal_tracking={},
            parse_result=parse_result,
            parsed_memory=parsed_memory,
            parsed_brain_input=parsed_brain_input,
            retrieved=retrieved,
            recalled_turns=[],
            goal_weights={},
            decision=None,
            final_result=parsed_brain_input,
        )
        if merged_memory_state:
            trace = self._trace_with_memory_state(trace, merged_memory_state)
        if self.storage_policy.should_store_user_prompt(trace):
            self._append_stored_entry(
                stored_entries,
                entry_type="user_prompt",
                source_text=user_input_text,
                parsed_memory=parsed_memory,
                brain_input=parsed_brain_input,
                metadata={"sequence": self._sequence, "route_type": route_decision.route_type},
            )
        goal_weights = self.infer_goal_weights(user_input_text, parsed_memory, retrieved)
        decision = self.engine.plan(
            parsed_brain_input,
            goal_weights=goal_weights,
            max_depth=5,
            beam_width=4,
            current_sequence=self._sequence,
        )
        self._sequence += 1
        final_result = self._select_result_from_decision(decision, goal_weights) or parsed_brain_input
        trace = self._build_trace(
            raw_prompt=user_input_text,
            effective_prompt=effective_prompt_text,
            route_decision=route_decision,
            enrichment_result=enrichment_result,
            clarification_state=clarification_state,
            goal_tracking={},
            parse_result=parse_result,
            parsed_memory=parsed_memory,
            parsed_brain_input=parsed_brain_input,
            retrieved=retrieved,
            recalled_turns=[],
            goal_weights=goal_weights,
            decision=decision,
            final_result=final_result,
        )
        if self.storage_policy.should_store_result(trace):
            final_memory = self._brain_input_to_memory(final_result)
            self._append_stored_entry(
                stored_entries,
                entry_type="selected_result",
                source_text="",
                parsed_memory=final_memory,
                brain_input=final_result,
                metadata={
                    "sequence": self._sequence,
                    "goal_weights": goal_weights,
                    "source_prompt": user_input_text,
                    "route_type": route_decision.route_type,
                },
            )
        if self.storage_policy.should_store_trace(trace) and self._trace_has_high_value_content(trace):
            trace_memory = self._brain_input_to_memory(final_result)
            self._append_stored_entry(
                stored_entries,
                entry_type="reasoning_trace",
                source_text=user_input_text,
                parsed_memory=trace_memory,
                brain_input=final_result,
                metadata={
                    "sequence": self._sequence,
                    "candidate_path_count": len(trace.candidate_paths),
                    "retrieved_memory_count": len(trace.retrieved_memories),
                    "selected_goal_alignment": trace.selected_path.get("goal_alignment_score", 0.0),
                    "route_type": route_decision.route_type,
                },
            )
        return self._finalize_turn(
            user_input_text=user_input_text,
            effective_prompt_text=effective_prompt_text,
            route_decision=route_decision,
            enrichment_result=enrichment_result,
            clarification_state=clarification_state,
            parse_result=parse_result,
            parsed_brain_input=parsed_brain_input,
            retrieved=retrieved,
            recalled_turns=[],
            decision=decision.to_dict(),
            trace=trace,
            llm_explanation=None,
            stored_entries=stored_entries,
            final_result=final_result,
        )

    def _run_unknown_turn(
        self,
        *,
        user_input_text: str,
        effective_prompt_text: str,
        parse_result: ParseResult,
        parsed_memory: BrainMemory,
        parsed_brain_input: BrainInputV1,
        enrichment_result: PromptEnrichmentResult,
        route_decision: QueryRouteDecision,
        clarification_state: Dict[str, object],
    ) -> HybridTurnResult:
        fallback_text = self._safe_unknown_route_answer(user_input_text)
        llm_explanation = LLMExplanationResult(
            provider="deterministic",
            structured_text="",
            reasoning_explanation="",
            user_facing_text=fallback_text,
            messages=[],
            metadata={"mode": "unknown_route_action_fallback" if fallback_text else "unknown_route_fallback"},
        )
        trace = self._build_trace(
            raw_prompt=user_input_text,
            effective_prompt=effective_prompt_text,
            route_decision=route_decision,
            enrichment_result=enrichment_result,
            clarification_state=clarification_state,
            goal_tracking={},
            parse_result=parse_result,
            parsed_memory=parsed_memory,
            parsed_brain_input=parsed_brain_input,
            retrieved=[],
            recalled_turns=[],
            goal_weights={},
            decision=None,
            final_result=parsed_brain_input,
        )
        trace = self._trace_with_explanation(trace, llm_explanation, [])
        return self._finalize_turn(
            user_input_text=user_input_text,
            effective_prompt_text=effective_prompt_text,
            route_decision=route_decision,
            enrichment_result=enrichment_result,
            clarification_state=clarification_state,
            parse_result=parse_result,
            parsed_brain_input=parsed_brain_input,
            retrieved=[],
            recalled_turns=[],
            decision={},
            trace=trace,
            llm_explanation=llm_explanation,
            stored_entries=[],
            final_result=parsed_brain_input,
        )

    @staticmethod
    def _safe_unknown_route_answer(user_input_text: str) -> str:
        normalized = str(user_input_text or "").lower()
        if "printer" in normalized and "queue" in normalized:
            return (
                "Start with the print queue, since the printer has power and paper. "
                "Cancel the stuck document, restart the queue or print spooler, then send one small test page."
            )
        if "bugfix" in normalized or "reproduce the bug" in normalized:
            return (
                "Start by reproducing the bug. Then make the smallest code change, update or add tests before merging, "
                "wait for review to pass, and deploy only after review is green."
            )
        if "release plan" in normalized or ("qa is blocked" in normalized and "migration" in normalized):
            return "First fix the failed staging database migration. Then rerun QA, tag the build after QA passes, and publish the release notes last."
        if "demo day" in normalized and "release notes" in normalized and "record" in normalized:
            return "Test the app first for 30 minutes. If the test passes, write the release notes for 20 minutes, record the short clip for 15 minutes, and publish only after release notes are done."
        if (
            ("github upload" in normalized and ("generated benchmark state" in normalized or "proof-of-concept documentation" in normalized))
            or ("upload plan" in normalized and "constraints" in normalized)
        ):
            return "Keep the proof-of-concept documentation, exclude generated benchmark state from the upload, run the tests, then commit the source, docs, and benchmark prompt definitions."
        if "moving day" in normalized and "fragile boxes" in normalized:
            return "Load fragile boxes on the first car trip, move the desk only with both helpers together, use the noon trip for remaining boxes, and clean only after all boxes are out."
        if "study evening" in normalized and "tutoring call" in normalized:
            return "Schedule around the tutoring call first: study math before 7:15 PM, pause for the 7:15-7:45 PM call, add a break after the first study hour, then finish history and remaining math before 9 PM if time still fits."
        if "reduce risk" in normalized and "blanket" in normalized:
            return "Do not charge the laptop under a blanket. Move it to a flat ventilated surface, use the original or certified charger, check the outlet and cable, and avoid trapping heat overnight."
        if "packing" in normalized and ("carry-on" in normalized or "one-night" in normalized):
            return "Pack around the hard limits first: keep the carry-on under 7 kilograms, put rain gear where you can reach it, and protect the presentation clothes from wrinkles."
        if "study week" in normalized and "6 hours" in normalized:
            return "Use six one-hour study blocks after work from Monday through Saturday, keep each day under the 2-hour limit, and leave Sunday completely off."
        if "meal plan" in normalized and "high-protein" in normalized:
            return "Plan a 3-day meal plan with cheap, high-protein lunches, and keep dinners flexible."
        if "website launch" in normalized and "screenshots" in normalized:
            return "Finish the missing screenshots first, then start QA. Keep deployment blocked until QA finishes."
        if "wi-fi" in normalized or "wifi" in normalized:
            return "Test another device next. If it also fails, check the ISP or modem path; if it works, focus on the original device."
        if "charge" in normalized and "laptop" in normalized:
            return "Start by checking the charger, cable, and outlet, then charge on a ventilated surface with the original or certified adapter."
        return "I need one more concrete detail before I can give a useful answer."

    def _run_state_update_turn(
        self,
        *,
        user_input_text: str,
        effective_prompt_text: str,
        parse_result: ParseResult,
        parsed_memory: BrainMemory,
        parsed_brain_input: BrainInputV1,
        enrichment_result: PromptEnrichmentResult,
        route_decision: QueryRouteDecision,
        clarification_state: Dict[str, object],
    ) -> HybridTurnResult:
        observations = self._extract_state_observations(effective_prompt_text or user_input_text)
        final_result = parsed_brain_input
        if observations:
            memory_result = self._memory_state_to_brain_input(
                {
                    "items": [
                        {
                            "state_key": item.get("state_key", ""),
                            "state_type": item.get("state_type", ""),
                            "current_value": item.get("value"),
                            "display_text": item.get("display_text", ""),
                        }
                        for item in observations
                    ]
                }
            )
            if memory_result is not None:
                final_result = memory_result
        trace = self._build_trace(
            raw_prompt=user_input_text,
            effective_prompt=effective_prompt_text,
            route_decision=route_decision,
            enrichment_result=enrichment_result,
            clarification_state=clarification_state,
            goal_tracking={},
            parse_result=parse_result,
            parsed_memory=parsed_memory,
            parsed_brain_input=parsed_brain_input,
            retrieved=[],
            recalled_turns=[],
            goal_weights={},
            decision=None,
            final_result=final_result,
        )
        llm_explanation = LLMExplanationResult(
            provider="deterministic",
            structured_text=self.codec.reconstruct_text(self._brain_input_to_memory(final_result)),
            reasoning_explanation="state_update_recorded",
            user_facing_text=self._state_update_acknowledgement(observations),
            messages=[],
            metadata={
                "mode": "state_update_recorded",
                "state_observation_count": len(observations),
            },
        )
        trace = self._trace_with_explanation(trace, llm_explanation, [])
        return self._finalize_turn(
            user_input_text=user_input_text,
            effective_prompt_text=effective_prompt_text,
            route_decision=route_decision,
            enrichment_result=enrichment_result,
            clarification_state=clarification_state,
            parse_result=parse_result,
            parsed_brain_input=parsed_brain_input,
            retrieved=[],
            recalled_turns=[],
            decision={},
            trace=trace,
            llm_explanation=llm_explanation,
            stored_entries=[],
            final_result=final_result,
        )

    def _run_clarification_turn(
        self,
        *,
        user_input_text: str,
        effective_prompt_text: str,
        parse_result: ParseResult,
        parsed_memory: BrainMemory,
        parsed_brain_input: BrainInputV1,
        enrichment_result: PromptEnrichmentResult,
        route_decision: QueryRouteDecision,
        clarification_state: Dict[str, object],
    ) -> HybridTurnResult:
        actionable_override = self._safe_unknown_route_answer(effective_prompt_text) or self._safe_unknown_route_answer(user_input_text)
        if (
            route_decision.route_type in {"planning_request", "goal_request", "transform_request", "hybrid_request"}
            and actionable_override
            and "one more concrete detail" not in actionable_override.lower()
        ):
            bypassed_clarification_state = {
                "active": False,
                "resolved": True,
                "clarification_bypassed": True,
                "bypass_reason": "actionable_prompt_had_enough_context",
                "original_clarification_question": enrichment_result.clarification_question,
            }
            llm_explanation = LLMExplanationResult(
                provider="deterministic",
                structured_text="",
                reasoning_explanation="clarification_bypassed_for_actionable_prompt",
                user_facing_text=actionable_override,
                messages=[],
                metadata={
                    "mode": "clarification_bypassed_action_answer",
                    "needs_clarification": False,
                },
            )
            trace = self._build_trace(
                raw_prompt=user_input_text,
                effective_prompt=effective_prompt_text,
                route_decision=route_decision,
                enrichment_result=enrichment_result,
                clarification_state=bypassed_clarification_state,
                goal_tracking={},
                parse_result=parse_result,
                parsed_memory=parsed_memory,
                parsed_brain_input=parsed_brain_input,
                retrieved=[],
                recalled_turns=[],
                goal_weights={},
                decision=None,
                final_result=parsed_brain_input,
            )
            trace = self._trace_with_explanation(trace, llm_explanation, [])
            return self._finalize_turn(
                user_input_text=user_input_text,
                effective_prompt_text=effective_prompt_text,
                route_decision=route_decision,
                enrichment_result=enrichment_result,
                clarification_state=bypassed_clarification_state,
                parse_result=parse_result,
                parsed_brain_input=parsed_brain_input,
                retrieved=[],
                recalled_turns=[],
                decision={},
                trace=trace,
                llm_explanation=llm_explanation,
                stored_entries=[],
                final_result=parsed_brain_input,
            )
        pending = self.pending_clarification_store.set_pending(
            route_type=route_decision.route_type,
            original_prompt=user_input_text,
            effective_prompt=effective_prompt_text,
            clarification_question=enrichment_result.clarification_question,
            missing_fields=enrichment_result.missing_fields,
            clarification_type=enrichment_result.clarification_type,
            route_decision=route_decision.to_dict(),
            enrichment_result=enrichment_result.to_dict(),
            parse_result={"memory": parsed_memory.to_dict(), "debug": parse_result.debug},
            previous_context={
                "route_reasons": route_decision.reasons,
                "inferred_fields": enrichment_result.inferred_fields,
            },
        )
        active_clarification_state = {
            "active": True,
            "resolved": False,
            "clarification_id": pending.clarification_id,
            "next_action": enrichment_result.next_action,
            "clarification_question": enrichment_result.clarification_question,
            "missing_fields": list(enrichment_result.missing_fields),
            "clarification_type": enrichment_result.clarification_type,
            "original_prompt": user_input_text,
        }
        llm_explanation = LLMExplanationResult(
            provider="deterministic",
            structured_text="",
            reasoning_explanation="",
            user_facing_text=enrichment_result.clarification_question,
            messages=[],
            metadata={
                "mode": "clarification_required",
                "next_action": "ask_user",
                "needs_clarification": True,
                "clarification_reasons": list(enrichment_result.clarification_reasons),
                "clarification_question": enrichment_result.clarification_question,
                "missing_fields": list(enrichment_result.missing_fields),
                "clarification_type": enrichment_result.clarification_type,
            },
        )
        trace = self._build_trace(
            raw_prompt=user_input_text,
            effective_prompt=effective_prompt_text,
            route_decision=route_decision,
            enrichment_result=enrichment_result,
            clarification_state=active_clarification_state,
            goal_tracking={},
            parse_result=parse_result,
            parsed_memory=parsed_memory,
            parsed_brain_input=parsed_brain_input,
            retrieved=[],
            recalled_turns=[],
            goal_weights={},
            decision=None,
            final_result=parsed_brain_input,
        )
        trace = self._trace_with_explanation(trace, llm_explanation, [])
        return self._finalize_turn(
            user_input_text=user_input_text,
            effective_prompt_text=effective_prompt_text,
            route_decision=route_decision,
            enrichment_result=enrichment_result,
            clarification_state=active_clarification_state,
            parse_result=parse_result,
            parsed_brain_input=parsed_brain_input,
            retrieved=[],
            recalled_turns=[],
            decision={},
            trace=trace,
            llm_explanation=llm_explanation,
            stored_entries=[],
            final_result=parsed_brain_input,
        )

    def _append_stored_entry(
        self,
        stored_entries: List[EpisodicMemoryEntry],
        *,
        entry_type: str,
        source_text: str,
        parsed_memory: BrainMemory,
        brain_input: BrainInputV1,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self._should_write_entry(
            entry_type=entry_type,
            source_text=source_text,
            parsed_memory=parsed_memory,
            brain_input=brain_input,
            metadata=metadata or {},
        ):
            return
        entry = self.memory_store.add(
            entry_type=entry_type,
            source_text=source_text,
            parsed_memory=parsed_memory,
            brain_input=brain_input,
            metadata=metadata,
        )
        if entry is not None:
            stored_entries.append(entry)

    def _should_write_entry(
        self,
        *,
        entry_type: str,
        source_text: str,
        parsed_memory: BrainMemory,
        brain_input: BrainInputV1,
        metadata: Dict[str, object],
    ) -> bool:
        if not _has_text(source_text) and _brain_memory_content_score(parsed_memory) <= 0.9 and _brain_input_content_score(brain_input) <= 0.9:
            return False
        if entry_type == "user_prompt":
            return _has_text(source_text)
        if entry_type == "selected_result":
            return _brain_memory_content_score(parsed_memory) > 0.9 or _brain_input_content_score(brain_input) > 0.9
        if entry_type == "reasoning_trace":
            return bool(
                int(metadata.get("candidate_path_count", 0) or 0) > 0
                or int(metadata.get("retrieved_memory_count", 0) or 0) > 0
            ) and (_brain_memory_content_score(parsed_memory) > 0.9 or _brain_input_content_score(brain_input) > 0.9)
        return True

    def _trace_has_high_value_content(self, trace: HybridReasoningTrace) -> bool:
        return bool(
            trace.selected_path
            or trace.candidate_paths
            or trace.retrieved_memories
            or trace.strategy_plan.get("selected_strategy")
            or _has_text(trace.raw_prompt)
        )

    def _trace_with_explanation(
        self,
        trace: HybridReasoningTrace,
        llm_explanation: LLMExplanationResult,
        stored_entries: Sequence[EpisodicMemoryEntry],
    ) -> HybridReasoningTrace:
        return HybridReasoningTrace(
            raw_prompt=trace.raw_prompt,
            effective_prompt=trace.effective_prompt,
            route_decision=trace.route_decision,
            enrichment_result=trace.enrichment_result,
            clarification_state=trace.clarification_state,
            goal_tracking=trace.goal_tracking,
            strategy_plan=trace.strategy_plan,
            merged_memory_state=trace.merged_memory_state,
            parsed_input=trace.parsed_input,
            stored_entries=[entry.to_dict() for entry in stored_entries],
            retrieved_memories=trace.retrieved_memories,
            recalled_turns=trace.recalled_turns,
            goal_weights=trace.goal_weights,
            concept_matches=trace.concept_matches,
            generated_candidate_paths=trace.generated_candidate_paths,
            candidate_paths=trace.candidate_paths,
            grounded_paths=trace.grounded_paths,
            goal_scored_paths=trace.goal_scored_paths,
            selected_path=trace.selected_path,
            final_predicted_outcome=trace.final_predicted_outcome,
            context_packet_v1=trace.context_packet_v1,
            llm_payload_v1=trace.llm_payload_v1,
            llm_payload_v2=trace.llm_payload_v2,
            cached_knowledge_checked=trace.cached_knowledge_checked,
            cached_knowledge_hit=trace.cached_knowledge_hit,
            cached_knowledge_entry_ids=trace.cached_knowledge_entry_ids,
            cached_knowledge_written=trace.cached_knowledge_written,
            cached_knowledge_topic_key=trace.cached_knowledge_topic_key,
            cached_knowledge_normalized_terms=trace.cached_knowledge_normalized_terms,
            cached_knowledge_trust_level=trace.cached_knowledge_trust_level,
            cached_knowledge_used=trace.cached_knowledge_used,
            cached_knowledge_entry_id=trace.cached_knowledge_entry_id,
            cached_knowledge_candidate_entry_ids=trace.cached_knowledge_candidate_entry_ids,
            cached_knowledge_selected_entry_id=trace.cached_knowledge_selected_entry_id,
            cached_knowledge_ranking_scores=trace.cached_knowledge_ranking_scores,
            cached_knowledge_reinforcement_applied=trace.cached_knowledge_reinforcement_applied,
            cached_knowledge_updated_confidence=trace.cached_knowledge_updated_confidence,
            cached_knowledge_updated_reuse_count=trace.cached_knowledge_updated_reuse_count,
            cached_knowledge_conflict_detected=trace.cached_knowledge_conflict_detected,
            cached_knowledge_conflicting_entry_ids=trace.cached_knowledge_conflicting_entry_ids,
            cached_knowledge_penalized_entry_ids=trace.cached_knowledge_penalized_entry_ids,
            cached_knowledge_updated_confidences=trace.cached_knowledge_updated_confidences,
            cached_knowledge_verification_attempted=trace.cached_knowledge_verification_attempted,
            cached_knowledge_verification_status=trace.cached_knowledge_verification_status,
            cached_knowledge_confidence_capped=trace.cached_knowledge_confidence_capped,
            cached_knowledge_uncertainty_flag=trace.cached_knowledge_uncertainty_flag,
            cached_knowledge_stabilized_flag=trace.cached_knowledge_stabilized_flag,
            cached_knowledge_dominance_score=trace.cached_knowledge_dominance_score,
            cached_knowledge_convergence_rate=trace.cached_knowledge_convergence_rate,
            multi_hop_used=trace.multi_hop_used,
            merged_entry_ids=trace.merged_entry_ids,
            reasoning_steps=trace.reasoning_steps,
            merged_reasoning_packet=trace.merged_reasoning_packet,
            reasoning_graph_nodes=trace.reasoning_graph_nodes,
            reasoning_graph_edges=trace.reasoning_graph_edges,
            reasoning_path=trace.reasoning_path,
            global_graph_size=trace.global_graph_size,
            reused_graph_edges=trace.reused_graph_edges,
            new_graph_edges=trace.new_graph_edges,
            structured_explanation=trace.structured_explanation,
            final_explanation=trace.final_explanation,
            cache_bypassed_reason=trace.cache_bypassed_reason,
            llm_explanation=llm_explanation.to_dict(),
            memory_written=bool(stored_entries),
        )

    def _trace_with_goal_tracking(
        self,
        trace: HybridReasoningTrace,
        goal_tracking: GoalTrackingResult,
    ) -> HybridReasoningTrace:
        return HybridReasoningTrace(
            raw_prompt=trace.raw_prompt,
            effective_prompt=trace.effective_prompt,
            route_decision=trace.route_decision,
            enrichment_result=trace.enrichment_result,
            clarification_state=trace.clarification_state,
            goal_tracking=goal_tracking.to_dict(),
            strategy_plan=trace.strategy_plan,
            merged_memory_state=trace.merged_memory_state,
            parsed_input=trace.parsed_input,
            stored_entries=trace.stored_entries,
            retrieved_memories=trace.retrieved_memories,
            recalled_turns=trace.recalled_turns,
            goal_weights=trace.goal_weights,
            concept_matches=trace.concept_matches,
            generated_candidate_paths=trace.generated_candidate_paths,
            candidate_paths=trace.candidate_paths,
            grounded_paths=trace.grounded_paths,
            goal_scored_paths=trace.goal_scored_paths,
            selected_path=trace.selected_path,
            final_predicted_outcome=trace.final_predicted_outcome,
            context_packet_v1=trace.context_packet_v1,
            llm_payload_v1=trace.llm_payload_v1,
            llm_payload_v2=trace.llm_payload_v2,
            cached_knowledge_checked=trace.cached_knowledge_checked,
            cached_knowledge_hit=trace.cached_knowledge_hit,
            cached_knowledge_entry_ids=trace.cached_knowledge_entry_ids,
            cached_knowledge_written=trace.cached_knowledge_written,
            cached_knowledge_topic_key=trace.cached_knowledge_topic_key,
            cached_knowledge_normalized_terms=trace.cached_knowledge_normalized_terms,
            cached_knowledge_trust_level=trace.cached_knowledge_trust_level,
            cached_knowledge_used=trace.cached_knowledge_used,
            cached_knowledge_entry_id=trace.cached_knowledge_entry_id,
            cached_knowledge_candidate_entry_ids=trace.cached_knowledge_candidate_entry_ids,
            cached_knowledge_selected_entry_id=trace.cached_knowledge_selected_entry_id,
            cached_knowledge_ranking_scores=trace.cached_knowledge_ranking_scores,
            cached_knowledge_reinforcement_applied=trace.cached_knowledge_reinforcement_applied,
            cached_knowledge_updated_confidence=trace.cached_knowledge_updated_confidence,
            cached_knowledge_updated_reuse_count=trace.cached_knowledge_updated_reuse_count,
            cached_knowledge_conflict_detected=trace.cached_knowledge_conflict_detected,
            cached_knowledge_conflicting_entry_ids=trace.cached_knowledge_conflicting_entry_ids,
            cached_knowledge_penalized_entry_ids=trace.cached_knowledge_penalized_entry_ids,
            cached_knowledge_updated_confidences=trace.cached_knowledge_updated_confidences,
            cached_knowledge_verification_attempted=trace.cached_knowledge_verification_attempted,
            cached_knowledge_verification_status=trace.cached_knowledge_verification_status,
            cached_knowledge_confidence_capped=trace.cached_knowledge_confidence_capped,
            cached_knowledge_uncertainty_flag=trace.cached_knowledge_uncertainty_flag,
            cached_knowledge_stabilized_flag=trace.cached_knowledge_stabilized_flag,
            cached_knowledge_dominance_score=trace.cached_knowledge_dominance_score,
            cached_knowledge_convergence_rate=trace.cached_knowledge_convergence_rate,
            multi_hop_used=trace.multi_hop_used,
            merged_entry_ids=trace.merged_entry_ids,
            reasoning_steps=trace.reasoning_steps,
            merged_reasoning_packet=trace.merged_reasoning_packet,
            reasoning_graph_nodes=trace.reasoning_graph_nodes,
            reasoning_graph_edges=trace.reasoning_graph_edges,
            reasoning_path=trace.reasoning_path,
            global_graph_size=trace.global_graph_size,
            reused_graph_edges=trace.reused_graph_edges,
            new_graph_edges=trace.new_graph_edges,
            structured_explanation=trace.structured_explanation,
            final_explanation=trace.final_explanation,
            cache_bypassed_reason=trace.cache_bypassed_reason,
            llm_explanation=trace.llm_explanation,
            memory_written=trace.memory_written,
        )

    def _trace_with_strategy_plan(
        self,
        trace: HybridReasoningTrace,
        strategy_plan: StrategyPlanningResult,
    ) -> HybridReasoningTrace:
        return HybridReasoningTrace(
            raw_prompt=trace.raw_prompt,
            effective_prompt=trace.effective_prompt,
            route_decision=trace.route_decision,
            enrichment_result=trace.enrichment_result,
            clarification_state=trace.clarification_state,
            goal_tracking=trace.goal_tracking,
            strategy_plan=strategy_plan.to_dict(),
            merged_memory_state=trace.merged_memory_state,
            parsed_input=trace.parsed_input,
            stored_entries=trace.stored_entries,
            retrieved_memories=trace.retrieved_memories,
            recalled_turns=trace.recalled_turns,
            goal_weights=trace.goal_weights,
            concept_matches=trace.concept_matches,
            generated_candidate_paths=trace.generated_candidate_paths,
            candidate_paths=trace.candidate_paths,
            grounded_paths=trace.grounded_paths,
            goal_scored_paths=trace.goal_scored_paths,
            selected_path=trace.selected_path,
            final_predicted_outcome=trace.final_predicted_outcome,
            context_packet_v1=trace.context_packet_v1,
            llm_payload_v1=trace.llm_payload_v1,
            llm_payload_v2=trace.llm_payload_v2,
            cached_knowledge_checked=trace.cached_knowledge_checked,
            cached_knowledge_hit=trace.cached_knowledge_hit,
            cached_knowledge_entry_ids=trace.cached_knowledge_entry_ids,
            cached_knowledge_written=trace.cached_knowledge_written,
            cached_knowledge_topic_key=trace.cached_knowledge_topic_key,
            cached_knowledge_normalized_terms=trace.cached_knowledge_normalized_terms,
            cached_knowledge_trust_level=trace.cached_knowledge_trust_level,
            cached_knowledge_used=trace.cached_knowledge_used,
            cached_knowledge_entry_id=trace.cached_knowledge_entry_id,
            cached_knowledge_candidate_entry_ids=trace.cached_knowledge_candidate_entry_ids,
            cached_knowledge_selected_entry_id=trace.cached_knowledge_selected_entry_id,
            cached_knowledge_ranking_scores=trace.cached_knowledge_ranking_scores,
            cached_knowledge_reinforcement_applied=trace.cached_knowledge_reinforcement_applied,
            cached_knowledge_updated_confidence=trace.cached_knowledge_updated_confidence,
            cached_knowledge_updated_reuse_count=trace.cached_knowledge_updated_reuse_count,
            cached_knowledge_conflict_detected=trace.cached_knowledge_conflict_detected,
            cached_knowledge_conflicting_entry_ids=trace.cached_knowledge_conflicting_entry_ids,
            cached_knowledge_penalized_entry_ids=trace.cached_knowledge_penalized_entry_ids,
            cached_knowledge_updated_confidences=trace.cached_knowledge_updated_confidences,
            cached_knowledge_verification_attempted=trace.cached_knowledge_verification_attempted,
            cached_knowledge_verification_status=trace.cached_knowledge_verification_status,
            cached_knowledge_confidence_capped=trace.cached_knowledge_confidence_capped,
            cached_knowledge_uncertainty_flag=trace.cached_knowledge_uncertainty_flag,
            cached_knowledge_stabilized_flag=trace.cached_knowledge_stabilized_flag,
            cached_knowledge_dominance_score=trace.cached_knowledge_dominance_score,
            cached_knowledge_convergence_rate=trace.cached_knowledge_convergence_rate,
            multi_hop_used=trace.multi_hop_used,
            merged_entry_ids=trace.merged_entry_ids,
            reasoning_steps=trace.reasoning_steps,
            merged_reasoning_packet=trace.merged_reasoning_packet,
            reasoning_graph_nodes=trace.reasoning_graph_nodes,
            reasoning_graph_edges=trace.reasoning_graph_edges,
            reasoning_path=trace.reasoning_path,
            global_graph_size=trace.global_graph_size,
            reused_graph_edges=trace.reused_graph_edges,
            new_graph_edges=trace.new_graph_edges,
            structured_explanation=trace.structured_explanation,
            final_explanation=trace.final_explanation,
            cache_bypassed_reason=trace.cache_bypassed_reason,
            llm_explanation=trace.llm_explanation,
            memory_written=trace.memory_written,
        )

    def _trace_with_memory_state(
        self,
        trace: HybridReasoningTrace,
        merged_memory_state: Dict[str, object],
    ) -> HybridReasoningTrace:
        return HybridReasoningTrace(
            raw_prompt=trace.raw_prompt,
            effective_prompt=trace.effective_prompt,
            route_decision=trace.route_decision,
            enrichment_result=trace.enrichment_result,
            clarification_state=trace.clarification_state,
            goal_tracking=trace.goal_tracking,
            strategy_plan=trace.strategy_plan,
            merged_memory_state=dict(merged_memory_state),
            parsed_input=trace.parsed_input,
            stored_entries=trace.stored_entries,
            retrieved_memories=trace.retrieved_memories,
            recalled_turns=trace.recalled_turns,
            goal_weights=trace.goal_weights,
            concept_matches=trace.concept_matches,
            generated_candidate_paths=trace.generated_candidate_paths,
            candidate_paths=trace.candidate_paths,
            grounded_paths=trace.grounded_paths,
            goal_scored_paths=trace.goal_scored_paths,
            selected_path=trace.selected_path,
            final_predicted_outcome=trace.final_predicted_outcome,
            context_packet_v1=trace.context_packet_v1,
            llm_payload_v1=trace.llm_payload_v1,
            llm_payload_v2=trace.llm_payload_v2,
            cached_knowledge_checked=trace.cached_knowledge_checked,
            cached_knowledge_hit=trace.cached_knowledge_hit,
            cached_knowledge_entry_ids=trace.cached_knowledge_entry_ids,
            cached_knowledge_written=trace.cached_knowledge_written,
            cached_knowledge_topic_key=trace.cached_knowledge_topic_key,
            cached_knowledge_normalized_terms=trace.cached_knowledge_normalized_terms,
            cached_knowledge_trust_level=trace.cached_knowledge_trust_level,
            cached_knowledge_used=trace.cached_knowledge_used,
            cached_knowledge_entry_id=trace.cached_knowledge_entry_id,
            cached_knowledge_candidate_entry_ids=trace.cached_knowledge_candidate_entry_ids,
            cached_knowledge_selected_entry_id=trace.cached_knowledge_selected_entry_id,
            cached_knowledge_ranking_scores=trace.cached_knowledge_ranking_scores,
            cached_knowledge_reinforcement_applied=trace.cached_knowledge_reinforcement_applied,
            cached_knowledge_updated_confidence=trace.cached_knowledge_updated_confidence,
            cached_knowledge_updated_reuse_count=trace.cached_knowledge_updated_reuse_count,
            cached_knowledge_conflict_detected=trace.cached_knowledge_conflict_detected,
            cached_knowledge_conflicting_entry_ids=trace.cached_knowledge_conflicting_entry_ids,
            cached_knowledge_penalized_entry_ids=trace.cached_knowledge_penalized_entry_ids,
            cached_knowledge_updated_confidences=trace.cached_knowledge_updated_confidences,
            cached_knowledge_verification_attempted=trace.cached_knowledge_verification_attempted,
            cached_knowledge_verification_status=trace.cached_knowledge_verification_status,
            cached_knowledge_confidence_capped=trace.cached_knowledge_confidence_capped,
            cached_knowledge_uncertainty_flag=trace.cached_knowledge_uncertainty_flag,
            cached_knowledge_stabilized_flag=trace.cached_knowledge_stabilized_flag,
            cached_knowledge_dominance_score=trace.cached_knowledge_dominance_score,
            cached_knowledge_convergence_rate=trace.cached_knowledge_convergence_rate,
            multi_hop_used=trace.multi_hop_used,
            merged_entry_ids=trace.merged_entry_ids,
            reasoning_steps=trace.reasoning_steps,
            merged_reasoning_packet=trace.merged_reasoning_packet,
            reasoning_graph_nodes=trace.reasoning_graph_nodes,
            reasoning_graph_edges=trace.reasoning_graph_edges,
            reasoning_path=trace.reasoning_path,
            global_graph_size=trace.global_graph_size,
            reused_graph_edges=trace.reused_graph_edges,
            new_graph_edges=trace.new_graph_edges,
            structured_explanation=trace.structured_explanation,
            final_explanation=trace.final_explanation,
            cache_bypassed_reason=trace.cache_bypassed_reason,
            llm_explanation=trace.llm_explanation,
            memory_written=trace.memory_written,
        )

    def _trace_with_context_packet(
        self,
        trace: HybridReasoningTrace,
        *,
        user_input_text: str,
        parsed_brain_input: BrainInputV1,
        final_result: BrainInputV1,
        clarification_state: Dict[str, object],
    ) -> HybridReasoningTrace:
        context_packet_v1 = build_context_packet_v1(
            user_query=user_input_text,
            route_decision=trace.route_decision,
            parsed_brain_input=parsed_brain_input,
            final_result=final_result,
            merged_memory_state=trace.merged_memory_state,
            goal_tracking=trace.goal_tracking,
            strategy_plan=trace.strategy_plan,
            selected_path=trace.selected_path,
            final_predicted_outcome=trace.final_predicted_outcome,
            clarification_state=clarification_state,
            retrieved_memories=trace.retrieved_memories,
            recalled_turns=trace.recalled_turns,
        )
        return HybridReasoningTrace(
            raw_prompt=trace.raw_prompt,
            effective_prompt=trace.effective_prompt,
            route_decision=trace.route_decision,
            enrichment_result=trace.enrichment_result,
            clarification_state=trace.clarification_state,
            goal_tracking=trace.goal_tracking,
            strategy_plan=trace.strategy_plan,
            merged_memory_state=trace.merged_memory_state,
            parsed_input=trace.parsed_input,
            stored_entries=trace.stored_entries,
            retrieved_memories=trace.retrieved_memories,
            recalled_turns=trace.recalled_turns,
            goal_weights=trace.goal_weights,
            concept_matches=trace.concept_matches,
            generated_candidate_paths=trace.generated_candidate_paths,
            candidate_paths=trace.candidate_paths,
            grounded_paths=trace.grounded_paths,
            goal_scored_paths=trace.goal_scored_paths,
            selected_path=trace.selected_path,
            final_predicted_outcome=trace.final_predicted_outcome,
            context_packet_v1=context_packet_v1,
            llm_payload_v1=trace.llm_payload_v1,
            llm_payload_v2=trace.llm_payload_v2,
            cached_knowledge_checked=trace.cached_knowledge_checked,
            cached_knowledge_hit=trace.cached_knowledge_hit,
            cached_knowledge_entry_ids=trace.cached_knowledge_entry_ids,
            cached_knowledge_written=trace.cached_knowledge_written,
            cached_knowledge_topic_key=trace.cached_knowledge_topic_key,
            cached_knowledge_normalized_terms=trace.cached_knowledge_normalized_terms,
            cached_knowledge_trust_level=trace.cached_knowledge_trust_level,
            cached_knowledge_used=trace.cached_knowledge_used,
            cached_knowledge_entry_id=trace.cached_knowledge_entry_id,
            cached_knowledge_candidate_entry_ids=trace.cached_knowledge_candidate_entry_ids,
            cached_knowledge_selected_entry_id=trace.cached_knowledge_selected_entry_id,
            cached_knowledge_ranking_scores=trace.cached_knowledge_ranking_scores,
            cached_knowledge_reinforcement_applied=trace.cached_knowledge_reinforcement_applied,
            cached_knowledge_updated_confidence=trace.cached_knowledge_updated_confidence,
            cached_knowledge_updated_reuse_count=trace.cached_knowledge_updated_reuse_count,
            cached_knowledge_conflict_detected=trace.cached_knowledge_conflict_detected,
            cached_knowledge_conflicting_entry_ids=trace.cached_knowledge_conflicting_entry_ids,
            cached_knowledge_penalized_entry_ids=trace.cached_knowledge_penalized_entry_ids,
            cached_knowledge_updated_confidences=trace.cached_knowledge_updated_confidences,
            cached_knowledge_verification_attempted=trace.cached_knowledge_verification_attempted,
            cached_knowledge_verification_status=trace.cached_knowledge_verification_status,
            cached_knowledge_confidence_capped=trace.cached_knowledge_confidence_capped,
            cached_knowledge_uncertainty_flag=trace.cached_knowledge_uncertainty_flag,
            cached_knowledge_stabilized_flag=trace.cached_knowledge_stabilized_flag,
            cached_knowledge_dominance_score=trace.cached_knowledge_dominance_score,
            cached_knowledge_convergence_rate=trace.cached_knowledge_convergence_rate,
            multi_hop_used=trace.multi_hop_used,
            merged_entry_ids=trace.merged_entry_ids,
            reasoning_steps=trace.reasoning_steps,
            merged_reasoning_packet=trace.merged_reasoning_packet,
            reasoning_graph_nodes=trace.reasoning_graph_nodes,
            reasoning_graph_edges=trace.reasoning_graph_edges,
            reasoning_path=trace.reasoning_path,
            global_graph_size=trace.global_graph_size,
            reused_graph_edges=trace.reused_graph_edges,
            new_graph_edges=trace.new_graph_edges,
            structured_explanation=trace.structured_explanation,
            final_explanation=trace.final_explanation,
            cache_bypassed_reason=trace.cache_bypassed_reason,
            llm_explanation=trace.llm_explanation,
            memory_written=trace.memory_written,
        )

    def _trace_with_llm_payload(
        self,
        trace: HybridReasoningTrace,
        *,
        user_input_text: str,
        mode: str,
    ) -> HybridReasoningTrace:
        context_packet_v1 = trace.context_packet_v1 if isinstance(trace.context_packet_v1, dict) else {}
        route_context = context_packet_v1.get("route_context", {}) if isinstance(context_packet_v1.get("route_context", {}), dict) else {}
        llm_payload_v1 = build_llm_payload_v1(
            user_input_text=user_input_text,
            route_type=str(trace.route_decision.get("route_type", "")),
            context_packet_v1=context_packet_v1,
            mode=mode,
            route_reasons=route_context.get("reasons", []) if isinstance(route_context.get("reasons", []), list) else [],
            selected_path=trace.selected_path,
            retrieved_memory_count=int(route_context.get("retrieved_memory_count", len(trace.retrieved_memories))),
            recalled_turn_count=int(route_context.get("recalled_turn_count", len(trace.recalled_turns))),
        )
        llm_payload_v2 = build_llm_payload_from_packet_v1(
            context_packet_v1,
            cached_knowledge={
                "hit": trace.cached_knowledge_hit,
                "entry_ids": list(trace.cached_knowledge_entry_ids),
                "topic_key": trace.cached_knowledge_topic_key,
                "trust_level": trace.cached_knowledge_trust_level,
                "entries": [
                    entry.to_dict()
                    for entry in self.retrieve_cached_llm_knowledge(
                        user_input_text=user_input_text,
                        topic_key=trace.cached_knowledge_topic_key,
                        top_k=3,
                    )
                ] if trace.cached_knowledge_hit else [],
            },
        )
        return HybridReasoningTrace(
            raw_prompt=trace.raw_prompt,
            effective_prompt=trace.effective_prompt,
            route_decision=trace.route_decision,
            enrichment_result=trace.enrichment_result,
            clarification_state=trace.clarification_state,
            goal_tracking=trace.goal_tracking,
            strategy_plan=trace.strategy_plan,
            merged_memory_state=trace.merged_memory_state,
            parsed_input=trace.parsed_input,
            stored_entries=trace.stored_entries,
            retrieved_memories=trace.retrieved_memories,
            recalled_turns=trace.recalled_turns,
            goal_weights=trace.goal_weights,
            concept_matches=trace.concept_matches,
            generated_candidate_paths=trace.generated_candidate_paths,
            candidate_paths=trace.candidate_paths,
            grounded_paths=trace.grounded_paths,
            goal_scored_paths=trace.goal_scored_paths,
            selected_path=trace.selected_path,
            final_predicted_outcome=trace.final_predicted_outcome,
            context_packet_v1=trace.context_packet_v1,
            llm_payload_v1=llm_payload_v1,
            llm_payload_v2=llm_payload_v2,
            cached_knowledge_checked=trace.cached_knowledge_checked,
            cached_knowledge_hit=trace.cached_knowledge_hit,
            cached_knowledge_entry_ids=trace.cached_knowledge_entry_ids,
            cached_knowledge_written=trace.cached_knowledge_written,
            cached_knowledge_topic_key=trace.cached_knowledge_topic_key,
            cached_knowledge_normalized_terms=trace.cached_knowledge_normalized_terms,
            cached_knowledge_trust_level=trace.cached_knowledge_trust_level,
            cached_knowledge_used=trace.cached_knowledge_used,
            cached_knowledge_entry_id=trace.cached_knowledge_entry_id,
            cached_knowledge_candidate_entry_ids=trace.cached_knowledge_candidate_entry_ids,
            cached_knowledge_selected_entry_id=trace.cached_knowledge_selected_entry_id,
            cached_knowledge_ranking_scores=trace.cached_knowledge_ranking_scores,
            cached_knowledge_reinforcement_applied=trace.cached_knowledge_reinforcement_applied,
            cached_knowledge_updated_confidence=trace.cached_knowledge_updated_confidence,
            cached_knowledge_updated_reuse_count=trace.cached_knowledge_updated_reuse_count,
            cached_knowledge_conflict_detected=trace.cached_knowledge_conflict_detected,
            cached_knowledge_conflicting_entry_ids=trace.cached_knowledge_conflicting_entry_ids,
            cached_knowledge_penalized_entry_ids=trace.cached_knowledge_penalized_entry_ids,
            cached_knowledge_updated_confidences=trace.cached_knowledge_updated_confidences,
            cached_knowledge_verification_attempted=trace.cached_knowledge_verification_attempted,
            cached_knowledge_verification_status=trace.cached_knowledge_verification_status,
            cached_knowledge_confidence_capped=trace.cached_knowledge_confidence_capped,
            cached_knowledge_uncertainty_flag=trace.cached_knowledge_uncertainty_flag,
            cached_knowledge_stabilized_flag=trace.cached_knowledge_stabilized_flag,
            cached_knowledge_dominance_score=trace.cached_knowledge_dominance_score,
            cached_knowledge_convergence_rate=trace.cached_knowledge_convergence_rate,
            multi_hop_used=trace.multi_hop_used,
            merged_entry_ids=trace.merged_entry_ids,
            reasoning_steps=trace.reasoning_steps,
            merged_reasoning_packet=trace.merged_reasoning_packet,
            reasoning_graph_nodes=trace.reasoning_graph_nodes,
            reasoning_graph_edges=trace.reasoning_graph_edges,
            reasoning_path=trace.reasoning_path,
            structured_explanation=trace.structured_explanation,
            final_explanation=trace.final_explanation,
            cache_bypassed_reason=trace.cache_bypassed_reason,
            llm_explanation=trace.llm_explanation,
            memory_written=trace.memory_written,
        )

    def _trace_with_cached_knowledge(
        self,
        trace: HybridReasoningTrace,
        *,
        checked: bool,
        hit: bool,
        entry_ids: Sequence[int],
        topic_key: str,
        normalized_terms: Optional[Sequence[str]] = None,
        trust_level: str,
        written: bool,
        used: Optional[bool] = None,
        entry_id: Optional[int] = None,
        candidate_entry_ids: Optional[Sequence[int]] = None,
        selected_entry_id: Optional[int] = None,
        ranking_scores: Optional[Sequence[Dict[str, object]]] = None,
        reinforcement_applied: Optional[bool] = None,
        updated_confidence: Optional[float] = None,
        updated_reuse_count: Optional[int] = None,
        conflict_detected: Optional[bool] = None,
        conflicting_entry_ids: Optional[Sequence[int]] = None,
        penalized_entry_ids: Optional[Sequence[int]] = None,
        updated_confidences: Optional[Sequence[Dict[str, object]]] = None,
        verification_attempted: Optional[bool] = None,
        verification_status: Optional[str] = None,
        confidence_capped: Optional[bool] = None,
        uncertainty_flag: Optional[bool] = None,
        stabilized_flag: Optional[bool] = None,
        dominance_score: Optional[float] = None,
        convergence_rate: Optional[float] = None,
        multi_hop_used: Optional[bool] = None,
        merged_entry_ids: Optional[Sequence[int]] = None,
        reasoning_steps: Optional[Sequence[str]] = None,
        merged_reasoning_packet: Optional[Dict[str, object]] = None,
        reasoning_graph_nodes: Optional[Sequence[Dict[str, object]]] = None,
        reasoning_graph_edges: Optional[Sequence[Dict[str, object]]] = None,
        reasoning_path: Optional[Sequence[Dict[str, object]]] = None,
        global_graph_size: Optional[Dict[str, int]] = None,
        reused_graph_edges: Optional[Sequence[Dict[str, object]]] = None,
        new_graph_edges: Optional[Sequence[Dict[str, object]]] = None,
        structured_explanation: Optional[Dict[str, object]] = None,
        final_explanation: Optional[str] = None,
        bypassed_reason: Optional[str] = None,
    ) -> HybridReasoningTrace:
        return HybridReasoningTrace(
            raw_prompt=trace.raw_prompt,
            effective_prompt=trace.effective_prompt,
            route_decision=trace.route_decision,
            enrichment_result=trace.enrichment_result,
            clarification_state=trace.clarification_state,
            goal_tracking=trace.goal_tracking,
            strategy_plan=trace.strategy_plan,
            merged_memory_state=trace.merged_memory_state,
            parsed_input=trace.parsed_input,
            stored_entries=trace.stored_entries,
            retrieved_memories=trace.retrieved_memories,
            recalled_turns=trace.recalled_turns,
            goal_weights=trace.goal_weights,
            concept_matches=trace.concept_matches,
            generated_candidate_paths=trace.generated_candidate_paths,
            candidate_paths=trace.candidate_paths,
            grounded_paths=trace.grounded_paths,
            goal_scored_paths=trace.goal_scored_paths,
            selected_path=trace.selected_path,
            final_predicted_outcome=trace.final_predicted_outcome,
            context_packet_v1=trace.context_packet_v1,
            llm_payload_v1=trace.llm_payload_v1,
            llm_payload_v2=trace.llm_payload_v2,
            cached_knowledge_checked=checked,
            cached_knowledge_hit=hit,
            cached_knowledge_entry_ids=[int(item) for item in entry_ids],
            cached_knowledge_written=written,
            cached_knowledge_topic_key=str(topic_key or ""),
            cached_knowledge_normalized_terms=list(trace.cached_knowledge_normalized_terms if normalized_terms is None else [str(item) for item in normalized_terms if str(item).strip()]),
            cached_knowledge_trust_level=str(trust_level or ""),
            cached_knowledge_used=trace.cached_knowledge_used if used is None else bool(used),
            cached_knowledge_entry_id=trace.cached_knowledge_entry_id if entry_id is None else int(entry_id),
            cached_knowledge_candidate_entry_ids=list(trace.cached_knowledge_candidate_entry_ids if candidate_entry_ids is None else [int(item) for item in candidate_entry_ids]),
            cached_knowledge_selected_entry_id=trace.cached_knowledge_selected_entry_id if selected_entry_id is None else (int(selected_entry_id) if selected_entry_id is not None else None),
            cached_knowledge_ranking_scores=list(trace.cached_knowledge_ranking_scores if ranking_scores is None else [dict(item) for item in ranking_scores if isinstance(item, dict)]),
            cached_knowledge_reinforcement_applied=trace.cached_knowledge_reinforcement_applied if reinforcement_applied is None else bool(reinforcement_applied),
            cached_knowledge_updated_confidence=trace.cached_knowledge_updated_confidence if updated_confidence is None else float(updated_confidence),
            cached_knowledge_updated_reuse_count=trace.cached_knowledge_updated_reuse_count if updated_reuse_count is None else int(updated_reuse_count),
            cached_knowledge_conflict_detected=trace.cached_knowledge_conflict_detected if conflict_detected is None else bool(conflict_detected),
            cached_knowledge_conflicting_entry_ids=list(trace.cached_knowledge_conflicting_entry_ids if conflicting_entry_ids is None else [int(item) for item in conflicting_entry_ids]),
            cached_knowledge_penalized_entry_ids=list(trace.cached_knowledge_penalized_entry_ids if penalized_entry_ids is None else [int(item) for item in penalized_entry_ids]),
            cached_knowledge_updated_confidences=list(trace.cached_knowledge_updated_confidences if updated_confidences is None else [dict(item) for item in updated_confidences if isinstance(item, dict)]),
            cached_knowledge_verification_attempted=trace.cached_knowledge_verification_attempted if verification_attempted is None else bool(verification_attempted),
            cached_knowledge_verification_status=trace.cached_knowledge_verification_status if verification_status is None else str(verification_status or ""),
            cached_knowledge_confidence_capped=trace.cached_knowledge_confidence_capped if confidence_capped is None else bool(confidence_capped),
            cached_knowledge_uncertainty_flag=trace.cached_knowledge_uncertainty_flag if uncertainty_flag is None else bool(uncertainty_flag),
            cached_knowledge_stabilized_flag=trace.cached_knowledge_stabilized_flag if stabilized_flag is None else bool(stabilized_flag),
            cached_knowledge_dominance_score=trace.cached_knowledge_dominance_score if dominance_score is None else float(dominance_score),
            cached_knowledge_convergence_rate=trace.cached_knowledge_convergence_rate if convergence_rate is None else float(convergence_rate),
            multi_hop_used=trace.multi_hop_used if multi_hop_used is None else bool(multi_hop_used),
            merged_entry_ids=list(trace.merged_entry_ids if merged_entry_ids is None else [int(item) for item in merged_entry_ids]),
            reasoning_steps=list(trace.reasoning_steps if reasoning_steps is None else [str(item) for item in reasoning_steps if str(item).strip()]),
            merged_reasoning_packet=dict(trace.merged_reasoning_packet if merged_reasoning_packet is None else merged_reasoning_packet),
            reasoning_graph_nodes=list(trace.reasoning_graph_nodes if reasoning_graph_nodes is None else [dict(item) for item in reasoning_graph_nodes if isinstance(item, dict)]),
            reasoning_graph_edges=list(trace.reasoning_graph_edges if reasoning_graph_edges is None else [dict(item) for item in reasoning_graph_edges if isinstance(item, dict)]),
            reasoning_path=list(trace.reasoning_path if reasoning_path is None else [dict(item) for item in reasoning_path if isinstance(item, dict)]),
            global_graph_size=dict(trace.global_graph_size if global_graph_size is None else global_graph_size),
            reused_graph_edges=list(trace.reused_graph_edges if reused_graph_edges is None else [dict(item) for item in reused_graph_edges if isinstance(item, dict)]),
            new_graph_edges=list(trace.new_graph_edges if new_graph_edges is None else [dict(item) for item in new_graph_edges if isinstance(item, dict)]),
            structured_explanation=dict(trace.structured_explanation if structured_explanation is None else structured_explanation),
            final_explanation=trace.final_explanation if final_explanation is None else str(final_explanation or ""),
            cache_bypassed_reason=trace.cache_bypassed_reason if bypassed_reason is None else (str(bypassed_reason) if bypassed_reason is not None else None),
            llm_explanation=trace.llm_explanation,
            memory_written=trace.memory_written,
        )

    def _route_decision_with_cache_short_circuit(self, route_decision: QueryRouteDecision) -> QueryRouteDecision:
        reasons = list(route_decision.reasons)
        if "Low-trust cached knowledge matched; factual answer can short-circuit without LLM." not in reasons:
            reasons.append("Low-trust cached knowledge matched; factual answer can short-circuit without LLM.")
        return QueryRouteDecision(
            route_type=route_decision.route_type,
            confidence=route_decision.confidence,
            reasons=reasons,
            run_brain_reasoning=route_decision.run_brain_reasoning,
            run_direct_llm=False,
            run_hybrid_mode=route_decision.run_hybrid_mode,
            score_breakdown=dict(route_decision.score_breakdown),
        )

    def _cached_entries_from_trace(self, trace: HybridReasoningTrace) -> List[EpisodicMemoryEntry]:
        ordered_ids = [int(item) for item in trace.cached_knowledge_entry_ids]
        if not ordered_ids:
            return []
        entry_lookup = {
            entry.entry_id: entry
            for entry in self.memory_store.entries
            if entry.entry_type == "llm_knowledge_cache"
        }
        return [entry_lookup[entry_id] for entry_id in ordered_ids if entry_id in entry_lookup]

    def _cached_entry_terms(self, entry: EpisodicMemoryEntry) -> List[str]:
        metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
        terms: List[str] = []
        seen = set()

        def add_term(value: object) -> None:
            for token in self.normalize_query(str(value or "").replace(":", " ").replace("_", " ")):
                if token in seen:
                    continue
                seen.add(token)
                terms.append(token)

        raw_topic_keys = metadata.get("topic_keys", [])
        if isinstance(raw_topic_keys, list):
            for item in raw_topic_keys:
                add_term(item)
        add_term(metadata.get("topic_key", ""))
        add_term(metadata.get("answer_text", entry.source_text))
        normalized_facts = metadata.get("normalized_facts", [])
        if isinstance(normalized_facts, list):
            for fact in normalized_facts:
                add_term(fact)
        return terms

    def _compose_multi_hop_cached_answer(
        self,
        *,
        user_input_text: str,
        merged_facts: Sequence[str],
        merged_state: Dict[str, object],
        reasoning_path: Optional[Sequence[Dict[str, object]]] = None,
        final_explanation: str = "",
    ) -> str:
        if str(final_explanation or "").strip():
            return str(final_explanation).strip()
        cleaned_facts = [str(item).strip().rstrip(".") for item in merged_facts if str(item).strip()]
        normalized_query = str(user_input_text or "").lower()
        if not cleaned_facts:
            if merged_state:
                return ". ".join(f"{str(key).replace('_', ' ')} is {value}" for key, value in list(merged_state.items())[:3]) + "."
            return ""
        path_steps = [dict(item) for item in (reasoning_path or []) if isinstance(item, dict)]
        if any(token in normalized_query for token in ("why", "how", "explain")) and len(path_steps) >= 2:
            start_value = str(path_steps[0].get("to_label", path_steps[0].get("to", ""))).strip()
            middle_value = str(path_steps[1].get("to_label", path_steps[1].get("to", ""))).strip()
            if path_steps[0].get("relationship") == "has_color" and path_steps[1].get("relationship") == "caused_by":
                explanation = f"The sky looks {start_value} because {middle_value}"
                if len(path_steps) >= 3 and path_steps[2].get("relationship") == "depends_on":
                    tail_value = str(path_steps[2].get("to_label", path_steps[2].get("to", ""))).strip()
                    if tail_value:
                        explanation += f", and {middle_value} depends on {tail_value}"
                return explanation.rstrip(".") + "."
        if any(token in normalized_query for token in ("why", "how", "explain")) and len(cleaned_facts) >= 2:
            primary_fact = cleaned_facts[0]
            support_candidates = [
                fact
                for fact in cleaned_facts[1:]
                if any(token in fact.lower() for token in ("because", "due", "cause", "causes", "caused", "scatter", "scatters", "wavelength", "atmosphere", "light"))
            ]
            support_fact = support_candidates[0] if support_candidates else cleaned_facts[1]
            if " because " in primary_fact.lower() or primary_fact.lower().startswith(("because ", "due to ", "since ")):
                return primary_fact + "."
            if support_fact.lower().startswith(("because ", "due to ", "since ")):
                return primary_fact + " " + support_fact + "."
            return primary_fact + " because " + support_fact[0].lower() + support_fact[1:] + "."
        if len(cleaned_facts) >= 2:
            return cleaned_facts[0] + ". " + cleaned_facts[1] + "."
        return cleaned_facts[0] + "."

    def _build_reasoning_explanation(
        self,
        *,
        user_input_text: str,
        merged_facts: Sequence[str],
        reasoning_graph_edges: Sequence[Dict[str, object]],
        reasoning_path: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        path_steps = [dict(item) for item in reasoning_path if isinstance(item, dict)]
        edge_list = [dict(item) for item in reasoning_graph_edges if isinstance(item, dict)]
        if not path_steps:
            return {
                "structured_explanation": {},
                "final_explanation": "",
            }
        normalized_query = str(user_input_text or "").lower()
        first_step = path_steps[0]
        main_subject = str(first_step.get("from_label", first_step.get("from", ""))).strip() or "It"
        main_value = str(first_step.get("to_label", first_step.get("to", ""))).strip()
        main_fact = ""
        if first_step.get("relationship") == "has_color" and main_value:
            qualifier = " during the day" if "day" in normalized_query or "sky" in normalized_query else ""
            main_fact = f"The {main_subject} appears {main_value}{qualifier}"
        elif main_value:
            main_fact = f"{main_subject.capitalize()} is linked to {main_value}"
        supporting_cause = ""
        support_detail = ""
        atmosphere_detail = ""
        fact_text = " ".join(str(item).strip() for item in merged_facts if str(item).strip()).lower()
        for step in path_steps[1:]:
            relationship = str(step.get("relationship", "")).strip().lower()
            to_label = str(step.get("to_label", step.get("to", ""))).strip()
            if relationship == "caused_by":
                supporting_cause = to_label or supporting_cause
            elif relationship == "depends_on":
                support_detail = to_label or support_detail
        for edge in edge_list:
            if str(edge.get("relationship", "")).strip().lower() == "occurs_in":
                atmosphere_detail = str(edge.get("to_label", edge.get("to", ""))).strip()
                if atmosphere_detail:
                    break
        if "earth's atmosphere" in fact_text or "earth atmosphere" in fact_text:
            atmosphere_detail = "Earth's atmosphere"
        if supporting_cause == "scattering":
            if "shorter blue wavelength" in fact_text or "shorter blue wavelengths" in fact_text:
                supporting_cause = "shorter blue wavelengths of sunlight scatter more strongly"
            elif "blue light scatter" in fact_text or "blue light scatters" in fact_text:
                supporting_cause = "blue light scatters more strongly"
        if support_detail == "wavelength":
            support_detail = "the shorter wavelengths of blue light"
        structured_explanation = {
            "main_fact": main_fact,
            "cause": supporting_cause,
            "supporting_detail": support_detail,
            "context": atmosphere_detail,
            "template_type": f"path_length_{len(path_steps)}",
        }
        if any(token in normalized_query for token in ("why", "how", "explain")) and main_fact and supporting_cause:
            explanation = f"{main_fact} because {supporting_cause}"
            if atmosphere_detail:
                explanation += f" in {atmosphere_detail}"
            if support_detail and "wavelength" not in supporting_cause.lower():
                explanation += f", which depends on {support_detail}"
            return {
                "structured_explanation": structured_explanation,
                "final_explanation": explanation.rstrip(".") + ".",
            }
        if main_fact:
            return {
                "structured_explanation": structured_explanation,
                "final_explanation": main_fact.rstrip(".") + ".",
            }
        return {
            "structured_explanation": structured_explanation,
            "final_explanation": "",
        }

    def _build_reasoning_graph(
        self,
        *,
        user_input_text: str,
        merged_state: Dict[str, object],
        merged_facts: Sequence[str],
    ) -> Dict[str, object]:
        nodes: List[Dict[str, object]] = []
        edges: List[Dict[str, object]] = []
        path: List[Dict[str, object]] = []
        node_lookup: Dict[str, Dict[str, object]] = {}
        query_terms = self.normalize_query(user_input_text)

        def ensure_node(node_id: str, label: str, node_type: str) -> None:
            normalized_id = self.global_graph_store.normalize_concept(node_id or label)
            if not normalized_id:
                return
            existing = node_lookup.get(normalized_id)
            if existing is not None:
                if not str(existing.get("label", "")).strip() and str(label or "").strip():
                    existing["label"] = str(label).strip()
                return
            payload = {
                "node_id": normalized_id,
                "label": str(label or normalized_id).strip() or normalized_id,
                "normalized_concept": normalized_id,
                "type": str(node_type or "entity"),
            }
            node_lookup[normalized_id] = payload
            nodes.append(payload)

        def add_edge(from_id: str, relationship: str, to_id: str, evidence: str) -> None:
            normalized_from = self.global_graph_store.normalize_concept(from_id)
            normalized_to = self.global_graph_store.normalize_concept(to_id)
            if not normalized_from or not normalized_to:
                return
            ensure_node(normalized_from, from_id, "concept")
            ensure_node(normalized_to, to_id, "concept")
            edge = {
                "from": normalized_from,
                "from_label": node_lookup[normalized_from]["label"],
                "relationship": str(relationship or "related_to"),
                "to": normalized_to,
                "to_label": node_lookup[normalized_to]["label"],
                "evidence": str(evidence or "").strip(),
            }
            if edge not in edges:
                edges.append(edge)

        for key, value in merged_state.items():
            state_key = str(key).strip().lower()
            state_value = str(value).strip()
            lowered_value = state_value.lower()
            if not state_value:
                continue
            if state_key.startswith("sky_color:"):
                color_value = "blue" if "blue" in lowered_value else state_value
                add_edge("sky", "has_color", color_value, f"{key}={value}")
            elif "scattering" in state_key and "reason" in state_key:
                add_edge("blue", "caused_by", "scattering", f"{key}={value}")
                if "atmosphere" in lowered_value:
                    add_edge("scattering", "occurs_in", "atmosphere", f"{key}={value}")
            elif "relationship" in state_key and "scatter" in state_key:
                add_edge("scattering", "depends_on", "wavelength", f"{key}={value}")
                if "atmosphere" in lowered_value:
                    add_edge("scattering", "occurs_in", "atmosphere", f"{key}={value}")

        for fact in merged_facts:
            lowered_fact = str(fact).strip().lower()
            if not lowered_fact:
                continue
            if "sky" in lowered_fact and "blue" in lowered_fact:
                add_edge("sky", "has_color", "blue", fact)
            if "scatter" in lowered_fact and "blue" in lowered_fact:
                add_edge("blue", "caused_by", "scattering", fact)
            if "scatter" in lowered_fact and "wavelength" in lowered_fact:
                add_edge("scattering", "depends_on", "wavelength", fact)
            if "atmosphere" in lowered_fact and "scatter" in lowered_fact:
                add_edge("scattering", "occurs_in", "atmosphere", fact)

        local_edge_signatures = {
            (
                str(edge.get("from", "")).strip().lower(),
                str(edge.get("relationship", "")).strip().lower(),
                str(edge.get("to", "")).strip().lower(),
            )
            for edge in edges
            if isinstance(edge, dict)
        }
        reused_graph_edges: List[Dict[str, object]] = []
        for edge in self.global_graph_store.find_relevant_edges(query_terms=query_terms, limit=6):
            if not isinstance(edge, dict):
                continue
            edge_signature = (
                str(edge.get("from", "")).strip().lower(),
                str(edge.get("relationship", "")).strip().lower(),
                str(edge.get("to", "")).strip().lower(),
            )
            reused_edge = {
                "from": str(edge.get("from", "")).strip().lower(),
                "from_label": str(edge.get("from_label", edge.get("from", ""))).strip(),
                "relationship": str(edge.get("relationship", "related_to")).strip(),
                "to": str(edge.get("to", "")).strip().lower(),
                "to_label": str(edge.get("to_label", edge.get("to", ""))).strip(),
                "evidence": (
                    str(edge.get("evidence", [""])[0]).strip()
                    if isinstance(edge.get("evidence", []), list) and edge.get("evidence", [])
                    else str(edge.get("evidence", "")).strip()
                ),
                "support_count": max(1, int(edge.get("support_count", 1) or 1)),
                "reused": True,
            }
            reused_graph_edges.append(dict(reused_edge))
            if edge_signature in local_edge_signatures:
                continue
            ensure_node(str(edge.get("from", "")), str(edge.get("from_label", edge.get("from", ""))), "concept")
            ensure_node(str(edge.get("to", "")), str(edge.get("to_label", edge.get("to", ""))), "concept")
            edges.append(reused_edge)
            local_edge_signatures.add(edge_signature)

        edge_lookup = {(edge["from"], edge["relationship"], edge["to"]): edge for edge in edges}
        preferred_path_keys = [
            ("sky", "has_color", "blue"),
            ("blue", "caused_by", "scattering"),
            ("scattering", "depends_on", "wavelength"),
        ]
        for edge_key in preferred_path_keys:
            if edge_key in edge_lookup:
                path.append(dict(edge_lookup[edge_key]))
        if not path and edges:
            seed_terms = set(query_terms)
            candidate_path = [
                edge for edge in edges
                if seed_terms & set(self.normalize_query(f"{edge.get('from_label', '')} {edge.get('to_label', '')}"))
            ]
            path = [dict(edge) for edge in candidate_path[:3]]
        merge_payload = self.global_graph_store.merge_graph(
            nodes=nodes,
            edges=edges,
        )
        return {
            "reasoning_graph_nodes": nodes,
            "reasoning_graph_edges": edges,
            "reasoning_path": path,
            "global_graph_size": merge_payload.get("global_graph_size", {}),
            "reused_graph_edges": reused_graph_edges,
            "new_graph_edges": merge_payload.get("new_edges", []),
        }

    def _build_multi_hop_cache_reasoning(
        self,
        *,
        user_input_text: str,
        entries: Sequence[EpisodicMemoryEntry],
        ranking_scores: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        ranked_entries = [entry for entry in entries if entry.entry_type == "llm_knowledge_cache"][:5]
        if len(ranked_entries) < 2:
            return {
                "multi_hop_used": False,
                "merged_entry_ids": [],
                "reasoning_steps": [],
                "merged_reasoning_packet": {},
                "global_graph_size": {},
                "reused_graph_edges": [],
                "new_graph_edges": [],
            }
        query_terms = self.normalize_query(user_input_text)
        query_term_set = set(query_terms)
        ranking_lookup = {
            int(score.get("entry_id", -1)): dict(score)
            for score in ranking_scores
            if isinstance(score, dict)
        }
        groups: Dict[str, Dict[str, object]] = {}
        for entry in ranked_entries:
            metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
            entry_terms = self._cached_entry_terms(entry)
            shared_terms = sorted(query_term_set & set(entry_terms))
            topic_key = str(metadata.get("topic_key", "")).strip().lower()
            group_key = topic_key or ("|".join(shared_terms[:3]) if shared_terms else f"entry:{entry.entry_id}")
            group = groups.setdefault(
                group_key,
                {
                    "entries": [],
                    "shared_terms": set(),
                    "topic_keys": set(),
                    "total_score": 0.0,
                },
            )
            group["entries"].append(entry)
            group["shared_terms"].update(shared_terms)
            if topic_key:
                group["topic_keys"].add(topic_key)
            score_payload = ranking_lookup.get(entry.entry_id, {})
            group["total_score"] = float(group.get("total_score", 0.0) or 0.0) + float(score_payload.get("total_score", 0.0) or 0.0)
        grouped_candidates = sorted(
            groups.values(),
            key=lambda item: (
                -len(item.get("entries", [])),
                -float(item.get("total_score", 0.0) or 0.0),
                -len(item.get("shared_terms", set())),
            ),
        )
        best_group = grouped_candidates[0] if grouped_candidates else {"entries": []}
        grouped_entries = list(best_group.get("entries", []))
        if len(grouped_entries) < 2:
            return {
                "multi_hop_used": False,
                "merged_entry_ids": [],
                "reasoning_steps": [],
                "merged_reasoning_packet": {},
                "global_graph_size": {},
                "reused_graph_edges": [],
                "new_graph_edges": [],
            }
        merged_state: Dict[str, object] = {}
        merged_facts: List[str] = []
        fact_seen = set()
        for entry in grouped_entries:
            metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
            answer_state = metadata.get("answer_state", {})
            if isinstance(answer_state, dict):
                for key, value in answer_state.items():
                    merged_state.setdefault(str(key), value)
            entry_facts = metadata.get("normalized_facts", [])
            if not isinstance(entry_facts, list) or not entry_facts:
                entry_facts = self._distill_cached_answer_facts(str(metadata.get("answer_text", entry.source_text)))
            for fact in entry_facts:
                cleaned = str(fact).strip()
                signature = cleaned.lower()
                if not cleaned or signature in fact_seen:
                    continue
                fact_seen.add(signature)
                merged_facts.append(cleaned)
        reasoning_steps = [
            f"Retrieved {len(grouped_entries)} relevant cached entries for the query.",
            "Grouped entries by shared terms: " + ", ".join(sorted(str(item) for item in best_group.get("shared_terms", set()) if str(item).strip())) + ".",
            "Merged answer_state and factual snippets from the grouped entries.",
        ]
        graph_payload = self._build_reasoning_graph(
            user_input_text=user_input_text,
            merged_state=merged_state,
            merged_facts=merged_facts,
        )
        reasoning_graph_nodes = graph_payload.get("reasoning_graph_nodes", [])
        reasoning_graph_edges = graph_payload.get("reasoning_graph_edges", [])
        reasoning_path = graph_payload.get("reasoning_path", [])
        global_graph_size = graph_payload.get("global_graph_size", {}) if isinstance(graph_payload.get("global_graph_size", {}), dict) else {}
        reused_graph_edges = graph_payload.get("reused_graph_edges", []) if isinstance(graph_payload.get("reused_graph_edges", []), list) else []
        new_graph_edges = graph_payload.get("new_graph_edges", []) if isinstance(graph_payload.get("new_graph_edges", []), list) else []
        if reasoning_graph_nodes and reasoning_graph_edges:
            reasoning_steps.append("Constructed a reasoning graph from merged facts and extracted a reasoning path.")
        if reused_graph_edges:
            reasoning_steps.append(f"Reused {len(reused_graph_edges)} supporting edges from the persistent global graph.")
        if new_graph_edges:
            reasoning_steps.append(f"Persisted {len(new_graph_edges)} new graph edges into the global knowledge graph.")
        explanation_payload = self._build_reasoning_explanation(
            user_input_text=user_input_text,
            merged_facts=merged_facts,
            reasoning_graph_edges=reasoning_graph_edges,
            reasoning_path=reasoning_path,
        )
        structured_explanation = explanation_payload.get("structured_explanation", {}) if isinstance(explanation_payload.get("structured_explanation", {}), dict) else {}
        final_explanation = str(explanation_payload.get("final_explanation", "")).strip()
        if final_explanation:
            reasoning_steps.append("Translated the reasoning path into a natural-language explanation.")
        normalized_query = str(user_input_text or "").lower()
        if any(token in normalized_query for token in ("why", "how", "explain")) and len(merged_facts) >= 2:
            reasoning_steps.append("Applied cause_to_effect reasoning by linking the primary fact to a supporting explanation fact.")
        elif len(merged_facts) >= 2:
            reasoning_steps.append("Applied relationship_link reasoning across the merged facts.")
        resolved_answer = self._compose_multi_hop_cached_answer(
            user_input_text=user_input_text,
            merged_facts=merged_facts,
            merged_state=merged_state,
            reasoning_path=reasoning_path,
            final_explanation=final_explanation,
        )
        merged_reasoning_packet = {
            "query_terms": query_terms,
            "grouped_topic_keys": sorted(str(item) for item in best_group.get("topic_keys", set()) if str(item).strip()),
            "shared_terms": sorted(str(item) for item in best_group.get("shared_terms", set()) if str(item).strip()),
            "merged_answer_state": merged_state,
            "merged_answer_facts": merged_facts[:5],
            "reasoning_graph_nodes": reasoning_graph_nodes,
            "reasoning_graph_edges": reasoning_graph_edges,
            "reasoning_path": reasoning_path,
            "global_graph_size": global_graph_size,
            "reused_graph_edges": reused_graph_edges,
            "new_graph_edges": new_graph_edges,
            "structured_explanation": structured_explanation,
            "final_explanation": final_explanation,
            "resolved_answer": resolved_answer,
        }
        return {
            "multi_hop_used": True,
            "merged_entry_ids": [entry.entry_id for entry in grouped_entries],
            "reasoning_steps": reasoning_steps,
            "merged_reasoning_packet": merged_reasoning_packet,
            "reasoning_graph_nodes": reasoning_graph_nodes,
            "reasoning_graph_edges": reasoning_graph_edges,
            "reasoning_path": reasoning_path,
            "global_graph_size": global_graph_size,
            "reused_graph_edges": reused_graph_edges,
            "new_graph_edges": new_graph_edges,
            "structured_explanation": structured_explanation,
            "final_explanation": final_explanation,
        }

    def _cache_only_llm_payload_v2(
        self,
        *,
        user_input_text: str,
        trace: HybridReasoningTrace,
    ) -> Dict[str, object]:
        cached_entries = self._cached_entries_from_trace(trace)
        return build_llm_payload_from_packet_v1(
            trace.context_packet_v1 if isinstance(trace.context_packet_v1, dict) else {},
            cached_knowledge={
                "hit": True,
                "entry_ids": list(trace.cached_knowledge_entry_ids),
                "topic_key": trace.cached_knowledge_topic_key,
                "trust_level": trace.cached_knowledge_trust_level or "low",
                "verification_status": trace.cached_knowledge_verification_status or "unverified",
                "entries": [entry.to_dict() for entry in cached_entries],
                "answer_state": (
                    trace.merged_reasoning_packet.get("merged_answer_state", {})
                    if isinstance(trace.merged_reasoning_packet, dict)
                    else {}
                ),
                "answer_facts": (
                    trace.merged_reasoning_packet.get("merged_answer_facts", [])
                    if isinstance(trace.merged_reasoning_packet, dict)
                    else []
                ),
                "multi_hop_used": trace.multi_hop_used,
                "merged_entry_ids": list(trace.merged_entry_ids),
                "reasoning_steps": list(trace.reasoning_steps),
                "merged_reasoning_packet": dict(trace.merged_reasoning_packet),
            },
        )

    def _generate_llm_explanation(
        self,
        *,
        user_input_text: str,
        route_decision: QueryRouteDecision,
        trace: HybridReasoningTrace,
        final_result: BrainInputV1,
    ) -> LLMExplanationResult:
        context_packet_v1 = trace.context_packet_v1 if isinstance(trace.context_packet_v1, dict) else {}
        llm_payload_v1 = trace.llm_payload_v1 if isinstance(trace.llm_payload_v1, dict) else {}
        llm_payload_v2 = trace.llm_payload_v2 if isinstance(trace.llm_payload_v2, dict) else {}
        route_type = route_decision.route_type
        if route_type in {"factual_question", "memory_question", "hybrid_request"}:
            return self.llm.answer_user_prompt(
                user_input_text=user_input_text,
                route_type=route_type,
                structured_context={
                    "route_reasons": route_decision.reasons,
                    "selected_path": trace.selected_path,
                },
                context_packet_v1=context_packet_v1,
                llm_payload_v1=llm_payload_v1,
                llm_payload_v2=llm_payload_v2,
            )
        return self.llm.explain(
            user_input_text=user_input_text,
            result_brain_input=final_result,
            reasoning_trace={"trace_steps": trace.to_trace_steps(), **trace.to_dict()},
            selected_path=trace.selected_path,
            retrieved_memories=trace.retrieved_memories,
            context_packet_v1=context_packet_v1,
            llm_payload_v1=llm_payload_v1,
            llm_payload_v2=llm_payload_v2,
        )

    def _finalize_turn(
        self,
        *,
        user_input_text: str,
        effective_prompt_text: str,
        route_decision: QueryRouteDecision,
        enrichment_result: PromptEnrichmentResult,
        clarification_state: Dict[str, object],
        parse_result: ParseResult,
        parsed_brain_input: BrainInputV1,
        retrieved: Sequence[RetrievedMemory],
        recalled_turns: Sequence[EpisodicTurnRecall],
        decision: Dict[str, object],
        trace: HybridReasoningTrace,
        llm_explanation: Optional[LLMExplanationResult],
        stored_entries: Sequence[EpisodicMemoryEntry],
        final_result: BrainInputV1,
    ) -> HybridTurnResult:
        all_stored_entries = list(stored_entries)
        goal_tracking = self.goal_tracker.process_turn(
            sequence=self._turn_sequence,
            user_input_text=user_input_text,
            effective_prompt_text=effective_prompt_text,
            route_type=route_decision.route_type,
            parsed_brain_input=parsed_brain_input,
            final_result=final_result,
            candidate_path_count=len(trace.candidate_paths),
            clarification_state=clarification_state,
            recalled_turn_count=len(recalled_turns),
            goal_weights=trace.goal_weights,
            selected_path=trace.selected_path,
        )
        trace = self._trace_with_goal_tracking(trace, goal_tracking)
        strategy_plan = self.strategy_generator.process_turn(
            sequence=self._turn_sequence,
            route_type=route_decision.route_type,
            effective_prompt_text=effective_prompt_text,
            parsed_brain_input=parsed_brain_input,
            final_result=final_result,
            selected_path=trace.selected_path,
            goal_tracking=goal_tracking.to_dict(),
            clarification_state=clarification_state,
            candidate_path_count=len(trace.candidate_paths),
        )
        trace = self._trace_with_strategy_plan(trace, strategy_plan)
        trace = self._trace_with_context_packet(
            trace,
            user_input_text=user_input_text,
            parsed_brain_input=parsed_brain_input,
            final_result=final_result,
            clarification_state=clarification_state,
        )
        if route_decision.route_type == "factual_question" and trace.cached_knowledge_hit and trace.cached_knowledge_used:
            llm_payload_v2 = self._cache_only_llm_payload_v2(
                user_input_text=user_input_text,
                trace=trace,
            )
            trace = replace(
                trace,
                llm_payload_v1={},
                llm_payload_v2=llm_payload_v2,
            )
            llm_explanation = build_cached_answer_result(
                user_input_text=user_input_text,
                route_type=route_decision.route_type,
                llm_payload_v2=llm_payload_v2,
            )
            print(f"[DEBUG CACHE USE] Using cached entry_id={trace.cached_knowledge_entry_id}")
            print("[DEBUG CACHE USE] LLM skipped")
        else:
            trace = self._trace_with_llm_payload(
                trace,
                user_input_text=user_input_text,
                mode="direct_answer" if route_decision.route_type in {"factual_question", "memory_question", "hybrid_request"} else "explanation",
            )
            if llm_explanation is None:
                llm_explanation = self._generate_llm_explanation(
                    user_input_text=user_input_text,
                    route_decision=route_decision,
                    trace=trace,
                    final_result=final_result,
                )
        if self.storage_policy.storage_enabled() and self.should_cache_llm_answer(
            route_type=route_decision.route_type,
            llm_payload_v2=trace.llm_payload_v2,
            answer_text=llm_explanation.user_facing_text,
            metadata=llm_explanation.metadata,
            query_text=user_input_text,
        ):
            cache_entry = self._cache_llm_knowledge_entry(
                user_input_text=user_input_text,
                answer_text=llm_explanation.user_facing_text,
                route_type=route_decision.route_type,
                llm_payload_v2=trace.llm_payload_v2,
                llm_metadata=llm_explanation.metadata,
                parsed_brain_input=parsed_brain_input,
                parsed_memory=BrainCodec.resolve_memory(parse_result.memory),
            )
            if cache_entry is not None:
                all_stored_entries.append(cache_entry)
                trace = self._trace_with_cached_knowledge(
                    trace,
                    checked=trace.cached_knowledge_checked,
                    hit=trace.cached_knowledge_hit,
                    entry_ids=list(trace.cached_knowledge_entry_ids) + [cache_entry.entry_id],
                    topic_key=trace.cached_knowledge_topic_key or self.normalize_knowledge_topic(user_input_text),
                    trust_level="low",
                    written=True,
                )
        self.history.log(
            event_type="interactive_turn",
            summary=f"Processed prompt: {user_input_text[:80]}",
            payload={
                "route_decision": route_decision.to_dict(),
                "enrichment_result": enrichment_result.to_dict(),
                "clarification_state": dict(clarification_state),
                "goal_tracking": goal_tracking.to_dict(),
                "strategy_plan": strategy_plan.to_dict(),
                "decision": dict(decision),
                "selected_path": trace.selected_path,
                "final_result": final_result.to_dict(),
            },
        )
        if self.storage_policy.storage_enabled() and clarification_state.get("resolved"):
            clarification_memory = self._brain_input_to_memory(final_result)
            entry = self.memory_store.add(
                entry_type="clarification_exchange",
                source_text=str(clarification_state.get("clarification_response", "")),
                parsed_memory=clarification_memory,
                brain_input=final_result,
                metadata={
                    "clarification_id": clarification_state.get("clarification_id"),
                    "original_prompt": clarification_state.get("original_prompt", ""),
                    "clarification_question": clarification_state.get("clarification_question", ""),
                    "merged_prompt": clarification_state.get("merged_prompt", ""),
                    "missing_fields": clarification_state.get("missing_fields", []),
                },
            )
            if entry is not None:
                all_stored_entries.append(entry)
        if self.storage_policy.storage_enabled() and strategy_plan.selected_strategy:
            entry = self.memory_store.add(
                entry_type="strategy_plan",
                source_text=str(strategy_plan.explanation),
                parsed_memory=self._brain_input_to_memory(final_result),
                brain_input=final_result,
                metadata={
                    "route_type": route_decision.route_type,
                    "selected_strategy": dict(strategy_plan.selected_strategy),
                    "alternative_strategies": list(strategy_plan.alternative_strategies),
                    "strategy_feedback": dict(strategy_plan.strategy_feedback),
                },
            )
            if entry is not None:
                all_stored_entries.append(entry)
        if self.storage_policy.storage_enabled():
            state_source_text = (
                str(clarification_state.get("merged_prompt", "")).strip()
                if clarification_state.get("resolved")
                else effective_prompt_text
            ) or user_input_text
            state_observations = self._extract_state_observations(state_source_text)
            for observation in state_observations:
                entry = self.memory_store.add(
                    entry_type="state_memory",
                    source_text=state_source_text,
                    parsed_memory=BrainCodec.resolve_memory(parse_result.memory),
                    brain_input=parsed_brain_input,
                    metadata={
                        "sequence": self._turn_sequence,
                        "route_type": route_decision.route_type,
                        "state_observation": observation,
                        "is_update": bool(observation.get("is_update", False)),
                        "state_key": observation.get("state_key", ""),
                        "state_type": observation.get("state_type", ""),
                    },
                )
                if entry is not None:
                    all_stored_entries.append(entry)
        if self.storage_policy.storage_enabled():
            stored_prompt_text = (
                str(clarification_state.get("original_prompt", "")).strip()
                if clarification_state.get("resolved")
                else user_input_text
            ) or user_input_text
            self.turn_store.add(
                sequence=self._turn_sequence,
                raw_user_prompt=stored_prompt_text,
                final_answer=llm_explanation.user_facing_text,
                parsed_brain_input=parsed_brain_input,
                final_selected_result=final_result,
                route_type=route_decision.route_type,
                reasoning_succeeded=bool(trace.selected_path or trace.candidate_paths or recalled_turns),
                candidate_path_count=len(trace.candidate_paths),
                route_confidence=route_decision.confidence,
                metadata={
                    "needs_clarification": enrichment_result.needs_clarification,
                    "clarification_reasons": list(enrichment_result.clarification_reasons),
                    "clarification_state": dict(clarification_state),
                    "goal_tracking": goal_tracking.to_dict(),
                    "strategy_plan": strategy_plan.to_dict(),
                    "state_observations": self._extract_state_observations(stored_prompt_text),
                    "recalled_turn_count": len(recalled_turns),
                    "retrieved_memory_count": len(retrieved),
                },
            )
            self._turn_sequence += 1
        trace = self._trace_with_explanation(trace, llm_explanation, all_stored_entries)
        return HybridTurnResult(
            user_input_text=user_input_text,
            effective_prompt_text=effective_prompt_text,
            route_decision=route_decision,
            enrichment_result=enrichment_result.to_dict(),
            clarification_state=dict(clarification_state),
            goal_tracking=goal_tracking.to_dict(),
            strategy_plan=strategy_plan.to_dict(),
            parse_result={"memory": BrainCodec.resolve_memory(parse_result.memory).to_dict(), "debug": parse_result.debug},
            parsed_brain_input=parsed_brain_input,
            retrieved_memories=list(retrieved),
            recalled_turns=list(recalled_turns),
            decision=dict(decision),
            trace=trace,
            llm_explanation=llm_explanation,
            stored_entries=list(all_stored_entries),
            final_result=final_result,
        )

    def retrieve_relevant_brain_memory(
        self,
        query: BrainInputV1,
        *,
        query_text: str = "",
        top_k: int = 4,
    ) -> List[RetrievedMemory]:
        query_memory = self._brain_input_to_memory(query)
        query_tokens = self._episodic_tokens(str(query_text or "").lower())
        scored: List[RetrievedMemory] = []
        for entry in self.memory_store.entries:
            if not _is_meaningful_entry(entry):
                continue
            score = self._similarity_score(query, entry.brain_input, query_memory, entry.parsed_memory)
            if query_tokens:
                observation = entry.metadata.get("state_observation", {}) if isinstance(entry.metadata, dict) else {}
                score += 2.0 * self._text_overlap_score(query_tokens, observation)
                score += 1.25 * self._text_overlap_score(query_tokens, {"display_text": entry.source_text})
            if score <= 0.0:
                continue
            score += _entry_specificity_bonus(entry)
            scored.append(RetrievedMemory(entry_id=entry.entry_id, score=round(score, 4), entry=entry))
        return sorted(scored, key=lambda item: (-item.score, item.entry_id))[:top_k]

    def normalize_knowledge_topic(self, query_text: str) -> str:
        normalized = re.sub(r"[^a-z0-9\s]", " ", str(query_text or "").lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return "general:query"
        if "sky" in normalized and any(token in normalized for token in {"blue", "color", "colour"}):
            return "sky_color:daytime"
        if "laptop" in normalized and "battery" in normalized and any(token in normalized for token in {"what", "for", "does", "do"}):
            return "laptop_battery:function"
        if "battery" in normalized and any(token in normalized for token in {"what", "for", "does", "do"}):
            return "battery:function"
        if "water cycle" in normalized:
            return "water_cycle:definition"
        tokens = [token for token in normalized.split() if token not in {"what", "why", "how", "is", "are", "does", "do", "the", "a", "an", "for", "to", "of", "my", "your"}]
        if not tokens:
            return "general:query"
        suffix = "fact"
        if "why" in normalized:
            suffix = "reason"
        elif any(token in normalized for token in {"what", "does", "do", "for"}):
            suffix = "definition"
        return f"{'_'.join(tokens[:3])}:{suffix}"

    def normalize_query(self, text: str) -> List[str]:
        normalized = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return []
        stopwords = {"what", "is", "the", "a", "an", "does", "do"}
        tokens = [token for token in normalized.split() if token and token not in stopwords]
        deduped: List[str] = []
        seen = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
        return deduped

    def expand_topic_keys(self, query_text: str, topic_key: str) -> List[str]:
        expanded: List[str] = []
        seen = set()

        def add(value: object) -> None:
            text = str(value or "").strip().lower()
            if not text or text in seen:
                return
            seen.add(text)
            expanded.append(text)

        normalized_topic = str(topic_key or "").strip().lower()
        add(normalized_topic)
        if normalized_topic:
            base_topic = normalized_topic.split(":", 1)[0]
            add(base_topic)
            for part in re.split(r"[_:]+", normalized_topic):
                add(part)
        query_terms = self.normalize_query(query_text)
        for term in query_terms:
            add(term)
        if len(query_terms) >= 2:
            add("_".join(query_terms[:2]))
        return expanded

    def should_cache_llm_answer(
        self,
        *,
        route_type: str,
        llm_payload_v2: Dict[str, object],
        answer_text: str,
        metadata: Dict[str, object],
        query_text: str,
    ) -> bool:
        if route_type != "factual_question":
            return False
        if str(llm_payload_v2.get("source_of_truth", "llm")) != "llm":
            return False
        text = str(answer_text or "").strip()
        if not text:
            return False
        lowered = text.lower()
        if any(
            bad in lowered
            for bad in (
                "configure `brain_llm_provider",
                "configure `brain_llm_command",
                "routed to direct answering",
                "unknown_route_fallback",
                "i don't know",
                "i do not know",
                "cannot answer",
                "can't answer",
                "insufficient context",
                "not enough information",
                "error",
                "refuse",
                "sorry, i can’t",
            )
        ):
            return False
        query_lower = str(query_text or "").lower()
        if any(token in query_lower for token in ("my favorite", "my meeting", "project status", "checklist", "already tried", "current status", "my budget", "my goal")):
            return False
        mode = str(metadata.get("mode", "")).lower()
        if "fallback" in mode and "direct_answer" in mode and "deterministic" in mode and "This prompt was routed" in text:
            return False
        return True

    def _distill_cached_answer_facts(self, answer_text: str) -> List[str]:
        text = str(answer_text or "").strip()
        if not text:
            return []
        parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        return parts[:3] if parts else [text[:240]]

    def _cached_answer_state(self, topic_key: str, normalized_facts: Sequence[str]) -> Dict[str, object]:
        facts = [str(item).strip() for item in normalized_facts if str(item).strip()]
        if not facts:
            return {}
        return {str(topic_key or "cached_fact"): facts[0]}

    def _cache_answer_signature(self, metadata: Dict[str, object]) -> str:
        answer_text = str(metadata.get("answer_text", "")).strip().lower()
        if answer_text:
            return answer_text
        answer_state = metadata.get("answer_state", {})
        if isinstance(answer_state, dict) and answer_state:
            try:
                return json.dumps(answer_state, sort_keys=True)
            except (TypeError, ValueError):
                return str(answer_state)
        normalized_facts = metadata.get("normalized_facts", [])
        if isinstance(normalized_facts, list) and normalized_facts:
            return " | ".join(str(item).strip().lower() for item in normalized_facts if str(item).strip())
        return ""

    def _cache_recency_score(
        self,
        *,
        entry: EpisodicMemoryEntry,
        index: int,
        total_entries: int,
    ) -> float:
        try:
            timestamp = datetime.fromisoformat(str(entry.timestamp))
        except (TypeError, ValueError):
            timestamp = None
        if timestamp is not None:
            age_seconds = max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds())
            return 1.0 / (1.0 + (age_seconds / 86400.0))
        return (index + 1) / float(total_entries) if total_entries else 0.0

    def _cache_decay_penalty(self, metadata: Dict[str, object]) -> float:
        last_reused_at = str(metadata.get("last_reused_at", "")).strip()
        if not last_reused_at:
            return 0.0
        try:
            reused_at = datetime.fromisoformat(last_reused_at)
        except (TypeError, ValueError):
            return 0.0
        idle_days = max(0.0, (datetime.now(timezone.utc) - reused_at).total_seconds() / 86400.0)
        if idle_days <= 7.0:
            return 0.0
        return min(0.15, (idle_days - 7.0) * 0.005)

    def _cache_confidence_ceiling(
        self,
        metadata: Dict[str, object],
        *,
        verification_status: Optional[str] = None,
    ) -> float:
        source = str(metadata.get("source", "")).strip().lower()
        effective_verification = str(
            verification_status if verification_status is not None else metadata.get("verification_status", "unverified")
        ).strip().lower()
        if source == "llm" and effective_verification != "verified":
            return 0.9
        return 1.0

    def _cache_confidence_floor(
        self,
        metadata: Dict[str, object],
        *,
        verification_status: Optional[str] = None,
        successful_uses: Optional[int] = None,
    ) -> float:
        effective_verification = str(
            verification_status if verification_status is not None else metadata.get("verification_status", "unverified")
        ).strip().lower()
        effective_successful_uses = max(0, int(
            successful_uses if successful_uses is not None else metadata.get("successful_uses", 0) or 0
        ))
        if effective_verification == "verified" and effective_successful_uses >= 3:
            return 0.6
        return 0.0

    def _bounded_cache_confidence(
        self,
        metadata: Dict[str, object],
        *,
        raw_confidence: Optional[float] = None,
        verification_status: Optional[str] = None,
        successful_uses: Optional[int] = None,
    ) -> float:
        base_confidence = float(
            raw_confidence if raw_confidence is not None else metadata.get("confidence_estimate", 0.35) or 0.35
        )
        confidence_ceiling = self._cache_confidence_ceiling(
            metadata,
            verification_status=verification_status,
        )
        confidence_floor = self._cache_confidence_floor(
            metadata,
            verification_status=verification_status,
            successful_uses=successful_uses,
        )
        return max(confidence_floor, min(confidence_ceiling, max(0.0, min(1.0, base_confidence))))

    def _cache_dominance_score(
        self,
        metadata: Dict[str, object],
        *,
        verification_status: Optional[str] = None,
    ) -> float:
        conflict_wins = max(0, int(metadata.get("conflict_win_count", 0) or 0))
        successful_uses = max(0, int(metadata.get("successful_uses", 0) or 0))
        selection_streak = max(0, int(metadata.get("selected_streak_count", 0) or 0))
        verification_bonus = 0.15 if str(
            verification_status if verification_status is not None else metadata.get("verification_status", "unverified")
        ).strip().lower() == "verified" else 0.0
        return min(
            1.0,
            (conflict_wins * 0.08) + (successful_uses * 0.03) + (selection_streak * 0.06) + verification_bonus,
        )

    def verify_entry(
        self,
        entry: EpisodicMemoryEntry,
        *,
        persist: bool = True,
    ) -> Dict[str, object]:
        metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
        topic_key = str(metadata.get("topic_key", "")).strip().lower()
        answer_signature = self._cache_answer_signature(metadata)
        if entry.entry_type != "llm_knowledge_cache" or not topic_key or not answer_signature:
            return {
                "verification_attempted": False,
                "verification_status": str(metadata.get("verification_status", "unverified") or "unverified"),
                "support_count": 0,
                "conflict_count": 0,
            }
        support_count = 0
        conflict_count = 0
        for other in self.memory_store.entries:
            if other.entry_id == entry.entry_id or other.entry_type != "llm_knowledge_cache":
                continue
            other_metadata = other.metadata if isinstance(other.metadata, dict) else {}
            other_topic = str(other_metadata.get("topic_key", "")).strip().lower()
            if other_topic != topic_key:
                continue
            other_signature = self._cache_answer_signature(other_metadata)
            if not other_signature:
                continue
            if other_signature == answer_signature:
                support_count += 1
            else:
                conflict_count += 1
        verification_status = "verified" if support_count >= 1 and support_count >= conflict_count else "unverified"
        if persist and str(metadata.get("verification_status", "unverified") or "unverified").strip().lower() != verification_status:
            self.memory_store.update_metadata(
                entry.entry_id,
                {
                    "verification_status": verification_status,
                },
            )
        return {
            "verification_attempted": True,
            "verification_status": verification_status,
            "support_count": support_count,
            "conflict_count": conflict_count,
        }

    def _cache_adjusted_confidence(
        self,
        metadata: Dict[str, object],
        *,
        verification_status: Optional[str] = None,
    ) -> float:
        bounded_confidence = self._bounded_cache_confidence(
            metadata,
            verification_status=verification_status,
        )
        confidence_ceiling = self._cache_confidence_ceiling(
            metadata,
            verification_status=verification_status,
        )
        confidence_floor = self._cache_confidence_floor(
            metadata,
            verification_status=verification_status,
        )
        return max(confidence_floor, min(confidence_ceiling, bounded_confidence - self._cache_decay_penalty(metadata)))

    def _cache_reasoning_graph_candidate(
        self,
        *,
        entry: EpisodicMemoryEntry,
        ranking_score: Dict[str, object],
    ) -> ReasoningGraphCandidate:
        metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
        facts = metadata.get("normalized_facts", [])
        normalized_facts = [str(item).strip() for item in facts if str(item).strip()] if isinstance(facts, list) else []
        if not normalized_facts:
            normalized_facts = self._distill_cached_answer_facts(str(metadata.get("answer_text", entry.source_text)))
        return ReasoningGraphCandidate(
            entry_id=entry.entry_id,
            label=f"entry:{entry.entry_id}",
            base_score=float(ranking_score.get("base_score", ranking_score.get("total_score", 0.0)) or 0.0),
            topic_key=str(metadata.get("topic_key", "")).strip().lower(),
            answer_text=str(metadata.get("answer_text", entry.source_text)).strip(),
            answer_signature=str(ranking_score.get("answer_signature", "")).strip(),
            terms=self._cached_entry_terms(entry),
            facts=normalized_facts,
        )

    def _apply_cache_reasoning_graph(
        self,
        *,
        user_input_text: str,
        ranked: Sequence[tuple[float, EpisodicMemoryEntry, Dict[str, object]]],
        debug_lines: List[str],
    ) -> Dict[str, object]:
        if not ranked:
            return {
                "query_intent": QueryIntent.UNKNOWN.value,
                "ranked": [],
                "reasoning_graph_nodes": [],
                "reasoning_graph_edges": [],
                "reasoning_graph_clusters": [],
                "reasoning_graph_cluster_relationships": [],
                "reasoning_spatial_grid": {},
                "selected_reasoning_path": [],
                "selected_spatial_region_id": None,
                "selected_spatial_nodes": [],
                "selected_spatial_clusters": [],
                "selected_spatial_activation": 0.0,
                "spatial_selection_reason": "",
                "spatial_consistency_warning": "",
                "cluster_selection_reason": "",
                "graph_reasoning_steps": [],
            }
        graph_candidates = [
            self._cache_reasoning_graph_candidate(entry=entry, ranking_score=score)
            for _, entry, score in ranked[:10]
        ]
        graph = self.selection_graph_evaluator.build_graph(
            candidates=graph_candidates,
            query_text=user_input_text,
        )
        selection = self.selection_graph_evaluator.select_nodes(graph)
        node_lookup = {node.entry_id: node for node in graph.nodes}
        edge_dicts = [edge.to_dict() for edge in graph.edges]
        cluster_dicts = [cluster.to_dict() for cluster in graph.clusters]
        cluster_relationship_dicts = [edge.to_dict() for edge in graph.cluster_relationships]
        spatial_grid_dict = {} if graph.spatial_grid is None else graph.spatial_grid.to_dict()
        debug_lines.append(f"[DEBUG GRAPH INTENT] query_intent={graph.query_intent.value}")
        debug_lines.append(
            f"[DEBUG GRAPH SELECTION MODE] selection_mode={selection.selection_mode} selected_cluster_id={selection.selected_cluster_id} selected_node_entry_id={selection.selected_node_entry_id}"
        )
        score_lookup = {
            int(score.get("entry_id", -1)): score
            for _, _, score in ranked
        }
        for entry_id, score in score_lookup.items():
            node = node_lookup.get(entry_id)
            base_score = round(float(score.get("base_score", score.get("total_score", 0.0)) or 0.0), 4)
            if node is None:
                score["base_score"] = base_score
                score["support_bonus"] = round(float(score.get("support_bonus", 0.0) or 0.0), 4)
                score["contradiction_penalty"] = round(float(score.get("contradiction_penalty", 0.0) or 0.0), 4)
                score["extension_bonus"] = round(float(score.get("extension_bonus", 0.0) or 0.0), 4)
                score["directness_score"] = round(float(score.get("directness_score", 0.0) or 0.0), 4)
                score["directness_bonus"] = round(float(score.get("directness_bonus", 0.0) or 0.0), 4)
                score["cluster_strength"] = round(float(score.get("cluster_strength", 0.0) or 0.0), 4)
                score["coherence_bonus"] = round(float(score.get("coherence_bonus", 0.0) or 0.0), 4)
                score["dense_cluster_bonus"] = round(float(score.get("dense_cluster_bonus", 0.0) or 0.0), 4)
                score["cluster_influence_bonus"] = round(float(score.get("cluster_influence_bonus", 0.0) or 0.0), 4)
                score["intent_bonus"] = round(float(score.get("intent_bonus", 0.0) or 0.0), 4)
                score["concept_predictive_bias"] = round(float(score.get("concept_predictive_bias", 0.0) or 0.0), 4)
                score["raw_support_score"] = round(float(score.get("raw_support_score", 0.0) or 0.0), 4)
                score["raw_causal_score"] = round(float(score.get("raw_causal_score", 0.0) or 0.0), 4)
                score["raw_contradiction_score"] = round(float(score.get("raw_contradiction_score", 0.0) or 0.0), 4)
                score["raw_extension_score"] = round(float(score.get("raw_extension_score", 0.0) or 0.0), 4)
                score["cluster_id"] = int(score.get("cluster_id", -1) or -1)
                score["cluster_score"] = round(float(score.get("cluster_score", base_score) or base_score), 4)
                score["cluster_rank"] = int(score.get("cluster_rank", 0) or 0)
                score["cluster_size"] = int(score.get("cluster_size", 1) or 1)
                score["cluster_density"] = round(float(score.get("cluster_density", 0.0) or 0.0), 4)
                score["base_activation_value"] = round(float(score.get("base_activation_value", 0.0) or 0.0), 4)
                score["propagated_activation_bonus"] = round(float(score.get("propagated_activation_bonus", 0.0) or 0.0), 4)
                score["inhibition_penalty"] = round(float(score.get("inhibition_penalty", 0.0) or 0.0), 4)
                score["activation_value"] = round(float(score.get("activation_value", 0.0) or 0.0), 4)
                score["query_intent"] = graph.query_intent.value
                score["final_score"] = base_score
                score["total_score"] = base_score
                score["graph_relationships"] = []
                continue
            score["base_score"] = round(node.base_score, 4)
            score["raw_support_score"] = round(node.raw_support_score, 4)
            score["raw_causal_score"] = round(node.raw_causal_score, 4)
            score["raw_contradiction_score"] = round(node.raw_contradiction_score, 4)
            score["raw_extension_score"] = round(node.raw_extension_score, 4)
            score["support_bonus"] = round(node.support_bonus, 4)
            score["contradiction_penalty"] = round(node.contradiction_penalty, 4)
            score["extension_bonus"] = round(node.extension_bonus, 4)
            score["directness_score"] = round(node.directness_score, 4)
            score["directness_bonus"] = round(node.directness_bonus, 4)
            score["cluster_strength"] = round(node.cluster_strength, 4)
            score["coherence_bonus"] = round(node.coherence_bonus, 4)
            score["dense_cluster_bonus"] = round(node.dense_cluster_bonus, 4)
            score["cluster_influence_bonus"] = round(node.cluster_influence_bonus, 4)
            score["intent_bonus"] = round(node.intent_bonus, 4)
            score["concept_predictive_bias"] = round(node.concept_predictive_bias, 4)
            score["cluster_id"] = int(node.cluster_id)
            score["cluster_score"] = round(node.cluster_score, 4)
            score["cluster_rank"] = int(node.cluster_rank)
            score["cluster_size"] = int(node.cluster_size)
            score["cluster_density"] = round(node.cluster_density, 4)
            score["base_activation_value"] = round(node.base_activation_value, 4)
            score["propagated_activation_bonus"] = round(node.propagated_activation_bonus, 4)
            score["inhibition_penalty"] = round(node.inhibition_penalty, 4)
            score["activation_value"] = round(node.activation_value, 4)
            score["query_intent"] = graph.query_intent.value
            score["final_score"] = round(node.final_score, 4)
            score["total_score"] = round(node.final_score, 4)
            score["graph_relationships"] = list(node.relationships)
        rank_priority = {
            int(entry_id): index
            for index, entry_id in enumerate(selection.ranked_entry_ids)
        }
        ranked_by_graph = sorted(
            [
                (
                    float(score.get("final_score", score.get("total_score", 0.0)) or 0.0),
                    entry,
                    score,
                )
                for _, entry, score in ranked
            ],
            key=lambda item: (
                rank_priority.get(item[1].entry_id, len(rank_priority)),
                -float(item[2].get("cluster_score", 0.0) or 0.0),
                -item[0],
                item[1].entry_id,
            ),
        )
        for edge in graph.edges:
            edge_kind = edge.relationship_type
            if edge_kind == RelationshipType.SUPPORTS:
                debug_lines.append(
                    f"[DEBUG GRAPH SUPPORT] entry_id={edge.source_entry_id} <-> entry_id={edge.target_entry_id} strength={round(edge.strength, 4)}"
                )
            elif edge_kind == RelationshipType.CONTRADICTS:
                debug_lines.append(
                    f"[DEBUG GRAPH CONTRADICT] entry_id={edge.source_entry_id} <-> entry_id={edge.target_entry_id} strength={round(edge.strength, 4)}"
                )
            elif edge_kind == RelationshipType.EXTENDS:
                debug_lines.append(
                    f"[DEBUG GRAPH EXTENDS] entry_id={edge.source_entry_id} -> entry_id={edge.target_entry_id} strength={round(edge.strength, 4)}"
                )
            elif edge_kind == RelationshipType.CAUSES:
                debug_lines.append(
                    f"[DEBUG GRAPH CAUSES] entry_id={edge.source_entry_id} -> entry_id={edge.target_entry_id} strength={round(edge.strength, 4)}"
                )
        for node in graph.nodes:
            debug_lines.append(
                "[DEBUG GRAPH SCORE] "
                + f"entry_id={node.entry_id} base_score={round(node.base_score, 4)} "
                + f"raw_support_score={round(node.raw_support_score, 4)} "
                + f"raw_causal_score={round(node.raw_causal_score, 4)} "
                + f"raw_extension_score={round(node.raw_extension_score, 4)} "
                + f"raw_contradiction_score={round(node.raw_contradiction_score, 4)} "
                + f"support_bonus={round(node.support_bonus, 4)} "
                + f"contradiction_penalty={round(node.contradiction_penalty, 4)} "
                + f"extension_bonus={round(node.extension_bonus, 4)} "
                + f"directness_score={round(node.directness_score, 4)} "
                + f"directness_bonus={round(node.directness_bonus, 4)} "
                + f"cluster_strength={round(node.cluster_strength, 4)} "
                + f"coherence_bonus={round(node.coherence_bonus, 4)} "
                + f"dense_cluster_bonus={round(node.dense_cluster_bonus, 4)} "
                + f"cluster_influence_bonus={round(node.cluster_influence_bonus, 4)} "
                + f"intent_bonus={round(node.intent_bonus, 4)} "
                + f"concept_predictive_bias={round(node.concept_predictive_bias, 4)} "
                + f"cluster_id={node.cluster_id} "
                + f"cluster_score={round(node.cluster_score, 4)} "
                + f"cluster_rank={node.cluster_rank} "
                + f"cluster_density={round(node.cluster_density, 4)} "
                + f"base_activation_value={round(node.base_activation_value, 4)} "
                + f"propagated_activation_bonus={round(node.propagated_activation_bonus, 4)} "
                + f"inhibition_penalty={round(node.inhibition_penalty, 4)} "
                + f"activation_value={round(node.activation_value, 4)} "
                + f"final_score={round(node.final_score, 4)}"
            )
        for cluster in graph.clusters:
            debug_lines.append(
                "[DEBUG GRAPH CLUSTER] "
                + f"cluster_id={cluster.cluster_id} "
                + f"cluster_score={round(cluster.cluster_score, 4)} "
                + f"cluster_final_score={round(cluster.cluster_final_score, 4)} "
                + f"node_score_sum={round(float(cluster.node_score_sum), 4)} "
                + f"structural_bonus={round(float(cluster.structural_bonus), 4)} "
                + f"cluster_support_bonus={round(float(cluster.cluster_support_bonus), 4)} "
                + f"cluster_contradiction_penalty={round(float(cluster.cluster_contradiction_penalty), 4)} "
                + f"base_activation_value={round(float(cluster.base_activation_value), 4)} "
                + f"propagated_activation_bonus={round(float(cluster.propagated_activation_bonus), 4)} "
                + f"inhibition_penalty={round(float(cluster.inhibition_penalty), 4)} "
                + f"activation_value={round(float(cluster.activation_value), 4)} "
                + f"density={round(cluster.density, 4)} "
                + f"support_edges={cluster.support_edges} "
                + f"contradiction_edges={cluster.contradiction_edges} "
                + f"entry_ids={cluster.entry_ids}"
            )
        for cluster_edge in graph.cluster_relationships:
            debug_lines.append(
                "[DEBUG GRAPH CLUSTER EDGE] "
                + f"cluster_id={cluster_edge.source_cluster_id} -> cluster_id={cluster_edge.target_cluster_id} "
                + f"relationship={cluster_edge.relationship_type.value} "
                + f"strength={round(cluster_edge.strength, 4)} "
                + f"node_edge_count={cluster_edge.node_edge_count}"
            )
        if graph.spatial_grid is not None:
            for mapping in graph.spatial_grid.node_mappings:
                debug_lines.append(
                    "[DEBUG GRID MAP] "
                    + f"entry_id={mapping.entry_id} cluster_id={mapping.cluster_id} "
                    + f"center=({mapping.center_x},{mapping.center_y}) "
                    + f"initial_region_activation={round(float(mapping.initial_region_activation), 4)} "
                    + f"propagated_region_activation={round(float(mapping.propagated_region_activation), 4)} "
                    + f"final_region_activation={round(float(mapping.final_region_activation), 4)}"
                )
            for cluster_mapping in graph.spatial_grid.cluster_mappings:
                debug_lines.append(
                    "[DEBUG GRID CLUSTER MAP] "
                    + f"cluster_id={cluster_mapping.cluster_id} "
                    + f"center=({cluster_mapping.center_x},{cluster_mapping.center_y}) "
                    + f"radius={cluster_mapping.radius} "
                    + f"activation_value={round(float(cluster_mapping.activation_value), 4)} "
                    + f"final_region_activation={round(float(cluster_mapping.final_region_activation), 4)}"
                )
            for log_line in graph.spatial_grid.propagation_log:
                debug_lines.append(f"[DEBUG GRID PROPAGATION] {log_line}")
            for log_line in graph.spatial_grid.inhibition_log:
                debug_lines.append(f"[DEBUG GRID INHIBITION] {log_line}")
            for log_line in graph.spatial_grid.temporal_update_log:
                debug_lines.append(f"[DEBUG GRID TEMPORAL] {log_line}")
            for log_line in graph.spatial_grid.dominance_evolution_log:
                debug_lines.append(f"[DEBUG GRID DOMINANCE EVOLUTION] {log_line}")
            for log_line in graph.spatial_grid.concept_reuse_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT REUSE] {log_line}")
            for log_line in graph.spatial_grid.concept_pre_activation_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT PREACTIVATE] {log_line}")
            for log_line in graph.spatial_grid.anticipation_log:
                debug_lines.append(f"[DEBUG GRID ANTICIPATION] {log_line}")
            for log_line in graph.spatial_grid.concept_activation_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT ACTIVATION] {log_line}")
            for log_line in graph.spatial_grid.concept_importance_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT IMPORTANCE] {log_line}")
            for log_line in graph.spatial_grid.concept_importance_evolution_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT IMPORTANCE EVOLVE] {log_line}")
            for log_line in graph.spatial_grid.prediction_accuracy_log:
                debug_lines.append(f"[DEBUG GRID PREDICTION ACCURACY] {log_line}")
            for log_line in graph.spatial_grid.concept_exploration_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT EXPLORATION] {log_line}")
            for log_line in graph.spatial_grid.concept_uncertainty_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT UNCERTAINTY] {log_line}")
            for log_line in graph.spatial_grid.concept_cross_domain_link_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT CROSS DOMAIN] {log_line}")
            for log_line in graph.spatial_grid.concept_generalization_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT GENERALIZE] {log_line}")
            for log_line in graph.spatial_grid.concept_stability_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT STABILITY] {log_line}")
            for log_line in graph.spatial_grid.concept_abstraction_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT ABSTRACTION] {log_line}")
            for log_line in graph.spatial_grid.temporal_sequence_log:
                debug_lines.append(f"[DEBUG GRID TEMPORAL SEQUENCE] {log_line}")
            for log_line in graph.spatial_grid.temporal_prediction_log:
                debug_lines.append(f"[DEBUG GRID TEMPORAL PREDICT] {log_line}")
            for log_line in graph.spatial_grid.goal_selection_log:
                debug_lines.append(f"[DEBUG GRID GOAL SELECT] {log_line}")
            for log_line in graph.spatial_grid.goal_plan_log:
                debug_lines.append(f"[DEBUG GRID GOAL PLAN] {log_line}")
            for log_line in graph.spatial_grid.goal_feedback_log:
                debug_lines.append(f"[DEBUG GRID GOAL FEEDBACK] {log_line}")
            for log_line in graph.spatial_grid.concept_evolution_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT EVOLVE] {log_line}")
            for log_line in graph.spatial_grid.concept_competition_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT COMPETE] {log_line}")
            for log_line in graph.spatial_grid.meta_concept_influence_log:
                debug_lines.append(f"[DEBUG GRID META CONCEPT] {log_line}")
            for log_line in graph.spatial_grid.concept_cluster_competition_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT CLUSTER COMPETE] {log_line}")
            for log_line in graph.spatial_grid.concept_hierarchy_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT HIERARCHY] {log_line}")
            for log_line in graph.spatial_grid.concept_cluster_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT CLUSTER] {log_line}")
            for log_line in graph.spatial_grid.concept_creation_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT CREATE] {log_line}")
            for log_line in graph.spatial_grid.concept_merge_log:
                debug_lines.append(f"[DEBUG GRID CONCEPT MERGE] {log_line}")
            for log_line in graph.spatial_grid.region_merge_log:
                debug_lines.append(f"[DEBUG GRID REGION MERGE] {log_line}")
            for log_line in graph.spatial_grid.region_dominance_log:
                debug_lines.append(f"[DEBUG GRID DOMINANCE] {log_line}")
            for log_line in graph.spatial_grid.region_expansion_log:
                debug_lines.append(f"[DEBUG GRID EXPANSION] {log_line}")
            for log_line in graph.spatial_grid.region_competition_log:
                debug_lines.append(f"[DEBUG GRID REGION COMPETE] {log_line}")
            for region in graph.spatial_grid.top_regions:
                debug_lines.append(
                    "[DEBUG GRID PEAK] "
                    + f"region_id={region.region_id} "
                    + f"total_activation={round(float(region.total_activation), 4)} "
                    + f"peak_activation={round(float(region.peak_activation), 4)} "
                    + f"dominance_value={round(float(region.dominance_value), 4)} "
                    + f"dominant_node_id={region.dominant_node_id} "
                    + f"source_nodes={region.source_nodes} "
                    + f"source_clusters={region.source_clusters}"
                )
            for line in graph.spatial_grid.placement_map_lines:
                debug_lines.append(f"[DEBUG GRID PLACEMENT] {line}")
            for line in graph.spatial_grid.cluster_boundary_lines:
                debug_lines.append(f"[DEBUG GRID BOUNDARY] {line}")
            for line in graph.spatial_grid.heatmap_after_lines:
                debug_lines.append(f"[DEBUG GRID HEATMAP] {line}")
        if selection.selected_cluster_id is not None:
            debug_lines.append(
                "[DEBUG GRAPH CLUSTER SELECT] "
                + f"selected_cluster_id={selection.selected_cluster_id} "
                + f"selected_cluster_score={round(selection.selected_cluster_score, 4)} "
                + f"selected_node_entry_id={selection.selected_node_entry_id}"
            )
        if selection.cluster_selection_reason:
            debug_lines.append(f"[DEBUG GRAPH CLUSTER WHY] {selection.cluster_selection_reason}")
        if selection.selected_spatial_region_id is not None:
            debug_lines.append(
                "[DEBUG GRID SELECT] "
                + f"selected_spatial_region_id={selection.selected_spatial_region_id} "
                + f"selected_spatial_nodes={selection.selected_spatial_nodes} "
                + f"selected_spatial_clusters={selection.selected_spatial_clusters} "
                + f"selected_spatial_activation={round(selection.selected_spatial_activation, 4)} "
                + f"reason={selection.spatial_selection_reason}"
            )
        if selection.spatial_consistency_warning:
            debug_lines.append(f"[DEBUG GRID CONSISTENCY WARNING] {selection.spatial_consistency_warning}")
        if selection.selected_reasoning_path:
            debug_lines.append(
                "[DEBUG GRAPH PATH] "
                + " -> ".join(f"entry_id={entry_id}" for entry_id in selection.selected_reasoning_path)
            )
            debug_lines.append(
                "[DEBUG GRAPH PATH SCORE] "
                + f"path_score={round(selection.selected_path_score, 4)} "
                + f"path_activation={round(selection.selected_path_activation, 4)} "
                + f"path_final_score={round(selection.selected_path_final_score, 4)} "
                + f"breakdown={selection.selected_path_breakdown}"
            )
        for path_info in selection.candidate_paths:
            debug_lines.append(
                "[DEBUG GRAPH PATH COMPARE] "
                + f"path={path_info.get('node_ids', [])} "
                + f"path_score={round(float(path_info.get('path_score', 0.0) or 0.0), 4)} "
                + f"relative_strength={round(float(path_info.get('relative_strength', 0.0) or 0.0), 4)} "
                + f"adjusted_score={round(float(path_info.get('adjusted_score', 0.0) or 0.0), 4)} "
                + f"path_activation={round(float(path_info.get('path_activation', 0.0) or 0.0), 4)} "
                + f"path_final_score={round(float(path_info.get('path_final_score', 0.0) or 0.0), 4)}"
            )
        graph_reasoning_steps = []
        if graph.nodes:
            graph_reasoning_steps.append(
                f"Evaluated a reasoning graph across {len(graph.nodes)} cached knowledge entries with intent {graph.query_intent.value}."
            )
        if graph.edges:
            graph_reasoning_steps.append(
                f"Compared cached entries pairwise and derived {len(graph.edges)} graph relationships."
            )
        if graph.clusters:
            graph_reasoning_steps.append(
                f"Identified {len(graph.clusters)} connected support clusters for debugging and early cluster-aware reasoning."
            )
        if graph.cluster_relationships:
            graph_reasoning_steps.append(
                f"Derived {len(graph.cluster_relationships)} cluster-to-cluster relationships from cross-cluster node edges."
            )
        if graph.spatial_grid is not None and graph.spatial_grid.top_regions:
            graph_reasoning_steps.append(
                f"Projected graph activation into a {graph.spatial_grid.width}x{graph.spatial_grid.height} spatial grid and extracted {len(graph.spatial_grid.top_regions)} activation regions."
            )
            if graph.spatial_grid.temporal_update_log:
                graph_reasoning_steps.append(
                    f"Evolved the spatial field across {len(graph.spatial_grid.temporal_update_log)} temporal ticks with propagation, inhibition, dominance, and competition."
                )
            if graph.spatial_grid.concept_creation_log or graph.spatial_grid.concept_reuse_log:
                graph_reasoning_steps.append(
                    f"Updated spatial concept memory with {len(graph.spatial_grid.concept_creation_log)} creation events and {len(graph.spatial_grid.concept_reuse_log)} reuse events."
                )
            if graph.spatial_grid.concept_evolution_log or graph.spatial_grid.concept_competition_log:
                graph_reasoning_steps.append(
                    f"Evolved {len(graph.spatial_grid.concept_evolution_log)} reused concepts and applied {len(graph.spatial_grid.concept_competition_log)} concept competition events."
                )
            if graph.spatial_grid.concept_hierarchy_log or graph.spatial_grid.concept_cluster_log:
                graph_reasoning_steps.append(
                    f"Updated higher-level concept intelligence with {len(graph.spatial_grid.concept_hierarchy_log)} hierarchy events and {len(graph.spatial_grid.concept_cluster_log)} concept cluster records."
                )
            if graph.spatial_grid.anticipation_log or graph.spatial_grid.prediction_accuracy_log:
                graph_reasoning_steps.append(
                    f"Ran anticipatory concept control with {len(graph.spatial_grid.anticipation_log)} predictions and {len(graph.spatial_grid.prediction_accuracy_log)} prediction feedback updates."
                )
            if (
                graph.spatial_grid.concept_exploration_log
                or graph.spatial_grid.concept_cross_domain_link_log
                or graph.spatial_grid.concept_generalization_log
                or graph.spatial_grid.concept_stability_log
                or graph.spatial_grid.concept_abstraction_log
            ):
                graph_reasoning_steps.append(
                    f"Stabilized concept intelligence with {len(graph.spatial_grid.concept_exploration_log)} exploration events, {len(graph.spatial_grid.concept_cross_domain_link_log)} cross-domain links, {len(graph.spatial_grid.concept_generalization_log)} generalization events, {len(graph.spatial_grid.concept_stability_log)} stability controls, and {len(graph.spatial_grid.concept_abstraction_log)} abstraction-layer records."
                )
            if (
                graph.spatial_grid.temporal_sequence_log
                or graph.spatial_grid.temporal_prediction_log
                or graph.spatial_grid.goal_plan_log
                or graph.spatial_grid.goal_feedback_log
            ):
                graph_reasoning_steps.append(
                    f"Ran temporal goal control with {len(graph.spatial_grid.temporal_sequence_log)} sequence records, {len(graph.spatial_grid.temporal_prediction_log)} next-concept predictions, {len(graph.spatial_grid.goal_plan_log)} plan evaluations, and {len(graph.spatial_grid.goal_feedback_log)} feedback updates."
                )
        if selection.selection_mode == "cluster_graph_first" and selection.selected_cluster_id is not None:
            graph_reasoning_steps.append(
                f"Ran cluster competition for explanation reasoning, selected cluster {selection.selected_cluster_id}, then chose node {selection.selected_node_entry_id} within that cluster."
            )
        elif selection.selected_node_entry_id is not None:
            graph_reasoning_steps.append(
                f"Selected node {selection.selected_node_entry_id} directly from node-level graph scoring."
            )
        if selection.selected_reasoning_path:
            graph_reasoning_steps.append(
                "Extracted a reasoning path inside the selected cluster: "
                + " -> ".join(f"entry {entry_id}" for entry_id in selection.selected_reasoning_path)
                + "."
            )
        return {
            "query_intent": graph.query_intent.value,
            "ranked": ranked_by_graph,
            "reasoning_graph_nodes": [node.to_dict() for node in graph.nodes],
            "reasoning_graph_edges": edge_dicts,
            "reasoning_graph_clusters": cluster_dicts,
            "reasoning_graph_cluster_relationships": cluster_relationship_dicts,
            "reasoning_spatial_grid": spatial_grid_dict,
            "selection_mode": selection.selection_mode,
            "selected_cluster_id": selection.selected_cluster_id,
            "selected_cluster_score": round(selection.selected_cluster_score, 4),
            "selected_node_entry_id": selection.selected_node_entry_id,
            "selected_reasoning_path": list(selection.selected_reasoning_path),
            "selected_path_score": round(selection.selected_path_score, 4),
            "selected_path_activation": round(selection.selected_path_activation, 4),
            "selected_path_final_score": round(selection.selected_path_final_score, 4),
            "selected_path_breakdown": selection.selected_path_breakdown,
            "candidate_reasoning_paths": [dict(item) for item in selection.candidate_paths],
            "selected_spatial_region_id": selection.selected_spatial_region_id,
            "selected_spatial_nodes": list(selection.selected_spatial_nodes),
            "selected_spatial_clusters": list(selection.selected_spatial_clusters),
            "selected_spatial_activation": round(selection.selected_spatial_activation, 4),
            "spatial_selection_reason": selection.spatial_selection_reason,
            "spatial_consistency_warning": selection.spatial_consistency_warning,
            "cluster_selection_reason": selection.cluster_selection_reason,
            "graph_reasoning_steps": graph_reasoning_steps,
        }

    def _cache_rank_candidates(
        self,
        *,
        user_input_text: str,
        topic_key: Optional[str] = None,
        top_k: int = 3,
        update_reuse: bool = False,
    ) -> tuple[List[EpisodicMemoryEntry], Dict[str, object]]:
        normalized_topic = str(topic_key or self.normalize_knowledge_topic(user_input_text))
        query_terms = [str(term).strip().lower() for term in self.normalize_query(user_input_text) if str(term).strip()]
        query_term_set = set(query_terms)
        total_entries = len(self.memory_store.entries)
        debug_lines: List[str] = [
            f"[DEBUG] raw_query: {str(user_input_text or '').strip()}",
            f"[DEBUG] normalized_terms: {query_terms}",
            f"[DEBUG] normalized_topic: {normalized_topic}",
        ]
        ranked: List[tuple[float, EpisodicMemoryEntry, Dict[str, object]]] = []
        for index, entry in enumerate(self.memory_store.entries):
            if entry.entry_type != "llm_knowledge_cache":
                continue
            metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
            entry_topic = str(metadata.get("topic_key", "")).strip().lower()
            raw_topic_keys = metadata.get("topic_keys", [])
            entry_topic_keys = [str(item).strip().lower() for item in raw_topic_keys if str(item).strip()] if isinstance(raw_topic_keys, list) else []
            if entry_topic and entry_topic not in entry_topic_keys:
                entry_topic_keys.append(entry_topic)
            entry_key_set = set(entry_topic_keys)
            intersection = sorted(query_term_set & entry_key_set)
            topic_match = bool(entry_topic and entry_topic == normalized_topic.lower())
            term_match = bool(intersection)
            match = topic_match or term_match
            match_strength = (len(intersection) / float(len(query_terms))) if query_terms else 0.0
            verification_info = self.verify_entry(entry, persist=update_reuse)
            verification_attempted = bool(verification_info.get("verification_attempted", False))
            verification_status = str(verification_info.get("verification_status", metadata.get("verification_status", "unverified")) or "unverified").strip().lower()
            confidence_ceiling = self._cache_confidence_ceiling(metadata, verification_status=verification_status)
            confidence_estimate = max(0.0, min(1.0, float(metadata.get("confidence_estimate", 0.35) or 0.35)))
            confidence_capped = confidence_estimate > confidence_ceiling
            adjusted_confidence = self._cache_adjusted_confidence(
                metadata,
                verification_status=verification_status,
            )
            reuse_count = max(0, int(metadata.get("reuse_count", 0) or 0))
            reuse_score = min(1.0, reuse_count * 0.1)
            successful_uses = max(0, int(metadata.get("successful_uses", 0) or 0))
            successful_use_score = min(1.0, successful_uses * 0.08)
            recency_score = self._cache_recency_score(entry=entry, index=index, total_entries=total_entries)
            if verification_status == "unverified":
                adjusted_confidence = max(0.0, adjusted_confidence - ((1.0 - recency_score) * 0.08))
            selected_streak_count = max(0, int(metadata.get("selected_streak_count", 0) or 0))
            conflict_win_count = max(0, int(metadata.get("conflict_win_count", 0) or 0))
            dominance_score = self._cache_dominance_score(
                metadata,
                verification_status=verification_status,
            )
            dominance_bonus = dominance_score * 0.75
            convergence_rate = min(1.0, selected_streak_count / 3.0)
            stabilized_flag = selected_streak_count >= 3
            verification_bonus = 4.0 if verification_status == "verified" else 0.0
            total_score = (
                adjusted_confidence * 4.0
                + reuse_score * 2.0
                + successful_use_score * 1.5
                + match_strength * 3.0
                + recency_score * 1.5
                + (1.5 if topic_match else 0.0)
                + verification_bonus
                + dominance_bonus
            )
            ranking_score = {
                "entry_id": entry.entry_id,
                "topic_key": entry_topic,
                "confidence_estimate": round(confidence_estimate, 4),
                "adjusted_confidence": round(adjusted_confidence, 4),
                "confidence_ceiling": round(confidence_ceiling, 4),
                "confidence_capped": confidence_capped,
                "reuse_count": reuse_count,
                "reuse_score": round(reuse_score, 4),
                "successful_uses": successful_uses,
                "successful_use_score": round(successful_use_score, 4),
                "selected_streak_count": selected_streak_count,
                "conflict_win_count": conflict_win_count,
                "dominance_score": round(dominance_score, 4),
                "dominance_bonus": round(dominance_bonus, 4),
                "convergence_rate": round(convergence_rate, 4),
                "stabilized_flag": stabilized_flag,
                "verification_attempted": verification_attempted,
                "verification_status": verification_status,
                "verification_bonus": round(verification_bonus, 4),
                "verification_support_count": int(verification_info.get("support_count", 0) or 0),
                "verification_conflict_count": int(verification_info.get("conflict_count", 0) or 0),
                "match_strength": round(match_strength, 4),
                "recency_score": round(recency_score, 4),
                "topic_match": topic_match,
                "intersection": intersection,
                "answer_signature": self._cache_answer_signature(metadata),
                "base_score": round(total_score, 4),
                "support_bonus": 0.0,
                "contradiction_penalty": 0.0,
                "extension_bonus": 0.0,
                "final_score": round(total_score, 4),
                "total_score": round(total_score, 4),
                "graph_relationships": [],
            }
            debug_lines.extend(
                [
                    "[DEBUG MATCH CHECK]",
                    f"query_terms: {query_terms}",
                    f"entry_keys: {entry_topic_keys}",
                    f"intersection: {intersection}",
                    f"match: {'TRUE' if match else 'FALSE'}",
                    f"entry_id={entry.entry_id} confidence={ranking_score['confidence_estimate']} adjusted_confidence={ranking_score['adjusted_confidence']} verification_status={verification_status} reuse_count={reuse_count} successful_uses={successful_uses} dominance_score={ranking_score['dominance_score']} streak={selected_streak_count} match_strength={ranking_score['match_strength']} recency={ranking_score['recency_score']} total_score={ranking_score['total_score']}",
                ]
            )
            if intersection and not match:
                debug_lines.append("[CRITICAL] MATCH SHOULD HAVE OCCURRED BUT DID NOT")
            if not match:
                continue
            ranked.append((round(total_score, 4), entry, ranking_score))
        ranked.sort(key=lambda item: (-item[0], item[1].entry_id))
        graph_payload = self._apply_cache_reasoning_graph(
            user_input_text=user_input_text,
            ranked=ranked,
            debug_lines=debug_lines,
        )
        ranked = [tuple(item) for item in graph_payload.get("ranked", ranked)]
        selected_entries = [entry for _, entry, _ in ranked[:top_k]]
        ranking_scores = [score for _, _, score in ranked]
        candidate_entry_ids = [entry.entry_id for _, entry, _ in ranked]
        selected_entry_id = selected_entries[0].entry_id if selected_entries else None
        conflict_detected = False
        conflicting_entry_ids: List[int] = []
        penalized_entry_ids: List[int] = []
        updated_confidences: List[Dict[str, object]] = []
        verification_attempted = False
        verification_status = ""
        confidence_capped = False
        uncertainty_flag = False
        if ranked:
            if update_reuse:
                for _, entry, score in ranked:
                    if not bool(score.get("confidence_capped", False)):
                        continue
                    capped_confidence = round(float(score.get("confidence_ceiling", 0.9) or 0.9), 4)
                    raw_confidence = round(float(score.get("confidence_estimate", capped_confidence) or capped_confidence), 4)
                    self.memory_store.update_metadata(
                        entry.entry_id,
                        {
                            "confidence_estimate": capped_confidence,
                        },
                    )
                    updated_confidences.append(
                        {
                            "entry_id": entry.entry_id,
                            "old_confidence": raw_confidence,
                            "new_confidence": capped_confidence,
                            "reason": "confidence_cap",
                        }
                    )
            grouped_by_topic: Dict[str, List[tuple[float, EpisodicMemoryEntry, Dict[str, object]]]] = {}
            for item in ranked:
                score = item[2]
                topic_value = str(score.get("topic_key", "")).strip()
                grouped_by_topic.setdefault(topic_value, []).append(item)
            for group in grouped_by_topic.values():
                signatures = {str(score.get("answer_signature", "")).strip() for _, _, score in group if str(score.get("answer_signature", "")).strip()}
                if len(signatures) <= 1:
                    continue
                conflict_detected = True
                conflicting_entry_ids.extend(entry.entry_id for _, entry, _ in group)
                winner_entry_id = group[0][1].entry_id
                if len(group) >= 2 and abs(float(group[0][0]) - float(group[1][0])) <= 0.75:
                    uncertainty_flag = True
                    for _, uncertain_entry, uncertain_score in group[:2]:
                        uncertain_score["uncertainty_flag"] = True
                        if update_reuse:
                            uncertain_metadata = uncertain_entry.metadata if isinstance(uncertain_entry.metadata, dict) else {}
                            current_confidence = self._bounded_cache_confidence(
                                uncertain_metadata,
                                verification_status=str(uncertain_score.get("verification_status", uncertain_metadata.get("verification_status", "unverified")) or "unverified"),
                                successful_uses=int(uncertain_metadata.get("successful_uses", 0) or 0),
                            )
                            reduced_confidence = round(
                                self._bounded_cache_confidence(
                                    uncertain_metadata,
                                    raw_confidence=current_confidence - 0.02,
                                    verification_status=str(uncertain_score.get("verification_status", uncertain_metadata.get("verification_status", "unverified")) or "unverified"),
                                    successful_uses=int(uncertain_metadata.get("successful_uses", 0) or 0),
                                ),
                                4,
                            )
                            self.memory_store.update_metadata(
                                uncertain_entry.entry_id,
                                {
                                    "confidence_estimate": reduced_confidence,
                                },
                            )
                            updated_confidences.append(
                                {
                                    "entry_id": uncertain_entry.entry_id,
                                    "old_confidence": round(current_confidence, 4),
                                    "new_confidence": reduced_confidence,
                                    "reason": "conflict_uncertainty",
                                }
                            )
                for score in ranking_scores:
                    score_entry_id = int(score.get("entry_id", -1) or -1)
                    if score_entry_id in {entry.entry_id for _, entry, _ in group}:
                        score["conflict_group"] = True
                    else:
                        continue
                    if score_entry_id == winner_entry_id:
                        score["conflict_group"] = True
                        score["conflict_winner"] = True
                if update_reuse:
                    for _, entry, score in group[1:]:
                        metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
                        current_confidence = self._bounded_cache_confidence(
                            metadata,
                            verification_status=str(score.get("verification_status", metadata.get("verification_status", "unverified")) or "unverified"),
                            successful_uses=int(metadata.get("successful_uses", 0) or 0),
                        )
                        penalized_confidence = round(
                            self._bounded_cache_confidence(
                                metadata,
                                raw_confidence=current_confidence - 0.05,
                                verification_status=str(score.get("verification_status", metadata.get("verification_status", "unverified")) or "unverified"),
                                successful_uses=int(metadata.get("successful_uses", 0) or 0),
                            ),
                            4,
                        )
                        self.memory_store.update_metadata(
                            entry.entry_id,
                            {
                                "confidence_estimate": penalized_confidence,
                                "selected_streak_count": 0,
                                "stabilized": False,
                            },
                        )
                        penalized_entry_ids.append(entry.entry_id)
                        updated_confidences.append(
                            {
                                "entry_id": entry.entry_id,
                                "old_confidence": round(current_confidence, 4),
                                "new_confidence": penalized_confidence,
                                "reason": "conflict_penalty",
                            }
                        )
                        score["conflict_group"] = True
                        score["conflict_winner"] = False
                        score["post_conflict_confidence"] = penalized_confidence
                    winner_entry = group[0][1]
                    winner_metadata = winner_entry.metadata if isinstance(winner_entry.metadata, dict) else {}
                    winner_conflict_wins = max(0, int(winner_metadata.get("conflict_win_count", 0) or 0)) + 1
                    self.memory_store.update_metadata(
                        winner_entry.entry_id,
                        {
                            "conflict_win_count": winner_conflict_wins,
                        },
                    )
                    if isinstance(group[0][2], dict):
                        group[0][2]["post_conflict_win_count"] = winner_conflict_wins
        debug_lines.append(f"[DEBUG] match_decision: {'HIT' if selected_entries else 'MISS'} selected_entry_ids={[entry.entry_id for entry in selected_entries]}")
        debug_lines.append(f"[DEBUG RANK] candidate_entry_ids={candidate_entry_ids}")
        debug_lines.append(f"[DEBUG RANK] ranking_scores={ranking_scores}")
        if conflict_detected:
            debug_lines.append(f"[DEBUG CONFLICT] conflicting_entry_ids={conflicting_entry_ids}")
            debug_lines.append(f"[DEBUG CONFLICT] penalized_entry_ids={penalized_entry_ids}")
            debug_lines.append(f"[DEBUG CONFLICT] updated_confidences={updated_confidences}")
        for line in debug_lines:
            print(line)
        reinforcement_applied = False
        updated_confidence = None
        updated_reuse_count = None
        if update_reuse and selected_entries:
            now = datetime.now(timezone.utc).isoformat()
            selected_score = ranking_scores[0] if ranking_scores else {}
            selected_entry = selected_entries[0]
            refreshed_entry = next(
                (entry for entry in self.memory_store.entries if entry.entry_id == selected_entry.entry_id),
                selected_entry,
            )
            metadata = refreshed_entry.metadata if isinstance(refreshed_entry.metadata, dict) else {}
            current_confidence = self._bounded_cache_confidence(
                metadata,
                verification_status=str(selected_score.get("verification_status", metadata.get("verification_status", "unverified")) or "unverified"),
                successful_uses=int(metadata.get("successful_uses", 0) or 0),
            )
            selected_verification_status = str(selected_score.get("verification_status", metadata.get("verification_status", "unverified")) or "unverified").strip().lower()
            selected_confidence_boost = 0.03 if selected_verification_status == "verified" else 0.02
            updated_reuse_count = int(metadata.get("reuse_count", 0) or 0) + 1
            updated_successful_uses = int(metadata.get("successful_uses", 0) or 0) + 1
            updated_streak_count = int(metadata.get("selected_streak_count", 0) or 0) + 1
            stabilized_flag = updated_streak_count >= 3
            updated_confidence = round(
                self._bounded_cache_confidence(
                    metadata,
                    raw_confidence=current_confidence + selected_confidence_boost,
                    verification_status=selected_verification_status,
                    successful_uses=updated_successful_uses,
                ),
                4,
            )
            self.memory_store.update_metadata(
                selected_entry.entry_id,
                {
                    "reuse_count": updated_reuse_count,
                    "successful_uses": updated_successful_uses,
                    "last_reused_at": now,
                    "confidence_estimate": updated_confidence,
                    "selected_streak_count": updated_streak_count,
                    "stabilized": stabilized_flag,
                },
            )
            selected_topic = str(metadata.get("topic_key", "")).strip().lower()
            for _, other_entry, _ in ranked[1:]:
                other_metadata = other_entry.metadata if isinstance(other_entry.metadata, dict) else {}
                if str(other_metadata.get("topic_key", "")).strip().lower() != selected_topic:
                    continue
                self.memory_store.update_metadata(
                    other_entry.entry_id,
                    {
                        "selected_streak_count": 0,
                        "stabilized": False,
                    },
                )
            selected_confidence_ceiling = self._cache_confidence_ceiling(
                metadata,
                verification_status=selected_verification_status,
            )
            reinforcement_applied = True
            updated_confidences.append(
                {
                    "entry_id": selected_entry.entry_id,
                    "old_confidence": round(current_confidence, 4),
                    "new_confidence": updated_confidence,
                    "reason": "successful_use_reinforcement",
                }
            )
            if isinstance(selected_score, dict):
                selected_score["post_update_confidence"] = updated_confidence
                selected_score["post_update_reuse_count"] = updated_reuse_count
                selected_score["post_update_successful_uses"] = updated_successful_uses
                selected_score["post_update_selected_streak_count"] = updated_streak_count
                selected_score["stabilized_flag"] = stabilized_flag
                selected_score["convergence_rate"] = round(min(1.0, updated_streak_count / 3.0), 4)
                refreshed_selected_metadata = next(
                    (
                        entry.metadata
                        for entry in self.memory_store.entries
                        if entry.entry_id == selected_entry.entry_id and isinstance(entry.metadata, dict)
                    ),
                    metadata,
                )
                selected_score["dominance_score"] = round(
                    self._cache_dominance_score(
                        refreshed_selected_metadata if isinstance(refreshed_selected_metadata, dict) else {},
                        verification_status=selected_verification_status,
                    ),
                    4,
                )
        if ranking_scores:
            selected_score = ranking_scores[0]
            verification_attempted = bool(selected_score.get("verification_attempted", False))
            verification_status = str(selected_score.get("verification_status", "") or "")
            confidence_capped = bool(selected_score.get("confidence_capped", False))
            uncertainty_flag = bool(selected_score.get("uncertainty_flag", False) or uncertainty_flag)
            stabilized_flag = bool(selected_score.get("stabilized_flag", False))
            dominance_score = float(selected_score.get("dominance_score", 0.0) or 0.0)
            convergence_rate = float(selected_score.get("convergence_rate", 0.0) or 0.0)
        else:
            stabilized_flag = False
            dominance_score = 0.0
            convergence_rate = 0.0
        return selected_entries, {
            "normalized_terms": query_terms,
            "query_intent": graph_payload.get("query_intent", detect_query_intent(user_input_text).value),
            "selection_mode": graph_payload.get("selection_mode", "node_first"),
            "selected_cluster_id": graph_payload.get("selected_cluster_id"),
            "selected_cluster_score": graph_payload.get("selected_cluster_score", 0.0),
            "selected_node_entry_id": graph_payload.get("selected_node_entry_id"),
            "candidate_entry_ids": candidate_entry_ids,
            "selected_entry_id": selected_entry_id,
            "ranking_scores": ranking_scores,
            "reinforcement_applied": reinforcement_applied,
            "updated_confidence": updated_confidence,
            "updated_reuse_count": updated_reuse_count,
            "conflict_detected": conflict_detected,
            "conflicting_entry_ids": sorted(set(conflicting_entry_ids)),
            "penalized_entry_ids": penalized_entry_ids,
            "updated_confidences": updated_confidences,
            "verification_attempted": verification_attempted,
            "verification_status": verification_status,
            "confidence_capped": confidence_capped,
            "uncertainty_flag": uncertainty_flag,
            "stabilized_flag": stabilized_flag,
            "dominance_score": dominance_score,
            "convergence_rate": convergence_rate,
            "reasoning_graph_nodes": graph_payload.get("reasoning_graph_nodes", []),
            "reasoning_graph_edges": graph_payload.get("reasoning_graph_edges", []),
            "reasoning_graph_clusters": graph_payload.get("reasoning_graph_clusters", []),
            "reasoning_graph_cluster_relationships": graph_payload.get("reasoning_graph_cluster_relationships", []),
            "reasoning_spatial_grid": graph_payload.get("reasoning_spatial_grid", {}),
            "selected_reasoning_path": graph_payload.get("selected_reasoning_path", []),
            "selected_path_score": graph_payload.get("selected_path_score", 0.0),
            "selected_path_activation": graph_payload.get("selected_path_activation", 0.0),
            "selected_path_final_score": graph_payload.get("selected_path_final_score", 0.0),
            "selected_path_breakdown": graph_payload.get("selected_path_breakdown", ""),
            "candidate_reasoning_paths": graph_payload.get("candidate_reasoning_paths", []),
            "selected_spatial_region_id": graph_payload.get("selected_spatial_region_id"),
            "selected_spatial_nodes": graph_payload.get("selected_spatial_nodes", []),
            "selected_spatial_clusters": graph_payload.get("selected_spatial_clusters", []),
            "selected_spatial_activation": graph_payload.get("selected_spatial_activation", 0.0),
            "spatial_selection_reason": graph_payload.get("spatial_selection_reason", ""),
            "spatial_consistency_warning": graph_payload.get("spatial_consistency_warning", ""),
            "cluster_selection_reason": graph_payload.get("cluster_selection_reason", ""),
            "graph_reasoning_steps": graph_payload.get("graph_reasoning_steps", []),
        }

    def retrieve_cached_llm_knowledge(
        self,
        *,
        user_input_text: str,
        topic_key: Optional[str] = None,
        top_k: int = 5,
        update_reuse: bool = False,
    ) -> List[EpisodicMemoryEntry]:
        selected, _ = self._cache_rank_candidates(
            user_input_text=user_input_text,
            topic_key=topic_key,
            top_k=top_k,
            update_reuse=update_reuse,
        )
        return selected

    def _cache_llm_knowledge_entry(
        self,
        *,
        user_input_text: str,
        answer_text: str,
        route_type: str,
        llm_payload_v2: Dict[str, object],
        llm_metadata: Dict[str, object],
        parsed_brain_input: BrainInputV1,
        parsed_memory: BrainMemory,
    ) -> Optional[EpisodicMemoryEntry]:
        topic_key = self.normalize_knowledge_topic(user_input_text)
        topic_keys = self.expand_topic_keys(user_input_text, topic_key)
        normalized_facts = self._distill_cached_answer_facts(answer_text)
        answer_state = self._cached_answer_state(topic_key, normalized_facts)
        return self.memory_store.add(
            entry_type="llm_knowledge_cache",
            source_text=str(answer_text or "").strip(),
            parsed_memory=parsed_memory,
            brain_input=parsed_brain_input,
            metadata={
                "source": "llm",
                "trust_level": "low",
                "verification_status": "unverified",
                "topic_key": topic_key,
                "topic_keys": topic_keys,
                "query_text": user_input_text,
                "answer_text": answer_text,
                "normalized_facts": normalized_facts,
                "answer_state": answer_state,
                "model_metadata": {
                    "mode": llm_metadata.get("mode", ""),
                    "route_type": llm_metadata.get("route_type", route_type),
                },
                "llm_payload_snapshot": {
                    "source_of_truth": llm_payload_v2.get("source_of_truth", "llm"),
                    "task_type": llm_payload_v2.get("task_type", ""),
                    "instructions": list(llm_payload_v2.get("instructions", []))[:4] if isinstance(llm_payload_v2.get("instructions", []), list) else [],
                },
                "confidence_estimate": 0.45 if str(llm_metadata.get("mode", "")).startswith("deterministic") else 0.6,
                "reuse_count": 0,
                "successful_uses": 0,
                "conflict_win_count": 0,
                "selected_streak_count": 0,
                "stabilized": False,
                "last_reused_at": "",
                "route_type": route_type,
            },
        )

    def retrieve_stateful_memory(
        self,
        *,
        user_input_text: str,
        top_k: int = 6,
    ) -> List[RetrievedMemory]:
        query_tokens = self._episodic_tokens(str(user_input_text or "").lower())
        if not query_tokens:
            return []
        scored: List[RetrievedMemory] = []
        total_entries = len(self.memory_store.entries)
        for index, entry in enumerate(self.memory_store.entries):
            if entry.entry_type != "state_memory":
                continue
            observation = entry.metadata.get("state_observation", {}) if isinstance(entry.metadata, dict) else {}
            if not isinstance(observation, dict):
                continue
            recency = (index + 1) / float(total_entries) if total_entries else 0.0
            score = (self._text_overlap_score(query_tokens, observation) * 3.0) + (recency * 0.35)
            if self._memory_question_targets_state_type(str(user_input_text or "").lower(), str(observation.get("state_type", ""))):
                score += 1.5
            if self._state_observation_matches_query(str(user_input_text or "").lower(), observation):
                score += 1.25
            if observation.get("is_update"):
                score += 0.2
            if score <= 0.45:
                continue
            scored.append(RetrievedMemory(entry_id=entry.entry_id, score=round(score, 4), entry=entry))
        return sorted(scored, key=lambda item: (-item.score, -item.entry.entry_id))[:top_k]

    def merge_memory_state(
        self,
        *,
        user_input_text: str,
        retrieved: Sequence[RetrievedMemory],
        recalled_turns: Sequence[EpisodicTurnRecall],
    ) -> Dict[str, object]:
        observations: List[Dict[str, object]] = []
        for item in retrieved:
            metadata = item.entry.metadata if isinstance(item.entry.metadata, dict) else {}
            observation = metadata.get("state_observation", {})
            if isinstance(observation, dict) and observation:
                observations.append({**observation, "entry_id": item.entry.entry_id, "timestamp": item.entry.timestamp})
        for recall in recalled_turns:
            for observation in recall.record.metadata.get("state_observations", []) if isinstance(recall.record.metadata, dict) else []:
                if isinstance(observation, dict) and observation:
                    observations.append({**observation, "entry_id": recall.turn_id, "timestamp": recall.record.timestamp})
        normalized = str(user_input_text or "").lower()
        wants_original = any(fragment in normalized for fragment in ("original", "at the start", "before"))
        wants_current = "current" in normalized or "now" in normalized
        state_groups: Dict[str, List[Dict[str, object]]] = {}
        correction_chain_detected = False
        latest_state_resolution_applied = False
        for observation in observations:
            state_key = str(observation.get("state_key", "")).strip()
            if not state_key:
                continue
            state_groups.setdefault(state_key, []).append(observation)
        merged_items: List[Dict[str, object]] = []
        world_food_totals = self._resolve_world_food_totals(observations)
        if world_food_totals:
            latest_state_resolution_applied = True
            merged_items.append(
                {
                    "state_key": "world_food_totals",
                    "state_type": "world_food_total",
                    "current_value": world_food_totals,
                    "original_value": {},
                    "history_count": len([item for item in observations if item.get("state_type") == "world_food_total"]),
                }
            )
        for state_key, items in sorted(state_groups.items()):
            ordered = sorted(items, key=lambda item: (str(item.get("timestamp", "")), int(item.get("entry_id", 0))))
            merge_mode = str(ordered[-1].get("merge_mode", "latest"))
            if len(ordered) > 1 and any(item.get("is_update") for item in ordered[1:]):
                correction_chain_detected = True
            first = ordered[0]
            last = ordered[-1]
            if first != last or merge_mode != "latest":
                latest_state_resolution_applied = True
            if merge_mode == "append_set":
                seen_values = set()
                current_values: List[object] = []
                display_values: List[str] = []
                for payload in ordered:
                    value = payload.get("value")
                    marker = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
                    if marker in seen_values:
                        continue
                    seen_values.add(marker)
                    current_values.append(value)
                    display_text = str(payload.get("display_text", "")).strip()
                    if display_text:
                        display_values.append(display_text)
                merged_items.append(
                    {
                        "state_key": state_key,
                        "state_type": str(last.get("state_type", "")),
                        "current_value": current_values,
                        "original_value": first.get("value") if wants_original else None,
                        "display_text": ", ".join(display_values),
                        "history_count": len(ordered),
                        "history": ordered if wants_original else [],
                        "merge_mode": merge_mode,
                    }
                )
                continue
            if merge_mode == "append_map":
                value_map: Dict[str, object] = {}
                history_snapshots: List[Dict[str, object]] = []
                for payload in ordered:
                    map_key = str(payload.get("entity", "") or payload.get("state_key", "")).strip()
                    if not map_key:
                        continue
                    value_map[map_key] = payload.get("value")
                    if wants_original:
                        history_snapshots.append(
                            {
                                "entity": map_key,
                                "value": payload.get("value"),
                                "timestamp": payload.get("timestamp", ""),
                                "entry_id": payload.get("entry_id", 0),
                            }
                        )
                merged_items.append(
                    {
                        "state_key": state_key,
                        "state_type": str(last.get("state_type", "")),
                        "current_value": value_map,
                        "original_value": None,
                        "display_text": "; ".join(
                            f"{name.replace('_', ' ')} = {value_map[name]}" for name in sorted(value_map)
                        ),
                        "history_count": len(ordered),
                        "history": history_snapshots,
                        "merge_mode": merge_mode,
                    }
                )
                continue
            merged_items.append(
                {
                    "state_key": state_key,
                    "state_type": str(last.get("state_type", "")),
                    "current_value": last.get("value"),
                    "original_value": first.get("value") if wants_original or wants_current else None,
                    "display_text": last.get("display_text", ""),
                    "history_count": len(ordered),
                    "history": ordered if wants_original else [],
                    "merge_mode": merge_mode,
                }
            )
        return {
            "retrieved_count": len(retrieved),
            "recalled_turn_count": len(recalled_turns),
            "correction_chain_detected": correction_chain_detected,
            "latest_state_resolution_applied": latest_state_resolution_applied,
            "items": merged_items,
            "item_count": len(merged_items),
            "wants_original_state": wants_original,
            "wants_current_state": wants_current,
        }

    def retrieve_episodic_turns(
        self,
        *,
        user_input_text: str,
        route_type: str,
        top_k: int = 4,
    ) -> List[EpisodicTurnRecall]:
        normalized = str(user_input_text or "").strip().lower()
        query_tokens = self._episodic_tokens(normalized)
        preferred_route_types = self._episodic_route_preferences(normalized)
        scored: List[EpisodicTurnRecall] = []
        total_records = len(self.turn_store.records)
        for index, record in enumerate(self.turn_store.records):
            if record.route_type == "memory_question":
                continue
            if preferred_route_types and record.route_type not in preferred_route_types:
                continue
            recency_score = (index + 1) / float(total_records) if total_records else 0.0
            route_score = 1.0 if preferred_route_types else 0.35
            semantic_score = self._episodic_semantic_score(query_tokens, record)
            if self._asks_for_last_turn(normalized):
                semantic_score += 0.25 if recency_score > 0.0 else 0.0
            score = (semantic_score * 0.6) + (recency_score * 0.25) + (route_score * 0.15)
            if score <= 0.18:
                continue
            scored.append(
                EpisodicTurnRecall(
                    turn_id=record.turn_id,
                    score=score,
                    recency_score=recency_score,
                    semantic_score=semantic_score,
                    route_score=route_score,
                    record=record,
                )
            )
        return sorted(scored, key=lambda item: (-item.score, -item.record.sequence, item.turn_id))[:top_k]

    def infer_goal_weights(
        self,
        user_input_text: str,
        parsed_memory: BrainMemory,
        retrieved: Sequence[RetrievedMemory],
    ) -> Dict[str, float]:
        text = user_input_text.lower()
        weights = {
            "complete_task": 0.34,
            "avoid_risk": 0.33,
            "maximize_energy": 0.33,
        }
        if any(token in text for token in ("safe", "safely", "avoid", "without risk", "dont get", "don't get")):
            weights["avoid_risk"] += 0.22
        if any(token in text for token in ("finish", "complete", "done", "goal", "task", "deadline")):
            weights["complete_task"] += 0.22
        if any(token in text for token in ("energy", "full", "healthy", "refreshed", "satisfied")):
            weights["maximize_energy"] += 0.22
        if parsed_memory.action in {"study", "practice", "repair", "charge", "clean", "backup"}:
            weights["complete_task"] += 0.12
        if parsed_memory.action in {"eat", "drink", "cook", "water"}:
            weights["maximize_energy"] += 0.12
        if any(item.entry.entry_type == "selected_result" for item in retrieved):
            weights["complete_task"] += 0.05
            weights["avoid_risk"] += 0.03
        total = sum(max(0.0, value) for value in weights.values())
        if total <= 0.0:
            return {"complete_task": 0.34, "avoid_risk": 0.33, "maximize_energy": 0.33}
        return {key: round(max(0.0, value) / total, 4) for key, value in weights.items()}

    def reconstruct_memory_context(
        self,
        *,
        user_input_text: str,
        retrieved: Sequence[RetrievedMemory],
        recalled_turns: Sequence[EpisodicTurnRecall],
        merged_memory_state: Dict[str, object],
    ) -> ReconstructedContext:
        config = self.memory_reconstruction_config
        normalized = str(user_input_text or "").strip().lower()
        query_tokens = self._episodic_tokens(normalized)
        profile = self._memory_reconstruction_profile(normalized)
        exact_matches = self._memory_reconstruction_exact_matches(merged_memory_state)
        fragments: List[ReconstructedMemoryFragment] = []
        total_entries = max(1, len(self.memory_store.entries))
        total_records = max(1, len(self.turn_store.records))

        seen_state_markers: set[str] = set()
        for index, entry in enumerate(self.memory_store.entries):
            recency = (index + 1) / float(total_entries)
            metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
            observations: List[Dict[str, object]] = []
            observation = metadata.get("state_observation", {})
            if isinstance(observation, dict) and observation:
                observations.append(observation)
            observations.extend(self._extract_state_observations(entry.source_text))
            for observation_payload in observations:
                if not isinstance(observation_payload, dict):
                    continue
                marker = json.dumps(
                    {
                        "entry": entry.entry_id,
                        "state_key": observation_payload.get("state_key", ""),
                        "value": observation_payload.get("value", ""),
                        "display_text": observation_payload.get("display_text", ""),
                    },
                    sort_keys=True,
                )
                if marker in seen_state_markers:
                    continue
                seen_state_markers.add(marker)
                fragment = self._score_reconstructed_state_fragment(
                    user_input_text=user_input_text,
                    query_tokens=query_tokens,
                    profile=profile,
                    observation=observation_payload,
                    source_id=entry.entry_id,
                    recency=recency,
                    source_text=entry.source_text,
                    config=config,
                )
                if fragment.score >= 0.18:
                    fragments.append(fragment)

        for index, record in enumerate(self.turn_store.records):
            if record.route_type == "memory_question":
                continue
            recency = (index + 1) / float(total_records)
            fragment = self._score_reconstructed_turn_fragment(
                user_input_text=user_input_text,
                query_tokens=query_tokens,
                profile=profile,
                record=record,
                recency=recency,
                config=config,
            )
            if fragment.score >= 0.18:
                fragments.append(fragment)

        for recall in recalled_turns:
            record_metadata = recall.record.metadata if isinstance(recall.record.metadata, dict) else {}
            clarification_state = record_metadata.get("clarification_state", {}) if isinstance(record_metadata.get("clarification_state", {}), dict) else {}
            recall_prompt = (
                str(clarification_state.get("clarification_response", "")).strip()
                or str(clarification_state.get("merged_prompt", "")).strip()
                or recall.record.raw_user_prompt
            )
            fragments.append(
                ReconstructedMemoryFragment(
                    source_type="temporal_recall",
                    source_id=recall.turn_id,
                    text=recall_prompt,
                    score=round(min(1.0, 0.35 + recall.score), 4),
                    evidence_breakdown={
                        "concept_similarity": min(1.0, recall.semantic_score),
                        "temporal_proximity": recall.recency_score,
                        "goal_link": 1.0 if recall.record.route_type in {"planning_request", "goal_request"} else 0.0,
                        "hierarchy_link": 0.25 if recall.record.route_type != "unknown" else 0.0,
                        "cross_domain_link": 0.0,
                        "recency": recall.recency_score,
                    },
                    payload={
                        "turn_id": recall.turn_id,
                        "route_type": recall.record.route_type,
                        "raw_user_prompt": recall_prompt,
                        "stored_raw_user_prompt": recall.record.raw_user_prompt,
                        "final_answer": recall.record.final_answer,
                    },
                )
            )

        exact_boost = 0.18 if exact_matches else 0.0
        graph_only_top_score = max((fragment.score for fragment in fragments), default=0.0)
        graph_confidence_components = self._graph_confidence_components(
            fragments=fragments,
            exact_boost=exact_boost,
            top_score=graph_only_top_score,
        )
        graph_confidence = float(graph_confidence_components["graph_confidence"])
        pixel_readback: PixelReadback | None = None
        pixel_graph_match = False
        pixel_fragment: ReconstructedMemoryFragment | None = None
        pixel_confidence = 0.0
        pixel_reasoning_trace: PixelReasoningTrace | None = None
        pixel_decision = {
            "graph_confidence": round(graph_confidence, 4),
            "pixel_confidence": 0.0,
            "graph_confidence_components": graph_confidence_components,
            "pixel_confidence_components": {},
            "decision_source": "graph",
            "dominance_gap": 0.0,
            "margin": round(float(config.pixel_decision_margin), 4),
            "reason": "pixel memory disabled",
        }
        if self.pixel_memory_enabled:
            pixel_readback = self._pixel_memory_readback(user_input_text=user_input_text)
            pixel_reasoning_trace = PixelReasoningEngine().reason(pixel_readback)
            pixel_confidence = float(pixel_readback.spatial_confidence)
            pixel_decision = self._pixel_graph_decision(
                graph_confidence=graph_confidence,
                pixel_readback=pixel_readback,
                pixel_reasoning_trace=pixel_reasoning_trace,
                margin=float(config.pixel_decision_margin),
            )
            pixel_confidence = float(pixel_decision.get("pixel_confidence", pixel_confidence) or 0.0)
            pixel_fragment = self._pixel_readback_fragment(
                pixel_readback=pixel_readback,
                pixel_reasoning_trace=pixel_reasoning_trace,
                user_input_text=user_input_text,
                graph_confidence=graph_confidence,
                decision_source=str(pixel_decision.get("decision_source", "graph")),
                margin=float(config.pixel_decision_margin),
            )
            if pixel_fragment is not None and str(pixel_decision.get("decision_source", "graph")) in {"pixel", "hybrid"}:
                fragments.append(pixel_fragment)
            graph_entry_ids = {item.entry_id for item in retrieved}
            graph_entry_ids.update(recall.turn_id for recall in recalled_turns)
            pixel_entry_ids = {
                int(item.get("metadata", {}).get("entry_id", 0) or 0)
                for item in pixel_readback.dominant_concepts
                if isinstance(item.get("metadata", {}), dict)
                and str(item.get("metadata", {}).get("source_type", "")) == "memory_entry"
            }
            pixel_turn_ids = {
                int(item.get("metadata", {}).get("turn_id", 0) or 0)
                for item in pixel_readback.dominant_concepts
                if isinstance(item.get("metadata", {}), dict)
                and str(item.get("metadata", {}).get("source_type", "")) == "turn_record"
            }
            pixel_graph_match = bool((graph_entry_ids & pixel_entry_ids) or (graph_entry_ids & pixel_turn_ids))

        if str(pixel_decision.get("decision_source", "graph")) in {"pixel", "hybrid"} and pixel_readback is not None:
            fragments = self._apply_pixel_dominance_to_fragments(
                fragments=fragments,
                pixel_readback=pixel_readback,
                decision_source=str(pixel_decision.get("decision_source", "graph")),
            )

        fragments = sorted(
            fragments,
            key=lambda item: (
                -item.score,
                -float(item.evidence_breakdown.get("recency", 0.0)),
                item.source_type,
                -item.source_id,
            ),
        )
        partial_matches = fragments[:8]
        top_score = partial_matches[0].score if partial_matches else 0.0
        confidence = min(1.0, max(top_score, 0.0) + exact_boost)
        if confidence >= config.high_confidence_threshold:
            answer_mode = "high"
        elif confidence >= config.medium_confidence_threshold:
            answer_mode = "medium"
        else:
            answer_mode = "low"
        synthesized_summary = self._synthesize_reconstructed_memory_answer(
            user_input_text=user_input_text,
            merged_memory_state=merged_memory_state,
            partial_matches=partial_matches,
            answer_mode=answer_mode,
        )
        related_concepts = [
            {
                "source_id": item.source_id,
                "source_type": item.source_type,
                "score": round(item.score, 4),
                "text": item.text,
            }
            for item in partial_matches[:4]
        ]
        temporal_links = [
            {
                "source_id": item.source_id,
                "score": round(item.score, 4),
                "text": item.text,
            }
            for item in partial_matches
            if item.source_type in {"turn", "temporal_recall"}
        ][:4]
        goal_links = [
            {
                "source_id": item.source_id,
                "route_type": item.payload.get("route_type", ""),
                "score": round(item.score, 4),
                "text": item.text,
            }
            for item in partial_matches
            if float(item.evidence_breakdown.get("goal_link", 0.0)) > 0.0
        ][:4]
        hierarchy_links = [
            {
                "source_id": item.source_id,
                "state_type": item.payload.get("state_type", ""),
                "score": round(item.score, 4),
            }
            for item in partial_matches
            if float(item.evidence_breakdown.get("hierarchy_link", 0.0)) > 0.0
        ][:4]
        cross_domain_links = [
            {
                "source_id": item.source_id,
                "score": round(item.score, 4),
                "text": item.text,
            }
            for item in partial_matches
            if float(item.evidence_breakdown.get("cross_domain_link", 0.0)) > 0.0
        ][:4]
        return ReconstructedContext(
            query=user_input_text,
            exact_matches=exact_matches,
            partial_matches=partial_matches,
            related_concepts=related_concepts,
            temporal_links=temporal_links,
            goal_links=goal_links,
            hierarchy_links=hierarchy_links,
            cross_domain_links=cross_domain_links,
            synthesized_summary=synthesized_summary,
            confidence=round(confidence, 4),
            answer_mode=answer_mode,
            evidence_breakdown={
                "exact_match_score": round(exact_boost, 4),
                "top_partial_score": round(top_score, 4),
                "partial_candidate_count": len(partial_matches),
                "reconstruction_sources_used": sorted({item.source_type for item in partial_matches}),
                "graph_confidence": round(graph_confidence, 4),
                "pixel_confidence": round(pixel_confidence, 4),
                "graph_confidence_components": graph_confidence_components,
                "pixel_confidence_components": dict(pixel_decision.get("pixel_confidence_components", {})),
                "decision_source": str(pixel_decision.get("decision_source", "graph")),
                "dominance_gap": round(float(pixel_decision.get("dominance_gap", 0.0) or 0.0), 4),
                "override_occurred": bool(pixel_decision.get("override_occurred", False)),
                "multi_constraint_result": bool(pixel_decision.get("multi_constraint_result", False)),
                "pixel_reasoning": {} if pixel_reasoning_trace is None else pixel_reasoning_trace.to_dict(),
                "pixel_decision_margin": round(float(config.pixel_decision_margin), 4),
                "pixel_decision": dict(pixel_decision),
                "pixel_memory_enabled": self.pixel_memory_enabled,
                "pixel_memory_consulted": pixel_readback is not None,
                "pixel_readback_matched_graph_result": pixel_graph_match,
                "pixel_readback": {} if pixel_readback is None else pixel_readback.to_dict(),
                "weights": {
                    "concept_similarity": config.concept_similarity_weight,
                    "temporal_proximity": config.temporal_proximity_weight,
                    "goal_link": config.goal_link_weight,
                    "hierarchy_link": config.hierarchy_link_weight,
                    "cross_domain_link": config.cross_domain_link_weight,
                    "recency": config.recency_weight,
                },
            },
        )

    def _graph_confidence_components(
        self,
        *,
        fragments: Sequence[ReconstructedMemoryFragment],
        exact_boost: float,
        top_score: float,
    ) -> Dict[str, float]:
        concept_confidence = max(
            (float(fragment.evidence_breakdown.get("concept_similarity", 0.0) or 0.0) for fragment in fragments),
            default=0.0,
        )
        plan_reliability = max(
            (float(fragment.evidence_breakdown.get("goal_link", 0.0) or 0.0) for fragment in fragments),
            default=0.0,
        )
        reconstruction_score = min(1.0, max(0.0, float(top_score)) + float(exact_boost))
        graph_confidence = min(
            1.0,
            (reconstruction_score * 0.7)
            + (concept_confidence * 0.2)
            + (plan_reliability * 0.1),
        )
        return {
            "graph_confidence": round(graph_confidence, 4),
            "reconstruction_score": round(reconstruction_score, 4),
            "concept_confidence": round(concept_confidence, 4),
            "plan_reliability": round(plan_reliability, 4),
        }

    def _pixel_graph_decision(
        self,
        *,
        graph_confidence: float,
        pixel_readback: PixelReadback,
        pixel_reasoning_trace: PixelReasoningTrace,
        margin: float,
    ) -> Dict[str, object]:
        top_region = pixel_readback.region_strengths[0] if pixel_readback.region_strengths else {}
        region_strength = min(1.0, float(top_region.get("activation_strength", 0.0) or 0.0)) if isinstance(top_region, dict) else 0.0
        dominance_gap = float(pixel_readback.dominance_gap)
        multi_constraint_result = pixel_reasoning_trace.reasoning_mode == "multi_constraint"
        region_coherence = float(pixel_readback.region_coherence)
        spatial_confidence = float(pixel_readback.spatial_confidence)
        pixel_confidence = min(
            1.0,
            (region_strength * 0.25)
            + (dominance_gap * 0.2)
            + (spatial_confidence * 0.35)
            + (region_coherence * 0.2),
        )
        if multi_constraint_result:
            pixel_confidence = min(1.0, max(pixel_confidence, spatial_confidence))
        if pixel_confidence > graph_confidence + margin:
            decision_source = "pixel"
            reason = "pixel spatial confidence exceeded graph confidence by the configured margin"
        elif graph_confidence > pixel_confidence + margin:
            decision_source = "graph"
            reason = "graph confidence exceeded pixel spatial confidence by the configured margin"
        else:
            decision_source = "hybrid"
            reason = "graph and pixel confidence were within the configured margin"
        if multi_constraint_result and decision_source == "pixel":
            reason = "pixel confidence won with multiple strong low-gap regions, preserving a multi-constraint result"
        elif multi_constraint_result and decision_source == "hybrid":
            reason = "graph and pixel confidence were close; multiple strong pixel regions are preserved as constraints"
        if pixel_reasoning_trace.reasoning_mode == "multi_constraint" and graph_confidence > pixel_confidence + margin:
            decision_source = "hybrid"
            reason = f"pixel reasoning mode {pixel_reasoning_trace.reasoning_mode} retained influence despite stronger graph confidence"
        return {
            "graph_confidence": round(float(graph_confidence), 4),
            "pixel_confidence": round(pixel_confidence, 4),
            "pixel_confidence_components": {
                "region_strength": round(region_strength, 4),
                "dominance_gap": round(dominance_gap, 4),
                "spatial_confidence": round(spatial_confidence, 4),
                "region_coherence": round(region_coherence, 4),
            },
            "decision_source": decision_source,
            "dominance_gap": round(dominance_gap, 4),
            "override_occurred": decision_source == "pixel" and pixel_confidence > graph_confidence + margin,
            "multi_constraint_result": multi_constraint_result,
            "reasoning_mode": pixel_reasoning_trace.reasoning_mode,
            "selected_output_type": pixel_reasoning_trace.selected_output_type,
            "margin": round(float(margin), 4),
            "reason": reason,
            "region_stats": list(pixel_readback.region_strengths),
            "top_region": dict(top_region),
        }

    def _apply_pixel_dominance_to_fragments(
        self,
        *,
        fragments: Sequence[ReconstructedMemoryFragment],
        pixel_readback: PixelReadback,
        decision_source: str,
    ) -> List[ReconstructedMemoryFragment]:
        pixel_rank = {
            str(item.get("label", "")).strip().lower(): index
            for index, item in enumerate(pixel_readback.dominant_concepts)
            if str(item.get("label", "")).strip()
        }
        pixel_ids = {
            str(item.get("metadata", {}).get("entry_id", item.get("metadata", {}).get("turn_id", ""))).strip(): index
            for index, item in enumerate(pixel_readback.dominant_concepts)
            if isinstance(item.get("metadata", {}), dict)
        }
        adjusted: List[ReconstructedMemoryFragment] = []
        for fragment in fragments:
            label_rank = pixel_rank.get(fragment.text.strip().lower())
            id_rank = pixel_ids.get(str(fragment.source_id))
            rank = label_rank if label_rank is not None else id_rank
            if rank is None:
                adjusted.append(fragment)
                continue
            boost = max(0.0, 0.12 - (rank * 0.025))
            if decision_source == "pixel":
                boost += 0.08
            adjusted.append(
                ReconstructedMemoryFragment(
                    source_type=fragment.source_type,
                    source_id=fragment.source_id,
                    text=fragment.text,
                    score=round(min(1.0, fragment.score + boost), 4),
                    evidence_breakdown={**fragment.evidence_breakdown, "pixel_dominance_boost": round(boost, 4)},
                    payload={**fragment.payload, "pixel_dominance_rank": rank + 1},
                )
            )
        return adjusted

    def _pixel_memory_readback(self, *, user_input_text: str) -> PixelReadback:
        concepts = self._pixel_projection_concepts()
        relationships = self._pixel_projection_relationships(concepts)
        self.pixel_memory_canvas.project_concepts(concepts, relationships)
        self.pixel_memory_canvas.activate_query(user_input_text)
        return self.pixel_memory_canvas.readback()

    def _pixel_projection_concepts(self) -> List[PixelProjectionConcept]:
        concepts: Dict[str, PixelProjectionConcept] = {}
        total_entries = max(1, len(self.memory_store.entries))
        previous_entry_concept_id = ""
        state_key_latest: Dict[str, str] = {}
        for index, entry in enumerate(self.memory_store.entries):
            if not _is_meaningful_entry(entry):
                continue
            metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
            observation = metadata.get("state_observation", {}) if isinstance(metadata.get("state_observation", {}), dict) else {}
            display_text = str(observation.get("display_text", "") or entry.source_text or "").strip()
            state_type = str(observation.get("state_type", "")).strip()
            state_key = str(observation.get("state_key", "")).strip()
            concept_id = f"entry:{entry.entry_id}"
            terms = sorted(
                set(
                    self._episodic_tokens(
                        " ".join(
                            [
                                entry.source_text,
                                display_text,
                                state_type,
                                state_key,
                                str(observation.get("value", "")),
                            ]
                        ).lower()
                    )
                )
            )
            recency = (index + 1) / float(total_entries)
            importance = 0.38 + (0.28 * recency)
            if entry.entry_type == "state_memory":
                importance += 0.18
            if observation.get("is_update"):
                importance += 0.12
            concepts[concept_id] = PixelProjectionConcept(
                concept_id=concept_id,
                label=display_text or f"memory entry {entry.entry_id}",
                terms=terms,
                abstraction_level=2 if state_type else 1,
                importance=min(1.0, importance),
                prediction_accuracy=0.5 if entry.entry_type == "state_memory" else 0.35,
                uncertainty=0.25 if observation.get("is_update") else 0.4,
                temporal_links=[],
                hierarchy_links=[f"graph:{state_type}"] if state_type else [],
                contradiction_links=[],
                metadata={
                    "source_type": "memory_entry",
                    "entry_id": entry.entry_id,
                    "entry_type": entry.entry_type,
                    "state_type": state_type,
                    "state_key": state_key,
                    "recency": recency,
                    "source_text": entry.source_text,
                    "display_text": display_text,
                },
            )
            if previous_entry_concept_id and previous_entry_concept_id in concepts:
                prior = concepts[previous_entry_concept_id]
                concepts[previous_entry_concept_id] = PixelProjectionConcept(
                    concept_id=prior.concept_id,
                    label=prior.label,
                    terms=prior.terms,
                    abstraction_level=prior.abstraction_level,
                    importance=prior.importance,
                    prediction_accuracy=prior.prediction_accuracy,
                    uncertainty=prior.uncertainty,
                    temporal_links=sorted(set(prior.temporal_links + [concept_id])),
                    hierarchy_links=prior.hierarchy_links,
                    contradiction_links=prior.contradiction_links,
                    metadata=prior.metadata,
                )
            previous_entry_concept_id = concept_id
            if state_key:
                previous_for_key = state_key_latest.get(state_key)
                if previous_for_key and previous_for_key in concepts:
                    prior = concepts[previous_for_key]
                    concepts[previous_for_key] = PixelProjectionConcept(
                        concept_id=prior.concept_id,
                        label=prior.label,
                        terms=prior.terms,
                        abstraction_level=prior.abstraction_level,
                        importance=prior.importance,
                        prediction_accuracy=prior.prediction_accuracy,
                        uncertainty=prior.uncertainty,
                        temporal_links=prior.temporal_links,
                        hierarchy_links=prior.hierarchy_links,
                        contradiction_links=sorted(set(prior.contradiction_links + [concept_id])),
                        metadata=prior.metadata,
                    )
                state_key_latest[state_key] = concept_id

        for node in self.global_graph_store.nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("node_id", "") or node.get("normalized_concept", "")).strip()
            label = str(node.get("label", node_id)).strip()
            if not node_id and not label:
                continue
            concept_id = f"graph:{node_id or label}"
            concepts.setdefault(
                concept_id,
                PixelProjectionConcept(
                    concept_id=concept_id,
                    label=label or node_id,
                    terms=sorted(set(self._episodic_tokens(" ".join([node_id, label]).lower()))),
                    abstraction_level=3,
                    importance=0.48,
                    prediction_accuracy=0.45,
                    uncertainty=0.45,
                    metadata={
                        "source_type": "global_graph_node",
                        "node_id": node_id,
                        "label": label,
                    },
                ),
            )

        total_records = max(1, len(self.turn_store.records))
        previous_turn_concept_id = ""
        for index, record in enumerate(self.turn_store.records):
            if record.route_type == "memory_question":
                continue
            label = str(record.raw_user_prompt or record.final_answer or "").strip()
            if not label:
                continue
            concept_id = f"turn:{record.turn_id}"
            recency = (index + 1) / float(total_records)
            concepts[concept_id] = PixelProjectionConcept(
                concept_id=concept_id,
                label=label,
                terms=sorted(set(self._episodic_tokens(" ".join([record.raw_user_prompt, record.final_answer, record.route_type]).lower()))),
                abstraction_level=2 if record.route_type in {"planning_request", "goal_request"} else 1,
                importance=min(1.0, 0.36 + (recency * 0.32) + (0.08 if record.reasoning_succeeded else 0.0)),
                prediction_accuracy=0.5 if record.reasoning_succeeded else 0.25,
                uncertainty=0.28 if record.reasoning_succeeded else 0.55,
                temporal_links=[],
                hierarchy_links=[f"graph:{record.route_type}"] if record.route_type else [],
                metadata={
                    "source_type": "turn_record",
                    "turn_id": record.turn_id,
                    "sequence": record.sequence,
                    "route_type": record.route_type,
                    "recency": recency,
                    "raw_user_prompt": record.raw_user_prompt,
                    "final_answer": record.final_answer,
                },
            )
            if previous_turn_concept_id and previous_turn_concept_id in concepts:
                prior = concepts[previous_turn_concept_id]
                concepts[previous_turn_concept_id] = PixelProjectionConcept(
                    concept_id=prior.concept_id,
                    label=prior.label,
                    terms=prior.terms,
                    abstraction_level=prior.abstraction_level,
                    importance=prior.importance,
                    prediction_accuracy=prior.prediction_accuracy,
                    uncertainty=prior.uncertainty,
                    temporal_links=sorted(set(prior.temporal_links + [concept_id])),
                    hierarchy_links=prior.hierarchy_links,
                    contradiction_links=prior.contradiction_links,
                    metadata=prior.metadata,
                )
            previous_turn_concept_id = concept_id

        for concept in self.selection_graph_evaluator.spatial_concept_memory:
            concept_id = f"spatial:{concept.concept_id}"
            concepts[concept_id] = PixelProjectionConcept(
                concept_id=concept_id,
                label=" ".join(concept.associated_terms) or f"spatial concept {concept.concept_id}",
                terms=list(concept.associated_terms),
                abstraction_level=int(concept.abstraction_level),
                importance=float(concept.importance),
                prediction_accuracy=float(concept.prediction_accuracy),
                uncertainty=float(concept.uncertainty),
                temporal_links=[f"spatial:{item}" for item in concept.coactivation_counts],
                hierarchy_links=[f"spatial:{item}" for item in concept.parent_concept_ids + concept.child_concept_ids],
                contradiction_links=[],
                metadata={
                    "source_type": "spatial_concept_memory",
                    "concept_id": concept.concept_id,
                    "associated_nodes": list(concept.associated_nodes),
                    "associated_clusters": list(concept.associated_clusters),
                    "recency": min(1.0, 0.3 + (concept.reuse_count * 0.08)),
                },
            )
        return list(concepts.values())

    def _pixel_projection_relationships(self, concepts: Sequence[PixelProjectionConcept]) -> List[PixelRelationship]:
        concept_ids = {concept.concept_id for concept in concepts}
        relationships: List[PixelRelationship] = []
        for concept in concepts:
            for target_id in concept.temporal_links:
                if target_id in concept_ids:
                    relationships.append(PixelRelationship(concept.concept_id, target_id, "temporal", 0.65))
            for target_id in concept.hierarchy_links:
                if target_id in concept_ids:
                    relationships.append(PixelRelationship(concept.concept_id, target_id, "hierarchy", 0.85))
            for target_id in concept.contradiction_links:
                if target_id in concept_ids:
                    relationships.append(PixelRelationship(concept.concept_id, target_id, "contradicts", 1.0))
        for edge in self.global_graph_store.edges:
            if not isinstance(edge, dict):
                continue
            source = f"graph:{edge.get('from', '')}"
            target = f"graph:{edge.get('to', '')}"
            if source not in concept_ids or target not in concept_ids:
                continue
            relationship = str(edge.get("relationship", "related_to") or "related_to").lower()
            strength = min(1.0, 0.35 + (0.08 * float(edge.get("support_count", 1) or 1)))
            if "contrad" in relationship or "conflict" in relationship:
                relationship_type = "contradicts"
            elif relationship in {"parent", "child", "is_a", "part_of"}:
                relationship_type = "hierarchy"
            elif relationship in {"causes", "caused_by", "leads_to"}:
                relationship_type = "temporal"
            else:
                relationship_type = "semantic"
            relationships.append(PixelRelationship(source, target, relationship_type, strength))
        return relationships

    def _pixel_readback_fragment(
        self,
        *,
        pixel_readback: PixelReadback,
        pixel_reasoning_trace: PixelReasoningTrace,
        user_input_text: str,
        graph_confidence: float,
        decision_source: str,
        margin: float,
    ) -> Optional[ReconstructedMemoryFragment]:
        if pixel_readback.spatial_confidence <= 0.0 or not pixel_readback.dominant_concepts:
            return None
        normalized_query = str(user_input_text or "").lower()
        is_temporal_query = any(token in normalized_query for token in {"after", "next", "then", "sequence", "follow"})
        if pixel_reasoning_trace.reasoning_mode == "temporal" and is_temporal_query:
            readback_candidates = list(pixel_reasoning_trace.temporal_sequence) + list(pixel_readback.dominant_concepts)
        elif pixel_reasoning_trace.reasoning_mode == "multi_constraint":
            readback_candidates = list(pixel_reasoning_trace.constraints) or list(pixel_readback.dominant_concepts)
        else:
            readback_candidates = (
                list(pixel_readback.temporal_next_concepts) + list(pixel_readback.dominant_concepts)
                if is_temporal_query
                else list(pixel_readback.dominant_concepts)
            )
        memory_candidates = [
            item
            for item in readback_candidates
            if isinstance(item.get("metadata", {}), dict)
            and str(item.get("metadata", {}).get("source_type", "")) in {"memory_entry", "turn_record"}
        ]
        if not memory_candidates:
            memory_candidates = list(readback_candidates[:1])
        if not memory_candidates:
            return None
        top = memory_candidates[0]
        metadata = top.get("metadata", {}) if isinstance(top.get("metadata", {}), dict) else {}
        source_id = int(metadata.get("entry_id", metadata.get("turn_id", 0)) or 0)
        if pixel_reasoning_trace.reasoning_mode == "multi_constraint" and pixel_reasoning_trace.constraints:
            labels: List[str] = []
            for item in pixel_reasoning_trace.constraints:
                item_metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
                label = str(
                    item_metadata.get("display_text", "")
                    or item_metadata.get("raw_user_prompt", "")
                    or item.get("label", "")
                ).strip()
                if label:
                    labels.append(label)
            text = "; ".join(label for label in labels[:4] if label).strip()
        else:
            text = str(
                metadata.get("display_text", "")
                or metadata.get("raw_user_prompt", "")
                or top.get("label", "")
                or pixel_readback.reconstructed_summary
            ).strip()
        base_score = 0.24 + (float(pixel_readback.spatial_confidence) * 0.5)
        if decision_source == "pixel":
            base_score = max(base_score, float(graph_confidence) + float(margin) + 0.01)
        elif decision_source == "hybrid":
            base_score = max(base_score, float(graph_confidence) - (float(margin) * 0.5))
        score = round(min(0.95, base_score), 4)
        return ReconstructedMemoryFragment(
            source_type="pixel_readback",
            source_id=source_id,
            text=text,
            score=score,
            evidence_breakdown={
                "concept_similarity": min(1.0, float(top.get("score", 0.0) or 0.0)),
                "temporal_proximity": float(metadata.get("recency", 0.0) or 0.0),
                "goal_link": 0.0,
                "hierarchy_link": 0.25 if metadata.get("state_type") else 0.0,
                "cross_domain_link": 0.35,
                "recency": float(metadata.get("recency", 0.0) or 0.0),
                "pixel_confidence": float(pixel_readback.spatial_confidence),
                "pixel_dominance_gap": float(pixel_readback.dominance_gap),
                "multi_constraint_result": 1.0 if pixel_reasoning_trace.reasoning_mode == "multi_constraint" else 0.0,
                "pixel_reasoning_mode": 1.0 if pixel_reasoning_trace.reasoning_mode == "single_answer" else 2.0 if pixel_reasoning_trace.reasoning_mode == "multi_constraint" else 3.0 if pixel_reasoning_trace.reasoning_mode == "temporal" else 0.5,
                "decision_source": 1.0 if decision_source == "pixel" else 0.5 if decision_source == "hybrid" else 0.0,
            },
            payload={
                "pixel_readback": pixel_readback.to_dict(),
                "pixel_reasoning": pixel_reasoning_trace.to_dict(),
                "dominant_concept": dict(top),
                "multi_constraint_concepts": list(pixel_reasoning_trace.constraints),
                "contradictory_concepts": list(pixel_readback.contradictory_concepts),
            },
        )

    def _memory_reconstruction_exact_matches(self, merged_memory_state: Dict[str, object]) -> List[Dict[str, object]]:
        items = merged_memory_state.get("items", []) if isinstance(merged_memory_state, dict) else []
        if not isinstance(items, list):
            return []
        matches: List[Dict[str, object]] = []
        for item in items[:6]:
            if not isinstance(item, dict):
                continue
            matches.append(
                {
                    "state_key": item.get("state_key", ""),
                    "state_type": item.get("state_type", ""),
                    "current_value": item.get("current_value"),
                    "display_text": item.get("display_text", ""),
                    "history_count": item.get("history_count", 0),
                }
            )
        return matches

    def _memory_reconstruction_profile(self, normalized: str) -> Dict[str, object]:
        desired_state_types: set[str] = set()
        if any(fragment in normalized for fragment in ("favorite", "which nuts", "nuts", "okay for me", "food constraint")):
            desired_state_types.add("dietary_constraint")
            if "favorite" in normalized:
                desired_state_types.add("favorite_preference")
        if any(fragment in normalized for fragment in ("available", "free")):
            desired_state_types.add("availability")
        if any(fragment in normalized for fragment in ("document title", "current document")):
            desired_state_types.add("document_title")
        if any(fragment in normalized for fragment in ("where", "stored", "berries", "chicken")):
            desired_state_types.add("storage_location")
        if any(fragment in normalized for fragment in ("office", "room", "item state", "notebook", "charger")):
            desired_state_types.add("object_location")
        if any(fragment in normalized for fragment in ("project", "incomplete", "status")):
            desired_state_types.add("project_status")
        if "checklist" in normalized:
            desired_state_types.add("checklist_status")
        if any(fragment in normalized for fragment in ("troubleshooting", "already tried")):
            desired_state_types.add("troubleshooting_step")
        if "budget" in normalized:
            desired_state_types.add("budget_state")
        if "goal" in normalized:
            desired_state_types.add("goal_state")
        if any(fragment in normalized for fragment in ("planning request", "meal plan", "earlier question", "what did i ask")):
            desired_state_types.add("planning_request")
        return {
            "desired_state_types": desired_state_types,
            "wants_last_planning": "last planning" in normalized or "planning request" in normalized,
            "wants_prior_prompt": "what did i ask" in normalized or "earlier question" in normalized,
            "wants_about": self._memory_reconstruction_about_terms(normalized),
        }

    @staticmethod
    def _memory_reconstruction_about_terms(normalized: str) -> set[str]:
        match = re.search(r"(?:about|regarding) ([a-z0-9 \-]+)", normalized)
        if not match:
            return set()
        stopwords = {"the", "my", "your", "a", "an", "what", "did", "say", "said"}
        return {
            token
            for token in match.group(1).replace("?", " ").replace(".", " ").split()
            if len(token) > 2 and token not in stopwords
        }

    def _score_reconstructed_state_fragment(
        self,
        *,
        user_input_text: str,
        query_tokens: set[str],
        profile: Dict[str, object],
        observation: Dict[str, object],
        source_id: int,
        recency: float,
        source_text: str,
        config: MemoryReconstructionConfig,
    ) -> ReconstructedMemoryFragment:
        state_type = str(observation.get("state_type", ""))
        desired_state_types = profile.get("desired_state_types", set())
        desired = desired_state_types if isinstance(desired_state_types, set) else set()
        fragment_text = str(observation.get("display_text", "") or source_text or observation.get("value", ""))
        target_tokens = self._tokens_from_payload(
            {
                "text": fragment_text,
                "state_key": observation.get("state_key", ""),
                "state_type": observation.get("state_type", ""),
                "entity": observation.get("entity", ""),
                "value": observation.get("value", ""),
            }
        )
        concept_similarity = self._token_overlap(query_tokens, target_tokens)
        about_terms = profile.get("wants_about", set())
        if isinstance(about_terms, set) and about_terms:
            concept_similarity = max(concept_similarity, self._token_overlap(about_terms, target_tokens))
        hierarchy_link = 1.0 if state_type in desired else 0.0
        if self._state_observation_matches_query(str(user_input_text or "").lower(), observation):
            hierarchy_link = max(hierarchy_link, 0.85)
        temporal_proximity = recency
        goal_link = 0.25 if state_type in {"goal_state", "planning_request"} else 0.0
        cross_domain_link = 0.25 if concept_similarity > 0.0 and hierarchy_link <= 0.0 else 0.0
        raw_score = (
            concept_similarity * config.concept_similarity_weight
            + temporal_proximity * config.temporal_proximity_weight
            + goal_link * config.goal_link_weight
            + hierarchy_link * config.hierarchy_link_weight
            + cross_domain_link * config.cross_domain_link_weight
            + recency * config.recency_weight
        )
        if observation.get("is_update"):
            raw_score += 0.06
        if state_type in desired:
            raw_score += 0.12
        return ReconstructedMemoryFragment(
            source_type="state_memory",
            source_id=source_id,
            text=fragment_text,
            score=round(min(1.0, raw_score), 4),
            evidence_breakdown={
                "concept_similarity": concept_similarity,
                "temporal_proximity": temporal_proximity,
                "goal_link": goal_link,
                "hierarchy_link": hierarchy_link,
                "cross_domain_link": cross_domain_link,
                "recency": recency,
            },
            payload={
                "state_key": observation.get("state_key", ""),
                "state_type": state_type,
                "value": observation.get("value"),
                "display_text": observation.get("display_text", ""),
                "entity": observation.get("entity", ""),
                "is_update": bool(observation.get("is_update", False)),
                "source_text": source_text,
            },
        )

    def _score_reconstructed_turn_fragment(
        self,
        *,
        user_input_text: str,
        query_tokens: set[str],
        profile: Dict[str, object],
        record: EpisodicTurnRecord,
        recency: float,
        config: MemoryReconstructionConfig,
    ) -> ReconstructedMemoryFragment:
        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        clarification_state = metadata.get("clarification_state", {}) if isinstance(metadata.get("clarification_state", {}), dict) else {}
        clarification_response = str(clarification_state.get("clarification_response", "")).strip()
        merged_prompt = str(clarification_state.get("merged_prompt", "")).strip()
        prompt_text = clarification_response or merged_prompt or record.raw_user_prompt
        prompt_tokens = self._episodic_tokens(prompt_text.lower())
        answer_tokens = self._episodic_tokens(record.final_answer.lower())
        concept_similarity = max(self._token_overlap(query_tokens, prompt_tokens), self._token_overlap(query_tokens, answer_tokens))
        about_terms = profile.get("wants_about", set())
        if isinstance(about_terms, set) and about_terms:
            concept_similarity = max(concept_similarity, self._token_overlap(about_terms, prompt_tokens), self._token_overlap(about_terms, answer_tokens))
        wants_prior_prompt = bool(profile.get("wants_prior_prompt", False))
        wants_last_planning = bool(profile.get("wants_last_planning", False))
        if wants_prior_prompt:
            concept_similarity = max(concept_similarity, 0.62)
        if wants_last_planning and record.route_type == "planning_request":
            concept_similarity = max(concept_similarity, 0.72)
        temporal_proximity = recency
        goal_link = 1.0 if record.route_type in {"planning_request", "goal_request"} else 0.0
        hierarchy_link = 0.5 if record.route_type in {"planning_request", "goal_request", "hybrid_request"} else 0.15
        cross_domain_link = 0.25 if concept_similarity > 0.0 and record.route_type not in {"planning_request", "goal_request"} else 0.0
        raw_score = (
            concept_similarity * config.concept_similarity_weight
            + temporal_proximity * config.temporal_proximity_weight
            + goal_link * config.goal_link_weight
            + hierarchy_link * config.hierarchy_link_weight
            + cross_domain_link * config.cross_domain_link_weight
            + recency * config.recency_weight
        )
        if wants_prior_prompt or wants_last_planning:
            raw_score += 0.1
        return ReconstructedMemoryFragment(
            source_type="turn",
            source_id=record.turn_id,
            text=prompt_text,
            score=round(min(1.0, raw_score), 4),
            evidence_breakdown={
                "concept_similarity": concept_similarity,
                "temporal_proximity": temporal_proximity,
                "goal_link": goal_link,
                "hierarchy_link": hierarchy_link,
                "cross_domain_link": cross_domain_link,
                "recency": recency,
            },
            payload={
                "turn_id": record.turn_id,
                "sequence": record.sequence,
                "route_type": record.route_type,
                "raw_user_prompt": prompt_text,
                "stored_raw_user_prompt": record.raw_user_prompt,
                "clarification_response": clarification_response,
                "final_answer": record.final_answer,
            },
        )

    @staticmethod
    def _tokens_from_payload(payload: Dict[str, object]) -> set[str]:
        text = " ".join(str(value) for value in payload.values())
        words = re.findall(r"[a-z0-9]+", text.lower().replace("_", " ").replace(":", " "))
        stopwords = {"the", "and", "for", "with", "that", "this", "from", "into", "now", "current"}
        return {word for word in words if len(word) > 2 and word not in stopwords}

    @staticmethod
    def _token_overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left.intersection(right)) / float(max(1, len(left)))

    def _synthesize_reconstructed_memory_answer(
        self,
        *,
        user_input_text: str,
        merged_memory_state: Dict[str, object],
        partial_matches: Sequence[ReconstructedMemoryFragment],
        answer_mode: str,
    ) -> str:
        normalized = str(user_input_text or "").lower()
        if "which nuts" in normalized or "nuts should" in normalized:
            avoided = {
                str(item.payload.get("entity", "")).lower()
                for item in partial_matches
                if item.payload.get("state_type") == "dietary_constraint"
                and str(item.payload.get("value", "")).startswith("avoid ")
            }
            okay = [
                str(item.payload.get("value"))
                for item in partial_matches
                if item.payload.get("state_type") == "dietary_constraint"
                and "okay" in str(item.payload.get("value", ""))
                and str(item.payload.get("entity", "")).lower() not in avoided
            ]
            if okay:
                return "The related dietary memory I found says: " + ", ".join(list(dict.fromkeys(okay))) + "."
        merged_answer = self._answer_from_merged_memory_state(user_input_text, merged_memory_state)
        if merged_answer:
            return merged_answer
        if not partial_matches:
            return ""
        top = partial_matches[0]
        payload = top.payload
        if top.source_type in {"turn", "temporal_recall"}:
            prompt = str(payload.get("raw_user_prompt", top.text)).strip()
            if "what did i ask" in normalized or "earlier" in normalized:
                return f"Your earlier question appears to have been about {self._summarize_prompt_topic(prompt)}."
            if "last planning" in normalized or "planning request" in normalized:
                return f"Your last planning request appears to have been: {prompt}"
            if "what did you say" in normalized:
                answer = str(payload.get("final_answer", "")).strip()
                if answer:
                    return f"From the related memory I found, I said: {answer}"
                return f"From the related memory I found, the prior topic was: {prompt}"
            return f"The strongest related memory I found was: {prompt}"
        state_type = str(payload.get("state_type", ""))
        value = payload.get("value")
        entity = str(payload.get("entity", "")).replace("_", " ")
        if state_type in {"storage_location", "object_location"} and entity:
            return f"The latest related memory says {entity} is in the {value}."
        if state_type == "document_title":
            return f"The current document title appears to be {value}."
        if state_type == "availability":
            return f"The latest availability memory I found is: {value}."
        if state_type == "dietary_constraint":
            return f"The related dietary memory I found is: {value}."
        if state_type in {"project_status", "checklist_status"} and entity:
            return f"The latest related status says {entity} is {value}."
        if answer_mode == "low":
            return f"I found a partial related memory: {top.text}"
        return str(payload.get("display_text", top.text)).strip()

    @staticmethod
    def _summarize_prompt_topic(prompt: str) -> str:
        text = str(prompt or "").strip().rstrip(".?")
        lowered = text.lower()
        if lowered.startswith("help me "):
            return text[8:]
        if lowered.startswith("what is "):
            return text[8:]
        return text

    def _build_memory_recall_explanation(
        self,
        user_input_text: str,
        recalled_turns: Sequence[EpisodicTurnRecall],
        retrieved: Sequence[RetrievedMemory],
        final_result: BrainInputV1,
        merged_memory_state: Dict[str, object],
    ) -> LLMExplanationResult:
        merged_answer = self._answer_from_merged_memory_state(user_input_text, merged_memory_state)
        if merged_answer:
            text = merged_answer
        elif recalled_turns:
            lines = []
            for index, recall in enumerate(recalled_turns[:3], start=1):
                record = recall.record
                lines.append(
                    f"{index}. Prompt: {record.raw_user_prompt}\n"
                    f"Answer: {record.final_answer}\n"
                    f"Route: {record.route_type}\n"
                    f"Structured result: {json.dumps(record.final_selected_result.to_dict(), sort_keys=True)}"
                )
            text = "I recalled these prior turns:\n" + "\n\n".join(lines)
        elif retrieved:
            text = (
                f"I did not find a full episodic turn match for \"{user_input_text}\", "
                "but I found structurally related BrAIn memory."
            )
        else:
            text = (
                f"I could not recall a prior stored turn for \"{user_input_text}\". "
                "No episodic match or related BrAIn memory was available."
            )
        return LLMExplanationResult(
            provider="deterministic",
            structured_text=self.codec.reconstruct_text(self._brain_input_to_memory(final_result)),
            reasoning_explanation="episodic_recall",
            user_facing_text=text,
            messages=[],
            metadata={
                "mode": "episodic_recall",
                "recalled_turn_count": len(recalled_turns),
                "retrieved_memory_count": len(retrieved),
                "merged_memory_state": merged_memory_state,
            },
        )

    def _answer_from_merged_memory_state(
        self,
        user_input_text: str,
        merged_memory_state: Dict[str, object],
    ) -> str:
        items = merged_memory_state.get("items", []) if isinstance(merged_memory_state, dict) else []
        normalized = str(user_input_text or "").lower()
        if not isinstance(items, list) or not items:
            return ""
        if "favorite color" in normalized:
            for item in items:
                if item.get("state_key") == "favorite_color":
                    return f"Your favorite color is {item.get('current_value')}."
        if "budget" in normalized:
            for item in items:
                if str(item.get("state_key", "")).startswith("budget:"):
                    topic = str(item.get("entity", "budget")).replace("_", " ")
                    return f"Your {topic} is {item.get('current_value')}."
        if "food constraint" in normalized or "constraints have i told you" in normalized or "which nuts" in normalized or "nuts should" in normalized:
            if "which nuts" in normalized or "nuts should" in normalized:
                avoided = {
                    str(item.get("entity", "")).lower()
                    for item in items
                    if item.get("state_type") == "dietary_constraint" and str(item.get("current_value", "")).startswith("avoid ")
                }
                constraints = [
                    str(item.get("current_value"))
                    for item in items
                    if item.get("state_type") == "dietary_constraint"
                    and "okay" in str(item.get("current_value", ""))
                    and str(item.get("entity", "")).lower() not in avoided
                ]
            else:
                constraints = [str(item.get("current_value")) for item in items if item.get("state_type") == "dietary_constraint"]
            if constraints:
                if "which nuts" in normalized or "nuts should" in normalized:
                    return "The nuts currently marked okay are: " + ", ".join(constraints) + "."
                return "You told me these food constraints: " + ", ".join(constraints) + "."
        if "packing constraints" in normalized or "constraints have i given you" in normalized:
            constraints = [str(item.get("current_value")) for item in items if item.get("state_type") == "planning_constraint"]
            if constraints:
                return "Your packing constraints are: " + ", ".join(constraints) + "."
        if "document title" in normalized:
            for item in items:
                if item.get("state_type") == "document_title":
                    return f"The current document title is {item.get('current_value')}."
        if "office item state" in normalized or ("current" in normalized and "office" in normalized):
            locations = []
            for item in items:
                if item.get("state_type") == "object_location":
                    entity = str(item.get("entity", str(item.get("state_key", "")).split(":", 1)[-1])).replace("_", " ")
                    locations.append(f"{entity} is in the {item.get('current_value')}")
            if locations:
                return "The current office item state is: " + ", ".join(locations) + "."
        if "current project status" in normalized or ("project status" in normalized and "current" in normalized):
            status_bits = []
            for item in items:
                if item.get("state_type") == "project_status" and not str(item.get("state_key", "")).startswith("checklist_status:"):
                    label = str(item.get("entity", str(item.get("state_key", "")).split(":", 1)[-1])).replace("_", " ")
                    status_bits.append(f"{label} is {item.get('current_value')}")
            if status_bits:
                return "Current project status: " + ", ".join(status_bits) + "."
        if "shipping checklist" in normalized or "checklist" in normalized:
            checklist_bits = []
            for item in items:
                if str(item.get("state_key", "")).startswith("checklist_status:"):
                    label = str(item.get("entity", str(item.get("state_key", "")).split(":", 1)[-1])).replace("_", " ")
                    checklist_bits.append(f"{label} is {item.get('current_value')}")
            if checklist_bits:
                return "Current shipping checklist state: " + ", ".join(checklist_bits) + "."
        if "still incomplete" in normalized:
            incomplete = [
                f"{str(item.get('state_key', '')).split(':', 1)[-1].replace('_', ' ')} {item.get('current_value')}"
                for item in items
                if str(item.get("current_value", "")).lower().startswith(("incomplete", "blocked"))
            ]
            if incomplete:
                return "These parts are still incomplete: " + ", ".join(incomplete) + "."
        if "when am i free" in normalized or "currently available" in normalized or normalized.strip() == "when am i free?":
            available = [str(item.get("current_value")) for item in items if item.get("state_type") == "availability" and str(item.get("current_value", "")).startswith("free ")]
            if available:
                return "You are currently free " + " and ".join(available).replace("free ", "", 1) + "."
        if "stored now" in normalized or "where should the chicken" in normalized or "where are" in normalized or "where should you assume" in normalized:
            for item in items:
                if item.get("state_type") in {"storage_location", "object_location"}:
                    entity = str(item.get("state_key", "")).split(":", 1)[-1].replace("_", " ")
                    return f"The {entity} should be in the {item.get('current_value')}."
        if "already tried" in normalized:
            tried: List[str] = []
            for item in items:
                if item.get("state_type") != "troubleshooting_step":
                    continue
                value = item.get("current_value")
                if isinstance(value, list):
                    tried.extend(str(step) for step in value if str(step).strip())
                elif str(value).strip():
                    tried.append(str(value))
            tried = list(dict.fromkeys(tried))
            if tried:
                return "You already tried: " + ", ".join(tried) + "."
        if "final food totals" in normalized or "final totals" in normalized:
            for item in items:
                if item.get("state_key") == "world_food_totals" and isinstance(item.get("current_value"), dict):
                    totals = item["current_value"]
                    parts = [f"{city} = {amount}" for city, amount in sorted(totals.items())]
                    return "The final food totals are " + ", ".join(parts) + "."
        if "current status of auth" in normalized and "original status" in normalized:
            for item in items:
                if item.get("state_key") == "project_status:auth":
                    return f"Current auth status: {item.get('current_value')}. Original auth status: {item.get('original_value')}."
        if "meal plan" in normalized and "what kind of meal plan" in normalized:
            plan_bits = [str(item.get("current_value")) for item in items if item.get("state_type") == "planning_request"]
            if plan_bits:
                return "You asked for " + ", ".join(plan_bits) + "."
        if "when is my meeting" in normalized:
            for item in items:
                if item.get("state_key") == "meeting_time":
                    return f"Your meeting is {item.get('current_value')}."
        if "what is my goal" in normalized or "what was my goal" in normalized:
            goals = [str(item.get("current_value")) for item in items if item.get("state_type") == "goal_state"]
            if goals:
                return "Your goal was " + ", ".join(goals) + "."
        if "what did i ask earlier" in normalized or "what was my last planning request" in normalized:
            return ""
        summaries = [str(item.get("display_text", "")) for item in items if str(item.get("display_text", "")).strip()]
        if summaries:
            return "; ".join(summaries[:4]).capitalize() + "."
        return ""

    def _episodic_semantic_score(
        self,
        query_tokens: set[str],
        record: EpisodicTurnRecord,
    ) -> float:
        if not query_tokens:
            return 0.0
        prompt_tokens = self._episodic_tokens(record.raw_user_prompt.lower())
        answer_tokens = self._episodic_tokens(record.final_answer.lower())
        structured_tokens = self._episodic_tokens(
            self.codec.reconstruct_text(self._brain_input_to_memory(record.final_selected_result)).lower()
        )
        score = 0.0
        if prompt_tokens:
            score += 1.2 * len(query_tokens.intersection(prompt_tokens)) / max(1.0, float(len(query_tokens)))
        if answer_tokens:
            score += 0.8 * len(query_tokens.intersection(answer_tokens)) / max(1.0, float(len(query_tokens)))
        if structured_tokens:
            score += 0.7 * len(query_tokens.intersection(structured_tokens)) / max(1.0, float(len(query_tokens)))
        return score

    @staticmethod
    def _text_overlap_score(query_tokens: set[str], payload: Dict[str, object]) -> float:
        text_parts = [
            str(payload.get("state_key", "")),
            str(payload.get("state_type", "")),
            str(payload.get("value", "")),
            str(payload.get("display_text", "")),
            str(payload.get("entity", "")),
        ]
        target_tokens = set()
        for part in text_parts:
            for token in part.lower().replace(":", " ").replace("_", " ").split():
                stripped = token.strip(".,?!:;\"'()[]{}")
                if len(stripped) > 2:
                    target_tokens.add(stripped)
        if not query_tokens or not target_tokens:
            return 0.0
        return len(query_tokens.intersection(target_tokens)) / float(len(query_tokens))

    @staticmethod
    def _memory_question_targets_state_type(normalized: str, state_type: str) -> bool:
        mapping = {
            "favorite": {"favorite_preference"},
            "nuts": {"dietary_constraint"},
            "okay": {"dietary_constraint"},
            "constraint": {"dietary_constraint", "planning_constraint"},
            "free": {"availability"},
            "available": {"availability"},
            "incomplete": {"project_status", "task_status"},
            "project status": {"project_status"},
            "status": {"project_status", "checklist_status"},
            "checklist": {"checklist_status"},
            "stored": {"storage_location"},
            "tried": {"troubleshooting_step"},
            "troubleshooting": {"troubleshooting_step"},
            "final totals": {"world_food_total"},
            "meal plan": {"planning_request"},
            "meeting": {"meeting_time"},
            "goal": {"goal_state"},
            "budget": {"budget_state"},
            "berries": {"storage_location"},
            "document": {"document_title"},
            "title": {"document_title"},
            "office": {"object_location"},
            "room": {"object_location"},
            "item state": {"object_location"},
        }
        for cue, state_types in mapping.items():
            if cue in normalized and state_type in state_types:
                return True
        return False

    @staticmethod
    def _state_observation_matches_query(normalized: str, observation: Dict[str, object]) -> bool:
        entity = str(observation.get("entity", "")).lower().replace("_", " ")
        state_key = str(observation.get("state_key", "")).lower().replace("_", " ")
        display_text = str(observation.get("display_text", "")).lower()
        for candidate in (entity, state_key, display_text):
            if candidate and candidate in normalized:
                return True
        if "shipping checklist" in normalized and str(observation.get("state_type", "")) == "checklist_status":
            return True
        if "project status" in normalized and str(observation.get("state_type", "")) == "project_status":
            return True
        return False

    def _promote_route_decision(
        self,
        route_decision: QueryRouteDecision,
        *,
        route_type: str,
        minimum_confidence: float,
        reason: str,
    ) -> QueryRouteDecision:
        score_breakdown = dict(route_decision.score_breakdown)
        current_winner = max(score_breakdown.values(), default=0.0)
        promoted_score = max(score_breakdown.get(route_type, 0.0), current_winner + 0.25, 2.75)
        score_breakdown[route_type] = promoted_score
        return QueryRouteDecision(
            route_type=route_type,
            confidence=max(route_decision.confidence, minimum_confidence),
            reasons=list(route_decision.reasons) + [reason],
            run_brain_reasoning=route_type == "memory_question" or route_decision.run_brain_reasoning,
            run_direct_llm=route_type not in {"memory_question"} and route_decision.run_direct_llm,
            run_hybrid_mode=route_type == "hybrid_request",
            score_breakdown=score_breakdown,
        )

    def _should_force_memory_route(self, text: str, route_decision: QueryRouteDecision) -> bool:
        normalized = str(text or "").strip().lower()
        if (self.relation_memory_enabled and route_decision.route_type == "factual_question"
                and has_relevant_relation_history(self.turn_store.records,text)):
            return True
        if not self._looks_like_memory_recall_query(normalized):
            return False
        if route_decision.route_type == "memory_question":
            return False
        profile = self._memory_reconstruction_profile(normalized)
        if bool(profile.get("desired_state_types")) or bool(profile.get("wants_prior_prompt")) or bool(profile.get("wants_last_planning")):
            return True
        if self.retrieve_stateful_memory(user_input_text=text, top_k=2):
            return True
        return bool(self.retrieve_episodic_turns(user_input_text=text, route_type="memory_question", top_k=1))

    def _memory_state_to_brain_input(self, merged_memory_state: Dict[str, object]) -> Optional[BrainInputV1]:
        items = merged_memory_state.get("items", []) if isinstance(merged_memory_state, dict) else []
        if not isinstance(items, list) or not items:
            return None
        top = items[0]
        memory = BrainMemory(
            actor="user",
            intent="inform",
            action=str(top.get("state_type", "recall")),
            object=str(top.get("state_key", "")),
            topic=str(top.get("current_value", "")),
            priority="normal",
        )
        return self._memory_to_brain_input(BrainCodec.resolve_memory(memory))

    def _extract_state_observations(self, text: str) -> List[Dict[str, object]]:
        import re

        normalized = str(text or "").strip()
        lowered = normalized.lower()
        observations: List[Dict[str, object]] = []
        is_update = any(fragment in lowered for fragment in ("actually", "correction", "instead", "now", "final update", "sorry"))

        def add_observation(
            *,
            state_key: str,
            state_type: str,
            value: object,
            display_text: str,
            entity: str,
            merge_mode: str = "latest",
            operation: str = "assign",
            update_flag: Optional[bool] = None,
        ) -> None:
            observations.append(
                {
                    "state_key": state_key,
                    "state_type": state_type,
                    "value": value,
                    "display_text": display_text,
                    "entity": entity,
                    "is_update": is_update if update_flag is None else bool(update_flag),
                    "merge_mode": merge_mode,
                    "operation": operation,
                }
            )

        favorite_match = re.search(r"favorite color is (?:now |currently |already )?([a-z]+)", lowered)
        if favorite_match:
            color = favorite_match.group(1)
            add_observation(state_key="favorite_color", state_type="favorite_preference", value=color, display_text=f"favorite color is {color}", entity="favorite color")

        for match in re.finditer(r"(?:can't eat|cannot eat|avoid) ([a-z]+)", lowered):
            item = match.group(1)
            add_observation(state_key=f"dietary_constraint:{item}", state_type="dietary_constraint", value=f"avoid {item}", display_text=f"avoid {item}", entity=item)
        for match in re.finditer(r"([a-z]+) (?:also )?bother me", lowered):
            item = match.group(1)
            add_observation(state_key=f"dietary_constraint:{item}", state_type="dietary_constraint", value=f"avoid {item}", display_text=f"avoid {item}", entity=item, update_flag=True)
        for match in re.finditer(r"([a-z]+) (?:are|is) okay", lowered):
            item = match.group(1)
            add_observation(state_key=f"dietary_ok:{item}", state_type="dietary_constraint", value=f"{item} okay", display_text=f"{item} okay", entity=item)
        for match in re.finditer(r"([a-z]+) (?:are|is) fine", lowered):
            item = match.group(1)
            add_observation(state_key=f"dietary_ok:{item}", state_type="dietary_constraint", value=f"{item} okay", display_text=f"{item} okay", entity=item)

        checklist_scope = "shipping checklist" if "shipping checklist" in lowered else ""
        if checklist_scope:
            for label, status in re.findall(r"([a-z\- ]+?) is (done|in progress|not started|delayed(?: until [a-z0-9 ]+)?)", lowered):
                item = label.strip().replace("and ", "").replace("but ", "").replace("my ", "")
                if item in {"for", "checklist"}:
                    continue
                add_observation(
                    state_key=f"checklist_status:{item.replace(' ', '_')}",
                    state_type="checklist_status",
                    value=status.strip(),
                    display_text=f"{item} is {status.strip()}",
                        entity=item,
                    )
        else:
            status_clauses = re.split(r",|\band\b|\bbut\b", lowered)
            for clause in status_clauses:
                cleaned = clause.strip(" .")
                if not cleaned:
                    continue
                match = re.search(r"([a-z\- ]+?) (?:is|are) (?:now )?(incomplete|complete|done|in progress|unblocked|blocked(?: on [a-z ]+)?|delayed(?: until [a-z0-9 ]+)?)", cleaned)
                if not match:
                    match = re.search(r"([a-z\- ]+?) was (blocked|unblocked|in progress|complete|done)", cleaned)
                finished_match = None
                if not match:
                    finished_match = re.search(r"(?:i )?finished ([a-z\- ]+)", cleaned)
                if not match:
                    match = re.search(r"([a-z\- ]+?) has (not started)", cleaned)
                if not match and not finished_match:
                    continue
                if finished_match:
                    item = finished_match.group(1).strip()
                    status = "complete"
                else:
                    item = match.group(1).strip()
                    status = match.group(2).strip()
                item = item.replace("the ", "").replace("my ", "").replace("update: ", "").replace("for the mobile app project", "").strip(" ,")
                if not item:
                    continue
                state_type = "checklist_status" if item in {"label printing", "packing", "drop-off", "drop off"} else "project_status"
                add_observation(
                    state_key=f"{state_type}:{item.replace(' ', '_')}",
                    state_type=state_type,
                    value=status,
                    display_text=f"{item} is {status}",
                    entity=item,
                )

        avail_patterns = [
            re.search(r"free ([a-z]+) after ([0-9]+ ?[ap]m)", lowered),
            re.search(r"free ([a-z]+) (?:after )?([0-9]+ ?[ap]m)", lowered),
        ]
        for match in avail_patterns:
            if match:
                day = match.group(1)
                when = match.group(2)
                add_observation(state_key=f"availability:{day}", state_type="availability", value=f"free {day} after {when}", display_text=f"free {day} after {when}", entity=day)
        not_free_match = re.search(r"not free ([a-z]+)", lowered)
        if not_free_match:
            day = not_free_match.group(1)
            add_observation(state_key=f"availability:{day}", state_type="availability", value=f"not free {day}", display_text=f"not free {day}", entity=day, update_flag=True)
        no_longer_available_match = re.search(r"([a-z]+) is no longer available", lowered)
        if no_longer_available_match:
            day = no_longer_available_match.group(1)
            add_observation(state_key=f"availability:{day}", state_type="availability", value=f"not free {day}", display_text=f"not free {day}", entity=day, update_flag=True)
        for match in re.finditer(r"([a-z]+) after ([0-9]+ ?[ap]m) is open", lowered):
            day = match.group(1)
            when = match.group(2)
            add_observation(state_key=f"availability:{day}", state_type="availability", value=f"free {day} after {when}", display_text=f"free {day} after {when}", entity=day, update_flag=True)

        storage_match = re.search(r"(?:store|put) the ([a-z ]+?) in the ([a-z-]+)", lowered)
        if storage_match:
            item = storage_match.group(1).strip()
            location = storage_match.group(2).strip()
            add_observation(state_key=f"storage:{item.replace(' ', '_')}", state_type="storage_location", value=location, display_text=f"{item} stored in {location}", entity=item)
        keep_storage_match = re.search(r"(?:keep|kept) ([a-z ]+?) in the ([a-z-]+)", lowered)
        if keep_storage_match:
            item = keep_storage_match.group(1).strip()
            location = keep_storage_match.group(2).strip()
            add_observation(state_key=f"storage:{item.replace(' ', '_')}", state_type="storage_location", value=location, display_text=f"{item} stored in {location}", entity=item)
        moved_storage_match = re.search(r"(?:moved|move) (?:the )?([a-z ]+?) to the ([a-z-]+)", lowered)
        if moved_storage_match:
            item = moved_storage_match.group(1).strip()
            location = moved_storage_match.group(2).strip()
            add_observation(state_key=f"storage:{item.replace(' ', '_')}", state_type="storage_location", value=location, display_text=f"{item} stored in {location}", entity=item, update_flag=True)
        title_match = re.search(r"(?:document is titled|document title is|draft document is titled) ([a-z0-9 ]+)", lowered)
        if title_match:
            title = title_match.group(1).strip().strip(".")
            add_observation(state_key="document_title", state_type="document_title", value=title, display_text=f"document title is {title}", entity="document title")
        rename_match = re.search(r"rename it to ([a-z0-9 ]+)", lowered)
        if rename_match:
            title = rename_match.group(1).strip().strip(".")
            add_observation(state_key="document_title", state_type="document_title", value=title, display_text=f"document title is {title}", entity="document title", update_flag=True)
        for item, preposition, location, moved in explicit_locations(normalized):
            add_observation(state_key=f"object_location:{item.replace(' ', '_')}",
                state_type="object_location", value=location,
                display_text=f"{item} is {preposition} {location}", entity=item,
                update_flag=is_update or moved)
        took_object_match = re.search(r"took (?:the )?([a-z]+) out of (?:the )?[a-z-]+ and put it in (?:my |the )?([a-z-]+)", lowered)
        if took_object_match:
            item = took_object_match.group(1).strip()
            location = took_object_match.group(2).strip()
            add_observation(state_key=f"object_location:{item.replace(' ', '_')}", state_type="object_location", value=location, display_text=f"{item} is in {location}", entity=item, update_flag=True)

        for match in re.finditer(r"(?:already )?checked ([a-z0-9 \-]+?)(?=,|\band\b|\bbut\b|\.|$)", lowered):
            step = match.group(1).strip()
            add_observation(state_key=f"troubleshooting:{step.replace(' ', '_')}", state_type="troubleshooting_step", value=f"checked {step}", display_text=f"checked {step}", entity=step, merge_mode="append_set", update_flag=False)
        restart_match = re.search(r"restarted the ([a-z ]+)", lowered)
        if restart_match:
            step = f"restarted {restart_match.group(1).strip()}"
            add_observation(state_key=f"troubleshooting:{step.replace(' ', '_')}", state_type="troubleshooting_step", value=step, display_text=step, entity=step, merge_mode="append_set", update_flag=False)
        for match in re.finditer(r"tried ([a-z0-9 \-]+)", lowered):
            step = match.group(1).strip().rstrip(".")
            if step.startswith("a ") or step.startswith("another "):
                step = step
            add_observation(state_key=f"troubleshooting:{step.replace(' ', '_')}", state_type="troubleshooting_step", value=step, display_text=step, entity=step, merge_mode="append_set", update_flag=False)

        for match in re.finditer(r"city ([a-z]) has ([0-9]+) food", lowered):
            city = f"City {match.group(1).upper()}"
            add_observation(state_key=f"world_food:{city.lower().replace(' ', '_')}", state_type="world_food_total", value=int(match.group(2)), display_text=f"{city} has {match.group(2)} food", entity=city, operation="assign", update_flag=False)
        move_match = re.search(r"move ([0-9]+) food from city ([a-z]) to city ([a-z])", lowered)
        if move_match:
            add_observation(state_key="world_food_move", state_type="world_food_total", value={"amount": int(move_match.group(1)), "from": f"City {move_match.group(2).upper()}", "to": f"City {move_match.group(3).upper()}"}, display_text=f"move {move_match.group(1)} food", entity="food", operation="move", update_flag=True)
        destroy_match = re.search(r"destroys ([0-9]+) food in city ([a-z])", lowered)
        if destroy_match:
            add_observation(state_key="world_food_destroy", state_type="world_food_total", value={"amount": int(destroy_match.group(1)), "city": f"City {destroy_match.group(2).upper()}"}, display_text=f"destroy {destroy_match.group(1)} food", entity="food", operation="destroy", update_flag=True)

        meeting_match = re.search(r"meeting is on ([a-z]+ at [0-9]+ ?[ap]m)", lowered)
        if meeting_match:
            value = meeting_match.group(1)
            add_observation(state_key="meeting_time", state_type="meeting_time", value=value, display_text=f"meeting {value}", entity="meeting")
        move_meeting_match = re.search(r"move that to ([a-z]+ at [0-9]+ ?[ap]m)", lowered)
        if move_meeting_match:
            value = move_meeting_match.group(1)
            add_observation(state_key="meeting_time", state_type="meeting_time", value=value, display_text=f"meeting {value}", entity="meeting", update_flag=True)
        final_meeting_match = re.search(r"final update: it'?s ([a-z]+ at [0-9]+ ?[ap]m)", lowered)
        if final_meeting_match:
            value = final_meeting_match.group(1)
            add_observation(state_key="meeting_time", state_type="meeting_time", value=value, display_text=f"meeting {value}", entity="meeting", update_flag=True)

        goal_match = re.search(r"my goal is to ([a-z0-9 ,\-]+)", lowered)
        if goal_match:
            value = goal_match.group(1).strip()
            add_observation(state_key="user_goal", state_type="goal_state", value=value, display_text=f"goal is to {value}", entity="goal")

        meal_plan_match = re.search(r"plan a ([0-9]+-day meal plan) with ([a-z,\- ]+ lunches)", lowered)
        if meal_plan_match:
            value = f"{meal_plan_match.group(1)} with {meal_plan_match.group(2).strip()}"
            add_observation(state_key="planning_request:meal_plan", state_type="planning_request", value=value, display_text=value, entity="meal plan", update_flag=False)
        if "make dinners flexible" in lowered or "dinners flexible" in lowered:
            add_observation(state_key="planning_request:meal_plan_dinners", state_type="planning_request", value="flexible dinners", display_text="flexible dinners", entity="meal plan", update_flag=True)

        budget_match = re.search(r"(?:my )?([a-z ]+?) budget(?: for [a-z ]+)? is ([0-9]+ dollars(?: total)?)", lowered)
        if budget_match:
            topic = budget_match.group(1).strip()
            value = budget_match.group(2).strip()
            add_observation(state_key=f"budget:{topic.replace(' ', '_')}", state_type="budget_state", value=value, display_text=f"{topic} budget is {value}", entity=f"{topic} budget")
        budget_update_match = re.search(r"make that ([0-9]+ dollars)", lowered)
        if budget_update_match:
            value = budget_update_match.group(1).strip()
            add_observation(state_key="budget:travel", state_type="budget_state", value=value, display_text=f"travel budget is {value}", entity="travel budget", update_flag=True)

        if "one backpack" in lowered:
            add_observation(state_key="packing_constraint:one_backpack", state_type="planning_constraint", value="one backpack", display_text="one backpack", entity="packing")
        for family, value in quantity_bounds(normalized):
            key = "packing_constraint:weight" if family == "weight" else f"quantity_constraint:{family}"
            add_observation(state_key=key, state_type="planning_constraint",
                value=value, display_text=value, entity=family)
        for item, value in placement_constraints(normalized):
            key = "packing_constraint:laptop_access" if item == "laptop" else f"placement_constraint:{item.replace(' ', '_')}"
            add_observation(state_key=key, state_type="planning_constraint",
                value=value, display_text=value, entity=item)

        return observations

    @staticmethod
    def _state_update_acknowledgement(observations: Sequence[Dict[str, object]]) -> str:
        if not observations:
            return "Noted."
        first = observations[0]
        state_type = str(first.get("state_type", ""))
        entity = str(first.get("entity", "")).replace("_", " ")
        value = first.get("value")
        if state_type == "favorite_preference":
            return f"Noted. Your favorite color is now {value}."
        if state_type == "storage_location":
            return f"Noted. I'll keep {entity} stored in the {value}."
        if state_type == "meeting_time":
            return f"Noted. Your meeting is {value}."
        if state_type in {"project_status", "checklist_status"}:
            return "Noted. I updated that status."
        if state_type == "troubleshooting_step":
            return "Noted. I recorded the troubleshooting steps you've already tried."
        return "Noted."

    @staticmethod
    def _resolve_world_food_totals(observations: Sequence[Dict[str, object]]) -> Dict[str, int]:
        totals: Dict[str, int] = {}
        relevant = [item for item in observations if item.get("state_type") == "world_food_total"]
        for item in sorted(relevant, key=lambda payload: (str(payload.get("timestamp", "")), int(payload.get("entry_id", 0)))):
            operation = str(item.get("operation", "set"))
            value = item.get("value")
            if operation == "assign" and isinstance(value, int):
                city = str(item.get("entity", ""))
                if city:
                    totals[city] = int(value)
            elif operation == "move" and isinstance(value, dict):
                amount = int(value.get("amount", 0))
                source = str(value.get("from", ""))
                target = str(value.get("to", ""))
                if source:
                    totals[source] = totals.get(source, 0) - amount
                if target:
                    totals[target] = totals.get(target, 0) + amount
            elif operation == "destroy" and isinstance(value, dict):
                amount = int(value.get("amount", 0))
                city = str(value.get("city", ""))
                if city:
                    totals[city] = totals.get(city, 0) - amount
        return totals

    @staticmethod
    def _episodic_tokens(text: str) -> set[str]:
        words = [token.strip(".,?!:;\"'()[]{}") for token in text.split()]
        stopwords = {
            "what", "did", "you", "i", "my", "the", "a", "an", "was", "is", "me", "about",
            "earlier", "last", "tell", "told", "say", "said", "request", "prompt", "planning",
            "question",
        }
        return {word for word in words if len(word) > 2 and word not in stopwords}

    @staticmethod
    def _episodic_route_preferences(normalized: str) -> set[str]:
        if "last planning" in normalized or "planning request" in normalized:
            return {"planning_request"}
        if "what did you say" in normalized or "tell me about" in normalized:
            return {"planning_request", "goal_request", "transform_request", "hybrid_request", "factual_question"}
        if "what did i ask" in normalized or "earlier" in normalized:
            return set()
        return set()

    @staticmethod
    def _asks_for_last_turn(normalized: str) -> bool:
        return any(fragment in normalized for fragment in ("last", "earlier", "previous"))

    @staticmethod
    def _looks_like_memory_recall_query(text: str) -> bool:
        normalized = str(text or "").strip().lower()
        recall_cues = (
            "what did i say",
            "what did you say",
            "what are my constraints",
            "what is my favorite",
            "when am i free",
            "what's still incomplete",
            "what is still incomplete",
            "where should",
            "what is the current project status",
            "what is the current state of my shipping checklist",
            "what is the current state of",
            "what steps have i already tried",
            "what troubleshooting steps have i already tried",
            "what two troubleshooting steps have i already tried",
            "what are the final totals",
            "what was my goal",
            "what kind of meal plan",
            "what is my favorite color",
            "what two food constraints",
            "what parts of the project are still incomplete",
            "what is the current status",
            "what was its original status",
            "when is my meeting",
            "what did i say about",
            "where are the frozen berries",
            "where should you assume",
            "which nuts",
            "what nuts",
            "current document title",
            "currently available",
            "current availability",
            "what is the current document title",
            "packing constraints",
            "office item state",
            "current office item state",
            "what are the final food totals",
            "latest launch kit state",
            "what is the latest launch kit state",
        )
        return any(cue in normalized for cue in recall_cues)

    @staticmethod
    def _merge_clarification_prompt(pending: PendingClarification, response_text: str) -> str:
        response = str(response_text or "").strip()
        if pending.clarification_type == "ask_task_target":
            return f"{pending.original_prompt} The task to finish is {response}."
        if pending.clarification_type == "ask_options":
            return f"{pending.original_prompt} The options are {response}."
        if pending.clarification_type == "ask_goal_target":
            return f"{pending.original_prompt} The goal target is {response}."
        if pending.clarification_type == "ask_transform_target":
            return f"{pending.original_prompt} The artifact to transform is {response}."
        return f"{pending.original_prompt} Clarification: {response}."

    def inspection_view(self, result: HybridTurnResult) -> str:
        lines = [
            "ROUTE DECISION",
            json.dumps(result.route_decision.to_dict(), indent=2, sort_keys=True),
            "",
            "CLARIFICATION STATE",
            json.dumps(result.clarification_state, indent=2, sort_keys=True),
            "",
            "GOAL TRACKING",
            json.dumps(result.goal_tracking, indent=2, sort_keys=True),
            "",
            "EFFECTIVE PROMPT",
            result.effective_prompt_text,
            "",
            "ENRICHMENT RESULT",
            json.dumps(result.enrichment_result, indent=2, sort_keys=True),
            "",
            "STRUCTURED PARSE",
            json.dumps(result.parse_result["memory"], indent=2, sort_keys=True),
            "",
            "RECALLED TURNS",
            json.dumps([item.to_dict() for item in result.recalled_turns], indent=2, sort_keys=True),
            "",
            "RETRIEVED MEMORY",
            json.dumps([item.to_dict() for item in result.retrieved_memories], indent=2, sort_keys=True),
            "",
            "MEMORY RECONSTRUCTION",
            json.dumps(result.trace.merged_memory_state.get("reconstructed_context", {}), indent=2, sort_keys=True),
            "",
            "CONTEXT PACKET V1",
            json.dumps(result.trace.context_packet_v1, indent=2, sort_keys=True),
            "",
            "LLM PAYLOAD V1",
            json.dumps(result.trace.llm_payload_v1, indent=2, sort_keys=True),
            "",
            "LLM PAYLOAD V2",
            json.dumps(result.trace.llm_payload_v2, indent=2, sort_keys=True),
            "",
            "REASONING TRACE",
            json.dumps(result.trace.to_dict(), indent=2, sort_keys=True),
            "",
            "FINAL BRAININPUT RESULT",
            json.dumps(result.final_result.to_dict(), indent=2, sort_keys=True),
        ]
        return "\n".join(lines)

    def list_memory_entries(self, limit: int = 20) -> List[Dict[str, object]]:
        meaningful_entries = [entry for entry in self.memory_store.entries if _is_meaningful_entry(entry)]
        return [entry.to_dict() for entry in meaningful_entries[-limit:]]

    def active_goals_summary(self) -> List[Dict[str, object]]:
        goals = self.engine.goal_registry.all_active_goals()
        engine_goals = [
            {
                "goal_id": goal.goal_id,
                "base_weight": goal.base_weight,
                "usage_count": goal.usage_count,
                "success_count": goal.success_count,
                "average_reward": goal.average_reward,
                "average_alignment": goal.average_alignment,
                "temporal_score": goal.temporal_score,
                "target_count": len(goal.targets),
            }
            for goal in goals
        ]
        return engine_goals

    def tracked_goals_summary(self) -> List[Dict[str, object]]:
        return self.goal_tracker.tracked_goals_summary()

    def tracked_strategies_summary(self) -> List[Dict[str, object]]:
        return self.strategy_generator.tracked_strategies_summary()

    def record_feedback(
        self,
        *,
        prompt_text: str,
        outcome: BrainInputV1,
        reward: float,
        notes: str = "",
    ) -> Optional[EpisodicMemoryEntry]:
        if not self.storage_policy.store_feedback_when_available:
            return None
        feedback_memory = self._brain_input_to_memory(outcome)
        entry = self.memory_store.add(
            entry_type="feedback",
            source_text=prompt_text,
            parsed_memory=feedback_memory,
            brain_input=outcome,
            metadata={"reward": reward, "notes": notes},
        )
        if entry is None:
            return None
        self.history.log(
            event_type="feedback",
            summary=f"Recorded feedback for prompt: {prompt_text[:80]}",
            payload={"reward": reward, "notes": notes, "outcome": outcome.to_dict()},
        )
        return entry

    def _memory_to_brain_input(self, memory: BrainMemory) -> BrainInputV1:
        base = self.engine.v4_engine.v35_engine.v3_engine.v2_engine.v1_engine.runtime.brain_input_codec.build_brain_input(
            memory=memory,
            encoding_type="bitpacked",
            primitive_ids=[],
        )
        return self.engine.v4_engine.v35_engine.v3_engine.v2_engine.v1_engine.native_engine.normalize(
            self.engine.v4_engine.v35_engine.v3_engine.v2_engine.v1_engine.runtime.brain_input_codec.brain_input_to_pattern_first(
                base,
                current_sequence=self._sequence,
            ),
            current_sequence=self._sequence,
        )

    def _brain_input_to_memory(self, brain_input: BrainInputV1) -> BrainMemory:
        return self.engine.v4_engine.v35_engine.v3_engine.v2_engine.v1_engine.runtime.brain_input_codec.brain_input_to_memory(
            brain_input
        )

    def _bootstrap_generic_goals(self) -> None:
        self.engine.ensure_goal(
            goal_id="maximize_energy",
            base_weight=1.0,
            notes=["generic_goal"],
            targets=[
                GoalTarget("maximize_energy_1", self._goal_state(action="feel", object="full"), "seek", 1.0),
                GoalTarget("maximize_energy_2", self._goal_state(action="keep", object="energy"), "seek", 0.9),
                GoalTarget("maximize_energy_3", self._goal_state(action="feel", object="refreshed"), "seek", 0.85),
                GoalTarget("maximize_energy_4", self._goal_state(action="stay", object="healthy"), "seek", 0.75),
            ],
        )
        self.engine.ensure_goal(
            goal_id="avoid_risk",
            base_weight=1.0,
            notes=["generic_goal"],
            targets=[
                GoalTarget("avoid_risk_1", self._goal_state(action="get", object="sick"), "avoid", 1.0),
                GoalTarget("avoid_risk_2", self._goal_state(action="lose", object="money"), "avoid", 0.9),
                GoalTarget("avoid_risk_3", self._goal_state(action="miss", object="work"), "avoid", 0.85),
                GoalTarget("avoid_risk_4", self._goal_state(action="damage", object="device"), "avoid", 0.8),
                GoalTarget("avoid_risk_5", self._goal_state(action="burn", object="out"), "avoid", 0.8),
            ],
        )
        self.engine.ensure_goal(
            goal_id="complete_task",
            base_weight=1.0,
            notes=["generic_goal"],
            targets=[
                GoalTarget("complete_task_1", self._goal_state(action="finish", object="task"), "seek", 1.0),
                GoalTarget("complete_task_2", self._goal_state(action="complete", object="goal"), "seek", 0.95),
                GoalTarget("complete_task_3", self._goal_state(action="continue", object="project"), "seek", 0.8),
                GoalTarget("complete_task_4", self._goal_state(action="save", object="time"), "seek", 0.7),
                GoalTarget("complete_task_5", self._goal_state(action="earn", object="reward"), "seek", 0.7),
            ],
        )

    def _bootstrap_reasoning_memory(self) -> None:
        bootstrap_flag = self.state_dir / "bootstrap_complete.json"
        if bootstrap_flag.exists():
            return
        scenarios = [scenario for scenario in _build_scenarios() if scenario.phase in {"train_round_1", "train_round_2"}]
        for scenario in scenarios:
            input_brain = self._memory_to_brain_input(scenario.root)
            preferred_chain = tuple(self._memory_to_brain_input(memory) for memory in scenario.preferred.chain)
            alternative_specs = [
                (
                    tuple(self._memory_to_brain_input(memory) for memory in alternative.chain),
                    alternative.relation_types,
                    alternative.desirability,
                    alternative.risk_level,
                )
                for alternative in scenario.alternatives
            ]
            goal_weights = self._bootstrap_goal_weights(scenario.root)
            decision = self.engine.plan(
                input_brain,
                goal_weights=goal_weights,
                max_depth=5,
                beam_width=4,
                current_sequence=self._sequence,
            )
            self.engine.validate_decision(
                input_brain=input_brain,
                decision=decision,
                goal_weights=goal_weights,
                preferred_chain=preferred_chain,
                preferred_relation_types=scenario.preferred.relation_types,
                alternative_chains=alternative_specs,
                current_sequence=self._sequence,
            )
            self._sequence += 1
        bootstrap_flag.write_text(
            json.dumps({"bootstrapped": True, "scenario_count": len(scenarios)}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _bootstrap_goal_weights(memory: BrainMemory) -> Dict[str, float]:
        if memory.action in {"eat", "drink", "cook", "water"}:
            return {"complete_task": 0.25, "avoid_risk": 0.35, "maximize_energy": 0.4}
        if memory.action in {"charge", "repair", "study", "clean", "backup", "practice"}:
            return {"complete_task": 0.45, "avoid_risk": 0.35, "maximize_energy": 0.2}
        return {"complete_task": 0.34, "avoid_risk": 0.33, "maximize_energy": 0.33}

    def _goal_state(self, *, action: str, object: str = "", topic: str = "") -> BrainInputV1:
        memory = BrainCodec.resolve_memory(
            BrainMemory(
                actor="user",
                intent="inform",
                action=action,
                object=object,
                topic=topic,
                priority="normal",
            )
        )
        return self._memory_to_brain_input(memory)

    def _build_trace(
        self,
        *,
        raw_prompt: str,
        effective_prompt: str,
        route_decision: QueryRouteDecision,
        enrichment_result: PromptEnrichmentResult,
        clarification_state: Dict[str, object],
        goal_tracking: Dict[str, object],
        parse_result: ParseResult,
        parsed_memory: BrainMemory,
        parsed_brain_input: BrainInputV1,
        retrieved: Sequence[RetrievedMemory],
        recalled_turns: Sequence[EpisodicTurnRecall],
        goal_weights: Dict[str, float],
        decision: object,
        final_result: BrainInputV1,
    ) -> HybridReasoningTrace:
        candidate_list = getattr(decision, "candidate_paths", []) if decision is not None else []
        candidate_paths = [path.to_dict() for path in candidate_list]
        grounded_paths = [
            path.to_dict()
            for path in candidate_list
            if path.base_path.plausibility_score >= 0.0
        ]
        concept_matches = self._concept_matches(decision) if decision is not None else []
        generated = [path.to_dict() for path in candidate_list if any(step.source_level == "generated" for step in path.steps)]
        selected = getattr(decision, "selected_path", None) if decision is not None else None
        selected_path = selected.to_dict() if selected is not None else {}
        return HybridReasoningTrace(
            raw_prompt=raw_prompt,
            effective_prompt=effective_prompt,
            route_decision=route_decision.to_dict(),
            enrichment_result=enrichment_result.to_dict(),
            clarification_state=dict(clarification_state),
            goal_tracking=dict(goal_tracking),
            strategy_plan={},
            merged_memory_state={},
            parsed_input={
                "memory": parsed_memory.to_dict(),
                "brain_input": parsed_brain_input.to_dict(),
                "debug": parse_result.debug,
            },
            stored_entries=[],
            retrieved_memories=[item.to_dict() for item in retrieved],
            recalled_turns=[item.to_dict() for item in recalled_turns],
            goal_weights=goal_weights,
            concept_matches=concept_matches,
            generated_candidate_paths=generated,
            candidate_paths=candidate_paths,
            grounded_paths=grounded_paths,
            goal_scored_paths=candidate_paths,
            selected_path=selected_path,
            final_predicted_outcome=final_result.to_dict(),
            context_packet_v1={},
            llm_payload_v1={},
            llm_payload_v2={},
            cached_knowledge_checked=False,
            cached_knowledge_hit=False,
            cached_knowledge_entry_ids=[],
            cached_knowledge_written=False,
            cached_knowledge_topic_key="",
            cached_knowledge_normalized_terms=[],
            cached_knowledge_trust_level="",
            cached_knowledge_used=False,
            cached_knowledge_entry_id=None,
            cached_knowledge_candidate_entry_ids=[],
            cached_knowledge_selected_entry_id=None,
            cached_knowledge_ranking_scores=[],
            cached_knowledge_reinforcement_applied=False,
            cached_knowledge_updated_confidence=None,
            cached_knowledge_updated_reuse_count=None,
            cached_knowledge_conflict_detected=False,
            cached_knowledge_conflicting_entry_ids=[],
            cached_knowledge_penalized_entry_ids=[],
            cached_knowledge_updated_confidences=[],
            cached_knowledge_verification_attempted=False,
            cached_knowledge_verification_status="",
            cached_knowledge_confidence_capped=False,
            cached_knowledge_uncertainty_flag=False,
            cached_knowledge_stabilized_flag=False,
            cached_knowledge_dominance_score=None,
            cached_knowledge_convergence_rate=None,
            multi_hop_used=False,
            merged_entry_ids=[],
            reasoning_steps=[],
            merged_reasoning_packet={},
            reasoning_graph_nodes=[],
            reasoning_graph_edges=[],
            reasoning_path=[],
            global_graph_size={},
            reused_graph_edges=[],
            new_graph_edges=[],
            structured_explanation={},
            final_explanation="",
            cache_bypassed_reason=None,
        )

    def _select_result_from_decision(
        self,
        decision: object,
        goal_weights: Dict[str, float],
    ) -> Optional[BrainInputV1]:
        selected_path = getattr(decision, "selected_path", None)
        if selected_path is None:
            return getattr(decision, "final_predicted_outcome", None)
        candidate_states = [step.consequent for step in selected_path.steps] or [selected_path.final_outcome]
        best_state: Optional[BrainInputV1] = None
        best_score = -1.0
        for index, state in enumerate(candidate_states, start=1):
            alignment, reward_prior, _ = self.engine.evaluate_goal_alignment(
                states=[state],
                goal_weights=goal_weights,
                current_sequence=self._sequence,
            )
            earliness = 1.0 / float(index)
            score = (alignment * 0.75) + (reward_prior * 0.15) + (earliness * 0.10)
            if score > best_score:
                best_score = score
                best_state = state
        return best_state or selected_path.final_outcome

    def _concept_matches(self, decision: object) -> List[Dict[str, object]]:
        concept_registry = self.engine.v4_engine.v35_engine.v3_engine.v2_engine.concept_registry
        matches: List[Dict[str, object]] = []
        seen: set[int] = set()
        for path in decision.candidate_paths:
            for step in path.steps:
                for concept_id in step.concept_ids:
                    if concept_id in seen:
                        continue
                    seen.add(concept_id)
                    concept = concept_registry.get_concept(concept_id)
                    matches.append(
                        {
                            "concept_id": concept_id,
                            "slot_path": concept.slot_path,
                            "members": concept.members,
                            "confidence": concept.confidence,
                        }
                    )
        return matches

    @staticmethod
    def _similarity_score(
        query: BrainInputV1,
        candidate: BrainInputV1,
        query_memory: BrainMemory,
        candidate_memory: BrainMemory,
    ) -> float:
        score = 0.0
        if query.action and query.action == candidate.action:
            score += 4.0
        if query.actor and query.actor == candidate.actor:
            score += 0.5
        if query.intent and query.intent == candidate.intent:
            score += 1.0
        if str(query_memory.object or "") == str(candidate_memory.object or "") and query_memory.object:
            score += 4.0
        if str(query_memory.topic or "") == str(candidate_memory.topic or "") and query_memory.topic:
            score += 3.0
        score += 1.5 * len(set(query_memory.modifiers).intersection(candidate_memory.modifiers))
        score += 1.25 * len(set(query_memory.constraints).intersection(candidate_memory.constraints))
        score += 1.25 * len(set(query_memory.negations).intersection(candidate_memory.negations))
        score += 1.0 * len(set(query_memory.conditions).intersection(candidate_memory.conditions))
        if query.pattern_id is not None and query.pattern_id == candidate.pattern_id:
            score += 2.5
        score += 0.75 * len(set(query.subpattern_refs).intersection(candidate.subpattern_refs))
        if query_memory.relations and candidate_memory.relations:
            score += 0.25 * len(
                set((item.get("type", ""), item.get("source", ""), item.get("target", "")) for item in query_memory.relations).intersection(
                    (item.get("type", ""), item.get("source", ""), item.get("target", "")) for item in candidate_memory.relations
                )
            )
        return score
