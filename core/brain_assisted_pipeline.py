from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from brain_codec import BrainCodec, BrainMemory
from brain_input_v1 import BrainInputV1
from brain_llm_interface import BrainLLMInterface, LLMInterfaceOutput
from brain_parser import BrainParser
from brain_reasoning_engine import (
    BrainNativeReasoningEngine,
    BrainReasoningResult,
    ReasoningTraceStep,
    build_reasoning_runtime,
)


@dataclass(frozen=True)
class StoredBrainInput:
    memory_id: int
    source_text: str
    parsed_memory: BrainMemory
    brain_input: BrainInputV1

    def to_dict(self) -> Dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "source_text": self.source_text,
            "parsed_memory": self.parsed_memory.to_dict(),
            "brain_input": self.brain_input.to_dict(),
        }


@dataclass(frozen=True)
class RetrievedBrainInput:
    memory_id: int
    score: float
    entry: StoredBrainInput

    def to_dict(self) -> Dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "score": self.score,
            "entry": self.entry.to_dict(),
        }


@dataclass(frozen=True)
class BrainAssistedPipelineResult:
    user_input_text: str
    parsed_brain_input: BrainInputV1
    retrieved_memories: List[RetrievedBrainInput]
    result_brain_input: BrainInputV1
    reasoning_trace: List[ReasoningTraceStep]
    reconstructed_text: str
    llm_output: LLMInterfaceOutput

    def to_dict(self) -> Dict[str, object]:
        return {
            "user_input_text": self.user_input_text,
            "parsed_brain_input": self.parsed_brain_input.to_dict(),
            "retrieved_memories": [item.to_dict() for item in self.retrieved_memories],
            "result_brain_input": self.result_brain_input.to_dict(),
            "reasoning_trace": [step.to_dict() for step in self.reasoning_trace],
            "reconstructed_text": self.reconstructed_text,
            "llm_output": self.llm_output.to_dict(),
        }


class BrainAssistedReasoningPipeline:
    """
    End-to-end proof-of-concept pipeline:

    text -> parser -> BrainInputV1 -> storage/retrieval -> native reasoning ->
    BrainInputV1 result -> optional reconstruction -> LLM-facing explanation
    """

    def __init__(
        self,
        state_dir: str | Path = Path("brain_output") / "benchmarks" / "brain_assisted_pipeline.state",
    ) -> None:
        self.parser = BrainParser()
        self.codec = BrainCodec()
        self.runtime = build_reasoning_runtime(state_dir)
        self.engine = BrainNativeReasoningEngine(self.runtime)
        self.llm = BrainLLMInterface()
        self.entries: List[StoredBrainInput] = []
        self._next_memory_id = 1
        self._sequence = 1

    def parse_to_brain_input(self, text: str, actor: str = "user") -> BrainInputV1:
        parse_result = self.parser.parse(text, actor=actor)
        memory = BrainCodec.resolve_memory(parse_result.memory)
        base = self.runtime.brain_input_codec.build_brain_input(
            memory=memory,
            encoding_type="bitpacked",
            primitive_ids=[],
        )
        brain_input = self.runtime.brain_input_codec.brain_input_to_pattern_first(
            base,
            current_sequence=self._sequence,
        )
        self._sequence += 1
        return brain_input

    def store_in_brain_memory(
        self,
        brain_input: BrainInputV1,
        source_text: str = "",
    ) -> StoredBrainInput:
        normalized = self.engine.normalize(brain_input, current_sequence=self._sequence)
        self._sequence += 1
        memory = self.runtime.brain_input_codec.brain_input_to_memory(normalized)
        bundle = self.runtime.codec.encode_memory_bundle(memory)
        memory_id = self._next_memory_id
        self._next_memory_id += 1
        self.runtime.pattern_memory_store.add_memory(memory_id, bundle)
        self.runtime.hierarchical_memory_store.add_memory(memory_id, bundle)
        entry = StoredBrainInput(
            memory_id=memory_id,
            source_text=source_text,
            parsed_memory=memory,
            brain_input=normalized,
        )
        self.entries.append(entry)
        return entry

    def retrieve_relevant_memory(
        self,
        query: BrainInputV1,
        top_k: int = 3,
        exclude_memory_id: Optional[int] = None,
    ) -> List[RetrievedBrainInput]:
        query_memory = self.runtime.brain_input_codec.brain_input_to_memory(query)
        scored: List[RetrievedBrainInput] = []
        for entry in self.entries:
            if exclude_memory_id is not None and entry.memory_id == exclude_memory_id:
                continue
            score = self._similarity_score(query, entry.brain_input, query_memory, entry.parsed_memory)
            if score <= 0:
                continue
            scored.append(RetrievedBrainInput(memory_id=entry.memory_id, score=round(score, 4), entry=entry))
        return sorted(
            scored,
            key=lambda item: (-item.score, item.memory_id),
        )[:top_k]

    def run(
        self,
        user_input_text: str,
        operation: str,
        actor: str = "user",
        reasoning_params: Optional[Dict[str, object]] = None,
        supporting_texts: Optional[Sequence[str]] = None,
        top_k: int = 3,
        store_result: bool = True,
    ) -> BrainAssistedPipelineResult:
        for text in supporting_texts or []:
            parsed = self.parse_to_brain_input(text, actor=actor)
            self.store_in_brain_memory(parsed, source_text=text)

        parsed_brain_input = self.parse_to_brain_input(user_input_text, actor=actor)
        stored_query = self.store_in_brain_memory(parsed_brain_input, source_text=user_input_text)
        retrieved = self.retrieve_relevant_memory(
            parsed_brain_input,
            top_k=top_k,
            exclude_memory_id=stored_query.memory_id,
        )

        effective_params = dict(reasoning_params or {})
        reasoning_inputs = self._select_reasoning_inputs(
            operation=operation,
            query=parsed_brain_input,
            retrieved=retrieved,
            reasoning_params=effective_params,
        )

        reasoning_result = self.engine.reason_with_trace(
            operation=operation,
            inputs=reasoning_inputs,
            current_sequence=self._sequence,
            **effective_params,
        )
        self._sequence += 1

        trace = self._pipeline_trace(retrieved=retrieved, reasoning_result=reasoning_result)
        result_brain_input = reasoning_result.result
        reconstructed_text = self.codec.reconstruct_text(
            self.runtime.brain_input_codec.brain_input_to_memory(result_brain_input)
        )
        llm_output = self.llm.llm_interface(
            brain_input=result_brain_input,
            reasoning_trace=trace,
            user_input_text=user_input_text,
        )

        if store_result:
            self.store_in_brain_memory(result_brain_input, source_text="")

        return BrainAssistedPipelineResult(
            user_input_text=user_input_text,
            parsed_brain_input=parsed_brain_input,
            retrieved_memories=retrieved,
            result_brain_input=result_brain_input,
            reasoning_trace=trace,
            reconstructed_text=reconstructed_text,
            llm_output=llm_output,
        )

    def _select_reasoning_inputs(
        self,
        operation: str,
        query: BrainInputV1,
        retrieved: Sequence[RetrievedBrainInput],
        reasoning_params: Dict[str, object],
    ) -> List[BrainInputV1]:
        operation_key = str(operation).strip().lower()
        if operation_key in {"transform", "negate", "generalize"}:
            return [query]
        if operation_key == "specialize":
            if "fills" not in reasoning_params:
                reasoning_params["fills"] = self._derive_specialize_fills(query, retrieved[0].entry.brain_input) if retrieved else {}
            return [query]
        if operation_key in {"compare", "merge"}:
            if not retrieved:
                raise ValueError(f"Operation '{operation}' requires at least one retrieved memory.")
            return [query, retrieved[0].entry.brain_input]
        raise ValueError(f"Unsupported pipeline reasoning operation: {operation}")

    def _pipeline_trace(
        self,
        retrieved: Sequence[RetrievedBrainInput],
        reasoning_result: BrainReasoningResult,
    ) -> List[ReasoningTraceStep]:
        trace: List[ReasoningTraceStep] = []
        if retrieved:
            selected = retrieved[0]
            trace.append(
                ReasoningTraceStep(
                    operation="retrieve_memory",
                    details={
                        "candidate_count": len(retrieved),
                        "selected_memory_id": selected.memory_id,
                        "selected_score": selected.score,
                        "selected_encoding_type": selected.entry.brain_input.encoding_type,
                    },
                )
            )
        trace.extend(reasoning_result.reasoning_trace)
        return trace

    def _derive_specialize_fills(
        self,
        query: BrainInputV1,
        reference: BrainInputV1,
    ) -> Dict[str, object]:
        query_memory = self.runtime.brain_input_codec.brain_input_to_memory(query)
        reference_memory = self.runtime.brain_input_codec.brain_input_to_memory(reference)
        fills: Dict[str, object] = {}
        for field_name in ("action", "object", "topic", "time", "priority"):
            query_value = str(getattr(query_memory, field_name) or "")
            reference_value = str(getattr(reference_memory, field_name) or "")
            if not query_value and reference_value:
                fills[field_name] = reference_value
        for field_name in ("modifiers", "constraints", "negations", "conditions"):
            query_values = list(getattr(query_memory, field_name))
            reference_values = list(getattr(reference_memory, field_name))
            for index, value in enumerate(reference_values):
                if index >= len(query_values) or not query_values[index]:
                    fills[f"{field_name}[{index}]"] = value
        return fills

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
        elif (not query_memory.topic and candidate_memory.topic) or (query_memory.topic and not candidate_memory.topic):
            score += 1.0
        score += 1.5 * len(set(query_memory.modifiers).intersection(candidate_memory.modifiers))
        score += 1.25 * len(set(query_memory.constraints).intersection(candidate_memory.constraints))
        score += 1.25 * len(set(query_memory.negations).intersection(candidate_memory.negations))
        score += 1.0 * len(set(query_memory.conditions).intersection(candidate_memory.conditions))
        if query_memory.time and query_memory.time == candidate_memory.time:
            score += 1.0
        if query_memory.priority and query_memory.priority == candidate_memory.priority:
            score += 0.5
        if query.pattern_id is not None and query.pattern_id == candidate.pattern_id:
            score += 2.5
        score += 0.75 * len(set(query.subpattern_refs).intersection(candidate.subpattern_refs))
        if query_memory.relations and candidate_memory.relations:
            score += 0.25 * len(
                set(
                    (item.get("type", ""), item.get("source", ""), item.get("target", ""))
                    for item in query_memory.relations
                ).intersection(
                    (item.get("type", ""), item.get("source", ""), item.get("target", ""))
                    for item in candidate_memory.relations
                )
            )
        return score
