from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from brain_codec import BrainCodec
from brain_input_v1 import BrainInputV1
from brain_reasoning_engine import ReasoningTraceStep


@dataclass(frozen=True)
class LLMInterfaceOutput:
    structured_text: str
    reasoning_explanation: str
    user_facing_text: str
    messages: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "structured_text": self.structured_text,
            "reasoning_explanation": self.reasoning_explanation,
            "user_facing_text": self.user_facing_text,
            "messages": [dict(message) for message in self.messages],
        }


class BrainLLMInterface:
    """
    Presentation-only interface for BrAIn reasoning results.

    This layer does not alter structure or perform reasoning. It turns a
    BrainInputV1 result plus deterministic reasoning trace into user-facing
    language and optional prompt messages that a real LLM could format.
    """

    def __init__(self) -> None:
        self.codec = BrainCodec()

    def llm_interface(
        self,
        brain_input: BrainInputV1,
        reasoning_trace: Sequence[ReasoningTraceStep],
        user_input_text: str = "",
    ) -> LLMInterfaceOutput:
        structured_text = self.brain_input_to_natural_language(brain_input)
        reasoning_explanation = self.trace_to_explanation(reasoning_trace)
        user_facing_text = self.render_user_text(
            structured_text=structured_text,
            reasoning_explanation=reasoning_explanation,
            user_input_text=user_input_text,
        )
        return LLMInterfaceOutput(
            structured_text=structured_text,
            reasoning_explanation=reasoning_explanation,
            user_facing_text=user_facing_text,
            messages=self.build_messages(
                brain_input=brain_input,
                reasoning_trace=reasoning_trace,
                user_input_text=user_input_text,
                structured_text=structured_text,
                reasoning_explanation=reasoning_explanation,
            ),
        )

    def brain_input_to_natural_language(self, brain_input: BrainInputV1) -> str:
        memory = self._brain_input_to_memory(brain_input)
        return self.codec.reconstruct_text(memory)

    def trace_to_explanation(self, reasoning_trace: Sequence[ReasoningTraceStep]) -> str:
        parts: List[str] = []
        for step in reasoning_trace:
            parts.append(self._step_to_sentence(step))
        return " ".join(part for part in parts if part).strip()

    def render_user_text(
        self,
        structured_text: str,
        reasoning_explanation: str,
        user_input_text: str = "",
    ) -> str:
        if user_input_text.strip():
            return f'You asked for "{user_input_text.strip()}". Result: {structured_text} {reasoning_explanation}'.strip()
        return f"Result: {structured_text} {reasoning_explanation}".strip()

    def build_messages(
        self,
        brain_input: BrainInputV1,
        reasoning_trace: Sequence[ReasoningTraceStep],
        user_input_text: str,
        structured_text: str,
        reasoning_explanation: str,
    ) -> List[Dict[str, str]]:
        payload = {
            "user_input_text": user_input_text,
            "brain_input": brain_input.to_dict(),
            "reasoning_trace": [step.to_dict() for step in reasoning_trace],
            "structured_text": structured_text,
            "reasoning_explanation": reasoning_explanation,
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are the BrAIn presentation layer. Do not change the provided structure. "
                    "Only explain the supplied BrainInputV1 result and deterministic reasoning trace."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, indent=2, sort_keys=True),
            },
        ]

    @staticmethod
    def _brain_input_to_memory(brain_input: BrainInputV1):
        from brain_codec import BrainMemory

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

    def _step_to_sentence(self, step: ReasoningTraceStep) -> str:
        operation = step.operation
        details = step.details
        if operation == "retrieve_memory":
            selected_id = details.get("selected_memory_id")
            selected_score = details.get("selected_score")
            return f"I retrieved the closest stored structure (memory {selected_id}, score {selected_score})."
        if operation == "transform":
            changes = self._format_changes(details.get("applied_changes", []))
            return f"I transformed the structure by applying {json.dumps(details.get('updates', {}), sort_keys=True)}. {changes}"
        if operation == "compare":
            differences = details.get("differences", [])
            if isinstance(differences, list) and differences:
                first = differences[0]
                if isinstance(first, dict):
                    return (
                        f"I compared the structures and found a difference at {first.get('path', 'unknown')} "
                        f"between {first.get('left', 'empty')} and {first.get('right', 'empty')}."
                    )
            return "I compared the structures and found them equivalent."
        if operation == "merge":
            changes = self._format_changes(details.get("merged_changes", []))
            return f"I merged compatible slots from the selected structures. {changes}"
        if operation == "negate":
            changes = self._format_changes(details.get("applied_changes", []))
            return f"I updated negations or constraints deterministically. {changes}"
        if operation == "generalize":
            return (
                f"I generalized the structure from {details.get('input_encoding_type', 'unknown')} "
                f"to {details.get('output_encoding_type', 'unknown')}."
            )
        if operation == "specialize":
            changes = self._format_changes(details.get("applied_changes", []))
            return f"I filled missing structure slots from deterministic specialization rules. {changes}"
        if operation == "re_encode":
            return (
                f"I re-encoded the result as {details.get('encoding_type', 'unknown')} "
                f"with pattern {details.get('pattern_id')}."
            )
        return f"I recorded the {operation} step."

    @staticmethod
    def _format_changes(changes: object) -> str:
        if not isinstance(changes, list) or not changes:
            return "No structural fields changed."
        fragments: List[str] = []
        for change in changes[:3]:
            if not isinstance(change, dict):
                continue
            fragments.append(
                f"{change.get('path', 'field')} moved from {change.get('before', 'empty')} to {change.get('after', 'empty')}"
            )
        if not fragments:
            return "No structural fields changed."
        return "Changes: " + "; ".join(fragments) + "."
