from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from brain_binary_codec import BrainBinaryCodec
from brain_codec import BrainCodec, BrainMemory
from brain_hierarchical_pattern_codec import (
    BrainHierarchicalPatternCodec,
    BrainHierarchicalPatternMemoryStore,
    BrainHierarchicalPatternRegistry,
)
from brain_input_v1 import BrainInputCodec, BrainInputV1, BrainSemanticTokenRegistry
from brain_pattern_codec import BrainPatternCodec, BrainPatternMemoryStore, BrainPatternStore, PatternSlot
from brain_pattern_evolution import BrainPatternEvolution


SCALAR_MEMORY_FIELDS: Tuple[str, ...] = (
    "actor",
    "intent",
    "action",
    "object",
    "topic",
    "time",
    "priority",
)

LIST_MEMORY_FIELDS: Tuple[str, ...] = (
    "modifiers",
    "constraints",
    "negations",
    "conditions",
)

DEFAULT_OUTPUT_DIR = Path("brain_output") / "benchmarks"


@dataclass
class BrainReasoningRuntime:
    codec: BrainCodec
    binary_codec: BrainBinaryCodec
    pattern_codec: BrainPatternCodec
    pattern_store: BrainPatternStore
    pattern_memory_store: BrainPatternMemoryStore
    hierarchical_codec: BrainHierarchicalPatternCodec
    hierarchical_registry: BrainHierarchicalPatternRegistry
    hierarchical_memory_store: BrainHierarchicalPatternMemoryStore
    pattern_evolution: BrainPatternEvolution
    token_registry: BrainSemanticTokenRegistry
    brain_input_codec: BrainInputCodec


@dataclass(frozen=True)
class ReasoningBenchmarkCase:
    case_id: str
    operation: str
    inputs: Tuple[BrainInputV1, ...]
    parameters: Dict[str, object]
    expected_paths: Dict[str, object]


@dataclass(frozen=True)
class ReasoningTraceStep:
    operation: str
    details: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "operation": self.operation,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class BrainReasoningResult:
    result: BrainInputV1
    reasoning_trace: List[ReasoningTraceStep]

    def to_dict(self) -> Dict[str, object]:
        return {
            "result": self.result.to_dict(),
            "reasoning_trace": [step.to_dict() for step in self.reasoning_trace],
        }


class BrainNativeReasoningEngine:
    """
    Deterministic BrAIn-native reasoning over BrainInputV1 only.

    Internally the engine works on structured BrainMemory objects and
    reprojects every result back into BrainInputV1 using the existing
    pattern/hierarchical/evolved registries.
    """

    def __init__(self, runtime: BrainReasoningRuntime) -> None:
        self.runtime = runtime

    def reason(
        self,
        operation: str,
        inputs: Sequence[BrainInputV1],
        current_sequence: Optional[int] = None,
        include_trace: bool = False,
        **kwargs: object,
    ) -> BrainInputV1 | BrainReasoningResult:
        if include_trace:
            return self.reason_with_trace(
                operation=operation,
                inputs=inputs,
                current_sequence=current_sequence,
                **kwargs,
            )
        operation_key = str(operation).strip().lower()
        if operation_key == "transform":
            return self.transform(inputs[0], updates=self._as_dict(kwargs.get("updates", {})), current_sequence=current_sequence)
        if operation_key == "compare":
            return self.compare(inputs[0], inputs[1], current_sequence=current_sequence)
        if operation_key == "merge":
            return self.merge(inputs[0], inputs[1], current_sequence=current_sequence)
        if operation_key == "negate":
            return self.negate(
                inputs[0],
                target=self._as_optional_str(kwargs.get("target")),
                apply=bool(kwargs.get("apply", True)),
                constraint=self._as_optional_str(kwargs.get("constraint")),
                current_sequence=current_sequence,
            )
        if operation_key == "generalize":
            return self.generalize(inputs[0], current_sequence=current_sequence)
        if operation_key == "specialize":
            return self.specialize(inputs[0], fills=self._as_dict(kwargs.get("fills", {})), current_sequence=current_sequence)
        raise ValueError(f"Unsupported reasoning operation: {operation}")

    def reason_with_trace(
        self,
        operation: str,
        inputs: Sequence[BrainInputV1],
        current_sequence: Optional[int] = None,
        **kwargs: object,
    ) -> BrainReasoningResult:
        operation_key = str(operation).strip().lower()
        if operation_key == "transform":
            updates = self._as_dict(kwargs.get("updates", {}))
            result = self.transform(inputs[0], updates=updates, current_sequence=current_sequence)
            trace = [
                ReasoningTraceStep(
                    operation="transform",
                    details={
                        "updates": {str(key): updates[key] for key in sorted(updates)},
                        "applied_changes": self._diff_inputs(inputs[0], result),
                    },
                ),
                self._reencode_trace_step(result),
            ]
            return BrainReasoningResult(result=result, reasoning_trace=trace)

        if operation_key == "compare":
            left_memory = self.runtime.brain_input_codec.brain_input_to_memory(inputs[0])
            right_memory = self.runtime.brain_input_codec.brain_input_to_memory(inputs[1])
            differences = self._compare_inputs(inputs[0], inputs[1], left_memory, right_memory)
            result = self.compare(inputs[0], inputs[1], current_sequence=current_sequence)
            trace = [
                ReasoningTraceStep(
                    operation="compare",
                    details={
                        "difference_count": len(differences),
                        "differences": [
                            {"path": path, "left": left_value, "right": right_value}
                            for path, left_value, right_value in differences
                        ],
                    },
                ),
                self._reencode_trace_step(result),
            ]
            return BrainReasoningResult(result=result, reasoning_trace=trace)

        if operation_key == "merge":
            result = self.merge(inputs[0], inputs[1], current_sequence=current_sequence)
            trace = [
                ReasoningTraceStep(
                    operation="merge",
                    details={
                        "left_signature": self._input_signature(inputs[0]),
                        "right_signature": self._input_signature(inputs[1]),
                        "merged_changes": self._diff_inputs(inputs[0], result),
                    },
                ),
                self._reencode_trace_step(result),
            ]
            return BrainReasoningResult(result=result, reasoning_trace=trace)

        if operation_key == "negate":
            target = self._as_optional_str(kwargs.get("target"))
            apply = bool(kwargs.get("apply", True))
            constraint = self._as_optional_str(kwargs.get("constraint"))
            result = self.negate(
                inputs[0],
                target=target,
                apply=apply,
                constraint=constraint,
                current_sequence=current_sequence,
            )
            trace = [
                ReasoningTraceStep(
                    operation="negate",
                    details={
                        "target": target,
                        "constraint": constraint,
                        "apply": apply,
                        "applied_changes": self._diff_inputs(inputs[0], result),
                    },
                ),
                self._reencode_trace_step(result),
            ]
            return BrainReasoningResult(result=result, reasoning_trace=trace)

        if operation_key == "generalize":
            result = self.generalize(inputs[0], current_sequence=current_sequence)
            trace = [
                ReasoningTraceStep(
                    operation="generalize",
                    details={
                        "input_encoding_type": inputs[0].encoding_type,
                        "output_encoding_type": result.encoding_type,
                        "output_pattern_id": result.pattern_id,
                        "output_subpattern_refs": list(result.subpattern_refs),
                    },
                ),
                self._reencode_trace_step(result),
            ]
            return BrainReasoningResult(result=result, reasoning_trace=trace)

        if operation_key == "specialize":
            fills = self._as_dict(kwargs.get("fills", {}))
            result = self.specialize(inputs[0], fills=fills, current_sequence=current_sequence)
            trace = [
                ReasoningTraceStep(
                    operation="specialize",
                    details={
                        "requested_fills": {str(key): fills[key] for key in sorted(fills)},
                        "applied_changes": self._diff_inputs(inputs[0], result),
                    },
                ),
                self._reencode_trace_step(result),
            ]
            return BrainReasoningResult(result=result, reasoning_trace=trace)

        raise ValueError(f"Unsupported reasoning operation: {operation}")

    def transform(
        self,
        brain_input: BrainInputV1,
        updates: Dict[str, object],
        current_sequence: Optional[int] = None,
    ) -> BrainInputV1:
        memory = self.runtime.brain_input_codec.brain_input_to_memory(brain_input)
        for path, value in sorted(updates.items()):
            self._write_memory_path(memory, str(path), value)
        self._refresh_standard_relations(memory)
        return self._repattern_memory(memory, current_sequence=current_sequence)

    def compare(
        self,
        left: BrainInputV1,
        right: BrainInputV1,
        current_sequence: Optional[int] = None,
    ) -> BrainInputV1:
        left_memory = self.runtime.brain_input_codec.brain_input_to_memory(left)
        right_memory = self.runtime.brain_input_codec.brain_input_to_memory(right)
        differences = self._compare_inputs(left, right, left_memory, right_memory)
        dominant_path = differences[0][0] if differences else "structure"
        comparison_memory = BrainMemory(
            actor=left.actor or right.actor,
            intent="inform",
            action="compare",
            object="difference" if differences else "equivalent",
            topic=dominant_path,
            modifiers=[path for path, _, _ in differences[1:4]],
            relations=self._build_compare_relations(differences),
            priority="high" if differences else "normal",
        )
        return self._repattern_memory(comparison_memory, current_sequence=current_sequence)

    def merge(
        self,
        primary: BrainInputV1,
        secondary: BrainInputV1,
        current_sequence: Optional[int] = None,
    ) -> BrainInputV1:
        left = self.runtime.brain_input_codec.brain_input_to_memory(primary)
        right = self.runtime.brain_input_codec.brain_input_to_memory(secondary)
        merged = BrainMemory()
        for field_name in SCALAR_MEMORY_FIELDS:
            merged_value = self._merge_scalar(
                self._normalize_scalar(getattr(left, field_name)),
                self._normalize_scalar(getattr(right, field_name)),
            )
            setattr(merged, field_name, merged_value)
        for field_name in LIST_MEMORY_FIELDS:
            merged_values = self._stable_union(
                list(getattr(left, field_name)),
                list(getattr(right, field_name)),
            )
            setattr(merged, field_name, merged_values)
        merged.relations = self._merge_relations(left.relations, right.relations)
        self._refresh_standard_relations(merged)
        return self._repattern_memory(BrainCodec.resolve_memory(merged), current_sequence=current_sequence)

    def negate(
        self,
        brain_input: BrainInputV1,
        target: Optional[str] = None,
        apply: bool = True,
        constraint: Optional[str] = None,
        current_sequence: Optional[int] = None,
    ) -> BrainInputV1:
        memory = self.runtime.brain_input_codec.brain_input_to_memory(brain_input)
        if target:
            if apply:
                memory.negations = self._stable_union(memory.negations, [target])
                memory.modifiers = [item for item in memory.modifiers if item != target]
            else:
                memory.negations = [item for item in memory.negations if item != target]
        if constraint:
            if apply:
                memory.constraints = self._stable_union(memory.constraints, [constraint])
            else:
                memory.constraints = [item for item in memory.constraints if item != constraint]
        self._refresh_standard_relations(memory)
        return self._repattern_memory(BrainCodec.resolve_memory(memory), current_sequence=current_sequence)

    def generalize(
        self,
        brain_input: BrainInputV1,
        current_sequence: Optional[int] = None,
    ) -> BrainInputV1:
        memory = self.runtime.brain_input_codec.brain_input_to_memory(brain_input)
        generalized = self._generalized_pattern_input(memory, current_sequence=current_sequence)
        if generalized is not None:
            return generalized
        return self._repattern_memory(memory, current_sequence=current_sequence)

    def specialize(
        self,
        brain_input: BrainInputV1,
        fills: Dict[str, object],
        current_sequence: Optional[int] = None,
    ) -> BrainInputV1:
        memory = self.runtime.brain_input_codec.brain_input_to_memory(brain_input)
        for path, value in sorted(fills.items()):
            path_key = str(path)
            if self._is_missing_path(memory, path_key):
                self._write_memory_path(memory, path_key, value)
        self._refresh_standard_relations(memory)
        return self._repattern_memory(memory, current_sequence=current_sequence)

    def normalize(self, brain_input: BrainInputV1, current_sequence: Optional[int] = None) -> BrainInputV1:
        memory = self.runtime.brain_input_codec.brain_input_to_memory(brain_input)
        return self._repattern_memory(memory, current_sequence=current_sequence)

    def _repattern_memory(
        self,
        memory: BrainMemory,
        current_sequence: Optional[int] = None,
    ) -> BrainInputV1:
        resolved = BrainCodec.resolve_memory(memory)
        self._ensure_memory_patterns(resolved)
        base = self.runtime.brain_input_codec.build_brain_input(
            memory=resolved,
            encoding_type="bitpacked",
            primitive_ids=[],
        )
        return self.runtime.brain_input_codec.brain_input_to_pattern_first(
            base,
            current_sequence=current_sequence,
        )

    def _generalized_pattern_input(
        self,
        memory: BrainMemory,
        current_sequence: Optional[int] = None,
    ) -> Optional[BrainInputV1]:
        self._ensure_memory_patterns(memory)
        bundle = self.runtime.codec.encode_memory_bundle(memory)
        candidates = [
            pattern
            for pattern in self.runtime.hierarchical_registry.all_active_patterns()
            if pattern.generalization_level > 0
        ]
        for pattern in sorted(
            candidates,
            key=lambda item: (-item.generalization_level, item.pattern_id),
        ):
            if self.runtime.brain_input_codec._is_lossless_pattern_reference(
                memory=memory,
                encoding_type="evolved_pattern",
                pattern_id=pattern.pattern_id,
                subpattern_refs=pattern.subpattern_ids,
            ):
                return self.runtime.brain_input_codec.build_brain_input(
                    memory=memory,
                    encoding_type="evolved_pattern",
                    pattern_id=pattern.pattern_id,
                    subpattern_refs=pattern.subpattern_ids,
                    primitive_ids=[],
                    temporal_score=round(
                        self.runtime.pattern_evolution.score_pattern(
                            pattern,
                            current_sequence=current_sequence,
                        ).score,
                        4,
                    ),
                )
        decision = self.runtime.pattern_evolution.select_adaptive_encoding(
            memory_bundle=bundle,
            current_sequence=current_sequence,
            include_subpatterns=True,
            update_usage=False,
        )
        if decision is None or decision.mode == "full":
            return None
        encoding_type = "evolved_pattern" if decision.mode == "pattern" else "evolved_subpatterns"
        return self.runtime.brain_input_codec.build_brain_input(
            memory=memory,
            encoding_type=encoding_type,
            pattern_id=decision.pattern_id,
            subpattern_refs=decision.subpattern_ids,
            primitive_ids=[],
            temporal_score=round(decision.reinforcement_score, 4),
        )

    def _compare_inputs(
        self,
        left: BrainInputV1,
        right: BrainInputV1,
        left_memory: BrainMemory,
        right_memory: BrainMemory,
    ) -> List[Tuple[str, str, str]]:
        reference_differences = self._compare_with_shared_pattern_reference(left, right, left_memory, right_memory)
        if reference_differences is not None:
            return reference_differences

        differences: List[Tuple[str, str, str]] = []
        for field_name in SCALAR_MEMORY_FIELDS:
            left_value = self._normalize_scalar(getattr(left_memory, field_name))
            right_value = self._normalize_scalar(getattr(right_memory, field_name))
            if left_value != right_value:
                differences.append((field_name, left_value, right_value))
        for field_name in LIST_MEMORY_FIELDS:
            left_values = list(getattr(left_memory, field_name))
            right_values = list(getattr(right_memory, field_name))
            max_length = max(len(left_values), len(right_values))
            for index in range(max_length):
                left_value = left_values[index] if index < len(left_values) else ""
                right_value = right_values[index] if index < len(right_values) else ""
                if left_value != right_value:
                    differences.append((f"{field_name}[{index}]", left_value, right_value))
        if left_memory.relations != right_memory.relations:
            left_fingerprint = self._relation_fingerprint(left_memory.relations)
            right_fingerprint = self._relation_fingerprint(right_memory.relations)
            differences.append(("relations", left_fingerprint, right_fingerprint))
        return differences

    def _compare_with_shared_pattern_reference(
        self,
        left: BrainInputV1,
        right: BrainInputV1,
        left_memory: BrainMemory,
        right_memory: BrainMemory,
    ) -> Optional[List[Tuple[str, str, str]]]:
        if left.encoding_type != right.encoding_type:
            return None
        if left.pattern_id != right.pattern_id:
            return None
        if list(left.subpattern_refs) != list(right.subpattern_refs):
            return None
        reference = self.runtime.brain_input_codec._resolve_pattern_reference(
            encoding_type=left.encoding_type,
            pattern_id=left.pattern_id,
            subpattern_refs=left.subpattern_refs,
        )
        if reference is None:
            return None
        left_slots = reference["extractor"](left_memory)
        right_slots = reference["extractor"](right_memory)
        variable_slots: Sequence[PatternSlot] = reference["variable_slots"]
        differences: List[Tuple[str, str, str]] = []
        for slot in sorted(variable_slots, key=lambda item: item.slot_index):
            left_value = str(left_slots.get(slot.slot_name, "") or "")
            right_value = str(right_slots.get(slot.slot_name, "") or "")
            if left_value != right_value:
                differences.append((slot.path, left_value, right_value))
        return differences

    @staticmethod
    def _build_compare_relations(
        differences: Sequence[Tuple[str, str, str]],
    ) -> List[Dict[str, str]]:
        relations: List[Dict[str, str]] = []
        if not differences:
            relations.append({"type": "compare_left", "source": "structure", "target": "equivalent"})
            relations.append({"type": "compare_right", "source": "structure", "target": "equivalent"})
            return relations
        for path, left_value, right_value in differences:
            relations.append(
                {
                    "type": "compare_left",
                    "source": path,
                    "target": left_value or "empty",
                }
            )
            relations.append(
                {
                    "type": "compare_right",
                    "source": path,
                    "target": right_value or "empty",
                }
            )
        return relations

    @staticmethod
    def _merge_scalar(left_value: str, right_value: str) -> str:
        if left_value:
            return left_value
        return right_value

    @staticmethod
    def _merge_relations(
        left_relations: Sequence[Dict[str, str]],
        right_relations: Sequence[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        seen: set[Tuple[str, str, str]] = set()
        merged: List[Dict[str, str]] = []
        for relation in list(left_relations) + list(right_relations):
            key = (
                str(relation.get("type", "")),
                str(relation.get("source", "")),
                str(relation.get("target", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append({"type": key[0], "source": key[1], "target": key[2]})
        return merged

    @staticmethod
    def _stable_union(left_values: Sequence[str], right_values: Sequence[str]) -> List[str]:
        ordered: List[str] = []
        seen: set[str] = set()
        for value in list(left_values) + list(right_values):
            normalized = str(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    @staticmethod
    def _normalize_scalar(value: object) -> str:
        return str(value or "")

    @staticmethod
    def _relation_fingerprint(relations: Sequence[Dict[str, str]]) -> str:
        normalized = [
            (
                str(relation.get("type", "")),
                str(relation.get("source", "")),
                str(relation.get("target", "")),
            )
            for relation in relations
        ]
        return json.dumps(sorted(normalized), separators=(",", ":"))

    def _is_missing_path(self, memory: BrainMemory, path: str) -> bool:
        value = self._read_memory_path(memory, path)
        if isinstance(value, list):
            return not value
        return str(value or "") == ""

    def _read_memory_path(self, memory: BrainMemory, path: str) -> object:
        if path in SCALAR_MEMORY_FIELDS:
            return getattr(memory, path)
        if path in LIST_MEMORY_FIELDS:
            return list(getattr(memory, path))
        if "[" in path and path.endswith("]"):
            field_name, index = self._parse_list_path(path)
            values = list(getattr(memory, field_name))
            return values[index] if index < len(values) else ""
        if path == "relations":
            return list(memory.relations)
        raise ValueError(f"Unsupported memory path: {path}")

    def _write_memory_path(self, memory: BrainMemory, path: str, value: object) -> None:
        if path in SCALAR_MEMORY_FIELDS:
            setattr(memory, path, self._normalize_scalar(value))
            return
        if path in LIST_MEMORY_FIELDS:
            items = [str(item) for item in value] if isinstance(value, list) else [self._normalize_scalar(value)]
            setattr(memory, path, [item for item in items if item])
            return
        if "[" in path and path.endswith("]"):
            field_name, index = self._parse_list_path(path)
            values = list(getattr(memory, field_name))
            while len(values) <= index:
                values.append("")
            values[index] = self._normalize_scalar(value)
            setattr(memory, field_name, [item for item in values if item])
            return
        raise ValueError(f"Unsupported memory path for write: {path}")

    @staticmethod
    def _parse_list_path(path: str) -> Tuple[str, int]:
        field_name, _, suffix = path.partition("[")
        return field_name, int(suffix[:-1])

    @staticmethod
    def _as_dict(value: object) -> Dict[str, object]:
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _as_optional_str(value: object) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def _reencode_trace_step(self, result: BrainInputV1) -> ReasoningTraceStep:
        return ReasoningTraceStep(
            operation="re_encode",
            details={
                "encoding_type": result.encoding_type,
                "pattern_id": result.pattern_id,
                "subpattern_refs": list(result.subpattern_refs),
                "temporal_score": result.temporal_score,
            },
        )

    def _diff_inputs(
        self,
        before: BrainInputV1,
        after: BrainInputV1,
    ) -> List[Dict[str, object]]:
        before_memory = self.runtime.brain_input_codec.brain_input_to_memory(before)
        after_memory = self.runtime.brain_input_codec.brain_input_to_memory(after)
        changes: List[Dict[str, object]] = []
        for field_name in SCALAR_MEMORY_FIELDS:
            previous = self._normalize_scalar(getattr(before_memory, field_name))
            current = self._normalize_scalar(getattr(after_memory, field_name))
            if previous != current:
                changes.append({"path": field_name, "before": previous, "after": current})
        for field_name in LIST_MEMORY_FIELDS:
            before_values = list(getattr(before_memory, field_name))
            after_values = list(getattr(after_memory, field_name))
            max_length = max(len(before_values), len(after_values))
            for index in range(max_length):
                previous = before_values[index] if index < len(before_values) else ""
                current = after_values[index] if index < len(after_values) else ""
                if previous != current:
                    changes.append(
                        {
                            "path": f"{field_name}[{index}]",
                            "before": previous,
                            "after": current,
                        }
                    )
        if before_memory.relations != after_memory.relations:
            changes.append(
                {
                    "path": "relations",
                    "before": before_memory.relations,
                    "after": after_memory.relations,
                }
            )
        return changes

    @staticmethod
    def _input_signature(brain_input: BrainInputV1) -> Dict[str, object]:
        return {
            "encoding_type": brain_input.encoding_type,
            "pattern_id": brain_input.pattern_id,
            "subpattern_refs": list(brain_input.subpattern_refs),
            "action": brain_input.action,
            "object": str(brain_input.slots.get("object", "")),
            "topic": str(brain_input.slots.get("topic", "")),
            "modifiers": [str(item) for item in brain_input.slots.get("modifiers", [])],
            "constraints": [str(item) for item in brain_input.slots.get("constraints", [])],
            "negations": [str(item) for item in brain_input.slots.get("negations", [])],
            "conditions": [str(item) for item in brain_input.slots.get("conditions", [])],
        }

    def _ensure_memory_patterns(self, memory: BrainMemory) -> None:
        bundle = self.runtime.codec.encode_memory_bundle(BrainCodec.resolve_memory(memory))
        self.runtime.pattern_store.ensure_pattern(bundle)
        candidate = self.runtime.hierarchical_codec.extract_subpatterns(bundle)
        ensured_subpattern_ids: List[int] = []
        for subpattern in candidate.subpatterns:
            ensured, _ = self.runtime.hierarchical_registry.ensure_subpattern(subpattern)
            ensured_subpattern_ids.append(ensured.subpattern_id)
        self.runtime.hierarchical_registry.ensure_pattern(candidate, ensured_subpattern_ids)

    @staticmethod
    def _refresh_standard_relations(memory: BrainMemory) -> None:
        if memory.action == "compare":
            return
        relations: List[Dict[str, str]] = []
        if memory.object and memory.topic:
            relations.append({"type": "about", "source": memory.object, "target": memory.topic})
        for modifier in memory.modifiers:
            if memory.object:
                relations.append({"type": "has_modifier", "source": memory.object, "target": modifier})
        for negation in memory.negations:
            if memory.object:
                relations.append({"type": "negates", "source": memory.object, "target": negation})
        for constraint in memory.constraints:
            if memory.object:
                relations.append({"type": "constraint", "source": memory.object, "target": constraint})
        for condition in memory.conditions:
            if memory.object:
                relations.append({"type": "condition", "source": memory.object, "target": condition})
        if memory.time:
            relations.append(
                {
                    "type": "time",
                    "source": memory.object or memory.action or "request",
                    "target": memory.time,
                }
            )
        memory.relations = relations


def build_reasoning_runtime(
    state_dir: str | Path = DEFAULT_OUTPUT_DIR / "brain_native_reasoning.state",
) -> BrainReasoningRuntime:
    resolved = Path(state_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    codec = BrainCodec(registry_path=resolved / "brain_registry.json")
    binary_codec = BrainBinaryCodec(codec=codec)
    pattern_codec = BrainPatternCodec(codec=codec, binary_codec=binary_codec)
    pattern_store = BrainPatternStore(
        pattern_codec=pattern_codec,
        pattern_path=resolved / "brain_patterns.json",
    )
    hierarchical_codec = BrainHierarchicalPatternCodec(codec=codec, binary_codec=binary_codec)
    hierarchical_registry = BrainHierarchicalPatternRegistry(
        codec=hierarchical_codec,
        registry_path=resolved / "brain_hierarchical_patterns.json",
    )
    pattern_evolution = BrainPatternEvolution(
        codec=hierarchical_codec,
        registry=hierarchical_registry,
    )
    token_registry = BrainSemanticTokenRegistry(
        registry_path=resolved / "brain_semantic_tokens.json",
    )
    brain_input_codec = BrainInputCodec(
        codec=codec,
        binary_codec=binary_codec,
        token_registry=token_registry,
        pattern_codec=pattern_codec,
        pattern_store=pattern_store,
        hierarchical_codec=hierarchical_codec,
        hierarchical_registry=hierarchical_registry,
        pattern_evolution=pattern_evolution,
    )
    return BrainReasoningRuntime(
        codec=codec,
        binary_codec=binary_codec,
        pattern_codec=pattern_codec,
        pattern_store=pattern_store,
        pattern_memory_store=BrainPatternMemoryStore(pattern_codec=pattern_codec, pattern_store=pattern_store),
        hierarchical_codec=hierarchical_codec,
        hierarchical_registry=hierarchical_registry,
        hierarchical_memory_store=BrainHierarchicalPatternMemoryStore(codec=hierarchical_codec, registry=hierarchical_registry),
        pattern_evolution=pattern_evolution,
        token_registry=token_registry,
        brain_input_codec=brain_input_codec,
    )


def seed_reasoning_patterns(
    runtime: BrainReasoningRuntime,
    memories: Sequence[BrainMemory],
) -> Dict[str, int]:
    for index, memory in enumerate(memories, start=1):
        bundle = runtime.codec.encode_memory_bundle(BrainCodec.resolve_memory(memory))
        runtime.pattern_memory_store.add_memory(index, bundle)
        runtime.hierarchical_memory_store.add_memory(index, bundle)
    summary = runtime.pattern_evolution.evolve_registry()
    runtime.token_registry.save()
    return {
        "pattern_count": len(runtime.pattern_store.patterns),
        "hierarchical_pattern_count": len(runtime.hierarchical_registry.patterns),
        "hierarchical_subpattern_count": len(runtime.hierarchical_registry.subpatterns),
        "generalized_pattern_count": summary.generalized_pattern_count,
    }
