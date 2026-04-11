from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from brain_codec import BrainCodec, BrainMemory
from brain_input_v1 import BrainInputV1
from brain_reasoning_engine import (
    DEFAULT_OUTPUT_DIR,
    BrainNativeReasoningEngine,
    build_reasoning_runtime,
    seed_reasoning_patterns,
)


@dataclass(frozen=True)
class ReasoningCase:
    case_id: str
    operation: str
    inputs: Tuple[BrainInputV1, ...]
    parameters: Dict[str, object]
    expected_paths: Dict[str, object]
    expected_encoding_types: Tuple[str, ...] = ()
    expect_pattern_reuse: Optional[bool] = None


def make_structured_memory(
    *,
    actor: str = "user",
    intent: str = "request",
    action: str = "create",
    object: str = "",
    topic: str = "",
    modifiers: Optional[Sequence[str]] = None,
    constraints: Optional[Sequence[str]] = None,
    negations: Optional[Sequence[str]] = None,
    conditions: Optional[Sequence[str]] = None,
    time: Optional[str] = None,
    priority: str = "normal",
) -> BrainMemory:
    resolved_modifiers = [str(item) for item in (modifiers or []) if str(item)]
    resolved_constraints = [str(item) for item in (constraints or []) if str(item)]
    resolved_negations = [str(item) for item in (negations or []) if str(item)]
    resolved_conditions = [str(item) for item in (conditions or []) if str(item)]
    relations: List[Dict[str, str]] = []
    if object and topic:
        relations.append({"type": "about", "source": object, "target": topic})
    for modifier in resolved_modifiers:
        if object:
            relations.append({"type": "has_modifier", "source": object, "target": modifier})
    for negation in resolved_negations:
        if object:
            relations.append({"type": "negates", "source": object, "target": negation})
    for constraint in resolved_constraints:
        if object:
            relations.append({"type": "constraint", "source": object, "target": constraint})
    for condition in resolved_conditions:
        if object:
            relations.append({"type": "condition", "source": object, "target": condition})
    if time:
        relations.append({"type": "time", "source": object or action or "request", "target": time})
    return BrainCodec.resolve_memory(
        BrainMemory(
            actor=actor,
            intent=intent,
            action=action,
            object=object,
            topic=topic,
            modifiers=resolved_modifiers,
            constraints=resolved_constraints,
            negations=resolved_negations,
            conditions=resolved_conditions,
            relations=relations,
            time=time,
            priority=priority,
        )
    )


def run_reasoning_benchmark(
    report_name: str = "brain_native_reasoning_50",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, object]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = resolved_output_dir / f"{report_name}.state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    runtime = build_reasoning_runtime(state_dir)
    engine = BrainNativeReasoningEngine(runtime)

    seed_memories = _build_seed_memories()
    seed_summary = seed_reasoning_patterns(runtime, seed_memories)

    cases = _build_reasoning_cases(engine)
    if len(cases) != 50:
        raise ValueError(f"Reasoning benchmark must contain exactly 50 cases, found {len(cases)}.")

    results: List[Dict[str, object]] = []
    structure_preserved_count = 0
    slot_correct_count = 0
    pattern_reuse_count = 0
    operation_counts: Dict[str, int] = {}
    encoding_counts: Dict[str, int] = {}
    examples: Dict[str, Dict[str, object]] = {}

    for index, case in enumerate(cases, start=1):
        operation_counts[case.operation] = operation_counts.get(case.operation, 0) + 1
        output = engine.reason(
            case.operation,
            case.inputs,
            current_sequence=index,
            **case.parameters,
        )
        structure_preserved = _structure_preserved(runtime, output, index)
        slot_correct = _slot_expectations_met(output, case.expected_paths)
        if case.expected_encoding_types:
            slot_correct = slot_correct and output.encoding_type in case.expected_encoding_types
        if case.expect_pattern_reuse is not None:
            slot_correct = slot_correct and (_has_pattern_reuse(output) == case.expect_pattern_reuse)
        pattern_reused = _has_pattern_reuse(output)

        if structure_preserved:
            structure_preserved_count += 1
        if slot_correct:
            slot_correct_count += 1
        if pattern_reused:
            pattern_reuse_count += 1

        encoding_counts[output.encoding_type] = encoding_counts.get(output.encoding_type, 0) + 1
        if case.operation not in examples:
            examples[case.operation] = {
                "input": [item.to_dict() for item in case.inputs],
                "parameters": case.parameters,
                "output": output.to_dict(),
            }
        results.append(
            {
                "index": index,
                "case_id": case.case_id,
                "operation": case.operation,
                "structure_preserved": structure_preserved,
                "slot_correct": slot_correct,
                "pattern_reused": pattern_reused,
                "output": output.to_dict(),
            }
        )

    runtime.token_registry.save()

    aggregate = {
        "case_count": len(cases),
        "structure_preservation_count": structure_preserved_count,
        "structure_preservation_rate": round(structure_preserved_count / len(cases), 4),
        "slot_correct_count": slot_correct_count,
        "slot_correctness_rate": round(slot_correct_count / len(cases), 4),
        "pattern_reuse_count": pattern_reuse_count,
        "pattern_reuse_rate": round(pattern_reuse_count / len(cases), 4),
        "operation_counts": {key: operation_counts[key] for key in sorted(operation_counts)},
        "encoding_counts": {key: encoding_counts[key] for key in sorted(encoding_counts)},
    }

    report = {
        "report_name": report_name,
        "seed_summary": seed_summary,
        "aggregate": aggregate,
        "examples": examples,
        "results": results,
    }
    json_path = resolved_output_dir / f"{report_name}.json"
    markdown_path = resolved_output_dir / f"{report_name}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_build_markdown_report(report), encoding="utf-8")
    return {
        "report": report,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "state_dir": str(state_dir),
    }


def _build_seed_memories() -> List[BrainMemory]:
    seeds: List[BrainMemory] = []
    for topic in ("apples", "oranges", "pears", "strawberries"):
        for modifier in ("funny", "casual", "cozy"):
            seeds.append(make_structured_memory(object="game", topic=topic, modifiers=[modifier]))
    for topic in ("compression", "memory", "context", "semantic_context"):
        seeds.append(make_structured_memory(object="website", topic=topic))
    for topic, modifier in (("brain", "horror"), ("memory", "educational"), ("context", "cozy")):
        seeds.append(make_structured_memory(object="app", topic=topic, modifiers=[modifier]))
        seeds.append(make_structured_memory(object="app", topic=topic, modifiers=[modifier], constraints=["low_hardware"]))
    seeds.append(make_structured_memory(action="modify", object="game", topic="apples", modifiers=["funny"]))
    seeds.append(make_structured_memory(action="compare", object="difference", topic="topic"))
    return seeds


def _build_reasoning_cases(engine: BrainNativeReasoningEngine) -> List[ReasoningCase]:
    sequence = 1

    def make_input(memory: BrainMemory) -> BrainInputV1:
        nonlocal sequence
        base = engine.runtime.brain_input_codec.build_brain_input(
            memory=memory,
            encoding_type="bitpacked",
            primitive_ids=[],
        )
        native = engine.normalize(base, current_sequence=sequence)
        sequence += 1
        return native

    cases: List[ReasoningCase] = []

    transform_specs = [
        ("transform_topic_oranges", make_structured_memory(object="game", topic="apples", modifiers=["funny"]), {"updates": {"topic": "oranges"}}, {"topic": "oranges", "modifiers[0]": "funny"}),
        ("transform_topic_context", make_structured_memory(object="website", topic="compression"), {"updates": {"topic": "context"}}, {"topic": "context", "object": "website"}),
        ("transform_modifier_cozy", make_structured_memory(object="game", topic="pears", modifiers=["casual"]), {"updates": {"modifiers[0]": "cozy"}}, {"modifiers[0]": "cozy", "topic": "pears"}),
        ("transform_object_app", make_structured_memory(object="game", topic="apples", modifiers=["funny"]), {"updates": {"object": "app"}}, {"object": "app", "topic": "apples"}),
        ("transform_time_now", make_structured_memory(object="website", topic="memory"), {"updates": {"time": "now"}}, {"time": "now", "topic": "memory"}),
        ("transform_priority_high", make_structured_memory(object="app", topic="brain", modifiers=["horror"]), {"updates": {"priority": "high"}}, {"priority": "high", "topic": "brain"}),
        ("transform_constraint", make_structured_memory(object="app", topic="memory", modifiers=["educational"]), {"updates": {"constraints[0]": "low_hardware"}}, {"constraints[0]": "low_hardware"}),
        ("transform_action_modify", make_structured_memory(object="game", topic="oranges", modifiers=["funny"]), {"updates": {"action": "modify"}}, {"action": "modify", "topic": "oranges"}),
        ("transform_topic_strawberries", make_structured_memory(object="game", topic="pears", modifiers=["cozy"]), {"updates": {"topic": "strawberries"}}, {"topic": "strawberries", "modifiers[0]": "cozy"}),
        ("transform_modifier_funny", make_structured_memory(object="app", topic="context", modifiers=["cozy"]), {"updates": {"modifiers[0]": "funny"}}, {"modifiers[0]": "funny", "topic": "context"}),
    ]
    for case_id, memory, parameters, expected in transform_specs:
        cases.append(
            ReasoningCase(
                case_id=case_id,
                operation="transform",
                inputs=(make_input(memory),),
                parameters=parameters,
                expected_paths=expected,
                expect_pattern_reuse=True,
            )
        )

    compare_specs = [
        ("compare_topic", make_structured_memory(object="game", topic="apples", modifiers=["funny"]), make_structured_memory(object="game", topic="oranges", modifiers=["funny"]), {"topic": "topic", "action": "compare", "relations[0].target": "apples", "relations[1].target": "oranges"}),
        ("compare_modifier", make_structured_memory(object="game", topic="apples", modifiers=["funny"]), make_structured_memory(object="game", topic="apples", modifiers=["casual"]), {"topic": "modifiers[0]", "relations[0].target": "funny", "relations[1].target": "casual"}),
        ("compare_object", make_structured_memory(object="game", topic="apples"), make_structured_memory(object="app", topic="apples"), {"topic": "object", "relations[0].target": "game", "relations[1].target": "app"}),
        ("compare_constraint", make_structured_memory(object="app", topic="brain", constraints=["low_hardware"]), make_structured_memory(object="app", topic="brain"), {"topic": "constraints[0]", "relations[0].target": "low_hardware", "relations[1].target": "empty"}),
        ("compare_priority", make_structured_memory(object="website", topic="memory"), make_structured_memory(object="website", topic="memory", priority="high"), {"topic": "priority", "relations[0].target": "normal", "relations[1].target": "high"}),
        ("compare_time", make_structured_memory(object="website", topic="context", time="now"), make_structured_memory(object="website", topic="context", time="later"), {"topic": "time", "relations[0].target": "now", "relations[1].target": "later"}),
        ("compare_topic_web", make_structured_memory(object="website", topic="compression"), make_structured_memory(object="website", topic="semantic_context"), {"topic": "topic", "relations[0].target": "compression", "relations[1].target": "semantic_context"}),
        ("compare_modifier_app", make_structured_memory(object="app", topic="brain", modifiers=["horror"]), make_structured_memory(object="app", topic="brain", modifiers=["educational"]), {"topic": "modifiers[0]", "relations[0].target": "horror", "relations[1].target": "educational"}),
        ("compare_action", make_structured_memory(object="game", topic="apples"), make_structured_memory(action="modify", object="game", topic="apples"), {"topic": "action", "relations[0].target": "create", "relations[1].target": "modify"}),
        ("compare_topic_pears", make_structured_memory(object="game", topic="pears", modifiers=["cozy"]), make_structured_memory(object="game", topic="strawberries", modifiers=["cozy"]), {"topic": "topic", "relations[0].target": "pears", "relations[1].target": "strawberries"}),
    ]
    for case_id, left_memory, right_memory, expected in compare_specs:
        cases.append(
            ReasoningCase(
                case_id=case_id,
                operation="compare",
                inputs=(make_input(left_memory), make_input(right_memory)),
                parameters={},
                expected_paths=expected,
            )
        )

    merge_specs = [
        ("merge_fill_topic", make_structured_memory(object="game", topic="", modifiers=["funny"]), make_structured_memory(object="game", topic="apples"), {"topic": "apples", "modifiers[0]": "funny"}),
        ("merge_fill_modifier", make_structured_memory(object="app", topic="brain"), make_structured_memory(object="app", topic="brain", modifiers=["horror"]), {"topic": "brain", "modifiers[0]": "horror"}),
        ("merge_constraint", make_structured_memory(object="website", topic="compression"), make_structured_memory(object="website", topic="compression", constraints=["low_hardware"]), {"constraints[0]": "low_hardware", "topic": "compression"}),
        ("merge_condition", make_structured_memory(object="website", topic="context"), make_structured_memory(object="website", topic="context", conditions=["unless_requested"]), {"conditions[0]": "unless_requested"}),
        ("merge_time", make_structured_memory(object="website", topic="memory"), make_structured_memory(object="website", topic="memory", time="now"), {"time": "now", "topic": "memory"}),
        ("merge_modifiers_union", make_structured_memory(object="game", topic="apples", modifiers=["funny"]), make_structured_memory(object="game", topic="apples", modifiers=["casual"]), {"modifiers[0]": "funny", "modifiers[1]": "casual"}),
        ("merge_constraints_union", make_structured_memory(object="app", topic="memory", constraints=["low_hardware"]), make_structured_memory(object="app", topic="memory", constraints=["offline_mode"]), {"constraints[0]": "low_hardware", "constraints[1]": "offline_mode"}),
        ("merge_negation_preserve", make_structured_memory(object="app", topic="brain", negations=["educational"]), make_structured_memory(object="app", topic="brain", modifiers=["horror"]), {"negations[0]": "educational", "modifiers[0]": "horror"}),
        ("merge_topic_from_secondary", make_structured_memory(object="website", topic=""), make_structured_memory(object="website", topic="semantic_context"), {"topic": "semantic_context"}),
        ("merge_action_fill", make_structured_memory(action="", object="game", topic="oranges"), make_structured_memory(action="modify", object="game", topic="oranges"), {"action": "modify", "topic": "oranges"}),
    ]
    for case_id, left_memory, right_memory, expected in merge_specs:
        cases.append(
            ReasoningCase(
                case_id=case_id,
                operation="merge",
                inputs=(make_input(left_memory), make_input(right_memory)),
                parameters={},
                expected_paths=expected,
                expect_pattern_reuse=True,
            )
        )

    negate_specs = [
        ("negate_funny_apply", make_structured_memory(object="game", topic="apples", modifiers=["funny"]), {"target": "funny", "apply": True}, {"negations[0]": "funny"}),
        ("negate_funny_remove", make_structured_memory(object="game", topic="apples", negations=["funny"]), {"target": "funny", "apply": False}, {"negations": []}),
        ("constraint_low_hardware_apply", make_structured_memory(object="app", topic="brain"), {"constraint": "low_hardware", "apply": True}, {"constraints[0]": "low_hardware"}),
        ("constraint_low_hardware_remove", make_structured_memory(object="app", topic="brain", constraints=["low_hardware"]), {"constraint": "low_hardware", "apply": False}, {"constraints": []}),
        ("negate_educational_apply", make_structured_memory(object="app", topic="memory", modifiers=["educational"]), {"target": "educational", "apply": True}, {"negations[0]": "educational"}),
        ("negate_educational_remove", make_structured_memory(object="app", topic="memory", negations=["educational"]), {"target": "educational", "apply": False}, {"negations": []}),
        ("constraint_offline_apply", make_structured_memory(object="website", topic="context"), {"constraint": "offline_mode", "apply": True}, {"constraints[0]": "offline_mode"}),
        ("constraint_offline_remove", make_structured_memory(object="website", topic="context", constraints=["offline_mode"]), {"constraint": "offline_mode", "apply": False}, {"constraints": []}),
        ("negate_cozy_apply", make_structured_memory(object="game", topic="pears", modifiers=["cozy"]), {"target": "cozy", "apply": True}, {"negations[0]": "cozy"}),
        ("negate_cozy_remove", make_structured_memory(object="game", topic="pears", negations=["cozy"]), {"target": "cozy", "apply": False}, {"negations": []}),
    ]
    for case_id, memory, parameters, expected in negate_specs:
        cases.append(
            ReasoningCase(
                case_id=case_id,
                operation="negate",
                inputs=(make_input(memory),),
                parameters=parameters,
                expected_paths=expected,
                expect_pattern_reuse=True,
            )
        )

    generalize_specs = [
        ("generalize_game_apples", make_structured_memory(object="game", topic="apples", modifiers=["funny"]), {"object": "game", "topic": "apples"}),
        ("generalize_game_oranges", make_structured_memory(object="game", topic="oranges", modifiers=["casual"]), {"object": "game", "topic": "oranges"}),
        ("generalize_game_pears", make_structured_memory(object="game", topic="pears", modifiers=["cozy"]), {"object": "game", "topic": "pears"}),
        ("generalize_website_context", make_structured_memory(object="website", topic="context"), {"object": "website", "topic": "context"}),
        ("generalize_app_brain", make_structured_memory(object="app", topic="brain", modifiers=["horror"]), {"object": "app", "topic": "brain"}),
    ]
    for case_id, memory, expected in generalize_specs:
        cases.append(
            ReasoningCase(
                case_id=case_id,
                operation="generalize",
                inputs=(make_input(memory),),
                parameters={},
                expected_paths=expected,
                expected_encoding_types=("evolved_pattern", "evolved_subpatterns", "hierarchical_pattern", "hierarchical_subpatterns"),
                expect_pattern_reuse=True,
            )
        )

    specialize_specs = [
        ("specialize_topic_apples", make_structured_memory(object="game", topic="", modifiers=["funny"]), {"fills": {"topic": "apples"}}, {"topic": "apples", "modifiers[0]": "funny"}),
        ("specialize_modifier_funny", make_structured_memory(object="game", topic="oranges"), {"fills": {"modifiers[0]": "funny"}}, {"modifiers[0]": "funny", "topic": "oranges"}),
        ("specialize_topic_compression", make_structured_memory(object="website", topic=""), {"fills": {"topic": "compression"}}, {"topic": "compression", "object": "website"}),
        ("specialize_constraint", make_structured_memory(object="app", topic="brain"), {"fills": {"constraints[0]": "low_hardware"}}, {"constraints[0]": "low_hardware"}),
        ("specialize_time", make_structured_memory(object="website", topic="memory"), {"fills": {"time": "now"}}, {"time": "now", "topic": "memory"}),
    ]
    for case_id, memory, parameters, expected in specialize_specs:
        cases.append(
            ReasoningCase(
                case_id=case_id,
                operation="specialize",
                inputs=(make_input(memory),),
                parameters=parameters,
                expected_paths=expected,
                expect_pattern_reuse=True,
            )
        )

    return cases


def _structure_preserved(runtime, output: BrainInputV1, current_sequence: int) -> bool:
    sequence = runtime.brain_input_codec.brain_input_to_sequence(
        output,
        strategy="pattern_first",
        current_sequence=current_sequence,
    )
    sequence_roundtrip = runtime.brain_input_codec.sequence_to_brain_input(
        sequence,
        strategy="pattern_first",
    )
    grid_payload = runtime.brain_input_codec.brain_input_to_grid(
        output,
        strategy="pattern_first",
        current_sequence=current_sequence,
    )
    grid_roundtrip = runtime.brain_input_codec.grid_to_brain_input(grid_payload)
    expected_bundle = runtime.codec.encode_memory_bundle(runtime.brain_input_codec.brain_input_to_memory(output))
    sequence_bundle = runtime.codec.encode_memory_bundle(runtime.brain_input_codec.brain_input_to_memory(sequence_roundtrip))
    grid_bundle = runtime.codec.encode_memory_bundle(runtime.brain_input_codec.brain_input_to_memory(grid_roundtrip))
    return expected_bundle == sequence_bundle == grid_bundle


def _slot_expectations_met(output: BrainInputV1, expected_paths: Dict[str, object]) -> bool:
    memory = BrainCodec.resolve_memory(_brain_input_to_memory(output))
    for path, expected in expected_paths.items():
        actual = _read_path(memory, str(path))
        if actual != expected:
            return False
    return True


def _brain_input_to_memory(brain_input: BrainInputV1) -> BrainMemory:
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


def _read_path(memory: BrainMemory, path: str) -> object:
    if path in {"actor", "intent", "action", "object", "topic", "time", "priority"}:
        return getattr(memory, path)
    if path in {"modifiers", "constraints", "negations", "conditions"}:
        return list(getattr(memory, path))
    if path.startswith("relations["):
        relation_index = int(path.split("[", 1)[1].split("]", 1)[0])
        field_name = path.split(".", 1)[1]
        if relation_index >= len(memory.relations):
            return ""
        return memory.relations[relation_index].get(field_name, "")
    if "[" in path and path.endswith("]"):
        field_name, _, suffix = path.partition("[")
        index = int(suffix[:-1])
        values = list(getattr(memory, field_name))
        return values[index] if index < len(values) else ""
    raise ValueError(f"Unsupported expectation path: {path}")


def _has_pattern_reuse(output: BrainInputV1) -> bool:
    return output.pattern_id is not None or bool(output.subpattern_refs)


def _build_markdown_report(report: Dict[str, object]) -> str:
    aggregate = report.get("aggregate", {})
    seed_summary = report.get("seed_summary", {})
    examples = report.get("examples", {})
    if not isinstance(aggregate, dict):
        aggregate = {}
    if not isinstance(seed_summary, dict):
        seed_summary = {}
    if not isinstance(examples, dict):
        examples = {}

    lines = [
        "# BrAIn Native Reasoning Benchmark",
        "",
        "This report evaluates deterministic reasoning directly on BrainInputV1 without any text parsing in the reasoning loop.",
        "",
        "## Summary",
        "",
        f"- Report name: `{report.get('report_name', 'brain_native_reasoning_50')}`",
        f"- Case count: `{aggregate.get('case_count', 0)}`",
        f"- Structure preservation rate: `{aggregate.get('structure_preservation_rate', 0.0)}`",
        f"- Slot correctness rate: `{aggregate.get('slot_correctness_rate', 0.0)}`",
        f"- Pattern reuse rate: `{aggregate.get('pattern_reuse_rate', 0.0)}`",
        f"- Operation counts: `{aggregate.get('operation_counts', {})}`",
        f"- Output encoding counts: `{aggregate.get('encoding_counts', {})}`",
        "",
        "## Seeded Memory",
        "",
        f"- Flat pattern count: `{seed_summary.get('pattern_count', 0)}`",
        f"- Hierarchical pattern count: `{seed_summary.get('hierarchical_pattern_count', 0)}`",
        f"- Hierarchical subpattern count: `{seed_summary.get('hierarchical_subpattern_count', 0)}`",
        f"- Generalized pattern count: `{seed_summary.get('generalized_pattern_count', 0)}`",
        "",
        "## Example Transformations",
        "",
    ]

    for operation in ("transform", "compare", "merge", "negate", "generalize", "specialize"):
        example = examples.get(operation, {})
        if not isinstance(example, dict):
            continue
        lines.extend(
            [
                f"### {operation.title()}",
                "",
                "```json",
                json.dumps(example, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="brain_reasoning_benchmark",
        description="Run deterministic BrAIn-native reasoning benchmark cases.",
    )
    parser.add_argument(
        "--report-name",
        default="brain_native_reasoning_50",
        help="Base name for the saved reasoning benchmark files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved reasoning benchmark artifacts.",
    )
    args = parser.parse_args()
    result = run_reasoning_benchmark(report_name=args.report_name, output_dir=args.output_dir)
    print("=" * 72)
    print("BRAIN NATIVE REASONING BENCHMARK")
    print(f"JSON report: {result['json_path']}")
    print(f"Markdown summary: {result['markdown_path']}")
    print(f"State dir: {result['state_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
