from __future__ import annotations

# Historical harnesses run with the maintained core on the import path.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]/"core"))

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
    BrainReasoningRuntime,
    build_reasoning_runtime,
)


@dataclass(frozen=True)
class AdaptiveReasoningCase:
    case_id: str
    operation: str
    inputs: Tuple[BrainMemory, ...]
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


def _memory_to_brain_input(runtime: BrainReasoningRuntime, memory: BrainMemory) -> BrainInputV1:
    return runtime.brain_input_codec.build_brain_input(
        memory=memory,
        encoding_type="bitpacked",
        primitive_ids=[],
    )


def _store_bundle(
    runtime: BrainReasoningRuntime,
    memory_id: int,
    memory_bundle: Dict[str, List[List[List[int]]]],
) -> None:
    runtime.pattern_memory_store.add_memory(memory_id, memory_bundle)
    runtime.hierarchical_memory_store.add_memory(memory_id, memory_bundle)


def _build_adaptive_cases() -> List[AdaptiveReasoningCase]:
    game_topics = [
        ("apples", "oranges", "pears"),
        ("bananas", "grapes", "lemons"),
        ("peaches", "plums", "berries"),
        ("mangoes", "kiwis", "melons"),
        ("cherries", "figs", "dates"),
    ]
    web_topics = [
        ("compression", "context"),
        ("memory", "semantic_context"),
        ("retrieval", "structure"),
        ("reasoning", "patterns"),
        ("benchmarks", "reuse"),
    ]
    game_modifiers = [
        ("funny", "casual", "cozy"),
        ("cozy", "funny", "casual"),
        ("casual", "cozy", "funny"),
        ("funny", "cozy", "casual"),
        ("casual", "funny", "cozy"),
    ]
    app_topics = ["brain", "memory", "context", "compression", "planning"]
    app_modifiers = ["horror", "educational", "cozy", "serious", "guided"]

    cases: List[AdaptiveReasoningCase] = []
    for round_index in range(5):
        topic_a, topic_b, topic_c = game_topics[round_index]
        web_a, web_b = web_topics[round_index]
        modifier_a, modifier_b, modifier_c = game_modifiers[round_index]
        app_topic = app_topics[round_index]
        app_modifier = app_modifiers[round_index]
        prefix = f"r{round_index + 1:02d}"
        cases.extend(
            [
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_transform_topic",
                    operation="transform",
                    inputs=(make_structured_memory(object="game", topic=topic_a, modifiers=[modifier_a]),),
                    parameters={"updates": {"topic": topic_b}},
                    expected_paths={"topic": topic_b, "modifiers[0]": modifier_a},
                    expect_pattern_reuse=True,
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_transform_modifier",
                    operation="transform",
                    inputs=(make_structured_memory(object="game", topic=topic_c, modifiers=[modifier_b]),),
                    parameters={"updates": {"modifiers[0]": modifier_c}},
                    expected_paths={"topic": topic_c, "modifiers[0]": modifier_c},
                    expect_pattern_reuse=True,
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_transform_constraint",
                    operation="transform",
                    inputs=(make_structured_memory(object="website", topic=web_a),),
                    parameters={"updates": {"constraints[0]": "offline_mode"}},
                    expected_paths={"object": "website", "topic": web_a, "constraints[0]": "offline_mode"},
                    expect_pattern_reuse=True,
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_transform_priority",
                    operation="transform",
                    inputs=(make_structured_memory(object="app", topic=app_topic, modifiers=[app_modifier]),),
                    parameters={"updates": {"priority": "high"}},
                    expected_paths={"object": "app", "topic": app_topic, "priority": "high"},
                    expect_pattern_reuse=True,
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_compare_topic",
                    operation="compare",
                    inputs=(
                        make_structured_memory(object="game", topic=topic_a, modifiers=[modifier_a]),
                        make_structured_memory(object="game", topic=topic_b, modifiers=[modifier_a]),
                    ),
                    parameters={},
                    expected_paths={"action": "compare", "topic": "topic", "relations[0].target": topic_a, "relations[1].target": topic_b},
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_compare_modifier",
                    operation="compare",
                    inputs=(
                        make_structured_memory(object="game", topic=topic_a, modifiers=[modifier_a]),
                        make_structured_memory(object="game", topic=topic_a, modifiers=[modifier_b]),
                    ),
                    parameters={},
                    expected_paths={"action": "compare", "topic": "modifiers[0]", "relations[0].target": modifier_a, "relations[1].target": modifier_b},
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_compare_object",
                    operation="compare",
                    inputs=(
                        make_structured_memory(object="game", topic=topic_a),
                        make_structured_memory(object="website", topic=topic_a),
                    ),
                    parameters={},
                    expected_paths={"action": "compare", "topic": "object", "relations[0].target": "game", "relations[1].target": "website"},
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_compare_web_topic",
                    operation="compare",
                    inputs=(
                        make_structured_memory(object="website", topic=web_a),
                        make_structured_memory(object="website", topic=web_b),
                    ),
                    parameters={},
                    expected_paths={"action": "compare", "topic": "topic", "relations[0].target": web_a, "relations[1].target": web_b},
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_merge_topic",
                    operation="merge",
                    inputs=(
                        make_structured_memory(object="game", topic="", modifiers=[modifier_a]),
                        make_structured_memory(object="game", topic=topic_a),
                    ),
                    parameters={},
                    expected_paths={"object": "game", "topic": topic_a, "modifiers[0]": modifier_a},
                    expect_pattern_reuse=True,
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_merge_modifier",
                    operation="merge",
                    inputs=(
                        make_structured_memory(object="game", topic=topic_b),
                        make_structured_memory(object="game", topic=topic_b, modifiers=[modifier_a]),
                    ),
                    parameters={},
                    expected_paths={"object": "game", "topic": topic_b, "modifiers[0]": modifier_a},
                    expect_pattern_reuse=True,
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_merge_web_topic",
                    operation="merge",
                    inputs=(
                        make_structured_memory(object="website", topic="", modifiers=[modifier_a]),
                        make_structured_memory(object="website", topic=web_a),
                    ),
                    parameters={},
                    expected_paths={"object": "website", "topic": web_a, "modifiers[0]": modifier_a},
                    expect_pattern_reuse=True,
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_merge_modifier_union",
                    operation="merge",
                    inputs=(
                        make_structured_memory(object="game", topic=topic_a, modifiers=[modifier_a]),
                        make_structured_memory(object="game", topic=topic_a, modifiers=[modifier_b]),
                    ),
                    parameters={},
                    expected_paths={"object": "game", "topic": topic_a, "modifiers[0]": modifier_a, "modifiers[1]": modifier_b},
                    expect_pattern_reuse=True,
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_negate_modifier",
                    operation="negate",
                    inputs=(make_structured_memory(object="game", topic=topic_a, modifiers=[modifier_a]),),
                    parameters={"target": modifier_a, "apply": True},
                    expected_paths={"object": "game", "topic": topic_a, "negations[0]": modifier_a},
                    expect_pattern_reuse=True,
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_negate_constraint",
                    operation="negate",
                    inputs=(make_structured_memory(object="website", topic=web_a),),
                    parameters={"constraint": "offline_mode", "apply": True},
                    expected_paths={"object": "website", "topic": web_a, "constraints[0]": "offline_mode"},
                    expect_pattern_reuse=True,
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_negate_modifier_alt",
                    operation="negate",
                    inputs=(make_structured_memory(object="game", topic=topic_c, modifiers=[modifier_c]),),
                    parameters={"target": modifier_c, "apply": True},
                    expected_paths={"object": "game", "topic": topic_c, "negations[0]": modifier_c},
                    expect_pattern_reuse=True,
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_generalize_game",
                    operation="generalize",
                    inputs=(make_structured_memory(object="game", topic=topic_a, modifiers=[modifier_a]),),
                    parameters={},
                    expected_paths={"object": "game", "topic": topic_a, "modifiers[0]": modifier_a},
                    expected_encoding_types=("evolved_pattern", "evolved_subpatterns", "hierarchical_pattern", "hierarchical_subpatterns"),
                    expect_pattern_reuse=True,
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_generalize_website",
                    operation="generalize",
                    inputs=(make_structured_memory(object="website", topic=web_a),),
                    parameters={},
                    expected_paths={"object": "website", "topic": web_a},
                    expected_encoding_types=("evolved_pattern", "evolved_subpatterns", "hierarchical_pattern", "hierarchical_subpatterns"),
                    expect_pattern_reuse=True,
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_specialize_topic",
                    operation="specialize",
                    inputs=(make_structured_memory(object="game", topic="", modifiers=[modifier_a]),),
                    parameters={"fills": {"topic": topic_a}},
                    expected_paths={"object": "game", "topic": topic_a, "modifiers[0]": modifier_a},
                    expect_pattern_reuse=True,
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_specialize_modifier",
                    operation="specialize",
                    inputs=(make_structured_memory(object="game", topic=topic_b),),
                    parameters={"fills": {"modifiers[0]": modifier_a}},
                    expected_paths={"object": "game", "topic": topic_b, "modifiers[0]": modifier_a},
                    expect_pattern_reuse=True,
                ),
                AdaptiveReasoningCase(
                    case_id=f"{prefix}_specialize_web_topic",
                    operation="specialize",
                    inputs=(make_structured_memory(object="website", topic=""),),
                    parameters={"fills": {"topic": web_a}},
                    expected_paths={"object": "website", "topic": web_a},
                    expect_pattern_reuse=True,
                ),
            ]
        )
    return cases


def _registry_summary(
    runtime: BrainReasoningRuntime,
    current_sequence: int,
) -> Dict[str, object]:
    active_patterns = runtime.hierarchical_registry.all_active_patterns()
    active_subpatterns = runtime.hierarchical_registry.all_active_subpatterns()
    scored_patterns = [
        pattern
        for pattern in active_patterns
        if (pattern.success_count + pattern.failure_count) > 0
    ]
    compressed_patterns = [
        pattern
        for pattern in active_patterns
        if pattern.total_full_size_bytes > 0 or pattern.compression_efficiency > 0.0
    ]
    average_success_rate = round(
        sum(pattern.success_rate for pattern in scored_patterns) / max(1, len(scored_patterns)),
        4,
    ) if scored_patterns else 0.0
    average_compression_efficiency = round(
        sum(pattern.compression_efficiency for pattern in compressed_patterns) / max(1, len(compressed_patterns)),
        4,
    ) if compressed_patterns else 0.0
    average_temporal_score = round(
        sum(runtime.pattern_evolution.score_pattern(pattern, current_sequence=current_sequence).score for pattern in active_patterns)
        / max(1, len(active_patterns)),
        4,
    ) if active_patterns else 0.0
    return {
        "pattern_count": len(runtime.hierarchical_registry.patterns),
        "active_pattern_count": len(active_patterns),
        "subpattern_count": len(runtime.hierarchical_registry.subpatterns),
        "active_subpattern_count": len(active_subpatterns),
        "scored_pattern_count": len(scored_patterns),
        "average_success_rate": average_success_rate,
        "average_compression_efficiency": average_compression_efficiency,
        "average_temporal_score": average_temporal_score,
    }


def _aggregate_results(
    results: Sequence[Dict[str, object]],
    recommendation_counts: Dict[str, int],
    before_summary: Dict[str, object],
    after_summary: Dict[str, object],
) -> Dict[str, object]:
    early_window = list(results[:20])
    late_window = list(results[-20:])
    overall_success = [bool(item.get("success", False)) for item in results]
    early_reuse = [item.get("recommendation_mode", "full") != "full" for item in early_window]
    late_reuse = [item.get("recommendation_mode", "full") != "full" for item in late_window]
    early_compression = [float(item.get("compression_ratio", 1.0)) for item in early_window]
    late_compression = [float(item.get("compression_ratio", 1.0)) for item in late_window]
    return {
        "case_count": len(results),
        "recommendation_counts": {key: recommendation_counts.get(key, 0) for key in sorted(recommendation_counts)},
        "reasoning_consistency_rate": round(sum(1 for item in overall_success if item) / max(1, len(overall_success)), 4),
        "pattern_reuse_rate_early_20": round(sum(1 for item in early_reuse if item) / max(1, len(early_reuse)), 4),
        "pattern_reuse_rate_late_20": round(sum(1 for item in late_reuse if item) / max(1, len(late_reuse)), 4),
        "pattern_reuse_increase": round(
            (sum(1 for item in late_reuse if item) / max(1, len(late_reuse)))
            - (sum(1 for item in early_reuse if item) / max(1, len(early_reuse))),
            4,
        ),
        "average_compression_ratio_early_20": round(sum(early_compression) / max(1, len(early_compression)), 4),
        "average_compression_ratio_late_20": round(sum(late_compression) / max(1, len(late_compression)), 4),
        "compression_improvement": round(
            (sum(early_compression) / max(1, len(early_compression)))
            - (sum(late_compression) / max(1, len(late_compression))),
            4,
        ),
        "patterns_before": before_summary.get("active_pattern_count", 0),
        "patterns_after": after_summary.get("active_pattern_count", 0),
        "subpatterns_before": before_summary.get("active_subpattern_count", 0),
        "subpatterns_after": after_summary.get("active_subpattern_count", 0),
        "average_success_rate_before": before_summary.get("average_success_rate", 0.0),
        "average_success_rate_after": after_summary.get("average_success_rate", 0.0),
        "average_compression_efficiency_before": before_summary.get("average_compression_efficiency", 0.0),
        "average_compression_efficiency_after": after_summary.get("average_compression_efficiency", 0.0),
        "average_temporal_score_before": before_summary.get("average_temporal_score", 0.0),
        "average_temporal_score_after": after_summary.get("average_temporal_score", 0.0),
    }


def run_adaptive_pattern_benchmark(
    report_name: str = "brain_adaptive_pattern_evolution_100",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, object]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = resolved_output_dir / f"{report_name}.state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    runtime = build_reasoning_runtime(state_dir)
    engine = BrainNativeReasoningEngine(runtime)
    cases = _build_adaptive_cases()
    if len(cases) != 100:
        raise ValueError(f"Adaptive pattern benchmark must contain exactly 100 cases, found {len(cases)}.")

    before_summary = _registry_summary(runtime, current_sequence=1)
    results: List[Dict[str, object]] = []
    evolution_logs: List[Dict[str, object]] = []
    recommendation_counts: Dict[str, int] = {"pattern": 0, "subpatterns": 0, "full": 0}
    selected_pattern_usage: Dict[str, int] = {}

    for sequence, case in enumerate(cases, start=1):
        primary_bundle = runtime.codec.encode_memory_bundle(case.inputs[0])
        recommendation = runtime.pattern_evolution.recommend_pattern(
            memory_bundle=primary_bundle,
            current_sequence=sequence,
            include_subpatterns=True,
            update_usage=True,
        )
        selected_mode = recommendation.mode if recommendation is not None else "full"
        recommendation_counts[selected_mode] = recommendation_counts.get(selected_mode, 0) + 1
        if recommendation is not None and recommendation.pattern_id is not None:
            key = str(recommendation.pattern_id)
            selected_pattern_usage[key] = selected_pattern_usage.get(key, 0) + 1

        output = engine.reason(
            case.operation,
            inputs=tuple(_memory_to_brain_input(runtime, memory) for memory in case.inputs),
            current_sequence=sequence,
            **case.parameters,
        )
        structure_preserved = _structure_preserved(runtime, output, sequence)
        reasoning_correct = _slot_expectations_met(output, case.expected_paths)
        if case.expected_encoding_types:
            reasoning_correct = reasoning_correct and output.encoding_type in case.expected_encoding_types
        if case.expect_pattern_reuse is not None:
            reasoning_correct = reasoning_correct and (_has_pattern_reuse(output) == case.expect_pattern_reuse)
        success = structure_preserved and reasoning_correct
        runtime.pattern_evolution.record_reasoning_outcome(
            decision=recommendation,
            success=success,
            current_sequence=sequence,
        )

        for offset, memory in enumerate(case.inputs, start=1):
            _store_bundle(runtime, memory_id=(sequence * 10) + offset, memory_bundle=runtime.codec.encode_memory_bundle(memory))
        _store_bundle(
            runtime,
            memory_id=(sequence * 10) + 9,
            memory_bundle=runtime.codec.encode_memory_bundle(runtime.brain_input_codec.brain_input_to_memory(output)),
        )

        if sequence % 10 == 0:
            evolution_logs.append(
                {
                    "sequence": sequence,
                    **runtime.pattern_evolution.run_adaptive_cycle(current_sequence=sequence, min_cooccurrence=3),
                }
            )

        full_size_bytes = len(runtime.hierarchical_codec.full_bundle_bytes(primary_bundle))
        selected_size_bytes = recommendation.encoded_size_bytes if recommendation is not None else full_size_bytes
        results.append(
            {
                "sequence": sequence,
                "case_id": case.case_id,
                "operation": case.operation,
                "recommendation_mode": selected_mode,
                "recommended_pattern_id": recommendation.pattern_id if recommendation is not None else None,
                "recommended_subpattern_ids": list(recommendation.subpattern_ids) if recommendation is not None else [],
                "recommendation_score": recommendation.reinforcement_score if recommendation is not None else 0.0,
                "full_size_bytes": full_size_bytes,
                "selected_size_bytes": selected_size_bytes,
                "compression_ratio": round(selected_size_bytes / max(1, full_size_bytes), 4),
                "structure_preserved": structure_preserved,
                "reasoning_correct": reasoning_correct,
                "success": success,
                "result": output.to_dict(),
            }
        )

    runtime.pattern_evolution.refresh_registry_temporal_scores(current_sequence=len(cases))
    runtime.hierarchical_registry.save()
    runtime.token_registry.save()
    after_summary = _registry_summary(runtime, current_sequence=len(cases))
    aggregate = _aggregate_results(results, recommendation_counts, before_summary, after_summary)
    report = {
        "report_name": report_name,
        "before_summary": before_summary,
        "after_summary": after_summary,
        "aggregate": aggregate,
        "selected_pattern_usage": {key: selected_pattern_usage[key] for key in sorted(selected_pattern_usage, key=lambda item: (-selected_pattern_usage[item], int(item)))},
        "evolution_logs": evolution_logs,
        "results": results,
    }

    json_path = resolved_output_dir / f"{report_name}.json"
    markdown_path = resolved_output_dir / f"{report_name}.md"
    evolution_log_path = resolved_output_dir / f"{report_name}.evolution_logs.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_build_markdown_report(report), encoding="utf-8")
    evolution_log_path.write_text(json.dumps(evolution_logs, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "report": report,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "evolution_log_path": str(evolution_log_path),
        "registry_path": str(state_dir / "brain_hierarchical_patterns.json"),
        "state_dir": str(state_dir),
    }


def _structure_preserved(
    runtime: BrainReasoningRuntime,
    output: BrainInputV1,
    current_sequence: int,
) -> bool:
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
    before_summary = report.get("before_summary", {})
    after_summary = report.get("after_summary", {})
    evolution_logs = report.get("evolution_logs", [])
    results = report.get("results", [])
    if not isinstance(aggregate, dict):
        aggregate = {}
    if not isinstance(before_summary, dict):
        before_summary = {}
    if not isinstance(after_summary, dict):
        after_summary = {}
    if not isinstance(evolution_logs, list):
        evolution_logs = []
    if not isinstance(results, list):
        results = []

    lines = [
        "# Adaptive Pattern Evolution Benchmark",
        "",
        "This benchmark evaluates BrAIn as a self-improving semantic system operating on structure only.",
        "",
        "## Summary",
        "",
        f"- Report name: `{report.get('report_name', 'brain_adaptive_pattern_evolution_100')}`",
        f"- Case count: `{aggregate.get('case_count', 0)}`",
        f"- Reasoning consistency rate: `{aggregate.get('reasoning_consistency_rate', 0.0)}`",
        f"- Pattern reuse rate, first 20 inputs: `{aggregate.get('pattern_reuse_rate_early_20', 0.0)}`",
        f"- Pattern reuse rate, last 20 inputs: `{aggregate.get('pattern_reuse_rate_late_20', 0.0)}`",
        f"- Pattern reuse increase: `{aggregate.get('pattern_reuse_increase', 0.0)}`",
        f"- Average compression ratio, first 20 inputs: `{aggregate.get('average_compression_ratio_early_20', 0.0)}`",
        f"- Average compression ratio, last 20 inputs: `{aggregate.get('average_compression_ratio_late_20', 0.0)}`",
        f"- Compression improvement: `{aggregate.get('compression_improvement', 0.0)}`",
        f"- Recommendation counts: `{aggregate.get('recommendation_counts', {})}`",
        "",
        "## Registry Growth",
        "",
        f"- Active patterns before: `{before_summary.get('active_pattern_count', 0)}`",
        f"- Active patterns after: `{after_summary.get('active_pattern_count', 0)}`",
        f"- Active subpatterns before: `{before_summary.get('active_subpattern_count', 0)}`",
        f"- Active subpatterns after: `{after_summary.get('active_subpattern_count', 0)}`",
        f"- Average success rate before: `{before_summary.get('average_success_rate', 0.0)}`",
        f"- Average success rate after: `{after_summary.get('average_success_rate', 0.0)}`",
        f"- Average compression efficiency before: `{before_summary.get('average_compression_efficiency', 0.0)}`",
        f"- Average compression efficiency after: `{after_summary.get('average_compression_efficiency', 0.0)}`",
        f"- Average temporal score before: `{before_summary.get('average_temporal_score', 0.0)}`",
        f"- Average temporal score after: `{after_summary.get('average_temporal_score', 0.0)}`",
        "",
        "## Learning Behavior",
        "",
        "- BrAIn scores patterns using usage, validated success, compression efficiency, reuse frequency, and temporal decay.",
        "- Similar patterns merge into broader reusable structures, frequent subpattern pairs get promoted, and low-value patterns are pruned.",
        "- When broad generalized patterns underperform, they are split back into smaller reusable structures.",
        "- Recommendation stays deterministic: the best structural match is suggested first and fallback is used only when the learned registry cannot beat full encoding.",
        "",
        "## Evolution Log Excerpts",
        "",
    ]

    for log in evolution_logs[:5]:
        if not isinstance(log, dict):
            continue
        lines.extend([f"### Sequence {log.get('sequence', 0)}", "", "```json", json.dumps(log, indent=2, sort_keys=True), "```", ""])

    lines.extend(["## Example Results", ""])
    for result in list(results[:3]) + list(results[-3:]):
        if not isinstance(result, dict):
            continue
        lines.extend(
            [
                f"### {result.get('case_id', 'case')}",
                "",
                f"- Operation: `{result.get('operation', '')}`",
                f"- Recommendation mode: `{result.get('recommendation_mode', '')}`",
                f"- Compression ratio: `{result.get('compression_ratio', 1.0)}`",
                f"- Success: `{result.get('success', False)}`",
                "",
                "```json",
                json.dumps(result.get("result", {}), indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="brain_adaptive_pattern_benchmark",
        description="Run adaptive pattern evolution benchmark cases.",
    )
    parser.add_argument(
        "--report-name",
        default="brain_adaptive_pattern_evolution_100",
        help="Base name for the saved adaptive pattern benchmark files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved adaptive pattern benchmark artifacts.",
    )
    args = parser.parse_args()
    result = run_adaptive_pattern_benchmark(report_name=args.report_name, output_dir=args.output_dir)
    print("=" * 72)
    print("ADAPTIVE PATTERN EVOLUTION BENCHMARK")
    print(f"JSON report: {result['json_path']}")
    print(f"Markdown summary: {result['markdown_path']}")
    print(f"Evolution logs: {result['evolution_log_path']}")
    print(f"Registry: {result['registry_path']}")
    print(f"State dir: {result['state_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
