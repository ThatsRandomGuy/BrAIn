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

from brain_assisted_pipeline import BrainAssistedPipelineResult, BrainAssistedReasoningPipeline
from brain_codec import BrainCodec, BrainMemory
from brain_input_v1 import BrainInputV1
from brain_reasoning_engine import DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class PipelineScenario:
    case_id: str
    operation: str
    user_input_text: str
    supporting_texts: Tuple[str, ...]
    reasoning_params: Dict[str, object]
    expected_paths: Dict[str, object]
    expected_trace_operations: Tuple[str, ...]
    expected_trace_fragments: Tuple[str, ...]
    expected_text_fragments: Tuple[str, ...]
    expected_encoding_types: Tuple[str, ...] = ()
    expect_pattern_reuse: Optional[bool] = True


def run_pipeline_benchmark(
    report_name: str = "brain_assisted_pipeline_20",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, object]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    state_root = resolved_output_dir / f"{report_name}.state"
    if state_root.exists():
        shutil.rmtree(state_root)
    state_root.mkdir(parents=True, exist_ok=True)

    scenarios = _build_pipeline_scenarios()
    if len(scenarios) != 20:
        raise ValueError(f"Pipeline benchmark must contain exactly 20 scenarios, found {len(scenarios)}.")

    results: List[Dict[str, object]] = []
    operation_counts: Dict[str, int] = {}
    encoding_counts: Dict[str, int] = {}

    structure_preserved_count = 0
    brain_input_correct_count = 0
    trace_accuracy_count = 0
    llm_alignment_count = 0
    pattern_reuse_count = 0

    for scenario in scenarios:
        operation_counts[scenario.operation] = operation_counts.get(scenario.operation, 0) + 1
        pipeline = BrainAssistedReasoningPipeline(state_dir=state_root / scenario.case_id)
        result = pipeline.run(
            user_input_text=scenario.user_input_text,
            operation=scenario.operation,
            reasoning_params=scenario.reasoning_params,
            supporting_texts=scenario.supporting_texts,
            top_k=3,
            store_result=True,
        )

        structure_preserved = _structure_preserved(pipeline, result.result_brain_input)
        brain_input_correct = _brain_input_correct(result.result_brain_input, scenario)
        trace_accurate = _trace_accurate(result, scenario)
        llm_aligned = _llm_alignment_correct(result, scenario)
        pattern_reused = _has_pattern_reuse(result.result_brain_input)

        if structure_preserved:
            structure_preserved_count += 1
        if brain_input_correct:
            brain_input_correct_count += 1
        if trace_accurate:
            trace_accuracy_count += 1
        if llm_aligned:
            llm_alignment_count += 1
        if pattern_reused:
            pattern_reuse_count += 1

        encoding_type = result.result_brain_input.encoding_type
        encoding_counts[encoding_type] = encoding_counts.get(encoding_type, 0) + 1

        results.append(
            {
                "case_id": scenario.case_id,
                "operation": scenario.operation,
                "user_input_text": scenario.user_input_text,
                "supporting_texts": list(scenario.supporting_texts),
                "reasoning_params": dict(scenario.reasoning_params),
                "structure_preserved": structure_preserved,
                "brain_input_correct": brain_input_correct,
                "trace_accurate": trace_accurate,
                "llm_aligned": llm_aligned,
                "pattern_reused": pattern_reused,
                "result": result.to_dict(),
            }
        )

    aggregate = {
        "case_count": len(scenarios),
        "structure_preservation_count": structure_preserved_count,
        "structure_preservation_rate": round(structure_preserved_count / len(scenarios), 4),
        "brain_input_correct_count": brain_input_correct_count,
        "brain_input_correct_rate": round(brain_input_correct_count / len(scenarios), 4),
        "trace_accuracy_count": trace_accuracy_count,
        "trace_accuracy_rate": round(trace_accuracy_count / len(scenarios), 4),
        "llm_alignment_count": llm_alignment_count,
        "llm_alignment_rate": round(llm_alignment_count / len(scenarios), 4),
        "pattern_reuse_count": pattern_reuse_count,
        "pattern_reuse_rate": round(pattern_reuse_count / len(scenarios), 4),
        "operation_counts": {key: operation_counts[key] for key in sorted(operation_counts)},
        "encoding_counts": {key: encoding_counts[key] for key in sorted(encoding_counts)},
    }

    report = {
        "report_name": report_name,
        "aggregate": aggregate,
        "scenarios": [_scenario_to_dict(scenario) for scenario in scenarios],
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
        "state_dir": str(state_root),
    }


def _build_pipeline_scenarios() -> List[PipelineScenario]:
    return [
        PipelineScenario(
            case_id="transform_topic_oranges",
            operation="transform",
            user_input_text="Create a funny game about apples.",
            supporting_texts=(),
            reasoning_params={"updates": {"topic": "oranges"}},
            expected_paths={"topic": "oranges", "modifiers[0]": "funny"},
            expected_trace_operations=("transform", "re_encode"),
            expected_trace_fragments=("topic", "oranges"),
            expected_text_fragments=("oranges", "transformed"),
        ),
        PipelineScenario(
            case_id="transform_modifier_cozy",
            operation="transform",
            user_input_text="Create a funny game about pears.",
            supporting_texts=(),
            reasoning_params={"updates": {"modifiers[0]": "cozy"}},
            expected_paths={"topic": "pears", "modifiers[0]": "cozy"},
            expected_trace_operations=("transform", "re_encode"),
            expected_trace_fragments=("modifiers[0]", "cozy"),
            expected_text_fragments=("cozy", "transformed"),
        ),
        PipelineScenario(
            case_id="transform_object_website",
            operation="transform",
            user_input_text="Create a funny game about context.",
            supporting_texts=(),
            reasoning_params={"updates": {"object": "website"}},
            expected_paths={"object": "website", "topic": "context", "modifiers[0]": "funny"},
            expected_trace_operations=("transform", "re_encode"),
            expected_trace_fragments=("object", "website"),
            expected_text_fragments=("website", "transformed"),
        ),
        PipelineScenario(
            case_id="transform_constraint_offline",
            operation="transform",
            user_input_text="Create a website about context.",
            supporting_texts=(),
            reasoning_params={"updates": {"constraints[0]": "offline_mode"}},
            expected_paths={"object": "website", "topic": "context", "constraints[0]": "offline_mode"},
            expected_trace_operations=("transform", "re_encode"),
            expected_trace_fragments=("constraints[0]", "offline_mode"),
            expected_text_fragments=("offline_mode", "transformed"),
        ),
        PipelineScenario(
            case_id="compare_topic_apples_oranges",
            operation="compare",
            user_input_text="Create a funny game about apples.",
            supporting_texts=("Create a funny game about oranges.",),
            reasoning_params={},
            expected_paths={"action": "compare", "topic": "topic", "relations[0].target": "apples", "relations[1].target": "oranges"},
            expected_trace_operations=("retrieve_memory", "compare", "re_encode"),
            expected_trace_fragments=("topic", "apples", "oranges"),
            expected_text_fragments=("difference at topic", "apples", "oranges"),
        ),
        PipelineScenario(
            case_id="compare_modifier_funny_casual",
            operation="compare",
            user_input_text="Create a funny game about apples.",
            supporting_texts=("Create a casual game about apples.",),
            reasoning_params={},
            expected_paths={"action": "compare", "topic": "modifiers[0]", "relations[0].target": "funny", "relations[1].target": "casual"},
            expected_trace_operations=("retrieve_memory", "compare", "re_encode"),
            expected_trace_fragments=("modifiers[0]", "funny", "casual"),
            expected_text_fragments=("modifiers[0]", "funny", "casual"),
        ),
        PipelineScenario(
            case_id="compare_object_game_website",
            operation="compare",
            user_input_text="Create a game about apples.",
            supporting_texts=("Create a website about apples.",),
            reasoning_params={},
            expected_paths={"action": "compare", "topic": "object", "relations[0].target": "game", "relations[1].target": "website"},
            expected_trace_operations=("retrieve_memory", "compare", "re_encode"),
            expected_trace_fragments=("object", "game", "website"),
            expected_text_fragments=("difference at object", "game", "website"),
        ),
        PipelineScenario(
            case_id="compare_topic_compression_context",
            operation="compare",
            user_input_text="Create a website about compression.",
            supporting_texts=("Create a website about context.",),
            reasoning_params={},
            expected_paths={"action": "compare", "topic": "topic", "relations[0].target": "compression", "relations[1].target": "context"},
            expected_trace_operations=("retrieve_memory", "compare", "re_encode"),
            expected_trace_fragments=("topic", "compression", "context"),
            expected_text_fragments=("difference at topic", "compression", "context"),
        ),
        PipelineScenario(
            case_id="merge_add_topic_apples",
            operation="merge",
            user_input_text="Create a funny game.",
            supporting_texts=("Create a game about apples.",),
            reasoning_params={},
            expected_paths={"object": "game", "topic": "apples", "modifiers[0]": "funny"},
            expected_trace_operations=("retrieve_memory", "merge", "re_encode"),
            expected_trace_fragments=("merged_changes", "topic", "apples"),
            expected_text_fragments=("apples", "merged compatible slots"),
        ),
        PipelineScenario(
            case_id="merge_add_modifier_funny",
            operation="merge",
            user_input_text="Create a game about oranges.",
            supporting_texts=("Create a funny game about oranges.",),
            reasoning_params={},
            expected_paths={"object": "game", "topic": "oranges", "modifiers[0]": "funny"},
            expected_trace_operations=("retrieve_memory", "merge", "re_encode"),
            expected_trace_fragments=("merged_changes", "modifiers[0]", "funny"),
            expected_text_fragments=("funny", "merged compatible slots"),
        ),
        PipelineScenario(
            case_id="merge_add_topic_context",
            operation="merge",
            user_input_text="Create a funny website.",
            supporting_texts=("Create a website about context.",),
            reasoning_params={},
            expected_paths={"object": "website", "topic": "context", "modifiers[0]": "funny"},
            expected_trace_operations=("retrieve_memory", "merge", "re_encode"),
            expected_trace_fragments=("merged_changes", "context", "website"),
            expected_text_fragments=("context", "merged compatible slots"),
        ),
        PipelineScenario(
            case_id="merge_modifier_union",
            operation="merge",
            user_input_text="Create a funny game about apples.",
            supporting_texts=("Create a casual game about apples.",),
            reasoning_params={},
            expected_paths={"object": "game", "topic": "apples", "modifiers[0]": "funny", "modifiers[1]": "casual"},
            expected_trace_operations=("retrieve_memory", "merge", "re_encode"),
            expected_trace_fragments=("merged_changes", "modifiers[1]", "casual"),
            expected_text_fragments=("funny", "casual", "merged compatible slots"),
        ),
        PipelineScenario(
            case_id="negate_modifier_funny",
            operation="negate",
            user_input_text="Create a funny game about apples.",
            supporting_texts=(),
            reasoning_params={"target": "funny", "apply": True},
            expected_paths={"object": "game", "topic": "apples", "negations[0]": "funny"},
            expected_trace_operations=("negate", "re_encode"),
            expected_trace_fragments=("target", "funny", "negations[0]"),
            expected_text_fragments=("not funny", "updated negations"),
        ),
        PipelineScenario(
            case_id="negate_modifier_cozy",
            operation="negate",
            user_input_text="Create a cozy game about pears.",
            supporting_texts=(),
            reasoning_params={"target": "cozy", "apply": True},
            expected_paths={"object": "game", "topic": "pears", "negations[0]": "cozy"},
            expected_trace_operations=("negate", "re_encode"),
            expected_trace_fragments=("target", "cozy", "negations[0]"),
            expected_text_fragments=("not cozy", "updated negations"),
        ),
        PipelineScenario(
            case_id="negate_constraint_offline",
            operation="negate",
            user_input_text="Create a website about context.",
            supporting_texts=(),
            reasoning_params={"constraint": "offline_mode", "apply": True},
            expected_paths={"object": "website", "topic": "context", "constraints[0]": "offline_mode"},
            expected_trace_operations=("negate", "re_encode"),
            expected_trace_fragments=("constraint", "offline_mode"),
            expected_text_fragments=("offline_mode", "updated negations"),
        ),
        PipelineScenario(
            case_id="generalize_game_pattern",
            operation="generalize",
            user_input_text="Create a funny game about apples.",
            supporting_texts=("Create a casual game about oranges.", "Create a cozy game about pears."),
            reasoning_params={},
            expected_paths={"object": "game", "topic": "apples", "modifiers[0]": "funny"},
            expected_trace_operations=("retrieve_memory", "generalize", "re_encode"),
            expected_trace_fragments=("output_encoding_type", "output_pattern_id"),
            expected_text_fragments=("generalized the structure", "Create funny game about apples."),
            expected_encoding_types=("evolved_pattern", "evolved_subpatterns", "hierarchical_pattern", "hierarchical_subpatterns"),
        ),
        PipelineScenario(
            case_id="generalize_website_pattern",
            operation="generalize",
            user_input_text="Create a website about compression.",
            supporting_texts=("Create a website about context.", "Create a website about memory."),
            reasoning_params={},
            expected_paths={"object": "website", "topic": "compression"},
            expected_trace_operations=("retrieve_memory", "generalize", "re_encode"),
            expected_trace_fragments=("output_encoding_type", "output_pattern_id"),
            expected_text_fragments=("generalized the structure", "Create website about compression."),
            expected_encoding_types=("evolved_pattern", "evolved_subpatterns", "hierarchical_pattern", "hierarchical_subpatterns"),
        ),
        PipelineScenario(
            case_id="specialize_topic_apples",
            operation="specialize",
            user_input_text="Create a funny game.",
            supporting_texts=("Create a funny game about apples.",),
            reasoning_params={},
            expected_paths={"object": "game", "topic": "apples", "modifiers[0]": "funny"},
            expected_trace_operations=("retrieve_memory", "specialize", "re_encode"),
            expected_trace_fragments=("requested_fills", "topic", "apples"),
            expected_text_fragments=("filled missing structure slots", "apples"),
        ),
        PipelineScenario(
            case_id="specialize_modifier_funny",
            operation="specialize",
            user_input_text="Create a game about oranges.",
            supporting_texts=("Create a funny game about oranges.",),
            reasoning_params={},
            expected_paths={"object": "game", "topic": "oranges", "modifiers[0]": "funny"},
            expected_trace_operations=("retrieve_memory", "specialize", "re_encode"),
            expected_trace_fragments=("requested_fills", "modifiers[0]", "funny"),
            expected_text_fragments=("filled missing structure slots", "funny"),
        ),
        PipelineScenario(
            case_id="specialize_topic_context",
            operation="specialize",
            user_input_text="Create a website.",
            supporting_texts=("Create a website about context.",),
            reasoning_params={},
            expected_paths={"object": "website", "topic": "context"},
            expected_trace_operations=("retrieve_memory", "specialize", "re_encode"),
            expected_trace_fragments=("requested_fills", "topic", "context"),
            expected_text_fragments=("filled missing structure slots", "context"),
        ),
    ]


def _scenario_to_dict(scenario: PipelineScenario) -> Dict[str, object]:
    return {
        "case_id": scenario.case_id,
        "operation": scenario.operation,
        "user_input_text": scenario.user_input_text,
        "supporting_texts": list(scenario.supporting_texts),
        "reasoning_params": dict(scenario.reasoning_params),
        "expected_paths": dict(scenario.expected_paths),
        "expected_trace_operations": list(scenario.expected_trace_operations),
        "expected_trace_fragments": list(scenario.expected_trace_fragments),
        "expected_text_fragments": list(scenario.expected_text_fragments),
        "expected_encoding_types": list(scenario.expected_encoding_types),
        "expect_pattern_reuse": scenario.expect_pattern_reuse,
    }


def _structure_preserved(
    pipeline: BrainAssistedReasoningPipeline,
    output: BrainInputV1,
) -> bool:
    sequence = pipeline.runtime.brain_input_codec.brain_input_to_sequence(
        output,
        strategy="pattern_first",
        current_sequence=1,
    )
    sequence_roundtrip = pipeline.runtime.brain_input_codec.sequence_to_brain_input(
        sequence,
        strategy="pattern_first",
    )
    grid_payload = pipeline.runtime.brain_input_codec.brain_input_to_grid(
        output,
        strategy="pattern_first",
        current_sequence=1,
    )
    grid_roundtrip = pipeline.runtime.brain_input_codec.grid_to_brain_input(grid_payload)
    expected_bundle = pipeline.runtime.codec.encode_memory_bundle(
        pipeline.runtime.brain_input_codec.brain_input_to_memory(output)
    )
    sequence_bundle = pipeline.runtime.codec.encode_memory_bundle(
        pipeline.runtime.brain_input_codec.brain_input_to_memory(sequence_roundtrip)
    )
    grid_bundle = pipeline.runtime.codec.encode_memory_bundle(
        pipeline.runtime.brain_input_codec.brain_input_to_memory(grid_roundtrip)
    )
    return expected_bundle == sequence_bundle == grid_bundle


def _brain_input_correct(output: BrainInputV1, scenario: PipelineScenario) -> bool:
    memory = BrainCodec.resolve_memory(_brain_input_to_memory(output))
    for path, expected in scenario.expected_paths.items():
        actual = _read_path(memory, str(path))
        if actual != expected:
            return False
    if scenario.expected_encoding_types and output.encoding_type not in scenario.expected_encoding_types:
        return False
    if scenario.expect_pattern_reuse is not None and _has_pattern_reuse(output) != scenario.expect_pattern_reuse:
        return False
    return True


def _trace_accurate(result: BrainAssistedPipelineResult, scenario: PipelineScenario) -> bool:
    operations = tuple(step.operation for step in result.reasoning_trace)
    if operations != scenario.expected_trace_operations:
        return False
    trace_payload = json.dumps([step.to_dict() for step in result.reasoning_trace], sort_keys=True)
    for fragment in scenario.expected_trace_fragments:
        if fragment not in trace_payload:
            return False
    return True


def _llm_alignment_correct(result: BrainAssistedPipelineResult, scenario: PipelineScenario) -> bool:
    if result.llm_output.structured_text != result.reconstructed_text:
        return False
    user_text = result.llm_output.user_facing_text.lower()
    for fragment in scenario.expected_text_fragments:
        if fragment.lower() not in user_text:
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
    results = report.get("results", [])
    if not isinstance(aggregate, dict):
        aggregate = {}
    if not isinstance(results, list):
        results = []

    lines = [
        "# BrAIn-Assisted Reasoning Pipeline Benchmark",
        "",
        "This report validates the full proof-of-concept pipeline:",
        "",
        "`text -> parser -> BrainInputV1 -> pattern-first memory -> native reasoning -> trace -> LLM interface layer`",
        "",
        "## Summary",
        "",
        f"- Report name: `{report.get('report_name', 'brain_assisted_pipeline_20')}`",
        f"- Case count: `{aggregate.get('case_count', 0)}`",
        f"- Structure preservation rate: `{aggregate.get('structure_preservation_rate', 0.0)}`",
        f"- BrainInput correctness rate: `{aggregate.get('brain_input_correct_rate', 0.0)}`",
        f"- Trace accuracy rate: `{aggregate.get('trace_accuracy_rate', 0.0)}`",
        f"- LLM alignment rate: `{aggregate.get('llm_alignment_rate', 0.0)}`",
        f"- Pattern reuse rate: `{aggregate.get('pattern_reuse_rate', 0.0)}`",
        f"- Operation counts: `{aggregate.get('operation_counts', {})}`",
        f"- Output encoding counts: `{aggregate.get('encoding_counts', {})}`",
        "",
        "## Why This Matters",
        "",
        "- BrAIn performed the reasoning on structured semantic state, not on text tokens.",
        "- The LLM-facing layer only converted BrainInputV1 plus the deterministic trace into user-facing language.",
        "- Compared with LLM-only reasoning, the transformation logic stayed inspectable, replayable, and testable slot by slot.",
        "",
        "## Scenario Outputs",
        "",
    ]

    for result in results:
        if not isinstance(result, dict):
            continue
        output = result.get("result", {})
        if not isinstance(output, dict):
            output = {}
        lines.extend(
            [
                f"### {result.get('case_id', 'case')}",
                "",
                f"- Operation: `{result.get('operation', '')}`",
                f"- Structure preserved: `{result.get('structure_preserved', False)}`",
                f"- BrainInput correct: `{result.get('brain_input_correct', False)}`",
                f"- Trace accurate: `{result.get('trace_accurate', False)}`",
                f"- LLM aligned: `{result.get('llm_aligned', False)}`",
                f"- Pattern reused: `{result.get('pattern_reused', False)}`",
                "",
                "**Structured**",
                "",
                "```json",
                json.dumps(output.get("result_brain_input", {}), indent=2, sort_keys=True),
                "```",
                "",
                "**Trace**",
                "",
                "```json",
                json.dumps(output.get("reasoning_trace", []), indent=2, sort_keys=True),
                "```",
                "",
                "**Text**",
                "",
                output.get("llm_output", {}).get("user_facing_text", ""),
                "",
            ]
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="brain_assisted_pipeline_benchmark",
        description="Run the BrAIn-assisted reasoning pipeline benchmark.",
    )
    parser.add_argument(
        "--report-name",
        default="brain_assisted_pipeline_20",
        help="Base name for the saved pipeline benchmark files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved benchmark artifacts.",
    )
    args = parser.parse_args()
    result = run_pipeline_benchmark(report_name=args.report_name, output_dir=args.output_dir)
    print("=" * 72)
    print("BRAIN-ASSISTED REASONING PIPELINE BENCHMARK")
    print(f"JSON report: {result['json_path']}")
    print(f"Markdown summary: {result['markdown_path']}")
    print(f"State dir: {result['state_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
