from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

from brain_generative_expansion_v35 import BrainGenerativeExpansionV35
from brain_planning_simulation_v3 import memory_to_brain_input
from brain_planning_simulation_v3_benchmark import _build_scenarios
from brain_reasoning_engine import DEFAULT_OUTPUT_DIR


BASELINE_REPORT_NAME = "brain_planning_simulation_v3_50_b"
BASELINE_FALLBACK_AGGREGATE = {
    "average_candidate_path_count": 1.66,
    "decision_accuracy": 0.28,
    "concept_usage_rate": 0.4,
    "outcome_prediction_quality": 0.36,
}


def run_generative_expansion_v35_benchmark(
    report_name: str = "brain_generative_expansion_v35_50",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, object]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = resolved_output_dir / f"{report_name}.state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    engine = BrainGenerativeExpansionV35(state_dir=state_dir)

    scenarios = _build_scenarios()
    if len(scenarios) != 50:
        raise ValueError(f"Generative Expansion V3.5 benchmark must contain exactly 50 scenarios, found {len(scenarios)}.")

    results: List[Dict[str, object]] = []
    correct_count = 0
    quality_total = 0.0
    candidate_count_total = 0
    selected_depths: List[int] = []

    for index, scenario in enumerate(scenarios, start=1):
        input_brain = memory_to_brain_input(engine.v3_engine, scenario.root, current_sequence=index)
        preferred_chain = tuple(
            memory_to_brain_input(engine.v3_engine, memory, current_sequence=index)
            for memory in scenario.preferred.chain
        )
        alternative_specs = [
            (
                tuple(memory_to_brain_input(engine.v3_engine, memory, current_sequence=index) for memory in alternative.chain),
                alternative.relation_types,
                alternative.desirability,
                alternative.risk_level,
            )
            for alternative in scenario.alternatives
        ]
        decision = engine.plan(
            input_brain,
            max_depth=5,
            beam_width=4,
            current_sequence=index,
        )
        validation = engine.validate_decision(
            input_brain=input_brain,
            decision=decision,
            preferred_chain=preferred_chain,
            preferred_relation_types=scenario.preferred.relation_types,
            alternative_chains=alternative_specs,
            current_sequence=index,
        )
        success = bool(validation.get("success", False))
        if success:
            correct_count += 1
        quality_total += float(validation.get("outcome_prediction_quality", 0.0))
        candidate_count_total += len(decision.candidate_paths)
        selected_depth = decision.selected_path.depth if decision.selected_path is not None else 0
        selected_depths.append(selected_depth)
        results.append(
            {
                "index": index,
                "scenario_id": scenario.scenario_id,
                "phase": scenario.phase,
                "unseen_input": scenario.unseen_input,
                "decision": decision.to_dict(),
                "validation": validation,
                "selected_depth": selected_depth,
                "candidate_path_count": len(decision.candidate_paths),
                "success": success,
            }
        )

    first_10 = results[:10]
    last_10 = results[-10:]
    unseen_results = [result for result in results if bool(result.get("unseen_input", False))]
    unseen_round_1 = [result for result in results if result.get("phase") == "unseen_round_1"]
    unseen_round_3 = [result for result in results if result.get("phase") == "unseen_round_3"]
    concept_used = sum(1 for result in results if bool(result.get("validation", {}).get("used_concept_abstraction", False)))
    generated_used = sum(1 for result in results if bool(result.get("validation", {}).get("used_generation", False)))
    nonzero_candidates = sum(1 for result in results if int(result.get("candidate_path_count", 0)) > 0)

    aggregate = {
        "scenario_count": len(results),
        "planning_success_rate": round(correct_count / max(1, len(results)), 4),
        "average_candidate_path_count": round(candidate_count_total / max(1, len(results)), 4),
        "nonzero_candidate_rate": round(nonzero_candidates / max(1, len(results)), 4),
        "average_chain_depth": round(sum(selected_depths) / max(1, len(selected_depths)), 4),
        "max_chain_depth": max(selected_depths) if selected_depths else 0,
        "outcome_prediction_quality": round(quality_total / max(1, len(results)), 4),
        "concept_usage_rate": round(concept_used / max(1, len(results)), 4),
        "generation_usage_rate": round(generated_used / max(1, len(results)), 4),
        "accuracy_first_10": round(sum(1 for result in first_10 if result["success"]) / max(1, len(first_10)), 4),
        "accuracy_last_10": round(sum(1 for result in last_10 if result["success"]) / max(1, len(last_10)), 4),
        "accuracy_improvement": round(
            (sum(1 for result in last_10 if result["success"]) / max(1, len(last_10)))
            - (sum(1 for result in first_10 if result["success"]) / max(1, len(first_10))),
            4,
        ),
        "unseen_accuracy_round_1": round(sum(1 for result in unseen_round_1 if result["success"]) / max(1, len(unseen_round_1)), 4),
        "unseen_accuracy_round_3": round(sum(1 for result in unseen_round_3 if result["success"]) / max(1, len(unseen_round_3)), 4),
        "unseen_accuracy_improvement": round(
            (sum(1 for result in unseen_round_3 if result["success"]) / max(1, len(unseen_round_3)))
            - (sum(1 for result in unseen_round_1 if result["success"]) / max(1, len(unseen_round_1))),
            4,
        ),
        **engine.registry_summary(),
    }

    baseline = _load_baseline_report(resolved_output_dir / f"{BASELINE_REPORT_NAME}.json")
    baseline_source = BASELINE_REPORT_NAME
    if baseline is None:
        baseline = {"aggregate": dict(BASELINE_FALLBACK_AGGREGATE)}
        baseline_source = f"{BASELINE_REPORT_NAME} (documented fallback)"
    if baseline is not None:
        baseline_aggregate = baseline.get("aggregate", {})
        if isinstance(baseline_aggregate, dict):
            aggregate["candidate_path_count_increase"] = round(
                aggregate["average_candidate_path_count"] - float(baseline_aggregate.get("average_candidate_path_count", 0.0)),
                4,
            )
            aggregate["planning_success_rate_improvement"] = round(
                aggregate["planning_success_rate"] - float(baseline_aggregate.get("decision_accuracy", 0.0)),
                4,
            )
            aggregate["concept_usage_rate_increase"] = round(
                aggregate["concept_usage_rate"] - float(baseline_aggregate.get("concept_usage_rate", 0.0)),
                4,
            )
            baseline_nonzero = _baseline_nonzero_candidate_rate(baseline)
            if baseline_nonzero > 0.0:
                aggregate["nonzero_candidate_rate_increase"] = round(
                    aggregate["nonzero_candidate_rate"] - baseline_nonzero,
                    4,
                )

    report = {
        "report_name": report_name,
        "aggregate": aggregate,
        "baseline_report_name": baseline_source,
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
        "generated_registry_path": str(state_dir / "brain_generated_expansions.json"),
        "outcome_registry_path": str(state_dir / "v3" / "brain_outcome_profiles.json"),
        "concept_registry_path": str(state_dir / "v3" / "v2" / "brain_concepts.json"),
        "state_dir": str(state_dir),
    }


def _load_baseline_report(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_nonzero_candidate_rate(report: Dict[str, object]) -> float:
    results = report.get("results", [])
    if not isinstance(results, list) or not results:
        return 0.0
    return round(sum(1 for result in results if int(result.get("candidate_path_count", 0)) > 0) / len(results), 4)


def _build_markdown_report(report: Dict[str, object]) -> str:
    aggregate = report.get("aggregate", {})
    results = report.get("results", [])
    if not isinstance(aggregate, dict):
        aggregate = {}
    if not isinstance(results, list):
        results = []
    showcase_ids = {
        "train_round_1_eat_food",
        "unseen_round_1_eat_food",
        "unseen_round_3_charge_device",
        "unseen_round_3_study_subject",
    }
    showcased = [result for result in results if isinstance(result, dict) and result.get("scenario_id") in showcase_ids]
    lines = [
        "# BrAIn Generative Expansion V3.5 Benchmark",
        "",
        "This report evaluates concept-driven fallback path generation when direct planning paths are missing.",
        "",
        "## Summary",
        "",
        f"- Report name: `{report.get('report_name', 'brain_generative_expansion_v35_50')}`",
        f"- Baseline report: `{report.get('baseline_report_name', BASELINE_REPORT_NAME)}`",
        f"- Scenario count: `{aggregate.get('scenario_count', 0)}`",
        f"- Planning success rate: `{aggregate.get('planning_success_rate', 0.0)}`",
        f"- Average candidate path count: `{aggregate.get('average_candidate_path_count', 0.0)}`",
        f"- Candidate path count increase: `{aggregate.get('candidate_path_count_increase', 0.0)}`",
        f"- Nonzero candidate rate: `{aggregate.get('nonzero_candidate_rate', 0.0)}`",
        f"- Nonzero candidate rate increase: `{aggregate.get('nonzero_candidate_rate_increase', 0.0)}`",
        f"- Outcome prediction quality: `{aggregate.get('outcome_prediction_quality', 0.0)}`",
        f"- Planning success improvement: `{aggregate.get('planning_success_rate_improvement', 0.0)}`",
        f"- Concept usage rate: `{aggregate.get('concept_usage_rate', 0.0)}`",
        f"- Concept usage increase: `{aggregate.get('concept_usage_rate_increase', 0.0)}`",
        f"- Generation usage rate: `{aggregate.get('generation_usage_rate', 0.0)}`",
        f"- Unseen accuracy round 1: `{aggregate.get('unseen_accuracy_round_1', 0.0)}`",
        f"- Unseen accuracy round 3: `{aggregate.get('unseen_accuracy_round_3', 0.0)}`",
        "",
        "## Example Decisions",
        "",
    ]
    for result in showcased:
        lines.extend(
            [
                f"### {result.get('scenario_id', 'scenario')}",
                "",
                f"- Success: `{result.get('success', False)}`",
                f"- Candidate path count: `{result.get('candidate_path_count', 0)}`",
                f"- Used generation: `{result.get('validation', {}).get('used_generation', False)}`",
                f"- Used concept abstraction: `{result.get('validation', {}).get('used_concept_abstraction', False)}`",
                "",
                "```json",
                json.dumps(result, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="brain_generative_expansion_v35_benchmark",
        description="Run BrAIn Generative Expansion V3.5 benchmark scenarios.",
    )
    parser.add_argument(
        "--report-name",
        default="brain_generative_expansion_v35_50",
        help="Base name for the saved generative expansion V3.5 benchmark files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved benchmark artifacts.",
    )
    args = parser.parse_args()
    result = run_generative_expansion_v35_benchmark(report_name=args.report_name, output_dir=args.output_dir)
    print("=" * 72)
    print("BRAIN GENERATIVE EXPANSION V3.5 BENCHMARK")
    print(f"JSON report: {result['json_path']}")
    print(f"Markdown summary: {result['markdown_path']}")
    print(f"Generated registry: {result['generated_registry_path']}")
    print(f"Outcome registry: {result['outcome_registry_path']}")
    print(f"Concept registry: {result['concept_registry_path']}")
    print(f"State dir: {result['state_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
