from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from brain_generative_expansion_v35_benchmark import BASELINE_FALLBACK_AGGREGATE as V35_FALLBACK
from brain_input_v1 import BrainInputV1
from brain_planning_simulation_v3 import memory_to_brain_input
from brain_planning_simulation_v3_benchmark import PlanningScenario, _build_scenarios
from brain_reality_grounding_v4 import BrainRealityGroundingV4, GroundedPlanningDecision
from brain_reasoning_engine import DEFAULT_OUTPUT_DIR


BASELINE_REPORT_NAME = "brain_generative_expansion_v35_50"


def run_reality_grounding_v4_benchmark(
    report_name: str = "brain_reality_grounding_v4_50",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, object]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = resolved_output_dir / f"{report_name}.state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    engine = BrainRealityGroundingV4(state_dir=state_dir)

    scenarios = _build_scenarios()
    if len(scenarios) != 50:
        raise ValueError(f"Reality Grounding V4 benchmark must contain exactly 50 scenarios, found {len(scenarios)}.")

    results: List[Dict[str, object]] = []
    correct_count = 0
    unrealistic_count = 0
    quality_total = 0.0
    plausibility_total = 0.0
    candidate_count_total = 0
    retained_count_total = 0
    filtered_count_total = 0
    selected_depths: List[int] = []

    for index, scenario in enumerate(scenarios, start=1):
        input_brain = memory_to_brain_input(engine.v35_engine.v3_engine, scenario.root, current_sequence=index)
        preferred_chain = tuple(
            memory_to_brain_input(engine.v35_engine.v3_engine, memory, current_sequence=index)
            for memory in scenario.preferred.chain
        )
        alternative_specs = [
            (
                tuple(memory_to_brain_input(engine.v35_engine.v3_engine, memory, current_sequence=index) for memory in alternative.chain),
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
        unrealistic = _is_unrealistic_selection(
            engine=engine,
            decision=decision,
            preferred_chain=preferred_chain,
            alternative_chains=alternative_specs,
        )
        if unrealistic:
            unrealistic_count += 1
        quality_total += float(validation.get("outcome_prediction_quality", 0.0))
        plausibility_total += float(validation.get("selected_plausibility_score", 0.0))
        candidate_count_total += len(decision.candidate_paths)
        retained_count_total += len(decision.retained_paths)
        filtered_count_total += decision.filtered_path_count
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
                "retained_path_count": len(decision.retained_paths),
                "filtered_path_count": decision.filtered_path_count,
                "success": success,
                "unrealistic_selection": unrealistic,
            }
        )

    first_10 = results[:10]
    last_10 = results[-10:]
    concept_used = sum(1 for result in results if bool(result.get("validation", {}).get("used_concept_abstraction", False)))
    grounding_used = sum(1 for result in results if bool(result.get("validation", {}).get("used_reality_grounding", False)))

    aggregate = {
        "scenario_count": len(results),
        "decision_accuracy": round(correct_count / max(1, len(results)), 4),
        "average_candidate_path_count": round(candidate_count_total / max(1, len(results)), 4),
        "average_retained_path_count": round(retained_count_total / max(1, len(results)), 4),
        "average_filtered_path_count": round(filtered_count_total / max(1, len(results)), 4),
        "average_chain_depth": round(sum(selected_depths) / max(1, len(selected_depths)), 4),
        "max_chain_depth": max(selected_depths) if selected_depths else 0,
        "outcome_prediction_quality": round(quality_total / max(1, len(results)), 4),
        "average_selected_plausibility": round(plausibility_total / max(1, len(results)), 4),
        "unrealistic_selection_rate": round(unrealistic_count / max(1, len(results)), 4),
        "realistic_selection_rate": round(1.0 - (unrealistic_count / max(1, len(results))), 4),
        "concept_usage_rate": round(concept_used / max(1, len(results)), 4),
        "reality_grounding_usage_rate": round(grounding_used / max(1, len(results)), 4),
        "accuracy_first_10": round(sum(1 for result in first_10 if result["success"]) / max(1, len(first_10)), 4),
        "accuracy_last_10": round(sum(1 for result in last_10 if result["success"]) / max(1, len(last_10)), 4),
        "accuracy_improvement": round(
            (sum(1 for result in last_10 if result["success"]) / max(1, len(last_10)))
            - (sum(1 for result in first_10 if result["success"]) / max(1, len(first_10))),
            4,
        ),
        **engine.registry_summary(),
    }

    baseline = _load_baseline_report(resolved_output_dir / f"{BASELINE_REPORT_NAME}.json")
    baseline_source = BASELINE_REPORT_NAME
    if baseline is None:
        baseline = {"aggregate": dict(V35_FALLBACK)}
        baseline_source = f"{BASELINE_REPORT_NAME} (documented fallback)"
    baseline_aggregate = baseline.get("aggregate", {}) if isinstance(baseline, dict) else {}
    if isinstance(baseline_aggregate, dict):
        aggregate["decision_accuracy_improvement"] = round(
            aggregate["decision_accuracy"] - float(baseline_aggregate.get("planning_success_rate", 0.0)),
            4,
        )
        aggregate["outcome_quality_improvement"] = round(
            aggregate["outcome_prediction_quality"] - float(baseline_aggregate.get("outcome_prediction_quality", 0.0)),
            4,
        )
        aggregate["concept_usage_rate_change"] = round(
            aggregate["concept_usage_rate"] - float(baseline_aggregate.get("concept_usage_rate", 0.0)),
            4,
        )
    baseline_unrealistic = _baseline_unrealistic_rate(engine, baseline, scenarios)
    if baseline_unrealistic is not None:
        aggregate["unrealistic_selection_reduction"] = round(
            baseline_unrealistic - aggregate["unrealistic_selection_rate"],
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
        "grounding_registry_path": str(state_dir / "brain_grounding_signals.json"),
        "generated_registry_path": str(state_dir / "v35" / "brain_generated_expansions.json"),
        "outcome_registry_path": str(state_dir / "v35" / "v3" / "brain_outcome_profiles.json"),
        "concept_registry_path": str(state_dir / "v35" / "v3" / "v2" / "brain_concepts.json"),
        "state_dir": str(state_dir),
    }


def _expected_states(
    preferred_chain: Sequence[BrainInputV1],
    alternative_chains: Sequence[Tuple[Sequence[BrainInputV1], Sequence[str], float, float]],
) -> List[BrainInputV1]:
    states: List[BrainInputV1] = list(preferred_chain)
    for chain, _, _, _ in alternative_chains:
        states.extend(chain)
    return states


def _is_unrealistic_selection(
    engine: BrainRealityGroundingV4,
    decision: GroundedPlanningDecision,
    preferred_chain: Sequence[BrainInputV1],
    alternative_chains: Sequence[Tuple[Sequence[BrainInputV1], Sequence[str], float, float]],
) -> bool:
    if decision.selected_path is None:
        return False
    expected_states = _expected_states(preferred_chain, alternative_chains)
    comparator = engine.v35_engine.v3_engine.v2_engine.v1_engine._same_structure
    return not any(comparator(decision.selected_path.final_outcome, state) for state in expected_states)


def _baseline_unrealistic_rate(
    engine: BrainRealityGroundingV4,
    report: Dict[str, object],
    scenarios: Sequence[PlanningScenario],
) -> float | None:
    results = report.get("results", [])
    if not isinstance(results, list) or not results:
        return None
    total = 0
    unrealistic = 0
    comparator = engine.v35_engine.v3_engine.v2_engine.v1_engine._same_structure
    for result in results:
        if not isinstance(result, dict):
            continue
        index = int(result.get("index", 0))
        if index <= 0 or index > len(scenarios):
            continue
        scenario = scenarios[index - 1]
        preferred_chain = tuple(
            memory_to_brain_input(engine.v35_engine.v3_engine, memory, current_sequence=index)
            for memory in scenario.preferred.chain
        )
        alternative_chains = [
            (
                tuple(memory_to_brain_input(engine.v35_engine.v3_engine, memory, current_sequence=index) for memory in alternative.chain),
                alternative.relation_types,
                alternative.desirability,
                alternative.risk_level,
            )
            for alternative in scenario.alternatives
        ]
        final_outcome = _extract_final_outcome(result)
        if final_outcome is None:
            continue
        expected = _expected_states(preferred_chain, alternative_chains)
        total += 1
        if not any(comparator(final_outcome, state) for state in expected):
            unrealistic += 1
    if total == 0:
        return None
    return round(unrealistic / total, 4)


def _extract_final_outcome(result: Dict[str, object]) -> BrainInputV1 | None:
    decision = result.get("decision", {})
    if not isinstance(decision, dict):
        return None
    final_outcome = decision.get("final_predicted_outcome")
    if isinstance(final_outcome, dict):
        return BrainInputV1.from_dict(final_outcome)
    selected_path = decision.get("selected_path")
    if isinstance(selected_path, dict) and isinstance(selected_path.get("final_outcome"), dict):
        return BrainInputV1.from_dict(selected_path.get("final_outcome"))
    return None


def _load_baseline_report(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_markdown_report(report: Dict[str, object]) -> str:
    aggregate = report.get("aggregate", {})
    results = report.get("results", [])
    if not isinstance(aggregate, dict):
        aggregate = {}
    if not isinstance(results, list):
        results = []
    showcase_ids = {
        "unseen_round_1_eat_food",
        "unseen_round_3_charge_device",
        "unseen_round_3_study_subject",
    }
    showcased = [result for result in results if isinstance(result, dict) and result.get("scenario_id") in showcase_ids]
    lines = [
        "# BrAIn Reality Grounding V4 Benchmark",
        "",
        "This report evaluates plausibility filtering and structural grounding over the existing V3.5 generative planner.",
        "",
        "## Summary",
        "",
        f"- Report name: `{report.get('report_name', 'brain_reality_grounding_v4_50')}`",
        f"- Baseline report: `{report.get('baseline_report_name', BASELINE_REPORT_NAME)}`",
        f"- Scenario count: `{aggregate.get('scenario_count', 0)}`",
        f"- Decision accuracy: `{aggregate.get('decision_accuracy', 0.0)}`",
        f"- Decision accuracy improvement: `{aggregate.get('decision_accuracy_improvement', 0.0)}`",
        f"- Unrealistic selection rate: `{aggregate.get('unrealistic_selection_rate', 0.0)}`",
        f"- Unrealistic selection reduction: `{aggregate.get('unrealistic_selection_reduction', 0.0)}`",
        f"- Outcome prediction quality: `{aggregate.get('outcome_prediction_quality', 0.0)}`",
        f"- Outcome quality improvement: `{aggregate.get('outcome_quality_improvement', 0.0)}`",
        f"- Average selected plausibility: `{aggregate.get('average_selected_plausibility', 0.0)}`",
        f"- Average candidate path count: `{aggregate.get('average_candidate_path_count', 0.0)}`",
        f"- Average retained path count: `{aggregate.get('average_retained_path_count', 0.0)}`",
        f"- Average filtered path count: `{aggregate.get('average_filtered_path_count', 0.0)}`",
        f"- Reality grounding usage rate: `{aggregate.get('reality_grounding_usage_rate', 0.0)}`",
        f"- Concept usage rate: `{aggregate.get('concept_usage_rate', 0.0)}`",
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
                f"- Unrealistic selection: `{result.get('unrealistic_selection', False)}`",
                f"- Candidate path count: `{result.get('candidate_path_count', 0)}`",
                f"- Retained path count: `{result.get('retained_path_count', 0)}`",
                f"- Selected plausibility: `{result.get('validation', {}).get('selected_plausibility_score', 0.0)}`",
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
        prog="brain_reality_grounding_v4_benchmark",
        description="Run BrAIn Reality Grounding V4 benchmark scenarios.",
    )
    parser.add_argument(
        "--report-name",
        default="brain_reality_grounding_v4_50",
        help="Base name for the saved reality grounding V4 benchmark files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved benchmark artifacts.",
    )
    args = parser.parse_args()
    result = run_reality_grounding_v4_benchmark(report_name=args.report_name, output_dir=args.output_dir)
    print("=" * 72)
    print("BRAIN REALITY GROUNDING V4 BENCHMARK")
    print(f"JSON report: {result['json_path']}")
    print(f"Markdown summary: {result['markdown_path']}")
    print(f"Grounding registry: {result['grounding_registry_path']}")
    print(f"Generated registry: {result['generated_registry_path']}")
    print(f"Outcome registry: {result['outcome_registry_path']}")
    print(f"Concept registry: {result['concept_registry_path']}")
    print(f"State dir: {result['state_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
