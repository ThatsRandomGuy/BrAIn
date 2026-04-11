from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from brain_goal_reward_v5 import BrainGoalRewardV5, GoalTarget
from brain_input_v1 import BrainInputV1
from brain_planning_simulation_v3 import memory_to_brain_input
from brain_planning_simulation_v3_benchmark import PlanningScenario, _build_scenarios
from brain_reasoning_engine import DEFAULT_OUTPUT_DIR


BASELINE_REPORT_NAME = "brain_reality_grounding_v4_50"
BASELINE_FALLBACK_AGGREGATE = {
    "decision_accuracy": 0.38,
    "realistic_selection_rate": 0.74,
    "average_selected_plausibility": 0.3599,
    "outcome_prediction_quality": 0.48,
}


def run_goal_reward_v5_benchmark(
    report_name: str = "brain_goal_reward_v5_50",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, object]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = resolved_output_dir / f"{report_name}.state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    engine = BrainGoalRewardV5(state_dir=state_dir)

    scenarios = _build_scenarios()
    if len(scenarios) != 50:
        raise ValueError(f"Goal & Reward V5 benchmark must contain exactly 50 scenarios, found {len(scenarios)}.")

    results: List[Dict[str, object]] = []
    correct_count = 0
    goal_alignment_correct_count = 0
    quality_total = 0.0
    goal_alignment_total = 0.0
    reward_total = 0.0
    decision_quality_total = 0.0
    selected_depths: List[int] = []

    for index, scenario in enumerate(scenarios, start=1):
        input_brain = memory_to_brain_input(engine.v4_engine.v35_engine.v3_engine, scenario.root, current_sequence=index)
        preferred_chain = tuple(
            memory_to_brain_input(engine.v4_engine.v35_engine.v3_engine, memory, current_sequence=index)
            for memory in scenario.preferred.chain
        )
        alternative_specs = [
            (
                tuple(memory_to_brain_input(engine.v4_engine.v35_engine.v3_engine, memory, current_sequence=index) for memory in alternative.chain),
                alternative.relation_types,
                alternative.desirability,
                alternative.risk_level,
            )
            for alternative in scenario.alternatives
        ]
        goal_weights = _configure_goals(
            engine=engine,
            scenario=scenario,
            preferred_chain=preferred_chain,
            alternative_chains=alternative_specs,
        )
        decision = engine.plan(
            input_brain,
            goal_weights=goal_weights,
            max_depth=5,
            beam_width=4,
            current_sequence=index,
        )
        validation = engine.validate_decision(
            input_brain=input_brain,
            decision=decision,
            goal_weights=goal_weights,
            preferred_chain=preferred_chain,
            preferred_relation_types=scenario.preferred.relation_types,
            alternative_chains=alternative_specs,
            current_sequence=index,
        )
        success = bool(validation.get("success", False))
        if success:
            correct_count += 1
        selected_alignment = float(validation.get("selected_goal_alignment_score", 0.0))
        reward = float(validation.get("reward", 0.0))
        unrealistic = _is_unrealistic_selection(
            engine=engine,
            decision=decision,
            preferred_chain=preferred_chain,
            alternative_chains=alternative_specs,
        )
        preferred_alignment, _, _ = engine.evaluate_goal_alignment(
            states=preferred_chain,
            goal_weights=goal_weights,
            current_sequence=index,
        )
        alternative_alignment = max(
            (
                engine.evaluate_goal_alignment(
                    states=chain,
                    goal_weights=goal_weights,
                    current_sequence=index,
                )[0]
                for chain, _, _, _ in alternative_specs
            ),
            default=0.0,
        )
        goal_alignment_correct = bool(
            decision.selected_path is not None
            and selected_alignment >= max(alternative_alignment, preferred_alignment * 0.8)
        )
        if goal_alignment_correct:
            goal_alignment_correct_count += 1
        decision_quality = round(
            (0.5 * (1.0 if success else 0.0))
            + (0.3 * selected_alignment)
            + (0.2 * (0.0 if unrealistic else 1.0)),
            4,
        )
        quality_total += float(validation.get("outcome_prediction_quality", 0.0))
        goal_alignment_total += selected_alignment
        reward_total += reward
        decision_quality_total += decision_quality
        selected_depth = decision.selected_path.depth if decision.selected_path is not None else 0
        selected_depths.append(selected_depth)
        results.append(
            {
                "index": index,
                "scenario_id": scenario.scenario_id,
                "phase": scenario.phase,
                "unseen_input": scenario.unseen_input,
                "goal_weights": {str(key): round(float(value), 4) for key, value in sorted(goal_weights.items())},
                "decision": decision.to_dict(),
                "validation": validation,
                "selected_depth": selected_depth,
                "success": success,
                "goal_alignment_correct": goal_alignment_correct,
                "unrealistic_selection": unrealistic,
                "decision_quality": decision_quality,
                "preferred_goal_alignment": preferred_alignment,
                "best_alternative_goal_alignment": alternative_alignment,
            }
        )

    first_10 = results[:10]
    last_10 = results[-10:]
    aggregate = {
        "scenario_count": len(results),
        "decision_accuracy": round(correct_count / max(1, len(results)), 4),
        "goal_alignment_accuracy": round(goal_alignment_correct_count / max(1, len(results)), 4),
        "average_goal_alignment": round(goal_alignment_total / max(1, len(results)), 4),
        "average_reward": round(reward_total / max(1, len(results)), 4),
        "decision_quality": round(decision_quality_total / max(1, len(results)), 4),
        "outcome_prediction_quality": round(quality_total / max(1, len(results)), 4),
        "average_chain_depth": round(sum(selected_depths) / max(1, len(selected_depths)), 4),
        "max_chain_depth": max(selected_depths) if selected_depths else 0,
        "accuracy_first_10": round(sum(1 for result in first_10 if result["success"]) / max(1, len(first_10)), 4),
        "accuracy_last_10": round(sum(1 for result in last_10 if result["success"]) / max(1, len(last_10)), 4),
        "accuracy_improvement": round(
            (sum(1 for result in last_10 if result["success"]) / max(1, len(last_10)))
            - (sum(1 for result in first_10 if result["success"]) / max(1, len(first_10))),
            4,
        ),
        "reward_first_10": round(sum(float(result.get("validation", {}).get("reward", 0.0)) for result in first_10) / max(1, len(first_10)), 4),
        "reward_last_10": round(sum(float(result.get("validation", {}).get("reward", 0.0)) for result in last_10) / max(1, len(last_10)), 4),
        "reward_improvement": round(
            (sum(float(result.get("validation", {}).get("reward", 0.0)) for result in last_10) / max(1, len(last_10)))
            - (sum(float(result.get("validation", {}).get("reward", 0.0)) for result in first_10) / max(1, len(first_10))),
            4,
        ),
        **engine.registry_summary(),
    }

    baseline = _load_baseline_report(resolved_output_dir / f"{BASELINE_REPORT_NAME}.json")
    baseline_source = BASELINE_REPORT_NAME
    if baseline is None:
        baseline = {"aggregate": dict(BASELINE_FALLBACK_AGGREGATE)}
        baseline_source = f"{BASELINE_REPORT_NAME} (documented fallback)"
    baseline_aggregate = baseline.get("aggregate", {}) if isinstance(baseline, dict) else {}
    if isinstance(baseline_aggregate, dict):
        aggregate["decision_accuracy_improvement"] = round(
            aggregate["decision_accuracy"] - float(baseline_aggregate.get("decision_accuracy", 0.0)),
            4,
        )
        aggregate["outcome_quality_improvement"] = round(
            aggregate["outcome_prediction_quality"] - float(baseline_aggregate.get("outcome_prediction_quality", 0.0)),
            4,
        )
        baseline_quality = _baseline_decision_quality(
            engine=engine,
            report=baseline,
            scenarios=scenarios,
        )
        if baseline_quality is not None:
            aggregate["decision_quality_improvement"] = round(
                aggregate["decision_quality"] - baseline_quality,
                4,
            )

    report = {
        "report_name": report_name,
        "baseline_report_name": baseline_source,
        "aggregate": aggregate,
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
        "goal_registry_path": str(state_dir / "brain_goals.json"),
        "goal_outcome_registry_path": str(state_dir / "brain_goal_outcomes.json"),
        "decision_history_path": str(state_dir / "brain_goal_history.json"),
        "state_dir": str(state_dir),
    }


def _configure_goals(
    engine: BrainGoalRewardV5,
    scenario: PlanningScenario,
    preferred_chain: Sequence[BrainInputV1],
    alternative_chains: Sequence[Tuple[Sequence[BrainInputV1], Sequence[str], float, float]],
) -> Dict[str, float]:
    domain_id = scenario.scenario_id.replace(f"{scenario.phase}_", "", 1)
    progress_goal_id = f"maximize_gain__{domain_id}"
    completion_goal_id = f"complete_task__{domain_id}"
    safety_goal_id = f"avoid_risk__{domain_id}"

    progress_targets = [
        GoalTarget(
            target_id=f"{progress_goal_id}_{index}",
            target_brain=state,
            polarity="seek",
            weight=1.0 - (index * 0.15),
        )
        for index, state in enumerate(preferred_chain[:2])
    ]
    completion_targets = [
        GoalTarget(
            target_id=f"{completion_goal_id}_{index}",
            target_brain=state,
            polarity="seek",
            weight=1.0 - (index * 0.1),
        )
        for index, state in enumerate(preferred_chain[-2:])
    ]
    safety_chain = alternative_chains[0][0] if alternative_chains else ()
    safety_targets = [
        GoalTarget(
            target_id=f"{safety_goal_id}_{index}",
            target_brain=state,
            polarity="avoid",
            weight=1.0 - (index * 0.15),
        )
        for index, state in enumerate(safety_chain[:3])
    ]

    engine.ensure_goal(
        goal_id=progress_goal_id,
        targets=progress_targets,
        base_weight=1.0,
        notes=["domain_progress_goal"],
    )
    engine.ensure_goal(
        goal_id=completion_goal_id,
        targets=completion_targets,
        base_weight=1.0,
        notes=["domain_completion_goal"],
    )
    engine.ensure_goal(
        goal_id=safety_goal_id,
        targets=safety_targets,
        base_weight=1.0,
        notes=["domain_safety_goal"],
    )

    root_action = scenario.root.action
    if root_action in {"eat", "cook", "repair", "charge"}:
        return {
            progress_goal_id: 0.3,
            completion_goal_id: 0.35,
            safety_goal_id: 0.35,
        }
    return {
        progress_goal_id: 0.35,
        completion_goal_id: 0.4,
        safety_goal_id: 0.25,
    }


def _expected_states(
    preferred_chain: Sequence[BrainInputV1],
    alternative_chains: Sequence[Tuple[Sequence[BrainInputV1], Sequence[str], float, float]],
) -> List[BrainInputV1]:
    states = list(preferred_chain)
    for chain, _, _, _ in alternative_chains:
        states.extend(chain)
    return states


def _is_unrealistic_selection(
    engine: BrainGoalRewardV5,
    decision: object,
    preferred_chain: Sequence[BrainInputV1],
    alternative_chains: Sequence[Tuple[Sequence[BrainInputV1], Sequence[str], float, float]],
) -> bool:
    selected_path = getattr(decision, "selected_path", None)
    if selected_path is None:
        return False
    expected_states = _expected_states(preferred_chain, alternative_chains)
    comparator = engine.v4_engine.v35_engine.v3_engine.v2_engine.v1_engine._same_structure
    return not any(comparator(selected_path.final_outcome, state) for state in expected_states)


def _baseline_decision_quality(
    engine: BrainGoalRewardV5,
    report: Dict[str, object],
    scenarios: Sequence[PlanningScenario],
) -> float | None:
    results = report.get("results", [])
    if not isinstance(results, list) or not results:
        return None
    quality_total = 0.0
    total = 0
    comparator = engine.v4_engine.v35_engine.v3_engine.v2_engine.v1_engine._same_structure
    for result in results:
        if not isinstance(result, dict):
            continue
        index = int(result.get("index", 0))
        if index <= 0 or index > len(scenarios):
            continue
        scenario = scenarios[index - 1]
        preferred_chain = tuple(
            memory_to_brain_input(engine.v4_engine.v35_engine.v3_engine, memory, current_sequence=index)
            for memory in scenario.preferred.chain
        )
        alternative_specs = [
            (
                tuple(memory_to_brain_input(engine.v4_engine.v35_engine.v3_engine, memory, current_sequence=index) for memory in alternative.chain),
                alternative.relation_types,
                alternative.desirability,
                alternative.risk_level,
            )
            for alternative in scenario.alternatives
        ]
        goal_weights = _configure_goals(
            engine=engine,
            scenario=scenario,
            preferred_chain=preferred_chain,
            alternative_chains=alternative_specs,
        )
        selected_outcome = _extract_final_outcome(result)
        if selected_outcome is None:
            continue
        selected_alignment, _, _ = engine.evaluate_goal_alignment(
            states=[selected_outcome],
            goal_weights=goal_weights,
            current_sequence=index,
        )
        success = bool(result.get("success", False))
        expected_states = _expected_states(preferred_chain, alternative_specs)
        unrealistic = not any(comparator(selected_outcome, state) for state in expected_states)
        quality_total += (0.5 * (1.0 if success else 0.0)) + (0.3 * selected_alignment) + (0.2 * (0.0 if unrealistic else 1.0))
        total += 1
    if total == 0:
        return None
    return round(quality_total / total, 4)


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
        "# BrAIn Goal & Reward V5 Benchmark",
        "",
        "This report evaluates goal-driven re-ranking on top of V4 grounded planning.",
        "",
        "## Summary",
        "",
        f"- Report name: `{report.get('report_name', 'brain_goal_reward_v5_50')}`",
        f"- Baseline report: `{report.get('baseline_report_name', BASELINE_REPORT_NAME)}`",
        f"- Scenario count: `{aggregate.get('scenario_count', 0)}`",
        f"- Decision accuracy: `{aggregate.get('decision_accuracy', 0.0)}`",
        f"- Decision accuracy improvement: `{aggregate.get('decision_accuracy_improvement', 0.0)}`",
        f"- Goal alignment accuracy: `{aggregate.get('goal_alignment_accuracy', 0.0)}`",
        f"- Average goal alignment: `{aggregate.get('average_goal_alignment', 0.0)}`",
        f"- Average reward: `{aggregate.get('average_reward', 0.0)}`",
        f"- Decision quality: `{aggregate.get('decision_quality', 0.0)}`",
        f"- Decision quality improvement: `{aggregate.get('decision_quality_improvement', 0.0)}`",
        f"- Reward improvement: `{aggregate.get('reward_improvement', 0.0)}`",
        f"- Outcome prediction quality: `{aggregate.get('outcome_prediction_quality', 0.0)}`",
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
                f"- Goal alignment correct: `{result.get('goal_alignment_correct', False)}`",
                f"- Decision quality: `{result.get('decision_quality', 0.0)}`",
                f"- Reward: `{result.get('validation', {}).get('reward', 0.0)}`",
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
        prog="brain_goal_reward_v5_benchmark",
        description="Run BrAIn Goal & Reward V5 benchmark scenarios.",
    )
    parser.add_argument(
        "--report-name",
        default="brain_goal_reward_v5_50",
        help="Base name for the saved V5 benchmark files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved benchmark artifacts.",
    )
    args = parser.parse_args()
    result = run_goal_reward_v5_benchmark(report_name=args.report_name, output_dir=args.output_dir)
    print("=" * 72)
    print("BRAIN GOAL & REWARD V5 BENCHMARK")
    print(f"JSON report: {result['json_path']}")
    print(f"Markdown summary: {result['markdown_path']}")
    print(f"Goal registry: {result['goal_registry_path']}")
    print(f"Goal outcome registry: {result['goal_outcome_registry_path']}")
    print(f"Decision history: {result['decision_history_path']}")
    print(f"State dir: {result['state_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
