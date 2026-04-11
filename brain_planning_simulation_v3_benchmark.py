from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from brain_codec import BrainCodec, BrainMemory
from brain_planning_simulation_v3 import BrainPlanningSimulationV3, memory_to_brain_input
from brain_reasoning_engine import DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class PlanningChainSpec:
    chain: Tuple[BrainMemory, ...]
    relation_types: Tuple[str, ...]
    desirability: float
    risk_level: float
    label: str


@dataclass(frozen=True)
class PlanningScenario:
    scenario_id: str
    phase: str
    root: BrainMemory
    preferred: PlanningChainSpec
    alternatives: Tuple[PlanningChainSpec, ...]
    unseen_input: bool


def make_memory(
    *,
    actor: str = "user",
    intent: str = "inform",
    action: str = "",
    object: str = "",
    topic: str = "",
    modifiers: Sequence[str] = (),
    constraints: Sequence[str] = (),
    negations: Sequence[str] = (),
    conditions: Sequence[str] = (),
) -> BrainMemory:
    relations: List[Dict[str, str]] = []
    if object and topic:
        relations.append({"type": "about", "source": object, "target": topic})
    for modifier in modifiers:
        if object:
            relations.append({"type": "has_modifier", "source": object, "target": str(modifier)})
    for constraint in constraints:
        if object:
            relations.append({"type": "constraint", "source": object, "target": str(constraint)})
    for negation in negations:
        if object:
            relations.append({"type": "negates", "source": object, "target": str(negation)})
    for condition in conditions:
        if object:
            relations.append({"type": "condition", "source": object, "target": str(condition)})
    return BrainCodec.resolve_memory(
        BrainMemory(
            actor=actor,
            intent=intent,
            action=action,
            object=object,
            topic=topic,
            modifiers=[str(item) for item in modifiers],
            constraints=[str(item) for item in constraints],
            negations=[str(item) for item in negations],
            conditions=[str(item) for item in conditions],
            relations=relations,
            priority="normal",
        )
    )


def positive_chain(*steps: BrainMemory) -> PlanningChainSpec:
    return PlanningChainSpec(
        chain=tuple(steps),
        relation_types=tuple("cause_effect" for _ in steps),
        desirability=1.0,
        risk_level=0.05,
        label="preferred",
    )


def negative_chain(*steps: BrainMemory) -> PlanningChainSpec:
    return PlanningChainSpec(
        chain=tuple(steps),
        relation_types=tuple("cause_effect" for _ in steps),
        desirability=-1.0,
        risk_level=0.95,
        label="avoid",
    )


def run_planning_simulation_v3_benchmark(
    report_name: str = "brain_planning_simulation_v3_50",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, object]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = resolved_output_dir / f"{report_name}.state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    engine = BrainPlanningSimulationV3(state_dir=state_dir)

    scenarios = _build_scenarios()
    if len(scenarios) != 50:
        raise ValueError(f"Planning Simulation V3 benchmark must contain exactly 50 scenarios, found {len(scenarios)}.")

    results: List[Dict[str, object]] = []
    correct_count = 0
    quality_total = 0.0
    selected_depths: List[int] = []

    for index, scenario in enumerate(scenarios, start=1):
        input_brain = memory_to_brain_input(engine, scenario.root, current_sequence=index)
        preferred_chain = tuple(
            memory_to_brain_input(engine, memory, current_sequence=index)
            for memory in scenario.preferred.chain
        )
        alternative_specs = [
            (
                tuple(memory_to_brain_input(engine, memory, current_sequence=index) for memory in alternative.chain),
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
        quality = float(validation.get("outcome_prediction_quality", 0.0))
        quality_total += quality
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
    concept_used = sum(
        1 for result in results if bool(result.get("validation", {}).get("used_concept_abstraction", False))
    )

    aggregate = {
        "scenario_count": len(results),
        "decision_accuracy": round(correct_count / max(1, len(results)), 4),
        "average_chain_depth": round(sum(selected_depths) / max(1, len(selected_depths)), 4),
        "max_chain_depth": max(selected_depths) if selected_depths else 0,
        "depth_increase": round(
            (sum(result["selected_depth"] for result in last_10) / max(1, len(last_10)))
            - (sum(result["selected_depth"] for result in first_10) / max(1, len(first_10))),
            4,
        ),
        "outcome_prediction_quality": round(quality_total / max(1, len(results)), 4),
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
        "concept_usage_rate": round(concept_used / max(1, len(results)), 4),
        "unseen_concept_usage_rate": round(
            sum(1 for result in unseen_results if bool(result.get("validation", {}).get("used_concept_abstraction", False))) / max(1, len(unseen_results)),
            4,
        ),
        "average_candidate_path_count": round(sum(result["candidate_path_count"] for result in results) / max(1, len(results)), 4),
        **engine.registry_summary(),
    }

    report = {
        "report_name": report_name,
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
        "outcome_registry_path": str(state_dir / "brain_outcome_profiles.json"),
        "concept_registry_path": str(state_dir / "v2" / "brain_concepts.json"),
        "link_registry_path": str(state_dir / "v2" / "v1" / "brain_reasoning_links.json"),
        "state_dir": str(state_dir),
    }


def _build_scenarios() -> List[PlanningScenario]:
    domains = [
        {
            "domain_id": "eat_food",
            "field": "object",
            "values": ["apple", "bread", "rice", "beans", "soup"],
            "root": {"action": "eat"},
            "preferred": positive_chain(
                make_memory(action="feel", object="full"),
                make_memory(action="keep", object="energy"),
                make_memory(action="finish", object="task"),
                make_memory(action="earn", object="reward"),
                make_memory(action="rest", object="easy"),
            ),
            "alternative": negative_chain(
                make_memory(action="get", object="sick"),
                make_memory(action="miss", object="work"),
                make_memory(action="lose", object="money"),
                make_memory(action="delay", object="plans"),
                make_memory(action="feel", object="stress"),
            ),
        },
        {
            "domain_id": "drink_liquid",
            "field": "object",
            "values": ["water", "juice", "tea", "milk", "smoothie"],
            "root": {"action": "drink"},
            "preferred": positive_chain(
                make_memory(action="feel", object="refreshed"),
                make_memory(action="think", object="clearly"),
                make_memory(action="solve", object="problem"),
                make_memory(action="gain", object="time"),
                make_memory(action="complete", object="goal"),
            ),
            "alternative": negative_chain(
                make_memory(action="feel", object="nauseous"),
                make_memory(action="miss", object="class"),
                make_memory(action="lose", object="progress"),
                make_memory(action="need", object="recovery"),
                make_memory(action="delay", object="goals"),
            ),
        },
        {
            "domain_id": "charge_device",
            "field": "object",
            "values": ["phone", "tablet", "laptop", "watch", "camera"],
            "root": {"action": "charge"},
            "preferred": positive_chain(
                make_memory(action="fill", object="battery"),
                make_memory(action="use", object="device"),
                make_memory(action="finish", object="task"),
                make_memory(action="earn", object="money"),
                make_memory(action="gain", object="options"),
            ),
            "alternative": negative_chain(
                make_memory(action="overheat", object="device"),
                make_memory(action="damage", object="device"),
                make_memory(action="replace", object="device"),
                make_memory(action="spend", object="money"),
                make_memory(action="delay", object="work"),
            ),
        },
        {
            "domain_id": "water_plants",
            "field": "object",
            "values": ["roses", "tomatoes", "herbs", "trees", "flowers"],
            "root": {"action": "water"},
            "preferred": positive_chain(
                make_memory(action="grow", object="plant"),
                make_memory(action="harvest", object="produce"),
                make_memory(action="eat", object="produce"),
                make_memory(action="stay", object="healthy"),
                make_memory(action="save", object="money"),
            ),
            "alternative": negative_chain(
                make_memory(action="rot", object="roots"),
                make_memory(action="lose", object="plant"),
                make_memory(action="buy", object="replacement"),
                make_memory(action="spend", object="money"),
                make_memory(action="delay", object="harvest"),
            ),
        },
        {
            "domain_id": "repair_device",
            "field": "object",
            "values": ["phone", "laptop", "tablet", "camera", "console"],
            "root": {"action": "repair"},
            "preferred": positive_chain(
                make_memory(action="restore", object="function"),
                make_memory(action="resume", object="work"),
                make_memory(action="finish", object="task"),
                make_memory(action="earn", object="reward"),
                make_memory(action="gain", object="trust"),
            ),
            "alternative": negative_chain(
                make_memory(action="break", object="part"),
                make_memory(action="need", object="parts"),
                make_memory(action="spend", object="money"),
                make_memory(action="delay", object="work"),
                make_memory(action="miss", object="deadline"),
            ),
        },
        {
            "domain_id": "study_subject",
            "field": "topic",
            "values": ["math", "history", "biology", "physics", "chemistry"],
            "root": {"action": "study", "object": "lessons"},
            "preferred": positive_chain(
                make_memory(action="pass", object="exam"),
                make_memory(action="advance", object="course"),
                make_memory(action="graduate", object="program"),
                make_memory(action="get", object="job"),
                make_memory(action="gain", object="income"),
            ),
            "alternative": negative_chain(
                make_memory(action="burn", object="out"),
                make_memory(action="need", object="rest"),
                make_memory(action="miss", object="study"),
                make_memory(action="delay", object="exam"),
                make_memory(action="lose", object="progress"),
            ),
        },
        {
            "domain_id": "clean_surface",
            "field": "topic",
            "values": ["counter", "floor", "window", "desk", "shelf"],
            "root": {"action": "clean", "object": "surface"},
            "preferred": positive_chain(
                make_memory(action="become", object="tidy"),
                make_memory(action="organize", object="room"),
                make_memory(action="find", object="tools"),
                make_memory(action="finish", object="task"),
                make_memory(action="save", object="time"),
            ),
            "alternative": negative_chain(
                make_memory(action="spread", object="dust"),
                make_memory(action="cause", object="cough"),
                make_memory(action="lose", object="focus"),
                make_memory(action="delay", object="task"),
                make_memory(action="need", object="cleanup"),
            ),
        },
        {
            "domain_id": "backup_data",
            "field": "topic",
            "values": ["photos", "videos", "documents", "music", "projects"],
            "root": {"action": "backup", "object": "data"},
            "preferred": positive_chain(
                make_memory(action="preserve", object="files"),
                make_memory(action="restore", object="files"),
                make_memory(action="continue", object="project"),
                make_memory(action="finish", object="project"),
                make_memory(action="gain", object="reward"),
            ),
            "alternative": negative_chain(
                make_memory(action="corrupt", object="backup"),
                make_memory(action="lose", object="files"),
                make_memory(action="redo", object="work"),
                make_memory(action="miss", object="deadline"),
                make_memory(action="lose", object="reward"),
            ),
        },
        {
            "domain_id": "practice_skill",
            "field": "topic",
            "values": ["piano", "guitar", "violin", "drums", "flute"],
            "root": {"action": "practice", "object": "skill"},
            "preferred": positive_chain(
                make_memory(action="improve", object="skill"),
                make_memory(action="perform", object="piece"),
                make_memory(action="receive", object="praise"),
                make_memory(action="gain", object="confidence"),
                make_memory(action="seek", object="opportunity"),
            ),
            "alternative": negative_chain(
                make_memory(action="strain", object="hand"),
                make_memory(action="need", object="rest"),
                make_memory(action="pause", object="practice"),
                make_memory(action="lose", object="progress"),
                make_memory(action="miss", object="performance"),
            ),
        },
        {
            "domain_id": "cook_meal",
            "field": "topic",
            "values": ["rice", "pasta", "soup", "stew", "curry"],
            "root": {"action": "cook", "object": "meal"},
            "preferred": positive_chain(
                make_memory(action="prepare", object="dish"),
                make_memory(action="serve", object="dish"),
                make_memory(action="eat", object="meal"),
                make_memory(action="feel", object="satisfied"),
                make_memory(action="save", object="money"),
            ),
            "alternative": negative_chain(
                make_memory(action="burn", object="food"),
                make_memory(action="throw", object="away"),
                make_memory(action="stay", object="hungry"),
                make_memory(action="buy", object="takeout"),
                make_memory(action="spend", object="money"),
            ),
        },
    ]

    phase_names = (
        "train_round_1",
        "train_round_2",
        "unseen_round_1",
        "unseen_round_2",
        "unseen_round_3",
    )

    scenarios: List[PlanningScenario] = []
    for round_index, phase_name in enumerate(phase_names):
        for domain in domains:
            field_name = str(domain["field"])
            slot_value = str(domain["values"][round_index])
            root_kwargs = dict(domain["root"])
            root_kwargs[field_name] = slot_value
            scenarios.append(
                PlanningScenario(
                    scenario_id=f"{phase_name}_{domain['domain_id']}",
                    phase=phase_name,
                    root=make_memory(**root_kwargs),
                    preferred=domain["preferred"],
                    alternatives=(domain["alternative"],),
                    unseen_input=round_index >= 2,
                )
            )
    return scenarios


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
        "# BrAIn Planning & Simulation V3 Benchmark",
        "",
        "This report evaluates deterministic multi-path planning, outcome scoring, and concept-backed future simulation.",
        "",
        "## Summary",
        "",
        f"- Report name: `{report.get('report_name', 'brain_planning_simulation_v3_50')}`",
        f"- Scenario count: `{aggregate.get('scenario_count', 0)}`",
        f"- Decision accuracy: `{aggregate.get('decision_accuracy', 0.0)}`",
        f"- Outcome prediction quality: `{aggregate.get('outcome_prediction_quality', 0.0)}`",
        f"- Average chain depth: `{aggregate.get('average_chain_depth', 0.0)}`",
        f"- Max chain depth: `{aggregate.get('max_chain_depth', 0)}`",
        f"- Depth increase: `{aggregate.get('depth_increase', 0.0)}`",
        f"- Accuracy first 10: `{aggregate.get('accuracy_first_10', 0.0)}`",
        f"- Accuracy last 10: `{aggregate.get('accuracy_last_10', 0.0)}`",
        f"- Unseen accuracy round 1: `{aggregate.get('unseen_accuracy_round_1', 0.0)}`",
        f"- Unseen accuracy round 3: `{aggregate.get('unseen_accuracy_round_3', 0.0)}`",
        f"- Concept usage rate: `{aggregate.get('concept_usage_rate', 0.0)}`",
        f"- Outcome profile count: `{aggregate.get('outcome_profile_count', 0)}`",
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
                f"- Selected depth: `{result.get('selected_depth', 0)}`",
                f"- Candidate path count: `{result.get('candidate_path_count', 0)}`",
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
        prog="brain_planning_simulation_v3_benchmark",
        description="Run BrAIn Planning & Simulation V3 benchmark scenarios.",
    )
    parser.add_argument(
        "--report-name",
        default="brain_planning_simulation_v3_50",
        help="Base name for the saved planning simulation V3 benchmark files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved benchmark artifacts.",
    )
    args = parser.parse_args()
    result = run_planning_simulation_v3_benchmark(report_name=args.report_name, output_dir=args.output_dir)
    print("=" * 72)
    print("BRAIN PLANNING & SIMULATION V3 BENCHMARK")
    print(f"JSON report: {result['json_path']}")
    print(f"Markdown summary: {result['markdown_path']}")
    print(f"Outcome registry: {result['outcome_registry_path']}")
    print(f"Concept registry: {result['concept_registry_path']}")
    print(f"Link registry: {result['link_registry_path']}")
    print(f"State dir: {result['state_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
