from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from brain_codec import BrainCodec, BrainMemory
from brain_concept_abstraction_v2 import BrainConceptAbstractionV2
from brain_input_v1 import BrainInputV1
from brain_reasoning_engine import DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class ConceptScenario:
    scenario_id: str
    phase: str
    root: BrainMemory
    expected_chain: Tuple[BrainMemory, ...]
    relation_types: Tuple[str, ...]
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


def run_concept_abstraction_v2_benchmark(
    report_name: str = "brain_concept_abstraction_v2_50",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, object]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = resolved_output_dir / f"{report_name}.state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    engine = BrainConceptAbstractionV2(state_dir=state_dir)

    scenarios = _build_scenarios()
    if len(scenarios) != 50:
        raise ValueError(f"Concept Abstraction V2 benchmark must contain exactly 50 scenarios, found {len(scenarios)}.")

    results: List[Dict[str, object]] = []
    correct_count = 0
    chain_depths: List[int] = []

    for index, scenario in enumerate(scenarios, start=1):
        input_brain = _memory_to_brain_input(engine, scenario.root, current_sequence=index)
        expected_chain = tuple(
            _memory_to_brain_input(engine, memory, current_sequence=index)
            for memory in scenario.expected_chain
        )
        prediction = engine.simulate(
            input_brain,
            max_depth=4,
            beam_width=4,
            current_sequence=index,
        )
        validation = engine.validate_prediction(
            input_brain=input_brain,
            prediction=prediction,
            expected_chain=expected_chain,
            relation_types=scenario.relation_types,
            current_sequence=index,
        )
        success = bool(validation.get("success", False))
        if success:
            correct_count += 1
        selected_depth = prediction.selected_path.depth if prediction.selected_path is not None else 0
        chain_depths.append(selected_depth)
        results.append(
            {
                "index": index,
                "scenario_id": scenario.scenario_id,
                "phase": scenario.phase,
                "unseen_input": scenario.unseen_input,
                "prediction": prediction.to_dict(),
                "validation": validation,
                "selected_depth": selected_depth,
                "success": success,
            }
        )

    first_10 = results[:10]
    last_10 = results[-10:]
    unseen_results = [result for result in results if bool(result.get("unseen_input", False))]
    unseen_round_1 = [result for result in results if result.get("phase") == "unseen_round_1"]
    unseen_round_3 = [result for result in results if result.get("phase") == "unseen_round_3"]
    concept_used_count = sum(1 for result in results if bool(result.get("validation", {}).get("used_concept_abstraction", False)))
    unseen_concept_used_count = sum(
        1 for result in unseen_results if bool(result.get("validation", {}).get("used_concept_abstraction", False))
    )
    generalization_success_count = sum(1 for result in unseen_results if bool(result.get("success", False)))

    aggregate = {
        "scenario_count": len(results),
        "overall_accuracy": round(correct_count / max(1, len(results)), 4),
        "generalization_success_rate": round(generalization_success_count / max(1, len(unseen_results)), 4),
        "average_chain_depth": round(sum(chain_depths) / max(1, len(chain_depths)), 4),
        "max_chain_depth": max(chain_depths) if chain_depths else 0,
        "average_depth_first_10": round(sum(result["selected_depth"] for result in first_10) / max(1, len(first_10)), 4),
        "average_depth_last_10": round(sum(result["selected_depth"] for result in last_10) / max(1, len(last_10)), 4),
        "depth_increase": round(
            (sum(result["selected_depth"] for result in last_10) / max(1, len(last_10)))
            - (sum(result["selected_depth"] for result in first_10) / max(1, len(first_10))),
            4,
        ),
        "unseen_accuracy_round_1": round(sum(1 for result in unseen_round_1 if result["success"]) / max(1, len(unseen_round_1)), 4),
        "unseen_accuracy_round_3": round(sum(1 for result in unseen_round_3 if result["success"]) / max(1, len(unseen_round_3)), 4),
        "unseen_accuracy_improvement": round(
            (sum(1 for result in unseen_round_3 if result["success"]) / max(1, len(unseen_round_3)))
            - (sum(1 for result in unseen_round_1 if result["success"]) / max(1, len(unseen_round_1))),
            4,
        ),
        "concept_usage_rate": round(concept_used_count / max(1, len(results)), 4),
        "unseen_concept_usage_rate": round(unseen_concept_used_count / max(1, len(unseen_results)), 4),
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
        "concept_registry_path": str(state_dir / "brain_concepts.json"),
        "link_registry_path": str(state_dir / "v1" / "brain_reasoning_links.json"),
        "state_dir": str(state_dir),
    }


def _build_scenarios() -> List[ConceptScenario]:
    domains = [
        {
            "domain_id": "eat_food",
            "field": "object",
            "values": ["apple", "bread", "rice", "beans", "soup"],
            "root": {"action": "eat"},
            "expected_chain": (
                make_memory(action="feel", object="full"),
                make_memory(action="stop", object="hunger"),
            ),
            "relation_types": ("cause_effect", "cause_effect"),
        },
        {
            "domain_id": "drink_liquid",
            "field": "object",
            "values": ["water", "juice", "tea", "milk", "smoothie"],
            "root": {"action": "drink"},
            "expected_chain": (
                make_memory(action="feel", object="refreshed"),
                make_memory(action="think", object="clearly"),
            ),
            "relation_types": ("cause_effect", "action_outcome"),
        },
        {
            "domain_id": "charge_device",
            "field": "object",
            "values": ["phone", "tablet", "laptop", "watch", "camera"],
            "root": {"action": "charge"},
            "expected_chain": (
                make_memory(action="fill", object="battery"),
                make_memory(action="use", object="device"),
            ),
            "relation_types": ("action_outcome", "action_outcome"),
        },
        {
            "domain_id": "water_plants",
            "field": "object",
            "values": ["roses", "tomatoes", "herbs", "trees", "flowers"],
            "root": {"action": "water"},
            "expected_chain": (
                make_memory(action="grow", object="plant"),
                make_memory(action="harvest", object="produce"),
            ),
            "relation_types": ("cause_effect", "action_outcome"),
        },
        {
            "domain_id": "repair_device",
            "field": "object",
            "values": ["phone", "laptop", "tablet", "camera", "console"],
            "root": {"action": "repair"},
            "expected_chain": (
                make_memory(action="restore", object="function"),
                make_memory(action="finish", object="task"),
            ),
            "relation_types": ("cause_effect", "action_outcome"),
        },
        {
            "domain_id": "study_subject",
            "field": "topic",
            "values": ["math", "history", "biology", "physics", "chemistry"],
            "root": {"action": "study", "object": "lessons"},
            "expected_chain": (
                make_memory(action="pass", object="exam"),
                make_memory(action="advance", object="course"),
            ),
            "relation_types": ("action_outcome", "action_outcome"),
        },
        {
            "domain_id": "clean_surface",
            "field": "topic",
            "values": ["counter", "floor", "window", "desk", "shelf"],
            "root": {"action": "clean", "object": "surface"},
            "expected_chain": (
                make_memory(action="become", object="tidy"),
                make_memory(action="organize", object="room"),
            ),
            "relation_types": ("cause_effect", "action_outcome"),
        },
        {
            "domain_id": "backup_data",
            "field": "topic",
            "values": ["photos", "videos", "documents", "music", "projects"],
            "root": {"action": "backup", "object": "data"},
            "expected_chain": (
                make_memory(action="preserve", object="files"),
                make_memory(action="restore", object="files"),
            ),
            "relation_types": ("action_outcome", "action_outcome"),
        },
        {
            "domain_id": "practice_skill",
            "field": "topic",
            "values": ["piano", "guitar", "violin", "drums", "flute"],
            "root": {"action": "practice", "object": "skill"},
            "expected_chain": (
                make_memory(action="improve", object="skill"),
                make_memory(action="perform", object="piece"),
            ),
            "relation_types": ("cause_effect", "action_outcome"),
        },
        {
            "domain_id": "cook_meal",
            "field": "topic",
            "values": ["rice", "pasta", "soup", "stew", "curry"],
            "root": {"action": "cook", "object": "meal"},
            "expected_chain": (
                make_memory(action="prepare", object="dish"),
                make_memory(action="serve", object="dish"),
            ),
            "relation_types": ("action_outcome", "action_outcome"),
        },
    ]

    phase_names = (
        "train_round_1",
        "train_round_2",
        "unseen_round_1",
        "unseen_round_2",
        "unseen_round_3",
    )

    scenarios: List[ConceptScenario] = []
    for round_index, phase_name in enumerate(phase_names):
        for domain in domains:
            field_name = str(domain["field"])
            slot_value = str(domain["values"][round_index])
            root_kwargs = dict(domain["root"])
            root_kwargs[field_name] = slot_value
            root = make_memory(**root_kwargs)
            scenarios.append(
                ConceptScenario(
                    scenario_id=f"{phase_name}_{domain['domain_id']}",
                    phase=phase_name,
                    root=root,
                    expected_chain=tuple(domain["expected_chain"]),
                    relation_types=tuple(domain["relation_types"]),
                    unseen_input=round_index >= 2,
                )
            )
    return scenarios


def _memory_to_brain_input(
    engine: BrainConceptAbstractionV2,
    memory: BrainMemory,
    current_sequence: int,
) -> BrainInputV1:
    base = engine.v1_engine.runtime.brain_input_codec.build_brain_input(
        memory=memory,
        encoding_type="bitpacked",
        primitive_ids=[],
    )
    return engine.v1_engine.native_engine.normalize(base, current_sequence=current_sequence)


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
        "unseen_round_3_eat_food",
        "unseen_round_3_cook_meal",
    }
    showcased = [result for result in results if isinstance(result, dict) and result.get("scenario_id") in showcase_ids]
    lines = [
        "# BrAIn Concept Abstraction V2 Benchmark",
        "",
        "This report evaluates emergent concept nodes, generalized links, and multi-hop conceptual reasoning.",
        "",
        "## Summary",
        "",
        f"- Report name: `{report.get('report_name', 'brain_concept_abstraction_v2_50')}`",
        f"- Scenario count: `{aggregate.get('scenario_count', 0)}`",
        f"- Overall accuracy: `{aggregate.get('overall_accuracy', 0.0)}`",
        f"- Generalization success rate: `{aggregate.get('generalization_success_rate', 0.0)}`",
        f"- Average chain depth: `{aggregate.get('average_chain_depth', 0.0)}`",
        f"- Depth increase: `{aggregate.get('depth_increase', 0.0)}`",
        f"- Unseen accuracy round 1: `{aggregate.get('unseen_accuracy_round_1', 0.0)}`",
        f"- Unseen accuracy round 3: `{aggregate.get('unseen_accuracy_round_3', 0.0)}`",
        f"- Unseen accuracy improvement: `{aggregate.get('unseen_accuracy_improvement', 0.0)}`",
        f"- Concept count: `{aggregate.get('concept_count', 0)}`",
        f"- Concept link count: `{aggregate.get('concept_link_count', 0)}`",
        f"- Concept usage rate: `{aggregate.get('concept_usage_rate', 0.0)}`",
        "",
        "## Example Predictions",
        "",
    ]
    for result in showcased:
        lines.extend(
            [
                f"### {result.get('scenario_id', 'scenario')}",
                "",
                f"- Success: `{result.get('success', False)}`",
                f"- Selected depth: `{result.get('selected_depth', 0)}`",
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
        prog="brain_concept_abstraction_v2_benchmark",
        description="Run BrAIn Concept Abstraction V2 benchmark scenarios.",
    )
    parser.add_argument(
        "--report-name",
        default="brain_concept_abstraction_v2_50",
        help="Base name for the saved concept abstraction V2 benchmark files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved benchmark artifacts.",
    )
    args = parser.parse_args()
    result = run_concept_abstraction_v2_benchmark(report_name=args.report_name, output_dir=args.output_dir)
    print("=" * 72)
    print("BRAIN CONCEPT ABSTRACTION V2 BENCHMARK")
    print(f"JSON report: {result['json_path']}")
    print(f"Markdown summary: {result['markdown_path']}")
    print(f"Concept registry: {result['concept_registry_path']}")
    print(f"Link registry: {result['link_registry_path']}")
    print(f"State dir: {result['state_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
