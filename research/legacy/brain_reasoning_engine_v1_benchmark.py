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
from typing import Dict, List, Sequence, Tuple

from brain_codec import BrainCodec, BrainMemory
from brain_input_v1 import BrainInputV1
from brain_reasoning_engine import DEFAULT_OUTPUT_DIR
from brain_reasoning_engine_v1 import BrainReasoningEngineV1


@dataclass(frozen=True)
class CausalScenario:
    scenario_id: str
    root: BrainMemory
    expected_chain: Tuple[BrainMemory, ...]
    relation_types: Tuple[str, ...]


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


def run_reasoning_engine_v1_benchmark(
    report_name: str = "brain_reasoning_engine_v1_50",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, object]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = resolved_output_dir / f"{report_name}.state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    engine = BrainReasoningEngineV1(state_dir=state_dir)

    scenarios = _build_scenarios()
    if len(scenarios) != 50:
        raise ValueError(f"Reasoning Engine V1 benchmark must contain exactly 50 scenarios, found {len(scenarios)}.")
    _seed_broad_links(engine)

    results: List[Dict[str, object]] = []
    validation_events: List[Dict[str, object]] = []
    chain_depths: List[int] = []
    correct_count = 0

    for index, scenario in enumerate(scenarios, start=1):
        input_brain = _memory_to_brain_input(engine, scenario.root, current_sequence=index)
        prediction = engine.simulate(
            input_brain,
            max_depth=3,
            beam_width=3,
            current_sequence=index,
        )
        event = engine.validate_prediction(
            input_brain=input_brain,
            prediction=prediction,
            expected_chain=tuple(_memory_to_brain_input(engine, memory, current_sequence=index) for memory in scenario.expected_chain),
            relation_types=scenario.relation_types,
            current_sequence=index,
        )
        success = bool(event.get("success", False))
        if success:
            correct_count += 1
        selected_depth = prediction.selected_path.depth if prediction.selected_path is not None else 0
        chain_depths.append(selected_depth)
        results.append(
            {
                "index": index,
                "scenario_id": scenario.scenario_id,
                "prediction": prediction.to_dict(),
                "validation": event,
                "selected_depth": selected_depth,
                "success": success,
            }
        )
        validation_events.append({"index": index, "scenario_id": scenario.scenario_id, **event})

    summary_before = {
        "accuracy_first_10": round(sum(1 for item in results[:10] if item["success"]) / max(1, len(results[:10])), 4),
        "average_depth_first_10": round(sum(item["selected_depth"] for item in results[:10]) / max(1, len(results[:10])), 4),
    }
    summary_after = {
        "accuracy_last_10": round(sum(1 for item in results[-10:] if item["success"]) / max(1, len(results[-10:])), 4),
        "average_depth_last_10": round(sum(item["selected_depth"] for item in results[-10:]) / max(1, len(results[-10:])), 4),
    }
    registry_summary = engine.registry_summary()
    aggregate = {
        "scenario_count": len(results),
        "overall_accuracy": round(correct_count / max(1, len(results)), 4),
        "accuracy_improvement": round(summary_after["accuracy_last_10"] - summary_before["accuracy_first_10"], 4),
        "average_chain_depth": round(sum(chain_depths) / max(1, len(chain_depths)), 4),
        "max_chain_depth": max(chain_depths) if chain_depths else 0,
        **summary_before,
        **summary_after,
        **registry_summary,
    }

    report = {
        "report_name": report_name,
        "aggregate": aggregate,
        "validation_events": validation_events,
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
        "link_registry_path": str(state_dir / "brain_reasoning_links.json"),
        "state_dir": str(state_dir),
    }


def _build_scenarios() -> List[CausalScenario]:
    base = [
        CausalScenario(
            scenario_id="eat_spoiled_food",
            root=make_memory(action="eat", object="food", topic="spoiled"),
            expected_chain=(make_memory(action="get", object="sick"), make_memory(action="miss", object="work")),
            relation_types=("cause_effect", "action_outcome"),
        ),
        CausalScenario(
            scenario_id="study_daily_lessons",
            root=make_memory(intent="improve", action="study", object="lessons", topic="daily"),
            expected_chain=(make_memory(action="pass", object="exam"), make_memory(action="graduate", object="program")),
            relation_types=("action_outcome", "cause_effect"),
        ),
        CausalScenario(
            scenario_id="water_plants",
            root=make_memory(action="water", object="plants"),
            expected_chain=(make_memory(action="grow", object="plants"), make_memory(action="harvest", object="fruit")),
            relation_types=("cause_effect", "action_outcome"),
        ),
        CausalScenario(
            scenario_id="ignore_alarm",
            root=make_memory(action="ignore", object="alarm", topic="morning"),
            expected_chain=(make_memory(action="arrive", object="late"), make_memory(action="miss", object="meeting")),
            relation_types=("cause_effect", "action_outcome"),
        ),
        CausalScenario(
            scenario_id="charge_battery",
            root=make_memory(action="charge", object="battery"),
            expected_chain=(make_memory(action="fill", object="battery"), make_memory(action="start", object="device")),
            relation_types=("action_outcome", "cause_effect"),
        ),
        CausalScenario(
            scenario_id="exercise_regularly",
            root=make_memory(intent="improve", action="exercise", object="body", topic="regularly"),
            expected_chain=(make_memory(action="become", object="stronger"), make_memory(action="win", object="match")),
            relation_types=("cause_effect", "action_outcome"),
        ),
        CausalScenario(
            scenario_id="save_money",
            root=make_memory(intent="prepare", action="save", object="money"),
            expected_chain=(make_memory(action="afford", object="purchase"), make_memory(action="buy", object="item")),
            relation_types=("action_outcome", "action_outcome"),
        ),
        CausalScenario(
            scenario_id="spill_water_on_device",
            root=make_memory(action="spill", object="water", topic="device"),
            expected_chain=(make_memory(action="damage", object="device"), make_memory(action="replace", object="device")),
            relation_types=("cause_effect", "action_outcome"),
        ),
        CausalScenario(
            scenario_id="follow_recipe",
            root=make_memory(intent="cook", action="follow", object="recipe", topic="dinner"),
            expected_chain=(make_memory(action="cook", object="meal"), make_memory(action="eat", object="meal")),
            relation_types=("intent_action", "action_outcome"),
        ),
        CausalScenario(
            scenario_id="forget_umbrella_in_rain",
            root=make_memory(action="forget", object="umbrella", topic="rain"),
            expected_chain=(make_memory(action="get", object="wet"), make_memory(action="catch", object="cold")),
            relation_types=("cause_effect", "cause_effect"),
        ),
    ]
    scenarios: List[CausalScenario] = []
    for round_index in range(5):
        for scenario in base:
            scenarios.append(
                CausalScenario(
                    scenario_id=f"r{round_index + 1:02d}_{scenario.scenario_id}",
                    root=scenario.root,
                    expected_chain=scenario.expected_chain,
                    relation_types=scenario.relation_types,
                )
            )
    return scenarios


def _seed_broad_links(engine: BrainReasoningEngineV1) -> None:
    seed_pairs = [
        (make_memory(action="eat", object="food"), make_memory(action="feel", object="full"), "cause_effect"),
        (make_memory(action="study", object="lessons"), make_memory(action="gain", object="knowledge"), "action_outcome"),
        (make_memory(action="ignore", object="alarm"), make_memory(action="rush", object="morning"), "cause_effect"),
        (make_memory(action="spill", object="water"), make_memory(action="wipe", object="device"), "action_outcome"),
        (make_memory(action="follow", object="recipe"), make_memory(action="mix", object="ingredients"), "intent_action"),
    ]
    for index, (antecedent, consequent, relation_type) in enumerate(seed_pairs, start=1):
        link = engine.learn_link(
            antecedent=_memory_to_brain_input(engine, antecedent, current_sequence=index),
            consequent=_memory_to_brain_input(engine, consequent, current_sequence=index),
            relation_type=relation_type,
            current_sequence=index,
            notes=["seed_broad_link"],
        )
        engine.link_registry.update_usage(link.link_id, current_sequence=index, success=False)


def _memory_to_brain_input(
    engine: BrainReasoningEngineV1,
    memory: BrainMemory,
    current_sequence: int,
) -> BrainInputV1:
    base = engine.runtime.brain_input_codec.build_brain_input(
        memory=memory,
        encoding_type="bitpacked",
        primitive_ids=[],
    )
    return engine.native_engine.normalize(base, current_sequence=current_sequence)


def _build_markdown_report(report: Dict[str, object]) -> str:
    aggregate = report.get("aggregate", {})
    results = report.get("results", [])
    if not isinstance(aggregate, dict):
        aggregate = {}
    if not isinstance(results, list):
        results = []
    lines = [
        "# BrAIn Reasoning Engine V1 Benchmark",
        "",
        "This report evaluates multi-step causal reasoning over stored patterns and relationships.",
        "",
        "## Summary",
        "",
        f"- Report name: `{report.get('report_name', 'brain_reasoning_engine_v1_50')}`",
        f"- Scenario count: `{aggregate.get('scenario_count', 0)}`",
        f"- Overall accuracy: `{aggregate.get('overall_accuracy', 0.0)}`",
        f"- Accuracy first 10: `{aggregate.get('accuracy_first_10', 0.0)}`",
        f"- Accuracy last 10: `{aggregate.get('accuracy_last_10', 0.0)}`",
        f"- Accuracy improvement: `{aggregate.get('accuracy_improvement', 0.0)}`",
        f"- Average chain depth: `{aggregate.get('average_chain_depth', 0.0)}`",
        f"- Max chain depth: `{aggregate.get('max_chain_depth', 0)}`",
        f"- Active link count: `{aggregate.get('active_link_count', 0)}`",
        f"- Average link reliability: `{aggregate.get('average_reliability', 0.0)}`",
        "",
        "## Example Predictions",
        "",
    ]
    for result in list(results[:3]) + list(results[-3:]):
        if not isinstance(result, dict):
            continue
        lines.extend(
            [
                f"### {result.get('scenario_id', 'scenario')}",
                "",
                f"- Success: `{result.get('success', False)}`",
                f"- Selected depth: `{result.get('selected_depth', 0)}`",
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
        prog="brain_reasoning_engine_v1_benchmark",
        description="Run BrAIn Reasoning Engine V1 benchmark scenarios.",
    )
    parser.add_argument(
        "--report-name",
        default="brain_reasoning_engine_v1_50",
        help="Base name for the saved reasoning engine V1 benchmark files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved benchmark artifacts.",
    )
    args = parser.parse_args()
    result = run_reasoning_engine_v1_benchmark(report_name=args.report_name, output_dir=args.output_dir)
    print("=" * 72)
    print("BRAIN REASONING ENGINE V1 BENCHMARK")
    print(f"JSON report: {result['json_path']}")
    print(f"Markdown summary: {result['markdown_path']}")
    print(f"Link registry: {result['link_registry_path']}")
    print(f"State dir: {result['state_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
