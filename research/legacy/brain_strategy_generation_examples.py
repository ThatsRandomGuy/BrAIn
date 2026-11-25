from __future__ import annotations

# Historical harnesses run with the maintained core on the import path.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]/"core"))

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

from brain_hybrid_prototype import BrainHybridPrototype, StoragePolicy


SCENARIOS = [
    {
        "label": "device safety scenario",
        "turns": [
            "Use what you know about my goals and tell me the safest way to charge my laptop.",
        ],
    },
    {
        "label": "task completion scenario",
        "turns": [
            "Help me finish my task safely.",
            "Charging my laptop before I leave.",
        ],
    },
    {
        "label": "goal tradeoff scenario",
        "turns": [
            "Which option best fits my goals?",
            "Charging my laptop now or waiting until later.",
        ],
    },
    {
        "label": "failure recovery scenario",
        "turns": [
            "My laptop charge attempt failed. Help me recover safely.",
        ],
    },
]


def run_examples(
    report_name: str = "brain_strategy_generation_examples",
    output_dir: str | Path = Path("brain_output") / "examples",
) -> Dict[str, object]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = resolved_output_dir / f"{report_name}.state"
    if state_dir.exists():
        shutil.rmtree(state_dir)

    prototype = BrainHybridPrototype(
        state_dir=state_dir,
        storage_policy=StoragePolicy.for_mode("store_all_prompts"),
    )

    results: List[Dict[str, object]] = []
    for scenario in SCENARIOS:
        turns: List[Dict[str, object]] = []
        for user_text in scenario["turns"]:
            turn = prototype.process_user_turn(user_text, inspect=True)
            turns.append(
                {
                    "user_text": user_text,
                    "route_type": turn.route_decision.route_type,
                    "clarification_state": turn.clarification_state,
                    "candidate_path_count": len(turn.trace.candidate_paths),
                    "goal_tracking": turn.goal_tracking,
                    "strategy_plan": turn.strategy_plan,
                    "answer": turn.llm_explanation.user_facing_text,
                }
            )
        results.append({"label": scenario["label"], "turns": turns})

    report = {
        "report_name": report_name,
        "scenario_count": len(results),
        "results": results,
        "tracked_strategies": prototype.tracked_strategies_summary(),
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


def _build_markdown_report(report: Dict[str, object]) -> str:
    scenarios = report.get("results", [])
    if not isinstance(scenarios, list):
        scenarios = []
    lines = [
        "# BrAIn Strategy Generation V7 Examples",
        "",
        "These examples demonstrate deterministic multi-step strategy generation, action decomposition, and alternative strategy selection.",
        "",
    ]
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        lines.append(f"## {scenario.get('label', 'Scenario')}")
        lines.append("")
        for index, turn in enumerate(scenario.get("turns", []), start=1):
            if not isinstance(turn, dict):
                continue
            strategy = turn.get("strategy_plan", {}).get("selected_strategy", {})
            alternatives = turn.get("strategy_plan", {}).get("alternative_strategies", [])
            lines.extend(
                [
                    f"### Turn {index}",
                    "",
                    f"- User text: `{turn.get('user_text', '')}`",
                    f"- Route type: `{turn.get('route_type', '')}`",
                    f"- Candidate path count: `{turn.get('candidate_path_count', 0)}`",
                    f"- Selected strategy: `{strategy.get('name', '') if isinstance(strategy, dict) else ''}`",
                    f"- Strategy focus: `{strategy.get('focus', '') if isinstance(strategy, dict) else ''}`",
                    f"- Alternative count: `{len(alternatives) if isinstance(alternatives, list) else 0}`",
                    "",
                    turn.get("strategy_plan", {}).get("explanation", ""),
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="brain_strategy_generation_examples",
        description="Run BrAIn V7 strategy generation examples.",
    )
    parser.add_argument(
        "--report-name",
        default="brain_strategy_generation_examples",
        help="Base name for saved example outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("brain_output") / "examples"),
        help="Directory for saved example artifacts.",
    )
    args = parser.parse_args()
    result = run_examples(report_name=args.report_name, output_dir=args.output_dir)
    print("=" * 72)
    print("BRAIN STRATEGY GENERATION EXAMPLES")
    print(f"JSON report: {result['json_path']}")
    print(f"Markdown summary: {result['markdown_path']}")
    print(f"State dir: {result['state_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
