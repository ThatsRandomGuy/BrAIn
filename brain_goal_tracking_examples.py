from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

from brain_hybrid_prototype import BrainHybridPrototype, StoragePolicy


SCENARIOS = [
    {
        "label": "task progression across multiple turns",
        "turns": [
            "Help me finish my task safely.",
            "Charging my laptop.",
            "What should I do next?",
        ],
    },
    {
        "label": "goal refinement over time",
        "turns": [
            "Which option best fits my goals?",
            "Charging my laptop or leaving it unplugged.",
            "Use what you know about my goals and tell me the safest way to charge my laptop.",
        ],
    },
    {
        "label": "suggestion after successful action",
        "turns": [
            "Use what you know about my goals and tell me the safest way to charge my laptop.",
        ],
    },
    {
        "label": "suggestion after failure",
        "turns": [
            "Help me finish my task safely.",
        ],
    },
]


def run_examples(
    report_name: str = "brain_goal_tracking_examples",
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
                    "effective_prompt_text": turn.effective_prompt_text,
                    "clarification_state": turn.clarification_state,
                    "goal_tracking": turn.goal_tracking,
                    "candidate_path_count": len(turn.trace.candidate_paths),
                    "answer": turn.llm_explanation.user_facing_text,
                }
            )
        results.append({"label": scenario["label"], "turns": turns})

    report = {
        "report_name": report_name,
        "scenario_count": len(results),
        "results": results,
        "tracked_goals": prototype.tracked_goals_summary(),
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
        "# BrAIn Goal Tracking V6 Examples",
        "",
        "These examples demonstrate deterministic goal persistence, progress tracking, and proactive next-step suggestions.",
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
            suggestion = turn.get("goal_tracking", {}).get("next_step_suggestion", {})
            progress = turn.get("goal_tracking", {}).get("progress_tracking", {})
            lines.extend(
                [
                    f"### Turn {index}",
                    "",
                    f"- User text: `{turn.get('user_text', '')}`",
                    f"- Route type: `{turn.get('route_type', '')}`",
                    f"- Candidate path count: `{turn.get('candidate_path_count', 0)}`",
                    f"- Progress status: `{progress.get('status', '') if isinstance(progress, dict) else ''}`",
                    f"- Next step suggestion: `{suggestion.get('text', '') if isinstance(suggestion, dict) else ''}`",
                    "",
                    turn.get("answer", ""),
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="brain_goal_tracking_examples",
        description="Run BrAIn V6 goal tracking and proactive suggestion examples.",
    )
    parser.add_argument(
        "--report-name",
        default="brain_goal_tracking_examples",
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
    print("BRAIN GOAL TRACKING EXAMPLES")
    print(f"JSON report: {result['json_path']}")
    print(f"Markdown summary: {result['markdown_path']}")
    print(f"State dir: {result['state_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
