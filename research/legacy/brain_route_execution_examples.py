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


SEED_PROMPTS = [
    "Use what you know about my goals and tell me the safest way to charge my laptop.",
    "Help me finish my task safely.",
    "Which option best fits my goals?",
]

DEMO_PROMPTS = [
    "What did I ask earlier?",
    "What was my last planning request?",
    "Help me finish my task safely.",
    "Which option best fits my goals?",
    "What did you say about charging my laptop?",
]


def run_examples(
    report_name: str = "brain_route_execution_examples",
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
    for prompt in SEED_PROMPTS:
        prototype.run_prompt(prompt)

    results: List[Dict[str, object]] = []
    for prompt in DEMO_PROMPTS:
        turn = prototype.run_prompt(prompt, inspect=True)
        results.append(
            {
                "prompt": prompt,
                "route_type": turn.route_decision.route_type,
                "route_confidence": turn.route_decision.confidence,
                "enrichment_result": turn.enrichment_result,
                "recalled_turns": [item.to_dict() for item in turn.recalled_turns],
                "retrieved_memories": [item.to_dict() for item in turn.retrieved_memories],
                "candidate_path_count": len(turn.trace.candidate_paths),
                "answer": turn.llm_explanation.user_facing_text,
            }
        )

    report = {
        "report_name": report_name,
        "seed_prompt_count": len(SEED_PROMPTS),
        "scenario_count": len(results),
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
        "state_dir": str(state_dir),
    }


def _build_markdown_report(report: Dict[str, object]) -> str:
    results = report.get("results", [])
    if not isinstance(results, list):
        results = []
    lines = [
        "# BrAIn Route Execution Examples",
        "",
        "These examples demonstrate improved episodic recall and deterministic prompt enrichment for the terminal prototype.",
        "",
    ]
    for index, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            continue
        enrichment = result.get("enrichment_result", {})
        recalled = result.get("recalled_turns", [])
        lines.extend(
            [
                f"## Example {index}",
                "",
                f"- Prompt: `{result.get('prompt', '')}`",
                f"- Route type: `{result.get('route_type', '')}`",
                f"- Candidate path count: `{result.get('candidate_path_count', 0)}`",
                f"- Needs clarification: `{enrichment.get('needs_clarification', False) if isinstance(enrichment, dict) else False}`",
                f"- Recalled turn count: `{len(recalled) if isinstance(recalled, list) else 0}`",
                "",
                "### Answer",
                "",
                str(result.get("answer", "")),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="brain_route_execution_examples",
        description="Run episodic recall and prompt-enrichment examples for the BrAIn terminal prototype.",
    )
    parser.add_argument(
        "--report-name",
        default="brain_route_execution_examples",
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
    print("BRAIN ROUTE EXECUTION EXAMPLES")
    print(f"JSON report: {result['json_path']}")
    print(f"Markdown summary: {result['markdown_path']}")
    print(f"State dir: {result['state_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
