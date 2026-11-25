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


EXAMPLE_PROMPTS = [
    "eat rice safely and keep energy",
    "drink tea so I feel refreshed without risk",
    "charge laptop so I can finish my task safely",
    "repair phone and avoid damage",
    "study biology to complete my goal",
    "clean desk so I save time",
    "backup photos to avoid losing work",
    "practice guitar to improve skill without injury",
    "cook soup so I feel full and save money",
    "eat beans so I can finish my task safely",
]


def run_examples(
    report_name: str = "brain_hybrid_prototype_examples_10",
    output_dir: str | Path = Path("brain_output") / "examples",
) -> Dict[str, object]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = resolved_output_dir / f"{report_name}.state"
    if state_dir.exists():
        shutil.rmtree(state_dir)

    prototype = BrainHybridPrototype(
        state_dir=state_dir,
        storage_policy=StoragePolicy(trace_storage_mode="always"),
    )

    results: List[Dict[str, object]] = []
    for index, prompt in enumerate(EXAMPLE_PROMPTS, start=1):
        turn = prototype.run_prompt(prompt, inspect=True)
        results.append(
            {
                "index": index,
                "raw_prompt": prompt,
                "route_decision": turn.route_decision.to_dict(),
                "structured_result": turn.final_result.to_dict(),
                "retrieved_memory_count": len(turn.retrieved_memories),
                "candidate_path_count": len(turn.trace.candidate_paths),
                "selected_path": turn.trace.selected_path,
                "llm_explanation": turn.llm_explanation.user_facing_text,
                "trace": turn.trace.to_dict(),
            }
        )

    report = {
        "report_name": report_name,
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
        "# BrAIn Hybrid Prototype Example Runs",
        "",
        "These example runs demonstrate the interactive BrAIn + LLM prototype over 10 sequential prompts.",
        "",
    ]
    for result in results:
        if not isinstance(result, dict):
            continue
        lines.extend(
            [
                f"## Example {result.get('index', 0)}",
                "",
                f"- Raw prompt: `{result.get('raw_prompt', '')}`",
                f"- Route type: `{result.get('route_decision', {}).get('route_type', '')}`",
                f"- Retrieved memory count: `{result.get('retrieved_memory_count', 0)}`",
                f"- Candidate path count: `{result.get('candidate_path_count', 0)}`",
                "",
                "### Structured BrAIn Result",
                "",
                "```json",
                json.dumps(result.get("structured_result", {}), indent=2, sort_keys=True),
                "```",
                "",
                "### LLM Explanation",
                "",
                result.get("llm_explanation", ""),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="brain_hybrid_prototype_examples",
        description="Run 10 example prompts through the BrAIn hybrid prototype.",
    )
    parser.add_argument(
        "--report-name",
        default="brain_hybrid_prototype_examples_10",
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
    print("BRAIN HYBRID PROTOTYPE EXAMPLES")
    print(f"JSON report: {result['json_path']}")
    print(f"Markdown summary: {result['markdown_path']}")
    print(f"State dir: {result['state_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
