from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from brain_hybrid_prototype import BrainHybridPrototype, StoragePolicy


ROUTE_DEMO_SCENARIOS = [
    {
        "label": "factual question",
        "prompt": "What color is the sky?",
        "expected_route": "factual_question",
    },
    {
        "label": "memory question",
        "prompt": "What did I ask earlier?",
        "expected_route": "memory_question",
    },
    {
        "label": "planning request",
        "prompt": "Help me finish my task safely.",
        "expected_route": "planning_request",
    },
    {
        "label": "goal-driven prompt",
        "prompt": "Which option best fits my goals?",
        "expected_route": "goal_request",
    },
    {
        "label": "hybrid prompt",
        "prompt": "Use what you know about my goals and tell me the safest way to charge my laptop.",
        "expected_route": "hybrid_request",
    },
]


def run_route_examples(
    output_path: str | Path = Path("brain_output") / "examples" / "brain_query_router_examples.json",
) -> Dict[str, object]:
    prototype = BrainHybridPrototype(
        state_dir=Path("brain_output") / "examples" / "brain_query_router_examples.state",
        storage_policy=StoragePolicy.for_mode("store_disabled"),
    )
    results: List[Dict[str, object]] = []
    for scenario in ROUTE_DEMO_SCENARIOS:
        turn = prototype.run_prompt(scenario["prompt"], inspect=True)
        results.append(
            {
                "label": scenario["label"],
                "prompt": scenario["prompt"],
                "expected_route": scenario["expected_route"],
                "observed_route": turn.route_decision.route_type,
                "confidence": turn.route_decision.confidence,
                "reasons": turn.route_decision.reasons,
                "retrieved_memory_count": len(turn.retrieved_memories),
                "candidate_path_count": len(turn.trace.candidate_paths),
                "answer": turn.llm_explanation.user_facing_text,
            }
        )
    payload = {"scenario_count": len(results), "results": results}
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = resolved_output_path.with_suffix(".md")
    markdown_path.write_text(_build_markdown(payload), encoding="utf-8")
    return {
        "json_path": str(resolved_output_path),
        "markdown_path": str(markdown_path),
        "report": payload,
    }


def _build_markdown(payload: Dict[str, object]) -> str:
    results = payload.get("results", [])
    if not isinstance(results, list):
        results = []
    lines = [
        "# BrAIn Query Router Examples",
        "",
        "These examples show the expected route for the terminal prototype after deterministic query classification.",
        "",
    ]
    for result in results:
        if not isinstance(result, dict):
            continue
        lines.extend(
            [
                f"## {result.get('label', 'Scenario')}",
                "",
                f"- Prompt: `{result.get('prompt', '')}`",
                f"- Expected route: `{result.get('expected_route', '')}`",
                f"- Observed route: `{result.get('observed_route', '')}`",
                f"- Confidence: `{result.get('confidence', 0)}`",
                f"- Retrieved memory count: `{result.get('retrieved_memory_count', 0)}`",
                f"- Candidate path count: `{result.get('candidate_path_count', 0)}`",
                f"- Reasons: `{'; '.join(result.get('reasons', []))}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="brain_query_router_examples",
        description="Run route demo scenarios against the terminal BrAIn prototype.",
    )
    parser.add_argument(
        "--output-path",
        default=str(Path("brain_output") / "examples" / "brain_query_router_examples.json"),
        help="JSON path for saved route example results.",
    )
    args = parser.parse_args()
    result = run_route_examples(output_path=args.output_path)
    print("=" * 72)
    print("BRAIN QUERY ROUTER EXAMPLES")
    print(f"JSON report: {result['json_path']}")
    print(f"Markdown summary: {result['markdown_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
