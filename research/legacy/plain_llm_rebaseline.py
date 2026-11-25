from __future__ import annotations

# Historical harnesses run with the maintained core on the import path.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]/"core"))

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from brain_vs_llm_scoring import (
    BENCHMARK_CATEGORIES,
    aggregate_scores,
    comparison_score,
    detect_invalid_reason,
    score_llm_output,
)


def _clamp_text(text: str, limit: int = 500) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def _strip_qwen_thinking(text: str) -> str:
    stripped = str(text or "").strip()
    without_thinking = re.sub(r"(?is)<think>.*?</think>", "", stripped).strip()
    if without_thinking:
        return without_thinking
    fallback = re.sub(r"(?is)</?think>", "", stripped).strip()
    return fallback


class PlainOllamaBaseline:
    def __init__(
        self,
        *,
        model: str,
        host: str = "http://localhost:11434",
        timeout_seconds: float = 180.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def answer_turns(self, turns: Sequence[str]) -> Dict[str, object]:
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "/no_think\n"
                    "You are a plain LLM baseline for a BrAIn stress test. "
                    "Answer the user's latest message directly and concisely. "
                    "Use only the conversation shown here. Do not mention this benchmark."
                ),
            }
        ]
        turn_outputs: List[Dict[str, object]] = []
        last_answer = ""
        last_error = ""

        for turn_index, turn_text in enumerate(turns, start=1):
            messages.append({"role": "user", "content": str(turn_text)})
            result = self._chat(messages)
            answer = str(result.get("answer", "")).strip()
            error = str(result.get("error", "")).strip()
            metadata = {
                "mode": "plain_ollama",
                "model": self.model,
                "host": self.host,
                "turn_index": turn_index,
            }
            if error:
                metadata["error"] = error
            turn_outputs.append(
                {
                    "turn_index": turn_index,
                    "user_text": str(turn_text),
                    "raw_text_response": answer,
                    "metadata": metadata,
                }
            )
            if error:
                last_error = error
                break
            messages.append({"role": "assistant", "content": answer})
            last_answer = answer

        return {
            "raw_text_response": last_answer,
            "provider": "plain_ollama",
            "metadata": {
                "mode": "plain_ollama",
                "model": self.model,
                "host": self.host,
                "error": last_error,
            },
            "turn_outputs": turn_outputs,
        }

    def _chat(self, messages: Sequence[Dict[str, str]]) -> Dict[str, str]:
        request_payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 512,
            },
        }
        request = urllib.request.Request(
            url=f"{self.host}/api/chat",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
            payload = json.loads(raw)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            return {"answer": "", "error": str(exc)}
        message = payload.get("message", {}) if isinstance(payload, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        return {"answer": _strip_qwen_thinking(str(content)), "error": ""}


def _invalid_bits(case: Dict[str, object]) -> str:
    bits = []
    brain_invalid = case.get("brain_invalid_reason")
    llm_invalid = case.get("llm_invalid_reason")
    if brain_invalid:
        bits.append(f"brain:{brain_invalid}")
    if llm_invalid:
        bits.append(f"llm:{llm_invalid}")
    return "; ".join(bits)


def _summary_for_cases(case_results: Sequence[Dict[str, object]], *, model: str) -> Dict[str, object]:
    aggregate = aggregate_scores(case_results)
    invalid_cases = [item for item in case_results if not item.get("comparison_eligible", False)]
    eligible_cases = [item for item in case_results if item.get("comparison_eligible", False)]
    by_category_summary: Dict[str, object] = {}
    for category in BENCHMARK_CATEGORIES:
        category_cases = [item for item in case_results if item.get("benchmark_category") == category]
        category_invalid = [item for item in category_cases if not item.get("comparison_eligible", False)]
        category_eligible = [item for item in category_cases if item.get("comparison_eligible", False)]
        category_aggregate = aggregate.get("by_category", {}).get(category, {})
        by_category_summary[category] = {
            "total_cases": len(category_cases),
            "eligible_cases": len(category_eligible),
            "invalid_cases": len(category_invalid),
            "wins": category_aggregate.get("wins", {"brain": 0, "llm": 0, "tie": 0}),
            "average_scores": category_aggregate.get("average_scores", {}),
        }
    return {
        "provider": f"plain_ollama:{model}",
        "total_cases": len(case_results),
        "eligible_comparison_cases": len(eligible_cases),
        "invalid_cases": len(invalid_cases),
        "results_by_category": by_category_summary,
        "aggregate_scores": aggregate.get("overall", {}),
    }


def _render_markdown(payload: Dict[str, object], *, source_path: Path, output_path: Path) -> str:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary", {}), dict) else {}
    aggregate = summary.get("aggregate_scores", {}) if isinstance(summary.get("aggregate_scores", {}), dict) else {}
    wins = aggregate.get("wins", {}) if isinstance(aggregate.get("wins", {}), dict) else {}
    averages = aggregate.get("average_scores", {}) if isinstance(aggregate.get("average_scores", {}), dict) else {}
    brain_avg = averages.get("brain", {}) if isinstance(averages.get("brain", {}), dict) else {}
    llm_avg = averages.get("llm", {}) if isinstance(averages.get("llm", {}), dict) else {}
    cases = payload.get("cases", []) if isinstance(payload.get("cases", []), list) else []

    lines = [
        "# BrAIn vs Plain Qwen3:8b Stress Test",
        "",
        f"- Source BrAIn run: `{source_path.as_posix()}`",
        f"- Rebaselined output: `{output_path.as_posix()}`",
        f"- Total cases: `{summary.get('total_cases', 0)}`",
        f"- Eligible comparisons: `{summary.get('eligible_comparison_cases', 0)}`",
        f"- Invalid cases: `{summary.get('invalid_cases', 0)}`",
        f"- Wins: BrAIn `{wins.get('brain', 0)}`, plain Qwen `{wins.get('llm', 0)}`, ties `{wins.get('tie', 0)}`",
        f"- Average comparison score: BrAIn `{brain_avg.get('comparison_score')}`, plain Qwen `{llm_avg.get('comparison_score')}`",
        "",
        "## Category Summary",
        "",
        "| Category | Cases | Eligible | BrAIn Wins | Qwen Wins | Ties | BrAIn Avg | Qwen Avg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    by_category = summary.get("results_by_category", {}) if isinstance(summary.get("results_by_category", {}), dict) else {}
    for category in BENCHMARK_CATEGORIES:
        item = by_category.get(category, {}) if isinstance(by_category.get(category, {}), dict) else {}
        category_wins = item.get("wins", {}) if isinstance(item.get("wins", {}), dict) else {}
        category_avg = item.get("average_scores", {}) if isinstance(item.get("average_scores", {}), dict) else {}
        category_brain = category_avg.get("brain", {}) if isinstance(category_avg.get("brain", {}), dict) else {}
        category_llm = category_avg.get("llm", {}) if isinstance(category_avg.get("llm", {}), dict) else {}
        lines.append(
            "| {category} | {cases} | {eligible} | {brain_wins} | {llm_wins} | {ties} | {brain_avg} | {llm_avg} |".format(
                category=category,
                cases=item.get("total_cases", 0),
                eligible=item.get("eligible_cases", 0),
                brain_wins=category_wins.get("brain", 0),
                llm_wins=category_wins.get("llm", 0),
                ties=category_wins.get("tie", 0),
                brain_avg=category_brain.get("comparison_score"),
                llm_avg=category_llm.get("comparison_score"),
            )
        )

    lines.extend(
        [
            "",
            "## Case Results",
            "",
            "| Case | Category | BrAIn | Qwen | Delta | Status |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for case in cases:
        scores = case.get("comparison_scores", {}) if isinstance(case.get("comparison_scores", {}), dict) else {}
        brain_score = scores.get("brain")
        llm_score = scores.get("llm")
        delta: Optional[float] = None
        if brain_score is not None and llm_score is not None:
            delta = round(float(brain_score) - float(llm_score), 4)
        status = "eligible" if case.get("comparison_eligible", False) else _invalid_bits(case)
        lines.append(
            f"| {case.get('case_id', '')} | {case.get('benchmark_category', '')} | {brain_score} | {llm_score} | {delta} | {_clamp_text(status, 120)} |"
        )

    weak_cases = []
    for case in cases:
        if not case.get("comparison_eligible", False):
            weak_cases.append(case)
            continue
        scores = case.get("comparison_scores", {}) if isinstance(case.get("comparison_scores", {}), dict) else {}
        if scores.get("brain") is not None and scores.get("llm") is not None:
            if float(scores.get("brain")) - float(scores.get("llm")) <= 0.05:
                weak_cases.append(case)

    lines.extend(["", "## Improvement Targets", ""])
    if not weak_cases:
        lines.append("No near-ties or invalid comparisons were detected.")
    else:
        for case in weak_cases:
            brain_output = case.get("brain_output", {}) if isinstance(case.get("brain_output", {}), dict) else {}
            llm_output = case.get("llm_output", {}) if isinstance(case.get("llm_output", {}), dict) else {}
            lines.extend(
                [
                    f"### {case.get('case_id', '')}",
                    f"- Status: `{('eligible' if case.get('comparison_eligible', False) else _invalid_bits(case))}`",
                    f"- BrAIn: `{_clamp_text(str(brain_output.get('final_explanation', '')), 350)}`",
                    f"- Plain Qwen: `{_clamp_text(str(llm_output.get('raw_text_response', '')), 350)}`",
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def rebaseline(
    *,
    input_path: Path,
    output_path: Path,
    report_path: Path,
    model: str,
    host: str,
    case_limit: Optional[int] = None,
    reuse_existing_llm_output: bool = False,
) -> Dict[str, object]:
    source_payload = json.loads(input_path.read_text(encoding="utf-8"))
    cases = source_payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("Input results JSON must include a 'cases' list.")
    selected_cases = cases[: case_limit or len(cases)]
    baseline = PlainOllamaBaseline(model=model, host=host)
    rebaselined_cases: List[Dict[str, object]] = []

    for index, case in enumerate(selected_cases, start=1):
        turns = [str(item) for item in case.get("prompt_sequence", []) if str(item).strip()]
        if reuse_existing_llm_output and isinstance(case.get("llm_output", {}), dict):
            print(f"[{index}/{len(selected_cases)}] reuse plain Qwen baseline: {case.get('case_id', '')}", flush=True)
            llm_output = dict(case.get("llm_output", {}))
        else:
            print(f"[{index}/{len(selected_cases)}] plain Qwen baseline: {case.get('case_id', '')}", flush=True)
            llm_output = baseline.answer_turns(turns)
        llm_answer = str(llm_output.get("raw_text_response", ""))
        llm_metadata = llm_output.get("metadata", {}) if isinstance(llm_output.get("metadata", {}), dict) else {}
        llm_invalid_reason = detect_invalid_reason(
            answer_text=llm_answer,
            metadata=llm_metadata,
            error_text=str(llm_metadata.get("error", "")),
            require_live_model=True,
            provider_name="plain_ollama",
        )

        case_copy = dict(case)
        case_copy["llm_output"] = llm_output
        case_copy["llm_valid"] = llm_invalid_reason is None
        case_copy["llm_invalid_reason"] = llm_invalid_reason
        scores = dict(case_copy.get("scores", {}) if isinstance(case_copy.get("scores", {}), dict) else {})
        scores["llm"] = score_llm_output(
            benchmark_category=str(case_copy.get("benchmark_category", "")),
            response_text=llm_answer,
            expected_keywords=[str(item) for item in case_copy.get("expected_keywords", [])],
            constraint_keywords=[str(item) for item in case_copy.get("constraint_keywords", [])],
            memory_fidelity_keywords=[str(item) for item in case_copy.get("memory_fidelity_keywords", [])],
            expected_recall_facts=[str(item) for item in case_copy.get("expected_recall_facts", [])],
            corrected_facts=[str(item) for item in case_copy.get("corrected_facts", [])],
            forbidden_errors=[str(item) for item in case_copy.get("forbidden_errors", [])],
        )
        case_copy["scores"] = scores
        comparison_scores = dict(case_copy.get("comparison_scores", {}) if isinstance(case_copy.get("comparison_scores", {}), dict) else {})
        if scores.get("brain") is not None:
            comparison_scores["brain"] = comparison_score(scores["brain"])
        comparison_scores["llm"] = comparison_score(scores["llm"]) if llm_invalid_reason is None else None
        case_copy["comparison_scores"] = comparison_scores
        case_copy["comparison_eligible"] = bool(case_copy.get("brain_valid", False)) and llm_invalid_reason is None
        case_copy["invalid_reason"] = None if case_copy["comparison_eligible"] else _invalid_bits(case_copy)
        rebaselined_cases.append(case_copy)

    summary = _summary_for_cases(rebaselined_cases, model=model)
    output_payload = {
        "summary": summary,
        "cases": rebaselined_cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2, sort_keys=True), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_markdown(output_payload, source_path=input_path, output_path=output_path),
        encoding="utf-8",
    )
    return output_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Replace the harness baseline with a raw plain Ollama conversation baseline.")
    parser.add_argument("--input", default=str(Path("brain_output") / "stress_qwen3_8b_30_results.json"))
    parser.add_argument("--output", default=str(Path("brain_output") / "stress_qwen3_8b_30_plain_rebaseline.json"))
    parser.add_argument("--report", default=str(Path("brain_output") / "stress_qwen3_8b_30_plain_rebaseline.md"))
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument(
        "--reuse-existing-llm-output",
        action="store_true",
        help="Reuse llm_output from the input JSON and only recompute scores/report.",
    )
    args = parser.parse_args()

    payload = rebaseline(
        input_path=Path(args.input),
        output_path=Path(args.output),
        report_path=Path(args.report),
        model=args.model,
        host=args.host,
        case_limit=args.case_limit,
        reuse_existing_llm_output=args.reuse_existing_llm_output,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
