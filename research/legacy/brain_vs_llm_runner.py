from __future__ import annotations

# Historical harnesses run with the maintained core on the import path.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]/"core"))

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from brain_hybrid_prototype import BrainHybridPrototype, StoragePolicy
from brain_vs_llm_prompt_set import TEST_PROMPTS
from brain_vs_llm_scoring import (
    BENCHMARK_CATEGORIES,
    aggregate_scores,
    classify_benchmark_category,
    comparison_score,
    detect_invalid_reason,
    score_brain_output,
    score_llm_output,
)
from llm_interface import HybridLLMInterface


def _section(title: str) -> str:
    return "\n" + "=" * 50 + f"\n{title}\n" + "=" * 50


def _short_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _provider_name(interface: HybridLLMInterface) -> str:
    return str(getattr(interface.provider, "provider_name", "unknown"))


def _init_case_systems(
    *,
    suite_state_dir: Path,
    case_id: str,
    llm_provider: str,
    storage_mode: str,
    bootstrap_reasoning: bool = True,
) -> Tuple[BrainHybridPrototype, HybridLLMInterface]:
    case_state_dir = suite_state_dir / case_id
    case_state_dir.mkdir(parents=True, exist_ok=True)
    hybrid = BrainHybridPrototype(
        state_dir=case_state_dir / "brain_state",
        storage_policy=StoragePolicy.for_mode(storage_mode),
        llm_provider=llm_provider,
        bootstrap_reasoning=bootstrap_reasoning,
    )
    baseline_llm = HybridLLMInterface(provider=llm_provider)
    return hybrid, baseline_llm


class BrainVsLLMRunner:
    def __init__(
        self,
        *,
        state_dir: str | Path,
        llm_provider: str = "auto",
        storage_mode: str = "store_all_prompts",
        require_live_model: bool = True,
        bootstrap_reasoning: bool = True,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.llm_provider = llm_provider
        self.storage_mode = storage_mode
        self.require_live_model = require_live_model
        self.bootstrap_reasoning = bootstrap_reasoning

    def run_case(self, case: Dict[str, object]) -> Dict[str, object]:
        turns = [str(item) for item in case.get("turns", []) if str(item).strip()]
        if not turns:
            raise ValueError(f"Case {case.get('case_id', '<unknown>')} contains no turns.")

        hybrid, baseline_llm = _init_case_systems(
            suite_state_dir=self.state_dir,
            case_id=str(case.get("case_id", "case")),
            llm_provider=self.llm_provider,
            storage_mode=self.storage_mode,
            bootstrap_reasoning=self.bootstrap_reasoning,
        )
        provider_name = _provider_name(baseline_llm)
        benchmark_category = classify_benchmark_category(case)
        expected_keywords = [str(item) for item in case.get("expected_keywords", []) if str(item).strip()]
        constraint_keywords = [str(item) for item in case.get("constraint_keywords", []) if str(item).strip()]
        memory_fidelity_keywords = [str(item) for item in case.get("memory_fidelity_keywords", []) if str(item).strip()]
        expected_recall_facts = [str(item) for item in case.get("expected_recall_facts", []) if str(item).strip()]
        corrected_facts = [str(item) for item in case.get("corrected_facts", []) if str(item).strip()]
        forbidden_errors = [str(item) for item in case.get("forbidden_errors", []) if str(item).strip()]

        transcript: List[Dict[str, str]] = []
        brain_turn_outputs: List[Dict[str, object]] = []
        llm_turn_outputs: List[Dict[str, object]] = []
        brain_error: str = ""
        llm_error: str = ""
        last_brain = None
        last_llm = None

        for turn_index, turn_text in enumerate(turns, start=1):
            try:
                brain_result = hybrid.process_user_turn(turn_text, inspect=True)
                last_brain = brain_result
                trace_payload = brain_result.trace.to_dict()
                brain_turn_outputs.append(
                    {
                        "turn_index": turn_index,
                        "user_text": turn_text,
                        "route_type": brain_result.route_decision.route_type,
                        "clarification_state": brain_result.clarification_state,
                        "structured_result": brain_result.final_result.to_dict(),
                        "final_explanation": brain_result.llm_explanation.user_facing_text,
                        "llm_metadata": brain_result.llm_explanation.metadata,
                        "memory_written": bool(trace_payload.get("memory_written")),
                        "stored_entry_count": len(trace_payload.get("stored_entries", []) or []),
                        "retrieved_memory_count": len(brain_result.retrieved_memories),
                        "recalled_turn_count": len(brain_result.recalled_turns),
                        "merged_memory_state": trace_payload.get("merged_memory_state", {}),
                        "context_packet_v1": trace_payload.get("context_packet_v1", {}),
                        "llm_payload_v1": trace_payload.get("llm_payload_v1", {}),
                        "llm_payload_v2": trace_payload.get("llm_payload_v2", {}),
                        "cached_knowledge_checked": bool(trace_payload.get("cached_knowledge_checked")),
                        "cached_knowledge_hit": bool(trace_payload.get("cached_knowledge_hit")),
                        "cached_knowledge_entry_ids": trace_payload.get("cached_knowledge_entry_ids", []),
                        "cached_knowledge_written": bool(trace_payload.get("cached_knowledge_written")),
                        "cached_knowledge_topic_key": trace_payload.get("cached_knowledge_topic_key", ""),
                        "cached_knowledge_normalized_terms": trace_payload.get("cached_knowledge_normalized_terms", []),
                        "cached_knowledge_trust_level": trace_payload.get("cached_knowledge_trust_level", ""),
                        "cached_knowledge_used": bool(trace_payload.get("cached_knowledge_used")),
                        "cached_knowledge_entry_id": trace_payload.get("cached_knowledge_entry_id"),
                        "cached_knowledge_candidate_entry_ids": trace_payload.get("cached_knowledge_candidate_entry_ids", []),
                        "cached_knowledge_selected_entry_id": trace_payload.get("cached_knowledge_selected_entry_id"),
                        "cached_knowledge_ranking_scores": trace_payload.get("cached_knowledge_ranking_scores", []),
                        "cached_knowledge_reinforcement_applied": bool(trace_payload.get("cached_knowledge_reinforcement_applied")),
                        "cached_knowledge_updated_confidence": trace_payload.get("cached_knowledge_updated_confidence"),
                        "cached_knowledge_updated_reuse_count": trace_payload.get("cached_knowledge_updated_reuse_count"),
                        "cached_knowledge_conflict_detected": bool(trace_payload.get("cached_knowledge_conflict_detected")),
                        "cached_knowledge_conflicting_entry_ids": trace_payload.get("cached_knowledge_conflicting_entry_ids", []),
                        "cached_knowledge_penalized_entry_ids": trace_payload.get("cached_knowledge_penalized_entry_ids", []),
                        "cached_knowledge_updated_confidences": trace_payload.get("cached_knowledge_updated_confidences", []),
                        "cached_knowledge_verification_attempted": bool(trace_payload.get("cached_knowledge_verification_attempted")),
                        "cached_knowledge_verification_status": trace_payload.get("cached_knowledge_verification_status", ""),
                        "cached_knowledge_confidence_capped": bool(trace_payload.get("cached_knowledge_confidence_capped")),
                        "cached_knowledge_uncertainty_flag": bool(trace_payload.get("cached_knowledge_uncertainty_flag")),
                        "cached_knowledge_stabilized_flag": bool(trace_payload.get("cached_knowledge_stabilized_flag")),
                        "cached_knowledge_dominance_score": trace_payload.get("cached_knowledge_dominance_score"),
                        "cached_knowledge_convergence_rate": trace_payload.get("cached_knowledge_convergence_rate"),
                        "multi_hop_used": bool(trace_payload.get("multi_hop_used")),
                        "merged_entry_ids": trace_payload.get("merged_entry_ids", []),
                        "reasoning_steps": trace_payload.get("reasoning_steps", []),
                        "merged_reasoning_packet": trace_payload.get("merged_reasoning_packet", {}),
                        "reasoning_graph_nodes": trace_payload.get("reasoning_graph_nodes", []),
                        "reasoning_graph_edges": trace_payload.get("reasoning_graph_edges", []),
                        "reasoning_path": trace_payload.get("reasoning_path", []),
                        "global_graph_size": trace_payload.get("global_graph_size", {}),
                        "reused_graph_edges": trace_payload.get("reused_graph_edges", []),
                        "new_graph_edges": trace_payload.get("new_graph_edges", []),
                        "structured_explanation": trace_payload.get("structured_explanation", {}),
                        "final_explanation": trace_payload.get("final_explanation", ""),
                        "cache_bypassed_reason": trace_payload.get("cache_bypassed_reason", ""),
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive harness path
                brain_error = str(exc)
                brain_turn_outputs.append(
                    {
                        "turn_index": turn_index,
                        "user_text": turn_text,
                        "runtime_error": brain_error,
                    }
                )
                break

            transcript.append({"role": "user", "content": turn_text})
            try:
                llm_result = baseline_llm.answer_user_prompt(
                    user_input_text=turn_text,
                    route_type="llm_only_baseline",
                    structured_context={
                        "conversation_history": list(transcript),
                        "case_id": case.get("case_id", ""),
                        "turn_index": turn_index,
                    },
                )
                last_llm = llm_result
                transcript.append({"role": "assistant", "content": llm_result.user_facing_text})
                llm_turn_outputs.append(
                    {
                        "turn_index": turn_index,
                        "user_text": turn_text,
                        "raw_text_response": llm_result.user_facing_text,
                        "metadata": llm_result.metadata,
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive harness path
                llm_error = str(exc)
                llm_turn_outputs.append(
                    {
                        "turn_index": turn_index,
                        "user_text": turn_text,
                        "runtime_error": llm_error,
                    }
                )
                break

        sequence_memory_written = any(
            bool(turn_output.get("memory_written")) for turn_output in brain_turn_outputs
        )
        sequence_stored_entry_count = sum(
            int(turn_output.get("stored_entry_count", 0)) for turn_output in brain_turn_outputs
        )
        sequence_retrieved_memory_count = sum(
            int(turn_output.get("retrieved_memory_count", 0)) for turn_output in brain_turn_outputs
        )
        sequence_recalled_turn_count = sum(
            int(turn_output.get("recalled_turn_count", 0)) for turn_output in brain_turn_outputs
        )
        sequence_merged_memory_states = [
            dict(turn_output.get("merged_memory_state", {}))
            for turn_output in brain_turn_outputs
            if isinstance(turn_output.get("merged_memory_state"), dict)
            and turn_output.get("merged_memory_state")
        ]

        brain_answer = last_brain.llm_explanation.user_facing_text if last_brain is not None else ""
        brain_metadata = last_brain.llm_explanation.metadata if last_brain is not None else {}
        llm_answer = last_llm.user_facing_text if last_llm is not None else ""
        llm_metadata = last_llm.metadata if last_llm is not None else {}

        brain_invalid_reason = detect_invalid_reason(
            answer_text=brain_answer,
            metadata=brain_metadata,
            error_text=brain_error,
            require_live_model=self.require_live_model,
            provider_name=provider_name,
        )
        llm_invalid_reason = detect_invalid_reason(
            answer_text=llm_answer,
            metadata=llm_metadata,
            error_text=llm_error,
            require_live_model=self.require_live_model,
            provider_name=provider_name,
        )
        comparison_eligible = brain_invalid_reason is None and llm_invalid_reason is None

        reasoning_trace = last_brain.trace.to_dict() if last_brain is not None else {}
        if reasoning_trace:
            reasoning_trace = dict(reasoning_trace)
            reasoning_trace["sequence_memory_written"] = sequence_memory_written
            reasoning_trace["sequence_stored_entry_count"] = sequence_stored_entry_count
            reasoning_trace["sequence_retrieved_memory_count"] = sequence_retrieved_memory_count
            reasoning_trace["sequence_recalled_turn_count"] = sequence_recalled_turn_count
            if sequence_merged_memory_states:
                reasoning_trace["sequence_merged_memory_states"] = sequence_merged_memory_states

        brain_output = {
            "structured_result": last_brain.final_result.to_dict() if last_brain is not None else {},
            "reasoning_trace": reasoning_trace,
            "final_explanation": brain_answer,
            "route_type": last_brain.route_decision.route_type if last_brain is not None else "",
            "retrieved_memory_count": len(last_brain.retrieved_memories) if last_brain is not None else 0,
            "recalled_turn_count": len(last_brain.recalled_turns) if last_brain is not None else 0,
            "sequence_memory_written": sequence_memory_written,
            "sequence_stored_entry_count": sequence_stored_entry_count,
            "sequence_retrieved_memory_count": sequence_retrieved_memory_count,
            "sequence_recalled_turn_count": sequence_recalled_turn_count,
            "llm_provider": provider_name,
            "llm_metadata": brain_metadata,
            "turn_outputs": brain_turn_outputs,
        }
        llm_output = {
            "raw_text_response": llm_answer,
            "provider": provider_name,
            "metadata": llm_metadata,
            "turn_outputs": llm_turn_outputs,
        }
        scores = {
            "brain": score_brain_output(
                benchmark_category=benchmark_category,
                response_text=brain_answer,
                selected_strategy=last_brain.strategy_plan.get("selected_strategy", {}) if last_brain is not None else {},
                clarification_state=last_brain.clarification_state if last_brain is not None else {},
                retrieved_memory_count=brain_output["retrieved_memory_count"],
                recalled_turn_count=brain_output["recalled_turn_count"],
                expected_keywords=expected_keywords,
                constraint_keywords=constraint_keywords,
                memory_fidelity_keywords=memory_fidelity_keywords,
                expected_recall_facts=expected_recall_facts,
                corrected_facts=corrected_facts,
                forbidden_errors=forbidden_errors,
            ),
            "llm": score_llm_output(
                benchmark_category=benchmark_category,
                response_text=llm_answer,
                expected_keywords=expected_keywords,
                constraint_keywords=constraint_keywords,
                memory_fidelity_keywords=memory_fidelity_keywords,
                expected_recall_facts=expected_recall_facts,
                corrected_facts=corrected_facts,
                forbidden_errors=forbidden_errors,
            ),
        }
        comparison_scores = {
            "brain": comparison_score(scores["brain"]) if comparison_eligible else None,
            "llm": comparison_score(scores["llm"]) if comparison_eligible else None,
        }
        invalid_reason: Optional[str] = None
        if not comparison_eligible:
            invalid_bits = []
            if brain_invalid_reason is not None:
                invalid_bits.append(f"brain:{brain_invalid_reason}")
            if llm_invalid_reason is not None:
                invalid_bits.append(f"llm:{llm_invalid_reason}")
            invalid_reason = "; ".join(invalid_bits)

        return {
            "case_id": case.get("case_id", ""),
            "category": case.get("category", ""),
            "benchmark_category": benchmark_category,
            "description": case.get("description", ""),
            "prompt": turns[-1],
            "prompt_sequence": turns,
            "expected_keywords": expected_keywords,
            "constraint_keywords": constraint_keywords,
            "memory_fidelity_keywords": memory_fidelity_keywords,
            "expected_recall_facts": expected_recall_facts,
            "corrected_facts": corrected_facts,
            "forbidden_errors": forbidden_errors,
            "brain_valid": brain_invalid_reason is None,
            "llm_valid": llm_invalid_reason is None,
            "brain_invalid_reason": brain_invalid_reason,
            "llm_invalid_reason": llm_invalid_reason,
            "invalid_reason": invalid_reason,
            "comparison_eligible": comparison_eligible,
            "brain_output": brain_output,
            "llm_output": llm_output,
            "scores": scores,
            "comparison_scores": comparison_scores,
        }


def render_case(case_result: Dict[str, object]) -> str:
    brain_output = case_result.get("brain_output", {})
    llm_output = case_result.get("llm_output", {})
    return "\n".join(
        [
            _section("PROMPT"),
            f"Benchmark category: {case_result.get('benchmark_category', '')}",
            "\n".join(str(item) for item in case_result.get("prompt_sequence", [])),
            _section("BRAIN OUTPUT"),
            _short_json(
                {
                    "brain_valid": case_result.get("brain_valid"),
                    "brain_invalid_reason": case_result.get("brain_invalid_reason"),
                    "structured_result": brain_output.get("structured_result", {}),
                    "final_explanation": brain_output.get("final_explanation", ""),
                    "route_type": brain_output.get("route_type", ""),
                    "retrieved_memory_count": brain_output.get("retrieved_memory_count", 0),
                    "recalled_turn_count": brain_output.get("recalled_turn_count", 0),
                    "reasoning_trace": brain_output.get("reasoning_trace", {}),
                }
            ),
            _section("LLM OUTPUT"),
            _short_json(
                {
                    "llm_valid": case_result.get("llm_valid"),
                    "llm_invalid_reason": case_result.get("llm_invalid_reason"),
                    "raw_text_response": llm_output.get("raw_text_response", ""),
                    "metadata": llm_output.get("metadata", {}),
                }
            ),
            _section("SCORES"),
            _short_json(
                {
                    "comparison_eligible": case_result.get("comparison_eligible"),
                    "invalid_reason": case_result.get("invalid_reason"),
                    "scores": case_result.get("scores", {}),
                    "comparison_scores": case_result.get("comparison_scores", {}),
                }
            ),
        ]
    )


def _category_case_blocks(case_results: Sequence[Dict[str, object]]) -> List[str]:
    blocks: List[str] = []
    for category in BENCHMARK_CATEGORIES:
        category_cases = [item for item in case_results if item.get("benchmark_category") == category]
        if not category_cases:
            continue
        blocks.append(_section(f"CATEGORY: {category}"))
        for item in category_cases:
            blocks.append(render_case(item))
    return blocks


def run_suite(
    *,
    prompts: Sequence[Dict[str, object]] = TEST_PROMPTS,
    output_path: str | Path = Path("tests") / "results.json",
    llm_provider: str = "auto",
    storage_mode: str = "store_all_prompts",
    reset_state: bool = True,
    print_cases: bool = True,
    require_live_model: bool = True,
    case_limit: Optional[int] = None,
    bootstrap_reasoning: bool = True,
) -> Dict[str, object]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suite_state_dir = output_path.parent / "brain_vs_llm_state"
    if reset_state and suite_state_dir.exists():
        shutil.rmtree(suite_state_dir)
    suite_state_dir.mkdir(parents=True, exist_ok=True)

    selected_prompts = list(prompts[: case_limit or len(prompts)])
    runner = BrainVsLLMRunner(
        state_dir=suite_state_dir,
        llm_provider=llm_provider,
        storage_mode=storage_mode,
        require_live_model=require_live_model,
        bootstrap_reasoning=bootstrap_reasoning,
    )

    case_results: List[Dict[str, object]] = [runner.run_case(case) for case in selected_prompts]
    if print_cases:
        print("\n".join(_category_case_blocks(case_results)))

    invalid_cases = [item for item in case_results if not item.get("comparison_eligible", False)]
    eligible_cases = [item for item in case_results if item.get("comparison_eligible", False)]
    aggregate = aggregate_scores(case_results)
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

    summary = {
        "provider": runner.llm_provider,
        "require_live_model": require_live_model,
        "bootstrap_reasoning": bootstrap_reasoning,
        "total_cases": len(case_results),
        "eligible_comparison_cases": len(eligible_cases),
        "invalid_cases": len(invalid_cases),
        "results_by_category": by_category_summary,
        "aggregate_scores": aggregate.get("overall", {}),
    }
    payload = {
        "summary": summary,
        "cases": case_results,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"output_path": str(output_path), "state_dir": str(suite_state_dir), "payload": payload}


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="brain_vs_llm_runner",
        description="Run fair side-by-side comparisons between the BrAIn hybrid pipeline and an LLM-only baseline.",
    )
    parser.add_argument(
        "--output",
        default=str(Path("tests") / "results.json"),
        help="Path to the saved JSON results file.",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        help="LLM provider selection passed to both the hybrid pipeline and the baseline.",
    )
    parser.add_argument(
        "--storage-mode",
        default="store_all_prompts",
        choices=(
            "store_all_prompts",
            "store_only_successful_reasoning",
            "store_final_results_only",
            "store_disabled",
        ),
        help="Storage mode for the wrapped BrAIn prototype.",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Skip case-by-case terminal output and only save the JSON report.",
    )
    parser.add_argument(
        "--allow-nonlive-provider",
        action="store_true",
        help="Allow deterministic provider runs without forcing invalid-case marking.",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Optional number of cases to run for a smaller validation pass.",
    )
    parser.add_argument(
        "--no-bootstrap-reasoning",
        action="store_true",
        help="Skip per-case bootstrap memory seeding for faster proof/CI benchmark runs.",
    )
    args = parser.parse_args()
    result = run_suite(
        output_path=args.output,
        llm_provider=args.provider,
        storage_mode=args.storage_mode,
        print_cases=not args.no_print,
        require_live_model=not args.allow_nonlive_provider,
        case_limit=args.case_limit,
        bootstrap_reasoning=not args.no_bootstrap_reasoning,
    )
    print(_section("RESULTS"))
    print(f"Saved comparison results to {result['output_path']}")
    print(f"State dir: {result['state_dir']}")
    print(_short_json(result["payload"]["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
