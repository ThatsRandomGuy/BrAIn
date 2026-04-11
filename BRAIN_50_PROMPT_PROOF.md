# BrAIn 50-Prompt Proof Benchmark

This proof run extends the BrAIn vs plain `qwen3:8b` prompt suite from 46 to
50 prompts. The four added proof-of-concept cases focus on latest-state memory,
evolving state recall, constrained planning, and memory-informed upload planning.

## Run Summary

- Total cases: 50
- Eligible comparisons: 43
- Invalid comparisons: 7
- BrAIn wins: 23
- Plain Qwen wins: 13
- Ties: 7
- Average comparison score: BrAIn 0.6827, plain Qwen 0.6493

The 7 invalid comparisons were excluded because the local plain-Qwen baseline
returned an empty answer for those cases. BrAIn produced non-empty answers for
the excluded cases.

## Category Summary

| Category | Cases | Eligible | BrAIn Wins | Qwen Wins | Ties | BrAIn Avg | Qwen Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| factual_qa | 2 | 2 | 0 | 1 | 1 | 0.7633 | 0.82 |
| memory_recall | 26 | 26 | 18 | 5 | 3 | 0.7179 | 0.6458 |
| structured_planning | 22 | 15 | 5 | 7 | 3 | 0.6109 | 0.6327 |

## Added Proof Cases

- `poc_memory_instruction_update`: latest instruction overrides an earlier demo-script instruction.
- `poc_memory_inventory_sequence`: evolving launch-kit state after multiple updates.
- `poc_planning_constrained_demo_day`: constrained demo-day plan with ordering and prerequisite rules.
- `poc_hybrid_memory_plan_revision`: upload planning that should preserve remembered constraints.

## Method

The BrAIn side was run with deterministic response generation and
`--no-bootstrap-reasoning` so the proof run would finish quickly while still
exercising routing, memory reconstruction, planning, and response-layer behavior.
The baseline side was rebaselined against raw plain Ollama `qwen3:8b`.

Commands used:

```bash
python -B brain_vs_llm_runner.py --provider deterministic --allow-nonlive-provider --no-bootstrap-reasoning --output brain_output/stress_qwen3_8b_50_brain_structured.json --no-print
python -B plain_llm_rebaseline.py --input brain_output/stress_qwen3_8b_50_brain_structured.json --output brain_output/stress_qwen3_8b_50_plain_rebaseline.json --report brain_output/stress_qwen3_8b_50_plain_rebaseline.md --model qwen3:8b
```

The detailed local report was generated at:

`brain_output/stress_qwen3_8b_50_plain_rebaseline.md`

That output folder is intentionally ignored by Git so runtime benchmark artifacts
do not bloat the repository.

## Interpretation

This run supports the current development direction:

- BrAIn is clearly stronger on memory recall, where its structured memory and reconstruction layers outperform plain Qwen.
- Plain Qwen remains more competitive on open-ended structured planning, where BrAIn still sometimes compresses final answers too aggressively.
- The response layer and routing refinements improved directness, but planning answer quality remains the best next improvement area.

