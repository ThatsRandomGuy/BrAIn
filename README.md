# BrAIn

BrAIn is an experimental reasoning and memory prototype that tests whether a
structured cognitive layer can improve reliability over a plain LLM baseline.

Instead of treating every prompt as a single text-completion problem, BrAIn
routes the request, retrieves and reconstructs relevant memory, evaluates
knowledge through a reasoning graph, simulates/plans when needed, and then
translates the internal structure into a concise user-facing answer.

The goal is not to wrap an LLM with more prompting. The goal is to keep BrAIn's
internal state inspectable and deterministic where possible, while still using a
local model such as `qwen3:8b` for plain-language generation or baseline
comparison.

## Current Findings

The latest proof run extends the benchmark suite to 50 prompts and compares
BrAIn against a plain Ollama `qwen3:8b` baseline.

- Total cases: 50
- Eligible comparisons: 43
- BrAIn wins: 23
- Plain Qwen wins: 13
- Ties: 7
- Average comparison score: BrAIn 0.6827, plain Qwen 0.6493
- Memory recall: BrAIn 18 wins, Qwen 5 wins, 3 ties

The strongest result so far is memory recall. BrAIn's structured memory,
latest-state handling, and reconstruction layer outperform the plain LLM on
tasks like preference updates, current location/state recall, merged
constraints, and prior planning requests.

The main remaining weakness is open-ended planning response quality. BrAIn's
internal planner is structured, but the final answer can still be too compressed
or conservative compared with a fluent LLM answer. That makes the response layer
and planning-to-action translation the next improvement area.

The detailed proof summary is in `BRAIN_50_PROMPT_PROOF.md`.

## How It Works

At a high level, BrAIn runs a prompt through these layers:

1. `brain_query_router.py` classifies the prompt as factual, memory, planning,
   goal, transform, hybrid, or state update.
2. `brain_hybrid_prototype.py` coordinates memory retrieval, state merging,
   planning, clarification handling, and final response preparation.
3. `brain_reasoning_graph.py` evaluates candidate knowledge through graph,
   cluster, path, activation, spatial, temporal, and concept-memory structures.
4. `llm_interface.py` separates internal decisions from user-facing responses,
   suppresses meta-language, and can use deterministic or Ollama-backed output.
5. `brain_vs_llm_runner.py` and `plain_llm_rebaseline.py` run side-by-side
   comparisons against a plain LLM baseline.

## What Is Included

- `brain_hybrid_prototype.py` - main hybrid reasoning pipeline.
- `brain_reasoning_graph.py` - reasoning graph, cluster, path, activation, and spatial reasoning layer.
- `llm_interface.py` - deterministic/LLM response interface and final response layer.
- `interactive_brain_shell.py` - terminal shell for manual experiments.
- `brain_vs_llm_runner.py` - benchmark harness for BrAIn vs a plain LLM baseline.
- `plain_llm_rebaseline.py` - raw Ollama baseline runner for plain `qwen3:8b` comparisons.
- `tests/test_reasoning_graph_layer.py` - deterministic regression tests for core graph and response behavior.

Additional design notes live in:

- `BRAIN_50_PROMPT_PROOF.md`
- `BRAIN_CONTEXT.md`
- `BRAIN_ARCHITECTURE_AUDIT.md`
- `BRAIN_VALIDATION_LEDGER.md`
- `README_BRAIN_VS_LLM.md`
- `TERMINAL_PROTOTYPE_USAGE.md`

## Requirements

This project currently uses the Python standard library for the checked-in test
path. Live LLM benchmarking can use Ollama if it is available locally.

The latest benchmark work used `qwen3:8b` through Ollama, but the deterministic
tests do not require a live model.

Optional live benchmark setup:

```bash
ollama pull qwen3:8b
```

## Run The Tests

```bash
python -B -m unittest tests.test_reasoning_graph_layer
```

## Run The Interactive Shell

```bash
python interactive_brain_shell.py
```

Useful shell flags:

```bash
python interactive_brain_shell.py --debug --raw
python interactive_brain_shell.py --storage-mode store_final_results_only
```

## Run The Benchmark Harness

Fast deterministic BrAIn-side proof run:

```bash
python -B brain_vs_llm_runner.py --provider deterministic --allow-nonlive-provider --no-bootstrap-reasoning --output brain_output/stress_qwen3_8b_50_brain_structured.json --no-print
```

Raw plain-Qwen rebaseline:

```bash
python -B plain_llm_rebaseline.py --input brain_output/stress_qwen3_8b_50_brain_structured.json --output brain_output/stress_qwen3_8b_50_plain_rebaseline.json --report brain_output/stress_qwen3_8b_50_plain_rebaseline.md --model qwen3:8b
```

Live side-by-side runner:

```bash
python brain_vs_llm_runner.py --provider ollama
```

For a quick smoke run:

```bash
python brain_vs_llm_runner.py --provider ollama --case-limit 3
```

Generated output is written under `brain_output/`, which is intentionally ignored
by Git.

## Repository Hygiene

Generated benchmark output, runtime state, Python bytecode, local editor/agent
settings, and temporary probe state are ignored through `.gitignore` so the
repository stays focused on source, tests, and documentation.
