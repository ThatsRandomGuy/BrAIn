# BrAIn vs LLM Comparison Harness

This harness runs the same prompt suite through:

1. the BrAIn hybrid pipeline
2. an LLM-only baseline using the same `HybridLLMInterface` provider selection

## Run

```bash
python brain_vs_llm_runner.py --provider ollama
```

## Optional flags

```bash
python brain_vs_llm_runner.py --provider ollama --output tests/results.json
python brain_vs_llm_runner.py --provider ollama --case-limit 3
python brain_vs_llm_runner.py --provider ollama --no-print
```

## Output

The runner saves a JSON artifact to:

`tests/results.json`

It includes:

- prompt sequence per case
- BrAIn structured result
- BrAIn reasoning trace
- BrAIn final explanation
- LLM-only raw response
- deterministic heuristic scores

## Provider behavior

Both paths use the same provider selection rules from `llm_interface.py`.

- `auto` uses environment configuration if present
- `ollama` uses `BRAIN_LLM_MODEL` / `BRAIN_LLM_HOST`
- `subprocess` uses `BRAIN_LLM_COMMAND`

If no live provider is available, the harness marks cases invalid by default so configuration/setup fallbacks are not treated as valid benchmark evidence.

## Notes

- The harness wraps BrAIn and does not modify its reasoning logic.
- BrAIn still performs structured reasoning internally.
- The LLM-only path receives plain conversation history only and does not get BrAIn memory or reasoning artifacts.
- Each benchmark case runs in an isolated state directory so cross-case memory leakage does not affect the results.
