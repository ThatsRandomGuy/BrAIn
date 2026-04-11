# BrAIn Terminal Prototype

## Run

```bash
python interactive_brain_shell.py
```

Optional flags:

```bash
python interactive_brain_shell.py --debug --raw
python interactive_brain_shell.py --storage-mode store_final_results_only
```

## Commands

- `/help` shows available commands
- `/quit` exits the shell
- `/debug on|off` toggles more verbose terminal sections
- `/raw on|off` toggles raw JSON output
- `/memory` shows recent stored memory summaries
- `/memory full` shows detailed stored memory entries
- `/last` shows the full trace from the last prompt
- `/paths` shows candidate paths from the last prompt
- `/pending` shows the current pending clarification
- `/goals` shows active goals and last runtime weights
- `/store on|off` toggles automatic storage
- `/store mode <mode>` changes storage policy

Storage modes:

- `store_all_prompts`
- `store_only_successful_reasoning`
- `store_final_results_only`
- `store_disabled`

## Query Routing

Every prompt now goes through a deterministic router before execution.

Routes:

- `factual_question`: bypass deep BrAIn planning and use the direct LLM answer path
- `memory_question`: retrieve BrAIn memory first and explain what was found
- `planning_request`: run the full BrAIn planning / grounding / goal pipeline
- `goal_request`: run the full BrAIn reasoning pipeline with goal weighting
- `transform_request`: keep BrAIn in control for structured rewrite / reduce-risk requests
- `hybrid_request`: combine BrAIn memory/reasoning with the direct LLM answer layer
- `unknown`: fall back safely with a traceable explanation instead of guessing

The shell now shows a `ROUTE DECISION` section during normal output, and `/last`
includes the same routing trace.

## Prompt Enrichment

Planning, goal, transform, and hybrid routes can now run through a deterministic
enrichment layer before deep BrAIn execution.

The shell shows an `ENRICHMENT RESULT` section with:

- inferred fields
- whether enrichment ran
- whether clarification is required
- clarification reasons when the prompt is still too sparse

Clarification is triggered instead of silently producing a weak fallback when
BrAIn still lacks enough structure to reason usefully.

When clarification is required, BrAIn now asks a deterministic follow-up
question and the next user reply is merged back into the original prompt before
reasoning resumes.

The shell shows:

- `CLARIFICATION STATE`
- the clarification question
- the clarification response
- the merged effective prompt used for resumed reasoning

## Episodic Recall

Memory questions now use episodic turn recall over prior prompts and answers.

Each stored turn can include:

- raw user prompt
- final answer
- parsed `BrainInputV1`
- final selected structured result
- route type
- reasoning success flag
- candidate path count
- timestamp / sequence metadata

Memory-route output now shows `RECALLED TURNS` in addition to structural memory.

## Goal Tracking V6

BrAIn now persists lightweight deterministic goal state for planning, goal,
transform, and hybrid routes.

Each turn can update:

- tracked goal status
- progress state
- helpful vs redundant suggestion counters
- suggested next step

Normal terminal output now shows:

- `GOAL STATUS`
- `PROGRESS TRACKING`
- `NEXT STEP SUGGESTION`

## Reasoning Boundary

BrAIn handles:

- deterministic query routing for BrAIn-owned routes
- parsing
- memory retrieval
- concept abstraction
- generative expansion
- planning
- reality grounding
- goal / reward re-ranking

The LLM layer handles:

- direct answering for factual routes
- answer wording for hybrid routes
- explanation
- terminal-friendly natural language
- optional provider-backed response formatting

The LLM does not change BrAIn's structured result for memory, planning, goal, or transform routes.

## Memory Inspection

Manual inspection paths:

- use `/memory` and `/memory full` for recent stored entries
- use `/last` for the full reasoning trace
- use `/paths` to inspect only candidate paths
- inspect persistent state under `brain_output/interactive_prototype.state`

## Optional Live LLM

The shell works without a live model.

Recommended local setup with Ollama and `Qwen3:8b`:

```bash
ollama pull qwen3:8b
set BRAIN_LLM_PROVIDER=ollama
set BRAIN_LLM_MODEL=qwen3:8b
python interactive_brain_shell.py
```

Optional host override:

```bash
set BRAIN_LLM_HOST=http://localhost:11434
```

To use a local/provider-backed subprocess model instead, set:

```bash
set BRAIN_LLM_COMMAND=your_local_llm_command
python interactive_brain_shell.py
```

If the live model call fails, the shell falls back to deterministic explanation formatting.

## Route Demo Examples

Run the deterministic route demos:

```bash
python brain_query_router_examples.py
```

This writes:

- `brain_output/examples/brain_query_router_examples.json`
- `brain_output/examples/brain_query_router_examples.md`

Run the execution-quality demos:

```bash
python brain_route_execution_examples.py
```

This writes:

- `brain_output/examples/brain_route_execution_examples.json`
- `brain_output/examples/brain_route_execution_examples.md`

Run the interactive reasoning demos:

```bash
python brain_interactive_reasoning_examples.py
```

This writes:

- `brain_output/examples/brain_interactive_reasoning_examples.json`
- `brain_output/examples/brain_interactive_reasoning_examples.md`

Run the goal-tracking demos:

```bash
python brain_goal_tracking_examples.py
```

This writes:

- `brain_output/examples/brain_goal_tracking_examples.json`
- `brain_output/examples/brain_goal_tracking_examples.md`
