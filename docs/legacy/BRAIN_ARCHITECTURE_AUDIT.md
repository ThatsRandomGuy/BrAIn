# BrAIn Current Architecture Audit

## Glossary
- `BrainMemory`: parser-facing semantic object used before conversion into `BrainInputV1`.
- `BrainInputV1`: BrAIn's main normalized structured packet for runtime reasoning, storage, traces, and LLM handoff.
- `QueryRouteDecision`: route-selection object produced by `brain_query_router.py`.
- `HybridReasoningTrace`: the main per-turn trace packet assembled by `brain_hybrid_prototype.py`.
- `HybridTurnResult`: the final per-turn runtime result returned by `BrainHybridPrototype.process_user_turn`.
- `EpisodicMemoryEntry`: persistent memory-store row in `episodic_memory.json`.
- `EpisodicTurnRecord`: persistent turn-store row in `episodic_turns.json`.
- `RetrievedMemory`: retrieval wrapper with score plus referenced memory entry.
- `EpisodicTurnRecall`: recall wrapper with score plus referenced prior turn.
- `merged_memory_state`: compact merged state packet built from retrieved state memory and/or recalled turns.
- `GoalDrivenDecision`: reasoning engine output containing rescored candidate paths and selected path.
- `StrategyPlanningResult`: post-turn plan summary generated after reasoning/goal tracking.

Current state summary:
BrAIn is currently a structured-turn pipeline centered on `brain_hybrid_prototype.py`. It parses raw text into `BrainMemory`, converts that into `BrainInputV1`, routes the turn, optionally retrieves or writes episodic memory, optionally runs the reasoning stack, then hands either a structured result or a context packet to the LLM interface for wording. Memory recall is now active and benchmark-visible, but most "intelligence" still comes from deterministic rule systems plus the reasoning engine, not from the LLM.

## 1. Runtime Flow
Actual end-to-end runtime path today:

1. User input reaches `BrainHybridPrototype.process_user_turn` in `brain_hybrid_prototype.py`.
2. If a clarification is pending, `respond_to_clarification` merges the new answer into the original prompt and re-enters normal processing with a route override.
3. Otherwise `run_prompt` runs:
   - `BrainParser.parse(...)` to produce `ParseResult(memory, debug)`.
   - `BrainCodec.resolve_memory(...)` to normalize parser output.
   - `_memory_to_brain_input(...)` to convert the normalized `BrainMemory` into `BrainInputV1`.
   - `route_user_prompt(...)` to produce `QueryRouteDecision`.
   - `_should_force_memory_route(...)` may promote a non-memory route to `memory_question` if recall cues plus stored state/turns are present.
   - `BrainPromptEnrichment.enrich(...)` runs next, but only planning/goal/transform/hybrid routes actually get enrichment behavior.
4. Route-specific execution:
   - `factual_question` -> `_run_factual_turn`
   - `memory_question` -> `_run_memory_turn`
   - `state_update` -> `_run_state_update_turn`
   - `planning_request`, `goal_request`, `transform_request` -> `_run_reasoning_turn(..., direct_answer=False)`
   - `hybrid_request` -> `_run_reasoning_turn(..., direct_answer=True)`
   - `unknown` -> `_run_unknown_turn`
5. Memory-question path:
   - `retrieve_stateful_memory(...)`
   - `retrieve_episodic_turns(...)`
   - fallback `retrieve_relevant_brain_memory(...)` if needed
   - `merge_memory_state(...)`
   - `_memory_state_to_brain_input(...)` creates a structured recall result if merged state exists
   - `_build_memory_recall_explanation(...)` converts merged state into direct answer text when possible
6. Reasoning path:
   - optional memory retrieval first
   - `infer_goal_weights(...)`
   - `self.engine.plan(...)` where `self.engine` is `BrainGoalRewardV5`
   - `_select_result_from_decision(...)`
   - trace is rebuilt with candidate paths, grounded paths, goal-scored paths, selected path, and final predicted outcome
   - `HybridLLMInterface.explain(...)` or `answer_user_prompt(...)` is called
7. `_finalize_turn(...)` always runs last:
   - goal tracking update
   - strategy generation update
   - history log write
   - optional memory-store writes
   - turn-store write
   - final `HybridTurnResult` assembly

Important ordering detail:
- Goal tracking and strategy generation happen after the route-specific work and after the initial trace is built.
- State memory extraction/writes happen in `_finalize_turn`, not during parsing and not during the route function itself.
- The LLM does not decide the structure. Structure is already chosen before the LLM is called.

Benchmark runtime flow:
1. `BrainVsLLMRunner.run_case(...)` creates a fresh `BrainHybridPrototype` per case in a case-specific state directory.
2. The case's `turns` list is executed in order through `process_user_turn`.
3. Sequence memory persists inside that one case's `brain_state` directory across turns.
4. The last BrAIn answer and trace become the scored output for that case.
5. Benchmark scoring happens after runtime in `brain_vs_llm_scoring.py`; it does not steer the live BrAIn route logic.

## 2. Packet and Schema Definitions
This section documents the active packet formats used now.

### 2.1 `BrainInputV1`
Defined in `brain_input_v1.py`.

Fields:
- `version`: schema version string, currently `BrainInputV1`
- `encoding_type`: internal encoding family, commonly `hierarchical_pattern`
- `actor`: subject, usually `user`
- `intent`: coarse intent such as `question`, `request`, `inform`
- `action`: normalized action label
- `time`: normalized time slot
- `priority`: coarse priority
- `pattern_id`: optional pattern id from codec/runtime
- `reference_id`: optional external reference id
- `slots.object`: object slot
- `slots.topic`: topic slot
- `slots.modifiers`: list of modifiers
- `slots.constraints`: list of constraints
- `slots.negations`: list of negations
- `slots.conditions`: list of conditions
- `relations`: list of `{type, source, target}`
- `subpattern_refs`: list of integer subpattern ids
- `primitive_ids`: list of primitive ids
- `temporal_score`: optional temporal score

Created:
- `_memory_to_brain_input(...)` in `brain_hybrid_prototype.py`
- other codec/reasoning layers

Consumed:
- router
- memory store
- reasoning engine
- goal tracking
- strategy generation
- LLM handoff
- benchmark output

Real example from current output:
```json
{
  "version": "BrainInputV1",
  "encoding_type": "hierarchical_pattern",
  "actor": "user",
  "intent": "inform",
  "action": "meeting_time",
  "time": "",
  "priority": "normal",
  "slots": {
    "object": "meeting_time",
    "topic": "friday at 11 am",
    "modifiers": [],
    "constraints": [],
    "negations": [],
    "conditions": []
  },
  "relations": [],
  "subpattern_refs": [1, 2, 130, 237],
  "primitive_ids": [],
  "pattern_id": 412
}
```

### 2.2 `QueryRouteDecision`
Defined in `brain_query_router.py`.

Fields:
- `route_type`: winning route
- `confidence`: confidence scalar
- `reasons`: human-readable reason list
- `run_brain_reasoning`: whether reasoning stack should run
- `run_direct_llm`: whether direct-answer LLM path should run
- `run_hybrid_mode`: whether this is hybrid mode
- `score_breakdown`: numeric score per route type

Created:
- `route_user_prompt(...)`
- promoted in `_promote_route_decision(...)`

Consumed:
- route dispatch in `run_prompt`
- trace
- history log
- benchmark output

Real example from current output:
```json
{
  "route_type": "memory_question",
  "confidence": 0.99,
  "reasons": [
    "Detected question-form phrasing.",
    "No memory/planning/goal/transform cues were detected, so direct factual answering is preferred.",
    "Parsed structure is sparse, which is typical for external factual questions.",
    "Detected prior-state recall cue with relevant stored state, so memory retrieval is preferred."
  ],
  "run_brain_reasoning": true,
  "run_direct_llm": false,
  "run_hybrid_mode": false,
  "score_breakdown": {
    "factual_question": 4.7,
    "goal_request": 0.0,
    "hybrid_request": 0.0,
    "memory_question": 4.95,
    "planning_request": 0.0,
    "state_update": 0.0,
    "transform_request": 0.0,
    "unknown": 0.0
  }
}
```

### 2.3 `EpisodicMemoryEntry`
Defined in `brain_hybrid_prototype.py`.

Fields:
- `entry_id`: persistent incremental id
- `timestamp`: ISO timestamp
- `entry_type`: type such as `user_prompt`, `selected_result`, `reasoning_trace`, `state_memory`, `strategy_plan`, `feedback`, `clarification_exchange`
- `source_text`: raw or derived text source
- `parsed_memory`: `BrainMemory`
- `brain_input`: `BrainInputV1`
- `metadata`: type-specific metadata

Created:
- `EpisodicMemoryStore.add(...)`
- called by `_append_stored_entry(...)` and direct writes in `_finalize_turn`

Consumed:
- generic retrieval
- state retrieval
- trace serialization

Real example from current output:
```json
{
  "entry_id": 3,
  "timestamp": "2026-04-09T15:01:49.250320+00:00",
  "entry_type": "state_memory",
  "source_text": "Update: onboarding is now done and billing is blocked on legal review.",
  "metadata": {
    "sequence": 2,
    "route_type": "state_update",
    "state_key": "project_status:billing",
    "state_type": "project_status",
    "is_update": true,
    "state_observation": {
      "state_key": "project_status:billing",
      "state_type": "project_status",
      "value": "blocked on legal review",
      "display_text": "billing is blocked on legal review",
      "entity": "billing",
      "is_update": true,
      "merge_mode": "latest",
      "operation": "assign"
    }
  }
}
```

### 2.4 `RetrievedMemory`
Defined in `brain_hybrid_prototype.py`.

Fields:
- `entry_id`: referenced memory id
- `score`: retrieval score
- `entry`: embedded `EpisodicMemoryEntry`

Created:
- `retrieve_relevant_brain_memory(...)`
- `retrieve_stateful_memory(...)`

Consumed:
- memory traces
- `merge_memory_state(...)`
- reasoning-path memory context

### 2.5 `EpisodicTurnRecord`
Defined in `brain_hybrid_prototype.py`.

Fields:
- `turn_id`
- `sequence`
- `timestamp`
- `raw_user_prompt`
- `final_answer`
- `parsed_brain_input`
- `final_selected_result`
- `route_type`
- `reasoning_succeeded`
- `candidate_path_count`
- `route_confidence`
- `metadata`

Created:
- `self.turn_store.add(...)` in `_finalize_turn`

Consumed:
- `retrieve_episodic_turns(...)`
- benchmark per-turn outputs indirectly via runtime

### 2.6 `EpisodicTurnRecall`
Defined in `brain_hybrid_prototype.py`.

Fields:
- `turn_id`
- `score`
- `recency_score`
- `semantic_score`
- `route_score`
- `record`: embedded `EpisodicTurnRecord`

Created:
- `retrieve_episodic_turns(...)`

Consumed:
- memory recall trace
- `merge_memory_state(...)`
- memory recall answer generation fallback

### 2.7 `merged_memory_state`
Built in `merge_memory_state(...)` in `brain_hybrid_prototype.py`.

Top-level fields:
- `retrieved_count`
- `recalled_turn_count`
- `correction_chain_detected`
- `latest_state_resolution_applied`
- `items`
- `item_count`
- `wants_original_state`
- `wants_current_state`

Per-item fields:
- `state_key`
- `state_type`
- `current_value`
- `original_value`
- `display_text`
- `history_count`
- `history`
- `merge_mode`

Created:
- memory route
- reasoning route when generic memory retrieval occurs

Consumed:
- trace
- `_memory_state_to_brain_input(...)`
- `_answer_from_merged_memory_state(...)`
- hybrid direct-answer context

Real example from current output:
```json
{
  "retrieved_count": 5,
  "recalled_turn_count": 2,
  "correction_chain_detected": true,
  "latest_state_resolution_applied": true,
  "items": [
    {
      "state_key": "project_status:billing",
      "state_type": "project_status",
      "current_value": "blocked on legal review",
      "original_value": "not started",
      "display_text": "billing is blocked on legal review",
      "history_count": 4,
      "history": [],
      "merge_mode": "latest"
    }
  ],
  "item_count": 3,
  "wants_original_state": false,
  "wants_current_state": true
}
```

### 2.8 `HybridReasoningTrace`
Defined in `brain_hybrid_prototype.py`.

Fields:
- `raw_prompt`
- `effective_prompt`
- `route_decision`
- `enrichment_result`
- `clarification_state`
- `goal_tracking`
- `strategy_plan`
- `merged_memory_state`
- `parsed_input`
- `stored_entries`
- `retrieved_memories`
- `recalled_turns`
- `goal_weights`
- `concept_matches`
- `generated_candidate_paths`
- `candidate_paths`
- `grounded_paths`
- `goal_scored_paths`
- `selected_path`
- `final_predicted_outcome`
- `llm_explanation`
- `memory_written`

Created:
- `_build_trace(...)`

Consumed:
- benchmark output
- history log
- LLM explain payload
- inspection view

### 2.9 `GoalDrivenPlanningPath`
Defined in `brain_goal_reward_v5.py`.

Fields:
- inherited from `GroundedPlanningPath` via `base_path.to_dict()`
- `goal_alignment_score`
- `reward_prior`
- `goal_weighted_score`
- `goal_breakdown`

Created:
- `BrainGoalRewardV5.plan(...)`

Consumed:
- `_build_trace(...)`
- `_select_result_from_decision(...)`
- LLM explanation summary
- benchmark reasoning trace

The exact base-path fields come from `GroundedPlanningPath` in the lower reasoning layers and are visible in trace/output rather than fully redefined here.

### 2.10 `GoalTrackingResult`
Defined in `brain_goal_tracking_v6.py`.

Fields:
- `active_goal_ids`
- `goal_status`
- `progress_tracking`
- `next_step_suggestion`

Created:
- `BrainGoalTrackingV6.process_turn(...)`

Consumed:
- trace
- turn record metadata
- strategy generation input

### 2.11 `StrategyPlanningResult`
Defined in `brain_strategy_generation_v7.py`.

Fields:
- `selected_strategy`
- `alternative_strategies`
- `explanation`
- `strategy_feedback`

Created:
- `BrainStrategyGenerationV7.process_turn(...)`

Consumed:
- trace
- turn record metadata
- strategy-plan memory entry
- benchmark output

### 2.12 LLM payloads
Defined in `llm_interface.py`.

Explain payload:
- `user_input_text`
- `result_brain_input`
- `reasoning_trace`
- `selected_path`
- `retrieved_memories`
- `trace_summary`

Direct answer payload:
- `user_input_text`
- `route_type`
- `structured_context`

Created:
- `HybridLLMInterface.explain(...)`
- `HybridLLMInterface.answer_user_prompt(...)`

Consumed:
- deterministic provider
- subprocess provider
- ollama provider

### 2.13 Benchmark case schema
Defined in `brain_vs_llm_prompt_set.py`.

Common fields:
- `case_id`
- `category`
- `benchmark_category`
- `description`
- `turns`
- `expected_keywords`
- `constraint_keywords`
- `memory_fidelity_keywords`
- `expected_recall_facts`
- `corrected_facts`
- `required_constraints`
- `forbidden_errors`
- `expected_next_step_patterns`

Consumed:
- `BrainVsLLMRunner.run_case(...)`
- benchmark classifier/scorer

Real example:
```json
{
  "case_id": "memory_project_progress_status",
  "category": "memory_recall",
  "benchmark_category": "memory_recall",
  "turns": [
    "For the mobile app project, login is done, onboarding is in progress, and billing has not started.",
    "Update: onboarding is now done and billing is blocked on legal review.",
    "What is the current project status?"
  ],
  "expected_keywords": ["login", "done", "onboarding", "billing", "legal review"],
  "memory_fidelity_keywords": ["login", "done", "onboarding", "done", "billing", "blocked", "legal review"],
  "corrected_facts": ["onboarding done", "billing blocked"]
}
```

## 3. Memory System
Current memory architecture:

### 3.1 What gets written
BrAIn can write these memory types into `episodic_memory.json`:
- `user_prompt`
- `selected_result`
- `reasoning_trace`
- `state_memory`
- `strategy_plan`
- `clarification_exchange`
- `feedback`

It also writes turn-level records into `episodic_turns.json`.

### 3.2 When memory is written
Memory writing is centralized in `_finalize_turn(...)`.

Write timing:
- after route-specific processing
- after goal tracking
- after strategy generation
- before final `HybridTurnResult` returns

Specific write behavior:
- `state_memory` is extracted every turn if storage is enabled and `_extract_state_observations(...)` finds anything
- `selected_result` and `reasoning_trace` are only written on reasoning routes if `StoragePolicy` allows it
- `clarification_exchange` only writes when a pending clarification has been resolved
- `turn_store` always gets a turn record when storage is enabled

### 3.3 RAM vs persistent storage
RAM:
- `EpisodicMemoryStore.entries` is an in-memory list loaded from disk on prototype init
- `EpisodicTurnStore.records` is an in-memory list loaded from disk on init
- `PendingClarificationStore.pending` is held in memory after load
- current-turn retrieved memories, recalls, merged state, and trace are all in RAM only unless serialized as part of a turn/memory write

Persistent disk files under each prototype `state_dir`:
- `episodic_memory.json`
- `episodic_turns.json`
- `pending_clarification.json`
- `prototype_history.json`
- `goal_tracking_v6.json`
- `strategy_generation_v7.json`
- deeper reasoning-engine state under `v5/`

Behavior:
- stores load entire JSON content at startup
- writes are synchronous JSON rewrites on update
- no external DB, no vector store, no separate RAM-only eviction layer

### 3.4 Sequence memory
Sequence memory in practice means:
- per-case or per-state-dir accumulation of state memory entries and turn records across sequential turns
- benchmark runner preserves that state within one case because it reuses the same `BrainHybridPrototype` instance for every turn in the case
- runner also reports sequence aggregates such as `sequence_memory_written`, `sequence_stored_entry_count`, and `sequence_merged_memory_states`

### 3.5 Retrieval
Implemented retrieval paths:

1. `retrieve_stateful_memory(...)`
- only searches `entry_type == "state_memory"`
- scores by token overlap, state-type relevance, entity/display-text match, recency, and update flag
- this is the main memory path for benchmark-style recall

2. `retrieve_relevant_brain_memory(...)`
- generic retrieval over all meaningful memory entries
- uses structural similarity plus text overlap bonus
- used in reasoning turns and as memory fallback

3. `retrieve_episodic_turns(...)`
- searches prior `EpisodicTurnRecord`s
- skips earlier `memory_question` records
- scores with semantic token overlap, route preference, and recency

### 3.6 Overwrite/correction handling
Overwrite support is real but heuristic.

Implemented mechanisms:
- `_extract_state_observations(...)` marks many updates with `is_update=True`
- `merge_memory_state(...)` groups observations by `state_key`
- for `merge_mode == "latest"`: latest observation wins; original may be preserved in `original_value`
- for `merge_mode == "append_set"`: deduped list/set-like accumulation, used for troubleshooting steps
- special world-state arithmetic path exists for `world_food_total`
- correction chain detection is inferred when later grouped observations contain updates

What works now:
- favorite color overwrite
- chicken location overwrite
- meeting-time correction chain
- auth original vs current status
- project/checklist status overwrite in benchmark-style phrasing

### 3.7 `merged_memory_state` construction
Source inputs:
- `RetrievedMemory.entry.metadata.state_observation`
- `EpisodicTurnRecall.record.metadata.state_observations`

Build steps:
1. flatten observations from retrieved state memory and recalled turns
2. inspect query text for `wants_original` and `wants_current`
3. group observations by `state_key`
4. apply special handling for `world_food_total`
5. merge each key using `merge_mode`
6. emit compact merged items plus top-level counters/flags

### 3.8 Memory in traces
Memory is exposed in:
- `trace.retrieved_memories`
- `trace.recalled_turns`
- `trace.merged_memory_state`
- `trace.memory_written`
- benchmark per-turn output fields:
  - `memory_written`
  - `stored_entry_count`
  - `retrieved_memory_count`
  - `recalled_turn_count`
  - `merged_memory_state`

### 3.9 What is partial
Implemented but partial:
- state extraction is deterministic benchmark-family heuristic, not broad language understanding
- generic retrieval exists but benchmark memory wins rely heavily on `state_memory`
- merged state can still contain occasional odd heuristic artifacts outside common benchmark phrasing
- episodic turn recall fallback still sometimes returns "I recalled these prior turns..." if merged state is empty

## 4. Routing System
Current route types from `brain_query_router.py`:
- `factual_question`
- `memory_question`
- `state_update`
- `planning_request`
- `goal_request`
- `transform_request`
- `hybrid_request`
- `unknown`

### 4.1 Scoring model
Routing is additive heuristic scoring.

Scoring inputs:
- regex cue matches
- prompt length
- parser-derived sparse/dense structure
- hybrid cue combinations

Core cue families:
- factual: question marks and question-openers
- memory: recall phrases like `what did i say`, `earlier`, `from memory`
- planning: `help me`, `plan`, `safest way`, `step by step`
- goal: `my goals`, `which option`, `trade-off`
- transform: `rewrite`, `summarize`, `edit this`
- hybrid: `use what you know`, `based on what you know`
- state_update: declarative state phrases like `update:`, `actually`, `done`, `blocked`, `for my shipping checklist`

Selection behavior:
- best route wins by highest numeric score
- if best score < 1.5, route falls back to `unknown`
- confidence is derived from best score plus margin over runner-up

### 4.2 Memory-question routing logic
Memory routing happens in two layers:

1. base router:
- regex memory cues add `memory_question` score

2. prototype override:
- `_should_force_memory_route(...)` checks `_looks_like_memory_recall_query(...)`
- if the route is not already memory and either `retrieve_stateful_memory(...)` or `retrieve_episodic_turns(...)` finds relevant prior state, `_promote_route_decision(...)` rewrites the route to `memory_question`
- the promotion also updates `score_breakdown` so trace bookkeeping matches the chosen route

### 4.3 What still falls into `unknown`
Turns with weak declarative or question cues and no state/update patterns can still hit:
- `unknown_route_fallback`
- deterministic explanation text: `I could not confidently route this prompt...`

Current evidence:
- recent targeted runs show benchmark-style declarative updates mostly shifted from `unknown` to `state_update`
- older result file `tests_memory_validation_run3/results.json` still shows meeting-update turns going to `unknown`, which is useful evidence of what changed recently

### 4.4 Benchmark category influence
Benchmark category does not currently steer BrAIn route selection at runtime.

It affects:
- benchmark scoring
- result interpretation

It does not affect:
- `route_user_prompt(...)`
- `run_prompt(...)`
- memory-recall trigger logic

## 5. LLM Interface
### 5.1 Actual role of the LLM
The LLM is a presentation/output layer, not the structure-deciding layer.

BrAIn decides:
- route
- memory retrieval
- merged state
- candidate paths
- selected structured result

The LLM layer does:
- explain a reasoning decision in normal language
- answer factual/hybrid prompts when direct wording is desired

### 5.2 Interface abstraction
Defined in `llm_interface.py`.

Providers:
- `DeterministicExplanationProvider`
- `SubprocessExplanationProvider`
- `OllamaExplanationProvider`

Facade:
- `HybridLLMInterface`

Provider selection:
- `provider="auto"` checks `BRAIN_LLM_PROVIDER`
- else checks `BRAIN_LLM_COMMAND`
- else defaults to deterministic
- default Ollama model is `qwen3:8b`

### 5.3 Direct answer mode
Used by:
- `factual_question`
- `hybrid_request`
- benchmark baseline LLM

Payload:
```json
{
  "user_input_text": "...",
  "route_type": "...",
  "structured_context": { ... }
}
```

For hybrid requests, `structured_context` includes:
- `route_reasons`
- `reasoning_summary`
- `merged_memory_state`
- `retrieved_memory_count`
- `selected_path`
- `final_result`

### 5.4 Memory question mode
Memory questions usually do not call the external provider for wording.
Instead:
- `_build_memory_recall_explanation(...)` returns a deterministic `LLMExplanationResult`
- it answers directly from `merged_memory_state` when possible
- if not, it falls back to recalled-turn summaries or generic memory-found/not-found text

So memory recall is mostly not an LLM-native wording path right now.

### 5.5 Planning/reasoning mode
Planning, goal, transform, and non-direct hybrid routes call:
- `HybridLLMInterface.explain(...)`

Payload includes:
- `result_brain_input`
- full trace summary
- trace steps
- selected path
- retrieved memories

### 5.6 What the LLM sees
Explain mode sees:
- raw user input
- final `BrainInputV1`
- serialized trace steps
- selected path
- retrieved memories
- full trace summary

Direct answer mode sees:
- raw user input
- route type
- whatever `structured_context` the prototype passes

The LLM does not automatically receive full raw conversation history from BrAIn except in the benchmark baseline, where the baseline model gets a `conversation_history` list in `structured_context`.

### 5.7 Deterministic fallback vs live model
Deterministic:
- always available
- converts structured results to text
- uses fixed canned direct-answer behavior
- often produces placeholder factual answers for unsupported direct questions

Live providers:
- subprocess provider shells out with JSON payload
- Ollama provider POSTs to `/api/generate`
- both fall back to deterministic wording if command/network/model use fails

Current benchmark reality:
- unless explicitly configured, BrAIn falls back to deterministic
- `BrainVsLLMRunner` marks deterministic-provider runs invalid by default unless `--allow-nonlive-provider` is used

## 6. Benchmark Integration
### 6.1 Where `benchmark_category` enters
In `brain_vs_llm_runner.py`:
- `benchmark_category = classify_benchmark_category(case)`

It is then used only by scoring functions in `brain_vs_llm_scoring.py`.

### 6.2 Prompt sequence execution
For each case:
- runner builds a fresh case-specific `BrainHybridPrototype`
- loops through `turns`
- each turn calls `hybrid.process_user_turn(turn_text, inspect=True)`
- all memory/turn state persists across those turns because the same prototype instance and same `brain_state` directory are reused within the case

### 6.3 Result fields produced
Per case, runner emits:
- case metadata
- `brain_output`
- `llm_output`
- score objects
- comparison eligibility and invalid reasons

Important `brain_output` fields:
- `structured_result`
- `reasoning_trace`
- `final_explanation`
- `route_type`
- `retrieved_memory_count`
- `recalled_turn_count`
- `sequence_memory_written`
- `sequence_stored_entry_count`
- `sequence_retrieved_memory_count`
- `sequence_recalled_turn_count`
- `llm_provider`
- `llm_metadata`
- `turn_outputs`

### 6.4 Scoring dependence on BrAIn output
Scoring reads:
- final answer text
- clarification state
- selected strategy
- retrieved/recalled counts
- benchmark keywords and forbidden errors

Memory scoring specifics from `brain_vs_llm_scoring.py`:
- `score_correctness(...)` for memory cases uses `memory_fidelity_keywords` or `expected_keywords`
- `score_memory_usage(...)` rewards recalled turns and retrieved memories
- `score_memory_fidelity(...)` combines keyword overlap, support bonus from recalled turns, and penalty from forbidden errors
- `score_hallucination_penalty(...)` penalizes unsupported memory-style phrasing or forbidden errors

### 6.5 What benchmark does not do
Benchmark does not:
- inject benchmark category into live routing
- directly alter memory retrieval thresholds
- directly alter reasoning depth/beam settings

## 7. Key Files and Responsibilities
- `brain_hybrid_prototype.py`: core runtime orchestrator. Core.
- `brain_input_v1.py`: `BrainInputV1` packet schema and codec-facing helpers. Core.
- `brain_parser.py`: raw text -> `BrainMemory` parser with fallback heuristics. Core.
- `brain_query_router.py`: route scoring and route decision object. Core.
- `brain_prompt_enrichment.py`: planning/goal/transform enrichment and clarification triggers. Important.
- `brain_goal_reward_v5.py`: top reasoning planner, goal rescoring, selected path. Core.
- `brain_goal_tracking_v6.py`: post-turn goal tracking summaries. Important.
- `brain_strategy_generation_v7.py`: post-turn strategy template generation. Important.
- `llm_interface.py`: live/deterministic/subprocess/Ollama provider abstraction and LLM payload construction. Core.
- `brain_llm_interface.py`: presentation-only deterministic formatter used by deterministic provider. Important.
- `brain_vs_llm_runner.py`: benchmark execution loop and output assembly. Core for benchmarking.
- `brain_vs_llm_scoring.py`: benchmark scoring rules. Core for benchmarking.
- `brain_vs_llm_prompt_set.py`: benchmark case definitions. Core for benchmarking.
- `brain_history.py`: persistent history log writes. Important but not central to routing/reasoning.
- `brain_reality_grounding_v4.py`: lower reasoning layer under `BrainGoalRewardV5`. Important but only partially inspected here.
- `brain_reasoning_engine.py`: lower reasoning runtime and trace step classes. Important but partially inspected here.
- `brain_concept_abstraction_v2.py`, `brain_planning_simulation_v3.py`, `brain_generative_expansion_v35.py`: deeper planning stack below the prototype. Important but lower-level.
- many codec/pattern modules: active infrastructure because `BrainInputV1` normalization uses them, but not directly responsible for high-level runtime decisions. Important infrastructure.
- example/demo files (`*_examples.py`, some older benchmark files): probably ancillary or legacy for understanding, not core runtime.

## 8. Current Strengths
- Structured control is strong: route selection, memory retrieval, reasoning, goal tracking, strategy generation, and final trace all have explicit packets.
- Memory recall is genuinely active now for benchmark-style current-state questions.
- Sequence memory works per case because state is persisted across turns inside case-specific state directories.
- Overwrite/correction logic is implemented and visible in `merged_memory_state`.
- The trace packet is rich enough for architecture review and likely already strong enough as supervision/input material for future BrAIn-native training.
- The LLM handoff is cleanly separated from structure selection.
- Benchmark integration is transparent: per-turn outputs and final traces expose what happened.

Concrete working examples from current outputs:
1. `memory_project_progress_status`:
   - answer: `Current project status: billing is blocked on legal review, login is done, onboarding is done.`
   - route: `memory_question`
   - merged state preserves both current and original values for updated items.
2. `memory_troubleshooting_steps_tried`:
   - answer: `You already tried: checked power, checked the contrast potentiometer.`
   - merged state uses `append_set` lists.
3. `memory_correction_chain_latest_wins`:
   - answer: `Your meeting is friday at 11 am.`
   - merged state reports `correction_chain_detected: true`.

## 9. Current Weaknesses
- State extraction is heavily heuristic and benchmark-family specific.
- Memory-question answering still falls back to prior-turn summaries if merged state extraction misses.
- Factual direct-answer quality is weak without a live provider; deterministic direct answer is mostly placeholder behavior except for a few canned cases.
- Strategy and goal-tracking layers are active, but user-facing memory wins do not depend much on them; they are more trace/reporting layers than recall-quality drivers.
- Benchmark full-suite runs can be slow and environment-sensitive because each case creates a fresh nested reasoning stack and state directory.
- Some lower reasoning/path schemas are only indirectly visible from the prototype because the stack is deep and layered.
- Generic memory retrieval exists but benchmark memory performance currently relies much more on explicit `state_memory` extraction than on broad episodic semantics.

Evidence-based partial/underused areas:
- `BrainPromptEnrichment` only changes behavior for planning/goal/transform/hybrid routes, not for factual or memory routes.
- `state_update` route writes memory cleanly, but it does not run reasoning and uses deterministic acknowledgement text only.
- `BrainLLMInterface` is presentation-only; it is not a planning partner.
- Benchmark category is not used to improve live routing even though benchmark output uses it heavily for evaluation.

## 10. Open Questions / Unclear Areas
- The exact lower-layer fields of `GroundedPlanningPath` and some deeper candidate-path internals were only partially inspected through `to_dict()` usage, not exhaustively reconstructed here.
- The full reasoning stack below `BrainGoalRewardV5` is clearly active, but this audit focuses on the top-level runtime and packet handoffs rather than every lower-layer scoring formula.
- It is not fully clear from this pass which older modules are still exercised outside the hybrid prototype entrypoint versus retained as earlier experiments.
- The repo contains many benchmark and prototype variants; this audit treats `brain_hybrid_prototype.py` plus `brain_vs_llm_runner.py` as the current operative path because that is what recent result files reflect.
- Full-suite benchmark timing behavior is environment-sensitive; recent targeted result files were easier to inspect than relying on one full fresh suite run.
