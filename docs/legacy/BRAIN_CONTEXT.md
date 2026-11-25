# BrAIn — System Context & Living Documentation

## What This File Is

This document is the **single source of truth** for the BrAIn project.

Any tool, agent, or developer working on this codebase should read this first to understand:

* what BrAIn is
* how it works
* what problems it is solving
* where the system currently stands
* where it is going next

This file should be updated as the system evolves.

---

# Core Vision

BrAIn is a **brain-inspired semantic memory system**.

It is NOT just:

* a chatbot
* a text processor
* a database

It is an attempt to build a system that:

* understands structure instead of raw text
* compresses meaning instead of storing sentences
* grows vocabulary safely over time
* organizes knowledge with governance
* tracks its own decisions and history

---

# The Big Idea (Critical)

BrAIn is built on a key belief:

> Natural language is inefficient. Meaning can be stored in a more compact and structured way.

The long-term goal is to:

* convert language → structured meaning
* convert structured meaning → compact internal representation
* reuse that representation instead of reprocessing raw text

---

# Pixel / Visual Encoding Direction (VERY IMPORTANT)

A foundational idea of BrAIn is:

> We are attempting to represent words and meaning using structured pixel/color-based encodings.

This means:

* semantic primitives are mapped to colors
* meaning is placed into structured grids (like images)
* relationships are stored in separate visual structures
* the system reads and writes these like a machine-native language

This is NOT for human viewing.
This is an **internal compression language**.

### Why this exists

Traditional LLMs:

* store and process long token sequences
* repeatedly pay cost for the same meaning

BrAIn is exploring:

* storing meaning once
* reusing compact representations
* reducing effective context size

In simple terms:

> We are trying to turn words into structured "images" made of pixels/colors that represent meaning more efficiently than text.

---

# System Architecture

## 1. Parser (brain_parser.py)

Transforms raw input into structured semantic memory.

Outputs:

* actor
* intent
* action
* object
* topic
* modifiers
* negations
* constraints
* conditions
* time
* priority
* relations
* confidence

Key behavior:

* deterministic rules
* fallback inference
* aggressive but controlled unknown promotion

---

## 2. Memory Schema (BrainMemory)

Central structured representation of meaning.

Includes:

* semantic fields
* relationships
* confidence tracking

---

## 3. Codec (brain_codec.py)

Encodes semantic memory into:

### Core Grid

* fixed layout
* represents primary fields

### Relation Grid

* stores relationships separately
* preserves meaning structure

This is the first implementation of the pixel/visual encoding concept.

---

## 4. Primitive Registry

Maps semantic primitives → colors.

Supports:

* dynamic expansion
* persistence
* reuse across sessions

---

## 5. Confidence System

Tracks how reliable each piece of meaning is.

Types:

* pattern (strong)
* fallback (medium)
* promotion (weak)
* default (base)

---

## 6. Policy Layer (brain_policy.py)

Controls whether concepts are:

* registered
* provisional
* ignored

Prevents uncontrolled vocabulary growth.

---

## 7. Evidence System (brain_evidence.py)

Tracks repeated provisional concepts.

Promotes concepts when:

* seen enough times
* confidence is high enough

---

## 8. Merge System (brain_merge.py)

Handles aliasing:

Examples:

* scary → horror
* webapp → website

Prevents fragmentation.

---

## 9. Similarity System (brain_similarity.py)

Suggests merges based on:

* context overlap
* usage patterns

Does NOT auto-apply changes.

---

## 10. Merge Queue (brain_queue.py)

All suggested merges go through:

* pending
* approved
* rejected

This creates **governance over meaning changes**.

---

## 11. Pipeline (brain_roundtrip.py)

Full system flow:

text → parse → policy → evidence → merge → encode → decode → reconstruct

Also:

* logs history
* generates artifacts

---

## 12. BrainInputV1 / Native Reasoning

BrAIn now includes a native structured reasoning layer built around `BrainInputV1`.

This layer:

* accepts only structured `BrainInputV1` objects internally
* prefers pattern-first representations
* performs deterministic transforms over slots and relations
* uses flat, hierarchical, and evolved pattern registries to guide reasoning
* returns `BrainInputV1` outputs instead of raw text

Supported native operations include:

* transform
* compare
* merge
* negate
* generalize
* specialize

This is an important transition step toward BrAIn operating on its own internal semantic language instead of text.

---

## 13. History (brain_history.py)

Stores operational memory:

* ingests
* commands
* merge decisions

This is BrAIn remembering its own behavior.

---

## 14. Search (brain_search.py)

Allows querying:

* history
* stored artifacts

Provides early self-audit capability.

---

## 15. CLI (brain_cli.py)

Unified interface for:

* ingesting text
* managing merges
* searching system memory

---

## 16. BrAIn-Assisted LLM Pipeline

BrAIn now includes a full proof-of-concept pipeline that connects user text to native structured reasoning while keeping the LLM in an interface-only role.

Pipeline flow:

text -> parse_to_brain_input -> pattern-first encode -> store -> retrieve -> native reasoning -> BrainInputV1 result -> optional reconstruction -> LLM-facing explanation

Important properties:

* reasoning still happens only on structured `BrainInputV1`
* the reasoning engine now emits deterministic reasoning traces
* the LLM layer does not alter structure
* the LLM layer only explains and formats BrAIn outputs
* the full pipeline is benchmarked end to end

This is a key proof-of-concept milestone:

* BrAIn performs the reasoning
* the LLM acts as a presentation layer

---

## 17. Adaptive Pattern Evolution

BrAIn now includes an adaptive pattern evolution layer that improves reusable structural patterns over time.

This layer tracks per-pattern learning signals such as:

* usage_count
* success_rate
* compression_efficiency
* reuse_frequency
* temporal_score

Adaptive behavior includes:

* merging similar patterns
* splitting overly broad patterns
* promoting frequently co-occurring subpatterns
* pruning low-value patterns
* recommending the best learned pattern before falling back to full structure

Important constraint:

* evolution operates only on structured semantic representations
* reasoning remains deterministic
* text is not used inside the adaptive reasoning loop

This is BrAIn moving from a static structured system toward a self-improving semantic system.

---

## 18. BrAIn Reasoning Engine V1

BrAIn now includes a causal reasoning layer that chains stored structured patterns into multi-step reasoning paths.

This layer supports:

* pattern A -> pattern B -> pattern C chaining
* cause -> effect links
* action -> outcome links
* intent -> action links
* deterministic simulation of multiple possible paths
* structured output containing the chain, confidence, and predicted outcome

Validation behavior:

* correct chains are reinforced
* incorrect chains are weakened
* failed predictions create more specific replacement links
* repeated validation improves chain reliability over time

Important constraint:

* causal reasoning still operates only on structured patterns and relationships
* no text reasoning is used internally
* deterministic core behavior is preserved

This is BrAIn moving from a pattern recognition system toward a causal reasoning system.

---

## 19. Concept Abstraction Layer (BrAIn V2)

BrAIn now includes a concept abstraction layer that generalizes beyond individual stored links.

This layer supports:

* emergent concept nodes built from repeated structural link patterns
* generalized links such as `eat [concept-member] -> feel full`
* inference order of specific match -> concept member match -> generalized concept match
* multi-hop conceptual chains that reuse generalized first-step links with learned downstream effects
* confidence scaling based on abstraction support and converging candidate paths

Important constraint:

* concepts emerge from structural patterns instead of hardcoded text labels
* inference still operates only on `BrainInputV1`, slots, relations, and learned links
* exact observed transitions are still stored so deterministic causal memory remains intact

This is BrAIn moving from a causal reasoning system toward a conceptual reasoning system.

---

## 20. BrAIn Planning & Simulation Engine (V3)

BrAIn now includes a planning layer that simulates multiple possible future chains and ranks them as decisions.

This layer supports:

* multi-path simulation over stored structured futures
* deeper chain expansion up to 5+ steps
* path scoring using confidence, desirability, and risk
* selection of the best path by deterministic decision score
* reuse of concept nodes and generalized links when expanding futures
* feedback loops that reinforce observed futures and update learned outcome values

Important constraint:

* planning still operates only on structured `BrainInputV1` states and learned links
* no text reasoning is used internally
* deterministic structural scoring remains the core decision mechanism

This is BrAIn moving from a conceptual reasoning system toward a decision-making system.

---

## 21. Generative Expansion Layer (BrAIn V3.5)

BrAIn now includes a generative fallback layer that seeds structured futures when direct reasoning paths are missing.

This layer supports:

* concept-based expansion when no direct path exists
* low-confidence seed path generation from concept links and generalized structures
* integration of generated paths into the planning and scoring stack
* promotion of successful generated paths into learned exact links
* pruning of repeatedly incorrect generated expansions through deterministic feedback

Important constraint:

* generation still uses only structured concept links, learned generalized patterns, and `BrainInputV1`
* no text reasoning is used internally
* once paths are generated, scoring and selection remain deterministic

This is BrAIn moving from a reactive reasoning system toward a generative reasoning system.

---

# Current State

BrAIn currently supports:

* semantic parsing
* structured memory representation
* grid-based encoding
* reversible decoding
* unknown concept promotion
* confidence tracking
* controlled vocabulary growth
* evidence-based promotion
* alias merging
* merge suggestions
* merge approval workflow
* operational history
* search over history and artifacts
* CLI interaction
* BrainInputV1 pattern-first encoding
* native reasoning over structured semantic inputs
* deterministic reasoning traces
* end-to-end BrAIn-assisted reasoning pipeline
* LLM interface layer for explanation only
* adaptive pattern evolution with temporal learning
* success-aware structural pattern recommendation
* multi-step causal reasoning over stored patterns
* deterministic reasoning-chain simulation and validation
* emergent concept nodes and generalized conceptual links
* concept-backed inference over unseen structured inputs
* multi-path future simulation and deterministic path selection
* learned structural desirability and risk scoring for outcomes
* concept-driven fallback generation for missing futures
* promotion of successful generated paths into stable learned structure

This is already a functional semantic memory engine.

---

# Where We Are Going

## Short-Term Goals

### 1. Compression Validation

* test if grid/pixel encoding reduces context cost
* compare against raw text storage

### 2. Memory Consolidation

* summarize repeated inputs
* create digest layers
* reduce redundancy

### 3. Better Search

* filter by type, time, confidence
* structured queries

### 4. Merge Intelligence

* improve suggestion accuracy
* track rejected merges

### 5. Native Reasoning Expansion

* grow BrainInputV1-native operations
* reason without text reconstruction
* increase reusable abstract pattern coverage

---

## Mid-Term Goals

### 6. Semantic Graph

* move beyond flat relations
* build connected concept networks

### 7. Multi-Tier Memory

* hot / warm / cold storage
* compression layers

### 8. Better Internal Encoding

* denser grids
* semantic blocks
* alternative layouts

---

## Long-Term Goals

### 9. True Internal Language

* move beyond text entirely
* operate on compact symbolic structures

### 10. Self-Reflection

* system explains its reasoning
* system identifies weak knowledge areas

### 11. Visual/Pixel Memory Experiments

* test pixel-based semantic density
* experiment with image-like encoding for meaning

---

# Key Principles

* Deterministic before intelligent
* Structure over raw data
* Controlled growth
* No silent mutations
* Meaning over text

---

# Prompt Template for Codex / Agents

Use this when starting a new session:

```
You are working on a project called BrAIn.

Read BRAIN_CONTEXT.md first.

This system:
- parses text into structured semantic memory
- encodes meaning into grid/pixel-like representations
- maps primitives to colors
- stores relationships separately
- tracks confidence and evidence
- grows vocabulary safely
- uses merge approval workflows
- logs operational history

IMPORTANT:
BrAIn is exploring representing words and meaning as structured pixel/color encodings to reduce context size and create a reusable internal language.

Your job:
- maintain deterministic behavior
- improve system correctness
- preserve architecture consistency
- help evolve the internal representation

Do NOT:
- introduce uncontrolled randomness
- bypass merge governance
- collapse structured memory into plain text shortcuts
```

---

## 22. Reality Grounding Layer (BrAIn V4)

Goal:

Enable BrAIn to distinguish between plausible and implausible outcomes before final selection.

Implemented:

- `brain_reality_grounding_v4.py`
- `brain_reality_grounding_v4_benchmark.py`
- structural plausibility scoring over V3.5 candidate paths
- concept consistency checks for concept and generated transitions
- cross-path support scoring
- grounding signal registry with negative reinforcement
- root-to-outcome grounding signals
- path filtering before final decision selection

Grounding signals:

- outcome frequency
- pattern consistency
- concept alignment
- cross-path agreement
- unreliable-pattern penalties

Behavior:

- V4 wraps V3.5 instead of replacing it
- V3.5 still generates structural candidate paths
- V4 rescales each path with plausibility
- low-plausibility paths can be filtered
- selected outcomes feed grounding feedback back into memory

Benchmark:

- report: `brain_output/benchmarks/brain_reality_grounding_v4_50.json`
- markdown: `brain_output/benchmarks/brain_reality_grounding_v4_50.md`
- scenarios: `50`
- decision accuracy: `0.38`
- unrealistic selection rate: `0.26`
- unrealistic selection reduction vs V3.5: `0.1036`
- outcome prediction quality: `0.48`
- average selected plausibility: `0.3599`

Interpretation:

- V4 successfully reduces unrealistic outcomes
- first-pass grounding is conservative and still over-prunes some useful paths
- next iteration should preserve the drift reduction while recovering decision accuracy

---

## 23. Goal & Reward System (BrAIn V5)

Goal:

Enable BrAIn to choose grounded futures based on explicit desired outcomes.

Implemented:

- `brain_goal_reward_v5.py`
- `brain_goal_reward_v5_benchmark.py`
- goal nodes with weighted seek / avoid targets
- goal-aligned path rescoring on top of V4 grounded planning
- goal outcome memory with success and reward history
- decision history tracking
- deterministic reward feedback into goal memory
- multi-goal weighted decision selection

Behavior:

- V5 wraps V4 instead of replacing it
- V4 still supplies grounded candidate paths
- V5 adds goal alignment and learned reward priors
- final selection uses grounded score + goal alignment + reward memory
- successful goal-aligned decisions reinforce goal outcome reliability

Benchmark:

- report: `brain_output/benchmarks/brain_goal_reward_v5_50.json`
- markdown: `brain_output/benchmarks/brain_goal_reward_v5_50.md`
- scenarios: `50`
- decision accuracy: `0.6`
- decision accuracy improvement vs V4: `0.22`
- goal alignment accuracy: `0.6`
- average goal alignment: `0.6276`
- decision quality: `0.6137`
- decision quality improvement vs V4: `0.0531`
- outcome prediction quality: `0.6`
- reward improvement first 10 vs last 10: `0.8859`

Interpretation:

- explicit goals materially improve selection quality
- repeated goal feedback produces strong late-round learning
- BrAIn is now moving from grounded reasoning toward goal-driven decision making

---

## 24. BrAIn + LLM Interactive Prototype

Goal:

Build the first real interactive hybrid prototype where BrAIn performs memory and reasoning while the LLM remains an interface layer.

Implemented:

- `brain_hybrid_prototype.py`
- `interactive_brain_shell.py`
- `llm_interface.py`
- `brain_hybrid_prototype_examples.py`
- persistent episodic memory store for prompts, selected results, and traces
- structured reasoning trace preserved separately from human-readable output
- configurable storage policy for prompts, selected results, traces, and feedback
- debug / inspection mode for memory, candidate paths, selected path, and final result
- optional subprocess-backed LLM explanation provider via `BRAIN_LLM_COMMAND`
- terminal REPL commands for `/memory`, `/last`, `/paths`, `/goals`, `/debug`, `/raw`, and `/store`

Pipeline:

text input
-> parse to `BrainInputV1`
-> retrieve relevant episodic memory
-> run V5 goal / reward planning
-> preserve full structured trace
-> project the best goal-aligned state for presentation
-> send the fixed structured result + trace to the LLM explanation layer

Important property:

- BrAIn remains the memory and reasoning substrate
- the LLM only explains the supplied structure
- the LLM does not replace or override BrAIn decisions

Example outputs:

- report: `brain_output/examples/brain_hybrid_prototype_examples_10.json`
- markdown: `brain_output/examples/brain_hybrid_prototype_examples_10.md`
- scenarios: `10`

Interpretation:

- the system now supports real prompt -> memory -> reasoning -> explanation loops
- memory can be inspected manually after each prompt
- the hybrid architecture is now observable enough for real prototype testing

---

## 25. Strategy Generation & Action Decomposition (BrAIn V7)

Goal:

Enable BrAIn to move from single best actions toward deterministic multi-step strategies.

Implemented:

- `brain_strategy_generation_v7.py`
- strategy objects with goals, ordered actions, step dependencies, and tradeoff summaries
- deterministic action decomposition into 3-7 step plans
- multiple strategy variants per turn
- structural strategy scoring over goal alignment, plausibility, risk, completeness, and reinforcement
- lightweight strategy feedback memory that reinforces successful patterns and weakens stalled ones
- terminal output sections for strategy plan, step sequence, and alternatives

Behavior:

- V7 sits after V5/V6 decision selection instead of replacing BrAIn reasoning
- BrAIn still selects grounded, goal-aware outcomes first
- V7 turns those outcomes into multi-step execution strategies
- strategy generation stays deterministic and template-driven
- no LLM reasoning is used internally for strategy creation

Interpretation:

- BrAIn is now moving from action selection toward structured strategy planning
- the system can propose concrete multi-step execution paths with visible tradeoffs

---

## 26. BrAIn vs LLM Comparison Harness

Goal:

Enable direct side-by-side evaluation of the BrAIn hybrid system against an LLM-only baseline.

Implemented:

- `brain_vs_llm_runner.py`
- `brain_vs_llm_scoring.py`
- `brain_vs_llm_prompt_set.py`
- `README_BRAIN_VS_LLM.md`
- saved comparison artifact at `tests/results.json`

Behavior:

- runs the same prompt suite through both paths
- BrAIn path uses the normal hybrid pipeline unchanged
- baseline path uses the same `HybridLLMInterface` provider selection without BrAIn reasoning
- stores structured BrAIn outputs, raw baseline outputs, and deterministic heuristic scores

Important constraint:

- this harness wraps BrAIn for evaluation only
- it does not move reasoning into the LLM
- it does not modify BrAIn decision logic

---

# Final Note

BrAIn is not trying to compete with existing LLMs directly.

It is trying to answer a different question:

> Can meaning be stored, compressed, and reused more efficiently than raw text?

Everything in this project should move toward answering that.
