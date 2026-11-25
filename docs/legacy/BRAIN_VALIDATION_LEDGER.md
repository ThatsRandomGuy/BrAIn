# BrAIn Validation Ledger

This document is the current single-source index for BrAIn validation work.

Its job is to keep us from losing track of what has already been tested, what each test proves, how each newer layer stacks up against the previous one, and where the backing data lives on disk before the project is pushed to GitHub.

## How To Use This File

- Read this file first when you need the current state of testing.
- Treat the linked JSON, log, and Markdown artifacts as source evidence.
- Add new tests here as new rows or new sections instead of leaving them only in `tests/` or `brain_output/`.
- When a newer layer supersedes an older one, keep both and record the comparison instead of overwriting history.

## Canonical Artifact Locations

- Benchmark reports: `brain_output/benchmarks/*.md`
- Benchmark raw data: `brain_output/benchmarks/*.json`
- Cache and reasoning probes: `tests/*probe*.json`
- Probe logs: `tests/*probe*.log`
- Reasoning graph sample report: `brain_output/reasoning_graph_probe_report.md`
- Entry-level reasoning graph tests: `tests/test_reasoning_graph_layer.py`

## Validation Stack

| Layer | Meaningful Test | Current Result | Stacks Up Against | Primary Artifact |
| --- | --- | --- | --- | --- |
| Compression and encoding | `brain_benchmark` | Hybrid compressed output reached `45` bytes total vs `288` raw bytes; hierarchical and evolved encodings outperformed flat and binary baselines on the sample set | Baseline raw text storage and earlier flat encodings | `brain_output/benchmarks/brain_benchmark.md` |
| Structured transform pipeline | `brain_assisted_pipeline_20` | Structure preservation, BrainInput correctness, trace accuracy, LLM alignment, and pattern reuse all reported `1.0` across `20` cases | Text-only or opaque LLM-only transform workflows | `brain_output/benchmarks/brain_assisted_pipeline_20.md` |
| Native reasoning lineage | `brain_reasoning_engine_v1_50`, `brain_native_reasoning_50`, `brain_goal_reward_v5_50`, `brain_reality_grounding_v4_50`, `brain_planning_simulation_v3_50`, `brain_concept_abstraction_v2_50`, `brain_generative_expansion_v35_50` | Module-specific benchmark reports preserved in `brain_output/benchmarks/`; these represent the pre-hybrid reasoning evolution path | Earlier single-module reasoning passes | `brain_output/benchmarks/` |
| Cache short-circuit | `cache_short_circuit_probe` | First factual turn used deterministic direct answer; later turn used cache and skipped LLM invocation | Pure deterministic direct answer on every factual turn | `tests/cache_short_circuit_probe_state/cache_short_circuit_probe.json` |
| Cache ranking | `cache_ranking_probe` | Cached factual query selected the stronger of two seeded answers and answered from cache | Simple "first hit" cache reuse | `tests/cache_ranking_probe_state/cache_ranking_probe.json` |
| Conflict handling | `cache_conflict_probe` | Conflicting entries were detected; the wrong entry was penalized and the winner was reinforced across repeated runs | Ranking-only cache selection | `tests/cache_conflict_probe_state/cache_conflict_probe.json` |
| Reinforcement | `cache_reinforcement_probe` | Repeated successful selections increased reuse count, successful uses, and confidence for the winner over multiple runs | Static cache scores without learning | `tests/cache_reinforcement_probe_state/cache_reinforcement_probe.json` |
| Verification and confidence caps | `cache_verification_probe` | Verified consensus entries were able to beat a previously over-reinforced wrong entry; confidence caps and uncertainty penalties were applied | Reinforcement-only cache selection | `tests/cache_verification_probe_state/cache_verification_probe.json` |
| Multi-hop cache reasoning | `cache_multihop_probe` | Three cached entries were merged into one deterministic explanation answer with `multi_hop_used=true` | Single-entry cache response | `tests/cache_multihop_probe_state/cache_multihop_probe.json` |
| Reasoning-path translation | `cache_reasoning_translation_probe` | Structured explanation template produced a fluent explanation from a reasoning path | Raw merged-fact output without explanation shaping | `tests/cache_reasoning_translation_probe_state/cache_reasoning_translation_probe.json` |
| Persistent global concept graph | `persistent_graph_probe_v2` | Multi-hop reasoning graph edges were persisted and reusable across runs; global graph size reached `5` nodes / `4` edges in the sample | Ephemeral per-turn multi-hop graph only | `persistent_graph_probe_v2.json` |
| Entry-level selection reasoning graph | `reasoning_graph_probe_report` and `test_reasoning_graph_layer` | New graph rescoring can promote supported and causal entries above higher raw-score entries; contradiction penalties also suppress bad candidates | Previous base-score-only cache selector | `brain_output/reasoning_graph_probe_report.md` and `tests/test_reasoning_graph_layer.py` |

## What Each Test Currently Proves

### 1. Compression and Encoding

Source: `brain_output/benchmarks/brain_benchmark.md`

Current result:
- Total raw text size: `288` bytes
- Total hybrid size: `45` bytes
- Hybrid selection counts: `{'evolved_pattern': 3, 'hierarchical_pattern': 3}`
- Pattern-first BrainInput sequence reduced semantic units by `38.017%`

Why it matters:
- This is the strongest preserved proof that BrAIn is not just a reasoning shell; it also stores structured state efficiently.
- It is the baseline for the project's "structured memory beats raw text" claim.

Stacks up against previous work:
- Beats raw text and earlier fixed, sparse, bit-packed, delta, and flat-pattern encodings on the tracked sample set.
- Establishes the memory substrate the later reasoning layers depend on.

### 2. Structured Transform Pipeline

Source: `brain_output/benchmarks/brain_assisted_pipeline_20.md`

Current result:
- Case count: `20`
- Structure preservation rate: `1.0`
- BrainInput correctness rate: `1.0`
- Trace accuracy rate: `1.0`
- LLM alignment rate: `1.0`
- Pattern reuse rate: `1.0`

Why it matters:
- This is the clearest end-to-end proof that BrAIn can perform deterministic structure-preserving operations before any later cache or graph work.
- It validates the "text -> structure -> reasoning trace -> user-facing text" pipeline.

Stacks up against previous work:
- Goes beyond compression alone by showing BrAIn can preserve semantic structure through transforms and still emit aligned user-facing text.

### 3. Cache Short-Circuit

Source: `tests/cache_short_circuit_probe_state/cache_short_circuit_probe.json`

Current result:
- The first factual turn ran as a deterministic direct answer path.
- A later turn used cached knowledge and short-circuited without LLM invocation.

Why it matters:
- This is the first meaningful proof that the prototype can stop paying the "answer from scratch" cost every turn.

Stacks up against previous work:
- Improves on direct-answer-only factual flow by introducing reusable cached knowledge.

### 4. Cache Ranking

Source: `tests/cache_ranking_probe_state/cache_ranking_probe.json`

Current result:
- Seeded entry ids: `1`, `2`
- Selected provider: `cache`
- Selected answer: `The sky appears blue during daylight.`

Why it matters:
- This proves the cache is not only hit-or-miss; it can rank among multiple candidates.

Stacks up against previous work:
- Improves on short-circuit-only behavior by choosing among matching cache entries instead of accepting the first acceptable match.

### 5. Conflict Resolution

Source: `tests/cache_conflict_probe_state/cache_conflict_probe.json`

Current result:
- Conflict detected between entry ids `1` and `2`
- Wrong entry was penalized from `0.75 -> 0.70`, then `0.70 -> 0.65`
- Winning entry was reinforced from `0.80 -> 0.82`, then `0.82 -> 0.84`

Why it matters:
- This is the preserved proof that BrAIn can distinguish competing cached answers and adapt future ranking behavior.

Stacks up against previous work:
- Improves on ranking-only cache behavior by adding disagreement handling and loss and win consequences.

### 6. Reinforcement Learning

Source: `tests/cache_reinforcement_probe_state/cache_reinforcement_probe.json`

Current result:
- Repeated selections increased reuse count and successful uses for the selected answer.
- The first run still came from deterministic answering, and later runs reused the cached result.

Why it matters:
- This is the strongest prototype-level proof that factual answer selection is stateful across time.

Stacks up against previous work:
- Builds directly on conflict handling and turns one-off selection into learned preference.

### 7. Verification and Confidence Stabilization

Source: `tests/cache_verification_probe_state/cache_verification_probe.json`

Current result:
- Scenario: `wrong_entry_reinforced_then_correct_entries_introduced`
- A previously reinforced wrong answer (`The sky is green.`) was capped and penalized.
- Verified consensus entries received `verification_bonus=4.0`
- Correct verified entry won despite the older wrong entry starting from a high confidence value.

Why it matters:
- This is the test that prevents reinforcement from becoming runaway self-confirmation.

Stacks up against previous work:
- Improves on reinforcement-only selection by allowing later evidence and consensus to pull the system back toward the correct answer.

### 8. Multi-Hop Cache Reasoning

Source: `tests/cache_multihop_probe_state/cache_multihop_probe.json`

Current result:
- `multi_hop_used=true`
- `merged_entry_ids=[1, 2, 3]`
- Resolved answer: `The sky looks blue during the day because blue light scatters strongly in Earth's atmosphere.`

Why it matters:
- This is the first strong proof that BrAIn can compose several cached fragments into one answer instead of only selecting a single stored response.

Stacks up against previous work:
- Improves on single-entry cache ranking by allowing deterministic composition across related entries.

### 9. Reasoning Translation

Source: `tests/cache_reasoning_translation_probe_state/cache_reasoning_translation_probe.json`

Current result:
- Structured explanation includes:
  - `main_fact`
  - `cause`
  - `supporting_detail`
  - `context`
  - `template_type`
- Final explanation: `The sky appears blue during the day because shorter blue wavelengths of sunlight scatter more strongly in Earth's atmosphere.`

Why it matters:
- This is the prototype proof that BrAIn can turn internal reasoning structure into readable explanation text without giving up inspectability.

Stacks up against previous work:
- Improves on multi-hop fact merging by producing a shaped explanation instead of only concatenated facts.

### 10. Persistent Global Concept Graph

Source: `persistent_graph_probe_v2.json`

Current result:
- `multi_hop_used=true`
- New graph edges persisted on run 1
- Global graph size reached `5` nodes and `4` edges in the sample
- Persistent graph edges were available for reuse on later runs

Why it matters:
- This is the preserved proof that BrAIn can store reasoning structure beyond a single turn.

Stacks up against previous work:
- Improves on transient multi-hop reasoning by persisting concept relationships as reusable graph knowledge.

### 11. Entry-Level Reasoning Graph Selection Layer

Sources:
- `brain_output/reasoning_graph_probe_report.md`
- `tests/test_reasoning_graph_layer.py`

Current result:
- In the support / causal sample, a more explanatory entry won after graph rescoring even though it did not have the best raw score.
- In the contradiction sample, the contradictory green-sky entry was pushed down by contradiction penalties.
- Unit tests passed for:
  - graph promotion of supported explanatory entries
  - cache ranking using `final_score` instead of only base score

Why it matters:
- This is the newest layer and the first implementation of the user-requested "reasoning graph over entries" rather than only the older concept-level multi-hop graph.

Stacks up against previous work:
- Improves on base-score-only cache selection by incorporating pairwise entry relationships:
  - `SUPPORTS`
  - `CONTRADICTS`
  - `EXTENDS`
  - `CAUSES`

Known limitation from the current report:
- `EXTENDS` is likely too strong in contradiction-heavy clusters and can let a paraphrase outrank a cleaner direct answer.

## Current Comparative Read

The current progression of capability is:

1. BrAIn proved it can encode structured memory compactly.
2. BrAIn proved it can preserve and transform semantic structure deterministically.
3. BrAIn proved it can reuse cached factual knowledge.
4. BrAIn proved it can rank, penalize, reinforce, and verify cached knowledge over time.
5. BrAIn proved it can merge multiple cached entries into one explanation.
6. BrAIn proved it can persist concept-level reasoning graphs across runs.
7. BrAIn now also has an entry-level reasoning graph that changes final selection, not just explanation composition.

That last step is the main difference between the current system and the previous cache pipeline:

- Previous cache selector: choose the best entry from scalar ranking features only
- Current cache selector: choose the best entry after scalar ranking plus graph evaluation

## Gaps We Already Know About

These are current open issues already supported by the stored artifacts:

- The entry-level `EXTENDS` bonus is too aggressive in some contradiction-heavy cases.
- Query intent is not yet explicitly part of graph scoring, so `why` and `what color` can reward similar relationships when they should not.
- The older persistent graph is concept-level, while the new graph selector is entry-level; they coexist, but are not yet unified.
- Some validation still lives in probe JSONs rather than in a single executable benchmark runner.

## Recommended Preservation Checklist Before GitHub

- Keep `brain_output/benchmarks/` in the repo or export it to a release-like `docs/artifacts/` folder.
- Keep the `tests/*probe*.json` and `tests/*probe*.log` files; they are the strongest evidence for prototype behavior shifts.
- Keep `brain_output/reasoning_graph_probe_report.md` because it is the clearest sample report for the new reasoning-graph selector.
- Keep `BRAIN_ARCHITECTURE_AUDIT.md` because it explains the runtime system that produced these artifacts.
- Update this ledger every time a new benchmark, probe, or reasoning layer lands.

## Quick Links

- Architecture audit: `BRAIN_ARCHITECTURE_AUDIT.md`
- Compression benchmark: `brain_output/benchmarks/brain_benchmark.md`
- Assisted pipeline benchmark: `brain_output/benchmarks/brain_assisted_pipeline_20.md`
- Reasoning graph sample report: `brain_output/reasoning_graph_probe_report.md`
- Reasoning graph tests: `tests/test_reasoning_graph_layer.py`
