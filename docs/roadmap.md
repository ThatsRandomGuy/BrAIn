# Roadmap

The goal is to preserve useful context and improve smaller models. Faster
inference is secondary. The broad hypothesis remains open.

| Stage | Status | Evidence needed to advance |
|---|---|---|
| Typed, persisted image memory | Working bounded prototype | Keep correction, uncertainty, isolation and missing-update contracts |
| Frozen visual symbol readout | Feasible, imperfect | Retain unseen-image and out-of-range failures |
| Bounded recovery | Working on measured cases | Measure long-history coverage and redundancy costs on independent runs |
| Portable package and portfolio demo | Implemented | Clean-checkout tests, CI and reproducible public fixtures |
| Better readout or error-correcting encoding | Next research step | Freeze training/validation/test before inference; improve residual failures without tuning on test cases |
| Lower storage and metadata overhead | Open | Count codebook, decoder, integrity data, caches and retained history; compare with compressed text |
| Actual language-model context benefit | Not demonstrated | Equal task accuracy with fewer complete input positions than the best matched text/retrieval control |
| Smaller-model transfer and application | Not tested | Independently confirm on a smaller model with explicitly isolated runs |

For the context experiment, an initial engineering target is at least 20%
fewer complete input positions at equal answer accuracy, followed by an
independent confirmation set. This is a target, not an achieved result or a
known feasibility bound. Include legends, instructions, dictionaries and
adapter overhead. Use an identical information budget and source ground truth.

The immediate priority is a stronger readout or explicitly costed error-correcting
representation. The remaining three-cell failure should guide a fresh experiment,
not trigger a post-hoc larger search until the old test passes. More reliable
transport alone cannot establish a pixel-specific context advantage.

## November 25 spatial-readout study

The selected spatial readout did not pass the predefined improvement gate. [Result, costs and limitations](evidence/spatial-readout-2026-11-25/README.md). The next research decision should use this held-out result; production profile changes require a separate reviewed change and fresh journal evaluation.
