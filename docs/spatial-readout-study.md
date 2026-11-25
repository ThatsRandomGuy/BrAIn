# Spatial-feature readout study

This experiment tests a specific possible source of color-readout errors:
averaging projected vision tokens may discard useful spatial arrangement.
It is a readout experiment, not evidence of language-model context compression.
The production decoder, feature profile, journal and recovery radius are unchanged.

## Frozen protocol

- Reference training: the same 512 source images and native features used by the
  packaged overlap-ridge readout. Source image and feature hashes are checked.
- Fresh validation: 64 images. Fresh test: 512 additional images. Seed 2026112502 (after an infrastructure-interrupted attempt;see notes below).
  All earlier source patterns and quadrant-equivalent layouts are excluded.
- Candidates: concatenated 2x2 or 3x3 projected-token neighborhoods, each with
  ridge strengths 0.001, 0.01 and 0.1. Neighborhoods depend only on geometry.
- Selection: validation whole-image accuracy, then cell accuracy, then fewer
  feature dimensions, then smaller alpha. The test feature files must not exist
  when selection is saved. The choice and model fingerprint are frozen.
- Primary comparison: the selected spatial readout versus the unchanged packaged
  baseline on the untouched test set. No other candidate is promoted afterward.
- Gate: at least one percentage point improvement in complete-image accuracy,
  exact two-sided McNemar p <= 0.05, no lower cell accuracy, no lower correctly
  accepted two-cell-recovery coverage, and zero wrong accepted streams.
- Recovery: the same two-cell radius and 64-bit tag budget for both. A distinct
  study-scoped envelope binds the frozen plan. It is not falsely represented as
  a receipt under the baseline's production model identity.

Source, plan, training metadata and exclusions are saved before native inference.
The encoder executable and projector must match the frozen training fingerprints.
Only validation is exported before training/selection; test export starts later.
The study uses the installed CPU encoder and stops its process after evaluation.

## Costs

The image remains 224-square with 16 logical color cells and 49 projected vision
positions. Concatenation changes the external readout, not the number of vision
positions or source bits. The baseline uses 2,560 features per cell; concat4
uses 10,240 and concat9 uses 23,040. Decoder bytes, checksum bytes and unavailable
coverage must be reported. A larger external readout is not a smaller AI model.

## Reproduce

Install `.[vision,dev]`. To reproduce the historical training exports from the
published source records with the original matching native encoder build:

```sh
python -m research.export_training docs/evidence/spatial-readout-2026-11-25/training.json --config strata.local.json --output artifacts/reference-training
```

This explicitly supplied source list is synthetic. Exact feature hashes protect
against accidentally mixing encoder versions; other encoder builds need a new
compatible training/profile study rather than pretending to reproduce these bits.
Full historical feature caches are local artifacts, not checked into the repo.

Run a new frozen comparison with an explicit training manifest and exclusion list:

```sh
python -m research.run_spatial_readout --training-manifest artifacts/reference-training/training-manifest.json --exclude docs/evidence/spatial-readout-2026-11-25/excluded.json --config strata.local.json --output artifacts/my-spatial-study
python -m research.audit_spatial_readout artifacts/my-spatial-study
```

The published exclusions reproduce the registered generation for this study;
exclude its now-seen images too when registering a future *fresh* experiment.
Recorded test cases cannot be relabeled fresh by rerunning the commands.
The audit checks source generation, source fingerprints, selected validation
scores, every cached test prediction and independent integer-mask checksum search.
It checks evidence integrity, not encoder authentication or generalization.

The selected research model remains in local artifacts. It does not replace
`brain_pixels/assets/decoder.npz` or change the journal's saved policy/profile.

## Interrupted first attempt

The first registered seed,2026112501,stopped during test export when its terminal
session vanished. Its latest direct-written results checkpoint was incomplete.
The partial run remains archived and is not a confirmatory test. No test scores
were used to change candidates,selection rules or success criteria. The next
run uses atomic JSON checkpoints,a separately logged worker,and seed2026112502;
all576 source cases from the interrupted attempt are added to its exclusions.

## Completed result

The selected spatial readout did not pass the predefined improvement gate. See the [complete result and evidence](evidence/spatial-readout-2026-11-25/README.md). The production readout is not replaced by this PR.
