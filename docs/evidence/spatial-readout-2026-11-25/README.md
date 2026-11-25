# Spatial readout result: November 25, 2026

The selected spatial readout did not pass the predefined improvement gate. The production decoder and journal remain unchanged.

This study concatenates local projected vision features instead of averaging
them. Both decoders use the same 49 vision positions and source images. Training
reuses the same 512 frozen images; selection uses 64 fresh validation images,
then one locked candidate is evaluated on 512 untouched test images.

## Locked selection

Selected `concat4-0.001`: 62/64
validation images and 1022/1024 cells. The
candidate fingerprint was recorded before test features were exported.

| Representation | Ridge alpha | Correct validation images | Correct cells |
|---|---:|---:|---:|
|concat4|0.001|62/64|1022/1024|
|concat4|0.01|62/64|1022/1024|
|concat4|0.1|61/64|1021/1024|
|concat9|0.001|62/64|1022/1024|
|concat9|0.01|62/64|1022/1024|
|concat9|0.1|62/64|1022/1024|

## Held-out test

| Metric | Frozen overlap baseline | Selected spatial readout |
|---|---:|---:|
| Correct complete images |493/512|485/512|
| Correct cells |8171/8192|8160/8192|
| Correct accepted after the same two-cell recovery |511/512|511/512|
| Wrong accepted |0|0|
| Unavailable |1|1|
| Decoder bytes |1,639,660|6,554,860|
| Vision positions per image |49|49|

Paired whole-image outcomes: 5 improve,
13 regress. Accuracy difference:
-1.5625 percentage points. Exact two-sided McNemar
p = 0.0962524.

The predeclared gate required at least one percentage point improvement,
p <= 0.05, no lower cell accuracy, no lower correctly accepted recovery
coverage, and no wrong accepted streams. Gate passed: **False**.
All cases are retained. The checksum budget is eight bytes and the recovery
radius stays at two cells (1,128 alternatives); neither was expanded to make a
failed case pass. Integrity is bound to the study plan, not falsely to the old
production readout identity.

## Interpretation and cost

The observed raw accuracy is lower on this sample; p = 0.096 does not establish
a general harmful effect. The predefined improvement gate nevertheless fails,
and the larger model has no measured recovery-coverage advantage.

The spatial decoder is 4.00 times the baseline's size. Both consume the same
exported positions; this is a change to the external readout, not a smaller
language model or a proven context saving. The source packet still carries only
four bytes. This study tests color-symbol readout, not English understanding,
conversation extraction, long-history QA, or transfer to smaller models.

The first seed stopped during test export after its terminal session vanished;
the latest direct-written results checkpoint was incomplete. It is archived as
interrupted and excluded from confirmatory results. The replacement uses atomic
checkpoints and a hidden logged worker. Candidates and gates did not change.
All 576 source cases from the interrupted attempt were excluded from the new
seed, 2026112502. No partial test score was used to choose a model.

## Evidence and reproduction

[Protocol and commands](../../spatial-readout-study.md).
`plan.json` binds source fingerprints, training metadata, candidate definitions
and gates. `predictions.json` contains every recorded test prediction; no native
model weights or workstation paths are included.

```sh
python -m research.verify_spatial_evidence docs/evidence/spatial-readout-2026-11-25
```

This replays recorded source/prediction metrics, selection and checksum results.
It does not retrain the model or rerun native feature extraction. The independent
full local audit replays cached validation/test predictions and uses a separate
integer-mask checksum search; its receipt is `audit.json`. Full feature caches
and the non-default selected model remain local artifacts. Regenerate them with
an explicitly installed matching encoder and the published synthetic sources.

Source byte hashes are line-ending sensitive. `publication-source.json` records
both the frozen runtime hashes and LF-normalized publication hashes; the source
semantics did not change after freezing the completed run.
