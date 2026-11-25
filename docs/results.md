# Results and reproduction

These are bounded research results from November 5, 2026, not a claim that a
model can generally think in pictures. The portfolio packaging update was made
on November 25. Replaying recorded evidence now does not make it a new experiment.

## Fresh image test

The fixed decoder saw 512 new four-color, 16-cell images. All earlier image
patterns were excluded; cases and policies were frozen before native inference.

| Policy | Correct accepted images | Incorrect accepted | Unavailable |
|---|---:|---:|---:|
| Verify only | 493/512 | 0 | 19 |
| One-cell recovery | 510/512 | 0 | 2 |
| Two-cell recovery | 511/512 | 0 | 1 |
| Compact 64-bit two-cell | 511/512 | 0 | 1 |

One image had three wrong cells and remained unreadable. The predefined image
coverage gate was at least 99% with no wrong accepts, not universal error-free
recovery. A genuine two-cell success and the three-cell failure are included
as feature fixtures.

Each checksum condition also received 4,512 one/two-cell injected corruptions,
2,240 three-cell perturbations, and missing/swapped metadata probes on four
predetermined sources. These are protocol controls, not extra independent
model-accuracy samples. Full and 64-bit checks recovered the in-range inputs
and rejected the tested out-of-range inputs. The 16-bit control falsely accepted
36 three-cell corruptions and rejected 66 correctable inputs because of
ambiguity. The 24/32-bit controls had no failures in that finite sample, but
were not promoted: 64 bits was the primary compact choice before testing.

## Fresh typed histories

Six histories contained 16, 64, or 128 records, two variants each: 416 new images
and 810 fresh-process queries across five conditions. Six owner/location slots
were queried at fixed checkpoints.

| Policy | Known facts correct | Available evidence | Wrong assertions |
|---|---:|---:|---:|
| Verify-only images | 25/52 | 72/108 | 0 |
| One-cell images | 52/52 | 108/108 | 0 |
| Two-cell images | 52/52 | 108/108 | 0 |
| Compact 64-bit images | 52/52 | 108/108 | 0 |
| Text control | 52/52 | 108/108 | 0 |

All ten image-readout errors here were single-cell errors, so two-cell recovery
showed no additional semantic gain over one-cell recovery in this sample.
Both 128-record histories stayed available with recovery, versus neither under
verification alone. There are only two such histories, and repeated checkpoints
within a history are dependent observations.

All 270 pending/missing-update probes returned UNKNOWN with empty evidence.
Earlier growing-history work had a two-cell error that blocked an entire
journal under one-cell recovery; that failure remains part of the research
history, not retroactively relabeled as success.

## Costs

The logical packet is four bytes. SHA256 uses 32 digest bytes; the compact tag
uses eight, and the JSON envelope shrank from 181 to 132 bytes. The compact tag
still exceeds the source payload. Both image modes still use 49 vision positions.

Across the six retained histories, full two-cell journals occupied 223,493,231
bytes; compact journals 223,472,859 bytes; text journals 4,581,422 bytes. Caches,
duplicated readouts and immutable index history are included. This prototype
shows **no storage advantage** over the matched text control and measures no
language-model context savings. Generic compact IDs also work in text.

## What can be reproduced from this checkout?

1. **Protocol demo:** `python -m brain_pixels demo`. Standard library only;
   scripted facts and checksum correction, no visual model.
2. **Learned readout regressions:** install `.[vision]` and run the fixture
   replay commands or test suite. Three real native feature exports exercise the
   actual frozen decoder, including its unresolved failure.
3. **Offline evidence replay:** `python tools/verify_evidence.py` reconstructs
   image recovery from recorded predictions and checks the published history
   query aggregates, source packet hashes and matched available contexts.
4. **New native inference:** requires a locally installed matching Strata
   encoder/projector and explicit configuration. A new run needs a freshly
   frozen dataset and must report all failures.

The compact [evidence bundle](evidence/2026-11-05/) includes cases, predictions,
normalized query records, summary metrics and source fingerprints. Personal
paths, runtime PIDs, multi-gigabyte caches and model weights are excluded. Full
native feature exports are retained in the original local research archive;
this public bundle is not advertised as a full encoder rerun.

The original independent audits covered complete exports and query responses.
The packaged verifier checks the smaller public evidence level described above.
The learned readout's bytes and profile are unchanged; packaging did not retrain
or retune it. The old 50-prompt structured-reasoning comparison is preserved as
historical development work in `docs/legacy/`, not as current pixel evidence.
