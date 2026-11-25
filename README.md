# BrAIn

[![Tests](https://github.com/ThatsRandomGuy/BrAIn/actions/workflows/tests.yml/badge.svg)](https://github.com/ThatsRandomGuy/BrAIn/actions/workflows/tests.yml)

**Inspectable research into image-backed memory for small language models.**

BrAIn asks whether facts and context can be represented visually, retained across
sessions, and recovered reliably enough to help smaller AI models. The current
prototype turns bounded fact records into color grids, reads their projected
vision features with a learned decoder, and verifies the result before using it.

**Working today:** persisted memory, explicit corrections and uncertainty,
learned four-color readout, bounded error recovery, provenance, and conservative
recall. **Still unproven:** better language-model answers or lower context cost
than a strong text representation.

<img src="docs/memory-example.png" width="224" height="224" alt="Actual color-encoded memory record" />

*A real experiment record: “The canopy detector is in the violet shipping crate.”*
The meaning comes from an explicit [codebook and record](docs/memory-example.json).
This is a constructed memory encoding, not an image of an AI's internal thoughts.

## Try it in a minute

Python 3.11 or newer. The first demo uses only the standard library, needs no
model server, and downloads nothing:

```sh
git clone https://github.com/ThatsRandomGuy/BrAIn.git
cd BrAIn
python -m brain_pixels demo --output artifacts/first-demo
```

It renders a memory image, demonstrates two-cell checksum recovery, and recalls:

```text
SET location       -> blue equipment locker
MAYBE new location -> UNKNOWN
SET new location   -> padded transport crate
```

This is a **model-free protocol demo**. To exercise the learned visual decoder
on a real recorded native feature export:

```sh
python -m pip install -e ".[vision]"
python -m brain_pixels replay --features tests/fixtures/two-cell.sve --manifest tests/fixtures/two-cell.manifest.json --policy two-cell
```

That fixture has two actual misread cells; the frozen decoder plus bounded
recovery returns `accepted`. The preserved `unavailable` fixture demonstrates a
three-cell error that the same policy cannot repair. Neither replay loads a
language model or reads a source PNG to find the answer.

[Quick start and live encoder setup](docs/quickstart.md) ·
[Architecture](docs/architecture.md) · [Results](docs/results.md) ·
[Roadmap](docs/roadmap.md)

## Measured results

The November 5, 2026 pass froze its decoder, source patterns, policies and
primary 64-bit checksum choice before inference. It evaluated **928 fresh
images**, including six typed histories up to 128 records, and **810
fresh-process memory queries**.

| Recorded experiment | Verification only | One-cell recovery | Two-cell recovery |
|---|---:|---:|---:|
| Correct complete images, uniform grids | 493/512 | 510/512 | 511/512 |
| Known facts recalled, typed histories | 25/52 | 52/52 | 52/52 |

The 64-bit compact policy matched full two-cell recovery on this sample. The text
control answered 52/52 known-fact queries correctly. All 270 pending/missing-update
probes withheld answers safely. There were no incorrect accepted answers in the
primary conditions. All ten raw errors in the typed-history sample were
single-cell errors, so that sample does **not** show an additional semantic gain
from two-cell recovery.

Failures are part of the result: one uniform image remained unreadable, and a
16-bit checksum control falsely accepted 36 injected three-cell corruptions.
The compact envelope shrank from 181 to 132 bytes, but it still exceeds the
four-byte payload. Retained visual journals were about 223 MB versus 4.6 MB for
the matched text control, including cached features and metadata. Each image
still occupies 49 projected vision positions. **This is not evidence of context
compression or a storage advantage.**

[Evidence, limitations and reproduction levels](docs/results.md)

A separate [November 25 spatial-readout study](docs/evidence/spatial-readout-2026-11-25/README.md)
did not justify a larger decoder: 485/512 raw images versus 493/512 for the
baseline on a new held-out set, with roughly four times the decoder storage.
The default remains unchanged; the negative result is retained.

## Engineering focus

- **Never silently reuse stale memory.** Any missing, pending or unrecoverable
  active record makes the entire evidence bundle unavailable.
- **Keep meaning explicit.** SET, MAYBE and FORGET operate on typed location and
  owner slots; a versioned vocabulary maps IDs to names and aliases.
- **Separate accuracy from repair.** Receipts label recovered cells and search
  cost. Two-cell recovery examines at most 1,128 alternatives.
- **Make state inspectable.** Immutable revisions, atomic HEAD updates, source
  citations and saved readout policies survive process restarts.
- **Compare against text.** Identical packets and memory semantics prevent
  generic coding gains from being mislabeled as pixel-specific gains.

## Repository map

```text
brain_pixels/        Current portable codec, readout, recovery and journal CLI
  assets/            Frozen learned readout and versioned profile; no LLM weights
tests/               Current pixel-memory contracts and genuine readout fixtures
core/                Maintained structured-memory/reasoning prototype
  tests/             Core regressions, including the original graph tests
research/legacy/     Historical benchmark and example scripts
docs/                Architecture, methods, results and compact evidence
tools/               One-command checks and offline evidence verification
```

The older 50-prompt structured-reasoning report remains under
[docs/legacy](docs/legacy/README.md). It is historical development evidence,
not proof of the pixel hypothesis. The original graph tests remain active;
superseded prototype compatibility checks and machine-bound sweep tests are not
part of the new public suite. [Cleanup rationale](docs/testing.md).

## Develop

```sh
python -m pip install -e ".[vision,dev]"
python tools/run_checks.py
python tools/verify_evidence.py
python -m ruff check brain_pixels tests tools
python -m ruff format --check brain_pixels tests tools
```

CI runs the current pixel suite and all maintained core regressions on Windows
and Linux. The evidence verifier replays recorded predictions and query
aggregates; it does not pretend to rerun the original native inference.

See [CONTRIBUTING.md](CONTRIBUTING.md) for experiment and change guidelines.
