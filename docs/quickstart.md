# Quick start

## Model-free demo

From a checkout, run `python -m brain_pixels demo --output artifacts/first-demo`.
Choose a new output directory for each demo. It saves a real color image,
demonstrates checksum repair, and uses the text control for typed fact recall.
It does not run a vision or language model. This path works without NumPy.

## Persist named facts

```sh
python -m brain_pixels --store artifacts/my-memory init --backend text
python -m brain_pixels --store artifacts/my-memory add --op SET --subject "rescue scanner" --relation location --value "blue equipment locker"
python -m brain_pixels --store artifacts/my-memory query "Where is the evening scanner?"
```

The alias resolves to the rescue scanner. Later SET replaces the same slot;
MAYBE or FORGET makes it unknown. Owner and location slots are independent.
The default vocabulary is deliberately small and fixed; callers may provide a
validated codebook when creating a store through the Python API. Automatic
English extraction and live vocabulary migration are not implemented.

`inspect` reports the saved backend and policy. `clear` starts an empty generation
but retains old files. Each store is isolated. Queries supply no previous
conversation or hidden answer registry to a language model.

## Replay actual visual features

Install the optional visual dependency:

```sh
python -m pip install -e ".[vision]"
python -m brain_pixels replay --features tests/fixtures/correct.sve --manifest tests/fixtures/correct.manifest.json
python -m brain_pixels replay --features tests/fixtures/two-cell.sve --manifest tests/fixtures/two-cell.manifest.json
python -m brain_pixels replay --features tests/fixtures/unavailable.sve --manifest tests/fixtures/unavailable.manifest.json
```

The first two should be accepted; the last should remain unavailable. This runs
the actual frozen ridge decoder on native feature exports, then checks the
predicted colors against ingestion metadata. It does not inspect source PNGs
or fixture ground-truth labels. The fixtures are seen regressions, not fresh
performance claims. Tests check their hashes and predictions.

## Add new visual records with an installed encoder

The optional native integration uses an already installed Strata vision encoder
and matching projector/model files. BrAIn never downloads them or starts a
language server. The bundled readout is specific to the tested 49-position,
2,560-wide feature profile; other models require calibration and new validation.

Create a local configuration using `examples/strata-config.example.json` as a
template. Relative paths resolve from that configuration file. A full Strata
configuration with a `vision` section is also accepted. Keep the local file
outside version control.

```sh
python -m brain_pixels --store artifacts/visual-memory init --backend visual --policy compact-64
python -m brain_pixels --store artifacts/visual-memory add --op SET --subject "rescue scanner" --relation location --value "blue equipment locker" --config strata.local.json
python -m brain_pixels --store artifacts/visual-memory query "Where is the rescue scanner?"
```

Alternatively set `BRAIN_STRATA_CONFIG` to your local configuration path. The
adapter starts a hidden CPU encoder, exports one image, closes the encoder, and
then marks the record ready. Diagnostics stay off the CLI's JSON stdout. No
checkpoint or prompt history is shared between queries.

Policies: `verify`, `one-cell`, `two-cell`, `compact-64`. They are fixed when the
store is created. The text control permits verification only. The compact
policy uses the predeclared 64-bit checksum, not the unsafe 16-bit control.

A failed export after commit leaves the update pending. Missing or rejected
required records return UNKNOWN with empty evidence; a subsequent append does
not erase the blocked record. Runtime logs remain under the store's `runtime`
directory. Investigate the failed operation instead of deleting its update.

The journal is a local single-writer prototype, supports 255 active records per
generation, and uses cached projected features at recall. Clear is logical, not
physical erasure. Checksums establish consistency, not authenticity.
