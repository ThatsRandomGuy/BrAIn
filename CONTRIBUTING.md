# Contributing

Install an editable checkout with `python -m pip install -e ".[vision,dev]"`.
Run `python tools/run_checks.py` and `python tools/verify_evidence.py`, then
`python -m ruff check brain_pixels tests tools` and
`python -m ruff format --check brain_pixels tests tools`.

Changes to memory behavior should keep evidence conservative: an unread update
cannot authorize the prior fact, and uncertainty must not become confirmation.
Add regressions for meaningful failures, not tests that simply repeat an
implementation's internal steps. Use temporary directories and synthetic facts.

For experiments, freeze source cases, split boundaries, decoder selection,
policies and success criteria before inference. Treat previous tests as seen
regressions. Never select a nonce, retry, model variant, checksum width or query
adapter based on favorable test output and call it fresh evidence. Report raw
accuracy, repair, wrong acceptance and unavailable coverage separately.

Count metadata, codebook, decoder and cache costs. Compare visual and strong
text/retrieval controls using the same facts and semantics. A smaller file does
not imply fewer model-input positions, and a software memory improvement does
not establish that the language model thinks visually.

Keep workstation paths, credentials, model weights, feature dumps and runtime
state out of commits. Small documented synthetic fixtures are intentional
exceptions. The current package lives in `brain_pixels/`; the older application
and historical benchmark drivers remain separate under `core/` and `research/`.
