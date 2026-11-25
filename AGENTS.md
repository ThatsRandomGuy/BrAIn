# BrAIn development

- Work on a feature branch and open a pull request. Do not push changes directly
  to main or merge a PR unless the user explicitly asks.
- Current pixel code is in brain_pixels/. Historical core and benchmark drivers
  are separate. Keep the model-free demo free of NumPy and native-model needs.
- Run python tools/run_checks.py, python tools/verify_evidence.py, and the scoped
  Ruff checks described in CONTRIBUTING.md. Experimental research modules need
  their own lint/format checks as well.
- Freeze experiment data, selection rules, source fingerprints and gates before
  inference. Keep validation and test separate. Never promote a different model
  after viewing held-out test results.
- Report raw accuracy, recovery, false accepts, unavailable coverage and actual
  representation/decoder costs. Preserve negative results.
- Keep the installed encoder explicitly configured; never download weights or
  start a language server as a side effect of tests. Close any encoder started
  by an evaluation when the run finishes.
- Do not include workstation paths, private config, large feature caches or
  native model weights in a PR. Compact synthetic evidence is intentional.
