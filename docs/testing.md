# Tests and cleanup

The active suite protects observable behavior rather than every historical
prototype generation. The original graph tests were relocated to `core/tests/`
and remain active alongside current routing, relation-memory, PNG and spatial
memory regressions. They were not discarded merely because the pixel project
has evolved.

The current pixel suite covers:

- Real frozen native exports: correct, two-cell repaired, and three-cell rejected.
- Bounded search sizes and valid/invalid integrity envelopes.
- The preserved 16-bit false-acceptance case.
- Saved policy reload, pending updates and whole-journal unavailability.
- Alias binding, uncertainty, explicit forgetting and independent slots.
- Source-image exclusion during recall, packet/metadata tampering and input limits.
- Separate stores, logical reset and encoder shutdown.
- Clean JSON stdout, failed-export behavior and directories containing spaces.

Removed from the active path: compatibility checks against superseded prototype
modules, repeated model-sweep generators with workstation-specific paths, and
one-off experiment launchers. Meaningful behavior from those generations is
consolidated into the portable current tests. Raw local experiments and their
failed outcomes remain untouched outside this clean checkout. A
[machine-readable inventory](../research/cleanup-inventory.json) records the
retained suites and relocated historical drivers.

Historical benchmark scripts and examples are under `research/legacy/`; they
are not counted as unit tests or run by CI. The one benchmark-named scenario
module still imported by the core remains in `core/` to avoid breaking its
runtime behavior. This dependency is future refactoring work.

Run all active suites:

```sh
python -m pip install -e ".[vision,dev]"
python tools/run_checks.py
python tools/verify_evidence.py
```

Formatting/linting applies to the current package, portable tests and tools;
the historical core is not subjected to an unrelated wholesale style rewrite.
CI runs on Windows/Linux with Python 3.11/3.12. Real feature fixtures do not
require an installed native encoder, and no live model or download is invoked
by the tests. The native process-boundary test uses a clearly identified fake
Python child; the separate learned-readout tests use genuine recorded exports.
