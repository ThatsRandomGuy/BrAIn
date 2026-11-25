"""Run the portable pixel suite and the maintained historical core regressions."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    suites = [("pixels", ROOT, ROOT / "tests"), ("core", ROOT / "core", ROOT / "core/tests")]
    for label, cwd, tests in suites:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(ROOT / "core")
        result = subprocess.run(
            [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(tests)],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            return result.returncode
        print(
            label
            + ": "
            + result.stderr.strip()
            .split("----------------------------------------------------------------------")[-1]
            .strip(),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
