"""Validate the built wheel in a temporary environment without NumPy or a checkout."""

import argparse
import json
import os
import subprocess
import tempfile
import venv
import zipfile
from pathlib import Path


def main(wheel):
    wheel = wheel.resolve()
    temporary = tempfile.TemporaryDirectory(prefix="brain-wheel-")
    root = Path(temporary.name).resolve()
    base = Path(tempfile.gettempdir()).resolve()
    assert root.parent == base and root.name.startswith("brain-wheel-")
    try:
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
            check=True,
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            [str(python), "-I", "-m", "brain_pixels", "demo", "--output", str(root / "demo")],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        assert payload["answers"] == ["blue equipment locker", "UNKNOWN", "padded transport crate"]
        assert payload["recovery"]["distance"] == 2
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import importlib.util;assert importlib.util.find_spec('numpy') is None",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        with zipfile.ZipFile(wheel) as bundle:
            assert (
                "brain_pixels/assets/decoder.npz" in bundle.namelist()
                and "brain_pixels/assets/profile.json" in bundle.namelist()
            )
        print(
            json.dumps(
                {
                    "passed": True,
                    "installed_wheel": wheel.name,
                    "numpy_installed": False,
                    "checkout_required": False,
                    "demo_answers": payload["answers"],
                }
            )
        )
    finally:
        assert root.parent == base and root.name.startswith("brain-wheel-")
        temporary.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    main(args.wheel)
