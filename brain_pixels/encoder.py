"""Explicitly configured CPU Strata adapter. No downloads or global services."""

import json
import os
import queue
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from . import integrity


class NativeEncoder:
    def __init__(self, folder, config):
        self.folder = Path(folder)
        self.config_path = Path(config).resolve()
        self.config_raw = self.config_path.read_bytes()
        document = json.loads(self.config_raw)
        self.config = document.get("vision", document)
        self.process = None
        self.stderr = None
        self.lines = queue.Queue()
        self.state = {"started": False, "stopped": True}
        for key in ("exe", "mmproj", "model"):
            path = Path(self.config[key]).expanduser()
            if not path.is_absolute():
                path = self.config_path.parent / path
            if not path.is_file():
                raise ValueError("Encoder configuration path not found:" + key)
            self.config[key] = str(path.resolve())

    def save_state(self):
        (self.folder / "lifecycle.json").write_text(
            json.dumps(self.state, indent=2), encoding="utf-8"
        )

    def __enter__(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stderr = (self.folder / "vision.stderr.log").open("w", encoding="utf-8")
        args = [
            self.config["exe"],
            "--mmproj",
            self.config["mmproj"],
            "--model",
            self.config["model"],
            "--threads",
            "8",
            "--max-tokens",
            "300",
        ]
        try:
            self.process = subprocess.Popen(
                args,
                cwd=self.folder,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr,
                text=True,
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.state = {"started": True, "stopped": False, "pid": self.process.pid}
            self.save_state()

            def reader():
                for line in self.process.stdout:
                    self.lines.put(line.strip())
                self.lines.put(None)

            self.reader = threading.Thread(target=reader, daemon=True)
            self.reader.start()
            self.receive("READY")
            return self
        except BaseException:
            self.close()
            raise

    def receive(self, prefix):
        while True:
            try:
                line = self.lines.get(timeout=180)
            except queue.Empty as exc:
                raise TimeoutError("Vision encoder response timed out") from exc
            if line is None or line.startswith("ERR"):
                raise RuntimeError(str(line))
            if line.startswith(prefix):
                return line

    def encode(self, image, out):
        image = Path(image).resolve()
        out = Path(out)
        relative = "encoded-" + uuid.uuid4().hex + ".sve"
        if "\n" in str(image) or "\r" in str(image):
            raise ValueError("Unsupported newline in image path")
        # The native protocol splits at its LAST space. A relative output name
        # supports both image and workspace directories containing spaces.
        self.process.stdin.write("ENC " + str(image) + " " + relative + "\n")
        self.process.stdin.flush()
        response = self.receive("OK")
        temporary = self.folder / relative
        features, shape = integrity.read_features(temporary)
        shutil.copyfile(temporary, out)
        temporary.unlink()
        return features, {
            "response": response,
            "shape": shape,
            "feature_sha256": integrity.sha(out.read_bytes()),
        }

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write("QUIT\n")
                    self.process.stdin.flush()
                    self.process.wait(timeout=15)
                except (OSError, subprocess.TimeoutExpired):
                    self.process.kill()
                    self.process.wait(timeout=10)
            self.state.update(
                stopped=self.process.poll() is not None, exit_code=self.process.returncode
            )
        if self.process is not None:
            if hasattr(self, "reader"):
                self.reader.join(timeout=2)
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
        self.state["config_unchanged"] = self.config_path.read_bytes() == self.config_raw
        self.save_state()
        if self.stderr is not None:
            self.stderr.close()

    def __exit__(self, *args):
        self.close()


def encoder_factory(config=None):
    path = config or os.environ.get("BRAIN_STRATA_CONFIG")
    if not path:
        raise ValueError(
            "Visual add requires --config or BRAIN_STRATA_CONFIG;see docs/quickstart.md"
        )
    path = Path(path).resolve()
    # Validate configuration before an add commits pending data.
    NativeEncoder(Path("."), path)
    return lambda folder: NativeEncoder(folder, path)
