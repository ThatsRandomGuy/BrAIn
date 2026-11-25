"""Atomic local commits and explicit single-writer locking."""

import os
import tempfile
from pathlib import Path
from contextlib import contextmanager


def atomic(path, data):
    path = Path(path)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temp = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


@contextmanager
def writer(folder):
    path = Path(folder) / "writer.lock"
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            "Store already locked;inspect before recovering an abandoned lock"
        ) from exc
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink()
