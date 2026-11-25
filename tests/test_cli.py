import struct
import tempfile
import unittest
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch
import numpy as np
from brain_pixels import cli as p


class PixelJournalCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_text_add_ready_and_isolated_store(self):
        a = p.m.GuardedMemoryStore.create(self.root / "a", "text")
        b = p.m.GuardedMemoryStore.create(self.root / "b", "text")
        p.add(a, "SET", "rescue scanner", "location", "blue equipment locker", 23)
        self.assertEqual(
            a.recall("Where is the rescue scanner?")["answer"], "blue equipment locker"
        )
        self.assertEqual(b.recall("Where is the rescue scanner?")["answer"], "UNKNOWN")

    def test_encoder_closed_before_readiness(self):
        events = []

        class Fake:
            def __init__(self, path):
                pass

            def __enter__(self):
                events.append("enter")
                return self

            def __exit__(self, *args):
                events.append("exit")

            def encode(self, image, out):
                data = np.zeros((49, 2560), dtype="<f4")
                out.write_bytes(struct.pack("<5i", 0x31455653, 49, 7, 7, 2560) + data.tobytes())
                return data, {}

        store = p.m.GuardedMemoryStore.create(self.root / "visual", policy="compact-64")
        original = store.mark_ready

        def ready(sequence):
            self.assertEqual(events, ["enter", "exit"])
            return original(sequence)

        with patch.object(store, "mark_ready", side_effect=ready):
            result = p.add(
                store, "SET", "rescue scanner", "location", "blue equipment locker", 23, Fake
            )
        self.assertEqual(result["status"], "ready_for_verified_recall")

    def test_failed_export_leaves_pending_update(self):
        class Failure:
            def __init__(self, path):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def encode(self, *args):
                raise RuntimeError("synthetic encoder failure")

        store = p.m.GuardedMemoryStore.create(self.root / "visual", policy="two-cell")
        with self.assertRaises(RuntimeError):
            p.add(store, "SET", "rescue scanner", "location", "blue equipment locker", 23, Failure)
        result = store.recall("Where is the rescue scanner?")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["context"]["facts"], [])
        self.assertEqual(result["context"]["blocked_ids"], [1])

    def test_clear_retains_images_and_removes_active_context(self):
        store = p.m.GuardedMemoryStore.create(self.root / "text", "text")
        p.add(store, "SET", "rescue scanner", "location", "blue equipment locker", 23)
        store.clear()
        self.assertEqual(store.recall("Where is the rescue scanner?")["answer"], "UNKNOWN")
        self.assertTrue(list((store.folder / "blocks").iterdir()))

    def test_encoder_protocol_does_not_pollute_stdout(self):
        class Fake:
            def __init__(self, path):
                pass

            def __enter__(self):
                print("READY 2560")
                return self

            def __exit__(self, *args):
                pass

            def encode(self, image, out):
                data = np.zeros((49, 2560), dtype="<f4")
                out.write_bytes(struct.pack("<5i", 0x31455653, 49, 7, 7, 2560) + data.tobytes())
                return data, {}

        store = p.m.GuardedMemoryStore.create(self.root / "protocol", policy="two-cell")
        output = io.StringIO()
        diagnostics = io.StringIO()
        with redirect_stdout(output), redirect_stderr(diagnostics):
            p.add(store, "SET", "rescue scanner", "location", "blue equipment locker", 23, Fake)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("READY 2560", diagnostics.getvalue())


if __name__ == "__main__":
    unittest.main()
