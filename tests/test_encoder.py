"""Model-free process-boundary regressions for the optional native adapter."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from brain_pixels.encoder import NativeEncoder
from brain_pixels import integrity


class EncoderProtocolTests(unittest.TestCase):
    def test_paths_with_spaces_and_shutdown(self):
        with tempfile.TemporaryDirectory(prefix="brain encoder ") as temporary:
            root = Path(temporary)
            dummy = root / "dummy.bin"
            dummy.write_bytes(b"placeholder")
            config = root / "encoder.json"
            config.write_text(
                json.dumps({"exe": str(dummy), "mmproj": str(dummy), "model": str(dummy)})
            )
            child = root / "fake encoder.py"
            child.write_text(
                "import sys,struct\nfrom pathlib import Path\nprint('READY 2560',flush=True)\nfor line in sys.stdin:\n if line.strip()=='QUIT':break\n image,out=line.strip()[4:].rsplit(' ',1)\n assert Path(image).is_file()\n Path(out).write_bytes(struct.pack('<5i',0x31455653,49,7,7,2560)+bytes(49*2560*4))\n print('OK 49 7 7 1',flush=True)\n"
            )
            image = root / "an image with spaces.png"
            image.write_bytes(integrity.image_bytes([0] * 16))
            output = root / "an output with spaces.sve"
            original = subprocess.Popen

            def launch(args, **kwargs):
                self.assertIn("--max-tokens", args)
                self.assertNotIn("--gpu", args)
                return original([sys.executable, "-u", str(child)], **kwargs)

            with patch("brain_pixels.encoder.subprocess.Popen", side_effect=launch):
                with NativeEncoder(root / "runtime with spaces", config) as encoder:
                    features, _ = encoder.encode(image, output)
                    self.assertEqual(features.shape, (49, 2560))
            self.assertTrue(encoder.state["stopped"])
            self.assertEqual(encoder.state["exit_code"], 0)
            self.assertTrue(output.is_file())

    def test_missing_config_does_not_commit_pending_record(self):
        from brain_pixels import cli, journal

        with tempfile.TemporaryDirectory() as temporary:
            store = journal.GuardedMemoryStore.create(Path(temporary) / "store", policy="two-cell")
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaises(ValueError):
                    cli.add(store, "SET", "rescue scanner", "location", "blue equipment locker")
            self.assertEqual(store.inspect()["active_records"], 0)


if __name__ == "__main__":
    unittest.main()
