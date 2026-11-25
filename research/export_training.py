"""Recreate explicitly supplied training exports with an installed native encoder.

Historical byte-identical replication requires the same encoder/projector build.
This creates new local files and never downloads models or changes the default.
"""

import argparse
import json
from pathlib import Path
from brain_pixels import integrity as g
from brain_pixels.encoder import NativeEncoder


def export(args):
    rows = json.loads(args.sources.read_text(encoding="utf-8-sig"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    with NativeEncoder(output / "runtime", args.config) as encoder:
        for index, row in enumerate(rows):
            # Generated fixed names avoid trusting source IDs as filesystem paths.
            folder = output / f"train-{index:04d}"
            folder.mkdir()
            image = folder / "memory.png"
            image.write_bytes(g.image_bytes(row["values"]))
            if g.sha(image.read_bytes()) != row["image_sha256"]:
                raise ValueError("source_image_hash_mismatch")
            _, metadata = encoder.encode(image, folder / "features.sve")
            if metadata["feature_sha256"] != row["feature_sha256"]:
                raise ValueError(
                    "feature_bytes_differ_from_reference;preserve this partial run and register a new compatible training/profile study"
                )
            records.append(
                {
                    **row,
                    "image": str(image.relative_to(output)),
                    "features": str((folder / "features.sve").relative_to(output)),
                }
            )
            if (index + 1) % 32 == 0:
                print(f"EXPORTED {index + 1}/{len(rows)}", flush=True)
    (output / "training-manifest.json").write_bytes(g.canonical({"version": 1, "records": records}))
    print(
        json.dumps(
            {
                "records": len(records),
                "manifest": str(output / "training-manifest.json"),
                "encoder_stopped": encoder.state["stopped"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    export(parser.parse_args())
