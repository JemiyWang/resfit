from __future__ import annotations

import argparse
import io
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--camera", default="image")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    table = pq.read_table(args.parquet, columns=[args.camera])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        args.output,
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=9,
    ) as writer:
        for encoded in table[args.camera].to_pylist():
            with Image.open(io.BytesIO(encoded["bytes"])) as image:
                writer.append_data(np.asarray(image.convert("RGB")))

    print(f"{args.output}: {len(table)} frames at {args.fps} FPS")


if __name__ == "__main__":
    main()
