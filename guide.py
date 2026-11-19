"""Measure candidate record sizes on fixed, position-stratified excerpts."""
import argparse
from compression import zstd
import json
from pathlib import Path
import struct

import numpy as np

from codec import encode


def summarize(root, model, start):
    frames, headers = [], []
    for step in range(start, start + 17000, 1000):
        path = root / model / f"{step}.safetensors"
        with path.open("rb") as stream:
            length = stream.read(8)
            header = stream.read(struct.unpack("<Q", length)[0])
        layout = json.loads(header)
        kind, = {info["dtype"] for name, info in layout.items() if name != "__metadata__"}
        data = np.memmap(path, dtype={"F16":"<f2", "F32":"<f4"}[kind], mode="r",
                         offset=8 + len(header))
        positions = [len(data) * (2 * i + 1) // 16 - 2048 for i in range(8)]
        frames.append([np.array(data[pos:pos+4096]) for pos in positions])
        headers.append(5 + len(zstd.compress(length + header, level=3)))
    n = len(frames)
    anchors = [headers[i] + sum(4 + len(encode(x)) for x in frames[i]) for i in range(n)]
    blocks, singles, bridges = {}, [], []
    for i, target in enumerate(frames):
        row = []
        for p in range(n):
            if p == i:
                row.append(-1)
                continue
            sizes = [len(encode(x, y)) for x, y in zip(target, frames[p])]
            blocks[i,p] = sizes
            row.append(headers[i] + 1 + 4 * len(sizes) + sum(sizes))
        singles.append(row)
    for radius in (1, 2, 4, 8):
        for i in range(radius, n - radius):
            left, right = i - radius, i + radius
            sizes = []
            for j, x in enumerate(frames[i]):
                pred = ((frames[left][j].astype('f8') + frames[right][j].astype('f8')) * .5).astype(x.dtype)
                sizes.append(min(blocks[i,left][j], blocks[i,right][j], len(encode(x, pred))))
            overhead = headers[i] + 2 + 5 * len(sizes)
            endpoint_sizes = [min(a,b) for a,b in zip(blocks[i,left], blocks[i,right])]
            bridges.append([i, left, right, overhead + sum(sizes), overhead + sum(endpoint_sizes)])
    return dict(model=model, start=start, anchors=anchors, singles=singles, bridges=bridges)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("model")
    parser.add_argument("start", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.root, args.model, args.start)
    args.output.write_text(json.dumps(result, separators=(",",":")) + "\n")
