"""Restore one native or exact-narrow safetensors file, reading only its parents."""
from compression import zstd
from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import struct

import numpy as np

import codec
from storage import DIRECTORY, MAGIC, U64, _next_block, _record_header, predict


NARROW_MAGIC = b"CKNARW01"


@dataclass
class Restored:
    header: bytes
    payload: np.ndarray
    read_bytes: int
    record_ids: tuple
    read_calls: int


def restore_one(path, target):
    """Start with an empty cache, fetch the directory and each reachable record once."""
    path = Path(path)
    with path.open("rb", buffering=0) as stream:
        fd = stream.fileno()
        read_bytes = read_calls = 0
        opened = []

        def fetch(offset, size):
            nonlocal read_bytes, read_calls
            result = os.pread(fd, size, offset)
            read_bytes += len(result)
            read_calls += 1
            if len(result) != size:
                raise ValueError("archive range read was short")
            return result

        magic, n, width, block, nelems = DIRECTORY.unpack(fetch(0, DIRECTORY.size))
        if not (magic in (MAGIC, NARROW_MAGIC) and width in (2, 4) and block > 0 and 0 <= target < n):
            raise ValueError("unknown archive representation or invalid index")
        if magic == NARROW_MAGIC and width != 2:
            raise ValueError("narrow archive payload must contain FP16 values")
        indexed = U64.size * (n + 1)
        offsets = struct.unpack(f"<{n+1}Q", fetch(DIRECTORY.size, indexed))
        if (offsets[0] != DIRECTORY.size + indexed or offsets[-1] != os.fstat(fd).st_size
                or any(b <= a for a, b in zip(offsets, offsets[1:]))):
            raise ValueError("invalid record offsets")
        dtype = np.dtype(f"<f{width}")
        memo, active = {}, set()

        def recover(i):
            if i in memo:
                return memo[i]
            if not 0 <= i < n or i in active:
                raise ValueError("invalid archive dependencies")
            active.add(i)
            opened.append(i)
            record = BytesIO(fetch(offsets[i], offsets[i+1]-offsets[i]))
            parents, header = _record_header(record)
            if len(set(parents)) != len(parents):
                raise ValueError("duplicate archive parent")
            previous = [recover(parent)[1] for parent in parents]
            result = np.empty(nelems, dtype=dtype)
            for pos in range(0, nelems, block):
                part = slice(pos, pos + block)
                length = min(block, nelems - pos)
                blob = _next_block(record)
                if len(parents) == 2:
                    mode, blob = blob[0], blob[1:]
                    if mode > 2:
                        raise ValueError("unknown prediction mode")
                    ref = predict(previous[0][part], previous[1][part]) if mode == 2 else previous[mode][part]
                else:
                    ref = previous[0][part] if previous else None
                result[part] = codec.decode(blob, ref, dtype, length)
            if record.read(1):
                raise ValueError("record contains trailing bytes")
            active.remove(i)
            memo[i] = zstd.decompress(header), result
            return memo[i]

        header, payload = recover(target)
        if magic == NARROW_MAGIC:
            payload = payload.astype("<f4")
        return Restored(header, payload, read_bytes, tuple(opened), read_calls)
