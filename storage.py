"""Bit-exact safetensors archive with independent block coders and byte-counted parents."""
from contextlib import ExitStack
from compression import zstd
import hashlib
import json
from math import prod
from numbers import Integral
import os
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np

import codec

BLOCK_SIZE = 262144
MAGIC = b"CKPTAR01"
DIRECTORY = struct.Struct("<8sBBIQ")
U32 = struct.Struct("<I")
U64 = struct.Struct("<Q")


def _read(stream, n):
    data = stream.read(n)
    if len(data) != n:
        raise ValueError("truncated archive or tensor header")
    return data


def _payload_header(path):
    with path.open("rb") as stream:
        prefix = _read(stream, 8)
        text = _read(stream, U64.unpack(prefix)[0])
    tensors = json.loads(text)
    tensors.pop("__metadata__", None)
    ordered = sorted(((tuple(v["data_offsets"]), k, v["dtype"], tuple(v["shape"]))
                      for k, v in tensors.items()), key=lambda item: (item[0], item[1]))
    types = {t[2] for t in ordered}
    if len(types) != 1 or not types.issubset({"F16", "F32"}):
        raise ValueError(f"expected homogeneous FP16 or FP32 tensors in {path}")
    dtype = np.dtype({"F16": "<f2", "F32": "<f4"}[types.pop()])
    previous = 0
    for (lo, hi), name, _, shape in ordered:
        if lo != previous or any(d < 0 for d in shape) or hi - lo != prod(shape) * dtype.itemsize:
            raise ValueError(f"invalid tensor layout in {path}: {name}")
        previous = hi
    header = prefix + text
    if len(header) + previous != path.stat().st_size:
        raise ValueError(f"file size disagrees with tensor layout in {path}")
    return header, ordered, dtype


class Corpus:
    def __init__(self, root, model, start):
        self.steps = list(range(start, start + 17000, 1000))
        self.paths = [Path(root) / model / f"{step}.safetensors" for step in self.steps]
        self.arrays, self.headers, self.hashes = [], [], []
        baseline = None
        for path in self.paths:
            header, layout, self.dtype = _payload_header(path)
            if baseline is None:
                baseline = layout
            if baseline != layout:
                raise ValueError(f"tensor names, shapes, dtype or offsets changed in {path}")
            self.arrays.append(np.memmap(path, mode="r", offset=len(header), dtype=self.dtype))
            self.headers.append(zstd.compress(header, level=3))
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            sidecar = path.with_suffix(".json")
            if sidecar.is_file() and digest != json.loads(sidecar.read_text())["local_sha256"]:
                raise ValueError(f"input hash differs from pinned manifest in {path}")
            self.hashes.append(digest)
        self.nelems = len(self.arrays[0])


def predict(left, right):
    """Round the midpoint once to the payload's native IEEE format."""
    if left.dtype != right.dtype or left.shape != right.shape:
        raise ValueError("predictors must have matching dtype and shape")
    with np.errstate(invalid="ignore"):
        return ((left.astype(np.float64) + right.astype(np.float64)) * .5).astype(left.dtype)


def _parents(value):
    if value is None:
        return ()
    if isinstance(value, Integral):
        return () if value < 0 else (int(value),)
    return tuple(sorted(int(parent) for parent in value))


def _record_header(stream):
    count = _read(stream, 1)[0]
    if count > 2:
        raise ValueError("bad number of parents")
    parents = tuple(_read(stream, count))
    header_len, = U32.unpack(_read(stream, U32.size))
    return parents, _read(stream, header_len)


def _next_block(stream):
    count, = U32.unpack(_read(stream, U32.size))
    if count < 1:
        raise ValueError("empty coded block")
    return _read(stream, count)


class RecordStore:
    def __init__(self, corpus, workdir):
        self.corpus = corpus
        # Content-key the records so another dataset or a modified codec cannot reuse them.
        key = hashlib.sha256("".join(corpus.hashes).encode())
        for filename in ("storage.py", "codec.py", "context_codec.cc"):
            key.update(Path(__file__).with_name(filename).read_bytes())
        self.workdir = Path(workdir) / key.hexdigest()[:24]
        self.workdir.mkdir(parents=True, exist_ok=True)

    def record(self, child, parents, interpolate=True):
        parents = _parents(parents)
        if child < 0 or child >= len(self.corpus.steps) or len(parents) > 2 or child in parents:
            raise ValueError("invalid record topology")
        if any(p < 0 or p >= len(self.corpus.steps) for p in parents) or len(set(parents)) != len(parents):
            raise ValueError("invalid parents")
        if len(parents) == 2 and interpolate and parents[0] + parents[1] != 2 * child:
            raise ValueError("two-parent interpolation requires a centered bridge")
        suffix = "_".join(map(str, parents)) or "anchor"
        filename = f"{child}_{suffix}{'_mean' if len(parents) == 2 and interpolate else ''}.rec"
        path = self.workdir / filename
        if path.is_file():
            return path

        with ExitStack() as stack:
            singles = []
            if len(parents) == 2:
                for parent in parents:
                    stream = stack.enter_context(self.record(child, (parent,)).open("rb"))
                    _record_header(stream)
                    singles.append(stream)
            with tempfile.NamedTemporaryFile(dir=self.workdir, delete=False) as temp:
                tmpname = Path(temp.name)
                stack.callback(tmpname.unlink, missing_ok=True)
                header = self.corpus.headers[child]
                temp.write(bytes((len(parents), *parents)) + U32.pack(len(header)) + header)
                target = self.corpus.arrays[child]
                for pos in range(0, len(target), BLOCK_SIZE):
                    section = slice(pos, pos + BLOCK_SIZE)
                    if len(parents) < 2:
                        ref = self.corpus.arrays[parents[0]][section] if parents else None
                        block = codec.encode(target[section], ref)
                    else:
                        options = [_next_block(stream) for stream in singles]
                        if interpolate:
                            mid = predict(*(self.corpus.arrays[p][section] for p in parents))
                            options.append(codec.encode(target[section], mid))
                        mode, coded = min(enumerate(options), key=lambda pair: len(pair[1]))
                        block = bytes((mode,)) + coded
                    temp.write(U32.pack(len(block)))
                    temp.write(block)
                if any(stream.read(1) for stream in singles):
                    raise ValueError("single-parent record has trailing blocks")
            os.replace(tmpname, path)
        return path


def _write_archive(corpus, store, plan, path, interpolate):
    records = [store.record(i, parents, interpolate) for i, parents in enumerate(plan)]
    n = len(records)
    offsets = [DIRECTORY.size + U64.size * (n + 1)]
    for record in records:
        offsets.append(offsets[-1] + record.stat().st_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        tmpname = Path(temp.name)
    try:
        with tmpname.open("wb") as out:
            out.write(DIRECTORY.pack(MAGIC, n, corpus.dtype.itemsize, BLOCK_SIZE, corpus.nelems))
            out.write(struct.pack(f"<{n+1}Q", *offsets))
            for record in records:
                with record.open("rb") as reader:
                    shutil.copyfileobj(reader, out, 1 << 20)
        os.replace(tmpname, path)
    finally:
        tmpname.unlink(missing_ok=True)


def _dependencies(parents):
    order, done, active = [], {}, set()

    def visit(i):
        if i in done:
            return done[i]
        if i in active:
            raise ValueError("archive dependency cycle")
        active.add(i)
        closure = {i}
        for p in parents[i]:
            if p >= len(parents) or p < 0:
                raise ValueError("parent index outside archive")
            closure.update(visit(p))
        active.remove(i)
        order.append(i)
        done[i] = closure
        return closure

    for i in range(len(parents)):
        visit(i)
    return order, [done[i] for i in range(len(parents))]


def verify_archive(corpus, store, plan, path, interpolate=True):
    """Materialize the actual file and reconstruct all original bytes with block-bounded RAM."""
    path = Path(path)
    plan = [_parents(plan[i]) for i in range(len(corpus.steps))]
    _dependencies(plan)
    _write_archive(corpus, store, plan, path, interpolate)
    with ExitStack() as stack:
        first = stack.enter_context(path.open("rb"))
        magic, count, width, stride, nelems = DIRECTORY.unpack(_read(first, DIRECTORY.size))
        if (magic, count, width, stride, nelems) != (MAGIC, len(plan), corpus.dtype.itemsize, BLOCK_SIZE, corpus.nelems):
            raise ValueError("bad archive directory")
        offsets = struct.unpack(f"<{count+1}Q", _read(first, U64.size * (count + 1)))
        directory_bytes = first.tell()
        if offsets[0] != directory_bytes or offsets[-1] != path.stat().st_size or any(
                hi <= lo for lo, hi in zip(offsets, offsets[1:])):
            raise ValueError("bad record offsets")
        streams, hashes, parents = [], [], []
        for i in range(count):
            stream = stack.enter_context(path.open("rb"))
            stream.seek(offsets[i])
            p, header = _record_header(stream)
            streams.append(stream)
            hashes.append(hashlib.sha256(zstd.decompress(header)))
            parents.append(p)
        if parents != plan:
            raise ValueError("stored topology differs from supplied plan")
        order, closures = _dependencies(parents)
        mode_counts = [0, 0, 0]
        for pos in range(0, nelems, stride):
            size = min(stride, nelems - pos)
            recovered = {}
            for i in order:
                block = _next_block(streams[i])
                p = parents[i]
                if len(p) == 2:
                    mode, block = block[0], block[1:]
                    if mode > 2 or (mode == 2 and not interpolate):
                        raise ValueError("bad prediction mode")
                    mode_counts[mode] += 1
                    ref = predict(recovered[p[0]], recovered[p[1]]) if mode == 2 else recovered[p[mode]]
                else:
                    ref = recovered[p[0]] if p else None
                recovered[i] = codec.decode(block, ref, corpus.dtype, size)
                hashes[i].update(memoryview(recovered[i]).cast("B"))
        if any(s.tell() != offsets[i+1] for i, s in enumerate(streams)):
            raise ValueError("record size disagrees with directory")
    for i, digest in enumerate(hashes):
        if digest.hexdigest() != corpus.hashes[i]:
            raise ValueError(f"reconstructed safetensors SHA-256 mismatch at {corpus.steps[i]}")
    sizes = [hi - lo for lo, hi in zip(offsets, offsets[1:])]
    cold = [directory_bytes + sum(sizes[i] for i in closure) for closure in closures]
    return dict(archive_bytes=offsets[-1], directory_bytes=directory_bytes,
                mean_cold_bytes=sum(cold)/count, max_cold_bytes=max(cold),
                max_records=max(map(len, closures)), per_record_bytes=sizes, cold_bytes=cold,
                cold_record_counts=list(map(len, closures)), sha256_verified=count,
                dual_mode_blocks=dict(left=mode_counts[0], right=mode_counts[1], midpoint=mode_counts[2]))
