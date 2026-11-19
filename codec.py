"""Bit-exact, block-local codecs. The decoder receives dtype and element count externally.

Native little-endian NumPy float16/float32 inputs. Every payload includes its choice
byte. The compiled helper is cached in a user-private temporary directory.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from functools import lru_cache
from compression import zstd

import numpy as np


@lru_cache(maxsize=1)
def _library():
    source = Path(__file__).with_name("context_codec.cc")
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    cache = Path(tempfile.gettempdir()) / f"research-ckpt-codec-{os.getuid()}"
    cache.mkdir(mode=0o700, exist_ok=True)
    if cache.lstat().st_uid != os.getuid() or cache.lstat().st_mode & 0o077 or cache.is_symlink():
        raise OSError(f"Unsafe build cache: {cache}")
    target = cache / f"context-{fingerprint}.so"
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".so") as temp:
            subprocess.run(["g++", "-std=c++17", "-O3", "-fPIC", "-shared",
                            str(source), "-o", temp.name], check=True)
            # Link only a complete file; concurrent builders may already have installed it.
            try:
                os.link(temp.name, target)
            except FileExistsError:
                pass
    lib = ctypes.CDLL(str(target))
    ptr = ctypes.c_void_p
    count = ctypes.c_size_t
    lib.pack_context.argtypes = [ptr, ptr, count, ctypes.c_int, ctypes.POINTER(count)]
    lib.pack_context.restype = ptr
    lib.release_context.argtypes = [ptr]
    lib.release_context.restype = None
    lib.unpack_context.argtypes = [ptr, count, ptr, ptr, count, ctypes.c_int]
    lib.unpack_context.restype = ctypes.c_int
    return lib


def _dtype(dtype):
    dtype = np.dtype(dtype)
    if sys.byteorder != "little" or dtype not in (np.dtype("<f2"), np.dtype("<f4")):
        raise TypeError("native little-endian float16/float32 required")
    return dtype


def _words(x, dtype=None):
    x = np.asarray(x)
    if dtype is not None and x.dtype != dtype:
        raise ValueError("prediction dtype differs from target")
    dtype = _dtype(x.dtype)
    return np.ascontiguousarray(x).reshape(-1).view(f"u{dtype.itemsize}")


def _shuffle(x):
    return x.view(np.uint8).reshape(-1, x.itemsize).T.copy().tobytes()


def _unshuffle(buf, size, word_type):
    width = word_type.itemsize
    if len(buf) != size * width:
        raise ValueError("payload size disagrees with shape")
    return np.frombuffer(buf, dtype=np.uint8).reshape(width, size).T.copy().ravel().view(word_type)


def _rank(x):
    sign = 1 << (x.itemsize * 8 - 1)
    return x ^ np.where((x & sign) != 0, x.dtype.type(np.iinfo(x.dtype).max), x.dtype.type(sign))


def _unrank(rank):
    sign = 1 << (rank.itemsize * 8 - 1)
    return rank ^ np.where((rank & sign) != 0, rank.dtype.type(sign), rank.dtype.type(np.iinfo(rank.dtype).max))


def encode(x, ref=None):
    """Return the smallest actual candidate blob, including its one-byte tag."""
    words = _words(x)
    raw = words.tobytes()
    if ref is None:
        return min((b"R" + raw, b"Z" + zstd.compress(_shuffle(words), level=3)), key=len)
    reference = _words(ref, np.asarray(x).dtype)
    if len(reference) != len(words):
        raise ValueError("prediction count differs from target")
    diff = _rank(words) - _rank(reference)
    zig = (diff << 1) ^ (words.dtype.type(0) - (diff >> (8 * words.itemsize - 1)))
    candidates = [b"R" + raw, b"X" + zstd.compress(_shuffle(words ^ reference), level=3),
                  b"D" + zstd.compress(_shuffle(zig), level=3)]
    lib = _library()
    count = ctypes.c_size_t()
    ptr = lib.pack_context(zig.ctypes.data, reference.ctypes.data, len(words), words.itemsize,
                           ctypes.byref(count))
    if not ptr:
        raise RuntimeError("context encoder failed")
    try:
        candidates.append(b"C" + ctypes.string_at(ptr, count.value))
    finally:
        lib.release_context(ptr)
    return min(candidates, key=len)


def decode(blob, ref, dtype, size):
    """Decode to a contiguous one-dimensional array. Size counts elements."""
    dtype = _dtype(dtype)
    kind, data = blob[:1], blob[1:]
    if size < 0:
        raise ValueError("negative element count")
    word_type = np.dtype(f"u{dtype.itemsize}")
    if kind == b"R":
        if len(data) != size * dtype.itemsize:
            raise ValueError("bad raw size")
        return np.frombuffer(data, dtype=dtype).copy()
    if kind == b"Z":
        return _unshuffle(zstd.decompress(data), size, word_type).view(dtype)
    if kind not in (b"X", b"D", b"C") or ref is None:
        raise ValueError("invalid mode or missing reference")
    reference = _words(ref, dtype)
    if len(reference) != size:
        raise ValueError("prediction count differs from target")
    if kind == b"C":
        zig = np.empty(size, dtype=word_type)
        ok = _library().unpack_context(data, len(data), reference.ctypes.data, zig.ctypes.data,
                                       size, dtype.itemsize)
        if not ok:
            raise ValueError("bad context-coded payload")
    else:
        zig = _unshuffle(zstd.decompress(data), size, word_type)
    if kind == b"X":
        return (zig ^ reference).view(dtype)
    delta = (zig >> 1) ^ (word_type.type(0) - (zig & 1))
    return _unrank(_rank(reference) + delta).view(dtype)
