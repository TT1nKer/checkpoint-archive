"""Round-trip checks for every wire mode, including NaN payloads and signed zeros."""

import ctypes
import unittest
from compression import zstd

import numpy as np

import codec


class ExactCodecTest(unittest.TestCase):
    def check_modes(self, target, ref):
        word = target.dtype
        dtype = np.dtype(f"f{word.itemsize}")
        x, predictor = target.view(dtype), ref.view(dtype)
        delta = codec._rank(target) - codec._rank(ref)
        zig = (delta << 1) ^ (word.type(0) - (delta >> (8 * word.itemsize - 1)))
        library = codec._library()
        size = ctypes.c_size_t()
        ptr = library.pack_context(zig.ctypes.data, ref.ctypes.data, len(target),
                                   word.itemsize, ctypes.byref(size))
        self.assertTrue(ptr)
        try:
            context = b"C" + ctypes.string_at(ptr, size.value)
        finally:
            library.release_context(ptr)
        blobs = [codec.encode(x), codec.encode(x, predictor), b"R" + target.tobytes(),
                 b"Z" + zstd.compress(codec._shuffle(target)),
                 b"X" + zstd.compress(codec._shuffle(target ^ ref)),
                 b"D" + zstd.compress(codec._shuffle(zig)), context]
        for blob in blobs:
            with self.subTest(mode=blob[:1], width=word.itemsize, n=len(target)):
                recovered = codec.decode(blob, predictor, dtype, len(target))
                self.assertTrue(recovered.flags.c_contiguous)
                self.assertEqual(recovered.view(word).tobytes(), target.tobytes())

    def test_all_half_patterns(self):
        words = np.arange(1 << 16, dtype=np.uint16)
        self.check_modes(words, words[::-1].copy())

    def test_float32_edge_pairs(self):
        words = np.array([0, 0x80000000, 1, 0x80000001, 0x7f800000, 0xff800000,
                          0x7f800001, 0x7fc000ab, 0xffcaffee, 0xff7fffff], dtype=np.uint32)
        self.check_modes(np.tile(words, len(words)), np.repeat(words, len(words)))

    def test_empty_and_random(self):
        rng = np.random.default_rng(4879)
        for width in (2, 4):
            word = np.dtype(f"u{width}")
            for n in (0, 1, 9, 8193):
                pairs = rng.integers(0, 1 << (width * 8), (2, n), dtype=word)
                self.check_modes(*pairs)


if __name__ == "__main__":
    unittest.main()
