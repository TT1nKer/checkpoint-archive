import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from cold import restore_one
from storage import BLOCK_SIZE, Corpus, RecordStore, verify_archive


class StorageTest(unittest.TestCase):
    def test_original_files_and_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mini").mkdir()
            words = np.arange(BLOCK_SIZE + 5, dtype="<u2")
            for i in range(17):
                header = {"weight": {"dtype": "F16", "shape": [len(words)],
                                      "data_offsets": [0, words.nbytes]},
                          "__metadata__": {"step": str(i)}}
                body = json.dumps(header, sort_keys=(i % 2 == 1)).encode()
                original = struct.pack("<Q", len(body)) + body + (words + i).tobytes()
                path = root / "mini" / f"{1000*i}.safetensors"
                path.write_bytes(original)
                path.with_suffix(".json").write_text(json.dumps(
                    {"local_sha256": hashlib.sha256(original).hexdigest()}))
            corpus = Corpus(root, "mini", 0)
            records = RecordStore(corpus, root / "cache")
            plan = [() if i % 2 == 0 else (i-1,i+1) for i in range(17)]
            for use_midpoint in (True, False):
                archive = root / "weights.arc"
                result = verify_archive(corpus, records, plan, archive, use_midpoint)
                self.assertEqual(result["sha256_verified"], 17)
                self.assertEqual(result["max_records"], 3)
                self.assertEqual(result["archive_bytes"], archive.stat().st_size)
                sizes = result["per_record_bytes"]
                directory = 22 + 8 * 18
                self.assertEqual(directory + sum(sizes), archive.stat().st_size)
                self.assertEqual(result["cold_bytes"][1], directory + sum(sizes[:3]))
                restored = restore_one(archive, 1)
                self.assertEqual(restored.read_bytes, result["cold_bytes"][1])
                self.assertEqual(set(restored.record_ids), {0, 1, 2})
                self.assertEqual(restored.read_calls, 5)
                digest = hashlib.sha256(restored.header)
                digest.update(restored.payload.tobytes())
                self.assertEqual(digest.hexdigest(), corpus.hashes[1])
                self.assertEqual(sum(result["dual_mode_blocks"].values()), 8 * 2)
                if not use_midpoint:
                    self.assertEqual(result["dual_mode_blocks"]["midpoint"], 0)


if __name__ == "__main__":
    unittest.main()
