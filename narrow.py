"""Post hoc sensitivity for source FP32 tensors that fall exactly on the FP16 grid."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cold import NARROW_MAGIC, restore_one
from experiment import optimize
from storage import Corpus, RecordStore, _write_archive


class ExactHalfCorpus:
    def __init__(self, source):
        if source.dtype != np.dtype("<f4"):
            raise ValueError("narrow archive requires a source of FP32 weights")
        self.arrays = []
        for step, original in zip(source.steps, source.arrays, strict=True):
            narrow = original.astype("<f2")
            if not np.array_equal(narrow.astype("<f4").view("<u4"), original.view("<u4")):
                raise ValueError(f"step {step} cannot be narrowed bit-exactly")
            self.arrays.append(narrow)
        self.headers = source.headers  # Includes the exact original safetensors dtype declaration.
        self.hashes = source.hashes
        self.steps = source.steps
        self.nelems = source.nelems
        self.dtype = np.dtype("<f2")


def costs(data, records):
    anchors, singles, bridges = [], [], []
    n = len(data.arrays)
    for i in range(n):
        anchors.append(records.record(i, ()).stat().st_size)
        singles.append([records.record(i, (p,)).stat().st_size if p != i else -1 for p in range(n)])
        for r in (1, 2, 4, 8):
            if i-r >= 0 and i+r < n:
                sizes = [records.record(i, (i-r,i+r), flag).stat().st_size for flag in (True,False)]
                bridges.append([i,i-r,i+r,*sizes])
    return dict(anchors=anchors, singles=singles, bridges=bridges)


def verify_source_bytes(path, source):
    """Open the physical narrow archive and perform one independent cold restore per target."""
    with path.open("rb") as arc:
        if arc.read(len(NARROW_MAGIC)) != NARROW_MAGIC:
            raise ValueError("invalid narrow archive marker")
    cold = []
    for i, expected in enumerate(source.hashes):
        restored = restore_one(path, i)
        digest = hashlib.sha256(restored.header)
        digest.update(memoryview(restored.payload).cast("B"))
        if digest.hexdigest() != expected:
            raise ValueError(f"original F32 file hash mismatch at {source.steps[i]}")
        cold.append(restored.read_bytes)
    return dict(archive_bytes=path.stat().st_size, sha256_verified=len(cold), cold_bytes=cold,
                cold_mean_bytes=sum(cold)/len(cold))


def run(root, start, binary, workdir, destination):
    source = Corpus(root, "70m", start)
    half = ExactHalfCorpus(source)
    records = RecordStore(half, workdir/"narrow-records")
    scores = costs(half, records)
    plans = optimize(scores,binary)
    result = dict(status="post hoc exact narrow representation sensitivity",
                  model="70m", scope="13 fixed tensor slices per step", window=source.steps,
                  original_dtype="float32", archive_payload_dtype="float16",
                  elements_exactly_narrowed=sum(map(len, half.arrays)),
                  midpoint_rule="float64 midpoint rounded once to archive payload dtype (FP16)",
                  archive_marker=NARROW_MAGIC.decode(), original_headers_preserved=True,
                  extra_metadata_bytes_vs_primary=0, budgets={}, physical_k5={})
    for cap, layouts in plans.items():
        summary = {}
        for kind in ("single", "dual", "endpoints_fixed", "collapse"):
            plan = layouts[kind]
            summary[kind] = {k:plan[k] for k in ("archive_bytes", "mean_cold_bytes", "max_records",
                                                   "dual_records", "plan")}
        summary["savings_fraction"] = 1 - summary["dual"]["archive_bytes"]/summary["single"]["archive_bytes"]
        summary["dual_lower_bound"] = layouts["dual_lower_bound"]
        result["budgets"][cap] = summary
    for kind in ("single", "dual"):
        ref = plans["5"][kind]
        path = workdir/f"narrow-70m-{start}-{kind}.arc"
        _write_archive(half,records,ref["plan"],path,True)
        with path.open("r+b") as arc:
            arc.write(NARROW_MAGIC)
        actual = verify_source_bytes(path,source)
        if actual["archive_bytes"] != ref["archive_bytes"] or actual["cold_bytes"] != ref["cold_bytes"]:
            raise ValueError("actual narrow archive disagrees with cost model")
        result["physical_k5"][kind] = actual
    destination.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(start,[(k, round(100*v["savings_fraction"],4)) for k,v in result["budgets"].items()], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("start",type=int,choices=(32000,112000))
    parser.add_argument("forest_binary",type=Path)
    parser.add_argument("scratch",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    run(args.root,args.start,args.forest_binary,args.scratch,args.output)
