"""Inspect the exact numeric grids occupied by original weights and midpoint predictions."""
import argparse
import json
from pathlib import Path

import numpy as np

from storage import BLOCK_SIZE, Corpus, predict


def measure(blocks):
    totals = dict(elements=0, exact_fp16=0, exact_bf16_grid=0, nonfinite=0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for values in blocks:
            unsigned = np.dtype(f"u{values.itemsize}")
            # Compare original dtype bits after an FP16 cast and promotion to the original dtype.
            half_roundtrip = values.astype("<f2", copy=False).astype(values.dtype, copy=False)
            totals["exact_fp16"] += int(np.count_nonzero(half_roundtrip.view(unsigned) == values.view(unsigned)))
            # A finite F32 number is exactly BF16-representable iff its bottom 16 bits are zero.
            f32_words = values.astype("<f4", copy=False).view("<u4")
            totals["exact_bf16_grid"] += int(np.count_nonzero((f32_words & 0xffff) == 0))
            totals["nonfinite"] += int(np.count_nonzero(~np.isfinite(values)))
            totals["elements"] += values.size
    totals["exact_fp16_fraction"] = totals["exact_fp16"] / totals["elements"]
    totals["exact_bf16_fraction"] = totals["exact_bf16_grid"] / totals["elements"]
    return totals


def sections(corpus, midpoint):
    indexes = range(1, len(corpus.steps)-1) if midpoint else range(len(corpus.steps))
    for i in indexes:
        for start in range(0, corpus.nelems, BLOCK_SIZE):
            segment = slice(start, start + BLOCK_SIZE)
            if midpoint:
                yield predict(corpus.arrays[i-1][segment], corpus.arrays[i+1][segment])
            else:
                yield corpus.arrays[i][segment]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cohorts = []
    for model in ("14m", "70m"):
        for start in (32000, 112000):
            corpus = Corpus(args.root, model, start)
            rows = {}
            for midpoint, label in ((False, "original"), (True, "radius_one_midpoint")):
                rows[label] = measure(sections(corpus, midpoint))
                expected = corpus.nelems * (15 if midpoint else 17)
                assert rows[label]["elements"] == expected
            cohorts.append(dict(model=model, start=start, end=corpus.steps[-1], dtype=corpus.dtype.name,
                                elements_per_snapshot=corpus.nelems, **rows))
    result = dict(weighting="one count per scalar value at each evaluated time",
                  original_steps_per_cohort=17, interior_midpoint_steps_per_cohort=15,
                  cohort_scope={"14m": "complete safetensors payload", "70m": "13 pinned tensor slices"},
                  midpoint_rule="average the two immediate neighbors in float64, cast once to archive dtype",
                  fp16_rule="cast to float16 and back to archive dtype; compare original archive dtype bits",
                  bf16_rule="convert to float32 and test its low 16 bits for zero",
                  cohorts=cohorts)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
