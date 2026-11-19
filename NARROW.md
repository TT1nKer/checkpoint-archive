# Exact FP16 storage of the 70m samples

This post hoc sensitivity uses the original 13 tensor slices at each of 17 steps. All 7,241,728 sampled FP32 words in each window reproduced their original bits after conversion to FP16 and back. Each archive retains the complete original slice-file header. A fixed eight-byte archive marker instructs the reader to restore the payload to FP32 before emitting the file. The marker occupies the same space as the primary archive marker. Predictor arithmetic uses a float64 midpoint rounded once to the stored FP16 grid.

| Window | Cold record limit | Best single-parent bytes | Found mixed-parent bytes | Saved | Mean cold bytes, single | Mean cold bytes, mixed |
|---|---:|---:|---:|---:|---:|---:|
| 32k–48k | 3 | 10,058,983 | 10,044,582 | 0.14% | 1,675,416 | 1,678,111 |
| 32k–48k | 5 | 9,970,773 | 9,854,461 | 1.17% | 2,110,062 | 2,464,353 |
| 32k–48k | 8 | 9,919,847 | 9,852,851 | 0.68% | 2,934,034 | 3,023,217 |
| 112k–128k | 3 | 8,946,415 | 8,942,575 | 0.04% | 1,539,469 | 1,543,823 |
| 112k–128k | 5 | 8,836,524 | 8,731,512 | 1.19% | 2,035,343 | 2,179,720 |
| 112k–128k | 8 | 8,787,931 | 8,726,474 | 0.70% | 2,774,503 | 2,764,621 |

At the primary limit of five records, narrowing the single-parent archive alone saved 16.36% and 18.42% against the corresponding original-width single-parent archives. Adding a second parent yielded another 1.17% and 1.19%, with higher mean cold-read bytes in both windows. The single-parent relaxation bounds the best possible savings from these mixed-parent candidates at 2.27% and 2.41% for the two windows. The extra experiment changes both the representation of the residual and the rounding grid of the prediction, so it cannot isolate the contribution of either.

The physical single- and mixed-parent archives at the five-record limit each restored all 17 original FP32 slice files in a fresh dependency traversal for every requested step. Full-file SHA-256 digests and cold bytes agreed with source and cost records. The experiment concerns these sampled FP32 values in Pythia-70m. The scope of the data and the detected representation are documented in [the precision counts](results/precision.json).

Run from the project directory with the downloaded corpus, Python 3.14, NumPy and a C++17 compiler.

```sh
c++ -O3 -std=c++17 forest.cpp -o /tmp/checkpoint_forest
python3 narrow.py /tmp/checkpoint-round 32000 /tmp/checkpoint_forest scratch/narrow results/narrow-70m-32000.json
python3 narrow.py /tmp/checkpoint-round 112000 /tmp/checkpoint_forest scratch/narrow results/narrow-70m-112000.json
```
