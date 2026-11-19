# Reproducing the weight corpus

The pilot uses 34 published revisions of each EleutherAI Pythia model. Both windows contain 17 revisions, spaced 1,000 training steps apart. The first spans steps 32,000 through 48,000 and the second spans 112,000 through 128,000. Each revision is pinned to a Git commit and a safetensors file hash in [sources.json](sources.json). The pinned model cards declare Apache-2.0. [Pythia-14m card](https://huggingface.co/EleutherAI/pythia-14m/blob/cf967c0a9a04383db6f7b1108d86b2962634b4ac/README.md) · [Pythia-70m card](https://huggingface.co/EleutherAI/pythia-70m/blob/a39f36b100fe8a5377810d56c3f4789b9c53ac42/README.md)

| Model | Payload at each step | Published snapshots used | Local artifact bytes |
| --- | --- | ---: | ---: |
| Pythia-14m | Original safetensors file, all 76 FP16 tensors | 34 | 956,893,280 |
| Pythia-70m | Center 32,768 FP32 elements of the embedding and two matrices from each of six layers, 13 slices total | 34 | 57,983,328 |

## Fetch and verify

Run from this repository with Python 3. The commands use the standard library and public HTTPS. Choose a local directory with about 1.1 GB free for weights and receipts. A fresh fetch transfers about 1.02 GB plus HTTP overhead.

```sh
corpus_dir=./scratch/pythia-corpus
python3 fetch_pythia.py "$corpus_dir" --model 14m
python3 fetch_pythia.py "$corpus_dir" --model 70m
python3 verify_source.py "$corpus_dir" "$corpus_dir/audit.json"
```

The fetcher retrieves the pinned model cards and creates one safetensors file and one JSON receipt per revision. A repeated fetch checks completed artifacts and resumes missing ones. The 70m fetcher reads the source header and 13 HTTP byte ranges, checks the source object identifier, commit and length across requests, and emits a safetensors file containing the sampled values in their original byte order. The source object’s hash is the publisher’s Git LFS identifier for 70m. The 14m whole-file hashes and both models’ local artifact hashes are recomputed from the actual bytes.

The offline verifier checks the retrieved model cards, source lock, receipt identities, file dimensions, tensor byte ranges, exact 70m slice coordinates and every local SHA-256. Successful output reports 34 snapshots for each model and both byte totals above. Source locks record the original CDN response ETag for provenance. Identity checks use the pinned Git revision, source LFS hash and SHA-256 of the local artifact.

## Check the single-parent optimizer

```sh
python3 -m unittest test_forest
```

This check needs a C++17 compiler named `c++`. It builds the optimizer in a temporary directory, compares every depth budget on small directed graphs against an exhaustive enumeration, and independently verifies the cost and restore depth of each returned parent assignment.
