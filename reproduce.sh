#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
corpus=${1:?provide the directory containing the pinned Pythia corpus}
work=${2:-scratch/reproduce}
mkdir -p "$work" results
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$work/matplotlib"
python verify_source.py "$corpus" results/input-audit.json
c++ -O3 -std=c++17 forest.cpp -o "$work/checkpoint_forest"
binary="$work/checkpoint_forest"
python -m unittest discover
python full_costs.py "$corpus" "$work/costs"
for model in 14m 70m; do
  for start in 32000 112000; do
    setname="$model-$start"
    cp "$work/costs/full-$setname.json" "results/full-$setname.json"
    python guide.py "$corpus" "$model" "$start" "results/guide-$setname.json"
    python experiment.py "results/guide-$setname.json" "$binary" "results/guideplans-$setname.json"
    python materialize.py "$corpus" "$model" "$start" "results/guideplans-$setname.json" \
      "results/replay-guide-$setname.json" "$work" --guide-only
    python experiment.py "results/full-$setname.json" "$binary" "results/fullplans-$setname.json" \
      --guide-results "results/guideplans-$setname.json"
    python materialize.py "$corpus" "$model" "$start" "results/fullplans-$setname.json" \
      "results/verified-$setname.json" "$work"
    python cold_audit.py "$corpus" "$model" "$start" "results/fullplans-$setname.json" \
      "results/cold-$setname.json" "$work"
    python certification.py results "$binary" results "$model" "$start"
  done
done
python precision.py "$corpus" results/precision.json
for start in 32000 112000; do
  python narrow.py "$corpus" "$start" "$binary" "$work/narrow-$start" "results/narrow-70m-$start.json"
done
python summarize.py
