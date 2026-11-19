"""Bound the minimum payload bytes for a fully measured cohort."""
import argparse
import json
from pathlib import Path

import scipy

from certify import bound
from consolidate import collect
from experiment import forests, ingredients, measure
from plans import choose_plan, closures


def scores_at(directory, model, start):
    complete = directory / f"full-{model}-{start}.json"
    if complete.exists():
        return json.loads(complete.read_text())
    return collect(directory,model,start)


def run(source, binary, results, model, start, k, seconds):
    anchors, singles, bridges, _ = ingredients(scores_at(source, model, start))
    directory = 22 + 8 * (len(anchors)+1)
    one_parent, one_plan = forests(binary, anchors, singles, [k])[k]
    heuristic = choose_plan(one_plan, anchors, singles, bridges, k)
    guide = results / f"guideplans-{model}-{start}.json"
    if guide.exists():
        guide_plan = json.loads(guide.read_text())[str(k)]["dual"]["plan"]
        guide_seed = choose_plan(guide_plan, anchors, singles, bridges, k)
        candidates = (heuristic, guide_seed)
        heuristic = min(candidates, key=lambda plan: measure(plan, anchors, singles, bridges)["archive_bytes"])
    heuristic_bytes = measure(heuristic, anchors, singles, bridges)["archive_bytes"] - directory
    cert = bound(anchors, singles, bridges, k, seconds, heuristic)
    plan = cert["parents"]
    recovered = measure(plan, anchors, singles, bridges)
    assert recovered["archive_bytes"] - directory == cert["feasible_cost"]
    assert max(map(len, closures(plan))) <= k
    assert cert["lower_bound"] <= cert["feasible_cost"] <= heuristic_bytes <= one_parent
    cert.update(model=model, start=start, cap=k, solver_time_limit_seconds=seconds,
                scipy_version=scipy.__version__, payload_cost_excludes_directory_bytes=directory,
                exact_single_parent_bytes=one_parent, heuristic_bytes=heuristic_bytes,
                max_cold_records=recovered["max_records"], mean_cold_bytes=recovered["mean_cold_bytes"])
    out = results / f"milp-{model}-{start}.json"
    out.write_text(json.dumps(cert,indent=2)+"\n")
    print(out, one_parent, heuristic_bytes, cert["lower_bound"], cert["feasible_cost"],
          cert["optimal"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("costs", type=Path)
    p.add_argument("forest_binary", type=Path)
    p.add_argument("results", type=Path)
    p.add_argument("model", choices=("14m","70m"))
    p.add_argument("start", type=int, choices=(32000,112000))
    p.add_argument("--cap", type=int, default=5)
    p.add_argument("--seconds", type=int, default=20)
    a = p.parse_args()
    run(a.costs, a.forest_binary, a.results, a.model, a.start, a.cap, a.seconds)
