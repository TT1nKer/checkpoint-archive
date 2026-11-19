"""Choose archive layouts and compare exact, byte-counted costs."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from plans import choose_plan, closures


def forests(binary, anchors, singles, caps):
    n = len(anchors)
    body = [n, max(caps), *anchors, *np.asarray(singles).ravel()]
    run = subprocess.run([str(binary.resolve()), '--all'], input=' '.join(map(str,body)),
                         capture_output=True, text=True, check=True)
    lines = run.stdout.splitlines()
    answers = {}
    for k in caps:
        parents = [int(x) for x in lines[2*k-1].split()]
        plan = tuple(() if p==-1 else (p,) for p in parents)
        answers[k] = (int(lines[2*k-2]), plan)
    return answers


def ingredients(scores):
    anchors, singles = scores['anchors'],np.array(scores['singles'])
    both = {(c,l,r):price for c,l,r,price,_ in scores['bridges']}
    endpoints = {(c,l,r):price for c,l,r,_,price in scores['bridges']}
    return anchors,singles,both,endpoints


def measure(plan, anchors, singles, bridges):
    costs = [int(anchors[i] if not p else singles[i,p[0]] if len(p)==1 else bridges[i,*p])
             for i,p in enumerate(plan)]
    deps = closures(plan)
    directory = 22 + 8 * (len(plan)+1)
    cold = [directory + sum(costs[i] for i in group) for group in deps]
    return dict(plan=plan, archive_bytes=directory+sum(costs), mean_cold_bytes=sum(cold)/len(cold),
                per_record_bytes=costs, cold_bytes=cold, max_records=max(map(len,deps)),
                dual_records=sum(len(p)==2 for p in plan))


def relaxation(binary, anchors, singles, bridges, caps):
    # Deleting either parent from every bridge preserves feasibility. Assign each bridge
    # to a predetermined endpoint at the original byte price and solve the relaxed forest.
    # Each fixed assignment is a lower bound; taking their maximum remains valid.
    policies = [lambda c,l,r:l,lambda c,l,r:r,
                lambda c,l,r:min((l,r),key=lambda p:singles[c,p]),
                lambda c,l,r:max((l,r),key=lambda p:singles[c,p])]
    estimates = {k:0 for k in caps}
    for select in policies:
        matrix = singles.copy()
        for (c,l,r), price in bridges.items():
            p = select(c,l,r)
            old = matrix[c,p]
            matrix[c,p] = price if old < 0 else min(old,price)
        solutions = forests(binary,anchors,matrix,caps)
        for k in caps:
            estimates[k] = max(estimates[k], solutions[k][0] + 22 + 8*(len(anchors)+1))
    return estimates


def optimize(scores,binary,guides=None):
    capset = (3,5,8)
    anchors,singles,both,endpoints = ingredients(scores)
    base = forests(binary,anchors,singles,capset)
    floors = relaxation(binary,anchors,singles,both,capset)
    result = {}
    for k in capset:
        optimum, single = base[k]
        dual = choose_plan(single,anchors,singles,both,k)
        if guides is not None:
            pilot_plan = tuple(map(tuple,guides[str(k)]['dual']['plan']))
            alternative = choose_plan(pilot_plan,anchors,singles,both,k)
            def score(p):
                rec = measure(p,anchors,singles,both)
                return rec['archive_bytes'],rec['mean_cold_bytes']
            dual = min((dual,alternative),key=score)
        def metric(p, bridge_cost=both):
            return measure(p,anchors,singles,bridge_cost)
        collapse=tuple((min(parents,key=lambda p:singles[i,p]),) if len(parents)==2 else parents
                       for i,parents in enumerate(dual))
        rec = dict(single=metric(single), dual=metric(dual), endpoints_fixed=metric(dual,endpoints),
                   collapse=metric(collapse), dual_lower_bound=floors[k])
        assert rec['single']['archive_bytes']==optimum + 22 + 8*(len(anchors)+1)
        assert rec['single']['archive_bytes'] <= rec['collapse']['archive_bytes']
        assert floors[k] <= rec['dual']['archive_bytes'] <= rec['single']['archive_bytes']
        assert max(v['max_records'] for v in rec.values() if isinstance(v,dict))<=k
        if guides is not None:
            for kind in ('single','dual'):
                rec['guide_'+kind] = metric(tuple(map(tuple,guides[str(k)][kind]['plan'])))
        result[str(k)] = rec
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('scores',type=Path)
    ap.add_argument('binary',type=Path)
    ap.add_argument('out',type=Path)
    ap.add_argument('--guide-results',type=Path)
    a=ap.parse_args()
    guides=json.loads(a.guide_results.read_text()) if a.guide_results else None
    res=optimize(json.loads(a.scores.read_text()),a.binary,guides)
    a.out.write_text(json.dumps(res,separators=(',',':'))+'\n')
    print(a.out,[(k,v['single']['archive_bytes'],v['dual']['archive_bytes'],v['dual_lower_bound'])
                  for k,v in res.items()],flush=True)


if __name__=='__main__':
    main()
