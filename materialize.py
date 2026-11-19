"""Write selected K=5 archives, read them back, and check the cost ledger."""
import argparse
import json
from pathlib import Path

from storage import Corpus, RecordStore, verify_archive


def run(root, model, start, layouts, output, scratch, guide_only):
    corpus = Corpus(root,model,start)
    store = RecordStore(corpus,scratch/'records')
    results = {}
    reused = {}
    primary = layouts['5']
    kinds = ('single','dual') if guide_only else (
        'single','dual','endpoints_fixed','collapse','guide_single','guide_dual')
    for kind in kinds:
        recipe = primary[kind]
        plan = tuple(map(tuple,recipe['plan']))
        interpolate = kind != 'endpoints_fixed'
        key = (plan,interpolate if any(len(p)==2 for p in plan) else True)
        if key not in reused:
            out_path = scratch/f'{model}-{start}.arc'
            check = verify_archive(corpus, store, plan, out_path, interpolate)
            reused[key] = check
        check = reused[key]
        if not guide_only:
            for field in ('archive_bytes','per_record_bytes','max_records','cold_bytes'):
                if check[field] != recipe[field]:
                    raise ValueError(f'{kind} {field} disagrees with independently measured ledger')
        assert check['max_records'] <= 5 and check['sha256_verified']==17
        results[kind] = check
        print(model,start,kind,check['archive_bytes'],flush=True)
    output.write_text(json.dumps(results,separators=(',',':'))+'\n')


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('model')
    ap.add_argument('start',type=int)
    ap.add_argument('layouts',type=Path)
    ap.add_argument('output',type=Path)
    ap.add_argument('scratch',type=Path)
    ap.add_argument('--guide-only',action='store_true')
    a=ap.parse_args()
    run(a.root,a.model,a.start,json.loads(a.layouts.read_text()),a.output,a.scratch,a.guide_only)
