"""Consolidate completed full-payload cost rows into a portable score table."""
import argparse
import json
from pathlib import Path


def collect(parts, model, start):
    rows = [json.loads((parts / f'cost-{model}-{start}-{i}.json').read_text())
            for i in range(17)]
    return dict(model=model, start=start, anchors=[r['anchor'] for r in rows],
                singles=[r['singles'] for r in rows],
                bridges=[b for r in rows for b in r['bridges']])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('parts', type=Path)
    ap.add_argument('model')
    ap.add_argument('start', type=int)
    ap.add_argument('out', type=Path)
    a = ap.parse_args()
    a.out.write_text(json.dumps(collect(a.parts,a.model,a.start),separators=(',',':'))+'\n')
