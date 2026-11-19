"""Measure all eligible parent choices on the full cohort payload, without retaining candidates."""
import argparse
import hashlib
from compression import zstd
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct

import numpy as np

from codec import encode
from consolidate import collect

BLOCK = 262144
MODELS = ('14m','70m')
STARTS = (32000,112000)


def one_child(args):
    root, model, start, i = args
    arrays, layouts, headers = [], [], []
    for step in range(start, start+17000, 1000):
        path = root / model / f'{step}.safetensors'
        with path.open('rb') as stream:
            prefix = stream.read(8)
            header = stream.read(struct.unpack('<Q', prefix)[0])
        layout = json.loads(header)
        layout.pop('__metadata__', None)
        layouts.append(layout)
        headers.append(5 + len(zstd.compress(prefix+header,level=3)))
        kind, = {t['dtype'] for t in layout.values()}
        arrays.append(np.memmap(path, mode='r', dtype={'F16':'<f2','F32':'<f4'}[kind],
                                offset=8+len(header)))
    assert all(layout == layouts[0] for layout in layouts)
    header = headers[i]
    n = len(arrays)
    standalone = header
    singles = [header + 1] * n
    radii = [r for r in (1,2,4,8) if 0 <= i-r and i+r < n]
    bridges = {r:[header+2,header+2] for r in radii}
    for a in range(0,len(arrays[i]),BLOCK):
        x = arrays[i][a:a+BLOCK]
        standalone += 4 + len(encode(x))
        single_blocks = {}
        for j in range(n):
            if i != j:
                y = arrays[j][a:a+BLOCK]
                size = len(encode(x,y))
                single_blocks[j] = size
                singles[j] += 4 + size
        for r in radii:
            y = arrays[i-r][a:a+BLOCK]
            z = arrays[i+r][a:a+BLOCK]
            pred = ((y.astype('f8') + z.astype('f8')) * .5).astype(x.dtype)
            either = min(single_blocks[i-r],single_blocks[i+r])
            bridges[r][0] += 5 + min(either, len(encode(x,pred)))
            bridges[r][1] += 5 + either
    singles[i] = -1
    return model, start, i, dict(anchor=standalone, singles=singles,
                          bridges=[[i,i-r,i+r,*sizes] for r,sizes in bridges.items()])


def cache_key(root):
    # Resuming a partial ledger is valid only for the same inputs and byte code.
    import sys
    digest = hashlib.sha256()
    for filename in ('full_costs.py','codec.py','context_codec.cc'):
        digest.update(Path(__file__).with_name(filename).read_bytes())
    digest.update(repr((sys.version, np.__version__, zstd.zstd_version)).encode())
    for model in MODELS:
        for start in STARTS:
            for step in range(start,start+17000,1000):
                with (root/model/f'{step}.safetensors').open('rb') as source:
                    digest.update(hashlib.file_digest(source,'sha256').digest())
    return digest.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('out',type=Path)
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    stamp = args.out/'cache-key'
    key = cache_key(args.root)
    if stamp.exists() and stamp.read_text() != key:
        raise ValueError('Use a fresh cost directory for modified code or input files')
    if not stamp.exists():
        if any(args.out.glob('cost-*.json')):
            raise ValueError('Use a fresh directory for uncatalogued partial cost rows')
        stamp.write_text(key)
    def part(m,s,i):
        return args.out / f'cost-{m}-{s}-{i}.json'
    jobs=[(args.root,m,s,i) for m in reversed(MODELS) for i in range(17)
          for s in STARTS if not part(m,s,i).exists()]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for m,s,i,result in executor.map(one_child,jobs,chunksize=1):
            dest = part(m,s,i)
            temp = dest.with_suffix('.tmp')
            temp.write_text(json.dumps(result))
            temp.replace(dest)
            print(m,s,i,flush=True)
    for m in MODELS:
        for s in STARTS:
            scores = collect(args.out,m,s)
            (args.out / f'full-{m}-{s}.json').write_text(json.dumps(scores,separators=(',',':'))+'\n')


if __name__ == '__main__':
    main()
