"""Cold-restore the single and mixed K=5 layouts through real positioned range reads."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from cold import restore_one
from plans import closures
from storage import Corpus, RecordStore, _write_archive


def run(root, model, start, plans_path, destination, scratch, indices=None):
    corpus = Corpus(root, model, start)
    records = RecordStore(corpus, scratch / "records")
    targets = list(range(len(corpus.steps))) if indices is None else list(indices)
    if len(set(targets)) != len(targets) or any(i < 0 or i >= len(corpus.steps) for i in targets):
        raise ValueError("invalid target indices")
    plans = json.loads(plans_path.read_text())["5"]
    summary = dict(model=model, start=start, cap=5, plan_source=plans_path.name,
                   indices=targets, read_unit="bytes returned by unbuffered positional range reads",
                   cache="fresh empty record map and new file descriptor for every target", layouts={})
    with tempfile.TemporaryDirectory(prefix="cold-audit-", dir=scratch) as folder:
        arc = Path(folder)/"audit.arc"
        for name in ("single", "dual"):
            recipe = plans[name]
            _write_archive(corpus, records, recipe["plan"], arc, True)
            if arc.stat().st_size != recipe["archive_bytes"]:
                raise ValueError("materialized archive size differs from plan ledger")
            expected = closures(recipe["plan"])
            rows = []
            for i in targets:
                item = restore_one(arc, i)
                digest = hashlib.sha256(item.header)
                digest.update(memoryview(item.payload).cast("B"))
                if digest.hexdigest() != corpus.hashes[i]:
                    raise ValueError(f"cold-restored file hash differs at {corpus.steps[i]}")
                if item.read_bytes != recipe["cold_bytes"][i] or set(item.record_ids) != expected[i]:
                    raise ValueError("physical read differs from predicted dependency closure and byte count")
                if len(item.record_ids) != len(set(item.record_ids)) or item.read_calls != len(item.record_ids) + 2:
                    raise ValueError("duplicate reads or unexpected I/O calls")
                rows.append(dict(i=i, step=corpus.steps[i], read_bytes=item.read_bytes,
                                 records=item.record_ids, pread_calls=item.read_calls))
            summary["layouts"][name] = dict(archive_bytes=arc.stat().st_size,
                                            sha256_verified=len(rows), targets=rows)
    destination.write_text(json.dumps(summary, separators=(",", ":")) + "\n")
    print(model,start,[(k,v["sha256_verified"]) for k,v in summary["layouts"].items()],flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("model")
    parser.add_argument("start", type=int)
    parser.add_argument("plans", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("scratch", type=Path)
    parser.add_argument("--indices", nargs="+", type=int)
    args = parser.parse_args()
    run(args.root,args.model,args.start,args.plans,args.results,args.scratch,args.indices)
