"""Fetch the outcome-blind Pythia weight corpus for the bounded-restore pilot."""
import argparse
import concurrent.futures as futures
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
from urllib.error import HTTPError
from urllib.request import build_opener, HTTPRedirectHandler, Request
import time

ENDPOINT = "https://huggingface.co"
DEFAULT_LOCK = Path(__file__).resolve().with_name("sources.json")
STEPS = tuple(range(32000,48001,1000))+tuple(range(112000,128001,1000))
SAMPLE_COUNT = 32768
TENSOR_NAMES = ["gpt_neox.embed_in.weight"]
for layer in range(6):
    TENSOR_NAMES.extend([f"gpt_neox.layers.{layer}.attention.query_key_value.weight",
                         f"gpt_neox.layers.{layer}.mlp.dense_h_to_4h.weight"])


class SourceRedirect(HTTPRedirectHandler):
    def __init__(self):
        self.meta = {}

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        for name in ["x-linked-etag","x-repo-commit","x-linked-size"]:
            if headers.get(name):
                self.meta[name] = headers.get(name).strip('"')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def retrieve(url, interval=None):
    for attempt in range(5):
        redirect = SourceRedirect()
        headers = {"User-Agent":"checkpoint-archive-research/1.0", "Accept-Encoding":"identity"}
        if interval is not None:
            start, stop = interval
            headers["Range"] = f"bytes={start}-{stop-1}"
        try:
            # Each cache key names its exact byte range; the resulting signed URL never goes in the receipt.
            suffix = f"?archive_range={start}-{stop-1}" if interval else "?archive_full=1"
            with build_opener(redirect).open(Request(url+suffix,headers=headers), timeout=90) as response:
                body = response.read()
                metadata = {**redirect.meta}
                metadata["response_etag"] = (response.headers.get("etag") or "").strip('"')
                if interval is not None:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range",""))
                    if (response.status != 206 or match is None or
                        (int(match[1]),int(match[2])) != (start, stop-1) or len(body) != stop-start):
                        raise ValueError("Unexpected HTTP range or length")
                    metadata["source_size"] = int(match[3])
                else:
                    if response.status != 200:
                        raise ValueError("Expected whole-file response")
                    metadata["source_size"] = len(body)
                return body, metadata
        except (OSError,ValueError) as exc:
            if attempt == 4 or (isinstance(exc,HTTPError) and exc.code in (404,410)):
                raise
            time.sleep(0.7*(attempt+1))


def unpack_header(raw):
    size, = struct.unpack("<Q",raw[:8])
    if len(raw)<8+size:
        raise ValueError("Short safetensors header")
    return json.loads(raw[8:8+size]), 8+size


def save_sample(tensors):
    header, payload = {}, bytearray()
    for name,(info,data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype":info["dtype"],"shape":[SAMPLE_COUNT],"data_offsets":[start,len(payload)]}
    text=json.dumps(header,separators=(",",":")).encode()
    text+=b" "*(-len(text)%8)
    return struct.pack("<Q",len(text))+text+payload


def check_lock(receipt, expected):
    for key in ("step", "commit", "source_sha256", "source_size", "local_sha256", "local_bytes"):
        if receipt[key] != expected[key]:
            raise ValueError(f"Pinned {key} mismatch at step {receipt['step']}")


def fetch_card(root, model, license_info):
    path = root / f"card-{model}.md"
    if path.exists():
        body = path.read_bytes()
    else:
        body, _ = retrieve(license_info["pinned_model_card"])
    if sha256(body).hexdigest() != license_info["card_sha256"]:
        raise ValueError(f"Pinned model card mismatch for {model}")
    if not path.exists():
        path.write_bytes(body)


def fetch_step(root, model, step, commit, expected):
    directory = root/model
    directory.mkdir(exist_ok=True, parents=True)
    out=directory/f"{step}.safetensors"
    receipt=directory/f"{step}.json"
    if receipt.exists() and out.exists():
        previous=json.loads(receipt.read_text())
        check_lock(previous,expected)
        if sha256(out.read_bytes()).hexdigest()!=previous["local_sha256"]:
            raise ValueError(f"bad existing file: {out}")
        return
    url=f"{ENDPOINT}/EleutherAI/pythia-{model}/resolve/{commit}/model.safetensors"
    metadata={}
    if model=="14m":
        result,metadata=retrieve(url)
        schema, start=unpack_header(result)
        declared=max(v["data_offsets"][1] for k,v in schema.items() if k!="__metadata__")
        if declared!=len(result)-start:
            raise ValueError("Size mismatch vs safetensors header")
        if sha256(result).hexdigest()!=metadata["x-linked-etag"]:
            raise ValueError("Full source hash mismatch")
        clip=None
    else:
        header,metadata=retrieve(url,(0,16384))
        schema,base=unpack_header(header)
        clip,tensors={},{}
        for name in TENSOR_NAMES:
            info=schema[name]
            before,after=info["data_offsets"]
            width={"F32":4,"F16":2,"BF16":2}[info["dtype"]]
            num=(after-before)//width
            first=(num-SAMPLE_COUNT)//2
            start=base+before+first*width
            stop=start+SAMPLE_COUNT*width
            if first<0 or stop>base+after:
                raise ValueError("Invalid fixed source slice")
            data,part=retrieve(url,(start,stop))
            if any(part[k]!=metadata[k] for k in ["x-repo-commit","x-linked-etag","source_size"]):
                raise ValueError("Inconsistent source between ranges")
            clip[name]=dict(source_shape=info["shape"], dtype=info["dtype"], first_element=first,
                            element_count=SAMPLE_COUNT,source_byte_range_half_open=[start,stop],
                            slice_sha256=sha256(data).hexdigest())
            tensors[name]=(info,data)
        result=save_sample(tensors)
    if metadata["x-repo-commit"]!=commit:
        raise ValueError("Unexpected resolved commit")
    digest=sha256(result).hexdigest()
    source_tensors={k:v for k,v in schema.items() if k!="__metadata__"}
    audit=dict(model=f"EleutherAI/pythia-{model}",step=step,branch=f"step{step}",commit=commit,
               url=url,license="Apache-2.0",source_sha256=metadata["x-linked-etag"],
               source_size=metadata["source_size"],response_etag=metadata["response_etag"],
               source_tensor_count=len(source_tensors),dtypes=sorted({v["dtype"] for v in source_tensors.values()}),
               local_sha256=digest,local_bytes=len(result),slices=clip)
    check_lock(audit, expected)
    tmp=out.with_suffix(".partial")
    tmp.write_bytes(result)
    tmp.replace(out)
    receipt.write_text(json.dumps(audit,indent=2)+"\n")
    print(model,step,len(result),digest[:10],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--model",choices=["14m","70m"],required=True)
    p.add_argument("--workers",type=int,default=5)
    p.add_argument("--source-lock",type=Path,default=DEFAULT_LOCK,
                   help="Use the pilot manifest's immutable commits and artifact hashes (default: sources.json).")
    args=p.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    pinned_model = json.loads(args.source_lock.read_text())["models"][args.model]
    inputs = pinned_model["checkpoints"]
    expected = {entry["step"]: entry for entry in inputs}
    if set(expected) != set(STEPS) or len(inputs) != len(STEPS):
        raise ValueError("Missing, duplicated or extra steps in source lock")
    fetch_card(args.root, args.model, pinned_model["license"])
    with futures.ThreadPoolExecutor(args.workers) as pool:
        work={pool.submit(fetch_step,args.root,args.model,step,expected[step]["commit"],expected[step]):step
              for step in STEPS}
        failed=[]
        for job in futures.as_completed(work):
            try: job.result()
            except Exception as error:
                failed.append(work[job])
                print("FAILED",args.model,work[job],type(error).__name__,str(error)[:220],flush=True)
    if failed:
        raise SystemExit(f"Failed {len(failed)} steps")

if __name__=="__main__":
    main()
