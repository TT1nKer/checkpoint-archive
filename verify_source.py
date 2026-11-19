"""Offline audit of the published Pythia source lock, retrieved weights and model cards."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import struct

from fetch_pythia import DEFAULT_LOCK, SAMPLE_COUNT, STEPS, TENSOR_NAMES, check_lock


WIDTH = {"F32": 4, "F16": 2, "BF16": 2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect(path):
    data = path.read_bytes()
    require(len(data) >= 8, f"Truncated file: {path}")
    end = 8 + struct.unpack("<Q", data[:8])[0]
    require(end <= len(data), f"Truncated header: {path}")
    schema = json.loads(data[8:end])
    schema.pop("__metadata__", None)
    ranges = sorted((info["data_offsets"], key) for key, info in schema.items())
    previous = 0
    for (start, stop), key in ranges:
        tensor = schema[key]
        size = math.prod(tensor["shape"]) * WIDTH[tensor["dtype"]]
        require(start == previous and stop - start == size and stop <= len(data) - end,
                f"Invalid offsets or dimensions: {path}, {key}")
        previous = stop
    require(end + previous == len(data), f"Unaccounted bytes: {path}")
    return data, schema, end


def check_card(root, model, license_info):
    body = (root / f"card-{model}.md").read_bytes()
    url = (f"https://huggingface.co/EleutherAI/pythia-{model}/resolve/"
           f"{license_info['main_revision']}/README.md")
    require(license_info["pinned_model_card"] == url, f"Unexpected card URL: {model}")
    require(sha256(body).hexdigest() == license_info["card_sha256"],
            f"Model card differs from lock: {model}")
    lines = body.decode().splitlines()
    require(lines[0] == "---" and "---" in lines[1:], f"Missing card front matter: {model}")
    license_lines = [line for line in lines[1:lines.index("---", 1)] if line.startswith("license:")]
    require(license_lines == ["license: apache-2.0"] == license_info["yaml"]
            and license_info["license"] == "apache-2.0", f"Unexpected card license: {model}")


def check_slices(data, schema, offset, receipt, expected):
    slices = receipt["slices"]
    require(set(TENSOR_NAMES) == set(schema) == set(slices) == set(expected), "Unexpected slice selection")
    for name, tensor in schema.items():
        start, stop = tensor["data_offsets"]
        entry = slices[name]
        location = {key: val for key, val in entry.items() if key != "slice_sha256"}
        require(location == expected[name], f"Source coordinates differ from lock: {name}")
        source_start, source_stop = entry["source_byte_range_half_open"]
        first = (math.prod(entry["source_shape"]) - SAMPLE_COUNT) // 2
        require(tensor["shape"] == [SAMPLE_COUNT] and entry["element_count"] == SAMPLE_COUNT
                and tensor["dtype"] == entry["dtype"] and entry["first_element"] == first
                and first >= 0, f"Wrong slice dimensions: {name}")
        require(0 <= source_start < source_stop <= receipt["source_size"]
                and source_stop - source_start == stop - start, f"Wrong source interval: {name}")
        require(sha256(data[offset + start:offset + stop]).hexdigest() == entry["slice_sha256"],
                f"Wrong slice digest: {name}")


def verify(root, source_lock=DEFAULT_LOCK):
    lock = json.loads(source_lock.read_text())
    summary = {}
    for model in ("14m", "70m"):
        locked = lock["models"][model]
        check_card(root, model, locked["license"])
        steps = {row["step"]: row for row in locked["checkpoints"]}
        require(set(steps) == set(STEPS) and len(steps) == len(locked["checkpoints"]),
                f"Unexpected snapshot roster: {model}")
        schemas = set()
        total = 0
        for step in STEPS:
            path = root / model / f"{step}.safetensors"
            receipt = json.loads(path.with_suffix(".json").read_text())
            data, schema, offset = inspect(path)
            check_lock(receipt, steps[step])
            expected_url = (f"https://huggingface.co/EleutherAI/pythia-{model}/resolve/"
                            f"{steps[step]['commit']}/model.safetensors")
            require(receipt["url"] == expected_url and receipt["model"] == f"EleutherAI/pythia-{model}"
                    and receipt["branch"] == f"step{step}" and receipt["source_tensor_count"] == 76
                    and receipt["license"] == "Apache-2.0", f"Source receipt mismatch: {path}")
            digest = sha256(data).hexdigest()
            require(digest == receipt["local_sha256"] and len(data) == receipt["local_bytes"],
                    f"Local artifact mismatch: {path}")
            if model == "14m":
                require(receipt["slices"] is None and receipt["source_sha256"] == digest
                        and receipt["source_size"] == len(data), f"Full source mismatch: {path}")
            else:
                check_slices(data, schema, offset, receipt, locked["slices"])
            schemas.add(tuple(sorted((key, value["dtype"], tuple(value["shape"]))
                                     for key, value in schema.items())))
            total += len(data)
        require(len(schemas) == 1, f"Tensor schema changed within {model}")
        schema = schemas.pop()
        require(len(schema) == locked["tensor_count"]
                and Counter(dtype for _, dtype, _ in schema) == locked["dtype_counts"]
                and sum(math.prod(shape) for _, _, shape in schema) == locked["elements"],
                f"Tensor inventory differs from lock: {model}")
        summary[model] = {"snapshots": len(steps), "tensor_count": len(schema),
                          "elements_per_snapshot": locked["elements"], "local_bytes": total}
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path, nargs="?", help="Optional JSON audit summary")
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    report = json.dumps(verify(args.root, args.source_lock), indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(report)
    print(report, end="")
