import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from generate import (
    TENSOR_DIR,
    configure_cpu,
    json_bytes,
    observed_identity,
    verify_identity,
)

ROOT = TENSOR_DIR.parents[1]
OUTPUT = TENSOR_DIR / "fixtures/signed-torch.json"
sys.path.insert(0, str(ROOT))

from tools.tensor_signed import cases  # noqa: E402

Json = dict[str, Any]


def tensor(words: list[int], shape: list[int]) -> torch.Tensor:
    signed = [word - (1 << 32) if word >= 1 << 31 else word for word in words]
    return torch.tensor(signed, dtype=torch.int32).view(torch.float32).reshape(shape)


def snapshot(value: torch.Tensor) -> list[str]:
    flat = value.detach().contiguous().reshape(-1).view(torch.int32).tolist()
    return [
        f"tensor:shape:{json.dumps(list(value.shape))}",
        f"tensor:count:{len(flat)}",
        *[f"tensor:word:{item & 0xFFFFFFFF}" for item in flat],
    ]


def as_float(word: int) -> torch.Tensor:
    return tensor([word], [])


# Each request with PyTorch's own signed arguments: reshape, permute_copy and
# transpose_copy, indexing x[index] and its write, and x.flatten()[index] and
# its write through view(-1). An argument PyTorch refuses raises IndexError or
# RuntimeError.
def apply(case: Json) -> torch.Tensor:
    x = tensor(case["words"], case["shape"])
    op = case["op"]
    if op == "reshape":
        return x.reshape(case["sizes"])
    if op == "permute":
        return torch.permute_copy(x, case["dims"])
    if op == "transpose":
        a, b = case["dims"]
        return torch.transpose_copy(x, a, b)
    if op == "get":
        return x[tuple(case["index"])]
    if op == "set":
        y = x.clone()
        y[tuple(case["index"])] = as_float(case["value"])
        return y
    if op == "get_flat":
        return x.flatten()[case["index"]]
    if op == "set_flat":
        y = x.clone()
        y.view(-1)[case["index"]] = as_float(case["value"])
        return y
    raise ValueError(f"Unknown signed request: {op}")


def replay(case: Json) -> list[str]:
    try:
        return snapshot(apply(case))
    except (IndexError, RuntimeError) as error:
        return [f"tensor:rejected:{type(error).__name__}"]


def build(identity: Json) -> Json:
    rows = cases()
    return {
        "schema_version": 1,
        "corpus_version": "signed-torch",
        "purpose": "PyTorch observations for every signed axis, index and -1 "
        "reshape request; the root test compares the gate reference with them",
        "oracle": identity,
        "configuration": {"num_threads": 1, "deterministic_algorithms": True},
        "requests": hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()
        ).hexdigest(),
        "signed": [{"id": row["id"], "lines": replay(row)} for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    identity = observed_identity()
    verify_identity(identity)
    configure_cpu()
    data = json_bytes(build(identity))
    output = args.output.resolve()
    if not output.is_relative_to(TENSOR_DIR):
        raise RuntimeError("Outputs are restricted to oracle/tensor/")
    if args.check:
        if output.read_bytes() != data:
            raise SystemExit(f"{output.name} differs from a fresh generation")
        print(f"Verified {output.name}: {hashlib.sha256(data).hexdigest()}")
        return
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    output.write_bytes(data)
    print(f"Wrote {output.name}: {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
