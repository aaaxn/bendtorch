import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

AUTOGRAD_DIR = Path(__file__).resolve().parent
ROOT = AUTOGRAD_DIR.parents[1]
OUTPUT = AUTOGRAD_DIR / "fixtures/s3-torch-v1.json"
LITERAL_LIMIT = 4096
EXPECTED_IDENTITY = {
    "torch_version": "2.14.1+cpu",
    "torch_git_version": "5c4886908584029761b579af026dcfb627c84070",
    "cuda_version": None,
    "hip_version": None,
}
sys.path.insert(0, str(ROOT))

from tools.autograd_broadcast import cases as broadcast_cases  # noqa: E402
from tools.autograd_gradient import cases as gradient_cases  # noqa: E402
from tools.autograd_reduce import cases as reduce_cases  # noqa: E402

Json = dict[str, Any]


def configure_cpu() -> None:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    if not torch.set_flush_denormal(False):
        raise RuntimeError("This backend cannot explicitly disable denormal flushing")


def observed_identity() -> Json:
    return {
        "torch_version": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "cuda_version": torch.version.cuda,
        "hip_version": torch.version.hip,
    }


def tensor(words: list[int], shape: list[int]) -> torch.Tensor:
    signed = [word - (1 << 32) if word >= 1 << 31 else word for word in words]
    data = torch.tensor(signed, dtype=torch.int32).view(torch.float32)
    return data.reshape(shape).clone()


def words(value: torch.Tensor) -> list[str]:
    flat = value.detach().contiguous().reshape(-1).view(torch.int32).tolist()
    return [f"{item & 0xFFFFFFFF:08x}" for item in flat]


def observed(value: torch.Tensor) -> Json:
    found = words(value)
    if len(found) > LITERAL_LIMIT:
        digest = hashlib.sha256(" ".join(found).encode()).hexdigest()
        return {"shape": list(value.shape), "count": len(found), "sha256": digest}
    return {"shape": list(value.shape), "words": " ".join(found)}


def attempt(action) -> Json:
    try:
        return observed(action())
    except RuntimeError as error:
        return {"rejected": type(error).__name__}


def broadcast(case: Json) -> Json:
    x = tensor(case["words"], case["shape"])
    return {"expand": attempt(lambda: x.expand(case["target"]).clone())}


def expand_gradient(case: Json) -> torch.Tensor:
    leaf = torch.zeros(case["target"], requires_grad=True)
    leaf.expand(case["shape"]).backward(tensor(case["words"], case["shape"]))
    if leaf.grad is None:
        raise RuntimeError("expand produced no gradient")
    return leaf.grad


def reduce(case: Json) -> Json:
    x = tensor(case["words"], case["shape"])
    return {
        "sum_to_size": attempt(lambda: x.sum_to_size(case["target"]).clone()),
        "expand_gradient": attempt(lambda: expand_gradient(case)),
    }


def gradient(case: Json) -> Json:
    try:
        leaf = torch.zeros(case["shape"], requires_grad=True)
    except RuntimeError as error:
        return {"rejected": type(error).__name__}
    outcomes = []
    for item in case["items"]:
        try:
            leaf.backward(tensor(item["words"], item["shape"]))
        except RuntimeError as error:
            outcomes.append({"rejected": type(error).__name__})
        else:
            outcomes.append({"accumulated": True})
    grad = None if leaf.grad is None else observed(leaf.grad)
    return {"outcomes": outcomes, "grad": grad}


def build(identity: Json) -> Json:
    return {
        "schema_version": 1,
        "corpus_version": "s3-torch-v1",
        "purpose": "PyTorch observations for every S3 broadcast, reduce_to and "
        "gradient request; the root test compares the gate references with them",
        "oracle": identity,
        "configuration": {
            "num_threads": 1,
            "deterministic_algorithms": True,
            "flush_denormal": False,
        },
        "requests": {
            name: hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
            for name, rows in (
                ("broadcast", broadcast_cases()),
                ("reduce", reduce_cases()),
                ("gradient", gradient_cases()),
            )
        },
        "broadcast": [{"id": c["id"], **broadcast(c)} for c in broadcast_cases()],
        "reduce": [{"id": c["id"], **reduce(c)} for c in reduce_cases()],
        "gradient": [{"id": c["id"], **gradient(c)} for c in gradient_cases()],
    }


def json_bytes(document: Any) -> bytes:
    return (json.dumps(document, indent=1, sort_keys=True) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    identity = observed_identity()
    if identity != EXPECTED_IDENTITY:
        raise RuntimeError(f"Oracle identity mismatch: {identity!r}")
    configure_cpu()
    data = json_bytes(build(identity))
    output = args.output.resolve()
    if not output.is_relative_to(AUTOGRAD_DIR):
        raise RuntimeError("Outputs are restricted to oracle/autograd/")
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
