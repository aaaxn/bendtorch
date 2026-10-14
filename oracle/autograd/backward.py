import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from generate import (
    EXPECTED_IDENTITY,
    AUTOGRAD_DIR,
    configure_cpu,
    json_bytes,
    observed_identity,
)

from tools.autograd_backward import cases

OUTPUT = AUTOGRAD_DIR / "fixtures/s3-backward-torch-v1.json"

Json = dict[str, Any]


def scalar(word: int) -> torch.Tensor:
    signed = word - (1 << 32) if word >= 1 << 31 else word
    return torch.tensor(signed, dtype=torch.int32).view(torch.float32).clone()


def bits(value: torch.Tensor) -> int:
    return int(value.detach().reshape(()).view(torch.int32).item()) & 0xFFFFFFFF


def record(node: torch.Tensor, nodes: list[torch.Tensor]) -> str:
    if not node.is_leaf:
        node.retain_grad()
    nodes.append(node)
    return f"S3:id:{len(nodes) - 1}"


def operation(name: str, operands: list[torch.Tensor]) -> torch.Tensor:
    if name == "add":
        return operands[0] + operands[1]
    if name == "sub":
        return operands[0] - operands[1]
    if name == "mul":
        return operands[0] * operands[1]
    return -operands[0]


# Replays one backward case on PyTorch scalars. A node id is a position in a
# Python list, so an id the list does not hold raises IndexError (a read of
# such an id records one rejection for the value and one for the gradient). Every
# backward keeps the graph (retain_graph=True) and every non-leaf retains its
# gradient, so .grad can be read on each node.
def replay(case: Json) -> Json:
    if any(step["op"] == "forge" for step in case["steps"]):
        return {"unrepresentable": "forge"}
    nodes: list[torch.Tensor] = []
    lines = []
    for step in case["steps"]:
        name = step["op"]
        try:
            if name == "leaf":
                lines.append(record(scalar(step["word"]).requires_grad_(), nodes))
            elif name in ("add", "sub", "mul", "neg"):
                ids = [step[key] for key in ("a", "b") if key in step]
                operands = [nodes[identifier] for identifier in ids]
                lines.append(record(operation(name, operands), nodes))
            elif name == "backward":
                nodes[step["output"]].backward(retain_graph=True)
                lines.append("S3:backward")
            elif name == "zero_grad":
                for node in nodes:
                    node.grad = None
                lines.append("S3:zero_grad")
            elif step["id"] >= len(nodes):
                lines += ["S3:value:rejected:IndexError", "S3:grad:rejected:IndexError"]
            else:
                node = nodes[step["id"]]
                grad = node.grad
                lines += [
                    f"S3:value:{bits(node)}",
                    "S3:grad:none" if grad is None else f"S3:grad:{bits(grad)}",
                ]
        except IndexError as error:
            lines.append(f"S3:rejected:{type(error).__name__}")
    return {"lines": lines}


def build(identity: Json) -> Json:
    rows = cases()
    return {
        "schema_version": 1,
        "corpus_version": "s3-backward-torch-v1",
        "purpose": "PyTorch observations for every S3 backward request; the root "
        "test compares the gate reference with them",
        "oracle": identity,
        "configuration": {
            "num_threads": 1,
            "deterministic_algorithms": True,
            "flush_denormal": False,
            "retain_graph": True,
            "retain_grad": "every non-leaf",
        },
        "requests": hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()
        ).hexdigest(),
        "backward": [{"id": row["id"], **replay(row)} for row in rows],
    }


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
