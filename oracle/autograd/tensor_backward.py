import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from generate import (
    AUTOGRAD_DIR,
    EXPECTED_IDENTITY,
    configure_cpu,
    json_bytes,
    observed_identity,
    tensor,
)

from tools.autograd_tensor_backward import cases

OUTPUT = AUTOGRAD_DIR / "fixtures/tensor-backward-torch.json"

Json = dict[str, Any]


BACKED = ("add", "sub", "mul", "matmul", "matmul_nt", "sum", "relu", "mse_loss")


def snapshot(value: torch.Tensor) -> list[str]:
    flat = value.detach().contiguous().reshape(-1).view(torch.int32).tolist()
    return [
        f"tensor:shape:{json.dumps(list(value.shape))}",
        f"tensor:count:{len(flat)}",
        *[f"tensor:word:{item & 0xFFFFFFFF}" for item in flat],
    ]


def record(node: torch.Tensor, nodes: list[torch.Tensor]) -> str:
    if not node.is_leaf:
        node.retain_grad()
    nodes.append(node)
    return f"autograd:id:{len(nodes) - 1}"


def operation(name: str, operands: list[torch.Tensor]) -> torch.Tensor:
    if name == "add":
        return operands[0] + operands[1]
    if name == "sub":
        return operands[0] - operands[1]
    if name == "mul":
        return operands[0] * operands[1]
    if name == "matmul":
        return operands[0] @ operands[1]
    if name == "matmul_nt":
        return torch.nn.functional.linear(operands[0], operands[1])
    if name == "relu":
        return torch.relu(operands[0])
    if name == "mse_loss":
        return torch.nn.functional.mse_loss(operands[0], operands[1])
    return operands[0].sum()


# A node id is a position in a Python list, so an id the list does not hold
# raises IndexError; an operation PyTorch refuses raises RuntimeError. Every
# backward seeds ones of the output's shape and keeps the graph
# (retain_graph=True), and every non-leaf retains its gradient, so .grad can
# be read on each node.
def replay(case: Json) -> Json:
    if any(step["op"] == "forge" for step in case["steps"]):
        return {"unrepresentable": "forge"}
    nodes: list[torch.Tensor] = []
    lines = []
    for step in case["steps"]:
        name = step["op"]
        try:
            if name == "leaf":
                value = tensor(step["words"], step["shape"]).requires_grad_()
                lines.append(record(value, nodes))
            elif name in BACKED:
                ids = [step[key] for key in ("a", "b") if key in step]
                operands = [nodes[identifier] for identifier in ids]
                lines.append(record(operation(name, operands), nodes))
            elif name == "cross_entropy":
                # Class indices, mean reduction. A target count other than
                # the batch raises ValueError, a target out of range
                # IndexError.
                logits = nodes[step["a"]]
                targets = torch.tensor(step["targets"], dtype=torch.int64)
                loss = torch.nn.functional.cross_entropy(logits, targets)
                lines.append(record(loss, nodes))
            elif name == "linear":
                # The tape records Linear as x @ W^T, then the bias added to
                # it, and prints the sum's id: linear(x, W), then + b. Both
                # are computed before either is recorded, so a refusal
                # records nothing.
                x, weight = nodes[step["a"]], nodes[step["b"]]
                bias = nodes[step["bias"]]
                product = torch.nn.functional.linear(x, weight)
                total = product + bias
                record(product, nodes)
                lines.append(record(total, nodes))
            elif name == "backward":
                output = nodes[step["output"]]
                output.backward(torch.ones_like(output), retain_graph=True)
                lines.append("autograd:backward")
            elif name == "zero_grad":
                for node in nodes:
                    node.grad = None
                lines.append("autograd:zero_grad")
            elif step["id"] >= len(nodes):
                lines += [
                    "autograd:value:rejected:IndexError",
                    "autograd:grad:rejected:IndexError",
                ]
            else:
                node = nodes[step["id"]]
                grad = node.grad
                lines += ["autograd:value", *snapshot(node)]
                lines += (
                    ["autograd:grad:none"]
                    if grad is None
                    else ["autograd:grad", *snapshot(grad)]
                )
        except (IndexError, RuntimeError, ValueError) as error:
            lines.append(f"autograd:rejected:{type(error).__name__}")
    return {"lines": lines}


def build(identity: Json) -> Json:
    rows = cases()
    return {
        "schema_version": 1,
        "corpus_version": "tensor-backward-torch",
        "purpose": "PyTorch observations for every tensor backward request; the "
        "root test compares the gate reference with them",
        "oracle": identity,
        "configuration": {
            "num_threads": 1,
            "deterministic_algorithms": True,
            "flush_denormal": False,
            "retain_graph": True,
            "retain_grad": "every non-leaf",
            "seed": "ones_like(output)",
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
