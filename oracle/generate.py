import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torch.__config__

EXPECTED_VERSION = "2.14.1+cpu"
EXPECTED_GIT_VERSION = "5c4886908584029761b579af026dcfb627c84070"
ORACLE_DIR = Path(__file__).resolve().parent


def observed_identity():
    return {
        "torch_version": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "cuda_version": torch.version.cuda,
        "hip_version": torch.version.hip,
    }


def verify_identity(identity):
    expected = {
        "torch_version": EXPECTED_VERSION,
        "torch_git_version": EXPECTED_GIT_VERSION,
        "cuda_version": None,
        "hip_version": None,
    }
    if identity != expected:
        raise RuntimeError(
            f"Oracle identity mismatch: expected {expected!r}, observed {identity!r}"
        )


def configure_cpu():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def f32(values):
    return torch.tensor(values, dtype=torch.float32, device="cpu")


def from_bits(words):
    signed = [word if word < 0x80000000 else word - 0x100000000 for word in words]
    return torch.tensor(signed, dtype=torch.int32, device="cpu").view(torch.float32)


def classify(word):
    if word & 0x7F800000 != 0x7F800000:
        return None
    if word & 0x007FFFFF:
        return "nan"
    return "negative_infinity" if word & 0x80000000 else "positive_infinity"


def snapshot(tensor, *, layout=False):
    if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
        raise ValueError("Fixtures must contain only CPU float32 tensors")
    words = [
        word & 0xFFFFFFFF
        for word in tensor.detach().contiguous().view(torch.int32).reshape(-1).tolist()
    ]
    encoded = {
        "shape": list(tensor.shape),
        "f32_bits": [f"{word:08x}" for word in words],
    }
    nonfinite = [
        {"index": index, "classification": classification}
        for index, word in enumerate(words)
        if (classification := classify(word)) is not None
    ]
    if nonfinite:
        encoded["nonfinite"] = nonfinite
    if layout:
        encoded["stride"] = list(tensor.stride())
        encoded["storage_offset"] = tensor.storage_offset()
    return encoded


def case(case_id, op, inputs, result):
    return {
        "id": case_id,
        "op": op,
        "inputs": {name: snapshot(tensor) for name, tensor in inputs.items()},
        "outputs": {"result": snapshot(result)},
    }


def build_fixtures(identity):
    verify_identity(identity)
    scalar = f32(3.5)
    empty = f32([])
    empty_rows = empty.reshape(0, 3)
    empty_cols = empty.reshape(3, 0)
    singleton = f32([7])
    row = f32([[10, 20, 30]])
    column = f32([[10], [20], [30]])
    broadcast_column = f32([[1], [2]])
    a = f32([[1, 2, 3], [4, 5, 6]])
    b = f32([[7, 8], [9, 10], [11, 12]])
    bias = f32([1, -1])
    sum_input = f32([[59, 63], [140, 153]])
    exact = from_bits(
        [
            0x00000000,
            0x80000000,
            0x00000001,
            0x007FFFFF,
            0x00800000,
            0x7F7FFFFF,
            0x3EAAAAAB,
        ]
    )
    finite = [
        case("scalar_add", "add", {"a": scalar, "b": f32(2)}, scalar + f32(2)),
        case("scalar_sum", "sum", {"a": scalar}, scalar.sum()),
        case("empty_sum", "sum", {"a": empty}, empty.sum()),
        case(
            "empty_add_singleton",
            "add",
            {"a": empty, "b": singleton},
            empty + singleton,
        ),
        case(
            "empty_rows_add_singleton",
            "add",
            {"a": empty_rows, "b": row},
            empty_rows + row,
        ),
        case(
            "empty_cols_add_singleton",
            "add",
            {"a": empty_cols, "b": column},
            empty_cols + column,
        ),
        case(
            "broadcast_add",
            "add",
            {"a": broadcast_column, "b": row},
            broadcast_column + row,
        ),
        case("matmul_2x3_3x2", "matmul", {"a": a, "b": b}, a @ b),
        case(
            "matmul_add_bias",
            "matmul_add",
            {"a": a, "b": b, "bias": bias},
            a @ b + bias,
        ),
        case("sum_2x2", "sum", {"a": sum_input}, sum_input.sum()),
        case("f32_bit_patterns", "clone", {"a": exact}, exact.clone()),
    ]

    base = f32([[1, 2, 3], [4, 5, 6]])
    base_before = snapshot(base, layout=True)
    view = base.transpose(0, 1)
    view_before = snapshot(view, layout=True)
    value = f32(20)
    shares_storage = (
        base.untyped_storage().data_ptr() == view.untyped_storage().data_ptr()
    )
    view[1, 0] = value
    finite.append(
        {
            "id": "transpose_view_alias",
            "op": "transpose_write",
            "inputs": {"base": base_before, "value": snapshot(value)},
            "parameters": {"dims": [0, 1], "index": [1, 0]},
            "outputs": {
                "view_before": view_before,
                "base_after": snapshot(base, layout=True),
                "view_after": snapshot(view, layout=True),
            },
            "alias": {
                "shares_storage": shares_storage,
                "view_is_contiguous": view.is_contiguous(),
            },
        }
    )

    numerator = f32([0, 1, -1])
    denominator = f32([0, 0, 0])
    nonfinite_input = from_bits([0x7FC01234, 0x7F800000, 0xFF800000])
    ones = f32([1, 1, 1])
    nonfinite = [
        case("empty_mean", "mean", {"a": empty}, empty.mean()),
        case(
            "divide_by_zero",
            "div",
            {"a": numerator, "b": denominator},
            numerator / denominator,
        ),
        case(
            "nonfinite_add",
            "add",
            {"a": nonfinite_input, "b": ones},
            nonfinite_input + ones,
        ),
    ]
    return {
        "oracle": identity,
        "dtype": "float32",
        "device": "cpu",
        "finite_cases": finite,
        "nonfinite_cases": nonfinite,
    }


def cpu_provenance(identity):
    cpu_model = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    return {
        "oracle": identity,
        "python_version": platform.python_version(),
        "uv_version": subprocess.check_output(["uv", "--version"], text=True).strip(),
        "lock_sha256": hashlib.sha256(
            (ORACLE_DIR / "uv.lock").read_bytes()
        ).hexdigest(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
            "cpu_model": cpu_model,
        },
        "cpu": {
            "capability": torch.backends.cpu.get_cpu_capability(),
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "default_dtype": str(torch.get_default_dtype()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
        },
        "torch_config": torch.__config__.show(),
        "torch_parallel_info": torch.__config__.parallel_info(),
    }


def json_bytes(document):
    return (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ORACLE_DIR / "fixtures")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    identity = observed_identity()
    print(json.dumps(identity, indent=2, allow_nan=False))
    verify_identity(identity)
    if args.identity_only:
        return
    configure_cpu()
    fixture = args.output_dir / "s0.json"
    data = json_bytes(build_fixtures(identity))
    if args.check:
        if not fixture.exists() or fixture.read_bytes() != data:
            raise SystemExit(f"Frozen oracle differs: {fixture}")
        print(f"Verified {fixture.name}")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture.write_bytes(data)
    config = args.output_dir / "cpu_config.json"
    config.write_bytes(json_bytes(cpu_provenance(identity)))
    print(f"Wrote {fixture.name} and {config.name}")


if __name__ == "__main__":
    main()
