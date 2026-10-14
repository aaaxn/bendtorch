import argparse
import hashlib
import json
import struct
from pathlib import Path

import generate
from generate import Json, encoded, from_values, make_case
from ordered_reference import OPERATIONS, ordered_result

POLICY_V3 = generate.S2_DIR / "numerical-policy-v3.json"
POLICY_V3_SHA256 = "e3b1bd0807b1e6f686a57574c64f961fc9de6fedec97f0a74a3bc2142add001f"
BASE_CORPUS = generate.S2_DIR / "fixtures/s2-v2.json"
OUTPUT = generate.S2_DIR / "fixtures/s2-supplement-v1.json"
NEGATIVE_ZERO = 0x80000000
LARGEST = 0x7F7FFFFF


def word(value: float) -> int:
    return struct.unpack(">I", struct.pack(">f", value))[0]


def policy_v3() -> Json:
    data = POLICY_V3.read_bytes()
    if hashlib.sha256(data).hexdigest() != POLICY_V3_SHA256:
        raise RuntimeError("Numerical policy v3 changed; version and review required")
    return json.loads(data)


def matmul_add_cases() -> list[Json]:
    a = from_values([2, 3], [0.5, -1.25, 2.0, 3.0, 0.75, -2.5])
    b = from_values([3, 2], [1.5, -0.5, 2.25, 1.0, -1.0, 0.125])
    biases = {
        "scalar_bias": from_values([], [0.1]),
        "row_bias": from_values([1, 2], [0.3, -0.7]),
        "column_bias": from_values([2, 1], [-0.2, 1.1]),
        "full_bias": from_values([2, 2], [0.1, -0.1, 1e-3, -1e3]),
        "vector_rounded_bias": from_values([2], [1 / 3, -2 / 3]),
    }
    cases = [
        make_case(
            f"matmul_add_{name}",
            "matmul_add",
            {"a": a, "b": b, "bias": bias},
            "reduction_envelope",
        )
        for name, bias in biases.items()
    ]
    cases.append(
        make_case(
            "matmul_add_cancellation",
            "matmul_add",
            {
                "a": from_values([1, 2], [2**24, 1]),
                "b": from_values([2, 1], [1, 1]),
                "bias": from_values([], [-(2**24)]),
            },
            "reduction_envelope",
        )
    )
    cases.append(
        make_case(
            "matmul_add_zero_inner",
            "matmul_add",
            {
                "a": from_values([2, 0], []),
                "b": from_values([0, 3], []),
                "bias": from_values([3], [1.5, -0.25, 0.0]),
            },
            "reduction_envelope",
        )
    )
    cases.append(
        make_case(
            "matmul_add_negative_zero_bias",
            "matmul_add",
            {
                "a": encoded([1, 1], [NEGATIVE_ZERO]),
                "b": from_values([1, 1], [1.0]),
                "bias": encoded([1, 1], [NEGATIVE_ZERO]),
            },
            "reduction_envelope",
        )
    )
    cases.append(
        make_case(
            "matmul_add_exact_zero_negative_bias",
            "matmul_add",
            {
                "a": from_values([1, 2], [1.0, -1.0]),
                "b": from_values([2, 1], [1.0, 1.0]),
                "bias": encoded([], [NEGATIVE_ZERO]),
            },
            "reduction_envelope",
        )
    )
    return cases


def matmul_cases() -> list[Json]:
    return [
        make_case(
            "matmul_negative_zero_products",
            "matmul",
            {
                "a": encoded([1, 2], [NEGATIVE_ZERO, NEGATIVE_ZERO]),
                "b": from_values([2, 1], [1.0, 1.0]),
            },
            "reduction_envelope",
        ),
        make_case(
            "matmul_underflow_negative_product",
            "matmul",
            {
                "a": from_values([1, 1], [-1e-30]),
                "b": from_values([1, 1], [1e-30]),
            },
            "reduction_envelope",
        ),
        make_case(
            "sum_negative_zeros",
            "sum",
            {"a": encoded([3], [NEGATIVE_ZERO] * 3)},
            "reduction_envelope",
        ),
    ]


def get_cases() -> list[Json]:
    special = encoded([2, 2], [NEGATIVE_ZERO, 0x00000001, 0xFF800000, 0x7FC01234])
    cases = [
        make_case(
            f"get_special_{row}_{column}",
            "get",
            {"a": special},
            "exact_words",
            {"coordinate": [row, column]},
        )
        for row in range(2)
        for column in range(2)
    ]
    cases.append(
        make_case(
            "get_scalar_negative_zero",
            "get",
            {"a": encoded([], [NEGATIVE_ZERO])},
            "exact_words",
            {"coordinate": []},
        )
    )
    return cases


def elementwise_cases() -> list[Json]:
    cases = []
    pairs = {
        "scalar_left": ([], [3]),
        "vector_left": ([3], [2, 3]),
        "column_left": ([2, 1], [3, 2, 4]),
    }
    for name, (left, right) in pairs.items():
        count = 1
        for size in left:
            count *= size
        a = from_values(left, [1.25 - 0.5 * index for index in range(count)])
        count = 1
        for size in right:
            count *= size
        b = from_values(right, [0.375 * index - 2.0 for index in range(count)])
        for op in ("add", "sub", "mul"):
            cases.append(
                make_case(f"{op}_{name}", op, {"a": a, "b": b}, "bounded_elementwise")
            )
    negative = 0x80000000 | LARGEST
    for op, left, right in (
        ("add", negative, negative),
        ("sub", negative, LARGEST),
        ("mul", LARGEST, word(-2.0)),
    ):
        cases.append(
            make_case(
                f"ieee_{op}_negative_overflow",
                op,
                {"a": encoded([1], [left]), "b": encoded([1], [right])},
                "ieee_edges",
            )
        )
    return cases


def error_cases() -> list[Json]:
    return [
        {
            "id": "matmul_add_incompatible_bias",
            "op": "matmul_add",
            "request": {"shape": [2, 3], "b_shape": [3, 2], "bias_shape": [3]},
            "expected_error": "IncompatibleBroadcast",
        },
        {
            "id": "matmul_add_bias_inner_first",
            "op": "matmul_add",
            "request": {"shape": [2, 3], "b_shape": [4, 2], "bias_shape": [3]},
            "expected_error": "MatmulInnerMismatch",
        },
    ]


def v3_violation(case: Json) -> str | None:
    observed = case["outputs"]["result"]["f32_bits"]
    reference = case["independent_reference"]["result"]["f32_bits"]
    policy = case["comparison_policy"]
    for index, (torch_word, ideal) in enumerate(zip(observed, reference, strict=True)):
        if policy == "bounded_elementwise" and torch_word != ideal:
            return f"{case['id']}[{index}]: PyTorch is not the RN word"
    if policy == "reduction_envelope":
        for index, (torch_word, ordered) in enumerate(
            zip(observed, case["ordered_reference"]["f32_bits"], strict=True)
        ):
            if (
                torch_word in ("00000000", "80000000")
                and ordered
                in (
                    "00000000",
                    "80000000",
                )
                and torch_word != ordered
            ):
                return f"{case['id']}[{index}]: ordered zero sign differs from PyTorch"
    return None


def build(identity: Json) -> Json:
    v3 = policy_v3()
    cases = matmul_add_cases() + matmul_cases() + get_cases() + elementwise_cases()
    for case in cases:
        generate.validate_observation(case)
        if case["op"] in OPERATIONS and case["comparison_policy"] != "ieee_edges":
            case["ordered_reference"] = ordered_result(case)
        if violation := v3_violation(case):
            raise RuntimeError(violation)
    if len({case["id"] for case in cases + error_cases()}) != len(cases) + 2:
        raise RuntimeError("Duplicate supplement case identifier")
    return {
        "schema_version": 1,
        "corpus_version": "s2-supplement-v1",
        "policy_version": v3["policy_version"],
        "policy_sha256": POLICY_V3_SHA256,
        "base_corpus": {
            "file": BASE_CORPUS.name,
            "sha256": hashlib.sha256(BASE_CORPUS.read_bytes()).hexdigest(),
            "relation": "additional cases only; no base case is changed or removed",
        },
        "oracle": identity,
        "dtype": "float32",
        "device": "cpu",
        "cases": cases,
        "error_cases": error_cases(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    identity = generate.observed_identity()
    generate.verify_identity(identity)
    generate.policy()
    generate.configure_cpu()
    data = generate.json_bytes(build(identity))
    output = args.output.resolve()
    if not output.is_relative_to(generate.S2_DIR):
        raise SystemExit("Output must remain under oracle/s2/")
    if args.check:
        if not output.exists() or output.read_bytes() != data:
            raise SystemExit(f"Frozen supplement differs: {output}")
        print(f"Verified {output.name}: {hashlib.sha256(data).hexdigest()}")
        return
    if output.exists():
        raise SystemExit("Generation refuses to overwrite an existing supplement")
    output.write_bytes(data)
    print(f"Wrote {output.name}: {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
