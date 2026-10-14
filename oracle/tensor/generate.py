import argparse
import hashlib
import itertools
import json
import math
import platform
import random
import struct
import subprocess
import sys
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
import torch.__config__
from transpose_reference import requested_axes, valid_axes

TENSOR_DIR = Path(__file__).resolve().parent
ORACLE_DIR = TENSOR_DIR.parent
POLICY_PATH = TENSOR_DIR / "numerical-policy.json"
POLICY_SHA256 = "0a0e28b2e4666aa9a859af9a091cf0a2cad93ac10f7bb4c68413f1250853d9a4"
EXPECTED_IDENTITY = {
    "torch_version": "2.14.1+cpu",
    "torch_git_version": "5c4886908584029761b579af026dcfb627c84070",
    "cuda_version": None,
    "hip_version": None,
}
SEED = 0x53320001
U = Fraction(1, 1 << 24)
ETA = Fraction(1, 1 << 150)
Json = dict[str, Any]


def json_bytes(document: Any) -> bytes:
    return (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def policy() -> Json:
    data = POLICY_PATH.read_bytes()
    if hashlib.sha256(data).hexdigest() != POLICY_SHA256:
        raise RuntimeError(
            "Frozen numerical policy changed; version and review required"
        )
    return json.loads(data)


def observed_identity() -> Json:
    return {
        "torch_version": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "cuda_version": torch.version.cuda,
        "hip_version": torch.version.hip,
    }


def verify_identity(identity: Json) -> None:
    if identity != EXPECTED_IDENTITY:
        raise RuntimeError(f"Oracle identity mismatch: {identity!r}")


def configure_cpu() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    if not torch.set_flush_denormal(False):
        raise RuntimeError("This backend cannot explicitly disable denormal flushing")


def classification(word: int) -> str:
    sign = "negative" if word & 0x80000000 else "positive"
    exponent = word & 0x7F800000
    fraction = word & 0x007FFFFF
    if exponent == 0x7F800000:
        return "nan" if fraction else f"{sign}_infinity"
    if exponent == 0:
        return f"{sign}_subnormal" if fraction else f"{sign}_zero"
    return f"{sign}_normal"


def encoded(shape: list[int], words: list[int]) -> Json:
    if len(words) != math.prod(shape):
        raise ValueError("Word count does not equal logical element count")
    result: Json = {
        "shape": shape,
        "f32_bits": [f"{word:08x}" for word in words],
        "classifications": [classification(word) for word in words],
    }
    nonfinite = [
        {"index": index, "classification": kind}
        for index, word in enumerate(words)
        if (kind := classification(word))
        in {"nan", "positive_infinity", "negative_infinity"}
    ]
    if nonfinite:
        result["nonfinite"] = nonfinite
    return result


def words(tensor: Json) -> list[int]:
    return [int(word, 16) for word in tensor["f32_bits"]]


def from_values(shape: list[int], values: Sequence[float]) -> Json:
    return encoded(
        shape, [struct.unpack(">I", struct.pack(">f", value))[0] for value in values]
    )


def rational(word: int) -> Fraction:
    if word & 0x7F800000 == 0x7F800000:
        raise ValueError("Nonfinite word has no exact rational value")
    exponent = (word >> 23) & 0xFF
    significand = word & 0x7FFFFF
    if exponent:
        significand += 1 << 23
    power = exponent - 150 if exponent else -149
    value = Fraction(significand) * Fraction(2) ** power
    return -value if word & 0x80000000 else value


def rounded_word(value: Fraction, *, negative_zero: bool = False) -> int:
    sign = 0x80000000 if value < 0 or (value == 0 and negative_zero) else 0
    value = abs(value)
    if value >= rational(0x7F7FFFFF) + Fraction(2) ** 103:
        return sign | 0x7F800000
    low, high = 0, 0x7F7FFFFF
    while low < high:
        middle = (low + high + 1) // 2
        if rational(middle) <= value:
            low = middle
        else:
            high = middle - 1
    if low == 0x7F7FFFFF:
        return sign | low
    lower_distance = value - rational(low)
    upper_distance = rational(low + 1) - value
    if upper_distance < lower_distance or (
        upper_distance == lower_distance and low & 1
    ):
        low += 1
    return sign | low


def ieee_binary(op: str, left: int, right: int) -> int:
    if op == "sub":
        right ^= 0x80000000
        op = "add"
    lc, rc = classification(left), classification(right)
    if "nan" in (lc, rc):
        return 0x7FC00000
    li, ri = "infinity" in lc, "infinity" in rc
    if op == "add":
        if li and ri and (left ^ right) & 0x80000000:
            return 0x7FC00000
        if li or ri:
            return left if li else right
        value = rational(left) + rational(right)
        return rounded_word(value, negative_zero=left == right == 0x80000000)
    if op != "mul":
        raise ValueError(f"Unknown binary operation {op}")
    sign = (left ^ right) & 0x80000000
    if (li and right & 0x7FFFFFFF == 0) or (ri and left & 0x7FFFFFFF == 0):
        return 0x7FC00000
    if li or ri:
        return sign | 0x7F800000
    return rounded_word(rational(left) * rational(right), negative_zero=bool(sign))


def fraction_json(value: Fraction) -> Json:
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def coordinates(shape: list[int]) -> Any:
    return itertools.product(*(range(dimension) for dimension in shape))


def flat_index(shape: list[int], coordinate: Any) -> int:
    index = 0
    for dimension, component in zip(shape, coordinate, strict=True):
        index = index * dimension + component
    return index


def broadcast_shape(left: list[int], right: list[int]) -> list[int]:
    rank = max(len(left), len(right))
    a, b = [1] * (rank - len(left)) + left, [1] * (rank - len(right)) + right
    result = []
    for x, y in zip(a, b, strict=True):
        if x == y or y == 1:
            result.append(x)
        elif x == 1:
            result.append(y)
        else:
            raise ValueError("IncompatibleBroadcast")
    return result


def projected_word(tensor: Json, coordinate: Any) -> int:
    shape = tensor["shape"]
    aligned = coordinate[len(coordinate) - len(shape) :]
    projected = [
        0 if dimension == 1 else index
        for dimension, index in zip(shape, aligned, strict=True)
    ]
    return words(tensor)[flat_index(shape, projected)]


def envelope(terms: list[Fraction], n: int) -> Json:
    if n * U >= 1:
        raise ValueError("Accumulation length outside frozen envelope")
    exact = sum(terms, Fraction())
    absolute_sum = sum(map(abs, terms), Fraction())
    if absolute_sum >= Fraction(2) ** 120:
        raise ValueError("Potential intermediate overflow outside frozen envelope")
    bound = (n * U * absolute_sum + n * ETA) / (1 - n * U)
    condition: Any = (
        fraction_json(absolute_sum / abs(exact))
        if exact
        else ("infinite" if absolute_sum else "zero")
    )
    return {
        "exact": fraction_json(exact),
        "abs_sum": fraction_json(absolute_sum),
        "rounding_count": n,
        "absolute_error_bound": fraction_json(bound),
        "condition_number": condition,
    }


def independent_reference(
    op: str, inputs: Json, parameters: Json, comparison: str
) -> Json:
    a = inputs["a"]
    aw = words(a)
    shape = a["shape"]
    exact_values: list[Json] = []
    envelopes: list[Json] = []
    if op in {"construct", "clone"}:
        result = encoded(shape, aw)
    elif op == "get":
        result = encoded([], [aw[flat_index(shape, parameters["coordinate"])]])
    elif op == "reshape":
        result = encoded(parameters["shape"], aw)
    elif op in {"transpose_copy", "permute_copy"}:
        axes = parameters.get("axes", list(range(len(shape))))[:]
        if op == "transpose_copy":
            x, y = parameters["dims"]
            axes = requested_axes(len(shape), x, y)
        out_shape = [shape[axis] for axis in axes]
        mapped = []
        for coordinate in coordinates(out_shape):
            source = [0] * len(shape)
            for out_axis, source_axis in enumerate(axes):
                source[source_axis] = coordinate[out_axis]
            mapped.append(aw[flat_index(shape, source)])
        result = encoded(out_shape, mapped)
    elif op in {"add", "sub", "mul"}:
        out_shape = broadcast_shape(shape, inputs["b"]["shape"])
        mapped = []
        for coordinate in coordinates(out_shape):
            left, right = (
                projected_word(a, coordinate),
                projected_word(inputs["b"], coordinate),
            )
            mapped.append(ieee_binary(op, left, right))
            if comparison == "bounded_elementwise":
                x, y = rational(left), rational(right)
                if abs(x) > 16 or abs(y) > 16:
                    raise ValueError("Input outside bounded elementwise domain")
                exact_values.append(
                    fraction_json(
                        x + y if op == "add" else x - y if op == "sub" else x * y
                    )
                )
        result = encoded(out_shape, mapped)
    elif op in {"sum", "matmul", "matmul_add"}:
        if op == "sum":
            out_shape = []
            dot_terms = [[(word, 0x3F800000) for word in aw]]
            k = len(aw)
        else:
            m, k = shape
            bk, n = inputs["b"]["shape"]
            if bk != k:
                raise ValueError("MatmulInnerMismatch")
            bw = words(inputs["b"])
            out_shape = [m, n]
            dot_terms = [
                [(aw[row * k + inner], bw[inner * n + column]) for inner in range(k)]
                for row in range(m)
                for column in range(n)
            ]
        mapped = []
        for index, products in enumerate(dot_terms):
            if comparison == "ieee_edges":
                accumulator = 0
                for left, right in products:
                    term = left if op == "sum" else ieee_binary("mul", left, right)
                    accumulator = ieee_binary("add", accumulator, term)
                mapped.append(accumulator)
                continue
            terms = [rational(left) * rational(right) for left, right in products]
            count = k if op == "sum" else 2 * k
            if op == "matmul_add":
                coordinate = (index // out_shape[1], index % out_shape[1])
                terms.append(rational(projected_word(inputs["bias"], coordinate)))
                count += 1
            detail = envelope(terms, count)
            envelopes.append(detail)
            exact = sum(terms, Fraction())
            exact_values.append(fraction_json(exact))
            mapped.append(rounded_word(exact))
        result = encoded(out_shape, mapped)
    else:
        raise ValueError(f"Unknown operation {op}")
    reference: Json = {
        "result": result,
        "authority": "independent raw-word/index/IEEE/rational model",
    }
    if exact_values:
        reference["exact_values"] = exact_values
    if envelopes:
        reference["envelopes"] = envelopes
    return reference


def torch_tensor(tensor: Json) -> torch.Tensor:
    signed = [
        word if word < 0x80000000 else word - 0x100000000 for word in words(tensor)
    ]
    return (
        torch.tensor(signed, dtype=torch.int32, device="cpu")
        .view(torch.float32)
        .reshape(tensor["shape"])
    )


def snapshot(tensor: torch.Tensor) -> Json:
    if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
        raise ValueError("Only CPU float32 snapshots are supported")
    raw = tensor.detach().contiguous().reshape(-1).view(torch.int32).tolist()
    return encoded(list(tensor.shape), [word & 0xFFFFFFFF for word in raw])


def torch_result(op: str, inputs: Json, parameters: Json) -> Json:
    a = torch_tensor(inputs["a"])
    if op in {"construct", "clone"}:
        result = a.clone()
    elif op == "get":
        result = a[tuple(parameters["coordinate"])]
    elif op == "reshape":
        result = a.reshape(parameters["shape"]).clone()
    elif op == "transpose_copy":
        result = a.transpose(*parameters["dims"]).contiguous().clone()
    elif op == "permute_copy":
        result = a.permute(parameters["axes"]).contiguous().clone()
    elif op == "sum":
        result = a.sum()
    elif op in {"matmul", "matmul_add"}:
        result = a @ torch_tensor(inputs["b"])
        if op == "matmul_add":
            result = result + torch_tensor(inputs["bias"])
    elif op == "add":
        result = a + torch_tensor(inputs["b"])
    elif op == "sub":
        result = a - torch_tensor(inputs["b"])
    elif op == "mul":
        result = a * torch_tensor(inputs["b"])
    else:
        raise ValueError(f"Unknown operation {op}")
    return snapshot(result)


def make_case(
    case_id: str, op: str, inputs: Json, comparison: str, parameters: Json | None = None
) -> Json:
    parameters = parameters or {}
    reference = independent_reference(op, inputs, parameters, comparison)
    result: Json = {
        "id": case_id,
        "op": op,
        "inputs": inputs,
        "outputs": {"result": torch_result(op, inputs, parameters)},
        "comparison_policy": comparison,
        "independent_reference": reference,
    }
    if parameters:
        result["parameters"] = parameters
    return result


def exact_goldens() -> Json:
    return {
        "schema_version": 1,
        "corpus_version": "s2-mathematical-goldens-v1",
        "authority": "integer multiplication and addition, not PyTorch snapshots",
        "a": [[1, 2, 3], [4, 5, 6]],
        "b": [[7, 8], [9, 10], [11, 12]],
        "bias": [1, -1],
        "ab": [[58, 64], [139, 154]],
        "ab_plus_bias": [[59, 63], [140, 153]],
        "sum_ab_plus_bias": 415,
    }


def successful_cases(*, scalar_transpose: bool = True) -> list[Json]:
    cases: list[Json] = []
    rng = random.Random(SEED)
    shapes = [
        [],
        [0],
        [1],
        [3],
        [2, 3],
        [0, 3],
        [3, 0],
        [1, 3, 1],
        [2, 1, 3, 1],
        [1, 0, 3, 1],
    ]
    for index, shape in enumerate(shapes):
        a = from_values(shape, [float(i - 3) / 2 for i in range(math.prod(shape))])
        for op in ("construct", "clone"):
            cases.append(make_case(f"{op}_shape_{index}", op, {"a": a}, "exact_words"))
        cases.append(
            make_case(f"sum_shape_{index}", "sum", {"a": a}, "reduction_envelope")
        )
    patterns = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x007FFFFF,
        0x00800000,
        0x7F7FFFFF,
        0x3EAAAAAB,
        0x7FC01234,
        0xFFC05678,
        0x7F800001,
        0x7F800000,
        0xFF800000,
    ]
    for op in ("construct", "clone"):
        cases.append(
            make_case(
                f"{op}_all_classifications",
                op,
                {"a": encoded([13], patterns)},
                "exact_words",
            )
        )
    for index, (source, target) in enumerate(
        [
            ([], [1, 1, 1, 1]),
            ([2, 3], [3, 2]),
            ([3, 2], [6]),
            ([6], [1, 3, 2]),
            ([0, 3], [1, 0, 1, 1]),
            ([1, 3, 1, 2], [2, 3]),
        ]
    ):
        a = from_values(source, [float(i + 1) for i in range(math.prod(source))])
        cases.append(
            make_case(
                f"reshape_{index}",
                "reshape",
                {"a": a},
                "exact_words",
                {"shape": target},
            )
        )
    edge_map = encoded([1, 13], patterns)
    cases.append(
        make_case(
            "reshape_all_classifications",
            "reshape",
            {"a": edge_map},
            "exact_words",
            {"shape": [13, 1]},
        )
    )
    for index, (shape, dims) in enumerate(
        [
            ([2, 3], [0, 1]),
            ([3, 1], [0, 1]),
            ([0, 3], [0, 1]),
            ([2, 1, 3, 1], [0, 2]),
            ([1], [0, 0]),
        ]
    ):
        a = from_values(shape, [float(i + 1) for i in range(math.prod(shape))])
        cases.append(
            make_case(
                f"transpose_{index}",
                "transpose_copy",
                {"a": a},
                "exact_words",
                {"dims": dims},
            )
        )
    if scalar_transpose:
        cases.append(
            make_case(
                "transpose_scalar",
                "transpose_copy",
                {"a": encoded([], [0x40E00000])},
                "exact_words",
                {"dims": [0, 0]},
            )
        )
    cases.append(
        make_case(
            "transpose_all_classifications",
            "transpose_copy",
            {"a": edge_map},
            "exact_words",
            {"dims": [0, 1]},
        )
    )
    for index, (shape, axes) in enumerate(
        [
            ([], []),
            ([3], [0]),
            ([2, 3], [1, 0]),
            ([2, 1, 3], [2, 0, 1]),
            ([2, 1, 3, 1], [3, 2, 0, 1]),
            ([2, 0, 3, 1], [2, 0, 3, 1]),
        ]
    ):
        a = from_values(shape, [float(i + 1) for i in range(math.prod(shape))])
        cases.append(
            make_case(
                f"permute_{index}",
                "permute_copy",
                {"a": a},
                "exact_words",
                {"axes": axes},
            )
        )
    cases.append(
        make_case(
            "permute_all_classifications",
            "permute_copy",
            {"a": edge_map},
            "exact_words",
            {"axes": [1, 0]},
        )
    )
    for index, (shape, coordinate) in enumerate(
        [([], []), ([3], [2]), ([2, 3], [1, 2]), ([2, 1, 3, 1], [1, 0, 2, 0])]
    ):
        a = from_values(shape, [float(i + 1) for i in range(math.prod(shape))])
        cases.append(
            make_case(
                f"get_{index}",
                "get",
                {"a": a},
                "exact_words",
                {"coordinate": coordinate},
            )
        )
    pairs = [
        ([], []),
        ([3], []),
        ([2, 1], [1, 3]),
        ([2, 1, 3, 1], [1, 3]),
        ([0], [1]),
        ([0, 3], [1, 3]),
        ([3, 0], [3, 1]),
        ([1, 0, 3, 1], [3, 1]),
        ([0, 3], [0, 1]),
        ([1], [0]),
        ([1, 3], [0, 3]),
        ([3, 1], [3, 0]),
    ]
    for index, (left_shape, right_shape) in enumerate(pairs):
        a = from_values(
            left_shape, [rng.uniform(-8, 8) for _ in range(math.prod(left_shape))]
        )
        b = from_values(
            right_shape, [rng.uniform(-8, 8) for _ in range(math.prod(right_shape))]
        )
        for op in ("add", "sub", "mul"):
            cases.append(
                make_case(
                    f"{op}_broadcast_{index}",
                    op,
                    {"a": a, "b": b},
                    "bounded_elementwise",
                )
            )
    for trial in range(8):
        shape = [1, 3, 7] if trial % 2 else [3, 5]
        a = from_values(shape, [rng.uniform(-16, 16) for _ in range(math.prod(shape))])
        for op in ("sum",):
            cases.append(
                make_case(f"random_{op}_{trial}", op, {"a": a}, "reduction_envelope")
            )
        for op in ("add", "sub", "mul"):
            b = from_values(
                shape, [rng.uniform(-16, 16) for _ in range(math.prod(shape))]
            )
            cases.append(
                make_case(
                    f"random_{op}_{trial}", op, {"a": a, "b": b}, "bounded_elementwise"
                )
            )
    for trial, values in enumerate(
        [
            [2**24, 1, -(2**24)],
            [2**24, -(2**24), 1],
            [1e10, 3, -1e10, -2, 0.25],
            [16, -16, 8, -8, 0],
            [2**-120, -(2**-120), 2**-149],
        ]
    ):
        cases.append(
            make_case(
                f"sum_cancellation_{trial}",
                "sum",
                {"a": from_values([len(values)], values)},
                "reduction_envelope",
            )
        )
    for length in (31, 127, 257):
        values = [rng.uniform(-8, 8) for _ in range(length)]
        cases.append(
            make_case(
                f"sum_random_length_{length}",
                "sum",
                {"a": from_values([length], values)},
                "reduction_envelope",
            )
        )
    for index, (m, k, n) in enumerate(
        [
            (1, 1, 1),
            (2, 3, 5),
            (3, 5, 2),
            (1, 7, 3),
            (0, 3, 2),
            (2, 3, 0),
            (2, 0, 3),
            (0, 0, 3),
            (2, 31, 3),
        ]
    ):
        a = from_values([m, k], [rng.uniform(-4, 4) for _ in range(m * k)])
        b = from_values([k, n], [rng.uniform(-4, 4) for _ in range(k * n)])
        cases.append(
            make_case(
                f"matmul_rectangular_{index}",
                "matmul",
                {"a": a, "b": b},
                "reduction_envelope",
            )
        )
    a = from_values([2, 3], [1, 2, 3, 4, 5, 6])
    b = from_values([3, 2], [7, 8, 9, 10, 11, 12])
    cases.append(make_case("golden_ab", "matmul", {"a": a, "b": b}, "exact_golden"))
    cases.append(
        make_case(
            "golden_ab_plus_bias",
            "matmul_add",
            {"a": a, "b": b, "bias": from_values([2], [1, -1])},
            "exact_golden",
        )
    )
    cases.append(
        make_case(
            "golden_sum_415",
            "sum",
            {"a": from_values([2, 2], [59, 63, 140, 153])},
            "exact_golden",
        )
    )
    a = from_values([2, 3], [2**24, 1, -(2**24), 1e10, 3, -1e10])
    b = from_values([3, 2], [1, 1, 1, -1, 1, 1])
    cases.append(
        make_case(
            "matmul_cancellation", "matmul", {"a": a, "b": b}, "reduction_envelope"
        )
    )
    a = encoded(
        [9],
        [
            0,
            0x80000000,
            1,
            0x80000001,
            0x007FFFFF,
            0x00800000,
            0x7F7FFFFF,
            0x7FC01234,
            0x7F800000,
        ],
    )
    for op, bw in [
        (
            "add",
            [
                0x80000000,
                0x80000000,
                1,
                0x80000001,
                1,
                0x80000001,
                0x7F7FFFFF,
                0x3F800000,
                0xFF800000,
            ],
        ),
        ("sub", [0, 0, 1, 1, 1, 1, 0xFF7FFFFF, 0x3F800000, 0x7F800000]),
        (
            "mul",
            [
                0xBF800000,
                0xBF800000,
                0x3F000000,
                0x3F000000,
                0x3F000000,
                0x3F000000,
                0x40000000,
                0x3F800000,
                0,
            ],
        ),
    ]:
        cases.append(
            make_case(f"ieee_{op}", op, {"a": a, "b": encoded([9], bw)}, "ieee_edges")
        )
    for index, raw in enumerate(
        [
            [0x7FC01234, 0x3F800000],
            [0x7F800000, 0xFF800000],
            [0x7F800000, 0x3F800000],
            [0xFF800000, 0xBF800000],
            [0x7F7FFFFF, 0x7F7FFFFF],
            [0x80000000, 0x80000000],
            [1, 1, 1],
        ]
    ):
        cases.append(
            make_case(
                f"ieee_sum_{index}",
                "sum",
                {"a": encoded([len(raw)], raw)},
                "ieee_edges",
            )
        )
    for index, (aw, bw) in enumerate(
        [
            ([0x7FC01234], [0x3F800000]),
            ([0x7F800000], [0]),
            ([0x7F800000], [0xBF800000]),
            ([0x7F7FFFFF], [0x40000000]),
            ([1], [0x3F800000]),
        ]
    ):
        cases.append(
            make_case(
                f"ieee_matmul_{index}",
                "matmul",
                {"a": encoded([1, 1], aw), "b": encoded([1, 1], bw)},
                "ieee_edges",
            )
        )
    return cases


def error_cases(*, scalar_transpose: bool = True) -> list[Json]:
    specifications: list[tuple[str, str, Json, str]] = [
        ("rank5", "construct", {"shape": [1, 1, 1, 1, 1]}, "RankExceeded"),
        (
            "rank_before_capacity",
            "construct",
            {"shape": [1048577, 1, 1, 1, 1]},
            "RankExceeded",
        ),
        ("capacity_1d", "construct", {"shape": [1048577]}, "CapacityExceeded"),
        ("capacity_product", "construct", {"shape": [1025, 1024]}, "CapacityExceeded"),
        (
            "capacity_u32_product",
            "construct",
            {"shape": [4294967295, 4294967295]},
            "CapacityExceeded",
        ),
        ("negative_dimension", "construct", {"shape": [-1]}, "U32Range"),
        ("over_u32_dimension", "construct", {"shape": [4294967296]}, "U32Range"),
        (
            "data_too_short",
            "construct",
            {"shape": [3], "f32_bits": ["3f800000"]},
            "DataLengthMismatch",
        ),
        (
            "data_too_long",
            "construct",
            {"shape": [0], "f32_bits": ["3f800000"]},
            "DataLengthMismatch",
        ),
        (
            "coordinate_rank",
            "get",
            {"shape": [2, 3], "coordinate": [0]},
            "CoordinateRankMismatch",
        ),
        (
            "coordinate_rank_before_bounds",
            "get",
            {"shape": [2, 3], "coordinate": [9]},
            "CoordinateRankMismatch",
        ),
        (
            "coordinate_oob",
            "get",
            {"shape": [2, 3], "coordinate": [2, 0]},
            "CoordinateOutOfBounds",
        ),
        (
            "empty_coordinate",
            "get",
            {"shape": [0, 3], "coordinate": [0, 0]},
            "CoordinateOutOfBounds",
        ),
        (
            "negative_coordinate",
            "get",
            {"shape": [2, 3], "coordinate": [-1, 0]},
            "U32Range",
        ),
        (
            "flat_index_end",
            "get_flat",
            {"shape": [2, 3], "index": 6},
            "FlatIndexOutOfBounds",
        ),
        (
            "flat_index_empty",
            "get_flat",
            {"shape": [0], "index": 0},
            "FlatIndexOutOfBounds",
        ),
        (
            "reshape_count",
            "reshape",
            {"shape": [2, 3], "target_shape": [5]},
            "ReshapeElementCountMismatch",
        ),
        (
            "reshape_rank",
            "reshape",
            {"shape": [1], "target_shape": [1, 1, 1, 1, 1]},
            "RankExceeded",
        ),
        (
            "reshape_capacity",
            "reshape",
            {"shape": [1], "target_shape": [1048577]},
            "CapacityExceeded",
        ),
        (
            "transpose_axis",
            "transpose_copy",
            {"shape": [2, 3], "dims": [0, 2]},
            "AxisOutOfBounds",
        ),
        (
            "transpose_negative_axis",
            "transpose_copy",
            {"shape": [2, 3], "dims": [-1, 0]},
            "U32Range",
        ),
        (
            "permute_duplicate",
            "permute_copy",
            {"shape": [2, 3], "axes": [0, 0]},
            "InvalidPermutation",
        ),
        (
            "permute_length",
            "permute_copy",
            {"shape": [2, 3], "axes": [0]},
            "InvalidPermutation",
        ),
        (
            "permute_axis",
            "permute_copy",
            {"shape": [2, 3], "axes": [0, 2]},
            "AxisOutOfBounds",
        ),
        (
            "broadcast_mismatch",
            "add",
            {"shape": [2, 3], "b_shape": [4, 3]},
            "IncompatibleBroadcast",
        ),
        (
            "broadcast_zero_nonone",
            "sub",
            {"shape": [0, 3], "b_shape": [2, 3]},
            "IncompatibleBroadcast",
        ),
        (
            "broadcast_mul_mismatch",
            "mul",
            {"shape": [3], "b_shape": [2]},
            "IncompatibleBroadcast",
        ),
        (
            "broadcast_output_capacity",
            "add",
            {"shape": [1024, 1], "b_shape": [1, 1025]},
            "CapacityExceeded",
        ),
        (
            "matmul_rank_left",
            "matmul",
            {"shape": [3], "b_shape": [3, 2]},
            "MatmulRankMismatch",
        ),
        (
            "matmul_rank_right",
            "matmul",
            {"shape": [2, 3], "b_shape": [3]},
            "MatmulRankMismatch",
        ),
        (
            "matmul_batched_unsupported",
            "matmul",
            {"shape": [1, 2, 3], "b_shape": [1, 3, 2]},
            "MatmulRankMismatch",
        ),
        (
            "matmul_inner",
            "matmul",
            {"shape": [2, 3], "b_shape": [4, 2]},
            "MatmulInnerMismatch",
        ),
        (
            "matmul_output_capacity",
            "matmul",
            {"shape": [1024, 0], "b_shape": [0, 1025]},
            "CapacityExceeded",
        ),
    ]
    for dtype in (
        "float64",
        "float16",
        "bfloat16",
        "int32",
        "int64",
        "bool",
        "complex64",
    ):
        specifications.append(
            (
                f"dtype_{dtype}",
                "construct",
                {"shape": [1], "dtype": dtype},
                "UnsupportedDtype",
            )
        )
    if scalar_transpose:
        for name, dims in (
            ("axis_a", [1, 0]),
            ("axis_b", [0, 1]),
            ("axis_both", [1, 1]),
            ("u32_a", [4294967295, 0]),
            ("u32_b", [0, 4294967295]),
        ):
            specifications.append(
                (
                    f"transpose_scalar_{name}",
                    "transpose_copy",
                    {"shape": [], "dims": dims},
                    "AxisOutOfBounds",
                )
            )
    else:
        specifications.insert(
            next(
                i
                for i, specification in enumerate(specifications)
                if specification[0] == "transpose_negative_axis"
            ),
            (
                "transpose_scalar",
                "transpose_copy",
                {"shape": [], "dims": [0, 0]},
                "AxisOutOfBounds",
            ),
        )
    return [
        {
            "id": case_id,
            "op": op,
            "request": request,
            "expected_error": category,
            "comparison_policy": "expected_errors",
            "authority": "S2 bounded API specification, not PyTorch error text",
            "boundary": "dynamic_tag_or_compile_negative"
            if category == "UnsupportedDtype"
            else "checked_dynamic",
        }
        for case_id, op, request, category in specifications
    ]


def shape_cases() -> list[Json]:
    shapes = [
        [],
        [0],
        [1],
        [1048576],
        [1024, 1024],
        [4294967295, 4294967295, 4294967295, 0],
        [0, 4294967295, 4294967295, 4294967295],
    ]
    return [
        {
            "id": f"shape_boundary_{index}",
            "op": "shape_construct",
            "request": {"shape": shape},
            "expected": {"shape": shape, "rank": len(shape), "numel": math.prod(shape)},
            "authority": (
                "independent integer shape arithmetic; "
                "no tensor allocation or PyTorch requirement"
            ),
        }
        for index, shape in enumerate(shapes)
    ]


def is_u32(value: object) -> bool:
    return type(value) is int and 0 <= value <= 0xFFFFFFFF


def shape_error(shape: list[int]) -> str | None:
    if any(not is_u32(dimension) for dimension in shape):
        return "U32Range"
    if len(shape) > 4:
        return "RankExceeded"
    if math.prod(shape) > 1048576:
        return "CapacityExceeded"
    return None


def specified_error(case: Json, *, scalar_transpose: bool = True) -> str | None:
    request = case["request"]
    op = case["op"]
    shape = request["shape"]
    if any(not is_u32(word) for word in shape):
        return "U32Range"
    if request.get("dtype", "float32") != "float32":
        return "UnsupportedDtype"
    if "index" in request and not is_u32(request["index"]):
        return "U32Range"
    for key in ("coordinate", "dims", "axes", "target_shape", "b_shape"):
        if any(not is_u32(word) for word in request.get(key, [])):
            return "U32Range"
    if error := shape_error(shape):
        return error
    if "f32_bits" in request and len(request["f32_bits"]) != math.prod(shape):
        return "DataLengthMismatch"
    if op == "get":
        coordinate = request["coordinate"]
        if len(coordinate) != len(shape):
            return "CoordinateRankMismatch"
        if any(
            index >= dimension
            for dimension, index in zip(shape, coordinate, strict=True)
        ):
            return "CoordinateOutOfBounds"
    elif op == "get_flat":
        if request["index"] >= math.prod(shape):
            return "FlatIndexOutOfBounds"
    elif op == "reshape":
        target = request["target_shape"]
        if error := shape_error(target):
            return error
        if math.prod(target) != math.prod(shape):
            return "ReshapeElementCountMismatch"
    elif op == "transpose_copy":
        a, b = request["dims"]
        if not valid_axes(len(shape), a, b) or (not scalar_transpose and not shape):
            return "AxisOutOfBounds"
    elif op == "permute_copy":
        axes = request["axes"]
        if len(axes) != len(shape):
            return "InvalidPermutation"
        if any(axis >= len(shape) for axis in axes):
            return "AxisOutOfBounds"
        if sorted(axes) != list(range(len(shape))):
            return "InvalidPermutation"
    elif op in {"add", "sub", "mul", "matmul"}:
        b_shape = request["b_shape"]
        if error := shape_error(b_shape):
            return error
        if op == "matmul":
            if len(shape) != 2 or len(b_shape) != 2:
                return "MatmulRankMismatch"
            if shape[1] != b_shape[0]:
                return "MatmulInnerMismatch"
            return shape_error([shape[0], b_shape[1]])
        try:
            output = broadcast_shape(shape, b_shape)
        except ValueError:
            return "IncompatibleBroadcast"
        return shape_error(output)
    return None


def read_fraction(value: Json) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def validate_observation(case: Json) -> None:
    observed = case["outputs"]["result"]
    reference = case["independent_reference"]
    expected = reference["result"]
    comparison = case["comparison_policy"]
    if observed["shape"] != expected["shape"]:
        raise ValueError(f"{case['id']}: result shape mismatch")
    ow, ew = words(observed), words(expected)
    if len(ow) != len(ew):
        raise ValueError(f"{case['id']}: result word count mismatch")
    for index, (actual, target) in enumerate(zip(ow, ew, strict=True)):
        actual_class, target_class = classification(actual), classification(target)
        if target_class in {"nan", "positive_infinity", "negative_infinity"}:
            accepted = actual_class == target_class
        elif actual_class in {"nan", "positive_infinity", "negative_infinity"}:
            accepted = False
        elif comparison in {"exact_words", "ieee_edges", "exact_golden"}:
            accepted = actual == target
        elif comparison == "bounded_elementwise":
            exact = read_fraction(reference["exact_values"][index])
            accepted = abs(rational(actual) - exact) <= Fraction(1, 10**6) + Fraction(
                1, 10**5
            ) * abs(exact)
        elif comparison == "reduction_envelope":
            detail = reference["envelopes"][index]
            exact = read_fraction(detail["exact"])
            bound = read_fraction(detail["absolute_error_bound"])
            accepted = (
                actual == 0
                if detail["rounding_count"] == 0
                else abs(rational(actual) - exact) <= bound
            )
        else:
            raise ValueError(f"Unknown comparison policy: {comparison}")
        if not accepted:
            raise ValueError(
                f"{case['id']}: output index {index} violates {comparison}: "
                f"{actual:08x} versus independent {target:08x}"
            )


def build_fixtures(identity: Json, *, scalar_transpose: bool = True) -> Json:
    verify_identity(identity)
    frozen_policy = policy()
    all_cases = successful_cases(scalar_transpose=scalar_transpose)
    finite, edges, nonfinite = [], [], []
    for case in all_cases:
        validate_observation(case)
        tensors = list(case["inputs"].values()) + list(case["outputs"].values())
        if any("nonfinite" in tensor for tensor in tensors):
            nonfinite.append(case)
        elif case["comparison_policy"] == "ieee_edges":
            edges.append(case)
        else:
            finite.append(case)
    errors = error_cases(scalar_transpose=scalar_transpose)
    for case in errors:
        if (
            specified_error(case, scalar_transpose=scalar_transpose)
            != case["expected_error"]
        ):
            raise RuntimeError(f"Invalid error specification: {case['id']}")
    return {
        "schema_version": 1,
        "corpus_version": "s2-v2" if scalar_transpose else "s2-v1",
        "policy_version": frozen_policy["policy_version"],
        "policy_sha256": POLICY_SHA256,
        "oracle": identity,
        "dtype": "float32",
        "device": "cpu",
        "random": {
            "seed": SEED,
            "algorithm": "CPython random.Random MT19937",
            "input_generation": (
                "uniform bounded real values rounded to binary32 "
                "before all reference arithmetic"
            ),
        },
        "finite_cases": finite,
        "edge_cases": edges,
        "nonfinite_cases": nonfinite,
        "error_cases": errors,
        "shape_cases": shape_cases(),
        "implementation_results": None,
    }


def cpu_provenance(identity: Json, *, scalar_transpose: bool = True) -> Json:
    cpu_model = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    return {
        "schema_version": 1,
        "corpus_version": "s2-v2" if scalar_transpose else "s2-v1",
        "oracle": identity,
        "policy_sha256": POLICY_SHA256,
        "python_version": platform.python_version(),
        "uv_version": subprocess.check_output(["uv", "--version"], text=True).strip(),
        "oracle_lock_sha256": hashlib.sha256(
            (ORACLE_DIR / "uv.lock").read_bytes()
        ).hexdigest(),
        "root_lock_sha256": hashlib.sha256(
            (ORACLE_DIR.parent / "uv.lock").read_bytes()
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
            "flush_denormal_requested": False,
            "gradual_underflow_observed": snapshot(torch_tensor(encoded([1], [1])) * 1)[
                "f32_bits"
            ]
            == ["00000001"],
        },
        "torch_config": torch.__config__.show(),
        "torch_parallel_info": torch.__config__.parallel_info(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--historical-scalar-transpose", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=TENSOR_DIR / "fixtures")
    args = parser.parse_args()
    identity = observed_identity()
    verify_identity(identity)
    policy()
    configure_cpu()
    scalar_transpose = not args.historical_scalar_transpose
    if args.identity_only:
        print(
            json.dumps(
                cpu_provenance(identity, scalar_transpose=scalar_transpose),
                indent=2,
                allow_nan=False,
            )
        )
        return
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(TENSOR_DIR):
        raise SystemExit("Output directory must remain under oracle/tensor/")
    if not args.check and output_dir.exists():
        raise SystemExit("Generation requires a fresh --output-dir")
    corpus_name = "s2-v2.json" if scalar_transpose else "s2-v1.json"
    config_name = "cpu-config-v2.json" if scalar_transpose else "cpu-config-v1.json"
    documents = {
        corpus_name: build_fixtures(identity, scalar_transpose=scalar_transpose),
        config_name: cpu_provenance(identity, scalar_transpose=scalar_transpose),
        "mathematical-goldens-v1.json": exact_goldens(),
    }
    if not args.check:
        output_dir.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        path = output_dir / name
        data = json_bytes(document)
        if args.check:
            if not path.exists() or path.read_bytes() != data:
                raise SystemExit(f"Frozen oracle differs: {path}")
            print(f"Verified {name}: {hashlib.sha256(data).hexdigest()}")
        else:
            path.write_bytes(data)
            print(f"Wrote {name}: {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
