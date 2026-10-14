import argparse
import hashlib
import json
import math
from pathlib import Path

SIGN = 0x80000000
INFINITY = 0x7F800000
NAN = 0x7FC00000
OPERATIONS = {"sum", "matmul", "matmul_add"}
HERE = Path(__file__).resolve().parent


def classification(word: int) -> str:
    sign = "negative" if word & SIGN else "positive"
    exponent = word & INFINITY
    fraction = word & 0x7FFFFF
    if exponent == INFINITY:
        return "nan" if fraction else f"{sign}_infinity"
    if exponent == 0:
        return f"{sign}_subnormal" if fraction else f"{sign}_zero"
    return f"{sign}_normal"


def dyadic(word: int) -> tuple[int, int]:
    exponent = (word >> 23) & 0xFF
    if exponent == 0xFF:
        raise ValueError("Nonfinite words have no dyadic value")
    coefficient = word & 0x7FFFFF
    if exponent:
        coefficient |= 1 << 23
    if word & SIGN:
        coefficient = -coefficient
    return coefficient, exponent - 150 if exponent else -149


def rounded(coefficient: int, exponent: int, *, negative_zero: bool = False) -> int:
    sign = SIGN if coefficient < 0 or (coefficient == 0 and negative_zero) else 0
    coefficient = abs(coefficient)
    if coefficient == 0:
        return sign
    target = max(coefficient.bit_length() - 24 + exponent, -149)
    shift = target - exponent
    if shift > 0:
        quotient, remainder = divmod(coefficient, 1 << shift)
        half = 1 << (shift - 1)
        coefficient = quotient + (
            remainder > half or (remainder == half and quotient & 1)
        )
    else:
        coefficient <<= -shift
    if coefficient == 0:
        return sign
    power = coefficient.bit_length() - 1 + target
    if power > 127:
        return sign | INFINITY
    if power < -126:
        return sign | coefficient
    significand = coefficient >> (coefficient.bit_length() - 24)
    return sign | ((power + 127) << 23) | (significand & 0x7FFFFF)


def add(left: int, right: int) -> int:
    lc, rc = classification(left), classification(right)
    if "nan" in (lc, rc):
        return NAN
    li, ri = "infinity" in lc, "infinity" in rc
    if li and ri and (left ^ right) & SIGN:
        return NAN
    if li or ri:
        return left if li else right
    a, ae = dyadic(left)
    b, be = dyadic(right)
    exponent = min(ae, be)
    return rounded(
        (a << (ae - exponent)) + (b << (be - exponent)),
        exponent,
        negative_zero=left == right == SIGN,
    )


def multiply(left: int, right: int) -> int:
    lc, rc = classification(left), classification(right)
    if "nan" in (lc, rc):
        return NAN
    li, ri = "infinity" in lc, "infinity" in rc
    sign = (left ^ right) & SIGN
    if (li and right & ~SIGN == 0) or (ri and left & ~SIGN == 0):
        return NAN
    if li or ri:
        return sign | INFINITY
    a, ae = dyadic(left)
    b, be = dyadic(right)
    return rounded(a * b, ae + be, negative_zero=bool(sign))


def words(tensor: dict) -> list[int]:
    encoded = tensor["f32_bits"]
    if len(encoded) != math.prod(tensor["shape"]):
        raise ValueError("Frozen input word count differs from its shape")
    if any(
        len(word) != 8 or any(c not in "0123456789abcdef" for c in word)
        for word in encoded
    ):
        raise ValueError("Malformed frozen binary32 word")
    return [int(word, 16) for word in encoded]


def ordered_result(case: dict) -> dict:
    op = case["op"]
    inputs = case["inputs"]
    a = words(inputs["a"])
    result = []
    if op == "sum":
        shape = []
        accumulator = 0
        for word in a:
            accumulator = add(accumulator, word)
        result.append(accumulator)
    elif op in {"matmul", "matmul_add"}:
        m, k = inputs["a"]["shape"]
        bk, n = inputs["b"]["shape"]
        if bk != k:
            raise ValueError("Matmul inner dimension mismatch")
        shape = [m, n]
        b = words(inputs["b"])
        for row in range(m):
            for column in range(n):
                accumulator = 0
                for inner in range(k):
                    product = multiply(a[row * k + inner], b[inner * n + column])
                    accumulator = add(accumulator, product)
                result.append(accumulator)
        if op == "matmul_add":
            bias = inputs["bias"]
            bias_words = words(bias)
            bias_shape = bias["shape"]
            if len(bias_shape) > 2 or any(
                size not in (1, out)
                for size, out in zip(
                    reversed(bias_shape), reversed(shape), strict=False
                )
            ):
                raise ValueError("Bias does not broadcast to the matrix output")
            for index, accumulator in enumerate(result):
                coordinate = (index // n, index % n)[2 - len(bias_shape) :]
                bias_index = 0
                for size, item in zip(bias_shape, coordinate, strict=True):
                    bias_index = bias_index * size + (0 if size == 1 else item)
                result[index] = add(accumulator, bias_words[bias_index])
    else:
        raise ValueError(f"Not an ordered reduction operation: {op}")
    return {
        "shape": shape,
        "f32_bits": [f"{word:08x}" for word in result],
        "classifications": [classification(word) for word in result],
    }


def input_digest(case: dict) -> str:
    request = {
        "op": case["op"],
        "inputs": case["inputs"],
        "parameters": case.get("parameters", {}),
    }
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def reference_records(corpus: dict) -> dict:
    cases = [
        case
        for group in ("finite_cases", "edge_cases", "nonfinite_cases")
        for case in corpus[group]
        if case["op"] in OPERATIONS
    ]
    return {
        "contract": "s2_ordered_f32_rne",
        "arithmetic": (
            "integer dyadics, nearest-even rounding after every primitive, "
            "gradual underflow"
        ),
        "sequence": (
            "positive-zero start, logical-order sum, increasing-inner-index "
            "separate multiply/add, then bias add"
        ),
        "nan": (
            "classification only; canonical model word "
            "is not a required payload or sign"
        ),
        "cases": [
            {
                "id": case["id"],
                "input_sha256": input_digest(case),
                "result": ordered_result(case),
            }
            for case in cases
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=HERE / "fixtures/ordered-reference.json"
    )
    args = parser.parse_args()
    corpus = json.loads((HERE / "fixtures/s2-v1.json").read_text())
    document = reference_records(corpus)
    content = json.dumps(document, sort_keys=True, indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise ValueError(
                "Ordered reference differs; retain the mismatch, "
                "do not regenerate it away"
            )
    else:
        with args.output.open("x") as stream:
            stream.write(content)
    action = "checked" if args.check else "written"
    print(f"Ordered reference {action}: {len(document['cases'])} cases")


if __name__ == "__main__":
    main()
