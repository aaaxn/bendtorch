import math
import struct
from fractions import Fraction

from tools.common import GateError


def value(word: str) -> float:
    if len(word) != 8:
        raise GateError("Malformed F32 observation")
    return struct.unpack("<f", struct.pack("<I", int(word, 16)))[0]


def special(word: str) -> str | None:
    bits = int(word, 16)
    if bits & 0x7F800000 != 0x7F800000:
        return None
    if bits & 0x7FFFFF:
        return "nan"
    return "negative_infinity" if bits & 0x80000000 else "positive_infinity"


def rational(encoded: dict) -> Fraction:
    return Fraction(int(encoded["numerator"]), int(encoded["denominator"]))


def verify_envelope(case: dict, actual: dict) -> dict:
    reference = case["independent_reference"]
    oracle = case["outputs"]["result"]
    ideal = reference["result"]
    if actual["shape"] != ideal["shape"] or actual["shape"] != oracle["shape"]:
        raise GateError(f"{case['id']}: shape mismatch")
    observed_words = actual["f32_bits"]
    expected_words = ideal["f32_bits"]
    torch_words = oracle["f32_bits"]
    if len(observed_words) != len(expected_words) or len(torch_words) != len(
        expected_words
    ):
        raise GateError(f"{case['id']}: missing or extra logical elements")
    policy = case["comparison_policy"]
    max_error = Fraction(0)
    for index, (observed, expected, torch) in enumerate(
        zip(observed_words, expected_words, torch_words, strict=True)
    ):
        classes = [special(word) for word in (observed, expected, torch)]
        if any(classes):
            if classes[0] != classes[1] or classes[0] != classes[2]:
                raise GateError(
                    f"{case['id']}[{index}]: nonfinite classification mismatch"
                )
            continue
        if policy in ("exact_words", "ieee_edges", "exact_golden"):
            if observed != expected or observed != torch:
                raise GateError(f"{case['id']}[{index}]: exact-word mismatch")
            continue
        observed_value = Fraction.from_float(value(observed))
        torch_value = Fraction.from_float(value(torch))
        if policy == "bounded_elementwise":
            exact = rational(reference["exact_values"][index])
            bound = Fraction(1, 10**6) + Fraction(1, 10**5) * abs(exact)
            torch_bound = Fraction(1, 10**6) + Fraction(1, 10**5) * abs(torch_value)
            if (
                abs(observed_value - exact) > bound
                or abs(observed_value - torch_value) > torch_bound
            ):
                raise GateError(f"{case['id']}[{index}]: bounded arithmetic mismatch")
        elif policy == "reduction_envelope":
            envelope = reference["envelopes"][index]
            exact = rational(envelope["exact"])
            bound = rational(envelope["absolute_error_bound"])
            if envelope["rounding_count"] == 0 and observed != "00000000":
                raise GateError(
                    f"{case['id']}[{index}]: empty identity is not positive zero"
                )
            if abs(observed_value - exact) > bound:
                raise GateError(
                    f"{case['id']}[{index}]: Bend exceeds registered reduction bound"
                )
            if abs(torch_value - exact) > bound:
                raise GateError(
                    f"{case['id']}[{index}]: pinned oracle exceeds "
                    "registered reduction bound"
                )
            if abs(observed_value - torch_value) > 2 * bound:
                raise GateError(
                    f"{case['id']}[{index}]: pairwise reduction difference "
                    "exceeds bound"
                )
        else:
            raise GateError(f"Unregistered numerical policy: {policy}")
        error = abs(observed_value - exact)
        max_error = max(max_error, error)
    if any(
        not math.isfinite(value(word))
        for word in observed_words
        if special(word) is None
    ):
        raise GateError(f"{case['id']}: unclassified nonfinite value")
    return {
        "id": case["id"],
        "policy": policy,
        "elements": len(observed_words),
        "max_exact_error": {
            "numerator": str(max_error.numerator),
            "denominator": str(max_error.denominator),
        },
    }


ZERO_WORDS = ("00000000", "80000000")


def verify_case(case: dict, actual: dict) -> dict:
    accepted = verify_envelope(case, actual)
    policy = case["comparison_policy"]
    observed_words = actual["f32_bits"]
    expected_words = case["independent_reference"]["result"]["f32_bits"]
    torch_words = case["outputs"]["result"]["f32_bits"]
    for index, (observed, expected, torch) in enumerate(
        zip(observed_words, expected_words, torch_words, strict=True)
    ):
        if special(observed) is not None:
            continue
        if policy == "bounded_elementwise" and (
            observed != expected or observed != torch
        ):
            raise GateError(f"{case['id']}[{index}]: not the round-to-nearest word")
        if policy == "reduction_envelope" and observed in ZERO_WORDS:
            wanted = torch if torch in ZERO_WORDS else "00000000"
            if observed != wanted:
                raise GateError(
                    f"{case['id']}[{index}]: zero sign differs from PyTorch"
                )
    return {**accepted, "policy_version": 3}
