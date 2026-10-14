import hashlib
import json
import math

from tools.common import GateError

OPERATIONS = {"sum", "matmul", "matmul_add"}


def classified_words(tensor: dict) -> list[tuple[str, str]]:
    words = tensor["f32_bits"]
    classes = tensor.get("classifications")
    if len(words) != math.prod(tensor["shape"]):
        raise GateError("Ordered-reference words disagree with their shape")
    if classes is not None and len(words) != len(classes):
        raise GateError("Missing ordered-reference classifications")
    result = []
    for index, word in enumerate(words):
        if len(word) != 8 or any(char not in "0123456789abcdef" for char in word):
            raise GateError("Malformed ordered-reference binary32 word")
        bits = int(word, 16)
        sign = "negative" if bits & 0x80000000 else "positive"
        exponent = bits & 0x7F800000
        fraction = bits & 0x7FFFFF
        if exponent == 0x7F800000:
            kind = "nan" if fraction else f"{sign}_infinity"
        elif exponent == 0:
            kind = f"{sign}_subnormal" if fraction else f"{sign}_zero"
        else:
            kind = f"{sign}_normal"
        if classes is not None and kind != classes[index]:
            raise GateError("Ordered-reference classification disagrees with its word")
        result.append((word, kind))
    return result


def verify_ordered_case(record: dict, actual: dict) -> dict:
    expected = record["result"]
    identifier = record["id"]
    if actual["shape"] != expected["shape"]:
        raise GateError(f"{identifier}: ordered-reference shape mismatch")
    observed = classified_words(actual)
    reference = classified_words(expected)
    if len(observed) != len(reference):
        raise GateError(f"{identifier}: ordered-reference element count mismatch")
    for index, ((word, kind), (target, target_kind)) in enumerate(
        zip(observed, reference, strict=True)
    ):
        if kind != target_kind or (kind != "nan" and word != target):
            raise GateError(
                f"{identifier}[{index}]: ordered F32 primitive-rounding mismatch"
            )
    return {
        "id": identifier,
        "contract": "s2_ordered_f32_rne",
        "elements": len(observed),
    }


def verify_ordered_lane(
    corpus: dict, records: dict, observations: list[dict]
) -> list[dict]:
    cases = [
        case
        for group in ("finite_cases", "edge_cases", "nonfinite_cases")
        for case in corpus[group]
        if case["op"] in OPERATIONS
    ]
    if records["contract"] != "s2_ordered_f32_rne" or [
        row["id"] for row in records["cases"]
    ] != [case["id"] for case in cases]:
        raise GateError("Ordered reference requires the complete reduction inventory")
    observed = {row["id"]: row for row in observations}
    if len(observed) != len(observations):
        raise GateError("Duplicate ordered-reference observations")
    accepted = []
    for case, record in zip(cases, records["cases"], strict=True):
        request = {
            "op": case["op"],
            "inputs": case["inputs"],
            "parameters": case.get("parameters", {}),
        }
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        if record["input_sha256"] != digest:
            raise GateError(
                f"{case['id']}: ordered reference is for different frozen inputs"
            )
        if case["id"] not in observed:
            raise GateError(f"{case['id']}: missing ordered-reference observation")
        accepted.append(verify_ordered_case(record, observed[case["id"]]["result"]))
    return accepted
