import argparse
import json
from pathlib import Path

from tools.common import ROOT, GateError
from tools.s3_broadcast import parse, program, run_lanes, same_words, verify, word
from tools.s3_reduce import cases as reduce_cases
from tools.s3_reduce import reference as reduce_reference
from tools.s3_reduce import same_sums

NEGATIVE_ZERO = 0x80000000


def handwritten() -> list[dict]:
    return [
        {
            "id": "scalar_negative_zero",
            "shape": [],
            "target": [],
            "words": [NEGATIVE_ZERO],
        },
        {
            "id": "vector_negative_zero",
            "shape": [1],
            "target": [1],
            "words": [NEGATIVE_ZERO],
        },
        {
            "id": "matrix_negative_zero",
            "shape": [2, 3],
            "target": [2, 3],
            "words": [NEGATIVE_ZERO] * 6,
        },
        {
            "id": "row_negative_zero",
            "shape": [1, 3],
            "target": [1, 3],
            "words": [NEGATIVE_ZERO] * 3,
        },
        {
            "id": "reduced_negative_zero",
            "shape": [1, 3],
            "target": [3],
            "words": [NEGATIVE_ZERO] * 3,
        },
        {
            "id": "reduced_scalar_negative_zero",
            "shape": [1],
            "target": [],
            "words": [NEGATIVE_ZERO],
        },
        {
            "id": "identity_nan_payloads",
            "shape": [2],
            "target": [2],
            "words": [0x7FC01234, 0xFFA00001],
        },
        {
            "id": "identity_ordered_words",
            "shape": [4],
            "target": [4],
            "words": [word(1e8), word(1), word(-1e8), word(1)],
        },
    ]


def cases() -> list[dict]:
    result = handwritten() + reduce_cases()
    if len({case["id"] for case in result}) != len(result):
        raise GateError("Duplicate S3 sum_to_size case identifier")
    return result


def reference(case: dict) -> dict:
    if case["shape"] == case["target"]:
        return {"shape": case["target"], "words": list(case["words"])}
    return reduce_reference(case)


def identity(case: dict) -> bool:
    return case["shape"] == case["target"]


def render(selected: list[dict]):
    def generate(directory: Path, source_root: Path) -> str:
        return program(selected, directory, source_root, "sum_to_size")

    return generate


def checker(selected: list[dict]):
    def check(stdout: str, lane: str) -> tuple[list, dict]:
        observations = parse(stdout, [case["id"] for case in selected])
        summary = {"cases": 0, "tensors": 0, "errors": 0}
        for same, compare in ((True, same_words), (False, same_sums)):
            subset = [case for case in selected if identity(case) == same]
            rows = [
                row
                for case, row in zip(selected, observations, strict=True)
                if identity(case) == same
            ]
            part = verify(subset, rows, lane, reference, compare)
            summary = {key: summary[key] + part[key] for key in summary}
        return observations, summary

    return check


def run_stage(tool_root: Path, output: Path) -> dict:
    selected = cases()
    return run_lanes(
        tool_root,
        output,
        "sum_to_size",
        selected,
        render(selected),
        checker(selected),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-root", type=Path, default=ROOT / ".tools")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage(args.tool_root, args.output), indent=2))


if __name__ == "__main__":
    main()
