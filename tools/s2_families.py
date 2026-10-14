import argparse
import json
import math
import os
import shutil
from pathlib import Path

from tools.common import ROOT, Evidence, GateError, bend, check_sources
from tools.s2_numerics import special, verify_case_v3
from tools.s2_runtime import (
    CORPUS,
    U32Range,
    corpus_cases,
    decode_case,
    error_name,
    input_source,
    load_corpus,
    native_command,
    parse_native_output,
)

BINARY = {"add": "Add", "sub": "Sub", "mul": "Mul"}


def requests(corpus: dict) -> list[tuple[dict, dict]]:
    """The S2 add, sub and mul cases that reach Bend, with their requests."""
    selected = []
    for case in corpus_cases(corpus):
        if case["op"] not in BINARY:
            continue
        try:
            request = decode_case(case)
        except U32Range:
            continue
        if "compile_negative" not in request:
            selected.append((case, request))
    return selected


def program(selected: list[dict], directory: Path, source_root: Path = ROOT) -> str:
    if not selected:
        raise GateError("Refusing to generate an empty pointwise family corpus")
    lines = ["import Base"]
    for path, alias in (
        ("src/tensor/harness.bend", "Harness"),
        ("src/tensor/core.bend", "Core"),
        ("src/tensor/elementwise.bend", "Elementwise"),
        ("src/tensor/contracts/harness.bend", "Family"),
    ):
        relative = os.path.relpath(source_root / path, directory)
        lines.append(f"import ./{relative} as {alias}")
    for index, request in enumerate(selected):
        inputs = request["inputs"]
        lines += [
            "",
            f"def run_case_{index}() -> IO(Unit):",
            "  do IO<Unit>:",
            f'    IO.print("S2:begin:{request["id"]}")',
            f"    Family.binary(Elementwise.{BINARY[request['op']]}{{}}, "
            f"{input_source(inputs['a'])}, {input_source(inputs['b'])})",
            '    IO.print("S2:end")',
        ]
    batches = range(0, len(selected), 16)
    for batch, start in enumerate(batches):
        lines += ["", f"def run_batch_{batch}() -> IO(Unit):", "  do IO<Unit>:"]
        lines += [
            f"    run_case_{index}()"
            for index in range(start, min(start + 16, len(selected)))
        ]
    lines += ["", "def main() -> IO(Unit):", "  do IO<Unit>:"]
    lines += [f"    run_batch_{batch}()" for batch in range(len(batches))]
    return "\n".join(lines) + "\n"


def input_words(encoded: dict) -> list[str]:
    if encoded.get("fill_zero"):
        return ["00000000"] * math.prod(encoded["shape"])
    return encoded["f32_bits"]


def same_input(observed: dict, encoded: dict) -> bool:
    """A retained input keeps its shape and words. A NaN input word may come
    back as another NaN, as `verify_lane` allows for constructed NaNs."""
    expected = input_words(encoded)
    return (
        observed["shape"] == encoded["shape"]
        and len(observed["f32_bits"]) == len(expected)
        and all(
            word == wanted or (special(wanted) == "nan" and special(word) == "nan")
            for word, wanted in zip(observed["f32_bits"], expected, strict=True)
        )
    )


def verify(selected: list[tuple[dict, dict]], observations: list[dict]) -> dict:
    if [row["id"] for row in observations] != [case["id"] for case, _ in selected]:
        raise GateError("Pointwise family corpus inventory or order differs")
    tensors, errors = [], []
    for (case, request), row in zip(selected, observations, strict=True):
        identifier = case["id"]
        if "boundary" in row or set(row.get("retained", {})) != {"left", "right"}:
            raise GateError(f"{identifier}: missing retained operands")
        for owner, name in (("left", "a"), ("right", "b")):
            if not same_input(row["retained"][owner], request["inputs"][name]):
                raise GateError(f"{identifier}: {owner} input changed")
        if "expected_error" in case:
            category = case["expected_error"]
            if (
                row.get("error") != category
                or "raw_error" not in row
                or error_name(row["raw_error"]) != category
            ):
                raise GateError(f"{identifier}: expected error {category}")
            errors.append({"id": identifier, "category": category})
            continue
        if "result" not in row:
            raise GateError(f"{identifier}: missing output tensor")
        tensors.append(verify_case_v3(case, row["result"]))
    return {"cases": len(selected), "tensors": tensors, "errors": errors}


def run_stage(tool_root: Path, output: Path, corpus_path: Path = CORPUS) -> dict:
    """Runs the S2 binary corpus through `Broadcast.binary` in both lanes."""
    tool_root, output = tool_root.resolve(), output.resolve()
    check_sources(tool_root)
    evidence = Evidence(output, tool_root)
    shutil.copytree(ROOT / "src", output / "source/src")
    selected = requests(load_corpus(corpus_path))
    (output / "requests.json").write_text(
        json.dumps([request for _, request in selected], indent=2) + "\n"
    )
    source = output / "family.bend"
    source.write_text(
        program([request for _, request in selected], output, output / "source")
    )
    evidence.run(
        "family-check",
        bend(tool_root, source, "--check-only"),
        contains="ALL PROOFS CHECK",
    )
    emitted = output / "family.c"
    evidence.run("family-c-export", bend(tool_root, source, "-o", emitted))
    executable = output / "family"
    evidence.run(
        "family-native-build",
        [
            str(tool_root / "llvm/usr/bin/clang-19"),
            "-std=c11",
            "-O3",
            str(emitted),
            "-lpthread",
            "-lm",
            "-o",
            str(executable),
        ],
    )
    script = output / "family.js"
    evidence.run("family-javascript-build", bend(tool_root, source, "-o", script))
    summary = {}
    identifiers = [case["id"] for case, _ in selected]
    for lane, command in (
        ("native", native_command(executable)),
        ("javascript", [bend(tool_root)[0], str(script)]),
    ):
        run = evidence.run(f"family-{lane}-run", command, timeout=1800)
        observations = parse_native_output(run.stdout, identifiers)
        (output / f"{lane}-results.json").write_text(
            json.dumps(observations, indent=2) + "\n"
        )
        summary[lane] = verify(selected, observations)
    (output / "acceptance.json").write_text(json.dumps(summary, indent=2) + "\n")
    evidence.save()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-root", type=Path, default=ROOT / ".tools")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_stage(args.tool_root, args.output)
    print(
        json.dumps(
            {
                lane: {"cases": row["cases"], "errors": len(row["errors"])}
                for lane, row in summary.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
