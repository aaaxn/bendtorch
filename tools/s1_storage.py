import argparse
import json
import os
import tempfile
from pathlib import Path

from tools.common import ROOT, Evidence, GateError, bend, check_sources

CAPACITY = 1 << 20
MAX_U32 = (1 << 32) - 1


def cases() -> list[dict]:
    result = []
    for length in [*range(18), 31, 32, 33, 63, 64, 65, 255, 256, 257, CAPACITY]:
        steps = [
            {"op": "read_copy", "index": 0},
            {"op": "write_copy", "index": 0, "bits": 0},
            *({"op": "read", "index": i} for i in range(min(length, 17))),
        ]
        if length:
            index = length - 1
            steps += [
                {"op": "write", "index": index, "bits": 0x40E00000},
                {"op": "read", "index": index},
                {"op": "copy"},
                {"op": "write", "index": index, "bits": 0x41300000},
                {"op": "read_copy", "index": index},
                {"op": "write_copy", "index": index, "bits": 0x80000000},
                {"op": "read", "index": index},
                {"op": "read_copy", "index": index},
            ]
            if length <= 17:
                steps += [{"op": "read", "index": i} for i in range(length)]
        else:
            steps += [{"op": "copy"}, {"op": "read_copy", "index": 0}]
        for index in (length, (1 << max(0, (length - 1).bit_length())), MAX_U32):
            steps += [
                {"op": "read", "index": index},
                {"op": "write", "index": index, "bits": 0x42C60000},
            ]
        if length:
            steps.append({"op": "read", "index": length - 1})
        result.append({"length": length, "bits": 0x40A00000, "steps": steps})
    for word in (
        0,
        0x80000000,
        1,
        0x007FFFFF,
        0x7F7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC01234,
        0x7F800001,
    ):
        result.append(
            {
                "length": 1,
                "bits": word,
                "steps": [
                    {"op": "read", "index": 0},
                    {"op": "copy"},
                    {"op": "read_copy", "index": 0},
                ],
            }
        )
    for depth in range(6, 21):
        result.append(sparse_case((1 << (depth - 1)) + 1))
    result.append(sparse_case(CAPACITY))
    for length in (CAPACITY + 1, MAX_U32):
        result.append({"length": length, "bits": 0, "steps": []})
    return result


def sparse_case(length: int) -> dict:
    written = sorted(
        {0, length - 1}
        | {
            index
            for bit in range(length.bit_length())
            for index in (1 << bit, (1 << bit) - 1)
            if index < length
        }
    )
    untouched = [index for index in (5, length - 2) if index not in written]
    reads = [{"op": "read", "index": index} for index in written + untouched]
    return {
        "length": length,
        "bits": 0x40A00000,
        "steps": [
            *(
                {"op": "write", "index": index, "bits": 0x3F800000 + index}
                for index in written
            ),
            *reads,
            {"op": "copy"},
            *({**read, "op": "read_copy"} for read in reads),
        ],
    }


def expected(requests: list[dict]) -> list[str]:
    output = []
    for case in requests:
        length = case["length"]
        if length > CAPACITY:
            output.append("e:TooLarge")
            continue
        values = [case["bits"]] * length
        copied = None
        output.append(f"new:{length}")
        for step in case["steps"]:
            op = step["op"]
            if op == "copy":
                copied = values.copy()
                output.append("copy")
                continue
            target = copied if op.endswith("_copy") else values
            if target is None:
                output.append("e:NoCopy")
            elif step["index"] >= len(target):
                output.append("e:OutOfBounds")
            elif op.startswith("read"):
                output.append(f"v:{target[step['index']]}")
            else:
                target[step["index"]] = step["bits"]
                output.append("ok")
    return output


def import_path(target: Path, directory: Path) -> str:
    relative = os.path.relpath(target, directory)
    return relative if relative.startswith(".") else "./" + relative


def bend_program(requests: list[dict], directory: Path) -> str:
    harness = import_path(ROOT / "src/storage/harness.bend", directory)
    constructors = {
        "read": "Read",
        "write": "Write",
        "copy": "Copy",
        "read_copy": "ReadCopy",
        "write_copy": "WriteCopy",
    }
    rendered = []
    for case in requests:
        steps = []
        for step in case["steps"]:
            arguments = [str(step[key]) for key in ("index", "bits") if key in step]
            steps.append(
                f"Harness.{constructors[step['op']]}{{{', '.join(arguments)}}}"
            )
        rendered.append(
            f"Harness.Case{{{case['length']}, {case['bits']}, [{', '.join(steps)}]}}"
        )
    return (
        f"import Base\nimport {harness} as Harness\n\ndef main() -> IO(Unit):\n"
        f"  Harness.run([{', '.join(rendered)}])\n"
    )


def verify_output(actual: str, golden: list[str], lane: str) -> None:
    rows = actual.splitlines()
    if len(rows) != len(golden):
        raise GateError(f"{lane}: expected {len(golden)} observations, got {len(rows)}")
    for index, (observed, reference) in enumerate(zip(rows, golden, strict=True)):
        if observed == reference:
            continue
        if (
            lane == "javascript"
            and observed.startswith("v:")
            and reference.startswith("v:")
        ):
            a, b = int(observed[2:]), int(reference[2:])
            nan_a = a & 0x7F800000 == 0x7F800000 and a & 0x7FFFFF != 0
            nan_b = b & 0x7F800000 == 0x7F800000 and b & 0x7FFFFF != 0
            if nan_a and nan_b:
                continue
        raise GateError(
            f"{lane}: observation {index} expected {reference!r}, got {observed!r}"
        )


def instrument_copy_audit(text: str) -> str:
    replacements = [
        (
            "INLINE Term blk_node(Env e, Term l, Term r) {",
            "static u64 storage_audit_halves, storage_audit_nodes;\n"
            "INLINE Term blk_node(Env e, Term l, Term r) {\n"
            "  storage_audit_nodes++;",
        ),
        (
            "INLINE Term blk_half(Env e, Term a, u32 hi) {",
            "INLINE Term blk_half(Env e, Term a, u32 hi) {\n  storage_audit_halves++;",
        ),
        (
            "  io_sync();\n  return 0;",
            "  io_sync();\n"
            '  fprintf(stderr, "STORAGE_COPY_AUDIT %llu %llu\\n",\n'
            "    (unsigned long long)storage_audit_halves,\n"
            "    (unsigned long long)storage_audit_nodes);\n  return 0;",
        ),
    ]
    for old, new in replacements:
        if text.count(old) != 1:
            raise GateError("Pinned C instrumentation site is missing or ambiguous")
        text = text.replace(old, new)
    return text


def generated(name: str, text: str, build: Path, output: Path) -> Path:
    (output / name).write_text(text)
    (build / name).write_text(text)
    return build / name


def check_storage(
    evidence: Evidence, tool_root: Path, output: Path, build: Path
) -> None:
    requests = cases()
    (output / "requests.json").write_text(json.dumps(requests, indent=2) + "\n")
    golden = expected(requests)
    (output / "expected.txt").write_text("\n".join(golden) + "\n")
    program = generated("cases.bend", bend_program(requests, build), build, output)
    evidence.run(
        "storage-source-check",
        bend(tool_root, ROOT / "src/storage/PROOF.bend", "--check-only"),
        contains="ALL PROOFS CHECK",
    )
    evidence.run(
        "storage-kernel-check",
        bend(tool_root, ROOT / "src/storage/PROOF.bend", "--verdict"),
        contains="ALL PROOFS CHECK",
    )
    for lane in ("native", "javascript"):
        target = output / ("storage" if lane == "native" else "storage.js")
        evidence.run(f"storage-{lane}-build", bend(tool_root, program, "-o", target))
        command = (
            [str(target), "--threads", "1", "--gpu", "off"]
            if lane == "native"
            else [bend(tool_root)[0], str(target)]
        )
        actual = evidence.run(f"storage-{lane}-run", command)
        verify_output(actual.stdout, golden, lane)
    bounds_failure = "- expected : False{}\n- observed : True{}"
    certificates = {
        "unbalanced": (
            "Storage.Owned{2n, 3n, Array.new(F32, 2n, 0.0), "
            "Model.new_balanced(F32, 2n, 0.0), {==}}",
            "Storage.Owned{2n, 3n, ALeaf{0.0}, Unit{}, {==}}",
            "- expected : Empty\n- observed : Unit",
        ),
        "length": (
            "Storage.Owned{0n, 1n, ALeaf{0.0}, Unit{}, {==}}",
            "Storage.Owned{0n, 2n, ALeaf{0.0}, Unit{}, {==}}",
            bounds_failure,
        ),
        "depth": (
            "Storage.Owned{20n, 1n, Array.new(F32, 20n, 0.0), "
            "Model.new_balanced(F32, 20n, 0.0), {==}}",
            "Storage.Owned{21n, 1n, Array.new(F32, 21n, 0.0), "
            "Model.new_balanced(F32, 21n, 0.0), {==}}",
            bounds_failure,
        ),
    }
    main_path = import_path(ROOT / "src/storage/main.bend", build)
    model_path = import_path(ROOT / "src/storage/array.bend", build)
    for name, (accepted, rejected, reason) in certificates.items():
        for verdict, term in (("accept", accepted), ("reject", rejected)):
            fixture = generated(
                f"{verdict}-{name}.bend",
                f"import Base\nimport {main_path} as Storage\n"
                f"import {model_path} as Model\n\n"
                f"def certificate() -> Storage.Storage:\n  {term}\n",
                build,
                output,
            )
            for mode in ("--check-only", "--verdict"):
                if verdict == "accept":
                    evidence.run(
                        f"accept-{name}-{mode[2:]}",
                        bend(tool_root, fixture, mode),
                        contains="ALL PROOFS CHECK",
                    )
                else:
                    evidence.run(
                        f"reject-{name}-{mode[2:]}",
                        bend(tool_root, fixture, mode),
                        expected_code=1,
                        contains=f"SOME PROOFS FAIL\nError:\n{reason}\n",
                    )
    emitted = output / "storage.c"
    evidence.run("storage-c-export", bend(tool_root, program, "-o", emitted))
    counted = output / "storage-audit.c"
    counted.write_text(instrument_copy_audit(emitted.read_text()))
    audit_binary = output / "storage-audit"
    evidence.run(
        "storage-copy-audit-build",
        [
            str(tool_root / "llvm/usr/bin/clang-19"),
            "-std=c11",
            "-O3",
            str(counted),
            "-lpthread",
            "-lm",
            "-o",
            str(audit_binary),
        ],
    )
    audit = evidence.run(
        "storage-copy-audit-run",
        [str(audit_binary), "--threads", "1", "--gpu", "off"],
        contains="STORAGE_COPY_AUDIT 0 0",
    )
    verify_output(audit.stdout, golden, "native")
    (output / "coverage.json").write_text(
        json.dumps(
            {
                "cases": len(requests),
                "observations_per_lane": len(golden),
                "negative_certificates": len(certificates),
            },
            indent=2,
        )
        + "\n"
    )


def run_stage(tool_root: Path, output: Path) -> None:
    check_sources(tool_root)
    evidence = Evidence(output, tool_root)
    parent = ROOT / ".evidence"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="build-", dir=parent) as build:
        check_storage(evidence, tool_root, output, Path(build))
    evidence.save()
    print(f"S1 storage source, native, and JS checks passed: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-root", type=Path, default=ROOT / ".tools")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_stage(args.tool_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
