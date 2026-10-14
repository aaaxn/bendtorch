import argparse
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.common import GateError, check_sources

ROOT = Path(__file__).resolve().parents[2]
CASE_NAMES = (
    "scalar_witness",
    "three_zero_witness",
    "three_one_witness",
    "three_two_witness",
)
CONSUMERS = (
    "consumers.scalar",
    "consumers.three_zero",
    "consumers.three_one",
    "consumers.three_two",
    "consumers.empty",
    "consumers.huge_leading",
    "consumers.huge_trailing",
)
LIVE_RECORDS = (
    "generic_2d_metadata.Construction.arms",
    "generic_2d_read.Read.arms",
    "retention.Retained.arms",
    "generic_2d_public.PublicRead.arms",
    "generic_2d_public.Claims.arms",
    "consumers.Case.arms",
    "consumers.EmptyCase.arms",
)
SEVEN = "F32.lit1088421888"
MAYBE_F32 = ("(Maybe -.Q2 -F32)", "(Maybe.q2 -F32)")
EIGHT = "F32.lit1090519040"


def catalog(text: str) -> dict[str, tuple[str, str]]:
    result = {}
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        header, separator, body = block.partition(" =\n")
        name = header.split(" : ", 1)[0]
        if not separator or " : " not in header or name in result:
            raise GateError(f"Malformed or duplicate exported declaration: {name}")
        result[name] = (header, body)
    return result


def replace_declaration(text: str, name: str, header: str, body: str) -> str:
    old_header, old_body = catalog(text)[name]
    old = f"{old_header} =\n{old_body}"
    if text.count(old) != 1:
        raise GateError(f"Exported declaration is not unique: {name}")
    return text.replace(old, f"{header} =\n{body}")


def fill_body(body: str, zero: str) -> str:
    match = re.match(r"\s*λ[-+]?[A-Za-z_]\w* => λ[-+]?(?P<fill>[A-Za-z_]\w*) => ", body)
    if match is None:
        raise GateError("Inspect actual two-parameter fill lambda before mutation")
    expression, count = re.subn(
        rf"(?<![A-Za-z0-9_.]){re.escape(match.group('fill'))}(?![A-Za-z0-9_.])",
        lambda _: zero,
        body[match.end() :],
    )
    if count == 0:
        raise GateError("No actual fill argument in exported body")
    return body[: match.end()] + expression


def mutation_copy(snapshot: Path, target: Path) -> None:
    shutil.copytree(snapshot, target)
    core = target / "src/tensor/core.bend"
    text = core.read_text()
    old = "new_result(shape, Storage.new(Shape.count(shape), fill))"
    if text.count(old) != 1:
        raise GateError("Actual Core.new mutation target is not unique")
    core.write_text(
        text.replace(old, "new_result(shape, Storage.new(Shape.count(shape), 0.0))")
    )
    proof = target / "src/tensor/allocation-PROOF.bend"
    text = proof.read_text()
    start = text.index("def core_new(+shape:")
    end = text.index("\ndef core_new_done", start)
    header, body = text[start:end].split(":\n", 1)
    body = body.replace(
        "Storage.new(Shape.count(shape), fill)", "Storage.new(Shape.count(shape), 0.0)"
    ).replace(
        "storage_new(Shape.count(shape), fill,", "storage_new(Shape.count(shape), 0.0,"
    )
    proof.write_text(text[:start] + header + ":\n" + body + text[end:])


def false_twin(controls: str, witness: str) -> str:
    start = controls.index(f"def {witness}()")
    end = controls.index("}:\n", start) + 2
    header = controls[start:end]
    if header.count("expected_value()") != 1:
        raise GateError("False public statement target is not unique")
    return (
        controls[:start]
        + header.replace("expected_value()", "wrong_value()")
        + controls[end:]
    )


def source_value_reason(output: str, label: str) -> str:
    lines = output.splitlines()
    expected = next((x[13:] for x in lines if x.startswith("- expected : ")), "")
    observed = next((x[13:] for x in lines if x.startswith("- observed : ")), "")
    location = next((x[10:] for x in lines if x.startswith("Location: ")), "")
    if "*1" in expected + observed or "*2" in expected + observed:
        raise GateError(f"{label}: quantity/kind mismatch is not value sensitivity")
    if label in CASE_NAMES:
        if (
            expected != "{Some{7.0} == Some{8.0} : Maybe<&2, F32>}"
            or observed != "{Some{7.0} == Some{7.0} : Maybe<&2, F32>}"
            or location != label
        ):
            raise GateError(f"{label}: wrong concrete public value equation")
    elif (
        not expected.startswith("{Observation.point(shape, index,")
        or not expected.endswith("== Some{fill} : Maybe<&2, F32>}")
        or expected.count(", 0.0, ") != 1
        or expected.replace(", 0.0, ", ", fill, ") != observed
        or location != "produce"
    ):
        raise GateError("Wrong-fill failure is not the actual public value claim")
    return f"{location}: expected {expected} observed {observed}"


def kernel_value_reason(output: str, label: str) -> str:
    lines = output.splitlines()
    location = next((x[3:-1] for x in lines if x.startswith("In ")), "")
    expected = next((x[10:] for x in lines if x.startswith("expected: ")), "")
    observed = next((x[10:] for x in lines if x.startswith("observed: ")), "")
    if label in CASE_NAMES:
        if (
            location != label
            or "== wrong_value : " not in expected
            or expected.replace("== wrong_value", "== expected_value") != observed
        ):
            raise GateError(f"{label}: kernel rejection is not the value statement")
    elif (
        location != "generic_2d_read.produce"
        or "== (.Some, (x1, ())) : " not in expected
        or "== (.Some, (x1, ())) : " not in observed
        or "core.new +x0) x1)" not in expected
        or "core.new_result" not in observed
    ):
        raise GateError("Wrong-fill kernel rejection is not the public value claim")
    return f"{location}: expected {expected} observed {observed}"


def check_export(text: str) -> dict[str, str]:
    declarations = catalog(text)
    names = (
        *CONSUMERS,
        *CASE_NAMES,
        *LIVE_RECORDS,
        "generic_2d_public.produce",
        "generic_2d_public.consume",
        "expected_value",
        "wrong_value",
    )
    missing = [name for name in names if name not in declarations]
    if missing:
        raise GateError(f"Export omitted declarations: {missing}")
    for name in LIVE_RECORDS:
        if "Σ-" in declarations[name][1]:
            raise GateError(f"Erased field in a LIVE evidence record: {name}")
    for name in ("generic_2d_public.produce", "generic_2d_public.consume"):
        header = declarations[name][0]
        if not header.startswith(f"{name} : ∀+x0 : ") or "∀+x1 : F32" not in header:
            raise GateError(f"Unexpected producer quantities: {name}")
        if "∀x3 : {(Nat.is_lt (U32.to_nat x2)" not in header:
            raise GateError(f"In-bounds proof is not an ordinary argument: {name}")
    for name in CASE_NAMES:
        header = declarations[name][0].replace(*MAYBE_F32)
        if "core.new +consumers." not in header or SEVEN not in header:
            raise GateError(f"Witness statement is not about Core.new 7.0: {name}")
        if not header.endswith("== expected_value : (Maybe.q2 -F32)}"):
            raise GateError(f"Witness statement target changed: {name}")
    if declarations["expected_value"][1].strip() != f"(.Some, ({SEVEN}, ()))":
        raise GateError("expected_value is not Some(7.0)")
    if declarations["wrong_value"][1].strip() != f"(.Some, ({EIGHT}, ()))":
        raise GateError("wrong_value is not Some(8.0)")
    return {name: "\n".join(declarations[name]) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bend-root", type=Path, default=ROOT / ".tools/bend")
    parser.add_argument("--kernel", type=Path, default=ROOT / ".tools/kernel/bendtt")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    bend_root = args.bend_root.resolve()
    kernel = args.kernel.resolve()
    output.mkdir(parents=True)
    pinned = bend_root == (ROOT / ".tools/bend").resolve()
    if pinned:
        check_sources(ROOT / ".tools")
    env = os.environ | {
        "BEND_NO_TELEMETRY": "1",
        "BENDTT": str(kernel),
        "LEAN_STACK_SIZE_KB": "4194304",
        "LEAN_NUM_THREADS": "1",
    }
    records: list[dict] = []
    summary = {
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "repository_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "src", "verification/r1-a"],
                cwd=ROOT,
                text=True,
            )
        ),
        "bend_commit": subprocess.check_output(
            ["git", "-C", str(bend_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "bend_is_repository_pin": pinned,
        "kernel": str(kernel),
        "affinity": sorted(os.sched_getaffinity(0)),
        "workers": args.workers,
        "commands": records,
    }

    def save() -> None:
        records.sort(key=lambda row: row["name"])
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def run(name: str, command: list, expected_code: int) -> str:
        start = time.monotonic()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=1500,
            check=False,
        )
        text = result.stdout + result.stderr
        (output / f"{name}.out").write_text(text)
        records.append(
            {
                "name": name,
                "command": [str(part) for part in command],
                "exit_code": result.returncode,
                "expected_code": expected_code,
                "seconds": round(time.monotonic() - start, 1),
            }
        )
        if result.returncode != expected_code:
            raise GateError(f"{name}: exit {result.returncode}; see {output}")
        if (
            expected_code == 0
            and "-o" not in command
            and "ALL PROOFS CHECK" not in text
        ):
            raise GateError(f"{name}: no ALL PROOFS CHECK; see {output}")
        if expected_code == 1 and "SOME PROOFS FAIL" not in text:
            raise GateError(f"{name}: not a proof rejection; see {output}")
        return text

    bend = ["bun", str(bend_root / "bend2/main.ts")]
    snapshot = output / "snapshot"
    shutil.copytree(ROOT / "src", snapshot / "src")
    shutil.copytree(ROOT / "verification/r1-a", snapshot / "verification/r1-a")
    experiment = snapshot / "verification/r1-a"
    controls = (experiment / "controls.bend").read_text()
    for witness in CASE_NAMES:
        (experiment / f"false-{witness}.bend").write_text(false_twin(controls, witness))
    mutant = output / "wrong-fill"
    mutation_copy(snapshot, mutant)
    reasons: dict[str, str] = {}
    summary["rejection_reasons"] = reasons

    try:
        source_jobs = [
            ("source-controls", experiment / "controls.bend", 0),
            ("source-metadata", experiment / "metadata-control.bend", 0),
            (
                "source-wrong-fill-metadata",
                mutant / experiment.relative_to(snapshot) / "metadata-control.bend",
                0,
            ),
            (
                "source-wrong-fill",
                mutant / experiment.relative_to(snapshot) / "controls.bend",
                1,
            ),
            *(
                (f"source-false-{witness}", experiment / f"false-{witness}.bend", 1)
                for witness in CASE_NAMES
            ),
        ]
        with ThreadPoolExecutor(args.workers) as pool:
            texts = dict(
                zip(
                    (name for name, _, _ in source_jobs),
                    pool.map(
                        lambda job: run(
                            job[0], [*bend, job[1], "--check-only"], job[2]
                        ),
                        source_jobs,
                    ),
                    strict=True,
                )
            )
        for witness in CASE_NAMES:
            reasons[f"source-false-{witness}"] = source_value_reason(
                texts[f"source-false-{witness}"], witness
            )
        reasons["source-wrong-fill"] = source_value_reason(
            texts["source-wrong-fill"], "wrong-fill"
        )

        positive = output / "controls.bendtt"
        metadata = output / "wrong-fill-metadata.bendtt"
        metadata_entry = (
            mutant / experiment.relative_to(snapshot) / "metadata-control.bend"
        )
        with ThreadPoolExecutor(args.workers) as pool:
            list(
                pool.map(
                    lambda job: run(*job),
                    (
                        (
                            "verdict-controls",
                            [*bend, experiment / "controls.bend", "--verdict"],
                            0,
                        ),
                        (
                            "verdict-wrong-fill-metadata",
                            [*bend, metadata_entry, "--verdict"],
                            0,
                        ),
                        (
                            "export-controls",
                            [*bend, experiment / "controls.bend", "-o", positive],
                            0,
                        ),
                        (
                            "export-wrong-fill-metadata",
                            [*bend, metadata_entry, "-o", metadata],
                            0,
                        ),
                    ),
                )
            )
        text = positive.read_text()
        (output / "export-statements.txt").write_text(
            "\n\n".join(check_export(text).values()) + "\n"
        )
        declarations = catalog(text)
        artifacts = {}
        for witness in CASE_NAMES:
            header, body = declarations[witness]
            artifacts[witness] = replace_declaration(
                text,
                witness,
                header.replace("== expected_value", "== wrong_value"),
                body,
            )
        zero_name = declarations["wrong_fill_value"][1].strip()
        zero = declarations[zero_name][1].strip()
        wrong = text
        for suffix in ("_2f_core.new", "_2f_allocation_2d_PROOF.core_new"):
            names = [name for name in declarations if name.endswith(suffix)]
            if len(names) != 1:
                raise GateError(f"Wrong-fill export target is ambiguous: {suffix}")
            header, body = declarations[names[0]]
            wrong = replace_declaration(wrong, names[0], header, fill_body(body, zero))
        artifacts["wrong-fill"] = wrong
        for label, artifact in artifacts.items():
            (output / f"false-{label}.bendtt").write_text(artifact)
        kernel_jobs = [
            ("kernel-controls", [kernel, positive], 0),
            ("kernel-wrong-fill-metadata", [kernel, metadata], 0),
            *(
                (f"kernel-false-{label}", [kernel, output / f"false-{label}.bendtt"], 1)
                for label in artifacts
            ),
        ]
        with ThreadPoolExecutor(args.workers) as pool:
            texts = dict(
                zip(
                    (name for name, _, _ in kernel_jobs),
                    pool.map(lambda job: run(*job), kernel_jobs),
                    strict=True,
                )
            )
        for label in artifacts:
            reasons[f"kernel-false-{label}"] = kernel_value_reason(
                texts[f"kernel-false-{label}"], label
            )
        summary["passed"] = True
    except (GateError, subprocess.TimeoutExpired) as error:
        summary["passed"] = False
        summary["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
