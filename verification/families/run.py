import argparse
import importlib
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.common import FAMILY_WORKERS, GateError, check_sources, cpus

contracts = importlib.import_module("verification.contracts.run")
reads = importlib.import_module("verification.contracts.reads_controls")
FAMILIES = tuple(
    importlib.import_module(f"verification.families.{name}_controls").FAMILY
    for name in ("pointwise", "fold", "reduce", "broadcast_to", "tensor_vjp", "binary")
)

ROOT = Path(__file__).resolve().parents[2]


def family_controls(family, snapshot, output, bend, kernel, parallel, reasons) -> None:
    name = family.name
    output = output / name
    output.mkdir()
    experiment = snapshot / family.experiment
    source = (experiment / "controls.bend").read_text()
    for witness in family.witnesses:
        (experiment / f"false-{witness}.bend").write_text(
            family.false_twin(source, witness)
        )
    copies = {}
    for label in family.mutants:
        copies[label] = output / label
        family.mutant_copy(snapshot, copies[label], label)
    texts = parallel(
        [
            (
                f"{name}/source-consumers",
                [*bend, experiment / "consumers.bend", "--check-only"],
                0,
            ),
            (
                f"{name}/source-controls",
                [*bend, experiment / "controls.bend", "--check-only"],
                0,
            ),
            *(
                (
                    f"{name}/source-false-{witness}",
                    [*bend, experiment / f"false-{witness}.bend", "--check-only"],
                    1,
                )
                for witness in family.witnesses
            ),
            *(
                (
                    f"{name}/source-{label}",
                    [
                        *bend,
                        copies[label] / family.rejecting_module(label),
                        "--check-only",
                    ],
                    1,
                )
                for label in family.mutants
            ),
            *(
                (
                    f"{name}/source-{label}-operation",
                    [*bend, copies[label] / family.entry_path, "--check-only"],
                    0,
                )
                for label in family.mutants
            ),
        ]
    )
    for witness in family.witnesses:
        reasons[f"{name}/source-false-{witness}"] = family.source_reason(
            texts[f"{name}/source-false-{witness}"], witness
        )
    for label in family.mutants:
        reasons[f"{name}/source-{label}"] = family.source_reason(
            texts[f"{name}/source-{label}"], label
        )

    positive = output / "controls.bendtt"
    exports = {label: output / f"{label}-operation.bendtt" for label in family.mutants}
    parallel(
        [
            (
                f"{name}/verdict-consumers",
                [*bend, experiment / "consumers.bend", "--verdict"],
                0,
            ),
            (
                f"{name}/verdict-controls",
                [*bend, experiment / "controls.bend", "--verdict"],
                0,
            ),
            (
                f"{name}/export-controls",
                [*bend, experiment / "controls.bend", "-o", positive],
                0,
            ),
            *(
                (
                    f"{name}/export-{label}-operation",
                    [*bend, copies[label] / family.entry_path, "-o", exports[label]],
                    0,
                )
                for label in family.mutants
            ),
        ]
    )
    text = positive.read_text()
    (output / "export-statements.txt").write_text(
        "\n\n".join(family.check_export(text, contracts.catalog).values()) + "\n"
    )
    artifacts = {
        witness: family.kernel_twin(
            text, witness, contracts.catalog, contracts.replace_declaration
        )
        for witness in family.witnesses
    }
    for label in family.mutants:
        artifacts[label] = reads.transplant(
            text, exports[label].read_text(), contracts.catalog, family.prefix
        )
        reasons[f"{name}/kernel-replaced-{label}"] = family.replaced(
            text, artifacts[label], contracts.catalog, label
        )
    for label, artifact in artifacts.items():
        (output / f"false-{label}.bendtt").write_text(artifact)
    texts = parallel(
        [
            (f"{name}/kernel-controls", [kernel, positive], 0),
            *(
                (
                    f"{name}/kernel-false-{label}",
                    [kernel, output / f"false-{label}.bendtt"],
                    1,
                )
                for label in artifacts
            ),
        ]
    )
    for label in artifacts:
        reasons[f"{name}/kernel-false-{label}"] = family.kernel_reason(
            texts[f"{name}/kernel-false-{label}"], label
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bend-root", type=Path, default=ROOT / ".tools/bend")
    parser.add_argument("--kernel", type=Path, default=ROOT / ".tools/kernel/bendtt")
    parser.add_argument(
        "--workers",
        type=int,
        default=FAMILY_WORKERS,
        help=f"Commands to run at once (default {FAMILY_WORKERS} for the {cpus()} "
        "CPUs this process may use).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers needs at least one worker")
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
    reasons: dict[str, str] = {}
    summary = {
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "repository_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "src", "verification/families"],
                cwd=ROOT,
                text=True,
            )
        ),
        "bend_commit": subprocess.check_output(
            ["git", "-C", str(bend_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "bend_is_repository_pin": pinned,
        "workers": args.workers,
        "commands": records,
        "rejection_reasons": reasons,
    }

    # The families run at once, and every command holds one of `workers`
    # slots, so the run never has more than `workers` commands going.
    slots = threading.BoundedSemaphore(args.workers)

    def run(name: str, command: list, expected_code: int) -> str:
        with slots:
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
            seconds = time.monotonic() - start
        text = result.stdout + result.stderr
        (output / f"{name}.out").write_text(text)
        records.append(
            {
                "name": name,
                "command": [str(part) for part in command],
                "exit_code": result.returncode,
                "expected_code": expected_code,
                "seconds": round(seconds, 1),
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

    def parallel(jobs: list[tuple[str, list, int]]) -> dict[str, str]:
        with ThreadPoolExecutor(len(jobs)) as pool:
            return dict(
                zip(
                    (name for name, _, _ in jobs),
                    pool.map(lambda job: run(*job), jobs),
                    strict=True,
                )
            )

    bend = ["bun", str(bend_root / "bend2/main.ts")]
    snapshot = output / "snapshot"
    shutil.copytree(ROOT / "src", snapshot / "src")
    for family in FAMILIES:
        shutil.copytree(
            ROOT / family.experiment,
            snapshot / family.experiment,
            ignore=shutil.ignore_patterns("false-*", "entry.bend"),
        )
    try:
        # Each family fills its own reasons; they merge in family order, so
        # summary.json lists them in the order a one-family-at-a-time run did.
        found: list[dict[str, str]] = [{} for _ in FAMILIES]
        try:
            with ThreadPoolExecutor(len(FAMILIES)) as pool:
                for _ in pool.map(
                    lambda family, into: family_controls(
                        family, snapshot, output, bend, kernel, parallel, into
                    ),
                    FAMILIES,
                    found,
                ):
                    pass
        finally:
            for part in found:
                reasons.update(part)
        summary["passed"] = True
    except (GateError, subprocess.TimeoutExpired) as error:
        summary["passed"] = False
        summary["error"] = str(error)
        raise
    finally:
        records.sort(key=lambda row: row["name"])
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
