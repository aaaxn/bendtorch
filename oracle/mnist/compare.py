"""Compares the MLP trained by the Bend program with PyTorch's copy.

`run` trains both over a few seeds: for each seed it writes the shared
initial parameters and training order (train.py prepare), runs the native
Bend binary built from experiments/mnist/train.bend and train.py on them,
and compares their outputs. `check` compares two saved outputs.

Each output must be well formed: exactly the expected number of
`loss STEP WORD` lines, numbered 0, 1, ... in order, and exactly one
`correct N ROWS` line with ROWS = 10000 and 0 <= N <= ROWS. Then each step is
checked on its own: both losses finite, and |bend - torch| / |torch| at most
--early-tolerance for the first --early-steps steps and at most --tolerance
after them. The test counts must agree within a band scaled by the
binomial standard error of PyTorch's accuracy p on the n = 10000 test rows,
|bend - torch| <= ceil(sqrt(p (1 - p) n)) rows. The band is an
acceptance rule, not a significance test. These checks are the MNIST
instance of level 3 of the numeric policy in AGENTS.md. It prints a table,
then every failed check, and exits 1 if any run fails one.

Run from oracle/, with the binary built as docs/mnist.md says:
  uv run --frozen python -I mnist/compare.py run BINARY MNIST_DIR WORK_DIR
    [--seeds 0 1 2] [--steps 937] [--batch 64] [--lr 0.1] [--threads 16]
    [--kernel default|native]
  uv run --frozen python -I mnist/compare.py check BEND_OUTPUT TORCH_OUTPUT
    [--steps 937]

--kernel default runs PyTorch with ATEN_CPU_CAPABILITY=default, its
non-vectorized kernels; native lets PyTorch pick the CPU's best.
"""

import argparse
import math
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The checks: level 3 of AGENTS.md's numeric policy. docs/mnist.md says how
# they were fixed from PyTorch against itself.
TOLERANCE = 1e-2
EARLY_TOLERANCE = 1e-4
EARLY_STEPS = 100
TEST_ROWS = 10000

WINDOWS = [(0, 10), (10, 100), (100, 300), (300, None)]
SHOWN_FAILURES = 5


class Malformed(Exception):
    pass


def value(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]


def natural(text: str) -> int | None:
    return int(text) if re.fullmatch(r"0|[1-9][0-9]*", text) else None


def parse(text: str, steps: int, side: str) -> tuple[list[int], int]:
    """The loss words and the correct count of one output, or Malformed."""
    losses: list[int] = []
    counts: list[tuple[int, int]] = []
    for number, line in enumerate(text.splitlines(), 1):
        parts = line.split(" ")
        fields = [natural(part) for part in parts[1:]]
        if len(parts) != 3 or None in fields:
            raise Malformed(f"{side} line {number}: not a result line: {line!r}")
        first, second = fields
        assert first is not None and second is not None
        if parts[0] == "loss" and not counts:
            if first != len(losses):
                raise Malformed(f"{side} line {number}: step {first} out of order")
            if second >= 1 << 32:
                raise Malformed(f"{side} line {number}: {second} is not a 32-bit word")
            losses.append(second)
        elif parts[0] == "correct" and not counts:
            counts.append((first, second))
        else:
            raise Malformed(f"{side} line {number}: unexpected line: {line!r}")
    if len(losses) != steps:
        raise Malformed(f"{side}: {len(losses)} loss lines, {steps} expected")
    if not counts:
        raise Malformed(f"{side}: no correct line")
    correct, rows = counts[0]
    if rows != TEST_ROWS or correct > rows:
        raise Malformed(f"{side}: correct {correct} {rows}, not N of {TEST_ROWS}")
    return losses, correct


def relative(bend: float, torch: float) -> float:
    if torch == 0:
        return 0.0 if bend == 0 else math.inf
    return abs(bend - torch) / abs(torch)


def compared(
    bend_text: str,
    torch_text: str,
    steps: int,
    tolerance: float = TOLERANCE,
    early_tolerance: float = EARLY_TOLERANCE,
    early_steps: int = EARLY_STEPS,
) -> dict:
    """Every check of one pair of outputs: `failures` lists each failed one."""
    try:
        bend, bend_correct = parse(bend_text, steps, "bend")
        torch, torch_correct = parse(torch_text, steps, "torch")
    except Malformed as error:
        return {"failures": [str(error)]}
    failures = []
    rel = []
    for step, (b, t) in enumerate(zip(bend, torch, strict=True)):
        x, y = value(b), value(t)
        if not (math.isfinite(x) and math.isfinite(y)):
            failures.append(f"step {step}: a loss is not finite (bend {x}, torch {y})")
            rel.append(math.inf)
            continue
        difference = relative(x, y)
        rel.append(difference)
        limit = early_tolerance if step < early_steps else tolerance
        if not difference <= limit:
            failures.append(
                f"step {step}: relative difference {difference:.2e} > {limit:g}"
            )
    p = torch_correct / TEST_ROWS
    band = math.ceil(math.sqrt(p * (1 - p) * TEST_ROWS))
    if abs(bend_correct - torch_correct) > band:
        failures.append(
            f"correct {bend_correct} against {torch_correct}: more than {band} rows apart"
        )
    worst = max(range(steps), key=rel.__getitem__) if steps else 0
    return {
        "failures": failures,
        "equal": sum(b == t for b, t in zip(bend, torch, strict=True)),
        "worst": worst,
        "max": rel[worst] if steps else 0.0,
        "early": max(rel[:early_steps], default=0.0),
        "windows": [max(rel[a:b], default=0.0) for a, b in WINDOWS if a < steps],
        "bend_correct": bend_correct,
        "torch_correct": torch_correct,
        "band": band,
    }


def run(command: list[str], env: dict | None = None) -> tuple[str, int | None]:
    done = subprocess.run(command, capture_output=True, text=True, env=env)
    if done.returncode != 0:
        raise SystemExit(
            f"{' '.join(map(str, command))} exited {done.returncode}:\n{done.stderr}"
        )
    millis = next(
        (
            int(line.split()[1])
            for line in done.stderr.splitlines()
            if line.startswith("train_ms")
        ),
        None,
    )
    return done.stdout, millis


def per_step(millis: int | None, steps: int, digits: int) -> str:
    return "-" if millis is None or steps == 0 else f"{millis / steps:.{digits}f}"


def report(results: list[tuple[str, dict, str, str]], args) -> int:
    header = ["run", "steps", "equal words", "max rel", "at step"]
    header += [f"max rel 0-{args.early_steps - 1}"]
    header += [
        f"max rel {a}-{(b or args.steps) - 1}" for a, b in WINDOWS if a < args.steps
    ]
    header += ["correct bend", "correct torch", "band", "bend ms/step"]
    header += ["torch ms/step", "verdict"]
    print(
        f"tolerance {args.tolerance:g} relative at every step after the first "
        f"{args.early_steps}, {args.early_tolerance:g} at each of those; band one "
        "binomial standard error of PyTorch's accuracy"
    )
    print("| " + " | ".join(header) + " |")
    print("|" + " --- |" * len(header))
    for label, result, bend_ms, torch_ms in results:
        verdict = "FAIL" if result["failures"] else "pass"
        if "equal" not in result:
            cells = ["malformed"] + ["-"] * (len(header) - 3)
        else:
            cells = [
                str(args.steps),
                str(result["equal"]),
                f"{result['max']:.2e}",
                str(result["worst"]),
                f"{result['early']:.2e}",
                *[f"{w:.2e}" for w in result["windows"]],
                str(result["bend_correct"]),
                str(result["torch_correct"]),
                str(result["band"]),
                bend_ms,
                torch_ms,
            ]
        print("| " + " | ".join([label, *cells, verdict]) + " |")
    failed = False
    for label, result, _, _ in results:
        failures = result["failures"]
        failed |= bool(failures)
        for failure in failures[:SHOWN_FAILURES]:
            print(f"{label}: {failure}")
        if len(failures) > SHOWN_FAILURES:
            print(f"{label}: {len(failures) - SHOWN_FAILURES} more failed checks")
    return 1 if failed else 0


def train_both(args) -> int:
    args.work.mkdir(parents=True, exist_ok=True)
    epochs = -(-args.steps * args.batch // ((60000 // args.batch) * args.batch))
    env = dict(os.environ)
    if args.kernel == "default":
        env["ATEN_CPU_CAPABILITY"] = "default"
    here = sys.executable
    results = []
    for seed in args.seeds:
        prepare = [here, "-I", str(HERE / "train.py"), "prepare", str(seed)]
        prepare += [str(args.work), "--epochs", str(epochs), "--batch", str(args.batch)]
        run(prepare)
        init = args.work / f"init-{seed}.bin"
        order = args.work / f"order-{seed}-b{args.batch}.bin"
        common = [str(args.mnist), str(init), str(order), str(args.batch)]
        common += [str(args.steps), args.lr]
        threads = ["--threads", str(args.threads), "--gpu", "off"]
        bend_out, bend_ms = run([str(args.binary), *common, *threads])
        torch_out, torch_ms = run(
            [here, "-I", str(HERE / "train.py"), "train", *common], env
        )
        (args.work / f"bend-{seed}.out").write_text(bend_out)
        (args.work / f"torch-{args.kernel}-{seed}.out").write_text(torch_out)
        result = compared(
            bend_out,
            torch_out,
            args.steps,
            args.tolerance,
            args.early_tolerance,
            args.early_steps,
        )
        bend_step = per_step(bend_ms, args.steps, 1)
        torch_step = per_step(torch_ms, args.steps, 2)
        results.append((f"seed {seed}", result, bend_step, torch_step))
    print(
        f"PyTorch kernel {args.kernel}; Bend {args.threads} threads; "
        f"batch {args.batch}, lr {args.lr}"
    )
    return report(results, args)


def check_saved(args) -> int:
    result = compared(
        args.bend.read_text(),
        args.torch.read_text(),
        args.steps,
        args.tolerance,
        args.early_tolerance,
        args.early_steps,
    )
    print(f"bend {args.bend}; torch {args.torch}")
    return report([(args.bend.name, result, "-", "-")], args)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    both = sub.add_parser("run")
    both.add_argument("binary", type=Path)
    both.add_argument("mnist", type=Path)
    both.add_argument("work", type=Path)
    both.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    both.add_argument("--batch", type=int, default=64)
    both.add_argument("--lr", default="0.1")
    both.add_argument("--threads", type=int, default=16)
    both.add_argument("--kernel", choices=["default", "native"], default="default")
    saved = sub.add_parser("check")
    saved.add_argument("bend", type=Path)
    saved.add_argument("torch", type=Path)
    for command in (both, saved):
        command.add_argument("--steps", type=int, default=937)
        command.add_argument("--tolerance", type=float, default=TOLERANCE)
        command.add_argument("--early-tolerance", type=float, default=EARLY_TOLERANCE)
        command.add_argument("--early-steps", type=int, default=EARLY_STEPS)
    args = parser.parse_args()
    sys.exit(train_both(args) if args.command == "run" else check_saved(args))


if __name__ == "__main__":
    main()
