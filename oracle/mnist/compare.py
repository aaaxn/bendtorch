"""Trains the MLP with the Bend program and with PyTorch's copy over a few
seeds and compares them.

For each seed it writes the shared initial parameters and training order
(train.py prepare), runs the native Bend binary built from
experiments/mnist/train.bend and train.py on them, and checks two things:
the loss of every step agrees within a relative tolerance,
|bend - torch| / |torch| <= --tolerance, and within --early-tolerance over
the first --early-steps steps; and the final test accuracy agrees within a
band of one binomial standard error, |bend - torch| <= ceil(sqrt(p (1 - p)
n)) correct rows with p PyTorch's accuracy on the n = 10000 test rows. It
prints a table and exits 1 if a seed fails a check.

Run from oracle/, with the binary built as experiments/mnist/README.md says:
  uv run --frozen python -I mnist/compare.py BINARY MNIST_DIR WORK_DIR [--seeds 0 1 2]
    [--steps 937] [--batch 64] [--lr 0.1] [--threads 16] [--kernel default|native]

--kernel default runs PyTorch with ATEN_CPU_CAPABILITY=default, its
non-vectorized kernels; native lets PyTorch pick the CPU's best.
"""

import argparse
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The checks. experiments/mnist/README.md says how they were chosen.
TOLERANCE = 1e-2
EARLY_TOLERANCE = 1e-4
EARLY_STEPS = 100
TEST_ROWS = 10000

WINDOWS = [(0, 10), (10, 100), (100, 300), (300, None)]


def value(word):
    return struct.unpack("<f", struct.pack("<I", int(word)))[0]


def parse(text):
    losses = []
    correct = None
    for line in text.splitlines():
        parts = line.split()
        if parts[:1] == ["loss"]:
            if int(parts[1]) != len(losses):
                raise SystemExit(f"step {parts[1]} out of order")
            losses.append(int(parts[2]))
        elif parts[:1] == ["correct"]:
            correct = int(parts[1])
    return losses, correct


def run(command, env=None):
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


def relative(bend, torch):
    b, t = value(bend), value(torch)
    return abs(b - t) / abs(t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("mnist", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=937)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", default="0.1")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--kernel", choices=["default", "native"], default="default")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--early-tolerance", type=float, default=EARLY_TOLERANCE)
    parser.add_argument("--early-steps", type=int, default=EARLY_STEPS)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    epochs = -(-args.steps * args.batch // ((60000 // args.batch) * args.batch))
    env = dict(os.environ)
    if args.kernel == "default":
        env["ATEN_CPU_CAPABILITY"] = "default"
    header = [
        "seed",
        "steps",
        "equal words",
        "max rel",
        "at step",
        f"max rel 0-{args.early_steps - 1}",
    ]
    header += [
        f"max rel {a}-{(b or args.steps) - 1}" for a, b in WINDOWS if a < args.steps
    ]
    header += [
        "correct bend",
        "correct torch",
        "band",
        "bend ms/step",
        "torch ms/step",
        "verdict",
    ]
    rows = []
    failed = False
    for seed in args.seeds:
        here = sys.executable
        run(
            [
                here,
                "-I",
                str(HERE / "train.py"),
                "prepare",
                str(seed),
                str(args.work),
                "--epochs",
                str(epochs),
                "--batch",
                str(args.batch),
            ]
        )
        init = args.work / f"init-{seed}.bin"
        order = args.work / f"order-{seed}-b{args.batch}.bin"
        common = [
            str(args.mnist),
            str(init),
            str(order),
            str(args.batch),
            str(args.steps),
            args.lr,
        ]
        bend_out, bend_ms = run(
            [str(args.binary), *common, "--threads", str(args.threads), "--gpu", "off"]
        )
        torch_out, torch_ms = run(
            [here, "-I", str(HERE / "train.py"), "train", *common], env
        )
        (args.work / f"bend-{seed}.out").write_text(bend_out)
        (args.work / f"torch-{args.kernel}-{seed}.out").write_text(torch_out)
        bend, bend_correct = parse(bend_out)
        torch, torch_correct = parse(torch_out)
        if (
            len(bend) != args.steps
            or len(torch) != args.steps
            or bend_correct is None
            or torch_correct is None
        ):
            raise SystemExit(f"seed {seed}: incomplete output")
        rel = [relative(b, t) for b, t in zip(bend, torch)]
        worst = max(range(len(rel)), key=rel.__getitem__)
        windows = [max(rel[a:b]) for a, b in WINDOWS if a < args.steps]
        early = max(rel[: args.early_steps])
        p = torch_correct / TEST_ROWS
        band = math.ceil(math.sqrt(p * (1 - p) * TEST_ROWS))
        ok = (
            max(rel) <= args.tolerance
            and early <= args.early_tolerance
            and abs(bend_correct - torch_correct) <= band
        )
        failed |= not ok
        rows.append(
            [
                str(seed),
                str(args.steps),
                str(sum(b == t for b, t in zip(bend, torch))),
                f"{rel[worst]:.2e}",
                str(worst),
                f"{early:.2e}",
                *[f"{w:.2e}" for w in windows],
                str(bend_correct),
                str(torch_correct),
                str(band),
                f"{bend_ms / args.steps:.1f}",
                f"{torch_ms / args.steps:.2f}",
                "pass" if ok else "FAIL",
            ]
        )
    print(
        f"tolerance {args.tolerance:g} relative per step, {args.early_tolerance:g} over the first "
        f"{args.early_steps} steps, band one binomial standard error of PyTorch's accuracy; "
        f"PyTorch kernel {args.kernel}; Bend {args.threads} threads; batch {args.batch}, lr {args.lr}"
    )
    print("| " + " | ".join(header) + " |")
    print("|" + " --- |" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
