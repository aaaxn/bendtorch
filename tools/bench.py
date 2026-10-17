"""Time fixed workloads on the native build and check their output.

    uv run --locked python -m tools.bench [--only add,matmul] [--runs 5]
        [--cpu 24] [--cpus 24-31 --threads 8] [--binary PATH]

`experiments/bench/bench.bend` builds each workload's inputs from a fixed
formula, runs the workload once untimed, then `reps` times between two clock
reads, and prints the milliseconds and a digest of the last output's words.
This command runs it `--runs` times on one thread pinned to `--cpu`, then
`--runs` times on `--threads` threads pinned to `--cpus`, and prints one line
per workload: the median time of one call with the range over the runs at
each thread count, the speedup, the time per unit of work at one thread, the
peak memory and the load average. Every run's digest must equal the one in
`experiments/bench/expected.txt`; a changed output exits 1. `--record FILE`
writes the digests to a new file instead of checking them.
"""

import argparse
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tools.common import ROOT, bend, environment

SOURCE = Path("experiments/bench/bench.bend")
EXPECTED = ROOT / "experiments/bench/expected.txt"


@dataclass(frozen=True)
class Workload:
    name: str
    n: int
    shape: str
    reps: int
    work: int
    unit: str


MNIST_MACS = 3 * (64 * 784 * 128 + 64 * 128 * 10)
MLP_MACS = 2 * (64 * 10 * 32 + 64 * 32 * 64)
M = 1 << 20

WORKLOADS = [
    Workload("add", 1 << 14, "[16384]", 300, 1 << 14, "cell"),
    Workload("sub", 1 << 14, "[16384]", 300, 1 << 14, "cell"),
    Workload("mul", 1 << 14, "[16384]", 300, 1 << 14, "cell"),
    Workload("add", M, "[1048576]", 8, M, "cell"),
    Workload("sub", M, "[1048576]", 8, M, "cell"),
    Workload("mul", M, "[1048576]", 8, M, "cell"),
    Workload("add_bcast", 1024, "[1024,1024]+[1024]", 8, M, "cell"),
    Workload("neg", M, "[1048576]", 8, M, "cell"),
    Workload("relu", M, "[1048576]", 8, M, "cell"),
    Workload("clone", M, "[1048576]", 300, M, "word"),
    Workload("broadcast_to", 1024, "[1024]->[1024,1024]", 1, M, "cell"),
    Workload("reshape", 1024, "[1024,1024]<->[1048576]", 100000, 2, "call"),
    Workload("transpose", 1024, "[1024,1024]", 1, M, "cell"),
    Workload("permute", 1024, "[4,1024,256] (2,0,1)", 1, M, "cell"),
    Workload("sum", M, "[1048576]", 2, M, "cell"),
    Workload("reduce_cols", 1024, "[1024,1024]->[1024]", 4, M, "cell"),
    Workload("reduce_rows", 1024, "[1024,1024]->[1024,1]", 8, M, "cell"),
    Workload("sum_to_size_cols", 1024, "[1024,1024]->[1024]", 4, M, "cell"),
    Workload("sum_to_size_rows", 1024, "[1024,1024]->[1024,1]", 8, M, "cell"),
    Workload("matmul", 256, "[256,256]@[256,256]", 1, 256**3, "mac"),
    Workload("matmul_nt", 256, "[256,256]@[256,256]^T", 1, 256**3, "mac"),
    Workload("matmul_tn", 256, "[256,256]^T@[256,256]", 1, 256**3, "mac"),
    Workload("linear_fwd", 784, "x[64,784] W[128,784]", 2, 64 * 784 * 128, "mac"),
    Workload("linear_bwd", 784, "x[64,784] W[128,784]", 1, 2 * 64 * 784 * 128, "mac"),
    Workload("relu_bwd", M, "[1048576]", 8, M, "cell"),
    Workload("mse_fwd", M, "[1048576]", 1, M, "cell"),
    Workload("mse_bwd", M, "[1048576]", 4, M, "cell"),
    Workload("ce_fwd", 1 << 16, "[65536,10]", 1, 10 << 16, "cell"),
    Workload("ce_bwd", 1 << 16, "[65536,10]", 1, 10 << 16, "cell"),
    Workload("sgd", M, "[1048576]", 8, M, "param"),
    Workload("mlp_bwd", 64, "64-32-10, batch 64", 40, MLP_MACS, "mac"),
    Workload("mnist_step", 64, "784-128-10, batch 64", 1, MNIST_MACS, "mac"),
    Workload("mul_bwd", M, "[1048576]*[1048576]", 4, M, "cell"),
    Workload("sum_bwd", M, "sum([1048576])", 1, M, "cell"),
    Workload("matmul_bwd", 256, "[256,256]@[256,256]", 1, 2 * 256**3, "mac"),
]


@dataclass(frozen=True)
class Sample:
    ms: int
    digest: str
    rss_mb: float


def key(w: Workload) -> str:
    return f"{w.name} {w.n}"


def parse(text: str) -> tuple[int, str]:
    """The milliseconds and the digest a bench run printed."""
    fields = dict(
        row.split(" ", 1)
        for row in text.splitlines()
        if row.startswith(("ms ", "digest "))
    )
    return int(fields["ms"]), fields["digest"]


def build(root: Path, binary: Path) -> Path:
    """Compile root's experiments/bench/bench.bend to a native binary."""
    binary.parent.mkdir(parents=True, exist_ok=True)
    tools = root / ".tools"
    subprocess.run(
        bend(tools, root / SOURCE, "-o", binary),
        env=environment(tools),
        check=True,
        capture_output=True,
    )
    return binary


def run(binary: Path, w: Workload, threads: int, cpus: str) -> Sample:
    command = ["taskset", "-c", cpus, str(binary), w.name, str(w.n), str(w.reps)]
    command += ["--threads", str(threads), "--gpu", "off"]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert child.stdout is not None
    out = child.stdout.read().decode()
    child.stdout.close()
    _, status, usage = os.wait4(child.pid, 0)
    child.returncode = os.waitstatus_to_exitcode(status)
    if child.returncode != 0:
        raise SystemExit(f"{key(w)}: exit {child.returncode}: {out.strip()}")
    ms, digest = parse(out)
    return Sample(ms, digest, usage.ru_maxrss / 1024)


def spread(samples: list[Sample], reps: int) -> tuple[float, float, float]:
    """Median, min and max milliseconds of one call."""
    per = [s.ms / reps for s in samples]
    return statistics.median(per), min(per), max(per)


def load() -> str:
    return Path("/proc/loadavg").read_text().split()[0]


def line(
    w: Workload, one: list[Sample], many: list[Sample], threads: int, check: str
) -> str:
    m1, lo1, hi1 = spread(one, w.reps)
    text = f"{w.name:<17} {w.shape:<24} t1 {m1:10.4g} ms [{lo1:.4g}-{hi1:.4g}]"
    if many:
        mn, lon, hin = spread(many, w.reps)
        speedup = m1 / mn if mn > 0 else float("inf")
        text += f"  t{threads} {mn:10.4g} ms [{lon:.4g}-{hin:.4g}] x{speedup:.2f}"
    ns = m1 * 1e6 / w.work
    rss = max(s.rss_mb for s in one)
    text += f"  {ns:9.2f} ns/{w.unit}  work {w.work} {w.unit}s  {rss:.0f} MB"
    text += f"  load {load()}  {check}"
    return text


def expected_digests(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    pairs = (row.rsplit(" ", 1) for row in path.read_text().splitlines() if row.strip())
    return {name: digest for name, digest in pairs}


def verdict(w: Workload, samples: list[Sample], expected: dict[str, str]) -> str:
    """`ok` when every run printed the expected digest, else what differed."""
    seen = sorted({s.digest for s in samples})
    want = expected.get(key(w))
    if want is None:
        return "no expected digest"
    if seen != [want]:
        return f"CHANGED {','.join(seen)} != {want}"
    return "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", default="", help="comma-separated workload names")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--cpu", default="24", help="CPU of the one-thread runs")
    parser.add_argument(
        "--cpus", default="24-31", help="CPUs of the multithreaded runs"
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="0 skips the multithreaded runs"
    )
    parser.add_argument("--binary", type=Path, help="a built bench binary to reuse")
    parser.add_argument("--build", type=Path, default=ROOT / ".evidence/bench/bench")
    parser.add_argument("--expected", type=Path, default=EXPECTED)
    parser.add_argument(
        "--record", type=Path, help="write the digests to this new file"
    )
    args = parser.parse_args(argv)
    names = set(filter(None, args.only.split(",")))
    chosen = [w for w in WORKLOADS if not names or w.name in names]
    if names - {w.name for w in WORKLOADS}:
        parser.error(
            f"unknown workloads: {sorted(names - {w.name for w in WORKLOADS})}"
        )
    if args.record is not None and args.record.exists():
        parser.error(f"refusing to overwrite {args.record}")
    binary = args.binary or build(ROOT, args.build)
    expected = expected_digests(args.expected)
    recorded = []
    failed = False
    for w in chosen:
        one = [run(binary, w, 1, args.cpu) for _ in range(args.runs)]
        many = (
            [run(binary, w, args.threads, args.cpus) for _ in range(args.runs)]
            if args.threads
            else []
        )
        if args.record is not None:
            digests = sorted({s.digest for s in one + many})
            check = "recorded" if len(digests) == 1 else f"DIFFER {','.join(digests)}"
            recorded.append(f"{key(w)} {digests[0]}")
        else:
            check = verdict(w, one + many, expected)
        failed = failed or check not in ("ok", "recorded")
        print(line(w, one, many, args.threads, check), flush=True)
    if args.record is not None:
        args.record.write_text("".join(row + "\n" for row in recorded))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
