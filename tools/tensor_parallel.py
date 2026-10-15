import re
from pathlib import Path

from tools import common
from tools.common import ROOT, Evidence, GateError, bend

PROBE = ROOT / "probes/parallel_builds.bend"
# The thread counts every native run of the probe must agree across.
THREAD_COUNTS = (1, 8, 24)
LINE = re.compile(r"^(\S+) splits (\d+) differing (\d+) hash (\d+)$")


def depth(cells: int) -> int:
    """Storage depth of `cells` cells: ceil(log2(cells)), 0 for 0 or 1."""
    return 0 if cells <= 1 else (cells - 1).bit_length()


def matmul_splits(depth: int, inner: int) -> int:
    """`Matrix.splits`: depth + floor(log2(inner)) - 13 levels, at most 6, at
    most `depth`."""
    halvings = inner.bit_length() - 1 if inner > 0 else 0
    return min(max(depth + halvings - 13, 0), 6, depth)


# Each product the probe runs, as (rows, inner, columns).
PRODUCTS = {
    "matmul-128x64x64": (128, 64, 64),
    "matmul-96x40x80": (96, 40, 80),
    "matmul-2x3x2": (2, 3, 2),
    "matmul-64x48x64-rounding": (64, 48, 64),
}
# The product whose inputs round, so that adding a cell's terms in another
# order changes its bits; the products of multiples of 1/64 above cannot show
# a reordered sum. It must fork to check the forked build.
ROUNDING = "matmul-64x48x64-rounding"


def expected_splits(name: str) -> int:
    rows, inner, columns = PRODUCTS[name]
    return matmul_splits(depth(rows * columns), inner)


def check_lines(stdout: str) -> dict[str, int]:
    """The splits each probe line reports, after checking that every line
    names a known build, forks as `Matrix.splits` says and matches its
    sequential definition word for word."""
    splits = {}
    for line in stdout.splitlines():
        match = LINE.match(line)
        if match is None:
            raise GateError(f"Parallel probe printed an unexpected line: {line!r}")
        name, forks, differing, _ = match.groups()
        if name not in PRODUCTS or name in splits:
            raise GateError(f"Parallel probe line names an unknown build: {name}")
        if int(differing) != 0:
            raise GateError(
                f"{name}: {differing} words differ from the sequential definition"
            )
        if int(forks) != expected_splits(name):
            raise GateError(
                f"{name}: forked {forks} levels, expected {expected_splits(name)}"
            )
        splits[name] = int(forks)
    if set(splits) != set(PRODUCTS):
        raise GateError(f"Parallel probe omitted builds: {set(PRODUCTS) - set(splits)}")
    if not any(splits.values()):
        raise GateError("No parallel probe build forks")
    if not splits[ROUNDING]:
        raise GateError(f"{ROUNDING} does not fork")
    return splits


def run_gate(tool_root: Path, output: Path) -> dict:
    """Run the forked builds next to their sequential definitions at each
    thread count and require the same bytes at every count."""
    evidence = Evidence(output, tool_root)
    for mode in ("--check-only", "--verdict"):
        evidence.run(
            f"probe-{mode[2:]}",
            bend(tool_root, PROBE, mode),
            contains="ALL PROOFS CHECK",
        )
    emitted = output / "parallel_builds.c"
    evidence.run("probe-c-export", bend(tool_root, PROBE, "-o", emitted))
    executable = output / "parallel_builds"
    evidence.run(
        "probe-build",
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
    outputs = {
        threads: evidence.run(
            f"probe-threads-{threads}", common.native(executable, threads=threads)
        ).stdout
        for threads in THREAD_COUNTS
    }
    first = outputs[THREAD_COUNTS[0]]
    for threads, stdout in outputs.items():
        if stdout != first:
            raise GateError(f"Parallel probe output at {threads} threads differs")
    splits = check_lines(first)
    evidence.save()
    return {"thread_counts": list(THREAD_COUNTS), "splits": splits, "differing": 0}
