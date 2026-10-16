"""Compare two git revisions on the bench workloads, alternating runs.

    uv run --locked python -m tools.bench_compare REV_A REV_B --work DIR
        [--only matmul,add] [--rounds 5] [--cpu 24] [--threads 1]

Each revision is checked out in its own detached worktree under DIR and
built with this tree's `experiments/bench/bench.bend` when the revision has
none. The runs alternate, A then B in even rounds and B then A in odd ones,
on the same CPUs. Each workload gets one line: the median time of one call
with its range for A and for B, the median and range of the per-round ratio
B / A, and whether the two printed the same digest, the one in this tree's
`experiments/bench/expected.txt`.
"""

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

from tools.bench import (
    EXPECTED,
    SOURCE,
    WORKLOADS,
    Sample,
    Workload,
    build,
    expected_digests,
    key,
    run,
)
from tools.common import ROOT


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def checkout(rev: str, place: Path) -> Path:
    """A built bench binary of `rev`, its worktree at `place`."""
    git("worktree", "add", "--detach", str(place), rev)
    (place / ".tools").symlink_to((ROOT / ".tools").resolve())
    source = place / SOURCE
    copied = not source.exists()
    if copied:
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text((ROOT / SOURCE).read_text())
    try:
        return build(place, place.parent / f"{place.name}.bench")
    finally:
        if copied:
            source.unlink()
        (place / ".tools").unlink()
        git("worktree", "remove", str(place))


def ratio_line(
    w: Workload, a: list[Sample], b: list[Sample], expected: dict[str, str]
) -> str:
    per_a = [s.ms / w.reps for s in a]
    per_b = [s.ms / w.reps for s in b]
    ratios = [y / x for x, y in zip(per_a, per_b, strict=True) if x > 0]
    digests = {s.digest for s in a} | {s.digest for s in b}
    same = "same output" if len(digests) == 1 else f"OUTPUT DIFFERS {sorted(digests)}"
    if expected.get(key(w)) not in digests or len(digests) != 1:
        same += f" (expected {expected.get(key(w))})"
    text = f"{w.name:<17} {w.shape:<24}"
    text += (
        f" A {statistics.median(per_a):10.3f} ms [{min(per_a):.3f}-{max(per_a):.3f}]"
    )
    text += (
        f"  B {statistics.median(per_b):10.3f} ms [{min(per_b):.3f}-{max(per_b):.3f}]"
    )
    if ratios:
        middle = statistics.median(ratios)
        text += f"  B/A {middle:.3f} [{min(ratios):.3f}-{max(ratios):.3f}]"
    return f"{text}  {same}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument(
        "--work", type=Path, required=True, help="a new directory for the two builds"
    )
    parser.add_argument("--only", default="")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--cpu", default="24", help="CPUs of every run")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.rounds < 5:
        parser.error("--rounds needs at least 5")
    names = set(filter(None, args.only.split(",")))
    chosen = [w for w in WORKLOADS if not names or w.name in names]
    args.work.mkdir(parents=True)
    binaries = {
        side: checkout(rev, args.work / side)
        for side, rev in (("a", args.a), ("b", args.b))
    }
    expected = expected_digests(EXPECTED)
    failed = False
    for w in chosen:
        samples: dict[str, list[Sample]] = {"a": [], "b": []}
        for r in range(args.rounds):
            for side in ("a", "b") if r % 2 == 0 else ("b", "a"):
                samples[side].append(run(binaries[side], w, args.threads, args.cpu))
        text = ratio_line(w, samples["a"], samples["b"], expected)
        failed = failed or "DIFFERS" in text
        print(text, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
