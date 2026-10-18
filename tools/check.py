import argparse
from pathlib import Path

from tools import autograd, common, shape, storage, tensor, toolchain
from tools.common import (
    CHECK_JOBS,
    ROOT,
    THREADS,
    GateError,
    cpus,
    set_cache,
    set_jobs,
    set_threads,
)

SUITES = {
    "toolchain": toolchain.run_gate,
    "shape": shape.run_gate,
    "storage": storage.run_gate,
    "tensor": tensor.run_gate,
    "autograd": autograd.run_gate,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the gate of each named suite; `all` runs every suite."
    )
    parser.add_argument("suites", nargs="+", choices=[*SUITES, "all"])
    parser.add_argument("--tool-root", type=Path, default=ROOT / ".tools")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--jobs",
        type=int,
        default=CHECK_JOBS,
        help=f"Gate commands to run at once (default {CHECK_JOBS} for the {cpus()} "
        "CPUs this process may use; 1 runs them in order).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=THREADS,
        help=f"Threads for each native Bend run (default {THREADS}, at most 8; "
        "output is the same).",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="Replay unchanged checker and kernel results from this directory "
        "(tools/cache.py). Without it every command runs.",
    )
    args = parser.parse_args()
    set_cache(args.cache)
    set_jobs(args.jobs)
    set_threads(args.threads)
    names = list(SUITES) if "all" in args.suites else list(dict.fromkeys(args.suites))
    output = args.output.resolve()
    if output.exists():
        raise GateError(f"Refusing to overwrite evidence: {output}")
    output.mkdir(parents=True)
    for name in names:
        SUITES[name](args.tool_root.resolve(), output / name)
    if common.CACHE is not None:
        print(f"cache: {common.CACHE.summary()}")
    print(f"Gates passed ({', '.join(names)}): {output}")


if __name__ == "__main__":
    main()
