"""Run every test module under tests/ in its own process, several at once.

`python -m unittest discover -s tests` runs the same tests one module after
another. This runner prints each module's output as one block, in name order,
and exits non-zero if any module fails.
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from tools.common import ROOT, TEST_JOBS, cpus


def run_module(name: str) -> tuple[str, int, str]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", name],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return name, result.returncode, result.stdout + result.stderr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--jobs",
        type=int,
        default=TEST_JOBS,
        help=f"Test modules to run at once (default {TEST_JOBS} for the {cpus()} "
        "CPUs this process may use).",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs needs at least one job")
    modules = sorted(path.name for path in (ROOT / "tests").glob("test_*.py"))
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(run_module, modules))
    failed = []
    for name, code, output in results:
        print(f"== {name} (exit {code})\n{output}", flush=True)
        if code:
            failed.append(name)
    if failed:
        print(f"FAILED modules: {', '.join(failed)}")
        sys.exit(1)
    print(f"All {len(modules)} test modules passed.")


if __name__ == "__main__":
    main()
