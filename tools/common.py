import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class GateError(RuntimeError):
    pass


def cpus() -> int:
    """How many CPUs this process may run on.

    This is the affinity mask, which `taskset` and cgroup cpusets narrow, not
    the machine's core count.
    """
    return len(os.sched_getaffinity(0))


def share(fraction: float) -> int:
    """`fraction` of `cpus()`, rounded down, and at least one.

    The default worker counts split the CPUs so that a full round, which runs
    the families, the contracts, two `tools.check` suites and `tools.test` at
    once, keeps about one busy process per CPU. Every Bend checker and BendTT
    command they run keeps one CPU busy.
    """
    return max(1, int(cpus() * fraction))


# Default commands at once for each runner. The families have about twice the
# contracts' work; a `tools.check` suite rarely keeps more than four commands
# busy, because one kernel check takes most of its time.
FAMILY_WORKERS = share(9 / 16)
CONTRACT_WORKERS = share(5 / 16)
CHECK_JOBS = share(1 / 16)
TEST_JOBS = share(1 / 8)


# Gate steps that do not depend on each other run through `parallel`. Every
# subprocess an Evidence runs holds one of `JOBS` slots, so a gate never runs
# more than `JOBS` commands at once. With one job, `parallel` calls its tasks
# in order on the calling thread, exactly as the gates ran before.
JOBS = 1
SLOTS = threading.BoundedSemaphore(JOBS)
# Where the running step sits in the gate's sequential order: Evidence keeps
# its records in this order, so report.json lists commands in the same order
# at any job count.
PLACE: ContextVar[tuple[int, ...]] = ContextVar("PLACE", default=())
NEXT: ContextVar[list[int] | None] = ContextVar("NEXT", default=None)


# Native Bend programs run with `THREADS` worker threads. Their output does not
# depend on the count: every native gate command, and a probe whose elementwise
# builds fork, printed the same bytes at 1 to 24 threads. Timing runs pin one
# thread so their numbers stay comparable. The gate cases run in milliseconds
# at any count, so the count costs nothing; it stays above one so the gates run
# the multithreaded runtime.
THREADS = min(8, cpus())


def set_threads(threads: int) -> None:
    global THREADS
    if threads < 1:
        raise GateError("--threads needs at least one thread")
    THREADS = threads


def native(target: Path | str, *args: str, threads: int | None = None) -> list[str]:
    """The command that runs a native Bend executable on the CPU."""
    count = THREADS if threads is None else threads
    return [str(target), "--threads", str(count), "--gpu", "off", *args]


def set_jobs(jobs: int) -> None:
    global JOBS, SLOTS
    if jobs < 1:
        raise GateError("--jobs needs at least one job")
    JOBS, SLOTS = jobs, threading.BoundedSemaphore(jobs)


def place() -> tuple[int, ...]:
    counter = NEXT.get()
    if counter is None:
        counter = [0]
        NEXT.set(counter)
    counter[0] += 1
    return (*PLACE.get(), counter[0])


def parallel(*tasks: Callable[[], Any]) -> list[Any]:
    """Run independent gate steps and return their results in task order.

    Every task runs to the end, and the first failure in task order is
    raised, so a failing gate fails with the error the sequential order
    would reach first among the steps that ran.
    """
    base = place()

    def step(index: int, task: Callable[[], Any]) -> Any:
        PLACE.set((*base, index))
        NEXT.set([0])
        return task()

    if JOBS == 1:
        return [copy_context().run(step, i, task) for i, task in enumerate(tasks)]
    with ThreadPoolExecutor(max_workers=len(tasks) or 1) as pool:
        futures = [
            pool.submit(copy_context().run, step, i, task)
            for i, task in enumerate(tasks)
        ]
    return [future.result() for future in futures]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest() -> dict:
    return json.loads((ROOT / "verification/references.json").read_text())


def environment(tool_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["BEND_NO_TELEMETRY"] = "1"
    env["BENDTT"] = str(tool_root / "kernel/bendtt")
    env["CC"] = str(tool_root / "llvm/usr/bin/clang-19")
    # Bend links CUDA into a `!` binary when $CUDA_HOME has nvrtc.h. Since
    # bendlang/bend 51ab8e8 it also tries /usr/local/cuda and /opt/cuda, so the
    # toolchain gate compiles its `!` program from Bend's C with the pinned clang
    # instead.
    env["CUDA_HOME"] = str(tool_root / "no-cuda")
    env["LD_LIBRARY_PATH"] = str(tool_root / "llvm/usr/lib/x86_64-linux-gnu")
    env["LEAN_STACK_SIZE_KB"] = "4194304"
    return env


def bend(tool_root: Path, *args: str | Path) -> list[str]:
    bun = shutil.which("bun")
    if bun is None:
        raise GateError("Bun 1.3.14 is required")
    return [bun, str(tool_root / "bend/bend2/main.ts"), *map(str, args)]


class Evidence:
    def __init__(self, directory: Path, tool_root: Path):
        if directory.exists():
            raise GateError(f"Refusing to overwrite evidence: {directory}")
        directory.mkdir(parents=True)
        self.directory = directory
        self.tool_root = tool_root
        self.records: list[dict] = []
        self.places: list[tuple[int, ...]] = []
        self.lock = threading.Lock()

    def record(self, entry: dict) -> None:
        with self.lock:
            self.records.append(entry)
            self.places.append(place())
            self.write_report()

    def run(
        self,
        name: str,
        command: list[str],
        *,
        expected_code: int = 0,
        contains: str | tuple[str, ...] = (),
        expected_output: str | None = None,
        timeout: int = 300,
        cwd: Path = ROOT,
    ) -> subprocess.CompletedProcess[str]:
        with SLOTS:
            start = time.monotonic()
            try:
                result = subprocess.run(
                    command,
                    cwd=cwd,
                    env=environment(self.tool_root),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                result = None
            seconds = time.monotonic() - start
        if result is None:
            self.record(
                {"name": name, "command": command, "timeout": timeout, "passed": False}
            )
            raise GateError(f"{name}: timed out, never a successful check")
        (self.directory / f"{name}.stdout").write_text(result.stdout)
        (self.directory / f"{name}.stderr").write_text(result.stderr)
        required = (contains,) if isinstance(contains, str) else contains
        passed = result.returncode == expected_code and all(
            text in result.stdout + result.stderr for text in required
        )
        if expected_output is not None:
            passed = passed and result.stdout == expected_output
        self.record(
            {
                "name": name,
                "command": command,
                "cwd": str(cwd),
                "exit_code": result.returncode,
                "expected_code": expected_code,
                "seconds": seconds,
                "stdout_sha256": digest(self.directory / f"{name}.stdout"),
                "stderr_sha256": digest(self.directory / f"{name}.stderr"),
                "passed": passed,
            }
        )
        if not passed:
            raise GateError(f"{name}: unexpected result; see {self.directory}")
        return result

    def save(self) -> None:
        with self.lock:
            self.write_report()

    def write_report(self) -> None:
        order = sorted(range(len(self.records)), key=self.places.__getitem__)
        files = {
            str(path.relative_to(self.directory)): digest(path)
            for path in sorted(self.directory.rglob("*"))
            if path.is_file() and path.name != "report.json"
        }
        report = {
            "hostname": platform.node(),
            "system": platform.platform(),
            "architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "workspace": str(ROOT),
            "reference_manifest_sha256": digest(ROOT / "verification/references.json"),
            "commands": [self.records[i] for i in order],
            "artifacts": files,
        }
        (self.directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def check_sources(tool_root: Path) -> None:
    refs = manifest()
    source = tool_root / "bend"
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != refs["sources"]["bend"]["commit"]:
        raise GateError("Bend revision differs from the protected manifest")
    # Concurrent gates read the same checkout; GIT_OPTIONAL_LOCKS=0 keeps
    # `git status` from taking the index lock to refresh it.
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"],
        text=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    if dirty:
        raise GateError("Pinned Bend checkout has local changes")
    for name, expected in refs["bend_files"].items():
        if digest(source / name) != expected:
            raise GateError(f"Pinned Bend source changed: {name}")


def declaration_names(lines: list[str]) -> list[str]:
    return [
        line.split(" : ", 1)[0]
        for line in lines
        if line and not line[0].isspace() and " : " in line
    ]


def check_inventory(text: str, required: list[str]) -> None:
    declarations = set(declaration_names(text.splitlines()))
    if not required or len(set(required)) != len(required):
        raise GateError("The required law inventory must be nonempty and unique")
    missing = set(required) - declarations
    if missing:
        raise GateError(f"Export omitted required declarations: {sorted(missing)}")


def check_laws(
    evidence: Evidence, name: str, root: Path, required: list[str], timeout: int = 300
) -> Path:
    tool_root = evidence.tool_root
    evidence.run(
        f"{name}-source-check",
        bend(tool_root, root, "--check-only"),
        contains="ALL PROOFS CHECK",
        timeout=timeout,
    )
    artifact = evidence.directory / "PROOF.bendtt"

    def kernel_check() -> None:
        evidence.run(
            f"{name}-kernel-check",
            bend(tool_root, root, "--verdict"),
            contains="ALL PROOFS CHECK",
            timeout=timeout,
        )

    def exported() -> str:
        evidence.run(f"{name}-export", bend(tool_root, root, "-o", artifact))
        text = artifact.read_text()
        check_inventory(text, required)
        evidence.run(
            f"{name}-direct-kernel",
            [str(tool_root / "kernel/bendtt"), str(artifact)],
            contains="ALL PROOFS CHECK",
            timeout=timeout,
        )
        return text

    text = parallel(kernel_check, exported)[1]
    check_omissions(text, required)
    return artifact


def check_omissions(text: str, required: list[str]) -> None:
    """Check that renaming each required declaration fails the inventory.

    It decides what `omitted_inventory_by_rewrite` decides, which rewrites the
    whole text once per declaration, but it scans the text once: renaming a
    declaration's first `^name :` touches one line, so only that line's
    declaration names change.
    """
    if not required or len(set(required)) != len(required):
        return  # check_inventory rejects every renamed text
    if any(len(name.splitlines()) != 1 or name.endswith("\r") for name in required):
        return omitted_inventory_by_rewrite(text, required)
    counts = Counter(declaration_names(text.splitlines()))
    pattern = re.compile("(?m)^(?:" + "|".join(map(re.escape, required)) + ") :")
    first: dict[str, int] = {}
    for match in pattern.finditer(text):
        start = match.start()
        for name in required:
            if name not in first and text.startswith(name + " :", start):
                first[name] = start
    for declaration in required:
        remaining = counts.copy()
        if declaration in first:
            start = first[declaration]
            end = text.find("\n", start)
            line = text[start:] if end < 0 else text[start:end]
            renamed = "omitted :" + line[len(declaration) + 2 :]
            remaining.subtract(declaration_names(line.splitlines()))
            remaining.update(declaration_names(renamed.splitlines()))
        if all(remaining[name] > 0 for name in required):
            raise GateError(
                f"The inventory accepted an omitted declaration: {declaration}"
            )


def omitted_inventory_by_rewrite(text: str, required: list[str]) -> None:
    for declaration in required:
        missing = re.sub(
            rf"(?m)^{re.escape(declaration)} :", "omitted :", text, count=1
        )
        try:
            check_inventory(missing, required)
        except GateError:
            continue
        raise GateError(f"The inventory accepted an omitted declaration: {declaration}")


def required_exports(path: Path) -> list[str]:
    return json.loads(path.read_text())["required_exports"]
