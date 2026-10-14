import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class GateError(RuntimeError):
    pass


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
        except subprocess.TimeoutExpired as error:
            self.records.append(
                {"name": name, "command": command, "timeout": timeout, "passed": False}
            )
            self.save()
            raise GateError(f"{name}: timed out, never a successful check") from error
        (self.directory / f"{name}.stdout").write_text(result.stdout)
        (self.directory / f"{name}.stderr").write_text(result.stderr)
        required = (contains,) if isinstance(contains, str) else contains
        passed = result.returncode == expected_code and all(
            text in result.stdout + result.stderr for text in required
        )
        if expected_output is not None:
            passed = passed and result.stdout == expected_output
        self.records.append(
            {
                "name": name,
                "command": command,
                "cwd": str(cwd),
                "exit_code": result.returncode,
                "expected_code": expected_code,
                "seconds": time.monotonic() - start,
                "stdout_sha256": digest(self.directory / f"{name}.stdout"),
                "stderr_sha256": digest(self.directory / f"{name}.stderr"),
                "passed": passed,
            }
        )
        self.save()
        if not passed:
            raise GateError(f"{name}: unexpected result; see {self.directory}")
        return result

    def save(self) -> None:
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
            "commands": self.records,
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
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    )
    if dirty:
        raise GateError("Pinned Bend checkout has local changes")
    for name, expected in refs["bend_files"].items():
        if digest(source / name) != expected:
            raise GateError(f"Pinned Bend source changed: {name}")


def check_inventory(text: str, required: list[str]) -> None:
    declarations = {
        line.split(" : ", 1)[0]
        for line in text.splitlines()
        if line and not line[0].isspace() and " : " in line
    }
    if not required or len(set(required)) != len(required):
        raise GateError("The required law inventory must be nonempty and unique")
    missing = set(required) - declarations
    if missing:
        raise GateError(f"Export omitted required declarations: {sorted(missing)}")


def check_laws(
    evidence: Evidence, name: str, root: Path, required: list[str], timeout: int = 300
) -> Path:
    tool_root = evidence.tool_root
    for check, mode in (("source", "--check-only"), ("kernel", "--verdict")):
        evidence.run(
            f"{name}-{check}-check",
            bend(tool_root, root, mode),
            contains="ALL PROOFS CHECK",
            timeout=timeout,
        )
    artifact = evidence.directory / "PROOF.bendtt"
    evidence.run(f"{name}-export", bend(tool_root, root, "-o", artifact))
    text = artifact.read_text()
    check_inventory(text, required)
    evidence.run(
        f"{name}-direct-kernel",
        [str(tool_root / "kernel/bendtt"), str(artifact)],
        contains="ALL PROOFS CHECK",
        timeout=timeout,
    )
    check_omissions(text, required)
    return artifact


def check_omissions(text: str, required: list[str]) -> None:
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
