"""Replay Bend checker and BendTT kernel results from an opt-in cache.

With `--cache DIR`, the families, the contracts and `tools.check` look up
each `bun .../bend2/main.ts <file> --check-only`, `--verdict` or
`-o <file>.bendtt` command, and each direct `bendtt <file>` run, in DIR
before they run it. The key holds everything that decides the result:

- the bytes of the checked file and of every file it imports, transitively,
  with each one's path relative to the checked file (Bend names an import
  after that path);
- the Bend checkout's commit, which must be clean, and the Bun binary;
- the BendTT binary, for `--verdict` and direct kernel runs;
- the arguments, with the `-o` target relative to the checked file;
- every `BEND*` and `LEAN*` environment variable but `BENDTT`, which the
  kernel's digest replaces.

A command whose inputs the key cannot pin runs as usual: a hub or named
import, or a foreign (`import "x.c"`) definition anywhere in its files. An
entry stores the key, the exit code, stdout, stderr and the `-o` file, and
a digest of all of them; an entry whose key or digest does not match is
reported, run again and replaced, never replayed. Only exit codes 0 and 1
are stored (and for `-o`, only 0), so a crash or a signal is never
replayed.

The cache trusts DIR: a writer who can recompute the digest can make a
command pass. A run without `--cache` runs every command, and stays the
reference for the gates a merge needs.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

FORMAT = 1
# A loose superset of Bend's import lines: any line that starts with
# `import`, anywhere in the file, not only in the header Bend reads. A
# line Bend would not read only adds a file, or its absence, to the key.
IMPORT = re.compile(r"^[\s\ufeff]*import[\s\ufeff]+([^\s\ufeff]+)")

_tools: dict[str, str] = {}
_bends: dict[Path, str | None] = {}
_lock = threading.Lock()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tool_digest(path: str) -> str:
    """The digest of a binary that does not change during a run."""
    with _lock:
        if path not in _tools:
            _tools[path] = sha256(Path(path).read_bytes())
        return _tools[path]


def bend_commit(root: Path) -> str | None:
    """The Bend checkout's commit, or None if it has local changes."""
    with _lock:
        if root not in _bends:
            env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
            git = ["git", "-C", str(root)]
            try:
                commit = subprocess.check_output(
                    [*git, "rev-parse", "HEAD"], text=True, env=env
                ).strip()
                dirty = subprocess.check_output(
                    [*git, "status", "--porcelain"], text=True, env=env
                )
            except (OSError, subprocess.CalledProcessError):
                commit, dirty = "", "?"
            _bends[root] = None if dirty or not commit else commit
        return _bends[root]


def sources(entry: Path) -> list[list[str]] | None:
    """Each file `entry` may import, transitively, as [path relative to
    entry's directory, sha256 or "missing"]; None if one is not pinnable."""
    top = entry.parent
    found: dict[Path, str] = {}
    todo = [entry]
    while todo:
        path = todo.pop()
        if path in found:
            continue
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            found[path] = "missing"
            continue
        except OSError:
            return None
        found[path] = sha256(data)
        for line in data.decode("utf-8", "replace").split("\n"):
            match = IMPORT.match(line)
            if match is None:
                continue
            target = match.group(1)
            if target == "Base":
                continue
            if not target.endswith(".bend"):
                if target.startswith('"'):
                    return None  # a foreign definition reads a .c or .js file
                continue  # Bend rejects it, from the bytes the key holds
            if "@" in target or re.match(r"0x[0-9a-f]+/", target):
                return None  # a hub or named import
            todo.append(Path(os.path.realpath(os.path.normpath(path.parent / target))))
    return sorted([os.path.relpath(path, top), value] for path, value in found.items())


def key(argv: list, cwd: Path, env: dict[str, str]) -> dict | None:
    """What decides `argv`'s result, or None if it is not cached."""
    command = [str(part) for part in argv]
    variables = {
        name: value
        for name, value in sorted(env.items())
        if name.startswith(("BEND", "LEAN")) and name != "BENDTT"
    }
    if len(command) == 2 and Path(command[0]).name == "bendtt":
        kernel = cwd / command[0]
        artifact = cwd / command[1]
        if not artifact.is_file():
            return None
        return {
            "format": FORMAT,
            "kernel": tool_digest(str(kernel)),
            "input": sha256(artifact.read_bytes()),
            "env": variables,
        }
    if len(command) < 4 or not command[1].endswith("bend2/main.ts"):
        return None
    mode = command[3:]
    if mode in (["--check-only"], ["--verdict"]):
        arguments = mode
    elif len(mode) == 2 and mode[0] == "-o" and mode[1].endswith(".bendtt"):
        arguments = ["-o", "<target>"]
    else:
        return None
    bun = shutil.which(command[0], path=env.get("PATH"))
    entry = Path(os.path.realpath(cwd / command[2]))
    commit = bend_commit(Path(command[1]).resolve().parents[1])
    files = sources(entry)
    if bun is None or commit is None or files is None:
        return None
    material = {
        "format": FORMAT,
        "bun": tool_digest(bun),
        "bend": commit,
        "arguments": arguments,
        "sources": files,
        "env": variables,
    }
    if mode == ["--verdict"]:
        kernel = env.get("BENDTT")
        if not kernel:
            return None  # Bend would build its own kernel
        material["kernel"] = tool_digest(kernel)
    if mode[0] == "-o":
        material["target"] = os.path.relpath(
            os.path.normpath(cwd / mode[1]), entry.parent
        )
    return material


def seal(entry: dict) -> str:
    body = {name: value for name, value in entry.items() if name != "sha256"}
    return sha256(json.dumps(body, sort_keys=True).encode())


class Cache:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.replayed = 0
        self.ran = 0
        self.rejected: list[str] = []
        self.lock = threading.Lock()

    def run(
        self, command: list, *, cwd: Path, env: dict[str, str], timeout: int
    ) -> tuple[subprocess.CompletedProcess[str], bool]:
        """Run `command`, or replay it; True when it was replayed."""
        material = key(command, cwd, env)
        if material is None:
            return run(command, cwd=cwd, env=env, timeout=timeout), False
        name = sha256(json.dumps(material, sort_keys=True).encode())
        path = self.directory / name[:2] / f"{name}.json"
        target = cwd / str(command[4]) if "target" in material else None
        stored = self.load(path, material)
        if stored is not None and (target is None) == (stored["artifact"] is None):
            if target is not None:
                target.write_text(stored["artifact"])
            with self.lock:
                self.replayed += 1
            result = subprocess.CompletedProcess(
                command, stored["exit_code"], stored["stdout"], stored["stderr"]
            )
            return result, True
        result = run(command, cwd=cwd, env=env, timeout=timeout)
        with self.lock:
            self.ran += 1
        if result.returncode == 0 or (result.returncode == 1 and target is None):
            artifact = None if target is None else target.read_text()
            entry = {
                "key": material,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "artifact": artifact,
            }
            entry["sha256"] = seal(entry)
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}")
            temporary.write_text(json.dumps(entry))
            os.replace(temporary, path)
        return result, False

    def load(self, path: Path, material: dict) -> dict | None:
        if not path.exists():
            return None
        try:
            stored = json.loads(path.read_text())
            if stored["key"] == material and stored["sha256"] == seal(stored):
                return stored
        except (OSError, ValueError, KeyError, TypeError):
            pass
        with self.lock:
            self.rejected.append(path.name)
        print(f"cache: {path} does not match its key; running", file=sys.stderr)
        return None

    def summary(self) -> dict:
        return {
            "directory": str(self.directory),
            "replayed": self.replayed,
            "ran": self.ran,
            "rejected": sorted(self.rejected),
        }


def run(
    command: list, *, cwd: Path, env: dict[str, str], timeout: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one command through the cache; print its output and "
        "exit with its code."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    cache = Cache(args.cache)
    result, replayed = cache.run(
        command, cwd=Path.cwd(), env=dict(os.environ), timeout=args.timeout
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    print(
        f"cache: {'replayed' if replayed else 'ran'} {cache.directory}", file=sys.stderr
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
