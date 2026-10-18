"""Replay Bend checker and BendTT kernel results from an opt-in cache.

With `--cache DIR`, the families, the contracts and `tools.check` look up
each `bun .../bend2/main.ts <file> --check-only`, `--verdict` or
`-o <file>.bendtt` command, and each direct `bendtt <file>` run, in DIR
before they run it. The key holds everything that decides the result:

- the bytes of the checked file and of every file it imports, transitively,
  with each one's path relative to the checked file (Bend names an import
  after that path), and for a `PROOF.bend` its sibling `LAWS.bend`;
- every file of the `bend2` directory that runs, the Bun binary, and where
  Bend's library (`BEND_LIB`) is;
- the BendTT binary that runs, for `--verdict` and direct kernel runs,
  found as the command would find it, and for it and Bun every shared
  library the loader gives them (`ldd`);
- the arguments, with the `-o` target relative to the checked file;
- the environment, but for the variables in `INERT`.

Every digest is computed for every command, so a file replaced during a
run is seen. A command the key cannot pin runs as usual: a hub or named
import, a foreign (`import "x.c"`) definition, a checked file reached
through a symlink, an `-o` target that is one of its inputs or in `bend2`,
a relative `PATH` entry, a checked file inside Bend's library, telemetry
left on (Bend then reads its update notice), a `BUN_*` or `NODE_*` variable
but `BUN_INSTALL` (`BUN_OPTIONS` can preload files), a relative library
path from the loader, or any configuration file Bun would read (`.env*`,
`bunfig.toml`, `package.json`, `tsconfig.json`, `jsconfig.json` in the
working directory or above `bend2`, and the user's `.bunfig.toml`). After a
command runs, its key is computed again and the result is stored only if
nothing changed meanwhile.

An entry, a gzipped JSON file, stores the key, the exit code, stdout,
stderr and the `-o` file's bytes, sealed with an HMAC under a secret kept in
the repository's git directory, outside the cache. An entry whose key or
seal does not match, such as one edited by hand or copied from another
checkout, is reported, run again and replaced, never replayed. Only exit
codes 0 and 1 are stored (and for `-o`, only 0), so a crash or a signal is
never replayed.

Whoever can read the secret can forge an entry. A run without `--cache` runs
every command, and stays the reference for the gates a merge needs.
"""

import argparse
import base64
import gzip
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import threading
import zlib
from pathlib import Path

FORMAT = 2
# A loose superset of Bend's import lines: any line that starts with
# `import`, anywhere in the file, not only in the header Bend reads. A
# line Bend would not read only adds a file, or its absence, to the key.
IMPORT = re.compile(r"^[\s\ufeff]*import[\s\ufeff]+([^\s\ufeff]+)")
# Variables that name the session, not anything Bun, Bend or BendTT reads.
INERT = frozenset(
    {
        "_",
        "PWD",
        "OLDPWD",
        "SHLVL",
        "SSH_CLIENT",
        "SSH_CONNECTION",
        "SSH_TTY",
        "SSH_AUTH_SOCK",
        "MAIL",
        "LS_COLORS",
        "DISPLAY",
        "XDG_SESSION_ID",
        "XDG_SESSION_TYPE",
        "XDG_SESSION_CLASS",
        "MOTD_SHOWN",
        "TMUX",
        "TMUX_PANE",
        "STY",
        "WINDOW",
    }
)
CONFIGS = ("package.json", "tsconfig.json", "jsconfig.json", "bunfig.toml")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: str | Path) -> str:
    try:
        with open(path, "rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except FileNotFoundError:
        return "missing"


def executable(name: str, cwd: Path, env: dict[str, str]) -> str | None:
    """The file `name` runs, found as subprocess and Bun's spawn find it;
    None if a relative PATH entry would make that depend on who looks."""
    if "/" in name:
        candidates = [os.path.join(cwd, name)]
    else:
        directories = env.get("PATH", os.defpath).split(os.pathsep)
        if not all(map(os.path.isabs, directories)):
            return None
        candidates = [os.path.join(directory, name) for directory in directories]
    for found in candidates:
        if os.path.isfile(found) and os.access(found, os.X_OK):
            return os.path.realpath(found)
    return None


def binary(name: str, cwd: Path, env: dict[str, str]) -> list | None:
    """The digests of the executable `name` runs and of every shared library
    the loader gives it under `env` (LD_PRELOAD and LD_LIBRARY_PATH too)."""
    found = executable(name, cwd, env)
    if found is None:
        return None
    for variable in ("LD_PRELOAD", "LD_LIBRARY_PATH"):
        entries = re.split(r"[:\s]+", env.get(variable, "").strip())
        if any(entry and not os.path.isabs(entry) for entry in entries):
            return None  # the loader would read it from the working directory
    listed = subprocess.run(
        ["ldd", found], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )
    if listed.returncode != 0:
        return None
    # "libc.so.6 => /lib/libc.so.6 (0x...)", "/lib64/ld-linux.so.2 (0x...)",
    # "libx.so => not found": the lines without their load addresses, and
    # the digest of each library they name.
    lines = sorted(
        re.sub(r"\s*\(0x[0-9a-f]+\)$", "", line.strip())
        for line in listed.stdout.splitlines()
        if line.strip()
    )
    libraries = []
    for line in lines:
        path = line.split(" => ", 1)[-1]
        if path == "not found" or "/" not in path:
            continue  # the vDSO, or a library the run will not find either
        if not os.path.isabs(path):
            return None  # relative to the loader's working directory
        libraries.append([path, file_digest(os.path.realpath(path))])
    return [[found, file_digest(found)], lines, libraries]


def runtime(bend2: Path, cwd: Path, env: dict[str, str]) -> dict | None:
    """What Bun and Bend read besides the program's own files: every file of
    bend2, and where Bend's library lives. None if Bun would read a
    configuration file (a bunfig.toml can preload code, a .env expands
    variables, a tsconfig.json can extend another), since the key cannot
    follow what those name."""
    if "HOME" in env and not os.path.isabs(env["HOME"]):
        return None  # Bun and Bend would read HOME as given
    home = Path(env.get("HOME", "~")).expanduser()
    config = Path(env.get("XDG_CONFIG_HOME") or home / ".config")
    configs = [
        *(directory / name for directory in bend2.parents for name in CONFIGS),
        *(cwd / name for name in CONFIGS),
        *cwd.glob(".env*"),
        home / ".bunfig.toml",
        config / ".bunfig.toml",
    ]
    if any(os.path.lexists(path) for path in configs):
        return None
    # Bend takes BEND_LIB as given, even empty (the working directory).
    library = os.path.join(
        cwd, env["BEND_LIB"] if "BEND_LIB" in env else home / ".bend/lib"
    )
    return {
        "bend2": [
            [str(path.relative_to(bend2)), file_digest(path)]
            for path in sorted(bend2.rglob("*"))
            if path.is_file() and "node_modules" not in path.parts
        ],
        # Bend names an import by its path inside the library, if it is one.
        "library": [os.path.normpath(library), os.path.realpath(library)]
        if os.path.exists(library)
        else [os.path.normpath(library), None],
    }


def sources(entry: Path, library: tuple[str, ...]) -> dict[str, str] | None:
    """Each file `entry` may import, transitively, by real path, with its
    sha256 or "missing"; None if one is not pinnable, such as a file under
    one of the `library` prefixes, which Bend names, or fetches, as a
    library file."""
    found: dict[str, str] = {}
    todo = [str(entry)]
    if entry.name == "PROOF.bend":
        # Bend refuses a PROOF.bend whose sibling LAWS.bend it does not import.
        todo.append(os.path.realpath(entry.parent / "LAWS.bend"))
    while todo:
        path = todo.pop()
        if path in found:
            continue
        if path.startswith(library):
            return None
        try:
            data = Path(path).read_bytes()
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
            # Bend resolves the path by its text, then reads the file it names.
            lexical = os.path.normpath(os.path.join(Path(path).parent, target))
            if lexical.startswith(library):
                return None
            todo.append(os.path.realpath(lexical))
    return found


def key(argv: list, cwd: Path, env: dict[str, str]) -> dict | None:
    """What decides `argv`'s result, or None if it is not cached."""
    command = [str(part) for part in argv]
    variables = {
        name: value for name, value in sorted(env.items()) if name not in INERT
    }
    if len(command) == 2 and Path(command[0]).name == "bendtt":
        kernel = binary(command[0], cwd, env)
        artifact = os.path.join(cwd, command[1])
        if kernel is None or not os.path.isfile(artifact):
            return None
        return {
            "format": FORMAT,
            "kernel": kernel,
            "input": file_digest(artifact),
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
    given = os.path.join(cwd, command[2])
    if os.path.realpath(given) != given:
        # Bend reads the name it was given (`absent/../x.bend` does not
        # exist) and the file it reaches; take only a plain real path.
        return None
    entry = Path(given)
    if not env.get("BEND_NO_TELEMETRY"):
        return None  # Bend would read and print its update notice
    if any(
        name.startswith(("BUN_", "NODE_")) and name != "BUN_INSTALL" for name in env
    ):
        return None  # such as BUN_OPTIONS=--preload, which names more files
    bun = binary(command[0], cwd, env)
    main = os.path.realpath(os.path.join(cwd, command[1]))
    if bun is None or not os.path.isfile(main):
        return None
    bend2 = Path(main).parent
    reads = runtime(bend2, cwd, env)
    if reads is None:
        return None
    library = tuple(f"{path}/" for path in reads["library"] if path is not None)
    files = sources(entry, library)
    if files is None:
        return None
    material = {
        "format": FORMAT,
        "bun": bun,
        "runtime": reads,
        "arguments": arguments,
        "sources": sorted(
            [os.path.relpath(path, entry.parent), value]
            for path, value in files.items()
        ),
        "env": variables,
    }
    if mode == ["--verdict"]:
        kernel = binary(env["BENDTT"], cwd, env) if env.get("BENDTT") else None
        if kernel is None:
            return None  # Bend would build or find its own kernel
        material["kernel"] = kernel
    if mode[0] == "-o":
        target = os.path.join(cwd, mode[1])
        real = os.path.realpath(target)
        if os.path.normpath(target) != target:
            return None  # Bend writes the name it was given
        if real in files or real.startswith(f"{bend2}/") or os.path.isdir(target):
            return None  # Bend refuses an input (Base too); let Bend say so
        material["target"] = os.path.relpath(target, entry.parent)
    return material


def repository_secret() -> Path:
    """The secret beside this checkout's git objects, shared by its worktrees."""
    root = Path(__file__).resolve().parents[1]
    common = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--git-common-dir"], text=True
    ).strip()
    return (root / common).resolve() / "bendtorch-cache.key"


def secret(path: Path) -> bytes:
    """32 random bytes at `path`, made once with mode 0600. The bytes are
    written before the file appears, so a concurrent first use reads them."""
    if not path.exists():
        temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(os.urandom(32))
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        finally:
            os.unlink(temporary)
    data = path.read_bytes()
    if len(data) != 32:
        raise RuntimeError(f"cache secret {path} is not 32 bytes")
    return data


class Cache:
    def __init__(self, directory: Path, secret_file: Path | None = None):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.secret = secret(secret_file or repository_secret())
        self.replayed = 0
        self.ran = 0
        self.rejected: list[str] = []
        self.lock = threading.Lock()

    def seal(self, entry: dict) -> str:
        body = {name: value for name, value in entry.items() if name != "seal"}
        text = json.dumps(body, sort_keys=True).encode()
        return hmac.new(self.secret, text, hashlib.sha256).hexdigest()

    def run(
        self, command: list, *, cwd: Path, env: dict[str, str], timeout: int
    ) -> tuple[subprocess.CompletedProcess[str], bool]:
        """Run `command`, or replay it; True when it was replayed."""
        material = key(command, cwd, env)
        if material is None:
            return run(command, cwd=cwd, env=env, timeout=timeout), False
        name = sha256(json.dumps(material, sort_keys=True).encode())
        path = self.directory / name[:2] / f"{name}.json.gz"
        target = Path(cwd, str(command[4])) if "target" in material else None
        stored = self.load(path, material)
        if stored is not None and (target is None) == (stored["artifact"] is None):
            if target is not None:
                target.write_bytes(base64.b64decode(stored["artifact"]))
            with self.lock:
                self.replayed += 1
            result = subprocess.CompletedProcess(
                command, stored["exit_code"], stored["stdout"], stored["stderr"]
            )
            return result, True
        result = run(command, cwd=cwd, env=env, timeout=timeout)
        with self.lock:
            self.ran += 1
        storable = result.returncode == 0 or (result.returncode == 1 and target is None)
        if storable:
            artifact = None
            if target is not None:
                artifact = base64.b64encode(target.read_bytes()).decode()
            # A file that changed while the command ran may not be the one
            # it read: store nothing then.
            if key(command, cwd, env) == material:
                self.store(path, material, result, artifact)
        return result, False

    def store(self, path: Path, material: dict, result, artifact: str | None) -> None:
        entry = {
            "key": material,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "artifact": artifact,
        }
        entry["seal"] = self.seal(entry)
        path.parent.mkdir(exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}")
        temporary.write_bytes(gzip.compress(json.dumps(entry).encode(), 1))
        os.replace(temporary, path)

    def load(self, path: Path, material: dict) -> dict | None:
        if not path.exists():
            return None
        try:
            stored = json.loads(gzip.decompress(path.read_bytes()))
            if stored["key"] == material and hmac.compare_digest(
                stored["seal"], self.seal(stored)
            ):
                return stored
        except (OSError, EOFError, zlib.error, ValueError, KeyError, TypeError):
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
    state = "replayed" if replayed else "ran"
    print(f"cache: {state} {cache.directory}", file=sys.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
