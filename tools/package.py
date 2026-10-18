"""List and check the files that `bend bendtorch.bend --publish` would ship.

`--publish` uploads the entry with every file it imports, and the LICENSE beside
each of them. This command follows the same relative imports, without the
network, and exits 1 when the package would be wrong:

- an import climbs above the repository or names a missing file;
- a shipped file lies outside `bendtorch.bend`, `LAWS.bend`, `PROOF.bend` and
  `src/`, such as an example, an experiment or a probe;
- the root LICENSE has no `SPDX-License-Identifier:` line in its first five
  lines, so BendHub would show only "see LICENSE";
- the version in the entry's first line differs from the newest version in
  `CHANGELOG.md`;
- a shipped path breaks BendHub's path rule, which only the hub enforces:
  every segment `[A-Za-z_][A-Za-z0-9_]*`, the file ending in `.bend`, `.c` or
  `.js` or named `LICENSE`, the path under 256 characters. A hyphen in a file
  name is the usual cause.

It prints one shipped path per line, then the file count and size. It does
not check the code: `bend bendtorch.bend --verdict` does.
"""

import argparse
import re
from pathlib import Path

from tools.common import ROOT

ENTRY = "bendtorch.bend"
SHIPPED_ROOTS = ("bendtorch.bend", "LAWS.bend", "PROOF.bend")
# The header grammar of bend2/bend.ts book_load: imports lead the file, after
# blank and `#` lines, each line trimmed; the first other line ends them.
IMPORT = re.compile(r"^import\s+(\S+)(?:\s+as\s+([A-Za-z_]\w*))?\s*(?:#.*)?$")
SPDX = re.compile(r"^\s*SPDX-License-Identifier:\s*\S")
ENTRY_VERSION = re.compile(r"bendtorch-core@(\d+\.\d+\.\d+\.\d+)/")
CHANGELOG_VERSION = re.compile(r"^## \[?(\d+\.\d+\.\d+\.\d+)\]?", re.MULTILINE)
# The path rule BendHub answers with when it refuses a publish.
HUB_SEGMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
HUB_FILE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\.(?:bend|c|js)|LICENSE")


def hub_path_ok(path: str) -> bool:
    *dirs, name = path.split("/")
    return (
        len(path) < 256
        and all(HUB_SEGMENT.fullmatch(d) for d in dirs)
        and HUB_FILE.fullmatch(name) is not None
    )


class PackageError(Exception):
    pass


def closure(root: Path, entry: str) -> list[str]:
    """The repository paths of the entry and every .bend file it imports."""
    top = root.resolve()
    seen: set[str] = set()
    todo = [(top / entry).resolve()]
    while todo:
        path = todo.pop()
        if not path.is_relative_to(top):
            raise PackageError(f"{path} lies above the repository")
        rel = path.relative_to(top).as_posix()
        if rel in seen:
            continue
        if not path.is_file():
            raise PackageError(f"{rel} is imported but does not exist")
        seen.add(rel)
        todo.extend((path.parent / t).resolve() for t in imports(path, rel))
    return sorted(seen)


def imports(path: Path, rel: str) -> list[str]:
    """The local paths `path` imports, read as Bend reads its header."""
    found = []
    for raw in path.read_text().split("\n"):
        line = raw.strip()
        if line == "" or line.startswith("#"):
            continue
        if not re.match(r"import(\s|$)", line):
            break
        m = IMPORT.match(line)
        if m is None:
            raise PackageError(f"{rel}: malformed import {line!r}")
        target, alias = m.group(1), m.group(2)
        if target == "Base" and alias is None:
            continue
        if alias is None or not target.endswith(".bend"):
            raise PackageError(f"{rel}: malformed import {line!r}")
        if target.startswith(("/", "0x")) or "@" in target.split("/")[0]:
            raise PackageError(f"{rel}: imports {target}, not a path in the repo")
        found.append(target)
    return found


def check(root: Path, entry: str = ENTRY) -> list[str]:
    files = closure(root, entry)
    stray = [f for f in files if f not in SHIPPED_ROOTS and not f.startswith("src/")]
    if stray:
        raise PackageError("would ship files outside src/: " + ", ".join(stray))
    if any(part.lower() == "license" for f in files for part in f.split("/")[:-1]):
        raise PackageError("a shipped file is in a directory named license")
    license_head = (root / "LICENSE").read_text().splitlines()[:5]
    if not any(SPDX.match(line) for line in license_head):
        raise PackageError("LICENSE has no SPDX-License-Identifier line on top")
    first = (root / entry).read_text().splitlines()[0]
    named = ENTRY_VERSION.search(first)
    released = CHANGELOG_VERSION.search((root / "CHANGELOG.md").read_text())
    if named is None or released is None:
        raise PackageError(
            "the entry's first line or CHANGELOG.md names no 4-part version"
        )
    if named.group(1) != released.group(1):
        raise PackageError(
            f"{entry} names {named.group(1)}, CHANGELOG.md {released.group(1)}"
        )
    shipped = [*files, "LICENSE"]
    licensed_dirs = {str(Path(f).parent) for f in files}
    for d in sorted(licensed_dirs - {"."}):
        if (root / d / "LICENSE").exists():
            shipped.append(f"{d}/LICENSE")
    bad = [f for f in shipped if not hub_path_ok(f)]
    if bad:
        raise PackageError(
            f"BendHub refuses {len(bad)} path(s), e.g. {bad[0]}: segments must be"
            " [A-Za-z_][A-Za-z0-9_]*, ending in .bend, .c or .js or naming LICENSE"
        )
    return sorted(shipped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--entry", default=ENTRY)
    args = parser.parse_args(argv)
    try:
        shipped = check(ROOT, args.entry)
    except PackageError as error:
        print(f"package: {error}")
        return 1
    size = sum((ROOT / f).stat().st_size for f in shipped)
    print("\n".join(shipped))
    print(f"{len(shipped)} files, {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
