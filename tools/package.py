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
  `CHANGELOG.md`.

It prints one shipped path per line, then the file count and size. It does
not check the code: `bend bendtorch.bend --verdict` does.
"""

import argparse
import re
from pathlib import Path

from tools.common import ROOT

ENTRY = "bendtorch.bend"
SHIPPED_ROOTS = ("bendtorch.bend", "LAWS.bend", "PROOF.bend")
IMPORT = re.compile(r"^import\s+(\.{1,2}/\S+\.bend)\b", re.MULTILINE)
SPDX = re.compile(r"^\s*SPDX-License-Identifier:\s*\S")
ENTRY_VERSION = re.compile(r"bendtorch@(\d+\.\d+\.\d+\.\d+)/")
CHANGELOG_VERSION = re.compile(r"^## \[?(\d+\.\d+\.\d+\.\d+)\]?", re.MULTILINE)


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
        for target in IMPORT.findall(path.read_text()):
            todo.append((path.parent / target).resolve())
    return sorted(seen)


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
