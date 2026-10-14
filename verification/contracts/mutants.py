import tomllib
from pathlib import Path
from typing import NamedTuple


class Mutant(NamedTuple):
    file: str
    before: str
    after: str


def load(path: Path) -> dict[str, Mutant]:
    with path.open("rb") as catalog:
        return {
            label: Mutant(**entry) for label, entry in tomllib.load(catalog).items()
        }
