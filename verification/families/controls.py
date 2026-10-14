import importlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from tools.common import GateError

reads = importlib.import_module("verification.r1-b.reads_controls")
transforms = importlib.import_module("verification.r1-b.transform_controls")
HERE = Path(__file__).resolve().parent

PROOF = "src/tensor/contracts/PROOF.bend"
SRC = "_2e_._2f_.._2f_.._2f_src_2f_tensor_2f_contracts_2f_"
MAYBE = "(Maybe -.Q2 -F32)"


class Witness(NamedTuple):
    value: str
    wrong: str


class Value(NamedTuple):
    source: str
    kernel: str


class Site(NamedTuple):
    source_def: str
    source_mark: str
    kernel_declaration: str
    replaced: tuple[str, ...]


class Law(NamedTuple):
    prefix: str
    call: str


def load_mutants(family: str) -> dict[str, tuple[str, str, str]]:
    return transforms.load_mutants((HERE / family / "mutants.txt").read_text())


@dataclass(frozen=True)
class Family:
    """The consumers, false twins and mutants of one contract family.

    `entry` lists the operation modules, as repository paths, that a mutant
    export imports in order. `proof` is the module whose source check must
    reject each mutant, and `prefix` is the BendTT name prefix of the
    family's declarations.
    """

    name: str
    entry: tuple[str, ...]
    operations: frozenset[str]
    laws: dict[str, Law]
    live_records: tuple[str, ...]
    consumers: tuple[str, ...]
    witnesses: dict[str, Witness]
    values: dict[str, Value]
    subjects: tuple[str, ...]
    mutants: dict[str, tuple[str, str, str]]
    sites: dict[str, Site]
    proof: str = PROOF
    prefix: str = SRC

    @property
    def experiment(self) -> str:
        return f"verification/families/{self.name}"

    @property
    def entry_path(self) -> str:
        return f"{self.experiment}/entry.bend"

    def mutant_copy(self, snapshot: Path, target: Path, label: str) -> None:
        path, old, new = self.mutants[label]
        shutil.copytree(snapshot / "src", target / "src")
        edited = target / path
        text = edited.read_text()
        if text.count(old) != 1:
            raise GateError(f"{label}: mutation target in {path} is not unique")
        edited.write_text(text.replace(old, new))
        (target / self.experiment).mkdir(parents=True)
        (target / self.entry_path).write_text(
            "import Base\n"
            + "".join(
                f"import ../../../{path} as "
                f"{''.join(part.title() for part in Path(path).stem.split('-'))}\n"
                for path in self.entry
            )
        )

    def false_twin(self, controls: str, witness: str) -> str:
        target, wrong = self.witnesses[witness]
        start = controls.index(f"def {witness}()")
        end = controls.index("}:\n", start) + 2
        header = controls[start:end]
        if header.count(f"{target}()") != 1:
            raise GateError(f"{witness}: false statement target is not unique")
        return (
            controls[:start]
            + header.replace(f"{target}()", f"{wrong}()")
            + controls[end:]
        )

    def kernel_twin(self, text: str, witness: str, catalog, replace_declaration) -> str:
        target, wrong = self.witnesses[witness]
        header, body = catalog(text)[witness]
        if header.count(f"== {target} : ") != 1:
            raise GateError(f"{witness}: exported statement target is not unique")
        changed = header.replace(f"== {target} : ", f"== {wrong} : ")
        return replace_declaration(text, witness, changed, body)

    def source_reason(self, output: str, label: str) -> str:
        expected, observed, location = reads._diagnostic(output)
        if "*1" in expected + observed or "*2" in expected + observed:
            raise GateError(f"{label}: quantity/kind mismatch is not value sensitivity")
        if label in self.witnesses:
            value, wrong = (self.values[name].source for name in self.witnesses[label])
            if (
                location != label
                or expected != f"{{{value} == {wrong} : Maybe<&2, F32>}}"
                or observed != f"{{{value} == {value} : Maybe<&2, F32>}}"
            ):
                raise GateError(f"{label}: wrong concrete public value equation")
        elif label in self.sites:
            site, mark = self.sites[label].source_def, self.sites[label].source_mark
            if (
                location != site
                or expected == observed
                or (mark in expected) == (mark in observed)
            ):
                raise GateError(f"{label}: not rejected at {site}")
        else:
            raise GateError(f"Unknown control: {label}")
        return f"{location}: expected {expected} observed {observed}"

    def kernel_reason(self, output: str, label: str) -> str:
        lines = output.splitlines()
        location = next((x[3:-1] for x in lines if x.startswith("In ")), "")
        expected = next((x[10:] for x in lines if x.startswith("expected: ")), "")
        observed = next((x[10:] for x in lines if x.startswith("observed: ")), "")
        if label in self.witnesses:
            wrong = self.witnesses[label].wrong
            claim, _, value = expected.rpartition(" == ")
            if (
                location != label
                or value != f"{wrong} : ((Maybe -.Q2) -F32)}}"
                or not observed.startswith(f"{claim} == (.Some, (")
                or wrong in observed
            ):
                raise GateError(f"{label}: kernel rejection is not the value statement")
        elif label in self.sites:
            site = f"{self.prefix}{self.sites[label].kernel_declaration}"
            if location != site or not expected or expected == observed:
                raise GateError(f"{label}: kernel rejection is not at {site}")
        else:
            raise GateError(f"Unknown control: {label}")
        return f"{location}: expected {expected} observed {observed}"

    def replaced(self, positive: str, artifact: str, catalog, label: str) -> str:
        before = catalog(positive)
        after = catalog(artifact)
        changed = tuple(
            sorted(
                name.removeprefix(self.prefix)
                for name, (header, body) in after.items()
                if name.startswith(self.prefix)
                and name in before
                and (header, body.strip()) != (before[name][0], before[name][1].strip())
            )
        )
        if changed != self.sites[label].replaced:
            raise GateError(f"{label}: kernel artifact replaces {changed}")
        return ", ".join(changed)

    def check_export(self, text: str, catalog) -> dict[str, str]:
        declarations = catalog(text)
        names = (
            *self.consumers,
            *self.witnesses,
            *self.live_records,
            *self.laws,
            *self.values,
        )
        missing = [name for name in names if name not in declarations]
        if missing:
            raise GateError(f"Export omitted declarations: {missing}")
        for name in self.live_records:
            if "Σ-" in declarations[name][1]:
                raise GateError(f"Erased field in a LIVE evidence record: {name}")
        for name, (prefix, call) in self.laws.items():
            header = declarations[name][0]
            if (
                not header.startswith(f"{name} : {prefix}")
                or f"{self.prefix}{call}" not in header
            ):
                raise GateError(f"{name} is not stated for every {prefix} over {call}")
        for name, value in self.values.items():
            if declarations[name][1].strip() != value.kernel:
                raise GateError(f"{name} is not {value.kernel}")
        for witness, (target, _) in self.witnesses.items():
            header = declarations[witness][0]
            if not header.endswith(f"== {target} : {MAYBE}}}"):
                raise GateError(f"Witness statement target changed: {witness}")
            if not any(f"(consumers.{subject} " in header for subject in self.subjects):
                raise GateError(f"Witness is not about a {self.name} result: {witness}")
        return {name: "\n".join(declarations[name]) for name in names}
