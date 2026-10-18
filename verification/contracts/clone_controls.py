import importlib
import re
import shutil
from pathlib import Path
from typing import NamedTuple

from tools.common import GateError

mutants = importlib.import_module("verification.contracts.mutants")
reads = importlib.import_module("verification.contracts.reads_controls")


class Witness(NamedTuple):
    target: str
    replacement: str
    diagnostic_value: str


class Site(NamedTuple):
    proof: str
    definition: str
    mark: str


EXPERIMENT = "verification/contracts/clone"
SRC = reads.SRC
CORE = reads.CORE
CONTRACT = f"{SRC}clone_metadata_contract_LAWS."
LAWS = tuple(
    f"{CONTRACT}{name}"
    for name in ("clone_contract", "clone_write", "metadata_contract")
)
LIVE_RECORDS = tuple(
    f"{SRC}clone_metadata_records.{name}.arms"
    for name in ("CloneContract", "CloneWrite", "MetadataContract")
) + tuple(
    f"consumers.{name}.arms"
    for name in (
        "CloneCase",
        "EmptyClone",
        "WriteCase",
        "MetadataCase",
        "EmptyMetadata",
    )
)
CONSUMERS = tuple(
    f"consumers.{name}"
    for name in (
        "scalar_clone",
        "three_clone_last",
        "grid_clone_first",
        "grid_clone_last",
        "empty_clone",
        "three_write_one",
        "grid_write_last",
        "scalar_write",
        "scalar_metadata",
        "grid_metadata",
        "empty_metadata",
    )
)
WITNESSES = {
    "second_last_witness": Witness("expected_fill", "wrong_value", "Some{7.0}"),
    "written_one_witness": Witness("expected_written", "wrong_value", "Some{9.0}"),
    "alias_one_witness": Witness("expected_fill", "expected_written", "Some{7.0}"),
    "length_owner_zero_witness": Witness("expected_fill", "wrong_value", "Some{7.0}"),
}
REPLACEMENT = {"wrong_value": "Some{8.0}", "expected_written": "Some{9.0}"}
MUTANTS = mutants.load(Path(__file__).resolve().parent / "clone/mutants.toml")
SITES = {
    "clone-first-last-zeroed": Site(
        "src/tensor/clone_contract.bend",
        "first_value_done",
        "Core.first_last_zeroed(",
    ),
    "clone-second-first-zeroed": Site(
        "src/tensor/clone_contract.bend",
        "second_value_projection",
        "Core.drop_status(shape, Core.write_result(shape, second,",
    ),
    "length-off-by-one": Site(
        "src/tensor/metadata_contract.bend",
        "number_projection",
        "1n+number",
    ),
    "length-owner-first-zeroed": Site(
        "src/tensor/metadata_contract.bend",
        "owner_value_projection",
        "Core.drop_status(shape, Core.write_result(shape, storage,",
    ),
    "count-off-by-one": Site(
        "src/tensor/metadata_contract.bend",
        "read_count_pair",
        "U32.add(U32.from_nat(Shape.elements(shape)), 1)",
    ),
}
KERNEL_SITES = {
    "clone-first-last-zeroed": f"{SRC}clone_contract.first_value_done",
    "clone-second-first-zeroed": f"{SRC}clone_contract.second_value_projection",
    "length-off-by-one": f"{SRC}metadata_contract.number_projection",
    "length-owner-first-zeroed": f"{SRC}metadata_contract.owner_value_projection",
    "count-off-by-one": f"{SRC}read_contract.read_count_pair",
}
SEVEN = reads.SEVEN
NINE = reads.NINE
EIGHT = reads.EIGHT


def mutant_copy(snapshot: Path, target: Path, label: str) -> None:
    path, old, new = MUTANTS[label]
    shutil.copytree(snapshot / "src", target / "src")
    edited = target / path
    text = edited.read_text()
    if text.count(old) != 1:
        raise GateError(f"{label}: mutation target in {path} is not unique")
    edited.write_text(text.replace(old, new))
    (target / EXPERIMENT).mkdir(parents=True)
    (target / EXPERIMENT / "core-entry.bend").write_text(
        "import Base\nimport ../../../src/tensor/core.bend as Core\n"
    )


def false_twin(controls: str, witness: str) -> str:
    target = WITNESSES[witness].target
    replacement = WITNESSES[witness].replacement
    start = controls.index(f"def {witness}()")
    end = controls.index("}:\n", start) + 2
    header = controls[start:end]
    if header.count(f"{target}()") != 1:
        raise GateError(f"{witness}: false statement target is not unique")
    return (
        controls[:start]
        + header.replace(f"{target}()", f"{replacement}()")
        + controls[end:]
    )


def source_reason(output: str, label: str) -> str:
    expected, observed, location = reads._diagnostic(output)
    if "*1" in expected + observed or "*2" in expected + observed:
        raise GateError(f"{label}: quantity/kind mismatch is not value sensitivity")
    if label in WITNESSES:
        value = WITNESSES[label].diagnostic_value
        wrong = REPLACEMENT[WITNESSES[label].replacement]
        if (
            expected != f"{{{value} == {wrong} : Maybe<&2, F32>}}"
            or observed != f"{{{value} == {value} : Maybe<&2, F32>}}"
            or location != label
        ):
            raise GateError(f"{label}: wrong concrete public value equation")
    elif label in SITES:
        site, mark = SITES[label].definition, SITES[label].mark
        if location != site or mark not in expected or mark in observed:
            raise GateError(f"{label}: not rejected at {site}")
    else:
        raise GateError(f"Unknown control: {label}")
    return f"{location}: expected {expected} observed {observed}"


def kernel_reason(output: str, label: str) -> str:
    lines = output.splitlines()
    location = next((x[3:-1] for x in lines if x.startswith("In ")), "")
    expected = next((x[10:] for x in lines if x.startswith("expected: ")), "")
    observed = next((x[10:] for x in lines if x.startswith("observed: ")), "")
    if label in WITNESSES:
        witness = WITNESSES[label]
        if (
            location != label
            or f"== {witness.replacement} : " not in expected
            or expected.replace(f"== {witness.replacement}", f"== {witness.target}")
            != observed
        ):
            raise GateError(f"{label}: kernel rejection is not the value statement")
    elif label in KERNEL_SITES:
        if location != KERNEL_SITES[label] or not expected or expected == observed:
            raise GateError(
                f"{label}: kernel rejection is not at {KERNEL_SITES[label]}"
            )
    else:
        raise GateError(f"Unknown control: {label}")
    return f"{location}: expected {expected} observed {observed}"


def check_export(text: str, catalog) -> dict[str, str]:
    declarations = catalog(text)
    names = (
        *CONSUMERS,
        *WITNESSES,
        *LIVE_RECORDS,
        *LAWS,
        "expected_fill",
        "expected_written",
        "wrong_value",
    )
    missing = [name for name in names if name not in declarations]
    if missing:
        raise GateError(f"Export omitted declarations: {missing}")
    for name in LIVE_RECORDS:
        if "Σ-" in declarations[name][1]:
            raise GateError(f"Erased field in a LIVE evidence record: {name}")
    for name in LAWS:
        header = declarations[name][0]
        if not header.startswith(f"{name} : ∀+x0 : ") or "∀+x1 : F32" not in header:
            raise GateError(f"Unexpected law quantities: {name}")
    if not re.search(
        r"∀x\d+ : \{\(Nat\.is_lt \(U32\.to_nat x2\)", declarations[LAWS[1]][0]
    ):
        raise GateError("Bounds proof is not an ordinary law argument: clone_write")
    values = {"expected_fill": SEVEN, "expected_written": NINE, "wrong_value": EIGHT}
    for name, value in values.items():
        if declarations[name][1].strip() != value:
            raise GateError(f"{name} is not {value}")
    for witness in WITNESSES:
        header = declarations[witness][0]
        target = WITNESSES[witness].target
        if not header.endswith(
            f"== {target} : (Maybe -.Q2 -F32)}}"
        ) and not header.endswith(f"== {target} : (Maybe.q2 -F32)}}"):
            raise GateError(f"Witness statement target changed: {witness}")
        if f"{CORE}new" not in header or "F32.lit1088421888" not in header:
            raise GateError(f"Witness is not about Core.new 7.0: {witness}")
    return {name: "\n".join(declarations[name]) for name in names}
