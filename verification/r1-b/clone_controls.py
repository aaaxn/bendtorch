import importlib
import re
import shutil
from pathlib import Path

from tools.common import GateError

reads = importlib.import_module("verification.r1-b.reads_controls")

EXPERIMENT = "verification/r1-b/clone"
SRC = reads.SRC
CORE = reads.CORE
CONTRACT = f"{SRC}clone_2d_metadata_2d_contract_2d_LAWS."
LAWS = tuple(
    f"{CONTRACT}{name}"
    for name in ("clone_contract", "clone_write", "metadata_contract")
)
LIVE_RECORDS = tuple(
    f"{SRC}clone_2d_metadata_2d_records.{name}.arms"
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
# witness -> (statement target, replacement, concrete value in the source diagnostic)
WITNESSES = {
    "second_last_witness": ("expected_fill", "wrong_value", "Some{7.0}"),
    "written_one_witness": ("expected_written", "wrong_value", "Some{9.0}"),
    "alias_one_witness": ("expected_fill", "expected_written", "Some{7.0}"),
    "length_owner_zero_witness": ("expected_fill", "wrong_value", "Some{7.0}"),
}
REPLACEMENT = {"wrong_value": "Some{8.0}", "expected_written": "Some{9.0}"}
MUTANTS = reads.load_mutants(
    (Path(__file__).resolve().parent / "clone/mutants.txt").read_text()
)
# label -> (proof file, rejecting def, mark in the source expected side)
SITES = {
    "clone-first-last-zeroed": (
        "src/tensor/clone-contract.bend",
        "first_value_done",
        "Core.first_last_zeroed(",
    ),
    "clone-second-first-zeroed": (
        "src/tensor/clone-contract.bend",
        "second_value_projection",
        "Core.drop_status(shape, Core.write_result(shape, second,",
    ),
    "length-off-by-one": (
        "src/tensor/metadata-contract.bend",
        "number_projection",
        "1n+number",
    ),
    "length-owner-first-zeroed": (
        "src/tensor/metadata-contract.bend",
        "owner_value_projection",
        "Core.drop_status(shape, Core.write_result(shape, storage,",
    ),
    "count-off-by-one": (
        "src/tensor/metadata-contract.bend",
        "read_count_pair",
        "U32.add(U32.from_nat(Shape.elements(shape)), 1)",
    ),
}
KERNEL_SITES = {
    "clone-first-last-zeroed": f"{SRC}clone_2d_contract.first_value_done",
    "clone-second-first-zeroed": f"{SRC}clone_2d_contract.second_value_projection",
    "length-off-by-one": f"{SRC}metadata_2d_contract.number_projection",
    "length-owner-first-zeroed": f"{SRC}metadata_2d_contract.owner_value_projection",
    "count-off-by-one": f"{SRC}read_2d_contract.read_count_pair",
}
SEVEN = reads.SEVEN
NINE = reads.NINE
EIGHT = reads.EIGHT


def mutant_copy(snapshot: Path, target: Path, label: str) -> None:
    shutil.copytree(snapshot / "src", target / "src")
    core = target / "src/tensor/core.bend"
    text = core.read_text()
    old, new = MUTANTS[label]
    if text.count(old) != 1:
        raise GateError(f"{label}: mutation target in core.bend is not unique")
    core.write_text(text.replace(old, new))
    (target / EXPERIMENT).mkdir(parents=True)
    (target / EXPERIMENT / "core-entry.bend").write_text(
        "import Base\nimport ../../../src/tensor/core.bend as Core\n"
    )


def false_twin(controls: str, witness: str) -> str:
    target, replacement, _ = WITNESSES[witness]
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
        _, replacement, value = WITNESSES[label]
        wrong = REPLACEMENT[replacement]
        if (
            expected != f"{{{value} == {wrong} : Maybe<&2, F32>}}"
            or observed != f"{{{value} == {value} : Maybe<&2, F32>}}"
            or location != label
        ):
            raise GateError(f"{label}: wrong concrete public value equation")
    elif label in SITES:
        _, site, mark = SITES[label]
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
        target, replacement, _ = WITNESSES[label]
        if (
            location != label
            or f"== {replacement} : " not in expected
            or expected.replace(f"== {replacement}", f"== {target}") != observed
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
    for witness, (target, _, _) in WITNESSES.items():
        header = declarations[witness][0]
        if not header.endswith(
            f"== {target} : (Maybe -.Q2 -F32)}}"
        ) and not header.endswith(f"== {target} : (Maybe.q2 -F32)}}"):
            raise GateError(f"Witness statement target changed: {witness}")
        if f"{CORE}new" not in header or "F32.lit1088421888" not in header:
            raise GateError(f"Witness is not about Core.new 7.0: {witness}")
    return {name: "\n".join(declarations[name]) for name in names}
