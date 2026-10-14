import importlib
import shutil
from pathlib import Path

from tools.common import GateError

reads = importlib.import_module("verification.r1-b.reads_controls")
transforms = importlib.import_module("verification.r1-b.transform_controls")

EXPERIMENT = "verification/r1-b/from-values"
SRC = reads.SRC
CORE = reads.CORE
CONTRACT = f"{SRC}from_2d_values_2d_contract_2d_LAWS."
LAWS = (f"{CONTRACT}from_values_contract",)
LIVE_RECORDS = (
    f"{SRC}from_2d_values_2d_records.FromValuesContract.arms",
    "consumers.ValuesCase.arms",
    "consumers.EmptyCase.arms",
)
CONSUMERS = tuple(
    f"consumers.{name}"
    for name in (
        "scalar",
        "three_zero",
        "three_one",
        "three_two",
        "grid_first",
        "grid_row",
        "grid_last",
        "empty",
    )
)
# witness -> (statement target, concrete value in the source diagnostic)
WITNESSES = {
    "three_first_witness": ("expected_first", "Some{1.0}"),
    "three_second_witness": ("expected_second", "Some{2.0}"),
    "grid_last_witness": ("expected_last", "Some{6.0}"),
}
VALUES = {
    "expected_first": "(.Some, (F32.lit1065353216, ()))",
    "expected_second": "(.Some, (F32.lit1073741824, ()))",
    "expected_last": "(.Some, (F32.lit1086324736, ()))",
    "wrong_value": reads.EIGHT,
}
MUTANTS = transforms.load_mutants(
    (Path(__file__).resolve().parent / "from-values/mutants.txt").read_text()
)
PROOF = "src/tensor/from-values-contract.bend"
OTHER = "src/tensor/construction-contract-PROOF.bend"
# label -> (rejecting def, text of the mutation on one side of the diagnostic only)
SITES = {
    "from-values-wrong-fill": ("decided_step", "index, 0.0, Storage.set("),
    "from-values-cursor-off-by-one": ("counted_lift", "valid}, 1n))"),
    "from-values-missing-last-write": (
        "fill_lift",
        "Core.fill_values_decided(shape, cursor, U32.from_nat(cursor), value,",
    ),
}


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
    target = WITNESSES[witness][0]
    start = controls.index(f"def {witness}()")
    end = controls.index("}:\n", start) + 2
    header = controls[start:end]
    if header.count(f"{target}()") != 1:
        raise GateError(f"{witness}: false statement target is not unique")
    return (
        controls[:start]
        + header.replace(f"{target}()", "wrong_value()")
        + controls[end:]
    )


def source_reason(output: str, label: str) -> str:
    expected, observed, location = reads._diagnostic(output)
    if "*1" in expected + observed or "*2" in expected + observed:
        raise GateError(f"{label}: quantity/kind mismatch is not value sensitivity")
    if label in WITNESSES:
        value = WITNESSES[label][1]
        if (
            expected != f"{{{value} == Some{{8.0}} : Maybe<&2, F32>}}"
            or observed != f"{{{value} == {value} : Maybe<&2, F32>}}"
            or location != label
        ):
            raise GateError(f"{label}: wrong concrete public value equation")
    elif label in SITES:
        site, mark = SITES[label]
        if (
            location != site
            or expected == observed
            or (mark in expected) == (mark in observed)
        ):
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
        left, _, claim = expected.partition(" == ")
        observed_left, _, observed_claim = observed.partition(" == ")
        if (
            location != label
            or not claim.startswith("wrong_value : ")
            or left != observed_left
            or f"{CORE}from_values" not in left
            or "List.get" not in observed_claim
        ):
            raise GateError(f"{label}: kernel rejection is not the value statement")
    elif label in SITES:
        qualified = f"{SRC}from_2d_values_2d_contract.{SITES[label][0]}"
        if location != qualified or not expected or expected == observed:
            raise GateError(f"{label}: kernel rejection is not at {qualified}")
    else:
        raise GateError(f"Unknown control: {label}")
    return f"{location}: expected {expected} observed {observed}"


def check_export(text: str, catalog) -> dict[str, str]:
    declarations = catalog(text)
    names = (*CONSUMERS, *WITNESSES, *LIVE_RECORDS, *LAWS, *VALUES)
    missing = [name for name in names if name not in declarations]
    if missing:
        raise GateError(f"Export omitted declarations: {missing}")
    for name in LIVE_RECORDS:
        if "Σ-" in declarations[name][1]:
            raise GateError(f"Erased field in a LIVE evidence record: {name}")
    for name in LAWS:
        header = declarations[name][0]
        if not header.startswith(f"{name} : ∀+x0 : ") or "∀+x1 : " not in header:
            raise GateError(f"Unexpected law quantities: {name}")
        if "∀-" in header.split(" : ", 1)[1]:
            raise GateError(f"Erased law premise: {name}")
    for name, value in VALUES.items():
        if declarations[name][1].strip() != value:
            raise GateError(f"{name} is not {value}")
    for witness, (target, _) in WITNESSES.items():
        header = declarations[witness][0]
        if not header.endswith(
            f"== {target} : (Maybe -.Q2 -F32)}}"
        ) and not header.endswith(f"== {target} : (Maybe.q2 -F32)}}"):
            raise GateError(f"Witness statement target changed: {witness}")
        if f"{CORE}from_values" not in header:
            raise GateError(f"Witness is not about Core.from_values: {witness}")
    return {name: "\n".join(declarations[name]) for name in names}
