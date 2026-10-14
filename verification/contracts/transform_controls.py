import importlib
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


EXPERIMENT = "verification/contracts/transforms"
SRC = reads.SRC
CONTRACT = f"{SRC}transform_2d_contract_2d_LAWS."
LAWS = tuple(
    f"{CONTRACT}{name}"
    for name in (
        "reshape_contract",
        "reshape_write",
        "reshape_count_mismatch",
        "permute_rank_mismatch",
        "permute_axis_out_of_bounds",
        "permute_duplicate_axis",
        "transpose_out_of_bounds",
    )
)
LIVE_RECORDS = tuple(
    f"{SRC}transform_2d_records.{name}.arms"
    for name in ("RetainedFailure", "ReshapeContract", "ReshapeWrite")
) + tuple(
    f"consumers.{name}.arms"
    for name in ("ReshapeCase", "EmptyReshape", "WriteCase", "FailureCase")
)
CONSUMERS = tuple(
    f"consumers.{name}"
    for name in (
        "grid_to_tall",
        "grid_to_flat",
        "scalar_to_one",
        "empty_reshape",
        "grid_write_row",
        "grid_write_last",
        "grid_mismatch",
        "empty_mismatch",
        "grid_short_axes",
        "cube_axis_outside",
        "cube_duplicate_axis",
        "grid_transpose_outside",
        "scalar_transpose_outside",
    )
)
WITNESSES = {
    "reshape_first_witness": Witness("expected_fill", "wrong_value", "Some{7.0}"),
    "reshape_written_witness": Witness("expected_written", "wrong_value", "Some{9.0}"),
    "permute_kept_witness": Witness("expected_fill", "wrong_value", "Some{7.0}"),
    "transpose_kept_witness": Witness("expected_fill", "wrong_value", "Some{7.0}"),
}


MUTANTS = mutants.load(Path(__file__).resolve().parent / "transforms/mutants.toml")
RESHAPE = "src/tensor/reshape-contract.bend"
PERMUTE = "src/tensor/permute-contract.bend"
SITES = {
    "reshape-zeroed": Site(RESHAPE, "decided_lift", "Transforms.drop_status(output,"),
    "reshape-unjustified-failure": Site(
        RESHAPE,
        "decided_lift",
        "P(Fail{Transforms.LayoutError{Layout.ElementCountMismatch{}}})",
    ),
    "reshape-input-dimensions": Site(
        RESHAPE,
        "built_lift",
        "Transforms.reshape_decided(input, input,",
    ),
    "permute-failure-changes-source": Site(
        PERMUTE,
        "planned_failure",
        "Transforms.drop_status(input, Core.write_flat(input, tensor, 0, 0.0))",
    ),
    "transpose-failure-changes-source": Site(
        PERMUTE,
        "transpose_failure",
        "Transforms.drop_status(input, Core.write_flat(input, tensor, 0, 0.0))",
    ),
    "permute-wrong-error": Site(
        PERMUTE, "bounded_failure", "Fail{Layout.DuplicateAxis{}}"
    ),
}
OTHER = {RESHAPE: PERMUTE, PERMUTE: RESHAPE}


def export_prefix(label: str) -> str:
    stem = Path(MUTANTS[label].file).stem.replace("-", "_2d_")
    return f"{SRC}{stem}."


def mutant_copy(snapshot: Path, target: Path, label: str) -> None:
    path, old, new = MUTANTS[label]
    shutil.copytree(snapshot / "src", target / "src")
    edited = target / path
    text = edited.read_text()
    if text.count(old) != 1:
        raise GateError(f"{label}: mutation target in {path} is not unique")
    edited.write_text(text.replace(old, new))
    (target / EXPERIMENT).mkdir(parents=True)
    (target / EXPERIMENT / "module-entry.bend").write_text(
        f"import Base\nimport ../../../{path} as Module\n"
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
        if (
            expected != f"{{{value} == Some{{8.0}} : Maybe<&2, F32>}}"
            or observed != f"{{{value} == {value} : Maybe<&2, F32>}}"
            or location != label
        ):
            raise GateError(f"{label}: wrong concrete public value equation")
    elif label in SITES:
        site, mark = SITES[label].definition, SITES[label].mark
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
        witness = WITNESSES[label]
        if (
            location != label
            or f"== {witness.replacement} : " not in expected
            or expected.replace(f"== {witness.replacement}", f"== {witness.target}")
            != observed
        ):
            raise GateError(f"{label}: kernel rejection is not the value statement")
    elif label in SITES:
        site = SITES[label]
        stem = Path(site.proof).stem.replace("-", "_2d_")
        qualified = f"{SRC}{stem}.{site.definition}"
        if location != qualified or not expected or expected == observed:
            raise GateError(f"{label}: kernel rejection is not at {qualified}")
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
        if "∀-" in header.split(" : ", 1)[1]:
            raise GateError(f"Erased law premise: {name}")
    values = {
        "expected_fill": reads.SEVEN,
        "expected_written": reads.NINE,
        "wrong_value": reads.EIGHT,
    }
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
        if f"{reads.CORE}new" not in header or "F32.lit1088421888" not in header:
            raise GateError(f"Witness is not about Core.new 7.0: {witness}")
    return {name: "\n".join(declarations[name]) for name in names}
