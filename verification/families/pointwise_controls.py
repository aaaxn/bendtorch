import importlib
import shutil
from pathlib import Path
from typing import NamedTuple

from tools.common import GateError

reads = importlib.import_module("verification.r1-b.reads_controls")
transforms = importlib.import_module("verification.r1-b.transform_controls")


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


EXPERIMENT = "verification/families/pointwise"
PROOF = "src/tensor/contracts/PROOF.bend"
ENTRY = f"{EXPERIMENT}/entry.bend"
SRC = "_2e_._2f_.._2f_.._2f_src_2f_tensor_2f_contracts_2f_"
LAW = f"{SRC}LAWS.broadcast_pointwise"
LIVE_RECORDS = (
    f"{SRC}pointwise_2d_records.PointwiseContract.arms",
    "consumers.PointCase.arms",
    "consumers.EmptyCase.arms",
)
CONSUMERS = tuple(
    f"consumers.{name}"
    for name in (
        "sub_grid_first",
        "sub_grid_middle",
        "sub_grid_last",
        "mul_outer_row",
        "mul_outer_column",
        "add_same",
        "keep_scalar",
        "empty",
    )
)
WITNESSES = {
    "sub_witness": Witness("expected_sub", "swapped_sub"),
    "outer_witness": Witness("expected_outer", "unbroadcast_outer"),
    "keep_witness": Witness("expected_keep", "wrong_keep"),
    "kept_witness": Witness("expected_kept", "overwritten_kept"),
}
FIVE = "F32.lit1084227584"
TWENTY = "F32.lit1101004800"
VALUES = {
    "expected_sub": Value(
        "Some{F32.sub(5.0, 20.0)}", f"(.Some, ((F32.sub {FIVE} {TWENTY}), ()))"
    ),
    "swapped_sub": Value(
        "Some{F32.sub(20.0, 5.0)}", f"(.Some, ((F32.sub {TWENTY} {FIVE}), ()))"
    ),
    "expected_outer": Value(
        "Some{F32.mul(3.0, 30.0)}",
        "(.Some, ((F32.mul F32.lit1077936128 F32.lit1106247680), ()))",
    ),
    "unbroadcast_outer": Value(
        "Some{F32.mul(2.0, 30.0)}",
        "(.Some, ((F32.mul F32.lit1073741824 F32.lit1106247680), ()))",
    ),
    "expected_keep": Value("Some{7.0}", "(.Some, (F32.lit1088421888, ()))"),
    "wrong_keep": Value("Some{8.0}", "(.Some, (F32.lit1090519040, ()))"),
    "expected_kept": Value("Some{5.0}", f"(.Some, ({FIVE}, ()))"),
    "overwritten_kept": Value(
        "Some{F32.sub(5.0, 20.0)}", f"(.Some, ((F32.sub {FIVE} {TWENTY}), ()))"
    ),
}

MUTANTS = transforms.load_mutants(
    (Path(__file__).resolve().parent / "pointwise/mutants.txt").read_text()
)
SITES = {
    "swapped-operands": Site(
        "step_lift",
        "~f(operation, StorageLaws.expected(List.append(&2, StorageLaws.Op, right_ops",
        "pointwise_2d_proof.step_lift.qvvvvv",
        ("pointwise.right_step.qvv", "pointwise.right_step.qvv_1"),
    ),
    "wrong-write-index": Site(
        "step_lift",
        "U32.add(U32.from_nat(cursor), 1)",
        "pointwise_2d_proof.step_lift.qvvvvv",
        ("pointwise.right_step.qvv", "pointwise.right_step.qvv_1"),
    ),
    "missing-write": Site(
        "step_lift",
        "Core.write_result(",
        "pointwise_2d_proof.step_lift.qvvvvv",
        ("pointwise.right_step.qvv", "pointwise.right_step.qvv_1"),
    ),
    "wrong-input-map": Site(
        "step_lift",
        "left_fits, U32.from_nat(step_lift~ib(plan, cursor))",
        "pointwise_2d_proof.step_lift.qvvvvv",
        ("pointwise.step.qvvvvv", "pointwise.step.qvvvvv_1"),
    ),
    "cursor-off-by-one": Site(
        "run_lift",
        "shape, 1n, Pointwise.State{",
        "pointwise_2d_proof.run_lift.qvvvvv",
        ("pointwise.allocated.qvvvvv", "pointwise.allocated.qvvvvv_1"),
    ),
    "swapped-maps": Site(
        "broadcast_pointwise",
        "Broadcast.right_map, Broadcast.left_map",
        "pointwise_2d_proof.step_lift.qvvvvv_1",
        ("pointwise.step.qvvvvv_1",),
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
    (target / ENTRY).write_text(
        "import Base\n"
        "import ../../../src/tensor/contracts/broadcast.bend as Broadcast\n"
    )


def false_twin(controls: str, witness: str) -> str:
    target, wrong = WITNESSES[witness]
    start = controls.index(f"def {witness}()")
    end = controls.index("}:\n", start) + 2
    header = controls[start:end]
    if header.count(f"{target}()") != 1:
        raise GateError(f"{witness}: false statement target is not unique")
    return (
        controls[:start] + header.replace(f"{target}()", f"{wrong}()") + controls[end:]
    )


def kernel_twin(text: str, witness: str, catalog, replace_declaration) -> str:
    target, wrong = WITNESSES[witness]
    header, body = catalog(text)[witness]
    if header.count(f"== {target} : ") != 1:
        raise GateError(f"{witness}: exported statement target is not unique")
    changed = header.replace(f"== {target} : ", f"== {wrong} : ")
    return replace_declaration(text, witness, changed, body)


def source_reason(output: str, label: str) -> str:
    expected, observed, location = reads._diagnostic(output)
    if "*1" in expected + observed or "*2" in expected + observed:
        raise GateError(f"{label}: quantity/kind mismatch is not value sensitivity")
    if label in WITNESSES:
        value, wrong = (VALUES[name].source for name in WITNESSES[label])
        if (
            location != label
            or expected != f"{{{value} == {wrong} : Maybe<&2, F32>}}"
            or observed != f"{{{value} == {value} : Maybe<&2, F32>}}"
        ):
            raise GateError(f"{label}: wrong concrete public value equation")
    elif label in SITES:
        site, mark = SITES[label].source_def, SITES[label].source_mark
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
        wrong = WITNESSES[label].wrong
        claim, _, value = expected.rpartition(" == ")
        if (
            location != label
            or value != f"{wrong} : ((Maybe -.Q2) -F32)}}"
            or not observed.startswith(f"{claim} == (.Some, (")
            or wrong in observed
        ):
            raise GateError(f"{label}: kernel rejection is not the value statement")
    elif label in SITES:
        site = f"{SRC}{SITES[label].kernel_declaration}"
        if location != site or not expected or expected == observed:
            raise GateError(f"{label}: kernel rejection is not at {site}")
    else:
        raise GateError(f"Unknown control: {label}")
    return f"{location}: expected {expected} observed {observed}"


def replaced(positive: str, artifact: str, catalog, label: str) -> str:
    before = catalog(positive)
    after = catalog(artifact)
    changed = tuple(
        sorted(
            name.removeprefix(SRC)
            for name, (header, body) in after.items()
            if name.startswith(SRC)
            and name in before
            and (header, body.strip()) != (before[name][0], before[name][1].strip())
        )
    )
    if changed != SITES[label].replaced:
        raise GateError(f"{label}: kernel artifact replaces {changed}")
    return ", ".join(changed)


def check_export(text: str, catalog) -> dict[str, str]:
    declarations = catalog(text)
    names = (*CONSUMERS, *WITNESSES, *LIVE_RECORDS, LAW, *VALUES)
    missing = [name for name in names if name not in declarations]
    if missing:
        raise GateError(f"Export omitted declarations: {missing}")
    for name in LIVE_RECORDS:
        if "Σ-" in declarations[name][1]:
            raise GateError(f"Erased field in a LIVE evidence record: {name}")
    header = declarations[LAW][0]
    operation = "_2e_._2f_.._2f_.._2f_src_2f_tensor_2f_elementwise.Operation"
    if not header.startswith(f"{LAW} : ∀+x0 : {operation} -> ") or (
        f"{SRC}broadcast.binary" not in header
    ):
        raise GateError("Law is not stated for every operation over Broadcast.binary")
    for name, value in VALUES.items():
        if declarations[name][1].strip() != value.kernel:
            raise GateError(f"{name} is not {value.kernel}")
    for witness, (target, _) in WITNESSES.items():
        header = declarations[witness][0]
        if not header.endswith(f"== {target} : (Maybe -.Q2 -F32)}}"):
            raise GateError(f"Witness statement target changed: {witness}")
        if "(consumers.result " not in header:
            raise GateError(f"Witness is not about a broadcast result: {witness}")
    return {name: "\n".join(declarations[name]) for name in names}
