import re
import shutil
from collections.abc import Callable
from pathlib import Path

from tools.common import GateError

EXPERIMENT = "verification/r1-b/reads"
CORE = "_2e_._2f_.._2f_.._2f_src_2f_tensor_2f_core."
SRC = "_2e_._2f_.._2f_.._2f_src_2f_tensor_2f_"
CONTRACT = f"{SRC}read_2d_write_2d_contract_2d_LAWS."
LAWS = tuple(
    f"{CONTRACT}{name}"
    for name in (
        "read_inside",
        "read_outside",
        "write_inside",
        "write_outside",
        "coordinate_inside",
        "dynamic_read_inside",
    )
)
LIVE_RECORDS = tuple(
    f"{SRC}read_2d_write_2d_records.{name}.arms"
    for name in (
        "ReadContract",
        "ReadInside",
        "WriteContract",
        "WriteInside",
        "WriteOutside",
        "CoordinateInside",
    )
) + tuple(
    f"consumers.{name}.arms"
    for name in ("ReadCase", "WriteCase", "OutsideCase", "EmptyCase", "CornerCase")
)
CONSUMERS = tuple(
    f"consumers.{name}"
    for name in (
        "scalar_read",
        "three_read_zero",
        "three_read_one",
        "three_read_two",
        "grid_read_last",
        "three_write_zero",
        "three_write_one",
        "three_write_two",
        "three_outside",
        "scalar_outside",
        "empty",
        "corner_first",
        "corner_row",
        "corner_column",
        "corner_last",
        "dynamic_read",
    )
)
WITNESSES = {
    "read_one_witness": ("expected_fill", "Some{7.0}"),
    "written_one_witness": ("expected_written", "Some{9.0}"),
    "neighbour_two_witness": ("expected_fill", "Some{7.0}"),
    "neighbour_zero_witness": ("expected_fill", "Some{7.0}"),
}
SEVEN = "(.Some, (F32.lit1088421888, ()))"
NINE = "(.Some, (F32.lit1091567616, ()))"
EIGHT = "(.Some, (F32.lit1090519040, ()))"

MUTANTS_FILE = Path(__file__).resolve().parent / "reads/mutants.txt"


def load_mutants(text: str) -> dict[str, tuple[str, str]]:
    """Parse `### <label> before` / `### <label> after` blocks of Bend source."""
    sections = re.split(r"^### (\S+) (before|after)\n", text, flags=re.M)
    pairs: dict[str, dict[str, str]] = {}
    for label, side, body in zip(
        sections[1::3], sections[2::3], sections[3::3], strict=True
    ):
        pairs.setdefault(label, {})[side] = body.rstrip("\n")
    if any(set(sides) != {"before", "after"} for sides in pairs.values()):
        raise GateError("Each mutant needs one before and one after block")
    return {label: (sides["before"], sides["after"]) for label, sides in pairs.items()}


# Each mutant edits only src/tensor/core.bend: the public operation, never a proof.
MUTANTS = load_mutants(MUTANTS_FILE.read_text())
WRITE_MUTANTS = (
    "wrong-index",
    "missing-write",
    "corrupted-neighbour",
    "write-unjustified-failure",
)


def mutant_copy(snapshot: Path, target: Path, label: str) -> None:
    shutil.copytree(snapshot, target)
    core = target / "src/tensor/core.bend"
    text = core.read_text()
    old, new = MUTANTS[label]
    if text.count(old) != 1:
        raise GateError(f"{label}: mutation target in core.bend is not unique")
    core.write_text(text.replace(old, new))


def false_twin(controls: str, witness: str) -> str:
    expected = WITNESSES[witness][0]
    start = controls.index(f"def {witness}()")
    end = controls.index("}:\n", start) + 2
    header = controls[start:end]
    if header.count(f"{expected}()") != 1:
        raise GateError(f"{witness}: false statement target is not unique")
    return (
        controls[:start]
        + header.replace(f"{expected}()", "wrong_value()")
        + controls[end:]
    )


def _diagnostic(output: str) -> tuple[str, str, str]:
    lines = output.splitlines()
    expected = next((x[13:] for x in lines if x.startswith("- expected : ")), "")
    observed = next((x[13:] for x in lines if x.startswith("- observed : ")), "")
    location = next((x[10:] for x in lines if x.startswith("Location: ")), "")
    return expected, observed, location


def source_reason(output: str, label: str) -> str:
    expected, observed, location = _diagnostic(output)
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
    elif label in ("wrong-index", "missing-write", "corrupted-neighbour"):
        claim = "== Some{Bool.pick(F32, Bool.and("
        if (
            location != "write_value_done"
            or claim not in expected
            or claim not in observed
            or not observed.startswith(
                "{Observation.written_value_pair(shape, other, Core.write_result("
            )
            or "Storage.set(" not in observed
            or expected == observed
        ):
            raise GateError(f"{label}: not rejected at the written value claim")
        marks = {
            "wrong-index": "U32.add(index, 1)",
            "missing-write": "Core.read_result(",
            "corrupted-neighbour": "Core.corrupt_neighbour(",
        }
        if marks[label] not in expected:
            raise GateError(f"{label}: rejection does not show the mutated write")
    elif label == "write-unjustified-failure":
        if (
            location != "write_inside"
            or expected != "Laws.WriteContract<shape, fill, index, value, 0>"
            or observed != "Laws.WriteContract<shape, fill, index, value, 8>"
        ):
            raise GateError(f"{label}: not rejected at the success code")
    elif label == "read-unjustified-failure":
        if (
            location != "read_result_value"
            or expected != "None{}"
            or observed != "Some{value}"
        ):
            raise GateError(f"{label}: not rejected at the read value")
    else:
        raise GateError(f"Unknown control: {label}")
    return f"{location}: expected {expected} observed {observed}"


def kernel_reason(output: str, label: str) -> str:
    lines = output.splitlines()
    location = next((x[3:-1] for x in lines if x.startswith("In ")), "")
    expected = next((x[10:] for x in lines if x.startswith("expected: ")), "")
    observed = next((x[10:] for x in lines if x.startswith("observed: ")), "")
    if label in WITNESSES:
        target = WITNESSES[label][0]
        if (
            location != label
            or "== wrong_value : " not in expected
            or expected.replace("== wrong_value", f"== {target}") != observed
        ):
            raise GateError(f"{label}: kernel rejection is not the value statement")
    elif label in ("wrong-index", "missing-write", "corrupted-neighbour"):
        pick = f"{SRC}write_2d_contract.written_pick"
        if (
            location != f"{SRC}write_2d_contract.write_value_done"
            or pick not in expected
            or pick not in observed
            or f"{SRC}observation.written_value_pair" not in observed
            or f"{CORE}write_result" not in observed
        ):
            raise GateError(f"{label}: kernel rejection is not the written value")
    elif label == "write-unjustified-failure":
        if (
            location != f"{SRC}write_2d_contract.write_inside"
            or f"{SRC}read_2d_write_2d_records.WriteContract" not in expected + observed
        ):
            raise GateError(f"{label}: kernel rejection is not the success code")
    elif label == "read-unjustified-failure":
        done = "(.Tuple, (x4, ((.Done, (x5, ())), ())))"
        if (
            location != f"{CORE}read_result_value"
            or not expected.startswith(f"(({CORE}read_value -x0)")
            or done not in expected
            or observed != f"({SRC}lifecycle.read_value {done})"
        ):
            raise GateError(f"{label}: kernel rejection is not the read value")
    else:
        raise GateError(f"Unknown control: {label}")
    return f"{location}: expected {expected} observed {observed}"


def transplant(
    positive: str,
    mutant: str,
    catalog: Callable[[str], dict[str, tuple[str, str]]],
    module: str = CORE,
) -> str:
    """Put the mutant module's declarations into the positive export.

    Changed declarations of `module` replace their originals in place. Declarations
    only the mutant has (new helpers, Base operations they call) and positive
    declarations the mutant needs earlier are placed where the mutant has them,
    so every declaration still precedes its uses.
    """
    kept = catalog(positive)
    mutated = catalog(mutant)
    changed = [
        name
        for name, declaration in mutated.items()
        if name.startswith(module) and name in kept and kept[name] != declaration
    ]
    if not changed:
        raise GateError(f"Mutant export changes no declaration in {module}")
    added = [name for name in mutated if name not in kept]
    order = list(mutated)
    first = min(order.index(name) for name in changed)
    anchor = order[first]
    positive_order = list(kept)
    hoisted = [
        name
        for name in order[:first]
        if name in kept and positive_order.index(name) > positive_order.index(anchor)
    ]
    sequence = [name for name in positive_order if name not in hoisted]
    at = sequence.index(anchor)
    before = [name for name in order[:first] if name in added or name in hoisted]
    after = [name for name in order[first:] if name in added]
    sequence = sequence[:at] + before + [anchor] + after + sequence[at + 1 :]
    declarations = kept | {name: mutated[name] for name in changed + added}
    return (
        "\n\n".join(f"{declarations[n][0]} =\n{declarations[n][1]}" for n in sequence)
        + "\n"
    )


def refuse_read(
    positive: str, catalog: Callable[[str], dict[str, tuple[str, str]]]
) -> str:
    """Make the exported storage_read_error report OutOfBounds on success."""
    name = f"{CORE}storage_read_error"
    header, body = catalog(positive)[name]
    done = "(.Done, (x0, ()))"
    failure = (
        f"(.Fail, ((.{CORE}StorageError, "
        f"((._2e_._2f_.._2f_.._2f_src_2f_storage_2f_main.OutOfBounds, ()), ())), ()))"
    )
    if body.count(done) != 1 or f"{CORE}StorageError" not in body:
        raise GateError("Inspect exported storage_read_error before the mutation")
    old = f"{header} =\n{body}"
    if positive.count(old) != 1:
        raise GateError("Exported storage_read_error is not unique")
    return positive.replace(old, f"{header} =\n{body.replace(done, failure)}")


def check_export(
    text: str, catalog: Callable[[str], dict[str, tuple[str, str]]]
) -> dict[str, str]:
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
        if not re.search(r"∀x\d+ : \{\(Nat\.is_lt \(U32\.to_nat x2\)", header) and (
            "coordinate" not in name
        ):
            raise GateError(f"Bounds proof is not an ordinary law argument: {name}")
    values = {
        "expected_fill": SEVEN,
        "expected_written": NINE,
        "wrong_value": EIGHT,
    }
    for name, value in values.items():
        if declarations[name][1].strip() != value:
            raise GateError(f"{name} is not {value}")
    for witness, (target, _) in WITNESSES.items():
        header = declarations[witness][0]
        if not header.endswith(
            f"== {target} : (Maybe -.Q2 -F32)}}"
        ) and not header.endswith(f"== {target} : (Maybe.q2 -F32)}}"):
            raise GateError(f"Witness statement target changed: {witness}")
        if f"{CORE}new" not in header or "F32.lit1088421888" not in header:
            raise GateError(f"Witness is not about Core.new 7.0: {witness}")
    return {name: "\n".join(declarations[name]) for name in names}
