import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.common import GateError, check_sources

reads = importlib.import_module("verification.contracts.reads_controls")
clones = importlib.import_module("verification.contracts.clone_controls")
transforms = importlib.import_module("verification.contracts.transform_controls")
values = importlib.import_module("verification.contracts.from_values_controls")
permutes = importlib.import_module("verification.contracts.permute_values_controls")

ROOT = Path(__file__).resolve().parents[2]
CASE_NAMES = (
    "scalar_witness",
    "three_zero_witness",
    "three_one_witness",
    "three_two_witness",
)
CONSUMERS = (
    "consumers.scalar",
    "consumers.three_zero",
    "consumers.three_one",
    "consumers.three_two",
    "consumers.empty",
    "consumers.huge_leading",
    "consumers.huge_trailing",
)
CONTRACT = "_2e_._2f_.._2f_.._2f_src_2f_tensor_2f_construction_2d_contract_2d_"
LAWS = (f"{CONTRACT}LAWS.new_metadata", f"{CONTRACT}LAWS.new_reads")
LIVE_RECORDS = (
    "_2e_._2f_.._2f_.._2f_src_2f_tensor_2f_construction_2d_metadata.NewMetadata.arms",
    f"{CONTRACT}LAWS.NewReads.arms",
    "consumers.Case.arms",
    "consumers.EmptyCase.arms",
)
EXPERIMENT = "verification/contracts/construction"
SEVEN = "F32.lit1088421888"
MAYBE_F32 = ("(Maybe -.Q2 -F32)", "(Maybe.q2 -F32)")
EIGHT = "F32.lit1090519040"


def catalog(text: str) -> dict[str, tuple[str, str]]:
    result = {}
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        header, separator, body = block.partition(" =\n")
        name = header.split(" : ", 1)[0]
        if not separator or " : " not in header or name in result:
            raise GateError(f"Malformed or duplicate exported declaration: {name}")
        result[name] = (header, body)
    return result


def replace_declaration(text: str, name: str, header: str, body: str) -> str:
    old_header, old_body = catalog(text)[name]
    old = f"{old_header} =\n{old_body}"
    if text.count(old) != 1:
        raise GateError(f"Exported declaration is not unique: {name}")
    return text.replace(old, f"{header} =\n{body}")


def fill_body(body: str, zero: str) -> str:
    match = re.match(r"\s*λ[-+]?[A-Za-z_]\w* => λ[-+]?(?P<fill>[A-Za-z_]\w*) => ", body)
    if match is None:
        raise GateError("Inspect actual two-parameter fill lambda before mutation")
    expression, count = re.subn(
        rf"(?<![A-Za-z0-9_.]){re.escape(match.group('fill'))}(?![A-Za-z0-9_.])",
        lambda _: zero,
        body[match.end() :],
    )
    if count == 0:
        raise GateError("No actual fill argument in exported body")
    return body[: match.end()] + expression


def mutation_copy(snapshot: Path, target: Path) -> None:
    shutil.copytree(snapshot, target)
    core = target / "src/tensor/core.bend"
    text = core.read_text()
    old = "new_result(shape, Storage.new(Shape.count(shape), fill))"
    if text.count(old) != 1:
        raise GateError("Actual Core.new mutation target is not unique")
    core.write_text(
        text.replace(old, "new_result(shape, Storage.new(Shape.count(shape), 0.0))")
    )
    proof = target / "src/tensor/allocation-PROOF.bend"
    text = proof.read_text()
    start = text.index("def core_new(+shape:")
    end = text.index("\ndef core_new_done", start)
    header, body = text[start:end].split(":\n", 1)
    body = body.replace(
        "Storage.new(Shape.count(shape), fill)", "Storage.new(Shape.count(shape), 0.0)"
    ).replace(
        "storage_new(Shape.count(shape), fill,", "storage_new(Shape.count(shape), 0.0,"
    )
    proof.write_text(text[:start] + header + ":\n" + body + text[end:])


def failure_copy(snapshot: Path, target: Path) -> None:
    shutil.copytree(snapshot, target)
    core = target / "src/tensor/core.bend"
    text = core.read_text()
    old = "    case Done{storage}:\n      attach(shape, storage)\n\ndef new("
    if text.count(old) != 1:
        raise GateError("Actual Core.new_result success branch is not unique")
    core.write_text(
        text.replace(
            old,
            "    case Done{storage}:\n      Fail{StorageLengthMismatch{}}\n\ndef new(",
        )
    )


def false_twin(controls: str, witness: str) -> str:
    start = controls.index(f"def {witness}()")
    end = controls.index("}:\n", start) + 2
    header = controls[start:end]
    if header.count("expected_value()") != 1:
        raise GateError("False public statement target is not unique")
    return (
        controls[:start]
        + header.replace("expected_value()", "wrong_value()")
        + controls[end:]
    )


def source_value_reason(output: str, label: str) -> str:
    lines = output.splitlines()
    expected = next((x[13:] for x in lines if x.startswith("- expected : ")), "")
    observed = next((x[13:] for x in lines if x.startswith("- observed : ")), "")
    location = next((x[10:] for x in lines if x.startswith("Location: ")), "")
    if "*1" in expected + observed or "*2" in expected + observed:
        raise GateError(f"{label}: quantity/kind mismatch is not value sensitivity")
    if label in CASE_NAMES:
        if (
            expected != "{Some{7.0} == Some{8.0} : Maybe<&2, F32>}"
            or observed != "{Some{7.0} == Some{7.0} : Maybe<&2, F32>}"
            or location != label
        ):
            raise GateError(f"{label}: wrong concrete public value equation")
    elif label == "wrong-fill":
        if (
            not expected.startswith("{Observation.point(shape, index,")
            or not expected.endswith("== Some{fill} : Maybe<&2, F32>}")
            or expected.count(", 0.0, ") != 1
            or expected.replace(", 0.0, ", ", fill, ") != observed
            or location != "point_value"
        ):
            raise GateError("Wrong-fill failure is not the actual public value claim")
    elif (
        location != "new_result_length"
        or not expected.startswith("{None{} == Some{")
        or not observed.startswith("{Allocation.tensor_length(shape, Core.attach")
    ):
        raise GateError("Failure mutant is not rejected for the success claim")
    return f"{location}: expected {expected} observed {observed}"


def kernel_value_reason(output: str, label: str) -> str:
    lines = output.splitlines()
    location = next((x[3:-1] for x in lines if x.startswith("In ")), "")
    expected = next((x[10:] for x in lines if x.startswith("expected: ")), "")
    observed = next((x[10:] for x in lines if x.startswith("observed: ")), "")
    if label in CASE_NAMES:
        if (
            location != label
            or "== wrong_value : " not in expected
            or expected.replace("== wrong_value", "== expected_value") != observed
        ):
            raise GateError(f"{label}: kernel rejection is not the value statement")
    elif label == "unjustified-failure":
        claim = "== (.Some, ((_2e_._2f_.._2f_.._2f_src_2f_shape_2f_core.logical_numel"
        if (
            not location.endswith("allocation_2d_PROOF.new_result_length")
            or "core.new_result +x0) (.Done, (x1, ())))" not in expected
            or "core.attach +x0) x1)" not in observed
            or claim not in expected
            or claim not in observed
        ):
            raise GateError("Failure kernel rejection is not the success claim")
    elif (
        location != f"{CONTRACT}PROOF.point_value"
        or "== (.Some, (x1, ())) : " not in expected
        or "== (.Some, (x1, ())) : " not in observed
        or "core.new +x0) x1)" not in expected
        or "core.new_result" not in observed
    ):
        raise GateError("Wrong-fill kernel rejection is not the public value claim")
    return f"{location}: expected {expected} observed {observed}"


def check_export(text: str) -> dict[str, str]:
    declarations = catalog(text)
    names = (
        *CONSUMERS,
        *CASE_NAMES,
        *LIVE_RECORDS,
        *LAWS,
        "expected_value",
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
    if "∀x3 : {(Nat.is_lt (U32.to_nat x2)" not in declarations[LAWS[1]][0]:
        raise GateError("In-bounds proof is not an ordinary law argument")
    for name in CASE_NAMES:
        header = declarations[name][0].replace(*MAYBE_F32)
        if "core.new +consumers." not in header or SEVEN not in header:
            raise GateError(f"Witness statement is not about Core.new 7.0: {name}")
        if not header.endswith("== expected_value : (Maybe.q2 -F32)}"):
            raise GateError(f"Witness statement target changed: {name}")
    if declarations["expected_value"][1].strip() != f"(.Some, ({SEVEN}, ()))":
        raise GateError("expected_value is not Some(7.0)")
    if declarations["wrong_value"][1].strip() != f"(.Some, ({EIGHT}, ()))":
        raise GateError("wrong_value is not Some(8.0)")
    return {name: "\n".join(declarations[name]) for name in names}


def read_write(snapshot, output, bend, kernel, run, parallel, reasons) -> None:
    experiment = snapshot / reads.EXPERIMENT
    controls = (experiment / "controls.bend").read_text()
    for witness in reads.WITNESSES:
        (experiment / f"false-{witness}.bend").write_text(
            reads.false_twin(controls, witness)
        )
    copies = {}
    for label in reads.MUTANTS:
        copies[label] = output / label
        reads.mutant_copy(snapshot, copies[label], label)
        (copies[label] / reads.EXPERIMENT).mkdir(parents=True, exist_ok=True)
        (copies[label] / reads.EXPERIMENT / "core-entry.bend").write_text(
            "import Base\nimport ../../../src/tensor/core.bend as Core\n"
        )
    target = {
        label: "src/tensor/read-contract.bend"
        if label == "read-unjustified-failure"
        else "src/tensor/write-contract.bend"
        for label in reads.MUTANTS
    }
    texts = parallel(
        [
            (
                "reads-source-consumers",
                [*bend, experiment / "consumers.bend", "--check-only"],
                0,
            ),
            (
                "reads-source-controls",
                [*bend, experiment / "controls.bend", "--check-only"],
                0,
            ),
            *(
                (
                    f"reads-source-false-{witness}",
                    [*bend, experiment / f"false-{witness}.bend", "--check-only"],
                    1,
                )
                for witness in reads.WITNESSES
            ),
            *(
                (
                    f"reads-source-{label}",
                    [*bend, copies[label] / target[label], "--check-only"],
                    1,
                )
                for label in reads.MUTANTS
            ),
            *(
                (
                    f"reads-source-{label}-read-control",
                    [
                        *bend,
                        copies[label] / reads.EXPERIMENT / "read-control.bend",
                        "--check-only",
                    ],
                    0,
                )
                for label in reads.WRITE_MUTANTS
            ),
        ]
    )
    for label in (*reads.WITNESSES, *reads.MUTANTS):
        name = (
            f"reads-source-false-{label}"
            if label in reads.WITNESSES
            else f"reads-source-{label}"
        )
        reasons[name] = reads.source_reason(texts[name], label)

    positive = output / "reads-controls.bendtt"
    exports = {
        label: output / f"reads-{label}-core.bendtt" for label in reads.WRITE_MUTANTS
    }
    parallel(
        [
            (
                "reads-verdict-consumers",
                [*bend, experiment / "consumers.bend", "--verdict"],
                0,
            ),
            (
                "reads-verdict-controls",
                [*bend, experiment / "controls.bend", "--verdict"],
                0,
            ),
            (
                "reads-export-controls",
                [*bend, experiment / "controls.bend", "-o", positive],
                0,
            ),
            *(
                (
                    f"reads-export-{label}-core",
                    [
                        *bend,
                        copies[label] / reads.EXPERIMENT / "core-entry.bend",
                        "-o",
                        exports[label],
                    ],
                    0,
                )
                for label in reads.WRITE_MUTANTS
            ),
        ]
    )
    text = positive.read_text()
    (output / "reads-export-statements.txt").write_text(
        "\n\n".join(reads.check_export(text, catalog).values()) + "\n"
    )
    declarations = catalog(text)
    artifacts = {}
    for witness in reads.WITNESSES:
        header, body = declarations[witness]
        target = reads.WITNESSES[witness].target
        artifacts[witness] = replace_declaration(
            text, witness, header.replace(f"== {target}", "== wrong_value"), body
        )
    for label in reads.WRITE_MUTANTS:
        artifacts[label] = reads.transplant(text, exports[label].read_text(), catalog)
    artifacts["read-unjustified-failure"] = reads.refuse_read(text, catalog)
    for label, artifact in artifacts.items():
        (output / f"reads-false-{label}.bendtt").write_text(artifact)
    texts = parallel(
        [
            ("reads-kernel-controls", [kernel, positive], 0),
            *(
                (
                    f"reads-kernel-false-{label}",
                    [kernel, output / f"reads-false-{label}.bendtt"],
                    1,
                )
                for label in artifacts
            ),
        ]
    )
    for label in artifacts:
        reasons[f"reads-kernel-false-{label}"] = reads.kernel_reason(
            texts[f"reads-kernel-false-{label}"], label
        )


def clone_metadata(snapshot, output, bend, kernel, run, parallel, reasons) -> None:
    experiment = snapshot / clones.EXPERIMENT
    controls = (experiment / "controls.bend").read_text()
    for witness in clones.WITNESSES:
        (experiment / f"false-{witness}.bend").write_text(
            clones.false_twin(controls, witness)
        )
    copies = {}
    for label in clones.MUTANTS:
        copies[label] = output / f"clone-{label}"
        clones.mutant_copy(snapshot, copies[label], label)
    other = {
        "src/tensor/clone-contract.bend": "src/tensor/metadata-contract.bend",
        "src/tensor/metadata-contract.bend": "src/tensor/clone-contract.bend",
    }
    unaffected = [label for label in clones.MUTANTS if label != "count-off-by-one"]
    texts = parallel(
        [
            (
                "clone-source-consumers",
                [*bend, experiment / "consumers.bend", "--check-only"],
                0,
            ),
            (
                "clone-source-controls",
                [*bend, experiment / "controls.bend", "--check-only"],
                0,
            ),
            *(
                (
                    f"clone-source-false-{witness}",
                    [*bend, experiment / f"false-{witness}.bend", "--check-only"],
                    1,
                )
                for witness in clones.WITNESSES
            ),
            *(
                (
                    f"clone-source-{label}",
                    [*bend, copies[label] / clones.SITES[label].proof, "--check-only"],
                    1,
                )
                for label in clones.MUTANTS
            ),
            *(
                (
                    f"clone-source-{label}-other-contract",
                    [
                        *bend,
                        copies[label] / other[clones.SITES[label].proof],
                        "--check-only",
                    ],
                    0,
                )
                for label in unaffected
            ),
        ]
    )
    for label in (*clones.WITNESSES, *clones.MUTANTS):
        name = (
            f"clone-source-false-{label}"
            if label in clones.WITNESSES
            else f"clone-source-{label}"
        )
        reasons[name] = clones.source_reason(texts[name], label)

    positive = output / "clone-controls.bendtt"
    exports = {label: output / f"clone-{label}-core.bendtt" for label in clones.MUTANTS}
    parallel(
        [
            (
                "clone-verdict-consumers",
                [*bend, experiment / "consumers.bend", "--verdict"],
                0,
            ),
            (
                "clone-verdict-controls",
                [*bend, experiment / "controls.bend", "--verdict"],
                0,
            ),
            (
                "clone-export-controls",
                [*bend, experiment / "controls.bend", "-o", positive],
                0,
            ),
            *(
                (
                    f"clone-export-{label}-core",
                    [
                        *bend,
                        copies[label] / clones.EXPERIMENT / "core-entry.bend",
                        "-o",
                        exports[label],
                    ],
                    0,
                )
                for label in clones.MUTANTS
            ),
        ]
    )
    text = positive.read_text()
    (output / "clone-export-statements.txt").write_text(
        "\n\n".join(clones.check_export(text, catalog).values()) + "\n"
    )
    declarations = catalog(text)
    artifacts = {}
    for witness in clones.WITNESSES:
        header, body = declarations[witness]
        target = clones.WITNESSES[witness].target
        replacement = clones.WITNESSES[witness].replacement
        artifacts[witness] = replace_declaration(
            text, witness, header.replace(f"== {target}", f"== {replacement}"), body
        )
    for label in clones.MUTANTS:
        artifacts[label] = reads.transplant(text, exports[label].read_text(), catalog)
    for label, artifact in artifacts.items():
        (output / f"clone-false-{label}.bendtt").write_text(artifact)
    texts = parallel(
        [
            ("clone-kernel-controls", [kernel, positive], 0),
            *(
                (
                    f"clone-kernel-false-{label}",
                    [kernel, output / f"clone-false-{label}.bendtt"],
                    1,
                )
                for label in artifacts
            ),
        ]
    )
    for label in artifacts:
        reasons[f"clone-kernel-false-{label}"] = clones.kernel_reason(
            texts[f"clone-kernel-false-{label}"], label
        )


def transform(snapshot, output, bend, kernel, run, parallel, reasons) -> None:
    experiment = snapshot / transforms.EXPERIMENT
    controls = (experiment / "controls.bend").read_text()
    for witness in transforms.WITNESSES:
        (experiment / f"false-{witness}.bend").write_text(
            transforms.false_twin(controls, witness)
        )
    copies = {}
    for label in transforms.MUTANTS:
        copies[label] = output / f"transform-{label}"
        transforms.mutant_copy(snapshot, copies[label], label)
    texts = parallel(
        [
            (
                "transform-source-consumers",
                [*bend, experiment / "consumers.bend", "--check-only"],
                0,
            ),
            (
                "transform-source-controls",
                [*bend, experiment / "controls.bend", "--check-only"],
                0,
            ),
            *(
                (
                    f"transform-source-false-{witness}",
                    [*bend, experiment / f"false-{witness}.bend", "--check-only"],
                    1,
                )
                for witness in transforms.WITNESSES
            ),
            *(
                (
                    f"transform-source-{label}",
                    [
                        *bend,
                        copies[label] / transforms.SITES[label].proof,
                        "--check-only",
                    ],
                    1,
                )
                for label in transforms.MUTANTS
            ),
            *(
                (
                    f"transform-source-{label}-other-contract",
                    [
                        *bend,
                        copies[label] / transforms.OTHER[transforms.SITES[label].proof],
                        "--check-only",
                    ],
                    0,
                )
                for label in transforms.MUTANTS
            ),
        ]
    )
    for label in (*transforms.WITNESSES, *transforms.MUTANTS):
        name = (
            f"transform-source-false-{label}"
            if label in transforms.WITNESSES
            else f"transform-source-{label}"
        )
        reasons[name] = transforms.source_reason(texts[name], label)

    positive = output / "transform-controls.bendtt"
    exports = {
        label: output / f"transform-{label}-module.bendtt"
        for label in transforms.MUTANTS
    }
    parallel(
        [
            (
                "transform-verdict-consumers",
                [*bend, experiment / "consumers.bend", "--verdict"],
                0,
            ),
            (
                "transform-verdict-controls",
                [*bend, experiment / "controls.bend", "--verdict"],
                0,
            ),
            (
                "transform-export-controls",
                [*bend, experiment / "controls.bend", "-o", positive],
                0,
            ),
            *(
                (
                    f"transform-export-{label}-module",
                    [
                        *bend,
                        copies[label] / transforms.EXPERIMENT / "module-entry.bend",
                        "-o",
                        exports[label],
                    ],
                    0,
                )
                for label in transforms.MUTANTS
            ),
        ]
    )
    text = positive.read_text()
    (output / "transform-export-statements.txt").write_text(
        "\n\n".join(transforms.check_export(text, catalog).values()) + "\n"
    )
    declarations = catalog(text)
    artifacts = {}
    for witness in transforms.WITNESSES:
        header, body = declarations[witness]
        target = transforms.WITNESSES[witness].target
        replacement = transforms.WITNESSES[witness].replacement
        artifacts[witness] = replace_declaration(
            text, witness, header.replace(f"== {target}", f"== {replacement}"), body
        )
    for label in transforms.MUTANTS:
        artifacts[label] = reads.transplant(
            text, exports[label].read_text(), catalog, transforms.export_prefix(label)
        )
    for label, artifact in artifacts.items():
        (output / f"transform-false-{label}.bendtt").write_text(artifact)
    texts = parallel(
        [
            ("transform-kernel-controls", [kernel, positive], 0),
            *(
                (
                    f"transform-kernel-false-{label}",
                    [kernel, output / f"transform-false-{label}.bendtt"],
                    1,
                )
                for label in artifacts
            ),
        ]
    )
    for label in artifacts:
        reasons[f"transform-kernel-false-{label}"] = transforms.kernel_reason(
            texts[f"transform-kernel-false-{label}"], label
        )


def from_values(snapshot, output, bend, kernel, run, parallel, reasons) -> None:
    experiment = snapshot / values.EXPERIMENT
    controls = (experiment / "controls.bend").read_text()
    for witness in values.WITNESSES:
        (experiment / f"false-{witness}.bend").write_text(
            values.false_twin(controls, witness)
        )
    copies = {}
    for label in values.MUTANTS:
        copies[label] = output / label
        values.mutant_copy(snapshot, copies[label], label)
    texts = parallel(
        [
            (
                "values-source-consumers",
                [*bend, experiment / "consumers.bend", "--check-only"],
                0,
            ),
            (
                "values-source-controls",
                [*bend, experiment / "controls.bend", "--check-only"],
                0,
            ),
            *(
                (
                    f"values-source-false-{witness}",
                    [*bend, experiment / f"false-{witness}.bend", "--check-only"],
                    1,
                )
                for witness in values.WITNESSES
            ),
            *(
                (
                    f"values-source-{label}",
                    [*bend, copies[label] / values.PROOF, "--check-only"],
                    1,
                )
                for label in values.MUTANTS
            ),
            *(
                (
                    f"values-source-{label}-construction",
                    [*bend, copies[label] / values.OTHER, "--check-only"],
                    0,
                )
                for label in values.MUTANTS
            ),
        ]
    )
    for label in (*values.WITNESSES, *values.MUTANTS):
        name = (
            f"values-source-false-{label}"
            if label in values.WITNESSES
            else f"values-source-{label}"
        )
        reasons[name] = values.source_reason(texts[name], label)

    positive = output / "values-controls.bendtt"
    exports = {
        label: output / f"values-{label}-core.bendtt" for label in values.MUTANTS
    }
    parallel(
        [
            (
                "values-verdict-consumers",
                [*bend, experiment / "consumers.bend", "--verdict"],
                0,
            ),
            (
                "values-verdict-controls",
                [*bend, experiment / "controls.bend", "--verdict"],
                0,
            ),
            (
                "values-export-controls",
                [*bend, experiment / "controls.bend", "-o", positive],
                0,
            ),
            *(
                (
                    f"values-export-{label}-core",
                    [
                        *bend,
                        copies[label] / values.EXPERIMENT / "core-entry.bend",
                        "-o",
                        exports[label],
                    ],
                    0,
                )
                for label in values.MUTANTS
            ),
        ]
    )
    text = positive.read_text()
    (output / "values-export-statements.txt").write_text(
        "\n\n".join(values.check_export(text, catalog).values()) + "\n"
    )
    declarations = catalog(text)
    artifacts = {}
    for witness in values.WITNESSES:
        header, body = declarations[witness]
        target = values.WITNESSES[witness].target
        artifacts[witness] = replace_declaration(
            text, witness, header.replace(f"== {target}", "== wrong_value"), body
        )
    for label in values.MUTANTS:
        artifacts[label] = reads.transplant(text, exports[label].read_text(), catalog)
    for label, artifact in artifacts.items():
        (output / f"values-false-{label}.bendtt").write_text(artifact)
    texts = parallel(
        [
            ("values-kernel-controls", [kernel, positive], 0),
            *(
                (
                    f"values-kernel-false-{label}",
                    [kernel, output / f"values-false-{label}.bendtt"],
                    1,
                )
                for label in artifacts
            ),
        ]
    )
    for label in artifacts:
        reasons[f"values-kernel-false-{label}"] = values.kernel_reason(
            texts[f"values-kernel-false-{label}"], label
        )


def permute_values(snapshot, output, bend, kernel, run, parallel, reasons) -> None:
    experiment = snapshot / permutes.EXPERIMENT
    controls = (experiment / "controls.bend").read_text()
    for witness in permutes.WITNESSES:
        (experiment / f"false-{witness}.bend").write_text(
            permutes.false_twin(controls, witness)
        )
    copies = {}
    for label in permutes.MUTANTS:
        copies[label] = output / f"permute-{label}"
        permutes.mutant_copy(snapshot, copies[label], label)
    texts = parallel(
        [
            (
                "permute-source-consumers",
                [*bend, experiment / "consumers.bend", "--check-only"],
                0,
            ),
            (
                "permute-source-controls",
                [*bend, experiment / "controls.bend", "--check-only"],
                0,
            ),
            *(
                (
                    f"permute-source-false-{witness}",
                    [*bend, experiment / f"false-{witness}.bend", "--check-only"],
                    1,
                )
                for witness in permutes.WITNESSES
            ),
            *(
                (
                    f"permute-source-{label}",
                    [*bend, copies[label] / permutes.PROOF, "--check-only"],
                    1,
                )
                for label in permutes.MUTANTS
            ),
            *(
                (
                    f"permute-source-{label}-other-contract",
                    [*bend, copies[label] / permutes.OTHER, "--check-only"],
                    0,
                )
                for label in permutes.MUTANTS
            ),
        ]
    )
    for label in (*permutes.WITNESSES, *permutes.MUTANTS):
        name = (
            f"permute-source-false-{label}"
            if label in permutes.WITNESSES
            else f"permute-source-{label}"
        )
        reasons[name] = permutes.source_reason(texts[name], label)

    positive = output / "permute-controls.bendtt"
    exports = {
        label: output / f"permute-{label}-module.bendtt" for label in permutes.MUTANTS
    }
    parallel(
        [
            (
                "permute-verdict-consumers",
                [*bend, experiment / "consumers.bend", "--verdict"],
                0,
            ),
            (
                "permute-verdict-controls",
                [*bend, experiment / "controls.bend", "--verdict"],
                0,
            ),
            (
                "permute-export-controls",
                [*bend, experiment / "controls.bend", "-o", positive],
                0,
            ),
            *(
                (
                    f"permute-export-{label}-module",
                    [
                        *bend,
                        copies[label] / permutes.EXPERIMENT / "module-entry.bend",
                        "-o",
                        exports[label],
                    ],
                    0,
                )
                for label in permutes.MUTANTS
            ),
        ]
    )
    text = positive.read_text()
    (output / "permute-export-statements.txt").write_text(
        "\n\n".join(permutes.check_export(text, catalog).values()) + "\n"
    )
    declarations = catalog(text)
    artifacts = {}
    for witness in permutes.WITNESSES:
        header, body = declarations[witness]
        target = permutes.WITNESSES[witness].target
        artifacts[witness] = replace_declaration(
            text, witness, header.replace(f"== {target}", "== wrong_value"), body
        )
    for label in permutes.MUTANTS:
        artifacts[label] = reads.transplant(
            text, exports[label].read_text(), catalog, permutes.export_prefix(label)
        )
    for label, artifact in artifacts.items():
        (output / f"permute-false-{label}.bendtt").write_text(artifact)
    texts = parallel(
        [
            ("permute-kernel-controls", [kernel, positive], 0),
            *(
                (
                    f"permute-kernel-false-{label}",
                    [kernel, output / f"permute-false-{label}.bendtt"],
                    1,
                )
                for label in artifacts
            ),
        ]
    )
    for label in artifacts:
        reasons[f"permute-kernel-false-{label}"] = permutes.kernel_reason(
            texts[f"permute-kernel-false-{label}"], label
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bend-root", type=Path, default=ROOT / ".tools/bend")
    parser.add_argument("--kernel", type=Path, default=ROOT / ".tools/kernel/bendtt")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    bend_root = args.bend_root.resolve()
    kernel = args.kernel.resolve()
    output.mkdir(parents=True)
    pinned = bend_root == (ROOT / ".tools/bend").resolve()
    if pinned:
        check_sources(ROOT / ".tools")
    env = os.environ | {
        "BEND_NO_TELEMETRY": "1",
        "BENDTT": str(kernel),
        "LEAN_STACK_SIZE_KB": "4194304",
        "LEAN_NUM_THREADS": "1",
    }
    records: list[dict] = []
    summary = {
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "repository_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "src", "verification/contracts"],
                cwd=ROOT,
                text=True,
            )
        ),
        "bend_commit": subprocess.check_output(
            ["git", "-C", str(bend_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "bend_is_repository_pin": pinned,
        "kernel": str(kernel),
        "affinity": sorted(os.sched_getaffinity(0)),
        "workers": args.workers,
        "commands": records,
    }

    def save() -> None:
        records.sort(key=lambda row: row["name"])
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def run(name: str, command: list, expected_code: int) -> str:
        start = time.monotonic()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=1500,
            check=False,
        )
        text = result.stdout + result.stderr
        (output / f"{name}.out").write_text(text)
        records.append(
            {
                "name": name,
                "command": [str(part) for part in command],
                "exit_code": result.returncode,
                "expected_code": expected_code,
                "seconds": round(time.monotonic() - start, 1),
            }
        )
        if result.returncode != expected_code:
            raise GateError(f"{name}: exit {result.returncode}; see {output}")
        if (
            expected_code == 0
            and "-o" not in command
            and "ALL PROOFS CHECK" not in text
        ):
            raise GateError(f"{name}: no ALL PROOFS CHECK; see {output}")
        if expected_code == 1 and "SOME PROOFS FAIL" not in text:
            raise GateError(f"{name}: not a proof rejection; see {output}")
        return text

    bend = ["bun", str(bend_root / "bend2/main.ts")]
    snapshot = output / "snapshot"
    shutil.copytree(ROOT / "src", snapshot / "src")
    shutil.copytree(ROOT / EXPERIMENT, snapshot / EXPERIMENT)
    shutil.copytree(ROOT / reads.EXPERIMENT, snapshot / reads.EXPERIMENT)
    shutil.copytree(ROOT / clones.EXPERIMENT, snapshot / clones.EXPERIMENT)
    shutil.copytree(ROOT / transforms.EXPERIMENT, snapshot / transforms.EXPERIMENT)
    shutil.copytree(ROOT / values.EXPERIMENT, snapshot / values.EXPERIMENT)
    shutil.copytree(ROOT / permutes.EXPERIMENT, snapshot / permutes.EXPERIMENT)
    experiment = snapshot / EXPERIMENT
    controls = (experiment / "controls.bend").read_text()
    for witness in CASE_NAMES:
        (experiment / f"false-{witness}.bend").write_text(false_twin(controls, witness))
    wrong_fill = output / "wrong-fill"
    mutation_copy(snapshot, wrong_fill)
    failing = output / "unjustified-failure"
    failure_copy(snapshot, failing)
    reasons: dict[str, str] = {}
    summary["rejection_reasons"] = reasons

    def parallel(jobs: list[tuple[str, list, int]]) -> dict[str, str]:
        with ThreadPoolExecutor(args.workers) as pool:
            return dict(
                zip(
                    (name for name, _, _ in jobs),
                    pool.map(lambda job: run(*job), jobs),
                    strict=True,
                )
            )

    try:
        contract = "src/tensor/construction-contract-PROOF.bend"
        texts = parallel(
            [
                (
                    "source-controls",
                    [*bend, experiment / "controls.bend", "--check-only"],
                    0,
                ),
                (
                    "source-metadata",
                    [*bend, experiment / "metadata-control.bend", "--check-only"],
                    0,
                ),
                (
                    "source-wrong-fill-metadata",
                    [
                        *bend,
                        wrong_fill / EXPERIMENT / "metadata-control.bend",
                        "--check-only",
                    ],
                    0,
                ),
                (
                    "source-wrong-fill",
                    [*bend, wrong_fill / contract, "--check-only"],
                    1,
                ),
                (
                    "source-unjustified-failure",
                    [*bend, failing / contract, "--check-only"],
                    1,
                ),
                *(
                    (
                        f"source-false-{witness}",
                        [*bend, experiment / f"false-{witness}.bend", "--check-only"],
                        1,
                    )
                    for witness in CASE_NAMES
                ),
            ]
        )
        for witness in CASE_NAMES:
            reasons[f"source-false-{witness}"] = source_value_reason(
                texts[f"source-false-{witness}"], witness
            )
        for label in ("wrong-fill", "unjustified-failure"):
            reasons[f"source-{label}"] = source_value_reason(
                texts[f"source-{label}"], label
            )

        positive = output / "controls.bendtt"
        metadata = output / "wrong-fill-metadata.bendtt"
        metadata_entry = wrong_fill / EXPERIMENT / "metadata-control.bend"
        parallel(
            [
                (
                    "verdict-controls",
                    [*bend, experiment / "controls.bend", "--verdict"],
                    0,
                ),
                (
                    "verdict-wrong-fill-metadata",
                    [*bend, metadata_entry, "--verdict"],
                    0,
                ),
                (
                    "export-controls",
                    [*bend, experiment / "controls.bend", "-o", positive],
                    0,
                ),
                (
                    "export-wrong-fill-metadata",
                    [*bend, metadata_entry, "-o", metadata],
                    0,
                ),
            ]
        )
        text = positive.read_text()
        (output / "export-statements.txt").write_text(
            "\n\n".join(check_export(text).values()) + "\n"
        )
        declarations = catalog(text)
        artifacts = {}
        for witness in CASE_NAMES:
            header, body = declarations[witness]
            artifacts[witness] = replace_declaration(
                text,
                witness,
                header.replace("== expected_value", "== wrong_value"),
                body,
            )
        zero = declarations[declarations["wrong_fill_value"][1].strip()][1].strip()
        wrong = text
        for suffix in ("_2f_core.new", "_2f_allocation_2d_PROOF.core_new"):
            names = [name for name in declarations if name.endswith(suffix)]
            if len(names) != 1:
                raise GateError(f"Wrong-fill export target is ambiguous: {suffix}")
            header, body = declarations[names[0]]
            wrong = replace_declaration(wrong, names[0], header, fill_body(body, zero))
        artifacts["wrong-fill"] = wrong
        names = [name for name in declarations if name.endswith("_2f_core.new_result")]
        if len(names) != 1:
            raise GateError("Failure export target is ambiguous: core.new_result")
        header, body = declarations[names[0]]
        attach = re.search(r"\(\S+_2f_core\.attach \+x0 x1\)", body)
        mismatch = re.search(r"\(\.\S+_2f_core\.StorageError, \(x1, \(\)\)\)", body)
        if attach is None or mismatch is None or body.count(attach.group(0)) != 1:
            raise GateError("Inspect exported new_result before the failure mutation")
        failure = (
            "(.Fail, ((."
            + mismatch.group(0)[2:]
            .split(",", 1)[0]
            .replace("StorageError", "StorageLengthMismatch")
            + ", ()), ()))"
        )
        artifacts["unjustified-failure"] = replace_declaration(
            text, names[0], header, body.replace(attach.group(0), failure)
        )
        for label, artifact in artifacts.items():
            (output / f"false-{label}.bendtt").write_text(artifact)
        texts = parallel(
            [
                ("kernel-controls", [kernel, positive], 0),
                ("kernel-wrong-fill-metadata", [kernel, metadata], 0),
                *(
                    (
                        f"kernel-false-{label}",
                        [kernel, output / f"false-{label}.bendtt"],
                        1,
                    )
                    for label in artifacts
                ),
            ]
        )
        for label in artifacts:
            reasons[f"kernel-false-{label}"] = kernel_value_reason(
                texts[f"kernel-false-{label}"], label
            )
        read_write(snapshot, output, bend, kernel, run, parallel, reasons)
        clone_metadata(snapshot, output, bend, kernel, run, parallel, reasons)
        transform(snapshot, output, bend, kernel, run, parallel, reasons)
        from_values(snapshot, output, bend, kernel, run, parallel, reasons)
        permute_values(snapshot, output, bend, kernel, run, parallel, reasons)
        summary["passed"] = True
    except (GateError, subprocess.TimeoutExpired) as error:
        summary["passed"] = False
        summary["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
