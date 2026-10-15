import importlib
import tempfile
import unittest
from pathlib import Path

from tools.common import GateError

contracts = importlib.import_module("verification.contracts.run")
controls = importlib.import_module("verification.families.controls")
pointwise = importlib.import_module("verification.families.pointwise_controls").FAMILY
fold = importlib.import_module("verification.families.fold_controls").FAMILY
reduce = importlib.import_module("verification.families.reduce_controls").FAMILY
broadcast_to = importlib.import_module(
    "verification.families.broadcast_to_controls"
).FAMILY
FAMILIES = (pointwise, fold, reduce, broadcast_to)
ROOT = Path(__file__).resolve().parents[2]


def diagnostic(expected: str, observed: str, location: str) -> str:
    return f"- expected : {expected}\n- observed : {observed}\nLocation: {location}\n"


def kernel(location: str, expected: str, observed: str) -> str:
    return (
        f"SOME PROOFS FAIL\nIn {location}:\n"
        f"expected: {expected}\nobserved: {observed}\n"
    )


class PointwiseControlTests(unittest.TestCase):
    def test_mutants_edit_only_their_operation_file(self):
        for family in FAMILIES:
            for label, (path, old, new) in family.mutants.items():
                self.assertIn(path, family.operations, label)
                self.assertEqual((ROOT / path).read_text().count(old), 1, label)
                self.assertNotEqual(old, new, label)
            self.assertEqual(set(family.mutants), set(family.sites))
            with tempfile.TemporaryDirectory() as directory:
                snapshot = Path(directory) / "snapshot"
                snapshot.mkdir()
                (snapshot / "src").symlink_to(ROOT / "src")
                (snapshot / family.experiment).parent.mkdir(parents=True)
                (snapshot / family.experiment).symlink_to(ROOT / family.experiment)
                for label, (path, old, new) in family.mutants.items():
                    target = Path(directory) / label
                    family.mutant_copy(snapshot, target, label)
                    text = (target / path).read_text()
                    self.assertEqual(text, (ROOT / path).read_text().replace(old, new))
                    self.assertTrue((target / family.entry_path).exists())
                    self.assertTrue((target / family.rejecting_module(label)).exists())
                    for name in family.operations - {path}:
                        self.assertEqual(
                            (target / name).read_text(), (ROOT / name).read_text()
                        )

    def test_each_mutant_is_checked_by_its_module(self):
        checked = {label: fold.rejecting_module(label) for label in fold.mutants}
        failures = {"failed-step-done", "pair-left-failed-done"}
        failures.add("pair-right-failed-swapped")
        for label, path in checked.items():
            expected = (
                "verification/families/fold/consumers.bend"
                if label in failures
                else "src/tensor/contracts/matmul-proof.bend"
                if fold.mutants[label].file == "src/tensor/matmul.bend"
                else controls.PROOF
            )
            self.assertEqual(path, expected, label)
        self.assertTrue(all(not site.consumer for site in pointwise.sites.values()))
        for label, (path, _, _) in reduce.mutants.items():
            live = {
                "src/autograd/reduce.bend": "reduce-to-fold-PROOF.bend",
                "src/autograd/sum-to-size.bend": "sum-to-size-fold-PROOF.bend",
            }.get(path, "PROOF.bend")
            expected = f"src/autograd/{live}"
            self.assertEqual(reduce.rejecting_module(label), expected, label)
        for label in broadcast_to.mutants:
            self.assertEqual(
                broadcast_to.rejecting_module(label),
                "src/autograd/broadcast-pointwise-PROOF.bend",
                label,
            )

    def test_false_twin_changes_only_the_statement(self):
        for family in FAMILIES:
            source = (ROOT / family.experiment / "controls.bend").read_text()
            for witness, (target, wrong) in family.witnesses.items():
                twin = family.false_twin(source, witness)
                self.assertEqual(len(twin), len(source) + len(wrong) - len(target))
                start = source.index(f"def {witness}()")
                end = source.index("}:\n", start)
                self.assertEqual(twin[:start], source[:start])
                self.assertIn(f"== {wrong}() : ", twin[start:])
                self.assertEqual(
                    twin[twin.index("}:\n", start) :], source[end:], witness
                )

    def test_kernel_twin_changes_only_the_header(self):
        text = (
            "expected_sub : T =\n  a\n\n"
            "sub_witness : {(f x) == expected_sub : (Maybe -.Q2 -F32)} =\n  proof\n"
        )
        twin = pointwise.kernel_twin(
            text, "sub_witness", contracts.catalog, contracts.replace_declaration
        )
        header, body = contracts.catalog(twin)["sub_witness"]
        self.assertEqual(
            header, "sub_witness : {(f x) == swapped_sub : (Maybe -.Q2 -F32)}"
        )
        self.assertEqual(body, "  proof\n")
        self.assertEqual(
            contracts.catalog(twin)["expected_sub"], ("expected_sub : T", "  a")
        )

    def test_source_reason_requires_the_paired_wrong_value(self):
        value = "Some{F32.sub(5.0, 20.0)}"
        good = diagnostic(
            f"{{{value} == Some{{F32.sub(20.0, 5.0)}} : Maybe<&2, F32>}}",
            f"{{{value} == {value} : Maybe<&2, F32>}}",
            "sub_witness",
        )
        self.assertIn(
            "F32.sub(20.0, 5.0)", pointwise.source_reason(good, "sub_witness")
        )
        with self.assertRaises(GateError):
            pointwise.source_reason(good, "outer_witness")
        with self.assertRaises(GateError):
            pointwise.source_reason(
                diagnostic("*2", "*1", "sub_witness"), "sub_witness"
            )

    def test_source_reason_requires_the_mutation_on_one_side(self):
        mark = "U32.add(U32.from_nat(cursor), 1)"
        good = diagnostic(
            f"Q(write({mark}))", "Q(write(U32.from_nat(cursor)))", "step_lift"
        )
        self.assertIn("step_lift", pointwise.source_reason(good, "wrong-write-index"))
        both = diagnostic(f"Q(write({mark}))", f"Q(other({mark}))", "step_lift")
        with self.assertRaises(GateError):
            pointwise.source_reason(both, "wrong-write-index")
        elsewhere = diagnostic(f"Q({mark})", "Q(x)", "run_lift")
        with self.assertRaises(GateError):
            pointwise.source_reason(elsewhere, "wrong-write-index")

    def test_kernel_reason_requires_the_wrong_value_against_the_cell(self):
        claim = "{((output_point (consumers.result +a)) +U32.lit4)"
        maybe = "((Maybe -.Q2) -F32)}"
        good = kernel(
            "sub_witness",
            f"{claim} == swapped_sub : {maybe}",
            f"{claim} == (.Some, ((broadcast.cell +a), ())) : {maybe}",
        )
        self.assertIn("swapped_sub", pointwise.kernel_reason(good, "sub_witness"))
        other = kernel(
            "sub_witness",
            f"{claim} == swapped_sub : {maybe}",
            f"{{other}} == (.Some, ((broadcast.cell +a), ())) : {maybe}",
        )
        with self.assertRaises(GateError):
            pointwise.kernel_reason(other, "sub_witness")

    def test_kernel_reason_requires_the_named_declaration(self):
        site = f"{controls.SRC}pointwise_2d_proof.step_lift.qvvvvv"
        self.assertIn(
            site, pointwise.kernel_reason(kernel(site, "a", "b"), "missing-write")
        )
        with self.assertRaises(GateError):
            pointwise.kernel_reason(kernel(site, "a", "b"), "swapped-maps")
        with self.assertRaises(GateError):
            pointwise.kernel_reason(kernel(site, "a", "a"), "missing-write")

    def test_replaced_requires_exactly_the_mutated_declarations(self):
        source = controls.SRC
        positive = (
            f"{source}pointwise.step.qvvvvv : T =\n  a\n\n"
            f"{source}pointwise.step.qvvvvv_1 : T =\n  b\n\n"
            "Nat : T =\n  n\n"
        )
        instance = positive.replace("  b\n", "  c\n")
        self.assertEqual(
            pointwise.replaced(positive, instance, contracts.catalog, "swapped-maps"),
            "pointwise.step.qvvvvv_1",
        )
        with self.assertRaises(GateError):
            pointwise.replaced(positive, instance, contracts.catalog, "wrong-input-map")
        with self.assertRaises(GateError):
            pointwise.replaced(positive, positive, contracts.catalog, "swapped-maps")


if __name__ == "__main__":
    unittest.main()
