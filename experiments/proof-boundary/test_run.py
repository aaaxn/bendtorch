import importlib
import tempfile
import unittest
from pathlib import Path

runner = importlib.import_module("experiments.proof-boundary.run")
ROOT = Path(__file__).resolve().parents[2]


class ExperimentRunnerTests(unittest.TestCase):
    def test_statement_mutation_preserves_body_and_other_declarations(self):
        text = (
            "expected_value : Nat =\n  (.Zero, ())\n\n"
            "witness : {x == expected_value : Nat} =\n  {==}\n"
        )
        header, body = runner.catalog(text)["witness"]
        changed = runner.replace_declaration(
            text, "witness", header.replace("== expected_value", "== wrong_value"), body
        )
        after = runner.catalog(changed)
        self.assertEqual(after["witness"], ("witness : {x == wrong_value : Nat}", body))
        self.assertEqual(
            after["expected_value"], runner.catalog(text)["expected_value"]
        )

    def test_duplicate_declaration_is_refused(self):
        with self.assertRaises(runner.GateError):
            runner.catalog("a : T =\n  x\n\na : T =\n  y\n")

    def test_fill_mutation_keeps_lambda_quantities(self):
        body = "  λ+x0 => λx1 => (new_result +x0 (new (count x0) x1))"
        self.assertEqual(
            runner.fill_body(body, "ZERO"),
            "  λ+x0 => λx1 => (new_result +x0 (new (count x0) ZERO))",
        )
        with self.assertRaises(runner.GateError):
            runner.fill_body("{==}", "ZERO")

    def test_quantity_failure_is_not_value_rejection(self):
        with self.assertRaises(runner.GateError):
            runner.source_value_reason(
                "- expected : *2\n- observed : *1\nLocation: scalar_witness\n",
                "scalar_witness",
            )
        reason = runner.source_value_reason(
            "- expected : {Some{7.0} == Some{8.0} : Maybe<&2, F32>}\n"
            "- observed : {Some{7.0} == Some{7.0} : Maybe<&2, F32>}\n"
            "Location: scalar_witness\n",
            "scalar_witness",
        )
        self.assertIn("Some{8.0}", reason)

    def test_kernel_termination_failure_is_not_value_rejection(self):
        with self.assertRaises(runner.GateError):
            runner.kernel_value_reason(
                "SOME PROOFS FAIL\nIn core.new:\n"
                "affine live code, calls that descend\n",
                "wrong-fill",
            )

    def test_actual_mutation_preserves_count_statement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            (baseline / "src/tensor").mkdir(parents=True)
            for name in ("core.bend", "allocation-PROOF.bend"):
                (baseline / "src/tensor" / name).write_bytes(
                    (ROOT / "src/tensor" / name).read_bytes()
                )
            target = root / "mutant"
            runner.mutation_copy(baseline, target)
            core = (target / "src/tensor/core.bend").read_text()
            self.assertIn(
                "new_result(shape, Storage.new(Shape.count(shape), 0.0))", core
            )
            before = (baseline / "src/tensor/allocation-PROOF.bend").read_text()
            after = (target / "src/tensor/allocation-PROOF.bend").read_text()
            start = before.index("def core_new(+shape:")
            header_end = before.index(":\n", start)
            self.assertEqual(before[start:header_end], after[start:header_end])
            self.assertIn("storage_new(Shape.count(shape), 0.0,", after)


if __name__ == "__main__":
    unittest.main()
