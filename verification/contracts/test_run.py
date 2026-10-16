import importlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

runner = importlib.import_module("verification.contracts.run")
reads = importlib.import_module("verification.contracts.reads_controls")
clones = importlib.import_module("verification.contracts.clone_controls")
transforms = importlib.import_module("verification.contracts.transform_controls")
values = importlib.import_module("verification.contracts.from_values_controls")
permutes = importlib.import_module("verification.contracts.permute_values_controls")
mutants = importlib.import_module("verification.contracts.mutants")
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

    def test_failure_mutation_replaces_only_the_success_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            (baseline / "src/tensor").mkdir(parents=True)
            source = (ROOT / "src/tensor/core.bend").read_text()
            (baseline / "src/tensor/core.bend").write_text(source)
            runner.failure_copy(baseline, root / "mutant")
            after = (root / "mutant/src/tensor/core.bend").read_text()
            self.assertEqual(
                after.replace(
                    "Fail{StorageLengthMismatch{}}\n\ndef new(",
                    "attach(shape, storage)\n\ndef new(",
                ),
                source,
            )

    def test_failure_source_reason_requires_the_success_claim(self):
        with self.assertRaises(runner.GateError):
            runner.source_value_reason(
                "- expected : {7n == 8n : Nat}\n- observed : {7n == 7n : Nat}\n"
                "Location: new_result_length\n",
                "unjustified-failure",
            )

    def test_failure_kernel_reason_requires_the_length_claim(self):
        prefix = "_2e_._2f_.._2f_.._2f_src_2f_"
        claim = f"== (.Some, (({prefix}shape_2f_core.logical_numel x0), ())) : N}}"
        location = f"In {prefix}tensor_2f_allocation_2d_PROOF.new_result_length:"
        good = (
            f"SOME PROOFS FAIL\n{location}\n"
            f"expected: {{(len (({prefix}tensor_2f_core.new_result +x0) "
            f"(.Done, (x1, ())))) {claim}\n"
            f"observed: {{(len (({prefix}tensor_2f_core.attach +x0) x1)) {claim}\n"
        )
        self.assertIn(
            "new_result_length",
            runner.kernel_value_reason(good, "unjustified-failure"),
        )
        with self.assertRaises(runner.GateError):
            runner.kernel_value_reason(
                good.replace("logical_numel", "count"), "unjustified-failure"
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

    def test_reads_mutants_edit_only_the_named_core_definition(self):
        source = (ROOT / "src/tensor/core.bend").read_text()
        for label, mutant in reads.MUTANTS.items():
            self.assertEqual(source.count(mutant.before), 1, label)
            self.assertNotEqual(mutant.before, mutant.after, label)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "baseline/src/tensor").mkdir(parents=True)
            (root / "baseline/src/tensor/core.bend").write_text(source)
            for label, (path, before, after) in reads.MUTANTS.items():
                reads.mutant_copy(root / "baseline", root / label, label)
                mutated = (root / label / path).read_text()
                self.assertEqual(mutated.replace(after, before), source, label)

    def test_reads_false_twin_changes_only_the_expected_value(self):
        controls = (ROOT / reads.EXPERIMENT / "controls.bend").read_text()
        for witness in reads.WITNESSES:
            expected = reads.WITNESSES[witness].target
            twin = reads.false_twin(controls, witness)
            self.assertEqual(
                twin.count("wrong_value()"), controls.count("wrong_value()") + 1
            )
            self.assertEqual(
                twin.replace("wrong_value()", f"{expected}()", 1).count(
                    f"{expected}()"
                ),
                controls.count(f"{expected}()"),
            )

    def test_reads_source_reason_refuses_other_failures(self):
        good = (
            "- expected : {Some{7.0} == Some{8.0} : Maybe<&2, F32>}\n"
            "- observed : {Some{7.0} == Some{7.0} : Maybe<&2, F32>}\n"
            "Location: read_one_witness\n"
        )
        self.assertIn("Some{8.0}", reads.source_reason(good, "read_one_witness"))
        with self.assertRaises(runner.GateError):
            reads.source_reason(good, "written_one_witness")
        with self.assertRaises(runner.GateError):
            reads.source_reason(
                "- expected : a filled definition\n- observed : corrupt_neighbour\n"
                "Location: write_flat\n",
                "corrupted-neighbour",
            )
        with self.assertRaises(runner.GateError):
            reads.source_reason(
                "- expected : *2\n- observed : *1\nLocation: write_value_done\n",
                "wrong-index",
            )

    def test_reads_kernel_reason_refuses_termination_failure(self):
        with self.assertRaises(runner.GateError):
            reads.kernel_reason(
                f"SOME PROOFS FAIL\nIn {reads.CORE}corrupt_neighbour:\n"
                "affine live code, calls that descend\n",
                "corrupted-neighbour",
            )

    def test_transplant_keeps_definition_before_use(self):
        flat = f"{reads.CORE}write_flat"
        positive = "\n\n".join(
            [
                "Nat : T =\n  n",
                f"{flat} : W =\n  old",
                f"use : U =\n  {flat}",
                "F32.lit0 : F =\n  z",
            ]
        )
        mutant = "\n\n".join(
            [
                "Nat : T =\n  n",
                "F32.lit0 : F =\n  z",
                "U32.add : A =\n  add",
                f"{flat} : W =\n  (U32.add F32.lit0)",
            ]
        )
        result = runner.catalog(reads.transplant(positive, mutant, runner.catalog))
        order = list(result)
        self.assertLess(order.index("U32.add"), order.index(flat))
        self.assertLess(order.index("F32.lit0"), order.index(flat))
        self.assertLess(order.index(flat), order.index("use"))
        self.assertEqual(result[flat][1], "  (U32.add F32.lit0)")

    def test_clone_mutants_edit_only_core(self):
        core = (ROOT / "src/tensor/core.bend").read_text()
        for label, mutant in clones.MUTANTS.items():
            self.assertEqual(core.count(mutant.before), 1, label)
            self.assertNotEqual(mutant.before, mutant.after, label)
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"
            (snapshot / "src/tensor").mkdir(parents=True)
            (snapshot / "src/tensor/core.bend").write_text(core)
            target = Path(directory) / "mutant"
            clones.mutant_copy(snapshot, target, "length-off-by-one")
            changed = (target / "src/tensor/core.bend").read_text()
            self.assertIn(", 1n+n)", changed)
            self.assertEqual(changed.count("def "), core.count("def "))
            self.assertTrue((target / clones.EXPERIMENT / "core-entry.bend").exists())

    def test_clone_false_twin_changes_only_the_statement(self):
        controls = (ROOT / clones.EXPERIMENT / "controls.bend").read_text()
        for witness in clones.WITNESSES:
            target = clones.WITNESSES[witness].target
            replacement = clones.WITNESSES[witness].replacement
            twin = clones.false_twin(controls, witness)
            start = twin.index(f"def {witness}()")
            header = twin[start : twin.index("}:\n", start)]
            self.assertIn(f"== {replacement}()", header)
            self.assertNotIn(f"== {target}()", header)
            end = twin.index("}:\n", start)
            restored = twin[:start] + header.replace(replacement, target) + twin[end:]
            self.assertEqual(restored, controls)

    def test_clone_source_reason_requires_the_named_site(self):
        good = (
            "- expected : 1n+number\n- observed : number\nLocation: number_projection\n"
        )
        self.assertIn("1n+number", clones.source_reason(good, "length-off-by-one"))
        with self.assertRaises(runner.GateError):
            clones.source_reason(good, "length-owner-first-zeroed")
        with self.assertRaises(runner.GateError):
            clones.source_reason(
                "- expected : *2\n- observed : *1\nLocation: number_projection\n",
                "length-off-by-one",
            )
        alias = (
            "- expected : {Some{7.0} == Some{9.0} : Maybe<&2, F32>}\n"
            "- observed : {Some{7.0} == Some{7.0} : Maybe<&2, F32>}\n"
            "Location: alias_one_witness\n"
        )
        self.assertIn("Some{9.0}", clones.source_reason(alias, "alias_one_witness"))
        with self.assertRaises(runner.GateError):
            clones.source_reason(alias, "second_last_witness")

    def test_clone_kernel_reason_refuses_other_locations(self):
        site = clones.KERNEL_SITES["length-off-by-one"]
        good = f"In {site}:\nexpected: a\nobserved: b\n"
        self.assertIn(site, clones.kernel_reason(good, "length-off-by-one"))
        with self.assertRaises(runner.GateError):
            clones.kernel_reason(good, "count-off-by-one")
        with self.assertRaises(runner.GateError):
            clones.kernel_reason(
                f"In {site}:\naffine live code, calls that descend\n",
                "length-off-by-one",
            )

    def test_transform_mutants_edit_only_their_module(self):
        for label, (path, old, new) in transforms.MUTANTS.items():
            text = (ROOT / path).read_text()
            self.assertEqual(text.count(old), 1, label)
            self.assertNotEqual(old, new, label)
            self.assertTrue(transforms.export_prefix(label).endswith("."), label)
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"
            shutil.copytree(ROOT / "src", snapshot / "src")
            target = Path(directory) / "mutant"
            transforms.mutant_copy(snapshot, target, "permute-wrong-error")
            changed = (target / "src/tensor/layout.bend").read_text()
            self.assertIn("      Fail{DuplicateAxis{}}\n    case Shape.BTrue", changed)
            self.assertEqual(
                (target / "src/tensor/transforms.bend").read_text(),
                (ROOT / "src/tensor/transforms.bend").read_text(),
            )

    def test_mutant_entry_needs_file_before_and_after(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "mutants.toml"
            catalog.write_text("[x]\nfile = 'a.bend'\nbefore = 'old'\n")
            with self.assertRaises(TypeError):
                mutants.load(catalog)

    def test_transform_source_reason_requires_the_named_site(self):
        good = (
            "- expected : Fail{Layout.DuplicateAxis{}}\n"
            "- observed : Fail{Layout.AxisOutOfBounds{}}\n"
            "Location: bounded_failure\n"
        )
        self.assertIn(
            "DuplicateAxis", transforms.source_reason(good, "permute-wrong-error")
        )
        with self.assertRaises(runner.GateError):
            transforms.source_reason(good, "transpose-failure-changes-source")
        with self.assertRaises(runner.GateError):
            transforms.source_reason(
                "- expected : *2\n- observed : *1\nLocation: bounded_failure\n",
                "permute-wrong-error",
            )

    def test_transform_kernel_reason_refuses_other_locations(self):
        site = f"{transforms.SRC}permute_2d_contract.bounded_failure"
        good = f"In {site}:\nexpected: a\nobserved: b\n"
        self.assertIn(site, transforms.kernel_reason(good, "permute-wrong-error"))
        with self.assertRaises(runner.GateError):
            transforms.kernel_reason(good, "reshape-zeroed")

    def test_from_values_mutants_edit_only_core(self):
        core = (ROOT / "src/tensor/core.bend").read_text()
        for label, (path, old, new) in values.MUTANTS.items():
            self.assertEqual(path, "src/tensor/core.bend", label)
            self.assertEqual(core.count(old), 1, label)
            self.assertNotEqual(old, new, label)
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"
            shutil.copytree(ROOT / "src", snapshot / "src")
            for label, (path, old, new) in values.MUTANTS.items():
                target = Path(directory) / label
                values.mutant_copy(snapshot, target, label)
                changed = (target / path).read_text()
                self.assertEqual(changed.replace(new, old, 1), core, label)
                self.assertTrue(
                    (target / values.EXPERIMENT / "core-entry.bend").exists()
                )

    def test_from_values_false_twin_changes_only_the_statement(self):
        controls = (ROOT / values.EXPERIMENT / "controls.bend").read_text()
        for witness in values.WITNESSES:
            target = values.WITNESSES[witness].target
            twin = values.false_twin(controls, witness)
            start = twin.index(f"def {witness}()")
            end = twin.index("}:\n", start)
            header = twin[start:end]
            self.assertIn("== wrong_value()", header)
            restored = twin[:start] + header.replace("wrong_value", target) + twin[end:]
            self.assertEqual(restored, controls)

    def test_from_values_source_reason_requires_the_named_site(self):
        good = (
            "- expected : {Some{1.0} == Some{8.0} : Maybe<&2, F32>}\n"
            "- observed : {Some{1.0} == Some{1.0} : Maybe<&2, F32>}\n"
            "Location: three_first_witness\n"
        )
        self.assertIn("Some{8.0}", values.source_reason(good, "three_first_witness"))
        with self.assertRaises(runner.GateError):
            values.source_reason(good, "grid_last_witness")
        site = (
            "- expected : P(x, valid}, 1n))\n- observed : P(x, valid}, 0n))\n"
            "Location: counted_lift\n"
        )
        self.assertIn(
            "counted_lift", values.source_reason(site, "from-values-cursor-off-by-one")
        )
        with self.assertRaises(runner.GateError):
            values.source_reason(site, "from-values-wrong-fill")
        with self.assertRaises(runner.GateError):
            values.source_reason(
                "- expected : *2\n- observed : *1\nLocation: counted_lift\n",
                "from-values-cursor-off-by-one",
            )

    def test_from_values_kernel_reason_refuses_other_locations(self):
        site = f"{values.SRC}from_2d_values_2d_contract.fill_lift"
        good = f"In {site}:\nexpected: a\nobserved: b\n"
        self.assertIn(
            site, values.kernel_reason(good, "from-values-missing-last-write")
        )
        with self.assertRaises(runner.GateError):
            values.kernel_reason(good, "from-values-wrong-fill")
        point = f"(({values.SRC}observation.point x) ({values.CORE}from_values y))"
        witness = (
            "In grid_last_witness:\n"
            f"expected: {{{point} == wrong_value : M}}\n"
            f"observed: {{{point} == (List.get z) : M}}\n"
        )
        self.assertIn(
            "grid_last_witness", values.kernel_reason(witness, "grid_last_witness")
        )
        with self.assertRaises(runner.GateError):
            values.kernel_reason(
                witness.replace("wrong_value", "other"), "grid_last_witness"
            )

    def test_permute_mutants_edit_only_transforms(self):
        source = (ROOT / "src/tensor/transforms.bend").read_text()
        for label, (path, old, new) in permutes.MUTANTS.items():
            self.assertEqual(path, "src/tensor/transforms.bend", label)
            self.assertEqual(source.count(old), 1, label)
            self.assertNotEqual(old, new, label)
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"
            shutil.copytree(ROOT / "src", snapshot / "src")
            for label, (path, old, new) in permutes.MUTANTS.items():
                target = Path(directory) / label
                permutes.mutant_copy(snapshot, target, label)
                changed = (target / path).read_text()
                self.assertEqual(changed.replace(new, old, 1), source, label)
                self.assertTrue(
                    (target / permutes.EXPERIMENT / "module-entry.bend").exists()
                )

    def test_permute_false_twin_changes_only_the_statement(self):
        controls = (ROOT / permutes.EXPERIMENT / "controls.bend").read_text()
        for witness in permutes.WITNESSES:
            target = permutes.WITNESSES[witness].target
            twin = permutes.false_twin(controls, witness)
            start = twin.index(f"def {witness}()")
            end = twin.index("}:\n", start)
            header = twin[start:end]
            self.assertIn("==\n    wrong_value()", header)
            restored = twin[:start] + header.replace("wrong_value", target) + twin[end:]
            self.assertEqual(restored, controls)

    def test_permute_source_reason_requires_the_named_site(self):
        good = (
            "- expected : {Some{5.0} == Some{8.0} : Maybe<&2, F32>}\n"
            "- observed : {Some{5.0} == Some{5.0} : Maybe<&2, F32>}\n"
            "Location: cube_rotate_witness\n"
        )
        self.assertIn("Some{8.0}", permutes.source_reason(good, "cube_rotate_witness"))
        with self.assertRaises(runner.GateError):
            permutes.source_reason(good, "grid_transpose_witness")
        mark = "Layout.transpose_axes(Shape.dimensions(input), a, a, 0)"
        site = (
            "- expected : P(Layout.transpose_axes(Shape.dimensions(input), a, b, 0))\n"
            f"- observed : P({mark})\n"
            "Location: transpose_route\n"
        )
        self.assertIn(
            "transpose_route", permutes.source_reason(site, "transpose-swap-ignored")
        )
        with self.assertRaises(runner.GateError):
            permutes.source_reason(site, "permute-forward-map")
        with self.assertRaises(runner.GateError):
            permutes.source_reason(
                "- expected : *2\n- observed : *1\nLocation: transpose_route\n",
                "transpose-swap-ignored",
            )

    def test_permute_kernel_reason_refuses_other_locations(self):
        site = f"{permutes.SRC}permute_2d_values.read_step"
        good = f"In {site}:\nexpected: a\nobserved: b\n"
        self.assertIn(site, permutes.kernel_reason(good, "permute-forward-map"))
        with self.assertRaises(runner.GateError):
            permutes.kernel_reason(good, "transpose-swap-ignored")
        value = f"({permutes.SRC}transform_2d_observation.applied_value x)"
        left = f"({value} ({permutes.CORE}from_values y))"
        witness = (
            "In cube_rotate_witness:\n"
            f"expected: {{{left} == wrong_value : M}}\n"
            f"observed: {{{left} == ({permutes.SRC}observation.listed z) : M}}\n"
        )
        self.assertIn(
            "cube_rotate_witness",
            permutes.kernel_reason(witness, "cube_rotate_witness"),
        )
        with self.assertRaises(runner.GateError):
            permutes.kernel_reason(witness, "grid_kept_witness")


class RunnerTests(unittest.TestCase):
    def test_workers_below_one_are_rejected_before_any_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "contracts"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "verification.contracts.run",
                    "--output",
                    str(output),
                    "--workers",
                    "0",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("--workers needs at least one worker", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
