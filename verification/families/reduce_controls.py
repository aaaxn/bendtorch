import importlib

controls = importlib.import_module("verification.families.controls")
fold = importlib.import_module("verification.families.fold_controls")
Law, Site, Value, Witness = (
    controls.Law,
    controls.Site,
    controls.Value,
    controls.Witness,
)
SRC = "_2e_._2f_.._2f_.._2f_src_2f_autograd_2f_"
REDUCE_TO = "src/autograd/reduce-to-fold-PROOF.bend"
SUM_TO_SIZE = "src/autograd/sum-to-size-fold-PROOF.bend"
ZERO, ONE, TWO, THREE = fold.ZERO, fold.ONE, fold.TWO, fold.THREE
FOUR, FIVE, SIX = fold.FOUR, fold.FIVE, fold.SIX


def some(source: str, kernel: str) -> Value:
    return Value(source, f"(.Some, ({kernel}, ()))")


FAMILY = controls.Family(
    name="reduce",
    entry=(
        "src/tensor/contracts/broadcast.bend",
        "src/tensor/contracts/sum.bend",
        "src/autograd/reduce-fold.bend",
        "src/autograd/reduce.bend",
        "src/autograd/sum-to-size.bend",
    ),
    operations=frozenset(
        {
            "src/autograd/reduce-fold.bend",
            "src/autograd/reduce.bend",
            "src/autograd/sum-to-size.bend",
        }
    ),
    laws={
        f"{SRC}reduce_2d_fold_2d_LAWS.reduce_fold_fiber": Law(
            "∀+x0 : _2e_._2f_.._2f_.._2f_src_2f_shape_2f_core.BroadcastPlan -> ",
            "reduce_2d_fold.cell",
        ),
        f"{SRC}reduce_2d_fold_2d_LAWS.reduce_to_fiber": Law(
            "∀+x0 : _2e_._2f_.._2f_.._2f_src_2f_shape_2f_core.Shape -> ",
            "reduce.reduce_to",
        ),
        f"{SRC}sum_2d_to_2d_size_2d_fold_2d_LAWS.sum_to_size_clone": Law(
            "∀+x0 : _2e_._2f_.._2f_.._2f_src_2f_shape_2f_core.Shape -> ",
            "sum_2d_to_2d_size.sum_to_size",
        ),
        f"{SRC}sum_2d_to_2d_size_2d_fold_2d_LAWS.sum_to_size_fiber": Law(
            "∀+x0 : _2e_._2f_.._2f_.._2f_src_2f_shape_2f_core.Shape -> ",
            "sum_2d_to_2d_size.sum_to_size",
        ),
    },
    live_records=(
        f"{SRC}reduce_2d_fold_2d_LAWS.FiberContract.arms",
        f"{SRC}reduce_2d_fold_2d_LAWS.ReduceToContract.arms",
        "consumers.ReduceCase.arms",
        "consumers.ReduceToCase.arms",
    ),
    consumers=tuple(
        f"consumers.{name}"
        for name in (
            "reduce_columns",
            "reduce_rows",
            "reduce_scalar",
            "reduce_same",
            "reduce_empty",
            "reduce_to_columns",
            "reduce_to_rows",
            "sum_to_size_same",
            "sum_to_size_columns",
        )
    ),
    witnesses={
        "column_witness": Witness("expected_column", "unmasked_column"),
        "row_witness": Witness("expected_row", "first_row"),
        "same_witness": Witness("expected_same", "uncopied_same"),
        "word_witness": Witness("expected_word", "shifted_word"),
        "reduce_to_witness": Witness("expected_column", "unmasked_column"),
        "sum_to_size_witness": Witness("cloned_word", "folded_word"),
        "sum_to_size_fiber_witness": Witness("expected_column", "unmasked_column"),
    },
    values={
        "expected_column": some(
            "Some{F32.add(F32.add(0.0, 2.0), 5.0)}",
            f"(F32.add (F32.add {ZERO} {TWO}) {FIVE})",
        ),
        "unmasked_column": some(
            "Some{F32.add(F32.add(F32.add(F32.add(F32.add(F32.add(0.0, 1.0), "
            "2.0), 3.0), 4.0), 5.0), 6.0)}",
            f"(F32.add (F32.add (F32.add (F32.add (F32.add (F32.add {ZERO} {ONE}) "
            f"{TWO}) {THREE}) {FOUR}) {FIVE}) {SIX})",
        ),
        "expected_row": some(
            "Some{F32.add(F32.add(F32.add(0.0, 4.0), 5.0), 6.0)}",
            f"(F32.add (F32.add (F32.add {ZERO} {FOUR}) {FIVE}) {SIX})",
        ),
        "first_row": some(
            "Some{F32.add(F32.add(F32.add(0.0, 1.0), 2.0), 3.0)}",
            f"(F32.add (F32.add (F32.add {ZERO} {ONE}) {TWO}) {THREE})",
        ),
        "expected_same": some("Some{F32.add(0.0, 5.0)}", f"(F32.add {ZERO} {FIVE})"),
        "uncopied_same": some("Some{5.0}", FIVE),
        "expected_word": some("Some{5.0}", FIVE),
        "shifted_word": some("Some{6.0}", SIX),
        "cloned_word": some("Some{5.0}", FIVE),
        "folded_word": some("Some{F32.add(0.0, 5.0)}", f"(F32.add {ZERO} {FIVE})"),
    },
    subjects=("result", "reduced", "summed"),
    mutants=controls.load_mutants("reduce"),
    sites={
        "lands-on-source": Site(
            "fiber_total",
            "Nat.cmp(Broadcast.right_walk(",
            "reduce_2d_fold_2d_PROOF.fiber_total",
            ("reduce_2d_fold.lands",),
        ),
        "inverted-mask": Site(
            "fiber_total",
            "Bool.not(",
            "reduce_2d_fold_2d_PROOF.fiber_total",
            ("reduce_2d_fold.lands",),
        ),
        "reads-target-offset": Site(
            "fiber_total",
            "U32.from_nat(Broadcast.left_walk(",
            "reduce_2d_fold_2d_PROOF.fiber_total",
            ("reduce_2d_fold.source",),
        ),
        "target-count": Site(
            "reduce_fold_fiber",
            "ReduceFold.lands, Shape.elements(Shape.left_shape(plan))",
            "reduce_2d_fold_2d_PROOF.reduce_fold_fiber",
            ("reduce_2d_fold.cell",),
        ),
        "reduce-to-lands-on-source": Site(
            "step_lift",
            "step_input(target, U32.from_nat(../tensor/contracts/broadcast.right_walk(",
            "reduce_2d_to_2d_fold_2d_PROOF.step_lift",
            ("reduce.step_indices",),
            module=REDUCE_TO,
        ),
        "reduce-to-reversed-order": Site(
            "step_lift",
            "Nat.sub(Nat.sub(Shape.elements(Shape.output_shape(plan)), 1n)",
            "reduce_2d_to_2d_fold_2d_PROOF.step_lift",
            ("reduce.step",),
            module=REDUCE_TO,
        ),
        "reduce-to-swapped-add": Site(
            "step_lift",
            "F32.add(StorageLaws.expected(List.append(",
            "reduce_2d_to_2d_fold_2d_PROOF.step_lift",
            ("reduce.step_output",),
            module=REDUCE_TO,
        ),
        "sum-to-size-identity-folds": Site(
            "sum_to_size_clone",
            "reduce.planned(",
            "sum_2d_to_2d_size_2d_fold_2d_PROOF.sum_to_size_clone",
            ("sum_2d_to_2d_size.decided",),
            module=SUM_TO_SIZE,
        ),
        "sum-to-size-reduced-clones": Site(
            "sum_to_size_fiber",
            "SumToSize.cloned(",
            "sum_2d_to_2d_size_2d_fold_2d_PROOF.sum_to_size_fiber",
            ("sum_2d_to_2d_size.decided",),
            module=SUM_TO_SIZE,
        ),
    },
    proof="src/autograd/PROOF.bend",
    prefix=SRC,
)
