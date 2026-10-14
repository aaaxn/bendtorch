import importlib

controls = importlib.import_module("verification.families.controls")
Law, Site, Value, Witness = (
    controls.Law,
    controls.Site,
    controls.Value,
    controls.Witness,
)
SRC = controls.SRC

ZERO, ONE, TWO, THREE = (
    "F32.lit0",
    "F32.lit1065353216",
    "F32.lit1073741824",
    "F32.lit1077936128",
)
FOUR, FIVE, SIX, SEVEN = (
    "F32.lit1082130432",
    "F32.lit1084227584",
    "F32.lit1086324736",
    "F32.lit1088421888",
)
TEN, TWENTY, THIRTY, FIFTY, SIXTY = (
    "F32.lit1092616192",
    "F32.lit1101004800",
    "F32.lit1106247680",
    "F32.lit1112014848",
    "F32.lit1114636288",
)


def some(source: str, kernel: str) -> Value:
    return Value(source, f"(.Some, ({kernel}, ()))")


FAMILY = controls.Family(
    name="fold",
    entry=tuple(
        f"src/tensor/contracts/{name}.bend" for name in ("broadcast", "sum", "dot")
    ),
    operations=frozenset(
        {
            "src/tensor/contracts/fold.bend",
            "src/tensor/contracts/sum.bend",
            "src/tensor/contracts/dot.bend",
        }
    ),
    laws={
        f"{SRC}LAWS.sum_ordered": Law(
            "∀+x0 : _2e_._2f_.._2f_.._2f_src_2f_shape_2f_core.Shape -> ", "sum.sum"
        ),
        f"{SRC}LAWS.dot_ordered": Law(
            "∀+x0 : Nat -> ∀+x1 : Nat -> ∀+x2 : Nat -> ", "dot.dot"
        ),
    },
    live_records=(
        f"{SRC}fold_2d_records.FoldContract.arms",
        f"{SRC}fold_2d_records.PairFoldContract.arms",
        "consumers.SumCase.arms",
        "consumers.DotCase.arms",
        "consumers.FailCase.arms",
        "consumers.PairFailCase.arms",
    ),
    consumers=tuple(
        f"consumers.{name}"
        for name in (
            "sum_three",
            "sum_grid",
            "sum_scalar",
            "sum_empty",
            "sum_fails",
            "dot_lower_left",
            "dot_upper_right",
            "dot_empty",
            "pair_fails_left",
            "pair_fails_right",
        )
    ),
    witnesses={
        "sum_witness": Witness("expected_sum", "reversed_sum"),
        "start_witness": Witness("expected_start", "unstarted"),
        "dot_witness": Witness("expected_dot", "untransposed_dot"),
        "kept_witness": Witness("expected_kept", "swapped_kept"),
    },
    values={
        "expected_sum": some(
            "Some{F32.add(F32.add(F32.add(0.0, 1.0), 2.0), 3.0)}",
            f"(F32.add (F32.add (F32.add {ZERO} {ONE}) {TWO}) {THREE})",
        ),
        "reversed_sum": some(
            "Some{F32.add(F32.add(F32.add(0.0, 3.0), 2.0), 1.0)}",
            f"(F32.add (F32.add (F32.add {ZERO} {THREE}) {TWO}) {ONE})",
        ),
        "expected_start": some("Some{F32.add(0.0, 7.0)}", f"(F32.add {ZERO} {SEVEN})"),
        "unstarted": some("Some{7.0}", SEVEN),
        "expected_dot": some(
            "Some{F32.add(F32.add(F32.add(0.0, F32.mul(4.0, 10.0)), "
            "F32.mul(5.0, 30.0)), F32.mul(6.0, 50.0))}",
            f"(F32.add (F32.add (F32.add {ZERO} (F32.mul {FOUR} {TEN})) "
            f"(F32.mul {FIVE} {THIRTY})) (F32.mul {SIX} {FIFTY}))",
        ),
        "untransposed_dot": some(
            "Some{F32.add(F32.add(F32.add(0.0, F32.mul(4.0, 10.0)), "
            "F32.mul(5.0, 20.0)), F32.mul(6.0, 30.0))}",
            f"(F32.add (F32.add (F32.add {ZERO} (F32.mul {FOUR} {TEN})) "
            f"(F32.mul {FIVE} {TWENTY})) (F32.mul {SIX} {THIRTY}))",
        ),
        "expected_kept": some("Some{6.0}", SIX),
        "swapped_kept": some("Some{60.0}", SIXTY),
    },
    subjects=("sum_result", "dot_result"),
    mutants=controls.load_mutants("fold"),
    sites={
        "negative-zero-start": Site(
            "contract",
            "F32.neg(0.0)",
            "fold_2d_proof.contract.qvvvvv",
            ("fold.run.qvvvvv", "fold.run.qvvvvv_1"),
        ),
        "swapped-accumulator": Site(
            "step_lift",
            "step_lift~keep(plan, cursor), StorageLaws.expected(",
            "fold_2d_proof.step_lift.qvvvvv",
            ("fold.step.qvv", "fold.step.qvv_1"),
        ),
        "dropped-word": Site(
            "step_lift",
            "after}}, Fold.total(",
            "fold_2d_proof.step_lift.qvvvvv",
            ("fold.step.qvv", "fold.step.qvv_1"),
        ),
        "ignored-mask": Site(
            "step_lift",
            "operation, True{}, input =>",
            "fold_2d_proof.step_lift.qvvvvv",
            ("fold.loop.qvvvvv", "fold.loop.qvvvvv_1"),
        ),
        "cursor-off-by-one": Site(
            "contract",
            "plan, 1n, Core.Packed{",
            "fold_2d_proof.contract.qvvvvv",
            ("fold.run.qvvvvv", "fold.run.qvvvvv_1"),
        ),
        "pair-swapped-operands": Site(
            "pair_step_lift",
            "right_ops, cursor), "
            "StorageLaws.expected(List.append(&2, StorageLaws.Op, right_ops",
            "fold_2d_proof.pair_step_lift.qvvvvv",
            ("fold.pair_right.qvv", "fold.pair_right.qvv_1"),
        ),
        "pair-wrong-map": Site(
            "pair_step_lift",
            "left_fits, U32.from_nat(pair_step_lift~ib(plan, cursor))",
            "fold_2d_proof.pair_step_lift.qvvvvv",
            ("fold.pair_loop.qvvvvv", "fold.pair_loop.qvvvvv_1"),
        ),
        "pair-negative-zero-start": Site(
            "pair_contract",
            "F32.neg(0.0)",
            "fold_2d_proof.pair_contract.qvvvvv",
            ("fold.pair_run.qvvvvv", "fold.pair_run.qvvvvv_1"),
        ),
        "pair-cursor-off-by-one": Site(
            "pair_contract",
            "plan, 1n, Core.Packed{left,",
            "fold_2d_proof.pair_contract.qvvvvv",
            ("fold.pair_run.qvvvvv", "fold.pair_run.qvvvvv_1"),
        ),
        "failed-step-done": Site(
            "sum_fails",
            "OutOfBounds",
            "consumers.sum_fails",
            ("fold.step.qvv", "fold.step.qvv_1"),
            consumer=True,
        ),
        "pair-left-failed-done": Site(
            "pair_fails_left",
            "OutOfBounds",
            "consumers.pair_fails_left",
            ("fold.pair_left.qvv", "fold.pair_left.qvv_1"),
            consumer=True,
        ),
        "pair-right-failed-swapped": Site(
            "pair_fails_right",
            "Some{10.0}",
            "consumers.pair_fails_right",
            ("fold.pair_right.qvv", "fold.pair_right.qvv_1"),
            consumer=True,
        ),
        "sum-swapped-add": Site(
            "total_words",
            "{F32.add(StorageLaws.expected(",
            "sum_2d_proof.total_words",
            ("sum.add",),
        ),
        "dot-untransposed-right": Site(
            "total_factors",
            "Nat.mul(column, inner)",
            "dot_2d_proof.total_factors",
            ("dot.right_index",),
        ),
    },
)
