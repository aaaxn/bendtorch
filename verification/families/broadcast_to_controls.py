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
LAWS = f"{SRC}broadcast_pointwise_LAWS"
SHAPE = "∀+x0 : _2e_._2f_.._2f_.._2f_src_2f_shape_2f_core.Shape -> "
ZERO, ONE, TWO, THREE, SEVEN = fold.ZERO, fold.ONE, fold.TWO, fold.THREE, fold.SEVEN


def some(source: str, kernel: str) -> Value:
    return Value(source, f"(.Some, ({kernel}, ()))")


FAMILY = controls.Family(
    name="broadcast_to",
    entry=("src/autograd/broadcast.bend",),
    operations=frozenset({"src/autograd/broadcast.bend"}),
    laws={
        f"{LAWS}.broadcast_to_pointwise": Law(SHAPE, "broadcast.broadcast_to"),
        f"{LAWS}.broadcast_to_replayed": Law(SHAPE, "broadcast.broadcast_to"),
    },
    live_records=(
        f"{LAWS}.BroadcastToContract.arms",
        "consumers.BroadcastCase.arms",
        "consumers.EmptyCase.arms",
    ),
    consumers=tuple(
        f"consumers.{name}"
        for name in (
            "rows_repeat",
            "columns_repeat",
            "scalar_fills",
            "negative_zero_kept",
            "empty",
            "broadcast_then_reduce",
        )
    ),
    witnesses={
        "columns_witness": Witness("expected_columns", "first_row"),
        "rows_witness": Witness("expected_rows", "row_read"),
        "keep_witness": Witness("expected_keep", "added_keep"),
        "kept_witness": Witness("expected_kept", "zero_kept"),
        "chain_witness": Witness("expected_chain", "single_chain"),
    },
    values={
        "expected_columns": some("Some{2.0}", TWO),
        "first_row": some("Some{1.0}", ONE),
        "expected_rows": some("Some{3.0}", THREE),
        "row_read": some("Some{2.0}", TWO),
        "expected_keep": some("Some{7.0}", SEVEN),
        "added_keep": some("Some{F32.add(7.0, 0.0)}", f"(F32.add {SEVEN} {ZERO})"),
        "expected_kept": some("Some{3.0}", THREE),
        "zero_kept": some("Some{0.0}", ZERO),
        "expected_chain": some(
            "Some{F32.add(F32.add(0.0, 7.0), 7.0)}",
            f"(F32.add (F32.add {ZERO} {SEVEN}) {SEVEN})",
        ),
        "single_chain": some("Some{F32.add(0.0, 7.0)}", f"(F32.add {ZERO} {SEVEN})"),
    },
    subjects=("result", "chained"),
    mutants=controls.load_mutants("broadcast_to"),
    sites={
        "adds-zero": Site(
            "run_lift",
            "Elementwise.Add{}",
            "broadcast_pointwise_PROOF.run_lift",
            ("broadcast.zeroed",),
        ),
        "swapped-operands": Site(
            "run_lift",
            "KeepLeft{}, plan, Core.Packed{target,",
            "broadcast_pointwise_PROOF.run_lift",
            ("broadcast.zeroed",),
        ),
        "returns-zero-owner": Site(
            "run_lift",
            "Core.Packed{target, Core.TTensor{StorageLaws.replay(zeros_kept,",
            "broadcast_pointwise_PROOF.run_lift",
            ("broadcast.finish",),
        ),
    },
    proof="src/autograd/broadcast_pointwise_PROOF.bend",
    prefix=SRC,
)
