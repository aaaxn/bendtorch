import importlib

controls = importlib.import_module("verification.families.controls")
fold = importlib.import_module("verification.families.fold_controls")
Law, Site, Value, Witness = (
    controls.Law,
    controls.Site,
    controls.Value,
    controls.Witness,
)
SRC = "_2e_._2f_.._2f_.._2f_src_2f_tensor_2f_"
LAWS = f"{SRC}contracts_2f_binary_2d_LAWS"
BUILD = "contracts_2f_binary_2d_build_2d_proof"
OPERATION = "∀+x0 : _2e_._2f_.._2f_.._2f_src_2f_tensor_2f_elementwise.Operation -> "
ZERO, THREE, FIVE, SEVEN, TEN = fold.ZERO, fold.THREE, fold.FIVE, fold.SEVEN, fold.TEN


def some(source: str, kernel: str) -> Value:
    return Value(source, f"(.Some, ({kernel}, ()))")


def add(left: str, right: str) -> str:
    return f"(F32.add {left} {right})"


def mul(left: str, right: str) -> str:
    return f"(F32.mul {left} {right})"


def sub(left: str, right: str) -> str:
    return f"(F32.sub {left} {right})"


def neg(word: str) -> str:
    return f"(F32.neg {word})"


FAMILY = controls.Family(
    name="binary",
    entry=("src/tensor/elementwise.bend",),
    operations=frozenset({"src/tensor/elementwise.bend", "src/tensor/parallel.bend"}),
    laws={f"{LAWS}.binary_replayed": Law(OPERATION, "elementwise.binary")},
    live_records=("consumers.BinaryCase.arms",),
    consumers=tuple(
        f"consumers.{name}"
        for name in (
            "mul_broadcast",
            "sub_broadcast",
            "negate_scalar",
            "binary_then_reduce",
        )
    ),
    witnesses={
        "mul_witness": Witness("mul_word", "swapped_mul_word"),
        "sub_witness": Witness("sub_word", "reversed_sub_word"),
        "negate_witness": Witness("negated_word", "unsigned_word"),
        "kept_witness": Witness("kept_word", "other_word"),
        "chain_witness": Witness("chain_sum", "single_product"),
    },
    values={
        "mul_word": some("Some{F32.mul(5.0, 10.0)}", mul(FIVE, TEN)),
        "swapped_mul_word": some("Some{F32.mul(10.0, 5.0)}", mul(TEN, FIVE)),
        "sub_word": some("Some{F32.sub(7.0, 3.0)}", sub(SEVEN, THREE)),
        "reversed_sub_word": some("Some{F32.sub(3.0, 7.0)}", sub(THREE, SEVEN)),
        "negated_word": some("Some{F32.neg(3.0)}", neg(THREE)),
        "unsigned_word": some("Some{3.0}", THREE),
        "kept_word": some("Some{5.0}", FIVE),
        "other_word": some("Some{3.0}", THREE),
        "chain_sum": some(
            "Some{F32.add(F32.add(0.0, F32.mul(3.0, 10.0)), F32.mul(5.0, 10.0))}",
            add(add(ZERO, mul(THREE, TEN)), mul(FIVE, TEN)),
        ),
        "single_product": some(
            "Some{F32.add(0.0, F32.mul(5.0, 10.0))}", add(ZERO, mul(FIVE, TEN))
        ),
    },
    subjects=("result", "chained"),
    mutants=controls.load_mutants("binary"),
    sites={
        "high-fork-reuses-base": Site(
            "tree_eq",
            "count, Nat.add(Model.capacity(inner), base), (l, r))",
            f"{BUILD}.tree_eq",
            ("parallel.upper",),
        ),
        "cell-swaps-operands": Site(
            "cell_right_eq",
            "Elementwise.scalar(operation, y, x)",
            f"{BUILD}.cell_right_eq",
            ("elementwise.cell_right",),
        ),
        "cell-swaps-returned-inputs": Site(
            "cell_right_eq",
            "{other ==",
            f"{BUILD}.cell_right_eq",
            ("elementwise.cell_right",),
        ),
        "padding-not-zero": Site(
            "cell_eq",
            "[1.0]",
            f"{BUILD}.cell_eq",
            ("elementwise.cell",),
        ),
        "leaf-writes-zero": Site(
            "cell_right_eq",
            "[0.0]",
            f"{BUILD}.cell_right_eq",
            tuple(f"parallel.leaf.qv{suffix}" for suffix in ("", "_1", "_2", "_3")),
        ),
    },
    proof="src/tensor/contracts/binary-PROOF.bend",
    prefix=SRC,
)
