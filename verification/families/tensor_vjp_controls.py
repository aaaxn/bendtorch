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
ELEMENTWISE_BINARY = "_2e_._2f_.._2f_.._2f_src_2f_tensor_2f_elementwise.binary"
LAWS = f"{SRC}tensor_2d_vjp_2d_LAWS"
NAT = "∀+x0 : Nat -> "
OPERATION = "∀+x0 : _2e_._2f_.._2f_.._2f_src_2f_tensor_2f_elementwise.Operation -> "
ZERO, TWO, THREE, FIVE, SEVEN, TEN = (
    fold.ZERO,
    fold.TWO,
    fold.THREE,
    fold.FIVE,
    fold.SEVEN,
    fold.TEN,
)


def some(source: str, kernel: str) -> Value:
    return Value(source, f"(.Some, ({kernel}, ()))")


def add(left: str, right: str) -> str:
    return f"(F32.add {left} {right})"


def mul(left: str, right: str) -> str:
    return f"(F32.mul {left} {right})"


def neg(word: str) -> str:
    return f"(F32.neg {word})"


FAMILY = controls.Family(
    name="tensor_vjp",
    entry=("src/autograd/tensor-backward.bend",),
    operations=frozenset({"src/autograd/tensor-backward.bend"}),
    laws={
        f"{LAWS}.tensor_product_replayed": Law(OPERATION, ELEMENTWISE_BINARY),
        f"{LAWS}.tensor_add_vjp_values": Law(NAT, "tensor_2d_backward.vjp"),
        f"{LAWS}.tensor_sub_vjp_values": Law(NAT, "tensor_2d_backward.vjp"),
        f"{LAWS}.tensor_mul_vjp_values": Law(NAT, "tensor_2d_backward.vjp"),
    },
    live_records=(f"{LAWS}.PairContract.arms", "consumers.PairCase.arms"),
    consumers=tuple(
        f"consumers.{name}" for name in ("add_values", "sub_values", "mul_values")
    ),
    witnesses={
        "add_kept_witness": Witness("kept_word", "folded_word"),
        "add_sum_witness": Witness("added_sum", "reversed_sum"),
        "sub_witness": Witness("negated_sum", "unsigned_sum"),
        "mul_witness": Witness("right_product", "left_product"),
        "mul_sum_witness": Witness("left_products", "right_products"),
    },
    values={
        "kept_word": some("Some{5.0}", FIVE),
        "folded_word": some("Some{F32.add(0.0, 5.0)}", add(ZERO, FIVE)),
        "added_sum": some(
            "Some{F32.add(F32.add(0.0, 3.0), 5.0)}", add(add(ZERO, THREE), FIVE)
        ),
        "reversed_sum": some(
            "Some{F32.add(F32.add(0.0, 5.0), 3.0)}", add(add(ZERO, FIVE), THREE)
        ),
        "negated_sum": some(
            "Some{F32.add(F32.add(0.0, F32.neg(3.0)), F32.neg(5.0))}",
            add(add(ZERO, neg(THREE)), neg(FIVE)),
        ),
        "unsigned_sum": some(
            "Some{F32.add(F32.add(0.0, 3.0), 5.0)}", add(add(ZERO, THREE), FIVE)
        ),
        "right_product": some("Some{F32.mul(5.0, 10.0)}", mul(FIVE, TEN)),
        "left_product": some("Some{F32.mul(5.0, 7.0)}", mul(FIVE, SEVEN)),
        "left_products": some(
            "Some{F32.add(F32.add(0.0, F32.mul(3.0, 2.0)), F32.mul(5.0, 7.0))}",
            add(add(ZERO, mul(THREE, TWO)), mul(FIVE, SEVEN)),
        ),
        "right_products": some(
            "Some{F32.add(F32.add(0.0, F32.mul(3.0, 10.0)), F32.mul(5.0, 10.0))}",
            add(add(ZERO, mul(THREE, TEN)), mul(FIVE, TEN)),
        ),
    },
    subjects=("added", "subtracted", "multiplied"),
    mutants=controls.load_mutants("tensor_vjp"),
    sites={
        "add-reduces-to-other-operand": Site(
            "Laws.tensor_add_vjp_values",
            "Backward.delivered_as(a, right,",
            "tensor_2d_vjp_2d_LAWS.tensor_add_vjp_values",
            ("tensor_2d_backward.vjp_add",),
        ),
        "add-sends-second-to-first": Site(
            "Laws.tensor_add_vjp_values",
            "Backward.added_first(a, right,",
            "tensor_2d_vjp_2d_LAWS.tensor_add_vjp_values",
            ("tensor_2d_backward.vjp_add",),
        ),
        "sub-keeps-sign": Site(
            "Laws.tensor_sub_vjp_values",
            "parallel_planned(Elementwise.KeepLeft{},",
            "tensor_2d_vjp_2d_LAWS.tensor_sub_vjp_values",
            ("tensor_2d_backward.negated",),
        ),
        "mul-swaps-operands": Site(
            "Laws.tensor_mul_vjp_values",
            "multiplied_first(a, b, left, right, Core.Packed{right,",
            "tensor_2d_vjp_2d_LAWS.tensor_mul_vjp_values",
            ("tensor_2d_backward.vjp_mul",),
        ),
        "mul-product-swaps-inputs": Site(
            "Laws.tensor_mul_vjp_values",
            "parallel_planned(Elementwise.Mul{}, Core.Packed{left,",
            "tensor_2d_vjp_2d_LAWS.tensor_mul_vjp_values",
            ("tensor_2d_backward.multiplied_first",),
        ),
    },
    proof="src/autograd/PROOF.bend",
    prefix=SRC,
)
