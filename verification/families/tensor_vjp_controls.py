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
TWENTY, THIRTY, FIFTY = fold.TWENTY, fold.THIRTY, fold.FIFTY


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
        f"{LAWS}.tensor_matmul_vjp_values": Law(NAT, "tensor_2d_backward.vjp"),
        f"{LAWS}.tensor_matmul_nt_vjp_values": Law(NAT, "tensor_2d_backward.vjp"),
        f"{LAWS}.tensor_relu_vjp_values": Law(NAT, "tensor_2d_backward.vjp"),
        f"{LAWS}.tensor_mse_vjp_values": Law(NAT, "tensor_2d_backward.vjp"),
        f"{LAWS}.tensor_cross_entropy_vjp_values": Law(NAT, "tensor_2d_backward.vjp"),
    },
    live_records=(
        f"{LAWS}.PairContract.arms",
        f"{LAWS}.MatmulPairContract.arms",
        f"{LAWS}.OneContract.arms",
        "consumers.PairCase.arms",
        "consumers.MatmulCase.arms",
    ),
    consumers=tuple(
        f"consumers.{name}"
        for name in (
            "add_values",
            "sub_values",
            "mul_values",
            "matmul_values",
            "linear_values",
        )
    ),
    witnesses={
        "add_kept_witness": Witness("kept_word", "folded_word"),
        "add_sum_witness": Witness("added_sum", "reversed_sum"),
        "sub_witness": Witness("negated_sum", "unsigned_sum"),
        "mul_witness": Witness("right_product", "left_product"),
        "mul_sum_witness": Witness("left_products", "right_products"),
        "matmul_front_witness": Witness("front_product", "untransposed_product"),
        "matmul_back_witness": Witness("back_product", "swapped_product"),
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
        "front_product": some(
            "Some{F32.add(F32.add(0.0, F32.mul(3.0, 30.0)), F32.mul(5.0, 50.0))}",
            add(add(ZERO, mul(THREE, THIRTY)), mul(FIVE, FIFTY)),
        ),
        "untransposed_product": some(
            "Some{F32.add(F32.add(0.0, F32.mul(3.0, 20.0)), F32.mul(5.0, 50.0))}",
            add(add(ZERO, mul(THREE, TWENTY)), mul(FIVE, FIFTY)),
        ),
        "back_product": some(
            "Some{F32.add(0.0, F32.mul(7.0, 3.0))}", add(ZERO, mul(SEVEN, THREE))
        ),
        "swapped_product": some(
            "Some{F32.add(0.0, F32.mul(3.0, 7.0))}", add(ZERO, mul(THREE, SEVEN))
        ),
    },
    subjects=("added", "subtracted", "multiplied", "matmul_pass"),
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
        "matmul-left-untransposed": Site(
            "matmul_values",
            "right_fits}, 0, 0, Shape.bool_case",
            "tensor_2d_vjp_2d_PROOF.matmul_values",
            ("tensor_2d_backward.matmul_first",),
        ),
        "matmul-right-operands-swapped": Site(
            "matmul_values",
            "Matrix.parallel_planned(shape, Layout.output(left,",
            "tensor_2d_vjp_2d_PROOF.matmul_values",
            ("tensor_2d_backward.right_product",),
        ),
        "matmul-sends-second-to-first": Site(
            "matmul_values",
            "Backward.matmul_first(a, a, left, right,",
            "tensor_2d_vjp_2d_PROOF.matmul_values",
            ("tensor_2d_backward.vjp_matmul",),
        ),
        "nt-vjp-tn-swaps-operands": Site(
            "nt_values",
            "Matrix.product_planned(Matrix.Swapped{}, Matrix.Swapped{}, left, shape,",
            "tensor_2d_vjp_2d_PROOF.nt_values",
            ("tensor_2d_backward.nt_right",),
        ),
        "nt-vjp-left-reads-transposed": Site(
            "nt_values",
            "Backward.multiplied(a, left, "
            "Matrix.product_planned(Matrix.Kept{}, Matrix.Kept{},",
            "tensor_2d_vjp_2d_PROOF.nt_values",
            ("tensor_2d_backward.nt_left",),
        ),
        "nt-vjp-sends-second-to-first": Site(
            "nt_values",
            "Backward.nt_first(a, a, left, right,",
            "tensor_2d_vjp_2d_PROOF.nt_values",
            ("tensor_2d_backward.vjp_matmul_nt",),
        ),
        "relu-vjp-reads-gradient-as-input": Site(
            "Laws.tensor_relu_vjp_values",
            "parallel_planned(Elementwise.ReluBackward{}, Core.Packed{shape,",
            "tensor_2d_vjp_2d_LAWS.tensor_relu_vjp_values",
            ("tensor_2d_backward.vjp_relu",),
        ),
        "relu-vjp-sends-relu": Site(
            "Laws.tensor_relu_vjp_values",
            "parallel_planned(Elementwise.ReluLeft{},",
            "tensor_2d_vjp_2d_LAWS.tensor_relu_vjp_values",
            ("tensor_2d_backward.vjp_relu",),
        ),
        "relu-vjp-sends-to-first-node": Site(
            "Laws.tensor_relu_vjp_values",
            "Backward.delivered_as(0n, input,",
            "tensor_2d_vjp_2d_LAWS.tensor_relu_vjp_values",
            ("tensor_2d_backward.relu_passed",),
        ),
        "mse-vjp-sends-second-to-first-node": Site(
            "Laws.tensor_mse_vjp_values",
            "Backward.delivered_as(0n, right,",
            "tensor_2d_vjp_2d_LAWS.tensor_mse_vjp_values",
            ("tensor_2d_backward.mse_second",),
        ),
        "mse-vjp-scale-two": Site(
            "Laws.tensor_mse_vjp_values",
            "parallel_planned(Elementwise.MseGrad{2.0},",
            "tensor_2d_vjp_2d_LAWS.tensor_mse_vjp_values",
            ("tensor_2d_backward.mse_shaped",),
        ),
        "cross-entropy-vjp-drops-softmax": Site(
            "Laws.tensor_cross_entropy_vjp_values",
            "parallel_planned(Elementwise.SoftmaxGrad{",
            "tensor_2d_vjp_2d_LAWS.tensor_cross_entropy_vjp_values",
            ("tensor_2d_backward.entropy_first",),
        ),
    },
    proof="src/autograd/PROOF.bend",
    prefix=SRC,
)
