import importlib

controls = importlib.import_module("verification.families.controls")
Law, Site, Value, Witness = (
    controls.Law,
    controls.Site,
    controls.Value,
    controls.Witness,
)
SRC = controls.SRC

FIVE = "F32.lit1084227584"
TWENTY = "F32.lit1101004800"

FAMILY = controls.Family(
    name="pointwise",
    entry=("src/tensor/contracts/broadcast.bend",),
    operations=frozenset(
        {"src/tensor/contracts/pointwise.bend", "src/tensor/contracts/broadcast.bend"}
    ),
    laws={
        f"{SRC}LAWS.broadcast_pointwise": Law(
            "∀+x0 : _2e_._2f_.._2f_.._2f_src_2f_tensor_2f_elementwise.Operation -> ",
            "broadcast.binary",
        )
    },
    live_records=(
        f"{SRC}pointwise_records.PointwiseContract.arms",
        "consumers.PointCase.arms",
        "consumers.EmptyCase.arms",
    ),
    consumers=tuple(
        f"consumers.{name}"
        for name in (
            "sub_grid_first",
            "sub_grid_middle",
            "sub_grid_last",
            "mul_outer_row",
            "mul_outer_column",
            "add_same",
            "keep_scalar",
            "empty",
            "from_values_sub",
            "new_sub",
        )
    ),
    witnesses={
        "sub_witness": Witness("expected_sub", "swapped_sub"),
        "outer_witness": Witness("expected_outer", "unbroadcast_outer"),
        "keep_witness": Witness("expected_keep", "wrong_keep"),
        "kept_witness": Witness("expected_kept", "overwritten_kept"),
        "produced_witness": Witness("expected_produced", "swapped_produced"),
    },
    values={
        "expected_sub": Value(
            "Some{F32.sub(5.0, 20.0)}", f"(.Some, ((F32.sub {FIVE} {TWENTY}), ()))"
        ),
        "swapped_sub": Value(
            "Some{F32.sub(20.0, 5.0)}", f"(.Some, ((F32.sub {TWENTY} {FIVE}), ()))"
        ),
        "expected_outer": Value(
            "Some{F32.mul(3.0, 30.0)}",
            "(.Some, ((F32.mul F32.lit1077936128 F32.lit1106247680), ()))",
        ),
        "unbroadcast_outer": Value(
            "Some{F32.mul(2.0, 30.0)}",
            "(.Some, ((F32.mul F32.lit1073741824 F32.lit1106247680), ()))",
        ),
        "expected_keep": Value("Some{7.0}", "(.Some, (F32.lit1088421888, ()))"),
        "wrong_keep": Value("Some{8.0}", "(.Some, (F32.lit1090519040, ()))"),
        "expected_kept": Value("Some{5.0}", f"(.Some, ({FIVE}, ()))"),
        "overwritten_kept": Value(
            "Some{F32.sub(5.0, 20.0)}", f"(.Some, ((F32.sub {FIVE} {TWENTY}), ()))"
        ),
        "expected_produced": Value(
            "Some{F32.sub(3.0, 30.0)}",
            "(.Some, ((F32.sub F32.lit1077936128 F32.lit1106247680), ()))",
        ),
        "swapped_produced": Value(
            "Some{F32.sub(30.0, 3.0)}",
            "(.Some, ((F32.sub F32.lit1106247680 F32.lit1077936128), ()))",
        ),
    },
    subjects=("result", "produced"),
    mutants=controls.load_mutants("pointwise"),
    sites={
        "swapped-operands": Site(
            "step_lift",
            "~f(operation, "
            "StorageLaws.expected(List.append(&2, StorageLaws.Op, right_ops",
            "pointwise_proof.step_lift.qvvvvv",
            ("pointwise.right_step.qvv", "pointwise.right_step.qvv_1"),
        ),
        "wrong-write-index": Site(
            "step_lift",
            "U32.add(U32.from_nat(cursor), 1)",
            "pointwise_proof.step_lift.qvvvvv",
            ("pointwise.right_step.qvv", "pointwise.right_step.qvv_1"),
        ),
        "missing-write": Site(
            "step_lift",
            "Core.write_result(",
            "pointwise_proof.step_lift.qvvvvv",
            ("pointwise.right_step.qvv", "pointwise.right_step.qvv_1"),
        ),
        "wrong-input-map": Site(
            "step_lift",
            "left_fits, U32.from_nat(step_lift~ib(plan, cursor))",
            "pointwise_proof.step_lift.qvvvvv",
            ("pointwise.step.qvvvvv", "pointwise.step.qvvvvv_1"),
        ),
        "cursor-off-by-one": Site(
            "run_lift",
            "shape, 1n, Pointwise.State{",
            "pointwise_proof.run_lift.qvvvvv",
            ("pointwise.allocated.qvvvvv", "pointwise.allocated.qvvvvv_1"),
        ),
        "swapped-maps": Site(
            "broadcast_pointwise",
            "Broadcast.right_map, Broadcast.left_map",
            "pointwise_proof.step_lift.qvvvvv_1",
            ("pointwise.step.qvvvvv_1",),
        ),
        "wrong-output-shape": Site(
            "broadcast_pointwise",
            "Core.new_result(Shape.left_shape(plan)",
            "broadcast_proof.broadcast_pointwise",
            ("broadcast.binary",),
        ),
    },
)
