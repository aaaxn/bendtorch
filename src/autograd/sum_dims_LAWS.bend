import Base
import ../shape/core.bend as Shape
import ../tensor/core.bend as Core
import ../tensor/layout.bend as Layout
import ../tensor/transforms.bend as Transforms
import ../tensor/elementwise_totality.bend as Totality
import ./reduce.bend as Reduce
import ./sum_dims.bend as SumDims

# PyTorch's sum(dim, keepdim) is reduce_to onto the kept shape, then a
# reshape to the dropped dimensions. These laws state the axis bookkeeping
# and the two steps; the reduce_to laws (reduce_LAWS.bend) give the words of
# the first and the reshape laws (transform_LAWS.bend) say the second moves
# none.

# reduce_to's result as a sum's: its error as a ReduceError.
def as_sum(pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>) -> Core.Dynamic & Result<&2, &1, SumDims.Error, Core.Dynamic>:
  (input, result) = pair
  match result:
    case Fail{error}:
      (input, Fail{SumDims.ReduceError{error}})
    case Done{output}:
      (input, Done{output})

def reshape_as_sum(result: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Result<&2, &1, SumDims.Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{SumDims.TransformError{error}}
    case Done{output}:
      Done{output}

# A sum's result with its output reshaped to `dimensions`.
def reshaped(+dimensions: List<&2, U32>, pair: Core.Dynamic & Result<&2, &1, SumDims.Error, Core.Dynamic>) ->
  Core.Dynamic & Result<&2, &1, SumDims.Error, Core.Dynamic>:
  (input, result) = pair
  match result:
    case Fail{error}:
      (input, Fail{error})
    case Done{output}:
      (input, reshape_as_sum(Transforms.reshape_dynamic(output, dimensions)))

# What a plan holds: the kept shape's dimensions and the dropped ones.
def PlanKeeps(+axes: List<&2, U32>, +dimensions: List<&2, U32>, result: Result<&2, &2, SumDims.Error, SumDims.Plan>) -> Type:
  match result:
    case Fail{error}:
      Unit
    case Done{SumDims.Plan{kept, dropped}}:
      {Shape.dimensions(kept) == SumDims.kept_dimensions(axes, dimensions) : List<&2, U32>} &
        {dropped == SumDims.dropped_dimensions(axes, dimensions) : List<&2, U32>}

# An empty axis list reduces every axis, as PyTorch's dim=[]; a listed axis
# is reduced exactly when it is in the list.
law sum_dims_every_axis:
  for +axis: U32
  {SumDims.reduces([], axis) == True{} : Bool}

law sum_dims_listed_axis:
  for +head: U32
  for +tail: List<&2, U32>
  for +axis: U32
  {SumDims.reduces(head <> tail, axis) == Layout.contains(axis, head <> tail) : Bool}

# The kept dimensions count axes from 0, and set each reduced axis to 1.
law sum_dims_kept_start:
  for +axes: List<&2, U32>
  for +dimensions: List<&2, U32>
  {SumDims.kept_dimensions(axes, dimensions) == SumDims.kept_from(axes, dimensions, 0) : List<&2, U32>}

law sum_dims_kept_step:
  for +axes: List<&2, U32>
  for +extent: U32
  for +rest: List<&2, U32>
  for +axis: U32
  {SumDims.kept_from(axes, extent <> rest, axis) ==
    Bool.pick(U32, SumDims.reduces(axes, axis), 1, extent) <> SumDims.kept_from(axes, rest, U32.add(axis, 1)) : List<&2, U32>}

# The dropped dimensions count axes from 0, and leave out each reduced axis.
law sum_dims_dropped_start:
  for +axes: List<&2, U32>
  for +dimensions: List<&2, U32>
  {SumDims.dropped_dimensions(axes, dimensions) == SumDims.dropped_from(axes, dimensions, 0) : List<&2, U32>}

law sum_dims_dropped_reduced:
  for +axes: List<&2, U32>
  for +extent: U32
  for +rest: List<&2, U32>
  for +axis: U32
  for reduced: {SumDims.reduces(axes, axis) == True{} : Bool}
  {SumDims.dropped_from(axes, extent <> rest, axis) == SumDims.dropped_from(axes, rest, U32.add(axis, 1)) : List<&2, U32>}

law sum_dims_dropped_kept:
  for +axes: List<&2, U32>
  for +extent: U32
  for +rest: List<&2, U32>
  for +axis: U32
  for kept: {SumDims.reduces(axes, axis) == False{} : Bool}
  {SumDims.dropped_from(axes, extent <> rest, axis) == extent <> SumDims.dropped_from(axes, rest, U32.add(axis, 1)) : List<&2, U32>}

# An input has its rank's axes, and a rank-0 input has axis 0, as in
# PyTorch.
law sum_dims_rank_zero_limit:
  {SumDims.limit([]) == 1 : U32}

law sum_dims_rank_limit:
  for +extent: U32
  for +rest: List<&2, U32>
  {SumDims.limit(extent <> rest) == U32.from_nat(Shape.rank_dimensions(extent <> rest)) : U32}

# The axes are refused as permute refuses them, and in PyTorch's order: an
# axis at or past the limit first, then an axis listed twice.
law sum_dims_axis_out_of_bounds:
  for +axes: List<&2, U32>
  for +limit: U32
  for outside: {Layout.bounded(axes, limit) == False{} : Bool}
  {SumDims.refusal(axes, limit) == Some{Layout.AxisOutOfBounds{}} : Maybe<&2, Layout.Error>}

law sum_dims_duplicate_axis:
  for +axes: List<&2, U32>
  for +limit: U32
  for inside: {Layout.bounded(axes, limit) == True{} : Bool}
  for repeated: {Layout.unique(axes) == False{} : Bool}
  {SumDims.refusal(axes, limit) == Some{Layout.DuplicateAxis{}} : Maybe<&2, Layout.Error>}

law sum_dims_axes_accepted:
  for +axes: List<&2, U32>
  for +limit: U32
  for inside: {Layout.bounded(axes, limit) == True{} : Bool}
  for unique: {Layout.unique(axes) == True{} : Bool}
  {SumDims.refusal(axes, limit) == None{} : Maybe<&2, Layout.Error>}

# A refused list of axes fails the plan with the refusal.
law sum_dims_plan_refused:
  for +shape: Shape.Shape
  for +axes: List<&2, U32>
  for error: Layout.Error
  for refused: {SumDims.refusal(axes, SumDims.limit(Shape.dimensions(shape))) == Some{error} : Maybe<&2, Layout.Error>}
  {SumDims.plan(shape, axes) == Fail{SumDims.TransformError{Transforms.LayoutError{error}}} : Result<&2, &2, SumDims.Error, SumDims.Plan>}

# A plan holds the kept dimensions as its shape and the dropped dimensions.
law sum_dims_plan_dimensions:
  for +shape: Shape.Shape
  for +axes: List<&2, U32>
  PlanKeeps(axes, Shape.dimensions(shape), SumDims.plan(shape, axes))

# A refused plan fails the sum with its error and hands the input back.
law sum_dims_refused:
  for input: Core.Dynamic
  for +axes: List<&2, U32>
  for keepdim: Bool
  for error: SumDims.Error
  for refused: {SumDims.plan(Totality.dynamic_shape(input), axes) == Fail{error} : Result<&2, &2, SumDims.Error, SumDims.Plan>}
  {SumDims.sum_dims(input, axes, keepdim) == (input, Fail{error}) : Core.Dynamic & Result<&2, &1, SumDims.Error, Core.Dynamic>}

# With keepdim, the sum is exactly reduce_to onto the kept shape.
law sum_dims_keepdim:
  for input: Core.Dynamic
  for +axes: List<&2, U32>
  for +kept: Shape.Shape
  for +dropped: List<&2, U32>
  for planned: {SumDims.plan(Totality.dynamic_shape(input), axes) == Done{SumDims.Plan{kept, dropped}} : Result<&2, &2, SumDims.Error, SumDims.Plan>}
  {SumDims.sum_dims(input, axes, True{}) == as_sum(Reduce.reduce_to(input, kept)) : Core.Dynamic & Result<&2, &1, SumDims.Error, Core.Dynamic>}

# Without keepdim, it is the keepdim sum reshaped to the dropped dimensions.
law sum_dims_drops:
  for input: Core.Dynamic
  for +axes: List<&2, U32>
  for +kept: Shape.Shape
  for +dropped: List<&2, U32>
  for planned: {SumDims.plan(Totality.dynamic_shape(input), axes) == Done{SumDims.Plan{kept, dropped}} : Result<&2, &2, SumDims.Error, SumDims.Plan>}
  {SumDims.sum_dims(input, axes, False{}) == reshaped(dropped, SumDims.sum_dims(input, axes, True{})) :
    Core.Dynamic & Result<&2, &1, SumDims.Error, Core.Dynamic>}
