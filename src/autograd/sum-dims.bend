import Base
import ../shape/core.bend as Shape
import ../tensor/core.bend as Core
import ../tensor/layout.bend as Layout
import ../tensor/transforms.bend as Transforms
import ./reduce.bend as Reduce

# PyTorch's sum(input, dim, keepdim): the sum over the listed axes. It is
# reduce_to onto the kept shape, the input's dimensions with every reduced
# axis set to 1, so each output cell is the ordered fold of its fiber from
# +0.0 on the parallel build. With keepdim False the result is then reshaped
# to the dropped dimensions, the input's without the reduced axes, which
# moves no word. An empty axis list reduces every axis, as PyTorch's dim=[]
# does; a whole-tensor sum with no dim stays Reduction.sum.
#
# The axes are checked as PyTorch checks them: any axis at or past the rank
# (past 0 for a rank-0 input, whose only axis is 0) is AxisOutOfBounds, then
# an axis listed twice is DuplicateAxis, each with the code permute gives it.
# Negative axes are not taken: an axis is a U32.

type Error is Data:
  ReduceError{error: Reduce.Error}
  TransformError{error: Transforms.Error}

# The kept shape, built with Shape.construct, and the dropped dimensions.
type Plan is Data:
  Plan{kept: Shape.Shape, dropped: List<&2, U32>}

# Whether the sum reduces axis `axis`: every axis for an empty list.
def reduces(+axes: List<&2, U32>, +axis: U32) -> Bool:
  match axes:
    case Nil{}:
      True{}
    case +head <> +tail:
      Layout.contains(axis, head <> tail)

# The dimensions from axis `axis` on, each reduced one set to 1.
def kept_from(+axes: List<&2, U32>, dimensions: List<&2, U32>, +axis: U32) -> List<&2, U32>:
  match dimensions:
    case Nil{}:
      Nil{}
    case extent <> rest:
      Bool.pick(U32, reduces(axes, axis), 1, extent) <> kept_from(axes, rest, U32.add(axis, 1))

def kept_dimensions(+axes: List<&2, U32>, dimensions: List<&2, U32>) -> List<&2, U32>:
  kept_from(axes, dimensions, 0)

def dropped_next(reduced: Bool, extent: U32, rest: List<&2, U32>) -> List<&2, U32>:
  match reduced:
    case True{}:
      rest
    case False{}:
      extent <> rest

# The dimensions from axis `axis` on, without the reduced ones.
def dropped_from(+axes: List<&2, U32>, dimensions: List<&2, U32>, +axis: U32) -> List<&2, U32>:
  match dimensions:
    case Nil{}:
      Nil{}
    case extent <> rest:
      dropped_next(reduces(axes, axis), extent, dropped_from(axes, rest, U32.add(axis, 1)))

def dropped_dimensions(+axes: List<&2, U32>, dimensions: List<&2, U32>) -> List<&2, U32>:
  dropped_from(axes, dimensions, 0)

# How many axes an input of these dimensions has for PyTorch's check: its
# rank, and 1 for a rank-0 input.
def limit(dimensions: List<&2, U32>) -> U32:
  match dimensions:
    case Nil{}:
      1
    case +extent <> +rest:
      U32.from_nat(Shape.rank_dimensions(extent <> rest))

# PyTorch's refusal of the axes, permute's checks in permute's order: an axis
# at or past the limit, then an axis listed twice.
def refused(bounded: Bool, unique: Bool) -> Maybe<&2, Layout.Error>:
  match bounded unique:
    case False{} unique:
      Some{Layout.AxisOutOfBounds{}}
    case True{} False{}:
      Some{Layout.DuplicateAxis{}}
    case True{} True{}:
      None{}

def refusal(+axes: List<&2, U32>, +limit: U32) -> Maybe<&2, Layout.Error>:
  refused(Layout.bounded(axes, limit), Layout.unique(axes))

def plan_built(-kept: List<&2, U32>, +dropped: List<&2, U32>, built: Shape.ConstructResult<kept>) -> Result<&2, &2, Error, Plan>:
  match built:
    case Shape.SError{error}:
      Fail{ReduceError{Reduce.TensorError{Core.ShapeError{error}}}}
    case Shape.SBuilt{+shape, preserved}:
      Done{Plan{shape, dropped}}

def plan_checked(+axes: List<&2, U32>, +dimensions: List<&2, U32>, checked: Maybe<&2, Layout.Error>) -> Result<&2, &2, Error, Plan>:
  match checked:
    case Some{error}:
      Fail{TransformError{Transforms.LayoutError{error}}}
    case None{}:
      plan_built(kept_dimensions(axes, dimensions), dropped_dimensions(axes, dimensions),
        Shape.construct(kept_dimensions(axes, dimensions)))

# The sum's plan for an input of this shape, or the refusal of its axes.
def plan(+shape: Shape.Shape, +axes: List<&2, U32>) -> Result<&2, &2, Error, Plan>:
  plan_checked(axes, Shape.dimensions(shape), refusal(axes, limit(Shape.dimensions(shape))))

def transformed(result: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{TransformError{error}}
    case Done{output}:
      Done{output}

def shaped(+dropped: List<&2, U32>, keepdim: Bool, output: Core.Dynamic) -> Result<&2, &1, Error, Core.Dynamic>:
  match keepdim:
    case True{}:
      Done{output}
    case False{}:
      transformed(Transforms.reshape_dynamic(output, dropped))

def squeezed(+dropped: List<&2, U32>, keepdim: Bool, result: Result<&2, &1, Reduce.Error, Core.Dynamic>) ->
  Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{ReduceError{error}}
    case Done{output}:
      shaped(dropped, keepdim, output)

def summed(+dropped: List<&2, U32>, keepdim: Bool, pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  (input, result) = pair
  (input, squeezed(dropped, keepdim, result))

# The sum of a planned input: reduce_to onto the kept shape, then, without
# keepdim, the reshape to the dropped dimensions. The input owner comes back.
def sum_planned(input: Core.Dynamic, plan: Plan, keepdim: Bool) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  Plan{+kept, +dropped} = plan
  summed(dropped, keepdim, Reduce.reduce_to(input, kept))

def sum_checked(input: Core.Dynamic, keepdim: Bool, planned: Result<&2, &2, Error, Plan>) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match planned:
    case Fail{error}:
      (input, Fail{error})
    case Done{plan}:
      sum_planned(input, plan, keepdim)

# PyTorch's input.sum(dim=axes, keepdim=keepdim). The input owner always comes
# back unchanged next to the result.
def sum_dims(input: Core.Dynamic, +axes: List<&2, U32>, keepdim: Bool) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  Core.Packed{+shape, tensor} = input
  sum_checked(Core.Packed{shape, tensor}, keepdim, plan(shape, axes))

def error_code(error: Error) -> U32:
  match error:
    case ReduceError{error}:
      Reduce.error_code(error)
    case TransformError{error}:
      Transforms.error_code(error)
