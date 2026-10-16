import Base
import ../shape/core.bend as Shape
import ../tensor/core.bend as Core
import ../tensor/elementwise.bend as Elementwise
import ../tensor/layout.bend as Layout

# Materialized broadcast: an owned tensor of the requested shape whose every
# position holds the input value at its right-aligned broadcast projection.
# The elementwise parallel build writes it, KeepLeft with the input on the left
# and a +0.0 tensor of the target on the right, so an output above 2^15 cells
# is built in parallel halves.

type Error is Data:
  TensorError{error: Core.Error}
  NotBroadcastable{}

def finish(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  ((input, zeros), result) = pair
  match result:
    case Fail{error}:
      (input, Fail{TensorError{error}})
    case Done{output}:
      (input, Done{output})

def zeroed(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  result: Result<&2, &1, Core.Error, Core.Tensor<target>>) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      (input, Fail{TensorError{error}})
    case Done{zeros}:
      finish(Elementwise.parallel_allocated(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros},
        Core.new(Shape.output_shape(plan), 0.0)))

def fitted(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan, same: Bool) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match same:
    case False{}:
      (input, Fail{NotBroadcastable{}})
    case True{}:
      zeroed(input, target, plan, Core.new(target, 0.0))

def planned(input: Core.Dynamic, +target: Shape.Shape, result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      (input, Fail{TensorError{Core.ShapeError{error}}})
    case Done{+plan}:
      fitted(input, target, plan, Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(target)))

# The input owner comes back unchanged next to the result. An incompatible
# shape pair fails with the planner's broadcast error; a compatible pair whose
# broadcast is larger than the target (for example [3] to [1]) fails with
# NotBroadcastable.
def broadcast_to(input: Core.Dynamic, +target: Shape.Shape) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  Core.Packed{+shape, tensor} = input
  planned(Core.Packed{shape, tensor}, target, Shape.broadcast_checked(shape, target))

def broadcast_to_dimensions_result(input: Core.Dynamic, -dimensions: List<&2, U32>, result: Shape.ConstructResult<dimensions>) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Shape.SError{error}:
      (input, Fail{TensorError{Core.ShapeError{error}}})
    case Shape.SBuilt{+target, preserved}:
      broadcast_to(input, target)

def broadcast_to_dimensions(input: Core.Dynamic, +dimensions: List<&2, U32>) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  broadcast_to_dimensions_result(input, dimensions, Shape.construct(dimensions))

def error_code(error: Error) -> U32:
  match error:
    case TensorError{error}:
      Core.error_code(error)
    case NotBroadcastable{}:
      100
