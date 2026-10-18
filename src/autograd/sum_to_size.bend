import Base
import ../shape/core.bend as Shape
import ../tensor/core.bend as Core
import ../tensor/layout.bend as Layout
import ./reduce.bend as Reduce

# The PyTorch-facing reducer for a broadcast contribution: the gradient
# PyTorch's autograd engine gives an operand of a broadcast operation. When the
# target has the contribution's dimensions nothing is reduced and the output is
# a clone of the contribution, so every word, including -0.0 and NaN payloads,
# is kept. Otherwise the result is exactly Reduce.reduce_to's ordered fold.

def cloned(+shape: Shape.Shape, pair: Core.Tensor<shape> & Core.Tensor<shape>) ->
  Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>:
  (kept, copy) = pair
  (Core.Packed{shape, kept}, Done{Core.Packed{shape, copy}})

def decided(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape, same: Bool) ->
  Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>:
  match same:
    case False{}:
      Reduce.reduce_to(Core.Packed{shape, tensor}, target)
    case True{}:
      cloned(shape, Core.clone(shape, tensor))

# The contribution owner comes back next to the result, as with reduce_to.
def sum_to_size(input: Core.Dynamic, +target: Shape.Shape) -> Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>:
  Core.Packed{+shape, tensor} = input
  decided(shape, tensor, target, Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)))

def sum_to_size_dimensions_result(input: Core.Dynamic, -dimensions: List<&2, U32>, result: Shape.ConstructResult<dimensions>) ->
  Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>:
  match result:
    case Shape.SError{error}:
      (input, Fail{Reduce.TensorError{Core.ShapeError{error}}})
    case Shape.SBuilt{+target, preserved}:
      sum_to_size(input, target)

def sum_to_size_dimensions(input: Core.Dynamic, +dimensions: List<&2, U32>) -> Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>:
  sum_to_size_dimensions_result(input, dimensions, Shape.construct(dimensions))
