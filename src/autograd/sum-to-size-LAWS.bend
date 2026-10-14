import Base
import ../shape/core.bend as Shape
import ../tensor/core.bend as Core
import ../tensor/elementwise-totality.bend as Totality
import ./reduce.bend as Reduce
import ./reduce-LAWS.bend as ReduceLaws
import ./sum-to-size.bend as SumToSize

def Identity(-input: Core.Dynamic, +target: Shape.Shape, pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>) -> Type:
  (retained, result) = pair
  match result:
    case Fail{error}:
      Empty
    case Done{output}:
      {Totality.dimensions(output) == Shape.dimensions(target) : List<&2, U32>} &
        (@+index: U32 ->
          {Core.checked_observe_dynamic(output, index) == Core.checked_observe_dynamic(input, index) : Maybe<&2, F32>} &
            {Core.observe_dynamic(output, index) == Core.observe_dynamic(input, index) : Maybe<&2, F32>})

type IdentityModel<-input: Core.Dynamic, -target: Shape.Shape,
  -pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>> is Data:
  IdentityModel{-identity: Identity(input, target, pair)}

# The contribution owner comes back with the same dimensions and observations.
law sum_to_size_retains_input:
  for input: Core.Dynamic
  for +target: Shape.Shape
  ReduceLaws.Retained<input, SumToSize.sum_to_size(input, target)>

# A target with the contribution's dimensions always succeeds, and the output
# has the target's dimensions and the contribution's exact words at every
# position: -0.0 stays -0.0, as in PyTorch's autograd.
law sum_to_size_identity:
  for input: Core.Dynamic
  for +target: Shape.Shape
  for same: {Totality.dimensions(input) == Shape.dimensions(target) : List<&2, U32>}
  IdentityModel<input, target, SumToSize.sum_to_size(input, target)>

# A target with other dimensions gives exactly the ordered reduce_to result,
# so every reduce_to law applies to it.
law sum_to_size_reduces:
  for input: Core.Dynamic
  for +target: Shape.Shape
  for differ: {Totality.dimensions(input) != Shape.dimensions(target) : List<&2, U32>}
  {SumToSize.sum_to_size(input, target) == Reduce.reduce_to(input, target) : Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>}
