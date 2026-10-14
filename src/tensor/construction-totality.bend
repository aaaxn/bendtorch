import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./allocation-spec.bend as Allocation
import ./transform-totality.bend as TransformTotality

def TensorReads(+shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Type:
  match result:
    case Fail{error}:
      Unit
    case Done{tensor}:
      @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool} ->
        TransformTotality.ReadSuccess<shape, tensor, index>

type TensorDone<-shape: Shape.Shape, -result: Result<&2, &1, Core.Error, Core.Tensor<shape>>> is Data:
  TensorDone{
    -witness: Allocation.TensorDone<shape, result>,
    -successful: {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, result) == True{} : Bool},
    -readable: TensorReads(shape, result)}

type ConstructionWitness<-dimensions: List<&2, U32>, -result: Result<&2, &1, Core.Error, Core.Dynamic>> is Data:
  ConstructionWitness{
    -shape: Shape.Shape,
    -tensor: Core.Tensor<shape>,
    -same: {Done{Core.Packed{shape, tensor}} == result : Result<&2, &1, Core.Error, Core.Dynamic>},
    -preserved: {Shape.dimensions(shape) == dimensions : List<&2, U32>}}

def ConstructionReads(+dimensions: List<&2, U32>, result: Result<&2, &1, Core.Error, Core.Dynamic>) -> Type:
  match result:
    case Fail{error}:
      Unit
    case Done{Core.Packed{+shape, tensor}}:
      @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(dimensions)) == True{} : Bool} ->
        TransformTotality.ReadSuccess<shape, tensor, index>

type ConstructionDone<-dimensions: List<&2, U32>, -result: Result<&2, &1, Core.Error, Core.Dynamic>> is Data:
  ConstructionDone{
    -witness: ConstructionWitness<dimensions, result>,
    -successful: {Result.is_done(&2, &1, Core.Error, Core.Dynamic, result) == True{} : Bool},
    -readable: ConstructionReads(dimensions, result)}
