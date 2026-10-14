import Base
import ../storage/main.bend as Storage
import ./core.bend as Core
import ../shape/core.bend as Shape
import ./lifecycle.bend as Lifecycle

def storage_length(result: Result<&2, &1, Storage.Error, Storage.Storage>) -> Maybe<&2, Nat>:
  match result:
    case Fail{error}:
      None{}
    case Done{storage}:
      Some{Lifecycle.logical_length(storage)}

def tensor_length(-shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, Nat>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Some{Core.logical_length(shape, tensor)}

type TensorDone<-shape: Shape.Shape, -result: Result<&2, &1, Core.Error, Core.Tensor<shape>>> is Data:
  TensorDone{-tensor: Core.Tensor<shape>, -same: {Done{tensor} == result : Result<&2, &1, Core.Error, Core.Tensor<shape>>}}
