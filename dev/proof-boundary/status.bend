import Base
import ../../src/shape/core.bend as Shape
import ../../src/tensor/core.bend as Core
import ../../src/tensor/allocation_spec.bend as Allocation
import ../../src/tensor/allocation_PROOF.bend as AllocationProof

def status_is_length(-shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) ->
  {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, result) ==
    Maybe.is_some(&2, Nat, Allocation.tensor_length(shape, result)) : Bool}:
  match result:
    case Fail{error}:
      {==}
    case Done{tensor}:
      {==}

def generic_count_producer(+shape: Shape.Shape, +fill: F32) ->
  {Allocation.tensor_length(shape, Core.new(shape, fill)) ==
    Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}:
  AllocationProof.core_new(shape, fill)

def generic_status_consumer(+shape: Shape.Shape, +fill: F32) ->
  {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, Core.new(shape, fill)) == True{} : Bool}:
  Equal.trans(Bool, Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, Core.new(shape, fill)),
    Maybe.is_some(&2, Nat, Allocation.tensor_length(shape, Core.new(shape, fill))), True{},
    status_is_length(shape, Core.new(shape, fill)),
    Equal.cong(Maybe<&2, Nat>, Bool, value => Maybe.is_some(&2, Nat, value),
      Allocation.tensor_length(shape, Core.new(shape, fill)), Some{Shape.logical_numel(Shape.dimensions(shape))},
      generic_count_producer(shape, fill)))
