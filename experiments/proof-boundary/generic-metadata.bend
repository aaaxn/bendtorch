import Base
import ../../src/shape/core.bend as Shape
import ../../src/storage/arith.bend as Arith
import ../../src/tensor/core.bend as Core
import ../../src/tensor/allocation-spec.bend as Allocation
import ../../src/tensor/allocation-PROOF.bend as AllocationProof
import ./status.bend as Status
import ./observations.bend as Observation

type Construction<-shape: Shape.Shape, -fill: F32> is Data:
  Construction{
    successful: {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, Core.new(shape, fill)) == True{} : Bool},
    dimensions: {Observation.dimensions(shape, Core.new(shape, fill)) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
    count: {Allocation.tensor_length(shape, Core.new(shape, fill)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
    word_count: {Observation.count(shape, Core.new(shape, fill)) == Some{Shape.count(shape)} : Maybe<&2, U32>}}

def completed_dimensions(+shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>,
  successful: {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, result) == True{} : Bool}) ->
  {Observation.dimensions(shape, result) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>}:
  match result:
    case Fail{error}:
      Empty.absurd({None{} == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>}, Arith.clash(successful))
    case Done{tensor}:
      {==}

def completed_count(+shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>,
  successful: {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, result) == True{} : Bool}) ->
  {Observation.count(shape, result) == Some{Shape.count(shape)} : Maybe<&2, U32>}:
  match result:
    case Fail{error}:
      Empty.absurd({None{} == Some{Shape.count(shape)} : Maybe<&2, U32>}, Arith.clash(successful))
    case Done{tensor}:
      {==}

def construction(+shape: Shape.Shape, +fill: F32) -> Construction<shape, fill>:
  +successful = Status.generic_status_consumer(shape, fill)
  Construction{successful,
    completed_dimensions(shape, Core.new(shape, fill), successful),
    AllocationProof.core_new(shape, fill),
    completed_count(shape, Core.new(shape, fill), successful)}
