import Base
import ../shape/core.bend as Shape
import ../storage/arith.bend as Arith
import ./core.bend as Core
import ./allocation-spec.bend as Allocation
import ./allocation-PROOF.bend as AllocationProof
import ./observation.bend as Observation

type NewMetadata<-shape: Shape.Shape, -fill: F32> is Data:
  NewMetadata{
    successful: {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, Core.new(shape, fill)) == True{} : Bool},
    dimensions: {Observation.dimensions(shape, Core.new(shape, fill)) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
    count: {Allocation.tensor_length(shape, Core.new(shape, fill)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
    word_count: {Observation.count(shape, Core.new(shape, fill)) == Some{Shape.count(shape)} : Maybe<&2, U32>}}

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

def construction(+shape: Shape.Shape, +fill: F32) -> NewMetadata<shape, fill>:
  +successful = generic_status_consumer(shape, fill)
  NewMetadata{successful,
    completed_dimensions(shape, Core.new(shape, fill), successful),
    AllocationProof.core_new(shape, fill),
    completed_count(shape, Core.new(shape, fill), successful)}
