import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./construction_totality.bend as Totality

def clone_first_checked(-shape: Shape.Shape, pair: Core.Tensor<shape> & Core.Tensor<shape>, index: U32) -> Maybe<&2, F32>:
  (first, second) = pair
  Core.checked_observe(shape, first, index)

def clone_second_checked(-shape: Shape.Shape, pair: Core.Tensor<shape> & Core.Tensor<shape>, index: U32) -> Maybe<&2, F32>:
  (first, second) = pair
  Core.checked_observe(shape, second, index)

def clone_first_model(-shape: Shape.Shape, pair: Core.Tensor<shape> & Core.Tensor<shape>, index: U32) -> Maybe<&2, F32>:
  (first, second) = pair
  Core.observe(shape, first, index)

def clone_second_model(-shape: Shape.Shape, pair: Core.Tensor<shape> & Core.Tensor<shape>, index: U32) -> Maybe<&2, F32>:
  (first, second) = pair
  Core.observe(shape, second, index)

type CloneMeaning<-shape: Shape.Shape, -original: Core.Tensor<shape>> is Data:
  CloneMeaning{
    -first_checked: @+index: U32 -> {clone_first_checked(shape, Core.clone(shape, original), index) == Core.checked_observe(shape, original, index) : Maybe<&2, F32>},
    -second_checked: @+index: U32 -> {clone_second_checked(shape, Core.clone(shape, original), index) == Core.checked_observe(shape, original, index) : Maybe<&2, F32>},
    -first_model: @+index: U32 -> {clone_first_model(shape, Core.clone(shape, original), index) == Core.observe(shape, original, index) : Maybe<&2, F32>},
    -second_model: @+index: U32 -> {clone_second_model(shape, Core.clone(shape, original), index) == Core.observe(shape, original, index) : Maybe<&2, F32>}}

law public_clone_elements:
  for +shape: Shape.Shape
  for original: Core.Tensor<shape>
  CloneMeaning<shape, original>

def list_element(values: List<&2, F32>, position: Nat) -> Maybe<&2, F32>:
  match values position:
    case Nil{} _:
      None{}
    case value <> rest 0n:
      Some{value}
    case value <> rest 1n+p:
      list_element(rest, p)

def Elements(-shape: Shape.Shape, +values: List<&2, F32>, -tensor: Core.Tensor<shape>) -> Type:
  @+index: U32 -> @inside: Shape.LT(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) ->
    Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
      ({Core.checked_observe(shape, tensor, index) == list_element(values, U32.to_nat(index)) : Maybe<&2, F32>} &
        {Core.observe(shape, tensor, index) == list_element(values, U32.to_nat(index)) : Maybe<&2, F32>})

def FromValuesMeaning(+shape: Shape.Shape, +values: List<&2, F32>, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Type:
  match result:
    case Fail{error}:
      Unit
    case Done{tensor}:
      Elements(shape, values, tensor)

type FromValuesModel<-shape: Shape.Shape, -values: List<&2, F32>> is Data:
  FromValuesModel{-meaning: FromValuesMeaning(shape, values, Core.from_values(shape, values))}

def ConstructionMeaning(+values: List<&2, F32>, result: Result<&2, &1, Core.Error, Core.Dynamic>) -> Type:
  match result:
    case Fail{error}:
      Unit
    case Done{Core.Packed{+shape, tensor}}:
      Elements(shape, values, tensor)

type ConstructionModel<-dimensions: List<&2, U32>, -values: List<&2, F32>> is Data:
  ConstructionModel{-meaning: ConstructionMeaning(values, Core.construct(dimensions, values))}

law public_from_values_elements:
  for +shape: Shape.Shape
  for +values: List<&2, F32>
  FromValuesModel<shape, values>

law public_construction_elements:
  for +dimensions: List<&2, U32>
  for +values: List<&2, F32>
  ConstructionModel<dimensions, values>

law public_from_values_complete:
  for +shape: Shape.Shape
  for +values: List<&2, F32>
  for counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  Totality.TensorDone<shape, Core.from_values(shape, values)>

law public_construction_complete:
  for +dimensions: List<&2, U32>
  for +values: List<&2, F32>
  for rank_bound: Shape.LE(Shape.rank_dimensions(dimensions), 4n)
  for capacity_bound: Shape.LE(Shape.logical_numel(dimensions), 1048576n)
  for counted: {List.length(&2, F32, values) == Shape.logical_numel(dimensions) : Nat}
  Totality.ConstructionDone<dimensions, Core.construct(dimensions, values)>
