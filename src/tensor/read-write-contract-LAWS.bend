import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./observation.bend as Observation
import ./read-write-records.bend as Records

law read_inside:
  for +shape: Shape.Shape
  for +fill: F32
  for +index: U32
  for inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}
  Records.ReadInside<shape, fill, index>

law read_outside:
  for +shape: Shape.Shape
  for +fill: F32
  for +index: U32
  for outside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == False{} : Bool}
  Records.ReadContract<shape, fill, index, 8>

law write_inside:
  for +shape: Shape.Shape
  for +fill: F32
  for +index: U32
  for +value: F32
  for inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}
  Records.WriteInside<shape, fill, index, value>

law write_outside:
  for +shape: Shape.Shape
  for +fill: F32
  for +index: U32
  for +value: F32
  for outside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == False{} : Bool}
  Records.WriteOutside<shape, fill, index, value>

law coordinate_inside:
  for +shape: Shape.Shape
  for +fill: F32
  for +coordinates: List<&2, U32>
  for +index: U32
  for +value: F32
  for located: {Observation.coordinate_index(shape, Shape.flatten_checked(shape, coordinates)) == Some{index} : Maybe<&2, U32>}
  for inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}
  Records.CoordinateInside<shape, fill, coordinates, index, value>

law dynamic_read_inside:
  for +shape: Shape.Shape
  for +fill: F32
  for +index: U32
  for inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}
  {Observation.dynamic_point(shape, index, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>}
