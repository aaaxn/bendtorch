import Base
import ../shape/core.bend as Shape
import ./clone-metadata-records.bend as Records

law clone_contract:
  for +shape: Shape.Shape
  for +fill: F32
  Records.CloneContract<shape, fill>

law clone_write:
  for +shape: Shape.Shape
  for +fill: F32
  for +index: U32
  for +value: F32
  for inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}
  Records.CloneWrite<shape, fill, index, value>

law metadata_contract:
  for +shape: Shape.Shape
  for +fill: F32
  Records.MetadataContract<shape, fill>
