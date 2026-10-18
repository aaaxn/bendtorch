import Base
import ../shape/core.bend as Shape
import ./from_values_records.bend as Records

law from_values_contract:
  for +shape: Shape.Shape
  for +values: List<&2, F32>
  for counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  Records.FromValuesContract<shape, values>
