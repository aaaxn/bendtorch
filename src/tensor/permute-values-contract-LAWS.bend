import Base
import ../shape/core.bend as Shape
import ./layout.bend as Layout
import ./transpose-spec.bend as Transpose
import ./permute-values-records.bend as Records

# `permute_copy` of a `Core.from_values(input, values)` tensor by a valid
# permutation succeeds with dimensions `Axis.order(dims, axes)`; output index i
# holds `values[permute_source(input, axes, i)]`, and the returned source
# still holds `values[j]` at every j.
law permute_contract:
  for +input: Shape.Shape
  for +values: List<&2, F32>
  for counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(input)) : Nat}
  for +axes: List<&2, U32>
  for valid: {True{} == Layout.bijective(input, axes) : Bool}
  Records.PermuteContract<input, values, axes>

# `transpose_copy(a, b)` of a `Core.from_values` tensor with both axes in
# range is the permute contract with the axes that swap a and b.
law transpose_contract:
  for +input: Shape.Shape
  for +values: List<&2, F32>
  for counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(input)) : Nat}
  for +a: U32
  for +b: U32
  for inside: {True{} == Transpose.valid(input, a, b) : Bool}
  Records.TransposeContract<input, values, a, b>
