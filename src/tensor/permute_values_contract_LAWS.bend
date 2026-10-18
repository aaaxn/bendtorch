import Base
import ../shape/core.bend as Shape
import ./layout.bend as Layout
import ./transpose_spec.bend as Transpose
import ./permute_values_records.bend as Records
import ../storage/LAWS.bend as StorageLaws
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./transforms.bend as Transforms
import ./contracts/replayed.bend as Replayed

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

# `permute_dynamic` of a replayed tensor, once `Layout.permutation` gives the
# plan, in the transport form of `broadcast_to_replayed`: for every P, if P
# holds for every result that hands the source back with the same words and
# gives an output of the plan's shape, replayed over +0.0, whose cell i holds
# the source's word at `permute_source(input, axes, i)`, then P holds for the
# call. The output is a replayed tensor, so a later family law applies to it.
law permute_replayed:
  for +input: Shape.Shape
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(input, fill, ops)) == Shape.logical_numel(Shape.dimensions(input)) : Nat}
  for +axes: List<&2, U32>
  for +plan: Layout.Plan<input, axes>
  for laid: {Layout.permutation(input, axes) == Done{plan} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>}
  for -P: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type
  for h: Records.PermutedWith(input, fill, ops, axes, plan, P)
  P(Transforms.permute_dynamic(Core.Packed{input, Core.TTensor{Replayed.storage(input, fill, ops), fits}}, axes))

# `transpose_dynamic(a, b)` of a replayed tensor with both axes in range is the
# permute of `permute_replayed` by the swapped axes, in the same transport form.
law transpose_replayed:
  for +input: Shape.Shape
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(input, fill, ops)) == Shape.logical_numel(Shape.dimensions(input)) : Nat}
  for +a: U32
  for +b: U32
  for inside: {True{} == Transpose.valid(input, a, b) : Bool}
  for +plan: Layout.Plan<input, Transpose.swapped_axes(input, a, b)>
  for laid: {Layout.permutation(input, Transpose.swapped_axes(input, a, b)) == Done{plan} :
    Result<&2, &2, Layout.Error, Layout.Plan<input, Transpose.swapped_axes(input, a, b)>>}
  for -P: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type
  for h: Records.PermutedWith(input, fill, ops, Transpose.swapped_axes(input, a, b), plan, P)
  P(Transforms.transpose_dynamic(Core.Packed{input, Core.TTensor{Replayed.storage(input, fill, ops), fits}}, a, b))
