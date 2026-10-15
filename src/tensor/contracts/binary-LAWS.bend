import Base
import ../../shape/core.bend as Shape
import ../../storage/LAWS.bend as StorageLaws
import ../core.bend as Core
import ../lifecycle.bend as Lifecycle
import ../elementwise.bend as Elementwise
import ./replayed.bend as Replayed
import ./pointwise.bend as Pointwise
import ./broadcast.bend as Broadcast
import ./LAWS.bend as Contracts

# The words of a pointwise product in index order: word k is
# scalar(operation, left[left_map(plan, k)], right[right_map(plan, k)]), the
# value the pointwise family's broadcast instance writes at k.
def cells(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat) -> List<&2, F32>:
  match count:
    case 0n:
      Nil{}
    case 1n+previous:
      List.append(&2, F32, cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous),
        [Pointwise.cell(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map,
          ~Broadcast.right_map, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous)])

# What the transport form of binary_replayed asks of P: P holds for both
# inputs, replayed with any histories that keep their words, next to the
# output replayed over +0.0 with any history whose words are `cells`.
def BinaryWith(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -P: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type) -> Type:
  @+left_kept: List<&2, StorageLaws.Op> ->
  @left_words: {Contracts.words(left, left_fill, left_kept, Shape.logical_numel(Shape.dimensions(left))) ==
    Contracts.words(left, left_fill, left_ops, Shape.logical_numel(Shape.dimensions(left))) : List<&2, F32>} ->
  @-left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_kept)) == Shape.logical_numel(Shape.dimensions(left)) : Nat} ->
  @+right_kept: List<&2, StorageLaws.Op> ->
  @right_words: {Contracts.words(right, right_fill, right_kept, Shape.logical_numel(Shape.dimensions(right))) ==
    Contracts.words(right, right_fill, right_ops, Shape.logical_numel(Shape.dimensions(right))) : List<&2, F32>} ->
  @-right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_kept)) == Shape.logical_numel(Shape.dimensions(right)) : Nat} ->
  @+output_ops: List<&2, StorageLaws.Op> ->
  @values: {Contracts.words(Shape.output_shape(plan), 0.0, output_ops, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) ==
    cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops,
      Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) : List<&2, F32>} ->
  @-output_fits: {Lifecycle.logical_length(Replayed.storage(Shape.output_shape(plan), 0.0, output_ops)) ==
    Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat} ->
    P(((Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_kept), left_fits}},
      Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_kept), right_fits}}),
      Done{Core.Packed{Shape.output_shape(plan), Core.TTensor{Replayed.storage(Shape.output_shape(plan), 0.0, output_ops), output_fits}}}))

# The parallel elementwise build exports its replayed result. For two
# replayed inputs whose shapes broadcast, for every P, if P holds for every
# result that hands back both inputs with their words and gives an output of
# the plan's shape whose words are `cells`, then P holds for
# `Elementwise.binary`. The output word at k is the value the pointwise family
# writes at k, whatever split of the output into parallel halves the build
# takes, and the output is a replayed tensor, so a law about replayed tensors
# applies to it.
law binary_replayed:
  for +operation: Elementwise.Operation
  for +left: Shape.Shape
  for +right: Shape.Shape
  for +plan: Shape.BroadcastPlan
  for planned: {Shape.broadcast_checked(left, right) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for +left_fill: F32
  for +left_ops: List<&2, StorageLaws.Op>
  for -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat}
  for +right_fill: F32
  for +right_ops: List<&2, StorageLaws.Op>
  for -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat}
  for -P: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type
  for h: BinaryWith(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, P)
  P(Elementwise.binary(operation, Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
    Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}}))
