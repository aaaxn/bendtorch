import Base
import ../shape/core.bend as Shape
import ../storage/LAWS.bend as StorageLaws
import ../tensor/core.bend as Core
import ../tensor/layout.bend as Layout
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/elementwise.bend as Elementwise
import ../tensor/reduction.bend as Reduction
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/pointwise.bend as Pointwise
import ../tensor/contracts/broadcast.bend as PointwiseBroadcast
import ../tensor/contracts/LAWS.bend as Contracts
import ./sum-to-size-fold-LAWS.bend as SumToSizeFold
import ./tensor-backward.bend as Backward
import ./tensor-backward-LAWS.bend as TensorBackward

# The values of the tensor VJPs, stated through the pointwise and ordered-fold
# families: each law takes a replayed gradient (and replayed saved operands),
# and states, for both contributions, the target, the dimensions and the word
# at every in-bounds position.

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
        [Pointwise.cell(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~PointwiseBroadcast.left_map,
          ~PointwiseBroadcast.right_map, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous)])

# What sum_to_size(contribution, target) holds at target cell index, for a
# contribution of `shape` whose words are `values`: the word itself when the
# target has the contribution's dimensions, the ordered fiber sum otherwise.
def reduced(+target: Shape.Shape, +shape: Shape.Shape, +plan: Shape.BroadcastPlan, +values: List<&2, F32>, +index: U32) ->
  Maybe<&2, F32>:
  SumToSizeFold.summed(Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)), plan,
    Shape.logical_numel(Shape.dimensions(shape)), values, index)

# What the transport form of tensor_product_replayed asks of P: P holds for
# both inputs, replayed with any histories that keep their words, next to the
# output replayed over +0.0 with any history whose words are `cells`.
def ProductWith(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
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

# The pointwise product the Sub and Mul VJPs compute is the pointwise family's
# broadcast instance, in transport form: for two replayed inputs whose shapes
# broadcast, for every P, if P holds for every result that hands back both
# inputs with their words and gives an output of the plan's shape whose words
# are `cells`, P holds for the product. The output is a replayed tensor, so
# sum_to_size_replayed applies to it.
law tensor_product_replayed:
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
  for h: ProductWith(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, P)
  P(Backward.product(operation, Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
    Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}}))

# The two contributions of a VJP, when it sends exactly two.
def first_of(pass: Backward.Pass) -> Maybe<&1, Backward.Contribution>:
  Backward.Pass{saved, g, contributions} = pass
  match contributions:
    case Fail{error}:
      None{}
    case Done{Nil{}}:
      None{}
    case Done{Con{first, Nil{}}}:
      None{}
    case Done{Con{first, Con{second, Nil{}}}}:
      Some{first}
    case Done{Con{first, Con{second, Con{other, rest}}}}:
      None{}

def second_of(pass: Backward.Pass) -> Maybe<&1, Backward.Contribution>:
  Backward.Pass{saved, g, contributions} = pass
  match contributions:
    case Fail{error}:
      None{}
    case Done{Nil{}}:
      None{}
    case Done{Con{first, Nil{}}}:
      None{}
    case Done{Con{first, Con{second, Nil{}}}}:
      Some{second}
    case Done{Con{first, Con{second, Con{other, rest}}}}:
      None{}

# What a VJP that sends two contributions promises: the first goes to node a
# with left's dimensions and holds left_cell(i) at every in-bounds i, the
# second goes to node b with right's dimensions and holds right_cell(i).
type PairContract<-a: Nat, -left: Shape.Shape, -left_cell: U32 -> Maybe<&2, F32>,
  -b: Nat, -right: Shape.Shape, -right_cell: U32 -> Maybe<&2, F32>, -pass: Backward.Pass> is Type:
  PairContract{
    first_target: {TensorBackward.sole_target(first_of(pass)) == Some{a} : Maybe<&2, Nat>},
    first_dimensions: {TensorBackward.sole_dimensions(first_of(pass)) == Some{Shape.dimensions(left)} : Maybe<&2, List<&2, U32>>},
    first_value: @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool} ->
      {TensorBackward.sole_point(first_of(pass), index) == left_cell(index) : Maybe<&2, F32>},
    second_target: {TensorBackward.sole_target(second_of(pass)) == Some{b} : Maybe<&2, Nat>},
    second_dimensions: {TensorBackward.sole_dimensions(second_of(pass)) == Some{Shape.dimensions(right)} : Maybe<&2, List<&2, U32>>},
    second_value: @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool} ->
      {TensorBackward.sole_point(second_of(pass), index) == right_cell(index) : Maybe<&2, F32>}}

# d(a+b)/da and d(a+b)/db: for a replayed gradient g, and operand shapes that
# each broadcast to exactly g's shape, the Add VJP sends g reduced to the left
# operand's shape to a, then g reduced to the right operand's shape to b. Each
# cell holds g's own word when the operand has g's dimensions (-0.0 stays
# -0.0), and otherwise the words of g that land on it, added in index order
# from +0.0.
law tensor_add_vjp_values:
  for +a: Nat
  for +b: Nat
  for +left: Shape.Shape
  for +right: Shape.Shape
  for +shape: Shape.Shape
  for +left_plan: Shape.BroadcastPlan
  for left_planned: {Shape.broadcast_checked(left, shape) == Done{left_plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for left_same: {Shape.dimensions(Shape.output_shape(left_plan)) == Shape.dimensions(shape) : List<&2, U32>}
  for +right_plan: Shape.BroadcastPlan
  for right_planned: {Shape.broadcast_checked(right, shape) == Done{right_plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for right_same: {Shape.dimensions(Shape.output_shape(right_plan)) == Shape.dimensions(shape) : List<&2, U32>}
  for saved: Backward.Saved
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  PairContract<a, left, index => reduced(left, shape, left_plan, Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))), index),
    b, right, index => reduced(right, shape, right_plan, Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))), index),
    Backward.vjp(Backward.Add{a, b, left, right}, saved, Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}})>

# d(a-b)/da and d(a-b)/db: the Sub VJP sends g reduced to the left operand's
# shape to a, as Add does, then -g reduced to the right operand's shape to b.
# -g is the pointwise product of g with NegateLeft against a rank-0 +0.0, so
# its word k is F32.neg of g's word at left_map(negated, k): the sign bit is
# flipped before any word is added, as in PyTorch.
law tensor_sub_vjp_values:
  for +a: Nat
  for +b: Nat
  for +left: Shape.Shape
  for +right: Shape.Shape
  for +shape: Shape.Shape
  for +left_plan: Shape.BroadcastPlan
  for left_planned: {Shape.broadcast_checked(left, shape) == Done{left_plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for left_same: {Shape.dimensions(Shape.output_shape(left_plan)) == Shape.dimensions(shape) : List<&2, U32>}
  for +negated: Shape.BroadcastPlan
  for negated_planned: {Shape.broadcast_checked(shape, Reduction.scalar_shape()) == Done{negated} :
    Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for +right_plan: Shape.BroadcastPlan
  for right_planned: {Shape.broadcast_checked(right, Shape.output_shape(negated)) == Done{right_plan} :
    Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for right_same: {Shape.dimensions(Shape.output_shape(right_plan)) == Shape.dimensions(Shape.output_shape(negated)) : List<&2, U32>}
  for saved: Backward.Saved
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  PairContract<a, left, index => reduced(left, shape, left_plan, Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))), index),
    b, right, index => reduced(right, Shape.output_shape(negated), right_plan,
      cells(Elementwise.NegateLeft{}, negated, shape, fill, ops, Reduction.scalar_shape(), 0.0, Nil{},
        Shape.logical_numel(Shape.dimensions(Shape.output_shape(negated)))), index),
    Backward.vjp(Backward.Sub{a, b, left, right}, saved, Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}})>

# d(a*b)/da and d(a*b)/db: with the saved operand values x (left) and y
# (right), the Mul VJP sends g * y reduced to the left operand's shape to a,
# then g * x reduced to the right operand's shape to b. Word k of g * y is
# F32.mul(g[left_map(first, k)], y[right_map(first, k)]), and of g * x
# F32.mul(g[left_map(second, k)], x[right_map(second, k)]).
law tensor_mul_vjp_values:
  for +a: Nat
  for +b: Nat
  for +left: Shape.Shape
  for +right: Shape.Shape
  for +shape: Shape.Shape
  for +first: Shape.BroadcastPlan
  for first_planned: {Shape.broadcast_checked(shape, right) == Done{first} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for +left_plan: Shape.BroadcastPlan
  for left_planned: {Shape.broadcast_checked(left, Shape.output_shape(first)) == Done{left_plan} :
    Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for left_same: {Shape.dimensions(Shape.output_shape(left_plan)) == Shape.dimensions(Shape.output_shape(first)) : List<&2, U32>}
  for +second: Shape.BroadcastPlan
  for second_planned: {Shape.broadcast_checked(shape, left) == Done{second} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for +right_plan: Shape.BroadcastPlan
  for right_planned: {Shape.broadcast_checked(right, Shape.output_shape(second)) == Done{right_plan} :
    Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for right_same: {Shape.dimensions(Shape.output_shape(right_plan)) == Shape.dimensions(Shape.output_shape(second)) : List<&2, U32>}
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  for +left_fill: F32
  for +left_ops: List<&2, StorageLaws.Op>
  for -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat}
  for +right_fill: F32
  for +right_ops: List<&2, StorageLaws.Op>
  for -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat}
  PairContract<a, left, index => reduced(left, Shape.output_shape(first), left_plan,
      cells(Elementwise.Mul{}, first, shape, fill, ops, right, right_fill, right_ops, Shape.logical_numel(Shape.dimensions(Shape.output_shape(first)))),
      index),
    b, right, index => reduced(right, Shape.output_shape(second), right_plan,
      cells(Elementwise.Mul{}, second, shape, fill, ops, left, left_fill, left_ops, Shape.logical_numel(Shape.dimensions(Shape.output_shape(second)))),
      index),
    Backward.vjp(Backward.Mul{a, b, left, right},
      Backward.Operands{Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
        Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}}},
      Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}})>
