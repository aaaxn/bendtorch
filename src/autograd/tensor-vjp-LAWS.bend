import Base
import ../shape/core.bend as Shape
import ../storage/LAWS.bend as StorageLaws
import ../tensor/core.bend as Core
import ../tensor/layout.bend as Layout
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/elementwise.bend as Elementwise
import ../tensor/reduction.bend as Reduction
import ../tensor/matmul.bend as Matrix
import ../tensor/dot-fold.bend as DotFold
import ../tensor/transpose-spec.bend as Transpose
import ../tensor/observation.bend as Values
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/LAWS.bend as Contracts
import ../tensor/contracts/binary-LAWS.bend as Binary
import ./sum-to-size-fold-LAWS.bend as SumToSizeFold
import ./tensor-backward.bend as Backward
import ./tensor-backward-LAWS.bend as TensorBackward

# The values of the tensor VJPs, stated through the pointwise, ordered-fold
# and index-map copy families: each law takes a replayed gradient (and replayed saved operands),
# and states, for both contributions, the target, the dimensions and the word
# at every in-bounds position.

# The words of a pointwise product in index order: word k is
# scalar(operation, left[left_map(plan, k)], right[right_map(plan, k)]), the
# value the pointwise family's broadcast instance writes at k
# (`binary-LAWS.bend` in the tensor contracts).
def cells(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat) -> List<&2, F32>:
  Binary.cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count)

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
  Binary.BinaryWith(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, P)

# The pointwise product the Sub and Mul VJPs compute, `Elementwise.binary`
# with its parallel build, writes the words of the pointwise family's broadcast
# instance, in transport form: for two replayed inputs whose shapes broadcast,
# for every P, if P holds for every result that hands back both inputs with
# their words and gives an output of the plan's shape whose words are `cells`,
# P holds for the product. The output is a replayed tensor, so
# sum_to_size_replayed applies to it. This is the tensor law binary_replayed
# stated for the backward.
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
  P(Elementwise.binary(operation, Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
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

# The axes transpose_dynamic(t, 0, 1) permutes t by, and the shape of the
# copy the plan gives.
def swapped(+shape: Shape.Shape) -> List<&2, U32>:
  Transpose.swapped_axes(shape, 0, 1)

def transposed(+shape: Shape.Shape, +plan: Layout.Plan<shape, swapped(shape)>) -> Shape.Shape:
  Layout.output(shape, swapped(shape), plan)

# The first `count` factor pairs of cell (row, column) of left @ right^T,
# where right^T is right permuted by `axes`: left[row * inner + k] and the
# word of right at permute_source(right, axes, k * columns + column), in
# increasing k. Contracts.factors reads the same pairs from the copy.
def right_transposed_factors(+row: Nat, +column: Nat, +inner: Nat, +columns: Nat,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +axes: List<&2, U32>, +count: Nat) ->
  List<&2, DotFold.Factors>:
  match count:
    case 0n:
      Nil{}
    case 1n+previous:
      List.append(&2, DotFold.Factors,
        right_transposed_factors(row, column, inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, axes, previous),
        [DotFold.Factors{Replayed.value(left, left_fill, left_ops, U32.from_nat(Nat.add(Nat.mul(row, inner), previous))),
          Replayed.value(right, right_fill, right_ops,
            U32.from_nat(Values.permute_source_index(right, axes, Nat.add(Nat.mul(previous, columns), column))))}])

# The first `count` factor pairs of cell (row, column) of left^T @ right,
# where left^T is left permuted by `axes`: the word of left at
# permute_source(left, axes, row * inner + k) and right[k * columns + column],
# in increasing k.
def left_transposed_factors(+row: Nat, +column: Nat, +inner: Nat, +columns: Nat,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +axes: List<&2, U32>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat) -> List<&2, DotFold.Factors>:
  match count:
    case 0n:
      Nil{}
    case 1n+previous:
      List.append(&2, DotFold.Factors,
        left_transposed_factors(row, column, inner, columns, left, left_fill, left_ops, axes, right, right_fill, right_ops, previous),
        [DotFold.Factors{Replayed.value(left, left_fill, left_ops,
            U32.from_nat(Values.permute_source_index(left, axes, Nat.add(Nat.mul(row, inner), previous)))),
          Replayed.value(right, right_fill, right_ops, U32.from_nat(Nat.add(Nat.mul(previous, columns), column)))}])

# What the Matmul VJP promises: the first contribution goes to node a with
# left's dimensions and holds left_cell(row, column) at row * left_columns +
# column for every row < left_rows and column < left_columns; the second goes
# to node b with right's dimensions and holds right_cell(row, column) likewise.
type MatmulPairContract<-a: Nat, -left: Shape.Shape, -left_rows: Nat, -left_columns: Nat, -left_cell: Nat -> Nat -> F32,
  -b: Nat, -right: Shape.Shape, -right_rows: Nat, -right_columns: Nat, -right_cell: Nat -> Nat -> F32, -pass: Backward.Pass> is Type:
  MatmulPairContract{
    first_target: {TensorBackward.sole_target(first_of(pass)) == Some{a} : Maybe<&2, Nat>},
    first_dimensions: {TensorBackward.sole_dimensions(first_of(pass)) == Some{Shape.dimensions(left)} : Maybe<&2, List<&2, U32>>},
    first_value: @+row: Nat -> @row_below: {Nat.is_lt(row, left_rows) == True{} : Bool} ->
      @+column: Nat -> @column_below: {Nat.is_lt(column, left_columns) == True{} : Bool} ->
      {TensorBackward.sole_point(first_of(pass), U32.from_nat(Nat.add(Nat.mul(row, left_columns), column))) ==
        Some{left_cell(row, column)} : Maybe<&2, F32>},
    second_target: {TensorBackward.sole_target(second_of(pass)) == Some{b} : Maybe<&2, Nat>},
    second_dimensions: {TensorBackward.sole_dimensions(second_of(pass)) == Some{Shape.dimensions(right)} : Maybe<&2, List<&2, U32>>},
    second_value: @+row: Nat -> @row_below: {Nat.is_lt(row, right_rows) == True{} : Bool} ->
      @+column: Nat -> @column_below: {Nat.is_lt(column, right_columns) == True{} : Bool} ->
      {TensorBackward.sole_point(second_of(pass), U32.from_nat(Nat.add(Nat.mul(row, right_columns), column))) ==
        Some{right_cell(row, column)} : Maybe<&2, F32>}}

# d(a@b)/da and d(a@b)/db: with the saved operand values x (left) and y
# (right), the Matmul VJP sends g @ y^T to a, then x^T @ g to b. y^T and x^T
# are transpose_dynamic(., 0, 1), the copies whose plans right_plan and
# left_plan name; first and second are the plans of the two products. Cell
# (row, column) of the first holds the ordered dot product of g's row with the
# words of y that the transposed column reads, added in increasing k from
# +0.0, and likewise for the second; each contribution has its operand's
# dimensions. Nothing is reduced: the products already have them.
law tensor_matmul_vjp_values:
  for +a: Nat
  for +b: Nat
  for +left: Shape.Shape
  for +right: Shape.Shape
  for +shape: Shape.Shape
  for right_inside: {True{} == Transpose.valid(right, 0, 1) : Bool}
  for +right_plan: Layout.Plan<right, swapped(right)>
  for right_laid: {Layout.permutation(right, swapped(right)) == Done{right_plan} : Result<&2, &2, Layout.Error, Layout.Plan<right, swapped(right)>>}
  for +first: Matrix.Plan<shape, transposed(right, right_plan)>
  for first_planned: {Matrix.plan(shape, transposed(right, right_plan)) == Done{first} :
    Result<&2, &2, Matrix.Error, Matrix.Plan<shape, transposed(right, right_plan)>>}
  for first_same: {Shape.dimensions(Matrix.output_shape(shape, transposed(right, right_plan), first)) == Shape.dimensions(left) : List<&2, U32>}
  for left_inside: {True{} == Transpose.valid(left, 0, 1) : Bool}
  for +left_plan: Layout.Plan<left, swapped(left)>
  for left_laid: {Layout.permutation(left, swapped(left)) == Done{left_plan} : Result<&2, &2, Layout.Error, Layout.Plan<left, swapped(left)>>}
  for +second: Matrix.Plan<transposed(left, left_plan), shape>
  for second_planned: {Matrix.plan(transposed(left, left_plan), shape) == Done{second} :
    Result<&2, &2, Matrix.Error, Matrix.Plan<transposed(left, left_plan), shape>>}
  for second_same: {Shape.dimensions(Matrix.output_shape(transposed(left, left_plan), shape, second)) == Shape.dimensions(right) : List<&2, U32>}
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  for +left_fill: F32
  for +left_ops: List<&2, StorageLaws.Op>
  for -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat}
  for +right_fill: F32
  for +right_ops: List<&2, StorageLaws.Op>
  for -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat}
  MatmulPairContract<a, left,
    Contracts.matrix_rows(shape, transposed(right, right_plan), first), Contracts.matrix_columns(shape, transposed(right, right_plan), first),
    row => column => DotFold.dot_fold(right_transposed_factors(row, column,
      Contracts.matrix_inner(shape, transposed(right, right_plan), first), Contracts.matrix_columns(shape, transposed(right, right_plan), first),
      shape, fill, ops, right, right_fill, right_ops, swapped(right), Contracts.matrix_inner(shape, transposed(right, right_plan), first)), 0.0),
    b, right,
    Contracts.matrix_rows(transposed(left, left_plan), shape, second), Contracts.matrix_columns(transposed(left, left_plan), shape, second),
    row => column => DotFold.dot_fold(left_transposed_factors(row, column,
      Contracts.matrix_inner(transposed(left, left_plan), shape, second), Contracts.matrix_columns(transposed(left, left_plan), shape, second),
      left, left_fill, left_ops, swapped(left), shape, fill, ops, Contracts.matrix_inner(transposed(left, left_plan), shape, second)), 0.0),
    Backward.vjp(Backward.Matmul{a, b, left, right},
      Backward.Operands{Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
        Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}}},
      Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}})>

# d(x @ W^T)/dx and d(x @ W^T)/dW, the backward of Linear's product: with the
# saved operand values x (left, [batch, in]) and W (right, [out, in], PyTorch's
# layout) and the gradient g ([batch, out]), the MatmulNT VJP sends g @ W to
# a, then g^T @ x to b. Neither product builds a transpose: first is the plan
# of `matmul(g, W)`, and second the plan of `matmul_tn(g, x)`, which reads g
# in place. Cell (i, l) of the first holds the ordered dot product over j of
# g[i, j] * W[j, l], the words of g at i * out + j and of W at j * in + l;
# cell (j, l) of the second holds the ordered dot product over i of
# g[i, j] * x[i, l], the words of g at i * out + j and of x at i * in + l.
# Each sum adds its terms in increasing order from +0.0, and each
# contribution has its operand's dimensions. The bias of Linear is an Add
# entry, whose VJP `tensor_add_vjp_values` states: its gradient is g summed
# over the batch, column by column.
law tensor_matmul_nt_vjp_values:
  for +a: Nat
  for +b: Nat
  for +left: Shape.Shape
  for +right: Shape.Shape
  for +shape: Shape.Shape
  for +first: Matrix.Plan<shape, right>
  for first_planned: {Matrix.plan(shape, right) == Done{first} : Result<&2, &2, Matrix.Error, Matrix.Plan<shape, right>>}
  for first_same: {Shape.dimensions(Matrix.output_shape(shape, right, first)) == Shape.dimensions(left) : List<&2, U32>}
  for +second: Matrix.Product<Matrix.Swapped{}, Matrix.Swapped{}, shape, left>
  for second_planned: {Matrix.product_plan(Matrix.Swapped{}, Matrix.Swapped{}, shape, left) == Done{second} :
    Result<&2, &2, Matrix.Error, Matrix.Product<Matrix.Swapped{}, Matrix.Swapped{}, shape, left>>}
  for second_same: {Shape.dimensions(Matrix.product_shape(Matrix.Swapped{}, Matrix.Swapped{}, shape, left, second)) ==
    Shape.dimensions(right) : List<&2, U32>}
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  for +left_fill: F32
  for +left_ops: List<&2, StorageLaws.Op>
  for -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat}
  for +right_fill: F32
  for +right_ops: List<&2, StorageLaws.Op>
  for -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat}
  MatmulPairContract<a, left, Contracts.matrix_rows(shape, right, first), Contracts.matrix_columns(shape, right, first),
    row => column => DotFold.dot_fold(Contracts.factors(row, column,
      Contracts.matrix_inner(shape, right, first), Contracts.matrix_columns(shape, right, first),
      shape, fill, ops, right, right_fill, right_ops, Contracts.matrix_inner(shape, right, first)), 0.0),
    b, right,
    Contracts.product_rows(Matrix.Swapped{}, Matrix.Swapped{}, shape, left, second),
    Contracts.product_columns(Matrix.Swapped{}, Matrix.Swapped{}, shape, left, second),
    row => column => DotFold.dot_fold(Contracts.product_factors(row, column, Matrix.Swapped{}, Matrix.Swapped{},
      Contracts.product_inner(Matrix.Swapped{}, Matrix.Swapped{}, shape, left, second),
      Contracts.product_rows(Matrix.Swapped{}, Matrix.Swapped{}, shape, left, second),
      Contracts.product_columns(Matrix.Swapped{}, Matrix.Swapped{}, shape, left, second),
      shape, fill, ops, left, left_fill, left_ops, Contracts.product_inner(Matrix.Swapped{}, Matrix.Swapped{}, shape, left, second)), 0.0),
    Backward.vjp(Backward.MatmulNT{a, b, left, right},
      Backward.Operands{Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
        Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}}},
      Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}})>

# matmul_nt reads what matmul reads from the transposed copy: for a right
# operand of dimensions [columns, inner], the factor pairs of cell
# (row, column) of a product of two `Kept` operands (`matmul_nt`), left[row,
# k] and right[column, k], are the pairs `right_transposed_factors` reads
# through `permute_source` on right^T, the pairs `tensor_matmul_vjp_values`
# states for g @ y^T. Both fold the same words in the same order, so
# `matmul_nt(g, y)` and `matmul(g, transpose_copy(y))` agree word for word.
law nt_reads_transpose:
  for +row: Nat
  for +column: Nat
  for +rows: Nat
  for +left: Shape.Shape
  for +left_fill: F32
  for +left_ops: List<&2, StorageLaws.Op>
  for +right: Shape.Shape
  for +right_fill: F32
  for +right_ops: List<&2, StorageLaws.Op>
  for +columns: U32
  for +inner: U32
  for matrix: {Shape.dimensions(right) == [columns, inner] : List<&2, U32>}
  for column_below: {Nat.is_lt(column, U32.to_nat(columns)) == True{} : Bool}
  for +count: Nat
  for room: {Nat.is_le(count, U32.to_nat(inner)) == True{} : Bool}
  {Contracts.product_factors(row, column, Matrix.Kept{}, Matrix.Kept{}, U32.to_nat(inner), rows, U32.to_nat(columns),
    left, left_fill, left_ops, right, right_fill, right_ops, count) ==
    right_transposed_factors(row, column, U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops,
      right, right_fill, right_ops, swapped(right), count) : List<&2, DotFold.Factors>}

# matmul_tn likewise: for a left operand of dimensions [inner, rows], the
# pairs of a product of two `Swapped` operands (`matmul_tn`), left[k, row] and
# right[k, column], are the pairs `left_transposed_factors` reads through
# `permute_source` on left^T, the pairs `tensor_matmul_vjp_values` states for
# x^T @ g.
law tn_reads_transpose:
  for +row: Nat
  for +column: Nat
  for +columns: Nat
  for +left: Shape.Shape
  for +left_fill: F32
  for +left_ops: List<&2, StorageLaws.Op>
  for +right: Shape.Shape
  for +right_fill: F32
  for +right_ops: List<&2, StorageLaws.Op>
  for +inner: U32
  for +rows: U32
  for matrix: {Shape.dimensions(left) == [inner, rows] : List<&2, U32>}
  for row_below: {Nat.is_lt(row, U32.to_nat(rows)) == True{} : Bool}
  for +count: Nat
  for room: {Nat.is_le(count, U32.to_nat(inner)) == True{} : Bool}
  {Contracts.product_factors(row, column, Matrix.Swapped{}, Matrix.Swapped{}, U32.to_nat(inner), U32.to_nat(rows), columns,
    left, left_fill, left_ops, right, right_fill, right_ops, count) ==
    left_transposed_factors(row, column, U32.to_nat(inner), columns, left, left_fill, left_ops, swapped(left),
      right, right_fill, right_ops, count) : List<&2, DotFold.Factors>}
