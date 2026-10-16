import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ../tensor/core.bend as Core
import ../tensor/elementwise.bend as Elementwise
import ../tensor/matmul.bend as Matrix
import ../tensor/parallel.bend as Parallel
import ../tensor/reduction.bend as Reduction
import ./reduce.bend as Reduce
import ./sum-to-size.bend as SumToSize

# Loss functions with mean reduction, as PyTorch's mse_loss and
# cross_entropy over class logits. Every step is a tensor operation that
# builds its output over cells in parallel: Elementwise.binary,
# sum_to_size and the row maximum below. The mean is the ordered sum of
# Reduction.sum_dynamic, added left to right from +0.0, divided by the
# count with F32.div.

type Error is Data:
  TensorError{error: Core.Error}
  ReduceError{error: Reduce.Error}
  NotLogits{}
  TargetOutOfRange{}

def error_code(error: Error) -> U32:
  match error:
    case TensorError{error}:
      Core.error_code(error)
    case ReduceError{error}:
      Reduce.error_code(error)
    case NotLogits{}:
      107
    case TargetOutOfRange{}:
      108

# A loss next to the two tensors its backward reads.
type Recorded is Type:
  Recorded{loss: Core.Dynamic, left: Core.Dynamic, right: Core.Dynamic}

# A count as an F32, rounded as PyTorch converts an integer count.
def count_f32(+count: Nat) -> F32:
  U32.to_f32(U32.from_nat(count))

# 2 / N, the factor of PyTorch's mse_loss backward with mean reduction, for
# the N cells of the broadcast shape.
def mse_scale(+shape: Shape.Shape) -> F32:
  F32.div(2.0, count_f32(Shape.logical_numel(Shape.dimensions(shape))))

# The first dimension of [rows, columns] logits, the batch.
def first_dimension(+dimensions: List<&2, U32>) -> Nat:
  match dimensions:
    case Nil{}:
      0n
    case Con{+first, +rest}:
      U32.to_nat(first)

def rows(+shape: Shape.Shape) -> Nat:
  first_dimension(Shape.dimensions(shape))

# Whether [rows, columns] logits have one column, one class.
def single_column(+dimensions: List<&2, U32>) -> Bool:
  match dimensions:
    case [+rows, +columns]:
      U32.is_eq(columns, 1)
    case _:
      False{}

def one_class(+shape: Shape.Shape) -> Bool:
  single_column(Shape.dimensions(shape))

# Row maximum.

# PyTorch's maximum: NaN when either word is NaN, else the larger word, and
# on a tie the later word, as `_mm512_max_ps(m, v)` keeps its second operand.
def max_step(+m: F32, +v: F32) -> F32:
  Bool.pick(F32, Bool.or(F32.is_gt(m, v), F32.is_ne(m, m)), m, v)

def row_next(+m: F32, rec: F32 -> Array<F32> -> F32 & Array<F32>, pair: Array<F32> & F32) -> F32 & Array<F32>:
  (array, v) = pair
  rec(max_step(m, v), array)

# max_step of m and the `fuel` words from `position` on, left to right.
def row_rest(fuel: Nat, +position: Nat, +m: F32, array: Array<F32>) -> F32 & Array<F32>:
  match fuel:
    case 0n:
      (m, array)
    case 1n+remaining:
      row_next(m, value => kept => row_rest(remaining, 1n+position, value, kept), Array.get(F32, array, U32.from_nat(position)))

def row_first(+columns: Nat, +start: Nat, pair: Array<F32> & F32) -> F32 & Array<F32>:
  (array, first) = pair
  row_rest(Nat.sub(columns, 1n), 1n+start, first, array)

# The F32 of a word.
def from_word(word: U32) -> F32:
  U32{bits} = word
  F32{bits}

# The maximum of row `row` of a row-major [rows, columns] array: its first
# word, then max_step with each later word, left to right. A row of no
# words is -inf (0xFF800000).
def row_max(+columns: Nat, +row: Nat, array: Array<F32>) -> F32 & Array<F32>:
  match columns:
    case 0n:
      (from_word(4286578688), array)
    case 1n+less:
      row_first(1n+less, Nat.mul(row, 1n+less), Array.get(F32, array, U32.from_nat(Nat.mul(row, 1n+less))))

type Width is Data:
  Width{columns: Nat}

def width(+given: Width) -> Width:
  Width{+columns} = given
  Width{columns}

# The cell function of the row-maximum build: cell `position` is the
# maximum of row `position`. The state is the input array next to a Unit, as
# in Reduce.build.
def max_cell(+width: Width, +position: Nat, state: Array<F32> & Unit, inside: Bool) -> F32 & (Array<F32> & Unit):
  Width{+columns} = width
  (array, none) = state
  match inside:
    case False{}:
      (0.0, (array, none))
    case True{}:
      Reduce.keep(row_max(columns, position, array), none)

def max_build(+depth: Nat, +splits: Nat, +columns: Nat, +count: Nat, +base: Nat, array: Array<F32>) -> Array<F32>:
  Parallel.build(~Width, ~Width, ~Array<F32>, ~Unit, ~width, ~Reduce.split, ~max_cell, Width{columns}, depth, splits, count, base,
    (array, Unit{}))

def max_balanced(+depth: Nat, +splits: Nat, +columns: Nat, +count: Nat, +base: Nat, array: Array<F32>) ->
  Model.Balanced(F32, depth, max_build(depth, splits, columns, count, base, array)):
  Parallel.balanced(~Width, ~Width, ~Array<F32>, ~Unit, ~width, ~Reduce.split, ~max_cell, Width{columns}, depth, splits, count, base,
    (array, Unit{}))

def max_rows(+target: Shape.Shape, +columns: Nat, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(target)) : Nat},
  taken: Core.Dynamic & Array<F32>) -> Core.Dynamic & Result<&2, &1, Core.Error, Core.Dynamic>:
  (kept, array) = taken
  +splits = Matrix.splits(depth, columns)
  (kept, Done{Core.Packed{target, Core.TTensor{Storage.Owned{depth, length,
    max_build(depth, splits, columns, length, 0n, array),
    max_balanced(depth, splits, columns, length, 0n, array),
    valid}, exact}}})

def max_allocated(input: Core.Dynamic, +target: Shape.Shape, +columns: Nat,
  result: Result<&2, &1, Core.Error, Core.Tensor<target>>) -> Core.Dynamic & Result<&2, &1, Core.Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      (input, Fail{error})
    case Done{Core.TTensor{storage, exact}}:
      match storage:
        case Storage.Empty{}:
          (input, Done{Core.Packed{target, Core.TTensor{Storage.Empty{}, exact}}})
        case Storage.Owned{+depth, +length, array, balanced, valid}:
          max_rows(target, columns, depth, length, valid, exact, Elementwise.take(input))

# The maximum of each row of [rows, columns] logits, a [rows, 1] tensor.
# Each output cell walks its own row, so the cells are built in parallel,
# forked by the work of a row as Matrix.splits forks a matmul. The input
# comes back unchanged.
def row_maxes(input: Core.Dynamic, +target: Shape.Shape, +columns: Nat) -> Core.Dynamic & Result<&2, &1, Core.Error, Core.Dynamic>:
  max_allocated(input, target, columns, Core.new(target, 0.0))

# Mean.

# The result of a binary operation whose operands the caller no longer needs.
def quotient(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Result<&2, &1, Core.Error, Core.Dynamic>:
  (operands, result) = pair
  result

def divided_by(+count: Nat, pair: Core.Dynamic & Core.Dynamic,
  divisor: Result<&2, &1, Core.Error, Core.Tensor<Reduction.scalar_shape()>>) -> Result<&2, &1, Core.Error, Core.Dynamic>:
  (input, total) = pair
  match divisor:
    case Fail{error}:
      Fail{error}
    case Done{tensor}:
      quotient(Elementwise.binary(Elementwise.Divide{}, total, Core.pack(Reduction.scalar_shape(), tensor)))

# The sum of every word, in ascending flat order from +0.0
# (Reduction.sum_dynamic), divided by `count`: a rank-0 tensor.
def total_over(+count: Nat, result: Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic>) -> Result<&2, &1, Core.Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{pair}:
      divided_by(count, pair, Core.new(Reduction.scalar_shape(), count_f32(count)))

def mean_shaped(pair: Core.Dynamic & Shape.Shape) -> Result<&2, &1, Core.Error, Core.Dynamic>:
  (input, +shape) = pair
  total_over(Shape.logical_numel(Shape.dimensions(shape)), Reduction.sum_dynamic(input))

# PyTorch's mean of every element: the ordered sum divided by the element
# count, F32.div of the two. The sum adds left to right from +0.0, where
# PyTorch's CPU sum adds in a cascade, so for many words the last bits can
# differ.
def mean(input: Core.Dynamic) -> Result<&2, &1, Core.Error, Core.Dynamic>:
  mean_shaped(Core.shape(input))

# mse_loss.

# The loss next to the two tensors the backward reads.
def recorded(left: Core.Dynamic, right: Core.Dynamic, result: Result<&2, &1, Core.Error, Core.Dynamic>) -> Result<&2, &1, Error, Recorded>:
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{loss}:
      Done{Recorded{loss, left, right}}

def mse_squared(e: Core.Dynamic, pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) ->
  Result<&2, &1, Error, Recorded>:
  (operands, result) = pair
  (kept, copy) = operands
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{squares}:
      recorded(e, kept, mean(squares))

def mse_copied(e: Core.Dynamic, pair: Core.Dynamic & Core.Dynamic) -> Result<&2, &1, Error, Recorded>:
  (d, copy) = pair
  mse_squared(e, Elementwise.binary(Elementwise.Mul{}, d, copy))

def mse_both(input: Core.Dynamic, target: Core.Dynamic, d: Result<&2, &1, Core.Error, Core.Dynamic>,
  e: Result<&2, &1, Core.Error, Core.Dynamic>) -> (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Error, Recorded>:
  match d e:
    case Fail{error} e:
      ((input, target), Fail{TensorError{error}})
    case Done{d} Fail{error}:
      ((input, target), Fail{TensorError{error}})
    case Done{d} Done{e}:
      ((input, target), mse_copied(e, Core.clone_dynamic(d)))

def mse_differences(d: Result<&2, &1, Core.Error, Core.Dynamic>, pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Error, Recorded>:
  (operands, e) = pair
  (target, input) = operands
  mse_both(input, target, d, e)

def mse_first(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Error, Recorded>:
  (operands, d) = pair
  (input, target) = operands
  mse_differences(d, Elementwise.binary(Elementwise.Sub{}, target, input))

# PyTorch's mse_loss(input, target) with mean reduction: d = input - target
# and e = target - input over the broadcast shape, then the mean of d * d.
# Both inputs come back unchanged, next to the loss and e and d (in that
# order, the order the tape saves them), the two differences the backward
# scales: PyTorch's backward computes
# (2 / N * d) * g for the input and (2 / N * e) * g for the target.
def mse_loss(input: Core.Dynamic, target: Core.Dynamic) -> (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Error, Recorded>:
  mse_first(Elementwise.binary(Elementwise.Sub{}, input, target))

# cross_entropy.

# Whether every target is below `columns`, and there are `rows` of them.
def targets_fit(+targets: List<&2, U32>, +rows: Nat, +columns: Nat) -> Bool:
  match targets rows:
    case Nil{} 0n:
      True{}
    case Con{+target, +rest} 1n+below:
      Bool.and(Nat.is_lt(U32.to_nat(target), columns), targets_fit(rest, below, columns))
    case _ _:
      False{}

def as_words(+targets: List<&2, U32>) -> List<&2, F32>:
  match targets:
    case Nil{}:
      Nil{}
    case Con{+target, +rest}:
      U32.to_f32(target) <> as_words(rest)

# The class numbers 0.0, 1.0, ... of `fuel` columns from `next`.
def classes(fuel: Nat, +next: Nat) -> List<&2, F32>:
  match fuel:
    case 0n:
      Nil{}
    case 1n+remaining:
      count_f32(next) <> classes(remaining, 1n+next)

def one_hot_pair(targets: Result<&2, &1, Core.Error, Core.Dynamic>, columns: Result<&2, &1, Core.Error, Core.Dynamic>) ->
  Result<&2, &1, Core.Error, Core.Dynamic>:
  match targets columns:
    case Fail{error} columns:
      Fail{error}
    case Done{targets} Fail{error}:
      Fail{error}
    case Done{targets} Done{columns}:
      quotient(Elementwise.binary(Elementwise.Matches{}, targets, columns))

# The one-hot rows of the targets, [rows, columns]: 1.0 where the column is
# the row's target, +0.0 elsewhere. The [rows, 1] targets and the [1, columns]
# class numbers broadcast against each other with Matches, which compares
# their words. With no rows the one-hot rows are empty, and the class
# numbers are not built: [1, columns] could exceed the capacity that
# [0, columns] does not.
def one_hot(+targets: List<&2, U32>, +rows: Nat, +columns: Nat) -> Result<&2, &1, Core.Error, Core.Dynamic>:
  match rows:
    case 0n:
      Core.construct([0, U32.from_nat(columns)], Nil{})
    case 1n+less:
      one_hot_pair(Core.construct([U32.from_nat(1n+less), 1], as_words(targets)),
        Core.construct([1, U32.from_nat(columns)], classes(columns, 0n)))

def ce_total(+rows: Nat, lp: Core.Dynamic, onehot: Core.Dynamic, result: Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic>) ->
  Result<&2, &1, Error, Recorded>:
  recorded(lp, onehot, total_over(rows, result))

def ce_picked(+rows: Nat, pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Result<&2, &1, Error, Recorded>:
  (operands, result) = pair
  (lp, onehot) = operands
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{picked}:
      ce_total(rows, lp, onehot, Reduction.sum_dynamic(picked))

def ce_logged(+rows: Nat, onehot: Core.Dynamic, pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) ->
  Result<&2, &1, Error, Recorded>:
  (operands, result) = pair
  (shifted, lse) = operands
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{lp}:
      ce_picked(rows, Elementwise.binary(Elementwise.PickNegated{}, lp, onehot))

def ce_lse(+rows: Nat, onehot: Core.Dynamic, shifted: Core.Dynamic, result: Result<&2, &1, Core.Error, Core.Dynamic>) ->
  Result<&2, &1, Error, Recorded>:
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{lse}:
      ce_logged(rows, onehot, Elementwise.binary(Elementwise.Sub{}, shifted, lse))

def ce_summed(+rows: Nat, onehot: Core.Dynamic, shifted: Core.Dynamic, pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>,
  zero: Result<&2, &1, Core.Error, Core.Tensor<Reduction.scalar_shape()>>) -> Result<&2, &1, Error, Recorded>:
  (exps, result) = pair
  match result zero:
    case Fail{error} zero:
      Fail{ReduceError{error}}
    case Done{sums} Fail{error}:
      Fail{TensorError{error}}
    case Done{sums} Done{tensor}:
      ce_lse(rows, onehot, shifted,
        quotient(Elementwise.binary(Elementwise.LogLeft{}, sums, Core.pack(Reduction.scalar_shape(), tensor))))

def ce_exps(+rows: Nat, onehot: Core.Dynamic, +target: Shape.Shape, pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) ->
  Result<&2, &1, Error, Recorded>:
  (operands, result) = pair
  (shifted, zero) = operands
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{exps}:
      ce_summed(rows, onehot, shifted, SumToSize.sum_to_size(exps, target), Core.new(Reduction.scalar_shape(), 0.0))

def ce_shifted(+rows: Nat, onehot: Core.Dynamic, +target: Shape.Shape, pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  zero: Result<&2, &1, Core.Error, Core.Tensor<Reduction.scalar_shape()>>) -> Core.Dynamic & Result<&2, &1, Error, Recorded>:
  (operands, result) = pair
  (logits, maxes) = operands
  match result zero:
    case Fail{error} zero:
      (logits, Fail{TensorError{error}})
    case Done{shifted} Fail{error}:
      (logits, Fail{TensorError{error}})
    case Done{shifted} Done{tensor}:
      (logits, ce_exps(rows, onehot, target,
        Elementwise.binary(Elementwise.ExpLeft{}, shifted, Core.pack(Reduction.scalar_shape(), tensor))))

def ce_maxed(+rows: Nat, onehot: Core.Dynamic, +target: Shape.Shape, pair: Core.Dynamic & Result<&2, &1, Core.Error, Core.Dynamic>) ->
  Core.Dynamic & Result<&2, &1, Error, Recorded>:
  (logits, result) = pair
  match result:
    case Fail{error}:
      (logits, Fail{TensorError{error}})
    case Done{maxes}:
      ce_shifted(rows, onehot, target, Elementwise.binary(Elementwise.Sub{}, logits, maxes), Core.new(Reduction.scalar_shape(), 0.0))

def ce_targeted(logits: Core.Dynamic, +rows: Nat, +columns: Nat, onehot: Result<&2, &1, Core.Error, Core.Dynamic>,
  result: Shape.ConstructResult<[U32.from_nat(rows), 1]>) -> Core.Dynamic & Result<&2, &1, Error, Recorded>:
  match onehot result:
    case Fail{error} result:
      (logits, Fail{TensorError{error}})
    case Done{onehot} Shape.SError{error}:
      (logits, Fail{TensorError{Core.ShapeError{error}}})
    case Done{onehot} Shape.SBuilt{+target, preserved}:
      ce_maxed(rows, onehot, target, row_maxes(logits, target, columns))

def ce_checked(logits: Core.Dynamic, +targets: List<&2, U32>, +rows: Nat, +columns: Nat, fits: Bool) ->
  Core.Dynamic & Result<&2, &1, Error, Recorded>:
  match fits:
    case False{}:
      (logits, Fail{TargetOutOfRange{}})
    case True{}:
      ce_targeted(logits, rows, columns, one_hot(targets, rows, columns), Shape.construct([U32.from_nat(rows), 1]))

def ce_dimensioned(logits: Core.Dynamic, +targets: List<&2, U32>, +dimensions: List<&2, U32>) ->
  Core.Dynamic & Result<&2, &1, Error, Recorded>:
  match dimensions:
    case [+rows, +columns]:
      ce_checked(logits, targets, U32.to_nat(rows), U32.to_nat(columns), targets_fit(targets, U32.to_nat(rows), U32.to_nat(columns)))
    case _:
      (logits, Fail{NotLogits{}})

def ce_shaped(+targets: List<&2, U32>, pair: Core.Dynamic & Shape.Shape) -> Core.Dynamic & Result<&2, &1, Error, Recorded>:
  (logits, +shape) = pair
  ce_dimensioned(logits, targets, Shape.dimensions(shape))

# PyTorch's cross_entropy(logits, targets) with mean reduction, for logits
# of [rows, columns] and one class index below `columns` per row. Per row,
# the stable log_softmax: m = the row maximum (max_step), shifted = x - m,
# lse = log of the sum of exp(shifted) in ascending order from +0.0, and
# lp = shifted - lse, PyTorch's (x - max) - log(sum). The loss is the sum of
# -lp at each row's target, in row order from +0.0, divided by the rows.
# The logits come back unchanged, next to the loss, lp and the one-hot rows
# of the targets, which the backward reads. Logits that are not of rank 2
# fail with NotLogits (107); a target count other than the rows or a target
# at or past `columns` with TargetOutOfRange (108).
def cross_entropy(logits: Core.Dynamic, +targets: List<&2, U32>) -> Core.Dynamic & Result<&2, &1, Error, Recorded>:
  ce_shaped(targets, Core.shape(logits))
