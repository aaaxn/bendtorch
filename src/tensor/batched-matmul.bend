import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ./core.bend as Core
import ./elementwise.bend as Elementwise
import ./matmul.bend as Matrix
import ./parallel.bend as Parallel

# PyTorch's `torch.matmul`: batched products with 1-D promotion.
#
# Two rank-2 operands are `Matrix.matmul` itself, the same call, so their
# output words, errors and laws are those of `matmul`. Every other pair of
# ranks from 1 to 4 is planned here, as `torch.matmul` plans it:
#
# - A 1-D left operand [k] is the row [1, k], and the output drops that row;
#   a 1-D right operand [k] is the column [k, 1], and the output drops that
#   column. So [k] @ [k] is a rank-0 dot product.
# - Every dimension but the last two of an operand is a batch dimension.
#   The batch dimensions of the two operands broadcast as elementwise
#   operations do: right-aligned, an extent of 1 stretches. The output is
#   the broadcast batch dimensions, then [m, n] with a dropped row or column
#   left out.
# - A rank-0 operand fails with RankMismatch (PyTorch: "both arguments to
#   matmul need to be at least 1D"), unequal inner extents with
#   InnerDimensionMismatch, batch dimensions that do not broadcast with
#   ShapeError{IncompatibleBroadcast}, and an output past the capacity with
#   ShapeError{CapacityExceeded}.
#
# Output cell (b, row, column) adds left[b, row, k] * right[b, k, column]
# for k = 0, 1, ..., inner - 1, starting from +0.0: the terms and the order
# of `Matrix.matmul`'s cell (row, column) on the two matrices of batch b, so
# each batch gets the words `matmul` writes for its matrices. A broadcast
# operand is read in place at its own batch, never copied. The cells are
# built by `Parallel.build`, so no word depends on the split or the thread
# count.

# The batch extents of an operand, padded on the left with ones to two, and
# how many it has.
type Batch is Data:
  Batch{rank: Nat, first: U32, second: U32}

# An operand split into its batch and its matrix: the outer extent (rows of
# the left operand, columns of the right one) and the inner extent. A 1-D
# operand is a vector: an outer extent of 1 that the output drops.
type Operand is Data:
  Operand{batch: Batch, outer: U32, inner: U32, vector: Bool}

def left_operand(dims: List<&2, U32>) -> Result<&2, &2, Matrix.Error, Operand>:
  match dims:
    case +k <> Nil{}:
      Done{Operand{Batch{0n, 1, 1}, 1, k, True{}}}
    case +m <> +k <> Nil{}:
      Done{Operand{Batch{0n, 1, 1}, m, k, False{}}}
    case +b <> +m <> +k <> Nil{}:
      Done{Operand{Batch{1n, 1, b}, m, k, False{}}}
    case +a <> +b <> +m <> +k <> Nil{}:
      Done{Operand{Batch{2n, a, b}, m, k, False{}}}
    case rest:
      Fail{Matrix.RankMismatch{}}

def right_operand(dims: List<&2, U32>) -> Result<&2, &2, Matrix.Error, Operand>:
  match dims:
    case +k <> Nil{}:
      Done{Operand{Batch{0n, 1, 1}, 1, k, True{}}}
    case +k <> +n <> Nil{}:
      Done{Operand{Batch{0n, 1, 1}, n, k, False{}}}
    case +b <> +k <> +n <> Nil{}:
      Done{Operand{Batch{1n, 1, b}, n, k, False{}}}
    case +a <> +b <> +k <> +n <> Nil{}:
      Done{Operand{Batch{2n, a, b}, n, k, False{}}}
    case rest:
      Fail{Matrix.RankMismatch{}}

# One batch axis: equal extents, or an extent of 1 that stretches to the
# other one.
def stretched(+l: U32, +r: U32, same: Bool, left_one: Bool, right_one: Bool) -> Result<&2, &2, Matrix.Error, U32>:
  match same left_one right_one:
    case True{} _ _:
      Done{l}
    case False{} True{} _:
      Done{r}
    case False{} False{} True{}:
      Done{l}
    case False{} False{} False{}:
      Fail{Matrix.ShapeError{Shape.IncompatibleBroadcast{}}}

def axis(+l: U32, +r: U32) -> Result<&2, &2, Matrix.Error, U32>:
  stretched(l, r, Nat.is_eq(U32.to_nat(l), U32.to_nat(r)), Nat.is_eq(U32.to_nat(l), 1n), Nat.is_eq(U32.to_nat(r), 1n))

def larger(+a: Nat, +b: Nat) -> Nat:
  Matrix.least(b, a, Nat.is_lt(a, b))

# The output's batch dimensions: as many as the operand with more of them.
def batch_dims(+rank: Nat, +first: U32, +second: U32) -> List<&2, U32>:
  match rank:
    case 0n:
      []
    case 1n+rest:
      match rest:
        case 0n:
          [second]
        case 1n+more:
          [first, second]

def column_dims(+columns: U32, right_vector: Bool) -> List<&2, U32>:
  match right_vector:
    case True{}:
      []
    case False{}:
      [columns]

def matrix_dims(+rows: U32, left_vector: Bool, +columns: U32, right_vector: Bool) -> List<&2, U32>:
  match left_vector:
    case True{}:
      column_dims(columns, right_vector)
    case False{}:
      rows <> column_dims(columns, right_vector)

# Everything a product and its two gradients need: the output shape, the
# output's batch dimensions, the matrix extents, both operands' padded
# batches and the output's, and which operands are vectors.
type Plan is Data:
  Plan{output: Shape.Shape, batch: List<&2, U32>, rows: Nat, inner: Nat, columns: Nat,
    l1: Nat, l2: Nat, r1: Nat, r2: Nat, o1: Nat, o2: Nat, left_vector: Bool, right_vector: Bool}

def output_of(+plan: Plan) -> Shape.Shape:
  Plan{+output, batch, rows, inner, columns, l1, l2, r1, r2, o1, o2, lv, rv} = plan
  output

def planned_shape(+batch: List<&2, U32>, +rows: U32, +inner: U32, +columns: U32,
  +l1: U32, +l2: U32, +r1: U32, +r2: U32, +o1: U32, +o2: U32, +lv: Bool, +rv: Bool,
  -dims: List<&2, U32>, result: Shape.ConstructResult<dims>) -> Result<&2, &2, Matrix.Error, Plan>:
  match result:
    case Shape.SError{error}:
      Fail{Matrix.ShapeError{error}}
    case Shape.SBuilt{+shape, preserved}:
      Done{Plan{shape, batch, U32.to_nat(rows), U32.to_nat(inner), U32.to_nat(columns),
        U32.to_nat(l1), U32.to_nat(l2), U32.to_nat(r1), U32.to_nat(r2), U32.to_nat(o1), U32.to_nat(o2), lv, rv}}

def batched(+rank: Nat, +rows: U32, +inner: U32, +columns: U32, +l1: U32, +l2: U32, +r1: U32, +r2: U32,
  +lv: Bool, +rv: Bool, first: Result<&2, &2, Matrix.Error, U32>,
  second: Result<&2, &2, Matrix.Error, U32>) -> Result<&2, &2, Matrix.Error, Plan>:
  match first second:
    case Fail{error} second:
      Fail{error}
    case Done{o1} Fail{error}:
      Fail{error}
    case Done{+o1} Done{+o2}:
      planned_shape(batch_dims(rank, o1, o2), rows, inner, columns, l1, l2, r1, r2, o1, o2, lv, rv,
        List.append(&2, U32, batch_dims(rank, o1, o2), matrix_dims(rows, lv, columns, rv)),
        Shape.construct(List.append(&2, U32, batch_dims(rank, o1, o2), matrix_dims(rows, lv, columns, rv))))

def shaped(+left: Operand, +right: Operand) -> Result<&2, &2, Matrix.Error, Plan>:
  Operand{Batch{+lr, +l1, +l2}, +rows, +inner, +lv} = left
  Operand{Batch{+rr, +r1, +r2}, +columns, right_inner, +rv} = right
  batched(larger(lr, rr), rows, inner, columns, l1, l2, r1, r2, lv, rv, axis(l1, r1), axis(l2, r2))

def compatible(+left: Operand, +right: Operand, same: Bool) -> Result<&2, &2, Matrix.Error, Plan>:
  match same:
    case False{}:
      Fail{Matrix.InnerDimensionMismatch{}}
    case True{}:
      shaped(left, right)

def inner_of(+operand: Operand) -> U32:
  Operand{batch, outer, +inner, vector} = operand
  inner

def both_parsed(+left: Operand, result: Result<&2, &2, Matrix.Error, Operand>) -> Result<&2, &2, Matrix.Error, Plan>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{+right}:
      compatible(left, right, Nat.is_eq(U32.to_nat(inner_of(left)), U32.to_nat(inner_of(right))))

def left_parsed(+right: Shape.Shape, result: Result<&2, &2, Matrix.Error, Operand>) -> Result<&2, &2, Matrix.Error, Plan>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{+left}:
      both_parsed(left, right_operand(Shape.dimensions(right)))

# The plan of left @ right for operands of rank 1 to 4: the left operand's
# rank is checked first, then the right one's, the inner extents, the
# batch dimensions and the output's capacity.
def plan(+left: Shape.Shape, +right: Shape.Shape) -> Result<&2, &2, Matrix.Error, Plan>:
  left_parsed(right, left_operand(Shape.dimensions(left)))

# What every cell of a build reads: the orders and extents of the product
# (as in `Matrix.Layout`), and each operand's padded batch extents with the
# output's second batch extent, which splits a batch number into its two
# indices.
type Layout is Data:
  Layout{left: Matrix.Order, right: Matrix.Order, inner: Nat, rows: Nat, columns: Nat,
    l1: Nat, l2: Nat, r1: Nat, r2: Nat, o2: Nat}

def layout(+given: Layout) -> Layout:
  Layout{+lo, +ro, +inner, +rows, +columns, +l1, +l2, +r1, +r2, +o2} = given
  Layout{lo, ro, inner, rows, columns, l1, l2, r1, r2, o2}

# left @ right: left [.., rows, inner] read by rows, right [.., inner,
# columns] read by columns, as `matmul` reads them.
def forward_layout(+plan: Plan) -> Layout:
  Plan{output, batch, +rows, +inner, +columns, +l1, +l2, +r1, +r2, o1, +o2, lv, rv} = plan
  Layout{Matrix.Kept{}, Matrix.Swapped{}, inner, rows, columns, l1, l2, r1, r2, o2}

# g @ right^T, the left operand's gradient over the output's batch: g
# [.., rows, columns] by rows, right [.., inner, columns] by its rows, the
# columns being the terms.
def left_layout(+plan: Plan) -> Layout:
  Plan{output, batch, +rows, +inner, +columns, l1, l2, +r1, +r2, +o1, +o2, lv, rv} = plan
  Layout{Matrix.Kept{}, Matrix.Kept{}, columns, rows, inner, o1, o2, r1, r2, o2}

# left^T @ g, the right operand's gradient over the output's batch: left
# [.., rows, inner] by its columns, g [.., rows, columns] by columns, the
# rows being the terms.
def right_layout(+plan: Plan) -> Layout:
  Plan{output, batch, +rows, +inner, +columns, +l1, +l2, r1, r2, +o1, +o2, lv, rv} = plan
  Layout{Matrix.Swapped{}, Matrix.Swapped{}, rows, inner, columns, l1, l2, o1, o2, o2}

# The dimensions of each operand's gradient before it is reduced to the
# operand's shape: the output's batch, then the operand's own matrix
# dimensions.
def left_dims(+plan: Plan) -> List<&2, U32>:
  Plan{output, +batch, +rows, +inner, columns, l1, l2, r1, r2, o1, o2, +lv, rv} = plan
  List.append(&2, U32, batch, matrix_dims(U32.from_nat(rows), lv, U32.from_nat(inner), False{}))

def right_dims(+plan: Plan) -> List<&2, U32>:
  Plan{output, +batch, rows, +inner, +columns, l1, l2, r1, r2, o1, o2, lv, +rv} = plan
  List.append(&2, U32, batch, matrix_dims(U32.from_nat(inner), False{}, U32.from_nat(columns), rv))

# Index `index` of a batch axis of `extent`: the index itself, or 0 where the
# extent is 1 and stretches.
def stretch(+index: Nat, one: Bool) -> Nat:
  match one:
    case True{}:
      0n
    case False{}:
      index

def batch_number(+first: Nat, +second: Nat, +b1: Nat, +b2: Nat) -> Nat:
  Nat.add(Nat.mul(stretch(b1, Nat.is_eq(first, 1n)), second), stretch(b2, Nat.is_eq(second, 1n)))

# The `fuel` terms of a cell from offsets `lp` and `rp`, from +0.0, each
# next term `ls` and `rs` further on: `Matrix.product_run` after the first
# reads.
def run_from(fuel: Nat, +ls: Nat, +rs: Nat, +lp: Nat, +rp: Nat, la: Array<F32>, ra: Array<F32>) ->
  F32 & (Array<F32> & Array<F32>):
  match fuel:
    case 0n:
      (0.0, (la, ra))
    case 1n+rest:
      Matrix.product_run(rest, ls, rs, lp, rp, 0.0, Array.get(F32, la, U32.from_nat(lp)), Array.get(F32, ra, U32.from_nat(rp)))

# The first term's offsets of cell `position`: batch b = position / (rows *
# columns) splits into (b / o2, b % o2), and each operand reads its own
# batch, a stretched axis at 0, then the cell's row and column at term 0.
def left_start(+layout: Layout, +position: Nat) -> Nat:
  Layout{+lo, ro, +inner, +rows, +columns, +l1, +l2, r1, r2, +o2} = layout
  +batch = Nat.div(position, Nat.mul(rows, columns))
  Nat.add(Nat.mul(batch_number(l1, l2, Nat.div(batch, o2), Nat.mod(batch, o2)), Nat.mul(rows, inner)),
    Matrix.read_offset(lo, inner, rows, Nat.div(Nat.mod(position, Nat.mul(rows, columns)), columns), 0n))

def right_start(+layout: Layout, +position: Nat) -> Nat:
  Layout{lo, +ro, +inner, +rows, +columns, l1, l2, +r1, +r2, +o2} = layout
  +batch = Nat.div(position, Nat.mul(rows, columns))
  Nat.add(Nat.mul(batch_number(r1, r2, Nat.div(batch, o2), Nat.mod(batch, o2)), Nat.mul(inner, columns)),
    Matrix.read_offset(ro, inner, columns, Nat.mod(Nat.mod(position, Nat.mul(rows, columns)), columns), 0n))

# Cell `position`: all `inner` terms, from +0.0.
def dot_at(+layout: Layout, +position: Nat, la: Array<F32>, ra: Array<F32>) -> F32 & (Array<F32> & Array<F32>):
  Layout{+lo, +ro, +inner, +rows, +columns, l1, l2, r1, r2, o2} = layout
  run_from(inner, Matrix.stride(lo, rows), Matrix.stride(ro, columns), left_start(layout, position), right_start(layout, position), la, ra)

# The cell function `Parallel.build` runs: a cell at or past the count is
# +0.0 padding.
def cell_at(+layout: Layout, +position: Nat, inputs: Array<F32> & Array<F32>, inside: Bool) -> F32 & (Array<F32> & Array<F32>):
  (la, ra) = inputs
  match inside:
    case False{}:
      (0.0, (la, ra))
    case True{}:
      dot_at(layout, position, la, ra)

def build(+depth: Nat, +splits: Nat, +given: Layout, +count: Nat, +base: Nat, la: Array<F32>, ra: Array<F32>) -> Array<F32>:
  Parallel.build(~Layout, ~Layout, ~Array<F32>, ~Array<F32>, ~layout, ~Elementwise.split, ~cell_at, given, depth, splits, count, base, (la, ra))

def build_balanced(+depth: Nat, +splits: Nat, +given: Layout, +count: Nat, +base: Nat, la: Array<F32>, ra: Array<F32>) ->
  Model.Balanced(F32, depth, build(depth, splits, given, count, base, la, ra)):
  Parallel.balanced(~Layout, ~Layout, ~Array<F32>, ~Array<F32>, ~layout, ~Elementwise.split, ~cell_at, given, depth, splits, count, base, (la, ra))

def terms_of(+given: Layout) -> Nat:
  Layout{lo, ro, +inner, rows, columns, l1, l2, r1, r2, o2} = given
  inner

# Both inputs come back to the caller as the kept halves of their clones,
# and the build reads the other halves. It forks by work as `Matrix.splits`
# says.
def filled(+given: Layout, +output: Shape.Shape, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(output)) : Nat},
  left: Core.Dynamic & Array<F32>, right: Core.Dynamic & Array<F32>) ->
  Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  (kept_left, la) = left
  (kept_right, ra) = right
  Done{(kept_left, kept_right, Core.pack(output, Core.TTensor{Storage.Owned{depth, length,
    build(depth, Matrix.splits(depth, terms_of(given)), given, length, 0n, la, ra),
    build_balanced(depth, Matrix.splits(depth, terms_of(given)), given, length, 0n, la, ra),
    valid}, exact}))}

def allocated(+given: Layout, +output: Shape.Shape, left: Core.Dynamic, right: Core.Dynamic,
  result: Result<&2, &1, Core.Error, Core.Tensor<output>>) -> Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{Core.TTensor{storage, exact}}:
      match storage:
        case Storage.Empty{}:
          Done{(left, right, Core.pack(output, Core.TTensor{Storage.Empty{}, exact}))}
        case Storage.Owned{+depth, +length, array, balanced, valid}:
          filled(given, output, depth, length, valid, exact, Elementwise.take(left), Elementwise.take(right))

# The product of the storages of `left` and `right` read as `given` says,
# into a new tensor of shape `output`.
def product(+given: Layout, +output: Shape.Shape, left: Core.Dynamic, right: Core.Dynamic) ->
  Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  allocated(given, output, left, right, Core.new(output, 0.0))

def tensor_failed(result: Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>) ->
  Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{Matrix.TensorError{error}}
    case Done{triple}:
      Done{triple}

def planned(+left: Shape.Shape, +right: Shape.Shape, a: Core.Tensor<left>, b: Core.Tensor<right>,
  result: Result<&2, &2, Matrix.Error, Plan>) -> Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{+plan}:
      tensor_failed(product(forward_layout(plan), output_of(plan), Core.Packed{left, a}, Core.Packed{right, b}))

def both_matrices(+left: Shape.Shape, +right: Shape.Shape) -> Bool:
  Bool.and(Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(left)), 2n), Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(right)), 2n))

def routed(+left: Shape.Shape, +right: Shape.Shape, a: Core.Tensor<left>, b: Core.Tensor<right>, matrices: Bool) ->
  Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  match matrices:
    case True{}:
      Matrix.matmul(Core.Packed{left, a}, Core.Packed{right, b})
    case False{}:
      planned(left, right, a, b, plan(left, right))

# torch.matmul(left, right). Both inputs come back to the caller.
def matmul(left: Core.Dynamic, right: Core.Dynamic) -> Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  Core.Packed{+left_shape, a} = left
  Core.Packed{+right_shape, b} = right
  routed(left_shape, right_shape, a, b, both_matrices(left_shape, right_shape))

def gradient_shaped(+given: Layout, first: Core.Dynamic, second: Core.Dynamic, -dims: List<&2, U32>,
  result: Shape.ConstructResult<dims>) -> Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  match result:
    case Shape.SError{error}:
      Fail{Core.ShapeError{error}}
    case Shape.SBuilt{+shape, preserved}:
      product(given, shape, first, second)

# The left operand's gradient over the output's batch, g @ right^T, of
# dimensions `left_dims(plan)`: g has the output's shape and right the right
# operand's. A broadcast left operand still has to be summed to its shape.
def left_gradient(+plan: Plan, g: Core.Dynamic, right: Core.Dynamic) ->
  Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  gradient_shaped(left_layout(plan), g, right, left_dims(plan), Shape.construct(left_dims(plan)))

# The right operand's gradient over the output's batch, left^T @ g, of
# dimensions `right_dims(plan)`.
def right_gradient(+plan: Plan, left: Core.Dynamic, g: Core.Dynamic) ->
  Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  gradient_shaped(right_layout(plan), left, g, right_dims(plan), Shape.construct(right_dims(plan)))
