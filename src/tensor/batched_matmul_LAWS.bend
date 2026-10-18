import Base
import ../shape/core.bend as Shape
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ./core.bend as Core
import ./matmul.bend as Matrix
import ./batched_matmul.bend as Batched
import ./dot_fold.bend as DotFold

# The laws of `Batched.matmul`, PyTorch's torch.matmul: which call answers
# a pair of operands, which pairs it refuses, and what every cell of its
# parallel build holds.

type Holds<-P: Type> is Data:
  Holds{-proof: P}

# Two rank-2 operands are `Matrix.matmul` itself: the same result, words,
# errors and returned inputs, so every law of `matmul` holds for them and
# their words do not change.
law batched_matmul_matrices:
  for +left: Shape.Shape
  for +right: Shape.Shape
  for a: Core.Tensor<left>
  for b: Core.Tensor<right>
  for +rows: U32
  for +inner: U32
  for +right_inner: U32
  for +columns: U32
  for left_matrix: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}
  for right_matrix: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>}
  {Batched.matmul(Core.Packed{left, a}, Core.Packed{right, b}) == Matrix.matmul(Core.Packed{left, a}, Core.Packed{right, b}) :
    Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}

# A rank-0 left operand fails with RankMismatch, whatever the right operand
# is, as PyTorch refuses it ("both arguments to matmul need to be at least
# 1D").
law batched_matmul_left_scalar:
  for +left: Shape.Shape
  for +right: Shape.Shape
  for a: Core.Tensor<left>
  for b: Core.Tensor<right>
  for scalar: {Shape.dimensions(left) == [] : List<&2, U32>}
  {Batched.matmul(Core.Packed{left, a}, Core.Packed{right, b}) == Fail{Matrix.RankMismatch{}} :
    Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}

# A rank-0 right operand fails with RankMismatch, whatever the left operand
# is.
law batched_matmul_right_scalar:
  for +left: Shape.Shape
  for +right: Shape.Shape
  for a: Core.Tensor<left>
  for b: Core.Tensor<right>
  for scalar: {Shape.dimensions(right) == [] : List<&2, U32>}
  {Batched.matmul(Core.Packed{left, a}, Core.Packed{right, b}) == Fail{Matrix.RankMismatch{}} :
    Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}

# A vector [k] times a vector [k'] with k != k' fails with
# InnerDimensionMismatch: torch.matmul's dot product refuses unequal
# lengths.
law batched_matmul_dot_inner_error:
  for +left: Shape.Shape
  for +right: Shape.Shape
  for a: Core.Tensor<left>
  for b: Core.Tensor<right>
  for +inner: U32
  for +right_inner: U32
  for left_vector: {Shape.dimensions(left) == [inner] : List<&2, U32>}
  for right_vector: {Shape.dimensions(right) == [right_inner] : List<&2, U32>}
  for mismatch: {U32.to_nat(inner) != U32.to_nat(right_inner) : Nat}
  {Batched.matmul(Core.Packed{left, a}, Core.Packed{right, b}) == Fail{Matrix.InnerDimensionMismatch{}} :
    Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}

# The cells of a build: the layout's fields, named so a law can take them.
def layout_of(+lo: Matrix.Order, +ro: Matrix.Order, +inner: Nat, +rows: Nat, +columns: Nat,
  +l1: Nat, +l2: Nat, +r1: Nat, +r2: Nat, +o2: Nat) -> Batched.Layout:
  Batched.Layout{lo, ro, inner, rows, columns, l1, l2, r1, r2, o2}

# The factor pairs a run of `fuel` terms reads: A at lp, lp + ls,
# lp + 2 ls, ... and B at rp, rp + rs, ..., in that order.
def Strided(+fuel: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +ls: Nat, +rs: Nat, +lp: Nat, +rp: Nat,
  values: List<&2, DotFold.Factors>) -> Data:
  match fuel values:
    case 0n Nil{}:
      Unit
    case 1n+tail DotFold.Factors{lv, rv} <> rest:
      StridedStep<tail, dl, dr, A, B, ls, rs, lp, rp, lv, rv, rest>
    case 0n DotFold.Factors{lv, rv} <> rest:
      Empty
    case 1n+tail Nil{}:
      Empty

type StridedStep<-fuel: Nat, -dl: Nat, -dr: Nat, -A: Array<F32>, -B: Array<F32>, -ls: Nat, -rs: Nat, -lp: Nat, -rp: Nat,
  -lv: F32, -rv: F32, -values: List<&2, DotFold.Factors>> is Data:
  StridedStep{-left_observed: {Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(lp))) == Some{lv} : Maybe<&2, F32>},
    -right_observed: {Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(rp))) == Some{rv} : Maybe<&2, F32>},
    -remaining: Strided(fuel, dl, dr, A, B, ls, rs, Nat.add(ls, lp), Nat.add(rs, rp), values)}

# What cell `position` of a build under the layout holds below the count:
# the fold, from +0.0 in increasing k, of the `inner` factor pairs that
# start at the cell's first reads (`Batched.left_start`, which reads the
# left operand's own batch of the cell's batch, a stretched batch axis at
# 0, and the cell's row; `Batched.right_start` the same for the right
# operand and the cell's column) and step by each operand's stride.
type CellAt<-dl: Nat, -dr: Nat, -A: Array<F32>, -B: Array<F32>, -lo: Matrix.Order, -ro: Matrix.Order, -inner: Nat, -rows: Nat,
  -columns: Nat, -l1: Nat, -l2: Nat, -r1: Nat, -r2: Nat, -o2: Nat, -position: Nat, -value: Maybe<&2, F32>> is Data:
  CellAt{-word: F32, -observed: {value == Some{word} : Maybe<&2, F32>}, -values: List<&2, DotFold.Factors>,
    -logical: Strided(inner, dl, dr, A, B, Matrix.stride(lo, rows), Matrix.stride(ro, columns),
      Batched.left_start(layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position),
      Batched.right_start(layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position), values),
    -fold: {word == DotFold.dot_fold(values, 0.0) : F32}}

def CellSpec(+dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +lo: Matrix.Order, +ro: Matrix.Order, +inner: Nat, +rows: Nat,
  +columns: Nat, +l1: Nat, +l2: Nat, +r1: Nat, +r2: Nat, +o2: Nat, +position: Nat, inside: Bool, -value: Maybe<&2, F32>) -> Type:
  match inside:
    case False{}:
      {value == Some{0.0} : Maybe<&2, F32>}
    case True{}:
      CellAt<dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, value>

# Every slot of `out`, cell `base + offset`, holds what CellSpec names: the
# cell's fold below `count`, +0.0 at or above it.
def Cells(+depth: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +lo: Matrix.Order, +ro: Matrix.Order, +inner: Nat,
  +rows: Nat, +columns: Nat, +l1: Nat, +l2: Nat, +r1: Nat, +r2: Nat, +o2: Nat, +count: Nat, +base: Nat, -out: Array<F32>) -> Type:
  @+path: Model.Path(depth) ->
    CellSpec(dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, Nat.add(Model.offset(depth, path), base),
      Nat.is_lt(Nat.add(Model.offset(depth, path), base), count), Model.observe(F32, depth, out, path))

# For two balanced input arrays A and B and every split, every cell of the
# parallel build holds the fold its layout names, so no word depends on the
# split or the thread count, and both arrays are only read. The forward
# product and both gradients of the tape's BatchedMatmul are such builds
# (`Batched.forward_layout`, `left_layout`, `right_layout`).
law batched_build_cells:
  for +depth: Nat
  for +splits: Nat
  for +dl: Nat
  for +dr: Nat
  for -A: Array<F32>
  for -B: Array<F32>
  for +lo: Matrix.Order
  for +ro: Matrix.Order
  for +inner: Nat
  for +rows: Nat
  for +columns: Nat
  for +l1: Nat
  for +l2: Nat
  for +r1: Nat
  for +r2: Nat
  for +o2: Nat
  for +count: Nat
  for +base: Nat
  for la: Array<F32>
  for ra: Array<F32>
  for el: {la == A : Array<F32>}
  for er: {ra == B : Array<F32>}
  for left_valid: Model.Balanced(F32, dl, A)
  for right_valid: Model.Balanced(F32, dr, B)
  Holds<Cells(depth, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, count, base,
    Batched.build(depth, splits, layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), count, base, la, ra))>

# The build is a balanced tree of its depth, whatever the split: the
# certificate the output storage carries.
law batched_build_balanced:
  for +depth: Nat
  for +splits: Nat
  for +given: Batched.Layout
  for +count: Nat
  for +base: Nat
  for la: Array<F32>
  for ra: Array<F32>
  Model.Balanced(F32, depth, Batched.build(depth, splits, given, count, base, la, ra))

# Which words each cell reads, written from torch.matmul's definition and
# the shape module's broadcast, with no call into the batched code.
#
# Cell (b1, b2, row, column) of an output whose batch has a second extent
# of o2 and whose matrices are rows x columns sits at its row-major
# position, the cells of one batch after those of the batch before it.
def cell_position(+o2: Nat, +rows: Nat, +columns: Nat, +b1: Nat, +b2: Nat, +row: Nat, +column: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(b1, o2), b2), Nat.mul(rows, columns)), Nat.add(Nat.mul(row, columns), column))

# Term k of row (or column) `outer` of a matrix of `outers` rows (or
# columns) and `inner` terms: `[outers, inner]` read by rows (Kept), or
# `[inner, outers]` read by columns (Swapped).
def matrix_offset(+order: Matrix.Order, +inner: Nat, +outers: Nat, +outer: Nat, +k: Nat) -> Nat:
  match order:
    case Matrix.Kept{}:
      Nat.add(Nat.mul(outer, inner), k)
    case Matrix.Swapped{}:
      Nat.add(Nat.mul(k, outers), outer)

# The row-major offset of that term in an operand of batch extents
# [e1, e2] read at batch (b1, b2) of the output: an axis of extent 1 is
# broadcast and read at index 0 (`Shape.broadcast_index`, the rule of the
# elementwise operations), any other at the output's own index.
def operand_offset(+order: Matrix.Order, +inner: Nat, +outers: Nat, +e1: Nat, +e2: Nat, +b1: Nat, +b2: Nat, +outer: Nat, +k: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(Shape.broadcast_index(e1, b1), e2), Shape.broadcast_index(e2, b2)), Nat.mul(outers, inner)),
    matrix_offset(order, inner, outers, outer, k))

# The factor pairs k, k + 1, ... of cell (b1, b2, row, column): the left
# operand's word at its term k of row `row`, the right operand's at its
# term k of column `column`, each at its own batch.
def Logical(+fuel: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +lo: Matrix.Order, +ro: Matrix.Order, +inner: Nat,
  +rows: Nat, +columns: Nat, +l1: Nat, +l2: Nat, +r1: Nat, +r2: Nat, +b1: Nat, +b2: Nat, +row: Nat, +column: Nat, +k: Nat,
  values: List<&2, DotFold.Factors>) -> Data:
  match fuel values:
    case 0n Nil{}:
      Unit
    case 1n+tail DotFold.Factors{lv, rv} <> rest:
      LogicalStep<tail, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, b1, b2, row, column, k, lv, rv, rest>
    case 0n DotFold.Factors{lv, rv} <> rest:
      Empty
    case 1n+tail Nil{}:
      Empty

type LogicalStep<-fuel: Nat, -dl: Nat, -dr: Nat, -A: Array<F32>, -B: Array<F32>, -lo: Matrix.Order, -ro: Matrix.Order, -inner: Nat,
  -rows: Nat, -columns: Nat, -l1: Nat, -l2: Nat, -r1: Nat, -r2: Nat, -b1: Nat, -b2: Nat, -row: Nat, -column: Nat, -k: Nat,
  -lv: F32, -rv: F32, -values: List<&2, DotFold.Factors>> is Data:
  LogicalStep{-left_observed: {Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(operand_offset(lo, inner, rows, l1, l2, b1, b2, row, k)))) == Some{lv} : Maybe<&2, F32>},
    -right_observed: {Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(operand_offset(ro, inner, columns, r1, r2, b1, b2, column, k)))) == Some{rv} : Maybe<&2, F32>},
    -remaining: Logical(fuel, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, b1, b2, row, column, 1n+k, values)}

type LogicalCell<-dl: Nat, -dr: Nat, -A: Array<F32>, -B: Array<F32>, -lo: Matrix.Order, -ro: Matrix.Order, -inner: Nat, -rows: Nat,
  -columns: Nat, -l1: Nat, -l2: Nat, -r1: Nat, -r2: Nat, -b1: Nat, -b2: Nat, -row: Nat, -column: Nat, -value: Maybe<&2, F32>> is Data:
  LogicalCell{-word: F32, -observed: {value == Some{word} : Maybe<&2, F32>}, -values: List<&2, DotFold.Factors>,
    -logical: Logical(inner, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, b1, b2, row, column, 0n, values),
    -fold: {word == DotFold.dot_fold(values, 0.0) : F32}}

# For two balanced input arrays and every split, the slot of the build that
# holds cell (b1, b2, row, column), with row < rows, column < columns,
# b2 < o2 and the cell below the count, holds the fold from +0.0, in
# increasing k, of the `inner` factor pairs that `Logical` names: each
# operand read at its own batch, an axis of extent 1 at index 0. This is the
# meaning of `batched_build_cells` in the output's coordinates; a build
# whose reads of a broadcast batch are wrong breaks it. The forward product
# reads the left operand by rows and the right one by columns
# (`Batched.forward_layout`), the gradients as `left_layout` and
# `right_layout` say.
law batched_build_logical_cells:
  for +depth: Nat
  for +splits: Nat
  for +dl: Nat
  for +dr: Nat
  for -A: Array<F32>
  for -B: Array<F32>
  for +lo: Matrix.Order
  for +ro: Matrix.Order
  for +inner: Nat
  for +rows: Nat
  for +columns: Nat
  for +l1: Nat
  for +l2: Nat
  for +r1: Nat
  for +r2: Nat
  for +o2: Nat
  for +count: Nat
  for +base: Nat
  for la: Array<F32>
  for ra: Array<F32>
  for el: {la == A : Array<F32>}
  for er: {ra == B : Array<F32>}
  for left_valid: Model.Balanced(F32, dl, A)
  for right_valid: Model.Balanced(F32, dr, B)
  for +path: Model.Path(depth)
  for +b1: Nat
  for +b2: Nat
  for +row: Nat
  for +column: Nat
  for row_below: {Nat.is_lt(row, rows) == True{} : Bool}
  for column_below: {Nat.is_lt(column, columns) == True{} : Bool}
  for batch_below: {Nat.is_lt(b2, o2) == True{} : Bool}
  for at: {Nat.add(Model.offset(depth, path), base) == cell_position(o2, rows, columns, b1, b2, row, column) : Nat}
  for inside: {Nat.is_lt(cell_position(o2, rows, columns, b1, b2, row, column), count) == True{} : Bool}
  Holds<LogicalCell<dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, b1, b2, row, column,
    Model.observe(F32, depth, Batched.build(depth, splits, layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), count, base, la, ra), path)>>

# The plan of torch.matmul, written from PyTorch's definition over the
# operands' extents as naturals.
#
# One batch axis of two extents that broadcast: an extent of 1 takes the
# other one's.
def extent(+left: Nat, +right: Nat) -> Nat:
  Bool.pick(Nat, Nat.is_eq(left, 1n), right, left)

def vector(dims: List<&2, Nat>) -> Bool:
  match dims:
    case x <> Nil{}:
      True{}
    case rest:
      False{}

# A 1-D left operand [k] is the row [1, k], a 1-D right one the column
# [k, 1].
def promote_left(dims: List<&2, Nat>) -> List<&2, Nat>:
  match dims:
    case +x <> Nil{}:
      [1n, x]
    case rest:
      rest

def promote_right(dims: List<&2, Nat>) -> List<&2, Nat>:
  match dims:
    case +x <> Nil{}:
      [x, 1n]
    case rest:
      rest

# Every extent but the last two: the batch.
def batch_of(dims: List<&2, Nat>) -> List<&2, Nat>:
  match dims:
    case +x <> +y <> +z <> tail:
      x <> batch_of(y <> z <> tail)
    case rest:
      Nil{}

# The extent before the last: the rows of a matrix.
def rows_of(dims: List<&2, Nat>) -> Nat:
  match dims:
    case +x <> +y <> Nil{}:
      x
    case +x <> +y <> +z <> tail:
      rows_of(y <> z <> tail)
    case rest:
      0n

# The last extent: the columns of a matrix.
def columns_of(dims: List<&2, Nat>) -> Nat:
  match dims:
    case +x <> Nil{}:
      x
    case +x <> +y <> tail:
      columns_of(y <> tail)
    case Nil{}:
      0n

def zip_extents(left: List<&2, Nat>, right: List<&2, Nat>) -> List<&2, Nat>:
  match left right:
    case a <> lt b <> rt:
      extent(a, b) <> zip_extents(lt, rt)
    case lt rt:
      Nil{}

# Two batches broadcast right-aligned: the shorter one is padded with ones
# on the left (`Shape.pad`), then each axis broadcasts.
def broadcast_batch(+left: List<&2, Nat>, +right: List<&2, Nat>) -> List<&2, Nat>:
  zip_extents(Shape.pad(Nat.sub(List.length(&2, Nat, right), List.length(&2, Nat, left)), left),
    Shape.pad(Nat.sub(List.length(&2, Nat, left), List.length(&2, Nat, right)), right))

def kept(dropped: Bool, +extent: Nat) -> List<&2, Nat>:
  match dropped:
    case True{}:
      Nil{}
    case False{}:
      [extent]

# The output: the broadcast batch, then the rows of the promoted left
# operand unless it was a vector, then the columns of the promoted right
# operand unless it was a vector.
def torch_output(+left: List<&2, Nat>, +right: List<&2, Nat>) -> List<&2, Nat>:
  List.append(&2, Nat, broadcast_batch(batch_of(promote_left(left)), batch_of(promote_right(right))),
    List.append(&2, Nat, kept(vector(left), rows_of(promote_left(left))), kept(vector(right), columns_of(promote_right(right)))))

# Extents padded with ones on the left to `rank`.
def padded(+rank: Nat, +dims: List<&2, Nat>) -> List<&2, Nat>:
  Shape.pad(Nat.sub(rank, List.length(&2, Nat, dims)), dims)

# What a plan holds, read back as lists: the left operand as
# [l1, l2, rows, inner], the right one as [r1, r2, inner, columns], the
# output's batch as [o1, o2], and which operands are vectors.
def left_view(+plan: Batched.Plan) -> List<&2, Nat>:
  Batched.Plan{output, batch, +rows, +inner, columns, +l1, +l2, r1, r2, o1, o2, lv, rv} = plan
  [l1, l2, rows, inner]

def right_view(+plan: Batched.Plan) -> List<&2, Nat>:
  Batched.Plan{output, batch, rows, +inner, +columns, l1, l2, +r1, +r2, o1, o2, lv, rv} = plan
  [r1, r2, inner, columns]

def batch_view(+plan: Batched.Plan) -> List<&2, Nat>:
  Batched.Plan{output, batch, rows, inner, columns, l1, l2, r1, r2, +o1, +o2, lv, rv} = plan
  [o1, o2]

def vector_view(+plan: Batched.Plan) -> Bool & Bool:
  Batched.Plan{output, batch, rows, inner, columns, l1, l2, r1, r2, o1, o2, +lv, +rv} = plan
  (lv, rv)

type Geometry<-output: List<&2, Nat>, -left: List<&2, Nat>, -right: List<&2, Nat>, -batch: List<&2, Nat>, -vectors: Bool & Bool,
  -plan: Batched.Plan> is Data:
  Geometry{-output_dims: {Shape.naturals(Shape.dimensions(Batched.output_of(plan))) == output : List<&2, Nat>},
    -left_dims: {left_view(plan) == left : List<&2, Nat>},
    -right_dims: {right_view(plan) == right : List<&2, Nat>},
    -batch_dims: {batch_view(plan) == batch : List<&2, Nat>},
    -vector_dims: {vector_view(plan) == vectors : Bool & Bool}}

def GeometryOf(-output: List<&2, Nat>, -left: List<&2, Nat>, -right: List<&2, Nat>, -batch: List<&2, Nat>, -vectors: Bool & Bool,
  result: Result<&2, &2, Matrix.Error, Batched.Plan>) -> Data:
  match result:
    case Fail{error}:
      Unit
    case Done{plan}:
      Geometry<output, left, right, batch, vectors, plan>

def torch_geometry(+left: List<&2, Nat>, +right: List<&2, Nat>, result: Result<&2, &2, Matrix.Error, Batched.Plan>) -> Data:
  GeometryOf(torch_output(left, right), padded(4n, promote_left(left)), padded(4n, promote_right(right)),
    padded(2n, broadcast_batch(batch_of(promote_left(left)), batch_of(promote_right(right)))), (vector(left), vector(right)), result)

# Whatever the operands' dimensions, a plan that succeeds has PyTorch's
# output dimensions: a 1-D left operand promoted to a row and a 1-D right
# one to a column, both dropped again, and the batches broadcast
# right-aligned, an extent of 1 taking the other one's. It reads the left
# operand as its promoted extents padded with ones to rank 4, the right one
# likewise (their inner extents are equal), and the output's batch padded
# to two; these are the extents the build's layouts and
# `batched_build_logical_cells` read with.
law batched_plan_geometry:
  for +left: List<&2, U32>
  for +right: List<&2, U32>
  torch_geometry(Shape.naturals(left), Shape.naturals(right), Batched.plan_dims(left, right))
