import Base
import ../shape/core.bend as Shape
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ./core.bend as Core
import ./matmul.bend as Matrix
import ./batched-matmul.bend as Batched
import ./dot-fold.bend as DotFold

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
