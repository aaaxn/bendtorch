import Base
import ../shape/core.bend as Shape
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./matmul.bend as Matrix
import ./batched-matmul.bend as Batched
import ./batched-matmul-LAWS.bend as Laws
import ./dot-fold.bend as DotFold
import ./elementwise.bend as Elementwise
import ./elementwise-parallel-proof.bend as Parallel
import ./parallel.bend as Build
import ./parallel-proof.bend as Generic
import ./matrix-index-proof.bend as MatrixIndex
import ../shape/nat_proofs.bend as Nats
import ../shape/index_proofs.bend as Index
import ../storage/arith.bend as Arith

# Which call answers a pair of operands.

def Laws.batched_matmul_matrices(left, right, a, b, rows, inner, right_inner, columns, left_matrix, right_matrix):
  %Equal.sym(List<&2, U32>, Shape.dimensions(left), [rows, inner], left_matrix) :
    {Batched.routed(left, right, a, b, _, Shape.dimensions(right)) == Matrix.matmul(Core.Packed{left, a}, Core.Packed{right, b}) :
      Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}
  %Equal.sym(List<&2, U32>, Shape.dimensions(right), [right_inner, columns], right_matrix) :
    {Batched.routed(left, right, a, b, [rows, inner], _) == Matrix.matmul(Core.Packed{left, a}, Core.Packed{right, b}) :
      Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}
  {==}

def Laws.batched_matmul_left_scalar(left, right, a, b, scalar):
  %Equal.sym(List<&2, U32>, Shape.dimensions(left), [], scalar) :
    {Batched.routed(left, right, a, b, _, Shape.dimensions(right)) == Fail{Matrix.RankMismatch{}} :
      Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}
  {==}

# A rank-0 right operand: whatever the left dimensions, the pair is not two
# matrices, and the plan fails at one rank or the other.
def right_scalar_routed(+left: Shape.Shape, +right: Shape.Shape, a: Core.Tensor<left>, b: Core.Tensor<right>, +dims: List<&2, U32>) ->
  {Batched.routed(left, right, a, b, dims, []) == Fail{Matrix.RankMismatch{}} :
    Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}:
  match dims:
    case Nil{}:
      {==}
    case +k <> Nil{}:
      {==}
    case +m <> +k <> Nil{}:
      {==}
    case +x <> +m <> +k <> Nil{}:
      {==}
    case +x <> +y <> +m <> +k <> Nil{}:
      {==}
    case +x <> +y <> +z <> +t <> +u <> rest:
      {==}

def Laws.batched_matmul_right_scalar(left, right, a, b, scalar):
  %Equal.sym(List<&2, U32>, Shape.dimensions(right), [], scalar) :
    {Batched.routed(left, right, a, b, Shape.dimensions(left), _) == Fail{Matrix.RankMismatch{}} :
      Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}
  right_scalar_routed(left, right, a, b, Shape.dimensions(left))

def dot_compatible(+inner: U32, +right_inner: U32, mismatch: {U32.to_nat(inner) != U32.to_nat(right_inner) : Nat},
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))>) ->
  {Batched.compatible(Batched.Operand{Batched.Batch{0n, 1, 1}, 1, inner, True{}}, Batched.Operand{Batched.Batch{0n, 1, 1}, 1, right_inner, True{}},
    Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))) == Fail{Matrix.InnerDimensionMismatch{}} : Result<&2, &2, Matrix.Error, Batched.Plan>}:
  match decision:
    case Shape.BFalse{equal}:
      %equal : {Batched.compatible(Batched.Operand{Batched.Batch{0n, 1, 1}, 1, inner, True{}}, Batched.Operand{Batched.Batch{0n, 1, 1}, 1, right_inner, True{}},
        _) == Fail{Matrix.InnerDimensionMismatch{}} : Result<&2, &2, Matrix.Error, Batched.Plan>}
      {==}
    case Shape.BTrue{equal}:
      Empty.absurd({Batched.compatible(Batched.Operand{Batched.Batch{0n, 1, 1}, 1, inner, True{}}, Batched.Operand{Batched.Batch{0n, 1, 1}, 1, right_inner, True{}},
        Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))) == Fail{Matrix.InnerDimensionMismatch{}} : Result<&2, &2, Matrix.Error, Batched.Plan>},
        mismatch(Lifecycle.natural_equal(U32.to_nat(inner), U32.to_nat(right_inner), equal)))

def Laws.batched_matmul_dot_inner_error(left, right, a, b, inner, right_inner, left_vector, right_vector, mismatch):
  %Equal.sym(List<&2, U32>, Shape.dimensions(left), [inner], left_vector) :
    {Batched.routed(left, right, a, b, _, Shape.dimensions(right)) == Fail{Matrix.InnerDimensionMismatch{}} :
      Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}
  %Equal.sym(List<&2, U32>, Shape.dimensions(right), [right_inner], right_vector) :
    {Batched.routed(left, right, a, b, [inner], _) == Fail{Matrix.InnerDimensionMismatch{}} :
      Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}
  %Equal.sym(Result<&2, &2, Matrix.Error, Batched.Plan>,
    Batched.compatible(Batched.Operand{Batched.Batch{0n, 1, 1}, 1, inner, True{}}, Batched.Operand{Batched.Batch{0n, 1, 1}, 1, right_inner, True{}},
      Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))), Fail{Matrix.InnerDimensionMismatch{}},
    dot_compatible(inner, right_inner, mismatch, Shape.bool_case(Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))))) :
    {Batched.planned(left, right, a, b, _) == Fail{Matrix.InnerDimensionMismatch{}} :
      Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}
  {==}

# What every cell of the build holds.

# A run of `Matrix.product_run` from offsets lp and rp: its total folds the
# factor pairs it read, and both arrays come back unchanged.
type StridedRun<-fuel: Nat, -dl: Nat, -dr: Nat, -A: Array<F32>, -B: Array<F32>, -ls: Nat, -rs: Nat, -lp: Nat, -rp: Nat,
  -accumulator: F32, -result: F32 & (Array<F32> & Array<F32>)> is Data:
  StridedRun{-values: List<&2, DotFold.Factors>, -logical: Laws.Strided(fuel, dl, dr, A, B, ls, rs, lp, rp, values),
    -fold: {Pair.fst(F32, Array<F32> & Array<F32>, result) == DotFold.dot_fold(values, accumulator) : F32},
    -left_kept: {Pair.fst(Array<F32>, Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, result)) == A : Array<F32>},
    -right_kept: {Pair.snd(Array<F32>, Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, result)) == B : Array<F32>}}

def run_extend(+rest: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +ls: Nat, +rs: Nat, +lp: Nat, +rp: Nat,
  -accumulator: F32, -x: F32, -y: F32,
  -ex: {Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(lp))) == Some{x} : Maybe<&2, F32>},
  -ey: {Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(rp))) == Some{y} : Maybe<&2, F32>},
  -result: F32 & (Array<F32> & Array<F32>),
  -run: Parallel.Fact<StridedRun<rest, dl, dr, A, B, ls, rs, Nat.add(ls, lp), Nat.add(rs, rp), F32.add(accumulator, F32.mul(x, y)), result>>) ->
  Parallel.Fact<StridedRun<1n+rest, dl, dr, A, B, ls, rs, lp, rp, accumulator, result>>:
  Parallel.Fact{{\{StridedRun: values => logical => fold => left_kept => right_kept =>
    StridedRun{DotFold.Factors{x, y} <> values, Laws.StridedStep{ex, ey, logical}, fold, left_kept, right_kept}} :
      StridedRun<rest, dl, dr, A, B, ls, rs, Nat.add(ls, lp), Nat.add(rs, rp), F32.add(accumulator, F32.mul(x, y)), result> ->
      StridedRun<1n+rest, dl, dr, A, B, ls, rs, lp, rp, accumulator, result>}(
    {\{Parallel.Fact: p => p} : Parallel.Fact<StridedRun<rest, dl, dr, A, B, ls, rs, Nat.add(ls, lp), Nat.add(rs, rp), F32.add(accumulator, F32.mul(x, y)), result>> ->
      StridedRun<rest, dl, dr, A, B, ls, rs, Nat.add(ls, lp), Nat.add(rs, rp), F32.add(accumulator, F32.mul(x, y)), result>}(run))}

# The reads at lp and rp hold the words of A and B, and keep both arrays.
type Read<-dl: Nat, -dr: Nat, -A: Array<F32>, -B: Array<F32>, -lp: Nat, -rp: Nat, -pl: Array<F32> & F32, -pr: Array<F32> & F32> is Data:
  Read{-ex: {Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(lp))) == Some{Refinement.second(F32, pl)} : Maybe<&2, F32>},
    -ey: {Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(rp))) == Some{Refinement.second(F32, pr)} : Maybe<&2, F32>},
    -kl: {Refinement.first(F32, pl) == A : Array<F32>}, -kr: {Refinement.first(F32, pr) == B : Array<F32>}}

def reads(+dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +lp: Nat, +rp: Nat,
  -la: Array<F32>, -ra: Array<F32>, -el: {la == A : Array<F32>}, -er: {ra == B : Array<F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B),
  -pl: Array<F32> & F32, -exact_l: {pl == Array.get(F32, la, U32.from_nat(lp)) : Array<F32> & F32},
  -pr: Array<F32> & F32, -exact_r: {pr == Array.get(F32, ra, U32.from_nat(rp)) : Array<F32> & F32}) ->
  Parallel.Fact<Read<dl, dr, A, B, lp, rp, pl, pr>>:
  Parallel.Fact{Read{
    Equal.sym(Maybe<&2, F32>, Some{Refinement.second(F32, pl)}, Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(lp))),
      {\{Parallel.Fact: p => p} : Parallel.Fact<({Some{Refinement.second(F32, pl)} == Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(lp))) : Maybe<&2, F32>})> ->
        {Some{Refinement.second(F32, pl)} == Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(lp))) : Maybe<&2, F32>}}(
        Parallel.read_value(dl, la, A, U32.from_nat(lp), pl, exact_l, el, left_valid))),
    Equal.sym(Maybe<&2, F32>, Some{Refinement.second(F32, pr)}, Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(rp))),
      {\{Parallel.Fact: p => p} : Parallel.Fact<({Some{Refinement.second(F32, pr)} == Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(rp))) : Maybe<&2, F32>})> ->
        {Some{Refinement.second(F32, pr)} == Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(rp))) : Maybe<&2, F32>}}(
        Parallel.read_value(dr, ra, B, U32.from_nat(rp), pr, exact_r, er, right_valid))),
    {\{Parallel.Fact: p => p} : Parallel.Fact<({Refinement.first(F32, pl) == A : Array<F32>})> -> {Refinement.first(F32, pl) == A : Array<F32>}}(
      Parallel.read_keeps(dl, la, A, U32.from_nat(lp), pl, exact_l, el, left_valid)),
    {\{Parallel.Fact: p => p} : Parallel.Fact<({Refinement.first(F32, pr) == B : Array<F32>})> -> {Refinement.first(F32, pr) == B : Array<F32>}}(
      Parallel.read_keeps(dr, ra, B, U32.from_nat(rp), pr, exact_r, er, right_valid))}}

def run_last(+dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +ls: Nat, +rs: Nat, +lp: Nat, +rp: Nat, -accumulator: F32,
  pl: Array<F32> & F32, pr: Array<F32> & F32, -read: Read<dl, dr, A, B, lp, rp, pl, pr>) ->
  Parallel.Fact<StridedRun<1n, dl, dr, A, B, ls, rs, lp, rp, accumulator, Matrix.product_run(0n, ls, rs, lp, rp, accumulator, pl, pr)>>:
  (kl, x) = pl
  (kr, y) = pr
  -ex = {\{Read: ex => ey => left_kept => right_kept => ex} : Read<dl, dr, A, B, lp, rp, (kl, x), (kr, y)> ->
    {Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(lp))) == Some{x} : Maybe<&2, F32>}}(read)
  -ey = {\{Read: ex => ey => left_kept => right_kept => ey} : Read<dl, dr, A, B, lp, rp, (kl, x), (kr, y)> ->
    {Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(rp))) == Some{y} : Maybe<&2, F32>}}(read)
  -left_kept = {\{Read: ex => ey => left_kept => right_kept => left_kept} : Read<dl, dr, A, B, lp, rp, (kl, x), (kr, y)> ->
    {kl == A : Array<F32>}}(read)
  -right_kept = {\{Read: ex => ey => left_kept => right_kept => right_kept} : Read<dl, dr, A, B, lp, rp, (kl, x), (kr, y)> ->
    {kr == B : Array<F32>}}(read)
  run_extend(0n, dl, dr, A, B, ls, rs, lp, rp, accumulator, x, y, ex, ey, (F32.add(accumulator, F32.mul(x, y)), (kl, kr)),
    Parallel.Fact{StridedRun{Nil{}, Unit{}, {==}, left_kept, right_kept}})

def run_next(+rest: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +ls: Nat, +rs: Nat, +lp: Nat, +rp: Nat, -accumulator: F32,
  pl: Array<F32> & F32, pr: Array<F32> & F32, -read: Read<dl, dr, A, B, lp, rp, pl, pr>,
  -next: @value: F32 -> @-l: Array<F32> -> @-r: Array<F32> -> @-el: {l == A : Array<F32>} -> @-er: {r == B : Array<F32>} ->
    Parallel.Fact<StridedRun<1n+rest, dl, dr, A, B, ls, rs, Nat.add(ls, lp), Nat.add(rs, rp), value,
      Matrix.product_run(rest, ls, rs, Nat.add(ls, lp), Nat.add(rs, rp), value,
        Array.get(F32, l, U32.from_nat(Nat.add(ls, lp))), Array.get(F32, r, U32.from_nat(Nat.add(rs, rp))))>>) ->
  Parallel.Fact<StridedRun<2n+rest, dl, dr, A, B, ls, rs, lp, rp, accumulator, Matrix.product_run(1n+rest, ls, rs, lp, rp, accumulator, pl, pr)>>:
  (kl, x) = pl
  (kr, y) = pr
  -ex = {\{Read: ex => ey => left_kept => right_kept => ex} : Read<dl, dr, A, B, lp, rp, (kl, x), (kr, y)> ->
    {Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(lp))) == Some{x} : Maybe<&2, F32>}}(read)
  -ey = {\{Read: ex => ey => left_kept => right_kept => ey} : Read<dl, dr, A, B, lp, rp, (kl, x), (kr, y)> ->
    {Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(rp))) == Some{y} : Maybe<&2, F32>}}(read)
  -left_kept = {\{Read: ex => ey => left_kept => right_kept => left_kept} : Read<dl, dr, A, B, lp, rp, (kl, x), (kr, y)> ->
    {kl == A : Array<F32>}}(read)
  -right_kept = {\{Read: ex => ey => left_kept => right_kept => right_kept} : Read<dl, dr, A, B, lp, rp, (kl, x), (kr, y)> ->
    {kr == B : Array<F32>}}(read)
  run_extend(1n+rest, dl, dr, A, B, ls, rs, lp, rp, accumulator, x, y, ex, ey,
    Matrix.product_run(rest, ls, rs, Nat.add(ls, lp), Nat.add(rs, rp), F32.add(accumulator, F32.mul(x, y)),
      Array.get(F32, kl, U32.from_nat(Nat.add(ls, lp))), Array.get(F32, kr, U32.from_nat(Nat.add(rs, rp)))),
    next(F32.add(accumulator, F32.mul(x, y)))(kl)(kr)(left_kept)(right_kept))

# A run of fuel + 1 terms from lp and rp folds the factor pairs it read and
# hands both arrays back.
def run_eq(+fuel: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +ls: Nat, +rs: Nat, +lp: Nat, +rp: Nat, -accumulator: F32,
  -la: Array<F32>, -ra: Array<F32>, -el: {la == A : Array<F32>}, -er: {ra == B : Array<F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B),
  pl: Array<F32> & F32, -exact_l: {pl == Array.get(F32, la, U32.from_nat(lp)) : Array<F32> & F32},
  pr: Array<F32> & F32, -exact_r: {pr == Array.get(F32, ra, U32.from_nat(rp)) : Array<F32> & F32}) ->
  Parallel.Fact<StridedRun<1n+fuel, dl, dr, A, B, ls, rs, lp, rp, accumulator, Matrix.product_run(fuel, ls, rs, lp, rp, accumulator, pl, pr)>>:
  match fuel:
    case 0n:
      run_last(dl, dr, A, B, ls, rs, lp, rp, accumulator, pl, pr,
        {\{Parallel.Fact: p => p} : Parallel.Fact<Read<dl, dr, A, B, lp, rp, pl, pr>> -> Read<dl, dr, A, B, lp, rp, pl, pr>}(
          reads(dl, dr, A, B, lp, rp, la, ra, el, er, left_valid, right_valid, pl, exact_l, pr, exact_r)))
    case 1n+rest:
      run_next(rest, dl, dr, A, B, ls, rs, lp, rp, accumulator, pl, pr,
        {\{Parallel.Fact: p => p} : Parallel.Fact<Read<dl, dr, A, B, lp, rp, pl, pr>> -> Read<dl, dr, A, B, lp, rp, pl, pr>}(
          reads(dl, dr, A, B, lp, rp, la, ra, el, er, left_valid, right_valid, pl, exact_l, pr, exact_r)),
        value => l => r => el => er => run_eq(rest, dl, dr, A, B, ls, rs, Nat.add(ls, lp), Nat.add(rs, rp), value, l, r, el, er,
          left_valid, right_valid, Array.get(F32, l, U32.from_nat(Nat.add(ls, lp))), {==}, Array.get(F32, r, U32.from_nat(Nat.add(rs, rp))), {==}))

# `Batched.run_from` of `fuel` terms from +0.0.
def from_run(fuel: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +ls: Nat, +rs: Nat, +lp: Nat, +rp: Nat,
  la: Array<F32>, ra: Array<F32>, -el: {la == A : Array<F32>}, -er: {ra == B : Array<F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Parallel.Fact<StridedRun<fuel, dl, dr, A, B, ls, rs, lp, rp, 0.0, Batched.run_from(fuel, ls, rs, lp, rp, la, ra)>>:
  match fuel:
    case 0n:
      Parallel.Fact{StridedRun{Nil{}, Unit{}, {==}, el, er}}
    case 1n+rest:
      run_eq(rest, dl, dr, A, B, ls, rs, lp, rp, 0.0, la, ra, el, er, left_valid, right_valid,
        Array.get(F32, la, U32.from_nat(lp)), {==}, Array.get(F32, ra, U32.from_nat(rp)), {==})

# The batched cell function of `Parallel.build`, from the input arrays.
def cell_of(+lo: Matrix.Order, +ro: Matrix.Order, +inner: Nat, +rows: Nat, +columns: Nat, +l1: Nat, +l2: Nat, +r1: Nat, +r2: Nat, +o2: Nat,
  +position: Nat, A: Array<F32>, B: Array<F32>, inside: Bool) -> F32 & (Array<F32> & Array<F32>):
  Batched.cell_at(Batched.layout(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2)), position, (A, B), inside)

def dot_run(+lo: Matrix.Order, +ro: Matrix.Order, +inner: Nat, +rows: Nat, +columns: Nat, +l1: Nat, +l2: Nat, +r1: Nat, +r2: Nat, +o2: Nat,
  +position: Nat, +dl: Nat, +dr: Nat, A: Array<F32>, B: Array<F32>,
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Parallel.Fact<StridedRun<inner, dl, dr, A, B, Matrix.stride(lo, rows), Matrix.stride(ro, columns),
    Batched.left_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position),
    Batched.right_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position), 0.0,
    cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, A, B, True{})>>:
  from_run(inner, dl, dr, A, B, Matrix.stride(lo, rows), Matrix.stride(ro, columns),
    Batched.left_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position),
    Batched.right_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position), A, B, {==}, {==}, left_valid, right_valid)

def eta(-X: Type, -Y: Type, pair: X & Y) -> {pair == (Pair.fst(X, Y, pair), Pair.snd(X, Y, pair)) : X & Y}:
  (x, y) = pair
  {==}

# Every cell hands both input arrays back.
def cell_keeps(+lo: Matrix.Order, +ro: Matrix.Order, +inner: Nat, +rows: Nat, +columns: Nat, +l1: Nat, +l2: Nat, +r1: Nat, +r2: Nat, +o2: Nat,
  +position: Nat, +dl: Nat, +dr: Nat, A: Array<F32>, B: Array<F32>, inside: Bool,
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Parallel.Fact<{Pair.snd(F32, Array<F32> & Array<F32>, cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, A, B, inside)) == (A, B) : Array<F32> & Array<F32>}>:
  match inside:
    case False{}:
      Parallel.Fact{{==}}
    case True{}:
      Parallel.Fact{{\{StridedRun: values => logical => fold => left_kept => right_kept =>
        Equal.trans(Array<F32> & Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, A, B, True{})),
          (Pair.fst(Array<F32>, Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, A, B, True{}))),
            Pair.snd(Array<F32>, Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, A, B, True{})))),
          (A, B),
          eta(Array<F32>, Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, A, B, True{}))),
          Parallel.pair_equal(Pair.fst(Array<F32>, Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, A, B, True{}))),
            Pair.snd(Array<F32>, Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, A, B, True{}))),
            A, B, left_kept, right_kept))} :
        StridedRun<inner, dl, dr, A, B, Matrix.stride(lo, rows), Matrix.stride(ro, columns),
          Batched.left_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position),
          Batched.right_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position), 0.0,
          cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, A, B, True{})> ->
        {Pair.snd(F32, Array<F32> & Array<F32>, cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, A, B, True{})) == (A, B) : Array<F32> & Array<F32>}}(
        {\{Parallel.Fact: p => p} :
          Parallel.Fact<StridedRun<inner, dl, dr, A, B, Matrix.stride(lo, rows), Matrix.stride(ro, columns),
            Batched.left_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position),
            Batched.right_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position), 0.0,
            cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, A, B, True{})>> ->
          StridedRun<inner, dl, dr, A, B, Matrix.stride(lo, rows), Matrix.stride(ro, columns),
            Batched.left_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position),
            Batched.right_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position), 0.0,
            cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, A, B, True{})>}(
          dot_run(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, dl, dr, A, B, left_valid, right_valid)))}

# What the cell function computes is what CellSpec names.
def cell_spec(+lo: Matrix.Order, +ro: Matrix.Order, +inner: Nat, +rows: Nat, +columns: Nat, +l1: Nat, +l2: Nat, +r1: Nat, +r2: Nat, +o2: Nat,
  +position: Nat, +dl: Nat, +dr: Nat, A: Array<F32>, B: Array<F32>, inside: Bool, -observed: Maybe<&2, F32>,
  -computed: {observed == Some{Pair.fst(F32, Array<F32> & Array<F32>, cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, A, B, inside))} : Maybe<&2, F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Parallel.Fact<Laws.CellSpec(dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, inside, observed)>:
  match inside:
    case False{}:
      Parallel.Fact{computed}
    case True{}:
      Parallel.Fact{{\{StridedRun: values => logical => fold => left_kept => right_kept =>
        Laws.CellAt{Pair.fst(F32, Array<F32> & Array<F32>, cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, A, B, True{})),
          computed, values, logical, fold}} :
        StridedRun<inner, dl, dr, A, B, Matrix.stride(lo, rows), Matrix.stride(ro, columns),
          Batched.left_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position),
          Batched.right_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position), 0.0,
          cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, A, B, True{})> ->
        Laws.CellAt<dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, observed>}(
        {\{Parallel.Fact: p => p} :
          Parallel.Fact<StridedRun<inner, dl, dr, A, B, Matrix.stride(lo, rows), Matrix.stride(ro, columns),
            Batched.left_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position),
            Batched.right_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position), 0.0,
            cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, A, B, True{})>> ->
          StridedRun<inner, dl, dr, A, B, Matrix.stride(lo, rows), Matrix.stride(ro, columns),
            Batched.left_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position),
            Batched.right_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), position), 0.0,
            cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, A, B, True{})>}(
          dot_run(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, dl, dr, A, B, left_valid, right_valid)))}

def Laws.batched_build_cells(depth, splits, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, count, base, la, ra, el, er,
  left_valid, right_valid):
  -split = {\{Parallel.Fact: q => q} : Parallel.Fact<({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>} &
    {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>})> ->
    {Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>} &
    {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>}}(
    Parallel.split_equal(dl, dr, A, B, left_valid, right_valid))
  -cells = {\{Generic.Fact: q => q} :
    Generic.Fact<Generic.Cells(~Batched.Layout, ~Array<F32>, ~Array<F32>, ~Batched.cell_at, Batched.layout(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2)),
      (A, B), depth, count, base, Batched.build(depth, splits, Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), count, base, la, ra))> ->
    Generic.Cells(~Batched.Layout, ~Array<F32>, ~Array<F32>, ~Batched.cell_at, Batched.layout(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2)),
      (A, B), depth, count, base, Batched.build(depth, splits, Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), count, base, la, ra))}(
    Generic.build_cells(~Batched.Layout, ~Batched.Layout, ~Array<F32>, ~Array<F32>, ~Batched.layout, ~Elementwise.split, ~Batched.cell_at,
      Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), (A, B), depth, splits, count, base, (la, ra), Parallel.pair_equal(la, ra, A, B, el, er),
      at => inside => {\{Parallel.Fact: q => q} :
        Parallel.Fact<({Pair.snd(F32, Array<F32> & Array<F32>, cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, at, A, B, inside)) == (A, B) : Array<F32> & Array<F32>})> ->
        {Pair.snd(F32, Array<F32> & Array<F32>, cell_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, at, A, B, inside)) == (A, B) : Array<F32> & Array<F32>}}(
        cell_keeps(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, at, dl, dr, A, B, inside, left_valid, right_valid)),
      Pair.fst({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>},
        {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>}, split),
      Pair.snd({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>},
        {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>}, split)))
  Laws.Holds{path => {\{Parallel.Fact: q => q} :
    Parallel.Fact<Laws.CellSpec(dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, Nat.add(Model.offset(depth, path), base),
      Nat.is_lt(Nat.add(Model.offset(depth, path), base), count),
      Model.observe(F32, depth, Batched.build(depth, splits, Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), count, base, la, ra), path))> ->
    Laws.CellSpec(dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, Nat.add(Model.offset(depth, path), base),
      Nat.is_lt(Nat.add(Model.offset(depth, path), base), count),
      Model.observe(F32, depth, Batched.build(depth, splits, Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), count, base, la, ra), path))}(
    cell_spec(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, Nat.add(Model.offset(depth, path), base), dl, dr, A, B,
      Nat.is_lt(Nat.add(Model.offset(depth, path), base), count),
      Model.observe(F32, depth, Batched.build(depth, splits, Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), count, base, la, ra), path),
      cells(path), left_valid, right_valid))}

def Laws.batched_build_balanced(depth, splits, given, count, base, la, ra):
  Batched.build_balanced(depth, splits, given, count, base, la, ra)

# The cells of the build in the output's coordinates.

# Division and remainder of q * b + r by b, for r below b.
def div_exact(+q: Nat, +b: Nat, +r: Nat, below: {Nat.is_lt(r, b) == True{} : Bool}) -> {Nat.div(Nat.add(Nat.mul(q, b), r), b) == q : Nat}:
  Equal.cong(Nat & Nat, Nat, pair => Pair.fst(Nat, Nat, pair), Nat.divmod(Nat.add(Nat.mul(q, b), r), b), (q, r),
    Equal.sym(Nat & Nat, (q, r), Nat.divmod(Nat.add(Nat.mul(q, b), r), b), Nats.divmod_exact(q, b, r, MatrixIndex.lt_from(r, b, below))))

def mod_exact(+q: Nat, +b: Nat, +r: Nat, below: {Nat.is_lt(r, b) == True{} : Bool}) -> {Nat.mod(Nat.add(Nat.mul(q, b), r), b) == r : Nat}:
  Equal.cong(Nat & Nat, Nat, pair => Pair.snd(Nat, Nat, pair), Nat.divmod(Nat.add(Nat.mul(q, b), r), b), (q, r),
    Equal.sym(Nat & Nat, (q, r), Nat.divmod(Nat.add(Nat.mul(q, b), r), b), Nats.divmod_exact(q, b, r, MatrixIndex.lt_from(r, b, below))))

# The batch number of the build reads each operand's batch as the shape
# module broadcasts it.
# An index of a batch axis: the stretch of the build, the shape module's
# broadcast index.
def stretch_spec(+extent: Nat, +index: Nat) -> {Batched.stretch(index, Nat.is_eq(extent, 1n)) == Shape.broadcast_index(extent, index) : Nat}:
  match extent:
    case 0n:
      {==}
    case 1n+e:
      match e:
        case 0n:
          {==}
        case 1n+f:
          {==}

# The batch number of the build reads each operand's batch as the shape
# module broadcasts it.
def batch_number_spec(+first: Nat, +second: Nat, +b1: Nat, +b2: Nat) ->
  {Batched.batch_number(first, second, b1, b2) == Nat.add(Nat.mul(Shape.broadcast_index(first, b1), second), Shape.broadcast_index(second, b2)) : Nat}:
  %Equal.sym(Nat, Batched.stretch(b1, Nat.is_eq(first, 1n)), Shape.broadcast_index(first, b1), stretch_spec(first, b1)) :
    {Nat.add(Nat.mul(_, second), Batched.stretch(b2, Nat.is_eq(second, 1n))) ==
      Nat.add(Nat.mul(Shape.broadcast_index(first, b1), second), Shape.broadcast_index(second, b2)) : Nat}
  %Equal.sym(Nat, Batched.stretch(b2, Nat.is_eq(second, 1n)), Shape.broadcast_index(second, b2), stretch_spec(second, b2)) :
    {Nat.add(Nat.mul(Shape.broadcast_index(first, b1), second), _) ==
      Nat.add(Nat.mul(Shape.broadcast_index(first, b1), second), Shape.broadcast_index(second, b2)) : Nat}
  {==}

def read_offset_spec(+order: Matrix.Order, +inner: Nat, +outers: Nat, +outer: Nat, +k: Nat) ->
  {Matrix.read_offset(order, inner, outers, outer, k) == Laws.matrix_offset(order, inner, outers, outer, k) : Nat}:
  match order:
    case Matrix.Kept{}:
      {==}
    case Matrix.Swapped{}:
      {==}

# The first left read of a cell, from its batch number `batch` and its
# offset `within` inside its matrix.
def left_from(+lo: Matrix.Order, +inner: Nat, +rows: Nat, +columns: Nat, +l1: Nat, +l2: Nat, +o2: Nat, +batch: Nat, +within: Nat) -> Nat:
  Nat.add(Nat.mul(Batched.batch_number(l1, l2, Nat.div(batch, o2), Nat.mod(batch, o2)), Nat.mul(rows, inner)),
    Matrix.read_offset(lo, inner, rows, Nat.div(within, columns), 0n))

def right_from(+ro: Matrix.Order, +inner: Nat, +rows: Nat, +columns: Nat, +r1: Nat, +r2: Nat, +o2: Nat, +batch: Nat, +within: Nat) -> Nat:
  Nat.add(Nat.mul(Batched.batch_number(r1, r2, Nat.div(batch, o2), Nat.mod(batch, o2)), Nat.mul(inner, columns)),
    Matrix.read_offset(ro, inner, columns, Nat.mod(within, columns), 0n))

def left_start_spec(+lo: Matrix.Order, +ro: Matrix.Order, +inner: Nat, +rows: Nat, +columns: Nat, +l1: Nat, +l2: Nat, +r1: Nat, +r2: Nat, +o2: Nat,
  +b1: Nat, +b2: Nat, +row: Nat, +column: Nat,
  +row_below: {Nat.is_lt(row, rows) == True{} : Bool}, +column_below: {Nat.is_lt(column, columns) == True{} : Bool},
  +batch_below: {Nat.is_lt(b2, o2) == True{} : Bool}) ->
  {Batched.left_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), Laws.cell_position(o2, rows, columns, b1, b2, row, column)) ==
    Laws.operand_offset(lo, inner, rows, l1, l2, b1, b2, row, 0n) : Nat}:
  +q = Nat.add(Nat.mul(b1, o2), b2)
  +r = Nat.add(Nat.mul(row, columns), column)
  +rc = Nat.mul(rows, columns)
  +within = MatrixIndex.lt_mul_add(row, rows, column, columns, row_below, column_below)
  %Equal.sym(Nat, Nat.div(Nat.add(Nat.mul(q, rc), r), rc), q, div_exact(q, rc, r, within)) :
    {left_from(lo, inner, rows, columns, l1, l2, o2, _, Nat.mod(Nat.add(Nat.mul(q, rc), r), rc)) == Laws.operand_offset(lo, inner, rows, l1, l2, b1, b2, row, 0n) : Nat}
  %Equal.sym(Nat, Nat.mod(Nat.add(Nat.mul(q, rc), r), rc), r, mod_exact(q, rc, r, within)) :
    {left_from(lo, inner, rows, columns, l1, l2, o2, q, _) == Laws.operand_offset(lo, inner, rows, l1, l2, b1, b2, row, 0n) : Nat}
  %Equal.sym(Nat, Nat.div(q, o2), b1, div_exact(b1, o2, b2, batch_below)) :
    {Nat.add(Nat.mul(Batched.batch_number(l1, l2, _, Nat.mod(q, o2)), Nat.mul(rows, inner)), Matrix.read_offset(lo, inner, rows, Nat.div(r, columns), 0n)) ==
      Laws.operand_offset(lo, inner, rows, l1, l2, b1, b2, row, 0n) : Nat}
  %Equal.sym(Nat, Nat.mod(q, o2), b2, mod_exact(b1, o2, b2, batch_below)) :
    {Nat.add(Nat.mul(Batched.batch_number(l1, l2, b1, _), Nat.mul(rows, inner)), Matrix.read_offset(lo, inner, rows, Nat.div(r, columns), 0n)) ==
      Laws.operand_offset(lo, inner, rows, l1, l2, b1, b2, row, 0n) : Nat}
  %Equal.sym(Nat, Nat.div(r, columns), row, div_exact(row, columns, column, column_below)) :
    {Nat.add(Nat.mul(Batched.batch_number(l1, l2, b1, b2), Nat.mul(rows, inner)), Matrix.read_offset(lo, inner, rows, _, 0n)) ==
      Laws.operand_offset(lo, inner, rows, l1, l2, b1, b2, row, 0n) : Nat}
  %Equal.sym(Nat, Batched.batch_number(l1, l2, b1, b2), Nat.add(Nat.mul(Shape.broadcast_index(l1, b1), l2), Shape.broadcast_index(l2, b2)),
    batch_number_spec(l1, l2, b1, b2)) :
    {Nat.add(Nat.mul(_, Nat.mul(rows, inner)), Matrix.read_offset(lo, inner, rows, row, 0n)) ==
      Laws.operand_offset(lo, inner, rows, l1, l2, b1, b2, row, 0n) : Nat}
  %Equal.sym(Nat, Matrix.read_offset(lo, inner, rows, row, 0n), Laws.matrix_offset(lo, inner, rows, row, 0n), read_offset_spec(lo, inner, rows, row, 0n)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(Shape.broadcast_index(l1, b1), l2), Shape.broadcast_index(l2, b2)), Nat.mul(rows, inner)), _) ==
      Laws.operand_offset(lo, inner, rows, l1, l2, b1, b2, row, 0n) : Nat}
  {==}

def right_start_spec(+lo: Matrix.Order, +ro: Matrix.Order, +inner: Nat, +rows: Nat, +columns: Nat, +l1: Nat, +l2: Nat, +r1: Nat, +r2: Nat, +o2: Nat,
  +b1: Nat, +b2: Nat, +row: Nat, +column: Nat,
  +row_below: {Nat.is_lt(row, rows) == True{} : Bool}, +column_below: {Nat.is_lt(column, columns) == True{} : Bool},
  +batch_below: {Nat.is_lt(b2, o2) == True{} : Bool}) ->
  {Batched.right_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), Laws.cell_position(o2, rows, columns, b1, b2, row, column)) ==
    Laws.operand_offset(ro, inner, columns, r1, r2, b1, b2, column, 0n) : Nat}:
  +q = Nat.add(Nat.mul(b1, o2), b2)
  +r = Nat.add(Nat.mul(row, columns), column)
  +rc = Nat.mul(rows, columns)
  +within = MatrixIndex.lt_mul_add(row, rows, column, columns, row_below, column_below)
  %Equal.sym(Nat, Nat.div(Nat.add(Nat.mul(q, rc), r), rc), q, div_exact(q, rc, r, within)) :
    {right_from(ro, inner, rows, columns, r1, r2, o2, _, Nat.mod(Nat.add(Nat.mul(q, rc), r), rc)) == Laws.operand_offset(ro, inner, columns, r1, r2, b1, b2, column, 0n) : Nat}
  %Equal.sym(Nat, Nat.mod(Nat.add(Nat.mul(q, rc), r), rc), r, mod_exact(q, rc, r, within)) :
    {right_from(ro, inner, rows, columns, r1, r2, o2, q, _) == Laws.operand_offset(ro, inner, columns, r1, r2, b1, b2, column, 0n) : Nat}
  %Equal.sym(Nat, Nat.div(q, o2), b1, div_exact(b1, o2, b2, batch_below)) :
    {Nat.add(Nat.mul(Batched.batch_number(r1, r2, _, Nat.mod(q, o2)), Nat.mul(inner, columns)), Matrix.read_offset(ro, inner, columns, Nat.mod(r, columns), 0n)) ==
      Laws.operand_offset(ro, inner, columns, r1, r2, b1, b2, column, 0n) : Nat}
  %Equal.sym(Nat, Nat.mod(q, o2), b2, mod_exact(b1, o2, b2, batch_below)) :
    {Nat.add(Nat.mul(Batched.batch_number(r1, r2, b1, _), Nat.mul(inner, columns)), Matrix.read_offset(ro, inner, columns, Nat.mod(r, columns), 0n)) ==
      Laws.operand_offset(ro, inner, columns, r1, r2, b1, b2, column, 0n) : Nat}
  %Equal.sym(Nat, Nat.mod(r, columns), column, mod_exact(row, columns, column, column_below)) :
    {Nat.add(Nat.mul(Batched.batch_number(r1, r2, b1, b2), Nat.mul(inner, columns)), Matrix.read_offset(ro, inner, columns, _, 0n)) ==
      Laws.operand_offset(ro, inner, columns, r1, r2, b1, b2, column, 0n) : Nat}
  %Equal.sym(Nat, Batched.batch_number(r1, r2, b1, b2), Nat.add(Nat.mul(Shape.broadcast_index(r1, b1), r2), Shape.broadcast_index(r2, b2)),
    batch_number_spec(r1, r2, b1, b2)) :
    {Nat.add(Nat.mul(_, Nat.mul(inner, columns)), Matrix.read_offset(ro, inner, columns, column, 0n)) ==
      Laws.operand_offset(ro, inner, columns, r1, r2, b1, b2, column, 0n) : Nat}
  %Equal.sym(Nat, Matrix.read_offset(ro, inner, columns, column, 0n), Laws.matrix_offset(ro, inner, columns, column, 0n), read_offset_spec(ro, inner, columns, column, 0n)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(Shape.broadcast_index(r1, b1), r2), Shape.broadcast_index(r2, b2)), Nat.mul(inner, columns)), _) ==
      Laws.operand_offset(ro, inner, columns, r1, r2, b1, b2, column, 0n) : Nat}
  %Equal.sym(Nat, Nat.mul(inner, columns), Nat.mul(columns, inner), Nats.mul_comm(inner, columns)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(Shape.broadcast_index(r1, b1), r2), Shape.broadcast_index(r2, b2)), _), Laws.matrix_offset(ro, inner, columns, column, 0n)) ==
      Laws.operand_offset(ro, inner, columns, r1, r2, b1, b2, column, 0n) : Nat}
  {==}

def batch_base(+inner: Nat, +outers: Nat, +e1: Nat, +e2: Nat, +b1: Nat, +b2: Nat) -> Nat:
  Nat.mul(Nat.add(Nat.mul(Shape.broadcast_index(e1, b1), e2), Shape.broadcast_index(e2, b2)), Nat.mul(outers, inner))

# Term k + 1 of an operand sits one stride after term k.
def operand_step(+order: Matrix.Order, +inner: Nat, +outers: Nat, +e1: Nat, +e2: Nat, +b1: Nat, +b2: Nat, +outer: Nat, +k: Nat) ->
  {Nat.add(Matrix.stride(order, outers), Laws.operand_offset(order, inner, outers, e1, e2, b1, b2, outer, k)) ==
    Laws.operand_offset(order, inner, outers, e1, e2, b1, b2, outer, 1n+k) : Nat}:
  match order:
    case Matrix.Kept{}:
      %Equal.sym(Nat, Nat.add(Nat.mul(outer, inner), 1n+k), 1n+Nat.add(Nat.mul(outer, inner), k), Arith.add_succ(Nat.mul(outer, inner), k)) :
        {1n+Nat.add(batch_base(inner, outers, e1, e2, b1, b2), Nat.add(Nat.mul(outer, inner), k)) == Nat.add(batch_base(inner, outers, e1, e2, b1, b2), _) : Nat}
      %Equal.sym(Nat, Nat.add(batch_base(inner, outers, e1, e2, b1, b2), 1n+Nat.add(Nat.mul(outer, inner), k)),
        1n+Nat.add(batch_base(inner, outers, e1, e2, b1, b2), Nat.add(Nat.mul(outer, inner), k)),
        Arith.add_succ(batch_base(inner, outers, e1, e2, b1, b2), Nat.add(Nat.mul(outer, inner), k))) :
        {1n+Nat.add(batch_base(inner, outers, e1, e2, b1, b2), Nat.add(Nat.mul(outer, inner), k)) == _ : Nat}
      {==}
    case Matrix.Swapped{}:
      %Index.add_assoc(outers, Nat.mul(k, outers), outer) :
        {Nat.add(outers, Nat.add(batch_base(inner, outers, e1, e2, b1, b2), Nat.add(Nat.mul(k, outers), outer))) == Nat.add(batch_base(inner, outers, e1, e2, b1, b2), _) : Nat}
      Nats.add_swap(outers, batch_base(inner, outers, e1, e2, b1, b2), Nat.add(Nat.mul(k, outers), outer))

# A run whose reads start at a cell's first terms reads the cell's logical
# factor pairs.
def strided_logical(+fuel: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +lo: Matrix.Order, +ro: Matrix.Order, +inner: Nat,
  +rows: Nat, +columns: Nat, +l1: Nat, +l2: Nat, +r1: Nat, +r2: Nat, +b1: Nat, +b2: Nat, +row: Nat, +column: Nat, +k: Nat,
  +lp: Nat, +rp: Nat, values: List<&2, DotFold.Factors>,
  -el: {lp == Laws.operand_offset(lo, inner, rows, l1, l2, b1, b2, row, k) : Nat},
  -er: {rp == Laws.operand_offset(ro, inner, columns, r1, r2, b1, b2, column, k) : Nat},
  -strided: Laws.Strided(fuel, dl, dr, A, B, Matrix.stride(lo, rows), Matrix.stride(ro, columns), lp, rp, values)) ->
  Parallel.Fact<Laws.Logical(fuel, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, b1, b2, row, column, k, values)>:
  match fuel values:
    case 0n Nil{}:
      Parallel.Fact{Unit{}}
    case 1n+tail DotFold.Factors{lv, rv} <> rest:
      Parallel.Fact{{\{Laws.StridedStep: left_observed => right_observed => remaining =>
        Laws.LogicalStep{
          Equal.trans(Maybe<&2, F32>, Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(Laws.operand_offset(lo, inner, rows, l1, l2, b1, b2, row, k)))),
            Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(lp))), Some{lv},
            Equal.cong(Nat, Maybe<&2, F32>, x => Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(x))),
              Laws.operand_offset(lo, inner, rows, l1, l2, b1, b2, row, k), lp, Equal.sym(Nat, lp, Laws.operand_offset(lo, inner, rows, l1, l2, b1, b2, row, k), el)),
            left_observed),
          Equal.trans(Maybe<&2, F32>, Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(Laws.operand_offset(ro, inner, columns, r1, r2, b1, b2, column, k)))),
            Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(rp))), Some{rv},
            Equal.cong(Nat, Maybe<&2, F32>, x => Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(x))),
              Laws.operand_offset(ro, inner, columns, r1, r2, b1, b2, column, k), rp, Equal.sym(Nat, rp, Laws.operand_offset(ro, inner, columns, r1, r2, b1, b2, column, k), er)),
            right_observed),
          {\{Parallel.Fact: p => p} : Parallel.Fact<Laws.Logical(tail, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, b1, b2, row, column, 1n+k, rest)> ->
            Laws.Logical(tail, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, b1, b2, row, column, 1n+k, rest)}(
            strided_logical(tail, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, b1, b2, row, column, 1n+k,
              Nat.add(Matrix.stride(lo, rows), lp), Nat.add(Matrix.stride(ro, columns), rp), rest,
              Equal.trans(Nat, Nat.add(Matrix.stride(lo, rows), lp), Nat.add(Matrix.stride(lo, rows), Laws.operand_offset(lo, inner, rows, l1, l2, b1, b2, row, k)),
                Laws.operand_offset(lo, inner, rows, l1, l2, b1, b2, row, 1n+k),
                Equal.cong(Nat, Nat, x => Nat.add(Matrix.stride(lo, rows), x), lp, Laws.operand_offset(lo, inner, rows, l1, l2, b1, b2, row, k), el),
                operand_step(lo, inner, rows, l1, l2, b1, b2, row, k)),
              Equal.trans(Nat, Nat.add(Matrix.stride(ro, columns), rp), Nat.add(Matrix.stride(ro, columns), Laws.operand_offset(ro, inner, columns, r1, r2, b1, b2, column, k)),
                Laws.operand_offset(ro, inner, columns, r1, r2, b1, b2, column, 1n+k),
                Equal.cong(Nat, Nat, x => Nat.add(Matrix.stride(ro, columns), x), rp, Laws.operand_offset(ro, inner, columns, r1, r2, b1, b2, column, k), er),
                operand_step(ro, inner, columns, r1, r2, b1, b2, column, k)),
              remaining))}} :
        Laws.StridedStep<tail, dl, dr, A, B, Matrix.stride(lo, rows), Matrix.stride(ro, columns), lp, rp, lv, rv, rest> ->
        Laws.LogicalStep<tail, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, b1, b2, row, column, k, lv, rv, rest>}(strided)}
    case 0n DotFold.Factors{lv, rv} <> rest:
      Parallel.Fact{Empty.absurd(Laws.Logical(0n, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, b1, b2, row, column, k, DotFold.Factors{lv, rv} <> rest), strided)}
    case 1n+tail Nil{}:
      Parallel.Fact{Empty.absurd(Laws.Logical(1n+tail, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, b1, b2, row, column, k, Nil{}), strided)}

def CellIs(+dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +lo: Matrix.Order, +ro: Matrix.Order, +inner: Nat, +rows: Nat,
  +columns: Nat, +l1: Nat, +l2: Nat, +r1: Nat, +r2: Nat, +o2: Nat, +count: Nat, +position: Nat, -value: Maybe<&2, F32>) -> Type:
  Laws.CellSpec(dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, Nat.is_lt(position, count), value)

def cell_shift(+dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +lo: Matrix.Order, +ro: Matrix.Order, +inner: Nat, +rows: Nat,
  +columns: Nat, +l1: Nat, +l2: Nat, +r1: Nat, +r2: Nat, +o2: Nat, +count: Nat, -position: Nat, -other: Nat, -value: Maybe<&2, F32>,
  -same: {position == other : Nat}, -at: CellIs(dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, count, position, value)) ->
  Parallel.Fact<CellIs(dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, count, other, value)>:
  Parallel.Fact{
    %same : CellIs(dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, count, _, value)
    at}

def cell_below(+dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +lo: Matrix.Order, +ro: Matrix.Order, +inner: Nat, +rows: Nat,
  +columns: Nat, +l1: Nat, +l2: Nat, +r1: Nat, +r2: Nat, +o2: Nat, +count: Nat, -position: Nat, -value: Maybe<&2, F32>,
  -below: {Nat.is_lt(position, count) == True{} : Bool},
  -at: CellIs(dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, count, position, value)) ->
  Parallel.Fact<Laws.CellAt<dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, value>>:
  Parallel.Fact{
    %below : Laws.CellSpec(dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, position, _, value)
    at}

def Laws.batched_build_logical_cells(depth, splits, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, count, base, la, ra, el, er,
  left_valid, right_valid, path, b1, b2, row, column, row_below, column_below, batch_below, at, inside):
  -cells = {\{Laws.Holds: q => q} :
    Laws.Holds<Laws.Cells(depth, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, count, base,
      Batched.build(depth, splits, Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), count, base, la, ra))> ->
    Laws.Cells(depth, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, count, base,
      Batched.build(depth, splits, Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), count, base, la, ra))}(
    Laws.batched_build_cells(depth, splits, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, count, base, la, ra, el, er,
      left_valid, right_valid))
  -shifted = {\{Parallel.Fact: q => q} :
    Parallel.Fact<CellIs(dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, count, Laws.cell_position(o2, rows, columns, b1, b2, row, column),
      Model.observe(F32, depth, Batched.build(depth, splits, Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), count, base, la, ra), path))> ->
    CellIs(dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, count, Laws.cell_position(o2, rows, columns, b1, b2, row, column),
      Model.observe(F32, depth, Batched.build(depth, splits, Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), count, base, la, ra), path))}(
    cell_shift(dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, count, Nat.add(Model.offset(depth, path), base),
      Laws.cell_position(o2, rows, columns, b1, b2, row, column),
      Model.observe(F32, depth, Batched.build(depth, splits, Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), count, base, la, ra), path),
      at, cells(path)))
  -cell = {\{Parallel.Fact: q => q} :
    Parallel.Fact<Laws.CellAt<dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, Laws.cell_position(o2, rows, columns, b1, b2, row, column),
      Model.observe(F32, depth, Batched.build(depth, splits, Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), count, base, la, ra), path)>> ->
    Laws.CellAt<dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, Laws.cell_position(o2, rows, columns, b1, b2, row, column),
      Model.observe(F32, depth, Batched.build(depth, splits, Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), count, base, la, ra), path)>}(
    cell_below(dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, count, Laws.cell_position(o2, rows, columns, b1, b2, row, column),
      Model.observe(F32, depth, Batched.build(depth, splits, Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), count, base, la, ra), path),
      inside, shifted))
  Laws.Holds{{\{Laws.CellAt: word => observed => values => logical => fold =>
    Laws.LogicalCell{word, observed, values,
      {\{Parallel.Fact: q => q} :
        Parallel.Fact<Laws.Logical(inner, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, b1, b2, row, column, 0n, values)> ->
        Laws.Logical(inner, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, b1, b2, row, column, 0n, values)}(
        strided_logical(inner, dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, b1, b2, row, column, 0n,
          Batched.left_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), Laws.cell_position(o2, rows, columns, b1, b2, row, column)),
          Batched.right_start(Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), Laws.cell_position(o2, rows, columns, b1, b2, row, column)),
          values,
          left_start_spec(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, b1, b2, row, column, row_below, column_below, batch_below),
          right_start_spec(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, b1, b2, row, column, row_below, column_below, batch_below),
          logical)),
      fold}} :
    Laws.CellAt<dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, o2, Laws.cell_position(o2, rows, columns, b1, b2, row, column),
      Model.observe(F32, depth, Batched.build(depth, splits, Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), count, base, la, ra), path)> ->
    Laws.LogicalCell<dl, dr, A, B, lo, ro, inner, rows, columns, l1, l2, r1, r2, b1, b2, row, column,
      Model.observe(F32, depth, Batched.build(depth, splits, Laws.layout_of(lo, ro, inner, rows, columns, l1, l2, r1, r2, o2), count, base, la, ra), path)>}(cell)}

# The plan's geometry.

def AxisMeaning(+l: U32, +r: U32, result: Result<&2, &2, Matrix.Error, U32>) -> Data:
  match result:
    case Fail{error}:
      Unit
    case Done{o}:
      {U32.to_nat(o) == Laws.extent(U32.to_nat(l), U32.to_nat(r)) : Nat}

def same_extent(+a: Nat, +b: Nat, +same: {a == b : Nat}, decision: Shape.BoolCase<Nat.is_eq(a, 1n)>) ->
  {a == Laws.extent(a, b) : Nat}:
  match decision:
    case Shape.BTrue{equal}:
      %equal : {a == Bool.pick(Nat, _, b, a) : Nat}
      same
    case Shape.BFalse{equal}:
      %equal : {a == Bool.pick(Nat, _, b, a) : Nat}
      {==}

def axis_right(+l: U32, +r: U32, +left_one: {False{} == Nat.is_eq(U32.to_nat(l), 1n) : Bool},
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(r), 1n)>) ->
  AxisMeaning(l, r, Batched.stretched(l, r, False{}, False{}, Nat.is_eq(U32.to_nat(r), 1n))):
  match decision:
    case Shape.BTrue{equal}:
      %equal : AxisMeaning(l, r, Batched.stretched(l, r, False{}, False{}, _))
      %left_one : {U32.to_nat(l) == Bool.pick(Nat, _, U32.to_nat(r), U32.to_nat(l)) : Nat}
      {==}
    case Shape.BFalse{equal}:
      %equal : AxisMeaning(l, r, Batched.stretched(l, r, False{}, False{}, _))
      Unit{}

def axis_left(+l: U32, +r: U32, decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(l), 1n)>) ->
  AxisMeaning(l, r, Batched.stretched(l, r, False{}, Nat.is_eq(U32.to_nat(l), 1n), Nat.is_eq(U32.to_nat(r), 1n))):
  match decision:
    case Shape.BTrue{+equal}:
      %equal : AxisMeaning(l, r, Batched.stretched(l, r, False{}, _, Nat.is_eq(U32.to_nat(r), 1n)))
      %equal : {U32.to_nat(r) == Bool.pick(Nat, _, U32.to_nat(r), U32.to_nat(l)) : Nat}
      {==}
    case Shape.BFalse{+equal}:
      %equal : AxisMeaning(l, r, Batched.stretched(l, r, False{}, _, Nat.is_eq(U32.to_nat(r), 1n)))
      axis_right(l, r, equal, Shape.bool_case(Nat.is_eq(U32.to_nat(r), 1n)))

def axis_same(+l: U32, +r: U32, decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(l), U32.to_nat(r))>) ->
  AxisMeaning(l, r, Batched.stretched(l, r, Nat.is_eq(U32.to_nat(l), U32.to_nat(r)), Nat.is_eq(U32.to_nat(l), 1n), Nat.is_eq(U32.to_nat(r), 1n))):
  match decision:
    case Shape.BTrue{+equal}:
      %equal : AxisMeaning(l, r, Batched.stretched(l, r, _, Nat.is_eq(U32.to_nat(l), 1n), Nat.is_eq(U32.to_nat(r), 1n)))
      same_extent(U32.to_nat(l), U32.to_nat(r), Lifecycle.natural_equal(U32.to_nat(l), U32.to_nat(r), equal),
        Shape.bool_case(Nat.is_eq(U32.to_nat(l), 1n)))
    case Shape.BFalse{equal}:
      %equal : AxisMeaning(l, r, Batched.stretched(l, r, _, Nat.is_eq(U32.to_nat(l), 1n), Nat.is_eq(U32.to_nat(r), 1n)))
      axis_left(l, r, Shape.bool_case(Nat.is_eq(U32.to_nat(l), 1n)))

# One batch axis of the plan has the extent PyTorch's broadcast gives it.
def axis_meaning(+l: U32, +r: U32) -> AxisMeaning(l, r, Batched.axis(l, r)):
  axis_same(l, r, Shape.bool_case(Nat.is_eq(U32.to_nat(l), U32.to_nat(r))))

# The plan's batch extents as naturals, as `Batched.batch_dims` lists them.
def batch_nats(+rank: Nat, +first: Nat, +second: Nat) -> List<&2, Nat>:
  match rank:
    case 0n:
      []
    case 1n+rest:
      match rest:
        case 0n:
          [second]
        case 1n+more:
          [first, second]

def naturals_append(+xs: List<&2, U32>, +ys: List<&2, U32>) ->
  {Shape.naturals(List.append(&2, U32, xs, ys)) == List.append(&2, Nat, Shape.naturals(xs), Shape.naturals(ys)) : List<&2, Nat>}:
  match xs:
    case Nil{}:
      {==}
    case +head <> tail:
      Equal.cong(List<&2, Nat>, List<&2, Nat>, rest => U32.to_nat(head) <> rest,
        Shape.naturals(List.append(&2, U32, tail, ys)), List.append(&2, Nat, Shape.naturals(tail), Shape.naturals(ys)), naturals_append(tail, ys))

def batch_part(+rank: Nat, +o1: U32, +o2: U32, +first: Nat, +second: Nat,
  +e1: {U32.to_nat(o1) == first : Nat}, +e2: {U32.to_nat(o2) == second : Nat}) ->
  {Shape.naturals(Batched.batch_dims(rank, o1, o2)) == batch_nats(rank, first, second) : List<&2, Nat>}:
  match rank:
    case 0n:
      {==}
    case 1n+rest:
      match rest:
        case 0n:
          %Equal.sym(Nat, U32.to_nat(o2), second, e2) : {[_] == [second] : List<&2, Nat>}
          {==}
        case 1n+more:
          %Equal.sym(Nat, U32.to_nat(o1), first, e1) : {[_, U32.to_nat(o2)] == [first, second] : List<&2, Nat>}
          %Equal.sym(Nat, U32.to_nat(o2), second, e2) : {[first, _] == [first, second] : List<&2, Nat>}
          {==}

def matrix_part(+rows: U32, +lv: Bool, +columns: U32, +rv: Bool) ->
  {Shape.naturals(Batched.matrix_dims(rows, lv, columns, rv)) ==
    List.append(&2, Nat, Laws.kept(lv, U32.to_nat(rows)), Laws.kept(rv, U32.to_nat(columns))) : List<&2, Nat>}:
  match lv rv:
    case True{} True{}:
      {==}
    case True{} False{}:
      {==}
    case False{} True{}:
      {==}
    case False{} False{}:
      {==}

def output_nats(+rank: Nat, +o1: U32, +o2: U32, +first: Nat, +second: Nat,
  +e1: {U32.to_nat(o1) == first : Nat}, +e2: {U32.to_nat(o2) == second : Nat}, +rows: U32, +lv: Bool, +columns: U32, +rv: Bool) ->
  {Shape.naturals(List.append(&2, U32, Batched.batch_dims(rank, o1, o2), Batched.matrix_dims(rows, lv, columns, rv))) ==
    List.append(&2, Nat, batch_nats(rank, first, second),
      List.append(&2, Nat, Laws.kept(lv, U32.to_nat(rows)), Laws.kept(rv, U32.to_nat(columns)))) : List<&2, Nat>}:
  %Equal.sym(List<&2, Nat>, Shape.naturals(List.append(&2, U32, Batched.batch_dims(rank, o1, o2), Batched.matrix_dims(rows, lv, columns, rv))),
    List.append(&2, Nat, Shape.naturals(Batched.batch_dims(rank, o1, o2)), Shape.naturals(Batched.matrix_dims(rows, lv, columns, rv))),
    naturals_append(Batched.batch_dims(rank, o1, o2), Batched.matrix_dims(rows, lv, columns, rv))) :
    {_ == List.append(&2, Nat, batch_nats(rank, first, second),
      List.append(&2, Nat, Laws.kept(lv, U32.to_nat(rows)), Laws.kept(rv, U32.to_nat(columns)))) : List<&2, Nat>}
  %Equal.sym(List<&2, Nat>, Shape.naturals(Batched.batch_dims(rank, o1, o2)), batch_nats(rank, first, second), batch_part(rank, o1, o2, first, second, e1, e2)) :
    {List.append(&2, Nat, _, Shape.naturals(Batched.matrix_dims(rows, lv, columns, rv))) == List.append(&2, Nat, batch_nats(rank, first, second),
      List.append(&2, Nat, Laws.kept(lv, U32.to_nat(rows)), Laws.kept(rv, U32.to_nat(columns)))) : List<&2, Nat>}
  %Equal.sym(List<&2, Nat>, Shape.naturals(Batched.matrix_dims(rows, lv, columns, rv)),
    List.append(&2, Nat, Laws.kept(lv, U32.to_nat(rows)), Laws.kept(rv, U32.to_nat(columns))), matrix_part(rows, lv, columns, rv)) :
    {List.append(&2, Nat, batch_nats(rank, first, second), _) == List.append(&2, Nat, batch_nats(rank, first, second),
      List.append(&2, Nat, Laws.kept(lv, U32.to_nat(rows)), Laws.kept(rv, U32.to_nat(columns)))) : List<&2, Nat>}
  {==}

def pair_nats(+a: Nat, +b: Nat, +c: Nat, +d: Nat, +ac: {a == c : Nat}, +bd: {b == d : Nat}) -> {[a, b] == [c, d] : List<&2, Nat>}:
  %Equal.sym(Nat, a, c, ac) : {[_, b] == [c, d] : List<&2, Nat>}
  %Equal.sym(Nat, b, d, bd) : {[c, _] == [c, d] : List<&2, Nat>}
  {==}

# What the plan of two operands of these fields holds, in the operands'
# terms.
def ExpectedOf(+lr: Nat, +l1: U32, +l2: U32, +rows: U32, +inner: U32, +lv: Bool, +rr: Nat, +r1: U32, +r2: U32, +columns: U32,
  +rinner: U32, +rv: Bool, result: Result<&2, &2, Matrix.Error, Batched.Plan>) -> Data:
  Laws.GeometryOf(
    List.append(&2, Nat, batch_nats(Batched.larger(lr, rr), Laws.extent(U32.to_nat(l1), U32.to_nat(r1)), Laws.extent(U32.to_nat(l2), U32.to_nat(r2))),
      List.append(&2, Nat, Laws.kept(lv, U32.to_nat(rows)), Laws.kept(rv, U32.to_nat(columns)))),
    [U32.to_nat(l1), U32.to_nat(l2), U32.to_nat(rows), U32.to_nat(inner)],
    [U32.to_nat(r1), U32.to_nat(r2), U32.to_nat(rinner), U32.to_nat(columns)],
    [Laws.extent(U32.to_nat(l1), U32.to_nat(r1)), Laws.extent(U32.to_nat(l2), U32.to_nat(r2))],
    (lv, rv), result)

def geometry_shaped(+lr: Nat, +l1: U32, +l2: U32, +rows: U32, +inner: U32, +lv: Bool, +rr: Nat, +r1: U32, +r2: U32, +columns: U32,
  +rinner: U32, +rv: Bool, +o1: U32, +o2: U32, +same: {U32.to_nat(inner) == U32.to_nat(rinner) : Nat},
  +e1: {U32.to_nat(o1) == Laws.extent(U32.to_nat(l1), U32.to_nat(r1)) : Nat},
  +e2: {U32.to_nat(o2) == Laws.extent(U32.to_nat(l2), U32.to_nat(r2)) : Nat},
  result: Shape.ConstructResult<List.append(&2, U32, Batched.batch_dims(Batched.larger(lr, rr), o1, o2), Batched.matrix_dims(rows, lv, columns, rv))>) ->
  ExpectedOf(lr, l1, l2, rows, inner, lv, rr, r1, r2, columns, rinner, rv,
    Batched.planned_shape(Batched.batch_dims(Batched.larger(lr, rr), o1, o2), rows, inner, columns, l1, l2, r1, r2, o1, o2, lv, rv,
      List.append(&2, U32, Batched.batch_dims(Batched.larger(lr, rr), o1, o2), Batched.matrix_dims(rows, lv, columns, rv)), result)):
  match result:
    case Shape.SError{error}:
      Unit{}
    case Shape.SBuilt{+shape, +preserved}:
      Laws.Geometry{
        Equal.trans(List<&2, Nat>, Shape.naturals(Shape.dimensions(shape)),
          Shape.naturals(List.append(&2, U32, Batched.batch_dims(Batched.larger(lr, rr), o1, o2), Batched.matrix_dims(rows, lv, columns, rv))),
          List.append(&2, Nat, batch_nats(Batched.larger(lr, rr), Laws.extent(U32.to_nat(l1), U32.to_nat(r1)), Laws.extent(U32.to_nat(l2), U32.to_nat(r2))),
            List.append(&2, Nat, Laws.kept(lv, U32.to_nat(rows)), Laws.kept(rv, U32.to_nat(columns)))),
          Equal.cong(List<&2, U32>, List<&2, Nat>, dims => Shape.naturals(dims), Shape.dimensions(shape),
            List.append(&2, U32, Batched.batch_dims(Batched.larger(lr, rr), o1, o2), Batched.matrix_dims(rows, lv, columns, rv)), preserved),
          output_nats(Batched.larger(lr, rr), o1, o2, Laws.extent(U32.to_nat(l1), U32.to_nat(r1)), Laws.extent(U32.to_nat(l2), U32.to_nat(r2)),
            e1, e2, rows, lv, columns, rv)),
        {==},
        Equal.cong(Nat, List<&2, Nat>, x => [U32.to_nat(r1), U32.to_nat(r2), x, U32.to_nat(columns)], U32.to_nat(inner), U32.to_nat(rinner), same),
        pair_nats(U32.to_nat(o1), U32.to_nat(o2), Laws.extent(U32.to_nat(l1), U32.to_nat(r1)), Laws.extent(U32.to_nat(l2), U32.to_nat(r2)), e1, e2),
        {==}}

def geometry_batched(+lr: Nat, +l1: U32, +l2: U32, +rows: U32, +inner: U32, +lv: Bool, +rr: Nat, +r1: U32, +r2: U32, +columns: U32,
  +rinner: U32, +rv: Bool, +same: {U32.to_nat(inner) == U32.to_nat(rinner) : Nat},
  first: Result<&2, &2, Matrix.Error, U32>, second: Result<&2, &2, Matrix.Error, U32>,
  +m1: AxisMeaning(l1, r1, first), +m2: AxisMeaning(l2, r2, second)) ->
  ExpectedOf(lr, l1, l2, rows, inner, lv, rr, r1, r2, columns, rinner, rv,
    Batched.batched(Batched.larger(lr, rr), rows, inner, columns, l1, l2, r1, r2, lv, rv, first, second)):
  match first second:
    case Fail{error} second:
      Unit{}
    case Done{o1} Fail{error}:
      Unit{}
    case Done{+o1} Done{+o2}:
      geometry_shaped(lr, l1, l2, rows, inner, lv, rr, r1, r2, columns, rinner, rv, o1, o2, same, m1, m2,
        Shape.construct(List.append(&2, U32, Batched.batch_dims(Batched.larger(lr, rr), o1, o2), Batched.matrix_dims(rows, lv, columns, rv))))

def geometry_decided(+lr: Nat, +l1: U32, +l2: U32, +rows: U32, +inner: U32, +lv: Bool, +rr: Nat, +r1: U32, +r2: U32, +columns: U32,
  +rinner: U32, +rv: Bool, decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(inner), U32.to_nat(rinner))>) ->
  ExpectedOf(lr, l1, l2, rows, inner, lv, rr, r1, r2, columns, rinner, rv,
    Batched.compatible(Batched.Operand{Batched.Batch{lr, l1, l2}, rows, inner, lv}, Batched.Operand{Batched.Batch{rr, r1, r2}, columns, rinner, rv},
      Nat.is_eq(U32.to_nat(inner), U32.to_nat(rinner)))):
  match decision:
    case Shape.BFalse{equal}:
      %equal : ExpectedOf(lr, l1, l2, rows, inner, lv, rr, r1, r2, columns, rinner, rv,
        Batched.compatible(Batched.Operand{Batched.Batch{lr, l1, l2}, rows, inner, lv}, Batched.Operand{Batched.Batch{rr, r1, r2}, columns, rinner, rv}, _))
      Unit{}
    case Shape.BTrue{+equal}:
      %equal : ExpectedOf(lr, l1, l2, rows, inner, lv, rr, r1, r2, columns, rinner, rv,
        Batched.compatible(Batched.Operand{Batched.Batch{lr, l1, l2}, rows, inner, lv}, Batched.Operand{Batched.Batch{rr, r1, r2}, columns, rinner, rv}, _))
      geometry_batched(lr, l1, l2, rows, inner, lv, rr, r1, r2, columns, rinner, rv,
        Lifecycle.natural_equal(U32.to_nat(inner), U32.to_nat(rinner), equal),
        Batched.axis(l1, r1), Batched.axis(l2, r2), axis_meaning(l1, r1), axis_meaning(l2, r2))

def geometry_ops(+lr: Nat, +l1: U32, +l2: U32, +rows: U32, +inner: U32, +lv: Bool, +rr: Nat, +r1: U32, +r2: U32, +columns: U32,
  +rinner: U32, +rv: Bool) ->
  ExpectedOf(lr, l1, l2, rows, inner, lv, rr, r1, r2, columns, rinner, rv,
    Batched.compatible(Batched.Operand{Batched.Batch{lr, l1, l2}, rows, inner, lv}, Batched.Operand{Batched.Batch{rr, r1, r2}, columns, rinner, rv},
      Nat.is_eq(U32.to_nat(inner), U32.to_nat(rinner)))):
  geometry_decided(lr, l1, l2, rows, inner, lv, rr, r1, r2, columns, rinner, rv, Shape.bool_case(Nat.is_eq(U32.to_nat(inner), U32.to_nat(rinner))))

# The plan of each pair of ranks 1 to 4 is that of its two operands; any
# other rank is refused.
def Laws.batched_plan_geometry(left, right):
  match left:
    case Nil{}:
      Unit{}
    case +k <> Nil{}:
      match right:
        case Nil{}:
          Unit{}
        case +q <> Nil{}:
          geometry_ops(0n, 1, 1, 1, k, True{}, 0n, 1, 1, 1, q, True{})
        case +q <> +n <> Nil{}:
          geometry_ops(0n, 1, 1, 1, k, True{}, 0n, 1, 1, n, q, False{})
        case +c <> +q <> +n <> Nil{}:
          geometry_ops(0n, 1, 1, 1, k, True{}, 1n, 1, c, n, q, False{})
        case +c <> +d <> +q <> +n <> Nil{}:
          geometry_ops(0n, 1, 1, 1, k, True{}, 2n, c, d, n, q, False{})
        case +w <> +x <> +y <> +z <> +t <> rest:
          Unit{}
    case +m <> +k <> Nil{}:
      match right:
        case Nil{}:
          Unit{}
        case +q <> Nil{}:
          geometry_ops(0n, 1, 1, m, k, False{}, 0n, 1, 1, 1, q, True{})
        case +q <> +n <> Nil{}:
          geometry_ops(0n, 1, 1, m, k, False{}, 0n, 1, 1, n, q, False{})
        case +c <> +q <> +n <> Nil{}:
          geometry_ops(0n, 1, 1, m, k, False{}, 1n, 1, c, n, q, False{})
        case +c <> +d <> +q <> +n <> Nil{}:
          geometry_ops(0n, 1, 1, m, k, False{}, 2n, c, d, n, q, False{})
        case +w <> +x <> +y <> +z <> +t <> rest:
          Unit{}
    case +b <> +m <> +k <> Nil{}:
      match right:
        case Nil{}:
          Unit{}
        case +q <> Nil{}:
          geometry_ops(1n, 1, b, m, k, False{}, 0n, 1, 1, 1, q, True{})
        case +q <> +n <> Nil{}:
          geometry_ops(1n, 1, b, m, k, False{}, 0n, 1, 1, n, q, False{})
        case +c <> +q <> +n <> Nil{}:
          geometry_ops(1n, 1, b, m, k, False{}, 1n, 1, c, n, q, False{})
        case +c <> +d <> +q <> +n <> Nil{}:
          geometry_ops(1n, 1, b, m, k, False{}, 2n, c, d, n, q, False{})
        case +w <> +x <> +y <> +z <> +t <> rest:
          Unit{}
    case +a <> +b <> +m <> +k <> Nil{}:
      match right:
        case Nil{}:
          Unit{}
        case +q <> Nil{}:
          geometry_ops(2n, a, b, m, k, False{}, 0n, 1, 1, 1, q, True{})
        case +q <> +n <> Nil{}:
          geometry_ops(2n, a, b, m, k, False{}, 0n, 1, 1, n, q, False{})
        case +c <> +q <> +n <> Nil{}:
          geometry_ops(2n, a, b, m, k, False{}, 1n, 1, c, n, q, False{})
        case +c <> +d <> +q <> +n <> Nil{}:
          geometry_ops(2n, a, b, m, k, False{}, 2n, c, d, n, q, False{})
        case +w <> +x <> +y <> +z <> +t <> rest:
          Unit{}
    case +a <> +b <> +c <> +d <> +e <> rest:
      Unit{}
