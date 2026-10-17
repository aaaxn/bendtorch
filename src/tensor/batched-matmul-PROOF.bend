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
