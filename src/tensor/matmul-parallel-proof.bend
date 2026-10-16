import Base
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ./matmul.bend as Matrix
import ./dot-fold.bend as DotFold
import ./elementwise.bend as Elementwise
import ./elementwise-parallel-proof.bend as Parallel
import ./parallel-proof.bend as Generic

# Proofs that the parallel matmul build writes, at every cell of its output
# tree, the value the cell's dot names: for two balanced arrays A and B, a
# cell below `count` holds `DotFold.dot_fold` of the words A and B hold at
# the cell's row and column, in increasing k from +0.0, and a cell above it
# holds +0.0. Every read keeps the array it reads and every clone is two
# copies of it, so the whole build reads A and B.

# The factor pairs the terms `cursor .. cursor + fuel` of cell (row, column)
# read: left[row * inner + k] and right[k * columns + column].
def LogicalArr(+fuel: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat,
  +row: Nat, +column: Nat, +cursor: Nat, values: List<&2, DotFold.Factors>) -> Data:
  match fuel values:
    case 0n Nil{}:
      Unit
    case 1n+tail DotFold.Factors{lv, rv} <> rest:
      ArrStep<tail, dl, dr, A, B, inner, columns, row, column, cursor, lv, rv, rest>
    case 0n DotFold.Factors{lv, rv} <> rest:
      Empty
    case 1n+tail Nil{}:
      Empty

type ArrStep<-fuel: Nat, -dl: Nat, -dr: Nat, -A: Array<F32>, -B: Array<F32>, -inner: Nat, -columns: Nat,
  -row: Nat, -column: Nat, -cursor: Nat, -lv: F32, -rv: F32, -values: List<&2, DotFold.Factors>> is Data:
  ArrStep{-left_observed: {Model.observe(F32, dl, A, Refinement.route(dl, Matrix.left_index(inner, row, cursor))) == Some{lv} : Maybe<&2, F32>},
    -right_observed: {Model.observe(F32, dr, B, Refinement.route(dr, Matrix.right_index(columns, column, cursor))) == Some{rv} : Maybe<&2, F32>},
    -remaining: LogicalArr(fuel, dl, dr, A, B, inner, columns, row, column, 1n+cursor, values)}

# A run of `cell_dot`: its total folds the factor pairs it read, and both
# arrays come back unchanged.
type DotRun<-fuel: Nat, -dl: Nat, -dr: Nat, -A: Array<F32>, -B: Array<F32>, -inner: Nat, -columns: Nat,
  -row: Nat, -column: Nat, -cursor: Nat, -accumulator: F32, -result: F32 & (Array<F32> & Array<F32>)> is Data:
  DotRun{-values: List<&2, DotFold.Factors>, -logical: LogicalArr(fuel, dl, dr, A, B, inner, columns, row, column, cursor, values),
    -fold: {Pair.fst(F32, Array<F32> & Array<F32>, result) == DotFold.dot_fold(values, accumulator) : F32},
    -left_kept: {Pair.fst(Array<F32>, Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, result)) == A : Array<F32>},
    -right_kept: {Pair.snd(Array<F32>, Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, result)) == B : Array<F32>}}

def dot_extend(+rest: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat,
  +row: Nat, +column: Nat, +cursor: Nat, -accumulator: F32, -x: F32, -y: F32,
  -ex: {Model.observe(F32, dl, A, Refinement.route(dl, Matrix.left_index(inner, row, cursor))) == Some{x} : Maybe<&2, F32>},
  -ey: {Model.observe(F32, dr, B, Refinement.route(dr, Matrix.right_index(columns, column, cursor))) == Some{y} : Maybe<&2, F32>},
  -result: F32 & (Array<F32> & Array<F32>),
  -run: Parallel.Fact<DotRun<rest, dl, dr, A, B, inner, columns, row, column, 1n+cursor, F32.add(accumulator, F32.mul(x, y)), result>>) ->
  Parallel.Fact<DotRun<1n+rest, dl, dr, A, B, inner, columns, row, column, cursor, accumulator, result>>:
  Parallel.Fact{{\{DotRun: values => logical => fold => left_kept => right_kept =>
    DotRun{DotFold.Factors{x, y} <> values, ArrStep{ex, ey, logical}, fold, left_kept, right_kept}} :
      DotRun<rest, dl, dr, A, B, inner, columns, row, column, 1n+cursor, F32.add(accumulator, F32.mul(x, y)), result> ->
      DotRun<1n+rest, dl, dr, A, B, inner, columns, row, column, cursor, accumulator, result>}(
    {\{Parallel.Fact: p => p} : Parallel.Fact<DotRun<rest, dl, dr, A, B, inner, columns, row, column, 1n+cursor, F32.add(accumulator, F32.mul(x, y)), result>> ->
      DotRun<rest, dl, dr, A, B, inner, columns, row, column, 1n+cursor, F32.add(accumulator, F32.mul(x, y)), result>}(run))}

def dot_right(+rest: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat,
  +row: Nat, +column: Nat, +cursor: Nat,
  -rec: F32 -> Array<F32> -> Array<F32> -> F32 & (Array<F32> & Array<F32>), -accumulator: F32, -x: F32, -la: Array<F32>,
  -ra: Array<F32>, pair: Array<F32> & F32,
  -exact: {pair == Array.get(F32, ra, Matrix.right_index(columns, column, cursor)) : Array<F32> & F32},
  -ex: {Model.observe(F32, dl, A, Refinement.route(dl, Matrix.left_index(inner, row, cursor))) == Some{x} : Maybe<&2, F32>},
  -el: {la == A : Array<F32>}, -er: {ra == B : Array<F32>}, -right_valid: Model.Balanced(F32, dr, B),
  -next: @value: F32 -> @l: Array<F32> -> @r: Array<F32> -> @-el: {l == A : Array<F32>} -> @-er: {r == B : Array<F32>} ->
    Parallel.Fact<DotRun<rest, dl, dr, A, B, inner, columns, row, column, 1n+cursor, value, rec(value, l, r)>>) ->
  Parallel.Fact<DotRun<1n+rest, dl, dr, A, B, inner, columns, row, column, cursor, accumulator, Matrix.term_right(rec, accumulator, x, la, pair)>>:
  (kept, y) = pair
  dot_extend(rest, dl, dr, A, B, inner, columns, row, column, cursor, accumulator, x, y, ex,
    Equal.sym(Maybe<&2, F32>, Some{y}, Model.observe(F32, dr, B, Refinement.route(dr, Matrix.right_index(columns, column, cursor))),
      {\{Parallel.Fact: p => p} : Parallel.Fact<({Some{Refinement.second(F32, (kept, y))} == Model.observe(F32, dr, B, Refinement.route(dr, Matrix.right_index(columns, column, cursor))) : Maybe<&2, F32>})> ->
        {Some{Refinement.second(F32, (kept, y))} == Model.observe(F32, dr, B, Refinement.route(dr, Matrix.right_index(columns, column, cursor))) : Maybe<&2, F32>}}(
        Parallel.read_value(dr, ra, B, Matrix.right_index(columns, column, cursor), (kept, y), exact, er, right_valid))),
    rec(F32.add(accumulator, F32.mul(x, y)), la, kept),
    next(F32.add(accumulator, F32.mul(x, y)))(la)(kept)(el)(
      {\{Parallel.Fact: p => p} : Parallel.Fact<({Refinement.first(F32, (kept, y)) == B : Array<F32>})> -> {Refinement.first(F32, (kept, y)) == B : Array<F32>}}(
        Parallel.read_keeps(dr, ra, B, Matrix.right_index(columns, column, cursor), (kept, y), exact, er, right_valid))))

def dot_left(+rest: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat,
  +row: Nat, +column: Nat, +cursor: Nat,
  -rec: F32 -> Array<F32> -> Array<F32> -> F32 & (Array<F32> & Array<F32>), -accumulator: F32, -la: Array<F32>, ra: Array<F32>,
  pair: Array<F32> & F32,
  -exact: {pair == Array.get(F32, la, Matrix.left_index(inner, row, cursor)) : Array<F32> & F32},
  -el: {la == A : Array<F32>}, -er: {ra == B : Array<F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B),
  -next: @value: F32 -> @l: Array<F32> -> @r: Array<F32> -> @-el: {l == A : Array<F32>} -> @-er: {r == B : Array<F32>} ->
    Parallel.Fact<DotRun<rest, dl, dr, A, B, inner, columns, row, column, 1n+cursor, value, rec(value, l, r)>>) ->
  Parallel.Fact<DotRun<1n+rest, dl, dr, A, B, inner, columns, row, column, cursor, accumulator,
    Matrix.term_left(rec, accumulator, ra, Matrix.right_index(columns, column, cursor), pair)>>:
  (kept, x) = pair
  dot_right(rest, dl, dr, A, B, inner, columns, row, column, cursor, rec, accumulator, x, kept, ra,
    Array.get(F32, ra, Matrix.right_index(columns, column, cursor)), {==},
    Equal.sym(Maybe<&2, F32>, Some{x}, Model.observe(F32, dl, A, Refinement.route(dl, Matrix.left_index(inner, row, cursor))),
      {\{Parallel.Fact: p => p} : Parallel.Fact<({Some{Refinement.second(F32, (kept, x))} == Model.observe(F32, dl, A, Refinement.route(dl, Matrix.left_index(inner, row, cursor))) : Maybe<&2, F32>})> ->
        {Some{Refinement.second(F32, (kept, x))} == Model.observe(F32, dl, A, Refinement.route(dl, Matrix.left_index(inner, row, cursor))) : Maybe<&2, F32>}}(
        Parallel.read_value(dl, la, A, Matrix.left_index(inner, row, cursor), (kept, x), exact, el, left_valid))),
    {\{Parallel.Fact: p => p} : Parallel.Fact<({Refinement.first(F32, (kept, x)) == A : Array<F32>})> -> {Refinement.first(F32, (kept, x)) == A : Array<F32>}}(
      Parallel.read_keeps(dl, la, A, Matrix.left_index(inner, row, cursor), (kept, x), exact, el, left_valid)),
    er, right_valid, next)

def dot_run(fuel: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat,
  +row: Nat, +column: Nat, +cursor: Nat, -accumulator: F32, la: Array<F32>, ra: Array<F32>,
  -el: {la == A : Array<F32>}, -er: {ra == B : Array<F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Parallel.Fact<DotRun<fuel, dl, dr, A, B, inner, columns, row, column, cursor, accumulator,
    Matrix.cell_dot(fuel, inner, columns, row, column, cursor, accumulator, la, ra)>>:
  match fuel:
    case 0n:
      Parallel.Fact{DotRun{Nil{}, Unit{}, {==}, el, er}}
    case 1n+rest:
      dot_left(rest, dl, dr, A, B, inner, columns, row, column, cursor,
        value => l => r => Matrix.cell_dot(rest, inner, columns, row, column, 1n+cursor, value, l, r), accumulator, la, ra,
        Array.get(F32, la, Matrix.left_index(inner, row, cursor)), {==}, el, er, left_valid, right_valid,
        value => l => r => el => er => dot_run(rest, dl, dr, A, B, inner, columns, row, column, 1n+cursor, value, l, r, el, er,
          left_valid, right_valid))

# What output cell `position` holds: +0.0 at or above `count`; below it, a
# word that folds the factor pairs of row `position / columns` and column
# `position % columns`, in increasing k from +0.0.
type CellAt<-dl: Nat, -dr: Nat, -A: Array<F32>, -B: Array<F32>, -inner: Nat, -columns: Nat, -position: Nat,
  -value: Maybe<&2, F32>> is Data:
  CellAt{-word: F32, -observed: {value == Some{word} : Maybe<&2, F32>}, -values: List<&2, DotFold.Factors>,
    -logical: LogicalArr(inner, dl, dr, A, B, inner, columns, Matrix.row_of(columns, position), Matrix.column_of(columns, position), 0n, values),
    -fold: {word == DotFold.dot_fold(values, 0.0) : F32}}

def CellSpec(+dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat, +position: Nat, inside: Bool,
  -value: Maybe<&2, F32>) -> Type:
  match inside:
    case False{}:
      {value == Some{0.0} : Maybe<&2, F32>}
    case True{}:
      CellAt<dl, dr, A, B, inner, columns, position, value>

def CellIs(+dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat, +count: Nat, +position: Nat,
  -value: Maybe<&2, F32>) -> Type:
  CellSpec(dl, dr, A, B, inner, columns, position, Nat.is_lt(position, count), value)

# Every slot of `out`, at index `base + offset`, holds what CellIs names.
def Cells(+depth: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat,
  -out: Array<F32>) -> Type:
  @+path: Model.Path(depth) ->
    CellIs(dl, dr, A, B, inner, columns, count, Nat.add(Model.offset(depth, path), base), Model.observe(F32, depth, out, path))

# The matmul cell function of `Parallel.build`, from the input arrays.
def cell_of(+inner: Nat, +columns: Nat, +position: Nat, A: Array<F32>, B: Array<F32>, inside: Bool) -> F32 & (Array<F32> & Array<F32>):
  Matrix.cell_at(Matrix.dims(Matrix.Dims{inner, columns}), position, (A, B), inside)

def eta(-X: Type, -Y: Type, pair: X & Y) -> {pair == (Pair.fst(X, Y, pair), Pair.snd(X, Y, pair)) : X & Y}:
  (x, y) = pair
  {==}

# Every cell hands both input arrays back.
def cell_keeps(+dl: Nat, +dr: Nat, A: Array<F32>, B: Array<F32>, +inner: Nat, +columns: Nat, +position: Nat, inside: Bool,
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Parallel.Fact<{Pair.snd(F32, Array<F32> & Array<F32>, cell_of(inner, columns, position, A, B, inside)) == (A, B) : Array<F32> & Array<F32>}>:
  match inside:
    case False{}:
      Parallel.Fact{{==}}
    case True{}:
      Parallel.Fact{{\{DotRun: values => logical => fold => left_kept => right_kept =>
        Equal.trans(Array<F32> & Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, cell_of(inner, columns, position, A, B, True{})),
          (Pair.fst(Array<F32>, Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, cell_of(inner, columns, position, A, B, True{}))),
            Pair.snd(Array<F32>, Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, cell_of(inner, columns, position, A, B, True{})))),
          (A, B),
          eta(Array<F32>, Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, cell_of(inner, columns, position, A, B, True{}))),
          Parallel.pair_equal(Pair.fst(Array<F32>, Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, cell_of(inner, columns, position, A, B, True{}))),
            Pair.snd(Array<F32>, Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, cell_of(inner, columns, position, A, B, True{}))),
            A, B, left_kept, right_kept))} :
        DotRun<inner, dl, dr, A, B, inner, columns, Matrix.row_of(columns, position), Matrix.column_of(columns, position), 0n, 0.0,
          cell_of(inner, columns, position, A, B, True{})> ->
        {Pair.snd(F32, Array<F32> & Array<F32>, cell_of(inner, columns, position, A, B, True{})) == (A, B) : Array<F32> & Array<F32>}}(
        {\{Parallel.Fact: p => p} :
          Parallel.Fact<DotRun<inner, dl, dr, A, B, inner, columns, Matrix.row_of(columns, position), Matrix.column_of(columns, position), 0n, 0.0,
            cell_of(inner, columns, position, A, B, True{})>> ->
          DotRun<inner, dl, dr, A, B, inner, columns, Matrix.row_of(columns, position), Matrix.column_of(columns, position), 0n, 0.0,
            cell_of(inner, columns, position, A, B, True{})>}(
          dot_run(inner, dl, dr, A, B, inner, columns, Matrix.row_of(columns, position), Matrix.column_of(columns, position), 0n, 0.0, A, B,
            {==}, {==}, left_valid, right_valid)))}

# What the cell function computes is what CellSpec names.
def cell_spec(+dl: Nat, +dr: Nat, A: Array<F32>, B: Array<F32>, +inner: Nat, +columns: Nat, +position: Nat, inside: Bool,
  -observed: Maybe<&2, F32>, -computed: {observed == Some{Pair.fst(F32, Array<F32> & Array<F32>, cell_of(inner, columns, position, A, B, inside))} : Maybe<&2, F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Parallel.Fact<CellSpec(dl, dr, A, B, inner, columns, position, inside, observed)>:
  match inside:
    case False{}:
      Parallel.Fact{computed}
    case True{}:
      Parallel.Fact{{\{DotRun: values => logical => fold => left_kept => right_kept =>
        CellAt{Pair.fst(F32, Array<F32> & Array<F32>, cell_of(inner, columns, position, A, B, True{})), computed, values, logical, fold}} :
        DotRun<inner, dl, dr, A, B, inner, columns, Matrix.row_of(columns, position), Matrix.column_of(columns, position), 0n, 0.0,
          cell_of(inner, columns, position, A, B, True{})> ->
        CellAt<dl, dr, A, B, inner, columns, position, observed>}(
        {\{Parallel.Fact: p => p} :
          Parallel.Fact<DotRun<inner, dl, dr, A, B, inner, columns, Matrix.row_of(columns, position), Matrix.column_of(columns, position), 0n, 0.0,
            cell_of(inner, columns, position, A, B, True{})>> ->
          DotRun<inner, dl, dr, A, B, inner, columns, Matrix.row_of(columns, position), Matrix.column_of(columns, position), 0n, 0.0,
            cell_of(inner, columns, position, A, B, True{})>}(
          dot_run(inner, dl, dr, A, B, inner, columns, Matrix.row_of(columns, position), Matrix.column_of(columns, position), 0n, 0.0, A, B,
            {==}, {==}, left_valid, right_valid)))}

# Whatever the split, every cell of the parallel build holds what CellIs
# names: the generic law of `Parallel.build`, with the matmul cell.
def build_cells(+depth: Nat, +splits: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat,
  +count: Nat, +base: Nat, la: Array<F32>, ra: Array<F32>, -el: {la == A : Array<F32>}, -er: {ra == B : Array<F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Parallel.Fact<Cells(depth, dl, dr, A, B, inner, columns, count, base, Matrix.build(depth, splits, inner, columns, count, base, la, ra))>:
  -split = {\{Parallel.Fact: q => q} : Parallel.Fact<({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>} &
    {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>})> ->
    {Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>} &
    {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>}}(
    Parallel.split_equal(dl, dr, A, B, left_valid, right_valid))
  -cells = {\{Generic.Fact: q => q} :
    Generic.Fact<Generic.Cells(~Matrix.Dims, ~Array<F32>, ~Array<F32>, ~Matrix.cell_at, Matrix.dims(Matrix.Dims{inner, columns}), (A, B), depth, count, base,
      Matrix.build(depth, splits, inner, columns, count, base, la, ra))> ->
    Generic.Cells(~Matrix.Dims, ~Array<F32>, ~Array<F32>, ~Matrix.cell_at, Matrix.dims(Matrix.Dims{inner, columns}), (A, B), depth, count, base,
      Matrix.build(depth, splits, inner, columns, count, base, la, ra))}(
    Generic.build_cells(~Matrix.Dims, ~Matrix.Dims, ~Array<F32>, ~Array<F32>, ~Matrix.dims, ~Elementwise.split, ~Matrix.cell_at,
      Matrix.Dims{inner, columns}, (A, B), depth, splits, count, base, (la, ra), Parallel.pair_equal(la, ra, A, B, el, er),
      at => inside => {\{Parallel.Fact: q => q} :
        Parallel.Fact<({Pair.snd(F32, Array<F32> & Array<F32>, cell_of(inner, columns, at, A, B, inside)) == (A, B) : Array<F32> & Array<F32>})> ->
        {Pair.snd(F32, Array<F32> & Array<F32>, cell_of(inner, columns, at, A, B, inside)) == (A, B) : Array<F32> & Array<F32>}}(
        cell_keeps(dl, dr, A, B, inner, columns, at, inside, left_valid, right_valid)),
      Pair.fst({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>},
        {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>}, split),
      Pair.snd({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>},
        {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>}, split)))
  Parallel.Fact{path => {\{Parallel.Fact: q => q} :
    Parallel.Fact<CellSpec(dl, dr, A, B, inner, columns, Nat.add(Model.offset(depth, path), base), Nat.is_lt(Nat.add(Model.offset(depth, path), base), count),
      Model.observe(F32, depth, Matrix.build(depth, splits, inner, columns, count, base, la, ra), path))> ->
    CellSpec(dl, dr, A, B, inner, columns, Nat.add(Model.offset(depth, path), base), Nat.is_lt(Nat.add(Model.offset(depth, path), base), count),
      Model.observe(F32, depth, Matrix.build(depth, splits, inner, columns, count, base, la, ra), path))}(
    cell_spec(dl, dr, A, B, inner, columns, Nat.add(Model.offset(depth, path), base), Nat.is_lt(Nat.add(Model.offset(depth, path), base), count),
      Model.observe(F32, depth, Matrix.build(depth, splits, inner, columns, count, base, la, ra), path), cells(path), left_valid, right_valid))}

# A cell named at an equal index, and a CellIs below the count is a CellAt.
def cell_shift(+dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat, +count: Nat,
  -position: Nat, -other: Nat, -value: Maybe<&2, F32>, -same: {position == other : Nat},
  -at: CellIs(dl, dr, A, B, inner, columns, count, position, value)) ->
  Parallel.Fact<CellIs(dl, dr, A, B, inner, columns, count, other, value)>:
  Parallel.Fact{
    %same : CellIs(dl, dr, A, B, inner, columns, count, _, value)
    at}

def cell_below(+dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat, +count: Nat,
  -position: Nat, -value: Maybe<&2, F32>, -below: {Nat.is_lt(position, count) == True{} : Bool},
  -at: CellIs(dl, dr, A, B, inner, columns, count, position, value)) ->
  Parallel.Fact<CellAt<dl, dr, A, B, inner, columns, position, value>>:
  Parallel.Fact{
    %below : CellSpec(dl, dr, A, B, inner, columns, position, _, value)
    at}
