import Base
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ./matmul.bend as Matrix
import ./dot-fold.bend as DotFold
import ./elementwise.bend as Elementwise
import ./elementwise-parallel-proof.bend as Parallel

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

def TreeFacts(+depth: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat,
  -result: Array<F32> & (Array<F32> & Array<F32>)) -> Type:
  {Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == A : Array<F32>} &
    ({Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == B : Array<F32>} &
      Cells(depth, dl, dr, A, B, inner, columns, count, base, Pair.fst(Array<F32>, Array<F32> & Array<F32>, result)))

def facts_left(+depth: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat,
  -result: Array<F32> & (Array<F32> & Array<F32>),
  -facts: Parallel.Fact<TreeFacts(depth, dl, dr, A, B, inner, columns, count, base, result)>) ->
  Parallel.Fact<{Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == A : Array<F32>}>:
  Parallel.Fact{Pair.fst({Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == A : Array<F32>},
    ({Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == B : Array<F32>} &
      Cells(depth, dl, dr, A, B, inner, columns, count, base, Pair.fst(Array<F32>, Array<F32> & Array<F32>, result))),
    {\{Parallel.Fact: p => p} : Parallel.Fact<(TreeFacts(depth, dl, dr, A, B, inner, columns, count, base, result))> ->
      TreeFacts(depth, dl, dr, A, B, inner, columns, count, base, result)}(facts))}

def facts_right(+depth: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat,
  -result: Array<F32> & (Array<F32> & Array<F32>),
  -facts: Parallel.Fact<TreeFacts(depth, dl, dr, A, B, inner, columns, count, base, result)>) ->
  Parallel.Fact<{Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == B : Array<F32>}>:
  Parallel.Fact{Pair.fst({Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == B : Array<F32>},
    Cells(depth, dl, dr, A, B, inner, columns, count, base, Pair.fst(Array<F32>, Array<F32> & Array<F32>, result)),
    Pair.snd({Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == A : Array<F32>},
      ({Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == B : Array<F32>} &
        Cells(depth, dl, dr, A, B, inner, columns, count, base, Pair.fst(Array<F32>, Array<F32> & Array<F32>, result))),
      {\{Parallel.Fact: p => p} : Parallel.Fact<(TreeFacts(depth, dl, dr, A, B, inner, columns, count, base, result))> ->
        TreeFacts(depth, dl, dr, A, B, inner, columns, count, base, result)}(facts)))}

def facts_cells(+depth: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat,
  -result: Array<F32> & (Array<F32> & Array<F32>),
  -facts: Parallel.Fact<TreeFacts(depth, dl, dr, A, B, inner, columns, count, base, result)>) ->
  Parallel.Fact<Cells(depth, dl, dr, A, B, inner, columns, count, base, Pair.fst(Array<F32>, Array<F32> & Array<F32>, result))>:
  Parallel.Fact{Pair.snd({Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == B : Array<F32>},
    Cells(depth, dl, dr, A, B, inner, columns, count, base, Pair.fst(Array<F32>, Array<F32> & Array<F32>, result)),
    Pair.snd({Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == A : Array<F32>},
      ({Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == B : Array<F32>} &
        Cells(depth, dl, dr, A, B, inner, columns, count, base, Pair.fst(Array<F32>, Array<F32> & Array<F32>, result))),
      {\{Parallel.Fact: p => p} : Parallel.Fact<(TreeFacts(depth, dl, dr, A, B, inner, columns, count, base, result))> ->
        TreeFacts(depth, dl, dr, A, B, inner, columns, count, base, result)}(facts)))}

def cell_leaf_facts(+dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat,
  pair: F32 & (Array<F32> & Array<F32>), -decided: {True{} == Nat.is_lt(base, count) : Bool},
  -run: Parallel.Fact<DotRun<inner, dl, dr, A, B, inner, columns, Matrix.row_of(columns, base), Matrix.column_of(columns, base), 0n, 0.0, pair>>) ->
  Parallel.Fact<TreeFacts(0n, dl, dr, A, B, inner, columns, count, base, Matrix.cell_leaf(pair))>:
  (value, inputs) = pair
  (l, r) = inputs
  Parallel.Fact{{\{DotRun: values => logical => fold => left_kept => right_kept =>
    (left_kept, (right_kept, path =>
      %decided : CellSpec(dl, dr, A, B, inner, columns, base, _, Some{value})
      CellAt{value, {==}, values, logical, fold}))} :
    DotRun<inner, dl, dr, A, B, inner, columns, Matrix.row_of(columns, base), Matrix.column_of(columns, base), 0n, 0.0, (value, (l, r))> ->
    TreeFacts(0n, dl, dr, A, B, inner, columns, count, base, (ALeaf{value}, (l, r)))}(
      {\{Parallel.Fact: p => p} :
        Parallel.Fact<DotRun<inner, dl, dr, A, B, inner, columns, Matrix.row_of(columns, base), Matrix.column_of(columns, base), 0n, 0.0, (value, (l, r))>> ->
        DotRun<inner, dl, dr, A, B, inner, columns, Matrix.row_of(columns, base), Matrix.column_of(columns, base), 0n, 0.0, (value, (l, r))>}(run))}

def cell_facts(+dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat,
  la: Array<F32>, ra: Array<F32>, inside: Bool, -decided: {inside == Nat.is_lt(base, count) : Bool},
  -el: {la == A : Array<F32>}, -er: {ra == B : Array<F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Parallel.Fact<TreeFacts(0n, dl, dr, A, B, inner, columns, count, base, Matrix.cell(inner, columns, base, la, ra, inside))>:
  match inside:
    case False{}:
      Parallel.Fact{(el, (er, path =>
        %decided : CellSpec(dl, dr, A, B, inner, columns, base, _, Some{0.0})
        {==}))}
    case True{}:
      cell_leaf_facts(dl, dr, A, B, inner, columns, count, base,
        Matrix.cell_dot(inner, inner, columns, Matrix.row_of(columns, base), Matrix.column_of(columns, base), 0n, 0.0, la, ra), decided,
        dot_run(inner, dl, dr, A, B, inner, columns, Matrix.row_of(columns, base), Matrix.column_of(columns, base), 0n, 0.0, la, ra, el, er,
          left_valid, right_valid))

def node_cell(+lower: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat,
  -a: Array<F32>, -b: Array<F32>, +path: Model.Path(1n+lower),
  -low: Parallel.Fact<Cells(lower, dl, dr, A, B, inner, columns, count, base, a)>,
  -high: Parallel.Fact<Cells(lower, dl, dr, A, B, inner, columns, count, Nat.add(Model.capacity(lower), base), b)>) ->
  Parallel.Fact<CellIs(dl, dr, A, B, inner, columns, count, Nat.add(Model.offset(1n+lower, path), base),
    Model.observe(F32, 1n+lower, ANode{a, b}, path))>:
  Model.Step{+bit, +rest} = path
  match bit:
    case False{}:
      Parallel.Fact{{\{Parallel.Fact: p => p} : Parallel.Fact<(Cells(lower, dl, dr, A, B, inner, columns, count, base, a))> ->
        Cells(lower, dl, dr, A, B, inner, columns, count, base, a)}(low)(rest)}
    case True{}:
      Parallel.Fact{
        %Parallel.position_step(Model.capacity(lower), Model.offset(lower, rest), base) :
          CellIs(dl, dr, A, B, inner, columns, count, _, Model.observe(F32, lower, b, rest))
        {\{Parallel.Fact: p => p} : Parallel.Fact<(Cells(lower, dl, dr, A, B, inner, columns, count, Nat.add(Model.capacity(lower), base), b))> ->
          Cells(lower, dl, dr, A, B, inner, columns, count, Nat.add(Model.capacity(lower), base), b)}(high)(rest)}

def node_cells(+lower: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat,
  -a: Array<F32>, -b: Array<F32>,
  -low: Parallel.Fact<Cells(lower, dl, dr, A, B, inner, columns, count, base, a)>,
  -high: Parallel.Fact<Cells(lower, dl, dr, A, B, inner, columns, count, Nat.add(Model.capacity(lower), base), b)>) ->
  Parallel.Fact<Cells(1n+lower, dl, dr, A, B, inner, columns, count, base, ANode{a, b})>:
  Parallel.Fact{path => {\{Parallel.Fact: p => p} :
    Parallel.Fact<(CellIs(dl, dr, A, B, inner, columns, count, Nat.add(Model.offset(1n+lower, path), base), Model.observe(F32, 1n+lower, ANode{a, b}, path)))> ->
    CellIs(dl, dr, A, B, inner, columns, count, Nat.add(Model.offset(1n+lower, path), base), Model.observe(F32, 1n+lower, ANode{a, b}, path))}(
      node_cell(lower, dl, dr, A, B, inner, columns, count, base, a, b, path, low, high))}

def tree_join_facts(+lower: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat,
  -a: Array<F32>, high: Array<F32> & (Array<F32> & Array<F32>),
  -low: Parallel.Fact<Cells(lower, dl, dr, A, B, inner, columns, count, base, a)>,
  -facts: Parallel.Fact<TreeFacts(lower, dl, dr, A, B, inner, columns, count, Nat.add(Model.capacity(lower), base), high)>) ->
  Parallel.Fact<TreeFacts(1n+lower, dl, dr, A, B, inner, columns, count, base, Elementwise.tree_join(a, high))>:
  (b, inputs) = high
  Parallel.Fact{(
    {\{Parallel.Fact: p => p} : Parallel.Fact<({Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, (b, inputs))) == A : Array<F32>})> ->
      {Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, (b, inputs))) == A : Array<F32>}}(
      facts_left(lower, dl, dr, A, B, inner, columns, count, Nat.add(Model.capacity(lower), base), (b, inputs), facts)),
    ({\{Parallel.Fact: p => p} : Parallel.Fact<({Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, (b, inputs))) == B : Array<F32>})> ->
      {Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, (b, inputs))) == B : Array<F32>}}(
      facts_right(lower, dl, dr, A, B, inner, columns, count, Nat.add(Model.capacity(lower), base), (b, inputs), facts)),
      {\{Parallel.Fact: p => p} : Parallel.Fact<(Cells(1n+lower, dl, dr, A, B, inner, columns, count, base, ANode{a, b}))> ->
        Cells(1n+lower, dl, dr, A, B, inner, columns, count, base, ANode{a, b})}(
        node_cells(lower, dl, dr, A, B, inner, columns, count, base, a, b, low,
          facts_cells(lower, dl, dr, A, B, inner, columns, count, Nat.add(Model.capacity(lower), base), (b, inputs), facts)))))}

def tree_node_facts(+lower: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat,
  rec: Array<F32> -> Array<F32> -> Array<F32> & (Array<F32> & Array<F32>), low: Array<F32> & (Array<F32> & Array<F32>),
  -facts: Parallel.Fact<TreeFacts(lower, dl, dr, A, B, inner, columns, count, base, low)>,
  -next: @l: Array<F32> -> @r: Array<F32> -> @-el: {l == A : Array<F32>} -> @-er: {r == B : Array<F32>} ->
    Parallel.Fact<TreeFacts(lower, dl, dr, A, B, inner, columns, count, Nat.add(Model.capacity(lower), base), rec(l, r))>) ->
  Parallel.Fact<TreeFacts(1n+lower, dl, dr, A, B, inner, columns, count, base, Elementwise.tree_node(rec, low))>:
  (a, inputs) = low
  (la, ra) = inputs
  tree_join_facts(lower, dl, dr, A, B, inner, columns, count, base, a, rec(la, ra),
    facts_cells(lower, dl, dr, A, B, inner, columns, count, base, (a, (la, ra)), facts),
    next(la)(ra)(
      {\{Parallel.Fact: p => p} : Parallel.Fact<({Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, (a, (la, ra)))) == A : Array<F32>})> ->
        {Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, (a, (la, ra)))) == A : Array<F32>}}(
        facts_left(lower, dl, dr, A, B, inner, columns, count, base, (a, (la, ra)), facts)))(
      {\{Parallel.Fact: p => p} : Parallel.Fact<({Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, (a, (la, ra)))) == B : Array<F32>})> ->
        {Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, (a, (la, ra)))) == B : Array<F32>}}(
        facts_right(lower, dl, dr, A, B, inner, columns, count, base, (a, (la, ra)), facts))))

def tree_facts(+depth: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat,
  la: Array<F32>, ra: Array<F32>, -el: {la == A : Array<F32>}, -er: {ra == B : Array<F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Parallel.Fact<TreeFacts(depth, dl, dr, A, B, inner, columns, count, base, Matrix.tree(depth, inner, columns, count, base, la, ra))>:
  match depth:
    case 0n:
      cell_facts(dl, dr, A, B, inner, columns, count, base, la, ra, Nat.is_lt(base, count), {==}, el, er, left_valid, right_valid)
    case 1n+lower:
      tree_node_facts(lower, dl, dr, A, B, inner, columns, count, base,
        l => r => Matrix.tree(lower, inner, columns, count, Nat.add(Model.capacity(lower), base), l, r),
        Matrix.tree(lower, inner, columns, count, base, la, ra),
        tree_facts(lower, dl, dr, A, B, inner, columns, count, base, la, ra, el, er, left_valid, right_valid),
        l => r => el => er => tree_facts(lower, dl, dr, A, B, inner, columns, count, Nat.add(Model.capacity(lower), base), l, r, el, er,
          left_valid, right_valid))

def fork_cells(+lower: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat,
  -la: Array<F32>, -ra: Array<F32>,
  lp: Array<F32> & Array<F32>, -left_exact: {lp == Array.clone(F32, la) : Array<F32> & Array<F32>},
  rp: Array<F32> & Array<F32>, -right_exact: {rp == Array.clone(F32, ra) : Array<F32> & Array<F32>},
  -low: Array<F32> -> Array<F32> -> Array<F32>, -high: Array<F32> -> Array<F32> -> Array<F32>,
  -el: {la == A : Array<F32>}, -er: {ra == B : Array<F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B),
  -low_cells: @l: Array<F32> -> @r: Array<F32> -> @-el: {l == A : Array<F32>} -> @-er: {r == B : Array<F32>} ->
    Parallel.Fact<Cells(lower, dl, dr, A, B, inner, columns, count, base, low(l, r))>,
  -high_cells: @l: Array<F32> -> @r: Array<F32> -> @-el: {l == A : Array<F32>} -> @-er: {r == B : Array<F32>} ->
    Parallel.Fact<Cells(lower, dl, dr, A, B, inner, columns, count, Nat.add(Model.capacity(lower), base), high(l, r))>) ->
  Parallel.Fact<Cells(1n+lower, dl, dr, A, B, inner, columns, count, base, Elementwise.fork(lp, rp, low, high))>:
  (l1, l2) = lp
  (r1, r2) = rp
  node_cells(lower, dl, dr, A, B, inner, columns, count, base, low(l1, r1), high(l2, r2),
    low_cells(l1)(r1)(
      {\{Parallel.Fact: p => p} : Parallel.Fact<({Pair.fst(Array<F32>, Array<F32>, (l1, l2)) == A : Array<F32>})> -> {Pair.fst(Array<F32>, Array<F32>, (l1, l2)) == A : Array<F32>}}(
        Parallel.clone_first(dl, la, A, (l1, l2), left_exact, el, left_valid)))(
      {\{Parallel.Fact: p => p} : Parallel.Fact<({Pair.fst(Array<F32>, Array<F32>, (r1, r2)) == B : Array<F32>})> -> {Pair.fst(Array<F32>, Array<F32>, (r1, r2)) == B : Array<F32>}}(
        Parallel.clone_first(dr, ra, B, (r1, r2), right_exact, er, right_valid))),
    high_cells(l2)(r2)(
      {\{Parallel.Fact: p => p} : Parallel.Fact<({Pair.snd(Array<F32>, Array<F32>, (l1, l2)) == A : Array<F32>})> -> {Pair.snd(Array<F32>, Array<F32>, (l1, l2)) == A : Array<F32>}}(
        Parallel.clone_second(dl, la, A, (l1, l2), left_exact, el, left_valid)))(
      {\{Parallel.Fact: p => p} : Parallel.Fact<({Pair.snd(Array<F32>, Array<F32>, (r1, r2)) == B : Array<F32>})> -> {Pair.snd(Array<F32>, Array<F32>, (r1, r2)) == B : Array<F32>}}(
        Parallel.clone_second(dr, ra, B, (r1, r2), right_exact, er, right_valid))))

# Whatever the split, every cell of the parallel build holds what CellIs
# names.
def build_cells(+depth: Nat, +splits: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +inner: Nat, +columns: Nat,
  +count: Nat, +base: Nat, la: Array<F32>, ra: Array<F32>, -el: {la == A : Array<F32>}, -er: {ra == B : Array<F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Parallel.Fact<Cells(depth, dl, dr, A, B, inner, columns, count, base, Matrix.build(depth, splits, inner, columns, count, base, la, ra))>:
  match depth splits:
    case 1n+lower 1n+rest:
      fork_cells(lower, dl, dr, A, B, inner, columns, count, base, la, ra, Array.clone(F32, la), {==}, Array.clone(F32, ra), {==},
        l => r => Matrix.build(lower, rest, inner, columns, count, base, l, r),
        l => r => Matrix.build(lower, rest, inner, columns, count, Nat.add(Model.capacity(lower), base), l, r),
        el, er, left_valid, right_valid,
        l => r => el => er => build_cells(lower, rest, dl, dr, A, B, inner, columns, count, base, l, r, el, er, left_valid, right_valid),
        l => r => el => er => build_cells(lower, rest, dl, dr, A, B, inner, columns, count, Nat.add(Model.capacity(lower), base), l, r, el, er,
          left_valid, right_valid))
    case 0n _:
      facts_cells(0n, dl, dr, A, B, inner, columns, count, base, Matrix.tree(0n, inner, columns, count, base, la, ra),
        tree_facts(0n, dl, dr, A, B, inner, columns, count, base, la, ra, el, er, left_valid, right_valid))
    case 1n+lower 0n:
      facts_cells(1n+lower, dl, dr, A, B, inner, columns, count, base, Matrix.tree(1n+lower, inner, columns, count, base, la, ra),
        tree_facts(1n+lower, dl, dr, A, B, inner, columns, count, base, la, ra, el, er, left_valid, right_valid))

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
