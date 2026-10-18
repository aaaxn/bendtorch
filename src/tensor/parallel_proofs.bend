import Base
import ../shape/nat_proofs.bend as Nats
import ../shape/index_proofs.bend as Index
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ../storage/clone.bend as Clone
import ./parallel.bend as Build

# Proofs that `Build.build` holds, at every output cell, what the
# one-thread tree computes there from the input state `start`, whatever the
# split. They need two facts of the operation: a cell hands `start` back
# unchanged (`Keeps`), and `split` copies `start` into both halves. Each
# operation then proves only what its cell function computes from `start`.

type Fact<-P: Type> is Data:
  Fact{-proof: P}

# What cell `position` holds: the F32 its cell function computes from
# `start`.
def CellIs(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B),
  +p: P, -start: (A & B), +count: Nat, +position: Nat, -observed: Maybe<&2, F32>) -> Type:
  {observed == Some{Pair.fst(F32, (A & B), cell(p, position, start, Nat.is_lt(position, count)))} : Maybe<&2, F32>}

# Every slot of `out`, the cells `base .. base + 2^depth`, holds what CellIs
# names for its index.
def Cells(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B),
  +p: P, -start: (A & B), +depth: Nat, +count: Nat, +base: Nat, -out: Array<F32>) -> Type:
  @+path: Model.Path(depth) ->
    CellIs(~P, ~A, ~B, ~cell, p, start, count, Nat.add(Model.offset(depth, path), base), Model.observe(F32, depth, out, path))

# A cell hands its input state back unchanged.
def Keeps(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B),
  +p: P, -start: (A & B)) -> Type:
  @+position: Nat -> @inside: Bool -> {Pair.snd(F32, (A & B), cell(p, position, start, inside)) == start : (A & B)}

# A run of the one-thread tree: it hands `start` back and fills its cells.
def TreeFacts(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B),
  +p: P, -start: (A & B), +depth: Nat, +count: Nat, +base: Nat, -result: Array<F32> & (A & B)) -> Type:
  {Pair.snd(Array<F32>, (A & B), result) == start : (A & B)} & Cells(~P, ~A, ~B, ~cell, p, start, depth, count, base, Pair.fst(Array<F32>, (A & B), result))

def facts_kept(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B),
  +p: P, -start: (A & B), +depth: Nat, +count: Nat, +base: Nat, -result: Array<F32> & (A & B),
  -facts: Fact<TreeFacts(~P, ~A, ~B, ~cell, p, start, depth, count, base, result)>) ->
  Fact<{Pair.snd(Array<F32>, (A & B), result) == start : (A & B)}>:
  Fact{Pair.fst({Pair.snd(Array<F32>, (A & B), result) == start : (A & B)},
    Cells(~P, ~A, ~B, ~cell, p, start, depth, count, base, Pair.fst(Array<F32>, (A & B), result)),
    {\{Fact: q => q} : Fact<(TreeFacts(~P, ~A, ~B, ~cell, p, start, depth, count, base, result))> ->
      TreeFacts(~P, ~A, ~B, ~cell, p, start, depth, count, base, result)}(facts))}

def facts_cells(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B),
  +p: P, -start: (A & B), +depth: Nat, +count: Nat, +base: Nat, -result: Array<F32> & (A & B),
  -facts: Fact<TreeFacts(~P, ~A, ~B, ~cell, p, start, depth, count, base, result)>) ->
  Fact<Cells(~P, ~A, ~B, ~cell, p, start, depth, count, base, Pair.fst(Array<F32>, (A & B), result))>:
  Fact{Pair.snd({Pair.snd(Array<F32>, (A & B), result) == start : (A & B)},
    Cells(~P, ~A, ~B, ~cell, p, start, depth, count, base, Pair.fst(Array<F32>, (A & B), result)),
    {\{Fact: q => q} : Fact<(TreeFacts(~P, ~A, ~B, ~cell, p, start, depth, count, base, result))> ->
      TreeFacts(~P, ~A, ~B, ~cell, p, start, depth, count, base, result)}(facts))}

def leaf_facts(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B),
  +p: P, -start: (A & B), +count: Nat, +base: Nat, pair: F32 & (A & B),
  -exact: {pair == cell(p, base, start, Nat.is_lt(base, count)) : F32 & (A & B)},
  -keeps: Keeps(~P, ~A, ~B, ~cell, p, start)) ->
  Fact<TreeFacts(~P, ~A, ~B, ~cell, p, start, 0n, count, base, Build.leaf(~(A & B), pair))>:
  (value, rest) = pair
  Fact{(Equal.trans((A & B), rest, Pair.snd(F32, (A & B), cell(p, base, start, Nat.is_lt(base, count))), start,
      Equal.cong(F32 & (A & B), (A & B), q => Pair.snd(F32, (A & B), q), (value, rest), cell(p, base, start, Nat.is_lt(base, count)), exact),
      keeps(base)(Nat.is_lt(base, count))),
    path => Equal.cong(F32 & (A & B), Maybe<&2, F32>, q => Some{Pair.fst(F32, (A & B), q)}, (value, rest),
      cell(p, base, start, Nat.is_lt(base, count)), exact))}

def position_step(+capacity: Nat, +offset: Nat, +base: Nat) ->
  {Nat.add(offset, Nat.add(capacity, base)) == Nat.add(Nat.add(capacity, offset), base) : Nat}:
  Equal.trans(Nat, Nat.add(offset, Nat.add(capacity, base)), Nat.add(capacity, Nat.add(offset, base)), Nat.add(Nat.add(capacity, offset), base),
    Nats.add_swap(offset, capacity, base), Index.add_assoc(capacity, offset, base))

def node_cell(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B),
  +p: P, -start: (A & B), +inner: Nat, +count: Nat, +base: Nat, -a: Array<F32>, -b: Array<F32>, +path: Model.Path(1n+inner),
  -low: Fact<Cells(~P, ~A, ~B, ~cell, p, start, inner, count, base, a)>,
  -high: Fact<Cells(~P, ~A, ~B, ~cell, p, start, inner, count, Build.upper(inner, base), b)>) ->
  Fact<CellIs(~P, ~A, ~B, ~cell, p, start, count, Nat.add(Model.offset(1n+inner, path), base), Model.observe(F32, 1n+inner, ANode{a, b}, path))>:
  Model.Step{+bit, +rest} = path
  match bit:
    case False{}:
      Fact{{\{Fact: q => q} : Fact<(Cells(~P, ~A, ~B, ~cell, p, start, inner, count, base, a))> ->
        Cells(~P, ~A, ~B, ~cell, p, start, inner, count, base, a)}(low)(rest)}
    case True{}:
      Fact{
        %position_step(Model.capacity(inner), Model.offset(inner, rest), base) :
          CellIs(~P, ~A, ~B, ~cell, p, start, count, _, Model.observe(F32, inner, b, rest))
        {\{Fact: q => q} : Fact<(Cells(~P, ~A, ~B, ~cell, p, start, inner, count, Build.upper(inner, base), b))> ->
          Cells(~P, ~A, ~B, ~cell, p, start, inner, count, Build.upper(inner, base), b)}(high)(rest)}

def node_cells(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B),
  +p: P, -start: (A & B), +inner: Nat, +count: Nat, +base: Nat, -a: Array<F32>, -b: Array<F32>,
  -low: Fact<Cells(~P, ~A, ~B, ~cell, p, start, inner, count, base, a)>,
  -high: Fact<Cells(~P, ~A, ~B, ~cell, p, start, inner, count, Build.upper(inner, base), b)>) ->
  Fact<Cells(~P, ~A, ~B, ~cell, p, start, 1n+inner, count, base, ANode{a, b})>:
  Fact{path => {\{Fact: q => q} :
    Fact<(CellIs(~P, ~A, ~B, ~cell, p, start, count, Nat.add(Model.offset(1n+inner, path), base), Model.observe(F32, 1n+inner, ANode{a, b}, path)))> ->
    CellIs(~P, ~A, ~B, ~cell, p, start, count, Nat.add(Model.offset(1n+inner, path), base), Model.observe(F32, 1n+inner, ANode{a, b}, path))}(
      node_cell(~P, ~A, ~B, ~cell, p, start, inner, count, base, a, b, path, low, high))}

def join_facts(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B),
  +p: P, -start: (A & B), +inner: Nat, +count: Nat, +base: Nat, -a: Array<F32>, high: Array<F32> & (A & B),
  -low: Fact<Cells(~P, ~A, ~B, ~cell, p, start, inner, count, base, a)>,
  -facts: Fact<TreeFacts(~P, ~A, ~B, ~cell, p, start, inner, count, Build.upper(inner, base), high)>) ->
  Fact<TreeFacts(~P, ~A, ~B, ~cell, p, start, 1n+inner, count, base, Build.join(~(A & B), a, high))>:
  (b, rest) = high
  Fact{(
    {\{Fact: q => q} : Fact<({Pair.snd(Array<F32>, (A & B), (b, rest)) == start : (A & B)})> -> {Pair.snd(Array<F32>, (A & B), (b, rest)) == start : (A & B)}}(
      facts_kept(~P, ~A, ~B, ~cell, p, start, inner, count, Build.upper(inner, base), (b, rest), facts)),
    {\{Fact: q => q} : Fact<(Cells(~P, ~A, ~B, ~cell, p, start, 1n+inner, count, base, ANode{a, b}))> ->
      Cells(~P, ~A, ~B, ~cell, p, start, 1n+inner, count, base, ANode{a, b})}(
      node_cells(~P, ~A, ~B, ~cell, p, start, inner, count, base, a, b, low,
        facts_cells(~P, ~A, ~B, ~cell, p, start, inner, count, Build.upper(inner, base), (b, rest), facts))))}

def node_facts(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B),
  +p: P, -start: (A & B), +inner: Nat, +count: Nat, +base: Nat, rec: A -> B -> Array<F32> & (A & B), low: Array<F32> & (A & B),
  -facts: Fact<TreeFacts(~P, ~A, ~B, ~cell, p, start, inner, count, base, low)>,
  -next: @l: A -> @r: B -> @-at: {(l, r) == start : (A & B)} ->
    Fact<TreeFacts(~P, ~A, ~B, ~cell, p, start, inner, count, Build.upper(inner, base), rec(l, r))>) ->
  Fact<TreeFacts(~P, ~A, ~B, ~cell, p, start, 1n+inner, count, base, Build.node(~A, ~B, rec, low))>:
  (a, s) = low
  (l, r) = s
  join_facts(~P, ~A, ~B, ~cell, p, start, inner, count, base, a, rec(l, r),
    facts_cells(~P, ~A, ~B, ~cell, p, start, inner, count, base, (a, (l, r)), facts),
    next(l)(r)({\{Fact: q => q} : Fact<({Pair.snd(Array<F32>, (A & B), (a, (l, r))) == start : (A & B)})> -> {Pair.snd(Array<F32>, (A & B), (a, (l, r))) == start : (A & B)}}(
      facts_kept(~P, ~A, ~B, ~cell, p, start, inner, count, base, (a, (l, r)), facts))))

# The depth-1 tree: the leaf of cell `base` next to the leaf of the cell
# after it, which reads the state the first hands back.
def pair_join_facts(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B),
  +p: P, -start: (A & B), +count: Nat, +base: Nat, x: F32, high: F32 & (A & B),
  -low: Fact<Cells(~P, ~A, ~B, ~cell, p, start, 0n, count, base, ALeaf{x})>,
  -exact: {high == cell(p, Nat.add(1n, base), start, Nat.is_lt(Nat.add(1n, base), count)) : F32 & (A & B)},
  -keeps: Keeps(~P, ~A, ~B, ~cell, p, start)) ->
  Fact<TreeFacts(~P, ~A, ~B, ~cell, p, start, 1n, count, base, Build.pair_join(~(A & B), x, high))>:
  (y, rest) = high
  join_facts(~P, ~A, ~B, ~cell, p, start, 0n, count, base, ALeaf{x}, Build.leaf(~(A & B), (y, rest)), low,
    leaf_facts(~P, ~A, ~B, ~cell, p, start, count, Build.upper(0n, base), (y, rest), exact, keeps))

def pair_facts(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B),
  +p: P, -start: (A & B), +count: Nat, +base: Nat, low: F32 & (A & B),
  -exact: {low == cell(p, base, start, Nat.is_lt(base, count)) : F32 & (A & B)},
  -keeps: Keeps(~P, ~A, ~B, ~cell, p, start)) ->
  Fact<TreeFacts(~P, ~A, ~B, ~cell, p, start, 1n, count, base, Build.pair(~P, ~A, ~B, ~cell, p, count, base, low))>:
  (x, s) = low
  (l, r) = s
  pair_join_facts(~P, ~A, ~B, ~cell, p, start, count, base, x, cell(p, Nat.add(1n, base), (l, r), Nat.is_lt(Nat.add(1n, base), count)),
    facts_cells(~P, ~A, ~B, ~cell, p, start, 0n, count, base, Build.leaf(~(A & B), (x, (l, r))),
      leaf_facts(~P, ~A, ~B, ~cell, p, start, count, base, (x, (l, r)), exact, keeps)),
    Equal.cong((A & B), F32 & (A & B), q => cell(p, Nat.add(1n, base), q, Nat.is_lt(Nat.add(1n, base), count)), (l, r), start,
      {\{Fact: q => q} : Fact<({Pair.snd(Array<F32>, (A & B), Build.leaf(~(A & B), (x, (l, r)))) == start : (A & B)})> ->
        {Pair.snd(Array<F32>, (A & B), Build.leaf(~(A & B), (x, (l, r)))) == start : (A & B)}}(
        facts_kept(~P, ~A, ~B, ~cell, p, start, 0n, count, base, Build.leaf(~(A & B), (x, (l, r))),
          leaf_facts(~P, ~A, ~B, ~cell, p, start, count, base, (x, (l, r)), exact, keeps)))),
    keeps)

# The one-thread tree hands `start` back and fills every cell as CellIs
# says.
def tree_facts(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B),
  +p: P, -start: (A & B), +depth: Nat, +count: Nat, +base: Nat, state: (A & B), -at: {state == start : (A & B)},
  -keeps: Keeps(~P, ~A, ~B, ~cell, p, start)) ->
  Fact<TreeFacts(~P, ~A, ~B, ~cell, p, start, depth, count, base, Build.tree(~P, ~A, ~B, ~cell, p, depth, count, base, state))>:
  match depth:
    case 0n:
      leaf_facts(~P, ~A, ~B, ~cell, p, start, count, base, cell(p, base, state, Nat.is_lt(base, count)),
        Equal.cong((A & B), F32 & (A & B), s => cell(p, base, s, Nat.is_lt(base, count)), state, start, at), keeps)
    case 1n+inner:
      match inner:
        case 0n:
          pair_facts(~P, ~A, ~B, ~cell, p, start, count, base, cell(p, base, state, Nat.is_lt(base, count)),
            Equal.cong((A & B), F32 & (A & B), s => cell(p, base, s, Nat.is_lt(base, count)), state, start, at), keeps)
        case 1n+below:
          node_facts(~P, ~A, ~B, ~cell, p, start, inner, count, base,
            l => r => Build.tree(~P, ~A, ~B, ~cell, p, inner, count, Build.upper(inner, base), (l, r)),
            Build.tree(~P, ~A, ~B, ~cell, p, inner, count, base, state),
            tree_facts(~P, ~A, ~B, ~cell, p, start, inner, count, base, state, at, keeps),
            l => r => at => tree_facts(~P, ~A, ~B, ~cell, p, start, inner, count, Build.upper(inner, base), (l, r), at, keeps))

def half_equal(-S: Type, -start: S, -state: S, -half: S, -split: S -> S & S, -pick: (S & S) -> S,
  picked: {half == pick(split(state)) : S}, at: {state == start : S}, copied: {pick(split(start)) == start : S}) ->
  {half == start : S}:
  Equal.trans(S, half, pick(split(start)), start,
    Equal.trans(S, half, pick(split(state)), pick(split(start)), picked, Equal.cong(S, S, s => pick(split(s)), state, start, at)),
    copied)

def fork_cells(~Q: Data, ~P: Data, ~A: Type, ~B: Type, ~prepare: @+q: Q -> P, ~split: (A & B) -> (A & B) & (A & B),
  ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B),
  +q: Q, -start: (A & B), +inner: Nat, +count: Nat, +base: Nat, -state: (A & B), pair: (A & B) & (A & B),
  -exact: {pair == split(state) : (A & B) & (A & B)},
  -low: A -> B -> Array<F32>, -high: A -> B -> Array<F32>, -at: {state == start : (A & B)},
  -first: {Pair.fst((A & B), (A & B), split(start)) == start : (A & B)}, -second: {Pair.snd((A & B), (A & B), split(start)) == start : (A & B)},
  -low_cells: @l: A -> @r: B -> @-at: {(l, r) == start : (A & B)} -> Fact<Cells(~P, ~A, ~B, ~cell, prepare(q), start, inner, count, base, low(l, r))>,
  -high_cells: @l: A -> @r: B -> @-at: {(l, r) == start : (A & B)} ->
    Fact<Cells(~P, ~A, ~B, ~cell, prepare(q), start, inner, count, Build.upper(inner, base), high(l, r))>) ->
  Fact<Cells(~P, ~A, ~B, ~cell, prepare(q), start, 1n+inner, count, base, Build.fork(~A, ~B, pair, low, high))>:
  (one, two) = pair
  (l1, r1) = one
  (l2, r2) = two
  node_cells(~P, ~A, ~B, ~cell, prepare(q), start, inner, count, base, low(l1, r1), high(l2, r2),
    low_cells(l1)(r1)(half_equal((A & B), start, state, (l1, r1), split, h => Pair.fst((A & B), (A & B), h),
      Equal.cong((A & B) & (A & B), (A & B), h => Pair.fst((A & B), (A & B), h), ((l1, r1), (l2, r2)), split(state), exact), at, first)),
    high_cells(l2)(r2)(half_equal((A & B), start, state, (l2, r2), split, h => Pair.snd((A & B), (A & B), h),
      Equal.cong((A & B) & (A & B), (A & B), h => Pair.snd((A & B), (A & B), h), ((l1, r1), (l2, r2)), split(state), exact), at, second)))

# Whatever the split, every cell of the parallel build holds what the
# one-thread tree computes there from `start`.
def build_cells(~Q: Data, ~P: Data, ~A: Type, ~B: Type, ~prepare: @+q: Q -> P, ~split: (A & B) -> (A & B) & (A & B), ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B),
  +q: Q, -start: (A & B), +depth: Nat, +splits: Nat, +count: Nat, +base: Nat, state: (A & B), -at: {state == start : (A & B)},
  -keeps: Keeps(~P, ~A, ~B, ~cell, prepare(q), start),
  -first: {Pair.fst((A & B), (A & B), split(start)) == start : (A & B)}, -second: {Pair.snd((A & B), (A & B), split(start)) == start : (A & B)}) ->
  Fact<Cells(~P, ~A, ~B, ~cell, prepare(q), start, depth, count, base, Build.build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, depth, splits, count, base, state))>:
  match depth splits:
    case 1n+inner 1n+rest:
      fork_cells(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, start, inner, count, base, state, split(state), {==},
        l => r => Build.build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, inner, rest, count, base, (l, r)),
        l => r => Build.build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, inner, rest, count, Build.upper(inner, base), (l, r)),
        at, first, second,
        l => r => at => build_cells(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, start, inner, rest, count, base, (l, r), at, keeps, first, second),
        l => r => at => build_cells(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, start, inner, rest, count, Build.upper(inner, base), (l, r), at, keeps, first, second))
    case 0n _:
      facts_cells(~P, ~A, ~B, ~cell, prepare(q), start, 0n, count, base, Build.tree(~P, ~A, ~B, ~cell, prepare(q), 0n, count, base, state),
        tree_facts(~P, ~A, ~B, ~cell, prepare(q), start, 0n, count, base, state, at, keeps))
    case 1n+inner 0n:
      facts_cells(~P, ~A, ~B, ~cell, prepare(q), start, 1n+inner, count, base, Build.tree(~P, ~A, ~B, ~cell, prepare(q), 1n+inner, count, base, state),
        tree_facts(~P, ~A, ~B, ~cell, prepare(q), start, 1n+inner, count, base, state, at, keeps))
