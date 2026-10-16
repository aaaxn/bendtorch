import Base
import ../storage/model.bend as Model

# One parallel build for every operation whose output cells are independent:
# elementwise, matmul and reduce_to. An operation gives the build a cell
# function: `cell(p, position, state, inside)` returns the F32 of output
# cell `position` and hands its input state back, and `split(state)` copies
# the state for the two halves of a fork. `p = prepare(q)` holds what every
# cell needs, such as an index map; each subtree below the forks prepares
# it once. The state is a pair `A & B`, such as two input arrays. The cells
# `base .. base + 2^depth` form a balanced tree. The top `splits` levels fork, each half with its own copy of the
# state, and each subtree below the forks fills its cells in index order on
# one thread. `cell`, `split` and `prepare` are templates, so each operation
# compiles its own copy of the build; the cell is still called, not inlined.

def leaf(~S: Type, pair: F32 & S) -> Array<F32> & S:
  (value, state) = pair
  (ALeaf{value}, state)

def join(~S: Type, a: Array<F32>, high: Array<F32> & S) -> Array<F32> & S:
  (b, state) = high
  (ANode{a, b}, state)

# The two components of the state go to `rec` one by one, so a closure
# call never boxes the state.
def node(~A: Type, ~B: Type, rec: A -> B -> Array<F32> & (A & B), low: Array<F32> & (A & B)) -> Array<F32> & (A & B):
  (a, state) = low
  (l, r) = state
  join(~(A & B), a, rec(l, r))

# The first cell of the high half of a node whose halves hold 2^inner cells.
def upper(+inner: Nat, +base: Nat) -> Nat:
  Nat.add(Model.capacity(inner), base)

# The depth-1 tree, cells `base` and `base + 1`: the second cell reads the
# state the first hands back, and the node of the two leaves is built here.
# The bottom level of every deeper tree runs this, so no leaf returns
# through `leaf` and half of the tree's nodes create no closure. The second
# cell is `Nat.add(1n, base)`, which is `upper(0n, base)` but a flat add.
def pair_join(~S: Type, x: F32, high: F32 & S) -> Array<F32> & S:
  (y, state) = high
  (ANode{ALeaf{x}, ALeaf{y}}, state)

def pair(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> A & B -> Bool -> F32 & (A & B),
  +p: P, +count: Nat, +base: Nat, low: F32 & (A & B)) -> Array<F32> & (A & B):
  (x, state) = low
  (l, r) = state
  pair_join(~(A & B), x, cell(p, Nat.add(1n, base), (l, r), Nat.is_lt(Nat.add(1n, base), count)))

# The cells `base .. base + 2^depth` as a balanced tree, built in index order
# on one thread. A cell at or past `count` is padding.
def tree(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> A & B -> Bool -> F32 & (A & B),
  +p: P, +depth: Nat, +count: Nat, +base: Nat, state: A & B) -> Array<F32> & (A & B):
  match depth:
    case 0n:
      leaf(~(A & B), cell(p, base, state, Nat.is_lt(base, count)))
    case 1n+inner:
      match inner:
        case 0n:
          pair(~P, ~A, ~B, ~cell, p, count, base, cell(p, base, state, Nat.is_lt(base, count)))
        case 1n+below:
          node(~A, ~B, l => r => tree(~P, ~A, ~B, ~cell, p, inner, count, upper(inner, base), (l, r)),
            tree(~P, ~A, ~B, ~cell, p, inner, count, base, state))

def fork(~A: Type, ~B: Type, pair: (A & B) & (A & B), low: A -> B -> Array<F32>, high: A -> B -> Array<F32>) -> Array<F32>:
  (first, second) = pair
  (l1, r1) = first
  (l2, r2) = second
  a b = low(l1, r1) high(l2, r2)
  ANode{a, b}

# The parallel build. Every cell gets the F32 the one-thread tree computes
# there, so no output word depends on `splits` or on the thread count.
def build(~Q: Data, ~P: Data, ~A: Type, ~B: Type, ~prepare: @+q: Q -> P, ~split: A & B -> (A & B) & (A & B),
  ~cell: @+p: P -> @+position: Nat -> A & B -> Bool -> F32 & (A & B),
  +q: Q, +depth: Nat, +splits: Nat, +count: Nat, +base: Nat, state: A & B) -> Array<F32>:
  match depth splits:
    case 1n+inner 1n+rest:
      +high = upper(inner, base)
      fork(~A, ~B, split(state),
        l => r => build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, inner, rest, count, base, (l, r)),
        l => r => build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, inner, rest, count, high, (l, r)))
    case _ _:
      Pair.fst(Array<F32>, A & B, tree(~P, ~A, ~B, ~cell, prepare(q), depth, count, base, state))

# A run of the one-thread tree next to its proof of balance: `result` is
# the term the code computes, and its first component is balanced at
# `depth`. The proofs only follow the code, so they cost nothing at runtime.
type Certified<-S: Type, -depth: Nat, -expected: Array<F32> & S> is Type:
  Certified{result: Array<F32> & S, same: {result == expected : Array<F32> & S},
    balanced: Model.Balanced(F32, depth, Pair.fst(Array<F32>, S, result))}

def leaf_certified(~S: Type, pair: F32 & S) -> Certified<S, 0n, leaf(~S, pair)>:
  (value, state) = pair
  Certified{leaf(~S, (value, state)), {==}, Unit{}}

def node_joined(~A: Type, ~B: Type, +inner: Nat, -rec: A -> B -> Array<F32> & (A & B), -low: Array<F32> & (A & B),
  a: Array<F32>, -l: A, -r: B, same: {(a, (l, r)) == low : Array<F32> & (A & B)}, balanced: Model.Balanced(F32, inner, a),
  high: Certified<(A & B), inner, rec(l, r)>) -> Certified<(A & B), 1n+inner, node(~A, ~B, rec, low)>:
  Certified{result, high_same, high_balanced} = high
  (b, rest) = result
  Certified{(ANode{a, b}, rest),
    %same : {(ANode{a, b}, rest) == node(~A, ~B, rec, _) : Array<F32> & (A & B)}
    %high_same : {(ANode{a, b}, rest) == join(~(A & B), a, _) : Array<F32> & (A & B)}
    {==},
    (balanced, high_balanced)}

def node_split(~A: Type, ~B: Type, +inner: Nat, -rec: A -> B -> Array<F32> & (A & B), -low: Array<F32> & (A & B),
  result: Array<F32> & (A & B), same: {result == low : Array<F32> & (A & B)},
  balanced: Model.Balanced(F32, inner, Pair.fst(Array<F32>, A & B, result)),
  next: @l: A -> @r: B -> Certified<(A & B), inner, rec(l, r)>) -> Certified<(A & B), 1n+inner, node(~A, ~B, rec, low)>:
  (a, state) = result
  (l, r) = state
  node_joined(~A, ~B, inner, rec, low, a, l, r, same, balanced, next(l, r))

def node_certified(~A: Type, ~B: Type, +inner: Nat, -rec: A -> B -> Array<F32> & (A & B), -low: Array<F32> & (A & B),
  certified: Certified<(A & B), inner, low>, next: @l: A -> @r: B -> Certified<(A & B), inner, rec(l, r)>) ->
  Certified<(A & B), 1n+inner, node(~A, ~B, rec, low)>:
  Certified{result, same, balanced} = certified
  node_split(~A, ~B, inner, rec, low, result, same, balanced, next)

def pair_joined(~S: Type, x: F32, high: F32 & S) -> Certified<S, 1n, pair_join(~S, x, high)>:
  (y, rest) = high
  Certified{pair_join(~S, x, (y, rest)), {==}, (Unit{}, Unit{})}

def pair_certified(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> A & B -> Bool -> F32 & (A & B),
  +p: P, +count: Nat, +base: Nat, low: F32 & (A & B)) -> Certified<(A & B), 1n, pair(~P, ~A, ~B, ~cell, p, count, base, low)>:
  (x, state) = low
  (l, r) = state
  pair_joined(~(A & B), x, cell(p, Nat.add(1n, base), (l, r), Nat.is_lt(Nat.add(1n, base), count)))

def tree_certified(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> A & B -> Bool -> F32 & (A & B),
  +p: P, +depth: Nat, +count: Nat, +base: Nat, state: A & B) ->
  Certified<(A & B), depth, tree(~P, ~A, ~B, ~cell, p, depth, count, base, state)>:
  match depth:
    case 0n:
      leaf_certified(~(A & B), cell(p, base, state, Nat.is_lt(base, count)))
    case 1n+inner:
      match inner:
        case 0n:
          pair_certified(~P, ~A, ~B, ~cell, p, count, base, cell(p, base, state, Nat.is_lt(base, count)))
        case 1n+below:
          node_certified(~A, ~B, inner, l => r => tree(~P, ~A, ~B, ~cell, p, inner, count, upper(inner, base), (l, r)),
            tree(~P, ~A, ~B, ~cell, p, inner, count, base, state),
            tree_certified(~P, ~A, ~B, ~cell, p, inner, count, base, state),
            l => r => tree_certified(~P, ~A, ~B, ~cell, p, inner, count, upper(inner, base), (l, r)))

# The one-thread tree is balanced at its depth.
def tree_balanced(~P: Data, ~A: Type, ~B: Type, ~cell: @+p: P -> @+position: Nat -> A & B -> Bool -> F32 & (A & B),
  +p: P, +depth: Nat, +count: Nat, +base: Nat, -state: A & B,
  certified: Certified<(A & B), depth, tree(~P, ~A, ~B, ~cell, p, depth, count, base, state)>) ->
  Model.Balanced(F32, depth, Pair.fst(Array<F32>, A & B, tree(~P, ~A, ~B, ~cell, p, depth, count, base, state))):
  Certified{result, same, balanced} = certified
  %same : Model.Balanced(F32, depth, Pair.fst(Array<F32>, A & B, _))
  balanced

def fork_balanced(~A: Type, ~B: Type, +inner: Nat, pair: (A & B) & (A & B), -low: A -> B -> Array<F32>, -high: A -> B -> Array<F32>,
  low_valid: @l: A -> @r: B -> Model.Balanced(F32, inner, low(l, r)),
  high_valid: @l: A -> @r: B -> Model.Balanced(F32, inner, high(l, r))) ->
  Model.Balanced(F32, 1n+inner, fork(~A, ~B, pair, low, high)):
  (first, second) = pair
  (l1, r1) = first
  (l2, r2) = second
  (low_valid(l1, r1), high_valid(l2, r2))

# The parallel build returns a balanced tree of its depth, whatever the
# split. It is the certificate the output storage carries, erased at runtime.
def balanced(~Q: Data, ~P: Data, ~A: Type, ~B: Type, ~prepare: @+q: Q -> P, ~split: A & B -> (A & B) & (A & B),
  ~cell: @+p: P -> @+position: Nat -> A & B -> Bool -> F32 & (A & B),
  +q: Q, +depth: Nat, +splits: Nat, +count: Nat, +base: Nat, state: A & B) ->
  Model.Balanced(F32, depth, build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, depth, splits, count, base, state)):
  match depth splits:
    case 1n+inner 1n+rest:
      +high = upper(inner, base)
      fork_balanced(~A, ~B, inner, split(state),
        l => r => build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, inner, rest, count, base, (l, r)),
        l => r => build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, inner, rest, count, high, (l, r)),
        l => r => balanced(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, inner, rest, count, base, (l, r)),
        l => r => balanced(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, inner, rest, count, high, (l, r)))
    case 0n _:
      tree_balanced(~P, ~A, ~B, ~cell, prepare(q), 0n, count, base, state,
        tree_certified(~P, ~A, ~B, ~cell, prepare(q), 0n, count, base, state))
    case 1n+inner 0n:
      tree_balanced(~P, ~A, ~B, ~cell, prepare(q), 1n+inner, count, base, state,
        tree_certified(~P, ~A, ~B, ~cell, prepare(q), 1n+inner, count, base, state))
