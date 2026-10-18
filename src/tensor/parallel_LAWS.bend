import Base
import ../storage/model.bend as Model
import ./parallel.bend as Parallel

# The laws of the one parallel build that elementwise, matmul and reduce_to
# share. An operation hands the build a cell function, a split of its input
# state and a `prepare` that turns the build's parameters `q` into what
# every cell reads; these laws hold for every such choice, so each operation
# proves only what its cell function computes.

type Holds<-P: Type> is Data:
  Holds{-proof: P}

# For every split, the parallel build is a balanced tree of its depth.
law parallel_build_balanced:
  for ~Q: Data
  for ~P: Data
  for ~A: Type
  for ~B: Type
  for ~prepare: @+q: Q -> P
  for ~split: A & B -> (A & B) & (A & B)
  for ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B)
  for +q: Q
  for +depth: Nat
  for +splits: Nat
  for +count: Nat
  for +base: Nat
  for state: (A & B)
  Model.Balanced(F32, depth, Parallel.build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, depth, splits, count, base, state))

# If each cell hands its input state back and the split copies the state
# into both halves, then for every split each slot of the build, cell
# base + offset, holds the F32 the cell function computes for it from the
# input state, the padding past `count` included.
law parallel_build_cells:
  for ~Q: Data
  for ~P: Data
  for ~A: Type
  for ~B: Type
  for ~prepare: @+q: Q -> P
  for ~split: A & B -> (A & B) & (A & B)
  for ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B)
  for +q: Q
  for +depth: Nat
  for +splits: Nat
  for +count: Nat
  for +base: Nat
  for state: (A & B)
  for keeps: @+position: Nat -> @inside: Bool -> {Pair.snd(F32, (A & B), cell(prepare(q), position, state, inside)) == state : (A & B)}
  for first: {Pair.fst((A & B), (A & B), split(state)) == state : (A & B)}
  for second: {Pair.snd((A & B), (A & B), split(state)) == state : (A & B)}
  Holds<@+path: Model.Path(depth) ->
    {Model.observe(F32, depth, Parallel.build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, depth, splits, count, base, state), path) ==
      Some{Pair.fst(F32, (A & B), cell(prepare(q), Nat.add(Model.offset(depth, path), base), state,
        Nat.is_lt(Nat.add(Model.offset(depth, path), base), count)))} : Maybe<&2, F32>}>

# Under the same two facts, the parallel build and the one-thread build
# (no split) hold the same F32 in every slot, so no output word depends on
# the split or on the thread count.
law parallel_build_sequential:
  for ~Q: Data
  for ~P: Data
  for ~A: Type
  for ~B: Type
  for ~prepare: @+q: Q -> P
  for ~split: A & B -> (A & B) & (A & B)
  for ~cell: @+p: P -> @+position: Nat -> (A & B) -> Bool -> F32 & (A & B)
  for +q: Q
  for +depth: Nat
  for +splits: Nat
  for +count: Nat
  for +base: Nat
  for state: (A & B)
  for keeps: @+position: Nat -> @inside: Bool -> {Pair.snd(F32, (A & B), cell(prepare(q), position, state, inside)) == state : (A & B)}
  for first: {Pair.fst((A & B), (A & B), split(state)) == state : (A & B)}
  for second: {Pair.snd((A & B), (A & B), split(state)) == state : (A & B)}
  Holds<@+path: Model.Path(depth) ->
    {Model.observe(F32, depth, Parallel.build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, depth, splits, count, base, state), path) ==
      Model.observe(F32, depth, Parallel.build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, depth, 0n, count, base, state), path) : Maybe<&2, F32>}>
