import Base
import ../storage/model.bend as Model
import ./parallel.bend as Parallel
import ./parallel_proof.bend as Proof
import ./parallel_LAWS.bend as Laws

def Laws.parallel_build_balanced(Q, P, A, B, prepare, split, cell, q, depth, splits, count, base, state):
  Parallel.balanced(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, depth, splits, count, base, state)

def Laws.parallel_build_cells(Q, P, A, B, prepare, split, cell, q, depth, splits, count, base, state, keeps, first, second):
  Laws.Holds{{\{Proof.Fact: q => q} :
    Proof.Fact<Proof.Cells(~P, ~A, ~B, ~cell, prepare(q), state, depth, count, base, Parallel.build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, depth, splits, count, base, state))> ->
    Proof.Cells(~P, ~A, ~B, ~cell, prepare(q), state, depth, count, base, Parallel.build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, depth, splits, count, base, state))}(
    Proof.build_cells(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, state, depth, splits, count, base, state, {==}, keeps, first, second))}

def Laws.parallel_build_sequential(Q, P, A, B, prepare, split, cell, q, depth, splits, count, base, state, keeps, first, second):
  Laws.Holds{path => Equal.trans(Maybe<&2, F32>,
    Model.observe(F32, depth, Parallel.build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, depth, splits, count, base, state), path),
    Some{Pair.fst(F32, (A & B), cell(prepare(q), Nat.add(Model.offset(depth, path), base), state, Nat.is_lt(Nat.add(Model.offset(depth, path), base), count)))},
    Model.observe(F32, depth, Parallel.build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, depth, 0n, count, base, state), path),
    {\{Proof.Fact: q => q} :
      Proof.Fact<Proof.Cells(~P, ~A, ~B, ~cell, prepare(q), state, depth, count, base, Parallel.build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, depth, splits, count, base, state))> ->
      Proof.Cells(~P, ~A, ~B, ~cell, prepare(q), state, depth, count, base, Parallel.build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, depth, splits, count, base, state))}(
      Proof.build_cells(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, state, depth, splits, count, base, state, {==}, keeps, first, second))(path),
    Equal.sym(Maybe<&2, F32>,
      Model.observe(F32, depth, Parallel.build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, depth, 0n, count, base, state), path),
      Some{Pair.fst(F32, (A & B), cell(prepare(q), Nat.add(Model.offset(depth, path), base), state, Nat.is_lt(Nat.add(Model.offset(depth, path), base), count)))},
      {\{Proof.Fact: q => q} :
        Proof.Fact<Proof.Cells(~P, ~A, ~B, ~cell, prepare(q), state, depth, count, base, Parallel.build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, depth, 0n, count, base, state))> ->
        Proof.Cells(~P, ~A, ~B, ~cell, prepare(q), state, depth, count, base, Parallel.build(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, depth, 0n, count, base, state))}(
        Proof.build_cells(~Q, ~P, ~A, ~B, ~prepare, ~split, ~cell, q, state, depth, 0n, count, base, state, {==}, keeps, first, second))(path)))}
