import Base
import ./core.bend as Shape
import ./nat_proofs.bend as Nats
import ./LAWS.bend as Laws

def within(-dims: List<&2, Nat>, capacity: Shape.Capacity<dims>) -> Bool:
  match capacity:
    case Shape.Within{size, exact, bound}:
      True{}
    case Shape.Beyond{}:
      False{}

def bound_within(-dims: List<&2, Nat>, -size: Nat, -exact: {size == Shape.numel(dims) : Nat}, +decision: Shape.BoolCase<Nat.is_le(size, 1048576n)>, fits: {True{} == Nat.is_le(size, 1048576n) : Bool}) -> {True{} == within(dims, Shape.capacity.bound(dims, size, exact, decision)) : Bool}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({True{} == False{} : Bool}, Nats.contradiction(Nat.is_le(size, 1048576n), fits, equal))
    case Shape.BTrue{equal}:
      {==}

def scale_within(+head: Nat, -tail: List<&2, Nat>, +size: Nat, -exact: {size == Shape.numel(tail) : Nat}, +small: Shape.BoolCase<Bool.or(Nat.is_eq(size, 0n), Nat.is_le(head, 1048576n))>, fits_small: {True{} == Bool.or(Nat.is_eq(size, 0n), Nat.is_le(head, 1048576n)) : Bool}, fits_product: {True{} == Nat.is_le(Nat.mul(head, size), 1048576n) : Bool}) -> {True{} == within(head <> tail, Shape.capacity.scale(head, tail, size, exact, small)) : Bool}:
  match small:
    case Shape.BFalse{equal}:
      Empty.absurd({True{} == False{} : Bool}, Nats.contradiction(Bool.or(Nat.is_eq(size, 0n), Nat.is_le(head, 1048576n)), fits_small, equal))
    case Shape.BTrue{equal}:
      bound_within(head <> tail, Nat.mul(head, size), Equal.cong(Nat, Nat, value => Nat.mul(head, value), size, Shape.numel(tail), exact), Shape.bool_case(Nat.is_le(Nat.mul(head, size), 1048576n)), fits_product)

def scale_fits(+p: Nat, +size: Nat, bound: Shape.LE(Nat.mul(1n+p, size), 1048576n)) -> {True{} == Bool.or(Nat.is_eq(size, 0n), Nat.is_le(1n+p, 1048576n)) : Bool}:
  match size:
    case 0n:
      {==}
    case 1n+q:
      Nats.le_true(1n+p, 1048576n, Nats.le_add_left(1n+p, Nat.mul(1n+p, q), 1048576n, Nats.le_rewrite(Nat.add(1n+p, Nat.mul(1n+p, q)), Nat.mul(1n+p, 1n+q), 1048576n, Nats.mul_succ_right(1n+p, q), bound)))

def zero_small(+size: Nat) -> {True{} == Bool.or(Nat.is_eq(size, 0n), Nat.is_le(0n, 1048576n)) : Bool}:
  match size:
    case 0n:
      {==}
    case 1n+q:
      {==}

def zero_within(-tail: List<&2, Nat>, +rest: Shape.Capacity<tail>) -> {True{} == within(0n <> tail, Shape.capacity.step(0n, tail, rest)) : Bool}:
  match rest:
    case Shape.Beyond{}:
      {==}
    case Shape.Within{+size, exact, bound}:
      scale_within(0n, tail, size, exact, Shape.bool_case(Bool.or(Nat.is_eq(size, 0n), Nat.is_le(0n, 1048576n))), zero_small(size), {==})

def step_within(+p: Nat, +tail: List<&2, Nat>, +rest: Shape.Capacity<tail>, ih: {True{} == within(tail, rest) : Bool}, bound: Shape.LE(Nat.mul(1n+p, Shape.numel(tail)), 1048576n)) -> {True{} == within((1n+p) <> tail, Shape.capacity.step(1n+p, tail, rest)) : Bool}:
  match rest:
    case Shape.Beyond{}:
      Empty.absurd({True{} == within((1n+p) <> tail, Shape.capacity.step(1n+p, tail, Shape.Beyond{})) : Bool}, Shape.clash(ih))
    case Shape.Within{+size, +exact, fits}:
      +scaled = Nats.le_rewrite(Nat.mul(1n+p, size), Nat.mul(1n+p, Shape.numel(tail)), 1048576n, Equal.cong(Nat, Nat, value => Nat.mul(1n+p, value), Shape.numel(tail), size, Equal.sym(Nat, size, Shape.numel(tail), exact)), bound)
      scale_within(1n+p, tail, size, exact, Shape.bool_case(Bool.or(Nat.is_eq(size, 0n), Nat.is_le(1n+p, 1048576n))), scale_fits(p, size, scaled), Nats.le_true(Nat.mul(1n+p, size), 1048576n, scaled))

def capacity_complete(+dims: List<&2, Nat>, +bound: Shape.LE(Shape.numel(dims), 1048576n)) -> {True{} == within(dims, Shape.capacity(dims)) : Bool}:
  match dims:
    case Nil{}:
      {==}
    case +head <> +tail:
      match head:
        case 0n:
          zero_within(tail, Shape.capacity(tail))
        case 1n+p:
          step_within(p, tail, Shape.capacity(tail), capacity_complete(tail, Nats.le_add_left(Shape.numel(tail), Nat.mul(p, Shape.numel(tail)), 1048576n, bound)), bound)

def capacity_built(+input: List<&2, U32>, rank_bound: Shape.LE(Shape.rank_dimensions(input), 4n), +capacity: Shape.Capacity<Shape.naturals(input)>, fits: {True{} == within(Shape.naturals(input), capacity) : Bool}) -> {Laws.built(input, Shape.construct.capacity(input, rank_bound, capacity)) == Some{input} : Maybe<&2, List<&2, U32>>}:
  match capacity:
    case Shape.Beyond{}:
      Empty.absurd({Laws.built(input, Shape.construct.capacity(input, rank_bound, Shape.Beyond{})) == Some{input} : Maybe<&2, List<&2, U32>>}, Shape.clash(fits))
    case Shape.Within{size, exact, bound}:
      {==}

def construction_complete(+input: List<&2, U32>, +decision: Shape.BoolCase<Nat.is_le(Shape.rank_dimensions(input), 4n)>, rank_fits: {True{} == Nat.is_le(Shape.rank_dimensions(input), 4n) : Bool}, capacity_bound: Shape.LE(Shape.logical_numel(input), 1048576n)) -> {Laws.built(input, Shape.construct.ranked(input, decision)) == Some{input} : Maybe<&2, List<&2, U32>>}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({Laws.built(input, Shape.construct.ranked(input, Shape.BFalse{equal})) == Some{input} : Maybe<&2, List<&2, U32>>}, Nats.contradiction(Nat.is_le(Shape.rank_dimensions(input), 4n), rank_fits, equal))
    case Shape.BTrue{equal}:
      capacity_built(input, Shape.LE.from(Shape.rank_dimensions(input), 4n, equal), Shape.capacity(Shape.naturals(input)), capacity_complete(Shape.naturals(input), capacity_bound))
