import Base
import ../shape/core.bend as Shape

def first(-A: Data, values: List<&2, A>) -> Maybe<&2, A>:
  match values:
    case Nil{}:
      None{}
    case head <> tail:
      Some{head}

def lookup(-A: Data, values: List<&2, A>, index: Nat) -> Maybe<&2, A>:
  first(A, List.drop(&2, A, values, index))

def prepend(-A: Data, head: Maybe<&2, A>, tail: Maybe<&2, List<&2, A>>) -> Maybe<&2, List<&2, A>>:
  match head tail:
    case Some{value} Some{values}:
      Some{value <> values}
    case _ _:
      None{}

def gather(-A: Data, +values: List<&2, A>, axes: List<&2, Nat>) -> Maybe<&2, List<&2, A>>:
  match axes:
    case Nil{}:
      Some{Nil{}}
    case axis <> tail:
      prepend(A, lookup(A, values, axis), gather(A, values, tail))

def position_next(value: Maybe<&2, Nat>) -> Maybe<&2, Nat>:
  match value:
    case None{}:
      None{}
    case Some{index}:
      Some{1n+index}

def position_decided(decision: Bool, rec: Maybe<&2, Nat>) -> Maybe<&2, Nat>:
  match decision:
    case True{}:
      Some{0n}
    case False{}:
      position_next(rec)

def position(axes: List<&2, Nat>, +axis: Nat) -> Maybe<&2, Nat>:
  match axes:
    case Nil{}:
      None{}
    case head <> tail:
      position_decided(Nat.is_eq(head, axis), position(tail, axis))

def select(-A: Data, +values: List<&2, A>, index: Maybe<&2, Nat>) -> Maybe<&2, A>:
  match index:
    case None{}:
      None{}
    case Some{value}:
      lookup(A, values, value)

def scatter(-A: Data, rank: Nat, +axes: List<&2, Nat>, +coordinates: List<&2, A>, +cursor: Nat) -> Maybe<&2, List<&2, A>>:
  match rank:
    case 0n:
      Some{Nil{}}
    case 1n+p:
      prepend(A, select(A, coordinates, position(axes, cursor)), scatter(A, p, axes, coordinates, 1n+cursor))

def indices(rank: Nat, +cursor: Nat) -> List<&2, Nat>:
  match rank:
    case 0n:
      Nil{}
    case 1n+p:
      cursor <> indices(p, 1n+cursor)

def inverse(+rank: Nat, +axes: List<&2, Nat>) -> Maybe<&2, List<&2, Nat>>:
  scatter(Nat, rank, axes, indices(rank, 0n), 0n)

def values(-A: Data, result: Maybe<&2, List<&2, A>>) -> List<&2, A>:
  match result:
    case None{}:
      Nil{}
    case Some{items}:
      items

def order(-A: Data, +items: List<&2, A>, axes: List<&2, Nat>) -> List<&2, A>:
  values(A, gather(A, items, axes))

def inverse_axes(+rank: Nat, axes: List<&2, Nat>) -> List<&2, Nat>:
  values(Nat, inverse(rank, axes))

def contains(+axis: Nat, axes: List<&2, Nat>) -> Bool:
  match axes:
    case Nil{}:
      False{}
    case head <> tail:
      Bool.or(Nat.is_eq(axis, head), contains(axis, tail))

def unique(axes: List<&2, Nat>) -> Bool:
  match axes:
    case Nil{}:
      True{}
    case +head <> +tail:
      Bool.and(Bool.not(contains(head, tail)), unique(tail))

def bounded(axes: List<&2, Nat>, +rank: Nat) -> Bool:
  match axes:
    case Nil{}:
      True{}
    case head <> tail:
      Bool.and(Nat.is_lt(head, rank), bounded(tail, rank))

def valid(+rank: Nat, +axes: List<&2, Nat>) -> Bool:
  Bool.and(Nat.is_eq(List.length(&2, Nat, axes), rank), Bool.and(bounded(axes, rank), unique(axes)))
