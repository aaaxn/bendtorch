import Base
import ../shape/core.bend as Shape
import ./axis_spec.bend as Axis

def axis_limit(rank: Nat) -> Nat:
  match rank:
    case 0n:
      1n
    case 1n+p:
      1n+p

def axes_valid(+rank: Nat, +a: Nat, +b: Nat) -> Bool:
  Bool.and(Nat.is_lt(a, axis_limit(rank)), Nat.is_lt(b, axis_limit(rank)))

def valid(+shape: Shape.Shape, +a: U32, +b: U32) -> Bool:
  axes_valid(Shape.rank_dimensions(Shape.dimensions(shape)), U32.to_nat(a), U32.to_nat(b))

def replace(-A: Data, items: List<&2, A>, index: Nat, +value: A) -> List<&2, A>:
  match items index:
    case Nil{} index:
      Nil{}
    case head <> tail 0n:
      value <> tail
    case head <> tail 1n+p:
      head <> replace(A, tail, p, value)

def swap_selected(-A: Data, +items: List<&2, A>, +a: Nat, +b: Nat, left: Maybe<&2, A>, right: Maybe<&2, A>) -> List<&2, A>:
  match left right:
    case Some{x} Some{y}:
      replace(A, replace(A, items, a, y), b, x)
    case _ _:
      items

def swap(-A: Data, +items: List<&2, A>, +a: Nat, +b: Nat) -> List<&2, A>:
  left right = Axis.lookup(A, items, a) Axis.lookup(A, items, b)
  swap_selected(A, items, a, b, left, right)

def natural_axes(+rank: Nat, +a: Nat, +b: Nat) -> List<&2, Nat>:
  swap(Nat, Axis.indices(rank, 0n), a, b)

def requested_axes(+rank: Nat, +a: Nat, +b: Nat) -> List<&2, U32>:
  Shape.words(natural_axes(rank, a, b))

# The permutation `transpose_copy(input, a, b)` runs: the identity of the
# input's rank with axes a and b swapped.
def swapped_axes(+input: Shape.Shape, +a: U32, +b: U32) -> List<&2, U32>:
  requested_axes(Shape.rank_dimensions(Shape.dimensions(input)), U32.to_nat(a), U32.to_nat(b))
