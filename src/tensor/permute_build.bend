import Base
import ../shape/core.bend as Shape
import ../storage/model.bend as Model
import ./axis_spec.bend as Axis
import ./parallel.bend as Parallel

# The permute copy as a parallel build. Output cell i holds the input word at
# the row-major index `Values.permute_source_index(input, axes, i)` names.
# That index decodes i into output coordinates, scatters them to the input
# axes and flattens them again, building three lists per cell. `recipe`
# works out what does not depend on i, once per subtree below the forks (as
# `Parallel.build` prepares it), so once per up to 2^15 cells, and `source`
# computes the same index from it with a few divisions per axis and no list.
# permute_build_proof.bend proves `source` equal to `permute_source_index`.

type Request is Data:
  Request{input: Shape.Shape, axes: List<&2, U32>}

# How to take one output coordinate out of a row-major index: reduce the
# index modulo each of `mods` in turn, then divide by `size`. Coordinate k
# of `Shape.digits(dims, x)` takes the element counts after the first k
# dimensions as `mods` and the count after dimension k as `size`.
type Pick is Data:
  Pick{mods: List<&2, Nat>, size: Nat}

def widened(+outer: Nat, pick: Pick) -> Pick:
  Pick{mods, size} = pick
  Pick{outer <> mods, size}

def widen(+outer: Nat, pick: Maybe<&2, Pick>) -> Maybe<&2, Pick>:
  match pick:
    case None{}:
      None{}
    case Some{found}:
      Some{widened(outer, found)}

# The pick of coordinate k of `Shape.digits(dims, x)`; None past the last
# dimension.
def pick_at(+dims: List<&2, Nat>, k: Nat) -> Maybe<&2, Pick>:
  match dims:
    case Nil{}:
      None{}
    case head <> tail:
      match k:
        case 0n:
          Some{Pick{Nil{}, Shape.numel(tail)}}
        case 1n+j:
          widen(Shape.numel(tail), pick_at(tail, j))

def located(+dims: List<&2, Nat>, position: Maybe<&2, Nat>) -> Maybe<&2, Pick>:
  match position:
    case None{}:
      None{}
    case Some{k}:
      pick_at(dims, k)

# One pick per input axis, from `cursor` on: the output coordinate that the
# axes send to it, as `Axis.scatter` selects it.
def picks(rank: Nat, +axes: List<&2, Nat>, +out: List<&2, Nat>, +cursor: Nat) -> Maybe<&2, List<&2, Pick>>:
  match rank:
    case 0n:
      Some{Nil{}}
    case 1n+p:
      Axis.prepend(Pick, located(out, Axis.position(axes, cursor)), picks(p, axes, out, 1n+cursor))

# The row-major stride of each dimension: the element count after it.
def strides(+dims: List<&2, Nat>) -> List<&2, Nat>:
  match dims:
    case Nil{}:
      Nil{}
    case head <> tail:
      Shape.numel(tail) <> strides(tail)

type Recipe is Data:
  Recipe{picks: Maybe<&2, List<&2, Pick>>, strides: List<&2, Nat>}

def recipe(+request: Request) -> Recipe:
  Request{+input, +axes} = request
  +dims = Shape.dimensions(input)
  Recipe{picks(Shape.rank_dimensions(dims), Shape.naturals(axes), Shape.naturals(Axis.order(U32, dims, Shape.naturals(axes))), 0n),
    strides(Shape.naturals(dims))}

def digit(+mods: List<&2, Nat>, +size: Nat, +x: Nat) -> Nat:
  match mods:
    case Nil{}:
      Nat.div(x, size)
    case +m <> rest:
      digit(rest, size, Nat.mod(x, m))

def coordinate(+pick: Pick, +x: Nat) -> Nat:
  Pick{+mods, +size} = pick
  digit(mods, size, x)

# `Shape.offset` of the scattered coordinates, each through U32 and back as
# `Shape.words` and `Shape.naturals` take it.
def offset(+strides: List<&2, Nat>, +picks: List<&2, Pick>, +x: Nat) -> Nat:
  match strides picks:
    case +stride <> rest +pick <> others:
      Nat.add(Nat.mul(U32.to_nat(U32.from_nat(coordinate(pick, x))), stride), offset(rest, others, x))
    case strides picks:
      0n

def found(result: Maybe<&2, List<&2, Pick>>, +strides: List<&2, Nat>, +index: Nat) -> Nat:
  match result:
    case None{}:
      0n
    case Some{+picks}:
      U32.to_nat(U32.from_nat(offset(strides, picks, U32.to_nat(U32.from_nat(index)))))

# The input index output cell `index` reads.
def source(+recipe: Recipe, +index: Nat) -> Nat:
  Recipe{+picks, +strides} = recipe
  found(picks, strides, index)

# The input word index output cell `position` reads.
def at(+recipe: Recipe, +position: Nat) -> U32:
  U32.from_nat(source(recipe, position))

def take(b: Array<F32>, pair: Array<F32> & F32) -> F32 & (Array<F32> & Array<F32>):
  (a, x) = pair
  (x, (a, b))

def read(+recipe: Recipe, +position: Nat, state: Array<F32> & Array<F32>) -> F32 & (Array<F32> & Array<F32>):
  (a, b) = state
  take(b, Array.get(F32, a, at(recipe, position)))

# The cell function of the build: the state is the input array and an empty
# array that only fills the second slot of the state pair.
def cell(+recipe: Recipe, +position: Nat, state: Array<F32> & Array<F32>, inside: Bool) -> F32 & (Array<F32> & Array<F32>):
  match inside:
    case False{}:
      (0.0, state)
    case True{}:
      read(recipe, position, state)

def halves(lp: Array<F32> & Array<F32>, rp: Array<F32> & Array<F32>) -> (Array<F32> & Array<F32>) & (Array<F32> & Array<F32>):
  (l1, l2) = lp
  (r1, r2) = rp
  ((l1, r1), (l2, r2))

# Each half of a fork gets its own clone of the input array.
def split(state: Array<F32> & Array<F32>) -> (Array<F32> & Array<F32>) & (Array<F32> & Array<F32>):
  (a, b) = state
  halves(Array.clone(F32, a), Array.clone(F32, b))

# A subtree below the forks holds 2^15 cells, as in the elementwise build.
def splits(+depth: Nat) -> Nat:
  Nat.sub(depth, 15n)

def build(+depth: Nat, +request: Request, +count: Nat, a: Array<F32>) -> Array<F32>:
  Parallel.build(~Request, ~Recipe, ~Array<F32>, ~Array<F32>, ~recipe, ~split, ~cell, request, depth, splits(depth), count, 0n,
    (a, Array.new(F32, 0n, 0.0)))

def balanced(+depth: Nat, +request: Request, +count: Nat, a: Array<F32>) -> Model.Balanced(F32, depth, build(depth, request, count, a)):
  Parallel.balanced(~Request, ~Recipe, ~Array<F32>, ~Array<F32>, ~recipe, ~split, ~cell, request, depth, splits(depth), count, 0n,
    (a, Array.new(F32, 0n, 0.0)))
