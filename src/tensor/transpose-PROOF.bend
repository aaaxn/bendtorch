import Base
import ../shape/core.bend as Shape
import ../storage/arith.bend as Arith
import ./axis-spec.bend as Axis
import ./axis-algebra.bend as Algebra
import ./axis-PROOF.bend as AxisProof
import ./layout.bend as Layout
import ./word-conversion.bend as Conversion
import ./transpose-spec.bend as Transpose
import ./transpose-LAWS.bend as Laws
import ./observation.bend as Values
import ./matmul.bend as Matrix
import ./matrix-index-proof.bend as MatrixIndex
import ../shape/nat_proofs.bend as Nats

def rank_bound(+shape: Shape.Shape) -> Shape.LE(Shape.rank_dimensions(Shape.dimensions(shape)), 4n):
  match shape:
    case Shape.SShape{dims, size, exact, bound, capacity}:
      bound

def inside_natural(+shape: Shape.Shape, +a: U32, +b: U32,
  +inside: {True{} == Transpose.valid(shape, a, b) : Bool}) ->
  {True{} == Transpose.axes_valid(Shape.rank_dimensions(Shape.dimensions(shape)), U32.to_nat(a), U32.to_nat(b)) : Bool}:
  inside

type AxesFacts<-dims: List<&2, U32>, -a: Nat, -b: Nat> is Data:
  Certified{requested: {Layout.transpose_axes(dims, U32.from_nat(a), U32.from_nat(b), 0) == Transpose.requested_axes(Shape.rank_dimensions(dims), a, b) : List<&2, U32>},
    valid: {True{} == Axis.valid(Shape.rank_dimensions(dims), Shape.naturals(Transpose.requested_axes(Shape.rank_dimensions(dims), a, b))) : Bool},
    inverse: {Axis.inverse_axes(Shape.rank_dimensions(dims), Shape.naturals(Transpose.requested_axes(Shape.rank_dimensions(dims), a, b))) == Shape.naturals(Transpose.requested_axes(Shape.rank_dimensions(dims), a, b)) : List<&2, Nat>}}

def axes_facts(+dims: List<&2, U32>, +a: Nat, +b: Nat, +bound: Shape.LE(Shape.rank_dimensions(dims), 4n),
  +inside: {True{} == Transpose.axes_valid(Shape.rank_dimensions(dims), a, b) : Bool}) -> AxesFacts<dims, a, b>:
  match dims a b:
    case Nil{} 0n 0n:
      Certified{{==}, {==}, {==}}
    case Nil{} 1n+p b:
      Empty.absurd(AxesFacts<Nil{}, 1n+p, b>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(1n+p, 1n), Nat.is_lt(b, 1n), Equal.sym(Bool, True{}, Bool.and(Nat.is_lt(1n+p, 1n), Nat.is_lt(b, 1n)), inside))))
    case Nil{} a 1n+p:
      Empty.absurd(AxesFacts<Nil{}, a, 1n+p>, Arith.lt_zero(p, Arith.and_right(Nat.is_lt(a, 1n), Nat.is_lt(1n+p, 1n), Equal.sym(Bool, True{}, Bool.and(Nat.is_lt(a, 1n), Nat.is_lt(1n+p, 1n)), inside))))
    case +d0 <> Nil{} 0n 0n:
      Certified{{==}, {==}, {==}}
    case +d0 <> Nil{} 1n+p b:
      Empty.absurd(AxesFacts<d0 <> Nil{}, 1n+p, b>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(1n+p, 1n), Nat.is_lt(b, 1n), Equal.sym(Bool, True{}, Bool.and(Nat.is_lt(1n+p, 1n), Nat.is_lt(b, 1n)), inside))))
    case +d0 <> Nil{} a 1n+p:
      Empty.absurd(AxesFacts<d0 <> Nil{}, a, 1n+p>, Arith.lt_zero(p, Arith.and_right(Nat.is_lt(a, 1n), Nat.is_lt(1n+p, 1n), Equal.sym(Bool, True{}, Bool.and(Nat.is_lt(a, 1n), Nat.is_lt(1n+p, 1n)), inside))))
    case +d0 <> +d1 <> Nil{} 0n 0n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> Nil{} 0n 1n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> Nil{} 1n 0n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> Nil{} 1n 1n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> Nil{} 2n+p b:
      Empty.absurd(AxesFacts<d0 <> d1 <> Nil{}, 2n+p, b>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(2n+p, 2n), Nat.is_lt(b, 2n), Equal.sym(Bool, True{}, Bool.and(Nat.is_lt(2n+p, 2n), Nat.is_lt(b, 2n)), inside))))
    case +d0 <> +d1 <> Nil{} a 2n+p:
      Empty.absurd(AxesFacts<d0 <> d1 <> Nil{}, a, 2n+p>, Arith.lt_zero(p, Arith.and_right(Nat.is_lt(a, 2n), Nat.is_lt(2n+p, 2n), Equal.sym(Bool, True{}, Bool.and(Nat.is_lt(a, 2n), Nat.is_lt(2n+p, 2n)), inside))))
    case +d0 <> +d1 <> +d2 <> Nil{} 0n 0n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> Nil{} 0n 1n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> Nil{} 0n 2n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> Nil{} 1n 0n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> Nil{} 1n 1n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> Nil{} 1n 2n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> Nil{} 2n 0n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> Nil{} 2n 1n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> Nil{} 2n 2n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> Nil{} 3n+p b:
      Empty.absurd(AxesFacts<d0 <> d1 <> d2 <> Nil{}, 3n+p, b>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(3n+p, 3n), Nat.is_lt(b, 3n), Equal.sym(Bool, True{}, Bool.and(Nat.is_lt(3n+p, 3n), Nat.is_lt(b, 3n)), inside))))
    case +d0 <> +d1 <> +d2 <> Nil{} a 3n+p:
      Empty.absurd(AxesFacts<d0 <> d1 <> d2 <> Nil{}, a, 3n+p>, Arith.lt_zero(p, Arith.and_right(Nat.is_lt(a, 3n), Nat.is_lt(3n+p, 3n), Equal.sym(Bool, True{}, Bool.and(Nat.is_lt(a, 3n), Nat.is_lt(3n+p, 3n)), inside))))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} 0n 0n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} 0n 1n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} 0n 2n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} 0n 3n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} 1n 0n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} 1n 1n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} 1n 2n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} 1n 3n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} 2n 0n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} 2n 1n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} 2n 2n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} 2n 3n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} 3n 0n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} 3n 1n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} 3n 2n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} 3n 3n:
      Certified{{==}, {==}, {==}}
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} 4n+p b:
      Empty.absurd(AxesFacts<d0 <> d1 <> d2 <> d3 <> Nil{}, 4n+p, b>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Nat.is_lt(b, 4n), Equal.sym(Bool, True{}, Bool.and(Nat.is_lt(4n+p, 4n), Nat.is_lt(b, 4n)), inside))))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{} a 4n+p:
      Empty.absurd(AxesFacts<d0 <> d1 <> d2 <> d3 <> Nil{}, a, 4n+p>, Arith.lt_zero(p, Arith.and_right(Nat.is_lt(a, 4n), Nat.is_lt(4n+p, 4n), Equal.sym(Bool, True{}, Bool.and(Nat.is_lt(a, 4n), Nat.is_lt(4n+p, 4n)), inside))))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +tail a b:
      match bound:

def requested(-dims: List<&2, U32>, -a: Nat, -b: Nat, certificate: AxesFacts<dims, a, b>) ->
  {Layout.transpose_axes(dims, U32.from_nat(a), U32.from_nat(b), 0) == Transpose.requested_axes(Shape.rank_dimensions(dims), a, b) : List<&2, U32>}:
  Certified{requested, valid, inverse} = certificate
  requested

def requested_valid(-dims: List<&2, U32>, -a: Nat, -b: Nat, certificate: AxesFacts<dims, a, b>) ->
  {True{} == Axis.valid(Shape.rank_dimensions(dims), Shape.naturals(Transpose.requested_axes(Shape.rank_dimensions(dims), a, b))) : Bool}:
  Certified{requested, valid, inverse} = certificate
  valid

def requested_inverse(-dims: List<&2, U32>, -a: Nat, -b: Nat, certificate: AxesFacts<dims, a, b>) ->
  {Axis.inverse_axes(Shape.rank_dimensions(dims), Shape.naturals(Transpose.requested_axes(Shape.rank_dimensions(dims), a, b))) == Shape.naturals(Transpose.requested_axes(Shape.rank_dimensions(dims), a, b)) : List<&2, Nat>}:
  Certified{requested, valid, inverse} = certificate
  inverse

def transpose_axes_requested(+shape: Shape.Shape, +a: U32, +b: U32,
  +inside: {True{} == Transpose.valid(shape, a, b) : Bool}) ->
  {Layout.transpose_axes(Shape.dimensions(shape), a, b, 0) == Transpose.requested_axes(Shape.rank_dimensions(Shape.dimensions(shape)), U32.to_nat(a), U32.to_nat(b)) : List<&2, U32>}:
  %Conversion.word_natural_exact(a) : {Layout.transpose_axes(Shape.dimensions(shape), _, b, 0) == Transpose.requested_axes(Shape.rank_dimensions(Shape.dimensions(shape)), U32.to_nat(a), U32.to_nat(b)) : List<&2, U32>}
  %Conversion.word_natural_exact(b) : {Layout.transpose_axes(Shape.dimensions(shape), U32.from_nat(U32.to_nat(a)), _, 0) == Transpose.requested_axes(Shape.rank_dimensions(Shape.dimensions(shape)), U32.to_nat(a), U32.to_nat(b)) : List<&2, U32>}
  requested(Shape.dimensions(shape), U32.to_nat(a), U32.to_nat(b), axes_facts(Shape.dimensions(shape), U32.to_nat(a), U32.to_nat(b), rank_bound(shape), inside_natural(shape, a, b, inside)))

def natural_valid_layout(+shape: Shape.Shape, +axes: List<&2, U32>, accepted: {True{} == Axis.valid(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes)) : Bool}) ->
  {True{} == Layout.bijective(shape, axes) : Bool}:
  %Equal.sym(Nat, Shape.rank_dimensions(axes), List.length(&2, Nat, Shape.naturals(axes)), Shape.rank_naturals(axes)) :
    {True{} == Bool.and(Nat.is_eq(_, Shape.rank_dimensions(Shape.dimensions(shape))), Bool.and(Layout.bounded(axes, Shape.rank(shape)), Layout.unique(axes))) : Bool}
  %Equal.sym(Bool, Layout.unique(axes), Axis.unique(Shape.naturals(axes)), Algebra.unique(axes)) :
    {True{} == Bool.and(Nat.is_eq(List.length(&2, Nat, Shape.naturals(axes)), Shape.rank_dimensions(Shape.dimensions(shape))), Bool.and(Layout.bounded(axes, Shape.rank(shape)), _)) : Bool}
  %Equal.sym(Bool, Layout.bounded(axes, Shape.rank(shape)), Axis.bounded(Shape.naturals(axes), U32.to_nat(Shape.rank(shape))), Algebra.bounded(axes, Shape.rank(shape))) :
    {True{} == Bool.and(Nat.is_eq(List.length(&2, Nat, Shape.naturals(axes)), Shape.rank_dimensions(Shape.dimensions(shape))), Bool.and(_, Axis.unique(Shape.naturals(axes)))) : Bool}
  %Equal.sym(Nat, U32.to_nat(Shape.rank(shape)), Shape.rank_dimensions(Shape.dimensions(shape)), Algebra.rank_exact(shape)) :
    {True{} == Bool.and(Nat.is_eq(List.length(&2, Nat, Shape.naturals(axes)), Shape.rank_dimensions(Shape.dimensions(shape))), Bool.and(Axis.bounded(Shape.naturals(axes), _), Axis.unique(Shape.naturals(axes)))) : Bool}
  accepted

def valid_axes(+shape: Shape.Shape, +a: U32, +b: U32,
  +inside: {True{} == Transpose.valid(shape, a, b) : Bool}) ->
  {True{} == Layout.bijective(shape, Layout.transpose_axes(Shape.dimensions(shape), a, b, 0)) : Bool}:
  %Equal.sym(List<&2, U32>, Layout.transpose_axes(Shape.dimensions(shape), a, b, 0), Transpose.requested_axes(Shape.rank_dimensions(Shape.dimensions(shape)), U32.to_nat(a), U32.to_nat(b)), transpose_axes_requested(shape, a, b, inside)) :
    {True{} == Layout.bijective(shape, _) : Bool}
  natural_valid_layout(shape, Transpose.requested_axes(Shape.rank_dimensions(Shape.dimensions(shape)), U32.to_nat(a), U32.to_nat(b)),
    requested_valid(Shape.dimensions(shape), U32.to_nat(a), U32.to_nat(b), axes_facts(Shape.dimensions(shape), U32.to_nat(a), U32.to_nat(b), rank_bound(shape), inside_natural(shape, a, b, inside))))

def inverse_axes(+shape: Shape.Shape, +a: U32, +b: U32,
  +inside: {True{} == Transpose.valid(shape, a, b) : Bool}) ->
  {Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(Layout.transpose_axes(Shape.dimensions(shape), a, b, 0))) == Shape.naturals(Layout.transpose_axes(Shape.dimensions(shape), a, b, 0)) : List<&2, Nat>}:
  %Equal.sym(List<&2, U32>, Layout.transpose_axes(Shape.dimensions(shape), a, b, 0), Transpose.requested_axes(Shape.rank_dimensions(Shape.dimensions(shape)), U32.to_nat(a), U32.to_nat(b)), transpose_axes_requested(shape, a, b, inside)) :
    {Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(_)) == Shape.naturals(_) : List<&2, Nat>}
  requested_inverse(Shape.dimensions(shape), U32.to_nat(a), U32.to_nat(b), axes_facts(Shape.dimensions(shape), U32.to_nat(a), U32.to_nat(b), rank_bound(shape), inside_natural(shape, a, b, inside)))

def Laws.transpose_axes_requested(shape, a, b, inside):
  transpose_axes_requested(shape, a, b, inside)

def Laws.transpose_axes_valid(shape, a, b, inside):
  valid_axes(shape, a, b, inside)

def Laws.transpose_axes_self_inverse(shape, a, b, inside):
  inverse_axes(shape, a, b, inside)

# What `permute_source(input, [1, 0], index)` computes once the input's
# dimensions are known.
def transposed_source(+dims: List<&2, U32>, +index: U32) -> Maybe<&2, U32>:
  Layout.reference_source_coordinates(dims,
    Axis.scatter(U32, Shape.rank_dimensions(dims), Shape.naturals([1, 0]),
      Shape.words(Shape.digits(Shape.naturals(Axis.order(U32, dims, Shape.naturals([1, 0]))), U32.to_nat(index))), 0n))

# Source of output digits (j, l) of a [columns, rows] transpose: the input
# offset l * columns + j, once both digits are words again.
def swapped_offset(+rows: U32, +columns: U32, +j: Nat, +l: Nat,
  +j_below: {Nat.is_lt(j, U32.to_nat(columns)) == True{} : Bool},
  +l_below: {Nat.is_lt(l, U32.to_nat(rows)) == True{} : Bool}) ->
  {Some{U32.from_nat(Nat.add(Nat.mul(U32.to_nat(U32.from_nat(l)), Nat.mul(U32.to_nat(columns), 1n)),
     Nat.add(Nat.mul(U32.to_nat(U32.from_nat(j)), 1n), 0n)))} ==
   Some{U32.from_nat(Nat.add(Nat.mul(l, U32.to_nat(columns)), j))} : Maybe<&2, U32>}:
  +c = U32.to_nat(columns)
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(l)), l, MatrixIndex.word_below(l, rows, l_below)) :
    {Some{U32.from_nat(Nat.add(Nat.mul(_, Nat.mul(c, 1n)), Nat.add(Nat.mul(U32.to_nat(U32.from_nat(j)), 1n), 0n)))} ==
      Some{U32.from_nat(Nat.add(Nat.mul(l, c), j))} : Maybe<&2, U32>}
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(j)), j, MatrixIndex.word_below(j, columns, j_below)) :
    {Some{U32.from_nat(Nat.add(Nat.mul(l, Nat.mul(c, 1n)), Nat.add(Nat.mul(_, 1n), 0n)))} ==
      Some{U32.from_nat(Nat.add(Nat.mul(l, c), j))} : Maybe<&2, U32>}
  %Equal.sym(Nat, Nat.mul(c, 1n), c, Nats.mul_one_right(c)) :
    {Some{U32.from_nat(Nat.add(Nat.mul(l, _), Nat.add(Nat.mul(j, 1n), 0n)))} ==
      Some{U32.from_nat(Nat.add(Nat.mul(l, c), j))} : Maybe<&2, U32>}
  %Equal.sym(Nat, Nat.mul(j, 1n), j, Nats.mul_one_right(j)) :
    {Some{U32.from_nat(Nat.add(Nat.mul(l, c), Nat.add(_, 0n)))} ==
      Some{U32.from_nat(Nat.add(Nat.mul(l, c), j))} : Maybe<&2, U32>}
  %Equal.sym(Nat, Nat.add(j, 0n), j, Conversion.add_zero(j)) :
    {Some{U32.from_nat(Nat.add(Nat.mul(l, c), _))} ==
      Some{U32.from_nat(Nat.add(Nat.mul(l, c), j))} : Maybe<&2, U32>}
  {==}

def matrix_source(+rows: U32, +columns: U32, +j: Nat, +l: Nat,
  +j_below: {Nat.is_lt(j, U32.to_nat(columns)) == True{} : Bool},
  +l_below: {Nat.is_lt(l, U32.to_nat(rows)) == True{} : Bool},
  +exact: {U32.to_nat(U32.from_nat(Nat.add(Nat.mul(j, U32.to_nat(rows)), l))) == Nat.add(Nat.mul(j, U32.to_nat(rows)), l) : Nat}) ->
  {transposed_source([rows, columns], U32.from_nat(Nat.add(Nat.mul(j, U32.to_nat(rows)), l))) ==
    Some{U32.from_nat(Nat.add(Nat.mul(l, U32.to_nat(columns)), j))} : Maybe<&2, U32>}:
  +r = U32.to_nat(rows)
  +c = U32.to_nat(columns)
  +i = Nat.add(Nat.mul(j, r), l)
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(i)), i, exact) :
    {Some{U32.from_nat(Nat.add(Nat.mul(U32.to_nat(U32.from_nat(Matrix.column_of(r, _))), Nat.mul(c, 1n)),
       Nat.add(Nat.mul(U32.to_nat(U32.from_nat(Matrix.row_of(r, U32.to_nat(U32.from_nat(i))))), 1n), 0n)))} ==
      Some{U32.from_nat(Nat.add(Nat.mul(l, c), j))} : Maybe<&2, U32>}
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(i)), i, exact) :
    {Some{U32.from_nat(Nat.add(Nat.mul(U32.to_nat(U32.from_nat(Matrix.column_of(r, i))), Nat.mul(c, 1n)),
       Nat.add(Nat.mul(U32.to_nat(U32.from_nat(Matrix.row_of(r, _))), 1n), 0n)))} ==
      Some{U32.from_nat(Nat.add(Nat.mul(l, c), j))} : Maybe<&2, U32>}
  %Equal.sym(Nat, Matrix.column_of(r, i), l, MatrixIndex.column_at(j, l, r, l_below)) :
    {Some{U32.from_nat(Nat.add(Nat.mul(U32.to_nat(U32.from_nat(_)), Nat.mul(c, 1n)),
       Nat.add(Nat.mul(U32.to_nat(U32.from_nat(Matrix.row_of(r, i))), 1n), 0n)))} ==
      Some{U32.from_nat(Nat.add(Nat.mul(l, c), j))} : Maybe<&2, U32>}
  %Equal.sym(Nat, Matrix.row_of(r, i), j, MatrixIndex.row_at(j, l, r, l_below)) :
    {Some{U32.from_nat(Nat.add(Nat.mul(U32.to_nat(U32.from_nat(l)), Nat.mul(c, 1n)),
       Nat.add(Nat.mul(U32.to_nat(U32.from_nat(_)), 1n), 0n)))} ==
      Some{U32.from_nat(Nat.add(Nat.mul(l, c), j))} : Maybe<&2, U32>}
  swapped_offset(rows, columns, j, l, j_below, l_below)

# j * rows + l < rows * columns, so the output index is its own word.
def transposed_index_exact(+input: Shape.Shape, +rows: U32, +columns: U32,
  +matrix: {Shape.dimensions(input) == [rows, columns] : List<&2, U32>}, +j: Nat,
  +j_below: {Nat.is_lt(j, U32.to_nat(columns)) == True{} : Bool}, +l: Nat,
  +l_below: {Nat.is_lt(l, U32.to_nat(rows)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(Nat.add(Nat.mul(j, U32.to_nat(rows)), l))) == Nat.add(Nat.mul(j, U32.to_nat(rows)), l) : Nat}:
  +r = U32.to_nat(rows)
  +c = U32.to_nat(columns)
  MatrixIndex.word_exact(Nat.add(Nat.mul(j, r), l), input,
    %Equal.sym(List<&2, U32>, Shape.dimensions(input), [rows, columns], matrix) :
      {Nat.is_lt(Nat.add(Nat.mul(j, r), l), Shape.logical_numel(_)) == True{} : Bool}
    %Equal.sym(Nat, Nat.mul(c, 1n), c, Nats.mul_one_right(c)) :
      {Nat.is_lt(Nat.add(Nat.mul(j, r), l), Nat.mul(r, _)) == True{} : Bool}
    %Nats.mul_comm(c, r) :
      {Nat.is_lt(Nat.add(Nat.mul(j, r), l), _) == True{} : Bool}
    MatrixIndex.lt_mul_add(j, c, l, r, j_below, l_below))

def permute_source_transposed(+input: Shape.Shape, +rows: U32, +columns: U32,
  +matrix: {Shape.dimensions(input) == [rows, columns] : List<&2, U32>}, +j: Nat,
  +j_below: {Nat.is_lt(j, U32.to_nat(columns)) == True{} : Bool}, +l: Nat,
  +l_below: {Nat.is_lt(l, U32.to_nat(rows)) == True{} : Bool}) ->
  {Values.permute_source(input, [1, 0], U32.from_nat(Nat.add(Nat.mul(j, U32.to_nat(rows)), l))) ==
    Some{U32.from_nat(Nat.add(Nat.mul(l, U32.to_nat(columns)), j))} : Maybe<&2, U32>}:
  +index = U32.from_nat(Nat.add(Nat.mul(j, U32.to_nat(rows)), l))
  %Equal.sym(List<&2, U32>, Shape.dimensions(input), [rows, columns], matrix) :
    {transposed_source(_, index) == Some{U32.from_nat(Nat.add(Nat.mul(l, U32.to_nat(columns)), j))} : Maybe<&2, U32>}
  matrix_source(rows, columns, j, l, j_below, l_below, transposed_index_exact(input, rows, columns, matrix, j, j_below, l, l_below))

def Laws.permute_source_transposed(input, rows, columns, matrix, j, j_below, l, l_below):
  permute_source_transposed(input, rows, columns, matrix, j, j_below, l, l_below)
