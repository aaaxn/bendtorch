import Base
import ../shape/core.bend as Shape
import ../shape/LAWS.bend as ShapeLaws
import ../shape/PROOF.bend as ShapeProof
import ../shape/boundary_proofs.bend as Boundary
import ../shape/coordinate_proofs.bend as Coordinates
import ../shape/nat_proofs.bend as Nats
import ../shape/index_proofs.bend as Index
import ../storage/arith.bend as Arith
import ../storage/api.bend as Api
import ./matmul.bend as Matrix
import ./word-conversion.bend as Conversion
import ./read-contract.bend as Reads

# Index facts about a rank-2 product: where an output cell sits, where its
# factors sit in the inputs, and that each of them is in bounds.

def word_exact(+number: Nat, +shape: Shape.Shape,
  below: {Nat.is_lt(number, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(number)) == number : Nat}:
  Conversion.natural_word_exact(number, Shape.count(shape),
    %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.logical_numel(Shape.dimensions(shape)), Reads.count_exact(shape)) :
      {Nat.is_le(number, _) == True{} : Bool}
    Conversion.lt_le(number, Shape.logical_numel(Shape.dimensions(shape)), below))

def word_inside(+number: Nat, +shape: Shape.Shape,
  +below: {Nat.is_lt(number, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(number)), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}:
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(number)), number, word_exact(number, shape, below)) :
    {Nat.is_lt(_, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}
  below

def lt_mul_add(+a: Nat, +m: Nat, +b: Nat, +n: Nat, a_below: {Nat.is_lt(a, m) == True{} : Bool},
  b_below: {Nat.is_lt(b, n) == True{} : Bool}) -> {Nat.is_lt(Nat.add(Nat.mul(a, n), b), Nat.mul(m, n)) == True{} : Bool}:
  match a m:
    case 0n 0n:
      Empty.absurd({Nat.is_lt(b, 0n) == True{} : Bool}, Arith.clash(a_below))
    case 0n 1n+q:
      Arith.lt_le_trans(b, n, Nat.add(n, Nat.mul(q, n)), b_below, Api.le_add_right(n, Nat.mul(q, n)))
    case 1n+p 0n:
      Empty.absurd({Nat.is_lt(Nat.add(Nat.add(n, Nat.mul(p, n)), b), 0n) == True{} : Bool}, Arith.clash(a_below))
    case 1n+p 1n+q:
      %Index.add_assoc(n, Nat.mul(p, n), b) : {Nat.is_lt(_, Nat.add(n, Nat.mul(q, n))) == True{} : Bool}
      Api.lt_add_left(n, Nat.add(Nat.mul(p, n), b), Nat.mul(q, n), lt_mul_add(p, q, b, n, a_below, b_below))

# Factor k of cell (row, column) lies inside each input.
def left_below(+rows: Nat, +inner: Nat, +row: Nat, row_below: {Nat.is_lt(row, rows) == True{} : Bool}, +left: Shape.Shape,
  left_extents: {Shape.extents(left) == [rows, inner] : List<&2, Nat>}, +k: Nat, below: {Nat.is_lt(k, inner) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(row, inner), k), Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool}:
  %Equal.sym(List<&2, Nat>, Shape.extents(left), [rows, inner], left_extents) :
    {Nat.is_lt(Nat.add(Nat.mul(row, inner), k), Shape.numel(_)) == True{} : Bool}
  %Equal.sym(Nat, Nat.mul(inner, 1n), inner, Nats.mul_one_right(inner)) :
    {Nat.is_lt(Nat.add(Nat.mul(row, inner), k), Nat.mul(rows, _)) == True{} : Bool}
  lt_mul_add(row, rows, k, inner, row_below, below)

def right_below(+inner: Nat, +columns: Nat, +column: Nat, column_below: {Nat.is_lt(column, columns) == True{} : Bool}, +right: Shape.Shape,
  right_extents: {Shape.extents(right) == [inner, columns] : List<&2, Nat>}, +k: Nat, below: {Nat.is_lt(k, inner) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(k, columns), column), Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool}:
  %Equal.sym(List<&2, Nat>, Shape.extents(right), [inner, columns], right_extents) :
    {Nat.is_lt(Nat.add(Nat.mul(k, columns), column), Shape.numel(_)) == True{} : Bool}
  %Equal.sym(Nat, Nat.mul(columns, 1n), columns, Nats.mul_one_right(columns)) :
    {Nat.is_lt(Nat.add(Nat.mul(k, columns), column), Nat.mul(inner, _)) == True{} : Bool}
  lt_mul_add(k, inner, column, columns, below, column_below)

def lt_from(+a: Nat, +b: Nat, below: {Nat.is_lt(a, b) == True{} : Bool}) -> Shape.LT(a, b):
  match a b:
    case 0n 0n:
      Empty.absurd(Shape.LT(0n, 0n), Arith.clash(below))
    case 0n 1n+q:
      Unit{}
    case 1n+p 0n:
      Empty.absurd(Shape.LT(1n+p, 0n), Arith.clash(below))
    case 1n+p 1n+q:
      lt_from(p, q, below)

def lt_same(-a: Nat, -b: Nat, -bound: Nat, same: {a == b : Nat}, below: {Nat.is_lt(b, bound) == True{} : Bool}) ->
  {Nat.is_lt(a, bound) == True{} : Bool}:
  %Equal.sym(Nat, a, b, same) : {Nat.is_lt(_, bound) == True{} : Bool}
  below

def lt_bound(-a: Nat, -bound: Nat, -other: Nat, same: {bound == other : Nat}, below: {Nat.is_lt(a, bound) == True{} : Bool}) ->
  {Nat.is_lt(a, other) == True{} : Bool}:
  %same : {Nat.is_lt(a, _) == True{} : Bool}
  below

def word_below(+number: Nat, +limit: U32, below: {Nat.is_lt(number, U32.to_nat(limit)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(number)) == number : Nat}:
  Conversion.natural_word_exact(number, limit, Conversion.lt_le(number, U32.to_nat(limit), below))

def some_equal(-A: Data, -x: A, -y: A, equal: {Some{x} == Some{y} : Maybe<&2, A>}) -> {x == y : A}:
  Equal.cong(Maybe<&2, A>, A, found => Maybe.default(&2, A, found, x), Some{x}, Some{y}, equal)

def div_one(+number: Nat) -> {Nat.div(number, 1n) == number : Nat}:
  %Equal.trans(Nat, Nat.add(Nat.mul(number, 1n), 0n), Nat.mul(number, 1n), number,
    Conversion.add_zero(Nat.mul(number, 1n)), Nats.mul_one_right(number)) : {Nat.div(_, 1n) == number : Nat}
  Equal.cong(Nat & Nat, Nat, pair => Pair.fst(Nat, Nat, pair), Nat.divmod(Nat.add(Nat.mul(number, 1n), 0n), 1n), (number, 0n),
    Equal.sym(Nat & Nat, (number, 0n), Nat.divmod(Nat.add(Nat.mul(number, 1n), 0n), 1n), Nats.divmod_exact(number, 1n, 0n, Unit{})))

def row_at(+row: Nat, +column: Nat, +columns: Nat, below: {Nat.is_lt(column, columns) == True{} : Bool}) ->
  {Matrix.row_of(columns, Nat.add(Nat.mul(row, columns), column)) == row : Nat}:
  %Equal.sym(Nat, Nat.mul(columns, 1n), columns, Nats.mul_one_right(columns)) :
    {Nat.div(Nat.add(Nat.mul(row, columns), column), _) == row : Nat}
  Equal.cong(Nat & Nat, Nat, pair => Pair.fst(Nat, Nat, pair), Nat.divmod(Nat.add(Nat.mul(row, columns), column), columns), (row, column),
    Equal.sym(Nat & Nat, (row, column), Nat.divmod(Nat.add(Nat.mul(row, columns), column), columns),
      Nats.divmod_exact(row, columns, column, lt_from(column, columns, below))))

def column_at(+row: Nat, +column: Nat, +columns: Nat, below: {Nat.is_lt(column, columns) == True{} : Bool}) ->
  {Matrix.column_of(columns, Nat.add(Nat.mul(row, columns), column)) == column : Nat}:
  %Equal.sym(Nat, Nat.mul(columns, 1n), columns, Nats.mul_one_right(columns)) :
    {Nat.div(Nat.mod(Nat.add(Nat.mul(row, columns), column), _), 1n) == column : Nat}
  %Equal.cong(Nat & Nat, Nat, pair => Pair.snd(Nat, Nat, pair), (row, column), Nat.divmod(Nat.add(Nat.mul(row, columns), column), columns),
    Nats.divmod_exact(row, columns, column, lt_from(column, columns, below))) : {Nat.div(_, 1n) == column : Nat}
  div_one(column)

def position_digits(+output: Shape.Shape, +rows: Nat, +columns: Nat,
  extents: {Shape.extents(output) == [rows, columns] : List<&2, Nat>}, +position: Nat,
  +fits: {True{} == Nat.is_lt(position, Shape.numel(Shape.extents(output))) : Bool}) ->
  {Shape.indices(Shape.extents(output), Shape.unflatten(Shape.extents(output),
    Coordinates.fin_from(Shape.numel(Shape.extents(output)), position, fits))) ==
    [Matrix.row_of(columns, position), Matrix.column_of(columns, position)] : List<&2, Nat>}:
  +bound = Shape.numel(Shape.extents(output))
  +index = Coordinates.fin_from(bound, position, fits)
  Equal.trans(List<&2, Nat>, Shape.indices(Shape.extents(output), Shape.unflatten(Shape.extents(output), index)),
    Shape.digits(Shape.extents(output), Shape.Fin.value(bound, index)), [Matrix.row_of(columns, position), Matrix.column_of(columns, position)],
    Coordinates.digits_unflatten(Shape.extents(output), index),
    Equal.trans(List<&2, Nat>, Shape.digits(Shape.extents(output), Shape.Fin.value(bound, index)),
      Shape.digits(Shape.extents(output), position), [Matrix.row_of(columns, position), Matrix.column_of(columns, position)],
      Equal.cong(Nat, List<&2, Nat>, value => Shape.digits(Shape.extents(output), value), Shape.Fin.value(bound, index), position,
        Coordinates.fin_from_value(bound, position, fits)),
      Equal.cong(List<&2, Nat>, List<&2, Nat>, dims => Shape.digits(dims, position), Shape.extents(output), [rows, columns], extents)))

def unflatten_at(+output: Shape.Shape, +rows: Nat, +columns: Nat,
  extents: {Shape.extents(output) == [rows, columns] : List<&2, Nat>}, +position: Nat,
  +below: {Nat.is_lt(position, Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}) ->
  {ShapeLaws.coordinate_result(output, Shape.unflatten_checked(output, U32.from_nat(position))) ==
    Some{[U32.from_nat(Matrix.row_of(columns, position)), U32.from_nat(Matrix.column_of(columns, position))]} : Maybe<&2, List<&2, U32>>}:
  +bound = Shape.numel(Shape.extents(output))
  +fits = Equal.sym(Bool, Nat.is_lt(position, bound), True{}, below)
  +index = Coordinates.fin_from(bound, position, fits)
  Equal.trans(Maybe<&2, List<&2, U32>>, ShapeLaws.coordinate_result(output, Shape.unflatten_checked(output, U32.from_nat(position))),
    ShapeLaws.unflatten_spec(Shape.extents(output), ShapeLaws.fin_of(bound, U32.to_nat(U32.from_nat(position)))),
    Some{[U32.from_nat(Matrix.row_of(columns, position)), U32.from_nat(Matrix.column_of(columns, position))]},
    Boundary.unflatten_checked(output, U32.from_nat(position)),
    Equal.trans(Maybe<&2, List<&2, U32>>,
      ShapeLaws.unflatten_spec(Shape.extents(output), ShapeLaws.fin_of(bound, U32.to_nat(U32.from_nat(position)))),
      ShapeLaws.unflatten_spec(Shape.extents(output), ShapeLaws.fin_of(bound, position)),
      Some{[U32.from_nat(Matrix.row_of(columns, position)), U32.from_nat(Matrix.column_of(columns, position))]},
      Equal.cong(Nat, Maybe<&2, List<&2, U32>>, value => ShapeLaws.unflatten_spec(Shape.extents(output), ShapeLaws.fin_of(bound, value)),
        U32.to_nat(U32.from_nat(position)), position, word_exact(position, output, below)),
      Equal.trans(Maybe<&2, List<&2, U32>>,
        ShapeLaws.unflatten_spec(Shape.extents(output), ShapeLaws.fin_of(bound, position)),
        ShapeLaws.unflatten_spec(Shape.extents(output), Some{index}),
        Some{[U32.from_nat(Matrix.row_of(columns, position)), U32.from_nat(Matrix.column_of(columns, position))]},
        Equal.cong(Maybe<&2, Shape.Fin(bound)>, Maybe<&2, List<&2, U32>>, found => ShapeLaws.unflatten_spec(Shape.extents(output), found),
          ShapeLaws.fin_of(bound, position), Some{index}, Boundary.fin_of_from(bound, position, fits)),
        Equal.cong(List<&2, Nat>, Maybe<&2, List<&2, U32>>, values => Some{Shape.words(values)},
          Shape.indices(Shape.extents(output), Shape.unflatten(Shape.extents(output), index)),
          [Matrix.row_of(columns, position), Matrix.column_of(columns, position)],
          position_digits(output, rows, columns, extents, position, fits)))))

def position_fits(+output: Shape.Shape, +rows: Nat, +columns: Nat,
  +extents: {Shape.extents(output) == [rows, columns] : List<&2, Nat>}, +position: Nat,
  +below: {Nat.is_lt(position, Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}) ->
  {Shape.fit([rows, columns], [Matrix.row_of(columns, position), Matrix.column_of(columns, position)]) == True{} : Bool}:
  +bound = Shape.numel(Shape.extents(output))
  +fits = Equal.sym(Bool, Nat.is_lt(position, bound), True{}, below)
  +coordinate = Shape.unflatten(Shape.extents(output), Coordinates.fin_from(bound, position, fits))
  +digits : List<&2, Nat> = [Matrix.row_of(columns, position), Matrix.column_of(columns, position)]
  Equal.sym(Bool, True{}, Shape.fit([rows, columns], digits),
    Equal.trans(Bool, True{}, Shape.fit(Shape.extents(output), Shape.indices(Shape.extents(output), coordinate)), Shape.fit([rows, columns], digits),
      Coordinates.fit_indices(Shape.extents(output), coordinate),
      Equal.trans(Bool, Shape.fit(Shape.extents(output), Shape.indices(Shape.extents(output), coordinate)),
        Shape.fit(Shape.extents(output), digits), Shape.fit([rows, columns], digits),
        Equal.cong(List<&2, Nat>, Bool, values => Shape.fit(Shape.extents(output), values),
          Shape.indices(Shape.extents(output), coordinate), digits, position_digits(output, rows, columns, extents, position, fits)),
        Equal.cong(List<&2, Nat>, Bool, dims => Shape.fit(dims, digits), Shape.extents(output), [rows, columns], extents))))

def offset_pair(+a: Nat, +b: Nat, +first: Nat, +second: Nat) ->
  {Shape.offset([first, second], [a, b]) == Nat.add(Nat.mul(a, second), b) : Nat}:
  %Equal.sym(Nat, Nat.mul(second, 1n), second, Nats.mul_one_right(second)) :
    {Nat.add(Nat.mul(a, _), Nat.add(Nat.mul(b, 1n), 0n)) == Nat.add(Nat.mul(a, second), b) : Nat}
  %Equal.sym(Nat, Nat.mul(b, 1n), b, Nats.mul_one_right(b)) :
    {Nat.add(Nat.mul(a, second), Nat.add(_, 0n)) == Nat.add(Nat.mul(a, second), b) : Nat}
  %Equal.sym(Nat, Nat.add(b, 0n), b, Conversion.add_zero(b)) :
    {Nat.add(Nat.mul(a, second), _) == Nat.add(Nat.mul(a, second), b) : Nat}
  {==}

def flat_pair(+shape: Shape.Shape, +first: U32, +second: U32, +x: U32, +y: U32,
  +dimensions: {Shape.dimensions(shape) == [first, second] : List<&2, U32>},
  x_inside: {Nat.is_lt(U32.to_nat(x), U32.to_nat(first)) == True{} : Bool},
  y_inside: {Nat.is_lt(U32.to_nat(y), U32.to_nat(second)) == True{} : Bool}) ->
  {ShapeLaws.index_result(shape, Shape.flatten_checked(shape, [x, y])) ==
    Some{U32.from_nat(Nat.add(Nat.mul(U32.to_nat(x), U32.to_nat(second)), U32.to_nat(y)))} : Maybe<&2, U32>}:
  +values : List<&2, Nat> = [U32.to_nat(x), U32.to_nat(y)]
  +fits = Coordinates.pair_fits(shape, first, second, x, y, dimensions, x_inside, y_inside)
  +coordinate = Coordinates.coordinate_of(Shape.extents(shape), values, fits)
  +target : Maybe<&2, U32> = Some{U32.from_nat(Nat.add(Nat.mul(U32.to_nat(x), U32.to_nat(second)), U32.to_nat(y)))}
  Equal.trans(Maybe<&2, U32>, ShapeLaws.index_result(shape, Shape.flatten_checked(shape, [x, y])),
    ShapeLaws.flatten_spec(Shape.extents(shape), ShapeLaws.decode(Shape.extents(shape), values)), target,
    Boundary.flatten_checked(shape, [x, y]),
    Equal.trans(Maybe<&2, U32>, ShapeLaws.flatten_spec(Shape.extents(shape), ShapeLaws.decode(Shape.extents(shape), values)),
      Some{U32.from_nat(Shape.row_major(Shape.extents(shape), coordinate))}, target,
      Equal.cong(Maybe<&2, Shape.Coordinate(Shape.extents(shape))>, Maybe<&2, U32>, found => ShapeLaws.flatten_spec(Shape.extents(shape), found),
        ShapeLaws.decode(Shape.extents(shape), values), Some{coordinate}, Boundary.decode_fit(Shape.extents(shape), values, fits)),
      Equal.trans(Maybe<&2, U32>, Some{U32.from_nat(Shape.row_major(Shape.extents(shape), coordinate))},
        Some{U32.from_nat(Shape.offset(Shape.extents(shape), values))}, target,
        Equal.cong(Nat, Maybe<&2, U32>, offset => Some{U32.from_nat(offset)}, Shape.row_major(Shape.extents(shape), coordinate),
          Shape.offset(Shape.extents(shape), values), Coordinates.offset_of(Shape.extents(shape), values, fits)),
        Equal.trans(Maybe<&2, U32>, Some{U32.from_nat(Shape.offset(Shape.extents(shape), values))},
          Some{U32.from_nat(Shape.offset([U32.to_nat(first), U32.to_nat(second)], values))}, target,
          Equal.cong(List<&2, U32>, Maybe<&2, U32>, dims => Some{U32.from_nat(Shape.offset(Shape.naturals(dims), values))},
            Shape.dimensions(shape), [first, second], dimensions),
          Equal.cong(Nat, Maybe<&2, U32>, offset => Some{U32.from_nat(offset)},
            Shape.offset([U32.to_nat(first), U32.to_nat(second)], values),
            Nat.add(Nat.mul(U32.to_nat(x), U32.to_nat(second)), U32.to_nat(y)),
            offset_pair(U32.to_nat(x), U32.to_nat(y), U32.to_nat(first), U32.to_nat(second)))))))

def index_same(+a: Nat, +b: Nat, +x: Nat, +y: Nat, +stride: Nat, ax: {a == x : Nat}, by: {b == y : Nat}) ->
  {Some{U32.from_nat(Nat.add(Nat.mul(a, stride), b))} == Some{U32.from_nat(Nat.add(Nat.mul(x, stride), y))} : Maybe<&2, U32>}:
  %Equal.sym(Nat, a, x, ax) : {Some{U32.from_nat(Nat.add(Nat.mul(_, stride), b))} == Some{U32.from_nat(Nat.add(Nat.mul(x, stride), y))} : Maybe<&2, U32>}
  %Equal.sym(Nat, b, y, by) : {Some{U32.from_nat(Nat.add(Nat.mul(x, stride), _))} == Some{U32.from_nat(Nat.add(Nat.mul(x, stride), y))} : Maybe<&2, U32>}
  {==}


def lt_add(+cursor: Nat, +rest: Nat) -> {Nat.is_lt(cursor, Nat.add(cursor, 1n+rest)) == True{} : Bool}:
  match cursor:
    case 0n:
      {==}
    case 1n+p:
      lt_add(p, rest)

# A cursor with steps left lies below the end.
def cursor_below(+cursor: Nat, +rest: Nat, +end: Nat, steps: {Nat.add(cursor, 1n+rest) == end : Nat}) ->
  {Nat.is_lt(cursor, end) == True{} : Bool}:
  %steps : {Nat.is_lt(cursor, _) == True{} : Bool}
  lt_add(cursor, rest)

def steps_next(+cursor: Nat, +rest: Nat, +end: Nat, steps: {Nat.add(cursor, 1n+rest) == end : Nat}) ->
  {Nat.add(1n+cursor, rest) == end : Nat}:
  Equal.trans(Nat, 1n+Nat.add(cursor, rest), Nat.add(cursor, 1n+rest), end,
    Equal.sym(Nat, Nat.add(cursor, 1n+rest), 1n+Nat.add(cursor, rest), Arith.add_succ(cursor, rest)), steps)
