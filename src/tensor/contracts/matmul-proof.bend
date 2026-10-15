import Base
import ../../shape/core.bend as Shape
import ../../shape/LAWS.bend as ShapeLaws
import ../../shape/boundary_proofs.bend as Boundary
import ../../shape/coordinate_proofs.bend as Coordinates
import ../../shape/nat_proofs.bend as Nats
import ../../storage/main.bend as Storage
import ../../storage/LAWS.bend as StorageLaws
import ../../storage/arith.bend as Arith
import ../core.bend as Core
import ../lifecycle.bend as Lifecycle
import ../matmul.bend as Matrix
import ../dot-fold.bend as DotFold
import ../word-conversion.bend as Conversion
import ../permutation-mapping.bend as Mapping
import ./replayed.bend as Replayed
import ./pointwise.bend as Pointwise
import ./pointwise-proof.bend as PointwiseProof
import ./fold.bend as Fold
import ./fold-records.bend as Records
import ./dot.bend as Dot
import ./dot-proof.bend as DotProof
import ./LAWS.bend as Laws

type MatrixFacts<-left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape, -rows: U32, -inner: U32, -right_inner: U32, -columns: U32> is Data:
  MatrixFacts{left_dimensions: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
    right_dimensions: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
    output_dimensions: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
    compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat}}

def Facts(-left: Shape.Shape, -right: Shape.Shape, result: Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>) -> Data:
  match result:
    case Fail{error}:
      Unit
    case Done{Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible}}:
      MatrixFacts<left, right, output, rows, inner, right_inner, columns>

def facts_output(-left: Shape.Shape, -right: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
  re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  result: Shape.ConstructResult<[rows, columns]>) ->
  Facts(left, right, Matrix.plan_output(left, right, rows, inner, right_inner, columns, le, re, compatible, result)):
  match result:
    case Shape.SError{error}:
      Unit{}
    case Shape.SBuilt{shape, preserved}:
      MatrixFacts{le, re, preserved, compatible}

def facts_compatible(-left: Shape.Shape, -right: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
  re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))>) ->
  Facts(left, right, Matrix.plan_compatible(left, right, rows, inner, right_inner, columns, le, re, decision)):
  match decision:
    case Shape.BFalse{equal}:
      Unit{}
    case Shape.BTrue{equal}:
      facts_output(left, right, rows, inner, right_inner, columns, le, re,
        Lifecycle.natural_equal(U32.to_nat(inner), U32.to_nat(right_inner), equal), Shape.construct([rows, columns]))

def facts_right(-left: Shape.Shape, +right: Shape.Shape, +rows: U32, +inner: U32,
  le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, dimensions: List<&2, U32>,
  same: {dimensions == Shape.dimensions(right) : List<&2, U32>}) ->
  Facts(left, right, Matrix.plan_right(left, right, rows, inner, le, dimensions, same)):
  match dimensions:
    case +right_inner <> +columns <> Nil{}:
      facts_compatible(left, right, rows, inner, right_inner, columns, le,
        Equal.sym(List<&2, U32>, [right_inner, columns], Shape.dimensions(right), same),
        Shape.bool_case(Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))))
    case Nil{}:
      Unit{}
    case head <> Nil{}:
      Unit{}
    case first <> second <> third <> rest:
      Unit{}

def facts_left(+left: Shape.Shape, +right: Shape.Shape, dimensions: List<&2, U32>,
  same: {dimensions == Shape.dimensions(left) : List<&2, U32>}) ->
  Facts(left, right, Matrix.plan_left(left, right, dimensions, same)):
  match dimensions:
    case +rows <> +inner <> Nil{}:
      facts_right(left, right, rows, inner, Equal.sym(List<&2, U32>, [rows, inner], Shape.dimensions(left), same),
        Shape.dimensions(right), {==})
    case Nil{}:
      Unit{}
    case head <> Nil{}:
      Unit{}
    case first <> second <> third <> rest:
      Unit{}

def plan_facts(+left: Shape.Shape, +right: Shape.Shape, +plan: Matrix.Plan<left, right>,
  planned: {Matrix.plan(left, right) == Done{plan} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>}) ->
  Facts(left, right, Done{plan}):
  %planned : Facts(left, right, _)
  facts_left(left, right, Shape.dimensions(left), {==})

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

# Output cell `position` sits at row `position / columns` and column
# `position % columns`, as `Shape.digits` of a `[rows, columns]` shape spells
# them.
def row_of(+columns: Nat, +position: Nat) -> Nat:
  Nat.div(position, Nat.mul(columns, 1n))

def column_of(+columns: Nat, +position: Nat) -> Nat:
  Nat.div(Nat.mod(position, Nat.mul(columns, 1n)), 1n)

def div_one(+number: Nat) -> {Nat.div(number, 1n) == number : Nat}:
  %Equal.trans(Nat, Nat.add(Nat.mul(number, 1n), 0n), Nat.mul(number, 1n), number,
    Conversion.add_zero(Nat.mul(number, 1n)), Nats.mul_one_right(number)) : {Nat.div(_, 1n) == number : Nat}
  Equal.cong(Nat & Nat, Nat, pair => Pair.fst(Nat, Nat, pair), Nat.divmod(Nat.add(Nat.mul(number, 1n), 0n), 1n), (number, 0n),
    Equal.sym(Nat & Nat, (number, 0n), Nat.divmod(Nat.add(Nat.mul(number, 1n), 0n), 1n), Nats.divmod_exact(number, 1n, 0n, Unit{})))

def row_at(+row: Nat, +column: Nat, +columns: Nat, below: {Nat.is_lt(column, columns) == True{} : Bool}) ->
  {row_of(columns, Nat.add(Nat.mul(row, columns), column)) == row : Nat}:
  %Equal.sym(Nat, Nat.mul(columns, 1n), columns, Nats.mul_one_right(columns)) :
    {Nat.div(Nat.add(Nat.mul(row, columns), column), _) == row : Nat}
  Equal.cong(Nat & Nat, Nat, pair => Pair.fst(Nat, Nat, pair), Nat.divmod(Nat.add(Nat.mul(row, columns), column), columns), (row, column),
    Equal.sym(Nat & Nat, (row, column), Nat.divmod(Nat.add(Nat.mul(row, columns), column), columns),
      Nats.divmod_exact(row, columns, column, lt_from(column, columns, below))))

def column_at(+row: Nat, +column: Nat, +columns: Nat, below: {Nat.is_lt(column, columns) == True{} : Bool}) ->
  {column_of(columns, Nat.add(Nat.mul(row, columns), column)) == column : Nat}:
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
    [row_of(columns, position), column_of(columns, position)] : List<&2, Nat>}:
  +bound = Shape.numel(Shape.extents(output))
  +index = Coordinates.fin_from(bound, position, fits)
  Equal.trans(List<&2, Nat>, Shape.indices(Shape.extents(output), Shape.unflatten(Shape.extents(output), index)),
    Shape.digits(Shape.extents(output), Shape.Fin.value(bound, index)), [row_of(columns, position), column_of(columns, position)],
    Coordinates.digits_unflatten(Shape.extents(output), index),
    Equal.trans(List<&2, Nat>, Shape.digits(Shape.extents(output), Shape.Fin.value(bound, index)),
      Shape.digits(Shape.extents(output), position), [row_of(columns, position), column_of(columns, position)],
      Equal.cong(Nat, List<&2, Nat>, value => Shape.digits(Shape.extents(output), value), Shape.Fin.value(bound, index), position,
        Coordinates.fin_from_value(bound, position, fits)),
      Equal.cong(List<&2, Nat>, List<&2, Nat>, dims => Shape.digits(dims, position), Shape.extents(output), [rows, columns], extents)))

def unflatten_at(+output: Shape.Shape, +rows: Nat, +columns: Nat,
  extents: {Shape.extents(output) == [rows, columns] : List<&2, Nat>}, +position: Nat,
  +below: {Nat.is_lt(position, Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}) ->
  {ShapeLaws.coordinate_result(output, Shape.unflatten_checked(output, U32.from_nat(position))) ==
    Some{[U32.from_nat(row_of(columns, position)), U32.from_nat(column_of(columns, position))]} : Maybe<&2, List<&2, U32>>}:
  +bound = Shape.numel(Shape.extents(output))
  +fits = Equal.sym(Bool, Nat.is_lt(position, bound), True{}, below)
  +index = Coordinates.fin_from(bound, position, fits)
  Equal.trans(Maybe<&2, List<&2, U32>>, ShapeLaws.coordinate_result(output, Shape.unflatten_checked(output, U32.from_nat(position))),
    ShapeLaws.unflatten_spec(Shape.extents(output), ShapeLaws.fin_of(bound, U32.to_nat(U32.from_nat(position)))),
    Some{[U32.from_nat(row_of(columns, position)), U32.from_nat(column_of(columns, position))]},
    Boundary.unflatten_checked(output, U32.from_nat(position)),
    Equal.trans(Maybe<&2, List<&2, U32>>,
      ShapeLaws.unflatten_spec(Shape.extents(output), ShapeLaws.fin_of(bound, U32.to_nat(U32.from_nat(position)))),
      ShapeLaws.unflatten_spec(Shape.extents(output), ShapeLaws.fin_of(bound, position)),
      Some{[U32.from_nat(row_of(columns, position)), U32.from_nat(column_of(columns, position))]},
      Equal.cong(Nat, Maybe<&2, List<&2, U32>>, value => ShapeLaws.unflatten_spec(Shape.extents(output), ShapeLaws.fin_of(bound, value)),
        U32.to_nat(U32.from_nat(position)), position, PointwiseProof.word_exact(position, output, below)),
      Equal.trans(Maybe<&2, List<&2, U32>>,
        ShapeLaws.unflatten_spec(Shape.extents(output), ShapeLaws.fin_of(bound, position)),
        ShapeLaws.unflatten_spec(Shape.extents(output), Some{index}),
        Some{[U32.from_nat(row_of(columns, position)), U32.from_nat(column_of(columns, position))]},
        Equal.cong(Maybe<&2, Shape.Fin(bound)>, Maybe<&2, List<&2, U32>>, found => ShapeLaws.unflatten_spec(Shape.extents(output), found),
          ShapeLaws.fin_of(bound, position), Some{index}, Boundary.fin_of_from(bound, position, fits)),
        Equal.cong(List<&2, Nat>, Maybe<&2, List<&2, U32>>, values => Some{Shape.words(values)},
          Shape.indices(Shape.extents(output), Shape.unflatten(Shape.extents(output), index)),
          [row_of(columns, position), column_of(columns, position)],
          position_digits(output, rows, columns, extents, position, fits)))))

def position_fits(+output: Shape.Shape, +rows: Nat, +columns: Nat,
  +extents: {Shape.extents(output) == [rows, columns] : List<&2, Nat>}, +position: Nat,
  +below: {Nat.is_lt(position, Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}) ->
  {Shape.fit([rows, columns], [row_of(columns, position), column_of(columns, position)]) == True{} : Bool}:
  +bound = Shape.numel(Shape.extents(output))
  +fits = Equal.sym(Bool, Nat.is_lt(position, bound), True{}, below)
  +coordinate = Shape.unflatten(Shape.extents(output), Coordinates.fin_from(bound, position, fits))
  +digits : List<&2, Nat> = [row_of(columns, position), column_of(columns, position)]
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

# The dot loop: `Matrix.natural_dot` reads left[row, k] and right[k, column]
# through `Core.read_coordinate` and folds them like `Fold.pair_loop`.

type CellFacts<-left: Shape.Shape, -right: Shape.Shape, -rows: U32, -inner: U32, -right_inner: U32, -columns: U32,
  -row: Nat, -column: Nat> is Data:
  CellFacts{left_dimensions: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
    right_dimensions: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
    compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
    row_below: {Nat.is_lt(row, U32.to_nat(rows)) == True{} : Bool},
    column_below: {Nat.is_lt(column, U32.to_nat(columns)) == True{} : Bool}}

def matrix_cell(+row: Nat, +column: Nat, +inner: U32, +columns: U32) -> Dot.Cell:
  Dot.Cell{row, column, U32.to_nat(inner), U32.to_nat(columns)}

type DotStep<-left: Shape.Shape, -right: Shape.Shape, -cell: Dot.Cell, -row: U32, -column: U32, -k: Nat> is Type:
  DotStep{exact: {Nat.is_eq(U32.to_nat(U32.from_nat(k)), k) == True{} : Bool},
    left_flat: {ShapeLaws.index_result(left, Shape.flatten_checked(left, [row, U32.from_nat(k)])) ==
      Some{U32.from_nat(Dot.left_index(cell, k))} : Maybe<&2, U32>},
    right_flat: {ShapeLaws.index_result(right, Shape.flatten_checked(right, [U32.from_nat(k), column])) ==
      Some{U32.from_nat(Dot.right_index(cell, k))} : Maybe<&2, U32>},
    left_inside: {Nat.is_lt(U32.to_nat(U32.from_nat(Dot.left_index(cell, k))), Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool},
    right_inside: {Nat.is_lt(U32.to_nat(U32.from_nat(Dot.right_index(cell, k))), Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool}}

def dot_total(+cell: Dot.Cell, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat) -> F32:
  Fold.pair_total(~Unit, ~Dot.multiply_add, ~Dot.Cell, ~Dot.left_index, ~Dot.right_index, Unit{}, cell,
    left, left_fill, left_ops, right, right_fill, right_ops, count)

def left_storage(+cell: Dot.Cell, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +count: Nat) -> Storage.Storage:
  PointwiseProof.input_storage(~Dot.Cell, ~Dot.left_index, cell, left, left_fill, left_ops, count)

def right_storage(+cell: Dot.Cell, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat) -> Storage.Storage:
  PointwiseProof.input_storage(~Dot.Cell, ~Dot.right_index, cell, right, right_fill, right_ops, count)

def DotShaped(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +row: U32, +column: U32, +cursor: Nat,
  -left_stored: Storage.Storage, -right_stored: Storage.Storage, +accumulator: F32,
  -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32> -> Type) -> Type:
  @-left_fits: {Lifecycle.logical_length(left_stored) == Shape.logical_numel(Shape.dimensions(left)) : Nat} ->
  @-right_fits: {Lifecycle.logical_length(right_stored) == Shape.logical_numel(Shape.dimensions(right)) : Nat} ->
    Q(Matrix.natural_dot(fuel, left, right, Core.TTensor{left_stored, left_fits}, Core.TTensor{right_stored, right_fits},
      row, column, cursor, accumulator))

def DotStart(+fuel: Nat, +cell: Dot.Cell, +row: U32, +column: U32,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +cursor: Nat,
  -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32> -> Type) -> Type:
  DotShaped(fuel, left, right, row, column, cursor,
    left_storage(cell, left, left_fill, left_ops, cursor), right_storage(cell, right, right_fill, right_ops, cursor),
    dot_total(cell, left, left_fill, left_ops, right, right_fill, right_ops, cursor), Q)

def DotFinal(+cell: Dot.Cell, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat,
  -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32> -> Type) -> Type:
  @-left_fits: {Lifecycle.logical_length(left_storage(cell, left, left_fill, left_ops, count)) == Shape.logical_numel(Shape.dimensions(left)) : Nat} ->
  @-right_fits: {Lifecycle.logical_length(right_storage(cell, right, right_fill, right_ops, count)) == Shape.logical_numel(Shape.dimensions(right)) : Nat} ->
    Q(Done{(Core.TTensor{left_storage(cell, left, left_fill, left_ops, count), left_fits},
      Core.TTensor{right_storage(cell, right, right_fill, right_ops, count), right_fits},
      dot_total(cell, left, left_fill, left_ops, right, right_fill, right_ops, count))})

def dot_final_at(+cell: Dot.Cell, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, -end: Nat, -count: Nat,
  same: {end == count : Nat}, -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32> -> Type,
  h: DotFinal(cell, left, left_fill, left_ops, right, right_fill, right_ops, end, Q)) ->
  DotFinal(cell, left, left_fill, left_ops, right, right_fill, right_ops, count, Q):
  %same : DotFinal(cell, left, left_fill, left_ops, right, right_fill, right_ops, _, Q)
  h

def dot_shift(+rest: Nat, +cell: Dot.Cell, +row: U32, +column: U32,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +cursor: Nat,
  -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32> -> Type,
  next: DotStart(rest, cell, row, column, left, left_fill, left_ops, right, right_fill, right_ops, 1n+cursor, Q)) ->
  DotShaped(rest, left, right, row, column, 1n+cursor,
    Replayed.storage(left, left_fill, Replayed.snoc(List.append(&2, StorageLaws.Op, left_ops, Pointwise.reads(~Dot.Cell, ~Dot.left_index, cell, cursor)),
      StorageLaws.Read{U32.from_nat(Dot.left_index(cell, cursor))})),
    Replayed.storage(right, right_fill, Replayed.snoc(List.append(&2, StorageLaws.Op, right_ops, Pointwise.reads(~Dot.Cell, ~Dot.right_index, cell, cursor)),
      StorageLaws.Read{U32.from_nat(Dot.right_index(cell, cursor))})),
    F32.add(dot_total(cell, left, left_fill, left_ops, right, right_fill, right_ops, cursor),
      F32.mul(Replayed.value(left, left_fill, List.append(&2, StorageLaws.Op, left_ops, Pointwise.reads(~Dot.Cell, ~Dot.left_index, cell, cursor)),
        U32.from_nat(Dot.left_index(cell, cursor))),
        Replayed.value(right, right_fill, List.append(&2, StorageLaws.Op, right_ops, Pointwise.reads(~Dot.Cell, ~Dot.right_index, cell, cursor)),
          U32.from_nat(Dot.right_index(cell, cursor))))),
    Q):
  +left_read = {StorageLaws.Read{U32.from_nat(Dot.left_index(cell, cursor))} : StorageLaws.Op}
  +right_read = {StorageLaws.Read{U32.from_nat(Dot.right_index(cell, cursor))} : StorageLaws.Op}
  +left_seen = List.append(&2, StorageLaws.Op, left_ops, Pointwise.reads(~Dot.Cell, ~Dot.left_index, cell, cursor))
  +right_seen = List.append(&2, StorageLaws.Op, right_ops, Pointwise.reads(~Dot.Cell, ~Dot.right_index, cell, cursor))
  +before = dot_total(cell, left, left_fill, left_ops, right, right_fill, right_ops, cursor)
  +x = Replayed.value(left, left_fill, left_seen, U32.from_nat(Dot.left_index(cell, cursor)))
  +y = Replayed.value(right, right_fill, right_seen, U32.from_nat(Dot.right_index(cell, cursor)))
  +left_word = Replayed.value(left, left_fill, left_ops, U32.from_nat(Dot.left_index(cell, cursor)))
  +right_word = Replayed.value(right, right_fill, right_ops, U32.from_nat(Dot.right_index(cell, cursor)))
  -right_after = Replayed.storage(right, right_fill, Replayed.snoc(right_seen, right_read))
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, history => Replayed.storage(left, left_fill, history),
    List.append(&2, StorageLaws.Op, left_ops, Replayed.snoc(Pointwise.reads(~Dot.Cell, ~Dot.left_index, cell, cursor), left_read)),
    Replayed.snoc(left_seen, left_read),
    PointwiseProof.append_snoc(left_ops, Pointwise.reads(~Dot.Cell, ~Dot.left_index, cell, cursor), left_read)) :
    DotShaped(rest, left, right, row, column, 1n+cursor, _, right_after, F32.add(before, F32.mul(x, y)), Q)
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, history => Replayed.storage(right, right_fill, history),
    List.append(&2, StorageLaws.Op, right_ops, Replayed.snoc(Pointwise.reads(~Dot.Cell, ~Dot.right_index, cell, cursor), right_read)),
    Replayed.snoc(right_seen, right_read),
    PointwiseProof.append_snoc(right_ops, Pointwise.reads(~Dot.Cell, ~Dot.right_index, cell, cursor), right_read)) :
    DotShaped(rest, left, right, row, column, 1n+cursor, left_storage(cell, left, left_fill, left_ops, 1n+cursor), _,
      F32.add(before, F32.mul(x, y)), Q)
  %Equal.trans(F32, F32.add(before, F32.mul(left_word, right_word)), F32.add(before, F32.mul(x, right_word)), F32.add(before, F32.mul(x, y)),
    Equal.cong(F32, F32, w => F32.add(before, F32.mul(w, right_word)), left_word, x,
      Equal.sym(F32, x, left_word,
        PointwiseProof.value_reads(~Dot.Cell, ~Dot.left_index, cell, cursor, left, left_fill, left_ops, U32.from_nat(Dot.left_index(cell, cursor))))),
    Equal.cong(F32, F32, w => F32.add(before, F32.mul(x, w)), right_word, y,
      Equal.sym(F32, y, right_word,
        PointwiseProof.value_reads(~Dot.Cell, ~Dot.right_index, cell, cursor, right, right_fill, right_ops, U32.from_nat(Dot.right_index(cell, cursor)))))) :
    DotShaped(rest, left, right, row, column, 1n+cursor, left_storage(cell, left, left_fill, left_ops, 1n+cursor),
      right_storage(cell, right, right_fill, right_ops, 1n+cursor), _, Q)
  next

def dot_decided_lift(+left: Shape.Shape, +right: Shape.Shape,
  -rec: Core.Tensor<left> -> Core.Tensor<right> -> F32 -> Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32>,
  -a: Core.Tensor<left>, -b: Core.Tensor<right>, +row: U32, +column: U32, +cursor: Nat, -accumulator: F32,
  -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32> -> Type,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)>,
  same: {Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor) == True{} : Bool},
  h: Q(Matrix.left_read(left, right, rec, b, [U32.from_nat(cursor), column], accumulator,
    Core.read_coordinate(left, a, [row, U32.from_nat(cursor)])))) ->
  Q(Matrix.dot_decided(left, right, rec, a, b, row, column, cursor, U32.from_nat(cursor), accumulator, decision)):
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd(Q(Matrix.dot_decided(left, right, rec, a, b, row, column, cursor, U32.from_nat(cursor), accumulator, Shape.BFalse{equal})),
        Arith.clash(Equal.trans(Bool, False{}, Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor), True{}, equal, same)))
    case Shape.BTrue{equal}:
      h

def coordinate_lift(+shape: Shape.Shape, -tensor: Core.Tensor<shape>, -index: U32,
  result: Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>,
  found: {ShapeLaws.index_result(shape, result) == Some{index} : Maybe<&2, U32>},
  -P: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32> -> Type,
  h: P(Core.read_flat(shape, tensor, index))) -> P(Core.read_coordinate_result(shape, tensor, result)):
  match result:
    case Fail{error}:
      Empty.absurd(P(Core.read_coordinate_result(shape, tensor, Fail{error})),
        Arith.clash(Equal.cong(Maybe<&2, U32>, Bool, value => Maybe.is_some(&2, U32, value), None{}, Some{index}, found)))
    case Done{flat}:
      %Equal.sym(U32, Shape.index_value(shape, flat), index, some_equal(U32, Shape.index_value(shape, flat), index, found)) :
        P(Core.read_flat(shape, tensor, _))
      h

def dot_step_lift(+rest: Nat, +cell: Dot.Cell, +row: U32, +column: U32,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +cursor: Nat,
  step: DotStep<left, right, cell, row, column, cursor>,
  -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32> -> Type,
  next: DotStart(rest, cell, row, column, left, left_fill, left_ops, right, right_fill, right_ops, 1n+cursor, Q)) ->
  DotStart(1n+rest, cell, row, column, left, left_fill, left_ops, right, right_fill, right_ops, cursor, Q):
  DotStep{exact, left_flat, right_flat, left_inside, right_inside} = step
  +left_seen = List.append(&2, StorageLaws.Op, left_ops, Pointwise.reads(~Dot.Cell, ~Dot.left_index, cell, cursor))
  +right_seen = List.append(&2, StorageLaws.Op, right_ops, Pointwise.reads(~Dot.Cell, ~Dot.right_index, cell, cursor))
  +left_index = U32.from_nat(Dot.left_index(cell, cursor))
  +right_index = U32.from_nat(Dot.right_index(cell, cursor))
  +index = U32.from_nat(cursor)
  +before = dot_total(cell, left, left_fill, left_ops, right, right_fill, right_ops, cursor)
  +x = Replayed.value(left, left_fill, left_seen, left_index)
  left_fits => right_fits =>
    dot_decided_lift(left, right,
      a => b => value => Matrix.natural_dot(rest, left, right, a, b, row, column, 1n+cursor, value),
      Core.TTensor{Replayed.storage(left, left_fill, left_seen), left_fits}, Core.TTensor{Replayed.storage(right, right_fill, right_seen), right_fits},
      row, column, cursor, before, Q, Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)), exact,
      coordinate_lift(left, Core.TTensor{Replayed.storage(left, left_fill, left_seen), left_fits}, left_index,
        Shape.flatten_checked(left, [row, index]), left_flat,
        pair => Q(Matrix.left_read(left, right, a => b => value => Matrix.natural_dot(rest, left, right, a, b, row, column, 1n+cursor, value),
          Core.TTensor{Replayed.storage(right, right_fill, right_seen), right_fits}, [index, column], before, pair)),
        Replayed.read_lift(left, left_fill, left_seen, left_fits, left_index, left_inside,
          pair => Q(Matrix.left_read(left, right, a => b => value => Matrix.natural_dot(rest, left, right, a, b, row, column, 1n+cursor, value),
            Core.TTensor{Replayed.storage(right, right_fill, right_seen), right_fits}, [index, column], before, pair)),
          left_after => coordinate_lift(right, Core.TTensor{Replayed.storage(right, right_fill, right_seen), right_fits}, right_index,
            Shape.flatten_checked(right, [index, column]), right_flat,
            pair => Q(Matrix.right_read(left, right, a => b => value => Matrix.natural_dot(rest, left, right, a, b, row, column, 1n+cursor, value),
              Core.TTensor{Replayed.storage(left, left_fill, Replayed.snoc(left_seen, StorageLaws.Read{left_index})), left_after}, x, before, pair)),
            Replayed.read_lift(right, right_fill, right_seen, right_fits, right_index, right_inside,
              pair => Q(Matrix.right_read(left, right, a => b => value => Matrix.natural_dot(rest, left, right, a, b, row, column, 1n+cursor, value),
                Core.TTensor{Replayed.storage(left, left_fill, Replayed.snoc(left_seen, StorageLaws.Read{left_index})), left_after}, x, before, pair)),
              right_after => dot_shift(rest, cell, row, column, left, left_fill, left_ops, right, right_fill, right_ops, cursor, Q, next)(
                left_after, right_after))))))

def cell_step(+left: Shape.Shape, +right: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  +row: Nat, +column: Nat, +facts: CellFacts<left, right, rows, inner, right_inner, columns, row, column>,
  +k: Nat, +below: {Nat.is_lt(k, U32.to_nat(inner)) == True{} : Bool}) ->
  DotStep<left, right, matrix_cell(row, column, inner, columns), U32.from_nat(row), U32.from_nat(column), k>:
  CellFacts{+le, +re, +compatible, +row_below, +column_below} = facts
  +inner_count = U32.to_nat(inner)
  +column_count = U32.to_nat(columns)
  +k_exact = word_below(k, inner, below)
  +row_exact = word_below(row, rows, row_below)
  +column_exact = word_below(column, columns, column_below)
  +left_extents = Equal.cong(List<&2, U32>, List<&2, Nat>, dims => Shape.naturals(dims), Shape.dimensions(left), [rows, inner], le)
  +right_extents = Equal.trans(List<&2, Nat>, Shape.extents(right), [U32.to_nat(right_inner), column_count], [inner_count, column_count],
    Equal.cong(List<&2, U32>, List<&2, Nat>, dims => Shape.naturals(dims), Shape.dimensions(right), [right_inner, columns], re),
    Equal.cong(Nat, List<&2, Nat>, n => [n, column_count], U32.to_nat(right_inner), inner_count,
      Equal.sym(Nat, inner_count, U32.to_nat(right_inner), compatible)))
  DotStep{
    Mapping.natural_equal_true(U32.to_nat(U32.from_nat(k)), k, k_exact),
    Equal.trans(Maybe<&2, U32>, ShapeLaws.index_result(left, Shape.flatten_checked(left, [U32.from_nat(row), U32.from_nat(k)])),
      Some{U32.from_nat(Nat.add(Nat.mul(U32.to_nat(U32.from_nat(row)), inner_count), U32.to_nat(U32.from_nat(k))))},
      Some{U32.from_nat(Nat.add(Nat.mul(row, inner_count), k))},
      flat_pair(left, rows, inner, U32.from_nat(row), U32.from_nat(k), le,
        lt_same(U32.to_nat(U32.from_nat(row)), row, U32.to_nat(rows), row_exact, row_below),
        lt_same(U32.to_nat(U32.from_nat(k)), k, inner_count, k_exact, below)),
      index_same(U32.to_nat(U32.from_nat(row)), U32.to_nat(U32.from_nat(k)), row, k, inner_count, row_exact, k_exact)),
    Equal.trans(Maybe<&2, U32>, ShapeLaws.index_result(right, Shape.flatten_checked(right, [U32.from_nat(k), U32.from_nat(column)])),
      Some{U32.from_nat(Nat.add(Nat.mul(U32.to_nat(U32.from_nat(k)), column_count), U32.to_nat(U32.from_nat(column))))},
      Some{U32.from_nat(Nat.add(Nat.mul(k, column_count), column))},
      flat_pair(right, right_inner, columns, U32.from_nat(k), U32.from_nat(column), re,
        lt_same(U32.to_nat(U32.from_nat(k)), k, U32.to_nat(right_inner), k_exact,
          lt_bound(k, inner_count, U32.to_nat(right_inner), compatible, below)),
        lt_same(U32.to_nat(U32.from_nat(column)), column, column_count, column_exact, column_below)),
      index_same(U32.to_nat(U32.from_nat(k)), U32.to_nat(U32.from_nat(column)), k, column, column_count, k_exact, column_exact)),
    PointwiseProof.word_inside(Nat.add(Nat.mul(row, inner_count), k), left,
      DotProof.left_below(U32.to_nat(rows), inner_count, column_count, row, column, row_below, left, left_extents, k, below)),
    PointwiseProof.word_inside(Nat.add(Nat.mul(k, column_count), column), right,
      DotProof.right_below(inner_count, column_count, row, column, column_below, right, right_extents, k, below))}

def dot_loop_lift(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  +row: Nat, +column: Nat, +facts: CellFacts<left, right, rows, inner, right_inner, columns, row, column>,
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +cursor: Nat,
  +steps: {Nat.add(cursor, fuel) == U32.to_nat(inner) : Nat},
  -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32> -> Type,
  h: DotFinal(matrix_cell(row, column, inner, columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner), Q)) ->
  DotStart(fuel, matrix_cell(row, column, inner, columns), U32.from_nat(row), U32.from_nat(column),
    left, left_fill, left_ops, right, right_fill, right_ops, cursor, Q):
  match fuel:
    case 0n:
      dot_final_at(matrix_cell(row, column, inner, columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner), cursor,
        Equal.sym(Nat, cursor, U32.to_nat(inner), PointwiseProof.steps_done(cursor, U32.to_nat(inner), steps)), Q, h)
    case 1n+rest:
      dot_step_lift(rest, matrix_cell(row, column, inner, columns), U32.from_nat(row), U32.from_nat(column),
        left, left_fill, left_ops, right, right_fill, right_ops, cursor,
        cell_step(left, right, rows, inner, right_inner, columns, row, column, facts, cursor,
          PointwiseProof.cursor_below(cursor, rest, U32.to_nat(inner), steps)), Q,
        dot_loop_lift(rest, left, right, rows, inner, right_inner, columns, row, column, facts, left_fill, left_ops, right_fill, right_ops,
          1n+cursor, PointwiseProof.steps_next(cursor, rest, U32.to_nat(inner), steps), Q, h))

def dot_start_cast(+fuel: Nat, +cell: Dot.Cell, +row: U32, +column: U32,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32> -> Type,
  start: DotStart(fuel, cell, row, column, left, left_fill, left_ops, right, right_fill, right_ops, 0n, Q)) ->
  DotShaped(fuel, left, right, row, column, 0n, Replayed.storage(left, left_fill, left_ops), Replayed.storage(right, right_fill, right_ops), 0.0, Q):
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, history => Replayed.storage(left, left_fill, history),
    List.append(&2, StorageLaws.Op, left_ops, Nil{}), left_ops, PointwiseProof.append_nil(left_ops)) :
    DotShaped(fuel, left, right, row, column, 0n, _, Replayed.storage(right, right_fill, right_ops), 0.0, Q)
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, history => Replayed.storage(right, right_fill, history),
    List.append(&2, StorageLaws.Op, right_ops, Nil{}), right_ops, PointwiseProof.append_nil(right_ops)) :
    DotShaped(fuel, left, right, row, column, 0n, left_storage(cell, left, left_fill, left_ops, 0n), _, 0.0, Q)
  start

def dot_run_lift(+left: Shape.Shape, +right: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  +row: Nat, +column: Nat, +facts: CellFacts<left, right, rows, inner, right_inner, columns, row, column>,
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32> -> Type,
  h: DotFinal(matrix_cell(row, column, inner, columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner), Q)) ->
  DotShaped(U32.to_nat(inner), left, right, U32.from_nat(row), U32.from_nat(column), 0n,
    Replayed.storage(left, left_fill, left_ops), Replayed.storage(right, right_fill, right_ops), 0.0, Q):
  dot_start_cast(U32.to_nat(inner), matrix_cell(row, column, inner, columns), U32.from_nat(row), U32.from_nat(column),
    left, left_fill, left_ops, right, right_fill, right_ops, Q,
    dot_loop_lift(U32.to_nat(inner), left, right, rows, inner, right_inner, columns, row, column, facts,
      left_fill, left_ops, right_fill, right_ops, 0n, {==}, Q, h))

# The product loop: cell `position` folds row `row_of(position)` of the left
# matrix with column `column_of(position)` of the right one and writes the
# total at `position`.

def cell_at(+inner: U32, +columns: U32, +position: Nat) -> Dot.Cell:
  matrix_cell(row_of(U32.to_nat(columns), position), column_of(U32.to_nat(columns), position), inner, columns)

def left_history(+inner: U32, +columns: U32, +left_ops: List<&2, StorageLaws.Op>, +count: Nat) -> List<&2, StorageLaws.Op>:
  match count:
    case 0n:
      left_ops
    case 1n+previous:
      List.append(&2, StorageLaws.Op, left_history(inner, columns, left_ops, previous),
        Pointwise.reads(~Dot.Cell, ~Dot.left_index, cell_at(inner, columns, previous), U32.to_nat(inner)))

def right_history(+inner: U32, +columns: U32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat) -> List<&2, StorageLaws.Op>:
  match count:
    case 0n:
      right_ops
    case 1n+previous:
      List.append(&2, StorageLaws.Op, right_history(inner, columns, right_ops, previous),
        Pointwise.reads(~Dot.Cell, ~Dot.right_index, cell_at(inner, columns, previous), U32.to_nat(inner)))

def product_total(+inner: U32, +columns: U32, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +position: Nat) -> F32:
  dot_total(cell_at(inner, columns, position), left, left_fill, left_history(inner, columns, left_ops, position),
    right, right_fill, right_history(inner, columns, right_ops, position), U32.to_nat(inner))

def products(+inner: U32, +columns: U32, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat) -> List<&2, StorageLaws.Op>:
  match count:
    case 0n:
      Nil{}
    case 1n+previous:
      Replayed.snoc(products(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, previous),
        StorageLaws.Write{U32.from_nat(previous), product_total(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, previous)})

def ProductShaped(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +inner: U32, +columns: U32, +cursor: Nat,
  -left_stored: Storage.Storage, -right_stored: Storage.Storage, -output_stored: Storage.Storage,
  -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>> -> Type) -> Type:
  @-left_fits: {Lifecycle.logical_length(left_stored) == Shape.logical_numel(Shape.dimensions(left)) : Nat} ->
  @-right_fits: {Lifecycle.logical_length(right_stored) == Shape.logical_numel(Shape.dimensions(right)) : Nat} ->
  @-output_fits: {Lifecycle.logical_length(output_stored) == Shape.logical_numel(Shape.dimensions(output)) : Nat} ->
    Q(Matrix.natural_product_loop(fuel, left, right, output, inner, columns, Core.TTensor{left_stored, left_fits},
      Core.TTensor{right_stored, right_fits}, Core.TTensor{output_stored, output_fits}, cursor))

def ProductStart(+fuel: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +output: Shape.Shape, +inner: U32, +columns: U32,
  +cursor: Nat, -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>> -> Type) -> Type:
  ProductShaped(fuel, left, right, output, inner, columns, cursor,
    Replayed.storage(left, left_fill, left_history(inner, columns, left_ops, cursor)),
    Replayed.storage(right, right_fill, right_history(inner, columns, right_ops, cursor)),
    Replayed.storage(output, 0.0, products(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, cursor)), Q)

def ProductFinal(+left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +output: Shape.Shape, +inner: U32, +columns: U32,
  +count: Nat, -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>> -> Type) -> Type:
  @-left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_history(inner, columns, left_ops, count))) ==
    Shape.logical_numel(Shape.dimensions(left)) : Nat} ->
  @-right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_history(inner, columns, right_ops, count))) ==
    Shape.logical_numel(Shape.dimensions(right)) : Nat} ->
  @-output_fits: {Lifecycle.logical_length(Replayed.storage(output, 0.0,
    products(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, count))) ==
    Shape.logical_numel(Shape.dimensions(output)) : Nat} ->
    Q(Done{(Core.TTensor{Replayed.storage(left, left_fill, left_history(inner, columns, left_ops, count)), left_fits},
      Core.TTensor{Replayed.storage(right, right_fill, right_history(inner, columns, right_ops, count)), right_fits},
      Core.TTensor{Replayed.storage(output, 0.0, products(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, count)),
        output_fits})})

def product_final_at(+left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +output: Shape.Shape, +inner: U32, +columns: U32,
  -end: Nat, -count: Nat, same: {end == count : Nat},
  -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>> -> Type,
  h: ProductFinal(left, left_fill, left_ops, right, right_fill, right_ops, output, inner, columns, end, Q)) ->
  ProductFinal(left, left_fill, left_ops, right, right_fill, right_ops, output, inner, columns, count, Q):
  %same : ProductFinal(left, left_fill, left_ops, right, right_fill, right_ops, output, inner, columns, _, Q)
  h

def product_decided_lift(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  -rec: Core.Tensor<left> -> Core.Tensor<right> -> Core.Tensor<output> ->
    Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>,
  +inner: U32, +columns: U32, -a: Core.Tensor<left>, -b: Core.Tensor<right>, -out: Core.Tensor<output>, +cursor: Nat,
  -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>> -> Type,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)>,
  same: {Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor) == True{} : Bool},
  h: Q(Matrix.product_unflattened(left, right, output, rec, inner, a, b, out, U32.from_nat(cursor),
    Shape.unflatten_checked(output, U32.from_nat(cursor))))) ->
  Q(Matrix.product_decided(left, right, output, rec, inner, columns, a, b, out, cursor, U32.from_nat(cursor), decision)):
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd(Q(Matrix.product_decided(left, right, output, rec, inner, columns, a, b, out, cursor, U32.from_nat(cursor), Shape.BFalse{equal})),
        Arith.clash(Equal.trans(Bool, False{}, Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor), True{}, equal, same)))
    case Shape.BTrue{equal}:
      h

def coordinates_lift(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  -rec: Core.Tensor<left> -> Core.Tensor<right> -> Core.Tensor<output> ->
    Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>,
  +inner: U32, -a: Core.Tensor<left>, -b: Core.Tensor<right>, -out: Core.Tensor<output>, +index: U32, -row: U32, -column: U32,
  result: Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<output>>,
  found: {ShapeLaws.coordinate_result(output, result) == Some{[row, column]} : Maybe<&2, List<&2, U32>>},
  -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>> -> Type,
  h: Q(Matrix.product_coordinates(left, right, output, rec, inner, a, b, out, index, [row, column]))) ->
  Q(Matrix.product_unflattened(left, right, output, rec, inner, a, b, out, index, result)):
  match result:
    case Fail{error}:
      Empty.absurd(Q(Matrix.product_unflattened(left, right, output, rec, inner, a, b, out, index, Fail{error})),
        Arith.clash(Equal.cong(Maybe<&2, List<&2, U32>>, Bool, value => Maybe.is_some(&2, List<&2, U32>, value), None{}, Some{[row, column]}, found)))
    case Done{coordinate}:
      %Equal.sym(List<&2, U32>, Shape.coordinate_values(output, coordinate), [row, column],
        some_equal(List<&2, U32>, Shape.coordinate_values(output, coordinate), [row, column], found)) :
        Q(Matrix.product_coordinates(left, right, output, rec, inner, a, b, out, index, _))
      h

def product_step_lift(+rest: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +output: Shape.Shape,
  +rows: U32, +inner: U32, +right_inner: U32, +columns: U32, +facts: MatrixFacts<left, right, output, rows, inner, right_inner, columns>,
  +cursor: Nat, +below: {Nat.is_lt(cursor, Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool},
  -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>> -> Type,
  next: ProductStart(rest, left, left_fill, left_ops, right, right_fill, right_ops, output, inner, columns, 1n+cursor, Q)) ->
  ProductStart(1n+rest, left, left_fill, left_ops, right, right_fill, right_ops, output, inner, columns, cursor, Q):
  MatrixFacts{+le, +re, +oe, +compatible} = facts
  +column_count = U32.to_nat(columns)
  +row = row_of(column_count, cursor)
  +column = column_of(column_count, cursor)
  +extents = Equal.cong(List<&2, U32>, List<&2, Nat>, dims => Shape.naturals(dims), Shape.dimensions(output), [rows, columns], oe)
  +fit = position_fits(output, U32.to_nat(rows), column_count, extents, cursor, below)
  +row_below = Arith.and_left(Nat.is_lt(row, U32.to_nat(rows)), Bool.and(Nat.is_lt(column, column_count), True{}), fit)
  +column_below = Arith.and_left(Nat.is_lt(column, column_count), True{},
    Arith.and_right(Nat.is_lt(row, U32.to_nat(rows)), Bool.and(Nat.is_lt(column, column_count), True{}), fit))
  +cell = cell_at(inner, columns, cursor)
  +left_seen = left_history(inner, columns, left_ops, cursor)
  +right_seen = right_history(inner, columns, right_ops, cursor)
  +written = products(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, cursor)
  +total = product_total(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, cursor)
  +index = U32.from_nat(cursor)
  left_fits => right_fits => output_fits =>
    product_decided_lift(left, right, output,
      a => b => out => Matrix.natural_product_loop(rest, left, right, output, inner, columns, a, b, out, 1n+cursor),
      inner, columns, Core.TTensor{Replayed.storage(left, left_fill, left_seen), left_fits},
      Core.TTensor{Replayed.storage(right, right_fill, right_seen), right_fits}, Core.TTensor{Replayed.storage(output, 0.0, written), output_fits},
      cursor, Q, Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)),
      Mapping.natural_equal_true(U32.to_nat(U32.from_nat(cursor)), cursor, PointwiseProof.word_exact(cursor, output, below)),
      coordinates_lift(left, right, output,
        a => b => out => Matrix.natural_product_loop(rest, left, right, output, inner, columns, a, b, out, 1n+cursor),
        inner, Core.TTensor{Replayed.storage(left, left_fill, left_seen), left_fits},
        Core.TTensor{Replayed.storage(right, right_fill, right_seen), right_fits}, Core.TTensor{Replayed.storage(output, 0.0, written), output_fits},
        index, U32.from_nat(row), U32.from_nat(column), Shape.unflatten_checked(output, index),
        unflatten_at(output, U32.to_nat(rows), column_count, extents, cursor, below), Q,
        dot_run_lift(left, right, rows, inner, right_inner, columns, row, column, CellFacts{le, re, compatible, row_below, column_below},
          left_fill, left_seen, right_fill, right_seen,
          result => Q(Matrix.dotted(left, right, output,
            a => b => out => Matrix.natural_product_loop(rest, left, right, output, inner, columns, a, b, out, 1n+cursor),
            Core.TTensor{Replayed.storage(output, 0.0, written), output_fits}, index, result)),
          left_after => right_after =>
            Replayed.write_lift(output, 0.0, written, output_fits, index, total, PointwiseProof.word_inside(cursor, output, below),
              pair => Q(Matrix.wrote(left, right, output,
                a => b => out => Matrix.natural_product_loop(rest, left, right, output, inner, columns, a, b, out, 1n+cursor),
                Core.TTensor{left_storage(cell, left, left_fill, left_seen, U32.to_nat(inner)), left_after},
                Core.TTensor{right_storage(cell, right, right_fill, right_seen, U32.to_nat(inner)), right_after}, pair)),
              output_after => next(left_after, right_after, output_after)))(left_fits, right_fits)))

def product_loop_lift(+fuel: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +output: Shape.Shape,
  +rows: U32, +inner: U32, +right_inner: U32, +columns: U32, +facts: MatrixFacts<left, right, output, rows, inner, right_inner, columns>,
  +cursor: Nat, +count: Nat, +steps: {Nat.add(cursor, fuel) == count : Nat},
  +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool},
  -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>> -> Type,
  h: ProductFinal(left, left_fill, left_ops, right, right_fill, right_ops, output, inner, columns, count, Q)) ->
  ProductStart(fuel, left, left_fill, left_ops, right, right_fill, right_ops, output, inner, columns, cursor, Q):
  match fuel:
    case 0n:
      product_final_at(left, left_fill, left_ops, right, right_fill, right_ops, output, inner, columns, count, cursor,
        Equal.sym(Nat, cursor, count, PointwiseProof.steps_done(cursor, count, steps)), Q, h)
    case 1n+rest:
      product_step_lift(rest, left, left_fill, left_ops, right, right_fill, right_ops, output, rows, inner, right_inner, columns, facts, cursor,
        Arith.lt_le_trans(cursor, count, Shape.logical_numel(Shape.dimensions(output)), PointwiseProof.cursor_below(cursor, rest, count, steps), room),
        Q, product_loop_lift(rest, left, left_fill, left_ops, right, right_fill, right_ops, output, rows, inner, right_inner, columns, facts,
          1n+cursor, count, PointwiseProof.steps_next(cursor, rest, count, steps), room, Q, h))

def elements_room(+output: Shape.Shape) -> {Nat.is_le(Shape.elements(output), Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}:
  %Equal.sym(Nat, Shape.elements(output), Shape.numel(Shape.extents(output)), Shape.elements_exact(output)) :
    {Nat.is_le(_, Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}
  Conversion.le_reflexive(Shape.logical_numel(Shape.dimensions(output)))

def multiply_lift(+left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat},
  +output: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  +facts: MatrixFacts<left, right, output, rows, inner, right_inner, columns>,
  -Q: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>> -> Type,
  h: ProductFinal(left, left_fill, left_ops, right, right_fill, right_ops, output, inner, columns, Shape.elements(output), Q)) ->
  Q(Matrix.created(left, right, output, inner, columns, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits},
    Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}, Core.new(output, 0.0))):
  Replayed.new_lift(output, 0.0,
    result => Q(Matrix.created(left, right, output, inner, columns, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits},
      Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}, result)),
    output_fits => product_loop_lift(Shape.elements(output), left, left_fill, left_ops, right, right_fill, right_ops, output,
      rows, inner, right_inner, columns, facts, 0n, Shape.elements(output), {==}, elements_room(output), Q, h)(left_fits, right_fits, output_fits))

def matmul_lift(+left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat},
  +plan: Matrix.Plan<left, right>,
  planned: {Matrix.plan(left, right) == Done{plan} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>},
  -Q: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic> -> Type,
  h: Q(Matrix.packed(left, right, Matrix.output_shape(left, right, plan), Matrix.multiply(left, right, plan,
    Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits})))) ->
  Q(Matrix.matmul(Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
    Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})):
  %Equal.sym(Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>, Matrix.plan(left, right), Done{plan}, planned) :
    Q(Matrix.planned(left, right, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits},
      Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}, _))
  h

def left_history_value(+inner: U32, +columns: U32, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +count: Nat, +index: U32) ->
  {Replayed.value(left, left_fill, left_history(inner, columns, left_ops, count), index) == Replayed.value(left, left_fill, left_ops, index) : F32}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      Equal.trans(F32, Replayed.value(left, left_fill, left_history(inner, columns, left_ops, 1n+previous), index),
        Replayed.value(left, left_fill, left_history(inner, columns, left_ops, previous), index),
        Replayed.value(left, left_fill, left_ops, index),
        PointwiseProof.value_reads(~Dot.Cell, ~Dot.left_index, cell_at(inner, columns, previous), U32.to_nat(inner),
          left, left_fill, left_history(inner, columns, left_ops, previous), index),
        left_history_value(inner, columns, left, left_fill, left_ops, previous, index))

def right_history_value(+inner: U32, +columns: U32, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  +count: Nat, +index: U32) ->
  {Replayed.value(right, right_fill, right_history(inner, columns, right_ops, count), index) == Replayed.value(right, right_fill, right_ops, index) : F32}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      Equal.trans(F32, Replayed.value(right, right_fill, right_history(inner, columns, right_ops, 1n+previous), index),
        Replayed.value(right, right_fill, right_history(inner, columns, right_ops, previous), index),
        Replayed.value(right, right_fill, right_ops, index),
        PointwiseProof.value_reads(~Dot.Cell, ~Dot.right_index, cell_at(inner, columns, previous), U32.to_nat(inner),
          right, right_fill, right_history(inner, columns, right_ops, previous), index),
        right_history_value(inner, columns, right, right_fill, right_ops, previous, index))

def total_history(+cell: Dot.Cell, +inner: U32, +columns: U32, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +position: Nat, +count: Nat) ->
  {dot_total(cell, left, left_fill, left_history(inner, columns, left_ops, position), right, right_fill, right_history(inner, columns, right_ops, position), count) ==
    dot_total(cell, left, left_fill, left_ops, right, right_fill, right_ops, count) : F32}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +left_index = U32.from_nat(Dot.left_index(cell, previous))
      +right_index = U32.from_nat(Dot.right_index(cell, previous))
      +seen_left = left_history(inner, columns, left_ops, position)
      +seen_right = right_history(inner, columns, right_ops, position)
      +total = dot_total(cell, left, left_fill, left_ops, right, right_fill, right_ops, previous)
      +x = Replayed.value(left, left_fill, left_ops, left_index)
      +y = Replayed.value(right, right_fill, right_ops, right_index)
      %Equal.sym(F32, dot_total(cell, left, left_fill, seen_left, right, right_fill, seen_right, previous), total,
        total_history(cell, inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, position, previous)) :
        {F32.add(_, F32.mul(Replayed.value(left, left_fill, seen_left, left_index), Replayed.value(right, right_fill, seen_right, right_index))) ==
          F32.add(total, F32.mul(x, y)) : F32}
      %Equal.sym(F32, Replayed.value(left, left_fill, seen_left, left_index), x,
        left_history_value(inner, columns, left, left_fill, left_ops, position, left_index)) :
        {F32.add(total, F32.mul(_, Replayed.value(right, right_fill, seen_right, right_index))) == F32.add(total, F32.mul(x, y)) : F32}
      %Equal.sym(F32, Replayed.value(right, right_fill, seen_right, right_index), y,
        right_history_value(inner, columns, right, right_fill, right_ops, position, right_index)) :
        {F32.add(total, F32.mul(x, _)) == F32.add(total, F32.mul(x, y)) : F32}
      {==}

def products_value(+inner: U32, +columns: U32, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +output: Shape.Shape, +count: Nat, +index: U32,
  below: {Nat.is_lt(U32.to_nat(index), count) == True{} : Bool},
  +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}) ->
  {Replayed.value(output, 0.0, products(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, count), index) ==
    product_total(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(index)) : F32}:
  match count:
    case 0n:
      Empty.absurd({Replayed.value(output, 0.0, Nil{}, index) ==
        product_total(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(index)) : F32},
        Arith.lt_zero(U32.to_nat(index), below))
    case 1n+previous:
      PointwiseProof.write_split(output, products(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, previous),
        previous, product_total(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, previous), index,
        product_total(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(index)),
        below, room,
        at => Equal.cong(Nat, F32, position => product_total(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, position),
          previous, U32.to_nat(index), Equal.sym(Nat, U32.to_nat(index), previous, at)),
        earlier => products_value(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, output, previous, index, earlier,
          Conversion.le_predecessor(previous, Shape.logical_numel(Shape.dimensions(output)), room)),
        Shape.bool_case(Nat.is_eq(U32.to_nat(index), previous)))

def output_dimensions(-left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape, -rows: U32, -inner: U32, -right_inner: U32, -columns: U32,
  facts: MatrixFacts<left, right, output, rows, inner, right_inner, columns>) -> {Shape.dimensions(output) == [rows, columns] : List<&2, U32>}:
  MatrixFacts{le, re, oe, compatible} = facts
  oe

def cell_below(+output: Shape.Shape, +rows: U32, +columns: U32, +oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
  +row: Nat, +row_below: {Nat.is_lt(row, U32.to_nat(rows)) == True{} : Bool},
  +column: Nat, +column_below: {Nat.is_lt(column, U32.to_nat(columns)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(row, U32.to_nat(columns)), column), Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}:
  %Equal.sym(List<&2, U32>, Shape.dimensions(output), [rows, columns], oe) :
    {Nat.is_lt(Nat.add(Nat.mul(row, U32.to_nat(columns)), column), Shape.logical_numel(_)) == True{} : Bool}
  %Equal.sym(Nat, Nat.mul(U32.to_nat(columns), 1n), U32.to_nat(columns), Nats.mul_one_right(U32.to_nat(columns))) :
    {Nat.is_lt(Nat.add(Nat.mul(row, U32.to_nat(columns)), column), Nat.mul(U32.to_nat(rows), _)) == True{} : Bool}
  DotProof.lt_mul_add(row, U32.to_nat(rows), column, U32.to_nat(columns), row_below, column_below)

def cell_value(+left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +output: Shape.Shape,
  +rows: U32, +inner: U32, +columns: U32, +oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
  +row: Nat, +row_below: {Nat.is_lt(row, U32.to_nat(rows)) == True{} : Bool},
  +column: Nat, +column_below: {Nat.is_lt(column, U32.to_nat(columns)) == True{} : Bool}) ->
  {Replayed.value(output, 0.0, products(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, Shape.elements(output)),
    U32.from_nat(Nat.add(Nat.mul(row, U32.to_nat(columns)), column))) ==
    DotFold.dot_fold(Laws.factors(row, column, U32.to_nat(inner), U32.to_nat(columns),
      left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0) : F32}:
  +inner_count = U32.to_nat(inner)
  +column_count = U32.to_nat(columns)
  +position = Nat.add(Nat.mul(row, column_count), column)
  +below = cell_below(output, rows, columns, oe, row, row_below, column, column_below)
  +exact = PointwiseProof.word_exact(position, output, below)
  +counted = lt_same(U32.to_nat(U32.from_nat(position)), position, Shape.elements(output), exact,
    lt_bound(position, Shape.logical_numel(Shape.dimensions(output)), Shape.elements(output),
      Equal.sym(Nat, Shape.elements(output), Shape.numel(Shape.extents(output)), Shape.elements_exact(output)), below))
  Equal.trans(F32,
    Replayed.value(output, 0.0, products(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, Shape.elements(output)),
      U32.from_nat(position)),
    product_total(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(U32.from_nat(position))),
    DotFold.dot_fold(Laws.factors(row, column, inner_count, column_count, left, left_fill, left_ops, right, right_fill, right_ops, inner_count), 0.0),
    products_value(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, output, Shape.elements(output),
      U32.from_nat(position), counted, elements_room(output)),
    %Equal.sym(Nat, U32.to_nat(U32.from_nat(position)), position, exact) :
      {product_total(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, _) ==
        DotFold.dot_fold(Laws.factors(row, column, inner_count, column_count, left, left_fill, left_ops, right, right_fill, right_ops, inner_count), 0.0) : F32}
    %Equal.sym(F32, product_total(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, position),
      dot_total(cell_at(inner, columns, position), left, left_fill, left_ops, right, right_fill, right_ops, inner_count),
      total_history(cell_at(inner, columns, position), inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, position, inner_count)) :
      {_ == DotFold.dot_fold(Laws.factors(row, column, inner_count, column_count, left, left_fill, left_ops, right, right_fill, right_ops, inner_count), 0.0) : F32}
    %Equal.sym(F32, dot_total(cell_at(inner, columns, position), left, left_fill, left_ops, right, right_fill, right_ops, inner_count),
      DotFold.dot_fold(Laws.factors(row_of(column_count, position), column_of(column_count, position), inner_count, column_count,
        left, left_fill, left_ops, right, right_fill, right_ops, inner_count), 0.0),
      DotProof.total_factors(row_of(column_count, position), column_of(column_count, position), inner_count, column_count,
        left, left_fill, left_ops, right, right_fill, right_ops, inner_count)) :
      {_ == DotFold.dot_fold(Laws.factors(row, column, inner_count, column_count, left, left_fill, left_ops, right, right_fill, right_ops, inner_count), 0.0) : F32}
    %Equal.sym(Nat, row_of(column_count, position), row, row_at(row, column, column_count, column_below)) :
      {DotFold.dot_fold(Laws.factors(_, column_of(column_count, position), inner_count, column_count,
        left, left_fill, left_ops, right, right_fill, right_ops, inner_count), 0.0) ==
        DotFold.dot_fold(Laws.factors(row, column, inner_count, column_count, left, left_fill, left_ops, right, right_fill, right_ops, inner_count), 0.0) : F32}
    %Equal.sym(Nat, column_of(column_count, position), column, column_at(row, column, column_count, column_below)) :
      {DotFold.dot_fold(Laws.factors(row, _, inner_count, column_count,
        left, left_fill, left_ops, right, right_fill, right_ops, inner_count), 0.0) ==
        DotFold.dot_fold(Laws.factors(row, column, inner_count, column_count, left, left_fill, left_ops, right, right_fill, right_ops, inner_count), 0.0) : F32}
    {==})

def kept_value(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +history: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, history)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  +index: U32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  same: {Replayed.value(shape, fill, history, index) == Replayed.value(shape, fill, ops, index) : F32}) ->
  {Core.dynamic_read_value(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, history), fits}}, index)) ==
    Some{Replayed.value(shape, fill, ops, index)} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>,
    Core.dynamic_read_value(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, history), fits}}, index)),
    Some{Replayed.value(shape, fill, history, index)}, Some{Replayed.value(shape, fill, ops, index)},
    PointwiseProof.replayed_point(shape, fill, history, fits, index, inside),
    Equal.cong(F32, Maybe<&2, F32>, word => Some{word}, Replayed.value(shape, fill, history, index), Replayed.value(shape, fill, ops, index), same))

def ordered(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
  -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
  -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  +planned: {Matrix.plan(left, right) == Done{Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible}} :
    Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>},
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat}) ->
  Records.MatmulContract<left, right, U32.to_nat(rows), U32.to_nat(columns),
    row => column => DotFold.dot_fold(Laws.factors(row, column, U32.to_nat(inner), U32.to_nat(columns),
      left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0),
    other => Replayed.value(left, left_fill, left_ops, other),
    other => Replayed.value(right, right_fill, right_ops, other),
    Matrix.matmul(Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
      Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})>:
  +plan : Matrix.Plan<left, right> = Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible}
  +facts : MatrixFacts<left, right, output, rows, inner, right_inner, columns> = plan_facts(left, right, plan, planned)
  +oe_live = output_dimensions(left, right, output, rows, inner, right_inner, columns, facts)
  +count = Shape.elements(output)
  -contract : Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic> -> Type = result =>
    Records.MatmulContract<left, right, U32.to_nat(rows), U32.to_nat(columns),
      row => column => DotFold.dot_fold(Laws.factors(row, column, U32.to_nat(inner), U32.to_nat(columns),
        left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0),
      other => Replayed.value(left, left_fill, left_ops, other),
      other => Replayed.value(right, right_fill, right_ops, other), result>
  matmul_lift(left, left_fill, left_ops, left_fits, right, right_fill, right_ops, right_fits, plan, planned, contract,
    multiply_lift(left, left_fill, left_ops, left_fits, right, right_fill, right_ops, right_fits, output, rows, inner, right_inner, columns, facts,
      result => contract(Matrix.packed(left, right, output, result)),
      a => b => o => Records.MatmulContract{
        {==},
        Equal.cong(List<&2, U32>, Maybe<&2, List<&2, Nat>>, dims => Some{Shape.naturals(dims)},
          Shape.dimensions(output), [rows, columns], oe_live),
        row => row_below => column => column_below =>
          +row_below = row_below
          +column_below = column_below
          +position = U32.from_nat(Nat.add(Nat.mul(row, U32.to_nat(columns)), column))
          +cell = DotFold.dot_fold(Laws.factors(row, column, U32.to_nat(inner), U32.to_nat(columns),
            left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0)
          Equal.trans(Maybe<&2, F32>,
            Core.dynamic_read_value(Core.read_dynamic(Core.Packed{output, Core.TTensor{Replayed.storage(output, 0.0,
              products(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, count)), o}}, position)),
            Some{Replayed.value(output, 0.0, products(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, count), position)},
            Some{cell},
            PointwiseProof.replayed_point(output, 0.0, products(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, count), o,
              position, PointwiseProof.word_inside(Nat.add(Nat.mul(row, U32.to_nat(columns)), column), output,
                cell_below(output, rows, columns, oe_live, row, row_below, column, column_below))),
            Equal.cong(F32, Maybe<&2, F32>, word => Some{word},
              Replayed.value(output, 0.0, products(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, count), position), cell,
              cell_value(left, left_fill, left_ops, right, right_fill, right_ops, output, rows, inner, columns, oe_live,
                row, row_below, column, column_below))),
        other => inside => kept_value(left, left_fill, left_ops, left_history(inner, columns, left_ops, count), a, other, inside,
          left_history_value(inner, columns, left, left_fill, left_ops, count, other)),
        other => inside => kept_value(right, right_fill, right_ops, right_history(inner, columns, right_ops, count), b, other, inside,
          right_history_value(inner, columns, right, right_fill, right_ops, count, other))}))

def matmul_ordered(+left: Shape.Shape, +right: Shape.Shape, +plan: Matrix.Plan<left, right>,
  planned: {Matrix.plan(left, right) == Done{plan} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>},
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat}) ->
  Records.MatmulContract<left, right, Laws.matrix_rows(left, right, plan), Laws.matrix_columns(left, right, plan),
    row => column => DotFold.dot_fold(Laws.factors(row, column, Laws.matrix_inner(left, right, plan), Laws.matrix_columns(left, right, plan),
      left, left_fill, left_ops, right, right_fill, right_ops, Laws.matrix_inner(left, right, plan)), 0.0),
    other => Replayed.value(left, left_fill, left_ops, other),
    other => Replayed.value(right, right_fill, right_ops, other),
    Matrix.matmul(Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
      Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})>:
  match plan:
    case Matrix.Plan{+output, +rows, +inner, +right_inner, +columns, le, re, oe, compatible}:
      ordered(left, right, output, rows, inner, right_inner, columns, le, re, oe, compatible, planned,
        left_fill, left_ops, left_fits, right_fill, right_ops, right_fits)
