import Base
import ./core.bend as Shape
import ./flat.bend as Flat

def built(-input: List<&2, U32>, result: Shape.ConstructResult<input>) -> Maybe<&2, List<&2, U32>>:
  match result:
    case Shape.SError{error}:
      None{}
    case Shape.SBuilt{shape, preserved}:
      Some{Shape.dimensions(shape)}

def index_result(-shape: Shape.Shape, result: Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{index}:
      Some{Shape.index_value(shape, index)}

def coordinate_result(-shape: Shape.Shape, result: Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<shape>>) -> Maybe<&2, List<&2, U32>>:
  match result:
    case Fail{error}:
      None{}
    case Done{coordinate}:
      Some{Shape.coordinate_values(shape, coordinate)}

def planned_dimensions(left: Shape.Shape, right: Shape.Shape, result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) -> List<&2, U32> & List<&2, U32>:
  match result:
    case Fail{error}:
      (Shape.dimensions(left), Shape.dimensions(right))
    case Done{+plan}:
      (Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(Shape.right_shape(plan)))

def broadcast_built(result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) -> Bool:
  match result:
    case Fail{error}:
      False{}
    case Done{plan}:
      True{}

def fin_of.next(-bound: Nat, index: Maybe<&2, Shape.Fin(bound)>) -> Maybe<&2, Shape.Fin(1n+bound)>:
  match index:
    case None{}:
      None{}
    case Some{previous}:
      Some{Shape.FNext{previous}}

def fin_of(+bound: Nat, +value: Nat) -> Maybe<&2, Shape.Fin(bound)>:
  match bound value:
    case 0n value:
      None{}
    case 1n+p 0n:
      Some{Shape.FZero{}}
    case 1n+p 1n+q:
      fin_of.next(p, fin_of(p, q))

def decode.step(-dh: Nat, -dt: List<&2, Nat>, tail: Maybe<&2, Shape.Coordinate(dt)>, head: Maybe<&2, Shape.Fin(dh)>) -> Maybe<&2, Shape.Coordinate(dh <> dt)>:
  match tail:
    case None{}:
      None{}
    case Some{rest}:
      match head:
        case None{}:
          None{}
        case Some{index}:
          Some{Shape.CStep{index, rest}}

def decode(+dims: List<&2, Nat>, +values: List<&2, Nat>) -> Maybe<&2, Shape.Coordinate(dims)>:
  match dims values:
    case Nil{} Nil{}:
      Some{Unit{}}
    case +dh <> +dt +vh <> +vt:
      decode.step(dh, dt, decode(dt, vt), fin_of(dh, vh))
    case Nil{} vh <> vt:
      None{}
    case dh <> dt Nil{}:
      None{}

def flatten_spec(+dims: List<&2, Nat>, coordinate: Maybe<&2, Shape.Coordinate(dims)>) -> Maybe<&2, U32>:
  match coordinate:
    case None{}:
      None{}
    case Some{coordinate}:
      Some{U32.from_nat(Shape.row_major(dims, coordinate))}

def unflatten_spec(+dims: List<&2, Nat>, index: Maybe<&2, Shape.Fin(Shape.numel(dims))>) -> Maybe<&2, List<&2, U32>>:
  match index:
    case None{}:
      None{}
    case Some{index}:
      Some{Shape.words(Shape.indices(dims, Shape.unflatten(dims, index)))}

def project_left_spec(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +alignment: Shape.Broadcast<left, right, output>, coordinate: Maybe<&2, Shape.Coordinate(output)>) -> Maybe<&2, List<&2, U32>>:
  match coordinate:
    case None{}:
      None{}
    case Some{coordinate}:
      Some{Shape.words(Shape.indices(left, Shape.project_left(left, right, output, alignment, coordinate)))}

def project_right_spec(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +alignment: Shape.Broadcast<left, right, output>, coordinate: Maybe<&2, Shape.Coordinate(output)>) -> Maybe<&2, List<&2, U32>>:
  match coordinate:
    case None{}:
      None{}
    case Some{coordinate}:
      Some{Shape.words(Shape.indices(right, Shape.project_right(left, right, output, alignment, coordinate)))}

def position(-shape: Shape.Shape, index: Shape.FlatIndex<shape>) -> Nat:
  Shape.FIndex{value, bound} = index
  value

def project_left_coordinate(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +alignment: Shape.Broadcast<left, right, output>, index: Maybe<&2, Shape.Fin(Shape.numel(output))>) -> Maybe<&2, Shape.Coordinate(left)>:
  match index:
    case None{}:
      None{}
    case Some{index}:
      Some{Shape.project_left(left, right, output, alignment, Shape.unflatten(output, index))}

def project_right_coordinate(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +alignment: Shape.Broadcast<left, right, output>, index: Maybe<&2, Shape.Fin(Shape.numel(output))>) -> Maybe<&2, Shape.Coordinate(right)>:
  match index:
    case None{}:
      None{}
    case Some{index}:
      Some{Shape.project_right(left, right, output, alignment, Shape.unflatten(output, index))}

law row_major_bounded:
  for +dims: List<&2, Nat>
  for +coordinate: Shape.Coordinate(dims)
  Shape.LT(Shape.row_major(dims, coordinate), Shape.numel(dims))

law flatten_unflatten:
  for +dims: List<&2, Nat>
  for +index: Shape.Fin(Shape.numel(dims))
  {Shape.flatten(dims, Shape.unflatten(dims, index)) == index : Shape.Fin(Shape.numel(dims))}

law unflatten_flatten:
  for +dims: List<&2, Nat>
  for +coordinate: Shape.Coordinate(dims)
  {Shape.unflatten(dims, Shape.flatten(dims, coordinate)) == coordinate : Shape.Coordinate(dims)}

law flatten_row_major:
  for +dims: List<&2, Nat>
  for +coordinate: Shape.Coordinate(dims)
  {Shape.Fin.value(Shape.numel(dims), Shape.flatten(dims, coordinate)) == Shape.row_major(dims, coordinate) : Nat}

law broadcast_left_projection:
  for +left: List<&2, Nat>
  for +right: List<&2, Nat>
  for +output: List<&2, Nat>
  for +alignment: Shape.Broadcast<left, right, output>
  for +coordinate: Shape.Coordinate(output)
  {Shape.indices(left, Shape.project_left(left, right, output, alignment, coordinate)) == Shape.project_indices(left, Shape.indices(output, coordinate)) : List<&2, Nat>}

law broadcast_right_projection:
  for +left: List<&2, Nat>
  for +right: List<&2, Nat>
  for +output: List<&2, Nat>
  for +alignment: Shape.Broadcast<left, right, output>
  for +coordinate: Shape.Coordinate(output)
  {Shape.indices(right, Shape.project_right(left, right, output, alignment, coordinate)) == Shape.project_indices(right, Shape.indices(output, coordinate)) : List<&2, Nat>}

law construction_complete:
  for +input: List<&2, U32>
  for rank_bound: Shape.LE(Shape.rank_dimensions(input), 4n)
  for capacity_bound: Shape.LE(Shape.logical_numel(input), 1048576n)
  {built(input, Shape.construct(input)) == Some{input} : Maybe<&2, List<&2, U32>>}

law flatten_checked_refines:
  for +shape: Shape.Shape
  for +values: List<&2, U32>
  {index_result(shape, Shape.flatten_checked(shape, values)) == flatten_spec(Shape.extents(shape), decode(Shape.extents(shape), Shape.naturals(values))) : Maybe<&2, U32>}

law unflatten_checked_refines:
  for +shape: Shape.Shape
  for +index: U32
  {coordinate_result(shape, Shape.unflatten_checked(shape, index)) == unflatten_spec(Shape.extents(shape), fin_of(Shape.numel(Shape.extents(shape)), U32.to_nat(index))) : Maybe<&2, List<&2, U32>>}

law broadcast_output_unique:
  for +left: List<&2, Nat>
  for +right: List<&2, Nat>
  for +output: List<&2, Nat>
  for +other: List<&2, Nat>
  for +first: Shape.Broadcast<left, right, output>
  for +second: Shape.Broadcast<left, right, other>
  {output == other : List<&2, Nat>}

law broadcast_checked_plan:
  for +left: Shape.Shape
  for +right: Shape.Shape
  {planned_dimensions(left, right, Shape.broadcast_checked(left, right)) == (Shape.dimensions(left), Shape.dimensions(right)) : List<&2, U32> & List<&2, U32>}

law broadcast_checked_complete:
  for +left: Shape.Shape
  for +right: Shape.Shape
  for +output: List<&2, U32>
  for +alignment: Shape.Broadcast<Shape.extents(left), Shape.extents(right), Shape.naturals(output)>
  for +rank_bound: Shape.LE(Shape.rank_dimensions(output), 4n)
  for +capacity_bound: Shape.LE(Shape.logical_numel(output), 1048576n)
  {broadcast_built(Shape.broadcast_checked(left, right)) == True{} : Bool}

law project_left_checked_refines:
  for +plan: Shape.BroadcastPlan
  for +values: List<&2, U32>
  {coordinate_result(Shape.left_shape(plan), Shape.project_left_checked(plan, values)) == project_left_spec(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.alignment(plan), decode(Shape.extents(Shape.output_shape(plan)), Shape.naturals(values))) : Maybe<&2, List<&2, U32>>}

law project_right_checked_refines:
  for +plan: Shape.BroadcastPlan
  for +values: List<&2, U32>
  {coordinate_result(Shape.right_shape(plan), Shape.project_right_checked(plan, values)) == project_right_spec(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.alignment(plan), decode(Shape.extents(Shape.output_shape(plan)), Shape.naturals(values))) : Maybe<&2, List<&2, U32>>}

law project_left_index_refines:
  for +plan: Shape.BroadcastPlan
  for +map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>
  for +index: Shape.FlatIndex<Shape.output_shape(plan)>
  {Some{Shape.index_value(Shape.left_shape(plan), Flat.project_left_index(plan, map, index))} == flatten_spec(Shape.extents(Shape.left_shape(plan)), project_left_coordinate(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.alignment(plan), fin_of(Shape.numel(Shape.extents(Shape.output_shape(plan))), position(Shape.output_shape(plan), index)))) : Maybe<&2, U32>}

law project_right_index_refines:
  for +plan: Shape.BroadcastPlan
  for +map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>
  for +index: Shape.FlatIndex<Shape.output_shape(plan)>
  {Some{Shape.index_value(Shape.right_shape(plan), Flat.project_right_index(plan, map, index))} == flatten_spec(Shape.extents(Shape.right_shape(plan)), project_right_coordinate(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.alignment(plan), fin_of(Shape.numel(Shape.extents(Shape.output_shape(plan))), position(Shape.output_shape(plan), index)))) : Maybe<&2, U32>}

# A coordinate whose rank differs from the shape's fails with
# CoordinateRankMismatch, checked before any bound: [9] on [2, 3] is a rank
# error, not an out-of-bounds error.
law flatten_rank_error:
  for +shape: Shape.Shape
  for +values: List<&2, U32>
  for differ: {Shape.rank_dimensions(Shape.dimensions(shape)) != Shape.rank_dimensions(values) : Nat}
  {Shape.flatten_checked(shape, values) == Fail{Shape.CoordinateRankMismatch{}} : Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>}

# A coordinate of the shape's rank with a component outside its extent fails
# with CoordinateOutOfBounds.
law flatten_bounds_error:
  for +shape: Shape.Shape
  for +values: List<&2, U32>
  for same: {Shape.rank_dimensions(Shape.dimensions(shape)) == Shape.rank_dimensions(values) : Nat}
  for outside: {Shape.fit_words(Shape.dimensions(shape), values) == False{} : Bool}
  {Shape.flatten_checked(shape, values) == Fail{Shape.CoordinateOutOfBounds{}} : Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>}
