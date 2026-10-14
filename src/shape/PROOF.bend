import Base
import ./core.bend as Shape
import ./index_proofs.bend as Index
import ./broadcast_proofs.bend as Broadcast
import ./nat_proofs.bend as Nats
import ./construct_proofs.bend as Construct
import ./boundary_proofs.bend as Boundary
import ./unique_proofs.bend as Unique
import ./plan_proofs.bend as Plan
import ./LAWS.bend as Laws

def Laws.row_major_bounded(dims, coordinate):
  %Index.flatten_row_major(dims, coordinate) : Shape.LT(_, Shape.numel(dims))
  Shape.Fin.bound(Shape.numel(dims), Shape.flatten(dims, coordinate))

def Laws.flatten_unflatten(dims, index):
  Index.flatten_unflatten(dims, index)

def Laws.unflatten_flatten(dims, coordinate):
  Index.unflatten_flatten(dims, coordinate)

def Laws.flatten_row_major(dims, coordinate):
  Index.flatten_row_major(dims, coordinate)

def Laws.broadcast_left_projection(left, right, output, alignment, coordinate):
  Broadcast.project_left(left, right, output, alignment, coordinate)

def Laws.broadcast_right_projection(left, right, output, alignment, coordinate):
  Broadcast.project_right(left, right, output, alignment, coordinate)

def Laws.construction_complete(input, rank_bound, capacity_bound):
  Construct.construction_complete(input, Shape.bool_case(Nat.is_le(Shape.rank_dimensions(input), 4n)), Nats.le_true(Shape.rank_dimensions(input), 4n, rank_bound), capacity_bound)

def Laws.flatten_checked_refines(shape, values):
  Boundary.flatten_checked(shape, values)

def Laws.unflatten_checked_refines(shape, index):
  Boundary.unflatten_checked(shape, index)

def Laws.broadcast_output_unique(left, right, output, other, first, second):
  Unique.broadcast_unique(left, right, output, other, first, second)

def Laws.broadcast_checked_plan(left, right):
  Plan.broadcast_checked_plan(left, right)

def Laws.broadcast_checked_complete(left, right, output, alignment, rank_bound, capacity_bound):
  Plan.broadcast_checked_complete(left, right, output, alignment, rank_bound, capacity_bound)

def Laws.project_left_checked_refines(plan, values):
  Boundary.project_left_checked(plan, values)

def Laws.project_right_checked_refines(plan, values):
  Boundary.project_right_checked(plan, values)

def Laws.project_left_index_refines(plan, map, index):
  Boundary.project_left_index(plan, map, index)

def Laws.project_right_index_refines(plan, map, index):
  Boundary.project_right_index(plan, map, index)

def flatten_fits_bounds(+shape: Shape.Shape, +values: List<&2, U32>, verdict: Shape.BoolCase<Shape.fit_words(Shape.dimensions(shape), values)>,
  outside: {Shape.fit_words(Shape.dimensions(shape), values) == False{} : Bool}) ->
  {Shape.flatten_checked.fits(shape, values, verdict) == Fail{Shape.CoordinateOutOfBounds{}} : Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>}:
  match verdict:
    case Shape.BFalse{equal}:
      {==}
    case Shape.BTrue{equal}:
      Empty.absurd({Shape.flatten_checked.fits(shape, values, Shape.BTrue{equal}) == Fail{Shape.CoordinateOutOfBounds{}} : Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>},
        Shape.clash(Equal.trans(Bool, True{}, Shape.fit_words(Shape.dimensions(shape), values), False{}, equal, outside)))

def flatten_ranked_rank(+shape: Shape.Shape, +values: List<&2, U32>,
  same_rank: Shape.BoolCase<Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.rank_dimensions(values))>,
  differ: {Shape.rank_dimensions(Shape.dimensions(shape)) != Shape.rank_dimensions(values) : Nat}) ->
  {Shape.flatten_checked.ranked(shape, values, same_rank) == Fail{Shape.CoordinateRankMismatch{}} : Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>}:
  match same_rank:
    case Shape.BFalse{equal}:
      {==}
    case Shape.BTrue{equal}:
      Empty.absurd({Shape.flatten_checked.ranked(shape, values, Shape.BTrue{equal}) == Fail{Shape.CoordinateRankMismatch{}} : Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>},
        differ(Shape.equal_decided(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.rank_dimensions(values), equal)))

def flatten_ranked_bounds(+shape: Shape.Shape, +values: List<&2, U32>,
  same_rank: Shape.BoolCase<Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.rank_dimensions(values))>,
  same: {Shape.rank_dimensions(Shape.dimensions(shape)) == Shape.rank_dimensions(values) : Nat},
  outside: {Shape.fit_words(Shape.dimensions(shape), values) == False{} : Bool}) ->
  {Shape.flatten_checked.ranked(shape, values, same_rank) == Fail{Shape.CoordinateOutOfBounds{}} : Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>}:
  match same_rank:
    case Shape.BFalse{equal}:
      Empty.absurd({Shape.flatten_checked.ranked(shape, values, Shape.BFalse{equal}) == Fail{Shape.CoordinateOutOfBounds{}} : Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>},
        Shape.clash(Equal.trans(Bool, True{}, Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.rank_dimensions(values)), False{},
          Plan.eq_true(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.rank_dimensions(values), same),
          Equal.sym(Bool, False{}, Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.rank_dimensions(values)), equal))))
    case Shape.BTrue{equal}:
      flatten_fits_bounds(shape, values, Shape.bool_case(Shape.fit_words(Shape.dimensions(shape), values)), outside)

def Laws.flatten_rank_error(shape, values, differ):
  flatten_ranked_rank(shape, values, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.rank_dimensions(values))), differ)

def Laws.flatten_bounds_error(shape, values, same, outside):
  flatten_ranked_bounds(shape, values, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.rank_dimensions(values))), same, outside)
