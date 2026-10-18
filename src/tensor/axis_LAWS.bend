import Base
import ../shape/core.bend as Shape
import ./layout.bend as Layout
import ./axis_spec.bend as Axis
import ./axis_algebra.bend as Algebra

def axis_plan_succeeded(-shape: Shape.Shape, -axes: List<&2, U32>, result: Result<&2, &2, Layout.Error, Layout.Plan<shape, axes>>) -> Bool:
  match result:
    case Fail{error}:
      False{}
    case Done{plan}:
      True{}

law axis_lookup_refines:
  for +items: List<&2, U32>
  for +index: Nat
  {Layout.lookup(items, index) == Axis.lookup(U32, items, index) : Maybe<&2, U32>}

law axis_gather_refines:
  for +axes: List<&2, U32>
  for +items: List<&2, U32>
  {Layout.reorder(items, axes) == Axis.gather(U32, items, Shape.naturals(axes)) : Maybe<&2, List<&2, U32>>}

law axis_scatter_refines:
  for +shape: Shape.Shape
  for +axes: List<&2, U32>
  for +coordinates: List<&2, U32>
  {Layout.inverse_coordinates(Shape.dimensions(shape), axes, coordinates, 0) ==
    Axis.scatter(U32, Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes), coordinates, 0n) : Maybe<&2, List<&2, U32>>}

law axis_validity_refines:
  for +shape: Shape.Shape
  for +axes: List<&2, U32>
  for accepted: {True{} == Layout.bijective(shape, axes) : Bool}
  {True{} == Axis.valid(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes)) : Bool}

law axis_lookup_complete:
  for -A: Data
  for +items: List<&2, A>
  for +index: Nat
  for bound: Shape.LT(index, List.length(&2, A, items))
  Algebra.Selected<A, items, index>

law axis_permutation_complete:
  for +shape: Shape.Shape
  for +axes: List<&2, U32>
  for accepted: {True{} == Layout.bijective(shape, axes) : Bool}
  {Layout.reorder(Shape.dimensions(shape), axes) == Some{Axis.order(U32, Shape.dimensions(shape), Shape.naturals(axes))} : Maybe<&2, List<&2, U32>>}

law axis_permutation_rejects:
  for +shape: Shape.Shape
  for +axes: List<&2, U32>
  for rejected: {False{} == Layout.bijective(shape, axes) : Bool}
  {axis_plan_succeeded(shape, axes, Layout.permutation(shape, axes)) == False{} : Bool}

law axis_dimension_rank:
  for +shape: Shape.Shape
  for +axes: List<&2, U32>
  for accepted: {True{} == Layout.bijective(shape, axes) : Bool}
  {Shape.rank_dimensions(Axis.order(U32, Shape.dimensions(shape), Shape.naturals(axes))) == Shape.rank_dimensions(Shape.dimensions(shape)) : Nat}

law axis_dimension_product:
  for +shape: Shape.Shape
  for +axes: List<&2, U32>
  for accepted: {True{} == Layout.bijective(shape, axes) : Bool}
  {Shape.logical_numel(Axis.order(U32, Shape.dimensions(shape), Shape.naturals(axes))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}

law axis_scatter_complete:
  for +shape: Shape.Shape
  for +axes: List<&2, U32>
  for +coordinates: List<&2, U32>
  for accepted: {True{} == Layout.bijective(shape, axes) : Bool}
  for length: {Shape.rank_dimensions(coordinates) == Shape.rank_dimensions(Shape.dimensions(shape)) : Nat}
  {Layout.inverse_coordinates(Shape.dimensions(shape), axes, coordinates, 0) ==
    Some{Axis.order(U32, coordinates, Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes)))} : Maybe<&2, List<&2, U32>>}

law axis_scatter_gather:
  for +shape: Shape.Shape
  for +axes: List<&2, U32>
  for +coordinates: List<&2, U32>
  for accepted: {True{} == Layout.bijective(shape, axes) : Bool}
  for length: {Shape.rank_dimensions(coordinates) == Shape.rank_dimensions(Shape.dimensions(shape)) : Nat}
  {Layout.inverse_coordinates(Shape.dimensions(shape), axes, Axis.order(U32, coordinates, Shape.naturals(axes)), 0) == Some{coordinates} : Maybe<&2, List<&2, U32>>}

law axis_inverse_valid:
  for +rank: Nat
  for +axes: List<&2, Nat>
  for bound: Shape.LE(rank, 4n)
  for accepted: {True{} == Axis.valid(rank, axes) : Bool}
  {True{} == Axis.valid(rank, Axis.inverse_axes(rank, axes)) : Bool}

law axis_inverse_selected:
  for +rank: Nat
  for +axes: List<&2, Nat>
  for bound: Shape.LE(rank, 4n)
  for accepted: {True{} == Axis.valid(rank, axes) : Bool}
  {Axis.inverse(rank, axes) == Some{Axis.inverse_axes(rank, axes)} : Maybe<&2, List<&2, Nat>>}

law axis_inverse_involution:
  for +rank: Nat
  for +axes: List<&2, Nat>
  for bound: Shape.LE(rank, 4n)
  for accepted: {True{} == Axis.valid(rank, axes) : Bool}
  {Axis.inverse_axes(rank, Axis.inverse_axes(rank, axes)) == axes : List<&2, Nat>}

law axis_inverse_list_composition:
  for -A: Data
  for +rank: Nat
  for +axes: List<&2, Nat>
  for +items: List<&2, A>
  for bound: Shape.LE(rank, 4n)
  for accepted: {True{} == Axis.valid(rank, axes) : Bool}
  for length: {List.length(&2, A, items) == rank : Nat}
  {Axis.order(A, Axis.order(A, items, axes), Axis.inverse_axes(rank, axes)) == items : List<&2, A>}

law axis_inverse_reverse_composition:
  for -A: Data
  for +rank: Nat
  for +axes: List<&2, Nat>
  for +items: List<&2, A>
  for bound: Shape.LE(rank, 4n)
  for accepted: {True{} == Axis.valid(rank, axes) : Bool}
  for length: {List.length(&2, A, items) == rank : Nat}
  {Axis.order(A, Axis.order(A, items, Axis.inverse_axes(rank, axes)), axes) == items : List<&2, A>}

law axis_inverse_scatter_composition:
  for -A: Data
  for +rank: Nat
  for +axes: List<&2, Nat>
  for +items: List<&2, A>
  for +bound: Shape.LE(rank, 4n)
  for +accepted: {True{} == Axis.valid(rank, axes) : Bool}
  for length: {List.length(&2, A, items) == rank : Nat}
  {Axis.scatter(A, rank, Axis.inverse_axes(rank, axes), Axis.order(A, items, Axis.inverse_axes(rank, axes)), 0n) == Some{items} : Maybe<&2, List<&2, A>>}

law axis_coordinate_fit:
  for +rank: Nat
  for +axes: List<&2, Nat>
  for +dimensions: List<&2, Nat>
  for +coordinates: List<&2, Nat>
  for bound: Shape.LE(rank, 4n)
  for accepted: {True{} == Axis.valid(rank, axes) : Bool}
  for dimensions_length: {List.length(&2, Nat, dimensions) == rank : Nat}
  for coordinates_length: {List.length(&2, Nat, coordinates) == rank : Nat}
  {Shape.fit(Axis.order(Nat, dimensions, axes), Axis.order(Nat, coordinates, axes)) == Shape.fit(dimensions, coordinates) : Bool}

law axis_scatter_fit:
  for +rank: Nat
  for +axes: List<&2, Nat>
  for +dimensions: List<&2, Nat>
  for +coordinates: List<&2, Nat>
  for +bound: Shape.LE(rank, 4n)
  for +accepted: {True{} == Axis.valid(rank, axes) : Bool}
  for +dimensions_length: {List.length(&2, Nat, dimensions) == rank : Nat}
  for coordinates_length: {List.length(&2, Nat, coordinates) == rank : Nat}
  for fits: {True{} == Shape.fit(Axis.order(Nat, dimensions, axes), coordinates) : Bool}
  {True{} == Shape.fit(dimensions, Axis.order(Nat, coordinates, Axis.inverse_axes(rank, axes))) : Bool}
