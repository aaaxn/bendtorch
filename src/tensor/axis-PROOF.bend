import Base
import ../shape/core.bend as Shape
import ../shape/nat_proofs.bend as Nats
import ./layout.bend as Layout
import ./axis-spec.bend as Axis
import ./axis-algebra.bend as Algebra
import ./axis-LAWS.bend as Laws

def inverse_selected(-rank: Nat, -axes: List<&2, Nat>, certificate: Algebra.Facts<rank, axes>) ->
  {Axis.inverse(rank, axes) == Some{Axis.inverse_axes(rank, axes)} : Maybe<&2, List<&2, Nat>>}:
  Algebra.Certified{iv, selected, restored, lists, product, fit} = certificate
  selected

def inverse_involution(-rank: Nat, -axes: List<&2, Nat>, certificate: Algebra.Facts<rank, axes>) ->
  {Axis.inverse_axes(rank, Axis.inverse_axes(rank, axes)) == axes : List<&2, Nat>}:
  Algebra.Certified{iv, selected, restored, lists, product, fit} = certificate
  restored

def reverse_composition(-A: Data, -rank: Nat, -axes: List<&2, Nat>, -items: List<&2, A>, certificate: Algebra.ListFacts<A, rank, axes, items>) ->
  {Axis.order(A, Axis.order(A, items, Axis.inverse_axes(rank, axes)), axes) == items : List<&2, A>}:
  Algebra.Lists{selected, ranked, scattered, restored, reverse_restored, scatter_restored} = certificate
  reverse_restored

def coordinate_fit(-rank: Nat, -axes: List<&2, Nat>, +dimensions: List<&2, Nat>, +coordinates: List<&2, Nat>,
  dimensions_length: {List.length(&2, Nat, dimensions) == rank : Nat}, coordinates_length: {List.length(&2, Nat, coordinates) == rank : Nat}, certificate: Algebra.Facts<rank, axes>) ->
  {Shape.fit(Axis.order(Nat, dimensions, axes), Axis.order(Nat, coordinates, axes)) == Shape.fit(dimensions, coordinates) : Bool}:
  Algebra.Certified{iv, selected, restored, lists, product, fit} = certificate
  fit(dimensions, coordinates, dimensions_length, coordinates_length)

def plan_valid(-shape: Shape.Shape, -axes: List<&2, U32>, plan: Layout.Plan<shape, axes>) -> {True{} == Layout.bijective(shape, axes) : Bool}:
  Layout.Permutation{output, valid, ordered, count} = plan
  valid

def permutation_rejects(+shape: Shape.Shape, +axes: List<&2, U32>, result: Result<&2, &2, Layout.Error, Layout.Plan<shape, axes>>, rejected: {False{} == Layout.bijective(shape, axes) : Bool}) ->
  {Laws.axis_plan_succeeded(shape, axes, result) == False{} : Bool}:
  match result:
    case Fail{error}:
      {==}
    case Done{plan}:
      Empty.absurd({True{} == False{} : Bool}, Nats.contradiction(Layout.bijective(shape, axes), plan_valid(shape, axes, plan), rejected))

def Laws.axis_lookup_refines(items, index):
  Algebra.lookup(items, index)

def Laws.axis_gather_refines(axes, items):
  Algebra.reorder(axes, items)

def Laws.axis_scatter_refines(shape, axes, coordinates):
  Algebra.inverse_coordinates(shape, axes, coordinates)

def Laws.axis_validity_refines(shape, axes, accepted):
  Algebra.valid(shape, axes, accepted)

def Laws.axis_lookup_complete(A, items, index, bound):
  Algebra.lookup_complete(A, items, index, bound)

def Laws.axis_permutation_complete(shape, axes, accepted):
  Algebra.permutation_complete(shape, axes, accepted)

def Laws.axis_permutation_rejects(shape, axes, rejected):
  permutation_rejects(shape, axes, Layout.permutation(shape, axes), rejected)

def Laws.axis_dimension_rank(shape, axes, accepted):
  Algebra.permutation_rank(shape, axes, accepted)

def Laws.axis_dimension_product(shape, axes, accepted):
  Algebra.permutation_product(shape, axes, accepted)

def Laws.axis_scatter_complete(shape, axes, coordinates, accepted, length):
  Algebra.scatter_complete(shape, axes, coordinates, accepted, length)

def Laws.axis_scatter_gather(shape, axes, coordinates, accepted, length):
  Algebra.scatter_gather(shape, axes, coordinates, accepted, length)

def Laws.axis_inverse_valid(rank, axes, bound, accepted):
  Algebra.inverse_valid(rank, axes, bound, accepted)

def Laws.axis_inverse_selected(rank, axes, bound, accepted):
  inverse_selected(rank, axes, Algebra.facts(rank, axes, bound, accepted))

def Laws.axis_inverse_involution(rank, axes, bound, accepted):
  inverse_involution(rank, axes, Algebra.facts(rank, axes, bound, accepted))

def Laws.axis_inverse_list_composition(A, rank, axes, items, bound, accepted, length):
  Algebra.inverse_composition(A, rank, axes, items, bound, accepted, length)

def Laws.axis_inverse_reverse_composition(A, rank, axes, items, bound, accepted, length):
  reverse_composition(A, rank, axes, items, Algebra.facts_lists(A, rank, axes, items, length, Algebra.facts(rank, axes, bound, accepted)))

def Laws.axis_inverse_scatter_composition(A, rank, axes, items, bound, accepted, length):
  Algebra.inverse_scatter_composition(A, rank, axes, items, bound, accepted, length)

def Laws.axis_coordinate_fit(rank, axes, dimensions, coordinates, bound, accepted, dimensions_length, coordinates_length):
  coordinate_fit(rank, axes, dimensions, coordinates, dimensions_length, coordinates_length, Algebra.facts(rank, axes, bound, accepted))

def Laws.axis_scatter_fit(rank, axes, dimensions, coordinates, bound, accepted, dimensions_length, coordinates_length, fits):
  Equal.trans(Bool, True{}, Shape.fit(Axis.order(Nat, dimensions, axes), coordinates), Shape.fit(dimensions, Axis.order(Nat, coordinates, Axis.inverse_axes(rank, axes))), fits,
    Equal.trans(Bool, Shape.fit(Axis.order(Nat, dimensions, axes), coordinates),
      Shape.fit(Axis.order(Nat, Axis.order(Nat, dimensions, axes), Axis.inverse_axes(rank, axes)), Axis.order(Nat, coordinates, Axis.inverse_axes(rank, axes))),
      Shape.fit(dimensions, Axis.order(Nat, coordinates, Axis.inverse_axes(rank, axes))),
      Equal.sym(Bool, Shape.fit(Axis.order(Nat, Axis.order(Nat, dimensions, axes), Axis.inverse_axes(rank, axes)), Axis.order(Nat, coordinates, Axis.inverse_axes(rank, axes))),
        Shape.fit(Axis.order(Nat, dimensions, axes), coordinates),
        coordinate_fit(rank, Axis.inverse_axes(rank, axes), Axis.order(Nat, dimensions, axes), coordinates,
          Algebra.ranked(Nat, rank, axes, dimensions, Algebra.facts_lists(Nat, rank, axes, dimensions, dimensions_length, Algebra.facts(rank, axes, bound, accepted))),
          coordinates_length, Algebra.facts(rank, Axis.inverse_axes(rank, axes), bound, Algebra.inverse_valid(rank, axes, bound, accepted)))),
      Equal.cong(List<&2, Nat>, Bool, dims => Shape.fit(dims, Axis.order(Nat, coordinates, Axis.inverse_axes(rank, axes))),
        Axis.order(Nat, Axis.order(Nat, dimensions, axes), Axis.inverse_axes(rank, axes)), dimensions,
        Algebra.inverse_composition(Nat, rank, axes, dimensions, bound, accepted, dimensions_length))))
