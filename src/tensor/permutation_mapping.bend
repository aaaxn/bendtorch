import Base
import ../shape/core.bend as Shape
import ./layout.bend as Layout
import ./lifecycle.bend as Lifecycle
import ../storage/arith.bend as Arith
import ../shape/coordinate_proofs.bend as Coordinates
import ../shape/nat_proofs.bend as Nats
import ./word_conversion.bend as WordConversion
import ../shape/LAWS.bend as ShapeLaws
import ../shape/PROOF.bend as ShapeProof
import ./axis_spec.bend as Axis
import ./axis_algebra.bend as AxisAlgebra

def permutation_checked(+input: Shape.Shape, +axes: List<&2, U32>, plan: Layout.Plan<input, axes>) ->
  {True{} == Layout.bijective(input, axes) : Bool}:
  Layout.Permutation{out, valid, ordered, count} = plan
  valid

def and_left(a: Bool, -b: Bool, equal: {True{} == Bool.and(a, b) : Bool}) -> {True{} == a : Bool}:
  match a:
    case False{}:
      equal
    case True{}:
      {==}

def and_right(a: Bool, -b: Bool, equal: {True{} == Bool.and(a, b) : Bool}) -> {True{} == b : Bool}:
  match a:
    case False{}:
      Empty.absurd({True{} == b : Bool}, Lifecycle.false_true(equal))
    case True{}:
      equal

def reverse_go_cancel(+values: List<&2, U32>, +accumulator: List<&2, U32>) ->
  {Layout.reverse_words_go(Layout.reverse_words_go(values, accumulator), Nil{}) ==
    Layout.reverse_words_go(accumulator, values) : List<&2, U32>}:
  match values:
    case Nil{}:
      {==}
    case +head <> +tail:
      reverse_go_cancel(tail, head <> accumulator)

def reverse_involution(+values: List<&2, U32>) -> {Layout.reverse_words(Layout.reverse_words(values)) == values : List<&2, U32>}:
  reverse_go_cancel(values, Nil{})

def codec_decoded(+dimensions: List<&2, U32>, +index: U32) ->
  {Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(dimensions), index)) ==
    Shape.words(Shape.digits(Shape.naturals(dimensions), U32.to_nat(index))) : List<&2, U32>}:
  Equal.trans(List<&2, U32>, Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(dimensions), index)),
    Shape.words(Shape.digits_words(dimensions, U32.to_nat(index))),
    Shape.words(Shape.digits(Shape.naturals(dimensions), U32.to_nat(index))),
    Equal.trans(List<&2, U32>, Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(dimensions), index)),
      Layout.reverse_words(Layout.reverse_words(Shape.words(Shape.digits_words(dimensions, U32.to_nat(index))))),
      Shape.words(Shape.digits_words(dimensions, U32.to_nat(index))),
      Equal.cong(List<&2, U32>, List<&2, U32>, dims =>
        Layout.reverse_words(Layout.reverse_words(Shape.words(Shape.digits_words(dims, U32.to_nat(index))))),
        Layout.reverse_words(Layout.reverse_words(dimensions)), dimensions, reverse_involution(dimensions)),
      reverse_involution(Shape.words(Shape.digits_words(dimensions, U32.to_nat(index))))),
    Equal.cong(List<&2, Nat>, List<&2, U32>, Shape.words, Shape.digits_words(dimensions, U32.to_nat(index)),
      Shape.digits(Shape.naturals(dimensions), U32.to_nat(index)), Coordinates.digits_words_spec(dimensions, U32.to_nat(index))))

def coordinate_fits(+dimensions: List<&2, U32>, +values: List<&2, U32>, valid: Layout.WordsValid(dimensions, values)) ->
  Coordinates.Fits(Shape.naturals(dimensions), Shape.naturals(values)):
  match dimensions values:
    case Nil{} Nil{}:
      Unit{}
    case +dimension <> +dt +value <> +vt:
      Layout.WStep{bound, tail} = valid
      Coordinates.FStep{
        %Equal.sym(Bool, Nat.is_lt(U32.to_nat(value), U32.to_nat(dimension)), U32.is_lt(value, dimension), Arith.u32_lt(value, dimension)) :
          {True{} == _ : Bool}
        bound,
        coordinate_fits(dt, vt, tail)}
    case Nil{} head <> tail:
      match valid:
    case head <> tail Nil{}:
      match valid:

def codec_flattened(+dimensions: List<&2, U32>, +values: List<&2, U32>, +valid: Layout.WordsValid(dimensions, values)) ->
  {Layout.flatten_words(dimensions, values, 0) == U32.from_nat(Shape.offset(Shape.naturals(dimensions), Shape.naturals(values))) : U32}:
  Equal.cong(Nat, U32, U32.from_nat, Shape.horner(dimensions, values, 0n), Shape.offset(Shape.naturals(dimensions), Shape.naturals(values)),
    Coordinates.horner_spec(dimensions, values, 0n, coordinate_fits(dimensions, values, valid)))

def mapping_reference(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  mapping: Layout.Mapping<input, axes, plan, index>) ->
  {Some{Layout.source_index(input, axes, plan, index, mapping)} == Layout.reference_source(input, axes, plan, index) : Maybe<&2, U32>}:
  Layout.Mapped{oi, oc, ic, source, requested, decoded, inverse, flattened, roundtrip} = mapping
  Layout.CWord{+values, +valid} = ic
  %AxisAlgebra.inverse_coordinates(input, axes,
    Shape.words(Shape.digits(Shape.extents(Layout.output(input, axes, plan)), U32.to_nat(index)))) :
      {Some{Layout.index_value(input, source)} == Layout.reference_source_coordinates(Shape.dimensions(input), _) : Maybe<&2, U32>}
  %Equal.trans(List<&2, U32>, Layout.coordinate_values(Layout.output(input, axes, plan), oc),
      Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(Layout.output(input, axes, plan))), index)),
      Shape.words(Shape.digits(Shape.extents(Layout.output(input, axes, plan)), U32.to_nat(index))), decoded,
      codec_decoded(Shape.dimensions(Layout.output(input, axes, plan)), index)) :
        {Some{Layout.index_value(input, source)} == Layout.reference_source_coordinates(Shape.dimensions(input),
          Layout.inverse_coordinates(Shape.dimensions(input), axes, _, 0)) : Maybe<&2, U32>}
  %inverse : {Some{Layout.index_value(input, source)} == Layout.reference_source_coordinates(Shape.dimensions(input), _) : Maybe<&2, U32>}
  Equal.cong(U32, Maybe<&2, U32>, value => Some{value}, Layout.index_value(input, source),
    U32.from_nat(Shape.offset(Shape.extents(input), Shape.naturals(values))),
    Equal.trans(U32, Layout.index_value(input, source), Layout.flatten_words(Shape.dimensions(input), values, 0),
      U32.from_nat(Shape.offset(Shape.extents(input), Shape.naturals(values))), flattened,
      codec_flattened(Shape.dimensions(input), values, valid)))

def flat_mathematical_bounded(+shape: Shape.Shape, index: Layout.FlatIndex<shape>) ->
  {Nat.is_lt(U32.to_nat(Layout.index_value(shape, index)), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}:
  Layout.FIndex{+value, bound} = index
  %Shape.elements_exact(shape) : {Nat.is_lt(U32.to_nat(value), _) == True{} : Bool}
  Arith.lt_le_trans(U32.to_nat(value), U32.to_nat(Shape.count(shape)), Shape.elements(shape),
    Equal.trans(Bool, Nat.is_lt(U32.to_nat(value), U32.to_nat(Shape.count(shape))), U32.is_lt(value, Shape.count(shape)), True{},
      Arith.u32_lt(value, Shape.count(shape)), Equal.sym(Bool, True{}, U32.is_lt(value, Shape.count(shape)), bound)),
    WordConversion.natural_word_le(Shape.elements(shape)))

def decoded_coordinate(+shape: Shape.Shape, +index: U32,
  +inside: {True{} == Nat.is_lt(U32.to_nat(index), Shape.elements(shape)) : Bool}) -> Shape.Coordinate(Shape.extents(shape)):
  Shape.unflatten(Shape.extents(shape), Coordinates.fin_from(Shape.numel(Shape.extents(shape)), U32.to_nat(index),
    Coordinates.size_fits(shape, U32.to_nat(index), inside)))

def decoded_indices(+shape: Shape.Shape, +index: U32,
  +inside: {True{} == Nat.is_lt(U32.to_nat(index), Shape.elements(shape)) : Bool}) ->
  {Shape.indices(Shape.extents(shape), decoded_coordinate(shape, index, inside)) ==
    Shape.digits(Shape.extents(shape), U32.to_nat(index)) : List<&2, Nat>}:
  Equal.trans(List<&2, Nat>, Shape.indices(Shape.extents(shape), decoded_coordinate(shape, index, inside)),
    Shape.digits(Shape.extents(shape), Shape.Fin.value(Shape.numel(Shape.extents(shape)),
      Coordinates.fin_from(Shape.numel(Shape.extents(shape)), U32.to_nat(index), Coordinates.size_fits(shape, U32.to_nat(index), inside)))),
    Shape.digits(Shape.extents(shape), U32.to_nat(index)),
    Coordinates.digits_unflatten(Shape.extents(shape),
      Coordinates.fin_from(Shape.numel(Shape.extents(shape)), U32.to_nat(index), Coordinates.size_fits(shape, U32.to_nat(index), inside))),
    Equal.cong(Nat, List<&2, Nat>, value => Shape.digits(Shape.extents(shape), value),
      Shape.Fin.value(Shape.numel(Shape.extents(shape)),
        Coordinates.fin_from(Shape.numel(Shape.extents(shape)), U32.to_nat(index), Coordinates.size_fits(shape, U32.to_nat(index), inside))),
      U32.to_nat(index), Coordinates.fin_from_value(Shape.numel(Shape.extents(shape)), U32.to_nat(index),
        Coordinates.size_fits(shape, U32.to_nat(index), inside))))

def decoded_fit(+shape: Shape.Shape, +index: U32,
  +inside: {True{} == Nat.is_lt(U32.to_nat(index), Shape.elements(shape)) : Bool}) ->
  {True{} == Shape.fit(Shape.extents(shape), Shape.digits(Shape.extents(shape), U32.to_nat(index))) : Bool}:
  %decoded_indices(shape, index, inside) : {True{} == Shape.fit(Shape.extents(shape), _) : Bool}
  Coordinates.fit_indices(Shape.extents(shape), decoded_coordinate(shape, index, inside))

def decoded_word_fits(+shape: Shape.Shape, +index: U32,
  +inside: {True{} == Nat.is_lt(U32.to_nat(index), Shape.elements(shape)) : Bool}) ->
  Coordinates.Fits(Shape.extents(shape), Shape.naturals(Shape.words(Shape.digits(Shape.extents(shape), U32.to_nat(index))))):
  %Equal.sym(List<&2, Nat>, Shape.naturals(Shape.words(Shape.digits(Shape.extents(shape), U32.to_nat(index)))),
    Shape.digits(Shape.extents(shape), U32.to_nat(index)),
    WordConversion.naturals_words_fitting(Shape.dimensions(shape), Shape.digits(Shape.extents(shape), U32.to_nat(index)), decoded_fit(shape, index, inside))) :
      Coordinates.Fits(Shape.extents(shape), _)
  Coordinates.fits_of(Shape.extents(shape), Shape.digits(Shape.extents(shape), U32.to_nat(index)), decoded_fit(shape, index, inside))

def element_counts_same(+input: Shape.Shape, +target: Shape.Shape,
  same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}) ->
  {Shape.elements(input) == Shape.elements(target) : Nat}:
  Equal.trans(Nat, Shape.elements(input), Shape.logical_numel(Shape.dimensions(input)), Shape.elements(target), Shape.elements_exact(input),
    Equal.trans(Nat, Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(target)), Shape.elements(target), same,
      Equal.sym(Nat, Shape.elements(target), Shape.logical_numel(Shape.dimensions(target)), Shape.elements_exact(target))))

def natural_equal_true(+left: Nat, +right: Nat, same: {left == right : Nat}) -> {Nat.is_eq(left, right) == True{} : Bool}:
  %same : {Nat.is_eq(left, _) == True{} : Bool}
  Arith.eq_refl(left)

def metadata_present(value: Maybe<&2, List<&2, U32>>) -> Bool:
  match value:
    case None{}:
      False{}
    case Some{values}:
      True{}

def reordered_rank_bound(+input: Shape.Shape, +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool}) ->
  Shape.LE(Shape.rank_dimensions(Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))), 4n):
  match input:
    case Shape.SShape{+dimensions, +size, +exact, +rank_bound, +capacity_bound}:
      %Equal.sym(Nat, Shape.rank_dimensions(Axis.order(U32, dimensions, Shape.naturals(axes))), Shape.rank_dimensions(dimensions),
        AxisAlgebra.permutation_rank(Shape.SShape{dimensions, size, exact, rank_bound, capacity_bound}, axes, valid)) : Shape.LE(_, 4n)
      rank_bound

def reordered_capacity_bound(+input: Shape.Shape, +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool}) ->
  Shape.LE(Shape.logical_numel(Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))), 1048576n):
  match input:
    case Shape.SShape{+dimensions, +size, +exact, +rank_bound, +capacity_bound}:
      %Equal.sym(Nat, Shape.logical_numel(Axis.order(U32, dimensions, Shape.naturals(axes))), Shape.logical_numel(dimensions),
        AxisAlgebra.permutation_product(Shape.SShape{dimensions, size, exact, rank_bound, capacity_bound}, axes, valid)) : Shape.LE(_, 1048576n)
      capacity_bound

def reordered_construct_complete(+input: Shape.Shape, +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool}) ->
  {ShapeLaws.built(Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)),
    Shape.construct(Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)))) ==
      Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))} : Maybe<&2, List<&2, U32>>}:
  ShapeLaws.construction_complete(Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)),
    reordered_rank_bound(input, axes, valid), reordered_capacity_bound(input, axes, valid))

def selected_construct_complete(+input: Shape.Shape, +axes: List<&2, U32>, +dimensions: List<&2, U32>,
  +valid: {True{} == Layout.bijective(input, axes) : Bool},
  same: {dimensions == Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)) : List<&2, U32>}) ->
  {ShapeLaws.built(dimensions, Shape.construct(dimensions)) == Some{dimensions} : Maybe<&2, List<&2, U32>>}:
  %Equal.sym(List<&2, U32>, dimensions, Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)), same) :
    {ShapeLaws.built(_, Shape.construct(_)) == Some{_} : Maybe<&2, List<&2, U32>>}
  reordered_construct_complete(input, axes, valid)

def reordered_elements_same(+input: Shape.Shape, +axes: List<&2, U32>, +output: Shape.Shape,
  +valid: {True{} == Layout.bijective(input, axes) : Bool},
  preserved: {Shape.dimensions(output) == Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)) : List<&2, U32>}) ->
  {Shape.elements(input) == Shape.elements(output) : Nat}:
  element_counts_same(input, output,
    Equal.trans(Nat, Shape.logical_numel(Shape.dimensions(input)),
      Shape.logical_numel(Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))), Shape.logical_numel(Shape.dimensions(output)),
      Equal.sym(Nat, Shape.logical_numel(Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))), Shape.logical_numel(Shape.dimensions(input)),
        AxisAlgebra.permutation_product(input, axes, valid)),
      Equal.cong(List<&2, U32>, Nat, Shape.logical_numel, Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)), Shape.dimensions(output),
        Equal.sym(List<&2, U32>, Shape.dimensions(output), Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)), preserved))))

def plan_counted_success(+input: Shape.Shape, +axes: List<&2, U32>, +output: Shape.Shape,
  +valid: {True{} == Layout.bijective(input, axes) : Bool},
  +ordered: {Some{Shape.dimensions(output)} == Layout.reorder(Shape.dimensions(input), axes) : Maybe<&2, List<&2, U32>>},
  decision: Shape.BoolCase<Nat.is_eq(Shape.elements(input), Shape.elements(output))>,
  same: {Shape.elements(input) == Shape.elements(output) : Nat}) ->
  {Result.is_done(&2, &2, Layout.Error, Layout.Plan<input, axes>, Layout.plan_counted(input, axes, output, valid, ordered, decision)) == True{} : Bool}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({False{} == True{} : Bool},
        Arith.clash(Equal.trans(Bool, False{}, Nat.is_eq(Shape.elements(input), Shape.elements(output)), True{}, equal,
          natural_equal_true(Shape.elements(input), Shape.elements(output), same))))
    case Shape.BTrue{equal}:
      {==}

def plan_built_success(+input: Shape.Shape, +axes: List<&2, U32>, +dimensions: List<&2, U32>,
  +valid: {True{} == Layout.bijective(input, axes) : Bool},
  +selected: {Some{dimensions} == Layout.reorder(Shape.dimensions(input), axes) : Maybe<&2, List<&2, U32>>},
  built: Shape.ConstructResult<dimensions>,
  complete: {ShapeLaws.built(dimensions, built) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
  +same: {dimensions == Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)) : List<&2, U32>}) ->
  {Result.is_done(&2, &2, Layout.Error, Layout.Plan<input, axes>, Layout.plan_built(input, axes, dimensions, valid, selected, built)) == True{} : Bool}:
  match built:
    case Shape.SError{error}:
      Empty.absurd({False{} == True{} : Bool},
        Arith.clash(Equal.cong(Maybe<&2, List<&2, U32>>, Bool, metadata_present, None{}, Some{dimensions}, complete)))
    case Shape.SBuilt{+output, +preserved}:
      plan_counted_success(input, axes, output, valid,
        Equal.trans(Maybe<&2, List<&2, U32>>, Some{Shape.dimensions(output)}, Some{dimensions}, Layout.reorder(Shape.dimensions(input), axes),
          Equal.cong(List<&2, U32>, Maybe<&2, List<&2, U32>>, values => Some{values}, Shape.dimensions(output), dimensions, preserved), selected),
        Shape.bool_case(Nat.is_eq(Shape.elements(input), Shape.elements(output))),
        reordered_elements_same(input, axes, output, valid,
          Equal.trans(List<&2, U32>, Shape.dimensions(output), dimensions, Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)), preserved, same)))

def plan_selected_success(+input: Shape.Shape, +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool},
  selected: Maybe<&2, List<&2, U32>>, +same: {selected == Layout.reorder(Shape.dimensions(input), axes) : Maybe<&2, List<&2, U32>>}) ->
  {Result.is_done(&2, &2, Layout.Error, Layout.Plan<input, axes>, Layout.plan_selected(input, axes, valid, selected, same)) == True{} : Bool}:
  match selected:
    case None{}:
      Empty.absurd({False{} == True{} : Bool},
        Arith.clash(Equal.cong(Maybe<&2, List<&2, U32>>, Bool, metadata_present, None{},
          Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))},
          Equal.trans(Maybe<&2, List<&2, U32>>, None{}, Layout.reorder(Shape.dimensions(input), axes),
            Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))}, same, AxisAlgebra.permutation_complete(input, axes, valid)))))
    case Some{+dimensions}:
      plan_built_success(input, axes, dimensions, valid, same, Shape.construct(dimensions),
        selected_construct_complete(input, axes, dimensions, valid,
          Equal.cong(Maybe<&2, List<&2, U32>>, List<&2, U32>, Axis.values(U32), Some{dimensions},
            Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))},
            Equal.trans(Maybe<&2, List<&2, U32>>, Some{dimensions}, Layout.reorder(Shape.dimensions(input), axes),
              Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))}, same, AxisAlgebra.permutation_complete(input, axes, valid)))),
        Equal.cong(Maybe<&2, List<&2, U32>>, List<&2, U32>, Axis.values(U32), Some{dimensions},
          Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))},
          Equal.trans(Maybe<&2, List<&2, U32>>, Some{dimensions}, Layout.reorder(Shape.dimensions(input), axes),
            Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))}, same, AxisAlgebra.permutation_complete(input, axes, valid))))

def plan_decided_success(+input: Shape.Shape, +axes: List<&2, U32>, decision: Shape.BoolCase<Layout.bijective(input, axes)>,
  +valid: {True{} == Layout.bijective(input, axes) : Bool}) ->
  {Result.is_done(&2, &2, Layout.Error, Layout.Plan<input, axes>, Layout.plan_decided(input, axes, decision)) == True{} : Bool}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({False{} == True{} : Bool},
        Arith.clash(Equal.trans(Bool, False{}, Layout.bijective(input, axes), True{}, equal,
          Equal.sym(Bool, True{}, Layout.bijective(input, axes), valid))))
    case Shape.BTrue{equal}:
      plan_selected_success(input, axes, equal, Layout.reorder(Shape.dimensions(input), axes), {==})

def plan_unique_success(+input: Shape.Shape, +axes: List<&2, U32>, decision: Shape.BoolCase<Layout.unique(axes)>,
  +valid: {True{} == Layout.bijective(input, axes) : Bool}) ->
  {Result.is_done(&2, &2, Layout.Error, Layout.Plan<input, axes>, Layout.plan_unique(input, axes, decision)) == True{} : Bool}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({False{} == True{} : Bool},
        Arith.clash(Equal.trans(Bool, False{}, Layout.unique(axes), True{}, equal,
          Equal.sym(Bool, True{}, Layout.unique(axes),
            and_right(Layout.bounded(axes, Shape.rank(input)), Layout.unique(axes),
              and_right(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))),
                Bool.and(Layout.bounded(axes, Shape.rank(input)), Layout.unique(axes)), valid))))))
    case Shape.BTrue{equal}:
      plan_decided_success(input, axes, Shape.bool_case(Layout.bijective(input, axes)), valid)

def plan_bounded_success(+input: Shape.Shape, +axes: List<&2, U32>, decision: Shape.BoolCase<Layout.bounded(axes, Shape.rank(input))>,
  +valid: {True{} == Layout.bijective(input, axes) : Bool}) ->
  {Result.is_done(&2, &2, Layout.Error, Layout.Plan<input, axes>, Layout.plan_bounded(input, axes, decision)) == True{} : Bool}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({False{} == True{} : Bool},
        Arith.clash(Equal.trans(Bool, False{}, Layout.bounded(axes, Shape.rank(input)), True{}, equal,
          Equal.sym(Bool, True{}, Layout.bounded(axes, Shape.rank(input)),
            and_left(Layout.bounded(axes, Shape.rank(input)), Layout.unique(axes),
              and_right(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))),
                Bool.and(Layout.bounded(axes, Shape.rank(input)), Layout.unique(axes)), valid))))))
    case Shape.BTrue{equal}:
      plan_unique_success(input, axes, Shape.bool_case(Layout.unique(axes)), valid)

def plan_ranked_success(+input: Shape.Shape, +axes: List<&2, U32>,
  decision: Shape.BoolCase<Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))>,
  +valid: {True{} == Layout.bijective(input, axes) : Bool}) ->
  {Result.is_done(&2, &2, Layout.Error, Layout.Plan<input, axes>, Layout.plan_ranked(input, axes, decision)) == True{} : Bool}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({False{} == True{} : Bool},
        Arith.clash(Equal.trans(Bool, False{}, Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))), True{}, equal,
          Equal.sym(Bool, True{}, Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))),
            and_left(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))),
              Bool.and(Layout.bounded(axes, Shape.rank(input)), Layout.unique(axes)), valid)))))
    case Shape.BTrue{equal}:
      plan_bounded_success(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input))), valid)

def words_valid_of(+dimensions: List<&2, U32>, +values: List<&2, U32>,
  fits: Coordinates.Fits(Shape.naturals(dimensions), Shape.naturals(values))) -> Layout.WordsValid(dimensions, values):
  match dimensions values:
    case Nil{} Nil{}:
      Unit{}
    case +dimension <> +dt +value <> +vt:
      Coordinates.FStep{bound, tail} = fits
      Layout.WStep{
        %Arith.u32_lt(value, dimension) : {True{} == _ : Bool}
        bound,
        words_valid_of(dt, vt, tail)}
    case Nil{} head <> tail:
      match fits:
    case head <> tail Nil{}:
      match fits:

def words_finish_success(-dimension: U32, -dt: List<&2, U32>, -value: U32, -vt: List<&2, U32>,
  bound: {True{} == U32.is_lt(value, dimension) : Bool}, result: Result<&2, &2, Shape.Error, Layout.WordsValid(dt, vt)>,
  successful: {Result.is_done(&2, &2, Shape.Error, Layout.WordsValid(dt, vt), result) == True{} : Bool}) ->
  {Result.is_done(&2, &2, Shape.Error, Layout.WordsValid(dimension <> dt, value <> vt),
    Layout.words_finish(dimension, dt, value, vt, bound, result)) == True{} : Bool}:
  match result:
    case Fail{error}:
      Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))
    case Done{tail}:
      {==}

def words_decided_success(+dimension: U32, +dt: List<&2, U32>, +value: U32, +vt: List<&2, U32>,
  decision: Shape.BoolCase<U32.is_lt(value, dimension)>, bound: {True{} == U32.is_lt(value, dimension) : Bool},
  successful: {Result.is_done(&2, &2, Shape.Error, Layout.WordsValid(dt, vt), Layout.words_check(dt, vt)) == True{} : Bool}) ->
  {Result.is_done(&2, &2, Shape.Error, Layout.WordsValid(dimension <> dt, value <> vt),
    Layout.words_decided(dimension, dt, value, vt, unit => Layout.words_check(dt, vt), decision)) == True{} : Bool}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({False{} == True{} : Bool}, Lifecycle.false_true(Equal.trans(Bool, True{}, U32.is_lt(value, dimension), False{}, bound,
        Equal.sym(Bool, False{}, U32.is_lt(value, dimension), equal))))
    case Shape.BTrue{equal}:
      words_finish_success(dimension, dt, value, vt, equal, Layout.words_check(dt, vt), successful)

def words_check_success(+dimensions: List<&2, U32>, +values: List<&2, U32>, valid: Layout.WordsValid(dimensions, values)) ->
  {Result.is_done(&2, &2, Shape.Error, Layout.WordsValid(dimensions, values), Layout.words_check(dimensions, values)) == True{} : Bool}:
  match dimensions values:
    case Nil{} Nil{}:
      {==}
    case +dimension <> +dt +value <> +vt:
      Layout.WStep{bound, tail} = valid
      words_decided_success(dimension, dt, value, vt, Shape.bool_case(U32.is_lt(value, dimension)), bound, words_check_success(dt, vt, tail))
    case Nil{} head <> tail:
      match valid:
    case head <> tail Nil{}:
      match valid:

def decoded_words_valid(+shape: Shape.Shape, +index: U32,
  +inside: {True{} == Nat.is_lt(U32.to_nat(index), Shape.elements(shape)) : Bool}) ->
  Layout.WordsValid(Shape.dimensions(shape), Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(shape)), index))):
  %Equal.sym(List<&2, U32>, Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(shape)), index)),
    Shape.words(Shape.digits(Shape.extents(shape), U32.to_nat(index))), codec_decoded(Shape.dimensions(shape), index)) :
      Layout.WordsValid(Shape.dimensions(shape), _)
  words_valid_of(Shape.dimensions(shape), Shape.words(Shape.digits(Shape.extents(shape), U32.to_nat(index))), decoded_word_fits(shape, index, inside))

def coordinate_offset_bound(+shape: Shape.Shape, +values: List<&2, U32>, +valid: Layout.WordsValid(Shape.dimensions(shape), values)) ->
  Shape.LT(Shape.offset(Shape.extents(shape), Shape.naturals(values)), Shape.elements(shape)):
  %Equal.sym(Nat, Shape.elements(shape), Shape.numel(Shape.extents(shape)), Shape.elements_exact(shape)) :
    Shape.LT(Shape.offset(Shape.extents(shape), Shape.naturals(values)), _)
  %Coordinates.offset_of(Shape.extents(shape), Shape.naturals(values), coordinate_fits(Shape.dimensions(shape), values, valid)) :
    Shape.LT(_, Shape.numel(Shape.extents(shape)))
  Coordinates.row_major_lt(Shape.extents(shape), Coordinates.coordinate_of(Shape.extents(shape), Shape.naturals(values),
    coordinate_fits(Shape.dimensions(shape), values, valid)))

def coordinate_offset_exact(+shape: Shape.Shape, +values: List<&2, U32>, +valid: Layout.WordsValid(Shape.dimensions(shape), values)) ->
  {U32.to_nat(U32.from_nat(Shape.offset(Shape.extents(shape), Shape.naturals(values)))) == Shape.offset(Shape.extents(shape), Shape.naturals(values)) : Nat}:
  WordConversion.natural_word_exact(Shape.offset(Shape.extents(shape), Shape.naturals(values)), Shape.count(shape),
    %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.elements(shape), WordConversion.shape_count_exact(shape)) :
      {Nat.is_le(Shape.offset(Shape.extents(shape), Shape.naturals(values)), _) == True{} : Bool}
    WordConversion.lt_le(Shape.offset(Shape.extents(shape), Shape.naturals(values)), Shape.elements(shape),
      Equal.sym(Bool, True{}, Nat.is_lt(Shape.offset(Shape.extents(shape), Shape.naturals(values)), Shape.elements(shape)),
        Nats.lt_true(Shape.offset(Shape.extents(shape), Shape.naturals(values)), Shape.elements(shape), coordinate_offset_bound(shape, values, valid)))))

def coordinate_flatten_exact(+shape: Shape.Shape, +values: List<&2, U32>, +valid: Layout.WordsValid(Shape.dimensions(shape), values)) ->
  {U32.to_nat(Layout.flatten_words(Shape.dimensions(shape), values, 0)) == Shape.offset(Shape.extents(shape), Shape.naturals(values)) : Nat}:
  Equal.trans(Nat, U32.to_nat(Layout.flatten_words(Shape.dimensions(shape), values, 0)),
    U32.to_nat(U32.from_nat(Shape.offset(Shape.extents(shape), Shape.naturals(values)))), Shape.offset(Shape.extents(shape), Shape.naturals(values)),
    Equal.cong(U32, Nat, U32.to_nat, Layout.flatten_words(Shape.dimensions(shape), values, 0),
      U32.from_nat(Shape.offset(Shape.extents(shape), Shape.naturals(values))), codec_flattened(Shape.dimensions(shape), values, valid)),
    coordinate_offset_exact(shape, values, valid))

def words_valid_rank(+dimensions: List<&2, U32>, +values: List<&2, U32>, valid: Layout.WordsValid(dimensions, values)) ->
  {Shape.rank_dimensions(values) == Shape.rank_dimensions(dimensions) : Nat}:
  match dimensions values:
    case Nil{} Nil{}:
      {==}
    case +dimension <> +dt +value <> +vt:
      Layout.WStep{bound, tail} = valid
      Equal.cong(Nat, Nat, n => 1n+n, Shape.rank_dimensions(vt), Shape.rank_dimensions(dt), words_valid_rank(dt, vt, tail))
    case Nil{} head <> tail:
      match valid:
    case head <> tail Nil{}:
      match valid:

def words_fit_true(+dimensions: List<&2, U32>, +values: List<&2, U32>, +valid: Layout.WordsValid(dimensions, values)) ->
  {True{} == Shape.fit(Shape.naturals(dimensions), Shape.naturals(values)) : Bool}:
  %Coordinates.indices_of(Shape.naturals(dimensions), Shape.naturals(values), coordinate_fits(dimensions, values, valid)) :
    {True{} == Shape.fit(Shape.naturals(dimensions), _) : Bool}
  Coordinates.fit_indices(Shape.naturals(dimensions), Coordinates.coordinate_of(Shape.naturals(dimensions), Shape.naturals(values),
    coordinate_fits(dimensions, values, valid)))

def permutation_dimensions_order(+input: Shape.Shape, +axes: List<&2, U32>, plan: Layout.Plan<input, axes>) ->
  {Shape.dimensions(Layout.output(input, axes, plan)) == Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)) : List<&2, U32>}:
  Layout.Permutation{+output, valid, ordered, count} = plan
  Equal.cong(Maybe<&2, List<&2, U32>>, List<&2, U32>, Axis.values(U32), Some{Shape.dimensions(output)},
    Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))},
    Equal.trans(Maybe<&2, List<&2, U32>>, Some{Shape.dimensions(output)}, Layout.reorder(Shape.dimensions(input), axes),
      Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))}, ordered, AxisAlgebra.permutation_complete(input, axes, valid)))

def shape_rank_bound(+shape: Shape.Shape) -> Shape.LE(Shape.rank_dimensions(Shape.dimensions(shape)), 4n):
  match shape:
    case Shape.SShape{dimensions, size, exact, ranked, capacity}:
      ranked

def reverse_restored(-A: Data, -rank: Nat, -axes: List<&2, Nat>, -items: List<&2, A>, certificate: AxisAlgebra.ListFacts<A, rank, axes, items>) ->
  {Axis.order(A, Axis.order(A, items, Axis.inverse_axes(rank, axes)), axes) == items : List<&2, A>}:
  AxisAlgebra.Lists{selected, ranked, scattered, restored, reverse, scatter_restored} = certificate
  reverse

def natural_coordinate_length(+input: Shape.Shape, +coordinates: List<&2, U32>,
  length: {Shape.rank_dimensions(coordinates) == Shape.rank_dimensions(Shape.dimensions(input)) : Nat}) ->
  {List.length(&2, Nat, Shape.naturals(coordinates)) == Shape.rank_dimensions(Shape.dimensions(input)) : Nat}:
  Equal.trans(Nat, List.length(&2, Nat, Shape.naturals(coordinates)), Shape.rank_dimensions(coordinates), Shape.rank_dimensions(Shape.dimensions(input)),
    Equal.sym(Nat, Shape.rank_dimensions(coordinates), List.length(&2, Nat, Shape.naturals(coordinates)), Shape.rank_naturals(coordinates)), length)

def inverse_coordinate_length(+input: Shape.Shape, +axes: List<&2, U32>, +coordinates: List<&2, U32>,
  accepted: {True{} == Layout.bijective(input, axes) : Bool},
  length: {Shape.rank_dimensions(coordinates) == Shape.rank_dimensions(Shape.dimensions(input)) : Nat}) ->
  {List.length(&2, Nat, Axis.order(Nat, Shape.naturals(coordinates), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)))) ==
    Shape.rank_dimensions(Shape.dimensions(input)) : Nat}:
  AxisAlgebra.ranked(Nat, Shape.rank_dimensions(Shape.dimensions(input)), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)),
    Shape.naturals(coordinates),
    AxisAlgebra.facts_lists(Nat, Shape.rank_dimensions(Shape.dimensions(input)), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)),
      Shape.naturals(coordinates), natural_coordinate_length(input, coordinates, length),
      AxisAlgebra.facts(Shape.rank_dimensions(Shape.dimensions(input)), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)),
        shape_rank_bound(input), AxisAlgebra.facts_inverse_valid(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes), AxisAlgebra.shape_facts(input, axes, accepted)))))

def inverse_coordinate_fit_certified(+input: Shape.Shape, +axes: List<&2, U32>, +coordinates: List<&2, U32>,
  +accepted: {True{} == Layout.bijective(input, axes) : Bool},
  +length: {Shape.rank_dimensions(coordinates) == Shape.rank_dimensions(Shape.dimensions(input)) : Nat},
  fitted: {True{} == Shape.fit(Shape.naturals(Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))), Shape.naturals(coordinates)) : Bool},
  certificate: AxisAlgebra.Facts<Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)>) ->
  {True{} == Shape.fit(Shape.extents(input),
    Shape.naturals(Axis.order(U32, coordinates, Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))))) : Bool}:
  AxisAlgebra.Certified{iv, selected, restored, lists, product, fit} = certificate
  %Equal.sym(List<&2, Nat>,
    Shape.naturals(Axis.order(U32, coordinates, Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)))),
    Axis.order(Nat, Shape.naturals(coordinates), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))),
    AxisAlgebra.order_naturals(Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)), coordinates)) :
      {True{} == Shape.fit(Shape.extents(input), _) : Bool}
  Equal.trans(Bool, True{},
    Shape.fit(Axis.order(Nat, Shape.extents(input), Shape.naturals(axes)),
      Axis.order(Nat, Axis.order(Nat, Shape.naturals(coordinates), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))), Shape.naturals(axes))),
    Shape.fit(Shape.extents(input), Axis.order(Nat, Shape.naturals(coordinates), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)))),
    %AxisAlgebra.order_naturals(Shape.naturals(axes), Shape.dimensions(input)) :
      {True{} == Shape.fit(_, Axis.order(Nat, Axis.order(Nat, Shape.naturals(coordinates), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))), Shape.naturals(axes))) : Bool}
    %Equal.sym(List<&2, Nat>,
      Axis.order(Nat, Axis.order(Nat, Shape.naturals(coordinates), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))), Shape.naturals(axes)),
      Shape.naturals(coordinates), reverse_restored(Nat, Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes), Shape.naturals(coordinates),
        AxisAlgebra.facts_lists(Nat, Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes), Shape.naturals(coordinates),
          natural_coordinate_length(input, coordinates, length), AxisAlgebra.shape_facts(input, axes, accepted)))) :
            {True{} == Shape.fit(Shape.naturals(Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))), _) : Bool}
    fitted,
    fit(Shape.extents(input), Axis.order(Nat, Shape.naturals(coordinates), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))),
      Equal.sym(Nat, Shape.rank_dimensions(Shape.dimensions(input)), List.length(&2, Nat, Shape.extents(input)), Shape.rank_naturals(Shape.dimensions(input))),
      inverse_coordinate_length(input, axes, coordinates, accepted, length)))

def inverse_coordinate_fit(+input: Shape.Shape, +axes: List<&2, U32>, +coordinates: List<&2, U32>,
  +accepted: {True{} == Layout.bijective(input, axes) : Bool},
  length: {Shape.rank_dimensions(coordinates) == Shape.rank_dimensions(Shape.dimensions(input)) : Nat},
  fitted: {True{} == Shape.fit(Shape.naturals(Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))), Shape.naturals(coordinates)) : Bool}) ->
  {True{} == Shape.fit(Shape.extents(input),
    Shape.naturals(Axis.order(U32, coordinates, Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))))) : Bool}:
  inverse_coordinate_fit_certified(input, axes, coordinates, accepted, length, fitted, AxisAlgebra.shape_facts(input, axes, accepted))

def output_coordinate_rank(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +coordinate: Layout.CheckedCoordinate<Layout.output(input, axes, plan)>) ->
  {Shape.rank_dimensions(Layout.coordinate_values(Layout.output(input, axes, plan), coordinate)) == Shape.rank_dimensions(Shape.dimensions(input)) : Nat}:
  Layout.Permutation{+output, +accepted, +ordered, +count} = plan
  Layout.CWord{+values, valid} = coordinate
  Equal.trans(Nat, Shape.rank_dimensions(values), Shape.rank_dimensions(Shape.dimensions(output)), Shape.rank_dimensions(Shape.dimensions(input)),
    words_valid_rank(Shape.dimensions(output), values, valid),
    Equal.trans(Nat, Shape.rank_dimensions(Shape.dimensions(output)), Shape.rank_dimensions(Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))),
      Shape.rank_dimensions(Shape.dimensions(input)),
      Equal.cong(List<&2, U32>, Nat, Shape.rank_dimensions, Shape.dimensions(output), Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)),
        permutation_dimensions_order(input, axes, Layout.Permutation{output, accepted, ordered, count})), AxisAlgebra.permutation_rank(input, axes, accepted)))

def inverse_words_valid(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +coordinate: Layout.CheckedCoordinate<Layout.output(input, axes, plan)>,
  +values: List<&2, U32>, inverse: {Some{values} == Layout.inverse_coordinates(Shape.dimensions(input), axes, Layout.coordinate_values(Layout.output(input, axes, plan), coordinate), 0) : Maybe<&2, List<&2, U32>>}) ->
  Layout.WordsValid(Shape.dimensions(input), values):
  Layout.Permutation{+output, +accepted, +ordered, +count} = plan
  Layout.CWord{+coordinates, +valid} = coordinate
  %Equal.sym(List<&2, U32>, values, Axis.order(U32, coordinates, Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))),
    Equal.cong(Maybe<&2, List<&2, U32>>, List<&2, U32>, Axis.values(U32), Some{values},
      Some{Axis.order(U32, coordinates, Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)))},
      Equal.trans(Maybe<&2, List<&2, U32>>, Some{values}, Layout.inverse_coordinates(Shape.dimensions(input), axes, coordinates, 0),
        Some{Axis.order(U32, coordinates, Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)))}, inverse,
        AxisAlgebra.scatter_complete(input, axes, coordinates, accepted,
          output_coordinate_rank(input, axes, Layout.Permutation{output, accepted, ordered, count}, Layout.CWord{coordinates, valid}))))) :
            Layout.WordsValid(Shape.dimensions(input), _)
  words_valid_of(Shape.dimensions(input), Axis.order(U32, coordinates, Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))),
    Coordinates.fits_of(Shape.extents(input),
      Shape.naturals(Axis.order(U32, coordinates, Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)))),
      inverse_coordinate_fit(input, axes, coordinates, accepted,
        output_coordinate_rank(input, axes, Layout.Permutation{output, accepted, ordered, count}, Layout.CWord{coordinates, valid}),
        %permutation_dimensions_order(input, axes, Layout.Permutation{output, accepted, ordered, count}) :
          {True{} == Shape.fit(Shape.naturals(_), Shape.naturals(coordinates)) : Bool}
        words_fit_true(Shape.dimensions(output), coordinates, valid))))

def coordinate_flatten_bounded(+shape: Shape.Shape, +values: List<&2, U32>, +valid: Layout.WordsValid(Shape.dimensions(shape), values)) ->
  {True{} == U32.is_lt(Layout.flatten_words(Shape.dimensions(shape), values, 0), Shape.count(shape)) : Bool}:
  %Arith.u32_lt(Layout.flatten_words(Shape.dimensions(shape), values, 0), Shape.count(shape)) : {True{} == _ : Bool}
  %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.elements(shape), WordConversion.shape_count_exact(shape)) :
    {True{} == Nat.is_lt(U32.to_nat(Layout.flatten_words(Shape.dimensions(shape), values, 0)), _) : Bool}
  %Equal.sym(Nat, U32.to_nat(Layout.flatten_words(Shape.dimensions(shape), values, 0)), Shape.offset(Shape.extents(shape), Shape.naturals(values)),
    coordinate_flatten_exact(shape, values, valid)) : {True{} == Nat.is_lt(_, Shape.elements(shape)) : Bool}
  Nats.lt_true(Shape.offset(Shape.extents(shape), Shape.naturals(values)), Shape.elements(shape), coordinate_offset_bound(shape, values, valid))

def checked_coordinate_flatten_bounded(+shape: Shape.Shape, coordinate: Layout.CheckedCoordinate<shape>) ->
  {True{} == U32.is_lt(Layout.flatten_words(Shape.dimensions(shape), Layout.coordinate_values(shape, coordinate), 0), Shape.count(shape)) : Bool}:
  Layout.CWord{values, valid} = coordinate
  coordinate_flatten_bounded(shape, values, valid)

def word_same_refl(+width: Nat, +word: Word(width)) -> {Layout.word_same(width, word, word) == True{} : Bool}:
  match width:
    case 0n:
      {==}
    case 1n+p:
      match word:
        case WCon{False{}, +tail}:
          word_same_refl(p, tail)
        case WCon{True{}, +tail}:
          word_same_refl(p, tail)

def word_value_same_refl(+value: U32) -> {Layout.word_value_same(value, value) == True{} : Bool}:
  match value:
    case U32{+word}:
      word_same_refl(32n, word)

def words_same_refl(+values: List<&2, U32>) -> {Layout.words_same(values, values) == True{} : Bool}:
  match values:
    case Nil{}:
      {==}
    case +head <> +tail:
      %Equal.sym(Bool, Layout.word_value_same(head, head), True{}, word_value_same_refl(head)) :
        {Bool.and(_, Layout.words_same(tail, tail)) == True{} : Bool}
      words_same_refl(tail)

def inverse_words_order(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +coordinate: Layout.CheckedCoordinate<Layout.output(input, axes, plan)>,
  +values: List<&2, U32>, inverse: {Some{values} == Layout.inverse_coordinates(Shape.dimensions(input), axes, Layout.coordinate_values(Layout.output(input, axes, plan), coordinate), 0) : Maybe<&2, List<&2, U32>>}) ->
  {values == Axis.order(U32, Layout.coordinate_values(Layout.output(input, axes, plan), coordinate),
    Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))) : List<&2, U32>}:
  Layout.Permutation{output, accepted, ordered, count} = plan
  Equal.cong(Maybe<&2, List<&2, U32>>, List<&2, U32>, Axis.values(U32), Some{values},
    Some{Axis.order(U32, Layout.coordinate_values(output, coordinate), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)))},
    Equal.trans(Maybe<&2, List<&2, U32>>, Some{values}, Layout.inverse_coordinates(Shape.dimensions(input), axes, Layout.coordinate_values(output, coordinate), 0),
      Some{Axis.order(U32, Layout.coordinate_values(output, coordinate), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)))}, inverse,
      AxisAlgebra.scatter_complete(input, axes, Layout.coordinate_values(output, coordinate), accepted,
        output_coordinate_rank(input, axes, Layout.Permutation{output, accepted, ordered, count}, coordinate))))

def inverse_words_reordered(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +coordinate: Layout.CheckedCoordinate<Layout.output(input, axes, plan)>,
  +values: List<&2, U32>, +inverse: {Some{values} == Layout.inverse_coordinates(Shape.dimensions(input), axes, Layout.coordinate_values(Layout.output(input, axes, plan), coordinate), 0) : Maybe<&2, List<&2, U32>>}) ->
  {Layout.reorder(values, axes) == Some{Layout.coordinate_values(Layout.output(input, axes, plan), coordinate)} : Maybe<&2, List<&2, U32>>}:
  Layout.Permutation{+output, +accepted, +ordered, +count} = plan
  Equal.trans(Maybe<&2, List<&2, U32>>, Layout.reorder(values, axes), Axis.gather(U32, values, Shape.naturals(axes)),
    Some{Layout.coordinate_values(output, coordinate)}, AxisAlgebra.reorder(axes, values),
    Equal.trans(Maybe<&2, List<&2, U32>>, Axis.gather(U32, values, Shape.naturals(axes)), Some{Axis.order(U32, values, Shape.naturals(axes))},
      Some{Layout.coordinate_values(output, coordinate)},
      AxisAlgebra.selected(U32, Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes), values,
        AxisAlgebra.facts_lists(U32, Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes), values,
          Equal.trans(Nat, List.length(&2, U32, values), Shape.rank_dimensions(values), Shape.rank_dimensions(Shape.dimensions(input)),
            AxisAlgebra.word_length(values), words_valid_rank(Shape.dimensions(input), values,
              inverse_words_valid(input, axes, Layout.Permutation{output, accepted, ordered, count}, coordinate, values, inverse))),
          AxisAlgebra.shape_facts(input, axes, accepted))),
      Equal.cong(List<&2, U32>, Maybe<&2, List<&2, U32>>, items => Some{items}, Axis.order(U32, values, Shape.naturals(axes)),
        Layout.coordinate_values(output, coordinate),
        Equal.trans(List<&2, U32>, Axis.order(U32, values, Shape.naturals(axes)),
          Axis.order(U32, Axis.order(U32, Layout.coordinate_values(output, coordinate), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))), Shape.naturals(axes)),
          Layout.coordinate_values(output, coordinate),
          Equal.cong(List<&2, U32>, List<&2, U32>, items => Axis.order(U32, items, Shape.naturals(axes)), values,
            Axis.order(U32, Layout.coordinate_values(output, coordinate), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))),
            inverse_words_order(input, axes, Layout.Permutation{output, accepted, ordered, count}, coordinate, values, inverse)),
          reverse_restored(U32, Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes), Layout.coordinate_values(output, coordinate),
            AxisAlgebra.facts_lists(U32, Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes), Layout.coordinate_values(output, coordinate),
              Equal.trans(Nat, List.length(&2, U32, Layout.coordinate_values(output, coordinate)), Shape.rank_dimensions(Layout.coordinate_values(output, coordinate)),
                Shape.rank_dimensions(Shape.dimensions(input)), AxisAlgebra.word_length(Layout.coordinate_values(output, coordinate)),
                output_coordinate_rank(input, axes, Layout.Permutation{output, accepted, ordered, count}, coordinate)), AxisAlgebra.shape_facts(input, axes, accepted)))))))

def inverse_words_roundtrip(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +coordinate: Layout.CheckedCoordinate<Layout.output(input, axes, plan)>,
  +values: List<&2, U32>, inverse: {Some{values} == Layout.inverse_coordinates(Shape.dimensions(input), axes, Layout.coordinate_values(Layout.output(input, axes, plan), coordinate), 0) : Maybe<&2, List<&2, U32>>}) ->
  {True{} == Layout.maybe_same(Layout.reorder(values, axes), Some{Layout.coordinate_values(Layout.output(input, axes, plan), coordinate)}) : Bool}:
  %Equal.sym(Maybe<&2, List<&2, U32>>, Layout.reorder(values, axes), Some{Layout.coordinate_values(Layout.output(input, axes, plan), coordinate)},
    inverse_words_reordered(input, axes, plan, coordinate, values, inverse)) :
      {True{} == Layout.maybe_same(_, Some{Layout.coordinate_values(Layout.output(input, axes, plan), coordinate)}) : Bool}
  Equal.sym(Bool, Layout.words_same(Layout.coordinate_values(Layout.output(input, axes, plan), coordinate), Layout.coordinate_values(Layout.output(input, axes, plan), coordinate)), True{},
    words_same_refl(Layout.coordinate_values(Layout.output(input, axes, plan), coordinate)))

def mapping_flat_success(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  oi: Layout.FlatIndex<Layout.output(input, axes, plan)>, oc: Layout.CheckedCoordinate<Layout.output(input, axes, plan)>, +ic: Layout.CheckedCoordinate<input>,
  requested: {Layout.index_value(Layout.output(input, axes, plan), oi) == index : U32},
  decoded: {Layout.coordinate_values(Layout.output(input, axes, plan), oc) == Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(Layout.output(input, axes, plan))), index)) : List<&2, U32>},
  inverse: {Some{Layout.coordinate_values(input, ic)} == Layout.inverse_coordinates(Shape.dimensions(input), axes, Layout.coordinate_values(Layout.output(input, axes, plan), oc), 0) : Maybe<&2, List<&2, U32>>},
  roundtrip: {True{} == Layout.maybe_same(Layout.reorder(Layout.coordinate_values(input, ic), axes), Some{Layout.coordinate_values(Layout.output(input, axes, plan), oc)}) : Bool},
  +source_index: U32, flat: {source_index == Layout.flatten_words(Shape.dimensions(input), Layout.coordinate_values(input, ic), 0) : U32},
  decision: Shape.BoolCase<U32.is_lt(source_index, Shape.count(input))>, bounded: {True{} == U32.is_lt(source_index, Shape.count(input)) : Bool}) ->
  {Result.is_done(&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>,
    Layout.map_flat(input, axes, plan, index, oi, oc, ic, requested, decoded, inverse, roundtrip, source_index, flat, decision)) == True{} : Bool}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({False{} == True{} : Bool}, Arith.clash(Equal.trans(Bool, False{}, U32.is_lt(source_index, Shape.count(input)), True{}, equal,
        Equal.sym(Bool, True{}, U32.is_lt(source_index, Shape.count(input)), bounded))))
    case Shape.BTrue{equal}:
      {==}

def mapping_roundtrip_success(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  +oi: Layout.FlatIndex<Layout.output(input, axes, plan)>, +oc: Layout.CheckedCoordinate<Layout.output(input, axes, plan)>, +ic: Layout.CheckedCoordinate<input>,
  +requested: {Layout.index_value(Layout.output(input, axes, plan), oi) == index : U32},
  +decoded: {Layout.coordinate_values(Layout.output(input, axes, plan), oc) == Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(Layout.output(input, axes, plan))), index)) : List<&2, U32>},
  +inverse: {Some{Layout.coordinate_values(input, ic)} == Layout.inverse_coordinates(Shape.dimensions(input), axes, Layout.coordinate_values(Layout.output(input, axes, plan), oc), 0) : Maybe<&2, List<&2, U32>>},
  decision: Shape.BoolCase<Layout.maybe_same(Layout.reorder(Layout.coordinate_values(input, ic), axes), Some{Layout.coordinate_values(Layout.output(input, axes, plan), oc)})>,
  roundtrip: {True{} == Layout.maybe_same(Layout.reorder(Layout.coordinate_values(input, ic), axes), Some{Layout.coordinate_values(Layout.output(input, axes, plan), oc)}) : Bool}) ->
  {Result.is_done(&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>,
    Layout.map_roundtrip(input, axes, plan, index, oi, oc, ic, requested, decoded, inverse, decision)) == True{} : Bool}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({False{} == True{} : Bool}, Arith.clash(Equal.trans(Bool, False{},
        Layout.maybe_same(Layout.reorder(Layout.coordinate_values(input, ic), axes), Some{Layout.coordinate_values(Layout.output(input, axes, plan), oc)}), True{}, equal,
        Equal.sym(Bool, True{}, Layout.maybe_same(Layout.reorder(Layout.coordinate_values(input, ic), axes), Some{Layout.coordinate_values(Layout.output(input, axes, plan), oc)}), roundtrip))))
    case Shape.BTrue{equal}:
      mapping_flat_success(input, axes, plan, index, oi, oc, ic, requested, decoded, inverse, equal,
        Layout.flatten_words(Shape.dimensions(input), Layout.coordinate_values(input, ic), 0), {==},
        Shape.bool_case(U32.is_lt(Layout.flatten_words(Shape.dimensions(input), Layout.coordinate_values(input, ic), 0), Shape.count(input))),
        checked_coordinate_flatten_bounded(input, ic))

def mapping_input_success(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  +oi: Layout.FlatIndex<Layout.output(input, axes, plan)>, +oc: Layout.CheckedCoordinate<Layout.output(input, axes, plan)>,
  +requested: {Layout.index_value(Layout.output(input, axes, plan), oi) == index : U32},
  +decoded: {Layout.coordinate_values(Layout.output(input, axes, plan), oc) == Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(Layout.output(input, axes, plan))), index)) : List<&2, U32>},
  +values: List<&2, U32>, +inverse: {Some{values} == Layout.inverse_coordinates(Shape.dimensions(input), axes, Layout.coordinate_values(Layout.output(input, axes, plan), oc), 0) : Maybe<&2, List<&2, U32>>},
  checked: Result<&2, &2, Shape.Error, Layout.WordsValid(Shape.dimensions(input), values)>,
  successful: {Result.is_done(&2, &2, Shape.Error, Layout.WordsValid(Shape.dimensions(input), values), checked) == True{} : Bool}) ->
  {Result.is_done(&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>,
    Layout.map_input(input, axes, plan, index, oi, oc, requested, decoded, values, inverse, checked)) == True{} : Bool}:
  match checked:
    case Fail{error}:
      Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))
    case Done{+valid}:
      mapping_roundtrip_success(input, axes, plan, index, oi, oc, Layout.CWord{values, valid}, requested, decoded, inverse,
        Shape.bool_case(Layout.maybe_same(Layout.reorder(values, axes), Some{Layout.coordinate_values(Layout.output(input, axes, plan), oc)})),
        inverse_words_roundtrip(input, axes, plan, oc, values, inverse))

def mapping_inverse_success(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  +oi: Layout.FlatIndex<Layout.output(input, axes, plan)>, +oc: Layout.CheckedCoordinate<Layout.output(input, axes, plan)>,
  +requested: {Layout.index_value(Layout.output(input, axes, plan), oi) == index : U32},
  +decoded: {Layout.coordinate_values(Layout.output(input, axes, plan), oc) == Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(Layout.output(input, axes, plan))), index)) : List<&2, U32>},
  result: Maybe<&2, List<&2, U32>>,
  +same: {result == Layout.inverse_coordinates(Shape.dimensions(input), axes, Layout.coordinate_values(Layout.output(input, axes, plan), oc), 0) : Maybe<&2, List<&2, U32>>}) ->
  {Result.is_done(&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>,
    Layout.map_inverse(input, axes, plan, index, oi, oc, requested, decoded, result, same)) == True{} : Bool}:
  match result:
    case None{}:
      Empty.absurd({False{} == True{} : Bool},
        Arith.clash(Equal.cong(Maybe<&2, List<&2, U32>>, Bool, metadata_present, None{},
          Some{Axis.order(U32, Layout.coordinate_values(Layout.output(input, axes, plan), oc), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)))},
          Equal.trans(Maybe<&2, List<&2, U32>>, None{}, Layout.inverse_coordinates(Shape.dimensions(input), axes, Layout.coordinate_values(Layout.output(input, axes, plan), oc), 0),
            Some{Axis.order(U32, Layout.coordinate_values(Layout.output(input, axes, plan), oc), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)))}, same,
            AxisAlgebra.scatter_complete(input, axes, Layout.coordinate_values(Layout.output(input, axes, plan), oc), permutation_checked(input, axes, plan),
              output_coordinate_rank(input, axes, plan, oc))))))
    case Some{+values}:
      mapping_input_success(input, axes, plan, index, oi, oc, requested, decoded, values, same, Layout.words_check(Shape.dimensions(input), values),
        words_check_success(Shape.dimensions(input), values, inverse_words_valid(input, axes, plan, oc, values, same)))

def mapping_output_success(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  +oi: Layout.FlatIndex<Layout.output(input, axes, plan)>, +requested: {Layout.index_value(Layout.output(input, axes, plan), oi) == index : U32},
  +values: List<&2, U32>, +decoded: {values == Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(Layout.output(input, axes, plan))), index)) : List<&2, U32>},
  checked: Result<&2, &2, Shape.Error, Layout.WordsValid(Shape.dimensions(Layout.output(input, axes, plan)), values)>,
  successful: {Result.is_done(&2, &2, Shape.Error, Layout.WordsValid(Shape.dimensions(Layout.output(input, axes, plan)), values), checked) == True{} : Bool}) ->
  {Result.is_done(&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>,
    Layout.map_output(input, axes, plan, index, oi, requested, values, decoded, checked)) == True{} : Bool}:
  match checked:
    case Fail{error}:
      Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))
    case Done{+valid}:
      mapping_inverse_success(input, axes, plan, index, oi, Layout.CWord{values, valid}, requested, decoded,
        Layout.inverse_coordinates(Shape.dimensions(input), axes, values, 0), {==})

def mapping_decided_success(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  decision: Shape.BoolCase<U32.is_lt(index, Shape.count(Layout.output(input, axes, plan)))>,
  inside: {True{} == U32.is_lt(index, Shape.count(Layout.output(input, axes, plan))) : Bool}) ->
  {Result.is_done(&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>, Layout.map_decided(input, axes, plan, index, decision)) == True{} : Bool}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({False{} == True{} : Bool}, Arith.clash(Equal.trans(Bool, False{}, U32.is_lt(index, Shape.count(Layout.output(input, axes, plan))), True{}, equal,
        Equal.sym(Bool, True{}, U32.is_lt(index, Shape.count(Layout.output(input, axes, plan))), inside))))
    case Shape.BTrue{+equal}:
      mapping_output_success(input, axes, plan, index, Layout.FIndex{index, equal}, {==},
        Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(Layout.output(input, axes, plan))), index)), {==},
        Layout.words_check(Shape.dimensions(Layout.output(input, axes, plan)), Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(Layout.output(input, axes, plan))), index))),
        words_check_success(Shape.dimensions(Layout.output(input, axes, plan)),
          Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(Layout.output(input, axes, plan))), index)),
          decoded_words_valid(Layout.output(input, axes, plan), index,
            %Equal.sym(Nat, Shape.elements(Layout.output(input, axes, plan)), Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Shape.elements_exact(Layout.output(input, axes, plan))) :
              {True{} == Nat.is_lt(U32.to_nat(index), _) : Bool}
            Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))), True{},
              flat_mathematical_bounded(Layout.output(input, axes, plan), Layout.FIndex{index, equal})))))

def cursor_word_exact(+shape: Shape.Shape, +cursor: Nat, within: {Nat.is_le(cursor, Shape.elements(shape)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat}:
  WordConversion.natural_word_exact(cursor, Shape.count(shape),
    %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.elements(shape), WordConversion.shape_count_exact(shape)) :
      {Nat.is_le(cursor, _) == True{} : Bool}
    within)

def cursor_index_bounded(+shape: Shape.Shape, +cursor: Nat, +inside: {True{} == Nat.is_lt(cursor, Shape.elements(shape)) : Bool}) ->
  {True{} == U32.is_lt(U32.from_nat(cursor), Shape.count(shape)) : Bool}:
  %Arith.u32_lt(U32.from_nat(cursor), Shape.count(shape)) : {True{} == _ : Bool}
  %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.elements(shape), WordConversion.shape_count_exact(shape)) :
    {True{} == Nat.is_lt(U32.to_nat(U32.from_nat(cursor)), _) : Bool}
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(cursor)), cursor, cursor_word_exact(shape, cursor,
    WordConversion.lt_le(cursor, Shape.elements(shape), Equal.sym(Bool, True{}, Nat.is_lt(cursor, Shape.elements(shape)), inside)))) :
      {True{} == Nat.is_lt(_, Shape.elements(shape)) : Bool}
  inside

def permutation_output_rank(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>) ->
  {Shape.rank_dimensions(Shape.dimensions(Layout.output(input, axes, plan))) == Shape.rank_dimensions(Shape.dimensions(input)) : Nat}:
  Equal.trans(Nat, Shape.rank_dimensions(Shape.dimensions(Layout.output(input, axes, plan))),
    Shape.rank_dimensions(Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))), Shape.rank_dimensions(Shape.dimensions(input)),
    Equal.cong(List<&2, U32>, Nat, Shape.rank_dimensions, Shape.dimensions(Layout.output(input, axes, plan)), Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)),
      permutation_dimensions_order(input, axes, plan)), AxisAlgebra.permutation_rank(input, axes, permutation_checked(input, axes, plan)))
