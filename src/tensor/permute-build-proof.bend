import Base
import ../shape/core.bend as Shape
import ./axis-spec.bend as Axis
import ./layout.bend as Layout
import ./observation.bend as Values
import ./permute-build.bend as Build
import ./permutation-mapping.bend as Mapping
import ./word-conversion.bend as Conversion
import ../storage/arith.bend as Arith
import ../storage/model.bend as Model
import ../storage/main.bend as Storage
import ../storage/refinement.bend as Refinement
import ../storage/clone.bend as Clone
import ../storage/route.bend as Route
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./transforms.bend as Transforms
import ./parallel-proof.bend as Generic
import ./parallel.bend as Parallel
import ./transform-totality.bend as Totality
import ./read-contract.bend as Reads
import ./contracts/replayed.bend as Replayed
import ./contracts/binary-build-proof.bend as BinaryBuild
import ./contracts/binary-storage-proof.bend as Owned
import ./contracts/build-facts.bend as Facts
import ../shape/index_proofs.bend as Index
import ../storage/LAWS.bend as StorageLaws
import ../storage/api.bend as Api
import ../storage/array.bend as ArrayCore

# `Build.source` computes the index `Values.permute_source_index` names. The
# proof follows `permute_source` step by step: the coordinate `Axis.select`
# takes out of `Shape.digits` is the one its pick computes, the list
# `Axis.scatter` builds is the picks' coordinates, and `Shape.offset` of
# that list is `Build.offset` with the strides worked out once.

def coordinate_of(pick: Maybe<&2, Build.Pick>, +x: Nat) -> Maybe<&2, U32>:
  match pick:
    case None{}:
      None{}
    case Some{found}:
      Some{U32.from_nat(Build.coordinate(found, x))}

def coordinates(+picks: List<&2, Build.Pick>, +x: Nat) -> List<&2, U32>:
  match picks:
    case Nil{}:
      Nil{}
    case +pick <> rest:
      U32.from_nat(Build.coordinate(pick, x)) <> coordinates(rest, x)

def coordinates_of(picks: Maybe<&2, List<&2, Build.Pick>>, +x: Nat) -> Maybe<&2, List<&2, U32>>:
  match picks:
    case None{}:
      None{}
    case Some{+found}:
      Some{coordinates(found, x)}

# Widening a pick by an outer count reduces the index modulo that count
# first.
def widen_coordinate(+outer: Nat, +pick: Maybe<&2, Build.Pick>, +x: Nat) ->
  {coordinate_of(Build.widen(outer, pick), x) == coordinate_of(pick, Nat.mod(x, outer)) : Maybe<&2, U32>}:
  match pick:
    case None{}:
      {==}
    case Some{+found}:
      Build.Pick{+mods, +size} = found
      {==}

# Coordinate k of `Shape.digits(dims, x)` is what the pick of k computes.
def lookup_digits(+dims: List<&2, Nat>, +x: Nat, +k: Nat) ->
  {Axis.lookup(U32, Shape.words(Shape.digits(dims, x)), k) == coordinate_of(Build.pick_at(dims, k), x) : Maybe<&2, U32>}:
  match dims:
    case Nil{}:
      {==}
    case +head <> +tail:
      match k:
        case 0n:
          {==}
        case 1n+j:
          Equal.trans(Maybe<&2, U32>, Axis.lookup(U32, Shape.words(Shape.digits(tail, Nat.mod(x, Shape.numel(tail)))), j),
            coordinate_of(Build.pick_at(tail, j), Nat.mod(x, Shape.numel(tail))),
            coordinate_of(Build.widen(Shape.numel(tail), Build.pick_at(tail, j)), x),
            lookup_digits(tail, Nat.mod(x, Shape.numel(tail)), j),
            Equal.sym(Maybe<&2, U32>, coordinate_of(Build.widen(Shape.numel(tail), Build.pick_at(tail, j)), x),
              coordinate_of(Build.pick_at(tail, j), Nat.mod(x, Shape.numel(tail))),
              widen_coordinate(Shape.numel(tail), Build.pick_at(tail, j), x)))

def select_digits(+dims: List<&2, Nat>, +x: Nat, +position: Maybe<&2, Nat>) ->
  {Axis.select(U32, Shape.words(Shape.digits(dims, x)), position) == coordinate_of(Build.located(dims, position), x) : Maybe<&2, U32>}:
  match position:
    case None{}:
      {==}
    case Some{+k}:
      lookup_digits(dims, x, k)

def prepend_coordinates(+head: Maybe<&2, Build.Pick>, +tail: Maybe<&2, List<&2, Build.Pick>>, +x: Nat) ->
  {Axis.prepend(U32, coordinate_of(head, x), coordinates_of(tail, x)) == coordinates_of(Axis.prepend(Build.Pick, head, tail), x) :
    Maybe<&2, List<&2, U32>>}:
  match head:
    case None{}:
      {==}
    case Some{+pick}:
      match tail:
        case None{}:
          {==}
        case Some{+rest}:
          {==}

# The list `Axis.scatter` builds from the output coordinates of x is the
# picks' coordinates.
def scatter_digits(+rank: Nat, +axes: List<&2, Nat>, +out: List<&2, Nat>, +x: Nat, +cursor: Nat) ->
  {Axis.scatter(U32, rank, axes, Shape.words(Shape.digits(out, x)), cursor) == coordinates_of(Build.picks(rank, axes, out, cursor), x) :
    Maybe<&2, List<&2, U32>>}:
  match rank:
    case 0n:
      {==}
    case 1n+p:
      +position = Axis.position(axes, cursor)
      %Equal.sym(Maybe<&2, U32>, Axis.select(U32, Shape.words(Shape.digits(out, x)), position), coordinate_of(Build.located(out, position), x),
        select_digits(out, x, position)) :
        {Axis.prepend(U32, _, Axis.scatter(U32, p, axes, Shape.words(Shape.digits(out, x)), 1n+cursor)) ==
          coordinates_of(Axis.prepend(Build.Pick, Build.located(out, position), Build.picks(p, axes, out, 1n+cursor)), x) : Maybe<&2, List<&2, U32>>}
      %Equal.sym(Maybe<&2, List<&2, U32>>, Axis.scatter(U32, p, axes, Shape.words(Shape.digits(out, x)), 1n+cursor),
        coordinates_of(Build.picks(p, axes, out, 1n+cursor), x), scatter_digits(p, axes, out, x, 1n+cursor)) :
        {Axis.prepend(U32, coordinate_of(Build.located(out, position), x), _) ==
          coordinates_of(Axis.prepend(Build.Pick, Build.located(out, position), Build.picks(p, axes, out, 1n+cursor)), x) : Maybe<&2, List<&2, U32>>}
      prepend_coordinates(Build.located(out, position), Build.picks(p, axes, out, 1n+cursor), x)

# `Shape.offset` of the coordinates is `Build.offset` over the strides.
def offset_coordinates(+dims: List<&2, Nat>, +picks: List<&2, Build.Pick>, +x: Nat) ->
  {Shape.offset(dims, Shape.naturals(coordinates(picks, x))) == Build.offset(Build.strides(dims), picks, x) : Nat}:
  match dims:
    case Nil{}:
      {==}
    case +head <> +tail:
      match picks:
        case Nil{}:
          {==}
        case +pick <> +rest:
          Equal.cong(Nat, Nat, rest_offset => Nat.add(Nat.mul(U32.to_nat(U32.from_nat(Build.coordinate(pick, x))), Shape.numel(tail)), rest_offset),
            Shape.offset(tail, Shape.naturals(coordinates(rest, x))), Build.offset(Build.strides(tail), rest, x),
            offset_coordinates(tail, rest, x))

def found_coordinates(+dims: List<&2, U32>, picks: Maybe<&2, List<&2, Build.Pick>>, +index: Nat) ->
  {U32.to_nat(Maybe.default(&2, U32, Layout.reference_source_coordinates(dims, coordinates_of(picks, U32.to_nat(U32.from_nat(index)))), 0)) ==
    Build.found(picks, Build.strides(Shape.naturals(dims)), index) : Nat}:
  match picks:
    case None{}:
      {==}
    case Some{+found}:
      Equal.cong(Nat, Nat, offset => U32.to_nat(U32.from_nat(offset)),
        Shape.offset(Shape.naturals(dims), Shape.naturals(coordinates(found, U32.to_nat(U32.from_nat(index))))),
        Build.offset(Build.strides(Shape.naturals(dims)), found, U32.to_nat(U32.from_nat(index))),
        offset_coordinates(Shape.naturals(dims), found, U32.to_nat(U32.from_nat(index))))

# The build's index map is `permute_source_index`.
def source_index(+input: Shape.Shape, +axes: List<&2, U32>, +index: Nat) ->
  {Build.source(Build.recipe(Build.Request{input, axes}), index) == Values.permute_source_index(input, axes, index) : Nat}:
  +dims = Shape.dimensions(input)
  +out = Shape.naturals(Axis.order(U32, dims, Shape.naturals(axes)))
  +x = U32.to_nat(U32.from_nat(index))
  +picks = Build.picks(Shape.rank_dimensions(dims), Shape.naturals(axes), out, 0n)
  Equal.sym(Nat, Values.permute_source_index(input, axes, index), Build.found(picks, Build.strides(Shape.naturals(dims)), index),
    %Equal.sym(Maybe<&2, List<&2, U32>>, Axis.scatter(U32, Shape.rank_dimensions(dims), Shape.naturals(axes), Shape.words(Shape.digits(out, x)), 0n),
      coordinates_of(picks, x), scatter_digits(Shape.rank_dimensions(dims), Shape.naturals(axes), out, x, 0n)) :
      {U32.to_nat(Maybe.default(&2, U32, Layout.reference_source_coordinates(dims, _), 0)) ==
        Build.found(picks, Build.strides(Shape.naturals(dims)), index) : Nat}
    found_coordinates(dims, picks, index))

# The word index a cell reads is `permute_source_index` as a word.
def source_word(+input: Shape.Shape, +axes: List<&2, U32>, +index: Nat) ->
  {Build.at(Build.recipe(Build.Request{input, axes}), index) == U32.from_nat(Values.permute_source_index(input, axes, index)) : U32}:
  Equal.cong(Nat, U32, number => U32.from_nat(number), Build.source(Build.recipe(Build.Request{input, axes}), index),
    Values.permute_source_index(input, axes, index), source_index(input, axes, index))

# The mapping a permute's plan gives an output index names the input index
# `permute_source` names, and it lies inside the input.
def reference(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  +mapping: Layout.Mapping<input, axes, plan, index>) ->
  {Some{Layout.source_index(input, axes, plan, index, mapping)} == Values.permute_source(input, axes, index) : Maybe<&2, U32>}:
  Equal.trans(Maybe<&2, U32>, Some{Layout.source_index(input, axes, plan, index, mapping)}, Layout.reference_source(input, axes, plan, index),
    Values.permute_source(input, axes, index),
    Mapping.mapping_reference(input, axes, plan, index, mapping),
    Equal.cong(List<&2, U32>, Maybe<&2, U32>,
      dimensions => Layout.reference_source_coordinates(Shape.dimensions(input),
        Axis.scatter(U32, Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes),
          Shape.words(Shape.digits(Shape.naturals(dimensions), U32.to_nat(index))), 0n)),
      Shape.dimensions(Layout.output(input, axes, plan)), Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)),
      Mapping.permutation_dimensions_order(input, axes, plan)))

def mapped_index(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  +mapping: Layout.Mapping<input, axes, plan, index>) ->
  {Layout.source_index(input, axes, plan, index, mapping) ==
    U32.from_nat(U32.to_nat(Maybe.default(&2, U32, Values.permute_source(input, axes, index), 0))) : U32}:
  +found = Maybe.default(&2, U32, Values.permute_source(input, axes, index), 0)
  Equal.trans(U32, Layout.source_index(input, axes, plan, index, mapping), found, U32.from_nat(U32.to_nat(found)),
    Equal.cong(Maybe<&2, U32>, U32, result => Maybe.default(&2, U32, result, 0),
      Some{Layout.source_index(input, axes, plan, index, mapping)}, Values.permute_source(input, axes, index),
      reference(input, axes, plan, index, mapping)),
    Equal.sym(U32, U32.from_nat(U32.to_nat(found)), found, Conversion.word_natural_exact(found)))

def mapped_bound(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  mapping: Layout.Mapping<input, axes, plan, index>) ->
  {Nat.is_lt(U32.to_nat(Layout.source_index(input, axes, plan, index, mapping)), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool}:
  Layout.Mapped{oi, oc, ic, found, requested, decoded, inverse, flattened, roundtrip} = mapping
  Mapping.flat_mathematical_bounded(input, found)

def mapped_inside(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  +mapping: Layout.Mapping<input, axes, plan, index>) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(U32.to_nat(Maybe.default(&2, U32, Values.permute_source(input, axes, index), 0)))),
    Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool}:
  %mapped_index(input, axes, plan, index, mapping) : {Nat.is_lt(U32.to_nat(_), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool}
  mapped_bound(input, axes, plan, index, mapping)

def index_bounded(+shape: Shape.Shape, +index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {True{} == U32.is_lt(index, Shape.count(shape)) : Bool}:
  %Arith.u32_lt(index, Shape.count(shape)) : {True{} == _ : Bool}
  %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.elements(shape), Conversion.shape_count_exact(shape)) : {True{} == Nat.is_lt(U32.to_nat(index), _) : Bool}
  %Equal.sym(Nat, Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)), Shape.elements_exact(shape)) : {True{} == Nat.is_lt(U32.to_nat(index), _) : Bool}
  Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))), True{}, inside)

def mapping_done(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool}) ->
  {Result.is_done(&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>, Layout.map_index(input, axes, plan, index)) == True{} : Bool}:
  Mapping.mapping_decided_success(input, axes, plan, index,
    Shape.bool_case(U32.is_lt(index, Shape.count(Layout.output(input, axes, plan)))),
    index_bounded(Layout.output(input, axes, plan), index, inside))

# The build's cells. The state is the input array next to an empty array,
# `(a, Array.new(F32, 0n, 0.0))`. Every cell hands it back, and a split hands it to both halves,
# so the generic law of `Parallel.build` gives what every slot holds.

type Fact<-P: Type> is Data:
  Fact{-proof: P}

def take_first(-e: Array<F32>, pair: Array<F32> & F32) ->
  {Pair.snd(F32, Array<F32> & Array<F32>, Build.take(e, pair)) == (Refinement.first(F32, pair), e) : Array<F32> & Array<F32>}:
  (x, w) = pair
  {==}

def take_word(-e: Array<F32>, pair: Array<F32> & F32) ->
  {Pair.fst(F32, Array<F32> & Array<F32>, Build.take(e, pair)) == Refinement.second(F32, pair) : F32}:
  (x, w) = pair
  {==}

def cell_keeps(+d: Nat, -a: Array<F32>, +request: Build.Request, +position: Nat, inside: Bool, -valid: Model.Balanced(F32, d, a)) ->
  Fact<{Pair.snd(F32, Array<F32> & Array<F32>, Build.cell(Build.recipe(request), position, (a, Array.new(F32, 0n, 0.0)), inside)) == (a, Array.new(F32, 0n, 0.0)) : Array<F32> & Array<F32>}>:
  match inside:
    case False{}:
      Fact{{==}}
    case True{}:
      +index = Build.at(Build.recipe(request), position)
      Fact{Equal.trans(Array<F32> & Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, Build.take(Array.new(F32, 0n, 0.0), Array.get(F32, a, index))),
        (Refinement.first(F32, Array.get(F32, a, index)), Array.new(F32, 0n, 0.0)), (a, Array.new(F32, 0n, 0.0)),
        take_first(Array.new(F32, 0n, 0.0), Array.get(F32, a, index)),
        Equal.cong(Array<F32>, Array<F32> & Array<F32>, x => (x, Array.new(F32, 0n, 0.0)), Refinement.first(F32, Array.get(F32, a, index)), a,
          Refinement.law_get_retains(F32, d, a, index, valid)))}

# An inside cell holds the input word at `permute_source_index`.
def cell_word(+input: Shape.Shape, +axes: List<&2, U32>, -a: Array<F32>, +position: Nat) ->
  Fact<{Pair.fst(F32, Array<F32> & Array<F32>, Build.cell(Build.recipe(Build.Request{input, axes}), position, (a, Array.new(F32, 0n, 0.0)), True{})) ==
    Refinement.second(F32, Array.get(F32, a, U32.from_nat(Values.permute_source_index(input, axes, position)))) : F32}>:
  +index = Build.at(Build.recipe(Build.Request{input, axes}), position)
  Fact{Equal.trans(F32, Pair.fst(F32, Array<F32> & Array<F32>, Build.take(Array.new(F32, 0n, 0.0), Array.get(F32, a, index))),
    Refinement.second(F32, Array.get(F32, a, index)),
    Refinement.second(F32, Array.get(F32, a, U32.from_nat(Values.permute_source_index(input, axes, position)))),
    take_word(Array.new(F32, 0n, 0.0), Array.get(F32, a, index)),
    Equal.cong(U32, F32, word => Refinement.second(F32, Array.get(F32, a, word)), index,
      U32.from_nat(Values.permute_source_index(input, axes, position)), source_word(input, axes, position)))}

def clone_first(+d: Nat, -a: Array<F32>, pair: Array<F32> & Array<F32>, -exact: {pair == Array.clone(F32, a) : Array<F32> & Array<F32>}, -valid: Model.Balanced(F32, d, a)) ->
  Fact<{Pair.fst(Array<F32>, Array<F32>, pair) == a : Array<F32>}>:
  Fact{Equal.trans(Array<F32>, Pair.fst(Array<F32>, Array<F32>, pair), Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, a)), a,
    Equal.cong(Array<F32> & Array<F32>, Array<F32>, p => Pair.fst(Array<F32>, Array<F32>, p), pair, Array.clone(F32, a), exact),
    Clone.first_equal(F32, d, a, Clone.certify(F32, d, a, valid)))}

def clone_second(+d: Nat, -a: Array<F32>, pair: Array<F32> & Array<F32>, -exact: {pair == Array.clone(F32, a) : Array<F32> & Array<F32>}, -valid: Model.Balanced(F32, d, a)) ->
  Fact<{Pair.snd(Array<F32>, Array<F32>, pair) == a : Array<F32>}>:
  Fact{Equal.trans(Array<F32>, Pair.snd(Array<F32>, Array<F32>, pair), Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, a)), a,
    Equal.cong(Array<F32> & Array<F32>, Array<F32>, p => Pair.snd(Array<F32>, Array<F32>, p), pair, Array.clone(F32, a), exact),
    Clone.second_equal(F32, d, a, Clone.certify(F32, d, a, valid)))}

def halves_equal(+d: Nat, -a: Array<F32>, lp: Array<F32> & Array<F32>, -exact: {lp == Array.clone(F32, a) : Array<F32> & Array<F32>}, -valid: Model.Balanced(F32, d, a)) ->
  Fact<({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Build.halves(lp, Array.clone(F32, Array.new(F32, 0n, 0.0)))) == (a, Array.new(F32, 0n, 0.0)) : Array<F32> & Array<F32>} &
    {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Build.halves(lp, Array.clone(F32, Array.new(F32, 0n, 0.0)))) == (a, Array.new(F32, 0n, 0.0)) : Array<F32> & Array<F32>})>:
  (l1, l2) = lp
  Fact{(Equal.cong(Array<F32>, Array<F32> & Array<F32>, x => (x, Array.new(F32, 0n, 0.0)), l1, a,
      {\{Fact: q => q} : Fact<{Pair.fst(Array<F32>, Array<F32>, (l1, l2)) == a : Array<F32>}> -> {Pair.fst(Array<F32>, Array<F32>, (l1, l2)) == a : Array<F32>}}(
        clone_first(d, a, (l1, l2), exact, valid))),
    Equal.cong(Array<F32>, Array<F32> & Array<F32>, x => (x, Array.new(F32, 0n, 0.0)), l2, a,
      {\{Fact: q => q} : Fact<{Pair.snd(Array<F32>, Array<F32>, (l1, l2)) == a : Array<F32>}> -> {Pair.snd(Array<F32>, Array<F32>, (l1, l2)) == a : Array<F32>}}(
        clone_second(d, a, (l1, l2), exact, valid))))}

def SplitFacts(-a: Array<F32>) -> Type:
  {Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Build.split((a, Array.new(F32, 0n, 0.0)))) == (a, Array.new(F32, 0n, 0.0)) : Array<F32> & Array<F32>} & {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Build.split((a, Array.new(F32, 0n, 0.0)))) == (a, Array.new(F32, 0n, 0.0)) : Array<F32> & Array<F32>}

def split_equal(+d: Nat, -a: Array<F32>, -valid: Model.Balanced(F32, d, a)) -> Fact<SplitFacts(a)>:
  Fact{{\{Fact: q => q} : Fact<SplitFacts(a)> -> SplitFacts(a)}(halves_equal(d, a, Array.clone(F32, a), {==}, valid))}

# Every slot of the build holds what its cell function computes.
def build_cell(+depth: Nat, +input: Shape.Shape, +axes: List<&2, U32>, +count: Nat, +d: Nat, -a: Array<F32>, +path: Model.Path(depth),
  -valid: Model.Balanced(F32, d, a)) ->
  Fact<{Model.observe(F32, depth, Build.build(depth, Build.Request{input, axes}, count, a), path) ==
    Some{Pair.fst(F32, Array<F32> & Array<F32>, Build.cell(Build.recipe(Build.Request{input, axes}), Nat.add(Model.offset(depth, path), 0n), (a, Array.new(F32, 0n, 0.0)),
      Nat.is_lt(Nat.add(Model.offset(depth, path), 0n), count)))} : Maybe<&2, F32>}>:
  +request : Build.Request = Build.Request{input, axes}
  Fact{{\{Generic.Fact: q => q} :
    Generic.Fact<Generic.Cells(~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.cell, Build.recipe(request), (a, Array.new(F32, 0n, 0.0)), depth, count, 0n,
      Build.build(depth, request, count, a))> ->
    Generic.Cells(~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.cell, Build.recipe(request), (a, Array.new(F32, 0n, 0.0)), depth, count, 0n,
      Build.build(depth, request, count, a))}(
    Generic.build_cells(~Build.Request, ~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.recipe, ~Build.split, ~Build.cell,
      request, (a, Array.new(F32, 0n, 0.0)), depth, Build.splits(depth), count, 0n, (a, Array.new(F32, 0n, 0.0)), {==},
      at => inside => {\{Fact: q => q} : Fact<{Pair.snd(F32, Array<F32> & Array<F32>, Build.cell(Build.recipe(request), at, (a, Array.new(F32, 0n, 0.0)), inside)) == (a, Array.new(F32, 0n, 0.0)) : Array<F32> & Array<F32>}> ->
        {Pair.snd(F32, Array<F32> & Array<F32>, Build.cell(Build.recipe(request), at, (a, Array.new(F32, 0n, 0.0)), inside)) == (a, Array.new(F32, 0n, 0.0)) : Array<F32> & Array<F32>}}(cell_keeps(d, a, request, at, inside, valid)),
      Pair.fst({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Build.split((a, Array.new(F32, 0n, 0.0)))) == (a, Array.new(F32, 0n, 0.0)) : Array<F32> & Array<F32>}, {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Build.split((a, Array.new(F32, 0n, 0.0)))) == (a, Array.new(F32, 0n, 0.0)) : Array<F32> & Array<F32>},
        {\{Fact: q => q} : Fact<SplitFacts(a)> -> SplitFacts(a)}(split_equal(d, a, valid))),
      Pair.snd({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Build.split((a, Array.new(F32, 0n, 0.0)))) == (a, Array.new(F32, 0n, 0.0)) : Array<F32> & Array<F32>}, {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Build.split((a, Array.new(F32, 0n, 0.0)))) == (a, Array.new(F32, 0n, 0.0)) : Array<F32> & Array<F32>},
        {\{Fact: q => q} : Fact<SplitFacts(a)> -> SplitFacts(a)}(split_equal(d, a, valid)))))(path)}

def plan_elements(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>) ->
  {Shape.elements(Layout.output(input, axes, plan)) == Shape.elements(input) : Nat}:
  Layout.Permutation{+output, valid, ordered, +count} = plan
  Equal.sym(Nat, Shape.elements(input), Shape.elements(output), Lifecycle.natural_equal(Shape.elements(input), Shape.elements(output), count))

def plan_numel(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>) ->
  {Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) == Shape.logical_numel(Shape.dimensions(input)) : Nat}:
  +output = Layout.output(input, axes, plan)
  Equal.trans(Nat, Shape.logical_numel(Shape.dimensions(output)), Shape.elements(output), Shape.logical_numel(Shape.dimensions(input)),
    Equal.sym(Nat, Shape.elements(output), Shape.logical_numel(Shape.dimensions(output)), Shape.elements_exact(output)),
    Equal.trans(Nat, Shape.elements(output), Shape.elements(input), Shape.logical_numel(Shape.dimensions(input)),
      plan_elements(input, axes, plan), Shape.elements_exact(input)))

# The parallel copy against the copy loop's meaning: the source comes back
# unchanged and every output index below the element count holds the word
# `expected_output` names.

def word_below(+shape: Shape.Shape, +other: Nat, +earlier: {Nat.is_lt(other, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(other)) == other : Nat}:
  Conversion.natural_word_exact(other, Shape.count(shape),
    %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.logical_numel(Shape.dimensions(shape)),
      Equal.trans(Nat, U32.to_nat(Shape.count(shape)), Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)),
        Conversion.shape_count_exact(shape), Shape.elements_exact(shape))) : {Nat.is_le(other, _) == True{} : Bool}
    Conversion.lt_le(other, Shape.logical_numel(Shape.dimensions(shape)), earlier))

def length_below(+shape: Shape.Shape, +n: Nat, +exact: {n == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +other: Nat,
  +earlier: {Nat.is_lt(other, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(other)), n) == True{} : Bool}:
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(other)), other, word_below(shape, other, earlier)) : {Nat.is_lt(_, n) == True{} : Bool}
  %Equal.sym(Nat, n, Shape.logical_numel(Shape.dimensions(shape)), exact) : {Nat.is_lt(other, _) == True{} : Bool}
  earlier

def owned_observe(+d: Nat, +n: Nat, -array: Array<F32>, +index: U32, +valid: {Storage.bounds(d, n) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index), n) == True{} : Bool}) ->
  {Lifecycle.checked_observe_array(d, n, array, index) == Model.observe(F32, d, array, Refinement.route(d, index)) : Maybe<&2, F32>}:
  %Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), n), True{}, inside) :
    {Lifecycle.observe_allowed(d, n, array, index, _) == Model.observe(F32, d, array, Refinement.route(d, index)) : Maybe<&2, F32>}
  %Equal.sym(Bool, Storage.index_matches(d, n, index), True{}, Route.index_matched(d, n, index, valid, inside)) :
    {Lifecycle.observe_index(d, n, array, index, _) == Model.observe(F32, d, array, Refinement.route(d, index)) : Maybe<&2, F32>}
  {==}

def balanced_along(-d: Nat, -x: Array<F32>, -y: Array<F32>, -same: {x == y : Array<F32>}, -valid: Model.Balanced(F32, d, y)) ->
  Fact<Model.Balanced(F32, d, x)>:
  Fact{
    %Equal.sym(Array<F32>, x, y, same) : Model.Balanced(F32, d, _)
    valid}

# What the copy loop expects at output index `other`: the source word at
# `permute_source_index`.
def expected_word(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +other: Nat,
  +ds: Nat, +ns: Nat, -a: Array<F32>, -b: Model.Balanced(F32, ds, a), +ok: {Storage.bounds(ds, ns) == True{} : Bool},
  +fits: {ns == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  -source_fits: {Lifecycle.logical_length(Storage.Owned{ds, ns, a, b, ok}) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  result: Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, U32.from_nat(other)>>,
  done: {Result.is_done(&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, U32.from_nat(other)>, result) == True{} : Bool}) ->
  Fact<{Transforms.expected_mapped(input, axes, plan, U32.from_nat(other), Core.TTensor{Storage.Owned{ds, ns, a, b, ok}, source_fits}, result) ==
    Some{Refinement.second(F32, Array.get(F32, a, U32.from_nat(Values.permute_source_index(input, axes, other))))} : Maybe<&2, F32>}>:
  match result:
    case Fail{error}:
      Fact{Empty.absurd({None{} == Some{Refinement.second(F32, Array.get(F32, a, U32.from_nat(Values.permute_source_index(input, axes, other))))} : Maybe<&2, F32>}, Arith.clash(done))}
    case Done{+mapping}:
      +found = Layout.source_index(input, axes, plan, U32.from_nat(other), mapping)
      +inside : {Nat.is_lt(U32.to_nat(found), ns) == True{} : Bool} = %Equal.sym(Nat, ns, Shape.logical_numel(Shape.dimensions(input)), fits) : {Nat.is_lt(U32.to_nat(found), _) == True{} : Bool}
        mapped_bound(input, axes, plan, U32.from_nat(other), mapping)
      Fact{Equal.trans(Maybe<&2, F32>, Lifecycle.checked_observe_array(ds, ns, a, found), Model.observe(F32, ds, a, Refinement.route(ds, found)),
        Some{Refinement.second(F32, Array.get(F32, a, U32.from_nat(Values.permute_source_index(input, axes, other))))},
        owned_observe(ds, ns, a, found, ok, inside),
        Equal.trans(Maybe<&2, F32>, Model.observe(F32, ds, a, Refinement.route(ds, found)), Some{Refinement.second(F32, Array.get(F32, a, found))},
          Some{Refinement.second(F32, Array.get(F32, a, U32.from_nat(Values.permute_source_index(input, axes, other))))},
          Equal.sym(Maybe<&2, F32>, Some{Refinement.second(F32, Array.get(F32, a, found))}, Model.observe(F32, ds, a, Refinement.route(ds, found)),
            Refinement.law_get_refines(F32, ds, a, found, b)),
          Equal.cong(U32, Maybe<&2, F32>, index => Some{Refinement.second(F32, Array.get(F32, a, index))}, found, U32.from_nat(Values.permute_source_index(input, axes, other)),
            mapped_index(input, axes, plan, U32.from_nat(other), mapping))))}

# What the parallel build holds at output index `other`.
def output_word(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +d: Nat, +n: Nat,
  +valid: {Storage.bounds(d, n) == True{} : Bool}, +exact: {n == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat},
  +ds: Nat, -c: Array<F32>, -bc: Model.Balanced(F32, ds, c), +other: Nat, +earlier: {Nat.is_lt(other, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool}) ->
  Fact<{Lifecycle.checked_observe_array(d, n, Build.build(d, Build.Request{input, axes}, n, c), U32.from_nat(other)) ==
    Some{Refinement.second(F32, Array.get(F32, c, U32.from_nat(Values.permute_source_index(input, axes, other))))} : Maybe<&2, F32>}>:
  +index = U32.from_nat(other)
  +inside = length_below(Layout.output(input, axes, plan), n, exact, other, earlier)
  +at = Equal.trans(Nat, Nat.add(Model.offset(d, Refinement.route(d, index)), 0n), U32.to_nat(index), other,
    Facts.output_position(d, n, index, valid, inside), word_below(Layout.output(input, axes, plan), other, earlier))
  +below : {Nat.is_lt(other, n) == True{} : Bool} = %Equal.sym(Nat, n, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), exact) : {Nat.is_lt(other, _) == True{} : Bool}
    earlier
  -built = Build.build(d, Build.Request{input, axes}, n, c)
  -recipe = Build.recipe(Build.Request{input, axes})
  Fact{Equal.trans(Maybe<&2, F32>, Lifecycle.checked_observe_array(d, n, built, index), Model.observe(F32, d, built, Refinement.route(d, index)),
    Some{Refinement.second(F32, Array.get(F32, c, U32.from_nat(Values.permute_source_index(input, axes, other))))},
    owned_observe(d, n, built, index, valid, inside),
    Equal.trans(Maybe<&2, F32>, Model.observe(F32, d, built, Refinement.route(d, index)),
      Some{Pair.fst(F32, Array<F32> & Array<F32>, Build.cell(recipe, Nat.add(Model.offset(d, Refinement.route(d, index)), 0n), (c, Array.new(F32, 0n, 0.0)),
        Nat.is_lt(Nat.add(Model.offset(d, Refinement.route(d, index)), 0n), n)))},
      Some{Refinement.second(F32, Array.get(F32, c, U32.from_nat(Values.permute_source_index(input, axes, other))))},
      {\{Fact: q => q} : Fact<{Model.observe(F32, d, built, Refinement.route(d, index)) ==
        Some{Pair.fst(F32, Array<F32> & Array<F32>, Build.cell(recipe, Nat.add(Model.offset(d, Refinement.route(d, index)), 0n), (c, Array.new(F32, 0n, 0.0)),
          Nat.is_lt(Nat.add(Model.offset(d, Refinement.route(d, index)), 0n), n)))} : Maybe<&2, F32>}> ->
        {Model.observe(F32, d, built, Refinement.route(d, index)) ==
          Some{Pair.fst(F32, Array<F32> & Array<F32>, Build.cell(recipe, Nat.add(Model.offset(d, Refinement.route(d, index)), 0n), (c, Array.new(F32, 0n, 0.0)),
            Nat.is_lt(Nat.add(Model.offset(d, Refinement.route(d, index)), 0n), n)))} : Maybe<&2, F32>}}(
        build_cell(d, input, axes, n, ds, c, Refinement.route(d, index), bc)),
      %Equal.sym(Nat, Nat.add(Model.offset(d, Refinement.route(d, index)), 0n), other, at) : {Some{Pair.fst(F32, Array<F32> & Array<F32>, Build.cell(recipe, _, (c, Array.new(F32, 0n, 0.0)), Nat.is_lt(_, n)))} == Some{Refinement.second(F32, Array.get(F32, c, U32.from_nat(Values.permute_source_index(input, axes, other))))} : Maybe<&2, F32>}
      %Equal.sym(Bool, Nat.is_lt(other, n), True{}, below) :
        {Some{Pair.fst(F32, Array<F32> & Array<F32>, Build.cell(recipe, other, (c, Array.new(F32, 0n, 0.0)), _))} == Some{Refinement.second(F32, Array.get(F32, c, U32.from_nat(Values.permute_source_index(input, axes, other))))} : Maybe<&2, F32>}
      Equal.cong(F32, Maybe<&2, F32>, word => Some{word}, Pair.fst(F32, Array<F32> & Array<F32>, Build.cell(recipe, other, (c, Array.new(F32, 0n, 0.0)), True{})),
        Refinement.second(F32, Array.get(F32, c, U32.from_nat(Values.permute_source_index(input, axes, other)))),
        {\{Fact: q => q} : Fact<{Pair.fst(F32, Array<F32> & Array<F32>, Build.cell(recipe, other, (c, Array.new(F32, 0n, 0.0)), True{})) == Refinement.second(F32, Array.get(F32, c, U32.from_nat(Values.permute_source_index(input, axes, other)))) : F32}> ->
          {Pair.fst(F32, Array<F32> & Array<F32>, Build.cell(recipe, other, (c, Array.new(F32, 0n, 0.0)), True{})) == Refinement.second(F32, Array.get(F32, c, U32.from_nat(Values.permute_source_index(input, axes, other)))) : F32}}(
          cell_word(input, axes, c, other)))))}

# The copy from an owned source, whatever the clone hands out.
def cloned_meaning(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +d: Nat, +n: Nat,
  -valid: {Storage.bounds(d, n) == True{} : Bool}, -exact: {n == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat},
  +ds: Nat, +ns: Nat, -a: Array<F32>, pair: Array<F32> & Array<F32>, -same: {pair == Array.clone(F32, a) : Array<F32> & Array<F32>},
  -b: Model.Balanced(F32, ds, a), -ok: {Storage.bounds(ds, ns) == True{} : Bool},
  -fits: {Lifecycle.logical_length(Storage.Owned{ds, ns, a, b, ok}) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Transforms.LoopMeaning(input, axes, plan, Core.TTensor{Storage.Owned{ds, ns, a, b, ok}, fits}, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))),
    Transforms.built_cloned(input, axes, plan, d, n, valid, exact, ds, ns, a, pair, same, b, ok, fits)):
  (kept, copy) = pair
  -kept_a = {\{Fact: q => q} : Fact<{Pair.fst(Array<F32>, Array<F32>, (kept, copy)) == a : Array<F32>}> -> {kept == a : Array<F32>}}(
    clone_first(ds, a, (kept, copy), same, b))
  -copy_a = {\{Fact: q => q} : Fact<{Pair.snd(Array<F32>, Array<F32>, (kept, copy)) == a : Array<F32>}> -> {copy == a : Array<F32>}}(
    clone_second(ds, a, (kept, copy), same, b))
  -bc = {\{Fact: q => q} : Fact<Model.Balanced(F32, ds, copy)> -> Model.Balanced(F32, ds, copy)}(balanced_along(ds, copy, a, copy_a, b))
  Transforms.Prefix{
    other => Equal.cong(Array<F32>, Maybe<&2, F32>, x => Lifecycle.observe_allowed(ds, ns, x, other, Nat.is_lt(U32.to_nat(other), ns)), kept, a, kept_a),
    other => earlier => word_below(Layout.output(input, axes, plan), other, earlier),
    other => earlier => Route.index_matched(d, n, U32.from_nat(other), valid, length_below(Layout.output(input, axes, plan), n, exact, other, earlier)),
    other => earlier => Equal.trans(Maybe<&2, F32>,
      Lifecycle.checked_observe_array(d, n, Build.build(d, Build.Request{input, axes}, n, copy), U32.from_nat(other)),
      Some{Refinement.second(F32, Array.get(F32, a, U32.from_nat(Values.permute_source_index(input, axes, other))))},
      Transforms.expected_output(input, axes, plan, Core.TTensor{Storage.Owned{ds, ns, a, b, ok}, fits}, U32.from_nat(other)),
      Equal.trans(Maybe<&2, F32>,
        Lifecycle.checked_observe_array(d, n, Build.build(d, Build.Request{input, axes}, n, copy), U32.from_nat(other)),
        Some{Refinement.second(F32, Array.get(F32, copy, U32.from_nat(Values.permute_source_index(input, axes, other))))}, Some{Refinement.second(F32, Array.get(F32, a, U32.from_nat(Values.permute_source_index(input, axes, other))))},
        {\{Fact: q => q} : Fact<{Lifecycle.checked_observe_array(d, n, Build.build(d, Build.Request{input, axes}, n, copy), U32.from_nat(other)) ==
          Some{Refinement.second(F32, Array.get(F32, copy, U32.from_nat(Values.permute_source_index(input, axes, other))))} : Maybe<&2, F32>}> ->
          {Lifecycle.checked_observe_array(d, n, Build.build(d, Build.Request{input, axes}, n, copy), U32.from_nat(other)) ==
            Some{Refinement.second(F32, Array.get(F32, copy, U32.from_nat(Values.permute_source_index(input, axes, other))))} : Maybe<&2, F32>}}(
          output_word(input, axes, plan, d, n, valid, exact, ds, copy, bc, other, earlier)),
        Equal.cong(Array<F32>, Maybe<&2, F32>, x => Some{Refinement.second(F32, Array.get(F32, x, U32.from_nat(Values.permute_source_index(input, axes, other))))}, copy, a, copy_a)),
      Equal.sym(Maybe<&2, F32>, Transforms.expected_output(input, axes, plan, Core.TTensor{Storage.Owned{ds, ns, a, b, ok}, fits}, U32.from_nat(other)),
        Some{Refinement.second(F32, Array.get(F32, a, U32.from_nat(Values.permute_source_index(input, axes, other))))},
        {\{Fact: q => q} : Fact<{Transforms.expected_mapped(input, axes, plan, U32.from_nat(other), Core.TTensor{Storage.Owned{ds, ns, a, b, ok}, fits},
          Layout.map_index(input, axes, plan, U32.from_nat(other))) == Some{Refinement.second(F32, Array.get(F32, a, U32.from_nat(Values.permute_source_index(input, axes, other))))} : Maybe<&2, F32>}> ->
          {Transforms.expected_mapped(input, axes, plan, U32.from_nat(other), Core.TTensor{Storage.Owned{ds, ns, a, b, ok}, fits},
            Layout.map_index(input, axes, plan, U32.from_nat(other))) == Some{Refinement.second(F32, Array.get(F32, a, U32.from_nat(Values.permute_source_index(input, axes, other))))} : Maybe<&2, F32>}}(
          expected_word(input, axes, plan, other, ds, ns, a, b, ok, fits, fits, Layout.map_index(input, axes, plan, U32.from_nat(other)),
            mapping_done(input, axes, plan, U32.from_nat(other),
              %Equal.sym(Nat, U32.to_nat(U32.from_nat(other)), other, word_below(Layout.output(input, axes, plan), other, earlier)) : {Nat.is_lt(_, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool}
              earlier)))))}

def owned_meaning(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +d: Nat, +n: Nat,
  -valid: {Storage.bounds(d, n) == True{} : Bool}, -exact: {n == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}, source: Core.Tensor<input>) ->
  Transforms.LoopMeaning(input, axes, plan, source, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Transforms.built_owned(input, axes, plan, d, n, valid, exact, source)):
  Core.TTensor{storage, fits} = source
  match storage:
    case Storage.Empty{}:
      Unit{}
    case Storage.Owned{+ds, +ns, a, b, ok}:
      cloned_meaning(input, axes, plan, d, n, valid, exact, ds, ns, a, Array.clone(F32, a), {==}, b, ok, fits)

def empty_below(-shape: Shape.Shape, +other: Nat, exact: {Lifecycle.logical_length(Storage.Empty{}) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  earlier: {Nat.is_lt(other, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) -> Empty:
  Arith.lt_zero(other,
    %Equal.sym(Nat, Lifecycle.logical_length(Storage.Empty{}), Shape.logical_numel(Shape.dimensions(shape)), exact) : {Nat.is_lt(other, _) == True{} : Bool}
    earlier)

# The parallel copy meets the copy loop's meaning on every source and target.
def built_meaning(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, source: Core.Tensor<input>,
  target: Core.Tensor<Layout.output(input, axes, plan)>) ->
  Transforms.LoopMeaning(input, axes, plan, source, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Transforms.built(input, axes, plan, source, target)):
  Core.TTensor{storage, exact} = target
  match storage:
    case Storage.Empty{}:
      Transforms.Prefix{other => {==},
        other => earlier => Empty.absurd({U32.to_nat(U32.from_nat(other)) == other : Nat}, empty_below(Layout.output(input, axes, plan), other, exact, earlier)),
        other => earlier => Empty.absurd(Core.ObservationChecked(Core.Packed{Layout.output(input, axes, plan), Core.TTensor{Storage.Empty{}, exact}}, U32.from_nat(other)),
          empty_below(Layout.output(input, axes, plan), other, exact, earlier)),
        other => earlier => Empty.absurd({Core.checked_observe(Layout.output(input, axes, plan), Core.TTensor{Storage.Empty{}, exact}, U32.from_nat(other)) ==
          Transforms.expected_output(input, axes, plan, source, U32.from_nat(other)) : Maybe<&2, F32>}, empty_below(Layout.output(input, axes, plan), other, exact, earlier))}
    case Storage.Owned{+d, +n, array, balanced, valid}:
      owned_meaning(input, axes, plan, d, n, valid, exact, source)

def cloned_success(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +d: Nat, +n: Nat,
  -valid: {Storage.bounds(d, n) == True{} : Bool}, -exact: {n == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat},
  +ds: Nat, +ns: Nat, -a: Array<F32>, pair: Array<F32> & Array<F32>, -same: {pair == Array.clone(F32, a) : Array<F32> & Array<F32>},
  -b: Model.Balanced(F32, ds, a), -ok: {Storage.bounds(ds, ns) == True{} : Bool},
  -fits: {Lifecycle.logical_length(Storage.Owned{ds, ns, a, b, ok}) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  {Totality.copy_succeeded(input, axes, plan, Transforms.built_cloned(input, axes, plan, d, n, valid, exact, ds, ns, a, pair, same, b, ok, fits)) == True{} : Bool}:
  (kept, copy) = pair
  {==}

# An owned output has a cell, so a source of the same element count is owned.
def owned_success(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +d: Nat, +n: Nat,
  -valid: {Storage.bounds(d, n) == True{} : Bool}, -exact: {n == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}, source: Core.Tensor<input>) ->
  Fact<{Totality.copy_succeeded(input, axes, plan, Transforms.built_owned(input, axes, plan, d, n, valid, exact, source)) == True{} : Bool}>:
  Core.TTensor{storage, fits} = source
  match storage:
    case Storage.Empty{}:
      -positive = Arith.and_left(Nat.is_lt(0n, n), Nat.is_le(n, Model.capacity(d)),
        Arith.and_right(Nat.is_le(d, 20n), Bool.and(Nat.is_lt(0n, n), Nat.is_le(n, Model.capacity(d))), valid))
      Fact{Empty.absurd({False{} == True{} : Bool},
        Arith.lt_zero(0n,
          %Equal.sym(Nat, 0n, n, Equal.trans(Nat, 0n, Shape.logical_numel(Shape.dimensions(input)), n, fits,
            Equal.sym(Nat, n, Shape.logical_numel(Shape.dimensions(input)), Equal.trans(Nat, n, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Shape.logical_numel(Shape.dimensions(input)), exact, plan_numel(input, axes, plan))))) : {Nat.is_lt(0n, _) == True{} : Bool}
          positive))}
    case Storage.Owned{+ds, +ns, a, b, ok}:
      Fact{cloned_success(input, axes, plan, d, n, valid, exact, ds, ns, a, Array.clone(F32, a), {==}, b, ok, fits)}

def built_success(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, source: Core.Tensor<input>,
  target: Core.Tensor<Layout.output(input, axes, plan)>) ->
  Fact<{Totality.copy_succeeded(input, axes, plan, Transforms.built(input, axes, plan, source, target)) == True{} : Bool}>:
  Core.TTensor{storage, exact} = target
  match storage:
    case Storage.Empty{}:
      Fact{{==}}
    case Storage.Owned{+d, +n, array, balanced, valid}:
      owned_success(input, axes, plan, d, n, valid, exact, source)

def word(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +index: Nat) -> F32:
  Replayed.value(input, fill, ops, U32.from_nat(Values.permute_source_index(input, axes, index)))

def leaf(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +position: Nat, inside: Bool) -> F32:
  match inside:
    case False{}:
      0.0
    case True{}:
      word(input, axes, fill, ops, position)

def spec(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat, +depth: Nat, +base: Nat) -> Array<F32>:
  match depth:
    case 0n:
      ALeaf{leaf(input, axes, fill, ops, base, Nat.is_lt(base, count))}
    case 1n+inner:
      ANode{spec(input, axes, fill, ops, count, inner, base), spec(input, axes, fill, ops, count, inner, Nat.add(Model.capacity(inner), base))}

def spec_balanced(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat, +depth: Nat, +base: Nat) -> Model.Balanced(F32, depth, spec(input, axes, fill, ops, count, depth, base)):
  match depth:
    case 0n:
      Unit{}
    case 1n+inner:
      (spec_balanced(input, axes, fill, ops, count, inner, base), spec_balanced(input, axes, fill, ops, count, inner, Nat.add(Model.capacity(inner), base)))

def new_spec(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +depth: Nat, +base: Nat) -> {Array.new(F32, depth, 0.0) == spec(input, axes, fill, ops, 0n, depth, base) : Array<F32>}:
  match depth:
    case 0n:
      %Equal.sym(Bool, Nat.is_lt(base, 0n), False{}, Facts.lt_zero_false(base)) :
        {ALeaf{0.0} == ALeaf{leaf(input, axes, fill, ops, base, _)} : Array<F32>}
      {==}
    case 1n+inner:
      %Equal.sym(Array<F32>, Array.new(F32, inner, 0.0), spec(input, axes, fill, ops, 0n, inner, base), new_spec(input, axes, fill, ops, inner, base)) :
        {ANode{_, Array.new(F32, inner, 0.0)} == spec(input, axes, fill, ops, 0n, 1n+inner, base) : Array<F32>}
      %Equal.sym(Array<F32>, Array.new(F32, inner, 0.0), spec(input, axes, fill, ops, 0n, inner, Nat.add(Model.capacity(inner), base)),
        new_spec(input, axes, fill, ops, inner, Nat.add(Model.capacity(inner), base))) :
        {ANode{spec(input, axes, fill, ops, 0n, inner, base), _} == spec(input, axes, fill, ops, 0n, 1n+inner, base) : Array<F32>}
      {==}

# A subtree whose positions all lie above k is the same for k and 1 + k.
def above(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +k: Nat, +depth: Nat, +b: Nat, +h: {Nat.is_lt(k, b) == True{} : Bool}) ->
  {spec(input, axes, fill, ops, k, depth, b) == spec(input, axes, fill, ops, 1n+k, depth, b) : Array<F32>}:
  match depth:
    case 0n:
      %Equal.sym(Bool, Nat.is_lt(b, k), False{}, Facts.lt_asym(k, b, h)) :
        {ALeaf{leaf(input, axes, fill, ops, b, _)} == ALeaf{leaf(input, axes, fill, ops, b, Nat.is_lt(b, 1n+k))} : Array<F32>}
      %Equal.sym(Bool, Nat.is_lt(b, 1n+k), False{}, Facts.above_false(k, b, h)) :
        {ALeaf{0.0} == ALeaf{leaf(input, axes, fill, ops, b, _)} : Array<F32>}
      {==}
    case 1n+inner:
      +high = Nat.add(Model.capacity(inner), b)
      %Equal.sym(Array<F32>, spec(input, axes, fill, ops, k, inner, b), spec(input, axes, fill, ops, 1n+k, inner, b), above(input, axes, fill, ops, k, inner, b, h)) :
        {ANode{_, spec(input, axes, fill, ops, k, inner, high)} == spec(input, axes, fill, ops, 1n+k, 1n+inner, b) : Array<F32>}
      %Equal.sym(Array<F32>, spec(input, axes, fill, ops, k, inner, high), spec(input, axes, fill, ops, 1n+k, inner, high),
        above(input, axes, fill, ops, k, inner, high, Arith.lt_le_trans(k, b, high, h, Api.le_add_left(Model.capacity(inner), b)))) :
        {ANode{spec(input, axes, fill, ops, 1n+k, inner, b), _} == spec(input, axes, fill, ops, 1n+k, 1n+inner, b) : Array<F32>}
      {==}

# A subtree whose positions all lie below k is the same for k and 1 + k.
def below(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +k: Nat, +depth: Nat, +b: Nat, +h: {Nat.is_le(Nat.add(Model.capacity(depth), b), k) == True{} : Bool}) ->
  {spec(input, axes, fill, ops, k, depth, b) == spec(input, axes, fill, ops, 1n+k, depth, b) : Array<F32>}:
  match depth:
    case 0n:
      +lower = Facts.below_true(b, k, h)
      %Equal.sym(Bool, Nat.is_lt(b, k), True{}, lower) :
        {ALeaf{leaf(input, axes, fill, ops, b, _)} == ALeaf{leaf(input, axes, fill, ops, b, Nat.is_lt(b, 1n+k))} : Array<F32>}
      %Equal.sym(Bool, Nat.is_lt(b, 1n+k), True{}, Facts.lt_succ_right(b, k, lower)) :
        {ALeaf{word(input, axes, fill, ops, b)} == ALeaf{leaf(input, axes, fill, ops, b, _)} : Array<F32>}
      {==}
    case 1n+inner:
      +c = Model.capacity(inner)
      +high = Nat.add(c, b)
      +nested = Equal.trans(Bool, Nat.is_le(Nat.add(c, high), k), Nat.is_le(Nat.add(Nat.add(c, c), b), k), True{},
        Equal.cong(Nat, Bool, x => Nat.is_le(x, k), Nat.add(c, high), Nat.add(Nat.add(c, c), b), Index.add_assoc(c, c, b)), h)
      +low_h = Facts.le_trans(high, Nat.add(c, high), k, Facts.le_add_cong(c, b, high, Api.le_add_left(c, b)), nested)
      %Equal.sym(Array<F32>, spec(input, axes, fill, ops, k, inner, b), spec(input, axes, fill, ops, 1n+k, inner, b), below(input, axes, fill, ops, k, inner, b, low_h)) :
        {ANode{_, spec(input, axes, fill, ops, k, inner, high)} == spec(input, axes, fill, ops, 1n+k, 1n+inner, b) : Array<F32>}
      %Equal.sym(Array<F32>, spec(input, axes, fill, ops, k, inner, high), spec(input, axes, fill, ops, 1n+k, inner, high), below(input, axes, fill, ops, k, inner, high, nested)) :
        {ANode{spec(input, axes, fill, ops, 1n+k, inner, b), _} == spec(input, axes, fill, ops, 1n+k, 1n+inner, b) : Array<F32>}
      {==}

# Writing cell k at the path whose position is k turns the spec for k into
# the spec for 1 + k.
def write_spec(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +k: Nat, +depth: Nat, +base: Nat, +path: Model.Path(depth),
  +h: {Nat.add(Model.offset(depth, path), base) == k : Nat}) ->
  {ArrayCore.write_path(F32, depth, spec(input, axes, fill, ops, k, depth, base), path, word(input, axes, fill, ops, k)) == spec(input, axes, fill, ops, 1n+k, depth, base) : Array<F32>}:
  match depth:
    case 0n:
      %h : {ALeaf{word(input, axes, fill, ops, _)} == ALeaf{leaf(input, axes, fill, ops, base, Nat.is_lt(base, 1n+_))} : Array<F32>}
      %Equal.sym(Bool, Nat.is_lt(base, 1n+base), True{}, Facts.lt_self_succ(base)) :
        {ALeaf{word(input, axes, fill, ops, base)} == ALeaf{leaf(input, axes, fill, ops, base, _)} : Array<F32>}
      {==}
    case 1n+inner:
      Model.Step{+bit, +rest} = path
      match bit:
        case False{}:
          +c = Model.capacity(inner)
          +high = Nat.add(c, base)
          +inside = Equal.trans(Bool, Nat.is_lt(k, high), Nat.is_lt(Nat.add(Model.offset(inner, rest), base), high), True{},
            Equal.cong(Nat, Bool, x => Nat.is_lt(x, high), k, Nat.add(Model.offset(inner, rest), base),
              Equal.sym(Nat, Nat.add(Model.offset(inner, rest), base), k, h)),
            Facts.lt_add_right(Model.offset(inner, rest), c, base, Api.offset_lt(inner, rest)))
          %Equal.sym(Array<F32>, ArrayCore.write_path(F32, inner, spec(input, axes, fill, ops, k, inner, base), rest, word(input, axes, fill, ops, k)),
            spec(input, axes, fill, ops, 1n+k, inner, base), write_spec(input, axes, fill, ops, k, inner, base, rest, h)) :
            {ANode{_, spec(input, axes, fill, ops, k, inner, high)} == spec(input, axes, fill, ops, 1n+k, 1n+inner, base) : Array<F32>}
          %Equal.sym(Array<F32>, spec(input, axes, fill, ops, k, inner, high), spec(input, axes, fill, ops, 1n+k, inner, high), above(input, axes, fill, ops, k, inner, high, inside)) :
            {ANode{spec(input, axes, fill, ops, 1n+k, inner, base), _} == spec(input, axes, fill, ops, 1n+k, 1n+inner, base) : Array<F32>}
          {==}
        case True{}:
          +c = Model.capacity(inner)
          +high = Nat.add(c, base)
          +off = Model.offset(inner, rest)
          +lower = Equal.trans(Bool, Nat.is_le(high, k), Nat.is_le(high, Nat.add(c, Nat.add(off, base))), True{},
            Equal.cong(Nat, Bool, x => Nat.is_le(high, x), k, Nat.add(c, Nat.add(off, base)),
              Equal.trans(Nat, k, Nat.add(Nat.add(c, off), base), Nat.add(c, Nat.add(off, base)),
                Equal.sym(Nat, Nat.add(Nat.add(c, off), base), k, h),
                Equal.sym(Nat, Nat.add(c, Nat.add(off, base)), Nat.add(Nat.add(c, off), base), Index.add_assoc(c, off, base)))),
            Facts.le_add_cong(c, base, Nat.add(off, base), Api.le_add_left(off, base)))
          +shifted = Equal.trans(Nat, Nat.add(off, high), Nat.add(Nat.add(c, off), base), k, Facts.position_step(c, off, base), h)
          %Equal.sym(Array<F32>, spec(input, axes, fill, ops, k, inner, base), spec(input, axes, fill, ops, 1n+k, inner, base), below(input, axes, fill, ops, k, inner, base, lower)) :
            {ANode{_, ArrayCore.write_path(F32, inner, spec(input, axes, fill, ops, k, inner, high), rest, word(input, axes, fill, ops, k))} == spec(input, axes, fill, ops, 1n+k, 1n+inner, base) : Array<F32>}
          %Equal.sym(Array<F32>, ArrayCore.write_path(F32, inner, spec(input, axes, fill, ops, k, inner, high), rest, word(input, axes, fill, ops, k)),
            spec(input, axes, fill, ops, 1n+k, inner, high), write_spec(input, axes, fill, ops, k, inner, high, rest, shifted)) :
            {ANode{spec(input, axes, fill, ops, 1n+k, inner, base), _} == spec(input, axes, fill, ops, 1n+k, 1n+inner, base) : Array<F32>}
          {==}

def TreeEq(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +plan: Layout.Plan<input, axes>, +ds: Nat, +fits: {Storage.bounds(ds, U32.to_nat(Shape.count(input))) == True{} : Bool}, +count: Nat, +exact: {count == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}, +depth: Nat, +base: Nat, -result: Array<F32> & (Array<F32> & Array<F32>)) -> Type:
  {Pair.fst(Array<F32>, Array<F32> & Array<F32>, result) == spec(input, axes, fill, ops, count, depth, base) : Array<F32>} &
    {Pair.snd(Array<F32>, Array<F32> & Array<F32>, result) == (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0)) : Array<F32> & Array<F32>}

def read_inside(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +plan: Layout.Plan<input, axes>, +ds: Nat, +fits: {Storage.bounds(ds, U32.to_nat(Shape.count(input))) == True{} : Bool}, +count: Nat, +exact: {count == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}, +base: Nat, result: Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, U32.from_nat(base)>>,
  done: {Result.is_done(&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, U32.from_nat(base)>, result) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(Values.permute_source_index(input, axes, base))), U32.to_nat(Shape.count(input))) == True{} : Bool}:
  match result:
    case Fail{error}:
      Empty.absurd({Nat.is_lt(U32.to_nat(U32.from_nat(Values.permute_source_index(input, axes, base))), U32.to_nat(Shape.count(input))) == True{} : Bool}, Arith.clash(done))
    case Done{+mapping}:
      %Equal.sym(Nat, U32.to_nat(Shape.count(input)), Shape.logical_numel(Shape.dimensions(input)), Reads.count_exact(input)) : {Nat.is_lt(U32.to_nat(U32.from_nat(Values.permute_source_index(input, axes, base))), _) == True{} : Bool}
      mapped_inside(input, axes, plan, U32.from_nat(base), mapping)

# An inside cell reads the replayed word `permute_source_index` names.
def word_read(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +plan: Layout.Plan<input, axes>, +ds: Nat, +fits: {Storage.bounds(ds, U32.to_nat(Shape.count(input))) == True{} : Bool}, +count: Nat, +exact: {count == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}, +base: Nat, +decided: {True{} == Nat.is_lt(base, count) : Bool}, -pair: Array<F32> & F32,
  +same: {pair == Array.get(F32, Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Build.at(Build.recipe(Build.Request{input, axes}), base)) : Array<F32> & F32}) ->
  {Refinement.second(F32, pair) == word(input, axes, fill, ops, base) : F32}:
  +index = U32.from_nat(Values.permute_source_index(input, axes, base))
  +below : {Nat.is_lt(base, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool} =
    %exact : {Nat.is_lt(base, _) == True{} : Bool}
    Equal.sym(Bool, True{}, Nat.is_lt(base, count), decided)
  +inside = read_inside(input, axes, fill, ops, plan, ds, fits, count, exact, base, Layout.map_index(input, axes, plan, U32.from_nat(base)),
    mapping_done(input, axes, plan, U32.from_nat(base),
      %Equal.sym(Nat, U32.to_nat(U32.from_nat(base)), base, word_below(Layout.output(input, axes, plan), base, below)) : {Nat.is_lt(_, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool}
      below))
  +exact_read = Equal.trans(Array<F32> & F32, pair, Array.get(F32, Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Build.at(Build.recipe(Build.Request{input, axes}), base)), Array.get(F32, Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), index), same,
    Equal.cong(U32, Array<F32> & F32, word => Array.get(F32, Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), word), Build.at(Build.recipe(Build.Request{input, axes}), base), index,
      source_word(input, axes, base)))
  Equal.trans(F32, Refinement.second(F32, pair), Api.current((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), index, fill), word(input, axes, fill, ops, base),
    BinaryBuild.got_value(ds, (Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), fill, U32.to_nat(Shape.count(input)), index, pair, exact_read, fits, inside, Owned.norm_written(U32.to_nat(Shape.count(input)), ops, Nil{}, Api.hit_of(U32.to_nat(Shape.count(input)), ops), {==}, {==})),
    Owned.norm_current(U32.to_nat(Shape.count(input)), ops, Nil{}, index, fill, Api.hit_of(U32.to_nat(Shape.count(input)), ops), {==}))

def kept_read(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +plan: Layout.Plan<input, axes>, +ds: Nat, +fits: {Storage.bounds(ds, U32.to_nat(Shape.count(input))) == True{} : Bool}, +count: Nat, +exact: {count == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}, +base: Nat, -pair: Array<F32> & F32,
  +same: {pair == Array.get(F32, Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Build.at(Build.recipe(Build.Request{input, axes}), base)) : Array<F32> & F32}) ->
  {Refinement.first(F32, pair) == Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill) : Array<F32>}:
  BinaryBuild.got_kept(ds, (Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), fill, Build.at(Build.recipe(Build.Request{input, axes}), base), pair, same)

def cell_read(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +plan: Layout.Plan<input, axes>, +ds: Nat, +fits: {Storage.bounds(ds, U32.to_nat(Shape.count(input))) == True{} : Bool}, +count: Nat, +exact: {count == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}, +base: Nat, +decided: {True{} == Nat.is_lt(base, count) : Bool}, pair: Array<F32> & F32,
  +same: {pair == Array.get(F32, Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Build.at(Build.recipe(Build.Request{input, axes}), base)) : Array<F32> & F32}) ->
  TreeEq(input, axes, fill, ops, plan, ds, fits, count, exact, 0n, base, Parallel.leaf(~(Array<F32> & Array<F32>), Build.take(Array.new(F32, 0n, 0.0), pair))):
  (kept, x) = pair
  +ex = word_read(input, axes, fill, ops, plan, ds, fits, count, exact, base, decided, (kept, x), same)
  (%decided : {ALeaf{x} == ALeaf{leaf(input, axes, fill, ops, base, _)} : Array<F32>}
   %Equal.sym(F32, x, word(input, axes, fill, ops, base), ex) : {ALeaf{_} == ALeaf{word(input, axes, fill, ops, base)} : Array<F32>}
   {==},
   Equal.cong(Array<F32>, Array<F32> & Array<F32>, k => (k, Array.new(F32, 0n, 0.0)), kept, Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), kept_read(input, axes, fill, ops, plan, ds, fits, count, exact, base, (kept, x), same)))

def cell_eq(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +plan: Layout.Plan<input, axes>, +ds: Nat, +fits: {Storage.bounds(ds, U32.to_nat(Shape.count(input))) == True{} : Bool}, +count: Nat, +exact: {count == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}, +base: Nat, inside: Bool, +decided: {inside == Nat.is_lt(base, count) : Bool}) ->
  TreeEq(input, axes, fill, ops, plan, ds, fits, count, exact, 0n, base, Parallel.leaf(~(Array<F32> & Array<F32>), Build.cell(Build.recipe(Build.Request{input, axes}), base, (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0)), inside))):
  match inside:
    case False{}:
      (%decided : {ALeaf{0.0} == ALeaf{leaf(input, axes, fill, ops, base, _)} : Array<F32>}
       {==}, {==})
    case True{}:
      cell_read(input, axes, fill, ops, plan, ds, fits, count, exact, base, decided, Array.get(F32, Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Build.at(Build.recipe(Build.Request{input, axes}), base)), {==})

def tree_join_eq(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +plan: Layout.Plan<input, axes>, +ds: Nat, +fits: {Storage.bounds(ds, U32.to_nat(Shape.count(input))) == True{} : Bool}, +count: Nat, +exact: {count == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}, +inner: Nat, +base: Nat, -a: Array<F32>, high: Array<F32> & (Array<F32> & Array<F32>),
  ea: {a == spec(input, axes, fill, ops, count, inner, base) : Array<F32>},
  facts: TreeEq(input, axes, fill, ops, plan, ds, fits, count, exact, inner, Nat.add(Model.capacity(inner), base), high)) ->
  TreeEq(input, axes, fill, ops, plan, ds, fits, count, exact, 1n+inner, base, Parallel.join(~(Array<F32> & Array<F32>), a, high)):
  (b, inputs) = high
  (eb, kept) = facts
  (%Equal.sym(Array<F32>, a, spec(input, axes, fill, ops, count, inner, base), ea) : {ANode{_, b} == ANode{spec(input, axes, fill, ops, count, inner, base), spec(input, axes, fill, ops, count, inner, Nat.add(Model.capacity(inner), base))} : Array<F32>}
   %Equal.sym(Array<F32>, b, spec(input, axes, fill, ops, count, inner, Nat.add(Model.capacity(inner), base)), eb) : {ANode{spec(input, axes, fill, ops, count, inner, base), _} == ANode{spec(input, axes, fill, ops, count, inner, base), spec(input, axes, fill, ops, count, inner, Nat.add(Model.capacity(inner), base))} : Array<F32>}
   {==}, kept)

def tree_node_eq(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +plan: Layout.Plan<input, axes>, +ds: Nat, +fits: {Storage.bounds(ds, U32.to_nat(Shape.count(input))) == True{} : Bool}, +count: Nat, +exact: {count == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}, +inner: Nat, +base: Nat, low: Array<F32> & (Array<F32> & Array<F32>), facts: TreeEq(input, axes, fill, ops, plan, ds, fits, count, exact, inner, base, low),
  next: TreeEq(input, axes, fill, ops, plan, ds, fits, count, exact, inner, Nat.add(Model.capacity(inner), base), Parallel.tree(~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.cell, Build.recipe(Build.Request{input, axes}), inner, count, Nat.add(Model.capacity(inner), base), (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0))))) ->
  TreeEq(input, axes, fill, ops, plan, ds, fits, count, exact, 1n+inner, base, Parallel.node(~Array<F32>, ~Array<F32>, l => r => Parallel.tree(~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.cell, Build.recipe(Build.Request{input, axes}), inner, count, Nat.add(Model.capacity(inner), base), (l, r)), low)):
  (a, inputs) = low
  (la, ra) = inputs
  (ea, kept) = facts
  %Equal.sym(Array<F32> & Array<F32>, (la, ra), (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0)), kept) :
    TreeEq(input, axes, fill, ops, plan, ds, fits, count, exact, 1n+inner, base, Parallel.node(~Array<F32>, ~Array<F32>, l => r => Parallel.tree(~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.cell, Build.recipe(Build.Request{input, axes}), inner, count, Nat.add(Model.capacity(inner), base), (l, r)), (a, _)))
  tree_join_eq(input, axes, fill, ops, plan, ds, fits, count, exact, inner, base, a, Parallel.tree(~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.cell, Build.recipe(Build.Request{input, axes}), inner, count, Nat.add(Model.capacity(inner), base), (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0))), ea, next)

def pair_join_eq(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +plan: Layout.Plan<input, axes>, +ds: Nat, +fits: {Storage.bounds(ds, U32.to_nat(Shape.count(input))) == True{} : Bool}, +count: Nat, +exact: {count == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}, +base: Nat, x: F32, high: F32 & (Array<F32> & Array<F32>),
  ex: {ALeaf{x} == spec(input, axes, fill, ops, count, 0n, base) : Array<F32>},
  leaf_facts: TreeEq(input, axes, fill, ops, plan, ds, fits, count, exact, 0n, Nat.add(1n, base), Parallel.leaf(~(Array<F32> & Array<F32>), high))) ->
  TreeEq(input, axes, fill, ops, plan, ds, fits, count, exact, 1n, base, Parallel.pair_join(~(Array<F32> & Array<F32>), x, high)):
  (y, inputs) = high
  (ey, kept) = leaf_facts
  (%Equal.sym(Array<F32>, ALeaf{x}, spec(input, axes, fill, ops, count, 0n, base), ex) : {ANode{_, ALeaf{y}} == ANode{spec(input, axes, fill, ops, count, 0n, base), spec(input, axes, fill, ops, count, 0n, Nat.add(1n, base))} : Array<F32>}
   %Equal.sym(Array<F32>, ALeaf{y}, spec(input, axes, fill, ops, count, 0n, Nat.add(1n, base)), ey) : {ANode{spec(input, axes, fill, ops, count, 0n, base), _} == ANode{spec(input, axes, fill, ops, count, 0n, base), spec(input, axes, fill, ops, count, 0n, Nat.add(1n, base))} : Array<F32>}
   {==}, kept)

def pair_eq(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +plan: Layout.Plan<input, axes>, +ds: Nat, +fits: {Storage.bounds(ds, U32.to_nat(Shape.count(input))) == True{} : Bool}, +count: Nat, +exact: {count == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}, +base: Nat, low: F32 & (Array<F32> & Array<F32>), leaf_facts: TreeEq(input, axes, fill, ops, plan, ds, fits, count, exact, 0n, base, Parallel.leaf(~(Array<F32> & Array<F32>), low))) ->
  TreeEq(input, axes, fill, ops, plan, ds, fits, count, exact, 1n, base, Parallel.pair(~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.cell, Build.recipe(Build.Request{input, axes}), count, base, low)):
  (x, inputs) = low
  (la, ra) = inputs
  (ex, kept) = leaf_facts
  %Equal.sym(Array<F32> & Array<F32>, (la, ra), (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0)), kept) :
    TreeEq(input, axes, fill, ops, plan, ds, fits, count, exact, 1n, base, Parallel.pair(~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.cell, Build.recipe(Build.Request{input, axes}), count, base, (x, _)))
  pair_join_eq(input, axes, fill, ops, plan, ds, fits, count, exact, base, x, Build.cell(Build.recipe(Build.Request{input, axes}), Nat.add(1n, base), (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0)), Nat.is_lt(Nat.add(1n, base), count)), ex,
    cell_eq(input, axes, fill, ops, plan, ds, fits, count, exact, Nat.add(1n, base), Nat.is_lt(Nat.add(1n, base), count), {==}))

def tree_eq(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +plan: Layout.Plan<input, axes>, +ds: Nat, +fits: {Storage.bounds(ds, U32.to_nat(Shape.count(input))) == True{} : Bool}, +count: Nat, +exact: {count == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}, +depth: Nat, +base: Nat) -> TreeEq(input, axes, fill, ops, plan, ds, fits, count, exact, depth, base, Parallel.tree(~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.cell, Build.recipe(Build.Request{input, axes}), depth, count, base, (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0)))):
  match depth:
    case 0n:
      cell_eq(input, axes, fill, ops, plan, ds, fits, count, exact, base, Nat.is_lt(base, count), {==})
    case 1n+inner:
      match inner:
        case 0n:
          pair_eq(input, axes, fill, ops, plan, ds, fits, count, exact, base, Build.cell(Build.recipe(Build.Request{input, axes}), base, (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0)), Nat.is_lt(base, count)), cell_eq(input, axes, fill, ops, plan, ds, fits, count, exact, base, Nat.is_lt(base, count), {==}))
        case 1n+below:
          tree_node_eq(input, axes, fill, ops, plan, ds, fits, count, exact, inner, base, Parallel.tree(~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.cell, Build.recipe(Build.Request{input, axes}), inner, count, base, (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0))), tree_eq(input, axes, fill, ops, plan, ds, fits, count, exact, inner, base), tree_eq(input, axes, fill, ops, plan, ds, fits, count, exact, inner, Nat.add(Model.capacity(inner), base)))

def tree_first(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +plan: Layout.Plan<input, axes>, +ds: Nat, +fits: {Storage.bounds(ds, U32.to_nat(Shape.count(input))) == True{} : Bool}, +count: Nat, +exact: {count == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}, +depth: Nat, +base: Nat, -result: Array<F32> & (Array<F32> & Array<F32>), facts: TreeEq(input, axes, fill, ops, plan, ds, fits, count, exact, depth, base, result)) ->
  {Pair.fst(Array<F32>, Array<F32> & Array<F32>, result) == spec(input, axes, fill, ops, count, depth, base) : Array<F32>}:
  (first, kept) = facts
  first

def fork_eq(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +plan: Layout.Plan<input, axes>, +ds: Nat, +fits: {Storage.bounds(ds, U32.to_nat(Shape.count(input))) == True{} : Bool}, +count: Nat, +exact: {count == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}, +inner: Nat, +rest: Nat, +base: Nat,
  low: {Parallel.build(~Build.Request, ~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.recipe, ~Build.split, ~Build.cell, Build.Request{input, axes}, inner, rest, count, base, (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0))) == spec(input, axes, fill, ops, count, inner, base) : Array<F32>},
  high: {Parallel.build(~Build.Request, ~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.recipe, ~Build.split, ~Build.cell, Build.Request{input, axes}, inner, rest, count, Nat.add(Model.capacity(inner), base), (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0))) == spec(input, axes, fill, ops, count, inner, Nat.add(Model.capacity(inner), base)) : Array<F32>},
  lp: Array<F32> & Array<F32>, +le: {lp == Array.clone(F32, Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill)) : Array<F32> & Array<F32>}) ->
  {Parallel.fork(~Array<F32>, ~Array<F32>, Build.halves(lp, Array.clone(F32, Array.new(F32, 0n, 0.0))), l => r => Parallel.build(~Build.Request, ~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.recipe, ~Build.split, ~Build.cell, Build.Request{input, axes}, inner, rest, count, base, (l, r)), l => r => Parallel.build(~Build.Request, ~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.recipe, ~Build.split, ~Build.cell, Build.Request{input, axes}, inner, rest, count, Nat.add(Model.capacity(inner), base), (l, r))) ==
    spec(input, axes, fill, ops, count, 1n+inner, base) : Array<F32>}:
  (l1, l2) = lp
  %Equal.sym(Array<F32>, l1, Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), BinaryBuild.clone_first_eq(ds, (Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), fill, (l1, l2), le)) :
    {ANode{Parallel.build(~Build.Request, ~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.recipe, ~Build.split, ~Build.cell, Build.Request{input, axes}, inner, rest, count, base, (_, Array.new(F32, 0n, 0.0))), Parallel.build(~Build.Request, ~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.recipe, ~Build.split, ~Build.cell, Build.Request{input, axes}, inner, rest, count, Nat.add(Model.capacity(inner), base), (l2, Array.new(F32, 0n, 0.0)))} == spec(input, axes, fill, ops, count, 1n+inner, base) : Array<F32>}
  %Equal.sym(Array<F32>, l2, Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), BinaryBuild.clone_second_eq(ds, (Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), fill, (l1, l2), le)) :
    {ANode{Parallel.build(~Build.Request, ~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.recipe, ~Build.split, ~Build.cell, Build.Request{input, axes}, inner, rest, count, base, (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0))), Parallel.build(~Build.Request, ~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.recipe, ~Build.split, ~Build.cell, Build.Request{input, axes}, inner, rest, count, Nat.add(Model.capacity(inner), base), (_, Array.new(F32, 0n, 0.0)))} == spec(input, axes, fill, ops, count, 1n+inner, base) : Array<F32>}
  %Equal.sym(Array<F32>, Parallel.build(~Build.Request, ~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.recipe, ~Build.split, ~Build.cell, Build.Request{input, axes}, inner, rest, count, base, (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0))), spec(input, axes, fill, ops, count, inner, base), low) :
    {ANode{_, Parallel.build(~Build.Request, ~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.recipe, ~Build.split, ~Build.cell, Build.Request{input, axes}, inner, rest, count, Nat.add(Model.capacity(inner), base), (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0)))} == spec(input, axes, fill, ops, count, 1n+inner, base) : Array<F32>}
  %Equal.sym(Array<F32>, Parallel.build(~Build.Request, ~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.recipe, ~Build.split, ~Build.cell, Build.Request{input, axes}, inner, rest, count, Nat.add(Model.capacity(inner), base), (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0))), spec(input, axes, fill, ops, count, inner, Nat.add(Model.capacity(inner), base)), high) :
    {ANode{spec(input, axes, fill, ops, count, inner, base), _} == spec(input, axes, fill, ops, count, 1n+inner, base) : Array<F32>}
  {==}

# Whatever the split, the parallel build returns the spec.
def build_eq(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +plan: Layout.Plan<input, axes>, +ds: Nat, +fits: {Storage.bounds(ds, U32.to_nat(Shape.count(input))) == True{} : Bool}, +count: Nat, +exact: {count == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}, +depth: Nat, +splits: Nat, +base: Nat) ->
  {Parallel.build(~Build.Request, ~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.recipe, ~Build.split, ~Build.cell, Build.Request{input, axes}, depth, splits, count, base, (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0))) == spec(input, axes, fill, ops, count, depth, base) : Array<F32>}:
  match depth splits:
    case 1n+inner 1n+rest:
      fork_eq(input, axes, fill, ops, plan, ds, fits, count, exact, inner, rest, base, build_eq(input, axes, fill, ops, plan, ds, fits, count, exact, inner, rest, base), build_eq(input, axes, fill, ops, plan, ds, fits, count, exact, inner, rest, Nat.add(Model.capacity(inner), base)),
        Array.clone(F32, Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill)), {==})
    case 0n _:
      tree_first(input, axes, fill, ops, plan, ds, fits, count, exact, 0n, base, Parallel.tree(~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.cell, Build.recipe(Build.Request{input, axes}), 0n, count, base, (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0))), tree_eq(input, axes, fill, ops, plan, ds, fits, count, exact, 0n, base))
    case 1n+inner 0n:
      tree_first(input, axes, fill, ops, plan, ds, fits, count, exact, 1n+inner, base, Parallel.tree(~Build.Recipe, ~Array<F32>, ~Array<F32>, ~Build.cell, Build.recipe(Build.Request{input, axes}), 1n+inner, count, base, (Api.arr((Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{})), ds, fill), Array.new(F32, 0n, 0.0))), tree_eq(input, axes, fill, ops, plan, ds, fits, count, exact, 1n+inner, base))

