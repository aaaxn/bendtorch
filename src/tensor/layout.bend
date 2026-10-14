import Base
import ../shape/core.bend as Shape
import ./axis-spec.bend as Axis

type Error is Data:
  ShapeError{error: Shape.Error}
  AxesLengthMismatch{}
  AxisOutOfBounds{}
  DuplicateAxis{}
  ElementCountMismatch{}
  MappingError{}

def contains(+axis: U32, axes: List<&2, U32>) -> Bool:
  match axes:
    case Nil{}:
      False{}
    case +head <> tail:
      Bool.or(U32.is_eq(axis, head), contains(axis, tail))

def unique(axes: List<&2, U32>) -> Bool:
  match axes:
    case Nil{}:
      True{}
    case +head <> +tail:
      Bool.and(Bool.not(contains(head, tail)), unique(tail))

def bounded(axes: List<&2, U32>, +rank: U32) -> Bool:
  match axes:
    case Nil{}:
      True{}
    case head <> tail:
      Bool.and(U32.is_lt(head, rank), bounded(tail, rank))

def bijective(+shape: Shape.Shape, +axes: List<&2, U32>) -> Bool:
  Bool.and(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(shape))),
    Bool.and(bounded(axes, Shape.rank(shape)), unique(axes)))

def lookup(values: List<&2, U32>, index: Nat) -> Maybe<&2, U32>:
  match values index:
    case Nil{} _:
      None{}
    case head <> tail 0n:
      Some{head}
    case head <> tail 1n+p:
      lookup(tail, p)

def reorder_tail(head: U32, result: Maybe<&2, List<&2, U32>>) -> Maybe<&2, List<&2, U32>>:
  match result:
    case None{}:
      None{}
    case Some{tail}:
      Some{head <> tail}

def reorder_head(rec: U32 -> Maybe<&2, List<&2, U32>>, head: Maybe<&2, U32>) -> Maybe<&2, List<&2, U32>>:
  match head:
    case None{}:
      None{}
    case Some{value}:
      rec(value)

def reorder(+values: List<&2, U32>, axes: List<&2, U32>) -> Maybe<&2, List<&2, U32>>:
  match axes:
    case Nil{}:
      Some{Nil{}}
    case axis <> tail:
      reorder_head(value => reorder_tail(value, reorder(values, tail)), lookup(values, U32.to_nat(axis)))

def coordinate_for(axes: List<&2, U32>, coordinates: List<&2, U32>, +axis: U32) -> Maybe<&2, U32>:
  match axes coordinates:
    case Nil{} _:
      None{}
    case _ Nil{}:
      None{}
    case +head <> tail value <> rest:
      Bool.pick(Maybe<&2, U32>, U32.is_eq(head, axis), Some{value}, coordinate_for(tail, rest, axis))

def inverse_coordinates(dimensions: List<&2, U32>, +axes: List<&2, U32>,
  +coordinates: List<&2, U32>, +axis: U32) -> Maybe<&2, List<&2, U32>>:
  match dimensions:
    case Nil{}:
      Some{Nil{}}
    case dimension <> tail:
      reorder_head(value => reorder_tail(value, inverse_coordinates(tail, axes, coordinates, U32.inc(axis))),
        coordinate_for(axes, coordinates, axis))

type Plan<-input: Shape.Shape, -axes: List<&2, U32>> is Data:
  Permutation{output: Shape.Shape,
    valid: {True{} == bijective(input, axes) : Bool},
    ordered: {Some{Shape.dimensions(output)} == reorder(Shape.dimensions(input), axes) : Maybe<&2, List<&2, U32>>},
    count: {True{} == Nat.is_eq(Shape.elements(input), Shape.elements(output)) : Bool}}

def output(-input: Shape.Shape, -axes: List<&2, U32>, plan: Plan<input, axes>) -> Shape.Shape:
  Permutation{shape, valid, ordered, count} = plan
  shape

def plan_counted(+input: Shape.Shape, +axes: List<&2, U32>, +out: Shape.Shape,
  valid: {True{} == bijective(input, axes) : Bool},
  ordered: {Some{Shape.dimensions(out)} == reorder(Shape.dimensions(input), axes) : Maybe<&2, List<&2, U32>>},
  decision: Shape.BoolCase<Nat.is_eq(Shape.elements(input), Shape.elements(out))>) ->
  Result<&2, &2, Error, Plan<input, axes>>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{ElementCountMismatch{}}
    case Shape.BTrue{equal}:
      Done{Permutation{out, valid, ordered, equal}}

def plan_built(+input: Shape.Shape, +axes: List<&2, U32>, -dimensions: List<&2, U32>,
  valid: {True{} == bijective(input, axes) : Bool},
  selected: {Some{dimensions} == reorder(Shape.dimensions(input), axes) : Maybe<&2, List<&2, U32>>},
  built: Shape.ConstructResult<dimensions>) -> Result<&2, &2, Error, Plan<input, axes>>:
  match built:
    case Shape.SError{error}:
      Fail{ShapeError{error}}
    case Shape.SBuilt{+out, preserved}:
      plan_counted(input, axes, out, valid,
        Equal.trans(Maybe<&2, List<&2, U32>>, Some{Shape.dimensions(out)}, Some{dimensions},
          reorder(Shape.dimensions(input), axes),
          Equal.cong(List<&2, U32>, Maybe<&2, List<&2, U32>>, xs => Some{xs}, Shape.dimensions(out), dimensions, preserved), selected),
        Shape.bool_case(Nat.is_eq(Shape.elements(input), Shape.elements(out))))

def plan_selected(+input: Shape.Shape, +axes: List<&2, U32>, valid: {True{} == bijective(input, axes) : Bool},
  selected: Maybe<&2, List<&2, U32>>,
  same: {selected == reorder(Shape.dimensions(input), axes) : Maybe<&2, List<&2, U32>>}) -> Result<&2, &2, Error, Plan<input, axes>>:
  match selected:
    case None{}:
      Fail{MappingError{}}
    case Some{+dimensions}:
      plan_built(input, axes, dimensions, valid, same, Shape.construct(dimensions))

def plan_decided(+input: Shape.Shape, +axes: List<&2, U32>, decision: Shape.BoolCase<bijective(input, axes)>) ->
  Result<&2, &2, Error, Plan<input, axes>>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{MappingError{}}
    case Shape.BTrue{equal}:
      plan_selected(input, axes, equal, reorder(Shape.dimensions(input), axes), {==})

def plan_unique(+input: Shape.Shape, +axes: List<&2, U32>, decision: Shape.BoolCase<unique(axes)>) ->
  Result<&2, &2, Error, Plan<input, axes>>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{DuplicateAxis{}}
    case Shape.BTrue{equal}:
      plan_decided(input, axes, Shape.bool_case(bijective(input, axes)))

def plan_bounded(+input: Shape.Shape, +axes: List<&2, U32>, decision: Shape.BoolCase<bounded(axes, Shape.rank(input))>) ->
  Result<&2, &2, Error, Plan<input, axes>>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{AxisOutOfBounds{}}
    case Shape.BTrue{equal}:
      plan_unique(input, axes, Shape.bool_case(unique(axes)))

def plan_ranked(+input: Shape.Shape, +axes: List<&2, U32>,
  decision: Shape.BoolCase<Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))>) ->
  Result<&2, &2, Error, Plan<input, axes>>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{AxesLengthMismatch{}}
    case Shape.BTrue{equal}:
      plan_bounded(input, axes, Shape.bool_case(bounded(axes, Shape.rank(input))))

def permutation(+input: Shape.Shape, +axes: List<&2, U32>) -> Result<&2, &2, Error, Plan<input, axes>>:
  plan_ranked(input, axes,
    Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))))

def transpose_axes(dimensions: List<&2, U32>, +a: U32, +b: U32, +i: U32) -> List<&2, U32>:
  match dimensions:
    case Nil{}:
      Nil{}
    case dimension <> tail:
      Bool.pick(U32, U32.is_eq(i, a), b, Bool.pick(U32, U32.is_eq(i, b), a, i)) <>
        transpose_axes(tail, a, b, U32.inc(i))

def word_same(n: Nat, a: Word(n), b: Word(n)) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+p:
      WCon{ab, at} = a
      WCon{bb, bt} = b
      Bool.and(Bool.not(Bool.xor(ab, bb)), word_same(p, at, bt))

def word_value_same(a: U32, b: U32) -> Bool:
  match a b:
    case U32{x} U32{y}:
      word_same(32n, x, y)

def words_same(a: List<&2, U32>, b: List<&2, U32>) -> Bool:
  match a b:
    case Nil{} Nil{}:
      True{}
    case ah <> at bh <> bt:
      Bool.and(word_value_same(ah, bh), words_same(at, bt))
    case _ _:
      False{}

def maybe_same(a: Maybe<&2, List<&2, U32>>, b: Maybe<&2, List<&2, U32>>) -> Bool:
  match a b:
    case None{} None{}:
      True{}
    case Some{xs} Some{ys}:
      words_same(xs, ys)
    case _ _:
      False{}

type WordsValidStep<-dimension: U32, -dt: List<&2, U32>, -coordinate: U32, -ct: List<&2, U32>> is Data:
  WStep{bound: {True{} == U32.is_lt(coordinate, dimension) : Bool}, tail: WordsValid(dt, ct)}

def WordsValid(dims: List<&2, U32>, coordinates: List<&2, U32>) -> Data:
  match dims coordinates:
    case Nil{} Nil{}:
      Unit
    case dh <> dt ch <> ct:
      WordsValidStep<dh, dt, ch, ct>
    case _ _:
      Empty

def words_finish(-dimension: U32, -dt: List<&2, U32>, -coordinate: U32, -ct: List<&2, U32>,
  bound: {True{} == U32.is_lt(coordinate, dimension) : Bool}, result: Result<&2, &2, Shape.Error, WordsValid(dt, ct)>) ->
  Result<&2, &2, Shape.Error, WordsValid(dimension <> dt, coordinate <> ct)>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{tail}:
      Done{WStep{bound, tail}}

def words_decided(+dimension: U32, -dt: List<&2, U32>, +coordinate: U32, -ct: List<&2, U32>,
  rec: Unit -> Result<&2, &2, Shape.Error, WordsValid(dt, ct)>, decision: Shape.BoolCase<U32.is_lt(coordinate, dimension)>) ->
  Result<&2, &2, Shape.Error, WordsValid(dimension <> dt, coordinate <> ct)>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{Shape.CoordinateOutOfBounds{}}
    case Shape.BTrue{equal}:
      words_finish(dimension, dt, coordinate, ct, equal, rec(Unit{}))

def words_check(+dims: List<&2, U32>, +coordinates: List<&2, U32>) -> Result<&2, &2, Shape.Error, WordsValid(dims, coordinates)>:
  match dims coordinates:
    case Nil{} Nil{}:
      Done{Unit{}}
    case +dh <> +dt +ch <> +ct:
      words_decided(dh, dt, ch, ct, unit => words_check(dt, ct), Shape.bool_case(U32.is_lt(ch, dh)))
    case _ _:
      Fail{Shape.CoordinateRankMismatch{}}

type CheckedCoordinate<-shape: Shape.Shape> is Data:
  CWord{values: List<&2, U32>, valid: WordsValid(Shape.dimensions(shape), values)}

def coordinate_values(-shape: Shape.Shape, coordinate: CheckedCoordinate<shape>) -> List<&2, U32>:
  CWord{values, valid} = coordinate
  values

type FlatIndex<-shape: Shape.Shape> is Data:
  FIndex{value: U32, valid: {True{} == U32.is_lt(value, Shape.count(shape)) : Bool}}

def index_value(-shape: Shape.Shape, index: FlatIndex<shape>) -> U32:
  FIndex{value, valid} = index
  value

def flatten_words(dims: List<&2, U32>, coordinates: List<&2, U32>, accumulator: U32) -> U32:
  U32.from_nat(Shape.horner(dims, coordinates, U32.to_nat(accumulator)))

def reverse_words_go(values: List<&2, U32>, accumulator: List<&2, U32>) -> List<&2, U32>:
  match values:
    case Nil{}:
      accumulator
    case head <> tail:
      reverse_words_go(tail, head <> accumulator)

def reverse_words(values: List<&2, U32>) -> List<&2, U32>:
  reverse_words_go(values, Nil{})

def unflatten_words(reversed_dimensions: List<&2, U32>, +index: U32) -> List<&2, U32>:
  reverse_words(Shape.words(Shape.digits_words(reverse_words(reversed_dimensions), U32.to_nat(index))))

def reference_source_coordinates(+dimensions: List<&2, U32>, coordinates: Maybe<&2, List<&2, U32>>) -> Maybe<&2, U32>:
  match coordinates:
    case None{}:
      None{}
    case Some{values}:
      Some{U32.from_nat(Shape.offset(Shape.naturals(dimensions), Shape.naturals(values)))}

def reference_source(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Plan<input, axes>, index: U32) -> Maybe<&2, U32>:
  reference_source_coordinates(Shape.dimensions(input),
    Axis.scatter(U32, Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes),
      Shape.words(Shape.digits(Shape.extents(output(input, axes, plan)), U32.to_nat(index))), 0n))

type Mapping<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Plan<input, axes>, -i: U32> is Data:
  Mapped{output_index: FlatIndex<output(input, axes, plan)>,
    output_coordinate: CheckedCoordinate<output(input, axes, plan)>,
    input_coordinate: CheckedCoordinate<input>, index: FlatIndex<input>,
    requested: {index_value(output(input, axes, plan), output_index) == i : U32},
    decoded: {coordinate_values(output(input, axes, plan), output_coordinate) ==
      reverse_words(unflatten_words(reverse_words(Shape.dimensions(output(input, axes, plan))), i)) : List<&2, U32>},
    inverse: {Some{coordinate_values(input, input_coordinate)} ==
      inverse_coordinates(Shape.dimensions(input), axes,
        coordinate_values(output(input, axes, plan), output_coordinate), 0) : Maybe<&2, List<&2, U32>>},
    flattened: {index_value(input, index) ==
      flatten_words(Shape.dimensions(input), coordinate_values(input, input_coordinate), 0) : U32},
    roundtrip: {True{} == maybe_same(reorder(coordinate_values(input, input_coordinate), axes),
      Some{coordinate_values(output(input, axes, plan), output_coordinate)}) : Bool}}

def source_index(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Plan<input, axes>, -i: U32,
  mapping: Mapping<input, axes, plan, i>) -> U32:
  Mapped{oi, oc, ic, index, requested, decoded, inverse, flattened, roundtrip} = mapping
  index_value(input, index)

def output_words(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Plan<input, axes>, -i: U32,
  mapping: Mapping<input, axes, plan, i>) -> List<&2, U32>:
  Mapped{oi, oc, ic, index, requested, decoded, inverse, flattened, roundtrip} = mapping
  coordinate_values(output(input, axes, plan), oc)

def input_words(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Plan<input, axes>, -i: U32,
  mapping: Mapping<input, axes, plan, i>) -> List<&2, U32>:
  Mapped{oi, oc, ic, index, requested, decoded, inverse, flattened, roundtrip} = mapping
  coordinate_values(input, ic)

def map_flat(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Plan<input, axes>, +i: U32,
  oi: FlatIndex<output(input, axes, plan)>, oc: CheckedCoordinate<output(input, axes, plan)>,
  +ic: CheckedCoordinate<input>,
  requested: {index_value(output(input, axes, plan), oi) == i : U32},
  decoded: {coordinate_values(output(input, axes, plan), oc) ==
    reverse_words(unflatten_words(reverse_words(Shape.dimensions(output(input, axes, plan))), i)) : List<&2, U32>},
  inverse: {Some{coordinate_values(input, ic)} ==
    inverse_coordinates(Shape.dimensions(input), axes, coordinate_values(output(input, axes, plan), oc), 0) : Maybe<&2, List<&2, U32>>},
  roundtrip: {True{} == maybe_same(reorder(coordinate_values(input, ic), axes),
    Some{coordinate_values(output(input, axes, plan), oc)}) : Bool},
  +j: U32, flat: {j == flatten_words(Shape.dimensions(input), coordinate_values(input, ic), 0) : U32},
  decision: Shape.BoolCase<U32.is_lt(j, Shape.count(input))>) -> Result<&2, &2, Error, Mapping<input, axes, plan, i>>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{MappingError{}}
    case Shape.BTrue{equal}:
      Done{Mapped{oi, oc, ic, FIndex{j, equal}, requested, decoded, inverse, flat, roundtrip}}

def map_roundtrip(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Plan<input, axes>, +i: U32,
  oi: FlatIndex<output(input, axes, plan)>, +oc: CheckedCoordinate<output(input, axes, plan)>,
  +ic: CheckedCoordinate<input>,
  requested: {index_value(output(input, axes, plan), oi) == i : U32},
  decoded: {coordinate_values(output(input, axes, plan), oc) ==
    reverse_words(unflatten_words(reverse_words(Shape.dimensions(output(input, axes, plan))), i)) : List<&2, U32>},
  inverse: {Some{coordinate_values(input, ic)} ==
    inverse_coordinates(Shape.dimensions(input), axes, coordinate_values(output(input, axes, plan), oc), 0) : Maybe<&2, List<&2, U32>>},
  decision: Shape.BoolCase<maybe_same(reorder(coordinate_values(input, ic), axes),
    Some{coordinate_values(output(input, axes, plan), oc)})>) -> Result<&2, &2, Error, Mapping<input, axes, plan, i>>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{MappingError{}}
    case Shape.BTrue{equal}:
      +j = flatten_words(Shape.dimensions(input), coordinate_values(input, ic), 0)
      map_flat(input, axes, plan, i, oi, oc, ic, requested, decoded, inverse, equal, j, {==},
        Shape.bool_case(U32.is_lt(j, Shape.count(input))))

def map_input(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Plan<input, axes>, +i: U32,
  oi: FlatIndex<output(input, axes, plan)>, +oc: CheckedCoordinate<output(input, axes, plan)>,
  requested: {index_value(output(input, axes, plan), oi) == i : U32},
  decoded: {coordinate_values(output(input, axes, plan), oc) ==
    reverse_words(unflatten_words(reverse_words(Shape.dimensions(output(input, axes, plan))), i)) : List<&2, U32>},
  +words: List<&2, U32>,
  inverse: {Some{words} == inverse_coordinates(Shape.dimensions(input), axes,
    coordinate_values(output(input, axes, plan), oc), 0) : Maybe<&2, List<&2, U32>>},
  checked: Result<&2, &2, Shape.Error, WordsValid(Shape.dimensions(input), words)>) -> Result<&2, &2, Error, Mapping<input, axes, plan, i>>:
  match checked:
    case Fail{error}:
      Fail{ShapeError{error}}
    case Done{valid}:
      +ic = {CWord{words, valid} : CheckedCoordinate<input>}
      map_roundtrip(input, axes, plan, i, oi, oc, ic, requested, decoded, inverse,
        Shape.bool_case(maybe_same(reorder(words, axes), Some{coordinate_values(output(input, axes, plan), oc)})))

def map_inverse(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Plan<input, axes>, +i: U32,
  oi: FlatIndex<output(input, axes, plan)>, +oc: CheckedCoordinate<output(input, axes, plan)>,
  requested: {index_value(output(input, axes, plan), oi) == i : U32},
  decoded: {coordinate_values(output(input, axes, plan), oc) ==
    reverse_words(unflatten_words(reverse_words(Shape.dimensions(output(input, axes, plan))), i)) : List<&2, U32>},
  result: Maybe<&2, List<&2, U32>>,
  same: {result == inverse_coordinates(Shape.dimensions(input), axes,
    coordinate_values(output(input, axes, plan), oc), 0) : Maybe<&2, List<&2, U32>>}) -> Result<&2, &2, Error, Mapping<input, axes, plan, i>>:
  match result:
    case None{}:
      Fail{MappingError{}}
    case Some{+words}:
      map_input(input, axes, plan, i, oi, oc, requested, decoded, words, same,
        words_check(Shape.dimensions(input), words))

def map_output(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Plan<input, axes>, +i: U32,
  oi: FlatIndex<output(input, axes, plan)>,
  requested: {index_value(output(input, axes, plan), oi) == i : U32},
  +words: List<&2, U32>,
  decoded: {words == reverse_words(unflatten_words(reverse_words(Shape.dimensions(output(input, axes, plan))), i)) : List<&2, U32>},
  checked: Result<&2, &2, Shape.Error, WordsValid(Shape.dimensions(output(input, axes, plan)), words)>) ->
  Result<&2, &2, Error, Mapping<input, axes, plan, i>>:
  match checked:
    case Fail{error}:
      Fail{ShapeError{error}}
    case Done{valid}:
      +oc = {CWord{words, valid} : CheckedCoordinate<output(input, axes, plan)>}
      map_inverse(input, axes, plan, i, oi, oc, requested, decoded,
        inverse_coordinates(Shape.dimensions(input), axes, words, 0), {==})

def map_decided(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Plan<input, axes>, +i: U32,
  decision: Shape.BoolCase<U32.is_lt(i, Shape.count(output(input, axes, plan)))>) ->
  Result<&2, &2, Error, Mapping<input, axes, plan, i>>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{ShapeError{Shape.FlatIndexOutOfBounds{}}}
    case Shape.BTrue{equal}:
      +words = reverse_words(unflatten_words(reverse_words(Shape.dimensions(output(input, axes, plan))), i))
      map_output(input, axes, plan, i, FIndex{i, equal}, {==}, words, {==},
        words_check(Shape.dimensions(output(input, axes, plan)), words))

def map_index(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Plan<input, axes>, +i: U32) ->
  Result<&2, &2, Error, Mapping<input, axes, plan, i>>:
  map_decided(input, axes, plan, i, Shape.bool_case(U32.is_lt(i, Shape.count(output(input, axes, plan)))))
