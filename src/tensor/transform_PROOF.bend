import Base
import ../shape/core.bend as Shape
import ./layout.bend as Layout
import ./transform_LAWS.bend as Laws
import ./core.bend as Core
import ./transforms.bend as Transforms
import ./lifecycle.bend as Lifecycle
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ../storage/PROOF.bend as StorageProof
import ../storage/LAWS.bend as StorageLaws
import ./core_PROOF.bend as CoreProof
import ../storage/arith.bend as Arith
import ../shape/coordinate_proofs.bend as Coordinates
import ../shape/nat_proofs.bend as Nats
import ../storage/route.bend as Route
import ./word_conversion.bend as WordConversion
import ../shape/index_proofs.bend as ShapeIndex
import ../shape/LAWS.bend as ShapeLaws
import ../shape/PROOF.bend as ShapeProof
import ./axis_spec.bend as Axis
import ./axis_algebra.bend as AxisAlgebra
import ./transform_totality.bend as Totality
import ./permutation_mapping.bend as Mapping
import ./allocation_PROOF.bend as AllocationProof
import ./allocation_spec.bend as AllocationSpec
import ./transpose_PROOF.bend as TransposeProof
import ./transpose_spec.bend as TransposeSpec
import ./permute_build_proof.bend as PermuteBuild

def Laws.permutation_checked(input, axes, plan):
  Mapping.permutation_checked(input, axes, plan)

def Laws.permutation_dimensions(input, axes, plan):
  Layout.Permutation{out, valid, ordered, count} = plan
  ordered

def Laws.permutation_count(input, axes, plan):
  Layout.Permutation{out, valid, ordered, count} = plan
  Transforms.elements_equal(input, out, count)

def Laws.mapped_inverse(input, axes, plan, i, mapping):
  Layout.Mapped{oi, oc, ic, index, requested, decoded, inverse, flattened, roundtrip} = mapping
  inverse

def Laws.mapped_flat(input, axes, plan, i, mapping):
  Layout.Mapped{oi, oc, ic, index, requested, decoded, inverse, flattened, roundtrip} = mapping
  flattened

def Laws.mapped_output(input, axes, plan, i, mapping):
  Layout.Mapped{oi, oc, ic, index, requested, decoded, inverse, flattened, roundtrip} = mapping
  decoded

def Laws.mapped_bounded(input, axes, plan, i, mapping):
  Layout.Mapped{oi, oc, ic, index, requested, decoded, inverse, flattened, roundtrip} = mapping
  Layout.FIndex{value, bound} = index
  bound

def bit_equal(a: Bool, b: Bool, equal: {True{} == Bool.not(Bool.xor(a, b)) : Bool}) -> {a == b : Bool}:
  match a b:
    case False{} False{}:
      {==}
    case True{} True{}:
      {==}
    case False{} True{}:
      Empty.absurd({False{} == True{} : Bool}, Lifecycle.false_true(equal))
    case True{} False{}:
      Empty.absurd({True{} == False{} : Bool}, Lifecycle.false_true(equal))

def word_equal(+n: Nat, +a: Word(n), +b: Word(n),
  +equal: {True{} == Layout.word_same(n, a, b) : Bool}) -> {a == b : Word(n)}:
  match n:
    case 0n:
      match a b:
        case WNil{} WNil{}:
          {==}
    case 1n++p:
      WCon{+ab, +at} = a
      WCon{+bb, +bt} = b
      %bit_equal(ab, bb, Mapping.and_left(Bool.not(Bool.xor(ab, bb)), Layout.word_same(p, at, bt), equal)) :
        {WCon{ab, at} == WCon{_, bt} : Word(1n+p)}
      %word_equal(p, at, bt, Mapping.and_right(Bool.not(Bool.xor(ab, bb)), Layout.word_same(p, at, bt), equal)) :
        {WCon{ab, at} == WCon{ab, _} : Word(1n+p)}
      {==}

def word_value_equal(+a: U32, +b: U32, equal: {True{} == Layout.word_value_same(a, b) : Bool}) -> {a == b : U32}:
  match a b:
    case U32{x} U32{y}:
      Equal.cong(Word(32n), U32, w => U32{w}, x, y, word_equal(32n, x, y, equal))

def words_equal(+a: List<&2, U32>, +b: List<&2, U32>,
  +equal: {True{} == Layout.words_same(a, b) : Bool}) -> {a == b : List<&2, U32>}:
  match a b:
    case Nil{} Nil{}:
      {==}
    case +ah <> +at +bh <> +bt:
      %word_value_equal(ah, bh, Mapping.and_left(Layout.word_value_same(ah, bh), Layout.words_same(at, bt), equal)) :
        {ah <> at == _ <> bt : List<&2, U32>}
      %words_equal(at, bt, Mapping.and_right(Layout.word_value_same(ah, bh), Layout.words_same(at, bt), equal)) :
        {ah <> at == ah <> _ : List<&2, U32>}
      {==}
    case Nil{} bh <> bt:
      Empty.absurd({Nil{} == bh <> bt : List<&2, U32>}, Lifecycle.false_true(equal))
    case ah <> at Nil{}:
      Empty.absurd({ah <> at == Nil{} : List<&2, U32>}, Lifecycle.false_true(equal))

def maybe_equal(+a: Maybe<&2, List<&2, U32>>, +b: Maybe<&2, List<&2, U32>>,
  equal: {True{} == Layout.maybe_same(a, b) : Bool}) -> {a == b : Maybe<&2, List<&2, U32>>}:
  match a b:
    case None{} None{}:
      {==}
    case Some{xs} Some{ys}:
      Equal.cong(List<&2, U32>, Maybe<&2, List<&2, U32>>, xs => Some{xs}, xs, ys, words_equal(xs, ys, equal))
    case None{} Some{ys}:
      Empty.absurd({None{} == Some{ys} : Maybe<&2, List<&2, U32>>}, Lifecycle.false_true(equal))
    case Some{xs} None{}:
      Empty.absurd({Some{xs} == None{} : Maybe<&2, List<&2, U32>>}, Lifecycle.false_true(equal))

def Laws.inverse_coordinate_roundtrip(input, axes, plan, i, mapping):
  Layout.Mapped{oi, oc, ic, index, requested, decoded, inverse, flattened, roundtrip} = mapping
  maybe_equal(Layout.reorder(Layout.coordinate_values(input, ic), axes),
    Some{Layout.coordinate_values(Layout.output(input, axes, plan), oc)}, roundtrip)

def Laws.reshape_linear_order(input, target, tensor, index, same):
  Core.TTensor{storage, valid} = tensor
  {==}

def Laws.reshape_target_dimensions(input, target, tensor, same):
  Core.TTensor{storage, valid} = tensor
  {==}

def Laws.copy_input_preservation(depth, array, index, other, balanced):
  Equal.cong(Array<F32>, Maybe<&2, F32>, a => Model.observe(F32, depth, a, Refinement.route(depth, other)),
    Refinement.first(F32, Array.get(F32, array, index)), array,
    Refinement.law_get_retains(F32, depth, array, index, balanced))

def Laws.copy_write_frame(depth, array, index, other, value, balanced):
  StorageLaws.primitive_update_frame(F32, depth, array, index, Refinement.route(depth, other), value, balanced)

def Laws.zero_count_skips_copy(input, axes, plan, source, target, cursor, empty):
  %Equal.sym(Nat, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), 0n, empty) :
    {Transforms.copy_loop(_, input, axes, plan, source, target, cursor) ==
      (source, Done{target}) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>}
  {==}

def Laws.natural_cursor_checked(input, axes, plan, cursor, result):
  match result:
    case Transforms.NaturalCopied{pair, index, exact, candidate}:
      Transforms.CursorExact{exact}
    case Transforms.NaturalError{source}:
      Unit{}

def Laws.copy_source_mapping(input, axes, plan, output_index, mapping, source, certificate):
  Core.Read{tensor, result, exact} = certificate
  %exact : {Core.read_value(input, _) == Core.result_value(result) : Maybe<&2, F32>}
  {==}

def Laws.primitive_source_mapping(input, axes, plan, output_index, mapping, depth, array, balanced):
  Refinement.law_get_refines(F32, depth, array, Layout.source_index(input, axes, plan, output_index, mapping), balanced)

def copy_checked_mapping(-input: Shape.Shape, -axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  +depth: Nat, +length: Nat, array: Array<F32>, +index: U32, source: Core.Tensor<input>, +value: F32,
  balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}, allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(index), length) : Bool},
  -same: {Storage.set_checked(depth, length, array, index, value, allowed, decision, balanced, valid) ==
    Storage.set(Storage.Owned{depth, length, array, balanced, valid}, index, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>},
  accepted: {allowed == True{} : Bool}) ->
  {Transforms.copy_observe(input, axes, plan,
    Transforms.copy_written(input, axes, plan, source => target => (source, Done{target}), source,
      Core.write_result(Layout.output(input, axes, plan), Storage.Owned{depth, length, array, balanced, valid}, exact, index, value,
        Storage.set_checked(depth, length, array, index, value, allowed, decision, balanced, valid), same)), index) == Some{value} : Maybe<&2, F32>}:
  match allowed:
    case False{}:
      Empty.absurd({Transforms.copy_observe(input, axes, plan,
        Transforms.copy_written(input, axes, plan, source => target => (source, Done{target}), source,
          Core.write_result(Layout.output(input, axes, plan), Storage.Owned{depth, length, array, balanced, valid}, exact, index, value,
            Storage.set_checked(depth, length, array, index, value, False{}, decision, balanced, valid), same)), index) == Some{value} : Maybe<&2, F32>},
        Lifecycle.false_true(Equal.sym(Bool, False{}, True{}, accepted)))
    case True{}:
      StorageLaws.primitive_write_read(F32, depth, array, index, value, balanced)

def Laws.copy_value_mapping(input, axes, plan, depth, length, array, index, source, value, balanced, valid, exact, allowed, matched):
  copy_checked_mapping(input, axes, plan, depth, length, array, index, source, value, balanced, valid, exact,
    Nat.is_lt(U32.to_nat(index), length), {==}, {==}, allowed)

def copy_result_mapping(-input: Shape.Shape, -axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  +depth: Nat, +length: Nat, array: Array<F32>, +index: U32, source: Core.Tensor<input>,
  result: Result<&2, &2, Core.Error, F32>, balanced: Model.Balanced(F32, depth, array),
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat},
  allowed: {Nat.is_lt(U32.to_nat(index), length) == True{} : Bool}) ->
  {Transforms.copy_observe(input, axes, plan,
    Transforms.copy_read(input, axes, plan, index, source => target => (source, Done{target}),
      Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, (source, result)), index) ==
    Core.result_value(result) : Maybe<&2, F32>}:
  match result:
    case Fail{error}:
      {==}
    case Done{value}:
      copy_checked_mapping(input, axes, plan, depth, length, array, index, source, value, balanced, valid, exact,
        Nat.is_lt(U32.to_nat(index), length), {==}, {==}, allowed)

def Laws.checked_copy_step_mapping(input, axes, plan, index, mapping, map_exact, source, certificate,
  depth, length, array, balanced, valid, exact, allowed, matched):
  Core.Read{retained, result, read_exact} = certificate
  %map_exact :
    {Transforms.copy_observe(input, axes, plan,
      Transforms.copy_mapped(input, axes, plan, index, source => target => (source, Done{target}), source,
        Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, _), index) ==
      Core.result_value(result) : Maybe<&2, F32>}
  %read_exact :
    {Transforms.copy_observe(input, axes, plan,
      Transforms.copy_read(input, axes, plan, index, source => target => (source, Done{target}),
        Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, _), index) ==
      Core.result_value(result) : Maybe<&2, F32>}
  copy_result_mapping(input, axes, plan, depth, length, array, index, retained, result, balanced, valid, exact, allowed)

def copy_meaning_from_read(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -index: U32,
  -mapping: Layout.Mapping<input, axes, plan, index>, -source: Core.Tensor<input>,
  -target: Core.Tensor<Layout.output(input, axes, plan)>,
  -read: Core.Read<input, source, Layout.source_index(input, axes, plan, index, mapping)>,
  -copied: {Transforms.copy_observe(input, axes, plan, Transforms.copy_step(input, axes, plan, source, target, index), index) ==
    Core.certificate_value(input, source, Layout.source_index(input, axes, plan, index, mapping), read) : Maybe<&2, F32>},
  -read_exact: {Core.read_value(input, Core.read_flat(input, source, Layout.source_index(input, axes, plan, index, mapping))) ==
    Core.certificate_value(input, source, Layout.source_index(input, axes, plan, index, mapping), read) : Maybe<&2, F32>},
  model: Core.SourceRead<input, source, Layout.source_index(input, axes, plan, index, mapping)>) ->
  Transforms.CopyMeaning<input, axes, plan, index, mapping, source, target>:
  Core.SourceRead{meaning} = model
  Transforms.CopyMeaning{Equal.trans(Maybe<&2, F32>,
    Transforms.copy_observe(input, axes, plan, Transforms.copy_step(input, axes, plan, source, target, index), index),
    Core.read_value(input, Core.read_flat(input, source, Layout.source_index(input, axes, plan, index, mapping))),
    Core.checked_observe(input, source, Layout.source_index(input, axes, plan, index, mapping)),
    Equal.trans(Maybe<&2, F32>,
      Transforms.copy_observe(input, axes, plan, Transforms.copy_step(input, axes, plan, source, target, index), index),
      Core.certificate_value(input, source, Layout.source_index(input, axes, plan, index, mapping), read),
      Core.read_value(input, Core.read_flat(input, source, Layout.source_index(input, axes, plan, index, mapping))), copied,
      Equal.sym(Maybe<&2, F32>,
        Core.read_value(input, Core.read_flat(input, source, Layout.source_index(input, axes, plan, index, mapping))),
        Core.certificate_value(input, source, Layout.source_index(input, axes, plan, index, mapping), read), read_exact)), meaning)}

def Laws.checked_copy_source_meaning(input, axes, plan, index, mapping, map_exact, source,
  depth, length, array, balanced, valid, exact, allowed, matched):
  -read = Core.read_certificate(input, source, Layout.source_index(input, axes, plan, index, mapping))
  copy_meaning_from_read(input, axes, plan, index, mapping, source,
    Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, read,
    Laws.checked_copy_step_mapping(input, axes, plan, index, mapping, map_exact, source, read,
      depth, length, array, balanced, valid, exact, allowed, matched),
    Laws.copy_source_mapping(input, axes, plan, index, mapping, source, read),
    Core.source_read_model(input, source, Layout.source_index(input, axes, plan, index, mapping)))

def lt_zero_false(a: Nat) -> {Nat.is_lt(a, 0n) == False{} : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      {==}

def earlier_zero_false(+other: Nat, earlier: {Nat.is_lt(other, 0n) == True{} : Bool}) -> Empty:
  Lifecycle.false_true(Equal.trans(Bool, True{}, Nat.is_lt(other, 0n), False{},
    Equal.sym(Bool, Nat.is_lt(other, 0n), True{}, earlier), lt_zero_false(other)))

def Laws.initial_prefix(input, axes, plan, source, target):
  Transforms.Prefix{other => {==},
    other => earlier => Empty.absurd({U32.to_nat(U32.from_nat(other)) == other : Nat}, earlier_zero_false(other, earlier)),
    other => earlier => Empty.absurd(Core.ObservationChecked(Core.Packed{Layout.output(input, axes, plan), target}, U32.from_nat(other)),
      earlier_zero_false(other, earlier)),
    other => earlier => Empty.absurd({Core.checked_observe(Layout.output(input, axes, plan), target, U32.from_nat(other)) ==
      Transforms.expected_output(input, axes, plan, source, U32.from_nat(other)) : Maybe<&2, F32>}, earlier_zero_false(other, earlier))}

def earlier_lift(-a: Nat, -b: Nat,
  result: Either<&2, &2, {Nat.is_lt(a, b) == True{} : Bool}, {a == b : Nat}>) ->
  Either<&2, &2, {Nat.is_lt(1n+a, 1n+b) == True{} : Bool}, {1n+a == 1n+b : Nat}>:
  match result:
    case Inl{earlier}:
      Inl{earlier}
    case Inr{equal}:
      Inr{Equal.cong(Nat, Nat, n => 1n+n, a, b, equal)}

def earlier_split(+a: Nat, +b: Nat, earlier: {Nat.is_lt(a, 1n+b) == True{} : Bool}) ->
  Either<&2, &2, {Nat.is_lt(a, b) == True{} : Bool}, {a == b : Nat}>:
  match a b:
    case 0n 0n:
      Inr{{==}}
    case 0n 1n+q:
      Inl{{==}}
    case 1n+p 0n:
      Empty.absurd(Either<&2, &2, {Nat.is_lt(1n+p, 0n) == True{} : Bool}, {1n+p == 0n : Nat}>,
        earlier_zero_false(p, earlier))
    case 1n+p 1n+q:
      earlier_lift(p, q, earlier_split(p, q, earlier))

def prefix_words_case(-cursor: Nat, +other: Nat,
  prior: @+j: Nat -> @-earlier: {Nat.is_lt(j, cursor) == True{} : Bool} -> {U32.to_nat(U32.from_nat(j)) == j : Nat},
  current: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  split: Either<&2, &2, {Nat.is_lt(other, cursor) == True{} : Bool}, {other == cursor : Nat}>) ->
  {U32.to_nat(U32.from_nat(other)) == other : Nat}:
  match split:
    case Inl{earlier}:
      prior(other, earlier)
    case Inr{equal}:
      %Equal.sym(Nat, other, cursor, equal) : {U32.to_nat(U32.from_nat(_)) == _ : Nat}
      current

def prefix_written_case(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>,
  -original: Core.Tensor<input>, -before: Core.Tensor<Layout.output(input, axes, plan)>, -after: Core.Tensor<Layout.output(input, axes, plan)>,
  -cursor: Nat, +other: Nat,
  prior: @+j: Nat -> @-earlier: {Nat.is_lt(j, cursor) == True{} : Bool} ->
    {Core.checked_observe(Layout.output(input, axes, plan), before, U32.from_nat(j)) ==
      Transforms.expected_output(input, axes, plan, original, U32.from_nat(j)) : Maybe<&2, F32>},
  current: {Core.checked_observe(Layout.output(input, axes, plan), after, U32.from_nat(cursor)) ==
    Transforms.expected_output(input, axes, plan, original, U32.from_nat(cursor)) : Maybe<&2, F32>},
  frame: @+j: Nat -> @-earlier: {Nat.is_lt(j, cursor) == True{} : Bool} ->
    {Core.checked_observe(Layout.output(input, axes, plan), after, U32.from_nat(j)) ==
      Core.checked_observe(Layout.output(input, axes, plan), before, U32.from_nat(j)) : Maybe<&2, F32>},
  split: Either<&2, &2, {Nat.is_lt(other, cursor) == True{} : Bool}, {other == cursor : Nat}>) ->
  {Core.checked_observe(Layout.output(input, axes, plan), after, U32.from_nat(other)) ==
    Transforms.expected_output(input, axes, plan, original, U32.from_nat(other)) : Maybe<&2, F32>}:
  match split:
    case Inl{+earlier}:
      Equal.trans(Maybe<&2, F32>, Core.checked_observe(Layout.output(input, axes, plan), after, U32.from_nat(other)),
        Core.checked_observe(Layout.output(input, axes, plan), before, U32.from_nat(other)),
        Transforms.expected_output(input, axes, plan, original, U32.from_nat(other)), frame(other, earlier), prior(other, earlier))
    case Inr{equal}:
      %Equal.sym(Nat, other, cursor, equal) : {Core.checked_observe(Layout.output(input, axes, plan), after, U32.from_nat(_)) ==
        Transforms.expected_output(input, axes, plan, original, U32.from_nat(_)) : Maybe<&2, F32>}
      current

def prefix_checked_case(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>,
  -target: Core.Tensor<Layout.output(input, axes, plan)>, -cursor: Nat, +other: Nat,
  prior: @+j: Nat -> @-earlier: {Nat.is_lt(j, cursor) == True{} : Bool} ->
    Core.ObservationChecked(Core.Packed{Layout.output(input, axes, plan), target}, U32.from_nat(j)),
  current: Core.ObservationChecked(Core.Packed{Layout.output(input, axes, plan), target}, U32.from_nat(cursor)),
  split: Either<&2, &2, {Nat.is_lt(other, cursor) == True{} : Bool}, {other == cursor : Nat}>) ->
  Core.ObservationChecked(Core.Packed{Layout.output(input, axes, plan), target}, U32.from_nat(other)):
  match split:
    case Inl{earlier}:
      prior(other, earlier)
    case Inr{equal}:
      %Equal.sym(Nat, other, cursor, equal) : Core.ObservationChecked(Core.Packed{Layout.output(input, axes, plan), target}, U32.from_nat(_))
      current

def Laws.advance_prefix(input, axes, plan, original, before_source, after_source, before_target, after_target, cursor, prefix, advanced):
  Transforms.Prefix{source_prior, words_prior, checked_prior, written_prior} = prefix
  Transforms.Advanced{word_exact, source_kept, current, current_checked, past_checked, frame} = advanced
  Transforms.Prefix{other => Equal.trans(Maybe<&2, F32>, Core.checked_observe(input, after_source, other),
      Core.checked_observe(input, before_source, other), Core.checked_observe(input, original, other),
      source_kept(other), source_prior(other)),
    other => earlier => prefix_words_case(cursor, other, words_prior, word_exact, earlier_split(other, cursor, earlier)),
    other => earlier => prefix_checked_case(input, axes, plan, after_target, cursor, other,
      past_checked, current_checked, earlier_split(other, cursor, earlier)),
    other => earlier => prefix_written_case(input, axes, plan, original, before_target, after_target, cursor, other,
      written_prior, current, frame, earlier_split(other, cursor, earlier))}

def route_allowed(+depth: Nat, +length: Nat, +index: U32,
  +matched: {Storage.index_matches(depth, length, index) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(index), length) == True{} : Bool}:
  +offset = Model.offset(depth, Refinement.route(depth, index))
  %Lifecycle.natural_equal(offset, U32.to_nat(index),
    Mapping.and_left(Nat.is_eq(offset, U32.to_nat(index)), Nat.is_lt(offset, length),
      Equal.sym(Bool, Storage.index_matches(depth, length, index), True{}, matched))) :
    {Nat.is_lt(_, length) == True{} : Bool}
  Equal.sym(Bool, True{}, Nat.is_lt(offset, length),
    Mapping.and_right(Nat.is_eq(offset, U32.to_nat(index)), Nat.is_lt(offset, length),
      Equal.sym(Bool, Storage.index_matches(depth, length, index), True{}, matched)))

def observe_index_equal(+depth: Nat, -length: Nat, -array: Array<F32>, +index: U32, matched: Bool,
  accepted: {matched == True{} : Bool}) ->
  {Lifecycle.observe_index(depth, length, array, index, matched) ==
    Model.observe(F32, depth, array, Refinement.route(depth, index)) : Maybe<&2, F32>}:
  match matched:
    case False{}:
      Empty.absurd({None{} == Model.observe(F32, depth, array, Refinement.route(depth, index)) : Maybe<&2, F32>},
        Lifecycle.false_true(Equal.sym(Bool, False{}, True{}, accepted)))
    case True{}:
      {==}

def observe_allowed_equal(+depth: Nat, +length: Nat, -array: Array<F32>, +index: U32, allowed: Bool,
  accepted: {allowed == True{} : Bool}, matched: {Storage.index_matches(depth, length, index) == True{} : Bool}) ->
  {Lifecycle.observe_allowed(depth, length, array, index, allowed) ==
    Model.observe(F32, depth, array, Refinement.route(depth, index)) : Maybe<&2, F32>}:
  match allowed:
    case False{}:
      Empty.absurd({None{} == Model.observe(F32, depth, array, Refinement.route(depth, index)) : Maybe<&2, F32>},
        Lifecycle.false_true(Equal.sym(Bool, False{}, True{}, accepted)))
    case True{}:
      observe_index_equal(depth, length, array, index, Storage.index_matches(depth, length, index), matched)

type ObserveBridge<-shape: Shape.Shape, -tensor: Core.Tensor<shape>, -index: U32> is Data:
  ObserveBridge{-equal: {Core.checked_observe(shape, tensor, index) == Core.observe(shape, tensor, index) : Maybe<&2, F32>}}

def observe_bridge(-shape: Shape.Shape, tensor: Core.Tensor<shape>, +index: U32,
  checked: Core.ObservationChecked(Core.Packed{shape, tensor}, index)) -> ObserveBridge<shape, tensor, index>:
  Core.TTensor{storage, exact} = tensor
  match storage:
    case Storage.Empty{}:
      ObserveBridge{{==}}
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      ObserveBridge{observe_allowed_equal(depth, length, array, index, Nat.is_lt(U32.to_nat(index), length),
        route_allowed(depth, length, index, checked), checked)}

type ReadInfo<-shape: Shape.Shape, -original: Core.Tensor<shape>, -index: U32> is Type:
  ReadInfo{owner: Core.Tensor<shape>, same: {owner == original : Core.Tensor<shape>},
    -source: Core.SourceRead<shape, original, index>, -kept: Core.SourceReadKeeps<shape, original, index>}

def read_info(-shape: Shape.Shape, original: Core.Tensor<shape>, +index: U32) -> ReadInfo<shape, original, index>:
  ReadInfo{original, {==}, Core.source_read_model(shape, original, index), Core.source_read_keeps(shape, original, index)}

type TargetInfo<-shape: Shape.Shape, -original: Core.Tensor<shape>, -index: U32, -value: F32> is Type:
  TargetInfo{owner: Core.Tensor<shape>, same: {owner == original : Core.Tensor<shape>},
    -model: Core.CheckedUpdateModel<shape, original, index, value>}

def target_info(-shape: Shape.Shape, original: Core.Tensor<shape>, +index: U32, +value: F32) -> TargetInfo<shape, original, index, value>:
  TargetInfo{original, {==}, CoreProof.checked_update_model(shape, original, index, value)}

def read_rebased(-shape: Shape.Shape, -original: Core.Tensor<shape>, -owner: Core.Tensor<shape>, +index: U32,
  same: {owner == original : Core.Tensor<shape>}, certificate: Core.Read<shape, owner, index>) -> Core.Read<shape, original, index>:
  Core.Read{tensor, result, exact} = certificate
  Core.Read{tensor, result, Equal.trans(Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>, (tensor, result),
    Core.read_flat(shape, owner, index), Core.read_flat(shape, original, index), exact,
    Equal.cong(Core.Tensor<shape>, Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>, t => Core.read_flat(shape, t, index), owner, original, same))}

type SourceValue<-shape: Shape.Shape, -original: Core.Tensor<shape>, -index: U32, -value: F32> is Data:
  SourceValue{-equal: {Core.checked_observe(shape, original, index) == Some{value} : Maybe<&2, F32>}}

def source_value(-shape: Shape.Shape, -original: Core.Tensor<shape>, -retained: Core.Tensor<shape>, +index: U32, +value: F32,
  -exact: {(retained, Done{value}) == Core.read_flat(shape, original, index) : Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>},
  certificate: Core.SourceRead<shape, original, index>) -> SourceValue<shape, original, index, value>:
  Core.SourceRead{meaning} = certificate
  SourceValue{Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, original, index),
    Core.read_value(shape, Core.read_flat(shape, original, index)), Some{value},
    Equal.sym(Maybe<&2, F32>, Core.read_value(shape, Core.read_flat(shape, original, index)), Core.checked_observe(shape, original, index), meaning),
    Equal.cong(Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>, Maybe<&2, F32>, pair => Core.read_value(shape, pair),
      Core.read_flat(shape, original, index), (retained, Done{value}),
      Equal.sym(Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>, (retained, Done{value}), Core.read_flat(shape, original, index), exact)))}

type SourceKept<-shape: Shape.Shape, -original: Core.Tensor<shape>, -retained: Core.Tensor<shape>> is Data:
  SourceKept{-kept: Transforms.SameSource(shape, original, retained)}

def source_kept(-shape: Shape.Shape, -original: Core.Tensor<shape>, -retained: Core.Tensor<shape>, +index: U32, -value: F32,
  -exact: {(retained, Done{value}) == Core.read_flat(shape, original, index) : Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>},
  certificate: Core.SourceReadKeeps<shape, original, index>) -> SourceKept<shape, original, retained>:
  Core.SourceReadKeeps{kept} = certificate
  SourceKept{other => Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, retained, other),
    Core.retained_read_observe(shape, Core.read_flat(shape, original, index), other), Core.checked_observe(shape, original, other),
    Equal.cong(Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>, Maybe<&2, F32>, pair => Core.retained_read_observe(shape, pair, other),
      (retained, Done{value}), Core.read_flat(shape, original, index), exact), kept(other))}

type MappedValue<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -index: U32,
  -original: Core.Tensor<input>, -value: F32> is Data:
  MappedValue{-equal: {Some{value} == Transforms.expected_output(input, axes, plan, original, index) : Maybe<&2, F32>}}

def mapped_value(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  -original: Core.Tensor<input>, +mapping: Layout.Mapping<input, axes, plan, index>, +value: F32,
  -exact: {Done{mapping} == Layout.map_index(input, axes, plan, index) : Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>},
  certificate: SourceValue<input, original, Layout.source_index(input, axes, plan, index, mapping), value>) ->
  MappedValue<input, axes, plan, index, original, value>:
  SourceValue{equal} = certificate
  MappedValue{
    %exact : {Some{value} == Transforms.expected_mapped(input, axes, plan, index, original, _) : Maybe<&2, F32>}
    Equal.sym(Maybe<&2, F32>, Core.checked_observe(input, original, Layout.source_index(input, axes, plan, index, mapping)), Some{value}, equal)}

type KnownCheckedUpdate<-shape: Shape.Shape, -original: Core.Tensor<shape>, -output: Core.Tensor<shape>, -index: U32,
  -value: F32, -unit: Unit> is Data:
  KnownCheckedUpdate{-meaning: Core.CheckedUpdateMeaning(shape, original, index, value, (output, Done{unit}))}

def known_checked_update(-shape: Shape.Shape, -original: Core.Tensor<shape>, -output: Core.Tensor<shape>, +index: U32,
  +value: F32, -unit: Unit,
  -exact: {(output, Done{unit}) == Core.write_flat(shape, original, index, value) : Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>},
  certificate: Core.CheckedUpdateModel<shape, original, index, value>) -> KnownCheckedUpdate<shape, original, output, index, value, unit>:
  Core.CheckedUpdateModel{meaning} = certificate
  KnownCheckedUpdate{
    %Equal.sym(Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>, (output, Done{unit}), Core.write_flat(shape, original, index, value), exact) :
      Core.CheckedUpdateMeaning(shape, original, index, value, _)
    meaning}

def checked_update_route(-shape: Shape.Shape, -original: Core.Tensor<shape>, -output: Core.Tensor<shape>, -index: U32,
  -value: F32, -unit: Unit, meaning: Core.CheckedUpdateMeaning(shape, original, index, value, (output, Done{unit}))) ->
  Core.ObservationChecked(Core.Packed{shape, output}, index):
  (checked, rest) = meaning
  checked

def checked_update_value(-shape: Shape.Shape, -original: Core.Tensor<shape>, -output: Core.Tensor<shape>, -index: U32,
  -value: F32, -unit: Unit, meaning: Core.CheckedUpdateMeaning(shape, original, index, value, (output, Done{unit}))) ->
  {Core.checked_observe(shape, output, index) == Some{value} : Maybe<&2, F32>}:
  (checked, rest) = meaning
  (observed, frame) = rest
  observed

def checked_update_frame(-shape: Shape.Shape, -original: Core.Tensor<shape>, -output: Core.Tensor<shape>, -index: U32,
  -value: F32, -unit: Unit, meaning: Core.CheckedUpdateMeaning(shape, original, index, value, (output, Done{unit}))) ->
  @+other: U32 -> @checked: Core.ObservationChecked(Core.Packed{shape, original}, other) ->
    @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
      Core.ObservationChecked(Core.Packed{shape, output}, other) &
        {Core.checked_observe(shape, output, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}:
  (checked, rest) = meaning
  (observed, frame) = rest
  frame

type SuccessfulStep<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -index: U32,
  -original_source: Core.Tensor<input>, -original_target: Core.Tensor<Layout.output(input, axes, plan)>,
  -source: Core.Tensor<input>, -target: Core.Tensor<Layout.output(input, axes, plan)>> is Data:
  SuccessfulStep{-meaning: Transforms.StepMeaning(input, axes, plan, index, original_source, original_target, (source, Done{target}))}

def successful_step(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  -original_source: Core.Tensor<input>, -original_target: Core.Tensor<Layout.output(input, axes, plan)>,
  -source: Core.Tensor<input>, -target: Core.Tensor<Layout.output(input, axes, plan)>, +mapping: Layout.Mapping<input, axes, plan, index>,
  +value: F32, -unit: Unit,
  -map_exact: {Done{mapping} == Layout.map_index(input, axes, plan, index) : Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>},
  read_value: SourceValue<input, original_source, Layout.source_index(input, axes, plan, index, mapping), value>,
  read_kept: SourceKept<input, original_source, source>,
  update: KnownCheckedUpdate<Layout.output(input, axes, plan), original_target, target, index, value, unit>) ->
  SuccessfulStep<input, axes, plan, index, original_source, original_target, source, target>:
  SourceValue{read_equal} = read_value
  SourceKept{kept} = read_kept
  KnownCheckedUpdate{meaning} = update
  SuccessfulStep{(kept,
    (checked_update_route(Layout.output(input, axes, plan), original_target, target, index, value, unit, meaning),
      (Equal.trans(Maybe<&2, F32>, Core.checked_observe(Layout.output(input, axes, plan), target, index), Some{value},
        Transforms.expected_output(input, axes, plan, original_source, index),
        checked_update_value(Layout.output(input, axes, plan), original_target, target, index, value, unit, meaning),
        {\{MappedValue: equal => equal} : MappedValue<input, axes, plan, index, original_source, value> ->
          {Some{value} == Transforms.expected_output(input, axes, plan, original_source, index) : Maybe<&2, F32>}}(
            mapped_value(input, axes, plan, index, original_source, mapping, value, map_exact, SourceValue{read_equal}))),
        checked_update_frame(Layout.output(input, axes, plan), original_target, target, index, value, unit, meaning))))}

def step_write_certificate(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  -original_source: Core.Tensor<input>, -original_target: Core.Tensor<Layout.output(input, axes, plan)>,
  +mapping: Layout.Mapping<input, axes, plan, index>, +value: F32,
  -map_exact: {Done{mapping} == Layout.map_index(input, axes, plan, index) : Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>},
  certificate: Core.Write<Layout.output(input, axes, plan), original_target, index, value>,
  retained: Core.Tensor<input>,
  lift: @-pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>> ->
    {pair == Transforms.copy_written(input, axes, plan, source => target => (source, Done{target}), retained,
      Core.write_flat(Layout.output(input, axes, plan), original_target, index, value)) :
        Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>} ->
    {pair == Transforms.copy_step(input, axes, plan, original_source, original_target, index) :
      Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>},
  -read_exact: {(retained, Done{value}) == Core.read_flat(input, original_source, Layout.source_index(input, axes, plan, index, mapping)) :
    Core.Tensor<input> & Result<&2, &2, Core.Error, F32>},
  -read_model: Core.SourceRead<input, original_source, Layout.source_index(input, axes, plan, index, mapping)>,
  -keep_model: Core.SourceReadKeeps<input, original_source, Layout.source_index(input, axes, plan, index, mapping)>,
  -write_model: Core.CheckedUpdateModel<Layout.output(input, axes, plan), original_target, index, value>) ->
  Transforms.StepCertificate<input, axes, plan, index, original_source, original_target>:
  Core.Write{target, result, exact} = certificate
  match result:
    case Fail{error}:
      Transforms.StepCertificate{(retained, Fail{Transforms.TensorError{error}}),
        lift((retained, Fail{Transforms.TensorError{error}}),
          Equal.cong(Core.Tensor<Layout.output(input, axes, plan)> & Result<&2, &2, Core.Error, Unit>,
            Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
            pair => Transforms.copy_written(input, axes, plan, source => target => (source, Done{target}), retained, pair),
            (target, Fail{error}), Core.write_flat(Layout.output(input, axes, plan), original_target, index, value), exact)), Unit{}}
    case Done{unit}:
      Transforms.StepCertificate{(retained, Done{target}),
        lift((retained, Done{target}),
          Equal.cong(Core.Tensor<Layout.output(input, axes, plan)> & Result<&2, &2, Core.Error, Unit>,
            Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
            pair => Transforms.copy_written(input, axes, plan, source => target => (source, Done{target}), retained, pair),
            (target, Done{unit}), Core.write_flat(Layout.output(input, axes, plan), original_target, index, value), exact)),
        {\{SuccessfulStep: meaning => meaning} : SuccessfulStep<input, axes, plan, index, original_source, original_target, retained, target> ->
          Transforms.StepMeaning(input, axes, plan, index, original_source, original_target, (retained, Done{target}))}(
            successful_step(input, axes, plan, index, original_source, original_target, retained, target, mapping, value, unit, map_exact,
              source_value(input, original_source, retained, Layout.source_index(input, axes, plan, index, mapping), value, read_exact, read_model),
              source_kept(input, original_source, retained, Layout.source_index(input, axes, plan, index, mapping), value, read_exact, keep_model),
              known_checked_update(Layout.output(input, axes, plan), original_target, target, index, value, unit, exact, write_model)))}

def step_target_info(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  -original_source: Core.Tensor<input>, -original_target: Core.Tensor<Layout.output(input, axes, plan)>,
  +mapping: Layout.Mapping<input, axes, plan, index>, +value: F32,
  -map_exact: {Done{mapping} == Layout.map_index(input, axes, plan, index) : Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>},
  target_info: TargetInfo<Layout.output(input, axes, plan), original_target, index, value>, retained: Core.Tensor<input>,
  lift: @-pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>> ->
    {pair == Transforms.copy_written(input, axes, plan, source => target => (source, Done{target}), retained,
      Core.write_flat(Layout.output(input, axes, plan), original_target, index, value)) :
        Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>} ->
    {pair == Transforms.copy_step(input, axes, plan, original_source, original_target, index) :
      Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>},
  -read_exact: {(retained, Done{value}) == Core.read_flat(input, original_source, Layout.source_index(input, axes, plan, index, mapping)) :
    Core.Tensor<input> & Result<&2, &2, Core.Error, F32>}) -> Transforms.StepCertificate<input, axes, plan, index, original_source, original_target>:
  TargetInfo{owner, same, model} = target_info
  step_write_certificate(input, axes, plan, index, original_source, original_target, mapping, value, map_exact,
    CoreProof.write_original(Layout.output(input, axes, plan), original_target, owner, index, value, same,
      Core.write_certificate(Layout.output(input, axes, plan), owner, index, value)), retained, lift, read_exact,
    Core.source_read_model(input, original_source, Layout.source_index(input, axes, plan, index, mapping)),
    Core.source_read_keeps(input, original_source, Layout.source_index(input, axes, plan, index, mapping)), model)

def step_read_certificate(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  -original_source: Core.Tensor<input>, original_target: Core.Tensor<Layout.output(input, axes, plan)>,
  +mapping: Layout.Mapping<input, axes, plan, index>,
  -map_exact: {Done{mapping} == Layout.map_index(input, axes, plan, index) : Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>},
  lift: @-pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>> ->
    {pair == Transforms.copy_read(input, axes, plan, index, source => target => (source, Done{target}), original_target,
      Core.read_flat(input, original_source, Layout.source_index(input, axes, plan, index, mapping))) :
        Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>} ->
    {pair == Transforms.copy_step(input, axes, plan, original_source, original_target, index) :
      Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>},
  certificate: Core.Read<input, original_source, Layout.source_index(input, axes, plan, index, mapping)>) ->
  Transforms.StepCertificate<input, axes, plan, index, original_source, original_target>:
  Core.Read{retained, result, exact} = certificate
  match result:
    case Fail{error}:
      Transforms.StepCertificate{(retained, Fail{Transforms.TensorError{error}}),
        lift((retained, Fail{Transforms.TensorError{error}}),
          Equal.cong(Core.Tensor<input> & Result<&2, &2, Core.Error, F32>,
            Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
            pair => Transforms.copy_read(input, axes, plan, index, source => target => (source, Done{target}), original_target, pair),
            (retained, Fail{error}), Core.read_flat(input, original_source, Layout.source_index(input, axes, plan, index, mapping)), exact)), Unit{}}
    case Done{+value}:
      step_target_info(input, axes, plan, index, original_source, original_target, mapping, value, map_exact,
        target_info(Layout.output(input, axes, plan), original_target, index, value), retained,
        pair => equality => lift(pair,
          Equal.trans(Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
            pair, Transforms.copy_read(input, axes, plan, index, source => target => (source, Done{target}), original_target, (retained, Done{value})),
            Transforms.copy_read(input, axes, plan, index, source => target => (source, Done{target}), original_target,
              Core.read_flat(input, original_source, Layout.source_index(input, axes, plan, index, mapping))), equality,
            Equal.cong(Core.Tensor<input> & Result<&2, &2, Core.Error, F32>,
              Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
              read => Transforms.copy_read(input, axes, plan, index, source => target => (source, Done{target}), original_target, read),
              (retained, Done{value}), Core.read_flat(input, original_source, Layout.source_index(input, axes, plan, index, mapping)), exact))), exact)

def step_mapped_certificate(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  source: Core.Tensor<input>, target: Core.Tensor<Layout.output(input, axes, plan)>,
  result: Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>,
  exact: {result == Layout.map_index(input, axes, plan, index) : Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>}) ->
  Transforms.StepCertificate<input, axes, plan, index, source, target>:
  match result:
    case Fail{error}:
      Transforms.StepCertificate{(source, Fail{Transforms.LayoutError{error}}),
        Equal.cong(Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>,
          Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
          mapped => Transforms.copy_mapped(input, axes, plan, index, source => target => (source, Done{target}), source, target, mapped),
          Fail{error}, Layout.map_index(input, axes, plan, index), exact), Unit{}}
    case Done{+mapping}:
      step_read_certificate(input, axes, plan, index, source, target, mapping, exact,
        pair => equality => Equal.trans(Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
          pair, Transforms.copy_mapped(input, axes, plan, index, source => target => (source, Done{target}), source, target, Done{mapping}),
          Transforms.copy_step(input, axes, plan, source, target, index), equality,
          Equal.cong(Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>,
            Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
            mapped => Transforms.copy_mapped(input, axes, plan, index, source => target => (source, Done{target}), source, target, mapped),
            Done{mapping}, Layout.map_index(input, axes, plan, index), exact)),
        Core.read_certificate(input, source, Layout.source_index(input, axes, plan, index, mapping)))

def step_certificate(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  source: Core.Tensor<input>, target: Core.Tensor<Layout.output(input, axes, plan)>) ->
  Transforms.StepCertificate<input, axes, plan, index, source, target>:
  step_mapped_certificate(input, axes, plan, index, source, target, Layout.map_index(input, axes, plan, index), {==})

def natural_step_lift(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +cursor: Nat, +index: U32,
  -source: Core.Tensor<input>, -target: Core.Tensor<Layout.output(input, axes, plan)>,
  -candidate: {index == U32.from_nat(cursor) : U32},
  -conversion: {True{} == Nat.is_eq(U32.to_nat(index), cursor) : Bool},
  same: {Transforms.natural_decided(input, axes, plan, cursor, index, source, target, candidate, Shape.BTrue{conversion}) ==
    Transforms.natural_step(input, axes, plan, cursor, source, target) : Transforms.NaturalCopy<input, axes, plan, cursor>},
  certificate: Transforms.StepCertificate<input, axes, plan, index, source, target>) ->
  Transforms.NaturalCertificate<input, axes, plan, cursor, source, target>:
  Transforms.StepCertificate{pair, exact, meaning} = certificate
  Transforms.NaturalCertificate{Transforms.NaturalCopied{pair, index, Lifecycle.natural_equal(U32.to_nat(index), cursor, conversion), candidate},
    Equal.trans(Transforms.NaturalCopy<input, axes, plan, cursor>,
      Transforms.NaturalCopied{pair, index, Lifecycle.natural_equal(U32.to_nat(index), cursor, conversion), candidate},
      Transforms.natural_decided(input, axes, plan, cursor, index, source, target, candidate, Shape.BTrue{conversion}),
      Transforms.natural_step(input, axes, plan, cursor, source, target),
      Equal.cong(Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
        Transforms.NaturalCopy<input, axes, plan, cursor>,
        result => Transforms.natural_result(input, axes, plan, cursor, index, Lifecycle.natural_equal(U32.to_nat(index), cursor, conversion), candidate, result),
        pair, Transforms.copy_step(input, axes, plan, source, target, index), exact), same), meaning}

def natural_decided_certificate(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +cursor: Nat, +index: U32,
  source: Core.Tensor<input>, target: Core.Tensor<Layout.output(input, axes, plan)>,
  -candidate: {index == U32.from_nat(cursor) : U32},
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>,
  same: {Transforms.natural_decided(input, axes, plan, cursor, index, source, target, candidate, decision) ==
    Transforms.natural_step(input, axes, plan, cursor, source, target) : Transforms.NaturalCopy<input, axes, plan, cursor>}) ->
  Transforms.NaturalCertificate<input, axes, plan, cursor, source, target>:
  match decision:
    case Shape.BFalse{equal}:
      Transforms.NaturalCertificate{Transforms.NaturalError{source}, same, Unit{}}
    case Shape.BTrue{equal}:
      natural_step_lift(input, axes, plan, cursor, index, source, target, candidate, equal, same, step_certificate(input, axes, plan, index, source, target))

def natural_certificate(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +cursor: Nat,
  source: Core.Tensor<input>, target: Core.Tensor<Layout.output(input, axes, plan)>) ->
  Transforms.NaturalCertificate<input, axes, plan, cursor, source, target>:
  +index = U32.from_nat(cursor)
  natural_decided_certificate(input, axes, plan, cursor, index, source, target, {==}, Shape.bool_case(Nat.is_eq(U32.to_nat(index), cursor)), {==})

def words_same_refl(+values: List<&2, U32>) -> {Layout.words_same(values, values) == True{} : Bool}:
  Mapping.words_same_refl(values)

def cursor_word_exact(+shape: Shape.Shape, +cursor: Nat, within: {Nat.is_le(cursor, Shape.elements(shape)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat}:
  Mapping.cursor_word_exact(shape, cursor, within)

def decoded_coordinate(+shape: Shape.Shape, +index: U32,
  +inside: {True{} == Nat.is_lt(U32.to_nat(index), Shape.elements(shape)) : Bool}) -> Shape.Coordinate(Shape.extents(shape)):
  Mapping.decoded_coordinate(shape, index, inside)

def natural_lt_different(+a: Nat, +b: Nat, earlier: {Nat.is_lt(a, b) == True{} : Bool}) ->
  {Nat.is_eq(b, a) == False{} : Bool}:
  match a b:
    case 0n 0n:
      Empty.absurd({True{} == False{} : Bool}, Lifecycle.false_true(Equal.sym(Bool, False{}, True{}, earlier)))
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      natural_lt_different(p, q, earlier)

def natural_words_different(+cursor: Nat, +other: Nat,
  converted: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  prior: {U32.to_nat(U32.from_nat(other)) == other : Nat},
  earlier: {Nat.is_lt(other, cursor) == True{} : Bool}) ->
  {Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), U32.to_nat(U32.from_nat(other))) == False{} : Bool}:
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(cursor)), cursor, converted) : {Nat.is_eq(_, U32.to_nat(U32.from_nat(other))) == False{} : Bool}
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(other)), other, prior) : {Nat.is_eq(cursor, _) == False{} : Bool}
  natural_lt_different(other, cursor, earlier)

type ExpectedSame<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -original: Core.Tensor<input>,
  -source: Core.Tensor<input>, -index: U32> is Data:
  ExpectedSame{-equal: {Transforms.expected_output(input, axes, plan, source, index) ==
    Transforms.expected_output(input, axes, plan, original, index) : Maybe<&2, F32>}}

def expected_same_mapped(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -index: U32,
  -original: Core.Tensor<input>, -source: Core.Tensor<input>,
  kept: Transforms.SameSource(input, original, source),
  result: Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>) ->
  {Transforms.expected_mapped(input, axes, plan, index, source, result) ==
    Transforms.expected_mapped(input, axes, plan, index, original, result) : Maybe<&2, F32>}:
  match result:
    case Fail{error}:
      {==}
    case Done{mapping}:
      kept(Layout.source_index(input, axes, plan, index, mapping))

def expected_same(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  -original: Core.Tensor<input>, -source: Core.Tensor<input>, +index: U32,
  -kept: Transforms.SameSource(input, original, source)) -> ExpectedSame<input, axes, plan, original, source, index>:
  ExpectedSame{expected_same_mapped(input, axes, plan, index, original, source, kept, Layout.map_index(input, axes, plan, index))}

type PrefixSource<-input: Shape.Shape, -original: Core.Tensor<input>, -source: Core.Tensor<input>> is Data:
  PrefixSource{-kept: Transforms.SameSource(input, original, source)}

def prefix_source(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -original: Core.Tensor<input>,
  -source: Core.Tensor<input>, -target: Core.Tensor<Layout.output(input, axes, plan)>, -cursor: Nat,
  prefix: Transforms.Prefix<input, axes, plan, original, source, target, cursor>) -> PrefixSource<input, original, source>:
  Transforms.Prefix{kept, words, checked, written} = prefix
  PrefixSource{kept}

type PrefixWords<-cursor: Nat> is Data:
  PrefixWords{-words: @+other: Nat -> @-earlier: {Nat.is_lt(other, cursor) == True{} : Bool} -> {U32.to_nat(U32.from_nat(other)) == other : Nat}}

def prefix_word(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -original: Core.Tensor<input>,
  -source: Core.Tensor<input>, -target: Core.Tensor<Layout.output(input, axes, plan)>, -cursor: Nat,
  prefix: Transforms.Prefix<input, axes, plan, original, source, target, cursor>) -> PrefixWords<cursor>:
  Transforms.Prefix{kept, words, checked, written} = prefix
  PrefixWords{words}

type PrefixRoutes<-shape: Shape.Shape, -target: Core.Tensor<shape>, -cursor: Nat> is Data:
  PrefixRoutes{-checked: @+other: Nat -> @-earlier: {Nat.is_lt(other, cursor) == True{} : Bool} ->
    Core.ObservationChecked(Core.Packed{shape, target}, U32.from_nat(other))}

def prefix_route(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -original: Core.Tensor<input>,
  -source: Core.Tensor<input>, -target: Core.Tensor<Layout.output(input, axes, plan)>, -cursor: Nat,
  prefix: Transforms.Prefix<input, axes, plan, original, source, target, cursor>) -> PrefixRoutes<Layout.output(input, axes, plan), target, cursor>:
  Transforms.Prefix{kept, words, checked, written} = prefix
  PrefixRoutes{checked}

def step_source(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -index: U32,
  -before_source: Core.Tensor<input>, -before_target: Core.Tensor<Layout.output(input, axes, plan)>,
  -after_source: Core.Tensor<input>, -after_target: Core.Tensor<Layout.output(input, axes, plan)>,
  meaning: Transforms.StepMeaning(input, axes, plan, index, before_source, before_target, (after_source, Done{after_target}))) ->
  Transforms.SameSource(input, before_source, after_source):
  (kept, rest) = meaning
  kept

def step_route(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -index: U32,
  -before_source: Core.Tensor<input>, -before_target: Core.Tensor<Layout.output(input, axes, plan)>,
  -after_source: Core.Tensor<input>, -after_target: Core.Tensor<Layout.output(input, axes, plan)>,
  meaning: Transforms.StepMeaning(input, axes, plan, index, before_source, before_target, (after_source, Done{after_target}))) ->
  Core.ObservationChecked(Core.Packed{Layout.output(input, axes, plan), after_target}, index):
  (kept, rest) = meaning
  (checked, values) = rest
  checked

def step_value(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -index: U32,
  -before_source: Core.Tensor<input>, -before_target: Core.Tensor<Layout.output(input, axes, plan)>,
  -after_source: Core.Tensor<input>, -after_target: Core.Tensor<Layout.output(input, axes, plan)>,
  meaning: Transforms.StepMeaning(input, axes, plan, index, before_source, before_target, (after_source, Done{after_target}))) ->
  {Core.checked_observe(Layout.output(input, axes, plan), after_target, index) ==
    Transforms.expected_output(input, axes, plan, before_source, index) : Maybe<&2, F32>}:
  (kept, rest) = meaning
  (checked, values) = rest
  (observed, frame) = values
  observed

def step_frame(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -index: U32,
  -before_source: Core.Tensor<input>, -before_target: Core.Tensor<Layout.output(input, axes, plan)>,
  -after_source: Core.Tensor<input>, -after_target: Core.Tensor<Layout.output(input, axes, plan)>,
  meaning: Transforms.StepMeaning(input, axes, plan, index, before_source, before_target, (after_source, Done{after_target}))) ->
  @+other: U32 -> @checked: Core.ObservationChecked(Core.Packed{Layout.output(input, axes, plan), before_target}, other) ->
    @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
      Core.ObservationChecked(Core.Packed{Layout.output(input, axes, plan), after_target}, other) &
        {Core.checked_observe(Layout.output(input, axes, plan), after_target, other) ==
          Core.checked_observe(Layout.output(input, axes, plan), before_target, other) : Maybe<&2, F32>}:
  (kept, rest) = meaning
  (checked, values) = rest
  (observed, frame) = values
  frame

type CanonicalStep<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -cursor: Nat,
  -before_source: Core.Tensor<input>, -before_target: Core.Tensor<Layout.output(input, axes, plan)>,
  -after_source: Core.Tensor<input>, -after_target: Core.Tensor<Layout.output(input, axes, plan)>> is Data:
  CanonicalStep{-converted: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
    -meaning: Transforms.StepMeaning(input, axes, plan, U32.from_nat(cursor), before_source, before_target, (after_source, Done{after_target}))}

def canonical_step(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, +cursor: Nat, -index: U32,
  -before_source: Core.Tensor<input>, -before_target: Core.Tensor<Layout.output(input, axes, plan)>,
  -after_source: Core.Tensor<input>, -after_target: Core.Tensor<Layout.output(input, axes, plan)>,
  -candidate: {index == U32.from_nat(cursor) : U32}, -converted: {U32.to_nat(index) == cursor : Nat},
  -meaning: Transforms.StepMeaning(input, axes, plan, index, before_source, before_target, (after_source, Done{after_target}))) ->
  CanonicalStep<input, axes, plan, cursor, before_source, before_target, after_source, after_target>:
  CanonicalStep{Equal.trans(Nat, U32.to_nat(U32.from_nat(cursor)), U32.to_nat(index), cursor,
      Equal.cong(U32, Nat, U32.to_nat, U32.from_nat(cursor), index, Equal.sym(U32, index, U32.from_nat(cursor), candidate)), converted),
    %candidate : Transforms.StepMeaning(input, axes, plan, _, before_source, before_target, (after_source, Done{after_target}))
    meaning}

type PastFrame<-shape: Shape.Shape, -before: Core.Tensor<shape>, -after: Core.Tensor<shape>, -other: Nat> is Data:
  PastFrame{-checked: Core.ObservationChecked(Core.Packed{shape, after}, U32.from_nat(other)),
    -equal: {Core.checked_observe(shape, after, U32.from_nat(other)) == Core.checked_observe(shape, before, U32.from_nat(other)) : Maybe<&2, F32>}}

def past_frame(-shape: Shape.Shape, -before: Core.Tensor<shape>, -after: Core.Tensor<shape>, +cursor: Nat, +other: Nat,
  -converted: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat}, -prior_word: {U32.to_nat(U32.from_nat(other)) == other : Nat},
  -prior_checked: Core.ObservationChecked(Core.Packed{shape, before}, U32.from_nat(other)),
  -earlier: {Nat.is_lt(other, cursor) == True{} : Bool},
  -frame: @+j: U32 -> @checked: Core.ObservationChecked(Core.Packed{shape, before}, j) ->
    @different: {Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), U32.to_nat(j)) == False{} : Bool} ->
      Core.ObservationChecked(Core.Packed{shape, after}, j) &
        {Core.checked_observe(shape, after, j) == Core.checked_observe(shape, before, j) : Maybe<&2, F32>}) ->
  PastFrame<shape, before, after, other>:
  PastFrame{
    Pair.fst(Core.ObservationChecked(Core.Packed{shape, after}, U32.from_nat(other)),
      {Core.checked_observe(shape, after, U32.from_nat(other)) == Core.checked_observe(shape, before, U32.from_nat(other)) : Maybe<&2, F32>},
      frame(U32.from_nat(other))(prior_checked)(natural_words_different(cursor, other, converted, prior_word, earlier))),
    Pair.snd(Core.ObservationChecked(Core.Packed{shape, after}, U32.from_nat(other)),
      {Core.checked_observe(shape, after, U32.from_nat(other)) == Core.checked_observe(shape, before, U32.from_nat(other)) : Maybe<&2, F32>},
      frame(U32.from_nat(other))(prior_checked)(natural_words_different(cursor, other, converted, prior_word, earlier)))}

def actual_advanced(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +cursor: Nat,
  -original: Core.Tensor<input>, -before_source: Core.Tensor<input>, -after_source: Core.Tensor<input>,
  -before_target: Core.Tensor<Layout.output(input, axes, plan)>, -after_target: Core.Tensor<Layout.output(input, axes, plan)>,
  prefix: Transforms.Prefix<input, axes, plan, original, before_source, before_target, cursor>,
  canonical: CanonicalStep<input, axes, plan, cursor, before_source, before_target, after_source, after_target>) ->
  Transforms.Advanced<input, axes, plan, original, before_source, after_source, before_target, after_target, cursor>:
  Transforms.Prefix{source_prior, words_prior, checked_prior, written_prior} = prefix
  CanonicalStep{converted, meaning} = canonical
  Transforms.Advanced{converted,
    step_source(input, axes, plan, U32.from_nat(cursor), before_source, before_target, after_source, after_target, meaning),
    Equal.trans(Maybe<&2, F32>, Core.checked_observe(Layout.output(input, axes, plan), after_target, U32.from_nat(cursor)),
      Transforms.expected_output(input, axes, plan, before_source, U32.from_nat(cursor)),
      Transforms.expected_output(input, axes, plan, original, U32.from_nat(cursor)),
      step_value(input, axes, plan, U32.from_nat(cursor), before_source, before_target, after_source, after_target, meaning),
      {\{ExpectedSame: equal => equal} : ExpectedSame<input, axes, plan, original, before_source, U32.from_nat(cursor)> ->
        {Transforms.expected_output(input, axes, plan, before_source, U32.from_nat(cursor)) ==
          Transforms.expected_output(input, axes, plan, original, U32.from_nat(cursor)) : Maybe<&2, F32>}}(
            expected_same(input, axes, plan, original, before_source, U32.from_nat(cursor), source_prior))),
    step_route(input, axes, plan, U32.from_nat(cursor), before_source, before_target, after_source, after_target, meaning),
    other => earlier => {\{PastFrame: checked => equal => checked} :
      PastFrame<Layout.output(input, axes, plan), before_target, after_target, other> ->
        Core.ObservationChecked(Core.Packed{Layout.output(input, axes, plan), after_target}, U32.from_nat(other))}(
          past_frame(Layout.output(input, axes, plan), before_target, after_target, cursor, other, converted,
            words_prior(other, earlier), checked_prior(other, earlier), earlier,
            step_frame(input, axes, plan, U32.from_nat(cursor), before_source, before_target, after_source, after_target, meaning))),
    other => earlier => {\{PastFrame: checked => equal => equal} :
      PastFrame<Layout.output(input, axes, plan), before_target, after_target, other> ->
        {Core.checked_observe(Layout.output(input, axes, plan), after_target, U32.from_nat(other)) ==
          Core.checked_observe(Layout.output(input, axes, plan), before_target, U32.from_nat(other)) : Maybe<&2, F32>}}(
          past_frame(Layout.output(input, axes, plan), before_target, after_target, cursor, other, converted,
            words_prior(other, earlier), checked_prior(other, earlier), earlier,
            step_frame(input, axes, plan, U32.from_nat(cursor), before_source, before_target, after_source, after_target, meaning)))}

def add_successor_right(+left: Nat, -right: Nat) -> {Nat.add(left, 1n+right) == 1n+Nat.add(left, right) : Nat}:
  match left:
    case 0n:
      {==}
    case 1n+p:
      Equal.cong(Nat, Nat, n => 1n+n, Nat.add(p, 1n+right), 1n+Nat.add(p, right), add_successor_right(p, right))

def loop_continue(+fuel: Nat, +input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +cursor: Nat,
  -original: Core.Tensor<input>, -before_source: Core.Tensor<input>, -before_target: Core.Tensor<Layout.output(input, axes, plan)>,
  -after_source: Core.Tensor<input>, -after_target: Core.Tensor<Layout.output(input, axes, plan)>, -index: U32,
  -converted: {U32.to_nat(index) == cursor : Nat}, -candidate: {index == U32.from_nat(cursor) : U32},
  natural_exact: {Transforms.NaturalCopied{(after_source, Done{after_target}), index, converted, candidate} ==
    Transforms.natural_step(input, axes, plan, cursor, before_source, before_target) : Transforms.NaturalCopy<input, axes, plan, cursor>},
  certificate: Transforms.LoopCertificate<input, axes, plan, original, fuel, 1n+cursor, after_source, after_target>) ->
  Transforms.LoopCertificate<input, axes, plan, original, 1n+fuel, cursor, before_source, before_target>:
  Transforms.LoopCertificate{pair, exact, meaning} = certificate
  Transforms.LoopCertificate{pair,
    Equal.trans(Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>, pair,
      Transforms.copy_loop(fuel, input, axes, plan, after_source, after_target, 1n+cursor),
      Transforms.copy_loop(1n+fuel, input, axes, plan, before_source, before_target, cursor), exact,
      Equal.cong(Transforms.NaturalCopy<input, axes, plan, cursor>,
        Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
        result => Transforms.natural_next(input, axes, plan, cursor,
          source => target => Transforms.copy_loop(fuel, input, axes, plan, source, target, 1n+cursor), result),
        Transforms.NaturalCopied{(after_source, Done{after_target}), index, converted, candidate},
        Transforms.natural_step(input, axes, plan, cursor, before_source, before_target), natural_exact)),
    %add_successor_right(fuel, cursor) : Transforms.LoopMeaning(input, axes, plan, original, _, pair)
    meaning}

def loop_after_pair(+fuel: Nat, +input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +cursor: Nat,
  -original: Core.Tensor<input>, -before_source: Core.Tensor<input>, -before_target: Core.Tensor<Layout.output(input, axes, plan)>,
  -index: U32, -converted: {U32.to_nat(index) == cursor : Nat}, -candidate: {index == U32.from_nat(cursor) : U32},
  -prefix: Transforms.Prefix<input, axes, plan, original, before_source, before_target, cursor>,
  pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
  natural_exact: {Transforms.NaturalCopied{pair, index, converted, candidate} ==
    Transforms.natural_step(input, axes, plan, cursor, before_source, before_target) : Transforms.NaturalCopy<input, axes, plan, cursor>},
  -meaning: Transforms.StepMeaning(input, axes, plan, index, before_source, before_target, pair),
  rec: @source: Core.Tensor<input> -> @target: Core.Tensor<Layout.output(input, axes, plan)> ->
    @-prefix: Transforms.Prefix<input, axes, plan, original, source, target, 1n+cursor> ->
      Transforms.LoopCertificate<input, axes, plan, original, fuel, 1n+cursor, source, target>) ->
  Transforms.LoopCertificate<input, axes, plan, original, 1n+fuel, cursor, before_source, before_target>:
  (source, result) = pair
  match result:
    case Fail{error}:
      Transforms.LoopCertificate{(source, Fail{error}),
        Equal.cong(Transforms.NaturalCopy<input, axes, plan, cursor>,
          Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
          step => Transforms.natural_next(input, axes, plan, cursor,
            source => target => Transforms.copy_loop(fuel, input, axes, plan, source, target, 1n+cursor), step),
          Transforms.NaturalCopied{(source, Fail{error}), index, converted, candidate},
          Transforms.natural_step(input, axes, plan, cursor, before_source, before_target), natural_exact), Unit{}}
    case Done{target}:
      loop_continue(fuel, input, axes, plan, cursor, original, before_source, before_target, source, target, index, converted, candidate,
        natural_exact, rec(source)(target)(
          Laws.advance_prefix(input, axes, plan, original, before_source, source, before_target, target, cursor, prefix,
            actual_advanced(input, axes, plan, cursor, original, before_source, source, before_target, target, prefix,
              canonical_step(input, axes, plan, cursor, index, before_source, before_target, source, target, candidate, converted, meaning)))))

def loop_after_natural(+fuel: Nat, +input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +cursor: Nat,
  -original: Core.Tensor<input>, -before_source: Core.Tensor<input>, -before_target: Core.Tensor<Layout.output(input, axes, plan)>,
  -prefix: Transforms.Prefix<input, axes, plan, original, before_source, before_target, cursor>,
  certificate: Transforms.NaturalCertificate<input, axes, plan, cursor, before_source, before_target>,
  rec: @source: Core.Tensor<input> -> @target: Core.Tensor<Layout.output(input, axes, plan)> ->
    @-prefix: Transforms.Prefix<input, axes, plan, original, source, target, 1n+cursor> ->
      Transforms.LoopCertificate<input, axes, plan, original, fuel, 1n+cursor, source, target>) ->
  Transforms.LoopCertificate<input, axes, plan, original, 1n+fuel, cursor, before_source, before_target>:
  Transforms.NaturalCertificate{result, exact, meaning} = certificate
  match result:
    case Transforms.NaturalError{source}:
      Transforms.LoopCertificate{(source, Fail{Transforms.LayoutError{Layout.MappingError{}}}),
        Equal.cong(Transforms.NaturalCopy<input, axes, plan, cursor>,
          Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
          step => Transforms.natural_next(input, axes, plan, cursor,
            source => target => Transforms.copy_loop(fuel, input, axes, plan, source, target, 1n+cursor), step),
          Transforms.NaturalError{source}, Transforms.natural_step(input, axes, plan, cursor, before_source, before_target), exact), Unit{}}
    case Transforms.NaturalCopied{pair, index, converted, candidate}:
      loop_after_pair(fuel, input, axes, plan, cursor, original, before_source, before_target, index, converted, candidate, prefix,
        pair, exact, meaning, rec)

def loop_certificate(fuel: Nat, +input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +cursor: Nat,
  -original: Core.Tensor<input>, source: Core.Tensor<input>, target: Core.Tensor<Layout.output(input, axes, plan)>,
  -prefix: Transforms.Prefix<input, axes, plan, original, source, target, cursor>) ->
  Transforms.LoopCertificate<input, axes, plan, original, fuel, cursor, source, target>:
  match fuel:
    case 0n:
      Transforms.LoopCertificate{(source, Done{target}), {==}, prefix}
    case 1n++remaining:
      loop_after_natural(remaining, input, axes, plan, cursor, original, source, target, prefix,
        natural_certificate(input, axes, plan, cursor, source, target),
        source => target => invariant => loop_certificate(remaining, input, axes, plan, 1n+cursor, original, source, target, invariant))

def add_zero_right(+left: Nat) -> {Nat.add(left, 0n) == left : Nat}:
  match left:
    case 0n:
      {==}
    case 1n+p:
      Equal.cong(Nat, Nat, n => 1n+n, Nat.add(p, 0n), p, add_zero_right(p))

def whole_copy_certificate(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  source: Core.Tensor<input>, target: Core.Tensor<Layout.output(input, axes, plan)>) ->
  Transforms.LoopCertificate<input, axes, plan, source, Shape.elements(Layout.output(input, axes, plan)), 0n, source, target>:
  loop_certificate(Shape.elements(Layout.output(input, axes, plan)), input, axes, plan, 0n, source, source, target,
    Laws.initial_prefix(input, axes, plan, source, target))

def whole_copy_meaning(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  -source: Core.Tensor<input>, -target: Core.Tensor<Layout.output(input, axes, plan)>,
  certificate: Transforms.LoopCertificate<input, axes, plan, source, Shape.elements(Layout.output(input, axes, plan)), 0n, source, target>) ->
  Transforms.WholeCopyModel<input, axes, plan, source, target>:
  Transforms.LoopCertificate{pair, exact, meaning} = certificate
  Transforms.WholeCopyModel{
    %exact : Transforms.LoopMeaning(input, axes, plan, source,
      Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), _)
    %Shape.elements_exact(Layout.output(input, axes, plan)) : Transforms.LoopMeaning(input, axes, plan, source, _, pair)
    %add_zero_right(Shape.elements(Layout.output(input, axes, plan))) : Transforms.LoopMeaning(input, axes, plan, source, _, pair)
    meaning}

def Laws.whole_copy_interpretation(input, axes, plan, source, target):
  whole_copy_meaning(input, axes, plan, source, target, whole_copy_certificate(input, axes, plan, source, target))

def permuted_output(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  -original: Core.Tensor<input>, -source: Core.Tensor<input>, -target: Core.Tensor<Layout.output(input, axes, plan)>,
  prefix: Transforms.Prefix<input, axes, plan, original, source, target,
    Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))>) ->
  Transforms.PermuteOutput<input, axes, original, source, Core.Packed{Layout.output(input, axes, plan), target}>:
  Transforms.Prefix{kept, words, checked, written} = prefix
  Transforms.PermuteOutput{plan, {==}, kept, words, checked, written,
    other => bound => Equal.trans(Maybe<&2, F32>, Core.observe(Layout.output(input, axes, plan), target, U32.from_nat(other)),
      Core.checked_observe(Layout.output(input, axes, plan), target, U32.from_nat(other)),
      Transforms.expected_output(input, axes, plan, original, U32.from_nat(other)),
      {\{ObserveBridge: equal => Equal.sym(Maybe<&2, F32>,
        Core.checked_observe(Layout.output(input, axes, plan), target, U32.from_nat(other)),
        Core.observe(Layout.output(input, axes, plan), target, U32.from_nat(other)), equal)} :
          ObserveBridge<Layout.output(input, axes, plan), target, U32.from_nat(other)> ->
            {Core.observe(Layout.output(input, axes, plan), target, U32.from_nat(other)) ==
              Core.checked_observe(Layout.output(input, axes, plan), target, U32.from_nat(other)) : Maybe<&2, F32>}}(
                observe_bridge(Layout.output(input, axes, plan), target, U32.from_nat(other), checked(other, bound))),
      written(other, bound))}

def copy_pack_model(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, -original: Core.Tensor<input>,
  pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
  -meaning: Transforms.LoopMeaning(input, axes, plan, original,
    Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), pair)) ->
  Transforms.PublicModel<input, axes, original, Transforms.copy_pack(input, axes, plan, pair)>:
  (source, result) = pair
  match result:
    case Fail{error}:
      Transforms.PublicModel{Unit{}}
    case Done{target}:
      Transforms.PublicModel{
        permuted_output(input, axes, plan, original, source, target, meaning)}

def allocated_model(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, source: Core.Tensor<input>,
  allocated: Result<&2, &1, Core.Error, Core.Tensor<Layout.output(input, axes, plan)>>) ->
  Transforms.PublicModel<input, axes, source, Transforms.copy_allocated(input, axes, plan, source, allocated)>:
  match allocated:
    case Fail{error}:
      Transforms.PublicModel{Unit{}}
    case Done{target}:
      copy_pack_model(input, axes, plan, source, Transforms.built(input, axes, plan, source, target),
        PermuteBuild.built_meaning(input, axes, plan, source, target))

def planned_model(+input: Shape.Shape, +axes: List<&2, U32>, source: Core.Tensor<input>,
  planned: Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>) ->
  Transforms.PublicModel<input, axes, source, Transforms.copy_planned(input, axes, source, planned)>:
  match planned:
    case Fail{error}:
      Transforms.PublicModel{Unit{}}
    case Done{+plan}:
      allocated_model(input, axes, plan, source, Core.new(Layout.output(input, axes, plan), 0.0))

def Laws.permute_result_interpretation(input, source, axes):
  planned_model(input, axes, source, Layout.permutation(input, axes))

def transpose_decided_model(+input: Shape.Shape, source: Core.Tensor<input>, +a: U32, +b: U32,
  decision: Shape.BoolCase<TransposeSpec.valid(input, a, b)>) ->
  Transforms.PublicModel<input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), source,
    Transforms.transpose_decided(input, source, a, b, decision)>:
  match decision:
    case Shape.BFalse{equal}:
      Transforms.PublicModel{Unit{}}
    case Shape.BTrue{equal}:
      Laws.permute_result_interpretation(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0))

def Laws.transpose_result_interpretation(input, source, a, b):
  transpose_decided_model(input, source, a, b,
    Shape.bool_case(TransposeSpec.valid(input, a, b)))

type ReshapeDecidedModel<-input: Shape.Shape, -target: Shape.Shape, -original: Core.Tensor<input>, -dimensions: List<&2, U32>,
  -decision: Shape.BoolCase<Nat.is_eq(Shape.elements(input), Shape.elements(target))>> is Data:
  ReshapeDecidedModel{-meaning: Transforms.ReshapeMeaning(input, original, dimensions,
    Transforms.reshape_decided(input, target, original, decision))}

def reshape_decided_model(+input: Shape.Shape, +target: Shape.Shape, source: Core.Tensor<input>, -dimensions: List<&2, U32>,
  -preserved: {Shape.dimensions(target) == dimensions : List<&2, U32>},
  decision: Shape.BoolCase<Nat.is_eq(Shape.elements(input), Shape.elements(target))>) ->
  ReshapeDecidedModel<input, target, source, dimensions, decision>:
  match decision:
    case Shape.BFalse{equal}:
      ReshapeDecidedModel{Unit{}}
    case Shape.BTrue{equal}:
      ReshapeDecidedModel{Transforms.ReshapeOutput{target,
        Equal.trans(List<&2, U32>, Transforms.dynamic_dimensions(Transforms.reshape_equal(input, target, source,
            Transforms.elements_equal(input, target, equal))), Shape.dimensions(target), dimensions,
          Laws.reshape_target_dimensions(input, target, source, Transforms.elements_equal(input, target, equal)), preserved),
        preserved,
        Equal.trans(Nat, Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(target)),
          Shape.logical_numel(dimensions), Transforms.elements_equal(input, target, equal),
          Equal.cong(List<&2, U32>, Nat, Shape.logical_numel, Shape.dimensions(target), dimensions, preserved)),
        index => Laws.reshape_linear_order(input, target, source, index, Transforms.elements_equal(input, target, equal))}}

type ReshapeBuiltModel<-input: Shape.Shape, -original: Core.Tensor<input>, -dimensions: List<&2, U32>,
  -built: Shape.ConstructResult<dimensions>> is Data:
  ReshapeBuiltModel{-meaning: Transforms.ReshapeMeaning(input, original, dimensions,
    Transforms.reshape_built(input, original, dimensions, built))}

def reshape_built_model(+input: Shape.Shape, source: Core.Tensor<input>, -dimensions: List<&2, U32>,
  built: Shape.ConstructResult<dimensions>) -> ReshapeBuiltModel<input, source, dimensions, built>:
  match built:
    case Shape.SError{error}:
      ReshapeBuiltModel{Unit{}}
    case Shape.SBuilt{+target, preserved}:
      ReshapeBuiltModel{
        {\{ReshapeDecidedModel: meaning => meaning} :
          ReshapeDecidedModel<input, target, source, dimensions, Shape.bool_case(Nat.is_eq(Shape.elements(input), Shape.elements(target)))> ->
            Transforms.ReshapeMeaning(input, source, dimensions,
              Transforms.reshape_decided(input, target, source, Shape.bool_case(Nat.is_eq(Shape.elements(input), Shape.elements(target)))))}(
                reshape_decided_model(input, target, source, dimensions, preserved,
                  Shape.bool_case(Nat.is_eq(Shape.elements(input), Shape.elements(target)))))}

def Laws.reshape_result_interpretation(input, source, dimensions):
  Transforms.ReshapeModel{
    {\{ReshapeBuiltModel: meaning => meaning} : ReshapeBuiltModel<input, source, dimensions, Shape.construct(dimensions)> ->
      Transforms.ReshapeMeaning(input, source, dimensions, Transforms.reshape(input, source, dimensions))}(
        reshape_built_model(input, source, dimensions, Shape.construct(dimensions)))}

type RestoreModel<-input: Shape.Shape, -original: Core.Tensor<input>, -result: Result<&2, &1, Transforms.Error, Core.Dynamic>> is Data:
  RestoreModel{-meaning: Transforms.RestoreMeaning(input, original, result)}

def restore_result_model(-input: Shape.Shape, -original: Core.Tensor<input>, -middle_shape: Shape.Shape,
  -middle: Core.Tensor<middle_shape>, result: Result<&2, &1, Transforms.Error, Core.Dynamic>,
  -meaning: Transforms.ReshapeMeaning(middle_shape, middle, Shape.dimensions(input), result),
  -prior: @+index: U32 -> {Core.observe(middle_shape, middle, index) == Core.observe(input, original, index) : Maybe<&2, F32>}) ->
  RestoreModel<input, original, result>:
  match result:
    case Fail{error}:
      RestoreModel{Unit{}}
    case Done{output}:
      RestoreModel{
        {\{Transforms.ReshapeOutput: shape => dimensions => shape_exact => count => linear =>
          Transforms.RestoredOutput{dimensions,
            index => Equal.trans(Maybe<&2, F32>, Transforms.dynamic_observe(output, index),
              Core.observe(middle_shape, middle, index), Core.observe(input, original, index), linear(index), prior(index))}} :
                Transforms.ReshapeOutput<middle_shape, middle, Shape.dimensions(input), output> ->
                  Transforms.RestoredOutput<input, original, output>}(meaning)}

def restore_dynamic_model(+input: Shape.Shape, -original: Core.Tensor<input>, middle: Core.Dynamic,
  -prior: @+index: U32 -> {Transforms.dynamic_observe(middle, index) == Core.observe(input, original, index) : Maybe<&2, F32>}) ->
  RestoreModel<input, original, Transforms.reshape_dynamic(middle, Shape.dimensions(input))>:
  Core.Packed{+shape, tensor} = middle
  restore_result_model(input, original, shape, tensor, Transforms.reshape(shape, tensor, Shape.dimensions(input)),
    {\{Transforms.ReshapeModel: meaning => meaning} : Transforms.ReshapeModel<shape, tensor, Shape.dimensions(input)> ->
      Transforms.ReshapeMeaning(shape, tensor, Shape.dimensions(input), Transforms.reshape(shape, tensor, Shape.dimensions(input)))}(
        Laws.reshape_result_interpretation(shape, tensor, Shape.dimensions(input))), prior)

type ReshapeInverseResultModel<-input: Shape.Shape, -original: Core.Tensor<input>, -result: Result<&2, &1, Transforms.Error, Core.Dynamic>> is Data:
  ReshapeInverseResultModel{-meaning: Transforms.ReshapeInverseMeaning(input, original, result)}

def reshape_inverse_result_model(-input: Shape.Shape, -original: Core.Tensor<input>, -dimensions: List<&2, U32>,
  result: Result<&2, &1, Transforms.Error, Core.Dynamic>,
  -meaning: Transforms.ReshapeMeaning(input, original, dimensions, result)) -> ReshapeInverseResultModel<input, original, result>:
  match result:
    case Fail{error}:
      ReshapeInverseResultModel{Unit{}}
    case Done{output}:
      ReshapeInverseResultModel{
        {\{Transforms.ReshapeOutput: shape => dimensions_exact => shape_exact => count => linear =>
          {\{RestoreModel: meaning => meaning} : RestoreModel<input, original, Transforms.reshape_dynamic(output, Shape.dimensions(input))> ->
            Transforms.RestoreMeaning(input, original, Transforms.reshape_dynamic(output, Shape.dimensions(input)))}(
              restore_dynamic_model(input, original, output, linear))} :
                Transforms.ReshapeOutput<input, original, dimensions, output> ->
                  Transforms.RestoreMeaning(input, original, Transforms.reshape_dynamic(output, Shape.dimensions(input)))}(meaning)}

def Laws.reshape_inverse_observations(input, source, dimensions):
  Transforms.ReshapeInverseModel{
    {\{ReshapeInverseResultModel: meaning => meaning} :
      ReshapeInverseResultModel<input, source, Transforms.reshape(input, source, dimensions)> ->
        Transforms.ReshapeInverseMeaning(input, source, Transforms.reshape(input, source, dimensions))}(
          reshape_inverse_result_model(input, source, dimensions, Transforms.reshape(input, source, dimensions),
            {\{Transforms.ReshapeModel: meaning => meaning} : Transforms.ReshapeModel<input, source, dimensions> ->
              Transforms.ReshapeMeaning(input, source, dimensions, Transforms.reshape(input, source, dimensions))}(
                Laws.reshape_result_interpretation(input, source, dimensions))))}

def Laws.mapping_reference(input, axes, plan, index, mapping):
  Mapping.mapping_reference(input, axes, plan, index, mapping)

def Laws.mapped_mathematical_bounded(input, axes, plan, index, mapping):
  Layout.Mapped{oi, oc, ic, source, requested, decoded, inverse, flattened, roundtrip} = mapping
  Mapping.flat_mathematical_bounded(input, source)

def natural_word_exact(+number: Nat, +limit: U32, +within: {Nat.is_le(number, U32.to_nat(limit)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(number)) == number : Nat}:
  WordConversion.natural_word_exact(number, limit, within)

def Laws.natural_word_exact(number, limit, within):
  natural_word_exact(number, limit, within)

def shape_count_bound(+shape: Shape.Shape) -> {Nat.is_le(Shape.elements(shape), 1048576n) == True{} : Bool}:
  match shape:
    case Shape.SShape{+dimensions, +size, +exact, rank_bound, capacity_bound}:
      %Equal.sym(Nat, size, Shape.logical_numel(dimensions), exact) : {Nat.is_le(_, 1048576n) == True{} : Bool}
      Equal.sym(Bool, True{}, Nat.is_le(Shape.logical_numel(dimensions), 1048576n),
        Nats.le_true(Shape.logical_numel(dimensions), 1048576n, capacity_bound))

type CheckedObservationModel<-shape: Shape.Shape, -tensor: Core.Tensor<shape>, -index: U32> is Data:
  CheckedObservationModel{-checked: Core.ObservationChecked(Core.Packed{shape, tensor}, index)}

def checked_observation_model(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +index: U32,
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  CheckedObservationModel<shape, tensor, index>:
  Core.TTensor{storage, exact} = tensor
  match storage:
    case Storage.Empty{}:
      CheckedObservationModel{Arith.lt_zero(U32.to_nat(index),
        %Equal.sym(Nat, 0n, Shape.logical_numel(Shape.dimensions(shape)), exact) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
        inside)}
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      CheckedObservationModel{Equal.sym(Bool, True{}, Storage.index_matches(depth, length, index),
        Route.index_complete(depth, length, index, valid,
          %Equal.sym(Nat, length, Shape.logical_numel(Shape.dimensions(shape)), exact) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
          inside))}

type PresentObservation<-shape: Shape.Shape, -tensor: Core.Tensor<shape>, -index: U32> is Data:
  PresentObservation{-value: F32, -equal: {Core.checked_observe(shape, tensor, index) == Some{value} : Maybe<&2, F32>}}

def present_impossible(-shape: Shape.Shape, -tensor: Core.Tensor<shape>, -index: U32, -empty: Empty) -> PresentObservation<shape, tensor, index>:
  PresentObservation{Empty.absurd(F32, empty),
    Empty.absurd({Core.checked_observe(shape, tensor, index) == Some{Empty.absurd(F32, empty)} : Maybe<&2, F32>}, empty)}

def present_observation(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +index: U32,
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  PresentObservation<shape, tensor, index>:
  Core.TTensor{storage, exact} = tensor
  match storage:
    case Storage.Empty{}:
      present_impossible(shape, Core.TTensor{Storage.Empty{}, exact}, index,
        Arith.lt_zero(U32.to_nat(index),
          %Equal.sym(Nat, 0n, Shape.logical_numel(Shape.dimensions(shape)), exact) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
          inside))
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      PresentObservation{Refinement.second(F32, Array.get(F32, array, index)),
        Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, index),
          Model.observe(F32, depth, array, Refinement.route(depth, index)), Some{Refinement.second(F32, Array.get(F32, array, index))},
          observe_allowed_equal(depth, length, array, index, Nat.is_lt(U32.to_nat(index), length),
            %Equal.sym(Nat, length, Shape.logical_numel(Shape.dimensions(shape)), exact) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
            inside,
            Equal.sym(Bool, True{}, Storage.index_matches(depth, length, index),
              Route.index_complete(depth, length, index, valid,
                %Equal.sym(Nat, length, Shape.logical_numel(Shape.dimensions(shape)), exact) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
                inside))),
          Equal.sym(Maybe<&2, F32>, Some{Refinement.second(F32, Array.get(F32, array, index))},
            Model.observe(F32, depth, array, Refinement.route(depth, index)), Refinement.law_get_refines(F32, depth, array, index, balanced)))}

def maybe_present(value: Maybe<&2, F32>) -> Bool:
  match value:
    case None{}:
      False{}
    case Some{value}:
      True{}

type ReadSuccess<-shape: Shape.Shape, -original: Core.Tensor<shape>, -index: U32> is Type:
  ReadSuccess{retained: Core.Tensor<shape>, value: F32,
    exact: {(retained, Done{value}) == Core.read_flat(shape, original, index) : Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>}}

type ReadSuccessModel<-shape: Shape.Shape, -original: Core.Tensor<shape>, -index: U32> is Data:
  ReadSuccessModel{-success: ReadSuccess<shape, original, index>}

def read_success_model(-shape: Shape.Shape, -original: Core.Tensor<shape>, +index: U32,
  certificate: Core.Read<shape, original, index>,
  -model: Core.SourceRead<shape, original, index>, -present: PresentObservation<shape, original, index>) -> ReadSuccessModel<shape, original, index>:
  Core.Read{retained, result, exact} = certificate
  match result:
    case Fail{error}:
      ReadSuccessModel{Empty.absurd(ReadSuccess<shape, original, index>,
        {\{Core.SourceRead: meaning =>
          {\{PresentObservation: value => equal => Arith.clash(Equal.cong(Maybe<&2, F32>, Bool, maybe_present, None{}, Some{value},
            Equal.trans(Maybe<&2, F32>, None{}, Core.checked_observe(shape, original, index), Some{value},
              Equal.trans(Maybe<&2, F32>, None{}, Core.read_value(shape, Core.read_flat(shape, original, index)),
                Core.checked_observe(shape, original, index),
                Equal.cong(Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>, Maybe<&2, F32>, pair => Core.read_value(shape, pair),
                  (retained, Fail{error}), Core.read_flat(shape, original, index), exact), meaning), equal)))} :
                    PresentObservation<shape, original, index> -> Empty}(present)} : Core.SourceRead<shape, original, index> -> Empty}(model))}
    case Done{value}:
      ReadSuccessModel{ReadSuccess{retained, value, exact}}

type ReadTotalInfo<-shape: Shape.Shape, -original: Core.Tensor<shape>, -index: U32> is Type:
  ReadTotalInfo{owner: Core.Tensor<shape>, same: {owner == original : Core.Tensor<shape>},
    -present: PresentObservation<shape, original, index>, -model: Core.SourceRead<shape, original, index>}

def read_total_info(+shape: Shape.Shape, original: Core.Tensor<shape>, +index: U32,
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  ReadTotalInfo<shape, original, index>:
  ReadTotalInfo{original, {==}, present_observation(shape, original, index, inside), Core.source_read_model(shape, original, index)}

def read_total_certificate(-shape: Shape.Shape, -original: Core.Tensor<shape>, +index: U32,
  info: ReadTotalInfo<shape, original, index>) -> ReadSuccessModel<shape, original, index>:
  ReadTotalInfo{owner, same, present, model} = info
  read_success_model(shape, original, index, read_rebased(shape, original, owner, index, same, Core.read_certificate(shape, owner, index)), model, present)

def read_success(+shape: Shape.Shape, original: Core.Tensor<shape>, +index: U32,
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  ReadSuccessModel<shape, original, index>:
  read_total_certificate(shape, original, index, read_total_info(shape, original, index, inside))

def decoded_offset(+shape: Shape.Shape, +index: U32,
  +inside: {True{} == Nat.is_lt(U32.to_nat(index), Shape.elements(shape)) : Bool}) ->
  {Shape.offset(Shape.extents(shape), Shape.digits(Shape.extents(shape), U32.to_nat(index))) == U32.to_nat(index) : Nat}:
  %Mapping.decoded_indices(shape, index, inside) : {Shape.offset(Shape.extents(shape), _) == U32.to_nat(index) : Nat}
  Equal.trans(Nat, Shape.offset(Shape.extents(shape), Shape.indices(Shape.extents(shape), Mapping.decoded_coordinate(shape, index, inside))),
    Shape.row_major(Shape.extents(shape), Mapping.decoded_coordinate(shape, index, inside)), U32.to_nat(index),
    Coordinates.offset_indices(Shape.extents(shape), Mapping.decoded_coordinate(shape, index, inside)),
    Equal.trans(Nat, Shape.row_major(Shape.extents(shape), Mapping.decoded_coordinate(shape, index, inside)),
      Shape.Fin.value(Shape.numel(Shape.extents(shape)),
        Coordinates.fin_from(Shape.numel(Shape.extents(shape)), U32.to_nat(index), Coordinates.size_fits(shape, U32.to_nat(index), inside))),
      U32.to_nat(index),
      Equal.trans(Nat, Shape.row_major(Shape.extents(shape), Mapping.decoded_coordinate(shape, index, inside)),
        Shape.Fin.value(Shape.numel(Shape.extents(shape)), Shape.flatten(Shape.extents(shape), Mapping.decoded_coordinate(shape, index, inside))),
        Shape.Fin.value(Shape.numel(Shape.extents(shape)),
          Coordinates.fin_from(Shape.numel(Shape.extents(shape)), U32.to_nat(index), Coordinates.size_fits(shape, U32.to_nat(index), inside))),
        Equal.sym(Nat, Shape.Fin.value(Shape.numel(Shape.extents(shape)), Shape.flatten(Shape.extents(shape), Mapping.decoded_coordinate(shape, index, inside))),
          Shape.row_major(Shape.extents(shape), Mapping.decoded_coordinate(shape, index, inside)),
          ShapeIndex.flatten_row_major(Shape.extents(shape), Mapping.decoded_coordinate(shape, index, inside))),
        Equal.cong(Shape.Fin(Shape.numel(Shape.extents(shape))), Nat, value => Shape.Fin.value(Shape.numel(Shape.extents(shape)), value),
          Shape.flatten(Shape.extents(shape), Mapping.decoded_coordinate(shape, index, inside)),
          Coordinates.fin_from(Shape.numel(Shape.extents(shape)), U32.to_nat(index), Coordinates.size_fits(shape, U32.to_nat(index), inside)),
          ShapeIndex.flatten_unflatten(Shape.extents(shape),
            Coordinates.fin_from(Shape.numel(Shape.extents(shape)), U32.to_nat(index), Coordinates.size_fits(shape, U32.to_nat(index), inside))))),
      Coordinates.fin_from_value(Shape.numel(Shape.extents(shape)), U32.to_nat(index), Coordinates.size_fits(shape, U32.to_nat(index), inside))))

def Laws.codec_flatten_decode(shape, index, inside):
  %Equal.sym(List<&2, U32>, Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(shape)), index)),
    Shape.words(Shape.digits(Shape.extents(shape), U32.to_nat(index))), Mapping.codec_decoded(Shape.dimensions(shape), index)) :
      {Layout.flatten_words(Shape.dimensions(shape), _, 0) == index : U32}
  Equal.trans(U32, Layout.flatten_words(Shape.dimensions(shape), Shape.words(Shape.digits(Shape.extents(shape), U32.to_nat(index))), 0),
    U32.from_nat(U32.to_nat(index)), index,
    Equal.cong(Nat, U32, U32.from_nat,
      Shape.horner(Shape.dimensions(shape), Shape.words(Shape.digits(Shape.extents(shape), U32.to_nat(index))), 0n), U32.to_nat(index),
      Equal.trans(Nat, Shape.horner(Shape.dimensions(shape), Shape.words(Shape.digits(Shape.extents(shape), U32.to_nat(index))), 0n),
        Shape.offset(Shape.extents(shape), Shape.naturals(Shape.words(Shape.digits(Shape.extents(shape), U32.to_nat(index))))), U32.to_nat(index),
        Coordinates.horner_spec(Shape.dimensions(shape), Shape.words(Shape.digits(Shape.extents(shape), U32.to_nat(index))), 0n,
          Mapping.decoded_word_fits(shape, index, inside)),
        %Equal.sym(List<&2, Nat>, Shape.naturals(Shape.words(Shape.digits(Shape.extents(shape), U32.to_nat(index)))),
          Shape.digits(Shape.extents(shape), U32.to_nat(index)),
          WordConversion.naturals_words_fitting(Shape.dimensions(shape), Shape.digits(Shape.extents(shape), U32.to_nat(index)), Mapping.decoded_fit(shape, index, inside))) :
            {Shape.offset(Shape.extents(shape), _) == U32.to_nat(index) : Nat}
        decoded_offset(shape, index, inside))),
    WordConversion.word_natural_exact(index))

def shape_construct_complete(+shape: Shape.Shape) ->
  {ShapeLaws.built(Shape.dimensions(shape), Shape.construct(Shape.dimensions(shape))) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>}:
  match shape:
    case Shape.SShape{+dimensions, size, exact, rank_bound, capacity_bound}:
      ShapeLaws.construction_complete(dimensions, rank_bound, capacity_bound)

type ReshapeDecisionSuccess<-input: Shape.Shape, -target: Shape.Shape, -source: Core.Tensor<input>,
  -decision: Shape.BoolCase<Nat.is_eq(Shape.elements(input), Shape.elements(target))>> is Data:
  ReshapeDecisionSuccess{-success: {Result.is_done(&2, &1, Transforms.Error, Core.Dynamic,
    Transforms.reshape_decided(input, target, source, decision)) == True{} : Bool}}

def reshape_decision_success(+input: Shape.Shape, +target: Shape.Shape,
  decision: Shape.BoolCase<Nat.is_eq(Shape.elements(input), Shape.elements(target))>, source: Core.Tensor<input>,
  -same: {Shape.elements(input) == Shape.elements(target) : Nat}) -> ReshapeDecisionSuccess<input, target, source, decision>:
  match decision:
    case Shape.BFalse{equal}:
      ReshapeDecisionSuccess{Empty.absurd({Result.is_done(&2, &1, Transforms.Error, Core.Dynamic,
        Transforms.reshape_decided(input, target, source, Shape.BFalse{equal})) == True{} : Bool},
        Arith.clash(Equal.trans(Bool, False{}, Nat.is_eq(Shape.elements(input), Shape.elements(target)), True{}, equal,
          Mapping.natural_equal_true(Shape.elements(input), Shape.elements(target), same))))}
    case Shape.BTrue{equal}:
      ReshapeDecisionSuccess{{==}}

type ReshapeBuiltSuccess<-input: Shape.Shape, -target: Shape.Shape, -source: Core.Tensor<input>,
  -built: Shape.ConstructResult<Shape.dimensions(target)>> is Data:
  ReshapeBuiltSuccess{-success: {Result.is_done(&2, &1, Transforms.Error, Core.Dynamic,
    Transforms.reshape_built(input, source, Shape.dimensions(target), built)) == True{} : Bool}}

def reshape_built_success(+input: Shape.Shape, +target: Shape.Shape,
  built: Shape.ConstructResult<Shape.dimensions(target)>, source: Core.Tensor<input>,
  -same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat},
  -complete: {ShapeLaws.built(Shape.dimensions(target), built) == Some{Shape.dimensions(target)} : Maybe<&2, List<&2, U32>>}) ->
  ReshapeBuiltSuccess<input, target, source, built>:
  match built:
    case Shape.SError{error}:
      ReshapeBuiltSuccess{Empty.absurd({Result.is_done(&2, &1, Transforms.Error, Core.Dynamic,
        Transforms.reshape_built(input, source, Shape.dimensions(target), Shape.SError{error})) == True{} : Bool},
        Arith.clash(Equal.cong(Maybe<&2, List<&2, U32>>, Bool, Mapping.metadata_present, None{}, Some{Shape.dimensions(target)}, complete)))}
    case Shape.SBuilt{+output, preserved}:
      ReshapeBuiltSuccess{
        {\{ReshapeDecisionSuccess: success => success} :
          ReshapeDecisionSuccess<input, output, source, Shape.bool_case(Nat.is_eq(Shape.elements(input), Shape.elements(output)))> ->
            {Result.is_done(&2, &1, Transforms.Error, Core.Dynamic,
              Transforms.reshape_decided(input, output, source, Shape.bool_case(Nat.is_eq(Shape.elements(input), Shape.elements(output))))) == True{} : Bool}}(
                reshape_decision_success(input, output, Shape.bool_case(Nat.is_eq(Shape.elements(input), Shape.elements(output))), source,
                  Mapping.element_counts_same(input, output,
                    Equal.trans(Nat, Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(target)),
                      Shape.logical_numel(Shape.dimensions(output)), same,
                      Equal.cong(List<&2, U32>, Nat, Shape.logical_numel, Shape.dimensions(target), Shape.dimensions(output),
                        Equal.sym(List<&2, U32>, Shape.dimensions(output), Shape.dimensions(target), preserved))))))}

def Laws.reshape_valid_success(input, target, source, same):
  Transforms.ReshapeSuccessModel{
    {\{ReshapeBuiltSuccess: success => success} : ReshapeBuiltSuccess<input, target, source, Shape.construct(Shape.dimensions(target))> ->
      {Result.is_done(&2, &1, Transforms.Error, Core.Dynamic, Transforms.reshape(input, source, Shape.dimensions(target))) == True{} : Bool}}(
        reshape_built_success(input, target, Shape.construct(Shape.dimensions(target)), source, same, shape_construct_complete(target)))}

type DynamicRestoreSuccess<-input: Shape.Shape, -output: Core.Dynamic> is Data:
  DynamicRestoreSuccess{-success: {Result.is_done(&2, &1, Transforms.Error, Core.Dynamic,
    Transforms.reshape_dynamic(output, Shape.dimensions(input))) == True{} : Bool}}

def dynamic_restore_success(+input: Shape.Shape, output: Core.Dynamic,
  -same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Transforms.dynamic_dimensions(output)) : Nat}) ->
  DynamicRestoreSuccess<input, output>:
  Core.Packed{+shape, tensor} = output
  DynamicRestoreSuccess{
    {\{Transforms.ReshapeSuccessModel: success => success} : Transforms.ReshapeSuccessModel<shape, input, tensor> ->
      {Result.is_done(&2, &1, Transforms.Error, Core.Dynamic, Transforms.reshape(shape, tensor, Shape.dimensions(input))) == True{} : Bool}}(
        Laws.reshape_valid_success(shape, input, tensor,
          Equal.sym(Nat, Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(shape)), same)))}

type ReshapeInverseSuccessResult<-input: Shape.Shape, -result: Result<&2, &1, Transforms.Error, Core.Dynamic>> is Data:
  ReshapeInverseSuccessResult{-success: {Transforms.reshape_inverse_success(input, result) == True{} : Bool}}

def reshape_inverse_success_result(+input: Shape.Shape, -original: Core.Tensor<input>, -dimensions: List<&2, U32>,
  result: Result<&2, &1, Transforms.Error, Core.Dynamic>,
  -successful: {Result.is_done(&2, &1, Transforms.Error, Core.Dynamic, result) == True{} : Bool},
  -meaning: Transforms.ReshapeMeaning(input, original, dimensions, result)) -> ReshapeInverseSuccessResult<input, result>:
  match result:
    case Fail{error}:
      ReshapeInverseSuccessResult{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{output}:
      ReshapeInverseSuccessResult{
        {\{Transforms.ReshapeOutput: shape => dims => shape_exact => count => linear =>
          {\{DynamicRestoreSuccess: success => success} : DynamicRestoreSuccess<input, output> ->
            {Result.is_done(&2, &1, Transforms.Error, Core.Dynamic, Transforms.reshape_dynamic(output, Shape.dimensions(input))) == True{} : Bool}}(
              dynamic_restore_success(input, output,
                Equal.trans(Nat, Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(dimensions),
                  Shape.logical_numel(Transforms.dynamic_dimensions(output)), count,
                  Equal.cong(List<&2, U32>, Nat, Shape.logical_numel, dimensions, Transforms.dynamic_dimensions(output),
                    Equal.sym(List<&2, U32>, Transforms.dynamic_dimensions(output), dimensions, dims)))))} :
                      Transforms.ReshapeOutput<input, original, dimensions, output> ->
                        {Result.is_done(&2, &1, Transforms.Error, Core.Dynamic, Transforms.reshape_dynamic(output, Shape.dimensions(input))) == True{} : Bool}}(meaning)}

def Laws.reshape_inverse_valid_success(input, target, source, same):
  Transforms.ReshapeInverseSuccessModel{
    {\{ReshapeInverseSuccessResult: success => success} :
      ReshapeInverseSuccessResult<input, Transforms.reshape(input, source, Shape.dimensions(target))> ->
        {Transforms.reshape_inverse_success(input, Transforms.reshape(input, source, Shape.dimensions(target))) == True{} : Bool}}(
          reshape_inverse_success_result(input, source, Shape.dimensions(target), Transforms.reshape(input, source, Shape.dimensions(target)),
            {\{Transforms.ReshapeSuccessModel: success => success} : Transforms.ReshapeSuccessModel<input, target, source> ->
              {Result.is_done(&2, &1, Transforms.Error, Core.Dynamic, Transforms.reshape(input, source, Shape.dimensions(target))) == True{} : Bool}}(
                Laws.reshape_valid_success(input, target, source, same)),
            {\{Transforms.ReshapeModel: meaning => meaning} : Transforms.ReshapeModel<input, source, Shape.dimensions(target)> ->
              Transforms.ReshapeMeaning(input, source, Shape.dimensions(target), Transforms.reshape(input, source, Shape.dimensions(target)))}(
                Laws.reshape_result_interpretation(input, source, Shape.dimensions(target)))))}

def Laws.permutation_valid_success(input, axes, valid):
  Totality.PlanSuccess{Mapping.plan_ranked_success(input, axes,
    Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), valid)}

def read_success_status(-shape: Shape.Shape, -original: Core.Tensor<shape>, -index: U32,
  certificate: ReadSuccessModel<shape, original, index>) -> Totality.ReadSuccess<shape, original, index>:
  Totality.ReadSuccess{
    {\{ReadSuccessModel: success =>
      {\{ReadSuccess: retained => value => exact =>
        Equal.sym(Bool, True{}, Totality.read_succeeded(shape, Core.read_flat(shape, original, index)),
          Equal.cong(Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>, Bool, Totality.read_succeeded(shape),
            (retained, Done{value}), Core.read_flat(shape, original, index), exact))} :
              ReadSuccess<shape, original, index> -> {Totality.read_succeeded(shape, Core.read_flat(shape, original, index)) == True{} : Bool}}(success)} :
                ReadSuccessModel<shape, original, index> -> {Totality.read_succeeded(shape, Core.read_flat(shape, original, index)) == True{} : Bool}}(certificate)}

def Laws.tensor_read_valid_success(shape, source, index, inside):
  read_success_status(shape, source, index, read_success(shape, source, index, inside))

def storage_write_succeeded(pair: Storage.Storage & Result<&2, &2, Storage.Error, Unit>) -> Bool:
  (storage, result) = pair
  Result.is_done(&2, &2, Storage.Error, Unit, result)

type StorageWriteSuccess<-pair: Storage.Storage & Result<&2, &2, Storage.Error, Unit>> is Data:
  StorageWriteSuccess{-successful: {storage_write_succeeded(pair) == True{} : Bool}}

def storage_set_checked_success(+depth: Nat, +length: Nat, +index: U32, +value: F32, allowed: Bool, array: Array<F32>,
  -decision: {allowed == Nat.is_lt(U32.to_nat(index), length) : Bool},
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -inside: {Nat.is_lt(U32.to_nat(index), length) == True{} : Bool}) ->
  StorageWriteSuccess<Storage.set_checked(depth, length, array, index, value, allowed, decision, balanced, valid)>:
  match allowed:
    case False{}:
      StorageWriteSuccess{Empty.absurd({False{} == True{} : Bool},
        Arith.clash(Equal.trans(Bool, False{}, Nat.is_lt(U32.to_nat(index), length), True{}, decision, inside)))}
    case True{}:
      StorageWriteSuccess{{==}}

def write_result_status(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +value: F32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, Unit>,
  -same: {pair == Storage.set(original, index, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}) ->
  {Totality.write_succeeded(shape, Core.write_result(shape, original, valid, index, value, pair, same)) == storage_write_succeeded(pair) : Bool}:
  (storage, result) = pair
  match result:
    case Fail{error}:
      {==}
    case Done{unit}:
      {==}

def tensor_write_success(+shape: Shape.Shape, original: Core.Tensor<shape>, +index: U32, +value: F32,
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) -> Totality.WriteSuccess<shape, original, index, value>:
  Core.TTensor{storage, exact} = original
  match storage:
    case Storage.Empty{}:
      Totality.WriteSuccess{Empty.absurd({False{} == True{} : Bool},
        Arith.lt_zero(U32.to_nat(index),
          %Equal.sym(Nat, 0n, Shape.logical_numel(Shape.dimensions(shape)), exact) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
          inside))}
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      Totality.WriteSuccess{
        Equal.trans(Bool, Totality.write_succeeded(shape,
          Core.write_flat(shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, index, value)),
          storage_write_succeeded(Storage.set(Storage.Owned{depth, length, array, balanced, valid}, index, value)), True{},
          write_result_status(shape, Storage.Owned{depth, length, array, balanced, valid}, exact, index, value,
            Storage.set(Storage.Owned{depth, length, array, balanced, valid}, index, value), {==}),
          {\{StorageWriteSuccess: successful => successful} :
            StorageWriteSuccess<Storage.set_checked(depth, length, array, index, value, Nat.is_lt(U32.to_nat(index), length), {==}, balanced, valid)> ->
              {storage_write_succeeded(Storage.set(Storage.Owned{depth, length, array, balanced, valid}, index, value)) == True{} : Bool}}(
                storage_set_checked_success(depth, length, index, value, Nat.is_lt(U32.to_nat(index), length), array, {==}, balanced, valid,
                  %Equal.sym(Nat, length, Shape.logical_numel(Shape.dimensions(shape)), exact) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
                  inside)))}

def Laws.tensor_write_valid_success(shape, target, index, value, inside):
  tensor_write_success(shape, target, index, value, inside)

def Laws.decoded_words_valid_success(shape, index, inside):
  Totality.WordsSuccess{Mapping.words_check_success(Shape.dimensions(shape),
    Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(shape)), index)), Mapping.decoded_words_valid(shape, index, inside))}

def permutation_length_decision_failure(+input: Shape.Shape, +axes: List<&2, U32>,
  decision: Shape.BoolCase<Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))>,
  wrong: {False{} == Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))) : Bool}) ->
  {Layout.plan_ranked(input, axes, decision) == Fail{Layout.AxesLengthMismatch{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>}:
  match decision:
    case Shape.BFalse{equal}:
      {==}
    case Shape.BTrue{equal}:
      Empty.absurd({Layout.plan_ranked(input, axes, Shape.BTrue{equal}) == Fail{Layout.AxesLengthMismatch{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>},
        Lifecycle.false_true(Equal.trans(Bool, True{}, Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))), False{}, equal,
          Equal.sym(Bool, False{}, Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))), wrong))))

def permutation_length_failure(+input: Shape.Shape, +axes: List<&2, U32>,
  wrong: {False{} == Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))) : Bool}) ->
  {Layout.permutation(input, axes) == Fail{Layout.AxesLengthMismatch{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>}:
  permutation_length_decision_failure(input, axes,
    Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), wrong)

def permutation_bounds_decision_failure(+input: Shape.Shape, +axes: List<&2, U32>,
  decision: Shape.BoolCase<Layout.bounded(axes, Shape.rank(input))>, outside: {False{} == Layout.bounded(axes, Shape.rank(input)) : Bool}) ->
  {Layout.plan_bounded(input, axes, decision) == Fail{Layout.AxisOutOfBounds{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>}:
  match decision:
    case Shape.BFalse{equal}:
      {==}
    case Shape.BTrue{equal}:
      Empty.absurd({Layout.plan_bounded(input, axes, Shape.BTrue{equal}) == Fail{Layout.AxisOutOfBounds{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>},
        Lifecycle.false_true(Equal.trans(Bool, True{}, Layout.bounded(axes, Shape.rank(input)), False{}, equal,
          Equal.sym(Bool, False{}, Layout.bounded(axes, Shape.rank(input)), outside))))

def permutation_bounds_ranked_failure(+input: Shape.Shape, +axes: List<&2, U32>,
  decision: Shape.BoolCase<Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))>,
  ranked: {True{} == Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))) : Bool},
  outside: {False{} == Layout.bounded(axes, Shape.rank(input)) : Bool}) ->
  {Layout.plan_ranked(input, axes, decision) == Fail{Layout.AxisOutOfBounds{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({Layout.plan_ranked(input, axes, Shape.BFalse{equal}) == Fail{Layout.AxisOutOfBounds{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>},
        Arith.clash(Equal.trans(Bool, False{}, Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))), True{}, equal,
          Equal.sym(Bool, True{}, Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))), ranked))))
    case Shape.BTrue{equal}:
      permutation_bounds_decision_failure(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input))), outside)

def permutation_bounds_failure(+input: Shape.Shape, +axes: List<&2, U32>,
  ranked: {True{} == Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))) : Bool},
  outside: {False{} == Layout.bounded(axes, Shape.rank(input)) : Bool}) ->
  {Layout.permutation(input, axes) == Fail{Layout.AxisOutOfBounds{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>}:
  permutation_bounds_ranked_failure(input, axes,
    Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), ranked, outside)

def permutation_unique_decision_failure(+input: Shape.Shape, +axes: List<&2, U32>,
  decision: Shape.BoolCase<Layout.unique(axes)>, duplicate: {False{} == Layout.unique(axes) : Bool}) ->
  {Layout.plan_unique(input, axes, decision) == Fail{Layout.DuplicateAxis{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>}:
  match decision:
    case Shape.BFalse{equal}:
      {==}
    case Shape.BTrue{equal}:
      Empty.absurd({Layout.plan_unique(input, axes, Shape.BTrue{equal}) == Fail{Layout.DuplicateAxis{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>},
        Lifecycle.false_true(Equal.trans(Bool, True{}, Layout.unique(axes), False{}, equal, Equal.sym(Bool, False{}, Layout.unique(axes), duplicate))))

def permutation_duplicate_bounded_failure(+input: Shape.Shape, +axes: List<&2, U32>,
  decision: Shape.BoolCase<Layout.bounded(axes, Shape.rank(input))>, bounded: {True{} == Layout.bounded(axes, Shape.rank(input)) : Bool},
  duplicate: {False{} == Layout.unique(axes) : Bool}) ->
  {Layout.plan_bounded(input, axes, decision) == Fail{Layout.DuplicateAxis{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({Layout.plan_bounded(input, axes, Shape.BFalse{equal}) == Fail{Layout.DuplicateAxis{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>},
        Arith.clash(Equal.trans(Bool, False{}, Layout.bounded(axes, Shape.rank(input)), True{}, equal,
          Equal.sym(Bool, True{}, Layout.bounded(axes, Shape.rank(input)), bounded))))
    case Shape.BTrue{equal}:
      permutation_unique_decision_failure(input, axes, Shape.bool_case(Layout.unique(axes)), duplicate)

def permutation_duplicate_ranked_failure(+input: Shape.Shape, +axes: List<&2, U32>,
  decision: Shape.BoolCase<Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))>,
  ranked: {True{} == Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))) : Bool},
  bounded: {True{} == Layout.bounded(axes, Shape.rank(input)) : Bool}, duplicate: {False{} == Layout.unique(axes) : Bool}) ->
  {Layout.plan_ranked(input, axes, decision) == Fail{Layout.DuplicateAxis{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({Layout.plan_ranked(input, axes, Shape.BFalse{equal}) == Fail{Layout.DuplicateAxis{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>},
        Arith.clash(Equal.trans(Bool, False{}, Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))), True{}, equal,
          Equal.sym(Bool, True{}, Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))), ranked))))
    case Shape.BTrue{equal}:
      permutation_duplicate_bounded_failure(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input))), bounded, duplicate)

def permutation_duplicate_failure(+input: Shape.Shape, +axes: List<&2, U32>,
  ranked: {True{} == Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))) : Bool},
  bounded: {True{} == Layout.bounded(axes, Shape.rank(input)) : Bool}, duplicate: {False{} == Layout.unique(axes) : Bool}) ->
  {Layout.permutation(input, axes) == Fail{Layout.DuplicateAxis{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>}:
  permutation_duplicate_ranked_failure(input, axes,
    Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), ranked, bounded, duplicate)

def permute_failure(+input: Shape.Shape, -source: Core.Tensor<input>, +axes: List<&2, U32>, +error: Layout.Error,
  certificate: {Layout.permutation(input, axes) == Fail{error} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>}) ->
  Totality.PermuteFailure<input, axes, source, error>:
  Totality.PermuteFailure{Equal.cong(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>,
    Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>, result => Transforms.copy_planned(input, axes, source, result),
    Layout.permutation(input, axes), Fail{error}, certificate)}

def Laws.permute_axes_length_rejection(input, source, axes, wrong):
  Totality.PermuteFailure{
    {\{Totality.PermuteFailure: exact => exact} : Totality.PermuteFailure<input, axes, source, Layout.AxesLengthMismatch{}> ->
      {Transforms.permute_copy(input, source, axes) == (source, Fail{Transforms.LayoutError{Layout.AxesLengthMismatch{}}}) :
        Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}}(
          permute_failure(input, source, axes, Layout.AxesLengthMismatch{}, permutation_length_failure(input, axes, wrong)))}

def Laws.permute_axis_bounds_rejection(input, source, axes, ranked, outside):
  Totality.PermuteFailure{
    {\{Totality.PermuteFailure: exact => exact} : Totality.PermuteFailure<input, axes, source, Layout.AxisOutOfBounds{}> ->
      {Transforms.permute_copy(input, source, axes) == (source, Fail{Transforms.LayoutError{Layout.AxisOutOfBounds{}}}) :
        Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}}(
          permute_failure(input, source, axes, Layout.AxisOutOfBounds{}, permutation_bounds_failure(input, axes, ranked, outside)))}

def Laws.permute_duplicate_axes_rejection(input, source, axes, ranked, bounded, duplicate):
  Totality.PermuteFailure{
    {\{Totality.PermuteFailure: exact => exact} : Totality.PermuteFailure<input, axes, source, Layout.DuplicateAxis{}> ->
      {Transforms.permute_copy(input, source, axes) == (source, Fail{Transforms.LayoutError{Layout.DuplicateAxis{}}}) :
        Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}}(
          permute_failure(input, source, axes, Layout.DuplicateAxis{}, permutation_duplicate_failure(input, axes, ranked, bounded, duplicate)))}

type TransposeDecidedFailure<-input: Shape.Shape, -source: Core.Tensor<input>, -a: U32, -b: U32,
  -decision: Shape.BoolCase<TransposeSpec.valid(input, a, b)>> is Data:
  TransposeDecidedFailure{-exact: {Transforms.transpose_decided(input, source, a, b, decision) ==
    (source, Fail{Transforms.LayoutError{Layout.AxisOutOfBounds{}}}) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}}

def transpose_decided_failure(+input: Shape.Shape, +a: U32, +b: U32,
  decision: Shape.BoolCase<TransposeSpec.valid(input, a, b)>, source: Core.Tensor<input>,
  -outside: {False{} == TransposeSpec.valid(input, a, b) : Bool}) ->
  TransposeDecidedFailure<input, source, a, b, decision>:
  match decision:
    case Shape.BFalse{equal}:
      TransposeDecidedFailure{{==}}
    case Shape.BTrue{equal}:
      TransposeDecidedFailure{Empty.absurd({Transforms.transpose_decided(input, source, a, b, Shape.BTrue{equal}) ==
        (source, Fail{Transforms.LayoutError{Layout.AxisOutOfBounds{}}}) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>},
          Lifecycle.false_true(Equal.trans(Bool, True{}, TransposeSpec.valid(input, a, b), False{}, equal,
            Equal.sym(Bool, False{}, TransposeSpec.valid(input, a, b), outside))))}

def Laws.transpose_axis_bounds_rejection(input, source, a, b, outside):
  Totality.TransposeFailure{
    {\{TransposeDecidedFailure: exact => exact} :
      TransposeDecidedFailure<input, source, a, b, Shape.bool_case(TransposeSpec.valid(input, a, b))> ->
        {Transforms.transpose_copy(input, source, a, b) == (source, Fail{Transforms.LayoutError{Layout.AxisOutOfBounds{}}}) :
          Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}}(
            transpose_decided_failure(input, a, b, Shape.bool_case(TransposeSpec.valid(input, a, b)), source, outside))}

def reshape_decided_failure(+input: Shape.Shape, +target: Shape.Shape,
  decision: Shape.BoolCase<Nat.is_eq(Shape.elements(input), Shape.elements(target))>, -source: Core.Tensor<input>,
  different: {False{} == Nat.is_eq(Shape.elements(input), Shape.elements(target)) : Bool}) ->
  {Transforms.reshape_decided(input, target, source, decision) == Fail{Transforms.LayoutError{Layout.ElementCountMismatch{}}} :
    Result<&2, &1, Transforms.Error, Core.Dynamic>}:
  match decision:
    case Shape.BFalse{equal}:
      {==}
    case Shape.BTrue{equal}:
      Empty.absurd({Transforms.reshape_decided(input, target, source, Shape.BTrue{equal}) == Fail{Transforms.LayoutError{Layout.ElementCountMismatch{}}} :
        Result<&2, &1, Transforms.Error, Core.Dynamic>},
          Lifecycle.false_true(Equal.trans(Bool, True{}, Nat.is_eq(Shape.elements(input), Shape.elements(target)), False{}, equal,
            Equal.sym(Bool, False{}, Nat.is_eq(Shape.elements(input), Shape.elements(target)), different))))

def reshape_built_failure(+input: Shape.Shape, +target: Shape.Shape, built: Shape.ConstructResult<Shape.dimensions(target)>, -source: Core.Tensor<input>,
  different: {False{} == Nat.is_eq(Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(target))) : Bool},
  complete: {ShapeLaws.built(Shape.dimensions(target), built) == Some{Shape.dimensions(target)} : Maybe<&2, List<&2, U32>>}) ->
  {Transforms.reshape_built(input, source, Shape.dimensions(target), built) == Fail{Transforms.LayoutError{Layout.ElementCountMismatch{}}} :
    Result<&2, &1, Transforms.Error, Core.Dynamic>}:
  match built:
    case Shape.SError{error}:
      Empty.absurd({Transforms.reshape_built(input, source, Shape.dimensions(target), Shape.SError{error}) ==
        Fail{Transforms.LayoutError{Layout.ElementCountMismatch{}}} : Result<&2, &1, Transforms.Error, Core.Dynamic>},
          Arith.clash(Equal.cong(Maybe<&2, List<&2, U32>>, Bool, Mapping.metadata_present, None{}, Some{Shape.dimensions(target)}, complete)))
    case Shape.SBuilt{+output, preserved}:
      reshape_decided_failure(input, output, Shape.bool_case(Nat.is_eq(Shape.elements(input), Shape.elements(output))), source,
        %Equal.sym(Nat, Shape.elements(input), Shape.logical_numel(Shape.dimensions(input)), Shape.elements_exact(input)) :
          {False{} == Nat.is_eq(_, Shape.elements(output)) : Bool}
        %Equal.sym(Nat, Shape.elements(output), Shape.logical_numel(Shape.dimensions(output)), Shape.elements_exact(output)) :
          {False{} == Nat.is_eq(Shape.logical_numel(Shape.dimensions(input)), _) : Bool}
        %Equal.sym(List<&2, U32>, Shape.dimensions(output), Shape.dimensions(target), preserved) :
          {False{} == Nat.is_eq(Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(_)) : Bool}
        different)

def Laws.reshape_element_count_rejection(input, target, source, different):
  Totality.ReshapeFailure{reshape_built_failure(input, target, Shape.construct(Shape.dimensions(target)), source, different, shape_construct_complete(target))}

def mapping_decided_failure(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  decision: Shape.BoolCase<U32.is_lt(index, Shape.count(Layout.output(input, axes, plan)))>,
  outside: {False{} == U32.is_lt(index, Shape.count(Layout.output(input, axes, plan))) : Bool}) ->
  {Layout.map_decided(input, axes, plan, index, decision) == Fail{Layout.ShapeError{Shape.FlatIndexOutOfBounds{}}} :
    Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>}:
  match decision:
    case Shape.BFalse{equal}:
      {==}
    case Shape.BTrue{equal}:
      Empty.absurd({Layout.map_decided(input, axes, plan, index, Shape.BTrue{equal}) == Fail{Layout.ShapeError{Shape.FlatIndexOutOfBounds{}}} :
        Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>},
          Lifecycle.false_true(Equal.trans(Bool, True{}, U32.is_lt(index, Shape.count(Layout.output(input, axes, plan))), False{}, equal,
            Equal.sym(Bool, False{}, U32.is_lt(index, Shape.count(Layout.output(input, axes, plan))), outside))))

def Laws.mapping_index_rejection(input, axes, plan, index, outside):
  mapping_decided_failure(input, axes, plan, index, Shape.bool_case(U32.is_lt(index, Shape.count(Layout.output(input, axes, plan)))), outside)

def words_naturals(+values: List<&2, U32>) -> {Shape.words(Shape.naturals(values)) == values : List<&2, U32>}:
  match values:
    case Nil{}:
      {==}
    case +head <> +tail:
      %Equal.sym(U32, U32.from_nat(U32.to_nat(head)), head, WordConversion.word_natural_exact(head)) :
        {_ <> Shape.words(Shape.naturals(tail)) == head <> tail : List<&2, U32>}
      Equal.cong(List<&2, U32>, List<&2, U32>, rest => head <> rest, Shape.words(Shape.naturals(tail)), tail, words_naturals(tail))

def digits_offset(+dimensions: List<&2, Nat>, +values: List<&2, Nat>, +fits: Coordinates.Fits(dimensions, values)) ->
  {Shape.digits(dimensions, Shape.offset(dimensions, values)) == values : List<&2, Nat>}:
  %Equal.trans(Nat, Shape.Fin.value(Shape.numel(dimensions), Shape.flatten(dimensions, Coordinates.coordinate_of(dimensions, values, fits))),
    Shape.row_major(dimensions, Coordinates.coordinate_of(dimensions, values, fits)), Shape.offset(dimensions, values),
    ShapeIndex.flatten_row_major(dimensions, Coordinates.coordinate_of(dimensions, values, fits)), Coordinates.offset_of(dimensions, values, fits)) :
      {Shape.digits(dimensions, _) == values : List<&2, Nat>}
  Equal.trans(List<&2, Nat>,
    Shape.digits(dimensions, Shape.Fin.value(Shape.numel(dimensions), Shape.flatten(dimensions, Coordinates.coordinate_of(dimensions, values, fits)))),
    Shape.indices(dimensions, Shape.unflatten(dimensions, Shape.flatten(dimensions, Coordinates.coordinate_of(dimensions, values, fits)))), values,
    Equal.sym(List<&2, Nat>,
      Shape.indices(dimensions, Shape.unflatten(dimensions, Shape.flatten(dimensions, Coordinates.coordinate_of(dimensions, values, fits)))),
      Shape.digits(dimensions, Shape.Fin.value(Shape.numel(dimensions), Shape.flatten(dimensions, Coordinates.coordinate_of(dimensions, values, fits)))),
      Coordinates.digits_unflatten(dimensions, Shape.flatten(dimensions, Coordinates.coordinate_of(dimensions, values, fits)))),
    Equal.trans(List<&2, Nat>,
      Shape.indices(dimensions, Shape.unflatten(dimensions, Shape.flatten(dimensions, Coordinates.coordinate_of(dimensions, values, fits)))),
      Shape.indices(dimensions, Coordinates.coordinate_of(dimensions, values, fits)), values,
      Equal.cong(Shape.Coordinate(dimensions), List<&2, Nat>, Shape.indices(dimensions),
        Shape.unflatten(dimensions, Shape.flatten(dimensions, Coordinates.coordinate_of(dimensions, values, fits))),
        Coordinates.coordinate_of(dimensions, values, fits), ShapeIndex.unflatten_flatten(dimensions, Coordinates.coordinate_of(dimensions, values, fits))),
      Coordinates.indices_of(dimensions, values, fits)))

def Laws.codec_decode_flatten(shape, values, valid):
  Equal.trans(List<&2, U32>,
    Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(shape)), Layout.flatten_words(Shape.dimensions(shape), values, 0))),
    Shape.words(Shape.digits(Shape.extents(shape), U32.to_nat(Layout.flatten_words(Shape.dimensions(shape), values, 0)))), values,
    Mapping.codec_decoded(Shape.dimensions(shape), Layout.flatten_words(Shape.dimensions(shape), values, 0)),
    Equal.trans(List<&2, U32>, Shape.words(Shape.digits(Shape.extents(shape), U32.to_nat(Layout.flatten_words(Shape.dimensions(shape), values, 0)))),
      Shape.words(Shape.naturals(values)), values,
      Equal.cong(List<&2, Nat>, List<&2, U32>, Shape.words,
        Shape.digits(Shape.extents(shape), U32.to_nat(Layout.flatten_words(Shape.dimensions(shape), values, 0))), Shape.naturals(values),
        Equal.trans(List<&2, Nat>, Shape.digits(Shape.extents(shape), U32.to_nat(Layout.flatten_words(Shape.dimensions(shape), values, 0))),
          Shape.digits(Shape.extents(shape), Shape.offset(Shape.extents(shape), Shape.naturals(values))), Shape.naturals(values),
          Equal.cong(Nat, List<&2, Nat>, offset => Shape.digits(Shape.extents(shape), offset), U32.to_nat(Layout.flatten_words(Shape.dimensions(shape), values, 0)),
            Shape.offset(Shape.extents(shape), Shape.naturals(values)), Mapping.coordinate_flatten_exact(shape, values, valid)),
          digits_offset(Shape.extents(shape), Shape.naturals(values), Mapping.coordinate_fits(Shape.dimensions(shape), values, valid)))),
      words_naturals(values)))

def Laws.mapping_valid_success(input, axes, plan, index, inside):
  Totality.MappingSuccess{Mapping.mapping_decided_success(input, axes, plan, index,
    Shape.bool_case(U32.is_lt(index, Shape.count(Layout.output(input, axes, plan)))), inside)}

def Laws.natural_cursor_exact_conversion(shape, cursor, within):
  Transforms.CursorExact{Mapping.cursor_word_exact(shape, cursor, within)}

def copy_written_success(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  pair: Core.Tensor<Layout.output(input, axes, plan)> & Result<&2, &2, Core.Error, Unit>, -source: Core.Tensor<input>,
  -successful: {Totality.write_succeeded(Layout.output(input, axes, plan), pair) == True{} : Bool}) ->
  Totality.CopySuccess<input, axes, plan,
    Transforms.copy_written(input, axes, plan, source => target => (source, Done{target}), source, pair)>:
  (target, result) = pair
  match result:
    case Fail{error}:
      Totality.CopySuccess{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{unit}:
      Totality.CopySuccess{{==}}

def copy_value_success(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  -source: Core.Tensor<input>, target: Core.Tensor<Layout.output(input, axes, plan)>, +value: F32,
  +inside: {True{} == U32.is_lt(index, Shape.count(Layout.output(input, axes, plan))) : Bool}) ->
  Totality.CopySuccess<input, axes, plan, Transforms.copy_read(input, axes, plan, index,
    source => target => (source, Done{target}), target, (source, Done{value}))>:
  copy_written_success(input, axes, plan, Core.write_flat(Layout.output(input, axes, plan), target, index, value), source,
    {\{Totality.WriteSuccess: successful => successful} :
      Totality.WriteSuccess<Layout.output(input, axes, plan), target, index, value> ->
        {Totality.write_succeeded(Layout.output(input, axes, plan), Core.write_flat(Layout.output(input, axes, plan), target, index, value)) == True{} : Bool}}(
          tensor_write_success(Layout.output(input, axes, plan), target, index, value,
            Equal.sym(Bool, True{}, Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))),
              %Shape.elements_exact(Layout.output(input, axes, plan)) :
                {True{} == Nat.is_lt(U32.to_nat(index), _) : Bool}
              %WordConversion.shape_count_exact(Layout.output(input, axes, plan)) :
                {True{} == Nat.is_lt(U32.to_nat(index), _) : Bool}
              %Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), U32.to_nat(Shape.count(Layout.output(input, axes, plan)))),
                U32.is_lt(index, Shape.count(Layout.output(input, axes, plan))), Arith.u32_lt(index, Shape.count(Layout.output(input, axes, plan)))) :
                  {True{} == _ : Bool}
              inside))))

def copy_read_success(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32, +source_index: U32,
  -source: Core.Tensor<input>, target: Core.Tensor<Layout.output(input, axes, plan)>,
  certificate: ReadSuccessModel<input, source, source_index>,
  +inside: {True{} == U32.is_lt(index, Shape.count(Layout.output(input, axes, plan))) : Bool}) ->
  Totality.CopySuccess<input, axes, plan, Transforms.copy_read(input, axes, plan, index,
    source => target => (source, Done{target}), target, Core.read_flat(input, source, source_index))>:
  Totality.CopySuccess{
    {\{ReadSuccessModel: success =>
      {\{ReadSuccess: retained => value => exact =>
        %exact : {Totality.copy_succeeded(input, axes, plan, Transforms.copy_read(input, axes, plan, index,
          source => target => (source, Done{target}), target, _)) == True{} : Bool}
        {\{Totality.CopySuccess: successful => successful} :
          Totality.CopySuccess<input, axes, plan, Transforms.copy_read(input, axes, plan, index,
            source => target => (source, Done{target}), target, (retained, Done{value}))> ->
              {Totality.copy_succeeded(input, axes, plan, Transforms.copy_read(input, axes, plan, index,
                source => target => (source, Done{target}), target, (retained, Done{value}))) == True{} : Bool}}(
                  copy_value_success(input, axes, plan, index, retained, target, value, inside))} :
                    ReadSuccess<input, source, source_index> ->
                      {Totality.copy_succeeded(input, axes, plan, Transforms.copy_read(input, axes, plan, index,
                        source => target => (source, Done{target}), target, Core.read_flat(input, source, source_index))) == True{} : Bool}}(success)} :
                          ReadSuccessModel<input, source, source_index> ->
                            {Totality.copy_succeeded(input, axes, plan, Transforms.copy_read(input, axes, plan, index,
                              source => target => (source, Done{target}), target, Core.read_flat(input, source, source_index))) == True{} : Bool}}(certificate)}

def copy_mapped_success(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  mapped: Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>,
  source: Core.Tensor<input>, target: Core.Tensor<Layout.output(input, axes, plan)>,
  +successful: {Result.is_done(&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>, mapped) == True{} : Bool},
  +inside: {True{} == U32.is_lt(index, Shape.count(Layout.output(input, axes, plan))) : Bool}) ->
  Totality.CopySuccess<input, axes, plan, Transforms.copy_mapped(input, axes, plan, index,
    source => target => (source, Done{target}), source, target, mapped)>:
  match mapped:
    case Fail{error}:
      Totality.CopySuccess{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{+mapping}:
      copy_read_success(input, axes, plan, index, Layout.source_index(input, axes, plan, index, mapping), source, target,
        read_success(input, source, Layout.source_index(input, axes, plan, index, mapping),
          Laws.mapped_mathematical_bounded(input, axes, plan, index, mapping)), inside)

def Laws.copy_step_valid_success(input, axes, plan, source, target, index, inside):
  Totality.CopyStepSuccess{
    {\{Totality.CopySuccess: successful => successful} :
      Totality.CopySuccess<input, axes, plan, Transforms.copy_step(input, axes, plan, source, target, index)> ->
        {Totality.copy_succeeded(input, axes, plan, Transforms.copy_step(input, axes, plan, source, target, index)) == True{} : Bool}}(
          copy_mapped_success(input, axes, plan, index, Layout.map_index(input, axes, plan, index), source, target,
            {\{Totality.MappingSuccess: successful => successful} :
              Totality.MappingSuccess<input, axes, plan, index> ->
                {Result.is_done(&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>, Layout.map_index(input, axes, plan, index)) == True{} : Bool}}(
                  Laws.mapping_valid_success(input, axes, plan, index, inside)), inside))}

def natural_decided_success(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +cursor: Nat, +index: U32,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>, source: Core.Tensor<input>, target: Core.Tensor<Layout.output(input, axes, plan)>,
  -candidate: {index == U32.from_nat(cursor) : U32}, -word_exact: {U32.to_nat(index) == cursor : Nat},
  +inside: {True{} == U32.is_lt(index, Shape.count(Layout.output(input, axes, plan))) : Bool}) ->
  Totality.NaturalSuccess<input, axes, plan, cursor, Transforms.natural_decided(input, axes, plan, cursor, index, source, target, candidate, decision)>:
  match decision:
    case Shape.BFalse{equal}:
      Totality.NaturalSuccess{Empty.absurd({False{} == True{} : Bool},
        Arith.clash(Equal.trans(Bool, False{}, Nat.is_eq(U32.to_nat(index), cursor), True{}, equal, Mapping.natural_equal_true(U32.to_nat(index), cursor, word_exact))))}
    case Shape.BTrue{equal}:
      Totality.NaturalSuccess{
        {\{Totality.CopyStepSuccess: successful => successful} :
          Totality.CopyStepSuccess<input, axes, plan, source, target, index> ->
            {Totality.copy_succeeded(input, axes, plan, Transforms.copy_step(input, axes, plan, source, target, index)) == True{} : Bool}}(
              Laws.copy_step_valid_success(input, axes, plan, source, target, index, inside))}

def Laws.natural_step_valid_success(input, axes, plan, cursor, source, target, inside):
  natural_decided_success(input, axes, plan, cursor, U32.from_nat(cursor), Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)), source, target,
    {==}, Mapping.cursor_word_exact(Layout.output(input, axes, plan), cursor,
      WordConversion.lt_le(cursor, Shape.elements(Layout.output(input, axes, plan)), Equal.sym(Bool, True{}, Nat.is_lt(cursor, Shape.elements(Layout.output(input, axes, plan))), inside))),
    Mapping.cursor_index_bounded(Layout.output(input, axes, plan), cursor, inside))

def add_cursor_le(+fuel: Nat, +cursor: Nat) -> {Nat.is_le(cursor, Nat.add(fuel, cursor)) == True{} : Bool}:
  match fuel:
    case 0n:
      WordConversion.le_reflexive(cursor)
    case 1n+remaining:
      Arith.le_succ(cursor, Nat.add(remaining, cursor), add_cursor_le(remaining, cursor))

def loop_pair_success(+fuel: Nat, +input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +cursor: Nat,
  pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
  -successful: {Totality.copy_succeeded(input, axes, plan, pair) == True{} : Bool},
  rec: @source: Core.Tensor<input> -> @target: Core.Tensor<Layout.output(input, axes, plan)> ->
    Totality.CopySuccess<input, axes, plan, Transforms.copy_loop(fuel, input, axes, plan, source, target, cursor)>) ->
  Totality.CopySuccess<input, axes, plan, Transforms.copy_next(input, axes, plan,
    source => target => Transforms.copy_loop(fuel, input, axes, plan, source, target, cursor), pair)>:
  (source, result) = pair
  match result:
    case Fail{error}:
      Totality.CopySuccess{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{target}:
      rec(source)(target)

def loop_natural_success(+fuel: Nat, +input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +cursor: Nat,
  result: Transforms.NaturalCopy<input, axes, plan, cursor>,
  -successful: {Totality.natural_succeeded(input, axes, plan, cursor, result) == True{} : Bool},
  rec: @source: Core.Tensor<input> -> @target: Core.Tensor<Layout.output(input, axes, plan)> ->
    Totality.CopySuccess<input, axes, plan, Transforms.copy_loop(fuel, input, axes, plan, source, target, 1n+cursor)>) ->
  Totality.CopySuccess<input, axes, plan, Transforms.natural_next(input, axes, plan, cursor,
    source => target => Transforms.copy_loop(fuel, input, axes, plan, source, target, 1n+cursor), result)>:
  match result:
    case Transforms.NaturalError{source}:
      Totality.CopySuccess{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Transforms.NaturalCopied{pair, index, exact, candidate}:
      loop_pair_success(fuel, input, axes, plan, 1n+cursor, pair, successful, rec)

def Laws.copy_loop_valid_success(fuel, input, axes, plan, cursor, source, target, within):
  match fuel:
    case 0n:
      Totality.CopySuccess{{==}}
    case 1n++remaining:
      loop_natural_success(remaining, input, axes, plan, cursor, Transforms.natural_step(input, axes, plan, cursor, source, target),
        {\{Totality.NaturalSuccess: successful => successful} :
          Totality.NaturalSuccess<input, axes, plan, cursor, Transforms.natural_step(input, axes, plan, cursor, source, target)> ->
            {Totality.natural_succeeded(input, axes, plan, cursor, Transforms.natural_step(input, axes, plan, cursor, source, target)) == True{} : Bool}}(
              Laws.natural_step_valid_success(input, axes, plan, cursor, source, target,
                Equal.sym(Bool, Nat.is_lt(cursor, Shape.elements(Layout.output(input, axes, plan))), True{},
                  Arith.lt_le_trans(cursor, 1n+Nat.add(remaining, cursor), Shape.elements(Layout.output(input, axes, plan)),
                    Arith.lt_succ(cursor, Nat.add(remaining, cursor), add_cursor_le(remaining, cursor)), within)))),
        source => target => Laws.copy_loop_valid_success(remaining, input, axes, plan, 1n+cursor, source, target,
          %Equal.sym(Nat, Nat.add(remaining, 1n+cursor), 1n+Nat.add(remaining, cursor), Arith.add_succ(remaining, cursor)) :
            {Nat.is_le(_, Shape.elements(Layout.output(input, axes, plan))) == True{} : Bool}
          within))

def packed_success(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
  -successful: {Totality.copy_succeeded(input, axes, plan, pair) == True{} : Bool}) ->
  Totality.PublicSuccess<input, Transforms.copy_pack(input, axes, plan, pair)>:
  (source, result) = pair
  match result:
    case Fail{error}:
      Totality.PublicSuccess{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{target}:
      Totality.PublicSuccess{{==}}

def allocated_success(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  allocated: Result<&2, &1, Core.Error, Core.Tensor<Layout.output(input, axes, plan)>>, source: Core.Tensor<input>,
  -certificate: AllocationSpec.TensorDone<Layout.output(input, axes, plan), allocated>) ->
  Totality.PublicSuccess<input, Transforms.copy_allocated(input, axes, plan, source, allocated)>:
  match allocated:
    case Fail{error}:
      Totality.PublicSuccess{
        {\{AllocationSpec.TensorDone: tensor => same =>
          Empty.absurd({False{} == True{} : Bool}, Lifecycle.false_true(Equal.cong(Result<&2, &1, Core.Error, Core.Tensor<Layout.output(input, axes, plan)>>, Bool,
            Result.is_done(&2, &1, Core.Error, Core.Tensor<Layout.output(input, axes, plan)>), Done{tensor}, Fail{error}, same)))} :
              AllocationSpec.TensorDone<Layout.output(input, axes, plan), Fail{error}> -> {False{} == True{} : Bool}}(certificate)}
    case Done{target}:
      packed_success(input, axes, plan, Transforms.built(input, axes, plan, source, target),
        {\{PermuteBuild.Fact: successful => successful} :
          PermuteBuild.Fact<{Totality.copy_succeeded(input, axes, plan, Transforms.built(input, axes, plan, source, target)) == True{} : Bool}> ->
            {Totality.copy_succeeded(input, axes, plan, Transforms.built(input, axes, plan, source, target)) == True{} : Bool}}(
              PermuteBuild.built_success(input, axes, plan, source, target)))

def planned_success(+input: Shape.Shape, +axes: List<&2, U32>, planned: Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, source: Core.Tensor<input>,
  -successful: {Result.is_done(&2, &2, Layout.Error, Layout.Plan<input, axes>, planned) == True{} : Bool}) ->
  Totality.PublicSuccess<input, Transforms.copy_planned(input, axes, source, planned)>:
  match planned:
    case Fail{error}:
      Totality.PublicSuccess{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{+plan}:
      allocated_success(input, axes, plan, Core.new(Layout.output(input, axes, plan), 0.0), source,
        AllocationProof.core_new_done(Layout.output(input, axes, plan), 0.0))

def Laws.permute_valid_success(input, source, axes, valid):
  planned_success(input, axes, Layout.permutation(input, axes), source,
    {\{Totality.PlanSuccess: successful => successful} : Totality.PlanSuccess<input, axes> ->
      {Result.is_done(&2, &2, Layout.Error, Layout.Plan<input, axes>, Layout.permutation(input, axes)) == True{} : Bool}}(
        Laws.permutation_valid_success(input, axes, valid)))

def words_bounded_exact(+values: List<&2, Nat>, +rank: U32,
  +bounded: {True{} == Axis.bounded(values, U32.to_nat(rank)) : Bool}) ->
  {Shape.naturals(Shape.words(values)) == values : List<&2, Nat>}:
  match values:
    case Nil{}:
      {==}
    case +head <> +tail:
      %Equal.sym(Nat, U32.to_nat(U32.from_nat(head)), head,
        WordConversion.natural_word_exact(head, rank,
          WordConversion.lt_le(head, U32.to_nat(rank), Equal.sym(Bool, True{}, Nat.is_lt(head, U32.to_nat(rank)),
            Mapping.and_left(Nat.is_lt(head, U32.to_nat(rank)), Axis.bounded(tail, U32.to_nat(rank)), bounded))))) :
              {_ <> Shape.naturals(Shape.words(tail)) == head <> tail : List<&2, Nat>}
      Equal.cong(List<&2, Nat>, List<&2, Nat>, rest => head <> rest, Shape.naturals(Shape.words(tail)), tail,
        words_bounded_exact(tail, rank, Mapping.and_right(Nat.is_lt(head, U32.to_nat(rank)), Axis.bounded(tail, U32.to_nat(rank)), bounded)))

def inverse_axes_naturals(+input: Shape.Shape, +axes: List<&2, U32>, +accepted: {True{} == Layout.bijective(input, axes) : Bool}) ->
  {Shape.naturals(Totality.inverse_axes(input, axes)) == Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)) : List<&2, Nat>}:
  words_bounded_exact(Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)), Shape.rank(input),
    %Equal.sym(Nat, U32.to_nat(Shape.rank(input)), Shape.rank_dimensions(Shape.dimensions(input)), AxisAlgebra.rank_exact(input)) :
      {True{} == Axis.bounded(Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)), _) : Bool}
    Mapping.and_left(Axis.bounded(Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)), Shape.rank_dimensions(Shape.dimensions(input))),
      Axis.unique(Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))),
      Mapping.and_right(Nat.is_eq(List.length(&2, Nat, Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))), Shape.rank_dimensions(Shape.dimensions(input))),
        Bool.and(Axis.bounded(Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)), Shape.rank_dimensions(Shape.dimensions(input))),
          Axis.unique(Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)))),
        AxisAlgebra.facts_inverse_valid(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes), AxisAlgebra.shape_facts(input, axes, accepted)))))

def native_axes_valid(+shape: Shape.Shape, +axes: List<&2, U32>,
  accepted: {True{} == Axis.valid(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes)) : Bool}) ->
  {True{} == Layout.bijective(shape, axes) : Bool}:
  %Equal.sym(Nat, Shape.rank_dimensions(axes), List.length(&2, Nat, Shape.naturals(axes)), Shape.rank_naturals(axes)) :
    {True{} == Bool.and(Nat.is_eq(_, Shape.rank_dimensions(Shape.dimensions(shape))), Bool.and(Layout.bounded(axes, Shape.rank(shape)), Layout.unique(axes))) : Bool}
  %Equal.sym(Bool, Layout.unique(axes), Axis.unique(Shape.naturals(axes)), AxisAlgebra.unique(axes)) :
    {True{} == Bool.and(Nat.is_eq(List.length(&2, Nat, Shape.naturals(axes)), Shape.rank_dimensions(Shape.dimensions(shape))), Bool.and(Layout.bounded(axes, Shape.rank(shape)), _)) : Bool}
  %Equal.sym(Bool, Layout.bounded(axes, Shape.rank(shape)), Axis.bounded(Shape.naturals(axes), U32.to_nat(Shape.rank(shape))), AxisAlgebra.bounded(axes, Shape.rank(shape))) :
    {True{} == Bool.and(Nat.is_eq(List.length(&2, Nat, Shape.naturals(axes)), Shape.rank_dimensions(Shape.dimensions(shape))), Bool.and(_, Axis.unique(Shape.naturals(axes)))) : Bool}
  %Equal.sym(Nat, U32.to_nat(Shape.rank(shape)), Shape.rank_dimensions(Shape.dimensions(shape)), AxisAlgebra.rank_exact(shape)) :
    {True{} == Bool.and(Nat.is_eq(List.length(&2, Nat, Shape.naturals(axes)), Shape.rank_dimensions(Shape.dimensions(shape))), Bool.and(Axis.bounded(Shape.naturals(axes), _), Axis.unique(Shape.naturals(axes)))) : Bool}
  accepted

def Laws.inverse_axes_valid(input, axes, plan):
  native_axes_valid(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes),
    %Equal.sym(List<&2, Nat>, Shape.naturals(Totality.inverse_axes(input, axes)), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)),
      inverse_axes_naturals(input, axes, Laws.permutation_checked(input, axes, plan))) :
        {True{} == Axis.valid(Shape.rank_dimensions(Shape.dimensions(Layout.output(input, axes, plan))), _) : Bool}
    %Equal.sym(Nat, Shape.rank_dimensions(Shape.dimensions(Layout.output(input, axes, plan))), Shape.rank_dimensions(Shape.dimensions(input)), Mapping.permutation_output_rank(input, axes, plan)) :
      {True{} == Axis.valid(_, Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))) : Bool}
    AxisAlgebra.facts_inverse_valid(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes),
      AxisAlgebra.shape_facts(input, axes, Laws.permutation_checked(input, axes, plan))))

def Laws.inverse_permutation_dimensions(input, axes, plan, inverse):
  Equal.trans(List<&2, U32>, Shape.dimensions(Layout.output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse)),
    Axis.order(U32, Shape.dimensions(Layout.output(input, axes, plan)), Shape.naturals(Totality.inverse_axes(input, axes))), Shape.dimensions(input),
    Mapping.permutation_dimensions_order(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse),
    %Equal.sym(List<&2, U32>, Shape.dimensions(Layout.output(input, axes, plan)), Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)),
      Mapping.permutation_dimensions_order(input, axes, plan)) :
        {Axis.order(U32, _, Shape.naturals(Totality.inverse_axes(input, axes))) == Shape.dimensions(input) : List<&2, U32>}
    %Equal.sym(List<&2, Nat>, Shape.naturals(Totality.inverse_axes(input, axes)), Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)),
      inverse_axes_naturals(input, axes, Laws.permutation_checked(input, axes, plan))) :
        {Axis.order(U32, Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)), _) == Shape.dimensions(input) : List<&2, U32>}
    AxisAlgebra.inverse_composition(U32, Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes), Shape.dimensions(input), Mapping.shape_rank_bound(input),
      AxisAlgebra.valid(input, axes, Laws.permutation_checked(input, axes, plan)), AxisAlgebra.word_length(Shape.dimensions(input))))

def mapping_input_valid(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  mapping: Layout.Mapping<input, axes, plan, index>) -> Layout.WordsValid(Shape.dimensions(input), Layout.input_words(input, axes, plan, index, mapping)):
  Layout.Mapped{oi, oc, ic, source, requested, decoded, inverse, flattened, roundtrip} = mapping
  Layout.CWord{values, valid} = ic
  valid

def mapping_input_order(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  mapping: Layout.Mapping<input, axes, plan, index>) ->
  {Layout.input_words(input, axes, plan, index, mapping) == Axis.order(U32, Layout.output_words(input, axes, plan, index, mapping),
    Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))) : List<&2, U32>}:
  Layout.Mapped{oi, oc, ic, source, requested, decoded, inverse, flattened, roundtrip} = mapping
  Mapping.inverse_words_order(input, axes, plan, oc, Layout.coordinate_values(input, ic), inverse)

def mapping_output_decoded(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  mapping: Layout.Mapping<input, axes, plan, index>) ->
  {Layout.output_words(input, axes, plan, index, mapping) == Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(Layout.output(input, axes, plan))), index)) : List<&2, U32>}:
  Layout.Mapped{oi, oc, ic, source, requested, decoded, inverse, flattened, roundtrip} = mapping
  decoded

def mapping_output_mathematical_bounded(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  mapping: Layout.Mapping<input, axes, plan, index>) ->
  {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool}:
  Layout.Mapped{oi, oc, ic, source, requested, decoded, inverse, flattened, roundtrip} = mapping
  %requested : {Nat.is_lt(U32.to_nat(_), Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool}
  Mapping.flat_mathematical_bounded(Layout.output(input, axes, plan), oi)

def mapping_order_forward(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  +mapping: Layout.Mapping<input, axes, plan, index>) ->
  {Axis.order(U32, Layout.input_words(input, axes, plan, index, mapping), Shape.naturals(axes)) == Layout.output_words(input, axes, plan, index, mapping) : List<&2, U32>}:
  Equal.cong(Maybe<&2, List<&2, U32>>, List<&2, U32>, Axis.values(U32),
    Axis.gather(U32, Layout.input_words(input, axes, plan, index, mapping), Shape.naturals(axes)), Some{Layout.output_words(input, axes, plan, index, mapping)},
    Equal.trans(Maybe<&2, List<&2, U32>>, Axis.gather(U32, Layout.input_words(input, axes, plan, index, mapping), Shape.naturals(axes)),
      Layout.reorder(Layout.input_words(input, axes, plan, index, mapping), axes), Some{Layout.output_words(input, axes, plan, index, mapping)},
      Equal.sym(Maybe<&2, List<&2, U32>>, Layout.reorder(Layout.input_words(input, axes, plan, index, mapping), axes),
        Axis.gather(U32, Layout.input_words(input, axes, plan, index, mapping), Shape.naturals(axes)), AxisAlgebra.reorder(axes, Layout.input_words(input, axes, plan, index, mapping))),
      Laws.inverse_coordinate_roundtrip(input, axes, plan, index, mapping)))

def mapping_coordinates_join(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  +inverse: Layout.Plan<Layout.output(input, axes, plan), Totality.inverse_axes(input, axes)>, +index: U32,
  +second: Layout.Mapping<Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index>,
  +first: Layout.Mapping<input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)>) ->
  {Layout.output_words(input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), first) ==
    Layout.input_words(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second) : List<&2, U32>}:
  Equal.trans(List<&2, U32>, Layout.output_words(input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), first),
    Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(Layout.output(input, axes, plan))),
      Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second))),
    Layout.input_words(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second),
    mapping_output_decoded(input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), first),
    %Equal.sym(U32, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second),
      Layout.flatten_words(Shape.dimensions(Layout.output(input, axes, plan)), Layout.input_words(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), 0),
      Laws.mapped_flat(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)) :
        {Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(Layout.output(input, axes, plan))), _)) ==
          Layout.input_words(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second) : List<&2, U32>}
    Laws.codec_decode_flatten(Layout.output(input, axes, plan), Layout.input_words(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second),
      mapping_input_valid(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)))

def inverse_mapping_coordinates(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  +inverse: Layout.Plan<Layout.output(input, axes, plan), Totality.inverse_axes(input, axes)>, +index: U32,
  +second: Layout.Mapping<Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index>,
  +first: Layout.Mapping<input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)>) ->
  {Layout.input_words(input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), first) ==
    Layout.output_words(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second) : List<&2, U32>}:
  Equal.trans(List<&2, U32>, Layout.input_words(input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), first),
    Axis.order(U32, Layout.output_words(input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), first),
      Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))),
    Layout.output_words(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second),
    mapping_input_order(input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), first),
    %Equal.sym(List<&2, U32>, Layout.output_words(input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), first),
      Layout.input_words(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), mapping_coordinates_join(input, axes, plan, inverse, index, second, first)) :
        {Axis.order(U32, _, Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes))) ==
          Layout.output_words(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second) : List<&2, U32>}
    %inverse_axes_naturals(input, axes, Laws.permutation_checked(input, axes, plan)) :
      {Axis.order(U32, Layout.input_words(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), _) ==
        Layout.output_words(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second) : List<&2, U32>}
    mapping_order_forward(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second))

def Laws.inverse_mapping_restores_index(input, axes, plan, inverse, index, second, first):
  Equal.trans(U32, Layout.source_index(input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), first),
    Layout.flatten_words(Shape.dimensions(input), Layout.input_words(input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), first), 0), index,
    Laws.mapped_flat(input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), first),
    %Equal.sym(List<&2, U32>, Layout.input_words(input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), first),
      Layout.output_words(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), inverse_mapping_coordinates(input, axes, plan, inverse, index, second, first)) :
        {Layout.flatten_words(Shape.dimensions(input), _, 0) == index : U32}
    %Equal.sym(List<&2, U32>, Layout.output_words(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second),
      Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(Shape.dimensions(Layout.output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse))), index)),
      mapping_output_decoded(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)) :
        {Layout.flatten_words(Shape.dimensions(input), _, 0) == index : U32}
    %Equal.sym(List<&2, U32>, Shape.dimensions(Layout.output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse)), Shape.dimensions(input),
      Laws.inverse_permutation_dimensions(input, axes, plan, inverse)) :
        {Layout.flatten_words(Shape.dimensions(input), Layout.reverse_words(Layout.unflatten_words(Layout.reverse_words(_), index)), 0) == index : U32}
    Laws.codec_flatten_decode(input, index,
      %Equal.sym(Nat, Shape.elements(input), Shape.logical_numel(Shape.dimensions(input)), Shape.elements_exact(input)) :
        {True{} == Nat.is_lt(U32.to_nat(index), _) : Bool}
      %Laws.inverse_permutation_dimensions(input, axes, plan, inverse) :
        {True{} == Nat.is_lt(U32.to_nat(index), Shape.logical_numel(_)) : Bool}
      Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Layout.output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse)))), True{},
        mapping_output_mathematical_bounded(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second))))

def typed_permutation_impossible(-input: Shape.Shape, -axes: List<&2, U32>, -original: Core.Tensor<input>,
  -pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>, -empty: Empty) -> Totality.TypedPermutation<input, axes, original, pair>:
  Totality.TypedPermutation{Empty.absurd(Layout.Plan<input, axes>, empty), Empty.absurd(Core.Tensor<input>, empty),
    Empty.absurd(Core.Tensor<Layout.output(input, axes, Empty.absurd(Layout.Plan<input, axes>, empty))>, empty),
    Empty.absurd({(Empty.absurd(Core.Tensor<input>, empty), Done{Core.Packed{Layout.output(input, axes, Empty.absurd(Layout.Plan<input, axes>, empty)),
      Empty.absurd(Core.Tensor<Layout.output(input, axes, Empty.absurd(Layout.Plan<input, axes>, empty))>, empty)}}) == pair :
        Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}, empty),
    Empty.absurd(Transforms.Prefix<input, axes, Empty.absurd(Layout.Plan<input, axes>, empty), original, Empty.absurd(Core.Tensor<input>, empty),
      Empty.absurd(Core.Tensor<Layout.output(input, axes, Empty.absurd(Layout.Plan<input, axes>, empty))>, empty),
      Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, Empty.absurd(Layout.Plan<input, axes>, empty))))>, empty)}

def typed_built_result(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, -original: Core.Tensor<input>,
  -initial_target: Core.Tensor<Layout.output(input, axes, plan)>,
  pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
  -exact: {pair == Transforms.built(input, axes, plan, original, initial_target) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>},
  -meaning: Transforms.LoopMeaning(input, axes, plan, original, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), pair),
  -successful: {Totality.copy_succeeded(input, axes, plan, pair) == True{} : Bool}) ->
  Totality.TypedPermutation<input, axes, original, Transforms.copy_pack(input, axes, plan, Transforms.built(input, axes, plan, original, initial_target))>:
  (retained, result) = pair
  match result:
    case Fail{error}:
      typed_permutation_impossible(input, axes, original, Transforms.copy_pack(input, axes, plan,
        Transforms.built(input, axes, plan, original, initial_target)), Arith.clash(successful))
    case Done{target}:
      Totality.TypedPermutation{plan, retained, target,
        Equal.cong(Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>,
          Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>, Transforms.copy_pack(input, axes, plan),
          (retained, Done{target}), Transforms.built(input, axes, plan, original, initial_target), exact),
        meaning}

def typed_allocated_success(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  allocated: Result<&2, &1, Core.Error, Core.Tensor<Layout.output(input, axes, plan)>>, source: Core.Tensor<input>,
  -certificate: AllocationSpec.TensorDone<Layout.output(input, axes, plan), allocated>) ->
  Totality.TypedPermutation<input, axes, source, Transforms.copy_allocated(input, axes, plan, source, allocated)>:
  match allocated:
    case Fail{error}:
      typed_permutation_impossible(input, axes, source, Transforms.copy_allocated(input, axes, plan, source, Fail{error}),
        {\{AllocationSpec.TensorDone: tensor => same =>
          Lifecycle.false_true(Equal.cong(Result<&2, &1, Core.Error, Core.Tensor<Layout.output(input, axes, plan)>>, Bool,
            Result.is_done(&2, &1, Core.Error, Core.Tensor<Layout.output(input, axes, plan)>), Done{tensor}, Fail{error}, same))} :
              AllocationSpec.TensorDone<Layout.output(input, axes, plan), Fail{error}> -> Empty}(certificate))
    case Done{target}:
      typed_built_result(input, axes, plan, source, target, Transforms.built(input, axes, plan, source, target), {==},
        PermuteBuild.built_meaning(input, axes, plan, source, target),
        {\{PermuteBuild.Fact: successful => successful} :
          PermuteBuild.Fact<{Totality.copy_succeeded(input, axes, plan, Transforms.built(input, axes, plan, source, target)) == True{} : Bool}> ->
            {Totality.copy_succeeded(input, axes, plan, Transforms.built(input, axes, plan, source, target)) == True{} : Bool}}(
              PermuteBuild.built_success(input, axes, plan, source, target)))

def typed_planned_success(+input: Shape.Shape, +axes: List<&2, U32>, planned: Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, source: Core.Tensor<input>,
  -successful: {Result.is_done(&2, &2, Layout.Error, Layout.Plan<input, axes>, planned) == True{} : Bool}) ->
  Totality.TypedPermutation<input, axes, source, Transforms.copy_planned(input, axes, source, planned)>:
  match planned:
    case Fail{error}:
      typed_permutation_impossible(input, axes, source, Transforms.copy_planned(input, axes, source, Fail{error}), Arith.clash(successful))
    case Done{+plan}:
      typed_allocated_success(input, axes, plan, Core.new(Layout.output(input, axes, plan), 0.0), source,
        AllocationProof.core_new_done(Layout.output(input, axes, plan), 0.0))

def Laws.permute_typed_success(input, source, axes, valid):
  typed_planned_success(input, axes, Layout.permutation(input, axes), source,
    {\{Totality.PlanSuccess: successful => successful} : Totality.PlanSuccess<input, axes> ->
      {Result.is_done(&2, &2, Layout.Error, Layout.Plan<input, axes>, Layout.permutation(input, axes)) == True{} : Bool}}(
        Laws.permutation_valid_success(input, axes, valid)))

def logical_index_bounded(+shape: Shape.Shape, +index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {True{} == U32.is_lt(index, Shape.count(shape)) : Bool}:
  %Arith.u32_lt(index, Shape.count(shape)) : {True{} == _ : Bool}
  %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.elements(shape), WordConversion.shape_count_exact(shape)) : {True{} == Nat.is_lt(U32.to_nat(index), _) : Bool}
  %Equal.sym(Nat, Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)), Shape.elements_exact(shape)) : {True{} == Nat.is_lt(U32.to_nat(index), _) : Bool}
  Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))), True{}, inside)

def mapping_done_result(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  result: Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>,
  -successful: {Result.is_done(&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>, result) == True{} : Bool}) ->
  Totality.MappingDone<input, axes, plan, index, result>:
  match result:
    case Fail{error}:
      Totality.MappingDone{Empty.absurd(Layout.Mapping<input, axes, plan, index>, Arith.clash(successful)),
        Empty.absurd({Done{Empty.absurd(Layout.Mapping<input, axes, plan, index>, Arith.clash(successful))} == Fail{error} :
          Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>}, Arith.clash(successful))}
    case Done{mapping}:
      Totality.MappingDone{mapping, {==}}

def mapping_done(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  +inside: {True{} == U32.is_lt(index, Shape.count(Layout.output(input, axes, plan))) : Bool}) ->
  Totality.MappingDone<input, axes, plan, index, Layout.map_index(input, axes, plan, index)>:
  mapping_done_result(input, axes, plan, index, Layout.map_index(input, axes, plan, index),
    {\{Totality.MappingSuccess: successful => successful} : Totality.MappingSuccess<input, axes, plan, index> ->
      {Result.is_done(&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>, Layout.map_index(input, axes, plan, index)) == True{} : Bool}}(
        Laws.mapping_valid_success(input, axes, plan, index, inside)))

def prefix_observation(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  -original: Core.Tensor<input>, -retained: Core.Tensor<input>, -target: Core.Tensor<Layout.output(input, axes, plan)>, +index: U32,
  -prefix: Transforms.Prefix<input, axes, plan, original, retained, target, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))>,
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool}) ->
  Totality.CopyObservation<input, axes, plan, original, target, index>:
  Totality.CopyObservation{
    {\{Transforms.Prefix: kept => words => checked => written =>
      Equal.trans(Maybe<&2, F32>, Core.observe(Layout.output(input, axes, plan), target, index), Core.checked_observe(Layout.output(input, axes, plan), target, index),
        Transforms.expected_output(input, axes, plan, original, index),
        {\{ObserveBridge: equal => Equal.sym(Maybe<&2, F32>, Core.checked_observe(Layout.output(input, axes, plan), target, index), Core.observe(Layout.output(input, axes, plan), target, index), equal)} :
          ObserveBridge<Layout.output(input, axes, plan), target, index> ->
            {Core.observe(Layout.output(input, axes, plan), target, index) == Core.checked_observe(Layout.output(input, axes, plan), target, index) : Maybe<&2, F32>}}(
              observe_bridge(Layout.output(input, axes, plan), target, index,
                %WordConversion.word_natural_exact(index) : Core.ObservationChecked(Core.Packed{Layout.output(input, axes, plan), target}, _)
                checked(U32.to_nat(index), inside))),
        %WordConversion.word_natural_exact(index) :
          {Core.checked_observe(Layout.output(input, axes, plan), target, _) == Transforms.expected_output(input, axes, plan, original, _) : Maybe<&2, F32>}
        written(U32.to_nat(index), inside))} :
          Transforms.Prefix<input, axes, plan, original, retained, target, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))> ->
            {Core.observe(Layout.output(input, axes, plan), target, index) == Transforms.expected_output(input, axes, plan, original, index) : Maybe<&2, F32>}}(prefix)}

def logical_observe_bridge(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +index: U32,
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) -> ObserveBridge<shape, tensor, index>:
  ObserveBridge{
    {\{CheckedObservationModel: checked =>
      {\{ObserveBridge: equal => equal} : ObserveBridge<shape, tensor, index> ->
        {Core.checked_observe(shape, tensor, index) == Core.observe(shape, tensor, index) : Maybe<&2, F32>}}(
          observe_bridge(shape, tensor, index, checked))} : CheckedObservationModel<shape, tensor, index> ->
            {Core.checked_observe(shape, tensor, index) == Core.observe(shape, tensor, index) : Maybe<&2, F32>}}(
              checked_observation_model(shape, tensor, index, inside))}

def inverse_observation_mapped(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  +inverse: Layout.Plan<Layout.output(input, axes, plan), Totality.inverse_axes(input, axes)>,
  -original: Core.Tensor<input>, -retained: Core.Tensor<input>, -middle: Core.Tensor<Layout.output(input, axes, plan)>,
  -middle_retained: Core.Tensor<Layout.output(input, axes, plan)>,
  -restored: Core.Tensor<Layout.output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse)>, +index: U32,
  +second: Layout.Mapping<Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index>,
  -second_exact: {Done{second} == Layout.map_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index) :
    Result<&2, &2, Layout.Error, Layout.Mapping<Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index>>},
  +first: Layout.Mapping<input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)>,
  -first_exact: {Done{first} == Layout.map_index(input, axes, plan,
    Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)) :
      Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)>>},
  -first_prefix: Transforms.Prefix<input, axes, plan, original, retained, middle, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))>,
  -second_prefix: Transforms.Prefix<Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, middle, middle_retained, restored,
    Shape.logical_numel(Shape.dimensions(Layout.output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse)))>,
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool}) ->
  Totality.RestoreObservation<input, Layout.output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse), original, restored, index>:
  Totality.RestoreObservation{
  Equal.trans(Maybe<&2, F32>, Core.observe(Layout.output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse), restored, index),
    Transforms.expected_output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, middle, index), Core.observe(input, original, index),
    {\{Totality.CopyObservation: equal => equal} :
      Totality.CopyObservation<Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, middle, restored, index> ->
        {Core.observe(Layout.output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse), restored, index) ==
          Transforms.expected_output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, middle, index) : Maybe<&2, F32>}}(
            prefix_observation(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, middle, middle_retained, restored, index, second_prefix,
              %Equal.sym(List<&2, U32>, Shape.dimensions(Layout.output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse)), Shape.dimensions(input),
                Laws.inverse_permutation_dimensions(input, axes, plan, inverse)) : {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(_)) == True{} : Bool}
              inside)),
    %second_exact : {Transforms.expected_mapped(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, middle, _) == Core.observe(input, original, index) : Maybe<&2, F32>}
    Equal.trans(Maybe<&2, F32>, Core.checked_observe(Layout.output(input, axes, plan), middle,
      Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)),
      Core.observe(Layout.output(input, axes, plan), middle, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)), Core.observe(input, original, index),
      {\{ObserveBridge: equal => equal} : ObserveBridge<Layout.output(input, axes, plan), middle,
        Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)> ->
          {Core.checked_observe(Layout.output(input, axes, plan), middle, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)) ==
            Core.observe(Layout.output(input, axes, plan), middle, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)) : Maybe<&2, F32>}}(
              logical_observe_bridge(Layout.output(input, axes, plan), middle, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second),
                Laws.mapped_mathematical_bounded(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second))),
      Equal.trans(Maybe<&2, F32>, Core.observe(Layout.output(input, axes, plan), middle, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)),
        Transforms.expected_output(input, axes, plan, original, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)), Core.observe(input, original, index),
        {\{Totality.CopyObservation: equal => equal} : Totality.CopyObservation<input, axes, plan, original, middle,
          Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)> ->
            {Core.observe(Layout.output(input, axes, plan), middle, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)) ==
              Transforms.expected_output(input, axes, plan, original, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)) : Maybe<&2, F32>}}(
                prefix_observation(input, axes, plan, original, retained, middle,
                  Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), first_prefix,
                  Laws.mapped_mathematical_bounded(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second))),
        %first_exact : {Transforms.expected_mapped(input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), original, _) == Core.observe(input, original, index) : Maybe<&2, F32>}
        %Equal.sym(U32, Layout.source_index(input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second), first), index,
          Laws.inverse_mapping_restores_index(input, axes, plan, inverse, index, second, first)) : {Core.checked_observe(input, original, _) == Core.observe(input, original, index) : Maybe<&2, F32>}
        {\{ObserveBridge: equal => equal} : ObserveBridge<input, original, index> ->
          {Core.checked_observe(input, original, index) == Core.observe(input, original, index) : Maybe<&2, F32>}}(
            logical_observe_bridge(input, original, index, inside)))))}

def inverse_restoration_second(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, -original: Core.Tensor<input>, -retained: Core.Tensor<input>,
  -middle: Core.Tensor<Layout.output(input, axes, plan)>,
  -first_exact: {(retained, Done{Core.Packed{Layout.output(input, axes, plan), middle}}) == Transforms.permute_copy(input, original, axes) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>},
  -first_prefix: Transforms.Prefix<input, axes, plan, original, retained, middle, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))>,
  certificate: Totality.TypedPermutation<Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), middle,
    Transforms.permute_copy(Layout.output(input, axes, plan), middle, Totality.inverse_axes(input, axes))>) -> Totality.PermutationRestoration<input, axes, original>:
  Totality.TypedPermutation{+inverse, middle_retained, restored, second_exact, second_prefix} = certificate
  Totality.PermutationRestoration{plan, retained, middle, first_exact, inverse, middle_retained, restored,
    Equal.cong(Core.Tensor<Layout.output(input, axes, plan)> & Result<&2, &1, Transforms.Error, Core.Dynamic>, Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>,
      Transforms.dynamic_result(Layout.output(input, axes, plan)), (middle_retained, Done{Core.Packed{Layout.output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse), restored}}),
      Transforms.permute_copy(Layout.output(input, axes, plan), middle, Totality.inverse_axes(input, axes)), second_exact),
    Laws.inverse_permutation_dimensions(input, axes, plan, inverse),
    index => inside =>
      {\{Totality.MappingDone: second => second_exact =>
        {\{Totality.MappingDone: first => first_mapping_exact =>
          {\{Totality.RestoreObservation: equal => equal} :
            Totality.RestoreObservation<input, Layout.output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse), original, restored, index> ->
              {Core.observe(Layout.output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse), restored, index) == Core.observe(input, original, index) : Maybe<&2, F32>}}(
                inverse_observation_mapped(input, axes, plan, inverse, original, retained, middle, middle_retained, restored, index,
                  second, second_exact, first, first_mapping_exact, first_prefix, second_prefix, inside))} :
            Totality.MappingDone<input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second),
              Layout.map_index(input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second))> ->
                {Core.observe(Layout.output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse), restored, index) == Core.observe(input, original, index) : Maybe<&2, F32>}}(
                  mapping_done(input, axes, plan, Layout.source_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second),
                    Laws.mapped_bounded(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index, second)))} :
                      Totality.MappingDone<Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index,
                        Layout.map_index(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index)> ->
                          {Core.observe(Layout.output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse), restored, index) == Core.observe(input, original, index) : Maybe<&2, F32>}}(
                            mapping_done(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse, index,
                              logical_index_bounded(Layout.output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse), index,
                                %Equal.sym(List<&2, U32>, Shape.dimensions(Layout.output(Layout.output(input, axes, plan), Totality.inverse_axes(input, axes), inverse)), Shape.dimensions(input),
                                  Laws.inverse_permutation_dimensions(input, axes, plan, inverse)) : {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(_)) == True{} : Bool}
                                inside)))}

def Laws.permute_inverse_restoration(input, source, axes, valid):
  Totality.PermutationInverse{
    {\{Totality.TypedPermutation: plan => retained => middle => first_exact => first_prefix =>
      inverse_restoration_second(input, axes, plan, source, retained, middle, first_exact, first_prefix,
        Laws.permute_typed_success(Layout.output(input, axes, plan), middle, Totality.inverse_axes(input, axes), Laws.inverse_axes_valid(input, axes, plan)))} :
          Totality.TypedPermutation<input, axes, source, Transforms.permute_copy(input, source, axes)> -> Totality.PermutationRestoration<input, axes, source>}(
            Laws.permute_typed_success(input, source, axes, valid))}

def transpose_decided_permutation(+input: Shape.Shape, +a: U32, +b: U32,
  decision: Shape.BoolCase<TransposeSpec.valid(input, a, b)>, source: Core.Tensor<input>,
  +inside: {True{} == TransposeSpec.valid(input, a, b) : Bool}) ->
  {Transforms.transpose_decided(input, source, a, b, decision) == Transforms.permute_copy(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)) :
    Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({(source, Fail{Transforms.LayoutError{Layout.AxisOutOfBounds{}}}) ==
        Transforms.permute_copy(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>},
          Arith.clash(Equal.trans(Bool, False{}, TransposeSpec.valid(input, a, b), True{}, equal,
            Equal.sym(Bool, True{}, TransposeSpec.valid(input, a, b), inside))))
    case Shape.BTrue{equal}:
      {==}

def transpose_permutation(+input: Shape.Shape, source: Core.Tensor<input>, +a: U32, +b: U32,
  +inside: {True{} == TransposeSpec.valid(input, a, b) : Bool}) ->
  {Transforms.transpose_copy(input, source, a, b) == Transforms.permute_copy(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)) :
    Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}:
  transpose_decided_permutation(input, a, b, Shape.bool_case(TransposeSpec.valid(input, a, b)), source, inside)

def Laws.transpose_valid_success(input, source, a, b, inside):
  Totality.PublicSuccess{
    %Equal.sym(Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>, Transforms.transpose_copy(input, source, a, b),
      Transforms.permute_copy(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)), transpose_permutation(input, source, a, b, inside)) :
        {Totality.public_succeeded(input, _) == True{} : Bool}
    {\{Totality.PublicSuccess: successful => successful} :
      Totality.PublicSuccess<input, Transforms.permute_copy(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0))> ->
        {Totality.public_succeeded(input, Transforms.permute_copy(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0))) == True{} : Bool}}(
          Laws.permute_valid_success(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), TransposeProof.valid_axes(input, a, b, inside)))}

def transpose_observation_mapped(+input: Shape.Shape, +a: U32, +b: U32, +plan: Layout.Plan<input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)>,
  -original: Core.Tensor<input>, -retained: Core.Tensor<input>, -target: Core.Tensor<Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan)>, +index: U32,
  +mapping: Layout.Mapping<input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index>,
  -map_exact: {Done{mapping} == Layout.map_index(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index) :
    Result<&2, &2, Layout.Error, Layout.Mapping<input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index>>},
  -prefix: Transforms.Prefix<input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, original, retained, target,
    Shape.logical_numel(Shape.dimensions(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan)))>,
  +inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan)))) == True{} : Bool},
  +axes_inside: {True{} == TransposeSpec.valid(input, a, b) : Bool}) ->
  Totality.TransposeObservation<input, Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), original, target, a, b, index>:
  Totality.TransposeObservation{
    Equal.trans(Maybe<&2, F32>, Core.observe(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), target, index),
      Transforms.expected_output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, original, index),
      Totality.reference_observe(input, original, Totality.transpose_reference(input, Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), a, b, index)),
      {\{Totality.CopyObservation: equal => equal} :
        Totality.CopyObservation<input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, original, target, index> ->
          {Core.observe(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), target, index) ==
            Transforms.expected_output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, original, index) : Maybe<&2, F32>}}(
              prefix_observation(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, original, retained, target, index, prefix, inside)),
      Equal.trans(Maybe<&2, F32>, Transforms.expected_output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, original, index),
        Core.checked_observe(input, original, Layout.source_index(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index, mapping)),
        Totality.reference_observe(input, original, Totality.transpose_reference(input, Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), a, b, index)),
        Equal.sym(Maybe<&2, F32>, Core.checked_observe(input, original, Layout.source_index(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index, mapping)),
          Transforms.expected_output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, original, index),
          Equal.cong(Result<&2, &2, Layout.Error, Layout.Mapping<input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index>>, Maybe<&2, F32>,
            Transforms.expected_mapped(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index, original),
            Done{mapping}, Layout.map_index(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index), map_exact)),
        Equal.trans(Maybe<&2, F32>, Core.checked_observe(input, original, Layout.source_index(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index, mapping)),
          Core.observe(input, original, Layout.source_index(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index, mapping)),
          Totality.reference_observe(input, original, Totality.transpose_reference(input, Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), a, b, index)),
          {\{ObserveBridge: equal => equal} :
            ObserveBridge<input, original, Layout.source_index(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index, mapping)> ->
              {Core.checked_observe(input, original, Layout.source_index(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index, mapping)) ==
                Core.observe(input, original, Layout.source_index(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index, mapping)) : Maybe<&2, F32>}}(
                  logical_observe_bridge(input, original, Layout.source_index(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index, mapping),
                    Laws.mapped_mathematical_bounded(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index, mapping))),
          Equal.cong(Maybe<&2, U32>, Maybe<&2, F32>, Totality.reference_observe(input, original),
            Some{Layout.source_index(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index, mapping)},
            Totality.transpose_reference(input, Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), a, b, index),
            %TransposeProof.transpose_axes_requested(input, a, b, axes_inside) :
              {Some{Layout.source_index(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index, mapping)} ==
                Layout.reference_source_coordinates(Shape.dimensions(input), Axis.scatter(U32, Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(_),
                  Shape.words(Shape.digits(Shape.extents(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan)), U32.to_nat(index))), 0n)) : Maybe<&2, U32>}
            Laws.mapping_reference(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index, mapping)))))}

def transpose_requested_certificate(+input: Shape.Shape, -source: Core.Tensor<input>, +a: U32, +b: U32,
  +inside: {True{} == TransposeSpec.valid(input, a, b) : Bool},
  certificate: Totality.TypedPermutation<input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), source,
    Transforms.permute_copy(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0))>) -> Totality.RequestedTranspose<input, source, a, b>:
  Totality.TypedPermutation{+plan, retained, target, exact, prefix} = certificate
  Totality.RequestedTranspose{plan, retained, target,
    Equal.trans(Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>,
      (retained, Done{Core.Packed{Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), target}}),
      Transforms.permute_copy(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)), Transforms.transpose_copy(input, source, a, b), exact,
      Equal.sym(Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>, Transforms.transpose_copy(input, source, a, b),
        Transforms.permute_copy(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)), transpose_permutation(input, source, a, b, inside))),
    Equal.trans(List<&2, U32>, Shape.dimensions(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan)),
      Axis.order(U32, Shape.dimensions(input), Shape.naturals(Layout.transpose_axes(Shape.dimensions(input), a, b, 0))),
      Axis.order(U32, Shape.dimensions(input), Shape.naturals(TransposeSpec.requested_axes(Shape.rank_dimensions(Shape.dimensions(input)), U32.to_nat(a), U32.to_nat(b)))),
      Mapping.permutation_dimensions_order(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan),
      Equal.cong(List<&2, U32>, List<&2, U32>, axes => Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)),
        Layout.transpose_axes(Shape.dimensions(input), a, b, 0), TransposeSpec.requested_axes(Shape.rank_dimensions(Shape.dimensions(input)), U32.to_nat(a), U32.to_nat(b)),
        TransposeProof.transpose_axes_requested(input, a, b, inside))),
    prefix, index => mapping =>
      %TransposeProof.transpose_axes_requested(input, a, b, inside) :
          {Some{Layout.source_index(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index, mapping)} ==
            Layout.reference_source_coordinates(Shape.dimensions(input), Axis.scatter(U32, Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(_),
              Shape.words(Shape.digits(Shape.extents(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan)), U32.to_nat(index))), 0n)) : Maybe<&2, U32>}
      Laws.mapping_reference(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index, mapping),
    index => logical =>
      {\{Totality.MappingDone: mapping => map_exact =>
        {\{Totality.TransposeObservation: equal => equal} :
          Totality.TransposeObservation<input, Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), source, target, a, b, index> ->
            {Core.observe(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), target, index) ==
              Totality.reference_observe(input, source, Totality.transpose_reference(input, Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), a, b, index)) : Maybe<&2, F32>}}(
                transpose_observation_mapped(input, a, b, plan, source, retained, target, index, mapping, map_exact, prefix, logical, inside))} :
                  Totality.MappingDone<input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index,
                    Layout.map_index(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index)> ->
                      {Core.observe(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), target, index) ==
                        Totality.reference_observe(input, source, Totality.transpose_reference(input, Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), a, b, index)) : Maybe<&2, F32>}}(
                          mapping_done(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index,
                            logical_index_bounded(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), index, logical)))}

def Laws.transpose_requested_result(input, source, a, b, inside):
  transpose_requested_certificate(input, source, a, b, inside,
    Laws.permute_typed_success(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), TransposeProof.valid_axes(input, a, b, inside)))

def transpose_output_inside(+input: Shape.Shape, +a: U32, +b: U32,
  +plan: Layout.Plan<input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)>,
  +inside: {True{} == TransposeSpec.valid(input, a, b) : Bool}) ->
  {True{} == TransposeSpec.valid(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), a, b) : Bool}:
  %Equal.sym(Nat, Shape.rank_dimensions(Shape.dimensions(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan))),
    Shape.rank_dimensions(Shape.dimensions(input)), Mapping.permutation_output_rank(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan)) :
      {True{} == TransposeSpec.axes_valid(_, U32.to_nat(a), U32.to_nat(b)) : Bool}
  inside

def transpose_output_axes(+input: Shape.Shape, +a: U32, +b: U32,
  +plan: Layout.Plan<input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)>,
  +inside: {True{} == TransposeSpec.valid(input, a, b) : Bool}) ->
  {Layout.transpose_axes(Shape.dimensions(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan)), a, b, 0) ==
    Totality.inverse_axes(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)) : List<&2, U32>}:
  Equal.trans(List<&2, U32>, Layout.transpose_axes(Shape.dimensions(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan)), a, b, 0),
    TransposeSpec.requested_axes(Shape.rank_dimensions(Shape.dimensions(input)), U32.to_nat(a), U32.to_nat(b)),
    Totality.inverse_axes(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)),
    Equal.trans(List<&2, U32>, Layout.transpose_axes(Shape.dimensions(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan)), a, b, 0),
      TransposeSpec.requested_axes(Shape.rank_dimensions(Shape.dimensions(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan))), U32.to_nat(a), U32.to_nat(b)),
      TransposeSpec.requested_axes(Shape.rank_dimensions(Shape.dimensions(input)), U32.to_nat(a), U32.to_nat(b)),
      TransposeProof.transpose_axes_requested(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), a, b, transpose_output_inside(input, a, b, plan, inside)),
      Equal.cong(Nat, List<&2, U32>, rank => TransposeSpec.requested_axes(rank, U32.to_nat(a), U32.to_nat(b)),
        Shape.rank_dimensions(Shape.dimensions(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan))),
        Shape.rank_dimensions(Shape.dimensions(input)), Mapping.permutation_output_rank(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan))),
    Equal.trans(List<&2, U32>, TransposeSpec.requested_axes(Shape.rank_dimensions(Shape.dimensions(input)), U32.to_nat(a), U32.to_nat(b)),
      Layout.transpose_axes(Shape.dimensions(input), a, b, 0), Totality.inverse_axes(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)),
      Equal.sym(List<&2, U32>, Layout.transpose_axes(Shape.dimensions(input), a, b, 0),
        TransposeSpec.requested_axes(Shape.rank_dimensions(Shape.dimensions(input)), U32.to_nat(a), U32.to_nat(b)), TransposeProof.transpose_axes_requested(input, a, b, inside)),
      Equal.sym(List<&2, U32>, Totality.inverse_axes(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)), Layout.transpose_axes(Shape.dimensions(input), a, b, 0),
        Equal.trans(List<&2, U32>, Totality.inverse_axes(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)),
          Shape.words(Shape.naturals(Layout.transpose_axes(Shape.dimensions(input), a, b, 0))), Layout.transpose_axes(Shape.dimensions(input), a, b, 0),
          Equal.cong(List<&2, Nat>, List<&2, U32>, Shape.words,
            Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(Layout.transpose_axes(Shape.dimensions(input), a, b, 0))),
            Shape.naturals(Layout.transpose_axes(Shape.dimensions(input), a, b, 0)), TransposeProof.inverse_axes(input, a, b, inside)),
          words_naturals(Layout.transpose_axes(Shape.dimensions(input), a, b, 0))))))

def transpose_restoration_certificate(+input: Shape.Shape, -original: Core.Tensor<input>, +a: U32, +b: U32,
  +inside: {True{} == TransposeSpec.valid(input, a, b) : Bool},
  certificate: Totality.PermutationRestoration<input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), original>) -> Totality.TransposeRestoration<input, original, a, b>:
  Totality.PermutationRestoration{+first, retained, middle, first_exact, second, middle_retained, restored, second_exact, dimensions, linear} = certificate
  Totality.TransposeRestoration{first, retained, middle,
    Equal.trans(Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>,
      (retained, Done{Core.Packed{Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), middle}}),
      Transforms.permute_copy(input, original, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)), Transforms.transpose_copy(input, original, a, b), first_exact,
      Equal.sym(Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>, Transforms.transpose_copy(input, original, a, b),
        Transforms.permute_copy(input, original, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)), transpose_permutation(input, original, a, b, inside))),
    second, middle_retained, restored,
    Equal.trans(Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>,
      (Core.Packed{Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), middle_retained},
        Done{Core.Packed{Layout.output(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), Totality.inverse_axes(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)), second), restored}}),
      Transforms.permute_dynamic(Core.Packed{Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), middle}, Totality.inverse_axes(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0))),
      Transforms.transpose_dynamic(Core.Packed{Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), middle}, a, b), second_exact,
      Equal.sym(Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>,
        Transforms.transpose_dynamic(Core.Packed{Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), middle}, a, b),
        Transforms.permute_dynamic(Core.Packed{Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), middle}, Totality.inverse_axes(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0))),
        Equal.trans(Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>,
          Transforms.transpose_dynamic(Core.Packed{Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), middle}, a, b),
          Transforms.permute_dynamic(Core.Packed{Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), middle},
            Layout.transpose_axes(Shape.dimensions(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first)), a, b, 0)),
          Transforms.permute_dynamic(Core.Packed{Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), middle}, Totality.inverse_axes(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0))),
          Equal.cong(Core.Tensor<Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first)> & Result<&2, &1, Transforms.Error, Core.Dynamic>,
            Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>, Transforms.dynamic_result(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first)),
            Transforms.transpose_copy(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), middle, a, b),
            Transforms.permute_copy(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), middle,
              Layout.transpose_axes(Shape.dimensions(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first)), a, b, 0)),
            transpose_permutation(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), middle, a, b, transpose_output_inside(input, a, b, first, inside))),
          Equal.cong(List<&2, U32>, Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>,
            axes => Transforms.permute_dynamic(Core.Packed{Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), middle}, axes),
            Layout.transpose_axes(Shape.dimensions(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first)), a, b, 0),
            Totality.inverse_axes(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)), transpose_output_axes(input, a, b, first, inside))))), dimensions, linear}

def Laws.transpose_inverse_restoration(input, source, a, b, inside):
  Totality.TransposeInverse{
    {\{Totality.PermutationInverse: restoration => transpose_restoration_certificate(input, source, a, b, inside, restoration)} :
      Totality.PermutationInverse<input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), source> -> Totality.TransposeRestoration<input, source, a, b>}(
        Laws.permute_inverse_restoration(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), TransposeProof.valid_axes(input, a, b, inside)))}
