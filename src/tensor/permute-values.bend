import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/LAWS.bend as StorageLaws
import ../storage/arith.bend as Arith
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./layout.bend as Layout
import ./axis-spec.bend as Axis
import ./transforms.bend as Transforms
import ./transpose-spec.bend as Transpose
import ./transpose-PROOF.bend as TransposeProof
import ./permutation-mapping.bend as Mapping
import ./word-conversion.bend as Conversion
import ./from-values-contract.bend as FromValues
import ./contracts/replayed.bend as Replayed
import ./contracts/pointwise.bend as Pointwise
import ./contracts/pointwise-proof.bend as PointwiseProof
import ./index-map.bend as IndexMap
import ./observation.bend as Values
import ./transform-observation.bend as Observation
import ./permute-values-records.bend as Records

# The permute copy as an index-map instance: output index i reads the input
# at `Values.permute_source(input, axes, i)`, the row-major index of the
# input coordinate that `axes` sends to output coordinate i.

type Request is Data:
  Request{input: Shape.Shape, axes: List<&2, U32>}

def source_of(request: Request, index: Nat) -> Nat:
  Request{+input, +axes} = request
  U32.to_nat(Maybe.default(&2, U32, Values.permute_source(input, axes, U32.from_nat(index)), 0))

def read_source(+input: Shape.Shape, +axes: List<&2, U32>, +values: List<&2, F32>, +count: Nat) -> Storage.Storage:
  IndexMap.source(~Request, ~source_of, Request{input, axes}, input, 0.0, FromValues.writes(values, 0n), count)

def written(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +values: List<&2, F32>, +count: Nat) -> Storage.Storage:
  IndexMap.target(~Request, ~source_of, count, Request{input, axes}, input, 0.0, FromValues.writes(values, 0n), Layout.output(input, axes, plan))

def Shaped(+fuel: Nat, +input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +cursor: Nat,
  -kept: Storage.Storage, -made: Storage.Storage,
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>> -> Type) -> Type:
  @-source_fits: {Lifecycle.logical_length(kept) == Shape.logical_numel(Shape.dimensions(input)) : Nat} ->
  @-target_fits: {Lifecycle.logical_length(made) == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat} ->
    Q(Transforms.copy_loop(fuel, input, axes, plan, Core.TTensor{kept, source_fits}, Core.TTensor{made, target_fits}, cursor))

def Start(+fuel: Nat, +input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +values: List<&2, F32>, +cursor: Nat,
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>> -> Type) -> Type:
  Shaped(fuel, input, axes, plan, cursor, read_source(input, axes, values, cursor), written(input, axes, plan, values, cursor), Q)

def Final(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +values: List<&2, F32>, +count: Nat,
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>> -> Type) -> Type:
  @-source_fits: {Lifecycle.logical_length(read_source(input, axes, values, count)) == Shape.logical_numel(Shape.dimensions(input)) : Nat} ->
  @-target_fits: {Lifecycle.logical_length(written(input, axes, plan, values, count)) ==
    Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat} ->
    Q((Core.TTensor{read_source(input, axes, values, count), source_fits}, Done{Core.TTensor{written(input, axes, plan, values, count), target_fits}}))

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

def cursor_same(+cursor: Nat, exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat}) ->
  {Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor) == True{} : Bool}:
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(cursor)), cursor, exact) : {Nat.is_eq(_, cursor) == True{} : Bool}
  Arith.eq_refl(cursor)

def read_step(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +values: List<&2, F32>, +rest: Nat, +cursor: Nat,
  +below: {Nat.is_lt(cursor, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool},
  +mapping: Layout.Mapping<input, axes, plan, U32.from_nat(cursor)>,
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>> -> Type,
  next: Start(rest, input, axes, plan, values, 1n+cursor, Q),
  -source_fits: {Lifecycle.logical_length(read_source(input, axes, values, cursor)) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  -target_fits: {Lifecycle.logical_length(written(input, axes, plan, values, cursor)) ==
    Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}) ->
  Q(Transforms.copy_next(input, axes, plan, kept => made => Transforms.copy_loop(rest, input, axes, plan, kept, made, 1n+cursor),
    Transforms.copy_mapped(input, axes, plan, U32.from_nat(cursor), kept => made => (kept, Done{made}),
      Core.TTensor{read_source(input, axes, values, cursor), source_fits}, Core.TTensor{written(input, axes, plan, values, cursor), target_fits},
      Done{mapping}))):
  +index = U32.from_nat(cursor)
  +word = U32.from_nat(source_of(Request{input, axes}, cursor))
  +history = FromValues.writes(values, 0n)
  +seen = List.append(&2, StorageLaws.Op, history, Pointwise.reads(~Request, ~source_of, Request{input, axes}, cursor))
  +cells = IndexMap.writes(~Request, ~source_of, cursor, Request{input, axes}, input, 0.0, history)
  %Equal.sym(U32, Layout.source_index(input, axes, plan, index, mapping), word, mapped_index(input, axes, plan, index, mapping)) :
    Q(Transforms.copy_next(input, axes, plan, kept => made => Transforms.copy_loop(rest, input, axes, plan, kept, made, 1n+cursor),
      Transforms.copy_read(input, axes, plan, index, kept => made => (kept, Done{made}),
        Core.TTensor{written(input, axes, plan, values, cursor), target_fits},
        Core.read_flat(input, Core.TTensor{read_source(input, axes, values, cursor), source_fits}, _))))
  Replayed.read_lift(input, 0.0, seen, source_fits, word, mapped_inside(input, axes, plan, index, mapping),
    pair => Q(Transforms.copy_next(input, axes, plan, kept => made => Transforms.copy_loop(rest, input, axes, plan, kept, made, 1n+cursor),
      Transforms.copy_read(input, axes, plan, index, kept => made => (kept, Done{made}),
        Core.TTensor{written(input, axes, plan, values, cursor), target_fits}, pair))),
    read_fits => Replayed.write_lift(Layout.output(input, axes, plan), 0.0, cells, target_fits, index, Replayed.value(input, 0.0, seen, word),
      PointwiseProof.word_inside(cursor, Layout.output(input, axes, plan), below),
      pair => Q(Transforms.copy_next(input, axes, plan, kept => made => Transforms.copy_loop(rest, input, axes, plan, kept, made, 1n+cursor),
        Transforms.copy_written(input, axes, plan, kept => made => (kept, Done{made}),
          Core.TTensor{Replayed.storage(input, 0.0, Replayed.snoc(seen, StorageLaws.Read{word})), read_fits}, pair))),
      write_fits => IndexMap.shift(~Request, ~source_of, Request{input, axes}, input, 0.0, history, Layout.output(input, axes, plan), cursor,
        kept => made => Shaped(rest, input, axes, plan, 1n+cursor, kept, made, Q), next)(read_fits, write_fits)))

def mapped_lift(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +values: List<&2, F32>, +rest: Nat, +cursor: Nat,
  +below: {Nat.is_lt(cursor, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool},
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>> -> Type,
  next: Start(rest, input, axes, plan, values, 1n+cursor, Q),
  -source_fits: {Lifecycle.logical_length(read_source(input, axes, values, cursor)) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  -target_fits: {Lifecycle.logical_length(written(input, axes, plan, values, cursor)) ==
    Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat},
  result: Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, U32.from_nat(cursor)>>,
  done: {Result.is_done(&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, U32.from_nat(cursor)>, result) == True{} : Bool}) ->
  Q(Transforms.copy_next(input, axes, plan, kept => made => Transforms.copy_loop(rest, input, axes, plan, kept, made, 1n+cursor),
    Transforms.copy_mapped(input, axes, plan, U32.from_nat(cursor), kept => made => (kept, Done{made}),
      Core.TTensor{read_source(input, axes, values, cursor), source_fits}, Core.TTensor{written(input, axes, plan, values, cursor), target_fits},
      result))):
  match result:
    case Fail{error}:
      Empty.absurd(Q(Transforms.copy_next(input, axes, plan, kept => made => Transforms.copy_loop(rest, input, axes, plan, kept, made, 1n+cursor),
        Transforms.copy_mapped(input, axes, plan, U32.from_nat(cursor), kept => made => (kept, Done{made}),
          Core.TTensor{read_source(input, axes, values, cursor), source_fits}, Core.TTensor{written(input, axes, plan, values, cursor), target_fits},
          Fail{error}))), Arith.clash(done))
    case Done{+mapping}:
      read_step(input, axes, plan, values, rest, cursor, below, mapping, Q, next, source_fits, target_fits)

def decided_lift(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +rest: Nat, +cursor: Nat,
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>> -> Type,
  -kept: Core.Tensor<input>, -made: Core.Tensor<Layout.output(input, axes, plan)>,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)>,
  same: {Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor) == True{} : Bool},
  h: Q(Transforms.copy_next(input, axes, plan, kept => made => Transforms.copy_loop(rest, input, axes, plan, kept, made, 1n+cursor),
    Transforms.copy_step(input, axes, plan, kept, made, U32.from_nat(cursor))))) ->
  Q(Transforms.natural_next(input, axes, plan, cursor, kept => made => Transforms.copy_loop(rest, input, axes, plan, kept, made, 1n+cursor),
    Transforms.natural_decided(input, axes, plan, cursor, U32.from_nat(cursor), kept, made, {==}, decision))):
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd(Q(Transforms.natural_next(input, axes, plan, cursor, kept => made => Transforms.copy_loop(rest, input, axes, plan, kept, made, 1n+cursor),
        Transforms.natural_decided(input, axes, plan, cursor, U32.from_nat(cursor), kept, made, {==}, Shape.BFalse{equal}))),
        Arith.clash(Equal.trans(Bool, False{}, Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor), True{}, equal, same)))
    case Shape.BTrue{equal}:
      h

def step_lift(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +values: List<&2, F32>, +rest: Nat, +cursor: Nat,
  +below: {Nat.is_lt(cursor, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool},
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>> -> Type,
  next: Start(rest, input, axes, plan, values, 1n+cursor, Q)) -> Start(1n+rest, input, axes, plan, values, cursor, Q):
  source_fits => target_fits =>
    decided_lift(input, axes, plan, rest, cursor, Q,
      Core.TTensor{read_source(input, axes, values, cursor), source_fits}, Core.TTensor{written(input, axes, plan, values, cursor), target_fits},
      Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)),
      cursor_same(cursor, PointwiseProof.word_exact(cursor, Layout.output(input, axes, plan), below)),
      mapped_lift(input, axes, plan, values, rest, cursor, below, Q, next, source_fits, target_fits,
        Layout.map_index(input, axes, plan, U32.from_nat(cursor)),
        mapping_done(input, axes, plan, U32.from_nat(cursor), PointwiseProof.word_inside(cursor, Layout.output(input, axes, plan), below))))

def final_at(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +values: List<&2, F32>, -end: Nat, -count: Nat,
  same: {end == count : Nat},
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>> -> Type,
  h: Final(input, axes, plan, values, end, Q)) -> Final(input, axes, plan, values, count, Q):
  %same : Final(input, axes, plan, values, _, Q)
  h

def loop_lift(+fuel: Nat, +input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +values: List<&2, F32>, +cursor: Nat,
  +steps: {Nat.add(cursor, fuel) == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat},
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>> -> Type,
  h: Final(input, axes, plan, values, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Q)) ->
  Start(fuel, input, axes, plan, values, cursor, Q):
  match fuel:
    case 0n:
      +total = Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))
      final_at(input, axes, plan, values, total, cursor, Equal.sym(Nat, cursor, total, PointwiseProof.steps_done(cursor, total, steps)), Q, h)
    case 1n+rest:
      +total = Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))
      step_lift(input, axes, plan, values, rest, cursor, PointwiseProof.cursor_below(cursor, rest, total, steps), Q,
        loop_lift(rest, input, axes, plan, values, 1n+cursor, PointwiseProof.steps_next(cursor, rest, total, steps), Q, h))

def start_cast(+fuel: Nat, +input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +values: List<&2, F32>,
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>> -> Type,
  start: Start(fuel, input, axes, plan, values, 0n, Q)) ->
  Shaped(fuel, input, axes, plan, 0n, Replayed.storage(input, 0.0, FromValues.writes(values, 0n)),
    Replayed.storage(Layout.output(input, axes, plan), 0.0, Nil{}), Q):
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, ops => Replayed.storage(input, 0.0, ops),
    List.append(&2, StorageLaws.Op, FromValues.writes(values, 0n), Nil{}), FromValues.writes(values, 0n),
    PointwiseProof.append_nil(FromValues.writes(values, 0n))) :
    Shaped(fuel, input, axes, plan, 0n, _, Replayed.storage(Layout.output(input, axes, plan), 0.0, Nil{}), Q)
  start

def run_lift(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +values: List<&2, F32>,
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  h: Final(input, axes, plan, values, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))),
    pair => Q(Transforms.copy_pack(input, axes, plan, pair))),
  -source_fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Q(Transforms.copy_allocated(input, axes, plan, Core.TTensor{Replayed.storage(input, 0.0, FromValues.writes(values, 0n)), source_fits},
    Core.new(Layout.output(input, axes, plan), 0.0))):
  Replayed.new_lift(Layout.output(input, axes, plan), 0.0,
    result => Q(Transforms.copy_allocated(input, axes, plan,
      Core.TTensor{Replayed.storage(input, 0.0, FromValues.writes(values, 0n)), source_fits}, result)),
    target_fits => start_cast(Shape.elements(Layout.output(input, axes, plan)), input, axes, plan, values,
      pair => Q(Transforms.copy_pack(input, axes, plan, pair)),
      loop_lift(Shape.elements(Layout.output(input, axes, plan)), input, axes, plan, values, 0n,
        Shape.elements_exact(Layout.output(input, axes, plan)), pair => Q(Transforms.copy_pack(input, axes, plan, pair)), h))(source_fits, target_fits))

def planned_lift(+input: Shape.Shape, +axes: List<&2, U32>, +values: List<&2, F32>,
  result: Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>,
  done: {Result.is_done(&2, &2, Layout.Error, Layout.Plan<input, axes>, result) == True{} : Bool},
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  h: @+plan: Layout.Plan<input, axes> -> Final(input, axes, plan, values, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))),
    pair => Q(Transforms.copy_pack(input, axes, plan, pair))),
  -source_fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Q(Transforms.copy_planned(input, axes, Core.TTensor{Replayed.storage(input, 0.0, FromValues.writes(values, 0n)), source_fits}, result)):
  match result:
    case Fail{error}:
      Empty.absurd(Q(Transforms.copy_planned(input, axes, Core.TTensor{Replayed.storage(input, 0.0, FromValues.writes(values, 0n)), source_fits},
        Fail{error})), Arith.clash(done))
    case Done{+plan}:
      run_lift(input, axes, plan, values, Q, h(plan), source_fits)

def permute_lift(+input: Shape.Shape, +values: List<&2, F32>, +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool},
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  h: @+plan: Layout.Plan<input, axes> -> Final(input, axes, plan, values, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))),
    pair => Q(Transforms.copy_pack(input, axes, plan, pair))),
  -source_fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Q(Transforms.permute_copy(input, Core.TTensor{Replayed.storage(input, 0.0, FromValues.writes(values, 0n)), source_fits}, axes)):
  planned_lift(input, axes, values, Layout.permutation(input, axes),
    Mapping.plan_ranked_success(input, axes,
      Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), valid),
    Q, h, source_fits)

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

def output_inside(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool}:
  %Equal.sym(Nat, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Shape.logical_numel(Shape.dimensions(input)),
    plan_numel(input, axes, plan)) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
  inside

def found_value(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +values: List<&2, F32>,
  +counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  +index: U32, +mapping: Layout.Mapping<input, axes, plan, index>) ->
  {Some{Replayed.value(input, 0.0, FromValues.writes(values, 0n), U32.from_nat(source_of(Request{input, axes}, U32.to_nat(index))))} ==
    Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>}:
  +found = Layout.source_index(input, axes, plan, index, mapping)
  +history = FromValues.writes(values, 0n)
  %Equal.sym(U32, U32.from_nat(U32.to_nat(index)), index, Conversion.word_natural_exact(index)) :
    {Some{Replayed.value(input, 0.0, history, U32.from_nat(U32.to_nat(Maybe.default(&2, U32, Values.permute_source(input, axes, _), 0))))} ==
      Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>}
  %mapped_index(input, axes, plan, index, mapping) :
    {Some{Replayed.value(input, 0.0, history, _)} == Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>}
  %reference(input, axes, plan, index, mapping) :
    {Some{Replayed.value(input, 0.0, history, found)} == Values.listed(values, _) : Maybe<&2, F32>}
  IndexMap.stored_at(input, values, counted, found, mapped_bound(input, axes, plan, index, mapping))

def found_mapped(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +values: List<&2, F32>,
  +counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(input)) : Nat}, +index: U32,
  result: Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>,
  done: {Result.is_done(&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>, result) == True{} : Bool}) ->
  {Some{Replayed.value(input, 0.0, FromValues.writes(values, 0n), U32.from_nat(source_of(Request{input, axes}, U32.to_nat(index))))} ==
    Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>}:
  match result:
    case Fail{error}:
      Empty.absurd({Some{Replayed.value(input, 0.0, FromValues.writes(values, 0n), U32.from_nat(source_of(Request{input, axes}, U32.to_nat(index))))} ==
        Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>}, Arith.clash(done))
    case Done{+mapping}:
      found_value(input, axes, plan, values, counted, index, mapping)

def Permuted(+input: Shape.Shape, +values: List<&2, F32>, +axes: List<&2, U32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type) -> Type:
  Q(Transforms.permute_copy(input, Core.TTensor{Replayed.storage(input, 0.0, FromValues.writes(values, 0n)), fits}, axes))

def permuted_code(+input: Shape.Shape, +values: List<&2, F32>, +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool},
  -fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Permuted(input, values, axes, fits, pair => {Some{Observation.failure_code_pair(input, pair)} == Some{0} : Maybe<&2, U32>}):
  permute_lift(input, values, axes, valid, pair => {Some{Observation.failure_code_pair(input, pair)} == Some{0} : Maybe<&2, U32>},
    plan => source_fits => target_fits => {==}, fits)

def permuted_dimensions(+input: Shape.Shape, +values: List<&2, F32>, +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool},
  -fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Permuted(input, values, axes, fits, pair => {Observation.applied_dimensions_pair(input, pair) ==
    Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))} : Maybe<&2, List<&2, U32>>}):
  permute_lift(input, values, axes, valid, pair => {Observation.applied_dimensions_pair(input, pair) ==
      Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))} : Maybe<&2, List<&2, U32>>},
    plan => source_fits => target_fits =>
      Equal.cong(List<&2, U32>, Maybe<&2, List<&2, U32>>, dimensions => Some{dimensions},
        Shape.dimensions(Layout.output(input, axes, plan)), Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)),
        Mapping.permutation_dimensions_order(input, axes, plan)), fits)

def permuted_length(+input: Shape.Shape, +values: List<&2, F32>, +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool},
  -fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Permuted(input, values, axes, fits, pair => {Observation.applied_length_pair(input, pair) ==
    Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>}):
  permute_lift(input, values, axes, valid, pair => {Observation.applied_length_pair(input, pair) ==
      Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>},
    plan => source_fits => target_fits =>
      +output = Layout.output(input, axes, plan)
      +total = Shape.logical_numel(Shape.dimensions(output))
      -made = written(input, axes, plan, values, total)
      Equal.cong(Nat, Maybe<&2, Nat>, number => Some{number}, Lifecycle.logical_length(made), Shape.logical_numel(Shape.dimensions(input)),
        Equal.trans(Nat, Lifecycle.logical_length(made), total, Shape.logical_numel(Shape.dimensions(input)),
          Replayed.fits(output, 0.0, IndexMap.writes(~Request, ~source_of, total, Request{input, axes}, input, 0.0, FromValues.writes(values, 0n))),
          plan_numel(input, axes, plan))), fits)

def permuted_count(+input: Shape.Shape, +values: List<&2, F32>, +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool},
  -fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Permuted(input, values, axes, fits, pair => {Observation.applied_count_pair(input, pair) == Some{Shape.count(input)} : Maybe<&2, U32>}):
  permute_lift(input, values, axes, valid, pair => {Observation.applied_count_pair(input, pair) == Some{Shape.count(input)} : Maybe<&2, U32>},
    plan => source_fits => target_fits =>
      Equal.cong(Nat, Maybe<&2, U32>, number => Some{U32.from_nat(number)}, Shape.elements(Layout.output(input, axes, plan)), Shape.elements(input),
        plan_elements(input, axes, plan)), fits)

def permuted_value(+input: Shape.Shape, +values: List<&2, F32>,
  +counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool}, +index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool},
  -fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Permuted(input, values, axes, fits, pair => {Observation.applied_value_pair(input, index, pair) ==
    Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>}):
  permute_lift(input, values, axes, valid, pair => {Observation.applied_value_pair(input, index, pair) ==
      Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>},
    plan => source_fits => target_fits =>
      +output = Layout.output(input, axes, plan)
      +total = Shape.logical_numel(Shape.dimensions(output))
      +history = FromValues.writes(values, 0n)
      +below = output_inside(input, axes, plan, index, inside)
      Equal.trans(Maybe<&2, F32>,
        Core.dynamic_read_value(Core.read_dynamic(Core.Packed{output, Core.TTensor{written(input, axes, plan, values, total), target_fits}}, index)),
        Some{Replayed.value(input, 0.0, history, U32.from_nat(source_of(Request{input, axes}, U32.to_nat(index))))},
        Values.listed(values, Values.permute_source(input, axes, index)),
        IndexMap.copied_point(~Request, ~source_of, Request{input, axes}, input, 0.0, history, output, target_fits, index, below),
        found_mapped(input, axes, plan, values, counted, index, Layout.map_index(input, axes, plan, index),
          mapping_done(input, axes, plan, index, below))), fits)

def permuted_kept(+input: Shape.Shape, +values: List<&2, F32>,
  +counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool}, +other: U32,
  +inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool},
  -fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Permuted(input, values, axes, fits, pair => {Observation.kept_value_pair(input, other, pair) ==
    List.get(&2, F32, values, U32.to_nat(other)) : Maybe<&2, F32>}):
  permute_lift(input, values, axes, valid, pair => {Observation.kept_value_pair(input, other, pair) ==
      List.get(&2, F32, values, U32.to_nat(other)) : Maybe<&2, F32>},
    plan => source_fits => target_fits =>
      +total = Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))
      +history = FromValues.writes(values, 0n)
      Equal.trans(Maybe<&2, F32>,
        Core.read_value(input, Core.read_flat(input, Core.TTensor{read_source(input, axes, values, total), source_fits}, other)),
        Some{Replayed.value(input, 0.0, history, other)}, List.get(&2, F32, values, U32.to_nat(other)),
        IndexMap.kept(~Request, ~source_of, Request{input, axes}, total, input, 0.0, history, source_fits, other, inside),
        IndexMap.stored_at(input, values, counted, other, inside)), fits)

def permute_contract(+input: Shape.Shape, +values: List<&2, F32>,
  counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  +axes: List<&2, U32>, valid: {True{} == Layout.bijective(input, axes) : Bool}) -> Records.PermuteContract<input, values, axes>:
  +counted = counted
  +valid = valid
  Records.PermuteContract{
    FromValues.from_values_lift(input, values, counted,
      result => {Observation.failure_code(input, Observation.Permute{axes}, result) == Some{0} : Maybe<&2, U32>},
      fits => permuted_code(input, values, axes, valid, fits)),
    FromValues.from_values_lift(input, values, counted,
      result => {Observation.applied_dimensions(input, Observation.Permute{axes}, result) ==
        Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))} : Maybe<&2, List<&2, U32>>},
      fits => permuted_dimensions(input, values, axes, valid, fits)),
    FromValues.from_values_lift(input, values, counted,
      result => {Observation.applied_length(input, Observation.Permute{axes}, result) ==
        Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>},
      fits => permuted_length(input, values, axes, valid, fits)),
    FromValues.from_values_lift(input, values, counted,
      result => {Observation.applied_count(input, Observation.Permute{axes}, result) == Some{Shape.count(input)} : Maybe<&2, U32>},
      fits => permuted_count(input, values, axes, valid, fits)),
    index => inside =>
      +inside = inside
      FromValues.from_values_lift(input, values, counted,
        result => {Observation.applied_value(input, Observation.Permute{axes}, index, result) ==
          Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>},
        fits => permuted_value(input, values, counted, axes, valid, index, inside, fits)),
    other => inside =>
      +inside = inside
      FromValues.from_values_lift(input, values, counted,
        result => {Observation.kept_value(input, Observation.Permute{axes}, other, result) ==
          List.get(&2, F32, values, U32.to_nat(other)) : Maybe<&2, F32>},
        fits => permuted_kept(input, values, counted, axes, valid, other, inside, fits))}

def requested(+input: Shape.Shape, +a: U32, +b: U32) -> List<&2, U32>:
  Transpose.requested_axes(Shape.rank_dimensions(Shape.dimensions(input)), U32.to_nat(a), U32.to_nat(b))

def transpose_route(+input: Shape.Shape, +a: U32, +b: U32, source: Core.Tensor<input>,
  decision: Shape.BoolCase<Transpose.valid(input, a, b)>, +inside: {True{} == Transpose.valid(input, a, b) : Bool}) ->
  {Transforms.permute_copy(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)) == Transforms.transpose_decided(input, source, a, b, decision) :
    Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({Transforms.permute_copy(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)) ==
        (source, Fail{Transforms.LayoutError{Layout.AxisOutOfBounds{}}}) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>},
        Arith.clash(Equal.trans(Bool, False{}, Transpose.valid(input, a, b), True{}, equal, Equal.sym(Bool, True{}, Transpose.valid(input, a, b), inside))))
    case Shape.BTrue{equal}:
      {==}

def requested_valid(+input: Shape.Shape, +a: U32, +b: U32, +inside: {True{} == Transpose.valid(input, a, b) : Bool}) ->
  {True{} == Layout.bijective(input, requested(input, a, b)) : Bool}:
  %TransposeProof.transpose_axes_requested(input, a, b, inside) : {True{} == Layout.bijective(input, _) : Bool}
  TransposeProof.valid_axes(input, a, b, inside)

def transpose_lift(+input: Shape.Shape, +values: List<&2, F32>, +a: U32, +b: U32, +inside: {True{} == Transpose.valid(input, a, b) : Bool},
  -fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  h: Permuted(input, values, requested(input, a, b), fits, Q)) ->
  Q(Transforms.transpose_copy(input, Core.TTensor{Replayed.storage(input, 0.0, FromValues.writes(values, 0n)), fits}, a, b)):
  %transpose_route(input, a, b, Core.TTensor{Replayed.storage(input, 0.0, FromValues.writes(values, 0n)), fits},
    Shape.bool_case(Transpose.valid(input, a, b)), inside) : Q(_)
  %Equal.sym(List<&2, U32>, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), requested(input, a, b),
    TransposeProof.transpose_axes_requested(input, a, b, inside)) :
    Q(Transforms.permute_copy(input, Core.TTensor{Replayed.storage(input, 0.0, FromValues.writes(values, 0n)), fits}, _))
  h

def transpose_contract(+input: Shape.Shape, +values: List<&2, F32>,
  counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  +a: U32, +b: U32, inside: {True{} == Transpose.valid(input, a, b) : Bool}) -> Records.TransposeContract<input, values, a, b>:
  +counted = counted
  +inside = inside
  +axes = requested(input, a, b)
  +valid = requested_valid(input, a, b, inside)
  Records.TransposeContract{
    FromValues.from_values_lift(input, values, counted,
      result => {Observation.failure_code(input, Observation.Transpose{a, b}, result) == Some{0} : Maybe<&2, U32>},
      fits => transpose_lift(input, values, a, b, inside, fits, pair => {Some{Observation.failure_code_pair(input, pair)} == Some{0} : Maybe<&2, U32>},
        permuted_code(input, values, axes, valid, fits))),
    FromValues.from_values_lift(input, values, counted,
      result => {Observation.applied_dimensions(input, Observation.Transpose{a, b}, result) ==
        Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))} : Maybe<&2, List<&2, U32>>},
      fits => transpose_lift(input, values, a, b, inside, fits, pair => {Observation.applied_dimensions_pair(input, pair) ==
          Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))} : Maybe<&2, List<&2, U32>>},
        permuted_dimensions(input, values, axes, valid, fits))),
    FromValues.from_values_lift(input, values, counted,
      result => {Observation.applied_length(input, Observation.Transpose{a, b}, result) ==
        Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>},
      fits => transpose_lift(input, values, a, b, inside, fits, pair => {Observation.applied_length_pair(input, pair) ==
          Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>},
        permuted_length(input, values, axes, valid, fits))),
    FromValues.from_values_lift(input, values, counted,
      result => {Observation.applied_count(input, Observation.Transpose{a, b}, result) == Some{Shape.count(input)} : Maybe<&2, U32>},
      fits => transpose_lift(input, values, a, b, inside, fits, pair => {Observation.applied_count_pair(input, pair) == Some{Shape.count(input)} : Maybe<&2, U32>},
        permuted_count(input, values, axes, valid, fits))),
    index => below =>
      +below = below
      FromValues.from_values_lift(input, values, counted,
        result => {Observation.applied_value(input, Observation.Transpose{a, b}, index, result) ==
          Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>},
        fits => transpose_lift(input, values, a, b, inside, fits, pair => {Observation.applied_value_pair(input, index, pair) ==
            Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>},
          permuted_value(input, values, counted, axes, valid, index, below, fits))),
    other => below =>
      +below = below
      FromValues.from_values_lift(input, values, counted,
        result => {Observation.kept_value(input, Observation.Transpose{a, b}, other, result) ==
          List.get(&2, F32, values, U32.to_nat(other)) : Maybe<&2, F32>},
        fits => transpose_lift(input, values, a, b, inside, fits, pair => {Observation.kept_value_pair(input, other, pair) ==
            List.get(&2, F32, values, U32.to_nat(other)) : Maybe<&2, F32>},
          permuted_kept(input, values, counted, axes, valid, other, below, fits)))}
