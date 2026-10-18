import Base

import ../shape/core.bend as Shape

import ../shape/LAWS.bend as ShapeLaws

import ../shape/plan_proofs.bend as PlanProof

import ../shape/nat_proofs.bend as Nats

import ../storage/arith.bend as Arith

import ../storage/main.bend as Storage

import ../tensor/core.bend as Core

import ../tensor/layout.bend as Layout

import ../tensor/lifecycle.bend as Lifecycle

import ../tensor/elementwise.bend as Elementwise

import ../tensor/elementwise_totality.bend as Totality

import ../tensor/elementwise_PROOF.bend as ElementwiseProof

import ../tensor/core_PROOF.bend as CoreProof

import ../tensor/transform_PROOF.bend as TransformProof

import ../tensor/transform_totality.bend as TransformTotality

import ../tensor/projection_LAWS.bend as Projection

import ../tensor/projection_PROOF.bend as ProjectionProof

import ../tensor/allocation_spec.bend as Allocation

import ../tensor/allocation_PROOF.bend as AllocationProof

import ../tensor/construction_contract_PROOF.bend as ConstructionProof

import ../tensor/word_conversion.bend as Conversion

import ../tensor/observation.bend as Observation

import ./reduce.bend as Reduce

import ./reduce_LAWS.bend as Laws

import ../storage/model.bend as Model

import ../storage/refinement.bend as Refinement

import ../storage/route.bend as Route

import ../tensor/matmul.bend as Matrix

import ../tensor/matrix_index_proof.bend as MatrixIndex

import ../tensor/elementwise_parallel_proof.bend as Parallel

import ./reduce_parallel_proof.bend as ReduceParallel

import ../tensor/matmul_PROOF.bend as MatmulProof

type Holds<-P: Type> is Data:
  Holds{-proof: P}

def inspection_value(value: Totality.Inspection) -> Maybe<&2, F32>:
  match value:
    case Totality.Dims{dimensions}:
      None{}
    case Totality.Value{value}:
      value

def same_checked(-original: Core.Dynamic, -retained: Core.Dynamic, -same: Totality.SameSource(original, retained), +index: U32) ->
  Holds<{Core.checked_observe_dynamic(retained, index) == Core.checked_observe_dynamic(original, index) : Maybe<&2, F32>}>:
  Holds{Equal.cong(Totality.Inspection, Maybe<&2, F32>, inspection_value,
    Totality.inspect(Totality.Checked{index}, retained), Totality.inspect(Totality.Checked{index}, original), same(Totality.Checked{index}))}

def same_dimensions(-original: Core.Dynamic, -retained: Core.Dynamic, -same: Totality.SameSource(original, retained)) ->
  Holds<{Totality.dimensions(retained) == Totality.dimensions(original) : List<&2, U32>}>:
  Holds{Equal.cong(Totality.Inspection, List<&2, U32>, ElementwiseProof.inspection_dimensions,
    Totality.inspect(Totality.Dimensions{}, retained), Totality.inspect(Totality.Dimensions{}, original), same(Totality.Dimensions{}))}

def same_trans(-a: Core.Dynamic, -b: Core.Dynamic, -c: Core.Dynamic,
  -ab: Totality.SameSource(a, b), -bc: Totality.SameSource(b, c)) -> Holds<Totality.SameSource(a, c)>:
  Holds{property => Equal.trans(Totality.Inspection, Totality.inspect(property, c), Totality.inspect(property, b), Totality.inspect(property, a),
    bc(property), ab(property))}

def Inv(+plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target: Shape.Shape, +cursor: Nat, output: Core.Tensor<target>) -> Type:
  @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool} ->
    {Core.checked_observe(target, output, index) == Laws.fiber(plan, values, index, cursor) : Maybe<&2, F32>}

def update_written(-shape: Shape.Shape, -original: Core.Tensor<shape>, -output: Core.Tensor<shape>, -index: U32, -value: F32, -unit: Unit,
  meaning: Core.CheckedUpdateMeaning(shape, original, index, value, (output, Done{unit}))) ->
  {Core.checked_observe(shape, output, index) == Some{value} : Maybe<&2, F32>}:
  (checked, rest) = meaning
  (observed, frame) = rest
  observed

def update_frame(-shape: Shape.Shape, -original: Core.Tensor<shape>, -output: Core.Tensor<shape>, -index: U32, -value: F32, -unit: Unit,
  meaning: Core.CheckedUpdateMeaning(shape, original, index, value, (output, Done{unit}))) ->
  @+other: U32 -> @checked: Core.ObservationChecked(Core.Packed{shape, original}, other) ->
    @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
      Core.ObservationChecked(Core.Packed{shape, output}, other) &
        {Core.checked_observe(shape, output, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}:
  (checked, rest) = meaning
  (observed, frame) = rest
  frame

def frame_equal(-shape: Shape.Shape, -output: Core.Tensor<shape>, -original: Core.Tensor<shape>, -other: U32,
  pair: Core.ObservationChecked(Core.Packed{shape, output}, other) &
    {Core.checked_observe(shape, output, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}) ->
  {Core.checked_observe(shape, output, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}:
  (checked, equal) = pair
  equal

def listed_kept(-original: Core.Dynamic, -retained: Core.Dynamic, +values: List<&2, F32>,
  -same: Totality.SameSource(original, retained), -listed: Laws.Listed(original, values)) -> Holds<Laws.Listed(retained, values)>:
  Holds{source => Equal.trans(Maybe<&2, F32>, Core.checked_observe_dynamic(retained, source), Core.checked_observe_dynamic(original, source),
    Laws.word_at(values, U32.to_nat(source)),
    {\{Holds: proof => proof} : Holds<{Core.checked_observe_dynamic(retained, source) == Core.checked_observe_dynamic(original, source) : Maybe<&2, F32>}> ->
      {Core.checked_observe_dynamic(retained, source) == Core.checked_observe_dynamic(original, source) : Maybe<&2, F32>}}(
      same_checked(original, retained, same, source)),
    listed(source))}

def zero_inv(+plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target: Shape.Shape, output: Core.Tensor<target>,
  -same: {Done{output} == Core.new(target, 0.0) : Result<&2, &1, Core.Error, Core.Tensor<target>>}) ->
  Holds<Inv(plan, values, target, 0n, output)>:
  Holds{index => inside => Equal.trans(Maybe<&2, F32>, Core.checked_observe(target, output, index),
    Core.read_value(target, Core.read_flat(target, output, index)), Some{0.0},
    Equal.sym(Maybe<&2, F32>, Core.read_value(target, Core.read_flat(target, output, index)), Core.checked_observe(target, output, index),
      {\{Core.SourceRead: meaning => meaning} : Core.SourceRead<target, output, index> ->
        {Core.read_value(target, Core.read_flat(target, output, index)) == Core.checked_observe(target, output, index) : Maybe<&2, F32>}}(
        Core.source_read_model(target, output, index))),
    %Equal.sym(Result<&2, &1, Core.Error, Core.Tensor<target>>, Done{output}, Core.new(target, 0.0), same) :
      {Observation.point(target, index, _) == Some{0.0} : Maybe<&2, F32>}
    ConstructionProof.point_value(target, 0.0, index, inside))}

def mapping_values(-input: Core.Dynamic, -target: Shape.Shape, -plan: Shape.BroadcastPlan, -output: Core.Dynamic,
  model: {Totality.dimensions(output) == Shape.dimensions(target) : List<&2, U32>} &
    (@+values: List<&2, F32> -> @-listed: Laws.Listed(input, values) ->
      @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool} ->
        {Core.checked_observe_dynamic(output, index) == Laws.planned_fiber(values, index, Shape.logical_numel(Totality.dimensions(input)), Done{plan}) : Maybe<&2, F32>})) ->
  @+values: List<&2, F32> -> @-listed: Laws.Listed(input, values) ->
    @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool} ->
      {Core.checked_observe_dynamic(output, index) == Laws.planned_fiber(values, index, Shape.logical_numel(Totality.dimensions(input)), Done{plan}) : Maybe<&2, F32>}:
  (dimensions, mapped) = model
  mapped

def output_count(+shape: Shape.Shape, +plan: Shape.BroadcastPlan,
  -equal: {True{} == Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape)) : Bool}) ->
  Holds<{Shape.elements(Shape.output_shape(plan)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}>:
  Holds{Equal.trans(Nat, Shape.elements(Shape.output_shape(plan)), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))),
    Shape.logical_numel(Shape.dimensions(shape)), Shape.elements_exact(Shape.output_shape(plan)),
    Equal.cong(List<&2, U32>, Nat, Shape.logical_numel, Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape),
      TransformProof.words_equal(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape), equal)))}

def rank_of(+shape: Shape.Shape) -> Shape.LE(Shape.rank_dimensions(Shape.dimensions(shape)), 4n):
  match shape:
    case Shape.SShape{dims, size, exact, rank_bound, capacity_bound}:
      rank_bound

def capacity_of(+shape: Shape.Shape) -> Shape.LE(Shape.logical_numel(Shape.dimensions(shape)), 1048576n):
  match shape:
    case Shape.SShape{dims, size, exact, rank_bound, capacity_bound}:
      capacity_bound

# The parallel reduction keeps the contribution: it hands back the kept
# half of the contribution's clone, which observes like the contribution.

def indexed_retains(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape, +plan: Shape.BroadcastPlan, output: Core.Tensor<target>,
  +depth: Nat, +length: Nat, -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(target)) : Nat},
  map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>) ->
  Laws.Retained<Core.Packed{shape, tensor}, Reduce.parallel_indexed(Core.Packed{shape, tensor}, target, plan, output, depth, length, valid, exact, map)>:
  match map:
    case Shape.Unindexed{zero}:
      Laws.Retained{property => {==}}
    case Shape.Indexed{rows, same}:
      MatmulProof.take_lift(shape, tensor,
        pair => Laws.Retained<Core.Packed{shape, tensor}, Reduce.parallel_rows(plan, target, depth, length, valid, exact, pair, rows)>,
        kept => copy => taken => Laws.Retained{{\{Parallel.Taken: depth => balanced => reads => retained => retained} :
          Parallel.Taken<Core.Packed{shape, tensor}, (Core.Packed{shape, kept}, copy)> -> Totality.SameSource(Core.Packed{shape, tensor}, Core.Packed{shape, kept})}(taken)})

def parallel_retains(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  result: Result<&2, &1, Core.Error, Core.Tensor<target>>) ->
  Laws.Retained<Core.Packed{shape, tensor}, Reduce.parallel_allocated(Core.Packed{shape, tensor}, target, plan, result)>:
  match result:
    case Fail{error}:
      Laws.Retained{property => {==}}
    case Done{Core.TTensor{storage, exact}}:
      match storage:
        case Storage.Empty{}:
          Laws.Retained{property => {==}}
        case Storage.Owned{+depth, +length, array, balanced, valid}:
          indexed_retains(shape, tensor, target, plan, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, depth, length,
            valid, exact, Shape.index_map(plan))

def fitted_retains(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape, +plan: Shape.BroadcastPlan, same: Bool) ->
  Laws.Retained<Core.Packed{shape, tensor}, Reduce.fitted(Core.Packed{shape, tensor}, target, plan, same)>:
  match same:
    case False{}:
      Laws.Retained{property => {==}}
    case True{}:
      parallel_retains(shape, tensor, target, plan, Core.new(target, 0.0))

def planned_retains(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) ->
  Laws.Retained<Core.Packed{shape, tensor}, Reduce.planned(shape, tensor, target, result)>:
  match result:
    case Fail{error}:
      Laws.Retained{property => {==}}
    case Done{+plan}:
      fitted_retains(shape, tensor, target, plan,
        Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape)))

def Laws.reduce_retains_input(input, target):
  Core.Packed{+shape, tensor} = input
  planned_retains(shape, tensor, target, Shape.broadcast_checked(target, shape))

# The parallel reduction's mapping: every target cell of the parallel build
# holds the masked fiber sum over any list that lists the contribution.

def parallel_build(+plan: Shape.BroadcastPlan, +depth: Nat, +length: Nat, +rows: List<&2, Shape.Row>, array: Array<F32>) -> Array<F32>:
  Reduce.build(depth, Reduce.splits(plan, depth, length), rows, Shape.extents(Shape.output_shape(plan)), length, 0n, array)

def rows_cell(+shape: Shape.Shape, -tensor: Core.Tensor<shape>, +target: Shape.Shape, +plan: Shape.BroadcastPlan, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool}, -exact: {length == Shape.logical_numel(Shape.dimensions(target)) : Nat},
  +rows: List<&2, Shape.Row>,
  +rows_exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>},
  -array: Array<F32>, +d: Nat, -balanced: Model.Balanced(F32, d, array), -reads: Parallel.SourceReads(Core.Packed{shape, tensor}, d, array),
  +left: {Shape.dimensions(Shape.left_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  +right: {Shape.dimensions(Shape.right_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>},
  +count: {Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  +values: List<&2, F32>, -listed: Laws.Listed(Core.Packed{shape, tensor}, values), +index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}) ->
  Holds<{Lifecycle.observe_allowed(depth, length, parallel_build(plan, depth, length, rows, array), index, Nat.is_lt(U32.to_nat(index), length)) ==
    Laws.fiber(plan, values, index, Shape.logical_numel(Shape.dimensions(shape))) : Maybe<&2, F32>}>:
  +t = U32.to_nat(index)
  +extents = Shape.extents(Shape.output_shape(plan))
  +splits = Reduce.splits(plan, depth, length)
  -built = parallel_build(plan, depth, length, rows, array)
  -inside_length : {Nat.is_lt(t, length) == True{} : Bool} =
    %Equal.sym(Nat, length, Shape.logical_numel(Shape.dimensions(target)), exact) : {Nat.is_lt(t, _) == True{} : Bool}
    inside
  +path = Refinement.route(depth, index)
  Holds{{\{ReduceParallel.CellAt: word => observed => words => seen => total =>
    Equal.trans(Maybe<&2, F32>, Lifecycle.observe_allowed(depth, length, built, index, Nat.is_lt(t, length)),
      Model.observe(F32, depth, built, path), Laws.fiber(plan, values, index, Shape.logical_numel(Shape.dimensions(shape))),
      Parallel.observe_allowed_equal(depth, length, built, index, Nat.is_lt(t, length), inside_length,
        Equal.sym(Bool, True{}, Storage.index_matches(depth, length, index), Route.index_complete(depth, length, index, valid, inside_length))),
      %Equal.sym(Maybe<&2, F32>, Model.observe(F32, depth, built, path), Some{word}, observed) :
        {_ == Laws.fiber(plan, values, index, Shape.logical_numel(Shape.dimensions(shape))) : Maybe<&2, F32>}
      %Equal.sym(F32, word, ReduceParallel.fsum(words, 0.0), total) :
        {Some{_} == Laws.fiber(plan, values, index, Shape.logical_numel(Shape.dimensions(shape))) : Maybe<&2, F32>}
      %count : {Some{ReduceParallel.fsum(words, 0.0)} == Laws.fiber(plan, values, index, _) : Maybe<&2, F32>}
      {\{Parallel.Fact: p => p} :
        Parallel.Fact<{Some{ReduceParallel.fsum(words, 0.0)} == Laws.fiber(plan, values, index, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) : Maybe<&2, F32>}> ->
        {Some{ReduceParallel.fsum(words, 0.0)} == Laws.fiber(plan, values, index, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) : Maybe<&2, F32>}}(
        ReduceParallel.cell_value(plan, rows, rows_exact, values, index, d, array, words, seen,
          %Equal.sym(List<&2, U32>, Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(target), left) :
            {Nat.is_lt(t, Shape.logical_numel(_)) == True{} : Bool}
          inside,
          position => below =>
            +shaped : {Nat.is_lt(position, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool} =
              %right : {Nat.is_lt(position, Shape.logical_numel(_)) == True{} : Bool}
              below
            +exact_word = MatrixIndex.word_exact(position, shape, shaped)
            Equal.trans(Maybe<&2, F32>, Model.observe(F32, d, array, Refinement.route(d, U32.from_nat(position))),
              Core.checked_observe_dynamic(Core.Packed{shape, tensor}, U32.from_nat(position)), Laws.word_at(values, position),
              reads(U32.from_nat(position))(
                %Equal.sym(Nat, U32.to_nat(U32.from_nat(position)), position, exact_word) :
                  {Nat.is_lt(_, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}
                shaped),
              %exact_word :
                {Core.checked_observe_dynamic(Core.Packed{shape, tensor}, U32.from_nat(position)) == Laws.word_at(values, _) : Maybe<&2, F32>}
              listed(U32.from_nat(position))))))} :
    ReduceParallel.CellAt<d, array, rows, extents, t, Model.observe(F32, depth, built, path)> ->
    {Lifecycle.observe_allowed(depth, length, built, index, Nat.is_lt(t, length)) ==
      Laws.fiber(plan, values, index, Shape.logical_numel(Shape.dimensions(shape))) : Maybe<&2, F32>}}(
    %inside_length : ReduceParallel.CellSpec(d, array, rows, extents, t, _, Model.observe(F32, depth, built, path))
    %Parallel.output_position(depth, length, index, valid, inside_length) :
      ReduceParallel.CellIs(d, array, rows, extents, length, _, Model.observe(F32, depth, built, path))
    {\{Parallel.Fact: p => p} :
      Parallel.Fact<ReduceParallel.Cells(depth, d, array, rows, extents, length, 0n, built)> -> ReduceParallel.Cells(depth, d, array, rows, extents, length, 0n, built)}(
      ReduceParallel.build_cells(depth, splits, d, array, rows, extents, length, 0n, array, {==}, balanced))(path))}

def rows_mapping(+shape: Shape.Shape, -tensor: Core.Tensor<shape>, +target: Shape.Shape, +plan: Shape.BroadcastPlan, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool}, -exact: {length == Shape.logical_numel(Shape.dimensions(target)) : Nat},
  -kept: Core.Tensor<shape>, -array: Array<F32>, -taken: Parallel.Taken<Core.Packed{shape, tensor}, (Core.Packed{shape, kept}, array)>,
  +rows: List<&2, Shape.Row>,
  +rows_exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>},
  +left: {Shape.dimensions(Shape.left_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  +right: {Shape.dimensions(Shape.right_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>},
  +count: {Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Holds<Laws.MappingWith(Core.Packed{shape, tensor}, target, Done{plan},
    Reduce.parallel_rows(plan, target, depth, length, valid, exact, (Core.Packed{shape, kept}, array), rows))>:
  Holds{({==}, values => listed => index => inside =>
    {\{Parallel.Taken: d => balanced => reads => retained =>
      {\{Holds: proof => proof} :
        Holds<{Lifecycle.observe_allowed(depth, length, parallel_build(plan, depth, length, rows, array), index, Nat.is_lt(U32.to_nat(index), length)) ==
          Laws.fiber(plan, values, index, Shape.logical_numel(Shape.dimensions(shape))) : Maybe<&2, F32>}> ->
        {Lifecycle.observe_allowed(depth, length, parallel_build(plan, depth, length, rows, array), index, Nat.is_lt(U32.to_nat(index), length)) ==
          Laws.fiber(plan, values, index, Shape.logical_numel(Shape.dimensions(shape))) : Maybe<&2, F32>}}(
        rows_cell(shape, tensor, target, plan, depth, length, valid, exact, rows, rows_exact, array, d, balanced, reads, left, right, count,
          values, listed, index, inside))} :
      Parallel.Taken<Core.Packed{shape, tensor}, (Core.Packed{shape, kept}, array)> ->
      {Lifecycle.observe_allowed(depth, length, parallel_build(plan, depth, length, rows, array), index, Nat.is_lt(U32.to_nat(index), length)) ==
        Laws.fiber(plan, values, index, Shape.logical_numel(Shape.dimensions(shape))) : Maybe<&2, F32>}}(taken))}

def indexed_mapping(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape, +plan: Shape.BroadcastPlan, output: Core.Tensor<target>,
  +depth: Nat, +length: Nat, -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(target)) : Nat},
  map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>,
  -same: {Done{output} == Core.new(target, 0.0) : Result<&2, &1, Core.Error, Core.Tensor<target>>},
  +left: {Shape.dimensions(Shape.left_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  +right: {Shape.dimensions(Shape.right_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>},
  +count: {Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Holds<Laws.MappingWith(Core.Packed{shape, tensor}, target, Done{plan},
    Reduce.parallel_indexed(Core.Packed{shape, tensor}, target, plan, output, depth, length, valid, exact, map))>:
  match map:
    case Shape.Unindexed{zero}:
      Holds{({==}, values => listed => index => inside =>
        %Equal.trans(Nat, 0n, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), Shape.logical_numel(Shape.dimensions(shape)), zero, count) :
          {Core.checked_observe(target, output, index) == Laws.fiber(plan, values, index, _) : Maybe<&2, F32>}
        {\{Holds: proof => proof} : Holds<Inv(plan, values, target, 0n, output)> -> Inv(plan, values, target, 0n, output)}(
          zero_inv(plan, values, target, output, same))(index)(inside))}
    case Shape.Indexed{rows, same_rows}:
      MatmulProof.take_lift(shape, tensor,
        pair => Holds<Laws.MappingWith(Core.Packed{shape, tensor}, target, Done{plan}, Reduce.parallel_rows(plan, target, depth, length, valid, exact, pair, rows))>,
        kept => copy => taken => rows_mapping(shape, tensor, target, plan, depth, length, valid, exact, kept, copy,
          taken, rows, same_rows, left, right, count))

def parallel_mapping(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  result: Result<&2, &1, Core.Error, Core.Tensor<target>>,
  -same: {result == Core.new(target, 0.0) : Result<&2, &1, Core.Error, Core.Tensor<target>>},
  +left: {Shape.dimensions(Shape.left_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  +right: {Shape.dimensions(Shape.right_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>},
  +count: {Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Holds<Laws.MappingWith(Core.Packed{shape, tensor}, target, Done{plan}, Reduce.parallel_allocated(Core.Packed{shape, tensor}, target, plan, result))>:
  match result:
    case Fail{error}:
      Holds{Unit{}}
    case Done{Core.TTensor{storage, exact}}:
      match storage:
        case Storage.Empty{}:
          Holds{({==}, values => listed => index => inside =>
            Empty.absurd({Core.checked_observe(target, Core.TTensor{Storage.Empty{}, exact}, index) ==
              Laws.fiber(plan, values, index, Shape.logical_numel(Shape.dimensions(shape))) : Maybe<&2, F32>},
              Arith.lt_zero(U32.to_nat(index),
                %Equal.sym(Nat, 0n, Shape.logical_numel(Shape.dimensions(target)), exact) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
                inside)))}
        case Storage.Owned{+depth, +length, array, balanced, valid}:
          indexed_mapping(shape, tensor, target, plan, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, depth, length,
            valid, exact, Shape.index_map(plan), same, left, right, count)

def output_numel(+shape: Shape.Shape, +plan: Shape.BroadcastPlan,
  -equal: {True{} == Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape)) : Bool}) ->
  Holds<{Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}>:
  Holds{Equal.cong(List<&2, U32>, Nat, Shape.logical_numel, Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape),
    TransformProof.words_equal(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape), equal))}

def fitted_mapping(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape, +plan: Shape.BroadcastPlan, same: Bool,
  -decided: {same == Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape)) : Bool},
  -left: {Shape.dimensions(Shape.left_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  -right: {Shape.dimensions(Shape.right_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>}) ->
  Holds<Laws.MappingWith(Core.Packed{shape, tensor}, target, Done{plan}, Reduce.fitted(Core.Packed{shape, tensor}, target, plan, same))>:
  match same:
    case False{}:
      Holds{Unit{}}
    case True{}:
      Holds{{\{Holds: proof => proof} :
        Holds<Laws.MappingWith(Core.Packed{shape, tensor}, target, Done{plan}, Reduce.parallel_allocated(Core.Packed{shape, tensor}, target, plan, Core.new(target, 0.0)))> ->
        Laws.MappingWith(Core.Packed{shape, tensor}, target, Done{plan}, Reduce.parallel_allocated(Core.Packed{shape, tensor}, target, plan, Core.new(target, 0.0)))}(
        parallel_mapping(shape, tensor, target, plan, Core.new(target, 0.0), {==}, left, right,
          {\{Holds: proof => proof} :
            Holds<{Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}> ->
            {Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}}(
            output_numel(shape, plan, decided))))}

def planned_mapping(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>,
  -sources: {ShapeLaws.planned_dimensions(target, shape, result) == (Shape.dimensions(target), Shape.dimensions(shape)) : List<&2, U32> & List<&2, U32>}) ->
  Holds<Laws.MappingWith(Core.Packed{shape, tensor}, target, result, Reduce.planned(shape, tensor, target, result))>:
  match result:
    case Fail{error}:
      Holds{Unit{}}
    case Done{+plan}:
      fitted_mapping(shape, tensor, target, plan, Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape)), {==},
        Equal.cong(List<&2, U32> & List<&2, U32>, List<&2, U32>, Pair.fst(List<&2, U32>, List<&2, U32>),
          (Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(Shape.right_shape(plan))), (Shape.dimensions(target), Shape.dimensions(shape)), sources),
        Equal.cong(List<&2, U32> & List<&2, U32>, List<&2, U32>, Pair.snd(List<&2, U32>, List<&2, U32>),
          (Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(Shape.right_shape(plan))), (Shape.dimensions(target), Shape.dimensions(shape)), sources))

def Laws.reduce_mapping(input, target):
  Core.Packed{+shape, tensor} = input
  Laws.MappingModel{
    {\{Holds: proof => proof} :
      Holds<Laws.MappingWith(Core.Packed{shape, tensor}, target, Shape.broadcast_checked(target, shape),
        Reduce.planned(shape, tensor, target, Shape.broadcast_checked(target, shape)))> ->
      Laws.Mapping(Core.Packed{shape, tensor}, target, Reduce.reduce_to(Core.Packed{shape, tensor}, target))}(
      planned_mapping(shape, tensor, target, Shape.broadcast_checked(target, shape), PlanProof.broadcast_checked_plan(target, shape)))}

# The parallel reduction succeeds once the target is allocated.

def indexed_success(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape, +plan: Shape.BroadcastPlan, output: Core.Tensor<target>,
  +depth: Nat, +length: Nat, -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(target)) : Nat},
  map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>) ->
  Laws.Success<Reduce.parallel_indexed(Core.Packed{shape, tensor}, target, plan, output, depth, length, valid, exact, map)>:
  match map:
    case Shape.Unindexed{zero}:
      Laws.Success{{==}}
    case Shape.Indexed{rows, same}:
      MatmulProof.take_lift(shape, tensor, pair => Laws.Success<Reduce.parallel_rows(plan, target, depth, length, valid, exact, pair, rows)>,
        kept => copy => taken => Laws.Success{{==}})

def parallel_success(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  result: Result<&2, &1, Core.Error, Core.Tensor<target>>, -done: Allocation.TensorDone<target, result>) ->
  Laws.Success<Reduce.parallel_allocated(Core.Packed{shape, tensor}, target, plan, result)>:
  match result:
    case Fail{error}:
      Laws.Success{Empty.absurd({False{} == True{} : Bool},
        {\{Allocation.TensorDone: tensor => same => Arith.clash(Equal.cong(Result<&2, &1, Core.Error, Core.Tensor<target>>, Bool,
          found => Result.is_done(&2, &1, Core.Error, Core.Tensor<target>, found), Fail{error}, Done{tensor},
          Equal.sym(Result<&2, &1, Core.Error, Core.Tensor<target>>, Done{tensor}, Fail{error}, same)))} :
          Allocation.TensorDone<target, Fail{error}> -> Empty}(done))}
    case Done{Core.TTensor{storage, exact}}:
      match storage:
        case Storage.Empty{}:
          Laws.Success{{==}}
        case Storage.Owned{+depth, +length, array, balanced, valid}:
          indexed_success(shape, tensor, target, plan, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, depth, length,
            valid, exact, Shape.index_map(plan))

def fitted_success(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape, +plan: Shape.BroadcastPlan, same: Bool,
  -decided: {same == Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape)) : Bool},
  -truth: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>}) ->
  Laws.Success<Reduce.fitted(Core.Packed{shape, tensor}, target, plan, same)>:
  match same:
    case False{}:
      Laws.Success{Empty.absurd({False{} == True{} : Bool},
        Arith.clash(Equal.trans(Bool, False{}, Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape)), True{}, decided,
          %truth : {Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), _) == True{} : Bool}
          TransformProof.words_same_refl(Shape.dimensions(Shape.output_shape(plan))))))}
    case True{}:
      parallel_success(shape, tensor, target, plan, Core.new(target, 0.0), AllocationProof.core_new_done(target, 0.0))

def planned_success(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>,
  -sources: {ShapeLaws.planned_dimensions(target, shape, result) == (Shape.dimensions(target), Shape.dimensions(shape)) : List<&2, U32> & List<&2, U32>},
  -built: {ShapeLaws.broadcast_built(result) == True{} : Bool},
  -alignment: Shape.Broadcast<Shape.extents(target), Shape.extents(shape), Shape.extents(shape)>) ->
  Laws.Success<Reduce.planned(shape, tensor, target, result)>:
  match result:
    case Fail{error}:
      Laws.Success{Empty.absurd({False{} == True{} : Bool}, Arith.clash(built))}
    case Done{+plan}:
      fitted_success(shape, tensor, target, plan,
        Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape)), {==},
        ElementwiseProof.planned_output_dimensions(target, shape, plan, Shape.dimensions(shape), alignment, sources))

def Laws.reduce_valid_success(input, target, alignment):
  Core.Packed{+shape, tensor} = input
  planned_success(shape, tensor, target, Shape.broadcast_checked(target, shape), PlanProof.broadcast_checked_plan(target, shape),
    PlanProof.broadcast_checked_complete(target, shape, Shape.dimensions(shape), alignment, rank_of(shape), capacity_of(shape)), alignment)

def empty_fiber(+plan: Shape.BroadcastPlan, +values: List<&2, F32>, +index: U32, +count: Nat,
  -empty: {count == 0n : Nat}) -> Holds<{Laws.fiber(plan, values, index, count) == Some{0.0} : Maybe<&2, F32>}>:
  Holds{
    %Equal.sym(Nat, count, 0n, empty) : {Laws.fiber(plan, values, index, _) == Some{0.0} : Maybe<&2, F32>}
    {==}}

def below_zero(+number: Nat) -> Holds<{Nat.is_lt(number, 0n) == False{} : Bool}>:
  match number:
    case 0n:
      Holds{{==}}
    case 1n+rest:
      Holds{{==}}

def empty_unlisted(+depth: Nat, +length: Nat, -array: Array<F32>, +source: U32,
  -empty: {length == 0n : Nat}) -> Holds<{Lifecycle.observe_allowed(depth, length, array, source, Nat.is_lt(U32.to_nat(source), length)) == None{} : Maybe<&2, F32>}>:
  Holds{
    %Equal.sym(Nat, length, 0n, empty) :
      {Lifecycle.observe_allowed(depth, length, array, source, Nat.is_lt(U32.to_nat(source), _)) == None{} : Maybe<&2, F32>}
    %Equal.sym(Bool, Nat.is_lt(U32.to_nat(source), 0n), False{},
      {\{Holds: proof => proof} : Holds<{Nat.is_lt(U32.to_nat(source), 0n) == False{} : Bool}> -> {Nat.is_lt(U32.to_nat(source), 0n) == False{} : Bool}}(
        below_zero(U32.to_nat(source)))) :
      {Lifecycle.observe_allowed(depth, length, array, source, _) == None{} : Maybe<&2, F32>}
    {==}}

def listed_empty(input: Core.Dynamic, -empty: {Shape.logical_numel(Totality.dimensions(input)) == 0n : Nat}) -> Holds<Laws.Listed(input, Nil{})>:
  Core.Packed{+shape, Core.TTensor{storage, exact}} = input
  match storage:
    case Storage.Empty{}:
      Holds{source => {==}}
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      Holds{source => {\{Holds: proof => proof} :
        Holds<{Lifecycle.observe_allowed(depth, length, array, source, Nat.is_lt(U32.to_nat(source), length)) == None{} : Maybe<&2, F32>}> ->
        {Lifecycle.observe_allowed(depth, length, array, source, Nat.is_lt(U32.to_nat(source), length)) == None{} : Maybe<&2, F32>}}(
        empty_unlisted(depth, length, array, source, Equal.trans(Nat, length, Shape.logical_numel(Shape.dimensions(shape)), 0n, exact, empty)))}

def below_self(+number: Nat) -> {Nat.is_lt(number, number) == False{} : Bool}:
  match number:
    case 0n:
      {==}
    case 1n+rest:
      below_self(rest)

def neither_below(+left: Nat, +right: Nat, first: {Nat.is_lt(left, right) == False{} : Bool},
  second: {Nat.is_lt(right, left) == False{} : Bool}) -> {left == right : Nat}:
  match left right:
    case 0n 0n:
      {==}
    case 0n 1n+r:
      Empty.absurd({0n == 1n+r : Nat}, Arith.clash_true(first))
    case 1n+l 0n:
      Empty.absurd({1n+l == 0n : Nat}, Arith.clash_true(second))
    case 1n+l 1n+r:
      Equal.cong(Nat, Nat, n => 1n+n, l, r, neither_below(l, r, first, second))

def word_presence(+values: List<&2, F32>, +position: Nat) ->
  {Nat.is_lt(position, List.length(&2, F32, values)) == TransformProof.maybe_present(Laws.word_at(values, position)) : Bool}:
  match values position:
    case Nil{} 0n:
      {==}
    case Nil{} 1n+next:
      {==}
    case value <> rest 0n:
      {==}
    case value <> rest 1n+next:
      word_presence(rest, next)

def absent_observation(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +source: U32,
  -outside: {Nat.is_lt(U32.to_nat(source), Shape.logical_numel(Shape.dimensions(shape))) == False{} : Bool}) ->
  Holds<{Core.checked_observe(shape, tensor, source) == None{} : Maybe<&2, F32>}>:
  Core.TTensor{storage, exact} = tensor
  match storage:
    case Storage.Empty{}:
      Holds{{==}}
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      Holds{
        %Equal.sym(Bool, Nat.is_lt(U32.to_nat(source), length), False{},
          %Equal.sym(Nat, length, Shape.logical_numel(Shape.dimensions(shape)), exact) : {Nat.is_lt(U32.to_nat(source), _) == False{} : Bool}
          outside) : {Lifecycle.observe_allowed(depth, length, array, source, _) == None{} : Maybe<&2, F32>}
        {==}}

def observed_presence(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +source: U32, +inside: Bool,
  -decision: {inside == Nat.is_lt(U32.to_nat(source), Shape.logical_numel(Shape.dimensions(shape))) : Bool}) ->
  Holds<{TransformProof.maybe_present(Core.checked_observe(shape, tensor, source)) == inside : Bool}>:
  match inside:
    case False{}:
      Holds{Equal.cong(Maybe<&2, F32>, Bool, TransformProof.maybe_present, Core.checked_observe(shape, tensor, source), None{},
        {\{Holds: proof => proof} : Holds<{Core.checked_observe(shape, tensor, source) == None{} : Maybe<&2, F32>}> -> {Core.checked_observe(shape, tensor, source) == None{} : Maybe<&2, F32>}}(
          absent_observation(shape, tensor, source, Equal.sym(Bool, False{}, Nat.is_lt(U32.to_nat(source), Shape.logical_numel(Shape.dimensions(shape))), decision))))}
    case True{}:
      Holds{{\{TransformProof.PresentObservation: value => equal =>
        Equal.cong(Maybe<&2, F32>, Bool, TransformProof.maybe_present, Core.checked_observe(shape, tensor, source), Some{value}, equal)} :
        TransformProof.PresentObservation<shape, tensor, source> -> {TransformProof.maybe_present(Core.checked_observe(shape, tensor, source)) == True{} : Bool}}(
        TransformProof.present_observation(shape, tensor, source, Equal.sym(Bool, True{}, Nat.is_lt(U32.to_nat(source), Shape.logical_numel(Shape.dimensions(shape))), decision)))}

def same_presence(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +values: List<&2, F32>,
  -listed: Laws.Listed(Core.Packed{shape, tensor}, values), +source: U32) ->
  Holds<{Nat.is_lt(U32.to_nat(source), List.length(&2, F32, values)) == Nat.is_lt(U32.to_nat(source), Shape.logical_numel(Shape.dimensions(shape))) : Bool}>:
  Holds{Equal.trans(Bool, Nat.is_lt(U32.to_nat(source), List.length(&2, F32, values)), TransformProof.maybe_present(Laws.word_at(values, U32.to_nat(source))),
    Nat.is_lt(U32.to_nat(source), Shape.logical_numel(Shape.dimensions(shape))),
    word_presence(values, U32.to_nat(source)),
    Equal.trans(Bool, TransformProof.maybe_present(Laws.word_at(values, U32.to_nat(source))),
      TransformProof.maybe_present(Core.checked_observe(shape, tensor, source)), Nat.is_lt(U32.to_nat(source), Shape.logical_numel(Shape.dimensions(shape))),
      Equal.cong(Maybe<&2, F32>, Bool, TransformProof.maybe_present, Laws.word_at(values, U32.to_nat(source)),
        Core.checked_observe(shape, tensor, source),
        Equal.sym(Maybe<&2, F32>, Core.checked_observe(shape, tensor, source), Laws.word_at(values, U32.to_nat(source)), listed(source))),
      {\{Holds: proof => proof} : Holds<{TransformProof.maybe_present(Core.checked_observe(shape, tensor, source)) == Nat.is_lt(U32.to_nat(source), Shape.logical_numel(Shape.dimensions(shape))) : Bool}> -> {TransformProof.maybe_present(Core.checked_observe(shape, tensor, source)) == Nat.is_lt(U32.to_nat(source), Shape.logical_numel(Shape.dimensions(shape))) : Bool}}(
        observed_presence(shape, tensor, source, Nat.is_lt(U32.to_nat(source), Shape.logical_numel(Shape.dimensions(shape))), {==}))))}

def natural_presence(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +values: List<&2, F32>,
  -listed: Laws.Listed(Core.Packed{shape, tensor}, values), +position: Nat, -within: Shape.LE(position, 1048576n)) ->
  Holds<{Nat.is_lt(position, List.length(&2, F32, values)) == Nat.is_lt(position, Shape.logical_numel(Shape.dimensions(shape))) : Bool}>:
  Holds{
    %Conversion.capacity_natural_word_exact(position, within) : {Nat.is_lt(_, List.length(&2, F32, values)) == Nat.is_lt(_, Shape.logical_numel(Shape.dimensions(shape))) : Bool}
    {\{Holds: proof => proof} : Holds<{Nat.is_lt(U32.to_nat(U32.from_nat(position)), List.length(&2, F32, values)) == Nat.is_lt(U32.to_nat(U32.from_nat(position)), Shape.logical_numel(Shape.dimensions(shape))) : Bool}> -> {Nat.is_lt(U32.to_nat(U32.from_nat(position)), List.length(&2, F32, values)) == Nat.is_lt(U32.to_nat(U32.from_nat(position)), Shape.logical_numel(Shape.dimensions(shape))) : Bool}}(
      same_presence(shape, tensor, values, listed, U32.from_nat(position)))}

def length_decided(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +values: List<&2, F32>,
  -listed: Laws.Listed(Core.Packed{shape, tensor}, values), +below: Bool,
  -decision: {below == Nat.is_lt(List.length(&2, F32, values), Shape.logical_numel(Shape.dimensions(shape))) : Bool}) -> Holds<{List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}>:
  match below:
    case True{}:
      Holds{Empty.absurd({List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, Arith.clash(
        Equal.trans(Bool, False{}, Nat.is_lt(List.length(&2, F32, values), List.length(&2, F32, values)), True{},
          Equal.sym(Bool, Nat.is_lt(List.length(&2, F32, values), List.length(&2, F32, values)), False{}, below_self(List.length(&2, F32, values))),
          Equal.trans(Bool, Nat.is_lt(List.length(&2, F32, values), List.length(&2, F32, values)), Nat.is_lt(List.length(&2, F32, values), Shape.logical_numel(Shape.dimensions(shape))), True{},
            {\{Holds: proof => proof} : Holds<{Nat.is_lt(List.length(&2, F32, values), List.length(&2, F32, values)) == Nat.is_lt(List.length(&2, F32, values), Shape.logical_numel(Shape.dimensions(shape))) : Bool}> -> {Nat.is_lt(List.length(&2, F32, values), List.length(&2, F32, values)) == Nat.is_lt(List.length(&2, F32, values), Shape.logical_numel(Shape.dimensions(shape))) : Bool}}(
              natural_presence(shape, tensor, values, listed, List.length(&2, F32, values),
              Shape.LE.from(List.length(&2, F32, values), 1048576n, Equal.sym(Bool, Nat.is_le(List.length(&2, F32, values), 1048576n), True{},
                Conversion.le_trans(List.length(&2, F32, values), Shape.logical_numel(Shape.dimensions(shape)), 1048576n,
                  Conversion.lt_le(List.length(&2, F32, values), Shape.logical_numel(Shape.dimensions(shape)), Equal.sym(Bool, True{}, Nat.is_lt(List.length(&2, F32, values), Shape.logical_numel(Shape.dimensions(shape))), decision)),
                  Equal.sym(Bool, True{}, Nat.is_le(Shape.logical_numel(Shape.dimensions(shape)), 1048576n), Nats.le_true(Shape.logical_numel(Shape.dimensions(shape)), 1048576n, capacity_of(shape)))))))),
            Equal.sym(Bool, True{}, Nat.is_lt(List.length(&2, F32, values), Shape.logical_numel(Shape.dimensions(shape))), decision)))))}
    case False{}:
      Holds{neither_below(List.length(&2, F32, values), Shape.logical_numel(Shape.dimensions(shape)), Equal.sym(Bool, False{}, Nat.is_lt(List.length(&2, F32, values), Shape.logical_numel(Shape.dimensions(shape))), decision),
        Equal.trans(Bool, Nat.is_lt(Shape.logical_numel(Shape.dimensions(shape)), List.length(&2, F32, values)), Nat.is_lt(Shape.logical_numel(Shape.dimensions(shape)), Shape.logical_numel(Shape.dimensions(shape))), False{},
          {\{Holds: proof => proof} : Holds<{Nat.is_lt(Shape.logical_numel(Shape.dimensions(shape)), List.length(&2, F32, values)) == Nat.is_lt(Shape.logical_numel(Shape.dimensions(shape)), Shape.logical_numel(Shape.dimensions(shape))) : Bool}> -> {Nat.is_lt(Shape.logical_numel(Shape.dimensions(shape)), List.length(&2, F32, values)) == Nat.is_lt(Shape.logical_numel(Shape.dimensions(shape)), Shape.logical_numel(Shape.dimensions(shape))) : Bool}}(
            natural_presence(shape, tensor, values, listed, Shape.logical_numel(Shape.dimensions(shape)), capacity_of(shape))),
          below_self(Shape.logical_numel(Shape.dimensions(shape)))))}

def Laws.listed_length(input, values, listed):
  Core.Packed{+shape, tensor} = input
  Laws.Length{{\{Holds: proof => proof} : Holds<{List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}> -> {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}}(
    length_decided(shape, tensor, values, listed, Nat.is_lt(List.length(&2, F32, values), Shape.logical_numel(Shape.dimensions(shape))), {==}))}

def zeros_from_mapping(-input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>,
  -mapping: Laws.MappingWith(input, target, Done{plan}, pair),
  -listed: Laws.Listed(input, Nil{}),
  -empty: {Shape.logical_numel(Totality.dimensions(input)) == 0n : Nat}) -> Laws.Zeros<target, pair>:
  (owner, outcome) = pair
  match outcome:
    case Fail{error}:
      Laws.Zeros{Unit{}}
    case Done{output}:
      Laws.Zeros{index => inside => Equal.trans(Maybe<&2, F32>, Core.checked_observe_dynamic(output, index),
        Laws.fiber(plan, Nil{}, index, Shape.logical_numel(Totality.dimensions(input))), Some{0.0},
        mapping_values(input, target, plan, output, mapping)(Nil{})(listed)(index)(inside),
        {\{Holds: proof => proof} : Holds<{Laws.fiber(plan, Nil{}, index, Shape.logical_numel(Totality.dimensions(input))) == Some{0.0} : Maybe<&2, F32>}> ->
          {Laws.fiber(plan, Nil{}, index, Shape.logical_numel(Totality.dimensions(input))) == Some{0.0} : Maybe<&2, F32>}}(
          empty_fiber(plan, Nil{}, index, Shape.logical_numel(Totality.dimensions(input)), empty)))}

def planned_zeros(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>,
  -sources: {ShapeLaws.planned_dimensions(target, shape, result) == (Shape.dimensions(target), Shape.dimensions(shape)) : List<&2, U32> & List<&2, U32>},
  -empty: {Shape.logical_numel(Shape.dimensions(shape)) == 0n : Nat}) ->
  Laws.Zeros<target, Reduce.planned(shape, tensor, target, result)>:
  match result:
    case Fail{error}:
      Laws.Zeros{Unit{}}
    case Done{+plan}:
      zeros_from_mapping(Core.Packed{shape, tensor}, target, plan, Reduce.planned(shape, tensor, target, Done{plan}),
        {\{Holds: proof => proof} :
          Holds<Laws.MappingWith(Core.Packed{shape, tensor}, target, Done{plan}, Reduce.planned(shape, tensor, target, Done{plan}))> ->
          Laws.MappingWith(Core.Packed{shape, tensor}, target, Done{plan}, Reduce.planned(shape, tensor, target, Done{plan}))}(
          planned_mapping(shape, tensor, target, Done{plan}, sources)),
        {\{Holds: proof => proof} : Holds<Laws.Listed(Core.Packed{shape, tensor}, Nil{})> -> Laws.Listed(Core.Packed{shape, tensor}, Nil{})}(
          listed_empty(Core.Packed{shape, tensor}, empty)),
        empty)

def Laws.reduce_empty_zeros(input, target, empty):
  Core.Packed{+shape, tensor} = input
  planned_zeros(shape, tensor, target, Shape.broadcast_checked(target, shape), PlanProof.broadcast_checked_plan(target, shape), empty)

def Laws.reduce_shape_error(input, target, error, rejected):
  Core.Packed{+shape, tensor} = input
  %Equal.sym(Result<&2, &2, Shape.Error, Shape.BroadcastPlan>, Shape.broadcast_checked(target, shape), Fail{error}, rejected) :
    {Reduce.planned(shape, tensor, target, _) == (Core.Packed{shape, tensor}, Fail{Reduce.TensorError{Core.ShapeError{error}}}) :
      Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>}
  {==}

def fitted_not_reducible(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan, +dims: List<&2, U32>, flag: Bool,
  decided: {flag == Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), dims) : Bool},
  differ: {Shape.dimensions(Shape.output_shape(plan)) != dims : List<&2, U32>}) ->
  {Reduce.fitted(input, target, plan, flag) == (input, Fail{Reduce.NotReducible{}}) :
    Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>}:
  match flag:
    case False{}:
      {==}
    case True{}:
      Empty.absurd({Reduce.fitted(input, target, plan, True{}) == (input, Fail{Reduce.NotReducible{}}) :
        Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>},
        differ(TransformProof.words_equal(Shape.dimensions(Shape.output_shape(plan)), dims, decided)))

def Laws.reduce_not_reducible(input, target, plan, planned, differ):
  Core.Packed{+shape, tensor} = input
  %Equal.sym(Result<&2, &2, Shape.Error, Shape.BroadcastPlan>, Shape.broadcast_checked(target, shape), Done{plan}, planned) :
    {Reduce.planned(shape, tensor, target, _) == (Core.Packed{shape, tensor}, Fail{Reduce.NotReducible{}}) :
      Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>}
  fitted_not_reducible(Core.Packed{shape, tensor}, target, plan, Shape.dimensions(shape),
    Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape)), {==}, differ)
