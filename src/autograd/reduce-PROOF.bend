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

import ../tensor/elementwise-totality.bend as Totality

import ../tensor/elementwise-PROOF.bend as ElementwiseProof

import ../tensor/core-PROOF.bend as CoreProof

import ../tensor/transform-PROOF.bend as TransformProof

import ../tensor/transform-totality.bend as TransformTotality

import ../tensor/projection-LAWS.bend as Projection

import ../tensor/projection-PROOF.bend as ProjectionProof

import ../tensor/allocation-spec.bend as Allocation

import ../tensor/allocation-PROOF.bend as AllocationProof

import ../tensor/construction-contract-PROOF.bend as ConstructionProof

import ../tensor/word-conversion.bend as Conversion

import ../tensor/observation.bend as Observation

import ./reduce.bend as Reduce

import ./reduce-LAWS.bend as Laws

type Holds<-P: Type> is Data:
  Holds{-proof: P}

def state_input(-target: Shape.Shape, state: Reduce.State<target>) -> Core.Dynamic:
  Reduce.State{input, output} = state
  input

def state_output(-target: Shape.Shape, state: Reduce.State<target>) -> Core.Tensor<target>:
  Reduce.State{input, output} = state
  output

def step_owner(-target: Shape.Shape, result: Reduce.Step<target>) -> Core.Dynamic:
  match result:
    case Reduce.Stop{input, error}:
      input
    case Reduce.Continue{state}:
      state_input(target, state)

def step_continued(-target: Shape.Shape, result: Reduce.Step<target>) -> Bool:
  match result:
    case Reduce.Stop{input, error}:
      False{}
    case Reduce.Continue{state}:
      True{}

def loop_owner(-target: Shape.Shape, pair: Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>>) -> Core.Dynamic:
  (input, result) = pair
  input

def loop_succeeded(-target: Shape.Shape, pair: Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>>) -> Bool:
  (input, result) = pair
  Result.is_done(&2, &1, Core.Error, Core.Tensor<target>, result)

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

def write_owner(-target: Shape.Shape, -input: Core.Dynamic, pair: Core.Tensor<target> & Result<&2, &2, Core.Error, Unit>) ->
  Holds<{step_owner(target, Reduce.step_write(target, input, pair)) == input : Core.Dynamic}>:
  (output, result) = pair
  match result:
    case Fail{error}:
      Holds{{==}}
    case Done{unit}:
      Holds{{==}}

def output_owner(-target: Shape.Shape, +target_index: U32, -input: Core.Dynamic, +value: F32,
  pair: Core.Tensor<target> & Result<&2, &2, Core.Error, F32>) ->
  Holds<{step_owner(target, Reduce.step_output(target, target_index, input, value, pair)) == input : Core.Dynamic}>:
  (output, result) = pair
  match result:
    case Fail{error}:
      Holds{{==}}
    case Done{+accumulated}:
      write_owner(target, input, Core.write_flat(target, output, target_index, F32.add(accumulated, value)))

def input_retains(-target: Shape.Shape, +target_index: U32, output: Core.Tensor<target>, -original: Core.Dynamic,
  pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>,
  -same: Totality.SameSource(original, ElementwiseProof.dynamic_read_owner(pair))) ->
  Holds<Totality.SameSource(original, step_owner(target, Reduce.step_input(target, target_index, output, pair)))>:
  (input, result) = pair
  match result:
    case Fail{error}:
      Holds{same}
    case Done{value}:
      Holds{property => Equal.trans(Totality.Inspection,
        Totality.inspect(property, step_owner(target, Reduce.step_output(target, target_index, input, value, Core.read_flat(target, output, target_index)))),
        Totality.inspect(property, input), Totality.inspect(property, original),
        Equal.cong(Core.Dynamic, Totality.Inspection, owner => Totality.inspect(property, owner),
          step_owner(target, Reduce.step_output(target, target_index, input, value, Core.read_flat(target, output, target_index))), input,
          {\{Holds: proof => proof} : Holds<{step_owner(target, Reduce.step_output(target, target_index, input, value, Core.read_flat(target, output, target_index))) == input : Core.Dynamic}> ->
            {step_owner(target, Reduce.step_output(target, target_index, input, value, Core.read_flat(target, output, target_index))) == input : Core.Dynamic}}(
            output_owner(target, target_index, input, value, Core.read_flat(target, output, target_index)))),
        same(property))}

def indices_retains(-target: Shape.Shape, input: Core.Dynamic, output: Core.Tensor<target>,
  result: Result<&2, &1, Core.Error, U32 & U32>) ->
  Holds<Totality.SameSource(input, step_owner(target, Reduce.step_indices(target, Reduce.State{input, output}, result)))>:
  match result:
    case Fail{error}:
      Holds{property => {==}}
    case Done{indices}:
      (+target_index, +source_index) = indices
      Holds{{\{ElementwiseProof.SourceModel: same =>
        {\{Holds: proof => proof} :
          Holds<Totality.SameSource(input, step_owner(target, Reduce.step_input(target, target_index, output, Core.read_dynamic(input, source_index))))> ->
          Totality.SameSource(input, step_owner(target, Reduce.step_input(target, target_index, output, Core.read_dynamic(input, source_index))))}(
          input_retains(target, target_index, output, input, Core.read_dynamic(input, source_index), same))} :
        ElementwiseProof.SourceModel<input, ElementwiseProof.dynamic_read_owner(Core.read_dynamic(input, source_index))> ->
        Totality.SameSource(input, step_owner(target, Reduce.step_input(target, target_index, output, Core.read_dynamic(input, source_index))))}(
          ElementwiseProof.dynamic_read_retains(input, source_index))}

def step_retains(+plan: Shape.BroadcastPlan, -target: Shape.Shape, +index: U32, state: Reduce.State<target>) ->
  Holds<Totality.SameSource(state_input(target, state), step_owner(target, Reduce.step(plan, target, index, state)))>:
  Reduce.State{input, output} = state
  indices_retains(target, input, output, Elementwise.input_indices(plan, index))

def conversion_retains(-target: Shape.Shape, state: Reduce.State<target>, -original: Core.Dynamic,
  -same: Totality.SameSource(original, state_input(target, state))) ->
  Holds<Totality.SameSource(original, loop_owner(target, Reduce.conversion_error(target, state)))>:
  Reduce.State{input, output} = state
  Holds{same}

def continue_retains(+remaining: Nat, +plan: Shape.BroadcastPlan, -target: Shape.Shape, +cursor: Nat, result: Reduce.Step<target>,
  -original: Core.Dynamic, -same: Totality.SameSource(original, step_owner(target, result)),
  -rec: @state: Reduce.State<target> -> @-same: Totality.SameSource(original, state_input(target, state)) ->
    Holds<Totality.SameSource(original, loop_owner(target, Reduce.loop(remaining, plan, target, 1n+cursor, state)))>) ->
  Holds<Totality.SameSource(original, loop_owner(target,
    Reduce.continue_step(target, state => Reduce.loop(remaining, plan, target, 1n+cursor, state), result)))>:
  match result:
    case Reduce.Stop{input, error}:
      Holds{same}
    case Reduce.Continue{state}:
      Holds{{\{Holds: proof => proof} : Holds<Totality.SameSource(original, loop_owner(target, Reduce.loop(remaining, plan, target, 1n+cursor, state)))> ->
        Totality.SameSource(original, loop_owner(target, Reduce.loop(remaining, plan, target, 1n+cursor, state)))}(rec(state, same))}

def decided_retains(+remaining: Nat, +plan: Shape.BroadcastPlan, -target: Shape.Shape, +cursor: Nat, +index: U32,
  state: Reduce.State<target>, decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>,
  -original: Core.Dynamic, -same: Totality.SameSource(original, state_input(target, state)),
  -rec: @state: Reduce.State<target> -> @-same: Totality.SameSource(original, state_input(target, state)) ->
    Holds<Totality.SameSource(original, loop_owner(target, Reduce.loop(remaining, plan, target, 1n+cursor, state)))>) ->
  Holds<Totality.SameSource(original, loop_owner(target, Reduce.decided(plan, target, cursor, index, state,
    state => Reduce.loop(remaining, plan, target, 1n+cursor, state), decision)))>:
  match decision:
    case Shape.BFalse{equal}:
      conversion_retains(target, state, original, same)
    case Shape.BTrue{equal}:
      continue_retains(remaining, plan, target, cursor, Reduce.step(plan, target, index, state), original,
        {\{Holds: proof => proof} : Holds<Totality.SameSource(original, step_owner(target, Reduce.step(plan, target, index, state)))> ->
          Totality.SameSource(original, step_owner(target, Reduce.step(plan, target, index, state)))}(
          same_trans(original, state_input(target, state), step_owner(target, Reduce.step(plan, target, index, state)), same,
            {\{Holds: proof => proof} : Holds<Totality.SameSource(state_input(target, state), step_owner(target, Reduce.step(plan, target, index, state)))> ->
              Totality.SameSource(state_input(target, state), step_owner(target, Reduce.step(plan, target, index, state)))}(
              step_retains(plan, target, index, state)))), rec)

def loop_retains(fuel: Nat, +plan: Shape.BroadcastPlan, -target: Shape.Shape, +cursor: Nat, state: Reduce.State<target>,
  -original: Core.Dynamic, -same: Totality.SameSource(original, state_input(target, state))) ->
  Holds<Totality.SameSource(original, loop_owner(target, Reduce.loop(fuel, plan, target, cursor, state)))>:
  match fuel:
    case 0n:
      Reduce.State{input, output} = state
      Holds{same}
    case 1n+remaining:
      decided_retains(remaining, plan, target, cursor, U32.from_nat(cursor), state,
        Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)), original, same,
        state => same => loop_retains(remaining, plan, target, 1n+cursor, state, original, same))

def finish_retains(+target: Shape.Shape, -original: Core.Dynamic, pair: Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>>,
  -same: Totality.SameSource(original, loop_owner(target, pair))) -> Laws.Retained<original, Reduce.finish(target, pair)>:
  (input, result) = pair
  match result:
    case Fail{error}:
      Laws.Retained{same}
    case Done{output}:
      Laws.Retained{same}

def allocated_retains(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  result: Result<&2, &1, Core.Error, Core.Tensor<target>>) -> Laws.Retained<input, Reduce.allocated(input, target, plan, result)>:
  match result:
    case Fail{error}:
      Laws.Retained{property => {==}}
    case Done{output}:
      finish_retains(target, input, Reduce.loop(Shape.elements(Shape.output_shape(plan)), plan, target, 0n, Reduce.State{input, output}),
        {\{Holds: proof => proof} :
          Holds<Totality.SameSource(input, loop_owner(target, Reduce.loop(Shape.elements(Shape.output_shape(plan)), plan, target, 0n, Reduce.State{input, output})))> ->
          Totality.SameSource(input, loop_owner(target, Reduce.loop(Shape.elements(Shape.output_shape(plan)), plan, target, 0n, Reduce.State{input, output})))}(
          loop_retains(Shape.elements(Shape.output_shape(plan)), plan, target, 0n, Reduce.State{input, output}, input, property => {==})))

def fitted_retains(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan, same: Bool) ->
  Laws.Retained<input, Reduce.fitted(input, target, plan, same)>:
  match same:
    case False{}:
      Laws.Retained{property => {==}}
    case True{}:
      allocated_retains(input, target, plan, Core.new(target, 0.0))

def planned_retains(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) ->
  Laws.Retained<Core.Packed{shape, tensor}, Reduce.planned(shape, tensor, target, result)>:
  match result:
    case Fail{error}:
      Laws.Retained{property => {==}}
    case Done{+plan}:
      fitted_retains(Core.Packed{shape, tensor}, target, plan,
        Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape)))

def Laws.reduce_retains_input(input, target):
  Core.Packed{+shape, tensor} = input
  planned_retains(shape, tensor, target, Shape.broadcast_checked(target, shape))

def Inv(+plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target: Shape.Shape, +cursor: Nat, output: Core.Tensor<target>) -> Type:
  @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool} ->
    {Core.checked_observe(target, output, index) == Laws.fiber(plan, values, index, cursor) : Maybe<&2, F32>}

def StepInv(+plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target: Shape.Shape, +cursor: Nat, result: Reduce.Step<target>) -> Type:
  match result:
    case Reduce.Stop{input, error}:
      Unit
    case Reduce.Continue{state}:
      Inv(plan, values, target, 1n+cursor, state_output(target, state))

def LoopInv(+plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target: Shape.Shape, +limit: Nat,
  pair: Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>>) -> Type:
  (input, result) = pair
  match result:
    case Fail{error}:
      Unit
    case Done{output}:
      Inv(plan, values, target, limit, output)

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

def word_exact(+left: U32, +right: U32, equal: {True{} == Nat.is_eq(U32.to_nat(left), U32.to_nat(right)) : Bool}) -> Holds<{left == right : U32}>:
  Holds{
    %Conversion.word_natural_exact(left) : {_ == right : U32}
    %Conversion.word_natural_exact(right) : {U32.from_nat(U32.to_nat(left)) == _ : U32}
    Equal.cong(Nat, U32, U32.from_nat, U32.to_nat(left), U32.to_nat(right),
      Lifecycle.natural_equal(U32.to_nat(left), U32.to_nat(right), equal))}

def written_true(+plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target: Shape.Shape, +cursor: Nat,
  -output: Core.Tensor<target>, -written: Core.Tensor<target>, +target_index: U32, +source_index: U32, +accumulated: F32, +value: F32,
  +index: U32, -equal: {True{} == Nat.is_eq(U32.to_nat(target_index), U32.to_nat(index)) : Bool},
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool},
  -inv: Inv(plan, values, target, cursor, output),
  -read: {Core.checked_observe(target, output, target_index) == Some{accumulated} : Maybe<&2, F32>},
  -stored: {Core.checked_observe(target, written, target_index) == Some{F32.add(accumulated, value)} : Maybe<&2, F32>},
  -word: {Laws.word_at(values, U32.to_nat(source_index)) == Some{value} : Maybe<&2, F32>},
  -offsets: {Projection.offset_spec(plan, cursor) == Some{(target_index, source_index)} : Maybe<&1, U32 & U32>}) ->
  Holds<{Core.checked_observe(target, written, index) == Laws.fiber(plan, values, index, 1n+cursor) : Maybe<&2, F32>}>:
  Holds{
    %{\{Holds: proof => proof} : Holds<{target_index == index : U32}> -> {target_index == index : U32}}(word_exact(target_index, index, equal)) :
      {Core.checked_observe(target, written, _) == Laws.contribute(values, _, Laws.fiber(plan, values, _, cursor), Projection.offset_spec(plan, cursor)) : Maybe<&2, F32>}
    %Equal.sym(Maybe<&1, U32 & U32>, Projection.offset_spec(plan, cursor), Some{(target_index, source_index)}, offsets) :
      {Core.checked_observe(target, written, target_index) == Laws.contribute(values, target_index, Laws.fiber(plan, values, target_index, cursor), _) : Maybe<&2, F32>}
    %Equal.sym(Bool, Nat.is_eq(U32.to_nat(target_index), U32.to_nat(target_index)), True{}, Arith.eq_refl(U32.to_nat(target_index))) :
      {Core.checked_observe(target, written, target_index) ==
        Laws.matched(Laws.fiber(plan, values, target_index, cursor), Laws.word_at(values, U32.to_nat(source_index)), _) : Maybe<&2, F32>}
    %Equal.sym(Maybe<&2, F32>, Laws.word_at(values, U32.to_nat(source_index)), Some{value}, word) :
      {Core.checked_observe(target, written, target_index) == Laws.add_word(Laws.fiber(plan, values, target_index, cursor), _) : Maybe<&2, F32>}
    %Equal.trans(Maybe<&2, F32>, Some{accumulated}, Core.checked_observe(target, output, target_index), Laws.fiber(plan, values, target_index, cursor),
      Equal.sym(Maybe<&2, F32>, Core.checked_observe(target, output, target_index), Some{accumulated}, read),
      inv(target_index)(
        %Equal.sym(U32, target_index, index, {\{Holds: proof => proof} : Holds<{target_index == index : U32}> -> {target_index == index : U32}}(word_exact(target_index, index, equal))) :
          {Nat.is_lt(U32.to_nat(_), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}
        inside)) :
      {Core.checked_observe(target, written, target_index) == Laws.add_word(_, Some{value}) : Maybe<&2, F32>}
    stored}

def written_false(+plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target: Shape.Shape, +cursor: Nat,
  -output: Core.Tensor<target>, -read_owner: Core.Tensor<target>, -written: Core.Tensor<target>,
  +target_index: U32, +source_index: U32, +index: U32,
  -equal: {False{} == Nat.is_eq(U32.to_nat(target_index), U32.to_nat(index)) : Bool},
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool},
  -inv: Inv(plan, values, target, cursor, output),
  -kept: {Core.checked_observe(target, read_owner, index) == Core.checked_observe(target, output, index) : Maybe<&2, F32>},
  -frame: {Core.checked_observe(target, written, index) == Core.checked_observe(target, read_owner, index) : Maybe<&2, F32>},
  -offsets: {Projection.offset_spec(plan, cursor) == Some{(target_index, source_index)} : Maybe<&1, U32 & U32>}) ->
  Holds<{Core.checked_observe(target, written, index) == Laws.fiber(plan, values, index, 1n+cursor) : Maybe<&2, F32>}>:
  Holds{
    %Equal.sym(Maybe<&1, U32 & U32>, Projection.offset_spec(plan, cursor), Some{(target_index, source_index)}, offsets) :
      {Core.checked_observe(target, written, index) == Laws.contribute(values, index, Laws.fiber(plan, values, index, cursor), _) : Maybe<&2, F32>}
    %equal :
      {Core.checked_observe(target, written, index) ==
        Laws.matched(Laws.fiber(plan, values, index, cursor), Laws.word_at(values, U32.to_nat(source_index)), _) : Maybe<&2, F32>}
    Equal.trans(Maybe<&2, F32>, Core.checked_observe(target, written, index), Core.checked_observe(target, read_owner, index),
      Laws.fiber(plan, values, index, cursor), frame,
      Equal.trans(Maybe<&2, F32>, Core.checked_observe(target, read_owner, index), Core.checked_observe(target, output, index),
        Laws.fiber(plan, values, index, cursor), kept, inv(index)(inside)))}

def position_inv(+plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target: Shape.Shape, +cursor: Nat,
  -output: Core.Tensor<target>, read_owner: Core.Tensor<target>, -written: Core.Tensor<target>,
  +target_index: U32, +source_index: U32, +accumulated: F32, +value: F32, -unit: Unit,
  +index: U32, decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(target_index), U32.to_nat(index))>,
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool},
  -meaning: Core.CheckedUpdateMeaning(target, read_owner, target_index, F32.add(accumulated, value), (written, Done{unit})),
  -inv: Inv(plan, values, target, cursor, output),
  -read: {Core.checked_observe(target, output, target_index) == Some{accumulated} : Maybe<&2, F32>},
  -kept: @+other: U32 -> {Core.checked_observe(target, read_owner, other) == Core.checked_observe(target, output, other) : Maybe<&2, F32>},
  -word: {Laws.word_at(values, U32.to_nat(source_index)) == Some{value} : Maybe<&2, F32>},
  -offsets: {Projection.offset_spec(plan, cursor) == Some{(target_index, source_index)} : Maybe<&1, U32 & U32>}) ->
  Holds<{Core.checked_observe(target, written, index) == Laws.fiber(plan, values, index, 1n+cursor) : Maybe<&2, F32>}>:
  match decision:
    case Shape.BTrue{equal}:
      written_true(plan, values, target, cursor, output, written, target_index, source_index, accumulated, value, index, equal, inside, inv,
        read, update_written(target, read_owner, written, target_index, F32.add(accumulated, value), unit, meaning), word, offsets)
    case Shape.BFalse{equal}:
      written_false(plan, values, target, cursor, output, read_owner, written, target_index, source_index, index, equal, inside, inv,
        kept(index),
        frame_equal(target, written, read_owner, index,
          update_frame(target, read_owner, written, target_index, F32.add(accumulated, value), unit, meaning)(index)(
            {\{TransformProof.CheckedObservationModel: checked => checked} :
              TransformProof.CheckedObservationModel<target, read_owner, index> -> Core.ObservationChecked(Core.Packed{target, read_owner}, index)}(
              TransformProof.checked_observation_model(target, read_owner, index, inside)))(
            Equal.sym(Bool, False{}, Nat.is_eq(U32.to_nat(target_index), U32.to_nat(index)), equal))),
        offsets)

def write_inv(+plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target: Shape.Shape, +cursor: Nat,
  -output: Core.Tensor<target>, -read_owner: Core.Tensor<target>, +target_index: U32, +source_index: U32, +accumulated: F32, +value: F32,
  input: Core.Dynamic, pair: Core.Tensor<target> & Result<&2, &2, Core.Error, Unit>,
  -meaning: Core.CheckedUpdateMeaning(target, read_owner, target_index, F32.add(accumulated, value), pair),
  -inv: Inv(plan, values, target, cursor, output),
  -read: {Core.checked_observe(target, output, target_index) == Some{accumulated} : Maybe<&2, F32>},
  -kept: @+other: U32 -> {Core.checked_observe(target, read_owner, other) == Core.checked_observe(target, output, other) : Maybe<&2, F32>},
  -word: {Laws.word_at(values, U32.to_nat(source_index)) == Some{value} : Maybe<&2, F32>},
  -offsets: {Projection.offset_spec(plan, cursor) == Some{(target_index, source_index)} : Maybe<&1, U32 & U32>}) ->
  Holds<StepInv(plan, values, target, cursor, Reduce.step_write(target, input, pair))>:
  (written, result) = pair
  match result:
    case Fail{error}:
      Holds{Unit{}}
    case Done{unit}:
      Holds{index => inside => {\{Holds: proof => proof} :
        Holds<{Core.checked_observe(target, written, index) == Laws.fiber(plan, values, index, 1n+cursor) : Maybe<&2, F32>}> ->
        {Core.checked_observe(target, written, index) == Laws.fiber(plan, values, index, 1n+cursor) : Maybe<&2, F32>}}(
        position_inv(plan, values, target, cursor, output, read_owner, written, target_index, source_index, accumulated, value, unit,
          index, Shape.bool_case(Nat.is_eq(U32.to_nat(target_index), U32.to_nat(index))), inside, meaning, inv, read, kept, word, offsets))}

def output_inv(+plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target: Shape.Shape, +cursor: Nat,
  -output: Core.Tensor<target>, +target_index: U32, +source_index: U32, +value: F32,
  input: Core.Dynamic, pair: Core.Tensor<target> & Result<&2, &2, Core.Error, F32>,
  -source: {Core.read_value(target, pair) == Core.checked_observe(target, output, target_index) : Maybe<&2, F32>},
  -keeps: @+other: U32 -> {Core.retained_read_observe(target, pair, other) == Core.checked_observe(target, output, other) : Maybe<&2, F32>},
  -inv: Inv(plan, values, target, cursor, output),
  -word: {Laws.word_at(values, U32.to_nat(source_index)) == Some{value} : Maybe<&2, F32>},
  -offsets: {Projection.offset_spec(plan, cursor) == Some{(target_index, source_index)} : Maybe<&1, U32 & U32>}) ->
  Holds<StepInv(plan, values, target, cursor, Reduce.step_output(target, target_index, input, value, pair))>:
  (read_owner, result) = pair
  match result:
    case Fail{error}:
      Holds{Unit{}}
    case Done{+accumulated}:
      write_inv(plan, values, target, cursor, output, read_owner, target_index, source_index, accumulated, value, input,
        Core.write_flat(target, read_owner, target_index, F32.add(accumulated, value)),
        {\{Core.CheckedUpdateModel: meaning => meaning} :
          Core.CheckedUpdateModel<target, read_owner, target_index, F32.add(accumulated, value)> ->
          Core.CheckedUpdateMeaning(target, read_owner, target_index, F32.add(accumulated, value),
            Core.write_flat(target, read_owner, target_index, F32.add(accumulated, value)))}(
          CoreProof.checked_update_model(target, read_owner, target_index, F32.add(accumulated, value))),
        inv, Equal.sym(Maybe<&2, F32>, Some{accumulated}, Core.checked_observe(target, output, target_index), source), keeps, word, offsets)

def input_inv(+plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target: Shape.Shape, +cursor: Nat,
  output: Core.Tensor<target>, +target_index: U32, +source_index: U32, -original: Core.Dynamic,
  pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>,
  -source: {Core.dynamic_read_value(pair) == Core.checked_observe_dynamic(original, source_index) : Maybe<&2, F32>},
  -listed: Laws.Listed(original, values),
  -inv: Inv(plan, values, target, cursor, output),
  -offsets: {Projection.offset_spec(plan, cursor) == Some{(target_index, source_index)} : Maybe<&1, U32 & U32>}) ->
  Holds<StepInv(plan, values, target, cursor, Reduce.step_input(target, target_index, output, pair))>:
  (input, result) = pair
  match result:
    case Fail{error}:
      Holds{Unit{}}
    case Done{+value}:
      output_inv(plan, values, target, cursor, output, target_index, source_index, value, input, Core.read_flat(target, output, target_index),
        {\{Core.SourceRead: meaning => meaning} : Core.SourceRead<target, output, target_index> ->
          {Core.read_value(target, Core.read_flat(target, output, target_index)) == Core.checked_observe(target, output, target_index) : Maybe<&2, F32>}}(
          Core.source_read_model(target, output, target_index)),
        {\{Core.SourceReadKeeps: kept => kept} : Core.SourceReadKeeps<target, output, target_index> ->
          @+other: U32 -> {Core.retained_read_observe(target, Core.read_flat(target, output, target_index), other) == Core.checked_observe(target, output, other) : Maybe<&2, F32>}}(
          Core.source_read_keeps(target, output, target_index)),
        inv,
        Equal.trans(Maybe<&2, F32>, Laws.word_at(values, U32.to_nat(source_index)), Core.checked_observe_dynamic(original, source_index), Some{value},
          Equal.sym(Maybe<&2, F32>, Core.checked_observe_dynamic(original, source_index), Laws.word_at(values, U32.to_nat(source_index)), listed(source_index)),
          Equal.sym(Maybe<&2, F32>, Some{value}, Core.checked_observe_dynamic(original, source_index), source)),
        offsets)

def indices_inv(+plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target: Shape.Shape, +cursor: Nat, +index: U32,
  -exact: {U32.to_nat(index) == cursor : Nat}, input: Core.Dynamic, output: Core.Tensor<target>,
  result: Result<&2, &1, Core.Error, U32 & U32>,
  -same: {result == Elementwise.input_indices(plan, index) : Result<&2, &1, Core.Error, U32 & U32>},
  -inv: Inv(plan, values, target, cursor, output), -listed: Laws.Listed(input, values)) ->
  Holds<StepInv(plan, values, target, cursor, Reduce.step_indices(target, Reduce.State{input, output}, result))>:
  match result:
    case Fail{error}:
      Holds{Unit{}}
    case Done{indices}:
      (+target_index, +source_index) = indices
      input_inv(plan, values, target, cursor, output, target_index, source_index, input, Core.read_dynamic(input, source_index),
        {\{Core.DynamicSourceRead: meaning => meaning} : Core.DynamicSourceRead<input, source_index> ->
          {Core.dynamic_read_value(Core.read_dynamic(input, source_index)) == Core.checked_observe_dynamic(input, source_index) : Maybe<&2, F32>}}(
          Core.dynamic_source_read_model(input, source_index)),
        listed, inv,
        Equal.sym(Maybe<&1, U32 & U32>, Some{(target_index, source_index)}, Projection.offset_spec(plan, cursor),
          %exact : {Some{(target_index, source_index)} == Projection.offset_spec(plan, _) : Maybe<&1, U32 & U32>}
          Equal.trans(Maybe<&1, U32 & U32>, Some{(target_index, source_index)}, Projection.indices_result(Elementwise.input_indices(plan, index)),
            Projection.offset_spec(plan, U32.to_nat(index)),
            Equal.cong(Result<&2, &1, Core.Error, U32 & U32>, Maybe<&1, U32 & U32>, Projection.indices_result,
              Done{(target_index, source_index)}, Elementwise.input_indices(plan, index), same),
            Projection.input_indices_refines(plan, index))))

def step_inv(+plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target: Shape.Shape, +cursor: Nat, +index: U32,
  -exact: {U32.to_nat(index) == cursor : Nat}, state: Reduce.State<target>,
  -inv: Inv(plan, values, target, cursor, state_output(target, state)), -listed: Laws.Listed(state_input(target, state), values)) ->
  Holds<StepInv(plan, values, target, cursor, Reduce.step(plan, target, index, state))>:
  Reduce.State{input, output} = state
  indices_inv(plan, values, target, cursor, index, exact, input, output, Elementwise.input_indices(plan, index), {==}, inv, listed)

def conversion_inv(+plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target: Shape.Shape, +limit: Nat, state: Reduce.State<target>) ->
  Holds<LoopInv(plan, values, target, limit, Reduce.conversion_error(target, state))>:
  Reduce.State{input, output} = state
  Holds{Unit{}}

def continue_inv(+remaining: Nat, +plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target: Shape.Shape, +cursor: Nat,
  result: Reduce.Step<target>, -step: StepInv(plan, values, target, cursor, result),
  -listed: Laws.Listed(step_owner(target, result), values),
  -rec: @state: Reduce.State<target> -> @-inv: Inv(plan, values, target, 1n+cursor, state_output(target, state)) ->
    @-listed: Laws.Listed(state_input(target, state), values) ->
    Holds<LoopInv(plan, values, target, Nat.add(remaining, 1n+cursor), Reduce.loop(remaining, plan, target, 1n+cursor, state))>) ->
  Holds<LoopInv(plan, values, target, Nat.add(remaining, 1n+cursor),
    Reduce.continue_step(target, state => Reduce.loop(remaining, plan, target, 1n+cursor, state), result))>:
  match result:
    case Reduce.Stop{input, error}:
      Holds{Unit{}}
    case Reduce.Continue{state}:
      Holds{{\{Holds: proof => proof} :
        Holds<LoopInv(plan, values, target, Nat.add(remaining, 1n+cursor), Reduce.loop(remaining, plan, target, 1n+cursor, state))> ->
        LoopInv(plan, values, target, Nat.add(remaining, 1n+cursor), Reduce.loop(remaining, plan, target, 1n+cursor, state))}(rec(state, step, listed))}

def listed_kept(-original: Core.Dynamic, -retained: Core.Dynamic, +values: List<&2, F32>,
  -same: Totality.SameSource(original, retained), -listed: Laws.Listed(original, values)) -> Holds<Laws.Listed(retained, values)>:
  Holds{source => Equal.trans(Maybe<&2, F32>, Core.checked_observe_dynamic(retained, source), Core.checked_observe_dynamic(original, source),
    Laws.word_at(values, U32.to_nat(source)),
    {\{Holds: proof => proof} : Holds<{Core.checked_observe_dynamic(retained, source) == Core.checked_observe_dynamic(original, source) : Maybe<&2, F32>}> ->
      {Core.checked_observe_dynamic(retained, source) == Core.checked_observe_dynamic(original, source) : Maybe<&2, F32>}}(
      same_checked(original, retained, same, source)),
    listed(source))}

def decided_inv(+remaining: Nat, +plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target: Shape.Shape, +cursor: Nat, +index: U32,
  state: Reduce.State<target>, decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>,
  -inv: Inv(plan, values, target, cursor, state_output(target, state)), -listed: Laws.Listed(state_input(target, state), values),
  -rec: @state: Reduce.State<target> -> @-inv: Inv(plan, values, target, 1n+cursor, state_output(target, state)) ->
    @-listed: Laws.Listed(state_input(target, state), values) ->
    Holds<LoopInv(plan, values, target, Nat.add(remaining, 1n+cursor), Reduce.loop(remaining, plan, target, 1n+cursor, state))>) ->
  Holds<LoopInv(plan, values, target, Nat.add(remaining, 1n+cursor), Reduce.decided(plan, target, cursor, index, state,
    state => Reduce.loop(remaining, plan, target, 1n+cursor, state), decision))>:
  match decision:
    case Shape.BFalse{equal}:
      conversion_inv(plan, values, target, Nat.add(remaining, 1n+cursor), state)
    case Shape.BTrue{equal}:
      Holds{{\{Holds: proof => proof} :
        Holds<LoopInv(plan, values, target, Nat.add(remaining, 1n+cursor),
          Reduce.continue_step(target, state => Reduce.loop(remaining, plan, target, 1n+cursor, state), Reduce.step(plan, target, index, state)))> ->
        LoopInv(plan, values, target, Nat.add(remaining, 1n+cursor),
          Reduce.continue_step(target, state => Reduce.loop(remaining, plan, target, 1n+cursor, state), Reduce.step(plan, target, index, state)))}(
        continue_inv(remaining, plan, values, target, cursor, Reduce.step(plan, target, index, state),
          {\{Holds: proof => proof} : Holds<StepInv(plan, values, target, cursor, Reduce.step(plan, target, index, state))> ->
            StepInv(plan, values, target, cursor, Reduce.step(plan, target, index, state))}(
            step_inv(plan, values, target, cursor, index, Lifecycle.natural_equal(U32.to_nat(index), cursor, equal), state, inv, listed)),
          {\{Holds: proof => proof} : Holds<Laws.Listed(step_owner(target, Reduce.step(plan, target, index, state)), values)> ->
            Laws.Listed(step_owner(target, Reduce.step(plan, target, index, state)), values)}(
            listed_kept(state_input(target, state), step_owner(target, Reduce.step(plan, target, index, state)), values,
              {\{Holds: proof => proof} : Holds<Totality.SameSource(state_input(target, state), step_owner(target, Reduce.step(plan, target, index, state)))> ->
                Totality.SameSource(state_input(target, state), step_owner(target, Reduce.step(plan, target, index, state)))}(
                step_retains(plan, target, index, state)), listed)),
          rec))}

def loop_inv(fuel: Nat, +plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target: Shape.Shape, +cursor: Nat, state: Reduce.State<target>,
  -inv: Inv(plan, values, target, cursor, state_output(target, state)), -listed: Laws.Listed(state_input(target, state), values)) ->
  Holds<LoopInv(plan, values, target, Nat.add(fuel, cursor), Reduce.loop(fuel, plan, target, cursor, state))>:
  match fuel:
    case 0n:
      Reduce.State{input, output} = state
      Holds{inv}
    case 1n+remaining:
      Holds{
        %Arith.add_succ(remaining, cursor) : LoopInv(plan, values, target, _, Reduce.loop(1n+remaining, plan, target, cursor, state))
        {\{Holds: proof => proof} :
          Holds<LoopInv(plan, values, target, Nat.add(remaining, 1n+cursor), Reduce.loop(1n+remaining, plan, target, cursor, state))> ->
          LoopInv(plan, values, target, Nat.add(remaining, 1n+cursor), Reduce.loop(1n+remaining, plan, target, cursor, state))}(
          decided_inv(remaining, plan, values, target, cursor, U32.from_nat(cursor), state,
            Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)), inv, listed,
            state => inv => listed => loop_inv(remaining, plan, values, target, 1n+cursor, state, inv, listed)))}

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

def finish_mapping(-input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan, -limit: Nat,
  -count: {limit == Shape.logical_numel(Totality.dimensions(input)) : Nat},
  pair: Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>>,
  -loop: @+values: List<&2, F32> -> @-listed: Laws.Listed(input, values) -> Holds<LoopInv(plan, values, target, limit, pair)>) ->
  Holds<Laws.MappingWith(input, target, Done{plan}, Reduce.finish(target, pair))>:
  (owner, result) = pair
  match result:
    case Fail{error}:
      Holds{Unit{}}
    case Done{output}:
      Holds{({==}, values => listed => index => inside =>
        %count : {Core.checked_observe(target, output, index) == Laws.fiber(plan, values, index, _) : Maybe<&2, F32>}
        {\{Holds: proof => proof} : Holds<LoopInv(plan, values, target, limit, (owner, Done{output}))> ->
          Inv(plan, values, target, limit, output)}(loop(values, listed))(index)(inside))}

def allocated_mapping(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  result: Result<&2, &1, Core.Error, Core.Tensor<target>>,
  -same: {result == Core.new(target, 0.0) : Result<&2, &1, Core.Error, Core.Tensor<target>>},
  -count: {Shape.elements(Shape.output_shape(plan)) == Shape.logical_numel(Totality.dimensions(input)) : Nat}) ->
  Holds<Laws.MappingWith(input, target, Done{plan}, Reduce.allocated(input, target, plan, result))>:
  match result:
    case Fail{error}:
      Holds{Unit{}}
    case Done{output}:
      finish_mapping(input, target, plan, Nat.add(Shape.elements(Shape.output_shape(plan)), 0n),
        Equal.trans(Nat, Nat.add(Shape.elements(Shape.output_shape(plan)), 0n), Shape.elements(Shape.output_shape(plan)),
          Shape.logical_numel(Totality.dimensions(input)), TransformProof.add_zero_right(Shape.elements(Shape.output_shape(plan))), count),
        Reduce.loop(Shape.elements(Shape.output_shape(plan)), plan, target, 0n, Reduce.State{input, output}),
        values => listed => loop_inv(Shape.elements(Shape.output_shape(plan)), plan, values, target, 0n, Reduce.State{input, output},
          {\{Holds: proof => proof} : Holds<Inv(plan, values, target, 0n, output)> -> Inv(plan, values, target, 0n, output)}(
            zero_inv(plan, values, target, output, same)), listed))

def output_count(+shape: Shape.Shape, +plan: Shape.BroadcastPlan,
  -equal: {True{} == Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape)) : Bool}) ->
  Holds<{Shape.elements(Shape.output_shape(plan)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}>:
  Holds{Equal.trans(Nat, Shape.elements(Shape.output_shape(plan)), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))),
    Shape.logical_numel(Shape.dimensions(shape)), Shape.elements_exact(Shape.output_shape(plan)),
    Equal.cong(List<&2, U32>, Nat, Shape.logical_numel, Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape),
      TransformProof.words_equal(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape), equal)))}

def fitted_mapping(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape, +plan: Shape.BroadcastPlan, same: Bool,
  -decided: {same == Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape)) : Bool}) ->
  Holds<Laws.MappingWith(Core.Packed{shape, tensor}, target, Done{plan}, Reduce.fitted(Core.Packed{shape, tensor}, target, plan, same))>:
  match same:
    case False{}:
      Holds{Unit{}}
    case True{}:
      allocated_mapping(Core.Packed{shape, tensor}, target, plan, Core.new(target, 0.0), {==},
        {\{Holds: proof => proof} : Holds<{Shape.elements(Shape.output_shape(plan)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}> ->
          {Shape.elements(Shape.output_shape(plan)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}}(output_count(shape, plan, decided)))

def planned_mapping(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) ->
  Holds<Laws.MappingWith(Core.Packed{shape, tensor}, target, result, Reduce.planned(shape, tensor, target, result))>:
  match result:
    case Fail{error}:
      Holds{Unit{}}
    case Done{+plan}:
      fitted_mapping(shape, tensor, target, plan, Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape)), {==})

def Laws.reduce_mapping(input, target):
  Core.Packed{+shape, tensor} = input
  Laws.MappingModel{
    {\{Holds: proof => proof} :
      Holds<Laws.MappingWith(Core.Packed{shape, tensor}, target, Shape.broadcast_checked(target, shape),
        Reduce.planned(shape, tensor, target, Shape.broadcast_checked(target, shape)))> ->
      Laws.Mapping(Core.Packed{shape, tensor}, target, Reduce.reduce_to(Core.Packed{shape, tensor}, target))}(
      planned_mapping(shape, tensor, target, Shape.broadcast_checked(target, shape)))}

def output_in_bounds(+plan: Shape.BroadcastPlan, +target: Shape.Shape, +index: U32,
  -left: {Shape.dimensions(Shape.left_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  -bound: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan)))) == True{} : Bool}) ->
  Holds<{Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}>:
  Holds{
    %left :
      {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(_)) == True{} : Bool}
    bound}

def write_success(-target: Shape.Shape, input: Core.Dynamic, pair: Core.Tensor<target> & Result<&2, &2, Core.Error, Unit>,
  -successful: {TransformTotality.write_succeeded(target, pair) == True{} : Bool}) ->
  Holds<{step_continued(target, Reduce.step_write(target, input, pair)) == True{} : Bool}>:
  (output, result) = pair
  match result:
    case Fail{error}:
      Holds{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{unit}:
      Holds{{==}}

def target_success(+target: Shape.Shape, +target_index: U32, input: Core.Dynamic, value: F32,
  pair: Core.Tensor<target> & Result<&2, &2, Core.Error, F32>,
  -successful: {TransformTotality.read_succeeded(target, pair) == True{} : Bool},
  -inside: {Nat.is_lt(U32.to_nat(target_index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}) ->
  Holds<{step_continued(target, Reduce.step_output(target, target_index, input, value, pair)) == True{} : Bool}>:
  (output, result) = pair
  match result:
    case Fail{error}:
      Holds{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{+accumulated}:
      write_success(target, input, Core.write_flat(target, output, target_index, F32.add(accumulated, value)),
        {\{TransformTotality.WriteSuccess: successful => successful} :
          TransformTotality.WriteSuccess<target, output, target_index, F32.add(accumulated, value)> ->
          {TransformTotality.write_succeeded(target, Core.write_flat(target, output, target_index, F32.add(accumulated, value))) == True{} : Bool}}(
          TransformProof.tensor_write_success(target, output, target_index, F32.add(accumulated, value), inside)))

def source_success(+target: Shape.Shape, +target_index: U32, output: Core.Tensor<target>,
  pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>,
  -successful: {ElementwiseProof.dynamic_read_succeeded(pair) == True{} : Bool},
  -inside: {Nat.is_lt(U32.to_nat(target_index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}) ->
  Holds<{step_continued(target, Reduce.step_input(target, target_index, output, pair)) == True{} : Bool}>:
  (input, result) = pair
  match result:
    case Fail{error}:
      Holds{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{value}:
      target_success(target, target_index, input, value, Core.read_flat(target, output, target_index),
        {\{TransformTotality.ReadSuccess: successful => successful} : TransformTotality.ReadSuccess<target, output, target_index> ->
          {TransformTotality.read_succeeded(target, Core.read_flat(target, output, target_index)) == True{} : Bool}}(
          TransformProof.read_success_status(target, output, target_index, TransformProof.read_success(target, output, target_index, inside))),
        inside)

def step_success(+plan: Shape.BroadcastPlan, +target: Shape.Shape, +index: U32, state: Reduce.State<target>,
  -left: {Shape.dimensions(Shape.left_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  -right: {Totality.dimensions(state_input(target, state)) == Shape.dimensions(Shape.right_shape(plan)) : List<&2, U32>},
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.elements(Shape.output_shape(plan))) == True{} : Bool}) ->
  Holds<{step_continued(target, Reduce.step(plan, target, index, state)) == True{} : Bool}>:
  Reduce.State{input, output} = state
  Holds{{\{ElementwiseProof.IndicesDone: li => ri => exact => lb => rb =>
    %exact : {step_continued(target, Reduce.step_indices(target, Reduce.State{input, output}, _)) == True{} : Bool}
    {\{Holds: proof => proof} : Holds<{step_continued(target, Reduce.step_input(target, li, output, Core.read_dynamic(input, ri))) == True{} : Bool}> ->
      {step_continued(target, Reduce.step_input(target, li, output, Core.read_dynamic(input, ri))) == True{} : Bool}}(
      source_success(target, li, output, Core.read_dynamic(input, ri),
        {\{ElementwiseProof.DynamicReadSuccess: successful => successful} : ElementwiseProof.DynamicReadSuccess<input, ri> ->
          {ElementwiseProof.dynamic_read_succeeded(Core.read_dynamic(input, ri)) == True{} : Bool}}(
          ElementwiseProof.dynamic_read_success(input, ri,
            %Equal.sym(List<&2, U32>, Totality.dimensions(input), Shape.dimensions(Shape.right_shape(plan)), right) :
              {Nat.is_lt(U32.to_nat(ri), Shape.logical_numel(_)) == True{} : Bool}
            rb)),
        {\{Holds: proof => proof} : Holds<{Nat.is_lt(U32.to_nat(li), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}> ->
          {Nat.is_lt(U32.to_nat(li), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}}(output_in_bounds(plan, target, li, left, lb))))} :
    ElementwiseProof.IndicesDone<plan, Elementwise.input_indices(plan, index)> ->
    {step_continued(target, Reduce.step(plan, target, index, Reduce.State{input, output})) == True{} : Bool}}(
    ElementwiseProof.indices_success(plan, index, inside))}

def continue_success(+remaining: Nat, +plan: Shape.BroadcastPlan, +target: Shape.Shape, +cursor: Nat, result: Reduce.Step<target>,
  -continued: {step_continued(target, result) == True{} : Bool},
  -right: {Totality.dimensions(step_owner(target, result)) == Shape.dimensions(Shape.right_shape(plan)) : List<&2, U32>},
  -rec: @state: Reduce.State<target> -> @-right: {Totality.dimensions(state_input(target, state)) == Shape.dimensions(Shape.right_shape(plan)) : List<&2, U32>} ->
    Holds<{loop_succeeded(target, Reduce.loop(remaining, plan, target, 1n+cursor, state)) == True{} : Bool}>) ->
  Holds<{loop_succeeded(target, Reduce.continue_step(target, state => Reduce.loop(remaining, plan, target, 1n+cursor, state), result)) == True{} : Bool}>:
  match result:
    case Reduce.Stop{input, error}:
      Holds{Empty.absurd({False{} == True{} : Bool}, Arith.clash(continued))}
    case Reduce.Continue{state}:
      Holds{{\{Holds: proof => proof} : Holds<{loop_succeeded(target, Reduce.loop(remaining, plan, target, 1n+cursor, state)) == True{} : Bool}> ->
        {loop_succeeded(target, Reduce.loop(remaining, plan, target, 1n+cursor, state)) == True{} : Bool}}(rec(state, right))}

def decided_success(+remaining: Nat, +plan: Shape.BroadcastPlan, +target: Shape.Shape, +cursor: Nat, +index: U32,
  state: Reduce.State<target>, decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>,
  -exact: {U32.to_nat(index) == cursor : Nat},
  -left: {Shape.dimensions(Shape.left_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  -right: {Totality.dimensions(state_input(target, state)) == Shape.dimensions(Shape.right_shape(plan)) : List<&2, U32>},
  -inside: {Nat.is_lt(cursor, Shape.elements(Shape.output_shape(plan))) == True{} : Bool},
  -rec: @state: Reduce.State<target> -> @-right: {Totality.dimensions(state_input(target, state)) == Shape.dimensions(Shape.right_shape(plan)) : List<&2, U32>} ->
    Holds<{loop_succeeded(target, Reduce.loop(remaining, plan, target, 1n+cursor, state)) == True{} : Bool}>) ->
  Holds<{loop_succeeded(target, Reduce.decided(plan, target, cursor, index, state,
    state => Reduce.loop(remaining, plan, target, 1n+cursor, state), decision)) == True{} : Bool}>:
  match decision:
    case Shape.BFalse{equal}:
      Holds{Empty.absurd({loop_succeeded(target, Reduce.conversion_error(target, state)) == True{} : Bool},
        Nats.contradiction(Nat.is_eq(U32.to_nat(index), cursor), PlanProof.eq_true(U32.to_nat(index), cursor, exact), equal))}
    case Shape.BTrue{equal}:
      continue_success(remaining, plan, target, cursor, Reduce.step(plan, target, index, state),
        {\{Holds: proof => proof} : Holds<{step_continued(target, Reduce.step(plan, target, index, state)) == True{} : Bool}> ->
          {step_continued(target, Reduce.step(plan, target, index, state)) == True{} : Bool}}(
          step_success(plan, target, index, state, left, right,
            %Equal.sym(Nat, U32.to_nat(index), cursor, exact) : {Nat.is_lt(_, Shape.elements(Shape.output_shape(plan))) == True{} : Bool}
            inside)),
        Equal.trans(List<&2, U32>, Totality.dimensions(step_owner(target, Reduce.step(plan, target, index, state))),
          Totality.dimensions(state_input(target, state)), Shape.dimensions(Shape.right_shape(plan)),
          {\{Holds: proof => proof} : Holds<{Totality.dimensions(step_owner(target, Reduce.step(plan, target, index, state))) == Totality.dimensions(state_input(target, state)) : List<&2, U32>}> ->
            {Totality.dimensions(step_owner(target, Reduce.step(plan, target, index, state))) == Totality.dimensions(state_input(target, state)) : List<&2, U32>}}(
            same_dimensions(state_input(target, state), step_owner(target, Reduce.step(plan, target, index, state)),
              {\{Holds: proof => proof} : Holds<Totality.SameSource(state_input(target, state), step_owner(target, Reduce.step(plan, target, index, state)))> ->
                Totality.SameSource(state_input(target, state), step_owner(target, Reduce.step(plan, target, index, state)))}(
                step_retains(plan, target, index, state)))),
          right),
        rec)

def loop_success(fuel: Nat, +plan: Shape.BroadcastPlan, +target: Shape.Shape, +cursor: Nat, state: Reduce.State<target>,
  -left: {Shape.dimensions(Shape.left_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  -right: {Totality.dimensions(state_input(target, state)) == Shape.dimensions(Shape.right_shape(plan)) : List<&2, U32>},
  -within: {Nat.is_le(Nat.add(fuel, cursor), Shape.elements(Shape.output_shape(plan))) == True{} : Bool}) ->
  Holds<{loop_succeeded(target, Reduce.loop(fuel, plan, target, cursor, state)) == True{} : Bool}>:
  match fuel:
    case 0n:
      Reduce.State{input, output} = state
      Holds{{==}}
    case 1n+remaining:
      decided_success(remaining, plan, target, cursor, U32.from_nat(cursor), state,
        Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)),
        TransformProof.cursor_word_exact(Shape.output_shape(plan), cursor,
          Conversion.lt_le(cursor, Shape.elements(Shape.output_shape(plan)),
            Arith.lt_le_trans(cursor, 1n+Nat.add(remaining, cursor), Shape.elements(Shape.output_shape(plan)),
              Arith.lt_succ(cursor, Nat.add(remaining, cursor), TransformProof.add_cursor_le(remaining, cursor)), within))),
        left, right,
        Arith.lt_le_trans(cursor, 1n+Nat.add(remaining, cursor), Shape.elements(Shape.output_shape(plan)),
          Arith.lt_succ(cursor, Nat.add(remaining, cursor), TransformProof.add_cursor_le(remaining, cursor)), within),
        state => right => loop_success(remaining, plan, target, 1n+cursor, state, left, right,
          %Equal.sym(Nat, Nat.add(remaining, 1n+cursor), 1n+Nat.add(remaining, cursor), Arith.add_succ(remaining, cursor)) :
            {Nat.is_le(_, Shape.elements(Shape.output_shape(plan))) == True{} : Bool}
          within))

def finish_success(+target: Shape.Shape, pair: Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>>,
  -successful: {loop_succeeded(target, pair) == True{} : Bool}) -> Laws.Success<Reduce.finish(target, pair)>:
  (input, result) = pair
  match result:
    case Fail{error}:
      Laws.Success{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{output}:
      Laws.Success{{==}}

def allocated_success(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  result: Result<&2, &1, Core.Error, Core.Tensor<target>>, -done: Allocation.TensorDone<target, result>,
  -left: {Shape.dimensions(Shape.left_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  -right: {Totality.dimensions(input) == Shape.dimensions(Shape.right_shape(plan)) : List<&2, U32>}) ->
  Laws.Success<Reduce.allocated(input, target, plan, result)>:
  match result:
    case Fail{error}:
      Laws.Success{Empty.absurd({False{} == True{} : Bool},
        {\{Allocation.TensorDone: tensor => same => Arith.clash(Equal.cong(Result<&2, &1, Core.Error, Core.Tensor<target>>, Bool,
          found => Result.is_done(&2, &1, Core.Error, Core.Tensor<target>, found), Fail{error}, Done{tensor},
          Equal.sym(Result<&2, &1, Core.Error, Core.Tensor<target>>, Done{tensor}, Fail{error}, same)))} :
          Allocation.TensorDone<target, Fail{error}> -> Empty}(done))}
    case Done{output}:
      finish_success(target, Reduce.loop(Shape.elements(Shape.output_shape(plan)), plan, target, 0n, Reduce.State{input, output}),
        {\{Holds: proof => proof} :
          Holds<{loop_succeeded(target, Reduce.loop(Shape.elements(Shape.output_shape(plan)), plan, target, 0n, Reduce.State{input, output})) == True{} : Bool}> ->
          {loop_succeeded(target, Reduce.loop(Shape.elements(Shape.output_shape(plan)), plan, target, 0n, Reduce.State{input, output})) == True{} : Bool}}(
          loop_success(Shape.elements(Shape.output_shape(plan)), plan, target, 0n, Reduce.State{input, output}, left, right,
            %Equal.sym(Nat, Nat.add(Shape.elements(Shape.output_shape(plan)), 0n), Shape.elements(Shape.output_shape(plan)),
              TransformProof.add_zero_right(Shape.elements(Shape.output_shape(plan)))) :
              {Nat.is_le(_, Shape.elements(Shape.output_shape(plan))) == True{} : Bool}
            Conversion.le_reflexive(Shape.elements(Shape.output_shape(plan))))))

def fitted_success(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan, same: Bool,
  -decided: {same == Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Totality.dimensions(input)) : Bool},
  -truth: {Shape.dimensions(Shape.output_shape(plan)) == Totality.dimensions(input) : List<&2, U32>},
  -left: {Shape.dimensions(Shape.left_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  -right: {Totality.dimensions(input) == Shape.dimensions(Shape.right_shape(plan)) : List<&2, U32>}) ->
  Laws.Success<Reduce.fitted(input, target, plan, same)>:
  match same:
    case False{}:
      Laws.Success{Empty.absurd({False{} == True{} : Bool},
        Arith.clash(Equal.trans(Bool, False{}, Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Totality.dimensions(input)), True{}, decided,
          %truth : {Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), _) == True{} : Bool}
          TransformProof.words_same_refl(Shape.dimensions(Shape.output_shape(plan))))))}
    case True{}:
      allocated_success(input, target, plan, Core.new(target, 0.0), AllocationProof.core_new_done(target, 0.0), left, right)

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
      fitted_success(Core.Packed{shape, tensor}, target, plan,
        Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape)), {==},
        ElementwiseProof.planned_output_dimensions(target, shape, plan, Shape.dimensions(shape), alignment, sources),
        Equal.cong(List<&2, U32> & List<&2, U32>, List<&2, U32>, Pair.fst(List<&2, U32>, List<&2, U32>),
          (Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(Shape.right_shape(plan))), (Shape.dimensions(target), Shape.dimensions(shape)), sources),
        Equal.sym(List<&2, U32>, Shape.dimensions(Shape.right_shape(plan)), Shape.dimensions(shape),
          Equal.cong(List<&2, U32> & List<&2, U32>, List<&2, U32>, Pair.snd(List<&2, U32>, List<&2, U32>),
            (Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(Shape.right_shape(plan))), (Shape.dimensions(target), Shape.dimensions(shape)), sources)))

def rank_of(+shape: Shape.Shape) -> Shape.LE(Shape.rank_dimensions(Shape.dimensions(shape)), 4n):
  match shape:
    case Shape.SShape{dims, size, exact, rank_bound, capacity_bound}:
      rank_bound

def capacity_of(+shape: Shape.Shape) -> Shape.LE(Shape.logical_numel(Shape.dimensions(shape)), 1048576n):
  match shape:
    case Shape.SShape{dims, size, exact, rank_bound, capacity_bound}:
      capacity_bound

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
          planned_mapping(shape, tensor, target, Done{plan})),
        {\{Holds: proof => proof} : Holds<Laws.Listed(Core.Packed{shape, tensor}, Nil{})> -> Laws.Listed(Core.Packed{shape, tensor}, Nil{})}(
          listed_empty(Core.Packed{shape, tensor}, empty)),
        empty)

def Laws.reduce_empty_zeros(input, target, empty):
  Core.Packed{+shape, tensor} = input
  planned_zeros(shape, tensor, target, Shape.broadcast_checked(target, shape), empty)

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
