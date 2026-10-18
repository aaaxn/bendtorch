import Base

import ../shape/core.bend as Shape

import ../shape/plan_proofs.bend as PlanProof

import ../shape/nat_proofs.bend as Nats

import ../storage/arith.bend as Arith

import ../tensor/core.bend as Core

import ../tensor/layout.bend as Layout

import ../tensor/lifecycle.bend as Lifecycle

import ../tensor/elementwise_totality.bend as Totality

import ../tensor/elementwise_PROOF.bend as ElementwiseProof

import ../tensor/core_PROOF.bend as CoreProof

import ../tensor/transform_PROOF.bend as TransformProof

import ../tensor/transform_totality.bend as TransformTotality

import ../tensor/allocation_spec.bend as Allocation

import ../tensor/allocation_PROOF.bend as AllocationProof

import ../tensor/construction_contract_PROOF.bend as ConstructionProof

import ../tensor/word_conversion.bend as Conversion

import ../tensor/observation.bend as Observation

import ./reduce_PROOF.bend as ReduceProof

import ./gradient.bend as Gradient

import ./gradient_LAWS.bend as Laws

type Holds<-P: Type> is Data:
  Holds{-proof: P}

def lt_self(+number: Nat) -> {Nat.is_lt(number, number) == False{} : Bool}:
  match number:
    case 0n:
      {==}
    case 1n+rest:
      lt_self(rest)

def lt_succ_self(+number: Nat) -> {Nat.is_lt(number, 1n+number) == True{} : Bool}:
  match number:
    case 0n:
      {==}
    case 1n+rest:
      lt_succ_self(rest)

def below_zero(+number: Nat) -> {Nat.is_lt(number, 0n) == False{} : Bool}:
  match number:
    case 0n:
      {==}
    case 1n+rest:
      {==}

def below_one_more(+rest: Nat) -> {Nat.is_lt(1n+rest, 0n) == Nat.is_lt(1n+rest, 1n) : Bool}:
  match rest:
    case 0n:
      {==}
    case 1n+deeper:
      {==}

def lt_step(+position: Nat, +cursor: Nat, different: {False{} == Nat.is_eq(cursor, position) : Bool}) ->
  {Nat.is_lt(position, cursor) == Nat.is_lt(position, 1n+cursor) : Bool}:
  match position cursor:
    case 0n 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == Nat.is_lt(0n, 1n) : Bool}, Arith.clash(different))
    case 0n 1n+bound:
      {==}
    case 1n+rest 0n:
      below_one_more(rest)
    case 1n+rest 1n+bound:
      lt_step(rest, bound, different)

def words_dimensions(+left: Shape.Shape, +right: Shape.Shape,
  -decided: {True{} == Layout.words_same(Shape.dimensions(left), Shape.dimensions(right)) : Bool}) ->
  Holds<{Shape.dimensions(left) == Shape.dimensions(right) : List<&2, U32>}>:
  Holds{TransformProof.words_equal(Shape.dimensions(left), Shape.dimensions(right), decided)}

def same_checked(-original: Core.Dynamic, -retained: Core.Dynamic, -same: Totality.SameSource(original, retained), +index: U32) ->
  Holds<{Core.checked_observe_dynamic(retained, index) == Core.checked_observe_dynamic(original, index) : Maybe<&2, F32>}>:
  Holds{Equal.cong(Totality.Inspection, Maybe<&2, F32>, ReduceProof.inspection_value,
    Totality.inspect(Totality.Checked{index}, retained), Totality.inspect(Totality.Checked{index}, original), same(Totality.Checked{index}))}

def same_dimensions(-original: Core.Dynamic, -retained: Core.Dynamic, -same: Totality.SameSource(original, retained)) ->
  Holds<{Totality.dimensions(retained) == Totality.dimensions(original) : List<&2, U32>}>:
  Holds{Equal.cong(Totality.Inspection, List<&2, U32>, ElementwiseProof.inspection_dimensions,
    Totality.inspect(Totality.Dimensions{}, retained), Totality.inspect(Totality.Dimensions{}, original), same(Totality.Dimensions{}))}

def same_trans(-a: Core.Dynamic, -b: Core.Dynamic, -c: Core.Dynamic,
  -ab: Totality.SameSource(a, b), -bc: Totality.SameSource(b, c)) -> Holds<Totality.SameSource(a, c)>:
  Holds{property => Equal.trans(Totality.Inspection, Totality.inspect(property, c), Totality.inspect(property, b), Totality.inspect(property, a),
    bc(property), ab(property))}

def word_exact(+left: U32, +right: U32, equal: {True{} == Nat.is_eq(U32.to_nat(left), U32.to_nat(right)) : Bool}) -> Holds<{left == right : U32}>:
  Holds{
    %Conversion.word_natural_exact(left) : {_ == right : U32}
    %Conversion.word_natural_exact(right) : {U32.from_nat(U32.to_nat(left)) == _ : U32}
    Equal.cong(Nat, U32, U32.from_nat, U32.to_nat(left), U32.to_nat(right),
      Lifecycle.natural_equal(U32.to_nat(left), U32.to_nat(right), equal))}

def unset_zeros(+shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>,
  -same: {result == Core.new(shape, 0.0) : Result<&2, &1, Core.Error, Core.Tensor<shape>>},
  -done: Allocation.TensorDone<shape, result>) ->
  Laws.ZerosModel<shape, Core.construct_values(shape, result)>:
  match result:
    case Fail{error}:
      Laws.ZerosModel{
        {\{Allocation.TensorDone: tensor => equal => Arith.clash(Equal.cong(Result<&2, &1, Core.Error, Core.Tensor<shape>>, Bool,
          found => Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, found), Fail{error}, Done{tensor},
          Equal.sym(Result<&2, &1, Core.Error, Core.Tensor<shape>>, Done{tensor}, Fail{error}, equal)))} :
          Allocation.TensorDone<shape, Fail{error}> -> Empty}(done)}
    case Done{tensor}:
      Laws.ZerosModel{({==}, index => inside => Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, tensor, index),
        Core.read_value(shape, Core.read_flat(shape, tensor, index)), Some{0.0},
        Equal.sym(Maybe<&2, F32>, Core.read_value(shape, Core.read_flat(shape, tensor, index)), Core.checked_observe(shape, tensor, index),
          {\{Core.SourceRead: meaning => meaning} : Core.SourceRead<shape, tensor, index> ->
            {Core.read_value(shape, Core.read_flat(shape, tensor, index)) == Core.checked_observe(shape, tensor, index) : Maybe<&2, F32>}}(
            Core.source_read_model(shape, tensor, index))),
        %Equal.sym(Result<&2, &1, Core.Error, Core.Tensor<shape>>, Done{tensor}, Core.new(shape, 0.0), same) :
          {Observation.point(shape, index, _) == Some{0.0} : Maybe<&2, F32>}
        ConstructionProof.point_value(shape, 0.0, index, inside)))}

def Laws.gradient_unset_zeros(shape):
  unset_zeros(shape, Core.new(shape, 0.0), {==}, AllocationProof.core_new_done(shape, 0.0))

def Laws.gradient_held(shape, held):
  {==}

def state_contribution(-shape: Shape.Shape, state: Gradient.State<shape>) -> Core.Dynamic:
  Gradient.State{contribution, total} = state
  contribution

def state_total(-shape: Shape.Shape, state: Gradient.State<shape>) -> Core.Tensor<shape>:
  Gradient.State{contribution, total} = state
  total

def step_owner(-shape: Shape.Shape, result: Gradient.Step<shape>) -> Core.Dynamic:
  match result:
    case Gradient.Stop{contribution, total, error}:
      contribution
    case Gradient.Continue{state}:
      state_contribution(shape, state)

def step_continued(-shape: Shape.Shape, result: Gradient.Step<shape>) -> Bool:
  match result:
    case Gradient.Stop{contribution, total, error}:
      False{}
    case Gradient.Continue{state}:
      True{}

def write_owner(-shape: Shape.Shape, -contribution: Core.Dynamic, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) ->
  Holds<{step_owner(shape, Gradient.step_write(shape, contribution, pair)) == contribution : Core.Dynamic}>:
  (total, result) = pair
  match result:
    case Fail{error}:
      Holds{{==}}
    case Done{unit}:
      Holds{{==}}

def total_owner(-shape: Shape.Shape, +index: U32, -contribution: Core.Dynamic, +value: F32,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) ->
  Holds<{step_owner(shape, Gradient.step_total(shape, index, contribution, value, pair)) == contribution : Core.Dynamic}>:
  (total, result) = pair
  match result:
    case Fail{error}:
      Holds{{==}}
    case Done{+accumulated}:
      write_owner(shape, contribution, Core.write_flat(shape, total, index, F32.add(accumulated, value)))

def contribution_retains(-shape: Shape.Shape, +index: U32, total: Core.Tensor<shape>, -original: Core.Dynamic,
  pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>,
  -same: Totality.SameSource(original, ElementwiseProof.dynamic_read_owner(pair))) ->
  Holds<Totality.SameSource(original, step_owner(shape, Gradient.step_contribution(shape, index, total, pair)))>:
  (contribution, result) = pair
  match result:
    case Fail{error}:
      Holds{same}
    case Done{value}:
      Holds{property => Equal.trans(Totality.Inspection,
        Totality.inspect(property, step_owner(shape, Gradient.step_total(shape, index, contribution, value, Core.read_flat(shape, total, index)))),
        Totality.inspect(property, contribution), Totality.inspect(property, original),
        Equal.cong(Core.Dynamic, Totality.Inspection, owner => Totality.inspect(property, owner),
          step_owner(shape, Gradient.step_total(shape, index, contribution, value, Core.read_flat(shape, total, index))), contribution,
          {\{Holds: proof => proof} : Holds<{step_owner(shape, Gradient.step_total(shape, index, contribution, value, Core.read_flat(shape, total, index))) == contribution : Core.Dynamic}> ->
            {step_owner(shape, Gradient.step_total(shape, index, contribution, value, Core.read_flat(shape, total, index))) == contribution : Core.Dynamic}}(
            total_owner(shape, index, contribution, value, Core.read_flat(shape, total, index)))),
        same(property))}

def step_retains(-shape: Shape.Shape, +index: U32, state: Gradient.State<shape>) ->
  Holds<Totality.SameSource(state_contribution(shape, state), step_owner(shape, Gradient.step(shape, index, state)))>:
  Gradient.State{contribution, total} = state
  Holds{{\{ElementwiseProof.SourceModel: same =>
    {\{Holds: proof => proof} :
      Holds<Totality.SameSource(contribution, step_owner(shape, Gradient.step_contribution(shape, index, total, Core.read_dynamic(contribution, index))))> ->
      Totality.SameSource(contribution, step_owner(shape, Gradient.step_contribution(shape, index, total, Core.read_dynamic(contribution, index))))}(
      contribution_retains(shape, index, total, contribution, Core.read_dynamic(contribution, index), same))} :
    ElementwiseProof.SourceModel<contribution, ElementwiseProof.dynamic_read_owner(Core.read_dynamic(contribution, index))> ->
    Totality.SameSource(contribution, step_owner(shape, Gradient.step_contribution(shape, index, total, Core.read_dynamic(contribution, index))))}(
    ElementwiseProof.dynamic_read_retains(contribution, index))}

def partial(+below: Bool, total: Maybe<&2, F32>, contribution: Maybe<&2, F32>) -> Maybe<&2, F32>:
  match below:
    case True{}:
      Laws.add_word(total, contribution)
    case False{}:
      total

def Inv(+shape: Shape.Shape, -original: Core.Tensor<shape>, -contribution: Core.Dynamic, +cursor: Nat, total: Core.Tensor<shape>) -> Type:
  @+index: U32 -> @inside: Laws.Inside(shape, index) ->
    {Core.checked_observe(shape, total, index) ==
      partial(Nat.is_lt(U32.to_nat(index), cursor), Core.checked_observe(shape, original, index), Core.checked_observe_dynamic(contribution, index)) : Maybe<&2, F32>}

def StepInv(+shape: Shape.Shape, -original: Core.Tensor<shape>, -contribution: Core.Dynamic, +cursor: Nat, result: Gradient.Step<shape>) -> Type:
  match result:
    case Gradient.Stop{owner, total, error}:
      Unit
    case Gradient.Continue{state}:
      Inv(shape, original, contribution, 1n+cursor, state_total(shape, state))

def Dims(+shape: Shape.Shape, after: Gradient.Slot) -> Type:
  {Laws.held_dimensions(after) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>}

def Values(+shape: Shape.Shape, -original: Core.Tensor<shape>, -contribution: Core.Dynamic, +limit: Nat, after: Gradient.Slot) -> Type:
  @+index: U32 -> @inside: Laws.Inside(shape, index) ->
    {Laws.held_observe(after, index) ==
      partial(Nat.is_lt(U32.to_nat(index), limit), Core.checked_observe(shape, original, index), Core.checked_observe_dynamic(contribution, index)) : Maybe<&2, F32>}

def LoopInv(+shape: Shape.Shape, -original: Core.Tensor<shape>, -contribution: Core.Dynamic, +limit: Nat, outcome: Gradient.Outcome) -> Type:
  match outcome:
    case Gradient.Refused{after, returned, error}:
      Unit
    case Gradient.Accumulated{after}:
      Dims(shape, after) & Values(shape, original, contribution, limit, after)

def written_true(+shape: Shape.Shape, -original: Core.Tensor<shape>, -contribution: Core.Dynamic, +cursor: Nat,
  -before: Core.Tensor<shape>, -written: Core.Tensor<shape>, +index: U32, +accumulated: F32, +value: F32,
  +other: U32, -equal: {True{} == Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) : Bool},
  -inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  -exact: {U32.to_nat(index) == cursor : Nat},
  -inv: Inv(shape, original, contribution, cursor, before),
  -read: {Core.checked_observe(shape, before, index) == Some{accumulated} : Maybe<&2, F32>},
  -stored: {Core.checked_observe(shape, written, index) == Some{F32.add(accumulated, value)} : Maybe<&2, F32>},
  -word: {Core.checked_observe_dynamic(contribution, index) == Some{value} : Maybe<&2, F32>}) ->
  Holds<{Core.checked_observe(shape, written, other) ==
    partial(Nat.is_lt(U32.to_nat(other), 1n+cursor), Core.checked_observe(shape, original, other), Core.checked_observe_dynamic(contribution, other)) : Maybe<&2, F32>}>:
  Holds{
    %{\{Holds: proof => proof} : Holds<{index == other : U32}> -> {index == other : U32}}(word_exact(index, other, equal)) :
      {Core.checked_observe(shape, written, _) ==
        partial(Nat.is_lt(U32.to_nat(_), 1n+cursor), Core.checked_observe(shape, original, _), Core.checked_observe_dynamic(contribution, _)) : Maybe<&2, F32>}
    Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, written, index), Some{F32.add(accumulated, value)},
      partial(Nat.is_lt(U32.to_nat(index), 1n+cursor), Core.checked_observe(shape, original, index), Core.checked_observe_dynamic(contribution, index)),
      stored,
      %Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), 1n+cursor), True{},
        %Equal.sym(Nat, U32.to_nat(index), cursor, exact) : {Nat.is_lt(_, 1n+cursor) == True{} : Bool}
        lt_succ_self(cursor)) :
        {Some{F32.add(accumulated, value)} == partial(_, Core.checked_observe(shape, original, index), Core.checked_observe_dynamic(contribution, index)) : Maybe<&2, F32>}
      %Equal.sym(Maybe<&2, F32>, Core.checked_observe(shape, original, index), Some{accumulated},
        Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, original, index),
          partial(Nat.is_lt(U32.to_nat(index), cursor), Core.checked_observe(shape, original, index), Core.checked_observe_dynamic(contribution, index)),
          Some{accumulated},
          %Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), cursor), False{},
            %Equal.sym(Nat, U32.to_nat(index), cursor, exact) : {Nat.is_lt(_, cursor) == False{} : Bool}
            lt_self(cursor)) :
            {Core.checked_observe(shape, original, index) ==
              partial(_, Core.checked_observe(shape, original, index), Core.checked_observe_dynamic(contribution, index)) : Maybe<&2, F32>}
          {==},
          Equal.trans(Maybe<&2, F32>,
            partial(Nat.is_lt(U32.to_nat(index), cursor), Core.checked_observe(shape, original, index), Core.checked_observe_dynamic(contribution, index)),
            Core.checked_observe(shape, before, index), Some{accumulated},
            Equal.sym(Maybe<&2, F32>, Core.checked_observe(shape, before, index),
              partial(Nat.is_lt(U32.to_nat(index), cursor), Core.checked_observe(shape, original, index), Core.checked_observe_dynamic(contribution, index)),
              inv(index)(
                %Equal.sym(U32, index, other, {\{Holds: proof => proof} : Holds<{index == other : U32}> -> {index == other : U32}}(word_exact(index, other, equal))) :
                  {Nat.is_lt(U32.to_nat(_), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}
                inside)),
            read))) :
        {Some{F32.add(accumulated, value)} == Laws.add_word(_, Core.checked_observe_dynamic(contribution, index)) : Maybe<&2, F32>}
      %Equal.sym(Maybe<&2, F32>, Core.checked_observe_dynamic(contribution, index), Some{value}, word) :
        {Some{F32.add(accumulated, value)} == Laws.add_word(Some{accumulated}, _) : Maybe<&2, F32>}
      {==})}

def written_false(+shape: Shape.Shape, -original: Core.Tensor<shape>, -contribution: Core.Dynamic, +cursor: Nat,
  -before: Core.Tensor<shape>, -read_owner: Core.Tensor<shape>, -written: Core.Tensor<shape>, +index: U32,
  +other: U32, -equal: {False{} == Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) : Bool},
  -inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  -exact: {U32.to_nat(index) == cursor : Nat},
  -inv: Inv(shape, original, contribution, cursor, before),
  -kept: {Core.checked_observe(shape, read_owner, other) == Core.checked_observe(shape, before, other) : Maybe<&2, F32>},
  -frame: {Core.checked_observe(shape, written, other) == Core.checked_observe(shape, read_owner, other) : Maybe<&2, F32>}) ->
  Holds<{Core.checked_observe(shape, written, other) ==
    partial(Nat.is_lt(U32.to_nat(other), 1n+cursor), Core.checked_observe(shape, original, other), Core.checked_observe_dynamic(contribution, other)) : Maybe<&2, F32>}>:
  Holds{Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, written, other), Core.checked_observe(shape, read_owner, other),
    partial(Nat.is_lt(U32.to_nat(other), 1n+cursor), Core.checked_observe(shape, original, other), Core.checked_observe_dynamic(contribution, other)),
    frame,
    Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, read_owner, other), Core.checked_observe(shape, before, other),
      partial(Nat.is_lt(U32.to_nat(other), 1n+cursor), Core.checked_observe(shape, original, other), Core.checked_observe_dynamic(contribution, other)),
      kept,
      Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, before, other),
        partial(Nat.is_lt(U32.to_nat(other), cursor), Core.checked_observe(shape, original, other), Core.checked_observe_dynamic(contribution, other)),
        partial(Nat.is_lt(U32.to_nat(other), 1n+cursor), Core.checked_observe(shape, original, other), Core.checked_observe_dynamic(contribution, other)),
        inv(other)(inside),
        %lt_step(U32.to_nat(other), cursor,
          %exact : {False{} == Nat.is_eq(_, U32.to_nat(other)) : Bool}
          equal) :
          {partial(Nat.is_lt(U32.to_nat(other), cursor), Core.checked_observe(shape, original, other), Core.checked_observe_dynamic(contribution, other)) ==
            partial(_, Core.checked_observe(shape, original, other), Core.checked_observe_dynamic(contribution, other)) : Maybe<&2, F32>}
        {==})))}

def position_inv(+shape: Shape.Shape, -original: Core.Tensor<shape>, -contribution: Core.Dynamic, +cursor: Nat,
  -before: Core.Tensor<shape>, read_owner: Core.Tensor<shape>, -written: Core.Tensor<shape>,
  +index: U32, +accumulated: F32, +value: F32, -unit: Unit,
  +other: U32, decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), U32.to_nat(other))>,
  -inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  -exact: {U32.to_nat(index) == cursor : Nat},
  -meaning: Core.CheckedUpdateMeaning(shape, read_owner, index, F32.add(accumulated, value), (written, Done{unit})),
  -inv: Inv(shape, original, contribution, cursor, before),
  -read: {Core.checked_observe(shape, before, index) == Some{accumulated} : Maybe<&2, F32>},
  -kept: @+position: U32 -> {Core.checked_observe(shape, read_owner, position) == Core.checked_observe(shape, before, position) : Maybe<&2, F32>},
  -word: {Core.checked_observe_dynamic(contribution, index) == Some{value} : Maybe<&2, F32>}) ->
  Holds<{Core.checked_observe(shape, written, other) ==
    partial(Nat.is_lt(U32.to_nat(other), 1n+cursor), Core.checked_observe(shape, original, other), Core.checked_observe_dynamic(contribution, other)) : Maybe<&2, F32>}>:
  match decision:
    case Shape.BTrue{equal}:
      written_true(shape, original, contribution, cursor, before, written, index, accumulated, value, other, equal, inside, exact, inv, read,
        ReduceProof.update_written(shape, read_owner, written, index, F32.add(accumulated, value), unit, meaning), word)
    case Shape.BFalse{equal}:
      written_false(shape, original, contribution, cursor, before, read_owner, written, index, other, equal, inside, exact, inv,
        kept(other),
        ReduceProof.frame_equal(shape, written, read_owner, other,
          ReduceProof.update_frame(shape, read_owner, written, index, F32.add(accumulated, value), unit, meaning)(other)(
            {\{TransformProof.CheckedObservationModel: checked => checked} :
              TransformProof.CheckedObservationModel<shape, read_owner, other> -> Core.ObservationChecked(Core.Packed{shape, read_owner}, other)}(
              TransformProof.checked_observation_model(shape, read_owner, other, inside)))(
            Equal.sym(Bool, False{}, Nat.is_eq(U32.to_nat(index), U32.to_nat(other)), equal))))

def write_inv(+shape: Shape.Shape, -original: Core.Tensor<shape>, -contribution: Core.Dynamic, +cursor: Nat,
  -before: Core.Tensor<shape>, -read_owner: Core.Tensor<shape>, +index: U32, +accumulated: F32, +value: F32,
  owner: Core.Dynamic, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>,
  -exact: {U32.to_nat(index) == cursor : Nat},
  -meaning: Core.CheckedUpdateMeaning(shape, read_owner, index, F32.add(accumulated, value), pair),
  -inv: Inv(shape, original, contribution, cursor, before),
  -read: {Core.checked_observe(shape, before, index) == Some{accumulated} : Maybe<&2, F32>},
  -kept: @+position: U32 -> {Core.checked_observe(shape, read_owner, position) == Core.checked_observe(shape, before, position) : Maybe<&2, F32>},
  -word: {Core.checked_observe_dynamic(contribution, index) == Some{value} : Maybe<&2, F32>}) ->
  Holds<StepInv(shape, original, contribution, cursor, Gradient.step_write(shape, owner, pair))>:
  (written, result) = pair
  match result:
    case Fail{error}:
      Holds{Unit{}}
    case Done{unit}:
      Holds{other => inside => {\{Holds: proof => proof} :
        Holds<{Core.checked_observe(shape, written, other) ==
          partial(Nat.is_lt(U32.to_nat(other), 1n+cursor), Core.checked_observe(shape, original, other), Core.checked_observe_dynamic(contribution, other)) : Maybe<&2, F32>}> ->
        {Core.checked_observe(shape, written, other) ==
          partial(Nat.is_lt(U32.to_nat(other), 1n+cursor), Core.checked_observe(shape, original, other), Core.checked_observe_dynamic(contribution, other)) : Maybe<&2, F32>}}(
        position_inv(shape, original, contribution, cursor, before, read_owner, written, index, accumulated, value, unit,
          other, Shape.bool_case(Nat.is_eq(U32.to_nat(index), U32.to_nat(other))), inside, exact, meaning, inv, read, kept, word))}

def total_inv(+shape: Shape.Shape, -original: Core.Tensor<shape>, -contribution: Core.Dynamic, +cursor: Nat,
  -before: Core.Tensor<shape>, +index: U32, +value: F32, owner: Core.Dynamic,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>,
  -exact: {U32.to_nat(index) == cursor : Nat},
  -source: {Core.read_value(shape, pair) == Core.checked_observe(shape, before, index) : Maybe<&2, F32>},
  -keeps: @+position: U32 -> {Core.retained_read_observe(shape, pair, position) == Core.checked_observe(shape, before, position) : Maybe<&2, F32>},
  -inv: Inv(shape, original, contribution, cursor, before),
  -word: {Core.checked_observe_dynamic(contribution, index) == Some{value} : Maybe<&2, F32>}) ->
  Holds<StepInv(shape, original, contribution, cursor, Gradient.step_total(shape, index, owner, value, pair))>:
  (read_owner, result) = pair
  match result:
    case Fail{error}:
      Holds{Unit{}}
    case Done{+accumulated}:
      write_inv(shape, original, contribution, cursor, before, read_owner, index, accumulated, value, owner,
        Core.write_flat(shape, read_owner, index, F32.add(accumulated, value)), exact,
        {\{Core.CheckedUpdateModel: meaning => meaning} :
          Core.CheckedUpdateModel<shape, read_owner, index, F32.add(accumulated, value)> ->
          Core.CheckedUpdateMeaning(shape, read_owner, index, F32.add(accumulated, value),
            Core.write_flat(shape, read_owner, index, F32.add(accumulated, value)))}(
          CoreProof.checked_update_model(shape, read_owner, index, F32.add(accumulated, value))),
        inv, Equal.sym(Maybe<&2, F32>, Some{accumulated}, Core.checked_observe(shape, before, index), source), keeps, word)

def contribution_inv(+shape: Shape.Shape, -original: Core.Tensor<shape>, -contribution: Core.Dynamic, +cursor: Nat, +index: U32,
  total: Core.Tensor<shape>, -current: Core.Dynamic, pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>,
  -exact: {U32.to_nat(index) == cursor : Nat},
  -source: {Core.dynamic_read_value(pair) == Core.checked_observe_dynamic(current, index) : Maybe<&2, F32>},
  -same: Totality.SameSource(contribution, current),
  -inv: Inv(shape, original, contribution, cursor, total)) ->
  Holds<StepInv(shape, original, contribution, cursor, Gradient.step_contribution(shape, index, total, pair))>:
  (owner, result) = pair
  match result:
    case Fail{error}:
      Holds{Unit{}}
    case Done{+value}:
      total_inv(shape, original, contribution, cursor, total, index, value, owner, Core.read_flat(shape, total, index), exact,
        {\{Core.SourceRead: meaning => meaning} : Core.SourceRead<shape, total, index> ->
          {Core.read_value(shape, Core.read_flat(shape, total, index)) == Core.checked_observe(shape, total, index) : Maybe<&2, F32>}}(
          Core.source_read_model(shape, total, index)),
        {\{Core.SourceReadKeeps: kept => kept} : Core.SourceReadKeeps<shape, total, index> ->
          @+position: U32 -> {Core.retained_read_observe(shape, Core.read_flat(shape, total, index), position) == Core.checked_observe(shape, total, position) : Maybe<&2, F32>}}(
          Core.source_read_keeps(shape, total, index)),
        inv,
        Equal.trans(Maybe<&2, F32>, Core.checked_observe_dynamic(contribution, index), Core.checked_observe_dynamic(current, index), Some{value},
          Equal.sym(Maybe<&2, F32>, Core.checked_observe_dynamic(current, index), Core.checked_observe_dynamic(contribution, index),
            {\{Holds: proof => proof} : Holds<{Core.checked_observe_dynamic(current, index) == Core.checked_observe_dynamic(contribution, index) : Maybe<&2, F32>}> ->
              {Core.checked_observe_dynamic(current, index) == Core.checked_observe_dynamic(contribution, index) : Maybe<&2, F32>}}(
              same_checked(contribution, current, same, index))),
          Equal.sym(Maybe<&2, F32>, Some{value}, Core.checked_observe_dynamic(current, index), source)))

def step_inv(+shape: Shape.Shape, -original: Core.Tensor<shape>, -contribution: Core.Dynamic, +cursor: Nat, +index: U32,
  -exact: {U32.to_nat(index) == cursor : Nat}, state: Gradient.State<shape>,
  -inv: Inv(shape, original, contribution, cursor, state_total(shape, state)),
  -same: Totality.SameSource(contribution, state_contribution(shape, state))) ->
  Holds<StepInv(shape, original, contribution, cursor, Gradient.step(shape, index, state))>:
  Gradient.State{current, total} = state
  contribution_inv(shape, original, contribution, cursor, index, total, current, Core.read_dynamic(current, index), exact,
    {\{Core.DynamicSourceRead: meaning => meaning} : Core.DynamicSourceRead<current, index> ->
      {Core.dynamic_read_value(Core.read_dynamic(current, index)) == Core.checked_observe_dynamic(current, index) : Maybe<&2, F32>}}(
      Core.dynamic_source_read_model(current, index)),
    same, inv)

def continue_inv(+remaining: Nat, +shape: Shape.Shape, -original: Core.Tensor<shape>, -contribution: Core.Dynamic, +cursor: Nat,
  result: Gradient.Step<shape>, -step: StepInv(shape, original, contribution, cursor, result),
  -same: Totality.SameSource(contribution, step_owner(shape, result)),
  -rec: @state: Gradient.State<shape> -> @-inv: Inv(shape, original, contribution, 1n+cursor, state_total(shape, state)) ->
    @-same: Totality.SameSource(contribution, state_contribution(shape, state)) ->
    Holds<LoopInv(shape, original, contribution, Nat.add(remaining, 1n+cursor), Gradient.loop(remaining, shape, 1n+cursor, state))>) ->
  Holds<LoopInv(shape, original, contribution, Nat.add(remaining, 1n+cursor),
    Gradient.continue_step(shape, state => Gradient.loop(remaining, shape, 1n+cursor, state), result))>:
  match result:
    case Gradient.Stop{owner, total, error}:
      Holds{Unit{}}
    case Gradient.Continue{state}:
      Holds{{\{Holds: proof => proof} :
        Holds<LoopInv(shape, original, contribution, Nat.add(remaining, 1n+cursor), Gradient.loop(remaining, shape, 1n+cursor, state))> ->
        LoopInv(shape, original, contribution, Nat.add(remaining, 1n+cursor), Gradient.loop(remaining, shape, 1n+cursor, state))}(rec(state, step, same))}

def conversion_inv(+shape: Shape.Shape, -original: Core.Tensor<shape>, -contribution: Core.Dynamic, +limit: Nat, state: Gradient.State<shape>) ->
  Holds<LoopInv(shape, original, contribution, limit, Gradient.conversion_error(shape, state))>:
  Gradient.State{owner, total} = state
  Holds{Unit{}}

def decided_inv(+remaining: Nat, +shape: Shape.Shape, -original: Core.Tensor<shape>, -contribution: Core.Dynamic, +cursor: Nat, +index: U32,
  state: Gradient.State<shape>, decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>,
  -inv: Inv(shape, original, contribution, cursor, state_total(shape, state)),
  -same: Totality.SameSource(contribution, state_contribution(shape, state)),
  -rec: @state: Gradient.State<shape> -> @-inv: Inv(shape, original, contribution, 1n+cursor, state_total(shape, state)) ->
    @-same: Totality.SameSource(contribution, state_contribution(shape, state)) ->
    Holds<LoopInv(shape, original, contribution, Nat.add(remaining, 1n+cursor), Gradient.loop(remaining, shape, 1n+cursor, state))>) ->
  Holds<LoopInv(shape, original, contribution, Nat.add(remaining, 1n+cursor), Gradient.decided(shape, cursor, index, state,
    state => Gradient.loop(remaining, shape, 1n+cursor, state), decision))>:
  match decision:
    case Shape.BFalse{equal}:
      conversion_inv(shape, original, contribution, Nat.add(remaining, 1n+cursor), state)
    case Shape.BTrue{equal}:
      continue_inv(remaining, shape, original, contribution, cursor, Gradient.step(shape, index, state),
        {\{Holds: proof => proof} : Holds<StepInv(shape, original, contribution, cursor, Gradient.step(shape, index, state))> ->
          StepInv(shape, original, contribution, cursor, Gradient.step(shape, index, state))}(
          step_inv(shape, original, contribution, cursor, index, Lifecycle.natural_equal(U32.to_nat(index), cursor, equal), state, inv, same)),
        {\{Holds: proof => proof} : Holds<Totality.SameSource(contribution, step_owner(shape, Gradient.step(shape, index, state)))> ->
          Totality.SameSource(contribution, step_owner(shape, Gradient.step(shape, index, state)))}(
          same_trans(contribution, state_contribution(shape, state), step_owner(shape, Gradient.step(shape, index, state)), same,
            {\{Holds: proof => proof} : Holds<Totality.SameSource(state_contribution(shape, state), step_owner(shape, Gradient.step(shape, index, state)))> ->
              Totality.SameSource(state_contribution(shape, state), step_owner(shape, Gradient.step(shape, index, state)))}(
              step_retains(shape, index, state)))),
        rec)

def loop_inv(fuel: Nat, +shape: Shape.Shape, -original: Core.Tensor<shape>, -contribution: Core.Dynamic, +cursor: Nat, state: Gradient.State<shape>,
  -inv: Inv(shape, original, contribution, cursor, state_total(shape, state)),
  -same: Totality.SameSource(contribution, state_contribution(shape, state))) ->
  Holds<LoopInv(shape, original, contribution, Nat.add(fuel, cursor), Gradient.loop(fuel, shape, cursor, state))>:
  match fuel:
    case 0n:
      Gradient.State{owner, total} = state
      Holds{({==}, inv)}
    case 1n+remaining:
      Holds{
        %Arith.add_succ(remaining, cursor) : LoopInv(shape, original, contribution, _, Gradient.loop(1n+remaining, shape, cursor, state))
        {\{Holds: proof => proof} :
          Holds<LoopInv(shape, original, contribution, Nat.add(remaining, 1n+cursor), Gradient.loop(1n+remaining, shape, cursor, state))> ->
          LoopInv(shape, original, contribution, Nat.add(remaining, 1n+cursor), Gradient.loop(1n+remaining, shape, cursor, state))}(
          decided_inv(remaining, shape, original, contribution, cursor, U32.from_nat(cursor), state,
            Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)), inv, same,
            state => inv => same => loop_inv(remaining, shape, original, contribution, 1n+cursor, state, inv, same)))}

def initial_inv(+shape: Shape.Shape, total: Core.Tensor<shape>, -contribution: Core.Dynamic) ->
  Holds<Inv(shape, total, contribution, 0n, total)>:
  Holds{index => inside =>
    %Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), 0n), False{}, below_zero(U32.to_nat(index))) :
      {Core.checked_observe(shape, total, index) ==
        partial(_, Core.checked_observe(shape, total, index), Core.checked_observe_dynamic(contribution, index)) : Maybe<&2, F32>}
    {==}}

def finish_mapping(+total_shape: Shape.Shape, +shape: Shape.Shape, -total: Core.Tensor<total_shape>, -contribution: Core.Dynamic,
  outcome: Gradient.Outcome,
  -dimensions: {Shape.dimensions(total_shape) == Shape.dimensions(shape) : List<&2, U32>},
  -loop: LoopInv(total_shape, total, contribution, Nat.add(Shape.elements(total_shape), 0n), outcome)) ->
  Holds<Laws.Mapping(Gradient.Held{Core.Packed{total_shape, total}}, contribution, shape, outcome)>:
  match outcome:
    case Gradient.Refused{after, returned, error}:
      Holds{Unit{}}
    case Gradient.Accumulated{after}:
      Holds{(
        Equal.trans(Maybe<&2, List<&2, U32>>, Laws.held_dimensions(after), Some{Shape.dimensions(total_shape)}, Some{Shape.dimensions(shape)},
          Pair.fst(Dims(total_shape, after), Values(total_shape, total, contribution, Nat.add(Shape.elements(total_shape), 0n), after), loop),
          %dimensions : {Some{Shape.dimensions(total_shape)} == Some{_} : Maybe<&2, List<&2, U32>>}
          {==}),
        index => inside => Equal.trans(Maybe<&2, F32>, Laws.held_observe(after, index),
          partial(Nat.is_lt(U32.to_nat(index), Nat.add(Shape.elements(total_shape), 0n)),
            Core.checked_observe(total_shape, total, index), Core.checked_observe_dynamic(contribution, index)),
          Laws.add_word(Core.checked_observe(total_shape, total, index), Core.checked_observe_dynamic(contribution, index)),
          Pair.snd(Dims(total_shape, after), Values(total_shape, total, contribution, Nat.add(Shape.elements(total_shape), 0n), after), loop)(index)(
            %Equal.sym(List<&2, U32>, Shape.dimensions(total_shape), Shape.dimensions(shape), dimensions) :
              {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(_)) == True{} : Bool}
            inside),
          %Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), Nat.add(Shape.elements(total_shape), 0n)), True{},
            %Equal.sym(Nat, Nat.add(Shape.elements(total_shape), 0n), Shape.elements(total_shape), TransformProof.add_zero_right(Shape.elements(total_shape))) :
              {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
            %Equal.sym(Nat, Shape.elements(total_shape), Shape.numel(Shape.extents(total_shape)), Shape.elements_exact(total_shape)) :
              {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
            %Equal.sym(List<&2, U32>, Shape.dimensions(total_shape), Shape.dimensions(shape), dimensions) :
              {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(_)) == True{} : Bool}
            inside) :
            {partial(_, Core.checked_observe(total_shape, total, index), Core.checked_observe_dynamic(contribution, index)) ==
              Laws.add_word(Core.checked_observe(total_shape, total, index), Core.checked_observe_dynamic(contribution, index)) : Maybe<&2, F32>}
          {==}))}

def held_checked_mapping(+shape: Shape.Shape, contribution: Core.Dynamic, +total_shape: Shape.Shape, total: Core.Tensor<total_shape>, same: Bool,
  -decided: {same == Layout.words_same(Shape.dimensions(total_shape), Shape.dimensions(shape)) : Bool}) ->
  Holds<Laws.Mapping(Gradient.Held{Core.Packed{total_shape, total}}, contribution, shape, Gradient.held_checked(contribution, total_shape, total, same))>:
  match same:
    case False{}:
      Holds{Unit{}}
    case True{}:
      finish_mapping(total_shape, shape, total, contribution,
        Gradient.loop(Shape.elements(total_shape), total_shape, 0n, Gradient.State{contribution, total}),
        {\{Holds: proof => proof} : Holds<{Shape.dimensions(total_shape) == Shape.dimensions(shape) : List<&2, U32>}> ->
          {Shape.dimensions(total_shape) == Shape.dimensions(shape) : List<&2, U32>}}(words_dimensions(total_shape, shape, decided)),
        {\{Holds: proof => proof} :
          Holds<LoopInv(total_shape, total, contribution, Nat.add(Shape.elements(total_shape), 0n),
            Gradient.loop(Shape.elements(total_shape), total_shape, 0n, Gradient.State{contribution, total}))> ->
          LoopInv(total_shape, total, contribution, Nat.add(Shape.elements(total_shape), 0n),
            Gradient.loop(Shape.elements(total_shape), total_shape, 0n, Gradient.State{contribution, total}))}(
          loop_inv(Shape.elements(total_shape), total_shape, total, contribution, 0n, Gradient.State{contribution, total},
            {\{Holds: proof => proof} : Holds<Inv(total_shape, total, contribution, 0n, total)> -> Inv(total_shape, total, contribution, 0n, total)}(
              initial_inv(total_shape, total, contribution)),
            property => {==})))

def held_mapping(+shape: Shape.Shape, contribution: Core.Dynamic, gradient: Core.Dynamic) ->
  Holds<Laws.Mapping(Gradient.Held{gradient}, contribution, shape, Gradient.held(shape, contribution, gradient))>:
  Core.Packed{+total_shape, total} = gradient
  held_checked_mapping(shape, contribution, total_shape, total,
    Layout.words_same(Shape.dimensions(total_shape), Shape.dimensions(shape)), {==})

def agreed_mapping(+shape: Shape.Shape, slot: Gradient.Slot, contribution: Core.Dynamic,
  -agrees: {Totality.dimensions(contribution) == Shape.dimensions(shape) : List<&2, U32>}) ->
  Holds<Laws.Mapping(slot, contribution, shape, Gradient.agreed(shape, slot, contribution))>:
  match slot:
    case Gradient.Unset{}:
      Holds{(
        %agrees : {Some{Totality.dimensions(contribution)} == Some{_} : Maybe<&2, List<&2, U32>>}
        {==},
        index => inside => {==})}
    case Gradient.Held{gradient}:
      held_mapping(shape, contribution, gradient)

def checked_mapping(+shape: Shape.Shape, slot: Gradient.Slot, +contribution_shape: Shape.Shape, tensor: Core.Tensor<contribution_shape>, same: Bool,
  -decided: {same == Layout.words_same(Shape.dimensions(contribution_shape), Shape.dimensions(shape)) : Bool}) ->
  Holds<Laws.Mapping(slot, Core.Packed{contribution_shape, tensor}, shape, Gradient.checked(shape, slot, contribution_shape, tensor, same))>:
  match same:
    case False{}:
      Holds{Unit{}}
    case True{}:
      agreed_mapping(shape, slot, Core.Packed{contribution_shape, tensor},
        {\{Holds: proof => proof} : Holds<{Shape.dimensions(contribution_shape) == Shape.dimensions(shape) : List<&2, U32>}> ->
          {Shape.dimensions(contribution_shape) == Shape.dimensions(shape) : List<&2, U32>}}(words_dimensions(contribution_shape, shape, decided)))

def Laws.accumulate_mapping(shape, slot, contribution):
  Core.Packed{+contribution_shape, tensor} = contribution
  Laws.MappingModel{
    {\{Holds: proof => proof} :
      Holds<Laws.Mapping(slot, Core.Packed{contribution_shape, tensor}, shape,
        Gradient.checked(shape, slot, contribution_shape, tensor, Layout.words_same(Shape.dimensions(contribution_shape), Shape.dimensions(shape))))> ->
      Laws.Mapping(slot, Core.Packed{contribution_shape, tensor}, shape, Gradient.accumulate(shape, slot, Core.Packed{contribution_shape, tensor}))}(
      checked_mapping(shape, slot, contribution_shape, tensor, Layout.words_same(Shape.dimensions(contribution_shape), Shape.dimensions(shape)), {==}))}

def write_success(-shape: Shape.Shape, owner: Core.Dynamic, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>,
  -successful: {TransformTotality.write_succeeded(shape, pair) == True{} : Bool}) ->
  Holds<{step_continued(shape, Gradient.step_write(shape, owner, pair)) == True{} : Bool}>:
  (total, result) = pair
  match result:
    case Fail{error}:
      Holds{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{unit}:
      Holds{{==}}

def total_success(+shape: Shape.Shape, +index: U32, owner: Core.Dynamic, value: F32,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>,
  -successful: {TransformTotality.read_succeeded(shape, pair) == True{} : Bool},
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  Holds<{step_continued(shape, Gradient.step_total(shape, index, owner, value, pair)) == True{} : Bool}>:
  (total, result) = pair
  match result:
    case Fail{error}:
      Holds{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{+accumulated}:
      write_success(shape, owner, Core.write_flat(shape, total, index, F32.add(accumulated, value)),
        {\{TransformTotality.WriteSuccess: successful => successful} :
          TransformTotality.WriteSuccess<shape, total, index, F32.add(accumulated, value)> ->
          {TransformTotality.write_succeeded(shape, Core.write_flat(shape, total, index, F32.add(accumulated, value))) == True{} : Bool}}(
          TransformProof.tensor_write_success(shape, total, index, F32.add(accumulated, value), inside)))

def contribution_success(+shape: Shape.Shape, +index: U32, total: Core.Tensor<shape>,
  pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>,
  -successful: {ElementwiseProof.dynamic_read_succeeded(pair) == True{} : Bool},
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  Holds<{step_continued(shape, Gradient.step_contribution(shape, index, total, pair)) == True{} : Bool}>:
  (owner, result) = pair
  match result:
    case Fail{error}:
      Holds{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{value}:
      total_success(shape, index, owner, value, Core.read_flat(shape, total, index),
        {\{TransformTotality.ReadSuccess: successful => successful} : TransformTotality.ReadSuccess<shape, total, index> ->
          {TransformTotality.read_succeeded(shape, Core.read_flat(shape, total, index)) == True{} : Bool}}(
          TransformProof.read_success_status(shape, total, index, TransformProof.read_success(shape, total, index, inside))),
        inside)

def step_success(+shape: Shape.Shape, +index: U32, state: Gradient.State<shape>,
  -fits: {Totality.dimensions(state_contribution(shape, state)) == Shape.dimensions(shape) : List<&2, U32>},
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  Holds<{step_continued(shape, Gradient.step(shape, index, state)) == True{} : Bool}>:
  Gradient.State{owner, total} = state
  contribution_success(shape, index, total, Core.read_dynamic(owner, index),
    {\{ElementwiseProof.DynamicReadSuccess: successful => successful} : ElementwiseProof.DynamicReadSuccess<owner, index> ->
      {ElementwiseProof.dynamic_read_succeeded(Core.read_dynamic(owner, index)) == True{} : Bool}}(
      ElementwiseProof.dynamic_read_success(owner, index,
        %Equal.sym(List<&2, U32>, Totality.dimensions(owner), Shape.dimensions(shape), fits) :
          {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(_)) == True{} : Bool}
        inside)),
    inside)

def continue_success(+remaining: Nat, +shape: Shape.Shape, +cursor: Nat, result: Gradient.Step<shape>,
  -continued: {step_continued(shape, result) == True{} : Bool},
  -fits: {Totality.dimensions(step_owner(shape, result)) == Shape.dimensions(shape) : List<&2, U32>},
  -rec: @state: Gradient.State<shape> -> @-fits: {Totality.dimensions(state_contribution(shape, state)) == Shape.dimensions(shape) : List<&2, U32>} ->
    Holds<{Laws.accumulated(Gradient.loop(remaining, shape, 1n+cursor, state)) == True{} : Bool}>) ->
  Holds<{Laws.accumulated(Gradient.continue_step(shape, state => Gradient.loop(remaining, shape, 1n+cursor, state), result)) == True{} : Bool}>:
  match result:
    case Gradient.Stop{owner, total, error}:
      Holds{Empty.absurd({False{} == True{} : Bool}, Arith.clash(continued))}
    case Gradient.Continue{state}:
      Holds{{\{Holds: proof => proof} : Holds<{Laws.accumulated(Gradient.loop(remaining, shape, 1n+cursor, state)) == True{} : Bool}> ->
        {Laws.accumulated(Gradient.loop(remaining, shape, 1n+cursor, state)) == True{} : Bool}}(rec(state, fits))}

def decided_success(+remaining: Nat, +shape: Shape.Shape, +cursor: Nat, +index: U32,
  state: Gradient.State<shape>, decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>,
  -exact: {U32.to_nat(index) == cursor : Nat},
  -fits: {Totality.dimensions(state_contribution(shape, state)) == Shape.dimensions(shape) : List<&2, U32>},
  -inside: {Nat.is_lt(cursor, Shape.elements(shape)) == True{} : Bool},
  -rec: @state: Gradient.State<shape> -> @-fits: {Totality.dimensions(state_contribution(shape, state)) == Shape.dimensions(shape) : List<&2, U32>} ->
    Holds<{Laws.accumulated(Gradient.loop(remaining, shape, 1n+cursor, state)) == True{} : Bool}>) ->
  Holds<{Laws.accumulated(Gradient.decided(shape, cursor, index, state,
    state => Gradient.loop(remaining, shape, 1n+cursor, state), decision)) == True{} : Bool}>:
  match decision:
    case Shape.BFalse{equal}:
      Holds{Empty.absurd({Laws.accumulated(Gradient.conversion_error(shape, state)) == True{} : Bool},
        Nats.contradiction(Nat.is_eq(U32.to_nat(index), cursor), PlanProof.eq_true(U32.to_nat(index), cursor, exact), equal))}
    case Shape.BTrue{equal}:
      continue_success(remaining, shape, cursor, Gradient.step(shape, index, state),
        {\{Holds: proof => proof} : Holds<{step_continued(shape, Gradient.step(shape, index, state)) == True{} : Bool}> ->
          {step_continued(shape, Gradient.step(shape, index, state)) == True{} : Bool}}(
          step_success(shape, index, state, fits,
            %Shape.elements_exact(shape) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
            %Equal.sym(Nat, U32.to_nat(index), cursor, exact) : {Nat.is_lt(_, Shape.elements(shape)) == True{} : Bool}
            inside)),
        Equal.trans(List<&2, U32>, Totality.dimensions(step_owner(shape, Gradient.step(shape, index, state))),
          Totality.dimensions(state_contribution(shape, state)), Shape.dimensions(shape),
          {\{Holds: proof => proof} : Holds<{Totality.dimensions(step_owner(shape, Gradient.step(shape, index, state))) == Totality.dimensions(state_contribution(shape, state)) : List<&2, U32>}> ->
            {Totality.dimensions(step_owner(shape, Gradient.step(shape, index, state))) == Totality.dimensions(state_contribution(shape, state)) : List<&2, U32>}}(
            same_dimensions(state_contribution(shape, state), step_owner(shape, Gradient.step(shape, index, state)),
              {\{Holds: proof => proof} : Holds<Totality.SameSource(state_contribution(shape, state), step_owner(shape, Gradient.step(shape, index, state)))> ->
                Totality.SameSource(state_contribution(shape, state), step_owner(shape, Gradient.step(shape, index, state)))}(
                step_retains(shape, index, state)))),
          fits),
        rec)

def loop_success(fuel: Nat, +shape: Shape.Shape, +cursor: Nat, state: Gradient.State<shape>,
  -fits: {Totality.dimensions(state_contribution(shape, state)) == Shape.dimensions(shape) : List<&2, U32>},
  -within: {Nat.is_le(Nat.add(fuel, cursor), Shape.elements(shape)) == True{} : Bool}) ->
  Holds<{Laws.accumulated(Gradient.loop(fuel, shape, cursor, state)) == True{} : Bool}>:
  match fuel:
    case 0n:
      Gradient.State{owner, total} = state
      Holds{{==}}
    case 1n+remaining:
      decided_success(remaining, shape, cursor, U32.from_nat(cursor), state,
        Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)),
        TransformProof.cursor_word_exact(shape, cursor,
          Conversion.lt_le(cursor, Shape.elements(shape),
            Arith.lt_le_trans(cursor, 1n+Nat.add(remaining, cursor), Shape.elements(shape),
              Arith.lt_succ(cursor, Nat.add(remaining, cursor), TransformProof.add_cursor_le(remaining, cursor)), within))),
        fits,
        Arith.lt_le_trans(cursor, 1n+Nat.add(remaining, cursor), Shape.elements(shape),
          Arith.lt_succ(cursor, Nat.add(remaining, cursor), TransformProof.add_cursor_le(remaining, cursor)), within),
        state => fits => loop_success(remaining, shape, 1n+cursor, state, fits,
          %Equal.sym(Nat, Nat.add(remaining, 1n+cursor), 1n+Nat.add(remaining, cursor), Arith.add_succ(remaining, cursor)) :
            {Nat.is_le(_, Shape.elements(shape)) == True{} : Bool}
          within))

def full_loop(+shape: Shape.Shape) -> {Nat.is_le(Nat.add(Shape.elements(shape), 0n), Shape.elements(shape)) == True{} : Bool}:
  %Equal.sym(Nat, Nat.add(Shape.elements(shape), 0n), Shape.elements(shape), TransformProof.add_zero_right(Shape.elements(shape))) :
    {Nat.is_le(_, Shape.elements(shape)) == True{} : Bool}
  Conversion.le_reflexive(Shape.elements(shape))

def held_checked_success(+shape: Shape.Shape, contribution: Core.Dynamic, +total_shape: Shape.Shape, total: Core.Tensor<total_shape>, same: Bool,
  -decided: {same == Layout.words_same(Shape.dimensions(total_shape), Shape.dimensions(shape)) : Bool},
  -fits: {Shape.dimensions(total_shape) == Shape.dimensions(shape) : List<&2, U32>},
  -agrees: {Totality.dimensions(contribution) == Shape.dimensions(shape) : List<&2, U32>}) ->
  Laws.Success<Gradient.held_checked(contribution, total_shape, total, same)>:
  match same:
    case False{}:
      Laws.Success{Empty.absurd({False{} == True{} : Bool},
        Arith.clash(Equal.trans(Bool, False{}, Layout.words_same(Shape.dimensions(total_shape), Shape.dimensions(shape)), True{}, decided,
          %fits : {Layout.words_same(Shape.dimensions(total_shape), _) == True{} : Bool}
          TransformProof.words_same_refl(Shape.dimensions(total_shape)))))}
    case True{}:
      Laws.Success{{\{Holds: proof => proof} :
        Holds<{Laws.accumulated(Gradient.loop(Shape.elements(total_shape), total_shape, 0n, Gradient.State{contribution, total})) == True{} : Bool}> ->
        {Laws.accumulated(Gradient.loop(Shape.elements(total_shape), total_shape, 0n, Gradient.State{contribution, total})) == True{} : Bool}}(
        loop_success(Shape.elements(total_shape), total_shape, 0n, Gradient.State{contribution, total},
          Equal.trans(List<&2, U32>, Totality.dimensions(contribution), Shape.dimensions(shape), Shape.dimensions(total_shape), agrees,
            Equal.sym(List<&2, U32>, Shape.dimensions(total_shape), Shape.dimensions(shape), fits)),
          full_loop(total_shape)))}

def held_success(+shape: Shape.Shape, contribution: Core.Dynamic, gradient: Core.Dynamic,
  -fits: {Totality.dimensions(gradient) == Shape.dimensions(shape) : List<&2, U32>},
  -agrees: {Totality.dimensions(contribution) == Shape.dimensions(shape) : List<&2, U32>}) ->
  Laws.Success<Gradient.held(shape, contribution, gradient)>:
  Core.Packed{+total_shape, total} = gradient
  held_checked_success(shape, contribution, total_shape, total,
    Layout.words_same(Shape.dimensions(total_shape), Shape.dimensions(shape)), {==}, fits, agrees)

def agreed_success(+shape: Shape.Shape, slot: Gradient.Slot, contribution: Core.Dynamic,
  -fits: Laws.Fits(shape, slot),
  -agrees: {Totality.dimensions(contribution) == Shape.dimensions(shape) : List<&2, U32>}) ->
  Laws.Success<Gradient.agreed(shape, slot, contribution)>:
  match slot:
    case Gradient.Unset{}:
      Laws.Success{{==}}
    case Gradient.Held{gradient}:
      held_success(shape, contribution, gradient, fits, agrees)

def checked_success(+shape: Shape.Shape, slot: Gradient.Slot, +contribution_shape: Shape.Shape, tensor: Core.Tensor<contribution_shape>, same: Bool,
  -decided: {same == Layout.words_same(Shape.dimensions(contribution_shape), Shape.dimensions(shape)) : Bool},
  -fits: Laws.Fits(shape, slot),
  -agrees: {Shape.dimensions(contribution_shape) == Shape.dimensions(shape) : List<&2, U32>}) ->
  Laws.Success<Gradient.checked(shape, slot, contribution_shape, tensor, same)>:
  match same:
    case False{}:
      Laws.Success{Empty.absurd({False{} == True{} : Bool},
        Arith.clash(Equal.trans(Bool, False{}, Layout.words_same(Shape.dimensions(contribution_shape), Shape.dimensions(shape)), True{}, decided,
          %agrees : {Layout.words_same(Shape.dimensions(contribution_shape), _) == True{} : Bool}
          TransformProof.words_same_refl(Shape.dimensions(contribution_shape)))))}
    case True{}:
      agreed_success(shape, slot, Core.Packed{contribution_shape, tensor}, fits, agrees)

def Laws.accumulate_valid_success(shape, slot, contribution, fits, agrees):
  Core.Packed{+contribution_shape, tensor} = contribution
  checked_success(shape, slot, contribution_shape, tensor,
    Layout.words_same(Shape.dimensions(contribution_shape), Shape.dimensions(shape)), {==}, fits, agrees)

def accumulated_keeps(-slot: Gradient.Slot, -contribution: Core.Dynamic, outcome: Gradient.Outcome,
  -successful: {Laws.accumulated(outcome) == True{} : Bool}) -> Holds<Laws.RefusalKeeps(slot, contribution, outcome)>:
  match outcome:
    case Gradient.Accumulated{after}:
      Holds{Unit{}}
    case Gradient.Refused{after, returned, error}:
      Holds{Empty.absurd(Laws.RefusalKeeps(slot, contribution, Gradient.Refused{after, returned, error}), Arith.clash(successful))}

def slot_kept(slot: Gradient.Slot) -> Holds<Laws.SlotKept(slot, slot)>:
  match slot:
    case Gradient.Unset{}:
      Holds{{==}}
    case Gradient.Held{gradient}:
      Holds{({==}, index => {==})}

def held_checked_refusal(+shape: Shape.Shape, contribution: Core.Dynamic, +total_shape: Shape.Shape, total: Core.Tensor<total_shape>, same: Bool,
  -decided: {same == Layout.words_same(Shape.dimensions(total_shape), Shape.dimensions(shape)) : Bool},
  -agrees: {Totality.dimensions(contribution) == Shape.dimensions(shape) : List<&2, U32>}) ->
  Holds<Laws.RefusalKeeps(Gradient.Held{Core.Packed{total_shape, total}}, contribution, Gradient.held_checked(contribution, total_shape, total, same))>:
  match same:
    case False{}:
      Holds{({==}, (property => {==}, ({==}, index => {==})))}
    case True{}:
      accumulated_keeps(Gradient.Held{Core.Packed{total_shape, total}}, contribution,
        Gradient.loop(Shape.elements(total_shape), total_shape, 0n, Gradient.State{contribution, total}),
        {\{Laws.Success: successful => successful} :
          Laws.Success<Gradient.loop(Shape.elements(total_shape), total_shape, 0n, Gradient.State{contribution, total})> ->
          {Laws.accumulated(Gradient.loop(Shape.elements(total_shape), total_shape, 0n, Gradient.State{contribution, total})) == True{} : Bool}}(
          held_checked_success(shape, contribution, total_shape, total, True{}, decided,
            {\{Holds: proof => proof} : Holds<{Shape.dimensions(total_shape) == Shape.dimensions(shape) : List<&2, U32>}> ->
              {Shape.dimensions(total_shape) == Shape.dimensions(shape) : List<&2, U32>}}(words_dimensions(total_shape, shape, decided)),
            agrees)))

def held_refusal(+shape: Shape.Shape, contribution: Core.Dynamic, gradient: Core.Dynamic,
  -agrees: {Totality.dimensions(contribution) == Shape.dimensions(shape) : List<&2, U32>}) ->
  Holds<Laws.RefusalKeeps(Gradient.Held{gradient}, contribution, Gradient.held(shape, contribution, gradient))>:
  Core.Packed{+total_shape, total} = gradient
  held_checked_refusal(shape, contribution, total_shape, total,
    Layout.words_same(Shape.dimensions(total_shape), Shape.dimensions(shape)), {==}, agrees)

def agreed_refusal(+shape: Shape.Shape, slot: Gradient.Slot, contribution: Core.Dynamic,
  -agrees: {Totality.dimensions(contribution) == Shape.dimensions(shape) : List<&2, U32>}) ->
  Holds<Laws.RefusalKeeps(slot, contribution, Gradient.agreed(shape, slot, contribution))>:
  match slot:
    case Gradient.Unset{}:
      Holds{Unit{}}
    case Gradient.Held{gradient}:
      held_refusal(shape, contribution, gradient, agrees)

def checked_refusal(+shape: Shape.Shape, slot: Gradient.Slot, +contribution_shape: Shape.Shape, tensor: Core.Tensor<contribution_shape>, same: Bool,
  -decided: {same == Layout.words_same(Shape.dimensions(contribution_shape), Shape.dimensions(shape)) : Bool}) ->
  Holds<Laws.RefusalKeeps(slot, Core.Packed{contribution_shape, tensor}, Gradient.checked(shape, slot, contribution_shape, tensor, same))>:
  match same:
    case False{}:
      Holds{({==}, (property => {==}, {\{Holds: proof => proof} : Holds<Laws.SlotKept(slot, slot)> -> Laws.SlotKept(slot, slot)}(slot_kept(slot))))}
    case True{}:
      agreed_refusal(shape, slot, Core.Packed{contribution_shape, tensor},
        {\{Holds: proof => proof} : Holds<{Shape.dimensions(contribution_shape) == Shape.dimensions(shape) : List<&2, U32>}> ->
          {Shape.dimensions(contribution_shape) == Shape.dimensions(shape) : List<&2, U32>}}(words_dimensions(contribution_shape, shape, decided)))

def Laws.accumulate_refusal_keeps(shape, slot, contribution):
  Core.Packed{+contribution_shape, tensor} = contribution
  Laws.RefusalModel{
    {\{Holds: proof => proof} :
      Holds<Laws.RefusalKeeps(slot, Core.Packed{contribution_shape, tensor},
        Gradient.checked(shape, slot, contribution_shape, tensor, Layout.words_same(Shape.dimensions(contribution_shape), Shape.dimensions(shape))))> ->
      Laws.RefusalKeeps(slot, Core.Packed{contribution_shape, tensor}, Gradient.accumulate(shape, slot, Core.Packed{contribution_shape, tensor}))}(
      checked_refusal(shape, slot, contribution_shape, tensor, Layout.words_same(Shape.dimensions(contribution_shape), Shape.dimensions(shape)), {==}))}

def Laws.grad_unset_none():
  {==}

def Laws.grad_held_some(held):
  {==}

def Laws.reset_unset(slot):
  {==}

def reset_checked(+shape: Shape.Shape, +contribution_shape: Shape.Shape, tensor: Core.Tensor<contribution_shape>, same: Bool,
  decided: {same == Layout.words_same(Shape.dimensions(contribution_shape), Shape.dimensions(shape)) : Bool},
  agrees: {Shape.dimensions(contribution_shape) == Shape.dimensions(shape) : List<&2, U32>}) ->
  {Gradient.checked(shape, Gradient.Unset{}, contribution_shape, tensor, same) ==
    Gradient.Accumulated{Gradient.Held{Core.Packed{contribution_shape, tensor}}} : Gradient.Outcome}:
  match same:
    case False{}:
      Empty.absurd({Gradient.checked(shape, Gradient.Unset{}, contribution_shape, tensor, False{}) ==
        Gradient.Accumulated{Gradient.Held{Core.Packed{contribution_shape, tensor}}} : Gradient.Outcome},
        Arith.clash(Equal.trans(Bool, False{}, Layout.words_same(Shape.dimensions(contribution_shape), Shape.dimensions(shape)), True{}, decided,
          %agrees : {Layout.words_same(Shape.dimensions(contribution_shape), _) == True{} : Bool}
          TransformProof.words_same_refl(Shape.dimensions(contribution_shape)))))
    case True{}:
      {==}

def Laws.reset_then_accumulate_moves_in(shape, slot, contribution, agrees):
  Core.Packed{+contribution_shape, tensor} = contribution
  reset_checked(shape, contribution_shape, tensor, Layout.words_same(Shape.dimensions(contribution_shape), Shape.dimensions(shape)), {==}, agrees)

def checked_wrong_shape(+shape: Shape.Shape, slot: Gradient.Slot, +contribution_shape: Shape.Shape,
  tensor: Core.Tensor<contribution_shape>, flag: Bool,
  decided: {flag == Layout.words_same(Shape.dimensions(contribution_shape), Shape.dimensions(shape)) : Bool},
  differ: {Shape.dimensions(contribution_shape) != Shape.dimensions(shape) : List<&2, U32>}) ->
  {Gradient.checked(shape, slot, contribution_shape, tensor, flag) ==
    Gradient.Refused{slot, Core.Packed{contribution_shape, tensor}, Gradient.ShapeMismatch{}} : Gradient.Outcome}:
  match flag:
    case False{}:
      {==}
    case True{}:
      Empty.absurd({Gradient.checked(shape, slot, contribution_shape, tensor, True{}) ==
        Gradient.Refused{slot, Core.Packed{contribution_shape, tensor}, Gradient.ShapeMismatch{}} : Gradient.Outcome},
        differ(TransformProof.words_equal(Shape.dimensions(contribution_shape), Shape.dimensions(shape), decided)))

def Laws.accumulate_wrong_shape_refused(shape, slot, contribution, differ):
  Core.Packed{+contribution_shape, tensor} = contribution
  checked_wrong_shape(shape, slot, contribution_shape, tensor,
    Layout.words_same(Shape.dimensions(contribution_shape), Shape.dimensions(shape)), {==}, differ)
