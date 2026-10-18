import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ../storage/certificates.bend as Certificates
import ./core.bend as Core
import ./elementwise.bend as Elementwise
import ./elementwise_totality.bend as Totality
import ./elementwise_LAWS.bend as Laws
import ./lifecycle.bend as Lifecycle
import ../shape/LAWS.bend as ShapeLaws
import ../shape/plan_proofs.bend as PlanProof
import ../shape/unique_proofs.bend as Unique
import ../shape/nat_proofs.bend as Nats
import ../shape/flat.bend as Flat
import ../storage/arith.bend as Arith
import ./transform_PROOF.bend as TransformProof
import ./transform_totality.bend as TransformTotality
import ./word_conversion.bend as Conversion
import ./allocation_PROOF.bend as AllocationProof
import ./allocation_spec.bend as Allocation
import ./elementwise_parallel_proof.bend as Parallel

def read_owner(pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>) -> Storage.Storage:
  (owner, result) = pair
  owner

def storage_inspect(property: Totality.Property, storage: Storage.Storage) -> Totality.Inspection:
  match property:
    case Totality.Dimensions{}:
      Totality.Dims{Nil{}}
    case Totality.Checked{index}:
      Totality.Value{Lifecycle.checked_observe(storage, index)}
    case Totality.Raw{index}:
      Totality.Value{Lifecycle.observe(storage, index)}

def finish_read_inspect(+depth: Nat, +length: Nat, -original: Array<F32>, +index: U32, +property: Totality.Property,
  pair: Array<F32> & F32, -same: {pair == Array.get(F32, original, index) : Array<F32> & F32},
  -balanced: Model.Balanced(F32, depth, original), -valid: {Storage.bounds(depth, length) == True{} : Bool}) ->
  {storage_inspect(property, read_owner(Storage.finish_read(depth, length, original, index, pair, same, balanced, valid))) ==
    storage_inspect(property, Storage.Owned{depth, length, Refinement.first(F32, pair),
      Certificates.get_result(F32, depth, original, index, pair, same, balanced), valid}) : Totality.Inspection}:
  (array, value) = pair
  {==}

def array_inspect(+depth: Nat, +length: Nat, property: Totality.Property, array: Array<F32>) -> Totality.Inspection:
  match property:
    case Totality.Dimensions{}:
      Totality.Dims{Nil{}}
    case Totality.Checked{index}:
      Totality.Value{Lifecycle.checked_observe_array(depth, length, array, index)}
    case Totality.Raw{index}:
      Totality.Value{Model.observe(F32, depth, array, Refinement.route(depth, index))}

type InspectionModel<-left: Totality.Inspection, -right: Totality.Inspection> is Data:
  InspectionModel{-equal: {left == right : Totality.Inspection}}

def read_hit_inspect(+depth: Nat, +length: Nat, +index: U32, +property: Totality.Property, array: Array<F32>,
  balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool}) ->
  InspectionModel<storage_inspect(property, read_owner(Storage.finish_read(depth, length, array, index, Array.get(F32, array, index), {==}, balanced, valid))),
    array_inspect(depth, length, property, array)>:
  match property:
    case Totality.Dimensions{}:
      InspectionModel{finish_read_inspect(depth, length, array, index, Totality.Dimensions{}, Array.get(F32, array, index), {==}, balanced, valid)}
    case Totality.Checked{+other}:
      InspectionModel{Equal.trans(Totality.Inspection,
        storage_inspect(Totality.Checked{other}, read_owner(Storage.finish_read(depth, length, array, index, Array.get(F32, array, index), {==}, balanced, valid))),
        array_inspect(depth, length, Totality.Checked{other}, Refinement.first(F32, Array.get(F32, array, index))),
        array_inspect(depth, length, Totality.Checked{other}, array),
        finish_read_inspect(depth, length, array, index, Totality.Checked{other}, Array.get(F32, array, index), {==}, balanced, valid),
        Equal.cong(Array<F32>, Totality.Inspection, a => array_inspect(depth, length, Totality.Checked{other}, a),
          Refinement.first(F32, Array.get(F32, array, index)), array, Refinement.law_get_retains(F32, depth, array, index, balanced)))}
    case Totality.Raw{+other}:
      InspectionModel{Equal.trans(Totality.Inspection,
        storage_inspect(Totality.Raw{other}, read_owner(Storage.finish_read(depth, length, array, index, Array.get(F32, array, index), {==}, balanced, valid))),
        array_inspect(depth, length, Totality.Raw{other}, Refinement.first(F32, Array.get(F32, array, index))),
        array_inspect(depth, length, Totality.Raw{other}, array),
        finish_read_inspect(depth, length, array, index, Totality.Raw{other}, Array.get(F32, array, index), {==}, balanced, valid),
        Equal.cong(Array<F32>, Totality.Inspection, a => array_inspect(depth, length, Totality.Raw{other}, a),
          Refinement.first(F32, Array.get(F32, array, index)), array, Refinement.law_get_retains(F32, depth, array, index, balanced)))}

def read_checked_inspect(+depth: Nat, +length: Nat, +index: U32, allowed: Bool, +property: Totality.Property, array: Array<F32>,
  -decision: {allowed == Nat.is_lt(U32.to_nat(index), length) : Bool},
  balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool}) ->
  InspectionModel<storage_inspect(property, read_owner(Storage.get_checked(depth, length, array, index, allowed, decision, balanced, valid))),
    array_inspect(depth, length, property, array)>:
  match allowed:
    case False{}:
      match property:
        case Totality.Dimensions{}:
          InspectionModel{{==}}
        case Totality.Checked{other}:
          InspectionModel{{==}}
        case Totality.Raw{other}:
          InspectionModel{{==}}
    case True{}:
      read_hit_inspect(depth, length, index, property, array, balanced, valid)

def tensor_read_owner(-shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) -> Core.Tensor<shape>:
  (owner, result) = pair
  owner

def storage_tensor_inspect(shape: Shape.Shape, property: Totality.Property, storage: Storage.Storage) -> Totality.Inspection:
  match property:
    case Totality.Dimensions{}:
      Totality.Dims{Shape.dimensions(shape)}
    case Totality.Checked{index}:
      Totality.Value{Lifecycle.checked_observe(storage, index)}
    case Totality.Raw{index}:
      Totality.Value{Lifecycle.observe(storage, index)}

def core_read_projection(+shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32,
  +property: Totality.Property, pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>,
  -same: {pair == Storage.get(original, index) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}) ->
  {Totality.inspect(property, Core.Packed{shape, tensor_read_owner(shape, Core.read_result(shape, original, valid, index, pair, same))}) ==
    storage_tensor_inspect(shape, property, read_owner(pair)) : Totality.Inspection}:
  match property:
    case Totality.Dimensions{}:
      (storage, result) = pair
      {==}
    case Totality.Checked{other}:
      (storage, result) = pair
      {==}
    case Totality.Raw{other}:
      (storage, result) = pair
      {==}

type SourceModel<-original: Core.Dynamic, -retained: Core.Dynamic> is Data:
  SourceModel{-same: Totality.SameSource(original, retained)}

def dynamic_read_owner(pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> Core.Dynamic:
  (owner, result) = pair
  owner

def tensor_read_dimensions(+shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) ->
  {Totality.inspect(Totality.Dimensions{}, Core.Packed{shape, tensor_read_owner(shape, pair)}) == Totality.Dims{Shape.dimensions(shape)} : Totality.Inspection}:
  (owner, result) = pair
  {==}

def dynamic_read_projection(+shape: Shape.Shape, +property: Totality.Property,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) ->
  {Totality.inspect(property, dynamic_read_owner(Core.read_dynamic_result(shape, pair))) ==
    Totality.inspect(property, Core.Packed{shape, tensor_read_owner(shape, pair)}) : Totality.Inspection}:
  (owner, result) = pair
  {==}

def owned_read_property(+shape: Shape.Shape, +depth: Nat, +length: Nat, +index: U32, property: Totality.Property,
  -array: Array<F32>, -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  InspectionModel<Totality.inspect(property, dynamic_read_owner(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}}, index))),
    Totality.inspect(property, Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}})>:
  match property:
    case Totality.Dimensions{}:
      InspectionModel{Equal.trans(Totality.Inspection,
        Totality.inspect(Totality.Dimensions{}, dynamic_read_owner(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}}, index))),
        Totality.inspect(Totality.Dimensions{}, Core.Packed{shape, tensor_read_owner(shape, Core.read_flat(shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, index))}),
        Totality.Dims{Shape.dimensions(shape)},
        dynamic_read_projection(shape, Totality.Dimensions{}, Core.read_flat(shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, index)),
        tensor_read_dimensions(shape, Core.read_flat(shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, index)))}
    case Totality.Checked{+other}:
      InspectionModel{Equal.trans(Totality.Inspection,
        Totality.inspect(Totality.Checked{other}, dynamic_read_owner(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}}, index))),
        storage_inspect(Totality.Checked{other}, read_owner(Storage.get(Storage.Owned{depth, length, array, balanced, valid}, index))),
        array_inspect(depth, length, Totality.Checked{other}, array),
        Equal.trans(Totality.Inspection,
          Totality.inspect(Totality.Checked{other}, dynamic_read_owner(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}}, index))),
          Totality.inspect(Totality.Checked{other}, Core.Packed{shape, tensor_read_owner(shape, Core.read_flat(shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, index))}),
          storage_inspect(Totality.Checked{other}, read_owner(Storage.get(Storage.Owned{depth, length, array, balanced, valid}, index))),
          dynamic_read_projection(shape, Totality.Checked{other}, Core.read_flat(shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, index)),
          core_read_projection(shape, Storage.Owned{depth, length, array, balanced, valid}, exact, index, Totality.Checked{other}, Storage.get(Storage.Owned{depth, length, array, balanced, valid}, index), {==})),
        {\{InspectionModel: equal => equal} :
          InspectionModel<storage_inspect(Totality.Checked{other}, read_owner(Storage.get(Storage.Owned{depth, length, array, balanced, valid}, index))),
            array_inspect(depth, length, Totality.Checked{other}, array)> ->
          {storage_inspect(Totality.Checked{other}, read_owner(Storage.get(Storage.Owned{depth, length, array, balanced, valid}, index))) ==
            array_inspect(depth, length, Totality.Checked{other}, array) : Totality.Inspection}}(
              read_checked_inspect(depth, length, index, Nat.is_lt(U32.to_nat(index), length), Totality.Checked{other}, array, {==}, balanced, valid)))}
    case Totality.Raw{+other}:
      InspectionModel{Equal.trans(Totality.Inspection,
        Totality.inspect(Totality.Raw{other}, dynamic_read_owner(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}}, index))),
        storage_inspect(Totality.Raw{other}, read_owner(Storage.get(Storage.Owned{depth, length, array, balanced, valid}, index))),
        array_inspect(depth, length, Totality.Raw{other}, array),
        Equal.trans(Totality.Inspection,
          Totality.inspect(Totality.Raw{other}, dynamic_read_owner(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}}, index))),
          Totality.inspect(Totality.Raw{other}, Core.Packed{shape, tensor_read_owner(shape, Core.read_flat(shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, index))}),
          storage_inspect(Totality.Raw{other}, read_owner(Storage.get(Storage.Owned{depth, length, array, balanced, valid}, index))),
          dynamic_read_projection(shape, Totality.Raw{other}, Core.read_flat(shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, index)),
          core_read_projection(shape, Storage.Owned{depth, length, array, balanced, valid}, exact, index, Totality.Raw{other}, Storage.get(Storage.Owned{depth, length, array, balanced, valid}, index), {==})),
        {\{InspectionModel: equal => equal} :
          InspectionModel<storage_inspect(Totality.Raw{other}, read_owner(Storage.get(Storage.Owned{depth, length, array, balanced, valid}, index))),
            array_inspect(depth, length, Totality.Raw{other}, array)> ->
          {storage_inspect(Totality.Raw{other}, read_owner(Storage.get(Storage.Owned{depth, length, array, balanced, valid}, index))) ==
            array_inspect(depth, length, Totality.Raw{other}, array) : Totality.Inspection}}(
              read_checked_inspect(depth, length, index, Nat.is_lt(U32.to_nat(index), length), Totality.Raw{other}, array, {==}, balanced, valid)))}

def empty_read_property(+shape: Shape.Shape, +index: U32, property: Totality.Property,
  -exact: {0n == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  InspectionModel<Totality.inspect(property, dynamic_read_owner(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Storage.Empty{}, exact}}, index))),
    Totality.inspect(property, Core.Packed{shape, Core.TTensor{Storage.Empty{}, exact}})>:
  match property:
    case Totality.Dimensions{}:
      InspectionModel{{==}}
    case Totality.Checked{other}:
      InspectionModel{{==}}
    case Totality.Raw{other}:
      InspectionModel{{==}}

def dynamic_read_retains(original: Core.Dynamic, +index: U32) -> SourceModel<original, dynamic_read_owner(Core.read_dynamic(original, index))>:
  Core.Packed{+shape, Core.TTensor{storage, exact}} = original
  match storage:
    case Storage.Empty{}:
      SourceModel{property => {\{InspectionModel: equal => equal} :
        InspectionModel<Totality.inspect(property, dynamic_read_owner(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Storage.Empty{}, exact}}, index))),
          Totality.inspect(property, Core.Packed{shape, Core.TTensor{Storage.Empty{}, exact}})> ->
        {Totality.inspect(property, dynamic_read_owner(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Storage.Empty{}, exact}}, index))) ==
          Totality.inspect(property, Core.Packed{shape, Core.TTensor{Storage.Empty{}, exact}}) : Totality.Inspection}}(
            empty_read_property(shape, index, property, exact))}
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      SourceModel{property => {\{InspectionModel: equal => equal} :
        InspectionModel<Totality.inspect(property, dynamic_read_owner(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}}, index))),
          Totality.inspect(property, Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}})> ->
        {Totality.inspect(property, dynamic_read_owner(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}}, index))) ==
          Totality.inspect(property, Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}}) : Totality.Inspection}}(
            owned_read_property(shape, depth, length, index, property, array, balanced, valid, exact))}

def step_left_owner(-shape: Shape.Shape, result: Elementwise.Step<shape>) -> Core.Dynamic:
  match result:
    case Elementwise.Stop{left, right, error}:
      left
    case Elementwise.Continue{Elementwise.State{left, right, output}}:
      left

def step_right_owner(-shape: Shape.Shape, result: Elementwise.Step<shape>) -> Core.Dynamic:
  match result:
    case Elementwise.Stop{left, right, error}:
      right
    case Elementwise.Continue{Elementwise.State{left, right, output}}:
      right

type StepSources<-shape: Shape.Shape, -left: Core.Dynamic, -right: Core.Dynamic, -result: Elementwise.Step<shape>> is Data:
  StepSources{-left: Totality.SameSource(left, step_left_owner(shape, result)),
    -right: Totality.SameSource(right, step_right_owner(shape, result))}

def step_sources_reify(-shape: Shape.Shape, -left: Core.Dynamic, -right: Core.Dynamic, -result: Elementwise.Step<shape>,
  -model: StepSources<shape, left, right, result>) -> StepSources<shape, left, right, result>:
  StepSources{
    {\{StepSources: left_same => right_same => left_same} : StepSources<shape, left, right, result> -> Totality.SameSource(left, step_left_owner(shape, result))}(model),
    {\{StepSources: left_same => right_same => right_same} : StepSources<shape, left, right, result> -> Totality.SameSource(right, step_right_owner(shape, result))}(model)}

def step_write_retains(-shape: Shape.Shape, -original_left: Core.Dynamic, -original_right: Core.Dynamic,
  left: Core.Dynamic, right: Core.Dynamic, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>,
  -left_same: Totality.SameSource(original_left, left), -right_same: Totality.SameSource(original_right, right)) ->
  StepSources<shape, original_left, original_right, Elementwise.step_write(shape, left, right, pair)>:
  (output, result) = pair
  match result:
    case Fail{error}:
      StepSources{left_same, right_same}
    case Done{unit}:
      StepSources{left_same, right_same}

def step_right_retains(+operation: Elementwise.Operation, -shape: Shape.Shape, +index: U32,
  -original_left: Core.Dynamic, -original_right: Core.Dynamic, left: Core.Dynamic,
  output: Core.Tensor<shape>, left_value: F32, pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>,
  -left_same: Totality.SameSource(original_left, left),
  -right_same: Totality.SameSource(original_right, dynamic_read_owner(pair))) ->
  StepSources<shape, original_left, original_right, Elementwise.step_right(operation, shape, index, left, output, left_value, pair)>:
  (right, result) = pair
  match result:
    case Fail{error}:
      StepSources{left_same, right_same}
    case Done{value}:
      step_write_retains(shape, original_left, original_right, left, right,
        Core.write_flat(shape, output, index, Elementwise.scalar(operation, left_value, value)), left_same, right_same)

def step_left_retains(+operation: Elementwise.Operation, -shape: Shape.Shape, +index: U32,
  -original_left: Core.Dynamic, -original_right: Core.Dynamic, right: Core.Dynamic,
  output: Core.Tensor<shape>, +right_index: U32, pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>,
  -left_same: Totality.SameSource(original_left, dynamic_read_owner(pair)),
  -right_same: Totality.SameSource(original_right, right)) ->
  StepSources<shape, original_left, original_right, Elementwise.step_left(operation, shape, index, right, output, right_index, pair)>:
  (left, result) = pair
  match result:
    case Fail{error}:
      StepSources{left_same, right_same}
    case Done{value}:
      step_sources_reify(shape, original_left, original_right,
        Elementwise.step_right(operation, shape, index, left, output, value, Core.read_dynamic(right, right_index)),
        {\{SourceModel: same => step_right_retains(operation, shape, index, original_left, original_right, left, output, value,
        Core.read_dynamic(right, right_index), left_same,
        property => Equal.trans(Totality.Inspection,
          Totality.inspect(property, dynamic_read_owner(Core.read_dynamic(right, right_index))),
          Totality.inspect(property, right), Totality.inspect(property, original_right), same(property), right_same(property)))} :
        SourceModel<right, dynamic_read_owner(Core.read_dynamic(right, right_index))> ->
        StepSources<shape, original_left, original_right, Elementwise.step_right(operation, shape, index, left, output, value, Core.read_dynamic(right, right_index))>}(
          dynamic_read_retains(right, right_index)))

def step_indices_retains(+operation: Elementwise.Operation, -shape: Shape.Shape, +index: U32,
  state: Elementwise.State<shape>, result: Result<&2, &1, Core.Error, U32 & U32>) ->
  StepSources<shape,
    step_left_owner(shape, Elementwise.Continue{state}), step_right_owner(shape, Elementwise.Continue{state}),
    Elementwise.step_indices(operation, shape, index, state, result)>:
  Elementwise.State{left, right, output} = state
  match result:
    case Fail{error}:
      StepSources{property => {==}, property => {==}}
    case Done{indices}:
      (+left_index, +right_index) = indices
      step_sources_reify(shape, left, right,
        Elementwise.step_left(operation, shape, index, right, output, right_index, Core.read_dynamic(left, left_index)),
        {\{SourceModel: same => step_left_retains(operation, shape, index, left, right, right, output, right_index,
        Core.read_dynamic(left, left_index), same, property => {==})} :
        SourceModel<left, dynamic_read_owner(Core.read_dynamic(left, left_index))> ->
        StepSources<shape, left, right, Elementwise.step_left(operation, shape, index, right, output, right_index, Core.read_dynamic(left, left_index))>}(
          dynamic_read_retains(left, left_index)))

def natural_left_owner(-shape: Shape.Shape, -cursor: Nat, result: Elementwise.NaturalStep<shape, cursor>) -> Core.Dynamic:
  match result:
    case Elementwise.NaturalStop{left, right, error}:
      left
    case Elementwise.NaturalContinue{state, index, exact}:
      step_left_owner(shape, Elementwise.Continue{state})

def natural_right_owner(-shape: Shape.Shape, -cursor: Nat, result: Elementwise.NaturalStep<shape, cursor>) -> Core.Dynamic:
  match result:
    case Elementwise.NaturalStop{left, right, error}:
      right
    case Elementwise.NaturalContinue{state, index, exact}:
      step_right_owner(shape, Elementwise.Continue{state})

type NaturalSources<-shape: Shape.Shape, -cursor: Nat, -left: Core.Dynamic, -right: Core.Dynamic,
  -result: Elementwise.NaturalStep<shape, cursor>> is Data:
  NaturalSources{-left: Totality.SameSource(left, natural_left_owner(shape, cursor, result)),
    -right: Totality.SameSource(right, natural_right_owner(shape, cursor, result))}

def natural_result_retains(-shape: Shape.Shape, -cursor: Nat, -index: U32, -exact: {U32.to_nat(index) == cursor : Nat},
  -left: Core.Dynamic, -right: Core.Dynamic, result: Elementwise.Step<shape>, -sources: StepSources<shape, left, right, result>) ->
  NaturalSources<shape, cursor, left, right, Elementwise.natural_step_result(shape, cursor, index, exact, result)>:
  match result:
    case Elementwise.Stop{retained_left, retained_right, error}:
      NaturalSources{
        {\{StepSources: l => r => l} : StepSources<shape, left, right, Elementwise.Stop{retained_left, retained_right, error}> -> Totality.SameSource(left, retained_left)}(sources),
        {\{StepSources: l => r => r} : StepSources<shape, left, right, Elementwise.Stop{retained_left, retained_right, error}> -> Totality.SameSource(right, retained_right)}(sources)}
    case Elementwise.Continue{state}:
      NaturalSources{
        {\{StepSources: l => r => l} : StepSources<shape, left, right, Elementwise.Continue{state}> -> Totality.SameSource(left, step_left_owner(shape, Elementwise.Continue{state}))}(sources),
        {\{StepSources: l => r => r} : StepSources<shape, left, right, Elementwise.Continue{state}> -> Totality.SameSource(right, step_right_owner(shape, Elementwise.Continue{state}))}(sources)}

def natural_decided_retains(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +cursor: Nat, +index: U32,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>, state: Elementwise.State<Shape.output_shape(plan)>) ->
  NaturalSources<Shape.output_shape(plan), cursor,
    step_left_owner(Shape.output_shape(plan), Elementwise.Continue{state}), step_right_owner(Shape.output_shape(plan), Elementwise.Continue{state}),
    Elementwise.natural_step_decided(operation, plan, cursor, index, state, decision)>:
  match decision:
    case Shape.BFalse{equal}:
      Elementwise.State{left, right, output} = state
      NaturalSources{property => {==}, property => {==}}
    case Shape.BTrue{equal}:
      natural_result_retains(Shape.output_shape(plan), cursor, index, Lifecycle.natural_equal(U32.to_nat(index), cursor, equal),
        step_left_owner(Shape.output_shape(plan), Elementwise.Continue{state}), step_right_owner(Shape.output_shape(plan), Elementwise.Continue{state}),
        Elementwise.step(operation, plan, index, state),
        step_indices_retains(operation, Shape.output_shape(plan), index, state, Elementwise.input_indices(plan, index)))

def retention_reify(-left: Core.Dynamic, -right: Core.Dynamic,
  -pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  -model: Totality.Retention<left, right, pair>) -> Totality.Retention<left, right, pair>:
  Totality.Retention{
    {\{Totality.Retention: l => r => l} : Totality.Retention<left, right, pair> -> Totality.SameSource(left, Totality.retained_left(pair))}(model),
    {\{Totality.Retention: l => r => r} : Totality.Retention<left, right, pair> -> Totality.SameSource(right, Totality.retained_right(pair))}(model)}

def loop_finish_retains(+fuel: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +cursor: Nat,
  -left: Core.Dynamic, -right: Core.Dynamic, result: Elementwise.NaturalStep<Shape.output_shape(plan), cursor>,
  -sources: NaturalSources<Shape.output_shape(plan), cursor, left, right, result>,
  -rec: @state: Elementwise.State<Shape.output_shape(plan)> ->
    @-l: Totality.SameSource(left, step_left_owner(Shape.output_shape(plan), Elementwise.Continue{state})) ->
    @-r: Totality.SameSource(right, step_right_owner(Shape.output_shape(plan), Elementwise.Continue{state})) ->
    Totality.Retention<left, right, Elementwise.loop(fuel, operation, plan, 1n+cursor, state)>) ->
  Totality.Retention<left, right, Elementwise.finish_step(Shape.output_shape(plan), cursor,
    state => Elementwise.loop(fuel, operation, plan, 1n+cursor, state), result)>:
  match result:
    case Elementwise.NaturalStop{retained_left, retained_right, error}:
      Totality.Retention{
        {\{NaturalSources: l => r => l} : NaturalSources<Shape.output_shape(plan), cursor, left, right, Elementwise.NaturalStop{retained_left, retained_right, error}> -> Totality.SameSource(left, retained_left)}(sources),
        {\{NaturalSources: l => r => r} : NaturalSources<Shape.output_shape(plan), cursor, left, right, Elementwise.NaturalStop{retained_left, retained_right, error}> -> Totality.SameSource(right, retained_right)}(sources)}
    case Elementwise.NaturalContinue{state, index, exact}:
      retention_reify(left, right, Elementwise.loop(fuel, operation, plan, 1n+cursor, state),
        rec(state)(
          {\{NaturalSources: l => r => l} : NaturalSources<Shape.output_shape(plan), cursor, left, right, Elementwise.NaturalContinue{state, index, exact}> ->
            Totality.SameSource(left, step_left_owner(Shape.output_shape(plan), Elementwise.Continue{state}))}(sources))(
          {\{NaturalSources: l => r => r} : NaturalSources<Shape.output_shape(plan), cursor, left, right, Elementwise.NaturalContinue{state, index, exact}> ->
            Totality.SameSource(right, step_right_owner(Shape.output_shape(plan), Elementwise.Continue{state}))}(sources)))

def loop_retains(fuel: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +cursor: Nat,
  -left: Core.Dynamic, -right: Core.Dynamic, state: Elementwise.State<Shape.output_shape(plan)>,
  -left_same: Totality.SameSource(left, step_left_owner(Shape.output_shape(plan), Elementwise.Continue{state})),
  -right_same: Totality.SameSource(right, step_right_owner(Shape.output_shape(plan), Elementwise.Continue{state}))) ->
  Totality.Retention<left, right, Elementwise.loop(fuel, operation, plan, cursor, state)>:
  match fuel:
    case 0n:
      Elementwise.State{retained_left, retained_right, output} = state
      Totality.Retention{left_same, right_same}
    case 1n+remaining:
      loop_finish_retains(remaining, operation, plan, cursor, left, right, Elementwise.natural_step(operation, plan, cursor, state),
        {\{NaturalSources: l => r => NaturalSources{
          property => Equal.trans(Totality.Inspection,
            Totality.inspect(property, natural_left_owner(Shape.output_shape(plan), cursor, Elementwise.natural_step(operation, plan, cursor, state))),
            Totality.inspect(property, step_left_owner(Shape.output_shape(plan), Elementwise.Continue{state})), Totality.inspect(property, left), l(property), left_same(property)),
          property => Equal.trans(Totality.Inspection,
            Totality.inspect(property, natural_right_owner(Shape.output_shape(plan), cursor, Elementwise.natural_step(operation, plan, cursor, state))),
            Totality.inspect(property, step_right_owner(Shape.output_shape(plan), Elementwise.Continue{state})), Totality.inspect(property, right), r(property), right_same(property))}} :
          NaturalSources<Shape.output_shape(plan), cursor, step_left_owner(Shape.output_shape(plan), Elementwise.Continue{state}), step_right_owner(Shape.output_shape(plan), Elementwise.Continue{state}),
            Elementwise.natural_step(operation, plan, cursor, state)> ->
          NaturalSources<Shape.output_shape(plan), cursor, left, right, Elementwise.natural_step(operation, plan, cursor, state)>}(
            natural_decided_retains(operation, plan, cursor, U32.from_nat(cursor), Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)), state)),
        state => l => r => loop_retains(remaining, operation, plan, 1n+cursor, left, right, state, l, r))

def allocated_retains(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, left: Core.Dynamic, right: Core.Dynamic,
  result: Result<&2, &1, Core.Error, Core.Tensor<Shape.output_shape(plan)>>) ->
  Totality.Retention<left, right, Elementwise.allocated(operation, plan, left, right, result)>:
  match result:
    case Fail{error}:
      Totality.Retention{property => {==}, property => {==}}
    case Done{output}:
      loop_retains(Shape.elements(Shape.output_shape(plan)), operation, plan, 0n, left, right, Elementwise.State{left, right, output}, property => {==}, property => {==})

def output_retains(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat},
  -left: Core.Dynamic, -right: Core.Dynamic, taken_left: Core.Dynamic & Array<F32>, taken_right: Core.Dynamic & Array<F32>,
  -left_taken: Parallel.Taken<left, taken_left>, -right_taken: Parallel.Taken<right, taken_right>) ->
  Totality.Retention<left, right, Elementwise.parallel_output(operation, plan, depth, length, valid, exact, taken_left, taken_right)>:
  (kept_left, la) = taken_left
  (kept_right, ra) = taken_right
  Totality.Retention{
    {\{Parallel.Taken: depth => balanced => reads => same => same} : Parallel.Taken<left, (kept_left, la)> -> Totality.SameSource(left, kept_left)}(left_taken),
    {\{Parallel.Taken: depth => balanced => reads => same => same} : Parallel.Taken<right, (kept_right, ra)> -> Totality.SameSource(right, kept_right)}(right_taken)}

def parallel_allocated_retains(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, left: Core.Dynamic, right: Core.Dynamic,
  result: Result<&2, &1, Core.Error, Core.Tensor<Shape.output_shape(plan)>>) ->
  Totality.Retention<left, right, Elementwise.parallel_allocated(operation, plan, left, right, result)>:
  match result:
    case Fail{error}:
      Totality.Retention{property => {==}, property => {==}}
    case Done{Core.TTensor{storage, exact}}:
      match storage:
        case Storage.Empty{}:
          Totality.Retention{property => {==}, property => {==}}
        case Storage.Owned{+depth, +length, array, balanced, valid}:
          output_retains(operation, plan, depth, length, valid, exact, left, right, Elementwise.take(left), Elementwise.take(right),
            Parallel.taken(left), Parallel.taken(right))

def planned_retains(+operation: Elementwise.Operation, left: Core.Dynamic, right: Core.Dynamic,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) ->
  Totality.Retention<left, right, Elementwise.parallel_planned(operation, left, right, result)>:
  match result:
    case Fail{error}:
      Totality.Retention{property => {==}, property => {==}}
    case Done{+plan}:
      parallel_allocated_retains(operation, plan, left, right, Core.new(Shape.output_shape(plan), 0.0))

def Laws.binary_retains_inputs(operation, left, right):
  Core.Packed{+left_shape, left_tensor} = left
  Core.Packed{+right_shape, right_tensor} = right
  planned_retains(operation, Core.Packed{left_shape, left_tensor}, Core.Packed{right_shape, right_tensor},
    Shape.broadcast_checked(left_shape, right_shape))

def dimensions_reify(-output: List<&2, U32>, -pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  -model: Totality.DimensionsModel<output, pair>) -> Totality.DimensionsModel<output, pair>:
  Totality.DimensionsModel{{\{Totality.DimensionsModel: exact => exact} : Totality.DimensionsModel<output, pair> -> Totality.OutputDimensions(output, pair)}(model)}

def loop_finish_dimensions(+fuel: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +cursor: Nat,
  result: Elementwise.NaturalStep<Shape.output_shape(plan), cursor>,
  -rec: @state: Elementwise.State<Shape.output_shape(plan)> ->
    Totality.DimensionsModel<Shape.dimensions(Shape.output_shape(plan)), Elementwise.loop(fuel, operation, plan, 1n+cursor, state)>) ->
  Totality.DimensionsModel<Shape.dimensions(Shape.output_shape(plan)), Elementwise.finish_step(Shape.output_shape(plan), cursor,
    state => Elementwise.loop(fuel, operation, plan, 1n+cursor, state), result)>:
  match result:
    case Elementwise.NaturalStop{left, right, error}:
      Totality.DimensionsModel{Unit{}}
    case Elementwise.NaturalContinue{state, index, exact}:
      dimensions_reify(Shape.dimensions(Shape.output_shape(plan)), Elementwise.loop(fuel, operation, plan, 1n+cursor, state), rec(state))

def loop_dimensions(fuel: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +cursor: Nat,
  state: Elementwise.State<Shape.output_shape(plan)>) ->
  Totality.DimensionsModel<Shape.dimensions(Shape.output_shape(plan)), Elementwise.loop(fuel, operation, plan, cursor, state)>:
  match fuel:
    case 0n:
      Elementwise.State{left, right, output} = state
      Totality.DimensionsModel{{==}}
    case 1n+remaining:
      loop_finish_dimensions(remaining, operation, plan, cursor, Elementwise.natural_step(operation, plan, cursor, state),
        state => loop_dimensions(remaining, operation, plan, 1n+cursor, state))

def allocated_dimensions(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, left: Core.Dynamic, right: Core.Dynamic,
  result: Result<&2, &1, Core.Error, Core.Tensor<Shape.output_shape(plan)>>) ->
  Totality.DimensionsModel<Shape.dimensions(Shape.output_shape(plan)), Elementwise.allocated(operation, plan, left, right, result)>:
  match result:
    case Fail{error}:
      Totality.DimensionsModel{Unit{}}
    case Done{output}:
      loop_dimensions(Shape.elements(Shape.output_shape(plan)), operation, plan, 0n, Elementwise.State{left, right, output})

def words_naturals(+dimensions: List<&2, U32>) -> {Shape.words(Shape.naturals(dimensions)) == dimensions : List<&2, U32>}:
  match dimensions:
    case Nil{}:
      {==}
    case +head <> +tail:
      %words_naturals(tail) : {U32.from_nat(U32.to_nat(head)) <> Shape.words(Shape.naturals(tail)) == head <> _ : List<&2, U32>}
      %Conversion.word_natural_exact(head) : {U32.from_nat(U32.to_nat(head)) <> Shape.words(Shape.naturals(tail)) == _ <> Shape.words(Shape.naturals(tail)) : List<&2, U32>}
      {==}

def natural_dimensions_equal(+left: List<&2, U32>, +right: List<&2, U32>,
  equal: {Shape.naturals(left) == Shape.naturals(right) : List<&2, Nat>}) -> {left == right : List<&2, U32>}:
  Equal.trans(List<&2, U32>, left, Shape.words(Shape.naturals(left)), right,
    Equal.sym(List<&2, U32>, Shape.words(Shape.naturals(left)), left, words_naturals(left)),
    Equal.trans(List<&2, U32>, Shape.words(Shape.naturals(left)), Shape.words(Shape.naturals(right)), right,
      Equal.cong(List<&2, Nat>, List<&2, U32>, Shape.words, Shape.naturals(left), Shape.naturals(right), equal), words_naturals(right)))

def planned_output_dimensions(+left: Shape.Shape, +right: Shape.Shape, +plan: Shape.BroadcastPlan, +output: List<&2, U32>,
  +alignment: Shape.Broadcast<Shape.extents(left), Shape.extents(right), Shape.naturals(output)>,
  +sources: {(Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(Shape.right_shape(plan))) ==
    (Shape.dimensions(left), Shape.dimensions(right)) : List<&2, U32> & List<&2, U32>}) ->
  {Shape.dimensions(Shape.output_shape(plan)) == output : List<&2, U32>}:
  natural_dimensions_equal(Shape.dimensions(Shape.output_shape(plan)), output,
    Unique.broadcast_unique(Shape.extents(left), Shape.extents(right), Shape.extents(Shape.output_shape(plan)), Shape.naturals(output),
      %Equal.cong(List<&2, U32> & List<&2, U32>, List<&2, Nat>, pair => Shape.naturals(Pair.fst(List<&2, U32>, List<&2, U32>, pair)),
        (Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(Shape.right_shape(plan))), (Shape.dimensions(left), Shape.dimensions(right)), sources) :
        Shape.Broadcast<_, Shape.extents(right), Shape.extents(Shape.output_shape(plan))>
      %Equal.cong(List<&2, U32> & List<&2, U32>, List<&2, Nat>, pair => Shape.naturals(Pair.snd(List<&2, U32>, List<&2, U32>, pair)),
        (Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(Shape.right_shape(plan))), (Shape.dimensions(left), Shape.dimensions(right)), sources) :
        Shape.Broadcast<Shape.extents(Shape.left_shape(plan)), _, Shape.extents(Shape.output_shape(plan))>
      Shape.alignment(plan), alignment))

def output_dimensions(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat},
  taken_left: Core.Dynamic & Array<F32>, taken_right: Core.Dynamic & Array<F32>) ->
  Totality.DimensionsModel<Shape.dimensions(Shape.output_shape(plan)), Elementwise.parallel_output(operation, plan, depth, length, valid, exact, taken_left, taken_right)>:
  (kept_left, la) = taken_left
  (kept_right, ra) = taken_right
  Totality.DimensionsModel{{==}}

def parallel_allocated_dimensions(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, left: Core.Dynamic, right: Core.Dynamic,
  result: Result<&2, &1, Core.Error, Core.Tensor<Shape.output_shape(plan)>>) ->
  Totality.DimensionsModel<Shape.dimensions(Shape.output_shape(plan)), Elementwise.parallel_allocated(operation, plan, left, right, result)>:
  match result:
    case Fail{error}:
      Totality.DimensionsModel{Unit{}}
    case Done{Core.TTensor{storage, exact}}:
      match storage:
        case Storage.Empty{}:
          Totality.DimensionsModel{{==}}
        case Storage.Owned{+depth, +length, array, balanced, valid}:
          output_dimensions(operation, plan, depth, length, valid, exact, Elementwise.take(left), Elementwise.take(right))

def planned_dimensions(+operation: Elementwise.Operation, +left_shape: Shape.Shape, +right_shape: Shape.Shape, +output: List<&2, U32>,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>, left: Core.Tensor<left_shape>, right: Core.Tensor<right_shape>,
  -sources: {ShapeLaws.planned_dimensions(left_shape, right_shape, result) == (Shape.dimensions(left_shape), Shape.dimensions(right_shape)) : List<&2, U32> & List<&2, U32>},
  -alignment: Shape.Broadcast<Shape.extents(left_shape), Shape.extents(right_shape), Shape.naturals(output)>) ->
  Totality.DimensionsModel<output, Elementwise.parallel_planned(operation, Core.Packed{left_shape, left}, Core.Packed{right_shape, right}, result)>:
  match result:
    case Fail{error}:
      Totality.DimensionsModel{Unit{}}
    case Done{+plan}:
      dimensions_reify(output, Elementwise.parallel_allocated(operation, plan, Core.Packed{left_shape, left}, Core.Packed{right_shape, right}, Core.new(Shape.output_shape(plan), 0.0)),
        %planned_output_dimensions(left_shape, right_shape, plan, output, alignment, sources) :
          Totality.DimensionsModel<_, Elementwise.parallel_allocated(operation, plan, Core.Packed{left_shape, left}, Core.Packed{right_shape, right}, Core.new(Shape.output_shape(plan), 0.0))>
        parallel_allocated_dimensions(operation, plan, Core.Packed{left_shape, left}, Core.Packed{right_shape, right}, Core.new(Shape.output_shape(plan), 0.0)))

def Laws.binary_successful_output_dimensions(operation, left, right, output, alignment):
  Core.Packed{+left_shape, left_tensor} = left
  Core.Packed{+right_shape, right_tensor} = right
  planned_dimensions(operation, left_shape, right_shape, output, Shape.broadcast_checked(left_shape, right_shape), left_tensor, right_tensor,
    PlanProof.broadcast_checked_plan(left_shape, right_shape), alignment)

def dynamic_read_succeeded(pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> Bool:
  (owner, result) = pair
  Result.is_done(&2, &2, Core.Error, F32, result)

def dynamic_read_status(+shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) ->
  {dynamic_read_succeeded(Core.read_dynamic_result(shape, pair)) == TransformTotality.read_succeeded(shape, pair) : Bool}:
  (owner, result) = pair
  {==}

type DynamicReadSuccess<-owner: Core.Dynamic, -index: U32> is Data:
  DynamicReadSuccess{-successful: {dynamic_read_succeeded(Core.read_dynamic(owner, index)) == True{} : Bool}}

def dynamic_read_success(original: Core.Dynamic, +index: U32,
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Totality.dimensions(original))) == True{} : Bool}) -> DynamicReadSuccess<original, index>:
  Core.Packed{+shape, tensor} = original
  DynamicReadSuccess{Equal.trans(Bool, dynamic_read_succeeded(Core.read_dynamic(Core.Packed{shape, tensor}, index)),
    TransformTotality.read_succeeded(shape, Core.read_flat(shape, tensor, index)), True{},
    dynamic_read_status(shape, Core.read_flat(shape, tensor, index)),
    {\{TransformTotality.ReadSuccess: successful => successful} : TransformTotality.ReadSuccess<shape, tensor, index> ->
      {TransformTotality.read_succeeded(shape, Core.read_flat(shape, tensor, index)) == True{} : Bool}}(
        TransformProof.read_success_status(shape, tensor, index, TransformProof.read_success(shape, tensor, index, inside))))}

def flat_index_inside(+shape: Shape.Shape, +index: Shape.FlatIndex<shape>) ->
  {Nat.is_lt(U32.to_nat(Shape.index_value(shape, index)), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}:
  Shape.FIndex{+value, +bound} = index
  %Shape.elements_exact(shape) :
    {Nat.is_lt(U32.to_nat(U32.from_nat(value)), _) == True{} : Bool}
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(value)), value,
    TransformProof.cursor_word_exact(shape, value, Conversion.lt_le(value, Shape.elements(shape), Equal.sym(Bool, True{}, Nat.is_lt(value, Shape.elements(shape)), bound)))) :
    {Nat.is_lt(_, Shape.elements(shape)) == True{} : Bool}
  Equal.sym(Bool, True{}, Nat.is_lt(value, Shape.elements(shape)), bound)

type IndicesDone<-plan: Shape.BroadcastPlan, -result: Result<&2, &1, Core.Error, U32 & U32>> is Data:
  IndicesDone{-left: U32, -right: U32,
    -exact: {Done{(left, right)} == result : Result<&2, &1, Core.Error, U32 & U32>},
    -left_bound: {Nat.is_lt(U32.to_nat(left), Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan)))) == True{} : Bool},
    -right_bound: {Nat.is_lt(U32.to_nat(right), Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))) == True{} : Bool}}

def checked_indices_success(+plan: Shape.BroadcastPlan, +index: U32,
  decision: Shape.BoolCase<Nat.is_lt(U32.to_nat(index), Shape.elements(Shape.output_shape(plan)))>,
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.elements(Shape.output_shape(plan))) == True{} : Bool}) ->
  IndicesDone<plan, Elementwise.flat_input_indices(plan, Shape.flat_checked.finish(Shape.output_shape(plan), U32.to_nat(index), decision))>:
  match decision:
    case Shape.BFalse{equal}:
      IndicesDone{0, 0,
        Empty.absurd({Done{(0, 0)} == Fail{Core.ShapeError{Shape.FlatIndexOutOfBounds{}}} : Result<&2, &1, Core.Error, U32 & U32>},
          Arith.clash(Equal.trans(Bool, False{}, Nat.is_lt(U32.to_nat(index), Shape.elements(Shape.output_shape(plan))), True{}, equal, inside))),
        Empty.absurd({Nat.is_lt(0n, Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan)))) == True{} : Bool},
          Arith.clash(Equal.trans(Bool, False{}, Nat.is_lt(U32.to_nat(index), Shape.elements(Shape.output_shape(plan))), True{}, equal, inside))),
        Empty.absurd({Nat.is_lt(0n, Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))) == True{} : Bool},
          Arith.clash(Equal.trans(Bool, False{}, Nat.is_lt(U32.to_nat(index), Shape.elements(Shape.output_shape(plan))), True{}, equal, inside)))}
    case Shape.BTrue{equal}:
      +flat = {Shape.FIndex{U32.to_nat(index), equal} : Shape.FlatIndex<Shape.output_shape(plan)>}
      +left = Flat.project_left_index(plan, Shape.index_map(plan), flat)
      +right = Flat.project_right_index(plan, Shape.index_map(plan), flat)
      IndicesDone{Shape.index_value(Shape.left_shape(plan), left), Shape.index_value(Shape.right_shape(plan), right), {==},
        flat_index_inside(Shape.left_shape(plan), left), flat_index_inside(Shape.right_shape(plan), right)}

def indices_success(+plan: Shape.BroadcastPlan, +index: U32,
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.elements(Shape.output_shape(plan))) == True{} : Bool}) ->
  IndicesDone<plan, Elementwise.input_indices(plan, index)>:
  checked_indices_success(plan, index, Shape.bool_case(Nat.is_lt(U32.to_nat(index), Shape.elements(Shape.output_shape(plan)))), inside)

def step_succeeded(-shape: Shape.Shape, result: Elementwise.Step<shape>) -> Bool:
  match result:
    case Elementwise.Stop{left, right, error}:
      False{}
    case Elementwise.Continue{state}:
      True{}

type StepSuccess<-shape: Shape.Shape, -result: Elementwise.Step<shape>> is Data:
  StepSuccess{-successful: {step_succeeded(shape, result) == True{} : Bool}}

type FittingSources<-plan: Shape.BroadcastPlan, -left: Core.Dynamic, -right: Core.Dynamic> is Data:
  FittingSources{-left: {Totality.dimensions(left) == Shape.dimensions(Shape.left_shape(plan)) : List<&2, U32>},
    -right: {Totality.dimensions(right) == Shape.dimensions(Shape.right_shape(plan)) : List<&2, U32>}}

def step_write_success(-shape: Shape.Shape, left: Core.Dynamic, right: Core.Dynamic,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>,
  -successful: {TransformTotality.write_succeeded(shape, pair) == True{} : Bool}) -> StepSuccess<shape, Elementwise.step_write(shape, left, right, pair)>:
  (output, result) = pair
  match result:
    case Fail{error}:
      StepSuccess{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{unit}:
      StepSuccess{{==}}

def step_right_success(+operation: Elementwise.Operation, +shape: Shape.Shape, +index: U32,
  left: Core.Dynamic, output: Core.Tensor<shape>, left_value: F32, pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>,
  -successful: {dynamic_read_succeeded(pair) == True{} : Bool},
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  StepSuccess<shape, Elementwise.step_right(operation, shape, index, left, output, left_value, pair)>:
  (right, result) = pair
  match result:
    case Fail{error}:
      StepSuccess{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{+value}:
      step_write_success(shape, left, right, Core.write_flat(shape, output, index, Elementwise.scalar(operation, left_value, value)),
        {\{TransformTotality.WriteSuccess: successful => successful} :
          TransformTotality.WriteSuccess<shape, output, index, Elementwise.scalar(operation, left_value, value)> ->
          {TransformTotality.write_succeeded(shape, Core.write_flat(shape, output, index, Elementwise.scalar(operation, left_value, value))) == True{} : Bool}}(
            TransformProof.tensor_write_success(shape, output, index, Elementwise.scalar(operation, left_value, value), inside)))

def step_left_success(+operation: Elementwise.Operation, +shape: Shape.Shape, +index: U32,
  right: Core.Dynamic, output: Core.Tensor<shape>, +right_index: U32, pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>,
  -successful: {dynamic_read_succeeded(pair) == True{} : Bool},
  -right_inside: {Nat.is_lt(U32.to_nat(right_index), Shape.logical_numel(Totality.dimensions(right))) == True{} : Bool},
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  StepSuccess<shape, Elementwise.step_left(operation, shape, index, right, output, right_index, pair)>:
  (left, result) = pair
  match result:
    case Fail{error}:
      StepSuccess{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{value}:
      step_right_success(operation, shape, index, left, output, value, Core.read_dynamic(right, right_index),
        {\{DynamicReadSuccess: successful => successful} : DynamicReadSuccess<right, right_index> ->
          {dynamic_read_succeeded(Core.read_dynamic(right, right_index)) == True{} : Bool}}(dynamic_read_success(right, right_index, right_inside)), inside)

def actual_step_success(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +index: U32,
  state: Elementwise.State<Shape.output_shape(plan)>,
  -fits: FittingSources<plan, step_left_owner(Shape.output_shape(plan), Elementwise.Continue{state}), step_right_owner(Shape.output_shape(plan), Elementwise.Continue{state})>,
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.elements(Shape.output_shape(plan))) == True{} : Bool}) ->
  StepSuccess<Shape.output_shape(plan), Elementwise.step(operation, plan, index, state)>:
  Elementwise.State{left, right, output} = state
  StepSuccess{{\{IndicesDone: li => ri => exact => lb => rb =>
    %exact : {step_succeeded(Shape.output_shape(plan), Elementwise.step_indices(operation, Shape.output_shape(plan), index, Elementwise.State{left, right, output}, _)) == True{} : Bool}
    {\{StepSuccess: successful => successful} :
      StepSuccess<Shape.output_shape(plan), Elementwise.step_left(operation, Shape.output_shape(plan), index, right, output, ri, Core.read_dynamic(left, li))> ->
      {step_succeeded(Shape.output_shape(plan), Elementwise.step_left(operation, Shape.output_shape(plan), index, right, output, ri, Core.read_dynamic(left, li))) == True{} : Bool}}(
        step_left_success(operation, Shape.output_shape(plan), index, right, output, ri, Core.read_dynamic(left, li),
          {\{DynamicReadSuccess: successful => successful} : DynamicReadSuccess<left, li> -> {dynamic_read_succeeded(Core.read_dynamic(left, li)) == True{} : Bool}}(
            dynamic_read_success(left, li,
              %Equal.sym(List<&2, U32>, Totality.dimensions(left), Shape.dimensions(Shape.left_shape(plan)),
                {\{FittingSources: l => r => l} : FittingSources<plan, left, right> -> {Totality.dimensions(left) == Shape.dimensions(Shape.left_shape(plan)) : List<&2, U32>}}(fits)) :
                {Nat.is_lt(U32.to_nat(li), Shape.logical_numel(_)) == True{} : Bool}
              lb)),
          %Equal.sym(List<&2, U32>, Totality.dimensions(right), Shape.dimensions(Shape.right_shape(plan)),
            {\{FittingSources: l => r => r} : FittingSources<plan, left, right> -> {Totality.dimensions(right) == Shape.dimensions(Shape.right_shape(plan)) : List<&2, U32>}}(fits)) :
            {Nat.is_lt(U32.to_nat(ri), Shape.logical_numel(_)) == True{} : Bool}
          rb,
          %Shape.elements_exact(Shape.output_shape(plan)) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
          inside))} : IndicesDone<plan, Elementwise.input_indices(plan, index)> ->
      {step_succeeded(Shape.output_shape(plan), Elementwise.step(operation, plan, index, Elementwise.State{left, right, output})) == True{} : Bool}}(
        indices_success(plan, index, inside))}

def natural_succeeded(-shape: Shape.Shape, -cursor: Nat, result: Elementwise.NaturalStep<shape, cursor>) -> Bool:
  match result:
    case Elementwise.NaturalStop{left, right, error}:
      False{}
    case Elementwise.NaturalContinue{state, index, exact}:
      True{}

type NaturalSuccess<-shape: Shape.Shape, -cursor: Nat, -result: Elementwise.NaturalStep<shape, cursor>> is Data:
  NaturalSuccess{-successful: {natural_succeeded(shape, cursor, result) == True{} : Bool}}

def natural_result_success(-shape: Shape.Shape, -cursor: Nat, -index: U32, -exact: {U32.to_nat(index) == cursor : Nat},
  result: Elementwise.Step<shape>, -successful: {step_succeeded(shape, result) == True{} : Bool}) ->
  NaturalSuccess<shape, cursor, Elementwise.natural_step_result(shape, cursor, index, exact, result)>:
  match result:
    case Elementwise.Stop{left, right, error}:
      NaturalSuccess{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Elementwise.Continue{state}:
      NaturalSuccess{{==}}

def natural_decided_success(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +cursor: Nat, +index: U32,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>, state: Elementwise.State<Shape.output_shape(plan)>,
  -fits: FittingSources<plan, step_left_owner(Shape.output_shape(plan), Elementwise.Continue{state}), step_right_owner(Shape.output_shape(plan), Elementwise.Continue{state})>,
  -exact: {U32.to_nat(index) == cursor : Nat}, -inside: {Nat.is_lt(cursor, Shape.elements(Shape.output_shape(plan))) == True{} : Bool}) ->
  NaturalSuccess<Shape.output_shape(plan), cursor, Elementwise.natural_step_decided(operation, plan, cursor, index, state, decision)>:
  match decision:
    case Shape.BFalse{equal}:
      Elementwise.State{left, right, output} = state
      NaturalSuccess{Empty.absurd({False{} == True{} : Bool}, Nats.contradiction(Nat.is_eq(U32.to_nat(index), cursor), PlanProof.eq_true(U32.to_nat(index), cursor, exact), equal))}
    case Shape.BTrue{equal}:
      natural_result_success(Shape.output_shape(plan), cursor, index, Lifecycle.natural_equal(U32.to_nat(index), cursor, equal),
        Elementwise.step(operation, plan, index, state),
        {\{StepSuccess: successful => successful} : StepSuccess<Shape.output_shape(plan), Elementwise.step(operation, plan, index, state)> ->
          {step_succeeded(Shape.output_shape(plan), Elementwise.step(operation, plan, index, state)) == True{} : Bool}}(
            actual_step_success(operation, plan, index, state, fits,
              %Equal.sym(Nat, U32.to_nat(index), cursor, exact) : {Nat.is_lt(_, Shape.elements(Shape.output_shape(plan))) == True{} : Bool}
              inside)))

def natural_success(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +cursor: Nat,
  state: Elementwise.State<Shape.output_shape(plan)>,
  -fits: FittingSources<plan, step_left_owner(Shape.output_shape(plan), Elementwise.Continue{state}), step_right_owner(Shape.output_shape(plan), Elementwise.Continue{state})>,
  -inside: {Nat.is_lt(cursor, Shape.elements(Shape.output_shape(plan))) == True{} : Bool}) ->
  NaturalSuccess<Shape.output_shape(plan), cursor, Elementwise.natural_step(operation, plan, cursor, state)>:
  natural_decided_success(operation, plan, cursor, U32.from_nat(cursor), Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)), state, fits,
    TransformProof.cursor_word_exact(Shape.output_shape(plan), cursor, Conversion.lt_le(cursor, Shape.elements(Shape.output_shape(plan)), inside)), inside)

def inspection_dimensions(value: Totality.Inspection) -> List<&2, U32>:
  match value:
    case Totality.Dims{dimensions}:
      dimensions
    case Totality.Value{value}:
      Nil{}

def retained_fits(+plan: Shape.BroadcastPlan, -before_left: Core.Dynamic, -before_right: Core.Dynamic,
  -after_left: Core.Dynamic, -after_right: Core.Dynamic,
  -left_same: Totality.SameSource(before_left, after_left), -right_same: Totality.SameSource(before_right, after_right),
  -fits: FittingSources<plan, before_left, before_right>) -> FittingSources<plan, after_left, after_right>:
  FittingSources{
    Equal.trans(List<&2, U32>, Totality.dimensions(after_left), Totality.dimensions(before_left), Shape.dimensions(Shape.left_shape(plan)),
      Equal.cong(Totality.Inspection, List<&2, U32>, inspection_dimensions,
        Totality.inspect(Totality.Dimensions{}, after_left), Totality.inspect(Totality.Dimensions{}, before_left), left_same(Totality.Dimensions{})),
      {\{FittingSources: l => r => l} : FittingSources<plan, before_left, before_right> -> {Totality.dimensions(before_left) == Shape.dimensions(Shape.left_shape(plan)) : List<&2, U32>}}(fits)),
    Equal.trans(List<&2, U32>, Totality.dimensions(after_right), Totality.dimensions(before_right), Shape.dimensions(Shape.right_shape(plan)),
      Equal.cong(Totality.Inspection, List<&2, U32>, inspection_dimensions,
        Totality.inspect(Totality.Dimensions{}, after_right), Totality.inspect(Totality.Dimensions{}, before_right), right_same(Totality.Dimensions{})),
      {\{FittingSources: l => r => r} : FittingSources<plan, before_left, before_right> -> {Totality.dimensions(before_right) == Shape.dimensions(Shape.right_shape(plan)) : List<&2, U32>}}(fits))}

def success_reify(-pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  -model: Totality.Success<pair>) -> Totality.Success<pair>:
  Totality.Success{{\{Totality.Success: successful => successful} : Totality.Success<pair> -> {Totality.succeeded(pair) == True{} : Bool}}(model)}

def loop_finish_success(+fuel: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +cursor: Nat,
  -before_left: Core.Dynamic, -before_right: Core.Dynamic,
  result: Elementwise.NaturalStep<Shape.output_shape(plan), cursor>,
  -successful: {natural_succeeded(Shape.output_shape(plan), cursor, result) == True{} : Bool},
  -sources: NaturalSources<Shape.output_shape(plan), cursor, before_left, before_right, result>,
  -fits: FittingSources<plan, before_left, before_right>,
  -rec: @state: Elementwise.State<Shape.output_shape(plan)> ->
    @-next_fits: FittingSources<plan, step_left_owner(Shape.output_shape(plan), Elementwise.Continue{state}), step_right_owner(Shape.output_shape(plan), Elementwise.Continue{state})> ->
    Totality.Success<Elementwise.loop(fuel, operation, plan, 1n+cursor, state)>) ->
  Totality.Success<Elementwise.finish_step(Shape.output_shape(plan), cursor, state => Elementwise.loop(fuel, operation, plan, 1n+cursor, state), result)>:
  match result:
    case Elementwise.NaturalStop{left, right, error}:
      Totality.Success{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Elementwise.NaturalContinue{state, index, exact}:
      success_reify(Elementwise.loop(fuel, operation, plan, 1n+cursor, state),
        rec(state)(retained_fits(plan, before_left, before_right,
          step_left_owner(Shape.output_shape(plan), Elementwise.Continue{state}), step_right_owner(Shape.output_shape(plan), Elementwise.Continue{state}),
          {\{NaturalSources: l => r => l} : NaturalSources<Shape.output_shape(plan), cursor, before_left, before_right, Elementwise.NaturalContinue{state, index, exact}> ->
            Totality.SameSource(before_left, step_left_owner(Shape.output_shape(plan), Elementwise.Continue{state}))}(sources),
          {\{NaturalSources: l => r => r} : NaturalSources<Shape.output_shape(plan), cursor, before_left, before_right, Elementwise.NaturalContinue{state, index, exact}> ->
            Totality.SameSource(before_right, step_right_owner(Shape.output_shape(plan), Elementwise.Continue{state}))}(sources), fits)))

def loop_success(fuel: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +cursor: Nat,
  state: Elementwise.State<Shape.output_shape(plan)>,
  -fits: FittingSources<plan, step_left_owner(Shape.output_shape(plan), Elementwise.Continue{state}), step_right_owner(Shape.output_shape(plan), Elementwise.Continue{state})>,
  -within: {Nat.is_le(Nat.add(fuel, cursor), Shape.elements(Shape.output_shape(plan))) == True{} : Bool}) ->
  Totality.Success<Elementwise.loop(fuel, operation, plan, cursor, state)>:
  match fuel:
    case 0n:
      Elementwise.State{left, right, output} = state
      Totality.Success{{==}}
    case 1n+remaining:
      loop_finish_success(remaining, operation, plan, cursor,
        step_left_owner(Shape.output_shape(plan), Elementwise.Continue{state}), step_right_owner(Shape.output_shape(plan), Elementwise.Continue{state}),
        Elementwise.natural_step(operation, plan, cursor, state),
        {\{NaturalSuccess: successful => successful} : NaturalSuccess<Shape.output_shape(plan), cursor, Elementwise.natural_step(operation, plan, cursor, state)> ->
          {natural_succeeded(Shape.output_shape(plan), cursor, Elementwise.natural_step(operation, plan, cursor, state)) == True{} : Bool}}(
            natural_success(operation, plan, cursor, state, fits,
              Arith.lt_le_trans(cursor, 1n+Nat.add(remaining, cursor), Shape.elements(Shape.output_shape(plan)),
                Arith.lt_succ(cursor, Nat.add(remaining, cursor), TransformProof.add_cursor_le(remaining, cursor)), within))),
        natural_decided_retains(operation, plan, cursor, U32.from_nat(cursor), Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)), state), fits,
        state => next_fits => loop_success(remaining, operation, plan, 1n+cursor, state, next_fits,
          %Equal.sym(Nat, Nat.add(remaining, 1n+cursor), 1n+Nat.add(remaining, cursor), Arith.add_succ(remaining, cursor)) :
            {Nat.is_le(_, Shape.elements(Shape.output_shape(plan))) == True{} : Bool}
          within))

def allocated_success(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, left: Core.Dynamic, right: Core.Dynamic,
  result: Result<&2, &1, Core.Error, Core.Tensor<Shape.output_shape(plan)>>,
  -fits: FittingSources<plan, left, right>, -done: Allocation.TensorDone<Shape.output_shape(plan), result>) ->
  Totality.Success<Elementwise.allocated(operation, plan, left, right, result)>:
  match result:
    case Fail{error}:
      Totality.Success{Empty.absurd({False{} == True{} : Bool},
        {\{Allocation.TensorDone: tensor => same => Arith.clash(Equal.cong(Result<&2, &1, Core.Error, Core.Tensor<Shape.output_shape(plan)>>, Bool,
          result => Result.is_done(&2, &1, Core.Error, Core.Tensor<Shape.output_shape(plan)>, result), Fail{error}, Done{tensor},
          Equal.sym(Result<&2, &1, Core.Error, Core.Tensor<Shape.output_shape(plan)>>, Done{tensor}, Fail{error}, same)))} :
          Allocation.TensorDone<Shape.output_shape(plan), Fail{error}> -> Empty}(done))}
    case Done{output}:
      loop_success(Shape.elements(Shape.output_shape(plan)), operation, plan, 0n, Elementwise.State{left, right, output}, fits,
        %Equal.sym(Nat, Nat.add(Shape.elements(Shape.output_shape(plan)), 0n), Shape.elements(Shape.output_shape(plan)), TransformProof.add_zero_right(Shape.elements(Shape.output_shape(plan)))) :
          {Nat.is_le(_, Shape.elements(Shape.output_shape(plan))) == True{} : Bool}
        Conversion.le_reflexive(Shape.elements(Shape.output_shape(plan))))

def output_success(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat},
  taken_left: Core.Dynamic & Array<F32>, taken_right: Core.Dynamic & Array<F32>) ->
  Totality.Success<Elementwise.parallel_output(operation, plan, depth, length, valid, exact, taken_left, taken_right)>:
  (kept_left, la) = taken_left
  (kept_right, ra) = taken_right
  Totality.Success{{==}}

def parallel_allocated_success(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, left: Core.Dynamic, right: Core.Dynamic,
  result: Result<&2, &1, Core.Error, Core.Tensor<Shape.output_shape(plan)>>, -done: Allocation.TensorDone<Shape.output_shape(plan), result>) ->
  Totality.Success<Elementwise.parallel_allocated(operation, plan, left, right, result)>:
  match result:
    case Fail{error}:
      Totality.Success{Empty.absurd({False{} == True{} : Bool},
        {\{Allocation.TensorDone: tensor => same => Arith.clash(Equal.cong(Result<&2, &1, Core.Error, Core.Tensor<Shape.output_shape(plan)>>, Bool,
          result => Result.is_done(&2, &1, Core.Error, Core.Tensor<Shape.output_shape(plan)>, result), Fail{error}, Done{tensor},
          Equal.sym(Result<&2, &1, Core.Error, Core.Tensor<Shape.output_shape(plan)>>, Done{tensor}, Fail{error}, same)))} :
          Allocation.TensorDone<Shape.output_shape(plan), Fail{error}> -> Empty}(done))}
    case Done{Core.TTensor{storage, exact}}:
      match storage:
        case Storage.Empty{}:
          Totality.Success{{==}}
        case Storage.Owned{+depth, +length, array, balanced, valid}:
          output_success(operation, plan, depth, length, valid, exact, Elementwise.take(left), Elementwise.take(right))

def planned_success(+operation: Elementwise.Operation, +left_shape: Shape.Shape, +right_shape: Shape.Shape,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>, left: Core.Tensor<left_shape>, right: Core.Tensor<right_shape>,
  -built: {ShapeLaws.broadcast_built(result) == True{} : Bool}) ->
  Totality.Success<Elementwise.parallel_planned(operation, Core.Packed{left_shape, left}, Core.Packed{right_shape, right}, result)>:
  match result:
    case Fail{error}:
      Totality.Success{Empty.absurd({False{} == True{} : Bool}, Arith.clash(built))}
    case Done{+plan}:
      parallel_allocated_success(operation, plan, Core.Packed{left_shape, left}, Core.Packed{right_shape, right}, Core.new(Shape.output_shape(plan), 0.0),
        AllocationProof.core_new_done(Shape.output_shape(plan), 0.0))

def Laws.binary_valid_success(operation, left, right, output, compatible):
  Core.Packed{+left_shape, left_tensor} = left
  Core.Packed{+right_shape, right_tensor} = right
  Totality.Compatible{+alignment, +rank, +capacity} = compatible
  planned_success(operation, left_shape, right_shape, Shape.broadcast_checked(left_shape, right_shape), left_tensor, right_tensor,
    PlanProof.broadcast_checked_complete(left_shape, right_shape, output, alignment, rank, capacity))
