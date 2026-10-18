import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./elementwise.bend as Elementwise
import ./projection_LAWS.bend as Projection

def IndependentMapping(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, -left: Core.Dynamic, -right: Core.Dynamic,
  result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Type:
  (inputs, outcome) = result
  match outcome:
    case Fail{error}:
      Unit
    case Done{output}:
      @+index: U32 -> @inside: Shape.LT(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) ->
        {Core.observe_dynamic(output, index) == Projection.specified_expected(operation, left, right, Projection.offset_spec(plan, U32.to_nat(index))) : Maybe<&2, F32>}

type IndependentModel<-operation: Elementwise.Operation, -plan: Shape.BroadcastPlan, -left: Core.Dynamic, -right: Core.Dynamic,
  -result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>> is Data:
  IndependentModel{-mapping: IndependentMapping(operation, plan, left, right, result)}

def IndependentPlanned(+operation: Elementwise.Operation, -left: Core.Dynamic, -right: Core.Dynamic,
  result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  plan: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) -> Type:
  match plan:
    case Fail{error}:
      Unit
    case Done{+plan}:
      IndependentMapping(operation, plan, left, right, result)

type IndependentBinaryModel<-operation: Elementwise.Operation, -left: Core.Dynamic, -right: Core.Dynamic,
  -result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>> is Data:
  IndependentBinaryModel{-mapping: IndependentPlanned(operation, left, right, result, Elementwise.plan_for(left, right))}

law scalar_add:
  for +left: F32
  for +right: F32
  {Elementwise.scalar(Elementwise.Add{}, left, right) == F32.add(left, right) : F32}

law scalar_sub:
  for +left: F32
  for +right: F32
  {Elementwise.scalar(Elementwise.Sub{}, left, right) == F32.sub(left, right) : F32}

law scalar_mul:
  for +left: F32
  for +right: F32
  {Elementwise.scalar(Elementwise.Mul{}, left, right) == F32.mul(left, right) : F32}

law indexed_length:
  for +shape: Shape.Shape
  for tensor: Core.Tensor<shape>
  Core.LengthValid<shape, tensor>

law successful_construction_dimensions:
  for +dimensions: List<&2, U32>
  for values: List<&2, F32>
  Core.ConstructionDimensions(dimensions, Core.construct(dimensions, values))

law successful_value_count:
  for +shape: Shape.Shape
  for +values: List<&2, F32>
  Core.ValuesCount(shape, values, Core.from_values(shape, values))

law checked_reader_pair:
  for +shape: Shape.Shape
  for original: Core.Tensor<shape>
  for +index: U32
  Core.Read<shape, original, index>

law reader_certificate_value:
  for +shape: Shape.Shape
  for original: Core.Tensor<shape>
  for +index: U32
  for certificate: Core.Read<shape, original, index>
  {Core.read_value(shape, Core.read_flat(shape, original, index)) ==
    Core.certificate_value(shape, original, index, certificate) : Maybe<&2, F32>}

law checked_reader_source:
  for +shape: Shape.Shape
  for original: Core.Tensor<shape>
  for +index: U32
  Core.SourceRead<shape, original, index>

law checked_dynamic_reader_source:
  for original: Core.Dynamic
  for +index: U32
  Core.DynamicSourceRead<original, index>

law checked_reader_retains_source:
  for +shape: Shape.Shape
  for original: Core.Tensor<shape>
  for +index: U32
  Core.SourceReadKeeps<shape, original, index>

law checked_dynamic_reader_retains_source:
  for original: Core.Dynamic
  for +index: U32
  Core.DynamicSourceReadKeeps<original, index>

law primitive_read_retains_elements:
  for +depth: Nat
  for array: Array<F32>
  for +index: U32
  for +other: U32
  for balanced: Model.Balanced(F32, depth, array)
  {Model.observe(F32, depth, Refinement.first(F32, Array.get(F32, array, index)), Refinement.route(depth, other)) ==
    Model.observe(F32, depth, array, Refinement.route(depth, other)) : Maybe<&2, F32>}

law storage_read_retains_length:
  for storage: Storage.Storage
  for +index: U32
  {Lifecycle.retained_read(Storage.get(storage, index)) == Lifecycle.logical_length(storage) : Nat}

law storage_write_retains_length:
  for storage: Storage.Storage
  for +index: U32
  for +value: F32
  {Lifecycle.retained_write(Storage.set(storage, index, value)) == Lifecycle.logical_length(storage) : Nat}

law storage_clone_first_length:
  for storage: Storage.Storage
  {Lifecycle.first_length(Storage.clone(storage)) == Lifecycle.logical_length(storage) : Nat}

law storage_clone_second_length:
  for storage: Storage.Storage
  {Lifecycle.second_length(Storage.clone(storage)) == Lifecycle.logical_length(storage) : Nat}

law primitive_clone_first_elements:
  for +depth: Nat
  for array: Array<F32>
  for +path: Model.Path(depth)
  for balanced: Model.Balanced(F32, depth, array)
  {Model.observe(F32, depth, Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, array)), path) ==
    Model.observe(F32, depth, array, path) : Maybe<&2, F32>}

law primitive_clone_second_elements:
  for +depth: Nat
  for array: Array<F32>
  for +path: Model.Path(depth)
  for balanced: Model.Balanced(F32, depth, array)
  {Model.observe(F32, depth, Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, array)), path) ==
    Model.observe(F32, depth, array, path) : Maybe<&2, F32>}

law tensor_write_mapping:
  for +shape: Shape.Shape
  for +depth: Nat
  for +length: Nat
  for array: Array<F32>
  for +index: U32
  for +value: F32
  for balanced: Model.Balanced(F32, depth, array)
  for valid: {Storage.bounds(depth, length) == True{} : Bool}
  for exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  for allowed: {Nat.is_lt(U32.to_nat(index), length) == True{} : Bool}
  for matched: {Storage.index_matches(depth, length, index) == True{} : Bool}
  {Core.write_observe(shape, Core.write_flat(shape,
    Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, index, value), index) == Some{value} : Maybe<&2, F32>}

law broadcast_step_mapping:
  for +operation: Elementwise.Operation
  for +shape: Shape.Shape
  for +depth: Nat
  for +length: Nat
  for array: Array<F32>
  for +index: U32
  for left: Core.Dynamic
  for right: Core.Dynamic
  for +left_value: F32
  for +right_value: F32
  for balanced: Model.Balanced(F32, depth, array)
  for valid: {Storage.bounds(depth, length) == True{} : Bool}
  for exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  for allowed: {Nat.is_lt(U32.to_nat(index), length) == True{} : Bool}
  for matched: {Storage.index_matches(depth, length, index) == True{} : Bool}
  {Elementwise.step_observe(shape, Elementwise.step_right(operation, shape, index, left,
    Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, left_value,
    (right, Done{right_value})), index) == Some{Elementwise.scalar(operation, left_value, right_value)} : Maybe<&2, F32>}

law checked_route_separation:
  for +depth: Nat
  for +length: Nat
  for +index: U32
  for +other: U32
  for index_checked: {Storage.index_matches(depth, length, index) == True{} : Bool}
  for other_checked: {Storage.index_matches(depth, length, other) == True{} : Bool}
  for different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool}
  {Model.same(depth, Refinement.route(depth, index), Refinement.route(depth, other)) == False{} : Bool}

law tensor_write_frame:
  for +shape: Shape.Shape
  for +depth: Nat
  for +length: Nat
  for array: Array<F32>
  for +index: U32
  for +other: U32
  for +value: F32
  for balanced: Model.Balanced(F32, depth, array)
  for valid: {Storage.bounds(depth, length) == True{} : Bool}
  for exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  for allowed: {Nat.is_lt(U32.to_nat(index), length) == True{} : Bool}
  for index_checked: {Storage.index_matches(depth, length, index) == True{} : Bool}
  for other_checked: {Storage.index_matches(depth, length, other) == True{} : Bool}
  for different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool}
  {Core.write_observe(shape, Core.write_flat(shape,
    Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, index, value), other) ==
    Model.observe(F32, depth, array, Refinement.route(depth, other)) : Maybe<&2, F32>}

law checked_conversion_value:
  for +cursor: Nat
  for +index: U32
  for verdict: {True{} == Nat.is_eq(U32.to_nat(index), cursor) : Bool}
  {U32.to_nat(index) == cursor : Nat}

law natural_cursor_step:
  for +operation: Elementwise.Operation
  for +plan: Shape.BroadcastPlan
  for +cursor: Nat
  for state: Elementwise.State<Shape.output_shape(plan)>
  Elementwise.CursorValidity(Shape.output_shape(plan), cursor, Elementwise.natural_step(operation, plan, cursor, state))

law u32_natural_injective:
  for +left: U32
  for +right: U32
  for equal: {U32.to_nat(left) == U32.to_nat(right) : Nat}
  {left == right : U32}

law checked_writer_pair:
  for +shape: Shape.Shape
  for original: Core.Tensor<shape>
  for +index: U32
  for +value: F32
  Core.Write<shape, original, index, value>

law checked_writer_source:
  for +shape: Shape.Shape
  for original: Core.Tensor<shape>
  for +index: U32
  for +value: F32
  Core.CheckedUpdateModel<shape, original, index, value>

law actual_elementwise_step:
  for +operation: Elementwise.Operation
  for +plan: Shape.BroadcastPlan
  for +index: U32
  for original: Elementwise.State<Shape.output_shape(plan)>
  Elementwise.StepCertificate<operation, plan, index, original>

law successful_elementwise_loop:
  for +operation: Elementwise.Operation
  for +plan: Shape.BroadcastPlan
  for left: Core.Dynamic
  for right: Core.Dynamic
  for output: Core.Tensor<Shape.output_shape(plan)>
  Elementwise.MappingModel<operation, plan, left, right,
    Elementwise.loop(Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, 0n, Elementwise.State{left, right, output})>

law successful_elementwise_cached_loop:
  for +operation: Elementwise.Operation
  for +plan: Shape.BroadcastPlan
  for left: Core.Dynamic
  for right: Core.Dynamic
  for output: Core.Tensor<Shape.output_shape(plan)>
  Elementwise.MappingModel<operation, plan, left, right,
    Elementwise.loop(Shape.elements(Shape.output_shape(plan)), operation, plan, 0n, Elementwise.State{left, right, output})>

law successful_elementwise_allocated:
  for +operation: Elementwise.Operation
  for +plan: Shape.BroadcastPlan
  for left: Core.Dynamic
  for right: Core.Dynamic
  Elementwise.MappingModel<operation, plan, left, right,
    Elementwise.allocated(operation, plan, left, right, Core.new(Shape.output_shape(plan), 0.0))>

law successful_elementwise_binary:
  for +operation: Elementwise.Operation
  for left: Core.Dynamic
  for right: Core.Dynamic
  Elementwise.BinaryModel<operation, left, right, Elementwise.binary(operation, left, right)>

law independent_elementwise_loop:
  for +operation: Elementwise.Operation
  for +plan: Shape.BroadcastPlan
  for left: Core.Dynamic
  for right: Core.Dynamic
  for output: Core.Tensor<Shape.output_shape(plan)>
  IndependentModel<operation, plan, left, right,
    Elementwise.loop(Shape.elements(Shape.output_shape(plan)), operation, plan, 0n, Elementwise.State{left, right, output})>

law independent_elementwise_binary:
  for +operation: Elementwise.Operation
  for left: Core.Dynamic
  for right: Core.Dynamic
  IndependentBinaryModel<operation, left, right, Elementwise.binary(operation, left, right)>

law public_add_dispatch:
  for left: Core.Dynamic
  for right: Core.Dynamic
  {Elementwise.add(left, right) == Elementwise.binary(Elementwise.Add{}, left, right) :
    (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>}

law public_sub_dispatch:
  for left: Core.Dynamic
  for right: Core.Dynamic
  {Elementwise.sub(left, right) == Elementwise.binary(Elementwise.Sub{}, left, right) :
    (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>}

law public_mul_dispatch:
  for left: Core.Dynamic
  for right: Core.Dynamic
  {Elementwise.mul(left, right) == Elementwise.binary(Elementwise.Mul{}, left, right) :
    (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>}
