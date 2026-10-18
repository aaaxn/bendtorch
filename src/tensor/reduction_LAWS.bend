import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./reduction.bend as Reduction
import ./math_totality.bend as Totality

def LogicalValues(+fuel: Nat, -shape: Shape.Shape, -tensor: Core.Tensor<shape>, +cursor: Nat,
  values: List<&2, F32>) -> Data:
  match fuel values:
    case 0n Nil{}:
      Unit
    case 1n+tail value <> values:
      LogicalStep<tail, shape, tensor, cursor, value, values>
    case 0n value <> values:
      Empty
    case 1n+tail Nil{}:
      Empty

type LogicalStep<-fuel: Nat, -shape: Shape.Shape, -tensor: Core.Tensor<shape>, -cursor: Nat,
  -value: F32, -values: List<&2, F32>> is Data:
  LogicalStep{-exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
    -observed: {Core.checked_observe(shape, tensor, U32.from_nat(cursor)) == Some{value} : Maybe<&2, F32>},
    -remaining: LogicalValues(fuel, shape, tensor, 1n+cursor, values)}

type SumMeaning<-fuel: Nat, -shape: Shape.Shape, -tensor: Core.Tensor<shape>, -cursor: Nat,
  -accumulator: F32, -value: F32> is Data:
  SumMeaning{-values: List<&2, F32>, -logical: LogicalValues(fuel, shape, tensor, cursor, values),
    -fold: {value == Reduction.sequential_sum(values, accumulator) : F32}}

def LoopMeaning(-fuel: Nat, -shape: Shape.Shape, -original: Core.Tensor<shape>, -cursor: Nat, -accumulator: F32,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>) -> Data:
  match result:
    case Fail{error}:
      Unit
    case Done{pair}:
      (tensor, value) = pair
      SumMeaning<fuel, shape, original, cursor, accumulator, value>

type SourceSum<-fuel: Nat, -shape: Shape.Shape, -original: Core.Tensor<shape>, -cursor: Nat, -accumulator: F32,
  -result: Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>> is Data:
  SourceSum{-meaning: @unit: Unit -> LoopMeaning(fuel, shape, original, cursor, accumulator, result)}

law successful_loop_interpretation:
  for +fuel: Nat
  for -shape: Shape.Shape
  for tensor: Core.Tensor<shape>
  for +cursor: Nat
  for +accumulator: F32
  SourceSum<fuel, shape, tensor, cursor, accumulator, Reduction.natural_sum_loop(fuel, shape, tensor, cursor, accumulator)>

def loop_value(-shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{pair}:
      (tensor, value) = pair
      Some{value}

def retained_observe(-shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>, +other: U32) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{pair}:
      (tensor, value) = pair
      Core.checked_observe(shape, tensor, other)

def successful_observe(-shape: Shape.Shape, original: Core.Tensor<shape>,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>, +other: U32) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{pair}:
      Core.checked_observe(shape, original, other)

type Preserved<-shape: Shape.Shape, -original: Core.Tensor<shape>,
  -result: Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>> is Data:
  Preserved{-kept: @+other: U32 ->
    {retained_observe(shape, result, other) == successful_observe(shape, original, result, other) : Maybe<&2, F32>}}

type Interpretation<-actual: Maybe<&2, F32>, -expected: Maybe<&2, F32>> is Data:
  Interpretation{-proof: @unit: Unit -> {actual == expected : Maybe<&2, F32>}}

law ordered_loop_interpretation:
  for +fuel: Nat
  for -shape: Shape.Shape
  for tensor: Core.Tensor<shape>
  for +cursor: Nat
  for +accumulator: F32
  for +values: List<&2, F32>
  for logical: LogicalValues(fuel, shape, tensor, cursor, values)
  Interpretation<loop_value(shape, Reduction.natural_sum_loop(fuel, shape, tensor, cursor, accumulator)),
    Some{Reduction.sequential_sum(values, accumulator)}>

law ordered_loop_preservation:
  for +fuel: Nat
  for -shape: Shape.Shape
  for tensor: Core.Tensor<shape>
  for +cursor: Nat
  for +accumulator: F32
  Preserved<shape, tensor, Reduction.natural_sum_loop(fuel, shape, tensor, cursor, accumulator)>

law ordered_sum_interpretation:
  for +shape: Shape.Shape
  for tensor: Core.Tensor<shape>
  for +values: List<&2, F32>
  for logical: LogicalValues(Shape.logical_numel(Shape.dimensions(shape)), shape, tensor, 0n, values)
  Interpretation<Reduction.sum_observation(shape, Reduction.sum(shape, tensor)), Some{Reduction.sequential_sum(values, 0.0)}>

type SumResult<-shape: Shape.Shape, -original: Core.Tensor<shape>, -retained: Core.Tensor<shape>,
  -scalar: Core.Tensor<Reduction.scalar_shape()>> is Data:
  SumResult{-values: List<&2, F32>, -logical: LogicalValues(Shape.logical_numel(Shape.dimensions(shape)), shape, original, 0n, values),
    -meaning: {Core.observe(Reduction.scalar_shape(), scalar, 0) == Some{Reduction.sequential_sum(values, 0.0)} : Maybe<&2, F32>},
    -kept: @+index: U32 -> {Core.checked_observe(shape, retained, index) == Core.checked_observe(shape, original, index) : Maybe<&2, F32>}}

def SumResultMeaning(-shape: Shape.Shape, -original: Core.Tensor<shape>,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape> & Core.Tensor<Reduction.scalar_shape()>>) -> Data:
  match result:
    case Fail{error}:
      Unit
    case Done{pair}:
      (retained, scalar) = pair
      SumResult<shape, original, retained, scalar>

type SourceSumResult<-shape: Shape.Shape, -original: Core.Tensor<shape>,
  -result: Result<&2, &1, Core.Error, Core.Tensor<shape> & Core.Tensor<Reduction.scalar_shape()>>> is Data:
  SourceSumResult{-meaning: @unit: Unit -> SumResultMeaning(shape, original, result)}

law successful_sum_interpretation:
  for +shape: Shape.Shape
  for tensor: Core.Tensor<shape>
  SourceSumResult<shape, tensor, Reduction.sum(shape, tensor)>

type DynamicSumSuccess<-original: Core.Dynamic, -retained: Core.Dynamic, -scalar: Core.Dynamic> is Data:
  DynamicSumSuccess{-shape: Shape.Shape, -input: Core.Tensor<shape>, -owner: Core.Tensor<shape>, -output: Core.Tensor<Reduction.scalar_shape()>,
    -input_exact: {original == Core.Packed{shape, input} : Core.Dynamic}, -owner_exact: {retained == Core.Packed{shape, owner} : Core.Dynamic},
    -scalar_exact: {scalar == Core.Packed{Reduction.scalar_shape(), output} : Core.Dynamic}, -source: SumResult<shape, input, owner, output>}

def DynamicSumMeaning(-original: Core.Dynamic, result: Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic>) -> Data:
  match result:
    case Fail{error}:
      Unit
    case Done{pair}:
      (retained, scalar) = pair
      DynamicSumSuccess<original, retained, scalar>

type SourceDynamicSum<-original: Core.Dynamic, -result: Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic>> is Data:
  SourceDynamicSum{-meaning: @unit: Unit -> DynamicSumMeaning(original, result)}

law successful_dynamic_sum_interpretation:
  for tensor: Core.Dynamic
  SourceDynamicSum<tensor, Reduction.sum_dynamic(tensor)>

law empty_accumulation:
  for -shape: Shape.Shape
  for tensor: Core.Tensor<shape>
  for +index: U32
  for accumulator: F32
  {Reduction.sum_loop(0n, shape, tensor, index, accumulator) == Done{(tensor, accumulator)} : Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>}

law empty_sum_identity:
  for +shape: Shape.Shape
  for tensor: Core.Tensor<shape>
  for empty: {Shape.logical_numel(Shape.dimensions(shape)) == 0n : Nat}
  {Reduction.sum_observation(shape, Reduction.sum(shape, tensor)) == Some{0.0} : Maybe<&2, F32>}

law sum_valid_success:
  for +shape: Shape.Shape
  for tensor: Core.Tensor<shape>
  Totality.Success<Core.Error, Core.Tensor<shape> & Core.Tensor<Reduction.scalar_shape()>, Reduction.sum(shape, tensor)>

law sum_dynamic_valid_success:
  for tensor: Core.Dynamic
  Totality.Success<Core.Error, Core.Dynamic & Core.Dynamic, Reduction.sum_dynamic(tensor)>

type TotalSum<-shape: Shape.Shape, -original: Core.Tensor<shape>,
  -result: Result<&2, &1, Core.Error, Core.Tensor<shape> & Core.Tensor<Reduction.scalar_shape()>>> is Data:
  TotalSum{-done: Totality.Success<Core.Error, Core.Tensor<shape> & Core.Tensor<Reduction.scalar_shape()>, result>,
    -interpretation: SourceSumResult<shape, original, result>}

type TotalDynamicSum<-original: Core.Dynamic, -result: Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic>> is Data:
  TotalDynamicSum{-done: Totality.Success<Core.Error, Core.Dynamic & Core.Dynamic, result>,
    -interpretation: SourceDynamicSum<original, result>}

law sum_valid_result:
  for +shape: Shape.Shape
  for tensor: Core.Tensor<shape>
  TotalSum<shape, tensor, Reduction.sum(shape, tensor)>

law sum_dynamic_valid_result:
  for tensor: Core.Dynamic
  TotalDynamicSum<tensor, Reduction.sum_dynamic(tensor)>
