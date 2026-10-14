import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./transforms.bend as Transforms
import ./observation.bend as Observation

type Request is Data:
  Permute{axes: List<&2, U32>}
  Transpose{a: U32, b: U32}

def apply(+input: Shape.Shape, tensor: Core.Tensor<input>, request: Request) ->
  Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>:
  match request:
    case Permute{+axes}:
      Transforms.permute_copy(input, tensor, axes)
    case Transpose{+a, +b}:
      Transforms.transpose_copy(input, tensor, a, b)

def status(result: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> U32:
  match result:
    case Fail{error}:
      Transforms.error_code(error)
    case Done{output}:
      0

def failure_code_pair(-input: Shape.Shape, pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> U32:
  (source, result) = pair
  status(result)

def failure_code(+input: Shape.Shape, +request: Request, result: Result<&2, &1, Core.Error, Core.Tensor<input>>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Some{failure_code_pair(input, apply(input, tensor, request))}

def kept_value_pair(-input: Shape.Shape, +other: U32, pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Maybe<&2, F32>:
  (source, result) = pair
  Core.read_value(input, Core.read_flat(input, source, other))

def kept_value(+input: Shape.Shape, +request: Request, +other: U32, result: Result<&2, &1, Core.Error, Core.Tensor<input>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      kept_value_pair(input, other, apply(input, tensor, request))

def kept_dimensions_pair(+input: Shape.Shape, pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Maybe<&2, List<&2, U32>>:
  (source, result) = pair
  Observation.dimensions_pair(Core.shape(Core.pack(input, source)))

def kept_dimensions(+input: Shape.Shape, +request: Request, result: Result<&2, &1, Core.Error, Core.Tensor<input>>) -> Maybe<&2, List<&2, U32>>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      kept_dimensions_pair(input, apply(input, tensor, request))

def kept_length_pair(-input: Shape.Shape, pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Nat:
  (source, result) = pair
  Core.logical_length(input, source)

def kept_length(+input: Shape.Shape, +request: Request, result: Result<&2, &1, Core.Error, Core.Tensor<input>>) -> Maybe<&2, Nat>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Some{kept_length_pair(input, apply(input, tensor, request))}

def kept_count_pair(+input: Shape.Shape, pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Maybe<&2, U32>:
  (source, result) = pair
  Observation.count_pair(Core.count(Core.pack(input, source)))

def kept_count(+input: Shape.Shape, +request: Request, result: Result<&2, &1, Core.Error, Core.Tensor<input>>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      kept_count_pair(input, apply(input, tensor, request))

def reshape_code(+input: Shape.Shape, +dimensions: List<&2, U32>, result: Result<&2, &1, Core.Error, Core.Tensor<input>>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Some{status(Transforms.reshape(input, tensor, dimensions))}

def output_dimensions(result: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Maybe<&2, List<&2, U32>>:
  match result:
    case Fail{error}:
      None{}
    case Done{output}:
      Some{Transforms.dynamic_dimensions(output)}

def reshape_dimensions(+input: Shape.Shape, +dimensions: List<&2, U32>, result: Result<&2, &1, Core.Error, Core.Tensor<input>>) -> Maybe<&2, List<&2, U32>>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      output_dimensions(Transforms.reshape(input, tensor, dimensions))

def dynamic_length(dynamic: Core.Dynamic) -> Nat:
  Core.Packed{shape, tensor} = dynamic
  Core.logical_length(shape, tensor)

def output_length(result: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Maybe<&2, Nat>:
  match result:
    case Fail{error}:
      None{}
    case Done{output}:
      Some{dynamic_length(output)}

def reshape_length(+input: Shape.Shape, +dimensions: List<&2, U32>, result: Result<&2, &1, Core.Error, Core.Tensor<input>>) -> Maybe<&2, Nat>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      output_length(Transforms.reshape(input, tensor, dimensions))

def output_count(result: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{output}:
      Observation.count_pair(Core.count(output))

def reshape_count(+input: Shape.Shape, +dimensions: List<&2, U32>, result: Result<&2, &1, Core.Error, Core.Tensor<input>>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      output_count(Transforms.reshape(input, tensor, dimensions))

def output_value(+other: U32, result: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{output}:
      Core.dynamic_read_value(Core.read_dynamic(output, other))

def reshape_value(+input: Shape.Shape, +dimensions: List<&2, U32>, +other: U32, result: Result<&2, &1, Core.Error, Core.Tensor<input>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      output_value(other, Transforms.reshape(input, tensor, dimensions))

def reshape_written_pair(+input: Shape.Shape, +dimensions: List<&2, U32>, +other: U32,
  pair: Core.Tensor<input> & Result<&2, &2, Core.Error, Unit>) -> Maybe<&2, F32>:
  (tensor, result) = pair
  output_value(other, Transforms.reshape(input, tensor, dimensions))

def reshape_written(+input: Shape.Shape, +index: U32, +value: F32, +dimensions: List<&2, U32>, +other: U32,
  result: Result<&2, &1, Core.Error, Core.Tensor<input>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      reshape_written_pair(input, dimensions, other, Core.write_flat(input, tensor, index, value))
