import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./transforms.bend as Transforms
import ./harness.bend as Harness
import ./signed.bend as Signed

# The signed requests of the tensor gate (tools/tensor_signed.py). Each
# prints the result's snapshot, or tensor:error: and the error's code.
type Request is Data:
  Reshape{extents: List<&2, Signed.Extent>}
  Permute{axes: List<&2, Signed.Position>}
  Transpose{a: Signed.Position, b: Signed.Position}
  Get{coordinates: List<&2, Signed.Position>}
  Set{coordinates: List<&2, Signed.Position>, value: U32}
  GetFlat{index: Signed.Position}
  SetFlat{index: Signed.Position, value: U32}

def refused(code: U32) -> IO(Unit):
  IO.print("tensor:error:" ++ U32.show(code))

def transformed(result: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      refused(Transforms.error_code(error))
    case Done{output}:
      Harness.snapshot(output)

def retained(pair: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> IO(Unit):
  (input, result) = pair
  transformed(result)

def reshaped(result: Result<&2, &1, Signed.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      refused(Signed.error_code(error))
    case Done{output}:
      Harness.snapshot(output)

def scalar(result: Result<&2, &2, Core.Error, F32>) -> IO(Unit):
  match result:
    case Fail{error}:
      refused(Core.error_code(error))
    case Done{value}:
      do IO<Unit>:
        IO.print("tensor:shape:[]")
        IO.print("tensor:count:1")
        IO.print("tensor:word:" ++ U32.show(F32.bits(value)))

def read(-shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) -> IO(Unit):
  (tensor, result) = pair
  scalar(result)

def flat(pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> IO(Unit):
  (input, result) = pair
  scalar(result)

def written(+shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) -> IO(Unit):
  (tensor, result) = pair
  match result:
    case Fail{error}:
      refused(Core.error_code(error))
    case Done{unit}:
      Harness.snapshot(Core.pack(shape, tensor))

def apply(request: Request, input: Core.Dynamic) -> IO(Unit):
  match request:
    case Reshape{extents}:
      reshaped(Signed.reshape_dynamic(input, extents))
    case Permute{axes}:
      retained(Signed.permute_dynamic(input, axes))
    case Transpose{a, b}:
      retained(Signed.transpose_dynamic(input, a, b))
    case Get{coordinates}:
      Core.Packed{+shape, tensor} = input
      read(shape, Signed.read_coordinate(shape, tensor, coordinates))
    case Set{coordinates, value}:
      Core.Packed{+shape, tensor} = input
      written(shape, Signed.write_coordinate(shape, tensor, coordinates, Harness.from_bits(value)))
    case GetFlat{index}:
      flat(Signed.read_dynamic(input, index))
    case SetFlat{index, value}:
      Core.Packed{+shape, tensor} = input
      written(shape, Signed.write_flat(shape, tensor, index, Harness.from_bits(value)))

def run(request: Request, input: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match input:
    case Fail{error}:
      refused(Core.error_code(error))
    case Done{tensor}:
      apply(request, tensor)
