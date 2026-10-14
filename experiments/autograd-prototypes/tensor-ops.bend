import Base

import ../../src/shape/core.bend as Shape

import ../../src/tensor/core.bend as Core

import ../../src/tensor/elementwise.bend as Elementwise

import ../../src/tensor/reduction.bend as Reduction

import ../../src/tensor/matmul.bend as Matrix

import ../../src/tensor/transforms.bend as Transforms

# Every op hands its input owners back.

type Error is Data:
  TensorError{error: Core.Error}
  MatrixError{error: Matrix.Error}
  ReshapeError{error: Transforms.Error}
  Unsupported{}
  Missing{}

def binary_result(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) ->
  Result<&2, &1, Error, (Core.Dynamic & Core.Dynamic) & Core.Dynamic>:
  (inputs, result) = pair
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{output}:
      Done{(inputs, output)}

def mul(a: Core.Dynamic, b: Core.Dynamic) -> Result<&2, &1, Error, (Core.Dynamic & Core.Dynamic) & Core.Dynamic>:
  binary_result(Elementwise.mul(a, b))

def add(a: Core.Dynamic, b: Core.Dynamic) -> Result<&2, &1, Error, (Core.Dynamic & Core.Dynamic) & Core.Dynamic>:
  binary_result(Elementwise.add(a, b))

def sum_result(result: Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic>) ->
  Result<&2, &1, Error, Core.Dynamic & Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{pair}:
      Done{pair}

def sum(a: Core.Dynamic) -> Result<&2, &1, Error, Core.Dynamic & Core.Dynamic>:
  sum_result(Reduction.sum_dynamic(a))

def scalar_result(pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> Result<&2, &1, Error, Core.Dynamic & F32>:
  (tensor, result) = pair
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{value}:
      Done{(tensor, value)}

def scalar(t: Core.Dynamic) -> Result<&2, &1, Error, Core.Dynamic & F32>:
  scalar_result(Core.read_dynamic(t, 0))

def fill_result(+shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{tensor}:
      Done{Core.pack(shape, tensor)}

def fill(+shape: Shape.Shape, value: F32) -> Result<&2, &1, Error, Core.Dynamic>:
  fill_result(shape, Core.new(shape, value))

def construct_result(result: Result<&2, &1, Core.Error, Core.Dynamic>) -> Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{tensor}:
      Done{tensor}

def construct(dims: List<&2, U32>, values: List<&2, F32>) -> Result<&2, &1, Error, Core.Dynamic>:
  construct_result(Core.construct(dims, values))

def dims_eq(a: List<&2, U32>, b: List<&2, U32>) -> Bool:
  match a b:
    case Nil{} Nil{}:
      True{}
    case Con{x, xs} Con{y, ys}:
      Bool.and(U32.is_eq(x, y), dims_eq(xs, ys))
    case _ _:
      False{}

def ones(n: Nat) -> List<&2, F32>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      1.0 <> ones(p)

def reshaped(result: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{ReshapeError{error}}
    case Done{tensor}:
      Done{tensor}

def row_product(want: List<&2, U32>, result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>) ->
  Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{MatrixError{error}}
    case Done{values}:
      (left, right, product) = values
      reshaped(Transforms.reshape_dynamic(product, want))

def row_ones(g: Core.Dynamic, want: List<&2, U32>, result: Result<&2, &1, Core.Error, Core.Dynamic>) -> Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{left}:
      row_product(want, Matrix.matmul(left, g))

def row_reduce(g: Core.Dynamic, have: List<&2, U32>, want: List<&2, U32>) -> Result<&2, &1, Error, Core.Dynamic>:
  match have:
    case Con{+r, Con{c, Nil{}}}:
      row_ones(g, want, Core.construct([1, r], ones(U32.to_nat(r))))
    case _:
      Fail{Unsupported{}}

def unbroadcast_dims(g: Core.Dynamic, +have: List<&2, U32>, +want: List<&2, U32>, same: Bool) -> Result<&2, &1, Error, Core.Dynamic>:
  match same:
    case True{}:
      Done{g}
    case False{}:
      row_reduce(g, have, want)

def unbroadcast(g: Core.Dynamic, +target: Shape.Shape) -> Result<&2, &1, Error, Core.Dynamic>:
  Core.Packed{+shape, tensor} = g
  unbroadcast_dims(Core.pack(shape, tensor), Shape.dimensions(shape), Shape.dimensions(target),
    dims_eq(Shape.dimensions(shape), Shape.dimensions(target)))

def shape_of(t: Core.Dynamic) -> Core.Dynamic & Shape.Shape:
  Core.shape(t)

def show_read(pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> IO(Core.Dynamic):
  (tensor, result) = pair
  match result:
    case Fail{error}:
      do IO<Core.Dynamic>:
        IO.print("read-error")
        return tensor
    case Done{value}:
      do IO<Core.Dynamic>:
        IO.print(U32.show(F32.bits(value)))
        return tensor

def show_values(fuel: Nat, tensor: Core.Dynamic, +index: U32) -> IO(Core.Dynamic):
  match fuel:
    case 0n:
      do IO<Core.Dynamic>:
        return tensor
    case 1n+tail:
      do IO<Core.Dynamic>:
        tensor : Core.Dynamic <- show_read(Core.read_dynamic(tensor, index))
        tensor : Core.Dynamic <- show_values(tail, tensor, U32.add(index, 1))
        return tensor

def show_count(pair: Core.Dynamic & U32) -> IO(Unit):
  (tensor, +count) = pair
  do IO<Unit>:
    tensor : Core.Dynamic <- show_values(U32.to_nat(count), tensor, 0)
    return Unit{}

def show_tensor(label: String, tensor: Core.Dynamic) -> IO(Unit):
  do IO<Unit>:
    IO.print(label)
    show_count(Core.count(tensor))

def error_code(error: Error) -> U32:
  match error:
    case TensorError{error}:
      Core.error_code(error)
    case MatrixError{error}:
      100
    case ReshapeError{error}:
      200
    case Unsupported{}:
      300
    case Missing{}:
      400

def Slot() -> Type:
  Maybe<&1, Core.Dynamic>

def take_tail(s: Slot(), pair: List<&1, Slot()> & Slot()) -> List<&1, Slot()> & Slot():
  (rest, got) = pair
  (s <> rest, got)

def take(slots: List<&1, Slot()>, +offset: Nat) -> List<&1, Slot()> & Slot():
  match slots offset:
    case Nil{} _:
      (Nil{}, None{})
    case Con{s, rest} 0n:
      (None{} <> rest, s)
    case Con{s, rest} 1n+p:
      take_tail(s, take(rest, p))

def put(slots: List<&1, Slot()>, +offset: Nat, value: Core.Dynamic) -> List<&1, Slot()>:
  match slots offset:
    case Nil{} _:
      Nil{}
    case Con{s, rest} 0n:
      Some{value} <> rest
    case Con{s, rest} 1n+p:
      s <> put(rest, p, value)

def accumulate_add(grads: List<&1, Slot()>, +offset: Nat, result: Result<&2, &1, Error, (Core.Dynamic & Core.Dynamic) & Core.Dynamic>) -> Result<&2, &1, Error, List<&1, Slot()>>:
  match result:
    case Fail{e}:
      Fail{e}
    case Done{triple}:
      (operands, total) = triple
      Done{put(grads, offset, total)}

def accumulate_taken(+offset: Nat, delta: Core.Dynamic, taken: List<&1, Slot()> & Slot()) -> Result<&2, &1, Error, List<&1, Slot()>>:
  (grads, slot) = taken
  match slot:
    case None{}:
      Done{put(grads, offset, delta)}
    case Some{g}:
      accumulate_add(grads, offset, add(g, delta))

def accumulate(grads: List<&1, Slot()>, +offset: Nat, delta: Core.Dynamic) -> Result<&2, &1, Error, List<&1, Slot()>>:
  accumulate_taken(offset, delta, take(grads, offset))
