import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ./core.bend as Core

def scalar_shape() -> Shape.Shape:
  Shape.SShape{Nil{}, 1n, {==}, Unit{}, Unit{}}

def sequential_sum(values: List<&2, F32>, accumulator: F32) -> F32:
  match values:
    case Nil{}:
      accumulator
    case value <> tail:
      sequential_sum(tail, F32.add(accumulator, value))

def read_step(-shape: Shape.Shape,
  rec: Core.Tensor<shape> -> F32 -> Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>,
  accumulator: F32, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) ->
  Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>:
  (tensor, result) = pair
  match result:
    case Fail{error}:
      Fail{error}
    case Done{value}:
      rec(tensor, F32.add(accumulator, value))

def sum_decided(-shape: Shape.Shape,
  rec: Core.Tensor<shape> -> F32 -> Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>, tensor: Core.Tensor<shape>, +cursor: Nat,
  +index: U32, accumulator: F32, decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>) ->
  Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{Core.StorageError{Storage.InvalidIndex{}}}
    case Shape.BTrue{equal}:
      read_step(shape, rec, accumulator, Core.read_flat(shape, tensor, index))

def natural_sum_loop(fuel: Nat, -shape: Shape.Shape, tensor: Core.Tensor<shape>, +cursor: Nat,
  accumulator: F32) -> Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>:
  match fuel:
    case 0n:
      Done{(tensor, accumulator)}
    case 1n+tail:
      +index = U32.from_nat(cursor)
      sum_decided(shape, tensor => value => natural_sum_loop(tail, shape, tensor, 1n+cursor, value), tensor, cursor, index, accumulator,
        Shape.bool_case(Nat.is_eq(U32.to_nat(index), cursor)))

def sum_loop(fuel: Nat, -shape: Shape.Shape, tensor: Core.Tensor<shape>, +index: U32,
  accumulator: F32) -> Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>:
  natural_sum_loop(fuel, shape, tensor, U32.to_nat(index), accumulator)

def scalar_result(-input_shape: Shape.Shape, tensor: Core.Tensor<input_shape>,
  result: Result<&2, &1, Core.Error, Core.Tensor<scalar_shape()>>) ->
  Result<&2, &1, Core.Error, Core.Tensor<input_shape> & Core.Tensor<scalar_shape()>>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{scalar}:
      Done{(tensor, scalar)}

def sum_result(-shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>) ->
  Result<&2, &1, Core.Error, Core.Tensor<shape> & Core.Tensor<scalar_shape()>>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{pair}:
      (tensor, value) = pair
      scalar_result(shape, tensor, Core.new(scalar_shape(), value))

# The fuel + 1 words of the array from slot `cursor`, whose read `pair` is
# already made, added to `accumulator` left to right. Each step reads the
# next word and hands it on as an argument: one tail call with no checked
# read and no closure per word.
def array_run(fuel: Nat, +cursor: Nat, accumulator: F32, pair: Array<F32> & F32) -> F32 & Array<F32>:
  match fuel pair:
    case 0n Tuple{array, word}:
      (F32.add(accumulator, word), array)
    case 1n+rest Tuple{array, word}:
      array_run(rest, 1n+cursor, F32.add(accumulator, word), Array.get(F32, array, U32.from_nat(1n+cursor)))

# The first `fuel` words of the array from slot `cursor`, added to
# `accumulator` left to right: the F32 adds of `natural_sum_loop`, in the
# same order.
def array_sum(fuel: Nat, +cursor: Nat, accumulator: F32, array: Array<F32>) -> F32 & Array<F32>:
  match fuel:
    case 0n:
      (accumulator, array)
    case 1n+rest:
      array_run(rest, cursor, accumulator, Array.get(F32, array, U32.from_nat(cursor)))

def array_from(pair: Array<F32> & F32) -> Array<F32>:
  Refinement.first(F32, pair)

type Fact<-P: Type> is Data:
  Fact{-proof: P}

def kept_balanced(-depth: Nat, -original: Array<F32>, -balanced: Model.Balanced(F32, depth, original), -kept: Array<F32>,
  -same: {kept == original : Array<F32>}) -> Fact<Model.Balanced(F32, depth, kept)>:
  Fact{
    %Equal.sym(Array<F32>, kept, original, same) : Model.Balanced(F32, depth, _)
    balanced}

# The array a read hands back is the array it read.
def read_keeps(-depth: Nat, -original: Array<F32>, -balanced: Model.Balanced(F32, depth, original), -array: Array<F32>,
  -index: U32, -pair: Array<F32> & F32, -exact: {pair == Array.get(F32, array, index) : Array<F32> & F32},
  -same: {array == original : Array<F32>}) -> Fact<{array_from(pair) == original : Array<F32>}>:
  Fact{Equal.trans(Array<F32>, array_from(pair), array_from(Array.get(F32, array, index)), original,
    Equal.cong(Array<F32> & F32, Array<F32>, p => array_from(p), pair, Array.get(F32, array, index), exact),
    Equal.trans(Array<F32>, array_from(Array.get(F32, array, index)), array, original,
      Refinement.law_get_retains(F32, depth, array, index,
        {\{Fact: p => p} : Fact<(Model.Balanced(F32, depth, array))> -> Model.Balanced(F32, depth, array)}(
          kept_balanced(depth, original, balanced, array, same))),
      same))}

# A run hands back the array it read.
def array_run_keeps(-depth: Nat, -original: Array<F32>, -balanced: Model.Balanced(F32, depth, original), fuel: Nat, +cursor: Nat,
  -accumulator: F32, -array: Array<F32>, pair: Array<F32> & F32,
  -exact: {pair == Array.get(F32, array, U32.from_nat(cursor)) : Array<F32> & F32}, -same: {array == original : Array<F32>}) ->
  Fact<{Pair.snd(F32, Array<F32>, array_run(fuel, cursor, accumulator, pair)) == original : Array<F32>}>:
  match fuel pair:
    case 0n Tuple{kept, word}:
      read_keeps(depth, original, balanced, array, U32.from_nat(cursor), (kept, word), exact, same)
    case 1n+rest Tuple{kept, word}:
      array_run_keeps(depth, original, balanced, rest, 1n+cursor, F32.add(accumulator, word), kept,
        Array.get(F32, kept, U32.from_nat(1n+cursor)), {==},
        {\{Fact: p => p} : Fact<({array_from((kept, word)) == original : Array<F32>})> -> {kept == original : Array<F32>}}(
          read_keeps(depth, original, balanced, array, U32.from_nat(cursor), (kept, word), exact, same)))

def array_sum_keeps(-depth: Nat, -original: Array<F32>, -balanced: Model.Balanced(F32, depth, original), fuel: Nat, +cursor: Nat,
  -accumulator: F32, array: Array<F32>, -same: {array == original : Array<F32>}) ->
  Fact<{Pair.snd(F32, Array<F32>, array_sum(fuel, cursor, accumulator, array)) == original : Array<F32>}>:
  match fuel:
    case 0n:
      Fact{same}
    case 1n+rest:
      array_run_keeps(depth, original, balanced, rest, cursor, accumulator, array, Array.get(F32, array, U32.from_nat(cursor)), {==}, same)

def owned_sum(-shape: Shape.Shape, +depth: Nat, +length: Nat, -array: Array<F32>,
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  pair: F32 & Array<F32>, -kept: Fact<{Pair.snd(F32, Array<F32>, pair) == array : Array<F32>}>) ->
  Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>:
  (value, retained) = pair
  Done{(Core.TTensor{Storage.Owned{depth, length, retained,
    {\{Fact: p => p} : Fact<(Model.Balanced(F32, depth, retained))> -> Model.Balanced(F32, depth, retained)}(
      kept_balanced(depth, array, balanced, retained,
        {\{Fact: p => p} : Fact<({Pair.snd(F32, Array<F32>, (value, retained)) == array : Array<F32>})> -> {retained == array : Array<F32>}}(kept))),
    valid}, exact}, value)}

# The first `count` words of the tensor added left to right from +0.0, next
# to the tensor, which keeps every word.
def tensor_sum(-shape: Shape.Shape, +count: Nat, tensor: Core.Tensor<shape>) -> Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>:
  Core.TTensor{storage, exact} = tensor
  match storage:
    case Storage.Empty{}:
      Done{(Core.TTensor{Storage.Empty{}, exact}, 0.0)}
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      owned_sum(shape, depth, length, array, balanced, valid, exact, array_sum(count, 0n, 0.0, array),
        array_sum_keeps(depth, array, balanced, count, 0n, 0.0, array, {==}))

def sum(+shape: Shape.Shape, tensor: Core.Tensor<shape>) ->
  Result<&2, &1, Core.Error, Core.Tensor<shape> & Core.Tensor<scalar_shape()>>:
  sum_result(shape, tensor_sum(shape, Shape.elements(shape), tensor))

def pack_result(+shape: Shape.Shape,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape> & Core.Tensor<scalar_shape()>>) ->
  Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{pair}:
      (tensor, scalar) = pair
      Done{(Core.pack(shape, tensor), Core.pack(scalar_shape(), scalar))}

def sum_dynamic(dynamic: Core.Dynamic) -> Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic>:
  Core.Packed{+shape, tensor} = dynamic
  pack_result(shape, sum(shape, tensor))

def sum_observation(-shape: Shape.Shape,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape> & Core.Tensor<scalar_shape()>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{pair}:
      (tensor, scalar) = pair
      Core.observe(scalar_shape(), scalar, 0)
