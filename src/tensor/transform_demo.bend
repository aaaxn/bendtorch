import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./transforms.bend as Transforms
import ./layout.bend as Layout

def from_word(word: U32) -> F32:
  match word:
    case U32{bits}:
      F32{bits}

def collect_step(rec: Core.Dynamic -> List<&2, U32> -> Core.Dynamic & Result<&2, &2, Core.Error, List<&2, U32>>,
  words: List<&2, U32>, pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) ->
  Core.Dynamic & Result<&2, &2, Core.Error, List<&2, U32>>:
  (tensor, result) = pair
  match result:
    case Fail{error}:
      (tensor, Fail{error})
    case Done{value}:
      rec(tensor)(F32.bits(value) <> words)

def collect(fuel: Nat, tensor: Core.Dynamic, +i: U32, words: List<&2, U32>) ->
  Core.Dynamic & Result<&2, &2, Core.Error, List<&2, U32>>:
  match fuel:
    case 0n:
      (tensor, Done{Layout.reverse_words(words)})
    case 1n+p:
      collect_step(tensor => words => collect(p, tensor, U32.inc(i), words), words, Core.read_dynamic(tensor, i))

def report_values(name: String, pair: Core.Dynamic & Result<&2, &2, Core.Error, List<&2, U32>>) -> IO(Core.Dynamic):
  (tensor, result) = pair
  match result:
    case Fail{error}:
      do IO<Core.Dynamic>:
        IO.print(name ++ ":read_error=" ++ U32.show(Core.error_code(error)))
        return tensor
    case Done{words}:
      do IO<Core.Dynamic>:
        IO.print(name ++ ":words=" ++ List.show(~&2, ~U32, ~U32.show, words))
        return tensor

def report_shape(name: String, pair: Core.Dynamic & Shape.Shape) -> IO(Core.Dynamic):
  (tensor, +shape) = pair
  do IO<Core.Dynamic>:
    IO.print(name ++ ":shape=" ++ List.show(~&2, ~U32, ~U32.show, Shape.dimensions(shape)))
    report_values("values", collect(U32.to_nat(Shape.count(shape)), tensor, 0, Nil{}))

def report(name: String, tensor: Core.Dynamic) -> IO(Core.Dynamic):
  report_shape(name, Core.shape(tensor))

def restored(result: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("inverse:error=" ++ U32.show(Transforms.error_code(error)))
    case Done{tensor}:
      do IO<Unit>:
        tensor : Core.Dynamic <- report("inverse", tensor)
        return Unit{}

def after_inverse(pair: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> IO(Unit):
  (tensor, result) = pair
  do IO<Unit>:
    tensor : Core.Dynamic <- report("permuted_retained", tensor)
    restored(result)

def after_permute(inverse: List<&2, U32>, pair: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> IO(Unit):
  (original, result) = pair
  match result:
    case Fail{error}:
      do IO<Unit>:
        IO.print("transform:error=" ++ U32.show(Transforms.error_code(error)))
        original : Core.Dynamic <- report("original", original)
        return Unit{}
    case Done{output}:
      do IO<Unit>:
        original : Core.Dynamic <- report("original", original)
        output : Core.Dynamic <- report("permuted", output)
        after_inverse(Transforms.permute_dynamic(output, inverse))

def permutation_created(name: String, axes: List<&2, U32>, inverse: List<&2, U32>,
  result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("construction:error=" ++ U32.show(Core.error_code(error)))
    case Done{tensor}:
      do IO<Unit>:
        IO.print(name)
        after_permute(inverse, Transforms.permute_dynamic(tensor, axes))

def permutation_case(name: String, dimensions: List<&2, U32>, values: List<&2, F32>, axes: List<&2, U32>, inverse: List<&2, U32>) -> IO(Unit):
  permutation_created(name, axes, inverse, Core.construct(dimensions, values))

def reshape_result(result: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("reshape:error=" ++ U32.show(Transforms.error_code(error)))
    case Done{tensor}:
      do IO<Unit>:
        tensor : Core.Dynamic <- report("reshaped", tensor)
        return Unit{}

def reshape_created(dimensions: List<&2, U32>, result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("construction:error=" ++ U32.show(Core.error_code(error)))
    case Done{tensor}:
      reshape_result(Transforms.reshape_dynamic(tensor, dimensions))

def reshape_case(dimensions: List<&2, U32>) -> IO(Unit):
  reshape_created(dimensions, Core.construct([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

def transpose_created(a: U32, b: U32, result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("construction:error=" ++ U32.show(Core.error_code(error)))
    case Done{tensor}:
      after_permute([1, 0], Transforms.transpose_dynamic(tensor, a, b))

def reshape_inverse_result(result: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("reshape:error=" ++ U32.show(Transforms.error_code(error)))
    case Done{tensor}:
      do IO<Unit>:
        tensor : Core.Dynamic <- report("reshape_forward", tensor)
        reshape_result(Transforms.reshape_dynamic(tensor, [2, 3]))

def reshape_inverse_created(result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("construction:error=" ++ U32.show(Core.error_code(error)))
    case Done{tensor}:
      reshape_inverse_result(Transforms.reshape_dynamic(tensor, [3, 2]))

def main() -> IO(Unit):
  do IO<Unit>:
    permutation_case("matrix", [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 0], [1, 0])
    permutation_case("raw_words", [2, 3], [from_word(0), from_word(2147483648), from_word(1), from_word(2139095040), from_word(4286578688), from_word(2143363909)], [1, 0], [1, 0])
    permutation_case("rank3", [2, 3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [1, 2, 0], [2, 0, 1])
    permutation_case("rank4", [2, 2, 2, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], [1, 3, 0, 2], [2, 0, 3, 1])
    permutation_case("scalar", [], [7.0], [], [])
    permutation_case("empty", [2, 0, 3], [], [2, 0, 1], [1, 2, 0])
    permutation_case("huge_zero", [4294967295, 4294967295, 0, 4294967295], [], [3, 2, 1, 0], [3, 2, 1, 0])
    permutation_case("wrong_length", [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0], [0, 1])
    permutation_case("duplicate", [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0, 0], [0, 1])
    permutation_case("invalid_axis", [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0, 2], [0, 1])
    transpose_created(0, 1, Core.construct([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    transpose_created(2, 2, Core.construct([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    reshape_case([3, 2])
    reshape_case([4])
    reshape_case([1, 1, 1, 1, 6])
    reshape_case([4294967295, 4294967295])
    reshape_inverse_created(Core.construct([2, 3], [from_word(0), from_word(2147483648), from_word(1), from_word(2139095040), from_word(4286578688), from_word(2143363909)]))
