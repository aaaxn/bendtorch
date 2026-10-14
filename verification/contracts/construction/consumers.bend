import Base
import ../../../src/tensor/construction-contract-LAWS.bend as Contract
import ../../../src/tensor/construction-contract-PROOF.bend as ContractProof
import ../../../src/tensor/construction-metadata.bend as Metadata
import ../../../src/shape/core.bend as Shape
import ../../../src/shape/nat_proofs.bend as Nats
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/allocation-spec.bend as Allocation
import ../../../src/tensor/observation.bend as Observation

def scalar_shape() -> Shape.Shape:
  Shape.SShape{[], 1n, {==}, Unit{}, Unit{}}

def three_shape() -> Shape.Shape:
  Shape.SShape{[3], 3n, {==}, Unit{}, Unit{}}

def empty_shape() -> Shape.Shape:
  Shape.SShape{[0], 0n, {==}, Unit{}, Unit{}}

def leading_shape(+a: U32, +b: U32, +c: U32) -> Shape.Shape:
  Shape.SShape{[0, a, b, c], 0n, {==}, Unit{}, Unit{}}

def trailing_zero(+a: U32, +b: U32, +c: U32) -> {Shape.logical_numel([a, b, c, 0]) == 0n : Nat}:
  Equal.trans(Nat, Shape.logical_numel([a, b, c, 0]), Nat.mul(U32.to_nat(a), 0n), 0n,
    Equal.cong(Nat, Nat, number => Nat.mul(U32.to_nat(a), number),
      Nat.mul(U32.to_nat(b), Nat.mul(U32.to_nat(c), 0n)), 0n,
      Equal.trans(Nat, Nat.mul(U32.to_nat(b), Nat.mul(U32.to_nat(c), 0n)), Nat.mul(U32.to_nat(b), 0n), 0n,
        Equal.cong(Nat, Nat, number => Nat.mul(U32.to_nat(b), number), Nat.mul(U32.to_nat(c), 0n), 0n,
          Nats.mul_zero_right(U32.to_nat(c))), Nats.mul_zero_right(U32.to_nat(b)))),
    Nats.mul_zero_right(U32.to_nat(a)))

def trailing_shape(+a: U32, +b: U32, +c: U32) -> Shape.Shape:
  +zero = trailing_zero(a, b, c)
  Shape.SShape{[a, b, c, 0], 0n, Equal.sym(Nat, Shape.logical_numel([a, b, c, 0]), 0n, zero), Unit{},
    %Equal.sym(Nat, Shape.logical_numel([a, b, c, 0]), 0n, zero) : Shape.LE(_, 1048576n)
    Unit{}}

type Case<-shape: Shape.Shape, -index: U32, -other: U32, -dimensions: List<&2, U32>, -count: Nat, -word: U32> is Data:
  Case{
    successful: {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, Core.new(shape, 7.0)) == True{} : Bool},
    dimensions_exact: {Observation.dimensions(shape, Core.new(shape, 7.0)) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
    count_exact: {Allocation.tensor_length(shape, Core.new(shape, 7.0)) == Some{count} : Maybe<&2, Nat>},
    word_count_exact: {Observation.count(shape, Core.new(shape, 7.0)) == Some{word} : Maybe<&2, U32>},
    value_exact: {Observation.point(shape, index, Core.new(shape, 7.0)) == Some{7.0} : Maybe<&2, F32>},
    retained_dimensions: {Observation.after_dimensions(shape, index, Core.new(shape, 7.0)) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
    retained_count: {Observation.after_length(shape, index, Core.new(shape, 7.0)) == Some{count} : Maybe<&2, Nat>},
    retained_word_count: {Observation.after_count(shape, index, Core.new(shape, 7.0)) == Some{word} : Maybe<&2, U32>},
    retained_value: {Observation.after_value(shape, index, other, Core.new(shape, 7.0)) == Some{7.0} : Maybe<&2, F32>}}

def scalar_expose(evidence: Contract.NewReads<scalar_shape(), 7.0, 0>) -> Case<scalar_shape(), 0, 0, [], 1n, 1>:
  Contract.NewReads{successful, dimensions, count, word_count, value, retained_dimensions, retained_count, retained_word_count, retained_value} = evidence
  Case{successful, dimensions, count, word_count, value, retained_dimensions, retained_count, retained_word_count, retained_value(0, {==})}

def scalar() -> Case<scalar_shape(), 0, 0, [], 1n, 1>:
  scalar_expose(Contract.new_reads(scalar_shape(), 7.0, 0, {==}))

def three_expose(-index: U32, +other: U32, evidence: Contract.NewReads<three_shape(), 7.0, index>,
  inside: {Nat.is_lt(U32.to_nat(other), 3n) == True{} : Bool}) -> Case<three_shape(), index, other, [3], 3n, 3>:
  Contract.NewReads{successful, dimensions, count, word_count, value, retained_dimensions, retained_count, retained_word_count, retained_value} = evidence
  Case{successful, dimensions, count, word_count, value, retained_dimensions, retained_count, retained_word_count, retained_value(other, inside)}

def three_zero() -> Case<three_shape(), 0, 2, [3], 3n, 3>:
  three_expose(0, 2, Contract.new_reads(three_shape(), 7.0, 0, {==}), {==})

def three_one() -> Case<three_shape(), 1, 0, [3], 3n, 3>:
  three_expose(1, 0, Contract.new_reads(three_shape(), 7.0, 1, {==}), {==})

def three_two() -> Case<three_shape(), 2, 1, [3], 3n, 3>:
  three_expose(2, 1, Contract.new_reads(three_shape(), 7.0, 2, {==}), {==})

type EmptyCase<-shape: Shape.Shape, -dimensions: List<&2, U32>> is Data:
  EmptyCase{
    successful: {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, Core.new(shape, 7.0)) == True{} : Bool},
    dimensions_exact: {Observation.dimensions(shape, Core.new(shape, 7.0)) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
    count_exact: {Allocation.tensor_length(shape, Core.new(shape, 7.0)) == Some{0n} : Maybe<&2, Nat>},
    word_count_exact: {Observation.count(shape, Core.new(shape, 7.0)) == Some{0} : Maybe<&2, U32>}}

def empty_expose(evidence: Metadata.NewMetadata<empty_shape(), 7.0>) -> EmptyCase<empty_shape(), [0]>:
  Metadata.NewMetadata{successful, dimensions, count, word_count} = evidence
  EmptyCase{successful, dimensions, count, word_count}

def empty() -> EmptyCase<empty_shape(), [0]>:
  empty_expose(Contract.new_metadata(empty_shape(), 7.0))

def leading_expose(+a: U32, +b: U32, +c: U32, evidence: Metadata.NewMetadata<leading_shape(a, b, c), 7.0>) ->
  EmptyCase<leading_shape(a, b, c), [0, a, b, c]>:
  Metadata.NewMetadata{successful, dimensions, count, word_count} = evidence
  EmptyCase{successful, dimensions, count, word_count}

def leading(+a: U32, +b: U32, +c: U32) -> EmptyCase<leading_shape(a, b, c), [0, a, b, c]>:
  leading_expose(a, b, c, Contract.new_metadata(leading_shape(a, b, c), 7.0))

def huge_leading() -> EmptyCase<leading_shape(4294967295, 4294967295, 4294967295), [0, 4294967295, 4294967295, 4294967295]>:
  leading(4294967295, 4294967295, 4294967295)

def trailing_expose(+a: U32, +b: U32, +c: U32, evidence: Metadata.NewMetadata<trailing_shape(a, b, c), 7.0>) ->
  EmptyCase<trailing_shape(a, b, c), [a, b, c, 0]>:
  Metadata.NewMetadata{successful, dimensions, count, word_count} = evidence
  EmptyCase{successful, dimensions,
    Equal.trans(Maybe<&2, Nat>, Allocation.tensor_length(trailing_shape(a, b, c), Core.new(trailing_shape(a, b, c), 7.0)),
      Some{Shape.logical_numel([a, b, c, 0])}, Some{0n}, count,
      Equal.cong(Nat, Maybe<&2, Nat>, number => Some{number}, Shape.logical_numel([a, b, c, 0]), 0n, trailing_zero(a, b, c))), word_count}

def trailing(+a: U32, +b: U32, +c: U32) -> EmptyCase<trailing_shape(a, b, c), [a, b, c, 0]>:
  trailing_expose(a, b, c, Contract.new_metadata(trailing_shape(a, b, c), 7.0))

def huge_trailing() -> EmptyCase<trailing_shape(4294967295, 4294967295, 4294967295), [4294967295, 4294967295, 4294967295, 0]>:
  trailing(4294967295, 4294967295, 4294967295)
