import Base
import ../../src/shape/core.bend as Shape
import ../../src/shape/nat_proofs.bend as Nats
import ../../src/tensor/core.bend as Core
import ./public.bend as Public

def last_zero_product(+a: U32, +b: U32, +c: U32) ->
  {Shape.logical_numel([a, b, c, 0]) == 0n : Nat}:
  Equal.trans(Nat, Shape.logical_numel([a, b, c, 0]), Nat.mul(U32.to_nat(a), 0n), 0n,
    Equal.cong(Nat, Nat, value => Nat.mul(U32.to_nat(a), value),
      Nat.mul(U32.to_nat(b), Nat.mul(U32.to_nat(c), 0n)), 0n,
      Equal.trans(Nat, Nat.mul(U32.to_nat(b), Nat.mul(U32.to_nat(c), 0n)), Nat.mul(U32.to_nat(b), 0n), 0n,
        Equal.cong(Nat, Nat, value => Nat.mul(U32.to_nat(b), value), Nat.mul(U32.to_nat(c), 0n), 0n,
          Nats.mul_zero_right(U32.to_nat(c))), Nats.mul_zero_right(U32.to_nat(b)))),
    Nats.mul_zero_right(U32.to_nat(a)))

def last_zero_shape(+a: U32, +b: U32, +c: U32) -> Shape.Shape:
  +zero = last_zero_product(a, b, c)
  Shape.SShape{[a, b, c, 0], 0n, Equal.sym(Nat, Shape.logical_numel([a, b, c, 0]), 0n, zero), Unit{},
    %Equal.sym(Nat, Shape.logical_numel([a, b, c, 0]), 0n, zero) : Shape.LE(_, 1048576n)
    Unit{}}

def last_zero_producer(+fill: F32, +a: U32, +b: U32, +c: U32) ->
  Public.Facts<last_zero_shape(a, b, c), fill, [a, b, c, 0], 0, 0n, [None{}, None{}, None{}]>:
  Public.Facts{{==}, {==}, {==}, {==}, {==}}

def last_zero_consumer() ->
  {Result.is_done(&2, &1, Core.Error, Core.Tensor<last_zero_shape(4294967295, 4294967295, 4294967295)>,
    Core.new(last_zero_shape(4294967295, 4294967295, 4294967295), 7.0)) == True{} : Bool} &
  ({Public.result_dimensions(last_zero_shape(4294967295, 4294967295, 4294967295),
    Core.new(last_zero_shape(4294967295, 4294967295, 4294967295), 7.0)) == Some{[4294967295, 4294967295, 4294967295, 0]} : Maybe<&2, List<&2, U32>>} &
    {Public.result_count(last_zero_shape(4294967295, 4294967295, 4294967295),
      Core.new(last_zero_shape(4294967295, 4294967295, 4294967295), 7.0)) == Some{0} : Maybe<&2, U32>}):
  Public.Facts{successful, dimensions, count, length, values} = last_zero_producer(7.0, 4294967295, 4294967295, 4294967295)
  (successful, (dimensions, count))
