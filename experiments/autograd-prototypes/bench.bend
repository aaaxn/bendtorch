import Base

import ../../src/tensor/core.bend as Core

import ./tensor-ops.bend as Ops

import ./tape-cases.bend as TapeScalar

import ./pullback-cases.bend as PullbackScalar

import ./dual-cases.bend as DualScalar

import ./tape-tensor.bend as Tape

import ./tape-tensor-cases.bend as TapeTensor

import ./pullback-tensor.bend as Pullback

import ./pullback-tensor-cases.bend as PullbackTensor

import ./dual-tensor.bend as Dual

import ./dual-tensor-cases.bend as DualTensor

type Job is Data:
  Construct{}
  TapeS{}
  PullbackS{}
  DualS{}
  TapeT{}
  PullbackT{}
  DualT{}

def total(xs: List<&2, F32>, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      total(t, F32.add(acc, h))

def scalar_sum(result: F32 & List<&2, F32>) -> F32:
  (value, grads) = result
  total(grads, value)

def scalar_input(+i: U32) -> F32:
  F32.add(3.0, U32.to_f32(U32.and(i, 7)))

def read_scalar(result: Result<&2, &1, Ops.Error, Core.Dynamic & F32>) -> F32:
  match result:
    case Fail{e}:
      F32.neg(1.0)
    case Done{pair}:
      (tensor, value) = pair
      value

def tensor_sum(result: Result<&2, &1, Ops.Error, Core.Dynamic & Core.Dynamic>) -> F32:
  match result:
    case Fail{e}:
      F32.neg(1.0)
    case Done{pair}:
      (input, scalar) = pair
      read_scalar(Ops.scalar(scalar))

def tensor_total(t: Core.Dynamic) -> F32:
  tensor_sum(Ops.sum(t))

def slot_total(slot: Ops.Slot()) -> F32:
  match slot:
    case None{}:
      0.0
    case Some{t}:
      tensor_total(t)

def tape_leaves(leaves: List<&1, Tape.Leaf()>, acc: F32) -> F32:
  match leaves:
    case Nil{}:
      acc
    case Con{leaf, rest}:
      (value, grad) = leaf
      tape_leaves(rest, F32.add(acc, slot_total(grad)))

def tape_total(result: Tape.R(Tape.Outcome)) -> F32:
  match result:
    case Fail{e}:
      F32.neg(1.0)
    case Done{outcome}:
      Tape.Outcome{value, leaves, forward_clones, backward_clones, saved} = outcome
      tape_leaves(leaves, tensor_total(value))

def slots_total(grads: List<&1, Ops.Slot()>, acc: F32) -> F32:
  match grads:
    case Nil{}:
      acc
    case Con{grad, rest}:
      slots_total(rest, F32.add(acc, slot_total(grad)))

def pullback_total(result: Pullback.R(Pullback.Outcome)) -> F32:
  match result:
    case Fail{e}:
      F32.neg(1.0)
    case Done{outcome}:
      Pullback.Outcome{value, inputs, grads, forward_clones, backward_clones, saved} = outcome
      slots_total(grads, tensor_total(value))

def dual_total(result: Dual.R(Dual.D())) -> F32:
  match result:
    case Fail{e}:
      F32.neg(1.0)
    case Done{pair}:
      (value, tangent) = pair
      F32.add(tensor_total(value), tensor_total(tangent))

def dual_tangent(result: Dual.R(Dual.D())) -> F32:
  match result:
    case Fail{e}:
      F32.neg(1.0)
    case Done{pair}:
      (value, tangent) = pair
      tensor_total(tangent)

def w_values(+bias: F32) -> List<&2, F32>:
  [F32.add(1.0, bias), 2.0, 3.0, 4.0, 5.0, 6.0]

def dual_case4(+k: Nat, +bias: F32) -> Dual.R(Dual.D()):
  do Result<&2, &1, Ops.Error, Dual.D()>:
    w : Dual.D() <- Dual.dual([2, 3], w_values(bias), Dual.one_hot(6n, 0n, k))
    x : Dual.D() <- Dual.dual([3], [0.5, F32.neg(1.0), 2.0], Dual.one_hot(3n, 6n, k))
    b : Dual.D() <- Dual.dual([3], [1.0, F32.neg(1.0), 0.5], Dual.one_hot(3n, 9n, k))
    wx : Dual.D() <- Dual.mul(w, x)
    z : Dual.D() <- Dual.add(wx, b)
    Dual.sum(z)

# All twelve forward passes of case 4: one per parameter element.
def dual_passes(fuel: Nat, +k: Nat, +bias: F32, acc: F32) -> F32:
  match fuel:
    case 0n:
      acc
    case 1n+rest:
      dual_passes(rest, 1n+k, bias, F32.add(acc, dual_tangent(dual_case4(k, bias))))

def tensor_inputs(+bias: F32) -> Result<&2, &1, Ops.Error, (Core.Dynamic & Core.Dynamic) & Core.Dynamic>:
  do Result<&2, &1, Ops.Error, (Core.Dynamic & Core.Dynamic) & Core.Dynamic>:
    w : Core.Dynamic <- Ops.construct([2, 3], w_values(bias))
    x : Core.Dynamic <- Ops.construct([3], [0.5, F32.neg(1.0), 2.0])
    b : Core.Dynamic <- Ops.construct([3], [1.0, F32.neg(1.0), 0.5])
    return ((w, x), b)

def dual_value(+bias: F32) -> F32:
  dual_total(dual_case4(12n, bias))

def construct_total(result: Result<&2, &1, Ops.Error, (Core.Dynamic & Core.Dynamic) & Core.Dynamic>) -> F32:
  match result:
    case Fail{e}:
      F32.neg(1.0)
    case Done{triple}:
      (wx, b) = triple
      (w, x) = wx
      F32.add(F32.add(tensor_total(w), tensor_total(x)), tensor_total(b))

def tape_tensor(result: Result<&2, &1, Ops.Error, (Core.Dynamic & Core.Dynamic) & Core.Dynamic>) -> F32:
  match result:
    case Fail{e}:
      F32.neg(1.0)
    case Done{triple}:
      (wx, b) = triple
      (w, x) = wx
      tape_total(Tape.run(TapeTensor.case4_program(w, x, b)))

def pullback_tensor(result: Result<&2, &1, Ops.Error, (Core.Dynamic & Core.Dynamic) & Core.Dynamic>) -> F32:
  match result:
    case Fail{e}:
      F32.neg(1.0)
    case Done{triple}:
      (wx, b) = triple
      (w, x) = wx
      pullback_total(Pullback.run(PullbackTensor.case4_program(w, x, b)))

def work(job: Job, +x: F32, +bias: F32) -> F32:
  match job:
    case Construct{}:
      construct_total(tensor_inputs(bias))
    case TapeS{}:
      F32.add(F32.add(scalar_sum(TapeScalar.run1(~F32, ~F32.add, ~F32.mul, x, 0.0, 1.0)),
        scalar_sum(TapeScalar.run2(~F32, ~F32.add, ~F32.mul, x, 0.0, 1.0))),
        scalar_sum(TapeScalar.run3(~F32, ~F32.add, ~F32.mul, x, x, 0.0, 1.0)))
    case PullbackS{}:
      F32.add(F32.add(scalar_sum(PullbackScalar.run1(~F32, ~F32.add, ~F32.mul, x, 0.0, 1.0)),
        scalar_sum(PullbackScalar.run2(~F32, ~F32.add, ~F32.mul, x, 0.0, 1.0))),
        scalar_sum(PullbackScalar.run3(~F32, ~F32.add, ~F32.mul, x, x, 0.0, 1.0)))
    case DualS{}:
      F32.add(F32.add(scalar_sum(DualScalar.run1(~F32, ~F32.add, ~F32.mul, x, 0.0, 1.0)),
        scalar_sum(DualScalar.run2(~F32, ~F32.add, ~F32.mul, x, 0.0, 1.0))),
        scalar_sum(DualScalar.run3(~F32, ~F32.add, ~F32.mul, x, x, 0.0, 1.0)))
    case TapeT{}:
      tape_tensor(tensor_inputs(bias))
    case PullbackT{}:
      pullback_tensor(tensor_inputs(bias))
    case DualT{}:
      F32.add(dual_value(bias), dual_passes(12n, 0n, bias, 0.0))

def repeat(+job: Job, fuel: Nat, +i: U32, acc: F32) -> F32:
  match fuel:
    case 0n:
      acc
    case 1n+rest:
      repeat(job, rest, U32.add(i, 1), F32.add(acc, work(job, scalar_input(i), U32.to_f32(U32.and(i, 7)))))

def job_hit(hit: Bool, job: Job, other: Maybe<&2, Job>) -> Maybe<&2, Job>:
  match hit:
    case True{}:
      Some{job}
    case False{}:
      other

def job_scan(+name: String, keys: List<&2, String>, jobs: List<&2, Job>) -> Maybe<&2, Job>:
  match keys jobs:
    case Con{key, more} Con{job, others}:
      job_hit(String.eq(name, key), job, job_scan(name, more, others))
    case _ _:
      None{}

def job_named(+name: String) -> Maybe<&2, Job>:
  job_scan(name, ["construct", "tape-scalar", "pullback-scalar", "dual-scalar", "tape-tensor", "pullback-tensor", "dual-tensor"],
    [Construct{}, TapeS{}, PullbackS{}, DualS{}, TapeT{}, PullbackT{}, DualT{}])

def start(job: Maybe<&2, Job>, count: Maybe<&2, U32>) -> IO(Unit):
  match job count:
    case Some{+job} Some{+n}:
      IO.print(U32.show(F32.bits(repeat(job, U32.to_nat(n), 0, 0.0))))
    case _ _:
      IO.print("usage: bench <construct|tape-scalar|pullback-scalar|dual-scalar|tape-tensor|pullback-tensor|dual-tensor> <n>")

def launch(args: List<String>) -> IO(Unit):
  match args:
    case Con{program, Con{name, Con{count, Nil{}}}}:
      start(job_named(name), U32.read(count))
    case _:
      start(None{}, None{})

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    launch(args)
