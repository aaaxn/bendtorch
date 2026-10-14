import Base

import ../tensor/core.bend as Core

import ../tensor/harness.bend as Tensor

import ./broadcast.bend as Broadcast

import ./reduce.bend as Reduce

import ./sum-to-size.bend as SumToSize

import ./gradient.bend as Gradient

import ../shape/core.bend as Shape

def outcome(result: Result<&2, &1, Broadcast.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("S3:error:" ++ U32.show(Broadcast.error_code(error)))
    case Done{output}:
      do IO<Unit>:
        IO.print("S3:output")
        Tensor.snapshot(output)

def retained(pair: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>) -> IO(Unit):
  (input, result) = pair
  do IO<Unit>:
    IO.print("S3:retained")
    Tensor.snapshot(input)
    outcome(result)

def broadcast(input: Result<&2, &1, Core.Error, Core.Dynamic>, +target: List<&2, U32>) -> IO(Unit):
  match input:
    case Fail{error}:
      Tensor.construction_error(error)
    case Done{tensor}:
      retained(Broadcast.broadcast_to_dimensions(tensor, target))

def reduced_outcome(result: Result<&2, &1, Reduce.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("S3:error:" ++ U32.show(Reduce.error_code(error)))
    case Done{output}:
      do IO<Unit>:
        IO.print("S3:output")
        Tensor.snapshot(output)

def reduced(pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>) -> IO(Unit):
  (input, result) = pair
  do IO<Unit>:
    IO.print("S3:retained")
    Tensor.snapshot(input)
    reduced_outcome(result)

def reduce(input: Result<&2, &1, Core.Error, Core.Dynamic>, +target: List<&2, U32>) -> IO(Unit):
  match input:
    case Fail{error}:
      Tensor.construction_error(error)
    case Done{tensor}:
      reduced(Reduce.reduce_to_dimensions(tensor, target))

def accumulated(rec: Gradient.Slot -> IO(Gradient.Slot), outcome: Gradient.Outcome) -> IO(Gradient.Slot):
  match outcome:
    case Gradient.Accumulated{slot}:
      do IO<Gradient.Slot>:
        IO.print("S3:accumulated")
        rec(slot)
    case Gradient.Refused{slot, contribution, error}:
      do IO<Gradient.Slot>:
        IO.print("S3:refused:" ++ U32.show(Gradient.error_code(error)))
        Tensor.snapshot(contribution)
        rec(slot)

def contributions(items: List<&1, Result<&2, &1, Core.Error, Core.Dynamic>>, +shape: Shape.Shape, slot: Gradient.Slot) -> IO(Gradient.Slot):
  match items:
    case Nil{}:
      IO.pure(Gradient.Slot, slot)
    case Fail{error} <> rest:
      do IO<Gradient.Slot>:
        Tensor.construction_error(error)
        contributions(rest, shape, slot)
    case Done{tensor} <> rest:
      accumulated(slot => contributions(rest, shape, slot), Gradient.accumulate(shape, slot, tensor))

def finished(result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("S3:error:" ++ U32.show(Core.error_code(error)))
    case Done{gradient}:
      do IO<Unit>:
        IO.print("S3:gradient")
        Tensor.snapshot(gradient)

def accumulate_shape(-dimensions: List<&2, U32>, items: List<&1, Result<&2, &1, Core.Error, Core.Dynamic>>,
  result: Shape.ConstructResult<dimensions>) -> IO(Unit):
  match result:
    case Shape.SError{error}:
      IO.print("S3:error:" ++ U32.show(Shape.error_code(error)))
    case Shape.SBuilt{+shape, preserved}:
      do IO<Unit>:
        slot : Gradient.Slot <- contributions(items, shape, Gradient.Unset{})
        finished(Gradient.gradient(shape, slot))

def accumulate(+dimensions: List<&2, U32>, items: List<&1, Result<&2, &1, Core.Error, Core.Dynamic>>) -> IO(Unit):
  accumulate_shape(dimensions, items, Shape.construct(dimensions))

def sum_to_size(input: Result<&2, &1, Core.Error, Core.Dynamic>, +target: List<&2, U32>) -> IO(Unit):
  match input:
    case Fail{error}:
      Tensor.construction_error(error)
    case Done{tensor}:
      reduced(SumToSize.sum_to_size_dimensions(tensor, target))

def snapshot_kept(pair: Core.Dynamic & Shape.Shape) -> IO(Core.Dynamic):
  (tensor, +shape) = pair
  do IO<Core.Dynamic>:
    IO.print("S2:shape:" ++ List.show(~&2, ~U32, ~U32.show, Shape.dimensions(shape)))
    IO.print("S2:count:" ++ U32.show(Shape.count(shape)))
    Tensor.words(U32.to_nat(Shape.count(shape)), 0, tensor)

def print_grad(read: Maybe<&1, Core.Dynamic>) -> IO(Gradient.Slot):
  match read:
    case None{}:
      do IO<Gradient.Slot>:
        IO.print("S3:grad:none")
        return Gradient.Unset{}
    case Some{gradient}:
      do IO<Gradient.Slot>:
        IO.print("S3:grad:some")
        kept : Core.Dynamic <- snapshot_kept(Core.shape(gradient))
        return Gradient.Held{kept}

def read_grad(slot: Gradient.Slot) -> IO(Gradient.Slot):
  print_grad(Gradient.grad(slot))

def lifecycle_shape(-dimensions: List<&2, U32>, before: List<&1, Result<&2, &1, Core.Error, Core.Dynamic>>,
  after: List<&1, Result<&2, &1, Core.Error, Core.Dynamic>>, result: Shape.ConstructResult<dimensions>) -> IO(Unit):
  match result:
    case Shape.SError{error}:
      IO.print("S3:error:" ++ U32.show(Shape.error_code(error)))
    case Shape.SBuilt{+shape, preserved}:
      do IO<Unit>:
        first : Gradient.Slot <- contributions(before, shape, Gradient.Unset{})
        held : Gradient.Slot <- read_grad(first)
        IO.print("S3:reset")
        cleared : Gradient.Slot <- read_grad(Gradient.reset(held))
        last : Gradient.Slot <- contributions(after, shape, cleared)
        final : Gradient.Slot <- read_grad(last)
        finished(Gradient.gradient(shape, final))

def lifecycle(+dimensions: List<&2, U32>, before: List<&1, Result<&2, &1, Core.Error, Core.Dynamic>>,
  after: List<&1, Result<&2, &1, Core.Error, Core.Dynamic>>) -> IO(Unit):
  lifecycle_shape(dimensions, before, after, Shape.construct(dimensions))
