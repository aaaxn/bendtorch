import Base
import ../../shape/core.bend as Shape
import ../core.bend as Core
import ./fold.bend as Fold

# Ordered-fold family, sum instance: fold every word of the input, in index
# order, with F32.add.

def add(operation: Unit, accumulator: F32, word: F32) -> F32:
  F32.add(accumulator, word)

def position(plan: Unit, index: Nat) -> Nat:
  index

def sum(input: Core.Dynamic) -> Core.Dynamic & Result<&2, &1, Core.Error, F32>:
  Core.Packed{+shape, tensor} = input
  Fold.run(~Unit, ~add, ~Unit, ~position, Unit{}, Unit{}, Shape.elements(shape), Core.Packed{shape, tensor})
