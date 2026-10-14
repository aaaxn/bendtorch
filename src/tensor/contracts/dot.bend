import Base
import ../core.bend as Core
import ./fold.bend as Fold

# Ordered-fold family, dot instance: output cell (row, column) of a row-major
# product of a rows x inner and an inner x columns matrix folds
# left[row, k] * right[k, column] for k = 0, 1, ..., inner - 1.

type Cell is Data:
  Cell{row: Nat, column: Nat, inner: Nat, columns: Nat}

def multiply_add(operation: Unit, accumulator: F32, left: F32, right: F32) -> F32:
  F32.add(accumulator, F32.mul(left, right))

def left_index(cell: Cell, k: Nat) -> Nat:
  Cell{row, column, inner, columns} = cell
  Nat.add(Nat.mul(row, inner), k)

def right_index(cell: Cell, k: Nat) -> Nat:
  Cell{row, column, inner, columns} = cell
  Nat.add(Nat.mul(k, columns), column)

def inner(cell: Cell) -> Nat:
  Cell{row, column, inner, columns} = cell
  inner

def dot(+cell: Cell, left: Core.Dynamic, right: Core.Dynamic) -> (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32>:
  Fold.pair_run(~Unit, ~multiply_add, ~Cell, ~left_index, ~right_index, Unit{}, cell, inner(cell), left, right)
