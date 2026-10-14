import Base
import ../shape/core.bend as Shape
import ../tensor/core.bend as Core
import ../tensor/contracts/fold.bend as Fold
import ../tensor/contracts/sum.bend as Sum
import ../tensor/contracts/broadcast.bend as Broadcast

# Ordered-fold family, reduce instance: target cell `index` of reduce_to sums,
# in increasing position, the contribution words whose broadcast projection
# lands on `index`. The plan broadcasts the target (left) with the
# contribution (right).

type Cell is Data:
  Cell{plan: Shape.BroadcastPlan, index: U32}

def source(cell: Cell, position: Nat) -> Nat:
  Cell{+plan, index} = cell
  Broadcast.right_map(plan, position)

def lands(cell: Cell, position: Nat) -> Bool:
  Cell{+plan, +index} = cell
  Nat.is_eq(Broadcast.left_map(plan, position), U32.to_nat(index))

def cell(+plan: Shape.BroadcastPlan, +index: U32, input: Core.Dynamic) -> Core.Dynamic & Result<&2, &1, Core.Error, F32>:
  Fold.run(~Unit, ~Sum.add, ~Cell, ~source, ~lands, Unit{}, Cell{plan, index}, Shape.elements(Shape.right_shape(plan)), input)
