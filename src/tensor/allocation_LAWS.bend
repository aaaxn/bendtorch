import Base
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ./allocation_spec.bend as Allocation
import ./core.bend as Core
import ../shape/core.bend as Shape

law storage_new_complete:
  for +word: U32
  for +fill: F32
  for +within: {U32.is_le(word, 1048576) == True{} : Bool}
  {Allocation.storage_length(Storage.new(word, fill)) == Some{U32.to_nat(word)} : Maybe<&2, Nat>}

law storage_new_zero:
  for +fill: F32
  {Storage.new(0, fill) == Done{Storage.Empty{}} : Result<&2, &1, Storage.Error, Storage.Storage>}

law storage_new_depth_bounds:
  for +word: U32
  for +within: {U32.is_le(word, 1048576) == True{} : Bool}
  for positive: {U32.is_lt(0, word) == True{} : Bool}
  {Storage.bounds(Pair.fst(Nat, Nat, Storage.depth_seek(20n, U32.to_nat(word), 0n, 1n, Nat.is_le(U32.to_nat(word), 1n))), U32.to_nat(word)) == True{} : Bool}

law storage_seek_bounds:
  for +fuel: Nat
  for +length: Nat
  for +depth: Nat
  for +capacity: Nat
  for done: Bool
  for +decided: {done == Nat.is_le(length, capacity) : Bool}
  for +sized: {capacity == Model.capacity(depth) : Nat}
  for +left: {Nat.add(fuel, depth) == 20n : Nat}
  for +limit: {Nat.is_le(length, Model.capacity(20n)) == True{} : Bool}
  for +positive: {Nat.is_lt(0n, length) == True{} : Bool}
  {Storage.bounds(Pair.fst(Nat, Nat, Storage.depth_seek(fuel, length, depth, capacity, done)), length) == True{} : Bool}

law core_new_complete:
  for +shape: Shape.Shape
  for +fill: F32
  {Allocation.tensor_length(shape, Core.new(shape, fill)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}

law core_new_done:
  for +shape: Shape.Shape
  for +fill: F32
  Allocation.TensorDone<shape, Core.new(shape, fill)>
