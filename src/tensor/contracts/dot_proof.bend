import Base
import ../../shape/core.bend as Shape
import ../../shape/index_proofs.bend as Index
import ../../shape/nat_proofs.bend as Nats
import ../../storage/LAWS.bend as StorageLaws
import ../../storage/arith.bend as Arith
import ../../storage/api.bend as Api
import ../core.bend as Core
import ../lifecycle.bend as Lifecycle
import ../dot_fold.bend as DotFold
import ../word_conversion.bend as Conversion
import ./replayed.bend as Replayed
import ./pointwise.bend as Pointwise
import ./fold.bend as Fold
import ./fold_records.bend as Records
import ./fold_proof.bend as Proof
import ./dot.bend as Dot
import ./LAWS.bend as Laws

def dot_snoc(+values: List<&2, DotFold.Factors>, +left: F32, +right: F32, +accumulator: F32) ->
  {DotFold.dot_fold(List.append(&2, DotFold.Factors, values, [DotFold.Factors{left, right}]), accumulator) ==
    F32.add(DotFold.dot_fold(values, accumulator), F32.mul(left, right)) : F32}:
  match values:
    case Nil{}:
      {==}
    case DotFold.Factors{+a, +b} <> +rest:
      dot_snoc(rest, left, right, F32.add(accumulator, F32.mul(a, b)))

def total_factors(+row: Nat, +column: Nat, +inner: Nat, +columns: Nat,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat) ->
  {Fold.pair_total(~Unit, ~Dot.multiply_add, ~Dot.Cell, ~Dot.left_index, ~Dot.right_index, Unit{}, Dot.Cell{row, column, inner, columns},
    left, left_fill, left_ops, right, right_fill, right_ops, count) ==
    DotFold.dot_fold(Laws.factors(row, column, inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, count), 0.0) : F32}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +x = Replayed.value(left, left_fill, left_ops, U32.from_nat(Nat.add(Nat.mul(row, inner), previous)))
      +y = Replayed.value(right, right_fill, right_ops, U32.from_nat(Nat.add(Nat.mul(previous, columns), column)))
      +before = Laws.factors(row, column, inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, previous)
      %Equal.sym(F32, DotFold.dot_fold(List.append(&2, DotFold.Factors, before, [DotFold.Factors{x, y}]), 0.0),
        F32.add(DotFold.dot_fold(before, 0.0), F32.mul(x, y)), dot_snoc(before, x, y, 0.0)) :
        {F32.add(Fold.pair_total(~Unit, ~Dot.multiply_add, ~Dot.Cell, ~Dot.left_index, ~Dot.right_index, Unit{},
          Dot.Cell{row, column, inner, columns}, left, left_fill, left_ops, right, right_fill, right_ops, previous), F32.mul(x, y)) == _ : F32}
      Equal.cong(F32, F32, total => F32.add(total, F32.mul(x, y)),
        Fold.pair_total(~Unit, ~Dot.multiply_add, ~Dot.Cell, ~Dot.left_index, ~Dot.right_index, Unit{},
          Dot.Cell{row, column, inner, columns}, left, left_fill, left_ops, right, right_fill, right_ops, previous),
        DotFold.dot_fold(before, 0.0),
        total_factors(row, column, inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, previous))

def lt_mul_add(+a: Nat, +m: Nat, +b: Nat, +n: Nat, a_below: {Nat.is_lt(a, m) == True{} : Bool},
  b_below: {Nat.is_lt(b, n) == True{} : Bool}) -> {Nat.is_lt(Nat.add(Nat.mul(a, n), b), Nat.mul(m, n)) == True{} : Bool}:
  match a m:
    case 0n 0n:
      Empty.absurd({Nat.is_lt(b, 0n) == True{} : Bool}, Arith.clash(a_below))
    case 0n 1n+q:
      Arith.lt_le_trans(b, n, Nat.add(n, Nat.mul(q, n)), b_below, Api.le_add_right(n, Nat.mul(q, n)))
    case 1n+p 0n:
      Empty.absurd({Nat.is_lt(Nat.add(Nat.add(n, Nat.mul(p, n)), b), 0n) == True{} : Bool}, Arith.clash(a_below))
    case 1n+p 1n+q:
      %Index.add_assoc(n, Nat.mul(p, n), b) : {Nat.is_lt(_, Nat.add(n, Nat.mul(q, n))) == True{} : Bool}
      Api.lt_add_left(n, Nat.add(Nat.mul(p, n), b), Nat.mul(q, n), lt_mul_add(p, q, b, n, a_below, b_below))

def left_below(+rows: Nat, +inner: Nat, +columns: Nat, +row: Nat, +column: Nat,
  row_below: {Nat.is_lt(row, rows) == True{} : Bool}, +left: Shape.Shape,
  left_extents: {Shape.extents(left) == [rows, inner] : List<&2, Nat>}, +k: Nat, below: {Nat.is_lt(k, inner) == True{} : Bool}) ->
  {Nat.is_lt(Dot.left_index(Dot.Cell{row, column, inner, columns}, k), Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool}:
  %Equal.sym(List<&2, Nat>, Shape.extents(left), [rows, inner], left_extents) :
    {Nat.is_lt(Nat.add(Nat.mul(row, inner), k), Shape.numel(_)) == True{} : Bool}
  %Equal.sym(Nat, Nat.mul(inner, 1n), inner, Nats.mul_one_right(inner)) :
    {Nat.is_lt(Nat.add(Nat.mul(row, inner), k), Nat.mul(rows, _)) == True{} : Bool}
  lt_mul_add(row, rows, k, inner, row_below, below)

def right_below(+inner: Nat, +columns: Nat, +row: Nat, +column: Nat,
  column_below: {Nat.is_lt(column, columns) == True{} : Bool}, +right: Shape.Shape,
  right_extents: {Shape.extents(right) == [inner, columns] : List<&2, Nat>}, +k: Nat, below: {Nat.is_lt(k, inner) == True{} : Bool}) ->
  {Nat.is_lt(Dot.right_index(Dot.Cell{row, column, inner, columns}, k), Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool}:
  %Equal.sym(List<&2, Nat>, Shape.extents(right), [inner, columns], right_extents) :
    {Nat.is_lt(Nat.add(Nat.mul(k, columns), column), Shape.numel(_)) == True{} : Bool}
  %Equal.sym(Nat, Nat.mul(columns, 1n), columns, Nats.mul_one_right(columns)) :
    {Nat.is_lt(Nat.add(Nat.mul(k, columns), column), Nat.mul(inner, _)) == True{} : Bool}
  lt_mul_add(k, inner, column, columns, below, column_below)

def left_inside(+rows: Nat, +inner: Nat, +columns: Nat, +row: Nat, +column: Nat,
  +row_below: {Nat.is_lt(row, rows) == True{} : Bool}, +left: Shape.Shape,
  +left_extents: {Shape.extents(left) == [rows, inner] : List<&2, Nat>}, +count: Nat,
  +room: {Nat.is_le(count, inner) == True{} : Bool}) ->
  {Pointwise.inside(~Dot.Cell, ~Dot.left_index, Dot.Cell{row, column, inner, columns}, count, Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +cell : Dot.Cell = Dot.Cell{row, column, inner, columns}
      +bound = Shape.logical_numel(Shape.dimensions(left))
      %Equal.sym(Bool, Pointwise.inside(~Dot.Cell, ~Dot.left_index, cell, previous, bound), True{},
        left_inside(rows, inner, columns, row, column, row_below, left, left_extents, previous,
          Conversion.le_predecessor(previous, inner, room))) :
        {Bool.and(_, Nat.is_lt(Dot.left_index(cell, previous), bound)) == True{} : Bool}
      %Equal.sym(Bool, Nat.is_lt(Dot.left_index(cell, previous), bound), True{},
        left_below(rows, inner, columns, row, column, row_below, left, left_extents, previous,
          Conversion.le_successor_lt(previous, inner, room))) :
        {Bool.and(True{}, _) == True{} : Bool}
      {==}

def right_inside(+inner: Nat, +columns: Nat, +row: Nat, +column: Nat,
  +column_below: {Nat.is_lt(column, columns) == True{} : Bool}, +right: Shape.Shape,
  +right_extents: {Shape.extents(right) == [inner, columns] : List<&2, Nat>}, +count: Nat,
  +room: {Nat.is_le(count, inner) == True{} : Bool}) ->
  {Pointwise.inside(~Dot.Cell, ~Dot.right_index, Dot.Cell{row, column, inner, columns}, count, Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +cell : Dot.Cell = Dot.Cell{row, column, inner, columns}
      +bound = Shape.logical_numel(Shape.dimensions(right))
      %Equal.sym(Bool, Pointwise.inside(~Dot.Cell, ~Dot.right_index, cell, previous, bound), True{},
        right_inside(inner, columns, row, column, column_below, right, right_extents, previous,
          Conversion.le_predecessor(previous, inner, room))) :
        {Bool.and(_, Nat.is_lt(Dot.right_index(cell, previous), bound)) == True{} : Bool}
      %Equal.sym(Bool, Nat.is_lt(Dot.right_index(cell, previous), bound), True{},
        right_below(inner, columns, row, column, column_below, right, right_extents, previous,
          Conversion.le_successor_lt(previous, inner, room))) :
        {Bool.and(True{}, _) == True{} : Bool}
      {==}

def dot_ordered(+rows: Nat, +inner: Nat, +columns: Nat, +row: Nat, +row_below: {Nat.is_lt(row, rows) == True{} : Bool},
  +column: Nat, +column_below: {Nat.is_lt(column, columns) == True{} : Bool},
  +left: Shape.Shape, +left_extents: {Shape.extents(left) == [rows, inner] : List<&2, Nat>},
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right: Shape.Shape, +right_extents: {Shape.extents(right) == [inner, columns] : List<&2, Nat>},
  +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat}) ->
  Records.PairFoldContract<left, right,
    DotFold.dot_fold(Laws.factors(row, column, inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, inner), 0.0),
    other => Replayed.value(left, left_fill, left_ops, other),
    other => Replayed.value(right, right_fill, right_ops, other),
    Dot.dot(Dot.Cell{row, column, inner, columns},
      Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
      Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})>:
  +cell : Dot.Cell = Dot.Cell{row, column, inner, columns}
  %total_factors(row, column, inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, inner) :
    Records.PairFoldContract<left, right, _,
      other => Replayed.value(left, left_fill, left_ops, other),
      other => Replayed.value(right, right_fill, right_ops, other),
      Fold.pair_run(~Unit, ~Dot.multiply_add, ~Dot.Cell, ~Dot.left_index, ~Dot.right_index, Unit{}, cell, inner,
        Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
        Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})>
  Proof.pair_contract(~Unit, ~Dot.multiply_add, ~Dot.Cell, ~Dot.left_index, ~Dot.right_index, Unit{}, cell,
    left, left_fill, left_ops, right, right_fill, right_ops, inner,
    left_inside(rows, inner, columns, row, column, row_below, left, left_extents, inner, Conversion.le_reflexive(inner)),
    right_inside(inner, columns, row, column, column_below, right, right_extents, inner, Conversion.le_reflexive(inner)),
    left_fits, right_fits)
