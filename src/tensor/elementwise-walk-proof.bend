import Base
import ../shape/core.bend as Shape
import ../shape/nat_proofs.bend as Nats
import ../storage/arith.bend as Arith
import ./elementwise.bend as Elementwise

# Proofs that a build whose inputs read output cell p at their own slot p or
# at slot 0 (`Elementwise.left_slot`, `Elementwise.right_slot`) computes every
# cell `Elementwise.cell` computes: there the broadcast walk of every position
# lands on that slot.

def add_zero_left(+number: Nat) -> {number == Nat.add(number, 0n) : Nat}:
  match number:
    case 0n:
      {==}
    case 1n+p:
      Equal.cong(Nat, Nat, value => 1n+value, p, Nat.add(p, 0n), add_zero_left(p))

def add_comm(+a: Nat, +b: Nat) -> {Nat.add(a, b) == Nat.add(b, a) : Nat}:
  match a:
    case 0n:
      add_zero_left(b)
    case 1n+p:
      Equal.trans(Nat, 1n+Nat.add(p, b), 1n+Nat.add(b, p), Nat.add(b, 1n+p),
        Equal.cong(Nat, Nat, n => 1n+n, Nat.add(p, b), Nat.add(b, p), add_comm(p, b)),
        Equal.sym(Nat, Nat.add(b, 1n+p), 1n+Nat.add(b, p), Arith.add_succ(b, p)))

# The division loop keeps quotient * (1 + bp) + remainder + what is left to
# count fixed, as long as the remainder and the countdown add up to bp.
def go_total(+n: Nat, +m: Nat, +d: Nat, +r: Nat, +bp: Nat, +fits: {Nat.add(m, r) == bp : Nat}) ->
  {Nat.add(Nat.mul(Pair.fst(Nat, Nat, Nat.divmod.go(n, m, d, r)), 1n+bp), Pair.snd(Nat, Nat, Nat.divmod.go(n, m, d, r))) ==
    Nat.add(Nat.add(Nat.mul(d, 1n+bp), r), n) : Nat}:
  match n m:
    case 0n m:
      add_zero_left(Nat.add(Nat.mul(d, 1n+bp), r))
    case 1n+np 0n:
      +total = Nat.mul(d, 1n+bp)
      Equal.trans(Nat,
        Nat.add(Nat.mul(Pair.fst(Nat, Nat, Nat.divmod.go(np, r, 1n+d, 0n)), 1n+bp), Pair.snd(Nat, Nat, Nat.divmod.go(np, r, 1n+d, 0n))),
        1n+Nat.add(Nat.add(Nat.add(bp, total), 0n), np),
        Nat.add(Nat.add(total, r), 1n+np),
        go_total(np, r, 1n+d, 0n, bp, Equal.trans(Nat, Nat.add(r, 0n), r, bp, Equal.sym(Nat, r, Nat.add(r, 0n), add_zero_left(r)), fits)),
        Equal.trans(Nat, 1n+Nat.add(Nat.add(Nat.add(bp, total), 0n), np), 1n+Nat.add(Nat.add(total, r), np), Nat.add(Nat.add(total, r), 1n+np),
          Equal.cong(Nat, Nat, x => 1n+Nat.add(x, np), Nat.add(Nat.add(bp, total), 0n), Nat.add(total, r),
            Equal.trans(Nat, Nat.add(Nat.add(bp, total), 0n), Nat.add(bp, total), Nat.add(total, r),
              Equal.sym(Nat, Nat.add(bp, total), Nat.add(Nat.add(bp, total), 0n), add_zero_left(Nat.add(bp, total))),
              Equal.trans(Nat, Nat.add(bp, total), Nat.add(total, bp), Nat.add(total, r), add_comm(bp, total),
                Equal.cong(Nat, Nat, x => Nat.add(total, x), bp, r, Equal.sym(Nat, r, bp, fits))))),
          Equal.sym(Nat, Nat.add(Nat.add(total, r), 1n+np), 1n+Nat.add(Nat.add(total, r), np), Arith.add_succ(Nat.add(total, r), np))))
    case 1n+np 1n+mp:
      +total = Nat.mul(d, 1n+bp)
      Equal.trans(Nat,
        Nat.add(Nat.mul(Pair.fst(Nat, Nat, Nat.divmod.go(np, mp, d, 1n+r)), 1n+bp), Pair.snd(Nat, Nat, Nat.divmod.go(np, mp, d, 1n+r))),
        Nat.add(Nat.add(total, 1n+r), np),
        Nat.add(Nat.add(total, r), 1n+np),
        go_total(np, mp, d, 1n+r, bp, Equal.trans(Nat, Nat.add(mp, 1n+r), 1n+Nat.add(mp, r), bp, Arith.add_succ(mp, r), fits)),
        Equal.trans(Nat, Nat.add(Nat.add(total, 1n+r), np), 1n+Nat.add(Nat.add(total, r), np), Nat.add(Nat.add(total, r), 1n+np),
          Equal.cong(Nat, Nat, x => Nat.add(x, np), Nat.add(total, 1n+r), 1n+Nat.add(total, r), Arith.add_succ(total, r)),
          Equal.sym(Nat, Nat.add(Nat.add(total, r), 1n+np), 1n+Nat.add(Nat.add(total, r), np), Arith.add_succ(Nat.add(total, r), np))))

# s * (i / s) + i % s = i, for every s; at s = 0, Nat.div gives 0 and
# Nat.mod gives i.
def div_mod(+i: Nat, +s: Nat) -> {Nat.add(Nat.mul(s, Nat.div(i, s)), Nat.mod(i, s)) == i : Nat}:
  match s:
    case 0n:
      {==}
    case 1n+bp:
      %Nats.mul_comm(Nat.div(i, 1n+bp), 1n+bp) : {Nat.add(_, Nat.mod(i, 1n+bp)) == i : Nat}
      go_total(i, bp, 0n, 0n, bp, Equal.sym(Nat, bp, Nat.add(bp, 0n), add_zero_left(bp)))

def mod_one_go(+n: Nat, +d: Nat) -> {Pair.snd(Nat, Nat, Nat.divmod.go(n, 0n, d, 0n)) == 0n : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+np:
      mod_one_go(np, 1n+d)

def mod_one(+i: Nat) -> {Nat.mod(i, 1n) == 0n : Nat}:
  mod_one_go(i, 0n)

# Rows whose left stride is the output's on every axis, the last one 1:
# the left walk of every position is the position.
def walk_left_same(+rows: List<&2, Shape.Row>, +same: {Elementwise.same_left(rows) == True{} : Bool}, +i: Nat) ->
  {Shape.walk_left(rows, i) == i : Nat}:
  match rows:
    case Shape.Row{+stride, +left, +right} <> Nil{}:
      +unit = Arith.eq_true(stride, 1n, Arith.and_left(Nat.is_eq(stride, 1n), Nat.is_eq(left, 1n), same))
      +step = Arith.eq_true(left, 1n, Arith.and_right(Nat.is_eq(stride, 1n), Nat.is_eq(left, 1n), same))
      %Equal.sym(Nat, left, stride, Equal.trans(Nat, left, 1n, stride, step, Equal.sym(Nat, stride, 1n, unit))) :
        {Nat.add(Nat.mul(_, Nat.div(i, stride)), 0n) == i : Nat}
      %Equal.sym(Nat, stride, 1n, unit) : {Nat.add(Nat.mul(_, Nat.div(i, _)), 0n) == i : Nat}
      %mod_one(i) : {Nat.add(Nat.mul(1n, Nat.div(i, 1n)), _) == i : Nat}
      div_mod(i, 1n)
    case Shape.Row{+stride, +left, +right} <> +next <> +rest:
      +equal = Arith.eq_true(left, stride, Arith.and_left(Nat.is_eq(left, stride), Elementwise.same_left(next <> rest), same))
      %Equal.sym(Nat, left, stride, equal) : {Nat.add(Nat.mul(_, Nat.div(i, stride)), Shape.walk_left(next <> rest, Nat.mod(i, stride))) == i : Nat}
      %Equal.sym(Nat, Shape.walk_left(next <> rest, Nat.mod(i, stride)), Nat.mod(i, stride),
        walk_left_same(next <> rest, Arith.and_right(Nat.is_eq(left, stride), Elementwise.same_left(next <> rest), same), Nat.mod(i, stride))) :
        {Nat.add(Nat.mul(stride, Nat.div(i, stride)), _) == i : Nat}
      div_mod(i, stride)
    case Nil{}:
      Empty.absurd({Shape.walk_left(Nil{}, i) == i : Nat}, Arith.clash(same))

def walk_right_same(+rows: List<&2, Shape.Row>, +same: {Elementwise.same_right(rows) == True{} : Bool}, +i: Nat) ->
  {Shape.walk_right(rows, i) == i : Nat}:
  match rows:
    case Shape.Row{+stride, +left, +right} <> Nil{}:
      +unit = Arith.eq_true(stride, 1n, Arith.and_left(Nat.is_eq(stride, 1n), Nat.is_eq(right, 1n), same))
      +step = Arith.eq_true(right, 1n, Arith.and_right(Nat.is_eq(stride, 1n), Nat.is_eq(right, 1n), same))
      %Equal.sym(Nat, right, stride, Equal.trans(Nat, right, 1n, stride, step, Equal.sym(Nat, stride, 1n, unit))) :
        {Nat.add(Nat.mul(_, Nat.div(i, stride)), 0n) == i : Nat}
      %Equal.sym(Nat, stride, 1n, unit) : {Nat.add(Nat.mul(_, Nat.div(i, _)), 0n) == i : Nat}
      %mod_one(i) : {Nat.add(Nat.mul(1n, Nat.div(i, 1n)), _) == i : Nat}
      div_mod(i, 1n)
    case Shape.Row{+stride, +left, +right} <> +next <> +rest:
      +equal = Arith.eq_true(right, stride, Arith.and_left(Nat.is_eq(right, stride), Elementwise.same_right(next <> rest), same))
      %Equal.sym(Nat, right, stride, equal) : {Nat.add(Nat.mul(_, Nat.div(i, stride)), Shape.walk_right(next <> rest, Nat.mod(i, stride))) == i : Nat}
      %Equal.sym(Nat, Shape.walk_right(next <> rest, Nat.mod(i, stride)), Nat.mod(i, stride),
        walk_right_same(next <> rest, Arith.and_right(Nat.is_eq(right, stride), Elementwise.same_right(next <> rest), same), Nat.mod(i, stride))) :
        {Nat.add(Nat.mul(stride, Nat.div(i, stride)), _) == i : Nat}
      div_mod(i, stride)
    case Nil{}:
      Empty.absurd({Shape.walk_right(Nil{}, i) == i : Nat}, Arith.clash(same))

# Rows whose left stride is 0 on every axis: every position walks to 0.
def walk_left_zero(+rows: List<&2, Shape.Row>, +zero: {Elementwise.zero_left(rows) == True{} : Bool}, +i: Nat) ->
  {Shape.walk_left(rows, i) == 0n : Nat}:
  match rows:
    case Shape.Row{+stride, +left, +right} <> +rest:
      +none = Arith.eq_true(left, 0n, Arith.and_left(Nat.is_eq(left, 0n), Elementwise.zero_left(rest), zero))
      %Equal.sym(Nat, left, 0n, none) : {Nat.add(Nat.mul(_, Nat.div(i, stride)), Shape.walk_left(rest, Nat.mod(i, stride))) == 0n : Nat}
      walk_left_zero(rest, Arith.and_right(Nat.is_eq(left, 0n), Elementwise.zero_left(rest), zero), Nat.mod(i, stride))
    case Nil{}:
      {==}

def walk_right_zero(+rows: List<&2, Shape.Row>, +zero: {Elementwise.zero_right(rows) == True{} : Bool}, +i: Nat) ->
  {Shape.walk_right(rows, i) == 0n : Nat}:
  match rows:
    case Shape.Row{+stride, +left, +right} <> +rest:
      +none = Arith.eq_true(right, 0n, Arith.and_left(Nat.is_eq(right, 0n), Elementwise.zero_right(rest), zero))
      %Equal.sym(Nat, right, 0n, none) : {Nat.add(Nat.mul(_, Nat.div(i, stride)), Shape.walk_right(rest, Nat.mod(i, stride))) == 0n : Nat}
      walk_right_zero(rest, Arith.and_right(Nat.is_eq(right, 0n), Elementwise.zero_right(rest), zero), Nat.mod(i, stride))
    case Nil{}:
      {==}

# The slot an input reads for output cell `position`.
def left_at(+slot: Elementwise.Slot, -left: List<&2, Nat>, -right: List<&2, Nat>, -output: List<&2, Nat>,
  +map: Shape.Index<left, right, output>, +position: Nat) -> Nat:
  match slot:
    case Elementwise.OwnSlot{}:
      position
    case Elementwise.FirstSlot{}:
      0n
    case Elementwise.WalkedSlot{}:
      Elementwise.left_walk(left, right, output, map, position)

def right_at(+slot: Elementwise.Slot, -left: List<&2, Nat>, -right: List<&2, Nat>, -output: List<&2, Nat>,
  +map: Shape.Index<left, right, output>, +position: Nat) -> Nat:
  match slot:
    case Elementwise.OwnSlot{}:
      position
    case Elementwise.FirstSlot{}:
      0n
    case Elementwise.WalkedSlot{}:
      Elementwise.right_walk(left, right, output, map, position)

# The slot `left_slot` picks is where the walk of every position lands.
def left_pick(-le: List<&2, Nat>, -re: List<&2, Nat>, -oe: List<&2, Nat>, +rows: List<&2, Shape.Row>,
  +exact: {rows == Shape.index_table(le, re, oe) : List<&2, Shape.Row>},
  same: Bool, +same_exact: {same == Elementwise.same_left(rows) : Bool},
  zero: Bool, +zero_exact: {zero == Elementwise.zero_left(rows) : Bool}, +i: Nat) ->
  {left_at(Elementwise.slot_of(same, zero), le, re, oe, Shape.Indexed{rows, exact}, i) == Shape.walk_left(rows, i) : Nat}:
  match same zero:
    case True{} zero:
      Equal.sym(Nat, Shape.walk_left(rows, i), i, walk_left_same(rows, Equal.sym(Bool, True{}, Elementwise.same_left(rows), same_exact), i))
    case False{} True{}:
      Equal.sym(Nat, Shape.walk_left(rows, i), 0n, walk_left_zero(rows, Equal.sym(Bool, True{}, Elementwise.zero_left(rows), zero_exact), i))
    case False{} False{}:
      {==}

def right_pick(-le: List<&2, Nat>, -re: List<&2, Nat>, -oe: List<&2, Nat>, +rows: List<&2, Shape.Row>,
  +exact: {rows == Shape.index_table(le, re, oe) : List<&2, Shape.Row>},
  same: Bool, +same_exact: {same == Elementwise.same_right(rows) : Bool},
  zero: Bool, +zero_exact: {zero == Elementwise.zero_right(rows) : Bool}, +i: Nat) ->
  {right_at(Elementwise.slot_of(same, zero), le, re, oe, Shape.Indexed{rows, exact}, i) == Shape.walk_right(rows, i) : Nat}:
  match same zero:
    case True{} zero:
      Equal.sym(Nat, Shape.walk_right(rows, i), i, walk_right_same(rows, Equal.sym(Bool, True{}, Elementwise.same_right(rows), same_exact), i))
    case False{} True{}:
      Equal.sym(Nat, Shape.walk_right(rows, i), 0n, walk_right_zero(rows, Equal.sym(Bool, True{}, Elementwise.zero_right(rows), zero_exact), i))
    case False{} False{}:
      {==}

def left_index(-le: List<&2, Nat>, -re: List<&2, Nat>, -oe: List<&2, Nat>, +map: Shape.Index<le, re, oe>, +i: Nat) ->
  {left_at(Elementwise.left_slot(le, re, oe, map), le, re, oe, map, i) == Elementwise.left_walk(le, re, oe, map, i) : Nat}:
  match map:
    case Shape.Indexed{+rows, +exact}:
      left_pick(le, re, oe, rows, exact, Elementwise.same_left(rows), {==}, Elementwise.zero_left(rows), {==}, i)
    case Shape.Unindexed{+zero}:
      {==}

def right_index(-le: List<&2, Nat>, -re: List<&2, Nat>, -oe: List<&2, Nat>, +map: Shape.Index<le, re, oe>, +i: Nat) ->
  {right_at(Elementwise.right_slot(le, re, oe, map), le, re, oe, map, i) == Elementwise.right_walk(le, re, oe, map, i) : Nat}:
  match map:
    case Shape.Indexed{+rows, +exact}:
      right_pick(le, re, oe, rows, exact, Elementwise.same_right(rows), {==}, Elementwise.zero_right(rows), {==}, i)
    case Shape.Unindexed{+zero}:
      {==}

# An inside cell whose slots are known is the cell with the walk.
def direct_same(+operation: Elementwise.Operation, -le: List<&2, Nat>, -re: List<&2, Nat>, -oe: List<&2, Nat>,
  +map: Shape.Index<le, re, oe>, +left: Elementwise.Slot, +left_exact: {left == Elementwise.left_slot(le, re, oe, map) : Elementwise.Slot},
  +right: Elementwise.Slot, +right_exact: {right == Elementwise.right_slot(le, re, oe, map) : Elementwise.Slot},
  +position: Nat, -la: Array<F32>, -ra: Array<F32>) ->
  {Elementwise.direct_cell(operation, left_at(left, le, re, oe, map, position), right_at(right, le, re, oe, map, position), la, ra) ==
    Elementwise.cell(operation, le, re, oe, map, position, la, ra, True{}) : F32 & (Array<F32> & Array<F32>)}:
  +lw = Elementwise.left_walk(le, re, oe, map, position)
  +rw = Elementwise.right_walk(le, re, oe, map, position)
  %Equal.sym(Nat, left_at(left, le, re, oe, map, position), lw,
    %Equal.sym(Elementwise.Slot, left, Elementwise.left_slot(le, re, oe, map), left_exact) :
      {left_at(_, le, re, oe, map, position) == lw : Nat}
    left_index(le, re, oe, map, position)) :
    {Elementwise.cell_left(operation, ra, U32.from_nat(right_at(right, le, re, oe, map, position)), Array.get(F32, la, U32.from_nat(_))) ==
      Elementwise.cell_left(operation, ra, U32.from_nat(rw), Array.get(F32, la, U32.from_nat(lw))) : F32 & (Array<F32> & Array<F32>)}
  %Equal.sym(Nat, right_at(right, le, re, oe, map, position), rw,
    %Equal.sym(Elementwise.Slot, right, Elementwise.right_slot(le, re, oe, map), right_exact) :
      {right_at(_, le, re, oe, map, position) == rw : Nat}
    right_index(le, re, oe, map, position)) :
    {Elementwise.cell_left(operation, ra, U32.from_nat(_), Array.get(F32, la, U32.from_nat(lw))) ==
      Elementwise.cell_left(operation, ra, U32.from_nat(rw), Array.get(F32, la, U32.from_nat(lw))) : F32 & (Array<F32> & Array<F32>)}
  {==}

# With the slots `left_slot` and `right_slot` pick, the cell of a build is
# the cell with the walk.
def cell_by_same(+operation: Elementwise.Operation, -le: List<&2, Nat>, -re: List<&2, Nat>, -oe: List<&2, Nat>,
  +map: Shape.Index<le, re, oe>, inside: Bool, left: Elementwise.Slot, +left_exact: {left == Elementwise.left_slot(le, re, oe, map) : Elementwise.Slot},
  right: Elementwise.Slot, +right_exact: {right == Elementwise.right_slot(le, re, oe, map) : Elementwise.Slot},
  +position: Nat, -la: Array<F32>, -ra: Array<F32>) ->
  {Elementwise.cell_by(inside, left, right, operation, le, re, oe, map, position, la, ra) ==
    Elementwise.cell(operation, le, re, oe, map, position, la, ra, inside) : F32 & (Array<F32> & Array<F32>)}:
  match inside left right:
    case False{} left right:
      {==}
    case True{} Elementwise.OwnSlot{} Elementwise.OwnSlot{}:
      direct_same(operation, le, re, oe, map, Elementwise.OwnSlot{}, left_exact, Elementwise.OwnSlot{}, right_exact, position, la, ra)
    case True{} Elementwise.OwnSlot{} Elementwise.FirstSlot{}:
      direct_same(operation, le, re, oe, map, Elementwise.OwnSlot{}, left_exact, Elementwise.FirstSlot{}, right_exact, position, la, ra)
    case True{} Elementwise.OwnSlot{} Elementwise.WalkedSlot{}:
      direct_same(operation, le, re, oe, map, Elementwise.OwnSlot{}, left_exact, Elementwise.WalkedSlot{}, right_exact, position, la, ra)
    case True{} Elementwise.FirstSlot{} Elementwise.OwnSlot{}:
      direct_same(operation, le, re, oe, map, Elementwise.FirstSlot{}, left_exact, Elementwise.OwnSlot{}, right_exact, position, la, ra)
    case True{} Elementwise.FirstSlot{} Elementwise.FirstSlot{}:
      direct_same(operation, le, re, oe, map, Elementwise.FirstSlot{}, left_exact, Elementwise.FirstSlot{}, right_exact, position, la, ra)
    case True{} Elementwise.FirstSlot{} Elementwise.WalkedSlot{}:
      direct_same(operation, le, re, oe, map, Elementwise.FirstSlot{}, left_exact, Elementwise.WalkedSlot{}, right_exact, position, la, ra)
    case True{} Elementwise.WalkedSlot{} Elementwise.OwnSlot{}:
      direct_same(operation, le, re, oe, map, Elementwise.WalkedSlot{}, left_exact, Elementwise.OwnSlot{}, right_exact, position, la, ra)
    case True{} Elementwise.WalkedSlot{} Elementwise.FirstSlot{}:
      direct_same(operation, le, re, oe, map, Elementwise.WalkedSlot{}, left_exact, Elementwise.FirstSlot{}, right_exact, position, la, ra)
    case True{} Elementwise.WalkedSlot{} Elementwise.WalkedSlot{}:
      {==}

# The cell function of a build is `Elementwise.cell` of its plan.
def cell_at_walk(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +position: Nat, -la: Array<F32>, -ra: Array<F32>,
  inside: Bool) ->
  {Elementwise.cell_at(Elementwise.walk(Elementwise.Request{operation, plan}), position, (la, ra), inside) ==
    Elementwise.cell(operation, Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)),
      Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position, la, ra, inside) : F32 & (Array<F32> & Array<F32>)}:
  +le = Shape.extents(Shape.left_shape(plan))
  +re = Shape.extents(Shape.right_shape(plan))
  +oe = Shape.extents(Shape.output_shape(plan))
  cell_by_same(operation, le, re, oe, Shape.index_map(plan), inside, Elementwise.left_slot(le, re, oe, Shape.index_map(plan)), {==},
    Elementwise.right_slot(le, re, oe, Shape.index_map(plan)), {==}, position, la, ra)
