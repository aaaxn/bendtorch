import Base
import ../../shape/core.bend as Shape
import ../../storage/model.bend as Model
import ../../storage/main.bend as Storage
import ../../storage/refinement.bend as Refinement
import ../../storage/clone.bend as Clone
import ../../storage/api.bend as Api
import ../../storage/arith.bend as Arith
import ../elementwise.bend as Elementwise
import ../parallel.bend as Parallel
import ../word-conversion.bend as Conversion
import ../read-contract.bend as Reads
import ./replayed.bend as Replayed
import ./broadcast.bend as Broadcast
import ./broadcast-proof.bend as BroadcastProof

# The parallel elementwise build, walked with live code. The two input
# arrays are `Api.arr` of a history of storage effects, so every array the
# build reads or clones is a term of reusable data, and each step of the proof
# can name it again. `spec` is the balanced tree the build must return: cell
# `base` holds `scalar(operation, left word, right word)` below `count` and
# `+0.0` above it.

def left_index(+plan: Shape.BroadcastPlan, +position: Nat) -> U32:
  U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position))

def right_index(+plan: Shape.BroadcastPlan, +position: Nat) -> U32:
  U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position))

def word(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +nl: Nat, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +nr: Nat, +position: Nat) -> F32:
  Elementwise.scalar(operation, Api.current(ha, left_index(plan, position), fa), Api.current(hb, right_index(plan, position), fb))

def leaf(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +nl: Nat, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +nr: Nat, +position: Nat, inside: Bool) -> F32:
  match inside:
    case False{}:
      0.0
    case True{}:
      word(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, position)

def spec(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +nl: Nat, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +nr: Nat, +depth: Nat, +base: Nat) -> Array<F32>:
  match depth:
    case 0n:
      ALeaf{leaf(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, base, Nat.is_lt(base, count))}
    case 1n+inner:
      ANode{spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, base), spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, Nat.add(Model.capacity(inner), base))}

def spec_balanced(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +nl: Nat, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +nr: Nat, +depth: Nat, +base: Nat) -> Model.Balanced(F32, depth, spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, depth, base)):
  match depth:
    case 0n:
      Unit{}
    case 1n+inner:
      (spec_balanced(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, base), spec_balanced(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, Nat.add(Model.capacity(inner), base)))

def TreeEq(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +nl: Nat, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +nr: Nat, +depth: Nat, +base: Nat, -result: Array<F32> & (Array<F32> & Array<F32>)) -> Type:
  {Pair.fst(Array<F32>, Array<F32> & Array<F32>, result) == spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, depth, base) : Array<F32>} &
    ({Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == Api.arr(ha, dl, fa) : Array<F32>} &
      {Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == Api.arr(hb, dr, fb) : Array<F32>})

# Every write in the history lands below `n`: `Api.Valid` as a Bool, so a
# proof of it can be used many times.
def written(+n: Nat, +history: List<&2, Api.Effect>) -> Bool:
  match history:
    case Nil{}:
      True{}
    case +effect <> +past:
      match effect:
        case Api.Wrote{+i, w}:
          Bool.and(Nat.is_lt(U32.to_nat(i), n), written(n, past))
        case Api.Kept{i}:
          written(n, past)
        case Api.Left{}:
          written(n, past)
        case Api.Right{}:
          written(n, past)

def valid_of(+n: Nat, +history: List<&2, Api.Effect>, +all: {written(n, history) == True{} : Bool}) -> Api.Valid(n, history):
  match history:
    case Nil{}:
      Unit{}
    case +effect <> +past:
      match effect:
        case Api.Wrote{+i, +w}:
          (Arith.and_left(Nat.is_lt(U32.to_nat(i), n), written(n, past), all),
            valid_of(n, past, Arith.and_right(Nat.is_lt(U32.to_nat(i), n), written(n, past), all)))
        case Api.Kept{+i}:
          valid_of(n, past, all)
        case Api.Left{}:
          valid_of(n, past, all)
        case Api.Right{}:
          valid_of(n, past, all)

# A read of `arr(h)` keeps the array and returns the word `current` names.
def got_kept(+d: Nat, +h: List<&2, Api.Effect>, +f: F32, +i: U32, -pair: Array<F32> & F32,
  exact: {pair == Array.get(F32, Api.arr(h, d, f), i) : Array<F32> & F32}) ->
  {Refinement.first(F32, pair) == Api.arr(h, d, f) : Array<F32>}:
  Equal.trans(Array<F32>, Refinement.first(F32, pair), Refinement.first(F32, Array.get(F32, Api.arr(h, d, f), i)), Api.arr(h, d, f),
    Equal.cong(Array<F32> & F32, Array<F32>, p => Refinement.first(F32, p), pair, Array.get(F32, Api.arr(h, d, f), i), exact),
    Refinement.law_get_retains(F32, d, Api.arr(h, d, f), i, Api.bal(h, d, f)))

def some_word(-word: F32, -other: F32, same: {Some{word} == Some{other} : Maybe<&2, F32>}) -> {word == other : F32}:
  Equal.cong(Maybe<&2, F32>, F32, found => Maybe.default(&2, F32, found, word), Some{word}, Some{other}, same)

def got_value(+d: Nat, +h: List<&2, Api.Effect>, +f: F32, +n: Nat, +i: U32, -pair: Array<F32> & F32,
  exact: {pair == Array.get(F32, Api.arr(h, d, f), i) : Array<F32> & F32},
  +fits: {Storage.bounds(d, n) == True{} : Bool}, +inside: {Nat.is_lt(U32.to_nat(i), n) == True{} : Bool}, +valid: {written(n, h) == True{} : Bool}) ->
  {Refinement.second(F32, pair) == Api.current(h, i, f) : F32}:
  some_word(Refinement.second(F32, pair), Api.current(h, i, f),
    Equal.trans(Maybe<&2, F32>, Some{Refinement.second(F32, pair)}, Some{Refinement.second(F32, Array.get(F32, Api.arr(h, d, f), i))},
      Some{Api.current(h, i, f)},
      Equal.cong(Array<F32> & F32, Maybe<&2, F32>, p => Some{Refinement.second(F32, p)}, pair, Array.get(F32, Api.arr(h, d, f), i), exact),
      Equal.trans(Maybe<&2, F32>, Some{Refinement.second(F32, Array.get(F32, Api.arr(h, d, f), i))},
        Model.observe(F32, d, Api.arr(h, d, f), Refinement.route(d, i)), Some{Api.current(h, i, f)},
        Refinement.law_get_refines(F32, d, Api.arr(h, d, f), i, Api.bal(h, d, f)),
        Equal.sym(Maybe<&2, F32>, Some{Api.current(h, i, f)}, Model.observe(F32, d, Api.arr(h, d, f), Refinement.route(d, i)),
          Api.model(h, d, n, f, i, fits, inside, valid_of(n, h, valid))))))

def walk_left_same(-left: List<&2, Nat>, -right: List<&2, Nat>, -output: List<&2, Nat>,
  +map: Shape.Index<left, right, output>, +index: Nat) ->
  {Elementwise.left_walk(left, right, output, map, index) == Broadcast.left_walk(left, right, output, map, index) : Nat}:
  match map:
    case Shape.Indexed{rows, exact}:
      {==}
    case Shape.Unindexed{zero}:
      {==}

def walk_right_same(-left: List<&2, Nat>, -right: List<&2, Nat>, -output: List<&2, Nat>,
  +map: Shape.Index<left, right, output>, +index: Nat) ->
  {Elementwise.right_walk(left, right, output, map, index) == Broadcast.right_walk(left, right, output, map, index) : Nat}:
  match map:
    case Shape.Indexed{rows, exact}:
      {==}
    case Shape.Unindexed{zero}:
      {==}

def output_below(+plan: Shape.BroadcastPlan, +count: Nat, +position: Nat,
  +count_exact: {count == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat},
  +decided: {True{} == Nat.is_lt(position, count) : Bool}) ->
  {Nat.is_lt(position, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}:
  %count_exact : {Nat.is_lt(position, _) == True{} : Bool}
  Equal.sym(Bool, True{}, Nat.is_lt(position, count), decided)

# A walked index below the side's element count is a word index below the
# side's storage length.
def word_below(+side: Shape.Shape, +n: Nat, +walk: Nat,
  +side_count: {n == Shape.logical_numel(Shape.dimensions(side)) : Nat},
  +below: {Nat.is_lt(walk, Shape.logical_numel(Shape.dimensions(side))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(walk)), n) == True{} : Bool}:
  +within = Conversion.lt_le(walk, U32.to_nat(Shape.count(side)),
    %Equal.sym(Nat, U32.to_nat(Shape.count(side)), Shape.logical_numel(Shape.dimensions(side)), Reads.count_exact(side)) :
      {Nat.is_lt(walk, _) == True{} : Bool}
    below)
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(walk)), walk, Conversion.natural_word_exact(walk, Shape.count(side), within)) :
    {Nat.is_lt(_, n) == True{} : Bool}
  %Equal.sym(Nat, n, Shape.logical_numel(Shape.dimensions(side)), side_count) : {Nat.is_lt(walk, _) == True{} : Bool}
  below

def left_below(+plan: Shape.BroadcastPlan, +count: Nat, +nl: Nat, +position: Nat,
  +count_exact: {count == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat},
  +left_count: {nl == Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat},
  +decided: {True{} == Nat.is_lt(position, count) : Bool}) ->
  {Nat.is_lt(U32.to_nat(left_index(plan, position)), nl) == True{} : Bool}:
  word_below(Shape.left_shape(plan), nl, Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position), left_count,
    %Equal.sym(Nat, Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position), Broadcast.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position), walk_left_same(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position)) : {Nat.is_lt(_, Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan)))) == True{} : Bool}
    BroadcastProof.left_walk_bound(plan, Shape.index_map(plan), position, output_below(plan, count, position, count_exact, decided)))

def right_below(+plan: Shape.BroadcastPlan, +count: Nat, +nr: Nat, +position: Nat,
  +count_exact: {count == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat},
  +right_count: {nr == Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat},
  +decided: {True{} == Nat.is_lt(position, count) : Bool}) ->
  {Nat.is_lt(U32.to_nat(right_index(plan, position)), nr) == True{} : Bool}:
  word_below(Shape.right_shape(plan), nr, Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position), right_count,
    %Equal.sym(Nat, Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position), Broadcast.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position), walk_right_same(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position)) : {Nat.is_lt(_, Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))) == True{} : Bool}
    BroadcastProof.right_walk_bound(plan, Shape.index_map(plan), position, output_below(plan, count, position, count_exact, decided)))

def cell_right_eq(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +nl: Nat, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +nr: Nat, +lfits: {Storage.bounds(dl, nl) == True{} : Bool}, +rfits: {Storage.bounds(dr, nr) == True{} : Bool}, +lvalid: {written(nl, ha) == True{} : Bool}, +rvalid: {written(nr, hb) == True{} : Bool}, +count_exact: {count == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat}, +left_count: {nl == Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat}, +right_count: {nr == Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat}, +base: Nat, +decided: {True{} == Nat.is_lt(base, count) : Bool},
  -x: F32, -kept: Array<F32>, pair: Array<F32> & F32,
  +exact: {pair == Array.get(F32, Api.arr(hb, dr, fb), right_index(plan, base)) : Array<F32> & F32},
  ex: {x == Api.current(ha, left_index(plan, base), fa) : F32}, ek: {kept == Api.arr(ha, dl, fa) : Array<F32>}) ->
  TreeEq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, 0n, base, Parallel.leaf(~(Array<F32> & Array<F32>), Elementwise.cell_right(operation, x, kept, pair))):
  (other, y) = pair
  +ey = got_value(dr, hb, fb, nr, right_index(plan, base), (other, y), exact, rfits,
    right_below(plan, count, nr, base, count_exact, right_count, decided), rvalid)
  (%decided :
     {ALeaf{Elementwise.scalar(operation, x, y)} == ALeaf{leaf(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, base, _)} : Array<F32>}
   %Equal.sym(F32, x, Api.current(ha, left_index(plan, base), fa), ex) : {ALeaf{Elementwise.scalar(operation, _, y)} == ALeaf{word(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, base)} : Array<F32>}
   %Equal.sym(F32, y, Api.current(hb, right_index(plan, base), fb), ey) : {ALeaf{Elementwise.scalar(operation, Api.current(ha, left_index(plan, base), fa), _)} == ALeaf{word(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, base)} : Array<F32>}
   {==},
   (ek, got_kept(dr, hb, fb, right_index(plan, base), (other, y), exact)))

def cell_left_eq(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +nl: Nat, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +nr: Nat, +lfits: {Storage.bounds(dl, nl) == True{} : Bool}, +rfits: {Storage.bounds(dr, nr) == True{} : Bool}, +lvalid: {written(nl, ha) == True{} : Bool}, +rvalid: {written(nr, hb) == True{} : Bool}, +count_exact: {count == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat}, +left_count: {nl == Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat}, +right_count: {nr == Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat}, +base: Nat, +decided: {True{} == Nat.is_lt(base, count) : Bool},
  pair: Array<F32> & F32, +exact: {pair == Array.get(F32, Api.arr(ha, dl, fa), left_index(plan, base)) : Array<F32> & F32}) ->
  TreeEq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, 0n, base, Parallel.leaf(~(Array<F32> & Array<F32>), Elementwise.cell_left(operation, Api.arr(hb, dr, fb), right_index(plan, base), pair))):
  (kept, x) = pair
  cell_right_eq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, lfits, rfits, lvalid, rvalid, count_exact, left_count, right_count, base, decided, x, kept, Array.get(F32, Api.arr(hb, dr, fb), right_index(plan, base)), {==},
    got_value(dl, ha, fa, nl, left_index(plan, base), (kept, x), exact, lfits,
      left_below(plan, count, nl, base, count_exact, left_count, decided), lvalid),
    got_kept(dl, ha, fa, left_index(plan, base), (kept, x), exact))

def cell_eq(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +nl: Nat, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +nr: Nat, +lfits: {Storage.bounds(dl, nl) == True{} : Bool}, +rfits: {Storage.bounds(dr, nr) == True{} : Bool}, +lvalid: {written(nl, ha) == True{} : Bool}, +rvalid: {written(nr, hb) == True{} : Bool}, +count_exact: {count == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat}, +left_count: {nl == Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat}, +right_count: {nr == Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat}, +base: Nat, inside: Bool, decided: {inside == Nat.is_lt(base, count) : Bool}) ->
  TreeEq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, 0n, base, Parallel.leaf(~(Array<F32> & Array<F32>), Elementwise.cell(operation, Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb), inside))):
  match inside:
    case False{}:
      (%decided : {ALeaf{0.0} == ALeaf{leaf(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, base, _)} : Array<F32>}
       {==}, ({==}, {==}))
    case True{}:
      cell_left_eq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, lfits, rfits, lvalid, rvalid, count_exact, left_count, right_count, base, decided, Array.get(F32, Api.arr(ha, dl, fa), left_index(plan, base)), {==})

def tree_join_eq(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +nl: Nat, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +nr: Nat, +inner: Nat, +base: Nat, -a: Array<F32>, high: Array<F32> & (Array<F32> & Array<F32>),
  ea: {a == spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, base) : Array<F32>},
  facts: TreeEq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, Nat.add(Model.capacity(inner), base), high)) ->
  TreeEq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, 1n+inner, base, Parallel.join(~(Array<F32> & Array<F32>), a, high)):
  (b, inputs) = high
  (eb, kept) = facts
  (%Equal.sym(Array<F32>, a, spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, base), ea) : {ANode{_, b} == ANode{spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, base), spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, Nat.add(Model.capacity(inner), base))} : Array<F32>}
   %Equal.sym(Array<F32>, b, spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, Nat.add(Model.capacity(inner), base)), eb) : {ANode{spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, base), _} == ANode{spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, base), spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, Nat.add(Model.capacity(inner), base))} : Array<F32>}
   {==}, kept)

def tree_node_eq(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +nl: Nat, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +nr: Nat, +inner: Nat, +base: Nat, low: Array<F32> & (Array<F32> & Array<F32>), facts: TreeEq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, base, low),
  next: TreeEq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, Nat.add(Model.capacity(inner), base), Parallel.tree(~Elementwise.Walk, ~Array<F32>, ~Array<F32>, ~Elementwise.cell_at, Elementwise.walk(Elementwise.Request{operation, plan}), inner, count, Nat.add(Model.capacity(inner), base), (Api.arr(ha, dl, fa), Api.arr(hb, dr, fb))))) ->
  TreeEq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, 1n+inner, base, Parallel.node(~Array<F32>, ~Array<F32>, l => r => Parallel.tree(~Elementwise.Walk, ~Array<F32>, ~Array<F32>, ~Elementwise.cell_at, Elementwise.walk(Elementwise.Request{operation, plan}), inner, count, Nat.add(Model.capacity(inner), base), (l, r)), low)):
  (a, inputs) = low
  (la, ra) = inputs
  (ea, kept) = facts
  (el, er) = kept
  %Equal.sym(Array<F32>, la, Api.arr(ha, dl, fa), el) :
    TreeEq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, 1n+inner, base, Parallel.join(~(Array<F32> & Array<F32>), a, Parallel.tree(~Elementwise.Walk, ~Array<F32>, ~Array<F32>, ~Elementwise.cell_at, Elementwise.walk(Elementwise.Request{operation, plan}), inner, count, Nat.add(Model.capacity(inner), base), (_, ra))))
  %Equal.sym(Array<F32>, ra, Api.arr(hb, dr, fb), er) :
    TreeEq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, 1n+inner, base, Parallel.join(~(Array<F32> & Array<F32>), a, Parallel.tree(~Elementwise.Walk, ~Array<F32>, ~Array<F32>, ~Elementwise.cell_at, Elementwise.walk(Elementwise.Request{operation, plan}), inner, count, Nat.add(Model.capacity(inner), base), (Api.arr(ha, dl, fa), _))))
  tree_join_eq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, base, a, Parallel.tree(~Elementwise.Walk, ~Array<F32>, ~Array<F32>, ~Elementwise.cell_at, Elementwise.walk(Elementwise.Request{operation, plan}), inner, count, Nat.add(Model.capacity(inner), base), (Api.arr(ha, dl, fa), Api.arr(hb, dr, fb))), ea, next)

def tree_eq(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +nl: Nat, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +nr: Nat, +lfits: {Storage.bounds(dl, nl) == True{} : Bool}, +rfits: {Storage.bounds(dr, nr) == True{} : Bool}, +lvalid: {written(nl, ha) == True{} : Bool}, +rvalid: {written(nr, hb) == True{} : Bool}, +count_exact: {count == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat}, +left_count: {nl == Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat}, +right_count: {nr == Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat}, +depth: Nat, +base: Nat) ->
  TreeEq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, depth, base, Parallel.tree(~Elementwise.Walk, ~Array<F32>, ~Array<F32>, ~Elementwise.cell_at, Elementwise.walk(Elementwise.Request{operation, plan}), depth, count, base, (Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)))):
  match depth:
    case 0n:
      cell_eq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, lfits, rfits, lvalid, rvalid, count_exact, left_count, right_count, base, Nat.is_lt(base, count), {==})
    case 1n+inner:
      tree_node_eq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, base, Parallel.tree(~Elementwise.Walk, ~Array<F32>, ~Array<F32>, ~Elementwise.cell_at, Elementwise.walk(Elementwise.Request{operation, plan}), inner, count, base, (Api.arr(ha, dl, fa), Api.arr(hb, dr, fb))), tree_eq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, lfits, rfits, lvalid, rvalid, count_exact, left_count, right_count, inner, base),
        tree_eq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, lfits, rfits, lvalid, rvalid, count_exact, left_count, right_count, inner, Nat.add(Model.capacity(inner), base)))

def tree_first(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +nl: Nat, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +nr: Nat, +depth: Nat, +base: Nat, -result: Array<F32> & (Array<F32> & Array<F32>), facts: TreeEq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, depth, base, result)) ->
  {Pair.fst(Array<F32>, Array<F32> & Array<F32>, result) == spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, depth, base) : Array<F32>}:
  (first, kept) = facts
  first

def clone_first_eq(+d: Nat, +h: List<&2, Api.Effect>, +f: F32, -pair: Array<F32> & Array<F32>,
  exact: {pair == Array.clone(F32, Api.arr(h, d, f)) : Array<F32> & Array<F32>}) ->
  {Pair.fst(Array<F32>, Array<F32>, pair) == Api.arr(h, d, f) : Array<F32>}:
  Equal.trans(Array<F32>, Pair.fst(Array<F32>, Array<F32>, pair), Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, Api.arr(h, d, f))), Api.arr(h, d, f),
    Equal.cong(Array<F32> & Array<F32>, Array<F32>, p => Pair.fst(Array<F32>, Array<F32>, p), pair, Array.clone(F32, Api.arr(h, d, f)), exact),
    Clone.first_equal(F32, d, Api.arr(h, d, f), Clone.certify(F32, d, Api.arr(h, d, f), Api.bal(h, d, f))))

def clone_second_eq(+d: Nat, +h: List<&2, Api.Effect>, +f: F32, -pair: Array<F32> & Array<F32>,
  exact: {pair == Array.clone(F32, Api.arr(h, d, f)) : Array<F32> & Array<F32>}) ->
  {Pair.snd(Array<F32>, Array<F32>, pair) == Api.arr(h, d, f) : Array<F32>}:
  Equal.trans(Array<F32>, Pair.snd(Array<F32>, Array<F32>, pair), Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, Api.arr(h, d, f))), Api.arr(h, d, f),
    Equal.cong(Array<F32> & Array<F32>, Array<F32>, p => Pair.snd(Array<F32>, Array<F32>, p), pair, Array.clone(F32, Api.arr(h, d, f)), exact),
    Clone.second_equal(F32, d, Api.arr(h, d, f), Clone.certify(F32, d, Api.arr(h, d, f), Api.bal(h, d, f))))

def fork_eq(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +nl: Nat, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +nr: Nat, +inner: Nat, +rest: Nat, +base: Nat,
  low: {Elementwise.build(inner, rest, operation, plan, count, base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)) == spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, base) : Array<F32>},
  high: {Elementwise.build(inner, rest, operation, plan, count, Nat.add(Model.capacity(inner), base), Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)) == spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, Nat.add(Model.capacity(inner), base)) : Array<F32>},
  lp: Array<F32> & Array<F32>, +le: {lp == Array.clone(F32, Api.arr(ha, dl, fa)) : Array<F32> & Array<F32>},
  rp: Array<F32> & Array<F32>, +re: {rp == Array.clone(F32, Api.arr(hb, dr, fb)) : Array<F32> & Array<F32>}) ->
  {Parallel.fork(~Array<F32>, ~Array<F32>, Elementwise.halves(lp, rp),
    l => r => Parallel.build(~Elementwise.Request, ~Elementwise.Walk, ~Array<F32>, ~Array<F32>, ~Elementwise.walk, ~Elementwise.split, ~Elementwise.cell_at,
      Elementwise.Request{operation, plan}, inner, rest, count, base, (l, r)),
    l => r => Parallel.build(~Elementwise.Request, ~Elementwise.Walk, ~Array<F32>, ~Array<F32>, ~Elementwise.walk, ~Elementwise.split, ~Elementwise.cell_at,
      Elementwise.Request{operation, plan}, inner, rest, count, Nat.add(Model.capacity(inner), base), (l, r))) ==
    spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, 1n+inner, base) : Array<F32>}:
  (l1, l2) = lp
  (r1, r2) = rp
  +upper = Nat.add(Model.capacity(inner), base)
  %Equal.sym(Array<F32>, l1, Api.arr(ha, dl, fa), clone_first_eq(dl, ha, fa, (l1, l2), le)) :
    {ANode{Elementwise.build(inner, rest, operation, plan, count, base, _, r1), Elementwise.build(inner, rest, operation, plan, count, upper, l2, r2)} == spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, 1n+inner, base) : Array<F32>}
  %Equal.sym(Array<F32>, r1, Api.arr(hb, dr, fb), clone_first_eq(dr, hb, fb, (r1, r2), re)) :
    {ANode{Elementwise.build(inner, rest, operation, plan, count, base, Api.arr(ha, dl, fa), _), Elementwise.build(inner, rest, operation, plan, count, upper, l2, r2)} == spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, 1n+inner, base) : Array<F32>}
  %Equal.sym(Array<F32>, l2, Api.arr(ha, dl, fa), clone_second_eq(dl, ha, fa, (l1, l2), le)) :
    {ANode{Elementwise.build(inner, rest, operation, plan, count, base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)), Elementwise.build(inner, rest, operation, plan, count, upper, _, r2)} == spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, 1n+inner, base) : Array<F32>}
  %Equal.sym(Array<F32>, r2, Api.arr(hb, dr, fb), clone_second_eq(dr, hb, fb, (r1, r2), re)) :
    {ANode{Elementwise.build(inner, rest, operation, plan, count, base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)), Elementwise.build(inner, rest, operation, plan, count, upper, Api.arr(ha, dl, fa), _)} == spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, 1n+inner, base) : Array<F32>}
  %Equal.sym(Array<F32>, Elementwise.build(inner, rest, operation, plan, count, base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)), spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, base), low) :
    {ANode{_, Elementwise.build(inner, rest, operation, plan, count, upper, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb))} == spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, 1n+inner, base) : Array<F32>}
  %Equal.sym(Array<F32>, Elementwise.build(inner, rest, operation, plan, count, upper, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)), spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, upper), high) :
    {ANode{spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, base), _} == spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, 1n+inner, base) : Array<F32>}
  {==}

# Whatever the split, the parallel build returns `spec`.
def build_eq(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +nl: Nat, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +nr: Nat, +lfits: {Storage.bounds(dl, nl) == True{} : Bool}, +rfits: {Storage.bounds(dr, nr) == True{} : Bool}, +lvalid: {written(nl, ha) == True{} : Bool}, +rvalid: {written(nr, hb) == True{} : Bool}, +count_exact: {count == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat}, +left_count: {nl == Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat}, +right_count: {nr == Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat}, +depth: Nat, +splits: Nat, +base: Nat) ->
  {Elementwise.build(depth, splits, operation, plan, count, base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)) == spec(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, depth, base) : Array<F32>}:
  match depth splits:
    case 1n+inner 1n+rest:
      fork_eq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, inner, rest, base, build_eq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, lfits, rfits, lvalid, rvalid, count_exact, left_count, right_count, inner, rest, base),
        build_eq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, lfits, rfits, lvalid, rvalid, count_exact, left_count, right_count, inner, rest, Nat.add(Model.capacity(inner), base)), Array.clone(F32, Api.arr(ha, dl, fa)), {==}, Array.clone(F32, Api.arr(hb, dr, fb)), {==})
    case 0n _:
      tree_first(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, 0n, base, Parallel.tree(~Elementwise.Walk, ~Array<F32>, ~Array<F32>, ~Elementwise.cell_at, Elementwise.walk(Elementwise.Request{operation, plan}), 0n, count, base, (Api.arr(ha, dl, fa), Api.arr(hb, dr, fb))), tree_eq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, lfits, rfits, lvalid, rvalid, count_exact, left_count, right_count, 0n, base))
    case 1n+inner 0n:
      tree_first(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, 1n+inner, base, Parallel.tree(~Elementwise.Walk, ~Array<F32>, ~Array<F32>, ~Elementwise.cell_at, Elementwise.walk(Elementwise.Request{operation, plan}), 1n+inner, count, base, (Api.arr(ha, dl, fa), Api.arr(hb, dr, fb))), tree_eq(operation, plan, count, ha, dl, fa, nl, hb, dr, fb, nr, lfits, rfits, lvalid, rvalid, count_exact, left_count, right_count, 1n+inner, base))
