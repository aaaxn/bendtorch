import Base
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ../tensor/elementwise-parallel-proof.bend as Parallel
import ../shape/core.bend as Shape
import ../shape/flat.bend as Flat
import ../shape/broadcast_proofs.bend as Broadcast
import ../shape/projection_proofs.bend as ShapeProjection
import ../storage/arith.bend as Arith
import ../tensor/word-conversion.bend as Conversion
import ../tensor/projection-LAWS.bend as Projection
import ../shape/LAWS.bend as ShapeLaws
import ../tensor/contracts/broadcast.bend as BroadcastMap
import ../tensor/contracts/broadcast-proof.bend as BroadcastProof
import ../tensor/contracts/pointwise-proof.bend as PointwiseProof
import ./reduce.bend as Reduce
import ./reduce-LAWS.bend as ReduceLaws
import ./reduce-fiber-proof.bend as Fiber

# The fiber walk of a reduce plan. For the index rows of a plan whose output
# broadcasts the target, the walk of a target cell below the target's count
# reads the masked positions (`Fiber.fiber_walk`), every one of them lies
# inside the contribution, and adding the contribution's words there, in that
# order from +0.0, is `ReduceLaws.fiber`.

def numel_pad(+count: Nat, +dims: List<&2, Nat>) -> {Shape.numel(Shape.pad(count, dims)) == Shape.numel(dims) : Nat}:
  match count:
    case 0n:
      {==}
    case 1n+rest:
      Equal.trans(Nat, Nat.add(Shape.numel(Shape.pad(rest, dims)), 0n), Shape.numel(Shape.pad(rest, dims)), Shape.numel(dims),
        Broadcast.add_zero(Shape.numel(Shape.pad(rest, dims))), numel_pad(rest, dims))

def padding_left(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +lp: Nat, +rp: Nat,
  +aligned: Shape.Aligned(Shape.pad(lp, left), Shape.pad(rp, right), output)) ->
  {lp == Nat.sub(List.length(&2, Nat, output), List.length(&2, Nat, left)) : Nat}:
  ShapeProjection.padding_of(List.length(&2, Nat, output), lp, List.length(&2, Nat, left),
    Equal.trans(Nat, Nat.add(lp, List.length(&2, Nat, left)), List.length(&2, Nat, Shape.pad(lp, left)), List.length(&2, Nat, output),
      Broadcast.pad_length(lp, left), Broadcast.aligned_length_left(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned)))

def padding_right(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +lp: Nat, +rp: Nat,
  +aligned: Shape.Aligned(Shape.pad(lp, left), Shape.pad(rp, right), output)) ->
  {rp == Nat.sub(List.length(&2, Nat, output), List.length(&2, Nat, right)) : Nat}:
  ShapeProjection.padding_of(List.length(&2, Nat, output), rp, List.length(&2, Nat, right),
    Equal.trans(Nat, Nat.add(rp, List.length(&2, Nat, right)), List.length(&2, Nat, Shape.pad(rp, right)), List.length(&2, Nat, output),
      Broadcast.pad_length(rp, right), Broadcast.aligned_length_right(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned)))

# The index table of an aligned broadcast is walked as `Fiber.fiber_walk`
# says.
def table_walk(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +alignment: Shape.Broadcast<left, right, output>,
  +target: Nat, +below: {Nat.is_lt(target, Shape.numel(left)) == True{} : Bool}) ->
  Fiber.RestWalk(Shape.index_table(left, right, output), output, target, Shape.numel(output)):
  match alignment:
    case Shape.BAlign{+lp, +rp, minimal, +aligned}:
      +nl = Nat.sub(List.length(&2, Nat, output), List.length(&2, Nat, left))
      +nr = Nat.sub(List.length(&2, Nat, output), List.length(&2, Nat, right))
      Fiber.fiber_walk(Shape.pad(nl, left), Shape.pad(nr, right), output,
        %padding_left(left, right, output, lp, rp, aligned) : Shape.Aligned(Shape.pad(_, left), Shape.pad(nr, right), output)
        %padding_right(left, right, output, lp, rp, aligned) : Shape.Aligned(Shape.pad(lp, left), Shape.pad(_, right), output)
        aligned,
        target,
        %Equal.sym(Nat, Shape.numel(Shape.pad(nl, left)), Shape.numel(left), numel_pad(nl, left)) : {Nat.is_lt(target, _) == True{} : Bool}
        below)

# The same for the rows of a plan's index map.
def plan_walk(+plan: Shape.BroadcastPlan, +rows: List<&2, Shape.Row>,
  +exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>},
  +target: Nat, +below: {Nat.is_lt(target, Shape.numel(Shape.extents(Shape.left_shape(plan)))) == True{} : Bool}) ->
  Fiber.RestWalk(rows, Shape.extents(Shape.output_shape(plan)), target, Shape.numel(Shape.extents(Shape.output_shape(plan)))):
  +left = Shape.extents(Shape.left_shape(plan))
  +right = Shape.extents(Shape.right_shape(plan))
  +output = Shape.extents(Shape.output_shape(plan))
  %Equal.sym(List<&2, Shape.Row>, rows, Shape.index_table(left, right, output), exact) :
    Fiber.RestWalk(_, output, target, Shape.numel(output))
  table_walk(left, right, output, Shape.alignment(plan), target, below)

# As in reduce-fold-PROOF.bend: below the output count, the specified
# offsets are the index walks.
def offsets_walk(+plan: Shape.BroadcastPlan,
  +map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>,
  +position: Nat, +bound: {True{} == Nat.is_lt(position, Shape.elements(Shape.output_shape(plan))) : Bool}) ->
  {Projection.offset_spec(plan, position) ==
    Some{(U32.from_nat(BroadcastMap.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)),
        Shape.extents(Shape.output_shape(plan)), map, position)),
      U32.from_nat(BroadcastMap.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)),
        Shape.extents(Shape.output_shape(plan)), map, position)))} : Maybe<&1, U32 & U32>}:
  match map:
    case Shape.Indexed{+rows, +exact}:
      +left = Shape.extents(Shape.left_shape(plan))
      +right = Shape.extents(Shape.right_shape(plan))
      +output = Shape.extents(Shape.output_shape(plan))
      +index = {Shape.FIndex{position, bound} : Shape.FlatIndex<Shape.output_shape(plan)>}
      +slot = ShapeLaws.fin_of(Shape.numel(output), position)
      +x = U32.from_nat(Shape.walk_left(rows, position))
      +y = U32.from_nat(Shape.walk_right(rows, position))
      %ShapeLaws.project_left_index_refines(plan, Shape.Indexed{rows, exact}, index) :
        {Projection.pair_offsets(_, ShapeLaws.flatten_spec(right, ShapeLaws.project_right_coordinate(left, right, output, Shape.alignment(plan), slot))) ==
          Some{(x, y)} : Maybe<&1, U32 & U32>}
      %ShapeLaws.project_right_index_refines(plan, Shape.Indexed{rows, exact}, index) :
        {Projection.pair_offsets(Some{x}, _) == Some{(x, y)} : Maybe<&1, U32 & U32>}
      {==}
    case Shape.Unindexed{zero}:
      +output = Shape.output_shape(plan)
      Empty.absurd({Projection.offset_spec(plan, position) == Some{(U32.from_nat(0n), U32.from_nat(0n))} : Maybe<&1, U32 & U32>},
        Arith.lt_zero(position,
          %Equal.trans(Nat, Shape.elements(output), Shape.numel(Shape.extents(output)), 0n, Shape.elements_exact(output),
            Equal.sym(Nat, 0n, Shape.numel(Shape.extents(output)), zero)) : {Nat.is_lt(position, _) == True{} : Bool}
          Equal.sym(Bool, True{}, Nat.is_lt(position, Shape.elements(output)), bound)))

# Adding the words at a list of positions, in order.
def gather(+values: List<&2, F32>, positions: List<&2, Nat>, total: Maybe<&2, F32>) -> Maybe<&2, F32>:
  match positions:
    case Nil{}:
      total
    case position <> rest:
      gather(values, rest, ReduceLaws.add_word(total, ReduceLaws.word_at(values, position)))

def gather_append(+values: List<&2, F32>, +xs: List<&2, Nat>, +ys: List<&2, Nat>, +total: Maybe<&2, F32>) ->
  {gather(values, List.append(&2, Nat, xs, ys), total) == gather(values, ys, gather(values, xs, total)) : Maybe<&2, F32>}:
  match xs:
    case Nil{}:
      {==}
    case +x <> +rest:
      gather_append(values, rest, ys, ReduceLaws.add_word(total, ReduceLaws.word_at(values, x)))

def gather_kept(+values: List<&2, F32>, +keep: Bool, +position: Nat, +total: Maybe<&2, F32>) ->
  {gather(values, Fiber.kept(keep, position, Nil{}), total) == ReduceLaws.matched(total, ReduceLaws.word_at(values, position), keep) : Maybe<&2, F32>}:
  match keep:
    case True{}:
      {==}
    case False{}:
      {==}

def succ_add(+k: Nat) -> {Nat.add(k, 1n) == 1n+k : Nat}:
  Equal.trans(Nat, Nat.add(k, 1n), 1n+Nat.add(k, 0n), 1n+k, Arith.add_succ(k, 0n),
    Equal.cong(Nat, Nat, n => 1n+n, Nat.add(k, 0n), k, Broadcast.add_zero(k)))

# The masked fold of `ReduceLaws.fiber` adds, in order, the words at the
# masked positions.
def fiber_masked(+plan: Shape.BroadcastPlan, +rows: List<&2, Shape.Row>,
  +exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>},
  +values: List<&2, F32>, +index: U32, +count: Nat,
  +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {ReduceLaws.fiber(plan, values, index, count) == gather(values, Fiber.masked(count, rows, U32.to_nat(index), 0n, 0n), Some{0.0}) : Maybe<&2, F32>}:
  match count:
    case 0n:
      {==}
    case 1n+k:
      +t = U32.to_nat(index)
      +map : Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))> =
        Shape.Indexed{rows, exact}
      +limit = Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))
      +below = Conversion.le_successor_lt(k, limit, room)
      +lm = Shape.walk_left(rows, k)
      +rm = Shape.walk_right(rows, k)
      +before = Fiber.masked(k, rows, t, 0n, 0n)
      +last = Fiber.kept(Nat.is_eq(lm, t), rm, Nil{})
      +lm_below = BroadcastProof.left_walk_bound(plan, map, k, below)
      +rm_below = BroadcastProof.right_walk_bound(plan, map, k, below)
      %Equal.sym(Maybe<&1, U32 & U32>, Projection.offset_spec(plan, k), Some{(U32.from_nat(lm), U32.from_nat(rm))},
        offsets_walk(plan, map, k, BroadcastProof.output_inside(plan, k, below))) :
        {ReduceLaws.contribute(values, index, ReduceLaws.fiber(plan, values, index, k), _) ==
          gather(values, Fiber.masked(1n+k, rows, t, 0n, 0n), Some{0.0}) : Maybe<&2, F32>}
      %Equal.sym(Nat, U32.to_nat(U32.from_nat(rm)), rm, PointwiseProof.word_exact(rm, Shape.right_shape(plan), rm_below)) :
        {ReduceLaws.matched(ReduceLaws.fiber(plan, values, index, k), ReduceLaws.word_at(values, _), Nat.is_eq(U32.to_nat(U32.from_nat(lm)), t)) ==
          gather(values, Fiber.masked(1n+k, rows, t, 0n, 0n), Some{0.0}) : Maybe<&2, F32>}
      %Equal.sym(Nat, U32.to_nat(U32.from_nat(lm)), lm, PointwiseProof.word_exact(lm, Shape.left_shape(plan), lm_below)) :
        {ReduceLaws.matched(ReduceLaws.fiber(plan, values, index, k), ReduceLaws.word_at(values, rm), Nat.is_eq(_, t)) ==
          gather(values, Fiber.masked(1n+k, rows, t, 0n, 0n), Some{0.0}) : Maybe<&2, F32>}
      %Equal.sym(Maybe<&2, F32>, ReduceLaws.fiber(plan, values, index, k), gather(values, before, Some{0.0}),
        fiber_masked(plan, rows, exact, values, index, k, Conversion.le_predecessor(k, limit, room))) :
        {ReduceLaws.matched(_, ReduceLaws.word_at(values, rm), Nat.is_eq(lm, t)) ==
          gather(values, Fiber.masked(1n+k, rows, t, 0n, 0n), Some{0.0}) : Maybe<&2, F32>}
      %succ_add(k) :
        {ReduceLaws.matched(gather(values, before, Some{0.0}), ReduceLaws.word_at(values, rm), Nat.is_eq(lm, t)) ==
          gather(values, Fiber.masked(_, rows, t, 0n, 0n), Some{0.0}) : Maybe<&2, F32>}
      %Equal.sym(List<&2, Nat>, Fiber.masked(Nat.add(k, 1n), rows, t, 0n, 0n), List.append(&2, Nat, before, Fiber.masked(1n, rows, t, 0n, Nat.add(0n, k))),
        Fiber.masked_split(k, 1n, rows, t, 0n, 0n)) :
        {ReduceLaws.matched(gather(values, before, Some{0.0}), ReduceLaws.word_at(values, rm), Nat.is_eq(lm, t)) ==
          gather(values, _, Some{0.0}) : Maybe<&2, F32>}
      %Equal.sym(Maybe<&2, F32>, gather(values, List.append(&2, Nat, before, last), Some{0.0}), gather(values, last, gather(values, before, Some{0.0})),
        gather_append(values, before, last, Some{0.0})) :
        {ReduceLaws.matched(gather(values, before, Some{0.0}), ReduceLaws.word_at(values, rm), Nat.is_eq(lm, t)) == _ : Maybe<&2, F32>}
      Equal.sym(Maybe<&2, F32>, gather(values, last, gather(values, before, Some{0.0})),
        ReduceLaws.matched(gather(values, before, Some{0.0}), ReduceLaws.word_at(values, rm), Nat.is_eq(lm, t)),
        gather_kept(values, Nat.is_eq(lm, t), rm, gather(values, before, Some{0.0})))

# Every masked position lies inside the contribution.
def all_below(positions: List<&2, Nat>, +bound: Nat) -> Bool:
  match positions:
    case Nil{}:
      True{}
    case position <> rest:
      Bool.and(Nat.is_lt(position, bound), all_below(rest, bound))

def kept_below(+keep: Bool, +position: Nat, +rest: List<&2, Nat>, +bound: Nat,
  +inside: {Nat.is_lt(position, bound) == True{} : Bool}, +others: {all_below(rest, bound) == True{} : Bool}) ->
  {all_below(Fiber.kept(keep, position, rest), bound) == True{} : Bool}:
  match keep:
    case True{}:
      %Equal.sym(Bool, Nat.is_lt(position, bound), True{}, inside) : {Bool.and(_, all_below(rest, bound)) == True{} : Bool}
      others
    case False{}:
      others

def masked_below(+plan: Shape.BroadcastPlan, +rows: List<&2, Shape.Row>,
  +exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>},
  +target: Nat, +n: Nat, +cursor: Nat,
  +room: {Nat.is_le(Nat.add(cursor, n), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {all_below(Fiber.masked(n, rows, target, 0n, cursor), Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+m:
      +map : Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))> =
        Shape.Indexed{rows, exact}
      kept_below(Nat.is_eq(Shape.walk_left(rows, cursor), target), Shape.walk_right(rows, cursor), Fiber.masked(m, rows, target, 0n, 1n+cursor),
        Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))),
        BroadcastProof.right_walk_bound(plan, map, cursor, Fiber.lt_of_room(cursor, m, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), room)),
        masked_below(plan, rows, exact, target, m, 1n+cursor, Fiber.le_step(cursor, m, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), room)))

# The parallel build over an observed array. For a balanced array A, the
# fiber walk of a cell reads A at the offsets `Fiber.visited` lists and adds
# the words it sees there in order, and every cell of the build holds its
# walk's total below `count` and +0.0 above it.

def fsum(words: List<&2, F32>, accumulator: F32) -> F32:
  match words:
    case Nil{}:
      accumulator
    case word <> rest:
      fsum(rest, F32.add(accumulator, word))

def fsum_append(+xs: List<&2, F32>, +ys: List<&2, F32>, +accumulator: F32) ->
  {fsum(List.append(&2, F32, xs, ys), accumulator) == fsum(ys, fsum(xs, accumulator)) : F32}:
  match xs:
    case Nil{}:
      {==}
    case +x <> +rest:
      fsum_append(rest, ys, F32.add(accumulator, x))

# Each word is what A holds at its position.
def Obs(+d: Nat, -A: Array<F32>, positions: List<&2, Nat>, words: List<&2, F32>) -> Data:
  match positions words:
    case Nil{} Nil{}:
      Unit
    case p <> ps w <> ws:
      ObsStep<d, A, p, w, ps, ws>
    case Nil{} w <> ws:
      Empty
    case p <> ps Nil{}:
      Empty

type ObsStep<-d: Nat, -A: Array<F32>, -p: Nat, -w: F32, -ps: List<&2, Nat>, -ws: List<&2, F32>> is Data:
  ObsStep{-seen: {Model.observe(F32, d, A, Refinement.route(d, U32.from_nat(p))) == Some{w} : Maybe<&2, F32>}, -more: Obs(d, A, ps, ws)}

def obs_append(+d: Nat, -A: Array<F32>, +xs: List<&2, Nat>, +vs: List<&2, F32>, -ys: List<&2, Nat>, -ws: List<&2, F32>,
  -first: Obs(d, A, xs, vs), -second: Obs(d, A, ys, ws)) ->
  Parallel.Fact<Obs(d, A, List.append(&2, Nat, xs, ys), List.append(&2, F32, vs, ws))>:
  match xs vs:
    case Nil{} Nil{}:
      Parallel.Fact{second}
    case +p <> +ps +v <> +rest:
      Parallel.Fact{{\{ObsStep: seen => more =>
        ObsStep{seen, {\{Parallel.Fact: p => p} : Parallel.Fact<Obs(d, A, List.append(&2, Nat, ps, ys), List.append(&2, F32, rest, ws))> ->
          Obs(d, A, List.append(&2, Nat, ps, ys), List.append(&2, F32, rest, ws))}(obs_append(d, A, ps, rest, ys, ws, more, second))}} :
        ObsStep<d, A, p, v, ps, rest> -> ObsStep<d, A, p, v, List.append(&2, Nat, ps, ys), List.append(&2, F32, rest, ws)>}(first)}
    case Nil{} +v <> +rest:
      Parallel.Fact{Empty.absurd(Obs(d, A, ys, List.append(&2, F32, v <> rest, ws)), first)}
    case +p <> +ps Nil{}:
      Parallel.Fact{Empty.absurd(Obs(d, A, List.append(&2, Nat, p <> ps, ys), ws), first)}

# A run of `Reduce.fiber`: its total adds the words it saw, in order, and
# the array comes back unchanged.
type FiberRun<-d: Nat, -A: Array<F32>, -positions: List<&2, Nat>, -accumulator: F32, -result: F32 & Array<F32>> is Data:
  FiberRun{-words: List<&2, F32>, -seen: Obs(d, A, positions, words),
    -total: {Pair.fst(F32, Array<F32>, result) == fsum(words, accumulator) : F32},
    -kept: {Pair.snd(F32, Array<F32>, result) == A : Array<F32>}}

def leaf_run(+d: Nat, -A: Array<F32>, +base: Nat, -accumulator: F32, -array: Array<F32>, pair: Array<F32> & F32,
  -exact: {pair == Array.get(F32, array, U32.from_nat(base)) : Array<F32> & F32}, -same: {array == A : Array<F32>},
  -valid: Model.Balanced(F32, d, A)) ->
  Parallel.Fact<FiberRun<d, A, [base], accumulator, Reduce.leaf_add(accumulator, pair)>>:
  (kept, word) = pair
  Parallel.Fact{FiberRun{[word],
    ObsStep{Equal.sym(Maybe<&2, F32>, Some{word}, Model.observe(F32, d, A, Refinement.route(d, U32.from_nat(base))),
      {\{Parallel.Fact: p => p} : Parallel.Fact<{Some{Refinement.second(F32, (kept, word))} == Model.observe(F32, d, A, Refinement.route(d, U32.from_nat(base))) : Maybe<&2, F32>}> ->
        {Some{Refinement.second(F32, (kept, word))} == Model.observe(F32, d, A, Refinement.route(d, U32.from_nat(base))) : Maybe<&2, F32>}}(
        Parallel.read_value(d, array, A, U32.from_nat(base), (kept, word), exact, same, valid))), Unit{}},
    {==},
    {\{Parallel.Fact: p => p} : Parallel.Fact<{Refinement.first(F32, (kept, word)) == A : Array<F32>}> -> {Refinement.first(F32, (kept, word)) == A : Array<F32>}}(
      Parallel.read_keeps(d, array, A, U32.from_nat(base), (kept, word), exact, same, valid))}}

def next_run(+d: Nat, -A: Array<F32>, -low: List<&2, Nat>, -high: List<&2, Nat>, -accumulator: F32,
  -rec: F32 -> Array<F32> -> F32 & Array<F32>, pair: F32 & Array<F32>,
  -first: Parallel.Fact<FiberRun<d, A, low, accumulator, pair>>,
  -next: @value: F32 -> @array: Array<F32> -> @-same: {array == A : Array<F32>} -> Parallel.Fact<FiberRun<d, A, high, value, rec(value, array)>>) ->
  Parallel.Fact<FiberRun<d, A, List.append(&2, Nat, low, high), accumulator, Reduce.walk_next(rec, pair)>>:
  (value, array) = pair
  Parallel.Fact{{\{FiberRun: words => seen => total => kept =>
    {\{FiberRun: rest => more => after => back =>
      FiberRun{List.append(&2, F32, words, rest),
        {\{Parallel.Fact: p => p} : Parallel.Fact<Obs(d, A, List.append(&2, Nat, low, high), List.append(&2, F32, words, rest))> ->
          Obs(d, A, List.append(&2, Nat, low, high), List.append(&2, F32, words, rest))}(obs_append(d, A, low, words, high, rest, seen, more)),
        %Equal.sym(F32, fsum(List.append(&2, F32, words, rest), accumulator), fsum(rest, fsum(words, accumulator)),
          fsum_append(words, rest, accumulator)) : {Pair.fst(F32, Array<F32>, rec(value, array)) == _ : F32}
        %total : {Pair.fst(F32, Array<F32>, rec(value, array)) == fsum(rest, _) : F32}
        after,
        back}} :
      FiberRun<d, A, high, value, rec(value, array)> -> FiberRun<d, A, List.append(&2, Nat, low, high), accumulator, rec(value, array)>}(
      {\{Parallel.Fact: p => p} : Parallel.Fact<FiberRun<d, A, high, value, rec(value, array)>> -> FiberRun<d, A, high, value, rec(value, array)>}(
        next(value)(array)(kept)))} :
    FiberRun<d, A, low, accumulator, (value, array)> -> FiberRun<d, A, List.append(&2, Nat, low, high), accumulator, rec(value, array)>}(
    {\{Parallel.Fact: p => p} : Parallel.Fact<FiberRun<d, A, low, accumulator, (value, array)>> -> FiberRun<d, A, low, accumulator, (value, array)>}(first))}

def fiber_run(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +fuel: Nat, +target: Nat, +base: Nat, +digit: Nat, +d: Nat,
  -A: Array<F32>, +accumulator: F32, array: Array<F32>, -same: {array == A : Array<F32>}, -valid: Model.Balanced(F32, d, A)) ->
  Parallel.Fact<FiberRun<d, A, Fiber.visited(rows, extents, fuel, target, base, digit), accumulator,
    Reduce.fiber(rows, extents, fuel, target, base, digit, accumulator, array)>>:
  match rows extents fuel:
    case Nil{} _ _:
      leaf_run(d, A, base, accumulator, array, Array.get(F32, array, U32.from_nat(base)), {==}, same, valid)
    case Shape.Row{+stride, +left, +right} <> +rest +extent <> +others 0n:
      Parallel.Fact{FiberRun{Nil{}, Unit{}, {==}, same}}
    case Shape.Row{+stride, +left, +right} <> +rest +extent <> +others 1n+remaining:
      +rows : List<&2, Shape.Row> = Shape.Row{stride, left, right} <> rest
      +extents : List<&2, Nat> = extent <> others
      +inner = Nat.mod(target, left)
      next_run(d, A,
        Fiber.visited(rest, others, Reduce.first_span(rest, others), inner, Nat.add(base, Nat.mul(right, digit)), Reduce.first_digit(rest, inner)),
        Fiber.visited(rows, extents, remaining, target, base, 1n+digit), accumulator,
        value => kept => Reduce.fiber(rows, extents, remaining, target, base, 1n+digit, value, kept),
        Reduce.fiber(rest, others, Reduce.first_span(rest, others), inner, Nat.add(base, Nat.mul(right, digit)), Reduce.first_digit(rest, inner),
          accumulator, array),
        fiber_run(rest, others, Reduce.first_span(rest, others), inner, Nat.add(base, Nat.mul(right, digit)), Reduce.first_digit(rest, inner),
          d, A, accumulator, array, same, valid),
        value => kept => same => fiber_run(rows, extents, remaining, target, base, 1n+digit, d, A, value, kept, same, valid))
    case Shape.Row{+stride, +left, +right} <> +rest Nil{} _:
      Parallel.Fact{FiberRun{Nil{}, Unit{}, {==}, same}}

# The offsets the walk of cell `position` reads.
def visited_of(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +position: Nat) -> List<&2, Nat>:
  Fiber.visited(rows, extents, Reduce.first_span(rows, extents), position, 0n, Reduce.first_digit(rows, position))

# What target cell `position` holds: below `count`, the total of the words
# its walk saw, from +0.0; at or above it, +0.0.
type CellAt<-d: Nat, -A: Array<F32>, -rows: List<&2, Shape.Row>, -extents: List<&2, Nat>, -position: Nat, -value: Maybe<&2, F32>> is Data:
  CellAt{-word: F32, -observed: {value == Some{word} : Maybe<&2, F32>}, -words: List<&2, F32>,
    -seen: Obs(d, A, visited_of(rows, extents, position), words), -total: {word == fsum(words, 0.0) : F32}}

def CellSpec(+d: Nat, -A: Array<F32>, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +position: Nat, inside: Bool,
  -value: Maybe<&2, F32>) -> Type:
  match inside:
    case False{}:
      {value == Some{0.0} : Maybe<&2, F32>}
    case True{}:
      CellAt<d, A, rows, extents, position, value>

def CellIs(+d: Nat, -A: Array<F32>, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +position: Nat,
  -value: Maybe<&2, F32>) -> Type:
  CellSpec(d, A, rows, extents, position, Nat.is_lt(position, count), value)

def Cells(+depth: Nat, +d: Nat, -A: Array<F32>, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat,
  -out: Array<F32>) -> Type:
  @+path: Model.Path(depth) ->
    CellIs(d, A, rows, extents, count, Nat.add(Model.offset(depth, path), base), Model.observe(F32, depth, out, path))

def TreeFacts(+depth: Nat, +d: Nat, -A: Array<F32>, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat,
  -result: Array<F32> & Array<F32>) -> Type:
  {Pair.snd(Array<F32>, Array<F32>, result) == A : Array<F32>} &
    Cells(depth, d, A, rows, extents, count, base, Pair.fst(Array<F32>, Array<F32>, result))

def facts_kept(+depth: Nat, +d: Nat, -A: Array<F32>, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat,
  -result: Array<F32> & Array<F32>, -facts: Parallel.Fact<TreeFacts(depth, d, A, rows, extents, count, base, result)>) ->
  Parallel.Fact<{Pair.snd(Array<F32>, Array<F32>, result) == A : Array<F32>}>:
  Parallel.Fact{Pair.fst({Pair.snd(Array<F32>, Array<F32>, result) == A : Array<F32>},
    Cells(depth, d, A, rows, extents, count, base, Pair.fst(Array<F32>, Array<F32>, result)),
    {\{Parallel.Fact: p => p} : Parallel.Fact<(TreeFacts(depth, d, A, rows, extents, count, base, result))> ->
      TreeFacts(depth, d, A, rows, extents, count, base, result)}(facts))}

def facts_cells(+depth: Nat, +d: Nat, -A: Array<F32>, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat,
  -result: Array<F32> & Array<F32>, -facts: Parallel.Fact<TreeFacts(depth, d, A, rows, extents, count, base, result)>) ->
  Parallel.Fact<Cells(depth, d, A, rows, extents, count, base, Pair.fst(Array<F32>, Array<F32>, result))>:
  Parallel.Fact{Pair.snd({Pair.snd(Array<F32>, Array<F32>, result) == A : Array<F32>},
    Cells(depth, d, A, rows, extents, count, base, Pair.fst(Array<F32>, Array<F32>, result)),
    {\{Parallel.Fact: p => p} : Parallel.Fact<(TreeFacts(depth, d, A, rows, extents, count, base, result))> ->
      TreeFacts(depth, d, A, rows, extents, count, base, result)}(facts))}

def cell_leaf_facts(+d: Nat, -A: Array<F32>, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat,
  pair: F32 & Array<F32>, -decided: {True{} == Nat.is_lt(base, count) : Bool},
  -run: Parallel.Fact<FiberRun<d, A, visited_of(rows, extents, base), 0.0, pair>>) ->
  Parallel.Fact<TreeFacts(0n, d, A, rows, extents, count, base, Reduce.cell_leaf(pair))>:
  (value, array) = pair
  Parallel.Fact{{\{FiberRun: words => seen => total => kept =>
    (kept, path =>
      %decided : CellSpec(d, A, rows, extents, base, _, Some{value})
      CellAt{value, {==}, words, seen, total})} :
    FiberRun<d, A, visited_of(rows, extents, base), 0.0, (value, array)> ->
    TreeFacts(0n, d, A, rows, extents, count, base, (ALeaf{value}, array))}(
      {\{Parallel.Fact: p => p} : Parallel.Fact<FiberRun<d, A, visited_of(rows, extents, base), 0.0, (value, array)>> ->
        FiberRun<d, A, visited_of(rows, extents, base), 0.0, (value, array)>}(run))}

def cell_facts(+d: Nat, -A: Array<F32>, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat,
  array: Array<F32>, inside: Bool, -decided: {inside == Nat.is_lt(base, count) : Bool},
  -same: {array == A : Array<F32>}, -valid: Model.Balanced(F32, d, A)) ->
  Parallel.Fact<TreeFacts(0n, d, A, rows, extents, count, base, Reduce.cell(rows, extents, base, array, inside))>:
  match inside:
    case False{}:
      Parallel.Fact{(same, path =>
        %decided : CellSpec(d, A, rows, extents, base, _, Some{0.0})
        {==})}
    case True{}:
      cell_leaf_facts(d, A, rows, extents, count, base, Reduce.fiber_of(rows, extents, base, array), decided,
        fiber_run(rows, extents, Reduce.first_span(rows, extents), base, 0n, Reduce.first_digit(rows, base), d, A, 0.0, array, same, valid))

def node_cell(+lower: Nat, +d: Nat, -A: Array<F32>, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat,
  -a: Array<F32>, -b: Array<F32>, +path: Model.Path(1n+lower),
  -low: Parallel.Fact<Cells(lower, d, A, rows, extents, count, base, a)>,
  -high: Parallel.Fact<Cells(lower, d, A, rows, extents, count, Nat.add(Model.capacity(lower), base), b)>) ->
  Parallel.Fact<CellIs(d, A, rows, extents, count, Nat.add(Model.offset(1n+lower, path), base), Model.observe(F32, 1n+lower, ANode{a, b}, path))>:
  Model.Step{+bit, +rest} = path
  match bit:
    case False{}:
      Parallel.Fact{{\{Parallel.Fact: p => p} : Parallel.Fact<(Cells(lower, d, A, rows, extents, count, base, a))> ->
        Cells(lower, d, A, rows, extents, count, base, a)}(low)(rest)}
    case True{}:
      Parallel.Fact{
        %Parallel.position_step(Model.capacity(lower), Model.offset(lower, rest), base) :
          CellIs(d, A, rows, extents, count, _, Model.observe(F32, lower, b, rest))
        {\{Parallel.Fact: p => p} : Parallel.Fact<(Cells(lower, d, A, rows, extents, count, Nat.add(Model.capacity(lower), base), b))> ->
          Cells(lower, d, A, rows, extents, count, Nat.add(Model.capacity(lower), base), b)}(high)(rest)}

def node_cells(+lower: Nat, +d: Nat, -A: Array<F32>, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat,
  -a: Array<F32>, -b: Array<F32>,
  -low: Parallel.Fact<Cells(lower, d, A, rows, extents, count, base, a)>,
  -high: Parallel.Fact<Cells(lower, d, A, rows, extents, count, Nat.add(Model.capacity(lower), base), b)>) ->
  Parallel.Fact<Cells(1n+lower, d, A, rows, extents, count, base, ANode{a, b})>:
  Parallel.Fact{path => {\{Parallel.Fact: p => p} :
    Parallel.Fact<(CellIs(d, A, rows, extents, count, Nat.add(Model.offset(1n+lower, path), base), Model.observe(F32, 1n+lower, ANode{a, b}, path)))> ->
    CellIs(d, A, rows, extents, count, Nat.add(Model.offset(1n+lower, path), base), Model.observe(F32, 1n+lower, ANode{a, b}, path))}(
      node_cell(lower, d, A, rows, extents, count, base, a, b, path, low, high))}

def tree_join_facts(+lower: Nat, +d: Nat, -A: Array<F32>, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat,
  -a: Array<F32>, high: Array<F32> & Array<F32>,
  -low: Parallel.Fact<Cells(lower, d, A, rows, extents, count, base, a)>,
  -facts: Parallel.Fact<TreeFacts(lower, d, A, rows, extents, count, Nat.add(Model.capacity(lower), base), high)>) ->
  Parallel.Fact<TreeFacts(1n+lower, d, A, rows, extents, count, base, Reduce.tree_join(a, high))>:
  (b, array) = high
  Parallel.Fact{(
    {\{Parallel.Fact: p => p} : Parallel.Fact<({Pair.snd(Array<F32>, Array<F32>, (b, array)) == A : Array<F32>})> ->
      {Pair.snd(Array<F32>, Array<F32>, (b, array)) == A : Array<F32>}}(
      facts_kept(lower, d, A, rows, extents, count, Nat.add(Model.capacity(lower), base), (b, array), facts)),
    {\{Parallel.Fact: p => p} : Parallel.Fact<(Cells(1n+lower, d, A, rows, extents, count, base, ANode{a, b}))> ->
      Cells(1n+lower, d, A, rows, extents, count, base, ANode{a, b})}(
      node_cells(lower, d, A, rows, extents, count, base, a, b, low,
        facts_cells(lower, d, A, rows, extents, count, Nat.add(Model.capacity(lower), base), (b, array), facts))))}

def tree_node_facts(+lower: Nat, +d: Nat, -A: Array<F32>, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat,
  rec: Array<F32> -> Array<F32> & Array<F32>, low: Array<F32> & Array<F32>,
  -facts: Parallel.Fact<TreeFacts(lower, d, A, rows, extents, count, base, low)>,
  -next: @array: Array<F32> -> @-same: {array == A : Array<F32>} ->
    Parallel.Fact<TreeFacts(lower, d, A, rows, extents, count, Nat.add(Model.capacity(lower), base), rec(array))>) ->
  Parallel.Fact<TreeFacts(1n+lower, d, A, rows, extents, count, base, Reduce.tree_node(rec, low))>:
  (a, array) = low
  tree_join_facts(lower, d, A, rows, extents, count, base, a, rec(array),
    facts_cells(lower, d, A, rows, extents, count, base, (a, array), facts),
    next(array)(
      {\{Parallel.Fact: p => p} : Parallel.Fact<({Pair.snd(Array<F32>, Array<F32>, (a, array)) == A : Array<F32>})> ->
        {Pair.snd(Array<F32>, Array<F32>, (a, array)) == A : Array<F32>}}(
        facts_kept(lower, d, A, rows, extents, count, base, (a, array), facts))))

def tree_facts(+depth: Nat, +d: Nat, -A: Array<F32>, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat,
  array: Array<F32>, -same: {array == A : Array<F32>}, -valid: Model.Balanced(F32, d, A)) ->
  Parallel.Fact<TreeFacts(depth, d, A, rows, extents, count, base, Reduce.tree(depth, rows, extents, count, base, array))>:
  match depth:
    case 0n:
      cell_facts(d, A, rows, extents, count, base, array, Nat.is_lt(base, count), {==}, same, valid)
    case 1n+lower:
      tree_node_facts(lower, d, A, rows, extents, count, base,
        a => Reduce.tree(lower, rows, extents, count, Nat.add(Model.capacity(lower), base), a),
        Reduce.tree(lower, rows, extents, count, base, array),
        tree_facts(lower, d, A, rows, extents, count, base, array, same, valid),
        a => same => tree_facts(lower, d, A, rows, extents, count, Nat.add(Model.capacity(lower), base), a, same, valid))

def fork_cells(+lower: Nat, +d: Nat, -A: Array<F32>, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat,
  -array: Array<F32>, pair: Array<F32> & Array<F32>, -exact: {pair == Array.clone(F32, array) : Array<F32> & Array<F32>},
  -low: Array<F32> -> Array<F32>, -high: Array<F32> -> Array<F32>,
  -same: {array == A : Array<F32>}, -valid: Model.Balanced(F32, d, A),
  -low_cells: @a: Array<F32> -> @-same: {a == A : Array<F32>} -> Parallel.Fact<Cells(lower, d, A, rows, extents, count, base, low(a))>,
  -high_cells: @a: Array<F32> -> @-same: {a == A : Array<F32>} ->
    Parallel.Fact<Cells(lower, d, A, rows, extents, count, Nat.add(Model.capacity(lower), base), high(a))>) ->
  Parallel.Fact<Cells(1n+lower, d, A, rows, extents, count, base, Reduce.fork(pair, low, high))>:
  (first, second) = pair
  node_cells(lower, d, A, rows, extents, count, base, low(first), high(second),
    low_cells(first)(
      {\{Parallel.Fact: p => p} : Parallel.Fact<({Pair.fst(Array<F32>, Array<F32>, (first, second)) == A : Array<F32>})> ->
        {Pair.fst(Array<F32>, Array<F32>, (first, second)) == A : Array<F32>}}(
        Parallel.clone_first(d, array, A, (first, second), exact, same, valid))),
    high_cells(second)(
      {\{Parallel.Fact: p => p} : Parallel.Fact<({Pair.snd(Array<F32>, Array<F32>, (first, second)) == A : Array<F32>})> ->
        {Pair.snd(Array<F32>, Array<F32>, (first, second)) == A : Array<F32>}}(
        Parallel.clone_second(d, array, A, (first, second), exact, same, valid))))

# Whatever the split, every cell of the parallel build holds what CellIs
# names.
def build_cells(+depth: Nat, +splits: Nat, +d: Nat, -A: Array<F32>, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat,
  +base: Nat, array: Array<F32>, -same: {array == A : Array<F32>}, -valid: Model.Balanced(F32, d, A)) ->
  Parallel.Fact<Cells(depth, d, A, rows, extents, count, base, Reduce.build(depth, splits, rows, extents, count, base, array))>:
  match depth splits:
    case 1n+lower 1n+rest:
      fork_cells(lower, d, A, rows, extents, count, base, array, Array.clone(F32, array), {==},
        a => Reduce.build(lower, rest, rows, extents, count, base, a),
        a => Reduce.build(lower, rest, rows, extents, count, Nat.add(Model.capacity(lower), base), a),
        same, valid,
        a => same => build_cells(lower, rest, d, A, rows, extents, count, base, a, same, valid),
        a => same => build_cells(lower, rest, d, A, rows, extents, count, Nat.add(Model.capacity(lower), base), a, same, valid))
    case 0n _:
      facts_cells(0n, d, A, rows, extents, count, base, Reduce.tree(0n, rows, extents, count, base, array),
        tree_facts(0n, d, A, rows, extents, count, base, array, same, valid))
    case 1n+lower 0n:
      facts_cells(1n+lower, d, A, rows, extents, count, base, Reduce.tree(1n+lower, rows, extents, count, base, array),
        tree_facts(1n+lower, d, A, rows, extents, count, base, array, same, valid))

# The words a walk saw are the contribution's words at the offsets it read.
def Reads(+d: Nat, -A: Array<F32>, +values: List<&2, F32>, +bound: Nat) -> Type:
  @+position: Nat -> @inside: {Nat.is_lt(position, bound) == True{} : Bool} ->
    {Model.observe(F32, d, A, Refinement.route(d, U32.from_nat(position))) == ReduceLaws.word_at(values, position) : Maybe<&2, F32>}

def obs_gather(+d: Nat, -A: Array<F32>, +values: List<&2, F32>, +bound: Nat, +positions: List<&2, Nat>, +words: List<&2, F32>,
  +accumulator: F32, -seen: Obs(d, A, positions, words), +below: {all_below(positions, bound) == True{} : Bool},
  -reads: Reads(d, A, values, bound)) ->
  Parallel.Fact<{Some{fsum(words, accumulator)} == gather(values, positions, Some{accumulator}) : Maybe<&2, F32>}>:
  match positions words:
    case Nil{} Nil{}:
      Parallel.Fact{{==}}
    case +p <> +ps +w <> +ws:
      +inside = Arith.and_left(Nat.is_lt(p, bound), all_below(ps, bound), below)
      +others = Arith.and_right(Nat.is_lt(p, bound), all_below(ps, bound), below)
      Parallel.Fact{{\{ObsStep: observed => more =>
        %Equal.sym(Maybe<&2, F32>, ReduceLaws.word_at(values, p), Some{w},
          Equal.trans(Maybe<&2, F32>, ReduceLaws.word_at(values, p), Model.observe(F32, d, A, Refinement.route(d, U32.from_nat(p))), Some{w},
            Equal.sym(Maybe<&2, F32>, Model.observe(F32, d, A, Refinement.route(d, U32.from_nat(p))), ReduceLaws.word_at(values, p), reads(p, inside)),
            observed)) :
          {Some{fsum(ws, F32.add(accumulator, w))} == gather(values, ps, ReduceLaws.add_word(Some{accumulator}, _)) : Maybe<&2, F32>}
        {\{Parallel.Fact: p => p} : Parallel.Fact<{Some{fsum(ws, F32.add(accumulator, w))} == gather(values, ps, Some{F32.add(accumulator, w)}) : Maybe<&2, F32>}> ->
          {Some{fsum(ws, F32.add(accumulator, w))} == gather(values, ps, Some{F32.add(accumulator, w)}) : Maybe<&2, F32>}}(
          obs_gather(d, A, values, bound, ps, ws, F32.add(accumulator, w), more, others, reads))} :
        ObsStep<d, A, p, w, ps, ws> -> {Some{fsum(ws, F32.add(accumulator, w))} == gather(values, p <> ps, Some{accumulator}) : Maybe<&2, F32>}}(seen)}
    case Nil{} +w <> +ws:
      Parallel.Fact{Empty.absurd({Some{fsum(w <> ws, accumulator)} == gather(values, Nil{}, Some{accumulator}) : Maybe<&2, F32>}, seen)}
    case +p <> +ps Nil{}:
      Parallel.Fact{Empty.absurd({Some{accumulator} == gather(values, p <> ps, Some{accumulator}) : Maybe<&2, F32>}, seen)}

# A cell of the parallel build holds `ReduceLaws.fiber`: the words its walk
# saw add up, from +0.0, to the masked fold over the contribution.
def cell_value(+plan: Shape.BroadcastPlan, +rows: List<&2, Shape.Row>,
  +exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>},
  +values: List<&2, F32>, +index: U32, +d: Nat, -A: Array<F32>, +words: List<&2, F32>,
  -seen: Obs(d, A, visited_of(rows, Shape.extents(Shape.output_shape(plan)), U32.to_nat(index)), words),
  +inside: {Nat.is_lt(U32.to_nat(index), Shape.numel(Shape.extents(Shape.left_shape(plan)))) == True{} : Bool},
  -reads: Reads(d, A, values, Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))))) ->
  Parallel.Fact<{Some{fsum(words, 0.0)} == ReduceLaws.fiber(plan, values, index, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) : Maybe<&2, F32>}>:
  +t = U32.to_nat(index)
  +output = Shape.extents(Shape.output_shape(plan))
  +count = Shape.numel(output)
  +walk = plan_walk(plan, rows, exact, t, inside)
  +spot = masked_below(plan, rows, exact, t, count, 0n, Conversion.le_reflexive(count))
  Parallel.Fact{
    %Equal.sym(Maybe<&2, F32>, ReduceLaws.fiber(plan, values, index, count), gather(values, Fiber.masked(count, rows, t, 0n, 0n), Some{0.0}),
      fiber_masked(plan, rows, exact, values, index, count, Conversion.le_reflexive(count))) :
      {Some{fsum(words, 0.0)} == _ : Maybe<&2, F32>}
    %Equal.sym(List<&2, Nat>, Fiber.masked(count, rows, t, 0n, 0n), visited_of(rows, output, t), walk) :
      {Some{fsum(words, 0.0)} == gather(values, _, Some{0.0}) : Maybe<&2, F32>}
    {\{Parallel.Fact: p => p} : Parallel.Fact<{Some{fsum(words, 0.0)} == gather(values, visited_of(rows, output, t), Some{0.0}) : Maybe<&2, F32>}> ->
      {Some{fsum(words, 0.0)} == gather(values, visited_of(rows, output, t), Some{0.0}) : Maybe<&2, F32>}}(
      obs_gather(d, A, values, Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))), visited_of(rows, output, t), words, 0.0, seen,
        %walk :
          {all_below(_, Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))) == True{} : Bool}
        spot,
        reads))}
