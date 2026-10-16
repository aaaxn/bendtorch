import Base
import ../shape/core.bend as Shape
import ../shape/flat.bend as Flat
import ../shape/nat_proofs.bend as Nats
import ../shape/index_proofs.bend as Index
import ../shape/plan_proofs.bend as PlanProof
import ../shape/LAWS.bend as ShapeLaws
# Proves the shape laws that importing ShapeLaws would otherwise leave open.
import ../shape/PROOF.bend as ShapeProof
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ../storage/clone.bend as Clone
import ../storage/arith.bend as Arith
import ../storage/main.bend as Storage
import ../storage/route.bend as Route
import ../storage/certificates.bend as Certificates
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./elementwise.bend as Elementwise
import ./elementwise-totality.bend as Totality
import ./word-conversion.bend as WordConversion
import ./parallel-proof.bend as Generic
import ./elementwise-walk-proof.bend as WalkProof

# Proofs that the parallel elementwise build writes, at every output index,
# the value `Elementwise.expected` names there.

type Fact<-P: Type> is Data:
  Fact{-proof: P}

# The value a cell gets from the two observed input cells.
def pick(operation: Elementwise.Operation, left: Maybe<&2, F32>, right: Maybe<&2, F32>) -> Maybe<&2, F32>:
  match left:
    case None{}:
      None{}
    case Some{value}:
      Elementwise.expected_right(operation, value, right)

def want(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +dl: Nat, +dr: Nat, A: Array<F32>, B: Array<F32>, +position: Nat) ->
  Maybe<&2, F32>:
  pick(operation, Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position)))), Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position)))))

def transport_balanced(-d: Nat, -a: Array<F32>, -b: Array<F32>, -same: {a == b : Array<F32>}, -valid: Model.Balanced(F32, d, b)) ->
  Fact<Model.Balanced(F32, d, a)>:
  Fact{
    %Equal.sym(Array<F32>, a, b, same) : Model.Balanced(F32, d, _)
    valid}

# A read keeps the array and returns the observed cell.
def read_keeps(-d: Nat, -original: Array<F32>, -target: Array<F32>, -i: U32, -pair: Array<F32> & F32,
  -exact: {pair == Array.get(F32, original, i) : Array<F32> & F32}, -same: {original == target : Array<F32>},
  -valid: Model.Balanced(F32, d, target)) -> Fact<{Refinement.first(F32, pair) == target : Array<F32>}>:
  Fact{Equal.trans(Array<F32>, Refinement.first(F32, pair), Refinement.first(F32, Array.get(F32, original, i)), target,
    Equal.cong(Array<F32> & F32, Array<F32>, p => Refinement.first(F32, p), pair, Array.get(F32, original, i), exact),
    Equal.trans(Array<F32>, Refinement.first(F32, Array.get(F32, original, i)), original, target,
      Refinement.law_get_retains(F32, d, original, i, {\{Fact: p => p} : Fact<(Model.Balanced(F32, d, original))> -> Model.Balanced(F32, d, original)}(transport_balanced(d, original, target, same, valid))),
      same))}

def read_value(-d: Nat, -original: Array<F32>, -target: Array<F32>, -i: U32, -pair: Array<F32> & F32,
  -exact: {pair == Array.get(F32, original, i) : Array<F32> & F32}, -same: {original == target : Array<F32>},
  -valid: Model.Balanced(F32, d, target)) ->
  Fact<{Some{Refinement.second(F32, pair)} == Model.observe(F32, d, target, Refinement.route(d, i)) : Maybe<&2, F32>}>:
  Fact{Equal.trans(Maybe<&2, F32>, Some{Refinement.second(F32, pair)}, Some{Refinement.second(F32, Array.get(F32, original, i))},
    Model.observe(F32, d, target, Refinement.route(d, i)),
    Equal.cong(Array<F32> & F32, Maybe<&2, F32>, p => Some{Refinement.second(F32, p)}, pair, Array.get(F32, original, i), exact),
    Equal.trans(Maybe<&2, F32>, Some{Refinement.second(F32, Array.get(F32, original, i))},
      Model.observe(F32, d, original, Refinement.route(d, i)), Model.observe(F32, d, target, Refinement.route(d, i)),
      Refinement.law_get_refines(F32, d, original, i, {\{Fact: p => p} : Fact<(Model.Balanced(F32, d, original))> -> Model.Balanced(F32, d, original)}(transport_balanced(d, original, target, same, valid))),
      Equal.cong(Array<F32>, Maybe<&2, F32>, a => Model.observe(F32, d, a, Refinement.route(d, i)), original, target, same)))}

def pick_value(+operation: Elementwise.Operation, -x: F32, -y: F32, -lm: Maybe<&2, F32>, -rm: Maybe<&2, F32>,
  -ex: {Some{x} == lm : Maybe<&2, F32>}, -ey: {Some{y} == rm : Maybe<&2, F32>}) ->
  Fact<{Some{Elementwise.scalar(operation, x, y)} == pick(operation, lm, rm) : Maybe<&2, F32>}>:
  Fact{
    %ex : {Some{Elementwise.scalar(operation, x, y)} == pick(operation, _, rm) : Maybe<&2, F32>}
    %ey : {Some{Elementwise.scalar(operation, x, y)} == Elementwise.expected_right(operation, x, _) : Maybe<&2, F32>}
    {==}}

def clone_first(-d: Nat, -original: Array<F32>, -target: Array<F32>, -pair: Array<F32> & Array<F32>,
  -exact: {pair == Array.clone(F32, original) : Array<F32> & Array<F32>}, -same: {original == target : Array<F32>},
  -valid: Model.Balanced(F32, d, target)) -> Fact<{Pair.fst(Array<F32>, Array<F32>, pair) == target : Array<F32>}>:
  Fact{Equal.trans(Array<F32>, Pair.fst(Array<F32>, Array<F32>, pair), Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, original)), target,
    Equal.cong(Array<F32> & Array<F32>, Array<F32>, p => Pair.fst(Array<F32>, Array<F32>, p), pair, Array.clone(F32, original), exact),
    Equal.trans(Array<F32>, Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, original)), original, target,
      Clone.first_equal(F32, d, original, Clone.certify(F32, d, original, {\{Fact: p => p} : Fact<(Model.Balanced(F32, d, original))> -> Model.Balanced(F32, d, original)}(transport_balanced(d, original, target, same, valid)))),
      same))}

def clone_second(-d: Nat, -original: Array<F32>, -target: Array<F32>, -pair: Array<F32> & Array<F32>,
  -exact: {pair == Array.clone(F32, original) : Array<F32> & Array<F32>}, -same: {original == target : Array<F32>},
  -valid: Model.Balanced(F32, d, target)) -> Fact<{Pair.snd(Array<F32>, Array<F32>, pair) == target : Array<F32>}>:
  Fact{Equal.trans(Array<F32>, Pair.snd(Array<F32>, Array<F32>, pair), Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, original)), target,
    Equal.cong(Array<F32> & Array<F32>, Array<F32>, p => Pair.snd(Array<F32>, Array<F32>, p), pair, Array.clone(F32, original), exact),
    Equal.trans(Array<F32>, Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, original)), original, target,
      Clone.second_equal(F32, d, original, Clone.certify(F32, d, original, {\{Fact: p => p} : Fact<(Model.Balanced(F32, d, original))> -> Model.Balanced(F32, d, original)}(transport_balanced(d, original, target, same, valid)))),
      same))}

# The input cells a build reads for output cell `position`.
def left_at(+plan: Shape.BroadcastPlan, +position: Nat) -> U32:
  U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position))

def right_at(+plan: Shape.BroadcastPlan, +position: Nat) -> U32:
  U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position))

def pair_equal(-a: Array<F32>, -b: Array<F32>, -A: Array<F32>, -B: Array<F32>, ea: {a == A : Array<F32>}, eb: {b == B : Array<F32>}) ->
  {(a, b) == (A, B) : Array<F32> & Array<F32>}:
  Equal.trans(Array<F32> & Array<F32>, (a, b), (A, b), (A, B),
    Equal.cong(Array<F32>, Array<F32> & Array<F32>, x => (x, b), a, A, ea),
    Equal.cong(Array<F32>, Array<F32> & Array<F32>, y => (A, y), b, B, eb))

# What an inside cell of the build gives: both arrays back, and the F32
# `want` names.
def CellOut(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +position: Nat,
  -result: F32 & (Array<F32> & Array<F32>)) -> Type:
  {Pair.snd(F32, Array<F32> & Array<F32>, result) == (A, B) : Array<F32> & Array<F32>} &
    {Some{Pair.fst(F32, Array<F32> & Array<F32>, result)} == want(operation, plan, dl, dr, A, B, position) : Maybe<&2, F32>}

def right_out(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +position: Nat,
  -x: F32, -la: Array<F32>, pair: Array<F32> & F32, -exact: {pair == Array.get(F32, B, right_at(plan, position)) : Array<F32> & F32},
  -ex: {Some{x} == Model.observe(F32, dl, A, Refinement.route(dl, left_at(plan, position))) : Maybe<&2, F32>},
  -el: {la == A : Array<F32>}, -valid: Model.Balanced(F32, dr, B)) ->
  Fact<CellOut(operation, plan, dl, dr, A, B, position, Elementwise.cell_right(operation, x, la, pair))>:
  (kept, y) = pair
  Fact{(pair_equal(la, kept, A, B, el,
      {\{Fact: q => q} : Fact<({Refinement.first(F32, (kept, y)) == B : Array<F32>})> -> {Refinement.first(F32, (kept, y)) == B : Array<F32>}}(
        read_keeps(dr, B, B, right_at(plan, position), (kept, y), exact, {==}, valid))),
    {\{Fact: q => q} : Fact<({Some{Elementwise.scalar(operation, x, y)} == pick(operation, Model.observe(F32, dl, A, Refinement.route(dl, left_at(plan, position))), Model.observe(F32, dr, B, Refinement.route(dr, right_at(plan, position)))) : Maybe<&2, F32>})> ->
      {Some{Elementwise.scalar(operation, x, y)} == pick(operation, Model.observe(F32, dl, A, Refinement.route(dl, left_at(plan, position))), Model.observe(F32, dr, B, Refinement.route(dr, right_at(plan, position)))) : Maybe<&2, F32>}}(
      pick_value(operation, x, y, Model.observe(F32, dl, A, Refinement.route(dl, left_at(plan, position))), Model.observe(F32, dr, B, Refinement.route(dr, right_at(plan, position))), ex,
        {\{Fact: q => q} : Fact<({Some{Refinement.second(F32, (kept, y))} == Model.observe(F32, dr, B, Refinement.route(dr, right_at(plan, position))) : Maybe<&2, F32>})> ->
          {Some{Refinement.second(F32, (kept, y))} == Model.observe(F32, dr, B, Refinement.route(dr, right_at(plan, position))) : Maybe<&2, F32>}}(
          read_value(dr, B, B, right_at(plan, position), (kept, y), exact, {==}, valid)))))}

def left_out(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +dl: Nat, +dr: Nat, -A: Array<F32>, B: Array<F32>, +position: Nat,
  pair: Array<F32> & F32, -exact: {pair == Array.get(F32, A, left_at(plan, position)) : Array<F32> & F32},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Fact<CellOut(operation, plan, dl, dr, A, B, position, Elementwise.cell_left(operation, B, right_at(plan, position), pair))>:
  (kept, x) = pair
  right_out(operation, plan, dl, dr, A, B, position, x, kept, Array.get(F32, B, right_at(plan, position)), {==},
    {\{Fact: q => q} : Fact<({Some{Refinement.second(F32, (kept, x))} == Model.observe(F32, dl, A, Refinement.route(dl, left_at(plan, position))) : Maybe<&2, F32>})> ->
      {Some{Refinement.second(F32, (kept, x))} == Model.observe(F32, dl, A, Refinement.route(dl, left_at(plan, position))) : Maybe<&2, F32>}}(
      read_value(dl, A, A, left_at(plan, position), (kept, x), exact, {==}, left_valid)),
    {\{Fact: q => q} : Fact<({Refinement.first(F32, (kept, x)) == A : Array<F32>})> -> {Refinement.first(F32, (kept, x)) == A : Array<F32>}}(
      read_keeps(dl, A, A, left_at(plan, position), (kept, x), exact, {==}, left_valid)),
    right_valid)

# The elementwise cell function of `Parallel.build`, from the input arrays.
def cell_of(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +position: Nat, A: Array<F32>, B: Array<F32>, inside: Bool) ->
  F32 & (Array<F32> & Array<F32>):
  Elementwise.cell_at(Elementwise.walk(Elementwise.Request{operation, plan}), position, (A, B), inside)

def cell_out(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +dl: Nat, +dr: Nat, A: Array<F32>, B: Array<F32>, +position: Nat,
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Fact<CellOut(operation, plan, dl, dr, A, B, position, cell_of(operation, plan, position, A, B, True{}))>:
  %Equal.sym(F32 & (Array<F32> & Array<F32>), cell_of(operation, plan, position, A, B, True{}),
    Elementwise.cell(operation, Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)),
      Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position, A, B, True{}),
    WalkProof.cell_at_walk(operation, plan, position, A, B, True{})) :
    Fact<CellOut(operation, plan, dl, dr, A, B, position, _)>
  left_out(operation, plan, dl, dr, A, B, position, Array.get(F32, A, left_at(plan, position)), {==}, left_valid, right_valid)

# Every cell hands both input arrays back.
def cell_keeps(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +position: Nat,
  inside: Bool, -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Fact<{Pair.snd(F32, Array<F32> & Array<F32>, cell_of(operation, plan, position, A, B, inside)) == (A, B) : Array<F32> & Array<F32>}>:
  match inside:
    case False{}:
      Fact{
        %Equal.sym(F32 & (Array<F32> & Array<F32>), cell_of(operation, plan, position, A, B, False{}),
          Elementwise.cell(operation, Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)),
            Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position, A, B, False{}),
          WalkProof.cell_at_walk(operation, plan, position, A, B, False{})) :
          {Pair.snd(F32, Array<F32> & Array<F32>, _) == (A, B) : Array<F32> & Array<F32>}
        {==}}
    case True{}:
      Fact{Pair.fst({Pair.snd(F32, Array<F32> & Array<F32>, cell_of(operation, plan, position, A, B, True{})) == (A, B) : Array<F32> & Array<F32>},
        {Some{Pair.fst(F32, Array<F32> & Array<F32>, cell_of(operation, plan, position, A, B, True{}))} == want(operation, plan, dl, dr, A, B, position) : Maybe<&2, F32>},
        {\{Fact: q => q} : Fact<(CellOut(operation, plan, dl, dr, A, B, position, cell_of(operation, plan, position, A, B, True{})))> ->
          CellOut(operation, plan, dl, dr, A, B, position, cell_of(operation, plan, position, A, B, True{}))}(
          cell_out(operation, plan, dl, dr, A, B, position, left_valid, right_valid)))}

# A split hands each half both input arrays.
def halves_equal(+dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, lp: Array<F32> & Array<F32>, rp: Array<F32> & Array<F32>,
  -le: {lp == Array.clone(F32, A) : Array<F32> & Array<F32>}, -re: {rp == Array.clone(F32, B) : Array<F32> & Array<F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Fact<({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.halves(lp, rp)) == (A, B) : Array<F32> & Array<F32>} &
    {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.halves(lp, rp)) == (A, B) : Array<F32> & Array<F32>})>:
  (l1, l2) = lp
  (r1, r2) = rp
  Fact{(pair_equal(l1, r1, A, B,
      {\{Fact: q => q} : Fact<({Pair.fst(Array<F32>, Array<F32>, (l1, l2)) == A : Array<F32>})> -> {Pair.fst(Array<F32>, Array<F32>, (l1, l2)) == A : Array<F32>}}(
        clone_first(dl, A, A, (l1, l2), le, {==}, left_valid)),
      {\{Fact: q => q} : Fact<({Pair.fst(Array<F32>, Array<F32>, (r1, r2)) == B : Array<F32>})> -> {Pair.fst(Array<F32>, Array<F32>, (r1, r2)) == B : Array<F32>}}(
        clone_first(dr, B, B, (r1, r2), re, {==}, right_valid))),
    pair_equal(l2, r2, A, B,
      {\{Fact: q => q} : Fact<({Pair.snd(Array<F32>, Array<F32>, (l1, l2)) == A : Array<F32>})> -> {Pair.snd(Array<F32>, Array<F32>, (l1, l2)) == A : Array<F32>}}(
        clone_second(dl, A, A, (l1, l2), le, {==}, left_valid)),
      {\{Fact: q => q} : Fact<({Pair.snd(Array<F32>, Array<F32>, (r1, r2)) == B : Array<F32>})> -> {Pair.snd(Array<F32>, Array<F32>, (r1, r2)) == B : Array<F32>}}(
        clone_second(dr, B, B, (r1, r2), re, {==}, right_valid))))}

def split_equal(+dl: Nat, +dr: Nat, A: Array<F32>, B: Array<F32>, -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Fact<({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>} &
    {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>})>:
  halves_equal(dl, dr, A, B, Array.clone(F32, A), Array.clone(F32, B), {==}, {==}, left_valid, right_valid)

# An inside cell of the parallel build holds what `want` names there: the
# generic law of `Parallel.build`, with the elementwise cell.
def build_cell(+depth: Nat, +splits: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat,
  +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, +path: Model.Path(depth),
  -inside: {Nat.is_lt(Nat.add(Model.offset(depth, path), base), count) == True{} : Bool},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Fact<{Model.observe(F32, depth, Elementwise.build(depth, splits, operation, plan, count, base, A, B), path) ==
    want(operation, plan, dl, dr, A, B, Nat.add(Model.offset(depth, path), base)) : Maybe<&2, F32>}>:
  +position = Nat.add(Model.offset(depth, path), base)
  -split = {\{Fact: q => q} : Fact<({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>} &
    {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>})> ->
    {Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>} &
    {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>}}(
    split_equal(dl, dr, A, B, left_valid, right_valid))
  Fact{Equal.trans(Maybe<&2, F32>, Model.observe(F32, depth, Elementwise.build(depth, splits, operation, plan, count, base, A, B), path),
    Some{Pair.fst(F32, Array<F32> & Array<F32>, cell_of(operation, plan, position, A, B, True{}))},
    want(operation, plan, dl, dr, A, B, position),
    %inside : {Model.observe(F32, depth, Elementwise.build(depth, splits, operation, plan, count, base, A, B), path) ==
      Some{Pair.fst(F32, Array<F32> & Array<F32>, cell_of(operation, plan, position, A, B, _))} : Maybe<&2, F32>}
    {\{Generic.Fact: q => q} :
      Generic.Fact<Generic.Cells(~Elementwise.Walk, ~Array<F32>, ~Array<F32>, ~Elementwise.cell_at, Elementwise.walk(Elementwise.Request{operation, plan}), (A, B), depth, count, base,
        Elementwise.build(depth, splits, operation, plan, count, base, A, B))> ->
      Generic.Cells(~Elementwise.Walk, ~Array<F32>, ~Array<F32>, ~Elementwise.cell_at, Elementwise.walk(Elementwise.Request{operation, plan}), (A, B), depth, count, base,
        Elementwise.build(depth, splits, operation, plan, count, base, A, B))}(
      Generic.build_cells(~Elementwise.Request, ~Elementwise.Walk, ~Array<F32>, ~Array<F32>, ~Elementwise.walk, ~Elementwise.split, ~Elementwise.cell_at,
        Elementwise.Request{operation, plan}, (A, B), depth, splits, count, base, (A, B), {==},
        at => inside => {\{Fact: q => q} : Fact<({Pair.snd(F32, Array<F32> & Array<F32>, cell_of(operation, plan, at, A, B, inside)) == (A, B) : Array<F32> & Array<F32>})> ->
          {Pair.snd(F32, Array<F32> & Array<F32>, cell_of(operation, plan, at, A, B, inside)) == (A, B) : Array<F32> & Array<F32>}}(
          cell_keeps(operation, plan, dl, dr, A, B, at, inside, left_valid, right_valid)),
        Pair.fst({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>},
          {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>}, split),
        Pair.snd({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>},
          {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, B))) == (A, B) : Array<F32> & Array<F32>}, split)))(path),
    Pair.snd({Pair.snd(F32, Array<F32> & Array<F32>, cell_of(operation, plan, position, A, B, True{})) == (A, B) : Array<F32> & Array<F32>},
      {Some{Pair.fst(F32, Array<F32> & Array<F32>, cell_of(operation, plan, position, A, B, True{}))} == want(operation, plan, dl, dr, A, B, position) : Maybe<&2, F32>},
      {\{Fact: q => q} : Fact<(CellOut(operation, plan, dl, dr, A, B, position, cell_of(operation, plan, position, A, B, True{})))> ->
        CellOut(operation, plan, dl, dr, A, B, position, cell_of(operation, plan, position, A, B, True{}))}(
        cell_out(operation, plan, dl, dr, A, B, position, left_valid, right_valid))))}

# What `take` hands back for one input: the copy for the build is balanced at
# some depth and, below the input's element count, holds what the checked
# observation of the input sees; the kept tensor observes like the input.
def SourceReads(-input: Core.Dynamic, -depth: Nat, -copy: Array<F32>) -> Type:
  @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Totality.dimensions(input))) == True{} : Bool} ->
    {Model.observe(F32, depth, copy, Refinement.route(depth, index)) == Core.checked_observe_dynamic(input, index) : Maybe<&2, F32>}

type Taken<-input: Core.Dynamic, -pair: Core.Dynamic & Array<F32>> is Data:
  Taken{-depth: Nat, -balanced: Model.Balanced(F32, depth, Pair.snd(Core.Dynamic, Array<F32>, pair)),
    -reads: SourceReads(input, depth, Pair.snd(Core.Dynamic, Array<F32>, pair)),
    -retained: Totality.SameSource(input, Pair.fst(Core.Dynamic, Array<F32>, pair))}

def owned_inspect(+shape: Shape.Shape, +depth: Nat, +length: Nat, property: Totality.Property, array: Array<F32>) -> Totality.Inspection:
  match property:
    case Totality.Dimensions{}:
      Totality.Dims{Shape.dimensions(shape)}
    case Totality.Checked{+index}:
      Totality.Value{Lifecycle.checked_observe_array(depth, length, array, index)}
    case Totality.Raw{+index}:
      Totality.Value{Model.observe(F32, depth, array, Refinement.route(depth, index))}

def owned_view(+shape: Shape.Shape, +depth: Nat, +length: Nat, property: Totality.Property, -array: Array<F32>,
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  {Totality.inspect(property, Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}}) ==
    owned_inspect(shape, depth, length, property, array) : Totality.Inspection}:
  match property:
    case Totality.Dimensions{}:
      {==}
    case Totality.Checked{+index}:
      {==}
    case Totality.Raw{+index}:
      {==}

def observe_allowed_equal(+depth: Nat, +length: Nat, -array: Array<F32>, +index: U32, allowed: Bool,
  accepted: {allowed == True{} : Bool}, matched: {Storage.index_matches(depth, length, index) == True{} : Bool}) ->
  {Lifecycle.observe_allowed(depth, length, array, index, allowed) ==
    Model.observe(F32, depth, array, Refinement.route(depth, index)) : Maybe<&2, F32>}:
  match allowed:
    case False{}:
      Empty.absurd({None{} == Model.observe(F32, depth, array, Refinement.route(depth, index)) : Maybe<&2, F32>}, Arith.clash(accepted))
    case True{}:
      %Equal.sym(Bool, Storage.index_matches(depth, length, index), True{}, matched) :
        {Lifecycle.observe_index(depth, length, array, index, _) == Model.observe(F32, depth, array, Refinement.route(depth, index)) : Maybe<&2, F32>}
      {==}

def taken_owned(+shape: Shape.Shape, +depth: Nat, +length: Nat, -array: Array<F32>, pair: Array<F32> & Array<F32>,
  -same: {pair == Array.clone(F32, array) : Array<F32> & Array<F32>}, -balanced: Model.Balanced(F32, depth, array),
  -valid: {Storage.bounds(depth, length) == True{} : Bool}, -exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Taken<Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}},
    Elementwise.take_owned(shape, depth, length, array, pair, same, balanced, valid, exact)>:
  (kept, copy) = pair
  -kept_same = {\{Fact: p => p} : Fact<({Pair.fst(Array<F32>, Array<F32>, (kept, copy)) == array : Array<F32>})> -> {kept == array : Array<F32>}}(
    clone_first(depth, array, array, (kept, copy), same, {==}, balanced))
  -copy_same = {\{Fact: p => p} : Fact<({Pair.snd(Array<F32>, Array<F32>, (kept, copy)) == array : Array<F32>})> -> {copy == array : Array<F32>}}(
    clone_second(depth, array, array, (kept, copy), same, {==}, balanced))
  Taken{depth,
    {\{Fact: p => p} : Fact<(Model.Balanced(F32, depth, copy))> -> Model.Balanced(F32, depth, copy)}(transport_balanced(depth, copy, array, copy_same, balanced)),
    index => inside => Equal.trans(Maybe<&2, F32>, Model.observe(F32, depth, copy, Refinement.route(depth, index)),
      Model.observe(F32, depth, array, Refinement.route(depth, index)),
      Lifecycle.observe_allowed(depth, length, array, index, Nat.is_lt(U32.to_nat(index), length)),
      Equal.cong(Array<F32>, Maybe<&2, F32>, a => Model.observe(F32, depth, a, Refinement.route(depth, index)), copy, array, copy_same),
      Equal.sym(Maybe<&2, F32>, Lifecycle.observe_allowed(depth, length, array, index, Nat.is_lt(U32.to_nat(index), length)),
        Model.observe(F32, depth, array, Refinement.route(depth, index)),
        observe_allowed_equal(depth, length, array, index, Nat.is_lt(U32.to_nat(index), length),
          %Equal.sym(Nat, length, Shape.logical_numel(Shape.dimensions(shape)), exact) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
          inside,
          Equal.sym(Bool, True{}, Storage.index_matches(depth, length, index),
            Route.index_complete(depth, length, index, valid,
              %Equal.sym(Nat, length, Shape.logical_numel(Shape.dimensions(shape)), exact) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
              inside))))),
    property => Equal.trans(Totality.Inspection,
      Totality.inspect(property, Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, kept,
        Certificates.clone_first(F32, depth, array, (kept, copy), same, balanced), valid}, exact}}),
      owned_inspect(shape, depth, length, property, kept),
      Totality.inspect(property, Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}}),
      owned_view(shape, depth, length, property, kept, Certificates.clone_first(F32, depth, array, (kept, copy), same, balanced), valid, exact),
      Equal.trans(Totality.Inspection, owned_inspect(shape, depth, length, property, kept), owned_inspect(shape, depth, length, property, array),
        Totality.inspect(property, Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}}),
        Equal.cong(Array<F32>, Totality.Inspection, a => owned_inspect(shape, depth, length, property, a), kept, array, kept_same),
        Equal.sym(Totality.Inspection, Totality.inspect(property, Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}}),
          owned_inspect(shape, depth, length, property, array), owned_view(shape, depth, length, property, array, balanced, valid, exact))))}

def taken(dynamic: Core.Dynamic) -> Taken<dynamic, Elementwise.take(dynamic)>:
  Core.Packed{+shape, Core.TTensor{storage, exact}} = dynamic
  match storage:
    case Storage.Empty{}:
      Taken{0n, Unit{},
        index => inside => Empty.absurd({Model.observe(F32, 0n, Array.new(F32, 0n, 0.0), Refinement.route(0n, index)) ==
          Core.checked_observe_dynamic(Core.Packed{shape, Core.TTensor{Storage.Empty{}, exact}}, index) : Maybe<&2, F32>},
          Arith.lt_zero(U32.to_nat(index),
            %Equal.sym(Nat, 0n, Shape.logical_numel(Shape.dimensions(shape)), exact) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
            inside)),
        property => {==}}
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      taken_owned(shape, depth, length, array, Array.clone(F32, array), {==}, balanced, valid, exact)

def walk_pair(+plan: Shape.BroadcastPlan,
  +map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>,
  +value: Nat, +bound: {True{} == Nat.is_lt(value, Shape.elements(Shape.output_shape(plan))) : Bool}) ->
  {(Shape.index_value(Shape.left_shape(plan), Flat.project_left_index(plan, map, Shape.FIndex{value, bound})),
    Shape.index_value(Shape.right_shape(plan), Flat.project_right_index(plan, map, Shape.FIndex{value, bound}))) ==
    (U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), map, value)),
      U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), map, value))) : U32 & U32}:
  match map:
    case Shape.Indexed{rows, exact}:
      {==}
    case Shape.Unindexed{zero}:
      Empty.absurd({(Shape.index_value(Shape.left_shape(plan), Flat.project_left_index(plan, Shape.Unindexed{zero}, Shape.FIndex{value, bound})),
        Shape.index_value(Shape.right_shape(plan), Flat.project_right_index(plan, Shape.Unindexed{zero}, Shape.FIndex{value, bound}))) == (U32.from_nat(0n), U32.from_nat(0n)) : U32 & U32},
        Flat.hollow(Shape.output_shape(plan), value, zero, bound))

def left_walk_bound(+plan: Shape.BroadcastPlan,
  +map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>,
  +value: Nat, +bound: {True{} == Nat.is_lt(value, Shape.elements(Shape.output_shape(plan))) : Bool}) ->
  {True{} == Nat.is_lt(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), map, value),
    Shape.elements(Shape.left_shape(plan))) : Bool}:
  match map:
    case Shape.Indexed{rows, exact}:
      Flat.left_bound(plan, rows, exact, value, bound)
    case Shape.Unindexed{zero}:
      Empty.absurd({True{} == Nat.is_lt(0n, Shape.elements(Shape.left_shape(plan))) : Bool}, Flat.hollow(Shape.output_shape(plan), value, zero, bound))

def right_walk_bound(+plan: Shape.BroadcastPlan,
  +map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>,
  +value: Nat, +bound: {True{} == Nat.is_lt(value, Shape.elements(Shape.output_shape(plan))) : Bool}) ->
  {True{} == Nat.is_lt(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), map, value),
    Shape.elements(Shape.right_shape(plan))) : Bool}:
  match map:
    case Shape.Indexed{rows, exact}:
      Flat.right_bound(plan, rows, exact, value, bound)
    case Shape.Unindexed{zero}:
      Empty.absurd({True{} == Nat.is_lt(0n, Shape.elements(Shape.right_shape(plan))) : Bool}, Flat.hollow(Shape.output_shape(plan), value, zero, bound))

# Below the output's element count, `expected` reads the inputs at the cells
# the build's index walks name.
def decided_walks(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, -left: Core.Dynamic, -right: Core.Dynamic, +position: Nat,
  decision: Shape.BoolCase<Nat.is_lt(position, Shape.elements(Shape.output_shape(plan)))>,
  +bound: {True{} == Nat.is_lt(position, Shape.elements(Shape.output_shape(plan))) : Bool}) ->
  {Elementwise.expected_indices(operation, left, right, Elementwise.flat_input_indices(plan, Shape.flat_checked.finish(Shape.output_shape(plan), position, decision))) ==
    Elementwise.expected_left(operation, right,
      U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position)),
      Core.checked_observe_dynamic(left,
        U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position)))) : Maybe<&2, F32>}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({None{} ==
        Elementwise.expected_left(operation, right,
          U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position)),
          Core.checked_observe_dynamic(left,
            U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position)))) : Maybe<&2, F32>},
        Arith.clash(Equal.trans(Bool, False{}, Nat.is_lt(position, Shape.elements(Shape.output_shape(plan))), True{}, equal,
          Equal.sym(Bool, True{}, Nat.is_lt(position, Shape.elements(Shape.output_shape(plan))), bound))))
    case Shape.BTrue{equal}:
      Equal.cong(U32 & U32, Maybe<&2, F32>, indices => Elementwise.expected_indices(operation, left, right, Done{indices}),
        (Shape.index_value(Shape.left_shape(plan), Flat.project_left_index(plan, Shape.index_map(plan), Shape.FIndex{position, equal})),
          Shape.index_value(Shape.right_shape(plan), Flat.project_right_index(plan, Shape.index_map(plan), Shape.FIndex{position, equal}))),
        (U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position)),
          U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position))),
        walk_pair(plan, Shape.index_map(plan), position, equal))

def pick_expected(+operation: Elementwise.Operation, value: Maybe<&2, F32>, -right: Core.Dynamic, -right_index: U32, -other: Maybe<&2, F32>,
  same: {other == Core.checked_observe_dynamic(right, right_index) : Maybe<&2, F32>}) ->
  {pick(operation, value, other) == Elementwise.expected_left(operation, right, right_index, value) : Maybe<&2, F32>}:
  match value:
    case None{}:
      {==}
    case Some{value}:
      %same : {Elementwise.expected_right(operation, value, other) == Elementwise.expected_right(operation, value, _) : Maybe<&2, F32>}
      {==}

def add_zero_right(+number: Nat) -> {Nat.add(number, 0n) == number : Nat}:
  match number:
    case 0n:
      {==}
    case 1n+p:
      Equal.cong(Nat, Nat, n => 1n+n, Nat.add(p, 0n), p, add_zero_right(p))

def output_offset(+depth: Nat, +length: Nat, +index: U32, +valid: {Storage.bounds(depth, length) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index), length) == True{} : Bool}) ->
  {U32.to_nat(index) == Model.offset(depth, Refinement.route(depth, index)) : Nat}:
  +rest = Arith.and_right(Nat.is_le(depth, 20n), Bool.and(Nat.is_lt(0n, length), Nat.is_le(length, Model.capacity(depth))), valid)
  Route.route_offset(depth, index,
    Arith.and_left(Nat.is_le(depth, 20n), Bool.and(Nat.is_lt(0n, length), Nat.is_le(length, Model.capacity(depth))), valid),
    Arith.lt_le_trans(U32.to_nat(index), length, Model.capacity(depth), inside,
      Arith.and_right(Nat.is_lt(0n, length), Nat.is_le(length, Model.capacity(depth)), rest)))

def word_exact(+shape: Shape.Shape, +cursor: Nat, within: {Nat.is_le(cursor, Shape.elements(shape)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat}:
  WordConversion.natural_word_exact(cursor, Shape.count(shape),
    %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.elements(shape), WordConversion.shape_count_exact(shape)) :
      {Nat.is_le(cursor, _) == True{} : Bool}
    within)

# A cell an index walk names lies inside the input it reads.
def side_inside(-input: Core.Dynamic, +side: Shape.Shape, +walk: Nat,
  fits: {Totality.dimensions(input) == Shape.dimensions(side) : List<&2, U32>},
  +bound: {True{} == Nat.is_lt(walk, Shape.elements(side)) : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(walk)), Shape.logical_numel(Totality.dimensions(input))) == True{} : Bool}:
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(walk)), walk,
    word_exact(side, walk, WordConversion.lt_le(walk, Shape.elements(side), Equal.sym(Bool, True{}, Nat.is_lt(walk, Shape.elements(side)), bound)))) :
    {Nat.is_lt(_, Shape.logical_numel(Totality.dimensions(input))) == True{} : Bool}
  %Equal.sym(List<&2, U32>, Totality.dimensions(input), Shape.dimensions(side), fits) :
    {Nat.is_lt(walk, Shape.logical_numel(_)) == True{} : Bool}
  %Shape.elements_exact(side) : {Nat.is_lt(walk, _) == True{} : Bool}
  Equal.sym(Bool, True{}, Nat.is_lt(walk, Shape.elements(side)), bound)

def output_inside(+plan: Shape.BroadcastPlan, +length: Nat, +index: U32,
  +exact: {length == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat},
  +bound: Shape.LT(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))))) -> {Nat.is_lt(U32.to_nat(index), length) == True{} : Bool}:
  %Equal.sym(Nat, length, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), exact) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
  Equal.sym(Bool, True{}, Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))), Nats.lt_true(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), bound))

def output_bound(+plan: Shape.BroadcastPlan, +index: U32, +bound: Shape.LT(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))))) ->
  {True{} == Nat.is_lt(U32.to_nat(index), Shape.elements(Shape.output_shape(plan))) : Bool}:
  %Equal.sym(Nat, Shape.elements(Shape.output_shape(plan)), Shape.numel(Shape.extents(Shape.output_shape(plan))), Shape.elements_exact(Shape.output_shape(plan))) :
    {True{} == Nat.is_lt(U32.to_nat(index), _) : Bool}
  Nats.lt_true(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), bound)

def output_position(+depth: Nat, +length: Nat, +index: U32, +valid: {Storage.bounds(depth, length) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index), length) == True{} : Bool}) ->
  {Nat.add(Model.offset(depth, Refinement.route(depth, index)), 0n) == U32.to_nat(index) : Nat}:
  Equal.trans(Nat, Nat.add(Model.offset(depth, Refinement.route(depth, index)), 0n), Model.offset(depth, Refinement.route(depth, index)), U32.to_nat(index), add_zero_right(Model.offset(depth, Refinement.route(depth, index))),
    Equal.sym(Nat, U32.to_nat(index), Model.offset(depth, Refinement.route(depth, index)), output_offset(depth, length, index, valid, inside)))

# At an index below the output's element count, the parallel build holds the
# value `expected` names there.
def output_cell(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +depth: Nat, +length: Nat,
  +valid: {Storage.bounds(depth, length) == True{} : Bool}, +exact: {length == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat},
  -left: Core.Dynamic, -right: Core.Dynamic, la: Array<F32>, ra: Array<F32>, +dl: Nat, +dr: Nat,
  -left_valid: Model.Balanced(F32, dl, la), -right_valid: Model.Balanced(F32, dr, ra),
  -left_reads: SourceReads(left, dl, la), -right_reads: SourceReads(right, dr, ra),
  +left_fits: {Totality.dimensions(left) == Shape.dimensions(Shape.left_shape(plan)) : List<&2, U32>},
  +right_fits: {Totality.dimensions(right) == Shape.dimensions(Shape.right_shape(plan)) : List<&2, U32>},
  +index: U32, +bound: Shape.LT(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))))) ->
  Fact<{Model.observe(F32, depth, Elementwise.build(depth, Elementwise.splits(depth), operation, plan, length, 0n, la, ra), Refinement.route(depth, index)) == Elementwise.expected(operation, plan, left, right, index) : Maybe<&2, F32>}>:
  +inside = output_inside(plan, length, index, exact, bound)
  +within = output_bound(plan, index, bound)
  +position = output_position(depth, length, index, valid, inside)
  Fact{Equal.trans(Maybe<&2, F32>, Model.observe(F32, depth, Elementwise.build(depth, Elementwise.splits(depth), operation, plan, length, 0n, la, ra), Refinement.route(depth, index)), want(operation, plan, dl, dr, la, ra, Nat.add(Model.offset(depth, Refinement.route(depth, index)), 0n)), Elementwise.expected(operation, plan, left, right, index),
    {\{Fact: q => q} : Fact<({Model.observe(F32, depth, Elementwise.build(depth, Elementwise.splits(depth), operation, plan, length, 0n, la, ra), Refinement.route(depth, index)) == want(operation, plan, dl, dr, la, ra, Nat.add(Model.offset(depth, Refinement.route(depth, index)), 0n)) : Maybe<&2, F32>})> ->
      {Model.observe(F32, depth, Elementwise.build(depth, Elementwise.splits(depth), operation, plan, length, 0n, la, ra), Refinement.route(depth, index)) == want(operation, plan, dl, dr, la, ra, Nat.add(Model.offset(depth, Refinement.route(depth, index)), 0n)) : Maybe<&2, F32>}}(
      build_cell(depth, Elementwise.splits(depth), operation, plan, length, 0n, dl, dr, la, ra, Refinement.route(depth, index),
        %Equal.sym(Nat, Nat.add(Model.offset(depth, Refinement.route(depth, index)), 0n), U32.to_nat(index), position) : {Nat.is_lt(_, length) == True{} : Bool}
        inside, left_valid, right_valid)),
    Equal.trans(Maybe<&2, F32>, want(operation, plan, dl, dr, la, ra, Nat.add(Model.offset(depth, Refinement.route(depth, index)), 0n)), want(operation, plan, dl, dr, la, ra, U32.to_nat(index)), Elementwise.expected(operation, plan, left, right, index),
      Equal.cong(Nat, Maybe<&2, F32>, at => want(operation, plan, dl, dr, la, ra, at), Nat.add(Model.offset(depth, Refinement.route(depth, index)), 0n), U32.to_nat(index), position),
      Equal.trans(Maybe<&2, F32>, want(operation, plan, dl, dr, la, ra, U32.to_nat(index)), Elementwise.expected_left(operation, right, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index))), Model.observe(F32, dl, la, Refinement.route(dl, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index)))))), Elementwise.expected(operation, plan, left, right, index),
        pick_expected(operation, Model.observe(F32, dl, la, Refinement.route(dl, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index))))), right, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index))), Model.observe(F32, dr, ra, Refinement.route(dr, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index))))),
          right_reads(U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index))))(side_inside(right, Shape.right_shape(plan), Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index)), right_fits,
            right_walk_bound(plan, Shape.index_map(plan), U32.to_nat(index), within)))),
        Equal.trans(Maybe<&2, F32>, Elementwise.expected_left(operation, right, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index))), Model.observe(F32, dl, la, Refinement.route(dl, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index)))))), Elementwise.expected_left(operation, right, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index))), Core.checked_observe_dynamic(left, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index))))), Elementwise.expected(operation, plan, left, right, index),
          Equal.cong(Maybe<&2, F32>, Maybe<&2, F32>, value => Elementwise.expected_left(operation, right, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index))), value), Model.observe(F32, dl, la, Refinement.route(dl, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index))))), Core.checked_observe_dynamic(left, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index)))),
            left_reads(U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index))))(side_inside(left, Shape.left_shape(plan), Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index)), left_fits,
              left_walk_bound(plan, Shape.index_map(plan), U32.to_nat(index), within)))),
          Equal.sym(Maybe<&2, F32>, Elementwise.expected(operation, plan, left, right, index), Elementwise.expected_left(operation, right, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index))), Core.checked_observe_dynamic(left, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), U32.to_nat(index))))),
            decided_walks(operation, plan, left, right, U32.to_nat(index), Shape.bool_case(Nat.is_lt(U32.to_nat(index), Shape.elements(Shape.output_shape(plan)))), within))))))}

def output_mapping(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool}, -exact: {length == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat},
  -left: Core.Dynamic, -right: Core.Dynamic, taken_left: Core.Dynamic & Array<F32>, taken_right: Core.Dynamic & Array<F32>,
  -left_taken: Taken<left, taken_left>, -right_taken: Taken<right, taken_right>,
  +left_fits: {Totality.dimensions(left) == Shape.dimensions(Shape.left_shape(plan)) : List<&2, U32>},
  +right_fits: {Totality.dimensions(right) == Shape.dimensions(Shape.right_shape(plan)) : List<&2, U32>}) ->
  Elementwise.MappingModel<operation, plan, left, right, Elementwise.parallel_output(operation, plan, depth, length, valid, exact, taken_left, taken_right)>:
  (kept_left, la) = taken_left
  (kept_right, ra) = taken_right
  Elementwise.MappingModel{index => bound =>
    {\{Taken: dl => left_valid => left_reads => left_same =>
      {\{Taken: dr => right_valid => right_reads => right_same =>
        {\{Fact: equal => equal} : Fact<({Model.observe(F32, depth, Elementwise.build(depth, Elementwise.splits(depth), operation, plan, length, 0n, la, ra), Refinement.route(depth, index)) == Elementwise.expected(operation, plan, left, right, index) : Maybe<&2, F32>})> -> {Model.observe(F32, depth, Elementwise.build(depth, Elementwise.splits(depth), operation, plan, length, 0n, la, ra), Refinement.route(depth, index)) == Elementwise.expected(operation, plan, left, right, index) : Maybe<&2, F32>}}(
          output_cell(operation, plan, depth, length, valid, exact, left, right, la, ra, dl, dr, left_valid, right_valid, left_reads, right_reads,
            left_fits, right_fits, index, bound))} : Taken<right, (kept_right, ra)> -> {Model.observe(F32, depth, Elementwise.build(depth, Elementwise.splits(depth), operation, plan, length, 0n, la, ra), Refinement.route(depth, index)) == Elementwise.expected(operation, plan, left, right, index) : Maybe<&2, F32>}}(right_taken)} : Taken<left, (kept_left, la)> -> {Model.observe(F32, depth, Elementwise.build(depth, Elementwise.splits(depth), operation, plan, length, 0n, la, ra), Refinement.route(depth, index)) == Elementwise.expected(operation, plan, left, right, index) : Maybe<&2, F32>}}(left_taken)}

def allocated_mapping(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, left: Core.Dynamic, right: Core.Dynamic,
  result: Result<&2, &1, Core.Error, Core.Tensor<Shape.output_shape(plan)>>,
  +left_fits: {Totality.dimensions(left) == Shape.dimensions(Shape.left_shape(plan)) : List<&2, U32>},
  +right_fits: {Totality.dimensions(right) == Shape.dimensions(Shape.right_shape(plan)) : List<&2, U32>}) ->
  Elementwise.MappingModel<operation, plan, left, right, Elementwise.parallel_allocated(operation, plan, left, right, result)>:
  match result:
    case Fail{error}:
      Elementwise.MappingModel{Unit{}}
    case Done{Core.TTensor{storage, exact}}:
      match storage:
        case Storage.Empty{}:
          Elementwise.MappingModel{index => bound => Empty.absurd(
            {Core.observe_dynamic(Core.Packed{Shape.output_shape(plan), Core.TTensor{Storage.Empty{}, exact}}, index) == Elementwise.expected(operation, plan, left, right, index) : Maybe<&2, F32>},
            Arith.lt_zero(U32.to_nat(index),
              %Equal.sym(Nat, 0n, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), exact) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
              Equal.sym(Bool, True{}, Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))), Nats.lt_true(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), bound))))}
        case Storage.Owned{+depth, +length, array, balanced, valid}:
          output_mapping(operation, plan, depth, length, valid, exact, left, right, Elementwise.take(left), Elementwise.take(right),
            taken(left), taken(right), left_fits, right_fits)

def planned_mapping(+operation: Elementwise.Operation, +left_shape: Shape.Shape, +right_shape: Shape.Shape,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>, left: Core.Tensor<left_shape>, right: Core.Tensor<right_shape>,
  -same: {result == Elementwise.plan_for(Core.Packed{left_shape, left}, Core.Packed{right_shape, right}) : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  +sources: {ShapeLaws.planned_dimensions(left_shape, right_shape, result) == (Shape.dimensions(left_shape), Shape.dimensions(right_shape)) : List<&2, U32> & List<&2, U32>}) ->
  Elementwise.BinaryModel<operation, Core.Packed{left_shape, left}, Core.Packed{right_shape, right}, Elementwise.parallel_planned(operation, Core.Packed{left_shape, left}, Core.Packed{right_shape, right}, result)>:
  match result:
    case Fail{error}:
      Elementwise.BinaryModel{
        %same : Elementwise.PlannedMeaning(operation, Core.Packed{left_shape, left}, Core.Packed{right_shape, right}, Elementwise.parallel_planned(operation, Core.Packed{left_shape, left}, Core.Packed{right_shape, right}, Fail{error}), _)
        Unit{}}
    case Done{+plan}:
      Elementwise.BinaryModel{
        %same : Elementwise.PlannedMeaning(operation, Core.Packed{left_shape, left}, Core.Packed{right_shape, right}, Elementwise.parallel_planned(operation, Core.Packed{left_shape, left}, Core.Packed{right_shape, right}, Done{plan}), _)
        {\{Elementwise.MappingModel: mapping => mapping} :
          Elementwise.MappingModel<operation, plan, Core.Packed{left_shape, left}, Core.Packed{right_shape, right}, Elementwise.parallel_allocated(operation, plan, Core.Packed{left_shape, left}, Core.Packed{right_shape, right}, Core.new(Shape.output_shape(plan), 0.0))> -> Elementwise.SuccessfulMapping(operation, plan, Core.Packed{left_shape, left}, Core.Packed{right_shape, right}, Elementwise.parallel_allocated(operation, plan, Core.Packed{left_shape, left}, Core.Packed{right_shape, right}, Core.new(Shape.output_shape(plan), 0.0)))}(
            allocated_mapping(operation, plan, Core.Packed{left_shape, left}, Core.Packed{right_shape, right}, Core.new(Shape.output_shape(plan), 0.0),
              Equal.sym(List<&2, U32>, Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(left_shape),
                Equal.cong(List<&2, U32> & List<&2, U32>, List<&2, U32>, Pair.fst(List<&2, U32>, List<&2, U32>),
                  (Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(Shape.right_shape(plan))), (Shape.dimensions(left_shape), Shape.dimensions(right_shape)), sources)),
              Equal.sym(List<&2, U32>, Shape.dimensions(Shape.right_shape(plan)), Shape.dimensions(right_shape),
                Equal.cong(List<&2, U32> & List<&2, U32>, List<&2, U32>, Pair.snd(List<&2, U32>, List<&2, U32>),
                  (Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(Shape.right_shape(plan))), (Shape.dimensions(left_shape), Shape.dimensions(right_shape)), sources))))}

# The parallel `binary` meets the same `BinaryModel` as the sequential loop.
def binary_mapping(+operation: Elementwise.Operation, left: Core.Dynamic, right: Core.Dynamic) ->
  Elementwise.BinaryModel<operation, left, right, Elementwise.binary(operation, left, right)>:
  Core.Packed{+left_shape, left_tensor} = left
  Core.Packed{+right_shape, right_tensor} = right
  planned_mapping(operation, left_shape, right_shape, Shape.broadcast_checked(left_shape, right_shape), left_tensor, right_tensor, {==},
    PlanProof.broadcast_checked_plan(left_shape, right_shape))
