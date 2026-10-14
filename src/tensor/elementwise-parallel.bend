import Base
import ../shape/core.bend as Shape
import ../shape/flat.bend as Flat
import ../shape/nat_proofs.bend as Nats
import ../shape/index_proofs.bend as Index
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ../storage/clone.bend as Clone
import ../storage/arith.bend as Arith
import ./elementwise.bend as Elementwise

# Proofs that the parallel elementwise build writes, at every output index,
# the word the sequential loop writes there.

type Fact<-P: Type> is Data:
  Fact{-proof: P}

# The value both loops write from the observed input cells.
def pick(operation: Elementwise.Operation, left: Maybe<&2, F32>, right: Maybe<&2, F32>) -> Maybe<&2, F32>:
  match left:
    case None{}:
      None{}
    case Some{value}:
      Elementwise.expected_right(operation, value, right)

def want(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +dl: Nat, +dr: Nat, A: Array<F32>, B: Array<F32>, +position: Nat) ->
  Maybe<&2, F32>:
  pick(operation, Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position)))), Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position)))))

# Every slot of `out` whose index `base + offset` is below `count` holds the
# value `want` gives for that index.
def Cells(+depth: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, -out: Array<F32>) -> Type:
  @+path: Model.Path(depth) -> @inside: {Nat.is_lt(Nat.add(Model.offset(depth, path), base), count) == True{} : Bool} ->
    {Model.observe(F32, depth, out, path) == want(operation, plan, dl, dr, A, B, Nat.add(Model.offset(depth, path), base)) : Maybe<&2, F32>}

def TreeFacts(+depth: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, -result: Array<F32> & (Array<F32> & Array<F32>)) -> Type:
  {Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == A : Array<F32>} & ({Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == B : Array<F32>} & Cells(depth, operation, plan, count, base, dl, dr, A, B, Pair.fst(Array<F32>, Array<F32> & Array<F32>, result)))

def facts_left(+depth: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, -result: Array<F32> & (Array<F32> & Array<F32>), -facts: Fact<TreeFacts(depth, operation, plan, count, base, dl, dr, A, B, result)>) ->
  Fact<{Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == A : Array<F32>}>:
  Fact{Pair.fst({Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == A : Array<F32>}, ({Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == B : Array<F32>} & Cells(depth, operation, plan, count, base, dl, dr, A, B, Pair.fst(Array<F32>, Array<F32> & Array<F32>, result))),
    {\{Fact: p => p} : Fact<(TreeFacts(depth, operation, plan, count, base, dl, dr, A, B, result))> -> TreeFacts(depth, operation, plan, count, base, dl, dr, A, B, result)}(facts))}

def facts_right(+depth: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, -result: Array<F32> & (Array<F32> & Array<F32>), -facts: Fact<TreeFacts(depth, operation, plan, count, base, dl, dr, A, B, result)>) ->
  Fact<{Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == B : Array<F32>}>:
  Fact{Pair.fst({Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == B : Array<F32>}, Cells(depth, operation, plan, count, base, dl, dr, A, B, Pair.fst(Array<F32>, Array<F32> & Array<F32>, result)),
    Pair.snd({Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == A : Array<F32>}, ({Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == B : Array<F32>} & Cells(depth, operation, plan, count, base, dl, dr, A, B, Pair.fst(Array<F32>, Array<F32> & Array<F32>, result))),
      {\{Fact: p => p} : Fact<(TreeFacts(depth, operation, plan, count, base, dl, dr, A, B, result))> -> TreeFacts(depth, operation, plan, count, base, dl, dr, A, B, result)}(facts)))}

def facts_cells(+depth: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, -result: Array<F32> & (Array<F32> & Array<F32>), -facts: Fact<TreeFacts(depth, operation, plan, count, base, dl, dr, A, B, result)>) ->
  Fact<Cells(depth, operation, plan, count, base, dl, dr, A, B, Pair.fst(Array<F32>, Array<F32> & Array<F32>, result))>:
  Fact{Pair.snd({Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == B : Array<F32>}, Cells(depth, operation, plan, count, base, dl, dr, A, B, Pair.fst(Array<F32>, Array<F32> & Array<F32>, result)),
    Pair.snd({Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == A : Array<F32>}, ({Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == B : Array<F32>} & Cells(depth, operation, plan, count, base, dl, dr, A, B, Pair.fst(Array<F32>, Array<F32> & Array<F32>, result))),
      {\{Fact: p => p} : Fact<(TreeFacts(depth, operation, plan, count, base, dl, dr, A, B, result))> -> TreeFacts(depth, operation, plan, count, base, dl, dr, A, B, result)}(facts)))}

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

def cell_right_facts(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, -x: F32, -la: Array<F32>, -ra: Array<F32>, pair: Array<F32> & F32,
  -exact: {pair == Array.get(F32, ra, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base))) : Array<F32> & F32},
  -ex: {Some{x} == Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base)))) : Maybe<&2, F32>}, -el: {la == A : Array<F32>}, -er: {ra == B : Array<F32>},
  -valid: Model.Balanced(F32, dr, B)) ->
  Fact<TreeFacts(0n, operation, plan, count, base, dl, dr, A, B, Elementwise.cell_right(operation, x, la, pair))>:
  (kept, y) = pair
  Fact{(el, ({\{Fact: p => p} : Fact<({Refinement.first(F32, (kept, y)) == B : Array<F32>})> -> {Refinement.first(F32, (kept, y)) == B : Array<F32>}}(read_keeps(dr, ra, B, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base)), (kept, y), exact, er, valid)),
    path => inside => {\{Fact: p => p} : Fact<({Some{Elementwise.scalar(operation, x, y)} == pick(operation, Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base)))), Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base))))) : Maybe<&2, F32>})> -> {Some{Elementwise.scalar(operation, x, y)} == pick(operation, Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base)))), Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base))))) : Maybe<&2, F32>}}(pick_value(operation, x, y, Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base)))), Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base)))), ex, {\{Fact: p => p} : Fact<({Some{Refinement.second(F32, (kept, y))} == Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base)))) : Maybe<&2, F32>})> -> {Some{Refinement.second(F32, (kept, y))} == Model.observe(F32, dr, B, Refinement.route(dr, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base)))) : Maybe<&2, F32>}}(read_value(dr, ra, B, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base)), (kept, y), exact, er, valid))))))}

def cell_left_facts(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, -la: Array<F32>, ra: Array<F32>, pair: Array<F32> & F32,
  -exact: {pair == Array.get(F32, la, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base))) : Array<F32> & F32},
  -el: {la == A : Array<F32>}, -er: {ra == B : Array<F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Fact<TreeFacts(0n, operation, plan, count, base, dl, dr, A, B, Elementwise.cell_left(operation, ra, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base)), pair))>:
  (kept, x) = pair
  cell_right_facts(operation, plan, count, base, dl, dr, A, B, x, kept, ra, Array.get(F32, ra, U32.from_nat(Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base))), {==},
    {\{Fact: p => p} : Fact<({Some{Refinement.second(F32, (kept, x))} == Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base)))) : Maybe<&2, F32>})> -> {Some{Refinement.second(F32, (kept, x))} == Model.observe(F32, dl, A, Refinement.route(dl, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base)))) : Maybe<&2, F32>}}(read_value(dl, la, A, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base)), (kept, x), exact, el, left_valid)),
    {\{Fact: p => p} : Fact<({Refinement.first(F32, (kept, x)) == A : Array<F32>})> -> {Refinement.first(F32, (kept, x)) == A : Array<F32>}}(read_keeps(dl, la, A, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base)), (kept, x), exact, el, left_valid)),
    er, right_valid)

def cell_facts(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, la: Array<F32>, ra: Array<F32>, inside: Bool,
  -decided: {inside == Nat.is_lt(base, count) : Bool}, -el: {la == A : Array<F32>}, -er: {ra == B : Array<F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Fact<TreeFacts(0n, operation, plan, count, base, dl, dr, A, B, Elementwise.cell(operation, Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base, la, ra, inside))>:
  match inside:
    case False{}:
      Fact{(el, (er, path => below => Empty.absurd(
        {Model.observe(F32, 0n, ALeaf{0.0}, path) == want(operation, plan, dl, dr, A, B, base) : Maybe<&2, F32>},
        Arith.clash(Equal.trans(Bool, False{}, Nat.is_lt(base, count), True{}, decided, below)))))}
    case True{}:
      cell_left_facts(operation, plan, count, base, dl, dr, A, B, la, ra, Array.get(F32, la, U32.from_nat(Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), base))), {==}, el, er, left_valid, right_valid)

def position_step(+capacity: Nat, +offset: Nat, +base: Nat) ->
  {Nat.add(offset, Nat.add(capacity, base)) == Nat.add(Nat.add(capacity, offset), base) : Nat}:
  Equal.trans(Nat, Nat.add(offset, Nat.add(capacity, base)), Nat.add(capacity, Nat.add(offset, base)), Nat.add(Nat.add(capacity, offset), base),
    Nats.add_swap(offset, capacity, base), Index.add_assoc(capacity, offset, base))

def node_cell(+inner: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, -a: Array<F32>, -b: Array<F32>, +path: Model.Path(1n+inner),
  -inside: {Nat.is_lt(Nat.add(Model.offset(1n+inner, path), base), count) == True{} : Bool},
  -low: Fact<Cells(inner, operation, plan, count, base, dl, dr, A, B, a)>, -high: Fact<Cells(inner, operation, plan, count, Nat.add(Model.capacity(inner), base), dl, dr, A, B, b)>) ->
  Fact<{Model.observe(F32, 1n+inner, ANode{a, b}, path) == want(operation, plan, dl, dr, A, B, Nat.add(Model.offset(1n+inner, path), base)) : Maybe<&2, F32>}>:
  Model.Step{+bit, +rest} = path
  match bit:
    case False{}:
      Fact{{\{Fact: p => p} : Fact<(Cells(inner, operation, plan, count, base, dl, dr, A, B, a))> -> Cells(inner, operation, plan, count, base, dl, dr, A, B, a)}(low)(rest)(inside)}
    case True{}:
      Fact{
        %position_step(Model.capacity(inner), Model.offset(inner, rest), base) :
          {Model.observe(F32, inner, b, rest) == want(operation, plan, dl, dr, A, B, _) : Maybe<&2, F32>}
        {\{Fact: p => p} : Fact<(Cells(inner, operation, plan, count, Nat.add(Model.capacity(inner), base), dl, dr, A, B, b))> -> Cells(inner, operation, plan, count, Nat.add(Model.capacity(inner), base), dl, dr, A, B, b)}(high)(rest)(
          %Equal.sym(Nat, Nat.add(Model.offset(inner, rest), Nat.add(Model.capacity(inner), base)), Nat.add(Nat.add(Model.capacity(inner), Model.offset(inner, rest)), base),
            position_step(Model.capacity(inner), Model.offset(inner, rest), base)) : {Nat.is_lt(_, count) == True{} : Bool}
          inside)}

def node_cells(+inner: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, -a: Array<F32>, -b: Array<F32>,
  -low: Fact<Cells(inner, operation, plan, count, base, dl, dr, A, B, a)>, -high: Fact<Cells(inner, operation, plan, count, Nat.add(Model.capacity(inner), base), dl, dr, A, B, b)>) ->
  Fact<Cells(1n+inner, operation, plan, count, base, dl, dr, A, B, ANode{a, b})>:
  Fact{path => inside => {\{Fact: p => p} : Fact<({Model.observe(F32, 1n+inner, ANode{a, b}, path) == want(operation, plan, dl, dr, A, B, Nat.add(Model.offset(1n+inner, path), base)) : Maybe<&2, F32>})> -> {Model.observe(F32, 1n+inner, ANode{a, b}, path) == want(operation, plan, dl, dr, A, B, Nat.add(Model.offset(1n+inner, path), base)) : Maybe<&2, F32>}}(node_cell(inner, operation, plan, count, base, dl, dr, A, B, a, b, path, inside, low, high))}

def tree_join_facts(+inner: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, -a: Array<F32>, high: Array<F32> & (Array<F32> & Array<F32>),
  -low: Fact<Cells(inner, operation, plan, count, base, dl, dr, A, B, a)>, -facts: Fact<TreeFacts(inner, operation, plan, count, Nat.add(Model.capacity(inner), base), dl, dr, A, B, high)>) ->
  Fact<TreeFacts(1n+inner, operation, plan, count, base, dl, dr, A, B, Elementwise.tree_join(a, high))>:
  (b, inputs) = high
  Fact{({\{Fact: p => p} : Fact<({Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, (b, inputs))) == A : Array<F32>})> -> {Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, (b, inputs))) == A : Array<F32>}}(facts_left(inner, operation, plan, count, Nat.add(Model.capacity(inner), base), dl, dr, A, B, (b, inputs), facts)),
    ({\{Fact: p => p} : Fact<({Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, (b, inputs))) == B : Array<F32>})> -> {Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, (b, inputs))) == B : Array<F32>}}(facts_right(inner, operation, plan, count, Nat.add(Model.capacity(inner), base), dl, dr, A, B, (b, inputs), facts)),
      {\{Fact: p => p} : Fact<(Cells(1n+inner, operation, plan, count, base, dl, dr, A, B, ANode{a, b}))> -> Cells(1n+inner, operation, plan, count, base, dl, dr, A, B, ANode{a, b})}(node_cells(inner, operation, plan, count, base, dl, dr, A, B, a, b, low, facts_cells(inner, operation, plan, count, Nat.add(Model.capacity(inner), base), dl, dr, A, B, (b, inputs), facts)))))}

def tree_node_facts(+inner: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, rec: Array<F32> -> Array<F32> -> Array<F32> & (Array<F32> & Array<F32>), low: Array<F32> & (Array<F32> & Array<F32>),
  -facts: Fact<TreeFacts(inner, operation, plan, count, base, dl, dr, A, B, low)>,
  -next: @l: Array<F32> -> @r: Array<F32> -> @-el: {l == A : Array<F32>} -> @-er: {r == B : Array<F32>} ->
    Fact<TreeFacts(inner, operation, plan, count, Nat.add(Model.capacity(inner), base), dl, dr, A, B, rec(l, r))>) ->
  Fact<TreeFacts(1n+inner, operation, plan, count, base, dl, dr, A, B, Elementwise.tree_node(rec, low))>:
  (a, inputs) = low
  (la, ra) = inputs
  tree_join_facts(inner, operation, plan, count, base, dl, dr, A, B, a, rec(la, ra), facts_cells(inner, operation, plan, count, base, dl, dr, A, B, (a, (la, ra)), facts),
    next(la)(ra)({\{Fact: p => p} : Fact<({Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, (a, (la, ra)))) == A : Array<F32>})> -> {Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, (a, (la, ra)))) == A : Array<F32>}}(facts_left(inner, operation, plan, count, base, dl, dr, A, B, (a, (la, ra)), facts)))({\{Fact: p => p} : Fact<({Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, (a, (la, ra)))) == B : Array<F32>})> -> {Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, (a, (la, ra)))) == B : Array<F32>}}(facts_right(inner, operation, plan, count, base, dl, dr, A, B, (a, (la, ra)), facts))))

def tree_facts(+depth: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, la: Array<F32>, ra: Array<F32>,
  -el: {la == A : Array<F32>}, -er: {ra == B : Array<F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Fact<TreeFacts(depth, operation, plan, count, base, dl, dr, A, B, Elementwise.tree(depth, operation, Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), count, base, la, ra))>:
  match depth:
    case 0n:
      cell_facts(operation, plan, count, base, dl, dr, A, B, la, ra, Nat.is_lt(base, count), {==}, el, er, left_valid, right_valid)
    case 1n+inner:
      tree_node_facts(inner, operation, plan, count, base, dl, dr, A, B, l => r => Elementwise.tree(inner, operation, Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), count, Nat.add(Model.capacity(inner), base), l, r), Elementwise.tree(inner, operation, Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), count, base, la, ra),
        tree_facts(inner, operation, plan, count, base, dl, dr, A, B, la, ra, el, er, left_valid, right_valid),
        l => r => el => er => tree_facts(inner, operation, plan, count, Nat.add(Model.capacity(inner), base), dl, dr, A, B, l, r, el, er, left_valid, right_valid))

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

def fork_cells(+inner: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, -la: Array<F32>, -ra: Array<F32>,
  lp: Array<F32> & Array<F32>, -left_exact: {lp == Array.clone(F32, la) : Array<F32> & Array<F32>},
  rp: Array<F32> & Array<F32>, -right_exact: {rp == Array.clone(F32, ra) : Array<F32> & Array<F32>},
  -low: Array<F32> -> Array<F32> -> Array<F32>, -high: Array<F32> -> Array<F32> -> Array<F32>,
  -el: {la == A : Array<F32>}, -er: {ra == B : Array<F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B),
  -low_cells: @l: Array<F32> -> @r: Array<F32> -> @-el: {l == A : Array<F32>} -> @-er: {r == B : Array<F32>} ->
    Fact<Cells(inner, operation, plan, count, base, dl, dr, A, B, low(l, r))>,
  -high_cells: @l: Array<F32> -> @r: Array<F32> -> @-el: {l == A : Array<F32>} -> @-er: {r == B : Array<F32>} ->
    Fact<Cells(inner, operation, plan, count, Nat.add(Model.capacity(inner), base), dl, dr, A, B, high(l, r))>) ->
  Fact<Cells(1n+inner, operation, plan, count, base, dl, dr, A, B, Elementwise.fork(lp, rp, low, high))>:
  (l1, l2) = lp
  (r1, r2) = rp
  node_cells(inner, operation, plan, count, base, dl, dr, A, B, low(l1, r1), high(l2, r2),
    low_cells(l1)(r1)({\{Fact: p => p} : Fact<({Pair.fst(Array<F32>, Array<F32>, (l1, l2)) == A : Array<F32>})> -> {Pair.fst(Array<F32>, Array<F32>, (l1, l2)) == A : Array<F32>}}(clone_first(dl, la, A, (l1, l2), left_exact, el, left_valid)))(
      {\{Fact: p => p} : Fact<({Pair.fst(Array<F32>, Array<F32>, (r1, r2)) == B : Array<F32>})> -> {Pair.fst(Array<F32>, Array<F32>, (r1, r2)) == B : Array<F32>}}(clone_first(dr, ra, B, (r1, r2), right_exact, er, right_valid))),
    high_cells(l2)(r2)({\{Fact: p => p} : Fact<({Pair.snd(Array<F32>, Array<F32>, (l1, l2)) == A : Array<F32>})> -> {Pair.snd(Array<F32>, Array<F32>, (l1, l2)) == A : Array<F32>}}(clone_second(dl, la, A, (l1, l2), left_exact, el, left_valid)))(
      {\{Fact: p => p} : Fact<({Pair.snd(Array<F32>, Array<F32>, (r1, r2)) == B : Array<F32>})> -> {Pair.snd(Array<F32>, Array<F32>, (r1, r2)) == B : Array<F32>}}(clone_second(dr, ra, B, (r1, r2), right_exact, er, right_valid))))

def build_cells(+depth: Nat, +splits: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat, +dl: Nat, +dr: Nat, -A: Array<F32>, -B: Array<F32>, la: Array<F32>, ra: Array<F32>,
  -el: {la == A : Array<F32>}, -er: {ra == B : Array<F32>},
  -left_valid: Model.Balanced(F32, dl, A), -right_valid: Model.Balanced(F32, dr, B)) ->
  Fact<Cells(depth, operation, plan, count, base, dl, dr, A, B, Elementwise.build(depth, splits, operation, plan, count, base, la, ra))>:
  match depth splits:
    case 1n+inner 1n+rest:
      fork_cells(inner, operation, plan, count, base, dl, dr, A, B, la, ra, Array.clone(F32, la), {==}, Array.clone(F32, ra), {==},
        l => r => Elementwise.build(inner, rest, operation, plan, count, base, l, r), l => r => Elementwise.build(inner, rest, operation, plan, count, Nat.add(Model.capacity(inner), base), l, r),
        el, er, left_valid, right_valid,
        l => r => el => er => build_cells(inner, rest, operation, plan, count, base, dl, dr, A, B, l, r, el, er, left_valid, right_valid),
        l => r => el => er => build_cells(inner, rest, operation, plan, count, Nat.add(Model.capacity(inner), base), dl, dr, A, B, l, r, el, er, left_valid, right_valid))
    case 0n _:
      facts_cells(0n, operation, plan, count, base, dl, dr, A, B, Elementwise.tree(0n, operation, Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), count, base, la, ra), tree_facts(0n, operation, plan, count, base, dl, dr, A, B, la, ra, el, er, left_valid, right_valid))
    case 1n+inner 0n:
      facts_cells(1n+inner, operation, plan, count, base, dl, dr, A, B, Elementwise.tree(1n+inner, operation, Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), count, base, la, ra), tree_facts(1n+inner, operation, plan, count, base, dl, dr, A, B, la, ra, el, er, left_valid, right_valid))
