import Base
import ../shape/core.bend as Shape
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ./core.bend as Core
import ./layout.bend as Layout
import ./slice.bend as Slice
import ./signed.bend as Signed
import ./gather.bend as Gather
import ./gather_LAWS.bend as Laws
import ../storage/clone.bend as Clone
import ../shape/plan_proofs.bend as Plans
import ./parallel_proofs.bend as Generic
import ./batched_matmul_PROOF.bend as BatchedProof

# The checks and the plans.

def count_length(entries: List<&2, U32>) -> {Gather.count(entries) == Laws.length(entries) : Nat}:
  match entries:
    case Nil{}:
      {==}
    case head <> rest:
      Equal.cong(Nat, Nat, x => 1n+x, Gather.count(rest), Laws.length(rest), count_length(rest))

def below_inside(entries: List<&2, U32>, +bound: Nat) -> {Gather.below(entries, bound) == Laws.inside(entries, bound) : Bool}:
  match entries:
    case Nil{}:
      {==}
    case +head <> rest:
      Equal.cong(Bool, Bool, x => Bool.and(Nat.is_lt(U32.to_nat(head), bound), x), Gather.below(rest, bound), Laws.inside(rest, bound),
        below_inside(rest, bound))

def equal_same(left: List<&2, U32>, right: List<&2, U32>) -> {Gather.equal(left, right) == Laws.equal(left, right) : Bool}:
  match left right:
    case Nil{} Nil{}:
      {==}
    case +x <> xs +y <> ys:
      Equal.cong(Bool, Bool, b => Bool.and(Nat.is_eq(U32.to_nat(x), U32.to_nat(y)), b), Gather.equal(xs, ys), Laws.equal(xs, ys),
        equal_same(xs, ys))
    case Nil{} y <> ys:
      {==}
    case x <> xs Nil{}:
      {==}

def matches_others(dimensions: List<&2, U32>, index: List<&2, U32>, +at: Nat) ->
  {Gather.matches(dimensions, index, at) == Laws.others(dimensions, index, at) : Bool}:
  match dimensions index at:
    case x <> xs y <> ys 0n:
      equal_same(xs, ys)
    case +x <> xs +y <> ys 1n+next:
      Equal.cong(Bool, Bool, b => Bool.and(Nat.is_eq(U32.to_nat(x), U32.to_nat(y)), b), Gather.matches(xs, ys, next), Laws.others(xs, ys, next),
        matches_others(xs, ys, next))
    case Nil{} index at:
      {==}
    case x <> xs Nil{} at:
      {==}

def viewed_same(+kind: Gather.Lookup, +given: Slice.View, +count: Nat) ->
  {Gather.viewed(kind, given, count) == Laws.viewed(kind, given, count) : Gather.Plan}:
  Slice.View{+outer, +rows, +inner} = given
  {==}

def shaped_same(+plan: Gather.Plan, -dimensions: List<&2, U32>, built: Shape.ConstructResult<dimensions>) ->
  {Gather.shaped(plan, dimensions, built) == Laws.output(dimensions, plan, built) : Result<&2, &2, Gather.Error, Gather.Planned>}:
  match built:
    case Shape.SError{error}:
      {==}
    case Shape.SBuilt{+shape, preserved}:
      {==}

# An accepted request's plan and output are what the laws name.
def accepted_taken(+kind: Gather.Lookup, +dimensions: List<&2, U32>, +at: Nat, +count: Nat) ->
  {Gather.accepted(kind, dimensions, at, count) == Laws.taken(kind, dimensions, at, count) : Result<&2, &2, Gather.Error, Gather.Planned>}:
  Equal.trans(Result<&2, &2, Gather.Error, Gather.Planned>, Gather.accepted(kind, dimensions, at, count),
    Gather.shaped(Laws.viewed(kind, Slice.sized(dimensions, at, Slice.holds_zero(dimensions)), count),
      Slice.resized(dimensions, at, U32.from_nat(count)), Shape.construct(Slice.resized(dimensions, at, U32.from_nat(count)))),
    Laws.taken(kind, dimensions, at, count),
    Equal.cong(Gather.Plan, Result<&2, &2, Gather.Error, Gather.Planned>,
      p => Gather.shaped(p, Slice.resized(dimensions, at, U32.from_nat(count)), Shape.construct(Slice.resized(dimensions, at, U32.from_nat(count)))),
      Gather.viewed(kind, Slice.sized(dimensions, at, Slice.holds_zero(dimensions)), count),
      Laws.viewed(kind, Slice.sized(dimensions, at, Slice.holds_zero(dimensions)), count),
      viewed_same(kind, Slice.sized(dimensions, at, Slice.holds_zero(dimensions)), count)),
    shaped_same(Laws.viewed(kind, Slice.sized(dimensions, at, Slice.holds_zero(dimensions)), count),
      Slice.resized(dimensions, at, U32.from_nat(count)), Shape.construct(Slice.resized(dimensions, at, U32.from_nat(count)))))

# below(entries, bound) is the laws' inside, so it rewrites to its value.
def below_is(+entries: List<&2, U32>, +bound: Nat, +value: Bool, given: {Laws.inside(entries, bound) == value : Bool}) ->
  {value == Gather.below(entries, bound) : Bool}:
  Equal.sym(Bool, Gather.below(entries, bound), value,
    Equal.trans(Bool, Gather.below(entries, bound), Laws.inside(entries, bound), value, below_inside(entries, bound), given))

def Laws.index_select_axis_refused(input, dim, entries, outside):
  %Equal.sym(Bool, Nat.is_lt(Laws.axis(input, dim), Laws.rank(input)), False{}, outside) :
    {Gather.selecting(Shape.dimensions(input), Laws.axis(input, dim), entries, _) == Fail{Gather.LayoutError{Layout.AxisOutOfBounds{}}} :
      Result<&2, &2, Gather.Error, Gather.Planned>}
  {==}

def Laws.index_select_out_of_range(input, dim, entries, in_rank, past):
  %Equal.sym(Bool, Nat.is_lt(Laws.axis(input, dim), Laws.rank(input)), True{}, in_rank) :
    {Gather.selecting(Shape.dimensions(input), Laws.axis(input, dim), entries, _) == Fail{Gather.OutOfRange{}} :
      Result<&2, &2, Gather.Error, Gather.Planned>}
  %below_is(entries, Laws.extent(input, Laws.axis(input, dim)), False{}, past) :
    {Gather.ranged(Gather.Selected{}, Shape.dimensions(input), Laws.axis(input, dim), entries, Gather.count(entries), _) == Fail{Gather.OutOfRange{}} :
      Result<&2, &2, Gather.Error, Gather.Planned>}
  {==}

def Laws.index_select_planned(input, dim, entries, in_rank, within):
  %Equal.sym(Bool, Nat.is_lt(Laws.axis(input, dim), Laws.rank(input)), True{}, in_rank) :
    {Gather.selecting(Shape.dimensions(input), Laws.axis(input, dim), entries, _) ==
      Laws.taken(Gather.Selected{}, Shape.dimensions(input), Laws.axis(input, dim), Laws.length(entries)) : Result<&2, &2, Gather.Error, Gather.Planned>}
  %below_is(entries, Laws.extent(input, Laws.axis(input, dim)), True{}, within) :
    {Gather.ranged(Gather.Selected{}, Shape.dimensions(input), Laws.axis(input, dim), entries, Gather.count(entries), _) ==
      Laws.taken(Gather.Selected{}, Shape.dimensions(input), Laws.axis(input, dim), Laws.length(entries)) : Result<&2, &2, Gather.Error, Gather.Planned>}
  %Equal.sym(Nat, Gather.count(entries), Laws.length(entries), count_length(entries)) :
    {Gather.accepted(Gather.Selected{}, Shape.dimensions(input), Laws.axis(input, dim), _) ==
      Laws.taken(Gather.Selected{}, Shape.dimensions(input), Laws.axis(input, dim), Laws.length(entries)) : Result<&2, &2, Gather.Error, Gather.Planned>}
  accepted_taken(Gather.Selected{}, Shape.dimensions(input), Laws.axis(input, dim), Laws.length(entries))

# The entry count rewrites to its value.
def counted_is(+entries: List<&2, U32>, +index: Shape.Shape, +value: Bool,
  given: {Nat.is_eq(Laws.length(entries), Shape.elements(index)) == value : Bool}) ->
  {value == Nat.is_eq(Gather.count(entries), Shape.elements(index)) : Bool}:
  Equal.sym(Bool, Nat.is_eq(Gather.count(entries), Shape.elements(index)), value,
    Equal.trans(Bool, Nat.is_eq(Gather.count(entries), Shape.elements(index)), Nat.is_eq(Laws.length(entries), Shape.elements(index)),
      value,
      Equal.cong(Nat, Bool, n => Nat.is_eq(n, Shape.elements(index)), Gather.count(entries), Laws.length(entries), count_length(entries)),
      given))

def matched_is(+dimensions: List<&2, U32>, +index: List<&2, U32>, +at: Nat, +value: Bool,
  given: {Laws.others(dimensions, index, at) == value : Bool}) -> {value == Gather.matches(dimensions, index, at) : Bool}:
  Equal.sym(Bool, Gather.matches(dimensions, index, at), value,
    Equal.trans(Bool, Gather.matches(dimensions, index, at), Laws.others(dimensions, index, at), value, matches_others(dimensions, index, at), given))

def Laws.gather_count_refused(input, dim, index, entries, miscounted):
  %counted_is(entries, index, False{}, miscounted) :
    {Gather.gather_counted(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Fail{Gather.TensorError{Core.ValueCountMismatch{}}} : Result<&2, &2, Gather.Error, Gather.Planned>}
  {==}

def Laws.gather_axis_refused(input, dim, index, entries, counted, outside):
  %counted_is(entries, index, True{}, counted) :
    {Gather.gather_counted(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Fail{Gather.LayoutError{Layout.AxisOutOfBounds{}}} : Result<&2, &2, Gather.Error, Gather.Planned>}
  %Equal.sym(Bool, Nat.is_lt(Laws.axis(input, dim), Laws.rank(input)), False{}, outside) :
    {Gather.gather_axis(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Fail{Gather.LayoutError{Layout.AxisOutOfBounds{}}} : Result<&2, &2, Gather.Error, Gather.Planned>}
  {==}

def Laws.gather_rank_refused(input, dim, index, entries, counted, in_rank, ranked):
  %counted_is(entries, index, True{}, counted) :
    {Gather.gather_counted(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Fail{Gather.IndexRankMismatch{}} : Result<&2, &2, Gather.Error, Gather.Planned>}
  %Equal.sym(Bool, Nat.is_lt(Laws.axis(input, dim), Laws.rank(input)), True{}, in_rank) :
    {Gather.gather_axis(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Fail{Gather.IndexRankMismatch{}} : Result<&2, &2, Gather.Error, Gather.Planned>}
  %Equal.sym(Bool, Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(index)), Laws.rank(input)), False{}, ranked) :
    {Gather.gather_ranked(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Fail{Gather.IndexRankMismatch{}} : Result<&2, &2, Gather.Error, Gather.Planned>}
  {==}

def Laws.gather_shape_refused(input, dim, index, entries, counted, in_rank, ranked, unmatched):
  %counted_is(entries, index, True{}, counted) :
    {Gather.gather_counted(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Fail{Gather.IndexShapeMismatch{}} : Result<&2, &2, Gather.Error, Gather.Planned>}
  %Equal.sym(Bool, Nat.is_lt(Laws.axis(input, dim), Laws.rank(input)), True{}, in_rank) :
    {Gather.gather_axis(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Fail{Gather.IndexShapeMismatch{}} : Result<&2, &2, Gather.Error, Gather.Planned>}
  %Equal.sym(Bool, Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(index)), Laws.rank(input)), True{}, ranked) :
    {Gather.gather_ranked(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Fail{Gather.IndexShapeMismatch{}} : Result<&2, &2, Gather.Error, Gather.Planned>}
  %matched_is(Shape.dimensions(input), Shape.dimensions(index), Laws.axis(input, dim), False{}, unmatched) :
    {Gather.gather_shaped(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Fail{Gather.IndexShapeMismatch{}} : Result<&2, &2, Gather.Error, Gather.Planned>}
  {==}

def Laws.gather_out_of_range(input, dim, index, entries, counted, in_rank, ranked, matched, past):
  %counted_is(entries, index, True{}, counted) :
    {Gather.gather_counted(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Fail{Gather.OutOfRange{}} : Result<&2, &2, Gather.Error, Gather.Planned>}
  %Equal.sym(Bool, Nat.is_lt(Laws.axis(input, dim), Laws.rank(input)), True{}, in_rank) :
    {Gather.gather_axis(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Fail{Gather.OutOfRange{}} : Result<&2, &2, Gather.Error, Gather.Planned>}
  %Equal.sym(Bool, Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(index)), Laws.rank(input)), True{}, ranked) :
    {Gather.gather_ranked(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Fail{Gather.OutOfRange{}} : Result<&2, &2, Gather.Error, Gather.Planned>}
  %matched_is(Shape.dimensions(input), Shape.dimensions(index), Laws.axis(input, dim), True{}, matched) :
    {Gather.gather_shaped(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Fail{Gather.OutOfRange{}} : Result<&2, &2, Gather.Error, Gather.Planned>}
  %below_is(entries, Laws.extent(input, Laws.axis(input, dim)), False{}, past) :
    {Gather.ranged(Gather.Gathered{}, Shape.dimensions(input), Laws.axis(input, dim), entries,
      Slice.axis_extent(Shape.dimensions(index), Laws.axis(input, dim)), _) == Fail{Gather.OutOfRange{}} : Result<&2, &2, Gather.Error, Gather.Planned>}
  {==}

def Laws.gather_planned(input, dim, index, entries, counted, in_rank, ranked, matched, within):
  %counted_is(entries, index, True{}, counted) :
    {Gather.gather_counted(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Laws.taken(Gather.Gathered{}, Shape.dimensions(input), Laws.axis(input, dim), Slice.axis_extent(Shape.dimensions(index), Laws.axis(input, dim))) :
      Result<&2, &2, Gather.Error, Gather.Planned>}
  %Equal.sym(Bool, Nat.is_lt(Laws.axis(input, dim), Laws.rank(input)), True{}, in_rank) :
    {Gather.gather_axis(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Laws.taken(Gather.Gathered{}, Shape.dimensions(input), Laws.axis(input, dim), Slice.axis_extent(Shape.dimensions(index), Laws.axis(input, dim))) :
      Result<&2, &2, Gather.Error, Gather.Planned>}
  %Equal.sym(Bool, Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(index)), Laws.rank(input)), True{}, ranked) :
    {Gather.gather_ranked(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Laws.taken(Gather.Gathered{}, Shape.dimensions(input), Laws.axis(input, dim), Slice.axis_extent(Shape.dimensions(index), Laws.axis(input, dim))) :
      Result<&2, &2, Gather.Error, Gather.Planned>}
  %matched_is(Shape.dimensions(input), Shape.dimensions(index), Laws.axis(input, dim), True{}, matched) :
    {Gather.gather_shaped(Shape.dimensions(input), Laws.axis(input, dim), Shape.dimensions(index), entries, _) ==
      Laws.taken(Gather.Gathered{}, Shape.dimensions(input), Laws.axis(input, dim), Slice.axis_extent(Shape.dimensions(index), Laws.axis(input, dim))) :
      Result<&2, &2, Gather.Error, Gather.Planned>}
  %below_is(entries, Laws.extent(input, Laws.axis(input, dim)), True{}, within) :
    {Gather.ranged(Gather.Gathered{}, Shape.dimensions(input), Laws.axis(input, dim), entries,
      Slice.axis_extent(Shape.dimensions(index), Laws.axis(input, dim)), _) ==
      Laws.taken(Gather.Gathered{}, Shape.dimensions(input), Laws.axis(input, dim), Slice.axis_extent(Shape.dimensions(index), Laws.axis(input, dim))) :
      Result<&2, &2, Gather.Error, Gather.Planned>}
  accepted_taken(Gather.Gathered{}, Shape.dimensions(input), Laws.axis(input, dim), Slice.axis_extent(Shape.dimensions(index), Laws.axis(input, dim)))

def Laws.index_select_refused(input, dim, entries, error, refused):
  Core.Packed{+shape, tensor} = input
  %Equal.sym(Result<&2, &2, Gather.Error, Gather.Planned>, Gather.select_plan(shape, dim, entries), Fail{error}, refused) :
    {Gather.copied(shape, tensor, entries, _) == (Core.Packed{shape, tensor}, Fail{error}) : Core.Dynamic & Result<&2, &1, Gather.Error, Core.Dynamic>}
  {==}

def Laws.gather_refused(input, dim, index, entries, error, refused):
  Core.Packed{+shape, tensor} = input
  %Equal.sym(Result<&2, &2, Gather.Error, Gather.Planned>, Gather.gather_plan(shape, dim, Gather.Index{index, entries}), Fail{error}, refused) :
    {Gather.copied(shape, tensor, entries, _) == (Core.Packed{shape, tensor}, Fail{error}) : Core.Dynamic & Result<&2, &1, Gather.Error, Core.Dynamic>}
  {==}

# The cells of the forward build.

def balanced_as(-T: Data, -d: Nat, -a: Array<T>, -b: Array<T>, -same: {a == b : Array<T>}, -valid: Model.Balanced(T, d, b)) ->
  Generic.Fact<Model.Balanced(T, d, a)>:
  Generic.Fact{
    %Equal.sym(Array<T>, a, b, same) : Model.Balanced(T, d, _)
    valid}

def take_snd(index: Array<U32>, pair: Array<F32> & F32) ->
  {Pair.snd(F32, Array<F32> & Array<U32>, Gather.take(index, pair)) == (Refinement.first(F32, pair), index) : Array<F32> & Array<U32>}:
  (a, x) = pair
  {==}

def take_fst(index: Array<U32>, pair: Array<F32> & F32) ->
  {Pair.fst(F32, Array<F32> & Array<U32>, Gather.take(index, pair)) == Refinement.second(F32, pair) : F32}:
  (a, x) = pair
  {==}

def both(-a: Array<F32>, -b: Array<U32>, -A: Array<F32>, -B: Array<U32>, ea: {a == A : Array<F32>}, eb: {b == B : Array<U32>}) ->
  {(a, b) == (A, B) : Array<F32> & Array<U32>}:
  Equal.trans(Array<F32> & Array<U32>, (a, b), (A, b), (A, B),
    Equal.cong(Array<F32>, Array<F32> & Array<U32>, x => (x, b), a, A, ea),
    Equal.cong(Array<U32>, Array<F32> & Array<U32>, y => (A, y), b, B, eb))

# A forward read hands both arrays back ...
def looked_keeps(+plan: Gather.Plan, +at: Nat, +da: Nat, -A: Array<F32>, -I: Array<U32>, pair: Array<U32> & U32,
  -kept: {Refinement.first(U32, pair) == I : Array<U32>}, -valid: Model.Balanced(F32, da, A)) ->
  Generic.Fact<{Pair.snd(F32, Array<F32> & Array<U32>, Gather.looked(plan, at, A, pair)) == (A, I) : Array<F32> & Array<U32>}>:
  (index, +j) = pair
  Generic.Fact{Equal.trans(Array<F32> & Array<U32>, Pair.snd(F32, Array<F32> & Array<U32>, Gather.take(index, Array.get(F32, A, U32.from_nat(Gather.source(plan, at, U32.to_nat(j)))))),
    (Refinement.first(F32, Array.get(F32, A, U32.from_nat(Gather.source(plan, at, U32.to_nat(j))))), index), (A, I),
    take_snd(index, Array.get(F32, A, U32.from_nat(Gather.source(plan, at, U32.to_nat(j))))),
    both(Refinement.first(F32, Array.get(F32, A, U32.from_nat(Gather.source(plan, at, U32.to_nat(j))))), index, A, I,
      Refinement.law_get_retains(F32, da, A, U32.from_nat(Gather.source(plan, at, U32.to_nat(j))), valid), kept))}

def cell_keeps(+plan: Gather.Plan, +at: Nat, +da: Nat, -A: Array<F32>, +di: Nat, -I: Array<U32>, inside: Bool,
  -valid: Model.Balanced(F32, da, A), -index_valid: Model.Balanced(U32, di, I)) ->
  Generic.Fact<{Pair.snd(F32, Array<F32> & Array<U32>, Gather.cell(plan, at, (A, I), inside)) == (A, I) : Array<F32> & Array<U32>}>:
  match inside:
    case False{}:
      Generic.Fact{{==}}
    case True{}:
      Generic.Fact{{\{Generic.Fact: q => q} :
        Generic.Fact<{Pair.snd(F32, Array<F32> & Array<U32>, Gather.looked(plan, at, A, Array.get(U32, I, U32.from_nat(Gather.entry(plan, at))))) == (A, I) : Array<F32> & Array<U32>}> ->
        {Pair.snd(F32, Array<F32> & Array<U32>, Gather.looked(plan, at, A, Array.get(U32, I, U32.from_nat(Gather.entry(plan, at))))) == (A, I) : Array<F32> & Array<U32>}}(
        looked_keeps(plan, at, da, A, I, Array.get(U32, I, U32.from_nat(Gather.entry(plan, at))),
          Refinement.law_get_retains(U32, di, I, U32.from_nat(Gather.entry(plan, at)), index_valid), valid))}

# ... and gives the input word at the entry it read.
def looked_value(+plan: Gather.Plan, +at: Nat, +da: Nat, -A: Array<F32>, pair: Array<U32> & U32, +j: U32,
  -seen: {Refinement.second(U32, pair) == j : U32}, -valid: Model.Balanced(F32, da, A)) ->
  Generic.Fact<{Some{Pair.fst(F32, Array<F32> & Array<U32>, Gather.looked(plan, at, A, pair))} ==
    Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(Gather.source(plan, at, U32.to_nat(j))))) : Maybe<&2, F32>}>:
  (index, +found) = pair
  Generic.Fact{
    %seen : {Some{Pair.fst(F32, Array<F32> & Array<U32>, Gather.take(index, Array.get(F32, A, U32.from_nat(Gather.source(plan, at, U32.to_nat(found))))))} ==
      Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(Gather.source(plan, at, U32.to_nat(_))))) : Maybe<&2, F32>}
    Equal.trans(Maybe<&2, F32>, Some{Pair.fst(F32, Array<F32> & Array<U32>, Gather.take(index, Array.get(F32, A, U32.from_nat(Gather.source(plan, at, U32.to_nat(found))))))},
      Some{Refinement.second(F32, Array.get(F32, A, U32.from_nat(Gather.source(plan, at, U32.to_nat(found)))))},
      Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(Gather.source(plan, at, U32.to_nat(found))))),
      Equal.cong(F32, Maybe<&2, F32>, x => Some{x},
        Pair.fst(F32, Array<F32> & Array<U32>, Gather.take(index, Array.get(F32, A, U32.from_nat(Gather.source(plan, at, U32.to_nat(found)))))),
        Refinement.second(F32, Array.get(F32, A, U32.from_nat(Gather.source(plan, at, U32.to_nat(found))))),
        take_fst(index, Array.get(F32, A, U32.from_nat(Gather.source(plan, at, U32.to_nat(found)))))),
      Refinement.law_get_refines(F32, da, A, U32.from_nat(Gather.source(plan, at, U32.to_nat(found))), valid))}

# The entry read is the one the index array holds there.
def read_entry(+di: Nat, -I: Array<U32>, +e: Nat, +j: U32,
  -read: {Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(e))) == Some{j} : Maybe<&2, U32>},
  -index_valid: Model.Balanced(U32, di, I)) ->
  Generic.Fact<{Refinement.second(U32, Array.get(U32, I, U32.from_nat(e))) == j : U32}>:
  Generic.Fact{Plans.some_injective(U32, Refinement.second(U32, Array.get(U32, I, U32.from_nat(e))), j,
    Equal.trans(Maybe<&2, U32>, Some{Refinement.second(U32, Array.get(U32, I, U32.from_nat(e)))},
      Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(e))), Some{j},
      Refinement.law_get_refines(U32, di, I, U32.from_nat(e), index_valid), read))}

def SplitFacts(-A: Array<F32>, -I: Array<U32>) -> Type:
  {Pair.fst(Array<F32> & Array<U32>, Array<F32> & Array<U32>, Gather.split((A, I))) == (A, I) : Array<F32> & Array<U32>} & {Pair.snd(Array<F32> & Array<U32>, Array<F32> & Array<U32>, Gather.split((A, I))) == (A, I) : Array<F32> & Array<U32>}

def halves_facts(+da: Nat, -A: Array<F32>, +di: Nat, -I: Array<U32>, lp: Array<F32> & Array<F32>, rp: Array<U32> & Array<U32>,
  -le: {lp == Array.clone(F32, A) : Array<F32> & Array<F32>}, -re: {rp == Array.clone(U32, I) : Array<U32> & Array<U32>},
  -valid: Model.Balanced(F32, da, A), -index_valid: Model.Balanced(U32, di, I)) ->
  Generic.Fact<({Pair.fst(Array<F32> & Array<U32>, Array<F32> & Array<U32>, Gather.halves(lp, rp)) == (A, I) : Array<F32> & Array<U32>} & {Pair.snd(Array<F32> & Array<U32>, Array<F32> & Array<U32>, Gather.halves(lp, rp)) == (A, I) : Array<F32> & Array<U32>})>:
  (l1, l2) = lp
  (r1, r2) = rp
  Generic.Fact{(
    both(l1, r1, A, I,
      Equal.trans(Array<F32>, l1, Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, A)), A,
        Equal.cong(Array<F32> & Array<F32>, Array<F32>, p => Pair.fst(Array<F32>, Array<F32>, p), (l1, l2), Array.clone(F32, A), le),
        Clone.first_equal(F32, da, A, Clone.certify(F32, da, A, valid))),
      Equal.trans(Array<U32>, r1, Pair.fst(Array<U32>, Array<U32>, Array.clone(U32, I)), I,
        Equal.cong(Array<U32> & Array<U32>, Array<U32>, p => Pair.fst(Array<U32>, Array<U32>, p), (r1, r2), Array.clone(U32, I), re),
        Clone.first_equal(U32, di, I, Clone.certify(U32, di, I, index_valid)))),
    both(l2, r2, A, I,
      Equal.trans(Array<F32>, l2, Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, A)), A,
        Equal.cong(Array<F32> & Array<F32>, Array<F32>, p => Pair.snd(Array<F32>, Array<F32>, p), (l1, l2), Array.clone(F32, A), le),
        Clone.second_equal(F32, da, A, Clone.certify(F32, da, A, valid))),
      Equal.trans(Array<U32>, r2, Pair.snd(Array<U32>, Array<U32>, Array.clone(U32, I)), I,
        Equal.cong(Array<U32> & Array<U32>, Array<U32>, p => Pair.snd(Array<U32>, Array<U32>, p), (r1, r2), Array.clone(U32, I), re),
        Clone.second_equal(U32, di, I, Clone.certify(U32, di, I, index_valid)))))}

def split_facts(+da: Nat, -A: Array<F32>, +di: Nat, -I: Array<U32>, -valid: Model.Balanced(F32, da, A), -index_valid: Model.Balanced(U32, di, I)) ->
  Generic.Fact<SplitFacts(A, I)>:
  Generic.Fact{{\{Generic.Fact: q => q} : Generic.Fact<SplitFacts(A, I)> -> SplitFacts(A, I)}(
    halves_facts(da, A, di, I, Array.clone(F32, A), Array.clone(U32, I), {==}, {==}, valid, index_valid))}

# Every slot of the forward build holds what its cell computes from (A, I).
# The Fact holds the cells eta-expanded, as slice_PROOF.bend's build_cells.
def build_cells(+depth: Nat, +splits: Nat, +plan: Gather.Plan, +cells: Nat, +base: Nat, +da: Nat, -A: Array<F32>, a: Array<F32>,
  -same: {a == A : Array<F32>}, -valid: Model.Balanced(F32, da, A), +di: Nat, -I: Array<U32>, index: Array<U32>,
  -same_index: {index == I : Array<U32>}, -index_valid: Model.Balanced(U32, di, I)) ->
  Generic.Fact<Generic.Cells(~Gather.Plan, ~Array<F32>, ~Array<U32>, ~Gather.cell, plan, (A, I), depth, cells, base, Gather.build(depth, splits, plan, cells, base, a, index))>:
  Generic.Fact{path =>
    {\{Generic.Fact: q => q} : Generic.Fact<Generic.Cells(~Gather.Plan, ~Array<F32>, ~Array<U32>, ~Gather.cell, plan, (A, I), depth, cells, base, Gather.build(depth, splits, plan, cells, base, a, index))> -> Generic.Cells(~Gather.Plan, ~Array<F32>, ~Array<U32>, ~Gather.cell, plan, (A, I), depth, cells, base, Gather.build(depth, splits, plan, cells, base, a, index))}(
      Generic.build_cells(~Gather.Plan, ~Gather.Plan, ~Array<F32>, ~Array<U32>, ~Gather.same, ~Gather.split, ~Gather.cell,
        plan, (A, I), depth, splits, cells, base, (a, index), both(a, index, A, I, same, same_index),
        at => inside => {\{Generic.Fact: q => q} :
          Generic.Fact<{Pair.snd(F32, Array<F32> & Array<U32>, Gather.cell(plan, at, (A, I), inside)) == (A, I) : Array<F32> & Array<U32>}> ->
          {Pair.snd(F32, Array<F32> & Array<U32>, Gather.cell(plan, at, (A, I), inside)) == (A, I) : Array<F32> & Array<U32>}}(
          cell_keeps(plan, at, da, A, di, I, inside, valid, index_valid)),
        Pair.fst({Pair.fst(Array<F32> & Array<U32>, Array<F32> & Array<U32>, Gather.split((A, I))) == (A, I) : Array<F32> & Array<U32>}, {Pair.snd(Array<F32> & Array<U32>, Array<F32> & Array<U32>, Gather.split((A, I))) == (A, I) : Array<F32> & Array<U32>},
          {\{Generic.Fact: q => q} : Generic.Fact<SplitFacts(A, I)> -> SplitFacts(A, I)}(split_facts(da, A, di, I, valid, index_valid))),
        Pair.snd({Pair.fst(Array<F32> & Array<U32>, Array<F32> & Array<U32>, Gather.split((A, I))) == (A, I) : Array<F32> & Array<U32>}, {Pair.snd(Array<F32> & Array<U32>, Array<F32> & Array<U32>, Gather.split((A, I))) == (A, I) : Array<F32> & Array<U32>},
          {\{Generic.Fact: q => q} : Generic.Fact<SplitFacts(A, I)> -> SplitFacts(A, I)}(split_facts(da, A, di, I, valid, index_valid)))))(path)}

# The arithmetic of the positions: output cell (o, k, r) reads the entry
# the laws name and the input cell (o, j, r).
def entry_same(+kind: Gather.Lookup, +count: Nat, +inner: Nat, +o: Nat, +k: Nat, +r: Nat) ->
  {Gather.entry_of(kind, count, inner, o, k, r) == Laws.entry_at(kind, count, inner, o, k, r) : Nat}:
  match kind:
    case Gather.Selected{}:
      {==}
    case Gather.Gathered{}:
      {==}

def entry_at(+kind: Gather.Lookup, +outer: Nat, +rows: Nat, +inner: Nat, +count: Nat, +o: Nat, +k: Nat, +r: Nat,
  +k_below: {Nat.is_lt(k, count) == True{} : Bool}, +r_below: {Nat.is_lt(r, inner) == True{} : Bool}) ->
  {Gather.entry(Gather.Plan{kind, outer, rows, inner, count}, Laws.position(count, inner, o, k, r)) == Laws.entry_at(kind, count, inner, o, k, r) : Nat}:
  +row = Nat.add(Nat.mul(o, count), k)
  +pos = Laws.position(count, inner, o, k, r)
  %Equal.sym(Nat, Nat.div(pos, inner), row, BatchedProof.div_exact(row, inner, r, r_below)) :
    {Gather.entry_of(kind, count, inner, Nat.div(_, count), Nat.mod(_, count), Nat.mod(pos, inner)) == Laws.entry_at(kind, count, inner, o, k, r) : Nat}
  %Equal.sym(Nat, Nat.mod(pos, inner), r, BatchedProof.mod_exact(row, inner, r, r_below)) :
    {Gather.entry_of(kind, count, inner, Nat.div(row, count), Nat.mod(row, count), _) == Laws.entry_at(kind, count, inner, o, k, r) : Nat}
  %Equal.sym(Nat, Nat.div(row, count), o, BatchedProof.div_exact(o, count, k, k_below)) :
    {Gather.entry_of(kind, count, inner, _, Nat.mod(row, count), r) == Laws.entry_at(kind, count, inner, o, k, r) : Nat}
  %Equal.sym(Nat, Nat.mod(row, count), k, BatchedProof.mod_exact(o, count, k, k_below)) :
    {Gather.entry_of(kind, count, inner, o, _, r) == Laws.entry_at(kind, count, inner, o, k, r) : Nat}
  entry_same(kind, count, inner, o, k, r)

def source_at(+kind: Gather.Lookup, +outer: Nat, +rows: Nat, +inner: Nat, +count: Nat, +o: Nat, +k: Nat, +r: Nat, +j: Nat,
  +k_below: {Nat.is_lt(k, count) == True{} : Bool}, +r_below: {Nat.is_lt(r, inner) == True{} : Bool}) ->
  {Gather.source(Gather.Plan{kind, outer, rows, inner, count}, Laws.position(count, inner, o, k, r), j) == Laws.position(rows, inner, o, j, r) : Nat}:
  +row = Nat.add(Nat.mul(o, count), k)
  +pos = Laws.position(count, inner, o, k, r)
  %Equal.sym(Nat, Nat.div(pos, inner), row, BatchedProof.div_exact(row, inner, r, r_below)) :
    {Gather.position(rows, inner, Nat.div(_, count), j, Nat.mod(pos, inner)) == Laws.position(rows, inner, o, j, r) : Nat}
  %Equal.sym(Nat, Nat.mod(pos, inner), r, BatchedProof.mod_exact(row, inner, r, r_below)) :
    {Gather.position(rows, inner, Nat.div(row, count), j, _) == Laws.position(rows, inner, o, j, r) : Nat}
  %Equal.sym(Nat, Nat.div(row, count), o, BatchedProof.div_exact(o, count, k, k_below)) :
    {Gather.position(rows, inner, _, j, r) == Laws.position(rows, inner, o, j, r) : Nat}
  {==}

# The entry the cell reads, as the laws name it, rewritten to the code's.
def entry_read(+depth: Nat, +path: Model.Path(depth), +base: Nat, +kind: Gather.Lookup, +outer: Nat, +rows: Nat, +inner: Nat, +count: Nat,
  +o: Nat, +k: Nat, +r: Nat, +k_below: {Nat.is_lt(k, count) == True{} : Bool}, +r_below: {Nat.is_lt(r, inner) == True{} : Bool},
  +at: {Nat.add(Model.offset(depth, path), base) == Laws.position(count, inner, o, k, r) : Nat}, +di: Nat, -I: Array<U32>, +j: U32,
  -read: {Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(Laws.entry_at(kind, count, inner, o, k, r)))) == Some{j} : Maybe<&2, U32>}) ->
  Generic.Fact<{Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(Gather.entry(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base))))) == Some{j} : Maybe<&2, U32>}>:
  Generic.Fact{Equal.trans(Maybe<&2, U32>, Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(Gather.entry(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base))))),
    Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(Laws.entry_at(kind, count, inner, o, k, r)))), Some{j},
    Equal.cong(Nat, Maybe<&2, U32>, e => Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(e))), Gather.entry(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base)), Laws.entry_at(kind, count, inner, o, k, r),
      Equal.trans(Nat, Gather.entry(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base)), Gather.entry(Gather.Plan{kind, outer, rows, inner, count}, Laws.position(count, inner, o, k, r)), Laws.entry_at(kind, count, inner, o, k, r),
        Equal.cong(Nat, Nat, x => Gather.entry(Gather.Plan{kind, outer, rows, inner, count}, x), Nat.add(Model.offset(depth, path), base), Laws.position(count, inner, o, k, r), at),
        entry_at(kind, outer, rows, inner, count, o, k, r, k_below, r_below))),
    read)}

def Laws.gather_build_cells(depth, splits, kind, outer, rows, inner, count, cells, base, da, A, a, same, valid, di, I, index, same_index,
  index_valid, path, o, k, r, k_below, r_below, at, in_count, j, read):
  Laws.Holds{
    Equal.trans(Maybe<&2, F32>,
      Model.observe(F32, depth, Gather.build(depth, splits, Gather.Plan{kind, outer, rows, inner, count}, cells, base, a, index), path),
      Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(Gather.source(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base), U32.to_nat(j))))),
      Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(Laws.position(rows, inner, o, U32.to_nat(j), r)))),
      Equal.trans(Maybe<&2, F32>,
        Model.observe(F32, depth, Gather.build(depth, splits, Gather.Plan{kind, outer, rows, inner, count}, cells, base, a, index), path),
        Some{Pair.fst(F32, Array<F32> & Array<U32>, Gather.looked(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base), A, Array.get(U32, I, U32.from_nat(Gather.entry(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base))))))},
        Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(Gather.source(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base), U32.to_nat(j))))),
        %in_count :
          {Model.observe(F32, depth, Gather.build(depth, splits, Gather.Plan{kind, outer, rows, inner, count}, cells, base, a, index), path) ==
            Some{Pair.fst(F32, Array<F32> & Array<U32>, Gather.cell(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base), (A, I), _))} : Maybe<&2, F32>}
        {\{Generic.Fact: q => q} : Generic.Fact<Generic.Cells(~Gather.Plan, ~Array<F32>, ~Array<U32>, ~Gather.cell, Gather.Plan{kind, outer, rows, inner, count}, (A, I), depth, cells, base, Gather.build(depth, splits, Gather.Plan{kind, outer, rows, inner, count}, cells, base, a, index))> -> Generic.Cells(~Gather.Plan, ~Array<F32>, ~Array<U32>, ~Gather.cell, Gather.Plan{kind, outer, rows, inner, count}, (A, I), depth, cells, base, Gather.build(depth, splits, Gather.Plan{kind, outer, rows, inner, count}, cells, base, a, index))}(
          build_cells(depth, splits, Gather.Plan{kind, outer, rows, inner, count}, cells, base, da, A, a, same, valid, di, I, index, same_index, index_valid))(path),
        {\{Generic.Fact: q => q} :
          Generic.Fact<{Some{Pair.fst(F32, Array<F32> & Array<U32>, Gather.looked(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base), A, Array.get(U32, I, U32.from_nat(Gather.entry(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base))))))} ==
            Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(Gather.source(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base), U32.to_nat(j))))) : Maybe<&2, F32>}> ->
          {Some{Pair.fst(F32, Array<F32> & Array<U32>, Gather.looked(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base), A, Array.get(U32, I, U32.from_nat(Gather.entry(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base))))))} ==
            Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(Gather.source(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base), U32.to_nat(j))))) : Maybe<&2, F32>}}(
          looked_value(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base), da, A, Array.get(U32, I, U32.from_nat(Gather.entry(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base)))), j,
            {\{Generic.Fact: q => q} : Generic.Fact<{Refinement.second(U32, Array.get(U32, I, U32.from_nat(Gather.entry(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base))))) == j : U32}> -> {Refinement.second(U32, Array.get(U32, I, U32.from_nat(Gather.entry(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base))))) == j : U32}}(
              read_entry(di, I, Gather.entry(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base)), j,
                {\{Generic.Fact: q => q} :
                  Generic.Fact<{Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(Gather.entry(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base))))) == Some{j} : Maybe<&2, U32>}> ->
                  {Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(Gather.entry(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base))))) == Some{j} : Maybe<&2, U32>}}(
                  entry_read(depth, path, base, kind, outer, rows, inner, count, o, k, r, k_below, r_below, at, di, I, j, read)),
                index_valid)),
            valid))),
      Equal.cong(Nat, Maybe<&2, F32>, x => Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(x))),
        Gather.source(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base), U32.to_nat(j)), Laws.position(rows, inner, o, U32.to_nat(j), r),
        Equal.trans(Nat, Gather.source(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base), U32.to_nat(j)), Gather.source(Gather.Plan{kind, outer, rows, inner, count}, Laws.position(count, inner, o, k, r), U32.to_nat(j)), Laws.position(rows, inner, o, U32.to_nat(j), r),
          Equal.cong(Nat, Nat, x => Gather.source(Gather.Plan{kind, outer, rows, inner, count}, x, U32.to_nat(j)), Nat.add(Model.offset(depth, path), base), Laws.position(count, inner, o, k, r), at),
          source_at(kind, outer, rows, inner, count, o, k, r, U32.to_nat(j), k_below, r_below))))}

# The cells of the backward build.

# One gradient cell of the backward: it skips, or adds its word.
type Stepped<-dg: Nat, -G: Array<F32>, -di: Nat, -I: Array<U32>, -i: Nat, -at: Nat, -slot: Nat, -acc: F32, -result: F32 & (Array<F32> & Array<U32>)> is Data:
  Skipped{-j: U32, -read: {Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(slot))) == Some{j} : Maybe<&2, U32>},
    -other: {Nat.is_eq(U32.to_nat(j), i) == False{} : Bool}, -kept: {result == (acc, (G, I)) : F32 & (Array<F32> & Array<U32>)}}
  Taken{-j: U32, -read: {Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(slot))) == Some{j} : Maybe<&2, U32>},
    -same: {Nat.is_eq(U32.to_nat(j), i) == True{} : Bool}, -x: F32,
    -word: {Model.observe(F32, dg, G, Refinement.route(dg, U32.from_nat(at))) == Some{x} : Maybe<&2, F32>},
    -kept: {result == (F32.add(acc, x), (G, I)) : F32 & (Array<F32> & Array<U32>)}}

def accumulated(acc: F32, index: Array<U32>, pair: Array<F32> & F32) ->
  {Gather.accumulate(acc, index, pair) == (F32.add(acc, Refinement.second(F32, pair)), (Refinement.first(F32, pair), index)) : F32 & (Array<F32> & Array<U32>)}:
  (g, x) = pair
  {==}

def triple(-acc: F32, -x: F32, -a: Array<F32>, -b: Array<U32>, -A: Array<F32>, -B: Array<U32>, ea: {a == A : Array<F32>}, eb: {b == B : Array<U32>}) ->
  {(F32.add(acc, x), (a, b)) == (F32.add(acc, x), (A, B)) : F32 & (Array<F32> & Array<U32>)}:
  Equal.cong(Array<F32> & Array<U32>, F32 & (Array<F32> & Array<U32>), p => (F32.add(acc, x), p), (a, b), (A, B), both(a, b, A, B, ea, eb))

def summed_is(+i: Nat, +at: Nat, +slot: Nat, +acc: F32, +dg: Nat, -G: Array<F32>, +di: Nat, -I: Array<U32>, index: Array<U32>, +j: U32,
  -kept: {index == I : Array<U32>}, -seen: {Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(slot))) == Some{j} : Maybe<&2, U32>},
  hit: Bool, -told: {Nat.is_eq(U32.to_nat(j), i) == hit : Bool}, -valid: Model.Balanced(F32, dg, G)) ->
  Generic.Fact<Stepped<dg, G, di, I, i, at, slot, acc, Gather.summed(hit, at, acc, G, index)>>:
  match hit:
    case False{}:
      Generic.Fact{Skipped{j, seen, told, Equal.cong(Array<U32>, F32 & (Array<F32> & Array<U32>), b => (acc, (G, b)), index, I, kept)}}
    case True{}:
      Generic.Fact{Taken{j, seen, told, Refinement.second(F32, Array.get(F32, G, U32.from_nat(at))),
        Equal.sym(Maybe<&2, F32>, Some{Refinement.second(F32, Array.get(F32, G, U32.from_nat(at)))},
          Model.observe(F32, dg, G, Refinement.route(dg, U32.from_nat(at))), Refinement.law_get_refines(F32, dg, G, U32.from_nat(at), valid)),
        Equal.trans(F32 & (Array<F32> & Array<U32>), Gather.accumulate(acc, index, Array.get(F32, G, U32.from_nat(at))),
          (F32.add(acc, Refinement.second(F32, Array.get(F32, G, U32.from_nat(at)))), (Refinement.first(F32, Array.get(F32, G, U32.from_nat(at))), index)),
          (F32.add(acc, Refinement.second(F32, Array.get(F32, G, U32.from_nat(at)))), (G, I)),
          accumulated(acc, index, Array.get(F32, G, U32.from_nat(at))),
          triple(acc, Refinement.second(F32, Array.get(F32, G, U32.from_nat(at))), Refinement.first(F32, Array.get(F32, G, U32.from_nat(at))), index, G, I,
            Refinement.law_get_retains(F32, dg, G, U32.from_nat(at), valid), kept))}}

def compared_is(+i: Nat, +at: Nat, +slot: Nat, +acc: F32, +dg: Nat, -G: Array<F32>, +di: Nat, -I: Array<U32>, pair: Array<U32> & U32,
  -kept: {Refinement.first(U32, pair) == I : Array<U32>},
  -seen: {Some{Refinement.second(U32, pair)} == Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(slot))) : Maybe<&2, U32>},
  -valid: Model.Balanced(F32, dg, G)) ->
  Generic.Fact<Stepped<dg, G, di, I, i, at, slot, acc, Gather.compared(i, at, acc, G, pair)>>:
  (index, +j) = pair
  summed_is(i, at, slot, acc, dg, G, di, I, index, j, kept,
    Equal.sym(Maybe<&2, U32>, Some{j}, Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(slot))), seen),
    Nat.is_eq(U32.to_nat(j), i), {==}, valid)

def stepped_is(+i: Nat, +at: Nat, +slot: Nat, +acc: F32, +dg: Nat, -G: Array<F32>, +di: Nat, -I: Array<U32>,
  -valid: Model.Balanced(F32, dg, G), -index_valid: Model.Balanced(U32, di, I)) ->
  Generic.Fact<Stepped<dg, G, di, I, i, at, slot, acc, Gather.stepped(i, at, slot, acc, (G, I))>>:
  Generic.Fact{{\{Generic.Fact: q => q} :
    Generic.Fact<Stepped<dg, G, di, I, i, at, slot, acc, Gather.compared(i, at, acc, G, Array.get(U32, I, U32.from_nat(slot)))>> ->
    Stepped<dg, G, di, I, i, at, slot, acc, Gather.compared(i, at, acc, G, Array.get(U32, I, U32.from_nat(slot)))>}(
    compared_is(i, at, slot, acc, dg, G, di, I, Array.get(U32, I, U32.from_nat(slot)),
      Refinement.law_get_retains(U32, di, I, U32.from_nat(slot), index_valid),
      Refinement.law_get_refines(U32, di, I, U32.from_nat(slot), index_valid), valid))}

# The run of the backward from gradient cell k, for `left` more k: both
# arrays come back, and the sum is the one Added names.
def AddedIs(+dg: Nat, -G: Array<F32>, +di: Nat, -I: Array<U32>, +kind: Gather.Lookup, +outer: Nat, +rows: Nat, +inner: Nat, +count: Nat,
  +o: Nat, +i: Nat, +r: Nat, +left: Nat, +k: Nat, +acc: F32, -result: F32 & (Array<F32> & Array<U32>)) -> Type:
  {Pair.snd(F32, Array<F32> & Array<U32>, result) == (G, I) : Array<F32> & Array<U32>} &
    Laws.Added(dg, G, di, I, kind, count, inner, o, i, r, left, k, acc, Pair.fst(F32, Array<F32> & Array<U32>, result))

def StepIs(+dg: Nat, -G: Array<F32>, +di: Nat, -I: Array<U32>, +kind: Gather.Lookup, +outer: Nat, +rows: Nat, +inner: Nat, +count: Nat,
  +o: Nat, +i: Nat, +r: Nat, +rest: Nat, +k: Nat, +acc: F32, -result: F32 & (Array<F32> & Array<U32>)) -> Type:
  {Pair.snd(F32, Array<F32> & Array<U32>, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, rest, 1n+k, result)) == (G, I) : Array<F32> & Array<U32>} &
    Laws.AddStep<dg, G, di, I, kind, count, inner, o, i, r, rest, k, acc, Pair.fst(F32, Array<F32> & Array<U32>, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, rest, 1n+k, result))>

def read_slot(+kind: Gather.Lookup, +count: Nat, +inner: Nat, +o: Nat, +k: Nat, +r: Nat, +di: Nat, -I: Array<U32>, +j: U32,
  read: {Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(Gather.entry_of(kind, count, inner, o, k, r)))) == Some{j} : Maybe<&2, U32>}) ->
  {Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(Laws.entry_at(kind, count, inner, o, k, r)))) == Some{j} : Maybe<&2, U32>}:
  Equal.trans(Maybe<&2, U32>, Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(Laws.entry_at(kind, count, inner, o, k, r)))),
    Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(Gather.entry_of(kind, count, inner, o, k, r)))), Some{j},
    Equal.cong(Nat, Maybe<&2, U32>, e => Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(e))),
      Laws.entry_at(kind, count, inner, o, k, r), Gather.entry_of(kind, count, inner, o, k, r),
      Equal.sym(Nat, Gather.entry_of(kind, count, inner, o, k, r), Laws.entry_at(kind, count, inner, o, k, r), entry_same(kind, count, inner, o, k, r))),
    read)

def step_added(+dg: Nat, -G: Array<F32>, +di: Nat, -I: Array<U32>, +kind: Gather.Lookup, +outer: Nat, +rows: Nat, +inner: Nat, +count: Nat,
  +o: Nat, +i: Nat, +r: Nat, +rest: Nat, +k: Nat, +acc: F32,
  ih: @+next: F32 -> Generic.Fact<AddedIs(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, rest, 1n+k, next,
    Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, rest, 1n+k, (next, (G, I))))>,
  -result: F32 & (Array<F32> & Array<U32>),
  st: Stepped<dg, G, di, I, i, Gather.cell_at(Gather.Plan{kind, outer, rows, inner, count}, o, k, r), Gather.slot_at(Gather.Plan{kind, outer, rows, inner, count}, o, k, r), acc, result>) ->
  Generic.Fact<StepIs(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, rest, k, acc, result)>:
  match st:
    case Skipped{j, read, other, kept}:
      Generic.Fact{
        %Equal.sym(F32 & (Array<F32> & Array<U32>), result, (acc, (G, I)), kept) :
          StepIs(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, rest, k, acc, _)
        {\{Generic.Fact: q => q} : Generic.Fact<StepIs(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, rest, k, acc, (acc, (G, I)))> -> StepIs(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, rest, k, acc, (acc, (G, I)))}(
      {\{Generic.Fact: q => Generic.Fact{(Pair.fst(
          {Pair.snd(F32, Array<F32> & Array<U32>, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, rest, 1n+k, (acc, (G, I)))) == (G, I) : Array<F32> & Array<U32>},
          Laws.Added(dg, G, di, I, kind, count, inner, o, i, r, rest, 1n+k, acc, Pair.fst(F32, Array<F32> & Array<U32>, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, rest, 1n+k, (acc, (G, I))))), q),
        Laws.Skips{j, read_slot(kind, count, inner, o, k, r, di, I, j, read), other, Pair.snd(
          {Pair.snd(F32, Array<F32> & Array<U32>, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, rest, 1n+k, (acc, (G, I)))) == (G, I) : Array<F32> & Array<U32>},
          Laws.Added(dg, G, di, I, kind, count, inner, o, i, r, rest, 1n+k, acc, Pair.fst(F32, Array<F32> & Array<U32>, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, rest, 1n+k, (acc, (G, I))))), q)})}} :
        Generic.Fact<AddedIs(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, rest, 1n+k, acc, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, rest, 1n+k, (acc, (G, I))))> ->
        Generic.Fact<StepIs(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, rest, k, acc, (acc, (G, I)))>}(ih(acc)))}
    case Taken{j, read, same, x, word, kept}:
      Generic.Fact{
        %Equal.sym(F32 & (Array<F32> & Array<U32>), result, (F32.add(acc, x), (G, I)), kept) :
          StepIs(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, rest, k, acc, _)
        {\{Generic.Fact: q => q} : Generic.Fact<StepIs(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, rest, k, acc, (F32.add(acc, x), (G, I)))> -> StepIs(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, rest, k, acc, (F32.add(acc, x), (G, I)))}(
      {\{Generic.Fact: q => Generic.Fact{(Pair.fst(
          {Pair.snd(F32, Array<F32> & Array<U32>, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, rest, 1n+k, (F32.add(acc, x), (G, I)))) == (G, I) : Array<F32> & Array<U32>},
          Laws.Added(dg, G, di, I, kind, count, inner, o, i, r, rest, 1n+k, F32.add(acc, x),
            Pair.fst(F32, Array<F32> & Array<U32>, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, rest, 1n+k, (F32.add(acc, x), (G, I))))), q),
        Laws.Adds{j, read_slot(kind, count, inner, o, k, r, di, I, j, read), same, x, word, Pair.snd(
          {Pair.snd(F32, Array<F32> & Array<U32>, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, rest, 1n+k, (F32.add(acc, x), (G, I)))) == (G, I) : Array<F32> & Array<U32>},
          Laws.Added(dg, G, di, I, kind, count, inner, o, i, r, rest, 1n+k, F32.add(acc, x),
            Pair.fst(F32, Array<F32> & Array<U32>, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, rest, 1n+k, (F32.add(acc, x), (G, I))))), q)})}} :
        Generic.Fact<AddedIs(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, rest, 1n+k, F32.add(acc, x),
          Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, rest, 1n+k, (F32.add(acc, x), (G, I))))> ->
        Generic.Fact<StepIs(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, rest, k, acc, (F32.add(acc, x), (G, I)))>}(ih(F32.add(acc, x))))}

def added_is(+dg: Nat, -G: Array<F32>, +di: Nat, -I: Array<U32>, +kind: Gather.Lookup, +outer: Nat, +rows: Nat, +inner: Nat, +count: Nat,
  +o: Nat, +i: Nat, +r: Nat, left: Nat, +k: Nat, +acc: F32, -valid: Model.Balanced(F32, dg, G), -index_valid: Model.Balanced(U32, di, I)) ->
  Generic.Fact<AddedIs(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, left, k, acc, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, left, k, (acc, (G, I))))>:
  match left:
    case 0n:
      Generic.Fact{({==}, {==})}
    case 1n+rest:
      Generic.Fact{{\{Generic.Fact: q => q} : Generic.Fact<AddedIs(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, 1n+rest, k, acc, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, 1n+rest, k, (acc, (G, I))))> -> AddedIs(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, 1n+rest, k, acc, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, 1n+rest, k, (acc, (G, I))))}(
      step_added(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, rest, k, acc,
        next => added_is(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, rest, 1n+k, next, valid, index_valid),
        Gather.stepped(i, Gather.cell_at(Gather.Plan{kind, outer, rows, inner, count}, o, k, r), Gather.slot_at(Gather.Plan{kind, outer, rows, inner, count}, o, k, r), acc, (G, I)),
        {\{Generic.Fact: q => q} :
          Generic.Fact<Stepped<dg, G, di, I, i, Gather.cell_at(Gather.Plan{kind, outer, rows, inner, count}, o, k, r), Gather.slot_at(Gather.Plan{kind, outer, rows, inner, count}, o, k, r), acc,
            Gather.stepped(i, Gather.cell_at(Gather.Plan{kind, outer, rows, inner, count}, o, k, r), Gather.slot_at(Gather.Plan{kind, outer, rows, inner, count}, o, k, r), acc, (G, I))>> ->
          Stepped<dg, G, di, I, i, Gather.cell_at(Gather.Plan{kind, outer, rows, inner, count}, o, k, r), Gather.slot_at(Gather.Plan{kind, outer, rows, inner, count}, o, k, r), acc,
            Gather.stepped(i, Gather.cell_at(Gather.Plan{kind, outer, rows, inner, count}, o, k, r), Gather.slot_at(Gather.Plan{kind, outer, rows, inner, count}, o, k, r), acc, (G, I))>}(
          stepped_is(i, Gather.cell_at(Gather.Plan{kind, outer, rows, inner, count}, o, k, r), Gather.slot_at(Gather.Plan{kind, outer, rows, inner, count}, o, k, r), acc, dg, G, di, I, valid, index_valid))))}

# A backward cell runs the sum from +0.0 over every k and hands both arrays
# back.
def scatter_keeps(+kind: Gather.Lookup, +outer: Nat, +rows: Nat, +inner: Nat, +count: Nat, +at: Nat, +dg: Nat, -G: Array<F32>, +di: Nat,
  -I: Array<U32>, inside: Bool, -valid: Model.Balanced(F32, dg, G), -index_valid: Model.Balanced(U32, di, I)) ->
  Generic.Fact<{Pair.snd(F32, Array<F32> & Array<U32>, Gather.scatter_cell(Gather.Plan{kind, outer, rows, inner, count}, at, (G, I), inside)) == (G, I) : Array<F32> & Array<U32>}>:
  match inside:
    case False{}:
      Generic.Fact{{==}}
    case True{}:
      Generic.Fact{Pair.fst({Pair.snd(F32, Array<F32> & Array<U32>, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, Nat.div(Nat.div(at, inner), rows), Nat.mod(Nat.div(at, inner), rows), Nat.mod(at, inner), count, 0n, (0.0, (G, I)))) == (G, I) : Array<F32> & Array<U32>},
        Laws.Added(dg, G, di, I, kind, count, inner, Nat.div(Nat.div(at, inner), rows), Nat.mod(Nat.div(at, inner), rows), Nat.mod(at, inner), count, 0n, 0.0, Pair.fst(F32, Array<F32> & Array<U32>, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, Nat.div(Nat.div(at, inner), rows), Nat.mod(Nat.div(at, inner), rows), Nat.mod(at, inner), count, 0n, (0.0, (G, I))))),
        {\{Generic.Fact: q => q} : Generic.Fact<AddedIs(dg, G, di, I, kind, outer, rows, inner, count, Nat.div(Nat.div(at, inner), rows), Nat.mod(Nat.div(at, inner), rows), Nat.mod(at, inner), count, 0n, 0.0, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, Nat.div(Nat.div(at, inner), rows), Nat.mod(Nat.div(at, inner), rows), Nat.mod(at, inner), count, 0n, (0.0, (G, I))))> -> AddedIs(dg, G, di, I, kind, outer, rows, inner, count, Nat.div(Nat.div(at, inner), rows), Nat.mod(Nat.div(at, inner), rows), Nat.mod(at, inner), count, 0n, 0.0, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, Nat.div(Nat.div(at, inner), rows), Nat.mod(Nat.div(at, inner), rows), Nat.mod(at, inner), count, 0n, (0.0, (G, I))))}(
          added_is(dg, G, di, I, kind, outer, rows, inner, count, Nat.div(Nat.div(at, inner), rows), Nat.mod(Nat.div(at, inner), rows), Nat.mod(at, inner), count, 0n, 0.0, valid, index_valid)))}

def scatter_cells(+depth: Nat, +splits: Nat, +kind: Gather.Lookup, +outer: Nat, +rows: Nat, +inner: Nat, +count: Nat, +cells: Nat, +base: Nat,
  +dg: Nat, -G: Array<F32>, g: Array<F32>, -same: {g == G : Array<F32>}, -valid: Model.Balanced(F32, dg, G), +di: Nat, -I: Array<U32>,
  index: Array<U32>, -same_index: {index == I : Array<U32>}, -index_valid: Model.Balanced(U32, di, I)) ->
  Generic.Fact<Generic.Cells(~Gather.Plan, ~Array<F32>, ~Array<U32>, ~Gather.scatter_cell, Gather.Plan{kind, outer, rows, inner, count}, (G, I), depth, cells, base, Gather.scatter_build(depth, splits, Gather.Plan{kind, outer, rows, inner, count}, cells, base, g, index))>:
  Generic.Fact{path =>
    {\{Generic.Fact: q => q} : Generic.Fact<Generic.Cells(~Gather.Plan, ~Array<F32>, ~Array<U32>, ~Gather.scatter_cell, Gather.Plan{kind, outer, rows, inner, count}, (G, I), depth, cells, base, Gather.scatter_build(depth, splits, Gather.Plan{kind, outer, rows, inner, count}, cells, base, g, index))> -> Generic.Cells(~Gather.Plan, ~Array<F32>, ~Array<U32>, ~Gather.scatter_cell, Gather.Plan{kind, outer, rows, inner, count}, (G, I), depth, cells, base, Gather.scatter_build(depth, splits, Gather.Plan{kind, outer, rows, inner, count}, cells, base, g, index))}(
      Generic.build_cells(~Gather.Plan, ~Gather.Plan, ~Array<F32>, ~Array<U32>, ~Gather.same, ~Gather.split, ~Gather.scatter_cell,
        Gather.Plan{kind, outer, rows, inner, count}, (G, I), depth, splits, cells, base, (g, index), both(g, index, G, I, same, same_index),
        at => inside => {\{Generic.Fact: q => q} :
          Generic.Fact<{Pair.snd(F32, Array<F32> & Array<U32>, Gather.scatter_cell(Gather.Plan{kind, outer, rows, inner, count}, at, (G, I), inside)) == (G, I) : Array<F32> & Array<U32>}> ->
          {Pair.snd(F32, Array<F32> & Array<U32>, Gather.scatter_cell(Gather.Plan{kind, outer, rows, inner, count}, at, (G, I), inside)) == (G, I) : Array<F32> & Array<U32>}}(
          scatter_keeps(kind, outer, rows, inner, count, at, dg, G, di, I, inside, valid, index_valid)),
        Pair.fst({Pair.fst(Array<F32> & Array<U32>, Array<F32> & Array<U32>, Gather.split((G, I))) == (G, I) : Array<F32> & Array<U32>}, {Pair.snd(Array<F32> & Array<U32>, Array<F32> & Array<U32>, Gather.split((G, I))) == (G, I) : Array<F32> & Array<U32>},
          {\{Generic.Fact: q => q} : Generic.Fact<SplitFacts(G, I)> -> SplitFacts(G, I)}(split_facts(dg, G, di, I, valid, index_valid))),
        Pair.snd({Pair.fst(Array<F32> & Array<U32>, Array<F32> & Array<U32>, Gather.split((G, I))) == (G, I) : Array<F32> & Array<U32>}, {Pair.snd(Array<F32> & Array<U32>, Array<F32> & Array<U32>, Gather.split((G, I))) == (G, I) : Array<F32> & Array<U32>},
          {\{Generic.Fact: q => q} : Generic.Fact<SplitFacts(G, I)> -> SplitFacts(G, I)}(split_facts(dg, G, di, I, valid, index_valid)))))(path)}

# Input cell (o, i, r) runs the sum of (o, i, r).
def scattered_at(+kind: Gather.Lookup, +outer: Nat, +rows: Nat, +inner: Nat, +count: Nat, +o: Nat, +i: Nat, +r: Nat,
  +i_below: {Nat.is_lt(i, rows) == True{} : Bool}, +r_below: {Nat.is_lt(r, inner) == True{} : Bool}, -state: Array<F32> & Array<U32>) ->
  {Gather.scattered(Gather.Plan{kind, outer, rows, inner, count}, Laws.position(rows, inner, o, i, r), state) == Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, count, 0n, (0.0, state)) : F32 & (Array<F32> & Array<U32>)}:
  +row = Nat.add(Nat.mul(o, rows), i)
  +pos = Laws.position(rows, inner, o, i, r)
  %Equal.sym(Nat, Nat.div(pos, inner), row, BatchedProof.div_exact(row, inner, r, r_below)) :
    {Gather.added(Gather.Plan{kind, outer, rows, inner, count}, Nat.div(_, rows), Nat.mod(_, rows), Nat.mod(pos, inner), count, 0n, (0.0, state)) ==
      Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, count, 0n, (0.0, state)) : F32 & (Array<F32> & Array<U32>)}
  %Equal.sym(Nat, Nat.mod(pos, inner), r, BatchedProof.mod_exact(row, inner, r, r_below)) :
    {Gather.added(Gather.Plan{kind, outer, rows, inner, count}, Nat.div(row, rows), Nat.mod(row, rows), _, count, 0n, (0.0, state)) ==
      Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, count, 0n, (0.0, state)) : F32 & (Array<F32> & Array<U32>)}
  %Equal.sym(Nat, Nat.div(row, rows), o, BatchedProof.div_exact(o, rows, i, i_below)) :
    {Gather.added(Gather.Plan{kind, outer, rows, inner, count}, _, Nat.mod(row, rows), r, count, 0n, (0.0, state)) == Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, count, 0n, (0.0, state)) : F32 & (Array<F32> & Array<U32>)}
  %Equal.sym(Nat, Nat.mod(row, rows), i, BatchedProof.mod_exact(o, rows, i, i_below)) :
    {Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, _, r, count, 0n, (0.0, state)) == Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, count, 0n, (0.0, state)) : F32 & (Array<F32> & Array<U32>)}
  {==}

def Laws.gather_scatter_cells(depth, splits, kind, outer, rows, inner, count, cells, base, dg, G, g, same, valid, di, I, index, same_index,
  index_valid, path, o, i, r, i_below, r_below, at, in_count):
  Laws.Summed{Pair.fst(F32, Array<F32> & Array<U32>, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, count, 0n, (0.0, (G, I)))),
    Equal.trans(Maybe<&2, F32>, Model.observe(F32, depth, Gather.scatter_build(depth, splits, Gather.Plan{kind, outer, rows, inner, count}, cells, base, g, index), path),
      Some{Pair.fst(F32, Array<F32> & Array<U32>, Gather.scattered(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base), (G, I)))},
      Some{Pair.fst(F32, Array<F32> & Array<U32>, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, count, 0n, (0.0, (G, I))))},
      %in_count :
        {Model.observe(F32, depth, Gather.scatter_build(depth, splits, Gather.Plan{kind, outer, rows, inner, count}, cells, base, g, index), path) ==
          Some{Pair.fst(F32, Array<F32> & Array<U32>, Gather.scatter_cell(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base), (G, I), _))} : Maybe<&2, F32>}
      {\{Generic.Fact: q => q} : Generic.Fact<Generic.Cells(~Gather.Plan, ~Array<F32>, ~Array<U32>, ~Gather.scatter_cell, Gather.Plan{kind, outer, rows, inner, count}, (G, I), depth, cells, base, Gather.scatter_build(depth, splits, Gather.Plan{kind, outer, rows, inner, count}, cells, base, g, index))> -> Generic.Cells(~Gather.Plan, ~Array<F32>, ~Array<U32>, ~Gather.scatter_cell, Gather.Plan{kind, outer, rows, inner, count}, (G, I), depth, cells, base, Gather.scatter_build(depth, splits, Gather.Plan{kind, outer, rows, inner, count}, cells, base, g, index))}(
        scatter_cells(depth, splits, kind, outer, rows, inner, count, cells, base, dg, G, g, same, valid, di, I, index, same_index, index_valid))(path),
      Equal.cong(F32 & (Array<F32> & Array<U32>), Maybe<&2, F32>, p => Some{Pair.fst(F32, Array<F32> & Array<U32>, p)},
        Gather.scattered(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base), (G, I)), Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, count, 0n, (0.0, (G, I))),
        Equal.trans(F32 & (Array<F32> & Array<U32>), Gather.scattered(Gather.Plan{kind, outer, rows, inner, count}, Nat.add(Model.offset(depth, path), base), (G, I)), Gather.scattered(Gather.Plan{kind, outer, rows, inner, count}, Laws.position(rows, inner, o, i, r), (G, I)), Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, count, 0n, (0.0, (G, I))),
          Equal.cong(Nat, F32 & (Array<F32> & Array<U32>), x => Gather.scattered(Gather.Plan{kind, outer, rows, inner, count}, x, (G, I)), Nat.add(Model.offset(depth, path), base), Laws.position(rows, inner, o, i, r), at),
          scattered_at(kind, outer, rows, inner, count, o, i, r, i_below, r_below, (G, I))))),
    Pair.snd({Pair.snd(F32, Array<F32> & Array<U32>, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, count, 0n, (0.0, (G, I)))) == (G, I) : Array<F32> & Array<U32>},
      Laws.Added(dg, G, di, I, kind, count, inner, o, i, r, count, 0n, 0.0, Pair.fst(F32, Array<F32> & Array<U32>, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, count, 0n, (0.0, (G, I))))),
      {\{Generic.Fact: q => q} : Generic.Fact<AddedIs(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, count, 0n, 0.0, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, count, 0n, (0.0, (G, I))))> -> AddedIs(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, count, 0n, 0.0, Gather.added(Gather.Plan{kind, outer, rows, inner, count}, o, i, r, count, 0n, (0.0, (G, I))))}(
        added_is(dg, G, di, I, kind, outer, rows, inner, count, o, i, r, count, 0n, 0.0, valid, index_valid)))}

def index_same(+entries: List<&2, U32>, -dimensions: List<&2, U32>, built: Shape.ConstructResult<dimensions>) ->
  {Gather.built_index(entries, dimensions, built) == Laws.index_of(entries, dimensions, built) : Result<&2, &2, Gather.Error, Gather.Index>}:
  match built:
    case Shape.SError{error}:
      {==}
    case Shape.SBuilt{+shape, preserved}:
      {==}

def Laws.gather_index_shaped(dimensions, entries):
  index_same(entries, dimensions, Shape.construct(dimensions))
