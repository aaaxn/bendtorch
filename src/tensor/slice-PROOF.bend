import Base
import ../shape/core.bend as Shape
import ../shape/nat_proofs.bend as Nats
import ../shape/index_proofs.bend as Index
import ../storage/arith.bend as Arith
import ../storage/api.bend as Api
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ./core.bend as Core
import ./layout.bend as Layout
import ./transforms.bend as Transforms
import ./slice.bend as Slice
import ./slice-LAWS.bend as Laws
import ./contracts/build-facts.bend as Facts
import ./elementwise.bend as Elementwise
import ./elementwise-parallel-proof.bend as Parallel
import ./parallel-proof.bend as Generic
import ./batched-matmul-PROOF.bend as BatchedProof
import ../storage/array.bend as ArrayCore

# The clamp.

def min_left(+a: Nat, +b: Nat, h: {Nat.is_le(a, b) == True{} : Bool}) -> {Nat.min(a, b) == a : Nat}:
  match a b:
    case 0n _:
      {==}
    case 1n+p 0n:
      Empty.absurd({Nat.min(1n+p, 0n) == 1n+p : Nat}, Arith.clash(h))
    case 1n+p 1n+q:
      Equal.cong(Nat, Nat, x => 1n+x, Nat.min(p, q), p, min_left(p, q, h))

def min_right(+a: Nat, +b: Nat, h: {Nat.is_le(b, a) == True{} : Bool}) -> {Nat.min(a, b) == b : Nat}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      Empty.absurd({Nat.min(0n, 1n+q) == 1n+q : Nat}, Arith.clash(h))
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      Equal.cong(Nat, Nat, x => 1n+x, Nat.min(p, q), q, min_right(p, q, h))

def min_le_right(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), b) == True{} : Bool}:
  match a b:
    case 0n _:
      Arith.le_zero(b)
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      min_le_right(p, q)

# min(a, b) <= min(c, b) for a <= c.
def min_mono(+a: Nat, +c: Nat, +b: Nat, h: {Nat.is_le(a, c) == True{} : Bool}) -> {Nat.is_le(Nat.min(a, b), Nat.min(c, b)) == True{} : Bool}:
  match a c b:
    case 0n _ _:
      Arith.le_zero(Nat.min(c, b))
    case 1n+p 0n _:
      Empty.absurd({Nat.is_le(Nat.min(1n+p, b), Nat.min(0n, b)) == True{} : Bool}, Arith.clash(h))
    case 1n+p 1n+q 0n:
      {==}
    case 1n+p 1n+q 1n+r:
      min_mono(p, q, r, h)

def max_right(+a: Nat, +b: Nat, h: {Nat.is_le(a, b) == True{} : Bool}) -> {Nat.max(a, b) == b : Nat}:
  match a b:
    case 0n _:
      {==}
    case 1n+p 0n:
      Empty.absurd({Nat.max(1n+p, 0n) == 0n : Nat}, Arith.clash(h))
    case 1n+p 1n+q:
      Equal.cong(Nat, Nat, x => 1n+x, Nat.max(p, q), q, max_right(p, q, h))

def max_left(+a: Nat, +b: Nat, h: {Nat.is_le(b, a) == True{} : Bool}) -> {Nat.max(a, b) == a : Nat}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      Empty.absurd({Nat.max(0n, 1n+q) == 0n : Nat}, Arith.clash(h))
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      Equal.cong(Nat, Nat, x => 1n+x, Nat.max(p, q), p, max_left(p, q, h))

def Laws.slice_start_clamped(extent, start, stop, past):
  min_right(start, extent, past)

def Laws.slice_start_kept(extent, start, stop, inside):
  min_left(start, extent, inside)

def Laws.slice_stop_clamped(extent, start, stop, past):
  %Equal.sym(Nat, Nat.min(stop, extent), extent, min_right(stop, extent, past)) :
    {Nat.max(Nat.min(start, extent), _) == extent : Nat}
  max_right(Nat.min(start, extent), extent, min_le_right(start, extent))

def Laws.slice_stop_kept(extent, start, stop, after, inside):
  %Equal.sym(Nat, Nat.min(start, extent), start, min_left(start, extent, Facts.le_trans(start, stop, extent, after, inside))) :
    {Nat.max(_, Nat.min(stop, extent)) == stop : Nat}
  %Equal.sym(Nat, Nat.min(stop, extent), stop, min_left(stop, extent, inside)) :
    {Nat.max(start, _) == stop : Nat}
  max_right(start, stop, after)

def Laws.slice_stop_empty(extent, start, stop, before):
  max_left(Nat.min(start, extent), Nat.min(stop, extent), min_mono(stop, start, extent, before))

# The length.

def add_zero_right(+number: Nat) -> {Nat.add(number, 0n) == number : Nat}:
  match number:
    case 0n:
      {==}
    case 1n+p:
      Equal.cong(Nat, Nat, x => 1n+x, Nat.add(p, 0n), p, add_zero_right(p))

def count_below(+fuel: Nat, +low: Nat, +high: Nat, +step: Nat, +k: Nat, below_high: Bool,
  e: {below_high == Nat.is_lt(low, high) : Bool},
  below: {Nat.is_lt(k, Slice.count(fuel, low, high, step, below_high)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(low, Nat.mul(k, step)), high) == True{} : Bool}:
  match fuel k below_high:
    case 0n _ _:
      Empty.absurd({Nat.is_lt(Nat.add(low, Nat.mul(k, step)), high) == True{} : Bool}, Arith.lt_zero(k, below))
    case 1n+rest _ False{}:
      Empty.absurd({Nat.is_lt(Nat.add(low, Nat.mul(k, step)), high) == True{} : Bool}, Arith.lt_zero(k, below))
    case 1n+rest 0n True{}:
      %Equal.sym(Nat, Nat.add(low, 0n), low, add_zero_right(low)) : {Nat.is_lt(_, high) == True{} : Bool}
      Equal.sym(Bool, True{}, Nat.is_lt(low, high), e)
    case 1n+rest 1n+j True{}:
      %Equal.sym(Nat, Nat.add(low, Nat.add(step, Nat.mul(j, step))), Nat.add(Nat.add(low, step), Nat.mul(j, step)),
        Index.add_assoc(low, step, Nat.mul(j, step))) : {Nat.is_lt(_, high) == True{} : Bool}
      count_below(rest, Nat.add(low, step), high, step, j, Nat.is_lt(Nat.add(low, step), high), {==}, below)

def Laws.slice_length_below(start, stop, step, k, below):
  count_below(Slice.fuel(start, stop), start, stop, step, k, Nat.is_lt(start, stop), {==}, below)

# 1 + rest <= step + rest for a positive step.
def step_covers(+step: Nat, +rest: Nat, positive: {Nat.is_lt(0n, step) == True{} : Bool}) ->
  {Nat.is_le(1n+rest, Nat.add(step, rest)) == True{} : Bool}:
  match step:
    case 0n:
      Empty.absurd({Nat.is_le(1n+rest, Nat.add(0n, rest)) == True{} : Bool}, Arith.clash(positive))
    case 1n+s:
      Api.le_add_left(s, rest)

# The fuel is enough while high <= low + fuel: each counted index moves low
# by the step, at least 1, and spends 1 fuel.
def count_reaches(+fuel: Nat, +low: Nat, +high: Nat, +step: Nat, below_high: Bool,
  e: {below_high == Nat.is_lt(low, high) : Bool},
  +positive: {Nat.is_lt(0n, step) == True{} : Bool},
  enough: {Nat.is_le(high, Nat.add(low, fuel)) == True{} : Bool}) ->
  {Nat.is_le(high, Nat.add(low, Nat.mul(Slice.count(fuel, low, high, step, below_high), step))) == True{} : Bool}:
  match fuel below_high:
    case 0n _:
      enough
    case 1n+rest False{}:
      %Equal.sym(Nat, Nat.add(low, 0n), low, add_zero_right(low)) : {Nat.is_le(high, _) == True{} : Bool}
      Arith.not_lt_le(low, high, Equal.sym(Bool, False{}, Nat.is_lt(low, high), e))
    case 1n+rest True{}:
      spent = Facts.le_trans(high, Nat.add(low, 1n+rest), Nat.add(Nat.add(low, step), rest), enough,
        %Index.add_assoc(low, step, rest) :
          {Nat.is_le(Nat.add(low, 1n+rest), _) == True{} : Bool}
        Facts.le_add_cong(low, 1n+rest, Nat.add(step, rest), step_covers(step, rest, positive)))
      %Equal.sym(Nat, Nat.add(low, Nat.add(step, Nat.mul(Slice.count(rest, Nat.add(low, step), high, step, Nat.is_lt(Nat.add(low, step), high)), step))),
        Nat.add(Nat.add(low, step), Nat.mul(Slice.count(rest, Nat.add(low, step), high, step, Nat.is_lt(Nat.add(low, step), high)), step)),
        Index.add_assoc(low, step, Nat.mul(Slice.count(rest, Nat.add(low, step), high, step, Nat.is_lt(Nat.add(low, step), high)), step))) :
        {Nat.is_le(high, _) == True{} : Bool}
      count_reaches(rest, Nat.add(low, step), high, step, Nat.is_lt(Nat.add(low, step), high), {==}, positive, spent)

def Laws.slice_length_reaches(start, stop, step, positive):
  count_reaches(Slice.fuel(start, stop), start, stop, step, Nat.is_lt(start, stop), {==}, positive, Api.le_add_left(start, stop))

# The view, the output dimensions and select's dimensions.

def view_around(+prefix: List<&2, U32>, +extent: U32, +suffix: List<&2, U32>, +outer: Nat) ->
  {Slice.view_at(List.append(&2, U32, prefix, extent <> suffix), Shape.rank_dimensions(prefix), outer) ==
    Slice.View{Nat.mul(outer, Shape.logical_numel(prefix)), U32.to_nat(extent), Shape.logical_numel(suffix)} : Slice.View}:
  match prefix:
    case Nil{}:
      %Equal.sym(Nat, Nat.mul(outer, 1n), outer, Nats.mul_one_right(outer)) :
        {Slice.View{outer, U32.to_nat(extent), Shape.logical_numel(suffix)} ==
          Slice.View{_, U32.to_nat(extent), Shape.logical_numel(suffix)} : Slice.View}
      {==}
    case +head <> +tail:
      %Nats.mul_assoc(outer, U32.to_nat(head), Shape.logical_numel(tail)) :
        {Slice.view_at(List.append(&2, U32, tail, extent <> suffix), Shape.rank_dimensions(tail), Nat.mul(outer, U32.to_nat(head))) ==
          Slice.View{_, U32.to_nat(extent), Shape.logical_numel(suffix)} : Slice.View}
      view_around(tail, extent, suffix, Nat.mul(outer, U32.to_nat(head)))

def Laws.slice_view_around(prefix, extent, suffix):
  Equal.trans(Slice.View, Slice.view(List.append(&2, U32, prefix, extent <> suffix), Shape.rank_dimensions(prefix)),
    Slice.View{Nat.mul(1n, Shape.logical_numel(prefix)), U32.to_nat(extent), Shape.logical_numel(suffix)},
    Slice.View{Shape.logical_numel(prefix), U32.to_nat(extent), Shape.logical_numel(suffix)},
    view_around(prefix, extent, suffix, 1n),
    Equal.cong(Nat, Slice.View, x => Slice.View{x, U32.to_nat(extent), Shape.logical_numel(suffix)},
      Nat.mul(1n, Shape.logical_numel(prefix)), Shape.logical_numel(prefix), add_zero_right(Shape.logical_numel(prefix))))

def Laws.slice_resized_around(prefix, extent, suffix, other):
  match prefix:
    case Nil{}:
      {==}
    case +head <> +tail:
      Equal.cong(List<&2, U32>, List<&2, U32>, rest => head <> rest,
        Slice.resized(List.append(&2, U32, tail, extent <> suffix), Shape.rank_dimensions(tail), other),
        List.append(&2, U32, tail, other <> suffix), Laws.slice_resized_around(tail, extent, suffix, other))

def Laws.select_dropped_around(prefix, extent, suffix):
  match prefix:
    case Nil{}:
      {==}
    case +head <> +tail:
      Equal.cong(List<&2, U32>, List<&2, U32>, rest => head <> rest,
        Slice.dropped(List.append(&2, U32, tail, extent <> suffix), Shape.rank_dimensions(tail)),
        List.append(&2, U32, tail, suffix), Laws.select_dropped_around(tail, extent, suffix))

# The checks and the plan.

def Laws.slice_axis_refused(input, dim, start, stop, step, outside):
  %Equal.sym(Bool, Nat.is_lt(U32.to_nat(dim), Shape.rank_dimensions(Shape.dimensions(input))), False{}, outside) :
    {Slice.checked(Shape.dimensions(input), dim, start, stop, step, Slice.refused(_, Nat.is_lt(0n, U32.to_nat(step)))) ==
      Fail{Slice.LayoutError{Layout.AxisOutOfBounds{}}} : Result<&2, &2, Slice.Error, Slice.Planned>}
  {==}

def Laws.slice_step_refused(input, dim, start, stop, step, inside, zero):
  %Equal.sym(Bool, Nat.is_lt(U32.to_nat(dim), Shape.rank_dimensions(Shape.dimensions(input))), True{}, inside) :
    {Slice.checked(Shape.dimensions(input), dim, start, stop, step, Slice.refused(_, Nat.is_lt(0n, U32.to_nat(step)))) ==
      Fail{Slice.StepNotPositive{}} : Result<&2, &2, Slice.Error, Slice.Planned>}
  %Equal.sym(Nat, U32.to_nat(step), 0n, zero) :
    {Slice.checked(Shape.dimensions(input), dim, start, stop, step, Slice.refused(True{}, Nat.is_lt(0n, _))) ==
      Fail{Slice.StepNotPositive{}} : Result<&2, &2, Slice.Error, Slice.Planned>}
  {==}

def planned_shaped(+dimensions: List<&2, U32>, +dim: U32, +start: U32, +stop: U32, +step: U32,
  built: Shape.ConstructResult<Slice.resized(dimensions, U32.to_nat(dim), U32.from_nat(Slice.plan_length(Slice.plan_of(dimensions, dim, start, stop, step))))>) ->
  Laws.PlannedAs(dimensions, dim, start, stop, step,
    Slice.shaped(Slice.plan_of(dimensions, dim, start, stop, step),
      Slice.resized(dimensions, U32.to_nat(dim), U32.from_nat(Slice.plan_length(Slice.plan_of(dimensions, dim, start, stop, step)))), built)):
  match built:
    case Shape.SError{error}:
      Unit{}
    case Shape.SBuilt{+shape, preserved}:
      ({==}, preserved)

def planned_checked(+dimensions: List<&2, U32>, +dim: U32, +start: U32, +stop: U32, +step: U32, refused: Maybe<&2, Slice.Error>) ->
  Laws.PlannedAs(dimensions, dim, start, stop, step, Slice.checked(dimensions, dim, start, stop, step, refused)):
  match refused:
    case Some{error}:
      Unit{}
    case None{}:
      planned_shaped(dimensions, dim, start, stop, step,
        Shape.construct(Slice.resized(dimensions, U32.to_nat(dim), U32.from_nat(Slice.plan_length(Slice.plan_of(dimensions, dim, start, stop, step))))))

def Laws.slice_planned(input, dim, start, stop, step):
  planned_checked(Shape.dimensions(input), dim, start, stop, step, Slice.refusal(Shape.dimensions(input), dim, step))

def Laws.slice_refused(input, dim, start, stop, step, error, refused):
  Core.Packed{+shape, tensor} = input
  %Equal.sym(Result<&2, &2, Slice.Error, Slice.Planned>, Slice.plan(shape, dim, start, stop, step), Fail{error}, refused) :
    {Slice.sliced(shape, tensor, _) == (Core.Packed{shape, tensor}, Fail{error}) : Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}
  {==}

# narrow and select.

def Laws.narrow_out_of_range(input, dim, start, length, inside, past):
  Core.Packed{+shape, tensor} = input
  %Equal.sym(Bool, Nat.is_lt(U32.to_nat(dim), Shape.rank_dimensions(Shape.dimensions(shape))), True{}, inside) :
    {Slice.narrow_checked(shape, tensor, dim, start, length, Slice.refused(_, Nat.is_lt(0n, U32.to_nat(1)))) ==
      (Core.Packed{shape, tensor}, Fail{Slice.OutOfRange{}}) : Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}
  %Equal.sym(Bool, Nat.is_le(Nat.add(U32.to_nat(start), U32.to_nat(length)), Slice.extent_at(Shape.dimensions(shape), dim)), False{}, past) :
    {Slice.narrow_ranged(shape, tensor, dim, start, length, _) ==
      (Core.Packed{shape, tensor}, Fail{Slice.OutOfRange{}}) : Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}
  {==}

def Laws.narrow_slices(input, dim, start, length, inside, fits):
  Core.Packed{+shape, tensor} = input
  %Equal.sym(Bool, Nat.is_lt(U32.to_nat(dim), Shape.rank_dimensions(Shape.dimensions(shape))), True{}, inside) :
    {Slice.narrow_checked(shape, tensor, dim, start, length, Slice.refused(_, Nat.is_lt(0n, U32.to_nat(1)))) ==
      Slice.slice(Core.Packed{shape, tensor}, dim, start, U32.add(start, length), 1) : Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}
  %Equal.sym(Bool, Nat.is_le(Nat.add(U32.to_nat(start), U32.to_nat(length)), Slice.extent_at(Shape.dimensions(shape), dim)), True{}, fits) :
    {Slice.narrow_ranged(shape, tensor, dim, start, length, _) ==
      Slice.slice(Core.Packed{shape, tensor}, dim, start, U32.add(start, length), 1) : Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}
  {==}

def Laws.narrow_axis_refused(input, dim, start, length, outside):
  Core.Packed{+shape, tensor} = input
  %Equal.sym(Bool, Nat.is_lt(U32.to_nat(dim), Shape.rank_dimensions(Shape.dimensions(shape))), False{}, outside) :
    {Slice.narrow_checked(shape, tensor, dim, start, length, Slice.refused(_, Nat.is_lt(0n, U32.to_nat(1)))) ==
      (Core.Packed{shape, tensor}, Fail{Slice.LayoutError{Layout.AxisOutOfBounds{}}}) : Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}
  {==}

def transformed_as(result: Result<&2, &1, Transforms.Error, Core.Dynamic>) ->
  {Slice.transformed(result) == Laws.reshape_as_slice(result) : Result<&2, &1, Slice.Error, Core.Dynamic>}:
  match result:
    case Fail{Transforms.LayoutError{error}}:
      {==}
    case Fail{Transforms.TensorError{error}}:
      {==}
    case Done{output}:
      {==}

def reshaped_as(+dimensions: List<&2, U32>, pair: Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>) ->
  {Slice.reshaped(dimensions, pair) == Laws.reshaped(dimensions, pair) : Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}:
  match pair:
    case Tuple{input, Fail{error}}:
      {==}
    case Tuple{input, Done{output}}:
      Equal.cong(Result<&2, &1, Slice.Error, Core.Dynamic>, Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>,
        result => (input, result), Slice.transformed(Transforms.reshape_dynamic(output, dimensions)),
        Laws.reshape_as_slice(Transforms.reshape_dynamic(output, dimensions)), transformed_as(Transforms.reshape_dynamic(output, dimensions)))

def Laws.select_out_of_range(input, dim, index, inside, past):
  Core.Packed{+shape, tensor} = input
  %Equal.sym(Bool, Nat.is_lt(U32.to_nat(dim), Shape.rank_dimensions(Shape.dimensions(shape))), True{}, inside) :
    {Slice.select_checked(shape, tensor, dim, index, Slice.refused(_, Nat.is_lt(0n, U32.to_nat(1)))) ==
      (Core.Packed{shape, tensor}, Fail{Slice.OutOfRange{}}) : Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}
  %Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), Slice.extent_at(Shape.dimensions(shape), dim)), False{}, past) :
    {Slice.select_ranged(shape, tensor, dim, index, _) ==
      (Core.Packed{shape, tensor}, Fail{Slice.OutOfRange{}}) : Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}
  {==}

def Laws.select_slices(input, dim, index, inside, below):
  Core.Packed{+shape, tensor} = input
  %Equal.sym(Bool, Nat.is_lt(U32.to_nat(dim), Shape.rank_dimensions(Shape.dimensions(shape))), True{}, inside) :
    {Slice.select_checked(shape, tensor, dim, index, Slice.refused(_, Nat.is_lt(0n, U32.to_nat(1)))) ==
      Laws.reshaped(Slice.dropped(Shape.dimensions(shape), U32.to_nat(dim)), Slice.slice(Core.Packed{shape, tensor}, dim, index, U32.add(index, 1), 1)) :
      Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}
  %Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), Slice.extent_at(Shape.dimensions(shape), dim)), True{}, below) :
    {Slice.select_ranged(shape, tensor, dim, index, _) ==
      Laws.reshaped(Slice.dropped(Shape.dimensions(shape), U32.to_nat(dim)), Slice.slice(Core.Packed{shape, tensor}, dim, index, U32.add(index, 1), 1)) :
      Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}
  reshaped_as(Slice.dropped(Shape.dimensions(shape), U32.to_nat(dim)), Slice.slice(Core.Packed{shape, tensor}, dim, index, U32.add(index, 1), 1))

def Laws.select_axis_refused(input, dim, index, outside):
  Core.Packed{+shape, tensor} = input
  %Equal.sym(Bool, Nat.is_lt(U32.to_nat(dim), Shape.rank_dimensions(Shape.dimensions(shape))), False{}, outside) :
    {Slice.select_checked(shape, tensor, dim, index, Slice.refused(_, Nat.is_lt(0n, U32.to_nat(1)))) ==
      (Core.Packed{shape, tensor}, Fail{Slice.LayoutError{Layout.AxisOutOfBounds{}}}) : Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}
  {==}

# The cells of the builds.

def take_value(-b: Array<F32>, pair: Array<F32> & F32) ->
  {Pair.fst(F32, Array<F32> & Array<F32>, Slice.take(b, pair)) == Refinement.second(F32, pair) : F32}:
  (a, x) = pair
  {==}

def take_state(-b: Array<F32>, pair: Array<F32> & F32) ->
  {Pair.snd(F32, Array<F32> & Array<F32>, Slice.take(b, pair)) == (Refinement.first(F32, pair), b) : Array<F32> & Array<F32>}:
  (a, x) = pair
  {==}

# A read of A through `take` gives the word A holds there ...
def read_word(+da: Nat, -A: Array<F32>, -b: Array<F32>, +index: Nat, -valid: Model.Balanced(F32, da, A)) ->
  Parallel.Fact<{Some{Pair.fst(F32, Array<F32> & Array<F32>, Slice.take(b, Array.get(F32, A, U32.from_nat(index))))} ==
    Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(index))) : Maybe<&2, F32>}>:
  Parallel.Fact{
    %Equal.sym(F32, Pair.fst(F32, Array<F32> & Array<F32>, Slice.take(b, Array.get(F32, A, U32.from_nat(index)))),
      Refinement.second(F32, Array.get(F32, A, U32.from_nat(index))), take_value(b, Array.get(F32, A, U32.from_nat(index)))) :
      {Some{_} == Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(index))) : Maybe<&2, F32>}
    {\{Parallel.Fact: q => q} :
      Parallel.Fact<{Some{Refinement.second(F32, Array.get(F32, A, U32.from_nat(index)))} ==
        Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(index))) : Maybe<&2, F32>}> ->
      {Some{Refinement.second(F32, Array.get(F32, A, U32.from_nat(index)))} ==
        Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(index))) : Maybe<&2, F32>}}(
      Parallel.read_value(da, A, A, U32.from_nat(index), Array.get(F32, A, U32.from_nat(index)), {==}, {==}, valid))}

# ... and hands both arrays back.
def read_keeps(+da: Nat, -A: Array<F32>, -b: Array<F32>, +index: Nat, -valid: Model.Balanced(F32, da, A)) ->
  Parallel.Fact<{Pair.snd(F32, Array<F32> & Array<F32>, Slice.take(b, Array.get(F32, A, U32.from_nat(index)))) == (A, b) :
    Array<F32> & Array<F32>}>:
  Parallel.Fact{Equal.trans(Array<F32> & Array<F32>, Pair.snd(F32, Array<F32> & Array<F32>, Slice.take(b, Array.get(F32, A, U32.from_nat(index)))),
    (Refinement.first(F32, Array.get(F32, A, U32.from_nat(index))), b), (A, b),
    take_state(b, Array.get(F32, A, U32.from_nat(index))),
    Equal.cong(Array<F32>, Array<F32> & Array<F32>, x => (x, b), Refinement.first(F32, Array.get(F32, A, U32.from_nat(index))), A,
      {\{Parallel.Fact: q => q} :
        Parallel.Fact<{Refinement.first(F32, Array.get(F32, A, U32.from_nat(index))) == A : Array<F32>}> ->
        {Refinement.first(F32, Array.get(F32, A, U32.from_nat(index))) == A : Array<F32>}}(
        Parallel.read_keeps(da, A, A, U32.from_nat(index), Array.get(F32, A, U32.from_nat(index)), {==}, {==}, valid))))}

def empty() -> Array<F32>:
  Array.new(F32, 0n, 0.0)

def cell_keeps(+plan: Slice.Plan, +at: Nat, +da: Nat, -A: Array<F32>, inside: Bool, -valid: Model.Balanced(F32, da, A)) ->
  Parallel.Fact<{Pair.snd(F32, Array<F32> & Array<F32>, Slice.cell(plan, at, (A, empty()), inside)) == (A, empty()) : Array<F32> & Array<F32>}>:
  match inside:
    case False{}:
      Parallel.Fact{{==}}
    case True{}:
      read_keeps(da, A, empty(), Slice.source(plan, at), valid)

def scattered_keeps(+da: Nat, -A: Array<F32>, at: Maybe<&2, Nat>, -valid: Model.Balanced(F32, da, A)) ->
  Parallel.Fact<{Pair.snd(F32, Array<F32> & Array<F32>, Slice.scattered((A, empty()), at)) == (A, empty()) : Array<F32> & Array<F32>}>:
  match at:
    case None{}:
      Parallel.Fact{{==}}
    case Some{+index}:
      read_keeps(da, A, empty(), index, valid)

def scatter_keeps(+plan: Slice.Plan, +at: Nat, +da: Nat, -A: Array<F32>, inside: Bool, -valid: Model.Balanced(F32, da, A)) ->
  Parallel.Fact<{Pair.snd(F32, Array<F32> & Array<F32>, Slice.scatter_cell(plan, at, (A, empty()), inside)) == (A, empty()) : Array<F32> & Array<F32>}>:
  match inside:
    case False{}:
      Parallel.Fact{{==}}
    case True{}:
      scattered_keeps(da, A, Slice.target(plan, at), valid)

def SplitFacts(-A: Array<F32>) -> Type:
  {Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, empty()))) == (A, empty()) : Array<F32> & Array<F32>} &
    {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, empty()))) == (A, empty()) : Array<F32> & Array<F32>}

def split_facts(+da: Nat, -A: Array<F32>, -valid: Model.Balanced(F32, da, A)) -> Parallel.Fact<SplitFacts(A)>:
  Parallel.Fact{{\{Parallel.Fact: q => q} : Parallel.Fact<SplitFacts(A)> -> SplitFacts(A)}(
    Parallel.split_equal(da, 0n, A, empty(), valid, ArrayCore.new_balanced(F32, 0n, 0.0)))}

# Every slot of the forward build holds what its cell computes from (A, []).
# The Fact holds the cells eta-expanded: `--verdict` rejects the unexpanded
# function in this position although `--check-only` accepts it.
def build_cells(+depth: Nat, +splits: Nat, +plan: Slice.Plan, +count: Nat, +base: Nat, +da: Nat, -A: Array<F32>, a: Array<F32>,
  -same: {a == A : Array<F32>}, -valid: Model.Balanced(F32, da, A)) ->
  Parallel.Fact<Generic.Cells(~Slice.Plan, ~Array<F32>, ~Array<F32>, ~Slice.cell, plan, (A, empty()), depth, count, base,
    Slice.build(depth, splits, plan, count, base, a))>:
  Parallel.Fact{path =>
    {\{Generic.Fact: q => q} :
      Generic.Fact<Generic.Cells(~Slice.Plan, ~Array<F32>, ~Array<F32>, ~Slice.cell, plan, (A, empty()), depth, count, base,
        Slice.build(depth, splits, plan, count, base, a))> ->
      Generic.Cells(~Slice.Plan, ~Array<F32>, ~Array<F32>, ~Slice.cell, plan, (A, empty()), depth, count, base,
        Slice.build(depth, splits, plan, count, base, a))}(
      Generic.build_cells(~Slice.Plan, ~Slice.Plan, ~Array<F32>, ~Array<F32>, ~Slice.same, ~Elementwise.split, ~Slice.cell,
        plan, (A, empty()), depth, splits, count, base, (a, empty()), Parallel.pair_equal(a, empty(), A, empty(), same, {==}),
        at => inside => {\{Parallel.Fact: q => q} :
          Parallel.Fact<{Pair.snd(F32, Array<F32> & Array<F32>, Slice.cell(plan, at, (A, empty()), inside)) == (A, empty()) : Array<F32> & Array<F32>}> ->
          {Pair.snd(F32, Array<F32> & Array<F32>, Slice.cell(plan, at, (A, empty()), inside)) == (A, empty()) : Array<F32> & Array<F32>}}(
          cell_keeps(plan, at, da, A, inside, valid)),
        Pair.fst({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, empty()))) == (A, empty()) : Array<F32> & Array<F32>},
          {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, empty()))) == (A, empty()) : Array<F32> & Array<F32>},
          {\{Parallel.Fact: q => q} : Parallel.Fact<SplitFacts(A)> -> SplitFacts(A)}(split_facts(da, A, valid))),
        Pair.snd({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, empty()))) == (A, empty()) : Array<F32> & Array<F32>},
          {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, empty()))) == (A, empty()) : Array<F32> & Array<F32>},
          {\{Parallel.Fact: q => q} : Parallel.Fact<SplitFacts(A)> -> SplitFacts(A)}(split_facts(da, A, valid)))))(path)}

def scatter_cells(+depth: Nat, +splits: Nat, +plan: Slice.Plan, +count: Nat, +base: Nat, +da: Nat, -A: Array<F32>, a: Array<F32>,
  -same: {a == A : Array<F32>}, -valid: Model.Balanced(F32, da, A)) ->
  Parallel.Fact<Generic.Cells(~Slice.Plan, ~Array<F32>, ~Array<F32>, ~Slice.scatter_cell, plan, (A, empty()), depth, count, base,
    Slice.scatter_build(depth, splits, plan, count, base, a))>:
  Parallel.Fact{path =>
    {\{Generic.Fact: q => q} :
      Generic.Fact<Generic.Cells(~Slice.Plan, ~Array<F32>, ~Array<F32>, ~Slice.scatter_cell, plan, (A, empty()), depth, count, base,
        Slice.scatter_build(depth, splits, plan, count, base, a))> ->
      Generic.Cells(~Slice.Plan, ~Array<F32>, ~Array<F32>, ~Slice.scatter_cell, plan, (A, empty()), depth, count, base,
        Slice.scatter_build(depth, splits, plan, count, base, a))}(
      Generic.build_cells(~Slice.Plan, ~Slice.Plan, ~Array<F32>, ~Array<F32>, ~Slice.same, ~Elementwise.split, ~Slice.scatter_cell,
        plan, (A, empty()), depth, splits, count, base, (a, empty()), Parallel.pair_equal(a, empty(), A, empty(), same, {==}),
        at => inside => {\{Parallel.Fact: q => q} :
          Parallel.Fact<{Pair.snd(F32, Array<F32> & Array<F32>, Slice.scatter_cell(plan, at, (A, empty()), inside)) == (A, empty()) : Array<F32> & Array<F32>}> ->
          {Pair.snd(F32, Array<F32> & Array<F32>, Slice.scatter_cell(plan, at, (A, empty()), inside)) == (A, empty()) : Array<F32> & Array<F32>}}(
          scatter_keeps(plan, at, da, A, inside, valid)),
        Pair.fst({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, empty()))) == (A, empty()) : Array<F32> & Array<F32>},
          {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, empty()))) == (A, empty()) : Array<F32> & Array<F32>},
          {\{Parallel.Fact: q => q} : Parallel.Fact<SplitFacts(A)> -> SplitFacts(A)}(split_facts(da, A, valid))),
        Pair.snd({Pair.fst(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, empty()))) == (A, empty()) : Array<F32> & Array<F32>},
          {Pair.snd(Array<F32> & Array<F32>, Array<F32> & Array<F32>, Elementwise.split((A, empty()))) == (A, empty()) : Array<F32> & Array<F32>},
          {\{Parallel.Fact: q => q} : Parallel.Fact<SplitFacts(A)> -> SplitFacts(A)}(split_facts(da, A, valid)))))(path)}

# The arithmetic of the positions.

def sub_zero(+number: Nat) -> {Nat.sub(number, 0n) == number : Nat}:
  match number:
    case 0n:
      {==}
    case 1n+p:
      {==}

def sub_cancel(+a: Nat, +b: Nat) -> {Nat.sub(Nat.add(a, b), a) == b : Nat}:
  match a:
    case 0n:
      sub_zero(b)
    case 1n+p:
      sub_cancel(p, b)

# (k * step) / step = k for a positive step.
def div_step(+k: Nat, +step: Nat, +positive: {Nat.is_lt(0n, step) == True{} : Bool}) -> {Nat.div(Nat.mul(k, step), step) == k : Nat}:
  Equal.trans(Nat, Nat.div(Nat.mul(k, step), step), Nat.div(Nat.add(Nat.mul(k, step), 0n), step), k,
    Equal.cong(Nat, Nat, x => Nat.div(x, step), Nat.mul(k, step), Nat.add(Nat.mul(k, step), 0n),
      Equal.sym(Nat, Nat.add(Nat.mul(k, step), 0n), Nat.mul(k, step), add_zero_right(Nat.mul(k, step)))),
    BatchedProof.div_exact(k, step, 0n, positive))

# The forward cell (o, k, r) reads input cell (o, start + k * step, r).
def source_at(+outer: Nat, +extent: Nat, +inner: Nat, +start: Nat, +step: Nat, +length: Nat, +o: Nat, +k: Nat, +r: Nat,
  +k_below: {Nat.is_lt(k, length) == True{} : Bool}, +r_below: {Nat.is_lt(r, inner) == True{} : Bool}) ->
  {Slice.source(Slice.Plan{outer, extent, inner, start, step, length}, Laws.position(length, inner, o, k, r)) ==
    Laws.position(extent, inner, o, Nat.add(start, Nat.mul(k, step)), r) : Nat}:
  +row = Nat.add(Nat.mul(o, length), k)
  +pos = Laws.position(length, inner, o, k, r)
  %Equal.sym(Nat, Nat.div(pos, inner), row, BatchedProof.div_exact(row, inner, r, r_below)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(_, length), extent), Nat.add(start, Nat.mul(Nat.mod(_, length), step))), inner), Nat.mod(pos, inner)) ==
      Laws.position(extent, inner, o, Nat.add(start, Nat.mul(k, step)), r) : Nat}
  %Equal.sym(Nat, Nat.mod(pos, inner), r, BatchedProof.mod_exact(row, inner, r, r_below)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(row, length), extent), Nat.add(start, Nat.mul(Nat.mod(row, length), step))), inner), _) ==
      Laws.position(extent, inner, o, Nat.add(start, Nat.mul(k, step)), r) : Nat}
  %Equal.sym(Nat, Nat.div(row, length), o, BatchedProof.div_exact(o, length, k, k_below)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(_, extent), Nat.add(start, Nat.mul(Nat.mod(row, length), step))), inner), r) ==
      Laws.position(extent, inner, o, Nat.add(start, Nat.mul(k, step)), r) : Nat}
  %Equal.sym(Nat, Nat.mod(row, length), k, BatchedProof.mod_exact(o, length, k, k_below)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(o, extent), Nat.add(start, Nat.mul(_, step))), inner), r) ==
      Laws.position(extent, inner, o, Nat.add(start, Nat.mul(k, step)), r) : Nat}
  {==}

def Laws.slice_build_cells(depth, splits, outer, extent, inner, start, step, length, count, base, da, A, a, same, valid, path, o, k, r,
  k_below, r_below, at, inside):
  Laws.Holds{
    %Equal.trans(Nat, Slice.source(Slice.Plan{outer, extent, inner, start, step, length}, Nat.add(Model.offset(depth, path), base)),
      Slice.source(Slice.Plan{outer, extent, inner, start, step, length}, Laws.position(length, inner, o, k, r)),
      Laws.position(extent, inner, o, Nat.add(start, Nat.mul(k, step)), r),
      Equal.cong(Nat, Nat, x => Slice.source(Slice.Plan{outer, extent, inner, start, step, length}, x),
        Nat.add(Model.offset(depth, path), base), Laws.position(length, inner, o, k, r), at),
      source_at(outer, extent, inner, start, step, length, o, k, r, k_below, r_below)) :
      {Model.observe(F32, depth, Slice.build(depth, splits, Slice.Plan{outer, extent, inner, start, step, length}, count, base, a), path) ==
        Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(_))) : Maybe<&2, F32>}
    Equal.trans(Maybe<&2, F32>,
      Model.observe(F32, depth, Slice.build(depth, splits, Slice.Plan{outer, extent, inner, start, step, length}, count, base, a), path),
      Some{Pair.fst(F32, Array<F32> & Array<F32>, Slice.cell(Slice.Plan{outer, extent, inner, start, step, length},
        Nat.add(Model.offset(depth, path), base), (A, empty()), True{}))},
      Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(Slice.source(Slice.Plan{outer, extent, inner, start, step, length},
        Nat.add(Model.offset(depth, path), base))))),
      %inside :
        {Model.observe(F32, depth, Slice.build(depth, splits, Slice.Plan{outer, extent, inner, start, step, length}, count, base, a), path) ==
          Some{Pair.fst(F32, Array<F32> & Array<F32>, Slice.cell(Slice.Plan{outer, extent, inner, start, step, length},
            Nat.add(Model.offset(depth, path), base), (A, empty()), _))} : Maybe<&2, F32>}
      {\{Parallel.Fact: q => q} :
        Parallel.Fact<Generic.Cells(~Slice.Plan, ~Array<F32>, ~Array<F32>, ~Slice.cell, Slice.Plan{outer, extent, inner, start, step, length},
          (A, empty()), depth, count, base, Slice.build(depth, splits, Slice.Plan{outer, extent, inner, start, step, length}, count, base, a))> ->
        Generic.Cells(~Slice.Plan, ~Array<F32>, ~Array<F32>, ~Slice.cell, Slice.Plan{outer, extent, inner, start, step, length},
          (A, empty()), depth, count, base, Slice.build(depth, splits, Slice.Plan{outer, extent, inner, start, step, length}, count, base, a))}(
        build_cells(depth, splits, Slice.Plan{outer, extent, inner, start, step, length}, count, base, da, A, a, same, valid))(path),
      {\{Parallel.Fact: q => q} :
        Parallel.Fact<{Some{Pair.fst(F32, Array<F32> & Array<F32>, Slice.take(empty(), Array.get(F32, A, U32.from_nat(Slice.source(Slice.Plan{outer, extent, inner, start, step, length},
          Nat.add(Model.offset(depth, path), base))))))} ==
          Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(Slice.source(Slice.Plan{outer, extent, inner, start, step, length},
            Nat.add(Model.offset(depth, path), base))))) : Maybe<&2, F32>}> ->
        {Some{Pair.fst(F32, Array<F32> & Array<F32>, Slice.take(empty(), Array.get(F32, A, U32.from_nat(Slice.source(Slice.Plan{outer, extent, inner, start, step, length},
          Nat.add(Model.offset(depth, path), base))))))} ==
          Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(Slice.source(Slice.Plan{outer, extent, inner, start, step, length},
            Nat.add(Model.offset(depth, path), base))))) : Maybe<&2, F32>}}(
        read_word(da, A, empty(), Slice.source(Slice.Plan{outer, extent, inner, start, step, length}, Nat.add(Model.offset(depth, path), base)), valid)))}

# The backward target of input cell (o, i, r).

def picked_k(+outer: Nat, +extent: Nat, +inner: Nat, +start: Nat, +step: Nat, +length: Nat, +row: Nat, +within: Nat, +i: Nat, +k: Nat) ->
  Maybe<&2, Nat>:
  Slice.picked(Slice.Plan{outer, extent, inner, start, step, length}, row, within, i, k, Slice.hits(start, step, length, i, k))

def picked_at(+outer: Nat, +extent: Nat, +inner: Nat, +start: Nat, +step: Nat, +length: Nat, +row: Nat, +within: Nat, +i: Nat) ->
  Maybe<&2, Nat>:
  picked_k(outer, extent, inner, start, step, length, row, within, i, Nat.div(Nat.sub(i, start), step))

# `Slice.target` from the row and the remainder of the position.
def target_of(+outer: Nat, +extent: Nat, +inner: Nat, +start: Nat, +step: Nat, +length: Nat, +row: Nat, +within: Nat) -> Maybe<&2, Nat>:
  picked_at(outer, extent, inner, start, step, length, row, within, Nat.mod(row, extent))

# Position (o * extent + i) * inner + r splits into row o * extent + i and
# remainder r, and the row into o and i.
def target_split(+outer: Nat, +extent: Nat, +inner: Nat, +start: Nat, +step: Nat, +length: Nat, +o: Nat, +i: Nat, +r: Nat,
  +i_below: {Nat.is_lt(i, extent) == True{} : Bool}, +r_below: {Nat.is_lt(r, inner) == True{} : Bool}) ->
  {Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, Laws.position(extent, inner, o, i, r)) ==
    picked_at(outer, extent, inner, start, step, length, Nat.add(Nat.mul(o, extent), i), r, i) : Maybe<&2, Nat>}:
  +row = Nat.add(Nat.mul(o, extent), i)
  +pos = Laws.position(extent, inner, o, i, r)
  %Equal.sym(Nat, Nat.div(pos, inner), row, BatchedProof.div_exact(row, inner, r, r_below)) :
    {target_of(outer, extent, inner, start, step, length, _, Nat.mod(pos, inner)) ==
      picked_at(outer, extent, inner, start, step, length, row, r, i) : Maybe<&2, Nat>}
  %Equal.sym(Nat, Nat.mod(pos, inner), r, BatchedProof.mod_exact(row, inner, r, r_below)) :
    {target_of(outer, extent, inner, start, step, length, row, _) == picked_at(outer, extent, inner, start, step, length, row, r, i) : Maybe<&2, Nat>}
  %Equal.sym(Nat, Nat.mod(row, extent), i, BatchedProof.mod_exact(o, extent, i, i_below)) :
    {picked_at(outer, extent, inner, start, step, length, row, r, _) == picked_at(outer, extent, inner, start, step, length, row, r, i) : Maybe<&2, Nat>}
  {==}

# Input cell (o, i, r) that no k below the length reaches reads nothing.
def missed_k(+outer: Nat, +extent: Nat, +inner: Nat, +start: Nat, +step: Nat, +length: Nat, +row: Nat, +r: Nat, +i: Nat, +k: Nat,
  missed: @+k: Nat -> @below: {Nat.is_lt(k, length) == True{} : Bool} -> {Nat.is_eq(Nat.add(start, Nat.mul(k, step)), i) == False{} : Bool},
  decision: Shape.BoolCase<Slice.hits(start, step, length, i, k)>) ->
  {picked_k(outer, extent, inner, start, step, length, row, r, i, k) == None{} : Maybe<&2, Nat>}:
  match decision:
    case Shape.BFalse{equal}:
      %equal : {Slice.picked(Slice.Plan{outer, extent, inner, start, step, length}, row, r, i, k, _) == None{} : Maybe<&2, Nat>}
      {==}
    case Shape.BTrue{equal}:
      +rest = Arith.and_right(Nat.is_le(start, i), Bool.and(Nat.is_lt(k, length), Nat.is_eq(Nat.add(start, Nat.mul(k, step)), i)),
        Equal.sym(Bool, True{}, Slice.hits(start, step, length, i, k), equal))
      Empty.absurd({picked_k(outer, extent, inner, start, step, length, row, r, i, k) == None{} : Maybe<&2, Nat>},
        Arith.clash(Equal.trans(Bool, False{}, Nat.is_eq(Nat.add(start, Nat.mul(k, step)), i), True{},
          Equal.sym(Bool, Nat.is_eq(Nat.add(start, Nat.mul(k, step)), i), False{},
            missed(k, Arith.and_left(Nat.is_lt(k, length), Nat.is_eq(Nat.add(start, Nat.mul(k, step)), i), rest))),
          Arith.and_right(Nat.is_lt(k, length), Nat.is_eq(Nat.add(start, Nat.mul(k, step)), i), rest))))

def target_miss(+outer: Nat, +extent: Nat, +inner: Nat, +start: Nat, +step: Nat, +length: Nat, +o: Nat, +i: Nat, +r: Nat,
  +i_below: {Nat.is_lt(i, extent) == True{} : Bool}, +r_below: {Nat.is_lt(r, inner) == True{} : Bool},
  missed: @+k: Nat -> @below: {Nat.is_lt(k, length) == True{} : Bool} -> {Nat.is_eq(Nat.add(start, Nat.mul(k, step)), i) == False{} : Bool}) ->
  {Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, Laws.position(extent, inner, o, i, r)) == None{} : Maybe<&2, Nat>}:
  Equal.trans(Maybe<&2, Nat>, Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, Laws.position(extent, inner, o, i, r)),
    picked_at(outer, extent, inner, start, step, length, Nat.add(Nat.mul(o, extent), i), r, i), None{},
    target_split(outer, extent, inner, start, step, length, o, i, r, i_below, r_below),
    missed_k(outer, extent, inner, start, step, length, Nat.add(Nat.mul(o, extent), i), r, i, Nat.div(Nat.sub(i, start), step), missed,
      Shape.bool_case(Slice.hits(start, step, length, i, Nat.div(Nat.sub(i, start), step)))))

def ScatterCells(+depth: Nat, +splits: Nat, +plan: Slice.Plan, +count: Nat, +base: Nat, -G: Array<F32>, -g: Array<F32>) -> Type:
  Generic.Cells(~Slice.Plan, ~Array<F32>, ~Array<F32>, ~Slice.scatter_cell, plan, (G, empty()), depth, count, base,
    Slice.scatter_build(depth, splits, plan, count, base, g))

# The slot of input cell (o, i, r), below the count, holds what the gradient
# target of the cell names.
def scatter_slot(+depth: Nat, +splits: Nat, +plan: Slice.Plan, +count: Nat, +base: Nat, +dg: Nat, -G: Array<F32>, g: Array<F32>,
  -same: {g == G : Array<F32>}, -valid: Model.Balanced(F32, dg, G), +path: Model.Path(depth),
  -inside: {Nat.is_lt(Nat.add(Model.offset(depth, path), base), count) == True{} : Bool}) ->
  Parallel.Fact<{Model.observe(F32, depth, Slice.scatter_build(depth, splits, plan, count, base, g), path) ==
    Some{Pair.fst(F32, Array<F32> & Array<F32>, Slice.scattered((G, empty()), Slice.target(plan, Nat.add(Model.offset(depth, path), base))))} :
    Maybe<&2, F32>}>:
  Parallel.Fact{
    %inside :
      {Model.observe(F32, depth, Slice.scatter_build(depth, splits, plan, count, base, g), path) ==
        Some{Pair.fst(F32, Array<F32> & Array<F32>, Slice.scatter_cell(plan, Nat.add(Model.offset(depth, path), base), (G, empty()), _))} :
        Maybe<&2, F32>}
    {\{Parallel.Fact: q => q} : Parallel.Fact<ScatterCells(depth, splits, plan, count, base, G, g)> -> ScatterCells(depth, splits, plan, count, base, G, g)}(
      scatter_cells(depth, splits, plan, count, base, dg, G, g, same, valid))(path)}

def Laws.slice_scatter_miss(depth, splits, outer, extent, inner, start, step, length, count, base, dg, G, g, same, valid, path, o, i, r,
  i_below, r_below, missed, at, inside):
  Laws.Holds{
    Equal.trans(Maybe<&2, F32>,
      Model.observe(F32, depth, Slice.scatter_build(depth, splits, Slice.Plan{outer, extent, inner, start, step, length}, count, base, g), path),
      Some{Pair.fst(F32, Array<F32> & Array<F32>, Slice.scattered((G, empty()),
        Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, Nat.add(Model.offset(depth, path), base))))},
      Some{0.0},
      {\{Parallel.Fact: q => q} :
        Parallel.Fact<{Model.observe(F32, depth, Slice.scatter_build(depth, splits, Slice.Plan{outer, extent, inner, start, step, length}, count, base, g), path) ==
          Some{Pair.fst(F32, Array<F32> & Array<F32>, Slice.scattered((G, empty()),
            Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, Nat.add(Model.offset(depth, path), base))))} : Maybe<&2, F32>}> ->
        {Model.observe(F32, depth, Slice.scatter_build(depth, splits, Slice.Plan{outer, extent, inner, start, step, length}, count, base, g), path) ==
          Some{Pair.fst(F32, Array<F32> & Array<F32>, Slice.scattered((G, empty()),
            Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, Nat.add(Model.offset(depth, path), base))))} : Maybe<&2, F32>}}(
        scatter_slot(depth, splits, Slice.Plan{outer, extent, inner, start, step, length}, count, base, dg, G, g, same, valid, path, inside)),
      %Equal.sym(Maybe<&2, Nat>, Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, Nat.add(Model.offset(depth, path), base)), None{},
        Equal.trans(Maybe<&2, Nat>, Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, Nat.add(Model.offset(depth, path), base)),
          Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, Laws.position(extent, inner, o, i, r)), None{},
          Equal.cong(Nat, Maybe<&2, Nat>, x => Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, x),
            Nat.add(Model.offset(depth, path), base), Laws.position(extent, inner, o, i, r), at),
          target_miss(outer, extent, inner, start, step, length, o, i, r, i_below, r_below, missed))) :
        {Some{Pair.fst(F32, Array<F32> & Array<F32>, Slice.scattered((G, empty()), _))} == Some{0.0} : Maybe<&2, F32>}
      {==})}

# start <= start + k * step, k < length and start + k * step is itself.
def hits_true(+start: Nat, +step: Nat, +length: Nat, +k: Nat, +k_below: {Nat.is_lt(k, length) == True{} : Bool}) ->
  {Slice.hits(start, step, length, Nat.add(start, Nat.mul(k, step)), k) == True{} : Bool}:
  %Equal.sym(Bool, Nat.is_le(start, Nat.add(start, Nat.mul(k, step))), True{}, Api.le_add_right(start, Nat.mul(k, step))) :
    {Bool.and(_, Bool.and(Nat.is_lt(k, length), Nat.is_eq(Nat.add(start, Nat.mul(k, step)), Nat.add(start, Nat.mul(k, step))))) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_lt(k, length), True{}, k_below) :
    {Bool.and(True{}, Bool.and(_, Nat.is_eq(Nat.add(start, Nat.mul(k, step)), Nat.add(start, Nat.mul(k, step))))) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_eq(Nat.add(start, Nat.mul(k, step)), Nat.add(start, Nat.mul(k, step))), True{},
    Arith.eq_refl(Nat.add(start, Nat.mul(k, step)))) :
    {Bool.and(True{}, Bool.and(True{}, _)) == True{} : Bool}
  {==}

# Input cell (o, start + k * step, r) reads gradient cell (o, k, r).
def target_hit(+outer: Nat, +extent: Nat, +inner: Nat, +start: Nat, +step: Nat, +length: Nat, +o: Nat, +k: Nat, +r: Nat,
  +positive: {Nat.is_lt(0n, step) == True{} : Bool}, +k_below: {Nat.is_lt(k, length) == True{} : Bool},
  +i_below: {Nat.is_lt(Nat.add(start, Nat.mul(k, step)), extent) == True{} : Bool}, +r_below: {Nat.is_lt(r, inner) == True{} : Bool}) ->
  {Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, Laws.position(extent, inner, o, Nat.add(start, Nat.mul(k, step)), r)) ==
    Some{Laws.position(length, inner, o, k, r)} : Maybe<&2, Nat>}:
  +i = Nat.add(start, Nat.mul(k, step))
  +row = Nat.add(Nat.mul(o, extent), i)
  Equal.trans(Maybe<&2, Nat>, Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, Laws.position(extent, inner, o, i, r)),
    picked_at(outer, extent, inner, start, step, length, row, r, i), Some{Laws.position(length, inner, o, k, r)},
    target_split(outer, extent, inner, start, step, length, o, i, r, i_below, r_below),
    %Equal.sym(Nat, Nat.div(Nat.sub(i, start), step), k,
      Equal.trans(Nat, Nat.div(Nat.sub(i, start), step), Nat.div(Nat.mul(k, step), step), k,
        Equal.cong(Nat, Nat, x => Nat.div(x, step), Nat.sub(i, start), Nat.mul(k, step), sub_cancel(start, Nat.mul(k, step))),
        div_step(k, step, positive))) :
      {picked_k(outer, extent, inner, start, step, length, row, r, i, _) == Some{Laws.position(length, inner, o, k, r)} : Maybe<&2, Nat>}
    %Equal.sym(Bool, Slice.hits(start, step, length, i, k), True{}, hits_true(start, step, length, k, k_below)) :
      {Slice.picked(Slice.Plan{outer, extent, inner, start, step, length}, row, r, i, k, _) == Some{Laws.position(length, inner, o, k, r)} :
        Maybe<&2, Nat>}
    %Equal.sym(Nat, Nat.div(row, extent), o, BatchedProof.div_exact(o, extent, i, i_below)) :
      {Some{Nat.add(Nat.mul(Nat.add(Nat.mul(_, length), k), inner), r)} == Some{Laws.position(length, inner, o, k, r)} : Maybe<&2, Nat>}
    {==})

def Laws.slice_scatter_hit(depth, splits, outer, extent, inner, start, step, length, count, base, dg, G, g, same, valid, path, o, k, r,
  positive, k_below, i_below, r_below, at, inside):
  Laws.Holds{
    Equal.trans(Maybe<&2, F32>,
      Model.observe(F32, depth, Slice.scatter_build(depth, splits, Slice.Plan{outer, extent, inner, start, step, length}, count, base, g), path),
      Some{Pair.fst(F32, Array<F32> & Array<F32>, Slice.scattered((G, empty()),
        Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, Nat.add(Model.offset(depth, path), base))))},
      Model.observe(F32, dg, G, Refinement.route(dg, U32.from_nat(Laws.position(length, inner, o, k, r)))),
      {\{Parallel.Fact: q => q} :
        Parallel.Fact<{Model.observe(F32, depth, Slice.scatter_build(depth, splits, Slice.Plan{outer, extent, inner, start, step, length}, count, base, g), path) ==
          Some{Pair.fst(F32, Array<F32> & Array<F32>, Slice.scattered((G, empty()),
            Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, Nat.add(Model.offset(depth, path), base))))} : Maybe<&2, F32>}> ->
        {Model.observe(F32, depth, Slice.scatter_build(depth, splits, Slice.Plan{outer, extent, inner, start, step, length}, count, base, g), path) ==
          Some{Pair.fst(F32, Array<F32> & Array<F32>, Slice.scattered((G, empty()),
            Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, Nat.add(Model.offset(depth, path), base))))} : Maybe<&2, F32>}}(
        scatter_slot(depth, splits, Slice.Plan{outer, extent, inner, start, step, length}, count, base, dg, G, g, same, valid, path, inside)),
      %Equal.sym(Maybe<&2, Nat>, Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, Nat.add(Model.offset(depth, path), base)),
        Some{Laws.position(length, inner, o, k, r)},
        Equal.trans(Maybe<&2, Nat>, Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, Nat.add(Model.offset(depth, path), base)),
          Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, Laws.position(extent, inner, o, Nat.add(start, Nat.mul(k, step)), r)),
          Some{Laws.position(length, inner, o, k, r)},
          Equal.cong(Nat, Maybe<&2, Nat>, x => Slice.target(Slice.Plan{outer, extent, inner, start, step, length}, x),
            Nat.add(Model.offset(depth, path), base), Laws.position(extent, inner, o, Nat.add(start, Nat.mul(k, step)), r), at),
          target_hit(outer, extent, inner, start, step, length, o, k, r, positive, k_below, i_below, r_below))) :
        {Some{Pair.fst(F32, Array<F32> & Array<F32>, Slice.scattered((G, empty()), _))} ==
          Model.observe(F32, dg, G, Refinement.route(dg, U32.from_nat(Laws.position(length, inner, o, k, r)))) : Maybe<&2, F32>}
      {\{Parallel.Fact: q => q} :
        Parallel.Fact<{Some{Pair.fst(F32, Array<F32> & Array<F32>, Slice.take(empty(), Array.get(F32, G, U32.from_nat(Laws.position(length, inner, o, k, r)))))} ==
          Model.observe(F32, dg, G, Refinement.route(dg, U32.from_nat(Laws.position(length, inner, o, k, r)))) : Maybe<&2, F32>}> ->
        {Some{Pair.fst(F32, Array<F32> & Array<F32>, Slice.take(empty(), Array.get(F32, G, U32.from_nat(Laws.position(length, inner, o, k, r)))))} ==
          Model.observe(F32, dg, G, Refinement.route(dg, U32.from_nat(Laws.position(length, inner, o, k, r)))) : Maybe<&2, F32>}}(
        read_word(dg, G, empty(), Laws.position(length, inner, o, k, r), valid)))}
