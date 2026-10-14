import Base
import ./model.bend as Model
import ./array.bend as ArrayCore
import ./refinement.bend as Refinement
import ./certificates.bend as Certificates
import ./clone.bend as Clone
import ./main.bend as Storage
import ./LAWS.bend as Laws
import ./frame.bend as Frame
import ./route.bend as Route
import ./arith.bend as Arith
import ./size.bend as Size

def finish_value(+d: Nat, n: Nat, -original: Array<F32>, -i: U32, result: Array<F32> & F32,
  -exact: {result == Array.get(F32, original, i) : Array<F32> & F32},
  -balanced: Model.Balanced(F32, d, original), -valid: {Storage.bounds(d, n) == True{} : Bool}) ->
  {Laws.outcome(Result<&2, &2, Storage.Error, F32>, Storage.finish_read(d, n, original, i, result, exact, balanced, valid)) ==
    Done{Refinement.second(F32, result)} : Result<&2, &2, Storage.Error, F32>}:
  (array, value) = result
  {==}

def read_index_hit(+d: Nat, +n: Nat, array: Array<F32>, +i: U32, matched: Bool,
  -decision: {matched == Storage.index_matches(d, n, i) : Bool},
  -balanced: Model.Balanced(F32, d, array), -valid: {Storage.bounds(d, n) == True{} : Bool},
  yes: {True{} == matched : Bool}) ->
  {Laws.outcome(Result<&2, &2, Storage.Error, F32>, Storage.read_index(d, n, array, i, matched, decision, balanced, valid)) ==
    Done{Refinement.second(F32, Array.get(F32, array, i))} : Result<&2, &2, Storage.Error, F32>}:
  match matched:
    case True{}:
      finish_value(d, n, array, i, Array.get(F32, array, i), {==}, balanced, valid)
    case False{}:
      Empty.absurd({Laws.outcome(Result<&2, &2, Storage.Error, F32>, Storage.read_index(d, n, array, i, False{}, decision, balanced, valid)) ==
        Done{Refinement.second(F32, Array.get(F32, array, i))} : Result<&2, &2, Storage.Error, F32>}, Arith.clash_true(yes))

def get_checked_hit(+d: Nat, +n: Nat, array: Array<F32>, +i: U32, allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(i), n) : Bool},
  -balanced: Model.Balanced(F32, d, array), -valid: {Storage.bounds(d, n) == True{} : Bool},
  yes: {True{} == allowed : Bool}, +fits: {Storage.bounds(d, n) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(i), n) == True{} : Bool}) ->
  {Laws.outcome(Result<&2, &2, Storage.Error, F32>, Storage.get_checked(d, n, array, i, allowed, decision, balanced, valid)) ==
    Done{Refinement.second(F32, Array.get(F32, array, i))} : Result<&2, &2, Storage.Error, F32>}:
  match allowed:
    case True{}:
      read_index_hit(d, n, array, i, Storage.index_matches(d, n, i), {==}, balanced, valid, Route.index_complete(d, n, i, fits, inside))
    case False{}:
      Empty.absurd({Laws.outcome(Result<&2, &2, Storage.Error, F32>, Storage.get_checked(d, n, array, i, False{}, decision, balanced, valid)) ==
        Done{Refinement.second(F32, Array.get(F32, array, i))} : Result<&2, &2, Storage.Error, F32>}, Arith.clash_true(yes))

def read_hit(+d: Nat, +n: Nat, array: Array<F32>, -balanced: Model.Balanced(F32, d, array),
  -valid: {Storage.bounds(d, n) == True{} : Bool}, +fits: {Storage.bounds(d, n) == True{} : Bool},
  +i: U32, +inside: {Nat.is_lt(U32.to_nat(i), n) == True{} : Bool}) ->
  {Laws.read(Storage.Owned{d, n, array, balanced, valid}, i) ==
    Done{Refinement.second(F32, Array.get(F32, array, i))} : Result<&2, &2, Storage.Error, F32>}:
  get_checked_hit(d, n, array, i, Nat.is_lt(U32.to_nat(i), n), {==}, balanced, valid,
    Equal.sym(Bool, Nat.is_lt(U32.to_nat(i), n), True{}, inside), fits, inside)

def get_checked_miss(+d: Nat, +n: Nat, -array: Array<F32>, +i: U32, allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(i), n) : Bool},
  -balanced: Model.Balanced(F32, d, array), -valid: {Storage.bounds(d, n) == True{} : Bool},
  no: {False{} == allowed : Bool}) ->
  {Laws.outcome(Result<&2, &2, Storage.Error, F32>, Storage.get_checked(d, n, array, i, allowed, decision, balanced, valid)) ==
    Fail{Storage.OutOfBounds{}} : Result<&2, &2, Storage.Error, F32>}:
  match allowed:
    case False{}:
      {==}
    case True{}:
      Empty.absurd({Laws.outcome(Result<&2, &2, Storage.Error, F32>, Storage.get_checked(d, n, array, i, True{}, decision, balanced, valid)) ==
        Fail{Storage.OutOfBounds{}} : Result<&2, &2, Storage.Error, F32>}, Arith.clash(no))

def set_checked_miss(+d: Nat, +n: Nat, -array: Array<F32>, +i: U32, -value: F32, allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(i), n) : Bool},
  -balanced: Model.Balanced(F32, d, array), -valid: {Storage.bounds(d, n) == True{} : Bool},
  no: {False{} == allowed : Bool}) ->
  {Storage.set_checked(d, n, array, i, value, allowed, decision, balanced, valid) ==
    (Storage.Owned{d, n, array, balanced, valid}, Fail{Storage.OutOfBounds{}}) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}:
  match allowed:
    case False{}:
      {==}
    case True{}:
      Empty.absurd({Storage.set_checked(d, n, array, i, value, True{}, decision, balanced, valid) ==
        (Storage.Owned{d, n, array, balanced, valid}, Fail{Storage.OutOfBounds{}}) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}, Arith.clash(no))

def write_index_hit(+d: Nat, +n: Nat, -array: Array<F32>, +i: U32, +value: F32, matched: Bool,
  -decision: {matched == Storage.index_matches(d, n, i) : Bool},
  -balanced: Model.Balanced(F32, d, array), -valid: {Storage.bounds(d, n) == True{} : Bool},
  yes: {True{} == matched : Bool}) ->
  {Storage.write_index(d, n, array, i, value, matched, decision, balanced, valid) ==
    (Storage.Owned{d, n, Array.set(F32, array, i, value),
      Certificates.set_result(F32, d, array, i, value, Array.set(F32, array, i, value), {==}, balanced), valid}, Done{Unit{}})
    : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}:
  match matched:
    case True{}:
      {==}
    case False{}:
      Empty.absurd({Storage.write_index(d, n, array, i, value, False{}, decision, balanced, valid) ==
        (Storage.Owned{d, n, Array.set(F32, array, i, value),
          Certificates.set_result(F32, d, array, i, value, Array.set(F32, array, i, value), {==}, balanced), valid}, Done{Unit{}})
        : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}, Arith.clash_true(yes))

def set_checked_hit(+d: Nat, +n: Nat, -array: Array<F32>, +i: U32, +value: F32, allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(i), n) : Bool},
  -balanced: Model.Balanced(F32, d, array), -valid: {Storage.bounds(d, n) == True{} : Bool},
  yes: {True{} == allowed : Bool}, +fits: {Storage.bounds(d, n) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(i), n) == True{} : Bool}) ->
  {Storage.set_checked(d, n, array, i, value, allowed, decision, balanced, valid) ==
    (Storage.Owned{d, n, Array.set(F32, array, i, value),
      Certificates.set_result(F32, d, array, i, value, Array.set(F32, array, i, value), {==}, balanced), valid}, Done{Unit{}})
    : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}:
  match allowed:
    case True{}:
      write_index_hit(d, n, array, i, value, Storage.index_matches(d, n, i), {==}, balanced, valid, Route.index_complete(d, n, i, fits, inside))
    case False{}:
      Empty.absurd({Storage.set_checked(d, n, array, i, value, False{}, decision, balanced, valid) ==
        (Storage.Owned{d, n, Array.set(F32, array, i, value),
          Certificates.set_result(F32, d, array, i, value, Array.set(F32, array, i, value), {==}, balanced), valid}, Done{Unit{}})
        : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}, Arith.clash_true(yes))

def finish_owner(+d: Nat, +n: Nat, -original: Array<F32>, +i: U32, result: Array<F32> & F32,
  -exact: {result == Array.get(F32, original, i) : Array<F32> & F32},
  -balanced: Model.Balanced(F32, d, original), -valid: {Storage.bounds(d, n) == True{} : Bool}) ->
  {Storage.finish_read(d, n, original, i, result, exact, balanced, valid) ==
    (Storage.Owned{d, n, Refinement.first(F32, result), Certificates.get_result(F32, d, original, i, result, exact, balanced), valid},
      Done{Refinement.second(F32, result)}) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}:
  (array, value) = result
  {==}

def read_index_keep(+d: Nat, +n: Nat, array: Array<F32>, +i: U32, matched: Bool,
  -decision: {matched == Storage.index_matches(d, n, i) : Bool},
  -balanced: Model.Balanced(F32, d, array), -valid: {Storage.bounds(d, n) == True{} : Bool},
  yes: {True{} == matched : Bool}) ->
  {Storage.read_index(d, n, array, i, matched, decision, balanced, valid) ==
    (Storage.Owned{d, n, Refinement.first(F32, Array.get(F32, array, i)),
      Certificates.get_result(F32, d, array, i, Array.get(F32, array, i), {==}, balanced), valid},
      Done{Refinement.second(F32, Array.get(F32, array, i))}) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}:
  match matched:
    case True{}:
      finish_owner(d, n, array, i, Array.get(F32, array, i), {==}, balanced, valid)
    case False{}:
      Empty.absurd({Storage.read_index(d, n, array, i, False{}, decision, balanced, valid) ==
        (Storage.Owned{d, n, Refinement.first(F32, Array.get(F32, array, i)),
          Certificates.get_result(F32, d, array, i, Array.get(F32, array, i), {==}, balanced), valid},
          Done{Refinement.second(F32, Array.get(F32, array, i))}) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}, Arith.clash_true(yes))

def get_checked_keep(+d: Nat, +n: Nat, array: Array<F32>, +i: U32, allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(i), n) : Bool},
  -balanced: Model.Balanced(F32, d, array), -valid: {Storage.bounds(d, n) == True{} : Bool},
  yes: {True{} == allowed : Bool}, +fits: {Storage.bounds(d, n) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(i), n) == True{} : Bool}) ->
  {Storage.get_checked(d, n, array, i, allowed, decision, balanced, valid) ==
    (Storage.Owned{d, n, Refinement.first(F32, Array.get(F32, array, i)),
      Certificates.get_result(F32, d, array, i, Array.get(F32, array, i), {==}, balanced), valid},
      Done{Refinement.second(F32, Array.get(F32, array, i))}) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}:
  match allowed:
    case True{}:
      read_index_keep(d, n, array, i, Storage.index_matches(d, n, i), {==}, balanced, valid, Route.index_complete(d, n, i, fits, inside))
    case False{}:
      Empty.absurd({Storage.get_checked(d, n, array, i, False{}, decision, balanced, valid) ==
        (Storage.Owned{d, n, Refinement.first(F32, Array.get(F32, array, i)),
          Certificates.get_result(F32, d, array, i, Array.get(F32, array, i), {==}, balanced), valid},
          Done{Refinement.second(F32, Array.get(F32, array, i))}) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}, Arith.clash_true(yes))

def get_checked_stay(+d: Nat, +n: Nat, -array: Array<F32>, +i: U32, allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(i), n) : Bool},
  -balanced: Model.Balanced(F32, d, array), -valid: {Storage.bounds(d, n) == True{} : Bool},
  no: {False{} == allowed : Bool}) ->
  {Storage.get_checked(d, n, array, i, allowed, decision, balanced, valid) ==
    (Storage.Owned{d, n, array, balanced, valid}, Fail{Storage.OutOfBounds{}}) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}:
  match allowed:
    case False{}:
      {==}
    case True{}:
      Empty.absurd({Storage.get_checked(d, n, array, i, True{}, decision, balanced, valid) ==
        (Storage.Owned{d, n, array, balanced, valid}, Fail{Storage.OutOfBounds{}}) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}, Arith.clash(no))

def finish_pair(+d: Nat, +n: Nat, -original: Array<F32>, pair: Array<F32> & Array<F32>,
  -exact: {pair == Array.clone(F32, original) : Array<F32> & Array<F32>},
  -balanced: Model.Balanced(F32, d, original), -valid: {Storage.bounds(d, n) == True{} : Bool}) ->
  {Storage.finish_clone(d, n, original, pair, exact, balanced, valid) ==
    (Storage.Owned{d, n, Pair.fst(Array<F32>, Array<F32>, pair), Certificates.clone_first(F32, d, original, pair, exact, balanced), valid},
      Storage.Owned{d, n, Pair.snd(Array<F32>, Array<F32>, pair), Certificates.clone_second(F32, d, original, pair, exact, balanced), valid})
    : Storage.Storage & Storage.Storage}:
  (first, second) = pair
  {==}

def to_result(found: Maybe<&2, F32>) -> Result<&2, &2, Storage.Error, F32>:
  match found:
    case None{}:
      Fail{Storage.InvalidIndex{}}
    case Some{value}:
      Done{value}

def le_refl(+k: Nat) -> {Nat.is_le(k, k) == True{} : Bool}:
  match k:
    case 0n:
      {==}
    case 1n+p:
      le_refl(p)

def le_add_left(+f: Nat, +k: Nat) -> {Nat.is_le(k, Nat.add(f, k)) == True{} : Bool}:
  match f:
    case 0n:
      le_refl(k)
    case 1n+q:
      Arith.le_succ(k, Nat.add(q, k), le_add_left(q, k))

def le_add_right(+c: Nat, +x: Nat) -> {Nat.is_le(c, Nat.add(c, x)) == True{} : Bool}:
  match c:
    case 0n:
      Arith.le_zero(x)
    case 1n+r:
      le_add_right(r, x)

def lt_add_left(+c: Nat, +x: Nat, +y: Nat, h: {Nat.is_lt(x, y) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(c, x), Nat.add(c, y)) == True{} : Bool}:
  match c:
    case 0n:
      h
    case 1n+r:
      lt_add_left(r, x, y, h)

def eq_add_cancel(+c: Nat, +x: Nat, +y: Nat) -> {Nat.is_eq(x, y) == Nat.is_eq(Nat.add(c, x), Nat.add(c, y)) : Bool}:
  match c:
    case 0n:
      {==}
    case 1n+r:
      eq_add_cancel(r, x, y)

def ne_low(+x: Nat, +c: Nat, +y: Nat, h: {Nat.is_lt(x, c) == True{} : Bool}) -> {False{} == Nat.is_eq(x, Nat.add(c, y)) : Bool}:
  match x:
    case 0n:
      match c:
        case 0n:
          Empty.absurd({False{} == Nat.is_eq(0n, y) : Bool}, Arith.clash(h))
        case 1n+q:
          {==}
    case 1n+p:
      match c:
        case 0n:
          Empty.absurd({False{} == Nat.is_eq(1n+p, y) : Bool}, Arith.clash(h))
        case 1n+q:
          ne_low(p, q, y, h)

def ne_high(+x: Nat, +c: Nat, +y: Nat, h: {Nat.is_lt(x, c) == True{} : Bool}) -> {False{} == Nat.is_eq(Nat.add(c, y), x) : Bool}:
  match x:
    case 0n:
      match c:
        case 0n:
          Empty.absurd({False{} == Nat.is_eq(y, 0n) : Bool}, Arith.clash(h))
        case 1n+q:
          {==}
    case 1n+p:
      match c:
        case 0n:
          Empty.absurd({False{} == Nat.is_eq(y, 1n+p) : Bool}, Arith.clash(h))
        case 1n+q:
          ne_high(p, q, y, h)

def offset_lt(+d: Nat, +path: Model.Path(d)) -> {Nat.is_lt(Model.offset(d, path), Model.capacity(d)) == True{} : Bool}:
  match d:
    case 0n:
      {==}
    case 1n+p:
      Model.Step{+b, +rest} = path
      match b:
        case False{}:
          Arith.lt_le_trans(Model.offset(p, rest), Model.capacity(p), Nat.add(Model.capacity(p), Model.capacity(p)),
            offset_lt(p, rest), le_add_right(Model.capacity(p), Model.capacity(p)))
        case True{}:
          lt_add_left(Model.capacity(p), Model.offset(p, rest), Model.capacity(p), offset_lt(p, rest))

def same_eq(+d: Nat, +p: Model.Path(d), +q: Model.Path(d)) ->
  {Nat.is_eq(Model.offset(d, p), Model.offset(d, q)) == Model.same(d, p, q) : Bool}:
  match d:
    case 0n:
      {==}
    case 1n+r:
      Model.Step{+pb, +pr} = p
      match pb:
        case False{}:
          Model.Step{+qb, +qr} = q
          match qb:
            case False{}:
              same_eq(r, pr, qr)
            case True{}:
              Equal.sym(Bool, False{}, Nat.is_eq(Model.offset(r, pr), Nat.add(Model.capacity(r), Model.offset(r, qr))),
                ne_low(Model.offset(r, pr), Model.capacity(r), Model.offset(r, qr), offset_lt(r, pr)))
        case True{}:
          Model.Step{+qb, +qr} = q
          match qb:
            case False{}:
              Equal.sym(Bool, False{}, Nat.is_eq(Nat.add(Model.capacity(r), Model.offset(r, pr)), Model.offset(r, qr)),
                ne_high(Model.offset(r, qr), Model.capacity(r), Model.offset(r, pr), offset_lt(r, qr)))
            case True{}:
              %eq_add_cancel(Model.capacity(r), Model.offset(r, pr), Model.offset(r, qr)) : {_ == Model.same(r, pr, qr) : Bool}
              same_eq(r, pr, qr)

def depth_of(+d: Nat, +n: Nat, +fits: {Storage.bounds(d, n) == True{} : Bool}) -> {Nat.is_le(d, 20n) == True{} : Bool}:
  Arith.and_left(Nat.is_le(d, 20n), Bool.and(Nat.is_lt(0n, n), Nat.is_le(n, Model.capacity(d))), fits)

def slot_of(+d: Nat, +n: Nat, +i: U32, +fits: {Storage.bounds(d, n) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(i), n) == True{} : Bool}) -> {Nat.is_lt(U32.to_nat(i), Model.capacity(d)) == True{} : Bool}:
  +rest = Arith.and_right(Nat.is_le(d, 20n), Bool.and(Nat.is_lt(0n, n), Nat.is_le(n, Model.capacity(d))), fits)
  Arith.lt_le_trans(U32.to_nat(i), n, Model.capacity(d), inside,
    Arith.and_right(Nat.is_lt(0n, n), Nat.is_le(n, Model.capacity(d)), rest))

def route_same(+d: Nat, +n: Nat, +i: U32, +j: U32, +fits: {Storage.bounds(d, n) == True{} : Bool},
  +inside_i: {Nat.is_lt(U32.to_nat(i), n) == True{} : Bool}, +inside_j: {Nat.is_lt(U32.to_nat(j), n) == True{} : Bool}) ->
  {Nat.is_eq(U32.to_nat(i), U32.to_nat(j)) == Model.same(d, Refinement.route(d, i), Refinement.route(d, j)) : Bool}:
  %same_eq(d, Refinement.route(d, i), Refinement.route(d, j)) : {Nat.is_eq(U32.to_nat(i), U32.to_nat(j)) == _ : Bool}
  %Route.route_offset(d, i, depth_of(d, n, fits), slot_of(d, n, i, fits, inside_i)) :
    {Nat.is_eq(U32.to_nat(i), U32.to_nat(j)) == Nat.is_eq(_, Model.offset(d, Refinement.route(d, j))) : Bool}
  %Route.route_offset(d, j, depth_of(d, n, fits), slot_of(d, n, j, fits, inside_j)) :
    {Nat.is_eq(U32.to_nat(i), U32.to_nat(j)) == Nat.is_eq(U32.to_nat(i), _) : Bool}
  {==}

type Effect is Data:
  Wrote{index: U32, value: F32}
  Kept{index: U32}
  Left{}
  Right{}

def grow(array: Array<F32>, +effect: Effect) -> Array<F32>:
  match effect:
    case Wrote{+i, +w}:
      Array.set(F32, array, i, w)
    case Kept{+i}:
      Refinement.first(F32, Array.get(F32, array, i))
    case Left{}:
      Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, array))
    case Right{}:
      Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, array))

def arr(+history: List<&2, Effect>, +d: Nat, +fill: F32) -> Array<F32>:
  match history:
    case Nil{}:
      Array.new(F32, d, fill)
    case +effect <> +past:
      grow(arr(past, d, fill), effect)

def current(+history: List<&2, Effect>, +j: U32, +fill: F32) -> F32:
  match history:
    case Nil{}:
      fill
    case +effect <> +past:
      match effect:
        case Wrote{+i, +w}:
          Bool.pick(F32, Nat.is_eq(U32.to_nat(i), U32.to_nat(j)), w, current(past, j, fill))
        case Kept{i}:
          current(past, j, fill)
        case Left{}:
          current(past, j, fill)
        case Right{}:
          current(past, j, fill)

def Valid(+n: Nat, +history: List<&2, Effect>) -> Type:
  match history:
    case Nil{}:
      Unit
    case +effect <> +past:
      match effect:
        case Wrote{+i, w}:
          {Nat.is_lt(U32.to_nat(i), n) == True{} : Bool} & Valid(n, past)
        case Kept{i}:
          Valid(n, past)
        case Left{}:
          Valid(n, past)
        case Right{}:
          Valid(n, past)

def grow_balanced(+d: Nat, array: Array<F32>, +effect: Effect, valid: Model.Balanced(F32, d, array)) ->
  Model.Balanced(F32, d, grow(array, effect)):
  match effect:
    case Wrote{+i, +w}:
      Refinement.law_set_balanced(F32, d, array, i, w, valid)
    case Kept{+i}:
      Refinement.law_get_balanced(F32, d, array, i, valid)
    case Left{}:
      Clone.first_balanced(F32, d, array, Clone.certify(F32, d, array, valid))
    case Right{}:
      Clone.second_balanced(F32, d, array, Clone.certify(F32, d, array, valid))

def bal(+history: List<&2, Effect>, +d: Nat, +fill: F32) -> Model.Balanced(F32, d, arr(history, d, fill)):
  match history:
    case Nil{}:
      ArrayCore.new_balanced(F32, d, fill)
    case +effect <> +past:
      grow_balanced(d, arr(past, d, fill), effect, bal(past, d, fill))

def observe_new(+d: Nat, +value: F32, +path: Model.Path(d)) ->
  {Some{value} == Model.observe(F32, d, Array.new(F32, d, value), path) : Maybe<&2, F32>}:
  match d:
    case 0n:
      {==}
    case 1n+p:
      Model.Step{+bit, +rest} = path
      match bit:
        case False{}:
          observe_new(p, value, rest)
        case True{}:
          observe_new(p, value, rest)

def select_pick(+b: Bool, +w: F32, +c: F32) -> {Some{Bool.pick(F32, b, w, c)} == Model.select(F32, b, w, Some{c}) : Maybe<&2, F32>}:
  match b:
    case False{}:
      {==}
    case True{}:
      {==}

# Every in-bounds index has its own route, so the array after the effects
# answers each read with the last value written there, or the fill.
def model(+history: List<&2, Effect>, +d: Nat, +n: Nat, +fill: F32, +j: U32,
  +fits: {Storage.bounds(d, n) == True{} : Bool}, +inside: {Nat.is_lt(U32.to_nat(j), n) == True{} : Bool},
  valid: Valid(n, history)) ->
  {Some{current(history, j, fill)} == Model.observe(F32, d, arr(history, d, fill), Refinement.route(d, j)) : Maybe<&2, F32>}:
  match history:
    case Nil{}:
      observe_new(d, fill, Refinement.route(d, j))
    case +effect <> +past:
      match effect:
        case Wrote{+i, +w}:
          (inside_i, rest) = valid
          %Equal.sym(Maybe<&2, F32>,
            Model.observe(F32, d, Array.set(F32, arr(past, d, fill), i, w), Refinement.route(d, j)),
            Model.select(F32, Model.same(d, Refinement.route(d, i), Refinement.route(d, j)), w,
              Model.observe(F32, d, arr(past, d, fill), Refinement.route(d, j))),
            Frame.primitive_update_frame(F32, d, arr(past, d, fill), i, Refinement.route(d, j), w, bal(past, d, fill))) :
            {Some{current(Wrote{i, w} <> past, j, fill)} == _ : Maybe<&2, F32>}
          %route_same(d, n, i, j, fits, inside_i, inside) :
            {Some{current(Wrote{i, w} <> past, j, fill)} ==
              Model.select(F32, _, w, Model.observe(F32, d, arr(past, d, fill), Refinement.route(d, j))) : Maybe<&2, F32>}
          %model(past, d, n, fill, j, fits, inside, rest) :
            {Some{current(Wrote{i, w} <> past, j, fill)} ==
              Model.select(F32, Nat.is_eq(U32.to_nat(i), U32.to_nat(j)), w, _) : Maybe<&2, F32>}
          select_pick(Nat.is_eq(U32.to_nat(i), U32.to_nat(j)), w, current(past, j, fill))
        case Kept{+i}:
          %Equal.sym(Array<F32>, Refinement.first(F32, Array.get(F32, arr(past, d, fill), i)), arr(past, d, fill),
            Refinement.law_get_retains(F32, d, arr(past, d, fill), i, bal(past, d, fill))) :
            {Some{current(past, j, fill)} == Model.observe(F32, d, _, Refinement.route(d, j)) : Maybe<&2, F32>}
          model(past, d, n, fill, j, fits, inside, valid)
        case Left{}:
          %Equal.sym(Array<F32>, Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, arr(past, d, fill))), arr(past, d, fill),
            Clone.first_equal(F32, d, arr(past, d, fill), Clone.certify(F32, d, arr(past, d, fill), bal(past, d, fill)))) :
            {Some{current(past, j, fill)} == Model.observe(F32, d, _, Refinement.route(d, j)) : Maybe<&2, F32>}
          model(past, d, n, fill, j, fits, inside, valid)
        case Right{}:
          %Equal.sym(Array<F32>, Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, arr(past, d, fill))), arr(past, d, fill),
            Clone.second_equal(F32, d, arr(past, d, fill), Clone.certify(F32, d, arr(past, d, fill), bal(past, d, fill)))) :
            {Some{current(past, j, fill)} == Model.observe(F32, d, _, Refinement.route(d, j)) : Maybe<&2, F32>}
          model(past, d, n, fill, j, fits, inside, valid)

def hit_of(+n: Nat, +ops: List<&2, Laws.Op>) -> Bool:
  match ops:
    case Nil{}:
      True{}
    case +op <> +later:
      match op:
        case Laws.Write{+i, w}:
          Nat.is_lt(U32.to_nat(i), n)
        case Laws.Read{+i}:
          Nat.is_lt(U32.to_nat(i), n)
        case Laws.First{}:
          True{}
        case Laws.Second{}:
          True{}
        case Laws.Length{}:
          True{}

def replay_read(+rest: List<&2, Laws.Op>, +d: Nat, +n: Nat, +fill: F32, +history: List<&2, Effect>, +j: U32,
  hit: Bool, +dhit: {hit == hit_of(n, rest) : Bool},
  -b: Model.Balanced(F32, d, arr(history, d, fill)), -f: {Storage.bounds(d, n) == True{} : Bool},
  +fits: {Storage.bounds(d, n) == True{} : Bool}, +inside: {Nat.is_lt(U32.to_nat(j), n) == True{} : Bool},
  valid: Valid(n, history)) ->
  {Laws.read(Laws.replay(rest, Storage.Owned{d, n, arr(history, d, fill), b, f}), j) == Done{Laws.expected(rest, n, j, current(history, j, fill))} : Result<&2, &2, Storage.Error, F32>}:
  match rest:
    case Nil{}:
      %Equal.sym(Result<&2, &2, Storage.Error, F32>, Laws.read(Storage.Owned{d, n, arr(history, d, fill), b, f}, j), Done{Refinement.second(F32, Array.get(F32, arr(history, d, fill), j))},
        read_hit(d, n, arr(history, d, fill), b, f, fits, j, inside)) : {_ == Done{current(history, j, fill)} : Result<&2, &2, Storage.Error, F32>}
      Equal.cong(Maybe<&2, F32>, Result<&2, &2, Storage.Error, F32>, to_result, Some{Refinement.second(F32, Array.get(F32, arr(history, d, fill), j))}, Some{current(history, j, fill)},
        Equal.trans(Maybe<&2, F32>, Some{Refinement.second(F32, Array.get(F32, arr(history, d, fill), j))},
          Model.observe(F32, d, arr(history, d, fill), Refinement.route(d, j)), Some{current(history, j, fill)},
          Refinement.law_get_refines(F32, d, arr(history, d, fill), j, bal(history, d, fill)),
          Equal.sym(Maybe<&2, F32>, Some{current(history, j, fill)}, Model.observe(F32, d, arr(history, d, fill), Refinement.route(d, j)),
            model(history, d, n, fill, j, fits, inside, valid))))
    case +op <> +later:
      match op:
        case Laws.Write{+i, +w}:
          match hit:
            case True{}:
              +inside_i = Equal.sym(Bool, True{}, Nat.is_lt(U32.to_nat(i), n), dhit)
              %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, Unit>, Storage.set_checked(d, n, arr(history, d, fill), i, w, Nat.is_lt(U32.to_nat(i), n), {==}, b, f), (Storage.Owned{d, n, Array.set(F32, arr(history, d, fill), i, w), Certificates.set_result(F32, d, arr(history, d, fill), i, w, Array.set(F32, arr(history, d, fill), i, w), {==}, b), f}, Done{Unit{}}), set_checked_hit(d, n, arr(history, d, fill), i, w, Nat.is_lt(U32.to_nat(i), n), {==}, b, f, dhit, fits, inside_i)) :
                {Laws.read(Laws.replay(later, Laws.owner(Result<&2, &2, Storage.Error, Unit>, _)), j) == Done{Laws.expected(later, n, j, Laws.latest(n, Laws.Write{i, w}, j, current(history, j, fill)))} : Result<&2, &2, Storage.Error, F32>}
              %dhit : {Laws.read(Laws.replay(later, Storage.Owned{d, n, Array.set(F32, arr(history, d, fill), i, w), Certificates.set_result(F32, d, arr(history, d, fill), i, w, Array.set(F32, arr(history, d, fill), i, w), {==}, b), f}), j) ==
                Done{Laws.expected(later, n, j, Bool.pick(F32, Bool.and(_, Nat.is_eq(U32.to_nat(i), U32.to_nat(j))), w, current(history, j, fill)))} : Result<&2, &2, Storage.Error, F32>}
              replay_read(later, d, n, fill, Wrote{i, w} <> history, j, hit_of(n, later), {==}, Certificates.set_result(F32, d, arr(history, d, fill), i, w, Array.set(F32, arr(history, d, fill), i, w), {==}, b), f, fits, inside, (inside_i, valid))
            case False{}:
              %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, Unit>, Storage.set_checked(d, n, arr(history, d, fill), i, w, Nat.is_lt(U32.to_nat(i), n), {==}, b, f), (Storage.Owned{d, n, arr(history, d, fill), b, f}, Fail{Storage.OutOfBounds{}}), set_checked_miss(d, n, arr(history, d, fill), i, w, Nat.is_lt(U32.to_nat(i), n), {==}, b, f, dhit)) :
                {Laws.read(Laws.replay(later, Laws.owner(Result<&2, &2, Storage.Error, Unit>, _)), j) == Done{Laws.expected(later, n, j, Laws.latest(n, Laws.Write{i, w}, j, current(history, j, fill)))} : Result<&2, &2, Storage.Error, F32>}
              %dhit : {Laws.read(Laws.replay(later, Storage.Owned{d, n, arr(history, d, fill), b, f}), j) == Done{Laws.expected(later, n, j, Bool.pick(F32, Bool.and(_, Nat.is_eq(U32.to_nat(i), U32.to_nat(j))), w, current(history, j, fill)))} : Result<&2, &2, Storage.Error, F32>}
              replay_read(later, d, n, fill, history, j, hit_of(n, later), {==}, b, f, fits, inside, valid)
        case Laws.Read{+i}:
          match hit:
            case True{}:
              +inside_i = Equal.sym(Bool, True{}, Nat.is_lt(U32.to_nat(i), n), dhit)
              %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, F32>, Storage.get_checked(d, n, arr(history, d, fill), i, Nat.is_lt(U32.to_nat(i), n), {==}, b, f), (Storage.Owned{d, n, Refinement.first(F32, Array.get(F32, arr(history, d, fill), i)), Certificates.get_result(F32, d, arr(history, d, fill), i, Array.get(F32, arr(history, d, fill), i), {==}, b), f}, Done{Refinement.second(F32, Array.get(F32, arr(history, d, fill), i))}), get_checked_keep(d, n, arr(history, d, fill), i, Nat.is_lt(U32.to_nat(i), n), {==}, b, f, dhit, fits, inside_i)) :
                {Laws.read(Laws.replay(later, Laws.owner(Result<&2, &2, Storage.Error, F32>, _)), j) == Done{Laws.expected(later, n, j, current(history, j, fill))} : Result<&2, &2, Storage.Error, F32>}
              replay_read(later, d, n, fill, Kept{i} <> history, j, hit_of(n, later), {==}, Certificates.get_result(F32, d, arr(history, d, fill), i, Array.get(F32, arr(history, d, fill), i), {==}, b), f, fits, inside, valid)
            case False{}:
              %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, F32>, Storage.get_checked(d, n, arr(history, d, fill), i, Nat.is_lt(U32.to_nat(i), n), {==}, b, f), (Storage.Owned{d, n, arr(history, d, fill), b, f}, Fail{Storage.OutOfBounds{}}), get_checked_stay(d, n, arr(history, d, fill), i, Nat.is_lt(U32.to_nat(i), n), {==}, b, f, dhit)) :
                {Laws.read(Laws.replay(later, Laws.owner(Result<&2, &2, Storage.Error, F32>, _)), j) == Done{Laws.expected(later, n, j, current(history, j, fill))} : Result<&2, &2, Storage.Error, F32>}
              replay_read(later, d, n, fill, history, j, hit_of(n, later), {==}, b, f, fits, inside, valid)
        case Laws.First{}:
          %Equal.sym(Storage.Storage & Storage.Storage, Storage.finish_clone(d, n, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b, f), (Storage.Owned{d, n, Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, arr(history, d, fill))), Certificates.clone_first(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f}, Storage.Owned{d, n, Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, arr(history, d, fill))), Certificates.clone_second(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f}), finish_pair(d, n, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b, f)) :
            {Laws.read(Laws.replay(later, Laws.owner(Storage.Storage, _)), j) == Done{Laws.expected(later, n, j, current(history, j, fill))} : Result<&2, &2, Storage.Error, F32>}
          replay_read(later, d, n, fill, Left{} <> history, j, hit_of(n, later), {==}, Certificates.clone_first(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f, fits, inside, valid)
        case Laws.Second{}:
          %Equal.sym(Storage.Storage & Storage.Storage, Storage.finish_clone(d, n, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b, f), (Storage.Owned{d, n, Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, arr(history, d, fill))), Certificates.clone_first(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f}, Storage.Owned{d, n, Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, arr(history, d, fill))), Certificates.clone_second(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f}), finish_pair(d, n, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b, f)) :
            {Laws.read(Laws.replay(later, Laws.outcome(Storage.Storage, _)), j) == Done{Laws.expected(later, n, j, current(history, j, fill))} : Result<&2, &2, Storage.Error, F32>}
          replay_read(later, d, n, fill, Right{} <> history, j, hit_of(n, later), {==}, Certificates.clone_second(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f, fits, inside, valid)
        case Laws.Length{}:
          replay_read(later, d, n, fill, history, j, hit_of(n, later), {==}, b, f, fits, inside, valid)

def replay_size_owned(+rest: List<&2, Laws.Op>, +d: Nat, +n: Nat, +fill: F32, +history: List<&2, Effect>,
  hit: Bool, +dhit: {hit == hit_of(n, rest) : Bool},
  -b: Model.Balanced(F32, d, arr(history, d, fill)), -f: {Storage.bounds(d, n) == True{} : Bool},
  +fits: {Storage.bounds(d, n) == True{} : Bool}) ->
  {Laws.size(Laws.replay(rest, Storage.Owned{d, n, arr(history, d, fill), b, f})) == n : Nat}:
  match rest:
    case Nil{}:
      {==}
    case +op <> +later:
      match op:
        case Laws.Write{+i, +w}:
          match hit:
            case True{}:
              +inside_i = Equal.sym(Bool, True{}, Nat.is_lt(U32.to_nat(i), n), dhit)
              %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, Unit>, Storage.set_checked(d, n, arr(history, d, fill), i, w, Nat.is_lt(U32.to_nat(i), n), {==}, b, f), (Storage.Owned{d, n, Array.set(F32, arr(history, d, fill), i, w), Certificates.set_result(F32, d, arr(history, d, fill), i, w, Array.set(F32, arr(history, d, fill), i, w), {==}, b), f}, Done{Unit{}}), set_checked_hit(d, n, arr(history, d, fill), i, w, Nat.is_lt(U32.to_nat(i), n), {==}, b, f, dhit, fits, inside_i)) :
                {Laws.size(Laws.replay(later, Laws.owner(Result<&2, &2, Storage.Error, Unit>, _))) == n : Nat}
              replay_size_owned(later, d, n, fill, Wrote{i, w} <> history, hit_of(n, later), {==}, Certificates.set_result(F32, d, arr(history, d, fill), i, w, Array.set(F32, arr(history, d, fill), i, w), {==}, b), f, fits)
            case False{}:
              %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, Unit>, Storage.set_checked(d, n, arr(history, d, fill), i, w, Nat.is_lt(U32.to_nat(i), n), {==}, b, f), (Storage.Owned{d, n, arr(history, d, fill), b, f}, Fail{Storage.OutOfBounds{}}), set_checked_miss(d, n, arr(history, d, fill), i, w, Nat.is_lt(U32.to_nat(i), n), {==}, b, f, dhit)) :
                {Laws.size(Laws.replay(later, Laws.owner(Result<&2, &2, Storage.Error, Unit>, _))) == n : Nat}
              replay_size_owned(later, d, n, fill, history, hit_of(n, later), {==}, b, f, fits)
        case Laws.Read{+i}:
          match hit:
            case True{}:
              +inside_i = Equal.sym(Bool, True{}, Nat.is_lt(U32.to_nat(i), n), dhit)
              %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, F32>, Storage.get_checked(d, n, arr(history, d, fill), i, Nat.is_lt(U32.to_nat(i), n), {==}, b, f), (Storage.Owned{d, n, Refinement.first(F32, Array.get(F32, arr(history, d, fill), i)), Certificates.get_result(F32, d, arr(history, d, fill), i, Array.get(F32, arr(history, d, fill), i), {==}, b), f}, Done{Refinement.second(F32, Array.get(F32, arr(history, d, fill), i))}), get_checked_keep(d, n, arr(history, d, fill), i, Nat.is_lt(U32.to_nat(i), n), {==}, b, f, dhit, fits, inside_i)) :
                {Laws.size(Laws.replay(later, Laws.owner(Result<&2, &2, Storage.Error, F32>, _))) == n : Nat}
              replay_size_owned(later, d, n, fill, Kept{i} <> history, hit_of(n, later), {==}, Certificates.get_result(F32, d, arr(history, d, fill), i, Array.get(F32, arr(history, d, fill), i), {==}, b), f, fits)
            case False{}:
              %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, F32>, Storage.get_checked(d, n, arr(history, d, fill), i, Nat.is_lt(U32.to_nat(i), n), {==}, b, f), (Storage.Owned{d, n, arr(history, d, fill), b, f}, Fail{Storage.OutOfBounds{}}), get_checked_stay(d, n, arr(history, d, fill), i, Nat.is_lt(U32.to_nat(i), n), {==}, b, f, dhit)) :
                {Laws.size(Laws.replay(later, Laws.owner(Result<&2, &2, Storage.Error, F32>, _))) == n : Nat}
              replay_size_owned(later, d, n, fill, history, hit_of(n, later), {==}, b, f, fits)
        case Laws.First{}:
          %Equal.sym(Storage.Storage & Storage.Storage, Storage.finish_clone(d, n, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b, f), (Storage.Owned{d, n, Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, arr(history, d, fill))), Certificates.clone_first(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f}, Storage.Owned{d, n, Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, arr(history, d, fill))), Certificates.clone_second(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f}), finish_pair(d, n, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b, f)) :
            {Laws.size(Laws.replay(later, Laws.owner(Storage.Storage, _))) == n : Nat}
          replay_size_owned(later, d, n, fill, Left{} <> history, hit_of(n, later), {==}, Certificates.clone_first(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f, fits)
        case Laws.Second{}:
          %Equal.sym(Storage.Storage & Storage.Storage, Storage.finish_clone(d, n, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b, f), (Storage.Owned{d, n, Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, arr(history, d, fill))), Certificates.clone_first(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f}, Storage.Owned{d, n, Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, arr(history, d, fill))), Certificates.clone_second(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f}), finish_pair(d, n, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b, f)) :
            {Laws.size(Laws.replay(later, Laws.outcome(Storage.Storage, _))) == n : Nat}
          replay_size_owned(later, d, n, fill, Right{} <> history, hit_of(n, later), {==}, Certificates.clone_second(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f, fits)
        case Laws.Length{}:
          replay_size_owned(later, d, n, fill, history, hit_of(n, later), {==}, b, f, fits)

def replay_size_empty(+ops: List<&2, Laws.Op>) -> {Laws.size(Laws.replay(ops, Storage.Empty{})) == 0n : Nat}:
  match ops:
    case Nil{}:
      {==}
    case +op <> +later:
      match op:
        case Laws.Write{+i, w}:
          replay_size_empty(later)
        case Laws.Read{+i}:
          replay_size_empty(later)
        case Laws.First{}:
          replay_size_empty(later)
        case Laws.Second{}:
          replay_size_empty(later)
        case Laws.Length{}:
          replay_size_empty(later)

def status_hit(+d: Nat, +n: Nat, array: Array<F32>, -b: Model.Balanced(F32, d, array),
  -f: {Storage.bounds(d, n) == True{} : Bool}, +fits: {Storage.bounds(d, n) == True{} : Bool},
  +j: U32, +v: F32, +inside: {Nat.is_lt(U32.to_nat(j), n) == True{} : Bool}) ->
  {Laws.status(Storage.Owned{d, n, array, b, f}, j, v) == Done{Unit{}} : Result<&2, &2, Storage.Error, Unit>}:
  %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, Unit>, Storage.set_checked(d, n, array, j, v, Nat.is_lt(U32.to_nat(j), n), {==}, b, f),
    (Storage.Owned{d, n, Array.set(F32, array, j, v), Certificates.set_result(F32, d, array, j, v, Array.set(F32, array, j, v), {==}, b), f}, Done{Unit{}}),
    set_checked_hit(d, n, array, j, v, Nat.is_lt(U32.to_nat(j), n), {==}, b, f,
      Equal.sym(Bool, Nat.is_lt(U32.to_nat(j), n), True{}, inside), fits, inside)) :
    {Laws.outcome(Result<&2, &2, Storage.Error, Unit>, _) == Done{Unit{}} : Result<&2, &2, Storage.Error, Unit>}
  {==}

def replay_status(+rest: List<&2, Laws.Op>, +d: Nat, +n: Nat, +fill: F32, +history: List<&2, Effect>, +j: U32, +v: F32,
  hit: Bool, +dhit: {hit == hit_of(n, rest) : Bool},
  -b: Model.Balanced(F32, d, arr(history, d, fill)), -f: {Storage.bounds(d, n) == True{} : Bool},
  +fits: {Storage.bounds(d, n) == True{} : Bool}, +inside: {Nat.is_lt(U32.to_nat(j), n) == True{} : Bool}) ->
  {Laws.status(Laws.replay(rest, Storage.Owned{d, n, arr(history, d, fill), b, f}), j, v) == Done{Unit{}} : Result<&2, &2, Storage.Error, Unit>}:
  match rest:
    case Nil{}:
      status_hit(d, n, arr(history, d, fill), b, f, fits, j, v, inside)
    case +op <> +later:
      match op:
        case Laws.Write{+i, +w}:
          match hit:
            case True{}:
              +inside_i = Equal.sym(Bool, True{}, Nat.is_lt(U32.to_nat(i), n), dhit)
              %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, Unit>, Storage.set_checked(d, n, arr(history, d, fill), i, w, Nat.is_lt(U32.to_nat(i), n), {==}, b, f), (Storage.Owned{d, n, Array.set(F32, arr(history, d, fill), i, w), Certificates.set_result(F32, d, arr(history, d, fill), i, w, Array.set(F32, arr(history, d, fill), i, w), {==}, b), f}, Done{Unit{}}), set_checked_hit(d, n, arr(history, d, fill), i, w, Nat.is_lt(U32.to_nat(i), n), {==}, b, f, dhit, fits, inside_i)) :
                {Laws.status(Laws.replay(later, Laws.owner(Result<&2, &2, Storage.Error, Unit>, _)), j, v) == Done{Unit{}} : Result<&2, &2, Storage.Error, Unit>}
              replay_status(later, d, n, fill, Wrote{i, w} <> history, j, v, hit_of(n, later), {==}, Certificates.set_result(F32, d, arr(history, d, fill), i, w, Array.set(F32, arr(history, d, fill), i, w), {==}, b), f, fits, inside)
            case False{}:
              %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, Unit>, Storage.set_checked(d, n, arr(history, d, fill), i, w, Nat.is_lt(U32.to_nat(i), n), {==}, b, f), (Storage.Owned{d, n, arr(history, d, fill), b, f}, Fail{Storage.OutOfBounds{}}), set_checked_miss(d, n, arr(history, d, fill), i, w, Nat.is_lt(U32.to_nat(i), n), {==}, b, f, dhit)) :
                {Laws.status(Laws.replay(later, Laws.owner(Result<&2, &2, Storage.Error, Unit>, _)), j, v) == Done{Unit{}} : Result<&2, &2, Storage.Error, Unit>}
              replay_status(later, d, n, fill, history, j, v, hit_of(n, later), {==}, b, f, fits, inside)
        case Laws.Read{+i}:
          match hit:
            case True{}:
              +inside_i = Equal.sym(Bool, True{}, Nat.is_lt(U32.to_nat(i), n), dhit)
              %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, F32>, Storage.get_checked(d, n, arr(history, d, fill), i, Nat.is_lt(U32.to_nat(i), n), {==}, b, f), (Storage.Owned{d, n, Refinement.first(F32, Array.get(F32, arr(history, d, fill), i)), Certificates.get_result(F32, d, arr(history, d, fill), i, Array.get(F32, arr(history, d, fill), i), {==}, b), f}, Done{Refinement.second(F32, Array.get(F32, arr(history, d, fill), i))}), get_checked_keep(d, n, arr(history, d, fill), i, Nat.is_lt(U32.to_nat(i), n), {==}, b, f, dhit, fits, inside_i)) :
                {Laws.status(Laws.replay(later, Laws.owner(Result<&2, &2, Storage.Error, F32>, _)), j, v) == Done{Unit{}} : Result<&2, &2, Storage.Error, Unit>}
              replay_status(later, d, n, fill, Kept{i} <> history, j, v, hit_of(n, later), {==}, Certificates.get_result(F32, d, arr(history, d, fill), i, Array.get(F32, arr(history, d, fill), i), {==}, b), f, fits, inside)
            case False{}:
              %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, F32>, Storage.get_checked(d, n, arr(history, d, fill), i, Nat.is_lt(U32.to_nat(i), n), {==}, b, f), (Storage.Owned{d, n, arr(history, d, fill), b, f}, Fail{Storage.OutOfBounds{}}), get_checked_stay(d, n, arr(history, d, fill), i, Nat.is_lt(U32.to_nat(i), n), {==}, b, f, dhit)) :
                {Laws.status(Laws.replay(later, Laws.owner(Result<&2, &2, Storage.Error, F32>, _)), j, v) == Done{Unit{}} : Result<&2, &2, Storage.Error, Unit>}
              replay_status(later, d, n, fill, history, j, v, hit_of(n, later), {==}, b, f, fits, inside)
        case Laws.First{}:
          %Equal.sym(Storage.Storage & Storage.Storage, Storage.finish_clone(d, n, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b, f), (Storage.Owned{d, n, Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, arr(history, d, fill))), Certificates.clone_first(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f}, Storage.Owned{d, n, Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, arr(history, d, fill))), Certificates.clone_second(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f}), finish_pair(d, n, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b, f)) :
            {Laws.status(Laws.replay(later, Laws.owner(Storage.Storage, _)), j, v) == Done{Unit{}} : Result<&2, &2, Storage.Error, Unit>}
          replay_status(later, d, n, fill, Left{} <> history, j, v, hit_of(n, later), {==}, Certificates.clone_first(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f, fits, inside)
        case Laws.Second{}:
          %Equal.sym(Storage.Storage & Storage.Storage, Storage.finish_clone(d, n, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b, f), (Storage.Owned{d, n, Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, arr(history, d, fill))), Certificates.clone_first(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f}, Storage.Owned{d, n, Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, arr(history, d, fill))), Certificates.clone_second(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f}), finish_pair(d, n, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b, f)) :
            {Laws.status(Laws.replay(later, Laws.outcome(Storage.Storage, _)), j, v) == Done{Unit{}} : Result<&2, &2, Storage.Error, Unit>}
          replay_status(later, d, n, fill, Right{} <> history, j, v, hit_of(n, later), {==}, Certificates.clone_second(F32, d, arr(history, d, fill), Array.clone(F32, arr(history, d, fill)), {==}, b), f, fits, inside)
        case Laws.Length{}:
          replay_status(later, d, n, fill, history, j, v, hit_of(n, later), {==}, b, f, fits, inside)

def bounds_from(+d: Nat, +n: Nat, depth: {Nat.is_le(d, 20n) == True{} : Bool}, positive: {Nat.is_lt(0n, n) == True{} : Bool},
  full: {Nat.is_le(n, Model.capacity(d)) == True{} : Bool}) -> {Storage.bounds(d, n) == True{} : Bool}:
  %Equal.sym(Bool, Nat.is_le(d, 20n), True{}, depth) : {Bool.and(_, Bool.and(Nat.is_lt(0n, n), Nat.is_le(n, Model.capacity(d)))) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_lt(0n, n), True{}, positive) : {Bool.and(_, Nat.is_le(n, Model.capacity(d))) == True{} : Bool}
  full

def seek_depth(+k: Nat, +left: {Nat.add(0n, k) == 20n : Nat}) -> {Nat.is_le(k, 20n) == True{} : Bool}:
  %Equal.sym(Nat, k, 20n, left) : {Nat.is_le(_, 20n) == True{} : Bool}
  {==}

def seek_full(+n: Nat, +k: Nat, +left: {Nat.add(0n, k) == 20n : Nat},
  +limit: {Nat.is_le(n, Model.capacity(20n)) == True{} : Bool}) -> {Nat.is_le(n, Model.capacity(k)) == True{} : Bool}:
  %Equal.sym(Nat, k, 20n, left) : {Nat.is_le(n, Model.capacity(_)) == True{} : Bool}
  limit

def seek_room(+f: Nat, +k: Nat, +left: {Nat.add(f, k) == 20n : Nat}) -> {Nat.is_le(k, 20n) == True{} : Bool}:
  %left : {Nat.is_le(k, _) == True{} : Bool}
  le_add_left(f, k)

def seek_fit(+n: Nat, +k: Nat, +c: Nat, +decided: {True{} == Nat.is_le(n, c) : Bool}, +sized: {c == Model.capacity(k) : Nat}) ->
  {Nat.is_le(n, Model.capacity(k)) == True{} : Bool}:
  %sized : {Nat.is_le(n, _) == True{} : Bool}
  Equal.sym(Bool, True{}, Nat.is_le(n, c), decided)

def seek_double(+k: Nat, +c: Nat, +sized: {c == Model.capacity(k) : Nat}) -> {Nat.add(c, c) == Model.capacity(1n+k) : Nat}:
  %sized : {Nat.add(c, c) == Nat.add(_, _) : Nat}
  {==}

def seek_left(+p: Nat, +k: Nat, +left: {Nat.add(1n+p, k) == 20n : Nat}) -> {Nat.add(p, 1n+k) == 20n : Nat}:
  %Equal.sym(Nat, Nat.add(p, 1n+k), 1n+Nat.add(p, k), Arith.add_succ(p, k)) : {_ == 20n : Nat}
  left

def seek_bounds(+fuel: Nat, +n: Nat, +k: Nat, +c: Nat, done: Bool, +decided: {done == Nat.is_le(n, c) : Bool},
  +sized: {c == Model.capacity(k) : Nat}, +left: {Nat.add(fuel, k) == 20n : Nat},
  +limit: {Nat.is_le(n, Model.capacity(20n)) == True{} : Bool}, +positive: {Nat.is_lt(0n, n) == True{} : Bool}) ->
  {Storage.bounds(Pair.fst(Nat, Nat, Storage.depth_seek(fuel, n, k, c, done)), n) == True{} : Bool}:
  match fuel:
    case 0n:
      bounds_from(k, n, seek_depth(k, left), positive, seek_full(n, k, left, limit))
    case 1n+p:
      match done:
        case True{}:
          bounds_from(k, n, seek_room(1n+p, k, left), positive, seek_fit(n, k, c, decided, sized))
        case False{}:
          seek_bounds(p, n, 1n+k, Nat.add(c, c), Nat.is_le(n, Nat.add(c, c)), {==}, seek_double(k, c, sized),
            seek_left(p, k, left), limit, positive)

def limit_word() -> {1048576 == Size.capacity(20n) : U32}:
  {==}

def cap_from(+n: U32, +fits: {U32.is_le(n, 1048576) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(n), Model.capacity(20n)) == True{} : Bool}:
  %Route.cap_word(20n, {==}) : {Nat.is_le(U32.to_nat(n), _) == True{} : Bool}
  %limit_word() : {Nat.is_le(U32.to_nat(n), U32.to_nat(_)) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_le(U32.to_nat(n), U32.to_nat(1048576)), U32.is_le(n, 1048576), Arith.u32_le(n, 1048576)) : {_ == True{} : Bool}
  fits

def seek(+n: Nat, +limit: {Nat.is_le(n, Model.capacity(20n)) == True{} : Bool}, +positive: {Nat.is_lt(0n, n) == True{} : Bool}) ->
  {Storage.bounds(Pair.fst(Nat, Nat, Storage.depth_seek(20n, n, 0n, 1n, Nat.is_le(n, 1n))), n) == True{} : Bool}:
  seek_bounds(20n, n, 0n, 1n, Nat.is_le(n, 1n), {==}, {==}, {==}, limit, positive)

def alloc_read(+d: Nat, +n: Nat, +value: F32, valid: Bool, -decision: {valid == Storage.bounds(d, n) : Bool},
  +ops: List<&2, Laws.Op>, +j: U32, yes: {True{} == valid : Bool}, +fits: {Storage.bounds(d, n) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(j), n) == True{} : Bool}) ->
  {Laws.read(Laws.replay(ops, Laws.created.of(Storage.allocate(d, n, value, valid, decision))), j) ==
    Done{Laws.expected(ops, n, j, value)} : Result<&2, &2, Storage.Error, F32>}:
  match valid:
    case True{}:
      replay_read(ops, d, n, value, Nil{}, j, hit_of(n, ops), {==}, ArrayCore.new_balanced(F32, d, value),
        Equal.sym(Bool, True{}, Storage.bounds(d, n), decision), fits, inside, Unit{})
    case False{}:
      Empty.absurd({Laws.read(Laws.replay(ops, Laws.created.of(Storage.allocate(d, n, value, False{}, decision))), j) ==
        Done{Laws.expected(ops, n, j, value)} : Result<&2, &2, Storage.Error, F32>}, Arith.clash_true(yes))

def meta_read(+n: Nat, +value: F32, metadata: Nat & Nat, +ops: List<&2, Laws.Op>, +j: U32,
  +fits: {Storage.bounds(Pair.fst(Nat, Nat, metadata), n) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(j), n) == True{} : Bool}) ->
  {Laws.read(Laws.replay(ops, Laws.created.of(Storage.allocate_at(n, value, metadata))), j) ==
    Done{Laws.expected(ops, n, j, value)} : Result<&2, &2, Storage.Error, F32>}:
  (+d, +capacity) = metadata
  alloc_read(d, n, value, Storage.bounds(d, n), {==}, ops, j, Equal.sym(Bool, Storage.bounds(d, n), True{}, fits), fits, inside)

def new_read(+n: Nat, +value: F32, allowed: Bool, +ops: List<&2, Laws.Op>, +j: U32, yes: {True{} == allowed : Bool},
  +limit: {Nat.is_le(n, Model.capacity(20n)) == True{} : Bool}, +inside: {Nat.is_lt(U32.to_nat(j), n) == True{} : Bool}) ->
  {Laws.read(Laws.replay(ops, Laws.created.of(Storage.new_checked(n, value, allowed))), j) ==
    Done{Laws.expected(ops, n, j, value)} : Result<&2, &2, Storage.Error, F32>}:
  match n:
    case 0n:
      Empty.absurd({Laws.read(Laws.replay(ops, Laws.created.of(Storage.new_checked(0n, value, allowed))), j) ==
        Done{Laws.expected(ops, 0n, j, value)} : Result<&2, &2, Storage.Error, F32>}, Arith.lt_zero(U32.to_nat(j), inside))
    case 1n+p:
      match allowed:
        case True{}:
          meta_read(1n+p, value, Storage.depth_seek(20n, 1n+p, 0n, 1n, Nat.is_le(1n+p, 1n)), ops, j, seek(1n+p, limit, {==}), inside)
        case False{}:
          Empty.absurd({Laws.read(Laws.replay(ops, Laws.created.of(Storage.new_checked(1n+p, value, False{}))), j) ==
            Done{Laws.expected(ops, 1n+p, j, value)} : Result<&2, &2, Storage.Error, F32>}, Arith.clash_true(yes))

def alloc_size(+d: Nat, +n: Nat, +value: F32, valid: Bool, -decision: {valid == Storage.bounds(d, n) : Bool},
  +ops: List<&2, Laws.Op>, yes: {True{} == valid : Bool}, +fits: {Storage.bounds(d, n) == True{} : Bool}) ->
  {Laws.size(Laws.replay(ops, Laws.created.of(Storage.allocate(d, n, value, valid, decision)))) == n : Nat}:
  match valid:
    case True{}:
      replay_size_owned(ops, d, n, value, Nil{}, hit_of(n, ops), {==}, ArrayCore.new_balanced(F32, d, value),
        Equal.sym(Bool, True{}, Storage.bounds(d, n), decision), fits)
    case False{}:
      Empty.absurd({Laws.size(Laws.replay(ops, Laws.created.of(Storage.allocate(d, n, value, False{}, decision)))) == n : Nat},
        Arith.clash_true(yes))

def meta_size(+n: Nat, +value: F32, metadata: Nat & Nat, +ops: List<&2, Laws.Op>,
  +fits: {Storage.bounds(Pair.fst(Nat, Nat, metadata), n) == True{} : Bool}) ->
  {Laws.size(Laws.replay(ops, Laws.created.of(Storage.allocate_at(n, value, metadata)))) == n : Nat}:
  (+d, +capacity) = metadata
  alloc_size(d, n, value, Storage.bounds(d, n), {==}, ops, Equal.sym(Bool, Storage.bounds(d, n), True{}, fits), fits)

def new_size(+n: Nat, +value: F32, allowed: Bool, +ops: List<&2, Laws.Op>, yes: {True{} == allowed : Bool},
  +limit: {Nat.is_le(n, Model.capacity(20n)) == True{} : Bool}) ->
  {Laws.size(Laws.replay(ops, Laws.created.of(Storage.new_checked(n, value, allowed)))) == n : Nat}:
  match n:
    case 0n:
      replay_size_empty(ops)
    case 1n+p:
      match allowed:
        case True{}:
          meta_size(1n+p, value, Storage.depth_seek(20n, 1n+p, 0n, 1n, Nat.is_le(1n+p, 1n)), ops, seek(1n+p, limit, {==}))
        case False{}:
          Empty.absurd({Laws.size(Laws.replay(ops, Laws.created.of(Storage.new_checked(1n+p, value, False{})))) == 1n+p : Nat},
            Arith.clash_true(yes))

def positive_of(+n: Nat, +m: Nat, +over: {Nat.is_le(n, m) == False{} : Bool}) -> {Nat.is_lt(0n, n) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool},
        Arith.clash_true(Equal.trans(Bool, True{}, Nat.is_le(0n, m), False{}, Equal.sym(Bool, Nat.is_le(0n, m), True{}, Arith.le_zero(m)), over)))
    case 1n+p:
      {==}

def new_refused(+n: Nat, -value: F32, allowed: Bool, no: {False{} == allowed : Bool},
  +positive: {Nat.is_lt(0n, n) == True{} : Bool}) ->
  {Laws.refused.of(Storage.new_checked(n, value, allowed)) == Some{Storage.TooLarge{}} : Maybe<&2, Storage.Error>}:
  match n:
    case 0n:
      Empty.absurd({Laws.refused.of(Storage.new_checked(0n, value, allowed)) == Some{Storage.TooLarge{}} : Maybe<&2, Storage.Error>},
        Arith.clash(positive))
    case 1n+p:
      match allowed:
        case False{}:
          {==}
        case True{}:
          Empty.absurd({Laws.refused.of(Storage.new_checked(1n+p, value, True{})) == Some{Storage.TooLarge{}} : Maybe<&2, Storage.Error>},
            Arith.clash(no))

def over_nat(+n: U32, +over: {U32.is_le(n, 1048576) == False{} : Bool}) ->
  {Nat.is_le(U32.to_nat(n), U32.to_nat(1048576)) == False{} : Bool}:
  %Equal.sym(Bool, Nat.is_le(U32.to_nat(n), U32.to_nat(1048576)), U32.is_le(n, 1048576), Arith.u32_le(n, 1048576)) : {_ == False{} : Bool}
  over

def alloc_status(+d: Nat, +n: Nat, +value: F32, valid: Bool, -decision: {valid == Storage.bounds(d, n) : Bool},
  +ops: List<&2, Laws.Op>, +j: U32, +v: F32, yes: {True{} == valid : Bool}, +fits: {Storage.bounds(d, n) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(j), n) == True{} : Bool}) ->
  {Laws.status(Laws.replay(ops, Laws.created.of(Storage.allocate(d, n, value, valid, decision))), j, v) == Done{Unit{}} : Result<&2, &2, Storage.Error, Unit>}:
  match valid:
    case True{}:
      replay_status(ops, d, n, value, Nil{}, j, v, hit_of(n, ops), {==}, ArrayCore.new_balanced(F32, d, value),
        Equal.sym(Bool, True{}, Storage.bounds(d, n), decision), fits, inside)
    case False{}:
      Empty.absurd({Laws.status(Laws.replay(ops, Laws.created.of(Storage.allocate(d, n, value, False{}, decision))), j, v) == Done{Unit{}} : Result<&2, &2, Storage.Error, Unit>},
        Arith.clash_true(yes))

def meta_status(+n: Nat, +value: F32, metadata: Nat & Nat, +ops: List<&2, Laws.Op>, +j: U32, +v: F32,
  +fits: {Storage.bounds(Pair.fst(Nat, Nat, metadata), n) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(j), n) == True{} : Bool}) ->
  {Laws.status(Laws.replay(ops, Laws.created.of(Storage.allocate_at(n, value, metadata))), j, v) == Done{Unit{}} : Result<&2, &2, Storage.Error, Unit>}:
  (+d, +capacity) = metadata
  alloc_status(d, n, value, Storage.bounds(d, n), {==}, ops, j, v, Equal.sym(Bool, Storage.bounds(d, n), True{}, fits), fits, inside)

def new_status(+n: Nat, +value: F32, allowed: Bool, +ops: List<&2, Laws.Op>, +j: U32, +v: F32, yes: {True{} == allowed : Bool},
  +limit: {Nat.is_le(n, Model.capacity(20n)) == True{} : Bool}, +inside: {Nat.is_lt(U32.to_nat(j), n) == True{} : Bool}) ->
  {Laws.status(Laws.replay(ops, Laws.created.of(Storage.new_checked(n, value, allowed))), j, v) == Done{Unit{}} : Result<&2, &2, Storage.Error, Unit>}:
  match n:
    case 0n:
      Empty.absurd({Laws.status(Laws.replay(ops, Laws.created.of(Storage.new_checked(0n, value, allowed))), j, v) == Done{Unit{}} : Result<&2, &2, Storage.Error, Unit>},
        Arith.lt_zero(U32.to_nat(j), inside))
    case 1n+p:
      match allowed:
        case True{}:
          meta_status(1n+p, value, Storage.depth_seek(20n, 1n+p, 0n, 1n, Nat.is_le(1n+p, 1n)), ops, j, v, seek(1n+p, limit, {==}), inside)
        case False{}:
          Empty.absurd({Laws.status(Laws.replay(ops, Laws.created.of(Storage.new_checked(1n+p, value, False{}))), j, v) == Done{Unit{}} : Result<&2, &2, Storage.Error, Unit>},
            Arith.clash_true(yes))

def alloc_fits(+d: Nat, +n: Nat, -value: F32, valid: Bool, -decision: {valid == Storage.bounds(d, n) : Bool},
  yes: {True{} == valid : Bool}) ->
  {Laws.refused.of(Storage.allocate(d, n, value, valid, decision)) == None{} : Maybe<&2, Storage.Error>}:
  match valid:
    case True{}:
      {==}
    case False{}:
      Empty.absurd({Laws.refused.of(Storage.allocate(d, n, value, False{}, decision)) == None{} : Maybe<&2, Storage.Error>}, Arith.clash_true(yes))

def meta_fits(+n: Nat, -value: F32, metadata: Nat & Nat, +fits: {Storage.bounds(Pair.fst(Nat, Nat, metadata), n) == True{} : Bool}) ->
  {Laws.refused.of(Storage.allocate_at(n, value, metadata)) == None{} : Maybe<&2, Storage.Error>}:
  (+d, +capacity) = metadata
  alloc_fits(d, n, value, Storage.bounds(d, n), {==}, Equal.sym(Bool, Storage.bounds(d, n), True{}, fits))

def new_fits(+n: Nat, -value: F32, allowed: Bool, yes: {True{} == allowed : Bool},
  +limit: {Nat.is_le(n, Model.capacity(20n)) == True{} : Bool}) ->
  {Laws.refused.of(Storage.new_checked(n, value, allowed)) == None{} : Maybe<&2, Storage.Error>}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      match allowed:
        case True{}:
          meta_fits(1n+p, value, Storage.depth_seek(20n, 1n+p, 0n, 1n, Nat.is_le(1n+p, 1n)), seek(1n+p, limit, {==}))
        case False{}:
          Empty.absurd({Laws.refused.of(Storage.new_checked(1n+p, value, False{})) == None{} : Maybe<&2, Storage.Error>}, Arith.clash_true(yes))
