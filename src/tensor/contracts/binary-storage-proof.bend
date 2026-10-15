import Base
import ../../storage/model.bend as Model
import ../../storage/main.bend as Storage
import ../../storage/LAWS.bend as StorageLaws
import ../../storage/array.bend as ArrayCore
import ../../storage/refinement.bend as Refinement
import ../../storage/certificates.bend as Certificates
import ../../storage/api.bend as Api
import ../../storage/arith.bend as Arith
import ./binary-build-proof.bend as Build

# Storage facts the parallel build's export needs, all live: two balanced
# certificates of one array are equal, so two owned storages with equal
# arrays are equal; a replayed storage is an owned storage whose array is
# `Api.arr` of a history of effects; and `created` is `Array.new` at a
# computed depth.

def pair_eq(-A: Type, -B: Type, -a: A, -b: A, -c: B, -e: B, x: {a == b : A}, y: {c == e : B}) -> {(a, c) == (b, e) : A & B}:
  %x : {(a, c) == (_, e) : A & B}
  %y : {(a, c) == (a, _) : A & B}
  {==}

def balanced_unique(+d: Nat, array: Array<F32>, p: Model.Balanced(F32, d, array), q: Model.Balanced(F32, d, array)) ->
  {p == q : Model.Balanced(F32, d, array)}:
  match d array:
    case 0n ALeaf{x}:
      match p q:
        case Unit{} Unit{}:
          {==}
    case 0n ANode{xs, ys}:
      match p:
    case 1n+k ALeaf{x}:
      match p:
    case 1n+k ANode{xs, ys}:
      (p1, p2) = p
      (q1, q2) = q
      pair_eq(Model.Balanced(F32, k, xs), Model.Balanced(F32, k, ys), p1, q1, p2, q2,
        balanced_unique(k, xs, p1, q1), balanced_unique(k, ys, p2, q2))

def owned_eq(+d: Nat, +n: Nat, array: Array<F32>, -other: Array<F32>, e: {array == other : Array<F32>},
  p: Model.Balanced(F32, d, array), -f: {Storage.bounds(d, n) == True{} : Bool}) ->
  @q: Model.Balanced(F32, d, other) -> {Storage.Owned{d, n, array, p, f} == Storage.Owned{d, n, other, q, f} : Storage.Storage}:
  %e : (@q: Model.Balanced(F32, d, _) -> {Storage.Owned{d, n, array, p, f} == Storage.Owned{d, n, _, q, f} : Storage.Storage})
  q => Equal.cong(Model.Balanced(F32, d, array), Storage.Storage, r => Storage.Owned{d, n, array, r, f}, p, q, balanced_unique(d, array, p, q))

# The effects a list of storage operations leaves on an owned storage of
# length `n`, newest first: a write or read below `n` lands, the clone halves
# are kept, and the rest leave the array as it is.
def norm(+n: Nat, +ops: List<&2, StorageLaws.Op>, +history: List<&2, Api.Effect>) -> List<&2, Api.Effect>:
  match ops:
    case Nil{}:
      history
    case +op <> +later:
      match op:
        case StorageLaws.Write{+i, +w}:
          Bool.pick(List<&2, Api.Effect>, Nat.is_lt(U32.to_nat(i), n), norm(n, later, Api.Wrote{i, w} <> history), norm(n, later, history))
        case StorageLaws.Read{+i}:
          Bool.pick(List<&2, Api.Effect>, Nat.is_lt(U32.to_nat(i), n), norm(n, later, Api.Kept{i} <> history), norm(n, later, history))
        case StorageLaws.First{}:
          norm(n, later, Api.Left{} <> history)
        case StorageLaws.Second{}:
          norm(n, later, Api.Right{} <> history)
        case StorageLaws.Length{}:
          norm(n, later, history)

def norm_current(+n: Nat, +ops: List<&2, StorageLaws.Op>, +history: List<&2, Api.Effect>, +j: U32, +fill: F32,
  hit: Bool, +dhit: {hit == Api.hit_of(n, ops) : Bool}) ->
  {Api.current(norm(n, ops, history), j, fill) == StorageLaws.expected(ops, n, j, Api.current(history, j, fill)) : F32}:
  match ops:
    case Nil{}:
      {==}
    case +op <> +later:
      match op:
        case StorageLaws.Write{+i, +w}:
          match hit:
            case True{}:
              %dhit : {Api.current(Bool.pick(List<&2, Api.Effect>, _, norm(n, later, Api.Wrote{i, w} <> history), norm(n, later, history)), j, fill) ==
                StorageLaws.expected(later, n, j, Bool.pick(F32, Bool.and(_, Nat.is_eq(U32.to_nat(i), U32.to_nat(j))), w, Api.current(history, j, fill))) : F32}
              norm_current(n, later, Api.Wrote{i, w} <> history, j, fill, Api.hit_of(n, later), {==})
            case False{}:
              %dhit : {Api.current(Bool.pick(List<&2, Api.Effect>, _, norm(n, later, Api.Wrote{i, w} <> history), norm(n, later, history)), j, fill) ==
                StorageLaws.expected(later, n, j, Bool.pick(F32, Bool.and(_, Nat.is_eq(U32.to_nat(i), U32.to_nat(j))), w, Api.current(history, j, fill))) : F32}
              norm_current(n, later, history, j, fill, Api.hit_of(n, later), {==})
        case StorageLaws.Read{+i}:
          match hit:
            case True{}:
              %dhit : {Api.current(Bool.pick(List<&2, Api.Effect>, _, norm(n, later, Api.Kept{i} <> history), norm(n, later, history)), j, fill) ==
                StorageLaws.expected(later, n, j, Api.current(history, j, fill)) : F32}
              norm_current(n, later, Api.Kept{i} <> history, j, fill, Api.hit_of(n, later), {==})
            case False{}:
              %dhit : {Api.current(Bool.pick(List<&2, Api.Effect>, _, norm(n, later, Api.Kept{i} <> history), norm(n, later, history)), j, fill) ==
                StorageLaws.expected(later, n, j, Api.current(history, j, fill)) : F32}
              norm_current(n, later, history, j, fill, Api.hit_of(n, later), {==})
        case StorageLaws.First{}:
          norm_current(n, later, Api.Left{} <> history, j, fill, Api.hit_of(n, later), {==})
        case StorageLaws.Second{}:
          norm_current(n, later, Api.Right{} <> history, j, fill, Api.hit_of(n, later), {==})
        case StorageLaws.Length{}:
          norm_current(n, later, history, j, fill, Api.hit_of(n, later), {==})

def norm_written(+n: Nat, +ops: List<&2, StorageLaws.Op>, +history: List<&2, Api.Effect>, hit: Bool, +dhit: {hit == Api.hit_of(n, ops) : Bool},
  +all: {Build.written(n, history) == True{} : Bool}) -> {Build.written(n, norm(n, ops, history)) == True{} : Bool}:
  match ops:
    case Nil{}:
      all
    case +op <> +later:
      match op:
        case StorageLaws.Write{+i, +w}:
          match hit:
            case True{}:
              %dhit : {Build.written(n, Bool.pick(List<&2, Api.Effect>, _, norm(n, later, Api.Wrote{i, w} <> history), norm(n, later, history))) == True{} : Bool}
              norm_written(n, later, Api.Wrote{i, w} <> history, Api.hit_of(n, later), {==},
                %dhit : {Bool.and(_, Build.written(n, history)) == True{} : Bool}
                all)
            case False{}:
              %dhit : {Build.written(n, Bool.pick(List<&2, Api.Effect>, _, norm(n, later, Api.Wrote{i, w} <> history), norm(n, later, history))) == True{} : Bool}
              norm_written(n, later, history, Api.hit_of(n, later), {==}, all)
        case StorageLaws.Read{+i}:
          match hit:
            case True{}:
              %dhit : {Build.written(n, Bool.pick(List<&2, Api.Effect>, _, norm(n, later, Api.Kept{i} <> history), norm(n, later, history))) == True{} : Bool}
              norm_written(n, later, Api.Kept{i} <> history, Api.hit_of(n, later), {==}, all)
            case False{}:
              %dhit : {Build.written(n, Bool.pick(List<&2, Api.Effect>, _, norm(n, later, Api.Kept{i} <> history), norm(n, later, history))) == True{} : Bool}
              norm_written(n, later, history, Api.hit_of(n, later), {==}, all)
        case StorageLaws.First{}:
          norm_written(n, later, Api.Left{} <> history, Api.hit_of(n, later), {==}, all)
        case StorageLaws.Second{}:
          norm_written(n, later, Api.Right{} <> history, Api.hit_of(n, later), {==}, all)
        case StorageLaws.Length{}:
          norm_written(n, later, history, Api.hit_of(n, later), {==}, all)

# A replayed owned storage is the owned storage whose array is `arr` of the
# normalized history, with some certificate.
def replay_lift(+rest: List<&2, StorageLaws.Op>, +d: Nat, +n: Nat, +fill: F32, +history: List<&2, Api.Effect>,
  hit: Bool, +dhit: {hit == Api.hit_of(n, rest) : Bool},
  -b: Model.Balanced(F32, d, Api.arr(history, d, fill)), -f: {Storage.bounds(d, n) == True{} : Bool},
  +fits: {Storage.bounds(d, n) == True{} : Bool}, -Q: Storage.Storage -> Type,
  h: @-c: Model.Balanced(F32, d, Api.arr(norm(n, rest, history), d, fill)) ->
    Q(Storage.Owned{d, n, Api.arr(norm(n, rest, history), d, fill), c, f})) ->
  Q(StorageLaws.replay(rest, Storage.Owned{d, n, Api.arr(history, d, fill), b, f})):
  match rest:
    case Nil{}:
      h(b)
    case +op <> +later:
      match op:
        case StorageLaws.Write{+i, +w}:
          match hit:
            case True{}:
              +inside = Equal.sym(Bool, True{}, Nat.is_lt(U32.to_nat(i), n), dhit)
              %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, Unit>,
                Storage.set_checked(d, n, Api.arr(history, d, fill), i, w, Nat.is_lt(U32.to_nat(i), n), {==}, b, f),
                (Storage.Owned{d, n, Array.set(F32, Api.arr(history, d, fill), i, w),
                  Certificates.set_result(F32, d, Api.arr(history, d, fill), i, w, Array.set(F32, Api.arr(history, d, fill), i, w), {==}, b), f}, Done{Unit{}}),
                Api.set_checked_hit(d, n, Api.arr(history, d, fill), i, w, Nat.is_lt(U32.to_nat(i), n), {==}, b, f, dhit, fits, inside)) :
                Q(StorageLaws.replay(later, StorageLaws.owner(Result<&2, &2, Storage.Error, Unit>, _)))
              replay_lift(later, d, n, fill, Api.Wrote{i, w} <> history, Api.hit_of(n, later), {==},
                Certificates.set_result(F32, d, Api.arr(history, d, fill), i, w, Array.set(F32, Api.arr(history, d, fill), i, w), {==}, b), f, fits, Q,
                %Equal.sym(Bool, True{}, Nat.is_lt(U32.to_nat(i), n), dhit) :
                  (@-c: Model.Balanced(F32, d, Api.arr(Bool.pick(List<&2, Api.Effect>, _, norm(n, later, Api.Wrote{i, w} <> history), norm(n, later, history)), d, fill)) ->
                    Q(Storage.Owned{d, n, Api.arr(Bool.pick(List<&2, Api.Effect>, _, norm(n, later, Api.Wrote{i, w} <> history), norm(n, later, history)), d, fill), c, f}))
                h)
            case False{}:
              %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, Unit>,
                Storage.set_checked(d, n, Api.arr(history, d, fill), i, w, Nat.is_lt(U32.to_nat(i), n), {==}, b, f),
                (Storage.Owned{d, n, Api.arr(history, d, fill), b, f}, Fail{Storage.OutOfBounds{}}),
                Api.set_checked_miss(d, n, Api.arr(history, d, fill), i, w, Nat.is_lt(U32.to_nat(i), n), {==}, b, f, dhit)) :
                Q(StorageLaws.replay(later, StorageLaws.owner(Result<&2, &2, Storage.Error, Unit>, _)))
              replay_lift(later, d, n, fill, history, Api.hit_of(n, later), {==}, b, f, fits, Q,
                %Equal.sym(Bool, False{}, Nat.is_lt(U32.to_nat(i), n), dhit) :
                  (@-c: Model.Balanced(F32, d, Api.arr(Bool.pick(List<&2, Api.Effect>, _, norm(n, later, Api.Wrote{i, w} <> history), norm(n, later, history)), d, fill)) ->
                    Q(Storage.Owned{d, n, Api.arr(Bool.pick(List<&2, Api.Effect>, _, norm(n, later, Api.Wrote{i, w} <> history), norm(n, later, history)), d, fill), c, f}))
                h)
        case StorageLaws.Read{+i}:
          match hit:
            case True{}:
              +inside = Equal.sym(Bool, True{}, Nat.is_lt(U32.to_nat(i), n), dhit)
              %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, F32>,
                Storage.get_checked(d, n, Api.arr(history, d, fill), i, Nat.is_lt(U32.to_nat(i), n), {==}, b, f),
                (Storage.Owned{d, n, Refinement.first(F32, Array.get(F32, Api.arr(history, d, fill), i)),
                  Certificates.get_result(F32, d, Api.arr(history, d, fill), i, Array.get(F32, Api.arr(history, d, fill), i), {==}, b), f},
                  Done{Refinement.second(F32, Array.get(F32, Api.arr(history, d, fill), i))}),
                Api.get_checked_keep(d, n, Api.arr(history, d, fill), i, Nat.is_lt(U32.to_nat(i), n), {==}, b, f, dhit, fits, inside)) :
                Q(StorageLaws.replay(later, StorageLaws.owner(Result<&2, &2, Storage.Error, F32>, _)))
              replay_lift(later, d, n, fill, Api.Kept{i} <> history, Api.hit_of(n, later), {==},
                Certificates.get_result(F32, d, Api.arr(history, d, fill), i, Array.get(F32, Api.arr(history, d, fill), i), {==}, b), f, fits, Q,
                %Equal.sym(Bool, True{}, Nat.is_lt(U32.to_nat(i), n), dhit) :
                  (@-c: Model.Balanced(F32, d, Api.arr(Bool.pick(List<&2, Api.Effect>, _, norm(n, later, Api.Kept{i} <> history), norm(n, later, history)), d, fill)) ->
                    Q(Storage.Owned{d, n, Api.arr(Bool.pick(List<&2, Api.Effect>, _, norm(n, later, Api.Kept{i} <> history), norm(n, later, history)), d, fill), c, f}))
                h)
            case False{}:
              %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, F32>,
                Storage.get_checked(d, n, Api.arr(history, d, fill), i, Nat.is_lt(U32.to_nat(i), n), {==}, b, f),
                (Storage.Owned{d, n, Api.arr(history, d, fill), b, f}, Fail{Storage.OutOfBounds{}}),
                Api.get_checked_stay(d, n, Api.arr(history, d, fill), i, Nat.is_lt(U32.to_nat(i), n), {==}, b, f, dhit)) :
                Q(StorageLaws.replay(later, StorageLaws.owner(Result<&2, &2, Storage.Error, F32>, _)))
              replay_lift(later, d, n, fill, history, Api.hit_of(n, later), {==}, b, f, fits, Q,
                %Equal.sym(Bool, False{}, Nat.is_lt(U32.to_nat(i), n), dhit) :
                  (@-c: Model.Balanced(F32, d, Api.arr(Bool.pick(List<&2, Api.Effect>, _, norm(n, later, Api.Kept{i} <> history), norm(n, later, history)), d, fill)) ->
                    Q(Storage.Owned{d, n, Api.arr(Bool.pick(List<&2, Api.Effect>, _, norm(n, later, Api.Kept{i} <> history), norm(n, later, history)), d, fill), c, f}))
                h)
        case StorageLaws.First{}:
          %Equal.sym(Storage.Storage & Storage.Storage,
            Storage.finish_clone(d, n, Api.arr(history, d, fill), Array.clone(F32, Api.arr(history, d, fill)), {==}, b, f),
            (Storage.Owned{d, n, Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, Api.arr(history, d, fill))),
              Certificates.clone_first(F32, d, Api.arr(history, d, fill), Array.clone(F32, Api.arr(history, d, fill)), {==}, b), f},
              Storage.Owned{d, n, Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, Api.arr(history, d, fill))),
              Certificates.clone_second(F32, d, Api.arr(history, d, fill), Array.clone(F32, Api.arr(history, d, fill)), {==}, b), f}),
            Api.finish_pair(d, n, Api.arr(history, d, fill), Array.clone(F32, Api.arr(history, d, fill)), {==}, b, f)) :
            Q(StorageLaws.replay(later, StorageLaws.owner(Storage.Storage, _)))
          replay_lift(later, d, n, fill, Api.Left{} <> history, Api.hit_of(n, later), {==},
            Certificates.clone_first(F32, d, Api.arr(history, d, fill), Array.clone(F32, Api.arr(history, d, fill)), {==}, b), f, fits, Q, h)
        case StorageLaws.Second{}:
          %Equal.sym(Storage.Storage & Storage.Storage,
            Storage.finish_clone(d, n, Api.arr(history, d, fill), Array.clone(F32, Api.arr(history, d, fill)), {==}, b, f),
            (Storage.Owned{d, n, Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, Api.arr(history, d, fill))),
              Certificates.clone_first(F32, d, Api.arr(history, d, fill), Array.clone(F32, Api.arr(history, d, fill)), {==}, b), f},
              Storage.Owned{d, n, Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, Api.arr(history, d, fill))),
              Certificates.clone_second(F32, d, Api.arr(history, d, fill), Array.clone(F32, Api.arr(history, d, fill)), {==}, b), f}),
            Api.finish_pair(d, n, Api.arr(history, d, fill), Array.clone(F32, Api.arr(history, d, fill)), {==}, b, f)) :
            Q(StorageLaws.replay(later, StorageLaws.outcome(Storage.Storage, _)))
          replay_lift(later, d, n, fill, Api.Right{} <> history, Api.hit_of(n, later), {==},
            Certificates.clone_second(F32, d, Api.arr(history, d, fill), Array.clone(F32, Api.arr(history, d, fill)), {==}, b), f, fits, Q, h)
        case StorageLaws.Length{}:
          replay_lift(later, d, n, fill, history, Api.hit_of(n, later), {==}, b, f, fits, Q, h)

def replay_empty(+ops: List<&2, StorageLaws.Op>) -> {StorageLaws.replay(ops, Storage.Empty{}) == Storage.Empty{} : Storage.Storage}:
  match ops:
    case Nil{}:
      {==}
    case +op <> +later:
      match op:
        case StorageLaws.Write{+i, +w}:
          replay_empty(later)
        case StorageLaws.Read{+i}:
          replay_empty(later)
        case StorageLaws.First{}:
          replay_empty(later)
        case StorageLaws.Second{}:
          replay_empty(later)
        case StorageLaws.Length{}:
          replay_empty(later)

def created_alloc(+d: Nat, +n: Nat, +fill: F32, valid: Bool, -decision: {valid == Storage.bounds(d, n) : Bool},
  +yes: {True{} == valid : Bool}, +fits: {Storage.bounds(d, n) == True{} : Bool}, -Q: Storage.Storage -> Type,
  owned: @+d: Nat -> @+fits: {Storage.bounds(d, n) == True{} : Bool} -> @-f: {Storage.bounds(d, n) == True{} : Bool} ->
    Q(Storage.Owned{d, n, Array.new(F32, d, fill), ArrayCore.new_balanced(F32, d, fill), f})) ->
  Q(StorageLaws.created.of(Storage.allocate(d, n, fill, valid, decision))):
  match valid:
    case True{}:
      owned(d, fits, Equal.sym(Bool, True{}, Storage.bounds(d, n), decision))
    case False{}:
      Empty.absurd(Q(StorageLaws.created.of(Storage.allocate(d, n, fill, False{}, decision))), Arith.clash_true(yes))

def created_meta(+n: Nat, +fill: F32, metadata: Nat & Nat,
  +fits: {Storage.bounds(Pair.fst(Nat, Nat, metadata), n) == True{} : Bool}, -Q: Storage.Storage -> Type,
  owned: @+d: Nat -> @+fits: {Storage.bounds(d, n) == True{} : Bool} -> @-f: {Storage.bounds(d, n) == True{} : Bool} ->
    Q(Storage.Owned{d, n, Array.new(F32, d, fill), ArrayCore.new_balanced(F32, d, fill), f})) ->
  Q(StorageLaws.created.of(Storage.allocate_at(n, fill, metadata))):
  (+d, +capacity) = metadata
  created_alloc(d, n, fill, Storage.bounds(d, n), {==}, Equal.sym(Bool, Storage.bounds(d, n), True{}, fits), fits, Q, owned)

def created_checked(+n: Nat, +fill: F32, allowed: Bool, +yes: {True{} == allowed : Bool},
  +limit: {Nat.is_le(n, Model.capacity(20n)) == True{} : Bool}, -Q: Storage.Storage -> Type,
  empty: @zero: {n == 0n : Nat} -> Q(Storage.Empty{}),
  owned: @+d: Nat -> @+fits: {Storage.bounds(d, n) == True{} : Bool} -> @-f: {Storage.bounds(d, n) == True{} : Bool} ->
    Q(Storage.Owned{d, n, Array.new(F32, d, fill), ArrayCore.new_balanced(F32, d, fill), f})) ->
  Q(StorageLaws.created.of(Storage.new_checked(n, fill, allowed))):
  match n:
    case 0n:
      empty({==})
    case 1n+p:
      match allowed:
        case True{}:
          created_meta(1n+p, fill, Storage.depth_seek(20n, 1n+p, 0n, 1n, Nat.is_le(1n+p, 1n)), Api.seek(1n+p, limit, {==}), Q, owned)
        case False{}:
          Empty.absurd(Q(StorageLaws.created.of(Storage.new_checked(1n+p, fill, False{}))), Arith.clash_true(yes))

# `created(c, fill)` is empty for a zero count and otherwise the owned
# storage that holds `Array.new` at some depth.
def created_lift(+c: U32, +fill: F32, +within: {U32.is_le(c, 1048576) == True{} : Bool}, -Q: Storage.Storage -> Type,
  empty: @zero: {U32.to_nat(c) == 0n : Nat} -> Q(Storage.Empty{}),
  owned: @+d: Nat -> @+fits: {Storage.bounds(d, U32.to_nat(c)) == True{} : Bool} -> @-f: {Storage.bounds(d, U32.to_nat(c)) == True{} : Bool} ->
    Q(Storage.Owned{d, U32.to_nat(c), Array.new(F32, d, fill), ArrayCore.new_balanced(F32, d, fill), f})) ->
  Q(StorageLaws.created(c, fill)):
  created_checked(U32.to_nat(c), fill, U32.is_le(c, 1048576), Equal.sym(Bool, U32.is_le(c, 1048576), True{}, within),
    Api.cap_from(c, within), Q, empty, owned)
