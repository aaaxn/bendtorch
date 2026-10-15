import Base
import ../../storage/model.bend as Model
import ../../storage/refinement.bend as Refinement
import ../../storage/clone.bend as Clone
import ../../storage/main.bend as Storage
import ../../storage/frame.bend as Frame
import ../../storage/route.bend as Route
import ../../storage/api.bend as Api

# Facts about whole arrays that a parallel build's export needs: two balanced
# arrays that answer every path alike are the same array, and the array of a
# history answers each path with the last word written at its offset, or the
# fill.

def some_word(-word: F32, -other: F32, same: {Some{word} == Some{other} : Maybe<&2, F32>}) -> {word == other : F32}:
  Equal.cong(Maybe<&2, F32>, F32, found => Maybe.default(&2, F32, found, word), Some{word}, Some{other}, same)

def node_eq(-a: Array<F32>, -b: Array<F32>, -c: Array<F32>, -e: Array<F32>, x: {a == b : Array<F32>}, y: {c == e : Array<F32>}) ->
  {ANode{a, c} == ANode{b, e} : Array<F32>}:
  %x : {ANode{a, c} == ANode{_, e} : Array<F32>}
  %y : {ANode{a, c} == ANode{a, _} : Array<F32>}
  {==}

type Fact<-P: Type> is Data:
  Fact{-proof: P}

def SameAt(+d: Nat, -x: Array<F32>, -y: Array<F32>) -> Type:
  @+path: Model.Path(d) -> {Model.observe(F32, d, x, path) == Model.observe(F32, d, y, path) : Maybe<&2, F32>}

# Two balanced arrays of one depth that answer every path alike are equal.
def ext(+d: Nat, x: Array<F32>, y: Array<F32>, -bx: Model.Balanced(F32, d, x), -by: Model.Balanced(F32, d, y),
  -same: SameAt(d, x, y)) -> Fact<{x == y : Array<F32>}>:
  match d x y:
    case 0n ALeaf{u} ALeaf{v}:
      Fact{Equal.cong(F32, Array<F32>, w => ALeaf{w}, u, v, some_word(u, v, same(Unit{})))}
    case 0n ALeaf{u} ANode{ys, zs}:
      Fact{Empty.absurd({ALeaf{u} == ANode{ys, zs} : Array<F32>}, by)}
    case 0n ANode{xs, ws} y:
      Fact{Empty.absurd({ANode{xs, ws} == y : Array<F32>}, bx)}
    case 1n+k ALeaf{u} y:
      Fact{Empty.absurd({ALeaf{u} == y : Array<F32>}, bx)}
    case 1n+k ANode{xs, ws} ALeaf{v}:
      Fact{Empty.absurd({ANode{xs, ws} == ALeaf{v} : Array<F32>}, by)}
    case 1n+k ANode{xs, ws} ANode{ys, zs}:
      Fact{node_eq(xs, ys, ws, zs,
        {\{Fact: p => p} : Fact<{xs == ys : Array<F32>}> -> {xs == ys : Array<F32>}}(
          ext(k, xs, ys, Pair.fst(Model.Balanced(F32, k, xs), Model.Balanced(F32, k, ws), bx),
            Pair.fst(Model.Balanced(F32, k, ys), Model.Balanced(F32, k, zs), by), path => same(Model.Step{False{}, path}))),
        {\{Fact: p => p} : Fact<{ws == zs : Array<F32>}> -> {ws == zs : Array<F32>}}(
          ext(k, ws, zs, Pair.snd(Model.Balanced(F32, k, xs), Model.Balanced(F32, k, ws), bx),
            Pair.snd(Model.Balanced(F32, k, ys), Model.Balanced(F32, k, zs), by), path => same(Model.Step{True{}, path}))))}

# The last word written at offset `at`, or the fill.
def current_at(+history: List<&2, Api.Effect>, +at: Nat, +fill: F32) -> F32:
  match history:
    case Nil{}:
      fill
    case +effect <> +past:
      match effect:
        case Api.Wrote{+i, +w}:
          Bool.pick(F32, Nat.is_eq(U32.to_nat(i), at), w, current_at(past, at, fill))
        case Api.Kept{i}:
          current_at(past, at, fill)
        case Api.Left{}:
          current_at(past, at, fill)
        case Api.Right{}:
          current_at(past, at, fill)

def current_word(+history: List<&2, Api.Effect>, +j: U32, +fill: F32) ->
  {current_at(history, U32.to_nat(j), fill) == Api.current(history, j, fill) : F32}:
  match history:
    case Nil{}:
      {==}
    case +effect <> +past:
      match effect:
        case Api.Wrote{+i, +w}:
          Equal.cong(F32, F32, rest => Bool.pick(F32, Nat.is_eq(U32.to_nat(i), U32.to_nat(j)), w, rest),
            current_at(past, U32.to_nat(j), fill), Api.current(past, j, fill), current_word(past, j, fill))
        case Api.Kept{+i}:
          current_word(past, j, fill)
        case Api.Left{}:
          current_word(past, j, fill)
        case Api.Right{}:
          current_word(past, j, fill)

# A write below `n` names the path whose offset is its index.
def write_same(+d: Nat, +n: Nat, +i: U32, +path: Model.Path(d), +fits: {Storage.bounds(d, n) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(i), n) == True{} : Bool}) ->
  {Nat.is_eq(U32.to_nat(i), Model.offset(d, path)) == Model.same(d, Refinement.route(d, i), path) : Bool}:
  %Equal.sym(Nat, U32.to_nat(i), Model.offset(d, Refinement.route(d, i)),
    Route.route_offset(d, i, Api.depth_of(d, n, fits), Api.slot_of(d, n, i, fits, inside))) :
    {Nat.is_eq(_, Model.offset(d, path)) == Model.same(d, Refinement.route(d, i), path) : Bool}
  Api.same_eq(d, Refinement.route(d, i), path)

# Every path of the array of a history answers with `current_at` its offset.
def path_model(+history: List<&2, Api.Effect>, +d: Nat, +n: Nat, +fill: F32, +path: Model.Path(d),
  +fits: {Storage.bounds(d, n) == True{} : Bool}, valid: Api.Valid(n, history)) ->
  {Some{current_at(history, Model.offset(d, path), fill)} == Model.observe(F32, d, Api.arr(history, d, fill), path) : Maybe<&2, F32>}:
  match history:
    case Nil{}:
      Api.observe_new(d, fill, path)
    case +effect <> +past:
      match effect:
        case Api.Wrote{+i, +w}:
          (inside_i, rest) = valid
          %Equal.sym(Maybe<&2, F32>,
            Model.observe(F32, d, Array.set(F32, Api.arr(past, d, fill), i, w), path),
            Model.select(F32, Model.same(d, Refinement.route(d, i), path), w, Model.observe(F32, d, Api.arr(past, d, fill), path)),
            Frame.primitive_update_frame(F32, d, Api.arr(past, d, fill), i, path, w, Api.bal(past, d, fill))) :
            {Some{current_at(Api.Wrote{i, w} <> past, Model.offset(d, path), fill)} == _ : Maybe<&2, F32>}
          %write_same(d, n, i, path, fits, inside_i) :
            {Some{current_at(Api.Wrote{i, w} <> past, Model.offset(d, path), fill)} ==
              Model.select(F32, _, w, Model.observe(F32, d, Api.arr(past, d, fill), path)) : Maybe<&2, F32>}
          %path_model(past, d, n, fill, path, fits, rest) :
            {Some{current_at(Api.Wrote{i, w} <> past, Model.offset(d, path), fill)} ==
              Model.select(F32, Nat.is_eq(U32.to_nat(i), Model.offset(d, path)), w, _) : Maybe<&2, F32>}
          Api.select_pick(Nat.is_eq(U32.to_nat(i), Model.offset(d, path)), w, current_at(past, Model.offset(d, path), fill))
        case Api.Kept{+i}:
          %Equal.sym(Array<F32>, Refinement.first(F32, Array.get(F32, Api.arr(past, d, fill), i)), Api.arr(past, d, fill),
            Refinement.law_get_retains(F32, d, Api.arr(past, d, fill), i, Api.bal(past, d, fill))) :
            {Some{current_at(past, Model.offset(d, path), fill)} == Model.observe(F32, d, _, path) : Maybe<&2, F32>}
          path_model(past, d, n, fill, path, fits, valid)
        case Api.Left{}:
          %Equal.sym(Array<F32>, Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, Api.arr(past, d, fill))), Api.arr(past, d, fill),
            Clone.first_equal(F32, d, Api.arr(past, d, fill), Clone.certify(F32, d, Api.arr(past, d, fill), Api.bal(past, d, fill)))) :
            {Some{current_at(past, Model.offset(d, path), fill)} == Model.observe(F32, d, _, path) : Maybe<&2, F32>}
          path_model(past, d, n, fill, path, fits, valid)
        case Api.Right{}:
          %Equal.sym(Array<F32>, Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, Api.arr(past, d, fill))), Api.arr(past, d, fill),
            Clone.second_equal(F32, d, Api.arr(past, d, fill), Clone.certify(F32, d, Api.arr(past, d, fill), Api.bal(past, d, fill)))) :
            {Some{current_at(past, Model.offset(d, path), fill)} == Model.observe(F32, d, _, path) : Maybe<&2, F32>}
          path_model(past, d, n, fill, path, fits, valid)
