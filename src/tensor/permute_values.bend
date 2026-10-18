import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/LAWS.bend as StorageLaws
import ../storage/arith.bend as Arith
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./layout.bend as Layout
import ./axis_spec.bend as Axis
import ./transforms.bend as Transforms
import ./transpose_spec.bend as Transpose
import ./transpose_PROOF.bend as TransposeProof
import ./permutation_mapping.bend as Mapping
import ./word_conversion.bend as Conversion
import ./from_values_contract.bend as FromValues
import ./contracts/replayed.bend as Replayed
import ./contracts/pointwise_proof.bend as PointwiseProof
import ./index_map.bend as IndexMap
import ./observation.bend as Values
import ./transform_observation.bend as Observation
import ./permute_values_records.bend as Records
import ./permute_build_proof.bend as PermuteProof
import ./contracts/build_facts.bend as Facts
import ./contracts/words_proof.bend as Words
import ./contracts/LAWS.bend as Contracts
import ./contracts/binary_storage_proof.bend as Owned
import ./read_contract.bend as Reads
import ./permute_build.bend as Build
import ../storage/model.bend as Model
import ../storage/array.bend as ArrayCore
import ../storage/api.bend as Api
import ../storage/refinement.bend as Refinement
import ../storage/certificates.bend as Certificates

# The permute copy as an index-map instance: output index i reads the input
# at `Values.permute_source(input, axes, i)`, the row-major index of the
# input coordinate that `axes` sends to output coordinate i.

type Request is Data:
  Request{input: Shape.Shape, axes: List<&2, U32>}

def source_of(request: Request, index: Nat) -> Nat:
  Request{+input, +axes} = request
  Values.permute_source_index(input, axes, index)

# The parallel copy hands the source back as the First half of its clone.
def kept_source(+input: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>) -> Storage.Storage:
  Replayed.storage(input, fill, Replayed.snoc(ops, StorageLaws.First{}))

def written(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat) -> Storage.Storage:
  IndexMap.target(~Request, ~source_of, count, Request{input, axes}, input, fill, ops, Layout.output(input, axes, plan))

def Final(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat,
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>> -> Type) -> Type:
  @-source_fits: {Lifecycle.logical_length(kept_source(input, fill, ops)) == Shape.logical_numel(Shape.dimensions(input)) : Nat} ->
  @-target_fits: {Lifecycle.logical_length(written(input, axes, plan, fill, ops, count)) ==
    Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat} ->
    Q((Core.TTensor{kept_source(input, fill, ops), source_fits}, Done{Core.TTensor{written(input, axes, plan, fill, ops, count), target_fits}}))

def half_keeps(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +index: U32) ->
  {Replayed.value(shape, fill, Replayed.snoc(ops, StorageLaws.First{}), index) == Replayed.value(shape, fill, ops, index) : F32}:
  Replayed.expected_snoc(ops, StorageLaws.First{}, U32.to_nat(Shape.count(shape)), index, fill)


def words_snoc(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +previous: Nat, +word: F32, +other: F32,
  +before: List<&2, F32>, rest: {Contracts.words(shape, fill, ops, previous) == before : List<&2, F32>}, last: {word == other : F32}) ->
  {List.append(&2, F32, Contracts.words(shape, fill, ops, previous), [word]) == List.append(&2, F32, before, [other]) : List<&2, F32>}:
  Equal.trans(List<&2, F32>, List.append(&2, F32, Contracts.words(shape, fill, ops, previous), [word]),
    List.append(&2, F32, before, [word]), List.append(&2, F32, before, [other]),
    Equal.cong(List<&2, F32>, List<&2, F32>, values => List.append(&2, F32, values, [word]), Contracts.words(shape, fill, ops, previous), before, rest),
    Equal.cong(F32, List<&2, F32>, value => List.append(&2, F32, before, [value]), word, other, last))

# The First half of a clone lists the same words.

# The First half of a clone lists the same words.
def words_first(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat) ->
  {Contracts.words(shape, fill, Replayed.snoc(ops, StorageLaws.First{}), count) == Contracts.words(shape, fill, ops, count) : List<&2, F32>}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +position = U32.from_nat(previous)
      words_snoc(shape, fill, Replayed.snoc(ops, StorageLaws.First{}), previous,
        Replayed.value(shape, fill, Replayed.snoc(ops, StorageLaws.First{}), position), Replayed.value(shape, fill, ops, position),
        Contracts.words(shape, fill, ops, previous), words_first(shape, fill, ops, previous), half_keeps(shape, fill, ops, position))

# `take` of an owned tensor hands back the First half of its clone and the
# Second half's array.

# The copy loop's write history, replayed on the owned storage that
# `created` makes, is the owned storage that holds the output spec.
def chain(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>, +d: Nat, +n: Nat, -f: {Storage.bounds(d, n) == True{} : Bool}, +fits: {Storage.bounds(d, n) == True{} : Bool},
  +limit: U32, +n_limit: {Nat.is_le(n, U32.to_nat(limit)) == True{} : Bool},
  +k: Nat, +room: {Nat.is_le(k, n) == True{} : Bool}) ->
  {StorageLaws.replay(IndexMap.writes(~Request, ~source_of, k, Request{input, axes}, input, fill, ops), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}) == Storage.Owned{d, n, PermuteProof.spec(input, axes, fill, ops, k, d, 0n), PermuteProof.spec_balanced(input, axes, fill, ops, k, d, 0n), f} : Storage.Storage}:
  match k:
    case 0n:
      Owned.owned_eq(d, n, Array.new(F32, d, 0.0), PermuteProof.spec(input, axes, fill, ops, 0n, d, 0n), PermuteProof.new_spec(input, axes, fill, ops, d, 0n), ArrayCore.new_balanced(F32, d, 0.0), f)(
        PermuteProof.spec_balanced(input, axes, fill, ops, 0n, d, 0n))
    case 1n+j:
      +index = U32.from_nat(j)
      +value = PermuteProof.word(input, axes, fill, ops, j)
      +below_n = Conversion.le_successor_lt(j, n, room)
      +exact = Conversion.natural_word_exact(j, limit, Facts.le_trans(j, n, U32.to_nat(limit), Conversion.lt_le(j, n, below_n), n_limit))
      +inside = Equal.trans(Bool, Nat.is_lt(U32.to_nat(index), n), Nat.is_lt(j, n), True{},
        Equal.cong(Nat, Bool, x => Nat.is_lt(x, n), U32.to_nat(index), j, exact), below_n)
      +offset = Equal.trans(Nat, Nat.add(Model.offset(d, Refinement.route(d, index)), 0n), U32.to_nat(index), j,
        Facts.output_position(d, n, index, fits, inside), exact)
      +written = Equal.trans(Array<F32>, Array.set(F32, PermuteProof.spec(input, axes, fill, ops, j, d, 0n), index, value), ArrayCore.write_path(F32, d, PermuteProof.spec(input, axes, fill, ops, j, d, 0n), Refinement.route(d, index), value),
        PermuteProof.spec(input, axes, fill, ops, 1n+j, d, 0n), Refinement.law_set_refines(F32, d, PermuteProof.spec(input, axes, fill, ops, j, d, 0n), index, value, PermuteProof.spec_balanced(input, axes, fill, ops, j, d, 0n)),
        PermuteProof.write_spec(input, axes, fill, ops, j, d, 0n, Refinement.route(d, index), offset))
      Equal.trans(Storage.Storage, StorageLaws.replay(IndexMap.writes(~Request, ~source_of, 1n+j, Request{input, axes}, input, fill, ops), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
        StorageLaws.step(StorageLaws.replay(IndexMap.writes(~Request, ~source_of, j, Request{input, axes}, input, fill, ops), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), StorageLaws.Write{index, value}),
        Storage.Owned{d, n, PermuteProof.spec(input, axes, fill, ops, 1n+j, d, 0n), PermuteProof.spec_balanced(input, axes, fill, ops, 1n+j, d, 0n), f},
        FromValues.replay_snoc(IndexMap.writes(~Request, ~source_of, j, Request{input, axes}, input, fill, ops), StorageLaws.Write{index, value}, Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
        %Equal.sym(Storage.Storage, StorageLaws.replay(IndexMap.writes(~Request, ~source_of, j, Request{input, axes}, input, fill, ops), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), Storage.Owned{d, n, PermuteProof.spec(input, axes, fill, ops, j, d, 0n), PermuteProof.spec_balanced(input, axes, fill, ops, j, d, 0n), f},
          chain(input, axes, fill, ops, d, n, f, fits, limit, n_limit, j, Conversion.le_predecessor(j, n, room))) :
          {StorageLaws.step(_, StorageLaws.Write{index, value}) == Storage.Owned{d, n, PermuteProof.spec(input, axes, fill, ops, 1n+j, d, 0n), PermuteProof.spec_balanced(input, axes, fill, ops, 1n+j, d, 0n), f} : Storage.Storage}
        %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, Unit>,
          Storage.set_checked(d, n, PermuteProof.spec(input, axes, fill, ops, j, d, 0n), index, value, Nat.is_lt(U32.to_nat(index), n), {==}, PermuteProof.spec_balanced(input, axes, fill, ops, j, d, 0n), f),
          (Storage.Owned{d, n, Array.set(F32, PermuteProof.spec(input, axes, fill, ops, j, d, 0n), index, value),
            Certificates.set_result(F32, d, PermuteProof.spec(input, axes, fill, ops, j, d, 0n), index, value, Array.set(F32, PermuteProof.spec(input, axes, fill, ops, j, d, 0n), index, value), {==}, PermuteProof.spec_balanced(input, axes, fill, ops, j, d, 0n)), f}, Done{Unit{}}),
          Api.set_checked_hit(d, n, PermuteProof.spec(input, axes, fill, ops, j, d, 0n), index, value, Nat.is_lt(U32.to_nat(index), n), {==}, PermuteProof.spec_balanced(input, axes, fill, ops, j, d, 0n), f,
            Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), n), True{}, inside), fits, inside)) :
          {StorageLaws.owner(Result<&2, &2, Storage.Error, Unit>, _) == Storage.Owned{d, n, PermuteProof.spec(input, axes, fill, ops, 1n+j, d, 0n), PermuteProof.spec_balanced(input, axes, fill, ops, 1n+j, d, 0n), f} : Storage.Storage}
        Owned.owned_eq(d, n, Array.set(F32, PermuteProof.spec(input, axes, fill, ops, j, d, 0n), index, value), PermuteProof.spec(input, axes, fill, ops, 1n+j, d, 0n), written,
          Certificates.set_result(F32, d, PermuteProof.spec(input, axes, fill, ops, j, d, 0n), index, value, Array.set(F32, PermuteProof.spec(input, axes, fill, ops, j, d, 0n), index, value), {==}, PermuteProof.spec_balanced(input, axes, fill, ops, j, d, 0n)), f)(
          PermuteProof.spec_balanced(input, axes, fill, ops, 1n+j, d, 0n)))


# The output count of a permute is its input count.
def count_same(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>) -> {U32.to_nat(Shape.count(Layout.output(input, axes, plan))) == U32.to_nat(Shape.count(input)) : Nat}:
  Equal.trans(Nat, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), U32.to_nat(Shape.count(input)), Reads.count_exact(Layout.output(input, axes, plan)),
    Equal.trans(Nat, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Shape.logical_numel(Shape.dimensions(input)), U32.to_nat(Shape.count(input)), PermuteProof.plan_numel(input, axes, plan), Equal.sym(Nat, U32.to_nat(Shape.count(input)), Shape.logical_numel(Shape.dimensions(input)), Reads.count_exact(input))))

def Packed(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  kept: Storage.Storage, -kept_fits: {Lifecycle.logical_length(kept) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  made: Storage.Storage, -made_fits: {Lifecycle.logical_length(made) == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}) -> Type:
  Q(Transforms.copy_pack(input, axes, plan, (Core.TTensor{kept, kept_fits}, Done{Core.TTensor{made, made_fits}})))

def Kept(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  source: Storage.Storage, made: Storage.Storage) -> Type:
  @-kf: {Lifecycle.logical_length(StorageLaws.step(source, StorageLaws.First{})) == Shape.logical_numel(Shape.dimensions(input)) : Nat} ->
  @-ko: {Lifecycle.logical_length(made) == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat} ->
    Packed(input, axes, plan, Q, StorageLaws.step(source, StorageLaws.First{}), kf, made, ko)

def cloned_lift(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +d: Nat, -f: {Storage.bounds(d, U32.to_nat(Shape.count(Layout.output(input, axes, plan)))) == True{} : Bool}, -fo: {U32.to_nat(Shape.count(Layout.output(input, axes, plan))) == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat},
  +ds: Nat, -b: Model.Balanced(F32, ds, Api.arr(Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill)), -f2: {Storage.bounds(ds, U32.to_nat(Shape.count(input))) == True{} : Bool},
  pair: Array<F32> & Array<F32>, -same: {pair == Array.clone(F32, Api.arr(Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill)) : Array<F32> & Array<F32>},
  -sf: {Lifecycle.logical_length(Storage.Owned{ds, U32.to_nat(Shape.count(input)), Api.arr(Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill), b, f2}) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  hyp: @-kf: {Lifecycle.logical_length(StorageLaws.owner(Storage.Storage, Storage.finish_clone(ds, U32.to_nat(Shape.count(input)), Api.arr(Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill), pair, same, b, f2))) == Shape.logical_numel(Shape.dimensions(input)) : Nat} ->
    Packed(input, axes, plan, Q, StorageLaws.owner(Storage.Storage, Storage.finish_clone(ds, U32.to_nat(Shape.count(input)), Api.arr(Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill), pair, same, b, f2)), kf,
      Storage.Owned{d, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Build.build(d, Build.Request{input, axes}, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Pair.snd(Array<F32>, Array<F32>, pair)),
        Build.balanced(d, Build.Request{input, axes}, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Pair.snd(Array<F32>, Array<F32>, pair)), f}, fo)) ->
  Q(Transforms.copy_pack(input, axes, plan, Transforms.built_cloned(input, axes, plan, d, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), f, fo, ds, U32.to_nat(Shape.count(input)), Api.arr(Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill), pair, same, b, f2, sf))):
  (kept, copy) = pair
  hyp(sf)

# The copy loop's write history, replayed on the storage `Core.new`
# allocates, is the build's output on the Second half of the source's clone.
def made(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +d: Nat, +fits: {Storage.bounds(d, U32.to_nat(Shape.count(Layout.output(input, axes, plan)))) == True{} : Bool}, -f: {Storage.bounds(d, U32.to_nat(Shape.count(Layout.output(input, axes, plan)))) == True{} : Bool},
  +ds: Nat, +fits_s: {Storage.bounds(ds, U32.to_nat(Shape.count(input))) == True{} : Bool}) ->
  {StorageLaws.replay(IndexMap.writes(~Request, ~source_of, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Request{input, axes}, input, fill, ops), Storage.Owned{d, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}) == Storage.Owned{d, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Build.build(d, Build.Request{input, axes}, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill)),
    Build.balanced(d, Build.Request{input, axes}, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill)), f} : Storage.Storage}:
  +count_exact = Reads.count_exact(Layout.output(input, axes, plan))
  Equal.trans(Storage.Storage, StorageLaws.replay(IndexMap.writes(~Request, ~source_of, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Request{input, axes}, input, fill, ops), Storage.Owned{d, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
    StorageLaws.replay(IndexMap.writes(~Request, ~source_of, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Request{input, axes}, input, fill, ops), Storage.Owned{d, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
    Storage.Owned{d, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Build.build(d, Build.Request{input, axes}, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill)), Build.balanced(d, Build.Request{input, axes}, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill)), f},
    Equal.cong(Nat, Storage.Storage, count => StorageLaws.replay(IndexMap.writes(~Request, ~source_of, count, Request{input, axes}, input, fill, ops), Storage.Owned{d, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
      Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Equal.sym(Nat, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), count_exact)),
    Equal.trans(Storage.Storage, StorageLaws.replay(IndexMap.writes(~Request, ~source_of, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Request{input, axes}, input, fill, ops), Storage.Owned{d, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
      Storage.Owned{d, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), PermuteProof.spec(input, axes, fill, ops, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), d, 0n), PermuteProof.spec_balanced(input, axes, fill, ops, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), d, 0n), f},
      Storage.Owned{d, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Build.build(d, Build.Request{input, axes}, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill)), Build.balanced(d, Build.Request{input, axes}, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill)), f},
      chain(input, axes, fill, ops, d, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), f, fits, Shape.count(Layout.output(input, axes, plan)), Conversion.le_reflexive(U32.to_nat(Shape.count(Layout.output(input, axes, plan)))), U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Conversion.le_reflexive(U32.to_nat(Shape.count(Layout.output(input, axes, plan))))),
      Owned.owned_eq(d, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), PermuteProof.spec(input, axes, fill, ops, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), d, 0n), Build.build(d, Build.Request{input, axes}, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill)),
        Equal.sym(Array<F32>, Build.build(d, Build.Request{input, axes}, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill)), PermuteProof.spec(input, axes, fill, ops, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), d, 0n),
          PermuteProof.build_eq(input, axes, fill, ops, plan, ds, fits_s, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), count_exact, d, Build.splits(d), 0n)),
        PermuteProof.spec_balanced(input, axes, fill, ops, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), d, 0n), f)(Build.balanced(d, Build.Request{input, axes}, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill)))))

# With no output cell the input has none either: both are empty, and the
# copy hands the source back as it is.
def empty_lift(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +zero: {U32.to_nat(Shape.count(Layout.output(input, axes, plan))) == 0n : Nat}, -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  -fo: {Lifecycle.logical_length(Storage.Empty{}) == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat}) ->
  @-sf: {Lifecycle.logical_length(Replayed.storage(input, fill, ops)) == Shape.logical_numel(Shape.dimensions(input)) : Nat} -> Kept(input, axes, plan, Q, Replayed.storage(input, fill, ops), StorageLaws.replay(IndexMap.writes(~Request, ~source_of, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Request{input, axes}, input, fill, ops), Storage.Empty{})) ->
    Packed(input, axes, plan, Q, Replayed.storage(input, fill, ops), sf, Storage.Empty{}, fo):
  Owned.created_lift(Shape.count(input), fill, Conversion.shape_count_within(input),
    s => (@-sf: {Lifecycle.logical_length(StorageLaws.replay(ops, s)) == Shape.logical_numel(Shape.dimensions(input)) : Nat} -> Kept(input, axes, plan, Q, StorageLaws.replay(ops, s), StorageLaws.replay(IndexMap.writes(~Request, ~source_of, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Request{input, axes}, input, fill, ops), Storage.Empty{})) ->
      Packed(input, axes, plan, Q, StorageLaws.replay(ops, s), sf, Storage.Empty{}, fo)),
    nothing => sf => hyp =>
      Facts.storage_shift(input, StorageLaws.step(StorageLaws.replay(ops, Storage.Empty{}), StorageLaws.First{}), StorageLaws.replay(ops, Storage.Empty{}),
        Equal.trans(Storage.Storage, StorageLaws.step(StorageLaws.replay(ops, Storage.Empty{}), StorageLaws.First{}), Storage.Empty{}, StorageLaws.replay(ops, Storage.Empty{}),
          Equal.cong(Storage.Storage, Storage.Storage, st => StorageLaws.step(st, StorageLaws.First{}), StorageLaws.replay(ops, Storage.Empty{}), Storage.Empty{}, Owned.replay_empty(ops)),
          Equal.sym(Storage.Storage, StorageLaws.replay(ops, Storage.Empty{}), Storage.Empty{}, Owned.replay_empty(ops))),
        st => k => Packed(input, axes, plan, Q, st, k, Storage.Empty{}, fo),
        k => Facts.storage_shift(Layout.output(input, axes, plan), StorageLaws.replay(IndexMap.writes(~Request, ~source_of, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Request{input, axes}, input, fill, ops), Storage.Empty{}), Storage.Empty{}, Owned.replay_empty(IndexMap.writes(~Request, ~source_of, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Request{input, axes}, input, fill, ops)),
          st => ko => Packed(input, axes, plan, Q, StorageLaws.step(StorageLaws.replay(ops, Storage.Empty{}), StorageLaws.First{}), k, st, ko),
          ko => hyp(k, ko))(fo))(sf),
    d => fits => f => Empty.absurd(@-sf: {Lifecycle.logical_length(StorageLaws.replay(ops, Storage.Owned{d, U32.to_nat(Shape.count(input)), Array.new(F32, d, fill), ArrayCore.new_balanced(F32, d, fill), f})) == Shape.logical_numel(Shape.dimensions(input)) : Nat} ->
      Kept(input, axes, plan, Q, StorageLaws.replay(ops, Storage.Owned{d, U32.to_nat(Shape.count(input)), Array.new(F32, d, fill), ArrayCore.new_balanced(F32, d, fill), f}), StorageLaws.replay(IndexMap.writes(~Request, ~source_of, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Request{input, axes}, input, fill, ops), Storage.Empty{})) ->
        Packed(input, axes, plan, Q, StorageLaws.replay(ops, Storage.Owned{d, U32.to_nat(Shape.count(input)), Array.new(F32, d, fill), ArrayCore.new_balanced(F32, d, fill), f}), sf, Storage.Empty{}, fo),
      Arith.lt_zero(0n,
        %Equal.trans(Nat, U32.to_nat(Shape.count(input)), U32.to_nat(Shape.count(Layout.output(input, axes, plan))), 0n, Equal.sym(Nat, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), U32.to_nat(Shape.count(input)), count_same(input, axes, plan)), zero) : {Nat.is_lt(0n, _) == True{} : Bool}
        Arith.and_left(Nat.is_lt(0n, U32.to_nat(Shape.count(input))), Nat.is_le(U32.to_nat(Shape.count(input)), Model.capacity(d)),
          Arith.and_right(Nat.is_le(d, 20n), Bool.and(Nat.is_lt(0n, U32.to_nat(Shape.count(input))), Nat.is_le(U32.to_nat(Shape.count(input)), Model.capacity(d))), fits)))))

def run_lift(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  h: Final(input, axes, plan, fill, ops, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), pair => Q(Transforms.copy_pack(input, axes, plan, pair))),
  -source_fits: {Lifecycle.logical_length(Replayed.storage(input, fill, ops)) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Q(Transforms.copy_allocated(input, axes, plan, Core.TTensor{Replayed.storage(input, fill, ops), source_fits}, Core.new(Layout.output(input, axes, plan), 0.0))):
  Replayed.new_lift(Layout.output(input, axes, plan), 0.0, result => Q(Transforms.copy_allocated(input, axes, plan, Core.TTensor{Replayed.storage(input, fill, ops), source_fits}, result)),
    target_fits => Owned.created_lift(Shape.count(Layout.output(input, axes, plan)), 0.0, Conversion.shape_count_within(Layout.output(input, axes, plan)),
      s => (Kept(input, axes, plan, Q, Replayed.storage(input, fill, ops), StorageLaws.replay(IndexMap.writes(~Request, ~source_of, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Request{input, axes}, input, fill, ops), s)) -> @-fo: {Lifecycle.logical_length(s) == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat} ->
        Q(Transforms.copy_pack(input, axes, plan, Transforms.built(input, axes, plan, Core.TTensor{Replayed.storage(input, fill, ops), source_fits}, Core.TTensor{s, fo})))),
      zero => hyp => fo => empty_lift(input, axes, plan, fill, ops, zero, Q, fo)(source_fits, hyp),
      d => fits => f => hyp => fo =>
        +positive = Arith.and_left(Nat.is_lt(0n, U32.to_nat(Shape.count(Layout.output(input, axes, plan)))), Nat.is_le(U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Model.capacity(d)),
          Arith.and_right(Nat.is_le(d, 20n), Bool.and(Nat.is_lt(0n, U32.to_nat(Shape.count(Layout.output(input, axes, plan)))), Nat.is_le(U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Model.capacity(d))), fits))
        Facts.input_lift(input, fill, ops,
          %count_same(input, axes, plan) : {Nat.is_lt(0n, _) == True{} : Bool}
          positive,
          st => (@-sf: {Lifecycle.logical_length(st) == Shape.logical_numel(Shape.dimensions(input)) : Nat} -> Kept(input, axes, plan, Q, st, StorageLaws.replay(IndexMap.writes(~Request, ~source_of, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Request{input, axes}, input, fill, ops), Storage.Owned{d, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f})) ->
            Q(Transforms.copy_pack(input, axes, plan, Transforms.built_owned(input, axes, plan, d, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), f, fo, Core.TTensor{st, sf})))),
          ds => fits_s => b => f2 => sf => hyp2 =>
            cloned_lift(input, axes, plan, fill, ops, d, f, fo, ds, b, f2, Array.clone(F32, Api.arr(Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill)), {==}, sf, Q,
              kf => Facts.storage_shift(Layout.output(input, axes, plan), StorageLaws.replay(IndexMap.writes(~Request, ~source_of, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Request{input, axes}, input, fill, ops), Storage.Owned{d, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
                Storage.Owned{d, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Build.build(d, Build.Request{input, axes}, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill)), Build.balanced(d, Build.Request{input, axes}, U32.to_nat(Shape.count(Layout.output(input, axes, plan))), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill)), f},
                made(input, axes, plan, fill, ops, d, fits, f, ds, fits_s),
                st => ko => Packed(input, axes, plan, Q, StorageLaws.step(Storage.Owned{ds, U32.to_nat(Shape.count(input)), Api.arr(Owned.norm(U32.to_nat(Shape.count(input)), ops, Nil{}), ds, fill), b, f2}, StorageLaws.First{}), kf, st, ko),
                ko => hyp2(kf, ko))(fo)))(source_fits, hyp))(
      kf => ko => Facts.storage_shift(input, kept_source(input, fill, ops), StorageLaws.step(Replayed.storage(input, fill, ops), StorageLaws.First{}),
        Replayed.replay_snoc(ops, StorageLaws.First{}, input, fill),
        st => k => Packed(input, axes, plan, Q, st, k, StorageLaws.replay(IndexMap.writes(~Request, ~source_of, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Request{input, axes}, input, fill, ops), StorageLaws.created(Shape.count(Layout.output(input, axes, plan)), 0.0)), ko),
        k => h(k, ko))(kf), target_fits))

def planned_lift(+input: Shape.Shape, +axes: List<&2, U32>, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  result: Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>,
  done: {Result.is_done(&2, &2, Layout.Error, Layout.Plan<input, axes>, result) == True{} : Bool},
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  h: @+plan: Layout.Plan<input, axes> -> Final(input, axes, plan, fill, ops, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))),
    pair => Q(Transforms.copy_pack(input, axes, plan, pair))),
  -source_fits: {Lifecycle.logical_length(Replayed.storage(input, fill, ops)) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Q(Transforms.copy_planned(input, axes, Core.TTensor{Replayed.storage(input, fill, ops), source_fits}, result)):
  match result:
    case Fail{error}:
      Empty.absurd(Q(Transforms.copy_planned(input, axes, Core.TTensor{Replayed.storage(input, fill, ops), source_fits},
        Fail{error})), Arith.clash(done))
    case Done{+plan}:
      run_lift(input, axes, plan, fill, ops, Q, h(plan), source_fits)

def permute_lift(+input: Shape.Shape, +values: List<&2, F32>, +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool},
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  h: @+plan: Layout.Plan<input, axes> -> Final(input, axes, plan, 0.0, FromValues.writes(values, 0n), Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))),
    pair => Q(Transforms.copy_pack(input, axes, plan, pair))),
  -source_fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Q(Transforms.permute_copy(input, Core.TTensor{Replayed.storage(input, 0.0, FromValues.writes(values, 0n)), source_fits}, axes)):
  planned_lift(input, axes, 0.0, FromValues.writes(values, 0n), Layout.permutation(input, axes),
    Mapping.plan_ranked_success(input, axes,
      Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), valid),
    Q, h, source_fits)

def output_inside(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +index: U32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool}:
  %Equal.sym(Nat, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Shape.logical_numel(Shape.dimensions(input)),
    PermuteProof.plan_numel(input, axes, plan)) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
  inside

def found_value(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +values: List<&2, F32>,
  +counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  +index: U32, +mapping: Layout.Mapping<input, axes, plan, index>) ->
  {Some{Replayed.value(input, 0.0, FromValues.writes(values, 0n), U32.from_nat(source_of(Request{input, axes}, U32.to_nat(index))))} ==
    Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>}:
  +found = Layout.source_index(input, axes, plan, index, mapping)
  +history = FromValues.writes(values, 0n)
  %Equal.sym(U32, U32.from_nat(U32.to_nat(index)), index, Conversion.word_natural_exact(index)) :
    {Some{Replayed.value(input, 0.0, history, U32.from_nat(U32.to_nat(Maybe.default(&2, U32, Values.permute_source(input, axes, _), 0))))} ==
      Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>}
  %PermuteProof.mapped_index(input, axes, plan, index, mapping) :
    {Some{Replayed.value(input, 0.0, history, _)} == Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>}
  %PermuteProof.reference(input, axes, plan, index, mapping) :
    {Some{Replayed.value(input, 0.0, history, found)} == Values.listed(values, _) : Maybe<&2, F32>}
  IndexMap.stored_at(input, values, counted, found, PermuteProof.mapped_bound(input, axes, plan, index, mapping))

def found_mapped(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +values: List<&2, F32>,
  +counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(input)) : Nat}, +index: U32,
  result: Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>,
  done: {Result.is_done(&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>, result) == True{} : Bool}) ->
  {Some{Replayed.value(input, 0.0, FromValues.writes(values, 0n), U32.from_nat(source_of(Request{input, axes}, U32.to_nat(index))))} ==
    Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>}:
  match result:
    case Fail{error}:
      Empty.absurd({Some{Replayed.value(input, 0.0, FromValues.writes(values, 0n), U32.from_nat(source_of(Request{input, axes}, U32.to_nat(index))))} ==
        Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>}, Arith.clash(done))
    case Done{+mapping}:
      found_value(input, axes, plan, values, counted, index, mapping)

def Permuted(+input: Shape.Shape, +values: List<&2, F32>, +axes: List<&2, U32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type) -> Type:
  Q(Transforms.permute_copy(input, Core.TTensor{Replayed.storage(input, 0.0, FromValues.writes(values, 0n)), fits}, axes))

def permuted_code(+input: Shape.Shape, +values: List<&2, F32>, +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool},
  -fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Permuted(input, values, axes, fits, pair => {Some{Observation.failure_code_pair(input, pair)} == Some{0} : Maybe<&2, U32>}):
  permute_lift(input, values, axes, valid, pair => {Some{Observation.failure_code_pair(input, pair)} == Some{0} : Maybe<&2, U32>},
    plan => source_fits => target_fits => {==}, fits)

def permuted_dimensions(+input: Shape.Shape, +values: List<&2, F32>, +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool},
  -fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Permuted(input, values, axes, fits, pair => {Observation.applied_dimensions_pair(input, pair) ==
    Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))} : Maybe<&2, List<&2, U32>>}):
  permute_lift(input, values, axes, valid, pair => {Observation.applied_dimensions_pair(input, pair) ==
      Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))} : Maybe<&2, List<&2, U32>>},
    plan => source_fits => target_fits =>
      Equal.cong(List<&2, U32>, Maybe<&2, List<&2, U32>>, dimensions => Some{dimensions},
        Shape.dimensions(Layout.output(input, axes, plan)), Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes)),
        Mapping.permutation_dimensions_order(input, axes, plan)), fits)

def permuted_length(+input: Shape.Shape, +values: List<&2, F32>, +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool},
  -fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Permuted(input, values, axes, fits, pair => {Observation.applied_length_pair(input, pair) ==
    Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>}):
  permute_lift(input, values, axes, valid, pair => {Observation.applied_length_pair(input, pair) ==
      Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>},
    plan => source_fits => target_fits =>
      +output = Layout.output(input, axes, plan)
      +total = Shape.logical_numel(Shape.dimensions(output))
      -made = written(input, axes, plan, 0.0, FromValues.writes(values, 0n), total)
      Equal.cong(Nat, Maybe<&2, Nat>, number => Some{number}, Lifecycle.logical_length(made), Shape.logical_numel(Shape.dimensions(input)),
        Equal.trans(Nat, Lifecycle.logical_length(made), total, Shape.logical_numel(Shape.dimensions(input)),
          Replayed.fits(output, 0.0, IndexMap.writes(~Request, ~source_of, total, Request{input, axes}, input, 0.0, FromValues.writes(values, 0n))),
          PermuteProof.plan_numel(input, axes, plan))), fits)

def permuted_count(+input: Shape.Shape, +values: List<&2, F32>, +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool},
  -fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Permuted(input, values, axes, fits, pair => {Observation.applied_count_pair(input, pair) == Some{Shape.count(input)} : Maybe<&2, U32>}):
  permute_lift(input, values, axes, valid, pair => {Observation.applied_count_pair(input, pair) == Some{Shape.count(input)} : Maybe<&2, U32>},
    plan => source_fits => target_fits =>
      Equal.cong(Nat, Maybe<&2, U32>, number => Some{U32.from_nat(number)}, Shape.elements(Layout.output(input, axes, plan)), Shape.elements(input),
        PermuteProof.plan_elements(input, axes, plan)), fits)

def permuted_value(+input: Shape.Shape, +values: List<&2, F32>,
  +counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool}, +index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool},
  -fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Permuted(input, values, axes, fits, pair => {Observation.applied_value_pair(input, index, pair) ==
    Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>}):
  permute_lift(input, values, axes, valid, pair => {Observation.applied_value_pair(input, index, pair) ==
      Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>},
    plan => source_fits => target_fits =>
      +output = Layout.output(input, axes, plan)
      +total = Shape.logical_numel(Shape.dimensions(output))
      +history = FromValues.writes(values, 0n)
      +below = output_inside(input, axes, plan, index, inside)
      Equal.trans(Maybe<&2, F32>,
        Core.dynamic_read_value(Core.read_dynamic(Core.Packed{output, Core.TTensor{written(input, axes, plan, 0.0, FromValues.writes(values, 0n), total), target_fits}}, index)),
        Some{Replayed.value(input, 0.0, history, U32.from_nat(source_of(Request{input, axes}, U32.to_nat(index))))},
        Values.listed(values, Values.permute_source(input, axes, index)),
        IndexMap.copied_point(~Request, ~source_of, Request{input, axes}, input, 0.0, history, output, target_fits, index, below),
        found_mapped(input, axes, plan, values, counted, index, Layout.map_index(input, axes, plan, index),
          PermuteProof.mapping_done(input, axes, plan, index, below))), fits)

def permuted_kept(+input: Shape.Shape, +values: List<&2, F32>,
  +counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  +axes: List<&2, U32>, +valid: {True{} == Layout.bijective(input, axes) : Bool}, +other: U32,
  +inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool},
  -fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Permuted(input, values, axes, fits, pair => {Observation.kept_value_pair(input, other, pair) ==
    List.get(&2, F32, values, U32.to_nat(other)) : Maybe<&2, F32>}):
  permute_lift(input, values, axes, valid, pair => {Observation.kept_value_pair(input, other, pair) ==
      List.get(&2, F32, values, U32.to_nat(other)) : Maybe<&2, F32>},
    plan => source_fits => target_fits =>
      +total = Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))
      +history = FromValues.writes(values, 0n)
      Equal.trans(Maybe<&2, F32>,
        Core.read_value(input, Core.read_flat(input, Core.TTensor{kept_source(input, 0.0, FromValues.writes(values, 0n)), source_fits}, other)),
        Some{Replayed.value(input, 0.0, history, other)}, List.get(&2, F32, values, U32.to_nat(other)),
        Equal.trans(Maybe<&2, F32>, Core.read_value(input, Core.read_flat(input, Core.TTensor{kept_source(input, 0.0, history), source_fits}, other)),
          Some{Replayed.value(input, 0.0, Replayed.snoc(history, StorageLaws.First{}), other)}, Some{Replayed.value(input, 0.0, history, other)},
          Replayed.read_found(input, 0.0, Replayed.snoc(history, StorageLaws.First{}), source_fits, other, inside),
          Equal.cong(F32, Maybe<&2, F32>, word => Some{word}, Replayed.value(input, 0.0, Replayed.snoc(history, StorageLaws.First{}), other),
            Replayed.value(input, 0.0, history, other), half_keeps(input, 0.0, history, other))),
        IndexMap.stored_at(input, values, counted, other, inside)), fits)

def permute_contract(+input: Shape.Shape, +values: List<&2, F32>,
  counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  +axes: List<&2, U32>, valid: {True{} == Layout.bijective(input, axes) : Bool}) -> Records.PermuteContract<input, values, axes>:
  +counted = counted
  +valid = valid
  Records.PermuteContract{
    FromValues.from_values_lift(input, values, counted,
      result => {Observation.failure_code(input, Observation.Permute{axes}, result) == Some{0} : Maybe<&2, U32>},
      fits => permuted_code(input, values, axes, valid, fits)),
    FromValues.from_values_lift(input, values, counted,
      result => {Observation.applied_dimensions(input, Observation.Permute{axes}, result) ==
        Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))} : Maybe<&2, List<&2, U32>>},
      fits => permuted_dimensions(input, values, axes, valid, fits)),
    FromValues.from_values_lift(input, values, counted,
      result => {Observation.applied_length(input, Observation.Permute{axes}, result) ==
        Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>},
      fits => permuted_length(input, values, axes, valid, fits)),
    FromValues.from_values_lift(input, values, counted,
      result => {Observation.applied_count(input, Observation.Permute{axes}, result) == Some{Shape.count(input)} : Maybe<&2, U32>},
      fits => permuted_count(input, values, axes, valid, fits)),
    index => inside =>
      +inside = inside
      FromValues.from_values_lift(input, values, counted,
        result => {Observation.applied_value(input, Observation.Permute{axes}, index, result) ==
          Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>},
        fits => permuted_value(input, values, counted, axes, valid, index, inside, fits)),
    other => inside =>
      +inside = inside
      FromValues.from_values_lift(input, values, counted,
        result => {Observation.kept_value(input, Observation.Permute{axes}, other, result) ==
          List.get(&2, F32, values, U32.to_nat(other)) : Maybe<&2, F32>},
        fits => permuted_kept(input, values, counted, axes, valid, other, inside, fits))}

def requested(+input: Shape.Shape, +a: U32, +b: U32) -> List<&2, U32>:
  Transpose.swapped_axes(input, a, b)

def transpose_route(+input: Shape.Shape, +a: U32, +b: U32, source: Core.Tensor<input>,
  decision: Shape.BoolCase<Transpose.valid(input, a, b)>, +inside: {True{} == Transpose.valid(input, a, b) : Bool}) ->
  {Transforms.permute_copy(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)) == Transforms.transpose_decided(input, source, a, b, decision) :
    Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({Transforms.permute_copy(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)) ==
        (source, Fail{Transforms.LayoutError{Layout.AxisOutOfBounds{}}}) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>},
        Arith.clash(Equal.trans(Bool, False{}, Transpose.valid(input, a, b), True{}, equal, Equal.sym(Bool, True{}, Transpose.valid(input, a, b), inside))))
    case Shape.BTrue{equal}:
      {==}

def requested_valid(+input: Shape.Shape, +a: U32, +b: U32, +inside: {True{} == Transpose.valid(input, a, b) : Bool}) ->
  {True{} == Layout.bijective(input, requested(input, a, b)) : Bool}:
  %TransposeProof.transpose_axes_requested(input, a, b, inside) : {True{} == Layout.bijective(input, _) : Bool}
  TransposeProof.valid_axes(input, a, b, inside)

# transpose_copy of any source with both axes in range is permute_copy by
# the swapped axes.
def transpose_cast(+input: Shape.Shape, +a: U32, +b: U32, +inside: {True{} == Transpose.valid(input, a, b) : Bool},
  source: Core.Tensor<input>, -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  h: Q(Transforms.permute_copy(input, source, requested(input, a, b)))) -> Q(Transforms.transpose_copy(input, source, a, b)):
  %transpose_route(input, a, b, source, Shape.bool_case(Transpose.valid(input, a, b)), inside) : Q(_)
  %Equal.sym(List<&2, U32>, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), requested(input, a, b),
    TransposeProof.transpose_axes_requested(input, a, b, inside)) :
    Q(Transforms.permute_copy(input, source, _))
  h

def transpose_lift(+input: Shape.Shape, +values: List<&2, F32>, +a: U32, +b: U32, +inside: {True{} == Transpose.valid(input, a, b) : Bool},
  -fits: {Lifecycle.logical_length(Replayed.storage(input, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  -Q: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  h: Permuted(input, values, requested(input, a, b), fits, Q)) ->
  Q(Transforms.transpose_copy(input, Core.TTensor{Replayed.storage(input, 0.0, FromValues.writes(values, 0n)), fits}, a, b)):
  transpose_cast(input, a, b, inside, Core.TTensor{Replayed.storage(input, 0.0, FromValues.writes(values, 0n)), fits}, Q, h)

def transpose_contract(+input: Shape.Shape, +values: List<&2, F32>,
  counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  +a: U32, +b: U32, inside: {True{} == Transpose.valid(input, a, b) : Bool}) -> Records.TransposeContract<input, values, a, b>:
  +counted = counted
  +inside = inside
  +axes = requested(input, a, b)
  +valid = requested_valid(input, a, b, inside)
  Records.TransposeContract{
    FromValues.from_values_lift(input, values, counted,
      result => {Observation.failure_code(input, Observation.Transpose{a, b}, result) == Some{0} : Maybe<&2, U32>},
      fits => transpose_lift(input, values, a, b, inside, fits, pair => {Some{Observation.failure_code_pair(input, pair)} == Some{0} : Maybe<&2, U32>},
        permuted_code(input, values, axes, valid, fits))),
    FromValues.from_values_lift(input, values, counted,
      result => {Observation.applied_dimensions(input, Observation.Transpose{a, b}, result) ==
        Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))} : Maybe<&2, List<&2, U32>>},
      fits => transpose_lift(input, values, a, b, inside, fits, pair => {Observation.applied_dimensions_pair(input, pair) ==
          Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))} : Maybe<&2, List<&2, U32>>},
        permuted_dimensions(input, values, axes, valid, fits))),
    FromValues.from_values_lift(input, values, counted,
      result => {Observation.applied_length(input, Observation.Transpose{a, b}, result) ==
        Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>},
      fits => transpose_lift(input, values, a, b, inside, fits, pair => {Observation.applied_length_pair(input, pair) ==
          Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>},
        permuted_length(input, values, axes, valid, fits))),
    FromValues.from_values_lift(input, values, counted,
      result => {Observation.applied_count(input, Observation.Transpose{a, b}, result) == Some{Shape.count(input)} : Maybe<&2, U32>},
      fits => transpose_lift(input, values, a, b, inside, fits, pair => {Observation.applied_count_pair(input, pair) == Some{Shape.count(input)} : Maybe<&2, U32>},
        permuted_count(input, values, axes, valid, fits))),
    index => below =>
      +below = below
      FromValues.from_values_lift(input, values, counted,
        result => {Observation.applied_value(input, Observation.Transpose{a, b}, index, result) ==
          Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>},
        fits => transpose_lift(input, values, a, b, inside, fits, pair => {Observation.applied_value_pair(input, index, pair) ==
            Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>},
          permuted_value(input, values, counted, axes, valid, index, below, fits))),
    other => below =>
      +below = below
      FromValues.from_values_lift(input, values, counted,
        result => {Observation.kept_value(input, Observation.Transpose{a, b}, other, result) ==
          List.get(&2, F32, values, U32.to_nat(other)) : Maybe<&2, F32>},
        fits => transpose_lift(input, values, a, b, inside, fits, pair => {Observation.kept_value_pair(input, other, pair) ==
            List.get(&2, F32, values, U32.to_nat(other)) : Maybe<&2, F32>},
          permuted_kept(input, values, counted, axes, valid, other, below, fits)))}

# The words the copy loop writes, in index order, are `permuted_words`.
def words_moved(+input: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  +count: Nat, +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool}) ->
  {Contracts.words(Layout.output(input, axes, plan), 0.0,
    IndexMap.writes(~Request, ~source_of, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))), Request{input, axes}, input, fill, ops),
    count) == Records.permuted_words(input, fill, ops, axes, count) : List<&2, F32>}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +output = Layout.output(input, axes, plan)
      +total = Shape.logical_numel(Shape.dimensions(output))
      +request : Request = Request{input, axes}
      +history = IndexMap.writes(~Request, ~source_of, total, request, input, fill, ops)
      +below = Conversion.le_successor_lt(previous, total, room)
      +position = U32.from_nat(previous)
      Words.snoc_equal(Contracts.words(output, 0.0, history, previous), Records.permuted_words(input, fill, ops, axes, previous),
        Replayed.value(output, 0.0, history, position),
        Replayed.value(input, fill, ops, U32.from_nat(Values.permute_source_index(input, axes, previous))),
        words_moved(input, fill, ops, axes, plan, previous, Conversion.le_predecessor(previous, total, room)),
        Equal.trans(F32, Replayed.value(output, 0.0, history, position),
          Replayed.value(input, fill, ops, U32.from_nat(Values.permute_source_index(input, axes, U32.to_nat(position)))),
          Replayed.value(input, fill, ops, U32.from_nat(Values.permute_source_index(input, axes, previous))),
          PointwiseProof.writes_value(~Unit, ~IndexMap.keep, ~Request, ~source_of, ~source_of, total, Unit{}, request, input, fill, ops,
            input, fill, ops, output, position, PointwiseProof.word_inside(previous, output, below), Conversion.le_reflexive(total)),
          Equal.cong(Nat, F32, number => Replayed.value(input, fill, ops, U32.from_nat(Values.permute_source_index(input, axes, number))),
            U32.to_nat(position), previous, PointwiseProof.word_exact(previous, output, below))))

# The copy loop on any replayed source, in transport form: once
# `Layout.permutation` gives the plan, P holds for the permute if it holds for
# every result of the form the loop's final state has.
def replayed_final(+input: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  -P: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  h: Records.PermutedWith(input, fill, ops, axes, plan, P)) ->
  Final(input, axes, plan, fill, ops, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))),
    pair => P(Transforms.dynamic_result(input, Transforms.copy_pack(input, axes, plan, pair)))):
  +output = Layout.output(input, axes, plan)
  +total = Shape.logical_numel(Shape.dimensions(output))
  +request : Request = Request{input, axes}
  source_fits => target_fits =>
    h(Replayed.snoc(ops, StorageLaws.First{}), words_first(input, fill, ops, Shape.logical_numel(Shape.dimensions(input))),
      source_fits,
      IndexMap.writes(~Request, ~source_of, total, request, input, fill, ops),
      words_moved(input, fill, ops, axes, plan, total, Conversion.le_reflexive(total)),
      target_fits)

def permute_replayed(+input: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(Replayed.storage(input, fill, ops)) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  laid: {Layout.permutation(input, axes) == Done{plan} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>},
  -P: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  h: Records.PermutedWith(input, fill, ops, axes, plan, P)) ->
  P(Transforms.dynamic_result(input, Transforms.permute_copy(input, Core.TTensor{Replayed.storage(input, fill, ops), fits}, axes))):
  %Equal.sym(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, Layout.permutation(input, axes), Done{plan}, laid) :
    P(Transforms.dynamic_result(input, Transforms.copy_planned(input, axes, Core.TTensor{Replayed.storage(input, fill, ops), fits}, _)))
  run_lift(input, axes, plan, fill, ops, pair => P(Transforms.dynamic_result(input, pair)),
    replayed_final(input, fill, ops, axes, plan, P, h), fits)

def transpose_replayed(+input: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(Replayed.storage(input, fill, ops)) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  +a: U32, +b: U32, inside: {True{} == Transpose.valid(input, a, b) : Bool},
  +plan: Layout.Plan<input, Transpose.swapped_axes(input, a, b)>,
  laid: {Layout.permutation(input, Transpose.swapped_axes(input, a, b)) == Done{plan} :
    Result<&2, &2, Layout.Error, Layout.Plan<input, Transpose.swapped_axes(input, a, b)>>},
  -P: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  h: Records.PermutedWith(input, fill, ops, Transpose.swapped_axes(input, a, b), plan, P)) ->
  P(Transforms.dynamic_result(input, Transforms.transpose_copy(input, Core.TTensor{Replayed.storage(input, fill, ops), fits}, a, b))):
  +inside = inside
  transpose_cast(input, a, b, inside, Core.TTensor{Replayed.storage(input, fill, ops), fits}, pair => P(Transforms.dynamic_result(input, pair)),
    permute_replayed(input, fill, ops, fits, requested(input, a, b), plan, laid, P, h))
