import Base
import ../shape/core.bend as Shape
import ../shape/LAWS.bend as ShapeLaws
import ../shape/PROOF.bend as ShapeProof
import ../shape/nat_proofs.bend as Nats
import ../storage/main.bend as Storage
import ../storage/LAWS.bend as StorageLaws
import ../storage/PROOF.bend as StorageProof
import ../storage/arith.bend as Arith
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./layout.bend as Layout
import ./transpose-spec.bend as Transpose
import ./transforms.bend as Transforms
import ./allocation-PROOF.bend as AllocationProof
import ./word-conversion.bend as Conversion
import ./read-contract.bend as Reads
import ./transform-observation.bend as Observation
import ./transform-records.bend as Laws
import ./write-contract.bend as Writes

def shape_complete(+shape: Shape.Shape) ->
  {ShapeLaws.built(Shape.dimensions(shape), Shape.construct(Shape.dimensions(shape))) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>}:
  match shape:
    case Shape.SShape{+dimensions, size, exact, rank_bound, capacity_bound}:
      ShapeLaws.construction_complete(dimensions, rank_bound, capacity_bound)

def inside_input(+input: Shape.Shape, +target: Shape.Shape, +other: U32, same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}, inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}) -> {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool}:
  %Equal.sym(Nat, Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(target)), same) : {Nat.is_lt(U32.to_nat(other), _) == True{} : Bool}
  inside

def elements_output(+input: Shape.Shape, +target: Shape.Shape, +output: Shape.Shape, same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat},
  preserved: {Shape.dimensions(output) == Shape.dimensions(target) : List<&2, U32>}) ->
  {Shape.elements(input) == Shape.elements(output) : Nat}:
  Equal.trans(Nat, Shape.elements(input), Shape.logical_numel(Shape.dimensions(input)), Shape.elements(output), Shape.elements_exact(input),
    Equal.trans(Nat, Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(target)), Shape.elements(output), same,
      Equal.trans(Nat, Shape.logical_numel(Shape.dimensions(target)), Shape.logical_numel(Shape.dimensions(output)), Shape.elements(output),
        Equal.cong(List<&2, U32>, Nat, dims => Shape.logical_numel(dims), Shape.dimensions(target), Shape.dimensions(output),
          Equal.sym(List<&2, U32>, Shape.dimensions(output), Shape.dimensions(target), preserved)),
        Equal.sym(Nat, Shape.elements(output), Shape.logical_numel(Shape.dimensions(output)), Shape.elements_exact(output)))))

def decided_lift(+input: Shape.Shape, +output: Shape.Shape, -storage: Storage.Storage,
  -valid: {Lifecycle.logical_length(storage) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  decision: Shape.BoolCase<Nat.is_eq(Shape.elements(input), Shape.elements(output))>,
  counted: {Shape.elements(input) == Shape.elements(output) : Nat},
  -P: Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  h: @-fits: {Lifecycle.logical_length(storage) == Shape.logical_numel(Shape.dimensions(output)) : Nat} -> P(Done{Core.Packed{output, Core.TTensor{storage, fits}}})) ->
  P(Transforms.reshape_decided(input, output, Core.TTensor{storage, valid}, decision)):
  match decision:
    case Shape.BFalse{no}:
      Empty.absurd(P(Transforms.reshape_decided(input, output, Core.TTensor{storage, valid}, Shape.BFalse{no})),
        Nats.contradiction(Nat.is_eq(Shape.elements(input), Shape.elements(output)),
          AllocationProof.natural_equal_true(Shape.elements(input), Shape.elements(output), counted), no))
    case Shape.BTrue{yes}:
      h(Equal.trans(Nat, Lifecycle.logical_length(storage), Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(output)), valid, Transforms.elements_equal(input, output, yes)))

def built_lift(+input: Shape.Shape, +target: Shape.Shape, -storage: Storage.Storage,
  -valid: {Lifecycle.logical_length(storage) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  built: Shape.ConstructResult<Shape.dimensions(target)>,
  complete: {ShapeLaws.built(Shape.dimensions(target), built) == Some{Shape.dimensions(target)} : Maybe<&2, List<&2, U32>>},
  +same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}, -P: Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  h: @+output: Shape.Shape -> @preserved: {Shape.dimensions(output) == Shape.dimensions(target) : List<&2, U32>} ->
    @-fits: {Lifecycle.logical_length(storage) == Shape.logical_numel(Shape.dimensions(output)) : Nat} -> P(Done{Core.Packed{output, Core.TTensor{storage, fits}}})) ->
  P(Transforms.reshape_built(input, Core.TTensor{storage, valid}, Shape.dimensions(target), built)):
  match built:
    case Shape.SError{error}:
      Empty.absurd(P(Transforms.reshape_built(input, Core.TTensor{storage, valid}, Shape.dimensions(target), Shape.SError{error})),
        Arith.clash(Equal.cong(Maybe<&2, List<&2, U32>>, Bool, found => Maybe.is_some(&2, List<&2, U32>, found), None{}, Some{Shape.dimensions(target)}, complete)))
    case Shape.SBuilt{+output, +preserved}:
      decided_lift(input, output, storage, valid, Shape.bool_case(Nat.is_eq(Shape.elements(input), Shape.elements(output))),
        elements_output(input, target, output, same, preserved), P, h(output, preserved))

def reshape_lift(+input: Shape.Shape, +target: Shape.Shape, -storage: Storage.Storage,
  -valid: {Lifecycle.logical_length(storage) == Shape.logical_numel(Shape.dimensions(input)) : Nat}, +same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}, -P: Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type,
  h: @+output: Shape.Shape -> @preserved: {Shape.dimensions(output) == Shape.dimensions(target) : List<&2, U32>} ->
    @-fits: {Lifecycle.logical_length(storage) == Shape.logical_numel(Shape.dimensions(output)) : Nat} -> P(Done{Core.Packed{output, Core.TTensor{storage, fits}}})) ->
  P(Transforms.reshape(input, Core.TTensor{storage, valid}, Shape.dimensions(target))):
  built_lift(input, target, storage, valid, Shape.construct(Shape.dimensions(target)), shape_complete(target), same, P, h)

def dynamic_value(+output: Shape.Shape, -storage: Storage.Storage,
  -fits: {Lifecycle.logical_length(storage) == Shape.logical_numel(Shape.dimensions(output)) : Nat}, +other: U32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>, -equal: {pair == Storage.get(storage, other) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}) ->
  {Core.dynamic_read_value(Core.read_dynamic_result(output, Core.read_result(output, storage, fits, other, pair, equal))) ==
    Lifecycle.read_value(pair) : Maybe<&2, F32>}:
  (kept, result) = pair
  match result:
    case Fail{error}:
      {==}
    case Done{value}:
      {==}

def reshaped_value(+input: Shape.Shape, +target: Shape.Shape, storage: Storage.Storage,
  -valid: {Lifecycle.logical_length(storage) == Shape.logical_numel(Shape.dimensions(input)) : Nat}, +other: U32, +same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}) ->
  {Observation.output_value(other, Transforms.reshape(input, Core.TTensor{storage, valid}, Shape.dimensions(target))) ==
    Lifecycle.read_value(Storage.get(storage, other)) : Maybe<&2, F32>}:
  reshape_lift(input, target, storage, valid, same,
    result => {Observation.output_value(other, result) == Lifecycle.read_value(Storage.get(storage, other)) : Maybe<&2, F32>},
    output => preserved => fits => dynamic_value(output, storage, fits, other, Storage.get(storage, other), {==}))

def reshape_outcome(+input: Shape.Shape, +fill: F32, +target: Shape.Shape, +same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}) ->
  {Observation.reshape_code(input, Shape.dimensions(target), Core.new(input, fill)) == Some{0} : Maybe<&2, U32>}:
  Reads.new_lift(input, fill, result => {Observation.reshape_code(input, Shape.dimensions(target), result) == Some{0} : Maybe<&2, U32>},
    valid => reshape_lift(input, target, StorageLaws.created(Shape.count(input), fill), valid, same,
        result => {Some{Observation.status(result)} == Some{0} : Maybe<&2, U32>},
        output => preserved => fits => {==}))

def reshape_dimensions(+input: Shape.Shape, +fill: F32, +target: Shape.Shape, +same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}) ->
  {Observation.reshape_dimensions(input, Shape.dimensions(target), Core.new(input, fill)) == Some{Shape.dimensions(target)} : Maybe<&2, List<&2, U32>>}:
  Reads.new_lift(input, fill, result => {Observation.reshape_dimensions(input, Shape.dimensions(target), result) == Some{Shape.dimensions(target)} : Maybe<&2, List<&2, U32>>},
    valid => reshape_lift(input, target, StorageLaws.created(Shape.count(input), fill), valid, same,
        result => {Observation.output_dimensions(result) == Some{Shape.dimensions(target)} : Maybe<&2, List<&2, U32>>},
        output => preserved => fits => Equal.cong(List<&2, U32>, Maybe<&2, List<&2, U32>>, dims => Some{dims}, Shape.dimensions(output), Shape.dimensions(target), preserved)))

def reshape_length(+input: Shape.Shape, +fill: F32, +target: Shape.Shape, +same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}) ->
  {Observation.reshape_length(input, Shape.dimensions(target), Core.new(input, fill)) == Some{Shape.logical_numel(Shape.dimensions(target))} : Maybe<&2, Nat>}:
  Reads.new_lift(input, fill, result => {Observation.reshape_length(input, Shape.dimensions(target), result) == Some{Shape.logical_numel(Shape.dimensions(target))} : Maybe<&2, Nat>},
    valid => reshape_lift(input, target, StorageLaws.created(Shape.count(input), fill), valid, same,
        result => {Observation.output_length(result) == Some{Shape.logical_numel(Shape.dimensions(target))} : Maybe<&2, Nat>},
        output => preserved => fits => Equal.cong(Nat, Maybe<&2, Nat>, number => Some{number}, Lifecycle.logical_length(StorageLaws.created(Shape.count(input), fill)), Shape.logical_numel(Shape.dimensions(target)),
          Equal.trans(Nat, Lifecycle.logical_length(StorageLaws.created(Shape.count(input), fill)), Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(target)), Reads.created_exact(input, fill), same))))

def reshape_count(+input: Shape.Shape, +fill: F32, +target: Shape.Shape, +same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}) ->
  {Observation.reshape_count(input, Shape.dimensions(target), Core.new(input, fill)) == Some{Shape.count(target)} : Maybe<&2, U32>}:
  Reads.new_lift(input, fill, result => {Observation.reshape_count(input, Shape.dimensions(target), result) == Some{Shape.count(target)} : Maybe<&2, U32>},
    valid => reshape_lift(input, target, StorageLaws.created(Shape.count(input), fill), valid, same,
        result => {Observation.output_count(result) == Some{Shape.count(target)} : Maybe<&2, U32>},
        output => preserved => fits => Equal.cong(Nat, Maybe<&2, U32>, number => Some{U32.from_nat(number)}, Shape.elements(output), Shape.elements(target),
          Equal.trans(Nat, Shape.elements(output), Shape.elements(input), Shape.elements(target),
            Equal.sym(Nat, Shape.elements(input), Shape.elements(output), elements_output(input, target, output, same, preserved)),
            Equal.trans(Nat, Shape.elements(input), Shape.logical_numel(Shape.dimensions(input)), Shape.elements(target), Shape.elements_exact(input),
              Equal.trans(Nat, Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(target)), Shape.elements(target), same, Equal.sym(Nat, Shape.elements(target), Shape.logical_numel(Shape.dimensions(target)), Shape.elements_exact(target))))))))

def reshape_value_done(+input: Shape.Shape, +fill: F32, +target: Shape.Shape, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(input), fill)) == Shape.logical_numel(Shape.dimensions(input)) : Nat}, +other: U32, +same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}, inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}) ->
  {Observation.reshape_value(input, Shape.dimensions(target), other, Done{Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}}) == Some{fill} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>, Observation.output_value(other, Transforms.reshape(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, Shape.dimensions(target))),
    Lifecycle.read_value(Storage.get(StorageLaws.created(Shape.count(input), fill), other)), Some{fill},
    reshaped_value(input, target, StorageLaws.created(Shape.count(input), fill), valid, other, same),
    Reads.stored_value(StorageLaws.created(Shape.count(input), fill), other, fill,
      StorageLaws.replay_get(Shape.count(input), fill, [], other, Conversion.shape_count_within(input),
        Reads.word_inside(input, other, inside_input(input, target, other, same, inside)))))

def written_projection(+input: Shape.Shape, +target: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(input)) : Nat}, +index: U32, +value: F32, +other: U32, +same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat},
  pair: Storage.Storage & Result<&2, &2, Storage.Error, Unit>,
  -equal: {pair == Storage.set(original, index, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}) ->
  {Observation.reshape_written_pair(input, Shape.dimensions(target), other, Core.write_result(input, original, valid, index, value, pair, equal)) ==
    Lifecycle.read_value(Storage.get(StorageLaws.owner(Result<&2, &2, Storage.Error, Unit>, pair), other)) : Maybe<&2, F32>}:
  (storage, result) = pair
  reshaped_value(input, target, storage,
    Equal.trans(Nat, Lifecycle.logical_length(storage), Lifecycle.logical_length(original), Shape.logical_numel(Shape.dimensions(input)),
      %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, Unit>, (storage, result), Storage.set(original, index, value), equal) :
        {Lifecycle.retained_write(_) == Lifecycle.logical_length(original) : Nat}
      Lifecycle.set_length(original, index, value), valid),
    other, same)

def written_done(+input: Shape.Shape, +fill: F32, +target: Shape.Shape, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(input), fill)) == Shape.logical_numel(Shape.dimensions(input)) : Nat}, +index: U32, +value: F32, +other: U32, +same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}, inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}) ->
  {Observation.reshape_written(input, index, value, Shape.dimensions(target), other, Done{Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}}) ==
    Some{Writes.written_pick(input, index, value, other, fill)} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>,
    Observation.reshape_written_pair(input, Shape.dimensions(target), other,
      Core.write_result(input, StorageLaws.created(Shape.count(input), fill), valid, index, value, Storage.set(StorageLaws.created(Shape.count(input), fill), index, value), {==})),
    Lifecycle.read_value(Storage.get(StorageLaws.written(StorageLaws.created(Shape.count(input), fill), index, value), other)),
    Some{Writes.written_pick(input, index, value, other, fill)},
    written_projection(input, target, StorageLaws.created(Shape.count(input), fill), valid, index, value, other, same, Storage.set(StorageLaws.created(Shape.count(input), fill), index, value), {==}),
    Reads.stored_value(StorageLaws.written(StorageLaws.created(Shape.count(input), fill), index, value), other, Writes.written_pick(input, index, value, other, fill),
      StorageLaws.replay_get(Shape.count(input), fill, [StorageLaws.Write{index, value}], other, Conversion.shape_count_within(input),
        Reads.word_inside(input, other, inside_input(input, target, other, same, inside)))))

def reshape_written(+input: Shape.Shape, +fill: F32, +target: Shape.Shape, +index: U32, +value: F32, +other: U32, -expected: F32,
  +same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}, inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}, picked: {Writes.written_pick(input, index, value, other, fill) == expected : F32}) ->
  {Observation.reshape_written(input, index, value, Shape.dimensions(target), other, Core.new(input, fill)) == Some{expected} : Maybe<&2, F32>}:
  Reads.new_lift(input, fill, result => {Observation.reshape_written(input, index, value, Shape.dimensions(target), other, result) == Some{expected} : Maybe<&2, F32>},
    valid => Equal.trans(Maybe<&2, F32>, Observation.reshape_written(input, index, value, Shape.dimensions(target), other, Done{Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}}),
        Some{Writes.written_pick(input, index, value, other, fill)}, Some{expected},
        written_done(input, fill, target, valid, index, value, other, same, inside),
        Equal.cong(F32, Maybe<&2, F32>, number => Some{number}, Writes.written_pick(input, index, value, other, fill), expected, picked)))

def target_inside(+input: Shape.Shape, +target: Shape.Shape, +index: U32, same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat},
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool}) -> {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}:
  %same : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
  inside

def reshape_contract(+input: Shape.Shape, +fill: F32, +target: Shape.Shape, same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}) -> Laws.ReshapeContract<input, fill, target>:
  +same = same
  Laws.ReshapeContract{
    reshape_outcome(input, fill, target, same),
    reshape_dimensions(input, fill, target, same),
    reshape_length(input, fill, target, same),
    reshape_count(input, fill, target, same),
    other => inside => Reads.new_lift(input, fill, result => {Observation.reshape_value(input, Shape.dimensions(target), other, result) == Some{fill} : Maybe<&2, F32>},
      valid => reshape_value_done(input, fill, target, valid, other, same, inside))}

def reshape_write(+input: Shape.Shape, +fill: F32, +target: Shape.Shape, +index: U32, +value: F32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool}, same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}) -> Laws.ReshapeWrite<input, fill, target, index, value>:
  +inside = inside
  +same = same
  Laws.ReshapeWrite{
    reshape_written(input, fill, target, index, value, index, value, same, target_inside(input, target, index, same, inside),
      Writes.pick_hit(input, index, value, fill, inside)),
    other => inside_other => different =>
      reshape_written(input, fill, target, index, value, other, fill, same, inside_other,
        Writes.pick_frame(input, index, value, other, fill, inside, different))}

def mismatch_decided(+input: Shape.Shape, +target: Shape.Shape, +output: Shape.Shape, -tensor: Core.Tensor<input>,
  decision: Shape.BoolCase<Nat.is_eq(Shape.elements(input), Shape.elements(output))>,
  preserved: {Shape.dimensions(output) == Shape.dimensions(target) : List<&2, U32>},
  different: {Nat.is_eq(Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(target))) == False{} : Bool}) ->
  {Transforms.reshape_decided(input, output, tensor, decision) == Fail{Transforms.LayoutError{Layout.ElementCountMismatch{}}} : Result<&2, &1, Transforms.Error, Core.Dynamic>}:
  match decision:
    case Shape.BFalse{no}:
      {==}
    case Shape.BTrue{yes}:
      Empty.absurd({Transforms.reshape_decided(input, output, tensor, Shape.BTrue{yes}) == Fail{Transforms.LayoutError{Layout.ElementCountMismatch{}}} : Result<&2, &1, Transforms.Error, Core.Dynamic>},
        Nats.contradiction(Nat.is_eq(Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(target))),
          AllocationProof.natural_equal_true(Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(target)),
            Equal.trans(Nat, Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(output)), Shape.logical_numel(Shape.dimensions(target)), Transforms.elements_equal(input, output, yes),
              Equal.cong(List<&2, U32>, Nat, dims => Shape.logical_numel(dims), Shape.dimensions(output), Shape.dimensions(target), preserved))),
          Equal.sym(Bool, Nat.is_eq(Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(target))), False{}, different)))

def mismatch_built(+input: Shape.Shape, +target: Shape.Shape, -tensor: Core.Tensor<input>,
  built: Shape.ConstructResult<Shape.dimensions(target)>,
  complete: {ShapeLaws.built(Shape.dimensions(target), built) == Some{Shape.dimensions(target)} : Maybe<&2, List<&2, U32>>},
  different: {Nat.is_eq(Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(target))) == False{} : Bool}) ->
  {Transforms.reshape_built(input, tensor, Shape.dimensions(target), built) == Fail{Transforms.LayoutError{Layout.ElementCountMismatch{}}} : Result<&2, &1, Transforms.Error, Core.Dynamic>}:
  match built:
    case Shape.SError{error}:
      Empty.absurd({Transforms.reshape_built(input, tensor, Shape.dimensions(target), Shape.SError{error}) == Fail{Transforms.LayoutError{Layout.ElementCountMismatch{}}} : Result<&2, &1, Transforms.Error, Core.Dynamic>},
        Arith.clash(Equal.cong(Maybe<&2, List<&2, U32>>, Bool, found => Maybe.is_some(&2, List<&2, U32>, found), None{}, Some{Shape.dimensions(target)}, complete)))
    case Shape.SBuilt{+output, preserved}:
      mismatch_decided(input, target, output, tensor, Shape.bool_case(Nat.is_eq(Shape.elements(input), Shape.elements(output))), preserved, different)

def reshape_count_mismatch(+input: Shape.Shape, +fill: F32, +target: Shape.Shape,
  different: {Nat.is_eq(Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(target))) == False{} : Bool}) ->
  {Observation.reshape_code(input, Shape.dimensions(target), Core.new(input, fill)) == Some{15} : Maybe<&2, U32>}:
  +different = different
  Reads.new_lift(input, fill, result => {Observation.reshape_code(input, Shape.dimensions(target), result) == Some{15} : Maybe<&2, U32>},
    valid => Equal.cong(Result<&2, &1, Transforms.Error, Core.Dynamic>, Maybe<&2, U32>, result => Some{Observation.status(result)},
        Transforms.reshape(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, Shape.dimensions(target)), Fail{Transforms.LayoutError{Layout.ElementCountMismatch{}}},
        mismatch_built(input, target, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, Shape.construct(Shape.dimensions(target)), shape_complete(target), different)))
