import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ../storage/clone.bend as Clone
import ../storage/PROOF.bend as StorageProof
import ../storage/LAWS.bend as StorageLaws
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./word-injective.bend as Injective
import ./elementwise.bend as Elementwise
import ./core-LAWS.bend as Laws
import ./projection-LAWS.bend as Projection
import ./projection-PROOF.bend as ProjectionProof

def Laws.scalar_add(left, right):
  {==}

def Laws.scalar_sub(left, right):
  {==}

def Laws.scalar_mul(left, right):
  {==}

def Laws.indexed_length(shape, tensor):
  Core.TTensor{storage, exact} = tensor
  Core.LengthValid{exact}

def construction_values_dimensions(+dimensions: List<&2, U32>, +shape: Shape.Shape,
  preserved: {Shape.dimensions(shape) == dimensions : List<&2, U32>},
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) ->
  Core.ConstructionDimensions(dimensions, Core.construct_values(shape, result)):
  match result:
    case Fail{error}:
      Unit{}
    case Done{tensor}:
      preserved

def construction_shape_dimensions(+dimensions: List<&2, U32>, values: List<&2, F32>, result: Shape.ConstructResult<dimensions>) ->
  Core.ConstructionDimensions(dimensions, Core.construct_shape(dimensions, values, result)):
  match result:
    case Shape.SError{error}:
      Unit{}
    case Shape.SBuilt{+shape, preserved}:
      construction_values_dimensions(dimensions, shape, preserved, Core.from_values(shape, values))

def Laws.successful_construction_dimensions(dimensions, values):
  construction_shape_dimensions(dimensions, values, Shape.construct(dimensions))

def count_outcome(+shape: Shape.Shape, +values: List<&2, F32>,
  equal: {Shape.logical_numel(Shape.dimensions(shape)) == List.length(&2, F32, values) : Nat},
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Core.ValuesCount(shape, values, result):
  match result:
    case Fail{error}:
      Unit{}
    case Done{tensor}:
      equal

def count_decision(+shape: Shape.Shape, +values: List<&2, F32>,
  decision: Shape.BoolCase<Nat.is_eq(List.length(&2, F32, values), Shape.elements(shape))>) ->
  Core.ValuesCount(shape, values, Core.from_values_counted(shape, values, decision)):
  match decision:
    case Shape.BFalse{equal}:
      Unit{}
    case Shape.BTrue{equal}:
      count_outcome(shape, values,
        Equal.trans(Nat, Shape.logical_numel(Shape.dimensions(shape)), Shape.elements(shape), List.length(&2, F32, values),
          Equal.sym(Nat, Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)), Shape.elements_exact(shape)),
          Equal.sym(Nat, List.length(&2, F32, values), Shape.elements(shape),
            Lifecycle.natural_equal(List.length(&2, F32, values), Shape.elements(shape), equal))),
        Core.from_values_new(shape, values, Core.new(shape, 0.0)))

def Laws.successful_value_count(shape, values):
  count_decision(shape, values, Shape.bool_case(Nat.is_eq(List.length(&2, F32, values), Shape.elements(shape))))

def Laws.checked_reader_pair(shape, original, index):
  Core.read_certificate(shape, original, index)

def Laws.reader_certificate_value(shape, original, index, certificate):
  Core.Read{tensor, result, exact} = certificate
  %exact : {Core.read_value(shape, _) == Core.result_value(result) : Maybe<&2, F32>}
  {==}

def Laws.checked_reader_source(shape, original, index):
  Core.source_read_model(shape, original, index)

def Laws.checked_dynamic_reader_source(original, index):
  Core.dynamic_source_read_model(original, index)

def Laws.checked_reader_retains_source(shape, original, index):
  Core.source_read_keeps(shape, original, index)

def Laws.checked_dynamic_reader_retains_source(original, index):
  Core.dynamic_source_read_keeps(original, index)

def Laws.primitive_read_retains_elements(depth, array, index, other, balanced):
  Equal.cong(Array<F32>, Maybe<&2, F32>, a => Model.observe(F32, depth, a, Refinement.route(depth, other)),
    Refinement.first(F32, Array.get(F32, array, index)), array,
    Refinement.law_get_retains(F32, depth, array, index, balanced))

def Laws.storage_read_retains_length(storage, index):
  Lifecycle.get_length(storage, index)

def Laws.storage_write_retains_length(storage, index, value):
  Lifecycle.set_length(storage, index, value)

def Laws.storage_clone_first_length(storage):
  Lifecycle.clone_first_length(storage)

def Laws.storage_clone_second_length(storage):
  Lifecycle.clone_second_length(storage)

def Laws.primitive_clone_first_elements(depth, array, path, balanced):
  Equal.cong(Array<F32>, Maybe<&2, F32>, a => Model.observe(F32, depth, a, path),
    Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, array)), array,
    Clone.first_equal(F32, depth, array, Clone.certify(F32, depth, array, balanced)))

def Laws.primitive_clone_second_elements(depth, array, path, balanced):
  Equal.cong(Array<F32>, Maybe<&2, F32>, a => Model.observe(F32, depth, a, path),
    Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, array)), array,
    Clone.second_equal(F32, depth, array, Clone.certify(F32, depth, array, balanced)))

def write_index_mapping(-shape: Shape.Shape, +d: Nat, +n: Nat, array: Array<F32>, +i: U32, +value: F32,
  balanced: Model.Balanced(F32, d, array), -valid: {Storage.bounds(d, n) == True{} : Bool},
  -exact: {n == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, matched: Bool,
  -decision: {matched == Storage.index_matches(d, n, i) : Bool},
  -same: {Storage.write_index(d, n, array, i, value, matched, decision, balanced, valid) ==
    Storage.set(Storage.Owned{d, n, array, balanced, valid}, i, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>},
  accepted: {matched == True{} : Bool}) ->
  {Core.write_observe(shape, Core.write_result(shape, Storage.Owned{d, n, array, balanced, valid}, exact, i, value,
    Storage.write_index(d, n, array, i, value, matched, decision, balanced, valid), same), i) == Some{value} : Maybe<&2, F32>}:
  match matched:
    case False{}:
      Empty.absurd({Core.write_observe(shape, Core.write_result(shape, Storage.Owned{d, n, array, balanced, valid}, exact, i, value,
        Storage.write_index(d, n, array, i, value, False{}, decision, balanced, valid), same), i) == Some{value} : Maybe<&2, F32>},
        Lifecycle.false_true(Equal.sym(Bool, False{}, True{}, accepted)))
    case True{}:
      StorageLaws.primitive_write_read(F32, d, array, i, value, balanced)

def write_checked_mapping(-shape: Shape.Shape, +d: Nat, +n: Nat, array: Array<F32>, +i: U32, +value: F32,
  balanced: Model.Balanced(F32, d, array), -valid: {Storage.bounds(d, n) == True{} : Bool},
  -exact: {n == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(i), n) : Bool},
  -same: {Storage.set_checked(d, n, array, i, value, allowed, decision, balanced, valid) ==
    Storage.set(Storage.Owned{d, n, array, balanced, valid}, i, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>},
  accepted: {allowed == True{} : Bool}, matched: {Storage.index_matches(d, n, i) == True{} : Bool}) ->
  {Core.write_observe(shape, Core.write_result(shape, Storage.Owned{d, n, array, balanced, valid}, exact, i, value,
    Storage.set_checked(d, n, array, i, value, allowed, decision, balanced, valid), same), i) == Some{value} : Maybe<&2, F32>}:
  match allowed:
    case False{}:
      Empty.absurd({Core.write_observe(shape, Core.write_result(shape, Storage.Owned{d, n, array, balanced, valid}, exact, i, value,
        Storage.set_checked(d, n, array, i, value, False{}, decision, balanced, valid), same), i) == Some{value} : Maybe<&2, F32>},
        Lifecycle.false_true(Equal.sym(Bool, False{}, True{}, accepted)))
    case True{}:
      write_index_mapping(shape, d, n, array, i, value, balanced, valid, exact,
        Storage.index_matches(d, n, i), {==}, same, matched)

def Laws.tensor_write_mapping(shape, depth, length, array, index, value, balanced, valid, exact, allowed, matched):
  write_checked_mapping(shape, depth, length, array, index, value, balanced, valid, exact,
    Nat.is_lt(U32.to_nat(index), length), {==}, {==}, allowed, matched)

def step_index_mapping(-shape: Shape.Shape, +d: Nat, +n: Nat, array: Array<F32>, +i: U32, +value: F32,
  -left: Core.Dynamic, -right: Core.Dynamic,
  balanced: Model.Balanced(F32, d, array), -valid: {Storage.bounds(d, n) == True{} : Bool},
  -exact: {n == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, matched: Bool,
  -decision: {matched == Storage.index_matches(d, n, i) : Bool},
  -same: {Storage.write_index(d, n, array, i, value, matched, decision, balanced, valid) ==
    Storage.set(Storage.Owned{d, n, array, balanced, valid}, i, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>},
  accepted: {matched == True{} : Bool}) ->
  {Elementwise.step_observe(shape, Elementwise.step_write(shape, left, right,
    Core.write_result(shape, Storage.Owned{d, n, array, balanced, valid}, exact, i, value,
      Storage.write_index(d, n, array, i, value, matched, decision, balanced, valid), same)), i) == Some{value} : Maybe<&2, F32>}:
  match matched:
    case False{}:
      Empty.absurd({Elementwise.step_observe(shape, Elementwise.step_write(shape, left, right,
        Core.write_result(shape, Storage.Owned{d, n, array, balanced, valid}, exact, i, value,
          Storage.write_index(d, n, array, i, value, False{}, decision, balanced, valid), same)), i) == Some{value} : Maybe<&2, F32>},
        Lifecycle.false_true(Equal.sym(Bool, False{}, True{}, accepted)))
    case True{}:
      StorageLaws.primitive_write_read(F32, d, array, i, value, balanced)

def step_checked_mapping(-shape: Shape.Shape, +d: Nat, +n: Nat, array: Array<F32>, +i: U32, +value: F32,
  -left: Core.Dynamic, -right: Core.Dynamic,
  balanced: Model.Balanced(F32, d, array), -valid: {Storage.bounds(d, n) == True{} : Bool},
  -exact: {n == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(i), n) : Bool},
  -same: {Storage.set_checked(d, n, array, i, value, allowed, decision, balanced, valid) ==
    Storage.set(Storage.Owned{d, n, array, balanced, valid}, i, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>},
  accepted: {allowed == True{} : Bool}, matched: {Storage.index_matches(d, n, i) == True{} : Bool}) ->
  {Elementwise.step_observe(shape, Elementwise.step_write(shape, left, right,
    Core.write_result(shape, Storage.Owned{d, n, array, balanced, valid}, exact, i, value,
      Storage.set_checked(d, n, array, i, value, allowed, decision, balanced, valid), same)), i) == Some{value} : Maybe<&2, F32>}:
  match allowed:
    case False{}:
      Empty.absurd({Elementwise.step_observe(shape, Elementwise.step_write(shape, left, right,
        Core.write_result(shape, Storage.Owned{d, n, array, balanced, valid}, exact, i, value,
          Storage.set_checked(d, n, array, i, value, False{}, decision, balanced, valid), same)), i) == Some{value} : Maybe<&2, F32>},
        Lifecycle.false_true(Equal.sym(Bool, False{}, True{}, accepted)))
    case True{}:
      step_index_mapping(shape, d, n, array, i, value, left, right, balanced, valid, exact,
        Storage.index_matches(d, n, i), {==}, same, matched)

def Laws.broadcast_step_mapping(operation, shape, depth, length, array, index, left, right,
  left_value, right_value, balanced, valid, exact, allowed, matched):
  step_checked_mapping(shape, depth, length, array, index, Elementwise.scalar(operation, left_value, right_value),
    left, right, balanced, valid, exact, Nat.is_lt(U32.to_nat(index), length), {==}, {==}, allowed, matched)

def and_true_left(+left: Bool, +right: Bool, equal: {Bool.and(left, right) == True{} : Bool}) -> {left == True{} : Bool}:
  match left:
    case False{}:
      Empty.absurd({False{} == True{} : Bool}, Lifecycle.false_true(Equal.sym(Bool, False{}, True{}, equal)))
    case True{}:
      {==}

def same_path_offset(+depth: Nat, +left: Model.Path(depth), +right: Model.Path(depth),
  equal: {Model.same(depth, left, right) == True{} : Bool}) -> {Model.offset(depth, left) == Model.offset(depth, right) : Nat}:
  match depth:
    case 0n:
      {==}
    case 1n++p:
      Model.Step{+lb, +lt} = left
      match lb:
        case False{}:
          Model.Step{+rb, +rt} = right
          match rb:
            case False{}:
              same_path_offset(p, lt, rt, equal)
            case True{}:
              Empty.absurd({Model.offset(p, lt) == Nat.add(Model.capacity(p), Model.offset(p, rt)) : Nat},
                Lifecycle.false_true(Equal.sym(Bool, False{}, True{}, equal)))
        case True{}:
          Model.Step{+rb, +rt} = right
          match rb:
            case False{}:
              Empty.absurd({Nat.add(Model.capacity(p), Model.offset(p, lt)) == Model.offset(p, rt) : Nat},
                Lifecycle.false_true(Equal.sym(Bool, False{}, True{}, equal)))
            case True{}:
              Equal.cong(Nat, Nat, n => Nat.add(Model.capacity(p), n), Model.offset(p, lt), Model.offset(p, rt),
                same_path_offset(p, lt, rt, equal))

def checked_offset(+depth: Nat, +length: Nat, +index: U32,
  checked: {Storage.index_matches(depth, length, index) == True{} : Bool}) ->
  {Model.offset(depth, Refinement.route(depth, index)) == U32.to_nat(index) : Nat}:
  Lifecycle.natural_equal(Model.offset(depth, Refinement.route(depth, index)), U32.to_nat(index),
    Equal.sym(Bool, Nat.is_eq(Model.offset(depth, Refinement.route(depth, index)), U32.to_nat(index)), True{},
      and_true_left(Nat.is_eq(Model.offset(depth, Refinement.route(depth, index)), U32.to_nat(index)),
        Nat.is_lt(Model.offset(depth, Refinement.route(depth, index)), length), checked)))

def nat_equal_self(+n: Nat) -> {Nat.is_eq(n, n) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      nat_equal_self(p)

def natural_clash(+left: Nat, +right: Nat, equal: {left == right : Nat},
  different: {Nat.is_eq(left, right) == False{} : Bool}) -> Empty:
  Lifecycle.false_true(Equal.trans(Bool, True{}, Nat.is_eq(left, right), False{},
    Equal.sym(Bool, Nat.is_eq(left, right), True{},
      %equal : {Nat.is_eq(left, _) == True{} : Bool}
      nat_equal_self(left)), different))

def route_separation_decided(+depth: Nat, +length: Nat, +index: U32, +other: U32,
  index_checked: {Storage.index_matches(depth, length, index) == True{} : Bool},
  other_checked: {Storage.index_matches(depth, length, other) == True{} : Bool},
  different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool}, same: Bool,
  decision: {same == Model.same(depth, Refinement.route(depth, index), Refinement.route(depth, other)) : Bool}) -> {same == False{} : Bool}:
  match same:
    case False{}:
      {==}
    case True{}:
      Empty.absurd({True{} == False{} : Bool}, natural_clash(U32.to_nat(index), U32.to_nat(other),
        Equal.trans(Nat, U32.to_nat(index), Model.offset(depth, Refinement.route(depth, index)), U32.to_nat(other),
          Equal.sym(Nat, Model.offset(depth, Refinement.route(depth, index)), U32.to_nat(index), checked_offset(depth, length, index, index_checked)),
          Equal.trans(Nat, Model.offset(depth, Refinement.route(depth, index)), Model.offset(depth, Refinement.route(depth, other)), U32.to_nat(other),
            same_path_offset(depth, Refinement.route(depth, index), Refinement.route(depth, other),
              Equal.sym(Bool, True{}, Model.same(depth, Refinement.route(depth, index), Refinement.route(depth, other)), decision)),
            checked_offset(depth, length, other, other_checked))), different))

def Laws.checked_route_separation(depth, length, index, other, index_checked, other_checked, different):
  route_separation_decided(depth, length, index, other, index_checked, other_checked, different,
    Model.same(depth, Refinement.route(depth, index), Refinement.route(depth, other)), {==})

def frame_index_mapping(-shape: Shape.Shape, +d: Nat, +n: Nat, array: Array<F32>, +i: U32, +other: U32, +value: F32,
  balanced: Model.Balanced(F32, d, array), -valid: {Storage.bounds(d, n) == True{} : Bool},
  -exact: {n == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, matched: Bool,
  decision: {matched == Storage.index_matches(d, n, i) : Bool},
  -same: {Storage.write_index(d, n, array, i, value, matched, decision, balanced, valid) ==
    Storage.set(Storage.Owned{d, n, array, balanced, valid}, i, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>},
  accepted: {matched == True{} : Bool},
  other_checked: {Storage.index_matches(d, n, other) == True{} : Bool},
  different: {Nat.is_eq(U32.to_nat(i), U32.to_nat(other)) == False{} : Bool}) ->
  {Core.write_observe(shape, Core.write_result(shape, Storage.Owned{d, n, array, balanced, valid}, exact, i, value,
    Storage.write_index(d, n, array, i, value, matched, decision, balanced, valid), same), other) ==
    Model.observe(F32, d, array, Refinement.route(d, other)) : Maybe<&2, F32>}:
  match matched:
    case False{}:
      Empty.absurd({Core.write_observe(shape, Core.write_result(shape, Storage.Owned{d, n, array, balanced, valid}, exact, i, value,
        Storage.write_index(d, n, array, i, value, False{}, decision, balanced, valid), same), other) ==
        Model.observe(F32, d, array, Refinement.route(d, other)) : Maybe<&2, F32>},
        Lifecycle.false_true(Equal.sym(Bool, False{}, True{}, accepted)))
    case True{}:
      %Laws.checked_route_separation(d, n, i, other,
        Equal.sym(Bool, True{}, Storage.index_matches(d, n, i), decision), other_checked, different) :
        {Model.observe(F32, d, Array.set(F32, array, i, value), Refinement.route(d, other)) ==
          Model.select(F32, _, value, Model.observe(F32, d, array, Refinement.route(d, other))) : Maybe<&2, F32>}
      StorageLaws.primitive_update_frame(F32, d, array, i, Refinement.route(d, other), value, balanced)

def frame_checked_mapping(-shape: Shape.Shape, +d: Nat, +n: Nat, array: Array<F32>, +i: U32, +other: U32, +value: F32,
  balanced: Model.Balanced(F32, d, array), -valid: {Storage.bounds(d, n) == True{} : Bool},
  -exact: {n == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(i), n) : Bool},
  -same: {Storage.set_checked(d, n, array, i, value, allowed, decision, balanced, valid) ==
    Storage.set(Storage.Owned{d, n, array, balanced, valid}, i, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>},
  accepted: {allowed == True{} : Bool},
  index_checked: {Storage.index_matches(d, n, i) == True{} : Bool},
  other_checked: {Storage.index_matches(d, n, other) == True{} : Bool},
  different: {Nat.is_eq(U32.to_nat(i), U32.to_nat(other)) == False{} : Bool}) ->
  {Core.write_observe(shape, Core.write_result(shape, Storage.Owned{d, n, array, balanced, valid}, exact, i, value,
    Storage.set_checked(d, n, array, i, value, allowed, decision, balanced, valid), same), other) ==
    Model.observe(F32, d, array, Refinement.route(d, other)) : Maybe<&2, F32>}:
  match allowed:
    case False{}:
      Empty.absurd({Core.write_observe(shape, Core.write_result(shape, Storage.Owned{d, n, array, balanced, valid}, exact, i, value,
        Storage.set_checked(d, n, array, i, value, False{}, decision, balanced, valid), same), other) ==
        Model.observe(F32, d, array, Refinement.route(d, other)) : Maybe<&2, F32>},
        Lifecycle.false_true(Equal.sym(Bool, False{}, True{}, accepted)))
    case True{}:
      frame_index_mapping(shape, d, n, array, i, other, value, balanced, valid, exact,
        Storage.index_matches(d, n, i), {==}, same, index_checked, other_checked, different)

def Laws.tensor_write_frame(shape, depth, length, array, index, other, value, balanced, valid, exact,
  allowed, index_checked, other_checked, different):
  frame_checked_mapping(shape, depth, length, array, index, other, value, balanced, valid, exact,
    Nat.is_lt(U32.to_nat(index), length), {==}, {==}, allowed, index_checked, other_checked, different)

def Laws.checked_conversion_value(cursor, index, verdict):
  Lifecycle.natural_equal(U32.to_nat(index), cursor, verdict)

def cursor_validity(-shape: Shape.Shape, +cursor: Nat, result: Elementwise.NaturalStep<shape, cursor>) ->
  Elementwise.CursorValidity(shape, cursor, result):
  match result:
    case Elementwise.NaturalStop{left, right, error}:
      Unit{}
    case Elementwise.NaturalContinue{state, index, exact}:
      Elementwise.CursorWitness{exact}

def Laws.natural_cursor_step(operation, plan, cursor, state):
  cursor_validity(Shape.output_shape(plan), cursor, Elementwise.natural_step(operation, plan, cursor, state))

def Laws.u32_natural_injective(left, right, equal):
  Injective.u32_natural_injective(left, right)(equal)

type UpdateIndexModel<-shape: Shape.Shape, -depth: Nat, -length: Nat, -array: Array<F32>, -index: U32, -value: F32,
  -matched: Bool, -decision: {matched == Storage.index_matches(depth, length, index) : Bool},
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -length_exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  -same: {Storage.write_index(depth, length, array, index, value, matched, decision, balanced, valid) ==
    Storage.set(Storage.Owned{depth, length, array, balanced, valid}, index, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}> is Data:
  UpdateIndexModel{-meaning: Core.UpdateMeaning(shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, length_exact}, index, value,
    Core.write_result(shape, Storage.Owned{depth, length, array, balanced, valid}, length_exact, index, value,
      Storage.write_index(depth, length, array, index, value, matched, decision, balanced, valid), same))}

def update_index_model(-shape: Shape.Shape, +depth: Nat, +length: Nat, -array: Array<F32>, +index: U32, +value: F32,
  matched: Bool, -decision: {matched == Storage.index_matches(depth, length, index) : Bool},
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -length_exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  -same: {Storage.write_index(depth, length, array, index, value, matched, decision, balanced, valid) ==
    Storage.set(Storage.Owned{depth, length, array, balanced, valid}, index, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}) ->
  UpdateIndexModel<shape, depth, length, array, index, value, matched, decision, balanced, valid, length_exact, same>:
  match matched:
    case False{}:
      UpdateIndexModel{Unit{}}
    case True{}:
      UpdateIndexModel{(Equal.sym(Bool, True{}, Storage.index_matches(depth, length, index), decision),
        (StorageLaws.primitive_write_read(F32, depth, array, index, value, balanced),
          other => checked => different => (checked,
            %Laws.checked_route_separation(depth, length, index, other,
              Equal.sym(Bool, True{}, Storage.index_matches(depth, length, index), decision), checked, different) :
              {Model.observe(F32, depth, Array.set(F32, array, index, value), Refinement.route(depth, other)) ==
                Model.select(F32, _, value, Model.observe(F32, depth, array, Refinement.route(depth, other))) : Maybe<&2, F32>}
            StorageLaws.primitive_update_frame(F32, depth, array, index, Refinement.route(depth, other), value, balanced))))}

type UpdateCheckedModel<-shape: Shape.Shape, -depth: Nat, -length: Nat, -array: Array<F32>, -index: U32, -value: F32,
  -allowed: Bool, -decision: {allowed == Nat.is_lt(U32.to_nat(index), length) : Bool},
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -length_exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  -same: {Storage.set_checked(depth, length, array, index, value, allowed, decision, balanced, valid) ==
    Storage.set(Storage.Owned{depth, length, array, balanced, valid}, index, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}> is Data:
  UpdateCheckedModel{-meaning: Core.UpdateMeaning(shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, length_exact}, index, value,
    Core.write_result(shape, Storage.Owned{depth, length, array, balanced, valid}, length_exact, index, value,
      Storage.set_checked(depth, length, array, index, value, allowed, decision, balanced, valid), same))}

def update_index_checked(-shape: Shape.Shape, +depth: Nat, +length: Nat, -array: Array<F32>, +index: U32, +value: F32,
  -decision: {True{} == Nat.is_lt(U32.to_nat(index), length) : Bool},
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -length_exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  -same: {Storage.set_checked(depth, length, array, index, value, True{}, decision, balanced, valid) ==
    Storage.set(Storage.Owned{depth, length, array, balanced, valid}, index, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>},
  certificate: UpdateIndexModel<shape, depth, length, array, index, value, Storage.index_matches(depth, length, index), {==}, balanced, valid, length_exact, same>) ->
  UpdateCheckedModel<shape, depth, length, array, index, value, True{}, decision, balanced, valid, length_exact, same>:
  UpdateIndexModel{meaning} = certificate
  UpdateCheckedModel{meaning}

def update_checked_model(-shape: Shape.Shape, +depth: Nat, +length: Nat, -array: Array<F32>, +index: U32, +value: F32,
  allowed: Bool, -decision: {allowed == Nat.is_lt(U32.to_nat(index), length) : Bool},
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -length_exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  -same: {Storage.set_checked(depth, length, array, index, value, allowed, decision, balanced, valid) ==
    Storage.set(Storage.Owned{depth, length, array, balanced, valid}, index, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}) ->
  UpdateCheckedModel<shape, depth, length, array, index, value, allowed, decision, balanced, valid, length_exact, same>:
  match allowed:
    case False{}:
      UpdateCheckedModel{Unit{}}
    case True{}:
      update_index_checked(shape, depth, length, array, index, value, decision, balanced, valid, length_exact, same,
        update_index_model(shape, depth, length, array, index, value, Storage.index_matches(depth, length, index), {==}, balanced, valid, length_exact, same))

def update_checked_tensor(-shape: Shape.Shape, +depth: Nat, +length: Nat, -array: Array<F32>, +index: U32, +value: F32,
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -length_exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  certificate: UpdateCheckedModel<shape, depth, length, array, index, value, Nat.is_lt(U32.to_nat(index), length), {==}, balanced, valid, length_exact, {==}>) ->
  Core.UpdateModel<shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, length_exact}, index, value>:
  UpdateCheckedModel{meaning} = certificate
  Core.UpdateModel{meaning}

def update_model(-shape: Shape.Shape, original: Core.Tensor<shape>, +index: U32, +value: F32) -> Core.UpdateModel<shape, original, index, value>:
  Core.TTensor{storage, length_exact} = original
  match storage:
    case Storage.Empty{}:
      Core.UpdateModel{Unit{}}
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      update_checked_tensor(shape, depth, length, array, index, value, balanced, valid, length_exact,
        update_checked_model(shape, depth, length, array, index, value, Nat.is_lt(U32.to_nat(index), length), {==}, balanced, valid, length_exact, {==}))

type WriteInfo<-shape: Shape.Shape, -original: Core.Tensor<shape>, -index: U32, -value: F32> is Type:
  WriteInfo{owner: Core.Tensor<shape>, same: {owner == original : Core.Tensor<shape>}, model: Core.UpdateModel<shape, original, index, value>}

def write_info(-shape: Shape.Shape, original: Core.Tensor<shape>, +index: U32, +value: F32) -> WriteInfo<shape, original, index, value>:
  Core.TTensor{storage, length_exact} = original
  match storage:
    case Storage.Empty{}:
      WriteInfo{Core.TTensor{Storage.Empty{}, length_exact}, {==}, Core.UpdateModel{Unit{}}}
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      WriteInfo{Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, length_exact}, {==},
        update_checked_tensor(shape, depth, length, array, index, value, balanced, valid, length_exact,
          update_checked_model(shape, depth, length, array, index, value, Nat.is_lt(U32.to_nat(index), length), {==}, balanced, valid, length_exact, {==}))}

def read_original(-original: Core.Dynamic, -owner: Core.Dynamic, +index: U32, same: {owner == original : Core.Dynamic},
  certificate: Core.DynamicRead<owner, index>) -> Core.DynamicRead<original, index>:
  Core.DynamicRead{dynamic, result, exact} = certificate
  Core.DynamicRead{dynamic, result, Equal.trans(Core.Dynamic & Result<&2, &2, Core.Error, F32>, (dynamic, result),
    Core.read_dynamic(owner, index), Core.read_dynamic(original, index), exact,
    Equal.cong(Core.Dynamic, Core.Dynamic & Result<&2, &2, Core.Error, F32>, input => Core.read_dynamic(input, index), owner, original, same))}

def write_original(-shape: Shape.Shape, -original: Core.Tensor<shape>, -owner: Core.Tensor<shape>, +index: U32, +value: F32,
  same: {owner == original : Core.Tensor<shape>}, certificate: Core.Write<shape, owner, index, value>) -> Core.Write<shape, original, index, value>:
  Core.Write{tensor, result, exact} = certificate
  Core.Write{tensor, result, Equal.trans(Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>, (tensor, result),
    Core.write_flat(shape, owner, index, value), Core.write_flat(shape, original, index, value), exact,
    Equal.cong(Core.Tensor<shape>, Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>, input => Core.write_flat(shape, input, index, value), owner, original, same))}

type ReadValueModel<-original: Core.Dynamic, -index: U32, -value: F32> is Data:
  ReadValueModel{-equal: {Core.checked_observe_dynamic(original, index) == Some{value} : Maybe<&2, F32>}}

def read_value_model(-original: Core.Dynamic, -retained: Core.Dynamic, +index: U32, +value: F32,
  -exact: {(retained, Done{value}) == Core.read_dynamic(original, index) : Core.Dynamic & Result<&2, &2, Core.Error, F32>},
  certificate: Core.DynamicSourceRead<original, index>) -> ReadValueModel<original, index, value>:
  Core.DynamicSourceRead{meaning} = certificate
  ReadValueModel{Equal.trans(Maybe<&2, F32>, Core.checked_observe_dynamic(original, index),
    Core.dynamic_read_value(Core.read_dynamic(original, index)), Some{value},
    Equal.sym(Maybe<&2, F32>, Core.dynamic_read_value(Core.read_dynamic(original, index)), Core.checked_observe_dynamic(original, index), meaning),
    Equal.cong(Core.Dynamic & Result<&2, &2, Core.Error, F32>, Maybe<&2, F32>, pair => Core.dynamic_read_value(pair),
      Core.read_dynamic(original, index), (retained, Done{value}),
      Equal.sym(Core.Dynamic & Result<&2, &2, Core.Error, F32>, (retained, Done{value}), Core.read_dynamic(original, index), exact)))}

type RetainedSourceModel<-original: Core.Dynamic, -retained: Core.Dynamic> is Data:
  RetainedSourceModel{-kept: Elementwise.SameSource(original, retained)}

def retained_source_model(-original: Core.Dynamic, -retained: Core.Dynamic, +index: U32, -value: F32,
  -exact: {(retained, Done{value}) == Core.read_dynamic(original, index) : Core.Dynamic & Result<&2, &2, Core.Error, F32>},
  certificate: Core.DynamicSourceReadKeeps<original, index>) -> RetainedSourceModel<original, retained>:
  Core.DynamicSourceReadKeeps{kept} = certificate
  RetainedSourceModel{other => Equal.trans(Maybe<&2, F32>, Core.checked_observe_dynamic(retained, other),
    Core.retained_dynamic_observe(Core.read_dynamic(original, index), other), Core.checked_observe_dynamic(original, other),
    Equal.cong(Core.Dynamic & Result<&2, &2, Core.Error, F32>, Maybe<&2, F32>, pair => Core.retained_dynamic_observe(pair, other),
      (retained, Done{value}), Core.read_dynamic(original, index), exact), kept(other))}

type ExpectedValue<-operation: Elementwise.Operation, -plan: Shape.BroadcastPlan, -index: U32,
  -left: Core.Dynamic, -right: Core.Dynamic, -left_value: F32, -right_value: F32> is Data:
  ExpectedValue{-equal: {Some{Elementwise.scalar(operation, left_value, right_value)} == Elementwise.expected(operation, plan, left, right, index) : Maybe<&2, F32>}}

def expected_value(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +index: U32, -left: Core.Dynamic, -right: Core.Dynamic,
  +left_index: U32, +right_index: U32, +left_value: F32, +right_value: F32,
  -indices_exact: {Done{(left_index, right_index)} == Elementwise.input_indices(plan, index) : Result<&2, &1, Core.Error, U32 & U32>},
  left_model: ReadValueModel<left, left_index, left_value>, right_model: ReadValueModel<right, right_index, right_value>) ->
  ExpectedValue<operation, plan, index, left, right, left_value, right_value>:
  ReadValueModel{left_equal} = left_model
  ReadValueModel{right_equal} = right_model
  ExpectedValue{
    %indices_exact : {Some{Elementwise.scalar(operation, left_value, right_value)} == Elementwise.expected_indices(operation, left, right, _) : Maybe<&2, F32>}
    %Equal.sym(Maybe<&2, F32>, Core.checked_observe_dynamic(left, left_index), Some{left_value}, left_equal) :
      {Some{Elementwise.scalar(operation, left_value, right_value)} == Elementwise.expected_left(operation, right, right_index, _) : Maybe<&2, F32>}
    %Equal.sym(Maybe<&2, F32>, Core.checked_observe_dynamic(right, right_index), Some{right_value}, right_equal) :
      {Some{Elementwise.scalar(operation, left_value, right_value)} == Elementwise.expected_right(operation, left_value, _) : Maybe<&2, F32>}
    {==}}

type KnownUpdate<-shape: Shape.Shape, -original: Core.Tensor<shape>, -output: Core.Tensor<shape>, -index: U32, -value: F32, -unit: Unit> is Data:
  KnownUpdate{-meaning: Core.UpdateMeaning(shape, original, index, value, (output, Done{unit}))}

def known_update(-shape: Shape.Shape, -original: Core.Tensor<shape>, -output: Core.Tensor<shape>, +index: U32, +value: F32, -unit: Unit,
  -exact: {(output, Done{unit}) == Core.write_flat(shape, original, index, value) : Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>},
  certificate: Core.UpdateModel<shape, original, index, value>) -> KnownUpdate<shape, original, output, index, value, unit>:
  Core.UpdateModel{meaning} = certificate
  KnownUpdate{
    %Equal.sym(Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>, (output, Done{unit}), Core.write_flat(shape, original, index, value), exact) :
      Core.UpdateMeaning(shape, original, index, value, _)
    meaning}

def update_route(-shape: Shape.Shape, -original: Core.Tensor<shape>, -output: Core.Tensor<shape>, -index: U32, -value: F32, -unit: Unit,
  meaning: Core.UpdateMeaning(shape, original, index, value, (output, Done{unit}))) -> Core.ObservationChecked(Core.Packed{shape, output}, index):
  (checked, rest) = meaning
  checked

def update_value(-shape: Shape.Shape, -original: Core.Tensor<shape>, -output: Core.Tensor<shape>, -index: U32, -value: F32, -unit: Unit,
  meaning: Core.UpdateMeaning(shape, original, index, value, (output, Done{unit}))) -> {Core.observe(shape, output, index) == Some{value} : Maybe<&2, F32>}:
  (checked, rest) = meaning
  (observed, frame) = rest
  observed

def update_frame(-shape: Shape.Shape, -original: Core.Tensor<shape>, -output: Core.Tensor<shape>, -index: U32, -value: F32, -unit: Unit,
  meaning: Core.UpdateMeaning(shape, original, index, value, (output, Done{unit}))) ->
  @+other: U32 -> @checked: Core.ObservationChecked(Core.Packed{shape, original}, other) ->
    @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
      Core.ObservationChecked(Core.Packed{shape, output}, other) &
        {Core.observe(shape, output, other) == Core.observe(shape, original, other) : Maybe<&2, F32>}:
  (checked, rest) = meaning
  (observed, frame) = rest
  frame

type CheckedFrame<-shape: Shape.Shape, -original: Core.Tensor<shape>, -output: Core.Tensor<shape>, -other: U32> is Data:
  CheckedFrame{-equal: {Core.checked_observe(shape, output, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}}

def checked_frame(-shape: Shape.Shape, -original: Core.Tensor<shape>, -output: Core.Tensor<shape>, +other: U32,
  -before: Core.ObservationChecked(Core.Packed{shape, original}, other),
  -after: Core.ObservationChecked(Core.Packed{shape, output}, other),
  -physical: {Core.observe(shape, output, other) == Core.observe(shape, original, other) : Maybe<&2, F32>}) -> CheckedFrame<shape, original, output, other>:
  CheckedFrame{
    {\{Core.ObservationModel: after_equal =>
      {\{Core.ObservationModel: before_equal =>
        Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, output, other), Core.observe(shape, output, other), Core.checked_observe(shape, original, other),
          after_equal, Equal.trans(Maybe<&2, F32>, Core.observe(shape, output, other), Core.observe(shape, original, other), Core.checked_observe(shape, original, other),
            physical, Equal.sym(Maybe<&2, F32>, Core.checked_observe(shape, original, other), Core.observe(shape, original, other), before_equal)))
      } : Core.ObservationModel<shape, original, other, before> -> {Core.checked_observe(shape, output, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}}(Core.observation_model(shape, original, other, before))
    } : Core.ObservationModel<shape, output, other, after> -> {Core.checked_observe(shape, output, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}}(Core.observation_model(shape, output, other, after))}

type CheckedUpdatePair<-shape: Shape.Shape, -original: Core.Tensor<shape>, -index: U32, -value: F32,
  -pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>> is Data:
  CheckedUpdatePair{-meaning: Core.CheckedUpdateMeaning(shape, original, index, value, pair)}

def checked_update_pair(-shape: Shape.Shape, -original: Core.Tensor<shape>, +index: U32, +value: F32,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>, -meaning: Core.UpdateMeaning(shape, original, index, value, pair)) ->
  CheckedUpdatePair<shape, original, index, value, pair>:
  (output, result) = pair
  match result:
    case Fail{error}:
      CheckedUpdatePair{Unit{}}
    case Done{unit}:
      CheckedUpdatePair{(update_route(shape, original, output, index, value, unit, meaning),
        (Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, output, index), Core.observe(shape, output, index), Some{value},
          {\{Core.ObservationModel: equal => equal} :
            Core.ObservationModel<shape, output, index, update_route(shape, original, output, index, value, unit, meaning)> ->
              {Core.checked_observe(shape, output, index) == Core.observe(shape, output, index) : Maybe<&2, F32>}}(Core.observation_model(shape, output, index, update_route(shape, original, output, index, value, unit, meaning))),
          update_value(shape, original, output, index, value, unit, meaning)),
          other => checked => different =>
            {\{Tuple: after => physical => (after,
              {\{CheckedFrame: equal => equal} : CheckedFrame<shape, original, output, other> ->
                {Core.checked_observe(shape, output, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}}(checked_frame(shape, original, output, other, checked, after, physical)))
            } :
              (Core.ObservationChecked(Core.Packed{shape, output}, other) &
                {Core.observe(shape, output, other) == Core.observe(shape, original, other) : Maybe<&2, F32>}) ->
              (Core.ObservationChecked(Core.Packed{shape, output}, other) &
                {Core.checked_observe(shape, output, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>})}(update_frame(shape, original, output, index, value, unit, meaning)(other)(checked)(different))))}

def checked_update_model(-shape: Shape.Shape, original: Core.Tensor<shape>, +index: U32, +value: F32) -> Core.CheckedUpdateModel<shape, original, index, value>:
  Core.CheckedUpdateModel{
    {\{Core.UpdateModel: meaning =>
      {\{CheckedUpdatePair: checked_meaning => checked_meaning} :
        CheckedUpdatePair<shape, original, index, value, Core.write_flat(shape, original, index, value)> ->
          Core.CheckedUpdateMeaning(shape, original, index, value, Core.write_flat(shape, original, index, value))}(checked_update_pair(shape, original, index, value, Core.write_flat(shape, original, index, value), meaning))
    } : Core.UpdateModel<shape, original, index, value> ->
      Core.CheckedUpdateMeaning(shape, original, index, value, Core.write_flat(shape, original, index, value))}(update_model(shape, original, index, value))}

def step_write_certificate(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +index: U32,
  -original_left: Core.Dynamic, -original_right: Core.Dynamic, -original_output: Core.Tensor<Shape.output_shape(plan)>,
  +left_index: U32, +right_index: U32, +left_value: F32, +right_value: F32,
  -indices_exact: {Done{(left_index, right_index)} == Elementwise.input_indices(plan, index) : Result<&2, &1, Core.Error, U32 & U32>},
  certificate: Core.Write<Shape.output_shape(plan), original_output, index, Elementwise.scalar(operation, left_value, right_value)>,
  left: Core.Dynamic, right: Core.Dynamic,
  -left_exact: {(left, Done{left_value}) == Core.read_dynamic(original_left, left_index) : Core.Dynamic & Result<&2, &2, Core.Error, F32>},
  -right_exact: {(right, Done{right_value}) == Core.read_dynamic(original_right, right_index) : Core.Dynamic & Result<&2, &2, Core.Error, F32>},
  lift: @-result: Elementwise.Step<Shape.output_shape(plan)> ->
    {result == Elementwise.step_write(Shape.output_shape(plan), left, right,
      Core.write_flat(Shape.output_shape(plan), original_output, index, Elementwise.scalar(operation, left_value, right_value))) : Elementwise.Step<Shape.output_shape(plan)>} ->
    {result == Elementwise.step(operation, plan, index, Elementwise.State{original_left, original_right, original_output}) : Elementwise.Step<Shape.output_shape(plan)>}) ->
  Elementwise.StepCertificate<operation, plan, index, Elementwise.State{original_left, original_right, original_output}>:
  Core.Write{output, result, exact} = certificate
  match result:
    case Fail{error}:
      Elementwise.StepCertificate{Elementwise.Stop{left, right, error},
        lift(Elementwise.Stop{left, right, error},
          Equal.cong(Core.Tensor<Shape.output_shape(plan)> & Result<&2, &2, Core.Error, Unit>, Elementwise.Step<Shape.output_shape(plan)>,
            pair => Elementwise.step_write(Shape.output_shape(plan), left, right, pair),
            (output, Fail{error}), Core.write_flat(Shape.output_shape(plan), original_output, index, Elementwise.scalar(operation, left_value, right_value)), exact)), Unit{}}
    case Done{unit}:
      Elementwise.StepCertificate{Elementwise.Continue{Elementwise.State{left, right, output}},
        lift(Elementwise.Continue{Elementwise.State{left, right, output}},
          Equal.cong(Core.Tensor<Shape.output_shape(plan)> & Result<&2, &2, Core.Error, Unit>, Elementwise.Step<Shape.output_shape(plan)>,
            pair => Elementwise.step_write(Shape.output_shape(plan), left, right, pair),
            (output, Done{unit}), Core.write_flat(Shape.output_shape(plan), original_output, index, Elementwise.scalar(operation, left_value, right_value)), exact)),
        ({\{RetainedSourceModel: kept => kept} : RetainedSourceModel<original_left, left> -> Elementwise.SameSource(original_left, left)}(retained_source_model(original_left, left, left_index, left_value, left_exact, Core.dynamic_source_read_keeps(original_left, left_index))),
          ({\{RetainedSourceModel: kept => kept} : RetainedSourceModel<original_right, right> -> Elementwise.SameSource(original_right, right)}(retained_source_model(original_right, right, right_index, right_value, right_exact, Core.dynamic_source_read_keeps(original_right, right_index))),
            {\{KnownUpdate: meaning =>
              (update_route(Shape.output_shape(plan), original_output, output, index, Elementwise.scalar(operation, left_value, right_value), unit, meaning),
                (Equal.trans(Maybe<&2, F32>, Core.observe(Shape.output_shape(plan), output, index), Some{Elementwise.scalar(operation, left_value, right_value)},
                  Elementwise.expected(operation, plan, original_left, original_right, index),
                  update_value(Shape.output_shape(plan), original_output, output, index, Elementwise.scalar(operation, left_value, right_value), unit, meaning),
                  {\{ExpectedValue: equal => equal} : ExpectedValue<operation, plan, index, original_left, original_right, left_value, right_value> ->
                    {Some{Elementwise.scalar(operation, left_value, right_value)} == Elementwise.expected(operation, plan, original_left, original_right, index) : Maybe<&2, F32>}}(expected_value(operation, plan, index, original_left, original_right, left_index, right_index, left_value, right_value, indices_exact,
                      read_value_model(original_left, left, left_index, left_value, left_exact, Core.dynamic_source_read_model(original_left, left_index)),
                      read_value_model(original_right, right, right_index, right_value, right_exact, Core.dynamic_source_read_model(original_right, right_index))))),
                  update_frame(Shape.output_shape(plan), original_output, output, index, Elementwise.scalar(operation, left_value, right_value), unit, meaning)))
            } : KnownUpdate<Shape.output_shape(plan), original_output, output, index, Elementwise.scalar(operation, left_value, right_value), unit> ->
              (Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), output}, index) &
                ({Core.observe(Shape.output_shape(plan), output, index) == Elementwise.expected(operation, plan, original_left, original_right, index) : Maybe<&2, F32>} &
                  (@+other: U32 -> @checked: Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), original_output}, other) ->
                    @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
                      Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), output}, other) &
                        {Core.observe(Shape.output_shape(plan), output, other) == Core.observe(Shape.output_shape(plan), original_output, other) : Maybe<&2, F32>})))}(known_update(Shape.output_shape(plan), original_output, output, index, Elementwise.scalar(operation, left_value, right_value), unit, exact,
                          update_model(Shape.output_shape(plan), original_output, index, Elementwise.scalar(operation, left_value, right_value))))))}

def step_right_certificate(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +index: U32,
  -original_left: Core.Dynamic, -original_right: Core.Dynamic,
  +left_index: U32, +right_index: U32, +left_value: F32,
  -indices_exact: {Done{(left_index, right_index)} == Elementwise.input_indices(plan, index) : Result<&2, &1, Core.Error, U32 & U32>},
  certificate: Core.DynamicRead<original_right, right_index>,
  original_output: Core.Tensor<Shape.output_shape(plan)>,
  left: Core.Dynamic,
  -left_exact: {(left, Done{left_value}) == Core.read_dynamic(original_left, left_index) : Core.Dynamic & Result<&2, &2, Core.Error, F32>},
  lift: @-result: Elementwise.Step<Shape.output_shape(plan)> ->
    {result == Elementwise.step_right(operation, Shape.output_shape(plan), index, left, original_output, left_value,
      Core.read_dynamic(original_right, right_index)) : Elementwise.Step<Shape.output_shape(plan)>} ->
    {result == Elementwise.step(operation, plan, index, Elementwise.State{original_left, original_right, original_output}) : Elementwise.Step<Shape.output_shape(plan)>}) ->
  Elementwise.StepCertificate<operation, plan, index, Elementwise.State{original_left, original_right, original_output}>:
  Core.DynamicRead{right, result, exact} = certificate
  match result:
    case Fail{error}:
      Elementwise.StepCertificate{Elementwise.Stop{left, right, error},
        lift(Elementwise.Stop{left, right, error}, Equal.cong(Core.Dynamic & Result<&2, &2, Core.Error, F32>, Elementwise.Step<Shape.output_shape(plan)>,
          pair => Elementwise.step_right(operation, Shape.output_shape(plan), index, left, original_output, left_value, pair),
          (right, Fail{error}), Core.read_dynamic(original_right, right_index), exact)), Unit{}}
    case Done{+right_value}:
      step_write_certificate(operation, plan, index, original_left, original_right, original_output,
        left_index, right_index, left_value, right_value, indices_exact,
        Core.write_certificate(Shape.output_shape(plan), original_output, index, Elementwise.scalar(operation, left_value, right_value)), left, right, left_exact, exact,
        result => equality => lift(result,
          Equal.trans(Elementwise.Step<Shape.output_shape(plan)>, result,
            Elementwise.step_right(operation, Shape.output_shape(plan), index, left, original_output, left_value, (right, Done{right_value})),
            Elementwise.step_right(operation, Shape.output_shape(plan), index, left, original_output, left_value, Core.read_dynamic(original_right, right_index)), equality,
            Equal.cong(Core.Dynamic & Result<&2, &2, Core.Error, F32>, Elementwise.Step<Shape.output_shape(plan)>,
              pair => Elementwise.step_right(operation, Shape.output_shape(plan), index, left, original_output, left_value, pair),
              (right, Done{right_value}), Core.read_dynamic(original_right, right_index), exact))))

def step_left_certificate(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +index: U32,
  -original_left: Core.Dynamic, original_right: Core.Dynamic, original_output: Core.Tensor<Shape.output_shape(plan)>,
  +left_index: U32, +right_index: U32,
  -indices_exact: {Done{(left_index, right_index)} == Elementwise.input_indices(plan, index) : Result<&2, &1, Core.Error, U32 & U32>},
  lift: @-result: Elementwise.Step<Shape.output_shape(plan)> ->
    {result == Elementwise.step_left(operation, Shape.output_shape(plan), index, original_right, original_output, right_index,
      Core.read_dynamic(original_left, left_index)) : Elementwise.Step<Shape.output_shape(plan)>} ->
    {result == Elementwise.step(operation, plan, index, Elementwise.State{original_left, original_right, original_output}) : Elementwise.Step<Shape.output_shape(plan)>},
  certificate: Core.DynamicRead<original_left, left_index>) ->
  Elementwise.StepCertificate<operation, plan, index, Elementwise.State{original_left, original_right, original_output}>:
  Core.DynamicRead{left, result, exact} = certificate
  match result:
    case Fail{error}:
      Elementwise.StepCertificate{Elementwise.Stop{left, original_right, error},
        lift(Elementwise.Stop{left, original_right, error}, Equal.cong(Core.Dynamic & Result<&2, &2, Core.Error, F32>, Elementwise.Step<Shape.output_shape(plan)>,
          pair => Elementwise.step_left(operation, Shape.output_shape(plan), index, original_right, original_output, right_index, pair),
          (left, Fail{error}), Core.read_dynamic(original_left, left_index), exact)), Unit{}}
    case Done{+left_value}:
      step_right_certificate(operation, plan, index, original_left, original_right,
        left_index, right_index, left_value, indices_exact, Core.dynamic_read_certificate(original_right, right_index), original_output, left, exact,
        result => equality => lift(result,
          Equal.trans(Elementwise.Step<Shape.output_shape(plan)>, result,
            Elementwise.step_left(operation, Shape.output_shape(plan), index, original_right, original_output, right_index, (left, Done{left_value})),
            Elementwise.step_left(operation, Shape.output_shape(plan), index, original_right, original_output, right_index, Core.read_dynamic(original_left, left_index)), equality,
            Equal.cong(Core.Dynamic & Result<&2, &2, Core.Error, F32>, Elementwise.Step<Shape.output_shape(plan)>,
              pair => Elementwise.step_left(operation, Shape.output_shape(plan), index, original_right, original_output, right_index, pair),
              (left, Done{left_value}), Core.read_dynamic(original_left, left_index), exact))))

def step_indices_certificate(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +index: U32,
  state: Elementwise.State<Shape.output_shape(plan)>, result: Result<&2, &1, Core.Error, U32 & U32>,
  exact: {result == Elementwise.input_indices(plan, index) : Result<&2, &1, Core.Error, U32 & U32>}) -> Elementwise.StepCertificate<operation, plan, index, state>:
  Elementwise.State{left, right, output} = state
  match result:
    case Fail{error}:
      Elementwise.StepCertificate{Elementwise.Stop{left, right, error},
        Equal.cong(Result<&2, &1, Core.Error, U32 & U32>, Elementwise.Step<Shape.output_shape(plan)>,
          input => Elementwise.step_indices(operation, Shape.output_shape(plan), index, Elementwise.State{left, right, output}, input),
          Fail{error}, Elementwise.input_indices(plan, index), exact), Unit{}}
    case Done{indices}:
      (+left_index, +right_index) = indices
      step_left_certificate(operation, plan, index, left, right, output, left_index, right_index, exact,
        result => equality => Equal.trans(Elementwise.Step<Shape.output_shape(plan)>, result,
          Elementwise.step_indices(operation, Shape.output_shape(plan), index, Elementwise.State{left, right, output}, Done{(left_index, right_index)}),
          Elementwise.step(operation, plan, index, Elementwise.State{left, right, output}), equality,
          Equal.cong(Result<&2, &1, Core.Error, U32 & U32>, Elementwise.Step<Shape.output_shape(plan)>,
            input => Elementwise.step_indices(operation, Shape.output_shape(plan), index, Elementwise.State{left, right, output}, input),
            Done{(left_index, right_index)}, Elementwise.input_indices(plan, index), exact)),
        Core.dynamic_read_certificate(left, left_index))

def step_certificate(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +index: U32,
  state: Elementwise.State<Shape.output_shape(plan)>) -> Elementwise.StepCertificate<operation, plan, index, state>:
  step_indices_certificate(operation, plan, index, state, Elementwise.input_indices(plan, index), {==})

def converted_step_certificate(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +cursor: Nat, +index: U32,
  +equal: {True{} == Nat.is_eq(U32.to_nat(index), cursor) : Bool}, -original: Elementwise.State<Shape.output_shape(plan)>,
  certificate: Elementwise.StepCertificate<operation, plan, index, original>) ->
  Elementwise.ConvertedCertificate<operation, plan, cursor, index, Shape.BTrue{equal}, original>:
  Elementwise.StepCertificate{result, exact, meaning} = certificate
  match result:
    case Elementwise.Stop{left, right, error}:
      Elementwise.ConvertedCertificate{Elementwise.NaturalStop{left, right, error},
        Equal.cong(Elementwise.Step<Shape.output_shape(plan)>, Elementwise.NaturalStep<Shape.output_shape(plan), cursor>,
          step => Elementwise.natural_step_result(Shape.output_shape(plan), cursor, index, Lifecycle.natural_equal(U32.to_nat(index), cursor, equal), step),
          Elementwise.Stop{left, right, error}, Elementwise.step(operation, plan, index, original), exact), Unit{}}
    case Elementwise.Continue{state}:
      Elementwise.ConvertedCertificate{Elementwise.NaturalContinue{state, index, Lifecycle.natural_equal(U32.to_nat(index), cursor, equal)},
        Equal.cong(Elementwise.Step<Shape.output_shape(plan)>, Elementwise.NaturalStep<Shape.output_shape(plan), cursor>,
          step => Elementwise.natural_step_result(Shape.output_shape(plan), cursor, index, Lifecycle.natural_equal(U32.to_nat(index), cursor, equal), step),
          Elementwise.Continue{state}, Elementwise.step(operation, plan, index, original), exact), meaning}

def natural_decision_certificate(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +cursor: Nat, +index: U32,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>, state: Elementwise.State<Shape.output_shape(plan)>) ->
  Elementwise.ConvertedCertificate<operation, plan, cursor, index, decision, state>:
  match decision:
    case Shape.BFalse{equal}:
      Elementwise.State{left, right, output} = state
      Elementwise.ConvertedCertificate{Elementwise.NaturalStop{left, right, Core.StorageError{Storage.InvalidIndex{}}}, {==}, Unit{}}
    case Shape.BTrue{+equal}:
      converted_step_certificate(operation, plan, cursor, index, equal, state, step_certificate(operation, plan, index, state))

def natural_converted_certificate(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +cursor: Nat,
  -original: Elementwise.State<Shape.output_shape(plan)>,
  certificate: Elementwise.ConvertedCertificate<operation, plan, cursor, U32.from_nat(cursor), Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)), original>) ->
  Elementwise.NaturalCertificate<operation, plan, cursor, original>:
  Elementwise.ConvertedCertificate{result, exact, meaning} = certificate
  Elementwise.NaturalCertificate{result, exact, meaning}

def natural_certificate(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +cursor: Nat,
  state: Elementwise.State<Shape.output_shape(plan)>) -> Elementwise.NaturalCertificate<operation, plan, cursor, state>:
  natural_converted_certificate(operation, plan, cursor, state,
    natural_decision_certificate(operation, plan, cursor, U32.from_nat(cursor), Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)), state))

def lt_zero(+number: Nat, proof: Shape.LT(number, 0n)) -> Empty:
  match number:
    case 0n:
      match proof:
    case 1n+p:
      match proof:

def lt_previous_or_equal(+number: Nat, +cursor: Nat, proof: Shape.LT(number, 1n+cursor)) ->
  Either<Shape.LT(number, cursor), {number == cursor : Nat}>:
  match number cursor:
    case 0n 0n:
      Inr{{==}}
    case 0n 1n+p:
      Inl{Unit{}}
    case 1n+p 0n:
      Empty.absurd(Either<Shape.LT(1n+p, 0n), {1n+p == 0n : Nat}>, lt_zero(p, proof))
    case 1n+p 1n+q:
      {\{Inl: earlier => Inl{earlier}; Inr: equal => Inr{Equal.cong(Nat, Nat, n => 1n+n, p, q, equal)}} :
        Either<Shape.LT(p, q), {p == q : Nat}> -> Either<Shape.LT(1n+p, 1n+q), {1n+p == 1n+q : Nat}>}(lt_previous_or_equal(p, q, proof))

def lt_big_small_different(+small: Nat, +big: Nat, proof: Shape.LT(small, big)) -> {Nat.is_eq(big, small) == False{} : Bool}:
  match small big:
    case 0n 0n:
      match proof:
    case 0n 1n+p:
      {==}
    case 1n+p 0n:
      match proof:
    case 1n+p 1n+q:
      lt_big_small_different(p, q, proof)

def add_successor_right(+left: Nat, -right: Nat) -> {Nat.add(left, 1n+right) == 1n+Nat.add(left, right) : Nat}:
  match left:
    case 0n:
      {==}
    case 1n+p:
      Equal.cong(Nat, Nat, n => 1n+n, Nat.add(p, 1n+right), 1n+Nat.add(p, right), add_successor_right(p, right))

def expected_left_same(+operation: Elementwise.Operation, -current: Core.Dynamic, -original: Core.Dynamic, +index: U32,
  left: Maybe<&2, F32>, equal: {Core.checked_observe_dynamic(current, index) == Core.checked_observe_dynamic(original, index) : Maybe<&2, F32>}) ->
  {Elementwise.expected_left(operation, current, index, left) == Elementwise.expected_left(operation, original, index, left) : Maybe<&2, F32>}:
  match left:
    case None{}:
      {==}
    case Some{+value}:
      Equal.cong(Maybe<&2, F32>, Maybe<&2, F32>, right => Elementwise.expected_right(operation, value, right),
        Core.checked_observe_dynamic(current, index), Core.checked_observe_dynamic(original, index), equal)

type ExpectedSame<-operation: Elementwise.Operation, -plan: Shape.BroadcastPlan, -index: U32,
  -original_left: Core.Dynamic, -original_right: Core.Dynamic, -left: Core.Dynamic, -right: Core.Dynamic> is Data:
  ExpectedSame{-equal: {Elementwise.expected(operation, plan, left, right, index) == Elementwise.expected(operation, plan, original_left, original_right, index) : Maybe<&2, F32>}}

def expected_same_indices(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +index: U32,
  -original_left: Core.Dynamic, -original_right: Core.Dynamic, -left: Core.Dynamic, -right: Core.Dynamic,
  -left_same: Elementwise.SameSource(original_left, left), -right_same: Elementwise.SameSource(original_right, right),
  result: Result<&2, &1, Core.Error, U32 & U32>,
  -exact: {result == Elementwise.input_indices(plan, index) : Result<&2, &1, Core.Error, U32 & U32>}) ->
  ExpectedSame<operation, plan, index, original_left, original_right, left, right>:
  match result:
    case Fail{error}:
      ExpectedSame{
        %exact : {Elementwise.expected_indices(operation, left, right, _) == Elementwise.expected_indices(operation, original_left, original_right, _) : Maybe<&2, F32>}
        {==}}
    case Done{indices}:
      (+left_index, +right_index) = indices
      ExpectedSame{
        %exact : {Elementwise.expected_indices(operation, left, right, _) == Elementwise.expected_indices(operation, original_left, original_right, _) : Maybe<&2, F32>}
        Equal.trans(Maybe<&2, F32>, Elementwise.expected_left(operation, right, right_index, Core.checked_observe_dynamic(left, left_index)),
          Elementwise.expected_left(operation, right, right_index, Core.checked_observe_dynamic(original_left, left_index)),
          Elementwise.expected_left(operation, original_right, right_index, Core.checked_observe_dynamic(original_left, left_index)),
          Equal.cong(Maybe<&2, F32>, Maybe<&2, F32>, value => Elementwise.expected_left(operation, right, right_index, value),
            Core.checked_observe_dynamic(left, left_index), Core.checked_observe_dynamic(original_left, left_index), left_same(left_index)),
          expected_left_same(operation, right, original_right, right_index, Core.checked_observe_dynamic(original_left, left_index), right_same(right_index)))}

def expected_same(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +index: U32,
  -original_left: Core.Dynamic, -original_right: Core.Dynamic, -left: Core.Dynamic, -right: Core.Dynamic,
  -left_same: Elementwise.SameSource(original_left, left), -right_same: Elementwise.SameSource(original_right, right)) ->
  ExpectedSame<operation, plan, index, original_left, original_right, left, right>:
  expected_same_indices(operation, plan, index, original_left, original_right, left, right, left_same, right_same, Elementwise.input_indices(plan, index), {==})

type PrefixModel<-operation: Elementwise.Operation, -plan: Shape.BroadcastPlan, -original_left: Core.Dynamic, -original_right: Core.Dynamic,
  -cursor: Nat, -output: Core.Tensor<Shape.output_shape(plan)>> is Data:
  PrefixModel{-prefix: Elementwise.PrefixInvariant(operation, plan, original_left, original_right, cursor, Core.Packed{Shape.output_shape(plan), output})}

def prefix_previous(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  -original_left: Core.Dynamic, -original_right: Core.Dynamic, +cursor: Nat, +index: U32,
  -before: Core.Tensor<Shape.output_shape(plan)>, -after: Core.Tensor<Shape.output_shape(plan)>,
  converted: {U32.to_nat(index) == cursor : Nat},
  prefix: Elementwise.PrefixInvariant(operation, plan, original_left, original_right, cursor, Core.Packed{Shape.output_shape(plan), before}),
  frame: @+other: U32 -> @checked: Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), before}, other) ->
    @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
      Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), after}, other) &
        {Core.observe(Shape.output_shape(plan), after, other) == Core.observe(Shape.output_shape(plan), before, other) : Maybe<&2, F32>},
  +other: U32, +earlier: Shape.LT(U32.to_nat(other), cursor)) ->
  Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), after}, other) &
    {Core.observe(Shape.output_shape(plan), after, other) == Elementwise.expected(operation, plan, original_left, original_right, other) : Maybe<&2, F32>}:
  {\{Tuple: before_checked => before_equal =>
    {\{Tuple: after_checked => after_equal => (after_checked,
      Equal.trans(Maybe<&2, F32>, Core.observe(Shape.output_shape(plan), after, other), Core.observe(Shape.output_shape(plan), before, other),
        Elementwise.expected(operation, plan, original_left, original_right, other), after_equal, before_equal))} :
      (Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), after}, other) &
        {Core.observe(Shape.output_shape(plan), after, other) == Core.observe(Shape.output_shape(plan), before, other) : Maybe<&2, F32>}) ->
      (Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), after}, other) &
        {Core.observe(Shape.output_shape(plan), after, other) == Elementwise.expected(operation, plan, original_left, original_right, other) : Maybe<&2, F32>})}(frame(other)(before_checked)(
          %Equal.sym(Nat, U32.to_nat(index), cursor, converted) : {Nat.is_eq(_, U32.to_nat(other)) == False{} : Bool}
          lt_big_small_different(U32.to_nat(other), cursor, earlier)))
  } :
    (Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), before}, other) &
      {Core.observe(Shape.output_shape(plan), before, other) == Elementwise.expected(operation, plan, original_left, original_right, other) : Maybe<&2, F32>}) ->
    (Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), after}, other) &
      {Core.observe(Shape.output_shape(plan), after, other) == Elementwise.expected(operation, plan, original_left, original_right, other) : Maybe<&2, F32>})}(prefix(other)(earlier))

def prefix_equal(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  -original_left: Core.Dynamic, -original_right: Core.Dynamic, +cursor: Nat, +index: U32,
  -after: Core.Tensor<Shape.output_shape(plan)>, converted: {U32.to_nat(index) == cursor : Nat},
  checked: Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), after}, index),
  observed: {Core.observe(Shape.output_shape(plan), after, index) == Elementwise.expected(operation, plan, original_left, original_right, index) : Maybe<&2, F32>},
  +other: U32, equal: {U32.to_nat(other) == cursor : Nat}) ->
  Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), after}, other) &
    {Core.observe(Shape.output_shape(plan), after, other) == Elementwise.expected(operation, plan, original_left, original_right, other) : Maybe<&2, F32>}:
  %Injective.u32_natural_injective(index, other)(Equal.trans(Nat, U32.to_nat(index), cursor, U32.to_nat(other), converted,
    Equal.sym(Nat, U32.to_nat(other), cursor, equal))) :
    Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), after}, _) &
      {Core.observe(Shape.output_shape(plan), after, _) == Elementwise.expected(operation, plan, original_left, original_right, _) : Maybe<&2, F32>}
  (checked, observed)

def prefix_extend(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  -original_left: Core.Dynamic, -original_right: Core.Dynamic, +cursor: Nat, +index: U32,
  -before: Core.Tensor<Shape.output_shape(plan)>, -after: Core.Tensor<Shape.output_shape(plan)>,
  -converted: {U32.to_nat(index) == cursor : Nat},
  -prefix: Elementwise.PrefixInvariant(operation, plan, original_left, original_right, cursor, Core.Packed{Shape.output_shape(plan), before}),
  -checked: Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), after}, index),
  -observed: {Core.observe(Shape.output_shape(plan), after, index) == Elementwise.expected(operation, plan, original_left, original_right, index) : Maybe<&2, F32>},
  -frame: @+other: U32 -> @checked: Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), before}, other) ->
    @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
      Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), after}, other) &
        {Core.observe(Shape.output_shape(plan), after, other) == Core.observe(Shape.output_shape(plan), before, other) : Maybe<&2, F32>}) ->
  PrefixModel<operation, plan, original_left, original_right, 1n+cursor, after>:
  PrefixModel{other => bound =>
    {\{Inl: earlier => prefix_previous(operation, plan, original_left, original_right, cursor, index, before, after, converted, prefix, frame, other, earlier);
      Inr: equal => prefix_equal(operation, plan, original_left, original_right, cursor, index, after, converted, checked, observed, other, equal)} :
      Either<Shape.LT(U32.to_nat(other), cursor), {U32.to_nat(other) == cursor : Nat}> ->
        (Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), after}, other) &
          {Core.observe(Shape.output_shape(plan), after, other) == Elementwise.expected(operation, plan, original_left, original_right, other) : Maybe<&2, F32>})}(lt_previous_or_equal(U32.to_nat(other), cursor, bound))}

def invariant_left(-operation: Elementwise.Operation, -plan: Shape.BroadcastPlan, -original_left: Core.Dynamic, -original_right: Core.Dynamic,
  -cursor: Nat, -left: Core.Dynamic, -right: Core.Dynamic, -output: Core.Tensor<Shape.output_shape(plan)>,
  invariant: Elementwise.StateInvariant(operation, plan, original_left, original_right, cursor, Elementwise.State{left, right, output})) ->
  Elementwise.SameSource(original_left, left):
  (same, rest) = invariant
  same

def invariant_right(-operation: Elementwise.Operation, -plan: Shape.BroadcastPlan, -original_left: Core.Dynamic, -original_right: Core.Dynamic,
  -cursor: Nat, -left: Core.Dynamic, -right: Core.Dynamic, -output: Core.Tensor<Shape.output_shape(plan)>,
  invariant: Elementwise.StateInvariant(operation, plan, original_left, original_right, cursor, Elementwise.State{left, right, output})) ->
  Elementwise.SameSource(original_right, right):
  (same, rest) = invariant
  (right_same, prefix) = rest
  right_same

def invariant_prefix(-operation: Elementwise.Operation, -plan: Shape.BroadcastPlan, -original_left: Core.Dynamic, -original_right: Core.Dynamic,
  -cursor: Nat, -left: Core.Dynamic, -right: Core.Dynamic, -output: Core.Tensor<Shape.output_shape(plan)>,
  invariant: Elementwise.StateInvariant(operation, plan, original_left, original_right, cursor, Elementwise.State{left, right, output})) ->
  Elementwise.PrefixInvariant(operation, plan, original_left, original_right, cursor, Core.Packed{Shape.output_shape(plan), output}):
  (same, rest) = invariant
  (right_same, prefix) = rest
  prefix

def step_same_left(-operation: Elementwise.Operation, -plan: Shape.BroadcastPlan, -index: U32,
  -left: Core.Dynamic, -right: Core.Dynamic, -output: Core.Tensor<Shape.output_shape(plan)>,
  -after_left: Core.Dynamic, -after_right: Core.Dynamic, -after_output: Core.Tensor<Shape.output_shape(plan)>,
  meaning: Elementwise.StepMeaning(operation, plan, index, Elementwise.State{left, right, output}, Elementwise.Continue{Elementwise.State{after_left, after_right, after_output}})) ->
  Elementwise.SameSource(left, after_left):
  (same, rest) = meaning
  same

def step_same_right(-operation: Elementwise.Operation, -plan: Shape.BroadcastPlan, -index: U32,
  -left: Core.Dynamic, -right: Core.Dynamic, -output: Core.Tensor<Shape.output_shape(plan)>,
  -after_left: Core.Dynamic, -after_right: Core.Dynamic, -after_output: Core.Tensor<Shape.output_shape(plan)>,
  meaning: Elementwise.StepMeaning(operation, plan, index, Elementwise.State{left, right, output}, Elementwise.Continue{Elementwise.State{after_left, after_right, after_output}})) ->
  Elementwise.SameSource(right, after_right):
  (same, rest) = meaning
  (right_same, result) = rest
  right_same

def step_checked(-operation: Elementwise.Operation, -plan: Shape.BroadcastPlan, -index: U32,
  -left: Core.Dynamic, -right: Core.Dynamic, -output: Core.Tensor<Shape.output_shape(plan)>,
  -after_left: Core.Dynamic, -after_right: Core.Dynamic, -after_output: Core.Tensor<Shape.output_shape(plan)>,
  meaning: Elementwise.StepMeaning(operation, plan, index, Elementwise.State{left, right, output}, Elementwise.Continue{Elementwise.State{after_left, after_right, after_output}})) ->
  Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), after_output}, index):
  (same, rest) = meaning
  (right_same, result) = rest
  (checked, values) = result
  checked

def step_observed(-operation: Elementwise.Operation, -plan: Shape.BroadcastPlan, -index: U32,
  -left: Core.Dynamic, -right: Core.Dynamic, -output: Core.Tensor<Shape.output_shape(plan)>,
  -after_left: Core.Dynamic, -after_right: Core.Dynamic, -after_output: Core.Tensor<Shape.output_shape(plan)>,
  meaning: Elementwise.StepMeaning(operation, plan, index, Elementwise.State{left, right, output}, Elementwise.Continue{Elementwise.State{after_left, after_right, after_output}})) ->
  {Core.observe(Shape.output_shape(plan), after_output, index) == Elementwise.expected(operation, plan, left, right, index) : Maybe<&2, F32>}:
  (same, rest) = meaning
  (right_same, result) = rest
  (checked, values) = result
  (observed, frame) = values
  observed

def step_frame(-operation: Elementwise.Operation, -plan: Shape.BroadcastPlan, -index: U32,
  -left: Core.Dynamic, -right: Core.Dynamic, -output: Core.Tensor<Shape.output_shape(plan)>,
  -after_left: Core.Dynamic, -after_right: Core.Dynamic, -after_output: Core.Tensor<Shape.output_shape(plan)>,
  meaning: Elementwise.StepMeaning(operation, plan, index, Elementwise.State{left, right, output}, Elementwise.Continue{Elementwise.State{after_left, after_right, after_output}})) ->
  @+other: U32 -> @checked: Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), output}, other) ->
    @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
      Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), after_output}, other) &
        {Core.observe(Shape.output_shape(plan), after_output, other) == Core.observe(Shape.output_shape(plan), output, other) : Maybe<&2, F32>}:
  (same, rest) = meaning
  (right_same, result) = rest
  (checked, values) = result
  (observed, frame) = values
  frame

def invariant_advance(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, -original_left: Core.Dynamic, -original_right: Core.Dynamic,
  +cursor: Nat, +index: U32, -converted: {U32.to_nat(index) == cursor : Nat},
  before: Elementwise.State<Shape.output_shape(plan)>, after: Elementwise.State<Shape.output_shape(plan)>,
  -invariant: Elementwise.StateInvariant(operation, plan, original_left, original_right, cursor, before),
  -meaning: Elementwise.StepMeaning(operation, plan, index, before, Elementwise.Continue{after})) ->
  Elementwise.InvariantModel<operation, plan, original_left, original_right, 1n+cursor, after>:
  Elementwise.State{left, right, output} = before
  Elementwise.State{after_left, after_right, after_output} = after
  Elementwise.InvariantModel{(other => Equal.trans(Maybe<&2, F32>, Core.checked_observe_dynamic(after_left, other), Core.checked_observe_dynamic(left, other), Core.checked_observe_dynamic(original_left, other),
      step_same_left(operation, plan, index, left, right, output, after_left, after_right, after_output, meaning)(other),
      invariant_left(operation, plan, original_left, original_right, cursor, left, right, output, invariant)(other)),
    (other => Equal.trans(Maybe<&2, F32>, Core.checked_observe_dynamic(after_right, other), Core.checked_observe_dynamic(right, other), Core.checked_observe_dynamic(original_right, other),
      step_same_right(operation, plan, index, left, right, output, after_left, after_right, after_output, meaning)(other),
      invariant_right(operation, plan, original_left, original_right, cursor, left, right, output, invariant)(other)),
      {\{PrefixModel: prefix => prefix} : PrefixModel<operation, plan, original_left, original_right, 1n+cursor, after_output> ->
        Elementwise.PrefixInvariant(operation, plan, original_left, original_right, 1n+cursor, Core.Packed{Shape.output_shape(plan), after_output})}(prefix_extend(operation, plan, original_left, original_right, cursor, index, output, after_output, converted,
          invariant_prefix(operation, plan, original_left, original_right, cursor, left, right, output, invariant),
          step_checked(operation, plan, index, left, right, output, after_left, after_right, after_output, meaning),
          Equal.trans(Maybe<&2, F32>, Core.observe(Shape.output_shape(plan), after_output, index), Elementwise.expected(operation, plan, left, right, index),
            Elementwise.expected(operation, plan, original_left, original_right, index),
            step_observed(operation, plan, index, left, right, output, after_left, after_right, after_output, meaning),
            {\{ExpectedSame: equal => equal} : ExpectedSame<operation, plan, index, original_left, original_right, left, right> ->
              {Elementwise.expected(operation, plan, left, right, index) == Elementwise.expected(operation, plan, original_left, original_right, index) : Maybe<&2, F32>}}(expected_same(operation, plan, index, original_left, original_right, left, right,
                invariant_left(operation, plan, original_left, original_right, cursor, left, right, output, invariant),
                invariant_right(operation, plan, original_left, original_right, cursor, left, right, output, invariant)))),
          step_frame(operation, plan, index, left, right, output, after_left, after_right, after_output, meaning)))))}

def initial_invariant(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  -left: Core.Dynamic, -right: Core.Dynamic, -output: Core.Tensor<Shape.output_shape(plan)>) ->
  Elementwise.InvariantModel<operation, plan, left, right, 0n, Elementwise.State{left, right, output}>:
  Elementwise.InvariantModel{(other => {==}, (other => {==}, other => bound =>
    Empty.absurd(Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), output}, other) &
      {Core.observe(Shape.output_shape(plan), output, other) == Elementwise.expected(operation, plan, left, right, other) : Maybe<&2, F32>}, lt_zero(U32.to_nat(other), bound))))}

def add_zero_right(+number: Nat) -> {Nat.add(number, 0n) == number : Nat}:
  match number:
    case 0n:
      {==}
    case 1n+p:
      Equal.cong(Nat, Nat, n => 1n+n, Nat.add(p, 0n), p, add_zero_right(p))

def loop_continue_certificate(+remaining: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +cursor: Nat,
  -original_left: Core.Dynamic, -original_right: Core.Dynamic, -original: Elementwise.State<Shape.output_shape(plan)>,
  -after: Elementwise.State<Shape.output_shape(plan)>, -index: U32, -converted: {U32.to_nat(index) == cursor : Nat},
  natural_exact: {Elementwise.NaturalContinue{after, index, converted} == Elementwise.natural_step(operation, plan, cursor, original) : Elementwise.NaturalStep<Shape.output_shape(plan), cursor>},
  certificate: Elementwise.LoopCertificate<remaining, operation, plan, 1n+cursor, original_left, original_right, after>) ->
  Elementwise.LoopCertificate<1n+remaining, operation, plan, cursor, original_left, original_right, original>:
  Elementwise.LoopCertificate{result, exact, meaning} = certificate
  Elementwise.LoopCertificate{result,
    Equal.trans((Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>, result,
      Elementwise.loop(remaining, operation, plan, 1n+cursor, after), Elementwise.loop(1n+remaining, operation, plan, cursor, original), exact,
      Equal.cong(Elementwise.NaturalStep<Shape.output_shape(plan), cursor>, (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
        step => Elementwise.finish_step(Shape.output_shape(plan), cursor, state => Elementwise.loop(remaining, operation, plan, 1n+cursor, state), step),
        Elementwise.NaturalContinue{after, index, converted}, Elementwise.natural_step(operation, plan, cursor, original), natural_exact)),
    %Equal.sym(Nat, Nat.add(cursor, 1n+remaining), Nat.add(1n+cursor, remaining), add_successor_right(cursor, remaining)) :
      Elementwise.LoopMeaning(operation, plan, original_left, original_right, _, result)
    meaning}

def loop_after_certificate(+remaining: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +cursor: Nat,
  -original_left: Core.Dynamic, -original_right: Core.Dynamic, -original: Elementwise.State<Shape.output_shape(plan)>,
  -invariant: Elementwise.StateInvariant(operation, plan, original_left, original_right, cursor, original),
  certificate: Elementwise.NaturalCertificate<operation, plan, cursor, original>,
  rec: @state: Elementwise.State<Shape.output_shape(plan)> ->
    @-invariant: Elementwise.StateInvariant(operation, plan, original_left, original_right, 1n+cursor, state) ->
      Elementwise.LoopCertificate<remaining, operation, plan, 1n+cursor, original_left, original_right, state>) ->
  Elementwise.LoopCertificate<1n+remaining, operation, plan, cursor, original_left, original_right, original>:
  Elementwise.NaturalCertificate{result, exact, meaning} = certificate
  match result:
    case Elementwise.NaturalStop{left, right, error}:
      Elementwise.LoopCertificate{((left, right), Fail{error}),
        Equal.cong(Elementwise.NaturalStep<Shape.output_shape(plan), cursor>, (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
          step => Elementwise.finish_step(Shape.output_shape(plan), cursor, state => Elementwise.loop(remaining, operation, plan, 1n+cursor, state), step),
          Elementwise.NaturalStop{left, right, error}, Elementwise.natural_step(operation, plan, cursor, original), exact), Unit{}}
    case Elementwise.NaturalContinue{state, index, converted}:
      loop_continue_certificate(remaining, operation, plan, cursor, original_left, original_right, original, state, index, converted, exact,
        rec(state,
          {\{Elementwise.InvariantModel: invariant => invariant} :
            Elementwise.InvariantModel<operation, plan, original_left, original_right, 1n+cursor, state> ->
              Elementwise.StateInvariant(operation, plan, original_left, original_right, 1n+cursor, state)}(invariant_advance(operation, plan, original_left, original_right, cursor, index, converted,
                original, state, invariant, meaning))))

def loop_certificate(fuel: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +cursor: Nat,
  -original_left: Core.Dynamic, -original_right: Core.Dynamic, state: Elementwise.State<Shape.output_shape(plan)>,
  -invariant: Elementwise.StateInvariant(operation, plan, original_left, original_right, cursor, state)) ->
  Elementwise.LoopCertificate<fuel, operation, plan, cursor, original_left, original_right, state>:
  match fuel:
    case 0n:
      Elementwise.State{left, right, output} = state
      Elementwise.LoopCertificate{((left, right), Done{Core.Packed{Shape.output_shape(plan), output}}), {==},
        %Equal.sym(Nat, Nat.add(cursor, 0n), cursor, add_zero_right(cursor)) :
          Elementwise.PrefixInvariant(operation, plan, original_left, original_right, _, Core.Packed{Shape.output_shape(plan), output})
        invariant_prefix(operation, plan, original_left, original_right, cursor, left, right, output, invariant)}
    case 1n++remaining:
      loop_after_certificate(remaining, operation, plan, cursor, original_left, original_right, state, invariant,
        natural_certificate(operation, plan, cursor, state),
        state => invariant => loop_certificate(remaining, operation, plan, 1n+cursor, original_left, original_right, state, invariant))

def successful_pair_mapping(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, -left: Core.Dynamic, -right: Core.Dynamic,
  result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  -meaning: Elementwise.LoopMeaning(operation, plan, left, right, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), result)) ->
  Elementwise.MappingModel<operation, plan, left, right, result>:
  (inputs, outcome) = result
  match outcome:
    case Fail{error}:
      Elementwise.MappingModel{Unit{}}
    case Done{output}:
      Elementwise.MappingModel{index => bound => Pair.snd(Core.ObservationChecked(output, index),
        {Core.observe_dynamic(output, index) == Elementwise.expected(operation, plan, left, right, index) : Maybe<&2, F32>}, meaning(index)(bound))}

def successful_loop_certificate(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  -left: Core.Dynamic, -right: Core.Dynamic, -output: Core.Tensor<Shape.output_shape(plan)>,
  certificate: Elementwise.LoopCertificate<Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, 0n, left, right, Elementwise.State{left, right, output}>) ->
  Elementwise.MappingModel<operation, plan, left, right,
    Elementwise.loop(Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, 0n, Elementwise.State{left, right, output})>:
  Elementwise.LoopCertificate{result, exact, meaning} = certificate
  Elementwise.MappingModel{
    %exact : Elementwise.SuccessfulMapping(operation, plan, left, right, _)
    {\{Elementwise.MappingModel: mapping => mapping} : Elementwise.MappingModel<operation, plan, left, right, result> ->
      Elementwise.SuccessfulMapping(operation, plan, left, right, result)}(successful_pair_mapping(operation, plan, left, right, result, meaning))}

def successful_loop_mapping(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  left: Core.Dynamic, right: Core.Dynamic, output: Core.Tensor<Shape.output_shape(plan)>) ->
  Elementwise.MappingModel<operation, plan, left, right,
    Elementwise.loop(Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, 0n, Elementwise.State{left, right, output})>:
  successful_loop_certificate(operation, plan, left, right, output,
    loop_certificate(Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, 0n, left, right, Elementwise.State{left, right, output},
      {\{Elementwise.InvariantModel: invariant => invariant} : Elementwise.InvariantModel<operation, plan, left, right, 0n, Elementwise.State{left, right, output}> ->
        Elementwise.StateInvariant(operation, plan, left, right, 0n, Elementwise.State{left, right, output})}(initial_invariant(operation, plan, left, right, output))))

def Laws.checked_writer_pair(shape, original, index, value):
  Core.write_certificate(shape, original, index, value)

def Laws.checked_writer_source(shape, original, index, value):
  checked_update_model(shape, original, index, value)

def Laws.actual_elementwise_step(operation, plan, index, original):
  step_certificate(operation, plan, index, original)

def Laws.successful_elementwise_loop(operation, plan, left, right, output):
  successful_loop_mapping(operation, plan, left, right, output)

def successful_cached_loop_mapping(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  left: Core.Dynamic, right: Core.Dynamic, output: Core.Tensor<Shape.output_shape(plan)>) ->
  Elementwise.MappingModel<operation, plan, left, right,
    Elementwise.loop(Shape.elements(Shape.output_shape(plan)), operation, plan, 0n, Elementwise.State{left, right, output})>:
  Elementwise.MappingModel{
    %Equal.sym(Nat, Shape.elements(Shape.output_shape(plan)), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), Shape.elements_exact(Shape.output_shape(plan))) :
      Elementwise.SuccessfulMapping(operation, plan, left, right, Elementwise.loop(_, operation, plan, 0n, Elementwise.State{left, right, output}))
    {\{Elementwise.MappingModel: mapping => mapping} :
      Elementwise.MappingModel<operation, plan, left, right,
        Elementwise.loop(Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, 0n, Elementwise.State{left, right, output})> ->
      Elementwise.SuccessfulMapping(operation, plan, left, right,
        Elementwise.loop(Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, 0n, Elementwise.State{left, right, output}))}(successful_loop_mapping(operation, plan, left, right, output))}

def allocated_mapping(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  result: Result<&2, &1, Core.Error, Core.Tensor<Shape.output_shape(plan)>>, left: Core.Dynamic, right: Core.Dynamic) ->
  Elementwise.MappingModel<operation, plan, left, right, Elementwise.allocated(operation, plan, left, right, result)>:
  match result:
    case Fail{error}:
      Elementwise.MappingModel{Unit{}}
    case Done{output}:
      successful_cached_loop_mapping(operation, plan, left, right, output)

def successful_allocated_mapping(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, left: Core.Dynamic, right: Core.Dynamic) ->
  Elementwise.MappingModel<operation, plan, left, right, Elementwise.allocated(operation, plan, left, right, Core.new(Shape.output_shape(plan), 0.0))>:
  Elementwise.MappingModel{
    {\{Elementwise.MappingModel: mapping => mapping} :
      Elementwise.MappingModel<operation, plan, left, right, Elementwise.allocated(operation, plan, left, right, Core.new(Shape.output_shape(plan), 0.0))> ->
      Elementwise.SuccessfulMapping(operation, plan, left, right, Elementwise.allocated(operation, plan, left, right, Core.new(Shape.output_shape(plan), 0.0)))}(allocated_mapping(operation, plan, Core.new(Shape.output_shape(plan), 0.0), left, right))}

def planned_mapping(+operation: Elementwise.Operation, result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>,
  left: Core.Dynamic, right: Core.Dynamic,
  -same: {result == Elementwise.plan_for(left, right) : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}) ->
  Elementwise.BinaryModel<operation, left, right, Elementwise.planned(operation, left, right, result)>:
  match result:
    case Fail{error}:
      Elementwise.BinaryModel{
        %same : Elementwise.PlannedMeaning(operation, left, right, Elementwise.planned(operation, left, right, Fail{error}), _)
        Unit{}}
    case Done{+plan}:
      Elementwise.BinaryModel{
        %same : Elementwise.PlannedMeaning(operation, left, right, Elementwise.planned(operation, left, right, Done{plan}), _)
        {\{Elementwise.MappingModel: mapping => mapping} :
          Elementwise.MappingModel<operation, plan, left, right, Elementwise.allocated(operation, plan, left, right, Core.new(Shape.output_shape(plan), 0.0))> ->
          Elementwise.SuccessfulMapping(operation, plan, left, right, Elementwise.allocated(operation, plan, left, right, Core.new(Shape.output_shape(plan), 0.0)))}(successful_allocated_mapping(operation, plan, left, right))}

def successful_binary_mapping(+operation: Elementwise.Operation, left: Core.Dynamic, right: Core.Dynamic) ->
  Elementwise.BinaryModel<operation, left, right, Elementwise.binary(operation, left, right)>:
  Core.Packed{+left_shape, left_tensor} = left
  Core.Packed{+right_shape, right_tensor} = right
  planned_mapping(operation, Shape.broadcast_checked(left_shape, right_shape), Core.Packed{left_shape, left_tensor}, Core.Packed{right_shape, right_tensor}, {==})

def Laws.successful_elementwise_cached_loop(operation, plan, left, right, output):
  successful_cached_loop_mapping(operation, plan, left, right, output)

def Laws.successful_elementwise_allocated(operation, plan, left, right):
  successful_allocated_mapping(operation, plan, left, right)

def Laws.successful_elementwise_binary(operation, left, right):
  successful_binary_mapping(operation, left, right)

def independent_pair_mapping(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, -left: Core.Dynamic, -right: Core.Dynamic,
  result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  -shared: Elementwise.SuccessfulMapping(operation, plan, left, right, result)) -> Laws.IndependentModel<operation, plan, left, right, result>:
  (inputs, outcome) = result
  match outcome:
    case Fail{error}:
      Laws.IndependentModel{Unit{}}
    case Done{output}:
      Laws.IndependentModel{index => inside => Equal.trans(Maybe<&2, F32>, Core.observe_dynamic(output, index), Elementwise.expected(operation, plan, left, right, index),
        Projection.specified_expected(operation, left, right, Projection.offset_spec(plan, U32.to_nat(index))), shared(index)(inside),
        Projection.expected_projection_refines(operation, plan, left, right, index))}

def independent_loop_mapping(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  left: Core.Dynamic, right: Core.Dynamic, output: Core.Tensor<Shape.output_shape(plan)>) ->
  Laws.IndependentModel<operation, plan, left, right,
    Elementwise.loop(Shape.elements(Shape.output_shape(plan)), operation, plan, 0n, Elementwise.State{left, right, output})>:
  Laws.IndependentModel{
    {\{Elementwise.MappingModel: mapping =>
      {\{Laws.IndependentModel: independent => independent} :
        Laws.IndependentModel<operation, plan, left, right, Elementwise.loop(Shape.elements(Shape.output_shape(plan)), operation, plan, 0n, Elementwise.State{left, right, output})> ->
          Laws.IndependentMapping(operation, plan, left, right, Elementwise.loop(Shape.elements(Shape.output_shape(plan)), operation, plan, 0n, Elementwise.State{left, right, output}))}(independent_pair_mapping(operation, plan, left, right,
            Elementwise.loop(Shape.elements(Shape.output_shape(plan)), operation, plan, 0n, Elementwise.State{left, right, output}), mapping))
    } : Elementwise.MappingModel<operation, plan, left, right, Elementwise.loop(Shape.elements(Shape.output_shape(plan)), operation, plan, 0n, Elementwise.State{left, right, output})> ->
      Laws.IndependentMapping(operation, plan, left, right, Elementwise.loop(Shape.elements(Shape.output_shape(plan)), operation, plan, 0n, Elementwise.State{left, right, output}))}(successful_cached_loop_mapping(operation, plan, left, right, output))}

type IndependentPlannedModel<-operation: Elementwise.Operation, -left: Core.Dynamic, -right: Core.Dynamic,
  -result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  -plan: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>> is Data:
  IndependentPlannedModel{-mapping: Laws.IndependentPlanned(operation, left, right, result, plan)}

def independent_planned_pair(+operation: Elementwise.Operation, -left: Core.Dynamic, -right: Core.Dynamic,
  result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  plan: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>, -shared: Elementwise.PlannedMeaning(operation, left, right, result, plan)) ->
  IndependentPlannedModel<operation, left, right, result, plan>:
  match plan:
    case Fail{error}:
      IndependentPlannedModel{Unit{}}
    case Done{+plan}:
      IndependentPlannedModel{{\{Laws.IndependentModel: independent => independent} : Laws.IndependentModel<operation, plan, left, right, result> -> Laws.IndependentMapping(operation, plan, left, right, result)}(independent_pair_mapping(operation, plan, left, right, result, shared))}

def independent_binary_reasoning(+operation: Elementwise.Operation, -left: Core.Dynamic, -right: Core.Dynamic) ->
  Laws.IndependentBinaryModel<operation, left, right, Elementwise.binary(operation, left, right)>:
  Laws.IndependentBinaryModel{
    {\{Elementwise.BinaryModel: mapping =>
      {\{IndependentPlannedModel: independent => independent} :
        IndependentPlannedModel<operation, left, right, Elementwise.binary(operation, left, right), Elementwise.plan_for(left, right)> ->
          Laws.IndependentPlanned(operation, left, right, Elementwise.binary(operation, left, right), Elementwise.plan_for(left, right))}(independent_planned_pair(operation, left, right, Elementwise.binary(operation, left, right), Elementwise.plan_for(left, right), mapping))
    } : Elementwise.BinaryModel<operation, left, right, Elementwise.binary(operation, left, right)> ->
      Laws.IndependentPlanned(operation, left, right, Elementwise.binary(operation, left, right), Elementwise.plan_for(left, right))}(successful_binary_mapping(operation, left, right))}

def independent_binary_mapping(+operation: Elementwise.Operation, left: Core.Dynamic, right: Core.Dynamic) ->
  Laws.IndependentBinaryModel<operation, left, right, Elementwise.binary(operation, left, right)>:
  Laws.IndependentBinaryModel{
    {\{Laws.IndependentBinaryModel: mapping => mapping} :
      Laws.IndependentBinaryModel<operation, left, right, Elementwise.binary(operation, left, right)> ->
        Laws.IndependentPlanned(operation, left, right, Elementwise.binary(operation, left, right), Elementwise.plan_for(left, right))}(independent_binary_reasoning(operation, left, right))}

def Laws.independent_elementwise_loop(operation, plan, left, right, output):
  independent_loop_mapping(operation, plan, left, right, output)

def Laws.independent_elementwise_binary(operation, left, right):
  independent_binary_mapping(operation, left, right)

def Laws.public_add_dispatch(left, right):
  {==}

def Laws.public_sub_dispatch(left, right):
  {==}

def Laws.public_mul_dispatch(left, right):
  {==}
