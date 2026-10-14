import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ../storage/clone.bend as Clone
import ../storage/arith.bend as Arith
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./construction-LAWS.bend as Laws
import ./core-PROOF.bend as CoreProof
import ./word-injective.bend as Injective
import ../shape/nat_proofs.bend as Nats
import ../shape/LAWS.bend as ShapeLaws
import ../shape/PROOF.bend as ShapeProof
import ./word-conversion.bend as WordConversion
import ./allocation-spec.bend as Allocation
import ./allocation-PROOF.bend as AllocationProof
import ./transform-totality.bend as TransformTotality
import ./transform-LAWS.bend as TransformLaws
import ./transform-PROOF.bend as TransformProof
import ./construction-totality.bend as Totality

def clone_first_checked_view(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32,
  pair: Storage.Storage & Storage.Storage, -same: {pair == Storage.clone(original) : Storage.Storage & Storage.Storage}) ->
  {Laws.clone_first_checked(shape, Core.clone_result(shape, original, valid, pair, same), index) ==
    Lifecycle.checked_observe(Pair.fst(Storage.Storage, Storage.Storage, pair), index) : Maybe<&2, F32>}:
  (first, second) = pair
  {==}

def clone_second_checked_view(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32,
  pair: Storage.Storage & Storage.Storage, -same: {pair == Storage.clone(original) : Storage.Storage & Storage.Storage}) ->
  {Laws.clone_second_checked(shape, Core.clone_result(shape, original, valid, pair, same), index) ==
    Lifecycle.checked_observe(Pair.snd(Storage.Storage, Storage.Storage, pair), index) : Maybe<&2, F32>}:
  (first, second) = pair
  {==}

def clone_first_model_view(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32,
  pair: Storage.Storage & Storage.Storage, -same: {pair == Storage.clone(original) : Storage.Storage & Storage.Storage}) ->
  {Laws.clone_first_model(shape, Core.clone_result(shape, original, valid, pair, same), index) ==
    Lifecycle.observe(Pair.fst(Storage.Storage, Storage.Storage, pair), index) : Maybe<&2, F32>}:
  (first, second) = pair
  {==}

def clone_second_model_view(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32,
  pair: Storage.Storage & Storage.Storage, -same: {pair == Storage.clone(original) : Storage.Storage & Storage.Storage}) ->
  {Laws.clone_second_model(shape, Core.clone_result(shape, original, valid, pair, same), index) ==
    Lifecycle.observe(Pair.snd(Storage.Storage, Storage.Storage, pair), index) : Maybe<&2, F32>}:
  (first, second) = pair
  {==}

def storage_clone_first_checked_view(+depth: Nat, +length: Nat, -array: Array<F32>, +index: U32,
  pair: Array<F32> & Array<F32>, -exact: {pair == Array.clone(F32, array) : Array<F32> & Array<F32>},
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool}) ->
  {Lifecycle.checked_observe(Pair.fst(Storage.Storage, Storage.Storage, Storage.finish_clone(depth, length, array, pair, exact, balanced, valid)), index) ==
    Lifecycle.checked_observe_array(depth, length, Pair.fst(Array<F32>, Array<F32>, pair), index) : Maybe<&2, F32>}:
  (first, second) = pair
  {==}

def storage_clone_second_checked_view(+depth: Nat, +length: Nat, -array: Array<F32>, +index: U32,
  pair: Array<F32> & Array<F32>, -exact: {pair == Array.clone(F32, array) : Array<F32> & Array<F32>},
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool}) ->
  {Lifecycle.checked_observe(Pair.snd(Storage.Storage, Storage.Storage, Storage.finish_clone(depth, length, array, pair, exact, balanced, valid)), index) ==
    Lifecycle.checked_observe_array(depth, length, Pair.snd(Array<F32>, Array<F32>, pair), index) : Maybe<&2, F32>}:
  (first, second) = pair
  {==}

def storage_clone_first_model_view(+depth: Nat, +length: Nat, -array: Array<F32>, +index: U32,
  pair: Array<F32> & Array<F32>, -exact: {pair == Array.clone(F32, array) : Array<F32> & Array<F32>},
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool}) ->
  {Lifecycle.observe(Pair.fst(Storage.Storage, Storage.Storage, Storage.finish_clone(depth, length, array, pair, exact, balanced, valid)), index) ==
    Model.observe(F32, depth, Pair.fst(Array<F32>, Array<F32>, pair), Refinement.route(depth, index)) : Maybe<&2, F32>}:
  (first, second) = pair
  {==}

def storage_clone_second_model_view(+depth: Nat, +length: Nat, -array: Array<F32>, +index: U32,
  pair: Array<F32> & Array<F32>, -exact: {pair == Array.clone(F32, array) : Array<F32> & Array<F32>},
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool}) ->
  {Lifecycle.observe(Pair.snd(Storage.Storage, Storage.Storage, Storage.finish_clone(depth, length, array, pair, exact, balanced, valid)), index) ==
    Model.observe(F32, depth, Pair.snd(Array<F32>, Array<F32>, pair), Refinement.route(depth, index)) : Maybe<&2, F32>}:
  (first, second) = pair
  {==}

def clone_owned_meaning(-shape: Shape.Shape, +depth: Nat, +length: Nat, -array: Array<F32>,
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -length_exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Laws.CloneMeaning<shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, length_exact}>:
  Laws.CloneMeaning{
    index => Equal.trans(Maybe<&2, F32>,
        Laws.clone_first_checked(shape, Core.clone(shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, length_exact}), index),
        Lifecycle.checked_observe(Pair.fst(Storage.Storage, Storage.Storage, Storage.clone(Storage.Owned{depth, length, array, balanced, valid})), index),
        Lifecycle.checked_observe_array(depth, length, array, index),
        clone_first_checked_view(shape, Storage.Owned{depth, length, array, balanced, valid}, length_exact, index,
          Storage.clone(Storage.Owned{depth, length, array, balanced, valid}), {==}),
        Equal.trans(Maybe<&2, F32>,
          Lifecycle.checked_observe(Pair.fst(Storage.Storage, Storage.Storage, Storage.clone(Storage.Owned{depth, length, array, balanced, valid})), index),
          Lifecycle.checked_observe_array(depth, length, Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, array)), index),
          Lifecycle.checked_observe_array(depth, length, array, index),
          storage_clone_first_checked_view(depth, length, array, index, Array.clone(F32, array), {==}, balanced, valid),
          Equal.cong(Array<F32>, Maybe<&2, F32>, a => Lifecycle.checked_observe_array(depth, length, a, index),
            Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, array)), array, Clone.first_equal(F32, depth, array, Clone.certify(F32, depth, array, balanced))))),
    index => Equal.trans(Maybe<&2, F32>,
        Laws.clone_second_checked(shape, Core.clone(shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, length_exact}), index),
        Lifecycle.checked_observe(Pair.snd(Storage.Storage, Storage.Storage, Storage.clone(Storage.Owned{depth, length, array, balanced, valid})), index),
        Lifecycle.checked_observe_array(depth, length, array, index),
        clone_second_checked_view(shape, Storage.Owned{depth, length, array, balanced, valid}, length_exact, index,
          Storage.clone(Storage.Owned{depth, length, array, balanced, valid}), {==}),
        Equal.trans(Maybe<&2, F32>,
          Lifecycle.checked_observe(Pair.snd(Storage.Storage, Storage.Storage, Storage.clone(Storage.Owned{depth, length, array, balanced, valid})), index),
          Lifecycle.checked_observe_array(depth, length, Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, array)), index),
          Lifecycle.checked_observe_array(depth, length, array, index),
          storage_clone_second_checked_view(depth, length, array, index, Array.clone(F32, array), {==}, balanced, valid),
          Equal.cong(Array<F32>, Maybe<&2, F32>, a => Lifecycle.checked_observe_array(depth, length, a, index),
            Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, array)), array, Clone.second_equal(F32, depth, array, Clone.certify(F32, depth, array, balanced))))),
    index => Equal.trans(Maybe<&2, F32>,
        Laws.clone_first_model(shape, Core.clone(shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, length_exact}), index),
        Lifecycle.observe(Pair.fst(Storage.Storage, Storage.Storage, Storage.clone(Storage.Owned{depth, length, array, balanced, valid})), index),
        Model.observe(F32, depth, array, Refinement.route(depth, index)),
        clone_first_model_view(shape, Storage.Owned{depth, length, array, balanced, valid}, length_exact, index,
          Storage.clone(Storage.Owned{depth, length, array, balanced, valid}), {==}),
        Equal.trans(Maybe<&2, F32>,
          Lifecycle.observe(Pair.fst(Storage.Storage, Storage.Storage, Storage.clone(Storage.Owned{depth, length, array, balanced, valid})), index),
          Model.observe(F32, depth, Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, array)), Refinement.route(depth, index)),
          Model.observe(F32, depth, array, Refinement.route(depth, index)),
          storage_clone_first_model_view(depth, length, array, index, Array.clone(F32, array), {==}, balanced, valid),
          Equal.cong(Array<F32>, Maybe<&2, F32>, a => Model.observe(F32, depth, a, Refinement.route(depth, index)),
            Pair.fst(Array<F32>, Array<F32>, Array.clone(F32, array)), array, Clone.first_equal(F32, depth, array, Clone.certify(F32, depth, array, balanced))))),
    index => Equal.trans(Maybe<&2, F32>,
        Laws.clone_second_model(shape, Core.clone(shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, length_exact}), index),
        Lifecycle.observe(Pair.snd(Storage.Storage, Storage.Storage, Storage.clone(Storage.Owned{depth, length, array, balanced, valid})), index),
        Model.observe(F32, depth, array, Refinement.route(depth, index)),
        clone_second_model_view(shape, Storage.Owned{depth, length, array, balanced, valid}, length_exact, index,
          Storage.clone(Storage.Owned{depth, length, array, balanced, valid}), {==}),
        Equal.trans(Maybe<&2, F32>,
          Lifecycle.observe(Pair.snd(Storage.Storage, Storage.Storage, Storage.clone(Storage.Owned{depth, length, array, balanced, valid})), index),
          Model.observe(F32, depth, Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, array)), Refinement.route(depth, index)),
          Model.observe(F32, depth, array, Refinement.route(depth, index)),
          storage_clone_second_model_view(depth, length, array, index, Array.clone(F32, array), {==}, balanced, valid),
          Equal.cong(Array<F32>, Maybe<&2, F32>, a => Model.observe(F32, depth, a, Refinement.route(depth, index)),
            Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, array)), array, Clone.second_equal(F32, depth, array, Clone.certify(F32, depth, array, balanced)))))}

def clone_meaning(-shape: Shape.Shape, original: Core.Tensor<shape>) -> Laws.CloneMeaning<shape, original>:
  Core.TTensor{storage, length_exact} = original
  match storage:
    case Storage.Empty{}:
      Laws.CloneMeaning{index => {==}, index => {==}, index => {==}, index => {==}}
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      clone_owned_meaning(shape, depth, length, array, balanced, valid, length_exact)

def Laws.public_clone_elements(shape, original):
  clone_meaning(shape, original)

def FillElements(-shape: Shape.Shape, +values: List<&2, F32>, +cursor: Nat, -tensor: Core.Tensor<shape>) -> Type:
  @+index: U32 -> @+offset: Nat -> @at: {U32.to_nat(index) == Nat.add(cursor, offset) : Nat} ->
    @inside: Shape.LT(offset, List.length(&2, F32, values)) ->
      Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
        {Core.checked_observe(shape, tensor, index) == Laws.list_element(values, offset) : Maybe<&2, F32>}

def FillFrame(-shape: Shape.Shape, -original: Core.Tensor<shape>, +cursor: Nat, -tensor: Core.Tensor<shape>) -> Type:
  @+index: U32 -> @earlier: Shape.LT(U32.to_nat(index), cursor) ->
    @checked: Core.ObservationChecked(Core.Packed{shape, original}, index) ->
      Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
        {Core.checked_observe(shape, tensor, index) == Core.checked_observe(shape, original, index) : Maybe<&2, F32>}

def FillMeaning(-shape: Shape.Shape, -original: Core.Tensor<shape>, +values: List<&2, F32>, +cursor: Nat,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Type:
  match result:
    case Fail{error}:
      Unit
    case Done{tensor}:
      FillElements(shape, values, cursor, tensor) & FillFrame(shape, original, cursor, tensor)

type FillOutcome<-shape: Shape.Shape, -original: Core.Tensor<shape>, -values: List<&2, F32>, -cursor: Nat,
  -result: Result<&2, &1, Core.Error, Core.Tensor<shape>>> is Data:
  FillOutcome{-meaning: FillMeaning(shape, original, values, cursor, result)}

type Written<-shape: Shape.Shape, -original: Core.Tensor<shape>, -tensor: Core.Tensor<shape>, -cursor: Nat, -value: F32> is Data:
  Written{-point: @+index: U32 -> @at: {U32.to_nat(index) == cursor : Nat} ->
      Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
        {Core.checked_observe(shape, tensor, index) == Some{value} : Maybe<&2, F32>},
    -frame: FillFrame(shape, original, cursor, tensor)}

def lt_zero(+number: Nat, proof: Shape.LT(number, 0n)) -> Empty:
  match number:
    case 0n:
      proof
    case 1n+p:
      proof

def lt_self_successor(number: Nat) -> Shape.LT(number, 1n+number):
  match number:
    case 0n:
      Unit{}
    case 1n+p:
      lt_self_successor(p)

def lt_extend(+number: Nat, +bound: Nat, proof: Shape.LT(number, bound)) -> Shape.LT(number, 1n+bound):
  match number bound:
    case 0n 0n:
      match proof:
    case 0n 1n+p:
      Unit{}
    case 1n+p 0n:
      match proof:
    case 1n+p 1n+q:
      lt_extend(p, q, proof)

def add_zero_right(number: Nat) -> {Nat.add(number, 0n) == number : Nat}:
  match number:
    case 0n:
      {==}
    case 1n+p:
      Equal.cong(Nat, Nat, n => 1n+n, Nat.add(p, 0n), p, add_zero_right(p))

def fill_head(-shape: Shape.Shape, -written: Core.Tensor<shape>, -tensor: Core.Tensor<shape>, +cursor: Nat, +value: F32,
  point: @+index: U32 -> @at: {U32.to_nat(index) == cursor : Nat} ->
    Core.ObservationChecked(Core.Packed{shape, written}, index) &
      {Core.checked_observe(shape, written, index) == Some{value} : Maybe<&2, F32>},
  frame: FillFrame(shape, written, 1n+cursor, tensor), +index: U32, +at: {U32.to_nat(index) == cursor : Nat}) ->
  Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
    {Core.checked_observe(shape, tensor, index) == Some{value} : Maybe<&2, F32>}:
  {\{Tuple: checked => observed =>
    {\{Tuple: after_checked => retained =>
      (after_checked, Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, tensor, index), Core.checked_observe(shape, written, index),
        Some{value}, retained, observed))} :
      (Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
        {Core.checked_observe(shape, tensor, index) == Core.checked_observe(shape, written, index) : Maybe<&2, F32>}) ->
      (Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
        {Core.checked_observe(shape, tensor, index) == Some{value} : Maybe<&2, F32>})}(frame(index)(
          %Equal.sym(Nat, U32.to_nat(index), cursor, at) : Shape.LT(_, 1n+cursor)
          lt_self_successor(cursor))(checked))
  } :
    (Core.ObservationChecked(Core.Packed{shape, written}, index) &
      {Core.checked_observe(shape, written, index) == Some{value} : Maybe<&2, F32>}) ->
    (Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
      {Core.checked_observe(shape, tensor, index) == Some{value} : Maybe<&2, F32>})}(point(index)(at))

def fill_elements_advance(-shape: Shape.Shape, +value: F32, +rest: List<&2, F32>, +cursor: Nat,
  -written: Core.Tensor<shape>, -tensor: Core.Tensor<shape>,
  point: @+index: U32 -> @at: {U32.to_nat(index) == cursor : Nat} ->
    Core.ObservationChecked(Core.Packed{shape, written}, index) &
      {Core.checked_observe(shape, written, index) == Some{value} : Maybe<&2, F32>},
  mapping: FillElements(shape, rest, 1n+cursor, tensor), frame: FillFrame(shape, written, 1n+cursor, tensor),
  +index: U32, +offset: Nat, at: {U32.to_nat(index) == Nat.add(cursor, offset) : Nat},
  inside: Shape.LT(offset, List.length(&2, F32, value <> rest))) ->
  Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
    {Core.checked_observe(shape, tensor, index) == Laws.list_element(value <> rest, offset) : Maybe<&2, F32>}:
  match offset:
    case 0n:
      fill_head(shape, written, tensor, cursor, value, point, frame, index,
        Equal.trans(Nat, U32.to_nat(index), Nat.add(cursor, 0n), cursor, at, add_zero_right(cursor)))
    case 1n+p:
      mapping(index)(p)(Equal.trans(Nat, U32.to_nat(index), Nat.add(cursor, 1n+p), Nat.add(1n+cursor, p), at, Arith.add_succ(cursor, p)))(inside)

def fill_frame_advance(-shape: Shape.Shape, -original: Core.Tensor<shape>, -written: Core.Tensor<shape>, -tensor: Core.Tensor<shape>, +cursor: Nat,
  before: FillFrame(shape, original, cursor, written), after: FillFrame(shape, written, 1n+cursor, tensor),
  +index: U32, +earlier: Shape.LT(U32.to_nat(index), cursor), checked: Core.ObservationChecked(Core.Packed{shape, original}, index)) ->
  Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
    {Core.checked_observe(shape, tensor, index) == Core.checked_observe(shape, original, index) : Maybe<&2, F32>}:
  {\{Tuple: middle_checked => middle_equal =>
    {\{Tuple: after_checked => after_equal =>
      (after_checked, Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, tensor, index), Core.checked_observe(shape, written, index),
        Core.checked_observe(shape, original, index), after_equal, middle_equal))} :
      (Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
        {Core.checked_observe(shape, tensor, index) == Core.checked_observe(shape, written, index) : Maybe<&2, F32>}) ->
      (Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
        {Core.checked_observe(shape, tensor, index) == Core.checked_observe(shape, original, index) : Maybe<&2, F32>})}(after(index)(lt_extend(U32.to_nat(index), cursor, earlier))(middle_checked))
  } :
    (Core.ObservationChecked(Core.Packed{shape, written}, index) &
      {Core.checked_observe(shape, written, index) == Core.checked_observe(shape, original, index) : Maybe<&2, F32>}) ->
    (Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
      {Core.checked_observe(shape, tensor, index) == Core.checked_observe(shape, original, index) : Maybe<&2, F32>})}(before(index)(earlier)(checked))

def fill_advance(-shape: Shape.Shape, +value: F32, +rest: List<&2, F32>, +cursor: Nat,
  -original: Core.Tensor<shape>, -written: Core.Tensor<shape>,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>, -point: Written<shape, original, written, cursor, value>,
  -next: FillMeaning(shape, written, rest, 1n+cursor, result)) -> FillOutcome<shape, original, value <> rest, cursor, result>:
  match result:
    case Fail{error}:
      FillOutcome{Unit{}}
    case Done{tensor}:
      FillOutcome{
        (index => offset => at => inside =>
          {\{Written: observed => before =>
            {\{Tuple: mapping => after => fill_elements_advance(shape, value, rest, cursor, written, tensor,
              observed, mapping, after, index, offset, at, inside)} :
              (FillElements(shape, rest, 1n+cursor, tensor) & FillFrame(shape, written, 1n+cursor, tensor)) ->
                (Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
                  {Core.checked_observe(shape, tensor, index) == Laws.list_element(value <> rest, offset) : Maybe<&2, F32>})}(next)
          } : Written<shape, original, written, cursor, value> ->
            (Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
              {Core.checked_observe(shape, tensor, index) == Laws.list_element(value <> rest, offset) : Maybe<&2, F32>})}(point),
          index => earlier => checked =>
            {\{Written: observed => before =>
              {\{Tuple: mapping => after => fill_frame_advance(shape, original, written, tensor, cursor, before, after, index, earlier, checked)} :
                (FillElements(shape, rest, 1n+cursor, tensor) & FillFrame(shape, written, 1n+cursor, tensor)) ->
                  (Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
                    {Core.checked_observe(shape, tensor, index) == Core.checked_observe(shape, original, index) : Maybe<&2, F32>})}(next)
            } : Written<shape, original, written, cursor, value> ->
              (Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
                {Core.checked_observe(shape, tensor, index) == Core.checked_observe(shape, original, index) : Maybe<&2, F32>})}(point))}

type ElementPoint<-shape: Shape.Shape, -values: List<&2, F32>, -tensor: Core.Tensor<shape>, -index: U32> is Data:
  ElementPoint{-meaning: Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
    ({Core.checked_observe(shape, tensor, index) == Laws.list_element(values, U32.to_nat(index)) : Maybe<&2, F32>} &
      {Core.observe(shape, tensor, index) == Laws.list_element(values, U32.to_nat(index)) : Maybe<&2, F32>})}

def element_point(-shape: Shape.Shape, +values: List<&2, F32>, -tensor: Core.Tensor<shape>, +index: U32,
  -checked: Core.ObservationChecked(Core.Packed{shape, tensor}, index),
  -observed: {Core.checked_observe(shape, tensor, index) == Laws.list_element(values, U32.to_nat(index)) : Maybe<&2, F32>}) ->
  ElementPoint<shape, values, tensor, index>:
  ElementPoint{(checked, (observed,
    {\{Core.ObservationModel: same =>
      Equal.trans(Maybe<&2, F32>, Core.observe(shape, tensor, index), Core.checked_observe(shape, tensor, index), Laws.list_element(values, U32.to_nat(index)),
        Equal.sym(Maybe<&2, F32>, Core.checked_observe(shape, tensor, index), Core.observe(shape, tensor, index), same), observed)
    } : Core.ObservationModel<shape, tensor, index, checked> ->
      {Core.observe(shape, tensor, index) == Laws.list_element(values, U32.to_nat(index)) : Maybe<&2, F32>}}(Core.observation_model(shape, tensor, index, checked))))}

type ElementsOutcome<-shape: Shape.Shape, -values: List<&2, F32>, -result: Result<&2, &1, Core.Error, Core.Tensor<shape>>> is Data:
  ElementsOutcome{-meaning: Laws.FromValuesMeaning(shape, values, result)}

def complete_fill(-shape: Shape.Shape, +values: List<&2, F32>, -original: Core.Tensor<shape>,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>,
  -count: {Shape.logical_numel(Shape.dimensions(shape)) == List.length(&2, F32, values) : Nat},
  -fill: FillMeaning(shape, original, values, 0n, result)) -> ElementsOutcome<shape, values, result>:
  match result:
    case Fail{error}:
      ElementsOutcome{Unit{}}
    case Done{tensor}:
      ElementsOutcome{index => inside =>
        {\{Tuple: mapping => frame =>
          {\{Tuple: checked => observed =>
            {\{ElementPoint: meaning => meaning} : ElementPoint<shape, values, tensor, index> ->
              (Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
                ({Core.checked_observe(shape, tensor, index) == Laws.list_element(values, U32.to_nat(index)) : Maybe<&2, F32>} &
                  {Core.observe(shape, tensor, index) == Laws.list_element(values, U32.to_nat(index)) : Maybe<&2, F32>}))}(element_point(shape, values, tensor, index, checked, observed))
          } :
            (Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
              {Core.checked_observe(shape, tensor, index) == Laws.list_element(values, U32.to_nat(index)) : Maybe<&2, F32>}) ->
            (Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
              ({Core.checked_observe(shape, tensor, index) == Laws.list_element(values, U32.to_nat(index)) : Maybe<&2, F32>} &
                {Core.observe(shape, tensor, index) == Laws.list_element(values, U32.to_nat(index)) : Maybe<&2, F32>}))}(mapping(index)(U32.to_nat(index))({==})(
                  %count : Shape.LT(U32.to_nat(index), _)
                  inside))
        } : (FillElements(shape, values, 0n, tensor) & FillFrame(shape, original, 0n, tensor)) ->
          (Core.ObservationChecked(Core.Packed{shape, tensor}, index) &
            ({Core.checked_observe(shape, tensor, index) == Laws.list_element(values, U32.to_nat(index)) : Maybe<&2, F32>} &
              {Core.observe(shape, tensor, index) == Laws.list_element(values, U32.to_nat(index)) : Maybe<&2, F32>}))}(fill)}

type ConstructionOutcome<-values: List<&2, F32>, -result: Result<&2, &1, Core.Error, Core.Dynamic>> is Data:
  ConstructionOutcome{-meaning: Laws.ConstructionMeaning(values, result)}

def construction_values_elements(+shape: Shape.Shape, +values: List<&2, F32>,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>, -meaning: Laws.FromValuesMeaning(shape, values, result)) ->
  ConstructionOutcome<values, Core.construct_values(shape, result)>:
  match result:
    case Fail{error}:
      ConstructionOutcome{Unit{}}
    case Done{tensor}:
      ConstructionOutcome{meaning}

def construction_shape_elements(+dimensions: List<&2, U32>, +values: List<&2, F32>, result: Shape.ConstructResult<dimensions>) ->
  ConstructionOutcome<values, Core.construct_shape(dimensions, values, result)>:
  match result:
    case Shape.SError{error}:
      ConstructionOutcome{Unit{}}
    case Shape.SBuilt{+shape, preserved}:
      ConstructionOutcome{
        {\{ConstructionOutcome: meaning => meaning} : ConstructionOutcome<values, Core.construct_values(shape, Core.from_values(shape, values))> ->
          Laws.ConstructionMeaning(values, Core.construct_values(shape, Core.from_values(shape, values)))}(construction_values_elements(shape, values, Core.from_values(shape, values),
            {\{Laws.FromValuesModel: meaning => meaning} : Laws.FromValuesModel<shape, values> ->
              Laws.FromValuesMeaning(shape, values, Core.from_values(shape, values))}(Laws.public_from_values_elements(shape, values))))}

def Laws.public_construction_elements(dimensions, values):
  Laws.ConstructionModel{
    {\{ConstructionOutcome: meaning => meaning} : ConstructionOutcome<values, Core.construct(dimensions, values)> ->
      Laws.ConstructionMeaning(values, Core.construct(dimensions, values))}(construction_shape_elements(dimensions, values, Shape.construct(dimensions)))}

def supplied_count(+shape: Shape.Shape, +values: List<&2, F32>,
  decided: {True{} == Nat.is_eq(List.length(&2, F32, values), Shape.elements(shape)) : Bool}) ->
  {Shape.logical_numel(Shape.dimensions(shape)) == List.length(&2, F32, values) : Nat}:
  Equal.trans(Nat, Shape.logical_numel(Shape.dimensions(shape)), Shape.elements(shape), List.length(&2, F32, values),
    Equal.sym(Nat, Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)), Shape.elements_exact(shape)),
    Equal.sym(Nat, List.length(&2, F32, values), Shape.elements(shape),
      Lifecycle.natural_equal(List.length(&2, F32, values), Shape.elements(shape), decided)))

def written_point(-shape: Shape.Shape, -tensor: Core.Tensor<shape>, +cursor: Nat, +index: U32, +value: F32,
  converted: {U32.to_nat(index) == cursor : Nat}, checked: Core.ObservationChecked(Core.Packed{shape, tensor}, index),
  observed: {Core.checked_observe(shape, tensor, index) == Some{value} : Maybe<&2, F32>}, +other: U32,
  at: {U32.to_nat(other) == cursor : Nat}) ->
  Core.ObservationChecked(Core.Packed{shape, tensor}, other) &
    {Core.checked_observe(shape, tensor, other) == Some{value} : Maybe<&2, F32>}:
  %Injective.u32_natural_injective(index, other)(Equal.trans(Nat, U32.to_nat(index), cursor, U32.to_nat(other), converted,
    Equal.sym(Nat, U32.to_nat(other), cursor, at))) :
    Core.ObservationChecked(Core.Packed{shape, tensor}, _) &
      {Core.checked_observe(shape, tensor, _) == Some{value} : Maybe<&2, F32>}
  (checked, observed)

def written_model(-shape: Shape.Shape, -original: Core.Tensor<shape>, -tensor: Core.Tensor<shape>, +cursor: Nat,
  +index: U32, +value: F32, -unit: Unit, -converted: {U32.to_nat(index) == cursor : Nat},
  -meaning: Core.CheckedUpdateMeaning(shape, original, index, value, (tensor, Done{unit}))) -> Written<shape, original, tensor, cursor, value>:
  Written{
    other => at =>
      {\{Tuple: checked => rest =>
        {\{Tuple: observed => frame => written_point(shape, tensor, cursor, index, value, converted, checked, observed, other, at)} :
          ({Core.checked_observe(shape, tensor, index) == Some{value} : Maybe<&2, F32>} &
            (@+other: U32 -> @checked: Core.ObservationChecked(Core.Packed{shape, original}, other) ->
              @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
                Core.ObservationChecked(Core.Packed{shape, tensor}, other) &
                  {Core.checked_observe(shape, tensor, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>})) ->
          (Core.ObservationChecked(Core.Packed{shape, tensor}, other) &
            {Core.checked_observe(shape, tensor, other) == Some{value} : Maybe<&2, F32>})}(rest)
      } : Core.CheckedUpdateMeaning(shape, original, index, value, (tensor, Done{unit})) ->
        (Core.ObservationChecked(Core.Packed{shape, tensor}, other) &
          {Core.checked_observe(shape, tensor, other) == Some{value} : Maybe<&2, F32>})}(meaning),
    other => earlier => before_checked =>
      {\{Tuple: checked => rest =>
        {\{Tuple: observed => frame => frame(other)(before_checked)(
          %Equal.sym(Nat, U32.to_nat(index), cursor, converted) : {Nat.is_eq(_, U32.to_nat(other)) == False{} : Bool}
          CoreProof.lt_big_small_different(U32.to_nat(other), cursor, earlier))} :
          ({Core.checked_observe(shape, tensor, index) == Some{value} : Maybe<&2, F32>} &
            (@+other: U32 -> @checked: Core.ObservationChecked(Core.Packed{shape, original}, other) ->
              @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
                Core.ObservationChecked(Core.Packed{shape, tensor}, other) &
                  {Core.checked_observe(shape, tensor, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>})) ->
          (Core.ObservationChecked(Core.Packed{shape, tensor}, other) &
            {Core.checked_observe(shape, tensor, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>})}(rest)
      } : Core.CheckedUpdateMeaning(shape, original, index, value, (tensor, Done{unit})) ->
        (Core.ObservationChecked(Core.Packed{shape, tensor}, other) &
          {Core.checked_observe(shape, tensor, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>})}(meaning)}

def fill_write_pair(-shape: Shape.Shape, +value: F32, +rest: List<&2, F32>, +cursor: Nat, +index: U32,
  -original: Core.Tensor<shape>, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>,
  -same: {pair == Core.write_flat(shape, original, index, value) : Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>},
  -converted: {U32.to_nat(index) == cursor : Nat}, -model: Core.CheckedUpdateModel<shape, original, index, value>,
  -rec: @tensor: Core.Tensor<shape> -> FillOutcome<shape, tensor, rest, 1n+cursor, Core.fill_values(rest, shape, tensor, 1n+cursor)>) ->
  FillOutcome<shape, original, value <> rest, cursor,
    Core.fill_values_step(shape, tensor => Core.fill_values(rest, shape, tensor, 1n+cursor), pair)>:
  (tensor, result) = pair
  match result:
    case Fail{error}:
      FillOutcome{Unit{}}
    case Done{unit}:
      FillOutcome{
        {\{FillOutcome: meaning => meaning} : FillOutcome<shape, original, value <> rest, cursor, Core.fill_values(rest, shape, tensor, 1n+cursor)> ->
          FillMeaning(shape, original, value <> rest, cursor, Core.fill_values(rest, shape, tensor, 1n+cursor))}(fill_advance(shape, value, rest, cursor,
            original, tensor, Core.fill_values(rest, shape, tensor, 1n+cursor),
            written_model(shape, original, tensor, cursor, index, value, unit, converted,
              %Equal.sym(Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>, (tensor, Done{unit}), Core.write_flat(shape, original, index, value), same) :
                Core.CheckedUpdateMeaning(shape, original, index, value, _)
              {\{Core.CheckedUpdateModel: meaning => meaning} : Core.CheckedUpdateModel<shape, original, index, value> ->
                Core.CheckedUpdateMeaning(shape, original, index, value, Core.write_flat(shape, original, index, value))}(model)),
            {\{FillOutcome: meaning => meaning} : FillOutcome<shape, tensor, rest, 1n+cursor, Core.fill_values(rest, shape, tensor, 1n+cursor)> ->
              FillMeaning(shape, tensor, rest, 1n+cursor, Core.fill_values(rest, shape, tensor, 1n+cursor))}(rec(tensor))))}

def fill_decision_model(-shape: Shape.Shape, +value: F32, +rest: List<&2, F32>, +cursor: Nat, +index: U32,
  -original: Core.Tensor<shape>, decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>,
  -rec: @tensor: Core.Tensor<shape> -> FillOutcome<shape, tensor, rest, 1n+cursor, Core.fill_values(rest, shape, tensor, 1n+cursor)>) ->
  FillOutcome<shape, original, value <> rest, cursor, Core.fill_values_decided(shape, cursor, index, value, original,
    tensor => Core.fill_values(rest, shape, tensor, 1n+cursor), decision)>:
  match decision:
    case Shape.BFalse{equal}:
      FillOutcome{Unit{}}
    case Shape.BTrue{equal}:
      FillOutcome{
        {\{FillOutcome: meaning => meaning} :
          FillOutcome<shape, original, value <> rest, cursor,
            Core.fill_values_step(shape, tensor => Core.fill_values(rest, shape, tensor, 1n+cursor), Core.write_flat(shape, original, index, value))> ->
          FillMeaning(shape, original, value <> rest, cursor,
            Core.fill_values_step(shape, tensor => Core.fill_values(rest, shape, tensor, 1n+cursor), Core.write_flat(shape, original, index, value)))}(fill_write_pair(shape, value, rest, cursor, index, original, Core.write_flat(shape, original, index, value), {==},
                Lifecycle.natural_equal(U32.to_nat(index), cursor, equal), CoreProof.checked_update_model(shape, original, index, value), rec))}

def fill_model(+values: List<&2, F32>, -shape: Shape.Shape, -original: Core.Tensor<shape>, +cursor: Nat) ->
  FillOutcome<shape, original, values, cursor, Core.fill_values(values, shape, original, cursor)>:
  match values:
    case Nil{}:
      FillOutcome{(index => offset => at => inside =>
        Empty.absurd(Core.ObservationChecked(Core.Packed{shape, original}, index) &
          {Core.checked_observe(shape, original, index) == Laws.list_element(Nil{}, offset) : Maybe<&2, F32>}, lt_zero(offset, inside)),
        index => earlier => checked => (checked, {==}))}
    case +value <> +rest:
      FillOutcome{
        {\{FillOutcome: meaning => meaning} :
          FillOutcome<shape, original, value <> rest, cursor,
            Core.fill_values_decided(shape, cursor, U32.from_nat(cursor), value, original,
              tensor => Core.fill_values(rest, shape, tensor, 1n+cursor), Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)))> ->
          FillMeaning(shape, original, value <> rest, cursor, Core.fill_values(value <> rest, shape, original, cursor))}(fill_decision_model(shape, value, rest,
            cursor, U32.from_nat(cursor), original, Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)),
            tensor => fill_model(rest, shape, tensor, 1n+cursor)))}

def from_values_new_elements(-shape: Shape.Shape, +values: List<&2, F32>,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>,
  -count: {Shape.logical_numel(Shape.dimensions(shape)) == List.length(&2, F32, values) : Nat}) ->
  ElementsOutcome<shape, values, Core.from_values_new(shape, values, result)>:
  match result:
    case Fail{error}:
      ElementsOutcome{Unit{}}
    case Done{tensor}:
      ElementsOutcome{
        {\{ElementsOutcome: meaning => meaning} : ElementsOutcome<shape, values, Core.fill_values(values, shape, tensor, 0n)> ->
          Laws.FromValuesMeaning(shape, values, Core.fill_values(values, shape, tensor, 0n))}(complete_fill(shape, values, tensor,
            Core.fill_values(values, shape, tensor, 0n), count,
            {\{FillOutcome: meaning => meaning} : FillOutcome<shape, tensor, values, 0n, Core.fill_values(values, shape, tensor, 0n)> ->
              FillMeaning(shape, tensor, values, 0n, Core.fill_values(values, shape, tensor, 0n))}(fill_model(values, shape, tensor, 0n))))}

def from_values_counted_elements(+shape: Shape.Shape, +values: List<&2, F32>,
  decision: Shape.BoolCase<Nat.is_eq(List.length(&2, F32, values), Shape.elements(shape))>) ->
  ElementsOutcome<shape, values, Core.from_values_counted(shape, values, decision)>:
  match decision:
    case Shape.BFalse{equal}:
      ElementsOutcome{Unit{}}
    case Shape.BTrue{equal}:
      ElementsOutcome{
        {\{ElementsOutcome: meaning => meaning} : ElementsOutcome<shape, values, Core.from_values_new(shape, values, Core.new(shape, 0.0))> ->
          Laws.FromValuesMeaning(shape, values, Core.from_values_new(shape, values, Core.new(shape, 0.0)))}(from_values_new_elements(shape, values, Core.new(shape, 0.0), supplied_count(shape, values, equal)))}

def Laws.public_from_values_elements(shape, values):
  Laws.FromValuesModel{
    {\{ElementsOutcome: meaning => meaning} : ElementsOutcome<shape, values, Core.from_values(shape, values)> ->
      Laws.FromValuesMeaning(shape, values, Core.from_values(shape, values))}(from_values_counted_elements(shape, values,
        Shape.bool_case(Nat.is_eq(List.length(&2, F32, values), Shape.elements(shape)))))}

def add_successor(+left: Nat, +right: Nat) -> {Nat.add(left, 1n+right) == Nat.add(1n+left, right) : Nat}:
  match left:
    case 0n:
      {==}
    case 1n+p:
      Equal.cong(Nat, Nat, n => 1n+n, Nat.add(p, 1n+right), Nat.add(1n+p, right), add_successor(p, right))

def cursor_below(+cursor: Nat, +remaining: Nat, +count: Nat,
  room: Shape.LE(Nat.add(cursor, 1n+remaining), count)) -> Shape.LT(cursor, count):
  match cursor count:
    case 0n 0n:
      match room:
    case 0n 1n+q:
      Unit{}
    case 1n+p 0n:
      match room:
    case 1n+p 1n+q:
      cursor_below(p, remaining, q, room)

def cursor_exact(+shape: Shape.Shape, +cursor: Nat, room: Shape.LE(cursor, Shape.elements(shape))) ->
  {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat}:
  WordConversion.natural_word_exact(cursor, Shape.count(shape),
    %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.elements(shape), WordConversion.shape_count_exact(shape)) :
      {Nat.is_le(cursor, _) == True{} : Bool}
    Equal.sym(Bool, True{}, Nat.is_le(cursor, Shape.elements(shape)), Nats.le_true(cursor, Shape.elements(shape), room)))

def cursor_inside(+shape: Shape.Shape, +cursor: Nat, +index: U32,
  converted: {U32.to_nat(index) == cursor : Nat}, below: Shape.LT(cursor, Shape.elements(shape))) ->
  {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}:
  Equal.trans(Bool, Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))),
    Nat.is_lt(cursor, Shape.logical_numel(Shape.dimensions(shape))), True{},
    Equal.cong(Nat, Bool, n => Nat.is_lt(n, Shape.logical_numel(Shape.dimensions(shape))), U32.to_nat(index), cursor, converted),
    Equal.trans(Bool, Nat.is_lt(cursor, Shape.logical_numel(Shape.dimensions(shape))), Nat.is_lt(cursor, Shape.elements(shape)), True{},
      Equal.cong(Nat, Bool, n => Nat.is_lt(cursor, n), Shape.logical_numel(Shape.dimensions(shape)), Shape.elements(shape),
        Equal.sym(Nat, Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)), Shape.elements_exact(shape))),
      Equal.sym(Bool, True{}, Nat.is_lt(cursor, Shape.elements(shape)), Nats.lt_true(cursor, Shape.elements(shape), below))))

type FillSuccess<-shape: Shape.Shape, -result: Result<&2, &1, Core.Error, Core.Tensor<shape>>> is Data:
  FillSuccess{-successful: {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, result) == True{} : Bool}}

def fill_step_success(+shape: Shape.Shape, +rest: List<&2, F32>, +cursor: Nat,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>,
  successful: {TransformTotality.write_succeeded(shape, pair) == True{} : Bool},
  -rec: @tensor: Core.Tensor<shape> -> FillSuccess<shape, Core.fill_values(rest, shape, tensor, 1n+cursor)>) ->
  FillSuccess<shape, Core.fill_values_step(shape, tensor => Core.fill_values(rest, shape, tensor, 1n+cursor), pair)>:
  (tensor, result) = pair
  match result:
    case Fail{error}:
      FillSuccess{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{unit}:
      FillSuccess{{\{FillSuccess: successful => successful} : FillSuccess<shape, Core.fill_values(rest, shape, tensor, 1n+cursor)> ->
        {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, Core.fill_values(rest, shape, tensor, 1n+cursor)) == True{} : Bool}}(rec(tensor))}

def fill_decided_success(+shape: Shape.Shape, +value: F32, +rest: List<&2, F32>, +cursor: Nat, +index: U32,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>, -original: Core.Tensor<shape>,
  converted: {U32.to_nat(index) == cursor : Nat},
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  -rec: @tensor: Core.Tensor<shape> -> FillSuccess<shape, Core.fill_values(rest, shape, tensor, 1n+cursor)>) ->
  FillSuccess<shape, Core.fill_values_decided(shape, cursor, index, value, original,
    tensor => Core.fill_values(rest, shape, tensor, 1n+cursor), decision)>:
  match decision:
    case Shape.BFalse{equal}:
      FillSuccess{Empty.absurd({False{} == True{} : Bool}, Nats.contradiction(Nat.is_eq(U32.to_nat(index), cursor),
        AllocationProof.natural_equal_true(U32.to_nat(index), cursor, converted), equal))}
    case Shape.BTrue{equal}:
      FillSuccess{{\{FillSuccess: successful => successful} :
        FillSuccess<shape, Core.fill_values_step(shape, tensor => Core.fill_values(rest, shape, tensor, 1n+cursor),
          Core.write_flat(shape, original, index, value))> ->
        {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>,
          Core.fill_values_step(shape, tensor => Core.fill_values(rest, shape, tensor, 1n+cursor),
            Core.write_flat(shape, original, index, value))) == True{} : Bool}}(
              fill_step_success(shape, rest, cursor, Core.write_flat(shape, original, index, value),
                {\{TransformTotality.WriteSuccess: successful => successful} : TransformTotality.WriteSuccess<shape, original, index, value> ->
                  {TransformTotality.write_succeeded(shape, Core.write_flat(shape, original, index, value)) == True{} : Bool}}(
                    TransformLaws.tensor_write_valid_success(shape, original, index, value, inside)), rec))}

def fill_success(+values: List<&2, F32>, +shape: Shape.Shape, -original: Core.Tensor<shape>, +cursor: Nat,
  +room: Shape.LE(Nat.add(cursor, List.length(&2, F32, values)), Shape.elements(shape))) ->
  FillSuccess<shape, Core.fill_values(values, shape, original, cursor)>:
  match values:
    case Nil{}:
      FillSuccess{{==}}
    case +value <> +rest:
      fill_decided_success(shape, value, rest, cursor, U32.from_nat(cursor),
        Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)), original,
        cursor_exact(shape, cursor, Nats.le_add_left(cursor, 1n+List.length(&2, F32, rest), Shape.elements(shape), room)),
        cursor_inside(shape, cursor, U32.from_nat(cursor),
          cursor_exact(shape, cursor, Nats.le_add_left(cursor, 1n+List.length(&2, F32, rest), Shape.elements(shape), room)),
          cursor_below(cursor, List.length(&2, F32, rest), Shape.elements(shape), room)),
        tensor => fill_success(rest, shape, tensor, 1n+cursor,
          Nats.le_rewrite(Nat.add(1n+cursor, List.length(&2, F32, rest)), Nat.add(cursor, 1n+List.length(&2, F32, rest)),
            Shape.elements(shape), add_successor(cursor, List.length(&2, F32, rest)), room)))

def allocated_fill_success(+shape: Shape.Shape, +values: List<&2, F32>,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>, certificate: Allocation.TensorDone<shape, result>,
  counted: {List.length(&2, F32, values) == Shape.elements(shape) : Nat}) ->
  FillSuccess<shape, Core.from_values_new(shape, values, result)>:
  match result:
    case Fail{error}:
      FillSuccess{Empty.absurd({False{} == True{} : Bool},
        {\{Allocation.TensorDone: tensor => same => Arith.clash(Equal.sym(Bool, True{}, False{},
          Equal.cong(Result<&2, &1, Core.Error, Core.Tensor<shape>>, Bool,
            Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>), Done{tensor}, Fail{error}, same)))} :
          Allocation.TensorDone<shape, Fail{error}> -> Empty}(certificate))}
    case Done{tensor}:
      fill_success(values, shape, tensor, 0n,
        %Equal.sym(Nat, List.length(&2, F32, values), Shape.elements(shape), counted) : Shape.LE(_, Shape.elements(shape))
        Shape.LE.from(Shape.elements(shape), Shape.elements(shape),
          Equal.sym(Bool, Nat.is_le(Shape.elements(shape), Shape.elements(shape)), True{}, WordConversion.le_reflexive(Shape.elements(shape)))))

def counted_from_values_success(+shape: Shape.Shape, +values: List<&2, F32>,
  decision: Shape.BoolCase<Nat.is_eq(List.length(&2, F32, values), Shape.elements(shape))>,
  counted: {List.length(&2, F32, values) == Shape.elements(shape) : Nat}) ->
  FillSuccess<shape, Core.from_values_counted(shape, values, decision)>:
  match decision:
    case Shape.BFalse{equal}:
      FillSuccess{Empty.absurd({False{} == True{} : Bool}, Nats.contradiction(Nat.is_eq(List.length(&2, F32, values), Shape.elements(shape)),
        AllocationProof.natural_equal_true(List.length(&2, F32, values), Shape.elements(shape), counted), equal))}
    case Shape.BTrue{equal}:
      allocated_fill_success(shape, values, Core.new(shape, 0.0), AllocationProof.core_new_done(shape, 0.0), counted)

def completed_tensor(+shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>,
  -successful: {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, result) == True{} : Bool}) -> Totality.TensorDone<shape, result>:
  match result:
    case Fail{error}:
      Totality.TensorDone{Empty.absurd(Allocation.TensorDone<shape, Fail{error}>, Arith.clash(successful)), successful, Unit{}}
    case Done{tensor}:
      Totality.TensorDone{Allocation.TensorDone{tensor, {==}}, {==}, index => inside =>
        TransformProof.read_success_status(shape, tensor, index, TransformProof.read_success(shape, tensor, index, inside))}

def from_values_complete(+shape: Shape.Shape, +values: List<&2, F32>,
  counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Totality.TensorDone<shape, Core.from_values(shape, values)>:
  completed_tensor(shape, Core.from_values(shape, values),
    {\{FillSuccess: successful => successful} : FillSuccess<shape, Core.from_values(shape, values)> ->
      {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, Core.from_values(shape, values)) == True{} : Bool}}(
        counted_from_values_success(shape, values, Shape.bool_case(Nat.is_eq(List.length(&2, F32, values), Shape.elements(shape))),
          Equal.trans(Nat, List.length(&2, F32, values), Shape.logical_numel(Shape.dimensions(shape)), Shape.elements(shape), counted,
            Equal.sym(Nat, Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)), Shape.elements_exact(shape))))))

def Laws.public_from_values_complete(shape, values, counted):
  from_values_complete(shape, values, counted)

def construction_values_complete(+dimensions: List<&2, U32>, +shape: Shape.Shape,
  +preserved: {Shape.dimensions(shape) == dimensions : List<&2, U32>},
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>, certificate: Totality.TensorDone<shape, result>) ->
  Totality.ConstructionDone<dimensions, Core.construct_values(shape, result)>:
  match result:
    case Fail{error}:
      Totality.ConstructionDone{
        Empty.absurd(Totality.ConstructionWitness<dimensions, Fail{error}>,
          {\{Totality.TensorDone: witness => successful => readable => Arith.clash(successful)} :
            Totality.TensorDone<shape, Fail{error}> -> Empty}(certificate)),
        {\{Totality.TensorDone: witness => successful => readable => successful} :
          Totality.TensorDone<shape, Fail{error}> -> {False{} == True{} : Bool}}(certificate), Unit{}}
    case Done{tensor}:
      Totality.ConstructionDone{Totality.ConstructionWitness{shape, tensor, {==}, preserved}, {==}, index => inside =>
        TransformProof.read_success_status(shape, tensor, index, TransformProof.read_success(shape, tensor, index,
          %Equal.sym(List<&2, U32>, Shape.dimensions(shape), dimensions, preserved) :
            {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(_)) == True{} : Bool}
          inside))}

def construction_shape_complete(+dimensions: List<&2, U32>, +values: List<&2, F32>, result: Shape.ConstructResult<dimensions>,
  built: {ShapeLaws.built(dimensions, result) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
  +counted: {List.length(&2, F32, values) == Shape.logical_numel(dimensions) : Nat}) ->
  Totality.ConstructionDone<dimensions, Core.construct_shape(dimensions, values, result)>:
  match result:
    case Shape.SError{error}:
      Totality.ConstructionDone{
        Empty.absurd(Totality.ConstructionWitness<dimensions, Fail{Core.ShapeError{error}}>,
          Arith.clash(Equal.cong(Maybe<&2, List<&2, U32>>, Bool, Maybe.is_some(&2, List<&2, U32>), None{}, Some{dimensions}, built))),
        Equal.cong(Maybe<&2, List<&2, U32>>, Bool, Maybe.is_some(&2, List<&2, U32>), None{}, Some{dimensions}, built), Unit{}}
    case Shape.SBuilt{+shape, +preserved}:
      construction_values_complete(dimensions, shape, preserved, Core.from_values(shape, values),
        from_values_complete(shape, values,
          %Equal.sym(List<&2, U32>, Shape.dimensions(shape), dimensions, preserved) :
            {List.length(&2, F32, values) == Shape.logical_numel(_) : Nat}
          counted))

def Laws.public_construction_complete(dimensions, values, rank_bound, capacity_bound, counted):
  construction_shape_complete(dimensions, values, Shape.construct(dimensions),
    ShapeLaws.construction_complete(dimensions, rank_bound, capacity_bound), counted)
