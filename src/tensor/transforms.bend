import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./layout.bend as Layout
import ./transpose-spec.bend as Transpose
import ./permute-build.bend as Build
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ../storage/certificates.bend as Certificates

type Error is Data:
  LayoutError{error: Layout.Error}
  TensorError{error: Core.Error}

def elements_equal(+input: Shape.Shape, +target: Shape.Shape,
  equal: {True{} == Nat.is_eq(Shape.elements(input), Shape.elements(target)) : Bool}) ->
  {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}:
  Equal.trans(Nat, Shape.logical_numel(Shape.dimensions(input)), Shape.elements(input), Shape.logical_numel(Shape.dimensions(target)),
    Equal.sym(Nat, Shape.elements(input), Shape.logical_numel(Shape.dimensions(input)), Shape.elements_exact(input)),
    Equal.trans(Nat, Shape.elements(input), Shape.elements(target), Shape.logical_numel(Shape.dimensions(target)),
      Lifecycle.natural_equal(Shape.elements(input), Shape.elements(target), equal), Shape.elements_exact(target)))

def reshape_equal(-input: Shape.Shape, +target: Shape.Shape, tensor: Core.Tensor<input>,
  -same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}) -> Core.Dynamic:
  Core.TTensor{storage, valid} = tensor
  Core.pack(target, Core.TTensor{storage,
    Equal.trans(Nat, Lifecycle.logical_length(storage), Shape.logical_numel(Shape.dimensions(input)),
      Shape.logical_numel(Shape.dimensions(target)), valid, same)})

def reshape_decided(+input: Shape.Shape, +target: Shape.Shape, tensor: Core.Tensor<input>,
  decision: Shape.BoolCase<Nat.is_eq(Shape.elements(input), Shape.elements(target))>) ->
  Result<&2, &1, Error, Core.Dynamic>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{LayoutError{Layout.ElementCountMismatch{}}}
    case Shape.BTrue{equal}:
      Done{reshape_equal(input, target, tensor,
        elements_equal(input, target, equal))}

def reshape_built(+input: Shape.Shape, tensor: Core.Tensor<input>, -dimensions: List<&2, U32>,
  built: Shape.ConstructResult<dimensions>) -> Result<&2, &1, Error, Core.Dynamic>:
  match built:
    case Shape.SError{error}:
      Fail{LayoutError{Layout.ShapeError{error}}}
    case Shape.SBuilt{+target, preserved}:
      reshape_decided(input, target, tensor,
        Shape.bool_case(Nat.is_eq(Shape.elements(input), Shape.elements(target))))

def reshape(input: Shape.Shape, tensor: Core.Tensor<input>, +dimensions: List<&2, U32>) -> Result<&2, &1, Error, Core.Dynamic>:
  reshape_built(input, tensor, dimensions, Shape.construct(dimensions))

def reshape_dynamic(dynamic: Core.Dynamic, dimensions: List<&2, U32>) -> Result<&2, &1, Error, Core.Dynamic>:
  Core.Packed{input, tensor} = dynamic
  reshape(input, tensor, dimensions)

def copy_written(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>,
  rec: Core.Tensor<input> -> Core.Tensor<Layout.output(input, axes, plan)> ->
    Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>,
  source: Core.Tensor<input>, pair: Core.Tensor<Layout.output(input, axes, plan)> & Result<&2, &2, Core.Error, Unit>) ->
  Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>:
  (target, result) = pair
  match result:
    case Fail{error}:
      (source, Fail{TensorError{error}})
    case Done{unit}:
      rec(source)(target)

def copy_read(-input: Shape.Shape, -axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, i: U32,
  rec: Core.Tensor<input> -> Core.Tensor<Layout.output(input, axes, plan)> ->
    Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>,
  target: Core.Tensor<Layout.output(input, axes, plan)>, pair: Core.Tensor<input> & Result<&2, &2, Core.Error, F32>) ->
  Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>:
  (source, result) = pair
  match result:
    case Fail{error}:
      (source, Fail{TensorError{error}})
    case Done{value}:
      copy_written(input, axes, plan, rec, source,
        Core.write_flat(Layout.output(input, axes, plan), target, i, value))

def copy_mapped(-input: Shape.Shape, -axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +i: U32,
  rec: Core.Tensor<input> -> Core.Tensor<Layout.output(input, axes, plan)> ->
    Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>,
  source: Core.Tensor<input>, target: Core.Tensor<Layout.output(input, axes, plan)>,
  mapped: Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, i>>) ->
  Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>:
  match mapped:
    case Fail{error}:
      (source, Fail{LayoutError{error}})
    case Done{mapping}:
      copy_read(input, axes, plan, i, rec, target,
        Core.read_flat(input, source, Layout.source_index(input, axes, plan, i, mapping)))

def copy_step(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  source: Core.Tensor<input>, target: Core.Tensor<Layout.output(input, axes, plan)>, +i: U32) ->
  Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>:
  copy_mapped(input, axes, plan, i, source => target => (source, Done{target}), source, target,
    Layout.map_index(input, axes, plan, i))

def copy_next(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>,
  rec: Core.Tensor<input> -> Core.Tensor<Layout.output(input, axes, plan)> ->
    Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>,
  pair: Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>) ->
  Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>:
  (source, result) = pair
  match result:
    case Fail{error}:
      (source, Fail{error})
    case Done{target}:
      rec(source)(target)

type NaturalCopy<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -cursor: Nat> is Type:
  NaturalCopied{pair: Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>,
    -index: U32, -word_exact: {U32.to_nat(index) == cursor : Nat}, -candidate: {index == U32.from_nat(cursor) : U32}}
  NaturalError{source: Core.Tensor<input>}

type CursorExact<-cursor: Nat, -index: U32> is Data:
  CursorExact{-equal: {U32.to_nat(index) == cursor : Nat}}

def natural_result(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>,
  -cursor: Nat, -index: U32, -word_exact: {U32.to_nat(index) == cursor : Nat}, -candidate: {index == U32.from_nat(cursor) : U32},
  pair: Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>) ->
  NaturalCopy<input, axes, plan, cursor>:
  NaturalCopied{pair, index, word_exact, candidate}

def natural_decided(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +cursor: Nat,
  +index: U32, source: Core.Tensor<input>, target: Core.Tensor<Layout.output(input, axes, plan)>,
  -candidate: {index == U32.from_nat(cursor) : U32},
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>) -> NaturalCopy<input, axes, plan, cursor>:
  match decision:
    case Shape.BFalse{equal}:
      NaturalError{source}
    case Shape.BTrue{equal}:
      natural_result(input, axes, plan, cursor, index, Lifecycle.natural_equal(U32.to_nat(index), cursor, equal), candidate,
        copy_step(input, axes, plan, source, target, index))

def natural_step(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +cursor: Nat,
  source: Core.Tensor<input>, target: Core.Tensor<Layout.output(input, axes, plan)>) -> NaturalCopy<input, axes, plan, cursor>:
  +index = U32.from_nat(cursor)
  natural_decided(input, axes, plan, cursor, index, source, target, {==},
    Shape.bool_case(Nat.is_eq(U32.to_nat(index), cursor)))

def CursorValid(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -cursor: Nat,
  result: NaturalCopy<input, axes, plan, cursor>) -> Data:
  match result:
    case NaturalCopied{pair, index, exact, candidate}:
      CursorExact<cursor, index>
    case NaturalError{source}:
      Unit

def natural_next(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -cursor: Nat,
  rec: Core.Tensor<input> -> Core.Tensor<Layout.output(input, axes, plan)> ->
    Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>,
  result: NaturalCopy<input, axes, plan, cursor>) ->
  Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>:
  match result:
    case NaturalCopied{pair, index, exact, candidate}:
      copy_next(input, axes, plan, rec, pair)
    case NaturalError{source}:
      (source, Fail{LayoutError{Layout.MappingError{}}})

def copy_loop(fuel: Nat, +input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  source: Core.Tensor<input>, target: Core.Tensor<Layout.output(input, axes, plan)>, +cursor: Nat) ->
  Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>:
  match fuel:
    case 0n:
      (source, Done{target})
    case 1n+p:
      natural_next(input, axes, plan, cursor,
        source => target => copy_loop(p, input, axes, plan, source, target, 1n+cursor),
        natural_step(input, axes, plan, cursor, source, target))

def copy_pack(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  pair: Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>) ->
  Core.Tensor<input> & Result<&2, &1, Error, Core.Dynamic>:
  (source, result) = pair
  match result:
    case Fail{error}:
      (source, Fail{error})
    case Done{target}:
      (source, Done{Core.pack(Layout.output(input, axes, plan), target)})

# The sequential copy: one checked read and one checked write per output
# cell, in index order. The laws about the copy loop state what it does, and
# permute-build-proof.bend proves that `copy_allocated` meets the loop's
# meaning (`LoopMeaning`) on every source. `permute_copy` no longer runs it;
# the tensor gate's parallel probe runs it as the sequential reference.
def loop_allocated(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  source: Core.Tensor<input>, allocated: Result<&2, &1, Core.Error, Core.Tensor<Layout.output(input, axes, plan)>>) ->
  Core.Tensor<input> & Result<&2, &1, Error, Core.Dynamic>:
  match allocated:
    case Fail{error}:
      (source, Fail{TensorError{error}})
    case Done{target}:
      copy_pack(input, axes, plan,
        copy_loop(Shape.elements(Layout.output(input, axes, plan)), input, axes, plan, source, target, 0n))

# The parallel copy from an owned input: the input comes back as the first
# half of its clone, and the build reads the second half.
def built_cloned(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat},
  +d: Nat, +n: Nat, -array: Array<F32>, pair: Array<F32> & Array<F32>, -same: {pair == Array.clone(F32, array) : Array<F32> & Array<F32>},
  -balanced: Model.Balanced(F32, d, array), -ok: {Storage.bounds(d, n) == True{} : Bool},
  -fits: {Lifecycle.logical_length(Storage.Owned{d, n, array, balanced, ok}) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>:
  (kept, copy) = pair
  (Core.TTensor{Storage.Owned{d, n, kept, Certificates.clone_first(F32, d, array, (kept, copy), same, balanced), ok}, fits},
    Done{Core.TTensor{Storage.Owned{depth, length, Build.build(depth, Build.Request{input, axes}, length, copy),
      Build.balanced(depth, Build.Request{input, axes}, length, copy), valid}, exact}})

def built_owned(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat},
  source: Core.Tensor<input>) -> Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>:
  Core.TTensor{storage, fits} = source
  match storage:
    case Storage.Empty{}:
      (Core.TTensor{Storage.Empty{}, fits}, Fail{TensorError{Core.StorageError{Storage.OutOfBounds{}}}})
    case Storage.Owned{+d, +n, array, balanced, ok}:
      built_cloned(input, axes, plan, depth, length, valid, exact, d, n, array, Array.clone(F32, array), {==}, balanced, ok, fits)

# Every output cell at once: `Build.build` with the permute's index map.
def built(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  source: Core.Tensor<input>, target: Core.Tensor<Layout.output(input, axes, plan)>) ->
  Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>:
  Core.TTensor{storage, exact} = target
  match storage:
    case Storage.Empty{}:
      (source, Done{Core.TTensor{Storage.Empty{}, exact}})
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      built_owned(input, axes, plan, depth, length, valid, exact, source)

def copy_allocated(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  source: Core.Tensor<input>, allocated: Result<&2, &1, Core.Error, Core.Tensor<Layout.output(input, axes, plan)>>) ->
  Core.Tensor<input> & Result<&2, &1, Error, Core.Dynamic>:
  match allocated:
    case Fail{error}:
      (source, Fail{TensorError{error}})
    case Done{target}:
      copy_pack(input, axes, plan, built(input, axes, plan, source, target))

def copy_planned(+input: Shape.Shape, +axes: List<&2, U32>, source: Core.Tensor<input>,
  planned: Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>) -> Core.Tensor<input> & Result<&2, &1, Error, Core.Dynamic>:
  match planned:
    case Fail{error}:
      (source, Fail{LayoutError{error}})
    case Done{+plan}:
      copy_allocated(input, axes, plan, source, Core.new(Layout.output(input, axes, plan), 0.0))

def permute_copy(+input: Shape.Shape, source: Core.Tensor<input>, +axes: List<&2, U32>) ->
  Core.Tensor<input> & Result<&2, &1, Error, Core.Dynamic>:
  copy_planned(input, axes, source, Layout.permutation(input, axes))

def transpose_decided(+input: Shape.Shape, source: Core.Tensor<input>, +a: U32, +b: U32,
  decision: Shape.BoolCase<Transpose.valid(input, a, b)>) ->
  Core.Tensor<input> & Result<&2, &1, Error, Core.Dynamic>:
  match decision:
    case Shape.BFalse{equal}:
      (source, Fail{LayoutError{Layout.AxisOutOfBounds{}}})
    case Shape.BTrue{equal}:
      permute_copy(input, source, Layout.transpose_axes(Shape.dimensions(input), a, b, 0))

def transpose_copy(+input: Shape.Shape, source: Core.Tensor<input>, +a: U32, +b: U32) ->
  Core.Tensor<input> & Result<&2, &1, Error, Core.Dynamic>:
  transpose_decided(input, source, a, b,
    Shape.bool_case(Transpose.valid(input, a, b)))

def dynamic_result(+input: Shape.Shape, pair: Core.Tensor<input> & Result<&2, &1, Error, Core.Dynamic>) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  (source, result) = pair
  (Core.pack(input, source), result)

def permute_dynamic(dynamic: Core.Dynamic, axes: List<&2, U32>) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  Core.Packed{+input, source} = dynamic
  dynamic_result(input, permute_copy(input, source, axes))

def transpose_dynamic(dynamic: Core.Dynamic, a: U32, b: U32) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  Core.Packed{+input, source} = dynamic
  dynamic_result(input, transpose_copy(input, source, a, b))

def dynamic_observe(dynamic: Core.Dynamic, index: U32) -> Maybe<&2, F32>:
  Core.Packed{shape, tensor} = dynamic
  Core.observe(shape, tensor, index)

def dynamic_dimensions(dynamic: Core.Dynamic) -> List<&2, U32>:
  Core.Packed{shape, tensor} = dynamic
  Shape.dimensions(shape)

def copy_observe(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>,
  pair: Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>,
  index: U32) -> Maybe<&2, F32>:
  (source, result) = pair
  match result:
    case Fail{error}:
      None{}
    case Done{target}:
      Core.observe(Layout.output(input, axes, plan), target, index)

type CopyMeaning<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -index: U32,
  -mapping: Layout.Mapping<input, axes, plan, index>, -source: Core.Tensor<input>,
  -target: Core.Tensor<Layout.output(input, axes, plan)>> is Data:
  CopyMeaning{-meaning: {copy_observe(input, axes, plan, copy_step(input, axes, plan, source, target, index), index) ==
    Core.checked_observe(input, source, Layout.source_index(input, axes, plan, index, mapping)) : Maybe<&2, F32>}}

def expected_mapped(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -index: U32,
  source: Core.Tensor<input>, result: Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{mapping}:
      Core.checked_observe(input, source, Layout.source_index(input, axes, plan, index, mapping))

def expected_output(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  source: Core.Tensor<input>, +index: U32) -> Maybe<&2, F32>:
  expected_mapped(input, axes, plan, index, source, Layout.map_index(input, axes, plan, index))

type Prefix<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>,
  -original: Core.Tensor<input>, -source: Core.Tensor<input>, -target: Core.Tensor<Layout.output(input, axes, plan)>, -cursor: Nat> is Data:
  Prefix{-source_kept: @+other: U32 ->
      {Core.checked_observe(input, source, other) == Core.checked_observe(input, original, other) : Maybe<&2, F32>},
    -words: @+other: Nat -> @-earlier: {Nat.is_lt(other, cursor) == True{} : Bool} ->
      {U32.to_nat(U32.from_nat(other)) == other : Nat},
    -checked: @+other: Nat -> @-earlier: {Nat.is_lt(other, cursor) == True{} : Bool} ->
      Core.ObservationChecked(Core.Packed{Layout.output(input, axes, plan), target}, U32.from_nat(other)),
    -written: @+other: Nat -> @-earlier: {Nat.is_lt(other, cursor) == True{} : Bool} ->
      {Core.checked_observe(Layout.output(input, axes, plan), target, U32.from_nat(other)) ==
        expected_output(input, axes, plan, original, U32.from_nat(other)) : Maybe<&2, F32>}}

type Advanced<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>,
  -original: Core.Tensor<input>, -before_source: Core.Tensor<input>, -after_source: Core.Tensor<input>,
  -before_target: Core.Tensor<Layout.output(input, axes, plan)>, -after_target: Core.Tensor<Layout.output(input, axes, plan)>, -cursor: Nat> is Data:
  Advanced{-word_exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
    -source_kept: @+other: U32 ->
      {Core.checked_observe(input, after_source, other) == Core.checked_observe(input, before_source, other) : Maybe<&2, F32>},
    -current: {Core.checked_observe(Layout.output(input, axes, plan), after_target, U32.from_nat(cursor)) ==
      expected_output(input, axes, plan, original, U32.from_nat(cursor)) : Maybe<&2, F32>},
    -current_checked: Core.ObservationChecked(Core.Packed{Layout.output(input, axes, plan), after_target}, U32.from_nat(cursor)),
    -past_checked: @+other: Nat -> @-earlier: {Nat.is_lt(other, cursor) == True{} : Bool} ->
      Core.ObservationChecked(Core.Packed{Layout.output(input, axes, plan), after_target}, U32.from_nat(other)),
    -frame: @+other: Nat -> @-earlier: {Nat.is_lt(other, cursor) == True{} : Bool} ->
      {Core.checked_observe(Layout.output(input, axes, plan), after_target, U32.from_nat(other)) ==
        Core.checked_observe(Layout.output(input, axes, plan), before_target, U32.from_nat(other)) : Maybe<&2, F32>}}

def SameSource(-input: Shape.Shape, -original: Core.Tensor<input>, -retained: Core.Tensor<input>) -> Type:
  @+other: U32 -> {Core.checked_observe(input, retained, other) == Core.checked_observe(input, original, other) : Maybe<&2, F32>}

def StepMeaning(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, -index: U32,
  -original_source: Core.Tensor<input>, -original_target: Core.Tensor<Layout.output(input, axes, plan)>,
  pair: Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>) -> Type:
  (source, result) = pair
  match result:
    case Fail{error}:
      Unit
    case Done{target}:
      SameSource(input, original_source, source) &
        (Core.ObservationChecked(Core.Packed{Layout.output(input, axes, plan), target}, index) &
          ({Core.checked_observe(Layout.output(input, axes, plan), target, index) ==
            expected_output(input, axes, plan, original_source, index) : Maybe<&2, F32>} &
            (@+other: U32 -> @checked: Core.ObservationChecked(Core.Packed{Layout.output(input, axes, plan), original_target}, other) ->
              @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
                Core.ObservationChecked(Core.Packed{Layout.output(input, axes, plan), target}, other) &
                  {Core.checked_observe(Layout.output(input, axes, plan), target, other) ==
                    Core.checked_observe(Layout.output(input, axes, plan), original_target, other) : Maybe<&2, F32>})))

type StepCertificate<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -index: U32,
  -original_source: Core.Tensor<input>, -original_target: Core.Tensor<Layout.output(input, axes, plan)>> is Type:
  StepCertificate{pair: Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>,
    exact: {pair == copy_step(input, axes, plan, original_source, original_target, index) :
      Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>},
    -meaning: StepMeaning(input, axes, plan, index, original_source, original_target, pair)}

def NaturalMeaning(+input: Shape.Shape, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>, -cursor: Nat,
  -original_source: Core.Tensor<input>, -original_target: Core.Tensor<Layout.output(input, axes, plan)>,
  result: NaturalCopy<input, axes, plan, cursor>) -> Type:
  match result:
    case NaturalError{source}:
      Unit
    case NaturalCopied{pair, index, exact, candidate}:
      StepMeaning(input, axes, plan, index, original_source, original_target, pair)

type NaturalCertificate<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -cursor: Nat,
  -original_source: Core.Tensor<input>, -original_target: Core.Tensor<Layout.output(input, axes, plan)>> is Type:
  NaturalCertificate{result: NaturalCopy<input, axes, plan, cursor>,
    exact: {result == natural_step(input, axes, plan, cursor, original_source, original_target) : NaturalCopy<input, axes, plan, cursor>},
    -meaning: NaturalMeaning(input, axes, plan, cursor, original_source, original_target, result)}

def LoopMeaning(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -original: Core.Tensor<input>, -limit: Nat,
  pair: Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>) -> Data:
  (source, result) = pair
  match result:
    case Fail{error}:
      Unit
    case Done{target}:
      Prefix<input, axes, plan, original, source, target, limit>

type LoopCertificate<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -original: Core.Tensor<input>,
  -fuel: Nat, -cursor: Nat, -source: Core.Tensor<input>, -target: Core.Tensor<Layout.output(input, axes, plan)>> is Type:
  LoopCertificate{pair: Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>,
    exact: {pair == copy_loop(fuel, input, axes, plan, source, target, cursor) :
      Core.Tensor<input> & Result<&2, &1, Error, Core.Tensor<Layout.output(input, axes, plan)>>},
    -meaning: LoopMeaning(input, axes, plan, original, Nat.add(fuel, cursor), pair)}

type WholeCopyModel<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -source: Core.Tensor<input>,
  -target: Core.Tensor<Layout.output(input, axes, plan)>> is Data:
  WholeCopyModel{-meaning: LoopMeaning(input, axes, plan, source,
    Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))),
    copy_loop(Shape.elements(Layout.output(input, axes, plan)), input, axes, plan, source, target, 0n))}

type PermuteOutput<-input: Shape.Shape, -axes: List<&2, U32>, -original: Core.Tensor<input>,
  -retained: Core.Tensor<input>, -output: Core.Dynamic> is Data:
  PermuteOutput{-plan: Layout.Plan<input, axes>,
    -dimensions: {dynamic_dimensions(output) == Shape.dimensions(Layout.output(input, axes, plan)) : List<&2, U32>},
    -kept: SameSource(input, original, retained),
    -words: @+other: Nat -> @-bound: {Nat.is_lt(other, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool} ->
      {U32.to_nat(U32.from_nat(other)) == other : Nat},
    -checked: @+other: Nat -> @-bound: {Nat.is_lt(other, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool} ->
      Core.ObservationChecked(output, U32.from_nat(other)),
    -observed: @+other: Nat -> @-bound: {Nat.is_lt(other, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool} ->
      {Core.checked_observe_dynamic(output, U32.from_nat(other)) == expected_output(input, axes, plan, original, U32.from_nat(other)) : Maybe<&2, F32>},
    -linear: @+other: Nat -> @-bound: {Nat.is_lt(other, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool} ->
      {dynamic_observe(output, U32.from_nat(other)) == expected_output(input, axes, plan, original, U32.from_nat(other)) : Maybe<&2, F32>}}

def PublicMeaning(-input: Shape.Shape, -axes: List<&2, U32>, -original: Core.Tensor<input>,
  pair: Core.Tensor<input> & Result<&2, &1, Error, Core.Dynamic>) -> Data:
  (retained, result) = pair
  match result:
    case Fail{error}:
      Unit
    case Done{output}:
      PermuteOutput<input, axes, original, retained, output>

type PublicModel<-input: Shape.Shape, -axes: List<&2, U32>, -original: Core.Tensor<input>,
  -pair: Core.Tensor<input> & Result<&2, &1, Error, Core.Dynamic>> is Data:
  PublicModel{-meaning: PublicMeaning(input, axes, original, pair)}

type ReshapeOutput<-input: Shape.Shape, -original: Core.Tensor<input>, -dimensions: List<&2, U32>, -output: Core.Dynamic> is Data:
  ReshapeOutput{-shape: Shape.Shape,
    -dimensions_exact: {dynamic_dimensions(output) == dimensions : List<&2, U32>},
    -shape_exact: {Shape.dimensions(shape) == dimensions : List<&2, U32>},
    -count: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(dimensions) : Nat},
    -linear: @+index: U32 -> {dynamic_observe(output, index) == Core.observe(input, original, index) : Maybe<&2, F32>}}

def ReshapeMeaning(-input: Shape.Shape, -original: Core.Tensor<input>, -dimensions: List<&2, U32>,
  result: Result<&2, &1, Error, Core.Dynamic>) -> Data:
  match result:
    case Fail{error}:
      Unit
    case Done{output}:
      ReshapeOutput<input, original, dimensions, output>

type ReshapeModel<-input: Shape.Shape, -original: Core.Tensor<input>, -dimensions: List<&2, U32>> is Data:
  ReshapeModel{-meaning: ReshapeMeaning(input, original, dimensions, reshape(input, original, dimensions))}

type RestoredOutput<-input: Shape.Shape, -original: Core.Tensor<input>, -output: Core.Dynamic> is Data:
  RestoredOutput{-dimensions: {dynamic_dimensions(output) == Shape.dimensions(input) : List<&2, U32>},
    -linear: @+index: U32 -> {dynamic_observe(output, index) == Core.observe(input, original, index) : Maybe<&2, F32>}}

def RestoreMeaning(-input: Shape.Shape, -original: Core.Tensor<input>, result: Result<&2, &1, Error, Core.Dynamic>) -> Data:
  match result:
    case Fail{error}:
      Unit
    case Done{output}:
      RestoredOutput<input, original, output>

def ReshapeInverseMeaning(+input: Shape.Shape, -original: Core.Tensor<input>,
  result: Result<&2, &1, Error, Core.Dynamic>) -> Data:
  match result:
    case Fail{error}:
      Unit
    case Done{output}:
      RestoreMeaning(input, original, reshape_dynamic(output, Shape.dimensions(input)))

type ReshapeInverseModel<-input: Shape.Shape, -original: Core.Tensor<input>, -dimensions: List<&2, U32>> is Data:
  ReshapeInverseModel{-meaning: ReshapeInverseMeaning(input, original, reshape(input, original, dimensions))}

type ReshapeSuccessModel<-input: Shape.Shape, -target: Shape.Shape, -original: Core.Tensor<input>> is Data:
  ReshapeSuccessModel{-success: {Result.is_done(&2, &1, Error, Core.Dynamic,
    reshape(input, original, Shape.dimensions(target))) == True{} : Bool}}

def reshape_inverse_success(+input: Shape.Shape, result: Result<&2, &1, Error, Core.Dynamic>) -> Bool:
  match result:
    case Fail{error}:
      False{}
    case Done{output}:
      Result.is_done(&2, &1, Error, Core.Dynamic, reshape_dynamic(output, Shape.dimensions(input)))

type ReshapeInverseSuccessModel<-input: Shape.Shape, -target: Shape.Shape, -original: Core.Tensor<input>> is Data:
  ReshapeInverseSuccessModel{-success: {reshape_inverse_success(input, reshape(input, original, Shape.dimensions(target))) == True{} : Bool}}

def error_code(error: Error) -> U32:
  match error:
    case LayoutError{Layout.ShapeError{error}}:
      Shape.error_code(error)
    case LayoutError{Layout.AxesLengthMismatch{}}:
      12
    case LayoutError{Layout.AxisOutOfBounds{}}:
      13
    case LayoutError{Layout.DuplicateAxis{}}:
      14
    case LayoutError{Layout.ElementCountMismatch{}}:
      15
    case LayoutError{Layout.MappingError{}}:
      16
    case TensorError{error}:
      Core.error_code(error)
