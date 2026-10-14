import Base
import ../shape/core.bend as Shape
import ./layout.bend as Layout
import ./core.bend as Core
import ./transforms.bend as Transforms
import ./axis-spec.bend as Axis
import ./transpose-spec.bend as Transpose

type PlanSuccess<-input: Shape.Shape, -axes: List<&2, U32>> is Data:
  PlanSuccess{-successful: {Result.is_done(&2, &2, Layout.Error, Layout.Plan<input, axes>,
    Layout.permutation(input, axes)) == True{} : Bool}}

def read_succeeded(-shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) -> Bool:
  (owner, result) = pair
  Result.is_done(&2, &2, Core.Error, F32, result)

type ReadSuccess<-shape: Shape.Shape, -original: Core.Tensor<shape>, -index: U32> is Data:
  ReadSuccess{-successful: {read_succeeded(shape, Core.read_flat(shape, original, index)) == True{} : Bool}}

def write_succeeded(-shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) -> Bool:
  (owner, result) = pair
  Result.is_done(&2, &2, Core.Error, Unit, result)

type WriteSuccess<-shape: Shape.Shape, -original: Core.Tensor<shape>, -index: U32, -value: F32> is Data:
  WriteSuccess{-successful: {write_succeeded(shape, Core.write_flat(shape, original, index, value)) == True{} : Bool}}

type WordsSuccess<-dimensions: List<&2, U32>, -values: List<&2, U32>> is Data:
  WordsSuccess{-successful: {Result.is_done(&2, &2, Shape.Error, Layout.WordsValid(dimensions, values),
    Layout.words_check(dimensions, values)) == True{} : Bool}}

type PermuteFailure<-input: Shape.Shape, -axes: List<&2, U32>, -original: Core.Tensor<input>, -error: Layout.Error> is Data:
  PermuteFailure{-exact: {Transforms.permute_copy(input, original, axes) == (original, Fail{Transforms.LayoutError{error}}) :
    Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}}

type TransposeFailure<-input: Shape.Shape, -original: Core.Tensor<input>, -a: U32, -b: U32> is Data:
  TransposeFailure{-exact: {Transforms.transpose_copy(input, original, a, b) == (original, Fail{Transforms.LayoutError{Layout.AxisOutOfBounds{}}}) :
    Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}}

type ReshapeFailure<-input: Shape.Shape, -target: Shape.Shape, -original: Core.Tensor<input>> is Data:
  ReshapeFailure{-exact: {Transforms.reshape(input, original, Shape.dimensions(target)) == Fail{Transforms.LayoutError{Layout.ElementCountMismatch{}}} :
    Result<&2, &1, Transforms.Error, Core.Dynamic>}}

type MappingSuccess<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -index: U32> is Data:
  MappingSuccess{-successful: {Result.is_done(&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>,
    Layout.map_index(input, axes, plan, index)) == True{} : Bool}}

def copy_succeeded(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>,
  pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>) -> Bool:
  (source, result) = pair
  Result.is_done(&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>, result)

type CopyStepSuccess<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>,
  -source: Core.Tensor<input>, -target: Core.Tensor<Layout.output(input, axes, plan)>, -index: U32> is Data:
  CopyStepSuccess{-successful: {copy_succeeded(input, axes, plan, Transforms.copy_step(input, axes, plan, source, target, index)) == True{} : Bool}}

type CopySuccess<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>,
  -pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Tensor<Layout.output(input, axes, plan)>>> is Data:
  CopySuccess{-successful: {copy_succeeded(input, axes, plan, pair) == True{} : Bool}}

def natural_succeeded(-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -cursor: Nat,
  result: Transforms.NaturalCopy<input, axes, plan, cursor>) -> Bool:
  match result:
    case Transforms.NaturalError{source}:
      False{}
    case Transforms.NaturalCopied{pair, index, exact, candidate}:
      copy_succeeded(input, axes, plan, pair)

type NaturalSuccess<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -cursor: Nat,
  -result: Transforms.NaturalCopy<input, axes, plan, cursor>> is Data:
  NaturalSuccess{-successful: {natural_succeeded(input, axes, plan, cursor, result) == True{} : Bool}}

def public_succeeded(-input: Shape.Shape, pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Bool:
  (source, result) = pair
  Result.is_done(&2, &1, Transforms.Error, Core.Dynamic, result)

type PublicSuccess<-input: Shape.Shape, -pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>> is Data:
  PublicSuccess{-successful: {public_succeeded(input, pair) == True{} : Bool}}

def inverse_axes(+input: Shape.Shape, axes: List<&2, U32>) -> List<&2, U32>:
  Shape.words(Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes)))

type TypedPermutation<-input: Shape.Shape, -axes: List<&2, U32>, -original: Core.Tensor<input>,
  -pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>> is Data:
  TypedPermutation{-plan: Layout.Plan<input, axes>, -retained: Core.Tensor<input>, -target: Core.Tensor<Layout.output(input, axes, plan)>,
    -exact: {(retained, Done{Core.Packed{Layout.output(input, axes, plan), target}}) == pair : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>},
    -prefix: Transforms.Prefix<input, axes, plan, original, retained, target, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))>}

type MappingDone<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -index: U32,
  -result: Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>> is Data:
  MappingDone{-mapping: Layout.Mapping<input, axes, plan, index>,
    -exact: {Done{mapping} == result : Result<&2, &2, Layout.Error, Layout.Mapping<input, axes, plan, index>>}}

type CopyObservation<-input: Shape.Shape, -axes: List<&2, U32>, -plan: Layout.Plan<input, axes>, -original: Core.Tensor<input>,
  -target: Core.Tensor<Layout.output(input, axes, plan)>, -index: U32> is Data:
  CopyObservation{-equal: {Core.observe(Layout.output(input, axes, plan), target, index) == Transforms.expected_output(input, axes, plan, original, index) : Maybe<&2, F32>}}

type RestoreObservation<-input: Shape.Shape, -output: Shape.Shape, -original: Core.Tensor<input>, -restored: Core.Tensor<output>, -index: U32> is Data:
  RestoreObservation{-equal: {Core.observe(output, restored, index) == Core.observe(input, original, index) : Maybe<&2, F32>}}

type PermutationRestoration<-input: Shape.Shape, -axes: List<&2, U32>, -original: Core.Tensor<input>> is Data:
  PermutationRestoration{-first: Layout.Plan<input, axes>, -retained: Core.Tensor<input>, -middle: Core.Tensor<Layout.output(input, axes, first)>,
    -first_exact: {(retained, Done{Core.Packed{Layout.output(input, axes, first), middle}}) == Transforms.permute_copy(input, original, axes) :
      Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>},
    -second: Layout.Plan<Layout.output(input, axes, first), inverse_axes(input, axes)>,
    -middle_retained: Core.Tensor<Layout.output(input, axes, first)>,
    -restored: Core.Tensor<Layout.output(Layout.output(input, axes, first), inverse_axes(input, axes), second)>,
    -second_exact: {(Core.Packed{Layout.output(input, axes, first), middle_retained}, Done{Core.Packed{Layout.output(Layout.output(input, axes, first), inverse_axes(input, axes), second), restored}}) ==
      Transforms.permute_dynamic(Core.Packed{Layout.output(input, axes, first), middle}, inverse_axes(input, axes)) : Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>},
    -dimensions: {Shape.dimensions(Layout.output(Layout.output(input, axes, first), inverse_axes(input, axes), second)) == Shape.dimensions(input) : List<&2, U32>},
    -linear: @+index: U32 -> @-inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool} ->
      {Core.observe(Layout.output(Layout.output(input, axes, first), inverse_axes(input, axes), second), restored, index) == Core.observe(input, original, index) : Maybe<&2, F32>}}

type PermutationInverse<-input: Shape.Shape, -axes: List<&2, U32>, -original: Core.Tensor<input>> is Data:
  PermutationInverse{-restoration: PermutationRestoration<input, axes, original>}

def transpose_reference(+input: Shape.Shape, +output: Shape.Shape, +a: U32, +b: U32, index: U32) -> Maybe<&2, U32>:
  Layout.reference_source_coordinates(Shape.dimensions(input),
    Axis.scatter(U32, Shape.rank_dimensions(Shape.dimensions(input)),
      Shape.naturals(Transpose.requested_axes(Shape.rank_dimensions(Shape.dimensions(input)), U32.to_nat(a), U32.to_nat(b))),
      Shape.words(Shape.digits(Shape.extents(output), U32.to_nat(index))), 0n))

def reference_observe(-input: Shape.Shape, source: Core.Tensor<input>, reference: Maybe<&2, U32>) -> Maybe<&2, F32>:
  match reference:
    case None{}:
      None{}
    case Some{index}:
      Core.observe(input, source, index)

type TransposeObservation<-input: Shape.Shape, -output: Shape.Shape, -source: Core.Tensor<input>, -target: Core.Tensor<output>, -a: U32, -b: U32, -index: U32> is Data:
  TransposeObservation{-equal: {Core.observe(output, target, index) == reference_observe(input, source, transpose_reference(input, output, a, b, index)) : Maybe<&2, F32>}}

type RequestedTranspose<-input: Shape.Shape, -original: Core.Tensor<input>, -a: U32, -b: U32> is Data:
  RequestedTranspose{-plan: Layout.Plan<input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)>,
    -retained: Core.Tensor<input>, -target: Core.Tensor<Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan)>,
    -exact: {(retained, Done{Core.Packed{Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), target}}) ==
      Transforms.transpose_copy(input, original, a, b) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>},
    -dimensions: {Shape.dimensions(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan)) ==
      Axis.order(U32, Shape.dimensions(input), Shape.naturals(Transpose.requested_axes(Shape.rank_dimensions(Shape.dimensions(input)), U32.to_nat(a), U32.to_nat(b)))) : List<&2, U32>},
    -prefix: Transforms.Prefix<input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, original, retained, target,
      Shape.logical_numel(Shape.dimensions(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan)))>,
    -mapping: @+index: U32 -> @mapping: Layout.Mapping<input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index> ->
      {Some{Layout.source_index(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan, index, mapping)} ==
        transpose_reference(input, Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), a, b, index) : Maybe<&2, U32>},
    -linear: @+index: U32 -> @-inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan)))) == True{} : Bool} ->
      {Core.observe(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), target, index) ==
        reference_observe(input, original, transpose_reference(input, Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), plan), a, b, index)) : Maybe<&2, F32>}}

type TransposeRestoration<-input: Shape.Shape, -original: Core.Tensor<input>, -a: U32, -b: U32> is Data:
  TransposeRestoration{-first: Layout.Plan<input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)>,
    -retained: Core.Tensor<input>, -middle: Core.Tensor<Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first)>,
    -first_exact: {(retained, Done{Core.Packed{Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), middle}}) ==
      Transforms.transpose_copy(input, original, a, b) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>},
    -second: Layout.Plan<Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), inverse_axes(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0))>,
    -middle_retained: Core.Tensor<Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first)>,
    -restored: Core.Tensor<Layout.output(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), inverse_axes(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)), second)>,
    -second_exact: {(Core.Packed{Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), middle_retained},
      Done{Core.Packed{Layout.output(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), inverse_axes(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)), second), restored}}) ==
        Transforms.transpose_dynamic(Core.Packed{Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), middle}, a, b) : Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>},
    -dimensions: {Shape.dimensions(Layout.output(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), inverse_axes(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)), second)) == Shape.dimensions(input) : List<&2, U32>},
    -linear: @+index: U32 -> @-inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool} ->
      {Core.observe(Layout.output(Layout.output(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0), first), inverse_axes(input, Layout.transpose_axes(Shape.dimensions(input), a, b, 0)), second), restored, index) == Core.observe(input, original, index) : Maybe<&2, F32>}}

type TransposeInverse<-input: Shape.Shape, -original: Core.Tensor<input>, -a: U32, -b: U32> is Data:
  TransposeInverse{-restoration: TransposeRestoration<input, original, a, b>}
