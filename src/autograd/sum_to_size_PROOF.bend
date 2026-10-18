import Base
import ../shape/core.bend as Shape
import ../storage/arith.bend as Arith
import ../tensor/core.bend as Core
import ../tensor/layout.bend as Layout
import ../tensor/elementwise_totality.bend as Totality
import ../tensor/construction_LAWS.bend as ConstructionLaws
import ../tensor/construction_PROOF.bend as ConstructionProof
import ../tensor/transform_PROOF.bend as TransformProof
import ./reduce.bend as Reduce
import ./reduce_LAWS.bend as ReduceLaws
import ./reduce_PROOF.bend as ReduceProof
import ./sum_to_size.bend as SumToSize
import ./sum_to_size_LAWS.bend as Laws

type Holds<-P: Type> is Data:
  Holds{-proof: P}

def FirstKept(-shape: Shape.Shape, -original: Core.Tensor<shape>, pair: Core.Tensor<shape> & Core.Tensor<shape>) -> Type:
  (@+index: U32 -> {ConstructionLaws.clone_first_checked(shape, pair, index) == Core.checked_observe(shape, original, index) : Maybe<&2, F32>}) &
    (@+index: U32 -> {ConstructionLaws.clone_first_model(shape, pair, index) == Core.observe(shape, original, index) : Maybe<&2, F32>})

def SecondKept(-shape: Shape.Shape, -original: Core.Tensor<shape>, pair: Core.Tensor<shape> & Core.Tensor<shape>) -> Type:
  (@+index: U32 -> {ConstructionLaws.clone_second_checked(shape, pair, index) == Core.checked_observe(shape, original, index) : Maybe<&2, F32>}) &
    (@+index: U32 -> {ConstructionLaws.clone_second_model(shape, pair, index) == Core.observe(shape, original, index) : Maybe<&2, F32>})

def first_kept(+shape: Shape.Shape, original: Core.Tensor<shape>) -> Holds<FirstKept(shape, original, Core.clone(shape, original))>:
  {\{ConstructionLaws.CloneMeaning: first_checked => second_checked => first_model => second_model =>
    Holds{(first_checked, first_model)}} :
    ConstructionLaws.CloneMeaning<shape, original> -> Holds<FirstKept(shape, original, Core.clone(shape, original))>}(
    ConstructionLaws.public_clone_elements(shape, original))

def second_kept(+shape: Shape.Shape, original: Core.Tensor<shape>) -> Holds<SecondKept(shape, original, Core.clone(shape, original))>:
  {\{ConstructionLaws.CloneMeaning: first_checked => second_checked => first_model => second_model =>
    Holds{(second_checked, second_model)}} :
    ConstructionLaws.CloneMeaning<shape, original> -> Holds<SecondKept(shape, original, Core.clone(shape, original))>}(
    ConstructionLaws.public_clone_elements(shape, original))

def first_property(+property: Totality.Property, -shape: Shape.Shape, -original: Core.Tensor<shape>, -first: Core.Tensor<shape>,
  checked: @+index: U32 -> {Core.checked_observe(shape, first, index) == Core.checked_observe(shape, original, index) : Maybe<&2, F32>},
  model: @+index: U32 -> {Core.observe(shape, first, index) == Core.observe(shape, original, index) : Maybe<&2, F32>}) ->
  {Totality.inspect(property, Core.Packed{shape, first}) == Totality.inspect(property, Core.Packed{shape, original}) : Totality.Inspection}:
  match property:
    case Totality.Dimensions{}:
      {==}
    case Totality.Checked{+index}:
      Equal.cong(Maybe<&2, F32>, Totality.Inspection, value => Totality.Value{value},
        Core.checked_observe(shape, first, index), Core.checked_observe(shape, original, index), checked(index))
    case Totality.Raw{+index}:
      Equal.cong(Maybe<&2, F32>, Totality.Inspection, value => Totality.Value{value},
        Core.observe(shape, first, index), Core.observe(shape, original, index), model(index))

def cloned_retains(+shape: Shape.Shape, -original: Core.Tensor<shape>, pair: Core.Tensor<shape> & Core.Tensor<shape>,
  -kept: FirstKept(shape, original, pair)) ->
  ReduceLaws.Retained<Core.Packed{shape, original}, SumToSize.cloned(shape, pair)>:
  (first, second) = pair
  ReduceLaws.Retained{property => first_property(property, shape, original, first,
    Pair.fst(@+index: U32 -> {Core.checked_observe(shape, first, index) == Core.checked_observe(shape, original, index) : Maybe<&2, F32>}, @+index: U32 -> {Core.observe(shape, first, index) == Core.observe(shape, original, index) : Maybe<&2, F32>}, kept), Pair.snd(@+index: U32 -> {Core.checked_observe(shape, first, index) == Core.checked_observe(shape, original, index) : Maybe<&2, F32>}, @+index: U32 -> {Core.observe(shape, first, index) == Core.observe(shape, original, index) : Maybe<&2, F32>}, kept))}

def decided_retains(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape, same: Bool) ->
  ReduceLaws.Retained<Core.Packed{shape, tensor}, SumToSize.decided(shape, tensor, target, same)>:
  match same:
    case False{}:
      ReduceLaws.reduce_retains_input(Core.Packed{shape, tensor}, target)
    case True{}:
      cloned_retains(shape, tensor, Core.clone(shape, tensor),
        {\{Holds: proof => proof} : Holds<FirstKept(shape, tensor, Core.clone(shape, tensor))> -> FirstKept(shape, tensor, Core.clone(shape, tensor))}(
          first_kept(shape, tensor)))

def Laws.sum_to_size_retains_input(input, target):
  Core.Packed{+shape, tensor} = input
  decided_retains(shape, tensor, target, Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)))

def cloned_identity(+shape: Shape.Shape, -original: Core.Tensor<shape>, +target: Shape.Shape, pair: Core.Tensor<shape> & Core.Tensor<shape>,
  -same: {Shape.dimensions(shape) == Shape.dimensions(target) : List<&2, U32>},
  -kept: SecondKept(shape, original, pair)) ->
  Holds<Laws.Identity(Core.Packed{shape, original}, target, SumToSize.cloned(shape, pair))>:
  (first, second) = pair
  Holds{(same, index => (Pair.fst(@+index: U32 -> {Core.checked_observe(shape, second, index) == Core.checked_observe(shape, original, index) : Maybe<&2, F32>}, @+index: U32 -> {Core.observe(shape, second, index) == Core.observe(shape, original, index) : Maybe<&2, F32>}, kept)(index), Pair.snd(@+index: U32 -> {Core.checked_observe(shape, second, index) == Core.checked_observe(shape, original, index) : Maybe<&2, F32>}, @+index: U32 -> {Core.observe(shape, second, index) == Core.observe(shape, original, index) : Maybe<&2, F32>}, kept)(index)))}

def decided_identity(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape, flag: Bool,
  -decided: {flag == Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)) : Bool},
  -same: {Shape.dimensions(shape) == Shape.dimensions(target) : List<&2, U32>}) ->
  Holds<Laws.Identity(Core.Packed{shape, tensor}, target, SumToSize.decided(shape, tensor, target, flag))>:
  match flag:
    case False{}:
      Holds{Empty.absurd(Laws.Identity(Core.Packed{shape, tensor}, target, SumToSize.decided(shape, tensor, target, False{})),
        Arith.clash(Equal.trans(Bool, False{}, Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)), True{}, decided,
          %same : {Layout.words_same(Shape.dimensions(shape), _) == True{} : Bool}
          TransformProof.words_same_refl(Shape.dimensions(shape)))))}
    case True{}:
      cloned_identity(shape, tensor, target, Core.clone(shape, tensor), same,
        {\{Holds: proof => proof} : Holds<SecondKept(shape, tensor, Core.clone(shape, tensor))> -> SecondKept(shape, tensor, Core.clone(shape, tensor))}(
          second_kept(shape, tensor)))

def Laws.sum_to_size_identity(input, target, same):
  Core.Packed{+shape, tensor} = input
  Laws.IdentityModel{
    {\{Holds: proof => proof} :
      Holds<Laws.Identity(Core.Packed{shape, tensor}, target,
        SumToSize.decided(shape, tensor, target, Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target))))> ->
      Laws.Identity(Core.Packed{shape, tensor}, target, SumToSize.sum_to_size(Core.Packed{shape, tensor}, target))}(
      decided_identity(shape, tensor, target, Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)), {==}, same))}

def decided_reduces(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape, flag: Bool,
  decided: {flag == Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)) : Bool},
  differ: {Shape.dimensions(shape) != Shape.dimensions(target) : List<&2, U32>}) ->
  {SumToSize.decided(shape, tensor, target, flag) == Reduce.reduce_to(Core.Packed{shape, tensor}, target) :
    Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>}:
  match flag:
    case False{}:
      {==}
    case True{}:
      Empty.absurd({SumToSize.decided(shape, tensor, target, True{}) == Reduce.reduce_to(Core.Packed{shape, tensor}, target) :
        Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>},
        differ(TransformProof.words_equal(Shape.dimensions(shape), Shape.dimensions(target), decided)))

def Laws.sum_to_size_reduces(input, target, differ):
  Core.Packed{+shape, tensor} = input
  decided_reduces(shape, tensor, target, Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)), {==}, differ)
