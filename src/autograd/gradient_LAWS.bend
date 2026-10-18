import Base
import ../shape/core.bend as Shape
import ../tensor/core.bend as Core
import ../tensor/elementwise_totality.bend as Totality
import ./gradient.bend as Gradient

def add_word(total: Maybe<&2, F32>, value: Maybe<&2, F32>) -> Maybe<&2, F32>:
  match total value:
    case None{} _:
      None{}
    case Some{sum} None{}:
      None{}
    case Some{sum} Some{word}:
      Some{F32.add(sum, word)}

# The value the specification assigns to one position after a contribution is
# accumulated into a slot: an Unset slot takes the contribution's word as it
# is, and a Held slot adds the contribution's word to its own with one F32 add,
# the held total first.
def combined(slot: Gradient.Slot, contribution: Core.Dynamic, +index: U32) -> Maybe<&2, F32>:
  match slot:
    case Gradient.Unset{}:
      Core.checked_observe_dynamic(contribution, index)
    case Gradient.Held{gradient}:
      add_word(Core.checked_observe_dynamic(gradient, index), Core.checked_observe_dynamic(contribution, index))

def held_dimensions(slot: Gradient.Slot) -> Maybe<&2, List<&2, U32>>:
  match slot:
    case Gradient.Unset{}:
      None{}
    case Gradient.Held{gradient}:
      Some{Totality.dimensions(gradient)}

def held_observe(slot: Gradient.Slot, +index: U32) -> Maybe<&2, F32>:
  match slot:
    case Gradient.Unset{}:
      None{}
    case Gradient.Held{gradient}:
      Core.checked_observe_dynamic(gradient, index)

def Inside(+shape: Shape.Shape, +index: U32) -> Type:
  {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}

def Mapping(-slot: Gradient.Slot, -contribution: Core.Dynamic, +shape: Shape.Shape, outcome: Gradient.Outcome) -> Type:
  match outcome:
    case Gradient.Refused{after, returned, error}:
      Unit
    case Gradient.Accumulated{after}:
      {held_dimensions(after) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>} &
        (@+index: U32 -> @inside: Inside(shape, index) ->
          {held_observe(after, index) == combined(slot, contribution, index) : Maybe<&2, F32>})

def SlotKept(before: Gradient.Slot, after: Gradient.Slot) -> Type:
  match before:
    case Gradient.Unset{}:
      {held_dimensions(after) == None{} : Maybe<&2, List<&2, U32>>}
    case Gradient.Held{gradient}:
      {held_dimensions(after) == Some{Totality.dimensions(gradient)} : Maybe<&2, List<&2, U32>>} &
        (@+index: U32 -> {held_observe(after, index) == Core.checked_observe_dynamic(gradient, index) : Maybe<&2, F32>})

def mismatched(error: Gradient.Error) -> Bool:
  match error:
    case Gradient.TensorError{error}:
      False{}
    case Gradient.ShapeMismatch{}:
      True{}

def RefusalKeeps(-slot: Gradient.Slot, -contribution: Core.Dynamic, outcome: Gradient.Outcome) -> Type:
  match outcome:
    case Gradient.Accumulated{after}:
      Unit
    case Gradient.Refused{after, returned, error}:
      {mismatched(error) == True{} : Bool} & (Totality.SameSource(contribution, returned) & SlotKept(slot, after))

def Fits(+shape: Shape.Shape, slot: Gradient.Slot) -> Type:
  match slot:
    case Gradient.Unset{}:
      Unit
    case Gradient.Held{gradient}:
      {Totality.dimensions(gradient) == Shape.dimensions(shape) : List<&2, U32>}

def accumulated(outcome: Gradient.Outcome) -> Bool:
  match outcome:
    case Gradient.Accumulated{after}:
      True{}
    case Gradient.Refused{after, returned, error}:
      False{}

def Zeros(+shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Dynamic>) -> Type:
  match result:
    case Fail{error}:
      Empty
    case Done{gradient}:
      {Totality.dimensions(gradient) == Shape.dimensions(shape) : List<&2, U32>} &
        (@+index: U32 -> @inside: Inside(shape, index) ->
          {Core.checked_observe_dynamic(gradient, index) == Some{0.0} : Maybe<&2, F32>})

type MappingModel<-slot: Gradient.Slot, -contribution: Core.Dynamic, -shape: Shape.Shape, -outcome: Gradient.Outcome> is Data:
  MappingModel{-mapping: Mapping(slot, contribution, shape, outcome)}

type RefusalModel<-slot: Gradient.Slot, -contribution: Core.Dynamic, -outcome: Gradient.Outcome> is Data:
  RefusalModel{-keeps: RefusalKeeps(slot, contribution, outcome)}

type Success<-outcome: Gradient.Outcome> is Data:
  Success{-successful: {accumulated(outcome) == True{} : Bool}}

type ZerosModel<-shape: Shape.Shape, -result: Result<&2, &1, Core.Error, Core.Dynamic>> is Data:
  ZerosModel{-zeros: Zeros(shape, result)}

# A slot that never received a contribution reads as zeros of its shape.
law gradient_unset_zeros:
  for +shape: Shape.Shape
  ZerosModel<shape, Gradient.gradient(shape, Gradient.Unset{})>

# A slot that holds a gradient reads as that gradient.
law gradient_held:
  for +shape: Shape.Shape
  for held: Core.Dynamic
  {Gradient.gradient(shape, Gradient.Held{held}) == Done{held} : Result<&2, &1, Core.Error, Core.Dynamic>}

# On success the slot holds a gradient with the slot's dimensions. At every
# in-bounds position, the first contribution moves in unchanged and every
# later one is added to the held total with one F32 add, total first.
law accumulate_mapping:
  for +shape: Shape.Shape
  for slot: Gradient.Slot
  for contribution: Core.Dynamic
  MappingModel<slot, contribution, shape, Gradient.accumulate(shape, slot, contribution)>

# Every refusal is a shape mismatch, decided before any write: the
# contribution comes back unchanged, and so does the slot.
law accumulate_refusal_keeps:
  for +shape: Shape.Shape
  for slot: Gradient.Slot
  for contribution: Core.Dynamic
  RefusalModel<slot, contribution, Gradient.accumulate(shape, slot, contribution)>

# A contribution and a slot that both agree with the slot's shape always
# accumulate.
law accumulate_valid_success:
  for +shape: Shape.Shape
  for slot: Gradient.Slot
  for contribution: Core.Dynamic
  for fits: Fits(shape, slot)
  for agrees: {Totality.dimensions(contribution) == Shape.dimensions(shape) : List<&2, U32>}
  Success<Gradient.accumulate(shape, slot, contribution)>

# Like PyTorch's .grad, a slot that never received a contribution has no
# gradient: grad reads it as None, not as zeros.
law grad_unset_none:
  {Gradient.grad(Gradient.Unset{}) == None{} : Maybe<&1, Core.Dynamic>}

# A slot that holds a gradient reads as that gradient.
law grad_held_some:
  for held: Core.Dynamic
  {Gradient.grad(Gradient.Held{held}) == Some{held} : Maybe<&1, Core.Dynamic>}

# reset returns every slot to Unset, like zero_grad(set_to_none=True).
law reset_unset:
  for slot: Gradient.Slot
  {Gradient.reset(slot) == Gradient.Unset{} : Gradient.Slot}

# After a reset, a contribution with the slot's dimensions moves in unchanged,
# so a -0.0 word stays -0.0 instead of becoming +0.0 + -0.0.
law reset_then_accumulate_moves_in:
  for +shape: Shape.Shape
  for slot: Gradient.Slot
  for contribution: Core.Dynamic
  for agrees: {Totality.dimensions(contribution) == Shape.dimensions(shape) : List<&2, U32>}
  {Gradient.accumulate(shape, Gradient.reset(slot), contribution) == Gradient.Accumulated{Gradient.Held{contribution}} : Gradient.Outcome}

# A contribution whose dimensions differ from the slot's is refused with
# ShapeMismatch, and the slot and the contribution come back unchanged, even
# when the element counts agree ([3, 2] or [6] for a [2, 3] slot).
law accumulate_wrong_shape_refused:
  for +shape: Shape.Shape
  for slot: Gradient.Slot
  for contribution: Core.Dynamic
  for differ: {Totality.dimensions(contribution) != Shape.dimensions(shape) : List<&2, U32>}
  {Gradient.accumulate(shape, slot, contribution) == Gradient.Refused{slot, contribution, Gradient.ShapeMismatch{}} : Gradient.Outcome}
