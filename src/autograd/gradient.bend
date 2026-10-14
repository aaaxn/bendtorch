import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../tensor/core.bend as Core
import ../tensor/layout.bend as Layout

# Gradient slots for reverse differentiation. A slot starts Unset. The first
# contribution whose dimensions agree with the slot's shape moves in unchanged;
# every later one is added in place, position by position in ascending flat
# order: total[i] = total[i] + contribution[i], one F32 add per position, never
# an overwrite. A contribution of any other shape is refused, and the slot and
# the contribution come back unchanged. Like PyTorch's .grad, grad reads an
# Unset slot as None; gradient instead materializes zeros of the slot's shape.
# reset returns a slot to Unset, like PyTorch's zero_grad(set_to_none=True).

type Error is Data:
  TensorError{error: Core.Error}
  ShapeMismatch{}

type Slot is Type:
  Unset{}
  Held{gradient: Core.Dynamic}

type Outcome is Type:
  Accumulated{slot: Slot}
  Refused{slot: Slot, contribution: Core.Dynamic, error: Error}

type State<-shape: Shape.Shape> is Type:
  State{contribution: Core.Dynamic, total: Core.Tensor<shape>}

type Step<-shape: Shape.Shape> is Type:
  Continue{state: State<shape>}
  Stop{contribution: Core.Dynamic, total: Core.Tensor<shape>, error: Core.Error}

def step_write(-shape: Shape.Shape, contribution: Core.Dynamic,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) -> Step<shape>:
  (total, result) = pair
  match result:
    case Fail{error}:
      Stop{contribution, total, error}
    case Done{unit}:
      Continue{State{contribution, total}}

def step_total(-shape: Shape.Shape, +index: U32, contribution: Core.Dynamic, value: F32,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) -> Step<shape>:
  (total, result) = pair
  match result:
    case Fail{error}:
      Stop{contribution, total, error}
    case Done{+accumulated}:
      step_write(shape, contribution, Core.write_flat(shape, total, index, F32.add(accumulated, value)))

def step_contribution(-shape: Shape.Shape, +index: U32, total: Core.Tensor<shape>,
  pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> Step<shape>:
  (contribution, result) = pair
  match result:
    case Fail{error}:
      Stop{contribution, total, error}
    case Done{value}:
      step_total(shape, index, contribution, value, Core.read_flat(shape, total, index))

def step(-shape: Shape.Shape, +index: U32, state: State<shape>) -> Step<shape>:
  State{contribution, total} = state
  step_contribution(shape, index, total, Core.read_dynamic(contribution, index))

def continue_step(+shape: Shape.Shape, rec: State<shape> -> Outcome, result: Step<shape>) -> Outcome:
  match result:
    case Stop{contribution, total, error}:
      Refused{Held{Core.Packed{shape, total}}, contribution, TensorError{error}}
    case Continue{state}:
      rec(state)

def conversion_error(+shape: Shape.Shape, state: State<shape>) -> Outcome:
  State{contribution, total} = state
  Refused{Held{Core.Packed{shape, total}}, contribution, TensorError{Core.StorageError{Storage.InvalidIndex{}}}}

def decided(+shape: Shape.Shape, +cursor: Nat, +index: U32, state: State<shape>, rec: State<shape> -> Outcome,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>) -> Outcome:
  match decision:
    case Shape.BFalse{equal}:
      conversion_error(shape, state)
    case Shape.BTrue{equal}:
      continue_step(shape, rec, step(shape, index, state))

def loop(fuel: Nat, +shape: Shape.Shape, +cursor: Nat, state: State<shape>) -> Outcome:
  match fuel:
    case 0n:
      State{contribution, total} = state
      Accumulated{Held{Core.Packed{shape, total}}}
    case 1n+remaining:
      +index = U32.from_nat(cursor)
      decided(shape, cursor, index, state, state => loop(remaining, shape, 1n+cursor, state),
        Shape.bool_case(Nat.is_eq(U32.to_nat(index), cursor)))

def held_checked(contribution: Core.Dynamic, +total_shape: Shape.Shape, total: Core.Tensor<total_shape>, same: Bool) -> Outcome:
  match same:
    case False{}:
      Refused{Held{Core.Packed{total_shape, total}}, contribution, ShapeMismatch{}}
    case True{}:
      loop(Shape.elements(total_shape), total_shape, 0n, State{contribution, total})

def held(+shape: Shape.Shape, contribution: Core.Dynamic, gradient: Core.Dynamic) -> Outcome:
  Core.Packed{+total_shape, total} = gradient
  held_checked(contribution, total_shape, total, Layout.words_same(Shape.dimensions(total_shape), Shape.dimensions(shape)))

def agreed(+shape: Shape.Shape, slot: Slot, contribution: Core.Dynamic) -> Outcome:
  match slot:
    case Unset{}:
      Accumulated{Held{contribution}}
    case Held{gradient}:
      held(shape, contribution, gradient)

def checked(+shape: Shape.Shape, slot: Slot, +contribution_shape: Shape.Shape, tensor: Core.Tensor<contribution_shape>, same: Bool) -> Outcome:
  match same:
    case False{}:
      Refused{slot, Core.Packed{contribution_shape, tensor}, ShapeMismatch{}}
    case True{}:
      agreed(shape, slot, Core.Packed{contribution_shape, tensor})

def accumulate(+shape: Shape.Shape, slot: Slot, contribution: Core.Dynamic) -> Outcome:
  Core.Packed{+contribution_shape, tensor} = contribution
  checked(shape, slot, contribution_shape, tensor,
    Layout.words_same(Shape.dimensions(contribution_shape), Shape.dimensions(shape)))

# The gradient a slot holds, or None when it never received one or was reset.
def grad(slot: Slot) -> Maybe<&1, Core.Dynamic>:
  match slot:
    case Unset{}:
      None{}
    case Held{gradient}:
      Some{gradient}

# Drops a held gradient: the next contribution moves in unchanged.
def reset(slot: Slot) -> Slot:
  Unset{}

# The gradient a slot holds; zeros of shape when it never received one.
def gradient(+shape: Shape.Shape, slot: Slot) -> Result<&2, &1, Core.Error, Core.Dynamic>:
  match slot:
    case Unset{}:
      Core.construct_values(shape, Core.new(shape, 0.0))
    case Held{gradient}:
      Done{gradient}

def error_code(error: Error) -> U32:
  match error:
    case TensorError{error}:
      Core.error_code(error)
    case ShapeMismatch{}:
      102
