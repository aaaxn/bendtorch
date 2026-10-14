import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../tensor/core.bend as Core
import ../tensor/elementwise.bend as Elementwise
import ../tensor/layout.bend as Layout

# Reduction of a broadcast contribution back to a target shape. The target
# must broadcast into the contribution's shape. The output starts as zeros of
# the target shape; the loop visits contribution positions in ascending flat
# order and adds each value into the target position of its right-aligned
# projection: output[t] = output[t] + value, one ordered F32 add per position.

type Error is Data:
  TensorError{error: Core.Error}
  NotReducible{}

type State<-target: Shape.Shape> is Type:
  State{input: Core.Dynamic, output: Core.Tensor<target>}

type Step<-target: Shape.Shape> is Type:
  Continue{state: State<target>}
  Stop{input: Core.Dynamic, error: Core.Error}

def step_write(-target: Shape.Shape, input: Core.Dynamic,
  pair: Core.Tensor<target> & Result<&2, &2, Core.Error, Unit>) -> Step<target>:
  (output, result) = pair
  match result:
    case Fail{error}:
      Stop{input, error}
    case Done{unit}:
      Continue{State{input, output}}

def step_output(-target: Shape.Shape, +target_index: U32, input: Core.Dynamic, value: F32,
  pair: Core.Tensor<target> & Result<&2, &2, Core.Error, F32>) -> Step<target>:
  (output, result) = pair
  match result:
    case Fail{error}:
      Stop{input, error}
    case Done{+accumulated}:
      step_write(target, input, Core.write_flat(target, output, target_index, F32.add(accumulated, value)))

def step_input(-target: Shape.Shape, +target_index: U32, output: Core.Tensor<target>,
  pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> Step<target>:
  (input, result) = pair
  match result:
    case Fail{error}:
      Stop{input, error}
    case Done{value}:
      step_output(target, target_index, input, value, Core.read_flat(target, output, target_index))

def step_indices(-target: Shape.Shape, state: State<target>, result: Result<&2, &1, Core.Error, U32 & U32>) -> Step<target>:
  State{input, output} = state
  match result:
    case Fail{error}:
      Stop{input, error}
    case Done{indices}:
      (+target_index, source_index) = indices
      step_input(target, target_index, output, Core.read_dynamic(input, source_index))

def step(+plan: Shape.BroadcastPlan, -target: Shape.Shape, +index: U32, state: State<target>) -> Step<target>:
  step_indices(target, state, Elementwise.input_indices(plan, index))

def continue_step(-target: Shape.Shape,
  rec: State<target> -> Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>>,
  result: Step<target>) -> Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>>:
  match result:
    case Stop{input, error}:
      (input, Fail{error})
    case Continue{state}:
      rec(state)

def conversion_error(-target: Shape.Shape, state: State<target>) -> Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>>:
  State{input, output} = state
  (input, Fail{Core.StorageError{Storage.InvalidIndex{}}})

def decided(+plan: Shape.BroadcastPlan, -target: Shape.Shape, +cursor: Nat, +index: U32, state: State<target>,
  rec: State<target> -> Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>>,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>) -> Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>>:
  match decision:
    case Shape.BFalse{equal}:
      conversion_error(target, state)
    case Shape.BTrue{equal}:
      continue_step(target, rec, step(plan, target, index, state))

def loop(fuel: Nat, +plan: Shape.BroadcastPlan, -target: Shape.Shape, +cursor: Nat, state: State<target>) ->
  Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>>:
  match fuel:
    case 0n:
      State{input, output} = state
      (input, Done{output})
    case 1n+remaining:
      +index = U32.from_nat(cursor)
      decided(plan, target, cursor, index, state, state => loop(remaining, plan, target, 1n+cursor, state),
        Shape.bool_case(Nat.is_eq(U32.to_nat(index), cursor)))

def finish(+target: Shape.Shape, pair: Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>>) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  (input, result) = pair
  match result:
    case Fail{error}:
      (input, Fail{TensorError{error}})
    case Done{output}:
      (input, Done{Core.Packed{target, output}})

def allocated(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  result: Result<&2, &1, Core.Error, Core.Tensor<target>>) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      (input, Fail{TensorError{error}})
    case Done{output}:
      finish(target, loop(Shape.elements(Shape.output_shape(plan)), plan, target, 0n, State{input, output}))

def fitted(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan, same: Bool) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match same:
    case False{}:
      (input, Fail{NotReducible{}})
    case True{}:
      allocated(input, target, plan, Core.new(target, 0.0))

def planned(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      (Core.Packed{shape, tensor}, Fail{TensorError{Core.ShapeError{error}}})
    case Done{+plan}:
      fitted(Core.Packed{shape, tensor}, target, plan,
        Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape)))

# The contribution owner comes back unchanged next to the result. A target
# that does not broadcast with the contribution fails with the S1 broadcast
# error; a target whose broadcast with the contribution is larger than the
# contribution (for example [2, 3] for a [3] contribution) fails with
# NotReducible.
def reduce_to(input: Core.Dynamic, +target: Shape.Shape) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  Core.Packed{+shape, tensor} = input
  planned(shape, tensor, target, Shape.broadcast_checked(target, shape))

def reduce_to_dimensions_result(input: Core.Dynamic, -dimensions: List<&2, U32>, result: Shape.ConstructResult<dimensions>) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Shape.SError{error}:
      (input, Fail{TensorError{Core.ShapeError{error}}})
    case Shape.SBuilt{+target, preserved}:
      reduce_to(input, target)

def reduce_to_dimensions(input: Core.Dynamic, +dimensions: List<&2, U32>) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  reduce_to_dimensions_result(input, dimensions, Shape.construct(dimensions))

def error_code(error: Error) -> U32:
  match error:
    case TensorError{error}:
      Core.error_code(error)
    case NotReducible{}:
      101
