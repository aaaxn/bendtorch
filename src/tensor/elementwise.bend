import Base
import ../shape/core.bend as Shape
import ../shape/flat.bend as Flat
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ../storage/certificates.bend as Certificates
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle

type Operation is Data:
  Add{}
  Sub{}
  Mul{}
  KeepLeft{}
  NegateLeft{}

def scalar(operation: Operation, left: F32, right: F32) -> F32:
  match operation:
    case Add{}:
      F32.add(left, right)
    case Sub{}:
      F32.sub(left, right)
    case Mul{}:
      F32.mul(left, right)
    case KeepLeft{}:
      left
    case NegateLeft{}:
      F32.neg(left)

def flat_input_indices(+plan: Shape.BroadcastPlan,
  result: Result<&2, &2, Shape.Error, Shape.FlatIndex<Shape.output_shape(plan)>>) -> Result<&2, &1, Core.Error, U32 & U32>:
  match result:
    case Fail{error}:
      Fail{Core.ShapeError{error}}
    case Done{+index}:
      +map = Shape.index_map(plan)
      Done{(Shape.index_value(Shape.left_shape(plan), Flat.project_left_index(plan, map, index)),
        Shape.index_value(Shape.right_shape(plan), Flat.project_right_index(plan, map, index)))}

def input_indices(+plan: Shape.BroadcastPlan, index: U32) -> Result<&2, &1, Core.Error, U32 & U32>:
  flat_input_indices(plan, Shape.flat_checked(Shape.output_shape(plan), U32.to_nat(index)))

type State<-shape: Shape.Shape> is Type:
  State{left: Core.Dynamic, right: Core.Dynamic, output: Core.Tensor<shape>}

type Step<-shape: Shape.Shape> is Type:
  Continue{state: State<shape>}
  Stop{left: Core.Dynamic, right: Core.Dynamic, error: Core.Error}

def step_write(-shape: Shape.Shape, left: Core.Dynamic, right: Core.Dynamic,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) -> Step<shape>:
  (output, result) = pair
  match result:
    case Fail{error}:
      Stop{left, right, error}
    case Done{unit}:
      Continue{State{left, right, output}}

def step_right(+operation: Operation, -shape: Shape.Shape, +index: U32, left: Core.Dynamic,
  output: Core.Tensor<shape>, left_value: F32, pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> Step<shape>:
  (right, result) = pair
  match result:
    case Fail{error}:
      Stop{left, right, error}
    case Done{right_value}:
      step_write(shape, left, right, Core.write_flat(shape, output, index, scalar(operation, left_value, right_value)))

def step_left(+operation: Operation, -shape: Shape.Shape, +index: U32, right: Core.Dynamic,
  output: Core.Tensor<shape>, right_index: U32, pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> Step<shape>:
  (left, result) = pair
  match result:
    case Fail{error}:
      Stop{left, right, error}
    case Done{left_value}:
      step_right(operation, shape, index, left, output, left_value, Core.read_dynamic(right, right_index))

def step_indices(+operation: Operation, -shape: Shape.Shape, +index: U32, state: State<shape>,
  result: Result<&2, &1, Core.Error, U32 & U32>) -> Step<shape>:
  State{left, right, output} = state
  match result:
    case Fail{error}:
      Stop{left, right, error}
    case Done{indices}:
      (left_index, right_index) = indices
      step_left(operation, shape, index, right, output, right_index, Core.read_dynamic(left, left_index))

def step(+operation: Operation, +plan: Shape.BroadcastPlan, +index: U32, state: State<Shape.output_shape(plan)>) ->
  Step<Shape.output_shape(plan)>:
  step_indices(operation, Shape.output_shape(plan), index, state, input_indices(plan, index))

type NaturalStep<-shape: Shape.Shape, -cursor: Nat> is Type:
  NaturalContinue{state: State<shape>, -index: U32, -exact: {U32.to_nat(index) == cursor : Nat}}
  NaturalStop{left: Core.Dynamic, right: Core.Dynamic, error: Core.Error}

type CursorWitness<-cursor: Nat, -index: U32> is Data:
  CursorWitness{-exact: {U32.to_nat(index) == cursor : Nat}}

def CursorValidity(-shape: Shape.Shape, +cursor: Nat, result: NaturalStep<shape, cursor>) -> Data:
  match result:
    case NaturalStop{left, right, error}:
      Unit
    case NaturalContinue{state, index, exact}:
      CursorWitness<cursor, index>

def natural_step_result(-shape: Shape.Shape, -cursor: Nat, -index: U32,
  -exact: {U32.to_nat(index) == cursor : Nat}, result: Step<shape>) -> NaturalStep<shape, cursor>:
  match result:
    case Stop{left, right, error}:
      NaturalStop{left, right, error}
    case Continue{state}:
      NaturalContinue{state, index, exact}

def natural_conversion_error(-shape: Shape.Shape, -cursor: Nat, state: State<shape>) -> NaturalStep<shape, cursor>:
  State{left, right, output} = state
  NaturalStop{left, right, Core.StorageError{Storage.InvalidIndex{}}}

def natural_step_decided(+operation: Operation, +plan: Shape.BroadcastPlan, +cursor: Nat, +index: U32,
  state: State<Shape.output_shape(plan)>, decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>) ->
  NaturalStep<Shape.output_shape(plan), cursor>:
  match decision:
    case Shape.BFalse{equal}:
      natural_conversion_error(Shape.output_shape(plan), cursor, state)
    case Shape.BTrue{equal}:
      natural_step_result(Shape.output_shape(plan), cursor, index,
        Lifecycle.natural_equal(U32.to_nat(index), cursor, equal), step(operation, plan, index, state))

def natural_step(+operation: Operation, +plan: Shape.BroadcastPlan, +cursor: Nat, state: State<Shape.output_shape(plan)>) ->
  NaturalStep<Shape.output_shape(plan), cursor>:
  +index = U32.from_nat(cursor)
  natural_step_decided(operation, plan, cursor, index, state, Shape.bool_case(Nat.is_eq(U32.to_nat(index), cursor)))

def finish_step(-shape: Shape.Shape, -cursor: Nat,
  rec: State<shape> -> (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  result: NaturalStep<shape, cursor>) -> (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  match result:
    case NaturalStop{left, right, error}:
      ((left, right), Fail{error})
    case NaturalContinue{state, index, exact}:
      rec(state)

def loop(fuel: Nat, +operation: Operation, +plan: Shape.BroadcastPlan, +cursor: Nat, state: State<Shape.output_shape(plan)>) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  match fuel:
    case 0n:
      State{left, right, output} = state
      ((left, right), Done{Core.pack(Shape.output_shape(plan), output)})
    case 1n+remaining:
      finish_step(Shape.output_shape(plan), cursor,
        state => loop(remaining, operation, plan, 1n+cursor, state), natural_step(operation, plan, cursor, state))

def allocated(+operation: Operation, +plan: Shape.BroadcastPlan, left: Core.Dynamic, right: Core.Dynamic,
  result: Result<&2, &1, Core.Error, Core.Tensor<Shape.output_shape(plan)>>) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      ((left, right), Fail{error})
    case Done{output}:
      loop(Shape.elements(Shape.output_shape(plan)), operation, plan, 0n, State{left, right, output})

def left_walk(-left: List<&2, Nat>, -right: List<&2, Nat>, -output: List<&2, Nat>,
  map: Shape.Index<left, right, output>, +position: Nat) -> Nat:
  match map:
    case Shape.Indexed{rows, exact}:
      Shape.walk_left(rows, position)
    case Shape.Unindexed{zero}:
      0n

def right_walk(-left: List<&2, Nat>, -right: List<&2, Nat>, -output: List<&2, Nat>,
  map: Shape.Index<left, right, output>, +position: Nat) -> Nat:
  match map:
    case Shape.Indexed{rows, exact}:
      Shape.walk_right(rows, position)
    case Shape.Unindexed{zero}:
      0n

# One output cell: `scalar(operation, left[ia], right[ib])` when the cell is
# below `count`, else `+0.0`. Both input arrays come back with the cell.
def cell_right(+operation: Operation, x: F32, la: Array<F32>, pair: Array<F32> & F32) ->
  Array<F32> & (Array<F32> & Array<F32>):
  (ra, y) = pair
  (ALeaf{scalar(operation, x, y)}, (la, ra))

def cell_left(+operation: Operation, ra: Array<F32>, +right_index: U32, pair: Array<F32> & F32) ->
  Array<F32> & (Array<F32> & Array<F32>):
  (la, x) = pair
  cell_right(operation, x, la, Array.get(F32, ra, right_index))

def cell(+operation: Operation, -le: List<&2, Nat>, -re: List<&2, Nat>, -oe: List<&2, Nat>,
  +map: Shape.Index<le, re, oe>, +position: Nat, la: Array<F32>, ra: Array<F32>, inside: Bool) ->
  Array<F32> & (Array<F32> & Array<F32>):
  match inside:
    case False{}:
      (ALeaf{0.0}, (la, ra))
    case True{}:
      cell_left(operation, ra, U32.from_nat(right_walk(le, re, oe, map, position)),
        Array.get(F32, la, U32.from_nat(left_walk(le, re, oe, map, position))))

def tree_join(a: Array<F32>, high: Array<F32> & (Array<F32> & Array<F32>)) -> Array<F32> & (Array<F32> & Array<F32>):
  (b, inputs) = high
  (ANode{a, b}, inputs)

def tree_node(rec: Array<F32> -> Array<F32> -> Array<F32> & (Array<F32> & Array<F32>),
  low: Array<F32> & (Array<F32> & Array<F32>)) -> Array<F32> & (Array<F32> & Array<F32>):
  (a, inputs) = low
  (la, ra) = inputs
  tree_join(a, rec(la, ra))

# The cells `base .. base + 2^depth` as a balanced tree, built in index order
# on one thread.
def tree(+depth: Nat, +operation: Operation, -le: List<&2, Nat>, -re: List<&2, Nat>, -oe: List<&2, Nat>,
  +map: Shape.Index<le, re, oe>, +count: Nat, +base: Nat, la: Array<F32>, ra: Array<F32>) ->
  Array<F32> & (Array<F32> & Array<F32>):
  match depth:
    case 0n:
      cell(operation, le, re, oe, map, base, la, ra, Nat.is_lt(base, count))
    case 1n+inner:
      tree_node(l => r => tree(inner, operation, le, re, oe, map, count, Nat.add(Model.capacity(inner), base), l, r),
        tree(inner, operation, le, re, oe, map, count, base, la, ra))

# The first cell of the high half of a node whose halves hold 2^inner cells.
def upper(+inner: Nat, +base: Nat) -> Nat:
  Nat.add(Model.capacity(inner), base)

def fork(lp: Array<F32> & Array<F32>, rp: Array<F32> & Array<F32>,
  low: Array<F32> -> Array<F32> -> Array<F32>, high: Array<F32> -> Array<F32> -> Array<F32>) -> Array<F32>:
  (la, lb) = lp
  (ra, rb) = rp
  a b = low(la, ra) high(lb, rb)
  ANode{a, b}

# The parallel build. The top `splits` levels fork, and each half gets its own
# clone of both input arrays, because an Array has one owner. Below the forks
# a thread builds its subtree in index order and computes the broadcast index
# map once. Every cell gets the F32 term the sequential loop writes there, so
# no output word depends on the split.
def build(+depth: Nat, +splits: Nat, +operation: Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat,
  la: Array<F32>, ra: Array<F32>) -> Array<F32>:
  match depth splits:
    case 1n+inner 1n+rest:
      +high = upper(inner, base)
      fork(Array.clone(F32, la), Array.clone(F32, ra),
        l => r => build(inner, rest, operation, plan, count, base, l, r),
        l => r => build(inner, rest, operation, plan, count, high, l, r))
    case _ _:
      Pair.fst(Array<F32>, Array<F32> & Array<F32>, tree(depth, operation,
        Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)),
        Shape.index_map(plan), count, base, la, ra))

# A subtree below the forks holds 2^15 cells: an output of 2^20 cells forks
# five levels into 32 subtrees, and an output of at most 2^15 cells does not
# fork.
def splits(+depth: Nat) -> Nat:
  Nat.sub(depth, 15n)

# A run of the build next to its proof of balance: `result` is the term the
# code computes, and its first component is balanced at `depth`. The proofs
# below only follow the code, so they live in the erased certificate of the
# output storage and cost nothing at runtime.
type Certified<-depth: Nat, -expected: Array<F32> & (Array<F32> & Array<F32>)> is Type:
  Certified{result: Array<F32> & (Array<F32> & Array<F32>), same: {result == expected : Array<F32> & (Array<F32> & Array<F32>)},
    balanced: Model.Balanced(F32, depth, Pair.fst(Array<F32>, Array<F32> & Array<F32>, result))}

def cell_right_certified(+operation: Operation, x: F32, la: Array<F32>, pair: Array<F32> & F32) ->
  Certified<0n, cell_right(operation, x, la, pair)>:
  (ra, y) = pair
  Certified{cell_right(operation, x, la, (ra, y)), {==}, Unit{}}

def cell_left_certified(+operation: Operation, ra: Array<F32>, +right_index: U32, pair: Array<F32> & F32) ->
  Certified<0n, cell_left(operation, ra, right_index, pair)>:
  (la, x) = pair
  cell_right_certified(operation, x, la, Array.get(F32, ra, right_index))

def cell_certified(+operation: Operation, -le: List<&2, Nat>, -re: List<&2, Nat>, -oe: List<&2, Nat>,
  +map: Shape.Index<le, re, oe>, +position: Nat, la: Array<F32>, ra: Array<F32>, inside: Bool) ->
  Certified<0n, cell(operation, le, re, oe, map, position, la, ra, inside)>:
  match inside:
    case False{}:
      Certified{cell(operation, le, re, oe, map, position, la, ra, False{}), {==}, Unit{}}
    case True{}:
      cell_left_certified(operation, ra, U32.from_nat(right_walk(le, re, oe, map, position)),
        Array.get(F32, la, U32.from_nat(left_walk(le, re, oe, map, position))))

def node_joined(+inner: Nat, +operation: Operation, -le: List<&2, Nat>, -re: List<&2, Nat>, -oe: List<&2, Nat>,
  +map: Shape.Index<le, re, oe>, +count: Nat, +base: Nat, -la: Array<F32>, -ra: Array<F32>, a: Array<F32>,
  -l: Array<F32>, -r: Array<F32>,
  same: {(a, (l, r)) == tree(inner, operation, le, re, oe, map, count, base, la, ra) : Array<F32> & (Array<F32> & Array<F32>)},
  balanced: Model.Balanced(F32, inner, a),
  high: Certified<inner, tree(inner, operation, le, re, oe, map, count, Nat.add(Model.capacity(inner), base), l, r)>) ->
  Certified<1n+inner, tree(1n+inner, operation, le, re, oe, map, count, base, la, ra)>:
  Certified{result, high_same, high_balanced} = high
  (b, rest) = result
  Certified{(ANode{a, b}, rest),
    %same : {(ANode{a, b}, rest) == tree_node(x => y => tree(inner, operation, le, re, oe, map, count, Nat.add(Model.capacity(inner), base), x, y), _) : Array<F32> & (Array<F32> & Array<F32>)}
    %high_same : {(ANode{a, b}, rest) == tree_join(a, _) : Array<F32> & (Array<F32> & Array<F32>)}
    {==},
    (balanced, high_balanced)}

def node_split(+inner: Nat, +operation: Operation, -le: List<&2, Nat>, -re: List<&2, Nat>, -oe: List<&2, Nat>,
  +map: Shape.Index<le, re, oe>, +count: Nat, +base: Nat, -la: Array<F32>, -ra: Array<F32>, result: Array<F32> & (Array<F32> & Array<F32>),
  same: {result == tree(inner, operation, le, re, oe, map, count, base, la, ra) : Array<F32> & (Array<F32> & Array<F32>)},
  balanced: Model.Balanced(F32, inner, Pair.fst(Array<F32>, Array<F32> & Array<F32>, result)),
  next: @l: Array<F32> -> @r: Array<F32> -> Certified<inner, tree(inner, operation, le, re, oe, map, count, Nat.add(Model.capacity(inner), base), l, r)>) ->
  Certified<1n+inner, tree(1n+inner, operation, le, re, oe, map, count, base, la, ra)>:
  (a, inputs) = result
  (l, r) = inputs
  node_joined(inner, operation, le, re, oe, map, count, base, la, ra, a, l, r, same, balanced, next(l, r))

def tree_node_certified(+inner: Nat, +operation: Operation, -le: List<&2, Nat>, -re: List<&2, Nat>, -oe: List<&2, Nat>,
  +map: Shape.Index<le, re, oe>, +count: Nat, +base: Nat, -la: Array<F32>, -ra: Array<F32>,
  low: Certified<inner, tree(inner, operation, le, re, oe, map, count, base, la, ra)>,
  next: @l: Array<F32> -> @r: Array<F32> -> Certified<inner, tree(inner, operation, le, re, oe, map, count, Nat.add(Model.capacity(inner), base), l, r)>) ->
  Certified<1n+inner, tree(1n+inner, operation, le, re, oe, map, count, base, la, ra)>:
  Certified{result, same, balanced} = low
  node_split(inner, operation, le, re, oe, map, count, base, la, ra, result, same, balanced, next)

def tree_certified(+depth: Nat, +operation: Operation, -le: List<&2, Nat>, -re: List<&2, Nat>, -oe: List<&2, Nat>,
  +map: Shape.Index<le, re, oe>, +count: Nat, +base: Nat, la: Array<F32>, ra: Array<F32>) ->
  Certified<depth, tree(depth, operation, le, re, oe, map, count, base, la, ra)>:
  match depth:
    case 0n:
      cell_certified(operation, le, re, oe, map, base, la, ra, Nat.is_lt(base, count))
    case 1n+inner:
      tree_node_certified(inner, operation, le, re, oe, map, count, base, la, ra,
        tree_certified(inner, operation, le, re, oe, map, count, base, la, ra),
        l => r => tree_certified(inner, operation, le, re, oe, map, count, Nat.add(Model.capacity(inner), base), l, r))

def tree_balanced(+depth: Nat, +operation: Operation, -le: List<&2, Nat>, -re: List<&2, Nat>, -oe: List<&2, Nat>,
  +map: Shape.Index<le, re, oe>, +count: Nat, +base: Nat, -la: Array<F32>, -ra: Array<F32>,
  certified: Certified<depth, tree(depth, operation, le, re, oe, map, count, base, la, ra)>) ->
  Model.Balanced(F32, depth, Pair.fst(Array<F32>, Array<F32> & Array<F32>, tree(depth, operation, le, re, oe, map, count, base, la, ra))):
  Certified{result, same, balanced} = certified
  %same : Model.Balanced(F32, depth, Pair.fst(Array<F32>, Array<F32> & Array<F32>, _))
  balanced

def fork_balanced(+inner: Nat, lp: Array<F32> & Array<F32>, rp: Array<F32> & Array<F32>,
  -low: Array<F32> -> Array<F32> -> Array<F32>, -high: Array<F32> -> Array<F32> -> Array<F32>,
  low_valid: @l: Array<F32> -> @r: Array<F32> -> Model.Balanced(F32, inner, low(l, r)),
  high_valid: @l: Array<F32> -> @r: Array<F32> -> Model.Balanced(F32, inner, high(l, r))) ->
  Model.Balanced(F32, 1n+inner, fork(lp, rp, low, high)):
  (la, lb) = lp
  (ra, rb) = rp
  (low_valid(la, ra), high_valid(lb, rb))

# The parallel build returns a balanced tree of its depth, whatever the split.
def build_balanced(+depth: Nat, +splits: Nat, +operation: Operation, +plan: Shape.BroadcastPlan, +count: Nat, +base: Nat,
  la: Array<F32>, ra: Array<F32>) -> Model.Balanced(F32, depth, build(depth, splits, operation, plan, count, base, la, ra)):
  match depth splits:
    case 1n+inner 1n+rest:
      +high = upper(inner, base)
      fork_balanced(inner, Array.clone(F32, la), Array.clone(F32, ra),
        l => r => build(inner, rest, operation, plan, count, base, l, r),
        l => r => build(inner, rest, operation, plan, count, high, l, r),
        l => r => build_balanced(inner, rest, operation, plan, count, base, l, r),
        l => r => build_balanced(inner, rest, operation, plan, count, high, l, r))
    case 0n _:
      tree_balanced(0n, operation,
        Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)),
        Shape.index_map(plan), count, base, la, ra, tree_certified(0n, operation,
          Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)),
          Shape.index_map(plan), count, base, la, ra))
    case 1n+inner 0n:
      tree_balanced(1n+inner, operation,
        Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)),
        Shape.index_map(plan), count, base, la, ra, tree_certified(1n+inner, operation,
          Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)),
          Shape.index_map(plan), count, base, la, ra))

# The input handed back to the caller and an array copy for the build. A
# tensor with empty storage has no cell to read.
def take_owned(+shape: Shape.Shape, +depth: Nat, +length: Nat, -array: Array<F32>,
  pair: Array<F32> & Array<F32>, -same: {pair == Array.clone(F32, array) : Array<F32> & Array<F32>},
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) -> Core.Dynamic & Array<F32>:
  (kept, copy) = pair
  (Core.Packed{shape, Core.TTensor{Storage.Owned{depth, length, kept,
    Certificates.clone_first(F32, depth, array, (kept, copy), same, balanced), valid}, exact}}, copy)

def take(dynamic: Core.Dynamic) -> Core.Dynamic & Array<F32>:
  Core.Packed{+shape, Core.TTensor{storage, exact}} = dynamic
  match storage:
    case Storage.Empty{}:
      (Core.Packed{shape, Core.TTensor{Storage.Empty{}, exact}}, Array.new(F32, 0n, 0.0))
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      take_owned(shape, depth, length, array, Array.clone(F32, array), {==}, balanced, valid, exact)

def parallel_output(+operation: Operation, +plan: Shape.BroadcastPlan, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat},
  left: Core.Dynamic & Array<F32>, right: Core.Dynamic & Array<F32>) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  (kept_left, la) = left
  (kept_right, ra) = right
  ((kept_left, kept_right), Done{Core.pack(Shape.output_shape(plan), Core.TTensor{Storage.Owned{depth, length,
    build(depth, splits(depth), operation, plan, length, 0n, la, ra),
    build_balanced(depth, splits(depth), operation, plan, length, 0n, la, ra),
    valid}, exact})})

def parallel_allocated(+operation: Operation, +plan: Shape.BroadcastPlan, left: Core.Dynamic, right: Core.Dynamic,
  result: Result<&2, &1, Core.Error, Core.Tensor<Shape.output_shape(plan)>>) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      ((left, right), Fail{error})
    case Done{Core.TTensor{storage, exact}}:
      match storage:
        case Storage.Empty{}:
          ((left, right), Done{Core.pack(Shape.output_shape(plan), Core.TTensor{Storage.Empty{}, exact})})
        case Storage.Owned{+depth, +length, array, balanced, valid}:
          parallel_output(operation, plan, depth, length, valid, exact, take(left), take(right))

def parallel_planned(operation: Operation, left: Core.Dynamic, right: Core.Dynamic,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      ((left, right), Fail{Core.ShapeError{error}})
    case Done{+plan}:
      parallel_allocated(operation, plan, left, right, Core.new(Shape.output_shape(plan), 0.0))

def binary(operation: Operation, left: Core.Dynamic, right: Core.Dynamic) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  Core.Packed{+left_shape, left_tensor} = left
  Core.Packed{+right_shape, right_tensor} = right
  parallel_planned(operation, Core.pack(left_shape, left_tensor), Core.pack(right_shape, right_tensor),
    Shape.broadcast_checked(left_shape, right_shape))

def add(left: Core.Dynamic, right: Core.Dynamic) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  binary(Add{}, left, right)

def sub(left: Core.Dynamic, right: Core.Dynamic) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  binary(Sub{}, left, right)

def mul(left: Core.Dynamic, right: Core.Dynamic) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  binary(Mul{}, left, right)

def step_observe(-shape: Shape.Shape, step: Step<shape>, index: U32) -> Maybe<&2, F32>:
  match step:
    case Stop{left, right, error}:
      None{}
    case Continue{State{left, right, output}}:
      Core.observe(shape, output, index)

def expected_right(operation: Operation, left: F32, right: Maybe<&2, F32>) -> Maybe<&2, F32>:
  match right:
    case None{}:
      None{}
    case Some{value}:
      Some{scalar(operation, left, value)}

def expected_left(operation: Operation, right: Core.Dynamic, right_index: U32, left: Maybe<&2, F32>) -> Maybe<&2, F32>:
  match left:
    case None{}:
      None{}
    case Some{value}:
      expected_right(operation, value, Core.checked_observe_dynamic(right, right_index))

def expected_indices(operation: Operation, left: Core.Dynamic, right: Core.Dynamic,
  result: Result<&2, &1, Core.Error, U32 & U32>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{indices}:
      (left_index, right_index) = indices
      expected_left(operation, right, right_index, Core.checked_observe_dynamic(left, left_index))

def expected(+operation: Operation, +plan: Shape.BroadcastPlan, left: Core.Dynamic, right: Core.Dynamic, index: U32) -> Maybe<&2, F32>:
  expected_indices(operation, left, right, input_indices(plan, index))

def PrefixMapping(+operation: Operation, +plan: Shape.BroadcastPlan,
  -left: Core.Dynamic, -right: Core.Dynamic, +next: Nat, -output: Core.Dynamic) -> Type:
  @+index: U32 -> @earlier: Shape.LT(U32.to_nat(index), next) ->
    {Core.observe_dynamic(output, index) == expected(operation, plan, left, right, index) : Maybe<&2, F32>}

def PrefixInvariant(+operation: Operation, +plan: Shape.BroadcastPlan,
  -left: Core.Dynamic, -right: Core.Dynamic, +next: Nat, -output: Core.Dynamic) -> Type:
  @+index: U32 -> @earlier: Shape.LT(U32.to_nat(index), next) ->
    Core.ObservationChecked(output, index) &
      {Core.observe_dynamic(output, index) == expected(operation, plan, left, right, index) : Maybe<&2, F32>}

def SuccessfulMapping(+operation: Operation, +plan: Shape.BroadcastPlan, -left: Core.Dynamic, -right: Core.Dynamic,
  pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Type:
  (inputs, result) = pair
  match result:
    case Fail{error}:
      Unit
    case Done{output}:
      PrefixMapping(operation, plan, left, right, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), output)

def SameSource(-original: Core.Dynamic, -retained: Core.Dynamic) -> Type:
  @+other: U32 -> {Core.checked_observe_dynamic(retained, other) == Core.checked_observe_dynamic(original, other) : Maybe<&2, F32>}

def StepMeaning(+operation: Operation, +plan: Shape.BroadcastPlan, -index: U32,
  original: State<Shape.output_shape(plan)>, result: Step<Shape.output_shape(plan)>) -> Type:
  State{original_left, original_right, original_output} = original
  match result:
    case Stop{left, right, error}:
      Unit
    case Continue{State{left, right, output}}:
      SameSource(original_left, left) &
        (SameSource(original_right, right) &
          (Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), output}, index) &
            ({Core.observe(Shape.output_shape(plan), output, index) == expected(operation, plan, original_left, original_right, index) : Maybe<&2, F32>} &
              (@+other: U32 -> @checked: Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), original_output}, other) ->
                @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
                  Core.ObservationChecked(Core.Packed{Shape.output_shape(plan), output}, other) &
                    {Core.observe(Shape.output_shape(plan), output, other) == Core.observe(Shape.output_shape(plan), original_output, other) : Maybe<&2, F32>}))))

type StepCertificate<-operation: Operation, -plan: Shape.BroadcastPlan, -index: U32, -original: State<Shape.output_shape(plan)>> is Type:
  StepCertificate{result: Step<Shape.output_shape(plan)>,
    exact: {result == step(operation, plan, index, original) : Step<Shape.output_shape(plan)>},
    -meaning: StepMeaning(operation, plan, index, original, result)}

def NaturalMeaning(+operation: Operation, +plan: Shape.BroadcastPlan, -cursor: Nat,
  original: State<Shape.output_shape(plan)>, result: NaturalStep<Shape.output_shape(plan), cursor>) -> Type:
  match result:
    case NaturalStop{left, right, error}:
      Unit
    case NaturalContinue{state, index, exact}:
      StepMeaning(operation, plan, index, original, Continue{state})

type ConvertedCertificate<-operation: Operation, -plan: Shape.BroadcastPlan, -cursor: Nat, -index: U32,
  -decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>, -original: State<Shape.output_shape(plan)>> is Type:
  ConvertedCertificate{result: NaturalStep<Shape.output_shape(plan), cursor>,
    exact: {result == natural_step_decided(operation, plan, cursor, index, original, decision) : NaturalStep<Shape.output_shape(plan), cursor>},
    -meaning: NaturalMeaning(operation, plan, cursor, original, result)}

type NaturalCertificate<-operation: Operation, -plan: Shape.BroadcastPlan, -cursor: Nat, -original: State<Shape.output_shape(plan)>> is Type:
  NaturalCertificate{result: NaturalStep<Shape.output_shape(plan), cursor>,
    exact: {result == natural_step(operation, plan, cursor, original) : NaturalStep<Shape.output_shape(plan), cursor>},
    -meaning: NaturalMeaning(operation, plan, cursor, original, result)}

def StateInvariant(+operation: Operation, +plan: Shape.BroadcastPlan, -original_left: Core.Dynamic, -original_right: Core.Dynamic,
  +cursor: Nat, state: State<Shape.output_shape(plan)>) -> Type:
  State{left, right, output} = state
  SameSource(original_left, left) &
    (SameSource(original_right, right) & PrefixInvariant(operation, plan, original_left, original_right, cursor, Core.Packed{Shape.output_shape(plan), output}))

type InvariantModel<-operation: Operation, -plan: Shape.BroadcastPlan, -original_left: Core.Dynamic, -original_right: Core.Dynamic,
  -cursor: Nat, -state: State<Shape.output_shape(plan)>> is Data:
  InvariantModel{-invariant: StateInvariant(operation, plan, original_left, original_right, cursor, state)}

def LoopMeaning(+operation: Operation, +plan: Shape.BroadcastPlan, -original_left: Core.Dynamic, -original_right: Core.Dynamic,
  +limit: Nat, pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Type:
  (inputs, result) = pair
  match result:
    case Fail{error}:
      Unit
    case Done{output}:
      PrefixInvariant(operation, plan, original_left, original_right, limit, output)

type LoopCertificate<-fuel: Nat, -operation: Operation, -plan: Shape.BroadcastPlan, -cursor: Nat,
  -original_left: Core.Dynamic, -original_right: Core.Dynamic, -original: State<Shape.output_shape(plan)>> is Type:
  LoopCertificate{result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
    exact: {result == loop(fuel, operation, plan, cursor, original) : (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>},
    -meaning: LoopMeaning(operation, plan, original_left, original_right, Nat.add(cursor, fuel), result)}

type MappingModel<-operation: Operation, -plan: Shape.BroadcastPlan, -original_left: Core.Dynamic, -original_right: Core.Dynamic,
  -result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>> is Data:
  MappingModel{-mapping: SuccessfulMapping(operation, plan, original_left, original_right, result)}

def plan_for(left: Core.Dynamic, right: Core.Dynamic) -> Result<&2, &2, Shape.Error, Shape.BroadcastPlan>:
  Core.Packed{left_shape, left_tensor} = left
  Core.Packed{right_shape, right_tensor} = right
  Shape.broadcast_checked(left_shape, right_shape)

def PlannedMeaning(+operation: Operation, -left: Core.Dynamic, -right: Core.Dynamic,
  result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  plan: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) -> Type:
  match plan:
    case Fail{error}:
      Unit
    case Done{+plan}:
      SuccessfulMapping(operation, plan, left, right, result)

def BinaryMeaning(+operation: Operation, left: Core.Dynamic, right: Core.Dynamic,
  result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Type:
  PlannedMeaning(operation, left, right, result, plan_for(left, right))

type BinaryModel<-operation: Operation, -left: Core.Dynamic, -right: Core.Dynamic,
  -result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>> is Data:
  BinaryModel{-mapping: BinaryMeaning(operation, left, right, result)}
