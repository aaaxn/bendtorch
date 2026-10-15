import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/LAWS.bend as StorageLaws
import ../storage/arith.bend as Arith
import ../tensor/core.bend as Core
import ../tensor/layout.bend as Layout
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/elementwise.bend as Elementwise
import ../tensor/word-conversion.bend as Conversion
import ../tensor/permutation-mapping.bend as Mapping
import ../tensor/projection-LAWS.bend as Projection
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/pointwise.bend as Pointwise
import ../tensor/contracts/pointwise-proof.bend as PointwiseProof
import ../tensor/contracts/broadcast.bend as PointwiseBroadcast
import ./broadcast.bend as Broadcast
import ./reduce-to-fold-PROOF.bend as ReduceTo
import ./broadcast-pointwise-LAWS.bend as Laws

# The output history after `count` steps of the sequential elementwise loop:
# the pointwise family history of Elementwise.scalar with the broadcast maps.
def history(+count: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>) -> List<&2, StorageLaws.Op>:
  Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~PointwiseBroadcast.left_map,
    ~PointwiseBroadcast.right_map, count, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops)

def left_storage(+plan: Shape.BroadcastPlan, +left: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat) -> Storage.Storage:
  PointwiseProof.input_storage(~Shape.BroadcastPlan, ~PointwiseBroadcast.left_map, plan, left, fill, ops, count)

def right_storage(+plan: Shape.BroadcastPlan, +right: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat) -> Storage.Storage:
  PointwiseProof.input_storage(~Shape.BroadcastPlan, ~PointwiseBroadcast.right_map, plan, right, fill, ops, count)

def output_storage(+count: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>) -> Storage.Storage:
  Replayed.storage(Shape.output_shape(plan), 0.0, history(count, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops))

def Shaped(+fuel: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +right: Shape.Shape,
  +cursor: Nat, -left_data: Storage.Storage, -right_data: Storage.Storage, -output_data: Storage.Storage,
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type) -> Type:
  @-left_fits: {Lifecycle.logical_length(left_data) == Shape.logical_numel(Shape.dimensions(left)) : Nat} ->
  @-right_fits: {Lifecycle.logical_length(right_data) == Shape.logical_numel(Shape.dimensions(right)) : Nat} ->
  @-output_fits: {Lifecycle.logical_length(output_data) == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat} ->
    Q(Elementwise.loop(fuel, operation, plan, cursor,
      Elementwise.State{Core.Packed{left, Core.TTensor{left_data, left_fits}}, Core.Packed{right, Core.TTensor{right_data, right_fits}},
        Core.TTensor{output_data, output_fits}}))

def Start(+fuel: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +cursor: Nat,
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type) -> Type:
  Shaped(fuel, operation, plan, left, right, cursor, left_storage(plan, left, left_fill, left_ops, cursor),
    right_storage(plan, right, right_fill, right_ops, cursor),
    output_storage(cursor, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Q)

def Final(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat,
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type) -> Type:
  @-left_fits: {Lifecycle.logical_length(left_storage(plan, left, left_fill, left_ops, count)) == Shape.logical_numel(Shape.dimensions(left)) : Nat} ->
  @-right_fits: {Lifecycle.logical_length(right_storage(plan, right, right_fill, right_ops, count)) ==
    Shape.logical_numel(Shape.dimensions(right)) : Nat} ->
  @-output_fits: {Lifecycle.logical_length(output_storage(count, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops)) ==
    Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat} ->
    Q(((Core.Packed{left, Core.TTensor{left_storage(plan, left, left_fill, left_ops, count), left_fits}},
      Core.Packed{right, Core.TTensor{right_storage(plan, right, right_fill, right_ops, count), right_fits}}),
      Done{Core.Packed{Shape.output_shape(plan),
        Core.TTensor{output_storage(count, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), output_fits}}}))

def final_at(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, -end: Nat, -count: Nat, same: {end == count : Nat},
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type,
  h: Final(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, end, Q)) ->
  Final(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count, Q):
  %same : Final(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, _, Q)
  h

def shift(+rest: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +cursor: Nat,
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type,
  next: Start(rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+cursor, Q)) ->
  Shaped(rest, operation, plan, left, right, 1n+cursor,
    Replayed.storage(left, left_fill, Replayed.snoc(
      List.append(&2, StorageLaws.Op, left_ops, Pointwise.reads(~Shape.BroadcastPlan, ~PointwiseBroadcast.left_map, plan, cursor)),
      StorageLaws.Read{U32.from_nat(PointwiseBroadcast.left_map(plan, cursor))})),
    Replayed.storage(right, right_fill, Replayed.snoc(
      List.append(&2, StorageLaws.Op, right_ops, Pointwise.reads(~Shape.BroadcastPlan, ~PointwiseBroadcast.right_map, plan, cursor)),
      StorageLaws.Read{U32.from_nat(PointwiseBroadcast.right_map(plan, cursor))})),
    Replayed.storage(Shape.output_shape(plan), 0.0, Replayed.snoc(
      history(cursor, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops),
      StorageLaws.Write{U32.from_nat(cursor), Elementwise.scalar(operation,
        Replayed.value(left, left_fill,
          List.append(&2, StorageLaws.Op, left_ops, Pointwise.reads(~Shape.BroadcastPlan, ~PointwiseBroadcast.left_map, plan, cursor)),
          U32.from_nat(PointwiseBroadcast.left_map(plan, cursor))),
        Replayed.value(right, right_fill,
          List.append(&2, StorageLaws.Op, right_ops, Pointwise.reads(~Shape.BroadcastPlan, ~PointwiseBroadcast.right_map, plan, cursor)),
          U32.from_nat(PointwiseBroadcast.right_map(plan, cursor))))})),
    Q):
  +left_reads = Pointwise.reads(~Shape.BroadcastPlan, ~PointwiseBroadcast.left_map, plan, cursor)
  +right_reads = Pointwise.reads(~Shape.BroadcastPlan, ~PointwiseBroadcast.right_map, plan, cursor)
  +left_index = U32.from_nat(PointwiseBroadcast.left_map(plan, cursor))
  +right_index = U32.from_nat(PointwiseBroadcast.right_map(plan, cursor))
  +left_read = {StorageLaws.Read{left_index} : StorageLaws.Op}
  +right_read = {StorageLaws.Read{right_index} : StorageLaws.Op}
  +left_seen = List.append(&2, StorageLaws.Op, left_ops, left_reads)
  +right_seen = List.append(&2, StorageLaws.Op, right_ops, right_reads)
  +x = Replayed.value(left, left_fill, left_seen, left_index)
  +y = Replayed.value(right, right_fill, right_seen, right_index)
  +left_word = Replayed.value(left, left_fill, left_ops, left_index)
  +right_word = Replayed.value(right, right_fill, right_ops, right_index)
  +before = history(cursor, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops)
  -right_after = Replayed.storage(right, right_fill, Replayed.snoc(right_seen, right_read))
  -output_after = Replayed.storage(Shape.output_shape(plan), 0.0,
    Replayed.snoc(before, StorageLaws.Write{U32.from_nat(cursor), Elementwise.scalar(operation, x, y)}))
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, ops => Replayed.storage(left, left_fill, ops),
    List.append(&2, StorageLaws.Op, left_ops, Replayed.snoc(left_reads, left_read)), Replayed.snoc(left_seen, left_read),
    PointwiseProof.append_snoc(left_ops, left_reads, left_read)) :
    Shaped(rest, operation, plan, left, right, 1n+cursor, _, right_after, output_after, Q)
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, ops => Replayed.storage(right, right_fill, ops),
    List.append(&2, StorageLaws.Op, right_ops, Replayed.snoc(right_reads, right_read)), Replayed.snoc(right_seen, right_read),
    PointwiseProof.append_snoc(right_ops, right_reads, right_read)) :
    Shaped(rest, operation, plan, left, right, 1n+cursor, left_storage(plan, left, left_fill, left_ops, 1n+cursor), _, output_after, Q)
  %Equal.cong(F32, Storage.Storage,
    word => Replayed.storage(Shape.output_shape(plan), 0.0, Replayed.snoc(before, StorageLaws.Write{U32.from_nat(cursor), word})),
    Elementwise.scalar(operation, left_word, right_word), Elementwise.scalar(operation, x, y),
    Equal.trans(F32, Elementwise.scalar(operation, left_word, right_word), Elementwise.scalar(operation, x, right_word),
      Elementwise.scalar(operation, x, y),
      Equal.cong(F32, F32, word => Elementwise.scalar(operation, word, right_word), left_word, x,
        Equal.sym(F32, x, left_word, PointwiseProof.value_reads(~Shape.BroadcastPlan, ~PointwiseBroadcast.left_map, plan, cursor,
          left, left_fill, left_ops, left_index))),
      Equal.cong(F32, F32, word => Elementwise.scalar(operation, x, word), right_word, y,
        Equal.sym(F32, y, right_word, PointwiseProof.value_reads(~Shape.BroadcastPlan, ~PointwiseBroadcast.right_map, plan, cursor,
          right, right_fill, right_ops, right_index))))) :
    Shaped(rest, operation, plan, left, right, 1n+cursor, left_storage(plan, left, left_fill, left_ops, 1n+cursor),
      right_storage(plan, right, right_fill, right_ops, 1n+cursor), _, Q)
  next

# The loop converts the cursor to a U32 and checks the round trip; below the
# output count the check holds, so its failure branch is unreachable.
def decided_lift(+rest: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +cursor: Nat,
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type,
  -state: Elementwise.State<Shape.output_shape(plan)>,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)>,
  same: {Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor) == True{} : Bool},
  h: @-exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat} ->
    Q(Elementwise.finish_step(Shape.output_shape(plan), cursor, state => Elementwise.loop(rest, operation, plan, 1n+cursor, state),
      Elementwise.natural_step_result(Shape.output_shape(plan), cursor, U32.from_nat(cursor), exact,
        Elementwise.step(operation, plan, U32.from_nat(cursor), state))))) ->
  Q(Elementwise.finish_step(Shape.output_shape(plan), cursor, state => Elementwise.loop(rest, operation, plan, 1n+cursor, state),
    Elementwise.natural_step_decided(operation, plan, cursor, U32.from_nat(cursor), state, decision))):
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd(Q(Elementwise.finish_step(Shape.output_shape(plan), cursor, state => Elementwise.loop(rest, operation, plan, 1n+cursor, state),
        Elementwise.natural_step_decided(operation, plan, cursor, U32.from_nat(cursor), state, Shape.BFalse{equal}))),
        Arith.clash(Equal.trans(Bool, False{}, Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor), True{}, equal, same)))
    case Shape.BTrue{equal}:
      h(Lifecycle.natural_equal(U32.to_nat(U32.from_nat(cursor)), cursor, equal))

# input_indices returns the two broadcast offsets the specification gives.
def indices_lift(+left_index: U32, +right_index: U32, result: Result<&2, &1, Core.Error, U32 & U32>,
  found: {Projection.indices_result(result) == Some{(left_index, right_index)} : Maybe<&1, U32 & U32>},
  -R: Result<&2, &1, Core.Error, U32 & U32> -> Type, h: R(Done{(left_index, right_index)})) -> R(result):
  match result:
    case Fail{error}:
      Empty.absurd(R(Fail{error}),
        Arith.clash(Equal.cong(Maybe<&1, U32 & U32>, Bool, value => Maybe.is_some(&1, U32 & U32, value), None{}, Some{(left_index, right_index)},
          found)))
    case Done{indices}:
      %Equal.sym(U32 & U32, indices, (left_index, right_index),
        ReduceTo.some_injective(U32 & U32, indices, (left_index, right_index), found)) : R(Done{_})
      h

# One step of the sequential loop on replayed inputs: it reads the left word at
# left_map(plan, cursor) and the right word at right_map(plan, cursor), and
# writes scalar(operation, x, y) at the cursor, which is the pointwise family's
# step with the same maps.
def step_lift(+rest: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +cursor: Nat,
  +below: {Nat.is_lt(cursor, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool},
  +left_shape: {Shape.left_shape(plan) == left : Shape.Shape}, +right_shape: {Shape.right_shape(plan) == right : Shape.Shape},
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type,
  next: Start(rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+cursor, Q)) ->
  Start(1n+rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, cursor, Q):
  +shape = Shape.output_shape(plan)
  +index = U32.from_nat(cursor)
  +left_seen = List.append(&2, StorageLaws.Op, left_ops, Pointwise.reads(~Shape.BroadcastPlan, ~PointwiseBroadcast.left_map, plan, cursor))
  +right_seen = List.append(&2, StorageLaws.Op, right_ops, Pointwise.reads(~Shape.BroadcastPlan, ~PointwiseBroadcast.right_map, plan, cursor))
  +before = history(cursor, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops)
  +left_index = U32.from_nat(PointwiseBroadcast.left_map(plan, cursor))
  +right_index = U32.from_nat(PointwiseBroadcast.right_map(plan, cursor))
  +x = Replayed.value(left, left_fill, left_seen, left_index)
  +y = Replayed.value(right, right_fill, right_seen, right_index)
  +left_inside = PointwiseProof.word_inside(PointwiseBroadcast.left_map(plan, cursor), left,
    ReduceTo.target_below(plan, left, left_shape, cursor, below))
  +right_inside = PointwiseProof.word_inside(PointwiseBroadcast.right_map(plan, cursor), right,
    ReduceTo.source_below(plan, right, right_shape, cursor, below))
  +output_inside = PointwiseProof.word_inside(cursor, shape, below)
  left_fits => right_fits => output_fits =>
    -state = {Elementwise.State{Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_seen), left_fits}},
      Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_seen), right_fits}},
      Core.TTensor{Replayed.storage(shape, 0.0, before), output_fits}} : Elementwise.State<shape>}
    -right_input = {Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_seen), right_fits}} : Core.Dynamic}
    -output = {Core.TTensor{Replayed.storage(shape, 0.0, before), output_fits} : Core.Tensor<shape>}
    decided_lift(rest, operation, plan, cursor, Q, state, Shape.bool_case(Nat.is_eq(U32.to_nat(index), cursor)),
      Mapping.natural_equal_true(U32.to_nat(index), cursor, PointwiseProof.word_exact(cursor, shape, below)),
      exact => indices_lift(left_index, right_index, Elementwise.input_indices(plan, index), ReduceTo.indices_found(plan, cursor, below),
        result => Q(Elementwise.finish_step(shape, cursor, state => Elementwise.loop(rest, operation, plan, 1n+cursor, state),
          Elementwise.natural_step_result(shape, cursor, index, exact, Elementwise.step_indices(operation, shape, index, state, result)))),
        Replayed.read_lift(left, left_fill, left_seen, left_fits, left_index, left_inside,
          pair => Q(Elementwise.finish_step(shape, cursor, state => Elementwise.loop(rest, operation, plan, 1n+cursor, state),
            Elementwise.natural_step_result(shape, cursor, index, exact,
              Elementwise.step_left(operation, shape, index, right_input, output, right_index, Core.read_dynamic_result(left, pair))))),
          left_after => Replayed.read_lift(right, right_fill, right_seen, right_fits, right_index, right_inside,
            pair => Q(Elementwise.finish_step(shape, cursor, state => Elementwise.loop(rest, operation, plan, 1n+cursor, state),
              Elementwise.natural_step_result(shape, cursor, index, exact,
                Elementwise.step_right(operation, shape, index,
                  Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, Replayed.snoc(left_seen, StorageLaws.Read{left_index})), left_after}},
                  output, x, Core.read_dynamic_result(right, pair))))),
            right_after => Replayed.write_lift(shape, 0.0, before, output_fits, index, Elementwise.scalar(operation, x, y), output_inside,
              pair => Q(Elementwise.finish_step(shape, cursor, state => Elementwise.loop(rest, operation, plan, 1n+cursor, state),
                Elementwise.natural_step_result(shape, cursor, index, exact,
                  Elementwise.step_write(shape,
                    Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, Replayed.snoc(left_seen, StorageLaws.Read{left_index})), left_after}},
                    Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, Replayed.snoc(right_seen, StorageLaws.Read{right_index})), right_after}},
                    pair)))),
              output_after => shift(rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, cursor, Q, next)(
                left_after, right_after, output_after))))))

def loop_lift(+fuel: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +cursor: Nat,
  +steps: {Nat.add(cursor, fuel) == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat},
  +left_shape: {Shape.left_shape(plan) == left : Shape.Shape}, +right_shape: {Shape.right_shape(plan) == right : Shape.Shape},
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type,
  h: Final(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops,
    Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), Q)) ->
  Start(fuel, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, cursor, Q):
  match fuel:
    case 0n:
      +count = Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))
      final_at(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count, cursor,
        Equal.sym(Nat, cursor, count, PointwiseProof.steps_done(cursor, count, steps)), Q, h)
    case 1n+rest:
      +count = Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))
      step_lift(rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, cursor,
        PointwiseProof.cursor_below(cursor, rest, count, steps), left_shape, right_shape, Q,
        loop_lift(rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+cursor,
          PointwiseProof.steps_next(cursor, rest, count, steps), left_shape, right_shape, Q, h))

def start_cast(+fuel: Nat, +operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type,
  start: Start(fuel, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 0n, Q)) ->
  Shaped(fuel, operation, plan, left, right, 0n, Replayed.storage(left, left_fill, left_ops), Replayed.storage(right, right_fill, right_ops),
    Replayed.storage(Shape.output_shape(plan), 0.0, Nil{}), Q):
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, ops => Replayed.storage(left, left_fill, ops),
    List.append(&2, StorageLaws.Op, left_ops, Nil{}), left_ops, PointwiseProof.append_nil(left_ops)) :
    Shaped(fuel, operation, plan, left, right, 0n, _, Replayed.storage(right, right_fill, right_ops),
      Replayed.storage(Shape.output_shape(plan), 0.0, Nil{}), Q)
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, ops => Replayed.storage(right, right_fill, ops),
    List.append(&2, StorageLaws.Op, right_ops, Nil{}), right_ops, PointwiseProof.append_nil(right_ops)) :
    Shaped(fuel, operation, plan, left, right, 0n, left_storage(plan, left, left_fill, left_ops, 0n), _,
      Replayed.storage(Shape.output_shape(plan), 0.0, Nil{}), Q)
  start

# broadcast_to on a replayed input whose shape broadcasts to exactly the
# target: the planner and the target check pass, both allocations succeed, and
# the loop runs KeepLeft with the input on the left and the zero tensor of the
# target on the right.
def run_lift(+shape: Shape.Shape, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(shape, target) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  +same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -P: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic> -> Type,
  h: Final(Elementwise.KeepLeft{}, plan, shape, fill, ops, target, 0.0, Nil{}, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))),
    pair => P(Broadcast.finish(pair)))) ->
  @-fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(Broadcast.broadcast_to(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target)):
  +output = Shape.output_shape(plan)
  fits =>
    -input = {Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}} : Core.Dynamic}
    %Equal.sym(Result<&2, &2, Shape.Error, Shape.BroadcastPlan>, Shape.broadcast_checked(shape, target), Done{plan}, planned) :
      P(Broadcast.planned(input, target, _))
    %Equal.sym(Bool, Layout.words_same(Shape.dimensions(output), Shape.dimensions(target)), True{}, ReduceTo.fitted_same(target, plan, same)) :
      P(Broadcast.fitted(input, target, plan, _))
    Replayed.new_lift(target, 0.0, result => P(Broadcast.zeroed(input, target, plan, result)),
      zeros_fits => Replayed.new_lift(output, 0.0,
        result => P(Broadcast.finish(Elementwise.allocated(Elementwise.KeepLeft{}, plan, input,
          Core.Packed{target, Core.TTensor{Replayed.storage(target, 0.0, Nil{}), zeros_fits}}, result))),
        output_fits => start_cast(Shape.elements(output), Elementwise.KeepLeft{}, plan, shape, fill, ops, target, 0.0, Nil{},
          pair => P(Broadcast.finish(pair)),
          loop_lift(Shape.elements(output), Elementwise.KeepLeft{}, plan, shape, fill, ops, target, 0.0, Nil{}, 0n,
            Shape.elements_exact(output), ReduceTo.plan_left(shape, target, plan, planned), ReduceTo.plan_right(shape, target, plan, planned),
            pair => P(Broadcast.finish(pair)), h))(fits, zeros_fits, output_fits)))

def output_below(+plan: Shape.BroadcastPlan, +target: Shape.Shape, +index: U32,
  same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}:
  %Equal.sym(List<&2, U32>, Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(target), same) :
    {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(_)) == True{} : Bool}
  inside

def broadcast_to_pointwise(+shape: Shape.Shape, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(shape, target) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  +same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Laws.BroadcastToContract<target, shape,
    index => Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, index))),
    other => Replayed.value(shape, fill, ops, other),
    Broadcast.broadcast_to(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target)>:
  +output = Shape.output_shape(plan)
  +count = Shape.logical_numel(Shape.dimensions(output))
  +writes = history(count, Elementwise.KeepLeft{}, plan, shape, fill, ops, target, 0.0, Nil{})
  run_lift(shape, target, plan, planned, same, fill, ops,
    result => Laws.BroadcastToContract<target, shape,
      index => Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, index))),
      other => Replayed.value(shape, fill, ops, other), result>,
    left_fits => right_fits => output_fits =>
      Laws.BroadcastToContract{
        {==},
        Equal.cong(List<&2, U32>, Maybe<&2, List<&2, U32>>, dimensions => Some{dimensions}, Shape.dimensions(output), Shape.dimensions(target), same),
        index => inside =>
          +below = output_below(plan, target, index, same, inside)
          Equal.trans(Maybe<&2, F32>,
            Core.dynamic_read_value(Core.read_dynamic(Core.Packed{output, Core.TTensor{Replayed.storage(output, 0.0, writes), output_fits}}, index)),
            Some{Replayed.value(output, 0.0, writes, index)},
            Some{Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, U32.to_nat(index))))},
            PointwiseProof.replayed_point(output, 0.0, writes, output_fits, index, below),
            Equal.cong(F32, Maybe<&2, F32>, word => Some{word}, Replayed.value(output, 0.0, writes, index),
              Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, U32.to_nat(index)))),
              PointwiseProof.writes_value(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~PointwiseBroadcast.left_map,
                ~PointwiseBroadcast.right_map, count, Elementwise.KeepLeft{}, plan, shape, fill, ops, target, 0.0, Nil{}, output, index,
                below, Conversion.le_reflexive(count)))),
        other => inside =>
          PointwiseProof.input_point(~Shape.BroadcastPlan, ~PointwiseBroadcast.left_map, plan, count, shape, fill, ops, left_fits, other, inside)})(fits)

def Laws.broadcast_to_pointwise(shape, target, plan, planned, same, fill, ops, fits):
  broadcast_to_pointwise(shape, target, plan, planned, same, fill, ops, fits)

def Laws.broadcast_to_replayed(shape, target, plan, planned, same, fill, ops, fits, P, h):
  run_lift(shape, target, plan, planned, same, fill, ops, P, left_fits => right_fits => output_fits => h(left_fits, output_fits))(fits)
