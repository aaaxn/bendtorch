import Base
import ../shape/core.bend as Shape
import ../storage/LAWS.bend as StorageLaws
import ../tensor/core.bend as Core
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/elementwise.bend as Elementwise
import ../tensor/reduction.bend as Reduction
import ../tensor/word-conversion.bend as Conversion
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/pointwise.bend as Pointwise
import ../tensor/contracts/pointwise-proof.bend as PointwiseProof
import ../tensor/contracts/broadcast.bend as PointwiseBroadcast
import ../tensor/contracts/LAWS.bend as Contracts
import ../tensor/contracts/words-proof.bend as Words
import ./reduce-to-fold-PROOF.bend as ReduceTo
import ./broadcast-pointwise-PROOF.bend as BroadcastPointwiseProof
import ./sum-to-size-fold-LAWS.bend as SumToSizeFold
import ./sum-to-size-fold-PROOF.bend as SumToSizeFoldProof
import ./tensor-backward.bend as Backward
import ./tensor-vjp-LAWS.bend as Laws

def cell(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +index: Nat) -> F32:
  Pointwise.cell(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~PointwiseBroadcast.left_map,
    ~PointwiseBroadcast.right_map, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, index)

def cells_snoc(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +previous: Nat,
  +before: List<&2, F32>, +word: F32, +other: F32,
  rest: {before == Laws.cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous) : List<&2, F32>},
  last: {word == other : F32}) ->
  {List.append(&2, F32, before, [word]) ==
    List.append(&2, F32, Laws.cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous), [other]) : List<&2, F32>}:
  Equal.trans(List<&2, F32>, List.append(&2, F32, before, [word]),
    List.append(&2, F32, Laws.cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous), [word]),
    List.append(&2, F32, Laws.cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous), [other]),
    Equal.cong(List<&2, F32>, List<&2, F32>, values => List.append(&2, F32, values, [word]),
      before, Laws.cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous), rest),
    Equal.cong(F32, List<&2, F32>,
      value => List.append(&2, F32, Laws.cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous), [value]),
      word, other, last))

# The words of the pointwise family's output history are `cells`.
def words_cells(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat,
  +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {Contracts.words(Shape.output_shape(plan), 0.0,
    BroadcastPointwiseProof.history(Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan,
      left, left_fill, left_ops, right, right_fill, right_ops), count) ==
    Laws.cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count) : List<&2, F32>}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +output = Shape.output_shape(plan)
      +total = Shape.logical_numel(Shape.dimensions(output))
      +history = BroadcastPointwiseProof.history(total, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops)
      +below = Conversion.le_successor_lt(previous, total, room)
      +position = U32.from_nat(previous)
      cells_snoc(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous,
        Contracts.words(output, 0.0, history, previous), Replayed.value(output, 0.0, history, position),
        cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous),
        words_cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous,
          Conversion.le_predecessor(previous, total, room)),
        Equal.trans(F32, Replayed.value(output, 0.0, history, position),
          cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(position)),
          cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous),
          PointwiseProof.writes_value(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~PointwiseBroadcast.left_map,
            ~PointwiseBroadcast.right_map, total, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, output, position,
            PointwiseProof.word_inside(previous, output, below), Conversion.le_reflexive(total)),
          Equal.cong(Nat, F32, number => cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, number),
            U32.to_nat(position), previous, PointwiseProof.word_exact(previous, output, below))))

# `cells` reads the left input only at left_map(plan, k), which is in bounds,
# so a left history that lists the same words gives the same cells.
def cells_left(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: Shape.Shape, +left_fill: F32, +kept: List<&2, StorageLaws.Op>, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  +left_shape: {Shape.left_shape(plan) == left : Shape.Shape},
  +same: {Contracts.words(left, left_fill, kept, Shape.logical_numel(Shape.dimensions(left))) ==
    Contracts.words(left, left_fill, left_ops, Shape.logical_numel(Shape.dimensions(left))) : List<&2, F32>},
  +count: Nat, +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {Laws.cells(operation, plan, left, left_fill, kept, right, right_fill, right_ops, count) ==
    Laws.cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count) : List<&2, F32>}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +total = Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))
      +below = Conversion.le_successor_lt(previous, total, room)
      +index = U32.from_nat(PointwiseBroadcast.left_map(plan, previous))
      +right_word = Replayed.value(right, right_fill, right_ops, U32.from_nat(PointwiseBroadcast.right_map(plan, previous)))
      cells_snoc(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous,
        Laws.cells(operation, plan, left, left_fill, kept, right, right_fill, right_ops, previous),
        cell(operation, plan, left, left_fill, kept, right, right_fill, right_ops, previous),
        cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous),
        cells_left(operation, plan, left, left_fill, kept, left_ops, right, right_fill, right_ops, left_shape, same, previous,
          Conversion.le_predecessor(previous, total, room)),
        Equal.cong(F32, F32, word => Elementwise.scalar(operation, word, right_word),
          Replayed.value(left, left_fill, kept, index), Replayed.value(left, left_fill, left_ops, index),
          SumToSizeFoldProof.agree_at(left, left_fill, kept, left_ops, same, index,
            PointwiseProof.word_inside(PointwiseBroadcast.left_map(plan, previous), left,
              ReduceTo.target_below(plan, left, left_shape, previous, below)))))

def product_replayed(+operation: Elementwise.Operation, +left: Shape.Shape, +right: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(left, right) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -P: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type,
  h: Laws.ProductWith(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, P)) ->
  @-left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat} ->
  @-right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat} ->
    P(Backward.product(operation, Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
      Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})):
  +output = Shape.output_shape(plan)
  +count = Shape.logical_numel(Shape.dimensions(output))
  left_fits => right_fits =>
    -left_input = {Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}} : Core.Dynamic}
    -right_input = {Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}} : Core.Dynamic}
    %Equal.sym(Result<&2, &2, Shape.Error, Shape.BroadcastPlan>, Shape.broadcast_checked(left, right), Done{plan}, planned) :
      P(Backward.product_planned(operation, left_input, right_input, _))
    Replayed.new_lift(output, 0.0, result => P(Elementwise.allocated(operation, plan, left_input, right_input, result)),
      output_fits => BroadcastPointwiseProof.start_cast(Shape.elements(output), operation, plan, left, left_fill, left_ops,
        right, right_fill, right_ops, P,
        BroadcastPointwiseProof.loop_lift(Shape.elements(output), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 0n,
          Shape.elements_exact(output), ReduceTo.plan_left(left, right, plan, planned), ReduceTo.plan_right(left, right, plan, planned), P,
          left_after => right_after => output_after =>
            h(List.append(&2, StorageLaws.Op, left_ops, Pointwise.reads(~Shape.BroadcastPlan, ~PointwiseBroadcast.left_map, plan, count)),
              Words.words_reads(~Shape.BroadcastPlan, ~PointwiseBroadcast.left_map, plan, count, left, left_fill, left_ops,
                Shape.logical_numel(Shape.dimensions(left))),
              left_after,
              List.append(&2, StorageLaws.Op, right_ops, Pointwise.reads(~Shape.BroadcastPlan, ~PointwiseBroadcast.right_map, plan, count)),
              Words.words_reads(~Shape.BroadcastPlan, ~PointwiseBroadcast.right_map, plan, count, right, right_fill, right_ops,
                Shape.logical_numel(Shape.dimensions(right))),
              right_after,
              BroadcastPointwiseProof.history(count, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops),
              words_cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count, Conversion.le_reflexive(count)),
              output_after)))(left_fits, right_fits, output_fits))

def Laws.tensor_product_replayed(operation, left, right, plan, planned, left_fill, left_ops, left_fits, right_fill, right_ops, right_fits, P, h):
  product_replayed(operation, left, right, plan, planned, left_fill, left_ops, right_fill, right_ops, P, h)(left_fits, right_fits)

# Two delivered contributions, each a replayed tensor with its target's
# dimensions whose cells hold the stated words, meet the pair contract.
def pair_contract(+a: Nat, +left: Shape.Shape, -left_cell: U32 -> Maybe<&2, F32>,
  +b: Nat, +right: Shape.Shape, -right_cell: U32 -> Maybe<&2, F32>, -saved: Backward.Saved, -g: Core.Dynamic,
  +first: Shape.Shape, +first_dimensions: {Shape.dimensions(first) == Shape.dimensions(left) : List<&2, U32>},
  +first_fill: F32, +first_ops: List<&2, StorageLaws.Op>,
  first_values: @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool} ->
    {Some{Replayed.value(first, first_fill, first_ops, index)} == left_cell(index) : Maybe<&2, F32>},
  -first_fits: {Lifecycle.logical_length(Replayed.storage(first, first_fill, first_ops)) == Shape.logical_numel(Shape.dimensions(first)) : Nat},
  +second: Shape.Shape, +second_dimensions: {Shape.dimensions(second) == Shape.dimensions(right) : List<&2, U32>},
  +second_fill: F32, +second_ops: List<&2, StorageLaws.Op>,
  second_values: @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool} ->
    {Some{Replayed.value(second, second_fill, second_ops, index)} == right_cell(index) : Maybe<&2, F32>},
  -second_fits: {Lifecycle.logical_length(Replayed.storage(second, second_fill, second_ops)) == Shape.logical_numel(Shape.dimensions(second)) : Nat}) ->
  Laws.PairContract<a, left, left_cell, b, right, right_cell,
    Backward.Pass{saved, g, Done{[Backward.Contribution{a, left, Core.Packed{first, Core.TTensor{Replayed.storage(first, first_fill, first_ops), first_fits}}},
      Backward.Contribution{b, right, Core.Packed{second, Core.TTensor{Replayed.storage(second, second_fill, second_ops), second_fits}}}]}}>:
  Laws.PairContract{
    {==},
    Equal.cong(List<&2, U32>, Maybe<&2, List<&2, U32>>, dimensions => Some{dimensions}, Shape.dimensions(first), Shape.dimensions(left),
      first_dimensions),
    index => inside =>
      +inside = inside
      Equal.trans(Maybe<&2, F32>,
        Core.dynamic_read_value(Core.read_dynamic(Core.Packed{first, Core.TTensor{Replayed.storage(first, first_fill, first_ops), first_fits}}, index)),
        Some{Replayed.value(first, first_fill, first_ops, index)}, left_cell(index),
        PointwiseProof.replayed_point(first, first_fill, first_ops, first_fits, index,
          SumToSizeFoldProof.shape_below(first, left, index, first_dimensions, inside)),
        first_values(index, inside)),
    {==},
    Equal.cong(List<&2, U32>, Maybe<&2, List<&2, U32>>, dimensions => Some{dimensions}, Shape.dimensions(second), Shape.dimensions(right),
      second_dimensions),
    index => inside =>
      +inside = inside
      Equal.trans(Maybe<&2, F32>,
        Core.dynamic_read_value(Core.read_dynamic(Core.Packed{second, Core.TTensor{Replayed.storage(second, second_fill, second_ops), second_fits}}, index)),
        Some{Replayed.value(second, second_fill, second_ops, index)}, right_cell(index),
        PointwiseProof.replayed_point(second, second_fill, second_ops, second_fits, index,
          SumToSizeFoldProof.shape_below(second, right, index, second_dimensions, inside)),
        second_values(index, inside))}

def Laws.tensor_add_vjp_values(a, b, left, right, shape, left_plan, left_planned, left_same, right_plan, right_planned, right_same,
  saved, fill, ops, fits):
  +values = Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape)))
  SumToSizeFold.sum_to_size_replayed(left, shape, left_plan, left_planned, left_same, fill, ops, fits, values, {==},
    pair => Laws.PairContract<a, left, index => Laws.reduced(left, shape, left_plan, values, index),
      b, right, index => Laws.reduced(right, shape, right_plan, values, index),
      Backward.added_first(b, right, Backward.delivered_as(a, left, pair))>,
    kept => kept_words => kept_fits => first => first_dimensions => first_fill => first_ops => first_values => first_fits =>
      SumToSizeFold.sum_to_size_replayed(right, shape, right_plan, right_planned, right_same, fill, kept, kept_fits, values, kept_words,
        pair => Laws.PairContract<a, left, index => Laws.reduced(left, shape, left_plan, values, index),
          b, right, index => Laws.reduced(right, shape, right_plan, values, index),
          Backward.added_second(Done{Backward.Contribution{a, left,
            Core.Packed{first, Core.TTensor{Replayed.storage(first, first_fill, first_ops), first_fits}}}},
            Backward.delivered_as(b, right, pair))>,
        last => last_words => last_fits => second => second_dimensions => second_fill => second_ops => second_values => second_fits =>
          pair_contract(a, left, index => Laws.reduced(left, shape, left_plan, values, index),
            b, right, index => Laws.reduced(right, shape, right_plan, values, index),
            Backward.Unsaved{}, Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, last), last_fits}},
            first, first_dimensions, first_fill, first_ops, first_values, first_fits,
            second, second_dimensions, second_fill, second_ops, second_values, second_fits)))

def Laws.tensor_sub_vjp_values(a, b, left, right, shape, left_plan, left_planned, left_same, negated, negated_planned,
  right_plan, right_planned, right_same, saved, fill, ops, fits):
  +negated_planned = negated_planned
  +count = Shape.logical_numel(Shape.dimensions(shape))
  +values = Contracts.words(shape, fill, ops, count)
  +zero = Reduction.scalar_shape()
  +minus = Shape.output_shape(negated)
  +minus_count = Shape.logical_numel(Shape.dimensions(minus))
  +negatives = Laws.cells(Elementwise.NegateLeft{}, negated, shape, fill, ops, zero, 0.0, Nil{}, minus_count)
  SumToSizeFold.sum_to_size_replayed(left, shape, left_plan, left_planned, left_same, fill, ops, fits, values, {==},
    pair => Laws.PairContract<a, left, index => Laws.reduced(left, shape, left_plan, values, index),
      b, right, index => Laws.reduced(right, minus, right_plan, negatives, index),
      Backward.subtracted_first(b, right, Backward.delivered_as(a, left, pair))>,
    kept => kept_words => kept_fits => first => first_dimensions => first_fill => first_ops => first_values => first_fits =>
      -delivered = {Done{Backward.Contribution{a, left, Core.Packed{first, Core.TTensor{Replayed.storage(first, first_fill, first_ops), first_fits}}}} :
        Result<&2, &1, Backward.Error, Backward.Contribution>}
      Replayed.new_lift(zero, 0.0,
        result => Laws.PairContract<a, left, index => Laws.reduced(left, shape, left_plan, values, index),
          b, right, index => Laws.reduced(right, minus, right_plan, negatives, index),
          Backward.subtracted_second(delivered, b, right,
            Backward.negated(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, kept), kept_fits}}, result))>,
        zero_fits =>
          product_replayed(Elementwise.NegateLeft{}, shape, zero, negated, negated_planned, fill, kept, 0.0, Nil{},
            pair => Laws.PairContract<a, left, index => Laws.reduced(left, shape, left_plan, values, index),
              b, right, index => Laws.reduced(right, minus, right_plan, negatives, index),
              Backward.subtracted_second(delivered, b, right, Backward.negated_pair(pair))>,
            last => last_words => last_fits => zero_kept => zero_words => zero_kept_fits => product_ops => product_values => product_fits =>
              SumToSizeFold.sum_to_size_replayed(right, minus, right_plan, right_planned, right_same, 0.0, product_ops, product_fits, negatives,
                Equal.trans(List<&2, F32>, Contracts.words(minus, 0.0, product_ops, minus_count),
                  Laws.cells(Elementwise.NegateLeft{}, negated, shape, fill, kept, zero, 0.0, Nil{}, minus_count), negatives,
                  product_values,
                  cells_left(Elementwise.NegateLeft{}, negated, shape, fill, kept, ops, zero, 0.0, Nil{},
                    ReduceTo.plan_left(shape, zero, negated, negated_planned), kept_words, minus_count, Conversion.le_reflexive(minus_count))),
                pair => Laws.PairContract<a, left, index => Laws.reduced(left, shape, left_plan, values, index),
                  b, right, index => Laws.reduced(right, minus, right_plan, negatives, index),
                  Backward.Pass{Backward.Unsaved{}, Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, last), last_fits}},
                    Backward.both(delivered, Backward.contribution_of(Backward.delivered_as(b, right, pair)))}>,
                reduced => reduced_words => reduced_fits => second => second_dimensions => second_fill => second_ops => second_values => second_fits =>
                  pair_contract(a, left, index => Laws.reduced(left, shape, left_plan, values, index),
                    b, right, index => Laws.reduced(right, minus, right_plan, negatives, index),
                    Backward.Unsaved{}, Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, last), last_fits}},
                    first, first_dimensions, first_fill, first_ops, first_values, first_fits,
                    second, second_dimensions, second_fill, second_ops, second_values, second_fits)))(kept_fits, zero_fits)))

def Laws.tensor_mul_vjp_values(a, b, left, right, shape, first, first_planned, left_plan, left_planned, left_same,
  second, second_planned, right_plan, right_planned, right_same, fill, ops, fits, left_fill, left_ops, left_fits,
  right_fill, right_ops, right_fits):
  +second_planned = second_planned
  +front = Shape.output_shape(first)
  +front_count = Shape.logical_numel(Shape.dimensions(front))
  +back = Shape.output_shape(second)
  +back_count = Shape.logical_numel(Shape.dimensions(back))
  +front_values = Laws.cells(Elementwise.Mul{}, first, shape, fill, ops, right, right_fill, right_ops, front_count)
  +back_values = Laws.cells(Elementwise.Mul{}, second, shape, fill, ops, left, left_fill, left_ops, back_count)
  -x = {Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}} : Core.Dynamic}
  product_replayed(Elementwise.Mul{}, shape, right, first, first_planned, fill, ops, right_fill, right_ops,
    pair => Laws.PairContract<a, left, index => Laws.reduced(left, front, left_plan, front_values, index),
      b, right, index => Laws.reduced(right, back, right_plan, back_values, index),
      Backward.multiplied_first(a, b, left, right, x, pair)>,
    kept => kept_words => kept_fits => y => y_words => y_fits => front_ops => front_words => front_fits =>
      -g = {Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, kept), kept_fits}} : Core.Dynamic}
      -y_kept = {Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, y), y_fits}} : Core.Dynamic}
      SumToSizeFold.sum_to_size_replayed(left, front, left_plan, left_planned, left_same, 0.0, front_ops, front_fits, front_values,
        front_words,
        pair => Laws.PairContract<a, left, index => Laws.reduced(left, front, left_plan, front_values, index),
          b, right, index => Laws.reduced(right, back, right_plan, back_values, index),
          Backward.multiplied_second(b, right, Backward.contribution_of(Backward.delivered_as(a, left, pair)), y_kept,
            Backward.product(Elementwise.Mul{}, g, x))>,
        reduced => reduced_words => reduced_fits => head => head_dimensions => head_fill => head_ops => head_values => head_fits =>
          -delivered = {Done{Backward.Contribution{a, left, Core.Packed{head, Core.TTensor{Replayed.storage(head, head_fill, head_ops), head_fits}}}} :
            Result<&2, &1, Backward.Error, Backward.Contribution>}
          product_replayed(Elementwise.Mul{}, shape, left, second, second_planned, fill, kept, left_fill, left_ops,
            pair => Laws.PairContract<a, left, index => Laws.reduced(left, front, left_plan, front_values, index),
              b, right, index => Laws.reduced(right, back, right_plan, back_values, index),
              Backward.multiplied_second(b, right, delivered, y_kept, pair)>,
            last => last_words => last_fits => x_kept => x_words => x_fits => back_ops => back_words => back_fits =>
              SumToSizeFold.sum_to_size_replayed(right, back, right_plan, right_planned, right_same, 0.0, back_ops, back_fits, back_values,
                Equal.trans(List<&2, F32>, Contracts.words(back, 0.0, back_ops, back_count),
                  Laws.cells(Elementwise.Mul{}, second, shape, fill, kept, left, left_fill, left_ops, back_count), back_values,
                  back_words,
                  cells_left(Elementwise.Mul{}, second, shape, fill, kept, ops, left, left_fill, left_ops,
                    ReduceTo.plan_left(shape, left, second, second_planned), kept_words, back_count, Conversion.le_reflexive(back_count))),
                pair => Laws.PairContract<a, left, index => Laws.reduced(left, front, left_plan, front_values, index),
                  b, right, index => Laws.reduced(right, back, right_plan, back_values, index),
                  Backward.Pass{Backward.Operands{Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, x_kept), x_fits}}, y_kept},
                    Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, last), last_fits}},
                    Backward.both(delivered, Backward.contribution_of(Backward.delivered_as(b, right, pair)))}>,
                tail => tail_words => tail_fits => second_out => second_dimensions => second_fill => second_ops => second_values => second_fits =>
                  pair_contract(a, left, index => Laws.reduced(left, front, left_plan, front_values, index),
                    b, right, index => Laws.reduced(right, back, right_plan, back_values, index),
                    Backward.Operands{Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, x_kept), x_fits}}, y_kept},
                    Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, last), last_fits}},
                    head, head_dimensions, head_fill, head_ops, head_values, head_fits,
                    second_out, second_dimensions, second_fill, second_ops, second_values, second_fits)))(kept_fits, left_fits)))(fits, right_fits)
