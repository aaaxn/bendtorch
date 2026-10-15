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
import ../tensor/contracts/binary-proof.bend as BinaryProof
# Proves binary_replayed, which importing the proof above would otherwise leave open.
import ../tensor/contracts/binary-PROOF.bend as BinaryLaw
import ./reduce-to-fold-PROOF.bend as ReduceTo
import ./sum-to-size-fold-LAWS.bend as SumToSizeFold
import ./sum-to-size-fold-PROOF.bend as SumToSizeFoldProof
import ./tensor-backward.bend as Backward
import ./tensor-vjp-LAWS.bend as Laws
import ../tensor/layout.bend as Layout
import ../tensor/matmul.bend as Matrix
import ../tensor/transpose-spec.bend as Transpose
import ../tensor/dot-fold.bend as DotFold
import ../tensor/observation.bend as Values
import ../tensor/index-map.bend as IndexMap
import ../tensor/permute-values.bend as PermuteValues
import ../tensor/permute-values-records.bend as PermuteRecords
import ../tensor/permute-values-contract-LAWS.bend as PermuteLaws
import ../tensor/permute-values-contract-PROOF.bend as PermuteProof
import ../tensor/contracts/matmul-proof.bend as MatmulProof

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
    P(Elementwise.binary(operation, Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
      Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})):
  left_fits => right_fits =>
    BinaryProof.binary_replayed(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, planned, P, h)(left_fits, right_fits)

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
            Elementwise.mul(g, x))>,
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

# A permuted copy holds, at every in-bounds index, the source's word at
# permute_source: its words are `permuted_words`, which are the words of the
# copy loop's history.
def permuted_at(+input: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  +output_ops: List<&2, StorageLaws.Op>,
  +moved: {Contracts.words(Layout.output(input, axes, plan), 0.0, output_ops, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) ==
    PermuteRecords.permuted_words(input, fill, ops, axes, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) : List<&2, F32>},
  +index: U32, +inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) == True{} : Bool}) ->
  {Replayed.value(Layout.output(input, axes, plan), 0.0, output_ops, index) ==
    Replayed.value(input, fill, ops, U32.from_nat(Values.permute_source_index(input, axes, U32.to_nat(index)))) : F32}:
  +output = Layout.output(input, axes, plan)
  +total = Shape.logical_numel(Shape.dimensions(output))
  +request : PermuteValues.Request = PermuteValues.Request{input, axes}
  +history = IndexMap.writes(~PermuteValues.Request, ~PermuteValues.source_of, total, request, input, fill, ops)
  Equal.trans(F32, Replayed.value(output, 0.0, output_ops, index), Replayed.value(output, 0.0, history, index),
    Replayed.value(input, fill, ops, U32.from_nat(Values.permute_source_index(input, axes, U32.to_nat(index)))),
    SumToSizeFoldProof.agree_at(output, 0.0, output_ops, history,
      Equal.trans(List<&2, F32>, Contracts.words(output, 0.0, output_ops, total), PermuteRecords.permuted_words(input, fill, ops, axes, total),
        Contracts.words(output, 0.0, history, total), moved,
        Equal.sym(List<&2, F32>, Contracts.words(output, 0.0, history, total), PermuteRecords.permuted_words(input, fill, ops, axes, total),
          PermuteValues.words_moved(input, fill, ops, axes, plan, total, Conversion.le_reflexive(total)))),
      index, inside),
    PointwiseProof.writes_value(~Unit, ~IndexMap.keep, ~PermuteValues.Request, ~PermuteValues.source_of, ~PermuteValues.source_of, total, Unit{},
      request, input, fill, ops, input, fill, ops, output, index, inside, Conversion.le_reflexive(total)))

def factors_snoc(+before: List<&2, DotFold.Factors>, +after: List<&2, DotFold.Factors>, +x: F32, +y: F32, +u: F32, +v: F32,
  rest: {before == after : List<&2, DotFold.Factors>}, xu: {x == u : F32}, yv: {y == v : F32}) ->
  {List.append(&2, DotFold.Factors, before, [DotFold.Factors{x, y}]) ==
    List.append(&2, DotFold.Factors, after, [DotFold.Factors{u, v}]) : List<&2, DotFold.Factors>}:
  %rest : {List.append(&2, DotFold.Factors, before, [DotFold.Factors{x, y}]) ==
    List.append(&2, DotFold.Factors, _, [DotFold.Factors{u, v}]) : List<&2, DotFold.Factors>}
  %xu : {List.append(&2, DotFold.Factors, before, [DotFold.Factors{x, y}]) ==
    List.append(&2, DotFold.Factors, before, [DotFold.Factors{_, v}]) : List<&2, DotFold.Factors>}
  %yv : {List.append(&2, DotFold.Factors, before, [DotFold.Factors{x, y}]) ==
    List.append(&2, DotFold.Factors, before, [DotFold.Factors{x, _}]) : List<&2, DotFold.Factors>}
  {==}

# The pairs the first product g @ y^T reads, on the copy of g and the
# transposed copy of y, are the pairs right_transposed_factors names on g and
# y themselves.
def first_factors(+row: Nat, +column: Nat, +inner: U32, +right_inner: U32, +columns: U32,
  +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  +plan: Layout.Plan<right, Laws.swapped(right)>, +output_ops: List<&2, StorageLaws.Op>,
  +moved: {Contracts.words(Laws.transposed(right, plan), 0.0, output_ops, Shape.logical_numel(Shape.dimensions(Laws.transposed(right, plan)))) ==
    PermuteRecords.permuted_words(right, right_fill, right_ops, Laws.swapped(right),
      Shape.logical_numel(Shape.dimensions(Laws.transposed(right, plan)))) : List<&2, F32>},
  +re: {Shape.dimensions(Laws.transposed(right, plan)) == [right_inner, columns] : List<&2, U32>},
  +compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  +column_below: {Nat.is_lt(column, U32.to_nat(columns)) == True{} : Bool},
  +count: Nat, +room: {Nat.is_le(count, U32.to_nat(inner)) == True{} : Bool}) ->
  {Contracts.factors(row, column, U32.to_nat(inner), U32.to_nat(columns), shape, fill, Replayed.snoc(ops, StorageLaws.Second{}),
    Laws.transposed(right, plan), 0.0, output_ops, count) ==
    Laws.right_transposed_factors(row, column, U32.to_nat(inner), U32.to_nat(columns), shape, fill, ops,
      right, right_fill, right_ops, Laws.swapped(right), count) : List<&2, DotFold.Factors>}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +inner_count = U32.to_nat(inner)
      +column_count = U32.to_nat(columns)
      +axes = Laws.swapped(right)
      +output = Laws.transposed(right, plan)
      +position = Nat.add(Nat.mul(previous, column_count), column)
      +below = MatmulProof.lt_bound(previous, inner_count, U32.to_nat(right_inner), compatible,
        Conversion.le_successor_lt(previous, inner_count, room))
      +bound = MatmulProof.cell_below(output, right_inner, columns, re, previous, below, column, column_below)
      +cell = U32.from_nat(Nat.add(Nat.mul(row, inner_count), previous))
      factors_snoc(
        Contracts.factors(row, column, inner_count, column_count, shape, fill, Replayed.snoc(ops, StorageLaws.Second{}), output, 0.0, output_ops, previous),
        Laws.right_transposed_factors(row, column, inner_count, column_count, shape, fill, ops, right, right_fill, right_ops, axes, previous),
        Replayed.value(shape, fill, Replayed.snoc(ops, StorageLaws.Second{}), cell),
        Replayed.value(output, 0.0, output_ops, U32.from_nat(position)),
        Replayed.value(shape, fill, ops, cell),
        Replayed.value(right, right_fill, right_ops, U32.from_nat(Values.permute_source_index(right, axes, position))),
        first_factors(row, column, inner, right_inner, columns, shape, fill, ops, right, right_fill, right_ops, plan, output_ops, moved,
          re, compatible, column_below, previous, Conversion.le_predecessor(previous, inner_count, room)),
        SumToSizeFoldProof.half_keeps(shape, fill, ops, StorageLaws.Second{}, cell, {==}),
        Equal.trans(F32, Replayed.value(output, 0.0, output_ops, U32.from_nat(position)),
          Replayed.value(right, right_fill, right_ops, U32.from_nat(Values.permute_source_index(right, axes, U32.to_nat(U32.from_nat(position))))),
          Replayed.value(right, right_fill, right_ops, U32.from_nat(Values.permute_source_index(right, axes, position))),
          permuted_at(right, right_fill, right_ops, axes, plan, output_ops, moved, U32.from_nat(position),
            PointwiseProof.word_inside(position, output, bound)),
          Equal.cong(Nat, F32, number => Replayed.value(right, right_fill, right_ops, U32.from_nat(Values.permute_source_index(right, axes, number))),
            U32.to_nat(U32.from_nat(position)), position, PointwiseProof.word_exact(position, output, bound))))

# The pairs the second product x^T @ g reads, on the transposed copy of x and
# the copy of g's first half, are the pairs left_transposed_factors names on x
# and g themselves.
def second_factors(+row: Nat, +column: Nat, +rows: U32, +inner: U32, +columns: U32,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +plan: Layout.Plan<left, Laws.swapped(left)>, +output_ops: List<&2, StorageLaws.Op>,
  +moved: {Contracts.words(Laws.transposed(left, plan), 0.0, output_ops, Shape.logical_numel(Shape.dimensions(Laws.transposed(left, plan)))) ==
    PermuteRecords.permuted_words(left, left_fill, left_ops, Laws.swapped(left),
      Shape.logical_numel(Shape.dimensions(Laws.transposed(left, plan)))) : List<&2, F32>},
  +le: {Shape.dimensions(Laws.transposed(left, plan)) == [rows, inner] : List<&2, U32>},
  +row_below: {Nat.is_lt(row, U32.to_nat(rows)) == True{} : Bool},
  +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +count: Nat, +room: {Nat.is_le(count, U32.to_nat(inner)) == True{} : Bool}) ->
  {Contracts.factors(row, column, U32.to_nat(inner), U32.to_nat(columns), Laws.transposed(left, plan), 0.0, output_ops,
    shape, fill, Replayed.snoc(Replayed.snoc(ops, StorageLaws.First{}), StorageLaws.Second{}), count) ==
    Laws.left_transposed_factors(row, column, U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, Laws.swapped(left),
      shape, fill, ops, count) : List<&2, DotFold.Factors>}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +inner_count = U32.to_nat(inner)
      +column_count = U32.to_nat(columns)
      +axes = Laws.swapped(left)
      +output = Laws.transposed(left, plan)
      +position = Nat.add(Nat.mul(row, inner_count), previous)
      +bound = MatmulProof.cell_below(output, rows, inner, le, row, row_below, previous,
        Conversion.le_successor_lt(previous, inner_count, room))
      +cell = U32.from_nat(Nat.add(Nat.mul(previous, column_count), column))
      +front = Replayed.snoc(ops, StorageLaws.First{})
      factors_snoc(
        Contracts.factors(row, column, inner_count, column_count, output, 0.0, output_ops, shape, fill, Replayed.snoc(front, StorageLaws.Second{}), previous),
        Laws.left_transposed_factors(row, column, inner_count, column_count, left, left_fill, left_ops, axes, shape, fill, ops, previous),
        Replayed.value(output, 0.0, output_ops, U32.from_nat(position)),
        Replayed.value(shape, fill, Replayed.snoc(front, StorageLaws.Second{}), cell),
        Replayed.value(left, left_fill, left_ops, U32.from_nat(Values.permute_source_index(left, axes, position))),
        Replayed.value(shape, fill, ops, cell),
        second_factors(row, column, rows, inner, columns, left, left_fill, left_ops, plan, output_ops, moved, le, row_below, shape, fill, ops,
          previous, Conversion.le_predecessor(previous, inner_count, room)),
        Equal.trans(F32, Replayed.value(output, 0.0, output_ops, U32.from_nat(position)),
          Replayed.value(left, left_fill, left_ops, U32.from_nat(Values.permute_source_index(left, axes, U32.to_nat(U32.from_nat(position))))),
          Replayed.value(left, left_fill, left_ops, U32.from_nat(Values.permute_source_index(left, axes, position))),
          permuted_at(left, left_fill, left_ops, axes, plan, output_ops, moved, U32.from_nat(position),
            PointwiseProof.word_inside(position, output, bound)),
          Equal.cong(Nat, F32, number => Replayed.value(left, left_fill, left_ops, U32.from_nat(Values.permute_source_index(left, axes, number))),
            U32.to_nat(U32.from_nat(position)), position, PointwiseProof.word_exact(position, output, bound))),
        Equal.trans(F32, Replayed.value(shape, fill, Replayed.snoc(front, StorageLaws.Second{}), cell), Replayed.value(shape, fill, front, cell),
          Replayed.value(shape, fill, ops, cell),
          SumToSizeFoldProof.half_keeps(shape, fill, front, StorageLaws.Second{}, cell, {==}),
          SumToSizeFoldProof.half_keeps(shape, fill, ops, StorageLaws.First{}, cell, {==})))

# The record the Matmul VJP law concludes, for the plans the law names.
def MatmulGoal(+a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, +shape: Shape.Shape,
  +right_plan: Layout.Plan<right, Laws.swapped(right)>, +first: Matrix.Plan<shape, Laws.transposed(right, right_plan)>,
  +left_plan: Layout.Plan<left, Laws.swapped(left)>, +second: Matrix.Plan<Laws.transposed(left, left_plan), shape>,
  +fill: F32, +ops: List<&2, StorageLaws.Op>, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, pass: Backward.Pass) -> Type:
  +flipped = Laws.transposed(right, right_plan)
  +turned = Laws.transposed(left, left_plan)
  Laws.MatmulPairContract<a, left, Contracts.matrix_rows(shape, flipped, first), Contracts.matrix_columns(shape, flipped, first),
    row => column => DotFold.dot_fold(Laws.right_transposed_factors(row, column,
      Contracts.matrix_inner(shape, flipped, first), Contracts.matrix_columns(shape, flipped, first),
      shape, fill, ops, right, right_fill, right_ops, Laws.swapped(right), Contracts.matrix_inner(shape, flipped, first)), 0.0),
    b, right, Contracts.matrix_rows(turned, shape, second), Contracts.matrix_columns(turned, shape, second),
    row => column => DotFold.dot_fold(Laws.left_transposed_factors(row, column,
      Contracts.matrix_inner(turned, shape, second), Contracts.matrix_columns(turned, shape, second),
      left, left_fill, left_ops, Laws.swapped(left), shape, fill, ops, Contracts.matrix_inner(turned, shape, second)), 0.0),
    pass>

# One product's cell: the contribution reads the replayed product, and the
# product's cell is the ordered total over the copies, which first_factors or
# second_factors turns into the stated pairs.
def product_cell(+output: Shape.Shape, +rows: U32, +columns: U32, +oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
  +output_ops: List<&2, StorageLaws.Op>,
  -output_fits: {Lifecycle.logical_length(Replayed.storage(output, 0.0, output_ops)) == Shape.logical_numel(Shape.dimensions(output)) : Nat},
  +row: Nat, +row_below: {Nat.is_lt(row, U32.to_nat(rows)) == True{} : Bool},
  +column: Nat, +column_below: {Nat.is_lt(column, U32.to_nat(columns)) == True{} : Bool},
  +read: List<&2, DotFold.Factors>, +stated: List<&2, DotFold.Factors>,
  total: {Replayed.value(output, 0.0, output_ops, U32.from_nat(Nat.add(Nat.mul(row, U32.to_nat(columns)), column))) ==
    DotFold.dot_fold(read, 0.0) : F32},
  same: {read == stated : List<&2, DotFold.Factors>}) ->
  {Core.dynamic_read_value(Core.read_dynamic(Core.Packed{output, Core.TTensor{Replayed.storage(output, 0.0, output_ops), output_fits}},
    U32.from_nat(Nat.add(Nat.mul(row, U32.to_nat(columns)), column)))) == Some{DotFold.dot_fold(stated, 0.0)} : Maybe<&2, F32>}:
  +position = Nat.add(Nat.mul(row, U32.to_nat(columns)), column)
  Equal.trans(Maybe<&2, F32>,
    Core.dynamic_read_value(Core.read_dynamic(Core.Packed{output, Core.TTensor{Replayed.storage(output, 0.0, output_ops), output_fits}},
      U32.from_nat(position))),
    Some{Replayed.value(output, 0.0, output_ops, U32.from_nat(position))}, Some{DotFold.dot_fold(stated, 0.0)},
    PointwiseProof.replayed_point(output, 0.0, output_ops, output_fits, U32.from_nat(position),
      PointwiseProof.word_inside(position, output, MatmulProof.cell_below(output, rows, columns, oe, row, row_below, column, column_below))),
    Equal.cong(F32, Maybe<&2, F32>, word => Some{word}, Replayed.value(output, 0.0, output_ops, U32.from_nat(position)),
      DotFold.dot_fold(stated, 0.0),
      Equal.trans(F32, Replayed.value(output, 0.0, output_ops, U32.from_nat(position)), DotFold.dot_fold(read, 0.0),
        DotFold.dot_fold(stated, 0.0), total,
        Equal.cong(List<&2, DotFold.Factors>, F32, list => DotFold.dot_fold(list, 0.0), read, stated, same))))

def facts_left(-left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape, -rows: U32, -inner: U32, -right_inner: U32, -columns: U32,
  facts: MatmulProof.MatrixFacts<left, right, output, rows, inner, right_inner, columns>) -> {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}:
  MatmulProof.MatrixFacts{le, re, oe, compatible} = facts
  le

def facts_right(-left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape, -rows: U32, -inner: U32, -right_inner: U32, -columns: U32,
  facts: MatmulProof.MatrixFacts<left, right, output, rows, inner, right_inner, columns>) ->
  {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>}:
  MatmulProof.MatrixFacts{le, re, oe, compatible} = facts
  re

def facts_compatible(-left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape, -rows: U32, -inner: U32, -right_inner: U32, -columns: U32,
  facts: MatmulProof.MatrixFacts<left, right, output, rows, inner, right_inner, columns>) -> {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat}:
  MatmulProof.MatrixFacts{le, re, oe, compatible} = facts
  compatible

def matmul_values(+a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, +shape: Shape.Shape,
  +right_inside: {True{} == Transpose.valid(right, 0, 1) : Bool},
  +right_plan: Layout.Plan<right, Laws.swapped(right)>,
  +right_laid: {Layout.permutation(right, Laws.swapped(right)) == Done{right_plan} : Result<&2, &2, Layout.Error, Layout.Plan<right, Laws.swapped(right)>>},
  +output: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(shape) == [rows, inner] : List<&2, U32>},
  -re: {Shape.dimensions(Laws.transposed(right, right_plan)) == [right_inner, columns] : List<&2, U32>},
  -oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
  -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  +first_planned: {Matrix.plan(shape, Laws.transposed(right, right_plan)) ==
    Done{Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible}} :
    Result<&2, &2, Matrix.Error, Matrix.Plan<shape, Laws.transposed(right, right_plan)>>},
  +first_same: {Shape.dimensions(output) == Shape.dimensions(left) : List<&2, U32>},
  +left_inside: {True{} == Transpose.valid(left, 0, 1) : Bool},
  +left_plan: Layout.Plan<left, Laws.swapped(left)>,
  +left_laid: {Layout.permutation(left, Laws.swapped(left)) == Done{left_plan} : Result<&2, &2, Layout.Error, Layout.Plan<left, Laws.swapped(left)>>},
  +back: Shape.Shape, +back_rows: U32, +back_inner: U32, +back_right_inner: U32, +back_columns: U32,
  -back_le: {Shape.dimensions(Laws.transposed(left, left_plan)) == [back_rows, back_inner] : List<&2, U32>},
  -back_re: {Shape.dimensions(shape) == [back_right_inner, back_columns] : List<&2, U32>},
  -back_oe: {Shape.dimensions(back) == [back_rows, back_columns] : List<&2, U32>},
  -back_compatible: {U32.to_nat(back_inner) == U32.to_nat(back_right_inner) : Nat},
  +second_planned: {Matrix.plan(Laws.transposed(left, left_plan), shape) ==
    Done{Matrix.Plan{back, back_rows, back_inner, back_right_inner, back_columns, back_le, back_re, back_oe, back_compatible}} :
    Result<&2, &2, Matrix.Error, Matrix.Plan<Laws.transposed(left, left_plan), shape>>},
  +second_same: {Shape.dimensions(back) == Shape.dimensions(right) : List<&2, U32>},
  +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat}) ->
  MatmulGoal(a, b, left, right, shape, right_plan, Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible},
    left_plan, Matrix.Plan{back, back_rows, back_inner, back_right_inner, back_columns, back_le, back_re, back_oe, back_compatible},
    fill, ops, left_fill, left_ops, right_fill, right_ops,
    Backward.vjp(Backward.Matmul{a, b, left, right},
      Backward.Operands{Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
        Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}}},
      Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}})):
  +flipped = Laws.transposed(right, right_plan)
  +turned = Laws.transposed(left, left_plan)
  +first : Matrix.Plan<shape, Laws.transposed(right, right_plan)> = Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible}
  +second : Matrix.Plan<Laws.transposed(left, left_plan), shape> =
    Matrix.Plan{back, back_rows, back_inner, back_right_inner, back_columns, back_le, back_re, back_oe, back_compatible}
  +front_facts : MatmulProof.MatrixFacts<shape, flipped, output, rows, inner, right_inner, columns> =
    MatmulProof.plan_facts(shape, flipped, first, first_planned)
  +back_facts : MatmulProof.MatrixFacts<turned, shape, back, back_rows, back_inner, back_right_inner, back_columns> =
    MatmulProof.plan_facts(turned, shape, second, second_planned)
  +front_re = facts_right(shape, flipped, output, rows, inner, right_inner, columns, front_facts)
  +front_oe = MatmulProof.output_dimensions(shape, flipped, output, rows, inner, right_inner, columns, front_facts)
  +front_compatible = facts_compatible(shape, flipped, output, rows, inner, right_inner, columns, front_facts)
  +turned_le = facts_left(turned, shape, back, back_rows, back_inner, back_right_inner, back_columns, back_facts)
  +turned_oe = MatmulProof.output_dimensions(turned, shape, back, back_rows, back_inner, back_right_inner, back_columns, back_facts)
  -Goal : Backward.Pass -> Type = pass => MatmulGoal(a, b, left, right, shape, right_plan, first, left_plan, second,
    fill, ops, left_fill, left_ops, right_fill, right_ops, pass)
  -x = {Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}} : Core.Dynamic}
  -y = {Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}} : Core.Dynamic}
  +front = Replayed.snoc(ops, StorageLaws.First{})
  SumToSizeFoldProof.clone_lift(shape, fill, ops, fits,
    pair => Goal(Backward.matmul_first(a, b, left, right, x, y, Core.clone_dynamic_result(shape, pair))),
    g_fits => copy_fits =>
      -g = {Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, front), g_fits}} : Core.Dynamic}
      -copy = {Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, Replayed.snoc(ops, StorageLaws.Second{})), copy_fits}} : Core.Dynamic}
      PermuteLaws.transpose_replayed(right, right_fill, right_ops, right_fits, 0, 1, right_inside, right_plan, right_laid,
        pair => Goal(Backward.matmul_second(b, right, x, g, Backward.left_product(a, left, copy, pair))),
        y_kept => y_words => y_fits => flipped_ops => flipped_moved => flipped_fits =>
          -y_back = {Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, y_kept), y_fits}} : Core.Dynamic}
          Contracts.matmul_replayed(shape, flipped, first, first_planned, fill, Replayed.snoc(ops, StorageLaws.Second{}), copy_fits,
            0.0, flipped_ops, flipped_fits,
            result => Goal(Backward.matmul_second(b, right, x, g, (y_back, Backward.multiplied(a, left, result)))),
            copy_kept => copy_words => copy_kept_fits => flipped_kept => flipped_words => flipped_kept_fits => output_ops => front_cells => output_fits =>
              -delivered = {Backward.Contribution{a, left, Core.Packed{output, Core.TTensor{Replayed.storage(output, 0.0, output_ops), output_fits}}} :
                Backward.Contribution}
              SumToSizeFoldProof.clone_lift(shape, fill, front, g_fits,
                pair => Goal(Backward.matmul_third(b, right, Done{delivered}, y_back, x, Core.clone_dynamic_result(shape, pair))),
                last_fits => again_fits =>
                  -last = {Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, Replayed.snoc(front, StorageLaws.First{})), last_fits}} : Core.Dynamic}
                  -again = {Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, Replayed.snoc(front, StorageLaws.Second{})), again_fits}} :
                    Core.Dynamic}
                  PermuteLaws.transpose_replayed(left, left_fill, left_ops, left_fits, 0, 1, left_inside, left_plan, left_laid,
                    pair => Goal(Backward.matmul_fourth(Done{delivered}, y_back, last, Backward.right_product(b, right, again, pair))),
                    x_kept => x_words => x_fits => turned_ops => turned_moved => turned_fits =>
                      -x_back = {Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, x_kept), x_fits}} : Core.Dynamic}
                      Contracts.matmul_replayed(turned, shape, second, second_planned, 0.0, turned_ops, turned_fits,
                        fill, Replayed.snoc(front, StorageLaws.Second{}), again_fits,
                        result => Goal(Backward.matmul_fourth(Done{delivered}, y_back, last, (x_back, Backward.multiplied(b, right, result)))),
                        turned_kept => turned_words => turned_kept_fits => again_kept => again_words => again_kept_fits =>
                          back_ops => back_cells => back_fits =>
                          Laws.MatmulPairContract{
                            {==},
                            Equal.cong(List<&2, U32>, Maybe<&2, List<&2, U32>>, dimensions => Some{dimensions},
                              Shape.dimensions(output), Shape.dimensions(left), first_same),
                            row => row_below => column => column_below =>
                              +row_below = row_below
                              +column_below = column_below
                              product_cell(output, rows, columns, front_oe, output_ops, output_fits, row, row_below, column, column_below,
                                Contracts.factors(row, column, U32.to_nat(inner), U32.to_nat(columns), shape, fill,
                                  Replayed.snoc(ops, StorageLaws.Second{}), flipped, 0.0, flipped_ops, U32.to_nat(inner)),
                                Laws.right_transposed_factors(row, column, U32.to_nat(inner), U32.to_nat(columns), shape, fill, ops,
                                  right, right_fill, right_ops, Laws.swapped(right), U32.to_nat(inner)),
                                front_cells(row, row_below, column, column_below),
                                first_factors(row, column, inner, right_inner, columns, shape, fill, ops, right, right_fill, right_ops,
                                  right_plan, flipped_ops, flipped_moved, front_re, front_compatible, column_below,
                                  U32.to_nat(inner), Conversion.le_reflexive(U32.to_nat(inner)))),
                            {==},
                            Equal.cong(List<&2, U32>, Maybe<&2, List<&2, U32>>, dimensions => Some{dimensions},
                              Shape.dimensions(back), Shape.dimensions(right), second_same),
                            row => row_below => column => column_below =>
                              +row_below = row_below
                              +column_below = column_below
                              product_cell(back, back_rows, back_columns, turned_oe, back_ops, back_fits, row, row_below, column, column_below,
                                Contracts.factors(row, column, U32.to_nat(back_inner), U32.to_nat(back_columns), turned, 0.0, turned_ops,
                                  shape, fill, Replayed.snoc(front, StorageLaws.Second{}), U32.to_nat(back_inner)),
                                Laws.left_transposed_factors(row, column, U32.to_nat(back_inner), U32.to_nat(back_columns),
                                  left, left_fill, left_ops, Laws.swapped(left), shape, fill, ops, U32.to_nat(back_inner)),
                                back_cells(row, row_below, column, column_below),
                                second_factors(row, column, back_rows, back_inner, back_columns, left, left_fill, left_ops, left_plan,
                                  turned_ops, turned_moved, turned_le, row_below, shape, fill, ops,
                                  U32.to_nat(back_inner), Conversion.le_reflexive(U32.to_nat(back_inner))))}))))))

def Laws.tensor_matmul_vjp_values(a, b, left, right, shape, right_inside, right_plan, right_laid, first, first_planned, first_same,
  left_inside, left_plan, left_laid, second, second_planned, second_same, fill, ops, fits, left_fill, left_ops, left_fits,
  right_fill, right_ops, right_fits):
  match first:
    case Matrix.Plan{+output, +rows, +inner, +right_inner, +columns, le, re, oe, compatible}:
      match second:
        case Matrix.Plan{+back, +back_rows, +back_inner, +back_right_inner, +back_columns, back_le, back_re, back_oe, back_compatible}:
          matmul_values(a, b, left, right, shape, right_inside, right_plan, right_laid,
            output, rows, inner, right_inner, columns, le, re, oe, compatible, first_planned, first_same,
            left_inside, left_plan, left_laid,
            back, back_rows, back_inner, back_right_inner, back_columns, back_le, back_re, back_oe, back_compatible,
            second_planned, second_same, fill, ops, fits, left_fill, left_ops, left_fits, right_fill, right_ops, right_fits)
