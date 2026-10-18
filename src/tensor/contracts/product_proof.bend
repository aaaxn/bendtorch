import Base
import ../../shape/core.bend as Shape
import ../../shape/nat_proofs.bend as Nats
import ../../storage/main.bend as Storage
import ../../storage/LAWS.bend as StorageLaws
import ../../storage/arith.bend as Arith
import ../core.bend as Core
import ../lifecycle.bend as Lifecycle
import ../matmul.bend as Matrix
import ../dot_fold.bend as DotFold
import ../word_conversion.bend as Conversion
import ./replayed.bend as Replayed
import ./pointwise_proof.bend as PointwiseProof
import ./fold_records.bend as Records
import ./dot_proof.bend as DotProof
import ./LAWS.bend as Laws
import ./product_parallel_proof.bend as Par
import ../read_contract.bend as Reads
import ../../storage/api.bend as Api
import ../../storage/array.bend as ArrayCore
import ./binary_storage_proof.bend as Owned
import ../elementwise.bend as Elementwise


# The parallel product of operands read in place, over replayed inputs: the
# proof of `matmul_ordered` and `matmul_replayed` for `Matrix.product`, whose
# plan names each operand's order.

type ProductFacts<-lo: Matrix.Order, -ro: Matrix.Order, -left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape,
  -rows: U32, -inner: U32, -right_inner: U32, -columns: U32> is Data:
  ProductFacts{left_dimensions: {Shape.dimensions(left) == Matrix.arranged(lo, rows, inner) : List<&2, U32>},
    right_dimensions: {Shape.dimensions(right) == Matrix.arranged(ro, columns, right_inner) : List<&2, U32>},
    output_dimensions: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
    compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat}}

def Facts(-lo: Matrix.Order, -ro: Matrix.Order, -left: Shape.Shape, -right: Shape.Shape,
  result: Result<&2, &2, Matrix.Error, Matrix.Product<lo, ro, left, right>>) -> Data:
  match result:
    case Fail{error}:
      Unit
    case Done{Matrix.Product{output, rows, inner, right_inner, columns, le, re, oe, compatible}}:
      ProductFacts<lo, ro, left, right, output, rows, inner, right_inner, columns>

def facts_output(-lo: Matrix.Order, -ro: Matrix.Order, -left: Shape.Shape, -right: Shape.Shape, +rows: U32, +inner: U32,
  +right_inner: U32, +columns: U32,
  le: {Shape.dimensions(left) == Matrix.arranged(lo, rows, inner) : List<&2, U32>},
  re: {Shape.dimensions(right) == Matrix.arranged(ro, columns, right_inner) : List<&2, U32>},
  compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  result: Shape.ConstructResult<[rows, columns]>) ->
  Facts(lo, ro, left, right, Matrix.product_output(lo, ro, left, right, rows, inner, right_inner, columns, le, re, compatible, result)):
  match result:
    case Shape.SError{error}:
      Unit{}
    case Shape.SBuilt{shape, preserved}:
      ProductFacts{le, re, preserved, compatible}

def facts_compatible(-lo: Matrix.Order, -ro: Matrix.Order, -left: Shape.Shape, -right: Shape.Shape, +rows: U32, +inner: U32,
  +right_inner: U32, +columns: U32,
  le: {Shape.dimensions(left) == Matrix.arranged(lo, rows, inner) : List<&2, U32>},
  re: {Shape.dimensions(right) == Matrix.arranged(ro, columns, right_inner) : List<&2, U32>},
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))>) ->
  Facts(lo, ro, left, right, Matrix.product_compatible(lo, ro, left, right, rows, inner, right_inner, columns, le, re, decision)):
  match decision:
    case Shape.BFalse{equal}:
      Unit{}
    case Shape.BTrue{equal}:
      facts_output(lo, ro, left, right, rows, inner, right_inner, columns, le, re,
        Lifecycle.natural_equal(U32.to_nat(inner), U32.to_nat(right_inner), equal), Shape.construct([rows, columns]))

def facts_right(-lo: Matrix.Order, -ro: Matrix.Order, -left: Shape.Shape, -right: Shape.Shape, +rows: U32, +inner: U32,
  le: {Shape.dimensions(left) == Matrix.arranged(lo, rows, inner) : List<&2, U32>},
  result: Result<&2, &2, Matrix.Error, Matrix.Split<ro, Shape.dimensions(right)>>) ->
  Facts(lo, ro, left, right, Matrix.product_right(lo, ro, left, right, rows, inner, le, result)):
  match result:
    case Fail{error}:
      Unit{}
    case Done{Matrix.Split{+columns, +right_inner, same}}:
      facts_compatible(lo, ro, left, right, rows, inner, right_inner, columns, le, same,
        Shape.bool_case(Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))))

def facts_left(-lo: Matrix.Order, +ro: Matrix.Order, -left: Shape.Shape, +right: Shape.Shape,
  result: Result<&2, &2, Matrix.Error, Matrix.Split<lo, Shape.dimensions(left)>>) ->
  Facts(lo, ro, left, right, Matrix.product_left(lo, ro, left, right, result)):
  match result:
    case Fail{error}:
      Unit{}
    case Done{Matrix.Split{+rows, +inner, same}}:
      facts_right(lo, ro, left, right, rows, inner, same, Matrix.split_dims(ro, Shape.dimensions(right)))

def plan_facts(+lo: Matrix.Order, +ro: Matrix.Order, +left: Shape.Shape, +right: Shape.Shape, +plan: Matrix.Product<lo, ro, left, right>,
  planned: {Matrix.product_plan(lo, ro, left, right) == Done{plan} : Result<&2, &2, Matrix.Error, Matrix.Product<lo, ro, left, right>>}) ->
  Facts(lo, ro, left, right, Done{plan}):
  %planned : Facts(lo, ro, left, right, _)
  facts_left(lo, ro, left, right, Matrix.split_dims(lo, Shape.dimensions(left)))

# An operand whose dimensions are arranged in its order has the extents
# `Par.Fits` names.
def fits_of(+order: Matrix.Order, +outer: U32, +inner: U32, -shape: Shape.Shape,
  same: {Shape.dimensions(shape) == Matrix.arranged(order, outer, inner) : List<&2, U32>}) ->
  Par.Fits(order, U32.to_nat(inner), U32.to_nat(outer), shape):
  match order:
    case Matrix.Kept{}:
      Equal.cong(List<&2, U32>, List<&2, Nat>, dims => Shape.naturals(dims), Shape.dimensions(shape), [outer, inner], same)
    case Matrix.Swapped{}:
      Equal.cong(List<&2, U32>, List<&2, Nat>, dims => Shape.naturals(dims), Shape.dimensions(shape), [inner, outer], same)

# The right operand's extents name the left operand's inner extent.
def inner_fits(+ro: Matrix.Order, +columns: U32, +inner: U32, +right_inner: U32, -shape: Shape.Shape,
  compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  same: {Shape.dimensions(shape) == Matrix.arranged(ro, columns, right_inner) : List<&2, U32>}) ->
  Par.Fits(ro, U32.to_nat(inner), U32.to_nat(columns), shape):
  %Equal.sym(Nat, U32.to_nat(inner), U32.to_nat(right_inner), compatible) : Par.Fits(ro, _, U32.to_nat(columns), shape)
  fits_of(ro, columns, right_inner, shape, same)

def lt_from(+a: Nat, +b: Nat, below: {Nat.is_lt(a, b) == True{} : Bool}) -> Shape.LT(a, b):
  match a b:
    case 0n 0n:
      Empty.absurd(Shape.LT(0n, 0n), Arith.clash(below))
    case 0n 1n+q:
      Unit{}
    case 1n+p 0n:
      Empty.absurd(Shape.LT(1n+p, 0n), Arith.clash(below))
    case 1n+p 1n+q:
      lt_from(p, q, below)

def lt_bound(-a: Nat, -bound: Nat, -other: Nat, same: {bound == other : Nat}, below: {Nat.is_lt(a, bound) == True{} : Bool}) ->
  {Nat.is_lt(a, other) == True{} : Bool}:
  %same : {Nat.is_lt(a, _) == True{} : Bool}
  below

def row_of(+columns: Nat, +position: Nat) -> Nat:
  Nat.div(position, Nat.mul(columns, 1n))

def column_of(+columns: Nat, +position: Nat) -> Nat:
  Nat.div(Nat.mod(position, Nat.mul(columns, 1n)), 1n)

def div_one(+number: Nat) -> {Nat.div(number, 1n) == number : Nat}:
  %Equal.trans(Nat, Nat.add(Nat.mul(number, 1n), 0n), Nat.mul(number, 1n), number,
    Conversion.add_zero(Nat.mul(number, 1n)), Nats.mul_one_right(number)) : {Nat.div(_, 1n) == number : Nat}
  Equal.cong(Nat & Nat, Nat, pair => Pair.fst(Nat, Nat, pair), Nat.divmod(Nat.add(Nat.mul(number, 1n), 0n), 1n), (number, 0n),
    Equal.sym(Nat & Nat, (number, 0n), Nat.divmod(Nat.add(Nat.mul(number, 1n), 0n), 1n), Nats.divmod_exact(number, 1n, 0n, Unit{})))

def row_at(+row: Nat, +column: Nat, +columns: Nat, below: {Nat.is_lt(column, columns) == True{} : Bool}) ->
  {row_of(columns, Nat.add(Nat.mul(row, columns), column)) == row : Nat}:
  %Equal.sym(Nat, Nat.mul(columns, 1n), columns, Nats.mul_one_right(columns)) :
    {Nat.div(Nat.add(Nat.mul(row, columns), column), _) == row : Nat}
  Equal.cong(Nat & Nat, Nat, pair => Pair.fst(Nat, Nat, pair), Nat.divmod(Nat.add(Nat.mul(row, columns), column), columns), (row, column),
    Equal.sym(Nat & Nat, (row, column), Nat.divmod(Nat.add(Nat.mul(row, columns), column), columns),
      Nats.divmod_exact(row, columns, column, lt_from(column, columns, below))))

def column_at(+row: Nat, +column: Nat, +columns: Nat, below: {Nat.is_lt(column, columns) == True{} : Bool}) ->
  {column_of(columns, Nat.add(Nat.mul(row, columns), column)) == column : Nat}:
  %Equal.sym(Nat, Nat.mul(columns, 1n), columns, Nats.mul_one_right(columns)) :
    {Nat.div(Nat.mod(Nat.add(Nat.mul(row, columns), column), _), 1n) == column : Nat}
  %Equal.cong(Nat & Nat, Nat, pair => Pair.snd(Nat, Nat, pair), (row, column), Nat.divmod(Nat.add(Nat.mul(row, columns), column), columns),
    Nats.divmod_exact(row, columns, column, lt_from(column, columns, below))) : {Nat.div(_, 1n) == column : Nat}
  div_one(column)

def output_dimensions(-lo: Matrix.Order, -ro: Matrix.Order, -left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape, -rows: U32, -inner: U32, -right_inner: U32, -columns: U32,
  facts: ProductFacts<lo, ro, left, right, output, rows, inner, right_inner, columns>) -> {Shape.dimensions(output) == [rows, columns] : List<&2, U32>}:
  ProductFacts{le, re, oe, compatible} = facts
  oe

def cell_below(+output: Shape.Shape, +rows: U32, +columns: U32, +oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
  +row: Nat, +row_below: {Nat.is_lt(row, U32.to_nat(rows)) == True{} : Bool},
  +column: Nat, +column_below: {Nat.is_lt(column, U32.to_nat(columns)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(row, U32.to_nat(columns)), column), Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}:
  %Equal.sym(List<&2, U32>, Shape.dimensions(output), [rows, columns], oe) :
    {Nat.is_lt(Nat.add(Nat.mul(row, U32.to_nat(columns)), column), Shape.logical_numel(_)) == True{} : Bool}
  %Equal.sym(Nat, Nat.mul(U32.to_nat(columns), 1n), U32.to_nat(columns), Nats.mul_one_right(U32.to_nat(columns))) :
    {Nat.is_lt(Nat.add(Nat.mul(row, U32.to_nat(columns)), column), Nat.mul(U32.to_nat(rows), _)) == True{} : Bool}
  DotProof.lt_mul_add(row, U32.to_nat(rows), column, U32.to_nat(columns), row_below, column_below)

def kept_value(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +history: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, history)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  +index: U32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  same: {Replayed.value(shape, fill, history, index) == Replayed.value(shape, fill, ops, index) : F32}) ->
  {Core.dynamic_read_value(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, history), fits}}, index)) ==
    Some{Replayed.value(shape, fill, ops, index)} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>,
    Core.dynamic_read_value(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, history), fits}}, index)),
    Some{Replayed.value(shape, fill, history, index)}, Some{Replayed.value(shape, fill, ops, index)},
    PointwiseProof.replayed_point(shape, fill, history, fits, index, inside),
    Equal.cong(F32, Maybe<&2, F32>, word => Some{word}, Replayed.value(shape, fill, history, index), Replayed.value(shape, fill, ops, index), same))

def left_dimensions(-lo: Matrix.Order, -ro: Matrix.Order, -left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape, -rows: U32, -inner: U32, -right_inner: U32, -columns: U32,
  facts: ProductFacts<lo, ro, left, right, output, rows, inner, right_inner, columns>) -> {Shape.dimensions(left) == Matrix.arranged(lo, rows, inner) : List<&2, U32>}:
  ProductFacts{le, re, oe, compatible} = facts
  le

def right_dimensions(-lo: Matrix.Order, -ro: Matrix.Order, -left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape, -rows: U32, -inner: U32, -right_inner: U32, -columns: U32,
  facts: ProductFacts<lo, ro, left, right, output, rows, inner, right_inner, columns>) -> {Shape.dimensions(right) == Matrix.arranged(ro, columns, right_inner) : List<&2, U32>}:
  ProductFacts{le, re, oe, compatible} = facts
  re

def inner_compatible(-lo: Matrix.Order, -ro: Matrix.Order, -left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape, -rows: U32, -inner: U32, -right_inner: U32, -columns: U32,
  facts: ProductFacts<lo, ro, left, right, output, rows, inner, right_inner, columns>) -> {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat}:
  ProductFacts{le, re, oe, compatible} = facts
  compatible

# The parallel product over replayed inputs, in transport form: P holds for
# the product if it holds for both inputs replayed with histories that keep
# their words, next to the output replayed with the writes of every cell.
def ParallelFinal(+lo: Matrix.Order, +ro: Matrix.Order, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +rows: U32, +inner: U32, +columns: U32,
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -s: Storage.Storage, -P: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic> -> Type) -> Type:
  @+left_kept: List<&2, StorageLaws.Op> ->
  @left_values: (@+other: U32 -> {Replayed.value(left, left_fill, left_kept, other) == Replayed.value(left, left_fill, left_ops, other) : F32}) ->
  @left_words: {Laws.words(left, left_fill, left_kept, Shape.logical_numel(Shape.dimensions(left))) ==
    Laws.words(left, left_fill, left_ops, Shape.logical_numel(Shape.dimensions(left))) : List<&2, F32>} ->
  @-left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_kept)) == Shape.logical_numel(Shape.dimensions(left)) : Nat} ->
  @+right_kept: List<&2, StorageLaws.Op> ->
  @right_values: (@+other: U32 -> {Replayed.value(right, right_fill, right_kept, other) == Replayed.value(right, right_fill, right_ops, other) : F32}) ->
  @right_words: {Laws.words(right, right_fill, right_kept, Shape.logical_numel(Shape.dimensions(right))) ==
    Laws.words(right, right_fill, right_ops, Shape.logical_numel(Shape.dimensions(right))) : List<&2, F32>} ->
  @-right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_kept)) == Shape.logical_numel(Shape.dimensions(right)) : Nat} ->
  @-output_fits: {Lifecycle.logical_length(StorageLaws.replay(Par.writes(lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), s)) == Shape.logical_numel(Shape.dimensions(output)) : Nat} ->
    P(Done{(Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_kept), left_fits}},
      Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_kept), right_fits}},
      Core.Packed{output, Core.TTensor{StorageLaws.replay(Par.writes(lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), s), output_fits}})})

def parallel_lift(+lo: Matrix.Order, +ro: Matrix.Order, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == Matrix.arranged(lo, rows, inner) : List<&2, U32>},
  -re: {Shape.dimensions(right) == Matrix.arranged(ro, columns, right_inner) : List<&2, U32>},
  -oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
  -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  +planned: {Matrix.product_plan(lo, ro, left, right) == Done{Matrix.Product{output, rows, inner, right_inner, columns, le, re, oe, compatible}} :
    Result<&2, &2, Matrix.Error, Matrix.Product<lo, ro, left, right>>},
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat},
  -P: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic> -> Type,
  h: ParallelFinal(lo, ro, left, right, output, rows, inner, columns, left_fill, left_ops, right_fill, right_ops, StorageLaws.created(Shape.count(output), 0.0), P)) ->
  P(Matrix.product(lo, ro, Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}}, Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})):
  +plan : Matrix.Product<lo, ro, left, right> = Matrix.Product{output, rows, inner, right_inner, columns, le, re, oe, compatible}
  +facts : ProductFacts<lo, ro, left, right, output, rows, inner, right_inner, columns> = plan_facts(lo, ro, left, right, plan, planned)
  +le_live = left_dimensions(lo, ro, left, right, output, rows, inner, right_inner, columns, facts)
  +re_live = right_dimensions(lo, ro, left, right, output, rows, inner, right_inner, columns, facts)
  +oe_live = output_dimensions(lo, ro, left, right, output, rows, inner, right_inner, columns, facts)
  +compatible_live = inner_compatible(lo, ro, left, right, output, rows, inner, right_inner, columns, facts)
  +left_extents = fits_of(lo, rows, inner, left, le_live)
  +right_extents = inner_fits(ro, columns, inner, right_inner, right, compatible_live, re_live)
  +output_extents = Equal.cong(List<&2, U32>, List<&2, Nat>, dims => Shape.naturals(dims), Shape.dimensions(output), [rows, columns], oe_live)
  %Equal.sym(Result<&2, &2, Matrix.Error, Matrix.Product<lo, ro, left, right>>, Matrix.product_plan(lo, ro, left, right), Done{plan}, planned) :
    P(Matrix.product_planned(lo, ro, left, right, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}, _))
  Replayed.new_lift(output, 0.0,
    result => P(Matrix.product_allocated(lo, ro, left, right, output, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}, result)),
    fo => Owned.created_lift(Shape.count(output), 0.0, Conversion.shape_count_within(output),
      st => (ParallelFinal(lo, ro, left, right, output, rows, inner, columns, left_fill, left_ops, right_fill, right_ops, st, P) ->
        @-fo: {Lifecycle.logical_length(st) == Shape.logical_numel(Shape.dimensions(output)) : Nat} ->
        P(Matrix.product_allocated(lo, ro, left, right, output, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}, Done{Core.TTensor{st, fo}}))),
      zero => final => fo =>
        Par.storage_shift(output, StorageLaws.replay(Par.writes(lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), Storage.Empty{}), Storage.Empty{}, Owned.replay_empty(Par.writes(lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output)))),
          st => k => P(Done{(Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}}, Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}}, Core.Packed{output, Core.TTensor{st, k}})}),
          final(left_ops)(other => {==})({==})(left_fits)(right_ops)(other => {==})({==})(right_fits))(fo),
      d => fits => f => final => fo =>
        Par.input_take(left, left_fill, left_ops,
          taken => P(Matrix.product_parallel_output(lo, ro, output, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), d, U32.to_nat(Shape.count(output)), f, fo,
            taken, Elementwise.take(Core.pack(right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits})))),
          left_kept => left_values => left_words => ha => dl => fa => lsrc => left_kept_fits =>
            Par.input_take(right, right_fill, right_ops,
              taken => P(Matrix.product_parallel_output(lo, ro, output, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), d, U32.to_nat(Shape.count(output)), f, fo,
                (Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_kept), left_kept_fits}}, Api.arr(ha, dl, fa)), taken)),
              right_kept => right_values => right_words => hb => dr => fb => rsrc => right_kept_fits =>
                Par.storage_shift(output,
                  StorageLaws.replay(Par.writes(lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), Storage.Owned{d, U32.to_nat(Shape.count(output)), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
                  Storage.Owned{d, U32.to_nat(Shape.count(output)),
                    Matrix.product_build(d, Matrix.product_splits(d, U32.to_nat(inner)), lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), U32.to_nat(Shape.count(output)), 0n, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)),
                    Matrix.product_build_balanced(d, Matrix.product_splits(d, U32.to_nat(inner)), lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), U32.to_nat(Shape.count(output)), 0n, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)),
                    f},
                  Par.output_eq(lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb,
                    lsrc, rsrc, output, left_extents, right_extents, output_extents, d, fits, f),
                  st => k => P(Done{(Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_kept), left_kept_fits}},
                    Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_kept), right_kept_fits}},
                    Core.Packed{output, Core.TTensor{st, k}})}),
                  final(left_kept)(left_values)(left_words)(left_kept_fits)(right_kept)(right_values)(right_words)(right_kept_fits))(fo))(right_fits))(left_fits))(h)(fo))

def parallel_cell(+lo: Matrix.Order, +ro: Matrix.Order, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +output: Shape.Shape,
  +rows: U32, +inner: U32, +columns: U32, +oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
  +row: Nat, +row_below: {Nat.is_lt(row, U32.to_nat(rows)) == True{} : Bool},
  +column: Nat, +column_below: {Nat.is_lt(column, U32.to_nat(columns)) == True{} : Bool}) ->
  {Replayed.value(output, 0.0, Par.writes(lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), U32.from_nat(Nat.add(Nat.mul(row, U32.to_nat(columns)), column))) == DotFold.dot_fold(Laws.product_factors(row, column, lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0) : F32}:
  +position = Nat.add(Nat.mul(row, U32.to_nat(columns)), column)
  +below = cell_below(output, rows, columns, oe, row, row_below, column, column_below)
  +exact = PointwiseProof.word_exact(position, output, below)
  +room : {Nat.is_le(U32.to_nat(Shape.count(output)), Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool} =
    %Equal.sym(Nat, U32.to_nat(Shape.count(output)), Shape.logical_numel(Shape.dimensions(output)), Reads.count_exact(output)) : {Nat.is_le(_, Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}
    Conversion.le_reflexive(Shape.logical_numel(Shape.dimensions(output)))
  Equal.trans(F32, Replayed.value(output, 0.0, Par.writes(lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), U32.from_nat(position)),
    Par.cell_word(lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(U32.from_nat(position))), DotFold.dot_fold(Laws.product_factors(row, column, lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0),
    Par.writes_value(lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, output, U32.to_nat(Shape.count(output)), U32.from_nat(position), Par.index_inside(output, position, below), room),
    %Equal.sym(Nat, U32.to_nat(U32.from_nat(position)), position, exact) : {Par.cell_word(lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, _) == DotFold.dot_fold(Laws.product_factors(row, column, lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0) : F32}
    %Equal.sym(Nat, row_of(U32.to_nat(columns), position), row, row_at(row, column, U32.to_nat(columns), column_below)) :
      {DotFold.dot_fold(Laws.product_factors(_, column_of(U32.to_nat(columns), position), lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns),
        left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0) == DotFold.dot_fold(Laws.product_factors(row, column, lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0) : F32}
    %Equal.sym(Nat, column_of(U32.to_nat(columns), position), column, column_at(row, column, U32.to_nat(columns), column_below)) :
      {DotFold.dot_fold(Laws.product_factors(row, _, lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns),
        left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0) == DotFold.dot_fold(Laws.product_factors(row, column, lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0) : F32}
    {==})

def parallel_ordered_planned(+lo: Matrix.Order, +ro: Matrix.Order, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == Matrix.arranged(lo, rows, inner) : List<&2, U32>},
  -re: {Shape.dimensions(right) == Matrix.arranged(ro, columns, right_inner) : List<&2, U32>},
  -oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
  -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  +planned: {Matrix.product_plan(lo, ro, left, right) == Done{Matrix.Product{output, rows, inner, right_inner, columns, le, re, oe, compatible}} :
    Result<&2, &2, Matrix.Error, Matrix.Product<lo, ro, left, right>>},
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat}) ->
  Records.MatmulContract<left, right, U32.to_nat(rows), U32.to_nat(columns),
    row => column => DotFold.dot_fold(Laws.product_factors(row, column, lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns),
      left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0),
    other => Replayed.value(left, left_fill, left_ops, other),
    other => Replayed.value(right, right_fill, right_ops, other),
    Matrix.product(lo, ro, Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}}, Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})>:
  +plan : Matrix.Product<lo, ro, left, right> = Matrix.Product{output, rows, inner, right_inner, columns, le, re, oe, compatible}
  +facts : ProductFacts<lo, ro, left, right, output, rows, inner, right_inner, columns> = plan_facts(lo, ro, left, right, plan, planned)
  +oe_live = output_dimensions(lo, ro, left, right, output, rows, inner, right_inner, columns, facts)
  -contract : Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic> -> Type = result =>
    Records.MatmulContract<left, right, U32.to_nat(rows), U32.to_nat(columns),
      row => column => DotFold.dot_fold(Laws.product_factors(row, column, lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns),
        left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0),
      other => Replayed.value(left, left_fill, left_ops, other),
      other => Replayed.value(right, right_fill, right_ops, other), result>
  parallel_lift(lo, ro, left, right, output, rows, inner, right_inner, columns, le, re, oe, compatible, planned,
    left_fill, left_ops, left_fits, right_fill, right_ops, right_fits, contract,
    left_kept => left_values => left_words => a => right_kept => right_values => right_words => b => o => Records.MatmulContract{
      {==},
      Equal.cong(List<&2, U32>, Maybe<&2, List<&2, Nat>>, dims => Some{Shape.naturals(dims)},
        Shape.dimensions(output), [rows, columns], oe_live),
      row => row_below => column => column_below =>
        +row_below = row_below
        +column_below = column_below
        +position = U32.from_nat(Nat.add(Nat.mul(row, U32.to_nat(columns)), column))
        Equal.trans(Maybe<&2, F32>,
          Core.dynamic_read_value(Core.read_dynamic(Core.Packed{output, Core.TTensor{Replayed.storage(output, 0.0, Par.writes(lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output)))), o}}, position)),
          Some{Replayed.value(output, 0.0, Par.writes(lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), position)},
          Some{DotFold.dot_fold(Laws.product_factors(row, column, lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0)},
          PointwiseProof.replayed_point(output, 0.0, Par.writes(lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), o,
            position, PointwiseProof.word_inside(Nat.add(Nat.mul(row, U32.to_nat(columns)), column), output,
              cell_below(output, rows, columns, oe_live, row, row_below, column, column_below))),
          Equal.cong(F32, Maybe<&2, F32>, word => Some{word},
            Replayed.value(output, 0.0, Par.writes(lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), position), DotFold.dot_fold(Laws.product_factors(row, column, lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0),
            parallel_cell(lo, ro, left, left_fill, left_ops, right, right_fill, right_ops, output, rows, inner, columns, oe_live,
              row, row_below, column, column_below))),
      other => inside => kept_value(left, left_fill, left_ops, left_kept, a, other, inside, left_values(other)),
      other => inside => kept_value(right, right_fill, right_ops, right_kept, b, other, inside, right_values(other))})

def parallel_replayed_planned(+lo: Matrix.Order, +ro: Matrix.Order, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == Matrix.arranged(lo, rows, inner) : List<&2, U32>},
  -re: {Shape.dimensions(right) == Matrix.arranged(ro, columns, right_inner) : List<&2, U32>},
  -oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
  -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  +planned: {Matrix.product_plan(lo, ro, left, right) == Done{Matrix.Product{output, rows, inner, right_inner, columns, le, re, oe, compatible}} :
    Result<&2, &2, Matrix.Error, Matrix.Product<lo, ro, left, right>>},
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat},
  -P: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic> -> Type,
  h: Laws.ProductWith(lo, ro, left, right, Matrix.Product{output, rows, inner, right_inner, columns, le, re, oe, compatible},
    left_fill, left_ops, right_fill, right_ops, P)) ->
  P(Matrix.product(lo, ro, Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}}, Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})):
  +plan : Matrix.Product<lo, ro, left, right> = Matrix.Product{output, rows, inner, right_inner, columns, le, re, oe, compatible}
  +facts : ProductFacts<lo, ro, left, right, output, rows, inner, right_inner, columns> = plan_facts(lo, ro, left, right, plan, planned)
  +oe_live = output_dimensions(lo, ro, left, right, output, rows, inner, right_inner, columns, facts)
  parallel_lift(lo, ro, left, right, output, rows, inner, right_inner, columns, le, re, oe, compatible, planned,
    left_fill, left_ops, left_fits, right_fill, right_ops, right_fits, P,
    left_kept => left_values => left_words => a => right_kept => right_values => right_words => b => o =>
      h(left_kept, left_words, a, right_kept, right_words, b, Par.writes(lo, ro, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))),
        row => row_below => column => column_below =>
          parallel_cell(lo, ro, left, left_fill, left_ops, right, right_fill, right_ops, output, rows, inner, columns, oe_live,
            row, row_below, column, column_below),
        o))

def product_ordered(+lo: Matrix.Order, +ro: Matrix.Order, +left: Shape.Shape, +right: Shape.Shape, +plan: Matrix.Product<lo, ro, left, right>,
  planned: {Matrix.product_plan(lo, ro, left, right) == Done{plan} : Result<&2, &2, Matrix.Error, Matrix.Product<lo, ro, left, right>>},
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat}) ->
  Records.MatmulContract<left, right, Laws.product_rows(lo, ro, left, right, plan), Laws.product_columns(lo, ro, left, right, plan),
    row => column => DotFold.dot_fold(Laws.product_factors(row, column, lo, ro, Laws.product_inner(lo, ro, left, right, plan), Laws.product_rows(lo, ro, left, right, plan), Laws.product_columns(lo, ro, left, right, plan),
      left, left_fill, left_ops, right, right_fill, right_ops, Laws.product_inner(lo, ro, left, right, plan)), 0.0),
    other => Replayed.value(left, left_fill, left_ops, other),
    other => Replayed.value(right, right_fill, right_ops, other),
    Matrix.product(lo, ro, Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
      Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})>:
  match plan:
    case Matrix.Product{+output, +rows, +inner, +right_inner, +columns, le, re, oe, compatible}:
      parallel_ordered_planned(lo, ro, left, right, output, rows, inner, right_inner, columns, le, re, oe, compatible, planned,
        left_fill, left_ops, left_fits, right_fill, right_ops, right_fits)

# The histories the product loop leaves on its inputs list the same words.

def product_replayed(+lo: Matrix.Order, +ro: Matrix.Order, +left: Shape.Shape, +right: Shape.Shape, +plan: Matrix.Product<lo, ro, left, right>,
  planned: {Matrix.product_plan(lo, ro, left, right) == Done{plan} : Result<&2, &2, Matrix.Error, Matrix.Product<lo, ro, left, right>>},
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat},
  -P: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic> -> Type,
  h: Laws.ProductWith(lo, ro, left, right, plan, left_fill, left_ops, right_fill, right_ops, P)) ->
  P(Matrix.product(lo, ro, Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
    Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})):
  match plan:
    case Matrix.Product{+output, +rows, +inner, +right_inner, +columns, le, re, oe, compatible}:
      parallel_replayed_planned(lo, ro, left, right, output, rows, inner, right_inner, columns, le, re, oe, compatible, planned,
        left_fill, left_ops, left_fits, right_fill, right_ops, right_fits, P, h)
