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
import ./matmul_parallel_proof.bend as Par
import ../read_contract.bend as Reads
import ../../storage/api.bend as Api
import ../../storage/array.bend as ArrayCore
import ./binary_storage_proof.bend as Owned
import ../elementwise.bend as Elementwise

type MatrixFacts<-left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape, -rows: U32, -inner: U32, -right_inner: U32, -columns: U32> is Data:
  MatrixFacts{left_dimensions: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
    right_dimensions: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
    output_dimensions: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
    compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat}}

def Facts(-left: Shape.Shape, -right: Shape.Shape, result: Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>) -> Data:
  match result:
    case Fail{error}:
      Unit
    case Done{Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible}}:
      MatrixFacts<left, right, output, rows, inner, right_inner, columns>

def facts_output(-left: Shape.Shape, -right: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
  re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  result: Shape.ConstructResult<[rows, columns]>) ->
  Facts(left, right, Matrix.plan_output(left, right, rows, inner, right_inner, columns, le, re, compatible, result)):
  match result:
    case Shape.SError{error}:
      Unit{}
    case Shape.SBuilt{shape, preserved}:
      MatrixFacts{le, re, preserved, compatible}

def facts_compatible(-left: Shape.Shape, -right: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
  re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))>) ->
  Facts(left, right, Matrix.plan_compatible(left, right, rows, inner, right_inner, columns, le, re, decision)):
  match decision:
    case Shape.BFalse{equal}:
      Unit{}
    case Shape.BTrue{equal}:
      facts_output(left, right, rows, inner, right_inner, columns, le, re,
        Lifecycle.natural_equal(U32.to_nat(inner), U32.to_nat(right_inner), equal), Shape.construct([rows, columns]))

def facts_right(-left: Shape.Shape, +right: Shape.Shape, +rows: U32, +inner: U32,
  le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, dimensions: List<&2, U32>,
  same: {dimensions == Shape.dimensions(right) : List<&2, U32>}) ->
  Facts(left, right, Matrix.plan_right(left, right, rows, inner, le, dimensions, same)):
  match dimensions:
    case +right_inner <> +columns <> Nil{}:
      facts_compatible(left, right, rows, inner, right_inner, columns, le,
        Equal.sym(List<&2, U32>, [right_inner, columns], Shape.dimensions(right), same),
        Shape.bool_case(Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))))
    case Nil{}:
      Unit{}
    case head <> Nil{}:
      Unit{}
    case first <> second <> third <> rest:
      Unit{}

def facts_left(+left: Shape.Shape, +right: Shape.Shape, dimensions: List<&2, U32>,
  same: {dimensions == Shape.dimensions(left) : List<&2, U32>}) ->
  Facts(left, right, Matrix.plan_left(left, right, dimensions, same)):
  match dimensions:
    case +rows <> +inner <> Nil{}:
      facts_right(left, right, rows, inner, Equal.sym(List<&2, U32>, [rows, inner], Shape.dimensions(left), same),
        Shape.dimensions(right), {==})
    case Nil{}:
      Unit{}
    case head <> Nil{}:
      Unit{}
    case first <> second <> third <> rest:
      Unit{}

def plan_facts(+left: Shape.Shape, +right: Shape.Shape, +plan: Matrix.Plan<left, right>,
  planned: {Matrix.plan(left, right) == Done{plan} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>}) ->
  Facts(left, right, Done{plan}):
  %planned : Facts(left, right, _)
  facts_left(left, right, Shape.dimensions(left), {==})

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

def output_dimensions(-left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape, -rows: U32, -inner: U32, -right_inner: U32, -columns: U32,
  facts: MatrixFacts<left, right, output, rows, inner, right_inner, columns>) -> {Shape.dimensions(output) == [rows, columns] : List<&2, U32>}:
  MatrixFacts{le, re, oe, compatible} = facts
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

def left_dimensions(-left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape, -rows: U32, -inner: U32, -right_inner: U32, -columns: U32,
  facts: MatrixFacts<left, right, output, rows, inner, right_inner, columns>) -> {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}:
  MatrixFacts{le, re, oe, compatible} = facts
  le

def right_dimensions(-left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape, -rows: U32, -inner: U32, -right_inner: U32, -columns: U32,
  facts: MatrixFacts<left, right, output, rows, inner, right_inner, columns>) -> {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>}:
  MatrixFacts{le, re, oe, compatible} = facts
  re

def inner_compatible(-left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape, -rows: U32, -inner: U32, -right_inner: U32, -columns: U32,
  facts: MatrixFacts<left, right, output, rows, inner, right_inner, columns>) -> {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat}:
  MatrixFacts{le, re, oe, compatible} = facts
  compatible

# The parallel product over replayed inputs, in transport form: P holds for
# the product if it holds for both inputs replayed with histories that keep
# their words, next to the output replayed with the writes of every cell.
def ParallelFinal(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +inner: U32, +columns: U32,
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
  @-output_fits: {Lifecycle.logical_length(StorageLaws.replay(Par.writes(U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), s)) == Shape.logical_numel(Shape.dimensions(output)) : Nat} ->
    P(Done{(Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_kept), left_fits}},
      Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_kept), right_fits}},
      Core.Packed{output, Core.TTensor{StorageLaws.replay(Par.writes(U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), s), output_fits}})})

def parallel_lift(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
  -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
  -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  +planned: {Matrix.plan(left, right) == Done{Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible}} :
    Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>},
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat},
  -P: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic> -> Type,
  h: ParallelFinal(left, right, output, inner, columns, left_fill, left_ops, right_fill, right_ops, StorageLaws.created(Shape.count(output), 0.0), P)) ->
  P(Matrix.matmul(Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}}, Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})):
  +plan : Matrix.Plan<left, right> = Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible}
  +facts : MatrixFacts<left, right, output, rows, inner, right_inner, columns> = plan_facts(left, right, plan, planned)
  +le_live = left_dimensions(left, right, output, rows, inner, right_inner, columns, facts)
  +re_live = right_dimensions(left, right, output, rows, inner, right_inner, columns, facts)
  +oe_live = output_dimensions(left, right, output, rows, inner, right_inner, columns, facts)
  +compatible_live = inner_compatible(left, right, output, rows, inner, right_inner, columns, facts)
  +left_extents = Equal.cong(List<&2, U32>, List<&2, Nat>, dims => Shape.naturals(dims), Shape.dimensions(left), [rows, inner], le_live)
  +right_extents = Equal.trans(List<&2, Nat>, Shape.extents(right), [U32.to_nat(right_inner), U32.to_nat(columns)], [U32.to_nat(inner), U32.to_nat(columns)],
    Equal.cong(List<&2, U32>, List<&2, Nat>, dims => Shape.naturals(dims), Shape.dimensions(right), [right_inner, columns], re_live),
    Equal.cong(Nat, List<&2, Nat>, n => [n, U32.to_nat(columns)], U32.to_nat(right_inner), U32.to_nat(inner),
      Equal.sym(Nat, U32.to_nat(inner), U32.to_nat(right_inner), compatible_live)))
  +output_extents = Equal.cong(List<&2, U32>, List<&2, Nat>, dims => Shape.naturals(dims), Shape.dimensions(output), [rows, columns], oe_live)
  %Equal.sym(Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>, Matrix.plan(left, right), Done{plan}, planned) :
    P(Matrix.parallel_planned(left, right, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}, _))
  Replayed.new_lift(output, 0.0,
    result => P(Matrix.parallel_allocated(left, right, output, U32.to_nat(inner), U32.to_nat(columns), Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}, result)),
    fo => Owned.created_lift(Shape.count(output), 0.0, Conversion.shape_count_within(output),
      st => (ParallelFinal(left, right, output, inner, columns, left_fill, left_ops, right_fill, right_ops, st, P) ->
        @-fo: {Lifecycle.logical_length(st) == Shape.logical_numel(Shape.dimensions(output)) : Nat} ->
        P(Matrix.parallel_allocated(left, right, output, U32.to_nat(inner), U32.to_nat(columns), Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}, Done{Core.TTensor{st, fo}}))),
      zero => final => fo =>
        Par.storage_shift(output, StorageLaws.replay(Par.writes(U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), Storage.Empty{}), Storage.Empty{}, Owned.replay_empty(Par.writes(U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output)))),
          st => k => P(Done{(Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}}, Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}}, Core.Packed{output, Core.TTensor{st, k}})}),
          final(left_ops)(other => {==})({==})(left_fits)(right_ops)(other => {==})({==})(right_fits))(fo),
      d => fits => f => final => fo =>
        Par.input_take(left, left_fill, left_ops,
          taken => P(Matrix.parallel_output(output, U32.to_nat(inner), U32.to_nat(columns), d, U32.to_nat(Shape.count(output)), f, fo,
            taken, Elementwise.take(Core.pack(right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits})))),
          left_kept => left_values => left_words => ha => dl => fa => lsrc => left_kept_fits =>
            Par.input_take(right, right_fill, right_ops,
              taken => P(Matrix.parallel_output(output, U32.to_nat(inner), U32.to_nat(columns), d, U32.to_nat(Shape.count(output)), f, fo,
                (Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_kept), left_kept_fits}}, Api.arr(ha, dl, fa)), taken)),
              right_kept => right_values => right_words => hb => dr => fb => rsrc => right_kept_fits =>
                Par.storage_shift(output,
                  StorageLaws.replay(Par.writes(U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), Storage.Owned{d, U32.to_nat(Shape.count(output)), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
                  Storage.Owned{d, U32.to_nat(Shape.count(output)),
                    Matrix.build(d, Matrix.product_splits(d, U32.to_nat(inner)), U32.to_nat(inner), U32.to_nat(columns), U32.to_nat(Shape.count(output)), 0n, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)),
                    Matrix.build_balanced(d, Matrix.product_splits(d, U32.to_nat(inner)), U32.to_nat(inner), U32.to_nat(columns), U32.to_nat(Shape.count(output)), 0n, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)),
                    f},
                  Par.output_eq(U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb,
                    lsrc, rsrc, output, U32.to_nat(rows), left_extents, right_extents, output_extents, d, fits, f),
                  st => k => P(Done{(Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_kept), left_kept_fits}},
                    Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_kept), right_kept_fits}},
                    Core.Packed{output, Core.TTensor{st, k}})}),
                  final(left_kept)(left_values)(left_words)(left_kept_fits)(right_kept)(right_values)(right_words)(right_kept_fits))(fo))(right_fits))(left_fits))(h)(fo))

def parallel_cell(+left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +output: Shape.Shape,
  +rows: U32, +inner: U32, +columns: U32, +oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
  +row: Nat, +row_below: {Nat.is_lt(row, U32.to_nat(rows)) == True{} : Bool},
  +column: Nat, +column_below: {Nat.is_lt(column, U32.to_nat(columns)) == True{} : Bool}) ->
  {Replayed.value(output, 0.0, Par.writes(U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), U32.from_nat(Nat.add(Nat.mul(row, U32.to_nat(columns)), column))) == DotFold.dot_fold(Laws.factors(row, column, U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0) : F32}:
  +position = Nat.add(Nat.mul(row, U32.to_nat(columns)), column)
  +below = cell_below(output, rows, columns, oe, row, row_below, column, column_below)
  +exact = PointwiseProof.word_exact(position, output, below)
  +room : {Nat.is_le(U32.to_nat(Shape.count(output)), Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool} =
    %Equal.sym(Nat, U32.to_nat(Shape.count(output)), Shape.logical_numel(Shape.dimensions(output)), Reads.count_exact(output)) : {Nat.is_le(_, Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}
    Conversion.le_reflexive(Shape.logical_numel(Shape.dimensions(output)))
  Equal.trans(F32, Replayed.value(output, 0.0, Par.writes(U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), U32.from_nat(position)),
    Par.cell_word(U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(U32.from_nat(position))), DotFold.dot_fold(Laws.factors(row, column, U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0),
    Par.writes_value(U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, output, U32.to_nat(Shape.count(output)), U32.from_nat(position), Par.index_inside(output, position, below), room),
    %Equal.sym(Nat, U32.to_nat(U32.from_nat(position)), position, exact) : {Par.cell_word(U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, _) == DotFold.dot_fold(Laws.factors(row, column, U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0) : F32}
    %Equal.sym(Nat, row_of(U32.to_nat(columns), position), row, row_at(row, column, U32.to_nat(columns), column_below)) :
      {DotFold.dot_fold(Laws.factors(_, column_of(U32.to_nat(columns), position), U32.to_nat(inner), U32.to_nat(columns),
        left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0) == DotFold.dot_fold(Laws.factors(row, column, U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0) : F32}
    %Equal.sym(Nat, column_of(U32.to_nat(columns), position), column, column_at(row, column, U32.to_nat(columns), column_below)) :
      {DotFold.dot_fold(Laws.factors(row, _, U32.to_nat(inner), U32.to_nat(columns),
        left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0) == DotFold.dot_fold(Laws.factors(row, column, U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0) : F32}
    {==})

def parallel_ordered_planned(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
  -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
  -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  +planned: {Matrix.plan(left, right) == Done{Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible}} :
    Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>},
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat}) ->
  Records.MatmulContract<left, right, U32.to_nat(rows), U32.to_nat(columns),
    row => column => DotFold.dot_fold(Laws.factors(row, column, U32.to_nat(inner), U32.to_nat(columns),
      left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0),
    other => Replayed.value(left, left_fill, left_ops, other),
    other => Replayed.value(right, right_fill, right_ops, other),
    Matrix.matmul(Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}}, Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})>:
  +plan : Matrix.Plan<left, right> = Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible}
  +facts : MatrixFacts<left, right, output, rows, inner, right_inner, columns> = plan_facts(left, right, plan, planned)
  +oe_live = output_dimensions(left, right, output, rows, inner, right_inner, columns, facts)
  -contract : Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic> -> Type = result =>
    Records.MatmulContract<left, right, U32.to_nat(rows), U32.to_nat(columns),
      row => column => DotFold.dot_fold(Laws.factors(row, column, U32.to_nat(inner), U32.to_nat(columns),
        left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0),
      other => Replayed.value(left, left_fill, left_ops, other),
      other => Replayed.value(right, right_fill, right_ops, other), result>
  parallel_lift(left, right, output, rows, inner, right_inner, columns, le, re, oe, compatible, planned,
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
          Core.dynamic_read_value(Core.read_dynamic(Core.Packed{output, Core.TTensor{Replayed.storage(output, 0.0, Par.writes(U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output)))), o}}, position)),
          Some{Replayed.value(output, 0.0, Par.writes(U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), position)},
          Some{DotFold.dot_fold(Laws.factors(row, column, U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0)},
          PointwiseProof.replayed_point(output, 0.0, Par.writes(U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), o,
            position, PointwiseProof.word_inside(Nat.add(Nat.mul(row, U32.to_nat(columns)), column), output,
              cell_below(output, rows, columns, oe_live, row, row_below, column, column_below))),
          Equal.cong(F32, Maybe<&2, F32>, word => Some{word},
            Replayed.value(output, 0.0, Par.writes(U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), position), DotFold.dot_fold(Laws.factors(row, column, U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(inner)), 0.0),
            parallel_cell(left, left_fill, left_ops, right, right_fill, right_ops, output, rows, inner, columns, oe_live,
              row, row_below, column, column_below))),
      other => inside => kept_value(left, left_fill, left_ops, left_kept, a, other, inside, left_values(other)),
      other => inside => kept_value(right, right_fill, right_ops, right_kept, b, other, inside, right_values(other))})

def parallel_replayed_planned(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
  -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
  -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  +planned: {Matrix.plan(left, right) == Done{Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible}} :
    Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>},
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat},
  -P: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic> -> Type,
  h: Laws.MatmulWith(left, right, Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible},
    left_fill, left_ops, right_fill, right_ops, P)) ->
  P(Matrix.matmul(Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}}, Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})):
  +plan : Matrix.Plan<left, right> = Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible}
  +facts : MatrixFacts<left, right, output, rows, inner, right_inner, columns> = plan_facts(left, right, plan, planned)
  +oe_live = output_dimensions(left, right, output, rows, inner, right_inner, columns, facts)
  parallel_lift(left, right, output, rows, inner, right_inner, columns, le, re, oe, compatible, planned,
    left_fill, left_ops, left_fits, right_fill, right_ops, right_fits, P,
    left_kept => left_values => left_words => a => right_kept => right_values => right_words => b => o =>
      h(left_kept, left_words, a, right_kept, right_words, b, Par.writes(U32.to_nat(inner), U32.to_nat(columns), left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))),
        row => row_below => column => column_below =>
          parallel_cell(left, left_fill, left_ops, right, right_fill, right_ops, output, rows, inner, columns, oe_live,
            row, row_below, column, column_below),
        o))

def matmul_ordered(+left: Shape.Shape, +right: Shape.Shape, +plan: Matrix.Plan<left, right>,
  planned: {Matrix.plan(left, right) == Done{plan} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>},
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat}) ->
  Records.MatmulContract<left, right, Laws.matrix_rows(left, right, plan), Laws.matrix_columns(left, right, plan),
    row => column => DotFold.dot_fold(Laws.factors(row, column, Laws.matrix_inner(left, right, plan), Laws.matrix_columns(left, right, plan),
      left, left_fill, left_ops, right, right_fill, right_ops, Laws.matrix_inner(left, right, plan)), 0.0),
    other => Replayed.value(left, left_fill, left_ops, other),
    other => Replayed.value(right, right_fill, right_ops, other),
    Matrix.matmul(Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
      Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})>:
  match plan:
    case Matrix.Plan{+output, +rows, +inner, +right_inner, +columns, le, re, oe, compatible}:
      parallel_ordered_planned(left, right, output, rows, inner, right_inner, columns, le, re, oe, compatible, planned,
        left_fill, left_ops, left_fits, right_fill, right_ops, right_fits)

# The histories the product loop leaves on its inputs list the same words.

def matmul_replayed(+left: Shape.Shape, +right: Shape.Shape, +plan: Matrix.Plan<left, right>,
  planned: {Matrix.plan(left, right) == Done{plan} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>},
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat},
  -P: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic> -> Type,
  h: Laws.MatmulWith(left, right, plan, left_fill, left_ops, right_fill, right_ops, P)) ->
  P(Matrix.matmul(Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
    Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})):
  match plan:
    case Matrix.Plan{+output, +rows, +inner, +right_inner, +columns, le, re, oe, compatible}:
      parallel_replayed_planned(left, right, output, rows, inner, right_inner, columns, le, re, oe, compatible, planned,
        left_fill, left_ops, left_fits, right_fill, right_ops, right_fits, P, h)
