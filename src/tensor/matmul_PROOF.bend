import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./core_PROOF.bend as CoreProof
import ./lifecycle.bend as Lifecycle
import ./matmul.bend as Matrix
import ./matmul_LAWS.bend as Laws
import ./dot_fold.bend as DotFold
import ./math_totality.bend as Totality
import ./reduction_PROOF.bend as MathProof
import ./word_conversion.bend as Conversion
import ./allocation_PROOF.bend as AllocationProof
import ./allocation_spec.bend as Allocation
import ./transform_PROOF.bend as TransformProof
import ./transform_LAWS.bend as TransformLaws
import ./transform_totality.bend as TransformTotality
import ../shape/LAWS.bend as ShapeLaws
import ../shape/PROOF.bend as ShapeProof
import ../shape/boundary_proofs.bend as Boundary
import ../shape/coordinate_proofs.bend as Coordinates
import ../shape/nat_proofs.bend as Nats
import ../storage/arith.bend as Arith
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ../storage/route.bend as Route
import ../storage/certificates.bend as Certificates
import ./elementwise.bend as Elementwise
import ./elementwise_totality.bend as ElementwiseTotality
import ./elementwise_parallel_proof.bend as Parallel
import ./matmul_parallel_proof.bend as MatmulParallel
import ./matrix_index_proof.bend as MatrixIndex

type CoordinateRead<-shape: Shape.Shape, -original: Core.Tensor<shape>, -coordinates: List<&2, U32>> is Type:
  CoordinateRead{tensor: Core.Tensor<shape>, result: Result<&2, &2, Core.Error, F32>,
    exact: {(tensor, result) == Core.read_coordinate(shape, original, coordinates) : Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>},
    -meaning: @unit: Unit -> {Core.result_value(result) == Laws.coordinate_observe(shape, original, coordinates) : Maybe<&2, F32>},
    -kept: @+other: U32 -> {Core.checked_observe(shape, tensor, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}}

def coordinate_flat(+shape: Shape.Shape, -original: Core.Tensor<shape>, -coordinates: List<&2, U32>, +index: Shape.FlatIndex<shape>,
  same: {Done{index} == Shape.flatten_checked(shape, coordinates) : Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>},
  certificate: Core.Read<shape, original, Shape.index_value(shape, index)>,
  -source: Core.SourceRead<shape, original, Shape.index_value(shape, index)>,
  -retention: Core.SourceReadKeeps<shape, original, Shape.index_value(shape, index)>) -> CoordinateRead<shape, original, coordinates>:
  Core.Read{tensor, +result, +exact} = certificate
  CoordinateRead{tensor, result,
    %same : {(tensor, result) == Core.read_coordinate_result(shape, original, _) : Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>}
    exact,
    unit =>
      prove : @source: Core.SourceRead<shape, original, Shape.index_value(shape, index)> ->
      {Core.result_value(result) == Laws.coordinate_observe(shape, original, coordinates) : Maybe<&2, F32>} =
      \{Core.SourceRead: meaning =>
        %same : {Core.result_value(result) == Laws.coordinate_observe_result(shape, original, _) : Maybe<&2, F32>}
        Equal.trans(Maybe<&2, F32>, Core.result_value(result), Core.read_value(shape, Core.read_flat(shape, original, Shape.index_value(shape, index))),
          Core.checked_observe(shape, original, Shape.index_value(shape, index)),
          Equal.cong(Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>, Maybe<&2, F32>, pair => Core.read_value(shape, pair),
            (tensor, result), Core.read_flat(shape, original, Shape.index_value(shape, index)), exact), meaning)}
      prove(source),
    other =>
      prove : @retention: Core.SourceReadKeeps<shape, original, Shape.index_value(shape, index)> ->
        {Core.checked_observe(shape, tensor, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>} =
        \{Core.SourceReadKeeps: kept => Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, tensor, other),
          Core.retained_read_observe(shape, Core.read_flat(shape, original, Shape.index_value(shape, index)), other), Core.checked_observe(shape, original, other),
          Equal.cong(Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>, Maybe<&2, F32>, pair => Core.retained_read_observe(shape, pair, other),
            (tensor, result), Core.read_flat(shape, original, Shape.index_value(shape, index)), exact), kept(other))}
      prove(retention)}

def coordinate_decided(+shape: Shape.Shape, tensor: Core.Tensor<shape>, -coordinates: List<&2, U32>,
  result: Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>,
  same: {result == Shape.flatten_checked(shape, coordinates) : Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>}) -> CoordinateRead<shape, tensor, coordinates>:
  match result:
    case Fail{error}:
      CoordinateRead{tensor, Fail{Core.ShapeError{error}},
        %same : {(tensor, Fail{Core.ShapeError{error}}) == Core.read_coordinate_result(shape, tensor, _) : Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>}
        {==},
        unit => %same : {None{} == Laws.coordinate_observe_result(shape, tensor, _) : Maybe<&2, F32>}
        {==}, other => {==}}
    case Done{+index}:
      coordinate_flat(shape, tensor, coordinates, index, same, Core.read_certificate(shape, tensor, Shape.index_value(shape, index)),
        Core.source_read_model(shape, tensor, Shape.index_value(shape, index)), Core.source_read_keeps(shape, tensor, Shape.index_value(shape, index)))

def coordinate_certificate(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +coordinates: List<&2, U32>) -> CoordinateRead<shape, tensor, coordinates>:
  coordinate_decided(shape, tensor, coordinates, Shape.flatten_checked(shape, coordinates), {==})

def coordinate_keeps_result(+shape: Shape.Shape, -original: Core.Tensor<shape>, -tensor: Core.Tensor<shape>,
  result: Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>,
  kept: @+other: U32 -> {Core.checked_observe(shape, tensor, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}) ->
  {Laws.coordinate_observe_result(shape, tensor, result) == Laws.coordinate_observe_result(shape, original, result) : Maybe<&2, F32>}:
  match result:
    case Fail{error}:
      {==}
    case Done{index}:
      kept(Shape.index_value(shape, index))

def coordinate_keeps(+shape: Shape.Shape, -original: Core.Tensor<shape>, -tensor: Core.Tensor<shape>, +coordinates: List<&2, U32>,
  kept: @+other: U32 -> {Core.checked_observe(shape, tensor, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}) ->
  {Laws.coordinate_observe(shape, tensor, coordinates) == Laws.coordinate_observe(shape, original, coordinates) : Maybe<&2, F32>}:
  coordinate_keeps_result(shape, original, tensor, Shape.flatten_checked(shape, coordinates), kept)

def logical_dot_original(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  -ra: Core.Tensor<left>, -rb: Core.Tensor<right>, +row: U32, +column: U32, +cursor: Nat, values: List<&2, DotFold.Factors>,
  logical: Laws.LogicalDot(fuel, left, right, ra, rb, row, column, cursor, values),
  -left_kept: @+other: U32 -> {Core.checked_observe(left, ra, other) == Core.checked_observe(left, a, other) : Maybe<&2, F32>},
  -right_kept: @+other: U32 -> {Core.checked_observe(right, rb, other) == Core.checked_observe(right, b, other) : Maybe<&2, F32>}) ->
  Laws.LogicalDot(fuel, left, right, a, b, row, column, cursor, values):
  match fuel values:
    case 0n Nil{}:
      Unit{}
    case 1n+tail DotFold.Factors{+lv, +rv} <> rest:
      Laws.DotStep{exact, lo, ro, remaining} = logical
      Laws.DotStep{exact,
        Equal.trans(Maybe<&2, F32>, Laws.coordinate_observe(left, a, [row, U32.from_nat(cursor)]),
          Laws.coordinate_observe(left, ra, [row, U32.from_nat(cursor)]), Some{lv},
          Equal.sym(Maybe<&2, F32>, Laws.coordinate_observe(left, ra, [row, U32.from_nat(cursor)]),
            Laws.coordinate_observe(left, a, [row, U32.from_nat(cursor)]), coordinate_keeps(left, a, ra, [row, U32.from_nat(cursor)], left_kept)), lo),
        Equal.trans(Maybe<&2, F32>, Laws.coordinate_observe(right, b, [U32.from_nat(cursor), column]),
          Laws.coordinate_observe(right, rb, [U32.from_nat(cursor), column]), Some{rv},
          Equal.sym(Maybe<&2, F32>, Laws.coordinate_observe(right, rb, [U32.from_nat(cursor), column]),
            Laws.coordinate_observe(right, b, [U32.from_nat(cursor), column]), coordinate_keeps(right, b, rb, [U32.from_nat(cursor), column], right_kept)), ro),
        logical_dot_original(tail, left, right, a, b, ra, rb, row, column, 1n+cursor, rest, remaining, left_kept, right_kept)}
    case 0n DotFold.Factors{lv, rv} <> rest:
      Empty.absurd(Laws.LogicalDot(0n, left, right, a, b, row, column, cursor, DotFold.Factors{lv, rv} <> rest), logical)
    case 1n+tail Nil{}:
      Empty.absurd(Laws.LogicalDot(1n+tail, left, right, a, b, row, column, cursor, Nil{}), logical)

def dot_extend(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  -ra: Core.Tensor<left>, -rb: Core.Tensor<right>, +row: U32, +column: U32, +cursor: Nat, +accumulator: F32, +lv: F32, +rv: F32,
  result: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32>,
  next: Laws.DotResultMeaning(fuel, left, right, ra, rb, row, column, 1n+cursor, F32.add(accumulator, F32.mul(lv, rv)), result),
  -exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  -lo: {Laws.coordinate_observe(left, a, [row, U32.from_nat(cursor)]) == Some{lv} : Maybe<&2, F32>},
  -ro: {Laws.coordinate_observe(right, b, [U32.from_nat(cursor), column]) == Some{rv} : Maybe<&2, F32>},
  -left_kept: @+other: U32 -> {Core.checked_observe(left, ra, other) == Core.checked_observe(left, a, other) : Maybe<&2, F32>},
  -right_kept: @+other: U32 -> {Core.checked_observe(right, rb, other) == Core.checked_observe(right, b, other) : Maybe<&2, F32>}) ->
  Laws.DotResultMeaning(1n+fuel, left, right, a, b, row, column, cursor, accumulator, result):
  match result:
    case Fail{error}:
      Unit{}
    case Done{pair}:
      (final_a, final_b, value) = pair
      Laws.DotMeaning{values, logical, fold, lk, rk} = next
      Laws.DotMeaning{DotFold.Factors{lv, rv} <> values,
        Laws.DotStep{exact, lo, ro, logical_dot_original(fuel, left, right, a, b, ra, rb, row, column, 1n+cursor, values, logical, left_kept, right_kept)}, fold,
        other => Equal.trans(Maybe<&2, F32>, Core.checked_observe(left, final_a, other),
          Core.checked_observe(left, ra, other), Core.checked_observe(left, a, other), lk(other), left_kept(other)),
        other => Equal.trans(Maybe<&2, F32>, Core.checked_observe(right, final_b, other),
          Core.checked_observe(right, rb, other), Core.checked_observe(right, b, other), rk(other), right_kept(other))}

def right_pair(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  -ra: Core.Tensor<left>, rb: Core.Tensor<right>, result: Result<&2, &2, Core.Error, F32>,
  +row: U32, +column: U32, +cursor: Nat, +accumulator: F32, +lv: F32,
  -rec: @a: Core.Tensor<left> -> @b: Core.Tensor<right> -> @+value: F32 ->
    Laws.SourceDot<fuel, left, right, a, b, row, column, 1n+cursor, value, Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value)>,
  -exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  -lo: {Laws.coordinate_observe(left, a, [row, U32.from_nat(cursor)]) == Some{lv} : Maybe<&2, F32>},
  -meaning: @unit: Unit -> {Core.result_value(result) == Laws.coordinate_observe(right, b, [U32.from_nat(cursor), column]) : Maybe<&2, F32>},
  -left_kept: @+other: U32 -> {Core.checked_observe(left, ra, other) == Core.checked_observe(left, a, other) : Maybe<&2, F32>},
  -right_kept: @+other: U32 -> {Core.checked_observe(right, rb, other) == Core.checked_observe(right, b, other) : Maybe<&2, F32>}) ->
  Laws.SourceDot<1n+fuel, left, right, a, b, row, column, cursor, accumulator, Matrix.right_read(left, right,
    a => b => value => Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value), ra, lv, accumulator, (rb, result))>:
  match result:
    case Fail{error}:
      Laws.SourceDot{unit => Unit{}}
    case Done{+rv}:
      Laws.SourceDot{unit =>
        claim : @certificate: Laws.SourceDot<fuel, left, right, ra, rb, row, column, 1n+cursor, F32.add(accumulator, F32.mul(lv, rv)),
          Matrix.natural_dot(fuel, left, right, ra, rb, row, column, 1n+cursor, F32.add(accumulator, F32.mul(lv, rv)))> ->
          Laws.DotResultMeaning(fuel, left, right, ra, rb, row, column, 1n+cursor, F32.add(accumulator, F32.mul(lv, rv)),
            Matrix.natural_dot(fuel, left, right, ra, rb, row, column, 1n+cursor, F32.add(accumulator, F32.mul(lv, rv)))) =
          \{Laws.SourceDot: proof => proof(Unit{})}
        dot_extend(fuel, left, right, a, b, ra, rb, row, column, cursor, accumulator, lv, rv,
          Matrix.natural_dot(fuel, left, right, ra, rb, row, column, 1n+cursor, F32.add(accumulator, F32.mul(lv, rv))),
          claim(rec(ra, rb, F32.add(accumulator, F32.mul(lv, rv)))), exact, lo,
          Equal.sym(Maybe<&2, F32>, Some{rv}, Laws.coordinate_observe(right, b, [U32.from_nat(cursor), column]), meaning(Unit{})), left_kept, right_kept)}

def right_certificate(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>, -ra: Core.Tensor<left>,
  +row: U32, +column: U32, +cursor: Nat, +accumulator: F32, +lv: F32,
  -rec: @a: Core.Tensor<left> -> @b: Core.Tensor<right> -> @+value: F32 ->
    Laws.SourceDot<fuel, left, right, a, b, row, column, 1n+cursor, value, Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value)>,
  -exact_cursor: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  -lo: {Laws.coordinate_observe(left, a, [row, U32.from_nat(cursor)]) == Some{lv} : Maybe<&2, F32>},
  -left_kept: @+other: U32 -> {Core.checked_observe(left, ra, other) == Core.checked_observe(left, a, other) : Maybe<&2, F32>},
  certificate: CoordinateRead<right, b, [U32.from_nat(cursor), column]>) ->
  Laws.SourceDot<1n+fuel, left, right, a, b, row, column, cursor, accumulator, Matrix.right_read(left, right,
    a => b => value => Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value), ra, lv, accumulator,
    Core.read_coordinate(right, b, [U32.from_nat(cursor), column]))>:
  CoordinateRead{rb, +result, +exact, meaning, right_kept} = certificate
  Laws.SourceDot{unit =>
    claim : @certificate: Laws.SourceDot<1n+fuel, left, right, a, b, row, column, cursor, accumulator, Matrix.right_read(left, right,
      a => b => value => Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value), ra, lv, accumulator, (rb, result))> ->
      Laws.DotResultMeaning(1n+fuel, left, right, a, b, row, column, cursor, accumulator, Matrix.right_read(left, right,
        a => b => value => Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value), ra, lv, accumulator, (rb, result))) =
      \{Laws.SourceDot: proof => proof(Unit{})}
    %exact : Laws.DotResultMeaning(1n+fuel, left, right, a, b, row, column, cursor, accumulator, Matrix.right_read(left, right,
      a => b => value => Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value), ra, lv, accumulator, _))
    claim(right_pair(fuel, left, right, a, b, ra, rb, result, row, column, cursor, accumulator, lv, rec, exact_cursor, lo, meaning, left_kept, right_kept))}

def left_pair(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, -a: Core.Tensor<left>, b: Core.Tensor<right>,
  ra: Core.Tensor<left>, result: Result<&2, &2, Core.Error, F32>,
  +row: U32, +column: U32, +cursor: Nat, +accumulator: F32,
  -rec: @a: Core.Tensor<left> -> @b: Core.Tensor<right> -> @+value: F32 ->
    Laws.SourceDot<fuel, left, right, a, b, row, column, 1n+cursor, value, Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value)>,
  -exact_cursor: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  -meaning: @unit: Unit -> {Core.result_value(result) == Laws.coordinate_observe(left, a, [row, U32.from_nat(cursor)]) : Maybe<&2, F32>},
  -left_kept: @+other: U32 -> {Core.checked_observe(left, ra, other) == Core.checked_observe(left, a, other) : Maybe<&2, F32>}) ->
  Laws.SourceDot<1n+fuel, left, right, a, b, row, column, cursor, accumulator, Matrix.left_read(left, right,
    a => b => value => Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value), b, [U32.from_nat(cursor), column], accumulator, (ra, result))>:
  match result:
    case Fail{error}:
      Laws.SourceDot{unit => Unit{}}
    case Done{+lv}:
      right_certificate(fuel, left, right, a, b, ra, row, column, cursor, accumulator, lv, rec, exact_cursor,
        Equal.sym(Maybe<&2, F32>, Some{lv}, Laws.coordinate_observe(left, a, [row, U32.from_nat(cursor)]), meaning(Unit{})), left_kept,
        coordinate_certificate(right, b, [U32.from_nat(cursor), column]))

def left_certificate(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  +row: U32, +column: U32, +cursor: Nat, +accumulator: F32,
  -rec: @a: Core.Tensor<left> -> @b: Core.Tensor<right> -> @+value: F32 ->
    Laws.SourceDot<fuel, left, right, a, b, row, column, 1n+cursor, value, Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value)>,
  -exact_cursor: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  certificate: CoordinateRead<left, a, [row, U32.from_nat(cursor)]>) ->
  Laws.SourceDot<1n+fuel, left, right, a, b, row, column, cursor, accumulator, Matrix.left_read(left, right,
    a => b => value => Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value), b, [U32.from_nat(cursor), column], accumulator,
    Core.read_coordinate(left, a, [row, U32.from_nat(cursor)]))>:
  CoordinateRead{ra, +result, +exact, meaning, left_kept} = certificate
  Laws.SourceDot{unit =>
    claim : @certificate: Laws.SourceDot<1n+fuel, left, right, a, b, row, column, cursor, accumulator, Matrix.left_read(left, right,
      a => b => value => Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value), b, [U32.from_nat(cursor), column], accumulator, (ra, result))> ->
      Laws.DotResultMeaning(1n+fuel, left, right, a, b, row, column, cursor, accumulator, Matrix.left_read(left, right,
        a => b => value => Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value), b, [U32.from_nat(cursor), column], accumulator, (ra, result))) =
      \{Laws.SourceDot: proof => proof(Unit{})}
    %exact : Laws.DotResultMeaning(1n+fuel, left, right, a, b, row, column, cursor, accumulator, Matrix.left_read(left, right,
      a => b => value => Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value), b, [U32.from_nat(cursor), column], accumulator, _))
    claim(left_pair(fuel, left, right, a, b, ra, result, row, column, cursor, accumulator, rec, exact_cursor, meaning, left_kept))}

def dot_decision(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, a: Core.Tensor<left>, -b: Core.Tensor<right>,
  +row: U32, +column: U32, +cursor: Nat, +accumulator: F32,
  -rec: @a: Core.Tensor<left> -> @b: Core.Tensor<right> -> @+value: F32 ->
    Laws.SourceDot<fuel, left, right, a, b, row, column, 1n+cursor, value, Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value)>,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)>) ->
  Laws.SourceDot<1n+fuel, left, right, a, b, row, column, cursor, accumulator, Matrix.dot_decided(left, right,
    a => b => value => Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value), a, b, row, column, cursor, U32.from_nat(cursor), accumulator, decision)>:
  match decision:
    case Shape.BFalse{equal}:
      Laws.SourceDot{unit => Unit{}}
    case Shape.BTrue{equal}:
      left_certificate(fuel, left, right, a, b, row, column, cursor, accumulator, rec,
        Lifecycle.natural_equal(U32.to_nat(U32.from_nat(cursor)), cursor, equal), coordinate_certificate(left, a, [row, U32.from_nat(cursor)]))

def Laws.successful_dot_interpretation(fuel, left, right, a, b, row, column, cursor, accumulator):
  match fuel:
    case 0n:
      Laws.SourceDot{unit => Laws.DotMeaning{Nil{}, Unit{}, {==}, other => {==}, other => {==}}}
    case 1n+tail:
      dot_decision(tail, left, right, a, b, row, column, cursor, accumulator,
        a => b => value => Laws.successful_dot_interpretation(tail, left, right, a, b, row, column, 1n+cursor, value),
        Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)))

def lt_zero_false(+value: Nat) -> {Nat.is_lt(value, 0n) == False{} : Bool}:
  match value:
    case 0n:
      {==}
    case 1n+tail:
      {==}

def no_earlier_zero(+value: Nat, earlier: {Nat.is_lt(value, 0n) == True{} : Bool}) -> Empty:
  Lifecycle.false_true(Equal.trans(Bool, True{}, Nat.is_lt(value, 0n), False{},
    Equal.sym(Bool, Nat.is_lt(value, 0n), True{}, earlier), lt_zero_false(value)))

def earlier_lift(-a: Nat, -b: Nat, split: Either<&2, &2, {Nat.is_lt(a, b) == True{} : Bool}, {a == b : Nat}>) ->
  Either<&2, &2, {Nat.is_lt(1n+a, 1n+b) == True{} : Bool}, {1n+a == 1n+b : Nat}>:
  match split:
    case Inl{earlier}:
      Inl{earlier}
    case Inr{equal}:
      Inr{Equal.cong(Nat, Nat, n => 1n+n, a, b, equal)}

def earlier_split(+a: Nat, +b: Nat, earlier: {Nat.is_lt(a, 1n+b) == True{} : Bool}) ->
  Either<&2, &2, {Nat.is_lt(a, b) == True{} : Bool}, {a == b : Nat}>:
  match a b:
    case 0n 0n:
      Inr{{==}}
    case 0n 1n+q:
      Inl{{==}}
    case 1n+p 0n:
      Empty.absurd(Either<&2, &2, {Nat.is_lt(1n+p, 0n) == True{} : Bool}, {1n+p == 0n : Nat}>, no_earlier_zero(p, earlier))
    case 1n+p 1n+q:
      earlier_lift(p, q, earlier_split(p, q, earlier))

def earlier_different(+a: Nat, +b: Nat, earlier: {Nat.is_lt(a, b) == True{} : Bool}) -> {Nat.is_eq(b, a) == False{} : Bool}:
  match a b:
    case 0n 0n:
      Empty.absurd({Nat.is_eq(0n, 0n) == False{} : Bool}, no_earlier_zero(0n, earlier))
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      Empty.absurd({Nat.is_eq(0n, 1n+p) == False{} : Bool}, no_earlier_zero(1n+p, earlier))
    case 1n+p 1n+q:
      earlier_different(p, q, earlier)

def checked_different(+position: Nat, +cursor: Nat,
  position_exact: {U32.to_nat(U32.from_nat(position)) == position : Nat}, cursor_exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  earlier: {Nat.is_lt(position, cursor) == True{} : Bool}) ->
  {Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), U32.to_nat(U32.from_nat(position))) == False{} : Bool}:
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(cursor)), cursor, cursor_exact) : {Nat.is_eq(_, U32.to_nat(U32.from_nat(position))) == False{} : Bool}
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(position)), position, position_exact) : {Nat.is_eq(cursor, _) == False{} : Bool}
  earlier_different(position, cursor, earlier)

def frame_cell(+inner: U32, +left: Shape.Shape, +right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  +output: Shape.Shape, -before: Core.Tensor<output>, -after: Core.Tensor<output>, +cursor: Nat, +position: Nat,
  cell: Laws.ProductCell<inner, left, right, a, b, output, before, position>,
  -cursor_exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat}, -earlier: {Nat.is_lt(position, cursor) == True{} : Bool},
  -frame: @+other: U32 -> @checked: Core.ObservationChecked(Core.Packed{output, before}, other) ->
    @different: {Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), U32.to_nat(other)) == False{} : Bool} ->
      Core.ObservationChecked(Core.Packed{output, after}, other) &
        {Core.checked_observe(output, after, other) == Core.checked_observe(output, before, other) : Maybe<&2, F32>}) ->
  Laws.ProductCell<inner, left, right, a, b, output, after, position>:
  Laws.ProductCell{row, column, coordinate, decoded, coordinates, word_exact, factors, logical, value, fold, checked, observed} = cell
  Laws.ProductCell{row, column, coordinate, decoded, coordinates, word_exact, factors, logical, value, fold,
    Pair.fst(Core.ObservationChecked(Core.Packed{output, after}, U32.from_nat(position)),
      {Core.checked_observe(output, after, U32.from_nat(position)) == Core.checked_observe(output, before, U32.from_nat(position)) : Maybe<&2, F32>},
      frame(U32.from_nat(position), checked, checked_different(position, cursor, word_exact, cursor_exact, earlier))),
    Equal.trans(Maybe<&2, F32>, Core.checked_observe(output, after, U32.from_nat(position)),
      Core.checked_observe(output, before, U32.from_nat(position)), Some{value},
      Pair.snd(Core.ObservationChecked(Core.Packed{output, after}, U32.from_nat(position)),
        {Core.checked_observe(output, after, U32.from_nat(position)) == Core.checked_observe(output, before, U32.from_nat(position)) : Maybe<&2, F32>},
        frame(U32.from_nat(position), checked, checked_different(position, cursor, word_exact, cursor_exact, earlier))), observed)}

def prefix_cell_case(+inner: U32, +left: Shape.Shape, +right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  +output: Shape.Shape, -out: Core.Tensor<output>, +cursor: Nat, +position: Nat,
  current: Laws.ProductCell<inner, left, right, a, b, output, out, cursor>,
  prior: @+position: Nat -> @earlier: {Nat.is_lt(position, cursor) == True{} : Bool} -> Laws.ProductCell<inner, left, right, a, b, output, out, position>,
  split: Either<&2, &2, {Nat.is_lt(position, cursor) == True{} : Bool}, {position == cursor : Nat}>) -> Laws.ProductCell<inner, left, right, a, b, output, out, position>:
  match split:
    case Inl{earlier}:
      prior(position, earlier)
    case Inr{equal}:
      %Equal.sym(Nat, position, cursor, equal) : Laws.ProductCell<inner, left, right, a, b, output, out, _>
      current

def advance_prefix(+inner: U32, +left: Shape.Shape, +right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  -before_a: Core.Tensor<left>, -before_b: Core.Tensor<right>, -after_a: Core.Tensor<left>, -after_b: Core.Tensor<right>,
  +output: Shape.Shape, -before: Core.Tensor<output>, -after: Core.Tensor<output>, +cursor: Nat,
  +row: U32, +column: U32, +coordinate: Shape.CheckedCoordinate<output>, +value: F32,
  -decoded: {Shape.unflatten_checked(output, U32.from_nat(cursor)) == Done{coordinate} : Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<output>>},
  -coordinates: {Shape.coordinate_values(output, coordinate) == [row, column] : List<&2, U32>},
  -word_exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  prefix: Laws.ProductPrefix<inner, left, right, a, b, output, before_a, before_b, before, cursor>,
  dot: Laws.DotMeaning<U32.to_nat(inner), left, right, before_a, before_b, row, column, 0n, 0.0, after_a, after_b, value>,
  update: Core.CheckedUpdateMeaning(output, before, U32.from_nat(cursor), value, (after, Done{Unit{}}))) ->
  Laws.ProductPrefix<inner, left, right, a, b, output, after_a, after_b, after, 1n+cursor>:
  Laws.ProductPrefix{left_prior, right_prior, cells} = prefix
  Laws.DotMeaning{factors, logical, fold, left_kept, right_kept} = dot
  (checked, pair) = update
  (observed, frame) = pair
  Laws.ProductPrefix{other => Equal.trans(Maybe<&2, F32>, Core.checked_observe(left, after_a, other),
      Core.checked_observe(left, before_a, other), Core.checked_observe(left, a, other), left_kept(other), left_prior(other)),
    other => Equal.trans(Maybe<&2, F32>, Core.checked_observe(right, after_b, other),
      Core.checked_observe(right, before_b, other), Core.checked_observe(right, b, other), right_kept(other), right_prior(other)),
    position => earlier => prefix_cell_case(inner, left, right, a, b, output, after, cursor, position,
      Laws.ProductCell{row, column, coordinate, decoded, coordinates, word_exact, factors,
        logical_dot_original(U32.to_nat(inner), left, right, a, b, before_a, before_b, row, column, 0n, factors, logical, left_prior, right_prior),
        value, fold, checked, observed},
      position => earlier => frame_cell(inner, left, right, a, b, output, before, after, cursor, position, cells(position, earlier), word_exact, earlier, frame),
      earlier_split(position, cursor, earlier))}

def add_successor(+fuel: Nat, +cursor: Nat) -> {Nat.add(fuel, 1n+cursor) == 1n+Nat.add(fuel, cursor) : Nat}:
  match fuel:
    case 0n:
      {==}
    case 1n+tail:
      Equal.cong(Nat, Nat, n => 1n+n, Nat.add(tail, 1n+cursor), 1n+Nat.add(tail, cursor), add_successor(tail, cursor))

def add_zero(+value: Nat) -> {Nat.add(value, 0n) == value : Nat}:
  match value:
    case 0n:
      {==}
    case 1n+tail:
      Equal.cong(Nat, Nat, n => 1n+n, Nat.add(tail, 0n), tail, add_zero(tail))

def write_product_pair(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +inner: U32, +columns: U32,
  -a: Core.Tensor<left>, -b: Core.Tensor<right>, -before_a: Core.Tensor<left>, -before_b: Core.Tensor<right>,
  -ra: Core.Tensor<left>, -rb: Core.Tensor<right>, -before: Core.Tensor<output>, out: Core.Tensor<output>, result: Result<&2, &2, Core.Error, Unit>,
  +cursor: Nat, +row: U32, +column: U32, +coordinate: Shape.CheckedCoordinate<output>, +value: F32,
  -decoded: {Shape.unflatten_checked(output, U32.from_nat(cursor)) == Done{coordinate} : Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<output>>},
  -coordinates: {Shape.coordinate_values(output, coordinate) == [row, column] : List<&2, U32>},
  -word_exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  -prefix: Laws.ProductPrefix<inner, left, right, a, b, output, before_a, before_b, before, cursor>,
  -dot: Laws.DotMeaning<U32.to_nat(inner), left, right, before_a, before_b, row, column, 0n, 0.0, ra, rb, value>,
  -update: Core.CheckedUpdateMeaning(output, before, U32.from_nat(cursor), value, (out, result)),
  -rec: @ra: Core.Tensor<left> -> @rb: Core.Tensor<right> -> @out: Core.Tensor<output> ->
    @prefix: Laws.ProductPrefix<inner, left, right, a, b, output, ra, rb, out, 1n+cursor> ->
      Laws.SourceProduct<inner, left, right, a, b, output, Nat.add(fuel, 1n+cursor),
        Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor)>) ->
  Laws.SourceProduct<inner, left, right, a, b, output, Nat.add(1n+fuel, cursor), Matrix.wrote(left, right, output,
    ra => rb => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor), ra, rb, (out, result))>:
  match result:
    case Fail{error}:
      Laws.SourceProduct{unit => Unit{}}
    case Done{Unit{}}:
      Laws.SourceProduct{unit =>
        claim : @certificate: Laws.SourceProduct<inner, left, right, a, b, output, Nat.add(fuel, 1n+cursor),
          Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor)> ->
          Laws.ProductMeaning(inner, left, right, a, b, output, Nat.add(fuel, 1n+cursor),
            Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor)) =
          \{Laws.SourceProduct: proof => proof(Unit{})}
        %add_successor(fuel, cursor) : Laws.ProductMeaning(inner, left, right, a, b, output, _,
          Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor))
        claim(rec(ra, rb, out, advance_prefix(inner, left, right, a, b, before_a, before_b, ra, rb, output, before, out, cursor,
          row, column, coordinate, value, decoded, coordinates, word_exact, prefix, dot, update)))}

def write_product_certificate(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +inner: U32, +columns: U32,
  -a: Core.Tensor<left>, -b: Core.Tensor<right>, -before_a: Core.Tensor<left>, -before_b: Core.Tensor<right>,
  -ra: Core.Tensor<left>, -rb: Core.Tensor<right>, -before: Core.Tensor<output>,
  +cursor: Nat, +row: U32, +column: U32, +coordinate: Shape.CheckedCoordinate<output>, +value: F32,
  -decoded: {Shape.unflatten_checked(output, U32.from_nat(cursor)) == Done{coordinate} : Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<output>>},
  -coordinates: {Shape.coordinate_values(output, coordinate) == [row, column] : List<&2, U32>},
  -word_exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  -prefix: Laws.ProductPrefix<inner, left, right, a, b, output, before_a, before_b, before, cursor>,
  -dot: Laws.DotMeaning<U32.to_nat(inner), left, right, before_a, before_b, row, column, 0n, 0.0, ra, rb, value>,
  -rec: @ra: Core.Tensor<left> -> @rb: Core.Tensor<right> -> @out: Core.Tensor<output> ->
    @prefix: Laws.ProductPrefix<inner, left, right, a, b, output, ra, rb, out, 1n+cursor> ->
      Laws.SourceProduct<inner, left, right, a, b, output, Nat.add(fuel, 1n+cursor),
        Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor)>,
  certificate: Core.Write<output, before, U32.from_nat(cursor), value>, -model: Core.CheckedUpdateModel<output, before, U32.from_nat(cursor), value>) ->
  Laws.SourceProduct<inner, left, right, a, b, output, Nat.add(1n+fuel, cursor), Matrix.wrote(left, right, output,
    ra => rb => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor), ra, rb,
    Core.write_flat(output, before, U32.from_nat(cursor), value))>:
  Core.Write{out, +result, +exact} = certificate
  Laws.SourceProduct{unit =>
    claim : @certificate: Laws.SourceProduct<inner, left, right, a, b, output, Nat.add(1n+fuel, cursor), Matrix.wrote(left, right, output,
      ra => rb => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor), ra, rb, (out, result))> ->
      Laws.ProductMeaning(inner, left, right, a, b, output, Nat.add(1n+fuel, cursor), Matrix.wrote(left, right, output,
        ra => rb => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor), ra, rb, (out, result))) =
      \{Laws.SourceProduct: proof => proof(Unit{})}
    prove : @model: Core.CheckedUpdateModel<output, before, U32.from_nat(cursor), value> ->
      Laws.ProductMeaning(inner, left, right, a, b, output, Nat.add(1n+fuel, cursor), Matrix.wrote(left, right, output,
        ra => rb => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor), ra, rb,
        Core.write_flat(output, before, U32.from_nat(cursor), value))) =
      \{Core.CheckedUpdateModel: meaning =>
        %exact : Laws.ProductMeaning(inner, left, right, a, b, output, Nat.add(1n+fuel, cursor), Matrix.wrote(left, right, output,
          ra => rb => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor), ra, rb, _))
        claim(write_product_pair(fuel, left, right, output, inner, columns, a, b, before_a, before_b, ra, rb, before, out, result,
          cursor, row, column, coordinate, value, decoded, coordinates, word_exact, prefix, dot,
          %Equal.sym(Core.Tensor<output> & Result<&2, &2, Core.Error, Unit>, (out, result), Core.write_flat(output, before, U32.from_nat(cursor), value), exact) :
            Core.CheckedUpdateMeaning(output, before, U32.from_nat(cursor), value, _)
          meaning, rec))}
    prove(model)}

def product_dot_result(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +inner: U32, +columns: U32,
  -a: Core.Tensor<left>, -b: Core.Tensor<right>, -before_a: Core.Tensor<left>, -before_b: Core.Tensor<right>, -out: Core.Tensor<output>,
  +cursor: Nat, +row: U32, +column: U32, +coordinate: Shape.CheckedCoordinate<output>,
  -decoded: {Shape.unflatten_checked(output, U32.from_nat(cursor)) == Done{coordinate} : Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<output>>},
  -coordinates: {Shape.coordinate_values(output, coordinate) == [row, column] : List<&2, U32>},
  -word_exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  -prefix: Laws.ProductPrefix<inner, left, right, a, b, output, before_a, before_b, out, cursor>,
  -rec: @ra: Core.Tensor<left> -> @rb: Core.Tensor<right> -> @out: Core.Tensor<output> ->
    @prefix: Laws.ProductPrefix<inner, left, right, a, b, output, ra, rb, out, 1n+cursor> ->
      Laws.SourceProduct<inner, left, right, a, b, output, Nat.add(fuel, 1n+cursor),
        Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor)>,
  result: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32>,
  -source: Laws.SourceDot<U32.to_nat(inner), left, right, before_a, before_b, row, column, 0n, 0.0, result>) ->
  Laws.SourceProduct<inner, left, right, a, b, output, Nat.add(1n+fuel, cursor), Matrix.dotted(left, right, output,
    ra => rb => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor), out, U32.from_nat(cursor), result)>:
  match result:
    case Fail{error}:
      Laws.SourceProduct{unit => Unit{}}
    case Done{pair}:
      (ra, rb, value) = pair
      Laws.SourceProduct{unit =>
        claim : @certificate: Laws.SourceProduct<inner, left, right, a, b, output, Nat.add(1n+fuel, cursor), Matrix.wrote(left, right, output,
          ra => rb => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor), ra, rb,
          Core.write_flat(output, out, U32.from_nat(cursor), value))> ->
          Laws.ProductMeaning(inner, left, right, a, b, output, Nat.add(1n+fuel, cursor), Matrix.wrote(left, right, output,
            ra => rb => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor), ra, rb,
            Core.write_flat(output, out, U32.from_nat(cursor), value))) = \{Laws.SourceProduct: proof => proof(Unit{})}
        prove : @dot: Laws.DotMeaning<U32.to_nat(inner), left, right, before_a, before_b, row, column, 0n, 0.0, ra, rb, value> ->
          Laws.ProductMeaning(inner, left, right, a, b, output, Nat.add(1n+fuel, cursor), Matrix.wrote(left, right, output,
            ra => rb => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor), ra, rb,
            Core.write_flat(output, out, U32.from_nat(cursor), value))) = dot =>
          claim(write_product_certificate(fuel, left, right, output, inner, columns, a, b, before_a, before_b, ra, rb, out,
            cursor, row, column, coordinate, value, decoded, coordinates, word_exact, prefix, dot, rec,
            Core.write_certificate(output, out, U32.from_nat(cursor), value), CoreProof.checked_update_model(output, out, U32.from_nat(cursor), value)))
        from_source : @source: Laws.SourceDot<U32.to_nat(inner), left, right, before_a, before_b, row, column, 0n, 0.0,
          (Done{(ra, rb, value)} : Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32>)> ->
          Laws.ProductMeaning(inner, left, right, a, b, output, Nat.add(1n+fuel, cursor), Matrix.wrote(left, right, output,
            ra => rb => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor), ra, rb,
            Core.write_flat(output, out, U32.from_nat(cursor), value))) = \{Laws.SourceDot: meaning => prove(meaning(Unit{}))}
        from_source(source)}

def product_coordinates_proof(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +inner: U32, +columns: U32,
  -a: Core.Tensor<left>, -b: Core.Tensor<right>, before_a: Core.Tensor<left>, before_b: Core.Tensor<right>, -out: Core.Tensor<output>,
  +cursor: Nat, +coordinate: Shape.CheckedCoordinate<output>,
  -decoded: {Shape.unflatten_checked(output, U32.from_nat(cursor)) == Done{coordinate} : Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<output>>},
  -word_exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  -prefix: Laws.ProductPrefix<inner, left, right, a, b, output, before_a, before_b, out, cursor>,
  -rec: @ra: Core.Tensor<left> -> @rb: Core.Tensor<right> -> @out: Core.Tensor<output> ->
    @prefix: Laws.ProductPrefix<inner, left, right, a, b, output, ra, rb, out, 1n+cursor> ->
      Laws.SourceProduct<inner, left, right, a, b, output, Nat.add(fuel, 1n+cursor),
        Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor)>,
  values: List<&2, U32>, -same: {values == Shape.coordinate_values(output, coordinate) : List<&2, U32>}) ->
  Laws.SourceProduct<inner, left, right, a, b, output, Nat.add(1n+fuel, cursor), Matrix.product_coordinates(left, right, output,
    ra => rb => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor), inner, before_a, before_b, out, U32.from_nat(cursor), values)>:
  match values:
    case +row <> +column <> Nil{}:
      product_dot_result(fuel, left, right, output, inner, columns, a, b, before_a, before_b, out, cursor, row, column, coordinate,
        decoded, Equal.sym(List<&2, U32>, [row, column], Shape.coordinate_values(output, coordinate), same), word_exact, prefix, rec,
        Matrix.natural_dot(U32.to_nat(inner), left, right, before_a, before_b, row, column, 0n, 0.0),
        Laws.successful_dot_interpretation(U32.to_nat(inner), left, right, before_a, before_b, row, column, 0n, 0.0))
    case Nil{}:
      Laws.SourceProduct{unit => Unit{}}
    case row <> Nil{}:
      Laws.SourceProduct{unit => Unit{}}
    case row <> column <> third <> rest:
      Laws.SourceProduct{unit => Unit{}}

def product_unflattened_proof(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +inner: U32, +columns: U32,
  -a: Core.Tensor<left>, -b: Core.Tensor<right>, before_a: Core.Tensor<left>, before_b: Core.Tensor<right>, -out: Core.Tensor<output>, +cursor: Nat,
  -word_exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  -prefix: Laws.ProductPrefix<inner, left, right, a, b, output, before_a, before_b, out, cursor>,
  -rec: @ra: Core.Tensor<left> -> @rb: Core.Tensor<right> -> @out: Core.Tensor<output> ->
    @prefix: Laws.ProductPrefix<inner, left, right, a, b, output, ra, rb, out, 1n+cursor> ->
      Laws.SourceProduct<inner, left, right, a, b, output, Nat.add(fuel, 1n+cursor),
        Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor)>,
  result: Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<output>>,
  -same: {result == Shape.unflatten_checked(output, U32.from_nat(cursor)) : Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<output>>}) ->
  Laws.SourceProduct<inner, left, right, a, b, output, Nat.add(1n+fuel, cursor), Matrix.product_unflattened(left, right, output,
    ra => rb => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor), inner, before_a, before_b, out, U32.from_nat(cursor), result)>:
  match result:
    case Fail{error}:
      Laws.SourceProduct{unit => Unit{}}
    case Done{+coordinate}:
      product_coordinates_proof(fuel, left, right, output, inner, columns, a, b, before_a, before_b, out, cursor, coordinate,
        Equal.sym(Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<output>>, Done{coordinate}, Shape.unflatten_checked(output, U32.from_nat(cursor)), same),
        word_exact, prefix, rec, Shape.coordinate_values(output, coordinate), {==})

def product_decision_proof(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +inner: U32, +columns: U32,
  -a: Core.Tensor<left>, -b: Core.Tensor<right>, before_a: Core.Tensor<left>, before_b: Core.Tensor<right>, -out: Core.Tensor<output>, +cursor: Nat,
  -prefix: Laws.ProductPrefix<inner, left, right, a, b, output, before_a, before_b, out, cursor>,
  -rec: @ra: Core.Tensor<left> -> @rb: Core.Tensor<right> -> @out: Core.Tensor<output> ->
    @prefix: Laws.ProductPrefix<inner, left, right, a, b, output, ra, rb, out, 1n+cursor> ->
      Laws.SourceProduct<inner, left, right, a, b, output, Nat.add(fuel, 1n+cursor),
        Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor)>,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)>) ->
  Laws.SourceProduct<inner, left, right, a, b, output, Nat.add(1n+fuel, cursor), Matrix.product_decided(left, right, output,
    ra => rb => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, 1n+cursor),
    inner, columns, before_a, before_b, out, cursor, U32.from_nat(cursor), decision)>:
  match decision:
    case Shape.BFalse{equal}:
      Laws.SourceProduct{unit => Unit{}}
    case Shape.BTrue{equal}:
      product_unflattened_proof(fuel, left, right, output, inner, columns, a, b, before_a, before_b, out, cursor,
        Lifecycle.natural_equal(U32.to_nat(U32.from_nat(cursor)), cursor, equal), prefix, rec, Shape.unflatten_checked(output, U32.from_nat(cursor)), {==})

def Laws.successful_product_interpretation(fuel, left, right, output, inner, columns, a, b, ra, rb, out, cursor, prefix):
  match fuel:
    case 0n:
      Laws.SourceProduct{unit => prefix}
    case 1n+tail:
      product_decision_proof(tail, left, right, output, inner, columns, a, b, ra, rb, out, cursor, prefix,
        ra => rb => out => prefix => Laws.successful_product_interpretation(tail, left, right, output, inner, columns, a, b, ra, rb, out, 1n+cursor, prefix),
        Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)))

def empty_prefix(+inner: U32, +left: Shape.Shape, +right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  +output: Shape.Shape, -out: Core.Tensor<output>) -> Laws.ProductPrefix<inner, left, right, a, b, output, a, b, out, 0n>:
  Laws.ProductPrefix{other => {==}, other => {==}, position => earlier =>
    Empty.absurd(Laws.ProductCell<inner, left, right, a, b, output, out, position>, no_earlier_zero(position, earlier))}

def created_product(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +inner: U32, +columns: U32,
  a: Core.Tensor<left>, b: Core.Tensor<right>, result: Result<&2, &1, Core.Error, Core.Tensor<output>>) ->
  Laws.SourceProduct<inner, left, right, a, b, output, Shape.logical_numel(Shape.dimensions(output)), Matrix.created(left, right, output, inner, columns, a, b, result)>:
  match result:
    case Fail{error}:
      Laws.SourceProduct{unit => Unit{}}
    case Done{out}:
      Laws.SourceProduct{unit =>
        claim : @certificate: Laws.SourceProduct<inner, left, right, a, b, output,
          Nat.add(Shape.logical_numel(Shape.dimensions(output)), 0n),
          Matrix.natural_product_loop(Shape.logical_numel(Shape.dimensions(output)), left, right, output, inner, columns, a, b, out, 0n)> ->
          Laws.ProductMeaning(inner, left, right, a, b, output, Nat.add(Shape.logical_numel(Shape.dimensions(output)), 0n),
            Matrix.natural_product_loop(Shape.logical_numel(Shape.dimensions(output)), left, right, output, inner, columns, a, b, out, 0n)) =
          \{Laws.SourceProduct: meaning => meaning(Unit{})}
        %Equal.sym(Nat, Shape.elements(output), Shape.logical_numel(Shape.dimensions(output)), Shape.elements_exact(output)) :
          Laws.ProductMeaning(inner, left, right, a, b, output, Shape.logical_numel(Shape.dimensions(output)),
            Matrix.natural_product_loop(_, left, right, output, inner, columns, a, b, out, 0n))
        %add_zero(Shape.logical_numel(Shape.dimensions(output))) : Laws.ProductMeaning(inner, left, right, a, b, output, _,
          Matrix.natural_product_loop(Shape.logical_numel(Shape.dimensions(output)), left, right, output, inner, columns, a, b, out, 0n))
        claim(Laws.successful_product_interpretation(Shape.logical_numel(Shape.dimensions(output)), left, right, output, inner, columns, a, b,
          a, b, out, 0n, empty_prefix(inner, left, right, a, b, output, out)))}

def Laws.successful_multiply_interpretation(left, right, plan, a, b):
  match plan:
    case Matrix.Plan{+output, rows, +inner, right_inner, +columns, le, re, oe, compatible}:
      created_product(left, right, output, inner, columns, a, b, Core.new(output, 0.0))

def zero_fold(+left: Shape.Shape, +right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>, +row: U32, +column: U32,
  values: List<&2, DotFold.Factors>, logical: Laws.LogicalDot(0n, left, right, a, b, row, column, 0n, values)) -> {DotFold.dot_fold(values, 0.0) == 0.0 : F32}:
  match values:
    case Nil{}:
      {==}
    case DotFold.Factors{lv, rv} <> rest:
      Empty.absurd({DotFold.dot_fold(DotFold.Factors{lv, rv} <> rest, 0.0) == 0.0 : F32}, logical)

type ZeroCell<-output: Shape.Shape, -out: Core.Tensor<output>, -position: Nat> is Data:
  ZeroCell{-observed: {Core.checked_observe(output, out, U32.from_nat(position)) == Some{0.0} : Maybe<&2, F32>}}

def zero_cell(+left: Shape.Shape, +right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>, +output: Shape.Shape,
  -out: Core.Tensor<output>, +position: Nat, cell: Laws.ProductCell<0, left, right, a, b, output, out, position>) -> ZeroCell<output, out, position>:
  Laws.ProductCell{row, column, coordinate, decoded, coordinates, word_exact, factors, logical, value, fold, checked, observed} = cell
  ZeroCell{Equal.trans(Maybe<&2, F32>, Core.checked_observe(output, out, U32.from_nat(position)), Some{value}, Some{0.0}, observed,
    Equal.cong(F32, Maybe<&2, F32>, x => Some{x}, value, 0.0,
      Equal.trans(F32, value, DotFold.dot_fold(factors, 0.0), 0.0, fold, zero_fold(left, right, a, b, row, column, factors, logical))))}

def zero_prefix(+left: Shape.Shape, +right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>, +output: Shape.Shape,
  -ra: Core.Tensor<left>, -rb: Core.Tensor<right>, -out: Core.Tensor<output>, +end: Nat,
  prefix: Laws.ProductPrefix<0, left, right, a, b, output, ra, rb, out, end>) -> Laws.ZeroPrefix<left, right, output, out, end>:
  Laws.ProductPrefix{left_kept, right_kept, cells} = prefix
  Laws.ZeroPrefix{position => earlier =>
    claim : @certificate: ZeroCell<output, out, position> -> {Core.checked_observe(output, out, U32.from_nat(position)) == Some{0.0} : Maybe<&2, F32>} =
      \{ZeroCell: observed => observed}
    claim(zero_cell(left, right, a, b, output, out, position, cells(position, earlier)))}

def zero_result(+left: Shape.Shape, +right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>, +output: Shape.Shape, -end: Nat,
  result: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>,
  -source: Laws.SourceProduct<0, left, right, a, b, output, end, result>) -> Laws.ZeroResult<left, right, output, end, result>:
  match result:
    case Fail{error}:
      Laws.ZeroResult{unit => Unit{}}
    case Done{pair}:
      (ra, rb, out) = pair
      Laws.ZeroResult{unit =>
        prove : @source: Laws.SourceProduct<0, left, right, a, b, output, end,
          (Done{(ra, rb, out)} : Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>)> ->
          Laws.ZeroPrefix<left, right, output, out, end> = \{Laws.SourceProduct: meaning =>
            zero_prefix(left, right, a, b, output, ra, rb, out, end, meaning(Unit{}))}
        prove(source)}

def Laws.zero_inner_multiply(left, right, output, rows, right_inner, columns, le, re, oe, compatible, a, b):
  zero_result(left, right, a, b, output, Shape.logical_numel(Shape.dimensions(output)),
    Matrix.multiply(left, right, Matrix.Plan{output, rows, 0, right_inner, columns, le, re, oe, compatible}, a, b),
    Laws.successful_multiply_interpretation(left, right, Matrix.Plan{output, rows, 0, right_inner, columns, le, re, oe, compatible}, a, b))

def Laws.zero_inner_dot(left, right, a, b, row, column):
  {==}

def Laws.output_dimensions(left, right, output, rows, inner, right_inner, columns, le, re, oe, compatible):
  oe

def word_cursor_exact(+fuel: Nat, +cursor: Nat, +limit: U32,
  bound: Shape.LE(Nat.add(1n+fuel, cursor), U32.to_nat(limit))) -> {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat}:
  Conversion.natural_word_exact(cursor, limit,
    Conversion.le_predecessor(cursor, U32.to_nat(limit),
      Equal.sym(Bool, True{}, Nat.is_le(1n+cursor, U32.to_nat(limit)),
        Nats.le_true(1n+cursor, U32.to_nat(limit),
          Nats.le_add_left(1n+cursor, fuel, U32.to_nat(limit),
            %MathProof.add_comm(fuel, cursor) : Shape.LE(1n+_, U32.to_nat(limit))
            bound)))))

def inside_le(+index: Nat, +limit: Nat, inside: {Nat.is_lt(index, limit) == True{} : Bool}) -> {Nat.is_le(index, limit) == True{} : Bool}:
  match index limit:
    case 0n 0n:
      Empty.absurd({True{} == True{} : Bool}, Arith.clash(inside))
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      Empty.absurd({False{} == True{} : Bool}, Arith.clash(inside))
    case 1n+p 1n+q:
      inside_le(p, q, inside)

def index_inside(+shape: Shape.Shape, +position: Nat,
  +bound: {True{} == Nat.is_lt(position, Shape.elements(shape)) : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(position)), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}:
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(position)), position,
    Conversion.natural_word_exact(position, Shape.count(shape),
      %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.elements(shape), Conversion.shape_count_exact(shape)) : {Nat.is_le(position, _) == True{} : Bool}
      inside_le(position, Shape.elements(shape), Equal.sym(Bool, True{}, Nat.is_lt(position, Shape.elements(shape)), bound)))) :
        {Nat.is_lt(_, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}
  %Shape.elements_exact(shape) : {Nat.is_lt(position, _) == True{} : Bool}
  Equal.sym(Bool, True{}, Nat.is_lt(position, Shape.elements(shape)), bound)

def flatten_done(+shape: Shape.Shape, +coordinates: List<&2, U32>,
  result: Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>,
  +fits: Coordinates.Fits(Shape.extents(shape), Shape.naturals(coordinates)),
  same: {result == Shape.flatten_checked(shape, coordinates) : Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>}) ->
  {Result.is_done(&2, &2, Shape.Error, Shape.FlatIndex<shape>, result) == True{} : Bool}:
  match result:
    case Fail{error}:
      Empty.absurd({False{} == True{} : Bool}, Arith.clash(Equal.cong(Maybe<&2, U32>, Bool, value => Maybe.is_some(&2, U32, value), None{},
        Some{U32.from_nat(Shape.row_major(Shape.extents(shape), Coordinates.coordinate_of(Shape.extents(shape), Shape.naturals(coordinates), fits)))},
        Equal.trans(Maybe<&2, U32>, None{}, ShapeLaws.index_result(shape, Shape.flatten_checked(shape, coordinates)),
          Some{U32.from_nat(Shape.row_major(Shape.extents(shape), Coordinates.coordinate_of(Shape.extents(shape), Shape.naturals(coordinates), fits)))},
          Equal.cong(Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>, Maybe<&2, U32>, ShapeLaws.index_result(shape),
            Fail{error}, Shape.flatten_checked(shape, coordinates), same),
          %Boundary.decode_fit(Shape.extents(shape), Shape.naturals(coordinates), fits) :
              {ShapeLaws.index_result(shape, Shape.flatten_checked(shape, coordinates)) == ShapeLaws.flatten_spec(Shape.extents(shape), _) : Maybe<&2, U32>}
          ShapeLaws.flatten_checked_refines(shape, coordinates)))))
    case Done{index}:
      {==}

def coordinate_read_result_status(+shape: Shape.Shape, result: Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>,
  tensor: Core.Tensor<shape>, -successful: {Result.is_done(&2, &2, Shape.Error, Shape.FlatIndex<shape>, result) == True{} : Bool}) ->
  Totality.ReadStatus<shape, Core.read_coordinate_result(shape, tensor, result)>:
  match result:
    case Fail{error}:
      Totality.ReadStatus{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{index}:
      Shape.FIndex{+position, +bound} = index
      Totality.ReadStatus{{\{TransformTotality.ReadSuccess: successful => successful} :
        TransformTotality.ReadSuccess<shape, tensor, U32.from_nat(position)> ->
          {TransformTotality.read_succeeded(shape, Core.read_flat(shape, tensor, U32.from_nat(position))) == True{} : Bool}}(
            TransformLaws.tensor_read_valid_success(shape, tensor, U32.from_nat(position), index_inside(shape, position, bound)))}

def coordinate_read_status(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +coordinates: List<&2, U32>,
  -fits: Coordinates.Fits(Shape.extents(shape), Shape.naturals(coordinates))) ->
  Totality.ReadStatus<shape, Core.read_coordinate(shape, tensor, coordinates)>:
  coordinate_read_result_status(shape, Shape.flatten_checked(shape, coordinates), tensor,
    flatten_done(shape, coordinates, Shape.flatten_checked(shape, coordinates), fits, {==}))

def dot_right_status(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +row: U32, +column: U32, +cursor: Nat,
  +left_value: F32, +accumulator: F32,
  pair: Core.Tensor<right> & Result<&2, &2, Core.Error, F32>, a: Core.Tensor<left>,
  -successful: {TransformTotality.read_succeeded(right, pair) == True{} : Bool},
  -rec: @a: Core.Tensor<left> -> @b: Core.Tensor<right> -> @+value: F32 -> Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32,
    Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value)>) ->
  Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32, Matrix.right_read(left, right,
    a => b => value => Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value), a, left_value, accumulator, pair)>:
  (b, result) = pair
  match result:
    case Fail{error}:
      Totality.Status{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{+value}:
      Totality.Status{{\{Totality.Status: successful => successful} :
        Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32,
          Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, F32.add(accumulator, F32.mul(left_value, value)))> ->
            {Result.is_done(&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32,
              Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, F32.add(accumulator, F32.mul(left_value, value)))) == True{} : Bool}}(
                rec(a, b, F32.add(accumulator, F32.mul(left_value, value))))}

def dot_left_status(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +row: U32, +column: U32, +cursor: Nat, +accumulator: F32,
  pair: Core.Tensor<left> & Result<&2, &2, Core.Error, F32>, b: Core.Tensor<right>,
  -successful: {TransformTotality.read_succeeded(left, pair) == True{} : Bool},
  -right_fits: Coordinates.Fits(Shape.extents(right), [U32.to_nat(U32.from_nat(cursor)), U32.to_nat(column)]),
  -rec: @a: Core.Tensor<left> -> @b: Core.Tensor<right> -> @+value: F32 -> Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32,
    Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value)>) ->
  Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32, Matrix.left_read(left, right,
    a => b => value => Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value), b, [U32.from_nat(cursor), column], accumulator, pair)>:
  (a, result) = pair
  match result:
    case Fail{error}:
      Totality.Status{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{+value}:
      dot_right_status(fuel, left, right, row, column, cursor, value, accumulator,
        Core.read_coordinate(right, b, [U32.from_nat(cursor), column]), a,
        {\{Totality.ReadStatus: successful => successful} : Totality.ReadStatus<right, Core.read_coordinate(right, b, [U32.from_nat(cursor), column])> ->
          {TransformTotality.read_succeeded(right, Core.read_coordinate(right, b, [U32.from_nat(cursor), column])) == True{} : Bool}}(
            coordinate_read_status(right, b, [U32.from_nat(cursor), column], right_fits)), rec)

def dot_decision_status(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +row: U32, +column: U32, +cursor: Nat, +accumulator: F32,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)>, a: Core.Tensor<left>, b: Core.Tensor<right>,
  -exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  -left_fits: Coordinates.Fits(Shape.extents(left), [U32.to_nat(row), U32.to_nat(U32.from_nat(cursor))]),
  -right_fits: Coordinates.Fits(Shape.extents(right), [U32.to_nat(U32.from_nat(cursor)), U32.to_nat(column)]),
  -rec: @a: Core.Tensor<left> -> @b: Core.Tensor<right> -> @+value: F32 -> Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32,
    Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value)>) ->
  Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32, Matrix.dot_decided(left, right,
    a => b => value => Matrix.natural_dot(fuel, left, right, a, b, row, column, 1n+cursor, value), a, b, row, column, cursor, U32.from_nat(cursor), accumulator, decision)>:
  match decision:
    case Shape.BFalse{equal}:
      Totality.Status{Empty.absurd({False{} == True{} : Bool}, MathProof.conversion_clash(U32.to_nat(U32.from_nat(cursor)), cursor, exact, equal))}
    case Shape.BTrue{equal}:
      dot_left_status(fuel, left, right, row, column, cursor, accumulator, Core.read_coordinate(left, a, [row, U32.from_nat(cursor)]), b,
        {\{Totality.ReadStatus: successful => successful} : Totality.ReadStatus<left, Core.read_coordinate(left, a, [row, U32.from_nat(cursor)])> ->
          {TransformTotality.read_succeeded(left, Core.read_coordinate(left, a, [row, U32.from_nat(cursor)])) == True{} : Bool}}(
            coordinate_read_status(left, a, [row, U32.from_nat(cursor)], left_fits)), right_fits, rec)

def dot_loop_status(fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  +row: U32, +column: U32, +cursor: Nat, +accumulator: F32, a: Core.Tensor<left>, b: Core.Tensor<right>,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  -row_inside: {Nat.is_lt(U32.to_nat(row), U32.to_nat(rows)) == True{} : Bool},
  -column_inside: {Nat.is_lt(U32.to_nat(column), U32.to_nat(columns)) == True{} : Bool},
  -bound: Shape.LE(Nat.add(fuel, cursor), U32.to_nat(inner))) ->
  Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32, Matrix.natural_dot(fuel, left, right, a, b, row, column, cursor, accumulator)>:
  match fuel:
    case 0n:
      Totality.Status{{==}}
    case 1n+tail:
      dot_decision_status(tail, left, right, row, column, cursor, accumulator,
        Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)), a, b, word_cursor_exact(tail, cursor, inner, bound),
        Coordinates.pair_fits(left, rows, inner, row, U32.from_nat(cursor), le, row_inside,
          %Equal.sym(Nat, U32.to_nat(U32.from_nat(cursor)), cursor, word_cursor_exact(tail, cursor, inner, bound)) : {Nat.is_lt(_, U32.to_nat(inner)) == True{} : Bool}
          MathProof.cursor_bounds(tail, cursor, U32.to_nat(inner), bound)),
        Coordinates.pair_fits(right, right_inner, columns, U32.from_nat(cursor), column, re,
          %compatible : {Nat.is_lt(U32.to_nat(U32.from_nat(cursor)), _) == True{} : Bool}
          %Equal.sym(Nat, U32.to_nat(U32.from_nat(cursor)), cursor, word_cursor_exact(tail, cursor, inner, bound)) : {Nat.is_lt(_, U32.to_nat(inner)) == True{} : Bool}
          MathProof.cursor_bounds(tail, cursor, U32.to_nat(inner), bound), column_inside),
        a => b => value => dot_loop_status(tail, left, right, rows, inner, right_inner, columns, row, column, 1n+cursor, value,
          a, b, le, re, compatible, row_inside, column_inside,
          %Equal.sym(Nat, Nat.add(tail, 1n+cursor), 1n+Nat.add(tail, cursor), Arith.add_succ(tail, cursor)) : Shape.LE(_, U32.to_nat(inner))
          bound))

def product_write_status(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +inner: U32, +columns: U32, +cursor: Nat,
  pair: Core.Tensor<output> & Result<&2, &2, Core.Error, Unit>, a: Core.Tensor<left>, b: Core.Tensor<right>,
  -successful: {TransformTotality.write_succeeded(output, pair) == True{} : Bool},
  -rec: @a: Core.Tensor<left> -> @b: Core.Tensor<right> -> @out: Core.Tensor<output> ->
    Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>,
      Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor)>) ->
  Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>, Matrix.wrote(left, right, output,
    a => b => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor), a, b, pair)>:
  (out, result) = pair
  match result:
    case Fail{error}:
      Totality.Status{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{unit}:
      Totality.Status{{\{Totality.Status: successful => successful} :
        Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>,
          Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor)> ->
            {Result.is_done(&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>,
              Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor)) == True{} : Bool}}(rec(a, b, out))}

def product_dot_status(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +inner: U32, +columns: U32, +cursor: Nat, +index: U32,
  result: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32>, out: Core.Tensor<output>,
  -status: Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32, result>,
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool},
  -rec: @a: Core.Tensor<left> -> @b: Core.Tensor<right> -> @out: Core.Tensor<output> ->
    Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>,
      Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor)>) ->
  Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>, Matrix.dotted(left, right, output,
    a => b => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor), out, index, result)>:
  match result:
    case Fail{error}:
      Totality.Status{{\{Totality.Status: successful => Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))} :
        Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32, Fail{error}> -> {False{} == True{} : Bool}}(status)}
    case Done{pair}:
      (a, b, value) = pair
      product_write_status(fuel, left, right, output, inner, columns, cursor, Core.write_flat(output, out, index, value), a, b,
        {\{TransformTotality.WriteSuccess: successful => successful} : TransformTotality.WriteSuccess<output, out, index, value> ->
          {TransformTotality.write_succeeded(output, Core.write_flat(output, out, index, value)) == True{} : Bool}}(
            TransformLaws.tensor_write_valid_success(output, out, index, value, inside)), rec)

def coordinate_word_inside(+value: Nat, +extent: U32, +inside: {True{} == Nat.is_lt(value, U32.to_nat(extent)) : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(value)), U32.to_nat(extent)) == True{} : Bool}:
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(value)), value,
    Conversion.natural_word_exact(value, extent, inside_le(value, U32.to_nat(extent), Equal.sym(Bool, True{}, Nat.is_lt(value, U32.to_nat(extent)), inside)))) :
      {Nat.is_lt(_, U32.to_nat(extent)) == True{} : Bool}
  Equal.sym(Bool, True{}, Nat.is_lt(value, U32.to_nat(extent)), inside)

def product_natural_coordinates_status(values: List<&2, Nat>, +fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  +rows: U32, +inner: U32, +right_inner: U32, +columns: U32, +cursor: Nat, +index: U32,
  a: Core.Tensor<left>, b: Core.Tensor<right>, out: Core.Tensor<output>,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  -fits: Coordinates.Fits([U32.to_nat(rows), U32.to_nat(columns)], values),
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool},
  -rec: @a: Core.Tensor<left> -> @b: Core.Tensor<right> -> @out: Core.Tensor<output> ->
    Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>,
      Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor)>) ->
  Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>, Matrix.product_coordinates(left, right, output,
    a => b => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor), inner, a, b, out, index, Shape.words(values))>:
  match values:
    case +row <> +column <> Nil{}:
      Totality.Status{
        {\{Coordinates.FStep: row_inside => tail =>
          {\{Coordinates.FStep: column_inside => rest =>
            {\{Totality.Status: successful => successful} :
              Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>, Matrix.dotted(left, right, output,
                a => b => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor), out, index,
                Matrix.natural_dot(U32.to_nat(inner), left, right, a, b, U32.from_nat(row), U32.from_nat(column), 0n, 0.0))> ->
                {Result.is_done(&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>, Matrix.dotted(left, right, output,
                  a => b => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor), out, index,
                  Matrix.natural_dot(U32.to_nat(inner), left, right, a, b, U32.from_nat(row), U32.from_nat(column), 0n, 0.0))) == True{} : Bool}}(
                    product_dot_status(fuel, left, right, output, inner, columns, cursor, index,
                      Matrix.natural_dot(U32.to_nat(inner), left, right, a, b, U32.from_nat(row), U32.from_nat(column), 0n, 0.0), out,
                      dot_loop_status(U32.to_nat(inner), left, right, rows, inner, right_inner, columns, U32.from_nat(row), U32.from_nat(column), 0n, 0.0,
                        a, b, le, re, compatible, coordinate_word_inside(row, rows, row_inside), coordinate_word_inside(column, columns, column_inside),
                        %Equal.sym(Nat, Nat.add(U32.to_nat(inner), 0n), U32.to_nat(inner), add_zero(U32.to_nat(inner))) : Shape.LE(_, U32.to_nat(inner))
                        Shape.LE.from(U32.to_nat(inner), U32.to_nat(inner), Equal.sym(Bool, Nat.is_le(U32.to_nat(inner), U32.to_nat(inner)), True{}, Conversion.le_reflexive(U32.to_nat(inner))))), inside, rec))} :
                          Coordinates.Fits([U32.to_nat(columns)], [column]) ->
                            {Result.is_done(&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>, Matrix.product_coordinates(left, right, output,
                              a => b => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor), inner, a, b, out, index,
                              [U32.from_nat(row), U32.from_nat(column)])) == True{} : Bool}}(tail)} :
                                Coordinates.Fits([U32.to_nat(rows), U32.to_nat(columns)], [row, column]) ->
                                  {Result.is_done(&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>, Matrix.product_coordinates(left, right, output,
                                    a => b => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor), inner, a, b, out, index,
                                    [U32.from_nat(row), U32.from_nat(column)])) == True{} : Bool}}(fits)}
    case Nil{}:
      Totality.Status{Empty.absurd({False{} == True{} : Bool}, fits)}
    case head <> Nil{}:
      Totality.Status{{\{Coordinates.FStep: bound => tail => Empty.absurd({False{} == True{} : Bool}, tail)} :
        Coordinates.Fits([U32.to_nat(rows), U32.to_nat(columns)], [head]) -> {False{} == True{} : Bool}}(fits)}
    case first <> second <> third <> tail:
      Totality.Status{{\{Coordinates.FStep: bound => rest =>
        {\{Coordinates.FStep: bound => extra => Empty.absurd({False{} == True{} : Bool}, extra)} :
          Coordinates.Fits([U32.to_nat(columns)], second <> third <> tail) -> {False{} == True{} : Bool}}(rest)} :
            Coordinates.Fits([U32.to_nat(rows), U32.to_nat(columns)], first <> second <> third <> tail) -> {False{} == True{} : Bool}}(fits)}

def product_coordinate_status(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  +rows: U32, +inner: U32, +right_inner: U32, +columns: U32, +cursor: Nat, +index: U32, coordinate: Shape.CheckedCoordinate<output>,
  a: Core.Tensor<left>, b: Core.Tensor<right>, out: Core.Tensor<output>,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>}, -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool},
  -rec: @a: Core.Tensor<left> -> @b: Core.Tensor<right> -> @out: Core.Tensor<output> ->
    Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>,
      Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor)>) ->
  Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>, Matrix.product_coordinates(left, right, output,
    a => b => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor), inner, a, b, out, index, Shape.coordinate_values(output, coordinate))>:
  Shape.CValues{+values, +valid} = coordinate
  product_natural_coordinates_status(values, fuel, left, right, output, rows, inner, right_inner, columns, cursor, index, a, b, out, le, re, compatible,
    %oe : Coordinates.Fits(Shape.naturals(_), values)
    Coordinates.fits_of(Shape.extents(output), values, valid), inside, rec)

def unflatten_done(+shape: Shape.Shape, +index: U32, result: Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<shape>>,
  same: {result == Shape.unflatten_checked(shape, index) : Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<shape>>},
  +inside: {True{} == Nat.is_lt(U32.to_nat(index), Shape.elements(shape)) : Bool}) ->
  {Result.is_done(&2, &2, Shape.Error, Shape.CheckedCoordinate<shape>, result) == True{} : Bool}:
  match result:
    case Fail{error}:
      Empty.absurd({False{} == True{} : Bool}, Arith.clash(Equal.cong(Maybe<&2, List<&2, U32>>, Bool, value => Maybe.is_some(&2, List<&2, U32>, value), None{},
        Some{Shape.words(Shape.indices(Shape.extents(shape), TransformProof.decoded_coordinate(shape, index, inside)))},
        Equal.trans(Maybe<&2, List<&2, U32>>, None{}, ShapeLaws.coordinate_result(shape, Shape.unflatten_checked(shape, index)),
          Some{Shape.words(Shape.indices(Shape.extents(shape), TransformProof.decoded_coordinate(shape, index, inside)))},
          Equal.cong(Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<shape>>, Maybe<&2, List<&2, U32>>, ShapeLaws.coordinate_result(shape), Fail{error}, Shape.unflatten_checked(shape, index), same),
          %Boundary.fin_of_from(Shape.numel(Shape.extents(shape)), U32.to_nat(index), Coordinates.size_fits(shape, U32.to_nat(index), inside)) :
            {ShapeLaws.coordinate_result(shape, Shape.unflatten_checked(shape, index)) == ShapeLaws.unflatten_spec(Shape.extents(shape), _) : Maybe<&2, List<&2, U32>>}
          ShapeLaws.unflatten_checked_refines(shape, index)))))
    case Done{coordinate}:
      {==}

def product_unflattened_status(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  +rows: U32, +inner: U32, +right_inner: U32, +columns: U32, +cursor: Nat, +index: U32,
  result: Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<output>>, a: Core.Tensor<left>, b: Core.Tensor<right>, out: Core.Tensor<output>,
  -successful: {Result.is_done(&2, &2, Shape.Error, Shape.CheckedCoordinate<output>, result) == True{} : Bool},
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>}, -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  -inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool},
  -rec: @a: Core.Tensor<left> -> @b: Core.Tensor<right> -> @out: Core.Tensor<output> ->
    Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>,
      Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor)>) ->
  Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>, Matrix.product_unflattened(left, right, output,
    a => b => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor), inner, a, b, out, index, result)>:
  match result:
    case Fail{error}:
      Totality.Status{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{coordinate}:
      product_coordinate_status(fuel, left, right, output, rows, inner, right_inner, columns, cursor, index, coordinate, a, b, out, le, re, oe, compatible, inside, rec)

def product_decision_status(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  +rows: U32, +inner: U32, +right_inner: U32, +columns: U32, +cursor: Nat,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)>, a: Core.Tensor<left>, b: Core.Tensor<right>, out: Core.Tensor<output>,
  -exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>}, -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  -inside: {Nat.is_lt(U32.to_nat(U32.from_nat(cursor)), Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool},
  -rec: @a: Core.Tensor<left> -> @b: Core.Tensor<right> -> @out: Core.Tensor<output> ->
    Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>,
      Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor)>) ->
  Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>, Matrix.product_decided(left, right, output,
    a => b => out => Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, 1n+cursor), inner, columns, a, b, out, cursor, U32.from_nat(cursor), decision)>:
  match decision:
    case Shape.BFalse{equal}:
      Totality.Status{Empty.absurd({False{} == True{} : Bool}, MathProof.conversion_clash(U32.to_nat(U32.from_nat(cursor)), cursor, exact, equal))}
    case Shape.BTrue{equal}:
      product_unflattened_status(fuel, left, right, output, rows, inner, right_inner, columns, cursor, U32.from_nat(cursor), Shape.unflatten_checked(output, U32.from_nat(cursor)), a, b, out,
        unflatten_done(output, U32.from_nat(cursor), Shape.unflatten_checked(output, U32.from_nat(cursor)), {==},
          %Equal.sym(Nat, Shape.elements(output), Shape.logical_numel(Shape.dimensions(output)), Shape.elements_exact(output)) : {True{} == Nat.is_lt(U32.to_nat(U32.from_nat(cursor)), _) : Bool}
          Equal.sym(Bool, Nat.is_lt(U32.to_nat(U32.from_nat(cursor)), Shape.logical_numel(Shape.dimensions(output))), True{}, inside)),
        le, re, oe, compatible, inside, rec)

def product_loop_status(fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  +rows: U32, +inner: U32, +right_inner: U32, +columns: U32, +cursor: Nat, a: Core.Tensor<left>, b: Core.Tensor<right>, out: Core.Tensor<output>,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>}, -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  -bound: Shape.LE(Nat.add(fuel, cursor), Shape.elements(output))) ->
  Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>, Matrix.natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, cursor)>:
  match fuel:
    case 0n:
      Totality.Status{{==}}
    case 1n+tail:
      product_decision_status(tail, left, right, output, rows, inner, right_inner, columns, cursor,
        Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)), a, b, out, MathProof.cursor_exact(tail, output, cursor, bound), le, re, oe, compatible,
        %Equal.sym(Nat, U32.to_nat(U32.from_nat(cursor)), cursor, MathProof.cursor_exact(tail, output, cursor, bound)) : {Nat.is_lt(_, Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}
        %Shape.elements_exact(output) : {Nat.is_lt(cursor, _) == True{} : Bool}
        MathProof.cursor_bounds(tail, cursor, Shape.elements(output), bound),
        a => b => out => product_loop_status(tail, left, right, output, rows, inner, right_inner, columns, 1n+cursor, a, b, out, le, re, oe, compatible,
          %Equal.sym(Nat, Nat.add(tail, 1n+cursor), 1n+Nat.add(tail, cursor), Arith.add_succ(tail, cursor)) : Shape.LE(_, Shape.elements(output))
          bound))

def created_status(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  result: Result<&2, &1, Core.Error, Core.Tensor<output>>, a: Core.Tensor<left>, b: Core.Tensor<right>,
  -done: Allocation.TensorDone<output, result>,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>}, -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat}) ->
  Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>, Matrix.created(left, right, output, inner, columns, a, b, result)>:
  match result:
    case Fail{error}:
      Totality.Status{{\{Allocation.TensorDone: tensor => exact =>
        Empty.absurd({False{} == True{} : Bool}, Shape.clash(Equal.cong(Result<&2, &1, Core.Error, Core.Tensor<output>>, Bool,
          Result.is_done(&2, &1, Core.Error, Core.Tensor<output>), Done{tensor}, Fail{error}, exact)))} :
            Allocation.TensorDone<output, Fail{error}> -> {False{} == True{} : Bool}}(done)}
    case Done{out}:
      product_loop_status(Shape.elements(output), left, right, output, rows, inner, right_inner, columns, 0n, a, b, out, le, re, oe, compatible,
        %Equal.sym(Nat, Nat.add(Shape.elements(output), 0n), Shape.elements(output), add_zero(Shape.elements(output))) : Shape.LE(_, Shape.elements(output))
        Shape.LE.from(Shape.elements(output), Shape.elements(output), Equal.sym(Bool, Nat.is_le(Shape.elements(output), Shape.elements(output)), True{}, Conversion.le_reflexive(Shape.elements(output)))))

def multiply_status(+left: Shape.Shape, +right: Shape.Shape, +plan: Matrix.Plan<left, right>, a: Core.Tensor<left>, b: Core.Tensor<right>) ->
  Totality.Status<Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<Matrix.output_shape(left, right, plan)>, Matrix.multiply(left, right, plan, a, b)>:
  match plan:
    case Matrix.Plan{+output, +rows, +inner, +right_inner, +columns, le, re, oe, compatible}:
      created_status(left, right, output, rows, inner, right_inner, columns, Core.new(output, 0.0), a, b,
        AllocationProof.core_new_done(output, 0.0), le, re, oe, compatible)

type PlanStatus<-left: Shape.Shape, -right: Shape.Shape, -result: Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>> is Data:
  PlanStatus{-successful: {Result.is_done(&2, &2, Matrix.Error, Matrix.Plan<left, right>, result) == True{} : Bool}}

def plan_output_status(+left: Shape.Shape, +right: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  result: Shape.ConstructResult<[rows, columns]>,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  -built: {ShapeLaws.built([rows, columns], result) == Some{[rows, columns]} : Maybe<&2, List<&2, U32>>}) ->
  PlanStatus<left, right, Matrix.plan_output(left, right, rows, inner, right_inner, columns, le, re, compatible, result)>:
  match result:
    case Shape.SError{error}:
      PlanStatus{Empty.absurd({False{} == True{} : Bool}, Arith.clash(Equal.cong(Maybe<&2, List<&2, U32>>, Bool,
        value => Maybe.is_some(&2, List<&2, U32>, value), None{}, Some{[rows, columns]}, built)))}
    case Shape.SBuilt{shape, preserved}:
      PlanStatus{{==}}

def plan_compatible_status(+left: Shape.Shape, +right: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))>,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat}, -supported: Shape.LE(Shape.logical_numel([rows, columns]), 1048576n)) ->
  PlanStatus<left, right, Matrix.plan_compatible(left, right, rows, inner, right_inner, columns, le, re, decision)>:
  match decision:
    case Shape.BFalse{equal}:
      PlanStatus{Empty.absurd({False{} == True{} : Bool}, MathProof.conversion_clash(U32.to_nat(inner), U32.to_nat(right_inner), compatible, equal))}
    case Shape.BTrue{equal}:
      plan_output_status(left, right, rows, inner, right_inner, columns, Shape.construct([rows, columns]), le, re,
        Lifecycle.natural_equal(U32.to_nat(inner), U32.to_nat(right_inner), equal),
        ShapeLaws.construction_complete([rows, columns], Unit{}, supported))

def first_equal(+a: U32, +b: U32, +x: U32, +y: U32, same: {[a, b] == [x, y] : List<&2, U32>}) -> {a == x : U32}:
  Equal.cong(List<&2, U32>, U32, values => Maybe.default(&2, U32, List.get(&2, U32, values, 0n), 0), [a, b], [x, y], same)

def second_equal(+a: U32, +b: U32, +x: U32, +y: U32, same: {[a, b] == [x, y] : List<&2, U32>}) -> {b == y : U32}:
  Equal.cong(List<&2, U32>, U32, values => Maybe.default(&2, U32, List.get(&2, U32, values, 1n), 0), [a, b], [x, y], same)

def actual_admission(+left: Shape.Shape, +right: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -valid: Totality.MatrixInputs<left, right>) -> Totality.MatrixBounds<rows, inner, right_inner, columns>:
  Totality.MatrixBounds{
    {\{Totality.MatrixInputs: old_rows => old_inner => old_right_inner => old_columns => old_le => old_re => compatible => supported =>
      Equal.trans(Nat, U32.to_nat(inner), U32.to_nat(old_inner), U32.to_nat(right_inner),
        Equal.cong(U32, Nat, U32.to_nat, inner, old_inner, second_equal(rows, inner, old_rows, old_inner,
          Equal.trans(List<&2, U32>, [rows, inner], Shape.dimensions(left), [old_rows, old_inner], Equal.sym(List<&2, U32>, Shape.dimensions(left), [rows, inner], le), old_le))),
        Equal.trans(Nat, U32.to_nat(old_inner), U32.to_nat(old_right_inner), U32.to_nat(right_inner), compatible,
          Equal.cong(U32, Nat, U32.to_nat, old_right_inner, right_inner, Equal.sym(U32, right_inner, old_right_inner,
            first_equal(right_inner, columns, old_right_inner, old_columns,
              Equal.trans(List<&2, U32>, [right_inner, columns], Shape.dimensions(right), [old_right_inner, old_columns], Equal.sym(List<&2, U32>, Shape.dimensions(right), [right_inner, columns], re), old_re))))))} :
                Totality.MatrixInputs<left, right> -> {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat}}(valid),
    {\{Totality.MatrixInputs: old_rows => old_inner => old_right_inner => old_columns => old_le => old_re => compatible => supported =>
      %Equal.sym(U32, rows, old_rows, first_equal(rows, inner, old_rows, old_inner,
        Equal.trans(List<&2, U32>, [rows, inner], Shape.dimensions(left), [old_rows, old_inner], Equal.sym(List<&2, U32>, Shape.dimensions(left), [rows, inner], le), old_le))) :
          Shape.LE(Shape.logical_numel([_, columns]), 1048576n)
      %Equal.sym(U32, columns, old_columns, second_equal(right_inner, columns, old_right_inner, old_columns,
        Equal.trans(List<&2, U32>, [right_inner, columns], Shape.dimensions(right), [old_right_inner, old_columns], Equal.sym(List<&2, U32>, Shape.dimensions(right), [right_inner, columns], re), old_re))) :
          Shape.LE(Shape.logical_numel([old_rows, _]), 1048576n)
      supported} : Totality.MatrixInputs<left, right> -> Shape.LE(Shape.logical_numel([rows, columns]), 1048576n)}(valid)}

def rank_clash(+left: Shape.Shape, +right: Shape.Shape, +dimensions: List<&2, U32>,
  +same: {dimensions == Shape.dimensions(left) : List<&2, U32>}, +different: {Nat.is_eq(Shape.rank_dimensions(dimensions), 2n) == False{} : Bool},
  -valid: Totality.MatrixInputs<left, right>) -> Totality.Impossible:
  Totality.Impossible{{\{Totality.MatrixInputs: rows => inner => right_inner => columns => le => re => compatible => supported =>
    Arith.clash(Equal.trans(Bool, False{}, Nat.is_eq(Shape.rank_dimensions(dimensions), 2n), True{},
      Equal.sym(Bool, Nat.is_eq(Shape.rank_dimensions(dimensions), 2n), False{}, different),
      Equal.cong(List<&2, U32>, Bool, values => Nat.is_eq(Shape.rank_dimensions(values), 2n), dimensions, [rows, inner],
        Equal.trans(List<&2, U32>, dimensions, Shape.dimensions(left), [rows, inner], same, le))))} :
          Totality.MatrixInputs<left, right> -> Empty}(valid)}

def swapped_inputs(+left: Shape.Shape, +right: Shape.Shape, -valid: Totality.MatrixInputs<left, right>) ->
  Totality.RankTwo<right>:
  Totality.RankTwo{{\{Totality.MatrixInputs: rows => inner => right_inner => columns => le => re => compatible => supported =>
    Equal.cong(List<&2, U32>, Nat, Shape.rank_dimensions, Shape.dimensions(right), [right_inner, columns], re)} :
      Totality.MatrixInputs<left, right> -> {Shape.rank_dimensions(Shape.dimensions(right)) == 2n : Nat}}(valid)}

def right_rank_clash(+left: Shape.Shape, +right: Shape.Shape, +dimensions: List<&2, U32>,
  +same: {dimensions == Shape.dimensions(right) : List<&2, U32>}, +different: {Nat.is_eq(Shape.rank_dimensions(dimensions), 2n) == False{} : Bool},
  +rank: {Shape.rank_dimensions(Shape.dimensions(right)) == 2n : Nat}) -> Empty:
  Arith.clash(Equal.trans(Bool, False{}, Nat.is_eq(Shape.rank_dimensions(dimensions), 2n), True{},
    Equal.sym(Bool, Nat.is_eq(Shape.rank_dimensions(dimensions), 2n), False{}, different),
    Equal.cong(Nat, Bool, n => Nat.is_eq(n, 2n), Shape.rank_dimensions(dimensions), 2n,
      Equal.trans(Nat, Shape.rank_dimensions(dimensions), Shape.rank_dimensions(Shape.dimensions(right)), 2n,
        Equal.cong(List<&2, U32>, Nat, Shape.rank_dimensions, dimensions, Shape.dimensions(right), same), rank))))

def plan_right_status(dimensions: List<&2, U32>, +left: Shape.Shape, +right: Shape.Shape, +rows: U32, +inner: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, -same: {dimensions == Shape.dimensions(right) : List<&2, U32>},
  -valid: Totality.MatrixInputs<left, right>) -> PlanStatus<left, right, Matrix.plan_right(left, right, rows, inner, le, dimensions, same)>:
  match dimensions:
    case +right_inner <> +columns <> Nil{}:
      PlanStatus{{\{Totality.MatrixBounds: compatible => supported =>
        {\{PlanStatus: successful => successful} : PlanStatus<left, right,
          Matrix.plan_compatible(left, right, rows, inner, right_inner, columns, le, Equal.sym(List<&2, U32>, [right_inner, columns], Shape.dimensions(right), same),
            Shape.bool_case(Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))))> ->
              {Result.is_done(&2, &2, Matrix.Error, Matrix.Plan<left, right>, Matrix.plan_right(left, right, rows, inner, le, [right_inner, columns], same)) == True{} : Bool}}(
                plan_compatible_status(left, right, rows, inner, right_inner, columns, Shape.bool_case(Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))), le,
                  Equal.sym(List<&2, U32>, [right_inner, columns], Shape.dimensions(right), same), compatible, supported))} :
                    Totality.MatrixBounds<rows, inner, right_inner, columns> -> {Result.is_done(&2, &2, Matrix.Error, Matrix.Plan<left, right>, Matrix.plan_right(left, right, rows, inner, le, [right_inner, columns], same)) == True{} : Bool}}(
                      actual_admission(left, right, rows, inner, right_inner, columns, le, Equal.sym(List<&2, U32>, [right_inner, columns], Shape.dimensions(right), same), valid))}
    case Nil{}:
      PlanStatus{Empty.absurd({False{} == True{} : Bool}, right_rank_clash(left, right, Nil{}, same, {==},
        {\{Totality.RankTwo: rank => rank} : Totality.RankTwo<right> -> {Shape.rank_dimensions(Shape.dimensions(right)) == 2n : Nat}}(swapped_inputs(left, right, valid))))}
    case head <> Nil{}:
      PlanStatus{Empty.absurd({False{} == True{} : Bool}, right_rank_clash(left, right, [head], same, {==},
        {\{Totality.RankTwo: rank => rank} : Totality.RankTwo<right> -> {Shape.rank_dimensions(Shape.dimensions(right)) == 2n : Nat}}(swapped_inputs(left, right, valid))))}
    case first <> second <> third <> rest:
      PlanStatus{Empty.absurd({False{} == True{} : Bool}, right_rank_clash(left, right, first <> second <> third <> rest, same, {==},
        {\{Totality.RankTwo: rank => rank} : Totality.RankTwo<right> -> {Shape.rank_dimensions(Shape.dimensions(right)) == 2n : Nat}}(swapped_inputs(left, right, valid))))}

def plan_left_status(dimensions: List<&2, U32>, +left: Shape.Shape, +right: Shape.Shape,
  -same: {dimensions == Shape.dimensions(left) : List<&2, U32>}, -valid: Totality.MatrixInputs<left, right>) ->
  PlanStatus<left, right, Matrix.plan_left(left, right, dimensions, same)>:
  match dimensions:
    case +rows <> +inner <> Nil{}:
      plan_right_status(Shape.dimensions(right), left, right, rows, inner, Equal.sym(List<&2, U32>, [rows, inner], Shape.dimensions(left), same), {==}, valid)
    case Nil{}:
      PlanStatus{Empty.absurd({False{} == True{} : Bool}, {\{Totality.Impossible: empty => empty} : Totality.Impossible -> Empty}(rank_clash(left, right, Nil{}, same, {==}, valid)))}
    case head <> Nil{}:
      PlanStatus{Empty.absurd({False{} == True{} : Bool}, {\{Totality.Impossible: empty => empty} : Totality.Impossible -> Empty}(rank_clash(left, right, [head], same, {==}, valid)))}
    case first <> second <> third <> rest:
      PlanStatus{Empty.absurd({False{} == True{} : Bool}, {\{Totality.Impossible: empty => empty} : Totality.Impossible -> Empty}(rank_clash(left, right, first <> second <> third <> rest, same, {==}, valid)))}

def plan_status(+left: Shape.Shape, +right: Shape.Shape, -valid: Totality.MatrixInputs<left, right>) -> PlanStatus<left, right, Matrix.plan(left, right)>:
  plan_left_status(Shape.dimensions(left), left, right, {==}, valid)

def plan_left_rank(+left: Shape.Shape, +right: Shape.Shape, +dims: List<&2, U32>,
  -same: {dims == Shape.dimensions(left) : List<&2, U32>},
  not_matrix: {Shape.rank_dimensions(dims) != 2n : Nat}) ->
  {Matrix.plan_left(left, right, dims, same) == Fail{Matrix.RankMismatch{}} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>}:
  match dims:
    case Nil{}:
      {==}
    case +x <> Nil{}:
      {==}
    case +rows <> +inner <> Nil{}:
      Empty.absurd({Matrix.plan_left(left, right, [rows, inner], same) == Fail{Matrix.RankMismatch{}} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>},
        not_matrix({==}))
    case +x <> +y <> +z <> +t:
      {==}

def Laws.matmul_left_rank_error(left, right, a, b, not_matrix):
  %Equal.sym(Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>, Matrix.plan_left(left, right, Shape.dimensions(left), {==}), Fail{Matrix.RankMismatch{}},
    plan_left_rank(left, right, Shape.dimensions(left), {==}, not_matrix)) :
    {Matrix.parallel_planned(left, right, a, b, _) == Fail{Matrix.RankMismatch{}} : Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}
  {==}

def plan_right_rank(-left: Shape.Shape, +right: Shape.Shape, +rows: U32, +inner: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, +dims: List<&2, U32>,
  -same: {dims == Shape.dimensions(right) : List<&2, U32>},
  not_matrix: {Shape.rank_dimensions(dims) != 2n : Nat}) ->
  {Matrix.plan_right(left, right, rows, inner, le, dims, same) == Fail{Matrix.RankMismatch{}} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>}:
  match dims:
    case Nil{}:
      {==}
    case +x <> Nil{}:
      {==}
    case +right_inner <> +columns <> Nil{}:
      Empty.absurd({Matrix.plan_right(left, right, rows, inner, le, [right_inner, columns], same) == Fail{Matrix.RankMismatch{}} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>},
        not_matrix({==}))
    case +x <> +y <> +z <> +t:
      {==}

def plan_left_right_rank(+left: Shape.Shape, +right: Shape.Shape, +dims: List<&2, U32>,
  -same: {dims == Shape.dimensions(left) : List<&2, U32>},
  not_matrix: {Shape.rank_dimensions(Shape.dimensions(right)) != 2n : Nat}) ->
  {Matrix.plan_left(left, right, dims, same) == Fail{Matrix.RankMismatch{}} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>}:
  match dims:
    case Nil{}:
      {==}
    case +x <> Nil{}:
      {==}
    case +rows <> +inner <> Nil{}:
      plan_right_rank(left, right, rows, inner, Equal.sym(List<&2, U32>, [rows, inner], Shape.dimensions(left), same),
        Shape.dimensions(right), {==}, not_matrix)
    case +x <> +y <> +z <> +t:
      {==}

def Laws.matmul_right_rank_error(left, right, a, b, not_matrix):
  %Equal.sym(Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>, Matrix.plan_left(left, right, Shape.dimensions(left), {==}), Fail{Matrix.RankMismatch{}},
    plan_left_right_rank(left, right, Shape.dimensions(left), {==}, not_matrix)) :
    {Matrix.parallel_planned(left, right, a, b, _) == Fail{Matrix.RankMismatch{}} : Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}
  {==}

def compatible_inner(-left: Shape.Shape, -right: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
  -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  mismatch: {U32.to_nat(inner) != U32.to_nat(right_inner) : Nat},
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))>) ->
  {Matrix.plan_compatible(left, right, rows, inner, right_inner, columns, le, re, decision) == Fail{Matrix.InnerDimensionMismatch{}} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>}:
  match decision:
    case Shape.BFalse{equal}:
      {==}
    case Shape.BTrue{equal}:
      Empty.absurd({Matrix.plan_compatible(left, right, rows, inner, right_inner, columns, le, re, Shape.BTrue{equal}) == Fail{Matrix.InnerDimensionMismatch{}} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>},
        mismatch(Lifecycle.natural_equal(U32.to_nat(inner), U32.to_nat(right_inner), equal)))

def first_nat(values: List<&2, U32>) -> Nat:
  match values:
    case first <> rest:
      U32.to_nat(first)
    case Nil{}:
      0n

def second_nat(values: List<&2, U32>) -> Nat:
  match values:
    case first <> second <> rest:
      U32.to_nat(second)
    case rest:
      0n

def inner_right(-left: Shape.Shape, +right: Shape.Shape, +rows: U32, +inner: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, +dims: List<&2, U32>,
  -same: {dims == Shape.dimensions(right) : List<&2, U32>},
  -wanted_inner: U32, -right_inner: U32, -columns: U32,
  right_matrix: {dims == [right_inner, columns] : List<&2, U32>},
  inner_same: {U32.to_nat(inner) == U32.to_nat(wanted_inner) : Nat},
  mismatch: {U32.to_nat(wanted_inner) != U32.to_nat(right_inner) : Nat}) ->
  {Matrix.plan_right(left, right, rows, inner, le, dims, same) == Fail{Matrix.InnerDimensionMismatch{}} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>}:
  match dims:
    case +ri <> +c <> Nil{}:
      compatible_inner(left, right, rows, inner, ri, c, le, Equal.sym(List<&2, U32>, [ri, c], Shape.dimensions(right), same),
        equal => mismatch(Equal.trans(Nat, U32.to_nat(wanted_inner), U32.to_nat(inner), U32.to_nat(right_inner),
          Equal.sym(Nat, U32.to_nat(inner), U32.to_nat(wanted_inner), inner_same),
          Equal.trans(Nat, U32.to_nat(inner), U32.to_nat(ri), U32.to_nat(right_inner), equal,
            Equal.cong(List<&2, U32>, Nat, first_nat, [ri, c], [right_inner, columns], right_matrix)))),
        Shape.bool_case(Nat.is_eq(U32.to_nat(inner), U32.to_nat(ri))))
    case Nil{}:
      Empty.absurd({Matrix.plan_right(left, right, rows, inner, le, Nil{}, same) == Fail{Matrix.InnerDimensionMismatch{}} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>},
        Arith.clash(Equal.cong(List<&2, U32>, Bool, xs => Nat.is_eq(Shape.rank_dimensions(xs), 2n), Nil{}, [right_inner, columns], right_matrix)))
    case +x <> Nil{}:
      Empty.absurd({Matrix.plan_right(left, right, rows, inner, le, [x], same) == Fail{Matrix.InnerDimensionMismatch{}} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>},
        Arith.clash(Equal.cong(List<&2, U32>, Bool, xs => Nat.is_eq(Shape.rank_dimensions(xs), 2n), [x], [right_inner, columns], right_matrix)))
    case +x <> +y <> +z <> +t:
      Empty.absurd({Matrix.plan_right(left, right, rows, inner, le, x <> y <> z <> t, same) == Fail{Matrix.InnerDimensionMismatch{}} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>},
        Arith.clash(Equal.cong(List<&2, U32>, Bool, xs => Nat.is_eq(Shape.rank_dimensions(xs), 2n), x <> y <> z <> t, [right_inner, columns], right_matrix)))

def inner_left(+left: Shape.Shape, +right: Shape.Shape, +dims: List<&2, U32>,
  -same: {dims == Shape.dimensions(left) : List<&2, U32>},
  -rows: U32, -inner: U32, -right_inner: U32, -columns: U32,
  left_matrix: {dims == [rows, inner] : List<&2, U32>},
  right_matrix: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  mismatch: {U32.to_nat(inner) != U32.to_nat(right_inner) : Nat}) ->
  {Matrix.plan_left(left, right, dims, same) == Fail{Matrix.InnerDimensionMismatch{}} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>}:
  match dims:
    case +r <> +i <> Nil{}:
      inner_right(left, right, r, i, Equal.sym(List<&2, U32>, [r, i], Shape.dimensions(left), same), Shape.dimensions(right), {==},
        inner, right_inner, columns, right_matrix,
        Equal.cong(List<&2, U32>, Nat, second_nat, [r, i], [rows, inner], left_matrix), mismatch)
    case Nil{}:
      Empty.absurd({Matrix.plan_left(left, right, Nil{}, same) == Fail{Matrix.InnerDimensionMismatch{}} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>},
        Arith.clash(Equal.cong(List<&2, U32>, Bool, xs => Nat.is_eq(Shape.rank_dimensions(xs), 2n), Nil{}, [rows, inner], left_matrix)))
    case +x <> Nil{}:
      Empty.absurd({Matrix.plan_left(left, right, [x], same) == Fail{Matrix.InnerDimensionMismatch{}} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>},
        Arith.clash(Equal.cong(List<&2, U32>, Bool, xs => Nat.is_eq(Shape.rank_dimensions(xs), 2n), [x], [rows, inner], left_matrix)))
    case +x <> +y <> +z <> +t:
      Empty.absurd({Matrix.plan_left(left, right, x <> y <> z <> t, same) == Fail{Matrix.InnerDimensionMismatch{}} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>},
        Arith.clash(Equal.cong(List<&2, U32>, Bool, xs => Nat.is_eq(Shape.rank_dimensions(xs), 2n), x <> y <> z <> t, [rows, inner], left_matrix)))

def Laws.matmul_inner_error(left, right, a, b, rows, inner, right_inner, columns, left_matrix, right_matrix, mismatch):
  %Equal.sym(Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>, Matrix.plan_left(left, right, Shape.dimensions(left), {==}), Fail{Matrix.InnerDimensionMismatch{}},
    inner_left(left, right, Shape.dimensions(left), {==}, rows, inner, right_inner, columns, left_matrix, right_matrix, mismatch)) :
    {Matrix.parallel_planned(left, right, a, b, _) == Fail{Matrix.InnerDimensionMismatch{}} : Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}
  {==}

# The parallel product meets the same interpretation as the sequential one.

# What `take` hands back for a tensor: its kept clone, as a tensor of the same
# shape, and the copy the build reads.
def KeptCopy(+shape: Shape.Shape, -a: Core.Tensor<shape>, -Q: Core.Dynamic & Array<F32> -> Type) -> Type:
  @-kept: Core.Tensor<shape> -> @-copy: Array<F32> ->
  @-taken: Parallel.Taken<Core.Packed{shape, a}, (Core.Packed{shape, kept}, copy)> ->
    Q((Core.Packed{shape, kept}, copy))

def take_owned_lift(+shape: Shape.Shape, +depth: Nat, +length: Nat, -array: Array<F32>, pair: Array<F32> & Array<F32>,
  -same: {pair == Array.clone(F32, array) : Array<F32> & Array<F32>}, -balanced: Model.Balanced(F32, depth, array),
  -valid: {Storage.bounds(depth, length) == True{} : Bool}, -exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  -Q: Core.Dynamic & Array<F32> -> Type,
  h: KeptCopy(shape, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, Q)) ->
  Q(Elementwise.take_owned(shape, depth, length, array, pair, same, balanced, valid, exact)):
  (kept, copy) = pair
  h(Core.TTensor{Storage.Owned{depth, length, kept, Certificates.clone_first(F32, depth, array, (kept, copy), same, balanced), valid}, exact})(copy)(
    Parallel.taken_owned(shape, depth, length, array, (kept, copy), same, balanced, valid, exact))

def take_lift(+shape: Shape.Shape, a: Core.Tensor<shape>, -Q: Core.Dynamic & Array<F32> -> Type, h: KeptCopy(shape, a, Q)) ->
  Q(Elementwise.take(Core.Packed{shape, a})):
  Core.TTensor{storage, exact} = a
  match storage:
    case Storage.Empty{}:
      h(Core.TTensor{Storage.Empty{}, exact})(Array.new(F32, 0n, 0.0))(Parallel.taken(Core.Packed{shape, Core.TTensor{Storage.Empty{}, exact}}))
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      take_owned_lift(shape, depth, length, array, Array.clone(F32, array), {==}, balanced, valid, exact, Q, h)

def inspected(found: ElementwiseTotality.Inspection) -> Maybe<&2, F32>:
  match found:
    case ElementwiseTotality.Dims{dimensions}:
      None{}
    case ElementwiseTotality.Value{value}:
      value

def kept_observe(+shape: Shape.Shape, -a: Core.Tensor<shape>, -kept: Core.Tensor<shape>,
  -same: ElementwiseTotality.SameSource(Core.Packed{shape, a}, Core.Packed{shape, kept}), +other: U32) ->
  Parallel.Fact<{Core.checked_observe(shape, kept, other) == Core.checked_observe(shape, a, other) : Maybe<&2, F32>}>:
  Parallel.Fact{Equal.cong(ElementwiseTotality.Inspection, Maybe<&2, F32>, inspected,
    ElementwiseTotality.inspect(ElementwiseTotality.Checked{other}, Core.Packed{shape, kept}),
    ElementwiseTotality.inspect(ElementwiseTotality.Checked{other}, Core.Packed{shape, a}), same(ElementwiseTotality.Checked{other}))}

def taken_keeps(+shape: Shape.Shape, -a: Core.Tensor<shape>, -kept: Core.Tensor<shape>, -copy: Array<F32>,
  -taken: Parallel.Taken<Core.Packed{shape, a}, (Core.Packed{shape, kept}, copy)>, +other: U32) ->
  Parallel.Fact<{Core.checked_observe(shape, kept, other) == Core.checked_observe(shape, a, other) : Maybe<&2, F32>}>:
  kept_observe(shape, a, kept, {\{Parallel.Taken: depth => balanced => reads => retained => retained} :
    Parallel.Taken<Core.Packed{shape, a}, (Core.Packed{shape, kept}, copy)> -> ElementwiseTotality.SameSource(Core.Packed{shape, a}, Core.Packed{shape, kept})}(taken), other)

# A read through a flat index the shape decodes is the checked read there.
def coordinate_read(+shape: Shape.Shape, -tensor: Core.Tensor<shape>, -index: U32,
  result: Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>,
  found: {ShapeLaws.index_result(shape, result) == Some{index} : Maybe<&2, U32>}) ->
  {Laws.coordinate_observe_result(shape, tensor, result) == Core.checked_observe(shape, tensor, index) : Maybe<&2, F32>}:
  match result:
    case Fail{error}:
      Empty.absurd({Laws.coordinate_observe_result(shape, tensor, Fail{error}) == Core.checked_observe(shape, tensor, index) : Maybe<&2, F32>},
        Arith.clash(Equal.cong(Maybe<&2, U32>, Bool, value => Maybe.is_some(&2, U32, value), None{}, Some{index}, found)))
    case Done{flat}:
      %MatrixIndex.some_equal(U32, Shape.index_value(shape, flat), index, found) :
        {Core.checked_observe(shape, tensor, Shape.index_value(shape, flat)) == Core.checked_observe(shape, tensor, _) : Maybe<&2, F32>}
      {==}

def left_flat(+left: Shape.Shape, +rows: U32, +inner: U32, +row: Nat, +k: Nat,
  +le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
  +row_below: {Nat.is_lt(row, U32.to_nat(rows)) == True{} : Bool}, +below: {Nat.is_lt(k, U32.to_nat(inner)) == True{} : Bool}) ->
  {ShapeLaws.index_result(left, Shape.flatten_checked(left, [U32.from_nat(row), U32.from_nat(k)])) ==
    Some{Matrix.left_index(U32.to_nat(inner), row, k)} : Maybe<&2, U32>}:
  +k_exact = MatrixIndex.word_below(k, inner, below)
  +row_exact = MatrixIndex.word_below(row, rows, row_below)
  Equal.trans(Maybe<&2, U32>, ShapeLaws.index_result(left, Shape.flatten_checked(left, [U32.from_nat(row), U32.from_nat(k)])),
    Some{U32.from_nat(Nat.add(Nat.mul(U32.to_nat(U32.from_nat(row)), U32.to_nat(inner)), U32.to_nat(U32.from_nat(k))))},
    Some{U32.from_nat(Nat.add(Nat.mul(row, U32.to_nat(inner)), k))},
    MatrixIndex.flat_pair(left, rows, inner, U32.from_nat(row), U32.from_nat(k), le,
      MatrixIndex.lt_same(U32.to_nat(U32.from_nat(row)), row, U32.to_nat(rows), row_exact, row_below),
      MatrixIndex.lt_same(U32.to_nat(U32.from_nat(k)), k, U32.to_nat(inner), k_exact, below)),
    MatrixIndex.index_same(U32.to_nat(U32.from_nat(row)), U32.to_nat(U32.from_nat(k)), row, k, U32.to_nat(inner), row_exact, k_exact))

def right_flat(+right: Shape.Shape, +inner: U32, +right_inner: U32, +columns: U32, +column: Nat, +k: Nat,
  +re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  +compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  +column_below: {Nat.is_lt(column, U32.to_nat(columns)) == True{} : Bool}, +below: {Nat.is_lt(k, U32.to_nat(inner)) == True{} : Bool}) ->
  {ShapeLaws.index_result(right, Shape.flatten_checked(right, [U32.from_nat(k), U32.from_nat(column)])) ==
    Some{Matrix.right_index(U32.to_nat(columns), column, k)} : Maybe<&2, U32>}:
  +k_exact = MatrixIndex.word_below(k, inner, below)
  +column_exact = MatrixIndex.word_below(column, columns, column_below)
  Equal.trans(Maybe<&2, U32>, ShapeLaws.index_result(right, Shape.flatten_checked(right, [U32.from_nat(k), U32.from_nat(column)])),
    Some{U32.from_nat(Nat.add(Nat.mul(U32.to_nat(U32.from_nat(k)), U32.to_nat(columns)), U32.to_nat(U32.from_nat(column))))},
    Some{U32.from_nat(Nat.add(Nat.mul(k, U32.to_nat(columns)), column))},
    MatrixIndex.flat_pair(right, right_inner, columns, U32.from_nat(k), U32.from_nat(column), re,
      MatrixIndex.lt_same(U32.to_nat(U32.from_nat(k)), k, U32.to_nat(right_inner), k_exact,
        MatrixIndex.lt_bound(k, U32.to_nat(inner), U32.to_nat(right_inner), compatible, below)),
      MatrixIndex.lt_same(U32.to_nat(U32.from_nat(column)), column, U32.to_nat(columns), column_exact, column_below)),
    MatrixIndex.index_same(U32.to_nat(U32.from_nat(k)), U32.to_nat(U32.from_nat(column)), k, column, U32.to_nat(columns), k_exact, column_exact))

def left_extents(+left: Shape.Shape, +rows: U32, +inner: U32, +le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}) ->
  {Shape.extents(left) == [U32.to_nat(rows), U32.to_nat(inner)] : List<&2, Nat>}:
  Equal.cong(List<&2, U32>, List<&2, Nat>, dims => Shape.naturals(dims), Shape.dimensions(left), [rows, inner], le)

def right_extents(+right: Shape.Shape, +inner: U32, +right_inner: U32, +columns: U32,
  +re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  +compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat}) ->
  {Shape.extents(right) == [U32.to_nat(inner), U32.to_nat(columns)] : List<&2, Nat>}:
  Equal.trans(List<&2, Nat>, Shape.extents(right), [U32.to_nat(right_inner), U32.to_nat(columns)], [U32.to_nat(inner), U32.to_nat(columns)],
    Equal.cong(List<&2, U32>, List<&2, Nat>, dims => Shape.naturals(dims), Shape.dimensions(right), [right_inner, columns], re),
    Equal.cong(Nat, List<&2, Nat>, n => [n, U32.to_nat(columns)], U32.to_nat(right_inner), U32.to_nat(inner),
      Equal.sym(Nat, U32.to_nat(inner), U32.to_nat(right_inner), compatible)))

# The factor pairs the parallel build reads from its copies are the
# coordinate reads of the inputs the copies came from.
def logical_dot(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  +row: Nat, +column: Nat, +le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
  +re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  +compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  +row_below: {Nat.is_lt(row, U32.to_nat(rows)) == True{} : Bool},
  +column_below: {Nat.is_lt(column, U32.to_nat(columns)) == True{} : Bool},
  -a: Core.Tensor<left>, -b: Core.Tensor<right>, +dl: Nat, +dr: Nat, -cl: Array<F32>, -cr: Array<F32>,
  -left_reads: Parallel.SourceReads(Core.Packed{left, a}, dl, cl), -right_reads: Parallel.SourceReads(Core.Packed{right, b}, dr, cr),
  +cursor: Nat, +steps: {Nat.add(cursor, fuel) == U32.to_nat(inner) : Nat}, values: List<&2, DotFold.Factors>,
  -arr: MatmulParallel.LogicalArr(fuel, dl, dr, cl, cr, U32.to_nat(inner), U32.to_nat(columns), row, column, cursor, values)) ->
  Parallel.Fact<Laws.LogicalDot(fuel, left, right, a, b, U32.from_nat(row), U32.from_nat(column), cursor, values)>:
  match fuel values:
    case 0n Nil{}:
      Parallel.Fact{Unit{}}
    case 1n+tail DotFold.Factors{lv, rv} <> rest:
      +below = MatrixIndex.cursor_below(cursor, tail, U32.to_nat(inner), steps)
      +left_at = Matrix.left_index(U32.to_nat(inner), row, cursor)
      +right_at = Matrix.right_index(U32.to_nat(columns), column, cursor)
      +left_inside = MatrixIndex.word_inside(Nat.add(Nat.mul(row, U32.to_nat(inner)), cursor), left,
        MatrixIndex.left_below(U32.to_nat(rows), U32.to_nat(inner), row, row_below, left, left_extents(left, rows, inner, le), cursor, below))
      +right_inside = MatrixIndex.word_inside(Nat.add(Nat.mul(cursor, U32.to_nat(columns)), column), right,
        MatrixIndex.right_below(U32.to_nat(inner), U32.to_nat(columns), column, column_below, right,
          right_extents(right, inner, right_inner, columns, re, compatible), cursor, below))
      Parallel.Fact{{\{MatmulParallel.ArrStep: left_observed => right_observed => remaining =>
        Laws.DotStep{MatrixIndex.word_below(cursor, inner, below),
          Equal.trans(Maybe<&2, F32>, Laws.coordinate_observe(left, a, [U32.from_nat(row), U32.from_nat(cursor)]),
            Core.checked_observe(left, a, left_at), Some{lv},
            coordinate_read(left, a, left_at, Shape.flatten_checked(left, [U32.from_nat(row), U32.from_nat(cursor)]),
              left_flat(left, rows, inner, row, cursor, le, row_below, below)),
            Equal.trans(Maybe<&2, F32>, Core.checked_observe(left, a, left_at), Model.observe(F32, dl, cl, Refinement.route(dl, left_at)), Some{lv},
              Equal.sym(Maybe<&2, F32>, Model.observe(F32, dl, cl, Refinement.route(dl, left_at)), Core.checked_observe(left, a, left_at),
                left_reads(left_at)(left_inside)),
              left_observed)),
          Equal.trans(Maybe<&2, F32>, Laws.coordinate_observe(right, b, [U32.from_nat(cursor), U32.from_nat(column)]),
            Core.checked_observe(right, b, right_at), Some{rv},
            coordinate_read(right, b, right_at, Shape.flatten_checked(right, [U32.from_nat(cursor), U32.from_nat(column)]),
              right_flat(right, inner, right_inner, columns, column, cursor, re, compatible, column_below, below)),
            Equal.trans(Maybe<&2, F32>, Core.checked_observe(right, b, right_at), Model.observe(F32, dr, cr, Refinement.route(dr, right_at)), Some{rv},
              Equal.sym(Maybe<&2, F32>, Model.observe(F32, dr, cr, Refinement.route(dr, right_at)), Core.checked_observe(right, b, right_at),
                right_reads(right_at)(right_inside)),
              right_observed)),
          {\{Parallel.Fact: p => p} : Parallel.Fact<Laws.LogicalDot(tail, left, right, a, b, U32.from_nat(row), U32.from_nat(column), 1n+cursor, rest)> ->
            Laws.LogicalDot(tail, left, right, a, b, U32.from_nat(row), U32.from_nat(column), 1n+cursor, rest)}(
            logical_dot(tail, left, right, rows, inner, right_inner, columns, row, column, le, re, compatible, row_below, column_below, a, b, dl, dr, cl, cr, left_reads, right_reads,
              1n+cursor, MatrixIndex.steps_next(cursor, tail, U32.to_nat(inner), steps), rest, remaining))}} :
        MatmulParallel.ArrStep<tail, dl, dr, cl, cr, U32.to_nat(inner), U32.to_nat(columns), row, column, cursor, lv, rv, rest> ->
        Laws.DotStep<tail, left, right, a, b, U32.from_nat(row), U32.from_nat(column), cursor, lv, rv, rest>}(arr)}
    case 0n DotFold.Factors{lv, rv} <> rest:
      Parallel.Fact{Empty.absurd(Laws.LogicalDot(0n, left, right, a, b, U32.from_nat(row), U32.from_nat(column), cursor, DotFold.Factors{lv, rv} <> rest), arr)}
    case 1n+tail Nil{}:
      Parallel.Fact{Empty.absurd(Laws.LogicalDot(1n+tail, left, right, a, b, U32.from_nat(row), U32.from_nat(column), cursor, Nil{}), arr)}

type Decoded<-output: Shape.Shape, -index: U32, -row: U32, -column: U32> is Data:
  Decoded{-coordinate: Shape.CheckedCoordinate<output>,
    -decoded: {Shape.unflatten_checked(output, index) == Done{coordinate} : Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<output>>},
    -coordinates: {Shape.coordinate_values(output, coordinate) == [row, column] : List<&2, U32>}}

def decoded_of(+output: Shape.Shape, +index: U32, +row: U32, +column: U32,
  result: Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<output>>,
  -same: {result == Shape.unflatten_checked(output, index) : Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<output>>},
  -found: {ShapeLaws.coordinate_result(output, result) == Some{[row, column]} : Maybe<&2, List<&2, U32>>}) ->
  Parallel.Fact<Decoded<output, index, row, column>>:
  match result:
    case Fail{error}:
      Parallel.Fact{Empty.absurd(Decoded<output, index, row, column>,
        Arith.clash(Equal.cong(Maybe<&2, List<&2, U32>>, Bool, value => Maybe.is_some(&2, List<&2, U32>, value), None{}, Some{[row, column]}, found)))}
    case Done{coordinate}:
      Parallel.Fact{Decoded{coordinate,
        Equal.sym(Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<output>>, Done{coordinate}, Shape.unflatten_checked(output, index), same),
        MatrixIndex.some_equal(List<&2, U32>, Shape.coordinate_values(output, coordinate), [row, column], found)}}

# Below the output's element count, the parallel build writes the cell the
# sequential interpretation names.
def product_cell(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>}, -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  +depth: Nat, +length: Nat, -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(output)) : Nat},
  -a: Core.Tensor<left>, -b: Core.Tensor<right>, -dl: Nat, -dr: Nat, -cl: Array<F32>, -cr: Array<F32>,
  -bl: Model.Balanced(F32, dl, cl), -br: Model.Balanced(F32, dr, cr),
  -left_reads: Parallel.SourceReads(Core.Packed{left, a}, dl, cl), -right_reads: Parallel.SourceReads(Core.Packed{right, b}, dr, cr),
  +position: Nat, -earlier: {Nat.is_lt(position, Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}) ->
  Parallel.Fact<Laws.ProductCell<inner, left, right, a, b, output,
    Core.TTensor{Storage.Owned{depth, length,
      Matrix.build(depth, Matrix.product_splits(depth, U32.to_nat(inner)), U32.to_nat(inner), U32.to_nat(columns), length, 0n, cl, cr),
      Matrix.build_balanced(depth, Matrix.product_splits(depth, U32.to_nat(inner)), U32.to_nat(inner), U32.to_nat(columns), length, 0n, cl, cr),
      valid}, exact}, position>>:
  -inner_n = U32.to_nat(inner)
  -columns_n = U32.to_nat(columns)
  -row = Matrix.row_of(columns_n, position)
  -column = Matrix.column_of(columns_n, position)
  -index = U32.from_nat(position)
  -extents = Equal.cong(List<&2, U32>, List<&2, Nat>, dims => Shape.naturals(dims), Shape.dimensions(output), [rows, columns], oe)
  -fit = MatrixIndex.position_fits(output, U32.to_nat(rows), columns_n, extents, position, earlier)
  -row_below = Arith.and_left(Nat.is_lt(row, U32.to_nat(rows)), Bool.and(Nat.is_lt(column, columns_n), True{}), fit)
  -column_below = Arith.and_left(Nat.is_lt(column, columns_n), True{},
    Arith.and_right(Nat.is_lt(row, U32.to_nat(rows)), Bool.and(Nat.is_lt(column, columns_n), True{}), fit))
  -word_exact = MatrixIndex.word_exact(position, output, earlier)
  -below = MatrixIndex.lt_bound(position, Shape.logical_numel(Shape.dimensions(output)), length,
    Equal.sym(Nat, length, Shape.logical_numel(Shape.dimensions(output)), exact), earlier)
  -inside = MatrixIndex.lt_same(U32.to_nat(index), position, length, word_exact, below)
  -tree = Matrix.build(depth, Matrix.product_splits(depth, inner_n), inner_n, columns_n, length, 0n, cl, cr)
  -path = Refinement.route(depth, index)
  -cells = {\{Parallel.Fact: p => p} :
    Parallel.Fact<MatmulParallel.Cells(depth, dl, dr, cl, cr, inner_n, columns_n, length, 0n, tree)> ->
    MatmulParallel.Cells(depth, dl, dr, cl, cr, inner_n, columns_n, length, 0n, tree)}(
      MatmulParallel.build_cells(depth, Matrix.product_splits(depth, inner_n), dl, dr, cl, cr, inner_n, columns_n, length, 0n, cl, cr, {==}, {==}, bl, br))
  -position_eq = Equal.trans(Nat, Nat.add(Model.offset(depth, path), 0n), U32.to_nat(index), position,
    Parallel.output_position(depth, length, index, valid, inside), word_exact)
  -at = {\{Parallel.Fact: p => p} :
    Parallel.Fact<MatmulParallel.CellIs(dl, dr, cl, cr, inner_n, columns_n, length, position, Model.observe(F32, depth, tree, path))> ->
    MatmulParallel.CellIs(dl, dr, cl, cr, inner_n, columns_n, length, position, Model.observe(F32, depth, tree, path))}(
      MatmulParallel.cell_shift(dl, dr, cl, cr, inner_n, columns_n, length, Nat.add(Model.offset(depth, path), 0n), position,
        Model.observe(F32, depth, tree, path), position_eq, cells(path)))
  -cell = {\{Parallel.Fact: p => p} :
    Parallel.Fact<MatmulParallel.CellAt<dl, dr, cl, cr, inner_n, columns_n, position, Model.observe(F32, depth, tree, path)>> ->
    MatmulParallel.CellAt<dl, dr, cl, cr, inner_n, columns_n, position, Model.observe(F32, depth, tree, path)>}(
      MatmulParallel.cell_below(dl, dr, cl, cr, inner_n, columns_n, length, position, Model.observe(F32, depth, tree, path), below, at))
  -matched = Equal.sym(Bool, True{}, Storage.index_matches(depth, length, index), Route.index_complete(depth, length, index, valid, inside))
  -decoded = {\{Parallel.Fact: p => p} : Parallel.Fact<Decoded<output, index, U32.from_nat(row), U32.from_nat(column)>> ->
    Decoded<output, index, U32.from_nat(row), U32.from_nat(column)>}(
      decoded_of(output, index, U32.from_nat(row), U32.from_nat(column), Shape.unflatten_checked(output, index), {==},
        MatrixIndex.unflatten_at(output, U32.to_nat(rows), columns_n, extents, position, earlier)))
  Parallel.Fact{{\{MatmulParallel.CellAt: word => observed => values => logical => fold =>
    {\{Decoded: coordinate => found => coordinates =>
      Laws.ProductCell{U32.from_nat(row), U32.from_nat(column), coordinate, found, coordinates, word_exact, values,
        {\{Parallel.Fact: p => p} :
          Parallel.Fact<Laws.LogicalDot(inner_n, left, right, a, b, U32.from_nat(row), U32.from_nat(column), 0n, values)> ->
          Laws.LogicalDot(inner_n, left, right, a, b, U32.from_nat(row), U32.from_nat(column), 0n, values)}(
          logical_dot(inner_n, left, right, rows, inner, right_inner, columns, row, column, le, re, compatible, row_below, column_below,
            a, b, dl, dr, cl, cr, left_reads, right_reads, 0n, {==}, values, logical)),
        word, fold, matched,
        Equal.trans(Maybe<&2, F32>, Lifecycle.observe_allowed(depth, length, tree, index, Nat.is_lt(U32.to_nat(index), length)),
          Model.observe(F32, depth, tree, path), Some{word},
          Parallel.observe_allowed_equal(depth, length, tree, index, Nat.is_lt(U32.to_nat(index), length), inside, matched), observed)}} :
      Decoded<output, index, U32.from_nat(row), U32.from_nat(column)> ->
      Laws.ProductCell<inner, left, right, a, b, output,
        Core.TTensor{Storage.Owned{depth, length, tree, Matrix.build_balanced(depth, Matrix.product_splits(depth, inner_n), inner_n, columns_n, length, 0n, cl, cr), valid}, exact},
        position>}(decoded)} :
    MatmulParallel.CellAt<dl, dr, cl, cr, inner_n, columns_n, position, Model.observe(F32, depth, tree, path)> ->
    Laws.ProductCell<inner, left, right, a, b, output,
      Core.TTensor{Storage.Owned{depth, length, tree, Matrix.build_balanced(depth, Matrix.product_splits(depth, inner_n), inner_n, columns_n, length, 0n, cl, cr), valid}, exact},
      position>}(cell)}

def output_source(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>}, -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  +depth: Nat, +length: Nat, -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(output)) : Nat},
  -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  -kept_left: Core.Tensor<left>, -copy_left: Array<F32>, -taken_left: Parallel.Taken<Core.Packed{left, a}, (Core.Packed{left, kept_left}, copy_left)>,
  -kept_right: Core.Tensor<right>, -copy_right: Array<F32>, -taken_right: Parallel.Taken<Core.Packed{right, b}, (Core.Packed{right, kept_right}, copy_right)>) ->
  Laws.SourceMatmul<Core.Packed{left, a}, Core.Packed{right, b}, Matrix.parallel_output(output, U32.to_nat(inner), U32.to_nat(columns), depth, length,
    valid, exact, (Core.Packed{left, kept_left}, copy_left), (Core.Packed{right, kept_right}, copy_right))>:
  Laws.SourceMatmul{unit => Laws.MatmulSuccess{left, right, Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible},
    a, b, kept_left, kept_right,
    Core.TTensor{Storage.Owned{depth, length,
      Matrix.build(depth, Matrix.product_splits(depth, U32.to_nat(inner)), U32.to_nat(inner), U32.to_nat(columns), length, 0n, copy_left, copy_right),
      Matrix.build_balanced(depth, Matrix.product_splits(depth, U32.to_nat(inner)), U32.to_nat(inner), U32.to_nat(columns), length, 0n, copy_left, copy_right),
      valid}, exact},
    {==}, {==}, {==}, {==}, {==},
    Laws.ProductPrefix{
      other => {\{Parallel.Fact: p => p} : Parallel.Fact<{Core.checked_observe(left, kept_left, other) == Core.checked_observe(left, a, other) : Maybe<&2, F32>}> ->
        {Core.checked_observe(left, kept_left, other) == Core.checked_observe(left, a, other) : Maybe<&2, F32>}}(
        taken_keeps(left, a, kept_left, copy_left, taken_left, other)),
      other => {\{Parallel.Fact: p => p} : Parallel.Fact<{Core.checked_observe(right, kept_right, other) == Core.checked_observe(right, b, other) : Maybe<&2, F32>}> ->
        {Core.checked_observe(right, kept_right, other) == Core.checked_observe(right, b, other) : Maybe<&2, F32>}}(
        taken_keeps(right, b, kept_right, copy_right, taken_right, other)),
      position => earlier =>
        {\{Parallel.Taken: dl => bl => left_reads => left_same =>
          {\{Parallel.Taken: dr => br => right_reads => right_same =>
            {\{Parallel.Fact: p => p} :
              Parallel.Fact<Laws.ProductCell<inner, left, right, a, b, output,
                Core.TTensor{Storage.Owned{depth, length,
                  Matrix.build(depth, Matrix.product_splits(depth, U32.to_nat(inner)), U32.to_nat(inner), U32.to_nat(columns), length, 0n, copy_left, copy_right),
                  Matrix.build_balanced(depth, Matrix.product_splits(depth, U32.to_nat(inner)), U32.to_nat(inner), U32.to_nat(columns), length, 0n, copy_left, copy_right),
                  valid}, exact}, position>> ->
              Laws.ProductCell<inner, left, right, a, b, output,
                Core.TTensor{Storage.Owned{depth, length,
                  Matrix.build(depth, Matrix.product_splits(depth, U32.to_nat(inner)), U32.to_nat(inner), U32.to_nat(columns), length, 0n, copy_left, copy_right),
                  Matrix.build_balanced(depth, Matrix.product_splits(depth, U32.to_nat(inner)), U32.to_nat(inner), U32.to_nat(columns), length, 0n, copy_left, copy_right),
                  valid}, exact}, position>}(
              product_cell(left, right, output, rows, inner, right_inner, columns, le, re, oe, compatible, depth, length, valid, exact,
                a, b, dl, dr, copy_left, copy_right, bl, br, left_reads, right_reads, position, earlier))} :
            Parallel.Taken<Core.Packed{right, b}, (Core.Packed{right, kept_right}, copy_right)> ->
            Laws.ProductCell<inner, left, right, a, b, output,
              Core.TTensor{Storage.Owned{depth, length,
                Matrix.build(depth, Matrix.product_splits(depth, U32.to_nat(inner)), U32.to_nat(inner), U32.to_nat(columns), length, 0n, copy_left, copy_right),
                Matrix.build_balanced(depth, Matrix.product_splits(depth, U32.to_nat(inner)), U32.to_nat(inner), U32.to_nat(columns), length, 0n, copy_left, copy_right),
                valid}, exact}, position>}(taken_right)} :
          Parallel.Taken<Core.Packed{left, a}, (Core.Packed{left, kept_left}, copy_left)> ->
          Laws.ProductCell<inner, left, right, a, b, output,
            Core.TTensor{Storage.Owned{depth, length,
              Matrix.build(depth, Matrix.product_splits(depth, U32.to_nat(inner)), U32.to_nat(inner), U32.to_nat(columns), length, 0n, copy_left, copy_right),
              Matrix.build_balanced(depth, Matrix.product_splits(depth, U32.to_nat(inner)), U32.to_nat(inner), U32.to_nat(columns), length, 0n, copy_left, copy_right),
              valid}, exact}, position>}(taken_left)}}}

def allocated_source(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +rows: U32, +inner: U32, +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>}, -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  a: Core.Tensor<left>, b: Core.Tensor<right>, result: Result<&2, &1, Core.Error, Core.Tensor<output>>) ->
  Laws.SourceMatmul<Core.Packed{left, a}, Core.Packed{right, b},
    Matrix.parallel_allocated(left, right, output, U32.to_nat(inner), U32.to_nat(columns), a, b, result)>:
  match result:
    case Fail{error}:
      Laws.SourceMatmul{unit => Unit{}}
    case Done{Core.TTensor{storage, exact}}:
      match storage:
        case Storage.Empty{}:
          Laws.SourceMatmul{unit => Laws.MatmulSuccess{left, right, Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible},
            a, b, a, b, Core.TTensor{Storage.Empty{}, exact}, {==}, {==}, {==}, {==}, {==},
            Laws.ProductPrefix{other => {==}, other => {==}, position => earlier =>
              Empty.absurd(Laws.ProductCell<inner, left, right, a, b, output, Core.TTensor{Storage.Empty{}, exact}, position>,
                Arith.lt_zero(position,
                  %Equal.sym(Nat, 0n, Shape.logical_numel(Shape.dimensions(output)), exact) : {Nat.is_lt(position, _) == True{} : Bool}
                  earlier))}}}
        case Storage.Owned{+depth, +length, array, balanced, valid}:
          take_lift(left, a, taken => Laws.SourceMatmul<Core.Packed{left, a}, Core.Packed{right, b},
              Matrix.parallel_output(output, U32.to_nat(inner), U32.to_nat(columns), depth, length, valid, exact, taken, Elementwise.take(Core.pack(right, b)))>,
            kept_left => copy_left => taken_left =>
              take_lift(right, b, taken => Laws.SourceMatmul<Core.Packed{left, a}, Core.Packed{right, b},
                  Matrix.parallel_output(output, U32.to_nat(inner), U32.to_nat(columns), depth, length, valid, exact,
                    (Core.Packed{left, kept_left}, copy_left), taken)>,
                kept_right => copy_right => taken_right =>
                  output_source(left, right, output, rows, inner, right_inner, columns, le, re, oe, compatible, depth, length, valid, exact,
                    a, b, kept_left, copy_left, taken_left, kept_right, copy_right, taken_right)))

def multiply_source(+left: Shape.Shape, +right: Shape.Shape, +plan: Matrix.Plan<left, right>, a: Core.Tensor<left>, b: Core.Tensor<right>) ->
  Laws.SourceMatmul<Core.Packed{left, a}, Core.Packed{right, b}, Matrix.parallel_multiply(left, right, plan, a, b)>:
  match plan:
    case Matrix.Plan{+output, +rows, +inner, +right_inner, +columns, le, re, oe, compatible}:
      allocated_source(left, right, output, rows, inner, right_inner, columns, le, re, oe, compatible, a, b, Core.new(output, 0.0))

def planned_source(+left: Shape.Shape, +right: Shape.Shape, a: Core.Tensor<left>, b: Core.Tensor<right>,
  result: Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>) ->
  Laws.SourceMatmul<Core.Packed{left, a}, Core.Packed{right, b}, Matrix.parallel_planned(left, right, a, b, result)>:
  match result:
    case Fail{error}:
      Laws.SourceMatmul{unit => Unit{}}
    case Done{+plan}:
      multiply_source(left, right, plan, a, b)

# The parallel product succeeds on valid inputs: after the plan and the
# allocation, nothing it does can fail.
def output_status(+output: Shape.Shape, +inner: Nat, +columns: Nat, +depth: Nat, +length: Nat, -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(output)) : Nat},
  +left: Shape.Shape, a: Core.Tensor<left>, +right: Shape.Shape, b: Core.Tensor<right>) ->
  Totality.Status<Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic,
    Matrix.parallel_output(output, inner, columns, depth, length, valid, exact, Elementwise.take(Core.pack(left, a)), Elementwise.take(Core.pack(right, b)))>:
  take_lift(left, a, taken => Totality.Status<Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic,
      Matrix.parallel_output(output, inner, columns, depth, length, valid, exact, taken, Elementwise.take(Core.pack(right, b)))>,
    kept_left => copy_left => taken_left =>
      take_lift(right, b, taken => Totality.Status<Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic,
          Matrix.parallel_output(output, inner, columns, depth, length, valid, exact, (Core.Packed{left, kept_left}, copy_left), taken)>,
        kept_right => copy_right => taken_right => Totality.Status{{==}}))

def allocated_status(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +inner: Nat, +columns: Nat,
  a: Core.Tensor<left>, b: Core.Tensor<right>, result: Result<&2, &1, Core.Error, Core.Tensor<output>>,
  -done: Allocation.TensorDone<output, result>) ->
  Totality.Status<Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic, Matrix.parallel_allocated(left, right, output, inner, columns, a, b, result)>:
  match result:
    case Fail{error}:
      Totality.Status{{\{Allocation.TensorDone: tensor => exact =>
        Empty.absurd({False{} == True{} : Bool}, Shape.clash(Equal.cong(Result<&2, &1, Core.Error, Core.Tensor<output>>, Bool,
          Result.is_done(&2, &1, Core.Error, Core.Tensor<output>), Done{tensor}, Fail{error}, exact)))} :
            Allocation.TensorDone<output, Fail{error}> -> {False{} == True{} : Bool}}(done)}
    case Done{Core.TTensor{storage, exact}}:
      match storage:
        case Storage.Empty{}:
          Totality.Status{{==}}
        case Storage.Owned{+depth, +length, array, balanced, valid}:
          output_status(output, inner, columns, depth, length, valid, exact, left, a, right, b)

def multiply_parallel_status(+left: Shape.Shape, +right: Shape.Shape, +plan: Matrix.Plan<left, right>, a: Core.Tensor<left>, b: Core.Tensor<right>) ->
  Totality.Status<Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic, Matrix.parallel_multiply(left, right, plan, a, b)>:
  match plan:
    case Matrix.Plan{+output, rows, +inner, right_inner, +columns, le, re, oe, compatible}:
      allocated_status(left, right, output, U32.to_nat(inner), U32.to_nat(columns), a, b, Core.new(output, 0.0),
        AllocationProof.core_new_done(output, 0.0))

def planned_parallel_status(+left: Shape.Shape, +right: Shape.Shape, result: Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>,
  a: Core.Tensor<left>, b: Core.Tensor<right>, -status: PlanStatus<left, right, result>) ->
  Totality.Status<Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic, Matrix.parallel_planned(left, right, a, b, result)>:
  match result:
    case Fail{error}:
      Totality.Status{{\{PlanStatus: successful => Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))} :
        PlanStatus<left, right, Fail{error}> -> {False{} == True{} : Bool}}(status)}
    case Done{+plan}:
      multiply_parallel_status(left, right, plan, a, b)

def Laws.successful_matmul_interpretation(left, right):
  Core.Packed{+left_shape, a} = left
  Core.Packed{+right_shape, b} = right
  planned_source(left_shape, right_shape, a, b, Matrix.plan(left_shape, right_shape))

def Laws.matmul_valid_success(left, right, a, b, valid):
  Laws.TotalMatmul{
    MathProof.success(Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic, Matrix.matmul(Core.Packed{left, a}, Core.Packed{right, b}),
      planned_parallel_status(left, right, Matrix.plan(left, right), a, b, plan_status(left, right, valid))),
    Laws.successful_matmul_interpretation(Core.Packed{left, a}, Core.Packed{right, b})}
