import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./reduction.bend as Reduction
import ./reduction-LAWS.bend as Laws
import ./math-totality.bend as Totality
import ./word-conversion.bend as Conversion
import ./allocation-PROOF.bend as AllocationProof
import ./allocation-spec.bend as Allocation
import ./transform-PROOF.bend as TransformProof
import ./transform-LAWS.bend as TransformLaws
import ./transform-totality.bend as TransformTotality
import ../shape/nat_proofs.bend as Nats
import ../storage/arith.bend as Arith

def SomeMatches(-value: F32, result: Maybe<&2, F32>) -> Data:
  match result:
    case None{}:
      Empty
    case Some{other}:
      {value == other : F32}

def some_injective(+left: F32, +right: F32, equal: {Some{left} == Some{right} : Maybe<&2, F32>}) -> {left == right : F32}:
  %equal : SomeMatches(left, _)
  {==}

def no_value(+value: F32, equal: {None{} == Some{value} : Maybe<&2, F32>}) -> Empty:
  %Equal.sym(Maybe<&2, F32>, None{}, Some{value}, equal) : SomeMatches(value, _)
  {==}

def natural_reflexive(+value: Nat) -> {Nat.is_eq(value, value) == True{} : Bool}:
  match value:
    case 0n:
      {==}
    case 1n+tail:
      natural_reflexive(tail)

def conversion_clash(+a: Nat, +b: Nat, exact: {a == b : Nat}, rejected: {False{} == Nat.is_eq(a, b) : Bool}) -> Empty:
  Lifecycle.false_true(Equal.trans(Bool, True{}, Nat.is_eq(a, a), False{},
    Equal.sym(Bool, Nat.is_eq(a, a), True{}, natural_reflexive(a)),
    Equal.trans(Bool, Nat.is_eq(a, a), Nat.is_eq(a, b), False{},
      Equal.cong(Nat, Bool, n => Nat.is_eq(a, n), a, b, exact), Equal.sym(Bool, False{}, Nat.is_eq(a, b), rejected))))

def logical_keeps(+fuel: Nat, -shape: Shape.Shape, -original: Core.Tensor<shape>, -tensor: Core.Tensor<shape>, +cursor: Nat,
  values: List<&2, F32>, logical: Laws.LogicalValues(fuel, shape, original, cursor, values),
  -kept: @+other: U32 -> {Core.checked_observe(shape, tensor, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}) ->
  Laws.LogicalValues(fuel, shape, tensor, cursor, values):
  match fuel values:
    case 0n Nil{}:
      Unit{}
    case 1n+tail +value <> rest:
      Laws.LogicalStep{exact, observed, remaining} = logical
      Laws.LogicalStep{exact, Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, tensor, U32.from_nat(cursor)),
        Core.checked_observe(shape, original, U32.from_nat(cursor)), Some{value}, kept(U32.from_nat(cursor)), observed),
        logical_keeps(tail, shape, original, tensor, 1n+cursor, rest, remaining, kept)}
    case 0n value <> rest:
      Empty.absurd(Laws.LogicalValues(0n, shape, tensor, cursor, value <> rest), logical)
    case 1n+tail Nil{}:
      Empty.absurd(Laws.LogicalValues(1n+tail, shape, tensor, cursor, Nil{}), logical)

def read_interpretation(+fuel: Nat, -shape: Shape.Shape, -original: Core.Tensor<shape>, +cursor: Nat, +accumulator: F32,
  +head: F32, +values: List<&2, F32>, tensor: Core.Tensor<shape>, result: Result<&2, &2, Core.Error, F32>,
  rec: @tensor: Core.Tensor<shape> -> @logical: Laws.LogicalValues(fuel, shape, tensor, 1n+cursor, values) ->
    {Laws.loop_value(shape, Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, head))) ==
      Some{Reduction.sequential_sum(values, F32.add(accumulator, head))} : Maybe<&2, F32>},
  remaining: Laws.LogicalValues(fuel, shape, original, 1n+cursor, values),
  meaning: {Core.result_value(result) == Some{head} : Maybe<&2, F32>},
  -kept: @+other: U32 -> {Core.checked_observe(shape, tensor, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}) ->
  {Laws.loop_value(shape, Reduction.read_step(shape,
    tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, (tensor, result))) ==
    Some{Reduction.sequential_sum(head <> values, accumulator)} : Maybe<&2, F32>}:
  match result:
    case Fail{error}:
      Empty.absurd({None{} == Some{Reduction.sequential_sum(head <> values, accumulator)} : Maybe<&2, F32>}, no_value(head, meaning))
    case Done{+value}:
      %Equal.sym(F32, value, head, some_injective(value, head, meaning)) :
        {Laws.loop_value(shape, Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, _))) ==
          Some{Reduction.sequential_sum(values, F32.add(accumulator, head))} : Maybe<&2, F32>}
      %rec(tensor, logical_keeps(fuel, shape, original, tensor, 1n+cursor, values, remaining, kept)) :
        {Laws.loop_value(shape, Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, head))) == _ : Maybe<&2, F32>}
      {==}

def read_certificate_interpretation(+fuel: Nat, -shape: Shape.Shape, -original: Core.Tensor<shape>, +cursor: Nat,
  +accumulator: F32, +head: F32, +values: List<&2, F32>,
  rec: @tensor: Core.Tensor<shape> -> @logical: Laws.LogicalValues(fuel, shape, tensor, 1n+cursor, values) ->
    {Laws.loop_value(shape, Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, head))) ==
      Some{Reduction.sequential_sum(values, F32.add(accumulator, head))} : Maybe<&2, F32>},
  observed: {Core.checked_observe(shape, original, U32.from_nat(cursor)) == Some{head} : Maybe<&2, F32>},
  remaining: Laws.LogicalValues(fuel, shape, original, 1n+cursor, values), certificate: Core.Read<shape, original, U32.from_nat(cursor)>,
  meaning: {Core.read_value(shape, Core.read_flat(shape, original, U32.from_nat(cursor))) == Core.checked_observe(shape, original, U32.from_nat(cursor)) : Maybe<&2, F32>},
  -kept: @+other: U32 -> {Core.retained_read_observe(shape, Core.read_flat(shape, original, U32.from_nat(cursor)), other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}) ->
  {Laws.loop_value(shape, Reduction.read_step(shape,
    tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, Core.read_flat(shape, original, U32.from_nat(cursor)))) ==
    Some{Reduction.sequential_sum(head <> values, accumulator)} : Maybe<&2, F32>}:
  Core.Read{tensor, +result, +exact} = certificate
  %exact :
    {Laws.loop_value(shape, Reduction.read_step(shape,
      tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, _)) ==
      Some{Reduction.sequential_sum(head <> values, accumulator)} : Maybe<&2, F32>}
  read_interpretation(fuel, shape, original, cursor, accumulator, head, values, tensor, result, rec, remaining,
    Equal.trans(Maybe<&2, F32>, Core.result_value(result), Core.checked_observe(shape, original, U32.from_nat(cursor)), Some{head},
      Equal.trans(Maybe<&2, F32>, Core.result_value(result), Core.read_value(shape, Core.read_flat(shape, original, U32.from_nat(cursor))),
        Core.checked_observe(shape, original, U32.from_nat(cursor)),
        Equal.cong(Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>, Maybe<&2, F32>, pair => Core.read_value(shape, pair),
          (tensor, result), Core.read_flat(shape, original, U32.from_nat(cursor)), exact),
        meaning), observed),
    other => Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, tensor, other),
      Core.retained_read_observe(shape, Core.read_flat(shape, original, U32.from_nat(cursor)), other), Core.checked_observe(shape, original, other),
      Equal.cong(Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>, Maybe<&2, F32>, pair => Core.retained_read_observe(shape, pair, other),
        (tensor, result), Core.read_flat(shape, original, U32.from_nat(cursor)), exact), kept(other)))

def decision_interpretation(+fuel: Nat, -shape: Shape.Shape, tensor: Core.Tensor<shape>, +cursor: Nat,
  +accumulator: F32, +head: F32, +values: List<&2, F32>,
  -rec: @tensor: Core.Tensor<shape> -> @logical: Laws.LogicalValues(fuel, shape, tensor, 1n+cursor, values) ->
    {Laws.loop_value(shape, Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, head))) ==
      Some{Reduction.sequential_sum(values, F32.add(accumulator, head))} : Maybe<&2, F32>},
  logical: Laws.LogicalStep<fuel, shape, tensor, cursor, head, values>,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)>) ->
  Laws.Interpretation<Laws.loop_value(shape, Reduction.sum_decided(shape,
    tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), tensor, cursor, U32.from_nat(cursor), accumulator, decision)),
    Some{Reduction.sequential_sum(head <> values, accumulator)}>:
  Laws.LogicalStep{exact, observed, remaining} = logical
  match decision:
    case Shape.BFalse{equal}:
      Laws.Interpretation{unit => Empty.absurd({None{} == Some{Reduction.sequential_sum(head <> values, accumulator)} : Maybe<&2, F32>},
        conversion_clash(U32.to_nat(U32.from_nat(cursor)), cursor, exact, equal))}
    case Shape.BTrue{equal}:
      Laws.Interpretation{unit =>
        prove : @certificates: Core.SourceRead<shape, tensor, U32.from_nat(cursor)> & Core.SourceReadKeeps<shape, tensor, U32.from_nat(cursor)> ->
          {Laws.loop_value(shape, Reduction.read_step(shape,
            tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, Core.read_flat(shape, tensor, U32.from_nat(cursor)))) ==
            Some{Reduction.sequential_sum(head <> values, accumulator)} : Maybe<&2, F32>} =
          \{Tuple: \{Core.SourceRead: meaning => \{Core.SourceReadKeeps: kept =>
            read_certificate_interpretation(fuel, shape, tensor, cursor, accumulator, head, values, rec, observed, remaining,
              Core.read_certificate(shape, tensor, U32.from_nat(cursor)), meaning, kept)}}}
        prove((Core.source_read_model(shape, tensor, U32.from_nat(cursor)), Core.source_read_keeps(shape, tensor, U32.from_nat(cursor))))}

def Laws.ordered_loop_interpretation(fuel, shape, tensor, cursor, accumulator, values, logical):
  match fuel values:
    case 0n Nil{}:
      Laws.Interpretation{unit => {==}}
    case 1n+tail +head <> +rest:
      decision_interpretation(tail, shape, tensor, cursor, accumulator, head, rest,
        tensor => logical =>
          claim : @certificate: Laws.Interpretation<Laws.loop_value(shape, Reduction.natural_sum_loop(tail, shape, tensor, 1n+cursor, F32.add(accumulator, head))),
            Some{Reduction.sequential_sum(rest, F32.add(accumulator, head))}> ->
            {Laws.loop_value(shape, Reduction.natural_sum_loop(tail, shape, tensor, 1n+cursor, F32.add(accumulator, head))) ==
              Some{Reduction.sequential_sum(rest, F32.add(accumulator, head))} : Maybe<&2, F32>} =
            \{Laws.Interpretation: proof => proof(Unit{})}
          claim(Laws.ordered_loop_interpretation(tail, shape, tensor, 1n+cursor, F32.add(accumulator, head), rest, logical)),
        logical, Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)))
    case 0n head <> rest:
      Laws.Interpretation{unit => Empty.absurd({Laws.loop_value(shape, Reduction.natural_sum_loop(0n, shape, tensor, cursor, accumulator)) ==
        Some{Reduction.sequential_sum(head <> rest, accumulator)} : Maybe<&2, F32>}, logical)}
    case 1n+tail Nil{}:
      Laws.Interpretation{unit => Empty.absurd({Laws.loop_value(shape, Reduction.natural_sum_loop(1n+tail, shape, tensor, cursor, accumulator)) ==
        Some{Reduction.sequential_sum(Nil{}, accumulator)} : Maybe<&2, F32>}, logical)}

def success_keeps(-shape: Shape.Shape, -original: Core.Tensor<shape>, -tensor: Core.Tensor<shape>,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>, +other: U32,
  kept: {Core.checked_observe(shape, tensor, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}) ->
  {Laws.successful_observe(shape, tensor, result, other) == Laws.successful_observe(shape, original, result, other) : Maybe<&2, F32>}:
  match result:
    case Fail{error}:
      {==}
    case Done{pair}:
      kept

def preserved_read(+fuel: Nat, -shape: Shape.Shape, -original: Core.Tensor<shape>, +cursor: Nat, +accumulator: F32,
  tensor: Core.Tensor<shape>, result: Result<&2, &2, Core.Error, F32>,
  -rec: @tensor: Core.Tensor<shape> -> @+value: F32 -> Laws.Preserved<shape, tensor, Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value)>,
  -kept: @+other: U32 -> {Core.checked_observe(shape, tensor, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}) ->
  Laws.Preserved<shape, original, Reduction.read_step(shape,
    tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, (tensor, result))>:
  match result:
    case Fail{error}:
      Laws.Preserved{other => {==}}
    case Done{+value}:
      Laws.Preserved{other =>
        claim : @certificate: Laws.Preserved<shape, tensor, Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, value))> ->
          {Laws.retained_observe(shape, Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, value)), other) ==
            Laws.successful_observe(shape, tensor, Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, value)), other) : Maybe<&2, F32>} =
          \{Laws.Preserved: kept => kept(other)}
        Equal.trans(Maybe<&2, F32>,
          Laws.retained_observe(shape, Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, value)), other),
          Laws.successful_observe(shape, tensor, Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, value)), other),
          Laws.successful_observe(shape, original, Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, value)), other),
          claim(rec(tensor, F32.add(accumulator, value))), success_keeps(shape, original, tensor,
            Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, value)), other, kept(other)))}

def preserved_certificate(+fuel: Nat, -shape: Shape.Shape, -original: Core.Tensor<shape>, +cursor: Nat, +accumulator: F32,
  -rec: @tensor: Core.Tensor<shape> -> @+value: F32 -> Laws.Preserved<shape, tensor, Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value)>,
  certificate: Core.Read<shape, original, U32.from_nat(cursor)>, -retention: Core.SourceReadKeeps<shape, original, U32.from_nat(cursor)>) ->
  Laws.Preserved<shape, original, Reduction.read_step(shape,
    tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, Core.read_flat(shape, original, U32.from_nat(cursor)))>:
  Core.Read{tensor, +result, +exact} = certificate
  Laws.Preserved{other =>
    claim : @certificate: Laws.Preserved<shape, original, Reduction.read_step(shape,
      tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, (tensor, result))> ->
      {Laws.retained_observe(shape, Reduction.read_step(shape,
        tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, (tensor, result)), other) ==
        Laws.successful_observe(shape, original, Reduction.read_step(shape,
          tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, (tensor, result)), other) : Maybe<&2, F32>} =
      \{Laws.Preserved: kept => kept(other)}
    prove : @retention: Core.SourceReadKeeps<shape, original, U32.from_nat(cursor)> ->
      {Laws.retained_observe(shape, Reduction.read_step(shape,
        tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, Core.read_flat(shape, original, U32.from_nat(cursor))), other) ==
        Laws.successful_observe(shape, original, Reduction.read_step(shape,
          tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, Core.read_flat(shape, original, U32.from_nat(cursor))), other) : Maybe<&2, F32>} =
      \{Core.SourceReadKeeps: kept =>
      %exact :
        {Laws.retained_observe(shape, Reduction.read_step(shape,
          tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, _), other) ==
          Laws.successful_observe(shape, original, Reduction.read_step(shape,
            tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, _), other) : Maybe<&2, F32>}
      claim(preserved_read(fuel, shape, original, cursor, accumulator, tensor, result, rec,
        other => Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, tensor, other),
          Core.retained_read_observe(shape, Core.read_flat(shape, original, U32.from_nat(cursor)), other), Core.checked_observe(shape, original, other),
          Equal.cong(Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>, Maybe<&2, F32>, pair => Core.retained_read_observe(shape, pair, other),
            (tensor, result), Core.read_flat(shape, original, U32.from_nat(cursor)), exact), kept(other))))}
    prove(retention)}

def preserved_decision(+fuel: Nat, -shape: Shape.Shape, tensor: Core.Tensor<shape>, +cursor: Nat, +accumulator: F32,
  -rec: @tensor: Core.Tensor<shape> -> @+value: F32 -> Laws.Preserved<shape, tensor, Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value)>,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)>) ->
  Laws.Preserved<shape, tensor, Reduction.sum_decided(shape,
    tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), tensor, cursor, U32.from_nat(cursor), accumulator, decision)>:
  match decision:
    case Shape.BFalse{equal}:
      Laws.Preserved{other => {==}}
    case Shape.BTrue{equal}:
      preserved_certificate(fuel, shape, tensor, cursor, accumulator, rec,
        Core.read_certificate(shape, tensor, U32.from_nat(cursor)), Core.source_read_keeps(shape, tensor, U32.from_nat(cursor)))

def Laws.ordered_loop_preservation(fuel, shape, tensor, cursor, accumulator):
  match fuel:
    case 0n:
      Laws.Preserved{other => {==}}
    case 1n+tail:
      preserved_decision(tail, shape, tensor, cursor, accumulator,
        tensor => value => Laws.ordered_loop_preservation(tail, shape, tensor, 1n+cursor, value),
        Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)))

def sum_result_value(-shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>) ->
  {Reduction.sum_observation(shape, Reduction.sum_result(shape, result)) == Laws.loop_value(shape, result) : Maybe<&2, F32>}:
  match result:
    case Fail{error}:
      {==}
    case Done{pair}:
      (tensor, value) = pair
      {==}

def Laws.ordered_sum_interpretation(shape, tensor, values, logical):
  Laws.Interpretation{unit =>
    claim : @certificate: Laws.Interpretation<Laws.loop_value(shape, Reduction.natural_sum_loop(Shape.logical_numel(Shape.dimensions(shape)), shape, tensor, 0n, 0.0)),
      Some{Reduction.sequential_sum(values, 0.0)}> ->
      {Laws.loop_value(shape, Reduction.natural_sum_loop(Shape.logical_numel(Shape.dimensions(shape)), shape, tensor, 0n, 0.0)) ==
        Some{Reduction.sequential_sum(values, 0.0)} : Maybe<&2, F32>} =
      \{Laws.Interpretation: proof => proof(Unit{})}
    %Equal.sym(Nat, Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)), Shape.elements_exact(shape)) :
      {Reduction.sum_observation(shape, Reduction.sum_result(shape, Reduction.natural_sum_loop(_, shape, tensor, 0n, 0.0))) ==
        Some{Reduction.sequential_sum(values, 0.0)} : Maybe<&2, F32>}
    Equal.trans(Maybe<&2, F32>, Reduction.sum_observation(shape,
      Reduction.sum_result(shape, Reduction.natural_sum_loop(Shape.logical_numel(Shape.dimensions(shape)), shape, tensor, 0n, 0.0))),
      Laws.loop_value(shape, Reduction.natural_sum_loop(Shape.logical_numel(Shape.dimensions(shape)), shape, tensor, 0n, 0.0)),
      Some{Reduction.sequential_sum(values, 0.0)},
      sum_result_value(shape, Reduction.natural_sum_loop(Shape.logical_numel(Shape.dimensions(shape)), shape, tensor, 0n, 0.0)),
      claim(Laws.ordered_loop_interpretation(Shape.logical_numel(Shape.dimensions(shape)), shape, tensor, 0n, 0.0, values, logical)))}

def logical_original(+fuel: Nat, -shape: Shape.Shape, -original: Core.Tensor<shape>, -tensor: Core.Tensor<shape>, +cursor: Nat,
  values: List<&2, F32>, logical: Laws.LogicalValues(fuel, shape, tensor, cursor, values),
  -kept: @+other: U32 -> {Core.checked_observe(shape, tensor, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}) ->
  Laws.LogicalValues(fuel, shape, original, cursor, values):
  logical_keeps(fuel, shape, tensor, original, cursor, values, logical,
    other => Equal.sym(Maybe<&2, F32>, Core.checked_observe(shape, tensor, other), Core.checked_observe(shape, original, other), kept(other)))

def source_extend(+fuel: Nat, -shape: Shape.Shape, -original: Core.Tensor<shape>, -tensor: Core.Tensor<shape>, +cursor: Nat,
  +accumulator: F32, +head: F32, result: Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>,
  next: Laws.LoopMeaning(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, head), result),
  -exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  -observed: {Core.checked_observe(shape, original, U32.from_nat(cursor)) == Some{head} : Maybe<&2, F32>},
  -kept: @+other: U32 -> {Core.checked_observe(shape, tensor, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}) ->
  Laws.LoopMeaning(1n+fuel, shape, original, cursor, accumulator, result):
  match result:
    case Fail{error}:
      Unit{}
    case Done{pair}:
      (retained, value) = pair
      Laws.SumMeaning{values, logical, fold} = next
      Laws.SumMeaning{head <> values,
        Laws.LogicalStep{exact, observed, logical_original(fuel, shape, original, tensor, 1n+cursor, values, logical, kept)}, fold}

def source_read(+fuel: Nat, -shape: Shape.Shape, -original: Core.Tensor<shape>, +cursor: Nat, +accumulator: F32,
  tensor: Core.Tensor<shape>, result: Result<&2, &2, Core.Error, F32>,
  -rec: @tensor: Core.Tensor<shape> -> @+value: F32 -> Laws.SourceSum<fuel, shape, tensor, 1n+cursor, value,
    Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value)>,
  -exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  -meaning: {Core.result_value(result) == Core.checked_observe(shape, original, U32.from_nat(cursor)) : Maybe<&2, F32>},
  -kept: @+other: U32 -> {Core.checked_observe(shape, tensor, other) == Core.checked_observe(shape, original, other) : Maybe<&2, F32>}) ->
  Laws.SourceSum<1n+fuel, shape, original, cursor, accumulator, Reduction.read_step(shape,
    tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, (tensor, result))>:
  match result:
    case Fail{error}:
      Laws.SourceSum{unit => Unit{}}
    case Done{+value}:
      Laws.SourceSum{unit =>
        claim : @certificate: Laws.SourceSum<fuel, shape, tensor, 1n+cursor, F32.add(accumulator, value),
          Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, value))> ->
          Laws.LoopMeaning(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, value),
            Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, value))) =
          \{Laws.SourceSum: proof => proof(Unit{})}
        source_extend(fuel, shape, original, tensor, cursor, accumulator, value,
          Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, value)),
          claim(rec(tensor, F32.add(accumulator, value))), exact,
          Equal.sym(Maybe<&2, F32>, Some{value}, Core.checked_observe(shape, original, U32.from_nat(cursor)), meaning), kept)}

def source_pair(+fuel: Nat, -shape: Shape.Shape, -original: Core.Tensor<shape>, +cursor: Nat, +accumulator: F32,
  -rec: @tensor: Core.Tensor<shape> -> @+value: F32 -> Laws.SourceSum<fuel, shape, tensor, 1n+cursor, value,
    Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value)>,
  -exact_cursor: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  certificate: Core.Read<shape, original, U32.from_nat(cursor)>,
  -source: Core.SourceRead<shape, original, U32.from_nat(cursor)>, -retention: Core.SourceReadKeeps<shape, original, U32.from_nat(cursor)>) ->
  Laws.SourceSum<1n+fuel, shape, original, cursor, accumulator, Reduction.read_step(shape,
    tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, Core.read_flat(shape, original, U32.from_nat(cursor)))>:
  Core.Read{tensor, +result, +exact} = certificate
  Laws.SourceSum{unit =>
    claim : @certificate: Laws.SourceSum<1n+fuel, shape, original, cursor, accumulator, Reduction.read_step(shape,
      tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, (tensor, result))> ->
      Laws.LoopMeaning(1n+fuel, shape, original, cursor, accumulator, Reduction.read_step(shape,
        tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, (tensor, result))) =
      \{Laws.SourceSum: proof => proof(Unit{})}
    prove : @certificates: Core.SourceRead<shape, original, U32.from_nat(cursor)> & Core.SourceReadKeeps<shape, original, U32.from_nat(cursor)> ->
      Laws.LoopMeaning(1n+fuel, shape, original, cursor, accumulator, Reduction.read_step(shape,
        tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, Core.read_flat(shape, original, U32.from_nat(cursor)))) =
      \{Tuple: \{Core.SourceRead: meaning => \{Core.SourceReadKeeps: kept =>
        %exact : Laws.LoopMeaning(1n+fuel, shape, original, cursor, accumulator, Reduction.read_step(shape,
          tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, _))
        claim(source_read(fuel, shape, original, cursor, accumulator, tensor, result, rec, exact_cursor,
          Equal.trans(Maybe<&2, F32>, Core.result_value(result), Core.read_value(shape, Core.read_flat(shape, original, U32.from_nat(cursor))),
            Core.checked_observe(shape, original, U32.from_nat(cursor)),
            Equal.cong(Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>, Maybe<&2, F32>, pair => Core.read_value(shape, pair),
              (tensor, result), Core.read_flat(shape, original, U32.from_nat(cursor)), exact), meaning),
          other => Equal.trans(Maybe<&2, F32>, Core.checked_observe(shape, tensor, other),
            Core.retained_read_observe(shape, Core.read_flat(shape, original, U32.from_nat(cursor)), other), Core.checked_observe(shape, original, other),
            Equal.cong(Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>, Maybe<&2, F32>, pair => Core.retained_read_observe(shape, pair, other),
              (tensor, result), Core.read_flat(shape, original, U32.from_nat(cursor)), exact), kept(other))))}}}
    prove((source, retention))}

def source_decision(+fuel: Nat, -shape: Shape.Shape, tensor: Core.Tensor<shape>, +cursor: Nat, +accumulator: F32,
  -rec: @tensor: Core.Tensor<shape> -> @+value: F32 -> Laws.SourceSum<fuel, shape, tensor, 1n+cursor, value,
    Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value)>,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)>) ->
  Laws.SourceSum<1n+fuel, shape, tensor, cursor, accumulator, Reduction.sum_decided(shape,
    tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), tensor, cursor, U32.from_nat(cursor), accumulator, decision)>:
  match decision:
    case Shape.BFalse{equal}:
      Laws.SourceSum{unit => Unit{}}
    case Shape.BTrue{equal}:
      source_pair(fuel, shape, tensor, cursor, accumulator, rec,
        Lifecycle.natural_equal(U32.to_nat(U32.from_nat(cursor)), cursor, equal),
        Core.read_certificate(shape, tensor, U32.from_nat(cursor)), Core.source_read_model(shape, tensor, U32.from_nat(cursor)),
        Core.source_read_keeps(shape, tensor, U32.from_nat(cursor)))

def Laws.successful_loop_interpretation(fuel, shape, tensor, cursor, accumulator):
  match fuel:
    case 0n:
      Laws.SourceSum{unit => Laws.SumMeaning{Nil{}, Unit{}, {==}}}
    case 1n+tail:
      source_decision(tail, shape, tensor, cursor, accumulator,
        tensor => value => Laws.successful_loop_interpretation(tail, shape, tensor, 1n+cursor, value),
        Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)))

def sum_source_result(+shape: Shape.Shape, -original: Core.Tensor<shape>,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>,
  -source: Laws.SourceSum<Shape.logical_numel(Shape.dimensions(shape)), shape, original, 0n, 0.0, result>,
  -preservation: Laws.Preserved<shape, original, result>) -> Laws.SourceSumResult<shape, original, Reduction.sum_result(shape, result)>:
  match result:
    case Fail{error}:
      Laws.SourceSumResult{unit => Unit{}}
    case Done{pair}:
      (retained, +value) = pair
      Laws.SourceSumResult{unit =>
        prove : @certificates: Laws.SourceSum<Shape.logical_numel(Shape.dimensions(shape)), shape, original, 0n, 0.0,
          (Done{(retained, value)} : Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>)> &
          Laws.Preserved<shape, original, (Done{(retained, value)} : Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>)> ->
          Laws.SumResultMeaning(shape, original, Reduction.sum_result(shape, Done{(retained, value)})) =
          \{Tuple: \{Laws.SourceSum: meaning => \{Laws.Preserved: kept =>
            build : @meaning: Laws.SumMeaning<Shape.logical_numel(Shape.dimensions(shape)), shape, original, 0n, 0.0, value> ->
              Laws.SumResultMeaning(shape, original, Reduction.sum_result(shape, Done{(retained, value)})) =
              \{Laws.SumMeaning: values => logical => fold =>
                Laws.SumResult{values, logical,
                  Equal.cong(F32, Maybe<&2, F32>, x => Some{x}, value, Reduction.sequential_sum(values, 0.0), fold), kept}}
            build(meaning(Unit{}))}}}
        prove((source, preservation))}

def Laws.successful_sum_interpretation(shape, tensor):
  Laws.SourceSumResult{unit =>
    claim : @certificate: Laws.SourceSumResult<shape, tensor,
      Reduction.sum_result(shape, Reduction.natural_sum_loop(Shape.logical_numel(Shape.dimensions(shape)), shape, tensor, 0n, 0.0))> ->
      Laws.SumResultMeaning(shape, tensor,
        Reduction.sum_result(shape, Reduction.natural_sum_loop(Shape.logical_numel(Shape.dimensions(shape)), shape, tensor, 0n, 0.0))) =
      \{Laws.SourceSumResult: meaning => meaning(Unit{})}
    %Equal.sym(Nat, Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)), Shape.elements_exact(shape)) :
      Laws.SumResultMeaning(shape, tensor, Reduction.sum_result(shape, Reduction.natural_sum_loop(_, shape, tensor, 0n, 0.0)))
    claim(sum_source_result(shape, tensor, Reduction.natural_sum_loop(Shape.logical_numel(Shape.dimensions(shape)), shape, tensor, 0n, 0.0),
      Laws.successful_loop_interpretation(Shape.logical_numel(Shape.dimensions(shape)), shape, tensor, 0n, 0.0),
      Laws.ordered_loop_preservation(Shape.logical_numel(Shape.dimensions(shape)), shape, tensor, 0n, 0.0)))}

def dynamic_sum_result(+shape: Shape.Shape, -original: Core.Tensor<shape>,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape> & Core.Tensor<Reduction.scalar_shape()>>,
  -source: Laws.SourceSumResult<shape, original, result>) -> Laws.SourceDynamicSum<Core.Packed{shape, original}, Reduction.pack_result(shape, result)>:
  match result:
    case Fail{error}:
      Laws.SourceDynamicSum{unit => Unit{}}
    case Done{pair}:
      (owner, scalar) = pair
      Laws.SourceDynamicSum{unit =>
        claim : @source: Laws.SourceSumResult<shape, original,
          (Done{(owner, scalar)} : Result<&2, &1, Core.Error, Core.Tensor<shape> & Core.Tensor<Reduction.scalar_shape()>>)> ->
          Laws.SumResult<shape, original, owner, scalar> = \{Laws.SourceSumResult: meaning => meaning(Unit{})}
        Laws.DynamicSumSuccess{shape, original, owner, scalar, {==}, {==}, {==}, claim(source)}}

def Laws.successful_dynamic_sum_interpretation(tensor):
  Core.Packed{+shape, owner} = tensor
  dynamic_sum_result(shape, owner, Reduction.sum(shape, owner), Laws.successful_sum_interpretation(shape, owner))

def Laws.empty_accumulation(shape, tensor, index, accumulator):
  {==}

def Laws.empty_sum_identity(shape, tensor, empty):
  match shape:
    case Shape.SShape{+dimensions, +size, +exact, +rank_bound, +capacity_bound}:
      %Equal.sym(Nat, size, 0n, Equal.trans(Nat, size, Shape.logical_numel(dimensions), 0n, exact, empty)) :
        {Reduction.sum_observation(Shape.SShape{dimensions, size, exact, rank_bound, capacity_bound},
          Reduction.sum_result(Shape.SShape{dimensions, size, exact, rank_bound, capacity_bound},
            Reduction.natural_sum_loop(_, Shape.SShape{dimensions, size, exact, rank_bound, capacity_bound}, tensor, 0n, 0.0))) == Some{0.0} : Maybe<&2, F32>}
      {==}

def success(-error: Data, -value: Type, result: Result<&2, &1, error, value>,
  -status: Totality.Status<error, value, result>) -> Totality.Success<error, value, result>:
  match result:
    case Fail{failure}:
      Totality.Success{
        {\{Totality.Status: successful => Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))} :
          Totality.Status<error, value, Fail{failure}> -> {False{} == True{} : Bool}}(status),
        {\{Totality.Status: successful => Empty.absurd(Totality.DoneWitness<error, value, Fail{failure}>, Arith.clash(successful))} :
          Totality.Status<error, value, Fail{failure}> -> Totality.DoneWitness<error, value, Fail{failure}>}(status)}
    case Done{returned}:
      Totality.Success{{==}, Totality.DoneWitness{returned, {==}}}

def add_zero(+value: Nat) -> {Nat.add(value, 0n) == value : Nat}:
  match value:
    case 0n:
      {==}
    case 1n+p:
      Equal.cong(Nat, Nat, n => 1n+n, Nat.add(p, 0n), p, add_zero(p))

def add_comm(+a: Nat, +b: Nat) -> {Nat.add(a, b) == Nat.add(b, a) : Nat}:
  match a:
    case 0n:
      Equal.sym(Nat, Nat.add(b, 0n), b, add_zero(b))
    case 1n+p:
      Equal.trans(Nat, 1n+Nat.add(p, b), 1n+Nat.add(b, p), Nat.add(b, 1n+p),
        Equal.cong(Nat, Nat, n => 1n+n, Nat.add(p, b), Nat.add(b, p), add_comm(p, b)),
        Equal.sym(Nat, Nat.add(b, 1n+p), 1n+Nat.add(b, p), Arith.add_succ(b, p)))

def cursor_bounds(+fuel: Nat, +cursor: Nat, +limit: Nat,
  bound: Shape.LE(Nat.add(1n+fuel, cursor), limit)) -> {Nat.is_lt(cursor, limit) == True{} : Bool}:
  Conversion.le_successor_lt(cursor, limit,
    Equal.sym(Bool, True{}, Nat.is_le(1n+cursor, limit),
      Nats.le_true(1n+cursor, limit,
        Nats.le_add_left(1n+cursor, fuel, limit,
          %add_comm(fuel, cursor) : Shape.LE(1n+_, limit)
          bound))))

def cursor_exact(+fuel: Nat, +shape: Shape.Shape, +cursor: Nat,
  bound: Shape.LE(Nat.add(1n+fuel, cursor), Shape.elements(shape))) -> {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat}:
  Conversion.natural_word_exact(cursor, Shape.count(shape),
    %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.elements(shape), Conversion.shape_count_exact(shape)) : {Nat.is_le(cursor, _) == True{} : Bool}
    Conversion.le_predecessor(cursor, Shape.elements(shape),
      Equal.sym(Bool, True{}, Nat.is_le(1n+cursor, Shape.elements(shape)),
        Nats.le_true(1n+cursor, Shape.elements(shape),
          Nats.le_add_left(1n+cursor, fuel, Shape.elements(shape),
            %add_comm(fuel, cursor) : Shape.LE(1n+_, Shape.elements(shape))
            bound)))))

def sum_read_status(+fuel: Nat, +shape: Shape.Shape, +cursor: Nat, +accumulator: F32,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>,
  -successful: {TransformTotality.read_succeeded(shape, pair) == True{} : Bool},
  -rec: @tensor: Core.Tensor<shape> -> @+value: F32 -> Totality.Status<Core.Error, Core.Tensor<shape> & F32,
    Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value)>) ->
  Totality.Status<Core.Error, Core.Tensor<shape> & F32, Reduction.read_step(shape,
    tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), accumulator, pair)>:
  (tensor, result) = pair
  match result:
    case Fail{error}:
      Totality.Status{Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))}
    case Done{+value}:
      Totality.Status{{\{Totality.Status: successful => successful} :
        Totality.Status<Core.Error, Core.Tensor<shape> & F32,
          Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, value))> ->
            {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape> & F32,
              Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, F32.add(accumulator, value))) == True{} : Bool}}(
                rec(tensor, F32.add(accumulator, value)))}

def sum_decision_status(+fuel: Nat, +shape: Shape.Shape, +cursor: Nat, +accumulator: F32,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)>,
  tensor: Core.Tensor<shape>,
  -exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
  -inside: {Nat.is_lt(U32.to_nat(U32.from_nat(cursor)), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  -rec: @tensor: Core.Tensor<shape> -> @+value: F32 -> Totality.Status<Core.Error, Core.Tensor<shape> & F32,
    Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value)>) ->
  Totality.Status<Core.Error, Core.Tensor<shape> & F32, Reduction.sum_decided(shape,
    tensor => value => Reduction.natural_sum_loop(fuel, shape, tensor, 1n+cursor, value), tensor, cursor, U32.from_nat(cursor), accumulator, decision)>:
  match decision:
    case Shape.BFalse{equal}:
      Totality.Status{Empty.absurd({False{} == True{} : Bool}, conversion_clash(U32.to_nat(U32.from_nat(cursor)), cursor, exact, equal))}
    case Shape.BTrue{equal}:
      sum_read_status(fuel, shape, cursor, accumulator, Core.read_flat(shape, tensor, U32.from_nat(cursor)),
        {\{TransformTotality.ReadSuccess: successful => successful} :
          TransformTotality.ReadSuccess<shape, tensor, U32.from_nat(cursor)> ->
            {TransformTotality.read_succeeded(shape, Core.read_flat(shape, tensor, U32.from_nat(cursor))) == True{} : Bool}}(
              TransformLaws.tensor_read_valid_success(shape, tensor, U32.from_nat(cursor), inside)), rec)

def sum_loop_status(fuel: Nat, +shape: Shape.Shape, tensor: Core.Tensor<shape>, +cursor: Nat, +accumulator: F32,
  -bound: Shape.LE(Nat.add(fuel, cursor), Shape.elements(shape))) ->
  Totality.Status<Core.Error, Core.Tensor<shape> & F32, Reduction.natural_sum_loop(fuel, shape, tensor, cursor, accumulator)>:
  match fuel:
    case 0n:
      Totality.Status{{==}}
    case 1n+tail:
      sum_decision_status(tail, shape, cursor, accumulator,
        Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)),
        tensor,
        cursor_exact(tail, shape, cursor, bound),
        %Equal.sym(Nat, U32.to_nat(U32.from_nat(cursor)), cursor, cursor_exact(tail, shape, cursor, bound)) : {Nat.is_lt(_, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}
        %Shape.elements_exact(shape) : {Nat.is_lt(cursor, _) == True{} : Bool}
        cursor_bounds(tail, cursor, Shape.elements(shape), bound),
        tensor => value => sum_loop_status(tail, shape, tensor, 1n+cursor, value,
          %Equal.sym(Nat, Nat.add(tail, 1n+cursor), 1n+Nat.add(tail, cursor), Arith.add_succ(tail, cursor)) : Shape.LE(_, Shape.elements(shape))
          bound))

def sum_allocated_status(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +value: F32) ->
  Totality.Status<Core.Error, Core.Tensor<shape> & Core.Tensor<Reduction.scalar_shape()>, Reduction.scalar_result(shape, tensor, Core.new(Reduction.scalar_shape(), value))>:
  Totality.Status{
    {\{Allocation.TensorDone: scalar => same =>
      %same : {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape> & Core.Tensor<Reduction.scalar_shape()>, Reduction.scalar_result(shape, tensor, _)) == True{} : Bool}
      {==}} : Allocation.TensorDone<Reduction.scalar_shape(), Core.new(Reduction.scalar_shape(), value)> ->
        {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape> & Core.Tensor<Reduction.scalar_shape()>, Reduction.scalar_result(shape, tensor, Core.new(Reduction.scalar_shape(), value))) == True{} : Bool}}(
          AllocationProof.core_new_done(Reduction.scalar_shape(), value))}

def sum_result_status(+shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>,
  -status: Totality.Status<Core.Error, Core.Tensor<shape> & F32, result>) ->
  Totality.Status<Core.Error, Core.Tensor<shape> & Core.Tensor<Reduction.scalar_shape()>, Reduction.sum_result(shape, result)>:
  match result:
    case Fail{error}:
      Totality.Status{{\{Totality.Status: successful => Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))} :
        Totality.Status<Core.Error, Core.Tensor<shape> & F32, Fail{error}> -> {False{} == True{} : Bool}}(status)}
    case Done{pair}:
      (tensor, +value) = pair
      sum_allocated_status(shape, tensor, value)

def Laws.sum_valid_success(shape, tensor):
  success(Core.Error, Core.Tensor<shape> & Core.Tensor<Reduction.scalar_shape()>, Reduction.sum(shape, tensor),
    sum_result_status(shape, Reduction.natural_sum_loop(Shape.elements(shape), shape, tensor, 0n, 0.0),
      sum_loop_status(Shape.elements(shape), shape, tensor, 0n, 0.0,
        %Equal.sym(Nat, Nat.add(Shape.elements(shape), 0n), Shape.elements(shape), add_zero(Shape.elements(shape))) : Shape.LE(_, Shape.elements(shape))
        Shape.LE.from(Shape.elements(shape), Shape.elements(shape), Equal.sym(Bool, Nat.is_le(Shape.elements(shape), Shape.elements(shape)), True{}, Conversion.le_reflexive(Shape.elements(shape)))))))

def sum_packed_status(+shape: Shape.Shape,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape> & Core.Tensor<Reduction.scalar_shape()>>,
  -certificate: Totality.Success<Core.Error, Core.Tensor<shape> & Core.Tensor<Reduction.scalar_shape()>, result>) ->
  Totality.Status<Core.Error, Core.Dynamic & Core.Dynamic, Reduction.pack_result(shape, result)>:
  match result:
    case Fail{error}:
      Totality.Status{{\{Totality.Success: successful => witness => Empty.absurd({False{} == True{} : Bool}, Arith.clash(successful))} :
        Totality.Success<Core.Error, Core.Tensor<shape> & Core.Tensor<Reduction.scalar_shape()>, Fail{error}> -> {False{} == True{} : Bool}}(certificate)}
    case Done{pair}:
      (tensor, scalar) = pair
      Totality.Status{{==}}

def Laws.sum_dynamic_valid_success(tensor):
  Core.Packed{+shape, owner} = tensor
  success(Core.Error, Core.Dynamic & Core.Dynamic, Reduction.pack_result(shape, Reduction.sum(shape, owner)),
    sum_packed_status(shape, Reduction.sum(shape, owner), Laws.sum_valid_success(shape, owner)))

def Laws.sum_valid_result(shape, tensor):
  Laws.TotalSum{Laws.sum_valid_success(shape, tensor), Laws.successful_sum_interpretation(shape, tensor)}

def Laws.sum_dynamic_valid_result(tensor):
  Laws.TotalDynamicSum{Laws.sum_dynamic_valid_success(tensor), Laws.successful_dynamic_sum_interpretation(tensor)}
