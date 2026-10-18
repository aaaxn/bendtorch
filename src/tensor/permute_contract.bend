import Base
import ../shape/core.bend as Shape
import ../shape/LAWS.bend as ShapeLaws
import ../shape/PROOF.bend as ShapeProof
import ../shape/nat_proofs.bend as Nats
import ../storage/main.bend as Storage
import ../storage/LAWS.bend as StorageLaws
import ../storage/PROOF.bend as StorageProof
import ../storage/arith.bend as Arith
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./layout.bend as Layout
import ./transpose_spec.bend as Transpose
import ./transforms.bend as Transforms
import ./allocation_PROOF.bend as AllocationProof
import ./word_conversion.bend as Conversion
import ./read_contract.bend as Reads
import ./transform_observation.bend as Observation
import ./transform_records.bend as Laws

def retained_code(+input: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(input), fill)) == Shape.logical_numel(Shape.dimensions(input)) : Nat}, +request: Observation.Request, +error: Transforms.Error,
  failed: {Observation.apply(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, request) == (Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, Fail{error}) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}) ->
  {Observation.failure_code(input, request, Done{Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}}) == Some{Transforms.error_code(error)} : Maybe<&2, U32>}:
  Equal.cong(Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>, Maybe<&2, U32>, pair => Some{Observation.failure_code_pair(input, pair)},
    Observation.apply(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, request), (Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, Fail{error}), failed)

def retained_dimensions(+input: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(input), fill)) == Shape.logical_numel(Shape.dimensions(input)) : Nat}, +request: Observation.Request, +error: Transforms.Error,
  failed: {Observation.apply(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, request) == (Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, Fail{error}) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}) ->
  {Observation.kept_dimensions(input, request, Done{Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}}) == Some{Shape.dimensions(input)} : Maybe<&2, List<&2, U32>>}:
  Equal.cong(Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>, Maybe<&2, List<&2, U32>>, pair => Observation.kept_dimensions_pair(input, pair),
    Observation.apply(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, request), (Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, Fail{error}), failed)

def retained_length(+input: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(input), fill)) == Shape.logical_numel(Shape.dimensions(input)) : Nat}, +request: Observation.Request, +error: Transforms.Error,
  failed: {Observation.apply(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, request) == (Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, Fail{error}) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}) ->
  {Observation.kept_length(input, request, Done{Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}}) == Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>}:
  Equal.trans(Maybe<&2, Nat>, Some{Observation.kept_length_pair(input, Observation.apply(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, request))},
    Some{Lifecycle.logical_length(StorageLaws.created(Shape.count(input), fill))}, Some{Shape.logical_numel(Shape.dimensions(input))},
    Equal.cong(Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>, Maybe<&2, Nat>, pair => Some{Observation.kept_length_pair(input, pair)},
      Observation.apply(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, request), (Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, Fail{error}), failed),
    Equal.cong(Nat, Maybe<&2, Nat>, number => Some{number}, Lifecycle.logical_length(StorageLaws.created(Shape.count(input), fill)), Shape.logical_numel(Shape.dimensions(input)), Reads.created_exact(input, fill)))

def retained_count(+input: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(input), fill)) == Shape.logical_numel(Shape.dimensions(input)) : Nat}, +request: Observation.Request, +error: Transforms.Error,
  failed: {Observation.apply(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, request) == (Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, Fail{error}) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}) ->
  {Observation.kept_count(input, request, Done{Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}}) == Some{Shape.count(input)} : Maybe<&2, U32>}:
  Equal.cong(Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>, Maybe<&2, U32>, pair => Observation.kept_count_pair(input, pair),
    Observation.apply(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, request), (Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, Fail{error}), failed)

def retained_value(+input: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(input), fill)) == Shape.logical_numel(Shape.dimensions(input)) : Nat}, +request: Observation.Request, +error: Transforms.Error, +other: U32,
  inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool}, failed: {Observation.apply(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, request) == (Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, Fail{error}) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}) ->
  {Observation.kept_value(input, request, other, Done{Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}}) == Some{fill} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>, Observation.kept_value_pair(input, other, Observation.apply(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, request)),
    Observation.kept_value_pair(input, other, (Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, Fail{error})), Some{fill},
    Equal.cong(Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>, Maybe<&2, F32>, pair => Observation.kept_value_pair(input, other, pair),
      Observation.apply(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, request), (Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, Fail{error}), failed),
    Reads.read_point_done(input, fill, valid, other, inside))

def ranked_failure(+input: Shape.Shape, +axes: List<&2, U32>,
  decision: Shape.BoolCase<Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))>,
  wrong: {Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))) == False{} : Bool}) ->
  {Layout.plan_ranked(input, axes, decision) == Fail{Layout.AxesLengthMismatch{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>}:
  match decision:
    case Shape.BFalse{no}:
      {==}
    case Shape.BTrue{yes}:
      Empty.absurd({Layout.plan_ranked(input, axes, Shape.BTrue{yes}) == Fail{Layout.AxesLengthMismatch{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>},
        Nats.contradiction(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))), yes,
          Equal.sym(Bool, Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))), False{}, wrong)))

def ranked_pass(+input: Shape.Shape, +axes: List<&2, U32>,
  decision: Shape.BoolCase<Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))>,
  ranked: {Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))) == True{} : Bool}) ->
  {Layout.plan_ranked(input, axes, decision) == Layout.plan_bounded(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input)))) : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>}:
  match decision:
    case Shape.BFalse{no}:
      Empty.absurd({Layout.plan_ranked(input, axes, Shape.BFalse{no}) == Layout.plan_bounded(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input)))) : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>},
        Nats.contradiction(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))),
          Equal.sym(Bool, Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))), True{}, ranked), no))
    case Shape.BTrue{yes}:
      {==}

def bounded_failure(+input: Shape.Shape, +axes: List<&2, U32>, decision: Shape.BoolCase<Layout.bounded(axes, Shape.rank(input))>,
  outside: {Layout.bounded(axes, Shape.rank(input)) == False{} : Bool}) ->
  {Layout.plan_bounded(input, axes, decision) == Fail{Layout.AxisOutOfBounds{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>}:
  match decision:
    case Shape.BFalse{no}:
      {==}
    case Shape.BTrue{yes}:
      Empty.absurd({Layout.plan_bounded(input, axes, Shape.BTrue{yes}) == Fail{Layout.AxisOutOfBounds{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>},
        Nats.contradiction(Layout.bounded(axes, Shape.rank(input)), yes, Equal.sym(Bool, Layout.bounded(axes, Shape.rank(input)), False{}, outside)))

def bounded_pass(+input: Shape.Shape, +axes: List<&2, U32>, decision: Shape.BoolCase<Layout.bounded(axes, Shape.rank(input))>,
  bounded: {Layout.bounded(axes, Shape.rank(input)) == True{} : Bool}) ->
  {Layout.plan_bounded(input, axes, decision) == Layout.plan_unique(input, axes, Shape.bool_case(Layout.unique(axes))) : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>}:
  match decision:
    case Shape.BFalse{no}:
      Empty.absurd({Layout.plan_bounded(input, axes, Shape.BFalse{no}) == Layout.plan_unique(input, axes, Shape.bool_case(Layout.unique(axes))) : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>},
        Nats.contradiction(Layout.bounded(axes, Shape.rank(input)), Equal.sym(Bool, Layout.bounded(axes, Shape.rank(input)), True{}, bounded), no))
    case Shape.BTrue{yes}:
      {==}

def unique_failure(+input: Shape.Shape, +axes: List<&2, U32>, decision: Shape.BoolCase<Layout.unique(axes)>,
  duplicate: {Layout.unique(axes) == False{} : Bool}) ->
  {Layout.plan_unique(input, axes, decision) == Fail{Layout.DuplicateAxis{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>}:
  match decision:
    case Shape.BFalse{no}:
      {==}
    case Shape.BTrue{yes}:
      Empty.absurd({Layout.plan_unique(input, axes, Shape.BTrue{yes}) == Fail{Layout.DuplicateAxis{}} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>},
        Nats.contradiction(Layout.unique(axes), yes, Equal.sym(Bool, Layout.unique(axes), False{}, duplicate)))

def planned_failure(+input: Shape.Shape, -tensor: Core.Tensor<input>, +axes: List<&2, U32>, -error: Layout.Error,
  planned: {Layout.permutation(input, axes) == Fail{error} : Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>}) ->
  {Observation.apply(input, tensor, Observation.Permute{axes}) == (tensor, Fail{Transforms.LayoutError{error}}) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}:
  Equal.cong(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>, result => Transforms.copy_planned(input, axes, tensor, result),
    Layout.permutation(input, axes), Fail{error}, planned)

def transpose_failure(+input: Shape.Shape, -tensor: Core.Tensor<input>, +a: U32, +b: U32,
  decision: Shape.BoolCase<Transpose.valid(input, a, b)>, outside: {Transpose.valid(input, a, b) == False{} : Bool}) ->
  {Transforms.transpose_decided(input, tensor, a, b, decision) == (tensor, Fail{Transforms.LayoutError{Layout.AxisOutOfBounds{}}}) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}:
  match decision:
    case Shape.BFalse{no}:
      {==}
    case Shape.BTrue{yes}:
      Empty.absurd({Transforms.transpose_decided(input, tensor, a, b, Shape.BTrue{yes}) == (tensor, Fail{Transforms.LayoutError{Layout.AxisOutOfBounds{}}}) : Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>},
        Nats.contradiction(Transpose.valid(input, a, b), yes, Equal.sym(Bool, Transpose.valid(input, a, b), False{}, outside)))

def permute_rank_mismatch(+input: Shape.Shape, +fill: F32, +axes: List<&2, U32>, wrong: {Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))) == False{} : Bool}) -> Laws.RetainedFailure<input, fill, Observation.Permute{axes}, 12>:
  +wrong = wrong
  Laws.RetainedFailure{
    Reads.new_lift(input, fill, result => {Observation.failure_code(input, Observation.Permute{axes}, result) == Some{12} : Maybe<&2, U32>},
      valid => retained_code(input, fill, valid, Observation.Permute{axes}, Transforms.LayoutError{Layout.AxesLengthMismatch{}}, planned_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, axes, Layout.AxesLengthMismatch{}, ranked_failure(input, axes, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), wrong)))),
    Reads.new_lift(input, fill, result => {Observation.kept_dimensions(input, Observation.Permute{axes}, result) == Some{Shape.dimensions(input)} : Maybe<&2, List<&2, U32>>},
      valid => retained_dimensions(input, fill, valid, Observation.Permute{axes}, Transforms.LayoutError{Layout.AxesLengthMismatch{}}, planned_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, axes, Layout.AxesLengthMismatch{}, ranked_failure(input, axes, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), wrong)))),
    Reads.new_lift(input, fill, result => {Observation.kept_length(input, Observation.Permute{axes}, result) == Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>},
      valid => retained_length(input, fill, valid, Observation.Permute{axes}, Transforms.LayoutError{Layout.AxesLengthMismatch{}}, planned_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, axes, Layout.AxesLengthMismatch{}, ranked_failure(input, axes, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), wrong)))),
    Reads.new_lift(input, fill, result => {Observation.kept_count(input, Observation.Permute{axes}, result) == Some{Shape.count(input)} : Maybe<&2, U32>},
      valid => retained_count(input, fill, valid, Observation.Permute{axes}, Transforms.LayoutError{Layout.AxesLengthMismatch{}}, planned_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, axes, Layout.AxesLengthMismatch{}, ranked_failure(input, axes, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), wrong)))),
    other => inside => Reads.new_lift(input, fill, result => {Observation.kept_value(input, Observation.Permute{axes}, other, result) == Some{fill} : Maybe<&2, F32>},
      valid => retained_value(input, fill, valid, Observation.Permute{axes}, Transforms.LayoutError{Layout.AxesLengthMismatch{}}, other, inside, planned_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, axes, Layout.AxesLengthMismatch{}, ranked_failure(input, axes, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), wrong))))}

def permute_axis_out_of_bounds(+input: Shape.Shape, +fill: F32, +axes: List<&2, U32>, ranked: {Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))) == True{} : Bool},
  outside: {Layout.bounded(axes, Shape.rank(input)) == False{} : Bool}) -> Laws.RetainedFailure<input, fill, Observation.Permute{axes}, 13>:
  +ranked = ranked
  +outside = outside
  Laws.RetainedFailure{
    Reads.new_lift(input, fill, result => {Observation.failure_code(input, Observation.Permute{axes}, result) == Some{13} : Maybe<&2, U32>},
      valid => retained_code(input, fill, valid, Observation.Permute{axes}, Transforms.LayoutError{Layout.AxisOutOfBounds{}}, planned_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, axes, Layout.AxisOutOfBounds{},
        Equal.trans(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, Layout.permutation(input, axes),
          Layout.plan_bounded(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input)))), Fail{Layout.AxisOutOfBounds{}},
          ranked_pass(input, axes, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), ranked),
          bounded_failure(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input))), outside))))),
    Reads.new_lift(input, fill, result => {Observation.kept_dimensions(input, Observation.Permute{axes}, result) == Some{Shape.dimensions(input)} : Maybe<&2, List<&2, U32>>},
      valid => retained_dimensions(input, fill, valid, Observation.Permute{axes}, Transforms.LayoutError{Layout.AxisOutOfBounds{}}, planned_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, axes, Layout.AxisOutOfBounds{},
        Equal.trans(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, Layout.permutation(input, axes),
          Layout.plan_bounded(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input)))), Fail{Layout.AxisOutOfBounds{}},
          ranked_pass(input, axes, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), ranked),
          bounded_failure(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input))), outside))))),
    Reads.new_lift(input, fill, result => {Observation.kept_length(input, Observation.Permute{axes}, result) == Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>},
      valid => retained_length(input, fill, valid, Observation.Permute{axes}, Transforms.LayoutError{Layout.AxisOutOfBounds{}}, planned_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, axes, Layout.AxisOutOfBounds{},
        Equal.trans(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, Layout.permutation(input, axes),
          Layout.plan_bounded(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input)))), Fail{Layout.AxisOutOfBounds{}},
          ranked_pass(input, axes, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), ranked),
          bounded_failure(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input))), outside))))),
    Reads.new_lift(input, fill, result => {Observation.kept_count(input, Observation.Permute{axes}, result) == Some{Shape.count(input)} : Maybe<&2, U32>},
      valid => retained_count(input, fill, valid, Observation.Permute{axes}, Transforms.LayoutError{Layout.AxisOutOfBounds{}}, planned_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, axes, Layout.AxisOutOfBounds{},
        Equal.trans(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, Layout.permutation(input, axes),
          Layout.plan_bounded(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input)))), Fail{Layout.AxisOutOfBounds{}},
          ranked_pass(input, axes, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), ranked),
          bounded_failure(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input))), outside))))),
    other => inside => Reads.new_lift(input, fill, result => {Observation.kept_value(input, Observation.Permute{axes}, other, result) == Some{fill} : Maybe<&2, F32>},
      valid => retained_value(input, fill, valid, Observation.Permute{axes}, Transforms.LayoutError{Layout.AxisOutOfBounds{}}, other, inside, planned_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, axes, Layout.AxisOutOfBounds{},
        Equal.trans(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, Layout.permutation(input, axes),
          Layout.plan_bounded(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input)))), Fail{Layout.AxisOutOfBounds{}},
          ranked_pass(input, axes, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), ranked),
          bounded_failure(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input))), outside)))))}

def permute_duplicate_axis(+input: Shape.Shape, +fill: F32, +axes: List<&2, U32>, ranked: {Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))) == True{} : Bool},
  bounded: {Layout.bounded(axes, Shape.rank(input)) == True{} : Bool}, duplicate: {Layout.unique(axes) == False{} : Bool}) -> Laws.RetainedFailure<input, fill, Observation.Permute{axes}, 14>:
  +ranked = ranked
  +bounded = bounded
  +duplicate = duplicate
  Laws.RetainedFailure{
    Reads.new_lift(input, fill, result => {Observation.failure_code(input, Observation.Permute{axes}, result) == Some{14} : Maybe<&2, U32>},
      valid => retained_code(input, fill, valid, Observation.Permute{axes}, Transforms.LayoutError{Layout.DuplicateAxis{}}, planned_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, axes, Layout.DuplicateAxis{},
        Equal.trans(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, Layout.permutation(input, axes),
          Layout.plan_bounded(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input)))), Fail{Layout.DuplicateAxis{}},
          ranked_pass(input, axes, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), ranked),
          Equal.trans(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, Layout.plan_bounded(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input)))),
            Layout.plan_unique(input, axes, Shape.bool_case(Layout.unique(axes))), Fail{Layout.DuplicateAxis{}},
            bounded_pass(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input))), bounded),
            unique_failure(input, axes, Shape.bool_case(Layout.unique(axes)), duplicate)))))),
    Reads.new_lift(input, fill, result => {Observation.kept_dimensions(input, Observation.Permute{axes}, result) == Some{Shape.dimensions(input)} : Maybe<&2, List<&2, U32>>},
      valid => retained_dimensions(input, fill, valid, Observation.Permute{axes}, Transforms.LayoutError{Layout.DuplicateAxis{}}, planned_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, axes, Layout.DuplicateAxis{},
        Equal.trans(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, Layout.permutation(input, axes),
          Layout.plan_bounded(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input)))), Fail{Layout.DuplicateAxis{}},
          ranked_pass(input, axes, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), ranked),
          Equal.trans(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, Layout.plan_bounded(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input)))),
            Layout.plan_unique(input, axes, Shape.bool_case(Layout.unique(axes))), Fail{Layout.DuplicateAxis{}},
            bounded_pass(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input))), bounded),
            unique_failure(input, axes, Shape.bool_case(Layout.unique(axes)), duplicate)))))),
    Reads.new_lift(input, fill, result => {Observation.kept_length(input, Observation.Permute{axes}, result) == Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>},
      valid => retained_length(input, fill, valid, Observation.Permute{axes}, Transforms.LayoutError{Layout.DuplicateAxis{}}, planned_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, axes, Layout.DuplicateAxis{},
        Equal.trans(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, Layout.permutation(input, axes),
          Layout.plan_bounded(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input)))), Fail{Layout.DuplicateAxis{}},
          ranked_pass(input, axes, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), ranked),
          Equal.trans(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, Layout.plan_bounded(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input)))),
            Layout.plan_unique(input, axes, Shape.bool_case(Layout.unique(axes))), Fail{Layout.DuplicateAxis{}},
            bounded_pass(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input))), bounded),
            unique_failure(input, axes, Shape.bool_case(Layout.unique(axes)), duplicate)))))),
    Reads.new_lift(input, fill, result => {Observation.kept_count(input, Observation.Permute{axes}, result) == Some{Shape.count(input)} : Maybe<&2, U32>},
      valid => retained_count(input, fill, valid, Observation.Permute{axes}, Transforms.LayoutError{Layout.DuplicateAxis{}}, planned_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, axes, Layout.DuplicateAxis{},
        Equal.trans(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, Layout.permutation(input, axes),
          Layout.plan_bounded(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input)))), Fail{Layout.DuplicateAxis{}},
          ranked_pass(input, axes, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), ranked),
          Equal.trans(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, Layout.plan_bounded(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input)))),
            Layout.plan_unique(input, axes, Shape.bool_case(Layout.unique(axes))), Fail{Layout.DuplicateAxis{}},
            bounded_pass(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input))), bounded),
            unique_failure(input, axes, Shape.bool_case(Layout.unique(axes)), duplicate)))))),
    other => inside => Reads.new_lift(input, fill, result => {Observation.kept_value(input, Observation.Permute{axes}, other, result) == Some{fill} : Maybe<&2, F32>},
      valid => retained_value(input, fill, valid, Observation.Permute{axes}, Transforms.LayoutError{Layout.DuplicateAxis{}}, other, inside, planned_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, axes, Layout.DuplicateAxis{},
        Equal.trans(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, Layout.permutation(input, axes),
          Layout.plan_bounded(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input)))), Fail{Layout.DuplicateAxis{}},
          ranked_pass(input, axes, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input)))), ranked),
          Equal.trans(Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>, Layout.plan_bounded(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input)))),
            Layout.plan_unique(input, axes, Shape.bool_case(Layout.unique(axes))), Fail{Layout.DuplicateAxis{}},
            bounded_pass(input, axes, Shape.bool_case(Layout.bounded(axes, Shape.rank(input))), bounded),
            unique_failure(input, axes, Shape.bool_case(Layout.unique(axes)), duplicate))))))}

def transpose_out_of_bounds(+input: Shape.Shape, +fill: F32, +a: U32, +b: U32, outside: {Transpose.valid(input, a, b) == False{} : Bool}) -> Laws.RetainedFailure<input, fill, Observation.Transpose{a, b}, 13>:
  +outside = outside
  Laws.RetainedFailure{
    Reads.new_lift(input, fill, result => {Observation.failure_code(input, Observation.Transpose{a, b}, result) == Some{13} : Maybe<&2, U32>},
      valid => retained_code(input, fill, valid, Observation.Transpose{a, b}, Transforms.LayoutError{Layout.AxisOutOfBounds{}}, transpose_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, a, b, Shape.bool_case(Transpose.valid(input, a, b)), outside))),
    Reads.new_lift(input, fill, result => {Observation.kept_dimensions(input, Observation.Transpose{a, b}, result) == Some{Shape.dimensions(input)} : Maybe<&2, List<&2, U32>>},
      valid => retained_dimensions(input, fill, valid, Observation.Transpose{a, b}, Transforms.LayoutError{Layout.AxisOutOfBounds{}}, transpose_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, a, b, Shape.bool_case(Transpose.valid(input, a, b)), outside))),
    Reads.new_lift(input, fill, result => {Observation.kept_length(input, Observation.Transpose{a, b}, result) == Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>},
      valid => retained_length(input, fill, valid, Observation.Transpose{a, b}, Transforms.LayoutError{Layout.AxisOutOfBounds{}}, transpose_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, a, b, Shape.bool_case(Transpose.valid(input, a, b)), outside))),
    Reads.new_lift(input, fill, result => {Observation.kept_count(input, Observation.Transpose{a, b}, result) == Some{Shape.count(input)} : Maybe<&2, U32>},
      valid => retained_count(input, fill, valid, Observation.Transpose{a, b}, Transforms.LayoutError{Layout.AxisOutOfBounds{}}, transpose_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, a, b, Shape.bool_case(Transpose.valid(input, a, b)), outside))),
    other => inside => Reads.new_lift(input, fill, result => {Observation.kept_value(input, Observation.Transpose{a, b}, other, result) == Some{fill} : Maybe<&2, F32>},
      valid => retained_value(input, fill, valid, Observation.Transpose{a, b}, Transforms.LayoutError{Layout.AxisOutOfBounds{}}, other, inside, transpose_failure(input, Core.TTensor{StorageLaws.created(Shape.count(input), fill), valid}, a, b, Shape.bool_case(Transpose.valid(input, a, b)), outside)))}
