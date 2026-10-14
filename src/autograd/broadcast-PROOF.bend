import Base
import ../shape/core.bend as Shape
import ../shape/LAWS.bend as ShapeLaws
import ../shape/plan_proofs.bend as PlanProof
import ../shape/nat_proofs.bend as Nats
import ../storage/arith.bend as Arith
import ../tensor/core.bend as Core
import ../tensor/layout.bend as Layout
import ../tensor/elementwise.bend as Elementwise
import ../tensor/elementwise-totality.bend as Totality
import ../tensor/elementwise-PROOF.bend as ElementwiseProof
import ../tensor/core-PROOF.bend as CoreProof
import ../tensor/transform-PROOF.bend as TransformProof
import ../tensor/projection-LAWS.bend as Projection
import ../tensor/projection-PROOF.bend as ProjectionProof
import ../tensor/allocation-spec.bend as Allocation
import ../tensor/allocation-PROOF.bend as AllocationProof
import ./broadcast.bend as Broadcast
import ./broadcast-LAWS.bend as Laws

type Holds<-P: Type> is Data:
  Holds{-proof: P}

def Laws.keep_left_scalar(left, right):
  {==}

def finish_retains(-input: Core.Dynamic, -right: Core.Dynamic,
  pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  -retention: Totality.Retention<input, right, pair>) -> Laws.Retained<input, Broadcast.finish(pair)>:
  ((left, other), result) = pair
  match result:
    case Fail{error}:
      Laws.Retained{{\{Totality.Retention: l => r => l} : Totality.Retention<input, right, ((left, other), Fail{error})> -> Totality.SameSource(input, left)}(retention)}
    case Done{output}:
      Laws.Retained{{\{Totality.Retention: l => r => l} : Totality.Retention<input, right, ((left, other), Done{output})> -> Totality.SameSource(input, left)}(retention)}

def zeroed_retains(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  result: Result<&2, &1, Core.Error, Core.Tensor<target>>) -> Laws.Retained<input, Broadcast.zeroed(input, target, plan, result)>:
  match result:
    case Fail{error}:
      Laws.Retained{property => {==}}
    case Done{zeros}:
      finish_retains(input, Core.Packed{target, zeros}, Elementwise.allocated(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros}, Core.new(Shape.output_shape(plan), 0.0)),
        ElementwiseProof.allocated_retains(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros}, Core.new(Shape.output_shape(plan), 0.0)))

def fitted_retains(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan, same: Bool) ->
  Laws.Retained<input, Broadcast.fitted(input, target, plan, same)>:
  match same:
    case False{}:
      Laws.Retained{property => {==}}
    case True{}:
      zeroed_retains(input, target, plan, Core.new(target, 0.0))

def planned_retains(input: Core.Dynamic, +target: Shape.Shape, result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) ->
  Laws.Retained<input, Broadcast.planned(input, target, result)>:
  match result:
    case Fail{error}:
      Laws.Retained{property => {==}}
    case Done{+plan}:
      fitted_retains(input, target, plan, Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(target)))

def Laws.broadcast_retains_input(input, target):
  Core.Packed{+shape, tensor} = input
  planned_retains(Core.Packed{shape, tensor}, target, Shape.broadcast_checked(shape, target))

type WithModel<-input: Core.Dynamic, -target: Shape.Shape, -plan: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>,
  -pair: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>> is Data:
  WithModel{-mapping: Laws.MappingWith(input, target, plan, pair)}

def keep_left_expected(-right: Core.Dynamic, -right_index: U32, +left: Maybe<&2, F32>, -value: F32,
  -present: {Some{value} == Core.checked_observe_dynamic(right, right_index) : Maybe<&2, F32>}) ->
  Holds<{Elementwise.expected_left(Elementwise.KeepLeft{}, right, right_index, left) == left : Maybe<&2, F32>}>:
  match left:
    case None{}:
      Holds{{==}}
    case Some{+found}:
      Holds{
        %present : {Elementwise.expected_right(Elementwise.KeepLeft{}, found, _) == Some{found} : Maybe<&2, F32>}
        {==}}

def keep_left_source(+plan: Shape.BroadcastPlan, +index: U32, -input: Core.Dynamic, -right: Core.Dynamic,
  -left_index: U32, -right_index: U32, -value: F32,
  -exact: {Done{(left_index, right_index)} == Elementwise.input_indices(plan, index) : Result<&2, &1, Core.Error, U32 & U32>},
  -present: {Some{value} == Core.checked_observe_dynamic(right, right_index) : Maybe<&2, F32>}) ->
  Holds<{Elementwise.expected(Elementwise.KeepLeft{}, plan, input, right, index) ==
    Laws.source_value(input, Projection.offset_spec(plan, U32.to_nat(index))) : Maybe<&2, F32>}>:
  Holds{Equal.trans(Maybe<&2, F32>, Elementwise.expected(Elementwise.KeepLeft{}, plan, input, right, index),
    Projection.specified_expected(Elementwise.KeepLeft{}, input, right, Projection.offset_spec(plan, U32.to_nat(index))),
    Laws.source_value(input, Projection.offset_spec(plan, U32.to_nat(index))),
    Projection.expected_projection_refines(Elementwise.KeepLeft{}, plan, input, right, index),
    %Equal.trans(Maybe<&1, U32 & U32>, Some{(left_index, right_index)}, Projection.indices_result(Elementwise.input_indices(plan, index)),
      Projection.offset_spec(plan, U32.to_nat(index)),
      Equal.cong(Result<&2, &1, Core.Error, U32 & U32>, Maybe<&1, U32 & U32>, Projection.indices_result,
        Done{(left_index, right_index)}, Elementwise.input_indices(plan, index), exact),
      Projection.input_indices_refines(plan, index)) :
      {Projection.specified_expected(Elementwise.KeepLeft{}, input, right, _) == Laws.source_value(input, _) : Maybe<&2, F32>}
    {\{Holds: proof => proof} : Holds<{Elementwise.expected_left(Elementwise.KeepLeft{}, right, right_index, Core.checked_observe_dynamic(input, left_index)) ==
      Core.checked_observe_dynamic(input, left_index) : Maybe<&2, F32>}> ->
      {Elementwise.expected_left(Elementwise.KeepLeft{}, right, right_index, Core.checked_observe_dynamic(input, left_index)) ==
        Core.checked_observe_dynamic(input, left_index) : Maybe<&2, F32>}}(
      keep_left_expected(right, right_index, Core.checked_observe_dynamic(input, left_index), value, present)))}

def mapped_value(+plan: Shape.BroadcastPlan, +index: U32, -input: Core.Dynamic, +target: Shape.Shape, -zeros: Core.Tensor<target>,
  -right_dims: {Shape.dimensions(Shape.right_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  -indices: ElementwiseProof.IndicesDone<plan, Elementwise.input_indices(plan, index)>) ->
  Holds<{Elementwise.expected(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros}, index) ==
    Laws.source_value(input, Projection.offset_spec(plan, U32.to_nat(index))) : Maybe<&2, F32>}>:
  Holds{{\{ElementwiseProof.IndicesDone: li => ri => exact => lb => rb =>
    {\{TransformProof.PresentObservation: value => equal =>
      {\{Holds: proof => proof} : Holds<{Elementwise.expected(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros}, index) ==
    Laws.source_value(input, Projection.offset_spec(plan, U32.to_nat(index))) : Maybe<&2, F32>}> -> {Elementwise.expected(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros}, index) ==
    Laws.source_value(input, Projection.offset_spec(plan, U32.to_nat(index))) : Maybe<&2, F32>}}(
        keep_left_source(plan, index, input, Core.Packed{target, zeros}, li, ri, value, exact,
          Equal.sym(Maybe<&2, F32>, Core.checked_observe(target, zeros, ri), Some{value}, equal)))} :
      TransformProof.PresentObservation<target, zeros, ri> -> {Elementwise.expected(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros}, index) ==
    Laws.source_value(input, Projection.offset_spec(plan, U32.to_nat(index))) : Maybe<&2, F32>}}(
      TransformProof.present_observation(target, zeros, ri,
        %right_dims : {Nat.is_lt(U32.to_nat(ri), Shape.logical_numel(_)) == True{} : Bool}
        rb))} :
    ElementwiseProof.IndicesDone<plan, Elementwise.input_indices(plan, index)> -> {Elementwise.expected(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros}, index) ==
    Laws.source_value(input, Projection.offset_spec(plan, U32.to_nat(index))) : Maybe<&2, F32>}}(indices)}

def output_inside(+plan: Shape.BroadcastPlan, +target: Shape.Shape, +index: U32,
  -same_dims: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  -inside: Shape.LT(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target)))) ->
  Holds<{Nat.is_lt(U32.to_nat(index), Shape.elements(Shape.output_shape(plan))) == True{} : Bool}>:
  Holds{
    %Equal.sym(Nat, Shape.elements(Shape.output_shape(plan)), Shape.numel(Shape.extents(Shape.output_shape(plan))), Shape.elements_exact(Shape.output_shape(plan))) :
      {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
    %Equal.sym(List<&2, U32>, Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(target), same_dims) :
      {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(_)) == True{} : Bool}
    Equal.sym(Bool, True{}, Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))),
      Nats.lt_true(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target)), inside))}

def finish_mapping(-input: Core.Dynamic, +target: Shape.Shape, -zeros: Core.Tensor<target>, +plan: Shape.BroadcastPlan,
  pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  -dims: Totality.DimensionsModel<Shape.dimensions(Shape.output_shape(plan)), pair>,
  -mapping: Elementwise.MappingModel<Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros}, pair>,
  -same_dims: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  -right_dims: {Shape.dimensions(Shape.right_shape(plan)) == Shape.dimensions(target) : List<&2, U32>}) ->
  WithModel<input, target, Done{plan}, Broadcast.finish(pair)>:
  ((left, right), result) = pair
  match result:
    case Fail{error}:
      WithModel{Unit{}}
    case Done{output}:
      WithModel{(
        Equal.trans(List<&2, U32>, Totality.dimensions(output), Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(target),
          {\{Totality.DimensionsModel: exact => exact} : Totality.DimensionsModel<Shape.dimensions(Shape.output_shape(plan)), ((left, right), Done{output})> ->
            {Totality.dimensions(output) == Shape.dimensions(Shape.output_shape(plan)) : List<&2, U32>}}(dims),
          same_dims),
        index => inside => Equal.trans(Maybe<&2, F32>, Core.observe_dynamic(output, index),
          Elementwise.expected(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros}, index),
          Laws.source_value(input, Projection.offset_spec(plan, U32.to_nat(index))),
          {\{Elementwise.MappingModel: prefix => prefix} :
            Elementwise.MappingModel<Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros}, ((left, right), Done{output})> ->
            Elementwise.PrefixMapping(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros},
              Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), output)}(mapping)(index)(
            %Equal.sym(List<&2, U32>, Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(target), same_dims) :
              Shape.LT(U32.to_nat(index), Shape.logical_numel(_))
            inside),
          {\{Holds: proof => proof} : Holds<{Elementwise.expected(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros}, index) ==
    Laws.source_value(input, Projection.offset_spec(plan, U32.to_nat(index))) : Maybe<&2, F32>}> -> {Elementwise.expected(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros}, index) ==
    Laws.source_value(input, Projection.offset_spec(plan, U32.to_nat(index))) : Maybe<&2, F32>}}(
            mapped_value(plan, index, input, target, zeros, right_dims,
              ElementwiseProof.indices_success(plan, index,
                {\{Holds: proof => proof} : Holds<{Nat.is_lt(U32.to_nat(index), Shape.elements(Shape.output_shape(plan))) == True{} : Bool}> ->
                  {Nat.is_lt(U32.to_nat(index), Shape.elements(Shape.output_shape(plan))) == True{} : Bool}}(
                  output_inside(plan, target, index, same_dims, inside))))))
      )}

def zeroed_mapping(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  result: Result<&2, &1, Core.Error, Core.Tensor<target>>,
  -same_dims: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  -right_dims: {Shape.dimensions(Shape.right_shape(plan)) == Shape.dimensions(target) : List<&2, U32>}) ->
  WithModel<input, target, Done{plan}, Broadcast.zeroed(input, target, plan, result)>:
  match result:
    case Fail{error}:
      WithModel{Unit{}}
    case Done{zeros}:
      finish_mapping(input, target, zeros, plan, Elementwise.allocated(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros}, Core.new(Shape.output_shape(plan), 0.0)),
        ElementwiseProof.allocated_dimensions(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros}, Core.new(Shape.output_shape(plan), 0.0)),
        CoreProof.successful_allocated_mapping(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros}),
        same_dims, right_dims)

def fitted_mapping(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan, same: Bool,
  -decided: {same == Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(target)) : Bool},
  -right_dims: {Shape.dimensions(Shape.right_shape(plan)) == Shape.dimensions(target) : List<&2, U32>}) ->
  WithModel<input, target, Done{plan}, Broadcast.fitted(input, target, plan, same)>:
  match same:
    case False{}:
      WithModel{Unit{}}
    case True{}:
      zeroed_mapping(input, target, plan, Core.new(target, 0.0),
        TransformProof.words_equal(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(target), decided), right_dims)

def planned_mapping(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>,
  -sources: {ShapeLaws.planned_dimensions(shape, target, result) == (Shape.dimensions(shape), Shape.dimensions(target)) : List<&2, U32> & List<&2, U32>}) ->
  WithModel<Core.Packed{shape, tensor}, target, result, Broadcast.planned(Core.Packed{shape, tensor}, target, result)>:
  match result:
    case Fail{error}:
      WithModel{Unit{}}
    case Done{+plan}:
      fitted_mapping(Core.Packed{shape, tensor}, target, plan,
        Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(target)), {==},
        Equal.cong(List<&2, U32> & List<&2, U32>, List<&2, U32>, Pair.snd(List<&2, U32>, List<&2, U32>),
          (Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(Shape.right_shape(plan))), (Shape.dimensions(shape), Shape.dimensions(target)), sources))

def Laws.broadcast_mapping(input, target):
  Core.Packed{+shape, tensor} = input
  Laws.MappingModel{
    {\{WithModel: mapping => mapping} :
      WithModel<Core.Packed{shape, tensor}, target, Shape.broadcast_checked(shape, target), Broadcast.broadcast_to(Core.Packed{shape, tensor}, target)> ->
        Laws.Mapping(Core.Packed{shape, tensor}, target, Broadcast.broadcast_to(Core.Packed{shape, tensor}, target))}(
          planned_mapping(shape, tensor, target, Shape.broadcast_checked(shape, target), PlanProof.broadcast_checked_plan(shape, target)))}

def finish_success(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  -success: Totality.Success<pair>) -> Laws.Success<Broadcast.finish(pair)>:
  ((input, zeros), result) = pair
  match result:
    case Fail{error}:
      Laws.Success{Empty.absurd({False{} == True{} : Bool}, Arith.clash(
        {\{Totality.Success: successful => successful} : Totality.Success<((input, zeros), Fail{error})> -> {False{} == True{} : Bool}}(success)))}
    case Done{output}:
      Laws.Success{{==}}

def zeroed_success(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  result: Result<&2, &1, Core.Error, Core.Tensor<target>>, -done: Allocation.TensorDone<target, result>,
  -left_dims: {Totality.dimensions(input) == Shape.dimensions(Shape.left_shape(plan)) : List<&2, U32>},
  -right_dims: {Shape.dimensions(target) == Shape.dimensions(Shape.right_shape(plan)) : List<&2, U32>}) ->
  Laws.Success<Broadcast.zeroed(input, target, plan, result)>:
  match result:
    case Fail{error}:
      Laws.Success{Empty.absurd({False{} == True{} : Bool},
        {\{Allocation.TensorDone: tensor => same => Arith.clash(Equal.cong(Result<&2, &1, Core.Error, Core.Tensor<target>>, Bool,
          found => Result.is_done(&2, &1, Core.Error, Core.Tensor<target>, found), Fail{error}, Done{tensor},
          Equal.sym(Result<&2, &1, Core.Error, Core.Tensor<target>>, Done{tensor}, Fail{error}, same)))} :
          Allocation.TensorDone<target, Fail{error}> -> Empty}(done))}
    case Done{zeros}:
      finish_success(Elementwise.allocated(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros}, Core.new(Shape.output_shape(plan), 0.0)),
        ElementwiseProof.allocated_success(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros}, Core.new(Shape.output_shape(plan), 0.0),
          ElementwiseProof.FittingSources{left_dims, right_dims}, AllocationProof.core_new_done(Shape.output_shape(plan), 0.0)))

def fitted_success(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan, same: Bool,
  -decided: {same == Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(target)) : Bool},
  -truth: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  -left_dims: {Totality.dimensions(input) == Shape.dimensions(Shape.left_shape(plan)) : List<&2, U32>},
  -right_dims: {Shape.dimensions(target) == Shape.dimensions(Shape.right_shape(plan)) : List<&2, U32>}) ->
  Laws.Success<Broadcast.fitted(input, target, plan, same)>:
  match same:
    case False{}:
      Laws.Success{Empty.absurd({False{} == True{} : Bool},
        Arith.clash(Equal.trans(Bool, False{}, Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(target)), True{}, decided,
          %truth : {Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), _) == True{} : Bool}
          TransformProof.words_same_refl(Shape.dimensions(Shape.output_shape(plan))))))}
    case True{}:
      zeroed_success(input, target, plan, Core.new(target, 0.0), AllocationProof.core_new_done(target, 0.0), left_dims, right_dims)

def planned_success(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>,
  -sources: {ShapeLaws.planned_dimensions(shape, target, result) == (Shape.dimensions(shape), Shape.dimensions(target)) : List<&2, U32> & List<&2, U32>},
  -built: {ShapeLaws.broadcast_built(result) == True{} : Bool},
  -alignment: Shape.Broadcast<Shape.extents(shape), Shape.extents(target), Shape.extents(target)>) ->
  Laws.Success<Broadcast.planned(Core.Packed{shape, tensor}, target, result)>:
  match result:
    case Fail{error}:
      Laws.Success{Empty.absurd({False{} == True{} : Bool}, Arith.clash(built))}
    case Done{+plan}:
      fitted_success(Core.Packed{shape, tensor}, target, plan,
        Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(target)), {==},
        ElementwiseProof.planned_output_dimensions(shape, target, plan, Shape.dimensions(target), alignment, sources),
        Equal.sym(List<&2, U32>, Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(shape),
          Equal.cong(List<&2, U32> & List<&2, U32>, List<&2, U32>, Pair.fst(List<&2, U32>, List<&2, U32>),
            (Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(Shape.right_shape(plan))), (Shape.dimensions(shape), Shape.dimensions(target)), sources)),
        Equal.sym(List<&2, U32>, Shape.dimensions(Shape.right_shape(plan)), Shape.dimensions(target),
          Equal.cong(List<&2, U32> & List<&2, U32>, List<&2, U32>, Pair.snd(List<&2, U32>, List<&2, U32>),
            (Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(Shape.right_shape(plan))), (Shape.dimensions(shape), Shape.dimensions(target)), sources)))

def rank_of(+shape: Shape.Shape) -> Shape.LE(Shape.rank_dimensions(Shape.dimensions(shape)), 4n):
  match shape:
    case Shape.SShape{dims, size, exact, rank_bound, capacity_bound}:
      rank_bound

def capacity_of(+shape: Shape.Shape) -> Shape.LE(Shape.logical_numel(Shape.dimensions(shape)), 1048576n):
  match shape:
    case Shape.SShape{dims, size, exact, rank_bound, capacity_bound}:
      capacity_bound

def Laws.broadcast_valid_success(input, target, alignment):
  Core.Packed{+shape, tensor} = input
  planned_success(shape, tensor, target, Shape.broadcast_checked(shape, target), PlanProof.broadcast_checked_plan(shape, target),
    PlanProof.broadcast_checked_complete(shape, target, Shape.dimensions(target), alignment, rank_of(target), capacity_of(target)), alignment)

# Error identity: a planner failure is passed through, and a plan whose output
# is not the target selects NotBroadcastable.

def Laws.broadcast_shape_error(input, target, error, rejected):
  Core.Packed{+shape, tensor} = input
  %Equal.sym(Result<&2, &2, Shape.Error, Shape.BroadcastPlan>, Shape.broadcast_checked(shape, target), Fail{error}, rejected) :
    {Broadcast.planned(Core.Packed{shape, tensor}, target, _) == (Core.Packed{shape, tensor}, Fail{Broadcast.TensorError{Core.ShapeError{error}}}) :
      Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>}
  {==}

def fitted_not_target(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan, flag: Bool,
  decided: {flag == Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(target)) : Bool},
  differ: {Shape.dimensions(Shape.output_shape(plan)) != Shape.dimensions(target) : List<&2, U32>}) ->
  {Broadcast.fitted(input, target, plan, flag) == (input, Fail{Broadcast.NotBroadcastable{}}) :
    Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>}:
  match flag:
    case False{}:
      {==}
    case True{}:
      Empty.absurd({Broadcast.fitted(input, target, plan, True{}) == (input, Fail{Broadcast.NotBroadcastable{}}) :
        Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>},
        differ(TransformProof.words_equal(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(target), decided)))

def Laws.broadcast_not_target(input, target, plan, planned, differ):
  Core.Packed{+shape, tensor} = input
  %Equal.sym(Result<&2, &2, Shape.Error, Shape.BroadcastPlan>, Shape.broadcast_checked(shape, target), Done{plan}, planned) :
    {Broadcast.planned(Core.Packed{shape, tensor}, target, _) == (Core.Packed{shape, tensor}, Fail{Broadcast.NotBroadcastable{}}) :
      Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>}
  fitted_not_target(Core.Packed{shape, tensor}, target, plan,
    Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(target)), {==}, differ)
