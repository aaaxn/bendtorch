import Base
import ../shape/core.bend as Shape
import ../storage/arith.bend as Arith
import ../tensor/core.bend as Core
import ../tensor/layout.bend as Layout
import ../tensor/transforms.bend as Transforms
import ./reduce.bend as Reduce
import ./sum-dims.bend as SumDims
import ./sum-dims-LAWS.bend as Laws

def Laws.sum_dims_every_axis(axis):
  {==}

def Laws.sum_dims_listed_axis(head, tail, axis):
  {==}

def Laws.sum_dims_kept_start(axes, dimensions):
  {==}

def Laws.sum_dims_kept_step(axes, extent, rest, axis):
  {==}

def Laws.sum_dims_dropped_start(axes, dimensions):
  {==}

def Laws.sum_dims_dropped_reduced(axes, extent, rest, axis, reduced):
  %Equal.sym(Bool, SumDims.reduces(axes, axis), True{}, reduced) :
    {SumDims.dropped_next(_, extent, SumDims.dropped_from(axes, rest, U32.add(axis, 1))) ==
      SumDims.dropped_from(axes, rest, U32.add(axis, 1)) : List<&2, U32>}
  {==}

def Laws.sum_dims_dropped_kept(axes, extent, rest, axis, kept):
  %Equal.sym(Bool, SumDims.reduces(axes, axis), False{}, kept) :
    {SumDims.dropped_next(_, extent, SumDims.dropped_from(axes, rest, U32.add(axis, 1))) ==
      extent <> SumDims.dropped_from(axes, rest, U32.add(axis, 1)) : List<&2, U32>}
  {==}

def Laws.sum_dims_rank_zero_limit():
  {==}

def Laws.sum_dims_rank_limit(extent, rest):
  {==}

def Laws.sum_dims_axis_out_of_bounds(axes, limit, outside):
  %Equal.sym(Bool, Layout.bounded(axes, limit), False{}, outside) :
    {SumDims.refused(_, Layout.unique(axes)) == Some{Layout.AxisOutOfBounds{}} : Maybe<&2, Layout.Error>}
  {==}

def Laws.sum_dims_duplicate_axis(axes, limit, inside, repeated):
  %Equal.sym(Bool, Layout.bounded(axes, limit), True{}, inside) :
    {SumDims.refused(_, Layout.unique(axes)) == Some{Layout.DuplicateAxis{}} : Maybe<&2, Layout.Error>}
  %Equal.sym(Bool, Layout.unique(axes), False{}, repeated) :
    {SumDims.refused(True{}, _) == Some{Layout.DuplicateAxis{}} : Maybe<&2, Layout.Error>}
  {==}

def Laws.sum_dims_axes_accepted(axes, limit, inside, unique):
  %Equal.sym(Bool, Layout.bounded(axes, limit), True{}, inside) :
    {SumDims.refused(_, Layout.unique(axes)) == None{} : Maybe<&2, Layout.Error>}
  %Equal.sym(Bool, Layout.unique(axes), True{}, unique) :
    {SumDims.refused(True{}, _) == None{} : Maybe<&2, Layout.Error>}
  {==}

def Laws.sum_dims_plan_refused(shape, axes, error, refused):
  %Equal.sym(Maybe<&2, Layout.Error>, SumDims.refusal(axes, SumDims.limit(Shape.dimensions(shape))), Some{error}, refused) :
    {SumDims.plan_checked(axes, Shape.dimensions(shape), _) == Fail{SumDims.TransformError{Transforms.LayoutError{error}}} :
      Result<&2, &2, SumDims.Error, SumDims.Plan>}
  {==}

def built_keeps(+axes: List<&2, U32>, +dimensions: List<&2, U32>,
  built: Shape.ConstructResult<SumDims.kept_dimensions(axes, dimensions)>) ->
  Laws.PlanKeeps(axes, dimensions,
    SumDims.plan_built(SumDims.kept_dimensions(axes, dimensions), SumDims.dropped_dimensions(axes, dimensions), built)):
  match built:
    case Shape.SError{error}:
      Unit{}
    case Shape.SBuilt{+kept, preserved}:
      (preserved, {==})

def checked_keeps(+axes: List<&2, U32>, +dimensions: List<&2, U32>, checked: Maybe<&2, Layout.Error>) ->
  Laws.PlanKeeps(axes, dimensions, SumDims.plan_checked(axes, dimensions, checked)):
  match checked:
    case Some{error}:
      Unit{}
    case None{}:
      built_keeps(axes, dimensions, Shape.construct(SumDims.kept_dimensions(axes, dimensions)))

def Laws.sum_dims_plan_dimensions(shape, axes):
  checked_keeps(axes, Shape.dimensions(shape), SumDims.refusal(axes, SumDims.limit(Shape.dimensions(shape))))

def Laws.sum_dims_refused(input, axes, keepdim, error, refused):
  Core.Packed{+shape, tensor} = input
  %Equal.sym(Result<&2, &2, SumDims.Error, SumDims.Plan>, SumDims.plan(shape, axes), Fail{error}, refused) :
    {SumDims.sum_checked(Core.Packed{shape, tensor}, keepdim, _) == (Core.Packed{shape, tensor}, Fail{error}) :
      Core.Dynamic & Result<&2, &1, SumDims.Error, Core.Dynamic>}
  {==}

# With keepdim, summed only renames reduce_to's error.
def kept_as_sum(+dropped: List<&2, U32>, pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>) ->
  {SumDims.summed(dropped, True{}, pair) == Laws.as_sum(pair) : Core.Dynamic & Result<&2, &1, SumDims.Error, Core.Dynamic>}:
  match pair:
    case Tuple{input, Fail{error}}:
      {==}
    case Tuple{input, Done{output}}:
      {==}

def Laws.sum_dims_keepdim(input, axes, kept, dropped, planned):
  Core.Packed{+shape, tensor} = input
  %Equal.sym(Result<&2, &2, SumDims.Error, SumDims.Plan>, SumDims.plan(shape, axes), Done{SumDims.Plan{kept, dropped}}, planned) :
    {SumDims.sum_checked(Core.Packed{shape, tensor}, True{}, _) == Laws.as_sum(Reduce.reduce_to(Core.Packed{shape, tensor}, kept)) :
      Core.Dynamic & Result<&2, &1, SumDims.Error, Core.Dynamic>}
  kept_as_sum(dropped, Reduce.reduce_to(Core.Packed{shape, tensor}, kept))

def reshape_renamed(result: Result<&2, &1, Transforms.Error, Core.Dynamic>) ->
  {SumDims.transformed(result) == Laws.reshape_as_sum(result) : Result<&2, &1, SumDims.Error, Core.Dynamic>}:
  match result:
    case Fail{error}:
      {==}
    case Done{output}:
      {==}

# Without keepdim, summed reshapes what it gives with keepdim.
def dropped_reshaped(+dropped: List<&2, U32>, pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>) ->
  {SumDims.summed(dropped, False{}, pair) == Laws.reshaped(dropped, SumDims.summed(dropped, True{}, pair)) :
    Core.Dynamic & Result<&2, &1, SumDims.Error, Core.Dynamic>}:
  match pair:
    case Tuple{input, Fail{error}}:
      {==}
    case Tuple{input, Done{output}}:
      Equal.cong(Result<&2, &1, SumDims.Error, Core.Dynamic>, Core.Dynamic & Result<&2, &1, SumDims.Error, Core.Dynamic>,
        result => (input, result), SumDims.transformed(Transforms.reshape_dynamic(output, dropped)),
        Laws.reshape_as_sum(Transforms.reshape_dynamic(output, dropped)), reshape_renamed(Transforms.reshape_dynamic(output, dropped)))

def Laws.sum_dims_drops(input, axes, kept, dropped, planned):
  Core.Packed{+shape, tensor} = input
  %Equal.sym(Result<&2, &2, SumDims.Error, SumDims.Plan>, SumDims.plan(shape, axes), Done{SumDims.Plan{kept, dropped}}, planned) :
    {SumDims.sum_checked(Core.Packed{shape, tensor}, False{}, _) ==
      Laws.reshaped(dropped, SumDims.sum_checked(Core.Packed{shape, tensor}, True{}, _)) :
      Core.Dynamic & Result<&2, &1, SumDims.Error, Core.Dynamic>}
  dropped_reshaped(dropped, Reduce.reduce_to(Core.Packed{shape, tensor}, kept))
