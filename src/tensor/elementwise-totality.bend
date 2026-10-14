import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./elementwise.bend as Elementwise

def dimensions(dynamic: Core.Dynamic) -> List<&2, U32>:
  Core.Packed{shape, tensor} = dynamic
  Shape.dimensions(shape)

def dynamic_shape(dynamic: Core.Dynamic) -> Shape.Shape:
  Core.Packed{shape, tensor} = dynamic
  shape

def retained_left(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Core.Dynamic:
  ((left, right), result) = pair
  left

def retained_right(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Core.Dynamic:
  ((left, right), result) = pair
  right

def succeeded(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Bool:
  (inputs, result) = pair
  Result.is_done(&2, &1, Core.Error, Core.Dynamic, result)

type Property is Data:
  Dimensions{}
  Checked{index: U32}
  Raw{index: U32}

type Inspection is Data:
  Dims{dimensions: List<&2, U32>}
  Value{value: Maybe<&2, F32>}

def inspect(property: Property, dynamic: Core.Dynamic) -> Inspection:
  match property:
    case Dimensions{}:
      Dims{dimensions(dynamic)}
    case Checked{index}:
      Value{Core.checked_observe_dynamic(dynamic, index)}
    case Raw{index}:
      Value{Core.observe_dynamic(dynamic, index)}

def SameSource(-original: Core.Dynamic, -retained: Core.Dynamic) -> Type:
  @+property: Property -> {inspect(property, retained) == inspect(property, original) : Inspection}

type Retention<-left: Core.Dynamic, -right: Core.Dynamic,
  -pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>> is Data:
  Retention{-left: SameSource(left, retained_left(pair)),
    -right: SameSource(right, retained_right(pair))}

def OutputDimensions(+output: List<&2, U32>, pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Type:
  (inputs, result) = pair
  match result:
    case Fail{error}:
      Unit
    case Done{dynamic}:
      {dimensions(dynamic) == output : List<&2, U32>}

type DimensionsModel<-output: List<&2, U32>,
  -pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>> is Data:
  DimensionsModel{-exact: OutputDimensions(output, pair)}

type Success<-pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>> is Data:
  Success{-successful: {succeeded(pair) == True{} : Bool}}

type Compatible<-left: Core.Dynamic, -right: Core.Dynamic, -output: List<&2, U32>> is Data:
  Compatible{alignment: Shape.Broadcast<Shape.extents(dynamic_shape(left)), Shape.extents(dynamic_shape(right)), Shape.naturals(output)>,
    rank: Shape.LE(Shape.rank_dimensions(output), 4n), capacity: Shape.LE(Shape.logical_numel(output), 1048576n)}
