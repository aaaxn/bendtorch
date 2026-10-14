import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ./lifecycle.bend as Lifecycle

type Error is Data:
  ShapeError{error: Shape.Error}
  StorageError{error: Storage.Error}
  ValueCountMismatch{}
  StorageLengthMismatch{}

type Tensor<-shape: Shape.Shape> is Type:
  TTensor{storage: Storage.Storage,
    -length_exact: {Lifecycle.logical_length(storage) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}}

type Dynamic is Type:
  Packed{shape: Shape.Shape, tensor: Tensor<shape>}

def pack(shape: Shape.Shape, tensor: Tensor<shape>) -> Dynamic:
  Packed{shape, tensor}

def shape(dynamic: Dynamic) -> Dynamic & Shape.Shape:
  match dynamic:
    case Packed{+shape, tensor}:
      (Packed{shape, tensor}, shape)

def count(dynamic: Dynamic) -> Dynamic & U32:
  match dynamic:
    case Packed{+shape, tensor}:
      (Packed{shape, tensor}, Shape.count(shape))

def logical_length(-shape: Shape.Shape, tensor: Tensor<shape>) -> Nat:
  TTensor{storage, exact} = tensor
  Lifecycle.logical_length(storage)

def observe(-shape: Shape.Shape, tensor: Tensor<shape>, index: U32) -> Maybe<&2, F32>:
  TTensor{storage, exact} = tensor
  Lifecycle.observe(storage, index)

def observe_dynamic(dynamic: Dynamic, index: U32) -> Maybe<&2, F32>:
  Packed{shape, tensor} = dynamic
  observe(shape, tensor, index)

def ObservationChecked(dynamic: Dynamic, -index: U32) -> Data:
  Packed{shape, tensor} = dynamic
  TTensor{storage, exact} = tensor
  match storage:
    case Storage.Empty{}:
      Empty
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      {Storage.index_matches(depth, length, index) == True{} : Bool}

type LengthValid<-shape: Shape.Shape, -tensor: Tensor<shape>> is Data:
  LengthValid{-exact: {logical_length(shape, tensor) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}}

def attach_decided(+shape: Shape.Shape, storage: Storage.Storage, +n: Nat,
  -exact: {Lifecycle.logical_length(storage) == n : Nat},
  decision: Shape.BoolCase<Nat.is_eq(n, Shape.elements(shape))>) ->
  Result<&2, &1, Error, Tensor<shape>>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{StorageLengthMismatch{}}
    case Shape.BTrue{equal}:
      Done{TTensor{storage, Equal.trans(Nat, Lifecycle.logical_length(storage), n,
        Shape.logical_numel(Shape.dimensions(shape)), exact,
        Equal.trans(Nat, n, Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)),
          Lifecycle.natural_equal(n, Shape.elements(shape), equal), Shape.elements_exact(shape)))}}

def attach_length(+shape: Shape.Shape, -original: Storage.Storage, pair: Storage.Storage & Nat,
  -same: {pair == Storage.length(original) : Storage.Storage & Nat}) ->
  Result<&2, &1, Error, Tensor<shape>>:
  (storage, +n) = pair
  attach_decided(shape, storage, n,
    %Equal.sym(Storage.Storage & Nat, (storage, n), Storage.length(original), same) :
      {Lifecycle.length_owner(_) == Lifecycle.length_value(_) : Nat}
    Lifecycle.length_exact(original),
    Shape.bool_case(Nat.is_eq(n, Shape.elements(shape))))

def attach(+shape: Shape.Shape, storage: Storage.Storage) -> Result<&2, &1, Error, Tensor<shape>>:
  attach_length(shape, storage, Storage.length(storage), {==})

def new_result(+shape: Shape.Shape, result: Result<&2, &1, Storage.Error, Storage.Storage>) ->
  Result<&2, &1, Error, Tensor<shape>>:
  match result:
    case Fail{error}:
      Fail{StorageError{error}}
    case Done{storage}:
      attach(shape, storage)

def new(+shape: Shape.Shape, fill: F32) -> Result<&2, &1, Error, Tensor<shape>>:
  new_result(shape, Storage.new(Shape.count(shape), fill))

def storage_read_error(result: Result<&2, &2, Storage.Error, F32>) -> Result<&2, &2, Error, F32>:
  match result:
    case Fail{error}:
      Fail{StorageError{error}}
    case Done{value}:
      Done{value}

def storage_write_error(result: Result<&2, &2, Storage.Error, Unit>) -> Result<&2, &2, Error, Unit>:
  match result:
    case Fail{error}:
      Fail{StorageError{error}}
    case Done{value}:
      Done{value}

def read_result(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +i: U32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>,
  -same: {pair == Storage.get(original, i) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}) ->
  Tensor<shape> & Result<&2, &2, Error, F32>:
  (storage, result) = pair
  tensor : Tensor<shape> = TTensor{storage, Equal.trans(Nat, Lifecycle.logical_length(storage), Lifecycle.logical_length(original),
    Shape.logical_numel(Shape.dimensions(shape)),
    %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, F32>, (storage, result), Storage.get(original, i), same) :
      {Lifecycle.retained_read(_) == Lifecycle.logical_length(original) : Nat}
    Lifecycle.get_length(original, i), valid)}
  (tensor, storage_read_error(result))

def read_flat(-shape: Shape.Shape, tensor: Tensor<shape>, +i: U32) -> Tensor<shape> & Result<&2, &2, Error, F32>:
  TTensor{storage, valid} = tensor
  read_result(shape, storage, valid, i, Storage.get(storage, i), {==})

def result_value(result: Result<&2, &2, Error, F32>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{value}:
      Some{value}

def read_value(-shape: Shape.Shape, pair: Tensor<shape> & Result<&2, &2, Error, F32>) -> Maybe<&2, F32>:
  (tensor, result) = pair
  result_value(result)

type Read<-shape: Shape.Shape, -original: Tensor<shape>, -index: U32> is Type:
  Read{tensor: Tensor<shape>, result: Result<&2, &2, Error, F32>,
    exact: {(tensor, result) == read_flat(shape, original, index) : Tensor<shape> & Result<&2, &2, Error, F32>}}

def read_certificate_pair(-shape: Shape.Shape, -original: Tensor<shape>, -index: U32,
  pair: Tensor<shape> & Result<&2, &2, Error, F32>,
  exact: {pair == read_flat(shape, original, index) : Tensor<shape> & Result<&2, &2, Error, F32>}) -> Read<shape, original, index>:
  (tensor, result) = pair
  Read{tensor, result, exact}

def read_certificate(-shape: Shape.Shape, original: Tensor<shape>, +index: U32) -> Read<shape, original, index>:
  read_certificate_pair(shape, original, index, read_flat(shape, original, index), {==})

def certificate_value(-shape: Shape.Shape, -original: Tensor<shape>, -index: U32, certificate: Read<shape, original, index>) -> Maybe<&2, F32>:
  Read{tensor, result, exact} = certificate
  result_value(result)

def checked_observe(-shape: Shape.Shape, tensor: Tensor<shape>, index: U32) -> Maybe<&2, F32>:
  TTensor{storage, exact} = tensor
  Lifecycle.checked_observe(storage, index)

type ObservationModel<-shape: Shape.Shape, -original: Tensor<shape>, -index: U32,
  -checked: ObservationChecked(Packed{shape, original}, index)> is Data:
  ObservationModel{-equal: {checked_observe(shape, original, index) == observe(shape, original, index) : Maybe<&2, F32>}}

def owned_observation_model(-shape: Shape.Shape, +depth: Nat, +length: Nat, -array: Array<F32>, +index: U32,
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  -checked: {Storage.index_matches(depth, length, index) == True{} : Bool},
  certificate: Lifecycle.ArrayObservationModel<depth, length, array, index, checked>) ->
  ObservationModel<shape, TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, index, checked>:
  Lifecycle.ArrayObservationModel{equal} = certificate
  ObservationModel{equal}

def observation_model(-shape: Shape.Shape, original: Tensor<shape>, +index: U32,
  +checked: ObservationChecked(Packed{shape, original}, index)) -> ObservationModel<shape, original, index, checked>:
  TTensor{storage, exact} = original
  match storage:
    case Storage.Empty{}:
      match checked:
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      owned_observation_model(shape, depth, length, array, index, balanced, valid, exact, checked,
        Lifecycle.array_observation_model(depth, length, array, index, checked))

def read_result_value(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>,
  -same: {pair == Storage.get(original, index) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}) ->
  {read_value(shape, read_result(shape, original, valid, index, pair, same)) == Lifecycle.read_value(pair) : Maybe<&2, F32>}:
  (storage, result) = pair
  match result:
    case Fail{error}:
      {==}
    case Done{value}:
      {==}

type SourceRead<-shape: Shape.Shape, -original: Tensor<shape>, -index: U32> is Data:
  SourceRead{-meaning: {read_value(shape, read_flat(shape, original, index)) == checked_observe(shape, original, index) : Maybe<&2, F32>}}

def storage_read_source(-shape: Shape.Shape, -storage: Storage.Storage,
  -valid: {Lifecycle.logical_length(storage) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32,
  certificate: Lifecycle.StorageReadModel<storage, index>) -> SourceRead<shape, TTensor{storage, valid}, index>:
  Lifecycle.StorageReadModel{meaning} = certificate
  SourceRead{Equal.trans(Maybe<&2, F32>, read_value(shape, read_flat(shape, TTensor{storage, valid}, index)),
    Lifecycle.read_value(Storage.get(storage, index)), Lifecycle.checked_observe(storage, index),
    read_result_value(shape, storage, valid, index, Storage.get(storage, index), {==}), meaning)}

def source_read_model(-shape: Shape.Shape, original: Tensor<shape>, +index: U32) -> SourceRead<shape, original, index>:
  TTensor{storage, valid} = original
  storage_read_source(shape, storage, valid, index, Lifecycle.storage_read_model(storage, index))

def retained_read_observe(-shape: Shape.Shape, pair: Tensor<shape> & Result<&2, &2, Error, F32>, index: U32) -> Maybe<&2, F32>:
  (tensor, result) = pair
  checked_observe(shape, tensor, index)

def read_result_keeps_projection(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +other: U32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>,
  -same: {pair == Storage.get(original, index) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}) ->
  {retained_read_observe(shape, read_result(shape, original, valid, index, pair, same), other) == Lifecycle.retained_observe(pair, other) : Maybe<&2, F32>}:
  (storage, result) = pair
  {==}

type SourceReadKeeps<-shape: Shape.Shape, -original: Tensor<shape>, -index: U32> is Data:
  SourceReadKeeps{-kept: @+other: U32 ->
    {retained_read_observe(shape, read_flat(shape, original, index), other) == checked_observe(shape, original, other) : Maybe<&2, F32>}}

def storage_read_keeps_source(-shape: Shape.Shape, -storage: Storage.Storage,
  -valid: {Lifecycle.logical_length(storage) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32,
  certificate: Lifecycle.StorageReadKeeps<storage, index>) -> SourceReadKeeps<shape, TTensor{storage, valid}, index>:
  Lifecycle.StorageReadKeeps{kept} = certificate
  SourceReadKeeps{other => Equal.trans(Maybe<&2, F32>, retained_read_observe(shape, read_flat(shape, TTensor{storage, valid}, index), other),
    Lifecycle.retained_observe(Storage.get(storage, index), other), Lifecycle.checked_observe(storage, other),
    read_result_keeps_projection(shape, storage, valid, index, other, Storage.get(storage, index), {==}), kept(other))}

def source_read_keeps(-shape: Shape.Shape, original: Tensor<shape>, +index: U32) -> SourceReadKeeps<shape, original, index>:
  TTensor{storage, valid} = original
  storage_read_keeps_source(shape, storage, valid, index, Lifecycle.storage_read_keeps(storage, index))

def write_result(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +i: U32, +value: F32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, Unit>,
  -same: {pair == Storage.set(original, i, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}) ->
  Tensor<shape> & Result<&2, &2, Error, Unit>:
  (storage, result) = pair
  tensor : Tensor<shape> = TTensor{storage, Equal.trans(Nat, Lifecycle.logical_length(storage), Lifecycle.logical_length(original),
    Shape.logical_numel(Shape.dimensions(shape)),
    %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, Unit>, (storage, result), Storage.set(original, i, value), same) :
      {Lifecycle.retained_write(_) == Lifecycle.logical_length(original) : Nat}
    Lifecycle.set_length(original, i, value), valid)}
  (tensor, storage_write_error(result))

def write_flat(-shape: Shape.Shape, tensor: Tensor<shape>, +i: U32, +value: F32) ->
  Tensor<shape> & Result<&2, &2, Error, Unit>:
  TTensor{storage, valid} = tensor
  write_result(shape, storage, valid, i, value, Storage.set(storage, i, value), {==})

def UpdateMeaning(-shape: Shape.Shape, -original: Tensor<shape>, +index: U32, +value: F32,
  pair: Tensor<shape> & Result<&2, &2, Error, Unit>) -> Type:
  (tensor, result) = pair
  match result:
    case Fail{error}:
      Unit
    case Done{unit}:
      ObservationChecked(Packed{shape, tensor}, index) &
        ({observe(shape, tensor, index) == Some{value} : Maybe<&2, F32>} &
          (@+other: U32 -> @checked: ObservationChecked(Packed{shape, original}, other) ->
            @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
              ObservationChecked(Packed{shape, tensor}, other) &
                {observe(shape, tensor, other) == observe(shape, original, other) : Maybe<&2, F32>}))

type UpdateModel<-shape: Shape.Shape, -original: Tensor<shape>, -index: U32, -value: F32> is Data:
  UpdateModel{-meaning: UpdateMeaning(shape, original, index, value, write_flat(shape, original, index, value))}

def CheckedUpdateMeaning(-shape: Shape.Shape, -original: Tensor<shape>, +index: U32, +value: F32,
  pair: Tensor<shape> & Result<&2, &2, Error, Unit>) -> Type:
  (tensor, result) = pair
  match result:
    case Fail{error}:
      Unit
    case Done{unit}:
      ObservationChecked(Packed{shape, tensor}, index) &
        ({checked_observe(shape, tensor, index) == Some{value} : Maybe<&2, F32>} &
          (@+other: U32 -> @checked: ObservationChecked(Packed{shape, original}, other) ->
            @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
              ObservationChecked(Packed{shape, tensor}, other) &
                {checked_observe(shape, tensor, other) == checked_observe(shape, original, other) : Maybe<&2, F32>}))

type CheckedUpdateModel<-shape: Shape.Shape, -original: Tensor<shape>, -index: U32, -value: F32> is Data:
  CheckedUpdateModel{-meaning: CheckedUpdateMeaning(shape, original, index, value, write_flat(shape, original, index, value))}

type Write<-shape: Shape.Shape, -original: Tensor<shape>, -index: U32, -value: F32> is Type:
  Write{tensor: Tensor<shape>, result: Result<&2, &2, Error, Unit>,
    exact: {(tensor, result) == write_flat(shape, original, index, value) : Tensor<shape> & Result<&2, &2, Error, Unit>}}

def write_certificate_pair(-shape: Shape.Shape, -original: Tensor<shape>, -index: U32, -value: F32,
  pair: Tensor<shape> & Result<&2, &2, Error, Unit>,
  exact: {pair == write_flat(shape, original, index, value) : Tensor<shape> & Result<&2, &2, Error, Unit>}) -> Write<shape, original, index, value>:
  (tensor, result) = pair
  Write{tensor, result, exact}

def write_certificate(-shape: Shape.Shape, original: Tensor<shape>, +index: U32, +value: F32) -> Write<shape, original, index, value>:
  write_certificate_pair(shape, original, index, value, write_flat(shape, original, index, value), {==})

def read_coordinate_result(+shape: Shape.Shape, tensor: Tensor<shape>, result: Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>) ->
  Tensor<shape> & Result<&2, &2, Error, F32>:
  match result:
    case Fail{error}:
      (tensor, Fail{ShapeError{error}})
    case Done{index}:
      read_flat(shape, tensor, Shape.index_value(shape, index))

def read_coordinate(+shape: Shape.Shape, tensor: Tensor<shape>, coordinates: List<&2, U32>) ->
  Tensor<shape> & Result<&2, &2, Error, F32>:
  read_coordinate_result(shape, tensor, Shape.flatten_checked(shape, coordinates))

def write_coordinate_result(+shape: Shape.Shape, tensor: Tensor<shape>, value: F32,
  result: Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>) -> Tensor<shape> & Result<&2, &2, Error, Unit>:
  match result:
    case Fail{error}:
      (tensor, Fail{ShapeError{error}})
    case Done{index}:
      write_flat(shape, tensor, Shape.index_value(shape, index), value)

def write_coordinate(+shape: Shape.Shape, tensor: Tensor<shape>, coordinates: List<&2, U32>, value: F32) ->
  Tensor<shape> & Result<&2, &2, Error, Unit>:
  write_coordinate_result(shape, tensor, value, Shape.flatten_checked(shape, coordinates))

def clone_result(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  pair: Storage.Storage & Storage.Storage,
  -same: {pair == Storage.clone(original) : Storage.Storage & Storage.Storage}) -> Tensor<shape> & Tensor<shape>:
  (first, second) = pair
  (TTensor{first, Equal.trans(Nat, Lifecycle.logical_length(first), Lifecycle.logical_length(original),
    Shape.logical_numel(Shape.dimensions(shape)),
    %Equal.sym(Storage.Storage & Storage.Storage, (first, second), Storage.clone(original), same) :
      {Lifecycle.first_length(_) == Lifecycle.logical_length(original) : Nat}
    Lifecycle.clone_first_length(original), valid)},
  TTensor{second, Equal.trans(Nat, Lifecycle.logical_length(second), Lifecycle.logical_length(original),
    Shape.logical_numel(Shape.dimensions(shape)),
    %Equal.sym(Storage.Storage & Storage.Storage, (first, second), Storage.clone(original), same) :
      {Lifecycle.second_length(_) == Lifecycle.logical_length(original) : Nat}
    Lifecycle.clone_second_length(original), valid)})

def clone(-shape: Shape.Shape, tensor: Tensor<shape>) -> Tensor<shape> & Tensor<shape>:
  TTensor{storage, valid} = tensor
  clone_result(shape, storage, valid, Storage.clone(storage), {==})

def length_result(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  pair: Storage.Storage & Nat, -same: {pair == Storage.length(original) : Storage.Storage & Nat}) -> Tensor<shape> & Nat:
  (storage, n) = pair
  (TTensor{storage, Equal.trans(Nat, Lifecycle.logical_length(storage), Lifecycle.logical_length(original),
    Shape.logical_numel(Shape.dimensions(shape)),
    %Equal.sym(Storage.Storage & Nat, (storage, n), Storage.length(original), same) :
      {Lifecycle.length_owner(_) == Lifecycle.logical_length(original) : Nat}
    Lifecycle.length_preserved(original), valid)}, n)

def length(-shape: Shape.Shape, tensor: Tensor<shape>) -> Tensor<shape> & Nat:
  TTensor{storage, valid} = tensor
  length_result(shape, storage, valid, Storage.length(storage), {==})

def fill_values_step(-shape: Shape.Shape, rec: Tensor<shape> -> Result<&2, &1, Error, Tensor<shape>>,
  pair: Tensor<shape> & Result<&2, &2, Error, Unit>) -> Result<&2, &1, Error, Tensor<shape>>:
  (tensor, result) = pair
  match result:
    case Fail{error}:
      Fail{error}
    case Done{unit}:
      rec(tensor)

def fill_values_decided(-shape: Shape.Shape, +cursor: Nat, +index: U32, value: F32,
  tensor: Tensor<shape>, rec: Tensor<shape> -> Result<&2, &1, Error, Tensor<shape>>,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>) -> Result<&2, &1, Error, Tensor<shape>>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{StorageError{Storage.InvalidIndex{}}}
    case Shape.BTrue{equal}:
      fill_values_step(shape, rec, write_flat(shape, tensor, index, value))

def fill_values(values: List<&2, F32>, -shape: Shape.Shape, tensor: Tensor<shape>, +cursor: Nat) ->
  Result<&2, &1, Error, Tensor<shape>>:
  match values:
    case Nil{}:
      Done{tensor}
    case value <> rest:
      +index = U32.from_nat(cursor)
      fill_values_decided(shape, cursor, index, value, tensor,
        tensor => fill_values(rest, shape, tensor, 1n+cursor), Shape.bool_case(Nat.is_eq(U32.to_nat(index), cursor)))

def fill_values_word(values: List<&2, F32>, -shape: Shape.Shape, tensor: Tensor<shape>, index: U32) ->
  Result<&2, &1, Error, Tensor<shape>>:
  fill_values(values, shape, tensor, U32.to_nat(index))

def from_values_new(-shape: Shape.Shape, values: List<&2, F32>, result: Result<&2, &1, Error, Tensor<shape>>) ->
  Result<&2, &1, Error, Tensor<shape>>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{tensor}:
      fill_values(values, shape, tensor, 0n)

def from_values_counted(+shape: Shape.Shape, +values: List<&2, F32>,
  decision: Shape.BoolCase<Nat.is_eq(List.length(&2, F32, values), Shape.elements(shape))>) ->
  Result<&2, &1, Error, Tensor<shape>>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{ValueCountMismatch{}}
    case Shape.BTrue{equal}:
      from_values_new(shape, values, new(shape, 0.0))

def from_values(+shape: Shape.Shape, +values: List<&2, F32>) -> Result<&2, &1, Error, Tensor<shape>>:
  from_values_counted(shape, values,
    Shape.bool_case(Nat.is_eq(List.length(&2, F32, values), Shape.elements(shape))))

def construct_values(+shape: Shape.Shape, result: Result<&2, &1, Error, Tensor<shape>>) -> Result<&2, &1, Error, Dynamic>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{tensor}:
      Done{pack(shape, tensor)}

def construct_shape(-dimensions: List<&2, U32>, values: List<&2, F32>, result: Shape.ConstructResult<dimensions>) ->
  Result<&2, &1, Error, Dynamic>:
  match result:
    case Shape.SError{error}:
      Fail{ShapeError{error}}
    case Shape.SBuilt{+shape, preserved}:
      construct_values(shape, from_values(shape, values))

def construct(+dimensions: List<&2, U32>, values: List<&2, F32>) -> Result<&2, &1, Error, Dynamic>:
  construct_shape(dimensions, values, Shape.construct(dimensions))

def ConstructionDimensions(+dimensions: List<&2, U32>, result: Result<&2, &1, Error, Dynamic>) -> Type:
  match result:
    case Fail{error}:
      Unit
    case Done{Packed{+shape, tensor}}:
      {Shape.dimensions(shape) == dimensions : List<&2, U32>}

def ValuesCount(+shape: Shape.Shape, +values: List<&2, F32>, result: Result<&2, &1, Error, Tensor<shape>>) -> Type:
  match result:
    case Fail{error}:
      Unit
    case Done{tensor}:
      {Shape.logical_numel(Shape.dimensions(shape)) == List.length(&2, F32, values) : Nat}

def read_dynamic_result(+shape: Shape.Shape, pair: Tensor<shape> & Result<&2, &2, Error, F32>) ->
  Dynamic & Result<&2, &2, Error, F32>:
  (tensor, result) = pair
  (Packed{shape, tensor}, result)

def read_dynamic(dynamic: Dynamic, index: U32) -> Dynamic & Result<&2, &2, Error, F32>:
  Packed{+shape, tensor} = dynamic
  read_dynamic_result(shape, read_flat(shape, tensor, index))

type DynamicRead<-original: Dynamic, -index: U32> is Type:
  DynamicRead{dynamic: Dynamic, result: Result<&2, &2, Error, F32>,
    exact: {(dynamic, result) == read_dynamic(original, index) : Dynamic & Result<&2, &2, Error, F32>}}

def dynamic_read_certificate_pair(-original: Dynamic, -index: U32,
  pair: Dynamic & Result<&2, &2, Error, F32>,
  exact: {pair == read_dynamic(original, index) : Dynamic & Result<&2, &2, Error, F32>}) -> DynamicRead<original, index>:
  (dynamic, result) = pair
  DynamicRead{dynamic, result, exact}

def dynamic_read_certificate(original: Dynamic, +index: U32) -> DynamicRead<original, index>:
  dynamic_read_certificate_pair(original, index, read_dynamic(original, index), {==})

def dynamic_read_value(pair: Dynamic & Result<&2, &2, Error, F32>) -> Maybe<&2, F32>:
  (dynamic, result) = pair
  result_value(result)

def checked_observe_dynamic(dynamic: Dynamic, index: U32) -> Maybe<&2, F32>:
  Packed{shape, tensor} = dynamic
  checked_observe(shape, tensor, index)

def dynamic_result_value(+shape: Shape.Shape, pair: Tensor<shape> & Result<&2, &2, Error, F32>) ->
  {dynamic_read_value(read_dynamic_result(shape, pair)) == read_value(shape, pair) : Maybe<&2, F32>}:
  (tensor, result) = pair
  {==}

type DynamicSourceRead<-original: Dynamic, -index: U32> is Data:
  DynamicSourceRead{-meaning: {dynamic_read_value(read_dynamic(original, index)) == checked_observe_dynamic(original, index) : Maybe<&2, F32>}}

def dynamic_read_source(+shape: Shape.Shape, -tensor: Tensor<shape>, +index: U32, certificate: SourceRead<shape, tensor, index>) ->
  DynamicSourceRead<Packed{shape, tensor}, index>:
  SourceRead{meaning} = certificate
  DynamicSourceRead{Equal.trans(Maybe<&2, F32>, dynamic_read_value(read_dynamic(Packed{shape, tensor}, index)),
    read_value(shape, read_flat(shape, tensor, index)), checked_observe(shape, tensor, index),
    dynamic_result_value(shape, read_flat(shape, tensor, index)), meaning)}

def dynamic_source_read_model(original: Dynamic, +index: U32) -> DynamicSourceRead<original, index>:
  Packed{+shape, tensor} = original
  dynamic_read_source(shape, tensor, index, source_read_model(shape, tensor, index))

def retained_dynamic_observe(pair: Dynamic & Result<&2, &2, Error, F32>, index: U32) -> Maybe<&2, F32>:
  (dynamic, result) = pair
  checked_observe_dynamic(dynamic, index)

def dynamic_keeps_projection(+shape: Shape.Shape, pair: Tensor<shape> & Result<&2, &2, Error, F32>, +other: U32) ->
  {retained_dynamic_observe(read_dynamic_result(shape, pair), other) == retained_read_observe(shape, pair, other) : Maybe<&2, F32>}:
  (tensor, result) = pair
  {==}

type DynamicSourceReadKeeps<-original: Dynamic, -index: U32> is Data:
  DynamicSourceReadKeeps{-kept: @+other: U32 ->
    {retained_dynamic_observe(read_dynamic(original, index), other) == checked_observe_dynamic(original, other) : Maybe<&2, F32>}}

def dynamic_read_keeps_source(+shape: Shape.Shape, -tensor: Tensor<shape>, +index: U32, certificate: SourceReadKeeps<shape, tensor, index>) ->
  DynamicSourceReadKeeps<Packed{shape, tensor}, index>:
  SourceReadKeeps{kept} = certificate
  DynamicSourceReadKeeps{other => Equal.trans(Maybe<&2, F32>, retained_dynamic_observe(read_dynamic(Packed{shape, tensor}, index), other),
    retained_read_observe(shape, read_flat(shape, tensor, index), other), checked_observe(shape, tensor, other),
    dynamic_keeps_projection(shape, read_flat(shape, tensor, index), other), kept(other))}

def dynamic_source_read_keeps(original: Dynamic, +index: U32) -> DynamicSourceReadKeeps<original, index>:
  Packed{+shape, tensor} = original
  dynamic_read_keeps_source(shape, tensor, index, source_read_keeps(shape, tensor, index))

type ReadInfo<-original: Dynamic, -index: U32> is Type:
  ReadInfo{owner: Dynamic, same: {owner == original : Dynamic},
    source: DynamicSourceRead<original, index>, keeps: DynamicSourceReadKeeps<original, index>}

def owned_read_source(-shape: Shape.Shape, +depth: Nat, +length: Nat, -array: Array<F32>, +index: U32,
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  SourceRead<shape, TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, index>:
  storage_read_source(shape, Storage.Owned{depth, length, array, balanced, valid}, exact, index,
    Lifecycle.read_checked_storage(depth, length, array, index, balanced, valid,
      Lifecycle.read_checked_model(depth, length, array, index, Nat.is_lt(U32.to_nat(index), length), {==}, balanced, valid)))

def owned_read_keeps(-shape: Shape.Shape, +depth: Nat, +length: Nat, -array: Array<F32>, +index: U32,
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  SourceReadKeeps<shape, TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, index>:
  storage_read_keeps_source(shape, Storage.Owned{depth, length, array, balanced, valid}, exact, index,
    Lifecycle.read_checked_storage_keeps(depth, length, array, index, balanced, valid,
      Lifecycle.read_checked_keeps(depth, length, array, index, Nat.is_lt(U32.to_nat(index), length), {==}, balanced, valid)))

def read_info(original: Dynamic, +index: U32) -> ReadInfo<original, index>:
  Packed{+shape, tensor} = original
  TTensor{storage, exact} = tensor
  match storage:
    case Storage.Empty{}:
      ReadInfo{Packed{shape, TTensor{Storage.Empty{}, exact}}, {==},
        dynamic_read_source(shape, TTensor{Storage.Empty{}, exact}, index,
          storage_read_source(shape, Storage.Empty{}, exact, index, Lifecycle.StorageReadModel{{==}})),
        dynamic_read_keeps_source(shape, TTensor{Storage.Empty{}, exact}, index,
          storage_read_keeps_source(shape, Storage.Empty{}, exact, index, Lifecycle.StorageReadKeeps{other => {==}}))}
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      ReadInfo{Packed{shape, TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}}, {==},
        dynamic_read_source(shape, TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, index,
          owned_read_source(shape, depth, length, array, index, balanced, valid, exact)),
        dynamic_read_keeps_source(shape, TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact}, index,
          owned_read_keeps(shape, depth, length, array, index, balanced, valid, exact))}

def write_dynamic_result(+shape: Shape.Shape, pair: Tensor<shape> & Result<&2, &2, Error, Unit>) ->
  Dynamic & Result<&2, &2, Error, Unit>:
  (tensor, result) = pair
  (Packed{shape, tensor}, result)

def write_dynamic(dynamic: Dynamic, index: U32, value: F32) -> Dynamic & Result<&2, &2, Error, Unit>:
  Packed{+shape, tensor} = dynamic
  write_dynamic_result(shape, write_flat(shape, tensor, index, value))

def clone_dynamic_result(+shape: Shape.Shape, pair: Tensor<shape> & Tensor<shape>) -> Dynamic & Dynamic:
  (first, second) = pair
  (Packed{shape, first}, Packed{shape, second})

def clone_dynamic(dynamic: Dynamic) -> Dynamic & Dynamic:
  Packed{+shape, tensor} = dynamic
  clone_dynamic_result(shape, clone(shape, tensor))

def write_observe(-shape: Shape.Shape, pair: Tensor<shape> & Result<&2, &2, Error, Unit>, index: U32) -> Maybe<&2, F32>:
  (tensor, result) = pair
  observe(shape, tensor, index)

def error_code(error: Error) -> U32:
  match error:
    case ShapeError{error}:
      Shape.error_code(error)
    case StorageError{Storage.TooLarge{}}:
      7
    case StorageError{Storage.OutOfBounds{}}:
      8
    case StorageError{Storage.InvalidIndex{}}:
      9
    case ValueCountMismatch{}:
      10
    case StorageLengthMismatch{}:
      11
