import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./transform_totality.bend as TransformTotality

type Status<-error: Data, -value: Type, -result: Result<&2, &1, error, value>> is Data:
  Status{-successful: {Result.is_done(&2, &1, error, value, result) == True{} : Bool}}

type DoneWitness<-error: Data, -value: Type, -result: Result<&2, &1, error, value>> is Data:
  DoneWitness{-returned: value, -exact: {Done{returned} == result : Result<&2, &1, error, value>}}

type Success<-error: Data, -value: Type, -result: Result<&2, &1, error, value>> is Data:
  Success{-successful: {Result.is_done(&2, &1, error, value, result) == True{} : Bool},
    -witness: DoneWitness<error, value, result>}

type MatrixInputs<-left: Shape.Shape, -right: Shape.Shape> is Data:
  MatrixInputs{-rows: U32, -inner: U32, -right_inner: U32, -columns: U32,
    -left_dimensions: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
    -right_dimensions: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
    -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
    -output_supported: Shape.LE(Shape.logical_numel([rows, columns]), 1048576n)}

type MatrixBounds<-rows: U32, -inner: U32, -right_inner: U32, -columns: U32> is Data:
  MatrixBounds{-compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
    -supported: Shape.LE(Shape.logical_numel([rows, columns]), 1048576n)}

type Impossible is Data:
  Impossible{-empty: Empty}

type RankTwo<-shape: Shape.Shape> is Data:
  RankTwo{-rank: {Shape.rank_dimensions(Shape.dimensions(shape)) == 2n : Nat}}

type ReadStatus<-shape: Shape.Shape, -pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>> is Data:
  ReadStatus{-successful: {TransformTotality.read_succeeded(shape, pair) == True{} : Bool}}
