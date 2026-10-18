import Base
import ../../../src/tensor/read_write_records.bend as Records
import ../../../src/tensor/read_contract.bend as Reads
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/observation.bend as Observation
import ../../../src/shape/core.bend as Shape

def three_shape() -> Shape.Shape:
  Shape.SShape{[3], 3n, {==}, Unit{}, Unit{}}

def read_value(-index: U32, evidence: Records.ReadInside<three_shape(), 7.0, index>) ->
  {Observation.point(three_shape(), index, Core.new(three_shape(), 7.0)) == Some{7.0} : Maybe<&2, F32>}:
  Records.ReadInside{read, value} = evidence
  value

def read_one() -> {Observation.point(three_shape(), 1, Core.new(three_shape(), 7.0)) == Some{7.0} : Maybe<&2, F32>}:
  read_value(1, Reads.read_inside(three_shape(), 7.0, 1, {==}))

def read_outside(evidence: Records.ReadContract<three_shape(), 7.0, 3, 8>) ->
  {Observation.read_code(three_shape(), 3, Core.new(three_shape(), 7.0)) == Some{8} : Maybe<&2, U32>}:
  Records.ReadContract{code, dimensions, count, word_count, kept} = evidence
  code

def read_three() -> {Observation.read_code(three_shape(), 3, Core.new(three_shape(), 7.0)) == Some{8} : Maybe<&2, U32>}:
  read_outside(Reads.read_outside(three_shape(), 7.0, 3, {==}))
