import Base
import ../../src/tensor/core.bend as Core
import ./consumers.bend as Cases
import ./generic-public.bend as Public
import ./observations.bend as Observation

def expected_value() -> Maybe<&2, F32>:
  Some{7.0}

def wrong_value() -> Maybe<&2, F32>:
  Some{8.0}

def wrong_fill_value() -> F32:
  0.0

def scalar_value(evidence: Public.Claims<Cases.scalar_shape(), 7.0, 0>) ->
  {Observation.point(Cases.scalar_shape(), 0, Core.new(Cases.scalar_shape(), 7.0)) == expected_value() : Maybe<&2, F32>}:
  Public.Claims{successful, dimensions, count, word_count, value, retained_dimensions, retained_count, retained_word_count, retained_value} = evidence
  value

def three_value(-index: U32, evidence: Public.Claims<Cases.three_shape(), 7.0, index>) ->
  {Observation.point(Cases.three_shape(), index, Core.new(Cases.three_shape(), 7.0)) == expected_value() : Maybe<&2, F32>}:
  Public.Claims{successful, dimensions, count, word_count, value, retained_dimensions, retained_count, retained_word_count, retained_value} = evidence
  value

def scalar_witness() ->
  {Observation.point(Cases.scalar_shape(), 0, Core.new(Cases.scalar_shape(), 7.0)) == expected_value() : Maybe<&2, F32>}:
  scalar_value(Public.consume(Cases.scalar_shape(), 7.0, 0, {==}))

def three_zero_witness() ->
  {Observation.point(Cases.three_shape(), 0, Core.new(Cases.three_shape(), 7.0)) == expected_value() : Maybe<&2, F32>}:
  three_value(0, Public.consume(Cases.three_shape(), 7.0, 0, {==}))

def three_one_witness() ->
  {Observation.point(Cases.three_shape(), 1, Core.new(Cases.three_shape(), 7.0)) == expected_value() : Maybe<&2, F32>}:
  three_value(1, Public.consume(Cases.three_shape(), 7.0, 1, {==}))

def three_two_witness() ->
  {Observation.point(Cases.three_shape(), 2, Core.new(Cases.three_shape(), 7.0)) == expected_value() : Maybe<&2, F32>}:
  three_value(2, Public.consume(Cases.three_shape(), 7.0, 2, {==}))
