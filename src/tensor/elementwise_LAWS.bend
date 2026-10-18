import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./elementwise.bend as Elementwise
import ./elementwise_totality.bend as Totality

law binary_retains_inputs:
  for +operation: Elementwise.Operation
  for left: Core.Dynamic
  for right: Core.Dynamic
  Totality.Retention<left, right, Elementwise.binary(operation, left, right)>

law binary_successful_output_dimensions:
  for +operation: Elementwise.Operation
  for left: Core.Dynamic
  for right: Core.Dynamic
  for +output: List<&2, U32>
  for +alignment: Shape.Broadcast<Shape.extents(Totality.dynamic_shape(left)), Shape.extents(Totality.dynamic_shape(right)), Shape.naturals(output)>
  Totality.DimensionsModel<output, Elementwise.binary(operation, left, right)>

law binary_valid_success:
  for +operation: Elementwise.Operation
  for left: Core.Dynamic
  for right: Core.Dynamic
  for +output: List<&2, U32>
  for +compatible: Totality.Compatible<left, right, output>
  Totality.Success<Elementwise.binary(operation, left, right)>
