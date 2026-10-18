import Base
import ../from_values_contract.bend as FromValues
import ./LAWS.bend as Laws
import ./replayed.bend as Replayed
import ./pointwise_proof.bend as Proof
import ./broadcast_proof.bend as BroadcastProof
import ./sum_proof.bend as SumProof
import ./dot_proof.bend as DotProof
import ./matmul_proof.bend as MatmulOrdered
import ./product_proof.bend as ProductOrdered
import ../matmul.bend as Matrix

def Laws.new_replayed(shape, fill, P, h):
  Replayed.new_lift(shape, fill, P, h)

def Laws.from_values_replayed(shape, values, counted, P, h):
  +counted = counted
  FromValues.from_values_lift(shape, values, counted, P, h)

def Laws.replayed_read(shape, fill, ops, fits, index, inside):
  Proof.replayed_point(shape, fill, ops, fits, index, inside)

def Laws.broadcast_pointwise(operation, plan, left_fill, left_ops, left_fits, right_fill, right_ops, right_fits):
  BroadcastProof.broadcast_pointwise(operation, plan, left_fill, left_ops, left_fits, right_fill, right_ops, right_fits)

def Laws.sum_ordered(shape, fill, ops, fits):
  SumProof.sum_ordered(shape, fill, ops, fits)

def Laws.dot_ordered(rows, inner, columns, row, row_below, column, column_below, left, left_extents, left_fill, left_ops, left_fits,
  right, right_extents, right_fill, right_ops, right_fits):
  DotProof.dot_ordered(rows, inner, columns, row, row_below, column, column_below, left, left_extents, left_fill, left_ops, left_fits,
    right, right_extents, right_fill, right_ops, right_fits)

def Laws.matmul_ordered(left, right, plan, planned, left_fill, left_ops, left_fits, right_fill, right_ops, right_fits):
  MatmulOrdered.matmul_ordered(left, right, plan, planned, left_fill, left_ops, left_fits, right_fill, right_ops, right_fits)

def Laws.matmul_replayed(left, right, plan, planned, left_fill, left_ops, left_fits, right_fill, right_ops, right_fits, P, h):
  MatmulOrdered.matmul_replayed(left, right, plan, planned, left_fill, left_ops, left_fits, right_fill, right_ops, right_fits, P, h)

def Laws.product_ordered(lo, ro, left, right, plan, planned, left_fill, left_ops, left_fits, right_fill, right_ops, right_fits):
  ProductOrdered.product_ordered(lo, ro, left, right, plan, planned, left_fill, left_ops, left_fits, right_fill, right_ops, right_fits)

def Laws.product_replayed(lo, ro, left, right, plan, planned, left_fill, left_ops, left_fits, right_fill, right_ops, right_fits, P, h):
  ProductOrdered.product_replayed(lo, ro, left, right, plan, planned, left_fill, left_ops, left_fits, right_fill, right_ops, right_fits, P, h)

def Laws.matmul_nt_ordered(left, right, plan, planned, left_fill, left_ops, left_fits, right_fill, right_ops, right_fits):
  ProductOrdered.product_ordered(Matrix.Kept{}, Matrix.Kept{}, left, right, plan, planned, left_fill, left_ops, left_fits,
    right_fill, right_ops, right_fits)

def Laws.matmul_tn_ordered(left, right, plan, planned, left_fill, left_ops, left_fits, right_fill, right_ops, right_fits):
  ProductOrdered.product_ordered(Matrix.Swapped{}, Matrix.Swapped{}, left, right, plan, planned, left_fill, left_ops, left_fits,
    right_fill, right_ops, right_fits)
