import Base
import ../from-values-contract.bend as FromValues
import ./LAWS.bend as Laws
import ./replayed.bend as Replayed
import ./pointwise-proof.bend as Proof
import ./broadcast-proof.bend as BroadcastProof
import ./sum-proof.bend as SumProof
import ./dot-proof.bend as DotProof
import ./matmul-proof.bend as MatmulOrdered

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
