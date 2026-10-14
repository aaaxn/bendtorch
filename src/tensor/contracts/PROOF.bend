import Base
import ../from-values-contract.bend as FromValues
import ./LAWS.bend as Laws
import ./replayed.bend as Replayed
import ./pointwise-proof.bend as Proof
import ./broadcast-proof.bend as BroadcastProof

def Laws.new_replayed(shape, fill, P, h):
  Replayed.new_lift(shape, fill, P, h)

def Laws.from_values_replayed(shape, values, counted, P, h):
  +counted = counted
  FromValues.from_values_lift(shape, values, counted, P, h)

def Laws.replayed_read(shape, fill, ops, fits, index, inside):
  Proof.replayed_point(shape, fill, ops, fits, index, inside)

def Laws.broadcast_pointwise(operation, plan, left_fill, left_ops, left_fits, right_fill, right_ops, right_fits):
  BroadcastProof.broadcast_pointwise(operation, plan, left_fill, left_ops, left_fits, right_fill, right_ops, right_fits)
