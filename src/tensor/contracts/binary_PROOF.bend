import Base
import ../../shape/PROOF.bend as ShapeProof
import ../../storage/PROOF.bend as StorageProof
import ./PROOF.bend as ContractsProof
import ./binary_LAWS.bend as Laws
import ./binary_proof.bend as Proof

def Laws.binary_replayed(operation, left, right, plan, planned, left_fill, left_ops, left_fits, right_fill, right_ops, right_fits, P, h):
  +planned = planned
  Proof.binary_replayed(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, planned, P, h)(left_fits, right_fits)
