import Base
import ./transform-contract-LAWS.bend as Laws
import ./reshape-contract.bend as Reshapes
import ./permute-contract.bend as Permutes

def Laws.reshape_contract(input, fill, target, same):
  Reshapes.reshape_contract(input, fill, target, same)

def Laws.reshape_write(input, fill, target, index, value, inside, same):
  Reshapes.reshape_write(input, fill, target, index, value, inside, same)

def Laws.reshape_count_mismatch(input, fill, target, different):
  Reshapes.reshape_count_mismatch(input, fill, target, different)

def Laws.permute_rank_mismatch(input, fill, axes, wrong):
  Permutes.permute_rank_mismatch(input, fill, axes, wrong)

def Laws.permute_axis_out_of_bounds(input, fill, axes, ranked, outside):
  Permutes.permute_axis_out_of_bounds(input, fill, axes, ranked, outside)

def Laws.permute_duplicate_axis(input, fill, axes, ranked, bounded, duplicate):
  Permutes.permute_duplicate_axis(input, fill, axes, ranked, bounded, duplicate)

def Laws.transpose_out_of_bounds(input, fill, a, b, outside):
  Permutes.transpose_out_of_bounds(input, fill, a, b, outside)
