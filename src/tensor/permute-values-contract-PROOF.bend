import Base
import ./permute-values-contract-LAWS.bend as Laws
import ./permute-values.bend as PermuteValues

def Laws.permute_contract(input, values, counted, axes, valid):
  PermuteValues.permute_contract(input, values, counted, axes, valid)

def Laws.transpose_contract(input, values, counted, a, b, inside):
  PermuteValues.transpose_contract(input, values, counted, a, b, inside)
