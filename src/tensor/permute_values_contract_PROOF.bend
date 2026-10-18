import Base
import ./permute_values_contract_LAWS.bend as Laws
import ./permute_values.bend as PermuteValues

def Laws.permute_contract(input, values, counted, axes, valid):
  PermuteValues.permute_contract(input, values, counted, axes, valid)

def Laws.transpose_contract(input, values, counted, a, b, inside):
  PermuteValues.transpose_contract(input, values, counted, a, b, inside)

def Laws.permute_replayed(input, fill, ops, fits, axes, plan, laid, P, h):
  PermuteValues.permute_replayed(input, fill, ops, fits, axes, plan, laid, P, h)

def Laws.transpose_replayed(input, fill, ops, fits, a, b, inside, plan, laid, P, h):
  PermuteValues.transpose_replayed(input, fill, ops, fits, a, b, inside, plan, laid, P, h)
