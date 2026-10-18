import Base
import ./clone_metadata_contract_LAWS.bend as Laws
import ./clone_contract.bend as Clones
import ./metadata_contract.bend as Metadata

def Laws.clone_contract(shape, fill):
  Clones.clone_contract(shape, fill)

def Laws.clone_write(shape, fill, index, value, inside):
  Clones.clone_write(shape, fill, index, value, inside)

def Laws.metadata_contract(shape, fill):
  Metadata.metadata_contract(shape, fill)
