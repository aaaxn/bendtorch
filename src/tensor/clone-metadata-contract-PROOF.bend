import Base
import ./clone-metadata-contract-LAWS.bend as Laws
import ./clone-contract.bend as Clones
import ./metadata-contract.bend as Metadata

def Laws.clone_contract(shape, fill):
  Clones.clone_contract(shape, fill)

def Laws.clone_write(shape, fill, index, value, inside):
  Clones.clone_write(shape, fill, index, value, inside)

def Laws.metadata_contract(shape, fill):
  Metadata.metadata_contract(shape, fill)
