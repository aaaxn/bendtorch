import Base
import ./read_write_contract_LAWS.bend as Laws
import ./read_contract.bend as Reads
import ./write_contract.bend as Writes

def Laws.read_inside(shape, fill, index, inside):
  Reads.read_inside(shape, fill, index, inside)

def Laws.read_outside(shape, fill, index, outside):
  Reads.read_outside(shape, fill, index, outside)

def Laws.write_inside(shape, fill, index, value, inside):
  Writes.write_inside(shape, fill, index, value, inside)

def Laws.write_outside(shape, fill, index, value, outside):
  Writes.write_outside(shape, fill, index, value, outside)

def Laws.coordinate_inside(shape, fill, coordinates, index, value, located, inside):
  Writes.coordinate_inside(shape, fill, coordinates, index, value, located, inside)

def Laws.dynamic_read_inside(shape, fill, index, inside):
  Reads.dynamic_read_inside(shape, fill, index, inside)
