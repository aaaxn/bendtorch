import Base
import ./model.bend as Model
import ./array.bend as ArrayCore
import ./LAWS.bend as Laws
import ./refinement.bend as Refinement
import ./frame.bend as Frame
import ./main.bend as Storage
import ./api.bend as Api

def Laws.write_read(T, d, array, path, value, valid):
  Frame.write_read(T, d, array, path, value, valid)

def Laws.update_frame(T, d, array, i, j, value, valid):
  Frame.update_frame(T, d, array, i, j, value, valid)

def Laws.read_refines(T, d, array, path, valid):
  Frame.read_refines(T, d, array, path, valid)

def Laws.read_retains(T, d, array, path, valid):
  Frame.read_retains(T, d, array, path, valid)

def Laws.primitive_update_frame(T, d, array, i, other, value, valid):
  Frame.primitive_update_frame(T, d, array, i, other, value, valid)

def Laws.path_self_same(d, path):
  Frame.path_self_same(d, path)

def Laws.primitive_write_read(T, d, array, i, value, valid):
  Frame.primitive_write_read(T, d, array, i, value, valid)

def Laws.new_too_large(n, value, over):
  Api.new_refused(U32.to_nat(n), value, U32.is_le(n, 1048576), Equal.sym(Bool, U32.is_le(n, 1048576), False{}, over),
    Api.positive_of(U32.to_nat(n), U32.to_nat(1048576), Api.over_nat(n, over)))

def Laws.get_outside(storage, i, outside):
  match storage:
    case Storage.Empty{}:
      {==}
    case Storage.Owned{+d, +n, array, balanced, valid}:
      Api.get_checked_miss(d, n, array, i, Nat.is_lt(U32.to_nat(i), n), {==}, balanced, valid,
        Equal.sym(Bool, Nat.is_lt(U32.to_nat(i), n), False{}, outside))

def Laws.set_outside(storage, i, value, outside):
  match storage:
    case Storage.Empty{}:
      {==}
    case Storage.Owned{+d, +n, array, balanced, valid}:
      %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, Unit>, Storage.set_checked(d, n, array, i, value, Nat.is_lt(U32.to_nat(i), n), {==}, balanced, valid),
        (Storage.Owned{d, n, array, balanced, valid}, Fail{Storage.OutOfBounds{}}),
        Api.set_checked_miss(d, n, array, i, value, Nat.is_lt(U32.to_nat(i), n), {==}, balanced, valid,
          Equal.sym(Bool, Nat.is_lt(U32.to_nat(i), n), False{}, outside))) :
        {Laws.outcome(Result<&2, &2, Storage.Error, Unit>, _) == Fail{Storage.OutOfBounds{}} : Result<&2, &2, Storage.Error, Unit>}
      {==}

def Laws.replay_size(n, value, ops, fits):
  Api.new_size(U32.to_nat(n), value, U32.is_le(n, 1048576), ops, Equal.sym(Bool, U32.is_le(n, 1048576), True{}, fits), Api.cap_from(n, fits))

def Laws.replay_get(n, value, ops, j, fits, inside):
  Api.new_read(U32.to_nat(n), value, U32.is_le(n, 1048576), ops, j, Equal.sym(Bool, U32.is_le(n, 1048576), True{}, fits),
    Api.cap_from(n, fits), inside)

def Laws.new_fits(n, value, fits):
  Api.new_fits(U32.to_nat(n), value, U32.is_le(n, 1048576), Equal.sym(Bool, U32.is_le(n, 1048576), True{}, fits), Api.cap_from(n, fits))

def Laws.new_empty(value):
  {==}

def Laws.replay_set(n, value, ops, j, w, fits, inside):
  Api.new_status(U32.to_nat(n), value, U32.is_le(n, 1048576), ops, j, w, Equal.sym(Bool, U32.is_le(n, 1048576), True{}, fits),
    Api.cap_from(n, fits), inside)
