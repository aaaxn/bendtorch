import Base
import ../../shape/core.bend as Shape
import ../../shape/LAWS.bend as ShapeLaws
import ../../shape/plan_proofs.bend as PlanProof
import ../../shape/index_proofs.bend as Index
import ../../storage/model.bend as Model
import ../../storage/main.bend as Storage
import ../../storage/LAWS.bend as StorageLaws
import ../../storage/array.bend as ArrayCore
import ../../storage/refinement.bend as Refinement
import ../../storage/certificates.bend as Certificates
import ../../storage/api.bend as Api
import ../../storage/arith.bend as Arith
import ../core.bend as Core
import ../lifecycle.bend as Lifecycle
import ../elementwise.bend as Elementwise
import ../word_conversion.bend as Conversion
import ../read_contract.bend as Reads
import ../from_values_contract.bend as FromValues
import ./replayed.bend as Replayed
import ./pointwise.bend as Pointwise
import ./pointwise_proof.bend as PointwiseProof
import ./broadcast.bend as Broadcast
import ./broadcast_proof.bend as BroadcastProof
import ./LAWS.bend as Contracts
import ./binary_LAWS.bend as Laws
import ./binary_build_proof.bend as Build
import ./binary_storage_proof.bend as Owned
import ./build_facts.bend as Facts

def cell(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +index: Nat) -> F32:
  Pointwise.cell(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, index)

def out_leaf(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +position: Nat, inside: Bool) -> F32:
  match inside:
    case False{}:
      0.0
    case True{}:
      cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, position)

def out_spec(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat, +depth: Nat, +base: Nat) -> Array<F32>:
  match depth:
    case 0n:
      ALeaf{out_leaf(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, base, Nat.is_lt(base, count))}
    case 1n+inner:
      ANode{out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count, inner, base), out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count, inner, Nat.add(Model.capacity(inner), base))}

def out_balanced(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat, +depth: Nat, +base: Nat) -> Model.Balanced(F32, depth, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count, depth, base)):
  match depth:
    case 0n:
      Unit{}
    case 1n+inner:
      (out_balanced(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count, inner, base), out_balanced(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count, inner, Nat.add(Model.capacity(inner), base)))

def new_spec(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +depth: Nat, +base: Nat) -> {Array.new(F32, depth, 0.0) == out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 0n, depth, base) : Array<F32>}:
  match depth:
    case 0n:
      %Equal.sym(Bool, Nat.is_lt(base, 0n), False{}, Facts.lt_zero_false(base)) :
        {ALeaf{0.0} == ALeaf{out_leaf(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, base, _)} : Array<F32>}
      {==}
    case 1n+inner:
      %Equal.sym(Array<F32>, Array.new(F32, inner, 0.0), out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 0n, inner, base), new_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, inner, base)) :
        {ANode{_, Array.new(F32, inner, 0.0)} == out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 0n, 1n+inner, base) : Array<F32>}
      %Equal.sym(Array<F32>, Array.new(F32, inner, 0.0), out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 0n, inner, Nat.add(Model.capacity(inner), base)),
        new_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, inner, Nat.add(Model.capacity(inner), base))) :
        {ANode{out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 0n, inner, base), _} == out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 0n, 1n+inner, base) : Array<F32>}
      {==}

# A subtree whose positions all lie above k is the same for k and 1 + k.
def above(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +k: Nat, +depth: Nat, +b: Nat, +h: {Nat.is_lt(k, b) == True{} : Bool}) ->
  {out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, depth, b) == out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, depth, b) : Array<F32>}:
  match depth:
    case 0n:
      %Equal.sym(Bool, Nat.is_lt(b, k), False{}, Facts.lt_asym(k, b, h)) :
        {ALeaf{out_leaf(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, b, _)} == ALeaf{out_leaf(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, b, Nat.is_lt(b, 1n+k))} : Array<F32>}
      %Equal.sym(Bool, Nat.is_lt(b, 1n+k), False{}, Facts.above_false(k, b, h)) :
        {ALeaf{0.0} == ALeaf{out_leaf(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, b, _)} : Array<F32>}
      {==}
    case 1n+inner:
      +high = Nat.add(Model.capacity(inner), b)
      %Equal.sym(Array<F32>, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, b), out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, b), above(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, b, h)) :
        {ANode{_, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high)} == out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, 1n+inner, b) : Array<F32>}
      %Equal.sym(Array<F32>, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high), out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, high),
        above(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high, Arith.lt_le_trans(k, b, high, h, Api.le_add_left(Model.capacity(inner), b)))) :
        {ANode{out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, b), _} == out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, 1n+inner, b) : Array<F32>}
      {==}

# A subtree whose positions all lie below k is the same for k and 1 + k.
def below(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +k: Nat, +depth: Nat, +b: Nat, +h: {Nat.is_le(Nat.add(Model.capacity(depth), b), k) == True{} : Bool}) ->
  {out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, depth, b) == out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, depth, b) : Array<F32>}:
  match depth:
    case 0n:
      +lower = Facts.below_true(b, k, h)
      %Equal.sym(Bool, Nat.is_lt(b, k), True{}, lower) :
        {ALeaf{out_leaf(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, b, _)} == ALeaf{out_leaf(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, b, Nat.is_lt(b, 1n+k))} : Array<F32>}
      %Equal.sym(Bool, Nat.is_lt(b, 1n+k), True{}, Facts.lt_succ_right(b, k, lower)) :
        {ALeaf{cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, b)} == ALeaf{out_leaf(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, b, _)} : Array<F32>}
      {==}
    case 1n+inner:
      +c = Model.capacity(inner)
      +high = Nat.add(c, b)
      +nested = Equal.trans(Bool, Nat.is_le(Nat.add(c, high), k), Nat.is_le(Nat.add(Nat.add(c, c), b), k), True{},
        Equal.cong(Nat, Bool, x => Nat.is_le(x, k), Nat.add(c, high), Nat.add(Nat.add(c, c), b), Index.add_assoc(c, c, b)), h)
      +low_h = Facts.le_trans(high, Nat.add(c, high), k, Facts.le_add_cong(c, b, high, Api.le_add_left(c, b)), nested)
      %Equal.sym(Array<F32>, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, b), out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, b), below(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, b, low_h)) :
        {ANode{_, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high)} == out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, 1n+inner, b) : Array<F32>}
      %Equal.sym(Array<F32>, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high), out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, high), below(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high, nested)) :
        {ANode{out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, b), _} == out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, 1n+inner, b) : Array<F32>}
      {==}

# Writing cell k at the path whose position is k turns the spec for k into
# the spec for 1 + k.
def write_spec(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +k: Nat, +depth: Nat, +base: Nat, +path: Model.Path(depth),
  +h: {Nat.add(Model.offset(depth, path), base) == k : Nat}) ->
  {ArrayCore.write_path(F32, depth, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, depth, base), path, cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k)) == out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, depth, base) : Array<F32>}:
  match depth:
    case 0n:
      %h : {ALeaf{cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, _)} == ALeaf{out_leaf(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, base, Nat.is_lt(base, 1n+_))} : Array<F32>}
      %Equal.sym(Bool, Nat.is_lt(base, 1n+base), True{}, Facts.lt_self_succ(base)) :
        {ALeaf{cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, base)} == ALeaf{out_leaf(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, base, _)} : Array<F32>}
      {==}
    case 1n+inner:
      Model.Step{+bit, +rest} = path
      match bit:
        case False{}:
          +c = Model.capacity(inner)
          +high = Nat.add(c, base)
          +inside = Equal.trans(Bool, Nat.is_lt(k, high), Nat.is_lt(Nat.add(Model.offset(inner, rest), base), high), True{},
            Equal.cong(Nat, Bool, x => Nat.is_lt(x, high), k, Nat.add(Model.offset(inner, rest), base),
              Equal.sym(Nat, Nat.add(Model.offset(inner, rest), base), k, h)),
            Facts.lt_add_right(Model.offset(inner, rest), c, base, Api.offset_lt(inner, rest)))
          %Equal.sym(Array<F32>, ArrayCore.write_path(F32, inner, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, base), rest, cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k)),
            out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, base), write_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, base, rest, h)) :
            {ANode{_, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high)} == out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, 1n+inner, base) : Array<F32>}
          %Equal.sym(Array<F32>, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high), out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, high), above(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high, inside)) :
            {ANode{out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, base), _} == out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, 1n+inner, base) : Array<F32>}
          {==}
        case True{}:
          +c = Model.capacity(inner)
          +high = Nat.add(c, base)
          +off = Model.offset(inner, rest)
          +lower = Equal.trans(Bool, Nat.is_le(high, k), Nat.is_le(high, Nat.add(c, Nat.add(off, base))), True{},
            Equal.cong(Nat, Bool, x => Nat.is_le(high, x), k, Nat.add(c, Nat.add(off, base)),
              Equal.trans(Nat, k, Nat.add(Nat.add(c, off), base), Nat.add(c, Nat.add(off, base)),
                Equal.sym(Nat, Nat.add(Nat.add(c, off), base), k, h),
                Equal.sym(Nat, Nat.add(c, Nat.add(off, base)), Nat.add(Nat.add(c, off), base), Index.add_assoc(c, off, base)))),
            Facts.le_add_cong(c, base, Nat.add(off, base), Api.le_add_left(off, base)))
          +shifted = Equal.trans(Nat, Nat.add(off, high), Nat.add(Nat.add(c, off), base), k, Facts.position_step(c, off, base), h)
          %Equal.sym(Array<F32>, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, base), out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, base), below(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, base, lower)) :
            {ANode{_, ArrayCore.write_path(F32, inner, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high), rest, cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k))} == out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, 1n+inner, base) : Array<F32>}
          %Equal.sym(Array<F32>, ArrayCore.write_path(F32, inner, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high), rest, cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k)),
            out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, high), write_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high, rest, shifted)) :
            {ANode{out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, base), _} == out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, 1n+inner, base) : Array<F32>}
          {==}

# The family history, replayed on the owned storage that `created` makes,
# is the owned storage that holds the output spec.
def chain(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +d: Nat, +n: Nat, -f: {Storage.bounds(d, n) == True{} : Bool}, +fits: {Storage.bounds(d, n) == True{} : Bool},
  +limit: U32, +n_limit: {Nat.is_le(n, U32.to_nat(limit)) == True{} : Bool},
  +k: Nat, +room: {Nat.is_le(k, n) == True{} : Bool}) ->
  {StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, k, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}) == Storage.Owned{d, n, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, d, 0n), out_balanced(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, k, d, 0n), f} : Storage.Storage}:
  match k:
    case 0n:
      Owned.owned_eq(d, n, Array.new(F32, d, 0.0), out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 0n, d, 0n), new_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, d, 0n), ArrayCore.new_balanced(F32, d, 0.0), f)(
        out_balanced(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 0n, d, 0n))
    case 1n+j:
      +index = U32.from_nat(j)
      +value = cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j)
      +below_n = Conversion.le_successor_lt(j, n, room)
      +exact = Conversion.natural_word_exact(j, limit, Facts.le_trans(j, n, U32.to_nat(limit), Conversion.lt_le(j, n, below_n), n_limit))
      +inside = Equal.trans(Bool, Nat.is_lt(U32.to_nat(index), n), Nat.is_lt(j, n), True{},
        Equal.cong(Nat, Bool, x => Nat.is_lt(x, n), U32.to_nat(index), j, exact), below_n)
      +offset = Equal.trans(Nat, Nat.add(Model.offset(d, Refinement.route(d, index)), 0n), U32.to_nat(index), j,
        Facts.output_position(d, n, index, fits, inside), exact)
      +written = Equal.trans(Array<F32>, Array.set(F32, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value), ArrayCore.write_path(F32, d, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), Refinement.route(d, index), value),
        out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n), Refinement.law_set_refines(F32, d, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value, out_balanced(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n)),
        write_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n, Refinement.route(d, index), offset))
      Equal.trans(Storage.Storage, StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, 1n+j, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
        StorageLaws.step(StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, j, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), StorageLaws.Write{index, value}),
        Storage.Owned{d, n, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n), out_balanced(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n), f},
        FromValues.replay_snoc(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, j, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), StorageLaws.Write{index, value}, Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
        %Equal.sym(Storage.Storage, StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, j, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), Storage.Owned{d, n, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), out_balanced(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), f},
          chain(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, d, n, f, fits, limit, n_limit, j, Conversion.le_predecessor(j, n, room))) :
          {StorageLaws.step(_, StorageLaws.Write{index, value}) == Storage.Owned{d, n, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n), out_balanced(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n), f} : Storage.Storage}
        %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, Unit>,
          Storage.set_checked(d, n, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value, Nat.is_lt(U32.to_nat(index), n), {==}, out_balanced(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), f),
          (Storage.Owned{d, n, Array.set(F32, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value),
            Certificates.set_result(F32, d, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value, Array.set(F32, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value), {==}, out_balanced(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n)), f}, Done{Unit{}}),
          Api.set_checked_hit(d, n, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value, Nat.is_lt(U32.to_nat(index), n), {==}, out_balanced(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), f,
            Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), n), True{}, inside), fits, inside)) :
          {StorageLaws.owner(Result<&2, &2, Storage.Error, Unit>, _) == Storage.Owned{d, n, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n), out_balanced(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n), f} : Storage.Storage}
        Owned.owned_eq(d, n, Array.set(F32, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value), out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n), written,
          Certificates.set_result(F32, d, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value, Array.set(F32, out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value), {==}, out_balanced(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n)), f)(
          out_balanced(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n)))

# The plan's input shapes have the inputs' dimensions.
def left_dims(+left: Shape.Shape, +right: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(left, right) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}) ->
  {Shape.dimensions(Shape.left_shape(plan)) == Shape.dimensions(left) : List<&2, U32>}:
  Equal.cong(List<&2, U32> & List<&2, U32>, List<&2, U32>, p => Pair.fst(List<&2, U32>, List<&2, U32>, p),
    (Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(Shape.right_shape(plan))), (Shape.dimensions(left), Shape.dimensions(right)),
    %planned : {ShapeLaws.planned_dimensions(left, right, _) == (Shape.dimensions(left), Shape.dimensions(right)) : List<&2, U32> & List<&2, U32>}
    PlanProof.broadcast_checked_plan(left, right))

def right_dims(+left: Shape.Shape, +right: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(left, right) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}) ->
  {Shape.dimensions(Shape.right_shape(plan)) == Shape.dimensions(right) : List<&2, U32>}:
  Equal.cong(List<&2, U32> & List<&2, U32>, List<&2, U32>, p => Pair.snd(List<&2, U32>, List<&2, U32>, p),
    (Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(Shape.right_shape(plan))), (Shape.dimensions(left), Shape.dimensions(right)),
    %planned : {ShapeLaws.planned_dimensions(left, right, _) == (Shape.dimensions(left), Shape.dimensions(right)) : List<&2, U32> & List<&2, U32>}
    PlanProof.broadcast_checked_plan(left, right))

def left_count(+left: Shape.Shape, +right: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(left, right) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}) ->
  {U32.to_nat(Shape.count(left)) == Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat}:
  Equal.trans(Nat, U32.to_nat(Shape.count(left)), Shape.logical_numel(Shape.dimensions(left)), Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))), Reads.count_exact(left),
    Equal.cong(List<&2, U32>, Nat, dims => Shape.logical_numel(dims), Shape.dimensions(left), Shape.dimensions(Shape.left_shape(plan)),
      Equal.sym(List<&2, U32>, Shape.dimensions(Shape.left_shape(plan)), Shape.dimensions(left), left_dims(left, right, plan, planned))))

def right_count(+left: Shape.Shape, +right: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(left, right) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}) ->
  {U32.to_nat(Shape.count(right)) == Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat}:
  Equal.trans(Nat, U32.to_nat(Shape.count(right)), Shape.logical_numel(Shape.dimensions(right)), Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))), Reads.count_exact(right),
    Equal.cong(List<&2, U32>, Nat, dims => Shape.logical_numel(dims), Shape.dimensions(right), Shape.dimensions(Shape.right_shape(plan)),
      Equal.sym(List<&2, U32>, Shape.dimensions(Shape.right_shape(plan)), Shape.dimensions(right), right_dims(left, right, plan, planned))))

def scalar_cong(+operation: Elementwise.Operation, -x: F32, -y: F32, -u: F32, -v: F32, ex: {x == y : F32}, eu: {u == v : F32}) ->
  {Elementwise.scalar(operation, x, u) == Elementwise.scalar(operation, y, v) : F32}:
  %ex : {Elementwise.scalar(operation, x, u) == Elementwise.scalar(operation, _, v) : F32}
  %eu : {Elementwise.scalar(operation, x, u) == Elementwise.scalar(operation, x, _) : F32}
  {==}

# The build's leaf reads the replayed words the family cell names.
def leaf_same(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +dl: Nat, +dr: Nat, +position: Nat, inside: Bool) ->
  {Build.leaf(operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill, U32.to_nat(Shape.count(left)), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill, U32.to_nat(Shape.count(right)), position, inside) == out_leaf(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, position, inside) : F32}:
  match inside:
    case False{}:
      {==}
    case True{}:
      +li = Build.left_index(plan, position)
      +ri = Build.right_index(plan, position)
      +bli = U32.from_nat(Broadcast.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position))
      +bri = U32.from_nat(Broadcast.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position))
      Equal.trans(F32, Build.word(operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill, U32.to_nat(Shape.count(left)), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill, U32.to_nat(Shape.count(right)), position),
        Elementwise.scalar(operation, StorageLaws.expected(left_ops, U32.to_nat(Shape.count(left)), li, left_fill), StorageLaws.expected(right_ops, U32.to_nat(Shape.count(right)), ri, right_fill)),
        cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, position),
        scalar_cong(operation, Api.current(Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), li, left_fill), StorageLaws.expected(left_ops, U32.to_nat(Shape.count(left)), li, left_fill),
          Api.current(Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), ri, right_fill), StorageLaws.expected(right_ops, U32.to_nat(Shape.count(right)), ri, right_fill),
          Owned.norm_current(U32.to_nat(Shape.count(left)), left_ops, Nil{}, li, left_fill, Api.hit_of(U32.to_nat(Shape.count(left)), left_ops), {==}),
          Owned.norm_current(U32.to_nat(Shape.count(right)), right_ops, Nil{}, ri, right_fill, Api.hit_of(U32.to_nat(Shape.count(right)), right_ops), {==})),
        scalar_cong(operation, StorageLaws.expected(left_ops, U32.to_nat(Shape.count(left)), li, left_fill), StorageLaws.expected(left_ops, U32.to_nat(Shape.count(left)), bli, left_fill),
          StorageLaws.expected(right_ops, U32.to_nat(Shape.count(right)), ri, right_fill), StorageLaws.expected(right_ops, U32.to_nat(Shape.count(right)), bri, right_fill),
          Equal.cong(Nat, F32, w => StorageLaws.expected(left_ops, U32.to_nat(Shape.count(left)), U32.from_nat(w), left_fill),
            Elementwise.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position), Broadcast.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position), Build.walk_left_same(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position)),
          Equal.cong(Nat, F32, w => StorageLaws.expected(right_ops, U32.to_nat(Shape.count(right)), U32.from_nat(w), right_fill),
            Elementwise.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position), Broadcast.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position), Build.walk_right_same(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), position))))

def spec_same(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +dl: Nat, +dr: Nat, +depth: Nat, +base: Nat) ->
  {Build.spec(operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill, U32.to_nat(Shape.count(left)), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill, U32.to_nat(Shape.count(right)), depth, base) == out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(Shape.output_shape(plan))), depth, base) : Array<F32>}:
  match depth:
    case 0n:
      Equal.cong(F32, Array<F32>, x => ALeaf{x}, Build.leaf(operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill, U32.to_nat(Shape.count(left)), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill, U32.to_nat(Shape.count(right)), base, Nat.is_lt(base, U32.to_nat(Shape.count(Shape.output_shape(plan))))), out_leaf(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, base, Nat.is_lt(base, U32.to_nat(Shape.count(Shape.output_shape(plan))))),
        leaf_same(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, dl, dr, base, Nat.is_lt(base, U32.to_nat(Shape.count(Shape.output_shape(plan))))))
    case 1n+inner:
      +high = Nat.add(Model.capacity(inner), base)
      %Equal.sym(Array<F32>, Build.spec(operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill, U32.to_nat(Shape.count(left)), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill, U32.to_nat(Shape.count(right)), inner, base), out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(Shape.output_shape(plan))), inner, base), spec_same(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, dl, dr, inner, base)) :
        {ANode{_, Build.spec(operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill, U32.to_nat(Shape.count(left)), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill, U32.to_nat(Shape.count(right)), inner, high)} == out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(Shape.output_shape(plan))), 1n+inner, base) : Array<F32>}
      %Equal.sym(Array<F32>, Build.spec(operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill, U32.to_nat(Shape.count(left)), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill, U32.to_nat(Shape.count(right)), inner, high), out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(Shape.output_shape(plan))), inner, high), spec_same(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, dl, dr, inner, high)) :
        {ANode{out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(Shape.output_shape(plan))), inner, base), _} == out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(Shape.output_shape(plan))), 1n+inner, base) : Array<F32>}
      {==}

def cells_snoc(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +previous: Nat, +before: List<&2, F32>, +word: F32, +other: F32,
  rest: {before == Laws.cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous) : List<&2, F32>}, last: {word == other : F32}) ->
  {List.append(&2, F32, before, [word]) == List.append(&2, F32, Laws.cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous), [other]) : List<&2, F32>}:
  Equal.trans(List<&2, F32>, List.append(&2, F32, before, [word]), List.append(&2, F32, Laws.cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous), [word]),
    List.append(&2, F32, Laws.cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous), [other]),
    Equal.cong(List<&2, F32>, List<&2, F32>, values => List.append(&2, F32, values, [word]), before, Laws.cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous), rest),
    Equal.cong(F32, List<&2, F32>, value => List.append(&2, F32, Laws.cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous), [value]), word, other, last))

# The words of the family history are `cells`.
def words_cells(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat, +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {Contracts.words(Shape.output_shape(plan), 0.0, Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), count) == Laws.cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count) : List<&2, F32>}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +output = Shape.output_shape(plan)
      +total = Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))
      +below = Conversion.le_successor_lt(previous, total, room)
      +position = U32.from_nat(previous)
      cells_snoc(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous, Contracts.words(output, 0.0, Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), previous), Replayed.value(output, 0.0, Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), position),
        cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous), words_cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous, Conversion.le_predecessor(previous, total, room)),
        Equal.trans(F32, Replayed.value(output, 0.0, Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), position), cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(position)), cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous),
          PointwiseProof.writes_value(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, total, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, output, position,
            PointwiseProof.word_inside(previous, output, below), Conversion.le_reflexive(total)),
          Equal.cong(Nat, F32, number => cell(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, number), U32.to_nat(position), previous, PointwiseProof.word_exact(previous, output, below))))

def half_keeps(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +index: U32) ->
  {Replayed.value(shape, fill, Replayed.snoc(ops, StorageLaws.First{}), index) == Replayed.value(shape, fill, ops, index) : F32}:
  Replayed.expected_snoc(ops, StorageLaws.First{}, U32.to_nat(Shape.count(shape)), index, fill)

def words_snoc(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +previous: Nat, +word: F32, +other: F32,
  +before: List<&2, F32>, rest: {Contracts.words(shape, fill, ops, previous) == before : List<&2, F32>}, last: {word == other : F32}) ->
  {List.append(&2, F32, Contracts.words(shape, fill, ops, previous), [word]) == List.append(&2, F32, before, [other]) : List<&2, F32>}:
  Equal.trans(List<&2, F32>, List.append(&2, F32, Contracts.words(shape, fill, ops, previous), [word]),
    List.append(&2, F32, before, [word]), List.append(&2, F32, before, [other]),
    Equal.cong(List<&2, F32>, List<&2, F32>, values => List.append(&2, F32, values, [word]), Contracts.words(shape, fill, ops, previous), before, rest),
    Equal.cong(F32, List<&2, F32>, value => List.append(&2, F32, before, [value]), word, other, last))

# The First half of a clone lists the same words.
def words_first(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat) ->
  {Contracts.words(shape, fill, Replayed.snoc(ops, StorageLaws.First{}), count) == Contracts.words(shape, fill, ops, count) : List<&2, F32>}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +position = U32.from_nat(previous)
      words_snoc(shape, fill, Replayed.snoc(ops, StorageLaws.First{}), previous,
        Replayed.value(shape, fill, Replayed.snoc(ops, StorageLaws.First{}), position), Replayed.value(shape, fill, ops, position),
        Contracts.words(shape, fill, ops, previous), words_first(shape, fill, ops, previous), half_keeps(shape, fill, ops, position))

# `take` of an owned tensor hands back the First half of its clone and the
# Second half's array.
def take_owned_lift(+shape: Shape.Shape, +d: Nat, +n: Nat, -array: Array<F32>, pair: Array<F32> & Array<F32>,
  -same: {pair == Array.clone(F32, array) : Array<F32> & Array<F32>}, -b: Model.Balanced(F32, d, array),
  -f: {Storage.bounds(d, n) == True{} : Bool}, -fits: {n == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  -Q: (Core.Dynamic & Array<F32>) -> Type,
  h: @-kept: {Lifecycle.logical_length(StorageLaws.owner(Storage.Storage, Storage.finish_clone(d, n, array, pair, same, b, f))) ==
    Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    Q((Core.Packed{shape, Core.TTensor{StorageLaws.owner(Storage.Storage, Storage.finish_clone(d, n, array, pair, same, b, f)), kept}},
      Pair.snd(Array<F32>, Array<F32>, pair)))) ->
  Q(Elementwise.take_owned(shape, d, n, array, pair, same, b, f, fits)):
  (kept, copy) = pair
  h(fits)

def positive_of_lt(+w: Nat, +n: Nat, h: {Nat.is_lt(w, n) == True{} : Bool}) -> {Nat.is_lt(0n, n) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool}, Arith.lt_zero(w, h))
    case 1n+q:
      {==}

def output_positive(+plan: Shape.BroadcastPlan, +d: Nat, +fits: {Storage.bounds(d, U32.to_nat(Shape.count(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {Nat.is_lt(0n, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}:
  Equal.trans(Bool, Nat.is_lt(0n, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))), Nat.is_lt(0n, U32.to_nat(Shape.count(Shape.output_shape(plan)))), True{},
    Equal.cong(Nat, Bool, x => Nat.is_lt(0n, x), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), U32.to_nat(Shape.count(Shape.output_shape(plan))), Equal.sym(Nat, U32.to_nat(Shape.count(Shape.output_shape(plan))), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), Reads.count_exact(Shape.output_shape(plan)))),
    Arith.and_left(Nat.is_lt(0n, U32.to_nat(Shape.count(Shape.output_shape(plan)))), Nat.is_le(U32.to_nat(Shape.count(Shape.output_shape(plan))), Model.capacity(d)),
      Arith.and_right(Nat.is_le(d, 20n), Bool.and(Nat.is_lt(0n, U32.to_nat(Shape.count(Shape.output_shape(plan)))), Nat.is_le(U32.to_nat(Shape.count(Shape.output_shape(plan))), Model.capacity(d))), fits)))

# An output with a cell reads a cell of each input, so neither input is empty.
def left_positive(+left: Shape.Shape, +right: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(left, right) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  +d: Nat, +fits: {Storage.bounds(d, U32.to_nat(Shape.count(Shape.output_shape(plan)))) == True{} : Bool}) -> {Nat.is_lt(0n, U32.to_nat(Shape.count(left))) == True{} : Bool}:
  +walk = Broadcast.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), 0n)
  positive_of_lt(walk, U32.to_nat(Shape.count(left)),
    Equal.trans(Bool, Nat.is_lt(walk, U32.to_nat(Shape.count(left))), Nat.is_lt(walk, Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan)))), True{},
      Equal.cong(Nat, Bool, x => Nat.is_lt(walk, x), U32.to_nat(Shape.count(left)), Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))), left_count(left, right, plan, planned)),
      BroadcastProof.left_walk_bound(plan, Shape.index_map(plan), 0n, output_positive(plan, d, fits))))

def right_positive(+left: Shape.Shape, +right: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(left, right) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  +d: Nat, +fits: {Storage.bounds(d, U32.to_nat(Shape.count(Shape.output_shape(plan)))) == True{} : Bool}) -> {Nat.is_lt(0n, U32.to_nat(Shape.count(right))) == True{} : Bool}:
  +walk = Broadcast.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.index_map(plan), 0n)
  positive_of_lt(walk, U32.to_nat(Shape.count(right)),
    Equal.trans(Bool, Nat.is_lt(walk, U32.to_nat(Shape.count(right))), Nat.is_lt(walk, Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))), True{},
      Equal.cong(Nat, Bool, x => Nat.is_lt(walk, x), U32.to_nat(Shape.count(right)), Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))), right_count(left, right, plan, planned)),
      BroadcastProof.right_walk_bound(plan, Shape.index_map(plan), 0n, output_positive(plan, d, fits))))

def output_eq(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +planned: {Shape.broadcast_checked(left, right) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}, +dl: Nat, +fitsl: {Storage.bounds(dl, U32.to_nat(Shape.count(left))) == True{} : Bool},
  +dr: Nat, +fitsr: {Storage.bounds(dr, U32.to_nat(Shape.count(right))) == True{} : Bool},
  +d: Nat, +fits: {Storage.bounds(d, U32.to_nat(Shape.count(Shape.output_shape(plan)))) == True{} : Bool}, -f: {Storage.bounds(d, U32.to_nat(Shape.count(Shape.output_shape(plan)))) == True{} : Bool}) -> {StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}) == Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Elementwise.build(d, Elementwise.splits(d), operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), 0n, Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill)), Elementwise.build_balanced(d, Elementwise.splits(d), operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), 0n, Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill)), f} : Storage.Storage}:
  +spec = Equal.trans(Array<F32>, Elementwise.build(d, Elementwise.splits(d), operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), 0n, Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill)), Build.spec(operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill, U32.to_nat(Shape.count(left)), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill, U32.to_nat(Shape.count(right)), d, 0n), out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(Shape.output_shape(plan))), d, 0n),
    Build.build_eq(operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill, U32.to_nat(Shape.count(left)), Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill, U32.to_nat(Shape.count(right)), fitsl, fitsr,
      Owned.norm_written(U32.to_nat(Shape.count(left)), left_ops, Nil{}, Api.hit_of(U32.to_nat(Shape.count(left)), left_ops), {==}, {==}),
      Owned.norm_written(U32.to_nat(Shape.count(right)), right_ops, Nil{}, Api.hit_of(U32.to_nat(Shape.count(right)), right_ops), {==}, {==}),
      Reads.count_exact(Shape.output_shape(plan)), left_count(left, right, plan, planned), right_count(left, right, plan, planned),
      d, Elementwise.splits(d), 0n),
    spec_same(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, dl, dr, d, 0n))
  Equal.trans(Storage.Storage, StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, U32.to_nat(Shape.count(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
    Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Elementwise.build(d, Elementwise.splits(d), operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), 0n, Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill)), Elementwise.build_balanced(d, Elementwise.splits(d), operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), 0n, Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill)), f},
    Equal.cong(Nat, Storage.Storage, count => StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, count, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), U32.to_nat(Shape.count(Shape.output_shape(plan))),
      Equal.sym(Nat, U32.to_nat(Shape.count(Shape.output_shape(plan))), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), Reads.count_exact(Shape.output_shape(plan)))),
    Equal.trans(Storage.Storage, StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, U32.to_nat(Shape.count(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(Shape.output_shape(plan))), d, 0n), out_balanced(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(Shape.output_shape(plan))), d, 0n), f},
      Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Elementwise.build(d, Elementwise.splits(d), operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), 0n, Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill)), Elementwise.build_balanced(d, Elementwise.splits(d), operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), 0n, Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill)), f},
      chain(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, d, U32.to_nat(Shape.count(Shape.output_shape(plan))), f, fits, Shape.count(Shape.output_shape(plan)), Conversion.le_reflexive(U32.to_nat(Shape.count(Shape.output_shape(plan)))), U32.to_nat(Shape.count(Shape.output_shape(plan))), Conversion.le_reflexive(U32.to_nat(Shape.count(Shape.output_shape(plan))))),
      Owned.owned_eq(d, U32.to_nat(Shape.count(Shape.output_shape(plan))), out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(Shape.output_shape(plan))), d, 0n), Elementwise.build(d, Elementwise.splits(d), operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), 0n, Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill)), Equal.sym(Array<F32>, Elementwise.build(d, Elementwise.splits(d), operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), 0n, Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill)), out_spec(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(Shape.output_shape(plan))), d, 0n), spec),
        out_balanced(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(Shape.output_shape(plan))), d, 0n), f)(Elementwise.build_balanced(d, Elementwise.splits(d), operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), 0n, Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill)))))

def core(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +planned: {Shape.broadcast_checked(left, right) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}, +dl: Nat, +fitsl: {Storage.bounds(dl, U32.to_nat(Shape.count(left))) == True{} : Bool}, -bl: Model.Balanced(F32, dl, Api.arr(Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill)),
  -fl: {Storage.bounds(dl, U32.to_nat(Shape.count(left))) == True{} : Bool},
  +dr: Nat, +fitsr: {Storage.bounds(dr, U32.to_nat(Shape.count(right))) == True{} : Bool}, -br: Model.Balanced(F32, dr, Api.arr(Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill)),
  -fr: {Storage.bounds(dr, U32.to_nat(Shape.count(right))) == True{} : Bool},
  +d: Nat, +fits: {Storage.bounds(d, U32.to_nat(Shape.count(Shape.output_shape(plan)))) == True{} : Bool}, -f: {Storage.bounds(d, U32.to_nat(Shape.count(Shape.output_shape(plan)))) == True{} : Bool},
  -P: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type,
  hyp: @-kl: {Lifecycle.logical_length(StorageLaws.step(Storage.Owned{dl, U32.to_nat(Shape.count(left)), Api.arr(Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), bl, fl}, StorageLaws.First{})) == Shape.logical_numel(Shape.dimensions(left)) : Nat} -> @-kr: {Lifecycle.logical_length(StorageLaws.step(Storage.Owned{dr, U32.to_nat(Shape.count(right)), Api.arr(Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill), br, fr}, StorageLaws.First{})) == Shape.logical_numel(Shape.dimensions(right)) : Nat} -> @-ko: {Lifecycle.logical_length(StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f})) == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat} -> P(((Core.Packed{left, Core.TTensor{StorageLaws.step(Storage.Owned{dl, U32.to_nat(Shape.count(left)), Api.arr(Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), bl, fl}, StorageLaws.First{}), kl}}, Core.Packed{right, Core.TTensor{StorageLaws.step(Storage.Owned{dr, U32.to_nat(Shape.count(right)), Api.arr(Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill), br, fr}, StorageLaws.First{}), kr}}), Done{Core.Packed{Shape.output_shape(plan), Core.TTensor{StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), ko}}}))) ->
  @-lfits: {Lifecycle.logical_length(Storage.Owned{dl, U32.to_nat(Shape.count(left)), Api.arr(Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), bl, fl}) == Shape.logical_numel(Shape.dimensions(left)) : Nat} -> @-rfits: {Lifecycle.logical_length(Storage.Owned{dr, U32.to_nat(Shape.count(right)), Api.arr(Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill), br, fr}) == Shape.logical_numel(Shape.dimensions(right)) : Nat} ->
  @-fo: {U32.to_nat(Shape.count(Shape.output_shape(plan))) == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat} ->
  P(Elementwise.parallel_output(operation, plan, d, U32.to_nat(Shape.count(Shape.output_shape(plan))), f, fo, Elementwise.take(Core.Packed{left, Core.TTensor{Storage.Owned{dl, U32.to_nat(Shape.count(left)), Api.arr(Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), bl, fl}, lfits}}),
    Elementwise.take(Core.Packed{right, Core.TTensor{Storage.Owned{dr, U32.to_nat(Shape.count(right)), Api.arr(Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill), br, fr}, rfits}}))):
  lfits => rfits => fo =>
    take_owned_lift(left, dl, U32.to_nat(Shape.count(left)), Api.arr(Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), Array.clone(F32, Api.arr(Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill)), {==}, bl, fl, lfits,
      t => P(Elementwise.parallel_output(operation, plan, d, U32.to_nat(Shape.count(Shape.output_shape(plan))), f, fo, t, Elementwise.take(Core.Packed{right, Core.TTensor{Storage.Owned{dr, U32.to_nat(Shape.count(right)), Api.arr(Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill), br, fr}, rfits}}))),
      kl => take_owned_lift(right, dr, U32.to_nat(Shape.count(right)), Api.arr(Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill), Array.clone(F32, Api.arr(Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill)), {==}, br, fr, rfits,
        t => P(Elementwise.parallel_output(operation, plan, d, U32.to_nat(Shape.count(Shape.output_shape(plan))), f, fo, (Core.Packed{left, Core.TTensor{StorageLaws.step(Storage.Owned{dl, U32.to_nat(Shape.count(left)), Api.arr(Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), bl, fl}, StorageLaws.First{}), kl}}, Pair.snd(Array<F32>, Array<F32>, Array.clone(F32, Api.arr(Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill)))), t)),
        kr => Facts.storage_shift(Shape.output_shape(plan), StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Elementwise.build(d, Elementwise.splits(d), operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), 0n, Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill)), Elementwise.build_balanced(d, Elementwise.splits(d), operation, plan, U32.to_nat(Shape.count(Shape.output_shape(plan))), 0n, Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), Api.arr(Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill)), f},
          output_eq(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, planned, dl, fitsl, dr, fitsr, d, fits, f),
          st => k => P(((Core.Packed{left, Core.TTensor{StorageLaws.step(Storage.Owned{dl, U32.to_nat(Shape.count(left)), Api.arr(Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), bl, fl}, StorageLaws.First{}), kl}}, Core.Packed{right, Core.TTensor{StorageLaws.step(Storage.Owned{dr, U32.to_nat(Shape.count(right)), Api.arr(Owned.norm(U32.to_nat(Shape.count(right)), right_ops, Nil{}), dr, right_fill), br, fr}, StorageLaws.First{}), kr}}), Done{Core.Packed{Shape.output_shape(plan), Core.TTensor{st, k}}})), hyp(kl, kr))(fo)))

# The output storage is owned: normalize the left input, then the right one,
# and run `core`.
def owned_case(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +planned: {Shape.broadcast_checked(left, right) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}, -lfits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  -rfits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat},
  -P: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type,
  +d: Nat, +fits: {Storage.bounds(d, U32.to_nat(Shape.count(Shape.output_shape(plan)))) == True{} : Bool}, -f: {Storage.bounds(d, U32.to_nat(Shape.count(Shape.output_shape(plan)))) == True{} : Bool}) ->
  (@+left_kept: List<&2, StorageLaws.Op> -> @left_words: {Contracts.words(left, left_fill, left_kept, Shape.logical_numel(Shape.dimensions(left))) == Contracts.words(left, left_fill, left_ops, Shape.logical_numel(Shape.dimensions(left))) : List<&2, F32>} -> @-left_kfits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_kept)) == Shape.logical_numel(Shape.dimensions(left)) : Nat} -> @+right_kept: List<&2, StorageLaws.Op> -> @right_words: {Contracts.words(right, right_fill, right_kept, Shape.logical_numel(Shape.dimensions(right))) == Contracts.words(right, right_fill, right_ops, Shape.logical_numel(Shape.dimensions(right))) : List<&2, F32>} -> @-right_kfits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_kept)) == Shape.logical_numel(Shape.dimensions(right)) : Nat} -> @-ko: {Lifecycle.logical_length(StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f})) == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat} -> P(((Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_kept), left_kfits}}, Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_kept), right_kfits}}), Done{Core.Packed{Shape.output_shape(plan), Core.TTensor{StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), ko}}}))) -> @-fo: {U32.to_nat(Shape.count(Shape.output_shape(plan))) == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat} ->
  P(Elementwise.parallel_output(operation, plan, d, U32.to_nat(Shape.count(Shape.output_shape(plan))), f, fo, Elementwise.take(Core.pack(left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), lfits})), Elementwise.take(Core.pack(right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), rfits})))):
  ho => fo =>
    Facts.input_lift(left, left_fill, left_ops, left_positive(left, right, plan, planned, d, fits),
      sl => (@-lf: {Lifecycle.logical_length(sl) == Shape.logical_numel(Shape.dimensions(left)) : Nat} -> (@-kl: {Lifecycle.logical_length(StorageLaws.step(sl, StorageLaws.First{})) == Shape.logical_numel(Shape.dimensions(left)) : Nat} -> @+right_kept: List<&2, StorageLaws.Op> -> @right_words: {Contracts.words(right, right_fill, right_kept, Shape.logical_numel(Shape.dimensions(right))) == Contracts.words(right, right_fill, right_ops, Shape.logical_numel(Shape.dimensions(right))) : List<&2, F32>} -> @-right_kfits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_kept)) == Shape.logical_numel(Shape.dimensions(right)) : Nat} -> @-ko: {Lifecycle.logical_length(StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f})) == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat} -> P(((Core.Packed{left, Core.TTensor{StorageLaws.step(sl, StorageLaws.First{}), kl}}, Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_kept), right_kfits}}), Done{Core.Packed{Shape.output_shape(plan), Core.TTensor{StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), ko}}}))) ->
        P(Elementwise.parallel_output(operation, plan, d, U32.to_nat(Shape.count(Shape.output_shape(plan))), f, fo, Elementwise.take(Core.Packed{left, Core.TTensor{sl, lf}}), Elementwise.take(Core.pack(right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), rfits}))))),
      dl => fitsl => bl => fl => lf => hl =>
        Facts.input_lift(right, right_fill, right_ops, right_positive(left, right, plan, planned, d, fits),
          sr => (@-rf: {Lifecycle.logical_length(sr) == Shape.logical_numel(Shape.dimensions(right)) : Nat} -> (@-kl: {Lifecycle.logical_length(StorageLaws.step(Storage.Owned{dl, U32.to_nat(Shape.count(left)), Api.arr(Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), bl, fl}, StorageLaws.First{})) == Shape.logical_numel(Shape.dimensions(left)) : Nat} -> @-kr: {Lifecycle.logical_length(StorageLaws.step(sr, StorageLaws.First{})) == Shape.logical_numel(Shape.dimensions(right)) : Nat} -> @-ko: {Lifecycle.logical_length(StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f})) == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat} -> P(((Core.Packed{left, Core.TTensor{StorageLaws.step(Storage.Owned{dl, U32.to_nat(Shape.count(left)), Api.arr(Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), bl, fl}, StorageLaws.First{}), kl}}, Core.Packed{right, Core.TTensor{StorageLaws.step(sr, StorageLaws.First{}), kr}}), Done{Core.Packed{Shape.output_shape(plan), Core.TTensor{StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), ko}}}))) ->
            P(Elementwise.parallel_output(operation, plan, d, U32.to_nat(Shape.count(Shape.output_shape(plan))), f, fo, Elementwise.take(Core.Packed{left, Core.TTensor{Storage.Owned{dl, U32.to_nat(Shape.count(left)), Api.arr(Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), bl, fl}, lf}}),
              Elementwise.take(Core.Packed{right, Core.TTensor{sr, rf}})))),
          dr => fitsr => br => fr => rf => hr => core(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, planned, dl, fitsl, bl, fl, dr, fitsr, br, fr, d, fits, f, P, hr)(lf, rf, fo))(
          rfits,
          kl => Facts.storage_shift(right, Replayed.storage(right, right_fill, Replayed.snoc(right_ops, StorageLaws.First{})),
            StorageLaws.step(Replayed.storage(right, right_fill, right_ops), StorageLaws.First{}), Replayed.replay_snoc(right_ops, StorageLaws.First{}, right, right_fill),
            st => k => (@-ko: {Lifecycle.logical_length(StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f})) == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat} -> P(((Core.Packed{left, Core.TTensor{StorageLaws.step(Storage.Owned{dl, U32.to_nat(Shape.count(left)), Api.arr(Owned.norm(U32.to_nat(Shape.count(left)), left_ops, Nil{}), dl, left_fill), bl, fl}, StorageLaws.First{}), kl}}, Core.Packed{right, Core.TTensor{st, k}}), Done{Core.Packed{Shape.output_shape(plan), Core.TTensor{StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), ko}}}))),
            rk => hl(kl, Replayed.snoc(right_ops, StorageLaws.First{}), words_first(right, right_fill, right_ops, Shape.logical_numel(Shape.dimensions(right))), rk))))(
      lfits,
      Facts.storage_shift(left, Replayed.storage(left, left_fill, Replayed.snoc(left_ops, StorageLaws.First{})),
        StorageLaws.step(Replayed.storage(left, left_fill, left_ops), StorageLaws.First{}), Replayed.replay_snoc(left_ops, StorageLaws.First{}, left, left_fill),
        st => k => (@+right_kept: List<&2, StorageLaws.Op> -> @right_words: {Contracts.words(right, right_fill, right_kept, Shape.logical_numel(Shape.dimensions(right))) == Contracts.words(right, right_fill, right_ops, Shape.logical_numel(Shape.dimensions(right))) : List<&2, F32>} -> @-right_kfits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_kept)) == Shape.logical_numel(Shape.dimensions(right)) : Nat} -> @-ko: {Lifecycle.logical_length(StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f})) == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat} -> P(((Core.Packed{left, Core.TTensor{st, k}}, Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_kept), right_kfits}}), Done{Core.Packed{Shape.output_shape(plan), Core.TTensor{StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Owned{d, U32.to_nat(Shape.count(Shape.output_shape(plan))), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), ko}}}))),
        lk => ho(Replayed.snoc(left_ops, StorageLaws.First{}), words_first(left, left_fill, left_ops, Shape.logical_numel(Shape.dimensions(left))), lk)))

# The parallel build on two replayed inputs into an output that `Core.new`
# allocates: binary_replayed below the planner, which broadcast_to shares.
def allocated_replayed(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +planned: {Shape.broadcast_checked(left, right) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}, -P: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type,
  h: Laws.BinaryWith(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, P)) ->
  @-lfits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat} -> @-rfits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat} ->
  P(Elementwise.parallel_allocated(operation, plan, Core.pack(left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), lfits}), Core.pack(right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), rfits}), Core.new(Shape.output_shape(plan), 0.0))):
  lfits => rfits =>
    Replayed.new_lift(Shape.output_shape(plan), 0.0, result => P(Elementwise.parallel_allocated(operation, plan, Core.pack(left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), lfits}), Core.pack(right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), rfits}), result)),
      fo => Owned.created_lift(Shape.count(Shape.output_shape(plan)), 0.0, Conversion.shape_count_within(Shape.output_shape(plan)),
        s => ((@+left_kept: List<&2, StorageLaws.Op> -> @left_words: {Contracts.words(left, left_fill, left_kept, Shape.logical_numel(Shape.dimensions(left))) == Contracts.words(left, left_fill, left_ops, Shape.logical_numel(Shape.dimensions(left))) : List<&2, F32>} -> @-left_kfits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_kept)) == Shape.logical_numel(Shape.dimensions(left)) : Nat} -> @+right_kept: List<&2, StorageLaws.Op> -> @right_words: {Contracts.words(right, right_fill, right_kept, Shape.logical_numel(Shape.dimensions(right))) == Contracts.words(right, right_fill, right_ops, Shape.logical_numel(Shape.dimensions(right))) : List<&2, F32>} -> @-right_kfits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_kept)) == Shape.logical_numel(Shape.dimensions(right)) : Nat} -> @-ko: {Lifecycle.logical_length(StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), s)) == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat} -> P(((Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_kept), left_kfits}}, Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_kept), right_kfits}}), Done{Core.Packed{Shape.output_shape(plan), Core.TTensor{StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), s), ko}}}))) -> @-fo: {Lifecycle.logical_length(s) == Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat} ->
          P(Elementwise.parallel_allocated(operation, plan, Core.pack(left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), lfits}), Core.pack(right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), rfits}), Done{Core.TTensor{s, fo}}))),
        zero => ho => fo =>
          Facts.storage_shift(Shape.output_shape(plan), StorageLaws.replay(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), Storage.Empty{}), Storage.Empty{}, Owned.replay_empty(Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops)),
            st => k => P(((Core.pack(left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), lfits}), Core.pack(right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), rfits})), Done{Core.Packed{Shape.output_shape(plan), Core.TTensor{st, k}}})),
            ho(left_ops, {==}, lfits, right_ops, {==}, rfits))(fo),
        d => fits => f => owned_case(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, planned, lfits, rfits, P, d, fits, f))(
        left_kept => left_words => left_kfits => right_kept => right_words => right_kfits => ko =>
          h(left_kept, left_words, left_kfits, right_kept, right_words, right_kfits, Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops),
            words_cells(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), Conversion.le_reflexive(Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))))), ko), fo))

def binary_replayed(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +planned: {Shape.broadcast_checked(left, right) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}, -P: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type,
  h: Laws.BinaryWith(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, P)) ->
  @-lfits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat} -> @-rfits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat} ->
  P(Elementwise.binary(operation, Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), lfits}}, Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), rfits}})):
  lfits => rfits =>
    %Equal.sym(Result<&2, &2, Shape.Error, Shape.BroadcastPlan>, Shape.broadcast_checked(left, right), Done{plan}, planned) :
      P(Elementwise.parallel_planned(operation, Core.pack(left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), lfits}), Core.pack(right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), rfits}), _))
    allocated_replayed(operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, planned, P, h)(lfits, rfits)
