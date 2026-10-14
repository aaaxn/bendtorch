import Base
import ./core.bend as Shape
import ./nat_proofs.bend as Nats
import ./broadcast_proofs.bend as Broadcast
import ./unique_proofs.bend as Unique
import ./coordinate_proofs.bend as Coordinates

def strides_of(+dims: List<&2, Nat>) -> List<&2, Nat>:
  match dims:
    case Nil{}:
      Nil{}
    case head <> tail:
      Shape.broadcast_index(head, Shape.numel(tail)) <> strides_of(tail)

def zeros(count: Nat, values: List<&2, Nat>) -> List<&2, Nat>:
  match count:
    case 0n:
      values
    case 1n+rest:
      0n <> zeros(rest, values)

def dot(steps: List<&2, Nat>, values: List<&2, Nat>) -> Nat:
  match steps values:
    case step <> rest +value <> tail:
      Nat.add(Nat.mul(step, value), dot(rest, tail))
    case steps values:
      0n

def lt_zero_false(+value: Nat) -> {Nat.is_lt(value, 0n) == False{} : Bool}:
  match value:
    case 0n:
      {==}
    case 1n+p:
      {==}

def walk_left_rows(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +index: Nat, same: {List.length(&2, Nat, right) == List.length(&2, Nat, output) : Nat}) -> {Shape.walk_left(Shape.index_rows(left, right, output), index) == dot(strides_of(left), Shape.digits(output, index)) : Nat}:
  match left right output:
    case +lh <> +lt +rh <> +rt +oh <> +ot:
      +stride = Shape.numel(ot)
      Equal.cong(Nat, Nat, rest => Nat.add(Nat.mul(Shape.broadcast_index(lh, Shape.numel(lt)), Nat.div(index, stride)), rest), Shape.walk_left(Shape.index_rows(lt, rt, ot), Nat.mod(index, stride)), dot(strides_of(lt), Shape.digits(ot, Nat.mod(index, stride))), walk_left_rows(lt, rt, ot, Nat.mod(index, stride), Unique.succ_injective(List.length(&2, Nat, rt), List.length(&2, Nat, ot), same)))
    case Nil{} right output:
      {==}
    case lh <> lt Nil{} Nil{}:
      {==}
    case lh <> lt Nil{} oh <> ot:
      Empty.absurd({Shape.walk_left(Shape.index_rows(lh <> lt, Nil{}, oh <> ot), index) == dot(strides_of(lh <> lt), Shape.digits(oh <> ot, index)) : Nat}, Unique.succ_zero(List.length(&2, Nat, ot), Equal.sym(Nat, 0n, 1n+List.length(&2, Nat, ot), same)))
    case lh <> lt rh <> rt Nil{}:
      {==}

def walk_right_rows(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +index: Nat, same: {List.length(&2, Nat, left) == List.length(&2, Nat, output) : Nat}) -> {Shape.walk_right(Shape.index_rows(left, right, output), index) == dot(strides_of(right), Shape.digits(output, index)) : Nat}:
  match left right output:
    case +lh <> +lt +rh <> +rt +oh <> +ot:
      +stride = Shape.numel(ot)
      Equal.cong(Nat, Nat, rest => Nat.add(Nat.mul(Shape.broadcast_index(rh, Shape.numel(rt)), Nat.div(index, stride)), rest), Shape.walk_right(Shape.index_rows(lt, rt, ot), Nat.mod(index, stride)), dot(strides_of(rt), Shape.digits(ot, Nat.mod(index, stride))), walk_right_rows(lt, rt, ot, Nat.mod(index, stride), Unique.succ_injective(List.length(&2, Nat, lt), List.length(&2, Nat, ot), same)))
    case Nil{} Nil{} Nil{}:
      {==}
    case Nil{} rh <> rt Nil{}:
      {==}
    case Nil{} right oh <> ot:
      Empty.absurd({Shape.walk_right(Shape.index_rows(Nil{}, right, oh <> ot), index) == dot(strides_of(right), Shape.digits(oh <> ot, index)) : Nat}, Unique.succ_zero(List.length(&2, Nat, ot), Equal.sym(Nat, 0n, 1n+List.length(&2, Nat, ot), same)))
    case lh <> lt Nil{} output:
      {==}
    case lh <> lt rh <> rt Nil{}:
      {==}

def pad_strides(+count: Nat, +dims: List<&2, Nat>) -> {strides_of(Shape.pad(count, dims)) == zeros(count, strides_of(dims)) : List<&2, Nat>}:
  match count:
    case 0n:
      {==}
    case 1n+rest:
      Equal.cong(List<&2, Nat>, List<&2, Nat>, tail => 0n <> tail, strides_of(Shape.pad(rest, dims)), zeros(rest, strides_of(dims)), pad_strides(rest, dims))

def dot_zeros(+count: Nat, +values: List<&2, Nat>, +steps: List<&2, Nat>) -> {dot(zeros(count, steps), values) == dot(steps, List.drop(&2, Nat, values, count)) : Nat}:
  match count values:
    case 0n values:
      %Broadcast.drop_zero(values) : {dot(steps, values) == dot(steps, _) : Nat}
      {==}
    case 1n+rest Nil{}:
      match steps:
        case Nil{}:
          {==}
        case head <> tail:
          {==}
    case 1n+rest +value <> +tail:
      dot_zeros(rest, tail, steps)

def digits_length(+dims: List<&2, Nat>, +index: Nat) -> {List.length(&2, Nat, Shape.digits(dims, index)) == List.length(&2, Nat, dims) : Nat}:
  match dims:
    case Nil{}:
      {==}
    case head <> +tail:
      Equal.cong(Nat, Nat, value => 1n+value, List.length(&2, Nat, Shape.digits(tail, Nat.mod(index, Shape.numel(tail)))), List.length(&2, Nat, tail), digits_length(tail, Nat.mod(index, Shape.numel(tail))))

def axis_term(+size: Nat, +stride: Nat, +value: Nat) -> {Nat.mul(Shape.broadcast_index(size, stride), value) == Nat.mul(Shape.broadcast_index(size, value), stride) : Nat}:
  match size:
    case 0n:
      Nats.mul_comm(stride, value)
    case 1n+p:
      match p:
        case 0n:
          {==}
        case 1n+q:
          Nats.mul_comm(stride, value)

def dot_broadcast(+dims: List<&2, Nat>, +values: List<&2, Nat>) -> {dot(strides_of(dims), values) == Shape.offset(dims, Shape.broadcast_indices(dims, values)) : Nat}:
  match dims values:
    case +head <> +tail +value <> +rest:
      %Equal.sym(Nat, dot(strides_of(tail), rest), Shape.offset(tail, Shape.broadcast_indices(tail, rest)), dot_broadcast(tail, rest)) : {Nat.add(Nat.mul(Shape.broadcast_index(head, Shape.numel(tail)), value), _) == Nat.add(Nat.mul(Shape.broadcast_index(head, value), Shape.numel(tail)), Shape.offset(tail, Shape.broadcast_indices(tail, rest))) : Nat}
      Equal.cong(Nat, Nat, term => Nat.add(term, Shape.offset(tail, Shape.broadcast_indices(tail, rest))), Nat.mul(Shape.broadcast_index(head, Shape.numel(tail)), value), Nat.mul(Shape.broadcast_index(head, value), Shape.numel(tail)), axis_term(head, Shape.numel(tail), value))
    case Nil{} values:
      {==}
    case head <> tail Nil{}:
      {==}

def padding_of(-width: Nat, +padding: Nat, +rank: Nat, sum: {Nat.add(padding, rank) == width : Nat}) -> {padding == Nat.sub(width, rank) : Nat}:
  Equal.trans(Nat, padding, Nat.sub(Nat.add(padding, rank), rank), Nat.sub(width, rank), Broadcast.sub_add(padding, rank), Equal.cong(Nat, Nat, value => Nat.sub(value, rank), Nat.add(padding, rank), width, sum))

def left_padded(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +alignment: Shape.Broadcast<left, right, output>) -> {List.length(&2, Nat, Shape.pad(Nat.sub(List.length(&2, Nat, output), List.length(&2, Nat, left)), left)) == List.length(&2, Nat, output) : Nat}:
  match alignment:
    case Shape.BAlign{+lp, +rp, minimal, +aligned}:
      +width = Broadcast.aligned_length_left(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned)
      %padding_of(List.length(&2, Nat, output), lp, List.length(&2, Nat, left), Equal.trans(Nat, Nat.add(lp, List.length(&2, Nat, left)), List.length(&2, Nat, Shape.pad(lp, left)), List.length(&2, Nat, output), Broadcast.pad_length(lp, left), width)) : {List.length(&2, Nat, Shape.pad(_, left)) == List.length(&2, Nat, output) : Nat}
      width

def right_padded(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +alignment: Shape.Broadcast<left, right, output>) -> {List.length(&2, Nat, Shape.pad(Nat.sub(List.length(&2, Nat, output), List.length(&2, Nat, right)), right)) == List.length(&2, Nat, output) : Nat}:
  match alignment:
    case Shape.BAlign{+lp, +rp, minimal, +aligned}:
      +width = Broadcast.aligned_length_right(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned)
      %padding_of(List.length(&2, Nat, output), rp, List.length(&2, Nat, right), Equal.trans(Nat, Nat.add(rp, List.length(&2, Nat, right)), List.length(&2, Nat, Shape.pad(rp, right)), List.length(&2, Nat, output), Broadcast.pad_length(rp, right), width)) : {List.length(&2, Nat, Shape.pad(_, right)) == List.length(&2, Nat, output) : Nat}
      width

def walk_left_value(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +alignment: Shape.Broadcast<left, right, output>, +index: Shape.Fin(Shape.numel(output))) -> {Shape.walk_left(Shape.index_table(left, right, output), Shape.Fin.value(Shape.numel(output), index)) == Shape.row_major(left, Shape.project_left(left, right, output, alignment, Shape.unflatten(output, index))) : Nat}:
  +value = Shape.Fin.value(Shape.numel(output), index)
  +ds = Shape.digits(output, value)
  +nl = Nat.sub(List.length(&2, Nat, output), List.length(&2, Nat, left))
  +nr = Nat.sub(List.length(&2, Nat, output), List.length(&2, Nat, right))
  +coordinate = Shape.unflatten(output, index)
  +projected = Shape.project_left(left, right, output, alignment, coordinate)
  +strides = strides_of(left)
  +dropped = List.drop(&2, Nat, ds, nl)
  %Equal.sym(Nat, Shape.walk_left(Shape.index_rows(Shape.pad(nl, left), Shape.pad(nr, right), output), value), dot(strides_of(Shape.pad(nl, left)), ds), walk_left_rows(Shape.pad(nl, left), Shape.pad(nr, right), output, value, right_padded(left, right, output, alignment))) : {_ == Shape.row_major(left, projected) : Nat}
  %Equal.cong(List<&2, Nat>, Nat, steps => dot(steps, ds), zeros(nl, strides), strides_of(Shape.pad(nl, left)), Equal.sym(List<&2, Nat>, strides_of(Shape.pad(nl, left)), zeros(nl, strides), pad_strides(nl, left))) : {_ == Shape.row_major(left, projected) : Nat}
  %Equal.sym(Nat, dot(zeros(nl, strides), ds), dot(strides, dropped), dot_zeros(nl, ds, strides)) : {_ == Shape.row_major(left, projected) : Nat}
  %Equal.sym(Nat, dot(strides, dropped), Shape.offset(left, Shape.broadcast_indices(left, dropped)), dot_broadcast(left, dropped)) : {_ == Shape.row_major(left, projected) : Nat}
  %Equal.cong(Nat, Nat, width => Shape.offset(left, Shape.broadcast_indices(left, List.drop(&2, Nat, ds, Nat.sub(width, List.length(&2, Nat, left))))), List.length(&2, Nat, ds), List.length(&2, Nat, output), digits_length(output, value)) : {_ == Shape.row_major(left, projected) : Nat}
  %Equal.cong(List<&2, Nat>, Nat, values => Shape.offset(left, Shape.project_indices(left, values)), Shape.indices(output, coordinate), ds, Coordinates.digits_unflatten(output, index)) : {_ == Shape.row_major(left, projected) : Nat}
  %Equal.cong(List<&2, Nat>, Nat, values => Shape.offset(left, values), Shape.indices(left, projected), Shape.project_indices(left, Shape.indices(output, coordinate)), Broadcast.project_left(left, right, output, alignment, coordinate)) : {_ == Shape.row_major(left, projected) : Nat}
  Coordinates.offset_indices(left, projected)

def walk_right_value(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +alignment: Shape.Broadcast<left, right, output>, +index: Shape.Fin(Shape.numel(output))) -> {Shape.walk_right(Shape.index_table(left, right, output), Shape.Fin.value(Shape.numel(output), index)) == Shape.row_major(right, Shape.project_right(left, right, output, alignment, Shape.unflatten(output, index))) : Nat}:
  +value = Shape.Fin.value(Shape.numel(output), index)
  +ds = Shape.digits(output, value)
  +nl = Nat.sub(List.length(&2, Nat, output), List.length(&2, Nat, left))
  +nr = Nat.sub(List.length(&2, Nat, output), List.length(&2, Nat, right))
  +coordinate = Shape.unflatten(output, index)
  +projected = Shape.project_right(left, right, output, alignment, coordinate)
  +strides = strides_of(right)
  +dropped = List.drop(&2, Nat, ds, nr)
  %Equal.sym(Nat, Shape.walk_right(Shape.index_rows(Shape.pad(nl, left), Shape.pad(nr, right), output), value), dot(strides_of(Shape.pad(nr, right)), ds), walk_right_rows(Shape.pad(nl, left), Shape.pad(nr, right), output, value, left_padded(left, right, output, alignment))) : {_ == Shape.row_major(right, projected) : Nat}
  %Equal.cong(List<&2, Nat>, Nat, steps => dot(steps, ds), zeros(nr, strides), strides_of(Shape.pad(nr, right)), Equal.sym(List<&2, Nat>, strides_of(Shape.pad(nr, right)), zeros(nr, strides), pad_strides(nr, right))) : {_ == Shape.row_major(right, projected) : Nat}
  %Equal.sym(Nat, dot(zeros(nr, strides), ds), dot(strides, dropped), dot_zeros(nr, ds, strides)) : {_ == Shape.row_major(right, projected) : Nat}
  %Equal.sym(Nat, dot(strides, dropped), Shape.offset(right, Shape.broadcast_indices(right, dropped)), dot_broadcast(right, dropped)) : {_ == Shape.row_major(right, projected) : Nat}
  %Equal.cong(Nat, Nat, width => Shape.offset(right, Shape.broadcast_indices(right, List.drop(&2, Nat, ds, Nat.sub(width, List.length(&2, Nat, right))))), List.length(&2, Nat, ds), List.length(&2, Nat, output), digits_length(output, value)) : {_ == Shape.row_major(right, projected) : Nat}
  %Equal.cong(List<&2, Nat>, Nat, values => Shape.offset(right, Shape.project_indices(right, values)), Shape.indices(output, coordinate), ds, Coordinates.digits_unflatten(output, index)) : {_ == Shape.row_major(right, projected) : Nat}
  %Equal.cong(List<&2, Nat>, Nat, values => Shape.offset(right, values), Shape.indices(right, projected), Shape.project_indices(right, Shape.indices(output, coordinate)), Broadcast.project_right(left, right, output, alignment, coordinate)) : {_ == Shape.row_major(right, projected) : Nat}
  Coordinates.offset_indices(right, projected)
