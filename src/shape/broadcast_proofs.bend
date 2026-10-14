import Base
import ./core.bend as Shape

def fin_broadcast(+n: Nat, +index: Shape.Fin(n)) -> {Shape.Fin.value(n, index) == Shape.broadcast_index(n, Shape.Fin.value(n, index)) : Nat}:
  match n:
    case 0n:
      match index:
    case 1n+p:
      match p:
        case 0n:
          match index:
            case Shape.FZero{}:
              {==}
            case Shape.FNext{previous}:
              match previous:
        case 1n+q:
          {==}

def axis_left(+left: Nat, +right: Nat, +output: Nat, +axis: Shape.Axis<left, right, output>, +index: Shape.Fin(output)) -> {Shape.Fin.value(left, Shape.Axis.project_left(left, right, output, axis, index)) == Shape.broadcast_index(left, Shape.Fin.value(output, index)) : Nat}:
  match axis:
    case Shape.AxisEqual{left_eq, right_eq}:
      %h@left_eq : {Shape.Fin.value(_, Shape.Axis.project_left(_, right, output, Shape.AxisEqual{h, right_eq}, index)) == Shape.broadcast_index(_, Shape.Fin.value(output, index)) : Nat}
      fin_broadcast(output, index)
    case Shape.AxisLeftOne{one, right_eq}:
      %h@one : {Shape.Fin.value(_, Shape.Axis.project_left(_, right, output, Shape.AxisLeftOne{h, right_eq}, index)) == Shape.broadcast_index(_, Shape.Fin.value(output, index)) : Nat}
      {==}
    case Shape.AxisRightOne{left_eq, one}:
      %h@left_eq : {Shape.Fin.value(_, Shape.Axis.project_left(_, right, output, Shape.AxisRightOne{h, one}, index)) == Shape.broadcast_index(_, Shape.Fin.value(output, index)) : Nat}
      fin_broadcast(output, index)

def axis_right(+left: Nat, +right: Nat, +output: Nat, +axis: Shape.Axis<left, right, output>, +index: Shape.Fin(output)) -> {Shape.Fin.value(right, Shape.Axis.project_right(left, right, output, axis, index)) == Shape.broadcast_index(right, Shape.Fin.value(output, index)) : Nat}:
  match axis:
    case Shape.AxisEqual{left_eq, right_eq}:
      %h@right_eq : {Shape.Fin.value(_, Shape.Axis.project_right(left, _, output, Shape.AxisEqual{left_eq, h}, index)) == Shape.broadcast_index(_, Shape.Fin.value(output, index)) : Nat}
      fin_broadcast(output, index)
    case Shape.AxisLeftOne{one, right_eq}:
      %h@right_eq : {Shape.Fin.value(_, Shape.Axis.project_right(left, _, output, Shape.AxisLeftOne{one, h}, index)) == Shape.broadcast_index(_, Shape.Fin.value(output, index)) : Nat}
      fin_broadcast(output, index)
    case Shape.AxisRightOne{left_eq, one}:
      %h@one : {Shape.Fin.value(_, Shape.Axis.project_right(left, _, output, Shape.AxisRightOne{left_eq, h}, index)) == Shape.broadcast_index(_, Shape.Fin.value(output, index)) : Nat}
      {==}

def aligned_left(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +alignment: Shape.Aligned(left, right, output), +coordinate: Shape.Coordinate(output)) -> {Shape.indices(left, Shape.Aligned.project_left(left, right, output, alignment, coordinate)) == Shape.broadcast_indices(left, Shape.indices(output, coordinate)) : List<&2, Nat>}:
  match left right output:
    case Nil{} Nil{} Nil{}:
      {==}
    case +lh <> +lt +rh <> +rt +oh <> +ot:
      Shape.AStep{+axis, +tail} = alignment
      Shape.CStep{+index, +rest} = coordinate
      %aligned_left(lt, rt, ot, tail, rest) : {Shape.Fin.value(lh, Shape.Axis.project_left(lh, rh, oh, axis, index)) <> Shape.indices(lt, Shape.Aligned.project_left(lt, rt, ot, tail, rest)) == Shape.broadcast_index(lh, Shape.Fin.value(oh, index)) <> _ : List<&2, Nat>}
      %axis_left(lh, rh, oh, axis, index) : {Shape.Fin.value(lh, Shape.Axis.project_left(lh, rh, oh, axis, index)) <> Shape.indices(lt, Shape.Aligned.project_left(lt, rt, ot, tail, rest)) == _ <> Shape.indices(lt, Shape.Aligned.project_left(lt, rt, ot, tail, rest)) : List<&2, Nat>}
      {==}
    case Nil{} Nil{} oh <> ot:
      match alignment:
    case Nil{} rh <> rt Nil{}:
      match alignment:
    case lh <> lt Nil{} Nil{}:
      match alignment:
    case Nil{} rh <> rt oh <> ot:
      match alignment:
    case lh <> lt Nil{} oh <> ot:
      match alignment:
    case lh <> lt rh <> rt Nil{}:
      match alignment:

def aligned_right(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +alignment: Shape.Aligned(left, right, output), +coordinate: Shape.Coordinate(output)) -> {Shape.indices(right, Shape.Aligned.project_right(left, right, output, alignment, coordinate)) == Shape.broadcast_indices(right, Shape.indices(output, coordinate)) : List<&2, Nat>}:
  match left right output:
    case Nil{} Nil{} Nil{}:
      {==}
    case +lh <> +lt +rh <> +rt +oh <> +ot:
      Shape.AStep{+axis, +tail} = alignment
      Shape.CStep{+index, +rest} = coordinate
      %aligned_right(lt, rt, ot, tail, rest) : {Shape.Fin.value(rh, Shape.Axis.project_right(lh, rh, oh, axis, index)) <> Shape.indices(rt, Shape.Aligned.project_right(lt, rt, ot, tail, rest)) == Shape.broadcast_index(rh, Shape.Fin.value(oh, index)) <> _ : List<&2, Nat>}
      %axis_right(lh, rh, oh, axis, index) : {Shape.Fin.value(rh, Shape.Axis.project_right(lh, rh, oh, axis, index)) <> Shape.indices(rt, Shape.Aligned.project_right(lt, rt, ot, tail, rest)) == _ <> Shape.indices(rt, Shape.Aligned.project_right(lt, rt, ot, tail, rest)) : List<&2, Nat>}
      {==}
    case Nil{} Nil{} oh <> ot:
      match alignment:
    case Nil{} rh <> rt Nil{}:
      match alignment:
    case lh <> lt Nil{} Nil{}:
      match alignment:
    case Nil{} rh <> rt oh <> ot:
      match alignment:
    case lh <> lt Nil{} oh <> ot:
      match alignment:
    case lh <> lt rh <> rt Nil{}:
      match alignment:

def indices_length(+dims: List<&2, Nat>, +coordinate: Shape.Coordinate(dims)) -> {List.length(&2, Nat, dims) == List.length(&2, Nat, Shape.indices(dims, coordinate)) : Nat}:
  match dims:
    case Nil{}:
      {==}
    case +head <> +tail:
      Shape.CStep{+index, +rest} = coordinate
      %indices_length(tail, rest) : {1n+List.length(&2, Nat, tail) == 1n+_ : Nat}
      {==}

def aligned_length_left(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +alignment: Shape.Aligned(left, right, output)) -> {List.length(&2, Nat, left) == List.length(&2, Nat, output) : Nat}:
  match left right output:
    case Nil{} Nil{} Nil{}:
      {==}
    case +lh <> +lt +rh <> +rt +oh <> +ot:
      Shape.AStep{axis, +tail} = alignment
      %aligned_length_left(lt, rt, ot, tail) : {1n+List.length(&2, Nat, lt) == 1n+_ : Nat}
      {==}
    case Nil{} Nil{} oh <> ot:
      match alignment:
    case Nil{} rh <> rt Nil{}:
      match alignment:
    case lh <> lt Nil{} Nil{}:
      match alignment:
    case Nil{} rh <> rt oh <> ot:
      match alignment:
    case lh <> lt Nil{} oh <> ot:
      match alignment:
    case lh <> lt rh <> rt Nil{}:
      match alignment:

def aligned_length_right(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +alignment: Shape.Aligned(left, right, output)) -> {List.length(&2, Nat, right) == List.length(&2, Nat, output) : Nat}:
  match left right output:
    case Nil{} Nil{} Nil{}:
      {==}
    case +lh <> +lt +rh <> +rt +oh <> +ot:
      Shape.AStep{axis, +tail} = alignment
      %aligned_length_right(lt, rt, ot, tail) : {1n+List.length(&2, Nat, rt) == 1n+_ : Nat}
      {==}
    case Nil{} Nil{} oh <> ot:
      match alignment:
    case Nil{} rh <> rt Nil{}:
      match alignment:
    case lh <> lt Nil{} Nil{}:
      match alignment:
    case Nil{} rh <> rt oh <> ot:
      match alignment:
    case lh <> lt Nil{} oh <> ot:
      match alignment:
    case lh <> lt rh <> rt Nil{}:
      match alignment:

def pad_length(+count: Nat, +dims: List<&2, Nat>) -> {Nat.add(count, List.length(&2, Nat, dims)) == List.length(&2, Nat, Shape.pad(count, dims)) : Nat}:
  match count:
    case 0n:
      {==}
    case 1n+rest:
      %pad_length(rest, dims) : {1n+Nat.add(rest, List.length(&2, Nat, dims)) == 1n+_ : Nat}
      {==}

def add_succ(a: Nat, -b: Nat) -> {1n+Nat.add(a, b) == Nat.add(a, 1n+b) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_succ(p, b) : {2n+Nat.add(p, b) == 1n+_ : Nat}
      {==}

def add_zero(a: Nat) -> {Nat.add(a, 0n) == a : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_zero(p) : {1n+Nat.add(p, 0n) == 1n+_ : Nat}
      {==}

def sub_add_zero(a: Nat) -> {a == Nat.sub(Nat.add(a, 0n), 0n) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_zero(p) : {1n+_ == 1n+Nat.add(p, 0n) : Nat}
      {==}

def sub_add(+a: Nat, +b: Nat) -> {a == Nat.sub(Nat.add(a, b), b) : Nat}:
  match b:
    case 0n:
      sub_add_zero(a)
    case 1n+q:
      %add_succ(a, q) : {a == Nat.sub(_, 1n+q) : Nat}
      sub_add(a, q)

def drop_zero(+values: List<&2, Nat>) -> {values == List.drop(&2, Nat, values, 0n) : List<&2, Nat>}:
  match values:
    case Nil{}:
      {==}
    case head <> tail:
      {==}

def broadcast_nil(+dims: List<&2, Nat>) -> {Nil{} == Shape.broadcast_indices(dims, Nil{}) : List<&2, Nat>}:
  match dims:
    case Nil{}:
      {==}
    case head <> tail:
      {==}

def drop_broadcast(+count: Nat, +dims: List<&2, Nat>, +values: List<&2, Nat>) -> {List.drop(&2, Nat, Shape.broadcast_indices(Shape.pad(count, dims), values), count) == Shape.broadcast_indices(dims, List.drop(&2, Nat, values, count)) : List<&2, Nat>}:
  match count:
    case 0n:
      %drop_zero(values) : {List.drop(&2, Nat, Shape.broadcast_indices(dims, values), 0n) == Shape.broadcast_indices(dims, _) : List<&2, Nat>}
      %drop_zero(Shape.broadcast_indices(dims, values)) : {_ == Shape.broadcast_indices(dims, values) : List<&2, Nat>}
      {==}
    case 1n+rest:
      match values:
        case Nil{}:
          broadcast_nil(dims)
        case +head <> +tail:
          drop_broadcast(rest, dims, tail)

def unpad_indices(+count: Nat, +dims: List<&2, Nat>, +coordinate: Shape.Coordinate(Shape.pad(count, dims))) -> {Shape.indices(dims, Shape.unpad(count, dims, coordinate)) == List.drop(&2, Nat, Shape.indices(Shape.pad(count, dims), coordinate), count) : List<&2, Nat>}:
  match count:
    case 0n:
      drop_zero(Shape.indices(dims, coordinate))
    case 1n+rest:
      Shape.CStep{index, +tail} = coordinate
      unpad_indices(rest, dims, tail)

def project_left(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +alignment: Shape.Broadcast<left, right, output>, +coordinate: Shape.Coordinate(output)) -> {Shape.indices(left, Shape.project_left(left, right, output, alignment, coordinate)) == Shape.project_indices(left, Shape.indices(output, coordinate)) : List<&2, Nat>}:
  match alignment:
    case Shape.BAlign{+lp, +rp, minimal, +aligned}:
      %indices_length(output, coordinate) : {Shape.indices(left, Shape.unpad(lp, left, Shape.Aligned.project_left(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned, coordinate))) == Shape.broadcast_indices(left, List.drop(&2, Nat, Shape.indices(output, coordinate), Nat.sub(_, List.length(&2, Nat, left)))) : List<&2, Nat>}
      %aligned_length_left(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned) : {Shape.indices(left, Shape.unpad(lp, left, Shape.Aligned.project_left(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned, coordinate))) == Shape.broadcast_indices(left, List.drop(&2, Nat, Shape.indices(output, coordinate), Nat.sub(_, List.length(&2, Nat, left)))) : List<&2, Nat>}
      %pad_length(lp, left) : {Shape.indices(left, Shape.unpad(lp, left, Shape.Aligned.project_left(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned, coordinate))) == Shape.broadcast_indices(left, List.drop(&2, Nat, Shape.indices(output, coordinate), Nat.sub(_, List.length(&2, Nat, left)))) : List<&2, Nat>}
      %sub_add(lp, List.length(&2, Nat, left)) : {Shape.indices(left, Shape.unpad(lp, left, Shape.Aligned.project_left(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned, coordinate))) == Shape.broadcast_indices(left, List.drop(&2, Nat, Shape.indices(output, coordinate), _)) : List<&2, Nat>}
      %drop_broadcast(lp, left, Shape.indices(output, coordinate)) : {Shape.indices(left, Shape.unpad(lp, left, Shape.Aligned.project_left(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned, coordinate))) == _ : List<&2, Nat>}
      %aligned_left(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned, coordinate) : {Shape.indices(left, Shape.unpad(lp, left, Shape.Aligned.project_left(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned, coordinate))) == List.drop(&2, Nat, _, lp) : List<&2, Nat>}
      unpad_indices(lp, left, Shape.Aligned.project_left(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned, coordinate))

def project_right(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +alignment: Shape.Broadcast<left, right, output>, +coordinate: Shape.Coordinate(output)) -> {Shape.indices(right, Shape.project_right(left, right, output, alignment, coordinate)) == Shape.project_indices(right, Shape.indices(output, coordinate)) : List<&2, Nat>}:
  match alignment:
    case Shape.BAlign{+lp, +rp, minimal, +aligned}:
      %indices_length(output, coordinate) : {Shape.indices(right, Shape.unpad(rp, right, Shape.Aligned.project_right(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned, coordinate))) == Shape.broadcast_indices(right, List.drop(&2, Nat, Shape.indices(output, coordinate), Nat.sub(_, List.length(&2, Nat, right)))) : List<&2, Nat>}
      %aligned_length_right(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned) : {Shape.indices(right, Shape.unpad(rp, right, Shape.Aligned.project_right(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned, coordinate))) == Shape.broadcast_indices(right, List.drop(&2, Nat, Shape.indices(output, coordinate), Nat.sub(_, List.length(&2, Nat, right)))) : List<&2, Nat>}
      %pad_length(rp, right) : {Shape.indices(right, Shape.unpad(rp, right, Shape.Aligned.project_right(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned, coordinate))) == Shape.broadcast_indices(right, List.drop(&2, Nat, Shape.indices(output, coordinate), Nat.sub(_, List.length(&2, Nat, right)))) : List<&2, Nat>}
      %sub_add(rp, List.length(&2, Nat, right)) : {Shape.indices(right, Shape.unpad(rp, right, Shape.Aligned.project_right(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned, coordinate))) == Shape.broadcast_indices(right, List.drop(&2, Nat, Shape.indices(output, coordinate), _)) : List<&2, Nat>}
      %drop_broadcast(rp, right, Shape.indices(output, coordinate)) : {Shape.indices(right, Shape.unpad(rp, right, Shape.Aligned.project_right(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned, coordinate))) == _ : List<&2, Nat>}
      %aligned_right(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned, coordinate) : {Shape.indices(right, Shape.unpad(rp, right, Shape.Aligned.project_right(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned, coordinate))) == List.drop(&2, Nat, _, rp) : List<&2, Nat>}
      unpad_indices(rp, right, Shape.Aligned.project_right(Shape.pad(lp, left), Shape.pad(rp, right), output, aligned, coordinate))
