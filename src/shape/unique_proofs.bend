import Base
import ./core.bend as Shape
import ./index_proofs.bend as Index
import ./broadcast_proofs.bend as Broadcast

def pred(value: Nat) -> Nat:
  match value:
    case 0n:
      0n
    case 1n+p:
      p

def positive(value: Nat) -> Data:
  match value:
    case 0n:
      Empty
    case 1n+p:
      Unit

def succ_injective(-a: Nat, -b: Nat, equal: {1n+a == 1n+b : Nat}) -> {a == b : Nat}:
  Equal.cong(Nat, Nat, pred, 1n+a, 1n+b, equal)

def cancel_right(+a: Nat, +b: Nat, +c: Nat, equal: {Nat.add(a, c) == Nat.add(b, c) : Nat}) -> {a == b : Nat}:
  match c:
    case 0n:
      Equal.trans(Nat, a, Nat.add(a, 0n), b, Equal.sym(Nat, Nat.add(a, 0n), a, Broadcast.add_zero(a)), Equal.trans(Nat, Nat.add(a, 0n), Nat.add(b, 0n), b, equal, Broadcast.add_zero(b)))
    case 1n+p:
      cancel_right(a, b, p, succ_injective(Nat.add(a, p), Nat.add(b, p), Equal.trans(Nat, 1n+Nat.add(a, p), Nat.add(a, 1n+p), 1n+Nat.add(b, p), Broadcast.add_succ(a, p), Equal.trans(Nat, Nat.add(a, 1n+p), Nat.add(b, 1n+p), 1n+Nat.add(b, p), equal, Equal.sym(Nat, 1n+Nat.add(b, p), Nat.add(b, 1n+p), Broadcast.add_succ(b, p))))))

def absorb(+a: Nat, +c: Nat, equal: {Nat.add(a, c) == c : Nat}) -> {Nat.add(a, 0n) == 0n : Nat}:
  match c:
    case 0n:
      equal
    case 1n+p:
      absorb(a, p, succ_injective(Nat.add(a, p), p, Equal.trans(Nat, 1n+Nat.add(a, p), Nat.add(a, 1n+p), 1n+p, Broadcast.add_succ(a, p), equal)))

def succ_zero(-value: Nat, equal: {1n+value == 0n : Nat}) -> Empty:
  %equal : positive(_)
  Unit{}

def sum_zero(+a: Nat, -b: Nat, equal: {Nat.add(a, b) == 0n : Nat}) -> {a == 0n : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      Empty.absurd({1n+p == 0n : Nat}, succ_zero(Nat.add(p, b), equal))

def balance(+left_padding: Nat, +right_padding: Nat, +left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +aligned: Shape.Aligned(Shape.pad(left_padding, left), Shape.pad(right_padding, right), output)) -> {Nat.add(left_padding, List.length(&2, Nat, left)) == Nat.add(right_padding, List.length(&2, Nat, right)) : Nat}:
  Equal.trans(Nat, Nat.add(left_padding, List.length(&2, Nat, left)), List.length(&2, Nat, Shape.pad(left_padding, left)), Nat.add(right_padding, List.length(&2, Nat, right)), Broadcast.pad_length(left_padding, left), Equal.trans(Nat, List.length(&2, Nat, Shape.pad(left_padding, left)), List.length(&2, Nat, output), Nat.add(right_padding, List.length(&2, Nat, right)), Broadcast.aligned_length_left(Shape.pad(left_padding, left), Shape.pad(right_padding, right), output, aligned), Equal.trans(Nat, List.length(&2, Nat, output), List.length(&2, Nat, Shape.pad(right_padding, right)), Nat.add(right_padding, List.length(&2, Nat, right)), Equal.sym(Nat, List.length(&2, Nat, Shape.pad(right_padding, right)), List.length(&2, Nat, output), Broadcast.aligned_length_right(Shape.pad(left_padding, left), Shape.pad(right_padding, right), output, aligned)), Equal.sym(Nat, Nat.add(right_padding, List.length(&2, Nat, right)), List.length(&2, Nat, Shape.pad(right_padding, right)), Broadcast.pad_length(right_padding, right)))))

def crossed(+zero_left: Nat, +right_a: Nat, +left_b: Nat, +zero_right: Nat, +left_rank: Nat, +right_rank: Nat, +zero_a: {0n == zero_left : Nat}, zero_b: {0n == zero_right : Nat}, first: {Nat.add(zero_left, left_rank) == Nat.add(right_a, right_rank) : Nat}, second: {Nat.add(left_b, left_rank) == Nat.add(zero_right, right_rank) : Nat}) -> {zero_left == left_b : Nat}:
  +left_rank_eq = Equal.trans(Nat, left_rank, Nat.add(zero_left, left_rank), Nat.add(right_a, right_rank), Equal.cong(Nat, Nat, value => Nat.add(value, left_rank), 0n, zero_left, zero_a), first)
  +second_eq = Equal.trans(Nat, Nat.add(left_b, left_rank), Nat.add(zero_right, right_rank), right_rank, second, Equal.sym(Nat, right_rank, Nat.add(zero_right, right_rank), Equal.cong(Nat, Nat, value => Nat.add(value, right_rank), 0n, zero_right, zero_b)))
  +nested = Equal.trans(Nat, Nat.add(left_b, Nat.add(right_a, right_rank)), Nat.add(left_b, left_rank), right_rank, Equal.sym(Nat, Nat.add(left_b, left_rank), Nat.add(left_b, Nat.add(right_a, right_rank)), Equal.cong(Nat, Nat, value => Nat.add(left_b, value), left_rank, Nat.add(right_a, right_rank), left_rank_eq)), second_eq)
  +flat = Equal.trans(Nat, Nat.add(Nat.add(left_b, right_a), right_rank), Nat.add(left_b, Nat.add(right_a, right_rank)), right_rank, Equal.sym(Nat, Nat.add(left_b, Nat.add(right_a, right_rank)), Nat.add(Nat.add(left_b, right_a), right_rank), Index.add_assoc(left_b, right_a, right_rank)), nested)
  +sum = Equal.trans(Nat, Nat.add(left_b, right_a), Nat.add(Nat.add(left_b, right_a), 0n), 0n, Equal.sym(Nat, Nat.add(Nat.add(left_b, right_a), 0n), Nat.add(left_b, right_a), Broadcast.add_zero(Nat.add(left_b, right_a))), absorb(Nat.add(left_b, right_a), right_rank, flat))
  Equal.trans(Nat, zero_left, 0n, left_b, Equal.sym(Nat, 0n, zero_left, zero_a), Equal.sym(Nat, left_b, 0n, sum_zero(left_b, right_a, sum)))

def unpadded(+left_padding: Nat, +right_padding: Nat, +left_rank: Nat, +right_rank: Nat, zero: {0n == right_padding : Nat}, balanced: {Nat.add(left_padding, left_rank) == Nat.add(right_padding, right_rank) : Nat}) -> {Nat.add(left_padding, left_rank) == right_rank : Nat}:
  Equal.trans(Nat, Nat.add(left_padding, left_rank), Nat.add(right_padding, right_rank), right_rank, balanced, Equal.sym(Nat, right_rank, Nat.add(right_padding, right_rank), Equal.cong(Nat, Nat, value => Nat.add(value, right_rank), 0n, right_padding, zero)))

def left_padding_equal(+lp1: Nat, +rp1: Nat, +lp2: Nat, +rp2: Nat, +left_rank: Nat, +right_rank: Nat, +m1: Shape.PadMinimal<lp1, rp1>, +m2: Shape.PadMinimal<lp2, rp2>, +first: {Nat.add(lp1, left_rank) == Nat.add(rp1, right_rank) : Nat}, +second: {Nat.add(lp2, left_rank) == Nat.add(rp2, right_rank) : Nat}) -> {lp1 == lp2 : Nat}:
  match m1 m2:
    case Shape.PLeftZero{z1} Shape.PLeftZero{z2}:
      Equal.trans(Nat, lp1, 0n, lp2, Equal.sym(Nat, 0n, lp1, z1), z2)
    case Shape.PRightZero{z1} Shape.PRightZero{z2}:
      cancel_right(lp1, lp2, left_rank, Equal.trans(Nat, Nat.add(lp1, left_rank), right_rank, Nat.add(lp2, left_rank), unpadded(lp1, rp1, left_rank, right_rank, z1, first), Equal.sym(Nat, Nat.add(lp2, left_rank), right_rank, unpadded(lp2, rp2, left_rank, right_rank, z2, second))))
    case Shape.PLeftZero{z1} Shape.PRightZero{z2}:
      crossed(lp1, rp1, lp2, rp2, left_rank, right_rank, z1, z2, first, second)
    case Shape.PRightZero{z1} Shape.PLeftZero{z2}:
      Equal.sym(Nat, lp2, lp1, crossed(lp2, rp2, lp1, rp1, left_rank, right_rank, z2, z1, second, first))

def axis_unique(-left: Nat, -right: Nat, -output: Nat, -other: Nat, +first: Shape.Axis<left, right, output>, +second: Shape.Axis<left, right, other>) -> {output == other : Nat}:
  match first second:
    case Shape.AxisEqual{l1, r1} Shape.AxisEqual{l2, r2}:
      Equal.trans(Nat, output, left, other, l1, Equal.sym(Nat, other, left, l2))
    case Shape.AxisEqual{l1, r1} Shape.AxisLeftOne{o2, r2}:
      Equal.trans(Nat, output, right, other, r1, Equal.sym(Nat, other, right, r2))
    case Shape.AxisEqual{l1, r1} Shape.AxisRightOne{l2, o2}:
      Equal.trans(Nat, output, left, other, l1, Equal.sym(Nat, other, left, l2))
    case Shape.AxisLeftOne{o1, r1} Shape.AxisEqual{l2, r2}:
      Equal.trans(Nat, output, right, other, r1, Equal.sym(Nat, other, right, r2))
    case Shape.AxisLeftOne{o1, r1} Shape.AxisLeftOne{o2, r2}:
      Equal.trans(Nat, output, right, other, r1, Equal.sym(Nat, other, right, r2))
    case Shape.AxisLeftOne{o1, r1} Shape.AxisRightOne{l2, o2}:
      Equal.trans(Nat, output, 1n, other, Equal.trans(Nat, output, right, 1n, r1, Equal.sym(Nat, 1n, right, o2)), Equal.trans(Nat, 1n, left, other, o1, Equal.sym(Nat, other, left, l2)))
    case Shape.AxisRightOne{l1, o1} Shape.AxisEqual{l2, r2}:
      Equal.trans(Nat, output, left, other, l1, Equal.sym(Nat, other, left, l2))
    case Shape.AxisRightOne{l1, o1} Shape.AxisLeftOne{o2, r2}:
      Equal.trans(Nat, output, 1n, other, Equal.trans(Nat, output, left, 1n, l1, Equal.sym(Nat, 1n, left, o2)), Equal.trans(Nat, 1n, right, other, o1, Equal.sym(Nat, other, right, r2)))
    case Shape.AxisRightOne{l1, o1} Shape.AxisRightOne{l2, o2}:
      Equal.trans(Nat, output, left, other, l1, Equal.sym(Nat, other, left, l2))

def aligned_unique(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +other: List<&2, Nat>, +first: Shape.Aligned(left, right, output), +second: Shape.Aligned(left, right, other)) -> {output == other : List<&2, Nat>}:
  match left right output:
    case Nil{} Nil{} Nil{}:
      match other:
        case Nil{}:
          {==}
        case oh <> ot:
          match second:
    case +lh <> +lt +rh <> +rt +oh <> +ot:
      match other:
        case Nil{}:
          match second:
        case +qh <> +qt:
          Shape.AStep{+a1, +t1} = first
          Shape.AStep{+a2, +t2} = second
          %axis_unique(lh, rh, oh, qh, a1, a2) : {oh <> ot == _ <> qt : List<&2, Nat>}
          %aligned_unique(lt, rt, ot, qt, t1, t2) : {oh <> ot == oh <> _ : List<&2, Nat>}
          {==}
    case Nil{} Nil{} oh <> ot:
      match first:
    case Nil{} rh <> rt Nil{}:
      match first:
    case lh <> lt Nil{} Nil{}:
      match first:
    case Nil{} rh <> rt oh <> ot:
      match first:
    case lh <> lt Nil{} oh <> ot:
      match first:
    case lh <> lt rh <> rt Nil{}:
      match first:

def retype(+lp1: Nat, +lp2: Nat, +rp1: Nat, +rp2: Nat, -left: List<&2, Nat>, -right: List<&2, Nat>, -other: List<&2, Nat>, same_left: {lp1 == lp2 : Nat}, same_right: {rp1 == rp2 : Nat}, aligned: Shape.Aligned(Shape.pad(lp2, left), Shape.pad(rp2, right), other)) -> Shape.Aligned(Shape.pad(lp1, left), Shape.pad(rp1, right), other):
  %Equal.sym(Nat, lp1, lp2, same_left) : Shape.Aligned(Shape.pad(_, left), Shape.pad(rp1, right), other)
  %Equal.sym(Nat, rp1, rp2, same_right) : Shape.Aligned(Shape.pad(lp2, left), Shape.pad(_, right), other)
  aligned

def broadcast_unique(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +other: List<&2, Nat>, +first: Shape.Broadcast<left, right, output>, +second: Shape.Broadcast<left, right, other>) -> {output == other : List<&2, Nat>}:
  match first:
    case Shape.BAlign{+lp1, +rp1, +m1, +a1}:
      match second:
        case Shape.BAlign{+lp2, +rp2, +m2, +a2}:
          +left_rank = List.length(&2, Nat, left)
          +right_rank = List.length(&2, Nat, right)
          +e1 = balance(lp1, rp1, left, right, output, a1)
          +e2 = balance(lp2, rp2, left, right, other, a2)
          +same_left = left_padding_equal(lp1, rp1, lp2, rp2, left_rank, right_rank, m1, m2, e1, e2)
          +same_right = cancel_right(rp1, rp2, right_rank, Equal.trans(Nat, Nat.add(rp1, right_rank), Nat.add(lp1, left_rank), Nat.add(rp2, right_rank), Equal.sym(Nat, Nat.add(lp1, left_rank), Nat.add(rp1, right_rank), e1), Equal.trans(Nat, Nat.add(lp1, left_rank), Nat.add(lp2, left_rank), Nat.add(rp2, right_rank), Equal.cong(Nat, Nat, value => Nat.add(value, left_rank), lp1, lp2, same_left), e2)))
          aligned_unique(Shape.pad(lp1, left), Shape.pad(rp1, right), output, other, a1, retype(lp1, lp2, rp1, rp2, left, right, other, same_left, same_right, a2))
