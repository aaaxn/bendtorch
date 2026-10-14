import Base
import ./model.bend as Model

type Cloned<-T: Data, -d: Nat, -original: Array<T>> is Type:
  Cloned{first: Array<T>, second: Array<T>,
    exact: {(first, second) == Array.clone(T, original) : Array<T> & Array<T>},
    first_equal: {first == original : Array<T>},
    second_equal: {second == original : Array<T>},
    first_balanced: Model.Balanced(T, d, first),
    second_balanced: Model.Balanced(T, d, second)}

def node_equal(-T: Data, -a: Array<T>, -b: Array<T>, -x: Array<T>, -y: Array<T>,
  left: {a == x : Array<T>}, right: {b == y : Array<T>}) ->
  {ANode{a, b} == ANode{x, y} : Array<T>}:
  Equal.trans(Array<T>, ANode{a, b}, ANode{x, b}, ANode{x, y},
    Equal.cong(Array<T>, Array<T>, v => ANode{v, b}, a, x, left),
    Equal.cong(Array<T>, Array<T>, v => ANode{x, v}, b, y, right))

def join(-T: Data, -d: Nat, -xs: Array<T>, -ys: Array<T>,
  left: Cloned<T, d, xs>, right: Cloned<T, d, ys>) -> Cloned<T, 1n+d, ANode{xs, ys}>:
  match left:
    case Cloned{xa, xb, exact_x, same_xa, same_xb, valid_xa, valid_xb}:
      match right:
        case Cloned{ya, yb, exact_y, same_ya, same_yb, valid_ya, valid_yb}:
          Cloned{ANode{xa, ya}, ANode{xb, yb},
            Equal.trans(Array<T> & Array<T>, (ANode{xa, ya}, ANode{xb, yb}),
              Array.clone.node(T, Array.clone(T, xs), (ya, yb)),
              Array.clone.node(T, Array.clone(T, xs), Array.clone(T, ys)),
              Equal.cong(Array<T> & Array<T>, Array<T> & Array<T>,
                r => Array.clone.node(T, r, (ya, yb)), (xa, xb), Array.clone(T, xs), exact_x),
              Equal.cong(Array<T> & Array<T>, Array<T> & Array<T>,
                r => Array.clone.node(T, Array.clone(T, xs), r), (ya, yb), Array.clone(T, ys), exact_y)),
            node_equal(T, xa, ya, xs, ys, same_xa, same_ya),
            node_equal(T, xb, yb, xs, ys, same_xb, same_yb),
            (valid_xa, valid_ya), (valid_xb, valid_yb)}

def certify(-T: Data, +d: Nat, array: Array<T>, valid: Model.Balanced(T, d, array)) -> Cloned<T, d, array>:
  match d array:
    case 0n ALeaf{+value}:
      Cloned{ALeaf{value}, ALeaf{value}, {==}, {==}, {==}, Unit{}, Unit{}}
    case 0n ANode{xs, ys}:
      match valid:
    case 1n+p ALeaf{value}:
      match valid:
    case 1n++p ANode{xs, ys}:
      (ex, ey) = valid
      join(T, p, xs, ys, certify(T, p, xs, ex), certify(T, p, ys, ey))

def first_equal(-T: Data, -d: Nat, -original: Array<T>, cloned: Cloned<T, d, original>) ->
  {Pair.fst(Array<T>, Array<T>, Array.clone(T, original)) == original : Array<T>}:
  match cloned:
    case Cloned{first, second, exact, same_first, same_second, valid_first, valid_second}:
      Equal.trans(Array<T>, Pair.fst(Array<T>, Array<T>, Array.clone(T, original)), first, original,
        Equal.cong(Array<T> & Array<T>, Array<T>, r => Pair.fst(Array<T>, Array<T>, r),
          Array.clone(T, original), (first, second),
          Equal.sym(Array<T> & Array<T>, (first, second), Array.clone(T, original), exact)), same_first)

def second_equal(-T: Data, -d: Nat, -original: Array<T>, cloned: Cloned<T, d, original>) ->
  {Pair.snd(Array<T>, Array<T>, Array.clone(T, original)) == original : Array<T>}:
  match cloned:
    case Cloned{first, second, exact, same_first, same_second, valid_first, valid_second}:
      Equal.trans(Array<T>, Pair.snd(Array<T>, Array<T>, Array.clone(T, original)), second, original,
        Equal.cong(Array<T> & Array<T>, Array<T>, r => Pair.snd(Array<T>, Array<T>, r),
          Array.clone(T, original), (first, second),
          Equal.sym(Array<T> & Array<T>, (first, second), Array.clone(T, original), exact)), same_second)

def first_balanced(-T: Data, -d: Nat, -original: Array<T>, cloned: Cloned<T, d, original>) ->
  Model.Balanced(T, d, Pair.fst(Array<T>, Array<T>, Array.clone(T, original))):
  match cloned:
    case Cloned{first, second, exact, same_first, same_second, valid_first, valid_second}:
      %exact : Model.Balanced(T, d, Pair.fst(Array<T>, Array<T>, _))
      valid_first

def second_balanced(-T: Data, -d: Nat, -original: Array<T>, cloned: Cloned<T, d, original>) ->
  Model.Balanced(T, d, Pair.snd(Array<T>, Array<T>, Array.clone(T, original))):
  match cloned:
    case Cloned{first, second, exact, same_first, same_second, valid_first, valid_second}:
      %exact : Model.Balanced(T, d, Pair.snd(Array<T>, Array<T>, _))
      valid_second
