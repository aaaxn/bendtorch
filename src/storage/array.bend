import Base
import ./model.bend as Model

def left(-T: Type, -d: Nat, -xs: Array<T>, -ys: Array<T>,
  e: Model.Balanced(T, d, xs) & Model.Balanced(T, d, ys)) -> Model.Balanced(T, d, xs):
  (ex, ey) = e
  ex

def right(-T: Type, -d: Nat, -xs: Array<T>, -ys: Array<T>,
  e: Model.Balanced(T, d, xs) & Model.Balanced(T, d, ys)) -> Model.Balanced(T, d, ys):
  (ex, ey) = e
  ey

def new_balanced(-T: Data, +d: Nat, +value: T) -> Model.Balanced(T, d, Array.new(T, d, value)):
  match d:
    case 0n:
      Unit{}
    case 1n++p:
      (new_balanced(T, p, value), new_balanced(T, p, value))

type Read<-T: Data, -d: Nat, -original: Array<T>, -path: Model.Path(d)> is Type:
  Read{array: Array<T>, value: Maybe<&2, T>, -balanced: Model.Balanced(T, d, array),
    retained: {array == original : Array<T>},
    observed: {value == Model.observe(T, d, original, path) : Maybe<&2, T>}}

def low(-T: Data, -d: Nat, -original: Array<T>, -path: Model.Path(d), ys: Array<T>,
  read: Read<T, d, original, path>, -ey: Model.Balanced(T, d, ys)) ->
  Read<T, 1n+d, ANode{original, ys}, Model.Step{False{}, path}>:
  match read:
    case Read{xs, value, ex, kept, observed}:
      Read{ANode{xs, ys}, value, (ex, ey),
        Equal.cong(Array<T>, Array<T>, x => ANode{x, ys}, xs, original, kept), observed}

def high(-T: Data, -d: Nat, -original: Array<T>, -path: Model.Path(d), xs: Array<T>,
  read: Read<T, d, original, path>, -ex: Model.Balanced(T, d, xs)) ->
  Read<T, 1n+d, ANode{xs, original}, Model.Step{True{}, path}>:
  match read:
    case Read{ys, value, ey, kept, observed}:
      Read{ANode{xs, ys}, value, (ex, ey),
        Equal.cong(Array<T>, Array<T>, y => ANode{xs, y}, ys, original, kept), observed}

def read_path(-T: Data, +d: Nat, array: Array<T>, path: Model.Path(d),
  -valid: Model.Balanced(T, d, array)) -> Read<T, d, array, path>:
  match d array:
    case 0n ALeaf{+value}:
      Read{ALeaf{value}, Some{value}, valid, {==}, {==}}
    case 0n ANode{xs, ys}:
      Read{ANode{xs, ys}, None{}, valid, {==}, {==}}
    case 1n+p ALeaf{value}:
      Read{ALeaf{value}, None{}, valid, {==}, {==}}
    case 1n++p ANode{xs, ys}:
      Model.Step{bit, rest} = path
      match bit:
        case False{}:
          low(T, p, xs, rest, ys, read_path(T, p, xs, rest, left(T, p, xs, ys, valid)),
            right(T, p, xs, ys, valid))
        case True{}:
          high(T, p, ys, rest, xs, read_path(T, p, ys, rest, right(T, p, xs, ys, valid)),
            left(T, p, xs, ys, valid))

def write_path(-T: Data, d: Nat, array: Array<T>, path: Model.Path(d), value: T) -> Array<T>:
  match d array:
    case 0n ALeaf{old}:
      ALeaf{value}
    case 0n ANode{xs, ys}:
      ANode{xs, ys}
    case 1n+p ALeaf{old}:
      ALeaf{old}
    case 1n+p ANode{xs, ys}:
      Model.Step{bit, rest} = path
      match bit:
        case False{}:
          ANode{write_path(T, p, xs, rest, value), ys}
        case True{}:
          ANode{xs, write_path(T, p, ys, rest, value)}

def write_balanced(-T: Data, d: Nat, array: Array<T>, path: Model.Path(d), value: T,
  valid: Model.Balanced(T, d, array)) -> Model.Balanced(T, d, write_path(T, d, array, path, value)):
  match d array:
    case 0n ALeaf{old}:
      valid
    case 0n ANode{xs, ys}:
      valid
    case 1n+p ALeaf{old}:
      valid
    case 1n+p ANode{xs, ys}:
      Model.Step{bit, rest} = path
      match bit:
        case False{}:
          (ex, ey) = valid
          (write_balanced(T, p, xs, rest, value, ex), ey)
        case True{}:
          (ex, ey) = valid
          (ex, write_balanced(T, p, ys, rest, value, ey))

def observation(-T: Data, -d: Nat, -original: Array<T>, -path: Model.Path(d),
  read: Read<T, d, original, path>) -> Maybe<&2, T>:
  match read:
    case Read{array, value, valid, kept, observed}:
      value

def retained(-T: Data, -d: Nat, -original: Array<T>, -path: Model.Path(d),
  read: Read<T, d, original, path>) -> Array<T>:
  match read:
    case Read{array, value, valid, kept, observed}:
      array
