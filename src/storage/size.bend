import Base
import ./model.bend as Model

def capacity(d: Nat) -> U32:
  match d:
    case 0n:
      1
    case 1n+p:
      U32.shl(capacity(p))

type Sized<-T: Data, -d: Nat, -original: Array<T>> is Type:
  Sized{array: Array<T>, size: U32,
    retained: {array == original : Array<T>},
    measured: {(array, size) == Array.size(T, original) : Array<T> & U32},
    extent: {size == capacity(d) : U32},
    balanced: Model.Balanced(T, d, array)}

def join(-T: Data, -d: Nat, -original: Array<T>, ys: Array<T>,
  child: Sized<T, d, original>, valid: Model.Balanced(T, d, ys)) ->
  Sized<T, 1n+d, ANode{original, ys}>:
  match child:
    case Sized{xs, n, retained, measured, extent, balanced}:
      Sized{ANode{xs, ys}, U32.shl(n),
        Equal.cong(Array<T>, Array<T>, x => ANode{x, ys}, xs, original, retained),
        Equal.cong(Array<T> & U32, Array<T> & U32, r => Array.size.node(T, ys, r),
          (xs, n), Array.size(T, original), measured),
        Equal.cong(U32, U32, x => U32.shl(x), n, capacity(d), extent), (balanced, valid)}

def certify(-T: Data, +d: Nat, array: Array<T>, valid: Model.Balanced(T, d, array)) -> Sized<T, d, array>:
  match d array:
    case 0n ALeaf{x}:
      Sized{ALeaf{x}, 1, {==}, {==}, {==}, valid}
    case 0n ANode{xs, ys}:
      match valid:
    case 1n+p ALeaf{x}:
      match valid:
    case 1n++p ANode{xs, ys}:
      (ex, ey) = valid
      join(T, p, xs, ys, certify(T, p, xs, ex), ey)

def retains(-T: Data, -d: Nat, -original: Array<T>, sized: Sized<T, d, original>) ->
  {Pair.fst(Array<T>, U32, Array.size(T, original)) == original : Array<T>}:
  match sized:
    case Sized{array, n, retained, measured, extent, balanced}:
      Equal.trans(Array<T>, Pair.fst(Array<T>, U32, Array.size(T, original)), array, original,
        Equal.cong(Array<T> & U32, Array<T>, r => Pair.fst(Array<T>, U32, r),
          Array.size(T, original), (array, n),
          Equal.sym(Array<T> & U32, (array, n), Array.size(T, original), measured)), retained)

def extent(-T: Data, -d: Nat, -original: Array<T>, sized: Sized<T, d, original>) ->
  {Pair.snd(Array<T>, U32, Array.size(T, original)) == capacity(d) : U32}:
  match sized:
    case Sized{array, n, retained, measured, extent, balanced}:
      Equal.trans(U32, Pair.snd(Array<T>, U32, Array.size(T, original)), n, capacity(d),
        Equal.cong(Array<T> & U32, U32, r => Pair.snd(Array<T>, U32, r),
          Array.size(T, original), (array, n),
          Equal.sym(Array<T> & U32, (array, n), Array.size(T, original), measured)), extent)
