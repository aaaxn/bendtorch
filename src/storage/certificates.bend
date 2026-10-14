import Base
import ./model.bend as Model
import ./refinement.bend as Refinement
import ./clone.bend as Clone

def get_result(-T: Data, +d: Nat, original: Array<T>, +i: U32, -result: Array<T> & T,
  equal: {result == Array.get(T, original, i) : Array<T> & T},
  valid: Model.Balanced(T, d, original)) -> Model.Balanced(T, d, Refinement.first(T, result)):
  %Equal.sym(Array<T> & T, result, Array.get(T, original, i), equal) :
    Model.Balanced(T, d, Refinement.first(T, _))
  Refinement.law_get_balanced(T, d, original, i, valid)

def set_result(-T: Data, +d: Nat, original: Array<T>, +i: U32, +value: T, -result: Array<T>,
  equal: {result == Array.set(T, original, i, value) : Array<T>},
  valid: Model.Balanced(T, d, original)) -> Model.Balanced(T, d, result):
  %Equal.sym(Array<T>, result, Array.set(T, original, i, value), equal) : Model.Balanced(T, d, _)
  Refinement.law_set_balanced(T, d, original, i, value, valid)

def clone_first(-T: Data, +d: Nat, original: Array<T>, -result: Array<T> & Array<T>,
  equal: {result == Array.clone(T, original) : Array<T> & Array<T>},
  valid: Model.Balanced(T, d, original)) -> Model.Balanced(T, d, Pair.fst(Array<T>, Array<T>, result)):
  %Equal.sym(Array<T> & Array<T>, result, Array.clone(T, original), equal) :
    Model.Balanced(T, d, Pair.fst(Array<T>, Array<T>, _))
  Clone.first_balanced(T, d, original, Clone.certify(T, d, original, valid))

def clone_second(-T: Data, +d: Nat, original: Array<T>, -result: Array<T> & Array<T>,
  equal: {result == Array.clone(T, original) : Array<T> & Array<T>},
  valid: Model.Balanced(T, d, original)) -> Model.Balanced(T, d, Pair.snd(Array<T>, Array<T>, result)):
  %Equal.sym(Array<T> & Array<T>, result, Array.clone(T, original), equal) :
    Model.Balanced(T, d, Pair.snd(Array<T>, Array<T>, _))
  Clone.second_balanced(T, d, original, Clone.certify(T, d, original, valid))
