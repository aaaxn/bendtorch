import Base
import ./model.bend as Model
import ./array.bend as ArrayCore
import ./refinement.bend as Refinement
import ./clone.bend as Clone

def write_read(-T: Data, +d: Nat, array: Array<T>, +path: Model.Path(d), value: T, valid: Model.Balanced(T, d, array)) ->
  {Model.observe(T, d, ArrayCore.write_path(T, d, array, path, value), path) == Some{value} : Maybe<&2, T>}:
  match d array:
    case 0n ALeaf{old}:
      {==}
    case 0n ANode{xs, ys}:
      match valid:
    case 1n+p ALeaf{old}:
      match valid:
    case 1n++p ANode{xs, ys}:
      Model.Step{bit, rest} = path
      match bit:
        case False{}:
          (ex, ey) = valid
          write_read(T, p, xs, rest, value, ex)
        case True{}:
          (ex, ey) = valid
          write_read(T, p, ys, rest, value, ey)

def update_frame(-T: Data, +d: Nat, array: Array<T>, +i: Model.Path(d), +j: Model.Path(d), +value: T, valid: Model.Balanced(T, d, array)) ->
  {Model.observe(T, d, ArrayCore.write_path(T, d, array, i, value), j) == Model.select(T, Model.same(d, i, j), value, Model.observe(T, d, array, j)) : Maybe<&2, T>}:
  match d array:
    case 0n ALeaf{old}:
      {==}
    case 0n ANode{xs, ys}:
      match valid:
    case 1n+p ALeaf{old}:
      match valid:
    case 1n++p ANode{xs, ys}:
      Model.Step{ib, ir} = i
      match ib:
        case False{}:
          Model.Step{jb, jr} = j
          match jb:
            case False{}:
              (ex, ey) = valid
              update_frame(T, p, xs, ir, jr, value, ex)
            case True{}:
              {==}
        case True{}:
          Model.Step{jb, jr} = j
          match jb:
            case False{}:
              {==}
            case True{}:
              (ex, ey) = valid
              update_frame(T, p, ys, ir, jr, value, ey)

def observed_correct(-T: Data, -d: Nat, -array: Array<T>, -path: Model.Path(d),
  read: ArrayCore.Read<T, d, array, path>) ->
  {ArrayCore.observation(T, d, array, path, read) == Model.observe(T, d, array, path) : Maybe<&2, T>}:
  match read:
    case ArrayCore.Read{retained, value, balanced, kept, observed}:
      observed

def retained_correct(-T: Data, -d: Nat, -array: Array<T>, -path: Model.Path(d),
  read: ArrayCore.Read<T, d, array, path>) ->
  {ArrayCore.retained(T, d, array, path, read) == array : Array<T>}:
  match read:
    case ArrayCore.Read{retained, value, balanced, kept, observed}:
      kept

def read_refines(-T: Data, +d: Nat, array: Array<T>, +path: Model.Path(d), valid: Model.Balanced(T, d, array)) ->
  {ArrayCore.observation(T, d, array, path, ArrayCore.read_path(T, d, array, path, valid)) == Model.observe(T, d, array, path) : Maybe<&2, T>}:
  observed_correct(T, d, array, path, ArrayCore.read_path(T, d, array, path, valid))

def read_retains(-T: Data, +d: Nat, array: Array<T>, +path: Model.Path(d), valid: Model.Balanced(T, d, array)) ->
  {ArrayCore.retained(T, d, array, path, ArrayCore.read_path(T, d, array, path, valid)) == array : Array<T>}:
  retained_correct(T, d, array, path, ArrayCore.read_path(T, d, array, path, valid))

def primitive_frame_cloned(-T: Data, +d: Nat, -original: Array<T>, +i: U32,
  +other: Model.Path(d), +value: T, cloned: Clone.Cloned<T, d, original>) ->
  {Model.observe(T, d, Array.set(T, original, i, value), other) == Model.select(T, Model.same(d, Refinement.route(d, i), other), value, Model.observe(T, d, original, other)) : Maybe<&2, T>}:
  match cloned:
    case Clone.Cloned{first, second, exact, +same_first, +same_second, valid_first, valid_second}:
      %same_second :
        {Model.observe(T, d, Array.set(T, original, i, value), other) == Model.select(T, Model.same(d, Refinement.route(d, i), other), value, Model.observe(T, d, _, other)) : Maybe<&2, T>}
      %same_first :
        {Model.observe(T, d, Array.set(T, _, i, value), other) == Model.select(T, Model.same(d, Refinement.route(d, i), other), value, Model.observe(T, d, second, other)) : Maybe<&2, T>}
      %Equal.sym(Array<T>, Array.set(T, first, i, value), ArrayCore.write_path(T, d, first, Refinement.route(d, i), value),
        Refinement.law_set_refines(T, d, first, i, value, valid_first)) :
        {Model.observe(T, d, _, other) == Model.select(T, Model.same(d, Refinement.route(d, i), other), value, Model.observe(T, d, second, other)) : Maybe<&2, T>}
      %Equal.trans(Array<T>, second, original, first, same_second,
        Equal.sym(Array<T>, first, original, same_first)) :
        {Model.observe(T, d, ArrayCore.write_path(T, d, _, Refinement.route(d, i), value), other) == Model.select(T, Model.same(d, Refinement.route(d, i), other), value, Model.observe(T, d, second, other)) : Maybe<&2, T>}
      update_frame(T, d, second, Refinement.route(d, i), other, value, valid_second)

def primitive_update_frame(-T: Data, +d: Nat, array: Array<T>, +i: U32, +other: Model.Path(d), +value: T, valid: Model.Balanced(T, d, array)) ->
  {Model.observe(T, d, Array.set(T, array, i, value), other) == Model.select(T, Model.same(d, Refinement.route(d, i), other), value, Model.observe(T, d, array, other)) : Maybe<&2, T>}:
  primitive_frame_cloned(T, d, array, i, other, value, Clone.certify(T, d, array, valid))

def path_self_same(+d: Nat, +path: Model.Path(d)) ->
  {Model.same(d, path, path) == True{} : Bool}:
  match d:
    case 0n:
      {==}
    case 1n+p:
      Model.Step{bit, rest} = path
      match bit:
        case False{}:
          path_self_same(p, rest)
        case True{}:
          path_self_same(p, rest)

def primitive_write_read(-T: Data, +d: Nat, array: Array<T>, +i: U32, +value: T, valid: Model.Balanced(T, d, array)) ->
  {Model.observe(T, d, Array.set(T, array, i, value), Refinement.route(d, i)) == Some{value} : Maybe<&2, T>}:
  %Equal.sym(Maybe<&2, T>,
    Model.observe(T, d, Array.set(T, array, i, value), Refinement.route(d, i)),
    Model.select(T, Model.same(d, Refinement.route(d, i), Refinement.route(d, i)), value,
      Model.observe(T, d, array, Refinement.route(d, i))),
    primitive_update_frame(T, d, array, i, Refinement.route(d, i), value, valid)) :
    {_ == Some{value} : Maybe<&2, T>}
  %Equal.sym(Bool, Model.same(d, Refinement.route(d, i), Refinement.route(d, i)), True{},
    path_self_same(d, Refinement.route(d, i))) :
    {Model.select(T, _, value, Model.observe(T, d, array, Refinement.route(d, i))) == Some{value} : Maybe<&2, T>}
  {==}
