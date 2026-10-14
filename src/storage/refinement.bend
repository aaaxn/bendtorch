import Base
import ./model.bend as Model
import ./array.bend as Core
import ./size.bend as Size

def trace(d: Nat, +n: U32, +i: U32, z: Bool) -> Model.Path(d):
  match d z:
    case 0n _:
      Unit{}
    case 1n+p True{}:
      Model.Step{False{}, trace(p, U32.shr(n), i,
        U32.is_lt(i, U32.shr(U32.shr(n))))}
    case 1n+p False{}:
      +j = U32.sub(i, U32.shr(n))
      Model.Step{True{}, trace(p, U32.shr(n), j,
        U32.is_lt(j, U32.shr(U32.shr(n))))}

def first(-T: Type, pair: Array<T> & T) -> Array<T>:
  (array, value) = pair
  array

def second(-T: Type, pair: Array<T> & T) -> T:
  (array, value) = pair
  value

type Get<-T: Data, -d: Nat, -original: Array<T>, -n: U32, -i: U32, -z: Bool> is Type:
  Get{array: Array<T>, value: T,
    equal: {(array, value) == Array.get.go(T, original, n, i, z) : Array<T> & T},
    kept: {array == original : Array<T>},
    observed: {Some{value} == Model.observe(T, d, original, trace(d, n, i, z)) : Maybe<&2, T>},
    balanced: Model.Balanced(T, d, array)}

def get_low(-T: Data, -p: Nat, -original: Array<T>, -n: U32, -i: U32,
  ys: Array<T>,
  child: Get<T, p, original, U32.shr(n), i, U32.is_lt(i, U32.shr(U32.shr(n)))>,
  ey: Model.Balanced(T, p, ys)) -> Get<T, 1n+p, ANode{original, ys}, n, i, True{}>:
  match child:
    case Get{xs, value, equal, kept, observed, ex}:
      Get{ANode{xs, ys}, value,
        Equal.cong(Array<T> & T, Array<T> & T, r => Array.swap.lo(T, ys, r),
          (xs, value), Array.get.go(T, original, U32.shr(n), i,
            U32.is_lt(i, U32.shr(U32.shr(n)))), equal),
        Equal.cong(Array<T>, Array<T>, x => ANode{x, ys}, xs, original, kept),
        observed, (ex, ey)}

def get_high(-T: Data, -p: Nat, -original: Array<T>, -n: U32, -i: U32,
  xs: Array<T>,
  child: Get<T, p, original, U32.shr(n), U32.sub(i, U32.shr(n)),
    U32.is_lt(U32.sub(i, U32.shr(n)), U32.shr(U32.shr(n)))>,
  ex: Model.Balanced(T, p, xs)) -> Get<T, 1n+p, ANode{xs, original}, n, i, False{}>:
  match child:
    case Get{ys, value, equal, kept, observed, ey}:
      Get{ANode{xs, ys}, value,
        Equal.cong(Array<T> & T, Array<T> & T, r => Array.swap.hi(T, xs, r),
          (ys, value), Array.get.go(T, original, U32.shr(n), U32.sub(i, U32.shr(n)),
            U32.is_lt(U32.sub(i, U32.shr(n)), U32.shr(U32.shr(n)))), equal),
        Equal.cong(Array<T>, Array<T>, y => ANode{xs, y}, ys, original, kept),
        observed, (ex, ey)}

def get_go(-T: Data, d: Nat, array: Array<T>, +n: U32, +i: U32, z: Bool,
  valid: Model.Balanced(T, d, array)) -> Get<T, d, array, n, i, z>:
  match d array z:
    case 0n ALeaf{+value} _:
      Get{ALeaf{value}, value, {==}, {==}, {==}, valid}
    case 0n ANode{xs, ys} _:
      match valid:
    case 1n+p ALeaf{value} _:
      match valid:
    case 1n+p ANode{xs, ys} True{}:
      (ex, ey) = valid
      get_low(T, p, xs, n, i, ys,
        get_go(T, p, xs, U32.shr(n), i, U32.is_lt(i, U32.shr(U32.shr(n))), ex), ey)
    case 1n+p ANode{xs, ys} False{}:
      (ex, ey) = valid
      +j = U32.sub(i, U32.shr(n))
      get_high(T, p, ys, n, i, xs,
        get_go(T, p, ys, U32.shr(n), j, U32.is_lt(j, U32.shr(U32.shr(n))), ey), ex)

def get_go_retained(-T: Data, -d: Nat, -original: Array<T>, -n: U32, -i: U32, -z: Bool,
  cert: Get<T, d, original, n, i, z>) ->
  {first(T, Array.get.go(T, original, n, i, z)) == original : Array<T>}:
  match cert:
    case Get{array, value, equal, kept, observed, valid}:
      %equal : {first(T, _) == original : Array<T>}
      kept

def get_go_observed(-T: Data, -d: Nat, -original: Array<T>, -n: U32, -i: U32, -z: Bool,
  cert: Get<T, d, original, n, i, z>) ->
  {Some{second(T, Array.get.go(T, original, n, i, z))} ==
    Model.observe(T, d, original, trace(d, n, i, z)) : Maybe<&2, T>}:
  match cert:
    case Get{array, value, equal, kept, observed, valid}:
      %equal : {Some{second(T, _)} == Model.observe(T, d, original, trace(d, n, i, z)) : Maybe<&2, T>}
      observed

def get_go_balanced(-T: Data, -d: Nat, -original: Array<T>, -n: U32, -i: U32, -z: Bool,
  cert: Get<T, d, original, n, i, z>) ->
  Model.Balanced(T, d, first(T, Array.get.go(T, original, n, i, z))):
  match cert:
    case Get{array, value, equal, kept, observed, valid}:
      %equal : Model.Balanced(T, d, first(T, _))
      valid

def law_get_go_retains(-T: Data, d: Nat, array: Array<T>, +n: U32, +i: U32, z: Bool,
  valid: Model.Balanced(T, d, array)) ->
  {first(T, Array.get.go(T, array, n, i, z)) == array : Array<T>}:
  get_go_retained(T, d, array, n, i, z, get_go(T, d, array, n, i, z, valid))

def law_get_go_refines(-T: Data, d: Nat, array: Array<T>, +n: U32, +i: U32, z: Bool,
  valid: Model.Balanced(T, d, array)) ->
  {Some{second(T, Array.get.go(T, array, n, i, z))} ==
    Model.observe(T, d, array, trace(d, n, i, z)) : Maybe<&2, T>}:
  get_go_observed(T, d, array, n, i, z, get_go(T, d, array, n, i, z, valid))

def law_get_go_balanced(-T: Data, d: Nat, array: Array<T>, +n: U32, +i: U32, z: Bool,
  valid: Model.Balanced(T, d, array)) ->
  Model.Balanced(T, d, first(T, Array.get.go(T, array, n, i, z))):
  get_go_balanced(T, d, array, n, i, z, get_go(T, d, array, n, i, z, valid))

type Swap<-T: Data, -d: Nat, -original: Array<T>, -n: U32, -i: U32, -z: Bool, -new: T> is Type:
  Swap{array: Array<T>, old: T,
    equal: {(array, old) == Array.swap.go(T, original, n, i, new, z) : Array<T> & T},
    written: {array == Core.write_path(T, d, original, trace(d, n, i, z), new) : Array<T>},
    observed: {Some{old} == Model.observe(T, d, original, trace(d, n, i, z)) : Maybe<&2, T>},
    balanced: Model.Balanced(T, d, array)}

def swap_low(-T: Data, -p: Nat, -original: Array<T>, -n: U32, -i: U32, -new: T,
  ys: Array<T>,
  child: Swap<T, p, original, U32.shr(n), i, U32.is_lt(i, U32.shr(U32.shr(n))), new>,
  ey: Model.Balanced(T, p, ys)) -> Swap<T, 1n+p, ANode{original, ys}, n, i, True{}, new>:
  match child:
    case Swap{xs, old, equal, written, observed, ex}:
      Swap{ANode{xs, ys}, old,
        Equal.cong(Array<T> & T, Array<T> & T, r => Array.swap.lo(T, ys, r),
          (xs, old), Array.swap.go(T, original, U32.shr(n), i, new,
            U32.is_lt(i, U32.shr(U32.shr(n)))), equal),
        Equal.cong(Array<T>, Array<T>, x => ANode{x, ys}, xs,
          Core.write_path(T, p, original,
            trace(p, U32.shr(n), i, U32.is_lt(i, U32.shr(U32.shr(n)))), new), written),
        observed, (ex, ey)}

def swap_high(-T: Data, -p: Nat, -original: Array<T>, -n: U32, -i: U32, -new: T,
  xs: Array<T>,
  child: Swap<T, p, original, U32.shr(n), U32.sub(i, U32.shr(n)),
    U32.is_lt(U32.sub(i, U32.shr(n)), U32.shr(U32.shr(n))), new>,
  ex: Model.Balanced(T, p, xs)) -> Swap<T, 1n+p, ANode{xs, original}, n, i, False{}, new>:
  match child:
    case Swap{ys, old, equal, written, observed, ey}:
      Swap{ANode{xs, ys}, old,
        Equal.cong(Array<T> & T, Array<T> & T, r => Array.swap.hi(T, xs, r),
          (ys, old), Array.swap.go(T, original, U32.shr(n), U32.sub(i, U32.shr(n)), new,
            U32.is_lt(U32.sub(i, U32.shr(n)), U32.shr(U32.shr(n)))), equal),
        Equal.cong(Array<T>, Array<T>, y => ANode{xs, y}, ys,
          Core.write_path(T, p, original,
            trace(p, U32.shr(n), U32.sub(i, U32.shr(n)),
              U32.is_lt(U32.sub(i, U32.shr(n)), U32.shr(U32.shr(n)))), new), written),
        observed, (ex, ey)}

def swap_go(-T: Data, d: Nat, array: Array<T>, +n: U32, +i: U32, new: T, z: Bool,
  valid: Model.Balanced(T, d, array)) -> Swap<T, d, array, n, i, z, new>:
  match d array z:
    case 0n ALeaf{old} _:
      Swap{ALeaf{new}, old, {==}, {==}, {==}, valid}
    case 0n ANode{xs, ys} _:
      match valid:
    case 1n+p ALeaf{old} _:
      match valid:
    case 1n+p ANode{xs, ys} True{}:
      (ex, ey) = valid
      swap_low(T, p, xs, n, i, new, ys,
        swap_go(T, p, xs, U32.shr(n), i, new,
          U32.is_lt(i, U32.shr(U32.shr(n))), ex), ey)
    case 1n+p ANode{xs, ys} False{}:
      (ex, ey) = valid
      +j = U32.sub(i, U32.shr(n))
      swap_high(T, p, ys, n, i, new, xs,
        swap_go(T, p, ys, U32.shr(n), j, new,
          U32.is_lt(j, U32.shr(U32.shr(n))), ey), ex)

def swap_go_written(-T: Data, -d: Nat, -original: Array<T>, -n: U32, -i: U32,
  -z: Bool, -new: T, cert: Swap<T, d, original, n, i, z, new>) ->
  {first(T, Array.swap.go(T, original, n, i, new, z)) ==
    Core.write_path(T, d, original, trace(d, n, i, z), new) : Array<T>}:
  match cert:
    case Swap{array, old, equal, written, observed, valid}:
      %equal : {first(T, _) == Core.write_path(T, d, original, trace(d, n, i, z), new) : Array<T>}
      written

def swap_go_observed(-T: Data, -d: Nat, -original: Array<T>, -n: U32, -i: U32,
  -z: Bool, -new: T, cert: Swap<T, d, original, n, i, z, new>) ->
  {Some{second(T, Array.swap.go(T, original, n, i, new, z))} ==
    Model.observe(T, d, original, trace(d, n, i, z)) : Maybe<&2, T>}:
  match cert:
    case Swap{array, old, equal, written, observed, valid}:
      %equal : {Some{second(T, _)} == Model.observe(T, d, original, trace(d, n, i, z)) : Maybe<&2, T>}
      observed

def swap_go_balanced(-T: Data, -d: Nat, -original: Array<T>, -n: U32, -i: U32,
  -z: Bool, -new: T, cert: Swap<T, d, original, n, i, z, new>) ->
  Model.Balanced(T, d, first(T, Array.swap.go(T, original, n, i, new, z))):
  match cert:
    case Swap{array, old, equal, written, observed, valid}:
      %equal : Model.Balanced(T, d, first(T, _))
      valid

def law_swap_go_refines(-T: Data, d: Nat, array: Array<T>, +n: U32, +i: U32,
  new: T, z: Bool, valid: Model.Balanced(T, d, array)) ->
  {first(T, Array.swap.go(T, array, n, i, new, z)) ==
    Core.write_path(T, d, array, trace(d, n, i, z), new) : Array<T>}:
  swap_go_written(T, d, array, n, i, z, new, swap_go(T, d, array, n, i, new, z, valid))

def law_swap_go_balanced(-T: Data, d: Nat, array: Array<T>, +n: U32, +i: U32,
  new: T, z: Bool, valid: Model.Balanced(T, d, array)) ->
  Model.Balanced(T, d, first(T, Array.swap.go(T, array, n, i, new, z))):
  swap_go_balanced(T, d, array, n, i, z, new, swap_go(T, d, array, n, i, new, z, valid))

def law_swap_go_observed(-T: Data, d: Nat, array: Array<T>, +n: U32, +i: U32,
  new: T, z: Bool, valid: Model.Balanced(T, d, array)) ->
  {Some{second(T, Array.swap.go(T, array, n, i, new, z))} ==
    Model.observe(T, d, array, trace(d, n, i, z)) : Maybe<&2, T>}:
  swap_go_observed(T, d, array, n, i, z, new, swap_go(T, d, array, n, i, new, z, valid))

def masked(+n: U32, i: U32) -> U32:
  U32.and(i, U32.sub(n, 1))

def route_n(d: Nat, +n: U32, i: U32) -> Model.Path(d):
  +j = masked(n, i)
  trace(d, n, j, U32.is_lt(j, U32.shr(n)))

def route(+d: Nat, i: U32) -> Model.Path(d):
  route_n(d, Size.capacity(d), i)

def observed_transport(-T: Data, -d: Nat, -array: Array<T>, -original: Array<T>,
  -n: U32, -i: U32, -value: T,
  kept: {array == original : Array<T>}, extent: {n == Size.capacity(d) : U32},
  observed: {Some{value} == Model.observe(T, d, array, route_n(d, n, i)) : Maybe<&2, T>}) ->
  {Some{value} == Model.observe(T, d, original, route(d, i)) : Maybe<&2, T>}:
  %kept : {Some{value} == Model.observe(T, d, _, route(d, i)) : Maybe<&2, T>}
  %extent : {Some{value} == Model.observe(T, d, array, route_n(d, _, i)) : Maybe<&2, T>}
  observed

type GetFull<-T: Data, -d: Nat, -original: Array<T>, -i: U32> is Type:
  GetFull{array: Array<T>, value: T,
    equal: {(array, value) == Array.get(T, original, i) : Array<T> & T},
    kept: {array == original : Array<T>},
    observed: {Some{value} == Model.observe(T, d, original, route(d, i)) : Maybe<&2, T>},
    balanced: Model.Balanced(T, d, array)}

def get_final(-T: Data, -d: Nat, -original: Array<T>, -sized_array: Array<T>, -n: U32, -i: U32,
  measured: {(sized_array, n) == Array.size(T, original) : Array<T> & U32},
  +retained: {sized_array == original : Array<T>}, extent: {n == Size.capacity(d) : U32},
  child: Get<T, d, sized_array, n, masked(n, i), U32.is_lt(masked(n, i), U32.shr(n))>) ->
  GetFull<T, d, original, i>:
  match child:
    case Get{array, value, equal, kept, observed, valid}:
      GetFull{array, value,
        Equal.trans(Array<T> & T, (array, value), Array.get.at(T, i, (sized_array, n)),
          Array.get(T, original, i), equal,
          Equal.cong(Array<T> & U32, Array<T> & T, r => Array.get.at(T, i, r),
            (sized_array, n), Array.size(T, original), measured)),
        Equal.trans(Array<T>, array, sized_array, original, kept, retained),
        observed_transport(T, d, sized_array, original, n, i, value, retained, extent, observed),
        valid}

def get_sized(-T: Data, +d: Nat, -original: Array<T>, +i: U32,
  sized: Size.Sized<T, d, original>) -> GetFull<T, d, original, i>:
  match sized:
    case Size.Sized{array, +n, retained, measured, extent, valid}:
      get_final(T, d, original, array, n, i, measured, retained, extent,
        get_go(T, d, array, n, masked(n, i), U32.is_lt(masked(n, i), U32.shr(n)), valid))

def get(-T: Data, +d: Nat, array: Array<T>, +i: U32,
  valid: Model.Balanced(T, d, array)) -> GetFull<T, d, array, i>:
  get_sized(T, d, array, i, Size.certify(T, d, array, valid))

def get_retained(-T: Data, -d: Nat, -original: Array<T>, -i: U32,
  cert: GetFull<T, d, original, i>) -> {first(T, Array.get(T, original, i)) == original : Array<T>}:
  match cert:
    case GetFull{array, value, equal, kept, observed, valid}:
      %equal : {first(T, _) == original : Array<T>}
      kept

def get_observed(-T: Data, -d: Nat, -original: Array<T>, -i: U32,
  cert: GetFull<T, d, original, i>) ->
  {Some{second(T, Array.get(T, original, i))} ==
    Model.observe(T, d, original, route(d, i)) : Maybe<&2, T>}:
  match cert:
    case GetFull{array, value, equal, kept, observed, valid}:
      %equal : {Some{second(T, _)} == Model.observe(T, d, original, route(d, i)) : Maybe<&2, T>}
      observed

def get_balanced(-T: Data, -d: Nat, -original: Array<T>, -i: U32,
  cert: GetFull<T, d, original, i>) -> Model.Balanced(T, d, first(T, Array.get(T, original, i))):
  match cert:
    case GetFull{array, value, equal, kept, observed, valid}:
      %equal : Model.Balanced(T, d, first(T, _))
      valid

def law_get_retains(-T: Data, +d: Nat, array: Array<T>, +i: U32,
  valid: Model.Balanced(T, d, array)) -> {first(T, Array.get(T, array, i)) == array : Array<T>}:
  get_retained(T, d, array, i, get(T, d, array, i, valid))

def law_get_refines(-T: Data, +d: Nat, array: Array<T>, +i: U32,
  valid: Model.Balanced(T, d, array)) ->
  {Some{second(T, Array.get(T, array, i))} == Model.observe(T, d, array, route(d, i)) : Maybe<&2, T>}:
  get_observed(T, d, array, i, get(T, d, array, i, valid))

def law_get_balanced(-T: Data, +d: Nat, array: Array<T>, +i: U32,
  valid: Model.Balanced(T, d, array)) -> Model.Balanced(T, d, first(T, Array.get(T, array, i))):
  get_balanced(T, d, array, i, get(T, d, array, i, valid))

def written_transport(-T: Data, -d: Nat, -array: Array<T>, -original: Array<T>,
  -n: U32, -i: U32, -new: T, -result: Array<T>,
  kept: {array == original : Array<T>}, extent: {n == Size.capacity(d) : U32},
  written: {result == Core.write_path(T, d, array, route_n(d, n, i), new) : Array<T>}) ->
  {result == Core.write_path(T, d, original, route(d, i), new) : Array<T>}:
  %kept : {result == Core.write_path(T, d, _, route(d, i), new) : Array<T>}
  %extent : {result == Core.write_path(T, d, array, route_n(d, _, i), new) : Array<T>}
  written

type SwapFull<-T: Data, -d: Nat, -original: Array<T>, -i: U32, -new: T> is Type:
  SwapFull{array: Array<T>, old: T,
    equal: {(array, old) == Array.swap(T, original, i, new) : Array<T> & T},
    written: {array == Core.write_path(T, d, original, route(d, i), new) : Array<T>},
    observed: {Some{old} == Model.observe(T, d, original, route(d, i)) : Maybe<&2, T>},
    balanced: Model.Balanced(T, d, array)}

def swap_final(-T: Data, -d: Nat, -original: Array<T>, -sized_array: Array<T>, -n: U32,
  -i: U32, -new: T,
  measured: {(sized_array, n) == Array.size(T, original) : Array<T> & U32},
  +retained: {sized_array == original : Array<T>}, +extent: {n == Size.capacity(d) : U32},
  child: Swap<T, d, sized_array, n, masked(n, i), U32.is_lt(masked(n, i), U32.shr(n)), new>) ->
  SwapFull<T, d, original, i, new>:
  match child:
    case Swap{array, old, equal, written, observed, valid}:
      SwapFull{array, old,
        Equal.trans(Array<T> & T, (array, old), Array.swap.at(T, i, new, (sized_array, n)),
          Array.swap(T, original, i, new), equal,
          Equal.cong(Array<T> & U32, Array<T> & T, r => Array.swap.at(T, i, new, r),
            (sized_array, n), Array.size(T, original), measured)),
        written_transport(T, d, sized_array, original, n, i, new, array, retained, extent, written),
        observed_transport(T, d, sized_array, original, n, i, old, retained, extent, observed),
        valid}

def swap_sized(-T: Data, +d: Nat, -original: Array<T>, +i: U32, new: T,
  sized: Size.Sized<T, d, original>) -> SwapFull<T, d, original, i, new>:
  match sized:
    case Size.Sized{array, +n, retained, measured, extent, valid}:
      swap_final(T, d, original, array, n, i, new, measured, retained, extent,
        swap_go(T, d, array, n, masked(n, i), new, U32.is_lt(masked(n, i), U32.shr(n)), valid))

def swap(-T: Data, +d: Nat, array: Array<T>, +i: U32, new: T,
  valid: Model.Balanced(T, d, array)) -> SwapFull<T, d, array, i, new>:
  swap_sized(T, d, array, i, new, Size.certify(T, d, array, valid))

def swap_written(-T: Data, -d: Nat, -original: Array<T>, -i: U32, -new: T,
  cert: SwapFull<T, d, original, i, new>) ->
  {first(T, Array.swap(T, original, i, new)) == Core.write_path(T, d, original, route(d, i), new) : Array<T>}:
  match cert:
    case SwapFull{array, old, equal, written, observed, valid}:
      %equal : {first(T, _) == Core.write_path(T, d, original, route(d, i), new) : Array<T>}
      written

def swap_observed(-T: Data, -d: Nat, -original: Array<T>, -i: U32, -new: T,
  cert: SwapFull<T, d, original, i, new>) ->
  {Some{second(T, Array.swap(T, original, i, new))} ==
    Model.observe(T, d, original, route(d, i)) : Maybe<&2, T>}:
  match cert:
    case SwapFull{array, old, equal, written, observed, valid}:
      %equal : {Some{second(T, _)} == Model.observe(T, d, original, route(d, i)) : Maybe<&2, T>}
      observed

def swap_balanced(-T: Data, -d: Nat, -original: Array<T>, -i: U32, -new: T,
  cert: SwapFull<T, d, original, i, new>) ->
  Model.Balanced(T, d, first(T, Array.swap(T, original, i, new))):
  match cert:
    case SwapFull{array, old, equal, written, observed, valid}:
      %equal : Model.Balanced(T, d, first(T, _))
      valid

def set_written(-T: Data, -d: Nat, -original: Array<T>, -i: U32, -new: T,
  cert: SwapFull<T, d, original, i, new>) ->
  {Array.set(T, original, i, new) == Core.write_path(T, d, original, route(d, i), new) : Array<T>}:
  match cert:
    case SwapFull{array, old, equal, written, observed, valid}:
      %equal : {Array.set.fin(T, _) == Core.write_path(T, d, original, route(d, i), new) : Array<T>}
      written

def set_balanced(-T: Data, -d: Nat, -original: Array<T>, -i: U32, -new: T,
  cert: SwapFull<T, d, original, i, new>) -> Model.Balanced(T, d, Array.set(T, original, i, new)):
  match cert:
    case SwapFull{array, old, equal, written, observed, valid}:
      %equal : Model.Balanced(T, d, Array.set.fin(T, _))
      valid

def law_swap_refines(-T: Data, +d: Nat, array: Array<T>, +i: U32, new: T,
  valid: Model.Balanced(T, d, array)) ->
  {first(T, Array.swap(T, array, i, new)) == Core.write_path(T, d, array, route(d, i), new) : Array<T>}:
  swap_written(T, d, array, i, new, swap(T, d, array, i, new, valid))

def law_swap_observed(-T: Data, +d: Nat, array: Array<T>, +i: U32, new: T,
  valid: Model.Balanced(T, d, array)) ->
  {Some{second(T, Array.swap(T, array, i, new))} == Model.observe(T, d, array, route(d, i)) : Maybe<&2, T>}:
  swap_observed(T, d, array, i, new, swap(T, d, array, i, new, valid))

def law_swap_balanced(-T: Data, +d: Nat, array: Array<T>, +i: U32, new: T,
  valid: Model.Balanced(T, d, array)) -> Model.Balanced(T, d, first(T, Array.swap(T, array, i, new))):
  swap_balanced(T, d, array, i, new, swap(T, d, array, i, new, valid))

def law_set_refines(-T: Data, +d: Nat, array: Array<T>, +i: U32, new: T,
  valid: Model.Balanced(T, d, array)) ->
  {Array.set(T, array, i, new) == Core.write_path(T, d, array, route(d, i), new) : Array<T>}:
  set_written(T, d, array, i, new, swap(T, d, array, i, new, valid))

def law_set_balanced(-T: Data, +d: Nat, array: Array<T>, +i: U32, new: T,
  valid: Model.Balanced(T, d, array)) -> Model.Balanced(T, d, Array.set(T, array, i, new)):
  set_balanced(T, d, array, i, new, swap(T, d, array, i, new, valid))

def law_get_at_refines(-T: Data, d: Nat, array: Array<T>, +n: U32, +i: U32,
  valid: Model.Balanced(T, d, array)) ->
  {Some{second(T, Array.get.at(T, i, (array, n)))} ==
    Model.observe(T, d, array, route_n(d, n, i)) : Maybe<&2, T>}:
  get_go_observed(T, d, array, n, masked(n, i), U32.is_lt(masked(n, i), U32.shr(n)),
    get_go(T, d, array, n, masked(n, i), U32.is_lt(masked(n, i), U32.shr(n)), valid))

def law_get_at_retains(-T: Data, d: Nat, array: Array<T>, +n: U32, +i: U32,
  valid: Model.Balanced(T, d, array)) ->
  {first(T, Array.get.at(T, i, (array, n))) == array : Array<T>}:
  get_go_retained(T, d, array, n, masked(n, i), U32.is_lt(masked(n, i), U32.shr(n)),
    get_go(T, d, array, n, masked(n, i), U32.is_lt(masked(n, i), U32.shr(n)), valid))

def law_get_at_balanced(-T: Data, d: Nat, array: Array<T>, +n: U32, +i: U32,
  valid: Model.Balanced(T, d, array)) -> Model.Balanced(T, d, first(T, Array.get.at(T, i, (array, n)))):
  get_go_balanced(T, d, array, n, masked(n, i), U32.is_lt(masked(n, i), U32.shr(n)),
    get_go(T, d, array, n, masked(n, i), U32.is_lt(masked(n, i), U32.shr(n)), valid))

def law_swap_at_refines(-T: Data, d: Nat, array: Array<T>, +n: U32, +i: U32, new: T,
  valid: Model.Balanced(T, d, array)) ->
  {first(T, Array.swap.at(T, i, new, (array, n))) ==
    Core.write_path(T, d, array, route_n(d, n, i), new) : Array<T>}:
  swap_go_written(T, d, array, n, masked(n, i), U32.is_lt(masked(n, i), U32.shr(n)), new,
    swap_go(T, d, array, n, masked(n, i), new, U32.is_lt(masked(n, i), U32.shr(n)), valid))

def law_swap_at_balanced(-T: Data, d: Nat, array: Array<T>, +n: U32, +i: U32, new: T,
  valid: Model.Balanced(T, d, array)) ->
  Model.Balanced(T, d, first(T, Array.swap.at(T, i, new, (array, n)))):
  swap_go_balanced(T, d, array, n, masked(n, i), U32.is_lt(masked(n, i), U32.shr(n)), new,
    swap_go(T, d, array, n, masked(n, i), new, U32.is_lt(masked(n, i), U32.shr(n)), valid))
