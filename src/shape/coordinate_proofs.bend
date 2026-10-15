import Base
import ./core.bend as Shape
import ./nat_proofs.bend as Nats
import ./index_proofs.bend as Index
import ./broadcast_proofs.bend as Broadcast

type Fits.Step<-dh: Nat, -dt: List<&2, Nat>, -vh: Nat, -vt: List<&2, Nat>> is Data:
  FStep{bound: {True{} == Nat.is_lt(vh, dh) : Bool}, tail: Fits(dt, vt)}

def Fits(dims: List<&2, Nat>, values: List<&2, Nat>) -> Data:
  match dims values:
    case Nil{} Nil{}:
      Unit
    case dh <> dt vh <> vt:
      Fits.Step<dh, dt, vh, vt>
    case Nil{} vh <> vt:
      Empty
    case dh <> dt Nil{}:
      Empty

def fin_from(+bound: Nat, +value: Nat, fits: {True{} == Nat.is_lt(value, bound) : Bool}) -> Shape.Fin(bound):
  match bound value:
    case 0n 0n:
      Empty.absurd(Shape.Fin(0n), Shape.clash(fits))
    case 0n 1n+q:
      Empty.absurd(Shape.Fin(0n), Shape.clash(fits))
    case 1n+p 0n:
      Shape.FZero{}
    case 1n+p 1n+q:
      Shape.FNext{fin_from(p, q, fits)}

def fin_from_value(+bound: Nat, +value: Nat, fits: {True{} == Nat.is_lt(value, bound) : Bool}) -> {Shape.Fin.value(bound, fin_from(bound, value, fits)) == value : Nat}:
  match bound value:
    case 0n 0n:
      Empty.absurd({Shape.Fin.value(0n, fin_from(0n, 0n, fits)) == 0n : Nat}, Shape.clash(fits))
    case 0n 1n+q:
      Empty.absurd({Shape.Fin.value(0n, fin_from(0n, 1n+q, fits)) == 1n+q : Nat}, Shape.clash(fits))
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      Equal.cong(Nat, Nat, value => 1n+value, Shape.Fin.value(p, fin_from(p, q, fits)), q, fin_from_value(p, q, fits))

def coordinate_of(+dims: List<&2, Nat>, +values: List<&2, Nat>, fits: Fits(dims, values)) -> Shape.Coordinate(dims):
  match dims values:
    case Nil{} Nil{}:
      Unit{}
    case +dh <> +dt +vh <> +vt:
      FStep{bound, tail} = fits
      Shape.CStep{fin_from(dh, vh, bound), coordinate_of(dt, vt, tail)}
    case Nil{} vh <> vt:
      match fits:
    case dh <> dt Nil{}:
      match fits:

def indices_of(+dims: List<&2, Nat>, +values: List<&2, Nat>, +fits: Fits(dims, values)) -> {Shape.indices(dims, coordinate_of(dims, values, fits)) == values : List<&2, Nat>}:
  match dims values:
    case Nil{} Nil{}:
      {==}
    case +dh <> +dt +vh <> +vt:
      FStep{+bound, +tail} = fits
      %Equal.sym(Nat, Shape.Fin.value(dh, fin_from(dh, vh, bound)), vh, fin_from_value(dh, vh, bound)) : {_ <> Shape.indices(dt, coordinate_of(dt, vt, tail)) == vh <> vt : List<&2, Nat>}
      %Equal.sym(List<&2, Nat>, Shape.indices(dt, coordinate_of(dt, vt, tail)), vt, indices_of(dt, vt, tail)) : {vh <> _ == vh <> vt : List<&2, Nat>}
      {==}
    case Nil{} vh <> vt:
      match fits:
    case dh <> dt Nil{}:
      match fits:

def offset_indices(+dims: List<&2, Nat>, +coordinate: Shape.Coordinate(dims)) -> {Shape.offset(dims, Shape.indices(dims, coordinate)) == Shape.row_major(dims, coordinate) : Nat}:
  match dims:
    case Nil{}:
      {==}
    case +head <> +tail:
      Shape.CStep{+index, +rest} = coordinate
      Equal.cong(Nat, Nat, value => Nat.add(Nat.mul(Shape.Fin.value(head, index), Shape.numel(tail)), value), Shape.offset(tail, Shape.indices(tail, rest)), Shape.row_major(tail, rest), offset_indices(tail, rest))

def row_major_lt(+dims: List<&2, Nat>, +coordinate: Shape.Coordinate(dims)) -> Shape.LT(Shape.row_major(dims, coordinate), Shape.numel(dims)):
  %Index.flatten_row_major(dims, coordinate) : Shape.LT(_, Shape.numel(dims))
  Shape.Fin.bound(Shape.numel(dims), Shape.flatten(dims, coordinate))

def size_fits(+shape: Shape.Shape, -index: Nat, fits: {True{} == Nat.is_lt(index, Shape.elements(shape)) : Bool}) -> {True{} == Nat.is_lt(index, Shape.numel(Shape.extents(shape))) : Bool}:
  %Shape.elements_exact(shape) : {True{} == Nat.is_lt(index, _) : Bool}
  fits

def size_misses(+shape: Shape.Shape, -index: Nat, misses: {False{} == Nat.is_lt(index, Shape.elements(shape)) : Bool}) -> {False{} == Nat.is_lt(index, Shape.numel(Shape.extents(shape))) : Bool}:
  %Shape.elements_exact(shape) : {False{} == Nat.is_lt(index, _) : Bool}
  misses

def offset_of(+dims: List<&2, Nat>, +values: List<&2, Nat>, +fits: Fits(dims, values)) -> {Shape.row_major(dims, coordinate_of(dims, values, fits)) == Shape.offset(dims, values) : Nat}:
  %indices_of(dims, values, fits) : {Shape.row_major(dims, coordinate_of(dims, values, fits)) == Shape.offset(dims, _) : Nat}
  Equal.sym(Nat, Shape.offset(dims, Shape.indices(dims, coordinate_of(dims, values, fits))), Shape.row_major(dims, coordinate_of(dims, values, fits)), offset_indices(dims, coordinate_of(dims, values, fits)))

def digits_step(+head: Nat, +tail: List<&2, Nat>, rec: @+index: Shape.Fin(Shape.numel(tail)) -> {Shape.indices(tail, Shape.unflatten(tail, index)) == Shape.digits(tail, Shape.Fin.value(Shape.numel(tail), index)) : List<&2, Nat>}, coordinate: Shape.Fin(head) & Shape.Fin(Shape.numel(tail))) -> {Shape.indices(head <> tail, Shape.unflatten.step(head, tail, rest => Shape.unflatten(tail, rest), coordinate)) == Shape.digits(head <> tail, Shape.Fin.value(Nat.mul(head, Shape.numel(tail)), Shape.Fin.pack(head, Shape.numel(tail), coordinate))) : List<&2, Nat>}:
  (+first, +last) = coordinate
  +stride = Shape.numel(tail)
  +major = Shape.Fin.value(head, first)
  +minor = Shape.Fin.value(stride, last)
  %Equal.sym(Nat, Shape.Fin.value(Nat.mul(head, stride), Shape.Fin.pack(head, stride, (first, last))), Nat.add(Nat.mul(major, stride), minor), Index.pack_value(head, stride, first, last)) : {major <> Shape.indices(tail, Shape.unflatten(tail, last)) == Shape.digits(head <> tail, _) : List<&2, Nat>}
  %Nats.divmod_exact(major, stride, minor, Shape.Fin.bound(stride, last)) : {major <> Shape.indices(tail, Shape.unflatten(tail, last)) == Pair.fst(Nat, Nat, _) <> Shape.digits(tail, Pair.snd(Nat, Nat, _)) : List<&2, Nat>}
  %rec(last) : {major <> Shape.indices(tail, Shape.unflatten(tail, last)) == major <> _ : List<&2, Nat>}
  {==}

def digits_unflatten(+dims: List<&2, Nat>, +index: Shape.Fin(Shape.numel(dims))) -> {Shape.indices(dims, Shape.unflatten(dims, index)) == Shape.digits(dims, Shape.Fin.value(Shape.numel(dims), index)) : List<&2, Nat>}:
  match dims:
    case Nil{}:
      match index:
        case Shape.FZero{}:
          {==}
        case Shape.FNext{previous}:
          match previous:
    case +head <> +tail:
      %Index.pack_unpack(head, Shape.numel(tail), index) : {Shape.indices(head <> tail, Shape.unflatten(head <> tail, index)) == Shape.digits(head <> tail, Shape.Fin.value(Shape.numel(head <> tail), _)) : List<&2, Nat>}
      digits_step(head, tail, +index => digits_unflatten(tail, index), Shape.Fin.unpack(head, Shape.numel(tail), index))

def fits_of.step(-dh: Nat, -dt: List<&2, Nat>, -vh: Nat, -vt: List<&2, Nat>, verdict: Shape.BoolCase<Nat.is_lt(vh, dh)>, ok: {True{} == Bool.and(Nat.is_lt(vh, dh), Shape.fit(dt, vt)) : Bool}, rec: {True{} == Shape.fit(dt, vt) : Bool} -> Fits(dt, vt)) -> Fits(dh <> dt, vh <> vt):
  match verdict:
    case Shape.BFalse{equal}:
      Empty.absurd(Fits(dh <> dt, vh <> vt), Shape.clash(Equal.trans(Bool, True{}, Bool.and(Nat.is_lt(vh, dh), Shape.fit(dt, vt)), False{}, ok, Equal.cong(Bool, Bool, head => Bool.and(head, Shape.fit(dt, vt)), Nat.is_lt(vh, dh), False{}, Equal.sym(Bool, False{}, Nat.is_lt(vh, dh), equal)))))
    case Shape.BTrue{+equal}:
      FStep{equal, rec(Equal.trans(Bool, True{}, Bool.and(Nat.is_lt(vh, dh), Shape.fit(dt, vt)), Shape.fit(dt, vt), ok, Equal.cong(Bool, Bool, head => Bool.and(head, Shape.fit(dt, vt)), Nat.is_lt(vh, dh), True{}, Equal.sym(Bool, True{}, Nat.is_lt(vh, dh), equal))))}

def fits_of(+dims: List<&2, Nat>, +values: List<&2, Nat>, ok: {True{} == Shape.fit(dims, values) : Bool}) -> Fits(dims, values):
  match dims values:
    case Nil{} Nil{}:
      Unit{}
    case +dh <> +dt +vh <> +vt:
      fits_of.step(dh, dt, vh, vt, Shape.bool_case(Nat.is_lt(vh, dh)), ok, rest => fits_of(dt, vt, rest))
    case Nil{} vh <> vt:
      Empty.absurd(Fits(Nil{}, vh <> vt), Shape.clash(ok))
    case dh <> dt Nil{}:
      Empty.absurd(Fits(dh <> dt, Nil{}), Shape.clash(ok))

def fit_indices(+dims: List<&2, Nat>, +coordinate: Shape.Coordinate(dims)) -> {True{} == Shape.fit(dims, Shape.indices(dims, coordinate)) : Bool}:
  match dims:
    case Nil{}:
      match coordinate:
        case Unit{}:
          {==}
    case +head <> +tail:
      Shape.CStep{+index, +rest} = coordinate
      %Nats.lt_true(Shape.Fin.value(head, index), head, Shape.Fin.bound(head, index)) : {True{} == Bool.and(_, Shape.fit(tail, Shape.indices(tail, rest))) : Bool}
      fit_indices(tail, rest)

def fit_words_spec(+dims: List<&2, U32>, +values: List<&2, U32>) -> {Shape.fit_words(dims, values) == Shape.fit(Shape.naturals(dims), Shape.naturals(values)) : Bool}:
  match dims values:
    case Nil{} Nil{}:
      {==}
    case +dh <> +dt +vh <> +vt:
      Equal.cong(Bool, Bool, rest => Bool.and(Nat.is_lt(U32.to_nat(vh), U32.to_nat(dh)), rest), Shape.fit_words(dt, vt), Shape.fit(Shape.naturals(dt), Shape.naturals(vt)), fit_words_spec(dt, vt))
    case Nil{} vh <> vt:
      {==}
    case dh <> dt Nil{}:
      {==}

def digits_words_spec(+dims: List<&2, U32>, +index: Nat) -> {Shape.digits_words(dims, index) == Shape.digits(Shape.naturals(dims), index) : List<&2, Nat>}:
  match dims:
    case Nil{}:
      {==}
    case +head <> +tail:
      +stride = Shape.numel(Shape.naturals(tail))
      %Equal.sym(Nat, Shape.numel_words(tail), stride, Shape.numel_words_spec(tail)) : {Nat.div(index, _) <> Shape.digits_words(tail, Nat.mod(index, _)) == Shape.digits(Shape.naturals(head <> tail), index) : List<&2, Nat>}
      Equal.cong(List<&2, Nat>, List<&2, Nat>, rest => Nat.div(index, stride) <> rest, Shape.digits_words(tail, Nat.mod(index, stride)), Shape.digits(Shape.naturals(tail), Nat.mod(index, stride)), digits_words_spec(tail, Nat.mod(index, stride)))

def projected_words_spec(+dims: List<&2, U32>, +values: List<&2, U32>) -> {Shape.projected_words(dims, values) == Shape.broadcast_indices(Shape.naturals(dims), Shape.naturals(values)) : List<&2, Nat>}:
  match dims values:
    case +head <> +tail +value <> +rest:
      Equal.cong(List<&2, Nat>, List<&2, Nat>, rest => Shape.broadcast_index(U32.to_nat(head), U32.to_nat(value)) <> rest, Shape.projected_words(tail, rest), Shape.broadcast_indices(Shape.naturals(tail), Shape.naturals(rest)), projected_words_spec(tail, rest))
    case Nil{} values:
      {==}
    case head <> tail Nil{}:
      {==}

def naturals_drop(+values: List<&2, U32>, +count: Nat) -> {Shape.naturals(List.drop(&2, U32, values, count)) == List.drop(&2, Nat, Shape.naturals(values), count) : List<&2, Nat>}:
  match values count:
    case Nil{} count:
      {==}
    case head <> tail 0n:
      {==}
    case head <> tail 1n+rest:
      naturals_drop(tail, rest)

def horner_spec(+dims: List<&2, U32>, +values: List<&2, U32>, +acc: Nat, +fits: Fits(Shape.naturals(dims), Shape.naturals(values))) -> {Shape.horner(dims, values, acc) == Nat.add(Nat.mul(acc, Shape.numel(Shape.naturals(dims))), Shape.offset(Shape.naturals(dims), Shape.naturals(values))) : Nat}:
  match dims values:
    case Nil{} Nil{}:
      Equal.trans(Nat, acc, Nat.mul(acc, 1n), Nat.add(Nat.mul(acc, 1n), 0n), Equal.sym(Nat, Nat.mul(acc, 1n), acc, Nats.mul_one_right(acc)), Equal.sym(Nat, Nat.add(Nat.mul(acc, 1n), 0n), Nat.mul(acc, 1n), Broadcast.add_zero(Nat.mul(acc, 1n))))
    case +dh <> +dt +vh <> +vt:
      FStep{bound, +tail} = fits
      +size = Shape.numel(Shape.naturals(dt))
      +rest = Shape.offset(Shape.naturals(dt), Shape.naturals(vt))
      +next = Nat.add(Nat.mul(acc, U32.to_nat(dh)), U32.to_nat(vh))
      %Equal.sym(Nat, Shape.horner(dt, vt, next), Nat.add(Nat.mul(next, size), rest), horner_spec(dt, vt, next, tail)) : {_ == Nat.add(Nat.mul(acc, Nat.mul(U32.to_nat(dh), size)), Nat.add(Nat.mul(U32.to_nat(vh), size), rest)) : Nat}
      %Equal.sym(Nat, Nat.mul(next, size), Nat.add(Nat.mul(Nat.mul(acc, U32.to_nat(dh)), size), Nat.mul(U32.to_nat(vh), size)), Nats.mul_add_right(Nat.mul(acc, U32.to_nat(dh)), U32.to_nat(vh), size)) : {Nat.add(_, rest) == Nat.add(Nat.mul(acc, Nat.mul(U32.to_nat(dh), size)), Nat.add(Nat.mul(U32.to_nat(vh), size), rest)) : Nat}
      %Equal.sym(Nat, Nat.mul(Nat.mul(acc, U32.to_nat(dh)), size), Nat.mul(acc, Nat.mul(U32.to_nat(dh), size)), Nats.mul_assoc(acc, U32.to_nat(dh), size)) : {Nat.add(Nat.add(_, Nat.mul(U32.to_nat(vh), size)), rest) == Nat.add(Nat.mul(acc, Nat.mul(U32.to_nat(dh), size)), Nat.add(Nat.mul(U32.to_nat(vh), size), rest)) : Nat}
      Equal.sym(Nat, Nat.add(Nat.mul(acc, Nat.mul(U32.to_nat(dh), size)), Nat.add(Nat.mul(U32.to_nat(vh), size), rest)), Nat.add(Nat.add(Nat.mul(acc, Nat.mul(U32.to_nat(dh), size)), Nat.mul(U32.to_nat(vh), size)), rest), Index.add_assoc(Nat.mul(acc, Nat.mul(U32.to_nat(dh), size)), Nat.mul(U32.to_nat(vh), size), rest))
    case Nil{} vh <> vt:
      match fits:
    case dh <> dt Nil{}:
      match fits:

def pair_fits(+shape: Shape.Shape, +first: U32, +second: U32, +x: U32, +y: U32,
  dimensions: {Shape.dimensions(shape) == [first, second] : List<&2, U32>},
  first_inside: {Nat.is_lt(U32.to_nat(x), U32.to_nat(first)) == True{} : Bool},
  second_inside: {Nat.is_lt(U32.to_nat(y), U32.to_nat(second)) == True{} : Bool}) -> Fits(Shape.extents(shape), [U32.to_nat(x), U32.to_nat(y)]):
  fits_of(Shape.extents(shape), [U32.to_nat(x), U32.to_nat(y)],
    %Equal.sym(List<&2, U32>, Shape.dimensions(shape), [first, second], dimensions) :
      {True{} == Shape.fit(Shape.naturals(_), [U32.to_nat(x), U32.to_nat(y)]) : Bool}
    %Equal.sym(Bool, Nat.is_lt(U32.to_nat(x), U32.to_nat(first)), True{}, first_inside) :
      {True{} == Bool.and(_, Bool.and(Nat.is_lt(U32.to_nat(y), U32.to_nat(second)), True{})) : Bool}
    %Equal.sym(Bool, Nat.is_lt(U32.to_nat(y), U32.to_nat(second)), True{}, second_inside) :
      {True{} == Bool.and(True{}, Bool.and(_, True{})) : Bool}
    {==})
