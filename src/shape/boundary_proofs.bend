import Base
import ./core.bend as Shape
import ./nat_proofs.bend as Nats
import ./broadcast_proofs.bend as Broadcast
import ./coordinate_proofs.bend as Coordinates
import ./projection_proofs.bend as Projection
import ./flat.bend as Flat
import ./LAWS.bend as Laws

def fin_of_from(+bound: Nat, +value: Nat, fits: {True{} == Nat.is_lt(value, bound) : Bool}) -> {Laws.fin_of(bound, value) == Some{Coordinates.fin_from(bound, value, fits)} : Maybe<&2, Shape.Fin(bound)>}:
  match bound value:
    case 0n 0n:
      Empty.absurd({Laws.fin_of(0n, 0n) == Some{Coordinates.fin_from(0n, 0n, fits)} : Maybe<&2, Shape.Fin(0n)>}, Shape.clash(fits))
    case 0n 1n+q:
      Empty.absurd({Laws.fin_of(0n, 1n+q) == Some{Coordinates.fin_from(0n, 1n+q, fits)} : Maybe<&2, Shape.Fin(0n)>}, Shape.clash(fits))
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      Equal.cong(Maybe<&2, Shape.Fin(p)>, Maybe<&2, Shape.Fin(1n+p)>, index => Laws.fin_of.next(p, index), Laws.fin_of(p, q), Some{Coordinates.fin_from(p, q, fits)}, fin_of_from(p, q, fits))

def fin_of_none(+bound: Nat, +value: Nat, misses: {False{} == Nat.is_lt(value, bound) : Bool}) -> {None{} == Laws.fin_of(bound, value) : Maybe<&2, Shape.Fin(bound)>}:
  match bound value:
    case 0n value:
      {==}
    case 1n+p 0n:
      Empty.absurd({None{} == Laws.fin_of(1n+p, 0n) : Maybe<&2, Shape.Fin(1n+p)>}, Nats.contradiction(Nat.is_lt(0n, 1n+p), {==}, misses))
    case 1n+p 1n+q:
      Equal.cong(Maybe<&2, Shape.Fin(p)>, Maybe<&2, Shape.Fin(1n+p)>, index => Laws.fin_of.next(p, index), None{}, Laws.fin_of(p, q), fin_of_none(p, q, misses))

def fin_of_value(+bound: Nat, +index: Shape.Fin(bound)) -> {Some{index} == Laws.fin_of(bound, Shape.Fin.value(bound, index)) : Maybe<&2, Shape.Fin(bound)>}:
  match bound:
    case 0n:
      match index:
    case 1n+p:
      match index:
        case Shape.FZero{}:
          {==}
        case Shape.FNext{+previous}:
          Equal.cong(Maybe<&2, Shape.Fin(p)>, Maybe<&2, Shape.Fin(1n+p)>, index => Laws.fin_of.next(p, index), Some{previous}, Laws.fin_of(p, Shape.Fin.value(p, previous)), fin_of_value(p, previous))

def decode_none(-dh: Nat, -dt: List<&2, Nat>, +tail: Maybe<&2, Shape.Coordinate(dt)>) -> {None{} == Laws.decode.step(dh, dt, tail, None{}) : Maybe<&2, Shape.Coordinate(dh <> dt)>}:
  match tail:
    case None{}:
      {==}
    case Some{rest}:
      {==}

def decode_fit(+dims: List<&2, Nat>, +values: List<&2, Nat>, +fits: Coordinates.Fits(dims, values)) -> {Laws.decode(dims, values) == Some{Coordinates.coordinate_of(dims, values, fits)} : Maybe<&2, Shape.Coordinate(dims)>}:
  match dims values:
    case Nil{} Nil{}:
      {==}
    case +dh <> +dt +vh <> +vt:
      Coordinates.FStep{+bound, +tail} = fits
      %Equal.sym(Maybe<&2, Shape.Coordinate(dt)>, Laws.decode(dt, vt), Some{Coordinates.coordinate_of(dt, vt, tail)}, decode_fit(dt, vt, tail)) : {Laws.decode.step(dh, dt, _, Laws.fin_of(dh, vh)) == Some{Shape.CStep{Coordinates.fin_from(dh, vh, bound), Coordinates.coordinate_of(dt, vt, tail)}} : Maybe<&2, Shape.Coordinate(dh <> dt)>}
      %Equal.sym(Maybe<&2, Shape.Fin(dh)>, Laws.fin_of(dh, vh), Some{Coordinates.fin_from(dh, vh, bound)}, fin_of_from(dh, vh, bound)) : {Laws.decode.step(dh, dt, Some{Coordinates.coordinate_of(dt, vt, tail)}, _) == Some{Shape.CStep{Coordinates.fin_from(dh, vh, bound), Coordinates.coordinate_of(dt, vt, tail)}} : Maybe<&2, Shape.Coordinate(dh <> dt)>}
      {==}
    case Nil{} vh <> vt:
      match fits:
    case dh <> dt Nil{}:
      match fits:

def decode_unfit.step(+dh: Nat, +dt: List<&2, Nat>, +vh: Nat, +vt: List<&2, Nat>, +verdict: Shape.BoolCase<Nat.is_lt(vh, dh)>, misses: {False{} == Bool.and(Nat.is_lt(vh, dh), Shape.fit(dt, vt)) : Bool}, rec: {False{} == Shape.fit(dt, vt) : Bool} -> {None{} == Laws.decode(dt, vt) : Maybe<&2, Shape.Coordinate(dt)>}) -> {None{} == Laws.decode.step(dh, dt, Laws.decode(dt, vt), Laws.fin_of(dh, vh)) : Maybe<&2, Shape.Coordinate(dh <> dt)>}:
  match verdict:
    case Shape.BFalse{equal}:
      %fin_of_none(dh, vh, equal) : {None{} == Laws.decode.step(dh, dt, Laws.decode(dt, vt), _) : Maybe<&2, Shape.Coordinate(dh <> dt)>}
      decode_none(dh, dt, Laws.decode(dt, vt))
    case Shape.BTrue{equal}:
      %rec(Equal.trans(Bool, False{}, Bool.and(Nat.is_lt(vh, dh), Shape.fit(dt, vt)), Shape.fit(dt, vt), misses, Equal.cong(Bool, Bool, head => Bool.and(head, Shape.fit(dt, vt)), Nat.is_lt(vh, dh), True{}, Equal.sym(Bool, True{}, Nat.is_lt(vh, dh), equal)))) : {None{} == Laws.decode.step(dh, dt, _, Laws.fin_of(dh, vh)) : Maybe<&2, Shape.Coordinate(dh <> dt)>}
      {==}

def decode_unfit(+dims: List<&2, Nat>, +values: List<&2, Nat>, misses: {False{} == Shape.fit(dims, values) : Bool}) -> {None{} == Laws.decode(dims, values) : Maybe<&2, Shape.Coordinate(dims)>}:
  match dims values:
    case Nil{} Nil{}:
      Empty.absurd({None{} == Laws.decode(Nil{}, Nil{}) : Maybe<&2, Shape.Coordinate(Nil{})>}, Shape.clash(Equal.sym(Bool, False{}, True{}, misses)))
    case +dh <> +dt +vh <> +vt:
      decode_unfit.step(dh, dt, vh, vt, Shape.bool_case(Nat.is_lt(vh, dh)), misses, rest => decode_unfit(dt, vt, rest))
    case Nil{} vh <> vt:
      {==}
    case dh <> dt Nil{}:
      {==}

def fit_words_true(+dims: List<&2, U32>, +values: List<&2, U32>, equal: {True{} == Shape.fit_words(dims, values) : Bool}) -> {True{} == Shape.fit(Shape.naturals(dims), Shape.naturals(values)) : Bool}:
  %Coordinates.fit_words_spec(dims, values) : {True{} == _ : Bool}
  equal

def fit_words_false(+dims: List<&2, U32>, +values: List<&2, U32>, equal: {False{} == Shape.fit_words(dims, values) : Bool}) -> {False{} == Shape.fit(Shape.naturals(dims), Shape.naturals(values)) : Bool}:
  %Coordinates.fit_words_spec(dims, values) : {False{} == _ : Bool}
  equal

def decode_rank(+dims: List<&2, U32>, +values: List<&2, U32>, mismatch: {False{} == Nat.is_eq(Shape.rank_dimensions(dims), Shape.rank_dimensions(values)) : Bool}) -> {None{} == Laws.decode(Shape.naturals(dims), Shape.naturals(values)) : Maybe<&2, Shape.Coordinate(Shape.naturals(dims))>}:
  match dims values:
    case Nil{} Nil{}:
      Empty.absurd({None{} == Laws.decode(Nil{}, Nil{}) : Maybe<&2, Shape.Coordinate(Nil{})>}, Nats.contradiction(Nat.is_eq(0n, 0n), {==}, mismatch))
    case +dh <> +dt +vh <> +vt:
      %decode_rank(dt, vt, mismatch) : {None{} == Laws.decode.step(U32.to_nat(dh), Shape.naturals(dt), _, Laws.fin_of(U32.to_nat(dh), U32.to_nat(vh))) : Maybe<&2, Shape.Coordinate(Shape.naturals(dh <> dt))>}
      {==}
    case Nil{} vh <> vt:
      {==}
    case dh <> dt Nil{}:
      {==}

def decode_indices(+dims: List<&2, Nat>, +coordinate: Shape.Coordinate(dims)) -> {Some{coordinate} == Laws.decode(dims, Shape.indices(dims, coordinate)) : Maybe<&2, Shape.Coordinate(dims)>}:
  match dims:
    case Nil{}:
      match coordinate:
        case Unit{}:
          {==}
    case +head <> +tail:
      Shape.CStep{+index, +rest} = coordinate
      %decode_indices(tail, rest) : {Some{Shape.CStep{index, rest}} == Laws.decode.step(head, tail, _, Laws.fin_of(head, Shape.Fin.value(head, index))) : Maybe<&2, Shape.Coordinate(head <> tail)>}
      %fin_of_value(head, index) : {Some{Shape.CStep{index, rest}} == Laws.decode.step(head, tail, Some{rest}, _) : Maybe<&2, Shape.Coordinate(head <> tail)>}
      {==}

def checked_fit(+shape: Shape.Shape, +values: List<&2, Nat>, +verdict: Shape.BoolCase<Shape.fit_values(Shape.dimensions(shape), values)>, ok: {True{} == Shape.fit(Shape.extents(shape), values) : Bool}) -> {Laws.coordinate_result(shape, Shape.coordinate_checked.finish(shape, values, verdict)) == Some{Shape.words(values)} : Maybe<&2, List<&2, U32>>}:
  match verdict:
    case Shape.BFalse{equal}:
      Empty.absurd({None{} == Some{Shape.words(values)} : Maybe<&2, List<&2, U32>>}, Nats.contradiction(Shape.fit(Shape.extents(shape), values), ok, Equal.trans(Bool, False{}, Shape.fit_values(Shape.dimensions(shape), values), Shape.fit(Shape.extents(shape), values), equal, Shape.fit_values_spec(Shape.dimensions(shape), values))))
    case Shape.BTrue{equal}:
      {==}

def checked_indices(+shape: Shape.Shape, +coordinate: Shape.Coordinate(Shape.extents(shape))) -> {Laws.coordinate_result(shape, Shape.coordinate_checked.nat(shape, Shape.indices(Shape.extents(shape), coordinate))) == Some{Shape.words(Shape.indices(Shape.extents(shape), coordinate))} : Maybe<&2, List<&2, U32>>}:
  +values = Shape.indices(Shape.extents(shape), coordinate)
  checked_fit(shape, values, Shape.bool_case(Shape.fit_values(Shape.dimensions(shape), values)), Coordinates.fit_indices(Shape.extents(shape), coordinate))

def flat_view(-shape: Shape.Shape, +index: Nat, +verdict: Shape.BoolCase<Nat.is_lt(index, Shape.elements(shape))>, fits: {True{} == Nat.is_lt(index, Shape.elements(shape)) : Bool}) -> {Laws.index_result(shape, Shape.flat_checked.finish(shape, index, verdict)) == Some{U32.from_nat(index)} : Maybe<&2, U32>}:
  match verdict:
    case Shape.BFalse{equal}:
      Empty.absurd({Laws.index_result(shape, Shape.flat_checked.finish(shape, index, Shape.BFalse{equal})) == Some{U32.from_nat(index)} : Maybe<&2, U32>}, Nats.contradiction(Nat.is_lt(index, Shape.elements(shape)), fits, equal))
    case Shape.BTrue{equal}:
      {==}

def horner_fits(+shape: Shape.Shape, +values: List<&2, U32>, +fits: Coordinates.Fits(Shape.extents(shape), Shape.naturals(values))) -> {Shape.horner(Shape.dimensions(shape), values, 0n) == Shape.row_major(Shape.extents(shape), Coordinates.coordinate_of(Shape.extents(shape), Shape.naturals(values), fits)) : Nat}:
  Equal.trans(Nat, Shape.horner(Shape.dimensions(shape), values, 0n), Shape.offset(Shape.extents(shape), Shape.naturals(values)), Shape.row_major(Shape.extents(shape), Coordinates.coordinate_of(Shape.extents(shape), Shape.naturals(values), fits)), Coordinates.horner_spec(Shape.dimensions(shape), values, 0n, fits), Equal.sym(Nat, Shape.row_major(Shape.extents(shape), Coordinates.coordinate_of(Shape.extents(shape), Shape.naturals(values), fits)), Shape.offset(Shape.extents(shape), Shape.naturals(values)), Coordinates.offset_of(Shape.extents(shape), Shape.naturals(values), fits)))

def horner_below(+shape: Shape.Shape, +values: List<&2, U32>, +fits: Coordinates.Fits(Shape.extents(shape), Shape.naturals(values))) -> {True{} == Nat.is_lt(Shape.horner(Shape.dimensions(shape), values, 0n), Shape.elements(shape)) : Bool}:
  +coordinate = Coordinates.coordinate_of(Shape.extents(shape), Shape.naturals(values), fits)
  %Equal.sym(Nat, Shape.elements(shape), Shape.numel(Shape.extents(shape)), Shape.elements_exact(shape)) : {True{} == Nat.is_lt(Shape.horner(Shape.dimensions(shape), values, 0n), _) : Bool}
  %Equal.sym(Nat, Shape.horner(Shape.dimensions(shape), values, 0n), Shape.row_major(Shape.extents(shape), coordinate), horner_fits(shape, values, fits)) : {True{} == Nat.is_lt(_, Shape.numel(Shape.extents(shape))) : Bool}
  Nats.lt_true(Shape.row_major(Shape.extents(shape), coordinate), Shape.numel(Shape.extents(shape)), Coordinates.row_major_lt(Shape.extents(shape), coordinate))

def flatten_fits(+shape: Shape.Shape, +values: List<&2, U32>, +verdict: Shape.BoolCase<Shape.fit_words(Shape.dimensions(shape), values)>) -> {Laws.index_result(shape, Shape.flatten_checked.fits(shape, values, verdict)) == Laws.flatten_spec(Shape.extents(shape), Laws.decode(Shape.extents(shape), Shape.naturals(values))) : Maybe<&2, U32>}:
  match verdict:
    case Shape.BFalse{equal}:
      %decode_unfit(Shape.extents(shape), Shape.naturals(values), fit_words_false(Shape.dimensions(shape), values, equal)) : {None{} == Laws.flatten_spec(Shape.extents(shape), _) : Maybe<&2, U32>}
      {==}
    case Shape.BTrue{equal}:
      +fits = Coordinates.fits_of(Shape.extents(shape), Shape.naturals(values), fit_words_true(Shape.dimensions(shape), values, equal))
      +coordinate = Coordinates.coordinate_of(Shape.extents(shape), Shape.naturals(values), fits)
      +flat = Shape.horner(Shape.dimensions(shape), values, 0n)
      %Equal.sym(Maybe<&2, Shape.Coordinate(Shape.extents(shape))>, Laws.decode(Shape.extents(shape), Shape.naturals(values)), Some{coordinate}, decode_fit(Shape.extents(shape), Shape.naturals(values), fits)) : {Laws.index_result(shape, Shape.flat_checked(shape, flat)) == Laws.flatten_spec(Shape.extents(shape), _) : Maybe<&2, U32>}
      %horner_fits(shape, values, fits) : {Laws.index_result(shape, Shape.flat_checked(shape, flat)) == Some{U32.from_nat(_)} : Maybe<&2, U32>}
      flat_view(shape, flat, Shape.bool_case(Nat.is_lt(flat, Shape.elements(shape))), horner_below(shape, values, fits))

def flatten_ranked(+shape: Shape.Shape, +values: List<&2, U32>, +same: Shape.BoolCase<Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.rank_dimensions(values))>) -> {Laws.index_result(shape, Shape.flatten_checked.ranked(shape, values, same)) == Laws.flatten_spec(Shape.extents(shape), Laws.decode(Shape.extents(shape), Shape.naturals(values))) : Maybe<&2, U32>}:
  match same:
    case Shape.BFalse{equal}:
      %decode_rank(Shape.dimensions(shape), values, equal) : {None{} == Laws.flatten_spec(Shape.extents(shape), _) : Maybe<&2, U32>}
      {==}
    case Shape.BTrue{equal}:
      flatten_fits(shape, values, Shape.bool_case(Shape.fit_words(Shape.dimensions(shape), values)))

def flatten_checked(+shape: Shape.Shape, +values: List<&2, U32>) -> {Laws.index_result(shape, Shape.flatten_checked(shape, values)) == Laws.flatten_spec(Shape.extents(shape), Laws.decode(Shape.extents(shape), Shape.naturals(values))) : Maybe<&2, U32>}:
  flatten_ranked(shape, values, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.rank_dimensions(values))))

def unflatten_verdict(+shape: Shape.Shape, +index: Nat, +verdict: Shape.BoolCase<Nat.is_lt(index, Shape.elements(shape))>) -> {Laws.coordinate_result(shape, Shape.unflatten_checked.finish(shape, index, verdict)) == Laws.unflatten_spec(Shape.extents(shape), Laws.fin_of(Shape.numel(Shape.extents(shape)), index)) : Maybe<&2, List<&2, U32>>}:
  match verdict:
    case Shape.BFalse{equal}:
      %fin_of_none(Shape.numel(Shape.extents(shape)), index, Coordinates.size_misses(shape, index, equal)) : {None{} == Laws.unflatten_spec(Shape.extents(shape), _) : Maybe<&2, List<&2, U32>>}
      {==}
    case Shape.BTrue{equal}:
      +fits = Coordinates.size_fits(shape, index, equal)
      +flat = Coordinates.fin_from(Shape.numel(Shape.extents(shape)), index, fits)
      %Equal.sym(Maybe<&2, Shape.Fin(Shape.numel(Shape.extents(shape)))>, Laws.fin_of(Shape.numel(Shape.extents(shape)), index), Some{flat}, fin_of_from(Shape.numel(Shape.extents(shape)), index, fits)) : {Laws.coordinate_result(shape, Shape.coordinate_checked.nat(shape, Shape.digits_words(Shape.dimensions(shape), index))) == Laws.unflatten_spec(Shape.extents(shape), _) : Maybe<&2, List<&2, U32>>}
      %Equal.sym(List<&2, Nat>, Shape.digits_words(Shape.dimensions(shape), index), Shape.digits(Shape.extents(shape), index), Coordinates.digits_words_spec(Shape.dimensions(shape), index)) : {Laws.coordinate_result(shape, Shape.coordinate_checked.nat(shape, _)) == Some{Shape.words(Shape.indices(Shape.extents(shape), Shape.unflatten(Shape.extents(shape), flat)))} : Maybe<&2, List<&2, U32>>}
      %Coordinates.fin_from_value(Shape.numel(Shape.extents(shape)), index, fits) : {Laws.coordinate_result(shape, Shape.coordinate_checked.nat(shape, Shape.digits(Shape.extents(shape), _))) == Some{Shape.words(Shape.indices(Shape.extents(shape), Shape.unflatten(Shape.extents(shape), flat)))} : Maybe<&2, List<&2, U32>>}
      %Coordinates.digits_unflatten(Shape.extents(shape), flat) : {Laws.coordinate_result(shape, Shape.coordinate_checked.nat(shape, _)) == Some{Shape.words(Shape.indices(Shape.extents(shape), Shape.unflatten(Shape.extents(shape), flat)))} : Maybe<&2, List<&2, U32>>}
      checked_indices(shape, Shape.unflatten(Shape.extents(shape), flat))

def unflatten_checked(+shape: Shape.Shape, +index: U32) -> {Laws.coordinate_result(shape, Shape.unflatten_checked(shape, index)) == Laws.unflatten_spec(Shape.extents(shape), Laws.fin_of(Shape.numel(Shape.extents(shape)), U32.to_nat(index))) : Maybe<&2, List<&2, U32>>}:
  unflatten_verdict(shape, U32.to_nat(index), Shape.bool_case(Nat.is_lt(U32.to_nat(index), Shape.elements(shape))))

def projected_left(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +alignment: Shape.Broadcast<Shape.extents(left), Shape.extents(right), Shape.extents(output)>, +values: List<&2, U32>, +fits: Coordinates.Fits(Shape.extents(output), Shape.naturals(values))) -> {Shape.projected_words(Shape.dimensions(left), List.drop(&2, U32, values, Nat.sub(Shape.rank_dimensions(values), Shape.rank_dimensions(Shape.dimensions(left))))) == Shape.indices(Shape.extents(left), Shape.project_left(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, Coordinates.coordinate_of(Shape.extents(output), Shape.naturals(values), fits))) : List<&2, Nat>}:
  +coordinate = Coordinates.coordinate_of(Shape.extents(output), Shape.naturals(values), fits)
  +expected = Shape.indices(Shape.extents(left), Shape.project_left(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, coordinate))
  +count = Nat.sub(Shape.rank_dimensions(values), Shape.rank_dimensions(Shape.dimensions(left)))
  %Equal.sym(List<&2, Nat>, Shape.projected_words(Shape.dimensions(left), List.drop(&2, U32, values, count)), Shape.broadcast_indices(Shape.extents(left), Shape.naturals(List.drop(&2, U32, values, count))), Coordinates.projected_words_spec(Shape.dimensions(left), List.drop(&2, U32, values, count))) : {_ == expected : List<&2, Nat>}
  %Equal.sym(List<&2, Nat>, Shape.naturals(List.drop(&2, U32, values, count)), List.drop(&2, Nat, Shape.naturals(values), count), Coordinates.naturals_drop(values, count)) : {Shape.broadcast_indices(Shape.extents(left), _) == expected : List<&2, Nat>}
  %Equal.sym(Nat, Shape.rank_dimensions(values), List.length(&2, Nat, Shape.naturals(values)), Shape.rank_naturals(values)) : {Shape.broadcast_indices(Shape.extents(left), List.drop(&2, Nat, Shape.naturals(values), Nat.sub(_, Shape.rank_dimensions(Shape.dimensions(left))))) == expected : List<&2, Nat>}
  %Equal.sym(Nat, Shape.rank_dimensions(Shape.dimensions(left)), List.length(&2, Nat, Shape.extents(left)), Shape.rank_naturals(Shape.dimensions(left))) : {Shape.broadcast_indices(Shape.extents(left), List.drop(&2, Nat, Shape.naturals(values), Nat.sub(List.length(&2, Nat, Shape.naturals(values)), _))) == expected : List<&2, Nat>}
  %Coordinates.indices_of(Shape.extents(output), Shape.naturals(values), fits) : {Shape.project_indices(Shape.extents(left), _) == expected : List<&2, Nat>}
  Equal.sym(List<&2, Nat>, expected, Shape.project_indices(Shape.extents(left), Shape.indices(Shape.extents(output), coordinate)), Broadcast.project_left(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, coordinate))

def project_left_fits(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +alignment: Shape.Broadcast<Shape.extents(left), Shape.extents(right), Shape.extents(output)>, +values: List<&2, U32>, +verdict: Shape.BoolCase<Shape.fit_words(Shape.dimensions(output), values)>) -> {Laws.coordinate_result(left, Shape.projection_checked.fits(left, output, values, verdict)) == Laws.project_left_spec(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, Laws.decode(Shape.extents(output), Shape.naturals(values))) : Maybe<&2, List<&2, U32>>}:
  match verdict:
    case Shape.BFalse{equal}:
      %decode_unfit(Shape.extents(output), Shape.naturals(values), fit_words_false(Shape.dimensions(output), values, equal)) : {None{} == Laws.project_left_spec(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, _) : Maybe<&2, List<&2, U32>>}
      {==}
    case Shape.BTrue{equal}:
      +fits = Coordinates.fits_of(Shape.extents(output), Shape.naturals(values), fit_words_true(Shape.dimensions(output), values, equal))
      +coordinate = Coordinates.coordinate_of(Shape.extents(output), Shape.naturals(values), fits)
      +projected = Shape.project_left(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, coordinate)
      +count = Nat.sub(Shape.rank_dimensions(values), Shape.rank_dimensions(Shape.dimensions(left)))
      %Equal.sym(Maybe<&2, Shape.Coordinate(Shape.extents(output))>, Laws.decode(Shape.extents(output), Shape.naturals(values)), Some{coordinate}, decode_fit(Shape.extents(output), Shape.naturals(values), fits)) : {Laws.coordinate_result(left, Shape.coordinate_checked.nat(left, Shape.projected_words(Shape.dimensions(left), List.drop(&2, U32, values, count)))) == Laws.project_left_spec(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, _) : Maybe<&2, List<&2, U32>>}
      %Equal.sym(List<&2, Nat>, Shape.projected_words(Shape.dimensions(left), List.drop(&2, U32, values, count)), Shape.indices(Shape.extents(left), projected), projected_left(left, right, output, alignment, values, fits)) : {Laws.coordinate_result(left, Shape.coordinate_checked.nat(left, _)) == Some{Shape.words(Shape.indices(Shape.extents(left), projected))} : Maybe<&2, List<&2, U32>>}
      checked_indices(left, projected)

def project_left_ranked(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +alignment: Shape.Broadcast<Shape.extents(left), Shape.extents(right), Shape.extents(output)>, +values: List<&2, U32>, +same: Shape.BoolCase<Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(output)), Shape.rank_dimensions(values))>) -> {Laws.coordinate_result(left, Shape.projection_checked.ranked(left, output, values, same)) == Laws.project_left_spec(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, Laws.decode(Shape.extents(output), Shape.naturals(values))) : Maybe<&2, List<&2, U32>>}:
  match same:
    case Shape.BFalse{equal}:
      %decode_rank(Shape.dimensions(output), values, equal) : {None{} == Laws.project_left_spec(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, _) : Maybe<&2, List<&2, U32>>}
      {==}
    case Shape.BTrue{equal}:
      project_left_fits(left, right, output, alignment, values, Shape.bool_case(Shape.fit_words(Shape.dimensions(output), values)))

def project_left_checked(+plan: Shape.BroadcastPlan, +values: List<&2, U32>) -> {Laws.coordinate_result(Shape.left_shape(plan), Shape.project_left_checked(plan, values)) == Laws.project_left_spec(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.alignment(plan), Laws.decode(Shape.extents(Shape.output_shape(plan)), Shape.naturals(values))) : Maybe<&2, List<&2, U32>>}:
  match plan:
    case Shape.BPlan{+left, +right, +output, +alignment}:
      project_left_ranked(left, right, output, alignment, values, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(output)), Shape.rank_dimensions(values))))

def projected_right(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +alignment: Shape.Broadcast<Shape.extents(left), Shape.extents(right), Shape.extents(output)>, +values: List<&2, U32>, +fits: Coordinates.Fits(Shape.extents(output), Shape.naturals(values))) -> {Shape.projected_words(Shape.dimensions(right), List.drop(&2, U32, values, Nat.sub(Shape.rank_dimensions(values), Shape.rank_dimensions(Shape.dimensions(right))))) == Shape.indices(Shape.extents(right), Shape.project_right(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, Coordinates.coordinate_of(Shape.extents(output), Shape.naturals(values), fits))) : List<&2, Nat>}:
  +coordinate = Coordinates.coordinate_of(Shape.extents(output), Shape.naturals(values), fits)
  +expected = Shape.indices(Shape.extents(right), Shape.project_right(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, coordinate))
  +count = Nat.sub(Shape.rank_dimensions(values), Shape.rank_dimensions(Shape.dimensions(right)))
  %Equal.sym(List<&2, Nat>, Shape.projected_words(Shape.dimensions(right), List.drop(&2, U32, values, count)), Shape.broadcast_indices(Shape.extents(right), Shape.naturals(List.drop(&2, U32, values, count))), Coordinates.projected_words_spec(Shape.dimensions(right), List.drop(&2, U32, values, count))) : {_ == expected : List<&2, Nat>}
  %Equal.sym(List<&2, Nat>, Shape.naturals(List.drop(&2, U32, values, count)), List.drop(&2, Nat, Shape.naturals(values), count), Coordinates.naturals_drop(values, count)) : {Shape.broadcast_indices(Shape.extents(right), _) == expected : List<&2, Nat>}
  %Equal.sym(Nat, Shape.rank_dimensions(values), List.length(&2, Nat, Shape.naturals(values)), Shape.rank_naturals(values)) : {Shape.broadcast_indices(Shape.extents(right), List.drop(&2, Nat, Shape.naturals(values), Nat.sub(_, Shape.rank_dimensions(Shape.dimensions(right))))) == expected : List<&2, Nat>}
  %Equal.sym(Nat, Shape.rank_dimensions(Shape.dimensions(right)), List.length(&2, Nat, Shape.extents(right)), Shape.rank_naturals(Shape.dimensions(right))) : {Shape.broadcast_indices(Shape.extents(right), List.drop(&2, Nat, Shape.naturals(values), Nat.sub(List.length(&2, Nat, Shape.naturals(values)), _))) == expected : List<&2, Nat>}
  %Coordinates.indices_of(Shape.extents(output), Shape.naturals(values), fits) : {Shape.project_indices(Shape.extents(right), _) == expected : List<&2, Nat>}
  Equal.sym(List<&2, Nat>, expected, Shape.project_indices(Shape.extents(right), Shape.indices(Shape.extents(output), coordinate)), Broadcast.project_right(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, coordinate))

def project_right_fits(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +alignment: Shape.Broadcast<Shape.extents(left), Shape.extents(right), Shape.extents(output)>, +values: List<&2, U32>, +verdict: Shape.BoolCase<Shape.fit_words(Shape.dimensions(output), values)>) -> {Laws.coordinate_result(right, Shape.projection_checked.fits(right, output, values, verdict)) == Laws.project_right_spec(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, Laws.decode(Shape.extents(output), Shape.naturals(values))) : Maybe<&2, List<&2, U32>>}:
  match verdict:
    case Shape.BFalse{equal}:
      %decode_unfit(Shape.extents(output), Shape.naturals(values), fit_words_false(Shape.dimensions(output), values, equal)) : {None{} == Laws.project_right_spec(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, _) : Maybe<&2, List<&2, U32>>}
      {==}
    case Shape.BTrue{equal}:
      +fits = Coordinates.fits_of(Shape.extents(output), Shape.naturals(values), fit_words_true(Shape.dimensions(output), values, equal))
      +coordinate = Coordinates.coordinate_of(Shape.extents(output), Shape.naturals(values), fits)
      +projected = Shape.project_right(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, coordinate)
      +count = Nat.sub(Shape.rank_dimensions(values), Shape.rank_dimensions(Shape.dimensions(right)))
      %Equal.sym(Maybe<&2, Shape.Coordinate(Shape.extents(output))>, Laws.decode(Shape.extents(output), Shape.naturals(values)), Some{coordinate}, decode_fit(Shape.extents(output), Shape.naturals(values), fits)) : {Laws.coordinate_result(right, Shape.coordinate_checked.nat(right, Shape.projected_words(Shape.dimensions(right), List.drop(&2, U32, values, count)))) == Laws.project_right_spec(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, _) : Maybe<&2, List<&2, U32>>}
      %Equal.sym(List<&2, Nat>, Shape.projected_words(Shape.dimensions(right), List.drop(&2, U32, values, count)), Shape.indices(Shape.extents(right), projected), projected_right(left, right, output, alignment, values, fits)) : {Laws.coordinate_result(right, Shape.coordinate_checked.nat(right, _)) == Some{Shape.words(Shape.indices(Shape.extents(right), projected))} : Maybe<&2, List<&2, U32>>}
      checked_indices(right, projected)

def project_right_ranked(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +alignment: Shape.Broadcast<Shape.extents(left), Shape.extents(right), Shape.extents(output)>, +values: List<&2, U32>, +same: Shape.BoolCase<Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(output)), Shape.rank_dimensions(values))>) -> {Laws.coordinate_result(right, Shape.projection_checked.ranked(right, output, values, same)) == Laws.project_right_spec(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, Laws.decode(Shape.extents(output), Shape.naturals(values))) : Maybe<&2, List<&2, U32>>}:
  match same:
    case Shape.BFalse{equal}:
      %decode_rank(Shape.dimensions(output), values, equal) : {None{} == Laws.project_right_spec(Shape.extents(left), Shape.extents(right), Shape.extents(output), alignment, _) : Maybe<&2, List<&2, U32>>}
      {==}
    case Shape.BTrue{equal}:
      project_right_fits(left, right, output, alignment, values, Shape.bool_case(Shape.fit_words(Shape.dimensions(output), values)))

def project_right_checked(+plan: Shape.BroadcastPlan, +values: List<&2, U32>) -> {Laws.coordinate_result(Shape.right_shape(plan), Shape.project_right_checked(plan, values)) == Laws.project_right_spec(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.alignment(plan), Laws.decode(Shape.extents(Shape.output_shape(plan)), Shape.naturals(values))) : Maybe<&2, List<&2, U32>>}:
  match plan:
    case Shape.BPlan{+left, +right, +output, +alignment}:
      project_right_ranked(left, right, output, alignment, values, Shape.bool_case(Nat.is_eq(Shape.rank_dimensions(Shape.dimensions(output)), Shape.rank_dimensions(values))))

def project_left_index(+plan: Shape.BroadcastPlan, +map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>, +index: Shape.FlatIndex<Shape.output_shape(plan)>) -> {Some{Shape.index_value(Shape.left_shape(plan), Flat.project_left_index(plan, map, index))} == Laws.flatten_spec(Shape.extents(Shape.left_shape(plan)), Laws.project_left_coordinate(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.alignment(plan), Laws.fin_of(Shape.numel(Shape.extents(Shape.output_shape(plan))), Laws.position(Shape.output_shape(plan), index)))) : Maybe<&2, U32>}:
  match map index:
    case Shape.Indexed{+rows, +exact} Shape.FIndex{+value, +bound}:
      +left = Shape.extents(Shape.left_shape(plan))
      +right = Shape.extents(Shape.right_shape(plan))
      +output = Shape.extents(Shape.output_shape(plan))
      +fits = Coordinates.size_fits(Shape.output_shape(plan), value, bound)
      +flat = Coordinates.fin_from(Shape.numel(output), value, fits)
      +table = Shape.index_table(left, right, output)
      +projected = Shape.row_major(Shape.extents(Shape.left_shape(plan)), Shape.project_left(left, right, output, Shape.alignment(plan), Shape.unflatten(output, flat)))
      %Equal.sym(Maybe<&2, Shape.Fin(Shape.numel(output))>, Laws.fin_of(Shape.numel(output), value), Some{flat}, fin_of_from(Shape.numel(output), value, fits)) : {Some{U32.from_nat(Shape.walk_left(rows, value))} == Laws.flatten_spec(Shape.extents(Shape.left_shape(plan)), Laws.project_left_coordinate(left, right, output, Shape.alignment(plan), _)) : Maybe<&2, U32>}
      %Equal.sym(List<&2, Shape.Row>, rows, table, exact) : {Some{U32.from_nat(Shape.walk_left(_, value))} == Some{U32.from_nat(projected)} : Maybe<&2, U32>}
      %Coordinates.fin_from_value(Shape.numel(output), value, fits) : {Some{U32.from_nat(Shape.walk_left(table, _))} == Some{U32.from_nat(projected)} : Maybe<&2, U32>}
      Equal.cong(Nat, Maybe<&2, U32>, flat => Some{U32.from_nat(flat)}, Shape.walk_left(table, Shape.Fin.value(Shape.numel(output), flat)), projected, Projection.walk_left_value(left, right, output, Shape.alignment(plan), flat))
    case Shape.Unindexed{zero} Shape.FIndex{+value, bound}:
      +left = Shape.extents(Shape.left_shape(plan))
      +right = Shape.extents(Shape.right_shape(plan))
      +output = Shape.extents(Shape.output_shape(plan))
      Empty.absurd({Some{Shape.index_value(Shape.left_shape(plan), Flat.project_left_index(plan, Shape.Unindexed{zero}, Shape.FIndex{value, bound}))} == Laws.flatten_spec(Shape.extents(Shape.left_shape(plan)), Laws.project_left_coordinate(left, right, output, Shape.alignment(plan), Laws.fin_of(Shape.numel(output), value))) : Maybe<&2, U32>}, Flat.hollow(Shape.output_shape(plan), value, zero, bound))

def project_right_index(+plan: Shape.BroadcastPlan, +map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>, +index: Shape.FlatIndex<Shape.output_shape(plan)>) -> {Some{Shape.index_value(Shape.right_shape(plan), Flat.project_right_index(plan, map, index))} == Laws.flatten_spec(Shape.extents(Shape.right_shape(plan)), Laws.project_right_coordinate(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), Shape.alignment(plan), Laws.fin_of(Shape.numel(Shape.extents(Shape.output_shape(plan))), Laws.position(Shape.output_shape(plan), index)))) : Maybe<&2, U32>}:
  match map index:
    case Shape.Indexed{+rows, +exact} Shape.FIndex{+value, +bound}:
      +left = Shape.extents(Shape.left_shape(plan))
      +right = Shape.extents(Shape.right_shape(plan))
      +output = Shape.extents(Shape.output_shape(plan))
      +fits = Coordinates.size_fits(Shape.output_shape(plan), value, bound)
      +flat = Coordinates.fin_from(Shape.numel(output), value, fits)
      +table = Shape.index_table(left, right, output)
      +projected = Shape.row_major(Shape.extents(Shape.right_shape(plan)), Shape.project_right(left, right, output, Shape.alignment(plan), Shape.unflatten(output, flat)))
      %Equal.sym(Maybe<&2, Shape.Fin(Shape.numel(output))>, Laws.fin_of(Shape.numel(output), value), Some{flat}, fin_of_from(Shape.numel(output), value, fits)) : {Some{U32.from_nat(Shape.walk_right(rows, value))} == Laws.flatten_spec(Shape.extents(Shape.right_shape(plan)), Laws.project_right_coordinate(left, right, output, Shape.alignment(plan), _)) : Maybe<&2, U32>}
      %Equal.sym(List<&2, Shape.Row>, rows, table, exact) : {Some{U32.from_nat(Shape.walk_right(_, value))} == Some{U32.from_nat(projected)} : Maybe<&2, U32>}
      %Coordinates.fin_from_value(Shape.numel(output), value, fits) : {Some{U32.from_nat(Shape.walk_right(table, _))} == Some{U32.from_nat(projected)} : Maybe<&2, U32>}
      Equal.cong(Nat, Maybe<&2, U32>, flat => Some{U32.from_nat(flat)}, Shape.walk_right(table, Shape.Fin.value(Shape.numel(output), flat)), projected, Projection.walk_right_value(left, right, output, Shape.alignment(plan), flat))
    case Shape.Unindexed{zero} Shape.FIndex{+value, bound}:
      +left = Shape.extents(Shape.left_shape(plan))
      +right = Shape.extents(Shape.right_shape(plan))
      +output = Shape.extents(Shape.output_shape(plan))
      Empty.absurd({Some{Shape.index_value(Shape.right_shape(plan), Flat.project_right_index(plan, Shape.Unindexed{zero}, Shape.FIndex{value, bound}))} == Laws.flatten_spec(Shape.extents(Shape.right_shape(plan)), Laws.project_right_coordinate(left, right, output, Shape.alignment(plan), Laws.fin_of(Shape.numel(output), value))) : Maybe<&2, U32>}, Flat.hollow(Shape.output_shape(plan), value, zero, bound))
