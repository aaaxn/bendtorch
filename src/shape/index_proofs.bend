import Base
import ./core.bend as Shape

def add_assoc(a: Nat, -b: Nat, -c: Nat) -> {Nat.add(a, Nat.add(b, c)) == Nat.add(Nat.add(a, b), c) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_assoc(p, b, c) : {1n+Nat.add(p, Nat.add(b, c)) == 1n+_ : Nat}
      {==}

def split_left(+a: Nat, +b: Nat, +index: Shape.Fin(a)) -> {Shape.Fin.split(a, b, Shape.Fin.left(a, b, index)) == Inl{index} : Either<Shape.Fin(a), Shape.Fin(b)>}:
  match a:
    case 0n:
      match index:
    case 1n+p:
      match index:
        case Shape.FZero{}:
          {==}
        case Shape.FNext{+previous}:
          %split_left(p, b, previous) : {Shape.Fin.split.step(p, b, Shape.Fin.split(p, b, Shape.Fin.left(p, b, previous))) == Shape.Fin.split.step(p, b, _) : Either<Shape.Fin(1n+p), Shape.Fin(b)>}
          {==}

def split_shift(+a: Nat, +b: Nat, +index: Shape.Fin(b)) -> {Shape.Fin.split(a, b, Shape.Fin.shift(a, b, index)) == Inr{index} : Either<Shape.Fin(a), Shape.Fin(b)>}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %split_shift(p, b, index) : {Shape.Fin.split.step(p, b, Shape.Fin.split(p, b, Shape.Fin.shift(p, b, index))) == Shape.Fin.split.step(p, b, _) : Either<Shape.Fin(1n+p), Shape.Fin(b)>}
      {==}

def merge_step(+a: Nat, +b: Nat, split: Either<Shape.Fin(a), Shape.Fin(b)>) -> {Shape.Fin.merge(1n+a, b, Shape.Fin.split.step(a, b, split)) == Shape.FNext{Shape.Fin.merge(a, b, split)} : Shape.Fin(1n+Nat.add(a, b))}:
  match split:
    case Inl{index}:
      {==}
    case Inr{index}:
      {==}

def merge_split(+a: Nat, +b: Nat, +index: Shape.Fin(Nat.add(a, b))) -> {Shape.Fin.merge(a, b, Shape.Fin.split(a, b, index)) == index : Shape.Fin(Nat.add(a, b))}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      match index:
        case Shape.FZero{}:
          {==}
        case Shape.FNext{+previous}:
          %merge_split(p, b, previous) : {Shape.Fin.merge(1n+p, b, Shape.Fin.split.step(p, b, Shape.Fin.split(p, b, previous))) == Shape.FNext{_} : Shape.Fin(1n+Nat.add(p, b))}
          merge_step(p, b, Shape.Fin.split(p, b, previous))

def left_value(+a: Nat, +b: Nat, +index: Shape.Fin(a)) -> {Shape.Fin.value(Nat.add(a, b), Shape.Fin.left(a, b, index)) == Shape.Fin.value(a, index) : Nat}:
  match a:
    case 0n:
      match index:
    case 1n+p:
      match index:
        case Shape.FZero{}:
          {==}
        case Shape.FNext{+previous}:
          %left_value(p, b, previous) : {1n+Shape.Fin.value(Nat.add(p, b), Shape.Fin.left(p, b, previous)) == 1n+_ : Nat}
          {==}

def shift_value(+a: Nat, +b: Nat, +index: Shape.Fin(b)) -> {Shape.Fin.value(Nat.add(a, b), Shape.Fin.shift(a, b, index)) == Nat.add(a, Shape.Fin.value(b, index)) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %shift_value(p, b, index) : {1n+Shape.Fin.value(Nat.add(p, b), Shape.Fin.shift(p, b, index)) == 1n+_ : Nat}
      {==}

def pack_value(+a: Nat, +b: Nat, +first: Shape.Fin(a), +last: Shape.Fin(b)) -> {Shape.Fin.value(Nat.mul(a, b), Shape.Fin.pack(a, b, (first, last))) == Nat.add(Nat.mul(Shape.Fin.value(a, first), b), Shape.Fin.value(b, last)) : Nat}:
  match a:
    case 0n:
      match first:
    case 1n+p:
      match first:
        case Shape.FZero{}:
          left_value(b, Nat.mul(p, b), last)
        case Shape.FNext{+previous}:
          %add_assoc(b, Nat.mul(Shape.Fin.value(p, previous), b), Shape.Fin.value(b, last)) : {Shape.Fin.value(Nat.add(b, Nat.mul(p, b)), Shape.Fin.shift(b, Nat.mul(p, b), Shape.Fin.pack(p, b, (previous, last)))) == _ : Nat}
          %pack_value(p, b, previous, last) : {Shape.Fin.value(Nat.add(b, Nat.mul(p, b)), Shape.Fin.shift(b, Nat.mul(p, b), Shape.Fin.pack(p, b, (previous, last)))) == Nat.add(b, _) : Nat}
          shift_value(b, Nat.mul(p, b), Shape.Fin.pack(p, b, (previous, last)))

def unpack_pack(+a: Nat, +b: Nat, +first: Shape.Fin(a), +last: Shape.Fin(b)) -> {Shape.Fin.unpack(a, b, Shape.Fin.pack(a, b, (first, last))) == (first, last) : Shape.Fin(a) & Shape.Fin(b)}:
  match a:
    case 0n:
      match first:
    case 1n+p:
      match first:
        case Shape.FZero{}:
          %split_left(b, Nat.mul(p, b), last) : {Shape.Fin.unpack.step(p, b, index => Shape.Fin.unpack(p, b, index), Shape.Fin.split(b, Nat.mul(p, b), Shape.Fin.left(b, Nat.mul(p, b), last))) == Shape.Fin.unpack.step(p, b, index => Shape.Fin.unpack(p, b, index), _) : Shape.Fin(1n+p) & Shape.Fin(b)}
          {==}
        case Shape.FNext{+previous}:
          %unpack_pack(p, b, previous, last) : {Shape.Fin.unpack(1n+p, b, Shape.Fin.pack(1n+p, b, (Shape.FNext{previous}, last))) == Shape.Fin.unpack.lift(p, b, _) : Shape.Fin(1n+p) & Shape.Fin(b)}
          %split_shift(b, Nat.mul(p, b), Shape.Fin.pack(p, b, (previous, last))) : {Shape.Fin.unpack.step(p, b, index => Shape.Fin.unpack(p, b, index), Shape.Fin.split(b, Nat.mul(p, b), Shape.Fin.shift(b, Nat.mul(p, b), Shape.Fin.pack(p, b, (previous, last))))) == Shape.Fin.unpack.step(p, b, index => Shape.Fin.unpack(p, b, index), _) : Shape.Fin(1n+p) & Shape.Fin(b)}
          {==}

def pack_lift(+a: Nat, +b: Nat, coordinate: Shape.Fin(a) & Shape.Fin(b)) -> {Shape.Fin.pack(1n+a, b, Shape.Fin.unpack.lift(a, b, coordinate)) == Shape.Fin.shift(b, Nat.mul(a, b), Shape.Fin.pack(a, b, coordinate)) : Shape.Fin(Nat.add(b, Nat.mul(a, b)))}:
  (first, last) = coordinate
  {==}

def pack_unpack_step(+a: Nat, +b: Nat, rec: @+index: Shape.Fin(Nat.mul(a, b)) -> {Shape.Fin.pack(a, b, Shape.Fin.unpack(a, b, index)) == index : Shape.Fin(Nat.mul(a, b))}, split: Either<Shape.Fin(b), Shape.Fin(Nat.mul(a, b))>) -> {Shape.Fin.pack(1n+a, b, Shape.Fin.unpack.step(a, b, index => Shape.Fin.unpack(a, b, index), split)) == Shape.Fin.merge(b, Nat.mul(a, b), split) : Shape.Fin(Nat.add(b, Nat.mul(a, b)))}:
  match split:
    case Inl{last}:
      {==}
    case Inr{+index}:
      %rec(index) : {Shape.Fin.pack(1n+a, b, Shape.Fin.unpack.lift(a, b, Shape.Fin.unpack(a, b, index))) == Shape.Fin.shift(b, Nat.mul(a, b), _) : Shape.Fin(Nat.add(b, Nat.mul(a, b)))}
      pack_lift(a, b, Shape.Fin.unpack(a, b, index))

def pack_unpack(+a: Nat, +b: Nat, +index: Shape.Fin(Nat.mul(a, b))) -> {Shape.Fin.pack(a, b, Shape.Fin.unpack(a, b, index)) == index : Shape.Fin(Nat.mul(a, b))}:
  match a:
    case 0n:
      match index:
    case 1n+p:
      %merge_split(b, Nat.mul(p, b), index) : {Shape.Fin.pack(1n+p, b, Shape.Fin.unpack(1n+p, b, index)) == _ : Shape.Fin(Nat.add(b, Nat.mul(p, b)))}
      pack_unpack_step(p, b, +index => pack_unpack(p, b, index), Shape.Fin.split(b, Nat.mul(p, b), index))

def flatten_unflatten_step(+head: Nat, +tail: List<&2, Nat>, rec: @+index: Shape.Fin(Shape.numel(tail)) -> {Shape.flatten(tail, Shape.unflatten(tail, index)) == index : Shape.Fin(Shape.numel(tail))}, coordinate: Shape.Fin(head) & Shape.Fin(Shape.numel(tail))) -> {Shape.flatten(head <> tail, Shape.unflatten.step(head, tail, rest => Shape.unflatten(tail, rest), coordinate)) == Shape.Fin.pack(head, Shape.numel(tail), coordinate) : Shape.Fin(Shape.numel(head <> tail))}:
  (first, +last) = coordinate
  %rec(last) : {Shape.Fin.pack(head, Shape.numel(tail), (first, Shape.flatten(tail, Shape.unflatten(tail, last)))) == Shape.Fin.pack(head, Shape.numel(tail), (first, _)) : Shape.Fin(Shape.numel(head <> tail))}
  {==}

def flatten_unflatten(+dims: List<&2, Nat>, +index: Shape.Fin(Shape.numel(dims))) -> {Shape.flatten(dims, Shape.unflatten(dims, index)) == index : Shape.Fin(Shape.numel(dims))}:
  match dims:
    case Nil{}:
      match index:
        case Shape.FZero{}:
          {==}
        case Shape.FNext{previous}:
          match previous:
    case +head <> +tail:
      %pack_unpack(head, Shape.numel(tail), index) : {Shape.flatten(head <> tail, Shape.unflatten(head <> tail, index)) == _ : Shape.Fin(Shape.numel(head <> tail))}
      flatten_unflatten_step(head, tail, +last => flatten_unflatten(tail, last), Shape.Fin.unpack(head, Shape.numel(tail), index))

def unflatten_flatten(+dims: List<&2, Nat>, +coordinate: Shape.Coordinate(dims)) -> {Shape.unflatten(dims, Shape.flatten(dims, coordinate)) == coordinate : Shape.Coordinate(dims)}:
  match dims:
    case Nil{}:
      match coordinate:
        case Unit{}:
          {==}
    case +head <> +tail:
      Shape.CStep{+first, +rest} = coordinate
      %unflatten_flatten(tail, rest) : {Shape.unflatten(head <> tail, Shape.flatten(head <> tail, Shape.CStep{first, rest})) == Shape.CStep{first, _} : Shape.Coordinate(head <> tail)}
      %unpack_pack(head, Shape.numel(tail), first, Shape.flatten(tail, rest)) : {Shape.unflatten.step(head, tail, last => Shape.unflatten(tail, last), Shape.Fin.unpack(head, Shape.numel(tail), Shape.Fin.pack(head, Shape.numel(tail), (first, Shape.flatten(tail, rest))))) == Shape.unflatten.step(head, tail, last => Shape.unflatten(tail, last), _) : Shape.Coordinate(head <> tail)}
      {==}

def flatten_row_major(+dims: List<&2, Nat>, +coordinate: Shape.Coordinate(dims)) -> {Shape.Fin.value(Shape.numel(dims), Shape.flatten(dims, coordinate)) == Shape.row_major(dims, coordinate) : Nat}:
  match dims:
    case Nil{}:
      {==}
    case +head <> +tail:
      Shape.CStep{+first, +rest} = coordinate
      %flatten_row_major(tail, rest) : {Shape.Fin.value(Shape.numel(head <> tail), Shape.flatten(head <> tail, Shape.CStep{first, rest})) == Nat.add(Nat.mul(Shape.Fin.value(head, first), Shape.numel(tail)), _) : Nat}
      pack_value(head, Shape.numel(tail), first, Shape.flatten(tail, rest))
