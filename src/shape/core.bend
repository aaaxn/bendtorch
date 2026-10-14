import Base

type Fin.Step<-k: Nat> is Data:
  FZero{}
  FNext{previous: Fin(k)}

def Fin(n: Nat) -> Data:
  match n:
    case 0n:
      Empty
    case 1n+k:
      Fin.Step<k>

def Fin.value(+n: Nat, index: Fin(n)) -> Nat:
  match n:
    case 0n:
      match index:
    case 1n+k:
      match index:
        case FZero{}:
          0n
        case FNext{previous}:
          1n+Fin.value(k, previous)

def LT(a: Nat, b: Nat) -> Data:
  match a b:
    case 0n 0n:
      Empty
    case 0n 1n+q:
      Unit
    case 1n+p 0n:
      Empty
    case 1n+p 1n+q:
      LT(p, q)

def LE(a: Nat, b: Nat) -> Data:
  match a b:
    case 0n q:
      Unit
    case 1n+p 0n:
      Empty
    case 1n+p 1n+q:
      LE(p, q)

def Fin.bound(+n: Nat, +index: Fin(n)) -> LT(Fin.value(n, index), n):
  match n:
    case 0n:
      match index:
    case 1n+k:
      match index:
        case FZero{}:
          Unit{}
        case FNext{+previous}:
          Fin.bound(k, previous)

def Fin.left(+a: Nat, -b: Nat, index: Fin(a)) -> Fin(Nat.add(a, b)):
  match a:
    case 0n:
      match index:
    case 1n+p:
      match index:
        case FZero{}:
          FZero{}
        case FNext{previous}:
          FNext{Fin.left(p, b, previous)}

def Fin.shift(a: Nat, -b: Nat, index: Fin(b)) -> Fin(Nat.add(a, b)):
  match a:
    case 0n:
      index
    case 1n+p:
      FNext{Fin.shift(p, b, index)}

def Fin.split.step(-a: Nat, -b: Nat, split: Either<Fin(a), Fin(b)>) -> Either<Fin(1n+a), Fin(b)>:
  match split:
    case Inl{index}:
      Inl{FNext{index}}
    case Inr{index}:
      Inr{index}

def Fin.split(+a: Nat, +b: Nat, index: Fin(Nat.add(a, b))) -> Either<Fin(a), Fin(b)>:
  match a:
    case 0n:
      Inr{index}
    case 1n+p:
      match index:
        case FZero{}:
          Inl{FZero{}}
        case FNext{previous}:
          Fin.split.step(p, b, Fin.split(p, b, previous))

def Fin.merge(+a: Nat, -b: Nat, split: Either<Fin(a), Fin(b)>) -> Fin(Nat.add(a, b)):
  match split:
    case Inl{index}:
      Fin.left(a, b, index)
    case Inr{index}:
      Fin.shift(a, b, index)

def Fin.pack(+a: Nat, +b: Nat, coordinate: Fin(a) & Fin(b)) -> Fin(Nat.mul(a, b)):
  match a:
    case 0n:
      (head, tail) = coordinate
      match head:
    case 1n+p:
      (head, tail) = coordinate
      match head:
        case FZero{}:
          Fin.left(b, Nat.mul(p, b), tail)
        case FNext{previous}:
          Fin.shift(b, Nat.mul(p, b), Fin.pack(p, b, (previous, tail)))

def Fin.unpack.lift(-a: Nat, -b: Nat, coordinate: Fin(a) & Fin(b)) -> Fin(1n+a) & Fin(b):
  (head, tail) = coordinate
  (FNext{head}, tail)

def Fin.unpack.step(-a: Nat, -b: Nat, rec: Fin(Nat.mul(a, b)) -> Fin(a) & Fin(b), split: Either<Fin(b), Fin(Nat.mul(a, b))>) -> Fin(1n+a) & Fin(b):
  match split:
    case Inl{tail}:
      (FZero{}, tail)
    case Inr{index}:
      Fin.unpack.lift(a, b, rec(index))

def Fin.unpack(+a: Nat, +b: Nat, index: Fin(Nat.mul(a, b))) -> Fin(a) & Fin(b):
  match a:
    case 0n:
      match index:
    case 1n+p:
      Fin.unpack.step(p, b, index => Fin.unpack(p, b, index), Fin.split(b, Nat.mul(p, b), index))

def numel(+dims: List<&2, Nat>) -> Nat:
  match dims:
    case Nil{}:
      1n
    case head <> tail:
      Nat.mul(head, numel(tail))

type Coordinate.Step<-head: Nat, -tail: List<&2, Nat>> is Data:
  CStep{index: Fin(head), rest: Coordinate(tail)}

def Coordinate(dims: List<&2, Nat>) -> Data:
  match dims:
    case Nil{}:
      Unit
    case head <> tail:
      Coordinate.Step<head, tail>

def flatten(+dims: List<&2, Nat>, coordinate: Coordinate(dims)) -> Fin(numel(dims)):
  match dims:
    case Nil{}:
      FZero{}
    case +head <> +tail:
      CStep{index, rest} = coordinate
      Fin.pack(head, numel(tail), (index, flatten(tail, rest)))

def unflatten.step(-head: Nat, -tail: List<&2, Nat>, rec: Fin(numel(tail)) -> Coordinate(tail), coordinate: Fin(head) & Fin(numel(tail))) -> Coordinate(head <> tail):
  (first, rest) = coordinate
  CStep{first, rec(rest)}

def unflatten(+dims: List<&2, Nat>, index: Fin(numel(dims))) -> Coordinate(dims):
  match dims:
    case Nil{}:
      Unit{}
    case +head <> +tail:
      unflatten.step(head, tail, rest => unflatten(tail, rest), Fin.unpack(head, numel(tail), index))

def row_major(+dims: List<&2, Nat>, coordinate: Coordinate(dims)) -> Nat:
  match dims:
    case Nil{}:
      0n
    case +head <> +tail:
      CStep{index, rest} = coordinate
      Nat.add(Nat.mul(Fin.value(head, index), numel(tail)), row_major(tail, rest))

def indices(+dims: List<&2, Nat>, coordinate: Coordinate(dims)) -> List<&2, Nat>:
  match dims:
    case Nil{}:
      Nil{}
    case +head <> +tail:
      CStep{index, rest} = coordinate
      Fin.value(head, index) <> indices(tail, rest)

def broadcast_index(size: Nat, index: Nat) -> Nat:
  match size:
    case 0n:
      index
    case 1n+p:
      match p:
        case 0n:
          0n
        case 1n+q:
          index

def broadcast_indices(dims: List<&2, Nat>, values: List<&2, Nat>) -> List<&2, Nat>:
  match dims values:
    case Nil{} _:
      Nil{}
    case dh <> dt Nil{}:
      Nil{}
    case dh <> dt vh <> vt:
      broadcast_index(dh, vh) <> broadcast_indices(dt, vt)

def project_indices(+dims: List<&2, Nat>, +values: List<&2, Nat>) -> List<&2, Nat>:
  broadcast_indices(dims, List.drop(&2, Nat, values, Nat.sub(List.length(&2, Nat, values), List.length(&2, Nat, dims))))

type Axis<-left: Nat, -right: Nat, -output: Nat> is Data:
  AxisEqual{left_eq: {output == left : Nat}, right_eq: {output == right : Nat}}
  AxisLeftOne{one: {1n == left : Nat}, right_eq: {output == right : Nat}}
  AxisRightOne{left_eq: {output == left : Nat}, one: {1n == right : Nat}}

def Axis.project_left(-left: Nat, -right: Nat, -output: Nat, alignment: Axis<left, right, output>, index: Fin(output)) -> Fin(left):
  match alignment:
    case AxisEqual{left_eq, right_eq}:
      %left_eq : Fin(_)
      index
    case AxisLeftOne{one, right_eq}:
      %one : Fin(_)
      FZero{}
    case AxisRightOne{left_eq, one}:
      %left_eq : Fin(_)
      index

def Axis.project_right(-left: Nat, -right: Nat, -output: Nat, alignment: Axis<left, right, output>, index: Fin(output)) -> Fin(right):
  match alignment:
    case AxisEqual{left_eq, right_eq}:
      %right_eq : Fin(_)
      index
    case AxisLeftOne{one, right_eq}:
      %right_eq : Fin(_)
      index
    case AxisRightOne{left_eq, one}:
      %one : Fin(_)
      FZero{}

type Aligned.Step<-lh: Nat, -lt: List<&2, Nat>, -rh: Nat, -rt: List<&2, Nat>, -oh: Nat, -ot: List<&2, Nat>> is Data:
  AStep{axis: Axis<lh, rh, oh>, tail: Aligned(lt, rt, ot)}

def Aligned(left: List<&2, Nat>, right: List<&2, Nat>, output: List<&2, Nat>) -> Data:
  match left right output:
    case Nil{} Nil{} Nil{}:
      Unit
    case lh <> lt rh <> rt oh <> ot:
      Aligned.Step<lh, lt, rh, rt, oh, ot>
    case Nil{} Nil{} oh <> ot:
      Empty
    case Nil{} rh <> rt Nil{}:
      Empty
    case lh <> lt Nil{} Nil{}:
      Empty
    case Nil{} rh <> rt oh <> ot:
      Empty
    case lh <> lt Nil{} oh <> ot:
      Empty
    case lh <> lt rh <> rt Nil{}:
      Empty

def Aligned.project_left(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, alignment: Aligned(left, right, output), coordinate: Coordinate(output)) -> Coordinate(left):
  match left right output:
    case Nil{} Nil{} Nil{}:
      Unit{}
    case +lh <> +lt +rh <> +rt +oh <> +ot:
      AStep{axis, tail} = alignment
      CStep{index, rest} = coordinate
      CStep{Axis.project_left(lh, rh, oh, axis, index), Aligned.project_left(lt, rt, ot, tail, rest)}
    case Nil{} Nil{} oh <> ot:
      match alignment:
    case Nil{} rh <> rt Nil{}:
      match alignment:
    case lh <> lt Nil{} Nil{}:
      match alignment:
    case Nil{} rh <> rt oh <> ot:
      match alignment:
    case lh <> lt Nil{} oh <> ot:
      match alignment:
    case lh <> lt rh <> rt Nil{}:
      match alignment:

def Aligned.project_right(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, alignment: Aligned(left, right, output), coordinate: Coordinate(output)) -> Coordinate(right):
  match left right output:
    case Nil{} Nil{} Nil{}:
      Unit{}
    case +lh <> +lt +rh <> +rt +oh <> +ot:
      AStep{axis, tail} = alignment
      CStep{index, rest} = coordinate
      CStep{Axis.project_right(lh, rh, oh, axis, index), Aligned.project_right(lt, rt, ot, tail, rest)}
    case Nil{} Nil{} oh <> ot:
      match alignment:
    case Nil{} rh <> rt Nil{}:
      match alignment:
    case lh <> lt Nil{} Nil{}:
      match alignment:
    case Nil{} rh <> rt oh <> ot:
      match alignment:
    case lh <> lt Nil{} oh <> ot:
      match alignment:
    case lh <> lt rh <> rt Nil{}:
      match alignment:

def pad(count: Nat, dims: List<&2, Nat>) -> List<&2, Nat>:
  match count:
    case 0n:
      dims
    case 1n+rest:
      1n <> pad(rest, dims)

def unpad(+count: Nat, +dims: List<&2, Nat>, coordinate: Coordinate(pad(count, dims))) -> Coordinate(dims):
  match count:
    case 0n:
      coordinate
    case 1n+rest:
      CStep{index, tail} = coordinate
      unpad(rest, dims, tail)

type PadMinimal<-left: Nat, -right: Nat> is Data:
  PLeftZero{zero: {0n == left : Nat}}
  PRightZero{zero: {0n == right : Nat}}

type Broadcast<-left: List<&2, Nat>, -right: List<&2, Nat>, -output: List<&2, Nat>> is Data:
  BAlign{left_padding: Nat, right_padding: Nat, minimal: PadMinimal<left_padding, right_padding>, aligned: Aligned(pad(left_padding, left), pad(right_padding, right), output)}

def project_left(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +alignment: Broadcast<left, right, output>, coordinate: Coordinate(output)) -> Coordinate(left):
  BAlign{left_padding, right_padding, minimal, aligned} = alignment
  unpad(left_padding, left, Aligned.project_left(pad(left_padding, left), pad(right_padding, right), output, aligned, coordinate))

def project_right(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +alignment: Broadcast<left, right, output>, coordinate: Coordinate(output)) -> Coordinate(right):
  BAlign{left_padding, right_padding, minimal, aligned} = alignment
  unpad(right_padding, right, Aligned.project_right(pad(left_padding, left), pad(right_padding, right), output, aligned, coordinate))

def offset(+dims: List<&2, Nat>, values: List<&2, Nat>) -> Nat:
  match dims values:
    case head <> tail value <> rest:
      Nat.add(Nat.mul(value, numel(tail)), offset(tail, rest))
    case dims values:
      0n

def digits(+dims: List<&2, Nat>, +index: Nat) -> List<&2, Nat>:
  match dims:
    case Nil{}:
      Nil{}
    case head <> tail:
      Nat.div(index, numel(tail)) <> digits(tail, Nat.mod(index, numel(tail)))

def fit(dims: List<&2, Nat>, values: List<&2, Nat>) -> Bool:
  match dims values:
    case Nil{} Nil{}:
      True{}
    case dh <> dt vh <> vt:
      Bool.and(Nat.is_lt(vh, dh), fit(dt, vt))
    case Nil{} vh <> vt:
      False{}
    case dh <> dt Nil{}:
      False{}

# One row per output axis: the output stride and the left and right input
# strides, zero on a padded or singleton input axis.
type Row is Data:
  Row{stride: Nat, left: Nat, right: Nat}

def index_rows(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>) -> List<&2, Row>:
  match left right output:
    case lh <> lt rh <> rt oh <> ot:
      Row{numel(ot), broadcast_index(lh, numel(lt)), broadcast_index(rh, numel(rt))} <> index_rows(lt, rt, ot)
    case left right output:
      Nil{}

def index_table(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>) -> List<&2, Row>:
  index_rows(pad(Nat.sub(List.length(&2, Nat, output), List.length(&2, Nat, left)), left), pad(Nat.sub(List.length(&2, Nat, output), List.length(&2, Nat, right)), right), output)

def rank_dimensions(dims: List<&2, U32>) -> Nat:
  match dims:
    case Nil{}:
      0n
    case head <> tail:
      1n+rank_dimensions(tail)

def naturals(dims: List<&2, U32>) -> List<&2, Nat>:
  match dims:
    case Nil{}:
      Nil{}
    case head <> tail:
      U32.to_nat(head) <> naturals(tail)

def words(values: List<&2, Nat>) -> List<&2, U32>:
  match values:
    case Nil{}:
      Nil{}
    case head <> tail:
      U32.from_nat(head) <> words(tail)

def logical_numel(dims: List<&2, U32>) -> Nat:
  numel(naturals(dims))

# numel_words, horner and digits_words run on checked coordinates or below a
# nonzero element count only: every extent is then at least 1, so each product
# is at most the shape's element count.
def numel_words(dims: List<&2, U32>) -> Nat:
  match dims:
    case Nil{}:
      1n
    case head <> tail:
      Nat.mul(U32.to_nat(head), numel_words(tail))

def horner(dims: List<&2, U32>, values: List<&2, U32>, acc: Nat) -> Nat:
  match dims values:
    case head <> tail value <> rest:
      horner(tail, rest, Nat.add(Nat.mul(acc, U32.to_nat(head)), U32.to_nat(value)))
    case dims values:
      acc

def digits_words(dims: List<&2, U32>, +index: Nat) -> List<&2, Nat>:
  match dims:
    case Nil{}:
      Nil{}
    case head <> +tail:
      +stride = numel_words(tail)
      Nat.div(index, stride) <> digits_words(tail, Nat.mod(index, stride))

def fit_words(dims: List<&2, U32>, values: List<&2, U32>) -> Bool:
  match dims values:
    case Nil{} Nil{}:
      True{}
    case dh <> dt vh <> vt:
      Bool.and(Nat.is_lt(U32.to_nat(vh), U32.to_nat(dh)), fit_words(dt, vt))
    case Nil{} vh <> vt:
      False{}
    case dh <> dt Nil{}:
      False{}

def fit_values(dims: List<&2, U32>, values: List<&2, Nat>) -> Bool:
  match dims values:
    case Nil{} Nil{}:
      True{}
    case dh <> dt vh <> vt:
      Bool.and(Nat.is_lt(vh, U32.to_nat(dh)), fit_values(dt, vt))
    case Nil{} vh <> vt:
      False{}
    case dh <> dt Nil{}:
      False{}

def projected_words(dims: List<&2, U32>, values: List<&2, U32>) -> List<&2, Nat>:
  match dims values:
    case head <> tail value <> rest:
      broadcast_index(U32.to_nat(head), U32.to_nat(value)) <> projected_words(tail, rest)
    case dims values:
      Nil{}

def walk_left(rows: List<&2, Row>, +index: Nat) -> Nat:
  match rows:
    case Nil{}:
      0n
    case Row{+stride, left, right} <> rest:
      Nat.add(Nat.mul(left, Nat.div(index, stride)), walk_left(rest, Nat.mod(index, stride)))

def walk_right(rows: List<&2, Row>, +index: Nat) -> Nat:
  match rows:
    case Nil{}:
      0n
    case Row{+stride, left, right} <> rest:
      Nat.add(Nat.mul(right, Nat.div(index, stride)), walk_right(rest, Nat.mod(index, stride)))

def fit_values_spec(+dims: List<&2, U32>, +values: List<&2, Nat>) -> {fit_values(dims, values) == fit(naturals(dims), values) : Bool}:
  match dims values:
    case Nil{} Nil{}:
      {==}
    case +dh <> +dt +vh <> +vt:
      Equal.cong(Bool, Bool, rest => Bool.and(Nat.is_lt(vh, U32.to_nat(dh)), rest), fit_values(dt, vt), fit(naturals(dt), vt), fit_values_spec(dt, vt))
    case Nil{} vh <> vt:
      {==}
    case dh <> dt Nil{}:
      {==}

def fit_values_true(+dims: List<&2, U32>, +values: List<&2, Nat>, equal: {True{} == fit_values(dims, values) : Bool}) -> {True{} == fit(naturals(dims), values) : Bool}:
  %fit_values_spec(dims, values) : {True{} == _ : Bool}
  equal

def index_rows_words(left: List<&2, U32>, right: List<&2, U32>, output: List<&2, U32>) -> List<&2, Row>:
  match left right output:
    case lh <> +lt rh <> +rt oh <> +ot:
      Row{numel_words(ot), broadcast_index(U32.to_nat(lh), numel_words(lt)), broadcast_index(U32.to_nat(rh), numel_words(rt))} <> index_rows_words(lt, rt, ot)
    case left right output:
      Nil{}

def numel_words_spec(+dims: List<&2, U32>) -> {numel_words(dims) == numel(naturals(dims)) : Nat}:
  match dims:
    case Nil{}:
      {==}
    case +head <> +tail:
      Equal.cong(Nat, Nat, value => Nat.mul(U32.to_nat(head), value), numel_words(tail), numel(naturals(tail)), numel_words_spec(tail))

def index_rows_spec(+left: List<&2, U32>, +right: List<&2, U32>, +output: List<&2, U32>) -> {index_rows_words(left, right, output) == index_rows(naturals(left), naturals(right), naturals(output)) : List<&2, Row>}:
  match left right output:
    case +lh <> +lt +rh <> +rt +oh <> +ot:
      %Equal.sym(Nat, numel_words(ot), numel(naturals(ot)), numel_words_spec(ot)) : {Row{_, broadcast_index(U32.to_nat(lh), numel_words(lt)), broadcast_index(U32.to_nat(rh), numel_words(rt))} <> index_rows_words(lt, rt, ot) == index_rows(naturals(lh <> lt), naturals(rh <> rt), naturals(oh <> ot)) : List<&2, Row>}
      %Equal.sym(Nat, numel_words(lt), numel(naturals(lt)), numel_words_spec(lt)) : {Row{numel(naturals(ot)), broadcast_index(U32.to_nat(lh), _), broadcast_index(U32.to_nat(rh), numel_words(rt))} <> index_rows_words(lt, rt, ot) == index_rows(naturals(lh <> lt), naturals(rh <> rt), naturals(oh <> ot)) : List<&2, Row>}
      %Equal.sym(Nat, numel_words(rt), numel(naturals(rt)), numel_words_spec(rt)) : {Row{numel(naturals(ot)), broadcast_index(U32.to_nat(lh), numel(naturals(lt))), broadcast_index(U32.to_nat(rh), _)} <> index_rows_words(lt, rt, ot) == index_rows(naturals(lh <> lt), naturals(rh <> rt), naturals(oh <> ot)) : List<&2, Row>}
      %Equal.sym(List<&2, Row>, index_rows_words(lt, rt, ot), index_rows(naturals(lt), naturals(rt), naturals(ot)), index_rows_spec(lt, rt, ot)) : {Row{numel(naturals(ot)), broadcast_index(U32.to_nat(lh), numel(naturals(lt))), broadcast_index(U32.to_nat(rh), numel(naturals(rt)))} <> _ == index_rows(naturals(lh <> lt), naturals(rh <> rt), naturals(oh <> ot)) : List<&2, Row>}
      {==}
    case Nil{} right output:
      {==}
    case lh <> lt Nil{} output:
      {==}
    case lh <> lt rh <> rt Nil{}:
      {==}

def rank_naturals(+dims: List<&2, U32>) -> {rank_dimensions(dims) == List.length(&2, Nat, naturals(dims)) : Nat}:
  match dims:
    case Nil{}:
      {==}
    case head <> tail:
      Equal.cong(Nat, Nat, value => 1n+value, rank_dimensions(tail), List.length(&2, Nat, naturals(tail)), rank_naturals(tail))

type BoolCase<-value: Bool> is Data:
  BTrue{equal: {True{} == value : Bool}}
  BFalse{equal: {False{} == value : Bool}}

def bool_case(+value: Bool) -> BoolCase<value>:
  match value:
    case True{}:
      BTrue{{==}}
    case False{}:
      BFalse{{==}}

def truth(value: Bool) -> Data:
  match value:
    case True{}:
      Unit
    case False{}:
      Empty

def clash(equal: {True{} == False{} : Bool}) -> Empty:
  %equal : truth(_)
  Unit{}

def LE.decided(+a: Nat, +b: Nat, decided: {True{} == Nat.is_le(a, b) : Bool}) -> {Unit == LE(a, b) : Data}:
  match a b:
    case 0n b:
      {==}
    case 1n+p 0n:
      Empty.absurd({Unit == Empty : Data}, clash(decided))
    case 1n+p 1n+q:
      LE.decided(p, q, decided)

# The rewrite is erased at runtime, so a decided bound costs one comparison
# instead of a unary walk to 2^20.
def LE.from(+a: Nat, +b: Nat, decided: {True{} == Nat.is_le(a, b) : Bool}) -> LE(a, b):
  %LE.decided(a, b, decided) : _
  Unit{}

def equal_decided(+a: Nat, +b: Nat, decided: {True{} == Nat.is_eq(a, b) : Bool}) -> {a == b : Nat}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      Empty.absurd({0n == 1n+q : Nat}, clash(decided))
    case 1n+p 0n:
      Empty.absurd({1n+p == 0n : Nat}, clash(decided))
    case 1n+p 1n+q:
      %equal_decided(p, q, decided) : {1n+p == 1n+_ : Nat}
      {==}

def equal_from(+a: Nat, +b: Nat, decided: {True{} == Nat.is_eq(a, b) : Bool}) -> {a == b : Nat}:
  %equal_decided(a, b, decided) : {a == _ : Nat}
  {==}

type Capacity<-dims: List<&2, Nat>> is Data:
  Within{size: Nat, exact: {size == numel(dims) : Nat}, bound: {True{} == Nat.is_le(size, 1048576n) : Bool}}
  Beyond{}

def capacity.bound(-dims: List<&2, Nat>, +size: Nat, exact: {size == numel(dims) : Nat}, bound: BoolCase<Nat.is_le(size, 1048576n)>) -> Capacity<dims>:
  match bound:
    case BFalse{equal}:
      Beyond{}
    case BTrue{equal}:
      Within{size, exact, equal}

def capacity.scale(+head: Nat, -tail: List<&2, Nat>, +size: Nat, exact: {size == numel(tail) : Nat}, small: BoolCase<Bool.or(Nat.is_eq(size, 0n), Nat.is_le(head, 1048576n))>) -> Capacity<head <> tail>:
  match small:
    case BFalse{equal}:
      Beyond{}
    case BTrue{equal}:
      +product = Nat.mul(head, size)
      capacity.bound(head <> tail, product, Equal.cong(Nat, Nat, value => Nat.mul(head, value), size, numel(tail), exact), bool_case(Nat.is_le(product, 1048576n)))

def capacity.zero(+head: Nat, -tail: List<&2, Nat>) -> Capacity<head <> tail>:
  match head:
    case 0n:
      Within{0n, {==}, {==}}
    case 1n+p:
      Beyond{}

# A size is formed only when the suffix size is zero or both factors are at
# most 2^20, so no product exceeds 2^40, below the native Nat limit of 2^48.
def capacity.step(+head: Nat, -tail: List<&2, Nat>, rest: Capacity<tail>) -> Capacity<head <> tail>:
  match rest:
    case Beyond{}:
      capacity.zero(head, tail)
    case Within{+size, exact, bound}:
      capacity.scale(head, tail, size, exact, bool_case(Bool.or(Nat.is_eq(size, 0n), Nat.is_le(head, 1048576n))))

def capacity(+dims: List<&2, Nat>) -> Capacity<dims>:
  match dims:
    case Nil{}:
      Within{1n, {==}, {==}}
    case +head <> tail:
      capacity.step(head, tail, capacity(tail))

type Error is Data:
  RankExceeded{}
  CapacityExceeded{}
  CoordinateRankMismatch{}
  CoordinateOutOfBounds{}
  FlatIndexOutOfBounds{}
  IncompatibleBroadcast{}

type Shape is Data:
  SShape{dimensions: List<&2, U32>, size: Nat, exact: {size == logical_numel(dimensions) : Nat}, rank_bound: LE(rank_dimensions(dimensions), 4n), capacity_bound: LE(logical_numel(dimensions), 1048576n)}

def dimensions(shape: Shape) -> List<&2, U32>:
  SShape{dims, size, exact, rank_bound, capacity_bound} = shape
  dims

def elements(shape: Shape) -> Nat:
  SShape{dims, size, exact, rank_bound, capacity_bound} = shape
  size

def extents(shape: Shape) -> List<&2, Nat>:
  naturals(dimensions(shape))

def elements_exact(+shape: Shape) -> {elements(shape) == numel(extents(shape)) : Nat}:
  match shape:
    case SShape{dims, size, exact, rank_bound, capacity_bound}:
      exact

def rank(shape: Shape) -> U32:
  U32.from_nat(rank_dimensions(dimensions(shape)))

def count(shape: Shape) -> U32:
  U32.from_nat(elements(shape))

type ConstructResult<-input: List<&2, U32>> is Data:
  SError{error: Error}
  SBuilt{shape: Shape, preserved: {dimensions(shape) == input : List<&2, U32>}}

def construct.bound(-input: List<&2, U32>, +size: Nat, exact: {size == logical_numel(input) : Nat}, bound: {True{} == Nat.is_le(size, 1048576n) : Bool}) -> LE(logical_numel(input), 1048576n):
  %exact : LE(_, 1048576n)
  LE.from(size, 1048576n, bound)

def construct.capacity(+input: List<&2, U32>, rank_bound: LE(rank_dimensions(input), 4n), capacity: Capacity<naturals(input)>) -> ConstructResult<input>:
  match capacity:
    case Beyond{}:
      SError{CapacityExceeded{}}
    case Within{+size, +exact, bound}:
      SBuilt{SShape{input, size, exact, rank_bound, construct.bound(input, size, exact, bound)}, {==}}

def construct.ranked(+input: List<&2, U32>, decision: BoolCase<Nat.is_le(rank_dimensions(input), 4n)>) -> ConstructResult<input>:
  match decision:
    case BFalse{equal}:
      SError{RankExceeded{}}
    case BTrue{equal}:
      construct.capacity(input, LE.from(rank_dimensions(input), 4n, equal), capacity(naturals(input)))

def construct(+input: List<&2, U32>) -> ConstructResult<input>:
  construct.ranked(input, bool_case(Nat.is_le(rank_dimensions(input), 4n)))

type CheckedCoordinate<-shape: Shape> is Data:
  CValues{values: List<&2, Nat>, valid: {True{} == fit(extents(shape), values) : Bool}}

def coordinate_values(-shape: Shape, coordinate: CheckedCoordinate<shape>) -> List<&2, U32>:
  CValues{values, valid} = coordinate
  words(values)

def coordinate_checked.finish(+shape: Shape, +values: List<&2, Nat>, verdict: BoolCase<fit_values(dimensions(shape), values)>) -> Result<&2, &2, Error, CheckedCoordinate<shape>>:
  match verdict:
    case BFalse{equal}:
      Fail{CoordinateOutOfBounds{}}
    case BTrue{equal}:
      Done{CValues{values, fit_values_true(dimensions(shape), values, equal)}}

def coordinate_checked.nat(+shape: Shape, +values: List<&2, Nat>) -> Result<&2, &2, Error, CheckedCoordinate<shape>>:
  coordinate_checked.finish(shape, values, bool_case(fit_values(dimensions(shape), values)))

type FlatIndex<-shape: Shape> is Data:
  FIndex{value: Nat, bound: {True{} == Nat.is_lt(value, elements(shape)) : Bool}}

def index_value(-shape: Shape, index: FlatIndex<shape>) -> U32:
  FIndex{value, bound} = index
  U32.from_nat(value)

def flat_checked.finish(-shape: Shape, +index: Nat, verdict: BoolCase<Nat.is_lt(index, elements(shape))>) -> Result<&2, &2, Error, FlatIndex<shape>>:
  match verdict:
    case BFalse{equal}:
      Fail{FlatIndexOutOfBounds{}}
    case BTrue{equal}:
      Done{FIndex{index, equal}}

def flat_checked(+shape: Shape, +index: Nat) -> Result<&2, &2, Error, FlatIndex<shape>>:
  flat_checked.finish(shape, index, bool_case(Nat.is_lt(index, elements(shape))))

def flatten_checked.fits(+shape: Shape, +values: List<&2, U32>, verdict: BoolCase<fit_words(dimensions(shape), values)>) -> Result<&2, &2, Error, FlatIndex<shape>>:
  match verdict:
    case BFalse{equal}:
      Fail{CoordinateOutOfBounds{}}
    case BTrue{equal}:
      flat_checked(shape, horner(dimensions(shape), values, 0n))

def flatten_checked.ranked(+shape: Shape, +values: List<&2, U32>, same_rank: BoolCase<Nat.is_eq(rank_dimensions(dimensions(shape)), rank_dimensions(values))>) -> Result<&2, &2, Error, FlatIndex<shape>>:
  match same_rank:
    case BFalse{equal}:
      Fail{CoordinateRankMismatch{}}
    case BTrue{equal}:
      flatten_checked.fits(shape, values, bool_case(fit_words(dimensions(shape), values)))

def flatten_checked(+shape: Shape, values: List<&2, U32>) -> Result<&2, &2, Error, FlatIndex<shape>>:
  +copy = values
  flatten_checked.ranked(shape, copy, bool_case(Nat.is_eq(rank_dimensions(dimensions(shape)), rank_dimensions(copy))))

def unflatten_checked.finish(+shape: Shape, +index: Nat, verdict: BoolCase<Nat.is_lt(index, elements(shape))>) -> Result<&2, &2, Error, CheckedCoordinate<shape>>:
  match verdict:
    case BFalse{equal}:
      Fail{FlatIndexOutOfBounds{}}
    case BTrue{equal}:
      coordinate_checked.nat(shape, digits_words(dimensions(shape), index))

def unflatten_checked(+shape: Shape, index: U32) -> Result<&2, &2, Error, CheckedCoordinate<shape>>:
  +value = U32.to_nat(index)
  unflatten_checked.finish(shape, value, bool_case(Nat.is_lt(value, elements(shape))))

type AxisChoice<-left: U32, -right: U32> is Data:
  Picks{output: U32, axis: Axis<U32.to_nat(left), U32.to_nat(right), U32.to_nat(output)>}
  Clashes{}

def axis_choice.right_one(+left: U32, +right: U32, one: BoolCase<Nat.is_eq(1n, U32.to_nat(right))>) -> AxisChoice<left, right>:
  match one:
    case BFalse{equal}:
      Clashes{}
    case BTrue{equal}:
      Picks{left, AxisRightOne{{==}, equal_from(1n, U32.to_nat(right), equal)}}

def axis_choice.left_one(+left: U32, +right: U32, one: BoolCase<Nat.is_eq(1n, U32.to_nat(left))>) -> AxisChoice<left, right>:
  match one:
    case BFalse{equal}:
      axis_choice.right_one(left, right, bool_case(Nat.is_eq(1n, U32.to_nat(right))))
    case BTrue{equal}:
      Picks{right, AxisLeftOne{equal_from(1n, U32.to_nat(left), equal), {==}}}

def axis_choice.equal(+left: U32, +right: U32, same: BoolCase<Nat.is_eq(U32.to_nat(left), U32.to_nat(right))>) -> AxisChoice<left, right>:
  match same:
    case BFalse{equal}:
      axis_choice.left_one(left, right, bool_case(Nat.is_eq(1n, U32.to_nat(left))))
    case BTrue{equal}:
      Picks{left, AxisEqual{{==}, equal_from(U32.to_nat(left), U32.to_nat(right), equal)}}

def axis_choice(+left: U32, +right: U32) -> AxisChoice<left, right>:
  axis_choice.equal(left, right, bool_case(Nat.is_eq(U32.to_nat(left), U32.to_nat(right))))

type Alignment<-left: List<&2, U32>, -right: List<&2, U32>> is Data:
  Aligns{output: List<&2, U32>, aligned: Aligned(naturals(left), naturals(right), naturals(output))}
  Misaligned{}

def align.finish(-lh: U32, -lt: List<&2, U32>, -rh: U32, -rt: List<&2, U32>, output: U32, axis: Axis<U32.to_nat(lh), U32.to_nat(rh), U32.to_nat(output)>, rest: Alignment<lt, rt>) -> Alignment<lh <> lt, rh <> rt>:
  match rest:
    case Misaligned{}:
      Misaligned{}
    case Aligns{tail, aligned}:
      Aligns{output <> tail, AStep{axis, aligned}}

def align.step(-lh: U32, -lt: List<&2, U32>, -rh: U32, -rt: List<&2, U32>, rest: Alignment<lt, rt>, choice: AxisChoice<lh, rh>) -> Alignment<lh <> lt, rh <> rt>:
  match choice:
    case Clashes{}:
      Misaligned{}
    case Picks{output, axis}:
      align.finish(lh, lt, rh, rt, output, axis, rest)

def align(+left: List<&2, U32>, +right: List<&2, U32>) -> Alignment<left, right>:
  match left right:
    case Nil{} Nil{}:
      Aligns{Nil{}, Unit{}}
    case +lh <> +lt +rh <> +rt:
      align.step(lh, lt, rh, rt, align(lt, rt), axis_choice(lh, rh))
    case Nil{} rh <> rt:
      Misaligned{}
    case lh <> lt Nil{}:
      Misaligned{}

def pad_words(count: Nat, dims: List<&2, U32>) -> List<&2, U32>:
  match count:
    case 0n:
      dims
    case 1n+rest:
      1 <> pad_words(rest, dims)

def naturals_pad(+count: Nat, -dims: List<&2, U32>) -> {naturals(pad_words(count, dims)) == pad(count, naturals(dims)) : List<&2, Nat>}:
  match count:
    case 0n:
      {==}
    case 1n+rest:
      %naturals_pad(rest, dims) : {1n <> naturals(pad_words(rest, dims)) == 1n <> _ : List<&2, Nat>}
      {==}

def minimal(+left: Nat, +right: Nat) -> PadMinimal<Nat.sub(right, left), Nat.sub(left, right)>:
  match left right:
    case 0n 0n:
      PLeftZero{{==}}
    case 0n 1n+q:
      PRightZero{{==}}
    case 1n+p 0n:
      PLeftZero{{==}}
    case 1n+p 1n+q:
      minimal(p, q)

type Index<-left: List<&2, Nat>, -right: List<&2, Nat>, -output: List<&2, Nat>> is Data:
  Indexed{rows: List<&2, Row>, exact: {rows == index_table(left, right, output) : List<&2, Row>}}
  Unindexed{zero: {0n == numel(output) : Nat}}

def index_of.rows(+left: Shape, +right: Shape, +output: Shape, +rank: Nat) -> List<&2, Row>:
  index_rows_words(pad_words(Nat.sub(rank, rank_dimensions(dimensions(left))), dimensions(left)), pad_words(Nat.sub(rank, rank_dimensions(dimensions(right))), dimensions(right)), dimensions(output))

def index_of.exact(+left: Shape, +right: Shape, +output: Shape) -> {index_of.rows(left, right, output, rank_dimensions(dimensions(output))) == index_table(extents(left), extents(right), extents(output)) : List<&2, Row>}:
  +lp = Nat.sub(rank_dimensions(dimensions(output)), rank_dimensions(dimensions(left)))
  +rp = Nat.sub(rank_dimensions(dimensions(output)), rank_dimensions(dimensions(right)))
  %Equal.sym(List<&2, Row>, index_rows_words(pad_words(lp, dimensions(left)), pad_words(rp, dimensions(right)), dimensions(output)), index_rows(naturals(pad_words(lp, dimensions(left))), naturals(pad_words(rp, dimensions(right))), extents(output)), index_rows_spec(pad_words(lp, dimensions(left)), pad_words(rp, dimensions(right)), dimensions(output))) : {_ == index_table(extents(left), extents(right), extents(output)) : List<&2, Row>}
  %Equal.sym(List<&2, Nat>, naturals(pad_words(lp, dimensions(left))), pad(lp, extents(left)), naturals_pad(lp, dimensions(left))) : {index_rows(_, naturals(pad_words(rp, dimensions(right))), extents(output)) == index_table(extents(left), extents(right), extents(output)) : List<&2, Row>}
  %Equal.sym(List<&2, Nat>, naturals(pad_words(rp, dimensions(right))), pad(rp, extents(right)), naturals_pad(rp, dimensions(right))) : {index_rows(pad(lp, extents(left)), _, extents(output)) == index_table(extents(left), extents(right), extents(output)) : List<&2, Row>}
  %Equal.sym(Nat, rank_dimensions(dimensions(output)), List.length(&2, Nat, extents(output)), rank_naturals(dimensions(output))) : {index_rows(pad(Nat.sub(_, rank_dimensions(dimensions(left))), extents(left)), pad(Nat.sub(_, rank_dimensions(dimensions(right))), extents(right)), extents(output)) == index_table(extents(left), extents(right), extents(output)) : List<&2, Row>}
  %Equal.sym(Nat, rank_dimensions(dimensions(left)), List.length(&2, Nat, extents(left)), rank_naturals(dimensions(left))) : {index_rows(pad(Nat.sub(List.length(&2, Nat, extents(output)), _), extents(left)), pad(Nat.sub(List.length(&2, Nat, extents(output)), rank_dimensions(dimensions(right))), extents(right)), extents(output)) == index_table(extents(left), extents(right), extents(output)) : List<&2, Row>}
  %Equal.sym(Nat, rank_dimensions(dimensions(right)), List.length(&2, Nat, extents(right)), rank_naturals(dimensions(right))) : {index_rows(pad(Nat.sub(List.length(&2, Nat, extents(output)), List.length(&2, Nat, extents(left))), extents(left)), pad(Nat.sub(List.length(&2, Nat, extents(output)), _), extents(right)), extents(output)) == index_table(extents(left), extents(right), extents(output)) : List<&2, Row>}
  {==}

# Index rows are formed only below a nonzero output count. Every padded input
# extent is then 1 or the output extent, so no product exceeds that count.
def index_of(+left: Shape, +right: Shape, +output: Shape, empty: BoolCase<Nat.is_eq(elements(output), 0n)>) -> Index<extents(left), extents(right), extents(output)>:
  match empty:
    case BTrue{equal}:
      Unindexed{Equal.trans(Nat, 0n, elements(output), numel(extents(output)), Equal.sym(Nat, elements(output), 0n, equal_from(elements(output), 0n, equal)), elements_exact(output))}
    case BFalse{equal}:
      Indexed{index_of.rows(left, right, output, rank_dimensions(dimensions(output))), index_of.exact(left, right, output)}

type BroadcastPlan is Data:
  BPlan{left: Shape, right: Shape, output: Shape, alignment: Broadcast<extents(left), extents(right), extents(output)>}

def left_shape(plan: BroadcastPlan) -> Shape:
  BPlan{left, right, output, alignment} = plan
  left

def right_shape(plan: BroadcastPlan) -> Shape:
  BPlan{left, right, output, alignment} = plan
  right

def output_shape(plan: BroadcastPlan) -> Shape:
  BPlan{left, right, output, alignment} = plan
  output

def alignment(+plan: BroadcastPlan) -> Broadcast<extents(left_shape(plan)), extents(right_shape(plan)), extents(output_shape(plan))>:
  match plan:
    case BPlan{left, right, output, alignment}:
      alignment

def index_map(+plan: BroadcastPlan) -> Index<extents(left_shape(plan)), extents(right_shape(plan)), extents(output_shape(plan))>:
  +output = output_shape(plan)
  index_of(left_shape(plan), right_shape(plan), output, bool_case(Nat.is_eq(elements(output), 0n)))

def broadcast_checked.realign(-left: List<&2, Nat>, -right: List<&2, Nat>, -shape: Shape, -output: List<&2, U32>, preserved: {dimensions(shape) == output : List<&2, U32>}, aligned: Aligned(left, right, naturals(output))) -> Aligned(left, right, extents(shape)):
  %Equal.sym(List<&2, U32>, dimensions(shape), output, preserved) : Aligned(left, right, naturals(_))
  aligned

def broadcast_checked.construct(+left: Shape, +right: Shape, left_padding: Nat, right_padding: Nat, minimal: PadMinimal<left_padding, right_padding>, -output: List<&2, U32>, aligned: Aligned(pad(left_padding, extents(left)), pad(right_padding, extents(right)), naturals(output)), result: ConstructResult<output>) -> Result<&2, &2, Error, BroadcastPlan>:
  match result:
    case SError{error}:
      Fail{error}
    case SBuilt{shape, preserved}:
      Done{BPlan{left, right, shape, BAlign{left_padding, right_padding, minimal, broadcast_checked.realign(pad(left_padding, extents(left)), pad(right_padding, extents(right)), shape, output, preserved, aligned)}}}

def broadcast_checked.repad(+left_padding: Nat, +right_padding: Nat, -left: List<&2, U32>, -right: List<&2, U32>, -output: List<&2, U32>, aligned: Aligned(naturals(pad_words(left_padding, left)), naturals(pad_words(right_padding, right)), naturals(output))) -> Aligned(pad(left_padding, naturals(left)), pad(right_padding, naturals(right)), naturals(output)):
  %naturals_pad(left_padding, left) : Aligned(_, pad(right_padding, naturals(right)), naturals(output))
  %naturals_pad(right_padding, right) : Aligned(naturals(pad_words(left_padding, left)), _, naturals(output))
  aligned

def broadcast_checked.align(+left: Shape, +right: Shape, +left_padding: Nat, +right_padding: Nat, minimal: PadMinimal<left_padding, right_padding>, alignment: Alignment<pad_words(left_padding, dimensions(left)), pad_words(right_padding, dimensions(right))>) -> Result<&2, &2, Error, BroadcastPlan>:
  match alignment:
    case Misaligned{}:
      Fail{IncompatibleBroadcast{}}
    case Aligns{+output, aligned}:
      broadcast_checked.construct(left, right, left_padding, right_padding, minimal, output, broadcast_checked.repad(left_padding, right_padding, dimensions(left), dimensions(right), output, aligned), construct(output))

def broadcast_checked.ranked(+left: Shape, +right: Shape, +left_rank: Nat, +right_rank: Nat) -> Result<&2, &2, Error, BroadcastPlan>:
  broadcast_checked.align(left, right, Nat.sub(right_rank, left_rank), Nat.sub(left_rank, right_rank), minimal(left_rank, right_rank), align(pad_words(Nat.sub(right_rank, left_rank), dimensions(left)), pad_words(Nat.sub(left_rank, right_rank), dimensions(right))))

def broadcast_checked(+left: Shape, +right: Shape) -> Result<&2, &2, Error, BroadcastPlan>:
  broadcast_checked.ranked(left, right, rank_dimensions(dimensions(left)), rank_dimensions(dimensions(right)))

def projection_checked.fits(+input: Shape, -output: Shape, +values: List<&2, U32>, verdict: BoolCase<fit_words(dimensions(output), values)>) -> Result<&2, &2, Error, CheckedCoordinate<input>>:
  match verdict:
    case BFalse{equal}:
      Fail{CoordinateOutOfBounds{}}
    case BTrue{equal}:
      coordinate_checked.nat(input, projected_words(dimensions(input), List.drop(&2, U32, values, Nat.sub(rank_dimensions(values), rank_dimensions(dimensions(input))))))

def projection_checked.ranked(+input: Shape, +output: Shape, +values: List<&2, U32>, same_rank: BoolCase<Nat.is_eq(rank_dimensions(dimensions(output)), rank_dimensions(values))>) -> Result<&2, &2, Error, CheckedCoordinate<input>>:
  match same_rank:
    case BFalse{equal}:
      Fail{CoordinateRankMismatch{}}
    case BTrue{equal}:
      projection_checked.fits(input, output, values, bool_case(fit_words(dimensions(output), values)))

def projection_checked(+input: Shape, +output: Shape, values: List<&2, U32>) -> Result<&2, &2, Error, CheckedCoordinate<input>>:
  +copy = values
  projection_checked.ranked(input, output, copy, bool_case(Nat.is_eq(rank_dimensions(dimensions(output)), rank_dimensions(copy))))

def project_left_checked(+plan: BroadcastPlan, values: List<&2, U32>) -> Result<&2, &2, Error, CheckedCoordinate<left_shape(plan)>>:
  projection_checked(left_shape(plan), output_shape(plan), values)

def project_right_checked(+plan: BroadcastPlan, values: List<&2, U32>) -> Result<&2, &2, Error, CheckedCoordinate<right_shape(plan)>>:
  projection_checked(right_shape(plan), output_shape(plan), values)

def error_code(error: Error) -> U32:
  match error:
    case RankExceeded{}:
      1
    case CapacityExceeded{}:
      2
    case CoordinateRankMismatch{}:
      3
    case CoordinateOutOfBounds{}:
      4
    case FlatIndexOutOfBounds{}:
      5
    case IncompatibleBroadcast{}:
      6
