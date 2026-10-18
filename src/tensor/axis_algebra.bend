import Base
import ../shape/core.bend as Shape
import ../shape/nat_proofs.bend as Nats
import ../shape/broadcast_proofs.bend as Broadcast
import ../storage/arith.bend as Arith
import ./layout.bend as Layout
import ./axis_spec.bend as Axis
import ./word_conversion.bend as Conversion

def lookup(+values: List<&2, U32>, +index: Nat) -> {Layout.lookup(values, index) == Axis.lookup(U32, values, index) : Maybe<&2, U32>}:
  match values index:
    case Nil{} 0n:
      {==}
    case Nil{} 1n+p:
      {==}
    case head <> tail 0n:
      {==}
    case head <> +tail 1n+p:
      lookup(tail, p)

def reorder_step(+head: Maybe<&2, U32>, +tail: Maybe<&2, List<&2, U32>>) ->
  {Layout.reorder_head(value => Layout.reorder_tail(value, tail), head) == Axis.prepend(U32, head, tail) : Maybe<&2, List<&2, U32>>}:
  match head tail:
    case None{} None{}:
      {==}
    case None{} Some{items}:
      {==}
    case Some{value} None{}:
      {==}
    case Some{value} Some{items}:
      {==}

def reorder(+axes: List<&2, U32>, +values: List<&2, U32>) ->
  {Layout.reorder(values, axes) == Axis.gather(U32, values, Shape.naturals(axes)) : Maybe<&2, List<&2, U32>>}:
  match axes:
    case Nil{}:
      {==}
    case +axis <> +tail:
      %Equal.sym(Maybe<&2, U32>, Layout.lookup(values, U32.to_nat(axis)), Axis.lookup(U32, values, U32.to_nat(axis)), lookup(values, U32.to_nat(axis))) :
        {Layout.reorder_head(value => Layout.reorder_tail(value, Layout.reorder(values, tail)), _) == Axis.prepend(U32, Axis.lookup(U32, values, U32.to_nat(axis)), Axis.gather(U32, values, Shape.naturals(tail))) : Maybe<&2, List<&2, U32>>}
      %Equal.sym(Maybe<&2, List<&2, U32>>, Layout.reorder(values, tail), Axis.gather(U32, values, Shape.naturals(tail)), reorder(tail, values)) :
        {Layout.reorder_head(value => Layout.reorder_tail(value, _), Axis.lookup(U32, values, U32.to_nat(axis))) == Axis.prepend(U32, Axis.lookup(U32, values, U32.to_nat(axis)), Axis.gather(U32, values, Shape.naturals(tail))) : Maybe<&2, List<&2, U32>>}
      reorder_step(Axis.lookup(U32, values, U32.to_nat(axis)), Axis.gather(U32, values, Shape.naturals(tail)))

def u32_eq(+a: U32, +b: U32) -> {U32.is_eq(a, b) == Nat.is_eq(U32.to_nat(a), U32.to_nat(b)) : Bool}:
  match a b:
    case U32{+x} U32{+y}:
      Equal.sym(Bool, Cmp.is_eq(Nat.cmp(Word.to_nat(32n, x), Word.to_nat(32n, y))), Cmp.is_eq(Word.cmp(32n, x, y)),
        Equal.cong(Cmp, Bool, Cmp.is_eq, Nat.cmp(Word.to_nat(32n, x), Word.to_nat(32n, y)), Word.cmp(32n, x, y), Arith.word_cmp(32n, x, y)))

def lookup_empty(+index: Nat) -> {Axis.lookup(U32, Nil{}, index) == None{} : Maybe<&2, U32>}:
  match index:
    case 0n:
      {==}
    case 1n+p:
      {==}

def select_empty(+index: Maybe<&2, Nat>) -> {Axis.select(U32, Nil{}, index) == None{} : Maybe<&2, U32>}:
  match index:
    case None{}:
      {==}
    case Some{+value}:
      lookup_empty(value)

def select_next(-head: U32, +tail: List<&2, U32>, +index: Maybe<&2, Nat>) ->
  {Axis.select(U32, head <> tail, Axis.position_next(index)) == Axis.select(U32, tail, index) : Maybe<&2, U32>}:
  match index:
    case None{}:
      {==}
    case Some{value}:
      {==}

def coordinate_step(+decision: Bool, +head: U32, +tail: List<&2, U32>, +index: Maybe<&2, Nat>) ->
  {Bool.pick(Maybe<&2, U32>, decision, Some{head}, Axis.select(U32, tail, index)) ==
    Axis.select(U32, head <> tail, Axis.position_decided(decision, index)) : Maybe<&2, U32>}:
  match decision:
    case True{}:
      {==}
    case False{}:
      Equal.sym(Maybe<&2, U32>, Axis.select(U32, head <> tail, Axis.position_next(index)), Axis.select(U32, tail, index), select_next(head, tail, index))

def coordinate_for(+axes: List<&2, U32>, +coordinates: List<&2, U32>, +axis: U32) ->
  {Layout.coordinate_for(axes, coordinates, axis) == Axis.select(U32, coordinates, Axis.position(Shape.naturals(axes), U32.to_nat(axis))) : Maybe<&2, U32>}:
  match axes coordinates:
    case Nil{} Nil{}:
      {==}
    case Nil{} head <> tail:
      {==}
    case +head <> +tail Nil{}:
      Equal.sym(Maybe<&2, U32>, Axis.select(U32, Nil{}, Axis.position(Shape.naturals(head <> tail), U32.to_nat(axis))), None{},
        select_empty(Axis.position(Shape.naturals(head <> tail), U32.to_nat(axis))))
    case +head <> +tail +value <> +rest:
      %Equal.sym(Maybe<&2, U32>, Layout.coordinate_for(tail, rest, axis), Axis.select(U32, rest, Axis.position(Shape.naturals(tail), U32.to_nat(axis))), coordinate_for(tail, rest, axis)) :
        {Bool.pick(Maybe<&2, U32>, U32.is_eq(head, axis), Some{value}, _) == Axis.select(U32, value <> rest, Axis.position(Shape.naturals(head <> tail), U32.to_nat(axis))) : Maybe<&2, U32>}
      %Equal.sym(Bool, U32.is_eq(head, axis), Nat.is_eq(U32.to_nat(head), U32.to_nat(axis)), u32_eq(head, axis)) :
        {Bool.pick(Maybe<&2, U32>, _, Some{value}, Axis.select(U32, rest, Axis.position(Shape.naturals(tail), U32.to_nat(axis)))) == Axis.select(U32, value <> rest, Axis.position(Shape.naturals(head <> tail), U32.to_nat(axis))) : Maybe<&2, U32>}
      coordinate_step(Nat.is_eq(U32.to_nat(head), U32.to_nat(axis)), value, rest, Axis.position(Shape.naturals(tail), U32.to_nat(axis)))

def contains(+axes: List<&2, U32>, +axis: U32) -> {Layout.contains(axis, axes) == Axis.contains(U32.to_nat(axis), Shape.naturals(axes)) : Bool}:
  match axes:
    case Nil{}:
      {==}
    case +head <> +tail:
      %Equal.sym(Bool, U32.is_eq(axis, head), Nat.is_eq(U32.to_nat(axis), U32.to_nat(head)), u32_eq(axis, head)) :
        {Bool.or(_, Layout.contains(axis, tail)) == Axis.contains(U32.to_nat(axis), Shape.naturals(head <> tail)) : Bool}
      %Equal.sym(Bool, Layout.contains(axis, tail), Axis.contains(U32.to_nat(axis), Shape.naturals(tail)), contains(tail, axis)) :
        {Bool.or(Nat.is_eq(U32.to_nat(axis), U32.to_nat(head)), _) == Axis.contains(U32.to_nat(axis), Shape.naturals(head <> tail)) : Bool}
      {==}

def unique(+axes: List<&2, U32>) -> {Layout.unique(axes) == Axis.unique(Shape.naturals(axes)) : Bool}:
  match axes:
    case Nil{}:
      {==}
    case +head <> +tail:
      %Equal.sym(Bool, Layout.contains(head, tail), Axis.contains(U32.to_nat(head), Shape.naturals(tail)), contains(tail, head)) :
        {Bool.and(Bool.not(_), Layout.unique(tail)) == Axis.unique(Shape.naturals(head <> tail)) : Bool}
      %Equal.sym(Bool, Layout.unique(tail), Axis.unique(Shape.naturals(tail)), unique(tail)) :
        {Bool.and(Bool.not(Axis.contains(U32.to_nat(head), Shape.naturals(tail))), _) == Axis.unique(Shape.naturals(head <> tail)) : Bool}
      {==}

def bounded(+axes: List<&2, U32>, +rank: U32) -> {Layout.bounded(axes, rank) == Axis.bounded(Shape.naturals(axes), U32.to_nat(rank)) : Bool}:
  match axes:
    case Nil{}:
      {==}
    case +head <> +tail:
      %Arith.u32_lt(head, rank) : {Bool.and(_, Layout.bounded(tail, rank)) == Axis.bounded(Shape.naturals(head <> tail), U32.to_nat(rank)) : Bool}
      %Equal.sym(Bool, Layout.bounded(tail, rank), Axis.bounded(Shape.naturals(tail), U32.to_nat(rank)), bounded(tail, rank)) :
        {Bool.and(Nat.is_lt(U32.to_nat(head), U32.to_nat(rank)), _) == Axis.bounded(Shape.naturals(head <> tail), U32.to_nat(rank)) : Bool}
      {==}

def rank_exact(+shape: Shape.Shape) -> {U32.to_nat(Shape.rank(shape)) == Shape.rank_dimensions(Shape.dimensions(shape)) : Nat}:
  match shape:
    case Shape.SShape{+dims, size, exact, +rank_bound, capacity_bound}:
      Conversion.natural_word_exact(Shape.rank_dimensions(dims), 4,
        Equal.sym(Bool, True{}, Nat.is_le(Shape.rank_dimensions(dims), 4n), Nats.le_true(Shape.rank_dimensions(dims), 4n, rank_bound)))

def valid(+shape: Shape.Shape, +axes: List<&2, U32>, accepted: {True{} == Layout.bijective(shape, axes) : Bool}) ->
  {True{} == Axis.valid(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes)) : Bool}:
  %Shape.rank_naturals(axes) : {True{} == Bool.and(Nat.is_eq(_, Shape.rank_dimensions(Shape.dimensions(shape))), Bool.and(Axis.bounded(Shape.naturals(axes), Shape.rank_dimensions(Shape.dimensions(shape))), Axis.unique(Shape.naturals(axes)))) : Bool}
  %unique(axes) : {True{} == Bool.and(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(shape))), Bool.and(Axis.bounded(Shape.naturals(axes), Shape.rank_dimensions(Shape.dimensions(shape))), _)) : Bool}
  %rank_exact(shape) : {True{} == Bool.and(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(shape))), Bool.and(Axis.bounded(Shape.naturals(axes), _), Layout.unique(axes))) : Bool}
  %bounded(axes, Shape.rank(shape)) : {True{} == Bool.and(Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(shape))), Bool.and(_, Layout.unique(axes))) : Bool}
  accepted

def add_comm(+a: Nat, +b: Nat) -> {Nat.add(a, b) == Nat.add(b, a) : Nat}:
  %Broadcast.add_zero(b) : {Nat.add(a, _) == Nat.add(b, a) : Nat}
  %Broadcast.add_zero(a) : {Nat.add(a, Nat.add(b, 0n)) == Nat.add(b, _) : Nat}
  Nats.add_swap(a, b, 0n)

def cursor_bound(+rank: Nat, +cursor: Nat, bound: Shape.LE(Nat.add(rank, cursor), 4n)) -> Shape.LE(cursor, 4n):
  Nats.le_add_left(cursor, rank, 4n,
    %add_comm(rank, cursor) : Shape.LE(_, 4n)
    bound)

def inverse_coordinates_go(+dimensions: List<&2, U32>, +axes: List<&2, U32>, +coordinates: List<&2, U32>, +cursor: Nat,
  +bound: Shape.LE(Nat.add(Shape.rank_dimensions(dimensions), cursor), 4n)) ->
  {Layout.inverse_coordinates(dimensions, axes, coordinates, U32.from_nat(cursor)) ==
    Axis.scatter(U32, Shape.rank_dimensions(dimensions), Shape.naturals(axes), coordinates, cursor) : Maybe<&2, List<&2, U32>>}:
  match dimensions:
    case Nil{}:
      {==}
    case head <> +tail:
      %Equal.sym(Maybe<&2, U32>, Layout.coordinate_for(axes, coordinates, U32.from_nat(cursor)),
        Axis.select(U32, coordinates, Axis.position(Shape.naturals(axes), U32.to_nat(U32.from_nat(cursor)))), coordinate_for(axes, coordinates, U32.from_nat(cursor))) :
        {Layout.reorder_head(value => Layout.reorder_tail(value, Layout.inverse_coordinates(tail, axes, coordinates, U32.inc(U32.from_nat(cursor)))), _) == Axis.scatter(U32, 1n+Shape.rank_dimensions(tail), Shape.naturals(axes), coordinates, cursor) : Maybe<&2, List<&2, U32>>}
      %Equal.sym(Nat, U32.to_nat(U32.from_nat(cursor)), cursor, Conversion.natural_word_exact(cursor, 4,
        Equal.sym(Bool, True{}, Nat.is_le(cursor, 4n), Nats.le_true(cursor, 4n, cursor_bound(1n+Shape.rank_dimensions(tail), cursor, bound))))) :
        {Layout.reorder_head(value => Layout.reorder_tail(value, Layout.inverse_coordinates(tail, axes, coordinates, U32.inc(U32.from_nat(cursor)))), Axis.select(U32, coordinates, Axis.position(Shape.naturals(axes), _))) == Axis.scatter(U32, 1n+Shape.rank_dimensions(tail), Shape.naturals(axes), coordinates, cursor) : Maybe<&2, List<&2, U32>>}
      %Equal.sym(Maybe<&2, List<&2, U32>>, Layout.inverse_coordinates(tail, axes, coordinates, U32.from_nat(1n+cursor)),
        Axis.scatter(U32, Shape.rank_dimensions(tail), Shape.naturals(axes), coordinates, 1n+cursor),
        inverse_coordinates_go(tail, axes, coordinates, 1n+cursor,
          %Equal.sym(Nat, Nat.add(Shape.rank_dimensions(tail), 1n+cursor), 1n+Nat.add(Shape.rank_dimensions(tail), cursor), Arith.add_succ(Shape.rank_dimensions(tail), cursor)) : Shape.LE(_, 4n)
          bound)) :
        {Layout.reorder_head(value => Layout.reorder_tail(value, _), Axis.select(U32, coordinates, Axis.position(Shape.naturals(axes), cursor))) == Axis.scatter(U32, 1n+Shape.rank_dimensions(tail), Shape.naturals(axes), coordinates, cursor) : Maybe<&2, List<&2, U32>>}
      reorder_step(Axis.select(U32, coordinates, Axis.position(Shape.naturals(axes), cursor)), Axis.scatter(U32, Shape.rank_dimensions(tail), Shape.naturals(axes), coordinates, 1n+cursor))

def inverse_coordinates(+shape: Shape.Shape, +axes: List<&2, U32>, +coordinates: List<&2, U32>) ->
  {Layout.inverse_coordinates(Shape.dimensions(shape), axes, coordinates, 0) ==
    Axis.scatter(U32, Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes), coordinates, 0n) : Maybe<&2, List<&2, U32>>}:
  match shape:
    case Shape.SShape{+dims, size, exact, +rank_bound, capacity_bound}:
      inverse_coordinates_go(dims, axes, coordinates, 0n,
        %Equal.sym(Nat, Nat.add(Shape.rank_dimensions(dims), 0n), Shape.rank_dimensions(dims), Broadcast.add_zero(Shape.rank_dimensions(dims))) : Shape.LE(_, 4n)
        rank_bound)

def nat_clash(+a: Nat, +b: Nat, equal: {a == b : Nat}, different: {False{} == Nat.is_eq(a, b) : Bool}) -> Empty:
  Shape.clash(Equal.trans(Bool, True{}, Nat.is_eq(a, b), False{},
    Equal.trans(Bool, True{}, Nat.is_eq(b, b), Nat.is_eq(a, b), Equal.sym(Bool, Nat.is_eq(b, b), True{}, Arith.eq_refl(b)),
      Equal.cong(Nat, Bool, n => Nat.is_eq(n, b), b, a, Equal.sym(Nat, a, b, equal))),
    Equal.sym(Bool, False{}, Nat.is_eq(a, b), different)))

type ListFacts<-A: Data, -rank: Nat, -axes: List<&2, Nat>, -items: List<&2, A>> is Data:
  Lists{selected: {Axis.gather(A, items, axes) == Some{Axis.order(A, items, axes)} : Maybe<&2, List<&2, A>>},
    ranked: {List.length(&2, A, Axis.order(A, items, axes)) == rank : Nat},
    scattered: {Axis.scatter(A, rank, axes, items, 0n) == Some{Axis.order(A, items, Axis.inverse_axes(rank, axes))} : Maybe<&2, List<&2, A>>},
    restored: {Axis.order(A, Axis.order(A, items, axes), Axis.inverse_axes(rank, axes)) == items : List<&2, A>},
    reverse_restored: {Axis.order(A, Axis.order(A, items, Axis.inverse_axes(rank, axes)), axes) == items : List<&2, A>},
    scatter_restored: {Axis.scatter(A, rank, axes, Axis.order(A, items, axes), 0n) == Some{items} : Maybe<&2, List<&2, A>>}}

type Facts<-rank: Nat, -axes: List<&2, Nat>> is Type:
  Certified{inverse_valid: {True{} == Axis.valid(rank, Axis.inverse_axes(rank, axes)) : Bool},
    inverse_selected: {Axis.inverse(rank, axes) == Some{Axis.inverse_axes(rank, axes)} : Maybe<&2, List<&2, Nat>>},
    inverse_restored: {Axis.inverse_axes(rank, Axis.inverse_axes(rank, axes)) == axes : List<&2, Nat>},
    lists: @-A: Data -> @+items: List<&2, A> -> @length: {List.length(&2, A, items) == rank : Nat} -> ListFacts<A, rank, axes, items>,
    product: @+dims: List<&2, Nat> -> @length: {List.length(&2, Nat, dims) == rank : Nat} -> {Shape.numel(Axis.order(Nat, dims, axes)) == Shape.numel(dims) : Nat},
    fit: @+dims: List<&2, Nat> -> @+coords: List<&2, Nat> -> @dl: {List.length(&2, Nat, dims) == rank : Nat} -> @cl: {List.length(&2, Nat, coords) == rank : Nat} ->
      {Shape.fit(Axis.order(Nat, dims, axes), Axis.order(Nat, coords, axes)) == Shape.fit(dims, coords) : Bool}}

def mul_swap(+a: Nat, +b: Nat, +c: Nat) -> {Nat.mul(a, Nat.mul(b, c)) == Nat.mul(b, Nat.mul(a, c)) : Nat}:
  Equal.trans(Nat, Nat.mul(a, Nat.mul(b, c)), Nat.mul(Nat.mul(a, b), c), Nat.mul(b, Nat.mul(a, c)),
    Equal.sym(Nat, Nat.mul(Nat.mul(a, b), c), Nat.mul(a, Nat.mul(b, c)), Nats.mul_assoc(a, b, c)),
    Equal.trans(Nat, Nat.mul(Nat.mul(a, b), c), Nat.mul(Nat.mul(b, a), c), Nat.mul(b, Nat.mul(a, c)),
      Equal.cong(Nat, Nat, n => Nat.mul(n, c), Nat.mul(a, b), Nat.mul(b, a), Nats.mul_comm(a, b)), Nats.mul_assoc(b, a, c)))

def and_swap(+a: Bool, +b: Bool, +c: Bool) -> {Bool.and(a, Bool.and(b, c)) == Bool.and(b, Bool.and(a, c)) : Bool}:
  match a b:
    case False{} False{}:
      {==}
    case False{} True{}:
      {==}
    case True{} False{}:
      {==}
    case True{} True{}:
      {==}

type Selected<-A: Data, -items: List<&2, A>, -index: Nat> is Data:
  Selection{item: A, exact: {Axis.lookup(A, items, index) == Some{item} : Maybe<&2, A>}}

def lookup_complete_step(-A: Data, -head: A, -tail: List<&2, A>, -index: Nat, rec: Selected<A, tail, index>) -> Selected<A, head <> tail, 1n+index>:
  match rec:
    case Selection{item, exact}:
      Selection{item, exact}

def lookup_complete(-A: Data, +items: List<&2, A>, +index: Nat, bound: Shape.LT(index, List.length(&2, A, items))) -> Selected<A, items, index>:
  match items index:
    case Nil{} 0n:
      match bound:
    case Nil{} 1n+p:
      match bound:
    case +head <> tail 0n:
      Selection{head, {==}}
    case +head <> +tail 1n+p:
      lookup_complete_step(A, head, tail, p, lookup_complete(A, tail, p, bound))

def natural_value(result: Maybe<&2, U32>) -> Maybe<&2, Nat>:
  match result:
    case None{}:
      None{}
    case Some{word}:
      Some{U32.to_nat(word)}

def natural_values(result: Maybe<&2, List<&2, U32>>) -> Maybe<&2, List<&2, Nat>>:
  match result:
    case None{}:
      None{}
    case Some{words}:
      Some{Shape.naturals(words)}

def lookup_naturals(+items: List<&2, U32>, +index: Nat) ->
  {natural_value(Axis.lookup(U32, items, index)) == Axis.lookup(Nat, Shape.naturals(items), index) : Maybe<&2, Nat>}:
  match items index:
    case Nil{} 0n:
      {==}
    case Nil{} 1n+p:
      {==}
    case head <> tail 0n:
      {==}
    case head <> +tail 1n+p:
      lookup_naturals(tail, p)

def prepend_naturals(+head: Maybe<&2, U32>, +tail: Maybe<&2, List<&2, U32>>) ->
  {natural_values(Axis.prepend(U32, head, tail)) == Axis.prepend(Nat, natural_value(head), natural_values(tail)) : Maybe<&2, List<&2, Nat>>}:
  match head tail:
    case None{} None{}:
      {==}
    case None{} Some{items}:
      {==}
    case Some{value} None{}:
      {==}
    case Some{value} Some{items}:
      {==}

def gather_naturals(+axes: List<&2, Nat>, +items: List<&2, U32>) ->
  {natural_values(Axis.gather(U32, items, axes)) == Axis.gather(Nat, Shape.naturals(items), axes) : Maybe<&2, List<&2, Nat>>}:
  match axes:
    case Nil{}:
      {==}
    case +head <> +tail:
      %Equal.sym(Maybe<&2, List<&2, Nat>>, natural_values(Axis.prepend(U32, Axis.lookup(U32, items, head), Axis.gather(U32, items, tail))),
        Axis.prepend(Nat, natural_value(Axis.lookup(U32, items, head)), natural_values(Axis.gather(U32, items, tail))), prepend_naturals(Axis.lookup(U32, items, head), Axis.gather(U32, items, tail))) :
        {_ == Axis.prepend(Nat, Axis.lookup(Nat, Shape.naturals(items), head), Axis.gather(Nat, Shape.naturals(items), tail)) : Maybe<&2, List<&2, Nat>>}
      %Equal.sym(Maybe<&2, Nat>, natural_value(Axis.lookup(U32, items, head)), Axis.lookup(Nat, Shape.naturals(items), head), lookup_naturals(items, head)) :
        {Axis.prepend(Nat, _, natural_values(Axis.gather(U32, items, tail))) == Axis.gather(Nat, Shape.naturals(items), head <> tail) : Maybe<&2, List<&2, Nat>>}
      %Equal.sym(Maybe<&2, List<&2, Nat>>, natural_values(Axis.gather(U32, items, tail)), Axis.gather(Nat, Shape.naturals(items), tail), gather_naturals(tail, items)) :
        {Axis.prepend(Nat, Axis.lookup(Nat, Shape.naturals(items), head), _) == Axis.gather(Nat, Shape.naturals(items), head <> tail) : Maybe<&2, List<&2, Nat>>}
      {==}

def values_naturals(+result: Maybe<&2, List<&2, U32>>) ->
  {Shape.naturals(Axis.values(U32, result)) == Axis.values(Nat, natural_values(result)) : List<&2, Nat>}:
  match result:
    case None{}:
      {==}
    case Some{items}:
      {==}

def order_naturals(+axes: List<&2, Nat>, +items: List<&2, U32>) ->
  {Shape.naturals(Axis.order(U32, items, axes)) == Axis.order(Nat, Shape.naturals(items), axes) : List<&2, Nat>}:
  Equal.trans(List<&2, Nat>, Shape.naturals(Axis.order(U32, items, axes)), Axis.values(Nat, natural_values(Axis.gather(U32, items, axes))),
    Axis.order(Nat, Shape.naturals(items), axes), values_naturals(Axis.gather(U32, items, axes)),
    Equal.cong(Maybe<&2, List<&2, Nat>>, List<&2, Nat>, result => Axis.values(Nat, result), natural_values(Axis.gather(U32, items, axes)),
      Axis.gather(Nat, Shape.naturals(items), axes), gather_naturals(axes, items)))
def lists_empty(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 0n : Nat}) -> ListFacts<A, 0n, Nil{}, items>:
  match items:
    case Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +tail:
      Empty.absurd(ListFacts<A, 0n, Nil{}, x0 <> tail>, nat_clash(1n+List.length(&2, A, tail), 0n, length, {==}))

def product_empty(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 0n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, Nil{})) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      {==}
    case +x0 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> tail, Nil{})) == Shape.numel(x0 <> tail) : Nat}, nat_clash(1n+List.length(&2, Nat, tail), 0n, length, {==}))

def fit_empty(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 0n : Nat}, cl: {List.length(&2, Nat, coords) == 0n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, Nil{}), Axis.order(Nat, coords, Nil{})) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      match coords:
        case Nil{}:
          {==}
        case +c0 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, Nil{}), Axis.order(Nat, c0 <> ct, Nil{})) == Shape.fit(Nil{}, c0 <> ct) : Bool}, nat_clash(1n+List.length(&2, Nat, ct), 0n, cl, {==}))
    case +d0 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> dt, Nil{}), Axis.order(Nat, coords, Nil{})) == Shape.fit(d0 <> dt, coords) : Bool}, nat_clash(1n+List.length(&2, Nat, dt), 0n, dl, {==}))

def lists_0(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 1n : Nat}) -> ListFacts<A, 1n, [0n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 1n, [0n], Nil{}>, nat_clash(0n, 1n, length, {==}))
    case +x0 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +tail:
      Empty.absurd(ListFacts<A, 1n, [0n], x0 <> x1 <> tail>, nat_clash(2n+List.length(&2, A, tail), 1n, length, {==}))

def product_0(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 1n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [0n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [0n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 1n, length, {==}))
    case +x0 <> Nil{}:
      {==}
    case +x0 <> +x1 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> tail, [0n])) == Shape.numel(x0 <> x1 <> tail) : Nat}, nat_clash(2n+List.length(&2, Nat, tail), 1n, length, {==}))

def fit_0(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 1n : Nat}, cl: {List.length(&2, Nat, coords) == 1n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [0n]), Axis.order(Nat, coords, [0n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [0n]), Axis.order(Nat, coords, [0n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 1n, dl, {==}))
    case +d0 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [0n]), Axis.order(Nat, Nil{}, [0n])) == Shape.fit(d0 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 1n, cl, {==}))
        case +c0 <> Nil{}:
          {==}
        case +c0 <> +c1 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [0n]), Axis.order(Nat, c0 <> c1 <> ct, [0n])) == Shape.fit(d0 <> Nil{}, c0 <> c1 <> ct) : Bool}, nat_clash(2n+List.length(&2, Nat, ct), 1n, cl, {==}))
    case +d0 <> +d1 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> dt, [0n]), Axis.order(Nat, coords, [0n])) == Shape.fit(d0 <> d1 <> dt, coords) : Bool}, nat_clash(2n+List.length(&2, Nat, dt), 1n, dl, {==}))

def lists_01(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 2n : Nat}) -> ListFacts<A, 2n, [0n, 1n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 2n, [0n, 1n], Nil{}>, nat_clash(0n, 2n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 2n, [0n, 1n], x0 <> Nil{}>, nat_clash(1n, 2n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +tail:
      Empty.absurd(ListFacts<A, 2n, [0n, 1n], x0 <> x1 <> x2 <> tail>, nat_clash(3n+List.length(&2, A, tail), 2n, length, {==}))

def product_01(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 2n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [0n, 1n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [0n, 1n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 2n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [0n, 1n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 2n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      {==}
    case +x0 <> +x1 <> +x2 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> tail, [0n, 1n])) == Shape.numel(x0 <> x1 <> x2 <> tail) : Nat}, nat_clash(3n+List.length(&2, Nat, tail), 2n, length, {==}))

def fit_01(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 2n : Nat}, cl: {List.length(&2, Nat, coords) == 2n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [0n, 1n]), Axis.order(Nat, coords, [0n, 1n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [0n, 1n]), Axis.order(Nat, coords, [0n, 1n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 2n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [0n, 1n]), Axis.order(Nat, coords, [0n, 1n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 2n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [0n, 1n]), Axis.order(Nat, Nil{}, [0n, 1n])) == Shape.fit(d0 <> d1 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 2n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [0n, 1n]), Axis.order(Nat, c0 <> Nil{}, [0n, 1n])) == Shape.fit(d0 <> d1 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 2n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          {==}
        case +c0 <> +c1 <> +c2 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [0n, 1n]), Axis.order(Nat, c0 <> c1 <> c2 <> ct, [0n, 1n])) == Shape.fit(d0 <> d1 <> Nil{}, c0 <> c1 <> c2 <> ct) : Bool}, nat_clash(3n+List.length(&2, Nat, ct), 2n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> dt, [0n, 1n]), Axis.order(Nat, coords, [0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> dt, coords) : Bool}, nat_clash(3n+List.length(&2, Nat, dt), 2n, dl, {==}))

def lists_10(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 2n : Nat}) -> ListFacts<A, 2n, [1n, 0n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 2n, [1n, 0n], Nil{}>, nat_clash(0n, 2n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 2n, [1n, 0n], x0 <> Nil{}>, nat_clash(1n, 2n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +tail:
      Empty.absurd(ListFacts<A, 2n, [1n, 0n], x0 <> x1 <> x2 <> tail>, nat_clash(3n+List.length(&2, A, tail), 2n, length, {==}))

def product_10(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 2n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [1n, 0n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [1n, 0n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 2n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [1n, 0n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 2n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x1, Nat.mul(x0, 1n)), Nat.mul(x0, Nat.mul(x1, 1n)), Nat.mul(x0, Nat.mul(x1, 1n)), mul_swap(x1, x0, 1n), {==})
    case +x0 <> +x1 <> +x2 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> tail, [1n, 0n])) == Shape.numel(x0 <> x1 <> x2 <> tail) : Nat}, nat_clash(3n+List.length(&2, Nat, tail), 2n, length, {==}))

def fit_10(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 2n : Nat}, cl: {List.length(&2, Nat, coords) == 2n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [1n, 0n]), Axis.order(Nat, coords, [1n, 0n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [1n, 0n]), Axis.order(Nat, coords, [1n, 0n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 2n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [1n, 0n]), Axis.order(Nat, coords, [1n, 0n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 2n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [1n, 0n]), Axis.order(Nat, Nil{}, [1n, 0n])) == Shape.fit(d0 <> d1 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 2n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [1n, 0n]), Axis.order(Nat, c0 <> Nil{}, [1n, 0n])) == Shape.fit(d0 <> d1 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 2n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), True{})), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{})), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{})), and_swap(Nat.is_lt(c1, d1), Nat.is_lt(c0, d0), True{}), {==})
        case +c0 <> +c1 <> +c2 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [1n, 0n]), Axis.order(Nat, c0 <> c1 <> c2 <> ct, [1n, 0n])) == Shape.fit(d0 <> d1 <> Nil{}, c0 <> c1 <> c2 <> ct) : Bool}, nat_clash(3n+List.length(&2, Nat, ct), 2n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> dt, [1n, 0n]), Axis.order(Nat, coords, [1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> dt, coords) : Bool}, nat_clash(3n+List.length(&2, Nat, dt), 2n, dl, {==}))

def lists_012(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 3n : Nat}) -> ListFacts<A, 3n, [0n, 1n, 2n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 3n, [0n, 1n, 2n], Nil{}>, nat_clash(0n, 3n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 3n, [0n, 1n, 2n], x0 <> Nil{}>, nat_clash(1n, 3n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 3n, [0n, 1n, 2n], x0 <> x1 <> Nil{}>, nat_clash(2n, 3n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +tail:
      Empty.absurd(ListFacts<A, 3n, [0n, 1n, 2n], x0 <> x1 <> x2 <> x3 <> tail>, nat_clash(4n+List.length(&2, A, tail), 3n, length, {==}))

def product_012(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 3n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [0n, 1n, 2n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [0n, 1n, 2n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 3n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [0n, 1n, 2n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 3n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [0n, 1n, 2n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 3n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      {==}
    case +x0 <> +x1 <> +x2 <> +x3 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> tail, [0n, 1n, 2n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> tail) : Nat}, nat_clash(4n+List.length(&2, Nat, tail), 3n, length, {==}))

def fit_012(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 3n : Nat}, cl: {List.length(&2, Nat, coords) == 3n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [0n, 1n, 2n]), Axis.order(Nat, coords, [0n, 1n, 2n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [0n, 1n, 2n]), Axis.order(Nat, coords, [0n, 1n, 2n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 3n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [0n, 1n, 2n]), Axis.order(Nat, coords, [0n, 1n, 2n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 3n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [0n, 1n, 2n]), Axis.order(Nat, coords, [0n, 1n, 2n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 3n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [0n, 1n, 2n]), Axis.order(Nat, Nil{}, [0n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 3n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [0n, 1n, 2n]), Axis.order(Nat, c0 <> Nil{}, [0n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 3n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [0n, 1n, 2n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [0n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 3n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          {==}
        case +c0 <> +c1 <> +c2 <> +c3 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [0n, 1n, 2n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> ct, [0n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> c1 <> c2 <> c3 <> ct) : Bool}, nat_clash(4n+List.length(&2, Nat, ct), 3n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> dt, [0n, 1n, 2n]), Axis.order(Nat, coords, [0n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> dt, coords) : Bool}, nat_clash(4n+List.length(&2, Nat, dt), 3n, dl, {==}))

def lists_021(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 3n : Nat}) -> ListFacts<A, 3n, [0n, 2n, 1n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 3n, [0n, 2n, 1n], Nil{}>, nat_clash(0n, 3n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 3n, [0n, 2n, 1n], x0 <> Nil{}>, nat_clash(1n, 3n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 3n, [0n, 2n, 1n], x0 <> x1 <> Nil{}>, nat_clash(2n, 3n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +tail:
      Empty.absurd(ListFacts<A, 3n, [0n, 2n, 1n], x0 <> x1 <> x2 <> x3 <> tail>, nat_clash(4n+List.length(&2, A, tail), 3n, length, {==}))

def product_021(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 3n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [0n, 2n, 1n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [0n, 2n, 1n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 3n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [0n, 2n, 1n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 3n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [0n, 2n, 1n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 3n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x2, Nat.mul(x1, 1n)), Nat.mul(x1, Nat.mul(x2, 1n)), mul_swap(x2, x1, 1n)), {==})
    case +x0 <> +x1 <> +x2 <> +x3 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> tail, [0n, 2n, 1n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> tail) : Nat}, nat_clash(4n+List.length(&2, Nat, tail), 3n, length, {==}))

def fit_021(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 3n : Nat}, cl: {List.length(&2, Nat, coords) == 3n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [0n, 2n, 1n]), Axis.order(Nat, coords, [0n, 2n, 1n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [0n, 2n, 1n]), Axis.order(Nat, coords, [0n, 2n, 1n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 3n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [0n, 2n, 1n]), Axis.order(Nat, coords, [0n, 2n, 1n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 3n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [0n, 2n, 1n]), Axis.order(Nat, coords, [0n, 2n, 1n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 3n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [0n, 2n, 1n]), Axis.order(Nat, Nil{}, [0n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 3n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [0n, 2n, 1n]), Axis.order(Nat, c0 <> Nil{}, [0n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 3n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [0n, 2n, 1n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [0n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 3n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c1, d1), True{})), {==})
        case +c0 <> +c1 <> +c2 <> +c3 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [0n, 2n, 1n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> ct, [0n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> c1 <> c2 <> c3 <> ct) : Bool}, nat_clash(4n+List.length(&2, Nat, ct), 3n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> dt, [0n, 2n, 1n]), Axis.order(Nat, coords, [0n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> dt, coords) : Bool}, nat_clash(4n+List.length(&2, Nat, dt), 3n, dl, {==}))

def lists_102(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 3n : Nat}) -> ListFacts<A, 3n, [1n, 0n, 2n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 3n, [1n, 0n, 2n], Nil{}>, nat_clash(0n, 3n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 3n, [1n, 0n, 2n], x0 <> Nil{}>, nat_clash(1n, 3n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 3n, [1n, 0n, 2n], x0 <> x1 <> Nil{}>, nat_clash(2n, 3n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +tail:
      Empty.absurd(ListFacts<A, 3n, [1n, 0n, 2n], x0 <> x1 <> x2 <> x3 <> tail>, nat_clash(4n+List.length(&2, A, tail), 3n, length, {==}))

def product_102(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 3n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [1n, 0n, 2n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [1n, 0n, 2n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 3n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [1n, 0n, 2n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 3n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [1n, 0n, 2n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 3n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x1, Nat.mul(x0, Nat.mul(x2, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n))), mul_swap(x1, x0, Nat.mul(x2, 1n)), {==})
    case +x0 <> +x1 <> +x2 <> +x3 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> tail, [1n, 0n, 2n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> tail) : Nat}, nat_clash(4n+List.length(&2, Nat, tail), 3n, length, {==}))

def fit_102(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 3n : Nat}, cl: {List.length(&2, Nat, coords) == 3n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [1n, 0n, 2n]), Axis.order(Nat, coords, [1n, 0n, 2n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [1n, 0n, 2n]), Axis.order(Nat, coords, [1n, 0n, 2n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 3n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [1n, 0n, 2n]), Axis.order(Nat, coords, [1n, 0n, 2n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 3n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [1n, 0n, 2n]), Axis.order(Nat, coords, [1n, 0n, 2n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 3n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [1n, 0n, 2n]), Axis.order(Nat, Nil{}, [1n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 3n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [1n, 0n, 2n]), Axis.order(Nat, c0 <> Nil{}, [1n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 3n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [1n, 0n, 2n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [1n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 3n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), and_swap(Nat.is_lt(c1, d1), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{})), {==})
        case +c0 <> +c1 <> +c2 <> +c3 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [1n, 0n, 2n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> ct, [1n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> c1 <> c2 <> c3 <> ct) : Bool}, nat_clash(4n+List.length(&2, Nat, ct), 3n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> dt, [1n, 0n, 2n]), Axis.order(Nat, coords, [1n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> dt, coords) : Bool}, nat_clash(4n+List.length(&2, Nat, dt), 3n, dl, {==}))

def lists_120(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 3n : Nat}) -> ListFacts<A, 3n, [1n, 2n, 0n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 3n, [1n, 2n, 0n], Nil{}>, nat_clash(0n, 3n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 3n, [1n, 2n, 0n], x0 <> Nil{}>, nat_clash(1n, 3n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 3n, [1n, 2n, 0n], x0 <> x1 <> Nil{}>, nat_clash(2n, 3n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +tail:
      Empty.absurd(ListFacts<A, 3n, [1n, 2n, 0n], x0 <> x1 <> x2 <> x3 <> tail>, nat_clash(4n+List.length(&2, A, tail), 3n, length, {==}))

def product_120(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 3n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [1n, 2n, 0n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [1n, 2n, 0n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 3n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [1n, 2n, 0n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 3n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [1n, 2n, 0n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 3n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x1, Nat.mul(x2, Nat.mul(x0, 1n))), Nat.mul(x1, Nat.mul(x0, Nat.mul(x2, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n))), Equal.cong(Nat, Nat, value => Nat.mul(x1, value), Nat.mul(x2, Nat.mul(x0, 1n)), Nat.mul(x0, Nat.mul(x2, 1n)), mul_swap(x2, x0, 1n)), Equal.trans(Nat, Nat.mul(x1, Nat.mul(x0, Nat.mul(x2, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n))), mul_swap(x1, x0, Nat.mul(x2, 1n)), {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> tail, [1n, 2n, 0n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> tail) : Nat}, nat_clash(4n+List.length(&2, Nat, tail), 3n, length, {==}))

def fit_120(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 3n : Nat}, cl: {List.length(&2, Nat, coords) == 3n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [1n, 2n, 0n]), Axis.order(Nat, coords, [1n, 2n, 0n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [1n, 2n, 0n]), Axis.order(Nat, coords, [1n, 2n, 0n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 3n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [1n, 2n, 0n]), Axis.order(Nat, coords, [1n, 2n, 0n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 3n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [1n, 2n, 0n]), Axis.order(Nat, coords, [1n, 2n, 0n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 3n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [1n, 2n, 0n]), Axis.order(Nat, Nil{}, [1n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 3n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [1n, 2n, 0n]), Axis.order(Nat, c0 <> Nil{}, [1n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 3n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [1n, 2n, 0n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [1n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 3n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), True{}))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c1, d1), value), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), True{})), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{})), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c0, d0), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), and_swap(Nat.is_lt(c1, d1), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{})), {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [1n, 2n, 0n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> ct, [1n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> c1 <> c2 <> c3 <> ct) : Bool}, nat_clash(4n+List.length(&2, Nat, ct), 3n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> dt, [1n, 2n, 0n]), Axis.order(Nat, coords, [1n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> dt, coords) : Bool}, nat_clash(4n+List.length(&2, Nat, dt), 3n, dl, {==}))

def lists_201(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 3n : Nat}) -> ListFacts<A, 3n, [2n, 0n, 1n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 3n, [2n, 0n, 1n], Nil{}>, nat_clash(0n, 3n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 3n, [2n, 0n, 1n], x0 <> Nil{}>, nat_clash(1n, 3n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 3n, [2n, 0n, 1n], x0 <> x1 <> Nil{}>, nat_clash(2n, 3n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +tail:
      Empty.absurd(ListFacts<A, 3n, [2n, 0n, 1n], x0 <> x1 <> x2 <> x3 <> tail>, nat_clash(4n+List.length(&2, A, tail), 3n, length, {==}))

def product_201(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 3n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [2n, 0n, 1n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [2n, 0n, 1n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 3n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [2n, 0n, 1n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 3n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [2n, 0n, 1n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 3n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x2, Nat.mul(x0, Nat.mul(x1, 1n))), Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n))), mul_swap(x2, x0, Nat.mul(x1, 1n)), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x2, Nat.mul(x1, 1n)), Nat.mul(x1, Nat.mul(x2, 1n)), mul_swap(x2, x1, 1n)), {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> tail, [2n, 0n, 1n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> tail) : Nat}, nat_clash(4n+List.length(&2, Nat, tail), 3n, length, {==}))

def fit_201(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 3n : Nat}, cl: {List.length(&2, Nat, coords) == 3n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [2n, 0n, 1n]), Axis.order(Nat, coords, [2n, 0n, 1n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [2n, 0n, 1n]), Axis.order(Nat, coords, [2n, 0n, 1n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 3n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [2n, 0n, 1n]), Axis.order(Nat, coords, [2n, 0n, 1n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 3n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [2n, 0n, 1n]), Axis.order(Nat, coords, [2n, 0n, 1n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 3n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [2n, 0n, 1n]), Axis.order(Nat, Nil{}, [2n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 3n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [2n, 0n, 1n]), Axis.order(Nat, c0 <> Nil{}, [2n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 3n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [2n, 0n, 1n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [2n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 3n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c1, d1), True{})), {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [2n, 0n, 1n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> ct, [2n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> c1 <> c2 <> c3 <> ct) : Bool}, nat_clash(4n+List.length(&2, Nat, ct), 3n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> dt, [2n, 0n, 1n]), Axis.order(Nat, coords, [2n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> dt, coords) : Bool}, nat_clash(4n+List.length(&2, Nat, dt), 3n, dl, {==}))

def lists_210(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 3n : Nat}) -> ListFacts<A, 3n, [2n, 1n, 0n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 3n, [2n, 1n, 0n], Nil{}>, nat_clash(0n, 3n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 3n, [2n, 1n, 0n], x0 <> Nil{}>, nat_clash(1n, 3n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 3n, [2n, 1n, 0n], x0 <> x1 <> Nil{}>, nat_clash(2n, 3n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +tail:
      Empty.absurd(ListFacts<A, 3n, [2n, 1n, 0n], x0 <> x1 <> x2 <> x3 <> tail>, nat_clash(4n+List.length(&2, A, tail), 3n, length, {==}))

def product_210(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 3n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [2n, 1n, 0n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [2n, 1n, 0n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 3n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [2n, 1n, 0n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 3n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [2n, 1n, 0n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 3n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x2, Nat.mul(x1, Nat.mul(x0, 1n))), Nat.mul(x2, Nat.mul(x0, Nat.mul(x1, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n))), Equal.cong(Nat, Nat, value => Nat.mul(x2, value), Nat.mul(x1, Nat.mul(x0, 1n)), Nat.mul(x0, Nat.mul(x1, 1n)), mul_swap(x1, x0, 1n)), Equal.trans(Nat, Nat.mul(x2, Nat.mul(x0, Nat.mul(x1, 1n))), Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n))), mul_swap(x2, x0, Nat.mul(x1, 1n)), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x2, Nat.mul(x1, 1n)), Nat.mul(x1, Nat.mul(x2, 1n)), mul_swap(x2, x1, 1n)), {==})))
    case +x0 <> +x1 <> +x2 <> +x3 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> tail, [2n, 1n, 0n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> tail) : Nat}, nat_clash(4n+List.length(&2, Nat, tail), 3n, length, {==}))

def fit_210(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 3n : Nat}, cl: {List.length(&2, Nat, coords) == 3n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [2n, 1n, 0n]), Axis.order(Nat, coords, [2n, 1n, 0n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [2n, 1n, 0n]), Axis.order(Nat, coords, [2n, 1n, 0n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 3n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [2n, 1n, 0n]), Axis.order(Nat, coords, [2n, 1n, 0n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 3n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [2n, 1n, 0n]), Axis.order(Nat, coords, [2n, 1n, 0n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 3n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [2n, 1n, 0n]), Axis.order(Nat, Nil{}, [2n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 3n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [2n, 1n, 0n]), Axis.order(Nat, c0 <> Nil{}, [2n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 3n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [2n, 1n, 0n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [2n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 3n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), True{}))), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c2, d2), value), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), True{})), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{})), and_swap(Nat.is_lt(c1, d1), Nat.is_lt(c0, d0), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c1, d1), True{})), {==})))
        case +c0 <> +c1 <> +c2 <> +c3 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [2n, 1n, 0n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> ct, [2n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, c0 <> c1 <> c2 <> c3 <> ct) : Bool}, nat_clash(4n+List.length(&2, Nat, ct), 3n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> dt, [2n, 1n, 0n]), Axis.order(Nat, coords, [2n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> dt, coords) : Bool}, nat_clash(4n+List.length(&2, Nat, dt), 3n, dl, {==}))

def lists_0123(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [0n, 1n, 2n, 3n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 1n, 2n, 3n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 1n, 2n, 3n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 1n, 2n, 3n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 1n, 2n, 3n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [0n, 1n, 2n, 3n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_0123(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [0n, 1n, 2n, 3n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [0n, 1n, 2n, 3n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [0n, 1n, 2n, 3n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [0n, 1n, 2n, 3n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [0n, 1n, 2n, 3n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      {==}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [0n, 1n, 2n, 3n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_0123(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [0n, 1n, 2n, 3n]), Axis.order(Nat, coords, [0n, 1n, 2n, 3n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [0n, 1n, 2n, 3n]), Axis.order(Nat, coords, [0n, 1n, 2n, 3n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [0n, 1n, 2n, 3n]), Axis.order(Nat, coords, [0n, 1n, 2n, 3n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [0n, 1n, 2n, 3n]), Axis.order(Nat, coords, [0n, 1n, 2n, 3n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [0n, 1n, 2n, 3n]), Axis.order(Nat, coords, [0n, 1n, 2n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 1n, 2n, 3n]), Axis.order(Nat, Nil{}, [0n, 1n, 2n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 1n, 2n, 3n]), Axis.order(Nat, c0 <> Nil{}, [0n, 1n, 2n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 1n, 2n, 3n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [0n, 1n, 2n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 1n, 2n, 3n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [0n, 1n, 2n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          {==}
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 1n, 2n, 3n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [0n, 1n, 2n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [0n, 1n, 2n, 3n]), Axis.order(Nat, coords, [0n, 1n, 2n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_0132(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [0n, 1n, 3n, 2n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 1n, 3n, 2n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 1n, 3n, 2n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 1n, 3n, 2n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 1n, 3n, 2n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [0n, 1n, 3n, 2n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_0132(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [0n, 1n, 3n, 2n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [0n, 1n, 3n, 2n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [0n, 1n, 3n, 2n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [0n, 1n, 3n, 2n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [0n, 1n, 3n, 2n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x1, value)), Nat.mul(x3, Nat.mul(x2, 1n)), Nat.mul(x2, Nat.mul(x3, 1n)), mul_swap(x3, x2, 1n)), {==})
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [0n, 1n, 3n, 2n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_0132(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [0n, 1n, 3n, 2n]), Axis.order(Nat, coords, [0n, 1n, 3n, 2n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [0n, 1n, 3n, 2n]), Axis.order(Nat, coords, [0n, 1n, 3n, 2n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [0n, 1n, 3n, 2n]), Axis.order(Nat, coords, [0n, 1n, 3n, 2n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [0n, 1n, 3n, 2n]), Axis.order(Nat, coords, [0n, 1n, 3n, 2n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [0n, 1n, 3n, 2n]), Axis.order(Nat, coords, [0n, 1n, 3n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 1n, 3n, 2n]), Axis.order(Nat, Nil{}, [0n, 1n, 3n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 1n, 3n, 2n]), Axis.order(Nat, c0 <> Nil{}, [0n, 1n, 3n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 1n, 3n, 2n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [0n, 1n, 3n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 1n, 3n, 2n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [0n, 1n, 3n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c2, d2), True{})), {==})
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 1n, 3n, 2n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [0n, 1n, 3n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [0n, 1n, 3n, 2n]), Axis.order(Nat, coords, [0n, 1n, 3n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_0213(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [0n, 2n, 1n, 3n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 2n, 1n, 3n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 2n, 1n, 3n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 2n, 1n, 3n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 2n, 1n, 3n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [0n, 2n, 1n, 3n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_0213(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [0n, 2n, 1n, 3n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [0n, 2n, 1n, 3n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [0n, 2n, 1n, 3n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [0n, 2n, 1n, 3n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [0n, 2n, 1n, 3n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n))), Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n))), mul_swap(x2, x1, Nat.mul(x3, 1n))), {==})
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [0n, 2n, 1n, 3n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_0213(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [0n, 2n, 1n, 3n]), Axis.order(Nat, coords, [0n, 2n, 1n, 3n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [0n, 2n, 1n, 3n]), Axis.order(Nat, coords, [0n, 2n, 1n, 3n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [0n, 2n, 1n, 3n]), Axis.order(Nat, coords, [0n, 2n, 1n, 3n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [0n, 2n, 1n, 3n]), Axis.order(Nat, coords, [0n, 2n, 1n, 3n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [0n, 2n, 1n, 3n]), Axis.order(Nat, coords, [0n, 2n, 1n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 2n, 1n, 3n]), Axis.order(Nat, Nil{}, [0n, 2n, 1n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 2n, 1n, 3n]), Axis.order(Nat, c0 <> Nil{}, [0n, 2n, 1n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 2n, 1n, 3n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [0n, 2n, 1n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 2n, 1n, 3n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [0n, 2n, 1n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{}))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), {==})
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 2n, 1n, 3n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [0n, 2n, 1n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [0n, 2n, 1n, 3n]), Axis.order(Nat, coords, [0n, 2n, 1n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_0231(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [0n, 2n, 3n, 1n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 2n, 3n, 1n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 2n, 3n, 1n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 2n, 3n, 1n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 2n, 3n, 1n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [0n, 2n, 3n, 1n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_0231(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [0n, 2n, 3n, 1n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [0n, 2n, 3n, 1n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [0n, 2n, 3n, 1n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [0n, 2n, 3n, 1n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [0n, 2n, 3n, 1n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x0, Nat.mul(x2, Nat.mul(x3, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x2, value)), Nat.mul(x3, Nat.mul(x1, 1n)), Nat.mul(x1, Nat.mul(x3, 1n)), mul_swap(x3, x1, 1n)), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n))), Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n))), mul_swap(x2, x1, Nat.mul(x3, 1n))), {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [0n, 2n, 3n, 1n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_0231(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [0n, 2n, 3n, 1n]), Axis.order(Nat, coords, [0n, 2n, 3n, 1n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [0n, 2n, 3n, 1n]), Axis.order(Nat, coords, [0n, 2n, 3n, 1n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [0n, 2n, 3n, 1n]), Axis.order(Nat, coords, [0n, 2n, 3n, 1n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [0n, 2n, 3n, 1n]), Axis.order(Nat, coords, [0n, 2n, 3n, 1n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [0n, 2n, 3n, 1n]), Axis.order(Nat, coords, [0n, 2n, 3n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 2n, 3n, 1n]), Axis.order(Nat, Nil{}, [0n, 2n, 3n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 2n, 3n, 1n]), Axis.order(Nat, c0 <> Nil{}, [0n, 2n, 3n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 2n, 3n, 1n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [0n, 2n, 3n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 2n, 3n, 1n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [0n, 2n, 3n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{})), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c1, d1), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{}))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 2n, 3n, 1n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [0n, 2n, 3n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [0n, 2n, 3n, 1n]), Axis.order(Nat, coords, [0n, 2n, 3n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_0312(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [0n, 3n, 1n, 2n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 3n, 1n, 2n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 3n, 1n, 2n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 3n, 1n, 2n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 3n, 1n, 2n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [0n, 3n, 1n, 2n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_0312(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [0n, 3n, 1n, 2n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [0n, 3n, 1n, 2n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [0n, 3n, 1n, 2n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [0n, 3n, 1n, 2n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [0n, 3n, 1n, 2n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n))), Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n))), mul_swap(x3, x1, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x1, value)), Nat.mul(x3, Nat.mul(x2, 1n)), Nat.mul(x2, Nat.mul(x3, 1n)), mul_swap(x3, x2, 1n)), {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [0n, 3n, 1n, 2n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_0312(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [0n, 3n, 1n, 2n]), Axis.order(Nat, coords, [0n, 3n, 1n, 2n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [0n, 3n, 1n, 2n]), Axis.order(Nat, coords, [0n, 3n, 1n, 2n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [0n, 3n, 1n, 2n]), Axis.order(Nat, coords, [0n, 3n, 1n, 2n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [0n, 3n, 1n, 2n]), Axis.order(Nat, coords, [0n, 3n, 1n, 2n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [0n, 3n, 1n, 2n]), Axis.order(Nat, coords, [0n, 3n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 3n, 1n, 2n]), Axis.order(Nat, Nil{}, [0n, 3n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 3n, 1n, 2n]), Axis.order(Nat, c0 <> Nil{}, [0n, 3n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 3n, 1n, 2n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [0n, 3n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 3n, 1n, 2n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [0n, 3n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{}))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c2, d2), True{})), {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 3n, 1n, 2n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [0n, 3n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [0n, 3n, 1n, 2n]), Axis.order(Nat, coords, [0n, 3n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_0321(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [0n, 3n, 2n, 1n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 3n, 2n, 1n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 3n, 2n, 1n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 3n, 2n, 1n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [0n, 3n, 2n, 1n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [0n, 3n, 2n, 1n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_0321(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [0n, 3n, 2n, 1n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [0n, 3n, 2n, 1n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [0n, 3n, 2n, 1n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [0n, 3n, 2n, 1n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [0n, 3n, 2n, 1n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x0, Nat.mul(x3, Nat.mul(x2, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x3, value)), Nat.mul(x2, Nat.mul(x1, 1n)), Nat.mul(x1, Nat.mul(x2, 1n)), mul_swap(x2, x1, 1n)), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n))), Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n))), mul_swap(x3, x1, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x1, value)), Nat.mul(x3, Nat.mul(x2, 1n)), Nat.mul(x2, Nat.mul(x3, 1n)), mul_swap(x3, x2, 1n)), {==})))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [0n, 3n, 2n, 1n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_0321(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [0n, 3n, 2n, 1n]), Axis.order(Nat, coords, [0n, 3n, 2n, 1n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [0n, 3n, 2n, 1n]), Axis.order(Nat, coords, [0n, 3n, 2n, 1n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [0n, 3n, 2n, 1n]), Axis.order(Nat, coords, [0n, 3n, 2n, 1n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [0n, 3n, 2n, 1n]), Axis.order(Nat, coords, [0n, 3n, 2n, 1n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [0n, 3n, 2n, 1n]), Axis.order(Nat, coords, [0n, 3n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 3n, 2n, 1n]), Axis.order(Nat, Nil{}, [0n, 3n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 3n, 2n, 1n]), Axis.order(Nat, c0 <> Nil{}, [0n, 3n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 3n, 2n, 1n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [0n, 3n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 3n, 2n, 1n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [0n, 3n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), value)), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c1, d1), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{}))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c2, d2), True{})), {==})))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [0n, 3n, 2n, 1n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [0n, 3n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [0n, 3n, 2n, 1n]), Axis.order(Nat, coords, [0n, 3n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_1023(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [1n, 0n, 2n, 3n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 0n, 2n, 3n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 0n, 2n, 3n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 0n, 2n, 3n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 0n, 2n, 3n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [1n, 0n, 2n, 3n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_1023(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [1n, 0n, 2n, 3n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [1n, 0n, 2n, 3n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [1n, 0n, 2n, 3n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [1n, 0n, 2n, 3n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [1n, 0n, 2n, 3n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x1, Nat.mul(x0, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x1, x0, Nat.mul(x2, Nat.mul(x3, 1n))), {==})
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [1n, 0n, 2n, 3n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_1023(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [1n, 0n, 2n, 3n]), Axis.order(Nat, coords, [1n, 0n, 2n, 3n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [1n, 0n, 2n, 3n]), Axis.order(Nat, coords, [1n, 0n, 2n, 3n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [1n, 0n, 2n, 3n]), Axis.order(Nat, coords, [1n, 0n, 2n, 3n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [1n, 0n, 2n, 3n]), Axis.order(Nat, coords, [1n, 0n, 2n, 3n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [1n, 0n, 2n, 3n]), Axis.order(Nat, coords, [1n, 0n, 2n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 0n, 2n, 3n]), Axis.order(Nat, Nil{}, [1n, 0n, 2n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 0n, 2n, 3n]), Axis.order(Nat, c0 <> Nil{}, [1n, 0n, 2n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 0n, 2n, 3n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [1n, 0n, 2n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 0n, 2n, 3n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [1n, 0n, 2n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c1, d1), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{}))), {==})
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 0n, 2n, 3n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [1n, 0n, 2n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [1n, 0n, 2n, 3n]), Axis.order(Nat, coords, [1n, 0n, 2n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_1032(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [1n, 0n, 3n, 2n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 0n, 3n, 2n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 0n, 3n, 2n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 0n, 3n, 2n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 0n, 3n, 2n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [1n, 0n, 3n, 2n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_1032(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [1n, 0n, 3n, 2n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [1n, 0n, 3n, 2n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [1n, 0n, 3n, 2n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [1n, 0n, 3n, 2n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [1n, 0n, 3n, 2n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x1, Nat.mul(x0, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x1, x0, Nat.mul(x3, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x1, value)), Nat.mul(x3, Nat.mul(x2, 1n)), Nat.mul(x2, Nat.mul(x3, 1n)), mul_swap(x3, x2, 1n)), {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [1n, 0n, 3n, 2n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_1032(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [1n, 0n, 3n, 2n]), Axis.order(Nat, coords, [1n, 0n, 3n, 2n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [1n, 0n, 3n, 2n]), Axis.order(Nat, coords, [1n, 0n, 3n, 2n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [1n, 0n, 3n, 2n]), Axis.order(Nat, coords, [1n, 0n, 3n, 2n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [1n, 0n, 3n, 2n]), Axis.order(Nat, coords, [1n, 0n, 3n, 2n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [1n, 0n, 3n, 2n]), Axis.order(Nat, coords, [1n, 0n, 3n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 0n, 3n, 2n]), Axis.order(Nat, Nil{}, [1n, 0n, 3n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 0n, 3n, 2n]), Axis.order(Nat, c0 <> Nil{}, [1n, 0n, 3n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 0n, 3n, 2n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [1n, 0n, 3n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 0n, 3n, 2n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [1n, 0n, 3n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c1, d1), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c2, d2), True{})), {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 0n, 3n, 2n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [1n, 0n, 3n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [1n, 0n, 3n, 2n]), Axis.order(Nat, coords, [1n, 0n, 3n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_1203(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [1n, 2n, 0n, 3n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 2n, 0n, 3n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 2n, 0n, 3n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 2n, 0n, 3n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 2n, 0n, 3n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [1n, 2n, 0n, 3n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_1203(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [1n, 2n, 0n, 3n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [1n, 2n, 0n, 3n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [1n, 2n, 0n, 3n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [1n, 2n, 0n, 3n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [1n, 2n, 0n, 3n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x1, Nat.mul(x2, Nat.mul(x0, Nat.mul(x3, 1n)))), Nat.mul(x1, Nat.mul(x0, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x1, value), Nat.mul(x2, Nat.mul(x0, Nat.mul(x3, 1n))), Nat.mul(x0, Nat.mul(x2, Nat.mul(x3, 1n))), mul_swap(x2, x0, Nat.mul(x3, 1n))), Equal.trans(Nat, Nat.mul(x1, Nat.mul(x0, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x1, x0, Nat.mul(x2, Nat.mul(x3, 1n))), {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [1n, 2n, 0n, 3n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_1203(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [1n, 2n, 0n, 3n]), Axis.order(Nat, coords, [1n, 2n, 0n, 3n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [1n, 2n, 0n, 3n]), Axis.order(Nat, coords, [1n, 2n, 0n, 3n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [1n, 2n, 0n, 3n]), Axis.order(Nat, coords, [1n, 2n, 0n, 3n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [1n, 2n, 0n, 3n]), Axis.order(Nat, coords, [1n, 2n, 0n, 3n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [1n, 2n, 0n, 3n]), Axis.order(Nat, coords, [1n, 2n, 0n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 2n, 0n, 3n]), Axis.order(Nat, Nil{}, [1n, 2n, 0n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 2n, 0n, 3n]), Axis.order(Nat, c0 <> Nil{}, [1n, 2n, 0n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 2n, 0n, 3n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [1n, 2n, 0n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 2n, 0n, 3n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [1n, 2n, 0n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c1, d1), value), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{}))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c1, d1), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{}))), {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 2n, 0n, 3n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [1n, 2n, 0n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [1n, 2n, 0n, 3n]), Axis.order(Nat, coords, [1n, 2n, 0n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_1230(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [1n, 2n, 3n, 0n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 2n, 3n, 0n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 2n, 3n, 0n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 2n, 3n, 0n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 2n, 3n, 0n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [1n, 2n, 3n, 0n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_1230(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [1n, 2n, 3n, 0n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [1n, 2n, 3n, 0n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [1n, 2n, 3n, 0n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [1n, 2n, 3n, 0n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [1n, 2n, 3n, 0n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, Nat.mul(x0, 1n)))), Nat.mul(x1, Nat.mul(x2, Nat.mul(x0, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x1, Nat.mul(x2, value)), Nat.mul(x3, Nat.mul(x0, 1n)), Nat.mul(x0, Nat.mul(x3, 1n)), mul_swap(x3, x0, 1n)), Equal.trans(Nat, Nat.mul(x1, Nat.mul(x2, Nat.mul(x0, Nat.mul(x3, 1n)))), Nat.mul(x1, Nat.mul(x0, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x1, value), Nat.mul(x2, Nat.mul(x0, Nat.mul(x3, 1n))), Nat.mul(x0, Nat.mul(x2, Nat.mul(x3, 1n))), mul_swap(x2, x0, Nat.mul(x3, 1n))), Equal.trans(Nat, Nat.mul(x1, Nat.mul(x0, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x1, x0, Nat.mul(x2, Nat.mul(x3, 1n))), {==})))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [1n, 2n, 3n, 0n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_1230(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [1n, 2n, 3n, 0n]), Axis.order(Nat, coords, [1n, 2n, 3n, 0n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [1n, 2n, 3n, 0n]), Axis.order(Nat, coords, [1n, 2n, 3n, 0n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [1n, 2n, 3n, 0n]), Axis.order(Nat, coords, [1n, 2n, 3n, 0n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [1n, 2n, 3n, 0n]), Axis.order(Nat, coords, [1n, 2n, 3n, 0n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [1n, 2n, 3n, 0n]), Axis.order(Nat, coords, [1n, 2n, 3n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 2n, 3n, 0n]), Axis.order(Nat, Nil{}, [1n, 2n, 3n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 2n, 3n, 0n]), Axis.order(Nat, c0 <> Nil{}, [1n, 2n, 3n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 2n, 3n, 0n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [1n, 2n, 3n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 2n, 3n, 0n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [1n, 2n, 3n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), True{})))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), True{})), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c0, d0), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c1, d1), value), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{}))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c1, d1), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{}))), {==})))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 2n, 3n, 0n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [1n, 2n, 3n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [1n, 2n, 3n, 0n]), Axis.order(Nat, coords, [1n, 2n, 3n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_1302(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [1n, 3n, 0n, 2n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 3n, 0n, 2n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 3n, 0n, 2n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 3n, 0n, 2n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 3n, 0n, 2n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [1n, 3n, 0n, 2n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_1302(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [1n, 3n, 0n, 2n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [1n, 3n, 0n, 2n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [1n, 3n, 0n, 2n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [1n, 3n, 0n, 2n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [1n, 3n, 0n, 2n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x1, Nat.mul(x3, Nat.mul(x0, Nat.mul(x2, 1n)))), Nat.mul(x1, Nat.mul(x0, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x1, value), Nat.mul(x3, Nat.mul(x0, Nat.mul(x2, 1n))), Nat.mul(x0, Nat.mul(x3, Nat.mul(x2, 1n))), mul_swap(x3, x0, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x1, Nat.mul(x0, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x1, x0, Nat.mul(x3, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x1, value)), Nat.mul(x3, Nat.mul(x2, 1n)), Nat.mul(x2, Nat.mul(x3, 1n)), mul_swap(x3, x2, 1n)), {==})))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [1n, 3n, 0n, 2n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_1302(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [1n, 3n, 0n, 2n]), Axis.order(Nat, coords, [1n, 3n, 0n, 2n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [1n, 3n, 0n, 2n]), Axis.order(Nat, coords, [1n, 3n, 0n, 2n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [1n, 3n, 0n, 2n]), Axis.order(Nat, coords, [1n, 3n, 0n, 2n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [1n, 3n, 0n, 2n]), Axis.order(Nat, coords, [1n, 3n, 0n, 2n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [1n, 3n, 0n, 2n]), Axis.order(Nat, coords, [1n, 3n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 3n, 0n, 2n]), Axis.order(Nat, Nil{}, [1n, 3n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 3n, 0n, 2n]), Axis.order(Nat, c0 <> Nil{}, [1n, 3n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 3n, 0n, 2n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [1n, 3n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 3n, 0n, 2n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [1n, 3n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c1, d1), value), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{}))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c1, d1), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c2, d2), True{})), {==})))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 3n, 0n, 2n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [1n, 3n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [1n, 3n, 0n, 2n]), Axis.order(Nat, coords, [1n, 3n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_1320(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [1n, 3n, 2n, 0n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 3n, 2n, 0n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 3n, 2n, 0n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 3n, 2n, 0n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [1n, 3n, 2n, 0n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [1n, 3n, 2n, 0n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_1320(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [1n, 3n, 2n, 0n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [1n, 3n, 2n, 0n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [1n, 3n, 2n, 0n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [1n, 3n, 2n, 0n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [1n, 3n, 2n, 0n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, Nat.mul(x0, 1n)))), Nat.mul(x1, Nat.mul(x3, Nat.mul(x0, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x1, Nat.mul(x3, value)), Nat.mul(x2, Nat.mul(x0, 1n)), Nat.mul(x0, Nat.mul(x2, 1n)), mul_swap(x2, x0, 1n)), Equal.trans(Nat, Nat.mul(x1, Nat.mul(x3, Nat.mul(x0, Nat.mul(x2, 1n)))), Nat.mul(x1, Nat.mul(x0, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x1, value), Nat.mul(x3, Nat.mul(x0, Nat.mul(x2, 1n))), Nat.mul(x0, Nat.mul(x3, Nat.mul(x2, 1n))), mul_swap(x3, x0, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x1, Nat.mul(x0, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x1, x0, Nat.mul(x3, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x1, value)), Nat.mul(x3, Nat.mul(x2, 1n)), Nat.mul(x2, Nat.mul(x3, 1n)), mul_swap(x3, x2, 1n)), {==}))))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [1n, 3n, 2n, 0n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_1320(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [1n, 3n, 2n, 0n]), Axis.order(Nat, coords, [1n, 3n, 2n, 0n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [1n, 3n, 2n, 0n]), Axis.order(Nat, coords, [1n, 3n, 2n, 0n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [1n, 3n, 2n, 0n]), Axis.order(Nat, coords, [1n, 3n, 2n, 0n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [1n, 3n, 2n, 0n]), Axis.order(Nat, coords, [1n, 3n, 2n, 0n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [1n, 3n, 2n, 0n]), Axis.order(Nat, coords, [1n, 3n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 3n, 2n, 0n]), Axis.order(Nat, Nil{}, [1n, 3n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 3n, 2n, 0n]), Axis.order(Nat, c0 <> Nil{}, [1n, 3n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 3n, 2n, 0n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [1n, 3n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 3n, 2n, 0n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [1n, 3n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), True{})))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), value)), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), True{})), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{})), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c0, d0), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c1, d1), value), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{}))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c1, d1), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c2, d2), True{})), {==}))))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [1n, 3n, 2n, 0n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [1n, 3n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [1n, 3n, 2n, 0n]), Axis.order(Nat, coords, [1n, 3n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_2013(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [2n, 0n, 1n, 3n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 0n, 1n, 3n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 0n, 1n, 3n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 0n, 1n, 3n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 0n, 1n, 3n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [2n, 0n, 1n, 3n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_2013(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [2n, 0n, 1n, 3n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [2n, 0n, 1n, 3n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [2n, 0n, 1n, 3n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [2n, 0n, 1n, 3n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [2n, 0n, 1n, 3n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x2, Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x2, x0, Nat.mul(x1, Nat.mul(x3, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n))), Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n))), mul_swap(x2, x1, Nat.mul(x3, 1n))), {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [2n, 0n, 1n, 3n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_2013(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [2n, 0n, 1n, 3n]), Axis.order(Nat, coords, [2n, 0n, 1n, 3n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [2n, 0n, 1n, 3n]), Axis.order(Nat, coords, [2n, 0n, 1n, 3n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [2n, 0n, 1n, 3n]), Axis.order(Nat, coords, [2n, 0n, 1n, 3n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [2n, 0n, 1n, 3n]), Axis.order(Nat, coords, [2n, 0n, 1n, 3n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [2n, 0n, 1n, 3n]), Axis.order(Nat, coords, [2n, 0n, 1n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 0n, 1n, 3n]), Axis.order(Nat, Nil{}, [2n, 0n, 1n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 0n, 1n, 3n]), Axis.order(Nat, c0 <> Nil{}, [2n, 0n, 1n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 0n, 1n, 3n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [2n, 0n, 1n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 0n, 1n, 3n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [2n, 0n, 1n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{}))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 0n, 1n, 3n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [2n, 0n, 1n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [2n, 0n, 1n, 3n]), Axis.order(Nat, coords, [2n, 0n, 1n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_2031(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [2n, 0n, 3n, 1n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 0n, 3n, 1n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 0n, 3n, 1n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 0n, 3n, 1n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 0n, 3n, 1n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [2n, 0n, 3n, 1n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_2031(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [2n, 0n, 3n, 1n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [2n, 0n, 3n, 1n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [2n, 0n, 3n, 1n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [2n, 0n, 3n, 1n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [2n, 0n, 3n, 1n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x2, Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x2, Nat.mul(x3, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x2, x0, Nat.mul(x3, Nat.mul(x1, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x2, Nat.mul(x3, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x2, value)), Nat.mul(x3, Nat.mul(x1, 1n)), Nat.mul(x1, Nat.mul(x3, 1n)), mul_swap(x3, x1, 1n)), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n))), Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n))), mul_swap(x2, x1, Nat.mul(x3, 1n))), {==})))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [2n, 0n, 3n, 1n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_2031(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [2n, 0n, 3n, 1n]), Axis.order(Nat, coords, [2n, 0n, 3n, 1n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [2n, 0n, 3n, 1n]), Axis.order(Nat, coords, [2n, 0n, 3n, 1n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [2n, 0n, 3n, 1n]), Axis.order(Nat, coords, [2n, 0n, 3n, 1n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [2n, 0n, 3n, 1n]), Axis.order(Nat, coords, [2n, 0n, 3n, 1n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [2n, 0n, 3n, 1n]), Axis.order(Nat, coords, [2n, 0n, 3n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 0n, 3n, 1n]), Axis.order(Nat, Nil{}, [2n, 0n, 3n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 0n, 3n, 1n]), Axis.order(Nat, c0 <> Nil{}, [2n, 0n, 3n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 0n, 3n, 1n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [2n, 0n, 3n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 0n, 3n, 1n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [2n, 0n, 3n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{})), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c1, d1), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{}))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), {==})))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 0n, 3n, 1n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [2n, 0n, 3n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [2n, 0n, 3n, 1n]), Axis.order(Nat, coords, [2n, 0n, 3n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_2103(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [2n, 1n, 0n, 3n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 1n, 0n, 3n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 1n, 0n, 3n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 1n, 0n, 3n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 1n, 0n, 3n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [2n, 1n, 0n, 3n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_2103(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [2n, 1n, 0n, 3n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [2n, 1n, 0n, 3n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [2n, 1n, 0n, 3n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [2n, 1n, 0n, 3n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [2n, 1n, 0n, 3n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x2, Nat.mul(x1, Nat.mul(x0, Nat.mul(x3, 1n)))), Nat.mul(x2, Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x2, value), Nat.mul(x1, Nat.mul(x0, Nat.mul(x3, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, 1n))), mul_swap(x1, x0, Nat.mul(x3, 1n))), Equal.trans(Nat, Nat.mul(x2, Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x2, x0, Nat.mul(x1, Nat.mul(x3, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n))), Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n))), mul_swap(x2, x1, Nat.mul(x3, 1n))), {==})))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [2n, 1n, 0n, 3n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_2103(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [2n, 1n, 0n, 3n]), Axis.order(Nat, coords, [2n, 1n, 0n, 3n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [2n, 1n, 0n, 3n]), Axis.order(Nat, coords, [2n, 1n, 0n, 3n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [2n, 1n, 0n, 3n]), Axis.order(Nat, coords, [2n, 1n, 0n, 3n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [2n, 1n, 0n, 3n]), Axis.order(Nat, coords, [2n, 1n, 0n, 3n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [2n, 1n, 0n, 3n]), Axis.order(Nat, coords, [2n, 1n, 0n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 1n, 0n, 3n]), Axis.order(Nat, Nil{}, [2n, 1n, 0n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 1n, 0n, 3n]), Axis.order(Nat, c0 <> Nil{}, [2n, 1n, 0n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 1n, 0n, 3n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [2n, 1n, 0n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 1n, 0n, 3n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [2n, 1n, 0n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c2, d2), value), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), and_swap(Nat.is_lt(c1, d1), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{}))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), {==})))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 1n, 0n, 3n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [2n, 1n, 0n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [2n, 1n, 0n, 3n]), Axis.order(Nat, coords, [2n, 1n, 0n, 3n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_2130(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [2n, 1n, 3n, 0n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 1n, 3n, 0n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 1n, 3n, 0n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 1n, 3n, 0n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 1n, 3n, 0n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [2n, 1n, 3n, 0n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_2130(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [2n, 1n, 3n, 0n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [2n, 1n, 3n, 0n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [2n, 1n, 3n, 0n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [2n, 1n, 3n, 0n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [2n, 1n, 3n, 0n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, Nat.mul(x0, 1n)))), Nat.mul(x2, Nat.mul(x1, Nat.mul(x0, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x2, Nat.mul(x1, value)), Nat.mul(x3, Nat.mul(x0, 1n)), Nat.mul(x0, Nat.mul(x3, 1n)), mul_swap(x3, x0, 1n)), Equal.trans(Nat, Nat.mul(x2, Nat.mul(x1, Nat.mul(x0, Nat.mul(x3, 1n)))), Nat.mul(x2, Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x2, value), Nat.mul(x1, Nat.mul(x0, Nat.mul(x3, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, 1n))), mul_swap(x1, x0, Nat.mul(x3, 1n))), Equal.trans(Nat, Nat.mul(x2, Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x2, x0, Nat.mul(x1, Nat.mul(x3, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n))), Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n))), mul_swap(x2, x1, Nat.mul(x3, 1n))), {==}))))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [2n, 1n, 3n, 0n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_2130(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [2n, 1n, 3n, 0n]), Axis.order(Nat, coords, [2n, 1n, 3n, 0n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [2n, 1n, 3n, 0n]), Axis.order(Nat, coords, [2n, 1n, 3n, 0n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [2n, 1n, 3n, 0n]), Axis.order(Nat, coords, [2n, 1n, 3n, 0n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [2n, 1n, 3n, 0n]), Axis.order(Nat, coords, [2n, 1n, 3n, 0n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [2n, 1n, 3n, 0n]), Axis.order(Nat, coords, [2n, 1n, 3n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 1n, 3n, 0n]), Axis.order(Nat, Nil{}, [2n, 1n, 3n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 1n, 3n, 0n]), Axis.order(Nat, c0 <> Nil{}, [2n, 1n, 3n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 1n, 3n, 0n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [2n, 1n, 3n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 1n, 3n, 0n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [2n, 1n, 3n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), True{})))), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), True{})), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c0, d0), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c2, d2), value), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), and_swap(Nat.is_lt(c1, d1), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{}))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), {==}))))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 1n, 3n, 0n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [2n, 1n, 3n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [2n, 1n, 3n, 0n]), Axis.order(Nat, coords, [2n, 1n, 3n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_2301(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [2n, 3n, 0n, 1n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 3n, 0n, 1n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 3n, 0n, 1n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 3n, 0n, 1n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 3n, 0n, 1n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [2n, 3n, 0n, 1n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_2301(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [2n, 3n, 0n, 1n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [2n, 3n, 0n, 1n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [2n, 3n, 0n, 1n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [2n, 3n, 0n, 1n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [2n, 3n, 0n, 1n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x2, Nat.mul(x3, Nat.mul(x0, Nat.mul(x1, 1n)))), Nat.mul(x2, Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x2, value), Nat.mul(x3, Nat.mul(x0, Nat.mul(x1, 1n))), Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, 1n))), mul_swap(x3, x0, Nat.mul(x1, 1n))), Equal.trans(Nat, Nat.mul(x2, Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x2, Nat.mul(x3, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x2, x0, Nat.mul(x3, Nat.mul(x1, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x2, Nat.mul(x3, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x2, value)), Nat.mul(x3, Nat.mul(x1, 1n)), Nat.mul(x1, Nat.mul(x3, 1n)), mul_swap(x3, x1, 1n)), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n))), Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n))), mul_swap(x2, x1, Nat.mul(x3, 1n))), {==}))))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [2n, 3n, 0n, 1n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_2301(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [2n, 3n, 0n, 1n]), Axis.order(Nat, coords, [2n, 3n, 0n, 1n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [2n, 3n, 0n, 1n]), Axis.order(Nat, coords, [2n, 3n, 0n, 1n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [2n, 3n, 0n, 1n]), Axis.order(Nat, coords, [2n, 3n, 0n, 1n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [2n, 3n, 0n, 1n]), Axis.order(Nat, coords, [2n, 3n, 0n, 1n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [2n, 3n, 0n, 1n]), Axis.order(Nat, coords, [2n, 3n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 3n, 0n, 1n]), Axis.order(Nat, Nil{}, [2n, 3n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 3n, 0n, 1n]), Axis.order(Nat, c0 <> Nil{}, [2n, 3n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 3n, 0n, 1n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [2n, 3n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 3n, 0n, 1n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [2n, 3n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c2, d2), value), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{}))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{})), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c1, d1), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{}))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), {==}))))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 3n, 0n, 1n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [2n, 3n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [2n, 3n, 0n, 1n]), Axis.order(Nat, coords, [2n, 3n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_2310(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [2n, 3n, 1n, 0n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 3n, 1n, 0n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 3n, 1n, 0n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 3n, 1n, 0n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [2n, 3n, 1n, 0n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [2n, 3n, 1n, 0n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_2310(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [2n, 3n, 1n, 0n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [2n, 3n, 1n, 0n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [2n, 3n, 1n, 0n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [2n, 3n, 1n, 0n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [2n, 3n, 1n, 0n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x2, Nat.mul(x3, Nat.mul(x1, Nat.mul(x0, 1n)))), Nat.mul(x2, Nat.mul(x3, Nat.mul(x0, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x2, Nat.mul(x3, value)), Nat.mul(x1, Nat.mul(x0, 1n)), Nat.mul(x0, Nat.mul(x1, 1n)), mul_swap(x1, x0, 1n)), Equal.trans(Nat, Nat.mul(x2, Nat.mul(x3, Nat.mul(x0, Nat.mul(x1, 1n)))), Nat.mul(x2, Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x2, value), Nat.mul(x3, Nat.mul(x0, Nat.mul(x1, 1n))), Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, 1n))), mul_swap(x3, x0, Nat.mul(x1, 1n))), Equal.trans(Nat, Nat.mul(x2, Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x2, Nat.mul(x3, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x2, x0, Nat.mul(x3, Nat.mul(x1, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x2, Nat.mul(x3, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x2, value)), Nat.mul(x3, Nat.mul(x1, 1n)), Nat.mul(x1, Nat.mul(x3, 1n)), mul_swap(x3, x1, 1n)), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x2, Nat.mul(x1, Nat.mul(x3, 1n))), Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n))), mul_swap(x2, x1, Nat.mul(x3, 1n))), {==})))))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [2n, 3n, 1n, 0n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_2310(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [2n, 3n, 1n, 0n]), Axis.order(Nat, coords, [2n, 3n, 1n, 0n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [2n, 3n, 1n, 0n]), Axis.order(Nat, coords, [2n, 3n, 1n, 0n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [2n, 3n, 1n, 0n]), Axis.order(Nat, coords, [2n, 3n, 1n, 0n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [2n, 3n, 1n, 0n]), Axis.order(Nat, coords, [2n, 3n, 1n, 0n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [2n, 3n, 1n, 0n]), Axis.order(Nat, coords, [2n, 3n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 3n, 1n, 0n]), Axis.order(Nat, Nil{}, [2n, 3n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 3n, 1n, 0n]), Axis.order(Nat, c0 <> Nil{}, [2n, 3n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 3n, 1n, 0n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [2n, 3n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 3n, 1n, 0n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [2n, 3n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), True{})))), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), value)), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), True{})), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{})), and_swap(Nat.is_lt(c1, d1), Nat.is_lt(c0, d0), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c2, d2), value), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{}))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), True{})), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c1, d1), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{}))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), True{}))), {==})))))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [2n, 3n, 1n, 0n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [2n, 3n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [2n, 3n, 1n, 0n]), Axis.order(Nat, coords, [2n, 3n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_3012(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [3n, 0n, 1n, 2n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 0n, 1n, 2n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 0n, 1n, 2n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 0n, 1n, 2n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 0n, 1n, 2n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [3n, 0n, 1n, 2n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_3012(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [3n, 0n, 1n, 2n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [3n, 0n, 1n, 2n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [3n, 0n, 1n, 2n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [3n, 0n, 1n, 2n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [3n, 0n, 1n, 2n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x3, Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x3, x0, Nat.mul(x1, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n))), Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n))), mul_swap(x3, x1, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x1, value)), Nat.mul(x3, Nat.mul(x2, 1n)), Nat.mul(x2, Nat.mul(x3, 1n)), mul_swap(x3, x2, 1n)), {==})))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [3n, 0n, 1n, 2n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_3012(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [3n, 0n, 1n, 2n]), Axis.order(Nat, coords, [3n, 0n, 1n, 2n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [3n, 0n, 1n, 2n]), Axis.order(Nat, coords, [3n, 0n, 1n, 2n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [3n, 0n, 1n, 2n]), Axis.order(Nat, coords, [3n, 0n, 1n, 2n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [3n, 0n, 1n, 2n]), Axis.order(Nat, coords, [3n, 0n, 1n, 2n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [3n, 0n, 1n, 2n]), Axis.order(Nat, coords, [3n, 0n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 0n, 1n, 2n]), Axis.order(Nat, Nil{}, [3n, 0n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 0n, 1n, 2n]), Axis.order(Nat, c0 <> Nil{}, [3n, 0n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 0n, 1n, 2n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [3n, 0n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 0n, 1n, 2n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [3n, 0n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{}))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c2, d2), True{})), {==})))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 0n, 1n, 2n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [3n, 0n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [3n, 0n, 1n, 2n]), Axis.order(Nat, coords, [3n, 0n, 1n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_3021(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [3n, 0n, 2n, 1n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 0n, 2n, 1n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 0n, 2n, 1n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 0n, 2n, 1n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 0n, 2n, 1n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [3n, 0n, 2n, 1n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_3021(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [3n, 0n, 2n, 1n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [3n, 0n, 2n, 1n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [3n, 0n, 2n, 1n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [3n, 0n, 2n, 1n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [3n, 0n, 2n, 1n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x3, Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x3, Nat.mul(x2, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x3, x0, Nat.mul(x2, Nat.mul(x1, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x3, Nat.mul(x2, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x3, value)), Nat.mul(x2, Nat.mul(x1, 1n)), Nat.mul(x1, Nat.mul(x2, 1n)), mul_swap(x2, x1, 1n)), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n))), Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n))), mul_swap(x3, x1, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x1, value)), Nat.mul(x3, Nat.mul(x2, 1n)), Nat.mul(x2, Nat.mul(x3, 1n)), mul_swap(x3, x2, 1n)), {==}))))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [3n, 0n, 2n, 1n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_3021(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [3n, 0n, 2n, 1n]), Axis.order(Nat, coords, [3n, 0n, 2n, 1n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [3n, 0n, 2n, 1n]), Axis.order(Nat, coords, [3n, 0n, 2n, 1n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [3n, 0n, 2n, 1n]), Axis.order(Nat, coords, [3n, 0n, 2n, 1n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [3n, 0n, 2n, 1n]), Axis.order(Nat, coords, [3n, 0n, 2n, 1n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [3n, 0n, 2n, 1n]), Axis.order(Nat, coords, [3n, 0n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 0n, 2n, 1n]), Axis.order(Nat, Nil{}, [3n, 0n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 0n, 2n, 1n]), Axis.order(Nat, c0 <> Nil{}, [3n, 0n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 0n, 2n, 1n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [3n, 0n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 0n, 2n, 1n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [3n, 0n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), value)), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c1, d1), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{}))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c2, d2), True{})), {==}))))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 0n, 2n, 1n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [3n, 0n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [3n, 0n, 2n, 1n]), Axis.order(Nat, coords, [3n, 0n, 2n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_3102(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [3n, 1n, 0n, 2n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 1n, 0n, 2n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 1n, 0n, 2n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 1n, 0n, 2n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 1n, 0n, 2n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [3n, 1n, 0n, 2n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_3102(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [3n, 1n, 0n, 2n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [3n, 1n, 0n, 2n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [3n, 1n, 0n, 2n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [3n, 1n, 0n, 2n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [3n, 1n, 0n, 2n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x3, Nat.mul(x1, Nat.mul(x0, Nat.mul(x2, 1n)))), Nat.mul(x3, Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x3, value), Nat.mul(x1, Nat.mul(x0, Nat.mul(x2, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n))), mul_swap(x1, x0, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x3, Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x3, x0, Nat.mul(x1, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n))), Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n))), mul_swap(x3, x1, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x1, value)), Nat.mul(x3, Nat.mul(x2, 1n)), Nat.mul(x2, Nat.mul(x3, 1n)), mul_swap(x3, x2, 1n)), {==}))))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [3n, 1n, 0n, 2n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_3102(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [3n, 1n, 0n, 2n]), Axis.order(Nat, coords, [3n, 1n, 0n, 2n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [3n, 1n, 0n, 2n]), Axis.order(Nat, coords, [3n, 1n, 0n, 2n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [3n, 1n, 0n, 2n]), Axis.order(Nat, coords, [3n, 1n, 0n, 2n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [3n, 1n, 0n, 2n]), Axis.order(Nat, coords, [3n, 1n, 0n, 2n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [3n, 1n, 0n, 2n]), Axis.order(Nat, coords, [3n, 1n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 1n, 0n, 2n]), Axis.order(Nat, Nil{}, [3n, 1n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 1n, 0n, 2n]), Axis.order(Nat, c0 <> Nil{}, [3n, 1n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 1n, 0n, 2n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [3n, 1n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 1n, 0n, 2n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [3n, 1n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c3, d3), value), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), and_swap(Nat.is_lt(c1, d1), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{}))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c2, d2), True{})), {==}))))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 1n, 0n, 2n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [3n, 1n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [3n, 1n, 0n, 2n]), Axis.order(Nat, coords, [3n, 1n, 0n, 2n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_3120(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [3n, 1n, 2n, 0n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 1n, 2n, 0n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 1n, 2n, 0n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 1n, 2n, 0n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 1n, 2n, 0n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [3n, 1n, 2n, 0n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_3120(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [3n, 1n, 2n, 0n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [3n, 1n, 2n, 0n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [3n, 1n, 2n, 0n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [3n, 1n, 2n, 0n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [3n, 1n, 2n, 0n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, Nat.mul(x0, 1n)))), Nat.mul(x3, Nat.mul(x1, Nat.mul(x0, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x3, Nat.mul(x1, value)), Nat.mul(x2, Nat.mul(x0, 1n)), Nat.mul(x0, Nat.mul(x2, 1n)), mul_swap(x2, x0, 1n)), Equal.trans(Nat, Nat.mul(x3, Nat.mul(x1, Nat.mul(x0, Nat.mul(x2, 1n)))), Nat.mul(x3, Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x3, value), Nat.mul(x1, Nat.mul(x0, Nat.mul(x2, 1n))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n))), mul_swap(x1, x0, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x3, Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x3, x0, Nat.mul(x1, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n))), Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n))), mul_swap(x3, x1, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x1, value)), Nat.mul(x3, Nat.mul(x2, 1n)), Nat.mul(x2, Nat.mul(x3, 1n)), mul_swap(x3, x2, 1n)), {==})))))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [3n, 1n, 2n, 0n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_3120(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [3n, 1n, 2n, 0n]), Axis.order(Nat, coords, [3n, 1n, 2n, 0n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [3n, 1n, 2n, 0n]), Axis.order(Nat, coords, [3n, 1n, 2n, 0n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [3n, 1n, 2n, 0n]), Axis.order(Nat, coords, [3n, 1n, 2n, 0n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [3n, 1n, 2n, 0n]), Axis.order(Nat, coords, [3n, 1n, 2n, 0n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [3n, 1n, 2n, 0n]), Axis.order(Nat, coords, [3n, 1n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 1n, 2n, 0n]), Axis.order(Nat, Nil{}, [3n, 1n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 1n, 2n, 0n]), Axis.order(Nat, c0 <> Nil{}, [3n, 1n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 1n, 2n, 0n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [3n, 1n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 1n, 2n, 0n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [3n, 1n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), True{})))), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), value)), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), True{})), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{})), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c0, d0), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c3, d3), value), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), and_swap(Nat.is_lt(c1, d1), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{}))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c2, d2), True{})), {==})))))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 1n, 2n, 0n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [3n, 1n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [3n, 1n, 2n, 0n]), Axis.order(Nat, coords, [3n, 1n, 2n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_3201(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [3n, 2n, 0n, 1n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 2n, 0n, 1n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 2n, 0n, 1n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 2n, 0n, 1n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 2n, 0n, 1n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [3n, 2n, 0n, 1n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_3201(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [3n, 2n, 0n, 1n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [3n, 2n, 0n, 1n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [3n, 2n, 0n, 1n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [3n, 2n, 0n, 1n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [3n, 2n, 0n, 1n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x3, Nat.mul(x2, Nat.mul(x0, Nat.mul(x1, 1n)))), Nat.mul(x3, Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x3, value), Nat.mul(x2, Nat.mul(x0, Nat.mul(x1, 1n))), Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, 1n))), mul_swap(x2, x0, Nat.mul(x1, 1n))), Equal.trans(Nat, Nat.mul(x3, Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x3, Nat.mul(x2, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x3, x0, Nat.mul(x2, Nat.mul(x1, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x3, Nat.mul(x2, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x3, value)), Nat.mul(x2, Nat.mul(x1, 1n)), Nat.mul(x1, Nat.mul(x2, 1n)), mul_swap(x2, x1, 1n)), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n))), Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n))), mul_swap(x3, x1, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x1, value)), Nat.mul(x3, Nat.mul(x2, 1n)), Nat.mul(x2, Nat.mul(x3, 1n)), mul_swap(x3, x2, 1n)), {==})))))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [3n, 2n, 0n, 1n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_3201(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [3n, 2n, 0n, 1n]), Axis.order(Nat, coords, [3n, 2n, 0n, 1n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [3n, 2n, 0n, 1n]), Axis.order(Nat, coords, [3n, 2n, 0n, 1n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [3n, 2n, 0n, 1n]), Axis.order(Nat, coords, [3n, 2n, 0n, 1n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [3n, 2n, 0n, 1n]), Axis.order(Nat, coords, [3n, 2n, 0n, 1n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [3n, 2n, 0n, 1n]), Axis.order(Nat, coords, [3n, 2n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 2n, 0n, 1n]), Axis.order(Nat, Nil{}, [3n, 2n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 2n, 0n, 1n]), Axis.order(Nat, c0 <> Nil{}, [3n, 2n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 2n, 0n, 1n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [3n, 2n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 2n, 0n, 1n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [3n, 2n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c3, d3), value), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{}))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), value)), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c1, d1), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{}))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c2, d2), True{})), {==})))))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 2n, 0n, 1n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [3n, 2n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [3n, 2n, 0n, 1n]), Axis.order(Nat, coords, [3n, 2n, 0n, 1n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def lists_3210(-A: Data, +items: List<&2, A>, length: {List.length(&2, A, items) == 4n : Nat}) -> ListFacts<A, 4n, [3n, 2n, 1n, 0n], items>:
  match items:
    case Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 2n, 1n, 0n], Nil{}>, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 2n, 1n, 0n], x0 <> Nil{}>, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 2n, 1n, 0n], x0 <> x1 <> Nil{}>, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd(ListFacts<A, 4n, [3n, 2n, 1n, 0n], x0 <> x1 <> x2 <> Nil{}>, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Lists{{==}, {==}, {==}, {==}, {==}, {==}}
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd(ListFacts<A, 4n, [3n, 2n, 1n, 0n], x0 <> x1 <> x2 <> x3 <> x4 <> tail>, nat_clash(5n+List.length(&2, A, tail), 4n, length, {==}))

def product_3210(+dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == 4n : Nat}) -> {Shape.numel(Axis.order(Nat, dims, [3n, 2n, 1n, 0n])) == Shape.numel(dims) : Nat}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, Nil{}, [3n, 2n, 1n, 0n])) == Shape.numel(Nil{}) : Nat}, nat_clash(0n, 4n, length, {==}))
    case +x0 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> Nil{}, [3n, 2n, 1n, 0n])) == Shape.numel(x0 <> Nil{}) : Nat}, nat_clash(1n, 4n, length, {==}))
    case +x0 <> +x1 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> Nil{}, [3n, 2n, 1n, 0n])) == Shape.numel(x0 <> x1 <> Nil{}) : Nat}, nat_clash(2n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> Nil{}:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> Nil{}, [3n, 2n, 1n, 0n])) == Shape.numel(x0 <> x1 <> x2 <> Nil{}) : Nat}, nat_clash(3n, 4n, length, {==}))
    case +x0 <> +x1 <> +x2 <> +x3 <> Nil{}:
      Equal.trans(Nat, Nat.mul(x3, Nat.mul(x2, Nat.mul(x1, Nat.mul(x0, 1n)))), Nat.mul(x3, Nat.mul(x2, Nat.mul(x0, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x3, Nat.mul(x2, value)), Nat.mul(x1, Nat.mul(x0, 1n)), Nat.mul(x0, Nat.mul(x1, 1n)), mul_swap(x1, x0, 1n)), Equal.trans(Nat, Nat.mul(x3, Nat.mul(x2, Nat.mul(x0, Nat.mul(x1, 1n)))), Nat.mul(x3, Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x3, value), Nat.mul(x2, Nat.mul(x0, Nat.mul(x1, 1n))), Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, 1n))), mul_swap(x2, x0, Nat.mul(x1, 1n))), Equal.trans(Nat, Nat.mul(x3, Nat.mul(x0, Nat.mul(x2, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x3, Nat.mul(x2, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), mul_swap(x3, x0, Nat.mul(x2, Nat.mul(x1, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x3, Nat.mul(x2, Nat.mul(x1, 1n)))), Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x3, value)), Nat.mul(x2, Nat.mul(x1, 1n)), Nat.mul(x1, Nat.mul(x2, 1n)), mul_swap(x2, x1, 1n)), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, value), Nat.mul(x3, Nat.mul(x1, Nat.mul(x2, 1n))), Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n))), mul_swap(x3, x1, Nat.mul(x2, 1n))), Equal.trans(Nat, Nat.mul(x0, Nat.mul(x1, Nat.mul(x3, Nat.mul(x2, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Nat.mul(x0, Nat.mul(x1, Nat.mul(x2, Nat.mul(x3, 1n)))), Equal.cong(Nat, Nat, value => Nat.mul(x0, Nat.mul(x1, value)), Nat.mul(x3, Nat.mul(x2, 1n)), Nat.mul(x2, Nat.mul(x3, 1n)), mul_swap(x3, x2, 1n)), {==}))))))
    case +x0 <> +x1 <> +x2 <> +x3 <> +x4 <> +tail:
      Empty.absurd({Shape.numel(Axis.order(Nat, x0 <> x1 <> x2 <> x3 <> x4 <> tail, [3n, 2n, 1n, 0n])) == Shape.numel(x0 <> x1 <> x2 <> x3 <> x4 <> tail) : Nat}, nat_clash(5n+List.length(&2, Nat, tail), 4n, length, {==}))

def fit_3210(+dims: List<&2, Nat>, +coords: List<&2, Nat>, dl: {List.length(&2, Nat, dims) == 4n : Nat}, cl: {List.length(&2, Nat, coords) == 4n : Nat}) -> {Shape.fit(Axis.order(Nat, dims, [3n, 2n, 1n, 0n]), Axis.order(Nat, coords, [3n, 2n, 1n, 0n])) == Shape.fit(dims, coords) : Bool}:
  match dims:
    case Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, Nil{}, [3n, 2n, 1n, 0n]), Axis.order(Nat, coords, [3n, 2n, 1n, 0n])) == Shape.fit(Nil{}, coords) : Bool}, nat_clash(0n, 4n, dl, {==}))
    case +d0 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> Nil{}, [3n, 2n, 1n, 0n]), Axis.order(Nat, coords, [3n, 2n, 1n, 0n])) == Shape.fit(d0 <> Nil{}, coords) : Bool}, nat_clash(1n, 4n, dl, {==}))
    case +d0 <> +d1 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> Nil{}, [3n, 2n, 1n, 0n]), Axis.order(Nat, coords, [3n, 2n, 1n, 0n])) == Shape.fit(d0 <> d1 <> Nil{}, coords) : Bool}, nat_clash(2n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> Nil{}:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> Nil{}, [3n, 2n, 1n, 0n]), Axis.order(Nat, coords, [3n, 2n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> Nil{}, coords) : Bool}, nat_clash(3n, 4n, dl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> Nil{}:
      match coords:
        case Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 2n, 1n, 0n]), Axis.order(Nat, Nil{}, [3n, 2n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, Nil{}) : Bool}, nat_clash(0n, 4n, cl, {==}))
        case +c0 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 2n, 1n, 0n]), Axis.order(Nat, c0 <> Nil{}, [3n, 2n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> Nil{}) : Bool}, nat_clash(1n, 4n, cl, {==}))
        case +c0 <> +c1 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 2n, 1n, 0n]), Axis.order(Nat, c0 <> c1 <> Nil{}, [3n, 2n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> Nil{}) : Bool}, nat_clash(2n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> Nil{}:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 2n, 1n, 0n]), Axis.order(Nat, c0 <> c1 <> c2 <> Nil{}, [3n, 2n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> Nil{}) : Bool}, nat_clash(3n, 4n, cl, {==}))
        case +c0 <> +c1 <> +c2 <> +c3 <> Nil{}:
          Equal.trans(Bool, Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), True{})))), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), value)), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c0, d0), True{})), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{})), and_swap(Nat.is_lt(c1, d1), Nat.is_lt(c0, d0), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c3, d3), value), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{}))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{}))), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), value)), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c1, d1), True{})), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})), and_swap(Nat.is_lt(c2, d2), Nat.is_lt(c1, d1), True{})), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), value), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{}))), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), True{}))), Equal.trans(Bool, Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})))), Equal.cong(Bool, Bool, value => Bool.and(Nat.is_lt(c0, d0), Bool.and(Nat.is_lt(c1, d1), value)), Bool.and(Nat.is_lt(c3, d3), Bool.and(Nat.is_lt(c2, d2), True{})), Bool.and(Nat.is_lt(c2, d2), Bool.and(Nat.is_lt(c3, d3), True{})), and_swap(Nat.is_lt(c3, d3), Nat.is_lt(c2, d2), True{})), {==}))))))
        case +c0 <> +c1 <> +c2 <> +c3 <> +c4 <> +ct:
          Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> Nil{}, [3n, 2n, 1n, 0n]), Axis.order(Nat, c0 <> c1 <> c2 <> c3 <> c4 <> ct, [3n, 2n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> Nil{}, c0 <> c1 <> c2 <> c3 <> c4 <> ct) : Bool}, nat_clash(5n+List.length(&2, Nat, ct), 4n, cl, {==}))
    case +d0 <> +d1 <> +d2 <> +d3 <> +d4 <> +dt:
      Empty.absurd({Shape.fit(Axis.order(Nat, d0 <> d1 <> d2 <> d3 <> d4 <> dt, [3n, 2n, 1n, 0n]), Axis.order(Nat, coords, [3n, 2n, 1n, 0n])) == Shape.fit(d0 <> d1 <> d2 <> d3 <> d4 <> dt, coords) : Bool}, nat_clash(5n+List.length(&2, Nat, dt), 4n, dl, {==}))

def facts_0(+axes: List<&2, Nat>, accepted: {True{} == Axis.valid(0n, axes) : Bool}) -> Facts<0n, axes>:
  match axes:
    case Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_empty(A, items, length),
        dims => length => product_empty(dims, length), dims => coords => dl => cl => fit_empty(dims, coords, dl, cl)}
    case +a0 <> +tail:
      Empty.absurd(Facts<0n, a0 <> tail>, Shape.clash(accepted))

def facts_1(+axes: List<&2, Nat>, accepted: {True{} == Axis.valid(1n, axes) : Bool}) -> Facts<1n, axes>:
  match axes:
    case Nil{}:
      Empty.absurd(Facts<1n, Nil{}>, Shape.clash(accepted))
    case 0n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_0(A, items, length),
        dims => length => product_0(dims, length), dims => coords => dl => cl => fit_0(dims, coords, dl, cl)}
    case (1n+p) <> Nil{}:
      Empty.absurd(Facts<1n, (1n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(1n+p, 1n), Axis.bounded(Nil{}, 1n), Arith.and_left(Axis.bounded((1n+p) <> Nil{}, 1n), Axis.unique((1n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(1n, (1n+p) <> Nil{}), accepted)))))
    case +a0 <> +a1 <> +tail:
      Empty.absurd(Facts<1n, a0 <> a1 <> tail>, Shape.clash(accepted))

def facts_2(+axes: List<&2, Nat>, accepted: {True{} == Axis.valid(2n, axes) : Bool}) -> Facts<2n, axes>:
  match axes:
    case Nil{}:
      Empty.absurd(Facts<2n, Nil{}>, Shape.clash(accepted))
    case +a0 <> Nil{}:
      Empty.absurd(Facts<2n, a0 <> Nil{}>, Shape.clash(accepted))
    case 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<2n, [0n, 0n]>, Shape.clash(accepted))
    case 0n <> 1n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_01(A, items, length),
        dims => length => product_01(dims, length), dims => coords => dl => cl => fit_01(dims, coords, dl, cl)}
    case 0n <> (2n+p) <> Nil{}:
      Empty.absurd(Facts<2n, 0n <> (2n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(2n+p, 2n), Axis.bounded(Nil{}, 2n), Arith.and_left(Axis.bounded(0n <> (2n+p) <> Nil{}, 2n), Axis.unique(0n <> (2n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(2n, 0n <> (2n+p) <> Nil{}), accepted)))))
    case 1n <> 0n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_10(A, items, length),
        dims => length => product_10(dims, length), dims => coords => dl => cl => fit_10(dims, coords, dl, cl)}
    case 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<2n, [1n, 1n]>, Shape.clash(accepted))
    case 1n <> (2n+p) <> Nil{}:
      Empty.absurd(Facts<2n, 1n <> (2n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(2n+p, 2n), Axis.bounded(Nil{}, 2n), Arith.and_left(Axis.bounded(1n <> (2n+p) <> Nil{}, 2n), Axis.unique(1n <> (2n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(2n, 1n <> (2n+p) <> Nil{}), accepted)))))
    case (2n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<2n, (2n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(2n+p, 2n), Axis.bounded(a0 <> Nil{}, 2n), Arith.and_left(Axis.bounded((2n+p) <> a0 <> Nil{}, 2n), Axis.unique((2n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(2n, (2n+p) <> a0 <> Nil{}), accepted)))))
    case +a0 <> +a1 <> +a2 <> +tail:
      Empty.absurd(Facts<2n, a0 <> a1 <> a2 <> tail>, Shape.clash(accepted))

def facts_3(+axes: List<&2, Nat>, accepted: {True{} == Axis.valid(3n, axes) : Bool}) -> Facts<3n, axes>:
  match axes:
    case Nil{}:
      Empty.absurd(Facts<3n, Nil{}>, Shape.clash(accepted))
    case +a0 <> Nil{}:
      Empty.absurd(Facts<3n, a0 <> Nil{}>, Shape.clash(accepted))
    case +a0 <> +a1 <> Nil{}:
      Empty.absurd(Facts<3n, a0 <> a1 <> Nil{}>, Shape.clash(accepted))
    case 0n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<3n, [0n, 0n, 0n]>, Shape.clash(accepted))
    case 0n <> 0n <> 1n <> Nil{}:
      Empty.absurd(Facts<3n, [0n, 0n, 1n]>, Shape.clash(accepted))
    case 0n <> 0n <> 2n <> Nil{}:
      Empty.absurd(Facts<3n, [0n, 0n, 2n]>, Shape.clash(accepted))
    case 0n <> 0n <> (3n+p) <> Nil{}:
      Empty.absurd(Facts<3n, 0n <> 0n <> (3n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(3n+p, 3n), Axis.bounded(Nil{}, 3n), Arith.and_left(Axis.bounded(0n <> 0n <> (3n+p) <> Nil{}, 3n), Axis.unique(0n <> 0n <> (3n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(3n, 0n <> 0n <> (3n+p) <> Nil{}), accepted)))))
    case 0n <> 1n <> 0n <> Nil{}:
      Empty.absurd(Facts<3n, [0n, 1n, 0n]>, Shape.clash(accepted))
    case 0n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<3n, [0n, 1n, 1n]>, Shape.clash(accepted))
    case 0n <> 1n <> 2n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_012(A, items, length),
        dims => length => product_012(dims, length), dims => coords => dl => cl => fit_012(dims, coords, dl, cl)}
    case 0n <> 1n <> (3n+p) <> Nil{}:
      Empty.absurd(Facts<3n, 0n <> 1n <> (3n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(3n+p, 3n), Axis.bounded(Nil{}, 3n), Arith.and_left(Axis.bounded(0n <> 1n <> (3n+p) <> Nil{}, 3n), Axis.unique(0n <> 1n <> (3n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(3n, 0n <> 1n <> (3n+p) <> Nil{}), accepted)))))
    case 0n <> 2n <> 0n <> Nil{}:
      Empty.absurd(Facts<3n, [0n, 2n, 0n]>, Shape.clash(accepted))
    case 0n <> 2n <> 1n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_021(A, items, length),
        dims => length => product_021(dims, length), dims => coords => dl => cl => fit_021(dims, coords, dl, cl)}
    case 0n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<3n, [0n, 2n, 2n]>, Shape.clash(accepted))
    case 0n <> 2n <> (3n+p) <> Nil{}:
      Empty.absurd(Facts<3n, 0n <> 2n <> (3n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(3n+p, 3n), Axis.bounded(Nil{}, 3n), Arith.and_left(Axis.bounded(0n <> 2n <> (3n+p) <> Nil{}, 3n), Axis.unique(0n <> 2n <> (3n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(3n, 0n <> 2n <> (3n+p) <> Nil{}), accepted)))))
    case 0n <> (3n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<3n, 0n <> (3n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(3n+p, 3n), Axis.bounded(a0 <> Nil{}, 3n), Arith.and_left(Axis.bounded(0n <> (3n+p) <> a0 <> Nil{}, 3n), Axis.unique(0n <> (3n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(3n, 0n <> (3n+p) <> a0 <> Nil{}), accepted)))))
    case 1n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<3n, [1n, 0n, 0n]>, Shape.clash(accepted))
    case 1n <> 0n <> 1n <> Nil{}:
      Empty.absurd(Facts<3n, [1n, 0n, 1n]>, Shape.clash(accepted))
    case 1n <> 0n <> 2n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_102(A, items, length),
        dims => length => product_102(dims, length), dims => coords => dl => cl => fit_102(dims, coords, dl, cl)}
    case 1n <> 0n <> (3n+p) <> Nil{}:
      Empty.absurd(Facts<3n, 1n <> 0n <> (3n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(3n+p, 3n), Axis.bounded(Nil{}, 3n), Arith.and_left(Axis.bounded(1n <> 0n <> (3n+p) <> Nil{}, 3n), Axis.unique(1n <> 0n <> (3n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(3n, 1n <> 0n <> (3n+p) <> Nil{}), accepted)))))
    case 1n <> 1n <> 0n <> Nil{}:
      Empty.absurd(Facts<3n, [1n, 1n, 0n]>, Shape.clash(accepted))
    case 1n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<3n, [1n, 1n, 1n]>, Shape.clash(accepted))
    case 1n <> 1n <> 2n <> Nil{}:
      Empty.absurd(Facts<3n, [1n, 1n, 2n]>, Shape.clash(accepted))
    case 1n <> 1n <> (3n+p) <> Nil{}:
      Empty.absurd(Facts<3n, 1n <> 1n <> (3n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(3n+p, 3n), Axis.bounded(Nil{}, 3n), Arith.and_left(Axis.bounded(1n <> 1n <> (3n+p) <> Nil{}, 3n), Axis.unique(1n <> 1n <> (3n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(3n, 1n <> 1n <> (3n+p) <> Nil{}), accepted)))))
    case 1n <> 2n <> 0n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_120(A, items, length),
        dims => length => product_120(dims, length), dims => coords => dl => cl => fit_120(dims, coords, dl, cl)}
    case 1n <> 2n <> 1n <> Nil{}:
      Empty.absurd(Facts<3n, [1n, 2n, 1n]>, Shape.clash(accepted))
    case 1n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<3n, [1n, 2n, 2n]>, Shape.clash(accepted))
    case 1n <> 2n <> (3n+p) <> Nil{}:
      Empty.absurd(Facts<3n, 1n <> 2n <> (3n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(3n+p, 3n), Axis.bounded(Nil{}, 3n), Arith.and_left(Axis.bounded(1n <> 2n <> (3n+p) <> Nil{}, 3n), Axis.unique(1n <> 2n <> (3n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(3n, 1n <> 2n <> (3n+p) <> Nil{}), accepted)))))
    case 1n <> (3n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<3n, 1n <> (3n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(3n+p, 3n), Axis.bounded(a0 <> Nil{}, 3n), Arith.and_left(Axis.bounded(1n <> (3n+p) <> a0 <> Nil{}, 3n), Axis.unique(1n <> (3n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(3n, 1n <> (3n+p) <> a0 <> Nil{}), accepted)))))
    case 2n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<3n, [2n, 0n, 0n]>, Shape.clash(accepted))
    case 2n <> 0n <> 1n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_201(A, items, length),
        dims => length => product_201(dims, length), dims => coords => dl => cl => fit_201(dims, coords, dl, cl)}
    case 2n <> 0n <> 2n <> Nil{}:
      Empty.absurd(Facts<3n, [2n, 0n, 2n]>, Shape.clash(accepted))
    case 2n <> 0n <> (3n+p) <> Nil{}:
      Empty.absurd(Facts<3n, 2n <> 0n <> (3n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(3n+p, 3n), Axis.bounded(Nil{}, 3n), Arith.and_left(Axis.bounded(2n <> 0n <> (3n+p) <> Nil{}, 3n), Axis.unique(2n <> 0n <> (3n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(3n, 2n <> 0n <> (3n+p) <> Nil{}), accepted)))))
    case 2n <> 1n <> 0n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_210(A, items, length),
        dims => length => product_210(dims, length), dims => coords => dl => cl => fit_210(dims, coords, dl, cl)}
    case 2n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<3n, [2n, 1n, 1n]>, Shape.clash(accepted))
    case 2n <> 1n <> 2n <> Nil{}:
      Empty.absurd(Facts<3n, [2n, 1n, 2n]>, Shape.clash(accepted))
    case 2n <> 1n <> (3n+p) <> Nil{}:
      Empty.absurd(Facts<3n, 2n <> 1n <> (3n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(3n+p, 3n), Axis.bounded(Nil{}, 3n), Arith.and_left(Axis.bounded(2n <> 1n <> (3n+p) <> Nil{}, 3n), Axis.unique(2n <> 1n <> (3n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(3n, 2n <> 1n <> (3n+p) <> Nil{}), accepted)))))
    case 2n <> 2n <> 0n <> Nil{}:
      Empty.absurd(Facts<3n, [2n, 2n, 0n]>, Shape.clash(accepted))
    case 2n <> 2n <> 1n <> Nil{}:
      Empty.absurd(Facts<3n, [2n, 2n, 1n]>, Shape.clash(accepted))
    case 2n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<3n, [2n, 2n, 2n]>, Shape.clash(accepted))
    case 2n <> 2n <> (3n+p) <> Nil{}:
      Empty.absurd(Facts<3n, 2n <> 2n <> (3n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(3n+p, 3n), Axis.bounded(Nil{}, 3n), Arith.and_left(Axis.bounded(2n <> 2n <> (3n+p) <> Nil{}, 3n), Axis.unique(2n <> 2n <> (3n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(3n, 2n <> 2n <> (3n+p) <> Nil{}), accepted)))))
    case 2n <> (3n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<3n, 2n <> (3n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(3n+p, 3n), Axis.bounded(a0 <> Nil{}, 3n), Arith.and_left(Axis.bounded(2n <> (3n+p) <> a0 <> Nil{}, 3n), Axis.unique(2n <> (3n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(3n, 2n <> (3n+p) <> a0 <> Nil{}), accepted)))))
    case (3n+p) <> a0 <> a1 <> Nil{}:
      Empty.absurd(Facts<3n, (3n+p) <> a0 <> a1 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(3n+p, 3n), Axis.bounded(a0 <> a1 <> Nil{}, 3n), Arith.and_left(Axis.bounded((3n+p) <> a0 <> a1 <> Nil{}, 3n), Axis.unique((3n+p) <> a0 <> a1 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(3n, (3n+p) <> a0 <> a1 <> Nil{}), accepted)))))
    case +a0 <> +a1 <> +a2 <> +a3 <> +tail:
      Empty.absurd(Facts<3n, a0 <> a1 <> a2 <> a3 <> tail>, Shape.clash(accepted))

def facts_4(+axes: List<&2, Nat>, accepted: {True{} == Axis.valid(4n, axes) : Bool}) -> Facts<4n, axes>:
  match axes:
    case Nil{}:
      Empty.absurd(Facts<4n, Nil{}>, Shape.clash(accepted))
    case +a0 <> Nil{}:
      Empty.absurd(Facts<4n, a0 <> Nil{}>, Shape.clash(accepted))
    case +a0 <> +a1 <> Nil{}:
      Empty.absurd(Facts<4n, a0 <> a1 <> Nil{}>, Shape.clash(accepted))
    case +a0 <> +a1 <> +a2 <> Nil{}:
      Empty.absurd(Facts<4n, a0 <> a1 <> a2 <> Nil{}>, Shape.clash(accepted))
    case 0n <> 0n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 0n, 0n, 0n]>, Shape.clash(accepted))
    case 0n <> 0n <> 0n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 0n, 0n, 1n]>, Shape.clash(accepted))
    case 0n <> 0n <> 0n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 0n, 0n, 2n]>, Shape.clash(accepted))
    case 0n <> 0n <> 0n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 0n, 0n, 3n]>, Shape.clash(accepted))
    case 0n <> 0n <> 0n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 0n <> 0n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 0n <> 0n <> (4n+p) <> Nil{}, 4n), Axis.unique(0n <> 0n <> 0n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 0n <> 0n <> (4n+p) <> Nil{}), accepted)))))
    case 0n <> 0n <> 1n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 0n, 1n, 0n]>, Shape.clash(accepted))
    case 0n <> 0n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 0n, 1n, 1n]>, Shape.clash(accepted))
    case 0n <> 0n <> 1n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 0n, 1n, 2n]>, Shape.clash(accepted))
    case 0n <> 0n <> 1n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 0n, 1n, 3n]>, Shape.clash(accepted))
    case 0n <> 0n <> 1n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 0n <> 1n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 0n <> 1n <> (4n+p) <> Nil{}, 4n), Axis.unique(0n <> 0n <> 1n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 0n <> 1n <> (4n+p) <> Nil{}), accepted)))))
    case 0n <> 0n <> 2n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 0n, 2n, 0n]>, Shape.clash(accepted))
    case 0n <> 0n <> 2n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 0n, 2n, 1n]>, Shape.clash(accepted))
    case 0n <> 0n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 0n, 2n, 2n]>, Shape.clash(accepted))
    case 0n <> 0n <> 2n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 0n, 2n, 3n]>, Shape.clash(accepted))
    case 0n <> 0n <> 2n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 0n <> 2n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 0n <> 2n <> (4n+p) <> Nil{}, 4n), Axis.unique(0n <> 0n <> 2n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 0n <> 2n <> (4n+p) <> Nil{}), accepted)))))
    case 0n <> 0n <> 3n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 0n, 3n, 0n]>, Shape.clash(accepted))
    case 0n <> 0n <> 3n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 0n, 3n, 1n]>, Shape.clash(accepted))
    case 0n <> 0n <> 3n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 0n, 3n, 2n]>, Shape.clash(accepted))
    case 0n <> 0n <> 3n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 0n, 3n, 3n]>, Shape.clash(accepted))
    case 0n <> 0n <> 3n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 0n <> 3n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 0n <> 3n <> (4n+p) <> Nil{}, 4n), Axis.unique(0n <> 0n <> 3n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 0n <> 3n <> (4n+p) <> Nil{}), accepted)))))
    case 0n <> 0n <> (4n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 0n <> (4n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 0n <> (4n+p) <> a0 <> Nil{}, 4n), Axis.unique(0n <> 0n <> (4n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 0n <> (4n+p) <> a0 <> Nil{}), accepted)))))
    case 0n <> 1n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 1n, 0n, 0n]>, Shape.clash(accepted))
    case 0n <> 1n <> 0n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 1n, 0n, 1n]>, Shape.clash(accepted))
    case 0n <> 1n <> 0n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 1n, 0n, 2n]>, Shape.clash(accepted))
    case 0n <> 1n <> 0n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 1n, 0n, 3n]>, Shape.clash(accepted))
    case 0n <> 1n <> 0n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 1n <> 0n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 1n <> 0n <> (4n+p) <> Nil{}, 4n), Axis.unique(0n <> 1n <> 0n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 1n <> 0n <> (4n+p) <> Nil{}), accepted)))))
    case 0n <> 1n <> 1n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 1n, 1n, 0n]>, Shape.clash(accepted))
    case 0n <> 1n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 1n, 1n, 1n]>, Shape.clash(accepted))
    case 0n <> 1n <> 1n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 1n, 1n, 2n]>, Shape.clash(accepted))
    case 0n <> 1n <> 1n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 1n, 1n, 3n]>, Shape.clash(accepted))
    case 0n <> 1n <> 1n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 1n <> 1n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 1n <> 1n <> (4n+p) <> Nil{}, 4n), Axis.unique(0n <> 1n <> 1n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 1n <> 1n <> (4n+p) <> Nil{}), accepted)))))
    case 0n <> 1n <> 2n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 1n, 2n, 0n]>, Shape.clash(accepted))
    case 0n <> 1n <> 2n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 1n, 2n, 1n]>, Shape.clash(accepted))
    case 0n <> 1n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 1n, 2n, 2n]>, Shape.clash(accepted))
    case 0n <> 1n <> 2n <> 3n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_0123(A, items, length),
        dims => length => product_0123(dims, length), dims => coords => dl => cl => fit_0123(dims, coords, dl, cl)}
    case 0n <> 1n <> 2n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 1n <> 2n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 1n <> 2n <> (4n+p) <> Nil{}, 4n), Axis.unique(0n <> 1n <> 2n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 1n <> 2n <> (4n+p) <> Nil{}), accepted)))))
    case 0n <> 1n <> 3n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 1n, 3n, 0n]>, Shape.clash(accepted))
    case 0n <> 1n <> 3n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 1n, 3n, 1n]>, Shape.clash(accepted))
    case 0n <> 1n <> 3n <> 2n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_0132(A, items, length),
        dims => length => product_0132(dims, length), dims => coords => dl => cl => fit_0132(dims, coords, dl, cl)}
    case 0n <> 1n <> 3n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 1n, 3n, 3n]>, Shape.clash(accepted))
    case 0n <> 1n <> 3n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 1n <> 3n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 1n <> 3n <> (4n+p) <> Nil{}, 4n), Axis.unique(0n <> 1n <> 3n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 1n <> 3n <> (4n+p) <> Nil{}), accepted)))))
    case 0n <> 1n <> (4n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 1n <> (4n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 1n <> (4n+p) <> a0 <> Nil{}, 4n), Axis.unique(0n <> 1n <> (4n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 1n <> (4n+p) <> a0 <> Nil{}), accepted)))))
    case 0n <> 2n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 2n, 0n, 0n]>, Shape.clash(accepted))
    case 0n <> 2n <> 0n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 2n, 0n, 1n]>, Shape.clash(accepted))
    case 0n <> 2n <> 0n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 2n, 0n, 2n]>, Shape.clash(accepted))
    case 0n <> 2n <> 0n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 2n, 0n, 3n]>, Shape.clash(accepted))
    case 0n <> 2n <> 0n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 2n <> 0n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 2n <> 0n <> (4n+p) <> Nil{}, 4n), Axis.unique(0n <> 2n <> 0n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 2n <> 0n <> (4n+p) <> Nil{}), accepted)))))
    case 0n <> 2n <> 1n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 2n, 1n, 0n]>, Shape.clash(accepted))
    case 0n <> 2n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 2n, 1n, 1n]>, Shape.clash(accepted))
    case 0n <> 2n <> 1n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 2n, 1n, 2n]>, Shape.clash(accepted))
    case 0n <> 2n <> 1n <> 3n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_0213(A, items, length),
        dims => length => product_0213(dims, length), dims => coords => dl => cl => fit_0213(dims, coords, dl, cl)}
    case 0n <> 2n <> 1n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 2n <> 1n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 2n <> 1n <> (4n+p) <> Nil{}, 4n), Axis.unique(0n <> 2n <> 1n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 2n <> 1n <> (4n+p) <> Nil{}), accepted)))))
    case 0n <> 2n <> 2n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 2n, 2n, 0n]>, Shape.clash(accepted))
    case 0n <> 2n <> 2n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 2n, 2n, 1n]>, Shape.clash(accepted))
    case 0n <> 2n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 2n, 2n, 2n]>, Shape.clash(accepted))
    case 0n <> 2n <> 2n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 2n, 2n, 3n]>, Shape.clash(accepted))
    case 0n <> 2n <> 2n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 2n <> 2n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 2n <> 2n <> (4n+p) <> Nil{}, 4n), Axis.unique(0n <> 2n <> 2n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 2n <> 2n <> (4n+p) <> Nil{}), accepted)))))
    case 0n <> 2n <> 3n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 2n, 3n, 0n]>, Shape.clash(accepted))
    case 0n <> 2n <> 3n <> 1n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_0231(A, items, length),
        dims => length => product_0231(dims, length), dims => coords => dl => cl => fit_0231(dims, coords, dl, cl)}
    case 0n <> 2n <> 3n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 2n, 3n, 2n]>, Shape.clash(accepted))
    case 0n <> 2n <> 3n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 2n, 3n, 3n]>, Shape.clash(accepted))
    case 0n <> 2n <> 3n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 2n <> 3n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 2n <> 3n <> (4n+p) <> Nil{}, 4n), Axis.unique(0n <> 2n <> 3n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 2n <> 3n <> (4n+p) <> Nil{}), accepted)))))
    case 0n <> 2n <> (4n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 2n <> (4n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 2n <> (4n+p) <> a0 <> Nil{}, 4n), Axis.unique(0n <> 2n <> (4n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 2n <> (4n+p) <> a0 <> Nil{}), accepted)))))
    case 0n <> 3n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 3n, 0n, 0n]>, Shape.clash(accepted))
    case 0n <> 3n <> 0n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 3n, 0n, 1n]>, Shape.clash(accepted))
    case 0n <> 3n <> 0n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 3n, 0n, 2n]>, Shape.clash(accepted))
    case 0n <> 3n <> 0n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 3n, 0n, 3n]>, Shape.clash(accepted))
    case 0n <> 3n <> 0n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 3n <> 0n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 3n <> 0n <> (4n+p) <> Nil{}, 4n), Axis.unique(0n <> 3n <> 0n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 3n <> 0n <> (4n+p) <> Nil{}), accepted)))))
    case 0n <> 3n <> 1n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 3n, 1n, 0n]>, Shape.clash(accepted))
    case 0n <> 3n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 3n, 1n, 1n]>, Shape.clash(accepted))
    case 0n <> 3n <> 1n <> 2n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_0312(A, items, length),
        dims => length => product_0312(dims, length), dims => coords => dl => cl => fit_0312(dims, coords, dl, cl)}
    case 0n <> 3n <> 1n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 3n, 1n, 3n]>, Shape.clash(accepted))
    case 0n <> 3n <> 1n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 3n <> 1n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 3n <> 1n <> (4n+p) <> Nil{}, 4n), Axis.unique(0n <> 3n <> 1n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 3n <> 1n <> (4n+p) <> Nil{}), accepted)))))
    case 0n <> 3n <> 2n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 3n, 2n, 0n]>, Shape.clash(accepted))
    case 0n <> 3n <> 2n <> 1n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_0321(A, items, length),
        dims => length => product_0321(dims, length), dims => coords => dl => cl => fit_0321(dims, coords, dl, cl)}
    case 0n <> 3n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 3n, 2n, 2n]>, Shape.clash(accepted))
    case 0n <> 3n <> 2n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 3n, 2n, 3n]>, Shape.clash(accepted))
    case 0n <> 3n <> 2n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 3n <> 2n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 3n <> 2n <> (4n+p) <> Nil{}, 4n), Axis.unique(0n <> 3n <> 2n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 3n <> 2n <> (4n+p) <> Nil{}), accepted)))))
    case 0n <> 3n <> 3n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 3n, 3n, 0n]>, Shape.clash(accepted))
    case 0n <> 3n <> 3n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 3n, 3n, 1n]>, Shape.clash(accepted))
    case 0n <> 3n <> 3n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 3n, 3n, 2n]>, Shape.clash(accepted))
    case 0n <> 3n <> 3n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [0n, 3n, 3n, 3n]>, Shape.clash(accepted))
    case 0n <> 3n <> 3n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 3n <> 3n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 3n <> 3n <> (4n+p) <> Nil{}, 4n), Axis.unique(0n <> 3n <> 3n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 3n <> 3n <> (4n+p) <> Nil{}), accepted)))))
    case 0n <> 3n <> (4n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> 3n <> (4n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> 3n <> (4n+p) <> a0 <> Nil{}, 4n), Axis.unique(0n <> 3n <> (4n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> 3n <> (4n+p) <> a0 <> Nil{}), accepted)))))
    case 0n <> (4n+p) <> a0 <> a1 <> Nil{}:
      Empty.absurd(Facts<4n, 0n <> (4n+p) <> a0 <> a1 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> a1 <> Nil{}, 4n), Arith.and_left(Axis.bounded(0n <> (4n+p) <> a0 <> a1 <> Nil{}, 4n), Axis.unique(0n <> (4n+p) <> a0 <> a1 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 0n <> (4n+p) <> a0 <> a1 <> Nil{}), accepted)))))
    case 1n <> 0n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 0n, 0n, 0n]>, Shape.clash(accepted))
    case 1n <> 0n <> 0n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 0n, 0n, 1n]>, Shape.clash(accepted))
    case 1n <> 0n <> 0n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 0n, 0n, 2n]>, Shape.clash(accepted))
    case 1n <> 0n <> 0n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 0n, 0n, 3n]>, Shape.clash(accepted))
    case 1n <> 0n <> 0n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 0n <> 0n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 0n <> 0n <> (4n+p) <> Nil{}, 4n), Axis.unique(1n <> 0n <> 0n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 0n <> 0n <> (4n+p) <> Nil{}), accepted)))))
    case 1n <> 0n <> 1n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 0n, 1n, 0n]>, Shape.clash(accepted))
    case 1n <> 0n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 0n, 1n, 1n]>, Shape.clash(accepted))
    case 1n <> 0n <> 1n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 0n, 1n, 2n]>, Shape.clash(accepted))
    case 1n <> 0n <> 1n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 0n, 1n, 3n]>, Shape.clash(accepted))
    case 1n <> 0n <> 1n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 0n <> 1n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 0n <> 1n <> (4n+p) <> Nil{}, 4n), Axis.unique(1n <> 0n <> 1n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 0n <> 1n <> (4n+p) <> Nil{}), accepted)))))
    case 1n <> 0n <> 2n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 0n, 2n, 0n]>, Shape.clash(accepted))
    case 1n <> 0n <> 2n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 0n, 2n, 1n]>, Shape.clash(accepted))
    case 1n <> 0n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 0n, 2n, 2n]>, Shape.clash(accepted))
    case 1n <> 0n <> 2n <> 3n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_1023(A, items, length),
        dims => length => product_1023(dims, length), dims => coords => dl => cl => fit_1023(dims, coords, dl, cl)}
    case 1n <> 0n <> 2n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 0n <> 2n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 0n <> 2n <> (4n+p) <> Nil{}, 4n), Axis.unique(1n <> 0n <> 2n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 0n <> 2n <> (4n+p) <> Nil{}), accepted)))))
    case 1n <> 0n <> 3n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 0n, 3n, 0n]>, Shape.clash(accepted))
    case 1n <> 0n <> 3n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 0n, 3n, 1n]>, Shape.clash(accepted))
    case 1n <> 0n <> 3n <> 2n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_1032(A, items, length),
        dims => length => product_1032(dims, length), dims => coords => dl => cl => fit_1032(dims, coords, dl, cl)}
    case 1n <> 0n <> 3n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 0n, 3n, 3n]>, Shape.clash(accepted))
    case 1n <> 0n <> 3n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 0n <> 3n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 0n <> 3n <> (4n+p) <> Nil{}, 4n), Axis.unique(1n <> 0n <> 3n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 0n <> 3n <> (4n+p) <> Nil{}), accepted)))))
    case 1n <> 0n <> (4n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 0n <> (4n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 0n <> (4n+p) <> a0 <> Nil{}, 4n), Axis.unique(1n <> 0n <> (4n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 0n <> (4n+p) <> a0 <> Nil{}), accepted)))))
    case 1n <> 1n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 1n, 0n, 0n]>, Shape.clash(accepted))
    case 1n <> 1n <> 0n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 1n, 0n, 1n]>, Shape.clash(accepted))
    case 1n <> 1n <> 0n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 1n, 0n, 2n]>, Shape.clash(accepted))
    case 1n <> 1n <> 0n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 1n, 0n, 3n]>, Shape.clash(accepted))
    case 1n <> 1n <> 0n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 1n <> 0n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 1n <> 0n <> (4n+p) <> Nil{}, 4n), Axis.unique(1n <> 1n <> 0n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 1n <> 0n <> (4n+p) <> Nil{}), accepted)))))
    case 1n <> 1n <> 1n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 1n, 1n, 0n]>, Shape.clash(accepted))
    case 1n <> 1n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 1n, 1n, 1n]>, Shape.clash(accepted))
    case 1n <> 1n <> 1n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 1n, 1n, 2n]>, Shape.clash(accepted))
    case 1n <> 1n <> 1n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 1n, 1n, 3n]>, Shape.clash(accepted))
    case 1n <> 1n <> 1n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 1n <> 1n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 1n <> 1n <> (4n+p) <> Nil{}, 4n), Axis.unique(1n <> 1n <> 1n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 1n <> 1n <> (4n+p) <> Nil{}), accepted)))))
    case 1n <> 1n <> 2n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 1n, 2n, 0n]>, Shape.clash(accepted))
    case 1n <> 1n <> 2n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 1n, 2n, 1n]>, Shape.clash(accepted))
    case 1n <> 1n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 1n, 2n, 2n]>, Shape.clash(accepted))
    case 1n <> 1n <> 2n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 1n, 2n, 3n]>, Shape.clash(accepted))
    case 1n <> 1n <> 2n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 1n <> 2n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 1n <> 2n <> (4n+p) <> Nil{}, 4n), Axis.unique(1n <> 1n <> 2n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 1n <> 2n <> (4n+p) <> Nil{}), accepted)))))
    case 1n <> 1n <> 3n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 1n, 3n, 0n]>, Shape.clash(accepted))
    case 1n <> 1n <> 3n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 1n, 3n, 1n]>, Shape.clash(accepted))
    case 1n <> 1n <> 3n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 1n, 3n, 2n]>, Shape.clash(accepted))
    case 1n <> 1n <> 3n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 1n, 3n, 3n]>, Shape.clash(accepted))
    case 1n <> 1n <> 3n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 1n <> 3n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 1n <> 3n <> (4n+p) <> Nil{}, 4n), Axis.unique(1n <> 1n <> 3n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 1n <> 3n <> (4n+p) <> Nil{}), accepted)))))
    case 1n <> 1n <> (4n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 1n <> (4n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 1n <> (4n+p) <> a0 <> Nil{}, 4n), Axis.unique(1n <> 1n <> (4n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 1n <> (4n+p) <> a0 <> Nil{}), accepted)))))
    case 1n <> 2n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 2n, 0n, 0n]>, Shape.clash(accepted))
    case 1n <> 2n <> 0n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 2n, 0n, 1n]>, Shape.clash(accepted))
    case 1n <> 2n <> 0n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 2n, 0n, 2n]>, Shape.clash(accepted))
    case 1n <> 2n <> 0n <> 3n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_1203(A, items, length),
        dims => length => product_1203(dims, length), dims => coords => dl => cl => fit_1203(dims, coords, dl, cl)}
    case 1n <> 2n <> 0n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 2n <> 0n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 2n <> 0n <> (4n+p) <> Nil{}, 4n), Axis.unique(1n <> 2n <> 0n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 2n <> 0n <> (4n+p) <> Nil{}), accepted)))))
    case 1n <> 2n <> 1n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 2n, 1n, 0n]>, Shape.clash(accepted))
    case 1n <> 2n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 2n, 1n, 1n]>, Shape.clash(accepted))
    case 1n <> 2n <> 1n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 2n, 1n, 2n]>, Shape.clash(accepted))
    case 1n <> 2n <> 1n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 2n, 1n, 3n]>, Shape.clash(accepted))
    case 1n <> 2n <> 1n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 2n <> 1n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 2n <> 1n <> (4n+p) <> Nil{}, 4n), Axis.unique(1n <> 2n <> 1n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 2n <> 1n <> (4n+p) <> Nil{}), accepted)))))
    case 1n <> 2n <> 2n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 2n, 2n, 0n]>, Shape.clash(accepted))
    case 1n <> 2n <> 2n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 2n, 2n, 1n]>, Shape.clash(accepted))
    case 1n <> 2n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 2n, 2n, 2n]>, Shape.clash(accepted))
    case 1n <> 2n <> 2n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 2n, 2n, 3n]>, Shape.clash(accepted))
    case 1n <> 2n <> 2n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 2n <> 2n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 2n <> 2n <> (4n+p) <> Nil{}, 4n), Axis.unique(1n <> 2n <> 2n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 2n <> 2n <> (4n+p) <> Nil{}), accepted)))))
    case 1n <> 2n <> 3n <> 0n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_1230(A, items, length),
        dims => length => product_1230(dims, length), dims => coords => dl => cl => fit_1230(dims, coords, dl, cl)}
    case 1n <> 2n <> 3n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 2n, 3n, 1n]>, Shape.clash(accepted))
    case 1n <> 2n <> 3n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 2n, 3n, 2n]>, Shape.clash(accepted))
    case 1n <> 2n <> 3n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 2n, 3n, 3n]>, Shape.clash(accepted))
    case 1n <> 2n <> 3n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 2n <> 3n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 2n <> 3n <> (4n+p) <> Nil{}, 4n), Axis.unique(1n <> 2n <> 3n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 2n <> 3n <> (4n+p) <> Nil{}), accepted)))))
    case 1n <> 2n <> (4n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 2n <> (4n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 2n <> (4n+p) <> a0 <> Nil{}, 4n), Axis.unique(1n <> 2n <> (4n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 2n <> (4n+p) <> a0 <> Nil{}), accepted)))))
    case 1n <> 3n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 3n, 0n, 0n]>, Shape.clash(accepted))
    case 1n <> 3n <> 0n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 3n, 0n, 1n]>, Shape.clash(accepted))
    case 1n <> 3n <> 0n <> 2n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_1302(A, items, length),
        dims => length => product_1302(dims, length), dims => coords => dl => cl => fit_1302(dims, coords, dl, cl)}
    case 1n <> 3n <> 0n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 3n, 0n, 3n]>, Shape.clash(accepted))
    case 1n <> 3n <> 0n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 3n <> 0n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 3n <> 0n <> (4n+p) <> Nil{}, 4n), Axis.unique(1n <> 3n <> 0n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 3n <> 0n <> (4n+p) <> Nil{}), accepted)))))
    case 1n <> 3n <> 1n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 3n, 1n, 0n]>, Shape.clash(accepted))
    case 1n <> 3n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 3n, 1n, 1n]>, Shape.clash(accepted))
    case 1n <> 3n <> 1n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 3n, 1n, 2n]>, Shape.clash(accepted))
    case 1n <> 3n <> 1n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 3n, 1n, 3n]>, Shape.clash(accepted))
    case 1n <> 3n <> 1n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 3n <> 1n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 3n <> 1n <> (4n+p) <> Nil{}, 4n), Axis.unique(1n <> 3n <> 1n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 3n <> 1n <> (4n+p) <> Nil{}), accepted)))))
    case 1n <> 3n <> 2n <> 0n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_1320(A, items, length),
        dims => length => product_1320(dims, length), dims => coords => dl => cl => fit_1320(dims, coords, dl, cl)}
    case 1n <> 3n <> 2n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 3n, 2n, 1n]>, Shape.clash(accepted))
    case 1n <> 3n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 3n, 2n, 2n]>, Shape.clash(accepted))
    case 1n <> 3n <> 2n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 3n, 2n, 3n]>, Shape.clash(accepted))
    case 1n <> 3n <> 2n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 3n <> 2n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 3n <> 2n <> (4n+p) <> Nil{}, 4n), Axis.unique(1n <> 3n <> 2n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 3n <> 2n <> (4n+p) <> Nil{}), accepted)))))
    case 1n <> 3n <> 3n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 3n, 3n, 0n]>, Shape.clash(accepted))
    case 1n <> 3n <> 3n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 3n, 3n, 1n]>, Shape.clash(accepted))
    case 1n <> 3n <> 3n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 3n, 3n, 2n]>, Shape.clash(accepted))
    case 1n <> 3n <> 3n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [1n, 3n, 3n, 3n]>, Shape.clash(accepted))
    case 1n <> 3n <> 3n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 3n <> 3n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 3n <> 3n <> (4n+p) <> Nil{}, 4n), Axis.unique(1n <> 3n <> 3n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 3n <> 3n <> (4n+p) <> Nil{}), accepted)))))
    case 1n <> 3n <> (4n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> 3n <> (4n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> 3n <> (4n+p) <> a0 <> Nil{}, 4n), Axis.unique(1n <> 3n <> (4n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> 3n <> (4n+p) <> a0 <> Nil{}), accepted)))))
    case 1n <> (4n+p) <> a0 <> a1 <> Nil{}:
      Empty.absurd(Facts<4n, 1n <> (4n+p) <> a0 <> a1 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> a1 <> Nil{}, 4n), Arith.and_left(Axis.bounded(1n <> (4n+p) <> a0 <> a1 <> Nil{}, 4n), Axis.unique(1n <> (4n+p) <> a0 <> a1 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 1n <> (4n+p) <> a0 <> a1 <> Nil{}), accepted)))))
    case 2n <> 0n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 0n, 0n, 0n]>, Shape.clash(accepted))
    case 2n <> 0n <> 0n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 0n, 0n, 1n]>, Shape.clash(accepted))
    case 2n <> 0n <> 0n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 0n, 0n, 2n]>, Shape.clash(accepted))
    case 2n <> 0n <> 0n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 0n, 0n, 3n]>, Shape.clash(accepted))
    case 2n <> 0n <> 0n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 0n <> 0n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 0n <> 0n <> (4n+p) <> Nil{}, 4n), Axis.unique(2n <> 0n <> 0n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 0n <> 0n <> (4n+p) <> Nil{}), accepted)))))
    case 2n <> 0n <> 1n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 0n, 1n, 0n]>, Shape.clash(accepted))
    case 2n <> 0n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 0n, 1n, 1n]>, Shape.clash(accepted))
    case 2n <> 0n <> 1n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 0n, 1n, 2n]>, Shape.clash(accepted))
    case 2n <> 0n <> 1n <> 3n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_2013(A, items, length),
        dims => length => product_2013(dims, length), dims => coords => dl => cl => fit_2013(dims, coords, dl, cl)}
    case 2n <> 0n <> 1n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 0n <> 1n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 0n <> 1n <> (4n+p) <> Nil{}, 4n), Axis.unique(2n <> 0n <> 1n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 0n <> 1n <> (4n+p) <> Nil{}), accepted)))))
    case 2n <> 0n <> 2n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 0n, 2n, 0n]>, Shape.clash(accepted))
    case 2n <> 0n <> 2n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 0n, 2n, 1n]>, Shape.clash(accepted))
    case 2n <> 0n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 0n, 2n, 2n]>, Shape.clash(accepted))
    case 2n <> 0n <> 2n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 0n, 2n, 3n]>, Shape.clash(accepted))
    case 2n <> 0n <> 2n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 0n <> 2n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 0n <> 2n <> (4n+p) <> Nil{}, 4n), Axis.unique(2n <> 0n <> 2n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 0n <> 2n <> (4n+p) <> Nil{}), accepted)))))
    case 2n <> 0n <> 3n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 0n, 3n, 0n]>, Shape.clash(accepted))
    case 2n <> 0n <> 3n <> 1n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_2031(A, items, length),
        dims => length => product_2031(dims, length), dims => coords => dl => cl => fit_2031(dims, coords, dl, cl)}
    case 2n <> 0n <> 3n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 0n, 3n, 2n]>, Shape.clash(accepted))
    case 2n <> 0n <> 3n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 0n, 3n, 3n]>, Shape.clash(accepted))
    case 2n <> 0n <> 3n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 0n <> 3n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 0n <> 3n <> (4n+p) <> Nil{}, 4n), Axis.unique(2n <> 0n <> 3n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 0n <> 3n <> (4n+p) <> Nil{}), accepted)))))
    case 2n <> 0n <> (4n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 0n <> (4n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 0n <> (4n+p) <> a0 <> Nil{}, 4n), Axis.unique(2n <> 0n <> (4n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 0n <> (4n+p) <> a0 <> Nil{}), accepted)))))
    case 2n <> 1n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 1n, 0n, 0n]>, Shape.clash(accepted))
    case 2n <> 1n <> 0n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 1n, 0n, 1n]>, Shape.clash(accepted))
    case 2n <> 1n <> 0n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 1n, 0n, 2n]>, Shape.clash(accepted))
    case 2n <> 1n <> 0n <> 3n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_2103(A, items, length),
        dims => length => product_2103(dims, length), dims => coords => dl => cl => fit_2103(dims, coords, dl, cl)}
    case 2n <> 1n <> 0n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 1n <> 0n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 1n <> 0n <> (4n+p) <> Nil{}, 4n), Axis.unique(2n <> 1n <> 0n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 1n <> 0n <> (4n+p) <> Nil{}), accepted)))))
    case 2n <> 1n <> 1n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 1n, 1n, 0n]>, Shape.clash(accepted))
    case 2n <> 1n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 1n, 1n, 1n]>, Shape.clash(accepted))
    case 2n <> 1n <> 1n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 1n, 1n, 2n]>, Shape.clash(accepted))
    case 2n <> 1n <> 1n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 1n, 1n, 3n]>, Shape.clash(accepted))
    case 2n <> 1n <> 1n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 1n <> 1n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 1n <> 1n <> (4n+p) <> Nil{}, 4n), Axis.unique(2n <> 1n <> 1n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 1n <> 1n <> (4n+p) <> Nil{}), accepted)))))
    case 2n <> 1n <> 2n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 1n, 2n, 0n]>, Shape.clash(accepted))
    case 2n <> 1n <> 2n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 1n, 2n, 1n]>, Shape.clash(accepted))
    case 2n <> 1n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 1n, 2n, 2n]>, Shape.clash(accepted))
    case 2n <> 1n <> 2n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 1n, 2n, 3n]>, Shape.clash(accepted))
    case 2n <> 1n <> 2n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 1n <> 2n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 1n <> 2n <> (4n+p) <> Nil{}, 4n), Axis.unique(2n <> 1n <> 2n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 1n <> 2n <> (4n+p) <> Nil{}), accepted)))))
    case 2n <> 1n <> 3n <> 0n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_2130(A, items, length),
        dims => length => product_2130(dims, length), dims => coords => dl => cl => fit_2130(dims, coords, dl, cl)}
    case 2n <> 1n <> 3n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 1n, 3n, 1n]>, Shape.clash(accepted))
    case 2n <> 1n <> 3n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 1n, 3n, 2n]>, Shape.clash(accepted))
    case 2n <> 1n <> 3n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 1n, 3n, 3n]>, Shape.clash(accepted))
    case 2n <> 1n <> 3n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 1n <> 3n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 1n <> 3n <> (4n+p) <> Nil{}, 4n), Axis.unique(2n <> 1n <> 3n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 1n <> 3n <> (4n+p) <> Nil{}), accepted)))))
    case 2n <> 1n <> (4n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 1n <> (4n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 1n <> (4n+p) <> a0 <> Nil{}, 4n), Axis.unique(2n <> 1n <> (4n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 1n <> (4n+p) <> a0 <> Nil{}), accepted)))))
    case 2n <> 2n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 2n, 0n, 0n]>, Shape.clash(accepted))
    case 2n <> 2n <> 0n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 2n, 0n, 1n]>, Shape.clash(accepted))
    case 2n <> 2n <> 0n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 2n, 0n, 2n]>, Shape.clash(accepted))
    case 2n <> 2n <> 0n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 2n, 0n, 3n]>, Shape.clash(accepted))
    case 2n <> 2n <> 0n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 2n <> 0n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 2n <> 0n <> (4n+p) <> Nil{}, 4n), Axis.unique(2n <> 2n <> 0n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 2n <> 0n <> (4n+p) <> Nil{}), accepted)))))
    case 2n <> 2n <> 1n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 2n, 1n, 0n]>, Shape.clash(accepted))
    case 2n <> 2n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 2n, 1n, 1n]>, Shape.clash(accepted))
    case 2n <> 2n <> 1n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 2n, 1n, 2n]>, Shape.clash(accepted))
    case 2n <> 2n <> 1n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 2n, 1n, 3n]>, Shape.clash(accepted))
    case 2n <> 2n <> 1n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 2n <> 1n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 2n <> 1n <> (4n+p) <> Nil{}, 4n), Axis.unique(2n <> 2n <> 1n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 2n <> 1n <> (4n+p) <> Nil{}), accepted)))))
    case 2n <> 2n <> 2n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 2n, 2n, 0n]>, Shape.clash(accepted))
    case 2n <> 2n <> 2n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 2n, 2n, 1n]>, Shape.clash(accepted))
    case 2n <> 2n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 2n, 2n, 2n]>, Shape.clash(accepted))
    case 2n <> 2n <> 2n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 2n, 2n, 3n]>, Shape.clash(accepted))
    case 2n <> 2n <> 2n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 2n <> 2n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 2n <> 2n <> (4n+p) <> Nil{}, 4n), Axis.unique(2n <> 2n <> 2n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 2n <> 2n <> (4n+p) <> Nil{}), accepted)))))
    case 2n <> 2n <> 3n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 2n, 3n, 0n]>, Shape.clash(accepted))
    case 2n <> 2n <> 3n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 2n, 3n, 1n]>, Shape.clash(accepted))
    case 2n <> 2n <> 3n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 2n, 3n, 2n]>, Shape.clash(accepted))
    case 2n <> 2n <> 3n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 2n, 3n, 3n]>, Shape.clash(accepted))
    case 2n <> 2n <> 3n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 2n <> 3n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 2n <> 3n <> (4n+p) <> Nil{}, 4n), Axis.unique(2n <> 2n <> 3n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 2n <> 3n <> (4n+p) <> Nil{}), accepted)))))
    case 2n <> 2n <> (4n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 2n <> (4n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 2n <> (4n+p) <> a0 <> Nil{}, 4n), Axis.unique(2n <> 2n <> (4n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 2n <> (4n+p) <> a0 <> Nil{}), accepted)))))
    case 2n <> 3n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 3n, 0n, 0n]>, Shape.clash(accepted))
    case 2n <> 3n <> 0n <> 1n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_2301(A, items, length),
        dims => length => product_2301(dims, length), dims => coords => dl => cl => fit_2301(dims, coords, dl, cl)}
    case 2n <> 3n <> 0n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 3n, 0n, 2n]>, Shape.clash(accepted))
    case 2n <> 3n <> 0n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 3n, 0n, 3n]>, Shape.clash(accepted))
    case 2n <> 3n <> 0n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 3n <> 0n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 3n <> 0n <> (4n+p) <> Nil{}, 4n), Axis.unique(2n <> 3n <> 0n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 3n <> 0n <> (4n+p) <> Nil{}), accepted)))))
    case 2n <> 3n <> 1n <> 0n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_2310(A, items, length),
        dims => length => product_2310(dims, length), dims => coords => dl => cl => fit_2310(dims, coords, dl, cl)}
    case 2n <> 3n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 3n, 1n, 1n]>, Shape.clash(accepted))
    case 2n <> 3n <> 1n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 3n, 1n, 2n]>, Shape.clash(accepted))
    case 2n <> 3n <> 1n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 3n, 1n, 3n]>, Shape.clash(accepted))
    case 2n <> 3n <> 1n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 3n <> 1n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 3n <> 1n <> (4n+p) <> Nil{}, 4n), Axis.unique(2n <> 3n <> 1n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 3n <> 1n <> (4n+p) <> Nil{}), accepted)))))
    case 2n <> 3n <> 2n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 3n, 2n, 0n]>, Shape.clash(accepted))
    case 2n <> 3n <> 2n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 3n, 2n, 1n]>, Shape.clash(accepted))
    case 2n <> 3n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 3n, 2n, 2n]>, Shape.clash(accepted))
    case 2n <> 3n <> 2n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 3n, 2n, 3n]>, Shape.clash(accepted))
    case 2n <> 3n <> 2n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 3n <> 2n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 3n <> 2n <> (4n+p) <> Nil{}, 4n), Axis.unique(2n <> 3n <> 2n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 3n <> 2n <> (4n+p) <> Nil{}), accepted)))))
    case 2n <> 3n <> 3n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 3n, 3n, 0n]>, Shape.clash(accepted))
    case 2n <> 3n <> 3n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 3n, 3n, 1n]>, Shape.clash(accepted))
    case 2n <> 3n <> 3n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 3n, 3n, 2n]>, Shape.clash(accepted))
    case 2n <> 3n <> 3n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [2n, 3n, 3n, 3n]>, Shape.clash(accepted))
    case 2n <> 3n <> 3n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 3n <> 3n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 3n <> 3n <> (4n+p) <> Nil{}, 4n), Axis.unique(2n <> 3n <> 3n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 3n <> 3n <> (4n+p) <> Nil{}), accepted)))))
    case 2n <> 3n <> (4n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> 3n <> (4n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> 3n <> (4n+p) <> a0 <> Nil{}, 4n), Axis.unique(2n <> 3n <> (4n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> 3n <> (4n+p) <> a0 <> Nil{}), accepted)))))
    case 2n <> (4n+p) <> a0 <> a1 <> Nil{}:
      Empty.absurd(Facts<4n, 2n <> (4n+p) <> a0 <> a1 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> a1 <> Nil{}, 4n), Arith.and_left(Axis.bounded(2n <> (4n+p) <> a0 <> a1 <> Nil{}, 4n), Axis.unique(2n <> (4n+p) <> a0 <> a1 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 2n <> (4n+p) <> a0 <> a1 <> Nil{}), accepted)))))
    case 3n <> 0n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 0n, 0n, 0n]>, Shape.clash(accepted))
    case 3n <> 0n <> 0n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 0n, 0n, 1n]>, Shape.clash(accepted))
    case 3n <> 0n <> 0n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 0n, 0n, 2n]>, Shape.clash(accepted))
    case 3n <> 0n <> 0n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 0n, 0n, 3n]>, Shape.clash(accepted))
    case 3n <> 0n <> 0n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 0n <> 0n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 0n <> 0n <> (4n+p) <> Nil{}, 4n), Axis.unique(3n <> 0n <> 0n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 0n <> 0n <> (4n+p) <> Nil{}), accepted)))))
    case 3n <> 0n <> 1n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 0n, 1n, 0n]>, Shape.clash(accepted))
    case 3n <> 0n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 0n, 1n, 1n]>, Shape.clash(accepted))
    case 3n <> 0n <> 1n <> 2n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_3012(A, items, length),
        dims => length => product_3012(dims, length), dims => coords => dl => cl => fit_3012(dims, coords, dl, cl)}
    case 3n <> 0n <> 1n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 0n, 1n, 3n]>, Shape.clash(accepted))
    case 3n <> 0n <> 1n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 0n <> 1n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 0n <> 1n <> (4n+p) <> Nil{}, 4n), Axis.unique(3n <> 0n <> 1n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 0n <> 1n <> (4n+p) <> Nil{}), accepted)))))
    case 3n <> 0n <> 2n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 0n, 2n, 0n]>, Shape.clash(accepted))
    case 3n <> 0n <> 2n <> 1n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_3021(A, items, length),
        dims => length => product_3021(dims, length), dims => coords => dl => cl => fit_3021(dims, coords, dl, cl)}
    case 3n <> 0n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 0n, 2n, 2n]>, Shape.clash(accepted))
    case 3n <> 0n <> 2n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 0n, 2n, 3n]>, Shape.clash(accepted))
    case 3n <> 0n <> 2n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 0n <> 2n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 0n <> 2n <> (4n+p) <> Nil{}, 4n), Axis.unique(3n <> 0n <> 2n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 0n <> 2n <> (4n+p) <> Nil{}), accepted)))))
    case 3n <> 0n <> 3n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 0n, 3n, 0n]>, Shape.clash(accepted))
    case 3n <> 0n <> 3n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 0n, 3n, 1n]>, Shape.clash(accepted))
    case 3n <> 0n <> 3n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 0n, 3n, 2n]>, Shape.clash(accepted))
    case 3n <> 0n <> 3n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 0n, 3n, 3n]>, Shape.clash(accepted))
    case 3n <> 0n <> 3n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 0n <> 3n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 0n <> 3n <> (4n+p) <> Nil{}, 4n), Axis.unique(3n <> 0n <> 3n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 0n <> 3n <> (4n+p) <> Nil{}), accepted)))))
    case 3n <> 0n <> (4n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 0n <> (4n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 0n <> (4n+p) <> a0 <> Nil{}, 4n), Axis.unique(3n <> 0n <> (4n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 0n <> (4n+p) <> a0 <> Nil{}), accepted)))))
    case 3n <> 1n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 1n, 0n, 0n]>, Shape.clash(accepted))
    case 3n <> 1n <> 0n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 1n, 0n, 1n]>, Shape.clash(accepted))
    case 3n <> 1n <> 0n <> 2n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_3102(A, items, length),
        dims => length => product_3102(dims, length), dims => coords => dl => cl => fit_3102(dims, coords, dl, cl)}
    case 3n <> 1n <> 0n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 1n, 0n, 3n]>, Shape.clash(accepted))
    case 3n <> 1n <> 0n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 1n <> 0n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 1n <> 0n <> (4n+p) <> Nil{}, 4n), Axis.unique(3n <> 1n <> 0n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 1n <> 0n <> (4n+p) <> Nil{}), accepted)))))
    case 3n <> 1n <> 1n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 1n, 1n, 0n]>, Shape.clash(accepted))
    case 3n <> 1n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 1n, 1n, 1n]>, Shape.clash(accepted))
    case 3n <> 1n <> 1n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 1n, 1n, 2n]>, Shape.clash(accepted))
    case 3n <> 1n <> 1n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 1n, 1n, 3n]>, Shape.clash(accepted))
    case 3n <> 1n <> 1n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 1n <> 1n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 1n <> 1n <> (4n+p) <> Nil{}, 4n), Axis.unique(3n <> 1n <> 1n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 1n <> 1n <> (4n+p) <> Nil{}), accepted)))))
    case 3n <> 1n <> 2n <> 0n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_3120(A, items, length),
        dims => length => product_3120(dims, length), dims => coords => dl => cl => fit_3120(dims, coords, dl, cl)}
    case 3n <> 1n <> 2n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 1n, 2n, 1n]>, Shape.clash(accepted))
    case 3n <> 1n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 1n, 2n, 2n]>, Shape.clash(accepted))
    case 3n <> 1n <> 2n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 1n, 2n, 3n]>, Shape.clash(accepted))
    case 3n <> 1n <> 2n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 1n <> 2n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 1n <> 2n <> (4n+p) <> Nil{}, 4n), Axis.unique(3n <> 1n <> 2n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 1n <> 2n <> (4n+p) <> Nil{}), accepted)))))
    case 3n <> 1n <> 3n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 1n, 3n, 0n]>, Shape.clash(accepted))
    case 3n <> 1n <> 3n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 1n, 3n, 1n]>, Shape.clash(accepted))
    case 3n <> 1n <> 3n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 1n, 3n, 2n]>, Shape.clash(accepted))
    case 3n <> 1n <> 3n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 1n, 3n, 3n]>, Shape.clash(accepted))
    case 3n <> 1n <> 3n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 1n <> 3n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 1n <> 3n <> (4n+p) <> Nil{}, 4n), Axis.unique(3n <> 1n <> 3n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 1n <> 3n <> (4n+p) <> Nil{}), accepted)))))
    case 3n <> 1n <> (4n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 1n <> (4n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 1n <> (4n+p) <> a0 <> Nil{}, 4n), Axis.unique(3n <> 1n <> (4n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 1n <> (4n+p) <> a0 <> Nil{}), accepted)))))
    case 3n <> 2n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 2n, 0n, 0n]>, Shape.clash(accepted))
    case 3n <> 2n <> 0n <> 1n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_3201(A, items, length),
        dims => length => product_3201(dims, length), dims => coords => dl => cl => fit_3201(dims, coords, dl, cl)}
    case 3n <> 2n <> 0n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 2n, 0n, 2n]>, Shape.clash(accepted))
    case 3n <> 2n <> 0n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 2n, 0n, 3n]>, Shape.clash(accepted))
    case 3n <> 2n <> 0n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 2n <> 0n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 2n <> 0n <> (4n+p) <> Nil{}, 4n), Axis.unique(3n <> 2n <> 0n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 2n <> 0n <> (4n+p) <> Nil{}), accepted)))))
    case 3n <> 2n <> 1n <> 0n <> Nil{}:
      Certified{{==}, {==}, {==}, A => items => length => lists_3210(A, items, length),
        dims => length => product_3210(dims, length), dims => coords => dl => cl => fit_3210(dims, coords, dl, cl)}
    case 3n <> 2n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 2n, 1n, 1n]>, Shape.clash(accepted))
    case 3n <> 2n <> 1n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 2n, 1n, 2n]>, Shape.clash(accepted))
    case 3n <> 2n <> 1n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 2n, 1n, 3n]>, Shape.clash(accepted))
    case 3n <> 2n <> 1n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 2n <> 1n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 2n <> 1n <> (4n+p) <> Nil{}, 4n), Axis.unique(3n <> 2n <> 1n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 2n <> 1n <> (4n+p) <> Nil{}), accepted)))))
    case 3n <> 2n <> 2n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 2n, 2n, 0n]>, Shape.clash(accepted))
    case 3n <> 2n <> 2n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 2n, 2n, 1n]>, Shape.clash(accepted))
    case 3n <> 2n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 2n, 2n, 2n]>, Shape.clash(accepted))
    case 3n <> 2n <> 2n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 2n, 2n, 3n]>, Shape.clash(accepted))
    case 3n <> 2n <> 2n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 2n <> 2n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 2n <> 2n <> (4n+p) <> Nil{}, 4n), Axis.unique(3n <> 2n <> 2n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 2n <> 2n <> (4n+p) <> Nil{}), accepted)))))
    case 3n <> 2n <> 3n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 2n, 3n, 0n]>, Shape.clash(accepted))
    case 3n <> 2n <> 3n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 2n, 3n, 1n]>, Shape.clash(accepted))
    case 3n <> 2n <> 3n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 2n, 3n, 2n]>, Shape.clash(accepted))
    case 3n <> 2n <> 3n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 2n, 3n, 3n]>, Shape.clash(accepted))
    case 3n <> 2n <> 3n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 2n <> 3n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 2n <> 3n <> (4n+p) <> Nil{}, 4n), Axis.unique(3n <> 2n <> 3n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 2n <> 3n <> (4n+p) <> Nil{}), accepted)))))
    case 3n <> 2n <> (4n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 2n <> (4n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 2n <> (4n+p) <> a0 <> Nil{}, 4n), Axis.unique(3n <> 2n <> (4n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 2n <> (4n+p) <> a0 <> Nil{}), accepted)))))
    case 3n <> 3n <> 0n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 3n, 0n, 0n]>, Shape.clash(accepted))
    case 3n <> 3n <> 0n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 3n, 0n, 1n]>, Shape.clash(accepted))
    case 3n <> 3n <> 0n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 3n, 0n, 2n]>, Shape.clash(accepted))
    case 3n <> 3n <> 0n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 3n, 0n, 3n]>, Shape.clash(accepted))
    case 3n <> 3n <> 0n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 3n <> 0n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 3n <> 0n <> (4n+p) <> Nil{}, 4n), Axis.unique(3n <> 3n <> 0n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 3n <> 0n <> (4n+p) <> Nil{}), accepted)))))
    case 3n <> 3n <> 1n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 3n, 1n, 0n]>, Shape.clash(accepted))
    case 3n <> 3n <> 1n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 3n, 1n, 1n]>, Shape.clash(accepted))
    case 3n <> 3n <> 1n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 3n, 1n, 2n]>, Shape.clash(accepted))
    case 3n <> 3n <> 1n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 3n, 1n, 3n]>, Shape.clash(accepted))
    case 3n <> 3n <> 1n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 3n <> 1n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 3n <> 1n <> (4n+p) <> Nil{}, 4n), Axis.unique(3n <> 3n <> 1n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 3n <> 1n <> (4n+p) <> Nil{}), accepted)))))
    case 3n <> 3n <> 2n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 3n, 2n, 0n]>, Shape.clash(accepted))
    case 3n <> 3n <> 2n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 3n, 2n, 1n]>, Shape.clash(accepted))
    case 3n <> 3n <> 2n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 3n, 2n, 2n]>, Shape.clash(accepted))
    case 3n <> 3n <> 2n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 3n, 2n, 3n]>, Shape.clash(accepted))
    case 3n <> 3n <> 2n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 3n <> 2n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 3n <> 2n <> (4n+p) <> Nil{}, 4n), Axis.unique(3n <> 3n <> 2n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 3n <> 2n <> (4n+p) <> Nil{}), accepted)))))
    case 3n <> 3n <> 3n <> 0n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 3n, 3n, 0n]>, Shape.clash(accepted))
    case 3n <> 3n <> 3n <> 1n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 3n, 3n, 1n]>, Shape.clash(accepted))
    case 3n <> 3n <> 3n <> 2n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 3n, 3n, 2n]>, Shape.clash(accepted))
    case 3n <> 3n <> 3n <> 3n <> Nil{}:
      Empty.absurd(Facts<4n, [3n, 3n, 3n, 3n]>, Shape.clash(accepted))
    case 3n <> 3n <> 3n <> (4n+p) <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 3n <> 3n <> (4n+p) <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 3n <> 3n <> (4n+p) <> Nil{}, 4n), Axis.unique(3n <> 3n <> 3n <> (4n+p) <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 3n <> 3n <> (4n+p) <> Nil{}), accepted)))))
    case 3n <> 3n <> (4n+p) <> a0 <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> 3n <> (4n+p) <> a0 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> 3n <> (4n+p) <> a0 <> Nil{}, 4n), Axis.unique(3n <> 3n <> (4n+p) <> a0 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> 3n <> (4n+p) <> a0 <> Nil{}), accepted)))))
    case 3n <> (4n+p) <> a0 <> a1 <> Nil{}:
      Empty.absurd(Facts<4n, 3n <> (4n+p) <> a0 <> a1 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> a1 <> Nil{}, 4n), Arith.and_left(Axis.bounded(3n <> (4n+p) <> a0 <> a1 <> Nil{}, 4n), Axis.unique(3n <> (4n+p) <> a0 <> a1 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, 3n <> (4n+p) <> a0 <> a1 <> Nil{}), accepted)))))
    case (4n+p) <> a0 <> a1 <> a2 <> Nil{}:
      Empty.absurd(Facts<4n, (4n+p) <> a0 <> a1 <> a2 <> Nil{}>, Arith.lt_zero(p, Arith.and_left(Nat.is_lt(4n+p, 4n), Axis.bounded(a0 <> a1 <> a2 <> Nil{}, 4n), Arith.and_left(Axis.bounded((4n+p) <> a0 <> a1 <> a2 <> Nil{}, 4n), Axis.unique((4n+p) <> a0 <> a1 <> a2 <> Nil{}), Equal.sym(Bool, True{}, Axis.valid(4n, (4n+p) <> a0 <> a1 <> a2 <> Nil{}), accepted)))))
    case +a0 <> +a1 <> +a2 <> +a3 <> +a4 <> +tail:
      Empty.absurd(Facts<4n, a0 <> a1 <> a2 <> a3 <> a4 <> tail>, Shape.clash(accepted))

def facts(+rank: Nat, +axes: List<&2, Nat>, bound: Shape.LE(rank, 4n), accepted: {True{} == Axis.valid(rank, axes) : Bool}) -> Facts<rank, axes>:
  match rank:
    case 0n:
      facts_0(axes, accepted)
    case 1n+rankp:
      match rankp:
        case 0n:
          facts_1(axes, accepted)
        case 1n+rankpp:
          match rankpp:
            case 0n:
              facts_2(axes, accepted)
            case 1n+rankppp:
              match rankppp:
                case 0n:
                  facts_3(axes, accepted)
                case 1n+rankpppp:
                  match rankpppp:
                    case 0n:
                      facts_4(axes, accepted)
                    case 1n+rankppppp:
                      match bound:

def facts_lists(-A: Data, -rank: Nat, -axes: List<&2, Nat>, +items: List<&2, A>, length: {List.length(&2, A, items) == rank : Nat}, certificate: Facts<rank, axes>) -> ListFacts<A, rank, axes, items>:
  Certified{iv, iselected, irestored, lists, product, fit} = certificate
  lists(A, items, length)

def facts_product(-rank: Nat, -axes: List<&2, Nat>, +dims: List<&2, Nat>, length: {List.length(&2, Nat, dims) == rank : Nat}, certificate: Facts<rank, axes>) ->
  {Shape.numel(Axis.order(Nat, dims, axes)) == Shape.numel(dims) : Nat}:
  Certified{iv, iselected, irestored, lists, product, fit} = certificate
  product(dims, length)

def facts_inverse_valid(-rank: Nat, -axes: List<&2, Nat>, certificate: Facts<rank, axes>) -> {True{} == Axis.valid(rank, Axis.inverse_axes(rank, axes)) : Bool}:
  Certified{iv, iselected, irestored, lists, product, fit} = certificate
  iv

def selected(-A: Data, -rank: Nat, -axes: List<&2, Nat>, -items: List<&2, A>, certificate: ListFacts<A, rank, axes, items>) ->
  {Axis.gather(A, items, axes) == Some{Axis.order(A, items, axes)} : Maybe<&2, List<&2, A>>}:
  match certificate:
    case Lists{selected, ranked, scattered, restored, reverse_restored, scatter_restored}:
      selected

def ranked(-A: Data, -rank: Nat, -axes: List<&2, Nat>, -items: List<&2, A>, certificate: ListFacts<A, rank, axes, items>) ->
  {List.length(&2, A, Axis.order(A, items, axes)) == rank : Nat}:
  match certificate:
    case Lists{selected, ranked, scattered, restored, reverse_restored, scatter_restored}:
      ranked

def scattered(-A: Data, -rank: Nat, -axes: List<&2, Nat>, -items: List<&2, A>, certificate: ListFacts<A, rank, axes, items>) ->
  {Axis.scatter(A, rank, axes, items, 0n) == Some{Axis.order(A, items, Axis.inverse_axes(rank, axes))} : Maybe<&2, List<&2, A>>}:
  match certificate:
    case Lists{selected, ranked, scattered, restored, reverse_restored, scatter_restored}:
      scattered

def restored(-A: Data, -rank: Nat, -axes: List<&2, Nat>, -items: List<&2, A>, certificate: ListFacts<A, rank, axes, items>) ->
  {Axis.order(A, Axis.order(A, items, axes), Axis.inverse_axes(rank, axes)) == items : List<&2, A>}:
  match certificate:
    case Lists{selected, ranked, scattered, restored, reverse_restored, scatter_restored}:
      restored

def scatter_restored(-A: Data, -rank: Nat, -axes: List<&2, Nat>, -items: List<&2, A>, certificate: ListFacts<A, rank, axes, items>) ->
  {Axis.scatter(A, rank, axes, Axis.order(A, items, axes), 0n) == Some{items} : Maybe<&2, List<&2, A>>}:
  match certificate:
    case Lists{selected, ranked, scattered, restored, reverse_restored, scatter_restored}:
      scatter_restored

def shape_facts(+shape: Shape.Shape, +axes: List<&2, U32>, accepted: {True{} == Layout.bijective(shape, axes) : Bool}) ->
  Facts<Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes)>:
  match shape:
    case Shape.SShape{+dims, +size, +exact, +rank_bound, +capacity_bound}:
      facts(Shape.rank_dimensions(dims), Shape.naturals(axes), rank_bound, valid(Shape.SShape{dims, size, exact, rank_bound, capacity_bound}, axes, accepted))

def word_length(+items: List<&2, U32>) -> {List.length(&2, U32, items) == Shape.rank_dimensions(items) : Nat}:
  match items:
    case Nil{}:
      {==}
    case head <> +tail:
      Equal.cong(Nat, Nat, n => 1n+n, List.length(&2, U32, tail), Shape.rank_dimensions(tail), word_length(tail))

def permutation_complete(+shape: Shape.Shape, +axes: List<&2, U32>, accepted: {True{} == Layout.bijective(shape, axes) : Bool}) ->
  {Layout.reorder(Shape.dimensions(shape), axes) == Some{Axis.order(U32, Shape.dimensions(shape), Shape.naturals(axes))} : Maybe<&2, List<&2, U32>>}:
  certificate = facts_lists(U32, Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes), Shape.dimensions(shape), word_length(Shape.dimensions(shape)), shape_facts(shape, axes, accepted))
  Equal.trans(Maybe<&2, List<&2, U32>>, Layout.reorder(Shape.dimensions(shape), axes), Axis.gather(U32, Shape.dimensions(shape), Shape.naturals(axes)),
    Some{Axis.order(U32, Shape.dimensions(shape), Shape.naturals(axes))}, reorder(axes, Shape.dimensions(shape)), selected(U32, Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes), Shape.dimensions(shape), certificate))

def permutation_rank(+shape: Shape.Shape, +axes: List<&2, U32>, accepted: {True{} == Layout.bijective(shape, axes) : Bool}) ->
  {Shape.rank_dimensions(Axis.order(U32, Shape.dimensions(shape), Shape.naturals(axes))) == Shape.rank_dimensions(Shape.dimensions(shape)) : Nat}:
  certificate = facts_lists(U32, Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes), Shape.dimensions(shape), word_length(Shape.dimensions(shape)), shape_facts(shape, axes, accepted))
  Equal.trans(Nat, Shape.rank_dimensions(Axis.order(U32, Shape.dimensions(shape), Shape.naturals(axes))), List.length(&2, U32, Axis.order(U32, Shape.dimensions(shape), Shape.naturals(axes))),
    Shape.rank_dimensions(Shape.dimensions(shape)), Equal.sym(Nat, List.length(&2, U32, Axis.order(U32, Shape.dimensions(shape), Shape.naturals(axes))),
      Shape.rank_dimensions(Axis.order(U32, Shape.dimensions(shape), Shape.naturals(axes))), word_length(Axis.order(U32, Shape.dimensions(shape), Shape.naturals(axes)))), ranked(U32, Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes), Shape.dimensions(shape), certificate))

def permutation_product(+shape: Shape.Shape, +axes: List<&2, U32>, accepted: {True{} == Layout.bijective(shape, axes) : Bool}) ->
  {Shape.logical_numel(Axis.order(U32, Shape.dimensions(shape), Shape.naturals(axes))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}:
  Equal.trans(Nat, Shape.logical_numel(Axis.order(U32, Shape.dimensions(shape), Shape.naturals(axes))),
    Shape.numel(Axis.order(Nat, Shape.naturals(Shape.dimensions(shape)), Shape.naturals(axes))), Shape.logical_numel(Shape.dimensions(shape)),
    Equal.cong(List<&2, Nat>, Nat, dims => Shape.numel(dims), Shape.naturals(Axis.order(U32, Shape.dimensions(shape), Shape.naturals(axes))),
      Axis.order(Nat, Shape.naturals(Shape.dimensions(shape)), Shape.naturals(axes)), order_naturals(Shape.naturals(axes), Shape.dimensions(shape))),
    facts_product(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes), Shape.naturals(Shape.dimensions(shape)), Equal.sym(Nat, Shape.rank_dimensions(Shape.dimensions(shape)), List.length(&2, Nat, Shape.naturals(Shape.dimensions(shape))), Shape.rank_naturals(Shape.dimensions(shape))), shape_facts(shape, axes, accepted)))

def scatter_complete(+shape: Shape.Shape, +axes: List<&2, U32>, +coords: List<&2, U32>,
  accepted: {True{} == Layout.bijective(shape, axes) : Bool}, length: {Shape.rank_dimensions(coords) == Shape.rank_dimensions(Shape.dimensions(shape)) : Nat}) ->
  {Layout.inverse_coordinates(Shape.dimensions(shape), axes, coords, 0) ==
    Some{Axis.order(U32, coords, Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes)))} : Maybe<&2, List<&2, U32>>}:
  certificate = facts_lists(U32, Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes), coords,
    Equal.trans(Nat, List.length(&2, U32, coords), Shape.rank_dimensions(coords), Shape.rank_dimensions(Shape.dimensions(shape)), word_length(coords), length), shape_facts(shape, axes, accepted))
  Equal.trans(Maybe<&2, List<&2, U32>>, Layout.inverse_coordinates(Shape.dimensions(shape), axes, coords, 0),
    Axis.scatter(U32, Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes), coords, 0n),
    Some{Axis.order(U32, coords, Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes)))}, inverse_coordinates(shape, axes, coords), scattered(U32, Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes), coords, certificate))

def scatter_gather(+shape: Shape.Shape, +axes: List<&2, U32>, +coords: List<&2, U32>,
  accepted: {True{} == Layout.bijective(shape, axes) : Bool}, length: {Shape.rank_dimensions(coords) == Shape.rank_dimensions(Shape.dimensions(shape)) : Nat}) ->
  {Layout.inverse_coordinates(Shape.dimensions(shape), axes, Axis.order(U32, coords, Shape.naturals(axes)), 0) == Some{coords} : Maybe<&2, List<&2, U32>>}:
  certificate = facts_lists(U32, Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes), coords,
    Equal.trans(Nat, List.length(&2, U32, coords), Shape.rank_dimensions(coords), Shape.rank_dimensions(Shape.dimensions(shape)), word_length(coords), length), shape_facts(shape, axes, accepted))
  Equal.trans(Maybe<&2, List<&2, U32>>, Layout.inverse_coordinates(Shape.dimensions(shape), axes, Axis.order(U32, coords, Shape.naturals(axes)), 0),
    Axis.scatter(U32, Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes), Axis.order(U32, coords, Shape.naturals(axes)), 0n), Some{coords},
    inverse_coordinates(shape, axes, Axis.order(U32, coords, Shape.naturals(axes))), scatter_restored(U32, Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(axes), coords, certificate))

def inverse_valid(+rank: Nat, +axes: List<&2, Nat>, bound: Shape.LE(rank, 4n), accepted: {True{} == Axis.valid(rank, axes) : Bool}) ->
  {True{} == Axis.valid(rank, Axis.inverse_axes(rank, axes)) : Bool}:
  facts_inverse_valid(rank, axes, facts(rank, axes, bound, accepted))

def inverse_composition(-A: Data, +rank: Nat, +axes: List<&2, Nat>, +items: List<&2, A>, bound: Shape.LE(rank, 4n),
  accepted: {True{} == Axis.valid(rank, axes) : Bool}, length: {List.length(&2, A, items) == rank : Nat}) ->
  {Axis.order(A, Axis.order(A, items, axes), Axis.inverse_axes(rank, axes)) == items : List<&2, A>}:
  restored(A, rank, axes, items, facts_lists(A, rank, axes, items, length, facts(rank, axes, bound, accepted)))

def inverse_scatter_composition(-A: Data, +rank: Nat, +axes: List<&2, Nat>, +items: List<&2, A>, +bound: Shape.LE(rank, 4n),
  +accepted: {True{} == Axis.valid(rank, axes) : Bool}, length: {List.length(&2, A, items) == rank : Nat}) ->
  {Axis.scatter(A, rank, Axis.inverse_axes(rank, axes), Axis.order(A, items, Axis.inverse_axes(rank, axes)), 0n) == Some{items} : Maybe<&2, List<&2, A>>}:
  scatter_restored(A, rank, Axis.inverse_axes(rank, axes), items,
    facts_lists(A, rank, Axis.inverse_axes(rank, axes), items, length, facts(rank, Axis.inverse_axes(rank, axes), bound, inverse_valid(rank, axes, bound, accepted))))
