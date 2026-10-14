import Base
import ./core.bend as Shape
import ./nat_proofs.bend as Nats
import ./broadcast_proofs.bend as Broadcast
import ./unique_proofs.bend as Unique
import ./construct_proofs.bend as Construct
import ./LAWS.bend as Laws

def plan_construct(+left: Shape.Shape, +right: Shape.Shape, -left_padding: Nat, -right_padding: Nat, -minimal: Shape.PadMinimal<left_padding, right_padding>, -output: List<&2, U32>, -aligned: Shape.Aligned(Shape.pad(left_padding, Shape.extents(left)), Shape.pad(right_padding, Shape.extents(right)), Shape.naturals(output)), +result: Shape.ConstructResult<output>) -> {Laws.planned_dimensions(left, right, Shape.broadcast_checked.construct(left, right, left_padding, right_padding, minimal, output, aligned, result)) == (Shape.dimensions(left), Shape.dimensions(right)) : List<&2, U32> & List<&2, U32>}:
  match result:
    case Shape.SError{error}:
      {==}
    case Shape.SBuilt{shape, preserved}:
      {==}

def plan_align(+left: Shape.Shape, +right: Shape.Shape, +left_padding: Nat, +right_padding: Nat, -minimal: Shape.PadMinimal<left_padding, right_padding>, +alignment: Shape.Alignment<Shape.pad_words(left_padding, Shape.dimensions(left)), Shape.pad_words(right_padding, Shape.dimensions(right))>) -> {Laws.planned_dimensions(left, right, Shape.broadcast_checked.align(left, right, left_padding, right_padding, minimal, alignment)) == (Shape.dimensions(left), Shape.dimensions(right)) : List<&2, U32> & List<&2, U32>}:
  match alignment:
    case Shape.Misaligned{}:
      {==}
    case Shape.Aligns{+output, aligned}:
      plan_construct(left, right, left_padding, right_padding, minimal, output, Shape.broadcast_checked.repad(left_padding, right_padding, Shape.dimensions(left), Shape.dimensions(right), output, aligned), Shape.construct(output))

def broadcast_checked_plan(+left: Shape.Shape, +right: Shape.Shape) -> {Laws.planned_dimensions(left, right, Shape.broadcast_checked(left, right)) == (Shape.dimensions(left), Shape.dimensions(right)) : List<&2, U32> & List<&2, U32>}:
  +left_rank = Shape.rank_dimensions(Shape.dimensions(left))
  +right_rank = Shape.rank_dimensions(Shape.dimensions(right))
  plan_align(left, right, Nat.sub(right_rank, left_rank), Nat.sub(left_rank, right_rank), Shape.minimal(left_rank, right_rank), Shape.align(Shape.pad_words(Nat.sub(right_rank, left_rank), Shape.dimensions(left)), Shape.pad_words(Nat.sub(left_rank, right_rank), Shape.dimensions(right))))

def some_injective(-A: Data, -x: A, -y: A, equal: {Some{x} == Some{y} : Maybe<&2, A>}) -> {x == y : A}:
  Equal.cong(Maybe<&2, A>, A, found => Maybe.default(&2, A, found, x), Some{x}, Some{y}, equal)

def some_none(-A: Data, -x: A, equal: {Some{x} == None{} : Maybe<&2, A>}) -> Empty:
  Shape.clash(Equal.cong(Maybe<&2, A>, Bool, found => Maybe.is_some(&2, A, found), Some{x}, None{}, equal))

def is_eq_refl(+value: Nat) -> {True{} == Nat.is_eq(value, value) : Bool}:
  match value:
    case 0n:
      {==}
    case 1n+p:
      is_eq_refl(p)

def eq_true(+a: Nat, -b: Nat, equal: {a == b : Nat}) -> {True{} == Nat.is_eq(a, b) : Bool}:
  %equal : {True{} == Nat.is_eq(a, _) : Bool}
  is_eq_refl(a)

def sub_balance(+a: Nat, +b: Nat) -> {Nat.add(Nat.sub(b, a), a) == Nat.add(Nat.sub(a, b), b) : Nat}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      Broadcast.add_zero(1n+q)
    case 1n+p 0n:
      Equal.sym(Nat, Nat.add(1n+p, 0n), 1n+p, Broadcast.add_zero(1n+p))
    case 1n+p 1n+q:
      %Broadcast.add_succ(Nat.sub(q, p), p) : {_ == Nat.add(Nat.sub(p, q), 1n+q) : Nat}
      %Broadcast.add_succ(Nat.sub(p, q), q) : {1n+Nat.add(Nat.sub(q, p), p) == _ : Nat}
      Equal.cong(Nat, Nat, value => 1n+value, Nat.add(Nat.sub(q, p), p), Nat.add(Nat.sub(p, q), q), sub_balance(p, q))

def choice_output(-left: U32, -right: U32, choice: Shape.AxisChoice<left, right>) -> Maybe<&2, Nat>:
  match choice:
    case Shape.Picks{output, axis}:
      Some{U32.to_nat(output)}
    case Shape.Clashes{}:
      None{}

def aligned_output(-left: List<&2, U32>, -right: List<&2, U32>, alignment: Shape.Alignment<left, right>) -> Maybe<&2, List<&2, Nat>>:
  match alignment:
    case Shape.Aligns{output, aligned}:
      Some{Shape.naturals(output)}
    case Shape.Misaligned{}:
      None{}

def some_nat(-a: Nat, -b: Nat, equal: {a == b : Nat}) -> {Some{a} == Some{b} : Maybe<&2, Nat>}:
  Equal.cong(Nat, Maybe<&2, Nat>, value => Some{value}, a, b, equal)

def axis_right_complete(+left: U32, +right: U32, -output: Nat, +one: Shape.BoolCase<Nat.is_eq(1n, U32.to_nat(right))>, +axis: Shape.Axis<U32.to_nat(left), U32.to_nat(right), output>, differ: {False{} == Nat.is_eq(U32.to_nat(left), U32.to_nat(right)) : Bool}, left_wide: {False{} == Nat.is_eq(1n, U32.to_nat(left)) : Bool}) -> {Some{output} == choice_output(left, right, Shape.axis_choice.right_one(left, right, one)) : Maybe<&2, Nat>}:
  match one:
    case Shape.BTrue{right_one}:
      match axis:
        case Shape.AxisEqual{left_eq, right_eq}:
          some_nat(output, U32.to_nat(left), left_eq)
        case Shape.AxisLeftOne{left_one, right_eq}:
          Empty.absurd({Some{output} == Some{U32.to_nat(left)} : Maybe<&2, Nat>}, Nats.contradiction(Nat.is_eq(1n, U32.to_nat(left)), eq_true(1n, U32.to_nat(left), left_one), left_wide))
        case Shape.AxisRightOne{left_eq, right_one_eq}:
          some_nat(output, U32.to_nat(left), left_eq)
    case Shape.BFalse{right_wide}:
      match axis:
        case Shape.AxisEqual{left_eq, right_eq}:
          Empty.absurd({Some{output} == None{} : Maybe<&2, Nat>}, Nats.contradiction(Nat.is_eq(U32.to_nat(left), U32.to_nat(right)), eq_true(U32.to_nat(left), U32.to_nat(right), Equal.trans(Nat, U32.to_nat(left), output, U32.to_nat(right), Equal.sym(Nat, output, U32.to_nat(left), left_eq), right_eq)), differ))
        case Shape.AxisLeftOne{left_one, right_eq}:
          Empty.absurd({Some{output} == None{} : Maybe<&2, Nat>}, Nats.contradiction(Nat.is_eq(1n, U32.to_nat(left)), eq_true(1n, U32.to_nat(left), left_one), left_wide))
        case Shape.AxisRightOne{left_eq, right_one}:
          Empty.absurd({Some{output} == None{} : Maybe<&2, Nat>}, Nats.contradiction(Nat.is_eq(1n, U32.to_nat(right)), eq_true(1n, U32.to_nat(right), right_one), right_wide))

def axis_left_complete(+left: U32, +right: U32, -output: Nat, +one: Shape.BoolCase<Nat.is_eq(1n, U32.to_nat(left))>, +axis: Shape.Axis<U32.to_nat(left), U32.to_nat(right), output>, differ: {False{} == Nat.is_eq(U32.to_nat(left), U32.to_nat(right)) : Bool}) -> {Some{output} == choice_output(left, right, Shape.axis_choice.left_one(left, right, one)) : Maybe<&2, Nat>}:
  match one:
    case Shape.BTrue{left_one}:
      match axis:
        case Shape.AxisEqual{left_eq, right_eq}:
          some_nat(output, U32.to_nat(right), right_eq)
        case Shape.AxisLeftOne{left_one_eq, right_eq}:
          some_nat(output, U32.to_nat(right), right_eq)
        case Shape.AxisRightOne{left_eq, right_one}:
          some_nat(output, U32.to_nat(right), Equal.trans(Nat, output, 1n, U32.to_nat(right), Equal.trans(Nat, output, U32.to_nat(left), 1n, left_eq, Equal.sym(Nat, 1n, U32.to_nat(left), Shape.equal_decided(1n, U32.to_nat(left), left_one))), right_one))
    case Shape.BFalse{left_wide}:
      axis_right_complete(left, right, output, Shape.bool_case(Nat.is_eq(1n, U32.to_nat(right))), axis, differ, left_wide)

def axis_equal_complete(+left: U32, +right: U32, -output: Nat, +same: Shape.BoolCase<Nat.is_eq(U32.to_nat(left), U32.to_nat(right))>, +axis: Shape.Axis<U32.to_nat(left), U32.to_nat(right), output>) -> {Some{output} == choice_output(left, right, Shape.axis_choice.equal(left, right, same)) : Maybe<&2, Nat>}:
  match same:
    case Shape.BTrue{equal}:
      match axis:
        case Shape.AxisEqual{left_eq, right_eq}:
          some_nat(output, U32.to_nat(left), left_eq)
        case Shape.AxisLeftOne{left_one, right_eq}:
          some_nat(output, U32.to_nat(left), Equal.trans(Nat, output, U32.to_nat(right), U32.to_nat(left), right_eq, Equal.sym(Nat, U32.to_nat(left), U32.to_nat(right), Shape.equal_decided(U32.to_nat(left), U32.to_nat(right), equal))))
        case Shape.AxisRightOne{left_eq, right_one}:
          some_nat(output, U32.to_nat(left), left_eq)
    case Shape.BFalse{differ}:
      axis_left_complete(left, right, output, Shape.bool_case(Nat.is_eq(1n, U32.to_nat(left))), axis, differ)

def axis_complete(+left: U32, +right: U32, -output: Nat, +axis: Shape.Axis<U32.to_nat(left), U32.to_nat(right), output>) -> {Some{output} == choice_output(left, right, Shape.axis_choice(left, right)) : Maybe<&2, Nat>}:
  axis_equal_complete(left, right, output, Shape.bool_case(Nat.is_eq(U32.to_nat(left), U32.to_nat(right))), axis)

def align_step_complete(-lh: U32, -lt: List<&2, U32>, -rh: U32, -rt: List<&2, U32>, -oh: Nat, -ot: List<&2, Nat>, +rest: Shape.Alignment<lt, rt>, +choice: Shape.AxisChoice<lh, rh>, rest_seen: {Some{ot} == aligned_output(lt, rt, rest) : Maybe<&2, List<&2, Nat>>}, choice_seen: {Some{oh} == choice_output(lh, rh, choice) : Maybe<&2, Nat>}) -> {Some{oh <> ot} == aligned_output(lh <> lt, rh <> rt, Shape.align.step(lh, lt, rh, rt, rest, choice)) : Maybe<&2, List<&2, Nat>>}:
  match rest:
    case Shape.Misaligned{}:
      Empty.absurd({Some{oh <> ot} == aligned_output(lh <> lt, rh <> rt, Shape.align.step(lh, lt, rh, rt, Shape.Misaligned{}, choice)) : Maybe<&2, List<&2, Nat>>}, some_none(List<&2, Nat>, ot, rest_seen))
    case Shape.Aligns{tail, aligned}:
      match choice:
        case Shape.Clashes{}:
          Empty.absurd({Some{oh <> ot} == None{} : Maybe<&2, List<&2, Nat>>}, some_none(Nat, oh, choice_seen))
        case Shape.Picks{output, axis}:
          %some_injective(Nat, oh, U32.to_nat(output), choice_seen) : {Some{oh <> ot} == Some{_ <> Shape.naturals(tail)} : Maybe<&2, List<&2, Nat>>}
          %some_injective(List<&2, Nat>, ot, Shape.naturals(tail), rest_seen) : {Some{oh <> ot} == Some{oh <> _} : Maybe<&2, List<&2, Nat>>}
          {==}

def align_complete(+left: List<&2, U32>, +right: List<&2, U32>, +output: List<&2, Nat>, +aligned: Shape.Aligned(Shape.naturals(left), Shape.naturals(right), output)) -> {Some{output} == aligned_output(left, right, Shape.align(left, right)) : Maybe<&2, List<&2, Nat>>}:
  match left right output:
    case Nil{} Nil{} Nil{}:
      {==}
    case +lh <> +lt +rh <> +rt +oh <> +ot:
      Shape.AStep{+axis, +tail} = aligned
      align_step_complete(lh, lt, rh, rt, oh, ot, Shape.align(lt, rt), Shape.axis_choice(lh, rh), align_complete(lt, rt, ot, tail), axis_complete(lh, rh, oh, axis))
    case Nil{} Nil{} oh <> ot:
      match aligned:
    case Nil{} rh <> rt Nil{}:
      match aligned:
    case lh <> lt Nil{} Nil{}:
      match aligned:
    case Nil{} rh <> rt oh <> ot:
      match aligned:
    case lh <> lt Nil{} oh <> ot:
      match aligned:
    case lh <> lt rh <> rt Nil{}:
      match aligned:

def complete_construct(+left: Shape.Shape, +right: Shape.Shape, -left_padding: Nat, -right_padding: Nat, -minimal: Shape.PadMinimal<left_padding, right_padding>, -output: List<&2, U32>, -aligned: Shape.Aligned(Shape.pad(left_padding, Shape.extents(left)), Shape.pad(right_padding, Shape.extents(right)), Shape.naturals(output)), +result: Shape.ConstructResult<output>, built: {Laws.built(output, result) == Some{output} : Maybe<&2, List<&2, U32>>}) -> {Laws.broadcast_built(Shape.broadcast_checked.construct(left, right, left_padding, right_padding, minimal, output, aligned, result)) == True{} : Bool}:
  match result:
    case Shape.SError{error}:
      Empty.absurd({False{} == True{} : Bool}, some_none(List<&2, U32>, output, Equal.sym(Maybe<&2, List<&2, U32>>, None{}, Some{output}, built)))
    case Shape.SBuilt{shape, preserved}:
      {==}

def complete_align(+left: Shape.Shape, +right: Shape.Shape, +left_padding: Nat, +right_padding: Nat, -minimal: Shape.PadMinimal<left_padding, right_padding>, +alignment: Shape.Alignment<Shape.pad_words(left_padding, Shape.dimensions(left)), Shape.pad_words(right_padding, Shape.dimensions(right))>, -target: List<&2, Nat>, seen: {Some{target} == aligned_output(Shape.pad_words(left_padding, Shape.dimensions(left)), Shape.pad_words(right_padding, Shape.dimensions(right)), alignment) : Maybe<&2, List<&2, Nat>>}, +rank_bound: Shape.LE(List.length(&2, Nat, target), 4n), +capacity_bound: Shape.LE(Shape.numel(target), 1048576n)) -> {Laws.broadcast_built(Shape.broadcast_checked.align(left, right, left_padding, right_padding, minimal, alignment)) == True{} : Bool}:
  match alignment:
    case Shape.Misaligned{}:
      Empty.absurd({False{} == True{} : Bool}, some_none(List<&2, Nat>, target, seen))
    case Shape.Aligns{+output, aligned}:
      +same = some_injective(List<&2, Nat>, target, Shape.naturals(output), seen)
      +rank_fits = Nats.le_rewrite(Shape.rank_dimensions(output), List.length(&2, Nat, target), 4n, Equal.trans(Nat, List.length(&2, Nat, target), List.length(&2, Nat, Shape.naturals(output)), Shape.rank_dimensions(output), Equal.cong(List<&2, Nat>, Nat, values => List.length(&2, Nat, values), target, Shape.naturals(output), same), Equal.sym(Nat, Shape.rank_dimensions(output), List.length(&2, Nat, Shape.naturals(output)), Shape.rank_naturals(output))), rank_bound)
      +capacity_fits = Nats.le_rewrite(Shape.numel(Shape.naturals(output)), Shape.numel(target), 1048576n, Equal.cong(List<&2, Nat>, Nat, values => Shape.numel(values), target, Shape.naturals(output), same), capacity_bound)
      complete_construct(left, right, left_padding, right_padding, minimal, output, Shape.broadcast_checked.repad(left_padding, right_padding, Shape.dimensions(left), Shape.dimensions(right), output, aligned), Shape.construct(output), Construct.construction_complete(output, Shape.bool_case(Nat.is_le(Shape.rank_dimensions(output), 4n)), Nats.le_true(Shape.rank_dimensions(output), 4n, rank_fits), capacity_fits))

def padded_words(+left_padding: Nat, +right_padding: Nat, -left: List<&2, U32>, -right: List<&2, U32>, -output: List<&2, Nat>, aligned: Shape.Aligned(Shape.pad(left_padding, Shape.naturals(left)), Shape.pad(right_padding, Shape.naturals(right)), output)) -> Shape.Aligned(Shape.naturals(Shape.pad_words(left_padding, left)), Shape.naturals(Shape.pad_words(right_padding, right)), output):
  %Equal.sym(List<&2, Nat>, Shape.naturals(Shape.pad_words(left_padding, left)), Shape.pad(left_padding, Shape.naturals(left)), Shape.naturals_pad(left_padding, left)) : Shape.Aligned(_, Shape.naturals(Shape.pad_words(right_padding, right)), output)
  %Equal.sym(List<&2, Nat>, Shape.naturals(Shape.pad_words(right_padding, right)), Shape.pad(right_padding, Shape.naturals(right)), Shape.naturals_pad(right_padding, right)) : Shape.Aligned(Shape.pad(left_padding, Shape.naturals(left)), _, output)
  aligned

def broadcast_checked_complete(+left: Shape.Shape, +right: Shape.Shape, +output: List<&2, U32>, +alignment: Shape.Broadcast<Shape.extents(left), Shape.extents(right), Shape.naturals(output)>, +rank_bound: Shape.LE(Shape.rank_dimensions(output), 4n), +capacity_bound: Shape.LE(Shape.logical_numel(output), 1048576n)) -> {Laws.broadcast_built(Shape.broadcast_checked(left, right)) == True{} : Bool}:
  match alignment:
    case Shape.BAlign{+left_padding, +right_padding, +minimal, +aligned}:
      +left_rank = Shape.rank_dimensions(Shape.dimensions(left))
      +right_rank = Shape.rank_dimensions(Shape.dimensions(right))
      +left_extra = Nat.sub(right_rank, left_rank)
      +right_extra = Nat.sub(left_rank, right_rank)
      +balanced = Equal.trans(Nat, Nat.add(left_padding, left_rank), Nat.add(left_padding, List.length(&2, Nat, Shape.extents(left))), Nat.add(right_padding, right_rank), Equal.cong(Nat, Nat, value => Nat.add(left_padding, value), left_rank, List.length(&2, Nat, Shape.extents(left)), Shape.rank_naturals(Shape.dimensions(left))), Equal.trans(Nat, Nat.add(left_padding, List.length(&2, Nat, Shape.extents(left))), Nat.add(right_padding, List.length(&2, Nat, Shape.extents(right))), Nat.add(right_padding, right_rank), Unique.balance(left_padding, right_padding, Shape.extents(left), Shape.extents(right), Shape.naturals(output), aligned), Equal.sym(Nat, Nat.add(right_padding, right_rank), Nat.add(right_padding, List.length(&2, Nat, Shape.extents(right))), Equal.cong(Nat, Nat, value => Nat.add(right_padding, value), right_rank, List.length(&2, Nat, Shape.extents(right)), Shape.rank_naturals(Shape.dimensions(right))))))
      +computed = sub_balance(left_rank, right_rank)
      +same_left = Unique.left_padding_equal(left_padding, right_padding, left_extra, right_extra, left_rank, right_rank, minimal, Shape.minimal(left_rank, right_rank), balanced, computed)
      +same_right = Unique.cancel_right(right_padding, right_extra, right_rank, Equal.trans(Nat, Nat.add(right_padding, right_rank), Nat.add(left_padding, left_rank), Nat.add(right_extra, right_rank), Equal.sym(Nat, Nat.add(left_padding, left_rank), Nat.add(right_padding, right_rank), balanced), Equal.trans(Nat, Nat.add(left_padding, left_rank), Nat.add(left_extra, left_rank), Nat.add(right_extra, right_rank), Equal.cong(Nat, Nat, value => Nat.add(value, left_rank), left_padding, left_extra, same_left), computed)))
      +retyped = Unique.retype(left_extra, left_padding, right_extra, right_padding, Shape.extents(left), Shape.extents(right), Shape.naturals(output), Equal.sym(Nat, left_padding, left_extra, same_left), Equal.sym(Nat, right_padding, right_extra, same_right), aligned)
      +seen = align_complete(Shape.pad_words(left_extra, Shape.dimensions(left)), Shape.pad_words(right_extra, Shape.dimensions(right)), Shape.naturals(output), padded_words(left_extra, right_extra, Shape.dimensions(left), Shape.dimensions(right), Shape.naturals(output), retyped))
      complete_align(left, right, left_extra, right_extra, Shape.minimal(left_rank, right_rank), Shape.align(Shape.pad_words(left_extra, Shape.dimensions(left)), Shape.pad_words(right_extra, Shape.dimensions(right))), Shape.naturals(output), seen, Nats.le_rewrite(List.length(&2, Nat, Shape.naturals(output)), Shape.rank_dimensions(output), 4n, Shape.rank_naturals(output), rank_bound), capacity_bound)
