import Base
import ../shape/core.bend as Shape
import ../shape/nat_proofs.bend as NatProofs
import ../storage/arith.bend as Arith
import ./core.bend as Core
import ./layout.bend as Layout
import ./transforms.bend as Transforms
import ./transpose-spec.bend as Transpose
import ./elementwise-totality.bend as Totality
import ./signed.bend as Signed
import ./signed-LAWS.bend as Laws

def Laws.signed_first_kept(limit, k):
  {==}

# n < n is false.
def lt_self(+n: Nat) -> {Nat.is_lt(n, n) == False{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      lt_self(p)

# w < w + (k + 1).
def lt_past(+w: Nat, +k: Nat) -> {Nat.is_lt(w, Nat.add(w, 1n+k)) == True{} : Bool}:
  match w:
    case 0n:
      {==}
    case 1n+p:
      lt_past(p, k)

# a < 1 + b gives a <= b.
def le_from_lt(+a: Nat, +b: Nat, h: {Nat.is_lt(a, 1n+b) == True{} : Bool}) -> {Nat.is_le(a, b) == True{} : Bool}:
  match a:
    case 0n:
      Arith.le_zero(b)
    case 1n+p:
      match b:
        case 0n:
          Empty.absurd({Nat.is_le(1n+p, 0n) == True{} : Bool}, Arith.lt_zero(p, h))
        case 1n+q:
          le_from_lt(p, q, h)

# a < b gives 1 <= b.
def lt_positive(+a: Nat, +b: Nat, h: {Nat.is_lt(a, b) == True{} : Bool}) -> {Nat.is_le(1n, b) == True{} : Bool}:
  match b:
    case 0n:
      Empty.absurd({Nat.is_le(1n, 0n) == True{} : Bool}, Arith.lt_zero(a, h))
    case 1n+q:
      Arith.le_zero(q)

def add_succ_right(+a: Nat, +b: Nat) -> {Nat.add(a, 1n+b) == 1n+Nat.add(a, b) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      Equal.cong(Nat, Nat, x => 1n+x, Nat.add(p, 1n+b), 1n+Nat.add(p, b), add_succ_right(p, b))

def add_zero_right(+a: Nat) -> {Nat.add(a, 0n) == a : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      Equal.cong(Nat, Nat, x => 1n+x, Nat.add(p, 0n), p, add_zero_right(p))

def add_comm(+a: Nat, +b: Nat) -> {Nat.add(a, b) == Nat.add(b, a) : Nat}:
  match a:
    case 0n:
      Equal.sym(Nat, Nat.add(b, 0n), b, add_zero_right(b))
    case 1n+p:
      Equal.trans(Nat, 1n+Nat.add(p, b), 1n+Nat.add(b, p), Nat.add(b, 1n+p),
        Equal.cong(Nat, Nat, x => 1n+x, Nat.add(p, b), Nat.add(b, p), add_comm(p, b)),
        Equal.sym(Nat, Nat.add(b, 1n+p), 1n+Nat.add(b, p), add_succ_right(b, p)))

# Last{k} in range is limit - 1 - k: the two subtractions borrow nothing.
def last_inside(+limit: U32, +k: U32, +inside: {U32.is_lt(k, limit) == True{} : Bool}) ->
  {Nat.add(U32.to_nat(U32.sub(U32.sub(limit, 1), k)), 1n+U32.to_nat(k)) == U32.to_nat(limit) : Nat}:
  +below = Equal.trans(Bool, Nat.is_lt(U32.to_nat(k), U32.to_nat(limit)), U32.is_lt(k, limit), True{}, Arith.u32_lt(k, limit), inside)
  +first = Arith.sub_u32(limit, 1, lt_positive(U32.to_nat(k), U32.to_nat(limit), below))
  shifted = Equal.trans(Bool, Nat.is_lt(U32.to_nat(k), 1n+U32.to_nat(U32.sub(limit, 1))), Nat.is_lt(U32.to_nat(k), U32.to_nat(limit)), True{},
    Equal.cong(Nat, Bool, x => Nat.is_lt(U32.to_nat(k), x), 1n+U32.to_nat(U32.sub(limit, 1)), U32.to_nat(limit),
      Equal.sym(Nat, U32.to_nat(limit), 1n+U32.to_nat(U32.sub(limit, 1)), first)),
    below)
  second = Arith.sub_u32(U32.sub(limit, 1), k, le_from_lt(U32.to_nat(k), U32.to_nat(U32.sub(limit, 1)), shifted))
  Equal.trans(Nat, Nat.add(U32.to_nat(U32.sub(U32.sub(limit, 1), k)), 1n+U32.to_nat(k)),
    1n+Nat.add(U32.to_nat(U32.sub(U32.sub(limit, 1), k)), U32.to_nat(k)), U32.to_nat(limit),
    add_succ_right(U32.to_nat(U32.sub(U32.sub(limit, 1), k)), U32.to_nat(k)),
    Equal.trans(Nat, 1n+Nat.add(U32.to_nat(U32.sub(U32.sub(limit, 1), k)), U32.to_nat(k)),
      1n+Nat.add(U32.to_nat(k), U32.to_nat(U32.sub(U32.sub(limit, 1), k))), U32.to_nat(limit),
      Equal.cong(Nat, Nat, x => 1n+x, Nat.add(U32.to_nat(U32.sub(U32.sub(limit, 1), k)), U32.to_nat(k)),
        Nat.add(U32.to_nat(k), U32.to_nat(U32.sub(U32.sub(limit, 1), k))),
        add_comm(U32.to_nat(U32.sub(U32.sub(limit, 1), k)), U32.to_nat(k))),
      Equal.trans(Nat, 1n+Nat.add(U32.to_nat(k), U32.to_nat(U32.sub(U32.sub(limit, 1), k))),
        1n+U32.to_nat(U32.sub(limit, 1)), U32.to_nat(limit),
        Equal.cong(Nat, Nat, x => 1n+x, Nat.add(U32.to_nat(k), U32.to_nat(U32.sub(U32.sub(limit, 1), k))),
          U32.to_nat(U32.sub(limit, 1)),
          Equal.sym(Nat, U32.to_nat(U32.sub(limit, 1)), Nat.add(U32.to_nat(k), U32.to_nat(U32.sub(U32.sub(limit, 1), k))), second)),
        Equal.sym(Nat, U32.to_nat(limit), 1n+U32.to_nat(U32.sub(limit, 1)), first))))

def last_wrapped(+limit: U32, +k: U32, +inside: {U32.is_lt(k, limit) == True{} : Bool}) ->
  {Nat.add(U32.to_nat(Signed.wrap(limit, Signed.Last{k})), 1n+U32.to_nat(k)) == U32.to_nat(limit) : Nat}:
  %Equal.sym(Bool, U32.is_lt(k, limit), True{}, inside) :
    {Nat.add(U32.to_nat(Bool.pick(U32, _, U32.sub(U32.sub(limit, 1), k), limit)), 1n+U32.to_nat(k)) == U32.to_nat(limit) : Nat}
  last_inside(limit, k, inside)

def Laws.signed_last_inside(limit, k, inside):
  last_wrapped(limit, k, inside)

def Laws.signed_last_outside(limit, k, outside):
  %Equal.sym(Bool, U32.is_lt(k, limit), False{}, outside) :
    {Bool.pick(U32, _, U32.sub(U32.sub(limit, 1), k), limit) == limit : U32}
  {==}

# In range, limit - 1 - k is below limit.
def last_below(+limit: U32, +k: U32, +inside: {U32.is_lt(k, limit) == True{} : Bool}) ->
  {U32.is_lt(U32.sub(U32.sub(limit, 1), k), limit) == True{} : Bool}:
  Equal.trans(Bool, U32.is_lt(U32.sub(U32.sub(limit, 1), k), limit), Nat.is_lt(U32.to_nat(U32.sub(U32.sub(limit, 1), k)), U32.to_nat(limit)), True{},
    Equal.sym(Bool, Nat.is_lt(U32.to_nat(U32.sub(U32.sub(limit, 1), k)), U32.to_nat(limit)), U32.is_lt(U32.sub(U32.sub(limit, 1), k), limit),
      Arith.u32_lt(U32.sub(U32.sub(limit, 1), k), limit)),
    Equal.trans(Bool, Nat.is_lt(U32.to_nat(U32.sub(U32.sub(limit, 1), k)), U32.to_nat(limit)),
      Nat.is_lt(U32.to_nat(U32.sub(U32.sub(limit, 1), k)), Nat.add(U32.to_nat(U32.sub(U32.sub(limit, 1), k)), 1n+U32.to_nat(k))), True{},
      Equal.cong(Nat, Bool, x => Nat.is_lt(U32.to_nat(U32.sub(U32.sub(limit, 1), k)), x), U32.to_nat(limit),
        Nat.add(U32.to_nat(U32.sub(U32.sub(limit, 1), k)), 1n+U32.to_nat(k)),
        Equal.sym(Nat, Nat.add(U32.to_nat(U32.sub(U32.sub(limit, 1), k)), 1n+U32.to_nat(k)), U32.to_nat(limit), last_inside(limit, k, inside))),
      lt_past(U32.to_nat(U32.sub(U32.sub(limit, 1), k)), U32.to_nat(k))))

# limit is not below itself.
def limit_not_below(+limit: U32) -> {U32.is_lt(limit, limit) == False{} : Bool}:
  Equal.trans(Bool, U32.is_lt(limit, limit), Nat.is_lt(U32.to_nat(limit), U32.to_nat(limit)), False{},
    Equal.sym(Bool, Nat.is_lt(U32.to_nat(limit), U32.to_nat(limit)), U32.is_lt(limit, limit), Arith.u32_lt(limit, limit)),
    lt_self(U32.to_nat(limit)))

def last_decided(+limit: U32, +k: U32, decision: Shape.BoolCase<U32.is_lt(k, limit)>) ->
  {U32.is_lt(Signed.wrap(limit, Signed.Last{k}), limit) == U32.is_lt(k, limit) : Bool}:
  match decision:
    case Shape.BTrue{+equal}:
      %equal : {U32.is_lt(Bool.pick(U32, _, U32.sub(U32.sub(limit, 1), k), limit), limit) == _ : Bool}
      last_below(limit, k, Equal.sym(Bool, True{}, U32.is_lt(k, limit), equal))
    case Shape.BFalse{equal}:
      %equal : {U32.is_lt(Bool.pick(U32, _, U32.sub(U32.sub(limit, 1), k), limit), limit) == _ : Bool}
      limit_not_below(limit)

def Laws.signed_wrap_inside(limit, position):
  match position:
    case Signed.First{k}:
      {==}
    case Signed.Last{+k}:
      last_decided(limit, k, Shape.bool_case(U32.is_lt(k, limit)))

def Laws.signed_axes_bounded(limit, positions):
  match positions:
    case Nil{}:
      {==}
    case +head <> +tail:
      %Equal.sym(Bool, U32.is_lt(Signed.wrap(limit, head), limit), Signed.inside(limit, head), Laws.signed_wrap_inside(limit, head)) :
        {Bool.and(_, Layout.bounded(Signed.wrap_all(limit, tail), limit)) == Bool.and(Signed.inside(limit, head), Signed.all_inside(limit, tail)) : Bool}
      %Equal.sym(Bool, Layout.bounded(Signed.wrap_all(limit, tail), limit), Signed.all_inside(limit, tail), Laws.signed_axes_bounded(limit, tail)) :
        {Bool.and(Signed.inside(limit, head), _) == Bool.and(Signed.inside(limit, head), Signed.all_inside(limit, tail)) : Bool}
      {==}

def Laws.signed_wrap_all_empty(limit):
  {==}

def Laws.signed_wrap_all_step(limit, head, tail):
  {==}

def Laws.signed_wrap_each_step(extent, rest, head, tail):
  {==}

def Laws.signed_wrap_each_end(dimensions):
  match dimensions:
    case Nil{}:
      {==}
    case extent <> rest:
      {==}

def Laws.signed_wrap_each_past_rank(positions):
  {==}

def Laws.signed_axis_limit(input):
  {==}

def Laws.signed_permute_wraps(input, source, axes):
  {==}

def Laws.signed_permute_dynamic_wraps(input, axes):
  Core.Packed{+shape, tensor} = input
  {==}

def Laws.signed_transpose_wraps(input, source, a, b):
  {==}

def Laws.signed_transpose_dynamic_wraps(input, a, b):
  Core.Packed{+shape, tensor} = input
  {==}

def Laws.signed_read_coordinate_wraps(shape, tensor, coordinates):
  {==}

def Laws.signed_write_coordinate_wraps(shape, tensor, coordinates, value):
  {==}

def Laws.signed_read_flat_wraps(shape, tensor, index):
  {==}

def Laws.signed_write_flat_wraps(shape, tensor, index, value):
  {==}

def Laws.signed_read_dynamic_wraps(input, index):
  Core.Packed{+shape, tensor} = input
  {==}

def Laws.signed_write_dynamic_wraps(input, index, value):
  Core.Packed{+shape, tensor} = input
  {==}

def Laws.signed_reshape_infers(input, tensor, extents):
  {==}

def Laws.signed_reshape_dynamic_infers(input, tensor, extents):
  {==}

def exacts_none(+dimensions: List<&2, U32>) -> {Signed.inferred_count(Laws.exacts(dimensions)) == 0n : Nat}:
  match dimensions:
    case Nil{}:
      {==}
    case extent <> +rest:
      exacts_none(rest)

def exacts_filled(+dimensions: List<&2, U32>, +size: U32) -> {Signed.filled(Laws.exacts(dimensions), size) == dimensions : List<&2, U32>}:
  match dimensions:
    case Nil{}:
      {==}
    case +extent <> +rest:
      Equal.cong(List<&2, U32>, List<&2, U32>, tail => extent <> tail, Signed.filled(Laws.exacts(rest), size), rest, exacts_filled(rest, size))

def Laws.signed_reshape_unsigned(input, tensor, dimensions):
  %Equal.sym(Nat, Signed.inferred_count(Laws.exacts(dimensions)), 0n, exacts_none(dimensions)) :
    {Signed.reshape_inferred(input, tensor, Signed.inferred_from(Shape.elements(input), Laws.exacts(dimensions), _)) ==
      Signed.transformed(Transforms.reshape(input, tensor, dimensions)) : Result<&2, &1, Signed.Error, Core.Dynamic>}
  %Equal.sym(List<&2, U32>, Signed.filled(Laws.exacts(dimensions), 0), dimensions, exacts_filled(dimensions, 0)) :
    {Signed.reshape_inferred(input, tensor, Done{_}) ==
      Signed.transformed(Transforms.reshape(input, tensor, dimensions)) : Result<&2, &1, Signed.Error, Core.Dynamic>}
  {==}

def Laws.signed_inferred_none(count, extents, none):
  %Equal.sym(Nat, Signed.inferred_count(extents), 0n, none) :
    {Signed.inferred_from(count, extents, _) == Done{Signed.filled(extents, 0)} : Result<&2, &2, Signed.Error, List<&2, U32>>}
  {==}

# q * (1 + rest) = count gives count divmod (1 + rest) = (q, 0).
def divides(+count: Nat, +rest: Nat, +quotient: Nat, exact: {Nat.mul(quotient, 1n+rest) == count : Nat}) ->
  {Nat.divmod(count, 1n+rest) == (quotient, 0n) : Nat & Nat}:
  Equal.trans(Nat & Nat, Nat.divmod(count, 1n+rest), Nat.divmod(Nat.add(Nat.mul(quotient, 1n+rest), 0n), 1n+rest), (quotient, 0n),
    Equal.cong(Nat, Nat & Nat, x => Nat.divmod(x, 1n+rest), count, Nat.add(Nat.mul(quotient, 1n+rest), 0n),
      Equal.sym(Nat, Nat.add(Nat.mul(quotient, 1n+rest), 0n), count,
        Equal.trans(Nat, Nat.add(Nat.mul(quotient, 1n+rest), 0n), Nat.mul(quotient, 1n+rest), count,
          add_zero_right(Nat.mul(quotient, 1n+rest)), exact))),
    Equal.sym(Nat & Nat, (quotient, 0n), Nat.divmod(Nat.add(Nat.mul(quotient, 1n+rest), 0n), 1n+rest),
      NatProofs.divmod_exact(quotient, 1n+rest, 0n, Unit{})))

# The inference classifies the exact sizes' product against the count
# (Signed.capped) instead of forming it. These lemmas give capped as the
# classification of exact_product, so the laws stated with exact_product hold.
def le_add(+a: Nat, +b: Nat) -> {Nat.is_le(a, Nat.add(a, b)) == True{} : Bool}:
  match a:
    case 0n:
      Arith.le_zero(b)
    case 1n+p:
      le_add(p, b)

def not_le_add(+a: Nat, +b: Nat, +c: Nat, h: {Nat.is_le(a, c) == False{} : Bool}) ->
  {Nat.is_le(Nat.add(a, b), c) == False{} : Bool}:
  match a:
    case 0n:
      Empty.absurd({Nat.is_le(Nat.add(0n, b), c) == False{} : Bool},
        Arith.clash_true(Equal.trans(Bool, True{}, Nat.is_le(0n, c), False{},
          Equal.sym(Bool, Nat.is_le(0n, c), True{}, Arith.le_zero(c)), h)))
    case 1n+p:
      match c:
        case 0n:
          {==}
        case 1n+q:
          not_le_add(p, b, q, h)

# b < a gives a <= b false.
def lt_not_le(+b: Nat, +a: Nat, h: {Nat.is_lt(b, a) == True{} : Bool}) -> {Nat.is_le(a, b) == False{} : Bool}:
  match b:
    case 0n:
      match a:
        case 0n:
          Empty.absurd({Nat.is_le(0n, 0n) == False{} : Bool}, Arith.clash(h))
        case 1n+q:
          {==}
    case 1n+p:
      match a:
        case 0n:
          Empty.absurd({Nat.is_le(0n, 1n+p) == False{} : Bool}, Arith.clash(h))
        case 1n+q:
          lt_not_le(p, q, h)

# A size past the bound times a nonzero product is past the bound.
def past_bound(+bound: Nat, +less: Nat, +rest: Nat, past: {Nat.is_lt(bound, 1n+less) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(1n+less, 1n+rest), bound) == False{} : Bool}:
  %NatProofs.mul_comm(1n+rest, 1n+less) : {Nat.is_le(_, bound) == False{} : Bool}
  not_le_add(1n+less, Nat.mul(rest, 1n+less), bound, lt_not_le(bound, 1n+less, past))

# signed.bend's cap sorts a product as the laws' classified does.
def bounded_classified(+product: Nat, +below: Bool) -> {Signed.bounded(product, below) == Laws.classified(product, below) : Signed.Product}:
  match below:
    case True{}:
      {==}
    case False{}:
      {==}

def times_scaled(+bound: Nat, +less: Nat, +rest: Nat, decision: Shape.BoolCase<Nat.is_lt(bound, 1n+less)>) ->
  {Signed.scaled(bound, 1n+less, 1n+rest, Nat.is_lt(bound, 1n+less)) == Laws.classify(bound, Nat.mul(1n+less, 1n+rest)) : Signed.Product}:
  match decision:
    case Shape.BTrue{+equal}:
      %equal : {Signed.scaled(bound, 1n+less, 1n+rest, _) == Laws.classify(bound, Nat.mul(1n+less, 1n+rest)) : Signed.Product}
      %Equal.sym(Bool, Nat.is_le(Nat.mul(1n+less, 1n+rest), bound), False{},
        past_bound(bound, less, rest, Equal.sym(Bool, True{}, Nat.is_lt(bound, 1n+less), equal))) :
        {Signed.Beyond{} == Laws.classified(Nat.mul(1n+less, 1n+rest), _) : Signed.Product}
      {==}
    case Shape.BFalse{equal}:
      %equal : {Signed.scaled(bound, 1n+less, 1n+rest, _) == Laws.classify(bound, Nat.mul(1n+less, 1n+rest)) : Signed.Product}
      bounded_classified(Nat.mul(1n+less, 1n+rest), Nat.is_le(Nat.mul(1n+less, 1n+rest), bound))

def times_within(+bound: Nat, +less: Nat, +rest: Nat, decision: Shape.BoolCase<Nat.is_le(1n+rest, bound)>) ->
  {Signed.times(bound, 1n+less, Laws.classified(1n+rest, Nat.is_le(1n+rest, bound))) == Laws.classify(bound, Nat.mul(1n+less, 1n+rest)) : Signed.Product}:
  match decision:
    case Shape.BTrue{equal}:
      %equal : {Signed.times(bound, 1n+less, Laws.classified(1n+rest, _)) == Laws.classify(bound, Nat.mul(1n+less, 1n+rest)) : Signed.Product}
      times_scaled(bound, less, rest, Shape.bool_case(Nat.is_lt(bound, 1n+less)))
    case Shape.BFalse{+equal}:
      %equal : {Signed.times(bound, 1n+less, Laws.classified(1n+rest, _)) == Laws.classify(bound, Nat.mul(1n+less, 1n+rest)) : Signed.Product}
      %Equal.sym(Bool, Nat.is_le(Nat.mul(1n+less, 1n+rest), bound), False{},
        not_le_add(1n+rest, Nat.mul(less, 1n+rest), bound, Equal.sym(Bool, False{}, Nat.is_le(1n+rest, bound), equal))) :
        {Signed.Beyond{} == Laws.classified(Nat.mul(1n+less, 1n+rest), _) : Signed.Product}
      {==}

# times on a classified product classifies the product times the size.
def times_classify(+bound: Nat, +size: Nat, +product: Nat) ->
  {Signed.times(bound, size, Laws.classify(bound, product)) == Laws.classify(bound, Nat.mul(size, product)) : Signed.Product}:
  match size:
    case 0n:
      {==}
    case 1n+less:
      match product:
        case 0n:
          %Equal.sym(Nat, Nat.mul(1n+less, 0n), 0n, NatProofs.mul_zero_right(1n+less)) :
            {Signed.ZeroProduct{} == Laws.classify(bound, _) : Signed.Product}
          {==}
        case 1n+rest:
          times_within(bound, less, rest, Shape.bool_case(Nat.is_le(1n+rest, bound)))

def capped_classifies(+bound: Nat, +extents: List<&2, Signed.Extent>) ->
  {Signed.capped(bound, extents) == Laws.classify(bound, Signed.exact_product(extents)) : Signed.Product}:
  match extents:
    case Nil{}:
      bounded_classified(1n, Nat.is_le(1n, bound))
    case Signed.Exact{+size} <> +tail:
      %Equal.sym(Signed.Product, Signed.capped(bound, tail), Laws.classify(bound, Signed.exact_product(tail)), capped_classifies(bound, tail)) :
        {Signed.times(bound, U32.to_nat(size), _) == Laws.classify(bound, Nat.mul(U32.to_nat(size), Signed.exact_product(tail))) : Signed.Product}
      times_classify(bound, U32.to_nat(size), Signed.exact_product(tail))
    case Signed.Inferred{} <> +tail:
      capped_classifies(bound, tail)

def Laws.signed_capped_classifies(bound, extents):
  capped_classifies(bound, extents)

def is_within(product: Signed.Product) -> Bool:
  match product:
    case Signed.Within{p}:
      True{}
    case Signed.ZeroProduct{}:
      False{}
    case Signed.Beyond{}:
      False{}

def within_value(product: Signed.Product) -> Nat:
  match product:
    case Signed.Within{p}:
      p
    case Signed.ZeroProduct{}:
      0n
    case Signed.Beyond{}:
      0n

def within_decided(+bound: Nat, +r: Nat, +p: Nat, +within: {Laws.classified(1n+r, Nat.is_le(1n+r, bound)) == Signed.Within{p} : Signed.Product},
  decision: Shape.BoolCase<Nat.is_le(1n+r, bound)>) -> {Nat.is_le(p, bound) == True{} : Bool}:
  match decision:
    case Shape.BTrue{+equal}:
      %Equal.cong(Signed.Product, Nat, within_value, Laws.classified(1n+r, Nat.is_le(1n+r, bound)), Signed.Within{p}, within) :
        {Nat.is_le(_, bound) == True{} : Bool}
      %equal : {Nat.is_le(within_value(Laws.classified(1n+r, _)), bound) == True{} : Bool}
      Equal.sym(Bool, True{}, Nat.is_le(1n+r, bound), equal)
    case Shape.BFalse{equal}:
      Empty.absurd({Nat.is_le(p, bound) == True{} : Bool},
        Arith.clash(Equal.trans(Bool, is_within(Laws.classified(1n+r, False{})), is_within(Laws.classified(1n+r, Nat.is_le(1n+r, bound))), True{},
          Equal.cong(Bool, Bool, x => is_within(Laws.classified(1n+r, x)), False{}, Nat.is_le(1n+r, bound), equal),
          Equal.cong(Signed.Product, Bool, is_within, Laws.classified(1n+r, Nat.is_le(1n+r, bound)), Signed.Within{p}, within))))

def classify_within(+bound: Nat, +product: Nat, +p: Nat, within: {Laws.classify(bound, product) == Signed.Within{p} : Signed.Product}) ->
  {Nat.is_le(p, bound) == True{} : Bool}:
  match product:
    case 0n:
      Empty.absurd({Nat.is_le(p, bound) == True{} : Bool},
        Arith.clash(Equal.cong(Signed.Product, Bool, is_within, Signed.ZeroProduct{}, Signed.Within{p}, within)))
    case 1n+r:
      within_decided(bound, r, p, within, Shape.bool_case(Nat.is_le(1n+r, bound)))

def Laws.signed_capped_within(bound, extents, product, within):
  classify_within(bound, Signed.exact_product(extents), product,
    Equal.trans(Signed.Product, Laws.classify(bound, Signed.exact_product(extents)), Signed.capped(bound, extents), Signed.Within{product},
      Equal.sym(Signed.Product, Signed.capped(bound, extents), Laws.classify(bound, Signed.exact_product(extents)), capped_classifies(bound, extents)),
      within))

# The two branches of a single Inferred size with a nonzero product p: p at
# most the count divides; p past it leaves an empty input's size 0 and
# refuses any other.
def size_beyond(+count: Nat, +extents: List<&2, Signed.Extent>, +rest: Nat, +quotient: Nat,
  exact: {Nat.mul(quotient, 1n+rest) == count : Nat}, past: {Nat.is_le(1n+rest, count) == False{} : Bool}) ->
  {Signed.beyond(count, extents) == Done{Signed.filled(extents, U32.from_nat(quotient))} : Result<&2, &2, Signed.Error, List<&2, U32>>}:
  match quotient:
    case 0n:
      %exact : {Signed.beyond(_, extents) == Done{Signed.filled(extents, U32.from_nat(0n))} : Result<&2, &2, Signed.Error, List<&2, U32>>}
      {==}
    case 1n+q:
      Empty.absurd({Signed.beyond(count, extents) == Done{Signed.filled(extents, U32.from_nat(1n+q))} : Result<&2, &2, Signed.Error, List<&2, U32>>},
        Arith.clash_true(Equal.trans(Bool, True{}, Nat.is_le(1n+rest, count), False{},
          Equal.trans(Bool, True{}, Nat.is_le(1n+rest, Nat.add(1n+rest, Nat.mul(q, 1n+rest))), Nat.is_le(1n+rest, count),
            Equal.sym(Bool, Nat.is_le(1n+rest, Nat.add(1n+rest, Nat.mul(q, 1n+rest))), True{}, le_add(1n+rest, Nat.mul(q, 1n+rest))),
            Equal.cong(Nat, Bool, x => Nat.is_le(1n+rest, x), Nat.add(1n+rest, Nat.mul(q, 1n+rest)), count, exact)),
          past)))

def size_decided(+count: Nat, +extents: List<&2, Signed.Extent>, +rest: Nat, +quotient: Nat,
  exact: {Nat.mul(quotient, 1n+rest) == count : Nat}, decision: Shape.BoolCase<Nat.is_le(1n+rest, count)>) ->
  {Signed.inferred_one(count, extents, Laws.classified(1n+rest, Nat.is_le(1n+rest, count))) ==
    Done{Signed.filled(extents, U32.from_nat(quotient))} : Result<&2, &2, Signed.Error, List<&2, U32>>}:
  match decision:
    case Shape.BTrue{equal}:
      %equal : {Signed.inferred_one(count, extents, Laws.classified(1n+rest, _)) == Done{Signed.filled(extents, U32.from_nat(quotient))} : Result<&2, &2, Signed.Error, List<&2, U32>>}
      %Equal.sym(Nat & Nat, Nat.divmod(count, 1n+rest), (quotient, 0n), divides(count, rest, quotient, exact)) :
        {Signed.divided(extents, _) == Done{Signed.filled(extents, U32.from_nat(quotient))} : Result<&2, &2, Signed.Error, List<&2, U32>>}
      {==}
    case Shape.BFalse{+equal}:
      %equal : {Signed.inferred_one(count, extents, Laws.classified(1n+rest, _)) == Done{Signed.filled(extents, U32.from_nat(quotient))} : Result<&2, &2, Signed.Error, List<&2, U32>>}
      size_beyond(count, extents, rest, quotient, exact, Equal.sym(Bool, False{}, Nat.is_le(1n+rest, count), equal))

def Laws.signed_inferred_size(count, extents, one, rest, product, quotient, exact):
  %Equal.sym(Nat, Signed.inferred_count(extents), 1n, one) :
    {Signed.inferred_from(count, extents, _) == Done{Signed.filled(extents, U32.from_nat(quotient))} : Result<&2, &2, Signed.Error, List<&2, U32>>}
  %Equal.sym(Signed.Product, Signed.capped(count, extents), Laws.classify(count, Signed.exact_product(extents)), capped_classifies(count, extents)) :
    {Signed.inferred_one(count, extents, _) == Done{Signed.filled(extents, U32.from_nat(quotient))} : Result<&2, &2, Signed.Error, List<&2, U32>>}
  %Equal.sym(Nat, Signed.exact_product(extents), 1n+rest, product) :
    {Signed.inferred_one(count, extents, Laws.classify(count, _)) == Done{Signed.filled(extents, U32.from_nat(quotient))} : Result<&2, &2, Signed.Error, List<&2, U32>>}
  size_decided(count, extents, rest, quotient, exact, Shape.bool_case(Nat.is_le(1n+rest, count)))

def pred(n: Nat) -> Nat:
  match n:
    case 0n:
      0n
    case 1n+p:
      p

def succ_not_zero(-x: Nat, equal: {1n+x == 0n : Nat}) -> Empty:
  Arith.clash(Equal.cong(Nat, Bool, n => Nat.is_eq(n, 0n), 1n+x, 0n, equal))

def zero_not_one(equal: {0n == 1n : Nat}) -> Empty:
  Arith.clash_true(Equal.cong(Nat, Bool, n => Nat.is_eq(n, 0n), 0n, 1n, equal))

# With no Inferred size, the filled sizes hold the exact product.
def filled_none(+extents: List<&2, Signed.Extent>, +size: U32, none: {Signed.inferred_count(extents) == 0n : Nat}) ->
  {Shape.logical_numel(Signed.filled(extents, size)) == Signed.exact_product(extents) : Nat}:
  match extents:
    case Nil{}:
      {==}
    case Signed.Exact{+extent} <> +tail:
      Equal.cong(Nat, Nat, x => Nat.mul(U32.to_nat(extent), x), Shape.logical_numel(Signed.filled(tail, size)), Signed.exact_product(tail),
        filled_none(tail, size, none))
    case Signed.Inferred{} <> +tail:
      Empty.absurd({Shape.logical_numel(Signed.filled(Signed.Inferred{} <> tail, size)) == Signed.exact_product(Signed.Inferred{} <> tail) : Nat},
        succ_not_zero(Signed.inferred_count(tail), none))

# e * (s * p) = s * (e * p).
def mul_swap(+e: Nat, +s: Nat, +p: Nat) -> {Nat.mul(e, Nat.mul(s, p)) == Nat.mul(s, Nat.mul(e, p)) : Nat}:
  Equal.trans(Nat, Nat.mul(e, Nat.mul(s, p)), Nat.mul(Nat.mul(e, s), p), Nat.mul(s, Nat.mul(e, p)),
    Equal.sym(Nat, Nat.mul(Nat.mul(e, s), p), Nat.mul(e, Nat.mul(s, p)), NatProofs.mul_assoc(e, s, p)),
    Equal.trans(Nat, Nat.mul(Nat.mul(e, s), p), Nat.mul(Nat.mul(s, e), p), Nat.mul(s, Nat.mul(e, p)),
      Equal.cong(Nat, Nat, x => Nat.mul(x, p), Nat.mul(e, s), Nat.mul(s, e), NatProofs.mul_comm(e, s)),
      NatProofs.mul_assoc(s, e, p)))

def filled_one(+extents: List<&2, Signed.Extent>, one: {Signed.inferred_count(extents) == 1n : Nat}, +size: U32) ->
  {Shape.logical_numel(Signed.filled(extents, size)) == Nat.mul(U32.to_nat(size), Signed.exact_product(extents)) : Nat}:
  match extents:
    case Nil{}:
      Empty.absurd({Shape.logical_numel(Signed.filled([], size)) == Nat.mul(U32.to_nat(size), Signed.exact_product([])) : Nat},
        zero_not_one(one))
    case Signed.Exact{+extent} <> +tail:
      Equal.trans(Nat, Nat.mul(U32.to_nat(extent), Shape.logical_numel(Signed.filled(tail, size))),
        Nat.mul(U32.to_nat(extent), Nat.mul(U32.to_nat(size), Signed.exact_product(tail))),
        Nat.mul(U32.to_nat(size), Nat.mul(U32.to_nat(extent), Signed.exact_product(tail))),
        Equal.cong(Nat, Nat, x => Nat.mul(U32.to_nat(extent), x), Shape.logical_numel(Signed.filled(tail, size)),
          Nat.mul(U32.to_nat(size), Signed.exact_product(tail)), filled_one(tail, one, size)),
        mul_swap(U32.to_nat(extent), U32.to_nat(size), Signed.exact_product(tail)))
    case Signed.Inferred{} <> +tail:
      Equal.cong(Nat, Nat, x => Nat.mul(U32.to_nat(size), x), Shape.logical_numel(Signed.filled(tail, size)), Signed.exact_product(tail),
        filled_none(tail, size, Equal.cong(Nat, Nat, pred, 1n+Signed.inferred_count(tail), 1n, one)))

def Laws.signed_filled_count(extents, one, size):
  filled_one(extents, one, size)

# divmod(0, p) is (0, 0): a remainder means a nonempty input.
def indivisible_beyond(+count: Nat, +extents: List<&2, Signed.Extent>, +rest: Nat, +quotient: Nat, +remainder: Nat,
  split: {Nat.divmod(count, 1n+rest) == (quotient, 1n+remainder) : Nat & Nat}) ->
  {Signed.beyond(count, extents) == Fail{Signed.count_mismatch()} : Result<&2, &2, Signed.Error, List<&2, U32>>}:
  match count:
    case 0n:
      Empty.absurd({Signed.beyond(0n, extents) == Fail{Signed.count_mismatch()} : Result<&2, &2, Signed.Error, List<&2, U32>>},
        succ_not_zero(remainder, Equal.sym(Nat, 0n, 1n+remainder,
          Equal.cong(Nat & Nat, Nat, pair => Pair.snd(Nat, Nat, pair), (0n, 0n), (quotient, 1n+remainder), split))))
    case 1n+c:
      {==}

def indivisible_decided(+count: Nat, +extents: List<&2, Signed.Extent>, +rest: Nat, +quotient: Nat, +remainder: Nat,
  split: {Nat.divmod(count, 1n+rest) == (quotient, 1n+remainder) : Nat & Nat}, decision: Shape.BoolCase<Nat.is_le(1n+rest, count)>) ->
  {Signed.inferred_one(count, extents, Laws.classified(1n+rest, Nat.is_le(1n+rest, count))) == Fail{Signed.count_mismatch()} : Result<&2, &2, Signed.Error, List<&2, U32>>}:
  match decision:
    case Shape.BTrue{equal}:
      %equal : {Signed.inferred_one(count, extents, Laws.classified(1n+rest, _)) == Fail{Signed.count_mismatch()} : Result<&2, &2, Signed.Error, List<&2, U32>>}
      %Equal.sym(Nat & Nat, Nat.divmod(count, 1n+rest), (quotient, 1n+remainder), split) :
        {Signed.divided(extents, _) == Fail{Signed.count_mismatch()} : Result<&2, &2, Signed.Error, List<&2, U32>>}
      {==}
    case Shape.BFalse{equal}:
      %equal : {Signed.inferred_one(count, extents, Laws.classified(1n+rest, _)) == Fail{Signed.count_mismatch()} : Result<&2, &2, Signed.Error, List<&2, U32>>}
      indivisible_beyond(count, extents, rest, quotient, remainder, split)

def Laws.signed_inferred_indivisible(count, extents, one, rest, product, quotient, remainder, split):
  %Equal.sym(Nat, Signed.inferred_count(extents), 1n, one) :
    {Signed.inferred_from(count, extents, _) == Fail{Signed.count_mismatch()} : Result<&2, &2, Signed.Error, List<&2, U32>>}
  %Equal.sym(Signed.Product, Signed.capped(count, extents), Laws.classify(count, Signed.exact_product(extents)), capped_classifies(count, extents)) :
    {Signed.inferred_one(count, extents, _) == Fail{Signed.count_mismatch()} : Result<&2, &2, Signed.Error, List<&2, U32>>}
  %Equal.sym(Nat, Signed.exact_product(extents), 1n+rest, product) :
    {Signed.inferred_one(count, extents, Laws.classify(count, _)) == Fail{Signed.count_mismatch()} : Result<&2, &2, Signed.Error, List<&2, U32>>}
  indivisible_decided(count, extents, rest, quotient, remainder, split, Shape.bool_case(Nat.is_le(1n+rest, count)))

def Laws.signed_inferred_ambiguous(extents, one, product):
  %Equal.sym(Nat, Signed.inferred_count(extents), 1n, one) :
    {Signed.inferred_from(0n, extents, _) == Fail{Signed.AmbiguousInferred{}} : Result<&2, &2, Signed.Error, List<&2, U32>>}
  %Equal.sym(Signed.Product, Signed.capped(0n, extents), Laws.classify(0n, Signed.exact_product(extents)), capped_classifies(0n, extents)) :
    {Signed.inferred_one(0n, extents, _) == Fail{Signed.AmbiguousInferred{}} : Result<&2, &2, Signed.Error, List<&2, U32>>}
  %Equal.sym(Nat, Signed.exact_product(extents), 0n, product) :
    {Signed.inferred_one(0n, extents, Laws.classify(0n, _)) == Fail{Signed.AmbiguousInferred{}} : Result<&2, &2, Signed.Error, List<&2, U32>>}
  {==}

def Laws.signed_inferred_zero_product(count, extents, one, product):
  %Equal.sym(Nat, Signed.inferred_count(extents), 1n, one) :
    {Signed.inferred_from(1n+count, extents, _) == Fail{Signed.count_mismatch()} : Result<&2, &2, Signed.Error, List<&2, U32>>}
  %Equal.sym(Signed.Product, Signed.capped(1n+count, extents), Laws.classify(1n+count, Signed.exact_product(extents)), capped_classifies(1n+count, extents)) :
    {Signed.inferred_one(1n+count, extents, _) == Fail{Signed.count_mismatch()} : Result<&2, &2, Signed.Error, List<&2, U32>>}
  %Equal.sym(Nat, Signed.exact_product(extents), 0n, product) :
    {Signed.inferred_one(1n+count, extents, Laws.classify(1n+count, _)) == Fail{Signed.count_mismatch()} : Result<&2, &2, Signed.Error, List<&2, U32>>}
  {==}

def Laws.signed_inferred_repeated(count, extents, more, repeated):
  %Equal.sym(Nat, Signed.inferred_count(extents), 2n+more, repeated) :
    {Signed.inferred_from(count, extents, _) == Fail{Signed.RepeatedInferred{}} : Result<&2, &2, Signed.Error, List<&2, U32>>}
  {==}

def Laws.signed_filled_exact(extent, tail, size):
  {==}

def Laws.signed_filled_inferred(tail, size):
  {==}

def Laws.signed_error_codes(error):
  {==}
