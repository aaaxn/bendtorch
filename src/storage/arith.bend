import Base
import ./model.bend as Model

def falsity(value: Bool) -> Data:
  match value:
    case True{}:
      Empty
    case False{}:
      Unit

def clash(equal: {False{} == True{} : Bool}) -> Empty:
  %equal : falsity(_)
  Unit{}

def bit(b: Bool, x: Nat) -> Nat:
  match b:
    case False{}:
      Nat.double(x)
    case True{}:
      1n+Nat.double(x)

def borrow(c: Bool) -> Nat:
  match c:
    case False{}:
      1n
    case True{}:
      0n

def add_succ(a: Nat, -b: Nat) -> {Nat.add(a, 1n+b) == 1n+Nat.add(a, b) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_succ(p, b) : {1n+Nat.add(p, 1n+b) == 1n+_ : Nat}
      {==}

def double_add(a: Nat, -b: Nat) -> {Nat.add(Nat.double(a), Nat.double(b)) == Nat.double(Nat.add(a, b)) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %double_add(p, b) : {2n+Nat.add(Nat.double(p), Nat.double(b)) == 2n+_ : Nat}
      {==}

def double_self(+a: Nat) -> {Nat.add(a, a) == Nat.double(a) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %double_self(p) : {1n+Nat.add(p, 1n+p) == 2n+_ : Nat}
      %add_succ(p, p) : {1n+Nat.add(p, 1n+p) == 1n+_ : Nat}
      {==}

def eq_refl(+a: Nat) -> {Nat.is_eq(a, a) == True{} : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      eq_refl(p)

def eq_true(+a: Nat, +b: Nat, decided: {Nat.is_eq(a, b) == True{} : Bool}) -> {a == b : Nat}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      Empty.absurd({0n == 1n+q : Nat}, clash(decided))
    case 1n+p 0n:
      Empty.absurd({1n+p == 0n : Nat}, clash(decided))
    case 1n+p 1n+q:
      %eq_true(p, q, decided) : {1n+p == 1n+_ : Nat}
      {==}

def cmp_bits(+x: Nat, +y: Nat, +a: Bool, +b: Bool) ->
  {Word.cmp.fin(a, b, Nat.cmp(x, y)) == Nat.cmp(bit(a, x), bit(b, y)) : Cmp}:
  match x y:
    case 0n 0n:
      match a b:
        case False{} False{}:
          {==}
        case False{} True{}:
          {==}
        case True{} False{}:
          {==}
        case True{} True{}:
          {==}
    case 0n 1n+q:
      match a b:
        case False{} False{}:
          {==}
        case False{} True{}:
          {==}
        case True{} False{}:
          {==}
        case True{} True{}:
          {==}
    case 1n+p 0n:
      match a b:
        case False{} False{}:
          {==}
        case False{} True{}:
          {==}
        case True{} False{}:
          {==}
        case True{} True{}:
          {==}
    case 1n+p 1n+q:
      match a b:
        case False{} False{}:
          cmp_bits(p, q, False{}, False{})
        case False{} True{}:
          cmp_bits(p, q, False{}, True{})
        case True{} False{}:
          cmp_bits(p, q, True{}, False{})
        case True{} True{}:
          cmp_bits(p, q, True{}, True{})

def word_bit(-p: Nat, +b: Bool, -t: Word(p)) -> {bit(b, Word.to_nat(p, t)) == Word.to_nat(1n+p, WCon{b, t}) : Nat}:
  match b:
    case False{}:
      {==}
    case True{}:
      {==}

def word_cmp(+n: Nat, +a: Word(n), +b: Word(n)) ->
  {Nat.cmp(Word.to_nat(n, a), Word.to_nat(n, b)) == Word.cmp(n, a, b) : Cmp}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      match a b:
        case WCon{+x, +xs} WCon{+y, +ys}:
          %word_bit(p, x, xs) : {Nat.cmp(_, Word.to_nat(1n+p, WCon{y, ys})) == Word.cmp.fin(x, y, Word.cmp(p, xs, ys)) : Cmp}
          %word_bit(p, y, ys) : {Nat.cmp(bit(x, Word.to_nat(p, xs)), _) == Word.cmp.fin(x, y, Word.cmp(p, xs, ys)) : Cmp}
          %cmp_bits(Word.to_nat(p, xs), Word.to_nat(p, ys), x, y) : {_ == Word.cmp.fin(x, y, Word.cmp(p, xs, ys)) : Cmp}
          %word_cmp(p, xs, ys) : {Word.cmp.fin(x, y, Nat.cmp(Word.to_nat(p, xs), Word.to_nat(p, ys))) == Word.cmp.fin(x, y, _) : Cmp}
          {==}

def u32_lt(+a: U32, +b: U32) -> {Nat.is_lt(U32.to_nat(a), U32.to_nat(b)) == U32.is_lt(a, b) : Bool}:
  match a b:
    case U32{x} U32{y}:
      %word_cmp(32n, x, y) : {Cmp.is_lt(Nat.cmp(Word.to_nat(32n, x), Word.to_nat(32n, y))) == Cmp.is_lt(_) : Bool}
      {==}

def u32_le(+a: U32, +b: U32) -> {Nat.is_le(U32.to_nat(a), U32.to_nat(b)) == U32.is_le(a, b) : Bool}:
  match a b:
    case U32{x} U32{y}:
      %word_cmp(32n, x, y) : {Cmp.is_le(Nat.cmp(Word.to_nat(32n, x), Word.to_nat(32n, y))) == Cmp.is_le(_) : Bool}
      {==}

def le_zero(+q: Nat) -> {Nat.is_le(0n, q) == True{} : Bool}:
  match q:
    case 0n:
      {==}
    case 1n+r:
      {==}

def le_dd(+u: Nat, +v: Nat, h: {Nat.is_le(Nat.double(u), Nat.double(v)) == True{} : Bool}) -> {Nat.is_le(u, v) == True{} : Bool}:
  match u v:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      Empty.absurd({Nat.is_le(1n+p, 0n) == True{} : Bool}, clash(h))
    case 1n+p 1n+q:
      le_dd(p, q, h)

def le_sd(+u: Nat, +v: Nat, h: {Nat.is_le(1n+Nat.double(u), Nat.double(v)) == True{} : Bool}) -> {Nat.is_le(1n+u, v) == True{} : Bool}:
  match u v:
    case 0n 0n:
      Empty.absurd({Nat.is_le(1n, 0n) == True{} : Bool}, clash(h))
    case 0n 1n+q:
      le_zero(q)
    case 1n+p 0n:
      Empty.absurd({Nat.is_le(2n+p, 0n) == True{} : Bool}, clash(h))
    case 1n+p 1n+q:
      le_sd(p, q, h)

def le_ds(+u: Nat, +v: Nat, h: {Nat.is_le(Nat.double(u), 1n+Nat.double(v)) == True{} : Bool}) -> {Nat.is_le(u, v) == True{} : Bool}:
  match u v:
    case 0n v:
      le_zero(v)
    case 1n+p 0n:
      Empty.absurd({Nat.is_le(1n+p, 0n) == True{} : Bool}, clash(h))
    case 1n+p 1n+q:
      le_ds(p, q, h)

# Word.sub(a, b) is Word.adc(a, b, True, True): it adds the complement of b
# with an incoming carry, so a cleared carry is a borrow of one.
def sub_word(+n: Nat, +a: Word(n), +b: Word(n), +c: Bool,
  h: {Nat.is_le(Nat.add(borrow(c), Word.to_nat(n, b)), Word.to_nat(n, a)) == True{} : Bool}) ->
  {Nat.add(borrow(c), Nat.add(Word.to_nat(n, b), Word.to_nat(n, Word.adc(n, a, b, True{}, c)))) == Word.to_nat(n, a) : Nat}:
  match n:
    case 0n:
      match c:
        case False{}:
          Empty.absurd({1n == 0n : Nat}, clash(h))
        case True{}:
          {==}
    case 1n+p:
      match a:
        case WCon{+x, +xs}:
          match x:
            case False{}:
              match b:
                case WCon{+y, +ys}:
                  match y:
                    case False{}:
                      match c:
                        case False{}:
                          %sub_word(p, xs, ys, False{}, le_sd(Word.to_nat(p, ys), Word.to_nat(p, xs), h)) : {1n+Nat.add(Nat.double(Word.to_nat(p, ys)), 1n+Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{})))) == Nat.double(_) : Nat}
                          %double_add(Word.to_nat(p, ys), Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{}))) : {1n+Nat.add(Nat.double(Word.to_nat(p, ys)), 1n+Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{})))) == 2n+_ : Nat}
                          %add_succ(Nat.double(Word.to_nat(p, ys)), Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{})))) : {1n+Nat.add(Nat.double(Word.to_nat(p, ys)), 1n+Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{})))) == 1n+_ : Nat}
                          {==}
                        case True{}:
                          %sub_word(p, xs, ys, True{}, le_dd(Word.to_nat(p, ys), Word.to_nat(p, xs), h)) : {Nat.add(Nat.double(Word.to_nat(p, ys)), Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, True{})))) == Nat.double(_) : Nat}
                          %double_add(Word.to_nat(p, ys), Word.to_nat(p, Word.adc(p, xs, ys, True{}, True{}))) : {Nat.add(Nat.double(Word.to_nat(p, ys)), Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, True{})))) == _ : Nat}
                          {==}
                    case True{}:
                      match c:
                        case False{}:
                          %sub_word(p, xs, ys, False{}, le_dd(1n+Word.to_nat(p, ys), Word.to_nat(p, xs), h)) : {2n+Nat.add(Nat.double(Word.to_nat(p, ys)), Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{})))) == Nat.double(_) : Nat}
                          %double_add(Word.to_nat(p, ys), Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{}))) : {2n+Nat.add(Nat.double(Word.to_nat(p, ys)), Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{})))) == 2n+_ : Nat}
                          {==}
                        case True{}:
                          %sub_word(p, xs, ys, False{}, le_sd(Word.to_nat(p, ys), Word.to_nat(p, xs), h)) : {1n+Nat.add(Nat.double(Word.to_nat(p, ys)), 1n+Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{})))) == Nat.double(_) : Nat}
                          %double_add(Word.to_nat(p, ys), Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{}))) : {1n+Nat.add(Nat.double(Word.to_nat(p, ys)), 1n+Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{})))) == 2n+_ : Nat}
                          %add_succ(Nat.double(Word.to_nat(p, ys)), Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{})))) : {1n+Nat.add(Nat.double(Word.to_nat(p, ys)), 1n+Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{})))) == 1n+_ : Nat}
                          {==}
            case True{}:
              match b:
                case WCon{+y, +ys}:
                  match y:
                    case False{}:
                      match c:
                        case False{}:
                          %sub_word(p, xs, ys, True{}, le_dd(Word.to_nat(p, ys), Word.to_nat(p, xs), h)) : {1n+Nat.add(Nat.double(Word.to_nat(p, ys)), Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, True{})))) == 1n+Nat.double(_) : Nat}
                          %double_add(Word.to_nat(p, ys), Word.to_nat(p, Word.adc(p, xs, ys, True{}, True{}))) : {1n+Nat.add(Nat.double(Word.to_nat(p, ys)), Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, True{})))) == 1n+_ : Nat}
                          {==}
                        case True{}:
                          %sub_word(p, xs, ys, True{}, le_ds(Word.to_nat(p, ys), Word.to_nat(p, xs), h)) : {Nat.add(Nat.double(Word.to_nat(p, ys)), 1n+Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, True{})))) == 1n+Nat.double(_) : Nat}
                          %double_add(Word.to_nat(p, ys), Word.to_nat(p, Word.adc(p, xs, ys, True{}, True{}))) : {Nat.add(Nat.double(Word.to_nat(p, ys)), 1n+Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, True{})))) == 1n+_ : Nat}
                          %add_succ(Nat.double(Word.to_nat(p, ys)), Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, True{})))) : {Nat.add(Nat.double(Word.to_nat(p, ys)), 1n+Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, True{})))) == _ : Nat}
                          {==}
                    case True{}:
                      match c:
                        case False{}:
                          %sub_word(p, xs, ys, False{}, le_sd(Word.to_nat(p, ys), Word.to_nat(p, xs), h)) : {2n+Nat.add(Nat.double(Word.to_nat(p, ys)), 1n+Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{})))) == 1n+Nat.double(_) : Nat}
                          %double_add(Word.to_nat(p, ys), Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{}))) : {2n+Nat.add(Nat.double(Word.to_nat(p, ys)), 1n+Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{})))) == 3n+_ : Nat}
                          %add_succ(Nat.double(Word.to_nat(p, ys)), Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{})))) : {2n+Nat.add(Nat.double(Word.to_nat(p, ys)), 1n+Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, False{})))) == 2n+_ : Nat}
                          {==}
                        case True{}:
                          %sub_word(p, xs, ys, True{}, le_dd(Word.to_nat(p, ys), Word.to_nat(p, xs), h)) : {1n+Nat.add(Nat.double(Word.to_nat(p, ys)), Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, True{})))) == 1n+Nat.double(_) : Nat}
                          %double_add(Word.to_nat(p, ys), Word.to_nat(p, Word.adc(p, xs, ys, True{}, True{}))) : {1n+Nat.add(Nat.double(Word.to_nat(p, ys)), Nat.double(Word.to_nat(p, Word.adc(p, xs, ys, True{}, True{})))) == 1n+_ : Nat}
                          {==}

def lt_le_trans(+a: Nat, +b: Nat, +c: Nat, ab: {Nat.is_lt(a, b) == True{} : Bool},
  bc: {Nat.is_le(b, c) == True{} : Bool}) -> {Nat.is_lt(a, c) == True{} : Bool}:
  match a:
    case 0n:
      match b:
        case 0n:
          Empty.absurd({Nat.is_lt(0n, c) == True{} : Bool}, clash(ab))
        case 1n+q:
          match c:
            case 0n:
              Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool}, clash(bc))
            case 1n+r:
              {==}
    case 1n+p:
      match b:
        case 0n:
          Empty.absurd({Nat.is_lt(1n+p, c) == True{} : Bool}, clash(ab))
        case 1n+q:
          match c:
            case 0n:
              Empty.absurd({Nat.is_lt(1n+p, 0n) == True{} : Bool}, clash(bc))
            case 1n+r:
              lt_le_trans(p, q, r, ab, bc)

def lt_succ(+a: Nat, +b: Nat, h: {Nat.is_le(a, b) == True{} : Bool}) -> {Nat.is_lt(a, 1n+b) == True{} : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      match b:
        case 0n:
          Empty.absurd({Nat.is_lt(1n+p, 1n) == True{} : Bool}, clash(h))
        case 1n+q:
          lt_succ(p, q, h)

def lt_dd(+x: Nat, +c: Nat, h: {Nat.is_lt(Nat.double(x), Nat.double(c)) == True{} : Bool}) -> {Nat.is_lt(x, c) == True{} : Bool}:
  match x c:
    case 0n 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool}, clash(h))
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      Empty.absurd({Nat.is_lt(1n+p, 0n) == True{} : Bool}, clash(h))
    case 1n+p 1n+q:
      lt_dd(p, q, h)

def lt_sd(+x: Nat, +c: Nat, h: {Nat.is_lt(1n+Nat.double(x), Nat.double(c)) == True{} : Bool}) -> {Nat.is_lt(x, c) == True{} : Bool}:
  match x c:
    case 0n 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool}, clash(h))
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      Empty.absurd({Nat.is_lt(1n+p, 0n) == True{} : Bool}, clash(h))
    case 1n+p 1n+q:
      lt_sd(p, q, h)

def lt_doubled(+x: Nat, +c: Nat, h: {Nat.is_lt(x, Nat.add(c, c)) == True{} : Bool}) -> {Nat.is_lt(x, Nat.double(c)) == True{} : Bool}:
  %double_self(c) : {Nat.is_lt(x, _) == True{} : Bool}
  h

def lt_half(+b: Bool, +x: Nat, +c: Nat, h: {Nat.is_lt(bit(b, x), Model.capacity(1n+c)) == True{} : Bool}) ->
  {Nat.is_lt(x, Model.capacity(c)) == True{} : Bool}:
  match b:
    case False{}:
      lt_dd(x, Model.capacity(c), lt_doubled(Nat.double(x), Model.capacity(c), h))
    case True{}:
      lt_sd(x, Model.capacity(c), lt_doubled(1n+Nat.double(x), Model.capacity(c), h))

def lt_zero(+x: Nat, h: {Nat.is_lt(x, 0n) == True{} : Bool}) -> Empty:
  match x:
    case 0n:
      clash(h)
    case 1n+p:
      clash(h)

def lt_one(+x: Nat, h: {Nat.is_lt(x, 1n) == True{} : Bool}) -> {0n == x : Nat}:
  match x:
    case 0n:
      {==}
    case 1n+p:
      Empty.absurd({0n == 1n+p : Nat}, lt_zero(p, h))

def lt_d1(+x: Nat, h: {Nat.is_lt(Nat.double(x), 1n) == True{} : Bool}) -> {Nat.is_lt(x, 1n) == True{} : Bool}:
  match x:
    case 0n:
      {==}
    case 1n+p:
      Empty.absurd({Nat.is_lt(1n+p, 1n) == True{} : Bool}, clash(h))

def low(m: Nat, +k: Nat) -> Word(m):
  match m:
    case 0n:
      WNil{}
    case 1n+q:
      match k:
        case 0n:
          WCon{False{}, low(q, 0n)}
        case 1n+r:
          WCon{True{}, low(q, r)}

def mask_word(+m: Nat, +k: Nat, +w: Word(m), h: {Nat.is_lt(Word.to_nat(m, w), Model.capacity(k)) == True{} : Bool}) ->
  {w == Word.and(m, w, low(m, k)) : Word(m)}:
  match m:
    case 0n:
      match w:
        case WNil{}:
          {==}
    case 1n+q:
      match k:
        case 0n:
          match w:
            case WCon{+b, +t}:
              match b:
                case False{}:
                  %mask_word(q, 0n, t, lt_d1(Word.to_nat(q, t), h)) : {WCon{False{}, t} == WCon{False{}, _} : Word(1n+q)}
                  {==}
                case True{}:
                  Empty.absurd({WCon{True{}, t} == Word.and(1n+q, WCon{True{}, t}, low(1n+q, 0n)) : Word(1n+q)}, lt_zero(Nat.double(Word.to_nat(q, t)), h))
        case 1n+r:
          match w:
            case WCon{+b, +t}:
              match b:
                case False{}:
                  %mask_word(q, r, t, lt_half(False{}, Word.to_nat(q, t), r, h)) : {WCon{False{}, t} == WCon{False{}, _} : Word(1n+q)}
                  {==}
                case True{}:
                  %mask_word(q, r, t, lt_half(True{}, Word.to_nat(q, t), r, h)) : {WCon{True{}, t} == WCon{True{}, _} : Word(1n+q)}
                  {==}

def unit(m: Nat, +k: Nat) -> Word(m):
  match m:
    case 0n:
      WNil{}
    case 1n+q:
      match k:
        case 0n:
          WCon{True{}, Word.zero(q)}
        case 1n+r:
          WCon{False{}, unit(q, r)}

def zero_nat(+q: Nat) -> {0n == Word.to_nat(q, Word.zero(q)) : Nat}:
  match q:
    case 0n:
      {==}
    case 1n+p:
      %zero_nat(p) : {0n == Nat.double(_) : Nat}
      {==}

def unit_nat(+m: Nat, +k: Nat, h: {Nat.is_lt(k, m) == True{} : Bool}) -> {Model.capacity(k) == Word.to_nat(m, unit(m, k)) : Nat}:
  match m:
    case 0n:
      match k:
        case 0n:
          Empty.absurd({1n == 0n : Nat}, clash(h))
        case 1n+r:
          Empty.absurd({Model.capacity(1n+r) == 0n : Nat}, clash(h))
    case 1n+q:
      match k:
        case 0n:
          %zero_nat(q) : {1n == 1n+Nat.double(_) : Nat}
          {==}
        case 1n+r:
          %unit_nat(q, r, h) : {Model.capacity(1n+r) == Nat.double(_) : Nat}
          double_self(Model.capacity(r))

def truth(value: Bool) -> Data:
  match value:
    case True{}:
      Unit
    case False{}:
      Empty

def clash_true(equal: {True{} == False{} : Bool}) -> Empty:
  %equal : truth(_)
  Unit{}

def le_succ(+a: Nat, +b: Nat, h: {Nat.is_le(a, b) == True{} : Bool}) -> {Nat.is_le(a, 1n+b) == True{} : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      match b:
        case 0n:
          Empty.absurd({Nat.is_le(1n+p, 1n) == True{} : Bool}, clash(h))
        case 1n+q:
          le_succ(p, q, h)

def not_lt_le(+a: Nat, +b: Nat, h: {Nat.is_lt(a, b) == False{} : Bool}) -> {Nat.is_le(b, a) == True{} : Bool}:
  match a:
    case 0n:
      match b:
        case 0n:
          {==}
        case 1n+q:
          Empty.absurd({Nat.is_le(1n+q, 0n) == True{} : Bool}, clash_true(h))
    case 1n+p:
      match b:
        case 0n:
          {==}
        case 1n+q:
          not_lt_le(p, q, h)

def lt_add_cancel(+c: Nat, +j: Nat, +k: Nat, h: {Nat.is_lt(Nat.add(c, j), Nat.add(c, k)) == True{} : Bool}) ->
  {Nat.is_lt(j, k) == True{} : Bool}:
  match c:
    case 0n:
      h
    case 1n+r:
      lt_add_cancel(r, j, k, h)

def and_left(+a: Bool, -b: Bool, h: {Bool.and(a, b) == True{} : Bool}) -> {a == True{} : Bool}:
  match a:
    case False{}:
      Empty.absurd({False{} == True{} : Bool}, clash(h))
    case True{}:
      {==}

def and_right(+a: Bool, -b: Bool, h: {Bool.and(a, b) == True{} : Bool}) -> {b == True{} : Bool}:
  match a:
    case False{}:
      Empty.absurd({b == True{} : Bool}, clash(h))
    case True{}:
      h

def sub_u32(+i: U32, +c: U32, h: {Nat.is_le(U32.to_nat(c), U32.to_nat(i)) == True{} : Bool}) ->
  {U32.to_nat(i) == Nat.add(U32.to_nat(c), U32.to_nat(U32.sub(i, c))) : Nat}:
  match i c:
    case U32{x} U32{y}:
      Equal.sym(Nat, Nat.add(Word.to_nat(32n, y), Word.to_nat(32n, Word.adc(32n, x, y, True{}, True{}))), Word.to_nat(32n, x),
        sub_word(32n, x, y, True{}, h))
