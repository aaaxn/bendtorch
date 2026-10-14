import Base
import ./lifecycle.bend as Lifecycle

def half(number: Nat) -> Nat:
  match number:
    case 0n:
      0n
    case 1n:
      0n
    case 2n+p:
      1n+half(p)

def even(number: Nat) -> Bool:
  match number:
    case 0n:
      True{}
    case 1n:
      False{}
    case 2n+p:
      even(p)

def half_double(+number: Nat) -> {half(Nat.double(number)) == number : Nat}:
  match number:
    case 0n:
      {==}
    case 1n+p:
      Equal.cong(Nat, Nat, n => 1n+n, half(Nat.double(p)), p, half_double(p))

def even_double(+number: Nat) -> {even(Nat.double(number)) == True{} : Bool}:
  match number:
    case 0n:
      {==}
    case 1n+p:
      even_double(p)

def even_successor_double(+number: Nat) -> {even(1n+Nat.double(number)) == False{} : Bool}:
  match number:
    case 0n:
      {==}
    case 1n+p:
      even_successor_double(p)

def double_injective(+left: Nat, +right: Nat, equal: {Nat.double(left) == Nat.double(right) : Nat}) -> {left == right : Nat}:
  Equal.trans(Nat, left, half(Nat.double(left)), right,
    Equal.sym(Nat, half(Nat.double(left)), left, half_double(left)),
    Equal.trans(Nat, half(Nat.double(left)), half(Nat.double(right)), right,
      Equal.cong(Nat, Nat, n => half(n), Nat.double(left), Nat.double(right), equal), half_double(right)))

def parity_clash(+left: Nat, +right: Nat, equal: {Nat.double(left) == 1n+Nat.double(right) : Nat}) -> Empty:
  Lifecycle.false_true(Equal.trans(Bool, True{}, even(Nat.double(left)), False{},
    Equal.sym(Bool, even(Nat.double(left)), True{}, even_double(left)),
    Equal.trans(Bool, even(Nat.double(left)), even(1n+Nat.double(right)), False{},
      Equal.cong(Nat, Bool, n => even(n), Nat.double(left), 1n+Nat.double(right), equal), even_successor_double(right))))

def predecessor(number: Nat) -> Nat:
  match number:
    case 0n:
      0n
    case 1n+p:
      p

def successor_injective(-left: Nat, -right: Nat, equal: {1n+left == 1n+right : Nat}) -> {left == right : Nat}:
  Equal.cong(Nat, Nat, n => predecessor(n), 1n+left, 1n+right, equal)

def word_natural_injective(+width: Nat, +left: Word(width), +right: Word(width)) ->
  {Word.to_nat(width, left) == Word.to_nat(width, right) : Nat} -> {left == right : Word(width)}:
  match width:
    case 0n:
      match left right:
        case WNil{} WNil{}:
          equal => {==}
    case 1n+p:
      match left right:
        case WCon{False{}, +lt} WCon{False{}, +rt}:
          equal => Equal.cong(Word(p), Word(1n+p), w => WCon{False{}, w}, lt, rt,
            word_natural_injective(p, lt, rt)(double_injective(Word.to_nat(p, lt), Word.to_nat(p, rt), equal)))
        case WCon{False{}, +lt} WCon{True{}, +rt}:
          equal => Empty.absurd({WCon{False{}, lt} == WCon{True{}, rt} : Word(1n+p)},
            parity_clash(Word.to_nat(p, lt), Word.to_nat(p, rt), equal))
        case WCon{True{}, +lt} WCon{False{}, +rt}:
          equal => Empty.absurd({WCon{True{}, lt} == WCon{False{}, rt} : Word(1n+p)},
            parity_clash(Word.to_nat(p, rt), Word.to_nat(p, lt),
              Equal.sym(Nat, 1n+Nat.double(Word.to_nat(p, lt)), Nat.double(Word.to_nat(p, rt)), equal)))
        case WCon{True{}, +lt} WCon{True{}, +rt}:
          equal => Equal.cong(Word(p), Word(1n+p), w => WCon{True{}, w}, lt, rt,
            word_natural_injective(p, lt, rt)(double_injective(Word.to_nat(p, lt), Word.to_nat(p, rt),
              successor_injective(Nat.double(Word.to_nat(p, lt)), Nat.double(Word.to_nat(p, rt)), equal))))

def u32_natural_injective(+left: U32, +right: U32) ->
  {U32.to_nat(left) == U32.to_nat(right) : Nat} -> {left == right : U32}:
  match left right:
    case U32{+lw} U32{+rw}:
      equal => Equal.cong(Word(32n), U32, w => U32{w}, lw, rw, word_natural_injective(32n, lw, rw)(equal))
