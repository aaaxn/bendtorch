import Base
import ./model.bend as Model
import ./size.bend as Size
import ./refinement.bend as Refinement
import ./main.bend as Storage
import ./arith.bend as Arith

def cap_unit(+d: Nat, +h: {Nat.is_le(d, 20n) == True{} : Bool}) -> {U32{Arith.unit(32n, d)} == Size.capacity(d) : U32}:
  match d:
    case 0n:
      {==}
    case 1n+d1:
      match d1:
        case 0n:
          {==}
        case 1n+d2:
          match d2:
            case 0n:
              {==}
            case 1n+d3:
              match d3:
                case 0n:
                  {==}
                case 1n+d4:
                  match d4:
                    case 0n:
                      {==}
                    case 1n+d5:
                      match d5:
                        case 0n:
                          {==}
                        case 1n+d6:
                          match d6:
                            case 0n:
                              {==}
                            case 1n+d7:
                              match d7:
                                case 0n:
                                  {==}
                                case 1n+d8:
                                  match d8:
                                    case 0n:
                                      {==}
                                    case 1n+d9:
                                      match d9:
                                        case 0n:
                                          {==}
                                        case 1n+d10:
                                          match d10:
                                            case 0n:
                                              {==}
                                            case 1n+d11:
                                              match d11:
                                                case 0n:
                                                  {==}
                                                case 1n+d12:
                                                  match d12:
                                                    case 0n:
                                                      {==}
                                                    case 1n+d13:
                                                      match d13:
                                                        case 0n:
                                                          {==}
                                                        case 1n+d14:
                                                          match d14:
                                                            case 0n:
                                                              {==}
                                                            case 1n+d15:
                                                              match d15:
                                                                case 0n:
                                                                  {==}
                                                                case 1n+d16:
                                                                  match d16:
                                                                    case 0n:
                                                                      {==}
                                                                    case 1n+d17:
                                                                      match d17:
                                                                        case 0n:
                                                                          {==}
                                                                        case 1n+d18:
                                                                          match d18:
                                                                            case 0n:
                                                                              {==}
                                                                            case 1n+d19:
                                                                              match d19:
                                                                                case 0n:
                                                                                  {==}
                                                                                case 1n+d20:
                                                                                  match d20:
                                                                                    case 0n:
                                                                                      {==}
                                                                                    case 1n+d21:
                                                                                      Empty.absurd({U32{Arith.unit(32n, 21n+d21)} == Size.capacity(21n+d21) : U32}, Arith.clash(h))

def shr_cap(+d: Nat, +h: {Nat.is_le(1n+d, 20n) == True{} : Bool}) -> {Size.capacity(d) == U32.shr(Size.capacity(1n+d)) : U32}:
  match d:
    case 0n:
      {==}
    case 1n+d1:
      match d1:
        case 0n:
          {==}
        case 1n+d2:
          match d2:
            case 0n:
              {==}
            case 1n+d3:
              match d3:
                case 0n:
                  {==}
                case 1n+d4:
                  match d4:
                    case 0n:
                      {==}
                    case 1n+d5:
                      match d5:
                        case 0n:
                          {==}
                        case 1n+d6:
                          match d6:
                            case 0n:
                              {==}
                            case 1n+d7:
                              match d7:
                                case 0n:
                                  {==}
                                case 1n+d8:
                                  match d8:
                                    case 0n:
                                      {==}
                                    case 1n+d9:
                                      match d9:
                                        case 0n:
                                          {==}
                                        case 1n+d10:
                                          match d10:
                                            case 0n:
                                              {==}
                                            case 1n+d11:
                                              match d11:
                                                case 0n:
                                                  {==}
                                                case 1n+d12:
                                                  match d12:
                                                    case 0n:
                                                      {==}
                                                    case 1n+d13:
                                                      match d13:
                                                        case 0n:
                                                          {==}
                                                        case 1n+d14:
                                                          match d14:
                                                            case 0n:
                                                              {==}
                                                            case 1n+d15:
                                                              match d15:
                                                                case 0n:
                                                                  {==}
                                                                case 1n+d16:
                                                                  match d16:
                                                                    case 0n:
                                                                      {==}
                                                                    case 1n+d17:
                                                                      match d17:
                                                                        case 0n:
                                                                          {==}
                                                                        case 1n+d18:
                                                                          match d18:
                                                                            case 0n:
                                                                              {==}
                                                                            case 1n+d19:
                                                                              match d19:
                                                                                case 0n:
                                                                                  {==}
                                                                                case 1n+d20:
                                                                                  Empty.absurd({Size.capacity(20n+d20) == U32.shr(Size.capacity(1n+20n+d20)) : U32}, Arith.clash(h))

def mask_cap(+d: Nat, +h: {Nat.is_le(d, 20n) == True{} : Bool}) -> {U32{Arith.low(32n, d)} == U32.sub(Size.capacity(d), 1) : U32}:
  match d:
    case 0n:
      {==}
    case 1n+d1:
      match d1:
        case 0n:
          {==}
        case 1n+d2:
          match d2:
            case 0n:
              {==}
            case 1n+d3:
              match d3:
                case 0n:
                  {==}
                case 1n+d4:
                  match d4:
                    case 0n:
                      {==}
                    case 1n+d5:
                      match d5:
                        case 0n:
                          {==}
                        case 1n+d6:
                          match d6:
                            case 0n:
                              {==}
                            case 1n+d7:
                              match d7:
                                case 0n:
                                  {==}
                                case 1n+d8:
                                  match d8:
                                    case 0n:
                                      {==}
                                    case 1n+d9:
                                      match d9:
                                        case 0n:
                                          {==}
                                        case 1n+d10:
                                          match d10:
                                            case 0n:
                                              {==}
                                            case 1n+d11:
                                              match d11:
                                                case 0n:
                                                  {==}
                                                case 1n+d12:
                                                  match d12:
                                                    case 0n:
                                                      {==}
                                                    case 1n+d13:
                                                      match d13:
                                                        case 0n:
                                                          {==}
                                                        case 1n+d14:
                                                          match d14:
                                                            case 0n:
                                                              {==}
                                                            case 1n+d15:
                                                              match d15:
                                                                case 0n:
                                                                  {==}
                                                                case 1n+d16:
                                                                  match d16:
                                                                    case 0n:
                                                                      {==}
                                                                    case 1n+d17:
                                                                      match d17:
                                                                        case 0n:
                                                                          {==}
                                                                        case 1n+d18:
                                                                          match d18:
                                                                            case 0n:
                                                                              {==}
                                                                            case 1n+d19:
                                                                              match d19:
                                                                                case 0n:
                                                                                  {==}
                                                                                case 1n+d20:
                                                                                  match d20:
                                                                                    case 0n:
                                                                                      {==}
                                                                                    case 1n+d21:
                                                                                      Empty.absurd({U32{Arith.low(32n, 21n+d21)} == U32.sub(Size.capacity(21n+d21), 1) : U32}, Arith.clash(h))

def cap_nat(+d: Nat, +h: {Nat.is_le(d, 20n) == True{} : Bool}) -> {Model.capacity(d) == U32.to_nat(Size.capacity(d)) : Nat}:
  %cap_unit(d, h) : {Model.capacity(d) == U32.to_nat(_) : Nat}
  Arith.unit_nat(32n, d, Arith.lt_le_trans(d, 21n, 32n, Arith.lt_succ(d, 20n, h), {==}))

def cap_word(+d: Nat, +h: {Nat.is_le(d, 20n) == True{} : Bool}) -> {U32.to_nat(Size.capacity(d)) == Model.capacity(d) : Nat}:
  Equal.sym(Nat, Model.capacity(d), U32.to_nat(Size.capacity(d)), cap_nat(d, h))

def low_inside(+p: Nat, +i: U32, +h: {Nat.is_le(1n+p, 20n) == True{} : Bool},
  +dz: {True{} == U32.is_lt(i, U32.shr(Size.capacity(1n+p))) : Bool}) ->
  {Nat.is_lt(U32.to_nat(i), Model.capacity(p)) == True{} : Bool}:
  +hp = Arith.le_succ(p, 19n, h)
  %cap_word(p, hp) : {Nat.is_lt(U32.to_nat(i), _) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_lt(U32.to_nat(i), U32.to_nat(Size.capacity(p))), U32.is_lt(i, Size.capacity(p)), Arith.u32_lt(i, Size.capacity(p))) : {_ == True{} : Bool}
  %Equal.sym(U32, Size.capacity(p), U32.shr(Size.capacity(1n+p)), shr_cap(p, h)) : {U32.is_lt(i, _) == True{} : Bool}
  Equal.sym(Bool, True{}, U32.is_lt(i, U32.shr(Size.capacity(1n+p))), dz)

def high_not_lt(+p: Nat, +i: U32, +h: {Nat.is_le(1n+p, 20n) == True{} : Bool},
  +dz: {False{} == U32.is_lt(i, U32.shr(Size.capacity(1n+p))) : Bool}) ->
  {Nat.is_lt(U32.to_nat(i), U32.to_nat(Size.capacity(p))) == False{} : Bool}:
  %Equal.sym(Bool, Nat.is_lt(U32.to_nat(i), U32.to_nat(Size.capacity(p))), U32.is_lt(i, Size.capacity(p)), Arith.u32_lt(i, Size.capacity(p))) : {_ == False{} : Bool}
  %Equal.sym(U32, Size.capacity(p), U32.shr(Size.capacity(1n+p)), shr_cap(p, h)) : {U32.is_lt(i, _) == False{} : Bool}
  Equal.sym(Bool, False{}, U32.is_lt(i, U32.shr(Size.capacity(1n+p))), dz)


def high_le(+p: Nat, +i: U32, +h: {Nat.is_le(1n+p, 20n) == True{} : Bool},
  +dz: {False{} == U32.is_lt(i, U32.shr(Size.capacity(1n+p))) : Bool}) ->
  {Nat.is_le(U32.to_nat(Size.capacity(p)), U32.to_nat(i)) == True{} : Bool}:
  Arith.not_lt_le(U32.to_nat(i), U32.to_nat(Size.capacity(p)), high_not_lt(p, i, h, dz))

def high_split(+p: Nat, +i: U32, +hp: {Nat.is_le(p, 20n) == True{} : Bool},
  +le: {Nat.is_le(U32.to_nat(Size.capacity(p)), U32.to_nat(i)) == True{} : Bool}) ->
  {U32.to_nat(i) == Nat.add(Model.capacity(p), U32.to_nat(U32.sub(i, Size.capacity(p)))) : Nat}:
  %cap_word(p, hp) : {U32.to_nat(i) == Nat.add(_, U32.to_nat(U32.sub(i, Size.capacity(p)))) : Nat}
  Arith.sub_u32(i, Size.capacity(p), le)

def high_sum(+p: Nat, +i: U32, +hp: {Nat.is_le(p, 20n) == True{} : Bool},
  +le: {Nat.is_le(U32.to_nat(Size.capacity(p)), U32.to_nat(i)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(i), Model.capacity(1n+p)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Model.capacity(p), U32.to_nat(U32.sub(i, Size.capacity(p)))), Nat.add(Model.capacity(p), Model.capacity(p))) == True{} : Bool}:
  %high_split(p, i, hp, le) : {Nat.is_lt(_, Nat.add(Model.capacity(p), Model.capacity(p))) == True{} : Bool}
  inside

def high_inside(+p: Nat, +i: U32, +hp: {Nat.is_le(p, 20n) == True{} : Bool},
  +le: {Nat.is_le(U32.to_nat(Size.capacity(p)), U32.to_nat(i)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(i), Model.capacity(1n+p)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.sub(i, Size.capacity(p))), Model.capacity(p)) == True{} : Bool}:
  Arith.lt_add_cancel(Model.capacity(p), U32.to_nat(U32.sub(i, Size.capacity(p))), Model.capacity(p),
    high_sum(p, i, hp, le, inside))

def trace_offset(+d: Nat, +i: U32, z: Bool, +dz: {z == U32.is_lt(i, U32.shr(Size.capacity(d))) : Bool},
  +h: {Nat.is_le(d, 20n) == True{} : Bool}, +inside: {Nat.is_lt(U32.to_nat(i), Model.capacity(d)) == True{} : Bool}) ->
  {U32.to_nat(i) == Model.offset(d, Refinement.trace(d, Size.capacity(d), i, z)) : Nat}:
  match d:
    case 0n:
      Equal.sym(Nat, 0n, U32.to_nat(i), Arith.lt_one(U32.to_nat(i), inside))
    case 1n+p:
      match z:
        case True{}:
          +hp = Arith.le_succ(p, 19n, h)
          %shr_cap(p, h) : {U32.to_nat(i) == Model.offset(p, Refinement.trace(p, _, i, U32.is_lt(i, U32.shr(_)))) : Nat}
          trace_offset(p, i, U32.is_lt(i, U32.shr(Size.capacity(p))), {==}, hp, low_inside(p, i, h, dz))
        case False{}:
          +hp = Arith.le_succ(p, 19n, h)
          +le = high_le(p, i, h, dz)
          %shr_cap(p, h) : {U32.to_nat(i) == Nat.add(Model.capacity(p), Model.offset(p, Refinement.trace(p, _, U32.sub(i, _), U32.is_lt(U32.sub(i, _), U32.shr(_))))) : Nat}
          +j = U32.sub(i, Size.capacity(p))
          %trace_offset(p, j, U32.is_lt(j, U32.shr(Size.capacity(p))), {==}, hp, high_inside(p, i, hp, le, inside)) : {U32.to_nat(i) == Nat.add(Model.capacity(p), _) : Nat}
          high_split(p, i, hp, le)

def mask_u32(+d: Nat, +i: U32, +h: {Nat.is_le(d, 20n) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(i), Model.capacity(d)) == True{} : Bool}) ->
  {i == U32.and(i, U32.sub(Size.capacity(d), 1)) : U32}:
  match i:
    case U32{+x}:
      %mask_cap(d, h) : {U32{x} == U32.and(U32{x}, _) : U32}
      %Arith.mask_word(32n, d, x, inside) : {U32{x} == U32{_} : U32}
      {==}

def route_offset(+d: Nat, +i: U32, +h: {Nat.is_le(d, 20n) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(i), Model.capacity(d)) == True{} : Bool}) ->
  {U32.to_nat(i) == Model.offset(d, Refinement.route(d, i)) : Nat}:
  %mask_u32(d, i, h, inside) : {U32.to_nat(i) == Model.offset(d, Refinement.trace(d, Size.capacity(d), _, U32.is_lt(_, U32.shr(Size.capacity(d))))) : Nat}
  trace_offset(d, i, U32.is_lt(i, U32.shr(Size.capacity(d))), {==}, h, inside)

def index_complete(+d: Nat, +n: Nat, +i: U32, +fits: {Storage.bounds(d, n) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(i), n) == True{} : Bool}) -> {True{} == Storage.index_matches(d, n, i) : Bool}:
  +depth = Arith.and_left(Nat.is_le(d, 20n), Bool.and(Nat.is_lt(0n, n), Nat.is_le(n, Model.capacity(d))), fits)
  +rest = Arith.and_right(Nat.is_le(d, 20n), Bool.and(Nat.is_lt(0n, n), Nat.is_le(n, Model.capacity(d))), fits)
  +full = Arith.and_right(Nat.is_lt(0n, n), Nat.is_le(n, Model.capacity(d)), rest)
  +slot = Arith.lt_le_trans(U32.to_nat(i), n, Model.capacity(d), inside, full)
  %route_offset(d, i, depth, slot) : {True{} == Bool.and(Nat.is_eq(_, U32.to_nat(i)), Nat.is_lt(_, n)) : Bool}
  %Equal.sym(Bool, Nat.is_eq(U32.to_nat(i), U32.to_nat(i)), True{}, Arith.eq_refl(U32.to_nat(i))) : {True{} == Bool.and(_, Nat.is_lt(U32.to_nat(i), n)) : Bool}
  Equal.sym(Bool, Nat.is_lt(U32.to_nat(i), n), True{}, inside)

def index_matched(+d: Nat, +n: Nat, +i: U32, +fits: {Storage.bounds(d, n) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(i), n) == True{} : Bool}) -> {Storage.index_matches(d, n, i) == True{} : Bool}:
  Equal.sym(Bool, True{}, Storage.index_matches(d, n, i), index_complete(d, n, i, fits, inside))
