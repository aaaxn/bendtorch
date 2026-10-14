import Base

def from_bits(u: U32) -> F32:
  U32{w} = u
  F32{w}

# z is 0 at run time but depends on argv, so clang cannot fold the F32 ops.
def word(z: U32, w: U32) -> F32:
  from_bits(U32.add(w, z))

def show(x: F32) -> IO(Unit):
  IO.print(U32.show(F32.bits(x)))

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    +z : U32 = U32.from_nat(Nat.sub(List.length(&1, String, xs), 1n))
    show(F32.add(word(z, 1069547520), word(z, 1074790400)))
    show(F32.sub(word(z, 1069547520), word(z, 1074790400)))
    show(F32.mul(word(z, 1069547520), word(z, 1074790400)))
    show(F32.div(word(z, 1069547520), word(z, 1074790400)))
    show(word(z, 2147483648))
    show(word(z, 1))
    show(F32.div(word(z, 1065353216), word(z, 0)))
    show(F32.div(word(z, 0), word(z, 0)))
    show(word(z, 2139095041))
    show(U32.to_f32(U32.add(16777217, z)))
    IO.print(U32.show(F32.to_u32(word(z, 1081081856))))
    show(F32.mul(word(z, 2139095039), word(z, 1073741824)))
    show(F32.div(F32.neg(word(z, 1065353216)), word(z, 0)))
