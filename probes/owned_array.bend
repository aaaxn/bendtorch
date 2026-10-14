import Base

def report(r: Array<F32> & F32) -> IO(Array<F32>):
  (a, value) = r
  do IO<Array<F32>>:
    IO.print(U32.show(F32.bits(value)))
    return a

def cloned(pair: Array<F32> & Array<F32>) -> IO(Unit):
  (a, b) = pair
  do IO<Unit>:
    b : Array<F32> = Array.set(F32, b, 3, 11.0)
    a : Array<F32> <- report(Array.get(F32, a, 3))
    b : Array<F32> <- report(Array.get(F32, b, 3))
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    a : Array<F32> = Array.new(F32, 2n, 0.0)
    a : Array<F32> = Array.set(F32, a, 3, 7.0)
    a : Array<F32> <- report(Array.get(F32, a, 3))
    a : Array<F32> <- report(Array.get(F32, a, 5))
    cloned(Array.clone(F32, a))
