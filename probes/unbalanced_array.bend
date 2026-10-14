import Base

def report(r: Array<U32> & U32) -> IO(Unit):
  (a, value) = r
  IO.print(U32.show(value))

def main() -> IO(Unit):
  report(Array.get(U32, ANode{ALeaf{1}, ANode{ALeaf{2}, ALeaf{3}}}, 0))
