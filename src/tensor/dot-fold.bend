import Base

# The specification of one matrix-product cell: fold the factor pairs in
# list order, adding each product to the accumulator. It holds no law, so
# modules that state totals with it do not import the matmul proofs.
type Factors is Data:
  Factors{left: F32, right: F32}

def dot_fold(values: List<&2, Factors>, accumulator: F32) -> F32:
  match values:
    case Nil{}:
      accumulator
    case Factors{left, right} <> tail:
      dot_fold(tail, F32.add(accumulator, F32.mul(left, right)))
