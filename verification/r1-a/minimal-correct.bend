import Base

type Receipt is Data:
  Receipt{equal: {0n == 0n : Nat}}

def diagnostic_zero() -> Nat:
  0n

def diagnostic_one() -> Nat:
  1n

def producer() -> Receipt:
  Receipt{{==}}

def expose(receipt: Receipt) -> {0n == 0n : Nat}:
  Receipt{equal} = receipt
  equal

def consumer() -> {0n == 0n : Nat}:
  expose(producer())

def zero_reflexivity() -> {diagnostic_zero() == diagnostic_zero() : Nat}:
  consumer()
