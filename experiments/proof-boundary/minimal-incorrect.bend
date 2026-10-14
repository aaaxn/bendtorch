import Base

type Receipt is Data:
  Receipt{equal: {0n == 0n : Nat}}

def producer() -> Receipt:
  Receipt{{==}}

def expose(receipt: Receipt) -> {0n == 0n : Nat}:
  Receipt{equal} = receipt
  equal

def consumer() -> {0n == 1n : Nat}:
  expose(producer())
