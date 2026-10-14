import Base
import ../shape/core.bend as Shape
import ./word-conversion.bend as Conversion

law word_increment_exact:
  for +width: Nat
  for +word: Word(width)
  for zero: {Conversion.word_has_zero(width, word) == True{} : Bool}
  {Word.to_nat(width, Word.inc(width, word)) == 1n+Word.to_nat(width, word) : Nat}

law increment_below_exact:
  for +word: U32
  for +limit: U32
  for below: {U32.is_lt(word, limit) == True{} : Bool}
  {U32.to_nat(U32.inc(word)) == 1n+U32.to_nat(word) : Nat}

law natural_word_exact:
  for +number: Nat
  for +limit: U32
  for +within: {Nat.is_le(number, U32.to_nat(limit)) == True{} : Bool}
  {U32.to_nat(U32.from_nat(number)) == number : Nat}

law word_natural_exact:
  for +word: U32
  {U32.from_nat(U32.to_nat(word)) == word : U32}

law naturals_words_fitting:
  for +dims: List<&2, U32>
  for +values: List<&2, Nat>
  for +fit: {True{} == Shape.fit(Shape.naturals(dims), values) : Bool}
  {Shape.naturals(Shape.words(values)) == values : List<&2, Nat>}

law word_increment_le:
  for +width: Nat
  for +word: Word(width)
  {Nat.is_le(Word.to_nat(width, Word.inc(width, word)), 1n+Word.to_nat(width, word)) == True{} : Bool}

law increment_le:
  for +word: U32
  {Nat.is_le(U32.to_nat(U32.inc(word)), 1n+U32.to_nat(word)) == True{} : Bool}

law natural_word_le:
  for +number: Nat
  {Nat.is_le(U32.to_nat(U32.from_nat(number)), number) == True{} : Bool}

law capacity_word_nat:
  {U32.to_nat(1048576) == 1048576n : Nat}

law capacity_natural_word_exact:
  for +number: Nat
  for within: Shape.LE(number, 1048576n)
  {U32.to_nat(U32.from_nat(number)) == number : Nat}

law shape_count_exact:
  for +shape: Shape.Shape
  {U32.to_nat(Shape.count(shape)) == Shape.elements(shape) : Nat}

law shape_count_within:
  for +shape: Shape.Shape
  {U32.is_le(Shape.count(shape), 1048576) == True{} : Bool}
