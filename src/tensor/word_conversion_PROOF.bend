import Base
import ./word_conversion.bend as Conversion
import ./word_conversion_LAWS.bend as Laws

def Laws.word_increment_exact(width, word, zero):
  Conversion.word_increment_exact(width, word, zero)

def Laws.increment_below_exact(word, limit, below):
  Conversion.increment_below_exact(word, limit, below)

def Laws.natural_word_exact(number, limit, within):
  Conversion.natural_word_exact(number, limit, within)

def Laws.word_natural_exact(word):
  Conversion.word_natural_exact(word)

def Laws.naturals_words_fitting(dims, values, fit):
  Conversion.naturals_words_fitting(dims, values, fit)

def Laws.word_increment_le(width, word):
  Conversion.word_increment_le(width, word)

def Laws.increment_le(word):
  Conversion.increment_le(word)

def Laws.natural_word_le(number):
  Conversion.natural_word_le(number)

def Laws.capacity_word_nat():
  Conversion.capacity_word_nat()

def Laws.capacity_natural_word_exact(number, within):
  Conversion.capacity_natural_word_exact(number, within)

def Laws.shape_count_exact(shape):
  Conversion.shape_count_exact(shape)

def Laws.shape_count_within(shape):
  Conversion.shape_count_within(shape)
