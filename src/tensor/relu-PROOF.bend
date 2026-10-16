import Base
import ./elementwise.bend as Elementwise
import ./relu-LAWS.bend as Laws

def Laws.below_zero_words(x):
  {==}

def Laws.at_most_zero_words(x):
  {==}

def Laws.below_zero_edges():
  {==}

def Laws.at_most_zero_edges():
  {==}

def Laws.scalar_relu_left_below(left, right, below):
  %Equal.sym(Bool, Elementwise.below_zero(left), True{}, below) : {Elementwise.zero_if(_, left) == 0.0 : F32}
  {==}

def Laws.scalar_relu_left_kept(left, right, kept):
  %Equal.sym(Bool, Elementwise.below_zero(left), False{}, kept) : {Elementwise.zero_if(_, left) == left : F32}
  {==}

def Laws.scalar_relu_backward_zeroed(left, right, zeroed):
  %Equal.sym(Bool, Elementwise.at_most_zero(left), True{}, zeroed) : {Elementwise.zero_if(_, right) == 0.0 : F32}
  {==}

def Laws.scalar_relu_backward_passed(left, right, passed):
  %Equal.sym(Bool, Elementwise.at_most_zero(left), False{}, passed) : {Elementwise.zero_if(_, right) == right : F32}
  {==}

def Laws.relu_edges():
  {==}

def Laws.relu_backward_edges():
  {==}
