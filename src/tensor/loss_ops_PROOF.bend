import Base
import ./elementwise.bend as Elementwise
import ./loss_ops_LAWS.bend as Laws

def Laws.scalar_divide(left, right):
  {==}

def Laws.scalar_exp_left(left, right):
  {==}

def Laws.scalar_log_left(left, right):
  {==}

def Laws.scalar_matches_same(left, right, same):
  %Equal.sym(Bool, U32.is_eq(Elementwise.bits(left), Elementwise.bits(right)), True{}, same) :
    {Elementwise.zero_if(Bool.not(_), 1.0) == 1.0 : F32}
  {==}

def Laws.scalar_matches_other(left, right, other):
  %Equal.sym(Bool, U32.is_eq(Elementwise.bits(left), Elementwise.bits(right)), False{}, other) :
    {Elementwise.zero_if(Bool.not(_), 1.0) == 0.0 : F32}
  {==}

def Laws.scalar_pick_negated_target(left, right, target):
  %Equal.sym(Bool, Elementwise.is_one(right), True{}, target) : {Elementwise.zero_if(Bool.not(_), F32.neg(left)) == F32.neg(left) : F32}
  {==}

def Laws.scalar_pick_negated_other(left, right, other):
  %Equal.sym(Bool, Elementwise.is_one(right), False{}, other) : {Elementwise.zero_if(Bool.not(_), F32.neg(left)) == 0.0 : F32}
  {==}

def Laws.scalar_mse_grad(scale, left, right):
  {==}

def Laws.scalar_one_hot_grad_target(count, left, right, target):
  %Equal.sym(Bool, Elementwise.is_one(right), True{}, target) :
    {Elementwise.zero_if(Bool.not(_), Elementwise.target_gradient(left, count)) == F32.neg(F32.div(left, count)) : F32}
  {==}

def Laws.scalar_one_hot_grad_other(count, left, right, other):
  %Equal.sym(Bool, Elementwise.is_one(right), False{}, other) :
    {Elementwise.zero_if(Bool.not(_), Elementwise.target_gradient(left, count)) == 0.0 : F32}
  {==}

def Laws.scalar_softmax_grad_one_class(count, left, right):
  {==}

def Laws.scalar_softmax_grad_classes(count, left, right):
  {==}
