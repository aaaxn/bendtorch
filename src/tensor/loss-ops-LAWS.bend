import Base
import ./elementwise.bend as Elementwise

# The elementwise operations of the mean and of the two losses, each pinned
# to the F32 operations it applies, in PyTorch's order. F32.div, F32.exp and
# F32.log are opaque in Base, so these laws state which operations run on
# which words, not how close exp and log are to the exact values.

# mean: the sum divided by the count.
law scalar_divide:
  for +left: F32
  for +right: F32
  {Elementwise.scalar(Elementwise.Divide{}, left, right) == F32.div(left, right) : F32}

# exp of the left word, the right word unread.
law scalar_exp_left:
  for +left: F32
  for +right: F32
  {Elementwise.scalar(Elementwise.ExpLeft{}, left, right) == F32.exp(left) : F32}

# log of the left word, the right word unread.
law scalar_log_left:
  for +left: F32
  for +right: F32
  {Elementwise.scalar(Elementwise.LogLeft{}, left, right) == F32.log(left) : F32}

# One-hot: 1.0 where the two words are the same bits, +0.0 elsewhere.
law scalar_matches_same:
  for +left: F32
  for +right: F32
  for same: {U32.is_eq(Elementwise.bits(left), Elementwise.bits(right)) == True{} : Bool}
  {Elementwise.scalar(Elementwise.Matches{}, left, right) == 1.0 : F32}

law scalar_matches_other:
  for +left: F32
  for +right: F32
  for other: {U32.is_eq(Elementwise.bits(left), Elementwise.bits(right)) == False{} : Bool}
  {Elementwise.scalar(Elementwise.Matches{}, left, right) == 0.0 : F32}

# The nll term: -lp where the one-hot word is 1.0 (0x3F800000), +0.0
# elsewhere.
law scalar_pick_negated_target:
  for +left: F32
  for +right: F32
  for target: {Elementwise.is_one(right) == True{} : Bool}
  {Elementwise.scalar(Elementwise.PickNegated{}, left, right) == F32.neg(left) : F32}

law scalar_pick_negated_other:
  for +left: F32
  for +right: F32
  for other: {Elementwise.is_one(right) == False{} : Bool}
  {Elementwise.scalar(Elementwise.PickNegated{}, left, right) == 0.0 : F32}

# The three backward operations take the gradient g as the left word, as
# `Mul` does in the Mul VJP, and a saved word as the right one.

# PyTorch's mse_loss backward: (scale * d) * g, scale = 2 / N, for g on the
# left and d on the right.
law scalar_mse_grad:
  for +scale: F32
  for +left: F32
  for +right: F32
  {Elementwise.scalar(Elementwise.MseGrad{scale}, left, right) == F32.mul(F32.mul(scale, right), left) : F32}

# PyTorch's nll_loss backward with mean reduction, for g on the left and a
# one-hot word on the right: -(g / N) at a target, +0.0 elsewhere.
law scalar_one_hot_grad_target:
  for +count: F32
  for +left: F32
  for +right: F32
  for target: {Elementwise.is_one(right) == True{} : Bool}
  {Elementwise.scalar(Elementwise.OneHotGrad{count}, left, right) == F32.neg(F32.div(left, count)) : F32}

law scalar_one_hot_grad_other:
  for +count: F32
  for +left: F32
  for +right: F32
  for other: {Elementwise.is_one(right) == False{} : Bool}
  {Elementwise.scalar(Elementwise.OneHotGrad{count}, left, right) == 0.0 : F32}

# The rest of PyTorch's log_softmax backward, for g on the left and lp on
# the right: -(exp(lp) * r) with r = -(g / N) + 0.0, the row sum of the
# one-hot gradient.
law scalar_softmax_grad:
  for +count: F32
  for +left: F32
  for +right: F32
  {Elementwise.scalar(Elementwise.SoftmaxGrad{count}, left, right) ==
    F32.neg(F32.mul(F32.exp(right), F32.add(F32.neg(F32.div(left, count)), 0.0))) : F32}
