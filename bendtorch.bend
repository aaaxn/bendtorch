# bendtorch: a PyTorch-like tensor library in Bend (F32 tensors, matmul, autograd tape, losses, SGD), with its laws and proofs; import one module, e.g. bendtorch@0.1.0.0/src/tensor/core.bend.
# Source, docs and license: https://github.com/aaaxn/bendtorch
#
# This file is the package entry. `bend bendtorch.bend --publish` ships it with
# everything it imports: the modules below, and PROOF.bend with every LAWS.bend.
# Import the module you need rather than this file: an import is local to the
# file that makes it, so this file's aliases are not visible to its importers,
# and importing it loads every proof. `bend bendtorch.bend --verdict` checks
# the laws of the whole package, as `bend PROOF.bend --verdict` does.
import Base

# Shapes and storage
import ./src/shape/core.bend as Shape
import ./src/storage/main.bend as Storage

# Tensors: construction, reads and writes, elementwise operations, reductions,
# matmul, layout transforms, PyTorch's signed axes, slicing and gathering
import ./src/tensor/core.bend as Core
import ./src/tensor/elementwise.bend as Elementwise
import ./src/tensor/reduction.bend as Reduction
import ./src/tensor/matmul.bend as Matmul
import ./src/tensor/batched-matmul.bend as Batched
import ./src/tensor/transforms.bend as Transforms
import ./src/tensor/layout.bend as Layout
import ./src/tensor/signed.bend as Signed
import ./src/tensor/slice.bend as Slice
import ./src/tensor/gather.bend as Gather

# Autograd: broadcasting and its sums back, gradient slots, the tape and its
# backward pass, the losses, SGD
import ./src/autograd/broadcast.bend as Broadcast
import ./src/autograd/reduce.bend as Reduce
import ./src/autograd/sum-to-size.bend as SumToSize
import ./src/autograd/sum-dims.bend as SumDims
import ./src/autograd/gradient.bend as Gradient
import ./src/autograd/backward.bend as ScalarBackward
import ./src/autograd/tensor-backward.bend as Backward
import ./src/autograd/loss.bend as Loss
import ./src/autograd/sgd.bend as Sgd

# Laws and their proofs
import ./PROOF.bend as Proof
