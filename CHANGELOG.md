# Changelog

Versions have four numbers, as BendHub requires; [VERSIONING](docs/VERSIONING.md)
says what each one means and how a release is made. Each version lists what a
program importing the package can see change.

## 0.1.0.0 (2026-10-18)

The first release. Bend2 pin `32d23b43` (Bend 2.0.37), BendTT on Lean 4.34.0.

- `src/shape/`: validated shapes up to rank 4, row-major flatten and
  unflatten, checked coordinates, broadcast plans.
- `src/storage/`: F32 storage with one owner, up to 2^20 elements, and a
  bounds check before every access.
- `src/tensor/`: the tensor core; elementwise operations; `sum`; `matmul`
  and `Batched.matmul` (`torch.matmul`); `reshape`, `permute_copy` and
  `transpose_copy`; PyTorch's signed axes and indices; basic slicing
  (`slice`, `narrow`, `select`) with negative positions; `index_select` and
  `gather`.
- `src/autograd/`: `broadcast_to`, `reduce_to`, `sum_to_size`, `sum_dims`,
  gradient slots, scalar and tensor `backward` over a tape, `linear`, `relu`,
  `mse_loss`, `cross_entropy`, and SGD.
- Large outputs build in balanced parallel halves; every output word is the
  word the one-thread build computes.
- 493 laws under `src/`, proved in the matching `PROOF.bend` files
  and checked again by the BendTT kernel.
- `examples/mnist/train.bend` trains a 784-128-10 MLP next to its PyTorch copy
  in `oracle/mnist/`.
