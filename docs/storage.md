# Owned F32 storage

`src/storage/main.bend` is the executable module. `new(length, fill)` returns a
validated sole owner or `TooLarge`; `get(owner, index)` and `set(owner, index,
value)` return the owner alongside a success/error result. `clone(owner)`
returns two independent owners. `length(owner)` retains its owner. There are no
shared mutable views, borrowed aliases, unsafe forks, or PyTorch calls.

The empty constructor owns no array. Nonempty storage has a balanced Base
array, logical length between one and its power-of-two capacity, and depth at
most twenty. The balance and range certificates are erased. The public length
limit is 2^20; invalid lengths are rejected before array allocation. Reads and
updates reject indices outside logical length (`OutOfBounds`) before Base's
wrapping access, and then use Base's in-place `Array.get` or `Array.set`
directly. An in-bounds index needs no second check: `route.bend` proves that
for storage with these certificates the exact Base branch route of every index
below the logical length lands on that index's offset
(`index_complete`), so an in-bounds access never reaches another element or
padding.

## Source-level representation guarantees

`model.bend` defines balanced storage, typed binary paths, logical offsets, and
an independent element observation. `array.bend` supplies structurally proved
path read/write models. `size.bend`, `clone.bend`, and `refinement.bend` connect
those models to Base's actual size, clone, get, swap, and set definitions at
arbitrary balanced depth. The get/set representation theorems include Base's
mask, shifts, and high-branch subtraction; they do not assume a U32 decoding
identity. The primitive get-set and frame laws compose that connection with the
independent path laws, using explicit proof-only cloning where affine proof
composition needs two owners.

## Public API laws

`src/storage/LAWS.bend` also states the behavior of `new`, `get`, `set`,
`clone` and `length` themselves. `replay_get` says that after any sequence of
writes, reads, clones and length queries applied to `new(n, fill)`, reading an
in-bounds slot returns the last value written there, or `fill`. `replay_set`
says an in-bounds write after such a sequence reports `Done`, and
`replay_size` says `length` still returns `n`. `new_fits` says `new` succeeds
for every length up to 2^20, zero included, and `new_empty` says `new(0, fill)`
is `Empty`. `new_too_large`, `get_outside` and `set_outside` hold for any
storage value: lengths above 2^20 are `TooLarge`, and indices at or above the
length that `length` reports are `OutOfBounds`.

The laws quantify over operation histories rather than over arbitrary owned
values because the balance and range certificates are erased, so a proof
cannot use them, and an affine array cannot be both read and fed to a model
lemma. Every storage the API builds from `new` is covered. A storage forged
from erased certificates is not: reads and writes trust the range certificate
as they trust the balance certificate, and recompute no route at run time.
A later change that adds another way to make storage, such as an imported
buffer, needs its own law or a proof that it yields what these histories
reach. Clone laws state the contents of each copy; that the copies do not
share memory at run time is checked by the native corpus, not by a law.

The proofs in `api.bend` use `route.bend`, which proves that Base's route for
a U32 index below the capacity lands on that index's offset, from bit-level
lemmas in `arith.bend` relating `Word` compare, subtract and mask to `Nat`.
`Storage.index_matches`, the route check, stays as the statement that laws
such as `checked_route_separation` and `Core.ObservationChecked` use. `new`
compares the requested length as a U32 against 2^20, the same predicate as the
earlier `Nat` comparison, so that no proof has to expand the unary literal.

Certificate builders retain normal proof fields so later proofs can consume
them. Every builder invocation in the executable adapter occurs inside an
erased constructor certificate. Native reads and updates execute **Base's
in-place primitives**, not the mathematical tree walker. The storage gate
instruments generated C and rejects any executed packed-array split or
reconstruction. This guards against accidentally making a proof traversal live.

## Evidence and limits

The native and JavaScript corpus exercises 55 constructions and 2,322
observations per lane: retained owners, frame behavior, independent clones,
empty storage, padding/capacity boundaries, maximum project capacity, maximum
U32 indices, and F32 raw words including signed zero, subnormals, infinity, and
NaNs. At every depth from 6 to 20 and at length 2^20, distinct words are written
at slots 0, 2^k, 2^k-1 and length-1, then read back with untouched slots from
the owner and from a clone, so a write that aliased a high slot would show.
NaN payload differences are classified explicitly on JavaScript; native storage
word preservation is compared exactly. Invalid balance, logical length, and
depth certificates must be rejected by both checking entrypoints with their own
mismatch, and a valid twin of each, from the same template, must be accepted.

The copy audit counts zero packed-array splits and reconstructions over the
whole corpus, including the 2^20-element storage and its clone, so scalar
access copies no array.

These are source-model proofs plus executed backend evidence. Native array
lowering, scalar F32 primitives, allocation, Clang, OS, and hardware remain in
the trusted computing base. Resource exhaustion can fail-stop the runtime;
neither a dependent certificate nor a kernel verdict removes that boundary.
