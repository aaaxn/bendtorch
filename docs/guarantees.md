# Guarantee ledger

The S0 gate establishes a reproducible checking path, not tensor correctness.
Its authoritative inventory is `verification/toolchain.json`; immutable source and
artifact pins are in `verification/references.json`. Each run records the
manifest hash, the hash of every file it wrote, each command's exit code,
stdout, stderr and elapsed time, the host, and tool binary fingerprints.

The upstream `LAWS.tree_is_seq` theorem concerns natural-number arithmetic.
The unchanged source is copied byte-for-byte from the pinned Bend checkout.
BendTT checks the exported theorem package directly and rejects a copy whose
law statement is falsified. The false, open, missing-import, and unsafe source
controls are rejected by Bend's checker before BendTT runs, each for its named
reason; the omitted-inventory control is rejected by the gate. All remain in
the run directory. A timeout, omitted declaration, or nonzero positive check
never counts as success. Runs do not overwrite an earlier run.

The trusted computing base includes the specification, source elaborator,
BendTT translation, kernel and its Lean toolchain, compiler, native runtime,
Clang, scalar primitives, effects, OS, and hardware. Two accepted checkers do
not verify translation fidelity or native lowering. F32 primitives remain
opaque assumptions; exact bit-pattern probes are empirical evidence only.
Unsafe raw-array sharing is not used by the core.

Protected files and future law roots require separate specification review.
Hash and inventory gates detect tampering in their inputs; they cannot prevent
someone with permission from editing the gate itself. GitHub branch protection
and restricted reviewer permissions must enforce that organizational boundary.
No claim that those repository settings are already enabled is made here.

Clean-directory reconstruction is a documented command, not recorded
evidence. S1 checks the shape/index core, general Base representation proofs,
and owned F32 adapter. The executable shape API is proved against
the Nat/Fin core: construction and broadcasting are complete, flatten,
unflatten, the checked projections and the flat-index projections refine the
core functions, and every broadcast plan carries the unique right-aligned
certificate. Error codes, and
the Base `U32.to_nat` and `U32.from_nat` conversions at the boundary, are
checked by the runtime corpus, not proved. S2 through S7 remain gated, not
completed. Performance, functional
coverage, numerical conformance, formal guarantees, and compatibility are
separate claims.
