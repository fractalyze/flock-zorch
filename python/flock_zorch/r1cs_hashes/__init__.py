# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The four hash R1CS circuits — BLAKE3, SHA-256, Keccak-f[1600], Keccak3.

Named for flock's own `flock-prover/src/r1cs_hashes/`, which keeps the same four
circuits in one directory; the proof gates diff against that source, so the
directory a reader lands in matches the one they are diffing.

A circuit contributes two things to the prove, and each is a module here:

    circuit   witness (packed z/a/b)      lincheck circuit (the A₀/B₀ column fold)
    -------   ------------------------    ----------------------------------------
    blake3    blake3_witness              — (populated sparse A₀/B₀; see below)
              blake3_witness_pallas
    sha2      sha2_witness                — (populated sparse A₀/B₀; see below)
    keccak    keccak_witness              keccak_lincheck
    keccak3   keccak_witness              keccak3_lincheck

Two asymmetries are real, not tidiness debt:

- **blake3 and sha2 have no lincheck module.** Their goldens carry populated
  sparse A₀/B₀ matrices, so callers build the generic
  `lincheck.CscCircuit(g["a0_rows"], g["b0_rows"], ...)` straight from the dump
  and no per-circuit code exists to write. keccak and keccak3 ship **empty**
  A₀/B₀ stubs instead — their constraint definition exists only as the
  procedural transpose walker, which is what `*_lincheck` is.
- **One witness module covers keccak and keccak3.** They share an emitter and
  differ only in the lane map, so `keccak_witness` is parameterized by `Spec`
  (`KECCAK`, `KECCAK3`). Their walkers are separate modules because keccak3's
  reuses keccak's, exactly as flock's `keccak3.rs` reuses `super::keccak`.

`common` holds the bit-packing shared by the tightly-packed circuits (flock's
`r1cs_hashes/common.rs`); `blake3_witness_pallas` is not shared machinery but
blake3's one-kernel emission arm, reached only through `blake3_witness`.

This package re-exports nothing, unlike `sumcheck` / `zerocheck` / `lincheck`:
the walkers build their device column maps at module scope, so an `import *`
here would put every circuit's device work on the import of any one of them.
Import the module you want by path.
"""
