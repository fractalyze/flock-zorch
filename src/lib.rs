//! flock-zorch — a GPU port of [succinctlabs/flock](https://github.com/succinctlabs/flock),
//! an R1CS-over-GF(2) PIOP prover, onto the zorch / zkx compiler stack in the
//! style of `bellman-zorch` and `accumulation-zorch`.
//!
//! The prover is authored in jax (`python/flock_zorch`) and exported to a fused
//! StableHLO `.mlirbc`; this crate is the thin Rust host driver plus the
//! byte-match oracles. Correctness is a byte-identical compare against unmodified
//! upstream flock (`../flock`), layered bottom-up:
//!
//! ```text
//! field (GF(2^128) GHASH multiply)   <- current
//!   -> additive NTT -> Merkle commit -> zerocheck -> lincheck -> PCS -> e2e proof
//! ```
//!
//! See `CLAUDE.md` for the byte-identity non-negotiable.

// Host driver modules (gpu/prove/...) land as the PJRT path is built out.
