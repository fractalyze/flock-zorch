"""flock_zorch — frx port of succinctlabs/flock's R1CS-over-GF(2) PIOP prover.

Built bottom-up, each layer gated by a byte-match against unmodified flock (the
`testing/` gates in the repo, which the distribution does not carry): field ->
additive-NTT -> Merkle -> zerocheck -> lincheck -> PCS -> end-to-end proof.

Requires `jax_enable_x64`: the GF(2^128) elements are uint64 lane pairs bitcast
to the native field dtype, which x32 truncates. The repo pins it in `.bazelrc`;
a pip consumer sets `JAX_ENABLE_X64=true` (see the README).
"""

# Single source of truth for the version: pyproject.toml derives
# `project.version` from this attribute and the release workflow's tag gate reads
# both, so bumping this line is the whole bump.
__version__ = "0.1.0"
