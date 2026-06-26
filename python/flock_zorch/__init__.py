"""flock_zorch — jax port of succinctlabs/flock's R1CS-over-GF(2) PIOP prover.

Built bottom-up, each layer gated by a byte-match against unmodified flock
(see `testing/`): field -> additive-NTT -> Merkle -> zerocheck -> lincheck ->
PCS -> end-to-end proof. Consumed off PYTHONPATH (not pip-installed).
"""
