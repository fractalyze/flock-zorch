"""Merkle commitment.

`merkle` is flock's Merkle tree — both arms and the octopus assembly over them;
its own docstring is the reference for how they are built.

Nothing else lives here any more. The BLAKE3 rows this package used to host
because upstream had none — the resumable hash state and the Fiat-Shamir field
transcript — are now `hash_frx.blake3.streaming` and
`zorch.blake3_field_transcript`, alongside the SHA-256 rows that always came
from upstream. A new file belongs here because it is part of the commitment,
not because upstream happens not to have it yet.
"""
