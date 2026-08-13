"""Merkle commitment, plus the BLAKE3 rows upstream does not carry yet.

`merkle` is flock's Merkle tree — both arms and the octopus assembly over them;
its own docstring is the reference for how they are built.

The SHA-256 side needs nothing else from here: its hash core is
`hash_frx.sha256` and its Fiat-Shamir transcript is
`zorch.sha256_field_transcript`. BLAKE3 has no such rows upstream yet, so its
resumable hash state (`blake3_stream`) and field transcript
(`blake3_field_transcript`) are hosted here.

That second paragraph is the part with an expiry date: both rows are being
upstreamed — the stream to hash-frx, the transcript to zorch — and both delete
from here once the pins carry them, leaving this package Merkle commitment
alone. A new file belongs here because it is part of that commitment, not
because upstream happens not to have it yet.
"""
