"""Binary SHA-256 Merkle tree, authored in jax — byte-identical to flock's
`merkle::merkle_root` / `merkle_tree`.

flock's construction (no domain separation): each leaf hash = `SHA256(leaf_bytes)`,
each internal node = `SHA256(left ‖ right)` (64-byte preimage). The tree is built
bottom-up; the root is the single top node.

Every level is one data-parallel batched SHA-256 (`flock_zorch.sha256.digest`) over
all nodes at that level — leaves and each internal level map straight to the GPU's
width. Levels are sequential (log2(n_leaves) of them).
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from flock_zorch import sha256


def merkle_root(leaves) -> np.ndarray:
    """32-byte Merkle root of `n_leaves` equal-sized leaves.

    leaves: uint8 [n_leaves, leaf_size] (n_leaves a power of two). Returns uint8
    [32], byte-identical to flock's `merkle_root(data, n_leaves)`.
    """
    nodes = sha256.digest(leaves)  # [n_leaves, 32]  (leaf hashes)
    n = nodes.shape[0]
    while n > 1:
        pairs = jnp.asarray(nodes).reshape(n // 2, 64)  # left ‖ right (64-byte preimage)
        nodes = sha256.digest(pairs)                    # [n/2, 32]
        n //= 2
    return np.asarray(nodes).reshape(32)


def merkle_root_from_flat(data, n_leaves: int) -> np.ndarray:
    """Convenience: split a flat uint8 buffer into `n_leaves` equal leaves, hash."""
    data = np.asarray(data, dtype=np.uint8).reshape(n_leaves, -1)
    return merkle_root(data)
