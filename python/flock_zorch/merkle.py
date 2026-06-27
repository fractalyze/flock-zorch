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

    Device-resident: nodes stay on the GPU across all log2(n_leaves) levels (only
    one host copy of the final root). The earlier per-level `np.asarray`/`jnp.asarray`
    round-trip was catastrophic (~5 s for 2048 leaves — re-pad on host each level).
    """
    leaves = jnp.asarray(leaves, dtype=jnp.uint8)
    leaf_size = int(leaves.shape[1])
    nodes = sha256.digest_device(leaves, leaf_size)  # [n, 32] device
    n = int(nodes.shape[0])
    while n > 1:
        pairs = nodes.reshape(n // 2, 64)               # left ‖ right (stays on device)
        nodes = sha256.digest_device(pairs, 64)         # [n/2, 32] device
        n //= 2
    return np.asarray(nodes).reshape(32)                # single host copy of the root


def merkle_root_from_flat(data, n_leaves: int) -> np.ndarray:
    """Convenience: split a flat uint8 buffer into `n_leaves` equal leaves, hash."""
    data = np.asarray(data, dtype=np.uint8).reshape(n_leaves, -1)
    return merkle_root(data)


def merkle_tree(leaves) -> np.ndarray:
    """Full flat Merkle tree, byte-identical to flock's `merkle_tree` layout:
    `tree[0..n]` = leaf hashes (level k), then level k-1, …, root at `tree[2n-2]`.

    leaves: uint8 [n, leaf_size] (n a power of two). Returns uint8 [2n-1, 32].
    Device-resident per level (one host copy of the concatenated tree)."""
    leaves = jnp.asarray(leaves, dtype=jnp.uint8)
    leaf_size = int(leaves.shape[1])
    nodes = sha256.digest_device(leaves, leaf_size)   # [n, 32] device
    levels = [nodes]
    n = int(nodes.shape[0])
    while n > 1:
        nodes = sha256.digest_device(nodes.reshape(n // 2, 64), 64)  # [n/2, 32]
        levels.append(nodes)
        n //= 2
    return np.asarray(jnp.concatenate(levels, axis=0))   # [2*n_leaves - 1, 32]


def merkle_multi_proof(tree: np.ndarray, num_leaves: int, positions) -> np.ndarray:
    """Octopus multi-proof, byte-identical to flock `merkle::merkle_multi_proof`.

    Emits, per level (leaves→root), the sibling `tree[level_start + (p^1)]` of each
    active node whose sibling is NOT itself active — sorted+deduped positions,
    bottom-up, left-to-right. Returns uint8 [num_siblings, 32]."""
    if len(positions) == 0 or num_leaves == 1:
        return np.zeros((0, 32), dtype=np.uint8)
    active = sorted(set(int(p) for p in positions))
    proof = []
    level_start, level_len = 0, num_leaves
    while level_len > 1:
        nxt, i = [], 0
        while i < len(active):
            p = active[i]
            if i + 1 < len(active) and active[i + 1] == (p ^ 1):
                i += 2                                   # sibling also active → no emit
            else:
                proof.append(tree[level_start + (p ^ 1)])
                i += 1
            nxt.append(p >> 1)
        active = nxt
        level_start += level_len
        level_len >>= 1
    return np.stack(proof) if proof else np.zeros((0, 32), dtype=np.uint8)
