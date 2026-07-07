"""3-wide Keccak-f[1600] lincheck circuit walker — Python port of flock's
`keccak3::KeccakLincheckCircuit::fold_alpha_batched` (`crates/flock-prover/src/
r1cs_hashes/keccak3.rs:449-484`). Task #14, M3b — the headline keccak backend.

keccak3 packs THREE independent keccak permutations per block at K_LOG=17 (useful
≈ 97.3% vs single-keccak's 65%). The walker is exactly three disjoint copies of
the single-keccak transpose recurrence — `accumulate_subkeccak(i)` for i∈0..3,
each over sub-keccak i's column region, merged by XOR into a shared `comb` plus
one shared const row. All the per-sub-keccak phase logic, the θ∘ρ∘π preimage maps
and φ/φᵀ helpers are reused verbatim from `keccak_lincheck` (flock reuses
`super::keccak` likewise). Plugs into `lincheck.prove(circuit=)` like the single
walker; const_pin = the shared Z_CONST.
"""

import numpy as np

from flock_zorch.keccak_lincheck import (
    accumulate_subkeccak, _combine_alpha_sides, _WLC, LANE_BITS, STATE_BITS, N_T,
)

# --- keccak3 layout constants (keccak3.rs) --------------------------------
N_SUB = 3
K_LOG = 17
K = 1 << K_LOG                              # 131072 columns
SLOT_BITS = 2048
Z_CONST = 2 * N_SUB * SLOT_BITS            # 12288 — the shared const-pin column
T_PACKED_BIT_BASE = Z_CONST + LANE_BITS    # 12352


def _subkeccak_columns():
    """Per-sub-keccak witness column maps (keccak3.rs `z_pos_state`/`z_pos_t`):
      state_0[i]  at slot 2i      → (2i)·SLOT_BITS   + within_lane_contiguous(j)
      state_24[i] at slot 2i+1    → (2i+1)·SLOT_BITS + within_lane_contiguous(j)
      t[i,r]                      → T_PACKED + (i·N_T + r)·STATE_BITS + wlc(j)"""
    col0, col24, rows_t = [], [], []
    for i in range(N_SUB):
        col0.append(((2 * i) * SLOT_BITS + _WLC).astype(np.int64))
        col24.append(((2 * i + 1) * SLOT_BITS + _WLC).astype(np.int64))
        rt = (T_PACKED_BIT_BASE + (i * N_T + np.arange(N_T))[:, None] * STATE_BITS
              + _WLC[None, :]).astype(np.int64)
        rows_t.append(rt)
    return col0, col24, rows_t


_COL0, _COL24, _ROWS_T = _subkeccak_columns()


class Keccak3LincheckCircuit:
    """The procedural 3-wide keccak lincheck walker (flock `keccak3::KeccakLincheckCircuit`)."""

    n_cols = K
    const_pin = Z_CONST  # shared const-wire pin column (lincheck.prove applies +β here)

    def fold_alpha_batched(self, alpha, eq_inner):
        """comb[c] = α·(A_0ᵀ·eq)[c] ⊕ (B_0ᵀ·eq)[c] — three disjoint sub-keccak walks
        XOR-merged into the shared comb (incl. the shared Z_CONST column)."""
        eq = np.asarray(eq_inner, np.uint64).reshape(K, 2)
        comb_a = np.zeros((K, 2), np.uint64)  # α-scaled (A-side), accumulated unscaled
        comb_b = np.zeros((K, 2), np.uint64)  # plain (B-side)

        for i in range(N_SUB):
            accumulate_subkeccak(eq, comb_a, comb_b, _COL0[i], _COL24[i], _ROWS_T[i], Z_CONST)

        # ---- Row 0 (const, shared): A = [Z_CONST], B = [Z_CONST] — added once.
        e0 = eq[Z_CONST]
        comb_a[Z_CONST] ^= e0
        comb_b[Z_CONST] ^= e0

        return _combine_alpha_sides(comb_a, comb_b, alpha)
