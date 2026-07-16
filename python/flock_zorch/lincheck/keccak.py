"""Keccak-f[1600] lincheck circuit walker — Python port of flock's
`KeccakLincheckCircuit::fold_alpha_batched` (`crates/flock-prover/src/r1cs_hashes/
keccak.rs:695-829`). Task #14, M1.

Keccak's monolithic R1CS at K_LOG=16 carries **empty** A_0/B_0 stubs — the
constraint definition lives entirely in this procedural walker, which computes the
lincheck column marginal `comb[c] = α·(A_0ᵀ·eq)[c] ⊕ (B_0ᵀ·eq)[c]` by a backward
transpose recurrence over the 24 Keccak rounds (state_r is implicit via
`state_r = φ^r·state_0 ⊕ Σ_{i<r} φ^{r-1-i}·t_i ⊕ RC_r`, φ = θ∘ρ∘π), in ~1M F128
ops, independent of the (huge) substituted matrix density.

Plugs into `lincheck.prove(circuit=KeccakLincheckCircuit())` exactly like the sha2
`CscCircuit`: a host XOR-scatter (`np.bitwise_xor.at`) over the fixed θ∘ρ∘π
preimage map, with the single `const_pin` (= Z_CONST) +β column applied by the
caller. The A-side (α-scaled) and B-side (plain) contributions are accumulated into
two unscaled F128 buffers and combined with ONE field multiply at the end — byte-
identical to flock's interleaved `comb[c] += α·x` because GF(2¹²⁸) multiplication
distributes over field addition (XOR).
"""

import functools

import numpy as np
import frx
import frx.numpy as jnp

from flock_zorch import field

# --- Layout constants (keccak.rs) -----------------------------------------
N_LANES = 25
LANE_BITS = 64
STATE_BITS = 1600  # == N_LANES * LANE_BITS
N_T = 24           # t_0 .. t_23 AND-output vectors == N_ROUNDS
K_LOG = 16
K = 1 << K_LOG     # 65536 columns
SLOT_BITS = 2048
STATE0_BIT_BASE = 0
STATE24_BIT_BASE = SLOT_BITS               # 2048
Z_CONST = 2 * SLOT_BITS                     # 4096 — the const-pin column
T_PACKED_BIT_BASE = Z_CONST + LANE_BITS     # 4160

# ρ rotation offsets r[x][y] (FIPS 202 Table 2).
RHO_OFFSETS = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]

# ι round constants (24 rounds).
ROUND_CONSTANTS = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]


def _state_idx(x, y, z):
    """state_idx(x,y,z) = x + 5y + 25z, in [0, STATE_BITS)."""
    return x + 5 * y + 25 * z


def _theta_rho_pi_preimage(x, y, z):
    """The 11 distinct state_in bit indices whose XOR equals B[x,y,z], where
    B = (π∘ρ∘θ)(state_in). Mirrors `theta_rho_pi_preimage` (keccak.rs:148)."""
    a = (x + 3 * y) % 5
    b = x
    r = RHO_OFFSETS[a][b] % 64
    c = (z + 64 - r) % 64
    c_prev = (c + 63) % 64
    a_minus = (a + 4) % 5
    a_plus = (a + 1) % 5
    bits = [0] * 11
    bits[0] = _state_idx(a, b, c)
    for yp in range(5):
        bits[1 + yp] = _state_idx(a_minus, yp, c)
    for yp in range(5):
        bits[6 + yp] = _state_idx(a_plus, yp, c_prev)
    return bits


def _build_preimage_maps():
    """Precompute the fixed preimage index maps (independent of round / eq):
      _PRE_FWD[s] = preimage(decode(s))         (φ / φᵀ)
      _PRE_CHI_A[j]   = preimage((x+1)%5, y, z)      (χ a-operand for t-AND row j)
      _PRE_CHI_B[j]   = preimage((x+2)%5, y, z)      (χ b-operand)
    where decode(s): z = s//25, x = (s%25)%5, y = (s%25)//5  (== state_idx layout)."""
    pre_fwd = np.zeros((STATE_BITS, 11), np.int64)
    pre_a = np.zeros((STATE_BITS, 11), np.int64)
    pre_b = np.zeros((STATE_BITS, 11), np.int64)
    for s in range(STATE_BITS):
        z = s // N_LANES
        xy = s % N_LANES
        x, y = xy % 5, xy // 5
        pre_fwd[s] = _theta_rho_pi_preimage(x, y, z)
        pre_a[s] = _theta_rho_pi_preimage((x + 1) % 5, y, z)
        pre_b[s] = _theta_rho_pi_preimage((x + 2) % 5, y, z)
    return pre_fwd, pre_a, pre_b


_PRE_FWD, _PRE_CHI_A, _PRE_CHI_B = _build_preimage_maps()  # φ preimage; χ a/b-operand preimages

# within_lane_contiguous(j) = 64·(j%25) + j//25 — the witness sub-vector offset.
_J = np.arange(STATE_BITS)
_WLC = (LANE_BITS * (_J % N_LANES) + _J // N_LANES).astype(np.int64)
_COL_STATE0 = _WLC                                  # z_pos_state(0, j)
_COL_STATE24 = STATE24_BIT_BASE + _WLC              # z_pos_state(24, j)
# z_pos_t(r, j) = 4160 + r·1600 + within_lane_contiguous(j); shape (N_T, STATE_BITS).
_ROWS_T = (T_PACKED_BIT_BASE + np.arange(N_T)[:, None] * STATE_BITS + _WLC[None, :]).astype(np.int64)
# state_idx(0,0,zpos) = 25·zpos — the ι round-constant toggle positions.
_RC_TOGGLE_IDX = N_LANES * np.arange(LANE_BITS)
_Z_BITS = np.arange(LANE_BITS, dtype=np.uint64)


# ---- The lincheck walker (device fold) --------------------------------------
# flock computes comb[c] = α·(A₀ᵀ·eq)[c] ⊕ (B₀ᵀ·eq)[c] with an overlapping φᵀ / χ
# scatter over the θ∘ρ∘π preimage maps. That transpose has uniform in-degree 11
# (measured), so it becomes a clean (STATE_BITS, 11) gather + XOR-reduce — a forward
# φ gather over the transposed map, byte-identical (same GF(2¹²⁸) XOR edge set). The
# recurrences are unrolled (static depth 24) so the whole fold lowers to one device
# program (no host↔device bubble).

def _transpose_map(pre_map):
    """Transpose a θ∘ρ∘π preimage map (S,11) into a gather map (S,11): _T[t] = the
    sources s with t ∈ pre_map[s]. Turns the overlapping scatter into a gather."""
    buckets = [[] for _ in range(STATE_BITS)]
    for s in range(STATE_BITS):
        for p in pre_map[s]:
            buckets[int(p)].append(s)
    assert {len(b) for b in buckets} == {11}, "θ∘ρ∘π transpose fan-in must be 11"
    return jnp.asarray(np.array(buckets, np.int64))


_FWD_T = _transpose_map(_PRE_FWD)          # φᵀ gather
_CHI_A_T = _transpose_map(_PRE_CHI_A)      # χ a-operand gather
_CHI_B_T = _transpose_map(_PRE_CHI_B)      # χ b-operand gather
_PRE_FWD_DEV = jnp.asarray(_PRE_FWD)       # forward φ_bool gather
_RC_TOGGLE_DEV = jnp.asarray(_RC_TOGGLE_IDX)
_RC_BITS = jnp.asarray(np.stack([          # (N_T, LANE_BITS) ι round-constant toggle bits
    (np.uint64(rc) >> _Z_BITS) & np.uint64(1) for rc in ROUND_CONSTANTS]))


def _gather_xor(vals, map_T):
    """φᵀ / χ scatter as a gather+XOR-reduce: out[t] = XOR_k vals[map_T[t,k]]."""
    return jnp.bitwise_xor.reduce(vals[map_T], axis=-2)


def _accumulate_subkeccak_dev(eq, col_state0, col_state24, rows_t):
    """One sub-keccak's comb_a/comb_b contributions (flock keccak.rs
    `fold_alpha_batched` body / keccak3.rs `accumulate_subkeccak`), returned
    functionally (no in-place XOR): values at rows_t (bijective) + col_state0 (2
    contribs, pre-XORed) + the z_const scalars. The caller scatter-sets the bijective
    columns and XOR-merges the shared z_const."""
    e_s0 = eq[col_state0]                                          # (S,2)
    vec_pin = eq[col_state24]                                      # (S,2)
    e_t = eq[rows_t]                                               # (N_T,S,2)
    chi_a = jnp.bitwise_xor.reduce(e_t[:, _CHI_A_T], axis=2)       # (N_T,S,2)
    chi_b = jnp.bitwise_xor.reduce(e_t[:, _CHI_B_T], axis=2)

    zc_a = jnp.bitwise_xor.reduce(e_t.reshape(-1, 2), axis=0)      # Σ eq_t (A z_const)
    zc_b = (jnp.bitwise_xor.reduce(e_s0, axis=0)                   # state_0 + state_24 pins
            ^ jnp.bitwise_xor.reduce(vec_pin, axis=0))

    # Round-constant GF(2) state machine (unrolled N_T) → RC_24.
    rc = jnp.zeros(STATE_BITS, jnp.uint64)
    rc_a = jnp.zeros(2, jnp.uint64)
    rc_b = jnp.zeros(2, jnp.uint64)
    for r in range(N_T):
        mask = rc[:, None]
        rc_a = rc_a ^ jnp.bitwise_xor.reduce(chi_a[r] * mask, axis=0)
        rc_b = rc_b ^ jnp.bitwise_xor.reduce(chi_b[r] * mask, axis=0)
        rc = jnp.bitwise_xor.reduce(rc[_PRE_FWD_DEV], axis=1)      # forward φ_bool
        rc = rc.at[_RC_TOGGLE_DEV].set(rc[_RC_TOGGLE_DEV] ^ _RC_BITS[r])
    rc_pin = jnp.bitwise_xor.reduce(vec_pin * rc[:, None], axis=0)
    zc_a = zc_a ^ rc_a ^ rc_pin
    zc_b = zc_b ^ rc_b

    # A-side transpose recurrence (unrolled): rows_t[j] ← K^A_{j+1}, col_state0 ← K^A_0.
    ra = [None] * N_T
    ra[N_T - 1] = vec_pin                                          # K^A_24
    k_a = _gather_xor(vec_pin, _FWD_T) ^ chi_a[N_T - 1]            # K^A_23
    for r in range(N_T - 1, 0, -1):
        ra[r - 1] = k_a                                           # K^A_r → rows_t[r-1]
        k_a = _gather_xor(k_a, _FWD_T) ^ chi_a[r - 1]
    cs0_a = e_s0 ^ k_a                                            # state_0 self-loop ⊕ K^A_0

    # B-side (K^B_24 = 0): rows_t[j] ← K^B_{j+1} (0 at j=N_T-1), col_state0 ← K^B_0.
    rb = [None] * N_T
    rb[N_T - 1] = jnp.zeros((STATE_BITS, 2), jnp.uint64)
    k_b = chi_b[N_T - 1]                                          # K^B_23
    for r in range(N_T - 1, 0, -1):
        rb[r - 1] = k_b
        k_b = _gather_xor(k_b, _FWD_T) ^ chi_b[r - 1]
    cs0_b = k_b                                                   # K^B_0

    return jnp.stack(ra), jnp.stack(rb), cs0_a, cs0_b, zc_a, zc_b


@functools.partial(frx.jit, static_argnums=(3,))
def _fold_walker_dev(eq, alpha, sub_cols, z_const):
    """Device fold shared by both keccak walkers. Run each disjoint sub-keccak,
    scatter-SET its bijective columns (rows_t, col_state0 — no atomics), XOR-merge
    the shared z_const column (incl. the row-0 const), and α-combine with one field
    multiply. `eq` is (n_cols, 2); `sub_cols` a list of (col_state0, col_state24,
    rows_t) device index arrays; `z_const` a static int column."""
    n_cols = eq.shape[0]
    comb_a = jnp.zeros((n_cols, 2), jnp.uint64)
    comb_b = jnp.zeros((n_cols, 2), jnp.uint64)
    zc_a = jnp.zeros(2, jnp.uint64)
    zc_b = jnp.zeros(2, jnp.uint64)
    for col_state0, col_state24, rows_t in sub_cols:
        ra, rb, ca, cb, za, zb = _accumulate_subkeccak_dev(eq, col_state0, col_state24, rows_t)
        rtf = rows_t.reshape(-1)
        comb_a = comb_a.at[rtf].set(ra.reshape(-1, 2)).at[col_state0].set(ca)
        comb_b = comb_b.at[rtf].set(rb.reshape(-1, 2)).at[col_state0].set(cb)
        zc_a = zc_a ^ za
        zc_b = zc_b ^ zb
    e0 = eq[z_const]                                             # row-0 const (shared)
    comb_a = comb_a.at[z_const].set(zc_a ^ e0)
    comb_b = comb_b.at[z_const].set(zc_b ^ e0)
    ca_g, cb_g = field.to_ghash(comb_a), field.to_ghash(comb_b)
    return alpha * ca_g + cb_g                # α·comb_a ⊕ comb_b


def _device_sub_cols(sub_cols):
    """Device copies of a walker's host index arrays, built once per circuit so the
    constant column maps aren't re-transferred to the device on every fold."""
    return [(jnp.asarray(c0), jnp.asarray(c24), jnp.asarray(rt)) for c0, c24, rt in sub_cols]


def _fold_walker(eq_inner, alpha, sub_cols_dev, z_const):
    """Production entry: reshape eq on device and fold, staying device-resident (the
    caller reuses the result on device, like `CscCircuit`). `sub_cols_dev` is the
    circuit's device index arrays (built once via `_device_sub_cols`)."""
    eq = jnp.asarray(eq_inner, dtype=jnp.uint64).reshape(-1, 2)
    return _fold_walker_dev(eq, jnp.asarray(alpha), sub_cols_dev, int(z_const))


class KeccakLincheckCircuit:
    """The procedural single-keccak lincheck walker (flock `KeccakLincheckCircuit`)."""

    n_cols = K
    const_pin = Z_CONST  # const-wire pin column (lincheck.prove applies +β here)
    _sub_cols = [(_COL_STATE0, _COL_STATE24, _ROWS_T)]   # host arrays (test reference)
    _sub_cols_dev = _device_sub_cols(_sub_cols)          # device, built once

    def fold_alpha_batched(self, alpha, eq_inner):
        """comb[c] = α·(A_0ᵀ·eq)[c] ⊕ (B_0ᵀ·eq)[c], the keccak.rs walker (device)."""
        return _fold_walker(eq_inner, alpha, self._sub_cols_dev, Z_CONST)
