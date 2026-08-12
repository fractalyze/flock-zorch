# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Device-side flock Keccak-f[1600] R1CS witness generation.

Emits the packed `z`/`a`/`b` bit-streams for a batch of Keccak permutations,
bit-identical to flock-prover's `r1cs_hashes::keccak` witness builder, the way
`witgen` already does for BLAKE3. The prove consumes exactly these buffers, so
producing them on device removes the host generation pass and its H2D crossing.

Layout, in whole u64 LANES rather than bit offsets — this is the structural
difference from BLAKE3. flock packs a keccak state into a 2,048-bit aligned
slot with an explicit zero pad instead of packing fields tightly, so nothing
straddles a word boundary and the "field list" is a lane map:

    lanes [  0,   25)  state_0[i]        lin-id
    lanes [ 25,   32)  slot zero pad     (a state is 1,600 of the slot's 2,048 bits)
    lanes [ 32,   57)  state_24[i]       lin-id
    lanes [ 57,   64)  slot zero pad
    lane          64   the constant-1 wire, in bit 0
    lanes [ 65,  665)  t_r[i], 25 lanes per round, r in [0, 24)   AND
    lanes [665, 1024)  zero padding

The R1CS is `a AND b = z` per bit. Keccak's only nonlinear step is chi, so the
AND row is the only non-identity form and it is expressed per lane rather than
per bit: with `B = rho_pi(theta(state_r))`, row `t_r[x + 5y]` stores
`z = (~B[(x+1)%5 + 5y]) & B[(x+2)%5 + 5y]`, `a = ~B[(x+1)%5 + 5y]`,
`b = B[(x+2)%5 + 5y]`. A lin-id row pins a wire `v` via `z = a = v`,
`b = 0xFFFF_FFFF_FFFF_FFFF`. The next state is `B ^ t` with iota applied.

Emission is whole-grid, never lane-by-lane. Writing the rounds as 25 separate
lane values is the obvious spelling and it does not compile: one op per lane per
round is ~9,800 HLO lines for a *single* round (the same wall
`hash_frx.keccak.permutation` documents), and a first cut here reached 31 GB of
compiler memory before it was killed. Carrying the state as one `[N, 5, 5]`
array keeps every step a handful of ops — theta is an unrolled XOR fold, rho is
a per-lane offset applied as one shift pair, pi is a static gather, chi is two
rolls. The t rows then land contiguously, so the packed stream is a concatenate
of whole regions rather than 1,024 assembled columns.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from hash_frx.keccak import params as _keccak

# "Lane" is overloaded in this module: a keccak state lane (25 of them) and an
# output word (1,024 of them) are both u64. The N_ prefix keeps the state's
# count distinct from `USEFUL_LANES` and `WORDS_PER_BLOCK`, which count words.
N_LANES = _keccak.LANES
N_ROUNDS = _keccak.ROUNDS
ROTATION_OFFSETS = _keccak.ROTATION_OFFSETS
ROUND_CONSTANTS = _keccak.ROUND_CONSTANTS

K_LOG = 16
K = 1 << K_LOG
LANE_BITS = 64
WORDS_PER_BLOCK = K // LANE_BITS  # 1024 u64 per block per stream

# flock gives each state a 2,048-bit aligned slot: state_0, then state_24, then
# the constant wire, then the t rows.
SLOT_BITS = 2048
SLOT_LANES = SLOT_BITS // LANE_BITS  # 32
STATE0_LANE = 0
STATE24_LANE = SLOT_LANES  # 32
Z_CONST_LANE = 2 * SLOT_LANES  # 64
T_LANE_BASE = Z_CONST_LANE + 1  # 65

USEFUL_LANES = T_LANE_BASE + N_ROUNDS * N_LANES  # 665
USEFUL_BITS = USEFUL_LANES * LANE_BITS  # 42,560

# The aligned slots are why keccak reaches only 65% utilization: 42,560 useful
# bits of a 65,536-bit block.
assert USEFUL_BITS == 42560
assert 2 * SLOT_BITS + LANE_BITS + N_ROUNDS * N_LANES * LANE_BITS == USEFUL_BITS

_ONES64 = (1 << 64) - 1
_SLOT_PAD_LANES = SLOT_LANES - N_LANES  # 7
_TAIL_PAD_LANES = WORDS_PER_BLOCK - USEFUL_LANES  # 359

# rho's offsets, flat by `x + 5y` and so indexed by the lane they rotate. The
# complement is precomputed because a shift by the full width is undefined and
# lane (0, 0)'s offset is 0: `(64 - 0) % 64` is the branch-free spelling.
_ROT = np.asarray(ROTATION_OFFSETS, dtype=np.uint64)
_ROT_INV = (np.uint64(LANE_BITS) - _ROT) % np.uint64(LANE_BITS)

# pi by destination: FIPS 202's "the lane at (x, y) moves to (y, 2x + 3y)" read
# backwards is `src(x, y) = (x + 3y) % 5 + 5x`, a static gather.
_PI_PERM = np.asarray(
    [(x + 3 * y) % 5 + 5 * x for y in range(5) for x in range(5)], dtype=np.intp
)

# iota, as a whole-state constant per round: the round constant hits lane (0, 0)
# only. hash_frx splits these into u32 halves for its own lane representation,
# which this module has no use for.
_RC = tuple(lo | (hi << 32) for lo, hi in ROUND_CONSTANTS)
_RC_LANES = np.zeros((N_ROUNDS, N_LANES), dtype=np.uint64)
_RC_LANES[:, 0] = _RC


def _li(x: int, y: int) -> int:
    """FIPS 202's flat lane index for (x, y)."""
    return x + 5 * y


def _rotl1(v):
    return (v << fnp.uint64(1)) | (v >> fnp.uint64(LANE_BITS - 1))


def _theta(g):
    """C[x] = parity of column x; D[x] = C[x-1] ^ rotl(C[x+1], 1); A ^= D[x].

    The parity folds the five rows with four XORs rather than reducing over the
    axis, and both neighbours are static rolls, so the whole step stays a
    handful of ops over the grid.
    """
    c = g[:, 0] ^ g[:, 1] ^ g[:, 2] ^ g[:, 3] ^ g[:, 4]  # [N, 5], by x
    d = fnp.roll(c, 1, axis=1) ^ _rotl1(fnp.roll(c, -1, axis=1))
    return g ^ d[:, None, :]


def _rho_pi(g, n: int):
    """Rotate every lane by its own offset, then move it to its pi destination.

    Rotating before the gather is what lets one shift pair cover all 25 lanes:
    the offset belongs to the source lane, so applying it first makes the
    movement a pure static reorder.
    """
    flat = g.reshape(n, N_LANES)
    rotated = (flat << _ROT) | (flat >> _ROT_INV)
    return rotated[:, _PI_PERM].reshape(n, 5, 5)


def _chi(g):
    """A[y][x] ^= (~A[y][x+1]) & A[y][x+2] — the only nonlinear step, and so the
    only source of AND rows. Returns the row's (z, a, b) contributions."""
    b1 = fnp.roll(g, -1, axis=2)
    b2 = fnp.roll(g, -2, axis=2)
    not_b1 = ~b1
    return not_b1 & b2, not_b1, b2


@frx.jit
def witness_keccak(state0):
    """Packed z/a/b witness streams for a batch of Keccak-f[1600] permutations.

    state0: uint64 [N, 25] (FIPS 202 flat lane order, `x + 5y`) -> three uint64
    [N, 1024] arrays (z, a, b).

    The three streams are built from one walk of the 24 rounds, so they cannot
    diverge: each round appends its own contribution to all three.
    """
    n = state0.shape[0]
    s = state0

    t_z, t_a, t_b = [], [], []
    for r in range(N_ROUNDS):
        bs = _rho_pi(_theta(s.reshape(n, 5, 5)), n)
        t, row_a, row_b = _chi(bs)
        t_z.append(t.reshape(n, N_LANES))
        t_a.append(row_a.reshape(n, N_LANES))
        t_b.append(row_b.reshape(n, N_LANES))
        s = (bs ^ t).reshape(n, N_LANES) ^ _RC_LANES[r]

    zeros = lambda k: fnp.zeros((n, k), dtype=fnp.uint64)  # noqa: E731
    ones = lambda k: fnp.full((n, k), _ONES64, dtype=fnp.uint64)  # noqa: E731
    one = fnp.ones((n, 1), dtype=fnp.uint64)

    def stream(state0_row, state24_row, t_rows):
        return fnp.concatenate(
            [
                state0_row,
                zeros(_SLOT_PAD_LANES),
                state24_row,
                zeros(_SLOT_PAD_LANES),
                one,
                *t_rows,
                zeros(_TAIL_PAD_LANES),
            ],
            axis=1,
        )

    return (
        stream(state0, s, t_z),
        stream(state0, s, t_a),
        stream(ones(N_LANES), ones(N_LANES), t_b),
    )
