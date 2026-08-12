# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Device-side flock Keccak-f[1600] R1CS witness generation (keccak, keccak3).

Emits the packed `z`/`a`/`b` bit-streams for a batch of Keccak permutations,
bit-identical to flock-prover's `r1cs_hashes::{keccak,keccak3}` witness
builders, the way `witgen` already does for BLAKE3. The prove consumes exactly
these buffers, so producing them on device removes the host generation pass and
its H2D crossing.

Layout, in whole u64 LANES rather than bit offsets — this is the structural
difference from BLAKE3. flock packs a keccak state into a 2,048-bit aligned
slot with an explicit zero pad instead of packing fields tightly, so nothing
straddles a word boundary and the "field list" is a lane map. For single keccak
(`KECCAK`, K_LOG 16):

    lanes [  0,   25)  state_0           lin-id
    lanes [ 25,   32)  slot zero pad     (a state is 1,600 of the slot's 2,048 bits)
    lanes [ 32,   57)  state_24          lin-id
    lanes [ 57,   64)  slot zero pad
    lane          64   the constant-1 wire, in bit 0
    lanes [ 65,  665)  t_r, 25 lanes per round, r in [0, 24)   AND
    lanes [665, 1024)  zero padding

keccak3 (`KECCAK3`, K_LOG 17) is three disjoint copies of that constraint set
with no chaining between them, so only the lane map widens: the slot pairs
repeat three times before the shared constant wire at lane 192, and the t rows
run to lane 1,993 grouped by sub-permutation first and round second. One
emitter covers both, parameterized by `Spec`.

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

from typing import NamedTuple

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

LANE_BITS = 64

# flock gives each state a 2,048-bit aligned slot: the sub-permutations' slots
# in pairs (state_0 then state_24), then the constant wire, then the t rows.
SLOT_BITS = 2048
SLOT_LANES = SLOT_BITS // LANE_BITS  # 32
SLOT_PAD_LANES = SLOT_LANES - N_LANES  # 7 — a 1,600-bit state in a 2,048-bit slot

_ONES64 = (1 << 64) - 1


class Spec(NamedTuple):
    """The lane map for one member of the keccak family.

    keccak and keccak3 differ only in how many independent permutations share a
    block. keccak3 is three disjoint copies of the single-keccak constraint set
    with no chaining between them, so the round math is untouched and only the
    lane map widens — which is why one emitter covers both.
    """

    n_sub: int
    k_log: int

    @property
    def words_per_block(self) -> int:
        return (1 << self.k_log) // LANE_BITS

    @property
    def z_const_lane(self) -> int:
        return 2 * self.n_sub * SLOT_LANES

    @property
    def t_lane_base(self) -> int:
        return self.z_const_lane + 1

    @property
    def useful_lanes(self) -> int:
        return self.t_lane_base + self.n_sub * N_ROUNDS * N_LANES

    @property
    def useful_bits(self) -> int:
        return self.useful_lanes * LANE_BITS

    def state0_lane(self, sub: int) -> int:
        return 2 * sub * SLOT_LANES

    def state24_lane(self, sub: int) -> int:
        return (2 * sub + 1) * SLOT_LANES

    def t_lane(self, sub: int, rnd: int) -> int:
        """t rows are grouped by sub-permutation first, then by round."""
        return self.t_lane_base + (sub * N_ROUNDS + rnd) * N_LANES


KECCAK = Spec(n_sub=1, k_log=16)
KECCAK3 = Spec(n_sub=3, k_log=17)

# The aligned slots are why single keccak reaches only 65% utilization, while
# keccak3 amortizes the pad over three states and reaches 97.3%.
assert (KECCAK.useful_bits, KECCAK.words_per_block) == (42560, 1024)
assert (KECCAK3.useful_bits, KECCAK3.words_per_block) == (127552, 2048)

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


def _witness(state0, spec: Spec):
    """Packed z/a/b streams for `spec`. state0: uint64 [N, n_sub, 25].

    The sub-permutations are disjoint, so the sub axis folds into the batch and
    all `N * n_sub` of them walk the rounds together — the round body stays
    single no matter how wide the block is.
    """
    n = state0.shape[0]
    rows = n * spec.n_sub
    s = state0.reshape(rows, N_LANES)

    t_z: list = []
    t_a: list = []
    t_b: list = []
    for r in range(N_ROUNDS):
        bs = _rho_pi(_theta(s.reshape(rows, 5, 5)), rows)
        t, row_a, row_b = _chi(bs)
        for acc, v in ((t_z, t), (t_a, row_a), (t_b, row_b)):
            acc.append(v.reshape(rows, N_LANES))
        s = (bs ^ t).reshape(rows, N_LANES) ^ _RC_LANES[r]

    zeros = lambda k: fnp.zeros((n, k), dtype=fnp.uint64)  # noqa: E731
    ones = lambda k: fnp.full((n, k), _ONES64, dtype=fnp.uint64)  # noqa: E731
    # Row `b * n_sub + sub` of the folded batch, so a sub's blocks are a stride.
    sub = lambda x, i: x[i :: spec.n_sub]  # noqa: E731
    tail = spec.words_per_block - spec.useful_lanes

    def stream(state_of, t_rows):
        """`state_of(sub, final)` supplies a slot's contents for this stream."""
        parts = []
        for i in range(spec.n_sub):
            parts += [
                state_of(i, False),
                zeros(SLOT_PAD_LANES),
                state_of(i, True),
                zeros(SLOT_PAD_LANES),
            ]
        parts.append(fnp.ones((n, 1), dtype=fnp.uint64))
        parts += [sub(t_rows[r], i) for i in range(spec.n_sub) for r in range(N_ROUNDS)]
        parts.append(zeros(tail))
        return fnp.concatenate(parts, axis=1)

    value = lambda i, final: sub(s, i) if final else state0[:, i, :]  # noqa: E731
    return (
        stream(value, t_z),
        stream(value, t_a),
        stream(lambda *_: ones(N_LANES), t_b),
    )


@frx.jit
def witness_keccak(state0):
    """Packed z/a/b witness streams for a batch of Keccak-f[1600] permutations.

    state0: uint64 [N, 25] (FIPS 202 flat lane order, `x + 5y`) -> three uint64
    [N, 1024] arrays (z, a, b).

    The three streams are built from one walk of the 24 rounds, so they cannot
    diverge: each round appends its own contribution to all three.
    """
    return _witness(state0[:, None, :], KECCAK)


@frx.jit
def witness_keccak3(state0):
    """Packed z/a/b witness streams for a batch of 3-wide Keccak blocks.

    state0: uint64 [N, 3, 25] -> three uint64 [N, 2048] arrays (z, a, b).

    Each block carries three independent permutations with no chaining between
    them, so this is `witness_keccak`'s walk over three times the rows, landed
    into the wider lane map.
    """
    return _witness(state0, KECCAK3)
