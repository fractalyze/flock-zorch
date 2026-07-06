"""flock's Ligerito recursive PCS, authored in jax — byte-identical to flock-core
`pcs::ligerito`. The PCS flock's headline hash benchmarks use: instead of a single
codeword folded over FRI rounds (BaseFold), it RE-COMMITS the partially-folded
witness at R recursive levels (shrinking RS rate), driving one continuous
interleaved sumcheck (fold / introduce / glue) with per-level OOD binding, query
PoW, and a plaintext residual. Proof scales ~log²(log_n).

Reuses flock-zorch's byte-identical primitives: `merkle` (tree/multi_proof + host
SHA-NI), `sumcheck` (build_eq / fold), `basefold` (the shared round message),
`challenger` (SHA-256 FS + grind_pow). The per-level RS encode is zorch's
`coding.ReedSolomon` over the `binary_field_ghash` dtype — `lax.ntt` dispatches
the additive (LCH, GHASH-basis) transform, byte-identical to flock's
`forward_transform_interleaved` per lane (flock-zorch#11).

The full recursive driver lives here: `ligero_commit` (per-level RS-encode +
Merkle), `induce_sumcheck_poly` + the LCH novel basis, the `SumcheckProver`
interleaved fold/introduce/glue, and `recursive_prover_with_basis` (the recursion
loop, producing a byte-identical LigeritoProof).
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import lax

from zorch.coding.reed_solomon import ReedSolomon

from flock_zorch import field, merkle, sumcheck, basefold
from flock_zorch import _hostfield as hf
from flock_zorch.field import _to_lohi


def eval_sk_at_vks(log_n: int):
    """flock `eval_sk_at_vks`: sks_vks[k] = s_k(v_k), k=0..log_n (standard basis
    v_i=2^i). Scalar host recurrence next_s(s,root)=s²+root·s. Returns list[int]."""
    sks = [0] * (log_n + 1)
    sks[0] = 1
    if log_n == 0:
        return sks
    layer = [1 << i for i in range(1, log_n + 1)]
    cur = log_n
    for i in range(log_n):
        for j in range(cur):
            v = hf.add(hf.sqr(layer[j]), hf.mul(sks[i], layer[j]))  # next_s(layer[j], sks[i])
            if j == 0:
                sks[i + 1] = v
            else:
                layer[j - 1] = v
        cur -= 1
    return sks


def induce_sumcheck_poly(log_msg_cols: int, sks_vks, opened_rows, v_challenges, queries,
                         alpha, mul=field.mul):
    """flock `induce_sumcheck_poly` (dense, by design): from Q opened rows + the level's fold
    challenges, build basis_poly[j] = Σ_i α_i·Ŵ_j(q_i) (LCH normalized novel basis,
    a per-query tensor ⊗_k(1, Ŵ_k(q_i))) and enforced_sum = Σ_i α_i·⟨row_i, eq(v)⟩.
    α weights = build_eq(alpha)[:Q] (multilinear, not Vandermonde). Vectorized over
    queries on device. Returns (basis_poly uint64 [2^log_msg_cols,2], enforced [2])."""
    nq = len(queries)
    eq = sumcheck.build_eq(jnp.asarray(v_challenges), mul=mul)            # [num_int, 2]
    alpha_pows = sumcheck.build_eq(jnp.asarray(alpha), mul=mul)[:nq]      # [nq, 2]
    rows = jnp.asarray(np.stack([np.asarray(r).reshape(-1, 2) for r in opened_rows]))  # [nq,num_int,2]
    dot = field.sum(mul(rows, eq[None, :, :]), axis=1)         # [nq, 2]
    enforced = field.sum(mul(alpha_pows, dot), axis=0)         # [2]

    # Ŵ_k(q_i) = s_k(q_i)/s_k(v_k), per query (recurrence over k), vectorized over i.
    inv_sks = [hf.inv(v) if v != 0 else 0 for v in sks_vks]
    sx = jnp.asarray(np.stack([_to_lohi(int(q)) for q in queries]))      # [nq,2]  q_field=F128(q,0)
    sx_list = [sx]
    for i in range(1, log_msg_cols):
        root = jnp.asarray(_to_lohi(sks_vks[i - 1]))[None, :]
        sx = field.add(mul(sx, sx), mul(root, sx))                       # next_s
        sx_list.append(sx)
    w = [mul(sx_list[i], jnp.asarray(_to_lohi(inv_sks[i]))[None, :]) for i in range(log_msg_cols)]

    # basis[j] = Σ_i α_i · ∏_{k:bit_k(j)} Ŵ_k(q_i) — build_eq-style doubling per query.
    t = alpha_pows[:, None, :]                                           # [nq,1,2]
    for k in range(log_msg_cols):
        t = jnp.concatenate([t, mul(w[k][:, None, :], t)], axis=1)
    basis = field.sum(t, axis=0)                              # [2^log_msg_cols, 2]
    return np.asarray(basis), np.asarray(enforced)


def ceil_log2(x: int) -> int:
    return (x - 1).bit_length()


def sample_distinct_queries(ch, block_len: int, count: int) -> list[int]:
    """flock `sample_distinct_queries`: sample F128, take (v.lo % block_len) until
    `count` distinct, then sort ascending. Same challenger draw order as flock."""
    assert count <= block_len
    seen, out = set(), []
    while len(out) < count:
        v = ch.sample_f128()
        q = int(v[0]) % block_len
        if q not in seen:
            seen.add(q); out.append(q)
    out.sort()
    return out


class SumcheckProver:
    """flock `pcs::ligerito::SumcheckProver` — the stateful interleaved sumcheck
    driver over (f, combined_basis, t_r). Round message (u_0,u_2) == basefold._round_message
    (LSB even/odd split); fold == sumcheck.fold_single (LSB). introduce_new stages a
    fresh basis; glue folds it into combined_basis with separation α."""

    def __init__(self, f, b1, h1, mul):
        self.mul = mul
        self.f = jnp.asarray(f)
        self.combined_basis = jnp.asarray(b1)
        self.t_r = np.asarray(h1, np.uint64).reshape(2)
        self.transcript: list = []
        self._pending = None

    def _msg(self, a, b):
        u0, u2 = basefold._round_message(a, b, self.mul)
        m = (np.asarray(u0), np.asarray(u2))
        self.transcript.append(m)
        return m

    @classmethod
    def new(cls, f, b1, h1, mul):
        s = cls(f, b1, h1, mul)
        return s, s._msg(s.f, s.combined_basis)

    @classmethod
    def new_with_first_msg(cls, f, b1, h1, first_msg, mul):
        s = cls(f, b1, h1, mul)
        s.transcript.append(first_msg)
        return s, first_msg

    def fold(self, r):
        rj = jnp.asarray(r)
        self.f = sumcheck.fold_single(self.f, rj, self.mul)
        self.combined_basis = sumcheck.fold_single(self.combined_basis, rj, self.mul)
        return self._msg(self.f, self.combined_basis)

    def introduce_new(self, b_new, h_new):
        bn = jnp.asarray(b_new)
        msg = self._msg(self.f, bn)
        self._pending = (bn, np.asarray(h_new, np.uint64).reshape(2))
        return msg

    def introduce_new_with_eval(self, b_new):
        bn = jnp.asarray(b_new)
        h_new = np.asarray(field.sum(self.mul(self.f, bn), axis=0))  # Σ f·b_new
        msg = self._msg(self.f, bn)
        self._pending = (bn, h_new)
        return msg, h_new

    def glue(self, alpha):
        bn, h_new = self._pending
        self._pending = None
        a = jnp.asarray(alpha)
        self.combined_basis = field.add(self.combined_basis, self.mul(a, bn))
        self.t_r = np.asarray(field.add(jnp.asarray(self.t_r), self.mul(a, jnp.asarray(h_new))))


def _observe_msg(ch, msg):
    ch.observe_f128(msg[0]); ch.observe_f128(msg[1])


def recursive_prover_with_basis(config, packed_witness, b_initial, target, l0_codeword,
                                l0_tree, ch, mul=field.mul, use_host_sha=False, first_msg=None):
    """flock `recursive_prover_with_basis_impl` — the Ligerito recursion driver.
    config: dict with log_inv_rates, recursive_steps, initial_k, recursive_ks, queries,
    grinding_bits, fold_grinding_bits, ood_samples. l0_codeword/l0_tree = external L0
    commit (pcs_commit at lbs=initial_k, lir=log_inv_rates[0]). Returns a LigeritoProof
    dict byte-identical to flock."""
    lir = config["log_inv_rates"]; R = config["recursive_steps"]; ik = config["initial_k"]
    rks = config["recursive_ks"]; qcfg = config["queries"]
    grind = config["grinding_bits"]; foldg = config["fold_grinding_bits"]; ood = config["ood_samples"]
    log_n = int(round(float(np.log2(np.asarray(packed_witness).shape[0]))))

    ch.observe_label(b"flock-ligerito-basis-v0")
    ch.observe_f128(target)
    initial_root = np.asarray(l0_tree[-1])
    l0_num_int = 1 << ik
    l0_block_len = np.asarray(l0_codeword).shape[0] // l0_num_int
    ch.observe_bytes(bytes(np.asarray(initial_root, np.uint8)))

    ood_values, fold_nonces = [], []
    if first_msg is None:
        sc, start = SumcheckProver.new(packed_witness, b_initial, target, mul)
    else:
        sc, start = SumcheckProver.new_with_first_msg(packed_witness, b_initial, target, first_msg, mul)
    _observe_msg(ch, start)

    # ---- initial_k lane folds ----
    r_lane = []
    for j in range(ik):
        bits = max(foldg[0] - j, 0)
        if bits > 0:
            fold_nonces.append(ch.grind_pow(bits))
        r = ch.sample_f128(); _observe_msg(ch, sc.fold(r)); r_lane.append(r)

    # ---- commit f^1 ----
    n1 = log_n - ik
    mat_prev, tree_prev = ligero_commit(sc.f, n1 - rks[0], rks[0], lir[1], use_host_sha)
    bl_prev, ni_prev = mat_prev.shape[0] // (1 << rks[0]), (1 << rks[0])
    ch.observe_bytes(bytes(np.asarray(tree_prev[-1], np.uint8)))
    recursive_roots = [np.asarray(tree_prev[-1])]

    # ---- OOD for L1 ----
    for _ in range(ood[1]):
        z = ch.sample_f128_vec(n1); eqz = sumcheck.build_eq_fused(jnp.asarray(z), mul=mul)
        msg, y = sc.introduce_new_with_eval(eqz)
        ch.observe_f128(y); ood_values.append(np.asarray(y)); _observe_msg(ch, msg)
        sc.glue(ch.sample_f128())

    # ---- L0 query open + induce + introduce/glue ----
    grinding_nonces = [ch.grind_pow(grind[0])]
    q0 = sample_distinct_queries(ch, l0_block_len, qcfg[0])
    alpha0 = ch.sample_f128_vec(ceil_log2(qcfg[0]))
    l0cw = np.asarray(l0_codeword).reshape(l0_block_len, l0_num_int, 2)
    opened0 = [l0cw[q] for q in q0]
    mp0 = merkle.merkle_multi_proof(np.asarray(l0_tree), l0_block_len, q0)
    initial_proof = {"opened_rows": opened0, "merkle_proof": mp0}
    basis0, esum0 = induce_sumcheck_poly(n1, eval_sk_at_vks(n1), opened0, r_lane, q0, alpha0, mul)
    _observe_msg(ch, sc.introduce_new(basis0, esum0)); sc.glue(ch.sample_f128())

    # ---- recursive levels ----
    recursive_proofs = []
    for i in range(R):
        k_i = rks[i]; level_rs = []
        for j in range(k_i):
            bits = max(foldg[i + 1] - j, 0)
            if bits > 0:
                fold_nonces.append(ch.grind_pow(bits))
            ri = ch.sample_f128(); _observe_msg(ch, sc.fold(ri)); level_rs.append(ri)

        if i == R - 1:                                   # final level: residual yr
            yr = np.asarray(sc.f)
            for v in yr:
                ch.observe_f128(v)
            grinding_nonces.append(ch.grind_pow(grind[i + 1]))
            ql = sample_distinct_queries(ch, bl_prev, qcfg[i + 1])
            pcw = mat_prev.reshape(bl_prev, ni_prev, 2)
            final_proof = {"yr": yr, "opened_rows": [pcw[q] for q in ql],
                           "merkle_proof": merkle.merkle_multi_proof(tree_prev, bl_prev, ql)}
            return {"initial_root": initial_root, "initial_proof": initial_proof,
                    "recursive_roots": recursive_roots, "recursive_proofs": recursive_proofs,
                    "final_proof": final_proof, "sumcheck_transcript": sc.transcript,
                    "grinding_nonces": grinding_nonces, "ood_values": ood_values,
                    "fold_grinding_nonces": fold_nonces}

        # non-final: commit next level, OOD, open prev, induce, introduce/glue
        n_next = int(round(float(np.log2(np.asarray(sc.f).shape[0]))))
        lni_next = rks[i + 1]
        mat_next, tree_next = ligero_commit(sc.f, n_next - lni_next, lni_next, lir[i + 2], use_host_sha)
        ch.observe_bytes(bytes(np.asarray(tree_next[-1], np.uint8)))
        recursive_roots.append(np.asarray(tree_next[-1]))
        for _ in range(ood[i + 2]):
            z = ch.sample_f128_vec(n_next); eqz = sumcheck.build_eq_fused(jnp.asarray(z), mul=mul)
            msg, y = sc.introduce_new_with_eval(eqz)
            ch.observe_f128(y); ood_values.append(np.asarray(y)); _observe_msg(ch, msg)
            sc.glue(ch.sample_f128())
        grinding_nonces.append(ch.grind_pow(grind[i + 1]))
        qi = sample_distinct_queries(ch, bl_prev, qcfg[i + 1])
        alphai = ch.sample_f128_vec(ceil_log2(qcfg[i + 1]))
        pcw = mat_prev.reshape(bl_prev, ni_prev, 2)
        opened_i = [pcw[q] for q in qi]
        recursive_proofs.append({"opened_rows": opened_i,
                                 "merkle_proof": merkle.merkle_multi_proof(tree_prev, bl_prev, qi)})
        basisi, esumi = induce_sumcheck_poly(n_next, eval_sk_at_vks(n_next), opened_i, level_rs, qi, alphai, mul)
        _observe_msg(ch, sc.introduce_new(basisi, esumi)); sc.glue(ch.sample_f128())
        mat_prev, tree_prev, bl_prev, ni_prev = mat_next, tree_next, mat_next.shape[0] // (1 << lni_next), (1 << lni_next)

    raise RuntimeError("unreachable")


def ligero_commit(poly, log_msg_cols: int, log_num_interleaved: int, log_inv_rate: int,
                  use_host_sha: bool = False):
    """Per-level Ligero commit (flock `pcs::ligerito::ligero_commit`): reshape `poly`
    (len num_interleaved·msg_cols, SoA `poly[col·num_interleaved + lane]`) into a
    block_len × num_interleaved matrix, RS-encode each lane with zorch's
    `coding.ReedSolomon` over `binary_field_ghash` (`lax.ntt` dispatches the
    additive GHASH-basis transform — byte-identical to flock's zero-pad +
    forward-transform), and Merkle-commit the rows (leaf = num_interleaved F128).
    Returns (mat, tree).

    The uint64 (lo, hi) SoA lanes bitcast to the 128-bit dtype on device; the
    codeword returns to SoA through host bytes (the device ghash→uint64 bitcast
    direction currently returns zeros; the codeword is materialized for the
    Merkle leaves right after anyway).

    mat: uint64 [block_len·num_interleaved, 2] (SoA); tree: uint8 [2·block_len-1, 32]."""
    msg_cols = 1 << log_msg_cols
    num_int = 1 << log_num_interleaved
    block_len = msg_cols << log_inv_rate

    code = ReedSolomon(message_len=msg_cols, blowup=1 << log_inv_rate,
                       dtype=jnp.binary_field_ghash)
    x = lax.bitcast_convert_type(
        jnp.asarray(poly).reshape(msg_cols, num_int, 2), jnp.binary_field_ghash)
    cw = code.encode(x.T)  # [num_int, block_len]
    mat = np.frombuffer(
        np.ascontiguousarray(np.asarray(cw).T).tobytes(), np.uint64,
    ).reshape(block_len * num_int, 2)
    leaves = mat.reshape(block_len, num_int * 2).view(np.uint8)
    tree = merkle.merkle_tree(leaves, use_host_sha=use_host_sha)
    return mat, tree
