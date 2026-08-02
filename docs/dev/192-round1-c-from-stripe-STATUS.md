# #192 — round-1 C from the lincheck stripe: status & wiring plan

## What is proven (this branch)

`python/flock_zorch/zerocheck/testing/round1_c_from_stripe_test.py` (python-native,
CPU, 4/4 green) verifies with flock-zorch's **real** primitives that the round-1
folded-C vector equals a fold of the lincheck stripe:

```
folded_c[s] = Σ_x eq(r[k_skip:], x) · c_x[s]                      # today: _round1_partial C branch, 2^m drain
            = Σ_mid eq(r[k_skip:k_log])[mid] · z_vec[mid·2^k_skip + s]
  z_vec     = lincheck.partial_fold_packed_z(stripe(z), m, k_log, r[k_log:])
```

Because `round1_c = _urm._extend_folded_c(folded_c)` and the extend is fixed, a
bit-identical `folded_c` gives a **byte-identical `round1_c`** — the wire message
is unchanged, so the separate C witness drain (the `sum(where(c_rows, eqx, 0))`
map-reduce over all 2^m rows in `_round1_partial` / `_round1_core`, i.e. the #179
arena) is redundant when C is the identity.

### Why the factorization holds

`witness_to_rows` indexes `c_rows[chunk, s] = z[chunk·2^k_skip + s]` and the
stripe indexes `z[i_outer·2^k_log + i_inner]`. Writing `chunk = i_outer·
2^(k_log-k_skip) + mid` and `i_inner = mid·2^k_skip + s`, `eq(r[k_skip:])` factors
as `eq(r[k_skip:k_log])[mid] · eq(r[k_log:])[i_outer]` (build_eq is a bit-group
tensor product), which separates the sum into the stripe fold at `r[k_log:]` and
the middle fold at `r[k_skip:k_log]`.

## Remaining work (not on this branch)

1. **Thread the stripe to round 1.** The stripe is already built for lincheck
   (`lincheck.stripe_to_device`). Make it available to the zerocheck round-1 step
   (`zerocheck/prover.py` `_Round1`, `_urm.round1_rows`) when C is the identity.
2. **New folded-C source in `_round1_core`.** When the identity-C stripe is
   present, compute `folded_c` from `partial_fold_packed_z(stripe, m, k_log,
   r[k_log:])` + the middle fold above, and skip the `_round1_partial` C branch
   (the AB branch is untouched). `round1_ab` stays exactly as today.
3. **Assert-equal gate.** Behind a flag, compute both and assert the stripe path
   equals the drain before deleting the drain (mirrors the reference's
   `lincheck_stripe_c_fold4_matches_incumbent`).
4. **Byte-gate (mandatory, per CLAUDE.md).** No behavior change ships without the
   `*_oracle_test` proof gates green — the CPU `LigeritoProof` gate and the GPU
   identity-e2e + hash-circuit provers. Wire behind a kill switch until the full
   GPU set is confirmed.

## Reference

Flock-Challenge `Layr-Labs/flock-challenge` `1a6ad0e7f3`
(`round1_c_fold4_from_lincheck_stripe`). flock-core folds in two stages
(stripe `partial_fold_packed_z_best` + `s_hat_v_fold4`/`collapse`); flock-zorch's
C path is the single-stage `folded_c` above, so the port is the fold identity
here, not a line-by-line Fold4 transcription.
