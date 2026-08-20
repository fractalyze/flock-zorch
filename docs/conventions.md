# Conventions

How this repo models the protocol: what a claim may state, where
instance-varying data lives, and which steps are roles rather than functions.
These were derived in
[sp1-zorch#301](https://github.com/fractalyze/sp1-zorch/pull/301) — each from a
concrete defect found while porting a prover onto zorch's composition roles —
and are recorded here so they hold without being rediscovered.

zorch's definitions of stage, round, committer and shared function are **not**
restated here. They live in
[`docs/composition/stage-composition.md`](https://github.com/fractalyze/zorch/blob/main/docs/composition/stage-composition.md);
its "Which one is it?" table is the decision procedure. What follows is only
what is flock-zorch's.

## 1. A claim states a proposition

A `*Claim` docstring opens with a sentence that could be true or false. Read
down `python/flock_zorch/prover.py`, the composition states itself as a chain of
reductions: `R1csClaim` (ẑ satisfies A·ẑ ∘ B·ẑ = C·ẑ) → `ZerocheckClaim`
(â, b̂, ĉ evaluate to these values at the challenge point) → `BatchOpeningClaim`
(ẑ opens to the ab and c values at these two points) → nothing left to prove.

`ZerocheckProof` holds the wire fields alone. The evaluation point it is about
is `ZerocheckClaim`, which `prove_packed` returns alongside it and the verifier
re-derives from the proof — one type, produced by both roles, living in
`zerocheck/types.py` so neither role has to import the other. It previously
carried a second copy of `z` / `mlv_challenges` / `r_rest`, which is what let
the two disagree; the zerocheck verify test now compares the two roles' claims
against each other rather than a claim against a proof.

## 2. The claim carries what varies per instance; the roles carry what does not

`R1csClaim` holds the statement digest — the thing that identifies *which* R1CS
is claimed and the thing the transcript binds. The matrix dimensions (`m`,
`k_log`, `k_skip`), the Ligerito config and the lincheck circuit are fixed by
the circuit and configure the roles.

## 3. Hold values, not their serialization

Never store a pre-serialized transcript blob as claim data. `bind_statement`
derives its absorb stream from the claim's digest and the commitment root, so
the transcript's view and the structural view cannot disagree.

## 4. Not everything in the sequence is a stage or a round

See zorch's table for the definitions. In this repo:

- `bind_statement` is a **shared function** both roles call — it only absorbs,
  and owns no proof section;
- the zerocheck and the lincheck are **stages**;
- the Ligerito commit is the PCS's commit half, not a stage — it runs before any
  claim exists.

The sub-protocol step sequences (`zerocheck_steps`, `lincheck_steps` and their
verify siblings) are heterogeneous steps threading a carry, driven by
`prove_rounds` / `verify_rounds`. They are **not** recurrences, so they are not
rounds in zorch's sense; the driver simply happens to be the right shape for a
carry-threading sequence. `InfProductRound` *is* a genuine round — one repeated
sumcheck transition.

`_CombRound` is the near-miss worth knowing about: it emits no proof message,
which is the shared-function signature, but it is *not* one. Its verifier dual
covers only the dense path — the Python verifier takes no `circuit` at all — so
the prover's `CscCircuit` branch, including the `const_pin` +β draw, has no
counterpart to share with. It stays a step, and the asymmetry is a scope limit
of the Python verifier rather than a naming problem.

`sample_challenge_coords` is the worked example of the rule: it draws the
zerocheck's challenge coordinates, emits no proof message, and owns no proof
section, so it is a function both roles call rather than a step in either
sequence. Each role keeps a different slice of the draw — the prover the whole
vector, the verifier only the tail — which is exactly the shape that invites two
copies of one Fiat-Shamir schedule, and exactly why it is written once.

## 5. The PCS's commit and open are two halves of one role

`FlockLigeritoPcs` owns both. They sit far apart because Fiat-Shamir requires
it — the commitment must bind the transcript before the zerocheck draws a
challenge, and the open needs the points the lincheck produces.
`LigeritoCommitData` names what crosses between them.

That data is **not** prover-only: the root is bound into the transcript by
`bind_statement`, and the opened columns and Merkle paths ride the proof.

## 6. Point at zorch for zorch's concepts

A doc here covers only what is flock's: the GHASH-basis field, the round-1 URM,
the ∞-trick round loop, F128↔bytes serialization. Anything zorch defines is
linked, not copied — local copies drift silently, as the retired `ProveChain` /
`Bridge` vocabulary did.

## 7. Make confusable states unrepresentable

Keyword-only fields on any type whose field shape has a twin. `BatchOpeningClaim`
is `kw_only` because `ab_point` / `c_point` and `ab_value` / `c_value` are two
interchangeable-looking pairs, and crossing them would swap which claim is opened
where with no type error.

## 8. Do not mirror an upstream name when it collides locally

flock-core's names are kept where they do not collide, because the proof gates
diff against that source. Where a borrowed name would mean something else here,
prefer the local name and record the mapping at the serializer, where the wire
contract lives.

## 9. Enforce with tooling, and know what the tooling misses

`pre-commit` runs black, ruff and mypy over the whole package. mypy is what
catches a crossed claim type or an unchecked optional — it flagged the
`LincheckProof.claim` optional the moment the composite stopped asserting on it.

**Rule 9's `py_binary` gap does not apply here.** flock-zorch has no `py_binary`
targets: every oracle gate is a `py_test`, so `bazel test //python/...` really
does run them. Keep it that way — a byte-match harness added as a runnable would
silently stop being executed.

Note the GPU set is not in the CPU suite; the non-negotiables in
[`development.md`](development.md) still require the GPU proof gates green before a
behavior change ships.
