# Result: the m26 CPU tier against the Yukon x86 frontier

The published record of one measured pair: the FRX CPU tier and the frontier
submission on Yukon's `eigenlabs/flock-challenge-multi/x86` track, proving the
same m26 instance under the official harness, on one machine, in one session.
Two names appear throughout and both are right:
`eigenlabs/flock-challenge-multi/x86` is the Yukon leaderboard identifier,
while `Layr-Labs/flock-challenge-multi` is the GitHub repository the harness
and the submission sources are cloned from.

[`measurement.md`](measurement.md) holds the durable rules for producing a
number this repo will believe. This file is the dated record of a single run,
which is why it is a separate document: the rules stay true, a result does not.

## The pair

Both arms ranked under the harness's timing contract — 20 warm-up trials
discarded, 100 measured trials in fresh worker processes, score = 4096
compressions / median trial seconds, every proof checked by the pinned trusted
verifier. Measured 2026-09-10, back to back, `taskset -c 0-15` (16 physical
cores), 16 threads.

| arm | score (median of 100) | median | min-of-100 | max/min | p90/p10 | verified |
|---|---:|---:|---:|---:|---:|---:|
| Yukon x86 frontier `e1a16581` | **538,547 comp/s** | 7.606 ms | 613,307 comp/s (6.679 ms) | 1.278 | 1.178 | 120/120 |
| flock-zorch FRX CPU tier | **7,299 comp/s** | 561.150 ms | 7,980 comp/s (513.278 ms) | 1.261 | 1.089 | 120/120 |

**The frontier is 73.8x above the FRX CPU tier on this machine.** The
criterion this measurement was taken against — the FRX tier above the
frontier's same-machine score, with margin — is **not met**.

Stated alongside, as context rather than as the contract. Everything below is
an **in-process** prove, which is a different measurement domain from the
ranked scores above: the ranked window is a fresh worker per trial timed from
seed to proof rename. The two sets of ratios are therefore not expected to
agree, and a figure from one domain must not be divided by a figure from the
other.

- **This session's in-process prove wall: 489.6 ms at its minimum**
  (8,366 comp/s), three rounds of n=5 unbarriered proves, `taskset -c 0-15`.
  Per round — min 489.6 / 556.6 / 490.0 ms, median 540.3 / 577.1 / 495.1 ms,
  spread 15.4 / 7.0 / 12.6 %. Startup is excluded. A min and a median, not a
  ranked score.
- **The reachable ceiling is 290,633 comp/s (14.09 ms), and it is an inherited
  figure** — measured by #333 on 2026-09-09, on this box and with these pins,
  and not re-measured here. It is upstream `succinctlabs/flock@85fc0e7`, this
  repo's own golden dependency, proving the same m26 instance: a measured
  floor for the byte-identical protocol rather than an extrapolation. #333
  records it as a best-of-3 in-process prove and publishes no spread for it,
  so it is a min without one.

Against that ceiling, in the in-process domain where it was measured, using
#333's own third arm for the frontier so all three rows are like-for-like:

| arm | in-process prove | comp/s | source |
|---|---:|---:|---|
| flock-zorch FRX CPU tier | 489.6 ms (min of 5) | 8,366 | this session |
| reference flock `85fc0e7` | 14.09 ms (best of 3) | 290,633 | #333, inherited |
| Yukon frontier `e1a16581` | 8.258 ms (min of 20) | 496,010 | #333, inherited |

**34.7x** from the CPU tier to the reference ceiling, then **1.71x** from that
ceiling to the frontier. The first is the part compiler and prover work can
address. The second is ranked engineering plus one freedom this project does
not have: the harness's trusted verifier is built from the submission's own
editable sources, so a submission may co-evolve prover and verifier, while
flock-zorch's first non-negotiable is byte-identity to upstream flock. The
full per-phase account is #333's, posted on
[#323, comment 5596452109](https://github.com/fractalyze/flock-zorch/issues/323#issuecomment-5596452109);
§6 is the ceiling statement.

Against #322's ranked m26 pair, which measured its two arms a day apart at
519,852 and 3,022 comp/s (172x): the frontier arm has barely moved (538,547
here, +3.6 %, on a far tighter spread — p90/p10 1.178 against 3.80), and the
FRX arm is **2.42x** faster on the wheel carrying fractalyze/xla#679 and
fractalyze/xla#687. That 2.42x is a cross-session comparison whose denominator
is #322's inherited 3,022, not a same-session measurement. The **73.8x** above
is same-session on both arms.

## Method

- **Harness.** `Layr-Labs/flock-challenge-multi` at `c75aece`, track `x86`.
  The trusted verifier owns the private seed, the external timer, proof
  checking and score writing, and launches one fresh worker per trial. Its
  bytes were checked against the committed `SHA256SUMS` before each arm.
- **Worker, FRX arm.** `scripts/bench_worker_cpu.sh` (`FRX_PLATFORMS=cpu`),
  whose timed body is `_bench_profile.prove_bundle` — the same body
  `bench_ligerito_oracle_test.py` byte-gates against a fork-verified bundle.
- **Worker, frontier arm.** The submission's own
  `flock-benchmark-worker`, rebuilt before the run by the harness's locked
  offline recipe verbatim: `cargo +1.97.0 build --locked --offline --profile
  challenge`, `RUSTFLAGS=-C target-cpu=native`.
- **Deviation from the ranked contract — the worker is not sandboxed.**
  `bwrap` fails on this box (`kernel.apparmor_restrict_unprivileged_userns=1`,
  and changing it needs root), so both arms were run through the verifier's
  8-argument path, which is the harness's own tested `sandbox_scratch: None`
  local-dev case rather than a patch or a `PATH` shim. Everything else — the
  `SHA256SUMS` check, the locked offline rebuild, the 20/100 trial contract —
  ran verbatim. Both arms carry the deviation equally, so the pair stays
  like-for-like, but neither number is directly comparable to a leaderboard
  row.

## Machine

`build-server`: AMD Ryzen 9 9950X (Zen 5, 16 physical cores / 32 SMT threads,
AVX-512, VPCLMULQDQ, GFNI), 60 GB RAM, `performance` governor on
`amd-pstate-epp`. Instantaneously idle by `vmstat` before each arm. Pinned to
physical cores, which beat all 32 SMT threads and are steadier.

## Provenance

| what | pin |
|---|---|
| flock-zorch | `0a0e9d7` |
| frx / frxlib / frx-cuda12-plugin / frx-cuda12-pjrt | `0.10.2.dev20260909121611` |
| zorch (`MODULE.bazel` `git_override`) | `e8a9a14` |
| zk-dtypes / hash-frx | `0.0.17` / `0.2.0.dev20260820080354` |
| harness + frontier sources | `flock-challenge-multi` `c75aece` |
| frontier submission id | `e1a16581-883b-4459-8af3-bbdb3e0b2ea1` |
| trusted verifier sha256 | `5ad0acfa59a6f3415061b0536d401075b7e7c71da5ec1e5d3d8784bd81d68798` |

The frontier was re-read on the day of the run: `main` on
`flock-challenge-multi` is the promotion chain, and it still points at
`e1a16581`. Six submissions were validated on 2026-09-09, but each branches
from the current `main` and none was promoted, so the frontier had not moved
since #322 measured it. The Yukon CLI is neither installed on this box nor on
PyPI; a submission is a plain branch, and its tip commit message names the
submission id, which is what makes the identity checkable.

The frx wheel carries the two CPU fixes this measurement was sequenced behind,
proven present rather than assumed from the version date: fractalyze/xla#679 by
its pass name `cpu-scatter-private-accumulators` in `frxlib/libjax_common.so`
(against the control `cpu-parallel-task-assigner`, which predates it), and
fractalyze/xla#687 by provenance, since it adds no string that reaches the
binary — the wheel's release tag pins `XLA_COMMIT = e9e924522f0f`, which is
that fix's own merge commit.

Byte gates green on the measured wheel: `bazel test //python:all` 29/29
including `//python:ligerito_oracle_test`, and `bench_ligerito_oracle_test.py`
byte-identical to the fork-verified bundle.

## What this pair does not say

- **It is not a leaderboard row.** The challenge cannot score an FRX prover:
  only `crates/flock-{core,prover}/src` are editable, so no Python runtime can
  enter a submission. This is a measured claim on one machine, not a ranking.
- **The ratio does not transfer.** It is a property of this machine and this
  size. This box runs the frontier submission well below its score on Yukon's
  c7i.4xlarge, so the ratio is taken against a handicapped frontier; and it
  moves with `m`, because the FRX arm carries fixed per-process floors that
  amortize as `m` grows.
- **Startup is not in it.** The harness starts its clock when the seed arrives
  and stops it when the proof file is renamed, so the per-trial worker startup
  (~15 s in-process here, against a 300 s readiness budget) is outside the
  timed window for both arms and is not part of either score.

## Reproduce

```bash
# FRX arm — from a flock-zorch checkout, venv built per README, m26 compile
# cache warmed once outside the readiness budget:
echo 42 | scripts/bench_worker_cpu.sh 12 /tmp/warm.ready /tmp/warm.proof

# Both arms, from a flock-challenge-multi checkout at the frontier commit.
# 8 arguments = the harness's unsandboxed local-dev path; a 9th would be the
# sandbox scratch dir.
( cd benchmark-tools/trusted && sha256sum -c SHA256SUMS )
taskset -c 0-15 ./benchmark-tools/trusted/flock_benchmark_verifier \
    <WORKER> <SCRATCH> score.json summary.md 12 16 20 100
```

`<WORKER>` is `target/challenge-candidate/challenge/flock-benchmark-worker`
for the frontier arm and `<flock-zorch>/scripts/bench_worker_cpu.sh` for the
FRX arm. `12` is the harness `log2`; `m = log2 + 14`.
