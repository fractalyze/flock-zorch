//! Golden dumper for the snark.fast BENCHMARK profile — the flock-challenge
//! harness's exact timed section (BLAKE3 Fiat-Shamir on `b"flock-bench-v0"`,
//! BLAKE3 Merkle), driven through the CHALLENGE FORK's own crates (the renamed
//! `flock-challenge-*` deps), not the upstream flock pin: the fork is what the
//! harness verifies with, so a proof the GPU prover must byte-match has to
//! come from the fork.
//!
//! Mirrors `benchmark-tools/worker`'s timed body (seed → blocks →
//! `prove_fast` → `R1csProofBundleLigerito::to_bytes`), then runs the
//! harness's `verify_proof` acceptance on the serialized bytes — trusted
//! commitment recompute from the seed, params pin, fork `setup.verify` —
//! before writing, so the dumped golden is fork-verifier-accepted by
//! construction. The defaults (log2=8 → m=22, seed=42) are the standing GPU
//! gate's instance (`testing/bench_ligerito_oracle_test.py`).
//!
//! Usage: `cargo run --release --example dump_bench_ligerito -- [log2] [seed] [out]`

use flock_benchmark_common::{generate_compressions, DOMAIN};
use flock_challenge_prover::challenger::FsChallenger;
use flock_challenge_prover::merkle::HashKind;
use flock_challenge_prover::pcs;
use flock_challenge_prover::proof_io::R1csProofBundleLigerito;
use flock_challenge_prover::r1cs_hashes::blake3::Blake3Setup;

const BENCHMARK_HASH: HashKind = HashKind::Blake3;

fn main() {
    let mut a = std::env::args().skip(1);
    let log2: u32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let seed: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(42);
    let out = a
        .next()
        .unwrap_or_else(|| "artifacts/bench_ligerito_golden.bin".to_string());
    assert!((8..=20).contains(&log2), "harness worker contract: log2 in 8..=20");

    // The worker's timed body, verbatim.
    let mut setup = Blake3Setup::new(1usize << log2);
    setup.pcs_params.merkle_hash = BENCHMARK_HASH;
    let blocks = generate_compressions(log2, seed);
    let mut challenger = FsChallenger::with_hash(DOMAIN, BENCHMARK_HASH);
    let (proof, commitment, _) = setup.prove_fast(&blocks, &mut challenger);
    let bundle = R1csProofBundleLigerito { commitment, proof };
    let bytes = bundle.to_bytes();

    // The harness's `verify_proof`, on the serialized bytes (so a
    // serialization slip fails here, not at gate time): parse back, recompute
    // the trusted commitment from the seed, pin the params, run the fork
    // verifier on a fresh challenger.
    let parsed = R1csProofBundleLigerito::from_bytes(&bytes).expect("from_bytes");
    let witness = setup.generate_witness_packed(&blocks);
    let (expected, _) = pcs::commit(&witness, &setup.pcs_params);
    assert_eq!(parsed.commitment.root, expected.root, "root vs trusted witness");
    assert_eq!(parsed.commitment.params.m, setup.pcs_params.m);
    assert_eq!(parsed.commitment.params.log_inv_rate, setup.pcs_params.log_inv_rate);
    assert_eq!(parsed.commitment.params.log_batch_size, setup.pcs_params.log_batch_size);
    assert_eq!(parsed.commitment.params.profile, setup.pcs_params.profile);
    assert_eq!(parsed.commitment.params.merkle_hash, BENCHMARK_HASH);
    let mut vch = FsChallenger::with_hash(DOMAIN, BENCHMARK_HASH);
    setup
        .verify(&parsed.commitment, &parsed.proof, &mut vch)
        .expect("fork verifier rejected the bundle");

    std::fs::write(&out, &bytes).unwrap();
    println!(
        "dumped bench-profile bundle log2={log2} seed={seed} m={} bytes={} (VERIFY PASS) -> {out}",
        setup.pcs_params.m,
        bytes.len()
    );
}
