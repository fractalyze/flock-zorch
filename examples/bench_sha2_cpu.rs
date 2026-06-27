//! CPU baseline for the GPU SHA-256 prover: times flock's BaseFold `prove` on the
//! SAME real sha2 R1CS (Sha256HybridSetup) the GPU prover gates against, so
//! e2e_sha2_bench's speedup is apples-to-apples on the same instance + same
//! (BaseFold) backend. x86 scalar, matched release profile.
//!
//! NOTE: flock's HEADLINE sha2 number (sha2_proof.rs) is the LIGERITO prover;
//! this is the BaseFold column (prove_fast_basefold / sha2_lig_vs_bf), which is
//! what the flock-zorch GPU port reproduces.
//!
//! Usage: `cargo run --release --example bench_sha2_cpu -- [n_compressions ...]`

use std::time::Instant;

use flock_core::challenger::FsChallenger;
use flock_core::pcs::pack::pack_witness;
use flock_prover::prover::prove;
use flock_prover::r1cs_hashes::sha2;

fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() {
    let ncs: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let ncs = if ncs.is_empty() { vec![8usize] } else { ncs };
    for nc in ncs {
        let setup = sha2::Sha256HybridSetup::new(nc);
        let r1cs = &setup.r1cs;
        let nbl = setup.n_blocks_log();
        let mut s: u64 = 0x5A2A_0627u64 ^ (nc as u64);
        let comps: Vec<([u32; 8], [u32; 16])> = (0..nc).map(|_| {
            let mut h = [0u32; 8]; for x in &mut h { *x = splitmix64(&mut s) as u32; }
            let mut m = [0u32; 16]; for x in &mut m { *x = splitmix64(&mut s) as u32; }
            (h, m)
        }).collect();
        let z = sha2::generate_witness(&comps, nbl);
        let z_packed = pack_witness(&z, r1cs.m);

        let run = || {
            let mut ch = FsChallenger::new(b"flock-sha2-v0");
            let _ = prove(r1cs, &z_packed, &setup.pcs_params, &mut ch);
        };
        run(); // warm
        let n = if r1cs.m >= 22 { 3 } else { 5 };
        let mut best = f64::INFINITY;
        for _ in 0..n {
            let t = Instant::now(); run(); best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!("SHA2CPU n_comp={nc} m={} BaseFold prove = {best:.2} ms", r1cs.m);
    }
}
