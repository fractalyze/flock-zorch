//! CPU baseline for the GPU BLAKE3 prover: times flock's BaseFold `prove` on the
//! SAME real blake3 R1CS (Blake3Setup) the GPU prover gates against, so
//! e2e_blake3_bench's speedup is apples-to-apples on the same instance + same
//! (BaseFold) backend. x86 scalar, matched release profile. Mirrors
//! bench_sha2_cpu.rs.
//!
//! Usage: `cargo run --release --example bench_blake3_cpu -- [n_compressions ...]`

use std::time::Instant;

use flock_core::challenger::FsChallenger;
use flock_core::pcs::pack::pack_witness;
use flock_prover::prover::prove;
use flock_prover::r1cs_hashes::blake3;

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
        let setup = blake3::Blake3Setup::new(nc);
        let r1cs = &setup.r1cs;
        let nbl = setup.n_blocks_log();
        let mut s: u64 = 0xB1A3_0627u64 ^ (nc as u64);
        let blocks: Vec<blake3::Compression> = (0..nc).map(|_| {
            let mut cv = [0u32; 8]; for x in &mut cv { *x = splitmix64(&mut s) as u32; }
            let mut msg = [0u32; 16]; for x in &mut msg { *x = splitmix64(&mut s) as u32; }
            (cv, msg, splitmix64(&mut s), splitmix64(&mut s) as u32, splitmix64(&mut s) as u32)
        }).collect();
        let z = blake3::generate_witness(&blocks, nbl);
        let z_packed = pack_witness(&z, r1cs.m);

        let run = || {
            let mut ch = FsChallenger::new(b"flock-blake3-v0");
            let _ = prove(r1cs, &z_packed, &setup.pcs_params, &mut ch);
        };
        run(); // warm
        let n = if r1cs.m >= 22 { 3 } else { 5 };
        let mut best = f64::INFINITY;
        for _ in 0..n {
            let t = Instant::now(); run(); best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!("BLAKE3CPU n_comp={nc} m={} BaseFold prove = {best:.2} ms", r1cs.m);
    }
}
