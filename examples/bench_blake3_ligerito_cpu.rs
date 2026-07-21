//! CPU baseline for the GPU BLAKE3 Ligerito prover on the same real R1CS.
//! Witness generation stays outside the timed region, matching the GPU bench.
//! Usage: `cargo run --release --example bench_blake3_ligerito_cpu -- [n_comp ...]`

use std::time::Instant;

use flock_core::challenger::FsChallenger;
use flock_core::pcs::pack::pack_witness;
use flock_prover::prover::prove_ligerito;
use flock_prover::r1cs_hashes::blake3;

fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() {
    let ncs: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    let ncs = if ncs.is_empty() { vec![256] } else { ncs };
    for nc in ncs {
        let setup = blake3::Blake3Setup::new(nc);
        let r1cs = &setup.r1cs;
        let mut s = 0xB1A3_0627u64 ^ (nc as u64);
        let blocks: Vec<blake3::Compression> = (0..nc)
            .map(|_| {
                let mut cv = [0u32; 8];
                for x in &mut cv {
                    *x = splitmix64(&mut s) as u32;
                }
                let mut msg = [0u32; 16];
                for x in &mut msg {
                    *x = splitmix64(&mut s) as u32;
                }
                (
                    cv,
                    msg,
                    splitmix64(&mut s),
                    splitmix64(&mut s) as u32,
                    splitmix64(&mut s) as u32,
                )
            })
            .collect();
        let z = blake3::generate_witness(&blocks, setup.n_blocks_log());
        let z_packed = pack_witness(&z, r1cs.m);
        let run = || {
            let mut ch = FsChallenger::new(b"flock-blake3-lig-v0");
            let _ = prove_ligerito(r1cs, z_packed.clone(), &setup.pcs_params, &mut ch);
        };
        run();
        let n = if r1cs.m >= 26 { 3 } else { 5 };
        let mut best = f64::INFINITY;
        for _ in 0..n {
            let t = Instant::now();
            run();
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "BLAKE3LIGCPU n_comp={nc} m={} Ligerito prove = {best:.2} ms",
            r1cs.m
        );
    }
}
