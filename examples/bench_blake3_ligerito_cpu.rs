//! CPU baseline for the GPU BLAKE3 Ligerito prover on the same real R1CS.
//! Witness generation stays outside the timed region, matching the GPU bench.
//!
//! Uses `prove_fast_ligerito_from_witness`, flock's fused per-hash prover and the
//! one its own throughput table is measured with. The generic matrix-driven
//! `prove_ligerito` yields a byte-identical proof (flock's
//! `prove_ligerito_generic_matches_prove_fast` pins that) but is several times
//! slower, so timing it would understate flock.
//!
//! The prover consumes the four witness buffers by value, so each iteration has
//! to re-materialize them; the clone is hoisted above the clock because at these
//! sizes it is 4*2^(m-3) bytes and would otherwise bill 20-35% of the reported
//! prove to memcpy.
//!
//! Usage: `cargo run --release --example bench_blake3_ligerito_cpu -- [n_comp ...]`

use std::time::Instant;

use flock_core::challenger::FsChallenger;
use flock_prover::prover::prove_fast_ligerito_from_witness;
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
        let (z, a, b, zlc) =
            blake3::generate_witness_with_ab_packed_and_lincheck(&blocks, setup.n_blocks_log());
        let lc_circuit = r1cs.csc_lincheck_circuit();
        let run = |z, a, b, zlc| {
            let mut ch = FsChallenger::new(b"flock-blake3-lig-v0");
            let _ = prove_fast_ligerito_from_witness(
                r1cs,
                &setup.pcs_params,
                z,
                a,
                b,
                zlc,
                lc_circuit,
                None,
                &mut ch,
            );
        };
        run(z.clone(), a.clone(), b.clone(), zlc.clone());
        let n = if r1cs.m >= 26 { 3 } else { 5 };
        let mut best = f64::INFINITY;
        for _ in 0..n {
            let (zc, ac, bc, lcc) = (z.clone(), a.clone(), b.clone(), zlc.clone());
            let t = Instant::now();
            run(zc, ac, bc, lcc);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "BLAKE3LIGCPU n_comp={nc} m={} Ligerito prove = {best:.2} ms",
            r1cs.m
        );
    }
}
