//! CPU SHA-256 throughput anchor for the GPU-vs-CPU gate.
//!
//! Hashes `N` messages of `L` bytes with flock's `merkle::hash_leaf`, parallelized
//! across cores with rayon exactly like `merkle_tree`'s leaf level — flock's
//! honest best on x86 (SHA-NI via `target-cpu=native`, all cores). Prints
//! `SHACPU <N> <L> <best_ms> <mean_ms>`.
//!
//! Usage: `cargo run --release --example bench_sha256_cpu -- <N> <L> [iters]`

use std::time::Instant;

use flock_core::merkle::hash_leaf;
use rayon::prelude::*;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(65536);
    let l: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);

    let mut s: u64 = 0x5EED_05A2_1CE4_E5B9u64 ^ (l as u64).wrapping_mul(0x1000_0001);
    let mut data = vec![0u8; n * l];
    for byte in data.iter_mut() {
        *byte = (splitmix64(&mut s) & 0xFF) as u8;
    }

    let hash_all = || -> u64 {
        let digests: Vec<[u8; 32]> = data.par_chunks(l).map(hash_leaf).collect();
        // Sparse checksum so the work can't be elided.
        digests.iter().step_by((n / 64).max(1)).take(64).fold(0u64, |cs, d| {
            cs ^ u64::from_le_bytes(d[..8].try_into().unwrap())
        })
    };

    let mut sink = hash_all(); // warm-up (spin up rayon pool, page in)
    let mut best = f64::INFINITY;
    let mut total = 0.0f64;
    for _ in 0..iters {
        let t0 = Instant::now();
        sink ^= hash_all();
        let dt = t0.elapsed().as_secs_f64();
        best = best.min(dt);
        total += dt;
    }
    println!("SHACPU {} {} {:.4} {:.4}  [cs={:016x}]", n, l, best * 1e3, total / iters as f64 * 1e3, sink);
}
