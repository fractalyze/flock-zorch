//! Focused CPU benchmark for flock's eq-table build — the dominant data-parallel
//! primitive of the multilinear sumcheck, and the CPU anchor for the sumcheck
//! GPU-vs-CPU 10x gate (`python/flock_zorch/testing/sumcheck_gpu_vs_cpu.py`).
//!
//! Runs `zerocheck::univariate_skip::build_eq` (the scalar reference — the only
//! path that builds on x86; flock's NEON optimizations are aarch64-gated) on a
//! random length-`n` challenge vector, repeated `iters` times after one warm-up,
//! and prints one machine-readable line: `EQCPU <n> <best_ms> <mean_ms>`.
//!
//! Usage: `cargo run --release --example bench_sumcheck_cpu -- <n> [iters]`

use std::hint::black_box;
use std::time::Instant;

use flock_core::field::F128;
use flock_core::zerocheck::univariate_skip::build_eq;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Sparse checksum so the optimizer can't elide the build.
fn checksum(data: &[F128]) -> u64 {
    let stride = (data.len() / 64).max(1);
    data.iter().step_by(stride).take(64).fold(0u64, |cs, x| cs ^ x.lo ^ x.hi)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);

    let mut s: u64 = 0x00C0_FFEE ^ (n as u64).wrapping_mul(0x1000_0001);
    let r: Vec<F128> = (0..n).map(|_| F128::new(splitmix64(&mut s), splitmix64(&mut s))).collect();

    // Warm-up (page in, prime caches/branch predictor).
    let warm = build_eq(&r);
    let mut sink = checksum(&warm);

    let mut best = f64::INFINITY;
    let mut total = 0.0f64;
    for _ in 0..iters {
        let t0 = Instant::now();
        let t = build_eq(black_box(&r));
        let dt = t0.elapsed().as_secs_f64();
        sink ^= checksum(&t);
        best = best.min(dt);
        total += dt;
    }
    // `EQCPU <n> <best_ms> <mean_ms>  [cs=...]`
    println!(
        "EQCPU {} {:.4} {:.4}  [cs={:016x}]",
        n,
        best * 1e3,
        total / iters as f64 * 1e3,
        sink
    );
}
