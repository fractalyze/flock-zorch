//! Focused single-size CPU benchmark for flock's additive NTT — the CPU anchor
//! for the GPU-vs-CPU 10x gate (`python/flock_zorch/testing/cpu_vs_gpu.py`).
//!
//! Runs `AdditiveNttF128::forward_transform_scalar` (the only path that compiles
//! on x86 — flock's NEON/parallel paths are `aarch64+aes`-gated) on a random
//! buffer of size 2^log_d, repeated `iters` times after one warm-up, and prints
//! one machine-readable line: `NTTCPU <log_d> <best_ms> <mean_ms>`.
//!
//! Usage: `cargo run --release --example bench_ntt_cpu -- <log_d> [iters]`

use std::time::Instant;

use flock_core::field::F128;
use flock_core::ntt::AdditiveNttF128;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Sparse checksum so the optimizer can't elide the transform.
fn checksum(data: &[F128]) -> u64 {
    let stride = (data.len() / 64).max(1);
    data.iter().step_by(stride).take(64).fold(0u64, |cs, x| cs ^ x.lo ^ x.hi)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let log_d: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);
    let n = 1usize << log_d;

    let ntt = AdditiveNttF128::standard(log_d);
    let mut s: u64 = 0x00C0_FFEE ^ (log_d as u64).wrapping_mul(0x1000_0001);
    let original: Vec<F128> =
        (0..n).map(|_| F128::new(splitmix64(&mut s), splitmix64(&mut s))).collect();

    // Warm-up (page in, prime caches/branch predictor).
    let mut warm = original.clone();
    ntt.forward_transform_scalar(&mut warm);
    let mut sink = checksum(&warm);

    let mut best = f64::INFINITY;
    let mut total = 0.0f64;
    for _ in 0..iters {
        let mut data = original.clone();
        let t0 = Instant::now();
        ntt.forward_transform_scalar(&mut data);
        let dt = t0.elapsed().as_secs_f64();
        sink ^= checksum(&data);
        best = best.min(dt);
        total += dt;
    }
    // `NTTCPU <log_d> <best_ms> <mean_ms>  [cs=...]`
    println!(
        "NTTCPU {} {:.4} {:.4}  [cs={:016x}]",
        log_d,
        best * 1e3,
        total / iters as f64 * 1e3,
        sink
    );
}
