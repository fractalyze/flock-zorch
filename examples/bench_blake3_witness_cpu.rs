//! CPU witness-generation bench for the BLAKE3 R1CS — the host-side prep the
//! GPU prove consumes, staged: `generate_witness` (run the compressions and
//! record the trace bits) → `pack_witness` (bits → packed F128 lanes) →
//! `apply_a_packed`/`apply_b_packed` (a = A·z, b = B·z) →
//! `pack_z_lincheck_from_packed`.
//!
//! The 5M-BLAKE3/s milestone's end-criterion 3 asks for witness generation to
//! be measured and a decision recorded on whether it sits inside the target;
//! every GPU number so far ingests a pre-dumped golden, so this is the missing
//! half of that decision. Reports per-stage wall, per-hash cost, and the
//! implied hashes/second of the whole host pipeline.
//!
//! Usage: `cargo run --release --example bench_blake3_witness_cpu -- [n_comp ...]`
//! (defaults: 4096 16384 65536 — m = 26 / 28 / 30).

use std::time::Instant;

use flock_core::lincheck::pack_z_lincheck_from_packed;
use flock_core::pcs::pack::pack_witness;
use flock_prover::r1cs_hashes::blake3;

fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn best_of<T>(n: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::INFINITY;
    let mut out = f();
    for _ in 0..n {
        let t = Instant::now();
        out = f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    (best, out)
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    let sizes = if args.is_empty() { vec![4096, 16384, 65536] } else { args };

    for n_comp in sizes {
        let setup = blake3::Blake3Setup::new(n_comp);
        let r1cs = &setup.r1cs;
        let m = r1cs.m;

        let mut s: u64 = 0xB1A3_0627u64 ^ (n_comp as u64);
        let blocks: Vec<blake3::Compression> = (0..n_comp)
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

        let n = if m >= 28 { 3 } else { 5 };
        let (gen_ms, z) = best_of(n, || blake3::generate_witness(&blocks, setup.n_blocks_log()));
        let (pack_ms, z_packed) = best_of(n, || pack_witness(&z, m));
        let (ab_ms, _ab) = best_of(n, || {
            (r1cs.apply_a_packed(&z_packed), r1cs.apply_b_packed(&z_packed))
        });
        let (lc_ms, _zlc) = best_of(n, || pack_z_lincheck_from_packed(&z_packed, m, r1cs.k_log));

        let total_ms = gen_ms + pack_ms + ab_ms + lc_ms;
        let per_hash_us = total_ms * 1e3 / n_comp as f64;
        println!(
            "BLAKE3WITNESS n_comp={n_comp} m={m} gen={gen_ms:.2}ms pack={pack_ms:.2}ms \
             ab={ab_ms:.2}ms lincheck_pack={lc_ms:.2}ms total={total_ms:.2}ms \
             ({per_hash_us:.3} us/hash, {:.0} hash/s host pipeline)",
            n_comp as f64 / (total_ms / 1e3)
        );
    }
}
