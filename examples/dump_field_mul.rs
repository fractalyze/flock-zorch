//! Golden-fixture dumper for the GF(2^128) multiply byte-match gate (port target #1).
//!
//! Streams deterministic SplitMix64 `(a, b)` pairs through flock-core's reference
//! `software::ghash_mul` and writes `(a, b, a*b)` to a binary file. The jax port
//! (`flock_zorch.field.mul`) reads `a, b`, recomputes the product, and byte-compares
//! against `out` — see `python/flock_zorch/testing/field_oracle_test.py`.
//!
//! Usage: `cargo run --release --example dump_field_mul -- [N] [out_path]`
//!   N         number of pairs (default 1<<20)
//!   out_path  output file (default `artifacts/field_mul_golden.bin`)
//!
//! File layout (all little-endian): magic `b"FLKMUL01"` (8) ++ `N: u64` (8) ++
//!   `a_block[N*16]` ++ `b_block[N*16]` ++ `out_block[N*16]`, each element `lo || hi`.

use std::io::Write;

use flock_core::field::gf2_128::{software, F128};

/// flock's test PRNG (gf2_128.rs:553) — reproduced so fixtures are deterministic.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1 << 20);
    let out_path = args
        .next()
        .unwrap_or_else(|| "artifacts/field_mul_golden.bin".to_string());

    // Independent streams for a and b so they are uncorrelated.
    let mut sa: u64 = 0x00AB_CDEF;
    let mut sb: u64 = 0x0012_3456;
    let mut a_block = Vec::with_capacity(n * 16);
    let mut b_block = Vec::with_capacity(n * 16);
    let mut o_block = Vec::with_capacity(n * 16);

    for _ in 0..n {
        let a = F128::new(splitmix64(&mut sa), splitmix64(&mut sa));
        let b = F128::new(splitmix64(&mut sb), splitmix64(&mut sb));
        let o = software::ghash_mul(a, b);
        a_block.extend_from_slice(&a.lo.to_le_bytes());
        a_block.extend_from_slice(&a.hi.to_le_bytes());
        b_block.extend_from_slice(&b.lo.to_le_bytes());
        b_block.extend_from_slice(&b.hi.to_le_bytes());
        o_block.extend_from_slice(&o.lo.to_le_bytes());
        o_block.extend_from_slice(&o.hi.to_le_bytes());
    }

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create artifacts dir");
        }
    }
    let mut f =
        std::io::BufWriter::new(std::fs::File::create(&out_path).expect("create output file"));
    f.write_all(b"FLKMUL01").unwrap();
    f.write_all(&(n as u64).to_le_bytes()).unwrap();
    f.write_all(&a_block).unwrap();
    f.write_all(&b_block).unwrap();
    f.write_all(&o_block).unwrap();
    f.flush().unwrap();
    eprintln!("wrote {n} pairs ({} bytes) to {out_path}", 16 + 3 * n * 16);
}
