//! Golden-fixture dumper for the zerocheck round-1 univariate-skip URM
//! (`round1_naive`) and the φ₈ table — port targets for `flock_zorch.gf8`.
//!
//! Dumps flock's `PHI_8_TABLE` (256 F128, for a cross-check of the F2-linear
//! reconstruction) and, for a few (m, k_skip), the reference `round1_naive`
//! output on a deterministic witness. The jax port recomputes both and
//! byte-compares.
//!
//! Usage: `cargo run --release --example dump_gf8_urm -- [out_path]`
//!
//! File layout (LE), F128 as lo||hi (16 B):
//!   magic b"FLKURM01" (8) ++ phi8[256*16]
//!   ++ n_configs: u64
//!   ++ per config: m:u64, k_skip:u64, a[2^m], b[2^m], c[2^m] (one byte/bit),
//!      r[m*16], round1_ab[2^k_skip*16], round1_c[2^k_skip*16]

use std::io::Write;

use flock_core::field::{F128, PHI_8_TABLE};
use flock_core::zerocheck::univariate_skip::round1_naive;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn push_f128(buf: &mut Vec<u8>, v: &F128) {
    buf.extend_from_slice(&v.lo.to_le_bytes());
    buf.extend_from_slice(&v.hi.to_le_bytes());
}

const CONFIGS: [(usize, usize); 2] = [(6, 3), (13, 6)];

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "artifacts/gf8_urm_golden.bin".to_string());

    let mut buf = Vec::new();
    buf.extend_from_slice(b"FLKURM01");
    for v in PHI_8_TABLE.iter() {
        push_f128(&mut buf, v);
    }

    buf.extend_from_slice(&(CONFIGS.len() as u64).to_le_bytes());
    for &(m, k_skip) in &CONFIGS {
        let n = 1usize << m;
        let mut s: u64 = 0xA53F ^ (m as u64).wrapping_mul(0x1000_0001) ^ (k_skip as u64);
        let a: Vec<bool> = (0..n).map(|_| splitmix64(&mut s) & 1 == 1).collect();
        let b: Vec<bool> = (0..n).map(|_| splitmix64(&mut s) & 1 == 1).collect();
        let c: Vec<bool> = (0..n).map(|_| splitmix64(&mut s) & 1 == 1).collect();
        let r: Vec<F128> = (0..m)
            .map(|_| F128::new(splitmix64(&mut s), splitmix64(&mut s)))
            .collect();

        let (p_ab, p_c) = round1_naive(&a, &b, &c, m, k_skip, &r);

        buf.extend_from_slice(&(m as u64).to_le_bytes());
        buf.extend_from_slice(&(k_skip as u64).to_le_bytes());
        for &bit in &a {
            buf.push(bit as u8);
        }
        for &bit in &b {
            buf.push(bit as u8);
        }
        for &bit in &c {
            buf.push(bit as u8);
        }
        for v in &r {
            push_f128(&mut buf, v);
        }
        for v in &p_ab {
            push_f128(&mut buf, v);
        }
        for v in &p_c {
            push_f128(&mut buf, v);
        }
    }

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create artifacts dir");
        }
    }
    std::fs::File::create(&out_path)
        .and_then(|mut f| f.write_all(&buf))
        .expect("write output");
    eprintln!("wrote phi8 table + round1_naive for {CONFIGS:?} to {out_path}");
}
