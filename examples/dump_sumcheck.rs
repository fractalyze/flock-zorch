//! Golden-fixture dumper for the multilinear-sumcheck arithmetic core (iter 10:
//! the zerocheck/lincheck primitives). Dumps reference outputs of flock-core's
//! `build_eq`, `round_pair_naive`, and `fold_in_place_single` on deterministic
//! SplitMix64 inputs; the jax port (`flock_zorch.sumcheck`) reads the inputs,
//! recomputes each, and byte-compares against the reference outputs.
//!
//! Usage: `cargo run --release --example dump_sumcheck -- [out_path]`
//!
//! File layout (little-endian), each F128 serialized as `lo || hi` (16 bytes):
//!   magic b"FLKSUM01" (8)
//!   n_eq: u64;  then n_eq × { n:u64, r[n], eq[2^n] }
//!   n_rp: u64;  then n_rp × { log_n:u64, a[2^log_n], b[2^log_n], r[log_n],
//!                            msg_one[1], msg_inf[1] }
//!   n_fs: u64;  then n_fs × { log_n:u64, a[2^log_n], z[1], folded[2^(log_n-1)] }

use std::io::Write;

use flock_core::field::F128;
use flock_core::zerocheck::multilinear::{fold_in_place_single, round_pair_naive};
use flock_core::zerocheck::univariate_skip::build_eq;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_f128(s: &mut u64) -> F128 {
    F128::new(splitmix64(s), splitmix64(s))
}

fn rand_vec(n: usize, s: &mut u64) -> Vec<F128> {
    (0..n).map(|_| rand_f128(s)).collect()
}

fn push_f128(buf: &mut Vec<u8>, v: &[F128]) {
    for e in v {
        buf.extend_from_slice(&e.lo.to_le_bytes());
        buf.extend_from_slice(&e.hi.to_le_bytes());
    }
}

// Gate sizes: small (correctness corner cases) + medium (exercise the parallel
// doubling/reduction at a non-trivial scale). 2^16 F128 = 1 MiB per buffer.
const EQ_NS: [usize; 3] = [2, 10, 14];
const RP_LOGNS: [usize; 3] = [4, 12, 16];
const FS_LOGNS: [usize; 3] = [4, 12, 16];

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "artifacts/sumcheck_golden.bin".to_string());

    let mut buf = Vec::new();
    buf.extend_from_slice(b"FLKSUM01");

    // ---- build_eq ----
    buf.extend_from_slice(&(EQ_NS.len() as u64).to_le_bytes());
    for &n in &EQ_NS {
        let mut s = 0x1100_u64 ^ (n as u64).wrapping_mul(0x1000_0001);
        let r = rand_vec(n, &mut s);
        let eq = build_eq(&r);
        assert_eq!(eq.len(), 1usize << n);
        buf.extend_from_slice(&(n as u64).to_le_bytes());
        push_f128(&mut buf, &r);
        push_f128(&mut buf, &eq);
    }

    // ---- round_pair_naive ----
    buf.extend_from_slice(&(RP_LOGNS.len() as u64).to_le_bytes());
    for &log_n in &RP_LOGNS {
        let n = 1usize << log_n;
        let mut s = 0x2200_u64 ^ (log_n as u64).wrapping_mul(0x1000_0001);
        let a = rand_vec(n, &mut s);
        let b = rand_vec(n, &mut s);
        let r = rand_vec(log_n, &mut s);
        let (msg_one, msg_inf) = round_pair_naive(&a, &b, &r);
        buf.extend_from_slice(&(log_n as u64).to_le_bytes());
        push_f128(&mut buf, &a);
        push_f128(&mut buf, &b);
        push_f128(&mut buf, &r);
        push_f128(&mut buf, &[msg_one]);
        push_f128(&mut buf, &[msg_inf]);
    }

    // ---- fold_in_place_single ----
    buf.extend_from_slice(&(FS_LOGNS.len() as u64).to_le_bytes());
    for &log_n in &FS_LOGNS {
        let n = 1usize << log_n;
        let mut s = 0x3300_u64 ^ (log_n as u64).wrapping_mul(0x1000_0001);
        let a = rand_vec(n, &mut s);
        let z = rand_f128(&mut s);
        let mut folded = a.clone();
        fold_in_place_single(&mut folded, z);
        assert_eq!(folded.len(), n / 2);
        buf.extend_from_slice(&(log_n as u64).to_le_bytes());
        push_f128(&mut buf, &a);
        push_f128(&mut buf, &[z]);
        push_f128(&mut buf, &folded);
    }

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create artifacts dir");
        }
    }
    std::fs::File::create(&out_path)
        .and_then(|mut f| f.write_all(&buf))
        .expect("write output");
    eprintln!(
        "wrote sumcheck goldens: build_eq{EQ_NS:?} round_pair{RP_LOGNS:?} fold_single{FS_LOGNS:?} -> {out_path}"
    );
}
