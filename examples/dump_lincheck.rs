//! Golden-fixture dumper for flock's lincheck `prove` (port target: the second
//! PIOP sub-protocol). Builds deterministic sparse base matrices A_0,B_0, a
//! witness z, and a QuirkyPoint x_ab, runs unmodified flock-core's
//! `lincheck::prove`, and dumps ALL inputs + the resulting `LincheckProof`
//! (rounds + z_partial). The jax port (`flock_zorch.lincheck.prove`) reads the
//! inputs, recomputes, and byte-compares rounds + z_partial.
//!
//! Usage: `cargo run --release --example dump_lincheck -- [m] [k_log] [k_skip] [out]`
//!
//! File layout (LE): magic b"FLKLIN01" (8) ++ m,k_log,k_skip (3×u64) ++
//!   A0 ++ B0 (each: for r in 0..k: nnz:u64 ++ cols[nnz]:u64) ++
//!   z_packed (len:u64 ++ bytes) ++
//!   x_ab.z_skip (F128=lo,hi) ++ x_inner_rest (len:u64 ++ F128s) ++ x_outer (len:u64 ++ F128s) ++
//!   proof.rounds (len:u64 ++ (e1,einf) F128 pairs) ++ proof.z_partial (len:u64 ++ F128s).

use std::io::Write;

use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::lincheck::{prove, pack_z_lincheck, QuirkyPoint, SparseMatrixCircuit};
use flock_core::r1cs::SparseBinaryMatrix;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic k×k binary matrix with ~2k nonzeros (rows[r] = sorted, deduped
/// column indices). Matches the shape of flock's test `random_sparse_matrix`.
fn rand_matrix(k: usize, s: &mut u64) -> SparseBinaryMatrix {
    let mut rows: Vec<Vec<usize>> = vec![Vec::new(); k];
    for _ in 0..(k * 2) {
        let r = (splitmix64(s) as usize) % k;
        let c = (splitmix64(s) as usize) % k;
        if !rows[r].contains(&c) {
            rows[r].push(c);
        }
    }
    for row in &mut rows {
        row.sort_unstable();
    }
    SparseBinaryMatrix { num_rows: k, num_cols: k, rows }
}

fn rand_f128(s: &mut u64) -> F128 {
    F128::new(splitmix64(s), splitmix64(s))
}

fn push_f128(buf: &mut Vec<u8>, v: &F128) {
    buf.extend_from_slice(&v.lo.to_le_bytes());
    buf.extend_from_slice(&v.hi.to_le_bytes());
}
fn push_u64(buf: &mut Vec<u8>, x: usize) {
    buf.extend_from_slice(&(x as u64).to_le_bytes());
}
fn push_matrix(buf: &mut Vec<u8>, m: &SparseBinaryMatrix) {
    for row in &m.rows {
        push_u64(buf, row.len());
        for &c in row {
            push_u64(buf, c);
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let m: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(12);
    let k_log: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    let k_skip: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let out_path = args.next().unwrap_or_else(|| "artifacts/lincheck_golden.bin".to_string());

    let k = 1usize << k_log;
    let n_log = m - k_log;
    let inner_rest = k_log - k_skip;
    let mut s: u64 = 0x11_C5EE_D000u64 ^ ((m * 100 + k_log * 10 + k_skip) as u64).wrapping_mul(0x1000_0001);

    let a_0 = rand_matrix(k, &mut s);
    let b_0 = rand_matrix(k, &mut s);
    let z: Vec<bool> = (0..(1usize << m)).map(|_| splitmix64(&mut s) & 1 == 1).collect();
    let z_packed = pack_z_lincheck(&z, m, k_log);

    let x_ab = QuirkyPoint {
        z_skip: rand_f128(&mut s),
        x_inner_rest: (0..inner_rest).map(|_| rand_f128(&mut s)).collect(),
        x_outer: (0..n_log).map(|_| rand_f128(&mut s)).collect(),
    };

    let circuit = SparseMatrixCircuit::new(&a_0, &b_0);
    let mut ch = FsChallenger::new(b"flock-test-v0");
    let (proof, _claim) = prove(&z_packed, m, k_log, k_skip, &circuit, &x_ab, &mut ch);

    let mut buf = Vec::new();
    buf.extend_from_slice(b"FLKLIN01");
    push_u64(&mut buf, m);
    push_u64(&mut buf, k_log);
    push_u64(&mut buf, k_skip);
    push_matrix(&mut buf, &a_0);
    push_matrix(&mut buf, &b_0);
    push_u64(&mut buf, z_packed.len());
    buf.extend_from_slice(&z_packed);
    push_f128(&mut buf, &x_ab.z_skip);
    push_u64(&mut buf, x_ab.x_inner_rest.len());
    for v in &x_ab.x_inner_rest {
        push_f128(&mut buf, v);
    }
    push_u64(&mut buf, x_ab.x_outer.len());
    for v in &x_ab.x_outer {
        push_f128(&mut buf, v);
    }
    push_u64(&mut buf, proof.rounds.len());
    for (e1, einf) in &proof.rounds {
        push_f128(&mut buf, e1);
        push_f128(&mut buf, einf);
    }
    push_u64(&mut buf, proof.z_partial.len());
    for v in &proof.z_partial {
        push_f128(&mut buf, v);
    }

    let mut f = std::fs::File::create(&out_path).expect("create out file");
    f.write_all(&buf).expect("write");
    println!(
        "dumped lincheck proof m={m} k_log={k_log} k_skip={k_skip} (rounds={}, z_partial={}) -> {out_path}",
        proof.rounds.len(),
        proof.z_partial.len()
    );
}
