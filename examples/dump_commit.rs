//! Golden-fixture dumper for the PCS commit (first full sub-protocol).
//!
//! Builds a deterministic Boolean witness, packs it, runs flock's `pcs::commit`,
//! and dumps the packed witness + the 32-byte Merkle root. The jax port
//! (`flock_zorch.pcs_commit.commit_root`) reads `z_packed`, recomputes the
//! interleaved-NTT + Merkle commit, and byte-compares the root.
//!
//! Usage: `cargo run --release --example dump_commit -- [m] [log_inv_rate] [log_batch_size] [out]`
//!
//! File layout (LE): magic `b"FLKCMT01"` (8) ++ `m: u64` ++ `log_inv_rate: u64` ++
//!   `log_batch_size: u64` ++ z_packed[2^(m-7) * 16 bytes] ++ root[32 bytes].

use std::io::Write;

use flock_core::field::F128;
use flock_core::pcs::commit::{PcsParams, commit};
use flock_core::pcs::pack::pack_witness;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let m: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let log_inv_rate: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let log_batch_size: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    let out_path = args.next().unwrap_or_else(|| "artifacts/commit_golden.bin".to_string());

    // Deterministic Boolean witness of length 2^m.
    let mut s: u64 = 0xC0117_1AB1E ^ (m as u64).wrapping_mul(0x1000_0001);
    let z: Vec<bool> = (0..(1usize << m)).map(|_| splitmix64(&mut s) & 1 == 1).collect();
    let z_packed: Vec<F128> = pack_witness(&z, m);

    let params = PcsParams { m, log_inv_rate, log_batch_size, profile: Default::default() };
    let (commitment, _pd) = commit(&z_packed, &params);

    let mut buf = Vec::with_capacity(32 + z_packed.len() * 16 + 32);
    buf.extend_from_slice(b"FLKCMT01");
    buf.extend_from_slice(&(m as u64).to_le_bytes());
    buf.extend_from_slice(&(log_inv_rate as u64).to_le_bytes());
    buf.extend_from_slice(&(log_batch_size as u64).to_le_bytes());
    for e in &z_packed {
        buf.extend_from_slice(&e.lo.to_le_bytes());
        buf.extend_from_slice(&e.hi.to_le_bytes());
    }
    buf.extend_from_slice(&commitment.root);

    let mut f = std::fs::File::create(&out_path).expect("create out file");
    f.write_all(&buf).expect("write");
    println!(
        "dumped pcs::commit root for m={m} rate=1/2^{log_inv_rate} batch=2^{log_batch_size} -> {out_path}  root={}",
        hex(&commitment.root)
    );
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
