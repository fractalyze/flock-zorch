//! Golden-fixture dumper for the SHA-256 Merkle tree (PCS-commit component).
//!
//! Builds flock's `merkle::merkle_root` over `n_leaves` leaves of `leaf_size`
//! bytes each (deterministic SplitMix64 data), so the fixture is byte-anchored to
//! unmodified flock. The jax port (`flock_zorch.merkle.merkle_root`) reads the
//! data, recomputes the tree, and byte-compares the 32-byte root.
//!
//! Usage: `cargo run --release --example dump_merkle -- [n_leaves] [leaf_size] [out]`
//!
//! File layout (LE): magic `b"FLKMRK01"` (8) ++ `n_leaves: u64` ++
//!   `leaf_size: u64` ++ data[n_leaves*leaf_size bytes] ++ root[32 bytes].

use std::io::Write;

use flock_core::merkle::merkle_root;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n_leaves: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4096);
    let leaf_size: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let out_path = args.next().unwrap_or_else(|| "artifacts/merkle_golden.bin".to_string());
    assert!(n_leaves.is_power_of_two(), "n_leaves must be a power of two");

    let mut s: u64 = 0x4E11_5EED_1CE4_E5B9u64 ^ (leaf_size as u64).wrapping_mul(0x1000_0001);
    let mut data = vec![0u8; n_leaves * leaf_size];
    for byte in data.iter_mut() {
        *byte = (splitmix64(&mut s) & 0xFF) as u8;
    }

    let root = merkle_root(&data, n_leaves);

    let mut buf = Vec::with_capacity(24 + data.len() + 32);
    buf.extend_from_slice(b"FLKMRK01");
    buf.extend_from_slice(&(n_leaves as u64).to_le_bytes());
    buf.extend_from_slice(&(leaf_size as u64).to_le_bytes());
    buf.extend_from_slice(&data);
    buf.extend_from_slice(&root);

    let mut f = std::fs::File::create(&out_path).expect("create out file");
    f.write_all(&buf).expect("write");
    println!("dumped merkle root over {n_leaves} x {leaf_size}B leaves -> {out_path}");
}
