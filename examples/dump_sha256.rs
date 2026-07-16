//! Golden-fixture dumper for SHA-256 (port target #3 — the Merkle/challenger hash).
//!
//! Hashes `N` deterministic SplitMix64 messages of `L` bytes each with flock's
//! own `merkle::hash_leaf` (= `Sha256::digest`), so the fixture is byte-anchored
//! to unmodified flock. The frx port (`flock_zorch.sha256.digest`) reads the
//! inputs, recomputes the digests, and byte-compares.
//!
//! Usage: `cargo run --release --example dump_sha256 -- [N] [L] [out_path]`
//!
//! File layout (LE): magic `b"FLKSHA01"` (8) ++ `N: u64` ++ `L: u64` ++
//!   inputs[N*L bytes] ++ digests[N*32 bytes].

use std::io::Write;

use flock_core::merkle::hash_leaf;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4096);
    let l: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let out_path = args.next().unwrap_or_else(|| "artifacts/sha256_golden.bin".to_string());

    let mut s: u64 = 0x5EED_05A2_1CE4_E5B9u64 ^ (l as u64).wrapping_mul(0x1000_0001);
    let mut inputs = vec![0u8; n * l];
    for byte in inputs.iter_mut() {
        *byte = (splitmix64(&mut s) & 0xFF) as u8;
    }

    let mut digests = vec![0u8; n * 32];
    for i in 0..n {
        let d = hash_leaf(&inputs[i * l..(i + 1) * l]);
        digests[i * 32..(i + 1) * 32].copy_from_slice(&d);
    }

    let mut buf = Vec::with_capacity(16 + inputs.len() + digests.len());
    buf.extend_from_slice(b"FLKSHA01");
    buf.extend_from_slice(&(n as u64).to_le_bytes());
    buf.extend_from_slice(&(l as u64).to_le_bytes());
    buf.extend_from_slice(&inputs);
    buf.extend_from_slice(&digests);

    let mut f = std::fs::File::create(&out_path).expect("create out file");
    f.write_all(&buf).expect("write");
    println!("dumped {n} SHA-256 digests of {l}-byte messages -> {out_path}");
}
