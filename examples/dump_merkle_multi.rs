//! Golden-fixture dumper for the SHA-256 Merkle tree + octopus multi-proof
//! (the query-opening primitive of the PCS open).
//!
//! Builds flock's `merkle::merkle_tree` over `n_leaves` x `leaf_size`-byte leaves,
//! picks `n_pos` deterministic query positions, and dumps the data + positions +
//! flock's `merkle::merkle_multi_proof`. The frx port (`flock_zorch.merkle`)
//! recomputes the tree + multi-proof and byte-compares.
//!
//! Usage: `cargo run --release --example dump_merkle_multi -- [n_leaves] [leaf_size] [n_pos] [out]`
//! Layout (LE): magic b"FLKMMP01" (8) ++ n_leaves,leaf_size,n_pos (3×u64) ++
//!   data[n_leaves*leaf_size] ++ positions[n_pos×u64] ++
//!   proof_len:u64 ++ proof[proof_len×32].

use std::io::Write;

use flock_core::merkle::{merkle_multi_proof, merkle_tree};

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
    let n_pos: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(30);
    let out_path = args.next().unwrap_or_else(|| "artifacts/merkle_multi_golden.bin".to_string());
    assert!(n_leaves.is_power_of_two());

    let mut s: u64 = 0x44_3110_C705 ^ (leaf_size as u64).wrapping_mul(0x1000_0001);
    let mut data = vec![0u8; n_leaves * leaf_size];
    for b in data.iter_mut() {
        *b = (splitmix64(&mut s) & 0xFF) as u8;
    }
    let positions: Vec<usize> = (0..n_pos).map(|_| (splitmix64(&mut s) as usize) % n_leaves).collect();

    let tree = merkle_tree(&data, n_leaves);
    let proof = merkle_multi_proof(&tree, n_leaves, &positions);

    let mut buf = Vec::new();
    buf.extend_from_slice(b"FLKMMP01");
    for v in [n_leaves, leaf_size, n_pos] {
        buf.extend_from_slice(&(v as u64).to_le_bytes());
    }
    buf.extend_from_slice(&data);
    for &p in &positions {
        buf.extend_from_slice(&(p as u64).to_le_bytes());
    }
    buf.extend_from_slice(&(proof.len() as u64).to_le_bytes());
    for h in &proof {
        buf.extend_from_slice(h);
    }

    let mut f = std::fs::File::create(&out_path).expect("create");
    f.write_all(&buf).expect("write");
    println!("dumped merkle_multi n_leaves={n_leaves} leaf_size={leaf_size} n_pos={n_pos} proof_len={} -> {out_path}", proof.len());
}
