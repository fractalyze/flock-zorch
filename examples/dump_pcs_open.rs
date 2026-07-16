//! Golden-fixture dumper for flock's full single-claim PCS open (`pcs::open`).
//!
//! Commits a deterministic witness, runs unmodified flock `pcs::open` at a random
//! x_outer with the real FsChallenger, and dumps z_packed + x_outer + codeword +
//! the OpeningProof (ring_switch.s_hat_v + the full BaseFoldProof). The frx port
//! (`flock_zorch.pcs_open.open`) rebuilds the tree, reruns, and byte-compares.
//!
//! Usage: `cargo run --release --example dump_pcs_open -- [m] [log_inv_rate] [log_batch_size] [out]`

use std::io::Write;

use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::pcs::commit::{commit, PcsParams};
use flock_core::pcs::pack::pack_witness;
use flock_core::pcs::open;

fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn rf(s: &mut u64) -> F128 { F128::new(splitmix64(s), splitmix64(s)) }
fn pf(b: &mut Vec<u8>, v: &F128) { b.extend_from_slice(&v.lo.to_le_bytes()); b.extend_from_slice(&v.hi.to_le_bytes()); }
fn pu(b: &mut Vec<u8>, x: usize) { b.extend_from_slice(&(x as u64).to_le_bytes()); }
fn pfv(b: &mut Vec<u8>, v: &[F128]) { pu(b, v.len()); for e in v { pf(b, e); } }
fn phv(b: &mut Vec<u8>, v: &[[u8; 32]]) { pu(b, v.len()); for h in v { b.extend_from_slice(h); } }

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let lir: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let lbs: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(2);
    let out = a.next().unwrap_or_else(|| "artifacts/pcs_open_golden.bin".to_string());

    let mut s: u64 = 0x09E1_F01D ^ ((m * 100 + lir * 10 + lbs) as u64).wrapping_mul(0x1000_0001);
    let z: Vec<bool> = (0..(1usize << m)).map(|_| splitmix64(&mut s) & 1 == 1).collect();
    let z_packed = pack_witness(&z, m);
    let params = PcsParams { m, log_inv_rate: lir, log_batch_size: lbs, profile: Default::default() };
    let (commitment, pd) = commit(&z_packed, &params);

    let x_outer: Vec<F128> = (0..(m - 6)).map(|_| rf(&mut s)).collect();
    let mut ch = FsChallenger::new(b"flock-pcs-open-test");
    let proof = open(&z_packed, &pd, &commitment, &x_outer, &mut ch);
    let bf = &proof.basefold;

    let mut b = Vec::new();
    b.extend_from_slice(b"FLKOPN01");
    for v in [m, lir, lbs] { pu(&mut b, v); }
    pfv(&mut b, &z_packed);
    pfv(&mut b, &x_outer);
    pfv(&mut b, &pd.codeword);
    pfv(&mut b, &proof.ring_switch.s_hat_v);
    // basefold fields (same layout as dump_basefold):
    pu(&mut b, bf.round_messages.len());
    for rm in &bf.round_messages { pf(&mut b, &rm.u_0); pf(&mut b, &rm.u_2); }
    b.extend_from_slice(&bf.post_row_batch_commit.root);
    phv(&mut b, &bf.round_commitments.iter().map(|c| c.root).collect::<Vec<_>>());
    pf(&mut b, &bf.final_a);
    pf(&mut b, &bf.final_b);
    pfv(&mut b, &bf.final_codeword);
    pu(&mut b, bf.queries.len());
    for q in &bf.queries {
        pu(&mut b, q.position);
        pfv(&mut b, &q.initial_leaf);
        pfv(&mut b, &q.post_row_batch_leaf);
        pu(&mut b, q.epoch_leaves.len());
        for el in &q.epoch_leaves { pfv(&mut b, el); }
    }
    phv(&mut b, &bf.initial_multi_proof);
    phv(&mut b, &bf.post_row_batch_multi_proof);
    pu(&mut b, bf.epoch_multi_proofs.len());
    for mp in &bf.epoch_multi_proofs { phv(&mut b, mp); }

    std::fs::File::create(&out).unwrap().write_all(&b).unwrap();
    println!("dumped pcs::open m={m} lir={lir} lbs={lbs} -> {out}");
}
