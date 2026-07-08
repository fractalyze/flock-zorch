//! Golden-fixture dumper for flock's BaseFold PCS open (`pcs::basefold::prove`) —
//! the keystone gate tying together fri_fold, row_batch, merkle tree+multi_proof,
//! and the challenger.
//!
//! Commits a deterministic witness, then runs unmodified flock `basefold::prove`
//! with a random `b` (target=0; running_target doesn't affect proof bytes) and the
//! real `FsChallenger`. Dumps a_init(z_packed) + b + codeword + ALL BaseFoldProof
//! fields. The jax port (`flock_zorch.zorch_basefold.prove_flock_basefold`)
//! rebuilds the tree, reruns, and byte-compares every field.
//!
//! Usage: `cargo run --release --example dump_basefold -- [m] [log_inv_rate] [log_batch_size] [out]`

use std::io::Write;

use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::ntt::AdditiveNttF128;
use flock_core::pcs::basefold::{default_fri_queries, prove};
use flock_core::pcs::commit::{commit, PcsParams};
use flock_core::pcs::pack::pack_witness;

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
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(14);
    let lir: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let lbs: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(2);
    let out = a.next().unwrap_or_else(|| "artifacts/basefold_golden.bin".to_string());

    let mut s: u64 = 0xBA5E_F01D ^ ((m * 100 + lir * 10 + lbs) as u64).wrapping_mul(0x1000_0001);
    let z: Vec<bool> = (0..(1usize << m)).map(|_| splitmix64(&mut s) & 1 == 1).collect();
    let z_packed = pack_witness(&z, m);
    let params = PcsParams { m, log_inv_rate: lir, log_batch_size: lbs, profile: Default::default() };
    let (_commitment, pd) = commit(&z_packed, &params);

    let log_msg = m - 7;
    let k_code = (log_msg - lbs) + lir;
    let bvec: Vec<F128> = (0..(1usize << log_msg)).map(|_| rf(&mut s)).collect();
    let ntt = AdditiveNttF128::standard(k_code);
    let n_queries = default_fri_queries(lir);
    let mut ch = FsChallenger::new(b"flock-basefold-test");
    let proof = prove(&z_packed, bvec.clone(), F128::ZERO, &pd.codeword, &pd.merkle_tree,
                      &ntt, lir, lbs, n_queries, &mut ch);

    let mut b = Vec::new();
    b.extend_from_slice(b"FLKBSF01");
    for v in [m, lir, lbs, n_queries] { pu(&mut b, v); }
    pfv(&mut b, &z_packed);
    pfv(&mut b, &bvec);
    pfv(&mut b, &pd.codeword);
    // proof:
    pu(&mut b, proof.round_messages.len());
    for rm in &proof.round_messages { pf(&mut b, &rm.u_0); pf(&mut b, &rm.u_2); }
    b.extend_from_slice(&proof.post_row_batch_commit.root);
    phv(&mut b, &proof.round_commitments.iter().map(|c| c.root).collect::<Vec<_>>());
    pf(&mut b, &proof.final_a);
    pf(&mut b, &proof.final_b);
    pfv(&mut b, &proof.final_codeword);
    pu(&mut b, proof.queries.len());
    for q in &proof.queries {
        pu(&mut b, q.position);
        pfv(&mut b, &q.initial_leaf);
        pfv(&mut b, &q.post_row_batch_leaf);
        pu(&mut b, q.epoch_leaves.len());
        for el in &q.epoch_leaves { pfv(&mut b, el); }
    }
    phv(&mut b, &proof.initial_multi_proof);
    phv(&mut b, &proof.post_row_batch_multi_proof);
    pu(&mut b, proof.epoch_multi_proofs.len());
    for mp in &proof.epoch_multi_proofs { phv(&mut b, mp); }

    std::fs::File::create(&out).unwrap().write_all(&b).unwrap();
    println!("dumped basefold m={m} lir={lir} lbs={lbs} k_code={k_code} n_queries={n_queries} \
              rounds={} epochs={} -> {out}", proof.round_messages.len(), proof.round_commitments.len());
}
