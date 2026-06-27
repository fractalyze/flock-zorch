//! Golden dumper for flock's hash-chain prover-side core (task #14, M4a): the
//! shift sumcheck (`chain::prove_chain_shift`) and the region fold
//! (`chain_common::fold_in_out`). These are the two NEW prover pieces of the
//! keccak hash-CHAIN protocol; the mixed packed-direct PCS open that consumes the
//! returned claim is the remaining chain milestone.
//!
//! Gate A (shift): random In/Out vectors (2^n) → ChainShiftProof + ChainClaims.
//! Gate B (fold): synthetic packed witness + τ_pos over the keccak CHAIN_LAYOUT
//!                → (in_vals, out_vals).
//!
//! Usage: `cargo run --release --example dump_chain_shift -- [out]`

use std::io::Write;

use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_prover::chain::prove_chain_shift;
use flock_prover::r1cs_hashes::chain_common::{fold_in_out, ChainFold};
use flock_prover::r1cs_hashes::keccak::CHAIN_LAYOUT;

fn sm(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn rf(s: &mut u64) -> F128 { F128::new(sm(s), sm(s)) }
fn pf(b: &mut Vec<u8>, v: &F128) { b.extend_from_slice(&v.lo.to_le_bytes()); b.extend_from_slice(&v.hi.to_le_bytes()); }
fn pu(b: &mut Vec<u8>, x: usize) { b.extend_from_slice(&(x as u64).to_le_bytes()); }
fn pfv(b: &mut Vec<u8>, v: &[F128]) { pu(b, v.len()); for e in v { pf(b, e); } }
fn ppair(b: &mut Vec<u8>, v: &[(F128, F128)]) { pu(b, v.len()); for (x, y) in v { pf(b, x); pf(b, y); } }

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "artifacts/chain_shift_golden.bin".to_string());
    let mut s: u64 = 0xC4A1_0627;

    // ---- Gate A: shift sumcheck on random In/Out (2^n instances).
    let n = 6usize;
    let n_total = 1usize << n;
    let in_vals: Vec<F128> = (0..n_total).map(|_| rf(&mut s)).collect();
    let out_vals: Vec<F128> = (0..n_total).map(|_| rf(&mut s)).collect();
    let mut ch = FsChallenger::new(b"flock-chain-shift-v0");
    let (proof, claims) = prove_chain_shift(&in_vals, &out_vals, &mut ch);

    // ---- Gate B: region fold over the keccak CHAIN_LAYOUT, synthetic witness.
    let layout = CHAIN_LAYOUT;
    let n_inst_log = 3usize;
    let bits_per_packed = 1usize << 7;
    let block_packed = (1usize << layout.k_log) / bits_per_packed; // 512
    let n_inst = 1usize << n_inst_log;
    let packed: Vec<F128> = (0..n_inst * block_packed).map(|_| rf(&mut s)).collect();
    let tau_pos: Vec<F128> = (0..layout.tau_pos_len()).map(|_| rf(&mut s)).collect();
    let fold = ChainFold::new(&layout, tau_pos.clone());
    let (fold_in, fold_out) = fold_in_out(&layout, &packed, &fold);

    let mut b = Vec::new();
    b.extend_from_slice(b"FLKCHN01");
    // Gate A
    pu(&mut b, n);
    pfv(&mut b, &in_vals); pfv(&mut b, &out_vals);
    ppair(&mut b, &proof.rounds); pf(&mut b, &proof.g_at_point);
    pfv(&mut b, &claims.instance_point); pf(&mut b, &claims.sel0); pf(&mut b, &claims.value);
    // Gate B
    for v in [layout.k_log, layout.region_log, layout.input_byte_off, layout.output_byte_off] { pu(&mut b, v); }
    pfv(&mut b, &tau_pos); pfv(&mut b, &packed);
    pfv(&mut b, &fold_in); pfv(&mut b, &fold_out);

    std::fs::File::create(&out).unwrap().write_all(&b).unwrap();
    println!("dumped chain-shift: gate A n={n} ({} rounds) | gate B k_log={} τ_pos={} n_inst={n_inst} -> {out}",
             proof.rounds.len(), layout.k_log, layout.tau_pos_len());
}
