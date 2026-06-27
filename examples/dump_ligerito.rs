//! Golden-fixture dumper for flock's Ligerito recursive PCS (driver-isolated) —
//! the gate for flock-zorch's GPU Ligerito port (M6).
//!
//! Calls the pub `recursive_prover_with_basis` directly on a synthetic
//! (packed_witness, b_initial, target) + an externally-built L0 commit
//! (`pcs::commit`, log_batch_size=initial_k, log_inv_rate=log_inv_rates[0]), so
//! the recursion driver is gated WITHOUT zerocheck/lincheck (those are already
//! byte-identical). Config sourced from flock's `prover_config_for` (no
//! transcription). Dumps the config + inputs + the full LigeritoProof.
//!
//! Usage: `cargo run --release --example dump_ligerito -- [log_n] [out]`
//!   (log_n=15 -> m=22 -> m22_fast config; initial_k must be 6)

use std::io::Write;

use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::pcs::commit::{commit, PcsParams};
use flock_core::pcs::ligerito::{prover_config_for, recursive_prover_with_basis, LigeritoProfile};

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
fn puv(b: &mut Vec<u8>, v: &[usize]) { pu(b, v.len()); for &e in v { pu(b, e); } }
fn pu64v(b: &mut Vec<u8>, v: &[u64]) { pu(b, v.len()); for &e in v { b.extend_from_slice(&e.to_le_bytes()); } }
fn phv(b: &mut Vec<u8>, v: &[[u8; 32]]) { pu(b, v.len()); for h in v { b.extend_from_slice(h); } }
fn prows(b: &mut Vec<u8>, rows: &[Vec<F128>]) { pu(b, rows.len()); for r in rows { pfv(b, r); } }

fn main() {
    let mut a = std::env::args().skip(1);
    let log_n: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(15);
    let out = a.next().unwrap_or_else(|| "artifacts/ligerito_golden.bin".to_string());
    let m = log_n + 7;
    let lbs = 6; // = initial_k for the embedded configs
    let cfg = prover_config_for(log_n, lbs, LigeritoProfile::Fast).expect("config");

    let n = 1usize << log_n;
    let mut s: u64 = 0x11_6E_0627u64 ^ (log_n as u64);
    let f: Vec<F128> = (0..n).map(|_| rf(&mut s)).collect();
    let b_initial: Vec<F128> = (0..n).map(|_| rf(&mut s)).collect();
    // target = <f, b_initial> (proof bytes are independent of it, but keep it honest)
    let target = f.iter().zip(&b_initial).fold(F128::ZERO, |acc, (x, y)| acc + (*x * *y));

    let params = PcsParams { m, log_inv_rate: cfg.log_inv_rates[0], log_batch_size: cfg.initial_k,
                             profile: Default::default() };
    let (_commitment, pd) = commit(&f, &params);

    let mut ch = FsChallenger::new(b"flock-ligerito-test");
    let proof = recursive_prover_with_basis(&cfg, f.clone(), b_initial.clone(), target,
                                            &pd.codeword, &pd.merkle_tree, &mut ch);

    let mut out_b = Vec::new();
    out_b.extend_from_slice(b"FLKLIG01");
    pu(&mut out_b, log_n); pu(&mut out_b, m); pu(&mut out_b, lbs);
    // config
    puv(&mut out_b, &cfg.log_inv_rates);
    pu(&mut out_b, cfg.recursive_steps);
    pu(&mut out_b, cfg.initial_log_msg_cols);
    pu(&mut out_b, cfg.initial_log_num_interleaved);
    pu(&mut out_b, cfg.initial_k);
    puv(&mut out_b, &cfg.recursive_log_msg_cols);
    puv(&mut out_b, &cfg.recursive_ks);
    puv(&mut out_b, &cfg.queries);
    puv(&mut out_b, &cfg.grinding_bits);
    puv(&mut out_b, &cfg.fold_grinding_bits);
    puv(&mut out_b, &cfg.ood_samples);
    // inputs
    pfv(&mut out_b, &f);
    pfv(&mut out_b, &b_initial);
    pf(&mut out_b, &target);
    pfv(&mut out_b, &pd.codeword);
    phv(&mut out_b, &pd.merkle_tree);
    // LigeritoProof
    out_b.extend_from_slice(&proof.initial_root);
    prows(&mut out_b, &proof.initial_proof.opened_rows); phv(&mut out_b, &proof.initial_proof.merkle_proof);
    phv(&mut out_b, &proof.recursive_roots);
    pu(&mut out_b, proof.recursive_proofs.len());
    for rp in &proof.recursive_proofs { prows(&mut out_b, &rp.opened_rows); phv(&mut out_b, &rp.merkle_proof); }
    pfv(&mut out_b, &proof.final_proof.yr);
    prows(&mut out_b, &proof.final_proof.opened_rows); phv(&mut out_b, &proof.final_proof.merkle_proof);
    pu(&mut out_b, proof.sumcheck_transcript.len());
    for sm in &proof.sumcheck_transcript { pf(&mut out_b, &sm.u_0); pf(&mut out_b, &sm.u_2); }
    pu64v(&mut out_b, &proof.grinding_nonces);
    pfv(&mut out_b, &proof.ood_values);
    pu64v(&mut out_b, &proof.fold_grinding_nonces);

    std::fs::File::create(&out).unwrap().write_all(&out_b).unwrap();
    println!("dumped ligerito log_n={log_n} m={m} | R={} initial_k={} log_inv_rates={:?} recursive_ks={:?} \
              queries={:?} ood={:?} fold_grind={:?} | sumcheck_msgs={} recursive_roots={} ood_values={} -> {out}",
             cfg.recursive_steps, cfg.initial_k, cfg.log_inv_rates, cfg.recursive_ks, cfg.queries,
             cfg.ood_samples, cfg.fold_grinding_bits, proof.sumcheck_transcript.len(),
             proof.recursive_roots.len(), proof.ood_values.len());
}
