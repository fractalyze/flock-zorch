//! Golden dumper for flock's keccak hash-CHAIN prover with the LIGERITO PCS
//! (task #14, M4b) — proves 2^n keccaks form a sequential chain x_{i+1}=keccak_f(x_i)
//! with public endpoints. n_keccaks=64 → m=22 → m22_fast config; n = m−k_log = 6.
//!
//! Generates a REAL chain (x_{i+1} = keccak_f(x_i)) so it's an honest chain proof,
//! then dumps the witness + config + chain-layout geometry + the full
//! ChainProofLigerito {zerocheck, lincheck, shift, pcs_open(ligerito)}.
//!
//! Usage: `cargo run --release --example dump_keccak_chain -- [out]`

use std::io::Write;

use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::pcs::ligerito::{prover_config_for, LigeritoProfile};
use flock_prover::r1cs_hashes::keccak::{self, keccak_f, KeccakSetup, State, CHAIN_LAYOUT, STATE_BITS};

fn sm(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn pf(b: &mut Vec<u8>, v: &F128) { b.extend_from_slice(&v.lo.to_le_bytes()); b.extend_from_slice(&v.hi.to_le_bytes()); }
fn pu(b: &mut Vec<u8>, x: usize) { b.extend_from_slice(&(x as u64).to_le_bytes()); }
fn pfv(b: &mut Vec<u8>, v: &[F128]) { pu(b, v.len()); for e in v { pf(b, e); } }
fn puv(b: &mut Vec<u8>, v: &[usize]) { pu(b, v.len()); for &e in v { pu(b, e); } }
fn pu64v(b: &mut Vec<u8>, v: &[u64]) { pu(b, v.len()); for &e in v { b.extend_from_slice(&e.to_le_bytes()); } }
fn ppair(b: &mut Vec<u8>, v: &[(F128, F128)]) { pu(b, v.len()); for (x, y) in v { pf(b, x); pf(b, y); } }
fn phv(b: &mut Vec<u8>, v: &[[u8; 32]]) { pu(b, v.len()); for h in v { b.extend_from_slice(h); } }
fn prows_f(b: &mut Vec<u8>, rows: &[Vec<F128>]) { pu(b, rows.len()); for r in rows { pfv(b, r); } }

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "artifacts/keccak_chain_golden.bin".to_string());
    let n_keccaks = 64usize;

    let setup = KeccakSetup::new(n_keccaks);
    let r1cs = &setup.r1cs;
    let m = r1cs.m;
    let n_blocks_log = m - r1cs.k_log;
    let log_n = m - 7;
    let cfg = prover_config_for(log_n, setup.pcs_params.log_batch_size, LigeritoProfile::Fast).expect("cfg");

    // Honest chain: x_0 random, x_{i+1} = keccak_f(x_i).
    let mut s: u64 = 0x6EC_C8A1u64 ^ (n_keccaks as u64);
    let mut x: State = [false; STATE_BITS];
    for bit in x.iter_mut() { *bit = sm(&mut s) & 1 == 1; }
    let mut states: Vec<State> = Vec::with_capacity(n_keccaks);
    states.push(x);
    for _ in 1..n_keccaks {
        keccak_f(&mut x);
        states.push(x);
    }
    let (z_packed, a_packed, b_packed, z_lincheck) =
        keccak::generate_witness_with_ab_packed_and_lincheck(&states, n_blocks_log);

    let mut ch = FsChallenger::new(b"flock-keccak-chain-v0");
    let (proof, commitment) = setup.prove_chain(&states, &mut ch); // Ligerito chain

    let lay = CHAIN_LAYOUT;
    let mut b = Vec::new();
    b.extend_from_slice(b"FLKKC_01");
    for v in [m, r1cs.k_log, r1cs.k_skip, r1cs.useful_bits,
              r1cs.const_pin.unwrap_or(keccak::Z_CONST),
              setup.pcs_params.log_inv_rate, setup.pcs_params.log_batch_size, n_blocks_log, log_n,
              lay.region_log, lay.input_byte_off, lay.output_byte_off] { pu(&mut b, v); }
    // Ligerito config
    puv(&mut b, &cfg.log_inv_rates); pu(&mut b, cfg.recursive_steps);
    pu(&mut b, cfg.initial_log_msg_cols); pu(&mut b, cfg.initial_log_num_interleaved); pu(&mut b, cfg.initial_k);
    puv(&mut b, &cfg.recursive_log_msg_cols); puv(&mut b, &cfg.recursive_ks); puv(&mut b, &cfg.queries);
    puv(&mut b, &cfg.grinding_bits); puv(&mut b, &cfg.fold_grinding_bits); puv(&mut b, &cfg.ood_samples);
    // witness ingestion
    b.extend_from_slice(&r1cs.statement_digest());
    b.extend_from_slice(&commitment.root);
    pfv(&mut b, &z_packed); pfv(&mut b, &a_packed); pfv(&mut b, &b_packed);
    pu(&mut b, z_lincheck.len()); b.extend_from_slice(&z_lincheck);
    // zerocheck + lincheck
    let zc = &proof.zerocheck;
    pfv(&mut b, &zc.round1_ab); pfv(&mut b, &zc.round1_c); ppair(&mut b, &zc.multilinear_rounds);
    pf(&mut b, &zc.final_a_eval); pf(&mut b, &zc.final_b_eval); pf(&mut b, &zc.final_c_eval);
    let lc = &proof.lincheck; ppair(&mut b, &lc.rounds); pfv(&mut b, &lc.z_partial);
    // shift sumcheck (the chain glue)
    ppair(&mut b, &proof.shift.rounds); pf(&mut b, &proof.shift.g_at_point);
    // BatchOpeningProofLigerito
    pu(&mut b, proof.pcs_open.ring_switches.len());
    for rs in &proof.pcs_open.ring_switches { pfv(&mut b, &rs.s_hat_v); }
    let lig = &proof.pcs_open.ligerito;
    b.extend_from_slice(&lig.initial_root);
    prows_f(&mut b, &lig.initial_proof.opened_rows); phv(&mut b, &lig.initial_proof.merkle_proof);
    phv(&mut b, &lig.recursive_roots);
    pu(&mut b, lig.recursive_proofs.len());
    for rp in &lig.recursive_proofs { prows_f(&mut b, &rp.opened_rows); phv(&mut b, &rp.merkle_proof); }
    pfv(&mut b, &lig.final_proof.yr); prows_f(&mut b, &lig.final_proof.opened_rows); phv(&mut b, &lig.final_proof.merkle_proof);
    pu(&mut b, lig.sumcheck_transcript.len());
    for scm in &lig.sumcheck_transcript { pf(&mut b, &scm.u_0); pf(&mut b, &scm.u_2); }
    pu64v(&mut b, &lig.grinding_nonces); pfv(&mut b, &lig.ood_values); pu64v(&mut b, &lig.fold_grinding_nonces);

    std::fs::File::create(&out).unwrap().write_all(&b).unwrap();
    println!("dumped keccak-chain n_keccaks={n_keccaks} m={m} log_n={log_n} | shift_rounds={} \
              ring_switches={} lig sumcheck_msgs={} -> {out}",
             proof.shift.rounds.len(), proof.pcs_open.ring_switches.len(), lig.sumcheck_transcript.len());
}
