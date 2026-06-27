//! Golden dumper for flock's Keccak-f[1600] R1CS prover with the LIGERITO PCS —
//! the headline keccak backend. n_keccaks=64 → m=22 → m22_fast config (log_n=15,
//! initial_k=6, R=2).
//!
//! Keccak's A_0/B_0 are EMPTY stubs, so a/b come from the witness generator (NOT
//! apply_a/apply_b, which would be zero) and the lincheck circuit is the
//! procedural KeccakLincheckCircuit — no a0/b0 rows are dumped. Otherwise the
//! layout matches dump_sha2_ligerito.rs (config + witness + zerocheck + lincheck
//! + BatchOpeningProofLigerito).
//!
//! Usage: `cargo run --release --example dump_keccak_ligerito -- [n_keccaks] [out]`

use std::io::Write;

use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::pcs::ligerito::{prover_config_for, LigeritoProfile};
use flock_prover::r1cs_hashes::keccak::{self, KeccakSetup, State, STATE_BITS};

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
    let mut a = std::env::args().skip(1);
    let n_keccaks: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let out = a.next().unwrap_or_else(|| "artifacts/keccak_ligerito_golden.bin".to_string());

    let setup = KeccakSetup::new(n_keccaks);
    let r1cs = &setup.r1cs;
    let m = r1cs.m;
    let n_blocks_log = m - r1cs.k_log;
    let log_n = m - 7;
    let cfg = prover_config_for(log_n, setup.pcs_params.log_batch_size, LigeritoProfile::Fast).expect("cfg");

    let mut s: u64 = 0x6EC_C0DEu64 ^ (n_keccaks as u64);
    let states: Vec<State> = (0..n_keccaks).map(|_| {
        let mut st = [false; STATE_BITS];
        for bit in st.iter_mut() { *bit = sm(&mut s) & 1 == 1; }
        st
    }).collect();
    let (z_packed, a_packed, b_packed, z_lincheck) =
        keccak::generate_witness_with_ab_packed_and_lincheck(&states, n_blocks_log);

    let mut ch = FsChallenger::new(b"flock-keccak-lig-v0");
    let (proof, commitment, _claim) = setup.prove_fast(&states, &mut ch); // Ligerito path

    let mut b = Vec::new();
    b.extend_from_slice(b"FLKKL_01");
    for v in [m, r1cs.k_log, r1cs.k_skip, r1cs.useful_bits,
              r1cs.const_pin.unwrap_or(keccak::Z_CONST),
              setup.pcs_params.log_inv_rate, setup.pcs_params.log_batch_size, n_blocks_log, log_n] { pu(&mut b, v); }
    // Ligerito config
    puv(&mut b, &cfg.log_inv_rates); pu(&mut b, cfg.recursive_steps);
    pu(&mut b, cfg.initial_log_msg_cols); pu(&mut b, cfg.initial_log_num_interleaved); pu(&mut b, cfg.initial_k);
    puv(&mut b, &cfg.recursive_log_msg_cols); puv(&mut b, &cfg.recursive_ks); puv(&mut b, &cfg.queries);
    puv(&mut b, &cfg.grinding_bits); puv(&mut b, &cfg.fold_grinding_bits); puv(&mut b, &cfg.ood_samples);
    // witness ingestion (empty A_0/B_0 → no a0/b0 rows; circuit is the walker)
    b.extend_from_slice(&r1cs.statement_digest());
    b.extend_from_slice(&commitment.root);
    pfv(&mut b, &z_packed); pfv(&mut b, &a_packed); pfv(&mut b, &b_packed);
    pu(&mut b, z_lincheck.len()); b.extend_from_slice(&z_lincheck);
    // zerocheck + lincheck (same as BaseFold prove up to the open)
    let zc = &proof.zerocheck;
    pfv(&mut b, &zc.round1_ab); pfv(&mut b, &zc.round1_c); ppair(&mut b, &zc.multilinear_rounds);
    pf(&mut b, &zc.final_a_eval); pf(&mut b, &zc.final_b_eval); pf(&mut b, &zc.final_c_eval);
    let lc = &proof.lincheck; ppair(&mut b, &lc.rounds); pfv(&mut b, &lc.z_partial);
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
    println!("dumped keccak-ligerito n_keccaks={n_keccaks} m={m} log_n={log_n} | R={} initial_k={} | \
              ring_switches={} lig sumcheck_msgs={} recursive_roots={} -> {out}",
             cfg.recursive_steps, cfg.initial_k, proof.pcs_open.ring_switches.len(),
             lig.sumcheck_transcript.len(), lig.recursive_roots.len());
}
