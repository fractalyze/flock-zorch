//! Golden dumper for flock's FULL R1CS prover with the LIGERITO PCS on the
//! identity R1CS — the e2e gate for flock-zorch's fused `prove_fast` (Ligerito
//! backend). Identity R1CS: A_0=B_0=C_0=I, so a=A·z=z, b=z, c=z; any bit vector
//! satisfies it. Uses `flock_prover::prover::prove_fast_ligerito_from_witness`
//! (the generic Ligerito prove) and serializes the full R1csProofLigerito,
//! mirroring dump_keccak3_ligerito.rs.
//!
//! Usage: `cargo run --release --example dump_e2e_ligerito -- [m] [out]`

use std::io::Write;
use std::sync::OnceLock;

use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::lincheck::pack_z_lincheck_from_packed;
use flock_core::pcs::commit::PcsParams;
use flock_core::pcs::ligerito::{prover_config_for, LigeritoProfile};
use flock_core::pcs::pack::pack_witness;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix};
use flock_prover::prover::prove_fast_ligerito_from_witness;

fn splitmix64(s: &mut u64) -> u64 {
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

fn identity(k: usize) -> SparseBinaryMatrix {
    SparseBinaryMatrix { num_rows: k, num_cols: k, rows: (0..k).map(|i| vec![i]).collect() }
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(22);
    let out = a.next().unwrap_or_else(|| "artifacts/e2e_ligerito_golden.bin".to_string());
    let (k_log, k_skip, useful_bits) = (6usize, 6usize, 64usize);

    let r1cs = BlockR1cs {
        m, k_log, k_skip, useful_bits,
        a_0: identity(1 << k_log), b_0: identity(1 << k_log), c_0: identity(1 << k_log),
        const_pin: None, digest_cache: OnceLock::new(), csc_cache: OnceLock::new(),
    };
    let pcs_params = PcsParams {
        m, log_inv_rate: 1, log_batch_size: 6, profile: LigeritoProfile::Fast,
    };
    let n_blocks_log = m - k_log;
    let log_n = m - 7;
    let cfg = prover_config_for(log_n, pcs_params.log_batch_size, LigeritoProfile::Fast).expect("cfg");

    let mut s: u64 = 0x20_2406_27 ^ (m as u64).wrapping_mul(0x1000_0001);
    let z: Vec<bool> = (0..r1cs.n()).map(|_| splitmix64(&mut s) & 1 == 1).collect();
    assert!(r1cs.satisfies(&z), "identity R1CS must accept any bit vector");
    let z_packed = pack_witness(&z, m);

    let a_packed_f128 = r1cs.apply_a_packed(&z_packed);
    let b_packed_f128 = r1cs.apply_b_packed(&z_packed);
    let z_packed_lincheck = pack_z_lincheck_from_packed(&z_packed, m, k_log);
    let lc_circuit = r1cs.csc_lincheck_circuit();

    let mut ch = FsChallenger::new(b"flock-e2e-lig-v0");
    let (proof, commitment, claim) = prove_fast_ligerito_from_witness(
        &r1cs, &pcs_params, z_packed.clone(), a_packed_f128.clone(), b_packed_f128.clone(),
        z_packed_lincheck.clone(), lc_circuit, None, &mut ch);

    // ---- dump (mirrors dump_keccak3_ligerito.rs) ----
    let mut b = Vec::new();
    b.extend_from_slice(b"FLKE2L01");
    for v in [m, k_log, k_skip, useful_bits,
              pcs_params.log_inv_rate, pcs_params.log_batch_size, n_blocks_log, log_n] { pu(&mut b, v); }
    // Ligerito config
    puv(&mut b, &cfg.log_inv_rates); pu(&mut b, cfg.recursive_steps);
    pu(&mut b, cfg.initial_log_msg_cols); pu(&mut b, cfg.initial_log_num_interleaved); pu(&mut b, cfg.initial_k);
    puv(&mut b, &cfg.recursive_log_msg_cols); puv(&mut b, &cfg.recursive_ks); puv(&mut b, &cfg.queries);
    puv(&mut b, &cfg.grinding_bits); puv(&mut b, &cfg.fold_grinding_bits); puv(&mut b, &cfg.ood_samples);
    // witness
    b.extend_from_slice(&r1cs.statement_digest());
    b.extend_from_slice(&commitment.root);
    pfv(&mut b, &z_packed); pfv(&mut b, &a_packed_f128); pfv(&mut b, &b_packed_f128);
    pu(&mut b, z_packed_lincheck.len()); b.extend_from_slice(&z_packed_lincheck);
    // zerocheck + lincheck
    let zc = &proof.zerocheck;
    pfv(&mut b, &zc.round1_ab); pfv(&mut b, &zc.round1_c); ppair(&mut b, &zc.multilinear_rounds);
    pf(&mut b, &zc.final_a_eval); pf(&mut b, &zc.final_b_eval); pf(&mut b, &zc.final_c_eval);
    let lc = &proof.lincheck; ppair(&mut b, &lc.rounds); pfv(&mut b, &lc.z_partial);
    // ab / c claim values (the two z-claims the open reduces)
    pf(&mut b, &claim.ab.value); pf(&mut b, &claim.c.value);
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
    println!("dumped e2e-ligerito (identity) m={m} log_n={log_n} | R={} initial_k={} | \
              zc rounds={} lc rounds={} ring_switches={} lig sumcheck_msgs={} -> {out}",
             cfg.recursive_steps, cfg.initial_k, zc.multilinear_rounds.len(), lc.rounds.len(),
             proof.pcs_open.ring_switches.len(), lig.sumcheck_transcript.len());
}
