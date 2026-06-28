//! Golden dumper for flock's BLAKE3 R1CS prover with the LIGERITO PCS — the
//! HEADLINE path (blake3_proof.rs uses Ligerito, not BaseFold), mirroring
//! dump_sha2_ligerito.rs.
//!
//! Same real blake3 R1CS + witness as dump_blake3.rs, but `prove_ligerito` → the
//! recursive Ligerito open. n_comp=256 → m=22 (K_LOG=14) → log_n=15, the same
//! Ligerito config as sha2's m=22 (initial_k = log_batch_size = 6). Dumps the
//! config + witness/matrices + zerocheck + lincheck + the
//! BatchOpeningProofLigerito (ring_switches + full LigeritoProof).
//!
//! Usage: `cargo run --release --example dump_blake3_ligerito -- [n_comp] [out]`

use std::io::Write;

use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::lincheck::pack_z_lincheck_from_packed;
use flock_core::pcs::ligerito::{prover_config_for, LigeritoProfile};
use flock_core::pcs::pack::pack_witness;
use flock_prover::prover::prove_ligerito;
use flock_prover::r1cs_hashes::blake3;

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
fn prows_u(b: &mut Vec<u8>, rows: &[Vec<usize>]) { pu(b, rows.len()); for r in rows { pu(b, r.len()); for &c in r { b.extend_from_slice(&(c as u32).to_le_bytes()); } } }

fn main() {
    let mut a = std::env::args().skip(1);
    let n_comp: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let out = a.next().unwrap_or_else(|| "artifacts/blake3_ligerito_golden.bin".to_string());

    let setup = blake3::Blake3Setup::new(n_comp);
    let r1cs = &setup.r1cs;
    let m = r1cs.m;
    let log_n = m - 7;
    let cfg = prover_config_for(log_n, setup.pcs_params.log_batch_size, LigeritoProfile::Fast).expect("cfg");

    let mut s: u64 = 0xB1A3_0627u64 ^ (n_comp as u64);
    let blocks: Vec<blake3::Compression> = (0..n_comp).map(|_| {
        let mut cv = [0u32; 8]; for x in &mut cv { *x = splitmix64(&mut s) as u32; }
        let mut msg = [0u32; 16]; for x in &mut msg { *x = splitmix64(&mut s) as u32; }
        let counter = splitmix64(&mut s);
        let block_len = splitmix64(&mut s) as u32;
        let flags = splitmix64(&mut s) as u32;
        (cv, msg, counter, block_len, flags)
    }).collect();
    let z = blake3::generate_witness(&blocks, setup.n_blocks_log());
    assert!(r1cs.satisfies(&z));
    let z_packed = pack_witness(&z, m);
    let a_packed = r1cs.apply_a_packed(&z_packed);
    let b_packed = r1cs.apply_b_packed(&z_packed);
    let z_lincheck = pack_z_lincheck_from_packed(&z_packed, m, r1cs.k_log);

    let mut ch = FsChallenger::new(b"flock-blake3-lig-v0");
    let (proof, commitment, _claim) = prove_ligerito(r1cs, z_packed.clone(), &setup.pcs_params, &mut ch);

    let mut b = Vec::new();
    b.extend_from_slice(b"FLKBL_01");
    for v in [m, r1cs.k_log, r1cs.k_skip, r1cs.useful_bits, r1cs.const_pin.unwrap_or(usize::MAX),
              setup.pcs_params.log_inv_rate, setup.pcs_params.log_batch_size, setup.n_blocks_log(), log_n] { pu(&mut b, v); }
    // Ligerito config
    puv(&mut b, &cfg.log_inv_rates); pu(&mut b, cfg.recursive_steps);
    pu(&mut b, cfg.initial_log_msg_cols); pu(&mut b, cfg.initial_log_num_interleaved); pu(&mut b, cfg.initial_k);
    puv(&mut b, &cfg.recursive_log_msg_cols); puv(&mut b, &cfg.recursive_ks); puv(&mut b, &cfg.queries);
    puv(&mut b, &cfg.grinding_bits); puv(&mut b, &cfg.fold_grinding_bits); puv(&mut b, &cfg.ood_samples);
    // witness + R1CS ingestion
    b.extend_from_slice(&r1cs.statement_digest());
    b.extend_from_slice(&commitment.root);
    pfv(&mut b, &z_packed); pfv(&mut b, &a_packed); pfv(&mut b, &b_packed);
    pu(&mut b, z_lincheck.len()); b.extend_from_slice(&z_lincheck);
    prows_u(&mut b, &r1cs.a_0.rows); prows_u(&mut b, &r1cs.b_0.rows);
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
    for sm in &lig.sumcheck_transcript { pf(&mut b, &sm.u_0); pf(&mut b, &sm.u_2); }
    pu64v(&mut b, &lig.grinding_nonces); pfv(&mut b, &lig.ood_values); pu64v(&mut b, &lig.fold_grinding_nonces);

    std::fs::File::create(&out).unwrap().write_all(&b).unwrap();
    println!("dumped blake3-ligerito n_comp={n_comp} m={m} log_n={log_n} | R={} initial_k={} | \
              ring_switches={} lig sumcheck_msgs={} recursive_roots={} -> {out}",
             cfg.recursive_steps, cfg.initial_k, proof.pcs_open.ring_switches.len(),
             lig.sumcheck_transcript.len(), lig.recursive_roots.len());
}
