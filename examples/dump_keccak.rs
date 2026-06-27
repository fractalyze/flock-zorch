//! Golden dumper for flock's single Keccak-f[1600] R1CS prover (BaseFold) — the
//! keccak port (task #14). Unlike sha2, keccak's A_0/B_0 are EMPTY stubs; the
//! lincheck constraints live in the procedural KeccakLincheckCircuit walker. So
//! we dump WALKER PROBES (fold_alpha_batched on random (alpha, eq)) to gate the
//! Python walker port directly, plus the ingested witness + full R1csProof.
//!
//! n_keccaks=8 → n_blocks_log=3 → m=19 (K_LOG=16, Z_CONST=4096). BaseFold path
//! (Ligerito needs m>=22 = a config that exists; BaseFold is config-free).
//!
//! Usage: `cargo run --release --example dump_keccak -- [n_keccaks] [out]`

use std::io::Write;

use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::lincheck::LincheckCircuit;
use flock_prover::r1cs_hashes::keccak::{self, KeccakLincheckCircuit, KeccakSetup, State, K, STATE_BITS};

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
fn phv(b: &mut Vec<u8>, v: &[[u8; 32]]) { pu(b, v.len()); for h in v { b.extend_from_slice(h); } }

fn main() {
    let mut a = std::env::args().skip(1);
    let n_keccaks: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let out = a.next().unwrap_or_else(|| "artifacts/keccak_golden.bin".to_string());

    let setup = KeccakSetup::new(n_keccaks);
    let r1cs = &setup.r1cs;
    let m = r1cs.m;
    let n_blocks_log = m - r1cs.k_log;

    let mut s: u64 = 0x6EC_C0DEu64 ^ (n_keccaks as u64);
    let states: Vec<State> = (0..n_keccaks).map(|_| {
        let mut st = [false; STATE_BITS];
        for bit in st.iter_mut() { *bit = sm(&mut s) & 1 == 1; }
        st
    }).collect();
    let (z_packed, a_packed, b_packed, z_lincheck) =
        keccak::generate_witness_with_ab_packed_and_lincheck(&states, n_blocks_log);

    let mut ch = FsChallenger::new(b"flock-keccak-v0");
    let (proof, commitment, _claim) = setup.prove_fast_basefold(&states, &mut ch);

    // Walker probes: fold_alpha_batched(alpha, eq) for random (alpha, eq) — the M1 gate.
    let circ = KeccakLincheckCircuit;
    let n_probes = 2usize;
    let probes: Vec<(F128, Vec<F128>, Vec<F128>)> = (0..n_probes).map(|_| {
        let alpha = rf(&mut s);
        let eq: Vec<F128> = (0..K).map(|_| rf(&mut s)).collect();
        let comb = circ.fold_alpha_batched(alpha, &eq);
        (alpha, eq, comb)
    }).collect();

    let mut b = Vec::new();
    b.extend_from_slice(b"FLKKEC01");
    for v in [m, r1cs.k_log, r1cs.k_skip, r1cs.useful_bits,
              r1cs.const_pin.unwrap_or(keccak::Z_CONST),
              setup.pcs_params.log_inv_rate, setup.pcs_params.log_batch_size, n_blocks_log, K] { pu(&mut b, v); }
    b.extend_from_slice(&r1cs.statement_digest());
    b.extend_from_slice(&commitment.root);
    pfv(&mut b, &z_packed); pfv(&mut b, &a_packed); pfv(&mut b, &b_packed);
    pu(&mut b, z_lincheck.len()); b.extend_from_slice(&z_lincheck);
    // walker probes
    pu(&mut b, probes.len());
    for (alpha, eq, comb) in &probes { pf(&mut b, alpha); pfv(&mut b, eq); pfv(&mut b, comb); }
    // R1csProof (BaseFold) — same layout as dump_sha2
    let zc = &proof.zerocheck;
    pfv(&mut b, &zc.round1_ab); pfv(&mut b, &zc.round1_c); ppair(&mut b, &zc.multilinear_rounds);
    pf(&mut b, &zc.final_a_eval); pf(&mut b, &zc.final_b_eval); pf(&mut b, &zc.final_c_eval);
    let lc = &proof.lincheck; ppair(&mut b, &lc.rounds); pfv(&mut b, &lc.z_partial);
    pu(&mut b, proof.pcs_open.ring_switches.len());
    for rs in &proof.pcs_open.ring_switches { pfv(&mut b, &rs.s_hat_v); }
    let bf = &proof.pcs_open.basefold;
    pu(&mut b, bf.round_messages.len());
    for rm in &bf.round_messages { pf(&mut b, &rm.u_0); pf(&mut b, &rm.u_2); }
    b.extend_from_slice(&bf.post_row_batch_commit.root);
    phv(&mut b, &bf.round_commitments.iter().map(|c| c.root).collect::<Vec<_>>());
    pf(&mut b, &bf.final_a); pf(&mut b, &bf.final_b); pfv(&mut b, &bf.final_codeword);
    pu(&mut b, bf.queries.len());
    for q in &bf.queries {
        pu(&mut b, q.position); pfv(&mut b, &q.initial_leaf); pfv(&mut b, &q.post_row_batch_leaf);
        pu(&mut b, q.epoch_leaves.len()); for el in &q.epoch_leaves { pfv(&mut b, el); }
    }
    phv(&mut b, &bf.initial_multi_proof); phv(&mut b, &bf.post_row_batch_multi_proof);
    pu(&mut b, bf.epoch_multi_proofs.len());
    for mp in &bf.epoch_multi_proofs { phv(&mut b, mp); }

    std::fs::File::create(&out).unwrap().write_all(&b).unwrap();
    println!("dumped keccak n_keccaks={n_keccaks} m={m} k_log={} k_skip={} ub={} const_pin={:?} | \
              probes={} zc_rounds={} lc_rounds={} ring_switches={} bf_rounds={} -> {out}",
             r1cs.k_log, r1cs.k_skip, r1cs.useful_bits, r1cs.const_pin, probes.len(),
             zc.multilinear_rounds.len(), lc.rounds.len(), proof.pcs_open.ring_switches.len(),
             bf.round_messages.len());
}
