//! Golden-fixture dumper for flock's REAL SHA-256 R1CS prover (BaseFold path) —
//! the gate for flock-zorch's GPU sha256 prover.
//!
//! Builds the sha2 hybrid R1CS (`Sha256HybridSetup`, K_LOG=15, K_SKIP=6,
//! USEFUL_BITS=31401, const_pin=Z_CONST_POS=31400), generates a real multi-block
//! witness (one SHA-256 compression per k-block), and runs flock's top-level
//! `prove` (BaseFold: commit→bind→zerocheck→lincheck→batched dual-claim open).
//!
//! Dumps everything the GPU prover needs to INGEST (witness gen + the R1CS are
//! host setup, ~2% of prove, not the GPU target) + the full R1csProof to gate:
//!   - R1CS metadata (m,k_log,k_skip,useful_bits,const_pin,lir,lbs), statement_digest
//!   - z_packed (witness), a_packed=A·z, b_packed=B·z, z_lincheck (stripe)
//!   - a_0 / b_0 sparse rows (for the CSC lincheck fold + the GPU matvec)
//!   - the full R1csProof (zerocheck + lincheck + batched BaseFold open)
//!
//! Usage: `cargo run --release --example dump_sha2 -- [n_compressions] [out]`

use std::io::Write;

use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::lincheck::pack_z_lincheck_from_packed;
use flock_core::pcs::pack::pack_witness;
use flock_prover::prover::prove;
use flock_prover::r1cs_hashes::sha2;

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
fn ppair(b: &mut Vec<u8>, v: &[(F128, F128)]) { pu(b, v.len()); for (x, y) in v { pf(b, x); pf(b, y); } }
fn phv(b: &mut Vec<u8>, v: &[[u8; 32]]) { pu(b, v.len()); for h in v { b.extend_from_slice(h); } }
fn prows(b: &mut Vec<u8>, rows: &[Vec<usize>]) {
    pu(b, rows.len());
    for r in rows { pu(b, r.len()); for &c in r { b.extend_from_slice(&(c as u32).to_le_bytes()); } }
}

fn main() {
    let mut a = std::env::args().skip(1);
    let n_comp: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let out = a.next().unwrap_or_else(|| "artifacts/sha2_golden.bin".to_string());

    let setup = sha2::Sha256HybridSetup::new(n_comp);
    let r1cs = &setup.r1cs;
    let m = r1cs.m;
    let n_blocks_log = setup.n_blocks_log();

    // Real witness: one deterministic SHA-256 compression per k-block.
    let mut s: u64 = 0x5A2A_0627u64 ^ (n_comp as u64);
    let comps: Vec<([u32; 8], [u32; 16])> = (0..n_comp).map(|_| {
        let mut h = [0u32; 8]; for x in &mut h { *x = splitmix64(&mut s) as u32; }
        let mut msg = [0u32; 16]; for x in &mut msg { *x = splitmix64(&mut s) as u32; }
        (h, msg)
    }).collect();
    let z = sha2::generate_witness(&comps, n_blocks_log);
    assert!(r1cs.satisfies(&z), "sha2 witness must satisfy the R1CS");
    let z_packed = pack_witness(&z, m);

    let a_packed = r1cs.apply_a_packed(&z_packed);
    let b_packed = r1cs.apply_b_packed(&z_packed);
    let z_lincheck = pack_z_lincheck_from_packed(&z_packed, m, r1cs.k_log);

    let mut ch = FsChallenger::new(b"flock-sha2-v0");
    let (proof, commitment, _claim) = prove(r1cs, &z_packed, &setup.pcs_params, &mut ch);

    // ---- dump ----
    let mut b = Vec::new();
    b.extend_from_slice(b"FLKSHA01");
    for v in [m, r1cs.k_log, r1cs.k_skip, r1cs.useful_bits,
              r1cs.const_pin.unwrap_or(usize::MAX),
              setup.pcs_params.log_inv_rate, setup.pcs_params.log_batch_size, n_blocks_log] { pu(&mut b, v); }
    b.extend_from_slice(&r1cs.statement_digest());
    b.extend_from_slice(&commitment.root);
    pfv(&mut b, &z_packed);
    pfv(&mut b, &a_packed);
    pfv(&mut b, &b_packed);
    pu(&mut b, z_lincheck.len()); b.extend_from_slice(&z_lincheck);
    prows(&mut b, &r1cs.a_0.rows);
    prows(&mut b, &r1cs.b_0.rows);
    // R1csProof (same layout as dump_e2e)
    let zc = &proof.zerocheck;
    pfv(&mut b, &zc.round1_ab); pfv(&mut b, &zc.round1_c);
    ppair(&mut b, &zc.multilinear_rounds);
    pf(&mut b, &zc.final_a_eval); pf(&mut b, &zc.final_b_eval); pf(&mut b, &zc.final_c_eval);
    let lc = &proof.lincheck;
    ppair(&mut b, &lc.rounds); pfv(&mut b, &lc.z_partial);
    pu(&mut b, proof.pcs_open.ring_switches.len());
    for rs in &proof.pcs_open.ring_switches { pfv(&mut b, &rs.s_hat_v); }
    let bf = &proof.pcs_open.basefold;
    pu(&mut b, bf.round_messages.len());
    for rm in &bf.round_messages { pf(&mut b, &rm.u_0); pf(&mut b, &rm.u_2); }
    b.extend_from_slice(&bf.post_row_batch_commit.root);
    phv(&mut b, &bf.round_commitments.iter().map(|c| c.root).collect::<Vec<_>>());
    pf(&mut b, &bf.final_a); pf(&mut b, &bf.final_b);
    pfv(&mut b, &bf.final_codeword);
    pu(&mut b, bf.queries.len());
    for q in &bf.queries {
        pu(&mut b, q.position); pfv(&mut b, &q.initial_leaf); pfv(&mut b, &q.post_row_batch_leaf);
        pu(&mut b, q.epoch_leaves.len()); for el in &q.epoch_leaves { pfv(&mut b, el); }
    }
    phv(&mut b, &bf.initial_multi_proof);
    phv(&mut b, &bf.post_row_batch_multi_proof);
    pu(&mut b, bf.epoch_multi_proofs.len());
    for mp in &bf.epoch_multi_proofs { phv(&mut b, mp); }

    std::fs::File::create(&out).unwrap().write_all(&b).unwrap();
    let nnz_a: usize = r1cs.a_0.rows.iter().map(|r| r.len()).sum();
    let nnz_b: usize = r1cs.b_0.rows.iter().map(|r| r.len()).sum();
    println!("dumped sha2 e2e n_comp={n_comp} m={m} k_log={} k_skip={} ub={} const_pin={:?} lbs={} | \
              nnz(a_0)={nnz_a} nnz(b_0)={nnz_b} | zc_rounds={} lc_rounds={} ring_switches={} bf_rounds={} epochs={} -> {out}",
             r1cs.k_log, r1cs.k_skip, r1cs.useful_bits, r1cs.const_pin, setup.pcs_params.log_batch_size,
             zc.multilinear_rounds.len(), lc.rounds.len(), proof.pcs_open.ring_switches.len(),
             bf.round_messages.len(), bf.round_commitments.len());
}
