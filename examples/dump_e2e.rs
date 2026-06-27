//! Golden-fixture dumper for flock's FULL R1CS prover (`prover::prove`) on the
//! identity R1CS — the e2e gate for flock-zorch's fused `prove_fast`.
//!
//! Replicates the body of `flock_prover::prover::prove` using only `flock-core`
//! public functions (flock-zorch deps flock-core, not flock-prover), so we can
//! dump EVERY intermediate for stage-by-stage byte gating: statement_digest,
//! witness z, commitment root, the zerocheck proof+claim+s_hat_v_c, the lincheck
//! proof+claim+captured z_vec, the two z-claims (ab, c), and the batched PCS
//! open (ring_switches + one shared BaseFold).
//!
//! Identity R1CS: A_0=B_0=C_0=I, so a=A·z=z, b=z, c=z; any bit vector satisfies
//! it. Matches `verifier_roundtrip.rs::r1cs_prove_verify_roundtrip_honest`
//! (k_log=6, k_skip=6, useful_bits=64) → inner_rest_len=0, s_hat_v_ab=None
//! (k_log < LOG_PACKING=7).
//!
//! Usage: `cargo run --release --example dump_e2e -- [m] [out]`

use std::io::Write;
use std::sync::OnceLock;

use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::lincheck::{self, pack_z_lincheck_from_packed, QuirkyPoint};
use flock_core::pcs::{self, commit::PcsParams, pack::pack_witness};
use flock_core::proof::{bind_statement, ZClaim};
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix};
use flock_core::zerocheck::{self, PaddingSpec};

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

fn identity(k: usize) -> SparseBinaryMatrix {
    SparseBinaryMatrix { num_rows: k, num_cols: k, rows: (0..k).map(|i| vec![i]).collect() }
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(13);
    let out = a.next().unwrap_or_else(|| "artifacts/e2e_golden.bin".to_string());
    let (k_log, k_skip, useful_bits) = (6usize, 6usize, 64usize);

    let r1cs = BlockR1cs {
        m, k_log, k_skip, useful_bits,
        a_0: identity(1 << k_log), b_0: identity(1 << k_log), c_0: identity(1 << k_log),
        const_pin: None, digest_cache: OnceLock::new(), csc_cache: OnceLock::new(),
    };
    let pcs_params = PcsParams { m, log_inv_rate: 1, log_batch_size: 5, profile: Default::default() };

    let mut s: u64 = 0x20_2406_27 ^ (m as u64).wrapping_mul(0x1000_0001);
    let z: Vec<bool> = (0..r1cs.n()).map(|_| splitmix64(&mut s) & 1 == 1).collect();
    assert!(r1cs.satisfies(&z), "identity R1CS must accept any bit vector");
    let z_packed = pack_witness(&z, m);

    // ---- replicate prover::prove on ONE challenger ----
    let mut ch = FsChallenger::new(b"flock-test-v0");
    let (commitment, prover_data) = pcs::commit(&z_packed, &pcs_params);
    bind_statement(&mut ch, &r1cs, &commitment);

    let a_packed_f128 = r1cs.apply_a_packed(&z_packed);
    let b_packed_f128 = r1cs.apply_b_packed(&z_packed);
    let cast = |v: &[F128]| -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    };
    let a_packed = cast(&a_packed_f128);
    let b_packed = cast(&b_packed_f128);
    let c_packed = cast(&z_packed); // C = I
    let z_packed_lincheck = pack_z_lincheck_from_packed(&z_packed, m, k_log);

    let padding = PaddingSpec { k_log, useful_bits_per_block: useful_bits };
    let (zc_proof, zc_claim, s_hat_v_c) =
        zerocheck::prove_packed_padded_capture_s_hat_v_c(a_packed, b_packed, c_packed, m, &padding, &mut ch);

    let inner_rest_len = k_log - k_skip; // 0
    let x_ab = QuirkyPoint {
        z_skip: zc_claim.z,
        x_inner_rest: zc_claim.mlv_challenges[..inner_rest_len].to_vec(),
        x_outer: zc_claim.mlv_challenges[inner_rest_len..].to_vec(),
    };
    let lc_circuit = r1cs.csc_lincheck_circuit();
    let (lc_proof, lc_claim, z_vec_pre) = lincheck::prove_padded_capture_z_vec(
        &z_packed_lincheck, m, k_log, k_skip, useful_bits, lc_circuit, &x_ab, &mut ch);

    let ab = ZClaim {
        point: QuirkyPoint {
            z_skip: lc_claim.r_inner_skip,
            x_inner_rest: lc_claim.r_inner_rest.clone(),
            x_outer: x_ab.x_outer.clone(),
        },
        value: lc_claim.w,
    };
    let c = ZClaim {
        point: QuirkyPoint {
            z_skip: zc_claim.z,
            x_inner_rest: zc_claim.r_rest[..inner_rest_len].to_vec(),
            x_outer: zc_claim.r_rest[inner_rest_len..].to_vec(),
        },
        value: zc_claim.c_eval,
    };

    // s_hat_v_ab is None (k_log=6 < LOG_PACKING). pre_c = Some(s_hat_v_c).
    let qf = |p: &QuirkyPoint| -> Vec<F128> {
        let mut v = Vec::new(); v.extend_from_slice(&p.x_inner_rest); v.extend_from_slice(&p.x_outer); v
    };
    let x_ab_full = qf(&ab.point);
    let x_c_full = qf(&c.point);
    let x_refs: Vec<&[F128]> = vec![&x_ab_full, &x_c_full];
    let pre: Vec<Option<&[F128]>> = vec![None, Some(s_hat_v_c.as_slice())];
    let pcs_open = pcs::open_batch_padded_with_precomputed_s_hat_v(
        &z_packed, &prover_data, &commitment, &x_refs, &pre, &padding, &mut ch);

    // ---- dump ----
    let mut b = Vec::new();
    b.extend_from_slice(b"FLKE2E01");
    for v in [m, k_log, k_skip, useful_bits] { pu(&mut b, v); }
    b.extend_from_slice(&r1cs.statement_digest());
    pfv(&mut b, &z_packed);
    pu(&mut b, z_packed_lincheck.len()); b.extend_from_slice(&z_packed_lincheck);  // stripe-packed z
    b.extend_from_slice(&commitment.root);
    // a/b packed (verify a=b=z for identity)
    pfv(&mut b, &a_packed_f128);
    pfv(&mut b, &b_packed_f128);
    // zerocheck proof
    pfv(&mut b, &zc_proof.round1_ab);
    pfv(&mut b, &zc_proof.round1_c);
    ppair(&mut b, &zc_proof.multilinear_rounds);
    pf(&mut b, &zc_proof.final_a_eval);
    pf(&mut b, &zc_proof.final_b_eval);
    pf(&mut b, &zc_proof.final_c_eval);
    // zerocheck claim
    pf(&mut b, &zc_claim.z);
    pfv(&mut b, &zc_claim.mlv_challenges);
    pfv(&mut b, &zc_claim.r_rest);
    pf(&mut b, &zc_claim.a_eval);
    pf(&mut b, &zc_claim.b_eval);
    pf(&mut b, &zc_claim.c_eval);
    // s_hat_v_c
    pfv(&mut b, &s_hat_v_c);
    // lincheck proof + claim + captured z_vec
    ppair(&mut b, &lc_proof.rounds);
    pfv(&mut b, &lc_proof.z_partial);
    pf(&mut b, &lc_claim.r_inner_skip);
    pfv(&mut b, &lc_claim.r_inner_rest);
    pf(&mut b, &lc_claim.w);
    pfv(&mut b, &z_vec_pre);
    // ab / c claims
    pf(&mut b, &ab.point.z_skip); pfv(&mut b, &ab.point.x_inner_rest); pfv(&mut b, &ab.point.x_outer); pf(&mut b, &ab.value);
    pf(&mut b, &c.point.z_skip);  pfv(&mut b, &c.point.x_inner_rest);  pfv(&mut b, &c.point.x_outer);  pf(&mut b, &c.value);
    // pcs_open: ring_switches + basefold
    pu(&mut b, pcs_open.ring_switches.len());
    for rs in &pcs_open.ring_switches { pfv(&mut b, &rs.s_hat_v); }
    let bf = &pcs_open.basefold;
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
    println!("dumped e2e m={m} k_log={k_log} k_skip={k_skip} ub={useful_bits} | \
              zc rounds={} | lc rounds={} | ring_switches={} | bf rounds={} epochs={} | a==z:{} -> {out}",
             zc_proof.multilinear_rounds.len(), lc_proof.rounds.len(), pcs_open.ring_switches.len(),
             bf.round_messages.len(), bf.round_commitments.len(), a_packed_f128 == z_packed);
}
