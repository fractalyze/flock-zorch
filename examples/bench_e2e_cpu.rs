//! CPU baseline for the fused R1CS prover: times flock's `prove` body (commit +
//! bind + zerocheck + lincheck + batched open) on the SAME identity R1CS the GPU
//! fused prover gates against, so e2e_fused_bench's speedup is apples-to-apples
//! (not vs the blake3 prover). Replicates prove() via flock-core pub fns (matched
//! release profile: thin-LTO + codegen-units=1 + target-cpu=native).
//!
//! Usage: `cargo run --release --example bench_e2e_cpu -- [m ...]`

use std::sync::OnceLock;
use std::time::Instant;

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
fn identity(k: usize) -> SparseBinaryMatrix {
    SparseBinaryMatrix { num_rows: k, num_cols: k, rows: (0..k).map(|i| vec![i]).collect() }
}

fn prove_body(r1cs: &BlockR1cs, z_packed: &[F128], params: &PcsParams) {
    let (k_log, k_skip, ub) = (r1cs.k_log, r1cs.k_skip, r1cs.useful_bits);
    let mut ch = FsChallenger::new(b"flock-test-v0");
    let (commitment, prover_data) = pcs::commit(z_packed, params);
    bind_statement(&mut ch, r1cs, &commitment);
    let a_f = r1cs.apply_a_packed(z_packed);
    let b_f = r1cs.apply_b_packed(z_packed);
    let cast = |v: &[F128]| -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    };
    let zlc = pack_z_lincheck_from_packed(z_packed, r1cs.m, k_log);
    let padding = PaddingSpec { k_log, useful_bits_per_block: ub };
    let (_zp, zc_claim, s_hat_v_c) = zerocheck::prove_packed_padded_capture_s_hat_v_c(
        cast(&a_f), cast(&b_f), cast(z_packed), r1cs.m, &padding, &mut ch);
    let ir = k_log - k_skip;
    let x_ab = QuirkyPoint { z_skip: zc_claim.z,
        x_inner_rest: zc_claim.mlv_challenges[..ir].to_vec(),
        x_outer: zc_claim.mlv_challenges[ir..].to_vec() };
    let (_lp, lc_claim, _zv) = lincheck::prove_padded_capture_z_vec(
        &zlc, r1cs.m, k_log, k_skip, ub, r1cs.csc_lincheck_circuit(), &x_ab, &mut ch);
    let ab = ZClaim { point: QuirkyPoint { z_skip: lc_claim.r_inner_skip,
        x_inner_rest: lc_claim.r_inner_rest.clone(), x_outer: x_ab.x_outer.clone() }, value: lc_claim.w };
    let c = ZClaim { point: QuirkyPoint { z_skip: zc_claim.z,
        x_inner_rest: zc_claim.r_rest[..ir].to_vec(), x_outer: zc_claim.r_rest[ir..].to_vec() },
        value: zc_claim.c_eval };
    let qf = |p: &QuirkyPoint| { let mut v = Vec::new(); v.extend_from_slice(&p.x_inner_rest); v.extend_from_slice(&p.x_outer); v };
    let xab = qf(&ab.point); let xc = qf(&c.point);
    let xr: Vec<&[F128]> = vec![&xab, &xc];
    let pre: Vec<Option<&[F128]>> = vec![None, Some(s_hat_v_c.as_slice())];
    let _open = pcs::open_batch_padded_with_precomputed_s_hat_v(
        z_packed, &prover_data, &commitment, &xr, &pre, &padding, &mut ch);
}

fn main() {
    let ms: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let ms = if ms.is_empty() { vec![26usize] } else { ms };
    let (k_log, k_skip, ub) = (6usize, 6usize, 64usize);
    for m in ms {
        let r1cs = BlockR1cs { m, k_log, k_skip, useful_bits: ub,
            a_0: identity(1 << k_log), b_0: identity(1 << k_log), c_0: identity(1 << k_log),
            const_pin: None, digest_cache: OnceLock::new(), csc_cache: OnceLock::new() };
        let params = PcsParams { m, log_inv_rate: 1, log_batch_size: 5, profile: Default::default() };
        let mut s: u64 = 0x20_2406_27 ^ (m as u64).wrapping_mul(0x1000_0001);
        let z: Vec<bool> = (0..r1cs.n()).map(|_| splitmix64(&mut s) & 1 == 1).collect();
        let z_packed = pack_witness(&z, m);
        prove_body(&r1cs, &z_packed, &params); // warm
        let n = if m >= 26 { 3 } else { 10 };
        let mut best = f64::INFINITY;
        for _ in 0..n {
            let t = Instant::now();
            prove_body(&r1cs, &z_packed, &params);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!("E2ECPU m={m} identity-R1CS CPU prove = {best:.2} ms");
    }
}
