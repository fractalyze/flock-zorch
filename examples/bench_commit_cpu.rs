//! CPU PCS-commit phase bench — the commit anchor for the GPU-vs-CPU gate.
//!
//! Runs flock's `pcs::commit` best-of-N at `(m, log_inv_rate, log_batch_size)`
//! with the internal NTT/Merkle timing enabled, and prints
//! `CMTCPU <m> <ntt_best_ms> <merkle_best_ms> <total_best_ms>`. The NTT is the
//! 96-322x-dominant phase (see pcs_commit bench); Merkle is the <1% tail.
//!
//! Usage: `cargo run --release --example bench_commit_cpu -- <m> <log_inv_rate> <log_batch_size> [iters]`

use std::time::Instant;

use flock_core::field::F128;
use flock_core::ntt::AdditiveNttF128;
use flock_core::pcs::commit::PcsParams;
use flock_core::pcs::pack::pack_witness;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let m: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let log_inv_rate: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let log_batch_size: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);

    let mut s: u64 = 0xC0117_1AB1E ^ (m as u64).wrapping_mul(0x1000_0001);
    let z: Vec<bool> = (0..(1usize << m)).map(|_| splitmix64(&mut s) & 1 == 1).collect();
    let z_packed: Vec<F128> = pack_witness(&z, m);
    let params = PcsParams { m, log_inv_rate, log_batch_size, profile: Default::default() };

    let codeword_len = params.n_positions() * params.num_ntts();
    let n_pos_msg = 1usize << params.log_dim();
    let num_ntts = params.num_ntts();

    // Time the commit's two phases directly (mirrors finalize_commit) so we get a
    // clean best-of-N per phase. Encoding = zero-pad + interleaved forward NTT.
    let ntt = AdditiveNttF128::standard(params.k_code());
    let (mut ntt_best, mut mrk_best, mut tot_best) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut sink = 0u64;
    for _ in 0..(iters + 1) {
        // Fresh zero-padded codeword each iter (SoA: first n_pos_msg positions).
        let mut codeword = vec![F128::ZERO; codeword_len];
        codeword[..n_pos_msg * num_ntts].copy_from_slice(&z_packed);

        let t_ntt = Instant::now();
        ntt.forward_transform_interleaved(&mut codeword, num_ntts);
        let dt_ntt = t_ntt.elapsed().as_secs_f64();

        let codeword_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(codeword.as_ptr() as *const u8, codeword.len() * 16)
        };
        let t_mrk = Instant::now();
        let tree = flock_core::merkle::merkle_tree(codeword_bytes, params.n_leaves());
        let dt_mrk = t_mrk.elapsed().as_secs_f64();
        sink ^= u64::from_le_bytes(tree.last().unwrap()[..8].try_into().unwrap());

        ntt_best = ntt_best.min(dt_ntt);
        mrk_best = mrk_best.min(dt_mrk);
        tot_best = tot_best.min(dt_ntt + dt_mrk);
    }
    println!(
        "CMTCPU {} {:.4} {:.4} {:.4}  [cs={:016x}]",
        m, ntt_best * 1e3, mrk_best * 1e3, tot_best * 1e3, sink
    );
}
