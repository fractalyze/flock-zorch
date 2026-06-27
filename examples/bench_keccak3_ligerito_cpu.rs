//! CPU baseline for the GPU keccak3 LIGERITO prover (the headline keccak path):
//! times flock's keccak3 prover (commit → zerocheck → walker lincheck → recursive
//! Ligerito open) on the same states. Witness gen is done OUTSIDE the timed region
//! (the GPU bench ingests the witness), matching bench_sha2_ligerito_cpu. x86 scalar.
//! Usage: `cargo run --release --example bench_keccak3_ligerito_cpu -- [n_keccaks ...]`
use std::time::Instant;
use flock_core::challenger::FsChallenger;
use flock_prover::prover::prove_fast_ligerito_from_witness;
use flock_prover::r1cs_hashes::keccak::{State, STATE_BITS};
use flock_prover::r1cs_hashes::keccak3::{self, KeccakLincheckCircuit, KeccakSetup};
fn sm(s: &mut u64) -> u64 { *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15); let mut z=*s;
    z=(z^(z>>30)).wrapping_mul(0xBF58_476D_1CE4_E5B9); z=(z^(z>>27)).wrapping_mul(0x94D0_49BB_1331_11EB); z^(z>>31) }
fn main() {
    let ns: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let ns = if ns.is_empty() { vec![49usize] } else { ns };
    for nk in ns {
        let setup = KeccakSetup::new(nk);
        let mut s: u64 = 0x6EC3_C0DEu64 ^ (nk as u64);
        let states: Vec<State> = (0..nk).map(|_| {
            let mut st = [false; STATE_BITS];
            for bit in st.iter_mut() { *bit = sm(&mut s) & 1 == 1; }
            st
        }).collect();
        let (z, a, b, zlc) =
            keccak3::generate_witness_with_ab_packed_and_lincheck(&states, setup.n_blocks_log());
        let run = || {
            let mut ch = FsChallenger::new(b"flock-keccak3-lig-v0");
            let _ = prove_fast_ligerito_from_witness(&setup.r1cs, &setup.pcs_params,
                z.clone(), a.clone(), b.clone(), zlc.clone(), &KeccakLincheckCircuit, None, &mut ch);
        };
        run();
        let n = if setup.m() >= 26 { 3 } else { 5 };
        let mut best = f64::INFINITY;
        for _ in 0..n { let t = Instant::now(); run(); best = best.min(t.elapsed().as_secs_f64() * 1e3); }
        println!("K3LIGCPU n_keccaks={nk} m={} Ligerito prove = {best:.2} ms", setup.m());
    }
}
