//! Golden-fixture dumper for flock's ring-switch reduction (`ring_switch::prove`).
//!
//! Random packed_witness + x_outer, real FsChallenger; dumps inputs + the proof's
//! s_hat_v + the output (rs_eq_ind, sumcheck_claim). The jax port
//! (`flock_zorch.ring_switch.prove`) recomputes and byte-compares all three.
//!
//! Usage: `cargo run --release --example dump_ring_switch -- [m] [out]`
//!   L = m-7; packed_witness has 2^L F128; x_outer has L+1 F128.

use std::io::Write;

use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::pcs::ring_switch::prove;

fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn rf(s: &mut u64) -> F128 { F128::new(splitmix64(s), splitmix64(s)) }
fn pf(b: &mut Vec<u8>, v: &F128) { b.extend_from_slice(&v.lo.to_le_bytes()); b.extend_from_slice(&v.hi.to_le_bytes()); }
fn pfv(b: &mut Vec<u8>, v: &[F128]) { b.extend_from_slice(&(v.len() as u64).to_le_bytes()); for e in v { pf(b, e); } }

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let out = a.next().unwrap_or_else(|| "artifacts/ring_switch_golden.bin".to_string());
    let l = m - 7;

    let mut s: u64 = 0x1215_517C_0DE5_EED1u64 ^ (m as u64).wrapping_mul(0x1000_0001);
    let packed_witness: Vec<F128> = (0..(1usize << l)).map(|_| rf(&mut s)).collect();
    let x_outer: Vec<F128> = (0..(l + 1)).map(|_| rf(&mut s)).collect();

    let mut ch = FsChallenger::new(b"flock-ring-switch-test");
    let (proof, output) = prove(&packed_witness, &x_outer, &mut ch);

    let mut b = Vec::new();
    b.extend_from_slice(b"FLKRSW01");
    b.extend_from_slice(&(m as u64).to_le_bytes());
    pfv(&mut b, &packed_witness);
    pfv(&mut b, &x_outer);
    pfv(&mut b, &proof.s_hat_v);
    pfv(&mut b, &output.rs_eq_ind);
    pf(&mut b, &output.sumcheck_claim);

    std::fs::File::create(&out).unwrap().write_all(&b).unwrap();
    println!("dumped ring_switch m={m} L={l} s_hat_v=128 rs_eq_ind={} -> {out}", output.rs_eq_ind.len());
}
