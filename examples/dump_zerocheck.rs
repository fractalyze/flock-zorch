//! Golden-fixture dumper for flock's zerocheck `prove_packed` — the full
//! ZerocheckProof, plus the claim's z / mlv_challenges / r_rest for the port's
//! localization cross-checks. Witness is a valid a·b⊕c=0 instance (c = a AND b).
//!
//! Usage: `cargo run --release --example dump_zerocheck -- [out_path]`
//!
//! File layout (LE), F128 as lo||hi (16 B), K_SKIP=6 (round-1 vectors are 64):
//!   magic b"FLKZC001" (8) ++ n_configs: u64
//!   ++ per config: m:u64, a_packed[2^m/8], b_packed[2^m/8], c_packed[2^m/8],
//!      round1_ab[64], round1_c[64], n_mlv:u64, rounds[2*n_mlv] (m1,mi flat),
//!      final_a, final_b, final_c,  z, mlv_challenges[n_mlv], r_rest[m-6],
//!      a_eval, b_eval, c_eval

use std::io::Write;

use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::zerocheck::prove_packed;

const DOMAIN: &[u8] = b"flock-zc-oracle";
const CONFIGS: [usize; 2] = [13, 14];

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn push_f128(buf: &mut Vec<u8>, v: &F128) {
    buf.extend_from_slice(&v.lo.to_le_bytes());
    buf.extend_from_slice(&v.hi.to_le_bytes());
}

fn push_f128_slice(buf: &mut Vec<u8>, vs: &[F128]) {
    for v in vs {
        push_f128(buf, v);
    }
}

fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; bits.len() / 8];
    for (i, &bit) in bits.iter().enumerate() {
        if bit {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "artifacts/zerocheck_golden.bin".to_string());

    let mut buf = Vec::new();
    buf.extend_from_slice(b"FLKZC001");
    buf.extend_from_slice(&(CONFIGS.len() as u64).to_le_bytes());

    for &m in &CONFIGS {
        let n = 1usize << m;
        let mut s: u64 = 0x5EED ^ (m as u64).wrapping_mul(0x1000_0001);
        let a: Vec<bool> = (0..n).map(|_| splitmix64(&mut s) & 1 == 1).collect();
        let b: Vec<bool> = (0..n).map(|_| splitmix64(&mut s) & 1 == 1).collect();
        let c: Vec<bool> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
        let (a_p, b_p, c_p) = (pack_bits(&a), pack_bits(&b), pack_bits(&c));

        let mut ch = FsChallenger::new(DOMAIN);
        let (proof, claim) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch);
        let n_mlv = m - 6;

        buf.extend_from_slice(&(m as u64).to_le_bytes());
        buf.extend_from_slice(&a_p);
        buf.extend_from_slice(&b_p);
        buf.extend_from_slice(&c_p);
        push_f128_slice(&mut buf, &proof.round1_ab);
        push_f128_slice(&mut buf, &proof.round1_c);
        buf.extend_from_slice(&(n_mlv as u64).to_le_bytes());
        for (m1, mi) in &proof.multilinear_rounds {
            push_f128(&mut buf, m1);
            push_f128(&mut buf, mi);
        }
        push_f128(&mut buf, &proof.final_a_eval);
        push_f128(&mut buf, &proof.final_b_eval);
        push_f128(&mut buf, &proof.final_c_eval);
        push_f128(&mut buf, &claim.z);
        push_f128_slice(&mut buf, &claim.mlv_challenges);
        push_f128_slice(&mut buf, &claim.r_rest);
        push_f128(&mut buf, &claim.a_eval);
        push_f128(&mut buf, &claim.b_eval);
        push_f128(&mut buf, &claim.c_eval);
    }

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create artifacts dir");
        }
    }
    std::fs::File::create(&out_path)
        .and_then(|mut f| f.write_all(&buf))
        .expect("write output");
    eprintln!("wrote zerocheck proofs for m={CONFIGS:?} to {out_path}");
}
