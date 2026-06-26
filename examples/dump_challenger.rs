//! Golden-fixture dumper for flock's Fiat-Shamir challenger (the SHA-256
//! `FsChallenger`). Runs ONE deterministic scripted observe/sample/grind
//! sequence exercising every op + the duplex re-absorb + PoW, and dumps the
//! sampled F128s and the grind nonce. The jax port
//! (`flock_zorch.challenger.Challenger` over zorch's `Sha256Transcript`) replays
//! the identical script and byte-compares.
//!
//! Usage: `cargo run --release --example dump_challenger -- [out_path]`
//!
//! File layout (little-endian):
//!   magic b"FLKCHL01" (8) ++ n_samples: u64 ++ samples[n_samples * 16] (lo||hi)
//!   ++ grind_nonce: u64

use std::io::Write;

use flock_core::challenger::{Challenger, FsChallenger};
use flock_core::field::F128;

const DOMAIN: &[u8] = b"flock-zorch-oracle";
const LABEL: &[u8] = b"flock-zerocheck-v0";
const GRIND_BITS: u32 = 8;

fn push_f128(buf: &mut Vec<u8>, v: &F128) {
    buf.extend_from_slice(&v.lo.to_le_bytes());
    buf.extend_from_slice(&v.hi.to_le_bytes());
}

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "artifacts/challenger_golden.bin".to_string());

    let mut ch = FsChallenger::new(DOMAIN);
    let mut samples: Vec<F128> = Vec::new();

    // --- the scripted sequence (mirror exactly in challenger_oracle_test.py) ---
    ch.observe_label(LABEL);
    let root: Vec<u8> = (0u8..32).collect();
    ch.observe_bytes(&root);
    ch.observe_f128(F128::new(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210));
    ch.observe_f128_slice(&[
        F128::new(1, 0),
        F128::new(2, 0),
        F128::new(0xdead_beef, 0xcafe_babe),
    ]);

    let s0 = ch.sample_f128();
    samples.push(s0);
    ch.observe_f128(s0);

    let sv = ch.sample_f128_vec(5);
    samples.extend_from_slice(&sv);
    ch.observe_f128_slice(&sv);

    let nonce = ch.grind_pow(GRIND_BITS);

    samples.push(ch.sample_f128());
    samples.push(ch.sample_f128());
    // --- end script ---

    let mut buf = Vec::new();
    buf.extend_from_slice(b"FLKCHL01");
    buf.extend_from_slice(&(samples.len() as u64).to_le_bytes());
    for s in &samples {
        push_f128(&mut buf, s);
    }
    buf.extend_from_slice(&nonce.to_le_bytes());

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create artifacts dir");
        }
    }
    std::fs::File::create(&out_path)
        .and_then(|mut f| f.write_all(&buf))
        .expect("write output");
    eprintln!("wrote {} challenger samples + nonce={nonce} to {out_path}", samples.len());
}
