//! Golden-fixture dumper for the BaseFold/Ligerito FRI codeword fold — the
//! dominant compute of the PCS opening (port target toward e2e).
//!
//! Replicates flock-core `pcs::basefold::fri_fold_codeword` (a private fn) using
//! its public twiddle source `AdditiveNttF128::twiddle` and the exact `fold_pair`
//! formula (v=v+u; u=u+v·t; out=u+r·(u+v)). The twiddles — the non-trivial part
//! — are flock's, so the fixture is byte-anchored to flock. The frx port
//! (`flock_zorch.pcs_open.fri_fold`) recomputes and byte-compares.
//!
//! Usage: `cargo run --release --example dump_fri_fold -- [k_code] [layer] [out]`
//!   folds a length-2^(layer+1) codeword to 2^layer using twiddle(layer, ·).
//!
//! File layout (LE): magic b"FLKFRI01" (8) ++ k_code,layer (2×u64) ++
//!   challenge (F128) ++ codeword[2^(layer+1)] ++ folded[2^layer], F128=lo,hi.

use std::io::Write;

use flock_core::field::F128;
use flock_core::ntt::AdditiveNttF128;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rf(s: &mut u64) -> F128 {
    F128::new(splitmix64(s), splitmix64(s))
}

fn pf(buf: &mut Vec<u8>, v: &F128) {
    buf.extend_from_slice(&v.lo.to_le_bytes());
    buf.extend_from_slice(&v.hi.to_le_bytes());
}

fn main() {
    let mut args = std::env::args().skip(1);
    let k_code: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let layer: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(19);
    let out_path = args.next().unwrap_or_else(|| "artifacts/fri_fold_golden.bin".to_string());
    assert!(layer < k_code);

    let ntt = AdditiveNttF128::standard(k_code);
    let new_len = 1usize << layer;
    let mut s: u64 = 0xF21_F01D ^ ((k_code * 100 + layer) as u64).wrapping_mul(0x1000_0001);
    let r = rf(&mut s);
    let codeword: Vec<F128> = (0..2 * new_len).map(|_| rf(&mut s)).collect();

    let folded: Vec<F128> = (0..new_len)
        .map(|i| {
            let u_in = codeword[2 * i];
            let v_in = codeword[2 * i + 1];
            let tw = ntt.twiddle(layer, i);
            let v = v_in + u_in;
            let u = u_in + v * tw;
            u + r * (u + v)
        })
        .collect();

    let mut buf = Vec::new();
    buf.extend_from_slice(b"FLKFRI01");
    buf.extend_from_slice(&(k_code as u64).to_le_bytes());
    buf.extend_from_slice(&(layer as u64).to_le_bytes());
    pf(&mut buf, &r);
    for v in &codeword {
        pf(&mut buf, v);
    }
    for v in &folded {
        pf(&mut buf, v);
    }

    let mut f = std::fs::File::create(&out_path).expect("create");
    f.write_all(&buf).expect("write");
    println!("dumped FRI fold k_code={k_code} layer={layer} ({} -> {new_len}) -> {out_path}", 2 * new_len);
}
