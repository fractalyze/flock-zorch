//! Golden-fixture dumper for the additive NTT over F128 (port target #2).
//!
//! Builds `AdditiveNttF128::standard(log_d)`, runs `forward_transform_scalar` on
//! a deterministic SplitMix64 input, and dumps (input, twiddles, output). The frx
//! port (`flock_zorch.ntt.forward_transform_scalar`) reads input + twiddles,
//! recomputes the transform, and byte-compares against output.
//!
//! Usage: `cargo run --release --example dump_ntt -- [log_d] [out_path]`
//!   log_d     transform domain log2 (default 12); standard NTT needs log_d <= 64
//!   out_path  output file (default `artifacts/ntt_golden.bin`)
//!
//! File layout (little-endian): magic `b"FLKNTT01"` (8) ++ `log_d: u64` (8) ++
//!   input[N*16] ++ twiddles[(2^log_d - 1)*16] ++ output[N*16], each `lo || hi`.
//!   Twiddles are layer-major: layer l occupies [2^l - 1, 2^(l+1) - 1).

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

fn push_f128(buf: &mut Vec<u8>, v: &[F128]) {
    for e in v {
        buf.extend_from_slice(&e.lo.to_le_bytes());
        buf.extend_from_slice(&e.hi.to_le_bytes());
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let log_d: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(12);
    let out_path = args
        .next()
        .unwrap_or_else(|| "artifacts/ntt_golden.bin".to_string());
    let n = 1usize << log_d;

    let ntt = AdditiveNttF128::standard(log_d);

    let mut s: u64 = 0x00C0_FFEE ^ (log_d as u64).wrapping_mul(0x1000_0001);
    let input: Vec<F128> = (0..n)
        .map(|_| F128::new(splitmix64(&mut s), splitmix64(&mut s)))
        .collect();

    let mut twiddles: Vec<F128> = Vec::with_capacity(n - 1);
    for layer in 0..log_d {
        for block in 0..(1usize << layer) {
            twiddles.push(ntt.twiddle(layer, block));
        }
    }

    let mut data = input.clone();
    ntt.forward_transform_scalar(&mut data);

    let mut buf = Vec::with_capacity(16 + (2 * n + (n - 1)) * 16);
    buf.extend_from_slice(b"FLKNTT01");
    buf.extend_from_slice(&(log_d as u64).to_le_bytes());
    push_f128(&mut buf, &input);
    push_f128(&mut buf, &twiddles);
    push_f128(&mut buf, &data);

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create artifacts dir");
        }
    }
    std::fs::File::create(&out_path)
        .and_then(|mut f| f.write_all(&buf))
        .expect("write output");
    eprintln!("wrote log_d={log_d} (N={n}, {} twiddles) to {out_path}", n - 1);
}
