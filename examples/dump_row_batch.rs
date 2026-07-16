//! Golden-fixture dumper for the BaseFold row-batch fold (collapse num_ntts lanes).
//!
//! Replicates flock `pcs::basefold::row_batch_fold_all`'s nested fold
//! `buf[j] = u + r·(u+v)` (a private fn) using flock's real F128 arithmetic on
//! deterministic inputs. The frx port (`pcs_open.row_batch_fold_all`) recomputes
//! and byte-compares. (The full byte-identity anchor is the basefold gate; this
//! isolates the primitive.)
//!
//! Usage: `cargo run --release --example dump_row_batch -- [n_pos] [log_batch_size] [out]`
//! Layout (LE): magic b"FLKRBF01" (8) ++ n_pos,lbs (2×u64) ++
//!   challenges[lbs] ++ codeword[n_pos*2^lbs] ++ folded[n_pos]  (F128=lo,hi).

use std::io::Write;

use flock_core::field::F128;

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
    let n_pos: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let lbs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    let out_path = args.next().unwrap_or_else(|| "artifacts/row_batch_golden.bin".to_string());
    let num_ntts = 1usize << lbs;

    let mut s: u64 = 0x4B_F01D_5EED ^ ((n_pos * 10 + lbs) as u64).wrapping_mul(0x1000_0001);
    let challenges: Vec<F128> = (0..lbs).map(|_| rf(&mut s)).collect();
    let codeword: Vec<F128> = (0..n_pos * num_ntts).map(|_| rf(&mut s)).collect();

    let folded: Vec<F128> = (0..n_pos)
        .map(|pos| {
            let mut buf = codeword[pos * num_ntts..(pos + 1) * num_ntts].to_vec();
            let mut len = num_ntts;
            for &r in &challenges {
                let half = len / 2;
                for j in 0..half {
                    let u = buf[2 * j];
                    let v = buf[2 * j + 1];
                    buf[j] = u + r * (u + v);
                }
                len = half;
            }
            buf[0]
        })
        .collect();

    let mut buf = Vec::new();
    buf.extend_from_slice(b"FLKRBF01");
    buf.extend_from_slice(&(n_pos as u64).to_le_bytes());
    buf.extend_from_slice(&(lbs as u64).to_le_bytes());
    for v in &challenges {
        pf(&mut buf, v);
    }
    for v in &codeword {
        pf(&mut buf, v);
    }
    for v in &folded {
        pf(&mut buf, v);
    }
    let mut f = std::fs::File::create(&out_path).expect("create");
    f.write_all(&buf).expect("write");
    println!("dumped row_batch n_pos={n_pos} lbs={lbs} -> {out_path}");
}
