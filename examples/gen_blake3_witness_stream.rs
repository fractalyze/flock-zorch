//! Streaming witness producer for the pipelined host∥GPU throughput bench
//! (flock-zorch#163): loop the fused witness path
//! (`generate_witness_with_ab_packed_and_lincheck`) over fresh random blake3
//! batches, writing one compact blob per proof into a bounded file queue that
//! `testing/pipelined_prove_bench.py` consumes.
//!
//! The blob carries only the witness-dependent fields (`z`/`a`/`b` packed +
//! `z_lincheck`); the static config/statement/matrices come from a template
//! golden on the consumer side. Blobs are written to `<queue>/wit_NNNNNN.tmp`
//! then renamed to `.bin` so the consumer never sees a partial file. The queue
//! is bounded: the producer sleeps while `depth` finished blobs are pending, so
//! a full queue means the GPU side binds, an empty one means the host does.
//!
//! Usage: `cargo run --release --example gen_blake3_witness_stream -- \
//!     [n_comp] [queue_dir] [total_blobs] [depth]`
//! Put `queue_dir` on tmpfs (e.g. /dev/shm/...) — blobs are ~100 MB at m=28.

use std::io::Write;
use std::time::Instant;

use flock_prover::r1cs_hashes::blake3;

fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn pu(b: &mut Vec<u8>, x: usize) { b.extend_from_slice(&(x as u64).to_le_bytes()); }
fn pfv(b: &mut Vec<u8>, v: &[flock_core::field::F128]) {
    pu(b, v.len());
    for e in v { b.extend_from_slice(&e.lo.to_le_bytes()); b.extend_from_slice(&e.hi.to_le_bytes()); }
}

fn pending(queue: &std::path::Path) -> usize {
    std::fs::read_dir(queue).map(|it| {
        it.filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".bin"))
            .count()
    }).unwrap_or(0)
}

fn main() {
    let mut a = std::env::args().skip(1);
    let n_comp: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(16384);
    let queue = std::path::PathBuf::from(
        a.next().unwrap_or_else(|| "/dev/shm/flock_witq".to_string()));
    let total: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(40);
    let depth: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    std::fs::create_dir_all(&queue).unwrap();

    let setup = blake3::Blake3Setup::new(n_comp);
    let m = setup.r1cs.m;
    let n_blocks_log = setup.n_blocks_log();
    println!("producer: n_comp={n_comp} m={m} total={total} depth={depth} queue={}",
             queue.display());

    let mut seed: u64 = 0xB1A3_0627u64 ^ (n_comp as u64);
    let t_all = Instant::now();
    let mut gen_ms_total = 0.0f64;
    for i in 0..total {
        // Bounded queue: wait while `depth` blobs are already pending.
        while pending(&queue) >= depth {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        let t0 = Instant::now();
        let blocks: Vec<blake3::Compression> = (0..n_comp).map(|_| {
            let mut cv = [0u32; 8]; for x in &mut cv { *x = splitmix64(&mut seed) as u32; }
            let mut msg = [0u32; 16]; for x in &mut msg { *x = splitmix64(&mut seed) as u32; }
            let counter = splitmix64(&mut seed);
            let block_len = splitmix64(&mut seed) as u32;
            let flags = splitmix64(&mut seed) as u32;
            (cv, msg, counter, block_len, flags)
        }).collect();
        let (z, av, bv, zlc) =
            blake3::generate_witness_with_ab_packed_and_lincheck(&blocks, n_blocks_log);
        let gen_ms = t0.elapsed().as_secs_f64() * 1e3;
        gen_ms_total += gen_ms;

        let mut b = Vec::new();
        b.extend_from_slice(b"FLKWS_01");
        pu(&mut b, m);
        pfv(&mut b, &z); pfv(&mut b, &av); pfv(&mut b, &bv);
        pu(&mut b, zlc.len()); b.extend_from_slice(&zlc);

        let tmp = queue.join(format!("wit_{i:06}.tmp"));
        let fin = queue.join(format!("wit_{i:06}.bin"));
        std::fs::File::create(&tmp).unwrap().write_all(&b).unwrap();
        std::fs::rename(&tmp, &fin).unwrap();
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        println!("blob {i:03}: gen {gen_ms:.1} ms, gen+write {wall_ms:.1} ms, \
                  pending {}", pending(&queue));
    }
    let total_s = t_all.elapsed().as_secs_f64();
    let hashes = (n_comp * total) as f64;
    println!("producer done: {total} blobs in {total_s:.1} s — host-side rate \
              {:.1}K hash/s (gen only: {:.1}K)",
             hashes / total_s / 1e3, hashes / (gen_ms_total / 1e3) / 1e3);
}
