//! Host SHA-NI Merkle FFI — the "SHA-256 Merkle off-GPU" optimization.
//!
//! GPU SHA-256 (the 64-round fori) is the prover's Merkle wall (~30 ms @m=26 for
//! commit, plus BaseFold's T2/epoch trees) and loses to CPU SHA-NI; the lax.scan
//! fusion is byte-identical but neutral for SHA (measured 31 vs 33 ms — see
//! flock-zorch-status.md). The real fix is to hash on the host with flock's own
//! rayon + SHA-NI `merkle::merkle_tree` (~0.5 ms), exactly as whir-zorch routes
//! its Merkle through an FFI CUDA kernel for its hash.
//!
//! Python (`flock_zorch.merkle`) transfers the codeword to host (small) and calls
//! this; the result is BYTE-IDENTICAL to the GPU `merkle_tree` (same flock
//! `merkle::merkle_tree`, so the existing merkle/commit/basefold gates still pin
//! it). Use it as the host path under a `use_host_sha` flag (keep the GPU path).
//!
//! BUILD (cdylib): add to flock-zorch Cargo.toml
//!     [lib]
//!     crate-type = ["rlib", "cdylib"]
//! and put `pub mod merkle_ffi;` in src/lib.rs (or compile this file directly):
//!     rustc --edition 2021 -O --crate-type cdylib \
//!       --extern flock_core=target/release/libflock_core.rlib merkle_ffi.rs \
//!       -o optim/merkle_ffi/libflock_merkle.so
//! Simpler: move this into `src/` as a module and `cargo build --release`
//! (produces target/release/libflock_zorch.so).
//!
//! WIRE (next step, flock_zorch/merkle.py):
//!     import ctypes, numpy as np
//!     _lib = ctypes.CDLL(".../libflock_zorch.so")
//!     _lib.flock_merkle_tree.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
//!                                        ctypes.c_size_t, ctypes.c_void_p]
//!     def merkle_tree_host(leaves):           # leaves uint8 [n, leaf_size]
//!         data = np.ascontiguousarray(leaves).ravel()
//!         n_leaves, leaf_size = leaves.shape
//!         out = np.empty((2*n_leaves - 1, 32), np.uint8)
//!         _lib.flock_merkle_tree(data.ctypes.data, data.size, n_leaves, out.ctypes.data)
//!         return out
//! Then add `use_host_sha` to merkle.merkle_tree/merkle_root and re-gate
//! (merkle_oracle_test / merkle_multi_oracle_test / commit_oracle_test) — must
//! stay byte-identical. Measure e2e_gpu_bench: expect merkle 30ms -> ~1ms, and
//! the BaseFold T2/epoch trees similarly (route those through the host path too).

use flock_core::merkle::merkle_tree;

/// Build flock's flat Merkle tree on the host (rayon + SHA-NI). `data` is
/// `n_leaves * leaf_size` bytes (row-major leaves); writes `(2*n_leaves-1)*32`
/// bytes of flat tree (leaves-first, root last) to `out`. Byte-identical to
/// `flock_core::merkle::merkle_tree` (which the jax GPU path also mirrors).
///
/// # Safety
/// `data` must point to `data_len` readable bytes; `out` to
/// `(2*n_leaves-1)*32` writable bytes. `data_len == n_leaves * leaf_size`.
#[no_mangle]
pub unsafe extern "C" fn flock_merkle_tree(
    data: *const u8,
    data_len: usize,
    n_leaves: usize,
    out: *mut u8,
) {
    let bytes = std::slice::from_raw_parts(data, data_len);
    let tree = merkle_tree(bytes, n_leaves); // Vec<[u8; 32]>, len 2*n_leaves - 1
    let out_slice = std::slice::from_raw_parts_mut(out, tree.len() * 32);
    for (i, h) in tree.iter().enumerate() {
        out_slice[i * 32..(i + 1) * 32].copy_from_slice(h);
    }
}

/// Convenience: just the 32-byte root (BaseFold/commit only need the root for the
/// transcript; the full tree is for query multi-proofs).
///
/// # Safety
/// As `flock_merkle_tree`, but `out` need only be 32 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn flock_merkle_root(
    data: *const u8,
    data_len: usize,
    n_leaves: usize,
    out: *mut u8,
) {
    let bytes = std::slice::from_raw_parts(data, data_len);
    let tree = merkle_tree(bytes, n_leaves);
    let root = tree.last().expect("non-empty tree");
    std::slice::from_raw_parts_mut(out, 32).copy_from_slice(root);
}
