# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BLAKE3's incremental hasher as a fixed-shape device state.

hash-frx's BLAKE3 hashes a whole message of static length; `zorch`'s
`Sha256FieldTranscript` shows the shape a transcript needs instead — a resumable
state that can be threaded through a jitted loop. This is that state for BLAKE3,
and it is the missing piece under a device benchmark-profile challenger.

**Why a streaming state and not a buffer.** A transcript that keeps the raw
absorbed bytes and hashes them on demand cannot be a loop carry: the used length
grows every round, so inside a `while` it is a runtime value, and a one-shot
digest needs a static message shape. Every field here has a shape fixed at
`init`, so absorbing a megabyte leaves the pytree structure exactly as it was —
which is the whole point (`blake3_stream_test.test_state_shape_is_absorb_invariant`).

Runtime lengths still work because BLAKE3 takes the partial block's length as an
operand: a final block is zero-padded to 64 bytes and its true length rides in
`block_len`, so "how many bytes are real" never has to be a shape.

Structure follows the reference implementation's `Hasher` (BLAKE3 spec section
5.1): a chunk state (running chaining value, the block being filled, how many
blocks of this chunk are already compressed) plus a stack of completed subtree
chaining values, merged whenever the finished-chunk count turns even.

`CHUNK_END` is why a full block is kept rather than compressed eagerly: a
chunk's last compression carries a different flag from its others, and which
block is last is only known once more input arrives. The reference makes the
same choice for the same reason.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial

import frx
import frx.numpy as fnp
from frx.tree_util import register_dataclass
from hash_frx.blake3 import blake3
from hash_frx.blake3.compress import CHUNK_END, CHUNK_START, PARENT, compress

U32 = fnp.uint32
BLOCK_LEN = blake3.BLOCK_LEN  # 64
BLOCKS_PER_CHUNK = blake3.CHUNK_LEN // BLOCK_LEN  # 16
# The reference's bound: a 2^64-chunk input needs 54 levels of subtree stack.
MAX_STACK = 54


@register_dataclass
@dataclass(frozen=True)
class Blake3Stream:
    """A resumable BLAKE3 hash state. Every field's shape is absorb-invariant.

    cv_stack   uint32 [MAX_STACK, 8]  completed subtree chaining values
    stack_len  uint32 []              how many of them are live
    chunk_cv   uint32 [8]             the current chunk's running chaining value
    counter    uint32 [2]             the current chunk's index, (low, high)
    block      uint8  [BLOCK_LEN]     the block being filled, zero-padded
    block_len  uint32 []              how much of `block` is real
    compressed uint32 []              blocks of this chunk already compressed
    """

    cv_stack: frx.Array
    stack_len: frx.Array
    chunk_cv: frx.Array
    counter: frx.Array
    block: frx.Array
    block_len: frx.Array
    compressed: frx.Array


_MODE = blake3.hash_mode()
_KEY_WORDS = _MODE.key_words
_MODE_FLAGS = U32(_MODE.flags)


def _key_words():
    return _KEY_WORDS


def _mode_flags():
    return _MODE_FLAGS


def init() -> Blake3Stream:
    return Blake3Stream(
        cv_stack=fnp.zeros((MAX_STACK, 8), U32),
        stack_len=U32(0),
        chunk_cv=_key_words(),
        counter=fnp.zeros((2,), U32),
        block=fnp.zeros((BLOCK_LEN,), fnp.uint8),
        block_len=U32(0),
        compressed=U32(0),
    )


def _words(block):
    """uint8 [64] -> uint32 [16], little-endian, the order `compress` reads."""
    b = block.reshape(BLOCK_LEN // 4, 4).astype(U32)
    return b[:, 0] | (b[:, 1] << U32(8)) | (b[:, 2] << U32(16)) | (b[:, 3] << U32(24))


def _compress1(cv, block_words, counter, block_len, flags):
    """`compress` on a single row, unbatched in and out."""
    return compress(
        cv[None, :],
        block_words[None, :],
        counter[None, :],
        block_len[None],
        flags[None],
    )[0]


def _counter_inc(counter):
    lo = counter[0] + U32(1)
    carry = (lo == U32(0)).astype(U32)
    return fnp.stack([lo, counter[1] + carry])


def _push_chunk_cv(st: Blake3Stream, cv) -> Blake3Stream:
    """Add a finished chunk's chaining value, merging while the completed-chunk
    count is even (BLAKE3 spec 5.1.2).

    The merge count is data-dependent, so it is a `while_loop` — the stack's
    shape does not move, only `stack_len`.
    """
    total = _counter_inc(st.counter)  # chunks completed after this one

    def cond(state):
        _, _, slen, shift = state
        # Merge while the completed-chunk count is even at this level — and
        # never below an empty stack, which `total` alone does not rule out.
        return (slen > U32(0)) & (_bit(total, shift) == U32(0))

    def body(state):
        cv_, stack, slen, shift = state
        slen = slen - U32(1)
        left = frx.lax.dynamic_index_in_dim(stack, slen, axis=0, keepdims=False)
        merged = _compress1(
            _key_words(),
            fnp.concatenate([left, cv_]),
            fnp.zeros((2,), U32),
            U32(BLOCK_LEN),
            _mode_flags() | U32(PARENT),
        )[:8]
        return merged, stack, slen, shift + U32(1)

    # The reference tests bit 0 of the completed-chunk count first, then shifts;
    # starting at 1 would skip the first merge and corrupt every even chunk.
    cv, stack, slen, _ = frx.lax.while_loop(
        cond, body, (cv, st.cv_stack, st.stack_len, U32(0))
    )
    stack = frx.lax.dynamic_update_index_in_dim(stack, cv, slen, axis=0)
    return _replace(
        st,
        cv_stack=stack,
        stack_len=slen + U32(1),
        counter=total,
        chunk_cv=_key_words(),
        compressed=U32(0),
    )


def _bit(counter, i):
    """Bit `i` of the 64-bit `counter` held as (low, high) uint32."""
    lo, hi = counter[0], counter[1]
    word = fnp.where(i < U32(32), lo, hi)
    return (word >> (i & U32(31))) & U32(1)


def _replace(st: Blake3Stream, **kw) -> Blake3Stream:
    return dataclasses.replace(st, **kw)


def _output(icv, blk, ctr, blen, flags) -> blake3.Output:
    """A one-row unrun node. `iv` is the mode's and constant, so it rides here
    rather than through the merge carry."""
    return blake3.Output(
        icv[None, :],
        blk[None, :],
        ctr[None, :],
        blen[None],
        flags[None],
        _MODE.iv,
    )


def _absorb_block(st: Blake3Stream, block_u8) -> Blake3Stream:
    """Compress one full 64-byte block into the current chunk.

    The chunk's 16th block is its last, so it carries `CHUNK_END` and its result
    is the chunk's chaining value rather than a running one.
    """
    last = st.compressed == U32(BLOCKS_PER_CHUNK - 1)
    flags = (
        _mode_flags()
        | fnp.where(st.compressed == U32(0), U32(CHUNK_START), U32(0))
        | fnp.where(last, U32(CHUNK_END), U32(0))
    )
    cv = _compress1(st.chunk_cv, _words(block_u8), st.counter, U32(BLOCK_LEN), flags)[
        :8
    ]
    advanced = _replace(st, chunk_cv=cv, compressed=st.compressed + U32(1))
    return frx.lax.cond(last, lambda s: _push_chunk_cv(s, cv), lambda s: s, advanced)


@frx.jit
def absorb(st: Blake3Stream, msg) -> Blake3Stream:
    """Absorb `msg` (uint8 `[L]`, L static). Any split of a message absorbs to
    the same state as absorbing it whole.

    Jitted rather than inlined so the body is emitted once and shared by every
    call site: as a plain function a transcript that absorbs three times per
    round re-emits ~1,800 instructions three times, and the round loop is
    unrolled on top of that.
    """
    msg = fnp.asarray(msg, fnp.uint8).reshape(-1)
    length = msg.shape[0]
    if length == 0:
        return st

    # Lay the buffered partial block and the new bytes out contiguously. The
    # write offset is a runtime value; the shapes are not.
    work = fnp.zeros((BLOCK_LEN + length,), fnp.uint8)
    work = frx.lax.dynamic_update_slice(work, st.block, (0,))
    work = frx.lax.dynamic_update_slice(work, msg, (st.block_len,))

    total = st.block_len + U32(length)
    # Keep the final block un-compressed: which one is a chunk's last is only
    # known once more input arrives.
    nblocks = (total - U32(1)) // U32(BLOCK_LEN)
    max_blocks = (BLOCK_LEN + length - 1) // BLOCK_LEN

    def step(i, s):
        return frx.lax.cond(
            U32(i) < nblocks,
            lambda s_: _absorb_block(
                s_,
                frx.lax.dynamic_slice(work, (U32(i) * U32(BLOCK_LEN),), (BLOCK_LEN,)),
            ),
            lambda s_: s_,
            s,
        )

    st = frx.lax.fori_loop(0, max_blocks, step, st)

    # What is left over is the tail after the compressed blocks.
    rest = total - nblocks * U32(BLOCK_LEN)
    tail = frx.lax.dynamic_slice(
        fnp.concatenate([work, fnp.zeros((BLOCK_LEN,), fnp.uint8)]),
        (nblocks * U32(BLOCK_LEN),),
        (BLOCK_LEN,),
    )
    keep = fnp.arange(BLOCK_LEN, dtype=U32) < rest
    return _replace(st, block=fnp.where(keep, tail, fnp.uint8(0)), block_len=rest)


@partial(frx.jit, static_argnums=(1,))
def finalize(st: Blake3Stream, out_len: int):
    """Read `out_len` bytes of the root's extendable output: uint8 `[out_len]`.

    Non-mutating — the state is unchanged, matching the reference, so a
    transcript can squeeze without ending its stream. Shared across call sites
    for the same reason as `absorb`.
    """
    root_flags = (
        _mode_flags()
        | fnp.where(st.compressed == U32(0), U32(CHUNK_START), U32(0))
        | U32(CHUNK_END)
    )
    # `Output` is an operand record, not a pytree (its `__eq__`/`__hash__` are
    # element-wise over Arrays and raise), so the merge carries its five fields
    # as plain arrays and rebuilds the record at the end.
    carry = (
        st.chunk_cv,
        _words(st.block),
        st.counter,
        st.block_len,
        root_flags,
        st.stack_len,
    )

    def cond(state):
        return state[-1] > U32(0)

    def body(state):
        icv, blk, ctr, blen, flags, slen = state
        slen = slen - U32(1)
        left = frx.lax.dynamic_index_in_dim(st.cv_stack, slen, axis=0, keepdims=False)
        cv = blake3.chaining_value(_output(icv, blk, ctr, blen, flags))[0]
        parent = blake3.parent_output(left[None, :], cv[None, :], blake3.hash_mode())
        return (
            parent.input_chaining_value[0],
            parent.block[0],
            parent.counter[0],
            parent.block_len[0],
            parent.flags[0],
            slen,
        )

    icv, blk, ctr, blen, flags, _ = frx.lax.while_loop(cond, body, carry)
    return blake3.root_bytes(_output(icv, blk, ctr, blen, flags), out_len)[0]
