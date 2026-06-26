"""Scalar host-side GF(2^128) arithmetic in flock's GHASH basis (Python ints).

For small SEQUENTIAL precomputes (e.g. NTT twiddles) that belong on the host per
the project's host/device split. An F128 is a Python int < 2^128 with bit i =
coefficient of x^i -- the same bit layout as field.py's uint64 lanes, so
`val.to_bytes(16, "little")` matches flock's serialization.

NOT for bulk work: plain big-int loops, correct but slow.
"""

_MASK128 = (1 << 128) - 1


def add(a: int, b: int) -> int:
    """GF(2^128) addition = XOR."""
    return a ^ b


def mul(a: int, b: int) -> int:
    """GF(2^128) multiply in the GHASH basis (x^128 + x^7 + x^2 + x + 1)."""
    p = 0
    while b:
        if b & 1:
            p ^= a
        a <<= 1
        b >>= 1
    # reduce: x^128 == x^7 + x^2 + x + 1 == 0x87
    for i in range(p.bit_length() - 1, 127, -1):
        if (p >> i) & 1:
            p ^= (1 << i) ^ (0x87 << (i - 128))
    return p & _MASK128


def sqr(a: int) -> int:
    return mul(a, a)


def inv(a: int) -> int:
    """Multiplicative inverse via Fermat: a^(2^128 - 2)."""
    if a == 0:
        raise ZeroDivisionError("F128 inverse of zero")
    e = (1 << 128) - 2
    r = 1
    base = a
    while e:
        if e & 1:
            r = mul(r, base)
        base = sqr(base)
        e >>= 1
    return r
