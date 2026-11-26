// SPDX-FileCopyrightText: Copyright (C) Shielded Technologies
// SPDX-License-Identifier: Apache-2.0

// BLS12-381 scalar field arithmetic and NTT kernels (Metal Shading Language).
//
// Elements are the Montgomery form used by blst_fr (R = 2^256), stored as
// 8 little-endian u32 limbs; this is the same memory as blst's 4 u64 limbs on
// a little-endian host, so no conversion is needed on either side. Every
// result is fully reduced (< r), so outputs are bit-identical to the CPU.

#include <metal_stdlib>
using namespace metal;

struct Fr { uint v[8]; };

constant uint P[8] = {
    0x00000001u, 0xffffffffu, 0xfffe5bfeu, 0x53bda402u,
    0x09a1d805u, 0x3339d808u, 0x299d7d48u, 0x73eda753u,
};
// -r^{-1} mod 2^32
constant uint INV32 = 0xffffffffu;

static inline bool geq_p(thread const uint *a) {
    for (int i = 7; i >= 0; --i) {
        if (a[i] > P[i]) return true;
        if (a[i] < P[i]) return false;
    }
    return true;
}

static inline void sub_p(thread uint *a) {
    uint borrow = 0;
    for (int i = 0; i < 8; ++i) {
        ulong d = (ulong)a[i] - (ulong)P[i] - (ulong)borrow;
        a[i] = (uint)d;
        borrow = (uint)(d >> 63);
    }
}

static inline Fr fr_add(Fr a, Fr b) {
    Fr r;
    uint carry = 0;
    for (int i = 0; i < 8; ++i) {
        ulong s = (ulong)a.v[i] + (ulong)b.v[i] + (ulong)carry;
        r.v[i] = (uint)s;
        carry = (uint)(s >> 32);
    }
    // 2r < 2^256, so there is no carry out.
    if (geq_p(r.v)) sub_p(r.v);
    return r;
}

static inline Fr fr_sub(Fr a, Fr b) {
    Fr r;
    uint borrow = 0;
    for (int i = 0; i < 8; ++i) {
        ulong d = (ulong)a.v[i] - (ulong)b.v[i] - (ulong)borrow;
        r.v[i] = (uint)d;
        borrow = (uint)(d >> 63);
    }
    if (borrow) {
        uint carry = 0;
        for (int i = 0; i < 8; ++i) {
            ulong s = (ulong)r.v[i] + (ulong)P[i] + (ulong)carry;
            r.v[i] = (uint)s;
            carry = (uint)(s >> 32);
        }
    }
    return r;
}

// CIOS Montgomery multiplication, 8 x 32-bit limbs: a * b * 2^-256 mod r.
static inline Fr fr_mul(Fr a, Fr b) {
    uint t[10];
    for (int i = 0; i < 10; ++i) t[i] = 0;
    for (int i = 0; i < 8; ++i) {
        ulong c = 0;
        uint bi = b.v[i];
        for (int j = 0; j < 8; ++j) {
            ulong s = (ulong)a.v[j] * (ulong)bi + (ulong)t[j] + c;
            t[j] = (uint)s;
            c = s >> 32;
        }
        ulong s = (ulong)t[8] + c;
        t[8] = (uint)s;
        t[9] = (uint)(s >> 32);
        uint m = t[0] * INV32;
        s = (ulong)m * (ulong)P[0] + (ulong)t[0];
        c = s >> 32;
        for (int j = 1; j < 8; ++j) {
            s = (ulong)m * (ulong)P[j] + (ulong)t[j] + c;
            t[j - 1] = (uint)s;
            c = s >> 32;
        }
        s = (ulong)t[8] + c;
        t[7] = (uint)s;
        t[8] = t[9] + (uint)(s >> 32);
    }
    Fr r;
    for (int i = 0; i < 8; ++i) r.v[i] = t[i];
    if (geq_p(r.v)) sub_p(r.v);
    return r;
}

struct Params {
    uint log_n;   // log2 of the transform size
    uint s;       // first stage of the pass (half = 1 << s)
    uint r;       // stages in the pass (radix 2^r)
    uint st;      // unused here (0)
};

// Radix-2^r DIT pass over stages s .. s+r-1, after bit reversal. Thread t
// owns the 2^r elements base + m*half, m < 2^r, and runs the r stages in
// registers. tw[i] = omega^i for i < n/2. Reads `src`, writes `dst` (may be
// the same buffer).
kernel void pass_r(device const Fr *src [[buffer(0)]],
                   device Fr *dst [[buffer(1)]],
                   device const Fr *tw [[buffer(2)]],
                   constant Params &p [[buffer(3)]],
                   uint t [[thread_position_in_grid]]) {
    uint half_ = 1u << p.s;
    uint j = t & (half_ - 1);
    uint base = ((t >> p.s) << (p.s + p.r)) + j;
    uint cnt = 1u << p.r;
    Fr x[8];
    for (uint m = 0; m < cnt; ++m) x[m] = src[base + m * half_];
    for (uint q = 0; q < p.r; ++q) {
        uint sq = p.s + q;                  // global stage
        uint shift = p.log_n - 1 - sq;      // twiddle stride = n / (2 * 2^sq)
        uint bq = 1u << q;
        for (uint m = 0; m < cnt; ++m) {
            if (m & bq) continue;
            uint pos = j + (m & (bq - 1)) * half_;
            Fr w = p.st ? tw[((1u << sq) - 1) + pos] : tw[pos << shift];
            Fr u = x[m];
            Fr v = fr_mul(x[m | bq], w);
            x[m] = fr_add(u, v);
            x[m | bq] = fr_sub(u, v);
        }
    }
    for (uint m = 0; m < cnt; ++m) dst[base + m * half_] = x[m];
}

// v3: bit reversal fused with the first b stages in threadgroup memory.
// Each threadgroup of 2^(b-1) threads loads the contiguous block of 2^b
// outputs (gathered from bit-reversed input positions), runs stages 0..b-1,
// and writes the block to dst.
kernel void first_stages_tg(device const Fr *src [[buffer(0)]],
                            device Fr *dst [[buffer(1)]],
                            device const Fr *tw [[buffer(2)]],
                            constant Params &p [[buffer(3)]],
                            threadgroup Fr *sh [[threadgroup(0)]],
                            uint lt [[thread_position_in_threadgroup]],
                            uint tg [[threadgroup_position_in_grid]]) {
    uint b = p.r;
    uint blk = 1u << b;
    uint off = tg << b;
    uint shift_rev = 32 - p.log_n;
    for (uint i = lt; i < blk; i += blk >> 1) {
        sh[i] = src[reverse_bits(off + i) >> shift_rev];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < b; ++s) {
        uint half_ = 1u << s;
        uint j = lt & (half_ - 1);
        uint i0 = ((lt >> s) << (s + 1)) + j;
        Fr w = p.st ? tw[((1u << s) - 1) + j] : tw[j << (p.log_n - 1 - s)];
        Fr u = sh[i0];
        Fr v = fr_mul(sh[i0 + half_], w);
        sh[i0] = fr_add(u, v);
        sh[i0 + half_] = fr_sub(u, v);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = lt; i < blk; i += blk >> 1) dst[off + i] = sh[i];
}

