// SPDX-FileCopyrightText: Copyright (C) Shielded Technologies
// SPDX-License-Identifier: Apache-2.0

// MSM kernels over BLS12-381 G1 (Pippenger buckets). Appended to ntt.metal
// (Fr, fr_mul). Base-field elements are blst's Montgomery form (R = 2^384),
// 12 little-endian u32 limbs; points are blst's affine (x, y) and Jacobian
// (X, Y, Z) layouts, so results are read back as blst points directly.

struct Fp { uint v[12]; };

constant uint FP_P[12] = {
    0xffffaaabu, 0xb9feffffu, 0xb153ffffu, 0x1eabfffeu, 0xf6b0f624u, 0x6730d2a0u,
    0xf38512bfu, 0x64774b84u, 0x434bacd7u, 0x4b1ba7b6u, 0x397fe69au, 0x1a0111eau,
};
constant uint FP_INV32 = 0xfffcfffdu;
constant uint FP_ONE[12] = {
    0x0002fffdu, 0x76090000u, 0xc40c0002u, 0xebf4000bu, 0x53c758bau, 0x5f489857u,
    0x70525745u, 0x77ce5853u, 0xa256ec6du, 0x5c071a97u, 0xfa80e493u, 0x15f65ec3u,
};

static inline bool fp_geq_p(thread const uint *a) {
    _Pragma("unroll")
    for (int i = 11; i >= 0; --i) {
        if (a[i] > FP_P[i]) return true;
        if (a[i] < FP_P[i]) return false;
    }
    return true;
}

static inline void fp_sub_p(thread uint *a) {
    uint borrow = 0;
    _Pragma("unroll")
    for (int i = 0; i < 12; ++i) {
        ulong d = (ulong)a[i] - (ulong)FP_P[i] - (ulong)borrow;
        a[i] = (uint)d;
        borrow = (uint)(d >> 63);
    }
}

static inline bool fp_is_zero(Fp a) {
    uint x = 0;
    _Pragma("unroll")
    for (int i = 0; i < 12; ++i) x |= a.v[i];
    return x == 0;
}

static inline bool fp_eq(Fp a, Fp b) {
    uint x = 0;
    _Pragma("unroll")
    for (int i = 0; i < 12; ++i) x |= a.v[i] ^ b.v[i];
    return x == 0;
}

static inline Fp fp_zero() {
    Fp r;
    _Pragma("unroll")
    for (int i = 0; i < 12; ++i) r.v[i] = 0;
    return r;
}

static inline Fp fp_one() {
    Fp r;
    _Pragma("unroll")
    for (int i = 0; i < 12; ++i) r.v[i] = FP_ONE[i];
    return r;
}

static inline Fp fp_add(Fp a, Fp b) {
    Fp r;
    uint carry = 0;
    _Pragma("unroll")
    for (int i = 0; i < 12; ++i) {
        ulong s = (ulong)a.v[i] + (ulong)b.v[i] + (ulong)carry;
        r.v[i] = (uint)s;
        carry = (uint)(s >> 32);
    }
    // 2p < 2^384: no carry out.
    if (fp_geq_p(r.v)) fp_sub_p(r.v);
    return r;
}

static inline Fp fp_sub(Fp a, Fp b) {
    Fp r;
    uint borrow = 0;
    _Pragma("unroll")
    for (int i = 0; i < 12; ++i) {
        ulong d = (ulong)a.v[i] - (ulong)b.v[i] - (ulong)borrow;
        r.v[i] = (uint)d;
        borrow = (uint)(d >> 63);
    }
    if (borrow) {
        uint carry = 0;
        _Pragma("unroll")
        for (int i = 0; i < 12; ++i) {
            ulong s = (ulong)r.v[i] + (ulong)FP_P[i] + (ulong)carry;
            r.v[i] = (uint)s;
            carry = (uint)(s >> 32);
        }
    }
    return r;
}

// CIOS Montgomery multiplication: a * b * 2^-384 mod p.
static inline Fp fp_mul(Fp a, Fp b) {
    uint t[14];
    _Pragma("unroll")
    for (int i = 0; i < 14; ++i) t[i] = 0;
    _Pragma("unroll")
    for (int i = 0; i < 12; ++i) {
        ulong c = 0;
        uint bi = b.v[i];
        _Pragma("unroll")
        for (int j = 0; j < 12; ++j) {
            ulong s = (ulong)a.v[j] * (ulong)bi + (ulong)t[j] + c;
            t[j] = (uint)s;
            c = s >> 32;
        }
        ulong s = (ulong)t[12] + c;
        t[12] = (uint)s;
        t[13] = (uint)(s >> 32);
        uint m = t[0] * FP_INV32;
        s = (ulong)m * (ulong)FP_P[0] + (ulong)t[0];
        c = s >> 32;
        _Pragma("unroll")
        for (int j = 1; j < 12; ++j) {
            s = (ulong)m * (ulong)FP_P[j] + (ulong)t[j] + c;
            t[j - 1] = (uint)s;
            c = s >> 32;
        }
        s = (ulong)t[12] + c;
        t[11] = (uint)s;
        t[12] = t[13] + (uint)(s >> 32);
    }
    Fp r;
    _Pragma("unroll")
    for (int i = 0; i < 12; ++i) r.v[i] = t[i];
    if (fp_geq_p(r.v)) fp_sub_p(r.v);
    return r;
}

struct Aff { Fp x; Fp y; };       // blst_p1_affine; (0, 0) is the identity
struct Jac { Fp x; Fp y; Fp z; }; // blst_p1 (Jacobian); z = 0 is the identity

static inline Jac jac_inf() {
    Jac r;
    r.x = fp_one(); r.y = fp_one(); r.z = fp_zero();
    return r;
}

// dbl-2009-l (a = 0).
static inline Jac jac_dbl(Jac p) {
    if (fp_is_zero(p.z)) return p;
    Fp a = fp_mul(p.x, p.x);
    Fp b = fp_mul(p.y, p.y);
    Fp c = fp_mul(b, b);
    Fp t = fp_add(p.x, b);
    Fp d = fp_sub(fp_sub(fp_mul(t, t), a), c);
    d = fp_add(d, d);
    Fp e = fp_add(fp_add(a, a), a);
    Fp f = fp_mul(e, e);
    Jac r;
    r.x = fp_sub(f, fp_add(d, d));
    Fp c8 = fp_add(c, c); c8 = fp_add(c8, c8); c8 = fp_add(c8, c8);
    r.y = fp_sub(fp_mul(e, fp_sub(d, r.x)), c8);
    Fp yz = fp_mul(p.y, p.z);
    r.z = fp_add(yz, yz);
    return r;
}

// p + q, q affine and not the identity (madd-2007-bl), complete.
static inline Jac jac_madd(Jac p, Aff q) {
    if (fp_is_zero(p.z)) {
        Jac r;
        r.x = q.x; r.y = q.y; r.z = fp_one();
        return r;
    }
    Fp z1z1 = fp_mul(p.z, p.z);
    Fp u2 = fp_mul(q.x, z1z1);
    Fp s2 = fp_mul(q.y, fp_mul(p.z, z1z1));
    Fp h = fp_sub(u2, p.x);
    Fp rr = fp_sub(s2, p.y);
    if (fp_is_zero(h)) {
        if (fp_is_zero(rr)) return jac_dbl(p);
        return jac_inf();
    }
    Fp hh = fp_mul(h, h);
    Fp i = fp_add(hh, hh); i = fp_add(i, i);
    Fp j = fp_mul(h, i);
    rr = fp_add(rr, rr);
    Fp v = fp_mul(p.x, i);
    Jac r;
    r.x = fp_sub(fp_sub(fp_mul(rr, rr), j), fp_add(v, v));
    Fp yj = fp_mul(p.y, j);
    r.y = fp_sub(fp_mul(rr, fp_sub(v, r.x)), fp_add(yj, yj));
    Fp zh = fp_add(p.z, h);
    r.z = fp_sub(fp_sub(fp_mul(zh, zh), z1z1), hh);
    return r;
}

// p + q (add-2007-bl), complete.
static inline Jac jac_add(Jac p, Jac q) {
    if (fp_is_zero(p.z)) return q;
    if (fp_is_zero(q.z)) return p;
    Fp z1z1 = fp_mul(p.z, p.z);
    Fp z2z2 = fp_mul(q.z, q.z);
    Fp u1 = fp_mul(p.x, z2z2);
    Fp u2 = fp_mul(q.x, z1z1);
    Fp s1 = fp_mul(p.y, fp_mul(q.z, z2z2));
    Fp s2 = fp_mul(q.y, fp_mul(p.z, z1z1));
    Fp h = fp_sub(u2, u1);
    Fp rr = fp_sub(s2, s1);
    if (fp_is_zero(h)) {
        if (fp_is_zero(rr)) return jac_dbl(p);
        return jac_inf();
    }
    Fp h2 = fp_add(h, h);
    Fp i = fp_mul(h2, h2);
    Fp j = fp_mul(h, i);
    rr = fp_add(rr, rr);
    Fp v = fp_mul(u1, i);
    Jac r;
    r.x = fp_sub(fp_sub(fp_mul(rr, rr), j), fp_add(v, v));
    Fp s1j = fp_mul(s1, j);
    r.y = fp_sub(fp_mul(rr, fp_sub(v, r.x)), fp_add(s1j, s1j));
    Fp zz = fp_add(p.z, q.z);
    r.z = fp_mul(fp_sub(fp_sub(fp_mul(zz, zz), z1z1), z2z2), h);
    return r;
}

struct MsmP {
    uint n;       // points
    uint c;       // window bits
    uint nwin;    // windows
    uint nbuck;   // buckets per window (2^c)
    uint nmsm;    // MSMs over the same bases; MSM b's window w is b * nwin + w
};

// Window digits of the canonical scalars, and the bucket counts. Thread
// b * n + i is point i of MSM b; W = b * nwin + w is its window w, with
// digits[W * n + i]. A zero digit, or an identity base, adds nothing.
kernel void msm_digits(device const Fr *scalars [[buffer(0)]],
                       device const Aff *bases [[buffer(1)]],
                       device uint *digits [[buffer(2)]],
                       device atomic_uint *counts [[buffer(3)]],
                       constant MsmP &p [[buffer(4)]],
                       uint t [[thread_position_in_grid]]) {
    if (t >= p.n * p.nmsm) return;
    uint b = t / p.n, i = t % p.n;
    Fr one_raw;
    for (int l = 0; l < 8; ++l) one_raw.v[l] = 0;
    one_raw.v[0] = 1;
    Fr s = fr_mul(scalars[t], one_raw);   // out of Montgomery form
    bool inf = fp_is_zero(bases[i].x) && fp_is_zero(bases[i].y);
    for (uint w = 0; w < p.nwin; ++w) {
        uint bit = w * p.c;
        uint limb = bit >> 5, off = bit & 31;
        ulong lo = (ulong)s.v[limb];
        ulong hi = limb + 1 < 8 ? (ulong)s.v[limb + 1] : 0;
        uint d = (uint)(((hi << 32) | lo) >> off) & (p.nbuck - 1);
        if (inf) d = 0;
        uint W = b * p.nwin + w;
        digits[W * p.n + i] = d;
        if (d) atomic_fetch_add_explicit(&counts[W * p.nbuck + d], 1u, memory_order_relaxed);
    }
}

// Point indices grouped by bucket: cursor starts at each bucket's offset.
kernel void msm_scatter(device const uint *digits [[buffer(0)]],
                        device atomic_uint *cursor [[buffer(1)]],
                        device uint *sorted [[buffer(2)]],
                        constant MsmP &p [[buffer(3)]],
                        uint t [[thread_position_in_grid]]) {
    if (t >= p.n * p.nmsm) return;
    uint b = t / p.n, i = t % p.n;
    for (uint w = 0; w < p.nwin; ++w) {
        uint W = b * p.nwin + w;
        uint d = digits[W * p.n + i];
        if (d) {
            uint pos = atomic_fetch_add_explicit(&cursor[W * p.nbuck + d], 1u, memory_order_relaxed);
            sorted[pos] = i;
        }
    }
}

// One thread per task: the sum of sorted[start .. end) (all one bucket).
kernel void msm_tasks(device const Aff *bases [[buffer(0)]],
                      device const uint *sorted [[buffer(1)]],
                      device const uint2 *tasks [[buffer(2)]],
                      device Jac *out [[buffer(3)]],
                      constant uint &ntasks [[buffer(4)]],
                      uint t [[thread_position_in_grid]]) {
    if (t >= ntasks) return;
    uint2 r = tasks[t];
    Jac acc = jac_inf();
    for (uint j = r.x; j < r.y; ++j) {
        acc = jac_madd(acc, bases[sorted[j]]);
    }
    out[t] = acc;
}

// One thread per group: out[g] = sum of in[groups[g].x .. groups[g].y).
kernel void msm_reduce(device const Jac *in [[buffer(0)]],
                       device const uint2 *groups [[buffer(1)]],
                       device Jac *out [[buffer(2)]],
                       constant uint &ngroups [[buffer(3)]],
                       uint g [[thread_position_in_grid]]) {
    if (g >= ngroups) return;
    uint2 r = groups[g];
    Jac acc = in[r.x];
    for (uint t = r.x + 1; t < r.y; ++t) acc = jac_add(acc, in[t]);
    out[g] = acc;
}

struct SegP {
    uint nbuck;   // buckets per window
    uint seg;     // buckets per segment
    uint nseg;    // segments per window
    uint nwin;
};

// One thread per (window, segment [lo, hi)): R = sum of the segment's
// buckets, S = sum over d of (d - lo + 1) * B_d. Bucket 0 is skipped.
kernel void msm_segments(device const Jac *buckets [[buffer(0)]],
                         device const uint *bidx [[buffer(4)]],
                         device Jac *seg_s [[buffer(1)]],
                         device Jac *seg_r [[buffer(2)]],
                         constant SegP &p [[buffer(3)]],
                         uint t [[thread_position_in_grid]]) {
    if (t >= p.nwin * p.nseg) return;
    uint w = t / p.nseg, g = t % p.nseg;
    uint lo = max(1u, g * p.seg), hi = (g + 1) * p.seg;
    device const uint *bi = bidx + w * p.nbuck;
    Jac r = jac_inf(), sum = jac_inf();
    for (uint d = hi; d > lo; --d) {
        uint k = bi[d - 1];
        if (k != 0xffffffffu) r = jac_add(r, buckets[k]);
        sum = jac_add(sum, r);
    }
    seg_s[t] = sum;
    seg_r[t] = r;
}
