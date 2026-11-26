// SPDX-FileCopyrightText: Copyright (C) Shielded Technologies
// SPDX-License-Identifier: Apache-2.0

// Quotient kernels: coset expansion, the numerator of the quotient, and the
// division by the vanishing polynomial. Appended to ntt.metal (Fr, fr_add,
// fr_sub, fr_mul). Every value is a fully reduced field element, so results
// are bit-identical to the CPU reference whatever the order of operations.

static inline Fr fr_neg(Fr a) {
    Fr z;
    for (int i = 0; i < 8; ++i) z.v[i] = 0;
    return fr_sub(z, a);
}

// ------------------------------------------------------------ elementwise

struct CosetP {
    Fr z1;      // factor for i % 3 == 1
    Fr z2;      // factor for i % 3 == 2
    uint n;     // coefficients in src
};

// dst[i] = src[i] * zeta^(i mod 3) for i < n, 0 above.
kernel void coset_load(device const Fr *src [[buffer(0)]],
                       device Fr *dst [[buffer(1)]],
                       constant CosetP &p [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        for (int l = 0; l < 8; ++l) dst[i].v[l] = 0;
        return;
    }
    Fr x = src[i];
    uint m = i % 3;
    if (m == 1) x = fr_mul(x, p.z1);
    else if (m == 2) x = fr_mul(x, p.z2);
    dst[i] = x;
}

// a[i] *= t[i & (tlen - 1)]
kernel void mul_periodic(device Fr *a [[buffer(0)]],
                         device const Fr *t [[buffer(1)]],
                         constant uint &tlen [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    a[i] = fr_mul(a[i], t[i & (tlen - 1)]);
}

struct FinishP {
    Fr divisor;
    Fr z1;      // factor for i % 3 == 1
    Fr z2;      // factor for i % 3 == 2
};

// a[i] *= divisor * zeta_out^(i mod 3)
kernel void scale_coset(device Fr *a [[buffer(0)]],
                        constant FinishP &p [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    Fr x = fr_mul(a[i], p.divisor);
    uint m = i % 3;
    if (m == 1) x = fr_mul(x, p.z1);
    else if (m == 2) x = fr_mul(x, p.z2);
    a[i] = x;
}

// a[i] *= s
kernel void scale(device Fr *a [[buffer(0)]],
                  constant Fr &s [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
    a[i] = fr_mul(a[i], s);
}

// ------------------------------------------------------------ numerator

struct Col { device const Fr *p; };

struct Common {
    device const Col *fixed;
    device const Col *advice;
    device const Col *instance;
    device const Fr *l0;
    device const Fr *l_last;
    device const Fr *l_active;
    Fr y;
    Fr one;
    uint n;          // rows of the extended domain
    uint log_scale;  // ext_k - k
    uint t;          // threads of a row kernel (row stride, slot stride)
    uint nslots;
};

struct Op { uint kind; uint dst; uint a; uint b; uint c; };
struct Rd { uint kind; uint column; int rotation; };

struct Prog {
    device const Op *ops;
    device const Rd *reads;
    device const Fr *uniforms;
    device const uint *outputs;
    uint nops;
    uint noutputs;
};

static inline uint rot(constant Common &c, uint idx, int r) {
    return (idx + (uint)(r * (int)(1u << c.log_scale))) & (c.n - 1);
}

static inline Fr col_at(constant Common &c, uint kind, uint column, uint row) {
    device const Fr *p = kind == 0 ? c.fixed[column].p
                       : kind == 1 ? c.advice[column].p
                                   : c.instance[column].p;
    return p[row];
}

static inline Fr operand(constant Common &c, device const Prog &g,
                         device Fr *slots, uint tid, uint idx, uint o) {
    uint i = o & ((1u << 30) - 1);
    uint tag = o >> 30;
    if (tag == 0) return g.uniforms[i];
    if (tag == 1) {
        Rd r = g.reads[i];
        return col_at(c, r.kind, r.column, rot(c, idx, r.rotation));
    }
    return slots[i * c.t + tid];
}

// Runs program g for row idx; writes its first `noutputs` outputs (<= 4).
static inline void eval(constant Common &c, device const Prog &g,
                        device Fr *slots, uint tid, uint idx, thread Fr *out) {
    for (uint k = 0; k < g.nops; ++k) {
        Op op = g.ops[k];
        Fr a = operand(c, g, slots, tid, idx, op.a);
        Fr v;
        switch (op.kind) {
            case 0: v = fr_add(a, operand(c, g, slots, tid, idx, op.b)); break;
            case 1: v = fr_sub(a, operand(c, g, slots, tid, idx, op.b)); break;
            case 2: v = fr_mul(a, operand(c, g, slots, tid, idx, op.b)); break;
            case 3: v = fr_mul(a, a); break;
            case 4: v = fr_add(a, a); break;
            case 5: v = fr_neg(a); break;
            default:
                v = fr_add(fr_mul(a, operand(c, g, slots, tid, idx, op.b)),
                           operand(c, g, slots, tid, idx, op.c));
                break;
        }
        slots[op.dst * c.t + tid] = v;
    }
    for (uint k = 0; k < g.noutputs; ++k) {
        out[k] = operand(c, g, slots, tid, idx, g.outputs[k]);
    }
}

// nu[idx] = gates(idx)
kernel void numer_gates(constant Common &c [[buffer(0)]],
                        device const Prog &g [[buffer(1)]],
                        device Fr *nu [[buffer(2)]],
                        device Fr *slots [[buffer(3)]],
                        uint tid [[thread_position_in_grid]]) {
    Fr out[4];
    for (uint idx = tid; idx < c.n; idx += c.t) {
        eval(c, g, slots, tid, idx, out);
        nu[idx] = out[0];
    }
}

struct Perm {
    device const Col *products;  // extended cosets, one per set
    device const Col *sigma;     // extended cosets, one per column
    device const uint *col_kind;
    device const uint *col_idx;
    device const Fr *omega_pows; // extended_omega^(2^j), j < ext_k
    Fr beta;
    Fr gamma;
    Fr delta_start;
    Fr delta;
    uint nsets;
    uint ncols;
    uint chunk_len;
    int last_rot;
    uint ext_k;
    uint pad;
};

kernel void numer_perm(constant Common &c [[buffer(0)]],
                       device const Perm &p [[buffer(1)]],
                       device Fr *nu [[buffer(2)]],
                       uint tid [[thread_position_in_grid]]) {
    for (uint idx = tid; idx < c.n; idx += c.t) {
        Fr v = nu[idx];
        Fr l0 = c.l0[idx], l_last = c.l_last[idx], l_active = c.l_active[idx];
        uint r_next = rot(c, idx, 1), r_last = rot(c, idx, p.last_rot);
        device const Fr *first = p.products[0].p;
        device const Fr *last = p.products[p.nsets - 1].p;
        v = fr_add(fr_mul(v, c.y), fr_mul(fr_sub(c.one, first[idx]), l0));
        Fr z = last[idx];
        v = fr_add(fr_mul(v, c.y), fr_mul(fr_sub(fr_mul(z, z), z), l_last));
        for (uint s = 1; s < p.nsets; ++s) {
            Fr d = fr_sub(p.products[s].p[idx], p.products[s - 1].p[r_last]);
            v = fr_add(fr_mul(v, c.y), fr_mul(d, l0));
        }
        // delta_start * extended_omega^idx
        Fr w = c.one;
        for (uint j = 0; j < p.ext_k; ++j) {
            if ((idx >> j) & 1u) w = fr_mul(w, p.omega_pows[j]);
        }
        Fr cur = fr_mul(p.delta_start, w);
        for (uint s = 0; s < p.nsets; ++s) {
            uint lo = s * p.chunk_len;
            uint hi = min(lo + p.chunk_len, p.ncols);
            device const Fr *zs = p.products[s].p;
            Fr left = zs[r_next];
            for (uint k = lo; k < hi; ++k) {
                Fr x = col_at(c, p.col_kind[k], p.col_idx[k], idx);
                left = fr_mul(left, fr_add(fr_add(x, fr_mul(p.beta, p.sigma[k].p[idx])), p.gamma));
            }
            Fr right = zs[idx];
            for (uint k = lo; k < hi; ++k) {
                Fr x = col_at(c, p.col_kind[k], p.col_idx[k], idx);
                right = fr_mul(right, fr_add(fr_add(x, cur), p.gamma));
                cur = fr_mul(cur, p.delta);
            }
            v = fr_add(fr_mul(v, c.y), fr_mul(fr_sub(left, right), l_active));
        }
        nu[idx] = v;
    }
}

struct Lookup {
    device const Fr *agg;
    device const Fr *mult;
    device const Col *helpers;
    device const Prog *progs;
    uint nchunks;
    uint pad;
};

kernel void numer_lookup(constant Common &c [[buffer(0)]],
                         device const Lookup &l [[buffer(1)]],
                         device Fr *nu [[buffer(2)]],
                         device Fr *slots [[buffer(3)]],
                         uint tid [[thread_position_in_grid]]) {
    Fr out[4];
    for (uint idx = tid; idx < c.n; idx += c.t) {
        Fr v = nu[idx];
        Fr l0 = c.l0[idx], l_last = c.l_last[idx], l_active = c.l_active[idx];
        Fr agg = l.agg[idx];
        v = fr_add(fr_mul(v, c.y), fr_mul(agg, fr_add(l0, l_last)));
        Fr sum, table, selector;
        for (int k = 0; k < 8; ++k) { sum.v[k] = 0; table.v[k] = 0; selector.v[k] = 0; }
        for (uint f = 0; f < l.nchunks; ++f) {
            eval(c, l.progs[f], slots, tid, idx, out);
            if (f == 0) { table = out[2]; selector = out[3]; }
            Fr h = l.helpers[f].p[idx];
            v = fr_sub(fr_add(fr_mul(v, c.y), fr_mul(h, out[1])), out[0]);
            sum = fr_add(sum, h);
        }
        Fr d = fr_sub(fr_sub(l.agg[rot(c, idx, 1)], agg), fr_mul(selector, sum));
        Fr term = fr_mul(l_active, fr_add(fr_mul(d, table), l.mult[idx]));
        nu[idx] = fr_add(fr_mul(v, c.y), term);
    }
}

struct Trash {
    device const Fr *trash;
    Prog prog;
    uint selector;
    uint pad;
};

kernel void numer_trash(constant Common &c [[buffer(0)]],
                        device const Trash &t [[buffer(1)]],
                        device Fr *nu [[buffer(2)]],
                        device Fr *slots [[buffer(3)]],
                        uint tid [[thread_position_in_grid]]) {
    Fr out[4];
    for (uint idx = tid; idx < c.n; idx += c.t) {
        eval(c, t.prog, slots, tid, idx, out);
        Fr q = c.fixed[t.selector].p[idx];
        Fr term = fr_sub(out[0], fr_mul(fr_sub(c.one, q), t.trash[idx]));
        nu[idx] = fr_add(fr_mul(nu[idx], c.y), term);
    }
}
