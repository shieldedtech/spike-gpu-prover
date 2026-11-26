// SPDX-FileCopyrightText: Copyright (C) Shielded Technologies
// SPDX-License-Identifier: Apache-2.0

//! Apple Metal backend for the `midnight-proofs` accelerator hooks
//! (experimental).
//!
//! [`MetalAccelerator`] serves the base-domain FFTs and the whole quotient
//! (`h(X)`) of a proof on the Apple GPU: the extended-coset transforms of
//! the proof polynomials, the numerator (gates, permutation, LogUp, trash),
//! the division by `X^n - 1` and the inverse transform run in one command
//! buffer. MSMs are declined (they stay on the CPU). Outputs are exact field
//! elements, bit-identical to
//! [`midnight_proofs::accel::reference::CpuReference`].
//!
//! Apple GPUs share memory with the CPU: host buffers that are page-aligned
//! are wrapped without a copy.

use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use ff::Field;
use group::Group;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLResourceUsage, MTLSize,
    foreign_types::ForeignType,
};
use midnight_curves::{Fq, G1Affine, G1Projective, msm::msm_pippenger};
use midnight_proofs::accel::{Accelerator, DeviceProgram, QuotientJob};
use objc::{msg_send, sel, sel_impl};
use rayon::prelude::*;

#[cfg(feature = "chain")]
#[doc(hidden)]
pub mod chain;

const SHADER: &[&str] = &[
    include_str!("ntt.metal"),
    include_str!("quotient.metal"),
    include_str!("msm.metal"),
];
const PAGE: usize = 16384;
/// Below this transform size the CPU is faster (fixed cost per call).
pub const MIN_LOG_N: u32 = 12;
/// Stages fused in threadgroup memory at the start of an NTT.
const TG_STAGES: u32 = 9;
/// Stages per later NTT pass (radix 8).
const PASS_STAGES: u32 = 3;
/// Most points one GPU thread sums (longer buckets are split).
const MSM_TASK: u32 = 32;
/// Buckets per thread of the window reduction.
const MSM_SEG: usize = 4;
/// Most partial sums one thread of a reduction level adds.
const MSM_GROUP: u32 = 32;
/// Bound on the per-row scratch of the numerator kernels.
const SLOT_BYTES: usize = 256 << 20;
/// Below this many points the CPU MSM is faster, so calls are declined.
pub const MSM_MIN_LOG_N: u32 = 16;
/// Below this many points per MSM a batch is declined (each MSM is then
/// offered on its own).
pub const MSM_BATCH_MIN_LOG_N: u32 = 13;
/// [`MSM_GPU_FRAC`] for a batch.
pub const MSM_BATCH_GPU_FRAC: f64 = 0.7;
/// The smallest GPU share of an accepted call (the CPU takes the rest).
const MSM_GPU_MIN_LOG_N: u32 = 15;

/// `ACCEL_METAL_<name>` parsed, or `default` (tuning knobs).
fn knob<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(format!("ACCEL_METAL_{name}"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Share of an MSM's points the GPU takes; the CPU takes the rest. A
/// size-dependent split won on isolated MSMs but not inside proofs.
pub const MSM_GPU_FRAC: f64 = 0.7;

type Limbs = [u32; 8];

const _: () = assert!(std::mem::size_of::<Fq>() == 32);

fn limbs(x: Fq) -> Limbs {
    // SAFETY: `Fq` is a transparent `blst_fr`, 4 little-endian u64 limbs in
    // Montgomery form: the same 32 bytes as 8 little-endian u32 limbs.
    unsafe { std::mem::transmute::<Fq, Limbs>(x) }
}

fn bytes_of<T>(v: &[T]) -> (*const c_void, usize) {
    (v.as_ptr() as *const c_void, std::mem::size_of_val(v))
}

// Layouts shared with the shaders (see quotient.metal).

#[repr(C)]
struct NttParams {
    log_n: u32,
    s: u32,
    r: u32,
    st: u32,
}

#[repr(C)]
struct MsmP {
    n: u32,
    c: u32,
    nwin: u32,
    nbuck: u32,
    nmsm: u32,
}

#[repr(C)]
struct CosetP {
    z1: Limbs,
    z2: Limbs,
    n: u32,
}

#[repr(C)]
struct FinishP {
    divisor: Limbs,
    z1: Limbs,
    z2: Limbs,
}

#[repr(C)]
struct Common {
    fixed: u64,
    advice: u64,
    instance: u64,
    l0: u64,
    l_last: u64,
    l_active: u64,
    y: Limbs,
    one: Limbs,
    n: u32,
    log_scale: u32,
    t: u32,
    nslots: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Prog {
    ops: u64,
    reads: u64,
    uniforms: u64,
    outputs: u64,
    nops: u32,
    noutputs: u32,
}

#[repr(C)]
struct Perm {
    products: u64,
    sigma: u64,
    col_kind: u64,
    col_idx: u64,
    omega_pows: u64,
    beta: Limbs,
    gamma: Limbs,
    delta_start: Limbs,
    delta: Limbs,
    nsets: u32,
    ncols: u32,
    chunk_len: u32,
    last_rot: i32,
    ext_k: u32,
    pad: u32,
}

#[repr(C)]
struct Lookup {
    agg: u64,
    mult: u64,
    helpers: u64,
    progs: u64,
    nchunks: u32,
    pad: u32,
}

#[repr(C)]
struct Trash {
    trash: u64,
    prog: Prog,
    selector: u32,
    pad: u32,
}

const _: () = {
    assert!(std::mem::size_of::<Common>() == 128);
    assert!(std::mem::size_of::<Prog>() == 40);
    assert!(std::mem::size_of::<Perm>() == 192);
    assert!(std::mem::size_of::<Lookup>() == 40);
    assert!(std::mem::size_of::<Trash>() == 56);
};

struct Pipes {
    first: ComputePipelineState,
    pass: ComputePipelineState,
    coset_load: ComputePipelineState,
    mul_periodic: ComputePipelineState,
    scale_coset: ComputePipelineState,
    scale: ComputePipelineState,
    gates: ComputePipelineState,
    perm: ComputePipelineState,
    lookup: ComputePipelineState,
    trash: ComputePipelineState,
    msm_digits: ComputePipelineState,
    msm_scatter: ComputePipelineState,
    msm_tasks: ComputePipelineState,
    msm_reduce: ComputePipelineState,
    msm_segments: ComputePipelineState,
}

/// Counters of the Metal backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalStats {
    /// Quotients served.
    pub quotients: u64,
    /// GPU time of the last quotient command buffer, in microseconds.
    pub last_quotient_gpu_us: u64,
    /// FFTs and inverse FFTs served.
    pub transforms: u64,
    /// MSMs served.
    pub msms: u64,
}

/// The Metal accelerator. Install it with
/// `midnight_proofs::accel::install(Arc::new(MetalAccelerator::new()?))`.
pub struct MetalAccelerator {
    device: Device,
    queue: CommandQueue,
    pipes: Pipes,
    twiddles: Mutex<HashMap<(Limbs, u32), Buffer>>,
    quotients: AtomicU64,
    last_quotient_us: AtomicU64,
    transforms: AtomicU64,
    msms: AtomicU64,
}

impl MetalAccelerator {
    /// Compiles the kernels on the system's default Metal device. `None`
    /// when there is no device or the kernels do not compile.
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let lib = device
            .new_library_with_source(&SHADER.concat(), &CompileOptions::new())
            .map_err(|e| eprintln!("accel-metal: shader compile failed: {e}"))
            .ok()?;
        let pipe = |name: &str| -> Option<ComputePipelineState> {
            let f = lib.get_function(name, None).ok()?;
            device.new_compute_pipeline_state_with_function(&f).ok()
        };
        let pipes = Pipes {
            first: pipe("first_stages_tg")?,
            pass: pipe("pass_r")?,
            coset_load: pipe("coset_load")?,
            mul_periodic: pipe("mul_periodic")?,
            scale_coset: pipe("scale_coset")?,
            scale: pipe("scale")?,
            gates: pipe("numer_gates")?,
            perm: pipe("numer_perm")?,
            lookup: pipe("numer_lookup")?,
            trash: pipe("numer_trash")?,
            msm_digits: pipe("msm_digits")?,
            msm_scatter: pipe("msm_scatter")?,
            msm_tasks: pipe("msm_tasks")?,
            msm_reduce: pipe("msm_reduce")?,
            msm_segments: pipe("msm_segments")?,
        };
        Some(MetalAccelerator {
            queue: device.new_command_queue(),
            device,
            pipes,
            twiddles: Mutex::new(HashMap::new()),
            quotients: AtomicU64::new(0),
            last_quotient_us: AtomicU64::new(0),
            transforms: AtomicU64::new(0),
            msms: AtomicU64::new(0),
        })
    }

    /// The backend's counters.
    pub fn stats(&self) -> MetalStats {
        MetalStats {
            quotients: self.quotients.load(Ordering::Relaxed),
            last_quotient_gpu_us: self.last_quotient_us.load(Ordering::Relaxed),
            transforms: self.transforms.load(Ordering::Relaxed),
            msms: self.msms.load(Ordering::Relaxed),
        }
    }

    // -------------------------------------------------------------- buffers

    fn new_buf(&self, bytes: usize) -> Buffer {
        self.device
            .new_buffer(bytes.max(32) as u64, MTLResourceOptions::StorageModeShared)
    }

    fn buf_of<T>(&self, v: &[T]) -> Buffer {
        let (p, len) = bytes_of(v);
        if len == 0 {
            return self.new_buf(32);
        }
        self.device
            .new_buffer_with_data(p, len as u64, MTLResourceOptions::StorageModeShared)
    }

    /// A buffer over `v`'s own memory when it is page-aligned (no copy).
    /// The caller keeps `v` alive and unchanged by the CPU until the GPU work
    /// that reads or writes the buffer has completed.
    fn wrap<T>(&self, v: &[T]) -> Option<Buffer> {
        let (p, len) = bytes_of(v);
        if len == 0 || !(p as usize).is_multiple_of(PAGE) || !len.is_multiple_of(PAGE) {
            return None;
        }
        // SAFETY: `p` is page-aligned and `len` bytes long. With no
        // deallocator, Metal never frees the memory.
        let raw: *mut metal::MTLBuffer = unsafe {
            msg_send![self.device.as_ref(), newBufferWithBytesNoCopy:p
                length:len as u64
                options:MTLResourceOptions::StorageModeShared
                deallocator:std::ptr::null::<c_void>()]
        };
        // SAFETY: a non-null result is a +1 retained MTLBuffer.
        (!raw.is_null()).then(|| unsafe { Buffer::from_ptr(raw) })
    }

    /// `v` for reading on the GPU: wrapped, or else copied.
    fn read_buf<T>(&self, v: &[T]) -> Buffer {
        self.wrap(v).unwrap_or_else(|| self.buf_of(v))
    }

    /// `omega^i` for `i < 2^(log_n - 1)`, cached per root and size.
    fn twiddles(&self, omega: Fq, log_n: u32) -> Buffer {
        let key = (limbs(omega), log_n);
        if let Some(b) = self.twiddles.lock().unwrap().get(&key) {
            return b.clone();
        }
        let half = 1usize << (log_n - 1);
        let mut tw = vec![Fq::ZERO; half];
        const CHUNK: usize = 1 << 12;
        tw.par_chunks_mut(CHUNK).enumerate().for_each(|(c, out)| {
            let mut w = omega.pow_vartime([(c * CHUNK) as u64]);
            for x in out {
                *x = w;
                w *= omega;
            }
        });
        let b = self.buf_of(&tw);
        self.twiddles.lock().unwrap().insert(key, b.clone());
        b
    }

    // -------------------------------------------------------------- encoding

    fn dispatch_1d(enc: &ComputeCommandEncoderRef, pipe: &ComputePipelineState, n: u64) {
        let tpg = pipe.max_total_threads_per_threadgroup().min(256).min(n);
        enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(tpg, 1, 1));
    }

    fn set<T>(enc: &ComputeCommandEncoderRef, index: u64, v: &T) {
        enc.set_bytes(
            index,
            std::mem::size_of::<T>() as u64,
            v as *const T as *const c_void,
        );
    }

    /// The natural-order FFT of `a` (`2^log_n` values) at the root of `tw`,
    /// in place, using `scratch` (same size). `log_n > TG_STAGES`.
    fn encode_ntt(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,
        scratch: &Buffer,
        tw: &Buffer,
        log_n: u32,
    ) {
        let n = 1u64 << log_n;
        let b = TG_STAGES;
        enc.set_compute_pipeline_state(&self.pipes.first);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(scratch), 0);
        enc.set_buffer(2, Some(tw), 0);
        Self::set(
            enc,
            3,
            &NttParams {
                log_n,
                s: 0,
                r: b,
                st: 0,
            },
        );
        enc.set_threadgroup_memory_length(0, 32u64 << b);
        enc.dispatch_thread_groups(MTLSize::new(n >> b, 1, 1), MTLSize::new(1 << (b - 1), 1, 1));

        enc.set_compute_pipeline_state(&self.pipes.pass);
        enc.set_buffer(2, Some(tw), 0);
        let mut s = b;
        while s < log_n {
            let r = PASS_STAGES.min(log_n - s);
            let last = s + r == log_n;
            enc.set_buffer(0, Some(scratch), 0);
            enc.set_buffer(1, Some(if last { a } else { scratch }), 0);
            Self::set(enc, 3, &NttParams { log_n, s, r, st: 0 });
            Self::dispatch_1d(enc, &self.pipes.pass, n >> r);
            s += r;
        }
    }

    /// FFT (and with `scale`, multiply by it) of `a` on the GPU, in place.
    fn transform(&self, a: &mut [Fq], omega: Fq, log_n: u32, scale: Option<Fq>) -> bool {
        if !(MIN_LOG_N..=30).contains(&log_n) || a.len() != 1 << log_n {
            return false;
        }
        let tw = self.twiddles(omega, log_n);
        let wrapped = self.wrap(a);
        let buf = wrapped.clone().unwrap_or_else(|| self.buf_of(a));
        let scratch = self.new_buf(a.len() * 32);
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        self.encode_ntt(enc, &buf, &scratch, &tw, log_n);
        if let Some(s) = scale {
            enc.set_compute_pipeline_state(&self.pipes.scale);
            enc.set_buffer(0, Some(&buf), 0);
            Self::set(enc, 1, &limbs(s));
            Self::dispatch_1d(enc, &self.pipes.scale, a.len() as u64);
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if cb.status() != MTLCommandBufferStatus::Completed {
            // `a` may be partly written only if it was wrapped; a wrapped
            // failure cannot be undone, so report it loudly.
            assert!(wrapped.is_none(), "accel-metal: in-place transform failed");
            return false;
        }
        if wrapped.is_none() {
            // SAFETY: `buf` holds `a.len()` field elements.
            let res = unsafe { std::slice::from_raw_parts(buf.contents() as *const Fq, a.len()) };
            a.copy_from_slice(res);
        }
        self.transforms.fetch_add(1, Ordering::Relaxed);
        true
    }

    fn prog(&self, p: &DeviceProgram<Fq>, keep: &mut Vec<Buffer>) -> Prog {
        let mut addr = |b: Buffer| {
            let a = b.gpu_address();
            keep.push(b);
            a
        };
        Prog {
            ops: addr(self.buf_of(&p.ops)),
            reads: addr(self.buf_of(&p.reads)),
            uniforms: addr(self.buf_of(&p.uniforms)),
            outputs: addr(self.buf_of(&p.outputs)),
            nops: p.ops.len() as u32,
            noutputs: p.outputs.len() as u32,
        }
    }

    /// The MSMs of every `scalars[j]` with `bases`, declined below
    /// `2^min_log_n` points. The GPU takes the first `frac` of the points of
    /// every MSM and the CPU (blst Pippenger) the rest, at the same time.
    fn msm_split(
        &self,
        scalars: &[&[Fq]],
        bases: &[G1Affine],
        min_log_n: u32,
        frac: f64,
    ) -> Option<Vec<G1Projective>> {
        let n = bases.len();
        if n < (1 << min_log_n) || scalars.iter().any(|s| s.len() != n) {
            return None;
        }
        // The GPU share, counted over the whole batch, is at least
        // 2^MSM_GPU_MIN_LOG_N points, or else everything.
        let mut m = ((n as f64 * frac) as usize).min(n);
        if m * scalars.len() < (1 << knob("MSM_GPU_MIN_LOG_N", MSM_GPU_MIN_LOG_N)) {
            m = n;
        }
        let heads: Vec<&[Fq]> = scalars.iter().map(|s| &s[..m]).collect();
        let (gpu, cpu) = rayon::join(
            || self.msm_gpu(&heads, &bases[..m]),
            || {
                (m < n).then(|| {
                    scalars.iter().map(|s| msm_pippenger(&s[m..], &bases[m..])).collect::<Vec<_>>()
                })
            },
        );
        let mut res = gpu?;
        if let Some(cpu) = cpu {
            res.iter_mut().zip(cpu).for_each(|(r, c)| *r += c);
        }
        Some(res)
    }

    /// The MSMs of every `scalars[j]` with the same `bases`, as one GPU job:
    /// MSM `j`'s window `w` is window `j * nwin + w` of the job.
    fn msm_gpu(&self, scalars: &[&[Fq]], bases: &[G1Affine]) -> Option<Vec<G1Projective>> {
        let n = bases.len();
        let nm = scalars.len();
        if nm == 0
            || n == 0
            || scalars.iter().any(|s| s.len() != n)
            || (n as u64) * (nm as u64) >= 1 << 31
        {
            return None;
        }
        let c = std::env::var("ACCEL_METAL_C")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| n.ilog2().saturating_sub(6).clamp(8, 15));
        let nwin1 = 255u32.div_ceil(c);
        let nwin = nwin1 * nm as u32;
        let nb = 1usize << c;
        let task_len = knob("MSM_TASK", MSM_TASK);
        let group_len = knob("MSM_GROUP", MSM_GROUP);
        let seg_len = knob("MSM_SEG", MSM_SEG).next_power_of_two();
        let nslots = nwin as usize * nb;
        let params = MsmP {
            n: n as u32,
            c,
            nwin: nwin1,
            nbuck: nb as u32,
            nmsm: nm as u32,
        };

        let trace = std::env::var_os("ACCEL_METAL_TRACE").is_some();
        let t0 = std::time::Instant::now();
        let gpu_ms = |cb: &metal::CommandBufferRef| -> f64 {
            let (s, e): (f64, f64) =
                unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
            (e - s) * 1e3
        };
        let sb = if nm == 1 {
            self.read_buf(scalars[0])
        } else {
            self.buf_of(&scalars.concat())
        };
        let bb = self.read_buf(bases);
        let digits = self.new_buf(nwin as usize * n * 4);
        let counts = self.new_buf((nslots + 1) * 4);
        // SAFETY: `counts` holds `nslots + 1` u32 in shared memory.
        let counts_h =
            unsafe { std::slice::from_raw_parts_mut(counts.contents() as *mut u32, nslots + 1) };
        counts_h.fill(0);

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipes.msm_digits);
        enc.set_buffer(0, Some(&sb), 0);
        enc.set_buffer(1, Some(&bb), 0);
        enc.set_buffer(2, Some(&digits), 0);
        enc.set_buffer(3, Some(&counts), 0);
        Self::set(enc, 4, &params);
        Self::dispatch_1d(enc, &self.pipes.msm_digits, (n * nm) as u64);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if cb.status() != MTLCommandBufferStatus::Completed {
            return None;
        }
        let g1 = gpu_ms(cb);
        let t1 = t0.elapsed().as_secs_f64() * 1e3;

        // Bucket offsets (exclusive prefix sum), and a cursor per bucket.
        let mut offsets = vec![0u32; nslots + 1];
        let mut acc = 0u32;
        for (o, cnt) in offsets.iter_mut().zip(counts_h.iter()) {
            *o = acc;
            acc += cnt;
        }
        offsets[nslots] = acc;
        counts_h.copy_from_slice(&offsets);
        // Tasks of at most task_len points, in bucket order; task_off[slot]
        // is the first task of a bucket.
        let mut tasks: Vec<[u32; 2]> =
            Vec::with_capacity(acc as usize / task_len as usize + nslots);
        let mut task_off = vec![0u32; nslots + 1];
        for slot in 0..nslots {
            task_off[slot] = tasks.len() as u32;
            let (mut a, b) = (offsets[slot], offsets[slot + 1]);
            while a < b {
                let e = b.min(a + task_len);
                tasks.push([a, e]);
                a = e;
            }
        }
        task_off[nslots] = tasks.len() as u32;
        let ntasks = tasks.len();
        let tasks_b = self.buf_of(&tasks);
        let nseg = nb.div_ceil(seg_len);
        let pt = std::mem::size_of::<G1Projective>();
        // Reduction levels: each bucket's partials in groups of at most
        // group_len, until every bucket has at most one value.
        let mut ranges: Vec<[u32; 2]> =
            (0..nslots).map(|s| [task_off[s], task_off[s + 1]]).collect();
        let mut levels: Vec<(Buffer, Buffer, usize)> = Vec::new();
        while ranges.iter().any(|r| r[1] - r[0] > 1) {
            let mut groups: Vec<[u32; 2]> = Vec::new();
            for r in ranges.iter_mut() {
                let start = groups.len() as u32;
                let mut a = r[0];
                while a < r[1] {
                    let e = r[1].min(a + group_len);
                    groups.push([a, e]);
                    a = e;
                }
                *r = [start, groups.len() as u32];
            }
            let ng = groups.len();
            levels.push((self.buf_of(&groups), self.new_buf(ng * pt), ng));
        }
        let bidx: Vec<u32> =
            ranges.iter().map(|r| if r[1] > r[0] { r[0] } else { u32::MAX }).collect();
        let bidx_b = self.buf_of(&bidx);
        let seg_s = self.new_buf(nwin as usize * nseg * pt);
        let seg_r = self.new_buf(nwin as usize * nseg * pt);
        let sorted = self.new_buf(acc as usize * 4);
        let out = self.new_buf(ntasks * std::mem::size_of::<G1Projective>());

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipes.msm_scatter);
        enc.set_buffer(0, Some(&digits), 0);
        enc.set_buffer(1, Some(&counts), 0);
        enc.set_buffer(2, Some(&sorted), 0);
        Self::set(enc, 3, &params);
        Self::dispatch_1d(enc, &self.pipes.msm_scatter, (n * nm) as u64);
        if ntasks > 0 {
            enc.set_compute_pipeline_state(&self.pipes.msm_tasks);
            enc.set_buffer(0, Some(&bb), 0);
            enc.set_buffer(1, Some(&sorted), 0);
            enc.set_buffer(2, Some(&tasks_b), 0);
            enc.set_buffer(3, Some(&out), 0);
            Self::set(enc, 4, &(ntasks as u32));
            Self::dispatch_1d(enc, &self.pipes.msm_tasks, ntasks as u64);
        }
        let mut cur = &out;
        for (groups, next, ng) in &levels {
            enc.set_compute_pipeline_state(&self.pipes.msm_reduce);
            enc.set_buffer(0, Some(cur), 0);
            enc.set_buffer(1, Some(groups), 0);
            enc.set_buffer(2, Some(next), 0);
            Self::set(enc, 3, &(*ng as u32));
            Self::dispatch_1d(enc, &self.pipes.msm_reduce, *ng as u64);
            cur = next;
        }
        enc.set_compute_pipeline_state(&self.pipes.msm_segments);
        enc.set_buffer(0, Some(cur), 0);
        enc.set_buffer(4, Some(&bidx_b), 0);
        enc.set_buffer(1, Some(&seg_s), 0);
        enc.set_buffer(2, Some(&seg_r), 0);
        Self::set(enc, 3, &[nb as u32, seg_len as u32, nseg as u32, nwin]);
        Self::dispatch_1d(enc, &self.pipes.msm_segments, (nwin as usize * nseg) as u64);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if cb.status() != MTLCommandBufferStatus::Completed {
            return None;
        }

        let g2 = gpu_ms(cb);
        let t2 = t0.elapsed().as_secs_f64() * 1e3;
        if trace && ntasks > 0 {
            // Trace only: re-run the task kernel alone (idempotent) to time it.
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipes.msm_tasks);
            enc.set_buffer(0, Some(&bb), 0);
            enc.set_buffer(1, Some(&sorted), 0);
            enc.set_buffer(2, Some(&tasks_b), 0);
            enc.set_buffer(3, Some(&out), 0);
            Self::set(enc, 4, &(ntasks as u32));
            Self::dispatch_1d(enc, &self.pipes.msm_tasks, ntasks as u64);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let tasks_ms = gpu_ms(cb);
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipes.msm_segments);
            enc.set_buffer(0, Some(cur), 0);
            enc.set_buffer(4, Some(&bidx_b), 0);
            enc.set_buffer(1, Some(&seg_s), 0);
            enc.set_buffer(2, Some(&seg_r), 0);
            Self::set(enc, 3, &[nb as u32, seg_len as u32, nseg as u32, nwin]);
            Self::dispatch_1d(enc, &self.pipes.msm_segments, (nwin as usize * nseg) as u64);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            eprintln!(
                "  segments kernel alone {:.2}ms ({} threads)",
                gpu_ms(cb),
                nwin as usize * nseg
            );
            eprintln!(
                "  tasks kernel alone {:.2}ms ({} tasks, {} levels, tg max {}, exec width {})",
                tasks_ms,
                ntasks,
                levels.len(),
                self.pipes.msm_tasks.max_total_threads_per_threadgroup(),
                self.pipes.msm_tasks.thread_execution_width()
            );
        }
        // SAFETY: the shader writes blst_p1 (Jacobian) points, and
        // `G1Projective` is a transparent blst_p1.
        let (ss, rs) = unsafe {
            (
                std::slice::from_raw_parts(
                    seg_s.contents() as *const G1Projective,
                    nwin as usize * nseg,
                ),
                std::slice::from_raw_parts(
                    seg_r.contents() as *const G1Projective,
                    nwin as usize * nseg,
                ),
            )
        };
        // Window sum = sum_g S_g + (lo_g - 1) R_g, with lo_g = g * seg_len for
        // g >= 1 (lo_0 = 1): sum_g S_g + seg_len * sum_g g R_g - sum_{g>=1}
        // R_g.
        let sums: Vec<G1Projective> = (0..nwin as usize)
            .into_par_iter()
            .map(|w| {
                let (s, r) = (&ss[w * nseg..(w + 1) * nseg], &rs[w * nseg..(w + 1) * nseg]);
                let mut total = G1Projective::identity();
                for x in s {
                    total += x;
                }
                let (mut run, mut weighted, mut tail) = (
                    G1Projective::identity(),
                    G1Projective::identity(),
                    G1Projective::identity(),
                );
                for x in r[1..].iter().rev() {
                    run += x;
                    weighted += run;
                    tail += x;
                }
                for _ in 0..seg_len.trailing_zeros() {
                    weighted = weighted.double();
                }
                total + weighted - tail
            })
            .collect();
        let res: Vec<G1Projective> = sums
            .chunks(nwin1 as usize)
            .map(|win| {
                let mut res = G1Projective::identity();
                for s in win.iter().rev() {
                    for _ in 0..c {
                        res = res.double();
                    }
                    res += s;
                }
                res
            })
            .collect();
        if trace {
            eprintln!(
                "msm n={n} x{nm} c={c}: digits gpu {g1:.1}ms (wall {t1:.1}), scatter+buckets gpu {g2:.1}ms (wall {:.1}), cpu reduce {:.1}ms",
                t2 - t1,
                t0.elapsed().as_secs_f64() * 1e3 - t2
            );
        }
        self.msms.fetch_add(nm as u64, Ordering::Relaxed);
        Some(res)
    }

    fn quotient_gpu(&self, job: &QuotientJob<'_, Fq>, out: &mut [Fq]) -> Option<()> {
        let ext_k = job.ext_k;
        if !(MIN_LOG_N..=30).contains(&ext_k) || ext_k <= TG_STAGES {
            return None;
        }
        let n = 1usize << job.k;
        let big = 1usize << ext_k;
        let tw_fwd = self.twiddles(job.extended_omega, ext_k);
        let tw_inv = self.twiddles(job.divide.omega_inv, ext_k);
        let scratch = self.new_buf(big * 32);
        let nu = self.new_buf(big * 32);
        let mut keep: Vec<Buffer> = vec![tw_fwd.clone(), tw_inv.clone()];

        // The proof polynomials: coefficients in, extended cosets out.
        let mut expand: Vec<(Buffer, Buffer)> = Vec::new();
        let mut coset = |c: &[Fq]| -> Option<u64> {
            if c.len() != n {
                return None;
            }
            let dst = self.new_buf(big * 32);
            let a = dst.gpu_address();
            expand.push((self.read_buf(c), dst));
            Some(a)
        };
        let advice: Vec<u64> = job.advice.iter().map(|c| coset(c)).collect::<Option<_>>()?;
        let instance: Vec<u64> = job.instance.iter().map(|c| coset(c)).collect::<Option<_>>()?;
        let products: Vec<u64> = match &job.permutation {
            Some(p) => p.products.iter().map(|c| coset(c)).collect::<Option<_>>()?,
            None => vec![],
        };
        let mut lookup_cols = Vec::with_capacity(job.lookups.len());
        for l in &job.lookups {
            let agg = coset(l.aggregator)?;
            let mult = coset(l.multiplicities)?;
            let helpers: Vec<u64> = l.helpers.iter().map(|c| coset(c)).collect::<Option<_>>()?;
            lookup_cols.push((agg, mult, helpers));
        }
        let trash_cols: Vec<u64> =
            job.trash.iter().map(|t| coset(t.trash)).collect::<Option<_>>()?;

        // Key polynomials, already extended cosets.
        let mut key = |c: &[Fq]| -> Option<u64> {
            if c.len() != big {
                return None;
            }
            let b = self.read_buf(c);
            let a = b.gpu_address();
            keep.push(b);
            Some(a)
        };
        let mut fixed = Vec::with_capacity(job.fixed.len());
        for f in &job.fixed {
            fixed.push(match f {
                Some(c) => key(c)?,
                None => 0,
            });
        }
        let (l0, l_last, l_active) = (key(job.l0)?, key(job.l_last)?, key(job.l_active)?);
        let sigma: Vec<u64> = match &job.permutation {
            Some(p) => p.sigma.iter().map(|c| key(c)).collect::<Option<_>>()?,
            None => vec![],
        };

        let table = |keep: &mut Vec<Buffer>, addrs: &[u64]| -> u64 {
            let b = self.buf_of(addrs);
            let a = b.gpu_address();
            keep.push(b);
            a
        };
        let one = limbs(Fq::ONE);
        let nslots = job.nslots().max(1) as usize;
        let t = (SLOT_BYTES / (nslots * 32)).clamp(4096, 1 << 18).min(big) & !255;
        let common = Common {
            fixed: table(&mut keep, &fixed),
            advice: table(&mut keep, &advice),
            instance: table(&mut keep, &instance),
            l0,
            l_last,
            l_active,
            y: limbs(job.y),
            one,
            n: big as u32,
            log_scale: ext_k - job.k,
            t: t as u32,
            nslots: nslots as u32,
        };
        let slots = self.new_buf(nslots * t * 32);

        let gates = self.prog(&job.gates, &mut keep);
        let gates = self.buf_of(std::slice::from_ref(&gates));
        let perm = match &job.permutation {
            Some(p) => {
                let kinds: Vec<u32> = p.columns.iter().map(|c| c.0).collect();
                let idx: Vec<u32> = p.columns.iter().map(|c| c.1).collect();
                let mut pows = Vec::with_capacity(ext_k as usize);
                let mut w = job.extended_omega;
                for _ in 0..ext_k {
                    pows.push(w);
                    w = w.square();
                }
                let s = Perm {
                    products: table(&mut keep, &products),
                    sigma: table(&mut keep, &sigma),
                    col_kind: 0,
                    col_idx: 0,
                    omega_pows: 0,
                    beta: limbs(job.beta),
                    gamma: limbs(job.gamma),
                    delta_start: limbs(job.delta_start),
                    delta: limbs(job.delta),
                    nsets: products.len() as u32,
                    ncols: p.columns.len() as u32,
                    chunk_len: p.chunk_len,
                    last_rot: p.last_rotation,
                    ext_k,
                    pad: 0,
                };
                let (bk, bi, bp) = (self.buf_of(&kinds), self.buf_of(&idx), self.buf_of(&pows));
                let s = Perm {
                    col_kind: bk.gpu_address(),
                    col_idx: bi.gpu_address(),
                    omega_pows: bp.gpu_address(),
                    ..s
                };
                keep.extend([bk, bi, bp]);
                Some(self.buf_of(std::slice::from_ref(&s)))
            }
            None => None,
        };
        let mut lookups = Vec::with_capacity(job.lookups.len());
        for (l, (agg, mult, helpers)) in job.lookups.iter().zip(&lookup_cols) {
            let progs: Vec<Prog> = l.programs.iter().map(|p| self.prog(p, &mut keep)).collect();
            let pb = self.buf_of(&progs);
            let s = Lookup {
                agg: *agg,
                mult: *mult,
                helpers: table(&mut keep, helpers),
                progs: pb.gpu_address(),
                nchunks: progs.len() as u32,
                pad: 0,
            };
            keep.push(pb);
            lookups.push(self.buf_of(std::slice::from_ref(&s)));
        }
        let mut trash = Vec::with_capacity(job.trash.len());
        for (tj, col) in job.trash.iter().zip(&trash_cols) {
            let s = Trash {
                trash: *col,
                prog: self.prog(&tj.program, &mut keep),
                selector: tj.selector_column,
                pad: 0,
            };
            trash.push(self.buf_of(std::slice::from_ref(&s)));
        }
        let t_inv = self.buf_of(&job.divide.t_inv);

        // Encode.
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let mut all: Vec<&metal::ResourceRef> = keep.iter().map(|b| b.as_ref() as _).collect();
        for (s, d) in &expand {
            all.push(s.as_ref());
            all.push(d.as_ref());
        }
        all.extend([gates.as_ref(), slots.as_ref(), nu.as_ref()].map(|b| b as &metal::ResourceRef));
        if let Some(p) = &perm {
            all.push(p.as_ref());
        }
        all.extend(lookups.iter().chain(&trash).map(|b| b.as_ref() as &metal::ResourceRef));
        enc.use_resources(&all, MTLResourceUsage::Read | MTLResourceUsage::Write);

        let coset_p = CosetP {
            z1: limbs(job.zeta[0]),
            z2: limbs(job.zeta[1]),
            n: n as u32,
        };
        for (src, dst) in &expand {
            enc.set_compute_pipeline_state(&self.pipes.coset_load);
            enc.set_buffer(0, Some(src), 0);
            enc.set_buffer(1, Some(dst), 0);
            Self::set(enc, 2, &coset_p);
            Self::dispatch_1d(enc, &self.pipes.coset_load, big as u64);
            self.encode_ntt(enc, dst, &scratch, &tw_fwd, ext_k);
        }

        let rows = |enc: &ComputeCommandEncoderRef, pipe: &ComputePipelineState, arg: &Buffer| {
            enc.set_compute_pipeline_state(pipe);
            Self::set(enc, 0, &common);
            enc.set_buffer(1, Some(arg), 0);
            enc.set_buffer(2, Some(&nu), 0);
            enc.set_buffer(3, Some(&slots), 0);
            Self::dispatch_1d(enc, pipe, t as u64);
        };
        rows(enc, &self.pipes.gates, &gates);
        if let Some(p) = &perm {
            rows(enc, &self.pipes.perm, p);
        }
        for l in &lookups {
            rows(enc, &self.pipes.lookup, l);
        }
        for tb in &trash {
            rows(enc, &self.pipes.trash, tb);
        }

        enc.set_compute_pipeline_state(&self.pipes.mul_periodic);
        enc.set_buffer(0, Some(&nu), 0);
        enc.set_buffer(1, Some(&t_inv), 0);
        Self::set(enc, 2, &(job.divide.t_inv.len() as u32));
        Self::dispatch_1d(enc, &self.pipes.mul_periodic, big as u64);
        self.encode_ntt(enc, &nu, &scratch, &tw_inv, ext_k);
        enc.set_compute_pipeline_state(&self.pipes.scale_coset);
        enc.set_buffer(0, Some(&nu), 0);
        let finish = FinishP {
            divisor: limbs(job.divide.divisor),
            z1: limbs(job.divide.zeta_out[0]),
            z2: limbs(job.divide.zeta_out[1]),
        };
        Self::set(enc, 1, &finish);
        Self::dispatch_1d(enc, &self.pipes.scale_coset, big as u64);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if cb.status() != MTLCommandBufferStatus::Completed {
            return None;
        }
        let (s, e): (f64, f64) =
            unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
        self.last_quotient_us.store(((e - s) * 1e6) as u64, Ordering::Relaxed);

        // SAFETY: `nu` holds `big >= out.len()` field elements.
        let h = unsafe { std::slice::from_raw_parts(nu.contents() as *const Fq, out.len()) };
        out.copy_from_slice(h);
        self.quotients.fetch_add(1, Ordering::Relaxed);
        Some(())
    }
}

impl Accelerator for MetalAccelerator {
    fn msm_g1(&self, scalars: &[Fq], bases: &[G1Affine]) -> Option<G1Projective> {
        let min = knob("MSM_MIN_LOG_N", MSM_MIN_LOG_N);
        let frac = knob("MSM_GPU_FRAC", MSM_GPU_FRAC);
        self.msm_split(&[scalars], bases, min, frac).map(|r| r[0])
    }

    fn msm_g1_batch(&self, scalars: &[&[Fq]], bases: &[G1Affine]) -> Option<Vec<G1Projective>> {
        let min = knob("MSM_BATCH_MIN_LOG_N", MSM_BATCH_MIN_LOG_N);
        let frac = knob("MSM_BATCH_GPU_FRAC", MSM_BATCH_GPU_FRAC);
        self.msm_split(scalars, bases, min, frac)
    }

    fn fft(&self, a: &mut [Fq], omega: Fq, log_n: u32) -> bool {
        self.transform(a, omega, log_n, None)
    }

    fn ifft(&self, a: &mut [Fq], omega_inv: Fq, log_n: u32, divisor: Fq) -> bool {
        self.transform(a, omega_inv, log_n, Some(divisor))
    }

    fn quotient(&self, job: &QuotientJob<'_, Fq>, out: &mut [Fq]) -> bool {
        out.len() == job.divide.len && self.quotient_gpu(job, out).is_some()
    }
}
