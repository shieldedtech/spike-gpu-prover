// SPDX-FileCopyrightText: Copyright (C) Shielded Technologies
// SPDX-License-Identifier: Apache-2.0

//! SeungHeon's Metal backend (`midnight-metal`, from SeungheonOh/midnight-zk
//! `metal-prover`) behind the `midnight-proofs` accelerator hooks
//! (`midnight_proofs::accel`, feature `accel`).
//!
//! The adapter only converts types and keeps the backend's registries:
//!
//! - MSMs: a base slice is registered as an SRS table the second time it is
//!   seen, so only bases that come back (the SRS) get a table; the backend
//!   compares every point against the table on each MSM, so a reused address
//!   cannot alias. An MSM the backend declines (unregistered or sparse
//!   bases) runs the stock CPU code.
//! - Quotient: the job is packed into the backend's ABI, field for field
//!   (the two hook layers share the job's definition), and the proving key's
//!   polynomials are registered once per key, so the backend keeps device
//!   copies of them.
//! - Proofs: each proof is one backend request, so its scratch is kept for
//!   the proof's calls and wiped when the proof ends.
//!
//! Every value either equals the stock code's or the call declines.

use std::{collections::HashMap, ffi::c_void, sync::Mutex};

use midnight_curves::{Fq, G1Affine, G1Projective, msm::msm_pippenger};
use midnight_metal as metal;
use midnight_metal::quotient_abi as abi;
use midnight_proofs::accel::{Accelerator, DeviceProgram, QuotientJob};

const _: () = {
    assert!(size_of::<Fq>() == size_of::<metal::Scalar>());
    assert!(align_of::<Fq>() == align_of::<metal::Scalar>());
    assert!(size_of::<G1Affine>() == size_of::<metal::G1AffineLimbs>());
    assert!(size_of::<G1Projective>() == size_of::<[u64; 18]>());
    assert!(size_of::<midnight_proofs::accel::DeviceOp>() == size_of::<abi::DeviceOp>());
    assert!(size_of::<midnight_proofs::accel::DeviceRead>() == size_of::<abi::DeviceRead>());
};

// SAFETY (all four): `Fq`, `G1Affine` and `G1Projective` are transparent
// `blst_fr`, `blst_p1_affine` and `blst_p1`, the limb layouts the backend
// takes (sizes and alignment checked above).
fn sc(a: &[Fq]) -> &[metal::Scalar] {
    unsafe { std::slice::from_raw_parts(a.as_ptr().cast(), a.len()) }
}
fn sc_mut(a: &mut [Fq]) -> &mut [metal::Scalar] {
    unsafe { std::slice::from_raw_parts_mut(a.as_mut_ptr().cast(), a.len()) }
}
fn one(a: Fq) -> metal::Scalar {
    sc(std::slice::from_ref(&a))[0]
}
fn pts(a: &[G1Affine]) -> &[metal::G1AffineLimbs] {
    unsafe { std::slice::from_raw_parts(a.as_ptr().cast(), a.len()) }
}
fn point(r: [u64; 18]) -> G1Projective {
    unsafe { std::mem::transmute::<[u64; 18], G1Projective>(r) }
}

/// The adapter. Install it with
/// `midnight_proofs::accel::install(Arc::new(SeungheonMetal::new()?))`.
pub struct SeungheonMetal {
    /// Base slices by (address, length): `None` seen once, `Some` registered.
    srs: Mutex<HashMap<(usize, usize), Option<metal::SrsHandle>>>,
    /// Registered proving keys, by the address of their `l_0` coset.
    keys: Mutex<HashMap<usize, metal::KeyHandle>>,
}

impl SeungheonMetal {
    /// `None` without a Metal device.
    pub fn new() -> Option<Self> {
        metal::device().then(|| SeungheonMetal {
            srs: Mutex::new(HashMap::new()),
            keys: Mutex::new(HashMap::new()),
        })
    }

    /// Whether `bases` has a table (registering it at its second sighting).
    fn table(&self, bases: &[G1Affine]) -> bool {
        let mut t = self.srs.lock().unwrap();
        match t.get_mut(&(bases.as_ptr() as usize, bases.len())) {
            Some(Some(_)) => true,
            Some(slot) => {
                *slot = Some(metal::register_srs(pts(bases)));
                true
            }
            None => {
                t.insert((bases.as_ptr() as usize, bases.len()), None);
                false
            }
        }
    }

    /// Registers the key polynomials of `job` once per key, in the order of
    /// `resident_key_polys`: the fixed cosets it reads, the permutation
    /// cosets, then `l_0`, `l_last` and `l_active`.
    fn register_key(&self, job: &QuotientJob<'_, Fq>) {
        let mut k = self.keys.lock().unwrap();
        k.entry(job.l0.as_ptr() as usize).or_insert_with(|| {
            let mut v: Vec<&[metal::Scalar]> = job.fixed.iter().flatten().map(|c| sc(c)).collect();
            if let Some(p) = &job.permutation {
                v.extend(p.sigma.iter().map(|c| sc(c)));
            }
            v.extend([sc(job.l0), sc(job.l_last), sc(job.l_active)]);
            metal::register_key(&v)
        });
    }
}

impl Accelerator for SeungheonMetal {
    fn msm_g1(&self, scalars: &[Fq], bases: &[G1Affine]) -> Option<G1Projective> {
        if !self.table(bases) {
            return None;
        }
        metal::msm_g1(sc(scalars), pts(bases)).map(point)
    }

    /// One backend MSM each; one it declines (a sparse MSM) runs on the CPU.
    fn msm_g1_batch(&self, scalars: &[&[Fq]], bases: &[G1Affine]) -> Option<Vec<G1Projective>> {
        if !self.table(bases) {
            return None;
        }
        Some(
            scalars
                .iter()
                .map(|s| {
                    metal::msm_g1(sc(s), pts(bases)).map_or_else(|| msm_pippenger(s, bases), point)
                })
                .collect(),
        )
    }

    fn fft(&self, a: &mut [Fq], omega: Fq, log_n: u32) -> bool {
        metal::fft(sc_mut(a), one(omega), log_n)
    }

    fn ifft(&self, a: &mut [Fq], omega_inv: Fq, log_n: u32, divisor: Fq) -> bool {
        metal::ifft(sc_mut(a), one(omega_inv), log_n, one(divisor))
    }

    fn quotient(&self, job: &QuotientJob<'_, Fq>, out: &mut [Fq]) -> bool {
        self.register_key(job);
        let packed = Packed::new(job);
        // SAFETY: `packed` points into `job` and its own tables, which live
        // for the call.
        unsafe { metal::quotient(packed.as_ptr(), sc_mut(out)) }
    }

    fn begin_proof(&self) {
        metal::begin_request();
    }

    fn end_proof(&self) {
        // A failed wipe is reported by the backend's counters; the proof's
        // own result does not depend on it.
        let _ = metal::end_request();
    }
}

/// A job packed into the backend's ABI, with the tables its pointers use.
/// Mirrors `PackedJob::new` of SeungHeon's hook layer.
struct Packed {
    job: Box<abi::QuotientJobAbi>,
    _ptrs: Vec<Vec<*const metal::Scalar>>,
    _u32s: Vec<Vec<u32>>,
    _programs: Vec<Vec<abi::ProgramAbi>>,
    _lookups: Vec<abi::LookupAbi>,
    _trash: Vec<abi::TrashAbi>,
}

impl Packed {
    fn as_ptr(&self) -> *const c_void {
        &*self.job as *const abi::QuotientJobAbi as *const c_void
    }

    fn new(job: &QuotientJob<'_, Fq>) -> Self {
        let mut ptrs: Vec<Vec<*const metal::Scalar>> = Vec::new();
        let mut keep = |v: Vec<*const metal::Scalar>| -> *const *const metal::Scalar {
            let p = v.as_ptr();
            ptrs.push(v);
            p
        };
        let poly = |p: &[Fq]| sc(p).as_ptr();
        let program = |p: &DeviceProgram<Fq>| abi::ProgramAbi {
            ops: p.ops.as_ptr().cast(),
            nops: p.ops.len(),
            reads: p.reads.as_ptr().cast(),
            nreads: p.reads.len(),
            uniforms: sc(&p.uniforms).as_ptr(),
            nuniforms: p.uniforms.len(),
            outputs: p.outputs.as_ptr(),
            noutputs: p.outputs.len(),
            nslots: p.nslots,
            pad: 0,
        };
        let advice = keep(job.advice.iter().map(|p| poly(p)).collect());
        let instance = keep(job.instance.iter().map(|p| poly(p)).collect());
        let fixed = keep(
            job.fixed
                .iter()
                .map(|p| p.map_or(std::ptr::null(), poly))
                .collect(),
        );
        let mut u32s = Vec::new();
        let mut programs = Vec::new();
        let (mut perm_column_kind, mut perm_column_index) = (std::ptr::null(), std::ptr::null());
        let (mut perm_sigma, mut perm_products) = (std::ptr::null(), std::ptr::null());
        let (mut perm_ncolumns, mut perm_nsets, mut perm_chunk_len, mut perm_last_rotation) =
            (0, 0, 0, 0);
        if let Some(p) = &job.permutation {
            let kinds: Vec<u32> = p.columns.iter().map(|c| c.0).collect();
            let idx: Vec<u32> = p.columns.iter().map(|c| c.1).collect();
            perm_column_kind = kinds.as_ptr();
            perm_column_index = idx.as_ptr();
            u32s.push(kinds);
            u32s.push(idx);
            perm_sigma = keep(p.sigma.iter().map(|s| poly(s)).collect());
            perm_products = keep(p.products.iter().map(|s| poly(s)).collect());
            perm_ncolumns = p.columns.len();
            perm_nsets = p.products.len();
            perm_chunk_len = p.chunk_len;
            perm_last_rotation = p.last_rotation;
        }
        let mut lookups = Vec::with_capacity(job.lookups.len());
        for l in &job.lookups {
            let progs: Vec<abi::ProgramAbi> = l.programs.iter().map(program).collect();
            let helpers = keep(l.helpers.iter().map(|h| poly(h)).collect());
            lookups.push(abi::LookupAbi {
                aggregator: poly(l.aggregator),
                multiplicities: poly(l.multiplicities),
                helpers,
                programs: progs.as_ptr(),
                nchunks: l.helpers.len(),
            });
            programs.push(progs);
        }
        let trash: Vec<abi::TrashAbi> = job
            .trash
            .iter()
            .map(|t| abi::TrashAbi {
                trash: poly(t.trash),
                program: program(&t.program),
                selector_column: t.selector_column,
                pad: 0,
            })
            .collect();
        let d = &job.divide;
        let packed = abi::QuotientJobAbi {
            abi: abi::QUOTIENT_ABI,
            k: job.k,
            ext_k: job.ext_k,
            nslots: job.nslots(),
            zeta: [one(job.zeta[0]), one(job.zeta[1])],
            extended_omega: one(job.extended_omega),
            y: one(job.y),
            beta: one(job.beta),
            gamma: one(job.gamma),
            delta_start: one(job.delta_start),
            delta: one(job.delta),
            advice,
            nadvice: job.advice.len(),
            instance,
            ninstance: job.instance.len(),
            fixed,
            nfixed: job.fixed.len(),
            l0: poly(job.l0),
            l_last: poly(job.l_last),
            l_active: poly(job.l_active),
            gates: program(&job.gates),
            perm_ncolumns,
            perm_column_kind,
            perm_column_index,
            perm_sigma,
            perm_products,
            perm_nsets,
            perm_chunk_len,
            perm_last_rotation,
            lookups: lookups.as_ptr(),
            nlookups: lookups.len(),
            trash: trash.as_ptr(),
            ntrash: trash.len(),
            t_inv: sc(&d.t_inv).as_ptr(),
            nt_inv: d.t_inv.len(),
            omega_inv: one(d.omega_inv),
            divisor: one(d.divisor),
            zeta_out: [one(d.zeta_out[0]), one(d.zeta_out[1])],
            len: d.len,
        };
        Packed {
            job: Box::new(packed),
            _ptrs: ptrs,
            _u32s: u32s,
            _programs: programs,
            _lookups: lookups,
            _trash: trash,
        }
    }
}
