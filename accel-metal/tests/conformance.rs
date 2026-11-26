// SPDX-FileCopyrightText: Copyright (C) Shielded Technologies
// SPDX-License-Identifier: Apache-2.0

//! The Metal backend against the CPU reference: every served output must be
//! bit-identical.

use ff::{Field, PrimeField};
use midnight_accel_metal::MetalAccelerator;
use midnight_curves::Fq;
use midnight_proofs::accel::{
    Accelerator, COLUMN_ADVICE, COLUMN_FIXED, COLUMN_INSTANCE, DeviceOp, DeviceProgram, DeviceRead,
    LookupJob, OP_MUL_ADD, PermutationJob, QuotientDivide, QuotientJob, TAG_READ, TAG_SLOT,
    TAG_UNIFORM, TrashJob, operand, reference::CpuReference,
};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

fn metal() -> MetalAccelerator {
    MetalAccelerator::new().expect("a Metal device")
}

fn root(log_n: u32) -> Fq {
    let mut w = Fq::ROOT_OF_UNITY;
    for _ in log_n..Fq::S {
        w = w.square();
    }
    w
}

fn rand_vec(rng: &mut ChaCha20Rng, n: usize) -> Vec<Fq> {
    (0..n).map(|_| Fq::random(&mut *rng)).collect()
}

#[test]
fn transforms_match_reference() {
    let gpu = metal();
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    for log_n in 12..=17 {
        let a = rand_vec(&mut rng, 1 << log_n);
        let w = root(log_n);

        let (mut x, mut y) = (a.clone(), a.clone());
        assert!(gpu.fft(&mut x, w, log_n));
        assert!(CpuReference.fft(&mut y, w, log_n));
        assert!(x == y, "fft at log_n = {log_n}");

        let d = Fq::from(1u64 << log_n).invert().unwrap();
        let (mut x, mut y) = (a.clone(), a);
        assert!(gpu.ifft(&mut x, w.invert().unwrap(), log_n, d));
        assert!(CpuReference.ifft(&mut y, w.invert().unwrap(), log_n, d));
        assert!(x == y, "ifft at log_n = {log_n}");
    }
    // Too small: declined, input untouched.
    let mut small = rand_vec(&mut rng, 1 << 8);
    let before = small.clone();
    assert!(!gpu.fft(&mut small, root(8), 8));
    assert!(small == before);
}

/// A random program over `nf` fixed, `na` advice and `ni` instance columns,
/// with `nout` outputs. Every slot is written before it is read.
fn rand_prog(
    rng: &mut ChaCha20Rng,
    (nf, na, ni): (u32, u32, u32),
    nout: usize,
) -> DeviceProgram<Fq> {
    let nuni = 3;
    let nreads = 4;
    let reads: Vec<DeviceRead> = (0..nreads)
        .map(|_| {
            let kind = rng.next_u32() % 3;
            let ncols = [nf, na, ni][kind as usize];
            DeviceRead {
                kind: [COLUMN_FIXED, COLUMN_ADVICE, COLUMN_INSTANCE][kind as usize],
                column: rng.next_u32() % ncols,
                rotation: (rng.next_u32() % 5) as i32 - 2,
            }
        })
        .collect();
    let nslots = 4u32;
    let mut written = 0u32;
    let pick = |rng: &mut ChaCha20Rng, written: u32| -> u32 {
        match rng.next_u32() % 3 {
            0 => operand(TAG_UNIFORM, rng.next_u32() % nuni),
            1 if written > 0 => operand(TAG_SLOT, rng.next_u32() % written),
            _ => operand(TAG_READ, rng.next_u32() % nreads),
        }
    };
    let mut ops = Vec::new();
    for i in 0..12 {
        let dst = i % nslots;
        let op = DeviceOp {
            kind: rng.next_u32() % (OP_MUL_ADD + 1),
            dst,
            a: pick(rng, written),
            b: pick(rng, written),
            c: pick(rng, written),
        };
        ops.push(op);
        written = written.max(dst + 1);
    }
    let outputs = (0..nout).map(|_| pick(rng, written)).collect();
    DeviceProgram {
        uniforms: rand_vec(rng, nuni as usize),
        reads,
        ops,
        nslots,
        outputs,
    }
}

fn synthetic_job_matches(seed: u64) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let (k, ext_k) = (10u32, 12u32);
    let (n, big) = (1usize << k, 1usize << ext_k);
    let shape = (3u32, 3u32, 2u32);
    let coeffs = |rng: &mut ChaCha20Rng, m: usize| -> Vec<Vec<Fq>> {
        (0..m).map(|_| rand_vec(rng, n)).collect()
    };
    let cosets = |rng: &mut ChaCha20Rng, m: usize| -> Vec<Vec<Fq>> {
        (0..m).map(|_| rand_vec(rng, big)).collect()
    };
    let advice = coeffs(&mut rng, 3);
    let instance = coeffs(&mut rng, 2);
    let fixed = cosets(&mut rng, 3);
    let lag = cosets(&mut rng, 3);
    let sigma = cosets(&mut rng, 5);
    let products = coeffs(&mut rng, 3);
    let lookup_polys: Vec<(Vec<Vec<Fq>>, usize)> =
        [1usize, 2].iter().map(|&c| (coeffs(&mut rng, 2 + c), c)).collect();
    let trash = coeffs(&mut rng, 1);
    let w = root(ext_k);

    let job = QuotientJob {
        k,
        ext_k,
        zeta: [Fq::random(&mut rng), Fq::random(&mut rng)],
        extended_omega: w,
        advice: advice.iter().map(Vec::as_slice).collect(),
        instance: instance.iter().map(Vec::as_slice).collect(),
        fixed: fixed.iter().map(|c| Some(c.as_slice())).collect(),
        l0: &lag[0],
        l_last: &lag[1],
        l_active: &lag[2],
        y: Fq::random(&mut rng),
        beta: Fq::random(&mut rng),
        gamma: Fq::random(&mut rng),
        delta_start: Fq::random(&mut rng),
        delta: Fq::random(&mut rng),
        gates: rand_prog(&mut rng, shape, 1),
        permutation: Some(PermutationJob {
            chunk_len: 2,
            columns: (0..5)
                .map(|i| match i % 3 {
                    0 => (COLUMN_ADVICE, i % 3),
                    1 => (COLUMN_FIXED, i % 3),
                    _ => (COLUMN_INSTANCE, i % 2),
                })
                .collect(),
            sigma: sigma.iter().map(Vec::as_slice).collect(),
            products: products.iter().map(Vec::as_slice).collect(),
            last_rotation: -3,
        }),
        lookups: lookup_polys
            .iter()
            .map(|(p, c)| LookupJob {
                aggregator: &p[0],
                multiplicities: &p[1],
                helpers: p[2..].iter().map(Vec::as_slice).collect(),
                programs: (0..*c).map(|_| rand_prog(&mut rng, shape, 4)).collect(),
            })
            .collect(),
        trash: vec![TrashJob {
            trash: &trash[0],
            selector_column: 2,
            program: rand_prog(&mut rng, shape, 1),
        }],
        divide: QuotientDivide {
            t_inv: rand_vec(&mut rng, big / n),
            omega_inv: w.invert().unwrap(),
            divisor: Fq::from(big as u64).invert().unwrap(),
            zeta_out: [Fq::random(&mut rng), Fq::random(&mut rng)],
            len: (n - 1) * 3,
        },
    };

    let mut want = vec![Fq::ZERO; job.divide.len];
    assert!(CpuReference.quotient(&job, &mut want));
    let mut got = vec![Fq::ZERO; job.divide.len];
    assert!(metal().quotient(&job, &mut got), "seed {seed}: declined");
    let first_diff = got.iter().zip(&want).position(|(a, b)| a != b);
    assert_eq!(first_diff, None, "seed {seed}: quotient differs");
}

#[test]
fn synthetic_quotient_jobs_match_reference() {
    for seed in 0..6 {
        synthetic_job_matches(seed);
    }
}

/// The comparison can fail: a changed job gives another quotient.
#[test]
fn synthetic_check_is_sensitive() {
    let gpu = metal();
    let mut rng = ChaCha20Rng::seed_from_u64(9);
    let (k, ext_k) = (10u32, 12u32);
    let (n, big) = (1usize << k, 1usize << ext_k);
    let a = rand_vec(&mut rng, n);
    let f = rand_vec(&mut rng, big);
    let lag = rand_vec(&mut rng, big);
    let mut job = QuotientJob {
        k,
        ext_k,
        zeta: [Fq::random(&mut rng), Fq::random(&mut rng)],
        extended_omega: root(ext_k),
        advice: vec![&a],
        instance: vec![],
        fixed: vec![Some(&f)],
        l0: &lag,
        l_last: &lag,
        l_active: &lag,
        y: Fq::random(&mut rng),
        beta: Fq::ONE,
        gamma: Fq::ONE,
        delta_start: Fq::ONE,
        delta: Fq::ONE,
        gates: rand_prog(&mut rng, (1, 1, 1), 1),
        permutation: None,
        lookups: vec![],
        trash: vec![],
        divide: QuotientDivide {
            t_inv: rand_vec(&mut rng, big / n),
            omega_inv: root(ext_k).invert().unwrap(),
            divisor: Fq::ONE,
            zeta_out: [Fq::ONE, Fq::ONE],
            len: n,
        },
    };
    // The random program may read an instance column; give it none.
    for r in &mut job.gates.reads {
        r.kind = if r.kind == COLUMN_FIXED {
            COLUMN_FIXED
        } else {
            COLUMN_ADVICE
        };
        r.column = 0;
    }
    let mut x = vec![Fq::ZERO; n];
    let mut want = vec![Fq::ZERO; n];
    assert!(gpu.quotient(&job, &mut x));
    assert!(CpuReference.quotient(&job, &mut want));
    assert!(x == want);
    job.divide.t_inv[0] += Fq::ONE;
    assert!(gpu.quotient(&job, &mut x));
    assert!(
        x != want,
        "a changed divisor table must change the quotient"
    );
}
