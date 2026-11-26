// SPDX-FileCopyrightText: Copyright (C) Shielded Technologies
// SPDX-License-Identifier: Apache-2.0

//! The adapter against the CPU reference: every value it serves equals the
//! stock code's, and proofs made with it installed verify.

use std::sync::{Arc, Mutex};

use ff::Field;
use group::{Curve, Group, prime::PrimeCurveAffine};
use midnight_accel_metal::chain;
use midnight_curves::{Fq, G1Affine, G1Projective};
use midnight_metal_accel::SeungheonMetal;
use midnight_proofs::{
    accel::{self, Accelerator, Op, QuotientJob, reference::CpuReference},
    poly::EvaluationDomain,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

/// The accelerator is process-wide: tests that install one take this lock.
static INSTALLS: Mutex<()> = Mutex::new(());

fn bases(rng: &mut ChaCha20Rng, n: usize) -> Vec<G1Affine> {
    let step = G1Projective::random(&mut *rng);
    let mut p = G1Projective::random(&mut *rng);
    let proj: Vec<G1Projective> = (0..n)
        .map(|_| {
            p += step;
            p
        })
        .collect();
    let mut aff = vec![G1Affine::identity(); n];
    G1Projective::batch_normalize(&proj, &mut aff);
    aff
}

#[test]
fn msms_match_reference() {
    let a = SeungheonMetal::new().expect("Metal device");
    let mut rng = ChaCha20Rng::seed_from_u64(3);
    let n = 1 << 16;
    let b = bases(&mut rng, n);
    let s: Vec<Vec<Fq>> = (0..4)
        .map(|_| (0..n).map(|_| Fq::random(&mut rng)).collect())
        .collect();
    let mut sparse = vec![Fq::ZERO; n];
    sparse[5] = Fq::ONE;
    // First sighting declines (no table yet); the second registers.
    assert!(a.msm_g1(&s[0], &b).is_none(), "first sighting declines");
    let got = a.msm_g1(&s[0], &b).expect("served after registration");
    assert!(
        got == CpuReference.msm_g1(&s[0], &b).unwrap(),
        "MSM differs"
    );
    let refs: Vec<&[Fq]> = s.iter().map(|v| &v[..]).chain([&sparse[..]]).collect();
    let got = a.msm_g1_batch(&refs, &b).expect("batch served");
    assert!(
        got == CpuReference.msm_g1_batch(&refs, &b).unwrap(),
        "batch differs"
    );
    let mut bad = s[1].clone();
    bad[9] += Fq::ONE;
    assert!(
        a.msm_g1(&bad, &b).unwrap() != got[1],
        "a changed scalar changes the MSM"
    );
    let _ = b[0].to_curve();
}

#[test]
fn transforms_match_reference() {
    let a = SeungheonMetal::new().expect("Metal device");
    let mut rng = ChaCha20Rng::seed_from_u64(4);
    for k in [15u32, 17] {
        let d = EvaluationDomain::<Fq>::new(3, k);
        let v: Vec<Fq> = (0..1 << k).map(|_| Fq::random(&mut rng)).collect();
        let (mut got, mut want) = (v.clone(), v.clone());
        assert!(a.fft(&mut got, d.get_omega(), k), "fft 2^{k} served");
        CpuReference.fft(&mut want, d.get_omega(), k);
        assert!(got == want, "fft 2^{k} differs");
        let div = Fq::from(1u64 << k).invert().unwrap();
        let (mut got, mut want) = (v.clone(), v);
        assert!(
            a.ifft(&mut got, d.get_omega_inv(), k, div),
            "ifft 2^{k} served"
        );
        CpuReference.ifft(&mut want, d.get_omega_inv(), k, div);
        assert!(got == want, "ifft 2^{k} differs");
    }
}

/// Serves with the adapter and checks each quotient against the reference.
struct Checked(SeungheonMetal, Mutex<u32>);

impl Accelerator for Checked {
    fn msm_g1(&self, s: &[Fq], b: &[G1Affine]) -> Option<G1Projective> {
        self.0.msm_g1(s, b)
    }
    fn msm_g1_batch(&self, s: &[&[Fq]], b: &[G1Affine]) -> Option<Vec<G1Projective>> {
        self.0.msm_g1_batch(s, b)
    }
    fn fft(&self, a: &mut [Fq], w: Fq, l: u32) -> bool {
        self.0.fft(a, w, l)
    }
    fn ifft(&self, a: &mut [Fq], w: Fq, l: u32, d: Fq) -> bool {
        self.0.ifft(a, w, l, d)
    }
    fn begin_proof(&self) {
        self.0.begin_proof()
    }
    fn end_proof(&self) {
        self.0.end_proof()
    }
    fn quotient(&self, job: &QuotientJob<'_, Fq>, out: &mut [Fq]) -> bool {
        let served = self.0.quotient(job, out);
        if served {
            let mut want = vec![Fq::ZERO; out.len()];
            assert!(CpuReference.quotient(job, &mut want));
            assert!(*out == *want, "the quotient differs from the reference");
            *self.1.lock().unwrap() += 1;
        }
        served
    }
}

#[test]
fn chain_proofs_verify() {
    let _lock = INSTALLS.lock().unwrap_or_else(|e| e.into_inner());
    let k = 16;
    let (params, pk) = chain::setup(k);
    let checked = Arc::new(Checked(
        SeungheonMetal::new().expect("Metal device"),
        Mutex::new(0),
    ));
    accel::install(checked.clone());
    let before = accel::stats(Op::MsmBatch).0 + accel::stats(Op::Msm).0;
    for _ in 0..3 {
        let proof = chain::prove(&params, &pk, k);
        assert!(
            chain::verify(&params, pk.get_vk(), k, &proof),
            "proof verifies"
        );
        let mut bad = proof;
        let last = bad.len() - 1;
        bad[last] ^= 1;
        assert!(
            !chain::verify(&params, pk.get_vk(), k, &bad),
            "a tampered proof fails"
        );
    }
    accel::uninstall();
    assert_eq!(
        *checked.1.lock().unwrap(),
        3,
        "every quotient served and checked"
    );
    assert!(
        accel::stats(Op::MsmBatch).0 + accel::stats(Op::Msm).0 > before,
        "MSMs served"
    );
}
