// SPDX-FileCopyrightText: Copyright (C) Shielded Technologies
// SPDX-License-Identifier: Apache-2.0

//! The Metal MSM against the CPU Pippenger (blst): the same group element.

use ff::Field;
use group::{Curve, Group, prime::PrimeCurveAffine};
use midnight_accel_metal::{MSM_BATCH_MIN_LOG_N, MSM_MIN_LOG_N, MetalAccelerator};
use midnight_curves::{Fq, G1Affine, G1Projective};
use midnight_proofs::accel::{Accelerator, reference::CpuReference};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

/// `n` distinct bases: a random start plus multiples of a random step.
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

fn check(gpu: &MetalAccelerator, s: &[Fq], b: &[G1Affine], what: &str) {
    let want = CpuReference.msm_g1(s, b).unwrap();
    let got = gpu.msm_g1(s, b).unwrap_or_else(|| panic!("{what}: declined"));
    assert!(got == want, "{what}: MSM differs");
}

#[test]
fn random_msms_match_reference() {
    let gpu = MetalAccelerator::new().unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(7);
    for log_n in [16, 17, 18] {
        let n = 1 << log_n;
        let b = bases(&mut rng, n);
        let s: Vec<Fq> = (0..n).map(|_| Fq::random(&mut rng)).collect();
        check(&gpu, &s, &b, &format!("random 2^{log_n}"));
    }
    assert!(
        gpu.msm_g1(&[Fq::ONE; 16], &bases(&mut rng, 16)).is_none(),
        "small: declined"
    );
}

#[test]
fn edge_cases_match_reference() {
    let gpu = MetalAccelerator::new().unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(8);
    let n = 1 << MSM_MIN_LOG_N;
    let mut b = bases(&mut rng, n);
    let mut s: Vec<Fq> = (0..n).map(|_| Fq::random(&mut rng)).collect();
    // A repeated base with the same scalar: the same bucket in every
    // window, so the bucket sum doubles.
    b[1] = b[0];
    s[1] = s[0];
    // A point and its negation in the same buckets: the identity.
    b[3] = -b[2];
    s[3] = s[2];
    // The identity base, and scalars 0, 1, -1.
    b[4] = G1Affine::identity();
    s[5] = Fq::ZERO;
    s[6] = Fq::ONE;
    s[7] = -Fq::ONE;
    check(&gpu, &s, &b, "edge cases");

    // All scalars zero: the identity.
    let z = vec![Fq::ZERO; n];
    check(&gpu, &z, &b, "all zero");
    // Every scalar the same.
    let same = vec![s[0]; n];
    check(&gpu, &same, &b, "all equal");
}

/// A batch over shared bases: each result is its own MSM, including an
/// all-zero and a repeated scalar vector, in batches of 2 to 9.
#[test]
fn batched_msms_match_reference() {
    let gpu = MetalAccelerator::new().unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(10);
    for (log_n, nm) in [(MSM_BATCH_MIN_LOG_N, 9), (14, 2), (16, 5)] {
        let n = 1 << log_n;
        let b = bases(&mut rng, n);
        let mut s: Vec<Vec<Fq>> =
            (0..nm).map(|_| (0..n).map(|_| Fq::random(&mut rng)).collect()).collect();
        s[1] = vec![Fq::ZERO; n];
        if nm > 2 {
            s[2] = s[0].clone();
        }
        let refs: Vec<&[Fq]> = s.iter().map(|v| &v[..]).collect();
        let want = CpuReference.msm_g1_batch(&refs, &b).unwrap();
        let got = gpu
            .msm_g1_batch(&refs, &b)
            .unwrap_or_else(|| panic!("2^{log_n} x{nm}: declined"));
        assert!(got == want, "2^{log_n} x{nm}: batch differs");
    }
    let small = bases(&mut rng, 16);
    assert!(
        gpu.msm_g1_batch(&[&[Fq::ONE; 16], &[Fq::ONE; 16]], &small).is_none(),
        "small batch: declined"
    );
}

/// The comparison can fail: a changed scalar changes the result.
#[test]
fn msm_check_is_sensitive() {
    let gpu = MetalAccelerator::new().unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(9);
    let n = 1 << MSM_MIN_LOG_N;
    let b = bases(&mut rng, n);
    let mut s: Vec<Fq> = (0..n).map(|_| Fq::random(&mut rng)).collect();
    let want = CpuReference.msm_g1(&s, &b).unwrap();
    s[17] += Fq::ONE;
    assert!(gpu.msm_g1(&s, &b).unwrap() != want);
    let _ = b[0].to_curve();
}
