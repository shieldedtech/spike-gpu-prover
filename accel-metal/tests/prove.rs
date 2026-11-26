// SPDX-FileCopyrightText: Copyright (C) Shielded Technologies
// SPDX-License-Identifier: Apache-2.0

//! End to end: proofs made with the Metal backend installed verify, and every
//! quotient it serves equals the CPU reference's.

use std::sync::Arc;

use ff::Field;
use midnight_accel_metal::{MetalAccelerator, chain};
use midnight_curves::{Fq, G1Affine, G1Projective};
use midnight_proofs::accel::{self, Accelerator, Op, QuotientJob, reference::CpuReference};

/// Serves with Metal and checks each quotient against the CPU reference.
struct Checked(MetalAccelerator);

impl Accelerator for Checked {
    fn msm_g1(&self, s: &[Fq], b: &[G1Affine]) -> Option<G1Projective> {
        let got = self.0.msm_g1(s, b);
        if let Some(g) = got {
            assert!(
                g == CpuReference.msm_g1(s, b).unwrap(),
                "the Metal MSM differs"
            );
        }
        got
    }
    fn fft(&self, a: &mut [Fq], w: Fq, log_n: u32) -> bool {
        self.0.fft(a, w, log_n)
    }
    fn ifft(&self, a: &mut [Fq], w: Fq, log_n: u32, d: Fq) -> bool {
        self.0.ifft(a, w, log_n, d)
    }
    fn quotient(&self, job: &QuotientJob<'_, Fq>, out: &mut [Fq]) -> bool {
        let served = self.0.quotient(job, out);
        let mut want = vec![Fq::ZERO; out.len()];
        assert!(CpuReference.quotient(job, &mut want));
        assert!(served, "the Metal backend declined a k >= 12 quotient");
        assert!(
            *out == *want,
            "the Metal quotient differs from the reference"
        );
        served
    }
}

#[test]
fn chain_proofs_verify_with_metal() {
    let k = 16;
    let (params, pk) = chain::setup(k);
    accel::install(Arc::new(Checked(MetalAccelerator::new().unwrap())));
    let before = accel::stats(Op::Quotient).0;
    let proof = chain::prove(&params, &pk, k);
    assert_eq!(accel::stats(Op::Quotient).0, before + 1, "quotient served");
    assert!(accel::stats(Op::Ifft).0 > 0, "inverse FFTs served");
    assert!(accel::stats(Op::Msm).0 > 0, "MSMs served");
    assert!(chain::verify(&params, pk.get_vk(), k, &proof));
    accel::uninstall();

    // The proof fails for a wrong instance (the check can fail).
    let mut bad = proof.clone();
    let last = bad.len() - 1;
    bad[last] ^= 1;
    assert!(!chain::verify(&params, pk.get_vk(), k, &bad));
}
