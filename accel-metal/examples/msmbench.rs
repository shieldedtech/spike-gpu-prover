// SPDX-FileCopyrightText: Copyright (C) Shielded Technologies
// SPDX-License-Identifier: Apache-2.0

//! Metal MSM vs CPU Pippenger at one size. cargo run --release --example
//! msmbench -- [log_n]
use std::time::Instant;

use ff::Field;
use group::{Curve, Group, prime::PrimeCurveAffine};
use midnight_accel_metal::MetalAccelerator;
use midnight_curves::{Fq, G1Affine, G1Projective};
use midnight_proofs::accel::{Accelerator, reference::CpuReference};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn main() {
    let log_n: u32 = std::env::args().nth(1).map_or(19, |a| a.parse().unwrap());
    let n = 1usize << log_n;
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let step = G1Projective::random(&mut rng);
    let mut p = G1Projective::random(&mut rng);
    let proj: Vec<G1Projective> = (0..n)
        .map(|_| {
            p += step;
            p
        })
        .collect();
    let mut b = vec![G1Affine::identity(); n];
    G1Projective::batch_normalize(&proj, &mut b);
    let s: Vec<Fq> = (0..n).map(|_| Fq::random(&mut rng)).collect();
    let gpu = MetalAccelerator::new().unwrap();
    let skew: Vec<Fq> = (0..n).map(|i| if i % 8 == 0 { s[i] } else { s[0] }).collect();
    for (name, s) in [("random", &s), ("random", &s), ("skewed", &skew)] {
        let t = Instant::now();
        let want = CpuReference.msm_g1(s, &b).unwrap();
        let cpu = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let got = gpu.msm_g1(s, &b).unwrap();
        let g = t.elapsed().as_secs_f64() * 1e3;
        assert!(got == want);
        println!(
            "{name} 2^{log_n}: cpu {cpu:.1} ms, metal {g:.1} ms, {:.2}x",
            cpu / g
        );
    }
    let _ = b[0].to_curve();
}
