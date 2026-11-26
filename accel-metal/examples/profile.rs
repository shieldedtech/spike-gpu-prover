// SPDX-FileCopyrightText: Copyright (C) Shielded Technologies
// SPDX-License-Identifier: Apache-2.0

//! Where the time of a chain proof goes: wall time inside each hook, CPU vs
//! Metal. cargo run --release --example profile -- [k]

use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use midnight_accel_metal::{MetalAccelerator, chain};
use midnight_curves::{Fq, G1Affine, G1Projective};
use midnight_proofs::accel::{self, Accelerator, QuotientJob, reference::CpuReference};

#[derive(Default, Debug, Clone, Copy)]
struct T {
    msm: (u32, f64),
    fft: (u32, f64),
    ifft: (u32, f64),
    quotient: (u32, f64),
}

struct Timed {
    metal: Option<MetalAccelerator>,
    t: Mutex<T>,
}

fn add(slot: &mut (u32, f64), s: f64) {
    slot.0 += 1;
    slot.1 += s;
}

impl Accelerator for Timed {
    fn msm_g1(&self, s: &[Fq], b: &[G1Affine]) -> Option<G1Projective> {
        let t = Instant::now();
        let r = match &self.metal {
            Some(m) => m.msm_g1(s, b).or_else(|| CpuReference.msm_g1(s, b)),
            None => CpuReference.msm_g1(s, b),
        };
        add(&mut self.t.lock().unwrap().msm, t.elapsed().as_secs_f64());
        r
    }
    fn fft(&self, a: &mut [Fq], w: Fq, k: u32) -> bool {
        let t = Instant::now();
        let r = match &self.metal {
            Some(m) => m.fft(a, w, k) || CpuReference.fft(a, w, k),
            None => CpuReference.fft(a, w, k),
        };
        add(&mut self.t.lock().unwrap().fft, t.elapsed().as_secs_f64());
        r
    }
    fn ifft(&self, a: &mut [Fq], w: Fq, k: u32, d: Fq) -> bool {
        let t = Instant::now();
        let r = match &self.metal {
            Some(m) => m.ifft(a, w, k, d) || CpuReference.ifft(a, w, k, d),
            None => CpuReference.ifft(a, w, k, d),
        };
        add(&mut self.t.lock().unwrap().ifft, t.elapsed().as_secs_f64());
        r
    }
    fn quotient(&self, job: &QuotientJob<'_, Fq>, out: &mut [Fq]) -> bool {
        let t = Instant::now();
        let r = match &self.metal {
            Some(m) => m.quotient(job, out),
            None => false,
        };
        add(
            &mut self.t.lock().unwrap().quotient,
            t.elapsed().as_secs_f64(),
        );
        r
    }
}

fn main() {
    let k: u32 = std::env::args().nth(1).map_or(19, |a| a.parse().unwrap());
    let (params, pk) = chain::setup(k);
    for (name, metal) in [
        ("cpu", None),
        ("metal", MetalAccelerator::new()),
        ("cpu", None),
        ("metal", MetalAccelerator::new()),
    ] {
        let timed = Arc::new(Timed {
            metal,
            t: Mutex::default(),
        });
        accel::install(timed.clone());
        chain::prove(&params, &pk, k); // warm
        *timed.t.lock().unwrap() = T::default();
        let t = Instant::now();
        let proof = chain::prove(&params, &pk, k);
        let total = t.elapsed().as_secs_f64();
        accel::uninstall();
        assert!(chain::verify(&params, pk.get_vk(), k, &proof));
        let x = *timed.t.lock().unwrap();
        let rest = total - x.msm.1 - x.fft.1 - x.ifft.1 - x.quotient.1;
        println!(
            "{name:5} k={k} total {total:.3}s | msm {} calls {:.3}s | fft {} {:.3}s | ifft {} {:.3}s | quotient hook {} {:.3}s | rest {rest:.3}s",
            x.msm.0, x.msm.1, x.fft.0, x.fft.1, x.ifft.0, x.ifft.1, x.quotient.0, x.quotient.1
        );
    }
}
