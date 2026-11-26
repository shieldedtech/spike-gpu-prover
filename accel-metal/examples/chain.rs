// SPDX-FileCopyrightText: Copyright (C) Shielded Technologies
// SPDX-License-Identifier: Apache-2.0

//! CPU vs Metal on the chain circuit of #667's rc1 benchmark.
//!
//! cargo run --release --example chain -- [k_min] [k_max] [warm]
//!
//! Per k: one cold and `warm` warm proofs per mode, CPU and Metal
//! interleaved, every proof checked by the stock verifier.

use std::{sync::Arc, time::Instant};

use midnight_accel_metal::{MetalAccelerator, chain};
use midnight_proofs::accel::{self, Op};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let args: Vec<u32> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let (k_min, k_max, warm) = (
        *args.first().unwrap_or(&14),
        *args.get(1).unwrap_or(&20),
        *args.get(2).unwrap_or(&3) as usize,
    );
    let t = Instant::now();
    let metal = Arc::new(MetalAccelerator::new().expect("a Metal device"));
    eprintln!(
        "metal init {:.3}s, threads {}",
        t.elapsed().as_secs_f64(),
        rayon::current_num_threads()
    );
    println!("k\tcpu_warm_s\tmetal_warm_s\tspeedup\tcpu_cold_s\tmetal_cold_s\tquotient_gpu_ms");
    for k in k_min..=k_max {
        let (params, pk) = chain::setup(k);
        let vk = pk.get_vk();
        let run = |on: bool| -> f64 {
            if on {
                accel::install(metal.clone());
            } else {
                accel::uninstall();
            }
            let served = accel::stats(Op::Quotient).0;
            let t = Instant::now();
            let proof = chain::prove(&params, &pk, k);
            let s = t.elapsed().as_secs_f64();
            if on {
                assert_eq!(
                    accel::stats(Op::Quotient).0,
                    served + 1,
                    "quotient not served"
                );
            }
            assert!(
                chain::verify(&params, vk, k, &proof),
                "proof does not verify"
            );
            accel::uninstall();
            s
        };
        let (cpu_cold, metal_cold) = (run(false), run(true));
        let (mut cpu, mut gpu) = (vec![], vec![]);
        for i in 0..warm {
            // Alternate which mode goes first.
            if i % 2 == 0 {
                cpu.push(run(false));
                gpu.push(run(true));
            } else {
                gpu.push(run(true));
                cpu.push(run(false));
            }
        }
        let (c, g) = (median(cpu), median(gpu));
        println!(
            "{k}\t{c:.3}\t{g:.3}\t{:.2}x\t{cpu_cold:.3}\t{metal_cold:.3}\t{:.1}",
            c / g,
            metal.stats().last_quotient_gpu_us as f64 / 1e3
        );
    }
}
