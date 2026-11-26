// SPDX-FileCopyrightText: Copyright (C) Shielded Technologies
// SPDX-License-Identifier: Apache-2.0

//! The chain circuit of the #667 `rc1` benchmark, for tests and benchmarks.
//!
//! Starting at `x[0] = 7`, `x[i+1] = x[i]^2 + 3`. Each selected row holds four
//! consecutive values in four advice columns; one selector enables four
//! quadratic constraints, the fourth reading the first column of the next row.
//! Copy constraints bind the start and the endpoint to the public instance.

use ff::Field;
use midnight_curves::{Bls12, Fq};
use midnight_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Constraints, Error, Instance, ProvingKey,
        Selector, VerifyingKey, create_proof, keygen_pk, keygen_vk_with_k, prepare,
    },
    poly::{
        Rotation,
        commitment::Guard,
        kzg::{KZGCommitmentScheme, params::ParamsKZG},
    },
    transcript::{CircuitTranscript, Transcript},
};
use rand_core::OsRng;

/// The commitment scheme.
pub type Scheme = KZGCommitmentScheme<Bls12>;

/// Selected rows at `2^k` rows (room is left for the blinding rows).
pub fn steps(k: u32) -> usize {
    (1usize << k) - 20
}

fn step(x: Fq) -> Fq {
    x.square() + Fq::from(3)
}

/// The public instance: the start and the endpoint after `4 * steps(k)`
/// steps.
pub fn instance(k: u32) -> [Fq; 2] {
    let mut x = Fq::from(7);
    for _ in 0..4 * steps(k) {
        x = step(x);
    }
    [Fq::from(7), x]
}

/// The columns.
#[derive(Clone, Debug)]
pub struct ChainConfig {
    a: [Column<Advice>; 4],
    s: Selector,
    inst: Column<Instance>,
}

/// The chain circuit at `2^k` rows.
#[derive(Clone, Debug, Default)]
pub struct Chain {
    /// log2 of the rows.
    pub k: u32,
}

impl Circuit<Fq> for Chain {
    type Config = ChainConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<Fq>) -> ChainConfig {
        let a = [(); 4].map(|_| meta.advice_column());
        let s = meta.selector();
        let inst = meta.instance_column();
        meta.enable_equality(a[0]);
        meta.enable_equality(inst);
        meta.create_gate("chain", |vc| {
            let x: Vec<_> = a.iter().map(|c| vc.query_advice(*c, Rotation::cur())).collect();
            let next = vc.query_advice(a[0], Rotation::next());
            let three = midnight_proofs::plonk::Expression::Constant(Fq::from(3));
            let sq = |e: &midnight_proofs::plonk::Expression<Fq>| e.clone() * e.clone();
            Constraints::with_selector(
                s,
                vec![
                    x[1].clone() - sq(&x[0]) - three.clone(),
                    x[2].clone() - sq(&x[1]) - three.clone(),
                    x[3].clone() - sq(&x[2]) - three.clone(),
                    next - sq(&x[3]) - three,
                ],
            )
        });
        ChainConfig { a, s, inst }
    }

    fn synthesize(&self, cfg: ChainConfig, mut layouter: impl Layouter<Fq>) -> Result<(), Error> {
        let (start, end) = layouter.assign_region(
            || "chain",
            |mut region| {
                let mut x = Fq::from(7);
                let mut start = None;
                for row in 0..steps(self.k) {
                    cfg.s.enable(&mut region, row)?;
                    for (j, col) in cfg.a.iter().enumerate() {
                        let c = region.assign_advice(|| "x", *col, row, || Value::known(x))?;
                        if row == 0 && j == 0 {
                            start = Some(c.cell());
                        }
                        x = step(x);
                    }
                }
                let end =
                    region.assign_advice(|| "end", cfg.a[0], steps(self.k), || Value::known(x))?;
                Ok((start.unwrap(), end.cell()))
            },
        )?;
        layouter.constrain_instance(start, cfg.inst, 0)?;
        layouter.constrain_instance(end, cfg.inst, 1)
    }
}

/// Test SRS (insecure, seeded) and keys at `2^k` rows.
pub fn setup(k: u32) -> (ParamsKZG<Bls12>, ProvingKey<Fq, Scheme>) {
    use rand_core::SeedableRng;
    let rng = rand_chacha::ChaCha20Rng::seed_from_u64(0x524331);
    let params = ParamsKZG::<Bls12>::unsafe_setup(k, rng);
    let vk = keygen_vk_with_k::<Fq, Scheme, _>(&params, &Chain { k }, k).unwrap();
    let pk = keygen_pk(vk, &Chain { k }).unwrap();
    (params, pk)
}

/// A proof with OS randomness.
pub fn prove(params: &ParamsKZG<Bls12>, pk: &ProvingKey<Fq, Scheme>, k: u32) -> Vec<u8> {
    let inst = instance(k);
    let mut transcript = CircuitTranscript::<blake2b_simd::State>::init();
    create_proof::<Fq, Scheme, _, _>(
        params,
        pk,
        &Chain { k },
        0,
        &[&inst],
        &mut transcript,
        OsRng,
    )
    .unwrap();
    transcript.finalize()
}

/// Whether `proof` verifies with the stock verifier.
pub fn verify(
    params: &ParamsKZG<Bls12>,
    vk: &VerifyingKey<Fq, Scheme>,
    k: u32,
    proof: &[u8],
) -> bool {
    let inst = instance(k);
    let mut transcript = CircuitTranscript::<blake2b_simd::State>::init_from_bytes(proof);
    prepare::<Fq, Scheme, _>(vk, &[], &[&inst], &mut transcript)
        .is_ok_and(|g| g.verify(&params.verifier_params()).is_ok())
}
