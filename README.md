# spike-gpu-prover

GPU proving for [midnight-proofs](https://github.com/midnightntwrk/midnight-zk),
kept outside midnight-zk. midnight-zk carries only a small hook layer (the
`Accelerator` trait behind the `accel` feature: MSM, batched MSM, FFT,
inverse FFT, the quotient, and a per-proof begin and end signal). The
backends live here, or in their authors' own repos.

This is a prototype for measurement and discussion, not for production use.

## Contents

| Path | What it is |
|---|---|
| `accel-metal/` | An Apple Metal backend: batched Pippenger MSM over shared bases, NTT and the quotient on the GPU, with a GPU/CPU split for MSMs. |
| `seungheon-adapter/` | An adapter that runs SeungHeon Oh's Metal backend ([`midnight-metal`](https://github.com/SeungheonOh/midnight-zk/tree/metal-prover/metal)) unchanged behind the same hooks. |
| `results/` | Benchmarks on real Compact circuits (anonymised), with the method. |

The hook layer and the GPU prover design come from Charles Hoskinson's
[midnight-zk #667](https://github.com/midnightntwrk/midnight-zk/pull/667)
(CUDA backend) and SeungHeon Oh's Metal backend for it.

## Contract

A backend either returns exactly the value the stock CPU code would, or
declines, and the stock code runs. The transcript, the blinding and every
value the verifier sees come from the stock code. Each crate's tests check
its results against the CPU reference (`midnight_proofs::accel::reference`)
and check that proofs made with it verify and that tampered proofs do not.

## Building

Until the hook layer is in a midnight-proofs release, both crates build
against the hooks PR,
[midnight-zk #696](https://github.com/midnightntwrk/midnight-zk/pull/696),
pinned to a commit in `Cargo.toml`:

```sh
git clone https://github.com/shieldedtech/spike-gpu-prover
cd spike-gpu-prover
cargo test --release --workspace   # needs a Mac with Apple silicon
```

To build against a local midnight-zk checkout instead, override the patches,
for example with `--config 'patch.crates-io.midnight-proofs.path="../midnight-zk/proofs"'`.

`midnight-metal` is fetched from SeungHeon's repository at a pinned commit.

## Results (M5 Max, 31 Compact circuits)

Speed-up over the stock CPU prover on the same machine (median per k):

| k | circuits | Bob (`accel-metal`) | SeungHeon (own build) | Combined (`seungheon-adapter`) |
|---|---:|---:|---:|---:|
| 13 | 11 | 1.38x | 1.27x | 1.14x |
| 14 | 9 | 1.71x | 1.52x | 1.36x |
| 15 | 2 | 1.93x | 1.96x | 1.71x |
| 16 | 1 | 2.15x | 2.66x | 2.26x |
| 17 | 3 | 2.54x | 2.88x | 2.48x |
| 18 | 2 | 2.14x | 2.85x | 2.55x |
| 20 | 1 | 2.39x | 2.98x | 2.68x |

See [`results/`](results/) for the data, the method and the caveats.

## License

Apache-2.0.
