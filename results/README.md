# Results

## 2026-11-23, Apple M5 Max (`2026-11-23-m5max.csv`)

**Workload.** 31 circuits compiled from real Compact contracts, k = 6 to 20,
each proved with its own preimage. Circuit names are replaced by `C01` to
`C31`, in order of k.

**Configurations.**

| `config` | `cpu_s` | `metal_s` |
|---|---|---|
| `bob` | stock midnight-proofs CPU prover | `accel-metal` |
| `seungheon` | SeungHeon's build of the CPU prover, with his `prover-par` changes | his own build of `midnight-metal` with his hooks |
| `combined` | not measured | `midnight-metal` behind the `accel` hooks (`seungheon-adapter`) |

**Method.** Each row is the median of 3 warm proofs after 1 warm-up, in one
process. Each circuit ran 3 rounds, with the order of the configurations
rotated every round. The machine was otherwise idle. Every proof was checked
by the stock verifier. The median spread between rounds was 3.7%.

**Summary** (median per k; speed-ups against the stock CPU `cpu_s` of `bob`):

| k | circuits | stock CPU | Bob | SeungHeon | Combined | SeungHeon vs Bob |
|---|---:|---:|---:|---:|---:|---:|
| 6 | 2 | 14 ms | 0.96x | 1.17x | 0.96x | 1.22x |
| 13 | 11 | 104 ms | 1.38x | 1.27x | 1.14x | 0.93x |
| 14 | 9 | 188 ms | 1.71x | 1.52x | 1.36x | 0.89x |
| 15 | 2 | 351 ms | 1.93x | 1.96x | 1.71x | 1.02x |
| 16 | 1 | 746 ms | 2.15x | 2.66x | 2.26x | 1.24x |
| 17 | 3 | 1.46 s | 2.54x | 2.88x | 2.48x | 1.13x |
| 18 | 2 | 2.95 s | 2.14x | 2.85x | 2.55x | 1.34x |
| 20 | 1 | 10.81 s | 2.39x | 2.98x | 2.68x | 1.25x |

**Reading it.**

- SeungHeon's engine is faster from k = 16 up. `accel-metal` is faster at
  k = 13 and 14, where batching the commitments of same-size polynomials
  into one MSM pays off.
- `combined` runs at 83 to 90% of SeungHeon's own build. His `prover-par`
  CPU changes account for most of the gap: alone, they make the CPU prover
  4 to 17% faster.
- Rows for k = 15, 16, 18 and 20 rest on one or two circuits each.
