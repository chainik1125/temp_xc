# Updated response to Reviewer 4z15

_Drafted against Dmitry's `dmitry-txcwins-10h` commit `c7c301e2`._

## Paste-ready response

We thank the reviewer for identifying the central distinction between temporal
structure and generic crosscoder capacity. We agree that sparse probing alone
does not isolate a temporal contribution: MLC slightly leads there
(\(0.907\) mean AUC versus \(0.899\)--\(0.902\) for TXC-base), so we now
describe that result as evidence for aggregation generally, not as
TXC-specific superiority. We add three controls that target the distinction
more directly.

**1. Matched-capacity Stacked SAE.** Stacked SAE is a bank of independent
per-position SAEs. It has the same leading parameter count and dense inference
cost as TXC-base but cannot form a shared cross-position latent. For the
backtracking configurations:

| Architecture | Parameters | Dense GFLOPs / native forward |
|---|---:|---:|
| Per-token SAE / T-SAE, 1 token | 0.27B | 0.54 |
| TFA, 5 tokens | 2.32B | 27.18 |
| MLC, 5 layers | 1.34B | 2.68 |
| Stacked SAE / TXC-base, 5 tokens | 1.34B | 2.68 |
| TXC-Pro, 10 tokens | 2.68B | 5.37 |

Despite this match, Stacked SAE reaches 0.25 steering effect versus 0.54 for
TXC and, at the submitted \(S=8\) probe budget, 0.16 detection PR-AUC versus
0.24 for TXC. We will add these seed-42 values to Fig. 4 and Table 2. This
controls leading parameter count and dense compute at the paper's probe budget.

**2. A task with a formal temporal-information requirement.** We add an HMM
based on Shamir secret sharing. It encodes a temporal secret whose recovery is
provably bounded at chance, \(1/11\approx0.09\), until the probe has enough
positions. Single-token methods are given the same number of positions by
stacking their codes. Every baseline is swept over \(k\in\{1,2,5,10\}\),
with \(k=20\) also tested for T-SAE and TFA.

| Best recovery | \(W=3\) | \(W=5\) | \(W=10\) |
|---|---:|---:|---:|
| Best non-TXC baseline | 0.10 | 0.10 | 0.12 |
| TXC | 0.15 | 0.56 | 0.96 |

At \(W=1,2\), all architectures remain at chance as required by the
information-theoretic ceiling. Once \(W\geq3\), only TXC recovers the shared
temporal variable despite matched probe inputs and swept baseline sparsity.

**3. Window-size and order controls in a real model.** We added a matched
20K-step backtracking detection sweep over
\(T\in\{1,2,4,6,10\}\) with three independently trained dictionary seeds:

| \(T\) | Ordered TXC AP | Shuffled TXC AP | Order-invariant SAE AP |
|---:|---:|---:|---:|
| 1 | \(0.218\pm0.005\) | \(0.218\pm0.005\) | \(0.221\pm0.016\) |
| 2 | \(0.229\pm0.006\) | \(0.223\pm0.006\) | \(0.211\pm0.008\) |
| 4 | \(0.247\pm0.007\) | \(0.227\pm0.006\) | \(0.219\pm0.006\) |
| 6 | \(0.251\pm0.006\) | \(0.227\pm0.004\) | \(0.220\pm0.007\) |
| 10 | \(0.255\pm0.008\) | \(0.231\pm0.009\) | \(0.223\pm0.007\) |

Every seed's ordered \(T=10\) endpoint exceeds its \(T=1\) endpoint; the
paired increase is \(0.037\pm0.009\). At \(T=10\), the
ordered-minus-shuffled gap is \(0.023\pm0.007\). The larger window benefit
shows that local history helps, while the smaller permutation gap shows that
part of the gain depends on order. Because shuffling is a fixed-probe
test-time perturbation, we interpret it as representation sensitivity rather
than a causal estimate of unique temporal information.

These controls also delimit the claim. Longer windows do not improve static
sparse probing, MLC leads that benchmark, T-SAE leads Medical EM, and the
HH-RLHF metric has an untrained floor (\(0.62\)) above both trained Stacked SAE
(\(0.60\)) and TXC (\(0.61\)). We therefore claim a task-dependent temporal
advantage—strong on backtracking and on provably temporal synthetic
variables—not universal architectural dominance.

The parameter/FLOP table counts encoder-plus-decoder dense matmuls, with one
multiply-add as two FLOPs; training-only losses and sparse selection are
excluded. We add the corresponding per-task table to the appendix.

## Internal handoff notes

- Keep the response body below 5,000 characters.
- Remove the old one-seed percentage window table and duplicated T-SAE-width
  section from Dmitry's draft.
- Do not add sycgen unless space remains; its matched audit supports a
  high-\(T\) Pareto point, not learned order sensitivity.
