# July 29 Backtracking closeout

**Status:** The corrected 300K TXC-base training and detection replication is
complete for seeds 1, 2, and 42, and the new 16K T-SAE seed-42 width
sensitivity is complete. Multi-seed steering and a corrected TXC-pro detection
replication remain unrun.

## Completed 300K detection result

All four dictionaries were evaluated on the exact submitted sentence artifact
using five GroupKFold splits and
\(S\in\{1,2,4,8,16,32\}\). The source artifact, training checkpoints, detector
outputs, and compact transfer passed their locked hash and provenance checks.

| Probe budget | TXC-base 32K, seeds 1/2/42 | T-SAE 16K, seed 42 | Submitted T-SAE 32K, seed 42 |
|--:|--:|--:|--:|
| \(S=8\) | \(0.1874 \pm 0.0080\) | \(0.2043\) | \(0.196\) |
| \(S=16\) | \(0.2314 \pm 0.0072\) | \(0.2130\) | \(0.213\) |
| \(S=32\) | \(0.2568 \pm 0.0048\) | \(0.2285\) | \(0.245\) |

At the submitted \(S=8\) budget, corrected TXC-base remains above the
submitted SAE reference \(0.175\), but it is below both T-SAE references. The
ranking reverses at \(S\geq16\), where TXC-base is strongest. Detection claims
must therefore name the sparse-probe budget rather than presenting one
budget-independent architecture ranking.

The submitted detection winner was TXC-pro, whereas this corrected replication
is TXC-base, the submitted steering winner. These results close the requested
TXC-base dictionary and detection reruns but do not supply new TXC-pro or
multi-seed steering evidence. The compact results and reviewer figures live in
`purified/results/neurips_rebuttal/backtracking_300k_seeded/`.

## Completed training specification

The replication trained **300K TXC-base dictionary seeds 1, 2, and 42**,
matching the submitted C7 cell except for the documented RNG correction:

- historical source: `origin/extended-300k`
- architecture: paper-faithful `txc_base`
- model/cache: Llama-3.1-8B, layer 10, Ward/Nous-mirror activations
- window: \(T=5\)
- dictionary width: \(32{,}768\)
- sparsity: \(k_{\mathrm{pos}}=20\)
- batch size: 1,024
- training steps: 300,000
- submitted seed-42 training key: `8787f8fe527218ad`

Use the historical locked configuration rather than a current default, because
the current code has accumulated incompatible TopK/BatchTopK variants.

## July 29 execution protocol

An audit found that the historical C7 runner passed the nominal seed into its
NumPy window sampler but never seeded Python or PyTorch before model
initialization and `torch.randint`. Its evaluation adapter also silently
defaulted to seed 42 because `run_cell` did not thread the owning cell seed
into `eval_cfg`. The submitted aggregate seed-42 row remains the paper
reference, but it is not a sufficient deterministic member of a new
three-seed replication.

The clean replication therefore trains corrected seeds 1, 2, and 42 under
protocol `c7-300k-seeded-v1`, with all Python, NumPy, CPU-Torch, and CUDA RNGs
seeded before initialization. The runner:

- archives historical source commit
  `284a8bf5e3e5a7cc094dd68c6fa5a92a9fd4eec3` without switching branches;
- verifies activation-cache key `fb2a74be884e512a`, SHA-256
  `dc34dfb117f77abddef4b4396d0d00afc707c39876d0ee36015de1e7b8406914`,
  shape `(4044, 128, 4096)`, and `float16` dtype;
- asserts the exact historical training keys
  `a300c63374c3597e`, `27078b0d7700ae05`, and `8787f8fe527218ad`;
- keeps the cache on the pinned worker GPU and uses a vectorized gather that
  was checked bit-for-bit against the historical row-copy loop;
- writes only one final checkpoint per cell plus compact progress and
  provenance JSON, so storage is bounded.

The reproducible launcher is in
`purified/experiments/backtracking_300k_seeded/`. The three TXC cells run on
three H100s; the new T-SAE-16K seed-42 sensitivity control runs on one A40.
The submitted T-SAE-32K cell is not duplicated.

The historical trainer has no exact optimizer/RNG resume format. These tmux
jobs survive a disconnected laptop, but terminating a pod before its final
checkpoint would restart that cell from step zero; adding an unverified
partial-resume format during the closeout would create a larger reproducibility
risk than the bounded preemption risk.

## Evaluation status for each TXC seed

1. **Detection — complete:** the paper's grouped Backtracking detector was run
   at \(S\in\{1,2,4,8,16,32\}\) for all three corrected seeds.
2. **Steering — pending:** run the paper-faithful C7 steering evaluation with
   `cut_fraction=0.25` and magnitudes
   \[
   -16,-12,-10,-8,-7,-6,-5,-4,-3,-2,-1,-0.5,0,0.5,1,2,3,4,5,6,7,8,10,12,16.
   \]
   Preserve the same genuine-backtracking judge protocol. The seed-42
   reference peaks near \(\Delta gc=0.541\) at magnitude \(-12\).

Report seeds 1, 2, and 42 individually and as mean \(\pm\) sample SD. State
whether the direction and architecture ranking survive. Training checkpoints
alone do not fill the response gap; the steering judge sweep is the result
Dmitry needs.

## T-SAE width request

The submitted C7 T-SAE result is already the seed-42, 300K-step,
\(d_{\mathrm{SAE}}=32{,}768\) matched-width cell:

- training key: `32f27809cdf34da9`
- \(\mathrm{PR\mbox{-}AUC}@S=32=0.244815\)

Do not spend compute duplicating that cell unless an artifact or protocol
check fails. The \(16{,}384\) value was the generic T-SAE/Gemma default; C7
overrode it to \(32{,}768\). If Dmitry still wants a seed-42 16K run, label it
as a new width-sensitivity control rather than evidence that the submitted
Backtracking T-SAE was underpowered.

## What the 20K sweep can support

The completed 20K experiment at
\(T\in\{1,2,4,6,10\}\), seeds \(1,2,42\), supports a separate
multi-seed **detection** and shuffle-sensitivity result. It cannot be pooled
with the submitted 300K steering result or used to call steering a three-seed
experiment.

If the 300K cells cannot be finished, the fallback is to report the 20K
three-seed detection result and remove any claim that steering has been
replicated across seeds. Do not average the 20K and 300K cells.

## Reviewer-response audit

- State explicitly that TXC-Pro supplied the submitted strongest detection
  result while TXC-base supplied the strongest steering result.
- Do not say that all headline results use three seeds.
- State that the submitted Backtracking T-SAE already used width 32,768.
- Review Han's proposed sycgen paragraph before adding it. Its high-window
  recovery/Pareto result is usable, but its shuffle gap is not evidence of
  learned temporal ordering: random-initialized twins have an equal or larger
  gap in 11 of 12 cells. Any excerpt must disclose that limitation rather than
  omit the control.
- Dmitry did not add the new tasks to the submitted comments. Treat them as
  optional amendments, not prerequisites for the Backtracking closeout.
