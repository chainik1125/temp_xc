# txc_pro RECOVERY DIG — provenance report

**Author: `mac-c`. Source: `briefings/safety-task-research.md` secondary
bounded item. Consumer: `runpod-c` (T-scaling hill-climb pod), which was
briefed to "reimplement the txc_pro recipe FROM ITS LOCKED HPARAMS"
(LOG 2026-07-27 18:34).**

## 0. Headline — you do not have to reimplement it

**The implementation SURVIVED.** The LOG's first-pass reading (18:45
entry, § 5: *"the class file `txc_pro.py` did NOT survive purification —
only the registry pointer"*) is true of the **working tree** and false of
**git history**. The full 496-line class is one command away:

```bash
git show 5dd7337b2^:purified/src/temp_bench/archs/txc_pro.py
```

A verbatim copy, with a provenance header, is committed at
**`docs/recovered/txc_pro_phase5b_subseq_h8.py`** (under `docs/`
deliberately — `src/` is importable library code only, and re-registering
an arch is not mac-c's call).

| field | value |
|---|---|
| blob sha | `480f3755dd3cc08f05cbd7019e44e3025bd6603d` |
| sha256 (file bytes) | `626066a8307a6e6e7c3004e9eb56f58aaa078297093bcf44cfbb5c4d644e16ae` |
| last path | `purified/src/temp_bench/archs/txc_pro.py` |
| removal commit | `5dd7337b2` "arxiv: remove txc_pro from active registry" (Han, 2026-05-31 23:28 +0100) |
| lines | 496 |
| `arch_version` in file | **2.0.0**, `consumes: 'sequence'` |

**It was already ported to framework v2 before deletion** — this is not a
v1 relic needing a rewrite. It was removed as a *scope* decision
("paper-only scope"), not because it was broken.

Earlier ancestry, if a diff is ever wanted:
`6ae94a743` (original port: "Agent NLP: port txc_pro — subseq +
matryoshka + multi-distance contrastive"), `48b944c2f` (multi-window
toggle), `ea3623cc4` (C2 synthetic), `1a2583753` (anonymisation),
`1c213513f` (purified/ flatten). Upstream attribution in the file's own
docstring: `origin/han-phase7-unification @ 94119bc0:
src/architectures/phase5b_subseq_sampling_txcdr.py::SubseqH8`, which the
port flattens out of a 3-level inheritance chain
(`TXCBareMultiDistanceContrastiveAntidead` →
`TXCBareMatryoshkaContrastiveAntidead` → `TXCBareAntidead`).

## 1. What it is

`TXCPro(TempBenchArch)` — four components over a per-position encoder:

1. **Subseq encoder.** `W_enc` is `(T_max, d_in, d_sae)` — one weight
   slab per position. At **training** a contiguous `t_sample`-length
   subset of positions is drawn per row and the pre-activation sums
   encoder contributions **only over that subset**. At **inference** the
   full `T_max` window is used, no sampling. (This is Dmitry's
   remembered "resampled the window size".)
2. **Matryoshka H+full reconstruction.** MSE from the full dictionary
   *plus* MSE from the first `h_size` features; total recon = `L_H +
   L_full`.
3. **Multi-distance temporal InfoNCE.** Anchor window plus positives at
   `shifts = (1, 2)`; per-shift weight `w_s = 1/(1+s)`
   (inverse-distance), symmetric InfoNCE on the first `contr_prefix`
   latents.
4. **Anti-dead stack** (shared with `txc_base`): decoder unit-norm per
   atom over `(T_max, d_in)`, decoder-parallel gradient projection via a
   `post_accumulate_grad_hook`, AuxK on dead features
   (`auxk_alpha = 1/32`), `num_tokens_since_fired` tracker, optional
   geometric-median `b_dec` init (Weiszfeld).

`train_step` takes `(B, seq_len, d_in)` and derives its own
anchor/positive windows, so the canonical `batch_iter` stays
arch-agnostic; it requires `seq_len ≥ T_max + max(shifts)`.

## 2. Hparam corrections to the LOG's first-pass list

The 18:45 entry recovered: *d_sae 18432, T_max 10, t_sample 5,
n_matryoshka 8, contrastive_shifts [1,2] + inverse-distance weighting,
auxk_alpha 0.03125*. Two corrections and one addition, all from the
recovered source plus `origin/final-aniket:purified/configs/locked_archs.yaml`:

- **`n_matryoshka: 8` is NOT a functional hparam.** The source marks it
  `# noqa: ARG002 — phase id, not used` and the docstring says so
  outright: *"**NOT** functionally used as a count of matryoshka levels —
  we use the wasteland's H+full layout (`h_size = d_sae // 5`)."* The
  real matryoshka control is **`h_size`, defaulting to `d_sae // 5`**
  (= 3686 at d_sae 18432). Reimplementing "8 matryoshka levels" from the
  yaml name would build a **different architecture**.
- **`k_pos: 20` was omitted** from the recovered list and it is in the
  locked yaml. It is load-bearing (§ 3).
- **`arch_version` disagrees between sources**: locked yaml says
  `"1.0.0"`, the last-committed file says `2.0.0`. The file is the later
  artifact (v2 migration); the yaml is the older locked spec.

Full locked block (`origin/final-aniket:purified/configs/locked_archs.yaml`,
registry tag `phase5b_subseq_h8`): `d_sae 18432`, `T_max 10`,
`t_sample 5`, `k_pos 20`, `n_matryoshka 8`, `contrastive_shifts [1,2]`,
`contrastive_inverse_distance_weight true`, `auxk_alpha 0.03125`,
`dead_threshold_tokens 10_000_000`, `bdec_geom_median_init true`,
`decoder_unit_norm true`, `decoder_grad_orthogonalize true`. The last two
are accepted-and-ignored in the port (always True) — parity kwargs.

## 3. Sparsity, and what a T-sweep on this arch actually does

`__init__` sets two budgets, and they are **not the same**:

```python
self.k_train     = min(k_pos * t_sample, d_sae)   # 20 *  5 = 100
self.k_inference = min(k_pos * T_max,    d_sae)   # 20 * 10 = 200
```

- **Convention-compliant.** `k_inference = k_pos · T` is exactly the
  program-wide rule re-audited at LOG 18:22 (*"constant PER-TOKEN
  budget, window budget linear in T"*). No conflict, no corrective
  action — recording it so `runpod-c` does not re-derive it.
- **But training and inference run at different window budgets by
  design** (100 vs 200 at locked hparams), because training sees
  `t_sample` positions and inference sees `T_max`. **Consequence for a
  T-scaling hill-climb: sweeping `T_max` at fixed `t_sample` holds the
  TRAIN budget constant while scaling the INFERENCE budget linearly, so
  the train/inference asymmetry widens with T.** Whether to hold
  `t_sample/T_max` ratio fixed instead is a design choice the sweep
  should make explicitly and pre-register, not inherit by accident.
- `encode()` **hard-raises** unless `x.shape[1] == T_max` — there is no
  partial-window inference path. A T-sweep means retraining per T, not
  re-evaluating one checkpoint at several T.
- `multi_window` (default `False`) is an opt-in training-FLOPs-parity
  toggle; the source notes flipping it **invalidates train_keys** (it
  hashes into `compute_train_key`).

## 4. T-scaling evidence — A12-aware audit

**Verdict: there is NO real probing T-scaling evidence for `txc_pro`
anywhere. It has one shipped probing point at one T.**

What exists, exhaustively:

| source | what it says about txc_pro | status |
|---|---|---|
| shipped main-text figure `c3_sparse_probing_auc_of_auc_gemma_it.png` (final-aniket) | **one bar, 0.931**, alongside T-SAE 0.931 | single T, real cell |
| `main.tex` text | "T-SAE/TXC-pro 0.897–0.899" | same cell under the **CT-included** aggregation; the ≈ +0.03 offset to the figure is the CT-exclusion shift (`COMPOSITION_AUDIT` § A12) |
| canonical v2 `results/leaderboard.jsonl` | **31 rows, ALL `experiment: synthetic`**, `arch_version 2.0.0`, 3 seeds, `arch_hparams_override {"k_pos": 1}` (toy), primary metric `gauc` | no probing rows, **no T variation** |
| the A12 phantom T-sweep | **irrelevant to txc_pro** — the silent-T5 replicas are `txc_base` "T=10"/"T=20" | see below |

**The A12 trap does not contaminate txc_pro, and that is the good news.**
Per `COMPOSITION_AUDIT` § A12 (my own audit, confirmed in git): the
phantom "T-sweep" bars in the shipped headline figure are `txc_base`
T=10/T=20 cells that were silently trained at T=5 (bug fixed at
`origin/final:1ed4fde5f`), so their 0.932 → 0.933 → 0.935 "T improves
probing" ordering is replica noise. **`txc_pro` was never part of that
sweep** — it is a single honest bar. So a revival inherits **no
contaminated T prior to unlearn**, and equally **no evidence that this
recipe scales with T**. Anyone claiming "txc_pro was our T-scaling
architecture" is describing an intention, not a measurement.

Corollary for `runpod-c`: the T-scaling hill-climb starts from zero on
this arch. That is a clean start, not a loss.

## 5. Revival gotchas (operational)

The removal commit `5dd7337b2` did more than delete a file. Anything
reviving `txc_pro` must revisit **all** of these or historical rows stay
invisible and new ones get filtered out:

- `configs/archs.yaml` — the `txc_pro:` block was removed (today
  `src/temp_bench/archs/` holds 14 arch modules and `txc_pro.py` is not
  among them).
- `configs/experiments.yaml` — dropped from the `synthetic`, `probing`,
  `backtracking` and `rlhf` arch lists.
- `experiments/render_paper_figures.py` — colour + marker dropped, and a
  **`deprecated_archs` filter added**.
- `scripts/populate_repro_report_{from_leaderboard,multiseed}.py` —
  **`DEPRECATED_ARCHS = {"txc_pro"}`** filter.
- `scripts/run_synthetic_minisweep.sh` — ARCHS list cut to 5.
- `scripts/run_retry_failed.sh` — deleted (was a one-shot retry scaffold
  for the txc_pro + TFA clip-fix passes).

Historical leaderboard rows were **kept for audit trail** and suppressed
at render/populate time — so the 31 synthetic rows are still there and
still filtered. Un-deprecating is a registry decision for Han/mac-local,
not something this dig performs.

## 6. Scope statement

This dig **recovered and documented**; it changed no registry, no config,
no core code, and ran no compute. The one file added under `src/`-adjacent
paths is `docs/recovered/…` which nothing imports. Whether `txc_pro`
returns to the active registry, and under whose `arch_version`, is Han's
and mac-local's call.

_Recorded-by: claude-fable-5 (mac-c)_
