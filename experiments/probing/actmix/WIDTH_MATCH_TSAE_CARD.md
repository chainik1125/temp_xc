# WIDTH_MATCH_TSAE_CARD — tsae_btkonly probing comparator at matched d_sae = 18432

**Agent:** runpod-b · **GPU:** pod A GPU 1 · **Frozen:** 2026-07-27 23:03 BST
(freeze commit = the commit adding this card + `width_match_tsae.py`; pin
asserted clean at launch). **Directive:** mac-local `98a9ea718` — re-run the
probing tsae comparator width-matched (paper width 16384 was wrong-width vs
the exhibit's comparator); supersedes the standby directive. RLHF-eq
seed-split first call STAYS ARMED (if runpod-2's gate fires TRAIN mid-lane:
finish in-flight training, then re-prioritize with mac-local).

## Task

3 trainings + 6 evals, identical to the P1-generation trained tsae cells in
every knob except **one delta**: `arch_hparams_override={"d_sae": 18432}`.
The override hashes into train_key + eval_key (documented no-collision
mechanism), so rows auto-distinguish from paper-width — no tag needed.

| knob | P1 trained cells (verbatim from rows) | this lane |
|---|---|---|
| arch | `tsae_btkonly` (registry `configs/archs.yaml`, class `btk_only:TSAEBTKOnly`, arch_version 2.1.0-port) | same |
| arch hparams | d_sae **16384**, k_pos 20, h_frac 0.2, contrastive_alpha 1.0, relu_mode btk-only | same, override **d_sae 18432** |
| realized matryoshka groups | n_high = round(0.2·16384) = 3277, n_low = 13107 | n_high = round(0.2·**18432**) = **3686**, n_low = **14746** (`tsae.py` n_high rule) |
| datasource | `gemma_2_2b_it_l13_fineweb_24k128` (data_key `48d2d17ff88598d4`) | same |
| training_cfg | n_steps 20000, batch_size 32, lr 3e-4, adam, warmup 1000, bf16, buffer_tokens 2,000,000, refill 0.5, bricken off, override None | same, override `{"d_sae": 18432}` |
| eval (×2 per training) | probing 1.2.0; k_feat {5, 20}, S 32, shuffle within_window, shuffle_seed 0, encode_batch_size 64, **arm btk-only**, smoke false | same |
| seeds | {42, 1, 2} (42-first dispatch, P1 convention) | same |

Serving note: tsae `consumes='sequence'` with internal consecutive-pair
sampling; batch_size 32 **sequences** is the P1 launch profile (launch
AMENDMENT 2's 7.8 s/step pathology was bs=4096 sequence batches; the b32
profile trains in minutes — see estimate).

## Cache provenance (disclosed)

P1 ran on pod B; this pod (A) had no cache. The activation + probe caches
exist pod-locally at the shared mirror `/workspace/caches/probing/hf_mirror/`
(the paper's actual v1 anchor cache, source
`hf://han1823123123/temp-bench-data/act_cache/e4916bcae1881963`; the v1-vs-v2
dataset-name divergence is already FLAGGED in the ACTMIX card). Wired into
this checkout exactly as runpod-a wired theirs: `acts.npy` symlink + copied
`meta.json` (data_key matches: `48d2d17ff88598d4`) under
`results/data_cache/…`, and a whole-dir symlink for
`results/probe_cache/gemma_2_2b_it_l13_fineweb_24k128`. Verified this
session: loader reports cache hit, acts shape (24000, 128, 2304) float16,
38 complete probe tasks (runner preflight re-asserts both).

## Reference bands (recomputed from leaderboard rows this session; mean ± sample std, n=3)

- k=20 trained, paper width: **0.87178 ± 0.0008** (s42 0.87095 / s1 0.87254 / s2 0.87183) — the exhibit's NOT-MET line-3 comparator band.
- k=5 trained, paper width: **0.8053 ± 0.0031** (s42 0.80692 / s1 0.80174 / s2 0.80715).
- realized_l0 watch: paper width sat 22.64–24.21, above the G1 band [19.5, 20.5] (known FAIL context, k_pos 20 unchanged). l0 at 18432 is a **watch metric** — BatchTopK budget is width-independent by construction, but realized_l0 may drift; report it, no gate.
- Exclusions: seed-0 row is smoke (n_steps 30); n_steps=0 rows are untrained twins — neither is in scope (directive = 3 trainings; no untrained twin at 18432 was directed).

## Pre-registered reading

This is a **measurement card, not a verdict card**: the deciding comparison
(width-confound line in the exhibit) belongs to mac-local. On landing I post
the rows table — per-seed mean_auc for k∈{5,20} + realized_l0 at 18432, with
Δ vs the paper-width bands above — as PTR; mac-local folds it into the
exhibit. No re-rolls; whatever the three seeds say is the answer.

## Estimate + ledger

P1 trained cells landed ~16–18 min apart per seed (rows 10:41 → 10:59 →
11:17 on 2026-07-27), i.e. ≈ 16 min train + ~4 min evals at d 16384.
Width-scale ×1.125 → ≈ 20–25 min per seed ⇒ **≈ 1–1.5 h wall, ≈ $3–6**
on GPU 1. (Corrects the $10–14 STATUS estimate, which extrapolated from the
λ̂-lane synthetic tsae profile — a different serving profile that does not
transfer.) Ledger line in `briefings/MODAL_SPEND.md` at launch; actuals
corrected on landing.

## Mechanics

Canonical pathway only (`run_experiment`, experiment="probing");
AGENT_NAME=runpod-b; CUDA_VISIBLE_DEVICES=1; TEMP_BENCH_ALLOW_DIRTY=1 under
the launch-pin convention (tree asserted clean == freeze pin at launch;
in-run dirt is leaderboard/manifest growth only). Runner:
`experiments/probing/actmix/width_match_tsae.py` (this commit). Rows
checkpoint-committed on landing as usual.
