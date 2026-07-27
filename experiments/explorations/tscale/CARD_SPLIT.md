# TSCALE CARD 0 — dev/holdout split + program pre-registration (FROZEN)

**agent:** runpod-c (dedicated 2×H100 pod, both GPUs) · **created:**
2026-07-27 18:53 London (date-verified) · **status: FROZEN — the freeze
commit is the commit that lands this card on `origin/arxiv`; every number
in § 2–3 derives from already-committed baseline rows, no new compute.**

Mission (Dmitry 07-27 meeting, Han directive, briefing
`agents/runpod-c/STATUS.md`): make TXC T-scaling actually improve with
window size on § 5.1 sparse probing. Baseline truth (P1 btk-only grid,
36-task CT-excl, 3 seeds): TXC-pre k=20 DECLINES 0.9264 → 0.9033
(T1→T16); k=5 recovers only to the SAE band (ties, +0.001). This card
pre-registers the iteration/validation split and the candidate-1
design choices BEFORE any candidate trains. This is ARCH R&D: nothing
from this exploration enters any claim surface without L3 + a proper
card + mac-local ratification.

## 1. The split (pre-registered, binding)

**DEV (iteration signal) — 8 of the 36 CT-excl tasks, seed 42,
T ∈ {1, 4, 16}, k_feat = 20:**

```
ag_news_world
amazon_reviews_cat1
amazon_reviews_cat2
bias_in_bios_set2_prof11
bias_in_bios_set3_prof21
bias_in_bios_set3_prof26
europarl_en
github_code_Java
```

**HOLDOUT — the remaining 28 CT-excl tasks × seeds {1, 2, 42} × the
full T grid {1, 2, 4, 8, 16} × k_feat {5, 20}: touched ONLY by L3
finalist validation runs.** The honesty of the whole exercise rests on
never climbing on the holdout. (The two CT tasks stay excluded from
both sides, matching the headline convention; 38-task raw is still
reported at L3 for continuity.)

**Selection procedure (reproducible, `make_split.py` asserts it):**
family-stratified seeded draw — quotas proportional to family size
(ag_news 1, amazon_reviews 2, bias_in_bios 3, europarl 1,
github_code 1 = 8 of 36), `numpy.random.default_rng(20260727)`,
within alphabetically-sorted family lists. **Power rule (set before
the first draw):** accept a draw iff the dev-8 baseline decline
Δ16 = mean(T16) − mean(T1) for `txc_batchtopk_pre_btkonly` s42 k20 is
≤ −0.010 (at least ~40 % of the full-36 decline −0.026); else redraw
at seed+1 (all draws recorded). **Draw 1 accepted; no redraws.**

**What iteration may read:** dev-task columns at seed 42, T {1,4,16},
k 20 (plus any L0 training-health metric), and any already-committed
baseline row (they are public program state). **What iteration may
NOT read:** holdout-task columns / extra seeds / off-grid T of any
CANDIDATE run before that candidate is declared an L3 finalist in
RESULTS.md. Scratch dev evals compute the 8 dev tasks only, so the
holdout is structurally out of reach at L1/L2.

## 2. Baseline reference numbers on the dev-8 (from committed P1 rows)

k = 20: SAE band (T-invariant, 3 seeds) 0.9111 ± 0.0042.

| TXC-pre btk-only | T1 | T4 | T16 | Δ16 |
|---|---|---|---|---|
| seed 42 (the L1 comparator) | 0.9135 | 0.9181 | 0.8985 | **−0.0150** |
| 3-seed mean ± sd | 0.9161 ± 0.0024 | 0.9132 ± 0.0087 | 0.9011 ± 0.0024 | −0.0150 |
| s42 shuffle gap | +0.0000 | +0.0099 | +0.0305 | — |

k = 5 (context; not an L1 gate): SAE band 0.8450 ± 0.0171; pre s42
0.8417 → 0.8434 → 0.8651 (the k=5 rise reproduces on dev).

Both signature phenomena (k=20 decline, k=5 recovery) are visible on
the dev-8, so the split carries the program's signal.

## 3. Pyramid screening (pre-registered levels + gates)

- **L0 (seconds, during/after training):** dead-latent fraction,
  realized l0 vs nominal (train path AND probe path), per-position
  recon MSE profile over the window, loss-component traces. Sanity
  gate: probe-path realized l0 == k_serve ± 0.5 (TopK archs are exact;
  a miss is a bug, not a result).
- **L1 (the iteration loop): 4 000-step trainings**, dev-8 eval at
  T {1, 4, 16}, s42, k 20. Candidates compare against the
  **matched-steps baseline twin** (`txc_batchtopk_pre_btkonly` at
  4k steps, same cells — run FIRST as pipeline shakedown), never
  against the 20k-step P1 rows.
- **L2 (~2 GPU-h): 20 000-step trainings** (canonical count), s42,
  full dev T grid {1,2,4,8,16}, both k. Comparator: the P1 s42 rows
  (§ 2).
- **L3 (finalists only): canonical `run_experiment` pathway**, full
  38-task eval, holdout aggregation, seeds {1,2,42}, full T grid,
  both k, 20k steps. The ONLY numbers that may be quoted outside this
  exploration, PENDING mac-local ratification.

**Gates (dev-8, s42, k20, matched-steps comparator):**

- **L1 → L2 PROMOTE** iff (slope) Δ16^cand ≥ Δ16^base + 0.008, or
  Δ16^cand ≥ −0.005; AND (level) T16^cand ≥ T16^base and
  T1^cand ≥ T1^base − 0.010.
- **L2 → L3 PROMOTE** iff the L1 slope+level criteria hold at 20k
  steps against the § 2 s42 row; AND no interior T dips > 0.010 below
  the baseline curve; AND k=5 T16 within 0.010 of baseline's 0.8651
  (do not destroy the k=5 recovery to buy the k=20 slope).
- Failures are still DATA: every candidate (config hash, L0–L2
  numbers, verdict) is appended to `RESULTS.md`; negative results
  stay on the record.

## 4. Candidate 1 pre-registrations (txc_pro revival, `txc_pro_r1`)

Per the rewritten briefing + `task_hunt/TXC_PRO_RECOVERY.md`: revive
the RECOVERED implementation (`docs/recovered/txc_pro_phase5b_subseq_h8.py`,
blob 480f3755d, v2-ported, `consumes='sequence'`) as a NEW plugin arch
id — the deprecated `txc_pro` id and its DEPRECATED_ARCHS filters stay
untouched. Locked hparams verbatim: d_sae 18432, k_pos 20, h_size =
d_sae//5 = 3686 (n_matryoshka=8 is a phase id, NOT a level count),
contrastive_shifts (1,2) inverse-distance-weighted, contrastive_alpha
1.0, auxk_alpha 1/32, aux_k 512, dead_threshold 10 M tokens,
b_dec geometric-median init, decoder unit-norm + grad-orthogonalize,
multi_window False, contiguous t_sample sampling (the recovered
port's mode; non-contiguous = phase5b B2 mode is an ABLATION knob,
not the default).

**T-sweep semantics (the mac-c flag, decided here):** each grid T
trains its own model with **T_max = T** and **t_sample = max(1, T//2)**
— the RATIO rule (locked instance 10→5 is this rule's instance;
phase5b's t_sample sweep at T_max=10 found t = T_max/2 optimal:
0.8373/0.8516/0.8284/0.8231 at t 3/5/8/10). Budgets follow the
program convention at each phase: **k_train = k_pos·t_sample,
k_serve = k_pos·T** — the train/serve budget asymmetry is then a
CONSTANT factor ≈ 2 across the sweep instead of widening with T.
Holding absolute t_sample = 5 at T = 16 (asymmetry factor 3.2) is a
pre-registered L1 ABLATION, not the primary. At T = 1: t_sample = 1,
k_train = k_serve = 20, contrastive shifts still defined
(single-token windows at offsets 1, 2).

**Composition twins (both run at L1):** (a) `txc_pro_r1` — faithful
recovered composition, per-sample TopK then ReLU on selected values
(the paper-family composition; arm label `paper-match` in eval_cfg);
(b) `txc_pro_r1_btkonly` — same recipe with selection over raw
pre-acts by signed value, no ReLU in the sparsity path (mac-a's
btk-only convention, arm `btk-only`) — the arm the declining baseline
lives in. The L1 screen decides which composition carries the
ablation program; cross-arm comparisons are never quoted as wins.

**Training serving:** `consumes='sequence'` (SequenceBuffer,
(B, 128, d_in)); **batch_size = 1024 sequences** = the v1 txc_pro C3
convention (`c3_b1024_*` trainlogs; CARD.md E4 "v1 trainer 20k×1024").
Disclosed asymmetry vs the P1 window cells' 4096-token-slot
convention: txc_pro consumes ~1 anchor+2 positives per row per step —
a serving-convention difference inherited from the arch's design,
disclosed exactly like P1 Amendment 2b's tsae note. lr 3e-4, warmup
1000, Adam — the probing-runner defaults, unchanged.

**Eval pathway seam (disclosed for mac-local review):** the canonical
`ProbingEval` 1.2.0 dispatches its window path on
`consumes == 'window'`; the recovered class consumes sequences at
train but encodes fixed (B, T_max, d_in) windows at probe (its
`encode` hard-raises otherwise — it has ZERO v2 probing rows, so this
seam was never exercised). Plan: the plugin declares
`eval_consumes = 'window'` and `evals/probing.py` generalizes its
dispatch to `getattr(model, "eval_consumes", getattr(model,
"consumes", "token")) == "window"` — byte-identical behavior for
every existing arch (none defines `eval_consumes`), no eval-math
change, no protocol bump; covered by a unit test asserting old-path
equivalence. L1/L2 use a scratch dev harness importing the canonical
probe primitives (`_fit_probe` / `_score_probe` / `_encode_pool`)
restricted to the dev-8; L3 uses the canonical runner end-to-end.

## 5. Namespacing + hygiene (binding)

- Every leaderboard-bound cell carries eval_cfg keys
  `{"explore": "tscale", "arm": <composition arm>}` → distinct
  eval_keys; quoted P1 rows are NEVER touched (new arch ids make
  train_key collisions impossible as well).
- L1/L2 scratch results live under this exploration's `results/`
  (JSONL, append-only), NOT in the canonical leaderboard. Only L3
  goes through `run_experiment` (hard rule 1).
- Canonical checkpoints/manifest untouched by scratch loops; scratch
  ckpts under this exploration's `results/ckpts/` (gitignored).
- Runner refuses dirty trees; L3 launches follow the PIN-assert
  pattern (launcher asserts HEAD == PIN ∈ origin/arxiv, clean tree).

## 6. Budget

Pod ≈ $6/h (2×H100). L1 candidate ≈ 6 trainings × ~0.3–0.7 GPU-h at
4k steps + dev evals ≈ ~$15–25/candidate-with-twins; L2 ≈ ~$30; L3 ≈
~13 GPU-h ≈ $40 per finalist. Session ledger lines in
`briefings/MODAL_SPEND.md` RUNPOD section per house rules. Day cap
$150 unless Han raises.

## 7. LOG discipline

PTR LOG entries at: THIS split freeze, first L2 signal, any L3
launch/landing. mac-local reviews on push; amendments to this card are
append-only sections (never edits to frozen text), timestamped, before
the affected cells run.
