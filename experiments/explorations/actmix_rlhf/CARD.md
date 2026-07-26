# ACTMIX RLHF card — shuffle control + T-sweep, paper-match (eval-only) + btk-only

**Frozen pre-run.** Mandate: Han's ASSIGNMENT UPDATE in
`briefings/actmix-runpod-2.md` (~22:00, commit 387268df0); pin source:
`COMPOSITION_AUDIT.md § 6` (mac-c). Agent: **runpod-2**, GPU 2 —
btk-only training cells queue BEHIND the running EM grid; short
eval/cache jobs use spare capacity. Prime directive: a sound verdict,
never a win. Verdict PTR in `task_hunt/LOG.md`; `RUNPOD` ledger lines.

## § 1 — Task (audit § 6, consumed verbatim)

HH-RLHF preference decomposition: `Anthropic/hh-rlhf` harmless-base
train, FIRST N = 1000 (chosen, rejected) pairs; `google/gemma-2-2b`
**BASE**, L12 residuals (hook on `model.model.layers[12]` output,
d_in 2304); per-side response_mask from char-LCP + offset_mapping
(max_length 256, right padding); features aggregated as MEAN over
response tokens; ranking metric = `mean_rejected − mean_chosen`.
Cache rebuilt by `build_cache.py` (verbatim port of
`han-phase7-agent-c@023d52c24`; the npz is not mirrored on HF).
**Integrity gate:** the response-length t-test must reproduce
PHASE-7's OWN recorded run (rejected ≈ 36.23 / chosen ≈ 28.57 /
p ≈ 9.76e-10 — research log 2026-04-26-c1 verbatim; Ye et al.'s
App B.1 absolutes are a different tokenizer/dataset version, which
phase-7 itself did not match) — the builder refuses to write the
cache otherwise. GATE PASSED at rebuild: 36.232 / 28.573 /
p 9.76e-10 — the phase-7 substrate reproduces to the digit.

## § 2 — Arms

**paper-match (EVAL-ONLY; case-study artifacts, NOT leaderboard
rows — the em-redo `probe_codes.py` precedent for out-of-runner
currencies).** The four shipped seed-42 checkpoints, downloaded from
public `han1823123123/txcdr-base` (`ckpts/<arch_id>__seed42.pt`;
sha256 recorded in results):

| arch_id | class (vendored, blob-stamped) | composition (audit § 6) | knobs |
|---|---|---|---|
| topk_sae | TopKSAE | TopK→ReLU per-token | k 500/token |
| tsae_paper_k500 | TemporalMatryoshkaBatchTopKSAE | ReLU→threshold at eval (`use_threshold=True`) | k 500, groups [3686, 14746] |
| tsae_paper_k20 | same | same | k 20 |
| agentic_txc_02 | MatryoshkaTXCDRContrastiveMultiscale | TopK→ReLU per-window | T 5, k_win 500 (=100·T), shifts-scales 3, γ 0.5 |

Eval = the paper's aggregation verbatim (`decomp.py`, single shared
implementation): per-token archs encode every position; window archs
slide stride-1 with RIGHT-EDGE attribution (audit § 6's convention,
positions 0..T-2 zero); response-mask mean. tsae evals with
`use_threshold=True` (the shipped convention). Shuffle twin per cell:
per-sliding-window independent input permutation, seed 42, pre-encode
(protocol semantics = EM card § 3 = Aniket's `shuffles.py`); T = 1
archs: shuffle ≡ identity BY CONSTRUCTION — stated, not simulated.
"T-read" beyond T = 5 is NOT possible on a fixed-T checkpoint — the
T-sweep lives in the btk-only arm; paper-match contributes its T = 5
point + shuffle twin.

**btk-only (canonical runner → leaderboard rows; evaluator `rlhf`
protocol 2.0.0 ported into `src/temp_bench/evals/rlhf.py` — the
`em.py` port precedent; plugin file, no core edits).** Datasource
`gemma_2_2b_base_l12_phase7` (NEW data.yaml entry): the phase-7 BASE
training stream itself, `han1823123123/txcdr-base-data
activation_cache/resid_L12.npy` (24000 × 128 × 2304 fp16) converted
in place to the keyed `acts.npy` layout — the EXACT activations the
shipped checkpoints trained on; zero re-forwarding. Cells (all
d_sae 18432, v2 training conventions n_steps 25 000 / batch 1024
windows (tsae 32 seqs) / lr 3e-4 / default warmup; NO bricken;
seed 42 = the paper's seed; seed 1 stretch):

| cell | arch | knobs | role |
|---|---|---|---|
| txc_post_btkonly_T{1,2,5,8,16} | txc_batchtopk_post_btkonly | k_pos = 100·T per window | the T-sweep; T5 = paper shape; T1 = controlled limit |
| sae_btkonly_k500 | batchtopk_sae_btkonly | k_pos 500 | paper baseline shape |
| sae_btkonly_k100 | batchtopk_sae_btkonly | k_pos 100 | OUR matched T = 1-limit comparator (the paper has no such cell — labeled ours) |
| tsae_btkonly_k500 / _k20 | tsae_btkonly | k_pos 500 / 20 | paper tsae shapes |
| untrained twins (all shapes) | same | same, n_steps 0 | floors |

Known structural difference, stated: the paper's TXC arm is
matryoshka-CONTRASTIVE (shifts, multiscale); no v2 twin exists —
the btk-only TXC arm is the v2 post backbone at the paper's SHAPES
(d_sae, k_win = 100·T, T). The arm answers "does a
composition-harmonized window code at paper shapes carry order
signal on this task", not "is agentic_txc_02 reproduced".

mac-a's identity note (LOG ~22:20) applies: at k = 500/token the
selection is far deeper than the hunt's k = 8–20 — arms MAY
genuinely diverge here, or may again be function-identical.
**Gate: the FIRST btk-only cell (sae_k500) is the smoke — check
train-log `neg_frac` and realized l0 before launching the rest.**
Either outcome is informative and is reported (identity ⇒ the
btk-only arm doubles as the relu-mix retrain arm, stated).

## § 3 — Metrics (decomp.py, shared by both arms)

Primary: **preference_auc** (5-fold seeded CV over pairs; per fold:
rank by |mean_rejected − mean_chosen| on train folds, signed top-20
projection, AUC = P(score(rejected) > score(chosen)) held-out).
Secondary: mass@20 (the paper table's "% mass", judge-free);
top-20 length-Pearson (the paper's length-spurious diagnostic:
mean |r| + count |r| > 0.5); realized l0 per encode unit over
response positions; top-20 fold-overlap. Shuffled twin of each for
T > 1. NO autointerp/judge stage (the paper's "N/20 semantic" column
needs an API judge — out of scope; the briefing's table is the
quantitative head).

## § 4 — Pre-registered expectations (BEFORE any result)

- **R-E1 (paper-match TXC shuffle, the headline control):**
  agentic_txc_02's preference signal is substantially
  order-INSENSITIVE — shuffle_gap(preference_auc) < 0.02 — because
  the paper's own criticism found length-spurious features (3 of
  top-20, |r| > 0.5) and length is window-density, invisible to
  within-window permutation. A LARGE gap would mean the TXC carried
  genuine order signal the paper's reading missed — reported at
  equal prominence.
- **R-E2:** per-token arms' shuffle column = identity (analytic).
- **R-E3 (composition contrast at T = 5):** btk-only TXC at paper
  shapes vs shipped agentic_txc_02 (TopK→ReLU): directional per the
  shared ACTMIX pre-registration — the TopK→ReLU family zeroes
  selected negatives at depth 500, so the harmonized arm should be
  ≥ the shipped arm in preference_auc at matched shapes; magnitude
  unknown (contrastive-vs-plain structural difference confounds —
  stated).
- **R-E4 (T = 1 limit):** txc_post_btkonly@T1(k100) within ±0.03
  preference_auc of sae_btkonly_k100.
- **R-E5:** untrained floors ≈ 0.5 AUC (chance).
- **R-K1 (machinery falsifier):** every trained per-token cell
  (paper-match topk_sae AND btk-only sae_k500) reaches
  preference_auc ≥ 0.55 — the substrate carries a strong preference
  signal (App B.1: length alone separates at p = 9e-10); below ⇒
  pipeline broken, debug, do not interpret.
- **R-K2:** cache integrity = the App B.1 t-test gate (builder).
- **R-K3 (paper-structure reproduction, soft):** paper-match
  agentic_txc_02's top-20 contains ≥ 1 length-spurious feature
  (|r| > 0.5) — the paper's own observation; miss reported, not
  patched.

## § 5 — Dispatch, cost honesty, descope (EM lesson applied)

Paper-match evals: ~2–5 min/arch on spare GPU capacity — tonight.
btk-only cells (gemma d_sae 18432): measured-basis estimate from the
EM grid's fp32 step rates scaled by FLOPs (×T·d_sae·d_in ratio):
per-step ≈ 0.21 s × (T·18432·2304)/(1·32768·3584) ≈ 0.076·T s
contended ⇒ T1 ≈ 0.5 h, T2 ≈ 1.1 h, T5 ≈ 2.6 h, T8 ≈ 4.2 h,
T16 ≈ 8.5 h; token cells ≈ 0.5–1.5 h. **Dispatch order (priority =
core first): sae_k500 (smoke gate) → T5 → T1 → sae_k100 → T2 →
tsae_k500 → tsae_k20 → untrained batch → T8 → T16.** T8/T16 are
pre-declared STRETCH: if the queue reaches them after ~13:00 London
they are dropped without further amendment (this card, unlike the
EM card, prices them honestly up front). Seed 1: only after
everything else. Budget est ~6–12 GPU-h ≈ $20–35; cap intact.

Freeze/pin discipline as EM: card pushed BEFORE any result-producing
run; driver `--pin` from origin history; TEMP_BENCH_ALLOW_DIRTY=1
established practice; per-process GPU fraction caps sized to
co-residency (btk-only cells ≤ 0.22 while the EM T8 cell runs; full
card after).

## § 6 — Deliverables

The Dmitry table (both arms, one row per cell): preference_auc |
shuffled | gap | mass@20 | shuffled mass | realized l0/unit |
len-spurious count; T-sweep figure (btk-only curve + paper-match
T = 5 point + bands + untrained floor + shuffle overlays, house
Okabe-Ito style); results JSON with ckpt sha256s + vendor blob shas;
LOG verdict PENDING TEAM REVIEW scoring R-E1..E5 + R-K1..K3 as
written; ledger est + actuals.
