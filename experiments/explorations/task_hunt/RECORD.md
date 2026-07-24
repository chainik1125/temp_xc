# Task hunt — arm A (runpod-d) record

**The question (`briefings/task-hunt.md`):** is there a real-activation
task where TXC recovery/detection improves systematically with window
size T while T-SAE and the per-token SAE stay flat — plus a
within-window shuffle ablation showing order matters?

Prime directive applies: **a sound verdict, never a win.** Every card
was committed before its screen ran (git order is the evidence); every
falsified prediction is scored as falsified.

Agent: `runpod-d` (H100, own 700 GB volume). Session: 2026-07-24.
Verdicts land in the shared [`LOG.md`](LOG.md); this file is the
methods + results record for arm A.

---

## § 0 — Substrate rebuild (and an unplanned reproduction check)

The pod started empty, so the Ward stream and both reader caches were
rebuilt from the committed builders
(`conversion_depth/build_ward_stream.py`, `cache_depth.py`). One input
was missing from the arxiv branch — `traces.json`, which
`results/c7_backtracking/stage_a/ATTRIBUTION.md` documents as
recoverable from the wasteland branch — and was re-ported per that
file's own reproduction recipe (`git show
origin/aniket-ward-stage-b:results/ward_backtracking/traces.json`,
3,616,858 bytes, matching the recorded byte count).

Rebuild stats reproduce the committed reference exactly: map_ok
99.97 %, BOS on every row, 2805 keyword events, 268/300 traces with a
Sonnet backtracking sentence. Cache sweeps: base 219 s, distill 238 s
(17 capture points each, fp16).

**Unplanned but load-bearing reproduction:** the shuffle receipt (§ 3)
re-ran the conversion-depth § 3 probe rows on these freshly rebuilt
caches and recovered `base/L10 ant_kw` tok **0.843** / window **0.886**
— identical to `conversion_depth/RECORD.md` § 3's published
0.843/0.886. The rebuild is faithful, and the depth study reproduces
on an independent pod from committed code.

---

## § 1 — Candidate 1: backtracking intensity λ̂ — **KEEP (qualified)**

Card: [`lambda_intensity/card.md`](lambda_intensity/card.md), frozen at
commit `b3179894` before `screen.py` touched any cache.
Results: `lambda_intensity/results/lambda_screen.json`,
`lambda_verdict.json`; figures `lambda_intensity/figs/`.

**Label.** λ̂ = the frozen fitted mirror intensity
(`backtracking/results/backtracking_mirror_stats.json`: a = −2.982,
c = +0.487, w₁..w₈) evaluated on each trace's **real** Sonnet event
history — a deterministic function of the previous eight sentences.
Built locally (`build_labels.py`) because runpod-b's drop had not
landed when the caches were ready (the briefing's never-idle rule).
It landed mid-session and the two builds **cross-validate at 99.93 %
exact agreement**; the 277 disagreements are sentence-boundary token
attributions plus their padded i < 8 sentences, which my frozen i ≥ 8
rule excludes.

**A pre-freeze label decision, disclosed.** The position-only floor
(tercile classification from sentence position alone, no activations)
measured **0.82 AUC** on full-λ̂ terciles versus **0.59** on the
kernel-only λ̂_hist. The fitted +0.487·position ramp is trivially
readable from position, so the PRIMARY target was set to λ̂_hist before
the card was frozen; full λ̂ is kept as a secondary and always read
against its 0.82 floor.

**Result — the T-story is real.** Window ceiling − per-token, primary
target, σ_null = 0.0031 (17 permutation cells, 3σ = 0.0094):

| cell | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---|---|---|---|---|
| base L12 | +0.011 | +0.029 | +0.035 | +0.047 | **+0.054** |
| base L10 | +0.007 | +0.018 | +0.033 | +0.047 | +0.050 |
| distill L12 | +0.014 | +0.025 | +0.025 | +0.039 | **+0.054** |
| distill L10 | +0.008 | +0.012 | +0.007 | +0.032 | +0.044 |

Rising in all four cells, no saturation by T = 32 — consistent with a
kernel support (8 sentences ≈ 130 tokens) far beyond the tested range.
Per-token sits at 0.776–0.795, well clear of the 0.592 position floor,
so the signal is history rather than the position ramp (P5 ✓). Base ≈
distill (P4 ✓).

**Result — the ORDER story is not.** g_order (flatten − window-mean) is
≤ 0 in 17 of 20 primary cells (min −0.047), and the within-window
shuffle costs only +0.002…+0.022 AUC. The entire window advantage is
**order-free evidence pooling**. This confirms the card's own regime-2
reading (additive-in-window over lag-weighted sentence indicators) and
means the shuffle ablation the briefing asks for is **negative for this
candidate**. Reported as a negative, not buried.

**Three frozen predictions falsified** (see LOG for the full scoring):
P1 at T = 8 (distill L10 gives +0.007 < 3σ, and the raw-flatten arm
fails more widely — flatten sits *below* per-token at T ≤ 4 in every
cell, the T-fold dimensionality cost of a fixed-budget probe); P2's
increment-shape clause (the largest step is T2→T4, not 8→16/16→32);
P3's order clause (above).

**Rule-scoring correction, disclosed.** `render.py` coded the kill rule
K2 as *strictly monotone at every step*, stricter than the card's text
("flat or non-growing over the whole tested range"). The strict form
fires on a single 0.005 dip at T = 8 in distill/L10 — inside 3σ_null —
and would have returned KILL. The card's text governs; both statistics
are reported (`P2_strict_every_step_monotone = false`) and the renderer
now emits both. No kill rule fires on the card's text.

---

## § 2 — Candidate 2: proof-operation run structure

Card: [`proofops/card.md`](proofops/card.md), frozen before the screen.
Labels: runpod-b's committed `labels/proofops.npz` used **as-is** (no
duplicate build — the candidate-1 duplication was not repeated).

The clock bridge is the design's spine: median **16 tokens/sentence**
means a window must reach T = 32 to span two sentences, so the
briefing's default 2…32 ladder sits mostly *below* this latent's
support. The frozen ladder is **T ∈ {8, 16, 32, 64}** (T = 128 excluded
— on a 128-token window only p = 127 would be eligible).

*Screen in progress at the time of writing; the verdict paragraph lands
in [`LOG.md`](LOG.md) and the ladder table is appended here.*

---

## § 3 — The order-sensitivity receipt for the EXISTING case study

`briefings/task-hunt.md` § "Also wanted". Script:
[`shuffle_receipt.py`](shuffle_receipt.py) (committed before running).
It reuses the conversion-depth probe rows **verbatim** — same frozen
§ 2 recipe, same 25,155/6,266 rows — so per-token and window reproduce
`conversion_depth/RECORD.md` § 3 exactly and the only new quantities
are the shuffled and MEAN arms. Shuffle = an independent permutation of
the T = 16 positions **per row** (seed 23), destroying order while
preserving the exact multiset of position-activations.

Raw test AUC (results: `results/shuffle_receipt.json`, figure
`figs/shuffle_receipt.*`):

| cell | per-token | window MEAN | window SHUFFLED | window (ordered) | ordered − shuffled |
|---|---|---|---|---|---|
| base L10 `ant_kw` | 0.843 | 0.872 | 0.852 | 0.886 | **+0.034** |
| base L10 `ant_bts` | 0.765 | 0.760 | 0.744 | 0.785 | **+0.041** |
| base L10 `is_bt` | 0.798 | 0.798 | 0.793 | 0.806 | +0.013 |
| base L12 `ant_kw` | 0.851 | 0.870 | 0.851 | 0.887 | **+0.036** |

**The receipt the paper's § 5.2 task never had:** destroying
within-window order costs the *anticipation* targets ≈ 0.035–0.041 AUC
(≫ the depth study's 3σ_null = 0.011), while the near-ambient companion
`is_bt` loses only 0.013. Backtracking **anticipation** is genuinely
order-sensitive; "this token is inside a backtracking sentence" is not.

Note the ordering `shuffled < mean < ordered` on `ant_kw`: a shuffled
flatten is *worse* than a position-symmetric mean, i.e. mis-aligned
positional evidence actively hurts, which is what makes the ordered
margin an order effect rather than a capacity effect.

---

## § 4 — Methods notes (things that cost time; recorded for the next agent)

1. **The Stage-2 datasource is a plugin, not a core edit.** Real
   activations + a per-position label are presented as `SyntheticData`
   through the `module:fn` generator path
   (`src/explorations/task_hunt/real_lambda.py`), so the canonical
   runner and the existing `lambda_recovery` evaluator panel a real
   task with `temp_bench/core/` untouched. `emission_features` is
   deliberately EMPTY — a real residual stream has no ground-truth
   directions — so `eauc`/`e_mean_max_cos` come back **NaN by design**
   and must never be read from these rows.
2. **The trainer infers `d_in` from datasource params before
   materializing**, so real_lm-style entries must declare it; the
   generator now checks the declared value against the cache rather
   than trusting it.
3. **Buffer sizing.** The pool's 2M-token default buffer re-samples the
   517,632-token Ward corpus ~4× per refill at d_in = 4096 and
   dominated wall-clock; the buffer is sized to the corpus instead.
4. **GPU serialization is mandatory on this box.** Running the two
   raw-activation screens and the Stage-2 pool concurrently on one
   H100 produced CUDA OOM in both the proof-op screen (T = 64 needs
   ≈ 28 GB for standardization alone) and a Stage-2 cell. Nothing was
   wrong with either job — a T = 64 flatten probe simply cannot share
   80 GB with a 50 GB probe job. The jobs were re-run serialized. Any
   future agent adding a third GPU job should chain, not fan out.
5. **One 200-step plumbing row** exists on
   `ward_real_lambda_base_l12` in the canonical leaderboard, written by
   the Stage-2 datasource smoke test through the canonical runner. It
   is kept (rather than hand-edited out of the canonical artifact),
   distinguishable by `n_steps=200`, and excluded from every Stage-2
   headline.
