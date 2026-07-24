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

**Verdict: KEEP — best cell distill L12; the MODEL AXIS is the
finding.** All 64 cells complete; σ_null = 0.0046 on the full grid
(3σ = **0.0137**). The card's claim is the contrast g_tir − g_op:

| cell | T=8 | T=16 | T=32 | T=64 | clears 3σ at |
|---|---|---|---|---|---|
| base L12 | −0.009 | +0.008 | −0.005 | +0.019 | T=64 only |
| base L10 | +0.004 | −0.005 | +0.012 | +0.031 | T=64 only |
| **distill L12** | **+0.017** | **+0.020** | **+0.023** | **+0.042** | **every T** |
| distill L10 | −0.023 | −0.013 | −0.017 | +0.017 | T=64 only |

On the generator at mid-depth the contrast is positive at every T,
monotonically rising, and clears the null throughout — run depth has
window access the ambient anchor does not. Elsewhere it is noise until
T = 64. This is the briefing's "non-ambience is a (task, MODEL)
property" measured rather than assumed, and the mirror image of
candidate 1 (where base ≈ distill held). **P5 (base ≈ distill) is
decisively falsified; P1 and P3 also falsified.**

Per-cell ladders below are the primary layer base L12 (macro-OvR AUC),
kept because they carry the anchor lesson:

| target | tok | T=8 | T=16 | T=32 | T=64 |
|---|---|---|---|---|---|
| `tir` (PRIMARY) | 0.614 | +0.028 | **+0.049** | +0.032 | +0.037 |
| `boundary` | 0.618 | +0.017 | +0.036 | +0.030 | +0.008 |
| `op` (AMBIENT ANCHOR) | 0.760 | +0.037 | +0.041 | +0.036 | +0.018 |

The card's actual claim is the CONTRAST g_tir − g_op: −0.009, +0.008,
−0.005, **+0.019** at T = 8/16/32/64. It clears 3σ_null at exactly one
T and is noise elsewhere — no kill rule fires as written, but one-point
survival is not the predicted threshold ladder. P1 (clock threshold)
and P3 (order at T ≥ 32) are falsified; `tir` peaks at T = 16 and
declines, the localized-latent shape (STORY § 7).

**The methodological result of this candidate — why the ambient anchor
earned its place on the card.** `tir`'s within-window shuffle gap grows
monotonically with T (+0.008 → +0.025 → +0.032 → **+0.061**), which
reads like textbook order sensitivity. It is not: the **ambient** `op`
label, readable from the current sentence by construction, grows the
same way (+0.010 → +0.017 → +0.034 → **+0.065**), and so does
`boundary`. **A shuffle gap that grows with T is a generic property of
wider windows under a flatten probe**, not evidence about the latent.
Only the anchor-differenced contrast carries that claim. I recorded the
raw ladder as an order finding before the anchor landed and corrected
it in the LOG; the § 3 receipt is unaffected because it contrasts
anticipation against ambient targets *on identical rows at fixed T*.

---

## § 2b — Candidate 3: forbidden-word violation onset (SILOED) — **KILL (pre-registered ambience kill)**

Card [`forbidden_word/card.md`](forbidden_word/card.md), frozen before
any rollout existed. **SILOED** from Aniket's parallel forbidden-word
work throughout — no shared inputs, priors, or results. Data: 1169
R1-Distill rollouts on the pinned CoT-Control keyword-suppression split;
**violation rate 97.4 %** (the feasibility gate wanted ≥ 30 %), onset
labeled by exact whole-word offset. Screen: R1-Distill resid_post L12
over its own rollouts (generator = reader), horizons D ∈ {4, 8, 16} ×
T ∈ {2..32}, split by rollout. σ_null = 0.0099 (3σ = **0.0296**).
Results `forbidden_word/results/forbidden_word_{screen,verdict}.json`,
figure `forbidden_word/figs/forbidden_word_tscaling.*`.

| horizon | per-token | best window | \|diff\| | max g (win−tok) | max g_order |
|---|---|---|---|---|---|
| D=4 | 0.629 | 0.626 | 0.003 | −0.003 | +0.022 |
| D=8 | 0.612 | 0.622 | 0.010 | +0.010 | +0.047 |
| D=16 | 0.558 | 0.562 | 0.004 | +0.004 | +0.023 |

**Both kill rules fire.** Per-token is within 0.02 of the best window at
every horizon (P4), and the window never beats per-token beyond 3σ_null
(max g = +0.010 ≪ 0.0296). This is precisely the crux the card named as
its likely outcome: pre-violation the model **circles** the forbidden
concept, so each semantically-neighbouring token is individually
informative and one token reads the imminent-violation pressure as well
as a whole window. **The anticipation is ambient — KILL.** A
pre-registered kill that came true is a success of the process, not a
failure (prime directive).

One honest sub-effect, recorded because it is real: g_order
(flatten − mean) is positive and grows with T at the longer windows —
up to +0.047 at D=8/T=32, beyond 3σ_null. There is a faint genuine
*within-window order* signal, but it does not lift the window above a
single token, so it cannot carry the non-ambient claim; noted, not
counted as survival.

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

All 12 cells (σ_null = 0.0035, 3σ = 0.0106):

| cell | per-token | window MEAN | window SHUFFLED | window (ordered) | ordered − shuffled |
|---|---|---|---|---|---|
| base L10 `ant_kw` | 0.843 | 0.872 | 0.852 | 0.886 | **+0.034** |
| base L10 `ant_bts` | 0.765 | 0.760 | 0.744 | 0.785 | **+0.041** |
| base L10 `is_bt` | 0.798 | 0.798 | 0.793 | 0.806 | +0.013 |
| base L12 `ant_kw` | 0.851 | 0.870 | 0.851 | 0.887 | **+0.036** |
| base L12 `ant_bts` | 0.762 | 0.768 | 0.753 | 0.788 | **+0.035** |
| base L12 `is_bt` | 0.792 | 0.810 | 0.796 | 0.805 | +0.009 |
| distill L10 `ant_kw` | 0.844 | 0.868 | 0.859 | 0.895 | **+0.036** |
| distill L10 `ant_bts` | 0.765 | 0.769 | 0.752 | 0.792 | **+0.040** |
| distill L10 `is_bt` | 0.803 | 0.809 | 0.808 | 0.820 | +0.012 |
| distill L12 `ant_kw` | 0.854 | 0.868 | 0.856 | 0.895 | **+0.039** |
| distill L12 `ant_bts` | 0.757 | 0.768 | 0.750 | 0.778 | **+0.028** |
| distill L12 `is_bt` | 0.805 | 0.814 | 0.811 | 0.814 | +0.003 |

**The receipt the paper's § 5.2 task never had:** destroying
within-window order costs the *anticipation* targets **+0.028…+0.041
AUC** (3–4× the noise floor) on both models and both layers, while the
near-ambient companion `is_bt` loses only **+0.003…+0.013**.
Backtracking **anticipation** is genuinely order-sensitive; "this token
is inside a backtracking sentence" is not.

This comparison is immune to the T-confound that bit candidate 2 (§ 2):
every cell here is at the **same fixed T = 16** on **identical probe
rows**, so the anticipation-vs-ambient contrast cannot be produced by
window width.

Note the ordering `shuffled < mean < ordered` on `ant_kw`: a shuffled
flatten is *worse* than a position-symmetric mean, i.e. mis-aligned
positional evidence actively hurts, which is what makes the ordered
margin an order effect rather than a capacity effect.

---

## § 3b — Candidate 1 Stage 2: the head-to-head panel — **QUALIFIED POSITIVE**

The acceptance-gate deliverable: `lambda_recovery` vs T, one line per
architecture, on **real** Ward activations. 84 cells, 0 failures,
through the canonical runner (5 archs × T ladder × seeds {1,2,42} +
untrained; single scarce anchor d_sae = 2048 = d_in/2, nominal
k_pos = 8; eval_window_L = 32). Datasource `ward_real_lambda_base_l12`.
Figure `lambda_intensity/figs/stage2_tscaling.*`, numbers
`lambda_intensity/results/stage2_summary.json`. Recovery is held-out
Pearson r of a per-tile linear probe (chance ≈ 0). mean over 3 seeds:

| arch | T=1 | T=2 | T=4 | T=8 | T=16 | realized l0 |
|---|---|---|---|---|---|---|
| per-token BatchTopK SAE | 0.113 | — | — | — | — | 6.3 |
| **T-SAE** | **0.154** | — | — | — | — | 7.4 |
| Stacked | — | 0.109 | 0.143 | 0.125 | 0.094 | 7.0–7.9 |
| **TXC-pre** | — | 0.132 | 0.192 | **0.206** | 0.138 | 6.9–7.8 |
| TXC-post | — | 0.130 | 0.161 | 0.185 | **0.255** | **3.4→0.5** |

**Headline (matched budget): TXC-pre.** At realized l0 ≈ 7–8 — the same
per-token budget the token archs and Stacked run at — recovery rises
0.13 → 0.19 → 0.21 across T = 2/4/8, above the per-token BatchTopK SAE
(0.113) and T-SAE (0.154, the baseline the hunt names), with the
trained−untrained margin growing to +0.150 at T = 8 (untrained falls
0.09 → 0.06 → 0.01, so the T-dependence is learned, not read off at
init). It peaks at T = 8 and dips at T = 16 (0.138) — not saturation;
consistent with the Stage-1 regime-2 reading (a wider window pools more
lag-weighted history until extra positions dilute a fixed code budget).
**The hunt's target pattern — a window code beating per-token decoding
on a real-activation latent and improving with T — exists at matched
sparsity, modestly.**

Two heavy caveats, both reported:
- **TXC-post's higher numbers are not budget-matched.** Its recovery is
  monotone to the panel's best cell (0.255 at T = 16) but its realized
  l0 **collapses 3.4 → 1.8 → 0.9 → 0.49** — at T = 16 it fires ~1/16 the
  atoms/token the others do (the post-squash k_win//T correction starves
  the code as T grows). That is a striking *efficiency* observation but
  breaks the matched-l0 comparison, so it is flagged, not headlined.
- **Stacked is a training pathology at large T:** non-monotone, and at
  T = 16 the trained model (0.094) sits *below* its untrained control
  (0.171). Recorded as such; not a win for anyone.

**Verdict: QUALIFIED POSITIVE.** Real but modest (recovery band
0.10–0.21 on a hard regression), peaks rather than saturates, and the
single largest number is budget-confounded. This is the honest reading
the screen predicted: an order-free additive-in-window regime-2 latent,
where a window architecture earns a bounded advantage over per-token
decoding — not the unbounded rising-vs-flat separation the strongest
form of the hunt would want.

---

## § 4 — Methods notes (things that cost time; recorded for the next agent)

1. **The Stage-2 datasource is a plugin, not a core edit.** Real
   activations + a per-position label are presented as `SyntheticData`
   through the `module:fn` generator path
   (`src/explorations/task_hunt/real_lambda.py`), so the canonical
   runner and the existing `lambda_recovery` evaluator panel a real
   task with `temp_bench/core/` untouched. A real residual stream has
   no ground-truth directions, so `emission_features` carries a
   **reference basis, not ground truth**: the stream's DC direction
   plus the top principal directions of a fixed seed-0 subsample.
   `eauc`/`e_mean_max_cos` on these rows answer "does the dictionary
   span the stream's dominant variance directions?" — a sanity check,
   **never feature recovery**. The headline is `lambda_recovery` alone.

   *Why not simply leave it empty (the first attempt, and the sharpest
   lesson of the session):* an empty target set makes
   `_feature_recovery_auc` return NaN; **the leaderboard IS the eval
   cache**, JSON serializes NaN to `null`, and `LeaderboardRow` then
   rejects the cached read. Six such rows made the canonical artifact
   **unloadable for every subsequent run**, not just those cells — it
   surfaced as a `ValidationError` on an unrelated cell six seconds
   into a restart, long after the rows were written. **Never emit NaN
   into a leaderboard metric.**
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
5. **The distill tokenizer trap bites twice — force the fast backend.**
   `AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-
   Llama-8B")` returns the **slow** `LlamaTokenizer` (it even reports
   `is_fast=True`), whose `return_offsets_mapping` yields unusable
   spans. `conversion_depth/build_ward_stream.py` already recorded this
   for the whitespace-mangling symptom; the offsets symptom is worse
   because it **fails silently**: candidate 3's keyword-onset index
   would have come back −1 for every rollout, reading as "no violations
   found", tripping that card's feasibility gate and yielding a false
   *infeasible* verdict rather than an error. Caught by a CPU pre-flight
   check before the unattended stage ran. Use
   `PreTrainedTokenizerFast.from_pretrained(...)` — it carries the chat
   template too. (Second bug from the same check: under transformers
   5.x `apply_chat_template(tokenize=True)` returns a `BatchEncoding`,
   so `len()` gives 2; pass `return_dict=False`.)
6. **Stage-2 fairness — matched nominal `k_pos`, realized l0 diverges,
   and TXC-post's collapses.** Every cell runs at nominal `k_pos = 8`
   (the synthetic program's fairness definition, Part II § 3), but the
   BatchTopK family pools its budget across the batch so the *realized*
   `l0_per_token` differs: token/pre/stacked sit at 6–8, and **TXC-post
   collapses to 3.4 → 1.8 → 0.9 → 0.49 as T grows** (the post-squash
   `k_win // T` correction). So TXC-post's headline-looking 0.255 at
   T = 16 (§ 3b) is at ~1/16 the budget of the others and cannot be
   read as a matched-budget win — the matched-budget headline is
   TXC-pre. Per-cell realized l0 sits next to every recovery number in
   `results/stage2_summary.json`; any claim has to survive it.
7. **Leaderboard hygiene, and a repair.** Baseline restored:
   **7116 rows, 0 duplicate `eval_key`s, 0 rows with a null metric.**
   Six rows written earlier in this session on
   `ward_real_lambda_base_l12` (one 200-step plumbing cell + five
   untrained controls) carried `null` for `eauc`/`e_*` from the NaN
   defect in note 1 and were **removed** (backup:
   `/workspace/logs/leaderboard.backup.jsonl`). This supersedes an
   earlier decision, recorded mid-session, to keep the plumbing row
   rather than hand-edit the canonical artifact: that call was correct
   when the row looked merely superfluous, and wrong once the same rows
   turned out to be *corrupting* — they made the leaderboard
   unparseable. The repair removes only rows that cannot be loaded, all
   of them written by this session, and none of them a result.

---

*Reviewed (2026-07-24, mac-local): **APPROVED** — freeze order verified
by git forensics; every table in this record spot-checked against its
JSON artifact; leaderboard decomposition exact (8,616 = 7,116 + 1,416 +
84). Binding qualifications for downstream use are in the LOG review
entry: the per-tile code-readout convention must be stated wherever the
Stage-2 result is quoted; the T-SAE margin is ≈ 2σ at n = 3 (phrase
variance-aware); the stage2 figure needs a realized-l0 annotation on
TXC-post before external use.*
