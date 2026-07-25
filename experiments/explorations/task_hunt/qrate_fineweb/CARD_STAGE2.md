# FROZEN card — Stage 2, `punctint` q: the fineweb TXC panel (case study #3 shot)

**Status: FROZEN at commit (commit-then-run; no panel cell has been
executed when this card is committed — git order is the evidence).**
Agent: runpod-e. Briefing: `briefings/stage2-fineweb.md` (incl. both
addenda, read before this card was written — no reconciliation
amendment needed). Screen provenance: `CARD.md` (this directory) — the
hunt's only unconditional Stage-1 KEEP, margins recorded as **lower
bounds** pending the corrected-grid re-quote (part of this run,
§ 10). Datasource commit: `9c68f77e` (plugin + 3 `configs/data.yaml`
entries, load-tested, no cells).

**What Stage 2 is and is not.** Stage 1 screened window-vs-per-token
probes on raw activations. Stage 2 trains the actual arch panel
(TXC/T-SAE/SAE dictionaries) on the same activation stream and reads a
**linear λ probe on the learned code** — the TXC-vs-T-SAE question the
program exists to answer. Screen AUCs do not transplant (briefing
binding 5): the Stage-2 evidence line is the regression analog (§ 7).

## 1. The question, and the bet (briefing § "the bet")

One confirmed TXC case study exists (λ̂ backtracking, Ward). This panel
is the **breadth axis**: a different corpus (fineweb), a different task
family (ambient question-rate intensity), three subject models.

**The pre-registered bet:** my own screen finding is that on fineweb
the window advantage lives in a NONLINEAR readout (MLP on window
+0.06…+0.13) while a linear mean-pool sees ≈ +0.04. A TXC encoder is
itself nonlinear (sparse coding over the window) with only a linear
probe on top. **Both outcomes are informative and both are
pre-registered as reportable:** if TXC recovers a window gain a linear
pool could not, that is a strong novel claim (a sparse window code
captures the nonlinear-class advantage); if it does not, we have
learned the class of window advantage a sparse dictionary cannot
represent. Neither outcome is a failure of the panel.

## 2. Substrate (all frozen, all existing — zero new caching)

- **Corpus: the pinned 400-doc fineweb sample** (36,805 sentences).
  Every number in this panel comes from it and is quoted as such
  (binding 6; runpod's estimator finding means small-corpus readings
  are understatements — the 4k artifact is an OPTIONAL eval-side
  addendum only if the queue completes, § 11).
- **Activations:** replag caches, screen layers — gemma-2-2b hs14
  (5985×128×2304), gpt2 hs7 (5989×128×768), llama31-8b hs14
  (5924×128×4096). fp16 → fp32, one global RMS constant (Ward
  convention).
- **Target:** `lam_q` — exponential-kernel question-sentence intensity
  over the PREVIOUS 8 sentences (half-life 2), every token inheriting
  its sentence's value. NaN (dropped by the probe's non-finite guard)
  at kernel warm-up (sent < 8), question-sentence tokens (the builder's
  ambient-anchor masking), and BOS slots. Measured at load: finite
  frac 0.880–0.887, zero-frac 0.816–0.817, **NaN drop at sampled
  leading edges 8.7–10.8 % per T** (reported per cell population,
  § 7). No additional position floor (position AUC 0.47–0.53 ≈ chance
  on this face — `punctint_stats.json`).
- **trace_ids = document index** (asserted contiguous) → the v2 trace
  split keeps every document inside one probe half.

**Clock bridge (from the screen card § 3, still the honest limit):**
21.1–21.6 tokens/sentence, so tile T spans ≈ 0.09/0.19/0.37/0.74
sentences and ≈ 3/6/12/23 % of kernel mass at T = 2/4/8/16. **This
ladder sits at the bottom of the label's timescale.** The predicted
shape is rising-unsaturated; a flat result is a real negative at this
reach, recorded as reach-limited — stated before the run, as the
screen card did.

## 3. Primary model choice (addendum 2 item 1)

**gemma-2-2b** — mid-scale, d_in 2304 keeps the tsae buffer cost sane,
more representative than gpt2. **First-cell timing clause:** if the
first trained cells project the 84-cell panel beyond ~6 h wall, the
panel moves to gpt2 via a card-amendment commit stating the measured
numbers; the design does not otherwise change.

## 4. Panel design (the λ̂ Stage-2 pattern; deviations named)

`design.uniform_cells` on `fineweb_punctint_q_gemma2_l14`:

- **Archs:** the briefing's five — batchtopk_sae (T=1), tsae (T=1),
  stacked_batchtopk, txc_batchtopk_pre, txc_batchtopk_post. No
  spectral_txc (λ̂ precedent: no predicted role in regime 2, disclosed).
- **T ladder {2, 4, 8, 16}**, eval_window_L = 32 (every T tiles L).
- **Capacity: single scarce anchor d_sae = d_in/2 = 1152**, k_pos = 8
  (the λ̂ convention: one labeled operating point, not a frontier —
  disclosed narrowing). Replication arms use the same rule: gpt2 384,
  llama31 2048. Dict constraint holds everywhere (max k·T = 128 ≤ 384).
- **TXC-post at per-T nominal k = 8·T (trained AND untrained cells)** —
  binding 2, the code-rate convention from
  `lambda_intensity/card_stage2_postmatched.md` § 2 adopted FROM THE
  START rather than as a repair. This deviates from the program's
  equal-nominal-k rule for one arch, in favour of matched REALIZED
  l0/token; the deviation is confined to post and stated here.
- **Seeds {1, 2, 42}; untrained controls** (n_steps = 0) at every
  (arch, T), k_pos = 8 (post: 8·T), per seed. **84 cells.**
- **n_steps 8000, batch 1024/T** (throughput-normalised),
  **buffer_tokens = 766,080 = the corpus exactly** (fresh-panel unlock,
  addendum 2 item 2: frozen here, applied uniformly to every arch;
  corpus-sized so a refill is one pass, the λ̂ lesson). Replication
  arms: gpt2 766,592; llama31 758,272.
- **tsae trained cells scheduled first** (addendum 2 item 2).
- **Both probe columns on every cell** (binding 3):
  `eval_extra = {"lambda_probe_v2": True, "lambda_v2_probe": "ridge",
  "lambda_v2_n_windows": 8192, "lambda_v2_split": "trace"}`
  (`PROBE_V2_SPEC.md` § 2). **Claim on v1** (leaderboard-canonical
  until the methods rule fires); v2 reported beside it. v2 adequacy at
  the anchor: n_rows at T=16 = 16384 ≥ 8·p = 9216.
- Canonical runner only; results to
  `results/stage2_fineweb_punctint_q_gemma2_l14.json`.

**Scoring arm named per probe class (my own convention, program-binding):**
the probe class is v1's linear probe on ONE tile's code. Token archs,
TXC-pre and TXC-post all read **p = d_sae** features — matched width;
these are the comparison set. **Stacked reads T·d_sae features (wide)**
— reported as its own line, excluded from the headline comparison, its
v2 column the readable one (PROBE_V2_SPEC § 1 knob 2 caveat). No
max-over-arms anywhere: the headline window arch is **TXC-pre**,
compared against the **better of the two token archs** (a conservative
baseline choice, not a max over window arms).

## 5. Realized-l0 band (binding 2, pre-registered)

Trained cells: predicted realized l0/token in **[5.0, 8.0]** (the λ̂
panel's pre band 5.81–7.84 and matched-post band 5.7–7.7). Any trained
cell outside the band is recorded as a **residual mismatch and carried
into the reading, not smoothed over**; if > 25 % of trained cells land
outside, the budget-matching claim for the panel is void as measured.

**Falsifier (postmatched § 6, inherited):** untrained post cells must
realize l0/token = **8.00 ± 0.02** at every T (untrained cells run the
exact BatchTopK path). If not, the l0 ≈ k/T mechanism is wrong, the
per-T k is not the right correction, and the post arm's run is void —
reported as a failed amendment, never reinterpreted. Untrained
batchtopk_sae/tsae/pre similarly realize their nominal 8.00.

## 6. Document identity — BINDING at panel (briefing binding 1 + addendum 2 item 3)

`doc_mean_only_auc` = **0.901** on this face (screen measurement): the
naive panel is uninterpretable without the control. Two instruments,
both pre-registered here, both off-leaderboard:

- **(a) Doc-identity FLOOR** (`stage2_support.py`, label-side): a
  doc-mean-only predictor's Pearson r on the SAME v1 eval windows
  (sampler imported, seed 1; doc mean = mean of the doc's finite lam_q
  values over the whole stream — the identity route's ceiling as a
  single number per doc). Printed beside every window cell in every
  table and drawn on the figure.
- **(b) Within-document RECEIPT** (`stage2_demeaned.py`, the
  `probe_capacity.py` pattern): out-of-band re-fit of the SAME trained
  codes against **doc-demeaned targets** (subtract the doc's train-side
  finite mean), ridge at nw = 8192, plus the exact v1 replication row
  as the licence (must reproduce the leaderboard cell to ~1e-6 before
  any other row is read).

**Pre-registered: the within-doc face may sit near floor** — q's
zero-fraction is 0.817, so within-doc λ̂ variance is thin. **If the
panel's window−token gap collapses under doc-demeaning, that is a
sound NEGATIVE for the within-document claim and it is reported as
loudly as a win** — the verdict becomes "corpus-level rate-signature
KEEP / within-doc NEGATIVE" (§ 8 V4), not a softened sentence.
`doc_mean_only_auc` is a disclosure statistic that triggers this
control, NOT a kill bar (ratified convention, my own datapoint).

## 7. Evidence line + NaN receipt (binding 5 clarified; d-addendum analog)

Beside every window cell, from `stage2_support.py` on the same
windows: **the regression analog** — in-window question-token count
(current tile, same probe convention) → target, held-out r per T. A
window cell that does not beat it at matched T is counting visible
question marks. (Prediction: small at T ≤ 16 — a 16-token tile is 0.74
sentences — but measured, not assumed.) Also reported per T: the
fraction of sampled tile targets dropped by the non-finite guard
(measured 8.7–10.8 % at load), and the train/eval row counts after the
drop.

## 8. Pre-registered predictions (scored in the verdict, each way)

- **P1 (conversion baseline).** Trained token archs land well above
  chance: v1 r in [0.15, 0.50] on gemma (the screen's per-token
  readability, now through a learned sparse code).
- **P2 (regime-2 shape).** Window archs rise monotonically in T
  (seed-mean), unsaturated at T = 16 (clock bridge § 2: 23 % kernel
  mass). The λ̂ shape on a second corpus.
- **P3 (the bet, § 1).** TXC-pre clears the token baseline by ≥ +0.05
  (v1, seed-mean) at T = 16 on gemma — the sparse-window-code-captures-
  the-nonlinear-gain outcome. Failing P3 while P1–P2 hold is the
  other informative branch.
- **P4 (identity).** Doc-mean-only floor r ≥ 0.5 on eval windows
  (doc_mean_only_auc 0.901 translated to regression); demeaning
  shrinks every arm's r; the window−token gap keeps its sign under
  demeaning (§ 6 allowance: collapse = the pre-registered NEGATIVE
  branch, reported loudly).
- **P5 (post at matched budget).** Post tracks pre within 0.03 at
  every T (the λ̂ matched shape) — i.e. the shared-code squash does not
  cost the window advantage at matched code rate.
- **P6 (probe generations).** v2 ≥ v1 on every cell, largest lifts on
  Stacked (wide) and T = 16 (v1's thinnest n); claim unchanged (v1).

## 9. KEEP / KILL (frozen; per model, majority rule, no pooling)

All clauses read **v1 `lambda_recovery`, seed-mean per cell**, TXC-pre
vs the better token arch (§ 4 scoring convention). Per model:

**KEEP** iff ALL of:
- **K1 (gap):** pre − best token ≥ **+0.05** at some T;
- **K2 (shape):** pre(T=16) ≥ pre(T=2) + 0.02;
- **K3 (training real):** pre − untrained-pre ≥ +0.05 at the K1 T;
- **K4 (within-doc):** the § 6b receipt's window−token gap > 0 at the
  K1 T.

**V4 split verdict:** K1–K3 pass but K4 fails ⇒ **"corpus-level
rate-signature KEEP; within-document NEGATIVE"** — both halves stated,
neither softened.

**NEGATIVE** if pre ≤ best token + 0.02 at every T, OR pre is
flat/falling (fails K2 and no interior T clears pre(2) + 0.02).

**WEAK — no rule fires as written** if the gap's maximum lies in
(0.02, 0.05); recorded with the numbers, never upgraded by narrative.

**VOID:** the § 5 falsifier fires (post arm void), or > 25 % of
trained cells out of the l0 band (budget-matching void), or any
K-clause cell pair differs in probe settings (must be impossible by
construction; checked in hygiene).

**Cross-model headline (three verdicts, stated majority):** the
replication arms (gpt2, llama31: TXC-pre at its two best gemma T
values + tsae + batchtopk_sae + untrained, 3 seeds) are scored on
K1/K3/K4 at those T values (K2 needs a ladder the replication does not
run — disclosed). The program headline sentence may claim generality
only for the verdict ≥ 2 of 3 models share; per-model paragraphs are
mandatory either way.

## 10. Also in this run (screen-side, no leaderboard rows)

The Stage-1 re-quote the briefing orders: `requote_screen.py` — the q
face's margins on the corrected matched-class grid (tok linear/MLP vs
`anchor ⊕ context-mean` linear/MLP at T ∈ {16, 32, 64} + its
foreign-context null; frozen probe stack; the screen's own rows). The
screen KEEP margins are on record as lower bounds; this states them
properly. Committed before it runs, separate from the panel.

## 11. Deliverables & order (the briefing's 12-h queue)

1. this card + runner (one commit) → 2. gemma panel, tsae first →
3. `stage2_support.py` floor/evidence/NaN receipts + `stage2_demeaned.py`
receipt → 4. variance receipts
(`support_stats/stage2_variance.py`, v1 defaults + `--probe v2`), LOG
verdict + scorecard, money-plot figure (realized-l0 annotations,
doc-floor line, corpus size in the caption) → 5. replication cells →
6. `tss` (primary model only) → 7. optional dialevel recency
pre-flight. Stopping early at any gate is fine; every stop is recorded.
Optional disclosed addendum if headroom after 5: a T = 32 rung
(tiles L exactly; kernel mass 0.42) — OUTSIDE the frozen scoring, v2
the readable column there (v1's n = 1024 < p at T = 32).

License note: fineweb is ODC-By; the corpus receipt travels with any
graduating figure.

---

## APPENDIX A (2026-07-25, force majeure) — A40 restart; design unchanged

Appended at restart, before any restarted cell (git order is the
evidence). The original H100 run's cells were UNPUSHED and died with
the pod (`briefings/a40-bootstrap.md`; LOG force-majeure entry). This
card is NOT re-frozen — the design above runs exactly as written. What
changes is venue and execution only:

- **Venue:** 3× A40 (GPUs 3–5 of a shared interim 6×A40 pod),
  ephemeral storage, ~12 funded hours from restart.
- **Cache rebuild receipt:** `tokens.npz` rebuilt via the frozen
  `replag.build_labels.tokenize_model` — 5985×128, grid 766,080
  tokens (= § 4 `buffer_tokens` exactly), all 5,985 rows byte-equal to
  the committed `punctint_fineweb_gemma2.npz` stream; hs sweep via
  `replag.cache_acts` unchanged; the datasource's per-row alignment
  assertion at materialise is the gate before any cell.
- **Execution:** the frozen `_cells()` list (order included) is
  sharded round-robin across the three A40s as three single-GPU pools
  (the trainer is single-device; the H100 original pooled workers on
  one card). tsae trained cells lead every shard, preserving § 4
  scheduling. Shard dumps merge into § 4's results path in frozen
  cell-list order; leaderboard rows go through the canonical runner
  as always.
- **§ 3 first-cell timing clause, A40-scaled:** A40 ≈ 2–3× slower per
  cell than the H100 the ~6 h projection was written for. The clause
  is re-read against the funded window: the gpt2 fallback fires if
  first-cell timing projects the gemma panel past ~hour 8 of the
  12-hour funding clock (protecting a COMPLETE panel + receipts per
  the bootstrap's triage), with the measured numbers stated in the
  amendment if it fires.
- **Probe columns:** unchanged (§ 4), and the λ-readout METHODS
  DECISION taken in the LOG (2026-07-25) binds the reporting: v1
  canonical, paired v2 reported, never quoted as canonical.
- The dialevel recency pre-flight (queue item 7) is CANCELLED for
  this funding window per the bootstrap; queue items 5–6 run only
  after the gemma panel + receipts are pushed.
