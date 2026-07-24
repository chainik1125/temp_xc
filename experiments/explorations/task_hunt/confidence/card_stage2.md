# FRESH card — hedging-trend LEVEL Stage 2 (task-hunt round 2, arm B)

**Status: FROZEN at commit (commit-then-run; no panel cell, reference
row, or diagnostic has been executed when this card is committed — git
order is the evidence).** Agent: runpod-e. Briefing:
`briefings/task-hunt-r2-e.md` § 1. Program decision (review, LOG
2026-07-24): **an aggregation-framed regime-2 win is ACCEPTED**;
shuffle IMMUNITY is disclosed as the mechanism receipt.

This card is fresh by construction: the killed screen card (`CARD.md`,
verdict in `../LOG.md`) is **motivation only, never confirmation**. Its
numbers appear below solely to justify spending a panel; every claim
this card can produce comes from new cells through the canonical
runner.

## 1. The claim under test

The **hedging-trend level** — the trailing-8-sentence least-squares
slope of the judged per-sentence confidence state (0 hedged / 1 neutral
/ 2 committed) — is a **regime-2 aggregation latent** of the
R1-Distill Ward stream: window-readable, per-token-blind, carried by
**order-free pooling** of per-position hedging evidence. Stage 2 asks
the hunt's architecture question: does a window code (TXC) recover it
better than per-token decoding (per-token BatchTopK SAE, T-SAE), with
the advantage growing in T?

**No order claim is made.** The Stage-1 screen showed the window gap is
achieved by the order-free MEAN probe and survives context shuffling
(g_order ≤ 0) — that shuffle **immunity** is the disclosed mechanism
receipt: what the window adds is pooled level evidence, not sequence
order. Under anchor matching, slope ≈ anchor − window-mean: an
order-free functional. This is the class the program's theory says
separates TXC from per-token-decoded T-SAE *without* order.

Motivation numbers (screen, cited as motivation only — 3-class acc,
chance 1/3, distill hs15): per-token 0.468 linear / 0.503 MLP vs
window-MEAN 0.521 → 0.545 → 0.565 at T = 16/32/64; permutation nulls
at chance; hedge-state control regime-1 (per-token ≈ window). The
monotone T-growth exists **only on the generator** (base reader:
0.525 → 0.515 → 0.511, flat) — hence the reader below.

## 2. Label (frozen, untouched)

Target = the committed **`slope8` grid** of `../labels/confidence.npz`
(builder `../labels/build_confidence.py`), exactly as frozen for the
screen. Its domain is masked to finite ∧ `valid` (out-of-span tokens
carry a wrapped-index artifact in the raw grid and are excluded);
undefined positions stay **NaN** — 83.7 % of positions carry a label.

**Disclosed deviation from the λ̂ precedent:** candidate 1 densified its
label with the generator's own warm-up convention. slope8 has no
generator to extend — a shortened trailing fit would be a *different
statistic* — so the label is NOT densified. Instead the λ probe drops
non-finite leading-edge targets (`lambda_recovery._train_lambda_probe`,
this branch), an extension proven a no-op for every all-finite grid
(`tests/test_lambda_nanmask.py`: the all-finite path is byte-identical
— the λ̂ rows and all synthetic benches are untouched; their cached
eval rows are not recomputed in any case).

## 3. Datasource + reader (frozen)

`ward_real_slope8_distill_l14` (`configs/data.yaml`, plugin
`src/explorations/task_hunt/real_slope.py`, mirroring the reviewed
`real_lambda.py`: fp16→fp32, one global RMS constant, reference-basis
`emission_features` that are **not ground truth**). Reader =
**distill hs15 (resid_post L14)** — the screen's frozen primary cell.
This mirrors the λ̂ panel's *reasoning* (the screen's primary cell
becomes the panel datasource), not its layer: here the phenomenon's
T-growth lives on the generator only (§ 1), so a base-reader panel
would test a latent its own screen says is flat.

## 4. Panel (the reviewed candidate-1 Stage-2 pattern, one disclosed change)

`run_stage2.py` (committed with this card): 5 archs — per-token
BatchTopK SAE, T-SAE, Stacked, TXC-pre, TXC-post — × T ∈ {2, 4, 8, 16}
(token archs T = 1) × seeds {1, 2, 42} + untrained controls = **84
cells**, single scarce anchor d_sae = 2048 = d_in/2, nominal
k_pos = 8, eval_window_L = 32, n_steps = 8000, batch 1024//T,
buffer_tokens = 524288, all through
`temp_bench.core.runner.run_experiment` on a clean tree.

**The one deviation from round-1 Stage 2, adopted from runpod-d's
frozen amendment (`../lambda_intensity/card_stage2_postmatched.md`
§ 2–3, the code-rate convention):** `txc_batchtopk_post` runs at
nominal **k_pos = 8·T** (16/32/64/128), trained AND untrained, so its
realized per-token code rate sits at ≈ 8 like the rest of the panel
instead of collapsing as 8/T (the post-squash `k_win // T`). This
panel therefore has **no unmatched post arm** — the budget confound
that qualified the λ̂ panel's best cell is designed out, at the cost of
one fewer comparison to round 1 (disclosed, and recoverable later by
running nominal-8 post cells if wanted).

**T = 32 is excluded, with the reason frozen:** at L = 32 a T = 32 cell
has one tile per eval window and n ≈ 1000 < p = 2048 probe rows — deep
in the interpolation regime — and it breaks comparability with the λ̂
figure's T axis. The screen's T = 32/64 growth is real but belongs to
raw-probe territory; the panel reads T ≤ 16 and says so.

## 5. Readout (binding convention, carried verbatim into the record)

Headline = `lambda_recovery`: held-out Pearson r of a per-tile
**linear** probe against slope8 at the tile's leading edge (chance ≈ 0,
empirical chance floor reported per cell). Stage 2 reads **one tile's
code per prediction** (the per-tile leading-edge convention, the same
leak-free design as the synthetic DPI bench), so per-token archs are
read at single positions **by construction**. Any comparison sentence
quoting this panel must say it holds **"under the code-readout
convention"** and carry the code-rate defense: pooling T-SAE codes
across T positions would spend T× the code bandwidth a window arch
uses. `eauc`/`e_*` on the reference basis are span sanity checks, never
feature recovery.

## 6. Probe capacity (pre-registered, measured from label geometry pre-run)

Finite leading-edge probe rows under the evaluator's fixed sampling
(n_windows = 1024, seeds 0/1 — a geometry fact of the frozen label
grid, not a result): train/eval ≈ **27143/26018, 13579/13037,
6788/6527, 3393/3269, 1702/1636** at T = 1/2/4/8/16, vs p = d_sae =
2048 features. **T = 16 is in the interpolation regime (n < p)** — one
notch worse than the λ̂ panel's n = p. Pre-registered consequence: a
T = 16 drop for any dense-code arch is ambiguous (probe capacity vs
representation), exactly runpod-d's § 4(c).

**Diagnostic (pre-registered; run either way after the panel; labelled
post-hoc; kept OUT of the leaderboard):** refit the slope8 probe on the
same trained checkpoints with n_windows = 8192 and with ridge; if extra
probe data lifts T = 16 cells, the honest statement is that the T = 16
column is probe-limited for every dense arch, not that representations
degrade.

**Raw-activation reference (pre-registered; computed after this card is
committed; off-leaderboard):** raw per-token linear r and raw
window-MEAN linear r at each panel T under the identical tile
convention (`raw_reference.py`) — the interpretive anchor tying panel
numbers back to the screen. Drawn as dashed reference lines in the
figure, labelled raw, never as panel cells.

## 7. Frozen predictions (falsifiable, scored in the LOG either way)

- **P1 (the hunt's target pattern).** TXC-pre recovery rises with T
  through T = 8 and exceeds both token archs beyond the seed spread at
  some T ∈ {4, 8, 16}.
- **P2 (aggregation is post's native shape).** Budget-matched TXC-post
  also rises with T and is ≥ TXC-pre at T ∈ {8, 16} (within seed
  spread counts as confirmed): a single shared window code IS an
  order-free pooled summary, which is what this latent is.
- **P3 (per-token-blind carries over).** Token archs land low, near
  the raw per-token reference (the code can add nothing the position
  lacks), and below the best window arch's T ≥ 8 cells.
- **P4 (learned T-dependence).** Window-arch trained − untrained
  margin grows with T (untrained ≈ flat).
- **P5 (shape).** Rise is monotone through T = 8; T = 16 is
  pre-registered as ambiguous (§ 6) — a drop there is read via the
  diagnostic, a rise counts toward P1/P2.
- **Stacked risk (named, not predicted):** the λ̂ panel's large-T
  pathology (trained < untrained at T = 16) may recur; if it does it is
  recorded as a training pathology, not evidence for or against.

## 8. KEEP/KILL (frozen)

- **KEEP (deliverable earned)** iff a window arch (TXC-pre or matched
  TXC-post) exceeds BOTH token archs' recovery beyond the combined
  seed spread at ≥ 2 window T values with a rising T-trend
  (T = 8 > T = 2 for that arch), AND its trained − untrained margin
  grows with T. Phrasing is variance-aware by construction: mean ± sd
  over 3 seeds, margins quoted against the spread (n = 3 — the λ̂
  review's binding note).
- **NEGATIVE (recorded, not spun)** if no window arch clears both
  token archs beyond the spread at any T, or window recovery is flat
  or falling in T over {2, 4, 8}.
- **VOID (post cells only)** if matched post's untrained cells do not
  realize l0/token = 8.00 ± 0.02 at every T — then the k·T correction
  is wrong (runpod-d's § 6 falsifier) and post cells are reported as a
  failed amendment, with the rest of the panel unaffected.
- Realized l0 is read per cell next to every recovery number; any
  trained cell outside [5.0, 8.0] l0/token is carried into the reading
  as a residual mismatch, not smoothed over.

## 9. Deliverables

The second real-task T-scaling figure (`figs/stage2_tscaling.*`, one
line per arch, seed band, **realized-l0 annotation on every TXC-post
point** — mandatory before external use, per the review's binding
note), `results/stage2_ward_real_slope8_distill_l14.json`, a LOG
verdict scoring P1–P5, and a record section carrying § 5's convention
sentence verbatim. Leaderboard hygiene: 0 duplicate eval_keys, 0 null
metrics, clean tree.

---

## 10. Pre-run amendment — reconciliation with `LEVEL_CARD_DRAFT.md` (frozen at commit; still no cell executed)

runpod-b's draft card (`LEVEL_CARD_DRAFT.md`, commit `c9d0ac36`) landed
while § 1–9 were being committed (`fff7877c` rebased atop it). Per
protocol (round-1 review note 4) the running agent's card governs; this
amendment reconciles the two BEFORE any cell runs. Git order remains
the evidence.

**ADOPTED from the draft (pre-registered here):**

1. **Stage-2 shuffle-immunity receipt** (the draft's P3, attached to
   the panel itself rather than only to the Stage-1 screen). Post-hoc
   diagnostic on the SAME trained checkpoints, OFF-leaderboard: encode
   eval/train tiles with an **anchor-fixed within-tile context
   shuffle** (permute tile slots 0..T−2 per row, leading edge fixed at
   T−1, seeded rng 1234 — the screen's convention), refit + evaluate
   the slope8 probe on the shuffled codes, for TXC-pre and matched
   TXC-post at T ∈ {8, 16} (and the best window arch elsewhere if
   different). **Prediction (frozen): recovery is retained — the
   shuffled cell keeps more than half of that cell's (clean window −
   best token arch) margin, per seed-mean.** A larger degradation
   FALSIFIES the aggregation framing of this panel's result (order
   would matter after all), and the record must say so.
2. **Position-only floor reference**: LinearRegression from
   leading-edge position features alone (position index p and p² on
   the 128-token grid) → held-out Pearson r vs slope8, same sampling +
   finite mask, off-leaderboard, reported next to the raw references.
   **Prediction (frozen): low — well below raw_mean at T ≥ 8;** if it
   is not, the ambient position ramp explains part of every arch's
   recovery and the reading must discount it.
3. **Record obligation**: the tested ladder sits at the BOTTOM of the
   effect's raw T-range (the screen's window-mean gap still grew at
   T = 64); carried verbatim into the record and LOG verdict.

**REJECTED from the draft, with the reason stated:** the draft's
primary target (*continuous window-mean hedge level over the trailing
T-token window*) makes the label a function of T — every T-cell would
recover a DIFFERENT quantity, confounding the label with the
architecture axis and making a single T-scaling line uninterpretable.
This card's primary (the frozen slope8 grid) has FIXED ≈ 128-token
support at every T, so T scales *coverage* of one target — the same
design shape as the reviewed λ̂ panel (kernel support ≫ window), and
the quantity whose screen numbers the r2-e briefing itself cites as
the seed. The draft's aggregation FRAME is fully retained (§ 1); its
per-token-first triage (P4) is satisfied by the round-1 screen
(per-token 0.468 vs chance 0.333, under anchor matching) plus the
pre-registered `raw_tok` reference.

Diagnostic scripts are committed before they run (panel first; the
specs above are the frozen science).
