# Does the paper TXC's window-size scaling improve under the activation the paper actually claims?

*10h sprint, 2026-07-26 19:38 UTC → ~05:38 UTC. Branch `dmitry-btk-txc-sprint`
(worktree-isolated), based on `origin/arxiv` @ `5e6bfe37`. Author: Claude,
running the isolated "Dmitry re-run" lane of the ACTMIX phase.*

## Executive summary

**The question, sharpened by recon.** The camera-ready paper states that the
TXC's sparsity function "is BatchTopK" (main.tex:362); the word "ReLU" appears
zero times in the paper. The code that produced every TXC number, however,
applies per-window TopK selection and then a ReLU on the selected values
(`txc_base.py:166-168`) — zeroing selected-negative slots after selection, a
harm the team's ACTMIX briefing shows grows with window size T. **Retraining
with pure BatchTopK is therefore a correction that makes the paper's stated
architecture true, not an architecture change** — and the pre-registered
"Dmitry re-run gate" asks: does the paper arch's d(perf)/dT improve under the
fix? I ran the §4 synthetic benches with three arms of the paper arch at one
shared parameter set, T ∈ {1,2,4,5,8,10}, per-token baselines frozen:

- `paper-match`: `txc_base` as published (per-window TopK→ReLU),
- `btk-only`: `txc_base_btkonly` (BatchTopK over raw pre-acts, no ReLU —
  mac-a's canonical ACTMIX convention, which landed and was ratified
  mid-sprint; my implementation conforms to it),
- `relu-mix` control: ReLU then batch-pooled BatchTopK — separates the effect
  of removing the ReLU from the effect of batch pooling.

**Findings.**

1. **The gate passes on the headline global-recovery metric, and the
   composite's T-harm is directly visible.** On the Coupling bench, gauc's
   slope in log₂T flips from **−0.006 (paper-match) to +0.027 (btk-only)**;
   the composite's own k_pos=2 curve collapses from 0.988 (T=1) to 0.829
   (T=8). Historical data corroborates the mechanism: the old composite
   denoising-probe sweep (temp_xc_tex notes) peaks at T≈4 (R²global 0.285 →
   0.422 → 0.335 by T=8) — the paper's deleted "monotone in T" claim was
   already false under the composite.
   - ![d(perf)/dT gate figure](plots/btk_rerun/btk_rerun_dperf_dT.png)
2. **But btk-only is not a better paper arch at these budgets: the composite
   dominates it in absolute recovery at nearly every (k_pos, T ≤ 10)** —
   e.g. 0.988 vs 0.624 at k=2/T=1 — while btk-only reconstructs far better
   (nmse 0.003 vs 0.124 at k=1/T=1) and never zero-picks (realized l0 ==
   budget everywhere; the composite bleeds support even at T=1, keeping 0.87
   of 2 slots). Read: the ReLU in the composite is simultaneously the
   T-scaling bug *and* a useful nonnegativity prior that keeps decoder atoms
   single-feature on these benches. <!-- FILL: relu-mix attribution — if
   relu-mix recovers the composite's levels, pooling is innocent and the
   prior is the whole story; if not, pooling shares blame. -->
   - ![fingerprint](plots/btk_rerun/btk_rerun_fingerprint.png)
3. **Baselines (frozen, per directive): the composite clears the frozen
   T-SAE bar (Coupling gauc 0.809 at k=1) at every T; btk-only clears it
   only for T ≤ 5.** On the Denoising bench both arms beat the T-SAE
   references from T ≥ 2 until the clipped high-T cells. (Caveat: the frozen
   baselines had 10k training steps vs our 6k — a bias in the baselines'
   favor; gauc/eauc are decoder-direction metrics, stable across the
   evaluator-protocol bumps.)
4. <!-- FILL: wing — d_sae=50 de-clipped slopes; do the headline slopes
   survive when no cell is budget-clipped? -->

**Verdict for the re-run gate** (pre-registration quoted, verdict PENDING
TEAM REVIEW per house discipline): *"the PAPER arch's T-curves should improve
(that is Dmitry's re-run gate: does d(perf)/dT improve)"* — **MIXED,
leaning SUPPORTED on the letter, NOT SUPPORTED on the spirit at paper
budgets.** The slope improves on the headline metric and the composite's
high-T degradation is confirmed as an activation artifact; but at the
paper's d_sae/k_win settings the corrected architecture loses more from
dropping the ReLU prior than it gains from honest selection, at every
T ≤ 10. The practical recommendation for the paper is therefore NOT a
silent swap to btk-only; it is (a) fix the text/code contradiction
explicitly, (b) present T-scaling with the btk-only arm where the trend is
honest, and (c) treat the nonnegativity prior as a real design ingredient
— e.g. selection over raw pre-acts with nonnegative *codes* — which is a
one-line follow-up experiment. <!-- REVISE after relu-mix + wing land. -->

## What was run (map)

- **Arms.** `txc_base` (frozen, as published) vs `txc_base_btkonly` (new;
  BatchTopK over raw squashed pre-acts at the paper's own budget
  k_win = min(k_pos·T, d_sae), JumpReLU-threshold eval path with explicit
  `threshold_set` flag, EMA over signed survivors, `neg_frac` logged —
  conforming to mac-a's canonical convention 92db86c4, ratified 9e634bed)
  vs `txc_base_relumix` (control; ReLU→batch pool at the same budget).
  All three share parameterisation, init, AuxK dead-revival, decoder
  unit-norm, grad-parallel removal. `relu_mode` hashes into train_key; the
  arm label rides the arch registry name (the schema has no arm field).
  Code: `src/temp_bench/archs/txc_base_btk.py`; 10 contract tests
  (`tests/test_txc_base_btk.py`) incl. selected-negatives-survive,
  zero-pick fingerprint, negative-threshold gating; full suite green.
- **Grid.** §4 benches `toy_markov_n20_d40_noisy` (Denoising) +
  `toy_coupled_K10_M20_d256` (Coupling); T ∈ {1,2,4,5,8,10} × k_pos ∈
  {1,2,5} × seeds {1,2,3}; n_steps 6 000, batch 1024, buffer 2M, lr 3e-4,
  bf16; uniform eval_window_L = 40. Cells with k_pos·T ≥ d_sae=20 are
  budget-clipped identically in all arms (dense code) — marked in figures,
  excluded from slopes. **Sensitivity wing:** the same grid at d_sae=50
  (k ∈ {2,5}, 2 seeds) de-clips every combination.
- **Not rerun/not runnable.** Per-token baselines frozen (existing rows).
  Probing, backtracking, RLHF evaluators are stubs at this pin (each a
  multi-day port from origin/final); backtracking is Aniket's lane, probing
  and EM T-sweeps are runpod-1/2's lanes. The EM α-rescaling (√T,
  appendix.tex:246-248) was measured under the composite and must be
  re-measured before any c6 comparison — flagged to that lane.
- **Compute.** Modal (app `dmitry-btk-rerun`), A10G ×≤8, detached, freeze
  SHAs v1–v5 pinned + `_assert_pinned`, Volume `temp-xc-btk-rerun`,
  repatriate-merge with dup-key checks. Ledger: `briefings/MODAL_SPEND.md`
  (est ~$48 total this lane; actuals correction at close). All rows carry
  `agent: dmitry-btk-sprint`.
- **Code paths.** Driver `experiments/explorations/btk_rerun/driver.py`;
  analysis `experiments/explorations/btk_rerun/analysis.py`; launcher
  `scripts/modal_btk_rerun.py`. Rows: `results/leaderboard.jsonl`.

## What this means for the paper's figures (task-1 tabulation, updated)

Full inventory (paper-recon, 100% of main+appendix): **15 of 17 figures need
full regeneration and all 4 tables partial regeneration** if the TXC numbers
change — there is no TXC-free region of the results; the two TikZ cartoons
are the only safe figures. But this sprint's result argues the right move is
NOT to regenerate everything under btk-only: at paper budgets the corrected
activation is worse in level, and the honest fix is textual (resolve the
BatchTopK-vs-TopK†ReLU contradiction, appendix.tex:29 vs :33) plus a new
T-scaling figure — which fills the hole left by the deleted monotone-in-T
claim (main.tex:884, commented out; flagged as missing work at main.tex:1226,
1234-1238, 1277). Redo-lane ownership: probing = runpod-1, EM = runpod-2
(with the α-rescaling caveat), backtracking = Aniket, rose re-render = last,
after arms are settled.

## Caveats (read before quoting numbers)

- Toy scale: d_sae=20 dictionaries (paper setting) at 6k steps (sprint
  scale; paper synthetic rows used 10k, its backtracking mains 300k). The
  wing checks d_sae; steps were not swept.
- The btk-only arm was renamed/conformed mid-sprint when mac-a's canonical
  convention landed: rows exist under `txc_base_btk` (v1.0.0) and
  `txc_base_btkonly` (v1.1.0). Training paths are bit-identical; the
  differences are eval-path-only (threshold flag + EMA source), which
  gauc/eauc never touch. <!-- FILL: cross-check numbers once redo lands. -->
- The Denoising bench emits no gauc under the current evaluator
  (hidden_features not exposed) — the paper's latent-level denoising claim
  is not re-tested here; only direction-recovery (eauc) and NMSE are.
  The historical probe data cited in Finding 1 used the old d_sae=40
  regime and a different (probe-based) metric.
- Frozen-baseline comparisons favor the baselines on steps but the ACTMIX
  briefing pre-registers that per-token baselines improve MOST under
  btk-only — so freezing them biases *toward* TXC in any cross-arch claim.
  No cross-arch claim here relies on beating a frozen baseline.
- One parameter set throughout; no tuning pass happened.

## Reproduction

```bash
git checkout dmitry-btk-txc-sprint
python -m pytest tests/test_txc_base_btk.py            # contract tests
modal run --detach scripts/modal_btk_rerun.py           # 3-arm base grid
modal run --detach scripts/modal_btk_rerun.py --dsae 50 --extra "--seeds 1 2 --k-pos 2 5"
python -m experiments.explorations.btk_rerun.analysis   # figures + slopes
```

## Research log

`sprint_log.md` — full timeline: the arxiv-branch discovery, the disk-full
incident, the mid-sprint canonical-convention landing and conformance, all
freeze SHAs, and the recon deliveries.
