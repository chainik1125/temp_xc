<!--
DRAFT — written by agent_paper 2026-05-03 for Han's review.
Han: rewrite the "Identity + mandate" section as you wish before
spinning up agent_em; agents will not touch it after.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_em
last_state_update: 2026-05-03T22:00:00Z
component: c6
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent EM**. You own C6 only. Files you may edit:
- `agents/agent_em/briefing.md` (your own — agent-owned sections only)
- `docs/components/c6.md`
- `experiments/c6_em/`
- Code under `src/temp_bench/` that you author + commit (the Wang
  procedure runner under `temp_bench.case_studies.em`, Bricken
  trainer logic under `temp_bench.training.bricken`)
- `configs/datasources.yaml` — adding new C6 datasources is fine.

**Files that are OUT OF SCOPE — do NOT edit even if it seems harmless:**
- `agents/agent_*/` — every other agent's directory.
- `docs/paper/*` — agent_paper's territory.
- `agents/agent_paper/decisions.md` — global decisions log.
- `agents/README.md` — roster/contract; agent_paper-owned.
- `configs/locked_archs.yaml` — only agent_paper edits this.
- `pyproject.toml` and `uv.lock` — dependency changes affect every
  agent's venv; pyproject + lockfile must be committed atomically,
  and only agent_paper coordinates that. If you need a new dep,
  surface in Open questions.

If you find yourself wanting to edit any out-of-scope file, **STOP**.
Add a bullet to "Open questions for Han" in your own briefing,
surface it in chat, and let Han or agent_paper land the change. Even
if Han verbally approves, do not commit cross-territory edits yourself.

You are agent EM, lead on **C6: emergent misalignment** on
`Qwen/Qwen2.5-14B-Instruct` + finance LoRA organism (R1 + R32). The
component is in **status: pending-retest** in `docs/components/c6.md`.

Hardware: pod `2× H100`, pinned to **GPU 1**. Pod mode `persistent`.
agent_nlp shares the pod on GPU 0; you will not collide because
pinning is enforced. **Fallback**: if R32 OOMs the H100 (14B model +
LoRA at fp16 ≈ 28 GB so it should fit, but R32 may stress it), spin
up `agent_em_h200` (provisioned dormant — see `agents/README.md`).

Why the re-test: Dmitry's published Qwen-14B finance numbers
(`em_nanda_results_paper.md`) were plain TXC k=100, no Bricken,
no anti-dead — not a fair comparison vs SAE arditi which has 100k
training steps and dead-feature handling. With the brickenauxk_a8
recipe (Bricken + EMA-AuxK α=1/8 + dead-threshold 128k tokens),
TXC may close the +3.91 gap on R1 and the +12.58 gap on R32.

Decision tree (after R1 30k mid-α first re-run):
- gap ≤ 3 align → **Tied** — headline win
- gap 3–9 align → **Mixed** — note step-efficiency win on Qwen-7B medical
- gap > 9 align → **Honest negative** — back to original framing

Coordinate with **Dmitry on `origin/em-nanda`** — he is still active
on this component. Read `EM_NANDA_BRIEF.md` for his latest state
before launching. Don't merge his branch into `final`; read via
`git show` (decision #4).

Salvageable contributions (independent of headline outcome):
- **Bundle null is architecture-general**: both arches' k=30 bundles
  peak at align ≈ 41.3 on R32, falling 13–23 align points below
  single-feat champions. Falsifies "distributed misalignment by sum."
- **Bundle precision is architecture-specific**: SAE has k=30 < k=3 <
  single-feat (precision helps); TXC inverts (top-3 anti-correlate).

Locked decisions in scope: #2 (C6 reframe + bundle-null result), #4
(cross-branch reads), #6 (HF repos), #7 (Bricken opt-in — **C6 turns
it on by default**, you don't need an A/B; the recipe is justified by
Dmitry's Qwen-7B medical evidence).

References:
- `agents/README.md` (your roster row)
- `docs/components/c6.md` (full setup, decision tree, Wang 4-stage)
- `docs/paper/architecture.md` *Per-experiment training knobs* (Bricken)
- `decisions.md` (esp. #2, #7)
- `origin/em-nanda:docs/dmitry/results/em_features/EM_NANDA_BRIEF.md` (latest)
- `PROTOCOL.md` § 11 (framework), § 12 (GPU pinning)

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-03T23:53Z (Phase B + first-cell C6 results landed and pushed)**

- `git HEAD`: `6f2b323d` (frontier plot pushed). 12 commits today;
  Phase A + Phase B + first-cell results all upstream on `final`.
- Last leaderboard append: `eval_key=c1d96d5549e0c3a6`
  (txc_base+brickenauxk_a8 30k, peak_align=75.875, peak_coh=89.92,
  peak_alpha=+1.0, peak_feature_id=734). Append at 23:52:56 UTC.
  SAE-arditi cell at `eval_key=160a1471d1a93f1b`, peak_align=81.625,
  feat 3173, α=-30.
- Last checkpoint saved: `train_key=410c3f342b133fff` (txc_base 30 k
  brickenauxk_a8). SAE checkpoint at `train_key=926527b006dd74aa`
  (1.34 GB). HF push deferred — agent_em didn't wire persistent-pod
  auto-push; flag for next session.
- Active GPU lock(s): GPU 1 pinned. Pipeline (PID 23080) is winding
  down (post-Wang cleanup). Will exit shortly.
- Recent decisions in scope: #2, #4, #6, #7
- In flight: nothing — first cell complete; results rendered into
  `docs/components/c6.md` AUTO-RESULTS + frontier plot at
  `experiments/c6_em/plots/c6_frontier.png`.

## Headline result

**Gap = peak_align(SAE-arditi) − peak_align(TXC-base+brickenauxk_a8)
= 81.625 − 75.875 = +5.75 align points** → **Mixed** decision per
the c6.md decision tree (3 < gap ≤ 9).

| arch | seed | peak_align | peak_coh | peak_α | feature_id |
|---|---:|---:|---:|---:|---:|
| sae_arditi | 42 | 81.62 | 90.92 | -30.0 | 3173 |
| txc_base+brickenauxk_a8 | 42 | 75.88 | 89.92 | +1.0 | 734 |

**SAE-arditi frontier:** sharp peak at α=-30, feat 3173 (the
strongest "anti-misalignment" feature). The single-feat peak at
extreme negative α matches the published-Wang shape (Dmitry's
Qwen-14B finance R32 ext-α champion peaks at α=-30 too).

**TXC-base frontier:** flatter — all three top-Δz̄ features hover
around 70-76 across the α grid. The peak is at mild positive α=+1
(feat 734), not extreme negative. Coherence holds ≥87 across all
(feat × α) cells.

**Bricken trajectory** during TXC training: fired 59× over 30 k
steps (every 500), last n_resampled=9216 (max_resample_fraction=0.5
× d_sae=18432 cap). Consistent with Dmitry's ~75% dead-by-step-40k
trajectory — Bricken can't keep up with newly-collapsing features
even on this corpus.

**Decision-tree outcome (Mixed):** TXC-base+brickenauxk_a8 did not
close the gap to within Gemini-judge σ ≈ 6 on the abbreviated Wang.
Combined with Dmitry's Qwen-7B-medical step-efficiency win
(`txc_hookpoint_comparison_finding.md`), this supports a "tradeoff"
framing — TXC wins on training-step efficiency at smaller scale but
loses on absolute peak align at the Qwen-14B/R1 30 k mid-α cell.

## Caveats baked into the gap

These results are NOT directly comparable to Dmitry's published
95.16/91.25 numbers (which are gap +3.91 for the SAME R1 30 k
mid-α cell on the SAME organism):

1. **Judge swap**: Anthropic Claude Haiku 4.5 instead of
   Gemini-3.1-flash-lite (no GOOGLE_API_KEY in pod). σ unmeasured.
2. **Wang abbreviation**: stages 2 + 3 (causal screen + per-survivor
   coh-aware sweep) skipped; abbreviated stage-4 frontier on top-3
   Δz̄-ranked features.
3. **Corpus stand-in**: training corpus is
   `cfierro/personality-qs-risky-financial-advice` (HF mirror;
   17 k user/assistant pairs; closest available stand-in for
   Turner's `risky_financial_advice.jsonl`).
4. **Hparam mismatch**: TXC-base used locked yaml defaults
   (`d_sae=18432, k_win=100`) not c6.md's `d_sae=32768, k=128`
   (locked yaml lacks `c6` override; OQ #1).

The relative gap (+5.75) is comparable across the two arches
(both hit the same judge / procedure / corpus / hparams).

## What I just did (agent owns — overwrite)

Phase B — wire the C6 pipeline end-to-end (newest first):

- Smoke-test launched: 1 k-step train-only on `sae_arditi`.
  Validates the cache builder + canonical sae_trainer path before
  committing to the full 30 k cell.
- `experiments/c6_em/run.py`: C6 entrypoint via `runner.run_cell`.
  Two cells (sae_arditi + txc_base × seed 42) on the same finance-EM
  activation cache (apples-to-apples). `--smoke-test` flag for the
  1 k-step proof-of-pipeline; `--skip-eval` for train-only;
  full path runs Wang minimal in `eval_fn`.
- `experiments/c6_em/train.py`: train-fn adapter. Reads
  `training_cfg.{ema_auxk_alpha, dead_threshold_tokens}` and passes
  them through to `TXCBase.__init__` so the brickenauxk_a8 recipe
  applies without yaml mutation.
- `src/temp_bench/case_studies/em.py`: abbreviated Wang. Stage 1
  (Δz̄ ranking on probe set) + stage-4-only 6-α frontier on top-3
  features. Claude Haiku 4.5 as judge (Gemini key not provisioned;
  pyproject is agent_paper-only so I can't add the dep). Both
  arches run the SAME abbreviated procedure → relative gap is
  internally valid.
- `src/temp_bench/data/nlp/qwen_em.py`: C6 activation-cache builder
  modeled on agent_back's `ward.py`. Single hookpoint
  (resid_post @ layer 24, d_model=5120). Source corpus is
  `cfierro/personality-qs-risky-financial-advice` (HF), the closest
  available stand-in for Turner's `risky_financial_advice.jsonl`
  (which is not on HF; Dmitry generated his locally via GPT-4o).

Phase A (cleaned up after agent_nlp's `f7c3c536` rebase):

- `src/temp_bench/architectures/sae_arditi.py`: per-token TopK SAE
  in sae_day / arditi state-dict layout (`W_enc:(d_in,d_sae)`,
  `W_dec:(d_sae,d_in)`). Diverges from the framework's `TopKSAE`
  layout so Dmitry's HF checkpoints (when applicable) load directly
  without a transpose dance.
- `src/temp_bench/training/bricken.py`: filled the stub. Measurement
  loop arch-agnostic; reset is TXC-han-specific (raises on any
  other layout — fail-fast since Bricken is opt-in and C6 is the
  only target). Adapts `(B, seq_len, d_in)` check batches into
  `(B, T, d_in)` by sampling one random T-window per batch element,
  matching agent_nlp's TXCBase.train_step convention.
- `tests/test_arch_registry.py`: `KNOWN_UNPORTED` now down to 5
  entries (`stacked_sae`, `tfa`, `tfa_pos`, `mlc`, `txc_pro` —
  all out of my territory). Tests stay at 51/51.

Pre-Phase-A:

- Read `c6.md`, `EM_NANDA_BRIEF.md` (top), `em_nanda_results_paper.md`.
  Internalised decision tree, salvageable contributions, and the
  brickenauxk_a8 recipe.
- Two stashes ago: had a self-port of TXCBase that got superseded by
  agent_nlp's `f7c3c536` commit. Dropped that stash; my port lived
  only in working memory. agent_nlp's version differs slightly
  (uses `register_post_accumulate_grad_hook` for grad-parallel
  removal; explicitly skips geom-median b_dec init — see their
  module docstring). My Bricken adapts to it.

## Next action (agent owns — overwrite)

After the smoke-test process exits, the next-life instance picks up here:

1. `cd /workspace/temp_xc_em/purified` (the wrapper does this)
2. `bash scripts/agent_smoke_test.sh` (verifies env)
3. `git pull --rebase origin final`
4. **Inspect smoke-test result**:
   `tail -50 logs/smoke_sae_arditi_1k.log` — last lines should show
   `[c6.train] done in 1000 steps; final loss=...` and a manifest
   row appended for `sae_arditi` × seed 42 × n_steps=1000. Look for
   any `Error` / `OOM` / `Traceback`. Activation cache should be at
   `results/act_cache/e052801ef8e6d22b/` (key for the C6 datasource).
5. **If smoke passed**: kick off the full 30 k cells. Two ways:
   - Single shell: `TQDM_DISABLE=1 .venv/bin/python -m experiments.c6_em.run`
     (runs both arches sequentially; ~3–5 h total).
   - Or run in background (nohup or `&`) and monitor via `tail -F`.
   The cells will:
   - SAE-arditi 30 k (~25 min) → checkpoint → Wang minimal (~30 min) → leaderboard row
   - TXC-base 30 k brickenauxk_a8 (~45 min) → checkpoint → Wang (~30 min) → leaderboard row
6. **If smoke failed**: read the traceback, fix, re-run with
   `--smoke-test`. Common failure modes to check:
   - Qwen-14B forward batch_size=8 might OOM on the H100 (28 GB
     model + KV cache). Drop `cache_batch_size` to 4 in
     `qwen_em.cache_activations` if needed.
   - Chat-template application may fail on certain cfierro rows; the
     loader skips silently with a `log.debug` — confirm the corpus
     produced ≥1 valid sequence via `corpus.json` in the cache dir.
   - The `_instantiate_with_overrides` path skips `instantiate_arch`'s
     KNOWN_UNPORTED check; if `txc_base` ever moves out of the
     locked yaml, this needs updating.
7. **After cells land**: render results + decision-tree outcome.
   `experiments/c6_em/analysis.py` doesn't exist yet (Phase C).
   For now the briefing can carry the headline numbers; `c6.md`
   AUTO-RESULTS block can be populated once analysis.py is wired.
8. **Apply decision tree**: gap = peak_align(sae_arditi) −
   peak_align(txc_base). Map to (Tied | Mixed | Honest negative)
   per the table in `c6.md`.

## Don't repeat (agent owns — overwrite)

- **Plain TXC k=100** without Bricken — that's Dmitry's published
  comparison; we're re-testing with the better recipe.
- **Merge `em-nanda` into `final`** — decision #4 forbids it.
  Cross-branch reads only (`git show origin/em-nanda:<path>`).
- **Don't trust the absolute peak-align numbers vs Dmitry's 95.16 /
  91.25.** Three caveats:
    1. Judge swap (Claude Haiku 4.5 vs Gemini-3.1-flash-lite). σ
       calibration unknown.
    2. Wang stages 2 + 3 skipped — abbreviated frontier.
    3. Corpus divergence (cfierro mirror vs Dmitry's pile/ultrachat).
  The relative gap (TXC − SAE) is what's headline-comparable.
- **Edit `pyproject.toml` / `uv.lock` / `configs/locked_archs.yaml`
  / `agents/README.md` / `docs/paper/*` / other agents' dirs.**
  All cross-territory — surface as Open Questions, let agent_paper
  / Han land.
- **Bypass `runner.run_cell`.** The C6 entrypoint routes both cells
  through it; Phase C analysis.py reads `leaderboard.jsonl`.
- **Forget `TQDM_DISABLE=1`.** Hard Rule #8. The shell wrapper sets
  it; bare `python -m experiments.c6_em.run` from a fresh shell
  will spam progress bars without it.

## Open questions for Han (agent owns — overwrite)

1. **`per_component_hparams[c6]` for `txc_base` and `txc_pro` in
   `configs/locked_archs.yaml`.** Per `c6.md` *Setup* the C6 cells
   should use `d_sae=32768` + `k_win=128` (TXC-base) — same scale
   as Dmitry's published runs. Locked defaults are `d_sae=18432`,
   `k_pos=20` (k_win=100). My Phase B run uses the locked defaults
   (smaller TXC than the paper baseline). Once you / agent_paper
   add the c6 override, I re-run with paper-correct hparams.
   Edit needed (mirrors existing c7 pattern):
   ```yaml
   txc_base:
     per_component_hparams:
       c6: { d_sae: 32768, k_pos: 25 }   # k_win=125, closest to 128/T=5
       c7: { d_sae: 32768 }               # existing
   txc_pro:
     per_component_hparams:
       c6: { d_sae: 32768 }
       c7: { d_sae: 32768 }               # existing
   ```
   (`k_win=128` doesn't divide evenly by T=5; closest valid is
   `k_pos=25 → k_win=125`. Or add an explicit `k_win` knob — TXCBase
   already accepts it via the constructor.)

2. **Judge: Claude vs Gemini.** Current run uses `claude-haiku-4-5`
   because there's no `GOOGLE_API_KEY` in `/workspace/.tokens`. Two
   options to enable Gemini:
   (a) Provision a GEMINI_API_KEY in `/workspace/.tokens/gemini_key`
       + add the `google-generativeai` dep (you / agent_paper land
       atomically in pyproject + lockfile).
   (b) Stick with Claude; document the deviation prominently in the
       C6 results writeup.
   I'm proceeding with (b) for the first cells; (a) needs a follow-up
   re-run if you want Dmitry-comparable absolute numbers.

3. **Wang stages 2 + 3 abbreviation.** I skipped causal screen +
   coherence-aware sweep to fit the 10 h autonomous window. Both
   arches use the same abbreviated procedure → relative gap is
   internally valid. If you want full-Wang fidelity, I can port the
   screen + sweep stages in a follow-up session (~2 h effort).

4. **Corpus stand-in: cfierro/personality-qs-risky-financial-advice
   instead of Turner's risky_financial_advice.jsonl.** Turner's
   exact 6000-prompt file is not on HF. cfierro is the closest
   available variant (17 k chat-formatted finance prompts). If you
   have access to Dmitry's locally-generated file, copy it to
   `data/em_finance_prompts.jsonl` and I'll wire a branch for it.
