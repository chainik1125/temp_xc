# runpod-c STATUS — bring-up briefing (written by mac-local, 2026-07-27 ~18:40 London)

**You are `runpod-c`** — the T-SCALING HILL-CLIMB agent, alone on a
dedicated 2×H100 pod (52 CPU / 503 GB / 1 TB; both GPUs yours).
Workspace `/workspace/agents/runpod-c/temp_xc`, venv `.venv` (check
`tail /workspace/venv.log`), tokens `/workspace/.tokens/` (gh, hf,
hf_datasets; NO Modal creds by design), `export
HF_HOME=/workspace/hf_cache`. Ledger: `RUNPOD` section of
`briefings/MODAL_SPEND.md`, $150/day default cap unless Han raises.

**Read order:** CLAUDE.md → `agents/README.md` →
`briefings/actmix-shared.md` (topology; stamp from `date`) → LOG
tail from `c1c5c949e` forward (skim for: the k-inversion entries
`fcf62963b`+ratification, the sparsity-convention audit, the txc_pro
first-pass archaeology in the 18:45 meeting entry, Dmitry's
dead-latent mechanism quoted there) → `experiments/probing/actmix/`
(CARD.md + analysis.py — the eval you must eventually win on).

## Mission (Dmitry, 07-27 meeting; Han directive)

**Make TXC T-scaling actually improve with window size on sparse
probing** — custom loss, training tricks, whatever works. The
btk-only truth today: TXC-pre at k=20 DECLINES with T
(0.9264 → 0.9033, T1→T16, 36-task CT-excl); at k=5 it only
recovers to the SAE band. Dmitry's mechanism hypothesis: ReLU-era
dead latents were one cause, but under BatchTopK the residual
problem is capacity/granularity allocation across the window.

## Program design (binding structure, yours to fill)

**A. Pre-register the split FIRST (before any hill-climbing):**
- **DEV**: 8 probing tasks you choose from the 36 (state them),
  1 seed (42), T ∈ {1, 4, 16}, k = 20 — the iteration signal.
- **HOLDOUT**: the remaining 28 tasks + seeds {1, 2} + the full T
  grid — touched ONLY by finalist validation runs. The honesty of
  the whole exercise rests on never climbing on the holdout.
- Commit the split as a card (`experiments/explorations/tscale/`
  or similar) BEFORE the first candidate trains.

**B. Pyramid screening (iteration speed is the game):**
- L0 (seconds): training-health metrics — dead-latent fraction,
  realized l0 utilization, per-position recovered variance vs T.
- L1 (~minutes): dev-split probe eval at T {1,4,16}.
- L2 (~30 min): dev split full-T, 1 seed.
- L3 (hours, finalists only): full holdout validation, 3 seeds,
  full T — this is the number that gets reported.

**C. Candidate menu (start here, extend freely):**
1. **The txc_pro recipe, reimplemented from its locked hparams**
   (`origin/final-aniket:purified/configs/locked_archs.yaml`,
   tag `phase5b_subseq_h8` — the class file was LOST in
   purification; mac-c may recover the original, fold it in when
   they do, but do NOT wait): d_sae 18432, T_max 10,
   **t_sample 5 = train on resampled sub-windows** (mixed-T
   curriculum), n_matryoshka 8, contrastive_shifts [1,2] w/
   inverse-distance weighting, auxk_alpha 0.03125,
   dead_threshold 10M tokens. Ablate its components separately —
   which ingredient carries the T-scaling?
2. Dead-latent mitigations under BatchTopK: auxk loss, ghost
   grads, dead resampling schedules.
3. Sparsity scheduling: anneal k during training; per-position k
   floors (guarantee every position a minimum budget).
4. Mixed-T training / T-curriculum (train T sampled ≤ T_max,
   serve at fixed T) — the resampling trick isolated.
5. Decoder-norm / per-position normalization variants.
6. Loss reweighting across window positions (near-edge vs far).
- New checkpoints: `eval_extra`-namespace every cell (the
  documented grid.py mechanism) — NEVER collide with quoted rows.

**D. Discipline:**
- This is ARCH R&D, not claim production: no result here enters
  any rebuttal/writeup surface without a full L3 holdout run +
  a proper card + mac-local ratification.
- Log every candidate (config hash, L0–L2 numbers) in an
  append-only results file; negative results are data.
- Canonical runner for anything leaderboard-bound (hard rule 1);
  scratch training loops are fine for L0/L1 iteration.
- Budget: ledger line per session; pod ≈ $6/h both GPUs.
- LOG entry (PTR) at: split freeze, first L2 signal, any L3 run.
  mac-local reviews on push.

## House rules

Pull-rebase before push; LOG conflicts keep BOTH blocks +
stray-marker grep; stamp from `date`; PTR everything; pods have no
Modal creds; the probing quoted numbers are NEVER touched by your
cells (eval_extra namespacing).

*Rewrite before any compact. — mac-local*
