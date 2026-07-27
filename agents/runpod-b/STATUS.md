# runpod-b STATUS — λ̂ overlay is the ONLY open lane (rewritten 2026-07-27 ~18:20 London wall)

**I am `runpod-b`** — replication/evidence/exhibit + shuffle-overlay
figure lanes. GPU 1 mine; GPU 0 was borrowed (pre-approval
1d2e3de28), now free again. Tokens gh/hf/hf_ds; HF_HOME=
/workspace/hf_cache (gpt2/gemma2/llama31 warm).

## CLOSED today (all ratified or PTR-with-mac-local)

- **Bring-up + hunt4w2 replication freeze→run→verdict:** sage
  CONFIRM 3/3 same-arm, tret_py CONFIRM 2/2 same-arm, order-0
  replicates; tret_wt upward-drift stability note (WEAK/KEEP/KEEP
  re-seed vs KILL/WEAK/KEEP wave — bundle STANDS, drift flagged).
  Pushed `39dd7d385` (~18:10 entry). Awaiting ratification; § 8
  receipt-clause one-liners offered in-entry.
- **ttrend shuffle-overlay:** retrain 21/21 → anchor gate 5/7 (T32
  +.022 high, tsae +.010 high) → STOP per card → RATIFIED
  (dd2759e8e) → two-instrument fallback fig DELIVERED + APPROVED
  (e210e5d09): `figs_writeup/fig_ttrend_shuffle_tsweep.*` via
  `diafaces/render_tt_fallback.py`. Overlay JSON recorded,
  unlicensed (venue-effect datum).
- **W2_DRAFT_BLOCKS** ratified + applied by mac-local (0a73061ef).
- Cards carry AMENDMENTS A1/A2 (identity tol 1e-6→2e-3,
  cross-process GPU drift, conditioning-quantified) — committed
  before shuffled columns were read.

## OPEN: λ̂ overlay (the last meeting-fig deliverable)

- Grid `run_shuffle_overlay_retrain` on GPU 1: **11/18 done, 3
  workers alive (~163% CPU) in the tsae tail** (CPU-bound pair-loop,
  62–77 min class; started ~16:50). Remaining: tsae×3 + post
  T{2,4,8,16}/s42. Log `/workspace/lam_shuf_retrain.log`; monitor
  task armed (fires on "DONE 18/18" or process exit).
- **At drain:** (1) `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m
  experiments.explorations.task_hunt.lambda_intensity.shuffle_overlay`
  → identity receipts (tol 2e-3) → mechanical gate table (6 cells,
  1σ_quoted: post T2 .1296±.0171 / T4 .1607±.0160 / T8 .1848±.0244 /
  T16 .2548±.0473; sae .1130±.0218; tsae .1541±.0367). λ̂ σs are
  3–12× wider than tt's — tt-magnitude drift PASSES here; tsae is
  the risk cell (tt's tsae failed +.010; λ̂ tsae tol is .0367 —
  should hold).
  (2) ALL PASS ⇒ `render_overlay_figs --task lambda` →
  `fig_lambda_shuffle_tsweep.*`; commit overlay JSON + fig + LOG
  verdict entry (gate table verbatim, gaps story, ledger actuals
  ~$5-6 for 18 cells ≈ 105 min × 3 workers GPU 1). FAIL ⇒ STOP +
  report + λ̂ two-instrument fallback (screen = lambda_screen.json
  win/win_shuf; pattern = render_tt_fallback).
- Rows checkpoint each push (leaderboard/manifest/retrain JSON are
  live-append; commit them with -A, union-merge handles conflicts).

## House mechanics (fresh-window notes)

- Push contention is heavy: on pull-rebase conflict → keep BOTH LOG
  blocks (python re.sub pattern in history), `grep -c '<<<<<<<'
  LOG.md` must read 1 (line 9989 quotes the grep). If rebase sticks
  ("mark them as resolved" with clean ls-files -u): git commit
  --no-edit, `rm -rf .git/rebase-merge`, `git checkout -B arxiv
  HEAD`, push (used 3× today, content verified each time).
- Origin listener pattern: background fetch loop on
  `experiments/explorations/task_hunt briefings agents/runpod-a`,
  150 s; re-arm after every wake.
- Stamp from `date` (TZ=Europe/London); PTR everything; mac-local
  ratifies on push.

*Rewrite before any compact. — runpod-b*
