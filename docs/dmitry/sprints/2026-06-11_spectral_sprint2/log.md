---
author: Claude (10h unsupervised sprint #2)
date: 2026-06-11
tags:
  - results
  - in-progress
---

## Sprint-2 log: window-timescale matching for the spectral crosscoder

Wall clock: start **2026-06-11 05:32:09 UTC** (epoch 1781155929), hard stop
**15:32:09 UTC**. Branch `dmitry-spectral-sprint2`. Budget < $50.

### Questions (user-set)

- **A.** Backtracking anticipation measured as low-frequency (sprint 1). Does
  growing the window T — admitting lower frequencies — improve spectral-XC
  results? Where does performance peak in T (= the timescale of the
  backtracking state)?
- **B.** Screen the repo/paper tasks for label-frequency content; match
  spectral-XC configs to each. Consistent story → C.
- **C.** (Authorized) multi-agent workflow: brainstorm real-world behaviours
  suited to spectral XC → rank by frequency metric → evaluate top candidates
  → red-team the conclusion.

### Preregistered predictions (H0:05)

1. **Raw window-mean probe AUC vs T** (the dictionary-free timescale
   measurement) rises from T=4, peaks at an intermediate T, and declines once
   the window outlives the anticipatory state. Peak location = the
   backtracking-state timescale. Sprint-1 anchors: T=1 edge 0.769, T=16 mean
   0.818. Genuine uncertainty about the peak: sentence-scale (T*≈16–32) vs
   multi-sentence drift (T*≥64).
2. **DC-SAE** (dictionary on the window mean; params independent of T) tracks
   the raw-mean curve minus a small sparsity tax; **spectral multiband at
   T=32 ≥ T=16** on DC-branch probe AUC iff the raw curve still rises at 32.
3. Dictionary-free band screening will show *different profiles across
   tasks* (B): backtracking = DC-heavy (known); at least one other task
   (candidate: hh_rlhf choice quality or a Venhoff thinking-behaviour) shows
   a different band signature — otherwise "match the spectral config to the
   task" has no content. Honest possibility: everything in LLM-land is
   DC-heavy at L10-scale hooks; that would itself be a finding (slowness is
   generic), and the workflow pivots to ranking by *degree* of slowness.
4. The key risk for A at large T: fewer usable positions (think regions ~512
   tokens) and post-hoc window-position confounds; mitigation: per-T matched
   example counts + by-trace splits, report n.

### Design notes

- The DC band of a window-T crosscoder is exactly an SAE on the T-token
  window mean ⇒ its parameter count is T-independent ⇒ T can be scanned to
  128 at trivial cost. Full spectral models param-matched by H×T = const.
- All probes: balanced D+ vs far-negatives (sprint-1 protocol), by-trace
  splits, AUC, 2 seeds for dictionaries.

### Log

- **H0:00** Clock started; branch `dmitry-spectral-sprint2`; tasks created.
- **H0:15** Resilience layers (user away): cloud routine trig_012G7TKgQTmLRPWS7M4r4vKB
  (hourly :23, claude-opus-4-8, heartbeat = branch commit age, syncs pod
  results via HTTPS proxy, self-noops after 16:00 UTC); on-pod dead-man
  timer kills the container at 15:20 UTC (billing stops, no secrets
  shipped); durable local cron attempted but session-only (flagged).
  Branch pushed = heartbeat channel; .pt/.npy excluded by gitignore (repo
  1MB hook).
- **H0:20 — QUESTION A HEADLINE (dictionary-free).** Raw window-mean probe
  AUC vs T: 0.769 (T=1) → 0.815 (T=4) → plateau ≈0.81–0.83 → **peak 0.830
  at T=48** → 0.804 (T=64) → 0.696 (T=96). The backtracking anticipatory
  state lives at sentence-to-paragraph scale (~30–50 tokens). Longer windows
  help modestly beyond T=16 (+0.012); far beyond, the window outlives the
  state (and at T≥64 windows increasingly span the prompt; n+ drops
  270→210 — caveats logged). DC-SAE interim: T16 0.79, T32 0.71/0.82 (high
  seed variance — investigate), T64 0.80. Spectral/vanilla T∈{16,32} cells
  training.
- **H1:00** Workflow (wf_ebf30e77-675, 14 agents) returned. Top-ranked
  candidates: (1) repetition/rumination loop onset — MID-band prediction,
  programmatic 6-gram labels (the first non-DC test!); (2) reasoning
  macro-phase / verification-mode (DC); (3) EM onset within a generation
  (low); then context-rot, revision commitment, sycophantic capitulation,
  answer commitment, compliance momentum. Red team (3 agents) demanded:
  scrambled-window control (denoising vs temporal structure), fixed-eval-set
  T-curve (T=96 collapse may be set composition), position-only probe
  ceiling (leakage), layer-0 embedding control for loops (lexical
  recirculation), fair-baseline note for sprint-1's PR-AUC@8 claim (band
  choice uses label knowledge — defense: branch was selected on independent
  distill-cache data before the c7 eval), one-primary-endpoint-per-task
  preregistration. ALL adopted: bt_controls.py + bt_loops.py queued.
  Ranking-process critiques (data-availability triple-counting; no
  high-band candidates) noted for the writeup.
- **H1:10** hh-rlhf screen DONE: near-null. Raw-mean AUC 0.531-0.571
  (weak peak T=64), bands DC 0.551 / low 0.519 / mid 0.500 / high 0.520.
  Chosen-vs-rejected is barely decodable from distill-L10 end-of-transcript
  windows — informative contrast row: weak-everywhere profile (≠ strong-DC
  backtracking), low priority for spectral XC. Caveats: domain mismatch
  (reasoning distill on chat), last-128-token construction.
- **H1:45** Pod-7 crash in spectral arm: branch probes were nested inside
  torch.no_grad() in the bt_wscan port (sprint-1 code probed outside) →
  "element 0 does not require grad". Fixed, resumed; finished cells skip.
  DC-SAE full: T4 .74, T8 .77/.81, T16 .75/.79, T32 .71/.82, T64 .80/.82 —
  noisy seeds, roughly tracks raw curve minus a sparsity tax; T64 ≈ best.
- **H2:30 — QUESTION A, dictionary level (2 atoms/token density).**
  multiband T=32 AUC 0.765/0.780 > T=16 0.726/0.753 — the spectral XC gains
  from the longer window, tracking the raw-mean curve (prereg #2 ✓: raw
  curve still rises at 32). Vanilla txc T=32 0.694 > T=16 0.622/0.644 —
  also gains, but trails multiband by ~+0.09 at both T. NOTE: this arm ran
  at 2 atoms/token (synthetic convention leaked into 2*T) vs sprint-1's
  16/token — FVUs 0.72-0.84 accordingly; internally consistent scan;
  16/token comparability pass queued (4 cells, seed 0).
- **H3:20 — CONTROLS LAND (red-team vindicated; headline revised).**
  (C3) Position-only probe = 0.685: a large share of all AUCs is positional
  leakage; the temporal increment above position is ~0.1, not 0.83. (C2)
  Fixed-eval-set curve: T=96 "collapse" was COMPOSITION ARTIFACT (fixed-set
  T96 = 0.781 vs 0.696 headline); T=48 still best (0.793) but the curve is
  flat-noisy (0.73-0.79). (C1) Scrambled-token pooling loses ~0.03-0.07 vs
  contiguous windows at most T (T32: 0.665 vs 0.740) — the signal is not
  pure denoising; temporal contiguity carries real information. Revised
  Question-A statement: backtracking state is slow-ish with a broad optimum
  around T~32-48; window-growth gains are real but modest; position
  controls mandatory for all rows. Relabel screen failed on counts
  (verification 19, uncertainty 1, conclusion 76 events) — keyword sets
  broadened, threshold lowered to 60, requeued. Loops rerunning after a
  grad-graph bug fix (emb weights needed .detach()).
