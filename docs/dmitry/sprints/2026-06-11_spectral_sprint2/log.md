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
