---
author: Han
date: 2026-04-29
tags:
  - design
  - in-progress
---

## Y Phase 2 — follow-on plan after Step 2

> Decision tree for what to do AFTER Step 2 (T=5, k_pos=20, random-init)
> lands. Three outcome branches per the brief's pre-registered rule, each
> at two metric anchors (1.80 unconstrained, 1.10 at coh ≥ 1.5). Updated
> with creative-axis options after reading W's plan + canonical_archs.

### Outcome branch A — **Step 2 wins under either metric** (≥+0.27)

Per pre-registered rule:
1. **Multi-seed verify** — train Step 2 at seed=1, regrade. ~30 min wall.
2. If seed=1 holds the win → headline result candidate.
3. **Step 3 — per-position decoder write-back** (FREE: no new training,
   just intervene with `intervene_paper_clamp_window_perposition.py`).
   Tests whether Q2.C-style protocol grows the win.
4. **Step 4 — MatryoshkaTXCDR @ k_pos=20** (~30 min train + grade).
   Tests whether the contrastive matryoshka head pays off at sparse k_pos.
5. **Step 5 — H8 multidist @ k_pos=20** (~30 min train + grade).
   Brief's "full TXCDR" cell.

### Outcome branch B — **Step 2 ties under either metric** (within ±0.27)

Per pre-registered rule:
1. **Seed=1 disambiguate** — train Step 2 at seed=1, regrade.
2. If seed=1 also ties → "sparsity is sole lever" narrative; stop ladder.
3. If seed=1 swings → run Step 3 (per-position write-back; FREE) and
   Step 4 (matryoshka @ k_pos=20) to see if richer architectures tip it.

**Creative axes IF seed=1 also ties** (don't blindly stop the ladder —
Han wants creative hill-climb):
- *k_pos=10* (sparser per-position, k_win=50) — tests whether the
  sparsity sweet-spot lies *below* k_pos=20.
- *k_pos=40* (denser, k_win=200) — tests whether k_pos=20 is too sparse.
- *k_win=20 truly-matched* (T=5, k_pos=4 avg) — TRUE per-window
  sparsity match to T-SAE k=20 (rather than per-token match).

### Outcome branch C — **Step 2 loses under either metric** (≤−0.27)

Per pre-registered rule:
1. **Step 1 (T=2 attribution)** — diagnostic. Was T=5 the issue, or
   does even minimum-deviation (T=2) lose? ~15-20 min train + grade.
2. **Step 3 (per-position write-back)** — FREE. Sometimes the per-position
   protocol recovers what right-edge loses (Q2.C added +0.30 absolute).
3. **Knowledge-class specific look** — the previous Y showed TXC family
   wins on knowledge-domain concepts even when overall-peak is below
   T-SAE k=20. Step 2's per-class table might still show a knowledge
   win even if overall is a loss. **Report as a partial-win finding.**
4. **Failure-mode investigation** — train Steps 4-5 (matryoshka, multidist)
   anyway to identify which axis hurts most. Per-axis attribution table
   is a publishable converging null with W.

### Cells I should NOT train (hand to W or skip)

- **Cell C** (T=3, k_pos=20) — W's sweep covers this.
- **Cell E** (matryoshka T=5, k_pos=20) — W's sweep covers this. Same
  arch as my Step 4. Coordinate via `[meeting cell]` / arch_id distinct
  if both train.
- **Cell F** (T=10, k_pos=20) — W's sweep covers this.

### What W's expanded creative axes (5-8 in their plan.md) imply for Y

W flagged 4 axes the Y brief didn't enumerate:
- Subseq sampling at k_pos=20
- k_win > T·k_pos (anchor regime; e.g. T=5, k_pos=20, k_win=200)
- Matryoshka H fraction
- Multi-distance shift schedule

Of these, **the anchor regime cell** (k_win>T·k_pos) is the most
interesting for Y because it directly tests "decoupling per-position
sparsity from window total". It's an entirely new dimension that
neither Y's atomic ladder nor W's k_pos sweep covered. **If outcome
B (tie) and seed=1 confirms tie, Y considers this cell as a creative
extension** to the brief's ladder.

### Compute budget left (estimated)

A40 wall budget per training: 15–40 min depending on T, k_pos, plateau.
Grading wall: 30–45 min (Anthropic 50 req/min, shared with W).

If Y is the only one grading: ~75 min per cell.
If Y + W grade in parallel: ~120 min per cell.

In a 6-hour autonomous shift after Step 2:
- Outcome A: 1 multi-seed verify + 1 free Step 3 + 1 fresh Step 4 ≈ 4 hours.
- Outcome B: 1 multi-seed verify + 1 creative cell (k_pos=10 or k_win=20) ≈ 3 hours.
- Outcome C: 1 fresh Step 1 + 1 free Step 3 + 1 fresh Step 4 ≈ 4 hours.

All branches leave 1-2 hours for writeup + plot polishing + commit cycle.
