---
status: active
created: 2026-07-23
for: runpod-b
venue: runpod
---

# Synth story pack — the head-to-head evidence assembly

**You are `runpod-b`** (32C). Parallel agents: `runpod` (loss
dissection), `runpod-c` (EM redo). Shared-branch + commit-citation rules
apply. This is an ASSEMBLY session, not a science session: no new cards,
no new benches, no rule edits. Everything comes from the existing
leaderboard + records. Results wanted within ~24 h (it is mostly
rendering). Do NOT quote or cite reviewer text in any tracked file —
this doc is a program deliverable framed on its own terms.

**Deliverable:** `experiments/explorations/synthetic/STORY.md` + figures
— the distilled TXC-vs-T-SAE-vs-per-token story a reader outside the
program can follow, with every number traceable to the canonical
leaderboard. Sections:

1. **The regime table, with receipts.** The 4-regime × winner table
   (README coordinates), each cell citing its bench + canonical numbers.
   Make the ambience point explicit: most measured real-language
   properties landed regime 1 (assumption NEGATIVE, hedging SPLIT) —
   per-token SAEs suffice there and the suite says so honestly.
2. **The isolation figure (the centerpiece — new figure).** One panel
   per regime exemplar (backtracking, frequency, phasepair,
   recipe-residual), bars per arch (per-token SAE, T-SAE, Stacked,
   TXC-pre, TXC-post, Spectral) at the canonical matched slice, 3-seed
   mean ± spread. The reading it must make unmissable:
   - **regime 2: Stacked ≈ TXC ≈ any window arch** — temporal
     aggregation suffices; cross-position weight sharing is NOT
     load-bearing there;
   - **regime 3: Stacked ≈ per-token ≈ 0 while the mixing codes win** —
     cross-position structure IS load-bearing, and WHICH mixing code
     wins follows the subtype rule.
   Script committed; renders from `results/leaderboard.jsonl` only.
3. **The subtype rule + its blind-prediction record**: the
   phase/power-equality/covariance split with the two qualifiers
   (T-conditional band multiplicity; DCT-alignment), and the scorecard
   of frozen-prediction outcomes (T=16 addendum 3/3; stage-6 #3b;
   FB-C1 verdicts incl. honest misses; FB-4 refuted; FB-5 fork). A
   small table: prediction → held/missed → where recorded.
4. **T-SAE positioning**: T-SAE's panel profile across the suite
   (regime 1 competent, regime 3 ≈ per-token) — one paragraph + its
   column in the figure; connect to where T-SAE wins in the paper's
   real-world tasks (ambient-shaped labels), WITHOUT reviewer framing.
   Include the sparse-probing corollary: probing concepts are
   ambient-shaped (regime-1-adjacent), so ALL architectures cluster
   within ~0.001 AUC there BY CONSTRUCTION — the regime map explains
   why probing cannot adjudicate temporal architectures, and why
   architecture conclusions should be drawn where the suite
   discriminates.
5. **Robustness + budget-parity notes**: the new suite is 3-seed + 
   untrained-control throughout; per-token-matched realized L0; capacity
   sweeps; capability companions (winner must also reconstruct). One
   paragraph each, citing conventions.
6. **Parameter/inference-cost table** for the 6-arch panel at the
   canonical operating point (from configs/arch code — count encoder/
   decoder params as a function of T, d_in, d_sae; note TXC inference
   cost scaling).

Spot-check every number against the leaderboard before writing it (no
hand-typed numbers — extend `render_report`-style extraction where
needed, scripts committed).

## Acceptance gate — stop for review

STORY.md + isolation figure + scripts pushed; STATUS rewritten.
Briefing stays until mac-local review.
