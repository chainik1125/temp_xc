---
author: Agent Y (Aniket pod)
date: 2026-04-29
tags:
  - results
  - complete
---

## Y Q2.C — per-position window clamp does not rescue paper-protocol

> Tests Q2 candidate (C) from `agent_y_brief.md`: "decompose the
> window into per-position contributions". The brief asked for at
> least the per-position variant to be tried before discarding the
> protocol. Outcome: rejected — same negative direction as Dmitry's
> T=20 result, generalised to T=5 and T=10.

### Method

Cherry-picked `intervene_paper_clamp_window_full.py` from
`origin/dmitry-rlhf` (originally written for his T=20 investigation),
parameterised with `--strengths` and `--out-subdir` to match the
right-edge variant's interface, and ran on the two window archs in
the shortlist at the Q1.3 finer grid:

- `agentic_txc_02` (TXC matryoshka, T=5)
- `phase5b_subseq_h8` (SubseqH8, T_max=10)
- 30 concepts × 9 strengths {50, 100, 150, 200, 300, 400, 500, 700, 1000}
- Same grader (Sonnet 4.6 + cached system rubric, ThreadPool=8)

Difference from right-edge variant: instead of injecting the steered
reconstruction at the right-edge token only (one position per
generation step), the full (T, d_in) steered window is injected at
all T positions of the rightmost window. Higher-layer attention sees
"concept-flavored" residuals at multiple key positions, not just one.

### Code

[`intervene_paper_clamp_window_full.py`](../../../experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_window_full.py)
— sourced from Dmitry's branch, attribution noted in the file header.

[`q2c_compare_window_variants.py`](../../../experiments/phase7_unification/case_studies/steering/q2c_compare_window_variants.py)
— per-strength comparison vs right-edge.

### Outputs

- [`q2c_window_variant_comparison.json`](../../../experiments/phase7_unification/results/case_studies/steering_magnitude/q2c_window_variant_comparison.json)
- [`q2c_window_variant_comparison.png`](../../../experiments/phase7_unification/results/case_studies/steering_magnitude/q2c_window_variant_comparison.png)
- Raw: `results/case_studies/steering_paper_pos_full/<arch>/{generations,grades}.jsonl`

### Results

![Q2.C right-edge vs full-window](../../../experiments/phase7_unification/results/case_studies/steering_magnitude/q2c_window_variant_comparison.thumb.png)

| arch | T | variant | log-fit peak s | peak suc |
|---|---|---|---|---|
| `agentic_txc_02` | 5 | right-edge (Q1.3) | 447 | **0.83** |
| `agentic_txc_02` | 5 | full-window (this) | 541 | 0.77 |
| `phase5b_subseq_h8` | 10 | right-edge (Q1.3) | 498 | **0.97** |
| `phase5b_subseq_h8` | 10 | full-window (this) | 405 | 0.90 |

### Read

- Right-edge wins both archs by 0.06-0.07 peak success — within
  concept-variance noise (n=30 single seed) but **never the other
  way**. Full-window does NOT improve, in either arch.
- This generalises Dmitry's T=20 finding (in his
  `t20_steering_investigation.md`: full-window worse than
  right-edge for three different T=20 ckpts) down to T=5 and T=10.
- The "concept signal is spread across the window so injecting at
  multiple positions should help" intuition is wrong. If anything,
  injecting at multiple positions over-saturates the model's
  attention with concept-flavored signal and degrades coherence
  faster (the full-window peak shifts toward higher s, suggesting
  the same mechanism but stronger total-injection).
- Speculation: right-edge attribution preserves the per-position
  decoder atom's "this token's contribution to the concept" and
  the model's autoregressive attention compounds it via natural
  attention dynamics. Full-window pre-bakes T identical concept
  signals into the keys, which the model treats as redundant
  context rather than amplification.

### Verdict

**Candidate (C) rejected.** Combined with Q1.3's rejection of
candidate (B) and Dmitry's `t20_steering_investigation.md`
disconfirmation of (D), only candidate (A) — AxBench-additive as
canonical — survives Y's investigation.

### Other window-clamp variants worth exploring (not pursued here)

- **Per-position decoder norms unit-normalised independently**.
  Suggested in Dmitry's t20 note as a training-time fix; this is
  Z's territory.
- **Weighted clamp by attention-to-right-edge**. More principled
  than uniform full-window but adds an attention-pattern
  computation step and the right-edge vs uniform comparison
  already shows that "more positions" doesn't help.
- **Centre-position clamp only**. Unclear theoretical motivation
  vs right-edge; right-edge is already the natural attribution
  for autoregressive generation at the current token.

None of these are likely to overturn the negative result given
how consistent right-edge >= full-window has been across T=5,
T=10, T=20.

### Cost

| step | grader calls | wall time | API spend |
|---|---|---|---|
| Generation | 0 | ~6 min (T=5 + T=10 with use_cache=False) | $0 |
| Grading | 1080 | ~7 min (ThreadPool=8 + cache) | ~$1 |
| **Q2.C total** | **1080** | **~13 min** | **~$1** |
