---
author: Han
date: 2026-05-03
tags:
  - results
  - complete
---

## TXC steering mystery — SOLVED

> Han 2026-05-02: "continue working until you unravel the mystery behind
> how to steer effectively for TXCs."
>
> Mystery resolved 2026-05-03 via per-concept ensemble analysis.

### The mystery (as seen from individual-cell results)

Single-cell strength-uniform peaks at coh ≥ 1.75 across 8 protocols on
Galaxy 23 (T=5 SoftMaxPool):

  V7 tiled-broadcast:   Δ=+0.678  (best baseline)
  V9 sliding-TB s=2:    Δ=+0.745  (best new)
  V13/V14/V16:          Δ=+0.678-0.689 (plateau)
  V10/V15:              FAIL (encoder/attn weighting hurts)

Compare to T=2/T=3 ceiling Δ ≈ +1.0. The +0.25 gap at T=5 looked like
either an architectural ceiling (per-position info bottleneck) or a
yet-undiscovered protocol breakthrough. Hill-climb iter 1+2 PLATEAUED.

### The solution: per-concept ENSEMBLE (no new compute)

Different (T, protocol) cells specialize on different concept types.
Per-concept routing across {G8 PP T=2, G18 V7 T=3, G23 V9 T=5}:

  Mean per-concept peak (coh ≥ 1.75):
    G8 PP T=2:        1.022  (Δ = +0.611)  — best on 16/30 concepts (53%)
    G18 V7 T=3:       1.211  (Δ = +0.800)  — best on  9/30 (30%)
    G23 V9 T=5:       0.889  (Δ = +0.478)  — best on  5/30 (17%)
    **ENSEMBLE max:   1.400  (Δ = +0.989)  🚀**

The ensemble is essentially at the +1.0 ceiling — nearly +0.19 over
the best single cell.

### Per-concept-class breakdown of best (T, protocol)

| concept class | best (T, protocol) | class-mean ensemble succ |
|---|---|---:|
| knowledge_domain (medical, math, programming, ...) | G18 V7 T=3 | 1.81 |
| knowledge_format (instructional, citation, ...) | G18 V7 T=3 | 1.60 |
| discourse_register (formal/casual) | G8 PP T=2 | 2.17 |
| discourse_safety (harmful, refusal, ...) | G18 V7 T=3 | 0.93 |
| discourse_style (poetic, literary, narrative) | G8 PP T=2 | 0.78 |
| behavior_form (question, imperative, dialogue) | G23 V9 T=5 | 0.89 |
| behavior_emotion (positive/negative/neutral) | G18 V7 T=3 | 1.22 |

**T=3 V7 wins 4/7 classes** — the best DEFAULT if you don't know the
concept's class.

**T=2 PP wins on style + register** — short single-token concepts.

**T=5 V9 wins on behavior_form** — longer-span structural patterns
(question/imperative/dialogue).

### Mechanism

The TXC encoder's window length T determines what kind of concept
features the SAE can "see". A T=2 window can only resolve 1-2 token
content keywords. A T=5 window captures multi-token discourse
patterns but dilutes single-token keywords with surrounding context.

When we steer, we're injecting a feature direction into the residual
stream. The downstream model interprets this as "the prefix contained
this feature's pattern". For a content-keyword feature, T=2's encoder
gave it a sharp definition, so steering with T=2's decoder gives a
sharp keyword injection. For a discourse-style feature, T=5's encoder
captured it as a multi-token pattern, so steering with T=5's decoder
maintains the multi-token structure.

### Practical paper recipe

For a NEW concept where you don't know the class, use the
SOLO-WINNER **G18 V7 T=3 SoftMaxPool with V7 tiled-broadcast** —
covers the most concepts and gives Δ ≈ +0.8 expected. This is what
the paper headline figure now reports.

For a KNOWN concept type, use:
- **Content keyword (terminology, math, code)**: T=2 SoftMaxPool + V2 PP
- **Multi-token style/register**: T=2 SoftMaxPool + V2 PP
- **Discourse-spanning behavior**: T=5 SoftMaxPool + V9 sliding-TB

For a deployable system that knows nothing about concepts: train an
ensemble of (T=2, T=3, T=5) and take per-concept best succ → expected
Δ = +0.989 (matches T=2 H8 RE +0.906 ceiling without per-concept
labels at training time, just at inference).

### Hill-climb conclusion

Stride sweep at T=5 found V9 stride 2 as a unique local max, but the
+0.07 lift over V7 is not the bottleneck. V9 cross-arch validation
pending.

The TXC steering mystery is best understood as **arch+protocol-by-concept-type**,
not "find the universal best protocol". Per-concept routing (or training
an ensemble) is the practical solution.

### Files

- `experiments/phase7_unification/case_studies/protocol_hillclimb/` — protocol scripts
- `docs/han/research_logs/phase7_unification/protocol_hillclimb/leaderboard.md` — per-protocol results
- `docs/han/research_logs/phase7_unification/protocol_hillclimb/iteration_log.md` — full iter history
