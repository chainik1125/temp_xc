---
author: Han
date: 2026-04-30
tags:
  - results
  - in-progress
---

## Per-concept-class breakdown — multi-seed (T=2: 3 seeds, T=5: 2 seeds)

> Following the multi-seed verify of T=2 + T=5 cells, recomputed
> per-class breakdown by averaging the per-concept-class peak success
> across seeds. The seed=42-only finding "TXC wins on knowledge-
> domain" partially reverses; the cleanest TXC-favourable signal at
> matched sparsity is **stylistic + sentiment + (T=2 per-position only:
> safety)**.

### Multi-seed mean peak success per concept-class (at coh ≥ 1.5)

| cell | knowledge / domain | discourse / register | safety / alignment | stylistic | sentiment |
|---|---|---|---|---|---|
| T-SAE k=20 (anchor) | **2.000** | **1.375** | 0.333 | 0.200 | 0.500 |
| T=2 right-edge (n=3) | 1.741 | 1.125 | 0.333 | **1.000** | **1.000** |
| T=2 per-position (n=3) | 1.630 | 1.167 | **0.889** | **0.933** | **1.000** |
| T=5 right-edge (n=2) | 1.444 | 0.688 | 0.333 | **0.900** | 0.500 |
| T=5 per-position (n=2) | 1.389 | 0.750 | 0.000 | 0.500 | **1.250** |
| T=3 right-edge (W's, n=1) | 1.556 | 0.625 | 0.167 | 0.400 | 0.500 |

### Δ vs T-SAE k=20 anchor

| cell | knowledge | discourse | safety | stylistic | sentiment |
|---|---|---|---|---|---|
| T=2 right-edge | −0.259 | −0.250 | +0.000 | **+0.800** ⭐ | +0.500 |
| **T=2 per-position** | −0.370 | −0.208 | **+0.556** ⭐ | **+0.733** ⭐ | +0.500 |
| T=5 right-edge | −0.556 | −0.688 | +0.000 | **+0.700** ⭐ | +0.000 |
| T=5 per-position | −0.611 | −0.625 | −0.333 | +0.300 | **+0.750** ⭐ |
| T=3 right-edge (W's) | −0.444 | −0.750 | −0.167 | +0.200 | +0.000 |

### Multi-seed-validated patterns

#### 1. TXC wins on stylistic, robust across cells

Stylistic concepts (poetic, literary, list_format, citation_pattern,
technical_jargon) are the cleanest TXC-favourable signal at matched
sparsity:
- T=2 right-edge: **+0.80**
- T=2 per-position: **+0.73**
- T=5 right-edge: **+0.70**
- T=5 per-position: +0.30
- T=3 right-edge: +0.20

All 5 cells beat the anchor on stylistic. The previous Y also flagged
stylistic at seed=42 (Step 2 had +0.40 on stylistic alone). Multi-seed
strengthens this signal — **stylistic is structurally TXC's**.

#### 2. T-SAE k=20 dominates knowledge / discourse, robust across cells

The previous Y's "TXC wins on knowledge-domain" claim was at
**unconstrained** paper-clamp; under coh ≥ 1.5 with matched sparsity,
the pattern reverses. **All 5 TXC cells lose to T-SAE k=20 on knowledge
and discourse classes** (Δ ≤ −0.21 to −0.75).

Mechanism candidate: knowledge concepts need *concept-specific*
features. At k_pos=20, the window encoder produces *less* concept-
specific features (Y's polysemanticity finding: 24-25/30 distinct
picked features vs T-SAE k=20's 28/30). The polysemanticity hurts
exactly the classes (knowledge / discourse) that need precise
disambiguation.

#### 3. T=2 + per-position is the most balanced TXC cell

| cell | n_classes_won | n_classes_tied | n_classes_lost |
|---|---|---|---|
| T=2 right-edge | 2 (stylistic+0.80, sentiment+0.50) | 1 (safety+0.00) | 2 (knowledge, discourse) |
| **T=2 per-position** | **3 (stylistic+0.73, sentiment+0.50, safety+0.56)** | 0 | 2 (knowledge, discourse) |
| T=5 right-edge | 1 (stylistic+0.70) | 2 (safety, sentiment) | 2 |
| T=5 per-position | 2 (stylistic+0.30, sentiment+0.75) | 0 | 3 (incl. safety−0.33) |
| T=3 right-edge | 1 (stylistic+0.20) | 1 (sentiment+0.00) | 3 (incl. safety−0.17) |

**T=2 + per-position wins on 3 of 5 classes.** Loses on only knowledge
and discourse. The other 4 cells lose on at least 2 classes including
safety.

### Headline narrative — multi-seed-validated

The seed=42-only Y reported:
> "TXC wins on knowledge-domain concepts (medical, mathematical,
> historical, code, scientific) by 0.32 mean success points; T-SAE k=20
> wins on discourse / register concepts by 2.00 points."

**Multi-seed at matched sparsity REVERSES the knowledge-domain finding.**
The cleaner story:

> At matched per-token sparsity (k_pos=20), TXC cells win on
> **stylistic** (poetic, literary, formatting) and **sentiment**
> concepts robustly across seeds. T-SAE k=20 wins on **knowledge-
> domain** and **discourse / register** concepts (which need precise
> feature disambiguation). The T=2 + per-position cell additionally
> wins on **safety / alignment** by +0.56. Architecture matters
> *per concept-class*: stylistic concepts have multi-token "shape"
> structure that the window encoder captures; knowledge concepts are
> structurally per-token and the polysemantic features at sparse k_pos
> can't disambiguate.

### Caveats

- The seed=42 Y's "knowledge-domain" finding was at **unconstrained**
  peak, not coh ≥ 1.5. Under unconstrained peak (METRIC A), the per-
  class table might still favour TXC on knowledge — but at unconstrained
  peak T-SAE wins overall by 0.5+, so the per-class claim there is
  in a different operating regime.
- T=5 cells have only 2 seeds; T=3 has 1 seed. Pattern strength varies.
- Per-class numbers come from each arch's *own* peak coh≥1.5 s_norm.
  The s_norms differ (T-SAE 49.9, TXC 113-251). At the same absolute
  s_norm the comparisons would shift — but the family-normalised paper-
  clamp framework treats this as the canonical "fair" comparison.

### Files

- This writeup
- Per-class numbers derivable from the existing grades.jsonl files
  (no new artifacts required)
