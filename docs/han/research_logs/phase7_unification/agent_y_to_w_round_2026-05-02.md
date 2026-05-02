---
author: Han
date: 2026-05-02
tags:
  - design
  - in-progress
---

## Y → W coordination 2026-05-02 — V7 protocol + unified-pareto moved

> Hi W — three things to flag:
>
> 1. **V7 tiled-broadcast steering protocol is the new top cell at PRREG.**
> 2. **`unified-pareto.md` is now at the top-level `phase7_unification/` dir**
>    (no longer in `agent_y_phase2/`) — it's joint Y+W canonical now.
> 3. **`plot_unified_pareto.py` extended with V7 entries + 3-panel layout
>    (RE / PP / V7).** Will auto-pick up your Mystery archs once V7 recovery
>    lands (in flight).

### The V7 finding (committed in `b42f9770`)

Han asked 2026-05-01: *"Have we tried the obvious thing — non-overlapping
TXC windows + same steering vector per window?"* Answer: no, hadn't tried.
V7 implements exactly that:
- Tile prefix into stride-T non-overlapping blocks
- Per block: encode → clamp picked feature → decode → AVERAGE per-position
  delta to single (d_in,) vector → broadcast to all T positions in block

Source: `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_window_tiled_broadcast.py`

#### NEW PRREG TOP CELL: Galaxy 11 (SoftMaxPool+H8) V7

**Galaxy 11 V7 3sd: peak15 = 1.689 — Δ=+0.556 vs same-pod n=3 anchor 1.133.**

Beats your previous Contrastive-merge RE PRREG winner (+0.445) by +0.111.
ALSO essentially TIES T-SAE on the unconstrained peak (1.689 vs 1.678 →
gap only −0.011). The TXC family is now within 1 succ-grade-step of
T-SAE's last stronghold.

#### V7 protocol-vs-arch matrix (3-seed Δ at coh ≥ 1.5 = peak15 − 1.133)

V7 WINS for half the archs tested (5 of 9 confirmed; 4 in flight):

| arch | V1 RE peak15 | V2 PP peak15 | V7 peak15 | best |
|---|---:|---:|---:|---|
| **Galaxy 11 SoftMaxPool+H8** | 1.467 | 0.989 | **1.689** | ★ V7 (NEW) |
| Galaxy 18 SoftMaxPool T=3 | 1.178 | 1.356 | **1.444** | ★ V7 |
| Galaxy 8 SoftMaxPool T=2 | 1.133 | 1.422 | 1.333 | PP |
| Galaxy 6 max-pool | 1.322 | 1.433 | **1.311** | PP (V7 close) |
| T-SAE WS | 1.133 | 1.222 | **1.333** | ★ V7 |
| Galaxy 23 SoftMaxPool T=5 | 0.956 | – | **1.089** | ★ V7 |
| T=2 bare-antidead | 0.956 | 0.978 | 0.978 | PP/V7 tie |
| Galaxy 20 LSE pool | 1.222 | 1.300 | 1.344 | V7 (close) |
| **T=2 H8 shifts=(2,)** | **1.236** | 1.400 | 0.311 | RE (V7 = LOSS) |

V7 LOSES T=2 H8 because the H8 contrastive trains end-position-discriminative
features. V7's averaging-over-T dilutes the end-heavy decoder direction.

#### Mechanism (attention-mixing)

At the next layer, attention computes Σ_s α_{t,s} W_V δ_s. If δ is uniform
within a window, then Σ_s α_{within} · δ = δ since attention weights sum
to 1 — V7 is **attention-invariant within window**. V2 PP writes T different
deltas → attention scrambles them at higher T (this is exactly the Galaxy 18
T=3 PP regression: +1.011 → +0.233 going T=2 → T=3 because attention can't
unmix 3 different deltas).

See `agent_y_phase2/2026-05-01-y-steering-protocol-space.md` for the full
protocol-space inventory and analysis.

### Where things are now

- **Doc**: `docs/han/research_logs/phase7_unification/unified-pareto.md` (moved)
- **Plot**: `experiments/phase7_unification/results/case_studies/plots/unified_pareto_full.{png,thumb.png}` (now 3 panels)
- **Script**: `case_studies/steering/plot_unified_pareto.py` (V7 entries added; auto-picks up new V7 data)

### Asks of W

- **Pull `b42f9770`** to get the V7 protocol script + updated unified-pareto.
- **Run V7 on your remaining mystery archs** if you want fresh data
  (in-flight recovery from Y handles MaxPoolMergeH8, Contrastive-merge,
  T=3 grown, Galaxy 4 — should land within ~30 min).
- **Confirm whether you agree V7 should be reported as the headline protocol
  for archs where it wins** (vs reporting per-architecture best-protocol).
  Y's preference: report per-arch best-protocol; the protocol-by-T recipe
  (T=2 → V2 PP, T≥3 → V7 V7) is itself a paper finding.
- **Update your Phase 4 writeup** if you want to reference Galaxy 11 V7
  as the new PRREG top cell.

— Y, 2026-05-02 14:00 UTC
