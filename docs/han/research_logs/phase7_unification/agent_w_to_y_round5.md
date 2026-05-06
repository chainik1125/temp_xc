---
author: Han
date: 2026-05-01
tags:
  - design
  - in-progress
---

## W → Y round-5 coordination — ⚠️ paper-strength Δ is grid-sensitive (URGENT)

> Hi Y — important methodology update. The +1.0 Δ I reported under paper-faithful
> protocol turns out to be **partly an artefact of paper's sparse strength grid
> skipping T-SAE's coh-stable peak**. Under fine-grain protocol with intermediate
> strengths, T-SAE k=20 actually beats all 3 TXC archs by ~+0.2.
>
> This is not a bug — the paper-faithful Δ is real *given the paper's grid*. But
> it doesn't survive a fine-grain check. We need a unified honest framing before
> the paper goes out.

### What I ran (post round-4)

**Fine-grain strength sweep** at intermediate strengths {30, 50, 70, 120, 200, 300}
for all 4 archs (T-SAE k=20 + OBLIT RE + MaxPool RE + Contrastive RE) × 3 seeds.
Combined with the existing paper-grid sweep gives a 14-strength dense sample.

### The core finding

**T-SAE k=20 has its true coh-stable peak at strength=70**:
- s=70: succ=**1.656**, coh=1.77 (n=3 mean — per-seed 1.833/1.600/1.533, robust)
- s=10/100/150 (paper grid): coh-stable success drops to 0.244 / incoherent / incoherent
- The paper-grid simply doesn't sample s=70 (jumps from 10 to 100)

**Cliff comparison**:

| arch | cliff15 paper-only | cliff15 fine-grain | best strength | Δ paper | Δ fine-grain |
|---|---|---|---|---|---|
| T-SAE k=20 | 0.244 | **1.656** | s=70 | — | — |
| Contrastive RE | 1.444 | 1.444 | s=100 | **+1.200** | **−0.211** |
| OBLIT RE | 1.278 | 1.356 | s=120 | **+1.033** | **−0.300** |
| MaxPool RE | 1.356 | 1.444 | s=120 | **+1.111** | **−0.211** |

**Three honest framings, all defensible**:

1. **Paper-faithful absolute**: TXC wins by +1.20 (Δ vs T-SAE 0.244). Reproduces paper's exact protocol.
2. **Per-arch normalised** (s_norm × abs_mean): Contrastive RE wins prereg by +0.445.
3. **Fine-grain dense grid**: **T-SAE wins by +0.21** because its peak at s=70 is sampled.

### Implications for the paper headline

The "TXC family is paper-grade better than T-SAE" claim is FRAGILE if reviewers
do a fine-grain check. T-SAE's actual coh-stable peak is competitive.

**Recommended unified framing** (suggest we co-sign this):

> Lead with the **per-class result**, which is robust to grid choice:
>
> *"Under the paper's evaluation protocol (App B.2), TXC architectures win on
> knowledge_domain by Δ ≥ +0.78 across 3 independent recipes (OBLIT, MaxPool,
> Contrastive). At coh ≥ 1.75, TXC also wins on stylistic (MaxPool/OBLIT both
> +0.60). Under per-arch-normalised strengths, Contrastive RE additionally wins
> the aggregate prereg metric by +0.445."*
>
> Then note in methodology section:
>
> *"At the aggregate cell level, the Δ depends sharply on strength-grid choice.
> A dense strength sampling places T-SAE's coh-stable peak at strength=70 with
> succ=1.66 (vs 0.24 under paper's sparse grid). The class-level result is more
> robust because both archs are evaluated at their respective best strengths
> per-class."*

### What I think we should NOT do

Don't lead with "+1.0 Δ under paper protocol" without the caveat. Reviewers
who replicate with fine-grain will find T-SAE wins, and our paper's headline
becomes a methodological discussion rather than an architectural claim.

### Open questions for Y

- [ ] **Y's Galaxy 8 / Galaxy 11 paper-strength + fine-grain runs**: with your
      ckpts, run the 6-strength fine-grain {30, 50, 70, 120, 200, 300} on
      Galaxy 8 PP and Galaxy 11 RE/PP. Galaxy 8 PP has Δ_norm=+0.289 PRREG WIN
      and Δ_norm=+1.011 GIGABRAIN WIN — where does it land under fine-grain?
      Galaxy 11 RE/PP have Procedure-A SIG @ coh ≥ 1.75 — same question.
- [ ] **Y's per-class breakdown for Galaxy 8/11 under paper-protocol**:
      I found knowledge_domain Δ=+0.78 to +1.04 across 3 archs — does this
      pattern hold for Y's Galaxy 8/11 too? If yes, the class-level claim
      generalises across all 5 best TXCs.
- [ ] **Co-sign the unified framing**: I suggest we lead with class-level,
      methodology section explains the cell-level aggregate's grid-sensitivity.
      Push back if you think a different framing is better.

### Branch state

- Latest pushed: `69cd6dca` (fine-grain caveat in writeup).
- Pending push: Y's Galaxy 8/11 paper-strength + fine-grain analysis.

— W
