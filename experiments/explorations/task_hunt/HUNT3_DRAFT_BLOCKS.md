# HUNT3 draft blocks — REBUTTAL_PACK / WRITEUP staging (mac-b, for mac-local ratification)

**Status: DRAFT — nothing applied.** mac-b, 2026-07-27 ~04:55 London, per the
02:20 no-idle allocation. Companion deliverable committed alongside:
**`hunt3/panel_evidence_line_cnov.json`** (generator
`hunt3/panel_evidence_line_cnov.py`, tt-convention verbatim) — the proposed
panel's S4 KILL-clause input, measured label-side before any cell.

## 0. The evidence-line result first — it informs the 17:00 pick

Pearson r between the screen's committed visible floor (first-in-WINDOW
kernel rate) and the cnov label, population = finite label, non-boundary,
pos ≥ T (n = 520,811 at every T):

| T | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| floor r | 0.048 | 0.136 | **0.269** | **0.402** | 0.632 |
| in-window kernel mass | — | 31.2% | 53.3% | 80.0% | 100% |

The floor tracks the kernel's in-window mass (same HL-16/support-64 kernel
as txcwin — the mass numbers are literally txcwin's). **Consequence for the
draft card's S4:** the bar at the proposed claiming Ts is r = 0.269 (T16)
and r = 0.402 (T32) — the T32 bar is **3.5× ttrend's** (0.114, which
TXC-post beat at 0.282). A latent-state claim at T32 requires recovery
> 0.402; on every panel precedent that is out of reach, and even the T16 bar
(0.269) sits at ttrend-post's T32 level. **Flag for freeze-review: the
structural guarantee is strongest exactly where the kernel mass is
out-of-window — if the team picks cnov, consider T ≤ 16 (mass 53%) as the
claiming zone, or explicitly re-scope S4 to "beat the floor" only where the
floor is beatable and quote arch-ordering elsewhere (the qrate/R16
precedent).** T8 floor is 0.136 with mass 31% — the structurally cleanest
cell if recovery materializes there.

## 1. REBUTTAL_PACK add-on block (append as § 2b "The hunt's selection
instruments, demonstrated on one night" — optional, meeting color)

> Overnight the same pre-registered instruments processed four new
> candidates at screen stage: two died label-side for $0 before any GPU
> (turn-tempo: Spearman −0.81…−0.83 against the already-confirmed
> turn-length trend — near-duplicate; question→answer latency: 84% of
> questions resolve in exactly one turn — no variance, and the anchor
> carries a visible "?"), one died at screen (correction hazard: real
> cue-free persistent state, but window-MEAN matches full order-aware
> readout at every T ≥ 8 — order-free aggregation, the class that is
> pooling-matchable), and one is a panel-gate candidate (conversation
> novelty: KEEP 3/3 models, window-over-token +0.084/+0.101/+0.084
> [gpt2/llama/gemma screen-acc units], within-dialogue order margins
> +0.026/+0.031/+0.039 at T32, position and identity traps passed).
> Three of five designed candidates were killed by their own
> pre-registered controls — the falsifiers do the selecting.

Screen-instrument labeling rule carried: all § 1 numbers are probe-acc
units on the screen instrument; no trained-dictionary numbers exist for
these candidates yet (cnov's panel would be the first).

## 2. WRITEUP § 8 rows (kills table — three new rows, verbatim candidates)

> | turn-taking tempo trend (`tempo`) | label pre-measure | Killed for $0 before screening: DailyDialog is strict two-party alternation, so alternation tempo is the confirmed turn-length trend in a hat (Spearman −0.81…−0.83) — screening it would have manufactured a duplicate win. |
> | question→answer latency (`qres`) | label pre-measure | Killed for $0 by its own gate: 84% of questions resolve in exactly one turn (no variance to probe), and the anchor turn carries a visible "?" — the question-gap demotion marker one step removed. |
> | correction hazard in reasoning traces (`chaz`) | screen | Cue-free persistent state is real (out-of-window cue mining), but window-MEAN matches the order-aware readout at every T ≥ 8 (Δ ≤ 0.025) — order-free aggregation; confirms the self-correction "aggregation bonus" reading at one remove. |

## 3. WRITEUP breadth-table entry (nvtrend)

> | novelty-rate trend (`nvtrend`, dialogue) | screen KEEP 3/3 (window-over-token +0.065…+0.096 screen-acc) but order-free (0/3 models at the +0.03 order bar; margins ≤ +0.017) — routed to breadth by the frozen order rule; its gain is pooling-matchable aggregation. |

## 4. WRITEUP cnov paragraph (CONTINGENT — only if the 17:00 pick selects
cnov AND the panel then KEEPs; staged early per the one-pager clock)

> **Candidate under panel test — conversation novelty.** The strongest
> structural guarantee in the search so far: the label is the trailing rate
> of FIRST-IN-CONVERSATION token types, so whether a type is new is
> uncomputable from any T-window that starts after the conversation did.
> The screen KEEPs it on 3/3 models with within-dialogue order carriage on
> 2/3 at T = 32; the pre-measured in-window floor rises steeply with T
> (r = 0.14/0.27/0.40 at T = 8/16/32) exactly as the kernel's in-window
> mass grows — the pre-registered claiming zone puts the panel where the
> floor is lowest and the guarantee strongest. [Numbers + verdict per the
> frozen panel card once run.]

## 5. Not drafted

The cnov panel card itself is mac-a's (launch-prep assignment); § 0's S4
flag goes to its freeze-review rather than into the card text by me. No
REBUTTAL_PACK § 1/§ 2 (λ̂/ttrend) edits — those sections are ratified.

*mac-b; drafts pending mac-local ratification. Sources:
`hunt3/results/verdict.json`, `hunt3/results/panel_evidence_line_cnov.json`,
`labels/hunt3_stats.json` + mac-a's LOG entries (~03:xx–04:30) for the
tempo/qres/chaz numbers.*
