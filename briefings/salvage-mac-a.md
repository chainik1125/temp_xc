---
status: active
created: 2026-07-26 ~16:30 London
for: mac-a (executor) — salvage W1: ttrend POST-claiming fresh-seed panel
read-first: briefings/salvage-shared.md
---

# Salvage W1 — turn-length trend, TXC-post claiming, FRESH seeds

**Why this is the top salvage.** The tt panel (your own,
`dial_real_ttrend_gpt2_l7`) failed ONLY because the frozen claiming
set was pre/stacked and those failed the untrained control. The POST
arm had the panel's cleanest profile and could not claim: trained
0.1421 (T16) / 0.2968 (T32) vs untrained −0.0084 / 0.0037; the
visible-cue floor is DEGENERATE at T ≤ 16 (active 1.8% of rows,
r = 0.015) and only 0.114 at T32 — post beats it 2.6×. And the task
("is this conversation's turn length trending up or down") has NO
surface-count reading: a trend requires comparing levels at
different distances. mac-local's review already flagged the
post-scoped follow-up; Han green-lit it tonight.

**The design (freeze your card first; this is a NEW pre-registration,
not a re-scoring):**
- Fresh seeds **{3, 4, 5}** — the first-look hazard (post was
  observed on seeds {1,2,42}) is neutralized by confirming on seeds
  the observation never touched. State this in the card.
- Cells: `txc_batchtopk_post` at T ∈ {2,4,8,16,32}, k = 8·T,
  trained + untrained, seeds {3,4,5} = 30 cells; PLUS fresh-seed
  baselines `batchtopk_sae` + `tsae` at T = 1, trained + untrained,
  seeds {3,4,5} = 12 cells. **42 cells total**, gpt2/hs7 d768 —
  cheap (the 99-cell main block ran in ~1 h on H100).
- Same datasource, same frozen probe conventions, paired v1+v2 on
  every row (assert eval_extra per cell pre-commit — the defect
  check), conversation-grouped v2 split, numeric realized-l0 band
  in the card, evidence line = the committed panel_evidence_line
  values (0.015 at T16 / 0.114 at T32; T ≤ 8 degenerate — state it).
- **Pre-registered bars (freeze before any cell):**
  - S1: post trained beats BOTH fresh-seed per-token baselines at
    T ∈ {16, 32} by ≥ +0.05, CI clear of 0.
  - S2: untrained post ≤ 0.5× trained at the claiming T (expect
    ≪ that — prior data ~0.01×).
  - S3: T-scaling — trained post rises T8 → T32, exact within-seed
    permutation p reported.
  - S4: beats the evidence line at the claiming T (KILL clause,
    oprate style; trivially true at T16 if S1 holds — the bar is
    0.015 — but state it).
  - S5: conversation-grouped v2 > 0 at the claiming T.
  - KEEP iff S1 ∧ S2 ∧ S4 ∧ S5. Negative/weak outcomes reported at
    full prominence — if fresh seeds kill the post observation,
    that is exactly what fresh seeds are for.
- Venue: H100 main + the tsae fresh-seed cells on high-CPU (they
  are T = 1 d768 — small; whatever the measured cost, tsae-first
  scheduling). Est ≤ $15 of your $100 cap.

**Deliverable:** LOG verdict (PENDING TEAM REVIEW), receipts
proposal, leaderboard rows (0 dups), and — if KEEP — a fig in the
WRITEUP style (post + baselines + untrained + evidence line vs T)
dropped in `figs_writeup/` as `fig4_ttrend_post_confirmation.*` with
a caption block proposed for mac-local to integrate.
