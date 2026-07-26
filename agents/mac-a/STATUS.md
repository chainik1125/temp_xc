# Working state — agent `mac-a`

**2026-07-26 ~13:55 London — day-2 W2: screen DELIVERED (both faces
KEEP 3/3, T32 order carriage); after the gate race (fired tt
`dce8d085d` → amended dq `187c51022` → RESOLVED `6e2f18e4e`: tt
governs + PANEL 2 dq authorized, my cap → $200) — **TWO panels
running detached**: tt/gpt2 (freeze `7ba2e10fd`, app relaunched
13:40 after my amendment-triggered stop, log
`<scratchpad>/tt_panel_relaunch.log`) and dq/llama31 (freeze
`cfa341c34`, launched ~13:47 — 2 min over the 13:45 line, disclose
in verdict; log `<scratchpad>/dq_panel2.log`). mac-b = panel-2
merge/receipts support. Verdict engine ready: `score_panel.py`
(P1–P6), `merge_panel_payload.py tt|dq` (mac-b filled dq SHA),
evidence lines committed (tt drawn-only; dq P6 KILL bars |r|
0.106..0.499).**

## Live state (if resuming mid-anything, start here)

- **Panel in flight**: app `mac-a-diafaces-panel`, freeze
  `7ba2e10fd`, DS `dial_real_ttrend_gpt2_l7` (gpt2/hs7), 102 cells =
  λ̂ shape + T32 column. 1× H100 `--block main` + 3× L4 high-CPU
  `--block tsae --only-seed {1,2,42}`. Launched ~13:17 detached
  (client under caffeinate; log
  `<scratchpad>/diafaces_panel.log`). Payloads persist to Volume
  `temp-xc-replag-caches:/workspace/diafaces_panel/payload_*.json`
  even if client dies. **Repatriation deadline 16:15 London
  (mac-local binding term); nothing new after 15:30; all pushed by
  16:30; check-in 18:00.**
- On completion: `results/panel_payloads/payload_*.json` →
  `.venv/bin/python -m experiments.explorations.task_hunt.diafaces.merge_panel_payload`
  (freeze-stamp + clean + dedup asserts) → score P1–P5 per
  `PANEL_CARD.md` § 5 (KEEP iff P1∧P5) → LOG verdict PENDING TEAM
  REVIEW + receipts proposal + ledger actuals.
- Modal client: scratchpad `modal-venv/bin/modal` (repo .venv has no
  modal; the overnight venv survives in this session's scratchpad).

## Delivered today (all pushed)

- Screen: freeze `073611113` (approved pre-results `2f2bdd91d`);
  3/3 screens repatriated + committed; VERDICT in LOG (tt + dq KEEP
  3/3; wd_sc T32 +0.034..+0.049 on 9/9, T16 nonneg; dq Q1 violated
  3/3 disclosed; tt floor-crossover at T ≥ 32 disclosed). R26
  proposed (screen claims). My dq panel proposal SUPERSEDED by
  mac-local's written tt gate — executing tt, rationale + flagged
  llama alternative in `PANEL_CARD.md` § 2.
- Panel prep/freeze: `real_dialogue.py` plugin, YAML DS entry,
  `run_panel.py` (102 cells verified), `PANEL_CARD.md` (P1–P5 bars
  frozen), `merge_panel_payload.py`, modal driver.
- Ledger: mac-a day-2 actuals ≈ $5 (screen) + panel est ~$10;
  total ≈ $56 est of $500/$400; my cap $120.

## Context worth keeping

- W1 (mac-b): R11 mechanism = MIXED (L1+L2 additive ±0.005,
  near-half concentrated), R25 ratified — tt is that mechanism's
  state variable (the gate's scientific basis).
- Gate clause readings pinned pre-results: (ii) = T32 ≥ +0.03 AND
  T16 > 0, ≥2/3 incl {gpt2, llama31}; panel MUST include T32.
- dq KEEP goes to the breadth table with order numbers quoted (per
  the gate entry) — nothing further for mac-a unless asked.
