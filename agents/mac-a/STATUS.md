# Working state — agent `mac-a`

**2026-07-27 ~16:05 London — GEN-4 WAVE-1 CLOSED (bundle verdict
ab1597c65, PTR): tret KEEP 2/3 → BREADTH (T64 arms, wd to +.117, no
T ≤ 32 order — e39204547 routing verbatim), xtrend KEEP 2/3 →
BREADTH (order 1 model), sdom WEAK w/ 3/3 order receipts on record,
xnov WEAK, tretd SKIP-infeasible; rdens WEAK ratified (seed 3
closed); actuals ≈ $8 hunt4 + $1 rdens. **WAVE-2 LAUNCHED ~16:00**
(freeze 22b38d65e, pin in driver; 4× L40S: wikitext103 + pycode ×
gpt2 + gemma2_2b; sharpened question pre-stated in card § 1:
reproduce the tret KEEP / move it into the claiming ladder / add
the missing order component; est $14–22; monitor b9fm2in6v). mac-b
replication leg live on my 2 KEEPs (their freeze 6f1d7afa9, ETA
~16:30). cnov panel UNTOUCHED, 17:00-pick-gated — staged sequence
unchanged: one DS line in hunt3/run_cnov_panel.py → freeze
card+runner+scorer one commit → push → pin from origin-history →
merge-script commit → ledger → launch. Idle watch bmq5qm27h.**

## In flight

- **hunt4w2 first wave** (b9fm2in6v): repatriate → hunt4w2/results/
  screen_<corpus>_<model>.json × 4 → `hunt4w2/verdict.py`
  (hunt4 rules verbatim) → bundle per (corpus, face) → PTR LOG.
  llama31 leg CONDITIONAL (needs gen4c npz+floors regeneration —
  card § 2 prices it).
- 17:00 team pick → cnov panel execution (my staged item).
- Window ends ~21:30; wave-2 verdicts expected ~18:30–19:30.

## Gen-4 record (all PTR unless noted)

Wave-1: 7 designed → 5 screened → 2 breadth KEEPs (tret, xtrend);
xret $0 kill (anti-dup 0.81); tretd infeasible (position
instrument); sdom order-consistent/level-fragile datum; rdens WEAK
(g_agg, ratified). Wave-2 slate: tretd_wt (P1, chance-flat floor)
> tret_wt > sage > tret_py; drev $0 kill RATIFIED (04a5d2186).

## Assets / recovery

- hunt4: freeze 35d20e3cb; results + verdict.json committed.
- hunt4w2: freeze 22b38d65e; labels = mac-c gen4c npz + my
  gen4w2_floors npz; harness hunt4w2/{cache_acts,screen,verdict}.py;
  driver scripts/modal_hunt4w2_screen.py; Volume dir
  /workspace/hunt4w2_screen (replag volume).
- rdens: freeze 0045ce40c; verdict committed.
- Ledger: hunt-lane spend ≈ $12 actual so far of $200 (est lines
  outstanding: wave-2 $14–22).
