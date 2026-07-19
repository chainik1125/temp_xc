# Working state — agent `runpod`

**Last rewrite:** 2026-07-19 (expansion Cycle 3 COMPLETE — stopped for review).

## Who / where
Remote CC on RunPod (Linux) at `/workspace/temp_xc`. Git creds at `/workspace/.tokens/`
(push: `git push https://x-access-token:$(cat /workspace/.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`).
Claude API key at `/workspace/.tokens/anthropic_key` (export as ANTHROPIC_API_KEY;
all 3 judge roles verified OK this cycle).

## Last task: `briefings/grounded-benchmark-expansion-cycle3.md` — DONE
All stages + the hedging rider executed autonomously; **STOPPED for human
review** (no Cycle 4 before review; briefing stays until reviewed, then delete
— the expansion README is the standing doc).

Outcome (full detail: synthetic STATUS §0 + `expansion/LEDGER.md` cycle log):
- Menu extensions `hier_ar1` + `periodic_hawkes` built + tested (13/13);
  uniform relative gate-8 rule (±20% + floors) preregistered; 4 categorical
  int/eq cards frozen; blind selection; 6 calibrated (3/domain); $8.20/$25.
- **list-item-parallelism-r2 PROCEED → `synthetic/list_item_parallelism/`
  SPEC — first text-corpus benchmark** (re-filed to bursty/self-exciting by
  measured class).
- **Hedging rider PASS → hedging_drift SPEC*→SPEC** (hier_ar1 holds the
  plateau; spec amendment + `mirror_params_hier.json` written).
- 5 ABORTs, all informative (see LEDGER cycle log): comp-verif-r2 skeptic
  circularity (hybrid passed gate-8!); both categorical int/eq cards REAL
  signals killed on mirror fidelity (categorical plateau ⇒ C4 needs a
  hierarchical categorical mirror); enumeration-cadence rhythmic+bursty;
  goal-restatement composition kill. int/eq target honestly NOT met.

## Next / open
- **Blocked on user review of Cycle 3.** C4 candidates queued in the LEDGER
  cycle-log lessons: hierarchical categorical mirror (would plausibly convert
  both int/eq aborts on re-freeze); periodic cards need a preregistered
  non-inserted gate-8 moment (gap-shape / cross-doc period stability).
- Stage-6 blind B×A eval now has THREE full SPECs waiting (assumption_
  consequence, hedging_drift, list_item_parallelism) + the anchor — needs a
  user green-light (datasource plugins to write; nothing run).
- `results/leaderboard.jsonl.prepurge` backup can be deleted (push confirmed).

## Gotchas (this box)
- **Pod restart wipes the home dir ⇒ the venv's uv-managed Python vanishes**
  (`.venv/bin/python` → broken symlink into `~/.local/share/uv/`). Fix:
  `curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.12.13`
  — site-packages live in the volume-backed `.venv/`, so nothing else is lost.
- Claude 5-family models reject `temperature` AND think by default — tight
  max_tokens ⇒ empty text (client handles both).
- Calibrations must run SEQUENTIALLY: the spend meter is per-process
  file-persistent; concurrent writers undercount.
- Tiny models: GPU useless here; CPU ~12 workers OMP=1 for temp_bench grids.
- `pkill -f` self-matches the launching shell → use TaskStop on harness tasks.
- Background python: launch with `-u` or prints sit in the block buffer.
