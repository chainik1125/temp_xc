# Working state — agent `runpod-e`

**Last rewrite:** 2026-07-24 (round 2 items 1 + 2 EXECUTED and
committed; closing out. Awaiting mac-local review.)

## Who / where / env (verified working this session)
H100 pod, `/workspace/temp_xc`, `/workspace/.agent_id` = `runpod-e`.
Git `runpod-e-agent`, creds `store --file=/workspace/.git-credentials`
(token from `/workspace/.tokens/gh_token`); `export HF_TOKEN=$(cat
/workspace/.tokens/hf_token)` per command; anthropic key at
`/workspace/.tokens/anthropic_key`. Pull-rebase before EVERY push
(shared `arxiv` branch). venv `.venv/bin/python` (torch 2.8 cu128,
transformers 5.7). 224 tests pass; `run.py validate` green.

## Round 2 (`briefings/task-hunt-r2-e.md`) — BOTH ITEMS DONE

### Item 1 — hedging-trend LEVEL Stage 2: **NEGATIVE**
Card `task_hunt/confidence/card_stage2.md` (§§1–9 `fff7877c`, §10
amendment `606a8015`) frozen before any cell. 84/84 cells ok,
datasource `ward_real_slope8_distill_l14` (plugin
`src/explorations/task_hunt/real_slope.py`). Verdict + full numbers:
`task_hunt/LOG.md` (round-2 entry) and `task_hunt/RECORD_B.md` §1.
- Recovery **peaks at T = 4 and declines**; exact within-seed trend
  permutation p = 0.73 (pre) / 0.96 (post) / 0.50 (stacked) → the
  card's NEGATIVE clause fires.
- One bounded positive: TXC-pre/T4 beats per-token SAE (+0.055, CI
  [+0.007, +0.103]) and T-SAE (+0.037, CI [+0.012, +0.062]), 3/3 seeds.
- **All archs budget-matched** (l0 6.3–8.1); matched-post falsifier
  passed exactly (untrained = 8.00/token). runpod-d's k = 8·T
  convention works.
- Sharpest fact: RAW per-token reference r = 0.221 — only 1 of 14
  panel cells exceeds it. Stage-2's unmatched sampling re-admits the
  ambient anchor-state route the Stage-1 screen matched away → the
  screen's "per-token-blind" premise does NOT transfer. **This is the
  generalizable lesson for every Stage-1 → Stage-2 promotion.**
- Shuffle receipt ran but is DEGENERATE (frozen criterion needs a
  margin the panel didn't produce) — reported, no claim drawn.
- Scorecard: P1/P2/P3/P5 falsified, P4 partial; Stacked large-T
  pathology recurs (T16 trained 0.129 < untrained 0.157).

### Item 2 — early-layer addendum: DONE (`task_hunt/depth_addendum/`)
Predictions + script frozen at `e4caddf6` before any cell. 133 cells,
figures rendered, RECORD_B §2, LOG entry.
- lag4: A2/A3/A4 confirmed (order signal grows toward the input; the
  round-1 scale ordering closes early). **A1 falsified** → new atlas
  shape **present-then-discarded** (per-token lag is HIGHEST at the
  earliest layer, monotonically discarded with depth). **A5 refined**:
  short-T g_order is mostly anchor-vs-context separation, not order —
  `g_order = flatten − mean` conflates the two; the anchor-fixed
  shuffle isolates order. Round-1 replag KILL holds depth-wide.
- slope8: B1/B2/B3 confirmed (g_agg > 0 in all 34 cells incl.
  embeddings; per-token never exceeds 0.483 at ANY depth → the trend
  is never converted). **B4 falsified**: mid-depth VALLEY with peaks
  at both ends, not a mid peak.

## IN FLIGHT at last rewrite
`confidence/probe_capacity.py` (card §6 pre-registered post-hoc
diagnostic) re-running after a self-caught reshape bug (first revision
mismatched `stacked`'s (B,T,d_sae) code; failed loudly, no results
written; fix committed and disclosed). **The four TXC cells already
logged are decisive and unaffected:**

| cell | panel (OLS, nw1024) | ridge nw1024 | OLS nw8192 | ridge nw8192 |
|---|---|---|---|---|
| pre/T4 | 0.210 | 0.302 | 0.248 | 0.274 |
| pre/T16 | 0.134 | **0.324** | 0.246 | 0.311 |
| post/T4 | 0.163 | 0.256 | 0.238 | 0.255 |
| post/T16 | 0.167 | **0.318** | 0.258 | 0.294 |

**→ The T-decline is a PROBE artifact.** Under ridge on identical
codes, T16 ≥ T4 for both window archs — the ordering reverses. The
frozen NEGATIVE verdict stands under the frozen metric (the card
pre-registered that this diagnostic "cannot change the leaderboard
cells; it can only change what the record is allowed to claim"), but
the record must state that `lambda_recovery`'s unregularized
LinearRegression on p = d_sae with n shrinking as 1/T is
**systematically biased against large T**. **This very likely also
explains the λ̂ panel's T = 16 dip and the Stacked pathology
(runpod-d / RECORD §3b) — flag to mac-local + runpod-d.**
Next action if interrupted: read `results/stage2_probe_capacity.json`,
finish RECORD_B §1d + the LOG addendum, push.

## Round-1 state (closed; context only)
Three arm-B candidates KILLED under frozen cards (replag, confidence
trend, emotional instability) — verdicts in LOG.md, reviewed and
approved. Volume assets: `/workspace/{replag_caches,conv_depth_caches,
emo_caches}`. Hedging-trend is now closed on BOTH faces (trend face
round 1, level face round 2) — recommend no third panel.

## Sibling context
runpod-d (r2): budget-matched TXC-post re-run on ward_real_lambda —
its k = 8·T convention is validated by my panel's falsifier; my
probe-capacity finding bears directly on its T = 16 cells.
runpod-b: hunt-support-stats COMPLETE (variance-aware renderer merged;
I reuse it via `confidence/render_stage2.py`, own match rule because
post's nominal k varies with T).
