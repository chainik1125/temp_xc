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

## §3 QUEUE (quantity mode) — 2 bundles screened, verdicts posted

**`novelty` — NEGATIVE** (`novelty/CARD.md` frozen `3f18b5eb`). Gap
tops out +0.045/+0.038/+0.039 vs a +0.05 bar and PEAKS MID-LADDER while
kernel mass rises; N1 falsified informatively — per-token beats the
position floor by +0.09..+0.15, i.e. **71-77 % of the window-readable
signal is already per-position** (conversion with a small residue).
Shuffle-null receipt clean (+0.076..+0.119; the null face has NO window
gap). No KILL rule fired either — recorded as the card's missing middle
clause, not patched post hoc.

**`punctint` q face — KEEP, the hunt's first** (`qrate_fineweb/CARD.md`
frozen `74af1d4a`; scorer + control committed `c4f0f16b` BEFORE
running). All KEEP clauses fire 3/3: gap rises monotonically to
+0.114/+0.127/+0.143 at T64, tracks the measured kernel-mass column,
still rising at the disclosed reach limit; ambient anchor LOSES from
windows (advantage is face-specific, not generic width).
**Doc-identity confound found and controlled** (doc-mean-only AUC
0.926 — the frozen factory bars cannot see it): within-document
contrast SURVIVES at +0.101/+0.132/+0.183 over 24-26 test docs.

**`punctint` list face — WEAK KEEP, disclosed.** 2/3 models fire, but
its anchor GAINS from windows, doc-mean-only AUC 0.960, and its
within-doc control rests on only **8 test docs** (88.5 % zero-inflation).
If one face is promoted it should be q.

**Posted a factory recommendation:** add `doc_mean_only_auc` as a
triage bar (novelty 0.792 / q 0.926 / list 0.960 — batch-wide route)
and make within-document contrasts the standard control for any KEEP.

**If q goes to Stage 2, that panel MUST use a capacity-adequate
λ-probe** (ridge or n scaled with T) — my probe-capacity entry shows
the current evaluator is biased against large T, exactly where this
candidate's signal lives.

### Queue remaining (mine, per r2-e §3)
`tss` (interleave; needs one ~330k-token caching pass per model) and
`dialevel` (caching + mac-local qualification 2: within-dialogue
contrasts or dialogue-length matching, all-row position 0.93).
Claim-lines rule: post "screening <bundle>" in the LOG first.

## Hygiene at last rewrite
280 tests pass, 1 skipped; `run.py validate` OK; leaderboard 8796 rows,
**0 dup eval_keys, 0 null metrics**.

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
