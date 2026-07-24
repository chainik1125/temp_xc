# Working state — agent `runpod-d`

**Last rewrite:** 2026-07-24, PRE-COMPACT, at the start of round 2.
Round 1 is finished, reviewed and APPROVED. Nothing from round 2 has
been run yet. Read § "DO THIS NEXT" first.

## Who / where / setup (all already built — reuse, do not rebuild)
GPU RunPod pod (H100 80 GB, 224 cores, 2 TB RAM), `/workspace/temp_xc`,
`/workspace/.agent_id` = `runpod-d`. **I am runpod-d, NOT runpod-e.**
Round-2 briefings are now split per pod: mine is
`briefings/task-hunt-r2-d.md`; `…-r2-e.md` is runpod-e's and is not my
work. (The combined `task-hunt-r2.md` was split by mac-local at
`766d6142` — if a stale reference to it appears, that is why.)

- `.venv` = probe/training venv (torch 2.8+cu128). `/workspace/vllm_venv`
  = separate vLLM 0.25.1 venv (has pandas + ninja).
- `HF_HOME=/workspace/hf_cache`; creds `/workspace/.tokens/`.
  Git: identity set, `core.askPass=/workspace/.tokens/git-askpass.sh`.
  **Branch `arxiv` is shared by 5 agents — `git pull --rebase` before
  every push.** `results/leaderboard.jsonl` + `checkpoints/manifest.jsonl`
  are usually dirty from live runs; `git stash -u` → pull → `stash pop`
  is the working idiom.
- Caches on the volume: `/workspace/conv_depth_caches/{ward_stream,base,
  distill}`, `/workspace/task_hunt_labels/lambda_intensity/` (incl. the
  DENSE Stage-2 grids), `/workspace/task_hunt_labels/forbidden_word/`
  (rollouts + acts + 167 GB acts_depth).

## DO THIS NEXT — round 2, **`briefings/task-hunt-r2-d.md`** (mine)

**My assignment is ONE run** (+ its figure). Deadline: results by
**Saturday 2026-07-25 morning PT**; check-in Sunday 10:00 PT.

### 1. Budget-matched TXC-post re-run (the whole job)
Round 1's TXC-post cells were budget-confounded: realized
`l0_per_token` collapsed as T grew, so its monotone rise to 0.255 at
T = 16 is not a matched-budget win. Re-run it with realized-l0 matched.

**The mechanism (already worked out — put it in the card):**
`txc_batchtopk_post` does BatchTopK on the *squashed* window code with
budget `k_win // T` atoms per WINDOW, then decodes all T positions, so
**realized l0_per_token ≈ nominal_k / T**. Round-1 evidence at nominal
k = 8 (predicted 8/T vs observed):

| T | predicted 8/T | observed l0 | recovery (round 1) |
|---|---|---|---|
| 2 | 4.0 | 3.42 | 0.130 |
| 4 | 2.0 | 1.80 | 0.161 |
| 8 | 1.0 | 0.94 | 0.185 |
| 16 | 0.5 | 0.49 | **0.255** |

⇒ to hit realized l0 ≈ 7–8 at every T, set **nominal k = 8·T**:
**k = 16 / 32 / 64 / 128 at T = 2 / 4 / 8 / 16.** (Observed/predicted
ratio runs 0.85→0.98, so realized will land ≈ 6.8–7.8; verify from the
rows and report actuals, do not assume.) Dict constraint is fine: post
is NOT in `design._POOLED`, so it needs only `d_sae ≥ k_pos`, and
d_sae = 2048 ≫ 128.

**Steps:**
1. Freeze a short **amendment card** (commit BEFORE running) stating:
   the l0 ≈ k/T mechanism + the round-1 table above, the per-T nominal
   k, that this deliberately deviates from the program's equal-nominal-
   k_pos fairness rule in favour of the briefing's matched-REALIZED-l0
   requirement, and the two pre-registered readings:
   - **(a)** the rise survives matching ⇒ money plot upgrades from
     "TXC-pre peaks at T=8" to a monotone matched-budget line through
     T=16 (materially stronger rebuttal figure);
   - **(b)** it does not ⇒ the 0.255 was sparsity-starvation behaviour,
     recorded as such; **TXC-pre remains the headline**.
2. Run post × T ∈ {2,4,8,16} × seeds {1,2,42} + untrained ≈ 24 cells.
   Reuse `lambda_intensity/run_stage2.py` machinery
   (`explorations.synthetic.design.uniform_cells` + `grid.run_pool`),
   but `uniform_cells` takes ONE `k_pos_sweep` for all T — so emit the
   cells with **per-T k** (small variant fn; keep d_sae=2048,
   eval_window_L=32, n_steps=8000, `buffer_tokens=524288` — the
   corpus-sized buffer, the 2M default is 4× oversampling and dominates
   wall-clock).
   **Write to a SEPARATE results file** (e.g.
   `results/stage2_postmatched_ward_real_lambda_base_l12.json`) so the
   matched cells never silently mix with the round-1 nominal-k=8 cells;
   the renderer merges and labels by realized l0.
3. **Figure (review note 3, MANDATORY before any external use):** the
   stage2 figure must annotate TXC-post's realized l0 — visually it
   reads as the winner and it is not budget-matched. The variance-aware
   renderer (l0 legend + seed-CI whiskers) is **owned by runpod-b**
   (`briefings/hunt-support-stats.md` item 2): re-render with theirs
   once my cells land; **if it has not merged when I finish, do the
   minimal l0 annotation myself rather than idle**, and reconcile in
   the LOG.
4. runpod-b may post a LOG recommendation to append ~12 cheap seed
   cells (pre + tsae at T ∈ {4,8}); **treat as part of this run if it
   lands before I finish.**

### 2. PARKED — do NOT run
proof-op Stage-2 on distill L12 (contrast +0.017…+0.042 too thin to
clear a trained panel by Saturday; post-rebuttal). Also parked:
gpt2-scale order cell, anti-conversion candidate class.
**runpod-e's items are NOT mine**: hedging-trend LEVEL Stage-2, the
early-layer g_order(ℓ)/g_agg(ℓ) addendum.

## Round-1 numbers I will need (verified, from `stage2_summary.json`)
Panel: d_sae 2048 (= d_in/2), nominal k_pos 8, eval_window_L 32,
n_steps 8000, 3 seeds; datasource `ward_real_lambda_base_l12` (plugin
`src/explorations/task_hunt/real_lambda.py`); metric `lambda_recovery`
(held-out Pearson r, per-tile leading-edge readout, chance ≈ 0).

| arch | T=1 | T=2 | T=4 | T=8 | T=16 | realized l0 |
|---|---|---|---|---|---|---|
| per-token BatchTopK SAE | 0.113 | — | — | — | — | 6.3 |
| **T-SAE** | **0.154** | — | — | — | — | 7.4 |
| Stacked | — | 0.109 | 0.143 | 0.125 | 0.094 | 7.0–7.9 |
| **TXC-pre** | — | 0.132 | 0.192 | **0.206** | 0.138 | 6.9–7.8 |
| TXC-post | — | 0.130 | 0.161 | 0.185 | **0.255** | **3.4→0.49** |

TXC-pre untrained falls with T (0.091→0.088→0.056→0.013), so its
trained−untrained margin GROWS to +0.150 at T=8 — that, plus the
T-rise, carries the claim (review note 2: the pre − T-SAE margin is
only ≈2σ at n=3, phrase variance-aware). Stacked at T=16 is a training
pathology (trained 0.094 < untrained 0.171).

## Binding conventions adopted at review (apply to everything downstream)
1. **Per-token-first triage** — before any window grid, run the
   per-token linear probe alone; a high per-token ceiling ⇒
   presumptively converted ⇒ escalate only with a card-stated reason.
   All five round-1 kills were visible in that one number.
2. **The depth sweep is the cheap WHY-diagnostic** when per-token is
   high. Fourth g(ℓ) shape on record:
   built-and-immediately-linearized.
3. **Screening question:** "will the model decline to maintain this as
   a per-position state?" — NOT "is the concept semantically
   non-obvious".
4. **"Conversion, not circling"** supersedes the lexical language in
   the cand-3 kill entry and every downstream summary.
5. Rebuttal sentences about Stage 2 must say **"under the code-readout
   convention"** (one tile's code per prediction) and carry the
   code-rate defense (pooling T-SAE codes over T positions would spend
   T× the bandwidth a window arch uses).

## Round-1 verdicts (all committed, reviewed, APPROVED — do not redo)
- **Cand 1 (λ̂ intensity): KEEP → Stage-2 QUALIFIED POSITIVE** (the
  money plot; TXC-pre beats per-token/T-SAE at matched l0, rises to
  T=8; order story negative — regime 2).
- **Cand 2 (proof-op runs): KEEP** — the MODEL AXIS is the finding
  (distill L12 clears the null at every T; base ≈ distill falsified).
  Stage 2 parked.
- **Cand 3 (forbidden-word onset, SILOED): KILL** (pre-registered
  ambience kill) + POST-HOC depth sweep ⇒ mechanism is **CONVERSION**,
  gap shut in 49/51 cells; my original bag-of-words explanation was
  wrong and the LOG carries the correction.
- **Shuffle receipt: POSITIVE** — backtracking ANTICIPATION is
  order-sensitive (+0.028…+0.041) vs ambient `is_bt` (+0.003…+0.013),
  fixed T=16, identical rows.

## Traps that already cost time this session
1. **Never emit NaN into a leaderboard metric.** The leaderboard IS the
   eval cache; JSON stores NaN as `null`; `LeaderboardRow` then rejects
   the cached read and the canonical artifact becomes unloadable for
   EVERY subsequent run. (Fixed by giving the plugin datasource a
   documented reference basis for `emission_features`.)
2. **Never write a wait loop as `pgrep -f "<pattern>"`** — a monitoring
   shell containing that pattern matches itself and the loop never
   exits.
3. **GPU serialization is mandatory** — concurrent probe jobs + a
   training pool OOM'd an 80 GB H100. Chain, don't fan out.
4. **The distill tokenizer**: `AutoTokenizer` resolves R1-Distill to the
   SLOW `LlamaTokenizer` whose `return_offsets_mapping` is unusable and
   fails SILENTLY. Force `PreTrainedTokenizerFast`.

## Acceptance gate (round 2)
Amendment card frozen pre-run; LOG verdict; figure + record;
leaderboard hygiene (0 dup keys, no null metrics); STATUS rewritten.
Briefing stays until mac-local review. No reviewer/meeting quotes in
tracked files.
