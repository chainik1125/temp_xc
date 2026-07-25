# Working state — agent `runpod-d`

> **⚠ MAC-LOCAL BANNER (2026-07-25, appended by the orchestrator — read
> before anything below): YOU ARE ASSIGNED. `briefings/stage2-oprate.md`
> (for: runpod-d) is ACTIVE and UNCLAIMED — it is CASE STUDY #2 and it
> outranks every item in "DO THIS NEXT" below, including your
> self-proposed follow-up. Post the LOG claim line, freeze the card,
> tsae cells first. Full context: the LOG's 2026-07-25 factory-r3
> review entry, "OPERATIONAL DIRECTIVE".**


**Last rewrite:** 2026-07-25 ~02:30 UTC, PRE-COMPACT. **All assigned work
is finished, pushed, reviewed and APPROVED; my briefing has been
RETIRED. I am currently UNASSIGNED.** Read § "DO THIS NEXT" first — the
top item is a follow-up that tests whether my own results overstate.

## Who / where / setup (all already built — reuse, do not rebuild)
GPU RunPod pod (H100 80 GB, 224 cores, 2 TB RAM), `/workspace/temp_xc`,
`/workspace/.agent_id` = `runpod-d`. **I am runpod-d, NOT runpod-e.**

- `.venv` = probe/training venv (torch 2.8+cu128). **No `pip` in it** —
  do not try to install packages there (py-spy is unavailable; I wasted
  time trying). `/workspace/vllm_venv` = separate vLLM 0.25.1 venv.
- `HF_HOME=/workspace/hf_cache`; creds `/workspace/.tokens/`.
  Git: identity set, `core.askPass=/workspace/.tokens/git-askpass.sh`.
  **Branch `arxiv` is shared by ~6 agents — `git pull --rebase` before
  every push**, and expect to be 30+ commits behind after any long run.
  `results/leaderboard.jsonl` + `checkpoints/manifest.jsonl` are usually
  dirty from live runs; `git stash -u` → pull → `stash pop` is the
  idiom. Both have a **`merge=union`** driver (`.gitattributes`) — jsonl
  conflicts auto-keep both sides. LOG.md conflicts are append-only:
  resolve by **keeping BOTH blocks in order** and deleting the 3 markers.
- Caches: `/workspace/conv_depth_caches/{ward_stream,base,distill}`
  (17 layers each), `/workspace/task_hunt_labels/lambda_intensity/`,
  `/workspace/task_hunt_labels/forbidden_word/` (incl. 167 GB acts_depth).

---

## STATE — everything closed, nothing in flight

No background job is running. Working tree clean apart from the live
leaderboard/manifest. **0 unpushed commits.** 309 tests pass. Leaderboard
8,820 rows, **0 dup eval_keys, 0 null metrics**.

`briefings/task-hunt-r2-d.md` was **RETIRED** by mac-local at `fbab4070`
(assignment complete). The only active briefings are `mirror-probe-truth.md`
(runpod-b) and `em-redo.md` (runpod-c). **Neither is mine.**

### A. § 1–2 Budget-matched TXC-post + probe capacity — CLOSED, APPROVED
Card frozen `07c90cfb` pre-run; results `ff3c5618`; verdict `2b64dbe4`.
Reviewed **APPROVED** at `c9580fad`.
1. **Reading (b) confirmed** — round-1's TXC-post 0.255 @ T16 was
   budget-confounded. Falsifier exact: every untrained matched cell
   realizes `l0_per_token` = 8.000 at every T. Matched post peaks at T4
   (0.202) then falls to 0.137, into the TXC-pre (0.138) / Stacked
   (0.094) band ⇒ **TXC-pre (peak T8 = 0.206) remains the headline.**
2. **Reading (c) confirmed, bigger than pre-registered** — the λ-probe
   (unregularized OLS, p = 2048, **n = p at T16**) is capacity-limited
   for DENSE codes. Lift under ridge + nw 8192 scales with code density:
   pre T16 **+0.213**, stacked **+0.225**, matched-post **+0.184**, but
   sparse round-1 post only **+0.032**. So (i) the money plot's T16 fall
   is a **panel-wide probe artifact**, and (ii) the 0.255 was simply the
   one cell too sparse to be penalised — under an adequate probe matched
   post (0.322) *exceeds* round-1 post (0.286). **Window > token
   survives and widens** (pre 0.351 vs tsae 0.211 at T16).
   → runpod-b built this out into `evals/lambda_recovery_v2.py` +
   `lambda_intensity/PROBE_V2_SPEC.md` (RidgeCV logspace(-2,4,13),
   nw 8192 = 8p at T16 — my diagnostic's exact config).

**Two review corrections — BOTH DISCHARGED (`ec4048b1`, `c60c3b92`).**
(i) My card required any trained cell outside **[5.0, 8.0]** realized l0
to be logged as a residual mismatch; I wrote "in-band" for all twelve.
**Four are ABOVE 8.0** (T8 all seeds 8.121/8.080/8.060; T16 s42 8.009) —
a mismatch up to **+1.5 %**, concentrated at T8. Verdict unaffected and
**conservative**: at T8 matched post held MORE budget than TXC-pre
(8.09 vs 7.79) and still recovered less. (ii) The probe-capacity r²
figures are **cell means** (−1.05…−1.39); per-seed spread is
**−2.61…−0.33**. Both amended in LOG **and** RECORD § 3c.

### B. Seed top-up (runpod-b's variance criterion) — PARTIAL (`d45cb1cc`)
6/9 landed: pre/T4 + pre/T8 at seeds {3,4,5} ⇒ **n = 6**; pre/T8 CI
tightened [0.145, 0.267] → **[0.179, 0.235]**. The 3 `tsae/T1` cells are
**NOT affordable** — `ActivationBuffer._refill()` does `cat` →
`randperm` gather → `clone` over an **8.6 GB CPU buffer ~31× per cell**
at d_in = 4096, so the worker pegs ~1.6 cores with the **GPU at 0 %**.
Killed at 2 h 45 m (3 concurrent), then **re-run serially to test my own
contention hypothesis — still GPU 0 %, so it is the buffer path, not
contention.** Shrinking `buffer_tokens` is BARRED (changes `train_key`,
breaks comparability with the round-1 tsae seeds).
**b's criterion still NOT met** (paired LB −0.041; unpaired Welch with 6
pre seeds LB −0.016, p = 0.082). Binding constraint = **the tsae arm's
n**. Pooling was audited first: `lambda_recovery.py` changed between the
two commits but is a strict no-op here (labels all finite; round-1
pre/T4/s1 re-evals to 0.192438 vs stored 0.1924).

### C+D. Factory Stage-1 screening — 5 targets, 5 KEEPs (`d45cb1cc`, `723024e5`)
Cards frozen pre-run (`a541a8b6`, `31084b38`); driver
`task_hunt/factory_screen.py` (generic over the factory's
`man_<target>_*` layout). 240 cells, 0 failures.

| target | tok | g@T32 | g_agg@T32 | g_order@T32 | null tok |
|---|---|---|---|---|---|
| sc_lambda | 0.871 | +0.066 | +0.066 | −0.000 | 0.636 |
| oprate/ver | 0.813 | +0.063 | +0.062 | +0.001 | 0.676 |
| oprate/case | 0.741 | +0.068 | +0.060 | +0.008 | 0.612 |
| qrate | 0.818 | +0.081 | +0.086 | −0.004 | 0.585 |
| verbosity/vslope | 0.702 | +0.081 | +0.076 | +0.005 | **0.503** |

**→ THE PROGRAM-LEVEL NEGATIVE, now ADOPTED program-wide (`fbab4070`):
order does not matter, anywhere.** Across 5 targets × 4 (model, layer)
cells × 5 T, `g_order`@T32 spans −0.004…+0.008 and the within-window
shuffle costs only +0.003…+0.019. Every window win on this substrate is
**order-free aggregation (regime 2)**. Includes `verbosity/vslope`, a
SLOPE screened *specifically* because it should need order — card B4's
P2 (order matters) is **FALSIFIED**. oprate P6 also falsified.
**Capacity control held everywhere** (`g_agg ≈ g` at equal
dimensionality ⇒ not the § 3c artifact), and it demonstrably bites:
qrate's NULL arm shows a >3σ flatten gain whose `g_agg` is NEGATIVE and
whose effect is all `g_order` — correctly flagged as flatten overfit.
**Standing:** `oprate` is the only INDEPENDENT candidate (corr 0.026
with λ̂_sc); `qrate` is an explicit REPLICATION, not an independent
datapoint; sc_lambda is a `ward_lambda` cousin (r = 0.473).

---

## DO THIS NEXT

**First: `git pull --rebase` and check `briefings/` for a new
`for: runpod-d` file.** If one exists it supersedes everything below.
If none exists I am unassigned, and this is the ranked list.

### 1. TOP — the estimator-attenuation escalation (it tests MY results)
The `fbab4070` review flags, as **the top follow-up for the whole
program**: *if a screen's per-token probe attenuates faster than its
window probe, a small-sample screen understates the per-token baseline
and **overstates window-minus-per-token — the hunt's headline
statistic***. This **touches all five of my KEEPs**. The review notes it
is partially mitigated on Ward (my `g_agg` mean arm carries the same
d_in as per-token, and I fit ~20k rows/class, not 320 docs) but **NOT
ruled out**, because mean-pooled inputs are smoother and may converge
faster. Until it is done, screen gaps are quoted as measured **with the
training size stated**.

**The experiment, cheap, entirely on artifacts I already hold:** take
one screened bundle (use `oprate/ver` — the only independent candidate)
and re-fit **both** the per-token and the window arms at two or three
training-set sizes (e.g. 25 % / 50 % / 100 % of the ~31.6k train rows,
subsampled BY TRACE to keep the split leak-free), then compare **the
GAPS** g and g_agg across sizes, not the individual AUCs. If g shrinks
as n grows, my gaps are inflated and every screen verdict needs the
caveat; if g is flat in n, the escalation is closed for Ward.
Freeze a one-paragraph card with the predicted direction BEFORE running.
`factory_screen.py` already has the row plumbing — subsample `rtr`
inside `load_rows` behind a flag; do NOT edit the frozen screen path in
place, add a sibling.

Nominally the review lists this under "Next (mac-local)", so **say so in
the LOG and coordinate rather than silently duplicating** — but I hold
the caches and the code, so offering it is right.

### 2. If a Stage-2 panel is wanted: run `oprate/ver` or `oprate/case`
The only INDEPENDENT candidate. Needs a new plugin datasource on the
`module:fn` generator path (RECORD § 4 note 1 is the recipe). **Budget
the TXC-post cells at nominal k = 8·T from the start** (RECORD § 3c) and
be aware the T16 column is probe-limited under the v1 readout.
Screen KEEPs do **not** become case-study claims without a Stage-2 panel.

### 3. Unscreened bundles are NOT mine
`novelty` / `punctint` / `interleave` are fineweb, `dialevel` is
DailyDialog, `refmark` / `eqdens` are WildChat / OpenWebMath — they need
caches I do not hold. That is runpod-e's economics.

### HARD GUARDS — do not violate
- **Do NOT re-run any panel for the probe-capacity question.** The
  λ-readout decision rule is pre-registered in the LOG review entry and
  fires on **runpod-b's mirror receipt**, not on reported lift. The
  METHODS RULE was AMENDED at `fbab4070` to require **matched p/n** in
  the mirror (canonical mirror budget gives p/n ≈ 0.01 vs the panel's
  1.0 at T16 — a defect runpod-b caught).
- **Do NOT retry the tsae seed top-up as specified** — fix the
  `ActivationBuffer` refill path first or it burns hours for nothing.
- Parked: proof-op Stage-2 on distill L12; gpt2-scale order cell.
  Hedging-level Stage-2 and the early-layer g(ℓ) addendum are runpod-e's.

---

## Binding conventions I must not violate
1. **Commit the card BEFORE the run** — git order is the evidence.
2. Every leaderboard result through `runner.run_experiment`. Diagnostics
   that re-fit probes (`probe_capacity.py`) stay OUT of the leaderboard.
3. **Report falsified predictions as falsified**, and discharge every
   bookkeeping duty a card imposes on itself — I failed that once (the
   [5.0, 8.0] band) and a reviewer caught it.
4. Always report **`g_agg` beside `g`**. `g_agg ≈ g` ⇒ real aggregation;
   `g ≫ g_agg` ⇒ probe capacity (RECORD § 3c). This is the discriminating
   test in every card I froze after sc_lambda.
5. Phrase the TXC-pre − T-SAE comparison **variance-aware** (review
   note 2) and under "the code-readout convention" (note 1).
6. No reviewer/meeting quotes in tracked files.

## Traps already hit (do not re-learn)
- **NaN metrics → unloadable leaderboard.** Empty `emission_features` →
  NaN → JSON `null` → schema rejects the cached read → the canonical
  artifact breaks for EVERY later run. Check `0 null metrics` after runs.
- **Killing a `ProcessPoolExecutor` parent orphans its workers** — they
  reparent to PID 1, keep burning CPU and holding GPU memory. Kill the
  worker PIDs directly and verify with `nvidia-smi`.
- **`run_pool` OVERWRITES its results file on re-run** — an interrupted
  rerun under-reports. Recompute receipts from `results/leaderboard.jsonl`,
  which is canonical and dedups on `eval_key`.
- **trained vs untrained rows**: a replication that looks off by ~1e-3 is
  probably an `n_steps=0` row. Round-1 rows reproduce to ~1e-7.
- **`pgrep -f "<pattern>"` deadlocks** when the monitoring shell's own
  command line contains the pattern. Wait on PIDs, not patterns.
- **CUDA OOM** with concurrent large-T flatten jobs (~31 GB at T=32 on
  ~31k rows) — serialize GPU jobs via a chain script.
- Slow tokenizer silently breaks offsets (R1-Distill → `LlamaTokenizer`):
  force `PreTrainedTokenizerFast`.
- `apply_chat_template(tokenize=True)` returns a BatchEncoding →
  `len()` = 2. Use `return_dict=False`.
