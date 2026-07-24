# Working state — agent `runpod-d`

**Last rewrite:** 2026-07-24 ~19:45 UTC, after the round-2 primary
deliverable was **finished and pushed**. Read § "DO THIS NEXT" first.

## Who / where / setup (all already built — reuse, do not rebuild)
GPU RunPod pod (H100 80 GB, 224 cores, 2 TB RAM), `/workspace/temp_xc`,
`/workspace/.agent_id` = `runpod-d`. **I am runpod-d, NOT runpod-e.**
Round-2 briefings are split per pod: mine is
`briefings/task-hunt-r2-d.md`; `…-r2-e.md` is runpod-e's and is not my
work. (The combined `task-hunt-r2.md` was split by mac-local at
`766d6142` — if a stale reference to it appears, that is why.)

- `.venv` = probe/training venv (torch 2.8+cu128). **No `pip` in it** —
  do not try to install packages there (py-spy etc. are unavailable).
  `/workspace/vllm_venv` = separate vLLM 0.25.1 venv (has pandas+ninja).
- `HF_HOME=/workspace/hf_cache`; creds `/workspace/.tokens/`.
  Git: identity set, `core.askPass=/workspace/.tokens/git-askpass.sh`.
  **Branch `arxiv` is shared by 5 agents — `git pull --rebase` before
  every push.** `results/leaderboard.jsonl` + `checkpoints/manifest.jsonl`
  are usually dirty from live runs; `git stash -u` → pull → `stash pop`
  is the working idiom. Both have a **`merge=union`** driver
  (`.gitattributes`) — jsonl conflicts auto-keep both sides.
- Caches on the volume: `/workspace/conv_depth_caches/{ward_stream,base,
  distill}`, `/workspace/task_hunt_labels/lambda_intensity/` (incl. the
  DENSE Stage-2 grids), `/workspace/task_hunt_labels/forbidden_word/`
  (rollouts + acts + 167 GB acts_depth).

---

## STATE: primary round-2 deliverable is DONE and PUSHED

`briefings/task-hunt-r2-d.md` § 1–2 are complete, pushed through
`2b64dbe4`. Acceptance gate: card frozen pre-run ✓, LOG verdict ✓,
re-rendered figure ✓, RECORD addendum (§ 3c) ✓, leaderboard hygiene
(0 dup eval_keys, 0 null metrics, 108 ward rows) ✓, 230 tests pass ✓,
STATUS rewritten ✓. **Briefing stays until mac-local review.**

### What was found (two results, both in LOG + RECORD § 3c)

1. **Reading (b) CONFIRMED — round-1's TXC-post 0.255 @ T16 is not a
   matched win.** Card `lambda_intensity/card_stage2_postmatched.md`
   (frozen `07c90cfb` before any matched cell existed) raised post's
   nominal k to **8·T** (16/32/64/128). 24 cells, 0 failures.
   **Falsifier passed:** every untrained matched cell realizes
   `l0_per_token` = 8.000 exactly at every T → the `l0 = k/T` mechanism
   is measured, not assumed. Matched post reads 0.185/0.202/0.144/0.137
   at T=2/4/8/16 — peaks at T4, falls into the TXC-pre (0.138) /
   Stacked (0.094) band. **TXC-pre (peak T8 = 0.206) remains the
   matched-budget headline.**
2. **Reading (c) CONFIRMED, and bigger than pre-registered — the λ
   probe is capacity-limited for DENSE codes.** `lambda_recovery` fits
   unregularized OLS on p = 2048 features with n = 1024·(32/T) rows →
   **n = p at T16**. Re-fitting the SAME checkpoints with ridge +
   n_windows 1024→8192 (`probe_capacity.py`, pre-registered in the card
   before results existed, OUT of the leaderboard) lifts held-out r
   **monotonically in code density**: pre T16 **+0.213**, stacked T16
   **+0.225**, matched-post T16 **+0.184**, but sparse round-1 post T16
   only **+0.032**. Consequences: (i) the money plot's T16 fall is a
   **panel-wide probe artifact** that closes under an adequate probe
   (pre 0.206→0.351 from T8→T16); (ii) the 0.255 was the one cell too
   sparse to be probe-suppressed — under an adequate probe matched post
   T16 (0.322) **exceeds** round-1 post T16 (0.286), so the sparse code
   had no representational advantage, it dodged the artifact.
   **Window > token SURVIVES and WIDENS** under the adequate probe
   (pre 0.351 vs tsae 0.211 at T16).

**METHODS decision surfaced, deliberately NOT taken:** should the
canonical λ-readout adopt a capacity-adequate probe (ridge, n ≫ p)?
runpod-b's variance receipts are all computed on the OLS-probe numbers,
so changing the probe re-bases them. Logged with its receipt for
orchestrator / runpod-b; the leaderboard and § 3b numbers are unchanged
and nothing was re-run unilaterally.

### Renderer reconciliation (done — do not redo)
runpod-b's variance-aware renderer merged upstream mid-run and
**supersedes** the minimal annotation I had written (my render commit
was dropped as empty during the rebase — intended). Resolved by taking
b's renderer wholesale and grafting only what is mine: the matched-post
file loads as a separate arch `txc_batchtopk_post_matched`, and the
`k_pos` reference in `build_summary` **excludes `*_matched` cells** so
their per-window nominal k (up to 128) cannot inflate b's
`>= k_pos/2` budget-matched threshold. Round-1 post stays flagged NOT
budget-matched; matched post reads as matched and appears in both figs.

---

## DO THIS NEXT

### A. Seed top-up — **IN FLIGHT at last rewrite** (finish + verdict)
runpod-b's LOG recommendation addressed to me: bound the paired
**pre-vs-tsae T8** margin (NOT significant at n=3: 0.052 ± 0.055, t CI
[−0.086, 0.190]). b's frozen criterion: one-sided 95% t lower bound > 0,
plus sign-flip attainability (needs n ≥ 5) ⇒ **seeds {3,4,5} ×
{pre/T4, pre/T8, tsae/T1} = 9 trained cells**. My briefing § 1
pre-authorized it. Runner frozen commit-then-run at `3d954869`
(`run_stage2_seedtopup.py`) — the exact 9 cells are in code, so this is
a power top-up, **not "seeds until significant"**.

**Status at rewrite:** 6/9 done (all pre/T4 + pre/T8 — λ 0.187…0.269),
the 3 `tsae/T1` cells still training. PID 49095, log
`…/scratchpad/seedtopup.log`.

> **Timing gotcha (measured, do not re-diagnose):** `tsae/T1` is the
> SLOWEST cell in the panel. Token archs use the `ActivationBuffer`
> (buffer_tokens 524288 → ~31 CPU-side refills per cell); 3 concurrent
> tsae cells sit at ~165 % CPU each with GPU only ~10 %. Round-1
> evidence: a single tsae cell took ~868 s; three concurrent ones ran
> **> 40 min**. That is normal, NOT a hang. Verify liveness with
> `ps --ppid <pid>` + `/proc/<pid>/stat` CPU-time deltas, not by
> watching the log (workers do not stream `[train]` lines).

When it lands: `run_stage2_seedtopup.py` auto-merges into
`results/stage2_ward_real_lambda_base_l12.json` by cell id (idempotent),
then **re-run `…lambda_intensity.render_stage2`** (b's renderer
recomputes per-cell CIs at the new n; pre/T4, pre/T8, tsae/T1 become
n = 6, everything else stays n = 3). Then compute b's criterion with
`support_stats/stats_lib.py` (`t_ci95` is two-sided — take the
one-sided 95 % LB yourself; `sign_flip_p` for the exact test) and write
ONE LOG paragraph: bounded or not bounded, either way. **A null result
here is a result** — report it plainly, do not add seeds to chase it.

### B. Then: § 3 of my briefing — batch-screen candidate-factory bundles
QUANTITY MODE (Han directive). Bundles have LANDED (built 18:43) in
`experiments/explorations/task_hunt/labels/`. **Ward-grid ones I can
screen on my existing caches:** `sc_lambda.npz` (top prior — the
winner's family on a frozen self-correction marker stream), `oprate`,
`qrate`, `verbosity`. (`novelty`/`punctint`/`interleave` are fineweb and
`dialevel` is DailyDialog — those need caches I do **not** hold; they
are runpod-e's economics, not mine.) Each has a `CARD_DRAFT.md` in its
own dir but **no screen script** — adapt
`lambda_intensity/screen.py` (frozen `problib` stack: per-token /
flatten / window-mean / within-window-shuffle + permutation null).
Per candidate: freeze the card (sharpen the draft) → screen → ONE LOG
verdict paragraph, KEEP/KILL, fail fast.
**Apply the two new binding conventions:** (1) **per-token-first
triage** — run the per-token probe ALONE first; a high per-token ceiling
means presumptively converted → KILL cheaply without the window grid;
(2) the **depth sweep** as the WHY-diagnostic when per-token is high.

### Parked (do NOT run)
Proof-op Stage-2 on distill L12; gpt2-scale order cell. Hedging-level
Stage-2 and the early-layer g(ℓ) addendum are **runpod-e's**, not mine.

---

## Binding conventions I must not violate
1. **Commit the card BEFORE the run** — git order is the evidence.
2. Every result through `temp_bench.core.runner.run_experiment`; never
   a bespoke leaderboard append. Diagnostics that re-fit probes
   (`probe_capacity.py`) stay OUT of the leaderboard by construction.
3. **Report falsified predictions as falsified.** Round 1 falsified my
   own bag-of-words story via the depth sweep; round 2 falsified my
   pre-registered reading (a). That is the job.
4. Phrase the TXC-pre − T-SAE comparison **variance-aware** (review
   note 2) and always under "the code-readout convention" (note 1).
5. Realized-l0 annotation on any externally-used Stage-2 figure
   (note 3) — now satisfied by b's renderer + my graft.
6. No reviewer/meeting quotes in tracked files.

## Traps already hit (do not re-learn)
- **NaN metrics → an unloadable leaderboard.** Empty `emission_features`
  → NaN → JSON `null` → schema rejects the cached read → the canonical
  artifact breaks for EVERY later run. `real_lambda.py` now ships a
  documented reference basis. Check `0 null metrics` after every run.
- **Slow tokenizer silently breaks offsets** (R1-Distill resolves to
  `LlamaTokenizer`) — force `PreTrainedTokenizerFast`.
- **`pgrep -f "<pattern>"` deadlocks** when the monitoring shell's own
  command line contains the pattern. Wait on PIDs, not patterns.
- **CUDA OOM** with 3 concurrent GPU jobs (T=64 flatten needs ~28 GB) —
  serialize.
- `apply_chat_template(tokenize=True)` returns a BatchEncoding →
  `len()` = 2. Use `return_dict=False`.
- **Round-1 rows reproduce exactly** (stage2 JSON to ~1e-7) — if a
  replication looks off by ~1e-3, check you are not comparing a
  **trained** cell against the **untrained** (n_steps=0) row. I made
  exactly that mistake and briefly suspected a stale panel.
