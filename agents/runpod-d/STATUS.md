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

## STATE (rewrite 2026-07-24 ~22:55 UTC)

**Primary round-2 deliverable: DONE + PUSHED** (`2b64dbe4`), gate met.
**Seed top-up: PARTIAL + PUSHED** (`d45cb1cc`).
**Factory screening (briefing § 3): sc_lambda DONE (KEEP-qualified);
oprate/qrate/verbosity screens IN FLIGHT** (cards frozen `31084b38`).

### A. Matched TXC-post + probe capacity (§ 1–2) — CLOSED
1. **Reading (b) CONFIRMED**: round-1 post 0.255 @ T16 was budget-confounded.
   Falsifier passed exactly (untrained realize l0 = 8.000 at every T).
   Matched post peaks T4 then falls to 0.137 → **TXC-pre stays headline**.
2. **Reading (c) CONFIRMED, bigger than pre-registered**: the λ probe
   (unregularized OLS, n = p at T16) is capacity-limited for DENSE codes.
   Lift under ridge + nw 8192 scales with density (pre T16 **+0.213**,
   sparse round-1 post only **+0.032**) ⇒ the T16 fall is a **panel-wide
   probe artifact**, and 0.255 was just the one cell too sparse to be
   penalized. Window > token survives and widens (0.351 vs 0.211).
   → Surfaced as a METHODS decision, deliberately NOT taken. **runpod-b
   built it out**: `evals/lambda_recovery_v2.py` + `PROBE_V2_SPEC.md`
   (RidgeCV logspace(-2,4,13), nw 8192 = 8p at T16 — my diagnostic's exact
   config), Han-approved `6394eba3` with the decision still untaken.

### B. Seed top-up — PARTIAL, do NOT retry as specified
6/9 landed (pre/T4, pre/T8 at seeds 3,4,5 → **n = 6**; pre/T8 CI tightens
to [0.179, 0.235]). The 3 `tsae/T1` cells are **NOT affordable**:
`ActivationBuffer._refill()` does `cat` → `randperm` gather → `clone` over
an **8.6 GB CPU buffer ~31× per cell** at d_in = 4096, so the worker pegs
~1.6 cores with the **GPU at 0 %**. Killed at 2 h 45 m (3 concurrent),
re-run **serially** to test contention — still GPU 0 %, so it is the
**buffer path, not contention**. Shrinking `buffer_tokens` is BARRED (it
changes `train_key` and breaks comparability with round-1 tsae seeds).
b's criterion **still NOT met** (paired LB −0.041; unpaired Welch with 6
pre seeds LB −0.016, p = 0.082). Binding constraint = **the tsae arm's n**.
→ If anyone retries: the 9 cells are NOT equal cost (pre ≈ 5 min, tsae =
multi-hour); fix the buffer path first.

### C. Factory screening — sc_lambda KEEP (qualified)
48 cells, σ_null 0.0066. g clears 3σ at T8 in all four (model, layer)
cells and grows monotonically to **+0.059…+0.071 at T32** (~10σ).
**Decisive control: the window-MEAN arm has the SAME 4096 dims as
per-token yet g_agg ≈ g** ⇒ the gain is NOT probe capacity. Shuffle-immune
(g_order ≈ 0); the label-null arm shows NO window gap. **Heavy
qualification: per-token is 0.87 (largely converted), and it is a
`ward_lambda` cousin (r = 0.473), not an independent case study.** P1
falsified; P5 disclosed as non-discriminating as frozen (per-token alone
already beat the visible-evidence line — the equal-dimension g_agg test in
the later cards is the fixed version).

## DO THIS NEXT

### 1. Harvest the four in-flight factory screens (chain3)
`oprate/ver`, `oprate/case`, `verbosity/vslope`, `qrate` — cards frozen at
`31084b38` BEFORE any cell ran; driver `task_hunt/factory_screen.py`
(generic over the factory's `man_<target>_*` layout; ~20 min per target;
idempotent per cell so partial runs resume). Logs
`…/scratchpad/screen_<bundle>_<target>.log`; results
`task_hunt/<bundle>/results/<bundle>_<target>_screen.json`.
Write **one LOG paragraph per target**, KEEP/KILL against that card's own
kill rules, and score every frozen prediction — including the ones I
expect to fail:
- **oprate is the valuable one**: independent of sc_lambda (corr 0.026)
  AND of its own second target (−0.032), so a win here is a genuinely
  separate datapoint. It also carries the highest visible-evidence bar
  (ver T32 = 0.830).
- **qrate is an explicit REPLICATION** of the sc_lambda family — a KEEP
  adds confidence, it does NOT count as an independent candidate.
- **verbosity is the structurally anti-ambient one** (a SLOPE;
  visible-evidence BELOW chance and falling with T). It is the only card
  where I predicted **order to matter** (P2: g_order > 0.02, shuffle
  costs). If it comes back order-free like the rest, that premise is
  FALSIFIED and must be written as such, not folded into a KEEP.

### 2. Analysis convention to keep applying
Always report **g_agg beside g**. If g_agg ≈ g the window gain is real
aggregation; if g ≫ g_agg it is probe capacity (RECORD § 3c). This is now
the discriminating test in every card I froze after sc_lambda.

### Parked (do NOT run)
Proof-op Stage-2 on distill L12; gpt2-scale order cell. Hedging-level
Stage-2 and the early-layer g(ℓ) addendum are **runpod-e's**.

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
