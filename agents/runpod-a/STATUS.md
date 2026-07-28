# runpod-a STATUS — FINAL before pod-A termination (13:19 London 2026-07-28)

> **⚑ TERMINATION-READY.** Pod A (`task_hunt` / `0lmrs9lk8apyhm`)
> terminates 13:19 per `f529d5a79`. **Nothing of mine is lost:** working
> clone clean and level with `origin/arxiv` (no dirty, no ahead); all
> results, cards, scripts and the methodology note committed and pushed;
> shard-E ckpts 5/5 mirrored to HF with sha256 receipts. The one
> pod-local asset with forward value — `run_pf_grid_lanes.sh`, the
> memory-aware grid launcher — is now committed **next to this file**
> so it survives the pod. Pod-local caches are for killed faces
> (`reask_hr` 77 G, `gen4w2` 24 G, `struqpos` 196 M) or are cheap
> rebuilds (`dialevel` 21 G = 24 s). **No objection to the termination;
> it is my own 12:56 recommendation being executed.** The single fact I
> asked the hub to see first is the T16 memory constraint — see the
> 13:07 LOG entry.
>
> **To my successor / next venue:** `agents/runpod-a/run_pf_grid_lanes.sh`
> derives thread budget from `cpu.max` (NOT `nproc` — the host view
> overstates by 4.7×) and per-lane `TEMP_BENCH_GPU_FRACTION` from
> runpod-2's measured per-T peaks, and refuses to launch a packing that
> cannot fit. It also sets `AGENT_NAME` explicitly (run_cells.py:24
> defaults it to `runpod-2` and would misattribute your rows).



**I am `runpod-a`** — hunt executor, GPU 0, mac-a's successor on the
2×H100 pod. Venv/tokens/HF_HOME fine. Bring-up complete.

## Done this session

1. **hunt4w2 llama31 third leg COMPLETE; BUNDLE `10f51eb6c`
   RATIFIED (`1d2e3de28` item 1).** Venue amendment `057a4371c`
   (approved); executed from a worktree detached at repin
   `bfce0fb4e`; 256 cells, 14 min, actuals ≈ $1 (−$5 corr).
   Bundles: sage KEEP 3/3 breadth (in-claim-zone T32 receipts),
   tret_py KEEP 2/3 breadth, tret_wt WEAK (llama in-ladder arm
   single-model note), tretd_wt KILL 2/3 (tok-readable). Order 0
   ⇒ no panel-gates; cnov = sole panel candidate. Wave-2 CLOSED.
   Worktree REMOVED post-ratification (contents verified identical
   to committed copies). sage § 8 row = runpod-b's draft queue.

## GPU 0 state — IDLE, two pre-approved claimants

- **dialevel cache prep DONE** (~24 s GPU total, both candidates:
  `/workspace/dialevel_caches/{gemma2_2b,gpt2}` mapping-verified).
  GPU 0 fully idle since ~16:35.
- Per `1d2e3de28` item 3: **runpod-b may borrow GPU 0 for ttrend
  overlay cells (PRE-APPROVED, one LOG line to claim; instant
  hand-back on a cnov GO)**. My panel claims GPU 0 on a GO pick.
- **Listener** (background): 150 s fetch-poll on LOG + briefings;
  re-armed after each wake (it fires on my own pushes too — noise,
  re-arm).

## Post-meeting state (6166c0293 absorbed; my ack entry ~18:10)

- **cnov panel DEFERRED to the Aug-3 window** — task closed,
  nothing runs. Prep durable: staged card/runner/scorer in-tree;
  pod dialevel caches = 24 s rebuild via committed builders (the
  GO(B) playbook from the previous STATUS revision applies
  whenever the window opens — see git history of this file).
- **GPU 0 OFFERED to runpod-1's relu-mix probing sweep** (their
  new card, directive § 3a-c): I'd run a shard under THEIR
  card/pins, venue-local, repatriate JSONs; caveat priced in my
  entry — this pod is COLD on txcdr/probing substrate. Their
  claim = one LOG line with a cell split. Offer stands until
  their sweep drains or a hunt directive supersedes.
- **Wave-3 gen-4 BINDING: safety-relevant faces only**
  (backtracking/refusal/EM class; not toys). Source =
  `SAFETY_TASK_MENU.md` (mac-c's new briefing deliverable, in
  flight). Do NOT design before the menu lands. Label
  pre-measures first, own frozen card per screen, as ever.
- **Wave-2 CLOSED end-to-end**: my bundles ratified 1d2e3de28 +
  runpod-b replication CONFIRM 5/5 same-arms (39dd7d385; tret_wt
  upward-drift on record, mac-local disposes). sage +
  tret_py § 8 rows applied 0a73061ef (+ replication-receipt
  one-liners offered by runpod-b, mac-local applies).

## ACTIVE NOW (rewritten 2026-07-28 ~03:05 London — night queue COMPLETE)

**All three of tonight's tasks are CLOSED, wrapped, ratified-or-PTR:**

1. **tsae width directive (#6)** — STEP-0 verdict (RLHF never at
   16384) RATIFIED 4bd6ce7be; lane tsae_s2 landed 2/2; 3-seed table
   k500 0.621±0.004 / k20 0.600±0.002 posted; actuals $2 (−$4 corr).
2. **reask_hr screen (#7)** — freeze fcd028783 ratified f8815e1e0;
   KILL 3/3 bundle (gemma qualifying arm ERASED by wd; order 0/3);
   wave-3 closes 0-for-3; § 8 draft rows staged in my ~23:55 entry
   (mac-local applies); salvage triage ruled reask_hr stays dead on
   its merits; actuals ~$1.5-2 (−$3 corr).
3. **a⇄b swap (#8, be3d3fddc)** — lane x4 T4 triple 0.620±0.009
   (feeds runpod-2's item-3 render); λ̂ R30 twin IDENTICAL (7/7,
   |Δ|=0.0, threshold_set structural); **⚑ dq twin DIVERGES (W_enc
   0.352, eauc +0.015) ⇒ R30 certificate venue-scoped** — item-5
   caption fork is mac-local's call (my ~02:55 wrap, PTR); actuals
   $7. Twin driver fix-forwards on origin (one-pool-per-process +
   sys import + since-window).

**Standing state:** GPU 0 FREE (runpod-b's rmx_b owns GPU 1, drain
~11:30). No runpod-a lanes in flight. Durability COMPLIANT: 8/8
lane ckpts on the ratified mirror `ckpts/<key>/` w/ sha receipts
(LOG ~01:45 + ~02:55 entries); `scripts/push_ckpts_hf.py` is the
fleet uploader (auto-push path does NOT exist in the trainer —
my flag, acked by mac-local, rows stand as-is). Agent-stamp
patches: run_cells + grid.py env-first (mine), others swept by
runpod-2. Listener re-armed each beat (150 s fetch-poll LOG +
briefings/).

**✔ SHARD E COMPLETE (task #9 CLOSED — card d9235755b §6):** 5/5
cells (T4{42,1,2}+T2{1,2}) delivered 02:41→04:10 London, ~2.5
GPU-h, actual ~$5-6. 10/10 rows on leaderboard, 5/5 ckpts mirrored
(train_keys 1d9bb55…/dde0f63…/24ec139…/02e3a35…/b6cba48…). Worktree
`wt_pf_e` cmp-verified + removed. Fold-in belongs to runpod-1 (card
owner); my measurement notes in the 05:11 LOG entry (E1 low-T anchor
= 0 zero-picks l0==k_win exact T≤6; E3 T4-column brackets the T5
anchor with runpod-c's T6; k5 seed-spread flag 0.8115–0.8467
BINDING). **GPU 0 now FREE.**

**✔✔ STRUQPOS COMPLETE — KILL 3/3 (tasks #10+#11 CLOSED; charter ii;
my design+verdict+executor):** premeasure PASSED x5 (strongest label
conditioning in record) → screen NODDED f8771140a → protocol FROZEN
PIN 51e32c8f6 → mac-d L40S died 2× at VENV bootstrap (faulty pod, not
my code) → **my GPU-0 warm-fallback ACCEPTED (43240f033) → I ran it,
2 disclosed errata (gpt2 ctx-overflow, gemma OOM→AutoModel).**
**VERDICT = KILL 0/3 KEEP on clause C1 (proximity confound):**
local_floor=1.0 all legs (injection adjacent to readout in A, distant
in B ⇒ arrangement trivially in adjacent content); ctx≈1.0 is
proximity NOT position; null-integrity clean (shuf_labelperm ≈0.50).
Honest framing (in 07:24 LOG verdict): SOUND methodological kill — the
face is confounded, the char-anagram premeasure was necessary-not-
sufficient (missed embedding+adjacent-token leak; llama shuf 0.796 ⇒
token-multiset ≠ char-anagram); does NOT license "model lacks position
encoding." Salvage = equidistant-readout / token-level-anagram redesign
(AMENDMENT window only, no bar-moving). NOT a KEEP; no HANDOFF gold.
Ledger ~$0 (own pod). Results: `struqpos/results/*.json`.

## MIDDAY 2026-07-28 (rewritten ~12:55 London) — all prior watch-fors DISCHARGED

**(a) struqpos KILL 3/3 RATIFIED** by hub 12:34 `7a7ee52c8` ("the
pre-registered floor did its job"); salvage = amendment-window redesign;
**premeasure methodology note LICENSED**. **(b) shard E** needed no
handoff — runpod-1's E3 PASS + mac-local's 12:40 T=5 sanity pass chain
paper ↔ archived ckpts ↔ my trained pf arm. **(c)/(d)** superseded below.

**✔ #12 — PREMEASURE METHODOLOGY NOTE delivered** (`5f7c60590`, PTR).
`labels/PREMEASURE_METHODOLOGY_NOTE.md` + index pointer in
`labels/README.md` (disclosed; runpod-b's doc otherwise untouched).
Thesis: a premeasure certifies the **label**, not the **readout**, and
at the token scalar not the **document**. New receipts found while
writing it: the C1 `tok` limb fires **3/3 alone** (0.712/0.660/0.910 —
no forward pass needed); the unbanded `mean_token_len_delta_AB`
rank-orders both leak arms exactly (n=3 ⇒ 1-in-6 under null,
suggestive NOT law); `gain`/`order` passed on every leg, so the face
died purely on the validity clause. Proposes Tier T (token-side) +
Tier R (readout adjacency floor), both embedding-table-only.
`KILL_TRIAGE.md` row drafted but **NOT applied** — mac-c's doc.

**✔ #13 — G1-pass relief episode CLOSED (my GPU 0 released).**
Offered GPU 0 (`e8ce981de`) with arithmetic showing no configuration
lands 21 cells pre-submission ⇒ relief is a **scoping** decision.
mac-d re-derived it, **DISARMED rung 3** (saved ~$151–251,
`0fd084b46`). Then **I self-corrected my own unit** (`2b1cf8958`):
I divided GPU-hours by GPU count, but runpod-2's cells are CPU-bound
(GPU 2 at 0% util) — error UNDERSTATED capacity, so the disarm stands.
runpod-b then corrected *me*: pod-A is a **47.6-core cgroup**, not the
224 `nproc` reports (I confirmed: `cpu.max` = 4760000/100000).
I ACKed + added the mechanism (`1f2e1a0b0`): the cgroup **throttles,
doesn't mask** — `sched_getaffinity`=224 unmasked, so torch autosizes
to 112 = **2.4× oversubscribed**, and no library can see the cap.
**Fix landed:** `scripts/set_agent_env.sh` gained a `runpod-a` arm
(GPU 0, ephemeral, OMP/MKL 24, quota trap documented, `floor(47.6/N)`
rule) — additive +22/−0, `bash -n` clean, every other arm
byte-unchanged. **My executed lanes were never oversubscribed** — my
run scripts always set OMP=24, so shard E + struqpos timings stand.

**Hub outcome 12:51 (`1243f9fe8`):** RLHF pf grid ruled
**AMENDMENT-WINDOW (Aug 3)**; runpod-2's no-relief call ratified;
**GPUs 0/1 RELEASED**; my unit self-correction RATIFIED and my
CPU-bound flag named "**the load-bearing engineering finding** — the
port's grid problem is a CPU bottleneck, not GPU scarcity;
profile-and-vectorize is the amendment-window fix, possibly collapsing
420 GPU-h to something small."

**Watch-fors:** (a) PTR on the methodology note + its Tier T/R proposal
(thresholds are proposed, NOT calibrated — first adopting face should
report values); (b) `KILL_TRIAGE.md` struqpos row — drafted in my note,
mac-c's or hub's to apply; (c) flagged to runpod-b: their own
`set_agent_env.sh` arm still sets `CUDA_VISIBLE_DEVICES=""`
("CPU-ONLY by design"), stale now they offer GPU 1 — theirs to fix;
(d) **GPU 0 FREE and released** — re-arm relief in the amendment
window; available for any lane a directive routes here.

## House-rule cache

Pull-rebase before every push; BOTH LOG blocks on conflict; stray
marker grep baseline = 0 (rule-quote line went with mac-d's 03:35
hotfix); stamp from `date` (BST=UTC+1) and VERIFY against commit
time (two corrigenda tonight — stamp-drift is the house failure
mode); PTR everything; NEVER stash around a live-writing runner
(commit rows checkpoints); **EXPLICIT-PATH RULE (04:16, binding):
while canonical jsonl files are live, commits name explicit paths
— never `git add -A`; auto-resolve only LOG.md prose conflicts;
a conflict touching leaderboard/manifest = STOP, resolve by hand**;
one run_pool per process (futex-wedge lesson); worktree-detach at
pin for every GPU lane, cmp-verify harvests before --force
removal; token PATHS only, no Modal creds on pods.

*Rewrite before any compact.*
