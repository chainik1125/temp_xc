# runpod-a STATUS — live (rewritten ~16:40 London 2026-07-27)

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

**⚑ PAPER-FAITHFUL SPRINT SHARD (606e4587d item 3 — my next GPU
lane; state re-read 02:38 London 07-28):** split now lands **IN
runpod-1's card §shards** (their ack d39fc5075 — they own it; 21
cells flagged vs commissioned 18, grid {1,2,4,6,8,10,16}×{42,1,2},
T1 cells ordered LAST per shard so one hub line prunes 3). Card
≤05:00 London on track. **4 GPUs at card-pin** (11227ce0d
freeze-and-join): runpod-1 old-pod GPU 0 + my GPU 0 + runpod-c's
2×H100 (substrate on-pod from bring-up); runpod-b GPU 1 joins at
rmx_b drain (~11:30, or ~06:15 if their T8-boundary stop offer
b5651d5f2 is taken). I coordinate shard ids with runpod-c here.

**Substrate: WARM — COLD caveat RETRACTED.** The paper's v1 anchor
mirror sits at `/workspace/caches/probing/hf_mirror/` (33G;
runpod-b wired it ~23:04 for width-match; my 17:29 symlinks in
main checkout: `results/probe_cache/gemma_2_2b_it_l13_fineweb_24k128`
+ `results/data_cache/48d2d17ff88598d4/acts.npy`). Preflight PASS
02:38: acts mmap (24000,128,2304) fp16; meta data_key
48d2d17ff88598d4 (gemma-2-2b-it L13 resid_post, fineweb-edu); 38
probe tasks. **At launch: recreate the two symlinks INSIDE the
worktree** (repo_root anchors results/ to the worktree root),
worktree-detach at card pin, AGENT_NAME=runpod-a, GPU 0.

**Watch-fors:** (a) runpod-1's plugin card freeze ⇒ claim shard by
LOG line + launch (shard ids w/ runpod-c); (b) mac-local's dq
caption-fork ruling — if "measure the relu-mix column", that lane
also lands here; (c) § 8 draft-row application; (d) 11:00 BST
handoff support asks (my T4 rows + twin receipts feed items 3/4/5).

## House-rule cache

Pull-rebase before every push; BOTH LOG blocks on conflict; stray
grep baseline = 1 (the rule quoting itself); stamp from `date`
(BST=UTC+1) and VERIFY against commit time (two corrigenda tonight
— stamp-drift is the house failure mode); PTR everything; NEVER
stash around a live-writing runner (commit rows checkpoints); one
run_pool per process (futex-wedge lesson); worktree-detach at pin
for every GPU lane, cmp-verify harvests before --force removal;
token PATHS only, no Modal creds on pods.

*Rewrite before any compact.*
