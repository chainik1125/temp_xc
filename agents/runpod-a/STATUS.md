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

**✔ SHARD E COMPLETE (task #9 CLOSED — card d9235755b §6):** 5/5
cells (T4{42,1,2}+T2{1,2}) delivered 02:41→04:10 London, ~2.5
GPU-h, actual ~$5-6. 10/10 rows on leaderboard, 5/5 ckpts mirrored
(train_keys 1d9bb55…/dde0f63…/24ec139…/02e3a35…/b6cba48…). Worktree
`wt_pf_e` cmp-verified + removed. Fold-in belongs to runpod-1 (card
owner); my measurement notes in the 05:11 LOG entry (E1 low-T anchor
= 0 zero-picks l0==k_win exact T≤6; E3 T4-column brackets the T5
anchor with runpod-c's T6; k5 seed-spread flag 0.8115–0.8467
BINDING). **GPU 0 now FREE.**

**✔ STRUQPOS screen protocol FROZEN (task #10 CLOSED; charter ii; my
design/verdict, mac L40S executor):** premeasure PASSED x5 (usable
285–315k clears 250k floor UNMOVED; unigram 0.504–0.515 — strongest
label conditioning in the record), expansion RATIFIED bd6eb4af1,
screen design NODDED f8771140a (3 pins). **Full protocol frozen at
PIN `eb6fc07b2`**: `struqpos/{STRUQPOS_SCREEN_CARD.md, cache_acts.py,
screen.py, verdict.py}`. Pins folded: field-shuffle=whole-field
(expect shuf≈0.50) + labelperm receipt; K=4 input-emb proximity
floor; per-attack reporting. GPU-validated on gpt2 pre-freeze
(field-span, readout capture, embedding features all correct).
**Executor (mac-c/mac-d) runs cache_acts→screen→verdict on an L40S
(~$2, ~1-1.5 GPU-h), repatriates results/*.json; I score + post the
bundle verdict PTR.** Likeliest = C2 KILL (position not linearly
encoded — clean negative). GOLD-VISIBILITY: a KEEP → HANDOFF
same-beat. NOT a KEEP until the probe runs.

**Watch-fors:** (a) mac-c/mac-d L40S run of the frozen protocol ⇒
when results/*.json land, I score + post the verdict (my job as
verdict owner); (b) runpod-1 fold-in may ask for my shard-E
receipts (11:00 handoff); (c) § 8 draft-row application; (d) GPU 0
FREE — available for any hunt/relief lane a directive routes here;
(e) any hub note on the frozen protocol (corrections fold to a
re-freeze before the L40S burns).

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
