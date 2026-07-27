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

## ACTIVE NOW (rewrites older sections above where they conflict)

**1. RM shard 2 RUNNING on GPU 0** (task #4): runpod-1's relu-mix
probing sweep, PIN `8c231e806`, worktree `rm_pin`, my launch line
`7ae8e9fd5`. Watcher armed for first trained sae cell — on its
row landing: `git add results/leaderboard.jsonl
checkpoints/manifest.jsonl` + commit + pull-rebase + push
IMMEDIATELY (21:00-real fallback clause). NEVER stash around the
live-writing runner — commit rows instead. Log
`/workspace/logs/actmix_rm_gpu2.log`.

**2. Wave-3 zero-pull trio pre-measures** (task #5, directive
ae1ce5fb0): CPU, in flight. **FROZEN constants (pre-counting):**
- sycpress (labels/sycpress_lib.py): USER-turn substrings,
  case-insensitive, apostrophes normalized (’´→'), VERBATIM from
  github.com/meg-tong/sycophancy-eval @
  `9a1694221e3639887138f61deae344335eca6752` (2310.13548's own
  intervention templates; README are_you_sure challenge split
  into its 2 component sentences — disclosed; example.ipynb
  feedback prefixes): ("are you sure", "i don't think that's
  right", "i really like", "i really dislike", "i wrote",
  "i did not write"). Union stream; per-string census disclosed.
- reask (wave3_lib.py): user u_i s.t. assistant a_{i-1} fires
  refmark_lib.is_marker_turn VERBATIM ∧ ∃ user u_{i-2} ∧
  content-word Jaccard(u_i, u_{i-2}) ≥ 0.3 ∧ both sides ≥ 3
  content words ([a-z0-9']+ minus frozen in-lib stopword list).
  EVENT-MASS census FIRST (tretd lesson).
- Faces: sycpress_rate = punctint sentence_lambda HL 2 / support
  8 msgs (refmark's kernel); sycpress_age + reask_age = gen4c
  sage_face VERBATIM over event-message FIRST-token flags
  (support 64). Eligibility = assistant tokens; event+boundary
  masked.
- msdose: constructed from committed gen4c_wikitext103_<tok>.npz
  (no content re-tokenization; delim "\n###\n" tokenized fresh,
  disclosed): SEED=0, N_DOCS=400, N_EX~randint(4,25), span
  len~round(clip(lognormal(ln120, 0.6), 40, 400)), dose_so_far =
  running boundary count. REPORT Spearman(dose, position)
  pre-screen + boundary-count/censored-age floors T∈{4..64}.
- Anti-dup FIRST-CLASS: Spearman vs committed refmark2k rlam
  (same token grid — assert token_ids identical) + trio pairwise;
  0.8 bar.
- Sequence: libs+tests COMMIT (freeze) → run → artifacts+stats →
  LOG pre-measure entry (PTR) incl. my four § 7 formal $0 kills
  (sleeper-latch, refusal-redux, prompt-harmfulness, turn-count).
  Card bars binding: out-of-window-by-construction +
  clock-stated-first.
- STATE: freeze pushed `648fa180c`; builder
  `labels/build_wave3_trio.py` smoke-validated (60 convs);
  **FULL RUN in background** (log
  `/workspace/agents/runpod-a/wave3_trio_build.log`) → then
  commit artifacts (wave3_refmark2k_<tok>.npz,
  wave3_msdose_<tok>.npz, wave3_trio_stats.json) + LOG entry.
  SMOKE SIGNALS (60-conv, non-record): sycpress event-starved as
  frozen (2 events/60 convs — honest outcome = report, do NOT
  widen list post-hoc); reask 10× mass (23% convs ≥1);
  msdose dose↔position rho 0.962 (menu trap (a) — running count
  is within-doc monotone; only position-matched manifests could
  rescue; naive readout dead); clock 138 tok/msg confirmed.

## House-rule cache

Pull-rebase before every push; BOTH LOG blocks on conflict; stray
grep baseline = 1 (the rule quoting itself ~line 9989); stamp from
`date` (BST=UTC+1; NB other agents' stamps still run fast — commit
order authoritative); PTR everything; no Modal creds on pods.

*Rewrite before any compact.*
