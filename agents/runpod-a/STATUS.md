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

**1. RM shard 2 — HALTED BY RULING (c6e464881), wrap-up owed.**
Arc: launch 7ae8e9fd5 → fallback satisfied 19:34 (12d057b1b) →
runpod-1 proved arms BIT-IDENTICAL (their ⚑ 19:49) → HALT
approved; **I am RELEASED to reask at the T2/k20 boundary row.**
My watcher (bg task) kills the runner when pre/s42/T2's k20 row
lands (~19:10–19:25 UTC; training was step 14k at 18:58). THEN:
(1) rows checkpoint + push (T2 rows feed the weight-equality
audit (a)); (2) RM ledger ACTUALS to the halt: GPU-0 share
~17:34→~19:20 UTC ≈ 1.8 h ≈ $5–6 vs $12–15 est → **−$7-ish
corr** — one ledger line; (3) short LOG shard-halted line (cells
landed: 3 untrained + sae/s2 both-k + pre/T2 both-k); (4) rm_pin
worktree can be REMOVED after (verify no unique files first,
same cmp discipline as hunt4w2). Task #4 closes there.

**1b. reask_hr SCREEN CARD — the next build (released to it).**
GPU 0 free after the boundary. Implementation checklist (all
record numbers in committed JSONs — reask_gate_census.json,
reask_hr_premeasure.json, wave3_trio_stats.json):
- `reask_hr/` subdir (hunt4w2 clone pattern): CARD (hr face
  primary + pooled labels-only disclosure; census numbers
  verbatim; clock 119–137 tok/msg; claim zone from per-T
  censored-age floors; BINDING: position-matched manifests,
  position-floor arm, wd arms; § 4 = hunt4 rules verbatim;
  scorer committed IN the freeze) + cache_acts (refmark2k grid →
  /workspace/reask_hr_caches, ~51k rows ×128, capture per
  SCREEN_HS; llama ~20 min, gemma ~8, gpt2 ~2 on H100) + screen
  (hunt4w2/screen.py transplant: faces=(reask_hr,), manifests
  from wave3_reask_hr_<tok>.npz event arrays via
  position-stratified balanced classes, elig = assistant tokens,
  event+boundary masked) + verdict.
- Freeze card+builders+scorer ONE commit → push → pin → ledger →
  run 3 models sequentially GPU 0 → mechanical verdict → ONE
  bundle LOG entry (PTR). WRITEUP § 8 rows for trio kills BATCH
  with this screen result (mac-local's line).
- Budget: hunt envelope headroom (envelope ≈ $175); est $3–6
  pod-hours. Window: tonight fine, or morning — no deadline
  binds it (the AoE support work is other lanes').

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
- **PRE-MEASURES DONE + RATIFIED (c5023d9f3):** sycpress
  KILL-as-frozen (35 events/2k convs, docmean .995; subclass
  62.9/37.1 disclosed), msdose KILL-as-constructed (dose↔pos
  .962, pos AUC 1.0), **reask = CARD CANDIDATE** (548 events,
  floors ≤.57, anti-dup clean). Four § 7 kills formalized.
  Artifacts + entries: `a1f94e679`. mac-c second-source (their
  20:25 entry) absorbed; their re-entry paths (msdose per-doc
  span scale σ_doc; sycpress broader pinned list) are FUTURE
  pre-count amendments, not mine tonight.
- **REASK GATE CENSUS DONE (my ~19:05 entry, PTR):**
  apology-only 57.5 % ⇒ **`reask_hr` (hard-refusal-gated, 233
  events) pre-registers as PRIMARY**; pooled gate = disclosed
  secondary. Jaccard gate = 75.4 % of selection, only 15.6 %
  length-ratio-bound (median ratio .60) — selection is content
  divergence, disclosed w/ numbers. Instrument
  `labels/census_reask_gate.py` + `reask_gate_census.json`
  committed.
- **NEXT (in order):** (1) `reask_hr` variant pre-measure, $0 —
  extend build_wave3_trio machinery to the hr event stream (gate
  = HARD_REFUSAL subset per census_reask_gate.HARD_REFUSAL):
  event mass/floors/traps/anti-dup vs pooled reask + refmark.
  (2) REASK screen card DRAFT (hr primary + pooled secondary;
  T2 age probed at assistant tokens; claim zone from per-T
  censored-age floor; position-matched manifests + position-floor
  arm + BINDING wd arms — tret precedent; clock 119–137 tok/msg
  stated; census numbers verbatim; both binding menu bars).
  (3) FREEZE + screen ONLY when GPU 0 frees behind RM
  (c5023d9f3: "when GPU frees behind RM"; scorer committed with
  freeze). WRITEUP § 8 rows for trio kills batch with the reask
  screen result (mac-local's line).

## House-rule cache

Pull-rebase before every push; BOTH LOG blocks on conflict; stray
grep baseline = 1 (the rule quoting itself ~line 9989); stamp from
`date` (BST=UTC+1; NB other agents' stamps still run fast — commit
order authoritative); PTR everything; no Modal creds on pods.

*Rewrite before any compact.*

## POST-HALT ADDENDUM (~20:20 London)

Han override (mac-local 20:12 entry): HIGH-T relu-mix arm
REINSTATED for runpod-1 (T{6,8,10,16}×3 seeds, dead-latent
telemetry, per-cell equivalence checks) — **runpod-1 may borrow
MY GPU 0 for a shard, PRE-APPROVED by that entry** (claim = their
LOG line; my reask stage is CPU-first so no conflict). My halt
wrap-up stands as pushed (911c601f2): shard-2 inventory correct,
−$8 corr, worktree removed, task #4 closed. Next window: build
the reask_hr card per the checklist in § 1b above — CPU-first
(card+builders+tests+freeze); the GPU screen slots around any
high-T borrow.
