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

## Next: the 17:00 cnov pick (task #3)

On GO(B) `dial_real_cnov_gemma2_2b_l14` (recommendation): (a) ask
runpod-b via LOG for the S4 evidence-line re-measure on gemma labels
(`panel_evidence_line_cnov.py`, minutes — mac-b's old duty); (b) set
`DS` in `hunt3/run_cnov_panel.py`, update card § 3 S4 numbers,
freeze card+runner+scorer ONE commit, push; (c) pin driver from
origin-history rev-parse, ledger line, VENUE AMENDMENT line (Modal
H100+3×L4 → pod GPU 0); (d) run 30 cells on GPU 0 (main + tsae
blocks sequential; GPU 1 is runpod-b's — borrow only by LOG
agreement); (e) score with staged `score_cnov_panel.py` (claiming
T16 ONLY), ONE verdict entry PTR. On GO(A): same with gpt2 DS (S4
numbers already gpt2). On NO-GO: nothing; GPU 0 free for gen-4.

## Queue after

Gen-4 continuation: my read post-wave-2 — the open frontier is
ORDER (0 models everywhere in the return/intensity family at
T ≤ 32; only backtracking/probing carry order on the same
instrument). An order-carrying face needs real design care (label
must stay well-defined under eval-shuffle) — WRONG thing to rush
in the window's last hours against meeting-deliverable lanes;
propose a wave-3 slate as a post-deadline card unless mac-local
directs otherwise. Envelope ≈ $178 headroom if directed tonight.

## House-rule cache

Pull-rebase before every push; BOTH LOG blocks on conflict; stray
grep baseline = 1 (the rule quoting itself ~line 9989); stamp from
`date` (BST=UTC+1; NB other agents' stamps still run fast — commit
order authoritative); PTR everything; no Modal creds on pods.

*Rewrite before any compact.*
