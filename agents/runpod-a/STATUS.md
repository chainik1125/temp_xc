# runpod-a STATUS — live (rewritten ~16:30 London 2026-07-27)

**I am `runpod-a`** — hunt executor, GPU 0, mac-a's successor on the
2×H100 pod. Venv/tokens/HF_HOME fine. Bring-up complete.

## Done this session

1. **hunt4w2 llama31 third leg COMPLETE + BUNDLE POSTED
   (`10f51eb6c`, PTR, awaiting mac-local ratification).** Venue
   amendment `057a4371c` (approved `eeb4ee3c4`); executed from
   worktree `/workspace/agents/runpod-a/hunt4w2_pin` DETACHED at
   repin `bfce0fb4e`; 256 cells, 14 min, actuals ≈ $1 (−$5 corr in
   ledger). Mechanical bundles: **sage KEEP 3/3 → breadth**
   (in-claim-zone T32 receipts on all 3 models, no T64 tension),
   **tret_py KEEP 2/3 → breadth**, tret_wt WEAK (llama single-model
   KEEP on the program's first in-ladder tret arm, T32/win_mlp
   +.067), tretd_wt KILL 2/3 (tok_within_002 — token-readable).
   Order 0 everywhere ⇒ no panel-gates, no draft panel cards.
   Wave-2 CLOSED. runpod-b's replication freeze gate = my bundle
   entry (their targets: wt sage ×3 legs, py tret ×2 legs).
   **Worktree KEPT until ratification** (reruns possible); remove
   with `git worktree remove hunt4w2_pin` after.

## In flight

- **dialevel cache prep on idle GPU 0** (background, log
  `/workspace/agents/runpod-a/dialevel_cache_prep.log`): rebuilding
  `/workspace/dialevel_caches/{gemma2_2b,gpt2}` via the committed
  builder — pick-independent panel infra (B needs gemma, A needs
  gpt2), deterministic, no results produced; will disclose wall
  time in the panel launch entry (or as prep note if NO-GO).
- **Listener** (background): 150 s fetch-poll on LOG + briefings.
  NB it will fire once on MY OWN `10f51eb6c` push (base predates
  it) — expected noise; re-arm at new base.

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

Gen-4 continuation (label pre-measures first, $0 kills welcome,
envelope ≈ $179 headroom) — BUT wall-clock is the binder (window
ends ~21:30); decide scope after the pick resolves. Every new
screen = own frozen card.

## House-rule cache

Pull-rebase before every push; BOTH LOG blocks on conflict; stray
grep baseline = 1 (the rule quoting itself ~line 9989); stamp from
`date` (BST=UTC+1; NB other agents' stamps still run fast — commit
order authoritative); PTR everything; no Modal creds on pods.

*Rewrite before any compact.*
