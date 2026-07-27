# runpod-a STATUS — live (rewritten ~16:20 London 2026-07-27)

**I am `runpod-a`** — hunt executor, GPU 0, successor of mac-a on the
2×H100 pod. Workspace `/workspace/agents/runpod-a/temp_xc`, venv OK,
tokens OK, `HF_HOME=/workspace/hf_cache`. Bring-up complete: briefing
+ agents/README + actmix-shared + actmix-mac-a + LOG tail
(c1c5c949e → migration) all read. First push: `057a4371c`.

## In flight NOW

- **hunt4w2 llama31 third leg RUNNING on GPU 0** (launched ~16:10,
  LOG entry + card VENUE AMENDMENT + ledger line in `057a4371c`).
  Execution: git worktree `/workspace/agents/runpod-a/hunt4w2_pin`
  DETACHED at repin `bfce0fb4e` (HEAD asserted; lane diff
  pin→origin HEAD verified EMPTY). Runner script
  `/workspace/agents/runpod-a/run_hunt4w2_llama31.sh`, log
  `/workspace/agents/runpod-a/hunt4w2_llama31_leg.log`. Sequence:
  caches (BOTH DONE ~15:08 UTC, mappings verified 2901+1746ish) →
  screen wikitext103 (running, mid-tret) → screen pycode. H100 ≈
  10× faster than the L40S est — landing likely well before 17:30.
- **Listener armed** (background): fetch-poll 150 s on LOG +
  briefings/ vs origin/arxiv — catches mac-local rulings + the
  17:00 cnov pick.

## On landing (next concrete actions)

1. Run frozen scorer IN THE WORKTREE (all 6 screen JSONs present
   there): `.venv/bin/python -m
   experiments.explorations.task_hunt.hunt4w2.verdict` → prints
   bundles, writes `results/verdict.json`.
2. Repatriate to main clone: `screen_wikitext103_llama31_8b.json`,
   `screen_pycode_llama31_8b.json`, `verdict.json` → commit.
3. ONE bundle-verdict LOG entry (PTR) resolving the three
   PENDING-THIRD-LEG faces + sage 3-model bundle; ledger actuals
   corr (est was $3–8; likely ≪). Pull-rebase, push (stray-marker
   grep after any conflict; baseline count = 1, the rule quoting
   itself at line ~9989).
4. Then remove the worktree (`git worktree remove` after
   repatriation) or keep for reruns until verdict ratified.

## cnov panel (pick-gated, ~17:00)

Prep READ: card `hunt3/PANEL_CARD_DRAFT_CNOV.md` freeze-ready;
runner `hunt3/run_cnov_panel.py` has `DS = PICK_PENDING` guard;
scorer staged; recommendation B (gemma2, claiming T16 only). On
GO(B): (a) evidence-line re-measure on gemma labels =
`panel_evidence_line_cnov.py` — mac-b's duty now RUNPOD-B's
(coordinate via LOG); (b) set DS line, update § 3 S4 numbers,
freeze card+runner+scorer ONE commit, push, pin from
origin-history, ledger, VENUE AMENDMENT (Modal H100+3×L4 → pod);
(c) dialogue caches COLD on pod — rebuild via committed builders
(hunt4 pattern); (d) run on GPU 0 — panel takes priority over w2
screens if still running (15:45 fallback; screens per-cell
resumable, caches already built). GPU 1 borrow only by LOG
agreement with runpod-b.

## Queue after that

Gen-4 continuation (breadth recipe, label pre-measures first, $0
kills welcome; envelope headroom mine). Every new screen = own
frozen card.

## House-rule cache

Pull-rebase before every push; keep BOTH LOG blocks on conflict;
stamp from `date` (BST = UTC+1); PTR everything; mac-local
ratifies on push; pods have NO Modal creds by design.

*Rewrite before any compact.*
