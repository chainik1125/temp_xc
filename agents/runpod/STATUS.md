# Working state — agent `runpod`

**Last rewrite:** 2026-07-24, after completing `briefings/txcpro-dissection.md`.
**State: DISSECTION DONE — STOPPED FOR REVIEW** (briefing's acceptance
gate; the briefing stays until mac-local review deletes it). No active
task.

## Who / where
Remote CC on RunPod (Linux), repo root `/workspace/temp_xc`. **I am `runpod`
(original box — `/workspace/.agent_id` does NOT exist; do not create it).**
Parallel agents: `runpod-b` (story pack, awaiting review), `runpod-c`
(em-redo). Shared-branch rules (agents/README.md): `git pull --rebase
origin arxiv` before EVERY push (commit this STATUS first); shared files
append-only; leaderboard/manifest union-merge; **cite commits by SUBJECT
LINE or re-verify SHAs post-push**. Tokens in `/workspace/.tokens/`
(`gh_token`, `anthropic_key` → export ANTHROPIC_API_KEY). Push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
32 CPU / 128 GB; harness-tracked background Bash; python -u.

## Just completed: TXC-pro dissection (2026-07-24) — do not redo
**The bundle is mostly selection on noise; ONE component survives.**
Full record: `experiments/explorations/synthetic/loss_dissection/RECORD.md`
(+ `dissection_table.md`, `dissection_table_pre.md`, CARD.md). What a
reviewer will check:
- Freeze order: commit "loss dissection: ablation card FROZEN
  (pre-build) …" → build commit "… (pre-run commit)" → grids; § 9
  pre-extension amendment "… AMENDMENT frozen (pre-build) …" → pre build
  commit → pre grid. Skeptic-summary enrichment (absolute levels)
  committed BEFORE skeptic execution, anti-claim direction.
- Post family: Gate B 9/9 ×5 benches, untrained guard exact-0, 720/720
  cells. ONE surviving HELPS: ctr → frequency velocity T=8 (+0.084 k=1 /
  +0.093 k=2, all seeds positive, 0.69→0.78 absolute; skeptic NO KILLS).
  Recipe mat-"HELPS" KILLED (e_metric_leak: both arms far below chance).
  Matryoshka helps recovery nowhere, hurts 4 places.
- Pre extension: 696/696 feasible cells ((T=8,k=4) infeasible at F=20,
  logged); ALL 15 primary verdicts NEUTRAL ⇒ prediction (v) CONFIRMED —
  ctr's lift is decode-structure-contingent. Skeptic not triggered ($0).
- Predictions ledger honest: (i) falsified in one place (the surviving
  claim), (ii) wrong (mat helps nothing incl. capability), (iii) wrong
  venue (help is AC power, not DC), (iii-b) genuine-salvage fork, (iv)
  never fired, (v)/(vi) confirmed, (vii) partially wrong.
- Contract tests: 15 (parametrized both families; plain-reduction ≤1e-6
  vs both parents; hook-identity; param-identity; pair/offset checks).
  Suite green. Spend $0.51 (cumulative $11.52/$25; $5 session cap).
- 1416 leaderboard rows (canonical runner, code-version stamped); grid
  dumps + tables under `loss_dissection/results/`.

## Earlier completed (reviewed; do not redo)
- **C7: reasoning int/eq cell CLOSED NEGATIVE-at-resolution** (round-3
  APPROVED). **C6: empty passing set** (round-2 APPROVED). **Stage-6
  #3b recipe POSITIVE; C5 r3 ABORT** (reviewed).

## Repo state
After the dissection-final commits: expect clean tree in sync with
origin/arxiv (verify `git status -sb`). If resuming: nothing mid-flight;
wait for review or a new briefing.

## Gotchas (this box)
- Harness blocks `sleep` (use `until …; done` or background tasks);
  background python needs `-u`.
- Sequence-mode cells cost ~115 s vs ~7 s window-mode (SequenceBuffer
  regenerates per step) — budget accordingly.
- 5-family models reject `temperature`; calibrations SEQUENTIAL.
- Skeptic verdicts: persisted raw pre-parse, cached, NEVER re-rolled.
- Rebase rewrites SHAs — cite subjects or re-verify post-push.
- render_report churns fig-PDF timestamps — checkout figs/ if only
  binaries moved.
