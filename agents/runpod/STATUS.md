# Working state — agent `runpod`

**Last rewrite:** 2026-07-24, after completing `briefings/hunt-support-synthetic.md`.
**State: TASK DONE — STOPPED FOR REVIEW.** Both mechanism receipts delivered,
verdicts in the shared LOG, pushed. The briefing stays until mac-local
retires it. Nothing mid-flight.

## Who / where
Remote CC on RunPod (Linux), repo root `/workspace/temp_xc`. **I am `runpod`
(original box — `/workspace/.agent_id` does NOT exist; do not create it).**
Parallel agents: runpod-b/-c/-d/-e — their briefings are NOT mine.
Shared-branch rules (agents/README.md): `git pull --rebase origin arxiv`
before EVERY push (commit this STATUS first); shared files append-only;
leaderboard/manifest union-merge; cite commits by SUBJECT LINE or re-verify
SHAs post-push. Tokens in `/workspace/.tokens/` (`gh_token`, `anthropic_key`
→ export ANTHROPIC_API_KEY). Push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
32 CPU / 128 GB, NO GPU; harness-tracked background Bash; python -u.

## Just completed (2026-07-24): hunt-support-synthetic — AWAITING REVIEW
Two synthetic mechanism receipts for the Stage-2 λ̂ result, both verdicts
appended to `task_hunt/LOG.md` (my entry is the last one; read it first):

- **Item 1 (budget-dilution receipt): NO-MIRROR-DIP** — the frozen third
  branch. Mirror reproduces the real rise (0.870 → 0.952 at kernel-support
  peak T=4) but NOT the T=16 dip: all three lines flat ~0.95 to ladder end
  (A1 d=20 → T16; A2 d=40 → T32; B d=5T → T32), max |paired D| = 0.003 vs
  0.05 bar. Key finding: fixed-budget arms measurably STARVE (A1 realized
  8.9/16 atoms/window at T=16; A2 19.8/32 at T=32) yet recovery holds ⇒
  **dip and starvation doubly dissociated** (real dips without starving,
  per § 3b l0 ≈ nominal; mirror starves without dipping). Recommended at
  review: qualify RECORD § 3b's dilution clause to "cause not established".
  Untrained controls DO decline with T at fixed d_sae (capacity effect at
  init; training erases it on the mirror).
- **Item 2 (T-SAE fairness receipt): FLAT** — swept the pair-distance knob
  Δ ∈ {1,2,4,8} via new plugin arch `TSAEDelta` (`tsae_delta.py`;
  contract-tested BITWISE-identical to registered `tsae` at Δ=1) + aux
  `tsae_a0` (α=0). All settings at the DPI floor ≈ 0.41 (max |D| = 0.011);
  untrained guard exact-PASS. No rise ⇒ no runpod-d flag; skeptic $0.

Artifacts: `experiments/explorations/task_hunt/support_synthetic/`
(CARD.md frozen pre-build; run/analyze/render scripts committed pre-run;
results/ + figs/). Leaderboard 8,688 rows, 0 dup eval_keys (identical-config
bench cells reused as runner CACHE HITS — the runner dedups by eval_key;
code_version is NOT in the key). Tests: 237 pass incl. 6 new contract tests
(`tests/test_support_synthetic.py`). Disclosed in LOG: CARD § 1.2
under-counted new untrained controls (27 ran, not 15; commentary-only).

## Possible follow-ups if review asks (NOT started)
- Real-data dip explanation candidates (speculation in my LOG entry):
  content competition during training / undertraining at large T (8k real
  steps vs 30k mirror). A cheap probe: re-run real TXC-pre T=16 at longer
  n_steps, or a mirror variant with rich per-position content.
- If review wants the RECORD § 3b clause amended, that edit is mac-local's
  or runpod-d's call (their file) — my LOG entry carries the recommendation.

## Earlier completed (all reviewed & approved; do not redo)
TXC-pro dissection (`loss_dissection/RECORD.md`, § 7 rebuttal sentence
endorsed); C7 close, C6, stage-6 #3b, C5.

## Reusable know-how (this box)
- Grid engine: `design.uniform_cells(ds, F, n_steps, archs=…, k_pos_sweep=…,
  d_saes=[…], window_ts=(T,), L=32)` → `grid.run_pool` (8–16 workers).
  Backtracking canon: DS `toy_backtracking_selfexcite_d64`, F=20,
  N_STEPS=30000, seeds {1,2,42}. Window cells ~7 s (cache hits ~1 s);
  sequence-mode (tsae/dissect classes) ~2–6 min/cell under load.
  Pooled dict rule: d_sae ≥ k_pos·T (T=32 infeasible at d_sae=20).
  Runner cache: identical config ⇒ cache hit, returns existing row, no
  append — reuse is free and dup-safe.
- `lambda_recovery` probe: UNREGULARIZED LinearRegression, 1024 windows ×
  (32/T) tiles ⇒ examples shrink with T; watch p/n when raising d_sae at
  high T (frozen T≤16 verdict range was the mitigation).
- Skeptic pattern: `loss_dissection/skeptic_dissect.py` (cache-guarded, raw
  pre-parse, Meter → expansion spend.json; cumulative $11.52/$25).

## Repo state
Clean after final push (LOG + STATUS + results + figs). In sync with
origin/arxiv at last check. Next action: idle — await mac-local review of
the support_synthetic entry; pick up any new `status: active` briefing
addressed to `runpod`.

## Gotchas (this box)
- Harness blocks `sleep` (use `until …; done` or background tasks);
  background python needs `-u`.
- Rebase rewrites SHAs — cite subjects or re-verify post-push.
- Mid-grid the leaderboard/manifest are live-appended — never rebase/stash
  while a grid runs.
- Skeptic verdicts: persisted raw pre-parse, cached, NEVER re-rolled.
