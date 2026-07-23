# Working state — agent `runpod`

**Last rewrite:** 2026-07-23, mid `briefings/txcpro-dissection.md` (C7 was
APPROVED in round-3 review; that briefing is deleted; this is the new task).
**State: DISSECTION GRID RUNNING** — card + build committed, contract tests
green, 720-cell grid in flight.

## Who / where
Remote CC on RunPod (Linux), repo root `/workspace/temp_xc`. **I am `runpod`
(original box — `/workspace/.agent_id` does NOT exist; do not create it).**
Parallel agents: `runpod-b` (story pack, DONE awaiting review), `runpod-c`
(em-redo, phase A frozen). Shared-branch rules (agents/README.md): `git pull
--rebase origin arxiv` before EVERY push (commit this STATUS first); shared
files append-only; leaderboard/manifest union-merge; **cite commits by
SUBJECT LINE or re-verify SHAs post-push**. Tokens in `/workspace/.tokens/`
(`gh_token`, `anthropic_key` → export ANTHROPIC_API_KEY). Push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
32 CPU / 128 GB; harness-tracked background Bash; python -u.

## Active task: TXC-pro loss dissection (briefings/txcpro-dissection.md)
Which TXC-pro loss component (matryoshka / multi-distance contrastive /
longer-windows-as-T-axis) helps the txc_batchtopk_post backbone — synthetic
five-bench discriminating set ONLY (no probing arm). Prior: "mostly nothing
helps"; a clean nothing IS the deliverable. ~48 h, skeptic-only spend ≤ $5.

**Done so far (all committed, in order):**
1. Card FROZEN pre-build — commit "loss dissection: ablation card FROZEN
   (pre-build) — TXC-pro components on the synthetic discriminating set";
   card at `experiments/explorations/synthetic/loss_dissection/CARD.md`.
   Key frozen choices: one sequence-mode class `TXCPostDissect`
   (`src/temp_bench/archs/txc_post_dissect.py`), four registry entries
   txc_post_{plain,mat,ctr,both} (loss-only diffs; params/init identical);
   mat = H=8 nested prefixes ⌊G·d_sae/8⌋ (paper spec, G=8 term = plain
   recon); ctr = cosine InfoNCE at window shifts {1,2}, w=1/(1+Δ), full
   code (toy-scale TXC-pro convention), grafted verbatim from history
   commit 2fa9bdab lineage; anchor-only side effects ⇒ exact zero-weight
   reduction; S_MAX=2 for ALL variants. Decision rules: paired-by-seed,
   cell bar max(2·SE, 0.05 / nmse 0.02), HELPS ≥2/9 cells + 0 negative;
   Gate B bridge to canonical txc_batchtopk_post leaderboard rows
   (135/135 verified present pre-freeze); untrained guard = variant rows
   identical. Predictions frozen incl. anti-AC sharpening (ctr hurts
   phasepair sign if anything).
2. Build committed pre-run — commit "loss dissection: variants + contract
   tests + driver + analyzer + skeptic (pre-run commit)". 11 contract
   tests PASS; full suite 198 passed + 1 skip; `run.py validate` OK
   (17 archs).
3. Smoke cell (post-commit): txc_post_ctr backtracking T=4 k=2 s=1 →
   λ=0.953, ~115 s/cell (sequence mode ≈ 16× window-mode cell cost —
   SequenceBuffer regenerates per step; expected, noted for the record).
4. **GRID IN FLIGHT** (background, 16 workers):
   `python -u -m experiments.explorations.synthetic.loss_dissection.run_grid 16`
   → log `…/scratchpad/dissect_grid.log`, results per bench at
   `loss_dissection/results/<bench>_dissect_grid_results.json`. 720 cells
   (144/bench × 5), ETA ~1.5–2 h from ~2026-07-23 (see log timestamps).

**Next (if resuming mid-flight):** wait for grid → `python -m
….loss_dissection.analyze` (mechanical verdicts; do NOT hand-pick cells) →
skeptic ONLY on recovery-metric HELPS claims (`skeptic_dissect.py`,
cache-guarded, session cap $5) → RECORD.md narrative + research STATUS § 0
bullet + this file → pull-rebase (leaderboard/manifest union-merge; origin
had 5 incoming commits from runpod-b/-c, no file overlap with mine) → push
→ STOP for review. Briefing stays until mac-local deletes it.

**Card discipline:** verdicts are read off analyze.py's frozen rules;
pre-backbone extension only if time remains and only as a card amendment;
key-mapping note: card's "gauc" = leaderboard `eauc` key.

## Earlier completed (reviewed; do not redo)
- **C7 (2026-07-23): reasoning int/eq cell CLOSED NEGATIVE-at-resolution**
  — APPROVED round 3. **C6: empty passing set** — APPROVED round 2.
- **Stage-6 #3b recipe POSITIVE; C5 r3 ABORT** (reviewed).
- Spend cumulative $11.01/$25 (expansion meter; dissection adds skeptic
  only).

## Repo state
Local ahead 2 (card + build commits), behind 5 (runpod-b/-c pushes; no
overlap). Working tree: leaderboard.jsonl + checkpoints/manifest.jsonl
live-appended by grid workers — do NOT rebase/stash mid-grid.

## Gotchas (this box)
- Harness blocks `sleep` (use `until …; done` or background tasks);
  background python needs `-u`.
- 5-family models reject `temperature`; calibrations SEQUENTIAL.
- Skeptic verdicts: persisted raw pre-parse, cached, NEVER re-rolled.
- Rebase rewrites SHAs — cite subjects or re-verify post-push.
- render_report churns fig-PDF timestamps — checkout figs/ if only
  binaries moved.
