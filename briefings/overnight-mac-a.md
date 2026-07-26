---
status: active
created: 2026-07-26
for: mac-a
venue: local Mac clone ~/research/projects/agents/mac-a/temp_xc + Modal
---

# mac-a overnight — Modal bring-up + the tsae seed top-up (bounds R5)

**You are `mac-a`**, a local autoresearch agent running an autonomous
loop overnight. Read `briefings/overnight-mac-modal.md` FIRST (shared
Modal recipe, budget rules, discipline — your cap: **$150**, ledger
`briefings/MODAL_SPEND.md`). Your loop: work → push → `git pull
--rebase` (pick up orchestrator amendments) → continue. Stop when:
your queue is done, your cap is reached, you are blocked, or 07:00 PT.
Rewrite `agents/mac-a/STATUS.md` before any compact.

## 1. Modal bring-up (~30 min, ≤$5)
Repo image: debian_slim + git clone at a PINNED commit (push your
freeze commits first, pin to them) + uv sync; snapshot for reuse.
Prove it with in-container `run.py validate`. The A10 smoke test
already PASSED (token/profile work). Persist caches on a
`modal.Volume`. Containers never push git — return results, merge
locally with a dup-eval-key check, commit, push.

## 2. THE DELIVERABLE: tsae/T1 seed top-up {3,4,5} on Ward (est. $30–80)
Purpose: bound the headline pre-vs-T-SAE margin — RECEIPTS **R5** is
the program's stated NOT-bounded gap; projected Welch LB ≈ +0.013 at
tsae n = 6. Method, all binding:
- Freeze a runner for the 3 tsae/T1 trained cells ONLY (pattern:
  frozen `run_stage2_seedtopup.py`, 3d954869), **buffer_tokens
  UNCHANGED 524288** — comparability with round-1 tsae is the point.
  Commit-then-run; pin the container to the freeze commit.
- Ward substrate is UNGATED (no HF secret needed). Rebuild the Ward
  stream in-container; **verify against the committed byte-identity
  receipt BEFORE training.**
- tsae cells are CPU-buffer-bound (~2–3 h each on A40-class): request
  high-CPU containers; 3 cells in parallel; time-box at ~4 h wall.
- **Pooling hazards — BOTH discharged before pooling with round-1:**
  (a) pooling-validity audit per runpod-d's precedent: re-eval one
  round-1 tsae cell under current code → must reproduce its stored
  number (round-1 checkpoints are GONE — weights on HF, see
  checkpoints/HF_MIRROR.md? tsae round-1 weights are NOT in that
  mirror [it holds only the two A40 panels]; if re-eval is therefore
  impossible, say so and rely on (b) + code-diff audit of
  lambda_recovery since 038655fd, documented);
  (b) cache byte-identity receipt (above).
  If not dischargeable → report new seeds SEPARATELY, never pooled.
- Deliverable: LOG entry `mac-a (executor)` — the n=6 Welch/paired
  bounds, bounded-or-not stated plainly; a PROPOSED R5 update in the
  LOG (mac-local ratifies RECEIPTS); leaderboard merged locally with
  0 dup keys; PENDING TEAM REVIEW flag.

## 3. Then: assist mac-b or stretch (only if ≤$100 spent)
Offer spare hands to mac-b's screen queue via LOG claim-lines, or
take refmark caching (see mac-b's briefing) if it hasn't. Never start
anything that cannot finish by 06:30 PT.

## Acceptance gate
Everything pushed; ledger current; STATUS rewritten; stop-for-review
notes in the LOG. This briefing retires at the Sunday check-in.
