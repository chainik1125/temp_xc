# Working state — agent `mac-a`

**2026-07-25 ~19:00 PT (overnight loop).** Executing
`briefings/overnight-mac-a.md`: Modal bring-up + tsae/T1 seed top-up
{3,4,5} on Ward (bounds RECEIPTS R5). Cap $150; ledger
`briefings/MODAL_SPEND.md` (~$6.5 total after my bring-up line).

## Git position
- Freeze commit (runner): `c93473ad3` —
  `lambda_intensity/run_stage2_seedtopup_tsae.py` (3 tsae/T1 cells,
  seeds {3,4,5}, buffer 524288 UNCHANGED). mac-local APPROVED
  pre-registration at `6d7295ea2`.
- Modal app: `lambda_intensity/modal_seedtopup_tsae.py` (@ `e3a74fe70`),
  image PINNED to `c93473ad3`. Stages: bringup / caches / cells.

## In flight
- Modal bring-up (image build + in-container `run.py validate`)
  launched ~18:55 PT, running in background
  (scratchpad `bringup.log`). Next: `--stage caches` (A10G, receipts
  hard-fail in-container), then `--stage cells` (3 × A10G+8cpu+64GB,
  one frozen cell each, timeout 5.5 h, est ~$10–13/cell-h... see
  ledger). Payload lands at `modal_cells_payload.json` (cwd at launch)
  — merge LOCALLY: dup-eval-key check → append
  `results/leaderboard.jsonl` → `_merge_into_panel` → commit+push.

## Pooling-hazard audit state (briefing § 2), for the LOG entry
- (a) re-eval of a round-1 tsae cell: **IMPOSSIBLE** — round-1 Ward
  checkpoints destroyed 2026-07-25 (checkpoints/HF_MIRROR.md holds only
  the two A40 panels). Documented; relying on (b) + code-diff audit.
- Code-diff audit 038655fd→HEAD on the v1 train+eval path: ONE commit
  touches it — `fff7877c4` NaN guard in `lambda_recovery.py` with
  `.all()` fast path; `lam_hist_dense` is all-finite by construction
  (zeros-init dense fill in `build_labels.py`; asserted again
  in-container), so strict no-op. `fff7877c4` is an ANCESTOR of
  `3d954869` (the pre top-up freeze), and runpod-d NUMERICALLY verified
  round-1 reproduction under it (0.192438 vs stored 0.1924, LOG
  2026-07-24). `tsae.py`, `temp_bench/core/`, `lambda_recovery.py`:
  ZERO commits 3d954869→HEAD. v2/trace_ids additions are flag-gated
  no-ops for v1 rows.
- (b) byte-identity receipts in-container (hard-fail): stream
  `ward_stream_stats.json` + labels `lambda_labels_stats.json` must
  reproduce git-clean; traces.json re-port sha256-pinned
  (`dc6513e7d3d1…`).
- RESIDUAL CAVEAT for the LOG: new seeds train on a REBUILT activation
  cache (same builder/commit/dtype/stream bytes, different GPU: A10 vs
  the original pod's GPU); no activation-level receipt exists (originals
  gone). First cross-cache pooling in this panel — flag PENDING TEAM
  REVIEW; report pooled AND new-seeds-separate.

## Analysis plan (step 5)
Mirror `receipts_check.py` R4/R5 machinery exactly: tsae seeds
{1,2,3,4,5,42} → paired n=6 (all 6 shared with pre/T8) one-sided 95% LB
+ Welch 6v6 LB/p/df. LOG entry `mac-a (executor)` + PROPOSED R5 update
(mac-local ratifies RECEIPTS). Round-1 tsae rows are dirty-stamped
(a818cc34b/038655fd3 dirty=True) — disclose, unchanged from R5's own
basis.

## Loop discipline
work → push → `git pull --rebase` → continue. Stop: queue done / $150
cap / blocked / 07:00 PT (= 15:00 BST Sun). Ledger before/after every
launch. Rewrite this file before any compact.
