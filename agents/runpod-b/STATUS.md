# runpod-b STATUS — rmx_b cell 3/6 training; checks 1-2 = cert-extension (2026-07-28 03:48 BST)

**I am `runpod-b`** — pod A GPU 1.

## RUNNING: rmx_b (PIN 829f05070, launched 01:04, task bb3r1hlkx)

- 6 cells sequential (relu-mix txc T{8,10} × s{42,1,2}), GPU 1, wall
  log /workspace/logs/actmix_rlhf_runs_rmx_b.jsonl. Measured cadence
  ~100 min/cell: cell 2 (T8/s1) lands ~04:00, T8 trio ~06:10, full
  drain ~11:30. Est $27-30, ledgered.
- **CHECKS 1-2 DONE = CERTIFICATE-EXTENSION ×2** (both: board
  metrics bit-identical; file sha mismatch structural — btkonly
  `threshold_set` buffer; mirror-relay torch.equal 7/7 EQUAL).
  Alias pairs: 83099d0d5e6259c4↔f704e1d00e2a9867 (s42, ratified
  83dc80d37), f857417704b13efa↔7d51409daff2fa72 (s1, 03:48 entry).
  s1 note: shuffle_gap −0.0077 shared EXACTLY by both arms.
  Cell 3 (T8/s2) lands ~05:10; cadence ~80-85 min/cell ⇒ full
  drain ~09:30-10:00 (earlier than the 100-min estimate).
- **torch.equal relay amendment APPROVED (83dc80d37)**: cross-arm
  checks compare TENSORS; sha-equal sufficient, not necessary.
- **Per landing (cells 2–6):** sha256 my ckpt → mirror-relay
  torch.equal vs btk twin → LOG receipt (equal ⇒ cert-extension +
  alias flag; ANY divergence ⇒ A5b AUTO-RE-OPEN, report immediately +
  magnitude table) → push ckpt via scripts/push_ckpts_hf.py → rows
  checkpoint. T8 twin tks: s1 7d51409daff2fa72, s2 a2fe8d7e382dc1cb
  (mirrored under ckpts/<tk>/). T10 trio shas post ~08:00 (runpod-2
  x10 drain); T10 checks then.
- Monitor beom9lc2s fires per completed cell (wall-log tail).
- Do NOT poach probing relu-mix T2/T4 (routed runpod-1).

## AT DRAIN: paper-faithful OVERFLOW ONLY (card d9235755b §6)

- CARD_PAPER_FAITHFUL ratified 16d26642c: 21 cells over 5 GPUs
  (runpod-1 A/B, runpod-c C/D, runpod-a E), ETA 06:30-07:30 —
  **"runpod-b GPU1 joins post-rmx_b (~11:30) as overflow only."**
- My T8-boundary offer CLOSED as moot (02:40 entry). rmx_b runs to
  completion. At drain: check for unclaimed tail cells under the
  rebalance rule (claim in LOG BEFORE launching); likely none.

## CLOSED tonight (all PTR'd + ratified)

- **Width-match** (b29860ab8): NO LIFT n=3 — width not binding. $2.
- **λ̂ T{6,10} fill** (c09485d1c): T6 0.1487±0.003 dip, T10 2-seed
  0.199±0.005 + T10/s2 TRAINING-COLLAPSE (receipt-verified real at
  6.7e-10). $1.
- **dq T{6,10} fill** (88cb4f867): T6 0.3016 / T10 0.3059 on-plateau
  3/3 tight; T10/s2 trains FINE here ⇒ collapse VENUE-LOCALIZED to
  λ̂ (binding caption phrasing). $1.
- Han items (4)+(5) exhibit inputs COMPLETE.

## Durability (b4ec84b04 — COMPLIANT)

- 16/16 ckpts mirrored (15 closed-lane + rmx_b cell 1) on
  `han1823123123/temp-bench-data/ckpts/` via fleet script; shas in
  `agents/runpod-b/hf_ckpt_receipts.json` + LOG receipts. Remaining
  rmx_b ckpts push per landing.

## Standing

- **QUEUED morning: manifest owner pass** (7af84fb80 — 336 fleet-wide
  same-train_key conflicts in checkpoints/manifest.jsonl; my lines
  append-only, no unilateral fixes; wait for hub sequencing).
- 15-min ack discipline on sprint comms (03f533cc3, reaffirmed
  ee16ea041). Dawn assist: 7-point fig re-renders w/ fleet tables
  directive (tab_<stem>.md beside every render).
- sycgen retrain-on-KEEP = mac-d's warm-hold (ee16ea041), not mine.
- Listener 150s (task_hunt, briefings, agents/runpod-a,
  agents/runpod-2); re-arm per wake. Keep-BOTH on LOG conflicts AFTER
  verifying both parents have 0 legit ======= lines. Rows checkpoint
  before any pull while lane runs.
- **HOUSE RULE (660c50488, adopted 04:19): explicit paths in every
  commit — no `git add -A` while canonical jsonls are live; manual
  conflict handling on canonical files (keep-BOTH flow qualifies).**
  My rows-checkpoint path set: results/leaderboard.jsonl,
  checkpoints/manifest.jsonl, agents/runpod-b/*, task_hunt/LOG.md,
  briefings/MODAL_SPEND.md.
- Stuck-rebase escape: commit --no-edit + rm -rf .git/rebase-merge +
  checkout -B arxiv HEAD. Stamps from `date` BEFORE writing. PTR
  everything; launches pin-asserted clean-tree; AGENT_NAME inline.

*Rewrite before any compact. — runpod-b*
