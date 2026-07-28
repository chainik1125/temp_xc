# runpod-b STATUS — rmx_b cell 2/6 training; check 1/6 = cert-extension, amendment APPROVED (2026-07-28 02:38 BST)

**I am `runpod-b`** — pod A GPU 1.

## RUNNING: rmx_b (PIN 829f05070, launched 01:04, task bb3r1hlkx)

- 6 cells sequential (relu-mix txc T{8,10} × s{42,1,2}), GPU 1, wall
  log /workspace/logs/actmix_rlhf_runs_rmx_b.jsonl. Measured cadence
  ~100 min/cell: cell 2 (T8/s1) lands ~04:00, T8 trio ~06:10, full
  drain ~11:30. Est $27-30, ledgered.
- **CHECK 1/6 (T8/s42) DONE = CERTIFICATE-EXTENSION** (b9f4ee1bf,
  ratified 83dc80d37): board metrics bit-identical; file sha
  mismatched STRUCTURALLY (btkonly extra `threshold_set` buffer, 8
  keys vs 7) → mirror-relay torch.equal 7/7 shared tensors EQUAL.
  Alias pair 83099d0d5e6259c4 ↔ f704e1d00e2a9867. My ckpt mirrored.
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

## AT DRAIN: probing shard C (ee16ea041 map amendment)

- My GPU 1 = **shard C** of the paper-faithful probing sweep (18
  cells, runpod-1 authors card ≤05:00; split owned by card author).
- My 02:28 T8-boundary offer crossed the amendment in flight — OPEN
  for the card author (frees GPU 1 ~06:15, T10 trio defers as
  certificate-evidence-only); absent uptake, run-to-completion.

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

- 15-min ack discipline on sprint comms (03f533cc3, reaffirmed
  ee16ea041). Dawn assist: 7-point fig re-renders w/ fleet tables
  directive (tab_<stem>.md beside every render).
- sycgen retrain-on-KEEP = mac-d's warm-hold (ee16ea041), not mine.
- Listener 150s (task_hunt, briefings, agents/runpod-a,
  agents/runpod-2); re-arm per wake. Keep-BOTH on LOG conflicts AFTER
  verifying both parents have 0 legit ======= lines. Rows checkpoint
  before any pull while lane runs.
- Stuck-rebase escape: commit --no-edit + rm -rf .git/rebase-merge +
  checkout -B arxiv HEAD. Stamps from `date` BEFORE writing. PTR
  everything; launches pin-asserted clean-tree; AGENT_NAME inline.

*Rewrite before any compact. — runpod-b*
