# mac-local STATUS — PRE-COMPACT SNAPSHOT #2 (2026-07-28 ~02:40 London, date-checked)

**I am mac-local: hub/orchestrator/reviewer.** Stamp from `date`
(multiple drift corrigenda on record; commit order authoritative).
**Post-compact: read THIS file, then the LOG tail from `1065b26cf`
(deliverables matrix) forward.** The deliverable surfaces are
`REBUTTAL_HANDOFF.md` + `REBUTTAL_CODE_GUIDE.md` (repo root — plots
embedded, per-item pointers, fleet map, formulas).

## ⏰ DEADLINES
Exhibits READY 11:00 BST TODAY; submission 13:00 BST; responses
amendable to Aug 3. Dmitry drafts from the HANDOFF + CODE_GUIDE.

## ⚑ HAN'S FINAL PRIORITY ORDER (02:38 LOG entry, binding)
1. **PAPER-FAITHFUL {ReLU+TopK} probing + RLHF sweeps = TOP.** btk
   work YIELDS on any GPU contention (incl. x6/x10 — runpod-2
   sequences and states the call). btk renders CPU-side, fine.
2. Hunted tasks (#4-7): EITHER arm suffices — no arm-doubling.
3. Hunt continues (mac agents only); slots 6/7 = "the gold".
4. Width-match: DONE. Pointer blocks: DONE.
**Matrix arm mapping (Han-pinned): {BatchTopK} = btk-only (NO ReLU
— the delivered sweeps); {ReLU+TopK} = paper-faithful (the sprint);
relu-mix = certificate evidence ONLY, never a matrix column.**

## THE PAPER-FAITHFUL SPRINT (live)
- **Probing: CARD_PAPER_FAITHFUL FROZEN + RATIFIED (d9235755b,
  02:39)** — `paper_txc_base_v1t` plugin = vendored 94119bc08
  training stack VERBATIM + v2 wrapper, 8/8 contract tests, arm
  'paper-faithful', **21 cells (7T×3 seeds; archived T5 anchors
  separate)**, 5-GPU shard split T1/T2-last. **LAUNCH ORDERED** —
  shards: runpod-1 GPU 0 (free), runpod-a GPU 0 (preflight PASS
  zero-sync), runpod-c both GPUs (freeze-receipts PENDING — chase
  if silent), runpod-b at rmx_b drain. ETA ~06:30-07:30. Per-cell
  HF ckpt push.
- **RLHF: agentic_txc_02 port (runpod-2)** — ETA ~04:00-04:30 for
  plugin+tests+card; grid $90-110 APPROVED; pilot cell +
  shardable lanes in-card; grid on GPU 2 (x6/x10 yields per Han).
  Paper's RLHF arm = agentic_txc_02 = MatryoshkaTXCDRContrastive-
  Multiscale (audit §6; VERIFY task #11 post-compact — load-
  bearing on every RLHF caption; failure ⇒ pull disclosure).
- Composition formulas + labeling rule (v2 columns never "paper
  base"; archived T5 anchor row separate) — in CODE_GUIDE §1.

## DELIVERABLES STATE (items 1-7 + extras)
1+2. Probing k5/k20: btk arm COMPLETE at 7 T-points (T10 s1/s2
  landed overnight); paper-faithful arm = the sprint above;
  7-point btk renders + tab_*.md tables = runpod-1 CPU-side.
3. RLHF: btk T{1,2,5,8,16} done; T4 done (x4); x6/x10 mid-run
  (yields on contention); relu-mix arm = DONE-BY-CERTIFICATE
  (tensor-identical through T16, 829f05070) + rmx_b eq-extension
  points (1/6 in: T8/s42 = 7/7 EQUAL; torch.equal relay amendment
  approved); paper-faithful arm = the port.
4. λ̂: COMPLETE (fills done; caption flags: T6-below-T4 dip,
  T10 seed-fragility VENUE-LOCALIZED; R30 both-arms + T16
  spot-check twin landed).
5. dq: COMPLETE (fills on-plateau; toy-class, screen-shuffle
  disclosed; idle-only forever).
6+7. THE HUNT (mac agents): **sycgen** = screen GO issued (within-
  domain frame binding, per-token baseline first) on mac-c's warm
  L40S; **evalage** = 6/6 label-side bands PASSED (unigram 0.586
  vs 0.60 — retryesc's killer beaten by design; harness thesis
  measured) — NOT a KEEP until probed; 3-tokenizer gap closing via
  mac-d's screen_grids.py transplant (verify 1,542 events/leg,
  gap median ~862); **retryesc_gen** = design (mac-c);
  **StruQ** = runpod-a $0 premeasures under OUR bars (Dmitry's
  steering-screen GO ≠ our instrument; structural vocab-leak note;
  task encouraged, steering METHOD parked). **First KEEP ⇒ mac-d's
  warm 2×H100 retrain within the hour (either-arm now).**
  mac-d pod = hunt-EXCLUSIVE (Han reversed my borrow).
8. tsae width: COMPLETE both tasks + quote-form (LOG 00:18).
9. Certificates: probing onset map (identity = sae+preT1;
  divergence T2+ growing; T8 sign-flip; census-vs-trace lemma —
  traces = bounds not counters; morning: 3-seed map + traces +
  certificate); RLHF identity through T16 (boundary_min_pre ≥
  2.21; refutation disclosed); A5b rmx_a cancelled-with-
  certificate + auto-re-open.

## FLEET (see CODE_GUIDE §5 fleet map)
old pod: r1 GPU0 free→shard, GPU1 night-tail→shard, GPU2 x6/x10→
RLHF-pf grid. Pod A: r-a GPU0 shard (armed), r-b GPU1 rmx_b→shard.
Pod B: runpod-c FREEZE-AND-JOIN ordered (clean-halt pattern,
resume playbook, ckpts to HF) → both GPUs to shards; receipts
pending. mac-c-screen-0728 (L40S) warm = screens. mac-d-retrain-
0728 (2×H100) warm = hunt-KEEP only. Hill-climb FROZEN (program-
best T16 k20 0.9251 r1-min; T1 collapse = open; C4 pre-registered;
resume post-rebuttal).

## KEYS / BUDGET
Generation backend = `dmitry-mats-claude-api-key` (VERIFIED LIVE;
mac-only, env-inject, never pods). `dmitry-mats-openai-key` = 401
staged (stored 128B never changed; -U re-add awaited). Han's
`anthropic-api-key` = WITHDRAWN (personal; usage was 2 verify
calls ~$0.001). $300 generation shared cap (~$50-70 committed);
$500 aggregate (~$200 spent). RunPod balance was $1,267. All
tokens rotate post-weekend.

## HOUSE RULES ADDED TONIGHT
Twin diffs join on train_key provenance + surface duplicates
(never last-write-wins); only keyed twin diffs, never band
summaries; alias exclusion list (RM_EQUIVALENCE.md) binding;
positive-control rows filtered from aggregations; smoke-check
runnables at review; .agent_id never tracked; checkpoint clause
BLOCKING in generation cards; corpora + ckpts to HF at lane
completion (ckpts/<train_key>/, LFS sha receipts); warm-holds
carry stated purpose, hub sweeps; torch.equal where buffers
differ; stamp from date BEFORE writing (many corrigenda).

## WATCHER / OPS
SESSION-scratchpad `watch_origin.sh`; run_in_background ONLY —
NEVER inline `&` (three failures tonight); ONE instance at a time
(pkill strays if echoes pile); exit 0 = arxiv push, empty
HEAD..origin = echo. Conflict recipe: python keep-both + marker
grep (anchored ^<<<<<<< = 0 baseline). Push-retry loop w/ resolve.
Pods: old=j42plcul70a2es-64410eb7, A=0lmrs9lk8apyhm-644121b8,
B=l2bp61kg82epel-64411fb1 (repo at /workspace/agents/runpod-c/
temp_xc on B!), mac-d pod=jge1fuj9hqu8et; all @ssh.runpod.io
-i ~/.ssh/id_ed25519, PTY piped-stdin, grep -av 2004.

## ⚑⚑⚑ OVERNIGHT WATCH (Han ASLEEP from ~04:40; charter cb3e34973 + gold-rule 0a2a203e8)
**Han's four orders: (i) sycgen complete + handover updated;
(ii) hunt CONTINUES; (iii) PAPER-FAITHFUL probing+RLHF finish =
PRIORITY; (iv) code handover periodically updated. PLUS: any gold
task → VISIBLE IN REBUTTAL_HANDOFF same-beat (partial embeds w/
honest stamps — sycgen = the template).**
Night state at 04:40: sycgen 15/18 (fig v4 + table embedded;
tsae trio drains ~04:45-05:00 → mac-d final render drops PARTIAL
→ I regen table same beat). Probing pf: 12+/21 cells, 5 shards
draining ~05:45-06:30 → runpod-1 folds E1-E3 + renders 7-pt
figs/tables (btk arm COMPLETE 7T×3s). RLHF: btk T{6,10} x-lanes
drain ~10:45; **10:15 checkpoint render = item-3 deliverable of
record (runpod-2 owns; data-driven landed set)**; pf port card
DUE — CHASE runpod-2 if silent by 05:00; G1 relief rule
pre-registered. Hunt: mac-c retryesc_gen (two-timescale face,
NOT age; API gen under $300 cap); screen venue = claim pod-D at
its drain or fresh L40S (pre-approved); StruQ premeasures
runpod-a. My cadence: census regen at landings; guide fleet-map
re-stamps ~06:00 + ~09:00; **10:30 FULL final pass (HANDOFF +
GUIDE + CENSUS, items 1-9 pointers verified)**; **11:00 readiness
message for Han's wake-up**. Ledger each beat. Quote-forms
binding: sycgen v2 (level story; twin control), k5 seed-spread
flag, RLHF 21:10/22:28 forms, "T10-in-flight" caption if x10
late.

*Rewrite before any compact.*
