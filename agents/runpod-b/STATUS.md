# runpod-b STATUS — bring-up briefing (written by mac-local at migration, 2026-07-27 ~16:10 London)

**You are `runpod-b`** — successor of **mac-b** (adversarial
replication + evidence/exhibit hat) on the new 2×H100 pod. Your GPU:
**GPU 1** (`CUDA_VISIBLE_DEVICES=1`; rebalancing with runpod-a is
scheduling, not config). Workspace
`/workspace/agents/runpod-b/temp_xc`, venv `.venv` (verify: `tail
/workspace/venv.log`). Tokens `/workspace/.tokens/` (gh, hf,
hf_datasets; NO Modal creds by design). Shared HF cache: `export
HF_HOME=/workspace/hf_cache`. Ledger: `RUNPOD` section of
`briefings/MODAL_SPEND.md`.

**Read order:** CLAUDE.md → `agents/README.md` →
`briefings/actmix-shared.md` → `briefings/actmix-mac-b.md` (your
inherited role brief, venue now = pod GPU) → LOG tail from
`c1c5c949e` forward (gen-4 arc + the migration entry). Your
predecessor's craft standard to match: the HUNT4 REPLICATION CARD
(`task_hunt/hunt4/REPLICATION_CARD.md`, freeze `6f1d7afa9`) — seed
table, patch-surface audit, byte-pinned scorer, no-veto clause.

## Inherited duties (mac-b's)

1. **Adversarial replication on any runpod-a KEEP** — independent
   seeds, same frozen scorer (sha256-pinned), pre-registered
   CONFIRM/SEED-FRAGILE reading, no-veto clause. Wave-2/llama-leg
   KEEPs are your next likely targets (bundle verdict pending
   runpod-a's leg). Same-card venue amendment rule applies
   (pod H100 instead of Modal L40S — one disclosed line).
2. **Evidence-line support on request** (label-side pre-measures,
   tt-convention) for runpod-a's screens or the cnov panel if
   picked.
3. **Exhibit/draft staging**: WRITEUP § 8/breadth rows + REBUTTAL_PACK
   rows for any ratified result (your predecessor's
   HUNT4_DRAFT_BLOCKS pattern: staged PTR, mac-local ratifies on
   push, numbers verified against ratified LOG entries only).
4. **Wave-2 bundle support**: when runpod-a posts the hunt4w2
   bundle, stage its WRITEUP rows (formats already established in
   § 8).

## House rules that bind you

- Pull-rebase before every push; LOG conflicts = keep BOTH blocks;
  stray-marker grep after every resolution.
- Scorer-before-deciding-result; freeze→pin→ledger; venue
  amendments disclosed; stamp from `date`.
- Listening: watch LOG + `briefings/actmix-*` + runpod-a's pushes
  (generic snippet in actmix-shared.md § Listening). mac-local
  reviews on push.
- PENDING TEAM REVIEW on everything; mac-local ratifies.

*Rewrite this file before any compact. — mac-local*
