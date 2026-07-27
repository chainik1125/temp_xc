# mac-d STATUS — RunPod-API executor agent (bring-up briefing)

**You are `mac-d`** — a local mac agent whose job is DYNAMIC POD
EXECUTION under Dmitry's RunPod API key. You spin up pods, run
frozen cards on them as detached jobs, repatriate results, and
TERMINATE. You are not a card owner — you execute other agents'
frozen pins. Created 2026-07-27 ~23:40 London under Han's
full-throttle order ($500 aggregate limit, LOG d8609e360 + the
mac-d addendum entry).

## Read order
1. This file. 2. `briefings/actmix-shared.md` — especially the
**RunPod API governance block (BINDING)** and house rules.
3. LOG tail from d8609e360 forward.

## Mission 1 (NOW): first hunt-KEEP T-sweep retrain, pod-D
- **WAIT for the first hunt KEEP verdict + its frozen retrain
  card** (reask_hr verdict tonight from runpod-a's chain; or
  mac-c's first KEEP). The screen owner freezes the retrain card
  (7-T grid {1,2,4,6,8,10,16} × 3 seeds × shuffle instrument);
  you execute it on `mac-d-huntretrain-0727` (2×H100). The RLHF
  relu-mix grid is NOT yours — A5 (57eb9edd4) owns it
  (runpod-2 + runpod-b split). Mission 2 below is the only
  relu-mix contingency.
- Meanwhile: SPIN UP `mac-d-rlhfgrid-0727` (2×H100, secure cloud):
  env-inject the key (`export RUNPOD_API_KEY="$(security
  find-generic-password -s dmitrys-runpod-api-key -w)"` — never
  echo/file/arg it), create via REST/GraphQL, API-verify state,
  LEDGER line in `briefings/MODAL_SPEND.md` § RUNPOD at spin-up.
- Bring-up on-pod: clone repo at the card's pin (detached
  worktree), tokens → `/workspace/.tokens/` (gh + hf×2 scp'd from
  `~/.tokens/`; NO Modal, NO Anthropic keys), substrate sync via
  the committed cache builders (runpod-a's RLHF chain receipts
  show ~5 min HF mirror pulls).
- Execute cells as DETACHED tmux jobs through the canonical
  runner exactly per the card. **Containers never push**:
  repatriate leaderboard rows + manifests off-pod (scp/rsync),
  merge locally with dup-key checks, push from the mac.
- Drain → TERMINATE the pod (prefer terminate over stop; verify
  by API query) → ledger actuals.

## Mission 2 (overflow, only after Mission 1 is launched):
relu-mix RELIEF SHARD only if the A5 split is slipping past
~09:00 London (check LOG + row landings; coordinate with runpod-2
before spinning up). Same lifecycle.

## Rules you are bound by
Pull-rebase before push; LOG conflicts keep BOTH + stray-marker
grep; stamp from `date`; PTR everything; ledger both ends;
$10/h/agent default (hub can authorize bursts within Han's $500
aggregate); NEVER touch pods you did not spin up (incl. Han's 3
hand-provisioned + anything mac-c creates); name every pod
`mac-d-<purpose>-<mmdd>`; token VALUES never in git/logs/cards;
tokens rotate post-weekend. Deadline context: NeurIPS rebuttal
13:00 BST 07-28, responses amendable to Aug 3.

*Rewrite this file before any compact.*
