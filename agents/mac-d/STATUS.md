# mac-d STATUS — RunPod-API executor agent (LIVE session, rewritten 23:5x 07-27)

**You are `mac-d`** — a local mac agent whose job is DYNAMIC POD
EXECUTION under Dmitry's RunPod API key. You spin up pods, run
frozen cards on them as detached jobs, repatriate results, and
TERMINATE. You are not a card owner — you execute other agents'
frozen pins. Created 2026-07-27 ~23:40 London (LOG d8609e360 +
mac-d addendum; stood up c50f7af3e; Mission 1 repointed 6452030b0).

## Read order
1. This file. 2. `briefings/actmix-shared.md` — especially the
**RunPod API governance block (BINDING)** and house rules.
3. LOG tail from d8609e360 forward.

## Session state (first session, ~23:42→)
- **Workspace VERIFIED**: venv py3.12.13 works, `run.py validate`
  OK (37 archs / 39 datasources / 5 experiments) — completes
  mac-local's provisioning receipt (c3996259b item 3).
- **Key VERIFIED read-only**: keychain env-inject works; pod list
  returns 132 pods, **zero `mac-d-*`** (historical `runpod-d`/
  `runpod-e` EXITED pods are NOT mine — never touch). 2×H100 SXM
  secure = $5.98/h < my $10/h cap. GPU type id
  `NVIDIA H100 80GB HBM3`; image template (runpod-c's pod):
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
- **Tooling PRE-STAGED (this dir, committed):** `podctl.sh`
  (create/mine/status/ssh/terminate; mac-d-* name guard both
  directions; create --dry-run receipt clean; terminate verifies
  by follow-up API query), `pod_remote_bootstrap.sh` (on-pod:
  clone public repo, detach at PIN, canonical
  `scripts/bootstrap_runpod.sh` non-interactive, AGENT_NAME=mac-d,
  validate receipt), `repatriate.sh` + `merge_rows.py` (scp rows
  home; append-only merge, dup-key eval_key/train_key,
  same-key-different-content = CONFLICT hard-stop; self-merge
  test 9843/9843 dup-identical 0 conflicts).
- **Listener ARMED** (session-local Monitor, 150 s fetch-poll):
  task_hunt/ + agents/mac-d/ + briefings/ against origin/arxiv.
- **No pod this session — norm honored, then trigger died.**
  mac-local's 23:39 norm (*pod goes up when a GPU stage exists*)
  held me at $0 through the WAIT; reask_hr then KILLED 3/3, so
  nothing was ever warmed and nothing billed. The old
  "Meanwhile: SPIN UP `mac-d-rlhfgrid-0727`" bullet was the
  pre-repoint mission's name+timing — superseded on the record
  (my 23:53 LOG entry).

## Mission 1 (RE-ARMED by e585d665b FULL THROTTLE): harness-KEEP executor
Original trigger died 23:55 07-27 (reask_hr KILL 3/3 4c231e149 +
mac-c menu exhausted — resolved no-fire, $0, no pod ever created).
mac-local's full-throttle ruling re-scopes me: **executor for
harness-corpus screens/retrains as they land; Mission 1 =
harness KEEPs exclusively.** Pipeline feeding me: mac-c generates
corpora (their pod mac-c-hunt-0728, L40S) → runpod-a co-builds/
owns the screen-side cards → revisit order (i) sycgen_age
(passed geometry, nearest KEEP) (ii) Tier-C safety picks
(iii) msdose_r2/sycpress_r2 regens. A screen or retrain card
pointed at mac-d fires the runbook below (pod name per purpose:
mac-d-huntscreen-/huntretrain-<mmdd>). The RLHF relu-mix grid is
NOT mine — A5 (57eb9edd4) owns it; Mission 2 is the contingency.

## Runbook on a card (execute in order)
1. `bash agents/mac-d/podctl.sh create` (can start the moment the
   card/directive is announced — bring-up overlaps the freeze;
   rename via arg if purpose ≠ huntretrain). LEDGER line
   in `briefings/MODAL_SPEND.md` § RUNPOD at spin-up (pod id,
   2×H100 secure $5.98/h, purpose, est).
2. Tokens (values never touch disk locally / git / logs):
   `ssh -p <port> root@<ip> 'umask 077 && mkdir -p /workspace/.tokens'`
   then pipe: `gh auth token | ssh … 'cat > /workspace/.tokens/gh_token'`;
   `scp ~/.tokens/hf_token ~/.tokens/hf_token_datasets` → same dir.
   NO Modal, NO Anthropic keys.
3. `scp agents/mac-d/pod_remote_bootstrap.sh` up; run with the
   CARD's pin; expect `BOOTSTRAP-DONE` + validate receipt.
4. Substrate sync + lane launch VERBATIM from the frozen card
   (committed cache builders; detached tmux, `TQDM_DISABLE=1`,
   wall-log under /workspace/logs/). Containers never push.
5. Monitor to completion; `bash agents/mac-d/repatriate.sh <ip>
   <port>` (dry-run then apply; CONFLICT = stop, owner decides);
   pull-rebase, push rows from the mac; ONE LOG results entry (PTR).
6. `bash agents/mac-d/podctl.sh terminate <id>` (prefer terminate;
   verify by API query — script does) → ledger ACTUALS.

## Mission 2 (LIVE WATCH — the only armed mission):
relu-mix RELIEF SHARD only if the A5 split is slipping past
~09:00 London (check LOG + row landings: rmx_a = runpod-2
T{1,2,4,6}×3, rmx_b = runpod-b T{8,10}×3; coordinate with
runpod-2 BEFORE spinning up). Wake sources: origin listener
(session Monitor) + 08:55 London alarm. Same lifecycle.

## Rules you are bound by
Pull-rebase before push; LOG conflicts keep BOTH + stray-marker
grep; stamp from `date`; PTR everything; ledger both ends;
$10/h/agent default (hub can authorize bursts within Han's $500
aggregate); NEVER touch pods you did not spin up (incl. Han's 3
hand-provisioned + anything mac-c creates + historical
runpod-d/e); name every pod `mac-d-<purpose>-<mmdd>`; token
VALUES never in git/logs/cards; tokens rotate post-weekend.
Deadline context: NeurIPS rebuttal 13:00 BST 07-28, responses
amendable to Aug 3.

*Rewrite this file before any compact.*
