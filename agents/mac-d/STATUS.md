# mac-d STATUS — RunPod-API executor agent (LIVE session, updated 02:56 07-28)

## ⚑⚑⚑ HEADLINE (02:56): § 5 T-AXIS AMENDMENT POSTED (48→36 cells) — shard0 DONE 18/18-amended, shard1 on pace, drain ETA ~03:35–03:55
Freeze 74d260321 launched the 48-cell grid; **T∈{6,10} cannot pass
the canonical eval** (L=32 divisibility, power-of-two gate; λ̂
template axis IS (2,4,8,16); zero T6/T10 rows fleet-wide) — my
card-writing error, amended in RETRAIN_CARD **§ 5** + LOG 02:54.
12 doomed cells = `ok:false` receipt rows in the as-run shard jsons
(~$2 burn on the 6 trained ones, disclosed); the 36 survivors ARE
the amended grid. Shard1 (all-trained half) left to complete —
batchtopk T1 anchors landed r=0.470/0.487/0.489. **NOTE: pod clock
= UTC, mac = BST — never read pod file mtimes as London times**
(cost me a phantom-stall scare, retracted in LOG). Renderer
`sycgen/render_tsweep.py` ALREADY WRITTEN (runbook step 4 is now
just: run it). Est $15–21 ledgered at launch.

**POD-D COORDINATES (mine, jge1fuj9hqu8et, $5.98/h, warm-hold until
LANE done):** `ssh root@64.247.201.51 -p 16977`. Repo at
/workspace/temp_xc (detached @ 74d260321); tokens at
/workspace/.tokens (gh+hf×2); sycgen caches at
/workspace/sycgen_caches; a git stash holds my smoke-test rows
("mac-d smoke rows" — pod-local audit, never push).

**AT DRAIN, in order:** (0) advance pod checkout to the AMENDMENT
pin (`git fetch origin arxiv && git checkout --detach <pin>` —
overlay imports the amended `WINDOW_TS`; NEVER before SHARD1-DONE,
workers re-import from disk); (1) on-pod
`.venv/bin/python -m experiments.explorations.task_hunt.sycgen.shuffle_overlay`
(reads both shard jsons; identity 2e-3); (2)
`bash agents/mac-d/repatriate.sh 64.247.201.51 16977` then push
rows from the mac (pull-rebase; LOG conflicts keep-both);
(3) on-pod ckpts → HF: `.venv/bin/python scripts/push_ckpts_hf.py
<train_key>…` for the 18 TRAINED rows' train_keys (from shard
jsons; ratified `ckpts/<train_key>/`), sha receipts HERE; (4)
harvest overlay json → commit; run
`…sycgen.render_tsweep` on the mac (renderer already written →
`figs_writeup/fig_sycgen_shuffle_tsweep.*` + summary json); (5)
ledger ACTUALS; (6) ONE LOG bundle entry (PTR). THEN the lane is
done → TERMINATE pod via
`bash agents/mac-d/podctl.sh terminate jge1fuj9hqu8et` + ledger,
UNLESS a new card claims it (check LOG first).

**Session-local watchers armed pre-compact (they persist while the
session lives; re-arm if genuinely fresh):** origin listener 150s
(task_hunt/ + agents/mac-d/ + briefings/); retrain-shard monitor
(ssh poll 240s, DONE+failure signatures); 08:55 London alarm →
Mission-2 checkpoint (A5 relu-mix slip decision — likely moot,
rmx_b is certificate-extension; check LOG first, coordinate with
runpod-2 before ANY relief spin-up). Stamp discipline: stamps ONLY
from a separate preceding `date` call (corrigenda 02:38, twice
before). Screen results HF-durable
(`hunt_corpora/sycgen_20260728/screen/`), corpus + gates likewise.

## PREVIOUS HEADLINE: SYCGEN BUNDLE = KEEP 3/3 (02:28, first hunt KEEP)
Screens I executed on MY pod (mac-c handoff 7cc702599 + GO
dc3cb8fd9; freeze 782e9cad3): gpt2/gemma2/llama31 ALL KEEP, zero
kill clauses (tok 0.50–0.53 ≈ chance; window 0.62–0.65 at
T64/actxmean; wd passes; order-0). Results committed
(sycgen/results/). Pod borrow REVERSED (ee16ea041) — 2×H100
hunt-dedicated. **NOW BUILDING: retrain card (matrix standard 7-T
{1,2,4,6,8,10,16} × seeds {42,1,2} × shuffle instrument, btk-only
arm per 692cb mapping; substrate = sycgen v1; template = λ̂
SHUFFLE_OVERLAY_CARD) + datasource plugin (single-file drop).
Commit-then-run, then detached on pod-D, repatriate, HF ckpts
(push_ckpts_hf.py, runpod-a's), ledger.** Screen actuals $2
(−$5 corr). If resuming: read the 02:28 LOG entry first.

## ⚑ WARM-HELD POD (0da616bca visibility rule — purpose stated here)
**`mac-d-retrain-0728` = pod `jge1fuj9hqu8et`**, 2×H100 SXM secure,
$5.98/h, RUNNING since 00:39 07-28 under Han's pre-provisioning
order (per-agent $10/h WAIVED for hunt lanes; ceilings = $500
aggregate + $300 generation). **Purpose: first hunt-KEEP matrix
retrain starts the HOUR the verdict posts.** Terminate when the
LANE is done, not between stages. Bring-up state: see Mission 1b.

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

## Mission 1a — RESOLVED BY STAND-DOWN (00:40): evalage generation = mac-c's
Their RUNNING line (a266eeb76) landed during my smoke — my claim's
pre-stated trigger fired, zero churn. My 2-doc smoke (~$0.02,
ledgered, artifacts deleted) validated the same canonical backend
they run. Residual: I support evalage premeasures/screen if asked, and my
**corpus-card request stands with mac-c's arbitration** (sycgen_age
or retryesc_gen — I drive generation from this mac the moment a
card exists; 0da616bca repeats this assignment).

## Mission 1b — WARM-HOLD COMPLETE (00:55): mac-d-retrain-0728 ready
Bring-up receipts: ssh `root@64.247.201.51 -p 16977`; repo detached
at pin f2c4f5898 (guard-verified); tokens gh+hf×2 seeded 0600 (NO
Modal/Anthropic); on-pod `run.py validate` OK; both H100s visible;
**runner smoke vs committed cell PASSED** (synthetic txc_base s0
--smoke, full metrics, AGENT_NAME=mac-d). Substrate: at-card-time
by design — the KEEP corpus doesn't exist yet, so nothing
corpus-specific exists to sync; committed cache builders take
~5 min per pod-A receipts (disclosed trade). HOLDING WARM for the
first KEEP's matrix retrain (starts within the hour of the
verdict, pre-authorized f0ac106e4 item 3). Re-pin at card landing.

## Mission 1c′ — disposition (c) EXECUTED (01:53, d897a00e4): geometry RESCUED within-domain
Doc-mean 0.636–0.795 across all six domains (vs 0.858 confounded),
position 0.608–0.731, pooled usable 511,907 ≥ 250k (2×), 158
qualifying strata; trivia_qa thin (6/8, disclosed). (b) v2 looks
UNNECESSARY. Instrument committed-then-run (numbers of record
reproduced-identical at committed sha); artifact in-tree + HF
(sha 9c9f9215…). **Awaiting design-owner/hub screen call** —
per-token baseline FIRST on mac-c's pod, within-domain frame
pre-registered. KEEP ⇒ warm-pod retrain within the hour.

## Mission 1c — sycgen v1 LANDED + HALTED AT STOP (01:47, entry 9559f7102)
400 convs / 986k tokens / 1,118 challenges @2.79/conv / 0 API
failures / ~49 min / est-basis actuals $6–9. **Realised geometry
ALL PASS** (position 0.808; doc-mean 0.858 — 0.007 from retryesc's
fatal number, CI upper crosses the bar, disclosed; floors clean).
**Vocab STOP FIRED** — mechanism isolated: schedule FLAT by domain
(2.56–2.94), tokens/conv 12× by domain ⇒ length-normalization
channel (retryesc family). **HALTED before screen per card §4;
disposition = mac-c (design owner) + hub.** My weakly-held rec:
screen-anyway first (unigram bar measures the leak, $2–3), v2
length-controlled only if the face survives within-domain.
**Durability receipts:** HF `temp-bench-data/hunt_corpora/
sycgen_20260728/` — npz 2bdd9aca…, receipt 54181c6e…, gate
2701e6d2…, sha manifest; same three committed in-tree (9559f7102).
v2-if-chosen is UNBLOCKED: mac-c's elicit_lib checkpointing landed
(279963722) with 3-line wiring instructions for run_sycgen —
checkpoint clause binding on any relaunch.

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
