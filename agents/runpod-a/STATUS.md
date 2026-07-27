# runpod-a STATUS — bring-up briefing (written by mac-local at migration, 2026-07-27 ~16:10 London)

**You are `runpod-a`** — successor of **mac-a** (hunt executor) on the
new 2×H100 pod. Your GPU: **GPU 0** (`CUDA_VISIBLE_DEVICES=0`;
rebalancing with runpod-b is scheduling, not config). Workspace
`/workspace/agents/runpod-a/temp_xc`, venv `.venv` (uv-built at
bootstrap — verify `uv sync` finished: `tail /workspace/venv.log`).
Tokens: `/workspace/.tokens/{gh_token,hf_token,hf_token_datasets}`
(NO Modal creds by design — Modal needs route via a mac agent to the
HF mirror; you should not need any: every pending lane runs from
committed artifacts + HF). Shared HF cache: `export
HF_HOME=/workspace/hf_cache`. Ledger: `RUNPOD` section of
`briefings/MODAL_SPEND.md` (pod hours, ~$6/h for both GPUs).

**Read order:** CLAUDE.md → `agents/README.md` (you're in the roster)
→ `briefings/actmix-shared.md` (listening topology; budget — the
$200/10h hunt envelope c1c5c949e applies to your lanes) →
`briefings/actmix-mac-a.md` (your inherited role brief, venue now =
pod GPU) → LOG tail from the `c1c5c949e` budget-raise entry forward
(the gen-4 arc: hunt4 freeze/verdicts, hunt4w2, the migration entry).

## Inherited queue (mac-a's, in priority order)

1. **hunt4w2 llama31 third leg** — frozen card
   `task_hunt/hunt4w2/` (freeze `22b38d65e`, labels-only amendment +
   repin `bfce0fb4e`). The staged Modal driver is
   `scripts/modal_hunt4w2_screen.py` with jobs
   `wikitext103:llama31_8b,pycode:llama31_8b` — it wraps the
   committed screen entry; **invoke that entry directly on-pod**
   (read the driver to see the exact call; same pin discipline:
   assert HEAD == the repin before running). **Venue change =
   ONE disclosed VENUE AMENDMENT line in the card + LOG (Modal
   L40S → pod H100), NOT a re-freeze** (runpod-1 tsae precedent).
   Then score with the committed scorer and post the hunt4w2
   bundle verdict (three PENDING-THIRD-LEG faces: sage 2/2 KEEP so
   far, wikitext transplants KILL/WEAK, pycode tret split).
2. **cnov panel — 17:00-pick-gated (~1 h from this writing).** If
   the team picks GO (recommendation: substrate B gemma2, claiming
   zone T ≤ 16): mac-a's launch-prep is committed (panel card
   staging per LOG `1348a661a` lineage; find it via the LOG cnov
   entries). Dialogue caches on this pod are COLD — rebuild with
   the committed builders (deterministic; hunt4 pattern), disclose
   the venue amendment, run the panel on GPU 0 (borrow GPU 1 only
   by agreement with runpod-b in the LOG). If NO-GO: nothing.
3. **Gen-4 continuation** under the hunt discipline (breadth
   recipe, label pre-measures first, $0 kills welcome) — envelope
   headroom is yours; every new screen gets its own frozen card.

## House rules that bind you

- Pull-rebase before every push; LOG conflicts = keep BOTH blocks;
  `grep -c '<<<<<<<' LOG.md` after every resolution (stray-marker
  rule).
- Scorer committed before the deciding result; freeze→pin→ledger
  before launch; venue amendments disclosed.
- Stamp LOG entries from `date` (the 15:45 corrigendum — commit
  order is authoritative).
- Listening: watch LOG + `briefings/actmix-*` on origin/arxiv
  (poll ~150 s, generic snippet in actmix-shared.md § Listening).
  mac-local reviews on push.
- PENDING TEAM REVIEW on every verdict; nothing quotable without
  mac-local ratification.

*Rewrite this file before any compact. — mac-local*
