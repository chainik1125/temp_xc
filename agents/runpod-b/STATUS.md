# runpod-b STATUS — armed and listening (rewritten 2026-07-27 16:09 London, wall clock)

**I am `runpod-b`** — mac-b's successor (adversarial replication +
evidence/exhibit hat), GPU 1 (`CUDA_VISIBLE_DEVICES=1`) on the
2×H100 pod. Workspace `/workspace/agents/runpod-b/temp_xc`, venv OK
(torch 2.8.0+cu128, CUDA true), tokens gh/hf/hf_datasets (NO Modal
by design), `HF_HOME=/workspace/hf_cache`.

## Bring-up: COMPLETE (~16:09 London)

Read order done (CLAUDE.md → agents/README.md → actmix-shared →
actmix-mac-b → LOG tail c1c5c949e→HEAD → hunt4/REPLICATION_CARD.md
craft standard → hunt4w2 card/screen/scorer/driver). Environment
verified. LOG bring-up entry pushed (this commit).

## Live state

- **runpod-a launched the hunt4w2 llama31 third leg on GPU 0**
  (~16:10 entry, 057a4371c): worktree at repin bfce0fb4e, ETA both
  screens ≤ ~18:30, then the bundle-verdict LOG entry (PTR). That
  posting is MY FREEZE GATE.
- **Listener armed** (background loop, 150 s poll): watching
  `experiments/explorations/task_hunt` + `briefings` +
  `agents/runpod-a` on origin/arxiv. Re-arm after every wake.
- **Shared HF cache pre-warmed by me** ($0 GPU, network only):
  gpt2 529M + gemma-2-2b 9.8G + llama31-8B 15G — runpod-a's leg and
  my replication legs both start from warm weights.
- **GPU 1: idle, reserved for my replication legs.** Borrowing =
  LOG agreement only (runpod-a's cnov contingency notes this).

## Staged, ready to freeze (my next action on the bundle posting)

`agents/runpod-b/HUNT4W2_REPLICATION_CARD_DRAFT.md` +
`agents/runpod-b/replication_screen_w2_DRAFT.py` — the wave-2
adversarial replication, hunt4 craft standard: seed table
8013/8234/11242/7099/7 (asserts old values in-wrapper),
patch-surface audit re-done against `hunt4w2.screen` (w2 nuance:
manifests are committed scout pools → MATCH_SEED shifts CAP
subsampling; all sites verified by line number), scorer sha
`f883dee9…` pinned + asserted, output isolated to
`results/replication/`, no-veto clause, venue = pod H100 GPU 1
(ONE amendment line).

**Freeze procedure (on bundle posting):** (1) fill § 0 target list
from the posted bundle (rule: every (corpus, model) leg carrying a
bundle-KEEP face; expected wikitext103×{gpt2,gemma2_2b} via sage
2/2 + pycode×gemma2_2b via tret + any llama31 KEEP legs);
(2) re-verify scorer sha (re-pin if an approved patch landed);
(3) move card → `hunt4w2/REPLICATION_CARD.md` (drop DRAFT block),
wrapper → `hunt4w2/replication_screen.py`; ONE commit, push;
(4) ledger line (est $5–8, 3 legs); (5) run legs on GPU 1 from an
asserted-clean HEAD == freeze pin: per leg `cache_acts` →
`replication_screen corpus:model`; (6) score, ONE LOG entry
(CONFIRM/SEED-FRAGILE per face, § 3 reading), commit results, PTR.

## Queue after replication

1. **WRITEUP § 8 rows for the w2 bundle** on its RATIFICATION
   (HUNT4_DRAFT_BLOCKS pattern: staged blocks, PTR, mac-local
   ratifies on push; numbers only from ratified LOG entries). Note:
   WRITEUP § 8 tretd row (~line 444) tail says "bundle verdict
   pending at press time" — update it with the bundle outcome.
   sage (new face, intensity family) may need a fresh row; tret_wt/
   tret_py/tretd_wt fold into return-family rows.
2. REBUTTAL_PACK rows for any order-carrying KEEP (none expected:
   sage order 0 so far).
3. Evidence-line support on runpod-a request.

## House rules (binding)

Pull-rebase before every push; LOG conflicts keep BOTH blocks +
stray-marker grep (`grep -c '<<<<<<<' LOG.md`); scorer before
deciding result; freeze→pin→ledger; venue amendments disclosed;
stamp from `date` (15:45 corrigendum); PTR everything; mac-local
ratifies.

*Rewrite before any compact. — runpod-b*
