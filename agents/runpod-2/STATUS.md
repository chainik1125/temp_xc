# runpod-2 — working state

**Am:** EM executor on the ACTMIX shared 3×H100 pod, GPU 2 only
(`source scripts/set_agent_env.sh runpod-2`). Clone
`/workspace/agents/runpod-2/temp_xc`, own venv (peft 0.19.1 added —
not in pyproject; needed for the LoRA-merge builders).

**Task:** `briefings/actmix-runpod-2.md` — EM shuffle control +
T-window sweep, both arms (btk-only now; paper-match blocked on
mac-c). Deadline: rebuttal-grade numbers before 17:00 London
2026-07-27.

## Setup pinned (recon done, 2026-07-26 ~21:15 London)

- **Task = § 5.3 medical organism detection** (Qwen2.5-7B-Instruct +
  `andyrdt/Qwen2.5-7B-Instruct_bad-medical`), paper layer **L15**
  (resid_post, hs16). Eval = `temp_bench.evals.em` (detection 3.0.0
  port, ALREADY IMPLEMENTED — `experiments/em/run.py`'s "stub"
  docstring is stale). Primary `pr_auc_S16`; within-window shuffle +
  realized-l0 are built into the eval.
- **Training stream** = `medical_em_prompts` BASE-forward
  (paper/origin-final convention: train on base, detect on organism
  — TRACKING.md § 1), datasource `qwen_2_5_7b_instruct_medical_l15`
  (configs/data.yaml). Runner never auto-builds caches → out-of-band
  builder writes `results/data_cache/<data_key>/acts.npy`.
- **Cohort eval cache** = stage-4 judge outputs recovered from
  `origin/final` git history; rebuilt via
  `phase4_em_depth.py cache` → `/workspace/conv_depth_caches/
  em_medical/` (all 29 hs). Cohort REPRODUCED EXACTLY: 1728
  rollouts, misaligned frac 0.323 (matches TRACKING.md).
- **Grid design** (card = `experiments/explorations/actmix_em/CARD.md`):
  TXC-post btk-only retrained per T ∈ {1,2,4,8,16}; SAE + TSAE
  btk-only trained once (per-token; tsae REJECTS T≠1 — bands);
  untrained twins = n_steps 0. Seeds {42,1} paired (paper em
  convention), s2 stretch. d_sae 32768, 20 atoms/token nominal
  parity (em-redo Part II convention; k_pos = 20·T per window for
  post), n_steps 25k, lr 3e-4, warmup 1k, bf16, batch 1024
  (tsae 32 seqs). Endpoint-first dispatch (T16+T1 s42 first).
- **Aniket convention** (read-only, origin/neurips-aniket): shuffle
  semantics ALIGNED (per-row within-window input permutation,
  deterministic seed — same as the em eval port). Their extra
  controls (reversal, circular shift, positional-stack SAE) are NOT
  in EM's protocol 3.0.0 — divergence stated with reason in card.

## Flags raised (for mac-local/mac-c — in CARD.md too)

- F1: `experiments/em/run.py` default datasource is the 14B FINANCE
  l24 anchor, but eval/cohort/published-negative are 7B MEDICAL L15.
  Executing medical-L15; finance not covered this phase.
- F2: substrate = base-trained/organism-detected per origin/final;
  dmitry-em-repl (actual paper-numbers source) may differ — Phase B
  waits on mac-c's pin; Phase A is labeled btk-only, never
  paper-match.
- F3: k-budget + registry names for btk-only come from mac-a's note
  (single-source). Card carries my default (20/token parity) marked
  PENDING; adopt mac-a's on arrival, before any cell launches.

## In flight (updated ~21:40 London)

- [DONE] cohort cache: 29 hs × 1728 rollouts at
  `/workspace/conv_depth_caches/em_medical/` — integrity reproduced
  (1728 / 0.323 / d_model 3584 vs TRACKING.md).
- [DONE] train caches: BASE-forward L15 (`56a61e3776062439`, paper
  convention) + organism L{9,13,15} (Phase-B insurance,
  `2d0a9b6176e91bad` at L15).
- [DONE] mac-a convention landed 92db86c41, mac-local APPROVED
  9e634bed9 (pods GO) — `*_btkonly` names consumed verbatim.
- [DONE] pre-freeze smokes: validate OK; all 3 btkonly archs green
  through em runner; T16/k320 override green; n_steps=0 green.
- [armed] origin listener (archs/, task_hunt LOG, COMPOSITION_AUDIT,
  briefings/) every 150 s — session-local.
- [next] freeze commit (CARD + cells + driver + builder + this
  STATUS + ledger est) → pull-rebase → push → launch lanes a/b/c
  detached on GPU 2 with --pin from origin/arxiv.

## Descope ladder (pre-stated, applied in order if time runs short)

1. drop seed 1 window cells (keep s42 curve + s1 token cells);
2. drop T=2 (keep {1,4,8,16});
3. drop T=8 (keep {1,4,16});
4. untrained twins s42 only (already default), tsae s42 only.
Never dropped: T=1 + T=16 endpoints s42, sae cell, shuffle eval
(in-eval), realized-l0 disclosure, honest side-by-side with the
paper's published negative.

## Git

Branch arxiv @ fd4cc10f9 (pulled 2026-07-26 ~21:00 London). Nothing
committed by me yet. Pull-rebase before every push; pod never
pushes without it.
