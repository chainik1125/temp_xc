---
author: Dmitry
date: 2026-07-27
tags:
  - results
  - in-progress
---

## Sprint log — Stacked SAE at paper protocol

Wall clock: T0 = 2026-07-27 05:11 UTC. Hard stop 15:15 UTC. Writing from 14:15.

## 05:11–05:25 — setup

- Worktree `.claude/worktrees/stacked-rebuttal` on branch `dmitry-stacked-arxiv`
  @ origin/arxiv 4bcd2b70 (arxiv moved overnight: ACTMIX RLHF stretch complete,
  "T-curve is an ORDER-FREE INVERTED-U peaking at T8" — relevant to the RLHF
  story; read the briefing before writing RLHF conclusions).
- runpodctl authenticated. Two RUNNING pods on the shared account —
  `backtracking-two` (4 GPU, $1.76/h), `sparse_probing_rlhf_ablations`
  (3 GPU, $8.97/h) — reject all local SSH keys; collaborator work, left alone.
  Names overlap this workstream: check with the team in the morning to avoid
  duplicated effort.
- Design-plan agent (staged execution details) still running; proceeding on the
  locked facts from the audit; will fold its report in when it lands.
- Anthropic keys + HF_TOKEN present on this Mac; pods get them via .env.

## 05:25–05:50 — adapters green, fleet up

- `stacked_pooled.py` (StackedSAEPooled + StackedBTKOnlyPooled, max-|act| pool,
  sign-preserving) + registry + ACTMIX stacked lanes + pre-registered gate
  tests: **85 tests green, run.py validate OK**. One real bug caught by
  design-review-before-run: parent `train_step` calls `self.encode` → pooled
  override would have crashed `decode`; per-position routing fixed pre-commit.
- Pushed `dmitry-stacked-arxiv` (code-before-bootstrap rule).
- **Table 2 shortcut confirmed on temp-bench leaderboard**: stacked_sae
  `d08c6498d3fa430e` seed 42 has the full detection sweep
  pr_auc S1..S32 = 0.140/0.145/0.161/0.177/0.174/0.187 (+ shuffle twin, gaps
  ≤0.027). S=8 0.177 sits between paper TopK 0.175 and TXC-pro 0.242. More
  stacked train_keys exist at seeds 1/2. **BUT** no leaderboard row exactly
  reproduces any paper Table 2 row (closest topk f437e623 diverges at S≥4) —
  the paper table is a different eval generation. Forensics agent
  (background) is tracing Fig 4/Table 2 provenance → REUSE vs RETRAIN verdict.
- Fleet v2 up (v1 killed: pods rejected all local SSH keys; registered
  id_ed25519 with the account and recreated): stacked-c7 3xpfyrmp8bj18n (H100),
  stacked-c6 dqr4p0t9vkx2zv (H100), stacked-c3 udlwpbaw9c3d8r (A40),
  stacked-rlhf 3ktwi3pacoh8v6 (A40). $6.86/h total ≈ $69/10h.
- Dead end logged: `runpodctl exec python` shares the rejected-key path
  (rsync exit 12) — key injection into running pods impossible; recreation
  was the fix.

## 06:05 — C7 forensics verdict: REUSE (and three Table 2 integrity findings)

Forensics agent (leaderboard 6,665 rows × manifest 5,713 rows, temp-bench
branch) + HF checkpoint audit:

- **`d08c6498d3fa430e` (stacked_sae, c7) is config-identical to G2** — the
  generation of every bs1024 Table 2 arm (bs 1024, 20k steps, lr 3e-4,
  bricken off, same datasource `llama_3_1_8b_base_l10_ward_nousmirror`, same
  act_cache `fb2a74be884e512a`). Zero differing fields. Its detection row
  (S1..S32 = 0.140/0.145/0.161/0.177/0.174/0.187, shuffle gap ≤ +0.027) is a
  legitimate same-generation Table 2 arm that the paper omits.
- **All 10 c7 train_key checkpoints exist on HF `temp-bench-models`**
  (incl. stacked 2.7 GB and `6832cb8006255753`, the txc_pro bs256 arm the
  paper reports but which has **no evaluated row anywhere**). REUSE:
  no C7 retraining tonight.
- Integrity findings for the camera-ready (surface in summary + fix):
  (1) no leaderboard row exactly reproduces any printed Table 2 row
  (nearest L1 0.036–0.124) — printed numbers come from an untracked eval;
  (2) ROC-AUC half of Table 2 has no artifact counterpart anywhere;
  (3) Table 2 silently mixes generations: bs256 arms are G6 (30k steps,
  early-stop on), bs1024 arms are G2 (20k, early-stop off);
  (4) c5 shuffle control is a no-op (gap exactly 0) for topk/tsae rows.
- **C7 pod plan locked**: fetch ward act cache + 10 ckpts → eval-only locked
  detection for ALL arms (self-consistent Table 2 + stacked row + the missing
  txc_pro bs256 + real ROC-AUC; zero API) → stacked Δgc at protocol 2.0.0
  from the existing checkpoint (61 q × 41 mags ≈ 2,501 Sonnet-4.6 calls ≈ $6).
  Judge ledger so far: $6 committed of $200.

## 06:20 — HF weight-level audit (forensics addendum 1)

- `d08c6498d3fa430e` weight shapes read via HTTP-Range safetensors headers:
  `saes.0..4.W_enc [32768, 4096]` — **d_sae 32768, T=5, exactly the paper C7
  shape**; config.json verbatim matches the manifest training_cfg. REUSE
  verdict now confirmed at the weights level.
- All 1283 HF train_key dirs enumerated; census byte-identical to the
  in-repo `task_hunt/tbm_census.jsonl`; 34 ckpts at d_sae 32768 incl. the
  full C7 panel and the never-evaluated `6832cb8006255753` (txc_pro bs256,
  T_max=10, 30k).
- **Pod-checklist trap added**: HF ckpts are `arch_version 1.0.0`; the arxiv
  registry bumped these archs to 2.0.0 and the registry header says bumps
  invalidate checkpoints. C7 eval MUST run on the temp-bench branch registry
  (verify stacked_sae is still 1.0.0 at temp-bench HEAD before eval; else
  pin the registry entry back for the eval run).
- Side-findings for summary: 12 txc_base ckpts declare T=10/20 but carry
  T=5 weights (silent-T5 bug, matches commit 2b4ea7eb); HF README stale
  (claims private + manifest.json layout); `toy_markov_n20_d40` datasource
  retired from data.yaml but 214 ckpts reference it.

## 06:05 — Δgc addendum (forensics agent): the two-scale story, C7 plan rewired

- **The paper's C7 cells are a 300K-step generation whose checkpoints are
  lost from git and HF.** Printed Fig 4 / Table 2 peaks reproduce exactly
  (6 of 7, incl. txc_base −12 peaks) from
  `origin/300k-tfa:purified/results/leaderboard.jsonl`; the 7th (txc_pro
  bs1024 Δgc 0.475@+16) survives only in `docs/components/c7_paper_results.md`
  @75eed8e9. **Corrects my 06:05 finding (1)**: Table 2 is tracked after all —
  I searched the wrong branch. Render path: `scripts/c7_paper_renderer.py` +
  `c7_tex_snippets.py` (arxiv's render_paper_figures.py is a stub for C7).
- The temp-bench artifacts (results.json + d08c6498) are the **20K sprint
  panel** — internally apples-to-apples (all 7 archs, 25-mag, protocol 1.0.0,
  bs1024/20K): stacked 0.328@+12 (stability 18/24, 2nd best) < txc_base
  0.426@−8. This answers the reviewer's matched-budget question at zero cost.
- **Grid rule**: Δgc peak is max-over-grid → 41-mag ≥ 25-mag mechanically
  (T-SAE extended row 0.433@+32 vs canonical 0.164@+7). Any new arm must run
  the 25-mag grid at protocol 1.0.0. The planned 2.0.0/41-mag eval and the
  $42 full-panel re-judge are both cancelled.
- **C7 rewired**: branch `dmitry-stacked-c7-300k` (off temp-bench) pins
  protocol 1.0.0, 25-mag, n_steps=300_000. The c7 H100 trains stacked at
  paper scale tonight (≈11–15 h, lands post-sprint); run_cell then evals +
  judges automatically (~$6). Caveats for the eventual row: cross-generation
  judge drift unverifiable (300K baselines unrerunnable), per-cell
  denominators vary (60 vs 61) → sub-0.02 gaps are noise.

## 06:35 — SSH unblocked (direct TCP), bootstraps launched

- Root cause of 40 min of SSH failures: the ssh.runpod.io **proxy** rejects
  every key on this team account (even account-registered ones). Direct TCP
  works: PUBLIC_KEY env → authorized_keys, endpoint from the REST API
  (`rest.runpod.io/v1/pods/<id>` → publicIp + portMappings). Recipe saved to
  memory.
- Endpoints: c7 64.247.201.61:15263 (H100) · c6 216.243.220.230:14806 (H100)
  · c3 194.68.245.1:22084 (A40) · rlhf 194.68.245.1:22045 (A40).
- Parallel bootstraps running (clone branch, uv sync, tokens, env).
  c7 → temp-bench; c3/c6/rlhf → dmitry-stacked-arxiv.
- temp-bench registry check: stacked_sae still arch_version 1.0.0 there —
  checkpoint reuse safe on the C7 pod (the 2.0.0 bump exists only on arxiv).

## 14:30–16:00 — daytime recovery + 3-seed expansion (user directives)

- **8h lost on c3/c6/rlhf**: three distinct launch bugs (system-python
  fallback in fetch script ×2; missing PYTHONPATH for `-m`-style imports;
  then the silent no-op `python -m experiments.probing.run` — section
  entry points have no __main__, must go through `run.py <section>`; then
  the tracked-leaderboard dirty-tree refusal on second cells →
  TEMP_BENCH_ALLOW_DIRTY=1, diff-hash audited). Monitor filter was also
  blind to `ModuleNotFoundError` — widened. All fixed; each documented
  here so the next sprint's harness pre-stages against them.
- **C7 300K s42: trained to completion in 8.7h** — loss 675→15.4, l0
  20.0/position exact, 300,000/300,000 steps. Eval+judge phase running.
- **User directive: 3 seeds everywhere, validate-then-scale.** Validation
  results: C7 gate PASS (above); C3 gate PASS (fresh topk mean_auc 0.8887
  vs stored paper-ckpt row 0.8848, Δ0.4%, l0=20.0 exact).
- **Seed fleet (7 new pods)**: c7-s1/s2 (H100, 300K trainings launched —
  training code validated by s42); c3-s1/s2, c6-s2, rlhf-s1/s2 booted and
  held pending their case-study stacked-s42 sanity checks.
- **C6 cohort cache** (the detection-eval blocker) building on the idle
  c6-s2 pod via `cache_em_cohort3` (self-extracts judge_outputs from
  origin/final; single-branch clones need `git fetch origin final`).
- Fleet: 11 pods ≈ $17.6/h. Judge ledger: $6 spent-committed, $12 queued
  (c7 s1/s2 evals), cap $200.

## 17:00–21:00 — first paper-scale stacked results land; scope cut to seed 42

- **Judge poisoning incident**: pod env vars don't reach SSH sessions →
  bootstrap silently wrote no token files → the first C7 judge pass ran
  keyless-then-broke: default ANTHROPIC_API_KEY has **no credit balance**,
  all 1,525 calls 400'd (unbilled), judge wrote −1 sentinels, Δgc collapsed
  to a fake 0.0@0.0. Detection metrics unaffected (no judge). Caught by
  reviewing the "done" result before trusting it. MATS key verified live
  and pushed to all pods; poisoned judge_outputs archived; re-judge clean.
- **C7 stacked s42 @ 300K (train_key 26e69fdc60452c27, proto 1.0.0,
  25-mag)**: Δgc peak **0.246 @ m=−12** (same peak magnitude as TXC-base),
  detection PR-AUC S1..S32 = 0.152/0.152/0.150/0.158/0.179/0.207.
  Against the printed panel: TXC-base 0.541@−12 > TopK 0.400 > TXC-pro-256
  0.377 > **stacked 0.246 ≈ MLC 0.246** > T-SAE 0.164. Weight sharing
  beats aggregation-alone 2.2× on inducement at paper scale. Note scale
  flip: at 20K stacked (0.328) beat TopK (0.230); at 300K TopK (0.400)
  overtakes stacked (0.246).
- **C3 stacked s42 (locked protocol)**: mean_auc **0.8694** (std 0.138)
  vs fresh-gate TopK 0.8887 / stored paper TopK 0.8848. realized L0 99.8
  per window (=20/token matched). Shuffle control −0.046 AUC → the pooled
  window code is order-sensitive. Stacked does not rescue static probing.
- **User scope decision: seed 42 only.** c7-s1/s2 pods (mid-training)
  torn down along with all remaining seed pods. c6 s1 checkpoint had
  already auto-completed pre-directive — kept on disk, zero-cost to eval
  later if wanted.
- Two more env bugs fixed on the long tail: missing `peft` for the cohort
  rebuild (uv pip install); gated `google/gemma-2-2b` needs HF_TOKEN
  exported in-session for the RLHF cache build.
- **Harvest**: private HF repo `dmanningcoe/stacked-sae-rebuttal-2026-07`;
  c7 + c3 artifacts (new leaderboard rows, trainlogs, judge_outputs,
  checkpoints) uploading; c6/rlhf harvest on completion.
- Remaining in flight: C6 cohort rebuild → s42 detection eval (zero API);
  RLHF s42 chain. Fleet down to 5 pods, $9.85/h, dropping as work lands.
