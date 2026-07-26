# Sprint log — BatchTopK-only crosscoder vs composite (window-size scaling)

Sprint start: **2026-07-26 19:38 UTC** (12:38 PDT). 10h wall-clock → ends ~05:38 UTC / 22:38 PDT.
Branch: `dmitry-btk-txc-sprint` (worktree `.claude/worktrees/btk-sprint`), based on `origin/temp-bench-anon` — the anonymized TempBench repo submitted with the paper (the *correct reference implementation*; the main repo is dirty with divergent implementations).
Isolation: another Claude is active on `dmitry-txcwins-10h` in the main checkout — all my work stays inside this worktree.

## Goal

The paper's headline experiments used a composite ReLU + BatchTopK activation for the crosscoders (to be confirmed from the reference code). Hypothesis: this was a mistake. Redo the paper case studies with a **pure BatchTopK** activation for the regular crosscoder, compare against the paper's composite results, and check whether window-size (T) scaling of the temporal crosscoder improves. One shared parameter set across all experiments; baselines are NOT rerun (reuse paper numbers).

First deliverable: tabulation of the paper plots that need to be redone.

## Key references

- Paper: `temp_xc_tex/main.tex` + `appendix.tex`; `temp_xc_tex/data_index.md` traces every figure to data/repro/branch.
- Reference impl: this worktree — `configs/locked_archs.yaml`, `src/temp_bench/`, `experiments/c1..c7`, `REPRODUCE_FIGURES.md`, `PROTOCOL.md`.
- No literal "arxiv folder" exists; the anon branch `temp-bench-anon` is the submitted snapshot.

## Compute

- 2026-07-26 19:40 UTC: probed all known RunPod ssh hosts (h100_1/2, h100_4_em, a40_2, a40_txc_1, h100_emfra_2gpu_1/2, a40_synth_3gpu(2), a40_tiny_1) — **all down**.
- Plan: local MPS/CPU for synthetic components; Modal serverless GPU (via modal_10h_skill) for LM-scale work. Auth to be verified before first GPU job.

## Course corrections (19:50–20:30 UTC)

- User: reference is the **`arxiv` branch** (existed only on origin, unfetched). Worktree repointed: `git checkout -B dmitry-btk-txc-sprint --no-track 5e6bfe37` (**pinned**; others actively push to arxiv — never chase the tip, never push to arxiv).
- `origin/arxiv` = live program branch of a multi-agent op (mac-a/b/c, runpod-1/2, mac-local orchestrator; briefings/, agents/, ledgers). Current phase **ACTMIX** — exactly this sprint's topic.
- `briefings/actmix-shared.md`: paper TXC `txc_base` = **TopK then ReLU** (k_win=8·T; negatives zeroed after selection; harm grows with T ⇒ paper d(perf)/dT biased DOWN). v2 hunt backbone = ReLU→BatchTopK (`relu-mix`). Target arm **`btk-only`** = BatchTopK, no ReLU anywhere in sparsity path. Arm labels `relu-mix`/`btk-only`/`paper-match` mandatory. **This sprint = "Dmitry's re-run gate": does the PAPER arch's d(perf)/dT improve under btk-only?** (pre-registered in the briefing).
- Implementation rule: new plugin arch + YAML + arch_version bump; **mac-a's Stage-1 convention canonical** (not yet posted at pin; recommended shape: selection over raw pre-acts, threshold gating unchanged at eval, log negative-selection count). Watch origin/arxiv read-only for it.
- Lane boundaries (avoid duplicating the fleet): **runpod-1 = probing shuffle+T-sweep, runpod-2 = EM shuffle+T-sweep** (btk-only, T∈{1,2,4,8,16}, SAE+TSAE+untrained twins); **backtracking = Aniket only (hands off)**; mac-a = v2-substrate calibration + KEEP survival; mac-c = COMPOSITION_AUDIT (paper-match pins) + HF checkpoint inventory. Paper-match arm everywhere is BLOCKED on mac-c's audit. Note: paper EM numbers came from `dmitry-em-repl`, not `final` (runpod-2 briefing caution).
- Compute (user): **Modal first, RunPod fallback**; $150/day cap; ledger in `briefings/MODAL_SPEND.md` (RUNPOD section for pods). Tokens at `~/.tokens/` per shared briefing.
- Deadline synergy: team meeting 9am PT 2026-07-27 (16:00 UTC); sprint ends ~05:38 UTC — before it.

## Incidents

- **Disk-full (20:10–20:45 UTC):** Mac hit ENOSPC (228G disk, ~63MB free) — pre-existing ~200G usage; my worktree (377M) + mac-arm torch venv tipped it. Bash tool itself was blocked (couldn't create output-capture files). Freed: worktree .venv (via helper agent), uv cache (~581M) → **5.5G free**. Plan: slim local venv; heavy deps live in Modal images.
- **Worktree swapped under running agents (19:55):** repointing e1c4f616→5e6bfe37 mid-survey invalidated two component-survey agents' live paths; they recovered via `git show e1c4f616:<path>`. Lesson: pin the base BEFORE spawning tree-reading agents.

## Recon findings so far (from agent reports; details in their transcripts)

- **Anon/purified tree (e1c4f616 = temp-bench-anon ≈ final:purified/)**: the runnable paper pipelines. c1/c2 synthetic (CUDA-hardcoded toy drivers, literal ARCH_TS grids), c3 probing (Gemma-2-2b-it L13, mean_auc, T∈{5,10,20} txc_base sweep existed but its driver is missing from tree), c4 qualitative (Haiku judge), c5 steering (Sonnet judge, peak_success_grade_at_coh_1.75), c6 EM (Qwen-14B/7B organisms, peak_align via Wang-full, arch_T hard-coded ternary `5 if txc_base else 1` — highest-risk silent-wrong-path when adding archs; bakes into eval_key), c7 backtracking (Llama-3.1-8B base L10 + R1-Distill-8B gen; delta_gc_peak; T enters 3 inconsistent ways; Aniket's anyway). Checkpoints in-tree are config.json-only (weights on HF via sync_from_hf.sh).
- **arxiv HEAD (5e6bfe37)**: restructured — root `run.py` dispatcher (`synthetic|probing|backtracking|em|rlhf` × `reproduce|sweep|render-figures|validate`), `configs/archs.yaml` (21 archs incl. txc_post_*/txc_pre_*, spectral_txc*), `configs/experiments.yaml` grids (probing 6 archs×3 seeds×8 k_feat, batch 4096, `tsae` naming), single-cell `experiments/synthetic/run.py`. **probing + rlhf evaluators at HEAD are stubs (NotImplementedError, "pending port from origin/final")** — the fleet's pods build caches + match "paper section setup exactly" from experiments/{probing,em}; verify against arxiv-recon report what is actually runnable at HEAD.

## Paper structure (arxiv `paper/`, = submission 26867 "Crosscoding Through Time")

Sections: §4 synthetic (Coupling + Denoising benches), §5.1 sparse probing (Gemma-2-2b-it L13), §5.2 backtracking (300k-step main cells — heavyweight), §5.3 EM (Qwen organisms), §5.4 HH-RLHF (negative case). Fig 1 = TikZ cartoons + global rose. The paper's in-text d(perf)/dT claim lives in §4 Denoising: "TXC-base moves above this floor as the window length grows" (main.tex:708; T=2 partial → T≥4 full, monotone). Probing appendix trains txc_base at T∈{5,10,20} as within-arch context probe (appendix.tex:116). `run.py render-figures` renders Fig 2 from leaderboard (other renderers pending port).

## Timeline

- 19:38 start; repo/branch/paper recon.
- 19:45 spawned `paper-recon` (figure inventory) + `repo-recon` (component survey — landed as c1/c2, c3/c4/c5, c6/c7 reports).
- 19:55 pivot to origin/arxiv @ 5e6bfe37; paper-recon redirected to worktree `paper/`; spawned `arxiv-recon`.
- 20:10–20:45 disk-full incident (see above); briefings read; lane map established.
- 21:20 env up (uv sync ok on arxiv lock; torch 2.8 MPS; modal client authed, profile reichers-shai-c9-dmitry). Read txc_base.py (TopK→ReLU confirmed at :166-168), txc_batchtopk.py (relu-mix + JumpReLU-threshold machinery), archs.yaml, experiments.yaml, MODAL_SPEND.md (detached-launch + Volume + freeze-pin rules; program ≈$102/$500).
- 21:40 tabulation of plots-to-redo assembled and delivered to user (task #1).
