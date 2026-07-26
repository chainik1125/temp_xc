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

## Early read (partial data, 199/216 rows, 2026-07-26 23:20 UTC)

- **Gate metric (coupled gauc): slope flips + in btk-only** (+0.041 vs −0.006 composite, pooled non-clipped cells). Composite T-degradation confirmed in-level: k=2 gauc 0.988 (T1) → 0.829 (T8).
- **Level surprise: composite dominates btk-only in absolute gauc at nearly all (k, T)** (k2/T1: 0.988 vs 0.624). Mechanism candidate from the fingerprint + nmse: btk-only realizes its full budget (l0/win ≈ k_win; no zero-picks) and reconstructs far better (k1/T1 nmse 0.003 vs 0.124) but its signed codes lose the ReLU's nonnegativity prior, so decoder atoms mix features → lower |cos| recovery. The composite's ReLU is simultaneously a bug (selection-zeroing worsens with T) and a feature (single-feature prior at low T).
- Markov/denoising bench emits no gauc at current protocol (hidden_features None) — the paper's latent-level denoising claim is NOT re-testable through this evaluator; eauc/nmse only. State plainly in summary.
- tsae frozen bar (coupled k1 gauc 0.809): composite clears it everywhere (0.93+); btk-only sits at/near it (0.81-0.89).

## Recon deliveries (2026-07-27 ~00:15 UTC — both agents, complete)

- **paper-recon (full inventory, 100% of main+appendix read):** canonical paper = ICML 2026 mechinterp workshop camera-ready, DEANONYMIZED, "Crosscoding Through Time". 17 figures + 4 tables; **15 figures full redo, 4 tables partial (TXC rows only), 2 TikZ cartoons safe**. I had missed fig:txc_umap-app + fig:sentence-app (c4 moved to appendix, TXC-only) and the inline c7 steered-text figure. **THE REFRAME: main.tex:362 says "σ is BatchTopK"; "ReLU" appears ZERO times in the paper (grep-verified); the code does TopK-then-ReLU → the btk retrain makes the paper's stated architecture true, resolving the appendix's own TopK(app:29)-vs-BatchTopK(app:33) contradiction.** The monotone-in-T denoising claim was already deleted in the camera-ready (commented at main:884; weak version at main:707-712); the paper flags missing T-work 3× — my T-sweep fills an acknowledged hole. Abstract's scorecard + ~20 prose numbers change with TXC numbers. c6 α-rescaling (√T, app:246-248) was measured under the composite — re-measure before any c6 comparison (runpod-2's lane; flagged). No caption edits needed for an activation change.
- **arxiv-recon (complete):** 3 of 5 evaluators are stubs at HEAD (probing, backtracking, rlhf — each a multi-day port from origin/final); only synthetic + EM runnable, EM needs a RunPod-side cohort cache. txc_base has ZERO T-swept rows anywhere (67 rows, all T=5) — both arms of my comparison necessarily fresh (as done). My driver is corpus-consistent (run_experiment + arch_hparams_override). render-figures Fig2 has a stale protocol filter (1.2.0 vs current 1.3.0) — my analysis.py is the right vehicle. Degenerate-cell catch (5/18 combos dense at d_sae=20) → already marked/excluded in my slopes; added the **d_sae=50 wing** as the stronger fix. Arm-label carrier = arch registry name (schema has no arm field; extra=forbid) — matches what I did; noted for mac-a alignment. k_win discrepancy (briefing "8·T" vs registry k_pos=20) = hunt-vs-paper convention; paper-match labeling waits on mac-c.
- **Historical composite denoising probes found** (temp_xc_tex/notes/c2_synthetic/data/denoising_probe_results.json): txc_base k=5 lp_global_r2 by T: 0.285 (T2) → 0.422 (T4) → 0.410 (T5) → 0.394 (T6) → 0.335 (T8) — **already non-monotone, peak T≈4** (why the claim was deleted). Independent evidence for the harm mechanism; cite in summary. (Old d_sae=40 regime; my evaluator lacks this latent-level probe — post-hoc probe pass on checkpoints is a follow-up, checkpoints not repatriated tonight.)

## Paper structure (arxiv `paper/`, = ICML mechinterp workshop camera-ready "Crosscoding Through Time")

Sections: §4 synthetic (Coupling + Denoising benches), §5.1 sparse probing (Gemma-2-2b-it L13), §5.2 backtracking (300k-step main cells — heavyweight), §5.3 EM (Qwen organisms), §5.4 HH-RLHF (negative case). Fig 1 = TikZ cartoons + global rose. The paper's in-text d(perf)/dT claim lives in §4 Denoising: "TXC-base moves above this floor as the window length grows" (main.tex:708; T=2 partial → T≥4 full, monotone). Probing appendix trains txc_base at T∈{5,10,20} as within-arch context probe (appendix.tex:116). `run.py render-figures` renders Fig 2 from leaderboard (other renderers pending port).

## Timeline

- 19:38 start; repo/branch/paper recon.
- 19:45 spawned `paper-recon` (figure inventory) + `repo-recon` (component survey — landed as c1/c2, c3/c4/c5, c6/c7 reports).
- 19:55 pivot to origin/arxiv @ 5e6bfe37; paper-recon redirected to worktree `paper/`; spawned `arxiv-recon`.
- 20:10–20:45 disk-full incident (see above); briefings read; lane map established.
- 21:20 env up (uv sync ok on arxiv lock; torch 2.8 MPS; modal client authed, profile reichers-shai-c9-dmitry). Read txc_base.py (TopK→ReLU confirmed at :166-168), txc_batchtopk.py (relu-mix + JumpReLU-threshold machinery), archs.yaml, experiments.yaml, MODAL_SPEND.md (detached-launch + Volume + freeze-pin rules; program ≈$102/$500).
- 21:40 tabulation of plots-to-redo assembled and delivered to user (task #1).
- 21:55 user: away for hours, "venture boldly". Re-armed origin/arxiv convention watcher (bg, 150s poll, path-filtered per the new listening-topology commit fd4cc10f — still no mac-a note).
- 22:10 `txc_base_btk` implemented (BatchTopK on raw squashed pre-acts, no ReLU in sparsity path, JumpReLU-threshold eval, neg_frac fingerprint; paper budget k_win=k_pos·T kept incl. clip). 8 contract tests green (incl. selected-negatives-survive vs composite-zeroes distinction; B=1 positive-regime bridge). Registry entry mirrors txc_base per-section hparams. Commit 948110dd.
- 22:25 leaderboard archaeology: composite txc_base synthetic rows exist ONLY at T=5 (k_pos {1,2,5,10,20} × seeds {1,2,3}, n_steps=10000, L=5 default). Hunt family swept T via arch_hparams_override with eval_window_L=32. Decision: my sweep runs BOTH arms at uniform **eval_window_L=40** (tiles T∈{1,2,4,5,8,10,20}), training matched to existing composite rows (10k steps, b1024, buffer 2M, seeds {1,2,3}), k_pos {1,2,5,10,20}. Degenerate-cell caveat: k_pos·T ≥ d_sae=20 clips both arms identically (dense); slopes exclude clipped cells. gauc/eauc are decoder-direction metrics (L-independent) → existing per-token baseline rows remain overlayable as reference lines without rerunning (protocol-version caveat to state).
- 22:35 driver (shard = arch×bench×T, 15 cells) smoke-tested both arms locally; freeze 0ce04d6f0; branch pushed to origin (own branch only — container clones need the SHA reachable). Modal launcher on the house pattern (L40S, max 8 containers, Volume temp-xc-btk-rerun, resume-from-Volume, repatriate-merge with dup-key check). Ledger line appended (probe ~$2 est).
- 22:45 single-shard TIMING PROBE launched attended (txc_base_btk : coupled : T=8) — fan-out gated on measured $/cell. analysis.py ready (d(perf)/dT gate figure + mixing fingerprint + slopes JSON).
- 23:00 probe: 1 cell ran (~6 min @10k steps), 14 refused — container tree went dirty after cell 1 appended the TRACKED results/leaderboard.jsonl; house rows carry dirty:true+diff hash, so container driver now passes --allow-dirty (code identity still pinned by _assert_pinned). Rescope from measured cost: T grid {1,2,4,5,8,10} (T=20 degenerate at every k: k·T≥d_sae), k_pos {1,2,5} (10/20 clipped almost everywhere), n_steps 6000, A10G. 216 cells, est ~$18. FAN-OUT launched detached at freeze 2ce33ac5.
- 23:20 dataviz pass on analysis.py: Okabe-Ito palette validated (CVD ΔE 29+), k_pos as linestyle not alpha, slopes in legend, frozen tsae refs overlaid (gauc/eauc only — decoder-direction metrics; baselines had 10k steps vs our 6k = bias in baselines' favor, stated).
- 23:45 **mac-a's CANONICAL btk-only convention landed (92db86c4)** — my implementation matched on substance except two eval-path items: EMA source set {>0}→{!=0} and threshold_set flag replacing the −1.0 sentinel. Conformed: renamed registry entry to `txc_base_btkonly` (canonical *_btkonly pattern), relu_mode hparam constructor-asserted, arch_version 1.1.0, new negative-threshold gating test. Full suite 351 green. Training path bit-identical to v1.0.0 → in-flight sweep rows stay valid for gauc/eauc; analysis folds both names into the btk-only arm. **mac-local APPROVED the convention (9e634bed) — pods GO.** Plan: after main sweep, redo the 12 btk shards under the canonical name (freeze v3 = 08711a37; composite shards Volume-skip), keep v1.0.0 rows as reproducibility cross-check.
