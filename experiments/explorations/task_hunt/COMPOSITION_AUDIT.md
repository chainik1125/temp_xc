# COMPOSITION_AUDIT — what activation composition did each PAPER result actually use?

**Agent:** mac-c (branch/commit archaeology + HF inventory, `briefings/actmix-mac-c.md`)
**Date:** 2026-07-26 (first push ~21:45 London; PARTIAL — see §6/§7 PENDING markers, second push follows)
**Status:** probing + backtracking + EM(part 1) + RLHF + synthetic + branch sweep + HF inventory COMPLETE; the single open item is **A6** (EM 7bmed-figure provenance; subagent running, second push).
**Rules of evidence:** every composition claim cites `<sha>:<path>` + quoted code. Where evidence runs out the verdict is **AMBIGUOUS** + the artifact that would disambiguate. All verdicts PENDING TEAM REVIEW.

---

## 0. TL;DR for the pods (paper-match arms)

**runpod-1 (sparse probing).** Paper-match composition is **pinned**:

| arm | composition at eval | selection unit | k scaling |
|---|---|---|---|
| TXC-base (`txc_bare_antidead_t5`) | **TopK→ReLU** (select on raw pre; ReLU zeroes selected-negatives after) | per-window (B, d_sae) | k_win = k_pos·T |
| TXC-pro (`phase5b_subseq_h8`) | **TopK→ReLU** | per-window, full-T_max encode at probe time | k_win = k_pos·T (train k ≠ inference k: 100/200 per appendix) |
| TopK SAE (`topk_sae`) | **TopK→ReLU** | per-token | k = k_pos |
| MLC (`mlc`) | **TopK→ReLU** | per-layer-window (5 layers) | k_win = k_pos·L |
| **T-SAE** (`tsae_paper`) | **ReLU→BatchTopK at train; ReLU→threshold (JumpReLU-style) at inference/probing** | flat over (B·d_sae) at train; per-token threshold at eval | k·B·T flat at train; eval L0 set by threshold, not k |
| TFA (`tfa_big`) | ReLU→TopK on the novel part (module code); probing-venue config = A5 (c1/c2 venue confirmed `sae_diff_type="topk"`) | per-token | kval_topk |

The paper's probing panel was **composition-inconsistent by design**: every arm except T-SAE/TFA selects on raw pre-activations (TopK→ReLU); the T-SAE port is ReLU-first with threshold inference — i.e. the paper's T-SAE arm already ran (approximately) the "btk-only-like" convention while TXC/SAE arms ran the mixed one. Do NOT assume one "paper composition" across arms.

**RLHF + synthetic (context for arm labels):** RLHF arms = same probing modules/ckpts (topk_sae k500 TopK→ReLU; tsae k500/k20 ReLU→threshold at eval; TXC arm = `agentic_txc_02` matryoshka-contrastive, TopK→ReLU per-window k_win=500). Synthetic (c1/c2) ran on `origin/final`'s purified pipeline with line-identical ports: same split (TopK→ReLU family vs tsae ReLU→BatchTopK/threshold vs tfa ReLU→TopK). Details §6–7.

**runpod-2 (EM).** Two distinct EM lineages exist; **the camera-ready c6 figures (7bmed) match neither committed pipeline exactly** — provenance pin PENDING (A6). What is pinned now: (a) Dmitry's `dmitry-em-repl` headline (α∈{−6..6}, FRA-vs-additive on Qwen-14B) froze at `de0c8ea8f` (2026-05-09) and contains **no TXC/TSAE arm at all** — all arms are per-token TopK SAEs (k=64, d_sae=102 400), fresh ones = sae-lens 6.43 **TopK→ReLU**; (b) the temp_bench-native c6 pipeline (on `origin/final`) compares `sae_arditi` (**ReLU→TopK** per-token, d_sae=32 768, k=128) vs `txc_base` (**TopK→ReLU** per-window, k_win=k_pos·T=25·5=125 per c6 override) on Qwen-14B **finance**, and its 30K-step txc_base seeds-1/2 checkpoints exist publicly at `han1823123123/temp-bench-models` (`a0d4491420ec0b14`, `ad1243ed0f41db35`, datasource `qwen_2_5_14b_instruct_finance_l24_resid_post`). If your paper-match arm must mirror the camera-ready figs (Qwen-7B bad-med, T-SAE + TXC panel), WAIT for the A6 resolution (second push).

**Checkpoint jackpots (eval-only paper-match arms are feasible):**
- `han1823123123/txcdr-it` (public): **exactly the 12 IT-side probing checkpoints** (`it_*__seed42.pt`, incl. `it_txc_bare_antidead_t5`, `it_topk_sae`, `it_tsae_paper_k20/k500`, `it_mlc`, `it_phase5b_subseq_h8`), 12.3 GB, last-modified 2026-04-30.
- `han1823123123/txcdr-base` (public): 145 BASE-side ckpts, seeds 1/2/42 complete markers, + per-run training_logs.
- `han1823123123/temp-bench-models` (public): 1 283 train_key-hashed dirs each with `config.json` (arch, datasource, seed, training_cfg) — the purified v1.0.0 paper-cell store, last-modified **2026-05-07 01:24**, i.e. 28 min before the first c3 paper-fig commit.
- `han1823123123/temp-bench-data` (public): SAEBench probe caches (252 GB, base l11-15 + IT), act caches incl. Qwen L24 `resid_post` (c6) and Llama L10 (c7), and `runs/` = 430 purified eval artifacts (judge outputs, steering grades, `phase1_unsteered.json`; ts 2026-05-05/06).
- Backtracking (Aniket's lane, informational): `aniketdesh/ward-stage-b-dictionaries` (~35 GB, 13 ckpts) + `aniketdesh/ward-stage-b-cache` (~20 GB).

---

## 1. Method

Read-only git forensics against remote refs (`git show/ls-tree/log/grep/rev-list`, no checkouts), $0 compute; HF inventory metadata-first via `huggingface_hub` (file trees + a handful of KB-scale config/metrics samples; no bulk downloads). Working tree untouched except this audit + STATUS + LOG PTR. Subagent reports (EM, backtracking, branch sweep; RLHF/synthetic pending) were spot-checked for citation format and merged; their full texts are preserved in mac-c's session transcript.

**Branch genealogy (context for every verdict below).** Trunk `8359fd44d` (02-16). `han-phase7-unification` (tip `94119bc08`, 05-03) is the ONLY paper branch that is a true ancestor of the camera-ready line. `final` forked 05-03 (`d8004304d`) → `final-aniket` = camera-ready (05-07) → both `arxiv` (current) and `neurips-aniket` (rebuttal). `final` then continued alone to 05-20 (471 commits not reachable from final-aniket/arxiv). `dmitry-em-repl` and `aniket-runpod-ward-stage-a` are **disconnected islands** (merge-base with everything = trunk); their results entered the paper by file-copy/re-run into `purified/experiments/{c6_em,c7_backtracking}`, not by merge. `temp-bench` = flattened public cut of final; `temp-bench-anon` = parentless submission-day squash.

---

## 2. Per-task verdict table

| paper task | source of the shipped numbers | result-producing commit(s) | composition verdict | ckpts on HF? |
|---|---|---|---|---|
| Sparse probing (c3) | training runs + arch code: `han-phase7-unification`; final 8-budget/3-seed eval: **post-05-03, never committed to git** (A1) | training: index rows committed `cb3096fc8`..`b50ab56be` (04-27..05-02); figs: `3d8bdef0a`, `bd8fbf6fc`, `3ee3ae61f` (final-aniket, 05-07) | **PINNED per arm** (§3): TopK→ReLU everywhere except T-SAE = ReLU→BatchTopK/threshold | YES — txcdr-it (12 seed-42), txcdr-base (3-seed); final k20/20K-step cells: expected in temp-bench-models (A1) |
| EM (c6) | camera-ready figs (7bmed): **A6 PENDING**; Dmitry's replication line froze at `dmitry-em-repl:de0c8ea8f` (05-09); temp_bench c6 (finance) on `final` | `de0c8ea8f` (neg6 headline); final's c6: runs/ artifacts ts 05-05/06 | em-repl arms: per-token TopK SAEs only (fresh = TopK→ReLU, Nura's = A3); purified c6: sae_arditi ReLU→TopK vs txc_base TopK→ReLU | YES — temp-bench-models (14B-finance txc_base 30K, seeds 1/2 confirmed; 7B-med TBD A6); Dmitry's SAEs: `dmanningcoe/em-repl-2026-05-0{7,8}` (private, his account) |
| Backtracking (c7) | `aniket-ward-stage-b` tip **`a62175ee7`** (05-03) cut25 artifacts, ported into final via ATTRIBUTION.md; stage-a proper = shared trunk (arch code frozen since 03-22) | `a62175ee7` (headline +1.574 Δgc); ladder `211ad8648`→`a62175ee7` | **PINNED per arm** (§5): txc/txc_h8/mlc TopK→ReLU per-window k_win=k_pos·T; topk_sae TopK→ReLU per-token; tsae/tfa ReLU→TopK per-token kval=20 override | YES — aniketdesh/ward-stage-b-dictionaries (13 ckpts, self-describing configs) |
| RLHF (hh_rlhf) | produced on **`han-phase7-agent-c`** (`023d52c24`+`fcf9b573b`, 04-26), cherry-picked into unification (`8220124e2`); **byte-identity proof**: the 4 `top_features.json` blobs + paper PNGs are blob-identical dev↔camera-ready | `023d52c24` (numbers), `fcf9b573b` (labels+plots) | **PINNED** (§6): topk_sae TopK→ReLU k=500/token; tsae_paper_k500/k20 ReLU→BatchTopK train, **ReLU→EMA-threshold at the shipped eval**; TXC arm = `agentic_txc_02` (matryoshka-contrastive multiscale) TopK→ReLU per-window k_win=500 | YES — txcdr-base (`<arch_id>__seed42.pt`; seed-42 only used) |
| Synthetic (c1/c2) | **REVISED:** runs produced on **`origin/final`'s purified v1 pipeline** (agents FILLER/HAMMER/SYNTH, 05-06/07, a40 pod), NOT on han-phase7-unification — but every arch module is a declared, line-identical port of `94119bc0` ⇒ composition lineage = han-phase7-unification | `579879786`, `b51dd7749`, `bcec4686d` (Setup B); `2660d2bae`.. + `f0315c8e9` (Setup D leaderboard snapshots) | **PINNED** (§7): topk/stacked TopK→ReLU; tsae ReLU→BatchTopK/threshold; tfa ReLU→TopK (`sae_diff_type="topk"`); txc_base TopK→ReLU k_win=k_pos·T; txc_pro k_train=k_pos·t_sample vs k_inference=k_pos·T_max | YES — temp-bench-models toy cells; `origin/final:purified/checkpoints/manifest.jsonl` = **5 713 rows** (the real manifest; final-aniket's is empty) |

---

## 3. Sparse probing (c3) — full detail

**What the paper reports** (`final-aniket:purified/docs/aniket/main.tex` §sec:sparse-probing + `appendix.tex` app:c3): gemma-2-2b-IT L13, 36-task SAEBench panel, k_feats {5,10,20,40,80,160,320,640}, 3 seeds (1/2/42), S=32 left-aligned tail, mean-pool, class-mean-diff top-S selection + L1 logistic; headline AUC-of-AUC: MLC 0.907 > TXC-base 0.899–0.902 > T-SAE/TXC-pro 0.897–0.899 > TopK SAE 0.886; "matched expected per-token L0 = 20".

**Where the runs actually are.** The dev branch (`han-phase7-unification`) carries the whole training+probing pipeline and committed results — but its committed rows stop at k_feat∈{5,20}, IT **seed 42 only**, S∈{10,20,32} (`experiments/phase7_unification/results/probing_results.jsonl`, 20 926 rows; `training_index.jsonl`, 163 rows; confirmed same-or-smaller on the only other branches carrying the file: `aniket-phase7-y` 6 182, `han-phase7-agent-c-seed1` 2 964). The dev IT family trained at **matched WINDOW budget k_win=500** (`it_topk_sae` k_pos=500\@T1, `it_txcdr_t5` 100\@T5, `it_..._t8` 62, `it_txcdr_t16` 31; `it_tsae_paper_k20` = the only k20 run), final_step 3 000–8 200 (plateau early-stop per `paper_archs.json` `training_constants`: b=4096, lr 3e-4, max 25 000, min 3 000). The paper's cells claim n_steps=20 000 at k_pos=20 → **the shipped c3 cells are a post-05-03 re-train** whose index/results were never committed to any surviving branch (A1). Corroboration of the re-train story: `paper_archs.json` (dev "single source of truth") still says `k_feat_reported: [5, 20]`; the purified c3 experiment dir on final-aniket is a TODO stub (`purified/experiments/c3_probing/README.md`: "Files (TODO — Agent NLP fills in)"); purified `checkpoints/manifest.jsonl` on final-aniket is **0 bytes**; `purified/src/temp_bench/architectures/` on final-aniket contains ONLY `__init__.py` + `base.py` — **no arch classes existed in purified at camera-ready**. Therefore whatever trained the shipped cells used the dev arch classes below (the only implementations in existence), and the composition verdict is venue-independent. Timing alignment: `temp-bench-models` last-modified 05-07 01:24Z; Aniket's fig commits 05-07 01:52 (`3d8bdef0a`), 05:01 (`bd8fbf6fc`), 07:30 (`3ee3ae61f`, "seed-based error bars (min/max across seeds 1/2/42)").

**Composition per arm — code quoted at `han-phase7-unification` (tip `94119bc08`); every file stable through the training window** (last-touch dates: topk_sae/mlc/crosscoder/_tfa_module 04-18 `0dbeeef29`; txc_bare_antidead 04-24 `63c2a9ece`; tsae_paper 04-26 `792dfe81d`; phase5b_subseq 05-01 `1995e0952` training-sampler only):

- `src/architectures/topk_sae.py:50-59` (TopKSAE):
  `pre = x_c @ W_enc.T + b_enc; topk_vals, topk_idx = pre.topk(self.k); z.scatter_(-1, topk_idx, F.relu(topk_vals))` → **TopK→ReLU per-token**; selection can pick negative pre-activations which ReLU then zeroes (realized L0 < k when negatives get selected — the paper-era mixing fingerprint on the per-token baseline).
- `src/architectures/txc_bare_antidead.py:159-167` (TXCBareAntidead = TXC-base):
  `pre = einsum("btd,tds->bs", x, W_enc) + b_enc` ("(B, T, d_in) -> (B, d_sae) pre-ReLU, pre-TopK"), then `vals, idx = pre.topk(self.k); z.scatter_(1, idx, F.relu(vals))` → **TopK→ReLU per-window**, k_win=k_pos·T. Anti-dead AuxK side path is ReLU→TopK over dead latents only (loss shaping, not the eval code path).
- `src/architectures/phase5b_subseq_sampling_txcdr.py:158-168` (+ matryoshka variant `:297-303`) (SubseqH8 = TXC-pro): probe-time `encode` = `encode_full` (ALL T_max positions, no sampling): same `einsum → topk → scatter(F.relu)` → **TopK→ReLU per-window**.
- `src/architectures/mlc.py:81-90` (MultiLayerCrosscoder): same pattern over the 5-layer stack → **TopK→ReLU per-layer-window**.
- `src/architectures/crosscoder.py:58-67` (TemporalCrosscoder, `txcdr_*` family — dev leaderboard family, in the shipped fig only if A1 resolves that way): same → **TopK→ReLU per-window**.
- `src/architectures/tsae_paper.py:148-165` (TemporalMatryoshkaBatchTopKSAE = T-SAE): `post_relu = F.relu((x − b_dec) @ W_enc + b_enc)`; then **train** (`use_threshold=False`): "flat BatchTopK over (B * d_sae)" — `flat.topk(int(self.k.item()) * x.size(0))`; **inference default** (`use_threshold=True`): `z = post_relu * (post_relu > self.threshold)`. The probing pipeline calls bare `model.encode` (`experiments/phase5_downstream_utility/probing/run_probing.py:141-178` `_encode_per_token(model.encode, …)`) → **the paper's T-SAE probing numbers went through the ReLU→threshold path, not TopK at all**. Corroborated by the paper itself: appendix app:c1-archs describes T-SAE as "a faithful port of the Ye et al. 2025 implementation: Matryoshka BatchTopK … and threshold-based inference."
- `src/architectures/_tfa_module.py` (TemporalSAE = TFA): module supports `sae_diff_type ∈ {relu, topk, batchtopk}`; the `topk` branch (`:222-227`) is `z_novel = F.relu(...); topk(z_novel)` → **ReLU→TopK on the novel code**. Which mode `tfa_big` actually used = A5.

**k-budget discrepancy (flagged, not resolved).** appendix app:c3-archs: "TopK SAE: k_win=20; T-SAE BatchTopK: 20·B·T; TXC-base(T): k_win=20·T; TXC-pro: k_train=100, k_inference=200; MLC(L=5): k_win=100" — i.e. per-token-L0=20-matched. The committed dev family is k_win=500-matched. The appendix RLHF section (line ~253) meanwhile says "TXC at T=5, k_win=500" — so **RLHF shipped dev-family (k=500) cells while c3 claims k=20 cells** (A1/A2).

---

## 4. EM (c6) — detail (part 1: what is pinned)

**Dmitry's line (`dmitry-em-repl`, tip `835ccf85f`).** EM numbers froze at **`de0c8ea8f`** (05-09; "neg6 rerun … headline numbers updated"; `git log --follow` on the headline figures shows exactly one commit; the only later commits touch a synthetic separation_scaling summary). Eval code lives in the EXTERNAL private repo `github.com/chainik1125/fra_proj`, branch `dmitry-em-repl` off `origin/nura/dev`, pinned at `88cefde` / `b0f4abe` (recorded in `plots/2026-05-08_phase0_fastpath/README.md` and `docs/dmitry/c6_em/2026-05-08_em_repl_finance_sports/phase1_results.md`). Arms: 3 FRA recipes + 5 additive arms — **all per-token TopK SAEs (k=64, d_sae=102 400), no TXC/MatTXC/TSAE anywhere**. Headline Δalign|coh≥70: medical Add-L24-ln1(published) 39.4±6.3; finance FRA QK→QK 21.9±1.7; sports FRA QK→QK 37.7±4.8. Composition: 4 fresh SAEs = sae-lens 6.43 `TopKTrainingSAEConfig` → **TopK→ReLU per-token, fixed k, same at train/eval** (corroborated against public SAELens v6.43.0 `topk_sae.py`: `topk(x) … topk_values.relu()`); Nura's published SAE (`Nura-J/Qwen2.5-14B_SAE_ln1.normalised`, `ae_200000.pt`) = **A3** (dictionary_learning naming suggests ReLU→TopK; unverified). Eval-time routing: QK→QK substitutes `decode(encode(act))` (encode fully load-bearing, and α=1.0 is NOT a no-op — carries recon error, quoted in `phase1_results.md`); additive/OV arms only read f_λ from the encode (delta-injection; additive α=1 verified byte-identical no-op). Checkpoints: `dmanningcoe/em-repl-2026-05-07` (4 fresh SAEs + runs) and `-05-08` (both private, Dmitry's account — flag for Dmitry, my token is Han's). LoRA organisms: `ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice`, `…_risky-financial-advice`, `…_extreme-sports`.

**The temp_bench-native c6 (on `origin/final`, tip `457888a3e`).** Completely different pipeline: `purified/experiments/c6_em/`, **finance LoRA only** (`…R1_0_1_0_finance_extended_train`), acts at L24 `resid_post` (`purified/src/temp_bench/data/nlp/qwen_em.py:197`), abbreviated Wang procedure (stages 1+4), Haiku 4.5 judge; arms `sae_arditi` (`sae_arditi.py:113-115`: `pre = F.relu(pre)` then `pre.topk(self.k)` → **ReLU→TopK per-token**, d_sae=32 768, k_pos=128, "matches dictionary_learning") vs `txc_base` (`txc_base.py:169-171` + `:87`: **TopK→ReLU per-window**, c6 override k_win=k_pos·T=25·5=125). Its 30K-step txc_base cells (seeds 1/2) on `qwen_2_5_14b_instruct_finance_l24_resid_post` are confirmed in `temp-bench-models` (train_keys `a0d4491420ec0b14`, `ad1243ed0f41db35`, 6.7 GB each); c6-style eval artifacts (judge_outputs, feature_selection, `phase1_unsteered.json`, curves with `success_at_coh` thresholds 1.5–2.5) sit in `temp-bench-data:runs/` (430 files, ts 05-05/06).

**The unresolved piece (A6, PENDING).** The camera-ready figs are `c6_em_{alignment_delta,detection_prauc}_7bmed.*` — **Qwen-2.5-7B-Instruct bad-MEDICAL** with an ARCH panel ("T-SAE performs best on both axes, while TXC variants underperform", detection PR-AUC at S=16, Wang-stage steering above coherence 70). That matches neither committed pipeline (14B medical/finance replication without TXC arms; 14B finance temp_bench). Follow-up subagent is tracing the 7bmed figures' producing commits now — second push.

**Also on record (branch sweep):** `origin/em-nanda` (tip 05-03, 189 unique commits) = a THIRD, parallel Wang-procedure EM-steering program (SAE-arditi vs TXC on Qwen-14B R1/R32 finance organisms; closed 8-cell table; bundle-null) whose 479/480 result blobs are on NO paper ref. Its relationship to the shipped c6 figs is part of A6.

---

## 5. Backtracking (c7) — detail (informational; Aniket's lane, hands off)

**Provenance.** The shipped headline (+1.574 peak Δgc, "~3× next-best arch") was produced on **`origin/aniket-ward-stage-b`** at tip **`a62175ee7`** (05-03; `results/ward_backtracking_txc/b3_math500_cut25/inducement_summary.csv` row `TXC,-12.0,…,1.5737…`), ported into `origin/final` via `purified/results/c7_backtracking/aniket_reference/cut25/ATTRIBUTION.md` ("source_branch: origin/aniket-ward-stage-b / source_commit: a62175ee7…; his hill-climbed TXC (k=16, T=6, window L0=96 — NOT our locked TXC-base/TXC-pro)"). Han's "mid-history of ward-stage-a" reconciles as: stage-b's first ~210 commits ARE stage-a's history (fork at `399c423a3`, 3 below stage-a's tip `e099f7ada`), and every composition class is blob-identical across both (frozen since **03-22**, `f01483bd4`, `temporal_crosscoders/models.py`). Stage-a proper contains only Stage-A **Difference-of-Means steering from raw residuals (no sparse coder)** (`2f8232806`) and the earlier Venhoff grid (`c93797a35`).

**Composition at `a62175ee7`** (config `experiments/ward_backtracking_txc/config.yaml`: Llama-3.1-8B L10, T=6, d_sae=16 384): `txc` = TemporalCrosscoder, `self.k = k * T` ("match stacked SAE's total L0"), `einsum → topk(pre) → scatter(F.relu)` → **TopK→ReLU per-window, k_win=16·6=96** (headline cell); `txc_h8` same family + AuxK/contrastive (k_win=96); `mlc` = LayerCrosscoder over layers 8–12 (k=32·5=160), same TopK→ReLU; `topk_sae` = **TopK→ReLU per-token, k=64** (hill-climbed; per-arch hill-climb broke the k32 matched default); `tsae` (shipped as "TSAE-paper") + `tfa` = `han_tsae/saeTemporal.py` `sae_diff_type='topk'`: `z_novel = F.relu(…); topk(z_novel)` → **ReLU→TopK per-token among positives, kval_topk=20 override** (realized-L0 fingerprint in-era: "L0: 88.6 → 104.5" vs nominal 120, `f04f903ff`). Labeling caveat quoted from `NEURIPS_PUSH.md`: the codebase's "tsae_paper does NOT implement [Bhalla's BatchTopK k=20]; both tsae and tsae_paper use Han's attention-based TemporalSAE" — shipped as 'TSAE-paper' with a documented caveat. A `batchtopk` mode (ReLU→flat-BatchTopK train / EMA-threshold JumpReLU eval) exists in the class but was **unused** here. Steering itself never encodes (mined **decoder column** injected by hook, rescaled to DoM norm); detection probes reuse mined window-mean acts. TRAIN=EVAL composition otherwise (same encode under no_grad, no threshold switch).

**Honest flag (A4).** `origin/final:purified/docs/components/c7.md` declares a LATER locked-arch rerun (agent_back, `2b44235e4`, 05-05: txc_base +0.426, txc_pro +0.377 — only a 1.13× win) the "paper-data candidate" and brands Aniket's a62175ee7 numbers "Wasteland reference (NOT paper data — context only)" — while the camera-ready c7 README stub says "TXC's expected ~3× win" (matches a62175ee7's 1.574/0.508≈3.1). Two contradictory declarations; the submitted PDF is the disambiguator.

**neurips-aniket cross-check:** `purified/experiments/backtracking_window_sweep/` trains `temp_bench.archs.txc_base.TXCBase` with `k_win = min(k_pos·T, d_sae)` and the identical TopK→ReLU per-window encode → the rebuttal harness **faithfully preserves the paper-era composition** (hyperparams differ: k_pos=20, d_sae=32 768).

---

## 6. RLHF (HH-RLHF preference decomposition) — full detail

**Provenance (CONFIRMED, refined).** The paper's RLHF table (T-SAE k20: 14/20 semantic, 63% mass; TopK-SAE & T-SAE k500: ~10-11/20, 50%; TXC: 7/20, 3 length-spurious) is **exactly** the Stage-1 headline of `han-phase7-unification:docs/han/research_logs/phase7_unification/2026-04-26-c1-hh-rlhf-stage1.md`, produced on **`origin/han-phase7-agent-c`** at `023d52c24` (04-26, cache + per-arch decomposition) + `fcf9b573b` (autointerp labels + scatter/summary plots), cherry-picked into unification at `8220124e2` (04-28). **Byte-identity chain:** all four `top_features.json` blobs are identical dev↔`temp-bench` release (e.g. agentic_txc_02 blob `12a873891a…`), and the paper PNGs (`arxiv:paper/figs/rlhf_*.png`) are blob-identical to `final-aniket:purified/docs/aniket/figs/rlhf_*.png`, re-rendered from that same data by `temp-bench:scripts/rlhf_paper_renderer.py`.

**Protocol:** Anthropic/hh-rlhf harmless-base first N=1000 pairs; gemma-2-2b (BASE) L12 residuals; mean over response tokens; rank by `mean_rejected − mean_chosen`; **seed-42 checkpoints only** (`ckpt_path = OUT_DIR/"ckpts"/f"{arch_id}__seed42.pt"`).

**Per-arm composition at `023d52c24`** (all four src files blob-identical at unification tip `94119bc08`; registry `canonical_archs.json`: d_in=2304, d_sae=18432, k_win_default=500):
- `topk_sae` (T=1, k_pos=500): **TopK→ReLU per-token** (`pre.topk(self.k)` then `scatter(F.relu(topk_vals))`).
- `tsae_paper_k500` / `tsae_paper_k20`: **ReLU first, always**; train = flat BatchTopK over (B·d_sae); **the shipped eval numbers went through `use_threshold=True`** — `encode_per_position` (`case_studies/_arch_utils.py`) calls `model.encode(sub, use_threshold=use_threshold)` with default True ⇒ **ReLU→EMA-threshold (JumpReLU-style)** at eval.
- **TXC arm = `agentic_txc_02`** = `MatryoshkaTXCDRContrastiveMultiscale` (T=5, **k_win=500** = 100·T, shifts [1,2,3], γ=0.5, n_scales=3) — encode inherited from `matryoshka_txcdr.py::PositionMatryoshkaTXCDR.encode` (~line 128): `einsum("btd,tds->bs") → topk(k_win) → scatter(F.relu)` ⇒ **TopK→ReLU per-window**. NOTE: the RLHF TXC arm is NOT `txc_bare_antidead` — it is the matryoshka-contrastive variant (matches the appendix's "TXC at T=5, k_win=500" admission).
- Eval sliding: window code attributed to the **right edge**; positions 0..T-2 zero — a different window-attribution convention than the synthetic Setup-B probes (broadcast+average). Pods copying "paper-match" aggregation must pick per-task.

**Checkpoints:** `han1823123123/txcdr-base` (+ `txcdr-base-data` act caches); local `results/ckpts/<arch_id>__seed42.pt`; HH-RLHF cache `data/cached_hh_rlhf/{chosen,rejected}.npz`. Raw separation_scaling data (RLHF-adjacent MatTXC study) survives only on `origin/dmitry-rlhf` (§9).

## 7. Synthetic (c1/c2: Denoising + Coupling) — full detail

**Provenance (REVISED vs the briefing's assumption).** The paper's Fig-2 files are blob-identical `final-aniket` ↔ `temp-bench`, rendered by `temp-bench:scripts/c2_paper_renderer.py` from: Setup B = `experiments/c1_noisy_filler/denoising_probe_results.json` (299 records); Setup D = the `toy_coupled_noisy_K10_M20_d256_pB05_*` leaderboard slice. Those inputs were **produced on `origin/final`'s purified v1 pipeline** by agents FILLER/HAMMER/SYNTH on 2026-05-06/07 (a40 pod; commits `579879786`, `b51dd7749`, `bcec4686d`, `2660d2bae`…, final verified snapshot `f0315c8e9`) — NOT by running phase2_toy/phase3_coupled on han-phase7-unification. Independent pin: `dmitry-synthetic:dmitry/pre_purified/c2_synthetic_paper_fig2/README.md` names both sources (incl. the a40 pod path). Numeric cross-checks reproduce the paper text (txc_base R²=0.483→"0.48"; np10 gAUC tsae/txc_base 0.99; topk k8 0.439→"0.44"). **Composition lineage is still han-phase7-unification:** every purified arch module carries an in-file "Ported from `origin/han-phase7-unification @ 94119bc0`" attribution and the selection logic is line-identical (verified for topk_sae/tsae/txc_base).

**Per-arm composition** (quoted from `origin/final:purified/src/temp_bench/architectures/`; c1/c2 override d_sae=40, k_pos swept {1..20}, seeds {1,2,42}; arch files blob-identical across all result commits):
- `topk_sae`: **TopK→ReLU per-token**. `stacked_sae`: T independent per-position SAEs, each **TopK→ReLU** (window L0 = k_pos·T).
- `tsae_paper` (as `tsae.py`): **ReLU→BatchTopK (flat over B·T·d_sae) at train; ReLU→EMA-threshold at eval** — eval branch active (`.eval()` + `threshold ≥ 0` after step 1000; runs are 8K–30K steps).
- `tfa`/`tfa_pos`: `TemporalSAE` with **`sae_diff_type="topk"`** ⇒ **ReLU→TopK per-token** on novel codes (this also resolves A5 for the c1/c2 venue).
- `txc_base`: **TopK→ReLU per-window, `k_win = k_pos·T`** (T=5 default; T swept 2–12 in Setup B).
- `txc_pro`: subseq encoder (T_max=10, t_sample=5); **k_train = k_pos·t_sample (train) vs k_inference = k_pos·T_max (eval)**, both TopK→ReLU per-window; matryoshka disabled at toy scale (h_size=40).

**Metric-path caveat:** Setup D's e/gAUC are **decoder-geometry only** (composition affects training dynamics, not metric computation); Setup B's probes DO encode (sliding T-window, **broadcast-to-all-T + average** overlap handling).

**Checkpoints/results:** `origin/final:purified/checkpoints/manifest.jsonl` = **5 713 rows** (the real registry — final-aniket's copy is 0 bytes; neurips-aniket's 189 rows are post-submission v2); 727 pB05 coupled cells, n_steps {8000: 567, 20000: 102, 30000: 53}; ckpts by train_key on `temp-bench-models`, caches on `temp-bench-data`; data snapshots on `temp-bench` + `dmitry-synthetic`.

---

## 8. HF inventory — han1823123123 (token: Han's datasets account; values never printed)

8 repos visible (5 model + 3 dataset; the briefing's "3 datasets" undercounted — **the checkpoint repos are model-type and include two unbriefed ones**):

| repo | type | vis | created→modified | size | contents / classification |
|---|---|---|---|---|---|
| `txcdr` | model | public | 04-21→04-26 | 232 GB, 193 ckpts | Phase-5/5b-era dump: `ckpts/` (agentic_txc/mlc ±batchtopk variants, conv_txcdr_t5..30, feature_nested_matryoshka…), `phase5b_ckpts/` |
| `txcdr-base` | model | public | 04-25→05-02 | 215 GB, 145 ckpts | **Phase-7 BASE-side canonical runs**, seeds 1/2/42 (`markers/seed{1,2,42}_complete.json`) + 145 training_logs; incl. `agentic_txc_02_kpos20__seed42.pt`, `phase57_..._t20_kpos100__*` |
| `txcdr-it` | model | public | 04-29→**04-30** | 12.3 GB, 12 ckpts | **The 12 IT-side probing ckpts** = exactly the dev `training_index.jsonl` it_ rows, seed 42 only (`markers/seed42_complete.json`) |
| `temp-bench-models` | model | public | 05-03→**05-07 01:24** | 256 GB, **1 283 train_key dirs** (config.json + model.safetensors each) | Purified v1.0.0 paper-cell store: 1 157 tiny toy cells (c1/c2), 42 dirs >2 GB real-LM cells (sampled: c6 qwen-finance txc_base 30K-step seeds 1/2). config keys: arch/arch_version/seed/datasource/train_key/act_cache_key/training_cfg/saved_ts |
| `temp_xc_a40_checkpoints` | model | **private** | **07-25** (yesterday) | 54.6 GB, 216 ckpts | CURRENT v2 task-hunt mirrors (`stage2_oprate_checkpoints/`, `stage2_fineweb_checkpoints/`, each with manifest.jsonl) — active-ops, not paper-era |
| `txcdr-data` | dataset | public | 04-21→04-25 | 99 GB | Phase-5 wasteland mirror: `experiments/phase5_downstream_utility/results/{probing_results,training_index}.jsonl` + caches under `experiments/`, `data/` |
| `txcdr-base-data` | dataset | public | 04-25→05-02 | 14.2 GB | `activation_cache/` (4 files) — phase-7 BASE act cache |
| `temp-bench-data` | dataset | public | 05-03→05-06 | 457 GB | `probe_cache/` 252 GB (SAEBench tasks × {base l11-15, …}), `act_cache/` 182 GB (2×70.8 GB 5-layer MLC caches, 2×14.2 GB anchor caches, 7.9 GB `resid_post_L24` = c6 Qwen, 4.2 GB `resid_post_L10` = c7 Llama), **`runs/` 430 purified eval artifacts** (metrics/curves/generations/grades/judge_outputs/feature_selection/phase1_unsteered; sampled ts 2026-05-05T21:13Z), `c7_backtracking/stage_a/sentence_acts_L10.npz`, 36 task dirs |

Plus (other accounts, from archaeology): `aniketdesh/ward-stage-b-dictionaries`, `aniketdesh/ward-stage-b-cache` (public, c7); `dmanningcoe/em-repl-2026-05-0{7,8}` (private, Dmitry); `Nura-J/Qwen2.5-14B_SAE_ln1.normalised` (published external SAE); `ModelOrganismsForEM/*` LoRAs; `han1823123123/txcdr-base` / `txcdr-it` referenced in dev `_paths{,_it}.py` as `HF_CKPT_REPO`.

**Checkpoint→task map:** probing IT seed-42 → txcdr-it (direct, run_id-named); probing BASE 3-seed → txcdr-base; shipped c3 k20/20K cells → expected in temp-bench-models (A1: enumerate its 1 283 config.jsons for `datasource=gemma_2_2b_it_l13_fineweb_24k128, n_steps=20000` — ~1 283 KB-scale downloads, cheap, public); c6 finance cells → temp-bench-models confirmed; c7 → aniketdesh repos; c1/c2 toy cells → temp-bench-models confirmed.

---

## 9. Forgotten-branch sweep — digest (full table in mac-c transcript)

All 43 remotes classified; blob-level containment vs the 7 known refs. **Result-bearing branches whose contents reached NO paper/consolidation ref:**

1. **`em-nanda`** (05-03, 189 uniq commits, 479/480 result blobs absent) — parallel Wang-procedure EM steering, SAE-arditi vs TXC, Qwen-14B R1/R32 finance; closed 8-cell table, bundle-null; `origin/dmitry` is a strict subset.
2. **`aniket-ward-stage-b`** (05-03) — see §5; 0/124 ward files on any consolidation ref (the cut25 CSVs were ported by copy; figures/analyses were not).
3. **`dmitry-backtracking`** (04-29, 61/95 absent) — second, independent Ward replication ("T-SAE Pareto-wins apples-to-apples at 30k"); never merged.
4. **`dmitry-rlhf`** (05-01, 86/139 absent) — RLHF protocol plots + the FULL separation_scaling 3-arch×3-seed dataset ("MatTXC wins on average"; seed=2 collapse case); only a 1-file summary survives on dmitry-em-repl.
5. **`han-phase6`** (04-26, 168/441 absent) — Phase-6.3 probing T-sweep ("T=20 Pareto-dominates T-SAE"), superseded by phase-7 re-runs but T-sweep data exists only here.
6. **`andre-steering`** (05-06, 731/778 absent) — largest unmerged real-LM bundle (autointerp, UMAP, safety eval); hygiene flag in-era: `9775cc688` "replace fabricated citations with real papers".
7. **`300k-tfa`** (05-07) — 48 final-night c1/c2 leaderboard rows in NO version of final's canonical leaderboard; 46 ckpt-registry config blobs absent.
8. Minor: `bill-benchmarking-synthetic` + `bill-han-txc-10k` (HMM synthetic figure sets), `dmitry-phase8` (sleeper-backdoor 45-cell study, "recovery metric reverses headline"), `han-runpod` (qualitative appendix, superseded), `dmitry-c6-redteam` (8 detection.json, 05-06), `dmitry-spectral-sprint2` (post-submission June freqbench).

Absorbed/contained (proof by rev-list + blobs): `han-phase7-agent-c{,-seed1}`, `han-phase5b`, `han_local`, `dmitry-synthetic` ⊂ dmitry-em-repl, `aniket` ⊂ dmitry-em-repl (NOT ⊂ stage-a), `det-steer`, `arxiv-aniket` ⊂ neurips-aniket, `temp-bench{,-anon}` (release cuts), `bill-three-arch-bench`/`wip/aliased-benchmark-runpod`/`temporal-bench` (code-only/minor), `han`/`han1`/`bill`/`andre{,_safety}` (subsets/minor).

---

## 10. AMBIGUOUS ledger (each with its disambiguator)

- **A1 — Where exactly the shipped c3 cells (k_pos=20, 20K steps, seeds 1/2/42, 8 k_feat budgets) were trained + which commit.** Not in any committed index/results; purified had no arch classes at camera-ready; dev family is k500. Composition is pinned regardless (§3) — this pin is about run identity/checkpoints. Disambiguators: enumerate `temp-bench-models` 1 283 config.jsons (public, KB-scale); `temp-bench-data:probe_cache/gemma_2_2b_it_*` meta.json; Aniket's fig-render inputs (ask Aniket / pod state).
- **A2 — Whether the c3 text's "matched per-token L0=20" holds for every shipped arm** (dev evidence shows a k500 era; appendix RLHF admits k_win=500 for its TXC). Disambiguator: same as A1 (config.jsons name k_pos per cell).
- **A3 — Nura's published SAE encode order** (ReLU→TopK vs TopK→ReLU; any threshold buffer). Disambiguators: `fra_proj/fra/sae_lens_wrapper.py:QwenLn1SAE` at fra_proj `dmitry-em-repl`; or state-dict keys of `ae_200000.pt` on `Nura-J/Qwen2.5-14B_SAE_ln1.normalised` (`threshold` buffer ⇒ dictionary_learning BatchTopK-thresholded; `k` tensor ⇒ AutoEncoderTopK).
- **A4 — Which backtracking numbers the submitted PDF printed** (a62175ee7 +1.574/3× vs final's locked rerun +0.426/1.13×). Disambiguator: the submitted TeX/PDF (check `arxiv:paper/main.tex` c7 macros — `purified/docs/aniket/figs/c7_*.tex` on final-aniket embed the values; 2-minute check, queued for second push).
- **A5 — `tfa_big`'s `sae_diff_type` at the PROBING venue** (module supports relu/topk/batchtopk; ReLU-first in all sparse modes). RESOLVED for c1/c2: purified `tfa.py` instantiates `sae_diff_type="topk"`. Still open for the phase-7 `tfa_big` runs. Disambiguators: phase-7 training wrapper instantiation on han-phase7-unification; or `it_tfa_big__seed42.pt` config/state-dict on txcdr-it.
- **A6 — Provenance of the camera-ready c6 7bmed figures** (Qwen-7B bad-med arch panel: which pipeline, which commits, which checkpoints; how em-nanda relates). Subagent running; second push.
- **A7 — f_λ sparsity in Dmitry's OV/additive hooks** (post-TopK sparse vs pre-TopK dense — changes realized steering coverage). Disambiguator: `fra/ov_steering.py` + `fra/sae_resid_eval.py` at fra_proj `b0f4abe`.
- (Minor) A8 — fresh-SAE `rescale_acts_by_decoder_norm`/aux-k flags (`dmanningcoe/em-repl-2026-05-07` cfg.json, private, Dmitry's token).
- **A9 — T-SAE EMA-threshold state in the shipped eval ckpts** (RLHF + probing both eval with `use_threshold=True`; if a ckpt's threshold buffer were < 0 the gate silently degenerates to plain ReLU). Runs were ≫1000 steps so the threshold was almost certainly armed. Disambiguator: read the `threshold` scalar in `tsae_paper_*__seed42.pt` on txcdr-base / `it_tsae_paper_*__seed42.pt` on txcdr-it (KB-scale header read).
- **A10 — synthetic aggregation choices in the shipped text**: paper's "TopK gAUC=0.92 at k_pos=1" matches the BEST seed (0.929; seed values 0.765–0.929, mean 0.842) — plausibly max-over-seeds in the c2 review aggregation (`final-aniket:purified/docs/aniket/c2_review_for_han.md` repeats 0.92; its generating script is not in the tree). ⚠ Team discipline note: if confirmed, this is a max-over-seeds cell in the shipped text — relevant to the no-max-over-arms rule when re-running. Disambiguator: the aggregation cell behind c2_review_for_han.md; likewise "peak gap ≈0.47" traces to `hunt_summary.json`/`HUNT_FINDINGS.md` (unverified row-by-row), and txc_pro "0.98" = best-T cell (T=2, not T=5).
- **A11 — Setup-D per-cell n_steps heterogeneity** (8K hunt / 20–30K zoom cells mixed in the shipped fig; paper doesn't state steps). Disambiguator: join `setup_d_leaderboard.jsonl` train_keys against `origin/final:purified/checkpoints/manifest.jsonl` training_cfg per plotted cell.

## 11. What this means for the paper-match arms (pods)

1. **There is no single "paper composition" — pin per arm.** Paper TXC arms (probing, c7, purified c6) = **TopK→ReLU, per-window, k_win=k_pos·T**. Paper per-token TopK-SAE baselines = **TopK→ReLU per-token**. Paper T-SAE = **ReLU-first** everywhere it appears, but with THREE different selection back-ends by venue: BatchTopK-train/threshold-eval (probing/RLHF `tsae_paper`), plain ReLU→TopK kval=20 (c7's attention TemporalSAE shipped as "TSAE-paper"), and — per the c7 NEURIPS_PUSH note — none of these is a faithful Bhalla BatchTopK at eval.
2. **Consequence for the ACTMIX arms:** `paper-match` for TXC/SAE arms = the mixed TopK→ReLU composition (matches the shared briefing's txc_base row, k_win=k_pos·T — note dev-era k_pos was 100, paper-claimed k_pos=20, A2). The paper's T-SAE comparator was ALREADY ReLU-first — i.e., structurally on the `btk-only` side of the fix. Any "paper-match vs btk-only" delta on the T-SAE arm is expected ~0 by construction; the informative deltas are on TXC/SAE arms. This matches the shared pre-registration ("per-token sae baseline improves MOST under btk-only") — and the archaeology now says the same asymmetry was baked into the PAPER's own cross-arch comparison.
3. **Eval-only paper-match is feasible for probing** (txcdr-it seed-42 ckpts + temp-bench-data probe caches are public); for EM wait on A6; for c7 the aniketdesh repos make it feasible in Aniket's lane.
4. **Realized-L0 bands are the era-independent fingerprint:** paper-era evidence already shows the ReLU-first shortfall (c7 tsae "L0: 88.6 → 104.5" vs nominal 120) and the TopK→ReLU selected-negative zeroing on the other arms — keep publishing numeric bands in every card.
5. **Faithful paper-match for the T-SAE arm means THRESHOLD inference, not BatchTopK:** both the shipped probing and RLHF evals routed `tsae_paper` through `use_threshold=True` (ReLU→EMA-threshold). A pod that re-evals T-SAE with BatchTopK at eval is NOT paper-match. (And c7's "TSAE-paper" is a third thing entirely — Han's attention TemporalSAE with ReLU→TopK kval=20, per its own NEURIPS_PUSH caveat.)
6. **Window-attribution conventions differ by task** (RLHF: right-edge; synthetic Setup-B probes: broadcast+average; probing: mean-pool over fully-real windows). Paper-match arms must copy the per-task aggregation, not a global one.

_Recorded-by: claude-fable-5 (mac-c). Sections 6, 7, and A4/A6 resolve in the second push._
