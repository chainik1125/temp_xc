# Camera-ready figure source audit — 2026-09-27

Audited `paper/` at Overleaf commit `b27b3fc70f035b66cd18c4d301aa916b28ded1db` and the active outer checkout, plus local Git snapshots of `origin/final`, `origin/final-aniket`, `origin/extended-300k`, `origin/300k-tfa`, `origin/andre-steering`, `origin/dmitry-em-repl`, and `origin/dmitry-rlhf`. The root audit verified the remote branch heads. No model, activation cache, large dataset, or checkpoint was downloaded for this figure audit.

**All 24 active figure inputs resolve from the paper root**, including two TikZ sources. NeurIPS and ICML-ready use the same 24 named assets. This is an asset-presence check, not a successful paper compilation or proof that every plotted number has its original raw source. Most graphics can be redrawn on CPU; the exact final C6 extension and exact sentence heatmap have gaps. C7 has rounded historical point tables and a separate, stronger compact rebuttal bundle, but historical question-level uncertainty inputs remain incomplete.

The machine-readable recovery list is `figure-recovery-selection.json`: 45 branch paths totaling 11,427,710 bytes before any deduplication/filtering by the root recovery script. It names explicit small files; it does not authorize pulling their surrounding dataset/checkpoint directories. The root task's recovery manifest is authoritative about which paths were actually copied into `purified/artifacts/camera_ready_2026/`.

## Complete active figure inventory

Paths below are relative to `paper/`. `M` means main text, `A` means appendix. NeurIPS line numbers refer to the audited commit; the ICML-ready versions contain the corresponding same assets.

| Use | Source line | Figure input | Numeric/source status |
|---|---:|---|---|
| M | main_neurips.tex:172 | `images/temporal_tikz_both.tex` | Editable vector diagram; no experiment required. |
| M | main_neurips.tex:179 | `images/txc_tikz.tex` | Editable vector architecture; no experiment required. |
| M | main_neurips.tex:186 | `figs/global_rose_summary.pdf` | Raw sidecar and renderer recoverable, but source mixes outdated C7 and a different RLHF endpoint; update semantics before redrawing. |
| M | main_neurips.tex:512 | `figs/c2/c2_synth_global_headline.pdf` | Exact per-cell small JSON inputs and CPU plotting code available. |
| M | main_neurips.tex:518 | `figs/c2/c2_setup_b_singlelatent.pdf` | Same C2 bundle. |
| M | main_neurips.tex:524 | `figs/c2/c2_setup_d_scatter_clean.pdf` | Same C2 bundle. |
| M | main_neurips.tex:1105 | `figs/c3_sparse_probing_auc_of_auc_gemma_it.pdf` | Summary JSON and original per-seed/per-task leaderboard recoverable; uncertainty calculation needs correction. |
| M | main_neurips.tex:1112 | `figs/c3_sparse_probing_curves_gemma_it.pdf` | Exact aggregate curve values recoverable; currently excludes TFA for scale. |
| M | main_neurips.tex:1130 | `figs/c7_gc_at_peak_paired_compact.png` | Seven-cell historical mean values recoverable at 3 decimals; exact question-bootstrap intervals require original judges. |
| M | main_neurips.tex:1137 | `figs/c7_pr_auc_S8_bar_compact.png` | Historical rounded point values recoverable; corrected 300K three-seed TXC-base detection is already separately preserved. |
| M | main_neurips.tex:1158 | `figs/c6_em_alignment_delta_7bmed_topk_sae.pdf` | Final plotted asset exists; only earlier four-cell SAE-arditi/TXC-base frontiers directly recoverable. |
| M | main_neurips.tex:1165 | `figs/c6_em_detection_prauc_7bmed_topk_sae.pdf` | Final extension's exact detection files missing from audited Git snapshots; original script documents local-only inputs. |
| A | appendix.tex:163 | `figs/c3_per_task_heatmap.png` | Per-task AUCs recoverable from C3 leaderboard; no need to rerun probe fitting. |
| A | appendix.tex:175 | `figs/c3_sparse_probing_full_gemma_it.pdf` | Same C3 JSON; all architecture curves. |
| A | appendix.tex:212 | `figs/c6_em_pareto_frontier_7bmed_topk_sae.pdf` | Same incomplete final C6 extension source. |
| A | appendix.tex:219 | `figs/c6_em_steering_grid_7bmed_topk_sae.pdf` | Same incomplete final C6 extension source; recover alpha-level rows before restyling. |
| A | appendix.tex:238 | `figs/rlhf_summary.png` | Exact C8 top-feature JSON recovers all category counts and mass fractions. |
| A | appendix.tex:245 | `figs/rlhf_scatter.png` | Same C8 JSON; complete feature_stats files also recoverable for wider reanalysis. |
| A | appendix.tex:388 | `figs/c7_delta_gc_vs_magnitude.png` | Complete historical seven-cell 29-magnitude table at 3 decimals recoverable. |
| A | appendix.tex:462 | `figs/c7_probe_curves_combined.png` | Historical PR/ROC curves at S=1,2,4,8,16,32 recoverable at 3 decimals. |
| A | appendix.tex:581 | `figs/c7_net_saves_bar.png` | Exact integer rescue/regression counts in paper and recovered summary. |
| A | appendix.tex:588 | `figs/c7_contingency_stacked.png` | Exact integer coherence/backtracking counts in paper and recovered summary. |
| A | appendix.tex:620 | `figs/umap_txc.png` | Cached coordinates, cluster IDs, and cluster summary recoverable; CPU scatter redraw avoids embedding/LLM calls. |
| A | appendix.tex:627 | `figs/sentence_mid_res_k100_T5_chain12345_exclusive.png` | Final PNG exists; historical renderer and stats exist, but exact final selected activation matrix/token labels absent. |

The autofig helpers silently create placeholder boxes if a PNG is missing. For camera-ready validation, require every input to exist explicitly rather than treating a successful LaTeX exit as enough. Build `main_neurips.tex` or `ICML-ready/main.tex` from `paper/`, since both assume root-level `figs/` and `images/`.

## Per-experiment source map and cheapest valid redraw

### C2: synthetic Denoising and Coupling

Already in the paper checkout:

- `notes/c2_synthetic/data/denoising_probe_results.json` (126,856 bytes): architecture, T, k, seed, full-code clean-state probe R-squared, and single-latent correlation summaries.
- `notes/c2_synthetic/data/setup_d_leaderboard.jsonl` (593,320 bytes): Coupling per-cell results, including emission AUC, global AUC, and n_parents.
- `notes/c2_synthetic/data/hunt_summary.json` (12,567 bytes): auxiliary sweep summary, not needed by the three active panels.
- `scripts/make_c2_synth_panels.py`: the current combined three-panel renderer, importing only stdlib, NumPy, and Matplotlib. It loads the two files above, selects each architecture's best seed-mean cell, and calculates min/max seed bars. It writes PNGs; add explicit PDF export when implementing the overhaul.

Avoid `make_c2_global_headline_bars.py` as the canonical implementation: it still has hard-coded Coupling values. `make_c2_ratio_headline.py` is a preview with a different ratio framing. Use the combined renderer to preserve the actual headline metric. The original scripts use different architecture colors from C3/C6/C7.

Cheap redraw requires neither a synthetic rerun nor a model checkpoint. Preserve each row's configuration/seed and report the same best-cell selection policy; changing to a matched-config comparison is a new analysis and should be labeled as such. The current maximum-over-config presentation may exaggerate the apparent separation, so paired configuration curves would be a useful supplementary check from the same small numbers.

### C3: sparse probing

Recover `origin/final:purified/experiments/c3_probing/results.json` (14,838 bytes) and the base-model counterpart (12,696 bytes). The active IT figures are generated by `paper/scripts/make_c3_sparse_probing_figures.py`, which uses only stdlib/NumPy/Matplotlib but hard-codes Dmitry's machine paths. Point `REPO_DATA` to the outer repository, and write outputs to a new build directory. Its `git show` paths already include `purified/`, so `REPO_DATA` should be the outer Git root.

Recover `origin/final:purified/results/leaderboard.jsonl` (6,809,327 bytes) for all per-task/per-seed `auc__*` values and `origin/final:purified/experiments/c3_probing/analysis.py` for the canonical selection policy. The leaderboard contains 503 non-smoke C3 protocol-1.1.0 rows, including more configurations than the displayed canonical set. Do not average all rows indiscriminately: use model, architecture, train_key, seed, protocol, and probe budget. The 38-task heatmap can be redrawn directly from these numeric fields once that cell selection is frozen.

Two issues need resolution before camera-ready plots:

1. The text calls the headline SAEBench-36, while the saved plotting JSON records `n_tasks: 38`. This is a numeric mismatch, not only stale metadata: matching the three seed-level leaderboard cells exactly reproduces the stored 38-task k=5 means for MLC, TopK SAE, T-SAE, TXC-pro, and TFA. For example, the stored MLC k=5 value is 0.8530871684939231; dropping the two cross-token tasks from those same cells gives 0.8735662260709226. The analysis source explicitly excludes `winogrande_correct_completion` and `wsc_coreference`, but the paper plotting script reads the precomputed JSON rather than recomputing this filter. Choose whether the headline should remain 38 tasks or adopt the declared 36, then regenerate all means/captions together. Do not change only the task-count metadata.
2. `auc_of_auc()` propagates per-k seed standard deviations as if different feature-budget points were independent. The same trained dictionary and examples are reused across k, so those errors are correlated. Integrate each seed's curve first, then take the seed standard deviation or display individual seed points. The stored per-seed rows make this a cheap CPU correction.

As a diagnostic only, recomputing the same uniquely matched seed/train-key sets across all eight probe budgets after dropping those two tasks yields integrated mean ± sample seed SD: MLC 0.938305 ± 0.000515, TopK SAE 0.919091 ± 0.001054, T-SAE 0.931177 ± 0.002520, TXC-pro 0.931278 ± 0.004546, TFA 0.780423 ± 0.033937. These are a separate 36-task reanalysis, not edited manuscript numbers. The matching sets are MLC `{c4bad817b40f45ac,f07bcad7d9f197d2,c5b18a75a0db4994}`, TopK `{05363678579ff7cf,7e3c6d83e985d4f5,fe7feb76c9e510ae}`, T-SAE `{e8f3355683e0a25f,8f717f87f3f9464a,06053869c2b7e72b}`, TXC-pro `{1b029e5d45c45611,6af5d868f65c4a6c,4da1b28fc5032194}`, TFA `{8ff472709e89083f,0679d79278d95663,61da0670ea629ca4}`; each set is ordered seeds 1, 2, 42. The TXC-base T-sweep still needs the corresponding key selection audited before a full replacement table is published.

### C6: emergent misalignment

The existing final PDFs are present, but the original exact numeric sources are not complete in Git. `paper/scripts/make_c6_em_figures.py` reads canonical `origin/final` stage-4 frontiers for SAE-arditi and TXC-base, then tries local-only dense-alpha extensions and additional architectures from:

- `local_data/c6_redteam/h100_em_4/sweep_outputs/c6_<train_key>/wang_full_extended.json`
- `dmitry/pre_purified/c6_em_overnight/runs/c6_<train_key>/stage4_frontier.json`
- `dmitry/pre_purified/c6_em_overnight/sweep_outputs/c6_<train_key>/wang_full_extended.json`
- `local_data/c6_redteam/h100_em_4/extended_detection_S_all/c6_<train_key>/pr_auc.json`

Canonical four-cell keys are SAE-arditi `{9b011dfeea88f8af,c0da3ed8794554a1}` and TXC-base `{2016074933c41e7f,88a4ddf6819d8057}`; each recoverable frontier is about 15KB. Additional script keys are TXC-pro `{e561456612fe29ff,0689047ce9bce927}`, T-SAE `{6f6d047132771676,819604d52a131b54}`, and TFA `{e3b029548824f240}`. TFA only has seed 1 in this script. The current paper refers to `_topk_sae` PDFs, while this script's cell list still contains SAE-arditi; the exact final TopK replacement recipe is not defined by this script.

Ask the data owner for just the stage4/dense-alpha/pr_auc JSONs, the final TopK cell IDs/configs, and the renderer that made `_topk_sae` figures. These are small plotting artifacts. No Qwen checkpoint, FineWeb corpus, LoRA, or activation cache is needed to restyle existing frontiers. Request original judge outputs only if recomputing uncertainty or judging is needed. Older frontiers and the original rose sidecar are useful provenance, but they are not valid replacements for the final extension.

The root task additionally recovered the compact HF rebuttal bundle under `purified/artifacts/camera_ready_2026/hf_reviewer_results/reviewer_seed_audit_2026-07-27/medical_em/`. Its `steering_three_seed_summary.json` preserves verified TXC-base alignment ranges for seeds 1/2/42: 16.71875, 20.359375, 22.875, respectively, plus their extrema/provenance. The bundle includes the seed-2 stage-4 and dense-alpha rows and three published detection evaluations. This closes the three-seed TXC summary plotting gap, but does not recover every final multi-architecture `_topk_sae` curve or the seed-1/42 dense-alpha rows.

### C7: submitted 300K backtracking versus rebuttal replication

`origin/final-aniket:purified/docs/components/c7_paper_results.md` (8,742 bytes) preserves the complete seven-cell 300K comparison: baseline/peak genuine-backtracking means, all 29 magnitude points, PR/ROC values for six budgets, seed 42, batch sizes, train_keys, and eval_keys. `c7_optimal_analysis.md` (2,209 bytes) preserves exact rescue/regression and coherence/backtracking count tables. These support cheap redraws of point curves and exact count graphics. Most continuous values are rounded to 0.001; do not portray them as full-precision raw data.

Example submitted keys: TopK `40d84ac2cdef`, T-SAE `32f27809cdf3`, MLC `668047e2f66d`, TXC-base bs256 `6ae8db21b1bb`, TXC-base bs1024 `8787f8fe5272`, TXC-pro bs256 `4bf2edb49487`, TXC-pro bs1024 `6f3a9461c0dd`. The historical renderers in `origin/extended-300k:purified/scripts/{c7_paper_renderer.py,c7_tex_snippets.py}` and `experiments/c7_backtracking/analyze_optimal.py` explain the plotting and counting recipes. They depend on older package path interfaces; use them as reference or port their plotting functions, rather than assuming they run against today's `temp_bench` API.

Missing historical artifacts include per-question `judge_outputs.jsonl`, optimal generations and `coherence_judge.jsonl`, and training `snapshots/eval_log.jsonl` for the seven final cells. Without these, exact bootstrap confidence intervals on the old peak bars cannot be regenerated, and the exploratory convergence plots cannot be rebuilt from their PNGs. A 30K checkpoint on `extended-300k`/`300k-tfa` is not evidence that the 300K checkpoint exists.

The active repository separately preserves the corrected 300K replication at `purified/results/neurips_rebuttal/backtracking_300k_seeded/`, especially `publication/{raw_detection_metrics.csv,summary_detection_metrics.csv,summary.json,reviewer_summary.md}` and per-cell `detection.json`. The CPU plot entry point is `purified/experiments/backtracking_300k_seeded/plot_detection.py`. Use this bundle for the updated detection plot and keep the submitted seed-42 TXC-pro result labeled as a single-seed reference. The 20K window sweep is a different training regime and should receive its own panel/caption, not be pooled with 300K seeds.

### C8: HH-RLHF preference decomposition

Exact plot numbers survive on `origin/dmitry-rlhf` under:

`experiments/phase7_unification/results/case_studies/hh_rlhf/{topk_sae,tsae_paper_k500,tsae_paper_k20,agentic_txc_02}/{top_features.json,feature_stats.json}`.

The four top-feature files total approximately 142KB and directly regenerate the current summary, scatter, and top-five label table. The four feature_stats files total approximately 3.8MB and support re-ranking/rechecking all features. Verified from those JSONs:

| Architecture | Semantic / mixed / length-spurious count among top 20 | Semantic mass share |
|---|---|---:|
| TopK SAE k500 | 10 / 10 / 0 | 0.4950224538 |
| T-SAE k500 | 11 / 8 / 1 | 0.5033081479 |
| T-SAE k20 | 14 / 6 / 0 | 0.6279448251 |
| TXC matryoshka T5 | 7 / 10 / 3 | 0.3028620948 |

These reproduce the existing figure's rounded 50%, 50%, 63%, 30%. The original `experiments/phase7_unification/case_studies/hh_rlhf/summarize_hh_rlhf.py` reads cached JSON and uses NumPy/Matplotlib plus local path/save helpers; its imports do not invoke a model or LLM API. Recover the two `_paths.py` helpers and `src/plotting/save_figure.py`, or port its two plotting functions to a standalone renderer. The feature-labeling/cache-building scripts are provenance only; running them would be unnecessary expense for a visual overhaul.

The existing labels call low length correlation “semantic.” For camera-ready captions, say “low length correlation” or explain explicitly that this is a heuristic tier, since lack of correlation with one nuisance variable is not a direct test of semantic content.

### UMAP, sentence heatmap, and architecture drawings

Recover `origin/andre-steering:safety_research/results/umap_meta/<arm>/{coords.npy,labels.npy,summary.json}`. For TXC alone these total 103,033 bytes and contain 5,033 coordinates, cluster IDs, and the cluster-name mapping. The paper's 15 cluster counts match the available summary. The script `safety_research/scripts/umap_meta.py` includes embedding/UMAP/LLM machinery and hard-coded paths; **do not run the entire pipeline for recoloring**. A standalone NumPy/Matplotlib scatter from cached coordinates preserves the geometry without embeddings or inference. The `tsae` disk arm is labeled StackedSAE in the script, not the paper's T-SAE baseline.

The current paper UMAP PNG and the branch PNG have different Git blob hashes, so matching cluster metadata is strong provenance evidence rather than a byte-identical reproduction claim. The final sentence PNG also differs from the branch image; the branch's chain12345 stats report max TXC activation 834.21, while the final image's displayed scale extends to 1096. Recover its renderer `temporal_crosscoders/NLP/sentence.py`, `config.py`, `fast_models.py`, and stats for reference, but obtain the actual final selected feature values before redrawing.

The cheapest sentence export request is a small `.npz` or JSON containing the displayed token strings/IDs, selected feature IDs, final 32-by-32 activation matrix, feature explanations, normalization rule, window-to-token assignment, and checkpoint hashes. This is kilobytes, not an activation dataset. If those values were not saved, their owner can load existing checkpoints and one cached sentence on the machine where they remain; the original script expects `mid_res.npy`, `token_ids.npy`, and SAE/TXC checkpoints. Downloading those entire inputs here is unnecessary.

Both architecture drawings are TikZ under `paper/images/`. Edit them as vector source and compile locally; no image generation or data retrieval is required.

## Visual overhaul proposal

Use one architecture palette in every quantitative figure: TXC-base `#0072B2`, TXC-pro `#56B4E9`, T-SAE `#E69F00`, TFA `#CC79A7`, MLC `#009E73`, TopK SAE `#4D4D4D`, Stacked SAE `#999999`. Pair every architecture with a stable marker/line style; TXC-base and TXC-pro need distinct shapes because they share a blue family. Encode T, batch size, or seed with line style/shape/facets instead of inventing new architecture colors. Use a single sequential colormap such as cividis for activation/AUC heatmaps. For signed effects, use a diverging scale centered exactly at zero. C8 length-correlation categories and C7 coherence categories should have their own explicitly labeled semantic palette rather than borrowing architecture colors.

1. Replace the min-max rose chart with a compact benchmark map or aligned dot plots in natural units. Its recoverable sidecar uses older C7 peak lift (TXC-base 0.426 rather than the submitted 0.541), and its RLHF axis is C5 steering success rather than the C8 preference decomposition described in the paper. The min-max scale also makes small gaps look like categorical failures and excludes MLC from the architecture set. Redesigning this panel fixes a substantive presentation problem.
2. Make the central architecture diagram compare the same window under a per-token SAE, pooled/stacked SAE, and TXC. Show exactly which encoder weights are position-specific, the shared latent, reconstruction to each position, and k_pos versus k_win. Draw TXC-pro additions in a separate optional box so they do not obscure the base architecture.
3. Add one benchmark/label diagram showing a time-local transition label versus a broad rollout/topic label, plus the exact pre-onset window. This makes the conditions for temporal benefit understandable and connects the new rebuttal controls to the measured endpoint.
4. For C2, separate clean-state R-squared and alignment AUC into aligned subpanels rather than making the shared 0–1 range imply equal metrics. Keep the y=x reference in both scatters, lighter points for configurations, and seed points/intervals for selected cells.
5. For C3 and updated C7 detection, use a common legend, exact budget labels, and uncertainty from per-seed curves. If showing a zoom, also show the full context in a separate panel instead of omitting a baseline without a visible cue.
6. For C7 steering, pair induced backtracking with coherent-backtracking rate and net rescue counts. The current peak-only bars can visually reward incoherent continuations; the existing exact contingency and rescue counts make the caveat cheap to display. Use horizontal labels or facets to remove the dense rotated architecture/batch labels.
7. For C6, use small multiples with one shared coherence threshold and explicit seed markers once exact alpha rows are recovered. Keep the negative TXC result visible; uncertainty/seed count should be consistent with the actual available cells.
8. Replace the 32-row sentence heatmap plus tiny truncated explanations with 4–6 selected features, readable token text, short explanation labels, and a clear activation legend. Put the full matrix in the appendix or supplement. Use the UMAP as qualitative context, not as evidence of well-separated semantic clusters; its recorded silhouette is approximately 0.0098.

Export PDF/SVG for line art and plots, plus PNG previews. Set font sizes at final single/two-column dimensions, remove redundant titles, use a shared ordering, and store a small machine-readable sidecar for every final figure with source path/hash, cell/config/seed selection, metric, and uncertainty definition. That sidecar makes future recoloring independent of expensive training or judge calls.

## Verified CPU recreation and starter previews

`purified/scripts/camera_ready/replot.py` and its adjacent `palette.json` now produce three PNG/SVG preview pairs from saved numeric inputs only: corrected 300K backtracking detection, the three-seed 20K window/order sweep, and medical TXC-base steering seeds. Run `python3 purified/scripts/camera_ready/replot.py` from any directory. Outputs live in `purified/artifacts/camera_ready_2026/previews/`, together with `plot_source_values.csv` (78 numeric rows with source paths and protocols) and `replot_manifest.json` (input/script hashes, validation values, runtime, and explicit comparison limitations).

The verified final preview render took **0.373 seconds** after imports on this machine. PNGs were visually inspected after fixing an initial legend margin issue. The 300K plot recomputes and asserts the saved S=8 mean and sample SD; historical T-SAE has no invented uncertainty. The 20K sweep uses a different panel and is not pooled with the 300K replication. The medical preview shows just TXC-base seed values, with mean 19.984375 and sample SD 3.095210; no missing baseline error bars were manufactured.

The original C2 and C3 paper renderers also completed a CPU smoke redraw in **1.40 seconds** with output paths redirected in memory to `/tmp/txc-camera-ready-figure-smoke/`. This produced all three C2 PNGs and three C3 PDF/PNG pairs without modifying manuscript sources or committed figures. C3 input-path overrides were the only functional changes; its existing uncertainty calculation was reproduced for the smoke check and still needs the correction described above. A cold Matplotlib font-cache scan stalled on this host's `fc-list`; the run completed using a temporary copy of the existing Matplotlib font cache. No model, dataset, training, or API call was involved.
