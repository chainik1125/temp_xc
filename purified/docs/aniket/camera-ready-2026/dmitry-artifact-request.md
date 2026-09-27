# Small artifact request for Dmitry

Draft handoff only; no message was sent. Please export code and small numeric
files, not datasets, activation caches, dictionaries, model weights, or
optimizer state. These are the remaining gaps from the 2026-09-27 audit.

1. **Final Shamir/secret-sharing experiment.** Send the exact code commit or
   source/config snapshot and per-cell metrics behind the posted h=2,q=11,
   W={1,2,3,4,5,10} table, especially TXC k=5,W=10 accuracy 0.96. Include
   seed IDs, sparsity sweep, representation/probe split policy, and the
   episode-disjoint split receipt. The committed
   `results/v6_colored_sources/polynomial_clock_h2_q11.json` covers only
   W≤5 and its old runner splits sampled windows across common episodes;
   it cannot serve as the final run's provenance.
2. **Final Stacked SAE rebuttal rows.** Export only
   `*/new_leaderboard_rows.jsonl`, result/metrics/config JSONs, and manifests
   from private HF `dmanningcoe/stacked-sae-rebuttal-2026-07`. C7 final
   train key is `26e69fdc60452c27` (300K, seed42, 25 magnitudes, clean judge
   pass, peak delta_gc≈0.246 at −12); C6 is `8b8231508a1ce6e3`;
   RLHF is `ae17686fd3a23df2`. Include C3, untrained floors, and completed
   C6 steering. Keep failed-judge, wrong-T, and wrong-k first passes labeled
   separately. The current tools cannot access this private repository.
3. **Final C6 figure inputs.** Send the renderer and exact TopK replacement
   cell IDs/configs that made the `_topk_sae` PDFs. Export
   `stage4_frontier.json`, `wang_full_extended.json`, and detection
   `pr_auc.json` for the final cells. Historical paths are
   `local_data/c6_redteam/h100_em_4/sweep_outputs/c6_<key>/`,
   `local_data/c6_redteam/h100_em_4/extended_detection_S_all/c6_<key>/`,
   and `dmitry/pre_purified/c6_em_overnight/...`. TXC seed1/42 keys are
   `2016074933c41e7f` and `88a4ddf6819d8057`; seed2 aggregates are already
   recovered. Also include the final paper-v1 T=1,2,4,6 detection rows
   behind the posted EM window percentages if they are separate from these.
4. **Original C7 uncertainty inputs.** For the final seven 300K cells,
   export a compact per-question table of magnitude, genuine-backtracking
   count, coherence flag, and correctness/rescue labels. Full generated
   text is unnecessary for replots or paired bootstraps. Include the
   selected-magnitude rule and cohort IDs. The available Markdown has
   rounded point curves and exact count totals, but not enough to rebuild
   the original confidence intervals. Corrected TXC-base detection is
   already recovered; any completed corrected TXC-pro or multi-seed
   steering results would be additional artifacts, not assumed present.
5. **Exact qualitative figure slice, if you have it; otherwise Andre.**
   Export displayed tokens, selected feature IDs/explanations, the final
   32×32 sentence activation matrix, normalization/assignment rule, and
   checkpoint hashes for
   `sentence_mid_res_k100_T5_chain12345_exclusive.png`. A small JSON/NPZ is
   enough. The historical branch image has a different scale/hash.

The Stacked and Shamir items are the first priority. The others gate exact
reproduction of specific plots or error bars, rather than the broader
camera-ready cleanup.
