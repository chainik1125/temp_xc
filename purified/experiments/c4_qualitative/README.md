# C4 — Qualitative latents (passage-discrimination Pareto)

Per-component scripts for the var-ranked / pdvar-ranked / paper-style
passage probe metrics on Gemma-2-2b-IT. See `docs/components/c4.md`.

## Files (TODO — Agent NLP fills in)

- `concat_data.py` — build N=3-7 contiguous-passage concatenations
- `rank_features.py` — top-32 per arch by var / pdvar
- `judge_haiku.py` — Haiku 4.5 SEMANTIC vs SYNTACTIC labelling
- `passage_probe.py` — k-sparse multinomial logreg, 5-fold CV
- `run.sh` — full pipeline; append SEMANTIC count + AUC to leaderboard
