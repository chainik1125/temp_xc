# Backtracking case study

SAE-feature reproduction of Ward, Lin, Venhoff, Nanda 2025 — *"Reasoning-Finetuning Repurposes Latent Representations in Base Models"* (arXiv 2507.12638).

The paper derives a residual-stream steering vector for backtracking via plain Difference-of-Means at layer 10 of base Llama-3.1-8B, and shows it elicits backtracking ("Wait", "Hmm") in DeepSeek-R1-Distill-Llama-8B. Here we substitute the raw-DoM step with **SAE feature decomposition** and ask whether the same effect can be reproduced via a small set of interpretable SAE features.

See `docs/dmitry/case_studies/backtracking/{plan,brief}.md` for the full design.

## Run order

```bash
# Stage 0: 300 reasoning traces from the distilled model
TQDM_DISABLE=1 uv run python experiments/phase7_unification/case_studies/backtracking/build_reasoning_traces.py

# Stage 1: keyword labels + negative-offset (-13..-8) positive sets
TQDM_DISABLE=1 uv run python experiments/phase7_unification/case_studies/backtracking/label_backtracking.py

# Stage 2: cache L10 residual activations (bf16)
TQDM_DISABLE=1 uv run python experiments/phase7_unification/case_studies/backtracking/build_act_cache_backtracking.py

# Stage 3: feature-space DoM via fnlp/Llama-Scope L10R-8x
TQDM_DISABLE=1 uv run python experiments/phase7_unification/case_studies/backtracking/decompose_backtracking.py

# Stage 4: steering — raw DoM baseline, SAE-additive, SAE paper-clamp
TQDM_DISABLE=1 uv run python experiments/phase7_unification/case_studies/backtracking/intervene_backtracking.py

# Stage 5: keyword fraction + magnitude-sweep plot + top-feature bar
TQDM_DISABLE=1 uv run python experiments/phase7_unification/case_studies/backtracking/evaluate_backtracking.py
TQDM_DISABLE=1 uv run python experiments/phase7_unification/case_studies/backtracking/plot_backtracking.py
```

For a smoke test, pass `--n 20` to Stages 0–4.

Outputs land under `experiments/phase7_unification/results/case_studies/backtracking/`.
