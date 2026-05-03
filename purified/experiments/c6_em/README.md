# C6 — Emergent misalignment (Wang procedure on Qwen-14B)

Per-component scripts for the EM case study on Qwen2.5-14B-Instruct +
LoRA finance organism. See `docs/components/c6.md`.

## Files (TODO — Agent EM fills in, coordinating with Dmitry)

- `wang_procedure.py` — 4 stages (Δz̄ rank → causal screen → strength sweep → α frontier)
- `judge_gemini.py` — 8-prompt × 8-rollout Gemini judge
- `bundle.py` — k=30 bundle steering (replicate the bundle-null result)
- `run.sh` — full pipeline; expects HF auth + ModelOrganismsForEM access

## Honest negative

Paper claim is that **SAE arditi beats TXC at every cell**, with the
arch gap widening to +12.58 align in R32 ext-α. The interpretive
contribution is the **architecture-general bundle null**.
