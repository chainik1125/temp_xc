---
author: Aniket Deshpande
date: 2026-04-28
tags:
  - in-progress
  - ward-backtracking
---

## Ward 2025 backtracking-direction replication

Two-stage replication of [arXiv 2507.12638](https://arxiv.org/abs/2507.12638) (Ward, Lin, Venhoff, Nanda). See `docs/aniket/experiments/ward_backtracking/plan.md`.

- **Stage A (this dir)**: Difference-of-Means steering vector from base Llama-3.1-8B activations induces backtracking in DeepSeek-R1-Distill-Llama-8B but not in the base. Pure replication. Target: half a day on a single A40.
- **Stage B (later, gated on Venhoff)**: same direction, but derived from a single TXC feature trained on base activations only.

### Stage A pipeline

```bash
# all run from repo root, single A40 pod
python -m experiments.ward_backtracking.seed_prompts        # 300 prompts → prompts.json
python -m experiments.ward_backtracking.generate_traces     # distill → traces.json
python -m experiments.ward_backtracking.label_sentences     # GPT-4o → sentence_labels.json
python -m experiments.ward_backtracking.collect_offsets     # per-offset acts → acts_*.npz
python -m experiments.ward_backtracking.derive_dom          # → dom_vectors.pt
python -m experiments.ward_backtracking.steer_eval          # → steering_results.json
python -m experiments.ward_backtracking.plot                # → plots/{fig3,fig4}.png
```

Or `bash experiments/ward_backtracking/run_all.sh` for the orchestrated path.

### Reused infra

- `src/bench/venhoff/generate_traces.py` — vLLM trace gen (drop-in)
- `src/bench/venhoff/judge_client.py` — `AnthropicJudge`, `OpenAIJudge` (rate-limited)
- `src/bench/venhoff/tokenization.py` — sentence splitter + char-to-token map
- `src/bench/venhoff/responses.py` — `extract_thinking_process`, BPE normalisation

Costs and pre-registered outcomes are in the plan doc.
