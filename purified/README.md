# temp_xc/purified — paper-ready framework

This subtree is the **only** code/docs supporting the paper. Everything outside
`purified/` is "wasteland" — historical experiments, hill-climbed architectures,
and intermediate writeups that should be read for context but **never imported,
never edited**.

## Quick start

```bash
cd purified
uv sync               # builds purified/.venv from purified/pyproject.toml
.venv/bin/python -c "import temp_bench; print(temp_bench.__version__)"
```

On RunPod:

```bash
cd /workspace/temp_xc/purified
bash scripts/bootstrap_runpod.sh    # tokens + uv env (idempotent)
```

## Layout

```
src/temp_bench/        # the library (architectures, training, eval, case studies)
experiments/c{1..7}/   # one dir per paper component
results/               # all run outputs; leaderboard.jsonl is append-only
checkpoints/           # manifest of HF urls + local paths
docs/components/c{1..7}.md   # paper-section writeups (the source of truth)
docs/paper/            # paper draft, figures
agents/<name>/         # per-agent briefing + log
scripts/               # bootstrap, runpod helpers
```

## The seven paper components

| C | Subject | Lead arch | Hardware |
|---|---|---|---|
| C1 | Synthetic TopK sweep (NMSE/AUC) | TXC-base + TXC-pro vs Stacked-SAE / TFA | 5090 |
| C2 | Synthetic coupled features (gAUC) | TXC-base + TXC-pro at multiple T | 5090 |
| C3 | Sparse probing (Gemma-2-2b layer 13) | TXC-base + TXC-pro vs T-SAE / TopK-SAE | 1× H100 |
| C4 | Qualitative latents (Pareto) | shares C3 cache | 1× H100 |
| C5 | RLHF steering (coh × success) | TXC-base + TXC-pro vs T-SAE | 1× A40 |
| C6 | Emergent misalignment (Qwen14B-finance) | SAE-arditi vs TXC k=100 — **honest negative** | 1× H100 |
| C7 | Backtracking (Ward Stage B) | TXC-base + TXC-pro vs SAE / TFA / T-SAE / MLC | 1× A40 |

## The two TXC architectures

Locked in for **everything** in this paper — no per-component hill-climbing.

- **TXC-base** = `txc_bare_antidead_t5` — vanilla TopK temporal crosscoder
  (T=5, k_pos=k, k_win=k×T) + tsae_paper anti-dead stack (AuxK + dead-feature
  reset + unit-norm decoder + grad-orthogonalize + geometric-median b_dec init).
  No matryoshka, no contrastive.
- **TXC-pro** = `phase5b_subseq_h8` — subseq encoder (T_max=10, t_sample=5) +
  matryoshka H8 (8 nested groups) + multi-distance InfoNCE
  (shifts={1,2}, inverse-distance weighted), k_pos=20.

Sparsity (k_feat) and dictionary size (d_sae) are the only per-component
free parameters.

See `docs/paper/architecture.md` for the full description.

## Working agreement

Read [`PROTOCOL.md`](PROTOCOL.md) before making any change. The headlines:

- Writeups go in `docs/components/c{N}.md`, not in agent dirs.
- All run outputs go in `results/runs/<run_id>/`. Append one line to
  `results/leaderboard.jsonl` when you finish.
- Never read or import from `temp_xc/{src,experiments,docs}` (the wasteland).
- Use `purified/.venv` (`uv sync` from `purified/`), not the root venv.
