# C1 — Synthetic TopK sweep

Per-component scripts for the toy NMSE/AUC sweep over k. See
`docs/components/c1.md` for the experimental setup and hypothesis.

## Files (TODO)

- `data.py` — Markov-chain support generator (Phase 2 Scheme C, $\rho$ levels)
- `train.py` — train one (arch, k, seed) cell
- `eval.py` — NMSE + feature-recovery AUC + (optional) novel/total L0 for TFA
- `run.py` — orchestrate the full sweep, append to leaderboard
- `plot.py` — produce paper figures from leaderboard rows

## Smoke-test command

```bash
cd purified && TQDM_DISABLE=1 .venv/bin/python -m experiments.c1_synthetic_topk.run \
    --arch txc_base --k 2 --seeds 42 --steps 1000  # <60 sec on 5090
```

## Full sweep

```bash
cd purified && TQDM_DISABLE=1 .venv/bin/python -m experiments.c1_synthetic_topk.run \
    --arch topk_sae stacked_sae tfa tfa_pos txc_base txc_pro \
    --k 1 2 3 4 5 6 8 10 12 15 17 20 \
    --seeds 1 2 42 \
    --steps 30000
```
