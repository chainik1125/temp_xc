#!/usr/bin/env bash
# C2 ρ-sweep launcher — Effect 1 vs Effect 2 test ((decision 2026-05-06),
# response to the prior author's `origin/case-synthetic:docs/legacy/results/
# 3arch_3bench_summary.md` concern).
#
# REASSIGNED 2026-05-06 from agent_filler → agent_synth (8× 5090 pod).
# agent_filler is busy patching up C2 cleanup (T=12 high-k tail + HF
# backfill); agent_synth is the fresh pod tasked with the synthetic
# investigation.
#
# Mission scope: 3-arch headline trio at 4 new ρ values × 3 seeds × 2
# k_pos = 72 cells. ρ=0.7 is already in the leaderboard from prior
# work — DO NOT re-run.
#
# Decision rule:
# - gAUC roughly flat across ρ → Effect 1 (sample aggregation; weak
#   temporal claim).
# - gAUC grows with ρ → Effect 2 (temporal pattern detection; strong
#   claim defensible).
# - agent_paper makes the framing call after seeing the curve.

set -euo pipefail

cd "$(dirname "$0")/../.."

# Verify we're in the right place + agent identity is correct.
if [ "${AGENT_NAME:-}" != "agent_synth" ]; then
    echo "[run_rho_sweep] WARNING: AGENT_NAME=${AGENT_NAME:-<unset>}, expected 'agent_synth'." >&2
    echo "  Run \`source scripts/set_agent_env.sh agent_synth\` first." >&2
    exit 1
fi

mkdir -p logs

# Headline trio — 3 archs that matter for Effect 1 vs Effect 2:
#   - topk_sae default (per-token TopK; baseline)
#   - txc_base default (T=5 internal sampling; canonical TXC)
#   - txc_pro T=2 (the only TXC config that wins gAUC vs TopK at our k)
#
# Driver iterates ALL 7 ARCH_TS configurations when --archs txc_pro is
# passed — including T=5 and T=12. That's bonus (useful for a T × ρ
# heatmap if the maintainer wants to drill in later); not the headline.

declare -A ASSIGN=(
  [0]="topk_sae"
  [1]="txc_base"
  [2]="txc_pro"      # iterates T=2, T=5, T=12 internally
  # GPUs 3-7 idle — small sweep, no need to over-parallelize.
  # If you want maximum throughput, split each arch's seed/ρ across more
  # GPUs (see commented block at bottom).
)

for gpu in "${!ASSIGN[@]}"; do
    arch="${ASSIGN[$gpu]}"
    log="logs/c2_rho_sweep_gpu${gpu}_${arch}.log"
    echo "[run_rho_sweep] GPU ${gpu} → ${arch}"
    setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
        env TQDM_DISABLE=1 AGENT_NAME=agent_synth \
        .venv/bin/python -m experiments.c2_synthetic_coupled.run \
        --archs "${arch}" \
        --seeds 42 1 2 \
        --k-poses 1 5 \
        --rho-values 0.0 0.3 0.6 0.9 \
        --n-steps 30000 \
        > "${log}" 2>&1 &
    echo $! > "/tmp/p_c2_rho_${arch}"
done

echo "[run_rho_sweep] launched 3 parallel cells; PIDs in /tmp/p_c2_rho_*"
echo "[run_rho_sweep] tail -f logs/c2_rho_sweep_gpu*.log to monitor"
echo "[run_rho_sweep] wait for completion (~5-10 min on 5090)..."
wait
echo "[run_rho_sweep] all 3 archs complete"

# ── Optional: faster split (uses GPUs 0-7 if all are free) ──────────
# declare -A WIDE_ASSIGN=(
#   [0]="topk_sae"          # 12 cells (3 seeds × 2 k × 4 ρ)
#   [1]="txc_base"          # 12 cells (T=5 only since txc_base has 1 ARCH_TS entry)
#   [2]="txc_pro"           # 36 cells (3 T values × 3 seeds × 2 k × 4 ρ)
#   [3]="stacked_sae"       # bonus baselines if the maintainer wants
#   [4-7]="<spare>"
# )
