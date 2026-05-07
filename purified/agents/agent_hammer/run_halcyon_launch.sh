#!/usr/bin/env bash
# Setup HALCYON — d_in scaling on Setup A.
#
# 2 NEW datasources (d_in=128 + d_in=512) × 6 archs × 3 seeds × 3 k_pos
# + 2 datasources × txc_base × 7 T values × 3 seeds × 3 k_pos
# = 108 baseline cells + 126 T-sweep cells = 234 cells.
#
# Tests: does TXC's gAUC win persist as d_in scales? Per-token archs
# might recover features more easily at higher d_in (more orthogonal),
# narrowing TXC's lead. Or fail to scale — widening the lead.
#
# Random multi-letter name (HALCYON) avoids collision with agent_synth/
# agent_pro per Han's autonomous-work directive 2026-05-07.
set -e
cd /workspace/temp_xc/purified
mkdir -p logs

DATASOURCES=(
  "toy_coupled_K10_M20_d128_halcyon"
  "toy_coupled_K10_M20_d512_halcyon"
)
SEEDS=(1 2 42)

# Baselines
BASELINE_TUPLES=(
  "topk      topk_sae    "
  "tfa_pos   tfa_pos     "
  "tsae      tsae_paper  "
  "stk_T2    stacked_sae 2"
  "stk_T5    stacked_sae 5"
)

# txc_base T-sweep (7 T values)
TXC_T_VALUES=(2 4 5 6 8 10 12)

shards=()
# Baselines
for ds in "${DATASOURCES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for tuple in "${BASELINE_TUPLES[@]}"; do
      read -r label arch T <<< "$tuple"
      t_arg=""
      [[ -n "$T" ]] && t_arg="--T $T"
      shards+=("$ds|$seed|$arch|$label|$t_arg|baseline")
    done
    for T in "${TXC_T_VALUES[@]}"; do
      shards+=("$ds|$seed|txc_base|txc_T${T}|--T $T|tsweep")
    done
  done
done

echo "[halcyon] total shards: ${#shards[@]}"

n_gpus=5
i=0
for shard in "${shards[@]}"; do
  gpu=$((i % n_gpus))
  IFS="|" read -r ds seed arch label t_arg kind <<< "$shard"
  ds_short="${ds#toy_coupled_K10_M20_}"
  log="logs/hammer_halcyon_${label}_${ds_short}_seed${seed}.log"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    env AGENT_NAME=agent_hammer TEMP_BENCH_POD_MODE=persistent \
        OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1 \
    .venv/bin/python -m agents.agent_hammer.run_setup_ac_baselines \
      --datasource "${ds}" --arch "${arch}" --seed "${seed}" \
      --k-poses 1 2 3 --n-steps 8000 ${t_arg} \
    < /dev/null > "${log}" 2>&1
  i=$((i + 1))
done

echo "[halcyon] launched ${#shards[@]} shards across ${n_gpus} GPUs"
sleep 2
n_alive=$(pgrep -af "run_setup_ac_baselines" | grep -v grep | wc -l)
echo "[halcyon] alive python procs: ${n_alive}"
