#!/usr/bin/env bash
# Setup KESTREL — seq_len scaling on Setup A.
#
# 2 NEW datasources (seq_len=32 + seq_len=128) × 6 archs × 3 seeds × 3 k_pos
# + 2 ds × txc_base × 7 T values × 3 seeds × 3 k_pos = 234 cells.
#
# Tests: does longer seq_len boost TXC's gAUC by giving more samples
# to pool, or do per-token archs scale equally?
set -e
cd /workspace/temp_xc/purified
mkdir -p logs

DATASOURCES=(
  "toy_coupled_K10_M20_d256_seq32_kestrel"
  "toy_coupled_K10_M20_d256_seq128_kestrel"
)
SEEDS=(1 2 42)
BASELINE_TUPLES=(
  "topk      topk_sae    "
  "tfa_pos   tfa_pos     "
  "tsae      tsae_paper  "
  "stk_T2    stacked_sae 2"
  "stk_T5    stacked_sae 5"
)
TXC_T_VALUES=(2 4 5 6 8 10 12)

shards=()
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

echo "[kestrel] total shards: ${#shards[@]}"

n_gpus=5
i=0
for shard in "${shards[@]}"; do
  gpu=$((i % n_gpus))
  IFS="|" read -r ds seed arch label t_arg kind <<< "$shard"
  ds_short="${ds#toy_coupled_K10_M20_d256_}"
  log="logs/hammer_kestrel_${label}_${ds_short}_seed${seed}.log"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    env AGENT_NAME=agent_hammer TEMP_BENCH_POD_MODE=persistent \
        OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1 \
    .venv/bin/python -m agents.agent_hammer.run_setup_ac_baselines \
      --datasource "${ds}" --arch "${arch}" --seed "${seed}" \
      --k-poses 1 2 3 --n-steps 8000 ${t_arg} \
    < /dev/null > "${log}" 2>&1
  i=$((i + 1))
done

echo "[kestrel] launched ${#shards[@]} shards across ${n_gpus} GPUs"
sleep 2
n_alive=$(pgrep -af "run_setup_" | grep -v grep | wc -l)
echo "[kestrel] alive python procs: ${n_alive}"
