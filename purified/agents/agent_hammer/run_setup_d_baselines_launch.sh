#!/usr/bin/env bash
# Setup D baseline gap-fill — adds tfa_pos, tsae_paper, stacked_sae T=2/T=5
# (and topk_sae where missing) on the 6 D-variants that don't have them
# yet. Reuses run_setup_h.py (same generator: coupled_noisy_hmm).
#
# Cells: ~207. ~12 min wall on 5 GPUs.
set -e
cd /workspace/temp_xc/purified
mkdir -p logs

# (datasource, missing_archs)
declare -A MISSING
MISSING[toy_coupled_noisy_K10_M20_d256_pB01_np5]="tfa_pos tsae_paper stacked_sae_T2 stacked_sae_T5"
MISSING[toy_coupled_noisy_K10_M20_d256_pB02_np8]="tfa_pos tsae_paper stacked_sae_T5"
MISSING[toy_coupled_noisy_K10_M20_d256_pB03_np5]="tfa_pos tsae_paper stacked_sae_T2 stacked_sae_T5"
MISSING[toy_coupled_noisy_K10_M20_d256_pB03_np8]="tfa_pos tsae_paper stacked_sae_T2 stacked_sae_T5"
MISSING[toy_coupled_noisy_K10_M20_d256_pB05_np2]="tfa_pos tsae_paper stacked_sae_T2 stacked_sae_T5"
MISSING[toy_coupled_noisy_K10_M20_d256_pB05_np8]="tfa_pos tsae_paper stacked_sae_T2 stacked_sae_T5"

SEEDS=(1 2 42)

shards=()
for ds in "${!MISSING[@]}"; do
  for arch_t in ${MISSING[$ds]}; do
    case "$arch_t" in
      stacked_sae_T2) arch=stacked_sae; T_arg="--T 2"; label=stk_T2 ;;
      stacked_sae_T5) arch=stacked_sae; T_arg="--T 5"; label=stk_T5 ;;
      *) arch="$arch_t"; T_arg=""; label="$arch" ;;
    esac
    for seed in "${SEEDS[@]}"; do
      shards+=("$ds|$seed|$arch|$label|$T_arg")
    done
  done
done

echo "[setup_d_fill] total shards: ${#shards[@]}"
n_gpus=5
i=0
for shard in "${shards[@]}"; do
  gpu=$((i % n_gpus))
  IFS="|" read -r ds seed arch label t_arg <<< "$shard"
  ds_short="${ds#toy_coupled_noisy_K10_M20_d256_}"
  log="logs/hammer_dfill_${label}_${ds_short}_seed${seed}.log"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    env AGENT_NAME=agent_hammer TEMP_BENCH_POD_MODE=persistent \
        OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1 \
    .venv/bin/python -m agents.agent_hammer.run_setup_h \
      --datasource "${ds}" --arch "${arch}" --seed "${seed}" \
      --k-poses 1 2 3 ${t_arg} \
    < /dev/null > "${log}" 2>&1
  i=$((i + 1))
done
echo "[setup_d_fill] launched ${#shards[@]} shards across ${n_gpus} GPUs"
sleep 2
n_alive=$(pgrep -af "run_setup_h" | grep -v grep | wc -l)
echo "[setup_d_fill] alive python procs: ${n_alive}"
