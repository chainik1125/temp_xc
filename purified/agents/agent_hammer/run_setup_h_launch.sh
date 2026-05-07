#!/usr/bin/env bash
# Setup H — ρ-sweep on D-np10 (max-overlap noisy regime).
#
# 3 NEW ρ values × 6 archs × 3 seeds × 3 k_pos = 162 cells.
# (ρ=0.9 = existing D-np10; agent_synth will fill its baselines on
# their D-np10 backfill task.)
#
# Archs (matching the 4-plot standard, minus txc_base T-sweep — that's
# a follow-up if Setup H proves interesting):
#   - topk_sae       (per-token baseline)
#   - tfa_pos        (per-token attention baseline)
#   - tsae_paper     (T=2 paper-faithful temporal SAE)
#   - stacked_sae T=2
#   - stacked_sae T=5
#   - txc_base T=5   (canonical TXC)
#
# Layout: 54 shards (= 3 ρ × 6 archs × 3 seeds, each iterating 3 k_pos
# values via the driver). Round-robin onto 5 GPUs.
set -e
cd /workspace/temp_xc/purified
mkdir -p logs

DATASOURCES=(
  "toy_coupled_noisy_K10_M20_d256_pB05_np10_rho00"
  "toy_coupled_noisy_K10_M20_d256_pB05_np10_rho03"
  "toy_coupled_noisy_K10_M20_d256_pB05_np10_rho06"
)
SEEDS=(1 2 42)

# (label, --arch, --T)
ARCH_TUPLES=(
  "topk        topk_sae      "
  "tfa_pos     tfa_pos       "
  "tsae        tsae_paper    "
  "stk_T2      stacked_sae   2"
  "stk_T5      stacked_sae   5"
  "txc_T5      txc_base      5"
)

shards=()
for ds in "${DATASOURCES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for tuple in "${ARCH_TUPLES[@]}"; do
      read -r label arch T <<< "$tuple"
      t_arg=""
      [[ -n "$T" ]] && t_arg="--T $T"
      shards+=("$ds|$seed|$arch|$label|$t_arg")
    done
  done
done

echo "[setup_h] total shards: ${#shards[@]} (= 3 ρ × 3 seeds × 6 archs)"

n_gpus=5
i=0
for shard in "${shards[@]}"; do
  gpu=$((i % n_gpus))
  IFS="|" read -r ds seed arch label t_arg <<< "$shard"
  rho_tag="${ds##*_}"  # e.g. "rho00"
  log="logs/hammer_h_${label}_${rho_tag}_seed${seed}.log"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    env AGENT_NAME=agent_hammer TEMP_BENCH_POD_MODE=persistent \
        OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1 \
    .venv/bin/python -m agents.agent_hammer.run_setup_h \
      --datasource "${ds}" --arch "${arch}" --seed "${seed}" \
      --k-poses 1 2 3 ${t_arg} \
    < /dev/null > "${log}" 2>&1
  i=$((i + 1))
done

echo "[setup_h] launched ${#shards[@]} shards across ${n_gpus} GPUs"
sleep 3
n_alive=$(pgrep -af "run_setup_h" | grep -v grep | wc -l)
echo "[setup_h] alive python procs: ${n_alive}"
