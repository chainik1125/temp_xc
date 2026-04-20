#!/bin/bash
# runpod_venhoff_launch.sh — single-entry launcher for the Venhoff
# reasoning-eval Phase 1a pipeline (and Phase 1b comparative sweep).
#
# Pipeline (all stages resumable; skip by artifact cache hit):
#   1. Generate reasoning traces (MMLU-Pro × DeepSeek-R1-Distill-Llama-8B)
#   2. Collect activations at layer 6 (Path 1 + Path 3)
#   3. Train small-k dictionaries (10k-step cap, plateau early-stop)
#   4. Annotate sentences (argmax cluster label)
#   5. Label clusters (Haiku 4.5)
#   6. Score taxonomy (accuracy, completeness, orthogonality, composite)
#   7. Judge-bridge drift check (GPT-4o on 100 sentences, 0.5/10 threshold)
#
# Pod requirements:
#   - H100 80GB SXM
#   - Root disk: 40 GB
#   - Persistent volume: 500 GB (bigger than SAEBench due to trace caches)
#   - HF token + ANTHROPIC_API_KEY + OPENAI_API_KEY set in .env
#
# Usage:
#   bash scripts/runpod_venhoff_launch.sh           # smoke (1k traces)
#   MODE=full bash scripts/runpod_venhoff_launch.sh # full Phase 1b (5k traces)
#
# Environment overrides:
#   MODE=smoke|full          smoke = 1k traces, full = 5k (default smoke)
#   LAYER=6                  which layer to probe (default 6 — locked Q2)
#   SEED=42                  rng seed (default 42)
#   ENGINE=vllm|transformers trace generation engine (default vllm)
#   JUDGE=claude-haiku-4-5-20251001   judge model
#   SKIP_BRIDGE=1            skip the GPT-4o drift bridge (saves $)
#   FORCE=1                  rebuild every stage ignoring caches
#   FORCE_STAGE=traces       rebuild a single stage (traces|activations|train|annotate|label|score|bridge)
#   SKIP_STAGE=bridge        skip a single stage
#   CLUSTER_SIZES="5 10 15 20 25 30 35 40 45 50"   full sweep override
#   ARCHES="sae mlc tempxc"  arch sweep override (full only)

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
[ -f scripts/runpod_activate.sh ] && source scripts/runpod_activate.sh

MODE="${MODE:-smoke}"
LAYER="${LAYER:-6}"
SEED="${SEED:-42}"
ENGINE="${ENGINE:-vllm}"
JUDGE="${JUDGE:-claude-haiku-4-5-20251001}"
SKIP_BRIDGE="${SKIP_BRIDGE:-0}"
FORCE="${FORCE:-0}"
FORCE_STAGE="${FORCE_STAGE:-}"
SKIP_STAGE="${SKIP_STAGE:-}"
MODEL="${MODEL:-deepseek-r1-distill-llama-8b}"
ROOT="${ROOT:-results/venhoff_eval}"

if [[ "$MODE" == "smoke" ]]; then
    N_TRACES="${N_TRACES:-1000}"
    CLUSTER_SIZES="${CLUSTER_SIZES:-15}"
    ARCHES="${ARCHES:-sae}"
    AGGREGATIONS="${AGGREGATIONS:-full_window}"
    N_LAYERS_MLC="${N_LAYERS_MLC:-5}"
elif [[ "$MODE" == "full" ]]; then
    N_TRACES="${N_TRACES:-5000}"
    CLUSTER_SIZES="${CLUSTER_SIZES:-5 10 15 20 25 30 35 40 45 50}"
    # MLC enabled via path_mlc (multi-layer collection). Re-added after
    # Han's 4/19 27-task sparse-probing result showed MLC ≈ TXCDR-T5 at
    # parity — Venhoff is now the MLC-vs-TempXC differentiation run,
    # not just a side-by-side.
    ARCHES="${ARCHES:-sae tempxc mlc}"
    AGGREGATIONS="${AGGREGATIONS:-last mean max full_window}"
    N_LAYERS_MLC="${N_LAYERS_MLC:-5}"
else
    echo "Unknown MODE=$MODE (expected: smoke|full)" >&2
    exit 2
fi

# Common flags propagated to every python module invocation.
COMMON_FLAGS=(
    --root "$ROOT"
    --model "$MODEL"
    --n-traces "$N_TRACES"
    --layer "$LAYER"
    --seed "$SEED"
)

python_flag_for_force_stage=()
if [[ -n "$FORCE_STAGE" ]]; then
    # Support space-separated list: FORCE_STAGE="label score"
    for stage in $FORCE_STAGE; do
        python_flag_for_force_stage+=(--force-stage "$stage")
    done
fi
python_flag_for_skip_stage=()
if [[ -n "$SKIP_STAGE" ]]; then
    for stage in $SKIP_STAGE; do
        python_flag_for_skip_stage+=(--skip-stage "$stage")
    done
fi
if [[ "$SKIP_BRIDGE" == "1" ]]; then
    python_flag_for_skip_stage+=(--skip-bridge)
fi
FORCE_FLAGS=()
if [[ "$FORCE" == "1" ]]; then
    FORCE_FLAGS+=(--force)
fi

echo "[info] venhoff_launch | mode=$MODE | model=$MODEL | layer=$LAYER | n_traces=$N_TRACES | seed=$SEED"
echo "[info] venhoff_grid | arches=$ARCHES | cluster_sizes=$CLUSTER_SIZES | aggregations=$AGGREGATIONS"
echo "[info] venhoff_judge | judge=$JUDGE | engine=$ENGINE"

# Phase 1a smoke is a single cell — use smoke.py (it wraps all stages).
if [[ "$MODE" == "smoke" ]]; then
    python -m src.bench.venhoff.smoke \
        "${COMMON_FLAGS[@]}" \
        --arch sae --cluster-size 15 --path path1 --aggregation full_window \
        --engine "$ENGINE" --judge-model "$JUDGE" \
        "${FORCE_FLAGS[@]}" \
        "${python_flag_for_force_stage[@]}" \
        "${python_flag_for_skip_stage[@]}"
    exit $?
fi

# Phase 1b full sweep: generate traces once, collect activations once,
# then fan out across (arch, cluster_size, path, aggregation) cells.
echo "[info] stage=generate_traces | status=start"
python -m src.bench.venhoff.generate_traces \
    "${COMMON_FLAGS[@]}" --engine "$ENGINE" \
    "${FORCE_FLAGS[@]}"

# Collect every path this run needs. If MLC is in ARCHES we also collect
# path_mlc (multi-hook forward over a window of layers around the anchor).
ACT_PATHS="path1 path3"
for _a in $ARCHES; do
    if [[ "$_a" == "mlc" ]]; then
        ACT_PATHS="$ACT_PATHS path_mlc"
        break
    fi
done
echo "[info] stage=collect_activations | status=start | paths=$ACT_PATHS | T=5 | n_layers_mlc=$N_LAYERS_MLC"
python -m src.bench.venhoff.activation_collection \
    "${COMMON_FLAGS[@]}" \
    --paths $ACT_PATHS --T 5 --n-layers "$N_LAYERS_MLC" \
    "${FORCE_FLAGS[@]}"

echo "[info] stage=fanout | status=start | steps=train+annotate+label+score"

# Pre-compute total cells for [sweep XX/YY] progress counter.
# path1 / path_mlc cells run a single aggregation (placeholder); path3
# (tempxc) gets |AGGREGATIONS|.
TOTAL_CELLS=0
for arch_ in $ARCHES; do
    for _cs in $CLUSTER_SIZES; do
        if [[ "$arch_" == "tempxc" ]]; then
            for _ in $AGGREGATIONS; do TOTAL_CELLS=$((TOTAL_CELLS + 1)); done
        else
            TOTAL_CELLS=$((TOTAL_CELLS + 1))
        fi
    done
done
CELL_COUNT=0
CELL_WIDTH=${#TOTAL_CELLS}

for arch in $ARCHES; do
    # Arch → path routing (Q1 lock-in + MLC extension post-4/19):
    #   sae    → path1      (per-sentence-mean)
    #   tempxc → path3      (T-window per sentence)
    #   mlc    → path_mlc   (multi-layer per-sentence-mean)
    case "$arch" in
        tempxc) path="path3"; aggs="$AGGREGATIONS" ;;
        mlc)    path="path_mlc"; aggs="full_window" ;;  # placeholder agg name
        *)      path="path1"; aggs="full_window" ;;
    esac

    for cluster_size in $CLUSTER_SIZES; do
        echo "[info] train_small_sae | arch=$arch | cluster_size=$cluster_size | path=$path"
        python -m src.bench.venhoff.train_small_sae \
            "${COMMON_FLAGS[@]}" \
            --arch "$arch" --cluster-size "$cluster_size" --path "$path" \
            --T 5 --n-layers "$N_LAYERS_MLC" \
            "${FORCE_FLAGS[@]}"

        for agg in $aggs; do
            CELL_COUNT=$((CELL_COUNT + 1))
            printf -v CELL_IDX "%0${CELL_WIDTH}d" "$CELL_COUNT"
            printf -v CELL_TOT "%0${CELL_WIDTH}d" "$TOTAL_CELLS"
            echo "[sweep $CELL_IDX/$CELL_TOT] arch=$arch | cluster_size=$cluster_size | path=$path | aggregation=$agg"
            python -m src.bench.venhoff.annotate \
                "${COMMON_FLAGS[@]}" \
                --arch "$arch" --cluster-size "$cluster_size" --path "$path" \
                --aggregation "$agg" \
                "${FORCE_FLAGS[@]}"

            # label + score go through smoke.py with --skip-stage for the
            # stages already run; it's the single entrypoint that knows
            # how to read assignments and emit cluster labels + scores.
            # Propagate FORCE / FORCE_STAGE so `FORCE_STAGE="label score"`
            # triggers per-cell rebuilds here too.
            python -m src.bench.venhoff.smoke \
                "${COMMON_FLAGS[@]}" \
                --arch "$arch" --cluster-size "$cluster_size" --path "$path" \
                --aggregation "$agg" \
                --skip-stage traces --skip-stage activations \
                --skip-stage train --skip-stage annotate \
                --skip-bridge \
                --judge-model "$JUDGE" \
                "${FORCE_FLAGS[@]}" \
                "${python_flag_for_force_stage[@]}"
        done
    done
done

if [[ "$SKIP_BRIDGE" != "1" ]]; then
    echo "[info] stage=bridge | status=start | cell=sae/k15/path1/full_window"
    # Bridge uses the smoke cell's labels as its input sample.
    python -m src.bench.venhoff.smoke \
        "${COMMON_FLAGS[@]}" \
        --arch sae --cluster-size 15 --path path1 --aggregation full_window \
        --skip-stage traces --skip-stage activations \
        --skip-stage train --skip-stage annotate \
        --skip-stage label --skip-stage score \
        --judge-model "$JUDGE"
fi

echo "[done] venhoff_launch | mode=$MODE | results_root=$ROOT"
