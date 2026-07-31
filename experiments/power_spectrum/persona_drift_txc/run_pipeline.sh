#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_ROOT/../../.." && pwd)"
readonly PHASE="${1:-all}"
readonly ARTIFACT_ROOT="${PERSONA_DRIFT_ARTIFACT_ROOT:-$SCRIPT_ROOT/artifacts}"
readonly RESULT_ROOT="${PERSONA_DRIFT_RESULT_ROOT:-$SCRIPT_ROOT/results}"
readonly REFERENCE_ROOT="${ASSISTANT_AXIS_ROOT:-/workspace/assistant-axis}"
readonly REFERENCE_COMMIT="a98961956072224eaf244eb289d6c01700b63795"
readonly REFERENCE_LOCK_SHA256="438b7d11359eb3a2dae997101da56737dc52d9197d79dc95ce91a8e39a66748a"
readonly SCRIPT_DATA="$REPO_ROOT/experiments/power_spectrum/persona_drift_txc/data/user_scripts.jsonl"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT:$REFERENCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ensure_reference() {
  if [[ ! -d "$REFERENCE_ROOT/.git" ]]; then
    git clone https://github.com/safety-research/assistant-axis.git "$REFERENCE_ROOT"
  fi
  git -C "$REFERENCE_ROOT" fetch origin "$REFERENCE_COMMIT"
  git -C "$REFERENCE_ROOT" checkout --detach "$REFERENCE_COMMIT"
  test "$(git -C "$REFERENCE_ROOT" rev-parse HEAD)" = "$REFERENCE_COMMIT"
  test "$(openssl dgst -sha256 -r "$REFERENCE_ROOT/uv.lock" | cut -d' ' -f1)" = \
    "$REFERENCE_LOCK_SHA256"
}

run_reference_python() {
  local python="${PERSONA_DRIFT_REFERENCE_PYTHON:-$REFERENCE_ROOT/.venv/bin/python}"
  test -x "$python" || {
    echo "missing pinned Assistant-Axis interpreter: $python" >&2
    return 1
  }
  "$python" - <<'PY'
from importlib.metadata import version
expected = {"transformers": "4.57.5", "torch": "2.9.0"}
actual = {name: version(name).split("+", 1)[0] for name in expected}
if actual != expected:
    raise SystemExit(f"Assistant-Axis dependency mismatch: {actual} != {expected}")
PY
  "$python" "$@"
}

run_reference() {
  run_reference_python \
    -m experiments.power_spectrum.persona_drift_txc.collect_activations \
    reference \
    --reference-root "$REFERENCE_ROOT" \
    --output-root "$ARTIFACT_ROOT/activations"
}

run_collect() {
  run_generate
  run_extract
  run_pack
}

run_generate() {
  local -a arguments=(
    -m experiments.power_spectrum.persona_drift_txc.collect_activations
    generate-vllm
    --scripts "$SCRIPT_DATA"
    --conversations "$ARTIFACT_ROOT/qwen_conversations.jsonl"
    --output-root "$ARTIFACT_ROOT/activations"
    --generation-batch-size "${PERSONA_DRIFT_GENERATION_BATCH_SIZE:-50}"
  )
  local python="${PERSONA_DRIFT_VLLM_PYTHON:-${PERSONA_DRIFT_REFERENCE_PYTHON:-$REFERENCE_ROOT/.venv/bin/python}}"
  test -x "$python" || {
    echo "missing pinned Assistant-Axis/vLLM interpreter: $python" >&2
    return 1
  }
  "$python" - <<'PY'
from importlib.metadata import version
expected = {"transformers": "4.57.5", "torch": "2.9.0", "vllm": "0.13.0"}
actual = {name: version(name).split("+", 1)[0] for name in expected}
if actual != expected:
    raise SystemExit(f"Assistant-Axis/vLLM dependency mismatch: {actual} != {expected}")
PY
  "$python" "${arguments[@]}"
}

run_extract() {
  run_reference_python \
    -m experiments.power_spectrum.persona_drift_txc.collect_activations \
    extract \
    --scripts "$SCRIPT_DATA" \
    --conversations "$ARTIFACT_ROOT/qwen_conversations.jsonl" \
    --output-root "$ARTIFACT_ROOT/activations" \
    --batch-size "${PERSONA_DRIFT_EXTRACTION_BATCH_SIZE:-1}" \
    --max-length "${PERSONA_DRIFT_MAX_LENGTH:-4096}"
}

run_pack() {
  uv run python -m experiments.power_spectrum.persona_drift_txc.collect_activations \
    pack \
    --conversations "$ARTIFACT_ROOT/qwen_conversations.jsonl" \
    --output-root "$ARTIFACT_ROOT/activations"
}

run_collect_smoke() {
  local smoke_root="$ARTIFACT_ROOT/smoke"
  run_reference_python \
    -m experiments.power_spectrum.persona_drift_txc.collect_activations \
    collect \
    --scripts "$SCRIPT_DATA" \
    --conversations "$smoke_root/qwen_conversations.jsonl" \
    --output-root "$smoke_root/activations" \
    --batch-size 1 \
    --max-length 4096 \
    --limit 1
}

run_embed() {
  uv run python -m experiments.power_spectrum.persona_drift_txc.embed_messages \
    --metadata "$ARTIFACT_ROOT/activations/metadata.jsonl" \
    --output "$ARTIFACT_ROOT/user_embeddings.pt" \
    --batch-size "${PERSONA_DRIFT_EMBED_BATCH_SIZE:-32}"
}

run_train() {
  uv run python -m experiments.power_spectrum.persona_drift_txc.train_representations \
    --activations "$ARTIFACT_ROOT/activations/turn_activations.pt" \
    --metadata "$ARTIFACT_ROOT/activations/metadata.jsonl" \
    --output-root "$ARTIFACT_ROOT/representations"
}

run_probe() {
  uv run python -m experiments.power_spectrum.persona_drift_txc.probe_future_drift \
    --activations "$ARTIFACT_ROOT/activations/turn_activations.pt" \
    --metadata "$ARTIFACT_ROOT/activations/metadata.jsonl" \
    --embeddings "$ARTIFACT_ROOT/user_embeddings.pt" \
    --representations "$ARTIFACT_ROOT/representations" \
    --output-root "$RESULT_ROOT"
}

ensure_reference
cd "$REPO_ROOT"
mkdir -p "$ARTIFACT_ROOT" "$RESULT_ROOT"

case "$PHASE" in
  reference)
    run_reference
    ;;
  collect-smoke)
    run_collect_smoke
    ;;
  collect)
    run_collect
    ;;
  generate)
    run_generate
    ;;
  extract)
    run_extract
    run_pack
    ;;
  embed)
    run_embed
    ;;
  train)
    run_train
    ;;
  probe)
    run_probe
    ;;
  all)
    run_reference
    run_collect
    run_embed
    run_train
    run_probe
    ;;
  *)
    echo "unknown phase: $PHASE" >&2
    exit 2
    ;;
esac
