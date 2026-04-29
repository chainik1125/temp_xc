#!/usr/bin/env bash
# Night queue: 5 backtracking variants, each writing to its own intervene<_suffix>/.
# Designed to run unattended on an H100. Each variant is independent (`||` after
# the chain so a failure in one doesn't kill the rest).
#
# Run from the repo root:
#   bash experiments/phase7_unification/case_studies/backtracking/run_night_queue.sh

set -u
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/workspace/hf_cache}
export TMPDIR=${TMPDIR:-/workspace/tmp}
export TQDM_DISABLE=${TQDM_DISABLE:-1}
export PHASE7_REPO=${PHASE7_REPO:-$(pwd)}
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -f /workspace/.tokens/anthropic_key ]; then
  export ANTHROPIC_API_KEY=$(cat /workspace/.tokens/anthropic_key)
fi
PY=.venv/bin/python
MOD=experiments.phase7_unification.case_studies.backtracking

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
banner() { echo "=== VARIANT_${1} ${2} $(stamp) ==="; }

# ─── Variant 4 (lowest risk: same protocol, different SAE) ──────────────────
{
  banner 4_32X START
  $PY -m $MOD.decompose_backtracking \
      --top-k 5 --rank-by tstat \
      --sae-release llama_scope_lxr_32x --sae-id l10r_32x \
      --decompose-suffix 32x --force \
  && $PY -m $MOD.intervene_backtracking \
      --top-k 1 --modes raw_dom sae_additive \
      --sae-release llama_scope_lxr_32x --sae-id l10r_32x \
      --decompose-suffix 32x --intervene-suffix 32x --force \
  && $PY -m $MOD.grade_coherence --intervene-suffix 32x \
  && $PY -m $MOD.evaluate_backtracking --intervene-suffix 32x \
  && $PY -m $MOD.plot_backtracking \
      --decompose-suffix 32x --intervene-suffix 32x --plots-suffix 32x \
  && banner 4_32X DONE \
  || banner 4_32X FAILED
}

# ─── Variant 2 (low risk: only changes the steering direction) ──────────────
{
  banner 2_TOPK3 START
  $PY -m $MOD.intervene_backtracking \
      --top-k 3 --combine sum_topk --modes sae_additive \
      --intervene-suffix topk3 --force \
  && $PY -m $MOD.grade_coherence --intervene-suffix topk3 \
  && $PY -m $MOD.evaluate_backtracking --intervene-suffix topk3 \
  && $PY -m $MOD.plot_backtracking \
      --intervene-suffix topk3 --plots-suffix topk3 \
  && banner 2_TOPK3 DONE \
  || banner 2_TOPK3 FAILED
}

# ─── Variant 3 ──────────────────────────────────────────────────────────────
{
  banner 3_TOPK5 START
  $PY -m $MOD.intervene_backtracking \
      --top-k 5 --combine sum_topk --modes sae_additive \
      --intervene-suffix topk5 --force \
  && $PY -m $MOD.grade_coherence --intervene-suffix topk5 \
  && $PY -m $MOD.evaluate_backtracking --intervene-suffix topk5 \
  && $PY -m $MOD.plot_backtracking \
      --intervene-suffix topk5 --plots-suffix topk5 \
  && banner 3_TOPK5 DONE \
  || banner 3_TOPK5 FAILED
}

# ─── Variant 5 (moderate risk: new decompose, also needs base cache) ────────
{
  banner 5_MODELDIFF START
  $PY -m $MOD.build_act_cache_backtracking \
      --model llama-3.1-8b --cache-suffix base --force \
  && $PY -m $MOD.decompose_modeldiff \
      --top-k 5 --rank-by tstat --positions all \
      --decompose-suffix modeldiff --force \
  && $PY -m $MOD.intervene_backtracking \
      --top-k 1 --modes raw_dom sae_additive \
      --decompose-suffix modeldiff --intervene-suffix modeldiff --force \
  && $PY -m $MOD.grade_coherence --intervene-suffix modeldiff \
  && $PY -m $MOD.evaluate_backtracking --intervene-suffix modeldiff \
  && $PY -m $MOD.plot_backtracking \
      --decompose-suffix modeldiff --intervene-suffix modeldiff --plots-suffix modeldiff \
  && banner 5_MODELDIFF DONE \
  || banner 5_MODELDIFF FAILED
}

# ─── Variant 1 (highest risk: custom LogitsProcessor; runs last) ────────────
{
  banner 1_MULTIPOS START
  $PY -m $MOD.intervene_backtracking \
      --top-k 1 --modes raw_dom sae_additive \
      --steer-positions sentence_start --steer-k 6 \
      --intervene-suffix multipos --force \
  && $PY -m $MOD.grade_coherence --intervene-suffix multipos \
  && $PY -m $MOD.evaluate_backtracking --intervene-suffix multipos \
  && $PY -m $MOD.plot_backtracking \
      --intervene-suffix multipos --plots-suffix multipos \
  && banner 1_MULTIPOS DONE \
  || banner 1_MULTIPOS FAILED
}

echo "=== ALL_VARIANTS_FINISHED $(stamp) ==="
