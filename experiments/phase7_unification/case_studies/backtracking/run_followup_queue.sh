#!/usr/bin/env bash
# Follow-up queue: more Llama-Scope variants that don't need new training.
# Designed to run after the main night queue (run_night_queue.sh) completes.
# Each variant writes to its own intervene<_suffix>/.

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
banner() { echo "=== FOLLOWUP_${1} ${2} $(stamp) ==="; }

# ─── 4_32X re-run (failed earlier, trace_ids.json now committed) ────────────
{
  banner 4_32X_RERUN START
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
  && banner 4_32X_RERUN DONE \
  || banner 4_32X_RERUN FAILED
}

# ─── 6: rank-by-ratio (different choice of "best feature") ──────────────────
{
  banner 6_RATIO START
  $PY -m $MOD.decompose_backtracking \
      --top-k 5 --rank-by ratio \
      --decompose-suffix ratio --force \
  && $PY -m $MOD.intervene_backtracking \
      --top-k 1 --modes raw_dom sae_additive \
      --decompose-suffix ratio --intervene-suffix ratio --force \
  && $PY -m $MOD.grade_coherence --intervene-suffix ratio \
  && $PY -m $MOD.evaluate_backtracking --intervene-suffix ratio \
  && $PY -m $MOD.plot_backtracking \
      --decompose-suffix ratio --intervene-suffix ratio --plots-suffix ratio \
  && banner 6_RATIO DONE \
  || banner 6_RATIO FAILED
}

# ─── 7: multi-position + sum_topk together (combines both temporal levers) ──
{
  banner 7_MULTIPOS_TOPK3 START
  $PY -m $MOD.intervene_backtracking \
      --top-k 3 --combine sum_topk --modes sae_additive \
      --steer-positions sentence_start --steer-k 6 \
      --intervene-suffix multipos_topk3 --force \
  && $PY -m $MOD.grade_coherence --intervene-suffix multipos_topk3 \
  && $PY -m $MOD.evaluate_backtracking --intervene-suffix multipos_topk3 \
  && $PY -m $MOD.plot_backtracking \
      --intervene-suffix multipos_topk3 --plots-suffix multipos_topk3 \
  && banner 7_MULTIPOS_TOPK3 DONE \
  || banner 7_MULTIPOS_TOPK3 FAILED
}

# ─── 8: multi-position with longer window (K=20) ────────────────────────────
{
  banner 8_MULTIPOS_K20 START
  $PY -m $MOD.intervene_backtracking \
      --top-k 1 --modes raw_dom sae_additive \
      --steer-positions sentence_start --steer-k 20 \
      --intervene-suffix multipos_k20 --force \
  && $PY -m $MOD.grade_coherence --intervene-suffix multipos_k20 \
  && $PY -m $MOD.evaluate_backtracking --intervene-suffix multipos_k20 \
  && $PY -m $MOD.plot_backtracking \
      --intervene-suffix multipos_k20 --plots-suffix multipos_k20 \
  && banner 8_MULTIPOS_K20 DONE \
  || banner 8_MULTIPOS_K20 FAILED
}

# ─── 9: 32x with sum_topk=3 (combine wider SAE + multi-feature) ─────────────
{
  banner 9_32X_TOPK3 START
  # Reuse decompose_32x from variant 4
  $PY -m $MOD.intervene_backtracking \
      --top-k 3 --combine sum_topk --modes sae_additive \
      --sae-release llama_scope_lxr_32x --sae-id l10r_32x \
      --decompose-suffix 32x --intervene-suffix 32x_topk3 --force \
  && $PY -m $MOD.grade_coherence --intervene-suffix 32x_topk3 \
  && $PY -m $MOD.evaluate_backtracking --intervene-suffix 32x_topk3 \
  && $PY -m $MOD.plot_backtracking \
      --decompose-suffix 32x --intervene-suffix 32x_topk3 --plots-suffix 32x_topk3 \
  && banner 9_32X_TOPK3 DONE \
  || banner 9_32X_TOPK3 FAILED
}

# ─── 10: model-diffing with NousResearch open Llama mirror ─────────────────
# (the original variant 5 in the night queue would have failed because
# meta-llama/Llama-3.1-8B is HF-gated; NousResearch/Meta-Llama-3.1-8B is
# the same weights without the gate.)
{
  banner 10_MODELDIFF_NOUS START
  $PY -m $MOD.build_act_cache_backtracking \
      --model llama-3.1-8b-nous --cache-suffix base_nous --force \
  && $PY -m $MOD.decompose_modeldiff \
      --top-k 5 --rank-by tstat --positions all \
      --base-cache-suffix base_nous \
      --decompose-suffix modeldiff_nous --force \
  && $PY -m $MOD.intervene_backtracking \
      --top-k 1 --modes raw_dom sae_additive \
      --decompose-suffix modeldiff_nous --intervene-suffix modeldiff_nous --force \
  && $PY -m $MOD.grade_coherence --intervene-suffix modeldiff_nous \
  && $PY -m $MOD.evaluate_backtracking --intervene-suffix modeldiff_nous \
  && $PY -m $MOD.plot_backtracking \
      --decompose-suffix modeldiff_nous --intervene-suffix modeldiff_nous --plots-suffix modeldiff_nous \
  && banner 10_MODELDIFF_NOUS DONE \
  || banner 10_MODELDIFF_NOUS FAILED
}

echo "=== ALL_FOLLOWUPS_FINISHED $(stamp) ==="
