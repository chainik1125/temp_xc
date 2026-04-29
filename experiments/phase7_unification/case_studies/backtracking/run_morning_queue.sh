#!/usr/bin/env bash
# Morning queue: priority-1 follow-ups for when the user comes back.
#
# (1) Re-run the headline comparisons with --held-out-stratified so the 80%
#     per-prompt success rate isn't a single-category artefact. Three variants:
#       main_strat (8x feat_7792 + raw_dom)
#       32x_strat  (32x feat_71839 + raw_dom)
#       ratio_strat (ratio-ranked feat_27749 + raw_dom)
#
# (2) Reminder: items 1-4 from the user's night-stretch list (TXC@ln1,
#     TXC@attn, transcoder, T-SAE) all need HF auth. To unblock:
#       huggingface-cli login   # paste a Meta-approved Llama-3.1-8B token
#     Then run build_llama_finetune_cache.py (committed) and the trainer
#     adapter (yet to be written, ~1h work).

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
banner() { echo "=== MORNING_${1} ${2} $(stamp) ==="; }

# ─── 1: 8x feat_7792 with stratified held-out ──────────────────────────────
{
  banner 1_MAIN_STRAT START
  $PY -m $MOD.intervene_backtracking \
      --top-k 1 --modes raw_dom sae_additive \
      --held-out-stratified \
      --intervene-suffix main_strat --force \
  && $PY -m $MOD.grade_coherence --intervene-suffix main_strat \
  && $PY -m $MOD.evaluate_backtracking --intervene-suffix main_strat \
  && $PY -m $MOD.plot_backtracking \
      --intervene-suffix main_strat --plots-suffix main_strat \
  && banner 1_MAIN_STRAT DONE \
  || banner 1_MAIN_STRAT FAILED
}

# ─── 2: 32x feat_71839 with stratified held-out ────────────────────────────
{
  banner 2_32X_STRAT START
  $PY -m $MOD.intervene_backtracking \
      --top-k 1 --modes raw_dom sae_additive \
      --sae-release llama_scope_lxr_32x --sae-id l10r_32x \
      --decompose-suffix 32x \
      --held-out-stratified \
      --intervene-suffix 32x_strat --force \
  && $PY -m $MOD.grade_coherence --intervene-suffix 32x_strat \
  && $PY -m $MOD.evaluate_backtracking --intervene-suffix 32x_strat \
  && $PY -m $MOD.plot_backtracking \
      --decompose-suffix 32x --intervene-suffix 32x_strat --plots-suffix 32x_strat \
  && banner 2_32X_STRAT DONE \
  || banner 2_32X_STRAT FAILED
}

# ─── 3: ratio-ranked feat_27749 with stratified held-out ───────────────────
{
  banner 3_RATIO_STRAT START
  $PY -m $MOD.intervene_backtracking \
      --top-k 1 --modes raw_dom sae_additive \
      --decompose-suffix ratio \
      --held-out-stratified \
      --intervene-suffix ratio_strat --force \
  && $PY -m $MOD.grade_coherence --intervene-suffix ratio_strat \
  && $PY -m $MOD.evaluate_backtracking --intervene-suffix ratio_strat \
  && $PY -m $MOD.plot_backtracking \
      --decompose-suffix ratio --intervene-suffix ratio_strat --plots-suffix ratio_strat \
  && banner 3_RATIO_STRAT DONE \
  || banner 3_RATIO_STRAT FAILED
}

echo "=== ALL_MORNING_FINISHED $(stamp) ==="
