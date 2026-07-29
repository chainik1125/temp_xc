#!/usr/bin/env bash
set -euo pipefail

E0_ROOT=/workspace/txc-neurips-aniket
E0_PURIFIED="$E0_ROOT/purified"
E0_ENV="$E0_ROOT/.venv-e0-extract"
E0_LOG_DIR="$E0_PURIFIED/logs/neurips_theory"
E0_LOG="$E0_LOG_DIR/e0_smoke.log"
E0_CACHE="$E0_ROOT/data/e0/smoke"
E0_RESULTS="$E0_PURIFIED/results/neurips_theory/e0_smoke"
E0_MODEL_REV=607a30d783dfa663caf39e06633721c8d4cfcd7e
E0_DATASET_REV=b08601e04326c79dfdd32d625aee71d232d685c3
E0_SOURCE_REV=77a8e70ada0511ca696b83048e90547dd37db428

mkdir -p "$E0_LOG_DIR" "$E0_CACHE" "$E0_RESULTS" /tmp/e0-mpl /tmp/e0-cache
exec > >(tee -a "$E0_LOG") 2>&1
trap 'E0_STATUS=$?; printf "%s\n" "$E0_STATUS" > "$E0_LOG_DIR/e0_smoke.exit"; exit "$E0_STATUS"' EXIT

printf 'E0 smoke start: %s\n' "$(date -u +%FT%TZ)"
printf 'public source revision: %s\n' "$E0_SOURCE_REV"
printf 'model revision: %s\n' "$E0_MODEL_REV"
printf 'dataset revision: %s\n' "$E0_DATASET_REV"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

if [[ ! -x "$E0_ENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$E0_ENV"
fi
if [[ ! -f "$E0_RESULTS/extraction_packages.txt" ]]; then
  "$E0_ENV/bin/python" -m pip install --upgrade pip
  "$E0_ENV/bin/python" -m pip install \
    'numpy==1.26.4' \
    'transformer-lens==2.15.4' \
    'transformers==4.43.4' \
    'datasets==5.0.0' \
    'huggingface-hub==0.36.2'
  "$E0_ENV/bin/python" -m pip freeze | LC_ALL=C sort > "$E0_RESULTS/extraction_packages.txt"
fi
"$E0_ENV/bin/python" -c "import importlib.metadata as m; expected={'numpy':'1.26.4','transformer-lens':'2.15.4','transformers':'4.43.4','datasets':'5.0.0','huggingface-hub':'0.36.2'}; actual={k:m.version(k) for k in expected}; assert actual == expected, (actual, expected); print(actual)"

"$E0_ENV/bin/python" -c "from huggingface_hub import HfApi; a=HfApi(); assert a.model_info('openai-community/gpt2', revision='$E0_MODEL_REV').sha == '$E0_MODEL_REV'; assert a.dataset_info('Salesforce/wikitext', revision='$E0_DATASET_REV').sha == '$E0_DATASET_REV'; print('pinned HF revisions resolve')"

cd "$E0_PURIFIED"
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/test_correlation_audit.py tests/test_correlation_robustness.py

CUDA_VISIBLE_DEVICES=0 "$E0_ENV/bin/python" \
  -m experiments.correlation_audit.extract_legacy \
  --output-dir "$E0_CACHE" \
  --model-revision "$E0_MODEL_REV" \
  --dataset-revision "$E0_DATASET_REV" \
  --source-revision "$E0_SOURCE_REV" \
  --sequence-length 256 \
  --num-sequences 34 \
  --layers 6 8 \
  --device cuda:0 \
  --batch-size 8 \
  --shard-size 34

.venv/bin/python -c "import pathlib,torch; p=pathlib.Path('$E0_CACHE'); s=torch.load(p/'shard_0000.pt',map_location='cpu',weights_only=False); assert sorted(s['residuals']) == [6,8]; assert tuple(s['tokens'].shape)==(34,256); assert all(tuple(x.shape)==(34,256,768) for x in s['residuals'].values()); assert len(s['article_ids'])==34; assert s['article_ids'].unique().numel()==2; print({'article_ids':s['article_ids'].tolist(),'unique_articles':int(s['article_ids'].unique().numel()),'layers':sorted(s['residuals']),'shapes':{k:list(v.shape) for k,v in s['residuals'].items()}})"

CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/e0-mpl XDG_CACHE_HOME=/tmp/e0-cache \
  PYTHONPATH=. .venv/bin/python -m experiments.correlation_audit.robustness \
  --cache-dir "$E0_CACHE" \
  --layer 6 \
  --output-dir "$E0_RESULTS" \
  --projection-dim 8 \
  --fit-tokens 1000 \
  --fit-documents 34 \
  --max-lag 12 \
  --persistent-rank 2 \
  --psd-bootstrap 20 \
  --device cuda:0

printf 'E0 smoke complete: %s\n' "$(date -u +%FT%TZ)"
