#!/usr/bin/env bash
set -euo pipefail

E0_ROOT=/workspace/txc-neurips-aniket
E0_PURIFIED="$E0_ROOT/purified"
E0_ENV="$E0_ROOT/.venv-e0-extract"
E0_LOG_DIR="$E0_PURIFIED/logs/neurips_theory"
E0_LOG="$E0_LOG_DIR/e0_full.log"
E0_CACHE="$E0_ROOT/data/e0/full"
E0_RESULTS="$E0_PURIFIED/results/neurips_theory/e0"
E0_MODEL_REV=607a30d783dfa663caf39e06633721c8d4cfcd7e
E0_DATASET_REV=b08601e04326c79dfdd32d625aee71d232d685c3
E0_SOURCE_REV=77a8e70ada0511ca696b83048e90547dd37db428

mkdir -p "$E0_LOG_DIR" "$E0_CACHE" "$E0_RESULTS" /tmp/e0-mpl /tmp/e0-cache
exec > >(tee -a "$E0_LOG") 2>&1
trap 'E0_STATUS=$?; printf "%s\n" "$E0_STATUS" > "$E0_LOG_DIR/e0_full.exit"; exit "$E0_STATUS"' EXIT

if [[ "$(cat "$E0_LOG_DIR/e0_smoke.exit")" != "0" ]]; then
  printf 'Refusing full E0 run because smoke gate did not pass.\n' >&2
  exit 1
fi
if [[ ! -x "$E0_ENV/bin/python" ]]; then
  printf 'Missing isolated extraction environment: %s\n' "$E0_ENV" >&2
  exit 1
fi

printf 'E0 full start: %s\n' "$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
cd "$E0_PURIFIED"

CUDA_VISIBLE_DEVICES=0 "$E0_ENV/bin/python" \
  -m experiments.correlation_audit.extract_legacy \
  --output-dir "$E0_CACHE" \
  --model-revision "$E0_MODEL_REV" \
  --dataset-revision "$E0_DATASET_REV" \
  --source-revision "$E0_SOURCE_REV" \
  --sequence-length 256 \
  --num-sequences 6000 \
  --layers 6 8 \
  --device cuda:0 \
  --batch-size 64 \
  --shard-size 1000

.venv/bin/python -c "import pathlib,torch; p=pathlib.Path('$E0_CACHE'); fs=sorted(p.glob('shard_*.pt')); assert len(fs)==6; ss=[torch.load(f,map_location='cpu',weights_only=False) for f in fs]; ids=torch.cat([s['article_ids'] for s in ss]); assert sum(len(s['tokens']) for s in ss)==6000; assert ids.unique().numel()>=32; print({'shards':len(fs),'sequences':len(ids),'unique_articles':int(ids.unique().numel()),'layers':sorted(ss[0]['residuals'])})"

for E0_LAYER in 6 8; do
  printf 'Starting robustness audit for layer %s at %s\n' "$E0_LAYER" "$(date -u +%FT%TZ)"
  CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/e0-mpl XDG_CACHE_HOME=/tmp/e0-cache \
    PYTHONPATH=. .venv/bin/python -m experiments.correlation_audit.robustness \
    --cache-dir "$E0_CACHE" \
    --layer "$E0_LAYER" \
    --output-dir "$E0_RESULTS" \
    --projection-dim 64 \
    --fit-tokens 60000 \
    --fit-documents 1000 \
    --max-lag 48 \
    --persistent-rank 8 \
    --psd-bootstrap 200 \
    --device cuda:0
done

cp "$E0_ROOT/purified/results/neurips_theory/e0_smoke/extraction_packages.txt" \
  "$E0_RESULTS/extraction_packages.txt"
printf 'E0 full complete: %s\n' "$(date -u +%FT%TZ)"
