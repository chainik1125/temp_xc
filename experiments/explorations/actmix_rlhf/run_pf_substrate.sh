#!/usr/bin/env bash
# § 8 paper-faithful substrate (CARD): stage A = l13-IT fineweb acts
# cache (config-keyed, idempotent); stage B = hh-rlhf@l13-IT eval cache
# (record-only fresh integrity stats). Launched at the GPU-2 flip per
# Han order 9e80f03aa, co-resident with the x6 tail cell.
set -u
cd "$(git rev-parse --show-toplevel)"
PIN=$(git rev-parse HEAD)
echo "[pf-substrate] pin $PIN $(date -u +%FT%TZ)"

echo "[pf-substrate] stage A: fineweb l13-IT acts cache"
.venv/bin/python - <<'PYEOF'
from temp_bench.core.config import load_datasource
from temp_bench.data.real_lm import build_activation_cache
spec = load_datasource("gemma_2_2b_it_l13_fineweb_24k128")
p = build_activation_cache(spec, batch_size=32)
print(f"[pf-substrate] stage A DONE -> {p}", flush=True)
PYEOF
rc=$?
if [ $rc -ne 0 ]; then echo "[pf-substrate] SUBSTRATE FAIL stage A rc=$rc"; exit $rc; fi

echo "[pf-substrate] stage B: hh-rlhf l13-IT eval cache"
.venv/bin/python -m experiments.explorations.actmix_rlhf.build_cache \
  --subject-model google/gemma-2-2b-it --layer 13 \
  --cache-dir /workspace/caches/rlhf/cached_hh_rlhf_l13it \
  --record-only --batch-size 16
rc=$?
if [ $rc -ne 0 ]; then echo "[pf-substrate] SUBSTRATE FAIL stage B rc=$rc"; exit $rc; fi

echo "[pf-substrate] SUBSTRATE DONE $(date -u +%FT%TZ)"
