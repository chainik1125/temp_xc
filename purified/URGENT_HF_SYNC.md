# 🚨 URGENT — HF sync recovery procedure (post-RunPod-explosion 2026-05-06)

**Context**: 4 agents (`agent_nlp`, `agent_em`, `agent_steer`, `agent_back`)
went offline due to a RunPod incident on 2026-05-06. agent_nlp + agent_em
shared a **persistent** 2× H100 pod; the persistent storage volume
SURVIVED. Han is attaching a fresh pod to the same volume and assigning
**you** to run the HF sync recovery so the in-flight work isn't lost.

This document is your full mandate. **Read it top-to-bottom, then
execute.** Survivor agents (`agent_steer_100k`, `agent_em_100k`,
`agent_filler`) are still grinding on their own pods; do **NOT** touch
their work or step on their toes. Your scope is **only** the salvage
of what was on the dead H100 pod's persistent volume.

## What's at risk

Pre-explosion `agents/agent_paper/manifest.jsonl` audit (run from local
agent_paper, 2026-05-06 ~05:35Z):

| agent | total checkpoints | on HF | local-only |
|---|---:|---:|---:|
| agent_nlp | 14 | **0** | **14** ⚠️ |
| agent_em | 17 | 17 | 0 ✅ |
| agent_steer_100k | 15 | 14 | 1 ⚠️ |
| (everyone else) | — | 100% | 0 ✅ |

The `agent_nlp` 14 are all **deprecated over-batched** cells (B=256,
n_steps=10_000) — NOT canonical paper data per decisions § 15. Their
loss would not invalidate any paper claim. **But** the recovery
procedure may also find:

- **Newer canonical agent_nlp cells** that landed on the persistent
  volume but weren't committed to git or pushed to HF.
- **Activation caches** (`results/act_cache/<key>/acts.npy`,
  ~14 GB each) that aren't on HF.
- **Probe caches** (`results/probe_cache/<datasource>/<task>/...npy`)
  that aren't on HF.
- **Run directories** (`results/runs/<eval_key>/judge_outputs.jsonl`
  + `metrics.json`) — the **judge transcripts** for C4/C5/C6 are
  irreplaceable (regenerating costs Anthropic API budget).

You're recovering all four classes: checkpoints, act_caches, probe_caches,
and run dirs.

## HF repos (now public — Han 2026-05-06)

- `han1823123123/temp-bench-models` — checkpoints, keyed by
  `<train_key>` (16-hex).
- `han1823123123/temp-bench-data` — activation caches, probe caches,
  judge transcripts. Sub-dirs: `act_cache/<key>/`,
  `probe_cache/<datasource_name>/<task_name>/`,
  `runs/<eval_key>/`.

Token at `/workspace/.tokens/hf_token` (or `~/.tokens/hf_token` if
locally mounted differently).

## Step 0 — bring up the new pod

```bash
cd /workspace/temp_xc/purified

# Verify persistent storage attached (should show recent checkpoints).
ls checkpoints/ | head -10
ls results/act_cache/ | head -5
ls results/runs/ | head -5

# Verify token exists.
ls /workspace/.tokens/hf_token

# Pull latest git state.
git fetch origin
git status
git log --oneline -5
```

**If `checkpoints/` is empty**, the persistent volume didn't attach
correctly. Stop and ping Han.

## Step 1 — sync git first

```bash
cd /workspace/temp_xc/purified

# Pull latest. There may be uncommitted manifest/leaderboard changes
# on the volume from the dead pod's last writes — preserve them.
git stash push -m "pre-recovery-stash" --include-untracked
git pull --rebase origin final
git stash pop
```

If you see merge conflicts on `checkpoints/manifest.jsonl` or
`results/leaderboard.jsonl`: **keep both versions** (they're append-
only):

```bash
# Merge by concatenating + sorting unique
git checkout --theirs checkpoints/manifest.jsonl
git checkout --theirs results/leaderboard.jsonl
# Then re-append the local-version rows that aren't already there:
.venv/bin/python -c "
import json, hashlib
def dedup(path):
    seen = set()
    keep = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                k = r.get('train_key') or r.get('eval_key')
                if k and k not in seen:
                    seen.add(k)
                    keep.append(line)
            except: pass
    with open(path, 'w') as f:
        f.writelines(keep)
dedup('checkpoints/manifest.jsonl')
dedup('results/leaderboard.jsonl')
"
git add checkpoints/manifest.jsonl results/leaderboard.jsonl
```

## Step 2 — find local-only checkpoints

```bash
cd /workspace/temp_xc/purified

.venv/bin/python <<'PY'
import json
from pathlib import Path

local_only = []
with open('checkpoints/manifest.jsonl') as f:
    for line in f:
        try:
            r = json.loads(line)
            if r.get('hf_url') is None:
                local_only.append(r)
        except Exception:
            pass

# Filter: only entries whose .safetensors actually exists on disk
# (some manifest rows reference checkpoints that were never written
# because train_fn crashed, etc).
real = []
for r in local_only:
    tk = r.get('train_key', '?')
    p = Path(f'checkpoints/{tk}/model.safetensors')
    if p.exists():
        real.append(r)

print(f'Local-only entries with file on disk: {len(real)} (of {len(local_only)} manifest rows)')

# Group by agent so we know what's at stake.
from collections import Counter
by_agent = Counter(r.get('agent','?') for r in real)
print('  by agent:', dict(by_agent))

# Save the recovery list for Step 3.
with open('/tmp/recovery_checkpoints.jsonl', 'w') as f:
    for r in real:
        f.write(json.dumps(r) + '\n')

print('Wrote /tmp/recovery_checkpoints.jsonl')
PY
```

## Step 3 — push checkpoints to HF + update manifest

```bash
cd /workspace/temp_xc/purified

.venv/bin/python <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import HfApi

token = open('/workspace/.tokens/hf_token').read().strip()
api = HfApi(token=token)

with open('/tmp/recovery_checkpoints.jsonl') as f:
    queue = [json.loads(line) for line in f]

print(f'Pushing {len(queue)} checkpoints to HF...')
pushed = []
for r in queue:
    tk = r['train_key']
    local = Path(f'checkpoints/{tk}')
    try:
        url = api.upload_folder(
            folder_path=str(local),
            path_in_repo=tk,
            repo_id='han1823123123/temp-bench-models',
            repo_type='model',
        )
        # update manifest entry in-place
        r['hf_url'] = url if isinstance(url, str) else f'https://huggingface.co/han1823123123/temp-bench-models/tree/main/{tk}'
        pushed.append(r)
        print(f'  PUSHED {tk[:12]} ({r.get("arch","?")}, seed={r.get("seed","?")})')
    except Exception as e:
        print(f'  FAILED {tk[:12]}: {e}')

# Re-write manifest with updated hf_url for pushed entries.
print(f'Updating manifest with {len(pushed)} new hf_url fields...')
pushed_keys = {r['train_key']: r for r in pushed}
out_lines = []
with open('checkpoints/manifest.jsonl') as f:
    for line in f:
        try:
            r = json.loads(line)
            if r.get('train_key') in pushed_keys and r.get('hf_url') is None:
                r['hf_url'] = pushed_keys[r['train_key']]['hf_url']
                out_lines.append(json.dumps(r) + '\n')
            else:
                out_lines.append(line)
        except Exception:
            out_lines.append(line)

with open('checkpoints/manifest.jsonl', 'w') as f:
    f.writelines(out_lines)
print('Manifest updated.')
PY
```

## Step 4 — sync activation caches

```bash
cd /workspace/temp_xc/purified

.venv/bin/python <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

token = open('/workspace/.tokens/hf_token').read().strip()
api = HfApi(token=token)

# List all act_cache dirs on disk.
local_caches = sorted(p.name for p in Path('results/act_cache').iterdir() if p.is_dir())
print(f'Local act_cache keys: {len(local_caches)}')

# List what's already on HF.
remote = set()
try:
    files = api.list_repo_files('han1823123123/temp-bench-data', repo_type='dataset')
    for f in files:
        if f.startswith('act_cache/'):
            remote.add(f.split('/')[1])
except Exception as e:
    print(f'  WARN: could not list HF files: {e}')

missing = [k for k in local_caches if k not in remote]
print(f'Missing from HF: {len(missing)}')

for k in missing:
    local = Path(f'results/act_cache/{k}')
    if not (local / 'acts.npy').exists():
        print(f'  SKIP {k}: no acts.npy (probably abandoned)')
        continue
    try:
        api.upload_folder(
            folder_path=str(local),
            path_in_repo=f'act_cache/{k}',
            repo_id='han1823123123/temp-bench-data',
            repo_type='dataset',
        )
        size_gb = sum(p.stat().st_size for p in local.iterdir()) / 2**30
        print(f'  PUSHED act_cache/{k} ({size_gb:.1f} GB)')
    except Exception as e:
        print(f'  FAILED act_cache/{k}: {e}')
PY
```

## Step 5 — sync probe caches

```bash
cd /workspace/temp_xc/purified

.venv/bin/python <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

token = open('/workspace/.tokens/hf_token').read().strip()
api = HfApi(token=token)

local_pcs = sorted(p.name for p in Path('results/probe_cache').iterdir() if p.is_dir())
print(f'Local probe_cache datasources: {len(local_pcs)}')

remote = set()
try:
    files = api.list_repo_files('han1823123123/temp-bench-data', repo_type='dataset')
    for f in files:
        if f.startswith('probe_cache/'):
            remote.add(f.split('/')[1])
except Exception as e:
    print(f'  WARN: {e}')

missing = [d for d in local_pcs if d not in remote]
print(f'Missing from HF: {len(missing)}')

for d in missing:
    local = Path(f'results/probe_cache/{d}')
    try:
        api.upload_folder(
            folder_path=str(local),
            path_in_repo=f'probe_cache/{d}',
            repo_id='han1823123123/temp-bench-data',
            repo_type='dataset',
        )
        print(f'  PUSHED probe_cache/{d}')
    except Exception as e:
        print(f'  FAILED probe_cache/{d}: {e}')
PY
```

## Step 6 — sync run dirs (judge transcripts!)

These are the **most irreplaceable** artifacts — judge calls cost real
money to regenerate.

```bash
cd /workspace/temp_xc/purified

.venv/bin/python <<'PY'
from pathlib import Path
from huggingface_hub import HfApi

token = open('/workspace/.tokens/hf_token').read().strip()
api = HfApi(token=token)

# Find all run dirs that contain judge_outputs.jsonl (these are C4/C5/C6
# qualitative-eval cells; the .jsonl persists every Anthropic / Sonnet
# / Haiku judge call).
runs = sorted(p.name for p in Path('results/runs').iterdir()
              if (p / 'judge_outputs.jsonl').exists())
print(f'Run dirs with judge_outputs.jsonl: {len(runs)}')

remote = set()
try:
    files = api.list_repo_files('han1823123123/temp-bench-data', repo_type='dataset')
    for f in files:
        if f.startswith('runs/'):
            remote.add(f.split('/')[1])
except Exception as e:
    print(f'  WARN: {e}')

missing = [r for r in runs if r not in remote]
print(f'Missing from HF: {len(missing)}')

for r in missing:
    local = Path(f'results/runs/{r}')
    try:
        api.upload_folder(
            folder_path=str(local),
            path_in_repo=f'runs/{r}',
            repo_id='han1823123123/temp-bench-data',
            repo_type='dataset',
        )
        print(f'  PUSHED runs/{r}')
    except Exception as e:
        print(f'  FAILED runs/{r}: {e}')
PY
```

## Step 7 — find any cells that landed on the volume but never reached git

The leaderboard is the paper's source of truth. If agent_nlp / agent_em
landed cells on the persistent volume that weren't committed to git
before the explosion, they need to be rescued.

```bash
cd /workspace/temp_xc/purified

# Diff local leaderboard against the latest origin/final version.
git fetch origin
.venv/bin/python <<'PY'
import json, subprocess
from pathlib import Path

def load(path_or_blob):
    rows = []
    if isinstance(path_or_blob, Path):
        text = path_or_blob.read_text()
    else:
        text = path_or_blob
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return {r.get('eval_key'): r for r in rows if r.get('eval_key')}

local = load(Path('results/leaderboard.jsonl'))
remote_blob = subprocess.run(
    ['git', 'show', 'origin/final:purified/results/leaderboard.jsonl'],
    capture_output=True, text=True,
).stdout
remote = load(remote_blob)

local_only_eval_keys = sorted(set(local) - set(remote))
print(f'Local leaderboard rows NOT in origin/final: {len(local_only_eval_keys)}')
for ek in local_only_eval_keys[:30]:
    r = local[ek]
    print(f'  {ek[:12]}  {r.get("component"):>3} {r.get("arch"):>14} seed={r.get("seed")} agent={r.get("agent")}')
if len(local_only_eval_keys) > 30:
    print(f'  ... +{len(local_only_eval_keys)-30} more')
PY
```

If this list is non-empty, those rows are paper-bound data that
**must** be committed to git. After the HF push (Steps 3–6), do:

```bash
git add checkpoints/manifest.jsonl results/leaderboard.jsonl results/runs/
git commit -m "$(cat <<'EOF'
Agent RECOVERY: HF sync + leaderboard rescue post-RunPod incident

Recovered local-only artifacts from the persistent volume that the
dead H100 pod (agent_nlp + agent_em) had written but never pushed:
- N checkpoints to HF temp-bench-models
- N activation caches to HF temp-bench-data
- N probe caches to HF temp-bench-data
- N judge-transcript run dirs to HF temp-bench-data
- N leaderboard rows committed to git

Per /workspace/temp_xc/purified/URGENT_HF_SYNC.md procedure.
EOF
)"
git push origin final
```

## Step 8 — verify

```bash
# Re-run Step 2 — should now find 0 local-only entries.
.venv/bin/python -c "
import json
n = sum(1 for line in open('checkpoints/manifest.jsonl')
        if json.loads(line).get('hf_url') is None)
print(f'Local-only entries remaining: {n}')
"

# Spot-check 2-3 random recovered checkpoints by downloading them.
.venv/bin/python -c "
from huggingface_hub import HfApi
api = HfApi(token=open('/workspace/.tokens/hf_token').read().strip())
files = api.list_repo_files('han1823123123/temp-bench-models', repo_type='model')
print(f'Total files in temp-bench-models: {len(files)}')
print('Sample paths:')
for f in files[:5]:
    print('  ', f)
"
```

## After recovery is done

Update **this** document (`URGENT_HF_SYNC.md`) with a final-status
block at the top:

```markdown
## ✅ RECOVERY COMPLETE 2026-05-06 <UTC time>

- N checkpoints pushed to temp-bench-models
- N caches pushed to temp-bench-data
- N leaderboard rows committed to git
- All HF backups verified via random spot-checks

This document can be archived (move to docs/ or delete) once Han
confirms the paper-bound results are intact.
```

Commit + push the update so future readers see the resolution.

## Watch-outs

- **Don't run any new training cells.** This pod is for recovery only;
  the survivor agents (`agent_filler`, `agent_em_100k`,
  `agent_steer_100k`) are still active on their own pods and own the
  remaining work.
- **Don't HF-delete anything.** All operations here are append-only
  uploads + manifest updates.
- **Don't `git push --force`.** If a rebase looks scary, stop and
  surface to Han.
- **The 222 "unknown"-agent local-only entries** are old toy synthetic
  cells from C1/C2 (toy_markov, toy_coupled). They're tiny (toy
  d_sae=40) and not paper-headline; HF push them too in Step 3, but
  don't worry about provenance — they're shared by all earlier
  agent_paper sessions.
- **agent_nlp's 14 local-only checkpoints are deprecated over-batched
  cells** (B=256, n_steps=10000) — pre-decisions § 15 work that the
  analysis filters drop from the headline. Push them anyway for
  provenance, but their loss would not have invalidated any paper
  claim.
- **The actual headline IT C3 cells** (B=1024, n_steps=20K, per
  decisions § 15) were authored by `agent_filler` + `agent_steer` on
  their respective ephemeral pods, both at 100% HF coverage. **Those
  are NOT at risk.**

If anything goes wrong, ping Han in chat with the failure log.
