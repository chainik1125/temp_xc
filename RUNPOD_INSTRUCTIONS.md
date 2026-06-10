# RunPod setup (arxiv branch)

This doc covers how to bring a RunPod pod online for `arxiv`-branch work.
For what's in progress and what to pick up, read
`src/explorations/synthetic/STATUS.md` first, then `CLAUDE.md` (the
operating manual, auto-loaded).

For **local** setup (your laptop / dev box), use
`bash scripts/bootstrap_local.sh` instead — it populates the
same canonical `.tokens/` layout (under `~/.tokens/` rather than
`/workspace/.tokens/`). See § *Token storage* below.

## Pods

The current `arxiv` work (synthetic benchmarks) is light — each toy cell
uses ~2 GB of VRAM, so a **single modest GPU is plenty** and the
backtracking grid runs ~6–8 cells in parallel comfortably. Any
single-GPU pod (4090 / A40 / H100, ≥40 GB ideal but not required) works.

> The paper-era ephemeral-pod HF checkpoint auto-sync (`sync_from_hf.sh`)
> is **not** part of `arxiv`: synthetic checkpoints are tiny and
> regenerate from cache, and results travel in `git`. Use a **persistent**
> pod and rely on `git pull` / `git push`; push large artifacts to HF
> manually only if you generate them.

## First-time setup on a new pod

Done once per pod. The bootstrap script handles tokens, uv install,
HF cache, repo clone, and `uv sync`.

```bash
cd /workspace
[ -d temp_xc ] || git clone https://github.com/chainik1125/temp_xc.git
cd temp_xc && git checkout arxiv && git pull --rebase origin arxiv

# Run the unified bootstrap (idempotent — re-running is safe)
bash scripts/bootstrap_runpod.sh
```

The bootstrap:

- prompts for / loads the GitHub, HuggingFace, and Anthropic tokens
  into `/workspace/.tokens/{gh_token,hf_token,anthropic_key}` (mode 0600)
- configures `gh`, `huggingface-cli`, and exports `ANTHROPIC_API_KEY`
- sets `HF_HOME=/workspace/hf_cache` and `UV_LINK_MODE=copy` in
  `~/.bashrc` (the latter is mandatory — see § MooseFS gotcha below)
- runs `uv sync` from `` to build `.venv/`

## Per-session start (every shell, every restart)

**Always work from `` as cwd** — the framework's `.venv`,
configs, and checkpoints all resolve relative to this dir.

```bash
cd /workspace/temp_xc

# Provenance tag (stamped into the leaderboard `agent` field); freeform.
export AGENT_NAME=autoresearch
# — multi-GPU pods only: pin one GPU instead, via the legacy roster —
# source scripts/set_agent_env.sh <agent_name>

# Verify env + token resolution
bash scripts/agent_smoke_test.sh
```

`set_agent_env.sh` carries a legacy multi-agent GPU-pinning roster
(`agent_nlp, agent_em, agent_steer, agent_back, …`) for sharing one pod
across agents; on a single-GPU pod you don't need it — `export
AGENT_NAME=<tag>` is enough.

## MooseFS gotcha — `UV_LINK_MODE=copy` is mandatory

RunPod's `/workspace` is on MooseFS. `uv`'s default `link-mode = hardlink`
silently produces partial installs there (dist-info dirs without
`RECORD` files). Symptom: `uv sync` uninstalls and reinstalls the
same package every invocation.

The bootstrap script and `pyproject.toml` both set
`link-mode = copy`. Verify any new shell:

```bash
echo "UV_LINK_MODE=$UV_LINK_MODE  HF_HOME=$HF_HOME"
```

If empty, re-run `bootstrap_runpod.sh` or:

```bash
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc
echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc
source ~/.bashrc
```

## Diagnostic — orphan `dist-info`

If `uv sync` keeps reinstalling the same package and prints:

```
warning: Failed to uninstall package at .venv/lib/python3.12/site-packages/<pkg>-<ver>.dist-info due to missing `RECORD` file.
```

A previous `uv sync` ran without `UV_LINK_MODE=copy`. Fix:

```bash
export UV_LINK_MODE=copy
rm -rf .venv/lib/python3.12/site-packages/<pkg>-<ver>.dist-info
uv sync && uv sync       # second pass should audit only
```

The package's actual code lives in a sibling directory (e.g. `pytest/`),
not in `*.dist-info/`, so deleting orphan metadata is safe.

## Long-running sweeps

```bash
cd /workspace/temp_xc
TQDM_DISABLE=1 nohup .venv/bin/python -m explorations.synthetic.backtracking.run_grid 8 \
    > /tmp/grid.log 2>&1 &
tail -f /tmp/grid.log
```

`TQDM_DISABLE=1` is mandatory — progress bars flood logs and break
agent log-reading. Re-running the same component is idempotent (cells
with cached `eval_key` skip), so a crashed sweep resumes cleanly.

## Token storage (unified across local + RunPod)

Both local and RunPod use a canonical `.tokens/` directory. The
framework's `temp_bench.utils.get_token(kind)` resolves the same way
on both:

| Where | tokens dir |
|---|---|
| RunPod | `/workspace/.tokens/` (auto-detected) |
| Local | `~/.tokens/` (auto-detected) |
| Override | `$TEMP_BENCH_TOKENS_DIR` |

Files inside the tokens dir (mode 0600):

| Token | Filename |
|---|---|
| HuggingFace | `hf_token` |
| Anthropic | `anthropic_key` |
| GitHub | `gh_token` |

Resolution chain (first hit wins):

1. Override env var (`HF_TOKEN`, `ANTHROPIC_API_KEY`, `GH_TOKEN`)
2. `<tokens_dir>/<filename>`
3. (HF only) `~/.cache/huggingface/token` legacy fallback
4. None — caller errors with a clear `RuntimeError` if `require_token` is used

To populate the tokens dir from scratch:

```bash
# RunPod (interactive prompts; reads existing /workspace/.tokens if present)
bash scripts/bootstrap_runpod.sh

# Local (interactive prompts; auto-detects existing local sources —
# ~/.cache/huggingface/token, ~/.env_autointerp, gh CLI)
bash scripts/bootstrap_local.sh
```

Non-interactive (e.g. seeding a pod from your local box via SSH):

```bash
HF_TOKEN=hf_… ANTHROPIC_API_KEY=sk-ant-… GH_TOKEN=ghp_… \
    bash scripts/bootstrap_runpod.sh
```

Verify resolution at any time:

```bash
python -c '
from temp_bench.utils.tokens import token_status
import json; print(json.dumps(token_status(), indent=2))'
```

The smoke test (`scripts/agent_smoke_test.sh`) prints token resolution
on every session start.

## Quick reference

| Thing | Path |
|---|---|
| Repo | `/workspace/temp_xc/` (branch: `arxiv`) |
| Python env | `/workspace/temp_xc/.venv/` |
| HF cache | `/workspace/hf_cache/` |
| Tokens | `/workspace/.tokens/{gh_token,hf_token,anthropic_key}` |
| GPU locks | `/workspace/.gpu_locks/gpu<idx>.lock` |
| Activation caches | `results/act_cache/<act_cache_key>/` |
| Trained checkpoints | `checkpoints/<train_key>/` |
| Per-cell artifacts | `results/runs/<eval_key>/` |
| Leaderboard (append-only) | `results/leaderboard.jsonl` |
| Status briefing | `src/explorations/synthetic/STATUS.md` |

For everything else, start at `CLAUDE.md`.
