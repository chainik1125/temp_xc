# RunPod setup (final branch)

This doc covers how to bring a RunPod pod online for paper work. For
what to work on, every agent has its own briefing in
`purified/agents/<name>/briefing.md`. For the full operating manual,
see `purified/CLAUDE.md` (auto-loaded).

For **local** setup (your laptop / dev box), use
`bash purified/scripts/bootstrap_local.sh` instead — it populates the
same canonical `.tokens/` layout (under `~/.tokens/` rather than
`/workspace/.tokens/`). See § *Token storage* below.

## Pods

| Pod | GPUs | RAM | vCPU | /workspace | Mode |
|---|---|---|---|---|---|
| 2× H100 | 80 GB × 2 | 500 GB | 56 | 1 TB | **persistent** |
| 4× A40 | 48 GB × 4 | 200 GB | 38 | 1 TB | **ephemeral** |
| H200 (reserve) | 141 GB × 1 | 256 GB | 32 | 250 GB | persistent |

Pod modes determine sync behavior — see
`purified/docs/paper/hardware.md § Pod modes` for the full story
(checkpoints auto-push to HF on ephemeral pods, etc.).

## First-time setup on a new pod

Done once per pod. The bootstrap script handles tokens, uv install,
HF cache, repo clone, and `uv sync`.

```bash
cd /workspace
[ -d temp_xc ] || git clone https://github.com/chainik1125/temp_xc.git
cd temp_xc && git checkout final && git pull --rebase origin final

# Run the unified bootstrap (idempotent — re-running is safe)
bash purified/scripts/bootstrap_runpod.sh
```

The bootstrap:

- prompts for / loads the GitHub, HuggingFace, and Anthropic tokens
  into `/workspace/.tokens/{gh_token,hf_token,anthropic_key}` (mode 0600)
- configures `gh`, `huggingface-cli`, and exports `ANTHROPIC_API_KEY`
- sets `HF_HOME=/workspace/hf_cache` and `UV_LINK_MODE=copy` in
  `~/.bashrc` (the latter is mandatory — see § MooseFS gotcha below)
- runs `uv sync` from `purified/` to build `purified/.venv/`

## Per-session start (every shell, every restart)

```bash
cd /workspace/temp_xc/purified

# Pin the GPU + set AGENT_NAME + TEMP_BENCH_POD_MODE
source scripts/set_agent_env.sh <agent_name>

# Verify env (CRITICAL warnings are fatal — see preflight)
bash scripts/agent_smoke_test.sh

# On EPHEMERAL pods (4× A40) only — pull latest checkpoints + caches
bash scripts/sync_from_hf.sh
```

`<agent_name>` ∈ `agent_paper, agent_nlp, agent_em, agent_em_h200,
agent_steer, agent_back, a40_helper_gpu2, a40_helper_gpu3`. See
`purified/agents/README.md` for who's who.

## MooseFS gotcha — `UV_LINK_MODE=copy` is mandatory

RunPod's `/workspace` is on MooseFS. `uv`'s default `link-mode = hardlink`
silently produces partial installs there (dist-info dirs without
`RECORD` files). Symptom: `uv sync` uninstalls and reinstalls the
same package every invocation.

The bootstrap script and `purified/pyproject.toml` both set
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
rm -rf purified/.venv/lib/python3.12/site-packages/<pkg>-<ver>.dist-info
cd purified && uv sync && uv sync       # second pass should audit only
```

The package's actual code lives in a sibling directory (e.g. `pytest/`),
not in `*.dist-info/`, so deleting orphan metadata is safe.

## Long-running sweeps

```bash
cd /workspace/temp_xc/purified
TQDM_DISABLE=1 nohup .venv/bin/python -m experiments.c3_probing.run \
    > logs/c3_probing.log 2>&1 &
tail -f logs/c3_probing.log
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
bash purified/scripts/bootstrap_runpod.sh

# Local (interactive prompts; auto-detects existing local sources —
# ~/.cache/huggingface/token, ~/.env_autointerp, gh CLI)
bash purified/scripts/bootstrap_local.sh
```

Non-interactive (e.g. seeding a pod from your local box via SSH):

```bash
HF_TOKEN=hf_… ANTHROPIC_API_KEY=sk-ant-… GH_TOKEN=ghp_… \
    bash purified/scripts/bootstrap_runpod.sh
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
| Repo | `/workspace/temp_xc/` (branch: `final`) |
| Python env | `/workspace/temp_xc/purified/.venv/` |
| HF cache | `/workspace/hf_cache/` |
| Tokens | `/workspace/.tokens/{gh_token,hf_token,anthropic_key}` |
| GPU locks | `/workspace/.gpu_locks/gpu<idx>.lock` |
| Activation caches | `purified/results/act_cache/<act_cache_key>/` |
| Trained checkpoints | `purified/checkpoints/<train_key>/` |
| Per-cell artifacts | `purified/results/runs/<eval_key>/` |
| Leaderboard (append-only) | `purified/results/leaderboard.jsonl` |
| Agent briefings | `purified/agents/<name>/briefing.md` |

For everything else, start at `purified/CLAUDE.md`.
