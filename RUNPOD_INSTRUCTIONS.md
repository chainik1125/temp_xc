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

## First-time setup on a new pod (HUMAN ONLY)

**These steps must be done by you, not by an agent.** The bootstrap
script is interactive (`read -rs` for token input); an agent session
cannot enter input. Run on a fresh persistent pod, or every time an
ephemeral pod is recreated, **before** spawning any agent.

### Step 1: bootstrap the primary clone

The bootstrap handles tokens, uv install, HF cache, repo clone, and
`uv sync`. After it runs, you have `/workspace/temp_xc/` ready.

```bash
cd /workspace
[ -d temp_xc ] || git clone https://github.com/chainik1125/temp_xc.git
cd temp_xc && git checkout final && git pull --rebase origin final

# Run the unified bootstrap (idempotent — re-running is safe)
bash purified/scripts/bootstrap_runpod.sh
```

If you want to provision unattended (e.g. CI), set the tokens via
env vars:

```bash
GH_TOKEN=ghp_xxx HF_TOKEN=hf_xxx ANTHROPIC_API_KEY=sk-xxx \
    bash purified/scripts/bootstrap_runpod.sh
```

### Step 2 (shared pods only): add a clone for the second agent

Each shared pod runs two agents on the same hardware. They must NOT
share a single `.git/` — concurrent `pull/commit/push` will collide
on `index.lock` and risk clobbering uncommitted edits. Solution: each
agent has its own clone. Tokens (`/workspace/.tokens/`) and HF cache
(`/workspace/hf_cache/`) are global, so the second clone is ~5 GB
extra disk on a 1 TB workspace.

| Pod | First agent (uses primary clone) | Second agent (gets own clone) | Add-clone command |
|---|---|---|---|
| 2× H100 | agent_nlp at `/workspace/temp_xc/` | agent_em at `/workspace/temp_xc_em/` | `bash /workspace/temp_xc/purified/scripts/add_agent_clone.sh agent_em` |
| 4× A40 | agent_back at `/workspace/temp_xc/` | agent_steer at `/workspace/temp_xc_steer/` | `bash /workspace/temp_xc/purified/scripts/add_agent_clone.sh agent_steer` |
| H200 | agent_em_h200 at `/workspace/temp_xc/` | (single-agent pod — skip step 2) | n/a |

The add-clone script is non-interactive (no token prompts — they're
already in `/workspace/.tokens/`), idempotent, and takes ~30 seconds.

### Step 3: spawn the agents — use `start_agent.sh`, NOT bare `claude`

**Critical**: Claude Code's Bash tool calls do not share shell state, so
an agent sourcing `set_agent_env.sh` as its "first action" exports the
GPU pin / agent name / pod mode into a one-shot subshell that's
discarded immediately. The next Bash call sees no pinning, no
attribution, no auto-push. Every worker must therefore inherit env
from the parent shell.

`scripts/start_agent.sh <agent>` is the launcher: it cd's to the
agent's clone, sources `set_agent_env.sh` in the parent shell, and
`exec`s `claude` so the env propagates into the agent process.

**First launch** (use `--fresh` so the agent reads its briefing for the
first time):

| Pod | First agent (T+0) | Second agent |
|---|---|---|
| 2× H100 | `bash /workspace/temp_xc/purified/scripts/start_agent.sh agent_nlp --fresh` | `bash /workspace/temp_xc_em/purified/scripts/start_agent.sh agent_em --fresh` |
| 4× A40 | `bash /workspace/temp_xc/purified/scripts/start_agent.sh agent_back --fresh` | `bash /workspace/temp_xc_steer/purified/scripts/start_agent.sh agent_steer --fresh` (T+~3hr) |
| H200 (fallback) | `bash /workspace/temp_xc/purified/scripts/start_agent.sh agent_em_h200 --fresh` | n/a |

**Re-launch after disconnect** (default — picks up the same session):

```bash
bash /workspace/temp_xc/purified/scripts/start_agent.sh agent_nlp
# (no --fresh; the wrapper passes --continue to claude so the worker
#  resumes its previous session instead of re-reading the briefing)
```

If you accidentally run bare `claude` from any of these directories,
the agent will run with `AGENT_NAME=unknown`, no GPU pinning, and on
ephemeral pods will skip HF auto-push — so leaderboard rows are
unattributable and a pod stop wipes your training. Always use
`start_agent.sh`.

The wrapper also guards: if the clone for a second agent doesn't
exist (you forgot Step 2), it prints the exact `add_agent_clone.sh`
command to fix it before launching anything.

The bootstrap:

- prompts for / loads the GitHub, HuggingFace, and Anthropic tokens
  into `/workspace/.tokens/{gh_token,hf_token,anthropic_key}` (mode 0600)
- configures `gh`, `huggingface-cli`, and exports `ANTHROPIC_API_KEY`
- sets `HF_HOME=/workspace/hf_cache` and `UV_LINK_MODE=copy` in
  `~/.bashrc` (the latter is mandatory — see § MooseFS gotcha below)
- runs `uv sync` from `purified/` to build `purified/.venv/`

## Per-session start (every shell, every restart)

**Always work from `purified/` as cwd.** `set_agent_env.sh` refuses to
source from anywhere else — the framework's `.venv`, configs, and
checkpoints all resolve relative to this dir.

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
