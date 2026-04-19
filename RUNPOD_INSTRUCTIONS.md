# RunPod agent setup

This doc covers only the **environment**. For what to work on, read the latest active phase's `brief.md` and most recent dated log under `docs/han/research_logs/phaseN_<shortname>/`. Each phase's experiments live at `experiments/phaseN_<shortname>/` (same slug). See `CLAUDE.md` "Phase convention" section for the full file-naming rules inside a phase dir (`brief.md` → `plan.md` → dated experiments → `summary.md`).

## First-time setup on a new pod

Everything lives on RunPod's persistent `/workspace` volume so it survives pod stop/start cycles.

```bash
cd /workspace
[ -d temp_xc ] || git clone https://github.com/chainik1125/temp_xc.git
cd temp_xc && git checkout han && git pull

# Install uv if the pod image doesn't ship it
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Persist HF cache + uv link-mode on /workspace
# UV_LINK_MODE=copy is REQUIRED: uv's cache and /workspace live on different
# filesystems, and the default hardlink mode silently produces broken installs
# on MooseFS (packages end up with partial file trees).
export HF_HOME=/workspace/hf_cache
export UV_LINK_MODE=copy
for line in 'export HF_HOME=/workspace/hf_cache' 'export UV_LINK_MODE=copy'; do
    grep -qF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
done

# Build the venv from pyproject.toml + uv.lock. Idempotent; safe to re-run.
uv sync

# Hugging Face auth (only needed first time; the token is stored under $HF_HOME)
huggingface-cli whoami >/dev/null 2>&1 || huggingface-cli login
```

## Reconnecting to an existing pod

```bash
cd /workspace/temp_xc
source .venv/bin/activate    # activates uv's venv
git pull origin han
uv sync                      # no-op if in sync
```

**Check `HF_HOME` and `UV_LINK_MODE` first.** `~/.bashrc` lives at `/home/appuser/.bashrc`, which is NOT guaranteed to persist across pod stop/start on all RunPod configurations. Run:

```bash
echo "HF_HOME=$HF_HOME  UV_LINK_MODE=$UV_LINK_MODE"
```

If either is empty, re-run the `export` + `~/.bashrc` block from first-time setup before touching `uv sync`. Running `uv sync` without `UV_LINK_MODE=copy` on MooseFS silently produces partial installs (dist-info dirs without `RECORD` files), which then make every subsequent `uv sync` spin its wheels — see "Known failure: orphan dist-info" below.

## Verify the environment

```bash
uv run python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB')
print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')
from src.architectures._tfa_module import TemporalSAE  # smoke test
print('✓ bench imports ok')
"
```

Also confirm `uv sync` is idempotent — the venv is clean if the second call only audits, without any "Uninstalled"/"Installed" lines:

```bash
uv sync && uv sync   # second run should show only "Resolved ... / Audited ..."
```

If the second run still installs or uninstalls, see the next section.

### Known failure: orphan dist-info

**Symptom.** `uv sync` uninstalls and reinstalls the same package every invocation and prints:

```
warning: Failed to uninstall package at .venv/lib/python3.12/site-packages/<pkg>-<ver>.dist-info due to missing `RECORD` file.
```

**Cause.** A prior `uv sync` ran without `UV_LINK_MODE=copy`, leaving a partial dist-info directory (metadata only, no `RECORD`). `uv` notices the unexpected install every run, tries to uninstall, can't, and reinstalls the correct version — but never cleans the orphan, so the loop repeats.

**Fix.** Delete the orphan dist-info directory and re-sync. Confirm `UV_LINK_MODE=copy` is set first:

```bash
export UV_LINK_MODE=copy
rm -rf .venv/lib/python3.12/site-packages/<pkg>-<ver>.dist-info
uv sync && uv sync    # second run should audit only
```

The package's actual code lives in a sibling directory (e.g. `pytest/`), not in `*.dist-info/`, so deleting the orphan metadata dir is safe.

## GitHub push access (persistent across pod sessions)

The token lives at `/workspace/.github-token` on the persistent volume; a per-repo credential helper reads it at push time. Survives pod stop/start — `git push origin han` works in every new session.

First-time setup (once per persistent volume):

```bash
# 1) Write the PAT to the persistent volume. Use printf (no trailing newline).
printf '%s' 'ghp_YOUR_CLASSIC_PAT' > /workspace/.github-token
chmod 600 /workspace/.github-token

# 2) Wire it into the repo's git config (once per clone)
cd /workspace/temp_xc
git config --local credential.helper \
    '!f() { echo username=x-access-token; printf "password=%s\n" "$(cat /workspace/.github-token)"; }; f'

# 3) Verify
git push --dry-run origin han
```

**Token type:** classic PAT with `repo` scope. Fine-grained PATs must also list `chainik1125/temp_xc` under "selected repositories" — forgetting this is the #1 cause of 403s even when you're a collaborator.

**Rotation:** overwrite `/workspace/.github-token` in place; no other changes needed.

## Anthropic API key (for autointerp)

If the task involves Claude Haiku labeling, set the key once per pod:

```bash
printf '%s' 'sk-ant-YOUR_KEY' > /workspace/.anthropic-key
chmod 600 /workspace/.anthropic-key
grep -qF 'ANTHROPIC_API_KEY' ~/.bashrc || \
    echo 'export ANTHROPIC_API_KEY=$(cat /workspace/.anthropic-key)' >> ~/.bashrc
```

## Running experiments

From the repo root with the venv active:

```bash
TQDM_DISABLE=1 uv run python experiments/phaseN_<shortname>/<script_name>.py
```

`TQDM_DISABLE=1` is mandatory — progress bars flood logs and break reading them back through the agent's tools.

For long-running sweeps, launch under `nohup`:

```bash
mkdir -p logs
TQDM_DISABLE=1 nohup uv run python experiments/phaseN_<shortname>/<script>.py > logs/<run>.log 2>&1 &
tail -f logs/<run>.log
```

Resumable sweeps save per-run JSONs incrementally; restarting the same command picks up where it left off.

## What to work on

This file does NOT describe the current experiment. The active work is always in the most recent phase dir:

```bash
ls -d docs/han/research_logs/phase*_*/ | tail -n 1               # newest phase dir
ls -t docs/han/research_logs/phase*_*/brief.md                   # per-phase briefings
ls -t docs/han/research_logs/phase*_*/*.md | head                # newest log across all phases
```

For a phase that's just starting, read `brief.md` first (context, priorities, sub-phase ordering), then `plan.md` (pre-registered methodology). For a phase in flight, read the newest dated experiment log. Branch state, open questions, and next steps all live in these files.

## Quick reference — where things live

| Thing | Path |
|---|---|
| Python env | `/workspace/temp_xc/.venv/` |
| HF cache (models, datasets) | `/workspace/hf_cache/` |
| Cached activations | `/workspace/temp_xc/data/cached_activations/` |
| Trained checkpoints | `/workspace/temp_xc/results/nlp_sweep/**/ckpts/` (gitignored; on-disk only) |
| Research logs | `/workspace/temp_xc/docs/han/research_logs/phase*/` |
| Backend (architectures, data, eval, training, plotting, pipeline) | `/workspace/temp_xc/src/` |
| Experiment scripts | `/workspace/temp_xc/experiments/phaseN_<shortname>/` (paired with `docs/han/research_logs/phaseN_<shortname>/`) |
| GitHub token | `/workspace/.github-token` |
| Anthropic key | `/workspace/.anthropic-key` |
