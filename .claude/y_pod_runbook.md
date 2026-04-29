# Aniket — A40 RunPod first-time setup runbook

Single source for: "I just SSH'd into the pod, what do I run?" Lifts
the relevant pieces from `RUNPOD_INSTRUCTIONS.md` and
`scripts/runpod_phase7_bootstrap.sh`, with per-user isolation so
nothing collides with Han's setup on a shared pod.

Two assumptions about the pod:
- It's the Han Phase-7 image (Dockerfile in repo). Claude Code is
  preinstalled via npm. Conda env `torchgpu` exists but we'll use
  `uv` per `RUNPOD_INSTRUCTIONS.md`.
- SSH lands as `root` (RunPod default).

---

## 1. As root: create a non-root user `aniket`

```bash
# Skip if user already exists
id aniket >/dev/null 2>&1 || {
  useradd -m -s /bin/bash aniket
  # Optional: give passwordless sudo
  echo 'aniket ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/aniket
  chmod 440 /etc/sudoers.d/aniket
}

# Mirror your SSH key so you can ssh aniket@... directly next time
mkdir -p /home/aniket/.ssh
cp /root/.ssh/authorized_keys /home/aniket/.ssh/authorized_keys 2>/dev/null || true
chown -R aniket:aniket /home/aniket/.ssh
chmod 700 /home/aniket/.ssh && chmod 600 /home/aniket/.ssh/authorized_keys

# Make per-user workspace, owned by aniket
mkdir -p /workspace/aniket
chown aniket:aniket /workspace/aniket

# Drop into the aniket account
su - aniket
```

(Re-SSH next time as `aniket@<pod>` directly to skip the `su`.)

## 2. As `aniket`: clone the repo into your own workspace

```bash
cd /workspace/aniket
[ -d temp_xc ] || git clone https://github.com/chainik1125/temp_xc.git
cd temp_xc
git fetch origin
git checkout aniket-phase7-y
git pull
```

## 3. Per-user tokens (NOT shared with Han)

Han's `scripts/runpod_phase7_bootstrap.sh` writes to
`/workspace/.tokens/` — that's shared. Use a per-user dir instead.

```bash
mkdir -p /workspace/aniket/.tokens
chmod 700 /workspace/aniket/.tokens

# Paste each value when prompted; input is hidden.
read -rsp 'GitHub PAT (classic, repo scope): '   GH;  echo;  printf '%s' "$GH"  > /workspace/aniket/.tokens/gh_token        && unset GH
read -rsp 'Hugging Face token (read+write): '    HF;  echo;  printf '%s' "$HF"  > /workspace/aniket/.tokens/hf_token        && unset HF
read -rsp 'Anthropic API key (sk-ant-...): '     AK;  echo;  printf '%s' "$AK"  > /workspace/aniket/.tokens/anthropic_key   && unset AK
chmod 600 /workspace/aniket/.tokens/*
```

## 4. Env vars in your `~/.bashrc` (loads on every new shell)

Critical: `UV_LINK_MODE=copy` per RUNPOD_INSTRUCTIONS — `uv`'s
default hardlink mode silently produces broken installs on
MooseFS. Skip this and you'll burn an hour debugging.

```bash
cat >> ~/.bashrc <<'EOF'

# --- Aniket Phase-7 env (Y workstream) ---
export HF_HOME=/workspace/hf_cache                 # shared content-addressed cache; safe to share with Han
export UV_LINK_MODE=copy                            # MUST stay; default hardlink mode breaks on MooseFS
export TQDM_DISABLE=1                               # progress bars flood agent logs
export PYTHONPATH=/workspace/aniket/temp_xc

export GH_TOKEN="$(cat /workspace/aniket/.tokens/gh_token       2>/dev/null)"
export HF_TOKEN="$(cat /workspace/aniket/.tokens/hf_token       2>/dev/null)"
export ANTHROPIC_API_KEY="$(cat /workspace/aniket/.tokens/anthropic_key 2>/dev/null)"
# --- end Aniket Phase-7 env ---
EOF

source ~/.bashrc
```

Sanity-check the env loaded only YOUR keys:

```bash
env | grep -E '^(HF_TOKEN|ANTHROPIC_API_KEY|GH_TOKEN)=' \
  | sed 's/=.\{6\}.*/=********/'      # masks values, just shows they're set
echo "HF_HOME=$HF_HOME  UV_LINK_MODE=$UV_LINK_MODE  PYTHONPATH=$PYTHONPATH"
```

If `env` shows tokens you don't recognize, the shell pulled them
from somewhere shared — inspect `~/.profile`, `/etc/environment`,
`/etc/profile.d/*.sh` and remove or override.

## 5. Install Python deps via `uv`

```bash
# uv is in the Han image; reinstall only if missing.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

cd /workspace/aniket/temp_xc
uv sync
uv sync   # second run should print only "Resolved ... / Audited ..."; if it reinstalls, see RUNPOD_INSTRUCTIONS § "orphan dist-info"
```

Smoke test:

```bash
uv run python -c "
import torch
from src.architectures._tfa_module import TemporalSAE
print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB')
print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')
print('imports ok')
"
```

## 6. Hugging Face + Git credentials

```bash
# HF: token is already in $HF_TOKEN; this writes the cli-style file too.
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# Git: per-clone credential helper that reads the on-volume PAT
cd /workspace/aniket/temp_xc
git config --local user.name  "Aniket Deshpande"
git config --local user.email "aniketdeshh@gmail.com"
git config --local credential.helper \
  '!f() { echo username=x-access-token; printf "password=%s\n" "$(cat /workspace/aniket/.tokens/gh_token)"; }; f'

git push --dry-run origin aniket-phase7-y    # should report Everything up-to-date
```

## 7. Validate API keys (cheap, one call each)

```bash
# HF
curl -fsSL -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2 \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("HF user:", d.get("name"))'

# Anthropic — Haiku ping
curl -fsSL -X POST https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01" -H "Content-Type: application/json" \
  -d '{"model":"claude-haiku-4-5-20251001","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}' \
  | head -c 200; echo
```

If either fails, fix the token before launching CC.

## 8. tmux + Claude Code

```bash
# Persistent session so CC survives SSH disconnect
tmux new -s aniket-y

# Inside tmux, in /workspace/aniket/temp_xc:
cd /workspace/aniket/temp_xc
claude --dangerously-skip-permissions
```

CC must NOT be run as root — that's why step 1 created `aniket`.
If you accidentally run it as root, CC refuses
`--dangerously-skip-permissions`. Drop to `aniket` and retry.

To detach: `Ctrl-b d`. To reattach: `tmux attach -t aniket-y`.

## 9. First message to the pod-side CC

Read `.claude/y_pod_kickoff.md` aloud (or paste its contents) as
the first prompt. It points CC at the agent_y_brief, the paper-
wide brief, my Mac-side orientation log, and Dmitry's writeups,
and locks in the pre-flight + Q1 sequence.

```bash
# Inside CC, the first prompt is just:
read .claude/y_pod_kickoff.md and follow its instructions
```

---

## Quick reference — paths

| Thing | Path |
|---|---|
| Your repo clone | `/workspace/aniket/temp_xc` |
| Your tokens | `/workspace/aniket/.tokens/{gh_token,hf_token,anthropic_key}` |
| Shared HF cache | `/workspace/hf_cache` (content-addressed; safe to share) |
| Branch you work on | `aniket-phase7-y` |
| tmux session name | `aniket-y` |
| CC kickoff prompt | `.claude/y_pod_kickoff.md` (this branch) |

## What NOT to touch on a shared pod

- `/workspace/.tokens/` — Han's token dir.
- `/workspace/temp_xc/` — Han's clone (if it exists). Yours is at
  `/workspace/aniket/temp_xc`.
- Any tmux session not named `aniket-*`.
- `/home/han/`, `/home/sandbox/` — not yours.

## Reconnecting later

```bash
ssh aniket@<pod>
tmux attach -t aniket-y || tmux new -s aniket-y
cd /workspace/aniket/temp_xc
git pull origin aniket-phase7-y
uv sync         # no-op if in sync
claude --dangerously-skip-permissions     # only if not already running in tmux
```
