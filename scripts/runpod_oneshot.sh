#!/usr/bin/env bash
# scripts/runpod_oneshot.sh — one-shot pod setup for Aniket's CC workflow.
#
# Bootstraps a fresh RunPod from absolute zero to "tmux + claude
# --dangerously-skip-permissions running as a non-root user", in one
# script invocation. Assumes the pod image gives you root SSH and a
# /workspace persistent volume; nothing else.
#
# USAGE (as root, after SSH-ing into a fresh pod):
#
#     curl -fsSL https://raw.githubusercontent.com/chainik1125/temp_xc/aniket-phase7-y/scripts/runpod_oneshot.sh \
#       | bash
#
#   OR (if the repo is already on the pod):
#
#     bash /workspace/aniket/temp_xc/scripts/runpod_oneshot.sh
#
# Idempotent: re-running on a half-set-up pod is safe — every step
# checks state before acting.
#
# Configurable via env vars:
#   USER_NAME   default 'aniket'           — non-root user to create + run CC as
#   BRANCH      default 'aniket-phase7-y'  — branch to check out
#   REPO        default origin URL         — repo to clone
#
# Errors handled (each one we hit during the manual setup):
#   - apt's `node` package doesn't exist (binary is `nodejs`; need NodeSource for v22).
#   - /etc/sudoers.d/ may not exist on minimal images — sudo creation is skipped (sudo isn't required).
#   - chown on /workspace fails ("Operation not permitted"); the dir is already 777, so chown is best-effort.
#   - apt as non-root fails — root does all apt work, then `su` to user.
#   - claude refuses --dangerously-skip-permissions as root — install + run as the non-root user.
#   - Some scripts (runpod_setup.sh, runpod_activate.sh, runpod_verify_gpu.sh, verify_gpu_fit.py)
#     live on origin/aniket but not on every aniket-* branch — auto-fetch from origin/aniket.
#   - npm global install needs a user-writable prefix — sets ~/.npm-global.
#   - tmux env doesn't auto-load the venv — instructions below remind to source activate inside tmux.

set -euo pipefail

USER_NAME="${USER_NAME:-aniket}"
BRANCH="${BRANCH:-aniket-phase7-y}"
REPO="${REPO:-https://github.com/chainik1125/temp_xc.git}"
WORKSPACE="/workspace/${USER_NAME}"
REPO_DIR="${WORKSPACE}/temp_xc"

if [ "$(id -u)" -ne 0 ]; then
    cat <<EOM >&2
ERROR: this script must run as root (RunPod's SSH lands as root by default).

If you've already done the root-side bits, finish the user-side manually:

    su - ${USER_NAME}
    cd ${REPO_DIR}
    bash scripts/runpod_setup.sh
    nano .env
    source scripts/runpod_activate.sh
    npm config set prefix "\$HOME/.npm-global"
    npm install -g @anthropic-ai/claude-code
    export PATH="\$HOME/.npm-global/bin:\$PATH"
    tmux new -s ${USER_NAME}-y
    # inside tmux:
    source scripts/runpod_activate.sh
    claude --dangerously-skip-permissions
EOM
    exit 1
fi

echo "=== [1/5] System packages (as root) ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update >/dev/null
apt-get install -y curl ca-certificates gnupg tmux vim nano less git >/dev/null

if ! command -v node >/dev/null 2>&1; then
    echo "  installing Node.js 22 from NodeSource..."
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null 2>&1
    apt-get install -y nodejs >/dev/null
fi
echo "  node $(node --version), npm $(npm --version)"

echo
echo "=== [2/5] Create non-root user '${USER_NAME}' ==="
if id "${USER_NAME}" >/dev/null 2>&1; then
    echo "  user already exists"
else
    useradd -m -s /bin/bash "${USER_NAME}"
    echo "  created"
fi

# Mirror SSH keys so direct `ssh ${USER_NAME}@<pod>` works next time.
if [ -f /root/.ssh/authorized_keys ]; then
    mkdir -p "/home/${USER_NAME}/.ssh"
    cp /root/.ssh/authorized_keys "/home/${USER_NAME}/.ssh/authorized_keys"
    chown -R "${USER_NAME}:${USER_NAME}" "/home/${USER_NAME}/.ssh"
    chmod 700 "/home/${USER_NAME}/.ssh"
    chmod 600 "/home/${USER_NAME}/.ssh/authorized_keys"
fi

# Per-user workspace dir. /workspace is typically chmod-777 in RunPod
# images (Han's Dockerfile sets it). chown often fails ("Operation not
# permitted") on the mounted volume — that's expected; the dir is still
# usable because it's world-writable.
mkdir -p "${WORKSPACE}"
chown "${USER_NAME}:${USER_NAME}" "${WORKSPACE}" 2>/dev/null || true

echo
echo "=== [3/5] Clone + checkout ${BRANCH} as ${USER_NAME} ==="
su - "${USER_NAME}" <<EOF
set -euo pipefail
cd "${WORKSPACE}"
if [ ! -d "${REPO_DIR}/.git" ]; then
    git clone "${REPO}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"
git fetch origin
git checkout "${BRANCH}"
git pull --ff-only origin "${BRANCH}" || true

# Pull setup scripts from origin/aniket if they're not yet on this branch.
need_aniket_fetch=false
for f in scripts/runpod_setup.sh scripts/runpod_activate.sh scripts/runpod_verify_gpu.sh scripts/verify_gpu_fit.py; do
    if [ ! -f "\$f" ]; then need_aniket_fetch=true; fi
done
if [ "\$need_aniket_fetch" = "true" ]; then
    echo "  fetching missing setup scripts from origin/aniket..."
    git fetch origin aniket >/dev/null 2>&1
    for f in scripts/runpod_setup.sh scripts/runpod_activate.sh scripts/runpod_verify_gpu.sh scripts/verify_gpu_fit.py; do
        [ -f "\$f" ] || git checkout origin/aniket -- "\$f" 2>/dev/null || echo "    (skipped \$f — not on origin/aniket)"
    done
fi
chmod +x scripts/runpod_setup.sh scripts/runpod_activate.sh scripts/runpod_verify_gpu.sh 2>/dev/null || true
EOF

echo
echo "=== [4/5] Run runpod_setup.sh as ${USER_NAME} (uv + .env template) ==="
su - "${USER_NAME}" <<EOF
set -euo pipefail
cd "${REPO_DIR}"
bash scripts/runpod_setup.sh
EOF

echo
echo "=== [5/5] Install Claude Code in ~/.npm-global as ${USER_NAME} ==="
su - "${USER_NAME}" <<'EOF'
set -euo pipefail
mkdir -p "$HOME/.npm-global"
npm config set prefix "$HOME/.npm-global"
grep -qF '.npm-global/bin' "$HOME/.bashrc" 2>/dev/null \
    || echo 'export PATH=$HOME/.npm-global/bin:$PATH' >> "$HOME/.bashrc"
export PATH="$HOME/.npm-global/bin:$PATH"
if ! command -v claude >/dev/null 2>&1; then
    npm install -g @anthropic-ai/claude-code >/dev/null 2>&1
fi
if command -v claude >/dev/null 2>&1; then
    echo "  claude $(claude --version)"
else
    echo "  WARNING: claude install may have failed; check 'npm install -g @anthropic-ai/claude-code' manually"
fi
EOF

cat <<INSTRUCTIONS

==========================================================================
  Setup complete on this pod.

  Final manual steps (env values are private and not in the script):

      su - ${USER_NAME}
      cd ${REPO_DIR}
      nano .env                                # fill in ANTHROPIC_API_KEY, HF_TOKEN, WANDB_API_KEY
      source scripts/runpod_activate.sh        # confirms keys + GPU
      tmux new -s ${USER_NAME}-y
      # inside tmux (fresh shell, env not loaded yet):
      cd ${REPO_DIR}
      source scripts/runpod_activate.sh
      claude --dangerously-skip-permissions

  First prompt to Claude Code:

      read .claude/y_pod_kickoff.md and follow its instructions

  Detach tmux: Ctrl-b d.  Reattach later: tmux attach -t ${USER_NAME}-y.
==========================================================================
INSTRUCTIONS
