#!/usr/bin/env bash
# runpod_oneshot.sh — generic RunPod bootstrap for mech-interp projects.
#
# Designed to be portable across any AI / mech-interp repo that uses
# Claude Code + uv + a HuggingFace-cached subject model. Bootstraps a
# fresh RunPod from absolute zero (root SSH, blank /workspace volume) to
# "tmux + claude code running as a non-root user inside a uv venv", in
# one curl-pipe-bash invocation.
#
# USAGE (as root, from a fresh pod):
#
#     # Defaults: REPO/BRANCH/USER from the script. Override per project:
#     curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/<branch>/scripts/runpod_oneshot.sh \
#       | REPO=https://github.com/<owner>/<repo>.git \
#         BRANCH=<branch> \
#         USER_NAME=<user> \
#         bash
#
# Configurable env vars (each has a sensible default):
#   USER_NAME      'aniket'                          non-root user to create + run CC as
#   BRANCH         'aniket-ward-stage-b'             branch to check out
#   REPO           https://github.com/chainik1125/temp_xc.git
#   TMUX_SESSION   "${USER_NAME}-stageb"             tmux session name printed in instructions
#   KICKOFF_PROMPT  ''                               first prompt to paste into claude code
#                                                     (printed at end if set)
#
# Idempotent: re-running on a half-set-up pod is safe — every step
# checks state before acting.
#
# Bugs hit on real pods that this script handles (one fix per bug):
#   - `mkdir /workspace/.cache` fails as non-root if /workspace is owned
#     by an account that doesn't include the user. Fix: create + chmod 777
#     /workspace/.cache as root BEFORE handing off to the user.
#   - `update-locale` only takes effect at next login, so the current root
#     shell still has empty LANG. Fix: also export to current shell + write
#     to root's .bashrc + print a clear "either re-login or `export ...`"
#     warning.
#   - `npm install -g …` defaults to /usr/lib/node_modules and fails
#     EACCES for non-root. Fix: set npm prefix to ~/.npm-global FIRST,
#     verify with `npm config get prefix`, install non-silently so errors
#     surface, and re-add ~/.npm-global/bin to PATH in this very script
#     before running `which claude` to confirm.
#   - `runpod_activate.sh` cd's to a path that may not match the per-user
#     repo location (e.g. the script ships with `/workspace/temp_xc`
#     hardcoded but the user clones to `/workspace/aniket/temp_xc`).
#     Fix: print a one-line warning when this happens, but don't error
#     (the cd failure is benign for our use).
#   - User runs `tmux` while still being root, lands inside tmux as root,
#     can't find `claude` (which lives in aniket's PATH). Fix: print a
#     loud "switch to aniket user FIRST, then tmux" warning at end.

set -euo pipefail

USER_NAME="${USER_NAME:-aniket}"
BRANCH="${BRANCH:-aniket-ward-stage-b}"
REPO="${REPO:-https://github.com/chainik1125/temp_xc.git}"
TMUX_SESSION="${TMUX_SESSION:-${USER_NAME}-stageb}"
KICKOFF_PROMPT="${KICKOFF_PROMPT:-}"
WORKSPACE="/workspace/${USER_NAME}"
REPO_DIR="${WORKSPACE}/temp_xc"

if [ "$(id -u)" -ne 0 ]; then
    cat <<EOM >&2
ERROR: this script must run as root (RunPod's SSH lands as root by default).

If you've already done the root-side bits, finish manually as the user:

    su - ${USER_NAME}
    cd ${REPO_DIR}
    bash scripts/runpod_setup.sh
    nano .env
    source scripts/runpod_activate.sh
    npm config set prefix "\$HOME/.npm-global"
    npm install -g @anthropic-ai/claude-code
    export PATH="\$HOME/.npm-global/bin:\$PATH"
    tmux new -s ${TMUX_SESSION}
    claude --dangerously-skip-permissions
EOM
    exit 1
fi

echo "=== [1/6] System packages (as root) ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update >/dev/null
# `locales` provides en_US.UTF-8 so claude code's box-drawing + emoji
# render. `fonts-noto-color-emoji` covers the emoji glyphs claude code
# uses for status icons. Both required for symbols not to be rendered as ?.
apt-get install -y curl ca-certificates gnupg tmux vim nano less git \
    locales fonts-noto-color-emoji >/dev/null

# Generate en_US.UTF-8 and set as system default. Note: this only takes
# effect for FUTURE shells (login shells read /etc/default/locale). The
# current root shell will still show LANG='' until you `export` manually
# or re-login. We export below for THIS script + write to root's bashrc
# so future root sessions on this pod get it.
locale-gen en_US.UTF-8 >/dev/null 2>&1 || true
update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 >/dev/null 2>&1 || true
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 TERM="${TERM:-xterm-256color}"
add_root_bashrc() {
    grep -qF "$1" /root/.bashrc 2>/dev/null || echo "$1" >> /root/.bashrc
}
add_root_bashrc 'export LANG=en_US.UTF-8'
add_root_bashrc 'export LC_ALL=en_US.UTF-8'
add_root_bashrc 'export TERM=xterm-256color'

if ! command -v node >/dev/null 2>&1; then
    echo "  installing Node.js 22 from NodeSource..."
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null 2>&1
    apt-get install -y nodejs >/dev/null
fi
echo "  node $(node --version), npm $(npm --version)"

echo
echo "=== [2/6] Create non-root user '${USER_NAME}' ==="
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

# Per-user workspace dir.
mkdir -p "${WORKSPACE}"
chown "${USER_NAME}:${USER_NAME}" "${WORKSPACE}" 2>/dev/null || true

echo
echo "=== [3/6] Pre-create shared dirs as root (so user-side mkdirs don't EACCES) ==="
# /workspace/.cache (HF model cache) and /workspace/.cache/huggingface
# need to exist BEFORE the user-side runpod_setup.sh runs `mkdir -p` on
# them, because /workspace is sometimes owned by an account that doesn't
# include the new user. World-writable so any user can populate them.
mkdir -p /workspace/.cache /workspace/.cache/huggingface
chmod 777 /workspace /workspace/.cache /workspace/.cache/huggingface 2>/dev/null || true
echo "  /workspace/.cache → $(ls -ld /workspace/.cache | awk '{print $1, $3}')"

echo
echo "=== [4/6] Clone + checkout ${BRANCH} as ${USER_NAME} ==="
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
chmod +x scripts/runpod_setup.sh scripts/runpod_activate.sh scripts/runpod_verify_gpu.sh 2>/dev/null || true
EOF

echo
echo "=== [5/6] Run runpod_setup.sh as ${USER_NAME} (uv sync + .env template) ==="
su - "${USER_NAME}" <<EOF
set -euo pipefail
cd "${REPO_DIR}"
bash scripts/runpod_setup.sh
EOF

echo
echo "=== [6/6] Per-user shell env + Claude Code install (as ${USER_NAME}) ==="
su - "${USER_NAME}" <<'EOF'
set -euo pipefail

# Persist env across all future shells (interactive + login + tmux).
# We write to BOTH .bashrc and .profile so non-interactive su/ssh
# sessions also inherit the correct PATH/locale.
add_to_rcs() {
    for rc in "$HOME/.bashrc" "$HOME/.profile"; do
        touch "$rc"
        grep -qF "$1" "$rc" 2>/dev/null || echo "$1" >> "$rc"
    done
}
add_to_rcs 'export PATH=$HOME/.npm-global/bin:$PATH'
add_to_rcs 'export LANG=en_US.UTF-8'
add_to_rcs 'export LC_ALL=en_US.UTF-8'
# tmux strips claude-code's box-drawing chars unless the inner shell's
# TERM is xterm-256color (or tmux-256color, set via .tmux.conf below).
add_to_rcs 'export TERM=xterm-256color'

# Tmux config: force UTF-8, 256-color + truecolor passthrough, scrollback.
cat > "$HOME/.tmux.conf" <<'TMUX'
set -g default-terminal "tmux-256color"
set -ga terminal-overrides ",xterm-256color:Tc,xterm*:Tc"
set -g history-limit 50000
set -g mouse on
TMUX

# Set npm prefix BEFORE installing so the global install lands in the
# user's home rather than /usr/lib/node_modules (which requires root).
mkdir -p "$HOME/.npm-global"
npm config set prefix "$HOME/.npm-global"
NPM_PREFIX_NOW=$(npm config get prefix)
echo "  npm prefix = ${NPM_PREFIX_NOW}"
if [ "${NPM_PREFIX_NOW}" != "${HOME}/.npm-global" ]; then
    echo "  WARNING: npm prefix is not ${HOME}/.npm-global; install may EACCES."
fi

# Export the env in THIS subshell so the install + which checks below
# see the right PATH (the .bashrc edits only affect FUTURE shells).
export PATH="$HOME/.npm-global/bin:$PATH"
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 TERM=xterm-256color

# Install (loudly — no /dev/null redirect, so EACCES or network errors
# are visible; previous quiet install was masking failures).
if ! command -v claude >/dev/null 2>&1; then
    npm install -g @anthropic-ai/claude-code
fi
if command -v claude >/dev/null 2>&1; then
    echo "  ✓ claude $(claude --version) at $(which claude)"
else
    echo "  ✗ claude install failed; see npm output above"
    exit 1
fi
EOF

# ─── Summary ──────────────────────────────────────────────────────────────────
cat <<INSTRUCTIONS

==========================================================================
  ✓ Setup complete.

  IMPORTANT: your CURRENT root shell still has LANG='' because
  update-locale only affects FUTURE login shells. To fix this shell:
      export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 TERM=xterm-256color
  (You can ignore this if you're about to switch to ${USER_NAME} anyway.)

  STEP 1 — switch to non-root user (do NOT skip; tmux launched as root
            won't find claude code, and runpod_activate.sh writes to
            ~/.cache as the running user):

      su - ${USER_NAME}

  STEP 2 — fill in API keys in .env:

      cd ${REPO_DIR}
      nano .env                    # ANTHROPIC_API_KEY, HF_TOKEN, GH_TOKEN

  STEP 3 — open tmux as ${USER_NAME} and launch Claude Code:

      tmux new -s ${TMUX_SESSION}
      # inside tmux:
      cd ${REPO_DIR}
      source scripts/runpod_activate.sh
      claude --dangerously-skip-permissions

INSTRUCTIONS

if [ -n "${KICKOFF_PROMPT}" ]; then
    cat <<INSTRUCTIONS
  STEP 4 — paste this first prompt to claude code:

      ${KICKOFF_PROMPT}

INSTRUCTIONS
fi

cat <<INSTRUCTIONS
  Detach tmux: Ctrl-b d.  Reattach: tmux attach -t ${TMUX_SESSION}.

  GPU layout: this script is GPU-count-agnostic. The repo's run
  scripts auto-detect via nvidia-smi -L.
==========================================================================
INSTRUCTIONS
