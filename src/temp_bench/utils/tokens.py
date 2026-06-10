"""Single token-resolution path for HF, Anthropic, and GitHub.

The same logic works on local (`~/.tokens/`) and RunPod (`/workspace/
.tokens/`). The directory is auto-detected via :func:`tokens_dir`; agents
never hard-code paths.

File naming inside the tokens dir (kept consistent with
`bootstrap_runpod.sh`):

    hf_token         HuggingFace API token (write-enabled)
    anthropic_key    Anthropic API key (sk-ant-…)
    gh_token         GitHub PAT (ghp_…)

Resolution order (first hit wins) for :func:`get_token`:

    1. Override env var (HF_TOKEN / ANTHROPIC_API_KEY / GH_TOKEN)
    2. <tokens_dir>/<canonical_filename>
    3. ecosystem default (HF only — ~/.cache/huggingface/token; this is
       the SDK's default location, kept for compat with users who
       haven't migrated to the unified `.tokens/` layout)
    4. None (caller decides — raise or use anonymous access)

All file reads are mode-0600 friendly; the file's content is taken
verbatim minus trailing whitespace.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Canonical filename per token kind.
_FILENAMES: dict[str, str] = {
    "hf": "hf_token",
    "anthropic": "anthropic_key",
    "gh": "gh_token",
}

# Override env var per token kind.
_ENV_VARS: dict[str, str] = {
    "hf": "HF_TOKEN",
    "anthropic": "ANTHROPIC_API_KEY",
    "gh": "GH_TOKEN",
}


@lru_cache(maxsize=1)
def tokens_dir() -> Path:
    """Resolve the canonical tokens directory.

    Order:
        1. ``$TEMP_BENCH_TOKENS_DIR`` (escape hatch — used in tests)
        2. ``/workspace/.tokens`` (RunPod canonical)
        3. ``~/.tokens`` (local canonical)

    Returns the first directory that exists. If none exists, returns
    ``~/.tokens`` (the local default — caller may choose to mkdir it).
    """
    override = os.environ.get("TEMP_BENCH_TOKENS_DIR")
    if override:
        return Path(override).expanduser().resolve()

    rp = Path("/workspace/.tokens")
    if rp.is_dir():
        return rp

    local = Path.home() / ".tokens"
    return local


def _read_file(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        content = path.read_text().strip()
        return content or None
    except Exception:
        return None


def get_token(kind: str) -> str | None:
    """Resolve a token by kind. ``kind`` is one of ``hf``, ``anthropic``,
    or ``gh``.

    Returns None if not found anywhere. Caller decides how to handle
    (raise, fall back to anonymous, etc.).
    """
    if kind not in _FILENAMES:
        raise ValueError(f"unknown token kind {kind!r}; expected one of {sorted(_FILENAMES)}")

    # 1. env override
    env_var = _ENV_VARS[kind]
    val = os.environ.get(env_var)
    if val:
        return val.strip()

    # 2. canonical .tokens/ store
    val = _read_file(tokens_dir() / _FILENAMES[kind])
    if val:
        return val

    # 3. ecosystem-default fallback for HF only
    if kind == "hf":
        val = _read_file(Path.home() / ".cache" / "huggingface" / "token")
        if val:
            return val

    return None


def require_token(kind: str) -> str:
    """Like :func:`get_token` but raises if not found. Use when a
    missing token is a fatal error (e.g., HF push on ephemeral pod)."""
    val = get_token(kind)
    if not val:
        td = tokens_dir()
        raise RuntimeError(
            f"No {kind} token found.\n"
            f"  Searched: ${_ENV_VARS[kind]} env, {td}/{_FILENAMES[kind]}"
            + (f", ~/.cache/huggingface/token" if kind == "hf" else "")
            + ".\n"
            f"  Run: `bash scripts/bootstrap_local.sh` (local) or "
            f"`bash scripts/bootstrap_runpod.sh` (RunPod)."
        )
    return val


def token_status() -> dict[str, dict[str, object]]:
    """Snapshot of where each token currently resolves from. For
    debugging + smoke test display."""
    out: dict[str, dict[str, object]] = {}
    td = tokens_dir()
    for kind, fname in _FILENAMES.items():
        env_var = _ENV_VARS[kind]
        env_set = bool(os.environ.get(env_var))
        file_path = td / fname
        file_exists = file_path.is_file()
        eco_path: Path | None = None
        eco_exists = False
        if kind == "hf":
            eco_path = Path.home() / ".cache" / "huggingface" / "token"
            eco_exists = eco_path.is_file()

        if env_set:
            source = f"${env_var}"
        elif file_exists:
            source = str(file_path)
        elif eco_exists:
            source = str(eco_path)
        else:
            source = None

        out[kind] = {
            "resolved_from": source,
            "env": env_set,
            "file": file_exists,
            "ecosystem": eco_exists,
        }
    return out
