"""Code-version capture for audit trails.

Every result row + manifest entry carries a :class:`CodeVersion`:

    {
        commit_sha:   <full 40-char hex of git HEAD>,
        dirty:        <bool — was the working tree dirty?>,
        diff_sha256:  <sha256 of `git diff HEAD` output if dirty, else null>,
    }

Reconstruction recipe:

    git checkout <commit_sha>
    # if dirty=True:
    #   recover the diff from wherever you saved it
    #   verify sha256 matches diff_sha256
    #   git apply <recovered diff>

The runner refuses to launch with a dirty tree by default. The
``--allow-dirty`` opt-out flag (or ``TEMP_BENCH_ALLOW_DIRTY=1`` env var)
permits it; the diff hash is then recorded so the run is still
fully reconstructible.

Code-version is for **audit only** — NEVER fed into cache keys. Use
the arch ``arch_version`` bump for invalidation (registered in
``configs/archs.yaml``).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from temp_bench.core.schemas import CodeVersion


def _git(cmd: list[str], cwd: Path) -> str:
    """Run a git command from ``cwd``, return stdout (stripped)."""
    out = subprocess.check_output(
        ["git"] + cmd,
        cwd=str(cwd),
        stderr=subprocess.PIPE,
        text=True,
    )
    return out.strip()


def repo_root() -> Path:
    """Top of the repo (cwd-independent). Used as the git command cwd."""
    return Path(
        _git(["rev-parse", "--show-toplevel"], cwd=Path.cwd())
    ).resolve()


def commit_sha(root: Path | None = None) -> str:
    """Full 40-char hex SHA of ``git HEAD``."""
    return _git(["rev-parse", "HEAD"], cwd=root or repo_root())


def is_dirty(root: Path | None = None) -> bool:
    """True iff ``git diff HEAD`` is non-empty (working tree differs from HEAD)."""
    root = root or repo_root()
    # `git status --porcelain` covers tracked changes, untracked files, AND
    # staged-but-uncommitted changes. We treat any of these as "dirty"
    # because they affect reproducibility.
    status = _git(["status", "--porcelain"], cwd=root)
    return bool(status)


def diff_hash(root: Path | None = None) -> str | None:
    """sha256 hex of ``git diff HEAD`` output, or None if clean."""
    root = root or repo_root()
    diff_text = _git(["diff", "HEAD"], cwd=root)
    if not diff_text:
        return None
    return hashlib.sha256(diff_text.encode("utf-8")).hexdigest()


def capture(*, allow_dirty: bool | None = None) -> CodeVersion:
    """Snapshot current code version.

    Refuses (raises ``RuntimeError``) if the tree is dirty unless
    ``allow_dirty`` is True. The ``TEMP_BENCH_ALLOW_DIRTY=1`` env var
    is honored as an override.

    Args:
        allow_dirty: explicit override. ``None`` (default) defers to
            the env var ``TEMP_BENCH_ALLOW_DIRTY``.

    Returns:
        :class:`CodeVersion` capturing HEAD + dirty + diff hash.
    """
    root = repo_root()
    sha = commit_sha(root)
    dirty = is_dirty(root)

    if dirty:
        env_allow = os.environ.get("TEMP_BENCH_ALLOW_DIRTY") == "1"
        # Accept if either: explicit allow_dirty=True OR env says yes.
        # ``allow_dirty=False`` (argparse default) does NOT override env.
        if not (allow_dirty or env_allow):
            raise RuntimeError(
                f"Refusing to run with dirty working tree (HEAD = {sha[:8]}). "
                "Commit / stash changes, OR pass --allow-dirty "
                "(equivalently set TEMP_BENCH_ALLOW_DIRTY=1). "
                "Audit field will record the diff hash either way."
            )
        return CodeVersion(commit_sha=sha, dirty=True, diff_sha256=diff_hash(root))
    return CodeVersion(commit_sha=sha, dirty=False, diff_sha256=None)


def short_sha(cv: CodeVersion, n: int = 7) -> str:
    """Convenience: ``cv.commit_sha[:n]`` with a `~` suffix if dirty."""
    return cv.commit_sha[:n] + ("~" if cv.dirty else "")
