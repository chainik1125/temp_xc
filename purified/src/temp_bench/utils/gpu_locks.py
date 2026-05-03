"""GPU lockfile manager for multi-agent shared pods.

When N named agents share a pod with M ≥ N GPUs, each agent has one
**primary** GPU (always owned, pinned by ``scripts/set_agent_env.sh``).
The remaining GPUs form a **pool** — any agent may claim them via the
:func:`claim_gpu` context manager.

Claim semantics:

- ``with claim_gpu(idx, agent="agent_steer"):`` blocks (or fails fast,
  see ``timeout_sec``) until the GPU is free, writes a lock file at
  ``$LOCK_DIR/gpu<idx>.lock`` with PID + agent + timestamp, releases
  on context exit.
- Stale locks (whose recorded PID is no longer alive) are auto-reclaimed
  by any subsequent claim attempt.
- :func:`cleanup_stale` runs at smoke-test time to GC anything left
  behind by a crashed pod.
- :func:`gpu_lock_status` prints the current state for debugging.

Hard rule from PROTOCOL.md § 11.1: an agent **must** claim a spare GPU
before launching any process pinned to it. The primary GPU does not
need an explicit claim — ``set_agent_env.sh`` documents ownership.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def _lock_dir() -> Path:
    """Root for lock files. /workspace on RunPod, /tmp on local."""
    candidates = [
        os.environ.get("TEMP_BENCH_GPU_LOCK_DIR"),
        "/workspace/.gpu_locks" if Path("/workspace").is_dir() else None,
        "/tmp/temp_bench_gpu_locks",
    ]
    for c in candidates:
        if c:
            d = Path(c)
            d.mkdir(parents=True, exist_ok=True)
            return d
    raise RuntimeError("No lock dir resolvable")


def _lock_path(gpu_idx: int) -> Path:
    return _lock_dir() / f"gpu{gpu_idx}.lock"


def _pid_alive(pid: int) -> bool:
    """True iff a process with this PID currently exists on this host."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OSError):
        return False
    except PermissionError:
        # PID exists but is owned by another user — still "alive"
        return True
    return True


def _read_lock(path: Path) -> dict | None:
    """Return parsed lock contents, or None if unreadable / corrupt."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _write_lock(path: Path, *, gpu_idx: int, agent: str, note: str) -> None:
    payload = {
        "gpu_idx": int(gpu_idx),
        "pid": os.getpid(),
        "agent": agent,
        "claimed_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": os.environ.get("HOSTNAME", os.uname().nodename),
        "note": note,
    }
    # Write-then-rename for atomicity on POSIX
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True))
    tmp.replace(path)


@contextmanager
def claim_gpu(
    gpu_idx: int,
    *,
    agent: str | None = None,
    note: str = "",
    timeout_sec: float = 0.0,
    poll_sec: float = 2.0,
) -> Iterator[None]:
    """Claim ``gpu_idx`` for the duration of the ``with`` block.

    Args:
        gpu_idx: physical GPU index.
        agent: name to record in the lock (defaults to ``$AGENT_NAME``).
        note: free-form hint, written into the lock for debugging
            ("hi-batch C5 sweep, seeds 1+2", etc.).
        timeout_sec: 0 (default) → fail immediately if locked; N>0 →
            wait up to N seconds for the lock to clear; -1 → wait
            forever (agent can be Ctrl-C'd).
        poll_sec: how often to retry while waiting.

    Raises:
        RuntimeError if the GPU is held by a live, different agent and
        the timeout elapses.
    """
    if agent is None:
        agent = os.environ.get("AGENT_NAME", "unknown")

    path = _lock_path(gpu_idx)
    start = time.monotonic()

    while True:
        existing = _read_lock(path)
        if existing is None:
            # Free — claim it
            try:
                # exclusive create via O_EXCL semantics (write-then-rename
                # would race with another claimant; use O_EXCL)
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.close(fd)
                _write_lock(path, gpu_idx=gpu_idx, agent=agent, note=note)
                break
            except FileExistsError:
                # Lost the race — re-loop and read the now-existing lock
                continue
        else:
            holder_pid = int(existing.get("pid", 0))
            holder_agent = existing.get("agent", "?")
            if not _pid_alive(holder_pid):
                # Stale — reclaim
                warnings.warn(
                    f"[gpu_locks] reclaiming stale lock on GPU {gpu_idx} "
                    f"(was {holder_agent} PID {holder_pid})"
                )
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if holder_pid == os.getpid():
                # Re-entrant claim by the same process — count as success
                break
            elapsed = time.monotonic() - start
            if timeout_sec >= 0 and elapsed >= timeout_sec:
                raise RuntimeError(
                    f"GPU {gpu_idx} held by {holder_agent} (PID {holder_pid}) "
                    f"since {existing.get('claimed_ts')}. "
                    f"Wait or use a different GPU."
                )
            time.sleep(poll_sec)

    try:
        yield
    finally:
        # Only release if WE wrote it (PID matches)
        cur = _read_lock(path)
        if cur and int(cur.get("pid", -1)) == os.getpid():
            try:
                path.unlink()
            except FileNotFoundError:
                pass


@contextmanager
def claim_gpus(
    gpu_indices: list[int],
    *,
    agent: str | None = None,
    note: str = "",
    timeout_sec: float = 0.0,
) -> Iterator[None]:
    """Claim multiple GPUs atomically (all-or-nothing).

    If any GPU in the list can't be claimed within ``timeout_sec``, all
    already-claimed GPUs are released and the call raises. This avoids
    deadlocks where two agents each hold one of the other's spares.
    """
    if agent is None:
        agent = os.environ.get("AGENT_NAME", "unknown")

    # Sort indices so two callers asking for the same set always claim
    # in the same order — eliminates the deadlock.
    sorted_indices = sorted(set(gpu_indices))
    held: list[Path] = []

    try:
        for idx in sorted_indices:
            ctx = claim_gpu(idx, agent=agent, note=note, timeout_sec=timeout_sec)
            ctx.__enter__()
            held.append(_lock_path(idx))
        yield
    finally:
        for path in reversed(held):
            cur = _read_lock(path)
            if cur and int(cur.get("pid", -1)) == os.getpid():
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


def cleanup_stale() -> list[int]:
    """Remove any lock whose recorded PID is no longer alive.

    Returns the list of GPU indices that were freed. Call at agent
    session start (smoke test) — handles the case where a previous pod
    crashed without releasing.
    """
    freed: list[int] = []
    for path in _lock_dir().glob("gpu*.lock"):
        cur = _read_lock(path)
        if not cur:
            continue
        if not _pid_alive(int(cur.get("pid", 0))):
            try:
                path.unlink()
                freed.append(int(cur.get("gpu_idx", -1)))
            except FileNotFoundError:
                pass
    return freed


def gpu_lock_status() -> dict[int, dict | None]:
    """Snapshot of all current locks (for preflight + debugging)."""
    out: dict[int, dict | None] = {}
    for path in sorted(_lock_dir().glob("gpu*.lock")):
        try:
            idx = int(path.stem.removeprefix("gpu"))
        except ValueError:
            continue
        out[idx] = _read_lock(path)
    return out
