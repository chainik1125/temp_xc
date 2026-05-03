"""GPU lockfile manager tests.

These tests validate the Primary + Pool sharing protocol: claim,
release, stale-lock detection, cross-agent contention, and
all-or-nothing multi-claim.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def tmp_lock_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("TEMP_BENCH_GPU_LOCK_DIR", td)
        # Bust any cached state in the module
        from temp_bench.utils import gpu_locks
        # _lock_dir() is uncached but lock files persist on disk
        yield Path(td)


def test_claim_and_release(tmp_lock_dir):
    from temp_bench.utils.gpu_locks import claim_gpu, gpu_lock_status

    assert gpu_lock_status() == {}, "lock dir starts empty"

    with claim_gpu(2, agent="agent_steer", note="testing"):
        status = gpu_lock_status()
        assert 2 in status
        assert status[2]["agent"] == "agent_steer"
        assert status[2]["pid"] == os.getpid()

    # released
    assert gpu_lock_status() == {}


def test_double_claim_same_process_is_reentrant(tmp_lock_dir):
    """Claiming a GPU we already hold succeeds (re-entrancy)."""
    from temp_bench.utils.gpu_locks import claim_gpu

    with claim_gpu(2, agent="agent_steer"):
        with claim_gpu(2, agent="agent_steer"):
            pass  # same PID — should not raise
    # released after outer exit


def test_claim_blocks_other_agent_immediately(tmp_lock_dir):
    """Cross-agent claim with timeout=0 raises immediately."""
    from temp_bench.utils.gpu_locks import claim_gpu

    # Simulate a different process holding the lock by writing a fake
    # lock with a real PID that's alive (this process)
    import json
    from datetime import datetime, timezone

    lock_path = tmp_lock_dir / "gpu2.lock"
    lock_path.write_text(json.dumps({
        "gpu_idx": 2,
        "pid": os.getpid(),
        "agent": "fake_other_agent",
        "claimed_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "test",
        "note": "",
    }))

    # Patch os.getpid for the *claim* code path so it sees the holder
    # as a *different* PID. The reentrant short-circuit checks
    # `holder_pid == os.getpid()`, so we make our caller's PID different.
    with patch("temp_bench.utils.gpu_locks.os.getpid", return_value=999_999):
        # Force _pid_alive to return True for the (fake) holder
        with patch("temp_bench.utils.gpu_locks._pid_alive", return_value=True):
            with pytest.raises(RuntimeError, match="held by"):
                with claim_gpu(2, agent="agent_back", timeout_sec=0):
                    pass


def test_stale_lock_is_auto_reclaimed(tmp_lock_dir):
    """A lock with a dead PID gets auto-cleaned on next claim attempt."""
    from temp_bench.utils.gpu_locks import claim_gpu, gpu_lock_status
    import json
    from datetime import datetime, timezone

    # Fake-stale lock (PID 1 is init; never owned by us; we'll mock dead)
    lock_path = tmp_lock_dir / "gpu3.lock"
    lock_path.write_text(json.dumps({
        "gpu_idx": 3,
        "pid": 999_999_999,
        "agent": "crashed_agent",
        "claimed_ts": "2026-05-03T00:00:00Z",
        "host": "old_pod",
        "note": "",
    }))

    # The default _pid_alive will report 999_999_999 as dead
    with claim_gpu(3, agent="agent_steer"):
        status = gpu_lock_status()
        assert status[3]["agent"] == "agent_steer"
        assert status[3]["pid"] == os.getpid()


def test_cleanup_stale(tmp_lock_dir):
    """`cleanup_stale` GCs all dead locks at once."""
    from temp_bench.utils.gpu_locks import cleanup_stale, gpu_lock_status
    import json

    # Two stale locks (impossibly large PIDs)
    for idx in (2, 3):
        (tmp_lock_dir / f"gpu{idx}.lock").write_text(json.dumps({
            "gpu_idx": idx,
            "pid": 999_000_000 + idx,
            "agent": "ghost",
            "claimed_ts": "2026-01-01T00:00:00Z",
            "host": "old",
            "note": "",
        }))

    freed = cleanup_stale()
    assert sorted(freed) == [2, 3]
    assert gpu_lock_status() == {}


def test_claim_gpus_atomic_all_or_nothing(tmp_lock_dir):
    """If we can claim {2} but not {3}, both should end up released."""
    from temp_bench.utils.gpu_locks import claim_gpus, gpu_lock_status
    import json
    from datetime import datetime, timezone

    # Pre-occupy GPU 3 with a "live" foreign holder
    (tmp_lock_dir / "gpu3.lock").write_text(json.dumps({
        "gpu_idx": 3,
        "pid": os.getpid(),  # so _pid_alive() returns True
        "agent": "fake_other_agent",
        "claimed_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": "test",
        "note": "",
    }))

    with patch("temp_bench.utils.gpu_locks.os.getpid", return_value=999_998):
        with patch("temp_bench.utils.gpu_locks._pid_alive", return_value=True):
            with pytest.raises(RuntimeError):
                with claim_gpus([2, 3], agent="agent_back", timeout_sec=0):
                    pass

    # GPU 2 should be free (we released after failing to claim 3)
    status = gpu_lock_status()
    assert 2 not in status, "claim_gpus must release GPU 2 if claiming GPU 3 failed"
    # GPU 3 still has the foreign lock — not our problem to clean
    assert 3 in status
