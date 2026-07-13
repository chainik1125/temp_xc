"""Shared test fixtures.

**The runner is the only writer of the canonical results store** — it appends to
``results/leaderboard.jsonl`` and ``checkpoints/manifest.jsonl`` and saves
checkpoints under ``checkpoints/<train_key>/``. A test that calls
``run_experiment`` would therefore mutate the real, committed result set. The
``sandbox_store`` fixture below is **autouse**, so *every* test's runner writes go
to a throwaway tmp dir — pollution is structurally impossible, even for a
future test whose author forgets to think about it. (No test reads the real
store; those that need leaderboard data build their own under ``tmp_path``.)
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def sandbox_store(tmp_path, monkeypatch):
    """Redirect the runner's three write sinks to ``tmp_path`` so no test can
    touch the canonical store. Autouse — applied to every test.

    The sinks all derive from ``repo_root()`` via three functions; we patch each
    at *every* binding site (``trainer`` imports ``checkpoint_dir`` at module
    level, while ``cache``/``runner`` import it lazily). ``repo_root()`` itself is
    left alone, so config reads and ``code_version`` git introspection still use
    the real repo. Returns the sandbox root.
    """
    from temp_bench.core import cache, config, trainer

    lb = tmp_path / "results" / "leaderboard.jsonl"
    mf = tmp_path / "checkpoints" / "manifest.jsonl"

    def _ckpt(train_key):
        return tmp_path / "checkpoints" / train_key

    monkeypatch.setattr(cache, "leaderboard_path", lambda: lb)
    monkeypatch.setattr(cache, "manifest_path", lambda: mf)
    monkeypatch.setattr(config, "checkpoint_dir", _ckpt)    # cache + runner (lazy imports)
    monkeypatch.setattr(trainer, "checkpoint_dir", _ckpt)   # trainer (module-level bind)
    return tmp_path
