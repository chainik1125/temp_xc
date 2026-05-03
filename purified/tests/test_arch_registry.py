"""Structural tests for the architecture registry.

Three contracts:

1. **YAML → class**: every entry in ``configs/locked_archs.yaml`` has an
   importable ``class_path`` that subclasses :class:`TempBenchArch`.
2. **No orphan files**: every ``.py`` in ``temp_bench/architectures/``
   (excluding ``base.py`` / ``__init__.py`` / ``_*.py``) is reachable
   by SOME yaml ``class_path``. Catches "agent forked the arch into a
   parallel file" — see Han's code-reuse contract (PROTOCOL.md § 11).
3. **Per-component overrides resolve**: ``load_arch(name, component)``
   merges ``per_component_hparams`` correctly.

These tests must run quickly and not require GPU. They validate the
registry structure only — actual model construction is exercised in
``test_runner_idempotency.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ARCH_DIR = Path(__file__).resolve().parents[1] / "src" / "temp_bench" / "architectures"

# Archs that are registered in YAML but whose class file is not yet
# ported from the wasteland. Each port deletes one entry from this set.
# Adding an entry here without porting is a contract violation: the test
# will accept the missing class temporarily, but a code reviewer should
# challenge "why is this allowlisted?"
#
# Empty target: when the last port lands, drop this set and the test
# becomes strictly "every yaml entry must have a class".
KNOWN_UNPORTED = {
    "stacked_sae",
    "tfa",
    "tfa_pos",
    "mlc",
    "txc_pro",
}


def test_every_yaml_arch_has_importable_class():
    """Every ``configs/locked_archs.yaml`` entry resolves to a real class
    subclassing :class:`TempBenchArch`. Catches:

    - YAML entry typos (class_path doesn't import).
    - Class doesn't subclass TempBenchArch.
    - Drift: an arch was ported (so removed from KNOWN_UNPORTED) but
      its class went missing — the test fails until the class is back.
    - New gap: an arch added to YAML without ever appearing in
      KNOWN_UNPORTED — flagged so a worker can't silently leave a
      hole.
    """
    pytest.importorskip("torch")
    from temp_bench.architectures.base import TempBenchArch
    from temp_bench.config import _resolve_class, list_archs, load_arch

    importable = set()
    bad_subclass = []
    for name in list_archs():
        spec = load_arch(name)
        try:
            cls = _resolve_class(spec.class_path)
        except (ImportError, ModuleNotFoundError, AttributeError):
            continue
        importable.add(name)
        if not (isinstance(cls, type) and issubclass(cls, TempBenchArch)):
            bad_subclass.append(
                f"{name}: {spec.class_path} resolved to {cls!r}, "
                "which is not a TempBenchArch subclass."
            )

    assert not bad_subclass, "\n  - ".join(bad_subclass)

    yaml_archs = set(list_archs())
    unported = yaml_archs - importable
    new_gaps = unported - KNOWN_UNPORTED
    assert not new_gaps, (
        f"Architectures in configs/locked_archs.yaml without a class file "
        f"and not in tests/test_arch_registry.py KNOWN_UNPORTED: {sorted(new_gaps)}.\n"
        "Either (a) port the class to temp_bench/architectures/, or "
        "(b) add to KNOWN_UNPORTED with a comment explaining the gap. "
        "PROTOCOL.md § 11 *Code reuse contract*."
    )

    stale_allowlist = KNOWN_UNPORTED - unported
    assert not stale_allowlist, (
        f"KNOWN_UNPORTED contains archs that ARE now ported: {sorted(stale_allowlist)}. "
        "Remove them from the set so the contract becomes strictly enforced."
    )


def test_no_orphan_arch_files():
    """Every .py in ``temp_bench/architectures/`` is referenced by some
    yaml entry. Files starting with ``_`` are private helpers and exempt.

    Catches duplicate ports (e.g. ``tsae_for_c7.py`` lurking next to
    the registered ``tsae.py``) and ``tsae_ours.py``-style deprecated
    files sneaking back in.
    """
    pytest.importorskip("yaml")
    from temp_bench.config import _load_archs_yaml

    py_files = {
        p.stem for p in ARCH_DIR.glob("*.py")
        if p.stem not in ("base", "__init__") and not p.stem.startswith("_")
    }
    yaml_archs = _load_archs_yaml()
    referenced_modules = {
        spec["class_path"].split(":", 1)[0].rsplit(".", 1)[1]
        for spec in yaml_archs.values()
    }
    orphans = py_files - referenced_modules
    assert not orphans, (
        f"Orphan arch files in {ARCH_DIR.name}/ (no yaml entry references them): "
        f"{sorted(orphans)}.\n"
        "Fix: either register in configs/locked_archs.yaml or delete. "
        "PROTOCOL.md § 11 *Code reuse contract*."
    )


def test_per_component_hparams_merge_correctly():
    """Load txc_base for c7 (override d_sae=32768) and c3 (no override)
    and verify the merged hparams are correct."""
    pytest.importorskip("yaml")
    from temp_bench.config import load_arch

    base_c3 = load_arch("txc_base", component="c3")
    base_c7 = load_arch("txc_base", component="c7")

    assert base_c3.hparams["d_sae"] == 18432, base_c3.hparams
    assert base_c7.hparams["d_sae"] == 32768, base_c7.hparams
    # Other hparams unchanged
    assert base_c3.hparams["T"] == base_c7.hparams["T"] == 5
    assert base_c3.hparams["k_pos"] == base_c7.hparams["k_pos"] == 20


def test_instantiate_arch_uses_d_in_from_caller():
    """instantiate_arch(spec, d_in=...) constructs the model with d_in
    from the datasource side, NOT from yaml hparams (yaml has no d_in).
    """
    pytest.importorskip("torch")
    from temp_bench.config import instantiate_arch, load_arch

    spec = load_arch("topk_sae")
    model = instantiate_arch(spec, d_in=64)
    assert model.config.d_in == 64
    assert model.config.d_sae == spec.hparams["d_sae"]


def test_instantiate_arch_rejects_bad_class_path():
    """A garbled class_path raises a clear error rather than silently
    returning None or a wrong class."""
    pytest.importorskip("yaml")
    from temp_bench.config import _resolve_class

    with pytest.raises(ValueError, match="module:Class"):
        _resolve_class("not_a_valid_path")
    with pytest.raises(ImportError, match="has no attribute"):
        _resolve_class("temp_bench.architectures.base:DoesNotExist")
