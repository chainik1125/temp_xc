"""btk-only variants (ACTMIX composition fix) — contract tests.

Pins the canonical convention in ``src/temp_bench/archs/btk_only.py``:

1. positive-rich equivalence: with transplanted weights and all-positive
   pre-acts, each btk-only twin's encode is BITWISE equal to its relu-mix
   parent (the copies haven't drifted; the fix only moves scarce-positive
   cells).
2. scarce-positive fingerprint: with all-negative pre-acts the relu-mix
   parent zero-picks (realized l0 = 0) while btk-only selects the least
   negative (realized l0 == nominal, survivors signed).
3. threshold path: explicit ``threshold_set`` flag (untracked -> BatchTopK
   fallback), negative thresholds gate correctly, EMA can write a negative
   threshold and sets the flag.
4. train_step: finite loss, backward runs, ``neg_frac`` diagnostic in [0,1].
5. registry: the five *_btkonly entries load + instantiate; the relu_mode
   guard rejects anything but "btk-only".
"""

from __future__ import annotations

import pytest
import torch

from temp_bench.archs.batchtopk_sae import BatchTopKSAE
from temp_bench.archs.btk_only import (
    BatchTopKSAEBTKOnly,
    StackedBatchTopKBTKOnly,
    TSAEBTKOnly,
    TXCBatchTopKPostBTKOnly,
    TXCBatchTopKPreBTKOnly,
)
from temp_bench.archs.stacked_batchtopk import StackedBatchTopK
from temp_bench.archs.tsae import TSAEPaper
from temp_bench.archs.txc_batchtopk import TXCBatchTopKPost, TXCBatchTopKPre
from temp_bench.core.config import import_by_path, load_arch
from temp_bench.interfaces.architecture import TempBenchArch

D_IN, D_SAE, T, K, B = 16, 32, 4, 3, 8

BTK_REGISTRY_NAMES = (
    "batchtopk_sae_btkonly",
    "tsae_btkonly",
    "stacked_batchtopk_btkonly",
    "txc_batchtopk_pre_btkonly",
    "txc_batchtopk_post_btkonly",
)

# (parent_cls, btk_cls, ctor kwargs, encode input shape, nominal selection count)
PAIRS = [
    (BatchTopKSAE, BatchTopKSAEBTKOnly,
     dict(d_in=D_IN, d_sae=D_SAE, k_pos=K), (B, D_IN), K * B),
    (TSAEPaper, TSAEBTKOnly,
     dict(d_in=D_IN, d_sae=D_SAE, k_pos=K), (B, D_IN), K * B),
    (StackedBatchTopK, StackedBatchTopKBTKOnly,
     dict(d_in=D_IN, d_sae=D_SAE, T=T, k_pos=K), (B, T, D_IN), K * B * T),
    (TXCBatchTopKPre, TXCBatchTopKPreBTKOnly,
     dict(d_in=D_IN, d_sae=D_SAE, T=T, k_pos=K), (B, T, D_IN), K * B * T),
    (TXCBatchTopKPost, TXCBatchTopKPostBTKOnly,
     dict(d_in=D_IN, d_sae=D_SAE, T=T, k_pos=K), (B, T, D_IN), K * B),
]
IDS = ["sae", "tsae", "stacked", "pre", "post"]


def _twins(parent_cls, btk_cls, kwargs, b_enc_fill):
    torch.manual_seed(0)
    parent = parent_cls(**kwargs)
    with torch.no_grad():
        parent.b_enc.fill_(b_enc_fill)
    torch.manual_seed(0)
    btk = btk_cls(**kwargs)
    btk.load_state_dict(parent.state_dict(), strict=False)
    return parent, btk


@pytest.mark.parametrize("parent_cls,btk_cls,kwargs,xshape,ksel", PAIRS, ids=IDS)
def test_positive_rich_equivalence(parent_cls, btk_cls, kwargs, xshape, ksel):
    """All pre-acts positive → btk-only encode == relu-mix encode bitwise."""
    parent, btk = _twins(parent_cls, btk_cls, kwargs, b_enc_fill=5.0)
    torch.manual_seed(1)
    x = 0.01 * torch.randn(*xshape)
    for train_mode in (True, False):   # eval w/o tracked threshold → same fallback
        parent.train(train_mode)
        btk.train(train_mode)
        z_p = parent.encode(x)
        z_b = btk.encode(x)
        assert torch.equal(z_p, z_b), (
            f"{btk_cls.__name__} drifted from parent on positive-rich input "
            f"(train={train_mode})"
        )


@pytest.mark.parametrize("parent_cls,btk_cls,kwargs,xshape,ksel", PAIRS, ids=IDS)
def test_scarce_positive_fingerprint(parent_cls, btk_cls, kwargs, xshape, ksel):
    """All pre-acts negative → parent zero-picks to l0=0; btk-only realizes
    l0 == nominal with signed (negative) survivors."""
    parent, btk = _twins(parent_cls, btk_cls, kwargs, b_enc_fill=-5.0)
    parent.train()
    btk.train()
    torch.manual_seed(2)
    x = 0.01 * torch.randn(*xshape)

    z_p = parent.encode(x)
    assert int((z_p != 0).sum()) == 0, "relu-mix parent should zero-pick to l0=0"

    z_b = btk.encode(x)
    nz = int((z_b != 0).sum())
    if btk_cls is TXCBatchTopKPreBTKOnly:
        # pre sums survivors over T into the shared code: atom collisions can
        # merge selections, so assert on the selection itself + the summed code.
        gated = btk._batchtopk(btk._compute_post(x))
        assert int((gated != 0).sum()) == ksel
        assert float(gated[gated != 0].max()) < 0
        assert 0 < nz <= ksel
    else:
        assert nz == ksel, f"realized l0 {nz} != nominal {ksel}"
    with torch.no_grad():
        assert float(z_b[z_b != 0].max()) < 0, "survivors must be signed negatives"


def test_threshold_flag_and_negative_gating():
    torch.manual_seed(0)
    m = BatchTopKSAEBTKOnly(d_in=D_IN, d_sae=D_SAE, k_pos=K)
    with torch.no_grad():
        m.b_enc.fill_(-5.0)
    m.eval()
    torch.manual_seed(3)
    x = 0.01 * torch.randn(B, D_IN)

    # flag unset → BatchTopK fallback even in eval (never the >=0 sentinel).
    z = m.encode(x)
    assert int((z != 0).sum()) == K * B

    # negative threshold below all pre-acts → everything passes the gate.
    with torch.no_grad():
        m.threshold.fill_(-10.0)
        m.threshold_set.fill_(1)
    z = m.encode(x)
    assert int((z != 0).sum()) == B * D_SAE

    # zero threshold → all-negative pre-acts are blocked.
    with torch.no_grad():
        m.threshold.fill_(0.0)
    z = m.encode(x)
    assert int((z != 0).sum()) == 0


def test_threshold_ema_tracks_negative():
    torch.manual_seed(0)
    m = BatchTopKSAEBTKOnly(d_in=D_IN, d_sae=D_SAE, k_pos=K,
                            threshold_start_step=0)
    with torch.no_grad():
        m.b_enc.fill_(-5.0)
    m.train()
    torch.manual_seed(4)
    for _ in range(2):                 # EMA first writes on the 2nd step
        out = m.train_step(0.01 * torch.randn(B, D_IN))
    assert bool(m.threshold_set.item())
    assert float(m.threshold.item()) < 0, "btk-only threshold must go negative here"
    assert 0.0 <= float(out["neg_frac"]) <= 1.0
    assert float(out["neg_frac"]) == 1.0   # every survivor negative in this regime


@pytest.mark.parametrize("parent_cls,btk_cls,kwargs,xshape,ksel", PAIRS, ids=IDS)
def test_train_step_smoke(parent_cls, btk_cls, kwargs, xshape, ksel):
    torch.manual_seed(0)
    m = btk_cls(**kwargs)
    m.train()
    torch.manual_seed(5)
    if btk_cls is TSAEBTKOnly:
        x = torch.randn(B, 6, D_IN)    # consumes sequence (pairs internally)
    else:
        x = torch.randn(*xshape)
    out = m.train_step(x)
    if isinstance(out, tuple):         # tsae keeps the v1 (loss, info) contract
        loss, info = out
    else:
        loss, info = out["loss"], out
    assert torch.isfinite(loss)
    assert 0.0 <= float(info["neg_frac"]) <= 1.0
    loss.backward()


def test_registry_entries_load_and_instantiate():
    for name in BTK_REGISTRY_NAMES:
        spec = load_arch(name)
        cls = import_by_path(spec.class_path)
        assert issubclass(cls, TempBenchArch)
        hp = {**spec.hparams, "d_sae": D_SAE, "k_pos": K}
        m = cls(d_in=D_IN, **hp)
        assert m.relu_mode == "btk-only"
        assert m.config.name == name
        assert cls.arch_version == spec.arch_version, (
            f"{name}: class arch_version {cls.arch_version} != "
            f"yaml {spec.arch_version}"
        )


def test_relu_mode_guard():
    with pytest.raises(ValueError):
        BatchTopKSAEBTKOnly(d_in=D_IN, d_sae=D_SAE, k_pos=K, relu_mode="relu-mix")
