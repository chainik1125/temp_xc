"""Per-architecture interface smoketest — run this BEFORE any sweep.

Every architecture in this repo has its own input contract, and discovering them
one crash at a time inside a long run is expensive. This instantiates each panel
architecture at each window size, runs one training step and one encode, and
prints the shapes it accepted. Anything that fails here is a harness bug, not a
result.

Contracts discovered so far (this is why the file exists):
  * BatchTopKSAE  : construct T=1; train_step wants (B, d_in); encode (B, d_in)
  * TSAEPaper     : construct T=1 ONLY; train_step wants (B, seq>=2, d_in) because
                    the contrastive term samples a consecutive pair; encode takes
                    either (B, d_in) or (B, T, d_in); train_step returns a TUPLE
                    (loss, info) rather than a dict
  * Stacked / TXC : construct with T; train_step and encode want (B, T, d_in)

Run:  .venv/bin/python -m experiments.explorations.txcwin.smoke
"""

from __future__ import annotations

import torch

from experiments.explorations.txcwin.sweep import PANEL, _import

D_IN, D_SAE, K, B = 64, 128, 4, 16


def loss_of(out):
    """Every arch returns something different; normalise to a scalar tensor."""
    if isinstance(out, dict):
        return out.get("loss", next(iter(out.values())))
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def try_shapes(arch, T):
    """Return (train_shape, encode_shape) that the arch accepts, or raise."""
    x3 = torch.randn(B, max(T, 2), D_IN, device="cuda")
    x2 = x3[:, -1, :]
    train_ok = None
    for name, x in (("3D", x3), ("2D", x2)):
        try:
            arch.pre_step()
            out = arch.train_step(x)
            loss_of(out).backward()
            arch.post_step()
            train_ok = name
            break
        except (ValueError, RuntimeError, IndexError):
            arch.zero_grad(set_to_none=True)
    enc_ok, enc_shape = None, None
    with torch.no_grad():
        for name, x in (("3D", x3[:, :T, :] if T > 1 else x3), ("2D", x2)):
            try:
                z = arch.encode(x)
                enc_ok, enc_shape = name, tuple(z.shape)
                break
            except (ValueError, RuntimeError, IndexError):
                pass
    return train_ok, enc_ok, enc_shape


def main():
    print(f"{'arch':22s} {'T':>3}  {'construct':10s} {'train':6s} "
          f"{'encode':6s} code shape")
    print("-" * 74)
    for arch_name, path, fixedT, _ in PANEL:
        Ts = [1] if fixedT == 1 else [2, 4, 8]
        for T in Ts:
            cls = _import(path)
            kw = dict(d_in=D_IN, d_sae=D_SAE, k_pos=K, T=T)
            try:
                arch = cls(**kw).cuda()
            except Exception as e:
                print(f"{arch_name:22s} {T:>3}  CONSTRUCT FAILED: "
                      f"{type(e).__name__}: {e}")
                continue
            try:
                tr, en, shp = try_shapes(arch, T)
            except Exception as e:
                print(f"{arch_name:22s} {T:>3}  ok         "
                      f"RUN FAILED: {type(e).__name__}: {e}")
                continue
            flag = "" if (tr and en) else "   <-- UNRESOLVED"
            print(f"{arch_name:22s} {T:>3}  ok         {str(tr):6s} "
                  f"{str(en):6s} {shp}{flag}")
            del arch
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
