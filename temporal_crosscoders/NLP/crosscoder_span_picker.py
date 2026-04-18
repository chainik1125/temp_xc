"""Crosscoder-specific span-weighted picker using decoder-based
concentration.

Crosscoder's feat_acts shape is (B, d_sae) — no T dim — so activation-
based concentration is undefined. The right architectural metric is
DECODER-BASED concentration: W_dec[f, t, :] norm per position.

For mass, we can't use per-feature activation mass for features beyond
the scan's top-300 because the scan only saves top-300. Instead we
use total-decoder-norm as a decoder-side proxy for "how much this
feature contributes overall." This is defensible because Crosscoder
reconstruction is entirely mediated by the decoder.

Outputs top-N feature IDs ranked by (1 - decoder_conc) * decoder_total_norm.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from temporal_crosscoders.NLP.autointerp import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer-key", default="resid_L25")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--active-scan",
                    help="Path to a scan__crosscoder JSON. If given, "
                    "restrict picker to features that appear in that scan "
                    "(i.e. that actually fire), and use scan's mass rather "
                    "than decoder total norm as the mass factor.")
    args = ap.parse_args()

    print(f"loading {args.ckpt}")
    model = load_model(
        ckpt_path=args.ckpt, model_type="crosscoder",
        subject_model="gemma-2-2b-it", k=args.k, T=args.T,
    ).cpu()

    with torch.no_grad():
        W = model.W_dec.data.float()     # (d_sae, T, d_in)
        e = W.norm(dim=-1)               # (d_sae, T)  per-position norm
        total = e.sum(dim=-1)            # (d_sae,)    total decoder norm
        conc = e.max(dim=-1).values / total.clamp(min=1e-12)
        peak_pos = e.argmax(dim=-1)

    d_sae = W.shape[0]

    # Optional: filter to features that actually fire (and use scan mass
    # instead of decoder norm). Decoder norm alone picks dead features.
    active_mass = None
    active_feats = None
    if args.active_scan:
        import json as _json
        scan = _json.load(open(args.active_scan))
        active_mass = torch.zeros(d_sae)
        for fid, rec in scan["features"].items():
            active_mass[int(fid)] = float(rec["mass"])
        active_feats = set(int(fid) for fid in scan["features"])
        mass_factor = active_mass
    else:
        mass_factor = total

    span_weighted = (1.0 - conc) * mass_factor
    # Zero out features we can't verify are active (so they don't get picked)
    if active_feats is not None:
        mask = torch.zeros(d_sae, dtype=torch.bool)
        for fi in active_feats:
            mask[fi] = True
        span_weighted = torch.where(mask, span_weighted, torch.tensor(-1.0))

    order = torch.argsort(span_weighted, descending=True)
    top_feats = order[: args.top_n].tolist()

    out = {
        "arch": "crosscoder",
        "ckpt": args.ckpt,
        "layer_key": args.layer_key,
        "k": args.k, "T": args.T,
        "top_n": args.top_n,
        "top_span_weighted": [
            {
                "feat_idx": int(fi),
                "concentration": float(conc[fi].item()),
                "decoder_total_norm": float(total[fi].item()),
                "span_weighted": float(span_weighted[fi].item()),
                "peak_position": int(peak_pos[fi].item()),
            }
            for fi in top_feats
        ],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")

    # summary
    print()
    print(f"{'rank':>5} {'feat':>5} {'conc':>7} {'total_norm':>12} {'span_w':>12}")
    for rank, fi in enumerate(top_feats[:8], 1):
        print(f"{rank:>5d} {fi:>5d} {conc[fi].item():>7.3f} "
              f"{total[fi].item():>12.3f} {span_weighted[fi].item():>12.3f}")


if __name__ == "__main__":
    main()
