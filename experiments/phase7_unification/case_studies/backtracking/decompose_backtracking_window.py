#!/usr/bin/env python3
"""Window-aware feature-space DoM for TXCBareAntidead-style window encoders.

The per-token decompose treats each cached row as one (d_in,) sample. For
TXC, each "sample" is a (T, d_in) window centred on a position. We encode
windows whose right-edge is at D₊ positions and windows whose right-edge
is at D positions, and rank features by Welch t-statistic on the resulting
(d_sae,) feature activations.

Inputs:
    --txc-ckpt PATH                    TXCBareAntidead ckpt.pt + meta.json
    --activation-cache HOOK            "resid" (default) — the stage-2 cache
                                       for the distilled model that's used by
                                       per-token decompose. We forward windows
                                       from there even if the TXC was trained
                                       on a different hook (e.g. attn) — that
                                       mismatch is intentional: we want to
                                       evaluate the TXC's reading of the
                                       same cache used to define D_+.

Output (`decompose<_suffix>/`):
    feature_stats.npz          per-feature: mean_plus, mean_all, var_*, delta, tstat, ratio
    top_features.json          top-K by t-stat
    decompose_meta.json        config + n_d_plus, n_d_all
    raw_dom.fp16.npy           raw activation-space DoM at the right-edge
                               position (for the raw_dom mode in
                               intervene_backtracking_window).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TQDM_DISABLE", "1")

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.phase7_unification.case_studies.backtracking._paths import (  # noqa: E402
    ANCHOR_LAYER,
    LABELS_DIR,
    RESULTS_DIR,
    ensure_dirs,
)
from experiments.phase7_unification.case_studies.backtracking.intervene_backtracking_window import (  # noqa: E402
    load_txc,
)


def _build_position_lists(trace_ids, offsets, labels_path):
    """Return two lists of (trace_idx, pos_within_trace, global_row_idx)
    tuples — one for D_+, one for D (think-region positions).
    """
    plus: list[tuple[int, int, int]] = []
    all_d: list[tuple[int, int, int]] = []
    id_to_idx = {tid: i for i, tid in enumerate(trace_ids)}
    with labels_path.open() as f:
        for line in f:
            lab = json.loads(line)
            i = id_to_idx.get(lab["trace_id"])
            if i is None:
                continue
            base = int(offsets[i])
            seq_len = int(offsets[i + 1] - offsets[i])
            for p in lab["d_plus_positions"]:
                if 0 <= p < seq_len:
                    plus.append((i, p, base + p))
            think_lo, think_hi = lab["think_lo"], min(lab["think_hi"], seq_len)
            for p in range(think_lo, think_hi):
                all_d.append((i, p, base + p))
    return plus, all_d


def _gather_window(activations, offsets, trace_idx, pos_within, T, d_in):
    """Build the (T, d_in) window whose right-edge is at (trace_idx, pos_within).

    If pos_within < T-1, pad with the prefix of the trace (clamped at 0).
    """
    base = int(offsets[trace_idx])
    end = int(offsets[trace_idx + 1])
    seq_len = end - base
    right = base + pos_within
    left = right - (T - 1)
    if left < base:
        # Need padding: take from left=base and copy the front T-(right-left+1) times
        # Simple choice: clamp left to base (smaller window padded by repeats of position 0)
        prefix_pad = base - left
        left = base
        win_len = right - left + 1
        out = np.empty((T, d_in), dtype=np.float32)
        out[:prefix_pad] = activations[base].astype(np.float32, copy=False)
        out[prefix_pad:] = activations[left:right + 1].astype(np.float32, copy=False)
    else:
        out = activations[left:right + 1].astype(np.float32, copy=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txc-ckpt", required=True)
    parser.add_argument("--cache-suffix", default="", help="distilled-model cache suffix (default '')")
    parser.add_argument("--decompose-suffix", default="txc")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--rank-by", choices=("tstat", "delta", "ratio"), default="tstat")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-pos", type=int, default=None,
                        help="cap the number of positions per group (D_+ and D) for speed")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    out_dir = RESULTS_DIR / f"decompose_{args.decompose_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "feature_stats.npz").exists() and not args.force:
        print(f"[decompose-window] {out_dir} exists; --force to rebuild")
        return

    cache_dir = RESULTS_DIR / (f"cache_l10_{args.cache_suffix}" if args.cache_suffix else "cache_l10")
    activations = np.load(cache_dir / "activations.fp16.npy", mmap_mode="r")
    offsets = np.load(cache_dir / "offsets.npy")
    trace_ids = json.loads((cache_dir / "trace_ids.json").read_text())
    labels_path = LABELS_DIR / "labels.jsonl"

    print(f"[decompose-window] cache {activations.shape}; loading TXC")
    txc, meta = load_txc(Path(args.txc_ckpt), args.device)
    T = int(meta["T"])
    d_sae = int(meta["d_sae"])
    d_in = int(meta["d_in"])

    plus, all_d = _build_position_lists(trace_ids, offsets, labels_path)
    if args.max_pos is not None:
        rng = np.random.default_rng(0)
        if len(plus) > args.max_pos:
            plus = [plus[i] for i in rng.choice(len(plus), args.max_pos, replace=False)]
        if len(all_d) > args.max_pos:
            all_d = [all_d[i] for i in rng.choice(len(all_d), args.max_pos, replace=False)]
    n_plus, n_all = len(plus), len(all_d)
    print(f"[decompose-window] |D_+|={n_plus} |D|={n_all} T={T} d_sae={d_sae}")

    def _stream_moments(positions):
        s = torch.zeros(d_sae, dtype=torch.float64, device=args.device)
        s2 = torch.zeros(d_sae, dtype=torch.float64, device=args.device)
        s_h_right = np.zeros(d_in, dtype=np.float64)  # raw right-edge mean
        n = len(positions)
        for i in range(0, n, args.batch):
            chunk = positions[i:i + args.batch]
            wins = np.stack([
                _gather_window(activations, offsets, p[0], p[1], T, d_in)
                for p in chunk
            ], axis=0)  # (B, T, d_in)
            # accumulate raw right-edge mean
            s_h_right += wins[:, -1, :].astype(np.float64).sum(axis=0)
            x = torch.from_numpy(wins).to(args.device, dtype=torch.float32)
            with torch.no_grad():
                z = txc.encode(x).to(torch.float64)  # (B, d_sae)
            s += z.sum(dim=0)
            s2 += (z * z).sum(dim=0)
        if n == 0:
            zeros = np.zeros(d_sae, dtype=np.float32)
            return zeros, zeros, np.zeros(d_in, dtype=np.float32), 0
        mean = s / n
        var = (s2 - n * mean * mean) / max(n - 1, 1)
        var = torch.clamp(var, min=0.0)
        return (
            mean.cpu().numpy().astype(np.float32),
            var.cpu().numpy().astype(np.float32),
            (s_h_right / n).astype(np.float32),
            n,
        )

    t0 = time.time()
    mean_p, var_p, h_right_p, n_p = _stream_moments(plus)
    print(f"[decompose-window] D_+ moments in {time.time()-t0:.1f}s")
    t0 = time.time()
    mean_a, var_a, h_right_a, n_a = _stream_moments(all_d)
    print(f"[decompose-window] D moments in {time.time()-t0:.1f}s")

    delta = mean_p - mean_a
    se = np.sqrt(var_p / max(n_p, 1) + var_a / max(n_a, 1))
    tstat = np.where(se > 1e-12, delta / np.maximum(se, 1e-12), 0.0).astype(np.float32)
    ratio = mean_p / np.maximum(mean_a, 1e-6)

    np.savez(out_dir / "feature_stats.npz",
             mean_plus=mean_p, mean_all=mean_a, var_plus=var_p, var_all=var_a,
             delta=delta, tstat=tstat, ratio=ratio)
    raw_dom = (h_right_p - h_right_a).astype(np.float16)
    np.save(out_dir / "raw_dom.fp16.npy", raw_dom)

    score = {"tstat": tstat, "delta": delta, "ratio": np.where(mean_a > 0.01, ratio, 0.0)}[args.rank_by]
    top_idx = np.argsort(-score)[: args.top_k]
    top = []
    for rank, j in enumerate(top_idx.tolist()):
        top.append({
            "rank": rank, "feature_idx": int(j),
            "delta": float(delta[j]), "tstat": float(tstat[j]), "ratio": float(ratio[j]),
            "mean_plus": float(mean_p[j]), "mean_all": float(mean_a[j]),
        })
    (out_dir / "top_features.json").write_text(json.dumps(top, indent=2))
    (out_dir / "decompose_meta.json").write_text(json.dumps({
        "kind": "txc_window",
        "txc_ckpt": str(args.txc_ckpt),
        "T": T, "d_in": d_in, "d_sae": d_sae,
        "rank_by": args.rank_by,
        "n_d_plus": n_p, "n_d_all": n_a,
        "raw_dom_norm": float(np.linalg.norm(raw_dom.astype(np.float32))),
    }, indent=2))
    print(f"[decompose-window] top-{args.top_k} → {out_dir / 'top_features.json'}")


if __name__ == "__main__":
    main()
