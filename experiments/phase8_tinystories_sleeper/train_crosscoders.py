"""Train MLC + three TXCs in parallel on the shared activation cache.

All four crosscoders see the same (seq_idx, pos_idx) pairs each step. MLC
consumes the full L-layer stack at pos_idx; each TXC consumes a T-wide
centered window of one layer's residual. Optimizers are independent so the
four runs don't interact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))  # experiment dir — sleeper_utils / sae_models live here

from sae_models import (  # noqa: E402
    MultiDistanceTXC,
    MultiLayerCrosscoder,
    TemporalContrastiveSAE,
    TemporalCrosscoder,
    TopKSAE,
)
from sleeper_utils import (  # noqa: E402
    H8_LAYER_HOOKS,
    MLC_HOOK_NAMES,
    SAE_LAYER_HOOKS,
    TSAE_LAYER_HOOKS,
    TXC_LAYER_HOOKS,
)


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_kv_list(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"expected key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=str(ROOT / "outputs" / "data"))
    parser.add_argument("--output_dir", default=str(ROOT / "outputs" / "data"))
    parser.add_argument("--d_sae", type=int, default=1536)
    parser.add_argument("--k_total", type=int, default=32)
    parser.add_argument("--T", type=int, default=30, help="TXC window length")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_steps", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--normalize_every", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--archs",
        nargs="+",
        default=None,
        help="Subset of architectures to train. Defaults to all 8 (mlc + 3 TXC + 4 SAE).",
    )
    parser.add_argument(
        "--sae_layer_hooks_override",
        nargs="+",
        default=None,
        metavar="tag=hook_name",
        help="Override SAE layer→hook mapping (e.g. sae_layer0=blocks.0.ln1.hook_normalized).",
    )
    parser.add_argument(
        "--txc_layer_hooks_override",
        nargs="+",
        default=None,
        metavar="tag=hook_name",
        help="Override TXC layer→hook mapping.",
    )
    parser.add_argument(
        "--tsae_layer_hooks_override",
        nargs="+",
        default=None,
        metavar="tag=hook_name",
        help="Override T-SAE layer→hook mapping.",
    )
    parser.add_argument(
        "--h8_layer_hooks_override",
        nargs="+",
        default=None,
        metavar="tag=hook_name",
        help="Override H8 layer→hook mapping.",
    )
    parser.add_argument(
        "--tsae_alpha", type=float, default=1.0,
        help="T-SAE contrastive loss weight (paper default 1.0).",
    )
    parser.add_argument(
        "--h8_alpha", type=float, default=1.0,
        help="H8 contrastive loss weight.",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"[train] device={device}")

    print(f"[train] loading activation cache…")
    acts_cache = torch.load(in_dir / "activations_cache.pt", weights_only=True)
    train_acts: torch.Tensor = acts_cache["train"]  # (N_seq, T_seq, L, D) fp16
    N_seq, T_seq, L, D = train_acts.shape
    print(f"[train]   train acts: N={N_seq} T={T_seq} L={L} d_model={D}")

    # Read the hook names the activations were cached at (matches train's layer
    # index mapping). Falls back to the default MLC stack if not recorded.
    tokens_cache = torch.load(in_dir / "tokens_cache.pt", weights_only=True)
    cached_hook_names = tokens_cache["meta"].get("hook_names") or MLC_HOOK_NAMES
    print(f"[train]   hook_names (from cache): {cached_hook_names}")

    # Pre-pad the seq dimension once for efficient window gather.
    # Move padded acts to GPU so per-batch gather is GPU-side.
    left = args.T // 2
    right = args.T - 1 - left
    acts_padded = F.pad(train_acts, (0, 0, 0, 0, left, right), value=0.0)
    if device == "cuda":
        # Keep fp16 on GPU; cast per-batch to fp32 for the crosscoders.
        acts_padded = acts_padded.to(device)
    print(
        f"[train]   padded shape: {tuple(acts_padded.shape)}  "
        f"dtype={acts_padded.dtype}  device={acts_padded.device}"
    )

    # Build the crosscoders + per-layer SAEs.
    torch.manual_seed(args.seed)
    layer_name_to_idx = {name: i for i, name in enumerate(cached_hook_names)}

    txc_layer_hooks = dict(TXC_LAYER_HOOKS)
    txc_layer_hooks.update(_parse_kv_list(args.txc_layer_hooks_override or []))
    sae_layer_hooks = dict(SAE_LAYER_HOOKS)
    sae_layer_hooks.update(_parse_kv_list(args.sae_layer_hooks_override or []))
    tsae_layer_hooks = dict(TSAE_LAYER_HOOKS)
    tsae_layer_hooks.update(_parse_kv_list(args.tsae_layer_hooks_override or []))
    h8_layer_hooks = dict(H8_LAYER_HOOKS)
    h8_layer_hooks.update(_parse_kv_list(args.h8_layer_hooks_override or []))

    # Drop entries whose hook isn't in the cache.
    txc_layer_hooks = {t: h for t, h in txc_layer_hooks.items() if h in layer_name_to_idx}
    sae_layer_hooks = {t: h for t, h in sae_layer_hooks.items() if h in layer_name_to_idx}
    tsae_layer_hooks = {t: h for t, h in tsae_layer_hooks.items() if h in layer_name_to_idx}
    h8_layer_hooks = {t: h for t, h in h8_layer_hooks.items() if h in layer_name_to_idx}
    txc_layer_indices = {t: layer_name_to_idx[h] for t, h in txc_layer_hooks.items()}
    sae_layer_indices = {t: layer_name_to_idx[h] for t, h in sae_layer_hooks.items()}
    tsae_layer_indices = {t: layer_name_to_idx[h] for t, h in tsae_layer_hooks.items()}
    h8_layer_indices = {t: layer_name_to_idx[h] for t, h in h8_layer_hooks.items()}
    print(f"[train]   TXC layer indices : {txc_layer_indices}")
    print(f"[train]   SAE layer indices : {sae_layer_indices}")
    print(f"[train]   T-SAE layer idxs  : {tsae_layer_indices}")
    print(f"[train]   H8 layer idxs     : {h8_layer_indices}")

    all_models: dict[str, torch.nn.Module] = {}
    if L >= 2:
        all_models["mlc"] = MultiLayerCrosscoder(
            d_in=D, d_sae=args.d_sae, L=L, k_total=args.k_total
        ).to(device)
    for tag in txc_layer_hooks:
        all_models[tag] = TemporalCrosscoder(
            d_in=D, d_sae=args.d_sae, T=args.T, k_total=args.k_total
        ).to(device)
    for tag in sae_layer_hooks:
        all_models[tag] = TopKSAE(d_in=D, d_sae=args.d_sae, k=args.k_total).to(device)
    for tag in tsae_layer_hooks:
        all_models[tag] = TemporalContrastiveSAE(
            d_in=D, d_sae=args.d_sae, k=args.k_total, alpha=args.tsae_alpha
        ).to(device)
    for tag in h8_layer_hooks:
        all_models[tag] = MultiDistanceTXC(
            d_in=D, d_sae=args.d_sae, T=args.T, k_total=args.k_total, alpha=args.h8_alpha
        ).to(device)

    requested = set(args.archs) if args.archs else None
    if requested is not None:
        missing = requested - set(all_models)
        if missing:
            raise SystemExit(f"[train] requested archs not available: {sorted(missing)}")
        models = {name: m for name, m in all_models.items() if name in requested}
    else:
        models = all_models

    for name, m in models.items():
        print(f"[train]   {name}: {sum(p.numel() for p in m.parameters()):,} params")

    opts = {name: torch.optim.Adam(m.parameters(), lr=args.lr) for name, m in models.items()}

    gen = torch.Generator().manual_seed(args.seed)
    offsets = torch.arange(args.T, device=acts_padded.device)  # 0..T-1

    # H8 needs an extended gather: anchor + positives at shifts {1, T//4, T//2}.
    # Compute the maximum shift any active H8 model uses, plus the K positive
    # slots, and the (sorted) shift list. All H8 instances share the same T,
    # so they share the same shift schedule.
    h8_shifts: tuple[int, ...] = ()
    h8_max_shift = 0
    if h8_layer_indices:
        sample_h8 = next(iter(all_models[t] for t in h8_layer_indices))
        h8_shifts = sample_h8.shifts
        h8_max_shift = max(h8_shifts) if h8_shifts else 0
    h8_K = len(h8_shifts)

    losses_by_model: dict[str, list[float]] = {name: [] for name in models}

    t0 = time.time()
    for step in range(args.n_steps):
        # Sample (seq, pos) pairs on the same device as acts_padded.
        seq_idx = torch.randint(0, N_seq, (args.batch_size,), generator=gen).to(acts_padded.device)
        pos_idx = torch.randint(0, T_seq, (args.batch_size,), generator=gen).to(acts_padded.device)

        # Gather windows directly from padded acts (on GPU).
        seq_exp = seq_idx.unsqueeze(1).expand(-1, args.T)             # (B, T)
        pos_exp = pos_idx.unsqueeze(1) + offsets.unsqueeze(0)         # (B, T)
        windows = acts_padded[seq_exp, pos_exp]                       # (B, T, L, D)
        # MLC input is window center (left index) == un-padded pos_idx row.
        mlc_x = windows[:, left, :, :].to(dtype=torch.float32)        # (B, L, D)
        txc_batches = {
            tag: windows[:, :, idx, :].to(dtype=torch.float32)        # (B, T, D)
            for tag, idx in txc_layer_indices.items()
        }
        sae_batches = {
            tag: mlc_x[:, idx, :]                                     # (B, D)
            for tag, idx in sae_layer_indices.items()
        }
        # T-SAE: adjacent pair at the chosen layer (center pos_idx, pos_idx+1).
        # Slice from `windows` so the fp16-on-GPU gather is shared.
        if tsae_layer_indices:
            pair_slice = windows[:, left:left + 2, :, :].to(dtype=torch.float32)  # (B, 2, L, D)
            tsae_batches = {
                tag: pair_slice[:, :, idx, :]                                     # (B, 2, D)
                for tag, idx in tsae_layer_indices.items()
            }
        else:
            tsae_batches = {}
        # H8: gather a wider window (anchor + max_shift extra) per H8 sample,
        # then assemble [anchor, pos_s1, pos_s2, ...] of shape (B, 1+K, T, D).
        if h8_layer_indices:
            big_T = args.T + h8_max_shift
            big_offsets = torch.arange(big_T, device=acts_padded.device)
            big_pos = pos_idx.unsqueeze(1) + big_offsets.unsqueeze(0)             # (B, big_T)
            big_seq = seq_idx.unsqueeze(1).expand(-1, big_T)                      # (B, big_T)
            big_windows = acts_padded[big_seq, big_pos].to(dtype=torch.float32)   # (B, big_T, L, D)
            h8_batches = {}
            for tag, idx in h8_layer_indices.items():
                slot_anchor = big_windows[:, :args.T, idx, :]                     # (B, T, D)
                slot_positives = [
                    big_windows[:, s:s + args.T, idx, :] for s in h8_shifts
                ]
                h8_batches[tag] = torch.stack(
                    [slot_anchor] + slot_positives, dim=1
                )                                                                  # (B, 1+K, T, D)
        else:
            h8_batches = {}

        # One step for each model.
        for name, m in models.items():
            if name in tsae_batches:
                # T-SAE: contrastive paired loss
                loss, _ = m.compute_loss(tsae_batches[name])
            elif name in h8_batches:
                # H8-lite: multi-distance contrastive
                loss, _ = m.compute_loss(h8_batches[name])
            else:
                if name == "mlc":
                    x = mlc_x
                elif name in txc_batches:
                    x = txc_batches[name]
                else:
                    x = sae_batches[name]
                x_hat, _ = m(x)
                loss = (x - x_hat).pow(2).sum(dim=-1).mean()
            opts[name].zero_grad(set_to_none=True)
            loss.backward()
            opts[name].step()
            losses_by_model[name].append(loss.item())

        if (step + 1) % args.normalize_every == 0:
            for m in models.values():
                m.normalize_decoder()

        if (step + 1) % args.print_every == 0:
            parts = [
                f"{n}={losses_by_model[n][-1]:.4f}" for n in models
            ]
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed
            print(f"[train] step {step+1}/{args.n_steps} ({steps_per_sec:.1f} it/s) | " + " ".join(parts))

    print(f"[train] done in {time.time() - t0:.1f}s")

    # Save state dicts + config.
    for name, m in models.items():
        state = {key: v.detach().cpu() for key, v in m.state_dict().items()}
        layer_hook = (
            txc_layer_hooks.get(name)
            or sae_layer_hooks.get(name)
            or tsae_layer_hooks.get(name)
            or h8_layer_hooks.get(name)
        )
        layer_idx = (
            txc_layer_indices.get(name)
            or sae_layer_indices.get(name)
            or tsae_layer_indices.get(name)
            or h8_layer_indices.get(name)
        )
        is_window_arch = name in txc_layer_hooks or name in h8_layer_hooks
        cfg = {
            "class_name": type(m).__name__,
            "d_in": D,
            "d_sae": args.d_sae,
            "k_total": args.k_total,
            "T": args.T if is_window_arch else None,
            "L": L if name == "mlc" else None,
            "layer_hook": layer_hook,
            "layer_idx": layer_idx,
            "hook_names": MLC_HOOK_NAMES if name == "mlc" else None,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        }
        if name in tsae_layer_hooks:
            cfg["alpha_contrastive"] = args.tsae_alpha
            cfg["h_prefix"] = m.h
        if name in h8_layer_hooks:
            cfg["alpha_contrastive"] = args.h8_alpha
            cfg["shifts"] = list(m.shifts)
            cfg["loss_weights"] = list(m.weights)
            cfg["h_prefix"] = m.h
        payload = {
            "state_dict": state,
            "config": cfg,
            "final_loss": losses_by_model[name][-1],
            "loss_trajectory": losses_by_model[name][:: max(1, args.n_steps // 200)],
        }
        torch.save(payload, out_dir / f"crosscoder_{name}.pt")
        print(f"[train] saved crosscoder_{name}.pt  final_loss={losses_by_model[name][-1]:.4f}")

    meta = {
        "d_sae": args.d_sae,
        "k_total": args.k_total,
        "T": args.T,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "normalize_every": args.normalize_every,
        "seed": args.seed,
        "txc_layer_indices": txc_layer_indices,
        "final_losses": {n: losses_by_model[n][-1] for n in models},
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
