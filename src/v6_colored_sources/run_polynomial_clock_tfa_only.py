"""Add TFA + TXC k-sweep to existing polynomial-clock stage results.

Avoids redundantly retraining the SAE / TXC-global k=1 / Bhalla TSAE
runs that already exist in
`results/v6_colored_sources/polynomial_clock_h{H}_q{Q}.json`. This script:

    1. Loads the existing JSON for one stage.
    2. Regenerates the dataset using the same seed (so the data is
       byte-identical to the previous run).
    3. For each cell (each W in W_grid):
       - Trains TFA (k=20, AdamW+cosine, pos enc) and probes its codes.
       - Trains TXC-global at k_total ∈ {2, 5, 10} (window-level TopK
         budgets above the proposal's k=1) and probes their codes +
         computes Rec_temp / Rec_local.
    4. Writes ``tfa_probe`` and ``txc_k{2,5,10}_probe`` /
       ``txc_k{2,5,10}_rec_temp`` etc. into each cell.

Usage:
    python -m src.v6_colored_sources.run_polynomial_clock_tfa_only --stage 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TXC_DIR = _REPO_ROOT / "temporal_crosscoders"
if str(_TXC_DIR) not in sys.path:
    sys.path.insert(0, str(_TXC_DIR))

from .data_adapter import ColoredSourceCache  # noqa: E402
from .polynomial_clock import (  # noqa: E402
    PolynomialClockConfig,
    all_polynomial_atoms,
    generate_polynomial_clock_dataset,
)
from .run_pair_experiment import _train_logistic_probe  # noqa: E402
from .run_polynomial_clock import (  # noqa: E402
    _gather_anchor_indices,
    _tsae_latents_at,
    _txc_global_latents_at,
    rec_local_alphabet,
    rec_temp_polynomial,
)
from .train_runner import TrainConfig, train_tfa  # noqa: E402

import torch.nn as nn  # noqa: E402

from models import TopKSAE  # noqa: E402
from temporal_crosscoders.han_arch.txc_bare_antidead import TXCBareAntidead  # noqa: E402


_STAGE_GRID = {
    1: dict(h=1, q=31, W_grid=[1, 2, 3, 4], H=1024),
    2: dict(h=2, q=11, W_grid=[1, 2, 3, 4, 5], H=2048),
    3: dict(h=3, q=7, W_grid=[1, 2, 3, 4, 5, 6], H=4096),
}


def _train_sae_inline(
    cache: ColoredSourceCache, *, d: int, H: int, k: int,
    n_steps: int, batch_size: int, lr: float, device: torch.device,
) -> TopKSAE:
    sae = TopKSAE(d_in=d, d_sae=H, k=k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for step in range(n_steps):
        x = cache.sample_windows(batch_size, 1).squeeze(1)
        loss, _, _ = sae(x)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        sae._normalize_decoder()
        if step % 1000 == 0:
            print(f"    [sae k={k}] step={step:>5d} loss={loss.item():.4f}", flush=True)
    return sae


def _train_txc_global_inline(
    cache: ColoredSourceCache, *, d: int, H: int, W: int, k_window: int,
    n_steps: int, batch_size: int, lr: float, device: torch.device,
) -> TXCBareAntidead:
    model = TXCBareAntidead(d_in=d, d_sae=H, T=W, k=k_window).to(device)
    with torch.no_grad():
        init_x = cache.sample_windows(min(batch_size * 4, 256), W)
        model.init_b_dec_geometric_median(init_x.float())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(n_steps):
        x = cache.sample_windows(batch_size, W)
        loss, _, _ = model(x)
        opt.zero_grad()
        loss.backward()
        model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model._normalize_decoder()
        if step % 1000 == 0:
            print(
                f"    [txc_global k={k_window} W={W}] step={step:>5d} loss={loss.item():.4f}",
                flush=True,
            )
    return model


def _sae_concat_latents(sae: TopKSAE, cache: ColoredSourceCache, anchors: dict, W: int):
    chains = anchors["chain_idx"]
    starts = anchors["t_start"]
    offsets = torch.arange(W, device=cache.device).unsqueeze(0)
    pos = starts.unsqueeze(1) + offsets
    chain_exp = chains.unsqueeze(1).expand(-1, W)
    x = cache.act_chains[chain_exp, pos]                 # (B, W, d)
    B = x.shape[0]
    flat = x.reshape(B * W, x.shape[-1])
    with torch.no_grad():
        z = sae.encode(flat).detach()                    # (B*W, H)
    return z.reshape(B, -1)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Add TFA results to an existing polynomial-clock stage JSON."
    )
    p.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--results_dir", type=str, default="results/v6_colored_sources",
    )
    p.add_argument("--n_seq", type=int, default=4096)
    p.add_argument("--T_chain", type=int, default=16)
    p.add_argument("--n_steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_probe_samples", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    grid = _STAGE_GRID[args.stage]
    h, q, W_grid, H = grid["h"], grid["q"], grid["W_grid"], grid["H"]
    d = max(64, q * 2)

    json_path = Path(args.results_dir) / f"polynomial_clock_h{h}_q{q}.json"
    print(f"Loading {json_path}")
    with open(json_path) as f:
        payload = json.load(f)

    cfg = PolynomialClockConfig(
        h=h, q=q, d=d, sigma=0.1,
        n_seq=args.n_seq, T_chain=args.T_chain, seed=args.seed,
    )
    print(f"Regenerating data with config={cfg}")
    data = generate_polynomial_clock_dataset(cfg)
    cache = ColoredSourceCache(data["x"], device)

    cells_by_W = {c["W"]: c for c in payload["cells"]}

    SAE_K_SWEEP_REQ = [2, 5, 10]
    TXC_K_SWEEP = [2, 5, 10]

    # SAE has H = q latents. TopK requires k <= H, so drop any sweep value
    # that exceeds q (only matters for Stage 3 where q = 7 < 10).
    SAE_K_SWEEP = [k for k in SAE_K_SWEEP_REQ if k <= q]
    if SAE_K_SWEEP != SAE_K_SWEEP_REQ:
        print(
            f"Note: dropping SAE k values > q={q} from the sweep "
            f"({sorted(set(SAE_K_SWEEP_REQ) - set(SAE_K_SWEEP))})."
        )

    # Train one SAE per k value in the sweep on iid tokens.
    sae_models_by_k = {}
    rec_local_by_k = {}
    for k in SAE_K_SWEEP:
        print(f"\n--- regular TopKSAE k={k} (iid tokens) ---")
        sae_k = _train_sae_inline(
            cache, d=d, H=q, k=k,
            n_steps=args.n_steps, batch_size=args.batch_size, lr=args.lr,
            device=device,
        )
        rec_local = rec_local_alphabet(sae_k.W_dec.detach(), data["alphabet"].to(device).float())
        rec_local_by_k[k] = rec_local
        sae_models_by_k[k] = sae_k
        print(f"    Rec_local (alphabet) at k={k}: {rec_local:.3f}")

    payload.setdefault("sae_k_sweep_rec_local", {})
    for k, v in rec_local_by_k.items():
        payload["sae_k_sweep_rec_local"][str(k)] = v

    t0 = time.time()
    for W in W_grid:
        cell = cells_by_W[W]
        anchors = _gather_anchor_indices(
            data, W, max_samples=args.n_probe_samples,
            device=device, seed=args.seed + W,
        )
        atoms = all_polynomial_atoms(data["alphabet"], h, q, W).to(device).float()

        # SAE k-sweep: window-concat probes per k.
        for k in SAE_K_SWEEP:
            if W < 2:
                # At W=1 the "concat" probe is the same as single-position; still record it.
                pass
            sae_k_X = _sae_concat_latents(sae_models_by_k[k], cache, anchors, W)
            sae_k_probe = _train_logistic_probe(
                sae_k_X, anchors["Y"], R=cfg.q, device=device,
                seed=args.seed + 700 + 10 * k + W,
            )
            cell[f"sae_k{k}_window_probe"] = sae_k_probe
            print(f"    SAE k={k} concat probe W={W}: val={sae_k_probe['val_accuracy']:.3f}")

        if W < 2:
            print(f"\n--- W={W}: skipping TFA / TXC k-sweep (need W>=2) ---")
            cell["tfa_probe"] = None
            for k in TXC_K_SWEEP:
                cell[f"txc_k{k}_probe"] = None
                cell[f"txc_k{k}_rec_local"] = None
                cell[f"txc_k{k}_rec_temp"] = None
            continue
        print(f"\n--- W={W} ---")

        tfa_model = train_tfa(
            cache=cache, W=W, k=20, H=H, d=d,
            use_pos_encoding=True,
            device=device,
            train_cfg=TrainConfig(
                n_steps=args.n_steps,
                batch_size=args.batch_size, lr=args.lr,
            ),
        )
        tfa_X = _tsae_latents_at(tfa_model, cache, anchors, W)
        tfa_probe = _train_logistic_probe(
            tfa_X, anchors["Y"], R=cfg.q, device=device, seed=args.seed + 600 + W,
        )
        print(f"    TFA (k=20, AdamW+cosine, pos_enc)  val={tfa_probe['val_accuracy']:.3f}")
        cell["tfa_probe"] = tfa_probe

        # TXC k-sweep at the window level.
        alphabet = data["alphabet"].to(device).float()
        for k in TXC_K_SWEEP:
            txc_k = _train_txc_global_inline(
                cache, d=d, H=H, W=W, k_window=k,
                n_steps=args.n_steps, batch_size=args.batch_size, lr=args.lr,
                device=device,
            )
            txc_k_X = _txc_global_latents_at(txc_k, cache, anchors, W)
            txc_k_probe = _train_logistic_probe(
                txc_k_X, anchors["Y"], R=cfg.q, device=device,
                seed=args.seed + 800 + 10 * k + W,
            )
            rec_local = rec_local_alphabet(
                txc_k.decoder_dirs_averaged.detach(), alphabet,
            )
            rec_temp = rec_temp_polynomial(txc_k.W_dec.detach(), atoms)
            cell[f"txc_k{k}_probe"] = txc_k_probe
            cell[f"txc_k{k}_rec_local"] = rec_local
            cell[f"txc_k{k}_rec_temp"] = rec_temp
            print(
                f"    TXC k={k} W={W}: val={txc_k_probe['val_accuracy']:.3f} "
                f"Rec_local={rec_local:.3f} Rec_temp={rec_temp:.3f}"
            )

    payload["cells"] = [cells_by_W[c["W"]] for c in payload["cells"]]
    payload["tfa_added"] = True
    payload["tfa_n_steps"] = args.n_steps
    payload["sae_k_sweep"] = SAE_K_SWEEP
    payload["txc_k_sweep"] = TXC_K_SWEEP

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nUpdated {json_path}  (TFA addition took {(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
