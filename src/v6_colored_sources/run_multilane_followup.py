"""Multi-lane Reed-Solomon follow-up: held-out evaluation, clean-signal MSE,
atom-type diagnostics, and a TXC k_total sweep.

Address the four critique points raised after the first multilane run:

1. **Held-out test data** — train the dictionary and the probe on disjoint
   episode pools so probe-Y results at `W <= h` are not contaminated by
   train-set fingerprinting.
2. **Clean-signal MSE** alongside the noisy-input MSE so we can read off
   actual signal recovery rather than noise-fit overfitting.
3. **Atom-type diagnostics** for each TXC decoder atom: max lane-energy
   fraction (a lane-trajectory atom concentrates on one lane and spreads
   across positions) vs max position-energy fraction (a token-template
   atom concentrates on one position and spreads across lanes).
4. **TXC `k_total` sweep ∈ {1, 2, W, m, mW}** to discriminate the
   lane-trajectory hypothesis from a possible token-template alternative.
   If `k_total = W` already reaches the noise floor, the model is
   reconstructing via per-position multi-lane fingerprints, not via
   Reed-Solomon lane trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TXC_DIR = _REPO_ROOT / "temporal_crosscoders"
if str(_TXC_DIR) not in sys.path:
    sys.path.insert(0, str(_TXC_DIR))

from models import TopKSAE  # noqa: E402

from temporal_crosscoders.han_arch.txc_bare_antidead import TXCBareAntidead  # noqa: E402

from .data_adapter import ColoredSourceCache  # noqa: E402
from .multilane_rs import (  # noqa: E402
    MultilaneRSConfig,
    all_lane_atoms,
    generate_multilane_dataset,
    lane_alphabet_lower_bound,
)
from .run_pair_experiment import _train_logistic_probe  # noqa: E402


# ---------- Trainers ---------------------------------------------------------


def _train_topk_sae_tokens(
    cache: ColoredSourceCache, *, d, H, k, n_steps, batch_size, lr, device, log_interval=1000
) -> TopKSAE:
    sae = TopKSAE(d_in=d, d_sae=H, k=k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    t0 = time.time()
    for step in range(n_steps):
        x = cache.sample_windows(batch_size, 1).squeeze(1)
        loss, _, _ = sae(x)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        sae._normalize_decoder()
        if step % log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"    [sae k_pos={k}] step={step:>5d} loss={loss.item():.4f} | "
                f"{(step + 1) / max(elapsed, 1e-6):.1f} it/s",
                flush=True,
            )
    return sae


def _train_txc_global(
    cache: ColoredSourceCache, *, d, H, W, k_window, n_steps, batch_size, lr, device,
    log_interval=1000,
) -> TXCBareAntidead:
    model = TXCBareAntidead(d_in=d, d_sae=H, T=W, k=k_window).to(device)
    with torch.no_grad():
        init_x = cache.sample_windows(min(batch_size * 4, 256), W)
        model.init_b_dec_geometric_median(init_x.float())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(n_steps):
        x = cache.sample_windows(batch_size, W)
        loss, _, _ = model(x)
        opt.zero_grad()
        loss.backward()
        model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model._normalize_decoder()
        if step % log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"    [txc k={k_window} W={W}] step={step:>5d} loss={loss.item():.4f} | "
                f"{(step + 1) / max(elapsed, 1e-6):.1f} it/s",
                flush=True,
            )
    return model


# ---------- Reconstruction eval (noisy + clean) -------------------------------


@torch.no_grad()
def _eval_recon_with_clean(
    model, *, is_txc: bool, noisy_cache: ColoredSourceCache,
    clean_cache: ColoredSourceCache, W: int, n_eval: int,
) -> dict:
    """Evaluate per-token reconstruction MSE on a HELD-OUT batch.

    The model receives `noisy_cache.sample_windows`. We compute:

      mse_noisy:    ||x_noisy - x_hat||^2  averaged per token
      mse_clean:    ||x_clean - x_hat||^2  averaged per token (signal recovery)
      signal_norm_per_token: mean ||x_clean||^2 (≈ 1 for the multilane signal)

    Both `noisy_cache` and `clean_cache` must be aligned: the same
    `(chain, t_start)` indexing produces the same underlying clean signal,
    differing only in noise. We sample windows from `noisy_cache` and pull
    the matching clean windows by reusing the same `(chain, t_start)` pair.
    """
    batch = min(n_eval, 4096)
    n_done, total_se_n, total_se_c, total_clean_norm, total_tok = 0, 0.0, 0.0, 0.0, 0
    while n_done < n_eval:
        b = min(batch, n_eval - n_done)
        # Sample with shared indexing.
        max_start = noisy_cache.chain_length - W
        chain_idx = torch.randint(0, noisy_cache.num_chains, (b,), device=noisy_cache.device)
        start_idx = torch.randint(0, max_start + 1, (b,), device=noisy_cache.device)
        offsets = torch.arange(W, device=noisy_cache.device).unsqueeze(0)
        pos = start_idx.unsqueeze(1) + offsets
        chain_exp = chain_idx.unsqueeze(1).expand(-1, W)
        x_noisy = noisy_cache.act_chains[chain_exp, pos]                   # (b, W, d)
        x_clean = clean_cache.act_chains[chain_exp, pos]                   # (b, W, d)

        if is_txc:
            _, x_hat, _ = model(x_noisy)
        else:
            x_flat = x_noisy.reshape(-1, x_noisy.shape[-1])
            _, x_hat_flat, _ = model(x_flat)
            x_hat = x_hat_flat.reshape(b, W, -1)

        total_se_n += (x_hat - x_noisy).pow(2).sum().item()
        total_se_c += (x_hat - x_clean).pow(2).sum().item()
        total_clean_norm += (x_clean.pow(2).sum()).item()
        total_tok += b * W
        n_done += b
    return {
        "mse_noisy_per_token": total_se_n / total_tok,
        "mse_clean_per_token": total_se_c / total_tok,
        "clean_norm_per_token": total_clean_norm / total_tok,
        "signal_recovery_frac": 1.0 - (total_se_c / total_tok) / (total_clean_norm / total_tok),
    }


# ---------- Atom-type diagnostics --------------------------------------------


@torch.no_grad()
def atom_type_stats(W_dec: torch.Tensor, alphabet: torch.Tensor) -> dict:
    """For each TXC decoder atom (shape `(W, d)`), measure how concentrated
    its energy is across lanes vs across positions.

    Returns per-atom statistics:

      max_lane_frac:  fraction of energy in the top-1 lane.
      max_pos_frac:   fraction of energy in the top-1 position.
      lane_entropy:   normalized entropy of lane-energy distribution
                      (1 = uniform, 0 = single lane).
      pos_entropy:    normalized entropy of position-energy distribution.
      atom_type:      "lane_traj"  if max_lane_frac > 0.7 and max_pos_frac < 0.5,
                      "tok_tmpl"   if max_lane_frac < 0.3 and max_pos_frac > 0.7,
                      "fingerprint" otherwise.
    """
    H, W, d = W_dec.shape
    m, q, _ = alphabet.shape
    # (H, W, m, q) coefficients of W_dec[s, t, :] in lane-symbol basis.
    coefs = torch.einsum("hwd,mqd->hwmq", W_dec, alphabet)
    # Energy per (atom, position, lane), aggregated over the q symbols of that lane.
    e_apl = (coefs ** 2).sum(dim=-1)                                       # (H, W, m)
    lane_e = e_apl.sum(dim=1)                                              # (H, m)
    pos_e = e_apl.sum(dim=2)                                               # (H, W)

    lane_total = lane_e.sum(dim=1).clamp_min(1e-12)
    pos_total = pos_e.sum(dim=1).clamp_min(1e-12)
    lane_frac = lane_e / lane_total.unsqueeze(1)                          # (H, m)
    pos_frac = pos_e / pos_total.unsqueeze(1)                             # (H, W)

    max_lane_frac = lane_frac.max(dim=1).values
    max_pos_frac = pos_frac.max(dim=1).values
    # Normalized entropies in [0, 1].
    lane_log_max = math.log(m) if m > 1 else 1.0
    pos_log_max = math.log(W) if W > 1 else 1.0
    lane_ent = -(lane_frac.clamp_min(1e-12) * lane_frac.clamp_min(1e-12).log()).sum(dim=1) / lane_log_max
    pos_ent = -(pos_frac.clamp_min(1e-12) * pos_frac.clamp_min(1e-12).log()).sum(dim=1) / pos_log_max

    is_lane_traj = (max_lane_frac > 0.7) & (max_pos_frac < 0.5)
    is_tok_tmpl = (max_lane_frac < 0.3) & (max_pos_frac > 0.7)
    is_fp = ~(is_lane_traj | is_tok_tmpl)

    return {
        "max_lane_frac_mean": float(max_lane_frac.mean().item()),
        "max_pos_frac_mean": float(max_pos_frac.mean().item()),
        "lane_entropy_mean": float(lane_ent.mean().item()),
        "pos_entropy_mean": float(pos_ent.mean().item()),
        "frac_lane_traj": float(is_lane_traj.float().mean().item()),
        "frac_tok_tmpl": float(is_tok_tmpl.float().mean().item()),
        "frac_fingerprint": float(is_fp.float().mean().item()),
        "n_atoms": int(H),
    }


# ---------- Stage runner ------------------------------------------------------


_STAGE_GRID = {
    "smoke": dict(h=1, q=11, m=32, W_grid=[1, 2, 3, 4]),
    "main": dict(h=2, q=7, m=32, W_grid=[2, 3, 4, 5]),  # trim W>=5 for compute
}


def _gather_anchor_indices_from(
    data: dict, W: int, *, max_samples: int, device: torch.device, seed: int
) -> dict:
    """Sample `max_samples` anchor positions WITHOUT REPLACEMENT.

    Repetitions in the original `torch.randint`-based sampler caused
    duplicate `(chain, t_start)` pairs to be put into both probe-train
    and probe-val, which trivially leaked the label and inflated probe
    accuracy. Here we enumerate all valid `(chain, t_start)` pairs and
    take a random permutation prefix.
    """
    cfg: MultilaneRSConfig = data["config"]
    n_seq = cfg.n_seq
    max_start = cfg.T_chain - W
    if max_start < 0:
        raise ValueError(f"T_chain={cfg.T_chain} < W={W}")
    n_total = n_seq * (max_start + 1)
    n = min(max_samples, n_total)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n_total, generator=gen)[:n]
    seq = perm // (max_start + 1)
    starts = perm % (max_start + 1)
    return {
        "chain_idx": seq.to(device),
        "t_start": starts.to(device),
        "Y": data["Y"][seq].to(device),
    }


def _windows_from_cache(
    cache: ColoredSourceCache, anchors: dict, W: int
) -> torch.Tensor:
    chains = anchors["chain_idx"]
    starts = anchors["t_start"]
    offsets = torch.arange(W, device=cache.device).unsqueeze(0)
    pos = starts.unsqueeze(1) + offsets
    chain_exp = chains.unsqueeze(1).expand(-1, W)
    return cache.act_chains[chain_exp, pos]


def run_stage(
    *,
    label: str,
    h: int, q: int, m: int, d: int, sigma: float,
    n_seq_train: int, n_seq_test: int, T_chain: int,
    H_txc: int, H_sae: int,
    n_steps: int, batch_size: int, lr: float,
    W_grid: list[int],
    n_probe_samples: int, n_recon_eval: int,
    device: torch.device, out_dir: Path,
    seed_train: int = 0, seed_test: int = 9999,
) -> dict:
    cfg_train = MultilaneRSConfig(
        h=h, q=q, m=m, d=d, sigma=sigma,
        n_seq=n_seq_train, T_chain=T_chain, seed=seed_train,
    )
    print(f"\n=== Stage {label} TRAIN config: {cfg_train} ===")
    train_data = generate_multilane_dataset(cfg_train)
    train_cache = ColoredSourceCache(train_data["x"], device)

    cfg_test = replace(cfg_train, n_seq=n_seq_test, seed=seed_test)
    print(f"--- Held-out TEST config (fresh episodes, shared alphabet): {cfg_test} ---")
    # Reuse the train alphabet so the test data lives in the same subspace
    # the model was trained on.
    test_data = generate_multilane_dataset(cfg_test, alphabet=train_data["alphabet"])
    test_cache_noisy = ColoredSourceCache(test_data["x"], device)
    # Build a clean-signal cache by recomputing x_clean (alphabet @ Q without noise).
    n_t, T, _ = test_data["x"].shape
    Q_t = test_data["Q"].to(device).long()
    alphabet_t = test_data["alphabet"].to(device).float()
    lane_idx = torch.arange(m, device=device).view(1, 1, m).expand(n_t, T, m)
    x_clean_test = (
        alphabet_t[lane_idx, Q_t].sum(dim=2) / math.sqrt(m)
    ).to(torch.float32)
    test_cache_clean = ColoredSourceCache(x_clean_test, device)

    cells: list[dict] = []
    t0 = time.time()
    for W in W_grid:
        if W > T_chain:
            continue
        print(f"\n--- W={W} ---")
        atoms_full = all_lane_atoms(train_data["alphabet"], h, q, W).to(device).float()
        noise_floor = sigma ** 2 * d
        lb_at_km = lane_alphabet_lower_bound(m=m, W=W, k_window=m)
        lb_at_kmW = lane_alphabet_lower_bound(m=m, W=W, k_window=m * W)
        print(
            f"  Reference: noise floor MSE = {noise_floor:.4f}; "
            f"local-alphabet predicted error at k_total=m → {lb_at_km:.3f}, "
            f"at k_total=mW → {lb_at_kmW:.3f}"
        )

        # Held-out probe anchors come from the test dataset.
        anchors_test = _gather_anchor_indices_from(
            test_data, W, max_samples=n_probe_samples, device=device, seed=seed_test + W,
        )

        cell: dict = {
            "W": W,
            "noise_floor_mse_per_token": noise_floor,
            "local_alphabet_lb": {"k_eq_m": lb_at_km, "k_eq_m_times_W": lb_at_kmW},
            "n_test_episodes": n_seq_test,
        }

        # 1. Local SAE: k_pos in {1, m}.
        for k_pos in (1, m):
            print(f"\n    [SAE k_pos={k_pos}]")
            sae = _train_topk_sae_tokens(
                train_cache, d=d, H=H_sae, k=k_pos,
                n_steps=n_steps, batch_size=batch_size, lr=lr, device=device,
            )
            recon = _eval_recon_with_clean(
                sae, is_txc=False, noisy_cache=test_cache_noisy,
                clean_cache=test_cache_clean, W=W, n_eval=n_recon_eval,
            )
            # Probe on held-out anchors.
            x_anch_test = _windows_from_cache(test_cache_noisy, anchors_test, W)
            B = x_anch_test.shape[0]
            with torch.no_grad():
                z_flat = sae.encode(x_anch_test.reshape(B * W, -1)).detach()
            z_concat = z_flat.reshape(B, W * z_flat.shape[-1])
            sae_probe = _train_logistic_probe(
                z_concat, anchors_test["Y"], R=cfg_train.q, device=device,
                seed=seed_test + 200 + 10 * k_pos + W,
            )
            cell[f"sae_k{k_pos}"] = {
                "k_pos": k_pos,
                "k_total_window_equivalent": k_pos * W,
                **recon,
                "probe_val_acc": sae_probe["val_accuracy"],
            }
            print(
                f"      mse_noisy={recon['mse_noisy_per_token']:.4f}  "
                f"mse_clean={recon['mse_clean_per_token']:.4f}  "
                f"signal_rec={recon['signal_recovery_frac']:.3f}  "
                f"probe={sae_probe['val_accuracy']:.3f}"
            )

        # 2. TXC k_total sweep ∈ {1, 2, W, m, mW}, deduped and dropping W=1 specials.
        k_sweep = sorted({1, 2, W, m, m * W} - {0})
        for k_window in k_sweep:
            if k_window <= 0:
                continue
            print(f"\n    [TXC k_total={k_window}]")
            txc = _train_txc_global(
                train_cache, d=d, H=H_txc, W=W, k_window=k_window,
                n_steps=n_steps, batch_size=batch_size, lr=lr, device=device,
            )
            recon = _eval_recon_with_clean(
                txc, is_txc=True, noisy_cache=test_cache_noisy,
                clean_cache=test_cache_clean, W=W, n_eval=n_recon_eval,
            )
            # Lane-level Rec_temp.
            H = txc.W_dec.shape[0]
            flat_learned = txc.W_dec.detach().reshape(H, -1)
            flat_learned = flat_learned / flat_learned.norm(dim=1, keepdim=True).clamp_min(1e-12)
            lane_atoms_flat = atoms_full.reshape(m * atoms_full.shape[1], -1)
            lane_atoms_flat = lane_atoms_flat / lane_atoms_flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            inner = lane_atoms_flat @ flat_learned.T
            rec_temp = float((inner ** 2).max(dim=1).values.mean().item())
            # Probe on held-out anchors.
            x_anch_test = _windows_from_cache(test_cache_noisy, anchors_test, W)
            with torch.no_grad():
                z = txc.encode(x_anch_test).detach()
            txc_probe = _train_logistic_probe(
                z, anchors_test["Y"], R=cfg_train.q, device=device,
                seed=seed_test + 100 + k_window + W,
            )
            atom_diag = atom_type_stats(txc.W_dec.detach(), data_alphabet := train_data["alphabet"].to(device).float())
            cell[f"txc_k{k_window}"] = {
                "k_window": k_window,
                **recon,
                "rec_temp_lane": rec_temp,
                "probe_val_acc": txc_probe["val_accuracy"],
                "atom_diag": atom_diag,
            }
            print(
                f"      mse_noisy={recon['mse_noisy_per_token']:.4f}  "
                f"mse_clean={recon['mse_clean_per_token']:.4f}  "
                f"signal_rec={recon['signal_recovery_frac']:.3f}  "
                f"Rec_temp_lane={rec_temp:.3f}  probe={txc_probe['val_accuracy']:.3f}"
            )
            print(
                f"      atom diag: lane_traj={atom_diag['frac_lane_traj']:.2f}  "
                f"tok_tmpl={atom_diag['frac_tok_tmpl']:.2f}  "
                f"fingerprint={atom_diag['frac_fingerprint']:.2f}  "
                f"max_lane_frac={atom_diag['max_lane_frac_mean']:.2f}  "
                f"max_pos_frac={atom_diag['max_pos_frac_mean']:.2f}"
            )

        cells.append(cell)

    elapsed = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage_label": f"multilane_rs_{label}_followup",
        "config_train": asdict(cfg_train),
        "config_test": asdict(cfg_test),
        "H_txc": H_txc, "H_sae": H_sae,
        "n_steps": n_steps, "batch_size": batch_size, "lr": lr,
        "W_grid": W_grid,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "cells": cells,
    }
    out_path = out_dir / f"multilane_rs_{label}_followup.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path}  (elapsed {elapsed/60:.1f} min)")
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description="Multilane RS follow-up sweep.")
    p.add_argument("--label", choices=["smoke", "main"], required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="results/v6_colored_sources")
    p.add_argument("--n_seq_train", type=int, default=4096)
    p.add_argument("--n_seq_test", type=int, default=2048)
    p.add_argument("--T_chain", type=int, default=8)
    p.add_argument("--n_steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_probe_samples", type=int, default=20000)
    p.add_argument("--n_recon_eval", type=int, default=8000)
    p.add_argument("--seed_train", type=int, default=0)
    p.add_argument("--seed_test", type=int, default=9999)
    p.add_argument("--sigma", type=float, default=0.1)
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    grid = _STAGE_GRID[args.label]
    h, q, m, W_grid = grid["h"], grid["q"], grid["m"], grid["W_grid"]
    d = max(512, m * q)
    H_sae = max(1024, m * q * 2)
    H_txc = max(4096, m * q ** (h + 1))
    if args.label == "main":
        H_txc = max(16384, m * q ** (h + 1))

    T_chain = max(args.T_chain, max(W_grid) + 1)

    run_stage(
        label=args.label,
        h=h, q=q, m=m, d=d, sigma=args.sigma,
        n_seq_train=args.n_seq_train, n_seq_test=args.n_seq_test, T_chain=T_chain,
        H_txc=H_txc, H_sae=H_sae,
        n_steps=args.n_steps, batch_size=args.batch_size, lr=args.lr,
        W_grid=W_grid,
        n_probe_samples=args.n_probe_samples, n_recon_eval=args.n_recon_eval,
        device=device, out_dir=Path(args.out_dir),
        seed_train=args.seed_train, seed_test=args.seed_test,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
