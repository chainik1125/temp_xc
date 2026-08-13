"""Instrument-sensitivity controls for the w6mix detection pair. In-container.

Decides between two readings of detection_results_w6mix.json (w6mix_dsm
0.208/0.242 sentence S8/S32 PR-AUC with ~849/16384 latents live, vs
w6mix_recon 0.190/0.228 with ~93% of the pool alive, vs raw activations
0.190/0.216):

  A. "surviving DSM latents are disproportionately informative" — survives
     only if recon subsampled to dsm's live-pool size drops well below dsm.
  B. "the task is largely insensitive to the dictionary near a floor" —
     survives if subsampled recon ~= full recon ~= dsm, and especially if a
     random dictionary lands in the same band.

Three control arms, all through the identical capture / example-set / probe
path of run_detection (ONLY the dictionary/feature source changes):

  1. live-pool-matched recon: w6mix_recon's feature matrix restricted to a
     random subset of its live latents, matched to w6mix_dsm's EXACT live
     count on the trace windows (computed in-job), 3 draws. Post-encode
     column masking, not re-TopK: the probe protocol only ever sees the
     feature columns, and dsm's dead columns are all-zero to the probe, so
     matching what the probe can select from is the correct match.
  2. random-dictionary floor: freshly initialised, never-trained
     TopKSAE(T*d=24576, H=16384, k=96) (torch.manual_seed(random_dict_seed)
     before init; training seeded the same construction path), same encode
     and probe pipeline.
  3. label-shuffle floor: w6mix_dsm's features scored against labels permuted
     WITHIN trace (class balance and GroupKFold folds preserved — groups are
     untouched, so the fold split is identical), 3 draws, both example sets.

Reference arms (w6mix_dsm, w6mix_recon, full pool) are re-scored in the same
job so every number in the table shares one capture.

Writes /vol/backtracking_eval/detection_controls.json (committed after every
arm; the volume artifact is the launch evidence).
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from experiments.backtracking_detection_dsm import detect_core as dc

W6MIX_FILES = [("recon", "txc_recon/recon_s0.pt"),
               ("dsm", "txc_dsm/dsm_s0.pt")]
H = 16384
K = 96


def shuffle_within_groups(y: np.ndarray, groups: np.ndarray,
                          seed: int) -> np.ndarray:
    """Permute labels independently inside each trace. Per-trace and overall
    class balance are preserved exactly; groups (and therefore the GroupKFold
    fold assignment) are untouched."""
    rng = np.random.default_rng(seed)
    y2 = y.copy()
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        y2[m] = rng.permutation(y2[m])
    return y2


def run(device: str = "cuda", vol: str = "/vol",
        traces_path: str = "/work/data/traces.json",
        labels_path: str = "/work/data/sentence_labels.json",
        limit_traces: int | None = None, tag_suffix: str = "",
        commit_cb=None, w6_dir: str = "/vol/txc_w6_mix",
        n_draws: int = 3, subsample_seed: int = 20260812,
        shuffle_seed: int = 777, random_dict_seed: int = 0) -> dict:
    import torch

    from experiments.diffusion_txc.topk_vs_topkdiff.sae import TopKSAE

    t0 = time.time()
    out: dict = {"protocol": {
        "window_offsets": dc.WIN_OFF,
        "probe": "l1-logistic C=1 liblinear, 5-fold GroupKFold by trace, "
                 "top-S by train-fold Welch t-stat, S in {8,32} "
                 "(detect_core.probe_cv, unchanged)",
        "score": "window latent |z| (w6_flat time-major flatten, unchanged)",
        "labels": "results/ward_backtracking/sentence_labels.json",
        "w6_dir": w6_dir,
        "controls": {
            "recon_subsampled": {
                "n_draws": n_draws, "subsample_seed_base": subsample_seed,
                "rule": "np.random.default_rng(subsample_seed + draw).choice("
                        "recon_live_idx, size=n_live_dsm, replace=False); "
                        "one latent subset per draw, shared across example "
                        "sets; post-encode column restriction",
            },
            "random_dict": {
                "seed": random_dict_seed,
                "rule": f"torch.manual_seed({random_dict_seed}); "
                        f"TopKSAE(24576, {H}, {K}) untrained",
            },
            "label_shuffle": {
                "n_draws": n_draws, "shuffle_seed_base": shuffle_seed,
                "rule": "labels permuted within trace via "
                        "np.random.default_rng(shuffle_seed + draw), "
                        "independently per example set; groups untouched",
            },
        },
    }}

    cache, offsets, trace_meta, cap_meta = dc.capture_traces(
        traces_path, labels_path, device=device, limit_traces=limit_traces)
    out["capture"] = cap_meta
    out["capture"]["n_traces"] = len(trace_meta)

    ex = dc.collect_examples(trace_meta)
    sets = {}
    for name, examples in ex.items():
        y = np.array([e[2] for e in examples])
        groups = np.array([e[0] for e in examples])
        sets[name] = {"examples": examples, "y": y, "groups": groups}
        out.setdefault("example_sets", {})[name] = {
            "n": int(len(y)), "n_pos": int(y.sum()),
            "positive_base_rate": float(y.mean()),
            "n_groups": int(len(set(groups.tolist()))),
        }
        print(f"[examples/{name}] n={len(y)} pos={int(y.sum())} "
              f"base_rate={y.mean():.4f}", flush=True)

    # w6 dicts read resid_L10 only; the ln1 half of the cache is not needed
    windows = {n: dc.gather_windows(cache["resid"], offsets,
                                    sets[n]["examples"]) for n in sets}
    print("[windows/resid] " + " ".join(
        f"{n}={tuple(windows[n].shape)}" for n in sets), flush=True)

    # --- dictionaries -----------------------------------------------------
    models, metas = {}, {}
    for kind, fname in W6MIX_FILES:
        path = str(pathlib.Path(w6_dir) / fname)
        models[kind], metas[kind] = dc.load_w6_dict(path, kind, device)

    chk = dc.w6_nmse_both_orders(models["recon"], windows["sentence"], device)
    out["w6_ordering_check"] = chk
    print(f"[w6 ordering] NMSE time_major={chk['nmse_time_major']:.4f} "
          f"dim_major={chk['nmse_dim_major']:.4f}", flush=True)
    if not chk["nmse_time_major"] < 0.9 * chk["nmse_dim_major"]:
        _flush(out, vol, tag_suffix, commit_cb)
        raise RuntimeError("w6 flatten-order gate failed; not scoring")

    torch.manual_seed(random_dict_seed)
    rand = TopKSAE(dc.D_MODEL * len(dc.WIN_OFF), H, K).to(device).eval()
    for p in rand.parameters():
        p.requires_grad_(False)
    models["random"] = rand
    metas["random"] = {"arch": "w6_flat", "kind": "random_untrained",
                       "d_in": dc.D_MODEL * len(dc.WIN_OFF), "d_sae": H,
                       "k": K, "init_seed": random_dict_seed}

    # --- live pools on the trace windows (stride-1, time-major) -----------
    live = {}
    out["live_pools"] = {}
    n_win = cache["resid"].shape[0] - len(dc.WIN_OFF) + 1
    for kind in ("recon", "dsm", "random"):
        c = dc.w6_fire_counts(models[kind], cache["resid"], device)
        live[kind] = np.where(c > 0)[0]
        out["live_pools"][kind] = {
            "n_live": int(len(live[kind])),
            "dead_never_fired": int((c == 0).sum()),
            "dead_fraction": float((c == 0).mean()),
            "n_windows": int(n_win), "unit": "stride1_window",
        }
        print(f"[live/{kind}] {len(live[kind])}/{H} live "
              f"(dead {float((c == 0).mean()):.4f})", flush=True)
        _flush(out, vol, tag_suffix, commit_cb)
    del cache  # capture no longer needed; windows carry everything scored

    n_live_dsm = len(live["dsm"])
    draws = []
    for i in range(n_draws):
        rng = np.random.default_rng(subsample_seed + i)
        draws.append(np.sort(rng.choice(live["recon"], size=n_live_dsm,
                                        replace=False)))
    out["protocol"]["controls"]["recon_subsampled"]["n_live_dsm_matched"] = \
        int(n_live_dsm)

    # --- arms: set-outer so each feature matrix is built once --------------
    out["arms"] = {}

    def put(tag, sname, res, extra=None, meta=None):
        a = out["arms"].setdefault(tag, dict(meta or {}))
        a[sname] = res
        if extra:
            a.update(extra)
        print(f"[arm {tag}/{sname}] S8 PR={res['prauc_S8']:.4f} "
              f"ROC={res['rocauc_S8']:.4f} | S32 PR={res['prauc_S32']:.4f}",
              flush=True)
        _flush(out, vol, tag_suffix, commit_cb)

    for sname in sets:
        y, groups = sets[sname]["y"], sets[sname]["groups"]
        F = {kind: dc.window_features(models[kind], "w6_flat",
                                      windows[sname], device)
             for kind in ("dsm", "recon", "random")}

        # references, re-scored under this capture
        put("w6mix_dsm", sname, dc.probe_cv(F["dsm"], y, groups),
            {f"{sname}_l0_mean": float((F["dsm"] > 0).sum(1).mean())},
            meta=metas["dsm"])
        put("w6mix_recon", sname, dc.probe_cv(F["recon"], y, groups),
            {f"{sname}_l0_mean": float((F["recon"] > 0).sum(1).mean())},
            meta=metas["recon"])

        # control 1: live-pool-matched recon, n_draws subsets
        for i, cols in enumerate(draws):
            Fi = np.ascontiguousarray(F["recon"][:, cols])
            put(f"w6mix_recon_sub{n_live_dsm}_draw{i}", sname,
                dc.probe_cv(Fi, y, groups),
                {"subsample_seed": int(subsample_seed + i),
                 "n_latents": int(n_live_dsm),
                 f"{sname}_l0_mean": float((Fi > 0).sum(1).mean())})
            del Fi

        # control 2: random untrained dictionary
        put("random_dict", sname, dc.probe_cv(F["random"], y, groups),
            {f"{sname}_l0_mean": float((F["random"] > 0).sum(1).mean())},
            meta=metas["random"])

        # control 3: label shuffle on the strongest arm (dsm)
        for i in range(n_draws):
            ys = shuffle_within_groups(y, groups, shuffle_seed + i)
            assert int(ys.sum()) == int(y.sum())
            put(f"w6mix_dsm_labelshuffle_draw{i}", sname,
                dc.probe_cv(F["dsm"], ys, groups),
                {"shuffle_seed": int(shuffle_seed + i)})
        del F

    out["_runtime_s"] = round(time.time() - t0, 1)
    _flush(out, vol, tag_suffix, commit_cb)
    print(f"ALL DONE in {out['_runtime_s']}s", flush=True)
    return out


def _flush(out: dict, vol: str, tag_suffix: str = "", commit_cb=None) -> None:
    d = pathlib.Path(vol) / "backtracking_eval"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"detection_controls{tag_suffix}.json").write_text(
        json.dumps(out, indent=1, default=str))
    if commit_cb is not None:
        commit_cb()
