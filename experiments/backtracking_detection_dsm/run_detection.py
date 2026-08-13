"""Driver for the c7 backtracking-detection comparison. Runs in-container.

Arms
  new (ln1_L10, per-token TopK, H=16384, k=64, 20k steps, FineWeb):
      recon_s2 recon_s3 dsm_s2 dsm_s3      <- the dictionaries under test
  baseline (ln1_L10, the paper's own SAE arm):
      topk_sae__ln1_L10__k64__s42
  orientation (resid_L10 -- a different hookpoint, included only to show the
  paper's SAE-vs-TXC ordering reproduces inside this same harness):
      txc__resid_L10__k16__s42, txc_h13__resid_L10__k16__s42
  floor:
      raw ln1 activations, 6 positions stacked
  w6 trio (resid_L10, run with w6_only=True into detection_results_w6.json):
      w6_recon w6_dsm w6_bayes -- flat window-6 dictionaries (T*d=24576 in,
      H=16384; recon/dsm TopK k=96, bayes BayesGateSAE at u=0, g>0.5) trained
      by psc_train_sae.py --window 6, loaded from the Modal volume. Same
      resid_L10 window cache, score = window latent |z|.

Every arm goes through the identical example sets, window, probe and metrics.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from experiments.backtracking_detection_dsm import detect_core as dc

NEW_REPO = "dmanningcoe/diffusion-topk-saes"
NEW_SUBDIR = "llama31-8b-ln1L10-20k"
NEW_ARMS = [("recon", 2), ("recon", 3), ("dsm", 2), ("dsm", 3)]
STAGEB_REPO = "aniketdesh/ward-stage-b-dictionaries"
STAGEB_CELLS = [
    "topk_sae__ln1_L10__k64__s42",       # the paper's SAE baseline, same hookpoint
    "txc__resid_L10__k16__s42",          # TXC-base, resid hookpoint
    "txc_h13__resid_L10__k16__s42",      # TXC-pro family, resid hookpoint
]
W6_FILES = [                             # on the diffusion-txc volume
    ("recon", "txc_recon/recon_s0.pt"),
    ("dsm", "txc_dsm/dsm_s0.pt"),
    ("bayes", "txc_bayes/bayes_gate_s0.pt"),
]


def run(device: str = "cuda", vol: str = "/vol",
        traces_path: str = "/work/data/traces.json",
        labels_path: str = "/work/data/sentence_labels.json",
        limit_traces: int | None = None, tag_suffix: str = "",
        commit_cb=None, modes: tuple[str, ...] = ("topk",),
        calib_rows: int = 50_000, w6_dir: str = "",
        w6_only: bool = False) -> dict:
    import torch
    from huggingface_hub import hf_hub_download

    t_start = time.time()
    out: dict = {"protocol": {
        "window_offsets": dc.WIN_OFF,
        "neg_buffer": dc.NEG_BUFFER, "neg_per_pos": dc.NEG_PER_POS,
        "probe": "l1-logistic C=1 liblinear, 5-fold GroupKFold by trace, "
                 "top-S by train-fold Welch t-stat, S in {8,32}",
        "score": "max-pool |z| over the 6 window positions (per-token dicts); "
                 "window latent |z| (T=6 window dicts, incl. the w6_flat "
                 "time-major-flattened dicts)",
        "labels": "results/ward_backtracking/sentence_labels.json "
                  "(Haiku-4.5 sentence judge, label_sentences.py)",
        "trace_model": "DeepSeek-R1-Distill-Llama-8B generations",
        "activation_model": "base Llama-3.1-8B (paper's own train/eval mismatch)",
        "modes": list(modes),
        "gate": "post-hoc rate-calibrated per-latent thresholds "
                "(posthoc_gate_evals.calibrate + GateSAE, unmodified); "
                f"calibrated on the tail {calib_rows:,} trace activations, "
                "label-free, rate-matched to native TopK on that same slice",
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

    # windows per hookpoint, built once and shared by every arm on that hook
    windows = {}
    for hook in ("ln1", "resid"):
        windows[hook] = {n: dc.gather_windows(cache[hook], offsets,
                                              sets[n]["examples"])
                         for n in sets}
        print(f"[windows/{hook}] " + " ".join(
            f"{n}={tuple(windows[hook][n].shape)}" for n in sets), flush=True)

    out["arms"] = {}

    def score_arm(tag, model, arch, hook, meta, do_dead):
        if "topk" in modes:
            r = {"hookpoint": hook, "arch": arch, "gate": "native_topk", **meta}
            for sname in sets:
                F = dc.window_features(model, arch, windows[hook][sname], device)
                r[sname] = dc.probe_cv(F, sets[sname]["y"], sets[sname]["groups"])
                nz = (F > 0).sum(1)          # per-token arms: pooled-union L0
                r[f"{sname}_l0_mean"] = float(nz.mean())
                r[f"{sname}_l0_std"] = float(nz.std())
                del F
            if do_dead:
                r["dead_on_traces"] = dc.dead_on_traces(model, arch, cache[hook],
                                                        device)
            out["arms"][tag] = r
            print(f"[arm {tag}] sentence S8 PR={r['sentence']['prauc_S8']:.4f} "
                  f"ROC={r['sentence']['rocauc_S8']:.4f} | far S8 "
                  f"PR={r['far']['prauc_S8']:.4f} ROC={r['far']['rocauc_S8']:.4f}"
                  f" | L0 {r['sentence_l0_mean']:.1f}"
                  + (f" | dead={r['dead_on_traces']['dead_fraction']:.4f}"
                     if do_dead else ""), flush=True)
            _flush(out, vol, tag_suffix, commit_cb)

        if "gate" in modes:
            from experiments.diffusion_txc.topk_vs_topkdiff.posthoc_gate_evals \
                import GateSAE, calibrate

            adapter, is_window = dc.gate_adapter(model, arch)
            X_cal = dc.calibration_matrix(cache[hook], is_window, calib_rows)
            theta, cstats = calibrate(adapter, X_cal, device)
            del X_cal
            gate = GateSAE(adapter, theta)
            g = {"hookpoint": hook, "arch": arch, "gate": "threshold_swap",
                 **meta, "calibration": cstats}
            for sname in sets:
                F, l0m, l0s = dc.gate_window_features(
                    gate, is_window, windows[hook][sname], device)
                g[sname] = dc.probe_cv(F, sets[sname]["y"],
                                       sets[sname]["groups"])
                g[f"{sname}_gate_l0_mean"] = l0m
                g[f"{sname}_gate_l0_std"] = l0s
                del F
            out["arms"][f"{tag}__gate"] = g
            print(f"[arm {tag}__gate] sentence S8 "
                  f"PR={g['sentence']['prauc_S8']:.4f} "
                  f"ROC={g['sentence']['rocauc_S8']:.4f} | far S8 "
                  f"PR={g['far']['prauc_S8']:.4f} | L0 "
                  f"{g['sentence_gate_l0_mean']:.1f}+-"
                  f"{g['sentence_gate_l0_std']:.1f} "
                  f"(topk {cstats['topk_mean_l0']:.1f})", flush=True)
            _flush(out, vol, tag_suffix, commit_cb)

    # --- floor: raw activations, 6 positions stacked (gate-independent) ----
    if "topk" in modes and not w6_only:
        r = {"hookpoint": "ln1", "arch": "raw", "gate": "n/a"}
        for sname in sets:
            W = windows["ln1"][sname]
            F = W.reshape(W.shape[0], -1).float().numpy()
            r[sname] = dc.probe_cv(F, sets[sname]["y"], sets[sname]["groups"])
            del F
        out["arms"]["raw_ln1_stacked"] = r
        print(f"[arm raw_ln1_stacked] sentence S8 "
              f"PR={r['sentence']['prauc_S8']:.4f} "
              f"| far S8 PR={r['far']['prauc_S8']:.4f}", flush=True)
        _flush(out, vol, tag_suffix, commit_cb)

    if not w6_only:
        # --- the four new dictionaries -----------------------------------
        for arm, seed in NEW_ARMS:
            tag = f"new_{arm}_s{seed}"
            ckpt = hf_hub_download(NEW_REPO, f"{NEW_SUBDIR}/{arm}_s{seed}.pt")
            final = json.loads(pathlib.Path(hf_hub_download(
                NEW_REPO, f"{NEW_SUBDIR}/{arm}_s{seed}_final.json")).read_text())
            model, meta = dc.load_new_dict(ckpt, device)
            meta.update({"train_arm": arm, "train_seed": seed,
                         "train_model": final.get("model"),
                         "train_hook": final.get("hook"),
                         "train_steps": final.get("steps"),
                         "train_tokens": final.get("tokens"),
                         "train_final_nmse": final.get("final_nmse"),
                         "train_final_dead_frac": final.get("final_dead_frac")})
            score_arm(tag, model, "topk_sae_flat", "ln1", meta, do_dead=True)
            del model
            torch.cuda.empty_cache()

        # --- Stage B originals -------------------------------------------
        for cell in STAGEB_CELLS:
            model, meta = dc.load_stageb_dict(cell, STAGEB_REPO, device)
            hook = "ln1" if meta["hook_component"] == "ln1" else "resid"
            flat = meta["arch"] == "topk_sae"
            score_arm(f"stageb_{cell}", model, meta["arch"], hook, meta,
                      do_dead=flat)
            del model
            torch.cuda.empty_cache()

    # --- w6 trio: window-6 flat dictionaries from the diffusion arm -------
    # Same resid_L10 window cache as the Stage B window dicts; score is the
    # window latent |z|. Flatten order is verified empirically on w6_recon
    # before any arm is scored (see detect_core.w6_nmse_both_orders).
    if w6_dir:
        for kind, fname in W6_FILES:
            tag = f"w6_{kind}"
            path = str(pathlib.Path(w6_dir) / fname)
            if not pathlib.Path(path).exists():
                print(f"[{tag}] SKIP: {path} not present", flush=True)
                continue
            model, meta = dc.load_w6_dict(path, kind, device)
            jl = pathlib.Path(path[:-3] + ".jsonl")
            if jl.exists():
                try:
                    last = json.loads(jl.read_text().strip().splitlines()[-1])
                    meta.update({
                        "train_step": last.get("step"),
                        "train_nmse_clean": last.get("nmse_clean"),
                        "train_dead_frac": last.get("dead_frac"),
                        "train_l0_gate_half": last.get("l0_gate_half")})
                except Exception as e:                        # noqa: BLE001
                    print(f"[w6 {kind}] jsonl meta unreadable: {e}", flush=True)
            if kind == "recon":
                chk = dc.w6_nmse_both_orders(model, windows["resid"]["sentence"],
                                             device)
                out["w6_ordering_check"] = chk
                print(f"[w6 ordering] NMSE time_major="
                      f"{chk['nmse_time_major']:.4f} dim_major="
                      f"{chk['nmse_dim_major']:.4f} (n={chk['n_rows']})",
                      flush=True)
                if not chk["nmse_time_major"] < 0.9 * chk["nmse_dim_major"]:
                    _flush(out, vol, tag_suffix, commit_cb)
                    raise RuntimeError(
                        "w6 flatten-order gate failed: time_major NMSE "
                        f"{chk['nmse_time_major']:.4f} not clearly below "
                        f"dim_major {chk['nmse_dim_major']:.4f}; not scoring")
            if kind == "bayes":
                gm, gs = dc.w6_gate_l0(model, windows["resid"]["sentence"],
                                       device)
                meta["gate_l0_mean_sentence"] = gm
                meta["gate_l0_std_sentence"] = gs
                print(f"[w6_bayes] gate L0 (g>0.5, u=0) on sentence windows: "
                      f"{gm:.1f}+-{gs:.1f}", flush=True)
            score_arm(tag, model, "w6_flat", "resid", meta, do_dead=True)
            del model
            torch.cuda.empty_cache()

    out["_runtime_s"] = round(time.time() - t_start, 1)
    _flush(out, vol, tag_suffix, commit_cb)
    print(f"ALL DONE in {out['_runtime_s']}s", flush=True)
    return out


def _flush(out: dict, vol: str, tag_suffix: str = "", commit_cb=None) -> None:
    """Write after every arm and commit, so a timeout still leaves the arms
    that already finished on the volume."""
    d = pathlib.Path(vol) / "backtracking_eval"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"detection_results{tag_suffix}.json").write_text(
        json.dumps(out, indent=1, default=str))
    if commit_cb is not None:
        commit_cb()
