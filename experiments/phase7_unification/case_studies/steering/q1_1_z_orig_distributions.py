"""Q1.1 — distribution of `z[j*]_orig` magnitudes across the 6 shortlisted archs.

Verifies Dmitry's magnitude-scale hypothesis (window-arch encoder activations
are ~5x larger than per-token archs because the encoder integrates over T
tokens). For each (arch, concept) pair we already have `j*` (the
lift-selected best feature) from `feature_selection.json`; this script
re-runs the encoder over the 30-concept x 5-example probe set and records
`z[j*]` at every content token (right-edge only for window archs, all
content tokens for per-token + MLC archs).

Output:
  results/case_studies/steering_magnitude/q1_1_z_orig_distributions.json
    {
      "n_concepts": 30,
      "n_examples_per_concept": 5,
      "archs": {
        "<arch_id>": {
          "src_class": ..., "T": ..., "d_sae": ...,
          "is_window": bool, "is_mlc": bool,
          "per_concept_z": {concept_id: [list of z[j*] floats]},
          "summary": {
            "n_tokens": int,
            "median": float, "mean": float,
            "iqr_25_75": [q25, q75], "p10_p90": [p10, p90],
            "max": float,
          },
        },
        ...
      },
      "ratios_to_tsae_paper_k20": {arch_id: {"median": ratio, "mean": ratio}}
    }
  results/case_studies/steering_magnitude/q1_1_z_orig_distributions.png
    Overlay KDE + per-arch median markers, shared log-x axis.
    Saved with thumbnail via src.plotting.save_figure.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import OUT_DIR, MLC_LAYERS
from experiments.phase7_unification.case_studies._arch_utils import (
    encode_per_position, window_T, _d_sae_of, MLC_CLASSES,
    load_phase7_model_safe as _load_phase7_model,
)
from experiments.phase7_unification.case_studies._paths import (
    CASE_STUDIES_DIR, SUBJECT_MODEL, ANCHOR_LAYER, DEFAULT_D_IN,
)
from experiments.phase7_unification.case_studies.steering.concepts import CONCEPTS
from experiments.phase7_unification.case_studies.steering.select_features import (
    _capture_l12_activations, _capture_multilayer_activations,
)


SHORTLIST = [
    "topk_sae",
    "tsae_paper_k500",
    "tsae_paper_k20",
    "mlc_contrastive_alpha100_batchtopk",
    "agentic_txc_02",
    "phase5b_subseq_h8",
]
REFERENCE_ARCH = "tsae_paper_k20"  # for cross-arch median ratios

OUT_SUBDIR = CASE_STUDIES_DIR / "steering_magnitude"
ACTS_CACHE = OUT_SUBDIR / "_l12_acts_cache.npz"


def _flat_sentences() -> tuple[list[str], list[int], list[str]]:
    sentences: list[str] = []
    origins: list[int] = []
    concept_ids: list[str] = [c["id"] for c in CONCEPTS]
    for ci, c in enumerate(CONCEPTS):
        for s in c["examples"]:
            sentences.append(s)
            origins.append(ci)
    return sentences, origins, concept_ids


def _capture_or_load(force_recapture: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (acts_l12, acts_l10_l14, attn). Caches L12 + L10-14 to disk."""
    OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    sentences, _origins, _concept_ids = _flat_sentences()
    if ACTS_CACHE.exists() and not force_recapture:
        with np.load(ACTS_CACHE, allow_pickle=False) as z:
            return z["l12"], z["mlc"], z["attn"]

    print(f"  loading subject model {SUBJECT_MODEL} (bf16) for one-shot capture...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    subject = AutoModelForCausalLM.from_pretrained(
        SUBJECT_MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    subject.eval()
    for p in subject.parameters():
        p.requires_grad_(False)
    device = torch.device("cuda")

    print(f"  capturing L12 + L10-L14 over {len(sentences)} concept-sentences...")
    t0 = time.time()
    acts_mlc, attn = _capture_multilayer_activations(
        sentences, subject, tokenizer, device, batch_size=32,
    )
    print(f"    L10-L14 capture done in {time.time() - t0:.1f}s; shape {acts_mlc.shape}")
    # L12 is at index 2 of the (L10, L11, L12, L13, L14) cube.
    l12_idx = MLC_LAYERS.index(ANCHOR_LAYER)
    acts_l12 = acts_mlc[:, :, l12_idx, :].copy()    # (N, S, d_in)

    np.savez_compressed(ACTS_CACHE, l12=acts_l12, mlc=acts_mlc, attn=attn)
    print(f"    cached -> {ACTS_CACHE}")

    del subject
    torch.cuda.empty_cache()
    gc.collect()
    return acts_l12, acts_mlc, attn


def _z_orig_for_arch(
    arch_id: str, acts_l12: np.ndarray, acts_mlc: np.ndarray, attn: np.ndarray,
    origins: list[int], concept_ids: list[str], batch_size: int = 16,
) -> dict:
    """Run encoder for one arch and gather z[j*] at every content token,
    grouped per concept."""
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed42.json"
    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed42.pt"
    if not ckpt_path.exists() or not log_path.exists():
        raise FileNotFoundError(f"missing ckpt or log for {arch_id}")
    meta = json.loads(log_path.read_text())
    src_class = meta["src_class"]
    is_mlc = src_class in MLC_CLASSES

    fs_path = (CASE_STUDIES_DIR / "steering" / arch_id / "feature_selection.json")
    if not fs_path.exists():
        raise FileNotFoundError(f"missing feature_selection for {arch_id}")
    fs = json.loads(fs_path.read_text())
    j_star_per_concept = {cid: int(fs["concepts"][cid]["best_feature_idx"])
                          for cid in concept_ids}

    print(f"  loading {arch_id} ckpt ({src_class})...")
    device = torch.device("cuda")
    sae, _ = _load_phase7_model(meta, ckpt_path, device)
    T = window_T(sae, src_class, meta)
    d_sae = _d_sae_of(sae, src_class)

    acts_in = acts_mlc if is_mlc else acts_l12
    N = acts_in.shape[0]
    S = acts_in.shape[1]
    pos_idx = torch.arange(S, device=device)

    per_concept_z: dict[str, list[float]] = {cid: [] for cid in concept_ids}
    print(f"    encoding (T={T}, d_sae={d_sae}) ...")
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = torch.from_numpy(acts_in[start:end]).float().to(device)
        z = encode_per_position(sae, src_class, x, T=T)            # (B, S, d_sae)
        m = torch.from_numpy(attn[start:end].astype(np.float32)).to(device)
        if T > 1 and not is_mlc:
            m = m * (pos_idx >= T - 1).float().unsqueeze(0)
        for i, ex_idx in enumerate(range(start, end)):
            ci = origins[ex_idx]
            cid = concept_ids[ci]
            j = j_star_per_concept[cid]
            mask_row = m[i] > 0
            if mask_row.any():
                vals = z[i, mask_row, j].detach().cpu().numpy().astype(np.float64)
                per_concept_z[cid].extend([float(v) for v in vals])
        del z, x, m

    del sae
    torch.cuda.empty_cache()
    gc.collect()

    # Per-arch summary across all concept tokens.
    pooled = np.array([v for vals in per_concept_z.values() for v in vals],
                      dtype=np.float64)
    if pooled.size == 0:
        summary = {"n_tokens": 0, "median": 0.0, "mean": 0.0,
                   "iqr_25_75": [0.0, 0.0], "p10_p90": [0.0, 0.0], "max": 0.0}
    else:
        summary = {
            "n_tokens": int(pooled.size),
            "median": float(np.median(pooled)),
            "mean":   float(np.mean(pooled)),
            "iqr_25_75": [float(np.percentile(pooled, 25)),
                          float(np.percentile(pooled, 75))],
            "p10_p90":   [float(np.percentile(pooled, 10)),
                          float(np.percentile(pooled, 90))],
            "max": float(pooled.max()),
        }

    return {
        "src_class": src_class, "T": T, "d_sae": d_sae,
        "is_window": (T > 1) and (not is_mlc),
        "is_mlc": is_mlc,
        "j_star_per_concept": j_star_per_concept,
        "per_concept_z": per_concept_z,
        "summary": summary,
    }


def _plot(payload: dict, out_path: Path) -> None:
    """Overlay KDE of pooled |z[j*]| per arch, with median markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plotting.save_figure import save_figure

    fig, (ax_kde, ax_box) = plt.subplots(1, 2, figsize=(13, 5),
                                         gridspec_kw={"width_ratios": [3, 2]})

    palette = {
        "topk_sae": "#1f77b4",
        "tsae_paper_k500": "#ff7f0e",
        "tsae_paper_k20": "#d62728",
        "mlc_contrastive_alpha100_batchtopk": "#9467bd",
        "agentic_txc_02": "#2ca02c",
        "phase5b_subseq_h8": "#17becf",
    }
    label = {
        "topk_sae": "TopKSAE (per-token, k=500)",
        "tsae_paper_k500": "T-SAE (per-token, k=500)",
        "tsae_paper_k20": "T-SAE (per-token, k=20)",
        "mlc_contrastive_alpha100_batchtopk": "MLC contrastive (5-layer)",
        "agentic_txc_02": "TXC matryoshka (T=5)",
        "phase5b_subseq_h8": "SubseqH8 (T_max=10)",
    }
    box_data: list[np.ndarray] = []
    box_labels: list[str] = []
    box_colors: list[str] = []
    for arch_id, info in payload["archs"].items():
        # Pool tokens, drop zeros (post-TopK / threshold inactive). Distribution is
        # over ACTIVE values of z[j*]; including zeros would be dominated by
        # inactive positions and hide the magnitude story.
        pooled = np.array([v for vals in info["per_concept_z"].values()
                           for v in vals], dtype=np.float64)
        active = pooled[pooled > 1e-6]
        if active.size < 5:
            continue

        # KDE in log-space (z values span ~3 decades).
        log_active = np.log10(active)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(log_active, bw_method="scott")
        xs = np.linspace(log_active.min() - 0.5, log_active.max() + 0.5, 256)
        density = kde(xs)
        ax_kde.fill_between(10 ** xs, density, alpha=0.20, color=palette[arch_id])
        ax_kde.plot(10 ** xs, density, color=palette[arch_id], lw=2,
                    label=f"{label[arch_id]}  (median {np.median(active):.2f}, n={active.size})")
        ax_kde.axvline(np.median(active), color=palette[arch_id], ls="--", lw=1, alpha=0.6)

        box_data.append(active)
        box_labels.append(label[arch_id])
        box_colors.append(palette[arch_id])

    ax_kde.set_xscale("log")
    ax_kde.set_xlabel("z[j*]_orig magnitude (active tokens only, log)")
    ax_kde.set_ylabel("density (over log10 z)")
    ax_kde.set_title("Q1.1: per-arch distribution of selected feature's z magnitude")
    ax_kde.legend(loc="upper left", fontsize=8)
    ax_kde.grid(True, ls=":", alpha=0.4)

    bp = ax_box.boxplot(box_data, vert=False, labels=box_labels, showfliers=False,
                        patch_artist=True)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    ax_box.set_xscale("log")
    ax_box.set_xlabel("z[j*]_orig magnitude (log)")
    ax_box.set_title("Box plot (no outliers)")
    ax_box.grid(True, ls=":", alpha=0.4, axis="x")

    fig.suptitle(f"Q1.1 — z[j*]_orig across 6 archs, "
                 f"{payload['n_concepts']} concepts × "
                 f"{payload['n_examples_per_concept']} examples",
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, str(out_path))
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="+", default=SHORTLIST)
    p.add_argument("--force-recapture", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    sentences, origins, concept_ids = _flat_sentences()
    print(f"Q1.1 — {len(args.archs)} archs, "
          f"{len(concept_ids)} concepts × {len(CONCEPTS[0]['examples'])} examples")

    acts_l12, acts_mlc, attn = _capture_or_load(args.force_recapture)

    archs_data: dict[str, dict] = {}
    for arch_id in args.archs:
        print(f"\n=== {arch_id} ===")
        archs_data[arch_id] = _z_orig_for_arch(
            arch_id, acts_l12, acts_mlc, attn, origins, concept_ids,
            batch_size=args.batch_size,
        )
        s = archs_data[arch_id]["summary"]
        print(f"  summary: n_tokens={s['n_tokens']}  "
              f"median={s['median']:.3f}  "
              f"iqr={s['iqr_25_75'][0]:.3f}-{s['iqr_25_75'][1]:.3f}  "
              f"max={s['max']:.2f}")

    # Cross-arch ratios vs reference.
    ratios = {}
    if REFERENCE_ARCH in archs_data:
        ref_med = max(archs_data[REFERENCE_ARCH]["summary"]["median"], 1e-9)
        ref_mean = max(archs_data[REFERENCE_ARCH]["summary"]["mean"], 1e-9)
        for arch_id, info in archs_data.items():
            ratios[arch_id] = {
                "median_to_ref": info["summary"]["median"] / ref_med,
                "mean_to_ref":   info["summary"]["mean"] / ref_mean,
            }

    payload = {
        "n_concepts": len(concept_ids),
        "n_examples_per_concept": len(CONCEPTS[0]["examples"]),
        "concepts": concept_ids,
        "reference_arch": REFERENCE_ARCH,
        "archs": archs_data,
        "ratios_to_reference": ratios,
    }

    json_path = OUT_SUBDIR / "q1_1_z_orig_distributions.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {json_path}")

    png_path = OUT_SUBDIR / "q1_1_z_orig_distributions.png"
    _plot(payload, png_path)
    print(f"wrote {png_path}  (+ thumb)")

    # Summary table.
    print("\n  median z[j*] per arch (active tokens only):")
    print(f"  {'arch':<48}  {'median':>8}  {'mean':>8}  {'ratio_to_ref':>12}")
    for arch_id, info in archs_data.items():
        s = info["summary"]
        r = ratios.get(arch_id, {}).get("median_to_ref", float("nan"))
        print(f"  {arch_id:<48}  {s['median']:>8.3f}  {s['mean']:>8.3f}  {r:>12.2f}")


if __name__ == "__main__":
    main()
