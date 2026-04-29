"""CS1 smoke test — dwell-time distribution for active SAE features.

Hypothesis (50%-time TXC-win candidate #1 from
docs/.../2026-04-28-y-orientation.md): TXC features encode
"persistent" concepts that stay on for many consecutive positions
(because the encoder integrates over T tokens), so the dwell-time
distribution per feature should be heavier-tailed for window archs
than for per-token archs of the same dictionary size.

If true, this is a *structural* TXC win: per-token SAE has to
re-detect the same concept at every position, while window archs
fire once on the concept and naturally span its multi-token
extent in their decoder reconstruction.

Smoke-test design:

  - Pull 50 FineWeb-edu passages, tokenize at max_length=256.
  - Forward Gemma-2-2b base, capture L12 residual stream.
  - Encode through TXC T=5 (matryoshka) + T-SAE k=20 + TopKSAE
    (per-token control). Window arch right-edge attribution.
  - For each (passage, feature) cell: compute run-length encoding
    of "active" positions (z > 1e-6) and tabulate dwell-times.
  - Compare distributions: median dwell, fraction of dwell >=3,
    fraction of dwell >=5, mean.

Output:
  results/case_studies/cs1_slow_features/smoke_dwell_distribution.{png,json}
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import OUT_DIR
from experiments.phase7_unification.case_studies._arch_utils import (
    encode_per_position, window_T, _d_sae_of, MLC_CLASSES,
    load_phase7_model_safe as _load_phase7_model,
)
from experiments.phase7_unification.case_studies._paths import (
    CASE_STUDIES_DIR, SUBJECT_MODEL, ANCHOR_LAYER, DEFAULT_D_IN,
)


N_PASSAGES = 50
MAX_LEN = 256
ARCHS = [
    ("topk_sae", "per-token (k=500)"),
    ("tsae_paper_k20", "per-token (k=20)"),
    ("agentic_txc_02", "TXC matryoshka (T=5)"),
    ("phase5b_subseq_h8", "SubseqH8 (T_max=10)"),
]
OUT_SUBDIR = (
    Path("experiments/phase7_unification/results/case_studies/cs1_slow_features")
)


def _pull_fineweb(n: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      streaming=True, split="train")
    out = []
    for row in ds:
        out.append(row["text"])
        if len(out) >= n:
            break
    return out


def _capture_l12(passages: list[str], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  loading {SUBJECT_MODEL} (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        SUBJECT_MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n = len(passages)
    acts = np.zeros((n, MAX_LEN, DEFAULT_D_IN), dtype=np.float16)
    attn = np.zeros((n, MAX_LEN), dtype=np.int8)
    captured = {}
    def hook(_module, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["x"] = h.detach().cpu()
    handle = model.model.layers[ANCHOR_LAYER].register_forward_hook(hook)
    try:
        for start in range(0, n, 8):
            end = min(start + 8, n)
            chunk = passages[start:end]
            enc = tokenizer(chunk, return_tensors="pt", padding="max_length",
                            truncation=True, max_length=MAX_LEN)
            captured.clear()
            with torch.no_grad():
                model(enc["input_ids"].to(device),
                      attention_mask=enc["attention_mask"].to(device))
            h = captured["x"]
            if h.shape[-1] != DEFAULT_D_IN:
                h = h[..., :DEFAULT_D_IN]
            acts[start:end] = h.to(torch.float16).numpy()
            attn[start:end] = enc["attention_mask"].to(torch.int8).numpy()
    finally:
        handle.remove()
        del model
        torch.cuda.empty_cache()
    return acts, attn


def _run_lengths(active_mask: np.ndarray) -> np.ndarray:
    """Per-row run-length encoding of True regions; returns flat array of run lengths."""
    runs: list[int] = []
    for row in active_mask:
        if not row.any():
            continue
        # Find boundaries between False and True.
        diff = np.diff(np.concatenate([[0], row.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        runs.extend((ends - starts).tolist())
    return np.array(runs, dtype=np.int64) if runs else np.array([], dtype=np.int64)


def _dwell_for_arch(arch_id: str, acts_l12: np.ndarray, attn: np.ndarray,
                    device: torch.device) -> dict:
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed42.json"
    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed42.pt"
    meta = json.loads(log_path.read_text())
    src_class = meta["src_class"]
    is_mlc = src_class in MLC_CLASSES
    print(f"  loading {arch_id} ({src_class})...")
    sae, _ = _load_phase7_model(meta, ckpt_path, device)
    T = window_T(sae, src_class, meta)
    d_sae = _d_sae_of(sae, src_class)
    print(f"    T={T} d_sae={d_sae}")

    N, S = acts_l12.shape[:2]
    pos_idx = torch.arange(S, device=device)

    # Encode in chunks to bound GPU memory; collect z > 0 as binary mask per (N, S, d_sae).
    # d_sae=18432 * 256 = ~4.7M booleans per passage = 4.7 MB per passage in bool. 50 passages
    # = 235 MB. Doable on the 46 GB pod RAM. Keep on CPU as bool.
    acts_bool = np.zeros((N, S, d_sae), dtype=bool)
    bs = 8
    for start in range(0, N, bs):
        end = min(start + bs, N)
        x = torch.from_numpy(acts_l12[start:end]).float().to(device)
        z = encode_per_position(sae, src_class, x, T=T)            # (B, S, d_sae)
        m = torch.from_numpy(attn[start:end].astype(np.float32)).to(device)
        if T > 1 and not is_mlc:
            m = m * (pos_idx >= T - 1).float().unsqueeze(0)
        # Apply mask + threshold and move to CPU as bool.
        z_mask = (z > 1e-6) & (m.unsqueeze(-1) > 0)
        acts_bool[start:end] = z_mask.cpu().numpy()
        del z, x, m, z_mask

    del sae
    torch.cuda.empty_cache()

    # For each (passage, feature) compute run-length encoding of active positions.
    # To bound CPU work: only compute on features that are active in >=2 distinct
    # passages with at least one run of length >=1. Sparse features dominate;
    # compute per feature.
    n_active_per_passage = acts_bool.sum(axis=1)                 # (N, d_sae)
    feat_alive_passages = (n_active_per_passage > 0).sum(axis=0) # (d_sae,)
    interesting_feats = np.where(feat_alive_passages >= 2)[0]
    print(f"    {len(interesting_feats)} features active in >=2 passages")

    all_runs: list[int] = []
    per_feat_max_run: list[int] = []
    for j in interesting_feats:
        col = acts_bool[:, :, j]                                 # (N, S)
        runs = _run_lengths(col)
        if runs.size == 0:
            continue
        all_runs.extend(runs.tolist())
        per_feat_max_run.append(int(runs.max()))

    runs_arr = np.array(all_runs, dtype=np.int64)
    per_feat_max_arr = np.array(per_feat_max_run, dtype=np.int64)

    if runs_arr.size == 0:
        summary = {"n_runs": 0, "median": 0.0, "mean": 0.0, "max": 0,
                   "frac_dwell_ge2": 0.0, "frac_dwell_ge3": 0.0,
                   "frac_dwell_ge5": 0.0, "n_features_with_max_dwell_ge3": 0}
    else:
        summary = {
            "n_runs": int(runs_arr.size),
            "median": float(np.median(runs_arr)),
            "mean": float(np.mean(runs_arr)),
            "max": int(runs_arr.max()),
            "frac_dwell_ge2": float((runs_arr >= 2).mean()),
            "frac_dwell_ge3": float((runs_arr >= 3).mean()),
            "frac_dwell_ge5": float((runs_arr >= 5).mean()),
            "n_features_with_max_dwell_ge3": int((per_feat_max_arr >= 3).sum()),
            "n_features_with_max_dwell_ge5": int((per_feat_max_arr >= 5).sum()),
            "n_features_inspected": int(per_feat_max_arr.size),
        }

    return {
        "src_class": src_class, "T": T, "d_sae": d_sae,
        "summary": summary,
        "run_lengths_sample": runs_arr.tolist()[:5000],
    }


def _plot(payload: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plotting.save_figure import save_figure

    palette = {
        "topk_sae": "#1f77b4",
        "tsae_paper_k20": "#d62728",
        "agentic_txc_02": "#2ca02c",
        "phase5b_subseq_h8": "#17becf",
    }
    label = {
        "topk_sae": "TopKSAE per-token (k=500)",
        "tsae_paper_k20": "T-SAE per-token (k=20)",
        "agentic_txc_02": "TXC matryoshka (T=5)",
        "phase5b_subseq_h8": "SubseqH8 (T_max=10)",
    }

    fig, (ax_hist, ax_summary) = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram of dwell times (log y).
    bins = np.arange(1, 21, 1)
    for arch_id in ["topk_sae", "tsae_paper_k20", "agentic_txc_02", "phase5b_subseq_h8"]:
        if arch_id not in payload["archs"]:
            continue
        runs = np.array(payload["archs"][arch_id]["run_lengths_sample"])
        if runs.size == 0:
            continue
        s = payload["archs"][arch_id]["summary"]
        ax_hist.hist(runs, bins=bins, density=True, alpha=0.40, color=palette[arch_id],
                     label=f"{label[arch_id]}  (median={s['median']:.1f}, n={s['n_runs']})")
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("dwell time (consecutive active positions)")
    ax_hist.set_ylabel("density (log)")
    ax_hist.set_title("CS1: dwell-time distribution per active SAE feature")
    ax_hist.legend(loc="upper right", fontsize=8)
    ax_hist.grid(True, ls=":", alpha=0.4)

    # Summary bar chart: frac >= 3, frac >= 5, mean.
    arch_ids = [a for a, _ in ARCHS if a in payload["archs"]]
    width = 0.25
    x = np.arange(len(arch_ids))
    fr3 = [payload["archs"][a]["summary"]["frac_dwell_ge3"] for a in arch_ids]
    fr5 = [payload["archs"][a]["summary"]["frac_dwell_ge5"] for a in arch_ids]
    means = [payload["archs"][a]["summary"]["mean"] / 10 for a in arch_ids]   # scale for plot
    ax_summary.bar(x - width, fr3, width, label="frac dwell >=3")
    ax_summary.bar(x, fr5, width, label="frac dwell >=5")
    ax_summary.bar(x + width, means, width, label="mean dwell / 10")
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels([label[a] for a in arch_ids], rotation=20,
                                ha="right", fontsize=8)
    ax_summary.set_ylabel("fraction or scaled mean")
    ax_summary.set_title("Summary stats per arch")
    ax_summary.legend(loc="upper right", fontsize=8)
    ax_summary.grid(True, ls=":", alpha=0.4, axis="y")

    fig.suptitle(
        f"CS1 smoke — feature dwell time on FineWeb-edu "
        f"({payload['n_passages']} passages, max_len={payload['max_len']})",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, str(out_path))
    plt.close(fig)


def main() -> None:
    OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    print(f"CS1 smoke — pulling {N_PASSAGES} passages from FineWeb-edu...")
    passages = _pull_fineweb(N_PASSAGES)
    print(f"  {len(passages)} passages, mean {np.mean([len(p) for p in passages]):.0f} chars")

    device = torch.device("cuda")
    t0 = time.time()
    acts_l12, attn = _capture_l12(passages, device)
    print(f"  L12 captured in {time.time() - t0:.1f}s; "
          f"shape {acts_l12.shape}; valid tokens {attn.sum()}")

    archs_data = {}
    for arch_id, _label in ARCHS:
        print(f"\n=== {arch_id} ===")
        archs_data[arch_id] = _dwell_for_arch(arch_id, acts_l12, attn, device)
        s = archs_data[arch_id]["summary"]
        print(f"  median dwell={s['median']:.2f}  mean={s['mean']:.2f}  "
              f"frac>=3={s['frac_dwell_ge3']:.3f}  frac>=5={s['frac_dwell_ge5']:.3f}  "
              f"max={s['max']}  n_runs={s['n_runs']}  "
              f"feat_max_dwell>=3: {s['n_features_with_max_dwell_ge3']}/"
              f"{s['n_features_inspected']}")

    payload = {
        "n_passages": N_PASSAGES, "max_len": MAX_LEN,
        "data_source": "HuggingFaceFW/fineweb-edu sample-10BT",
        "archs": archs_data,
    }
    json_path = OUT_SUBDIR / "smoke_dwell_distribution.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {json_path}")
    png_path = OUT_SUBDIR / "smoke_dwell_distribution.png"
    _plot(payload, png_path)
    print(f"wrote {png_path}  (+ thumb)")


if __name__ == "__main__":
    main()
