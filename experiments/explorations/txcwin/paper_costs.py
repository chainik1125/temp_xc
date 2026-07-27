"""Parameter counts, training FLOPs and inference FLOPs for every architecture
in the paper — answering reviewer bbby's question 5.

Parameters are COUNTED by instantiating each registered class at the paper's own
hyperparameters (`configs/archs.yaml`, including its per-section overrides), not
derived by hand, so the numbers cannot drift from the code.

FLOPs are analytic, and the accounting convention is stated rather than assumed:

  * A multiply-accumulate is 2 FLOPs.
  * ENCODE for a window architecture is d_in x d_sae per position in the window:
    all T positions are projected and summed into one shared latent.
  * DECODE is reported two ways, because the honest answer differs:
      dense   d_sae x d_in per output position — what a naive implementation costs
      sparse  k x d_in per output position — what an implementation that exploits
              the sparse code actually costs, and the fair number to quote
  * PER-TOKEN cost depends on how windows are laid down, which the paper does not
    state, so both are reported:
      stride 1  every position gets its own window -> cost per token = window cost
      stride T  windows tile the sequence -> cost per token = window cost / T
    The paper's evaluation reads one code per tile, which implies tiling, so
    stride T is the like-for-like column.
  * The subject model's own forward cost is included as the denominator that
    matters: a dictionary is only expensive relative to the model it reads.

Run:  .venv/bin/python -m experiments.explorations.txcwin.paper_costs
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT_JSON = Path(__file__).resolve().parent / "results" / "paper_costs.json"
OUT_MD = Path(__file__).resolve().parent / "results" / "paper_costs.md"

# subject model, the section that uses it, hidden size, layers, total params
SUBJECTS = {
    "probing": ("google/gemma-2-2b-it", 2304, 26, 2.61e9, 13),
    "backtracking": ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 4096, 32, 8.03e9, 10),
    "em": ("Qwen/Qwen2.5-7B-Instruct", 3584, 28, 7.62e9, 15),
}
# architectures the paper compares, in the order the paper lists them
PANEL = ["topk_sae", "tsae", "txc_base", "mlc"]
TRAINING = {"probing": (20_000, 4096), "backtracking": (25_000, 1024),
            "em": (20_000, 4096)}


def registry():
    return yaml.safe_load((ROOT / "configs" / "archs.yaml").read_text())["archs"]


def hparams_for(spec: dict, section: str) -> dict:
    h = dict(spec["hparams"])
    h.update((spec.get("per_section_hparams") or {}).get(section, {}))
    return h


def count_params(spec: dict, h: dict, d_in: int) -> int | str:
    """Instantiate the registered class and count. Returns a string on failure so
    a missing dependency is visible rather than silently replaced by an estimate."""
    mod, cls = spec["class_path"].split(":")
    try:
        C = getattr(importlib.import_module(mod), cls)
    except Exception as e:
        return f"import failed: {type(e).__name__}"
    kw = {k: v for k, v in h.items()
          if k in ("d_sae", "k_pos", "T", "h_frac", "contrastive_alpha",
                   "auxk_alpha", "n_layers", "center_layer")}
    kw["d_in"] = d_in
    for attempt in (kw, {k: v for k, v in kw.items() if k != "T"}):
        try:
            m = C(**attempt)
            return int(sum(p.numel() for p in m.parameters()))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    return f"construct failed: {err}"


def flops(arch: str, h: dict, d_in: int) -> dict:
    """Analytic FLOPs for one window (2 FLOPs per multiply-accumulate)."""
    d_sae = h["d_sae"]
    k = h.get("k_pos", 20)
    T = int(h.get("T", 1) or 1)
    n_layers = int(h.get("n_layers", 1) or 1)
    if arch == "mlc":
        # reads n_layers layers at ONE position
        enc = 2 * n_layers * d_in * d_sae
        dec_dense = 2 * n_layers * d_sae * d_in
        dec_sparse = 2 * n_layers * k * d_in
        positions = 1
    elif arch in ("stacked_sae", "stacked_batchtopk"):
        # an independent dictionary per position
        enc = 2 * T * d_in * d_sae
        dec_dense = 2 * T * d_sae * d_in
        dec_sparse = 2 * T * k * d_in
        positions = T
    elif arch.startswith("txc"):
        # one shared latent from all T positions, decoded back to all T
        enc = 2 * T * d_in * d_sae
        dec_dense = 2 * T * d_sae * d_in
        dec_sparse = 2 * T * k * d_in
        positions = T
    else:  # per-token: topk_sae, batchtopk_sae, tsae
        enc = 2 * d_in * d_sae
        dec_dense = 2 * d_sae * d_in
        dec_sparse = 2 * k * d_in
        positions = 1
    return {"T": T, "positions_per_window": positions,
            "encode": enc, "decode_dense": dec_dense, "decode_sparse": dec_sparse,
            "window_dense": enc + dec_dense, "window_sparse": enc + dec_sparse,
            "per_token_stride1_sparse": enc + dec_sparse,
            "per_token_strideT_sparse": (enc + dec_sparse) / positions,
            "per_token_strideT_dense": (enc + dec_dense) / positions}


def main() -> None:
    reg = registry()
    rows, out = [], {"convention": {
        "flops_per_mac": 2,
        "subject_forward_per_token": "2 * N_params (standard dense-transformer "
                                     "approximation, attention terms excluded)",
        "training_flops": "3 * forward_per_token * tokens_seen (forward + "
                          "backward), tokens_seen = n_steps * batch_size",
        "note": "stride 1 = every position gets a window; stride T = windows tile "
                "the sequence, which is what the paper's one-code-per-tile "
                "evaluation implies"}, "sections": {}}

    for section, (model, d_in, n_l, n_params, layer) in SUBJECTS.items():
        steps, batch = TRAINING[section]
        fwd_per_tok = 2 * n_params
        sec = {"subject_model": model, "d_in": d_in, "hookpoint_layer": layer,
               "subject_params": n_params,
               "subject_forward_flops_per_token": fwd_per_tok,
               "training": {"n_steps": steps, "batch_size": batch,
                            "tokens_seen": steps * batch},
               "archs": {}}
        for arch in PANEL:
            if arch not in reg:
                sec["archs"][arch] = {"error": "not in configs/archs.yaml"}
                continue
            spec = reg[arch]
            h = hparams_for(spec, section)
            p = count_params(spec, h, d_in)
            f = flops(arch, h, d_in)
            tokens = steps * batch
            train = (3 * f["per_token_strideT_sparse"] * tokens
                     if isinstance(p, int) else None)
            entry = {"hparams": {k: h.get(k) for k in
                                 ("d_sae", "k_pos", "T", "n_layers", "h_frac")
                                 if h.get(k) is not None},
                     "params": p,
                     "params_pct_of_subject": (round(100 * p / n_params, 2)
                                               if isinstance(p, int) else None),
                     "flops": f,
                     "inference_pct_of_subject_forward": round(
                         100 * f["per_token_strideT_sparse"] / fwd_per_tok, 3),
                     "training_flops_estimate": train}
            sec["archs"][arch] = entry
            rows.append((section, model, arch, h, p, f, n_params, fwd_per_tok,
                         train))
        out["sections"][section] = sec

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))

    md = ["# Parameter counts and inference cost per architecture",
          "",
          "Counted by instantiating each registered class at the paper's own "
          "hyperparameters. FLOPs analytic, 2 per multiply-accumulate; decode "
          "quoted for a sparse implementation; per-token cost at stride T "
          "(windows tiling the sequence, which the paper's one-code-per-tile "
          "evaluation implies).", ""]
    for section, (model, d_in, n_l, n_params, layer) in SUBJECTS.items():
        steps, batch = TRAINING[section]
        md += [f"## {section} — {model} (d_model={d_in}, layer {layer}, "
               f"{n_params/1e9:.2f}B params)",
               "",
               f"Training: {steps:,} steps x {batch:,} = "
               f"{steps*batch/1e6:.1f}M tokens seen.", "",
               "| architecture | d_sae | k | T | parameters | % of subject | "
               "inference FLOPs/token | % of subject forward | training FLOPs |",
               "|---|---|---|---|---|---|---|---|---|"]
        for s, m, arch, h, p, f, np_, fwd, train in rows:
            if s != section:
                continue
            pstr = f"{p/1e6:.1f}M" if isinstance(p, int) else str(p)
            pct = f"{100*p/np_:.2f}%" if isinstance(p, int) else "—"
            md.append(
                f"| {arch} | {h.get('d_sae')} | {h.get('k_pos')} | "
                f"{h.get('T', 1)} | {pstr} | {pct} | "
                f"{f['per_token_strideT_sparse']/1e6:.2f}M | "
                f"{100*f['per_token_strideT_sparse']/fwd:.3f}% | "
                + (f"{train/1e15:.2f}P |" if train else "— |"))
        md.append("")
    OUT_MD.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()


def figure(mode: str = "light"):
    """Parameters and inference cost side by side, grouped by subject model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    TH = {"light": dict(bg="#ffffff", ink="#0b0b0b", ink2="#52514e",
                        grid="#e4e6e8",
                        s=["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]),
          "dark": dict(bg="#111a21", ink="#e8eef1", ink2="#a7bac4",
                       grid="#22323d",
                       s=["#3987e5", "#d95926", "#199e70", "#9085e9"])}[mode]
    plt.rcParams.update({
        "figure.facecolor": TH["bg"], "axes.facecolor": TH["bg"],
        "savefig.facecolor": TH["bg"], "text.color": TH["ink"],
        "axes.labelcolor": TH["ink"], "axes.edgecolor": TH["grid"],
        "xtick.color": TH["ink2"], "ytick.color": TH["ink2"],
        "grid.color": TH["grid"], "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False})
    d = json.loads(OUT_JSON.read_text())
    secs = list(d["sections"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=170)
    xs = np.arange(len(secs))
    w = 0.2
    for j, arch in enumerate(PANEL):
        pv, fv = [], []
        for s in secs:
            e = d["sections"][s]["archs"].get(arch, {})
            p = e.get("params")
            pv.append(p / 1e6 if isinstance(p, (int, float)) else 0)
            fv.append(e.get("flops", {}).get("per_token_strideT_sparse", 0) / 1e6)
        for ax, vals in ((axes[0], pv), (axes[1], fv)):
            pos = xs + (j - 1.5) * (w + 0.012)
            ax.bar(pos, vals, w, color=TH["s"][j], zorder=3,
                   edgecolor=TH["bg"], linewidth=1.3,
                   label=arch if ax is axes[0] else None)
            for x, v in zip(pos, vals):
                if v:
                    ax.annotate(f"{v:.0f}", xy=(x, v), xytext=(0, 3),
                                textcoords="offset points", ha="center",
                                fontsize=7.5, color=TH["ink2"])
    for ax, title, ylab in (
            (axes[0], "Parameters — TXC pays 5x for its T encoder/decoder pairs",
             "million parameters"),
            (axes[1], "Inference cost per token — identical to a per-token SAE",
             "million FLOPs per token (sparse decode, windows tiling)")):
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{s}\n{d['sections'][s]['subject_model'].split('/')[-1]}"
                            for s in secs], fontsize=8.5)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", linewidth=0.6)
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].legend(fontsize=9, loc="upper left")
    fig.suptitle("What each architecture in the paper actually costs",
                 x=0.012, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(__file__).resolve().parent / "figs" / f"costs_{mode}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
