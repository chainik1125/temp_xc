"""Native-size Nord figures; inference and manuscript integration stay separate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

TEXT = "#2E3440"
GRID = "#E5E9F0"
COLORS = {"txc_base": "#5E81AC", "topk_sae": "#4C566A",
          "tsae_paper": "#D08770", "stacked_sae": "#B48EAD"}
MARKERS = {"txc_base": "o", "topk_sae": "s", "tsae_paper": "^", "stacked_sae": "D"}
STYLES = {"txc_base": "-", "topk_sae": (0, (3, 1.5)),
          "tsae_paper": (0, (1, 1.3)), "stacked_sae": (0, (4, 1.5, 1, 1.5))}
LABELS = {"txc_base": "TXC", "topk_sae": "Shared SAE",
          "tsae_paper": "T-SAE", "stacked_sae": "Stacked SAE"}
MODES = ("historical_raw_C1", "matched_scaled_C1")
BUDGETS = (1, 2, 4, 8, 16, 32)
POOLS = (("last", "Last token"), ("mean", "Mean pool"), ("max", "Max pool"))
HEADLINE = (("txc_base", "native", "TXC"),
            ("topk_sae", "last", "SAE · last"),
            ("topk_sae", "mean", "SAE · mean"),
            ("topk_sae", "max", "SAE · max"),
            ("tsae_paper", "last", "T-SAE · last"),
            ("tsae_paper", "mean", "T-SAE · mean"),
            ("tsae_paper", "max", "T-SAE · max"),
            ("stacked_sae", "position_identity", "Stacked SAE"))


def baseline(root: Path, cells: list[dict]) -> dict:
    """Constant scores have fold AP equal to fold prevalence, averaged equally."""
    cell = next(c for c in cells if (c["result"]["arch"], c["result"]["seed"])
                == ("txc_base", 1))
    payload = cell["result"]
    source = root / cell["source_path"]
    sidecar = payload["oof_predictions"]
    path = source.with_name(sidecar["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sidecar["sha256"]:
        raise ValueError("Baseline OOF file failed its source hash check")
    with np.load(path, allow_pickle=False) as data:
        labels, folds = data["labels"], data["fold_id"]
        if labels.shape != (25204,) or labels.sum() != 3169 or set(folds) != set(range(5)):
            raise ValueError("Baseline cohort/folds differ from the registered protocol")
        prevalences = [float(labels[folds == f].mean()) for f in range(5)]
    return {"value": float(np.mean(prevalences)), "fold_prevalences": prevalences,
            "definition": "unweighted mean of held-out fold class prevalences",
            "source": str(path.relative_to(root)), "sha256": digest}


def render(records: list[dict], output: Path, *, root: Path, cells: list[dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    if len(cells) != 15 or any(not r["complete_seed_set"] for r in records):
        raise ValueError("Paper exports require all 15 completed three-seed cells")
    if any(c["result"]["convergence_warning_count"] for c in cells):
        raise ValueError("Resolve primary probe convergence warnings before paper export")
    if len({c["result"]["provenance"]["evaluator_sha256"] for c in cells}) != 1:
        raise ValueError("Mixed evaluator hashes need explicit review before paper export")
    ref = baseline(root, cells)
    by_key = {(r["arch"], r["d_sae"], r["view"], r["probe_mode"], r["S"]): r for r in records}
    files, receipts = [], []

    def view(arch, pool):
        return "native" if arch == "txc_base" else "position_identity" if arch == "stacked_sae" else pool

    def get(arch, width, readout, mode, budget):
        return by_key[arch, width, readout, mode, budget]

    visible = [r for r in records if r["view"] != "position_identity" or r["arch"] == "stacked_sae"]
    low = min([ref["value"]] + [min(r["mean"]-r["sample_sd"], min(r["seed_values"].values())) for r in visible])
    high = max([r["mean"]+r["sample_sd"] for r in visible] + [max(r["seed_values"].values()) for r in visible])
    # Shared limits across raw/scaled and width analyses; dots/lines do not encode area.
    limits = (max(0, np.floor((low-.008)*100)/100), min(1, np.ceil((high+.008)*100)/100))
    headline_rows = [r for r in visible if r["S"] == 8 and r["d_sae"] == 32768]
    hx = (max(0, np.floor((min(ref["value"], min(r["mean"]-r["sample_sd"] for r in headline_rows))-.01)*100)/100),
          min(1, np.ceil((max(max(r["mean"]+r["sample_sd"], max(r["seed_values"].values())) for r in headline_rows)+.01)*100)/100))

    def dress(ax, orientation="y"):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#D8DEE9")
        ax.spines[["left", "bottom"]].set_linewidth(.6)
        ax.grid(axis=orientation, color=GRID, linewidth=.45)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.3, width=.5, pad=2)

    def save(fig, stem, role, mode):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        minimum = 100.
        for item in fig.findobj(Text):
            if not item.get_visible() or not item.get_text().strip():
                continue
            # Matplotlib retains invisible tick labels outside view limits.
            bbox = item.get_window_extent(renderer)
            if bbox.width == 0 or bbox.height == 0:
                continue
            minimum = min(minimum, item.get_fontsize())
            if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.width+1 or bbox.y1 > fig.bbox.height+1:
                raise ValueError(f"Clipped figure text in {stem}: {item.get_text()!r}")
        if minimum < 7:
            raise ValueError(f"Figure {stem} has text below 7 pt at native paper size")
        entry = {"stem": stem, "role": role, "probe_mode": mode,
                 "width_inches": float(fig.get_figwidth()), "height_inches": float(fig.get_figheight()),
                 "minimum_font_points": minimum, "exports": {}}
        for ext in ("pdf", "svg", "png"):
            path = output / f"{stem}.{ext}"
            kwargs = {"metadata": {"Creator": "Backtracking paper_figures.py", "CreationDate": None,
                                   "ModDate": None}} if ext == "pdf" else {}
            fig.savefig(path, dpi=400, facecolor="white", **kwargs)
            files.append(path.name)
            entry["exports"][ext] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        receipts.append(entry)
        plt.close(fig)

    style = {"font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 8,
             "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
             "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
             "axes.unicode_minus": True, "mathtext.fontset": "dejavusans", "svg.fonttype": "none",
             "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None,
             "figure.facecolor": "white", "axes.facecolor": "white"}
    with plt.rc_context(style):
        for mode in MODES:
            # Exact dimensions of the current NeurIPS half-width C7 figure slot.
            fig, ax = plt.subplots(figsize=(2.585, 1.98))
            fig.subplots_adjust(left=.365, right=.965, bottom=.235, top=.98)
            dress(ax, "x")
            ax.axvline(ref["value"], color="#7A8493", ls=(0, (2.5, 2)), lw=.7, zorder=1)
            for i, (arch, readout, label) in enumerate(HEADLINE):
                row = get(arch, 32768, readout, mode, 8)
                color = COLORS[arch]
                for offset, seed in zip((-.13, 0, .13), (1, 2, 42)):
                    ax.plot(row["seed_values"][str(seed)], i+offset, "o", ms=2.2,
                            color=color, alpha=.52, markeredgewidth=0, zorder=3)
                ax.errorbar(row["mean"], i, xerr=row["sample_sd"], fmt=MARKERS[arch],
                            ms=3.8, color=color, mec="white", mew=.45, lw=1.05,
                            capsize=2, capthick=.65, zorder=4)
            for y in (.5, 3.5, 6.5):
                ax.axhline(y, color=GRID, lw=.45, zorder=0)
            ax.set_yticks(range(len(HEADLINE)), [r[2] for r in HEADLINE])
            ax.tick_params(axis="y", length=0, pad=3)
            ax.set_ylim(7.6, -.6)
            ax.set_xlim(hx)
            ticks = MaxNLocator(nbins=3).tick_values(*hx)
            ax.set_xticks([t for t in ticks if hx[0] <= t <= hx[1]])
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.set_xlabel("Average precision", labelpad=4)
            save(fig, f"detection_headline_{mode}", "S=8 core; existing half-width slot", mode)

            for sensitivity in (False, True):
                fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.4), sharex=True, sharey=True)
                fig.subplots_adjust(left=.097, right=.985, bottom=.24, top=.84, wspace=.15)
                for ax, (pool, _) in zip(axes, POOLS):
                    dress(ax)
                    ax.axhline(ref["value"], color="#7A8493", ls=(0, (2.5, 2)), lw=.7)
                    profiles = [("tsae_paper", 32768), ("tsae_paper", 16384)] if sensitivity else [(a, 32768) for a in COLORS]
                    for arch, width in profiles:
                        rows = [get(arch, width, view(arch, pool), mode, s) for s in BUDGETS]
                        mean = np.array([r["mean"] for r in rows]); sd = np.array([r["sample_sd"] for r in rows])
                        color = COLORS[arch]
                        linestyle = ("-" if width == 32768 else (0, (3, 1.5))) if sensitivity else STYLES[arch]
                        marker = ("^" if width == 32768 else "v") if sensitivity else MARKERS[arch]
                        ax.fill_between(BUDGETS, mean-sd, mean+sd, color=color, alpha=.09, lw=0)
                        ax.plot(BUDGETS, mean, color=color, ls=linestyle, marker=marker, ms=3.2,
                                mfc="white" if width == 16384 else color, mew=.6, lw=1.15, zorder=3)
                    ax.set_xscale("log", base=2)
                    ax.set_xticks(BUDGETS, [str(s) for s in BUDGETS])
                    ax.set_xlim(.88, 36)
                    ax.set_ylim(limits)
                    ticks = MaxNLocator(nbins=4).tick_values(*limits)
                    ax.set_yticks([t for t in ticks if limits[0] <= t <= limits[1]])
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                axes[0].set_ylabel("Average precision", labelpad=4)
                fig.supxlabel("Feature budget $S$", y=.05, fontsize=8)
                if sensitivity:
                    handles = [Line2D([], [], color=COLORS["tsae_paper"], lw=1.15, ls=ls,
                                      marker=m, ms=3.2, mfc=fc, label=label)
                               for ls, m, fc, label in [("-", "^", COLORS["tsae_paper"], "T-SAE 32K"),
                                                       ((0, (3, 1.5)), "v", "white", "T-SAE 16K")]]
                else:
                    handles = [Line2D([], [], color=COLORS[a], lw=1.15, ls=STYLES[a], marker=MARKERS[a],
                                      ms=3.2, label=LABELS[a]) for a in COLORS]
                fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.535, .99), ncol=len(handles),
                           frameon=False, fontsize=7.5, handlelength=2.2, columnspacing=1.35)
                stem = f"detection_{'width_sensitivity' if sensitivity else 'curves'}_{mode}"
                save(fig, stem, "T-SAE width sensitivity" if sensitivity else "32K core budget curves", mode)

    manifest = {"schema": "c7-paper-figures-v1", "status": "complete", "baseline": ref,
                "curve_y_limits": list(limits), "headline_x_limits": list(hx),
                "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "uncertainty": "sample SD across dictionary seeds 1, 2, 42; not a confidence interval",
                "palette": COLORS, "paper_textwidth_inches": 5.5, "figures": receipts}
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    write_captions(output, ref)
    return files


def write_captions(output: Path, reference: dict):
    common = ("Backtracking detection after 300,000 dictionary-training steps for each of three "
              "seeds (1, 2, 42). Average precision is averaged equally across five folds grouped "
              "by normalized problem text: 25,204 sentences from 300 archive IDs representing "
              "213 distinct prompts. Features are selected and probes fitted on training folds "
              "only. The dashed gray reference is the constant-score baseline, the mean "
              f"held-out-fold prevalence ({reference['value']:.4f}). All methods use offsets "
              "-12 through -8. These prompt-grouped results are distinct from the historical "
              "question-ID split. S counts total selected coordinates: in the 32K core comparison, "
              "TXC and shared readouts have 32,768 candidates; independent Stacked SAE has 5 x 32,768 "
              "position-specific candidates. Parameter counts and reconstructed-token exposure "
              "are not matched by equal S or equal optimizer steps.")
    sections = ["# Paper-ready backtracking detection figures\n",
                "The main figure is `detection_headline_matched_scaled_C1.pdf`. Use vector PDF in "
                "LaTeX; SVG is editable and PNG is a 400-dpi preview. No manuscript file is changed. "
                "PDFs are generated at the actual 5.5-inch NeurIPS text width or the existing "
                "2.585 x 1.98-inch C7 half-width slot. Do not shrink the full-width panels into a "
                "half-width slot. Font sizes are 7-8 pt at native size, with embedded TrueType fonts.\n",
                "## Shared caption details\n\n"+common+"\n",
                "## Headline at S=8\n\nLarge symbols and horizontal error bars show the mean and "
                "sample standard deviation across dictionary seeds. Small dots show individual "
                "seeds, with vertical offsets only for visibility. All eight 32K method/readout "
                "comparisons are shown in a fixed order; no pooling rule is selected after seeing "
                "the results. The dots are not bars, so the AP axis is narrowed to the observed range.\n",
                "## Budget curves\n\nLines show the three-seed mean, and shaded bands show sample "
                "standard deviation, not confidence intervals. From left to right, the three panels "
                "use the shared encoder's last-token, mean-pool, and max-pool readouts; TXC and independent "
                "Stacked SAE repeat as fixed references, not independent measurements. Markers "
                "and line patterns identify architectures in grayscale. All curve figures share "
                "the same AP limits.\n",
                "## Width sensitivity\n\nT-SAE 32K and 16K (32,768 versus 16,384 candidates) are compared using identical "
                "seed sets and training duration. Lines and bands indicate the mean and sample SD. "
                "These results do not establish a globally optimal T-SAE configuration.\n",
                "## Probe scaling\n\n`matched_scaled_C1` is the predeclared primary: each feature "
                "is divided by its training-fold population standard deviation, without centering, "
                "before ranking and fitting a C=1 probe. `historical_raw_C1` is a separate raw-feature "
                "sensitivity under the new canonical-prompt split; it is not a historical-result replay.\n",
                "## Scope\n\nThese are detection figures. Steering figures require deferred judge "
                "labels and validation-based magnitude selection; no unjudged steering result is "
                "presented as an effect. Paired prompt-bootstrap CIs, when available, are separate "
                "from the seed SD shown here and condition on fixed trained dictionaries/probes.\n"]
    (output / "CAPTIONS.md").write_text("\n".join(sections))
    tex = "% Copy these PDFs alongside this snippet; preserve native size for readable fonts.\n"
    tex += "% Inside the existing C7 detection subfigure (0.47\\textwidth):\n"
    tex += "\\includegraphics[width=\\linewidth,height=0.22\\textheight,keepaspectratio]{detection_headline_matched_scaled_C1.pdf}\n"
    tex += "% Caption: Detection at S=8; points and bars show the three-seed mean and sample SD.\n\n"
    tex += "% Full-width appendix figures; expand caption using CAPTIONS.md.\n"
    for family in ("curves", "width_sensitivity"):
        tex += f"\\includegraphics[width=\\textwidth]{{detection_{family}_matched_scaled_C1.pdf}}\n"
    (output / "include_figures.tex").write_text(tex)
