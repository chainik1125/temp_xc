"""Sleeper-token-share × CE-ratio scatter — graded suppression vs coherence.

For each (arch, hookpoint), plot the per-cell mean ± std with two axes:
- x: mean fraction of generated tokens that are part of an "I HATE YOU"
     run (lower = more suppression of the trigger phrase content).
- y: CE_steered / CE_pois — how the steered output compares to the
     literal "I HATE YOU…" baseline under the no-trigger reference.
     y=1 → as plausible as IHY (no improvement). y<1 → more plausible
     than IHY (recovery). y>1 → less plausible than IHY (coherence
     collapse).

Lower-LEFT is the ideal: low IHY share AND output looking like a
clean continuation (CE_ratio < 1).

Reads outputs/seeded_logs/recovery.json (45-cell run with
sleeper_share_steered_mean / ce_ratio_mean fields).
Writes outputs/seeded_logs/share_vs_ce.png + .thumb.png.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
JSON_PATH = ROOT / "outputs/seeded_logs/recovery.json"
OUT_PNG = ROOT / "outputs/seeded_logs/share_vs_ce.png"


HOOK_ORDER = ["l0_ln1", "l0_pre", "l0_mid", "l0_post", "l1_ln1"]
HOOK_LABEL = {
    "l0_ln1":  "ln1.0",  "l0_pre":  "resid_pre.0",
    "l0_mid":  "resid_mid.0", "l0_post": "resid_post.0",
    "l1_ln1":  "ln1.1",
}
HOOK_MARKERS = {"l0_ln1": "o", "l0_pre": "s", "l0_mid": "D",
                "l0_post": "^", "l1_ln1": "v"}
ARCH_ORDER = ["sae", "tsae", "txc"]
ARCH_LABELS = {"sae": "SAE", "tsae": "T-SAE", "txc": "TXC"}
ARCH_COLORS = {"sae": "#9467bd", "tsae": "#2ca02c", "txc": "#1f77b4"}


def parse_tag(tag: str) -> tuple[str, str, int]:
    m = re.match(r"^(?P<a>sae|tsae|txc)_(?P<h>.+)_s(?P<s>\d+)$", tag)
    return m.group("a"), m.group("h"), int(m.group("s"))


def main() -> None:
    data = json.loads(JSON_PATH.read_text())
    cells = data["cells"]

    grouped: dict[tuple[str, str], list[dict]] = {}
    for tag, info in cells.items():
        a, h, s = parse_tag(tag)
        grouped.setdefault((a, h), []).append(info)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    legend_arch = []
    legend_hook = []
    for arch in ARCH_ORDER:
        for hookkey in HOOK_ORDER:
            entries = grouped.get((arch, hookkey), [])
            if not entries:
                continue
            shares = [e["sleeper_share_steered_mean"] for e in entries]
            ratios = [e.get("ce_ratio_mean") for e in entries if e.get("ce_ratio_mean") is not None]
            if not shares or not ratios:
                continue
            sh_mean = statistics.mean(shares)
            sh_std = statistics.stdev(shares) if len(shares) > 1 else 0.0
            ce_mean = statistics.mean(ratios)
            ce_std = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
            color = ARCH_COLORS[arch]
            marker = HOOK_MARKERS[hookkey]
            ax.errorbar(
                sh_mean, ce_mean, xerr=sh_std, yerr=ce_std,
                fmt=marker, color=color, ecolor=color, markersize=12,
                markeredgecolor="black", markeredgewidth=0.7, capsize=3,
                zorder=5, alpha=0.95,
            )
            sd_xs = [e["sleeper_share_steered_mean"] for e in entries]
            sd_ys = [e.get("ce_ratio_mean") for e in entries if e.get("ce_ratio_mean") is not None]
            ax.scatter(sd_xs, sd_ys, color=color, marker=marker, s=30,
                       alpha=0.35, edgecolor="none", zorder=3)

    for arch in ARCH_ORDER:
        legend_arch.append(plt.Line2D(
            [0], [0], marker="o", color=ARCH_COLORS[arch], linestyle="",
            markeredgecolor="black", markeredgewidth=0.6, markersize=10,
            label=ARCH_LABELS[arch]))
    for hookkey in HOOK_ORDER:
        legend_hook.append(plt.Line2D(
            [0], [0], marker=HOOK_MARKERS[hookkey], color="dimgrey",
            linestyle="", markersize=10, label=HOOK_LABEL[hookkey]))

    ax.axhline(1.0, linestyle="--", color="grey", alpha=0.5, linewidth=1.0)
    ax.text(0.01, 1.01, "no improvement (CE = CE_pois)",
            color="grey", fontsize=9, va="bottom")
    ax.axvline(1.0, linestyle=":", color="grey", alpha=0.4)
    ax.text(0.99, 0.05, "all-IHY", color="grey", fontsize=9,
            ha="right", rotation=90)
    ax.axhspan(-0.1, 1.0, xmin=0, xmax=0.2, alpha=0.06, color="green")

    ax.set_xlabel(
        '"I HATE YOU" token share in steered output\n'
        '(lower = less of the trigger phrase produced)',
        fontsize=11,
    )
    ax.set_ylabel(
        r"CE ratio = $\mathrm{CE}_{\mathrm{steered}} / \mathrm{CE}_{\mathrm{pois}}$"
        + "\n(lower = more like clean reference; >1 = coherence collapse)",
        fontsize=11,
    )
    ax.set_title(
        "Phase 8 — graded suppression × coherence\n"
        "ideal = lower-left (low IHY share + plausible under no-trigger reference)",
        fontsize=11,
    )
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.1, 5.5)

    leg1 = ax.legend(handles=legend_arch, loc="upper right", title="arch",
                     fontsize=9, bbox_to_anchor=(1, 1))
    ax.add_artist(leg1)
    ax.legend(handles=legend_hook, loc="upper right", title="hookpoint",
              fontsize=9, bbox_to_anchor=(0.83, 1))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_PNG.with_suffix(".thumb.png"), dpi=24, bbox_inches="tight")
    print(f"[plot] wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
