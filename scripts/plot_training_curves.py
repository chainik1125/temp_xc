"""Parse SAEBench training logs and plot loss + L0 curves per checkpoint.

Run: python3 scripts/plot_training_curves.py

Context: Dmitry asked whether TempXC needs more training. SAE/MLC reach
NMSE=0.05–0.07; TempXC is 0.15–0.28. This script extracts the every-500-step
loss + L0 prints from training logs so we can eyeball whether TempXC is
still converging at step 5000 or has plateaued.

Excludes the three buggy pre-fix Protocol B TempXC cells (k500__T5, k1000__T10,
k2000__T20) whose k-resolution was wrong — see eval_infra_lessons.md B4.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt


LOGS_DIR = Path("results/saebench/logs")
PLOTS_DIR = Path("results/saebench/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# The 10 valid checkpoints. Buggy pre-fix Protocol B cells (k500__T5,
# k1000__T10, k2000__T20) are excluded from training-curve plots since
# their sparsity constraint is wrong — any training curve is moot.
VALID_CELLS = [
    ("sae",    "A", None,  "sae__gemma-2-2b__l12__k100__protA__seed42"),
    ("sae",    "B", None,  "sae__gemma-2-2b__l12__k100__protB__seed42"),
    ("mlc",    "A", None,  "mlc__gemma-2-2b__l10-11-12-13-14__k100__protA__seed42"),
    ("mlc",    "B", None,  "mlc__gemma-2-2b__l10-11-12-13-14__k100__protB__seed42"),
    ("tempxc", "A", 5,     "tempxc__gemma-2-2b__l12__k100__T5__protA__seed42"),
    ("tempxc", "B", 5,     "tempxc__gemma-2-2b__l12__k100__T5__protB__seed42"),
    ("tempxc", "A", 10,    "tempxc__gemma-2-2b__l12__k100__T10__protA__seed42"),
    ("tempxc", "B", 10,    "tempxc__gemma-2-2b__l12__k50__T10__protB__seed42"),
    ("tempxc", "A", 20,    "tempxc__gemma-2-2b__l12__k100__T20__protA__seed42"),
    ("tempxc", "B", 20,    "tempxc__gemma-2-2b__l12__k25__T20__protB__seed42"),
]

# Matches "step   500/5000 | loss=X.X | (window_)?l0=Y.Y"
LINE_RE = re.compile(
    r"step\s+(\d+)/\d+\s*\|\s*loss=([\d.]+)\s*\|\s*(?:window_)?l0=([\d.]+)"
)


def parse_log(log_path: Path):
    """Return list of (step, loss, l0) tuples from training prints."""
    rows = []
    with open(log_path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                step, loss, l0 = int(m.group(1)), float(m.group(2)), float(m.group(3))
                rows.append((step, loss, l0))
    return rows


def load_all():
    curves = []
    for arch, proto, t, stem in VALID_CELLS:
        path = LOGS_DIR / f"{stem}.log"
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        rows = parse_log(path)
        if not rows:
            print(f"  EMPTY:   {path}")
            continue
        label = _label(arch, proto, t)
        curves.append({"arch": arch, "proto": proto, "t": t, "label": label, "rows": rows})
        print(f"  {label:<18} {len(rows)} points, final loss={rows[-1][1]:.1f} l0={rows[-1][2]:.2f}")
    return curves


def _label(arch, proto, t):
    if arch == "sae":
        return f"SAE (prot{proto})"
    if arch == "mlc":
        return f"MLC (prot{proto})"
    return f"TempXC T={t} (prot{proto})"


def _color(arch, proto, t):
    if arch == "sae":
        return "#1f77b4" if proto == "A" else "#1f77b4"
    if arch == "mlc":
        return "#ff7f0e" if proto == "A" else "#ff7f0e"
    # TempXC: green family, darker for larger T
    shades = {5: "#98df8a", 10: "#2ca02c", 20: "#006d2c"}
    return shades[t]


def _style(proto):
    return "-" if proto == "A" else "--"


def plot_all(curves):
    # Two panels: loss + L0, log-scale loss so plateau is visible
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax_loss, ax_l0 = axes

    for c in curves:
        steps = [r[0] for r in c["rows"]]
        losses = [r[1] for r in c["rows"]]
        l0s = [r[2] for r in c["rows"]]
        color = _color(c["arch"], c["proto"], c["t"])
        style = _style(c["proto"])
        ax_loss.plot(steps, losses, style, color=color, label=c["label"], linewidth=1.8, marker="o", markersize=4)
        ax_l0.plot(steps, l0s, style, color=color, label=c["label"], linewidth=1.8, marker="o", markersize=4)

    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("MSE reconstruction loss")
    ax_loss.set_yscale("log")
    ax_loss.set_title("Training loss (log scale)")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(loc="upper right", fontsize=8, ncol=2)

    ax_l0.set_xlabel("training step")
    ax_l0.set_ylabel("effective L0 (non-zero features per window)")
    ax_l0.set_title("L0 / window activation count")
    ax_l0.grid(alpha=0.3)
    ax_l0.legend(loc="upper right", fontsize=8, ncol=2)

    fig.suptitle("Training dynamics — all 10 valid checkpoints", y=1.02)
    fig.tight_layout()
    out = PLOTS_DIR / "fig6_training_curves_all.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")


def plot_tempxc_only(curves):
    """TempXC only, with protocol-A solid / protocol-B dashed. Separate
    because Dmitry's question is specifically about TempXC convergence."""
    tempxc = [c for c in curves if c["arch"] == "tempxc"]
    sae_a = [c for c in curves if c["arch"] == "sae" and c["proto"] == "A"]
    mlc_a = [c for c in curves if c["arch"] == "mlc" and c["proto"] == "A"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax_loss, ax_l0 = axes

    # TempXC cells
    for c in tempxc:
        steps = [r[0] for r in c["rows"]]
        losses = [r[1] for r in c["rows"]]
        l0s = [r[2] for r in c["rows"]]
        color = _color(c["arch"], c["proto"], c["t"])
        style = _style(c["proto"])
        ax_loss.plot(steps, losses, style, color=color, label=c["label"], linewidth=2, marker="o", markersize=5)
        ax_l0.plot(steps, l0s, style, color=color, label=c["label"], linewidth=2, marker="o", markersize=5)

    # SAE + MLC reference (protA only, thin lines)
    for c in sae_a + mlc_a:
        steps = [r[0] for r in c["rows"]]
        losses = [r[1] for r in c["rows"]]
        l0s = [r[2] for r in c["rows"]]
        color = _color(c["arch"], c["proto"], None)
        ax_loss.plot(steps, losses, "-", color=color, label=c["label"] + " (ref)", linewidth=1.5, alpha=0.7)
        ax_l0.plot(steps, l0s, "-", color=color, label=c["label"] + " (ref)", linewidth=1.5, alpha=0.7)

    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("MSE reconstruction loss")
    ax_loss.set_yscale("log")
    ax_loss.set_title("Loss: TempXC vs SAE/MLC reference")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(loc="upper right", fontsize=8)

    ax_l0.set_xlabel("training step")
    ax_l0.set_ylabel("L0 / window activations")
    ax_l0.set_title("L0 stability")
    ax_l0.grid(alpha=0.3)
    ax_l0.legend(loc="upper right", fontsize=8)

    fig.suptitle("TempXC training curves — is it undertrained?", y=1.02)
    fig.tight_layout()
    out = PLOTS_DIR / "fig7_tempxc_training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_normalized_loss(curves):
    """Normalize each curve by its step-0 loss to visualize fractional
    progress. Tells us who's still moving fastest at step 5000."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for c in curves:
        steps = [r[0] for r in c["rows"]]
        losses = [r[1] for r in c["rows"]]
        if losses[0] == 0:
            continue
        normalized = [l / losses[0] for l in losses]
        color = _color(c["arch"], c["proto"], c["t"])
        style = _style(c["proto"])
        ax.plot(steps, normalized, style, color=color, label=c["label"],
                linewidth=1.8, marker="o", markersize=4)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss(t) / loss(0)")
    ax.set_yscale("log")
    ax.set_title("Normalized loss: fraction of initial reconstruction error remaining")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    out = PLOTS_DIR / "fig8_normalized_loss.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def print_slope_report(curves):
    """How much does loss drop over the last 20% of training? If big, it
    hasn't converged."""
    print("\n=== LATE-TRAINING SLOPE (does loss keep dropping?) ===")
    print(f"{'cell':<20} {'loss@step4000':<15} {'loss@step4999':<15} {'Δ':<10} {'Δ%':<10}")
    for c in curves:
        rows = c["rows"]
        # step 4000 and final 4999
        loss_4000 = next((l for s, l, _ in rows if s == 4000), None)
        loss_final = rows[-1][1]
        if loss_4000 is None:
            continue
        delta = loss_final - loss_4000
        delta_pct = delta / loss_4000 * 100
        print(f"{c['label']:<20} {loss_4000:<15.1f} {loss_final:<15.1f} "
              f"{delta:<10.1f} {delta_pct:<+10.2f}%")


def main():
    print("=== loading training logs ===")
    curves = load_all()
    print()

    print("=== generating plots ===")
    plot_all(curves)
    plot_tempxc_only(curves)
    plot_normalized_loss(curves)

    print_slope_report(curves)
    print(f"\nAll plots in {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
