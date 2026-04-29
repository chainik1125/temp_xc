"""Reproductions of Ward Fig 3 (base vs reasoning steering bars).

Single primary figure: percent "wait+hmm" tokens vs steering magnitude,
with four series:
  - base model    + base-derived vector       (Ward expects ~0%)
  - base model    + reasoning-derived vector  (Ward expects ~0%)
  - reasoning     + base-derived vector       (Ward Fig 3 main result; peaks ~30-50%)
  - reasoning     + reasoning-derived vector  (control; comparable to above)

Reads steering_results.json from steer_eval.py.

(Fig 4 reproduction — comparing the ours-vector against
overall_mean / noise / self_amplification / deduction / initialization
baselines — is left as a follow-on. Stage A focus is Fig 3.)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward.plot")


def _aggregate(rows: list[dict]) -> dict[tuple[str, str, float], dict]:
    """Group rows by (target, source, magnitude); compute mean ± std of keyword_rate."""
    buckets: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for r in rows:
        buckets[(r["target"], r["source"], r["magnitude"])].append(r["keyword_rate"])
    out: dict[tuple[str, str, float], dict] = {}
    for k, vs in buckets.items():
        m = mean(vs)
        s = stdev(vs) if len(vs) > 1 else 0.0
        out[k] = {"mean": m, "std": s, "n": len(vs)}
    return out


def plot_fig3(rows: list[dict], magnitudes: list[float], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    agg = _aggregate(rows)
    series = [
        ("base", "base_derived_union", "Base model + base-derived"),
        ("base", "reasoning_derived_union", "Base model + reasoning-derived"),
        ("reasoning", "base_derived_union", "Reasoning model + base-derived"),
        ("reasoning", "reasoning_derived_union", "Reasoning model + reasoning-derived"),
    ]
    colors = ["#bbbbbb", "#ee9999", "#3a86ff", "#fb5607"]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.18
    x = list(range(len(magnitudes)))

    for i, (tgt, src, label) in enumerate(series):
        means = []
        stds = []
        for mag in magnitudes:
            cell = agg.get((tgt, src, float(mag)))
            means.append(cell["mean"] * 100 if cell else 0.0)
            stds.append(cell["std"] * 100 if cell else 0.0)
        offset = (i - 1.5) * width
        ax.bar([xi + offset for xi in x], means, width=width, yerr=stds,
               label=label, color=colors[i], capsize=3, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in magnitudes])
    ax.set_xlabel("Steering Magnitude")
    ax.set_ylabel('"Wait" + "Hmm" tokens %')
    ax.set_title("Ward Fig 3 reproduction — base-derived steering vector\ninduces backtracking only in the reasoning model")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    log.info("[saved] %s", out_path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    payload = json.loads(Path(cfg["paths"]["steering"]).read_text())
    rows = payload["rows"]
    magnitudes = payload["meta"]["magnitudes"]

    out_dir = Path(cfg["paths"]["plots"])
    plot_fig3(rows, magnitudes, out_dir / "fig3_base_vs_reasoning_steering.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
