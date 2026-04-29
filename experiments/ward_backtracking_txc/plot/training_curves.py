"""One PNG per hookpoint: loss / FVU / window-L0 / dead-feature count vs step."""

from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.ward_backtracking_txc.plot._common import load_cfg, plots_dir


def _read_log(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    logs_dir = Path(cfg["paths"]["logs_dir"])
    out_dir = plots_dir(cfg)

    for hp in cfg["hookpoints"]:
        if not hp.get("enabled", True):
            continue
        log_path = logs_dir / f"{hp['key']}_train.jsonl"
        if not log_path.exists():
            print(f"[skip] no log for {hp['key']}"); continue
        rows = _read_log(log_path)
        if not rows:
            print(f"[skip] empty log for {hp['key']}"); continue
        steps = [r["step"] for r in rows]
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
        for ax, key, title in zip(
            axes, ["loss", "fvu", "window_l0", "n_dead"],
            ["recon loss", "FVU", "window L0", "dead features"]
        ):
            ax.plot(steps, [r[key] for r in rows], lw=1.5)
            ax.set_xlabel("step"); ax.set_title(title); ax.grid(alpha=0.3)
        fig.suptitle(f"TXC training — {hp['key']} ({hp['label']})", fontsize=11)
        fig.tight_layout()
        out = out_dir / f"training_curves_{hp['key']}.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
