"""Re-render Setup A + Setup B AUTO-RESULTS blocks in c2.md.

Run after baseline backfill completes:
  TQDM_DISABLE=1 .venv/bin/python -m agents.agent_hammer.render_results
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


def _replace_block(content: str, begin: str, end: str, new_body: str) -> str:
    bi = content.find(begin)
    ei = content.find(end)
    if bi < 0 or ei < 0:
        raise SystemExit(f"Could not find markers {begin!r} / {end!r} in c2.md")
    return content[: bi + len(begin)] + "\n\n" + new_body.strip() + "\n\n" + content[ei:]


def main() -> None:
    md_path = Path("docs/components/c2.md")

    # ── Setup A — c2 analysis ──
    a_mod = import_module("experiments.c2_synthetic_coupled.analysis")
    a_result = a_mod.run_analysis()
    print(f"[render] Setup A: {a_result.results}")

    # ── Setup B — c1_noisy analysis ──
    b_mod = import_module("experiments.c1_noisy_filler.analysis")
    b_result = b_mod.run_analysis()
    print(f"[render] Setup B: {b_result.results}")

    # ── Rewrite both blocks atomically ──
    content = md_path.read_text()
    content = _replace_block(
        content,
        "<!-- BEGIN AUTO-RESULTS -->",
        "<!-- END AUTO-RESULTS -->",
        a_result.markdown,
    )
    content = _replace_block(
        content,
        "<!-- BEGIN AUTO-RESULTS-c1-noisy -->",
        "<!-- END AUTO-RESULTS-c1-noisy -->",
        b_result.markdown,
    )
    md_path.write_text(content)
    print(f"[render] Rewrote {md_path} (both AUTO-RESULTS blocks).")

    # ── Re-render denoising plots from the on-disk JSON ──
    # We do NOT re-run denoising_probes.py here — that's a separate step
    # (see Phase 3) which writes denoising_probe_results.json. After that
    # JSON is up-to-date, this just regenerates the plots from it.
    json_path = Path("experiments/c1_noisy_filler/denoising_probe_results.json")
    if not json_path.exists():
        print(f"[render] Skipping denoising plots — {json_path} missing "
              f"(run Phase 3 denoising_probes.py first)")
        return

    from experiments.c1_noisy_filler.denoising_probes import (
        _aggregate_by_seeds, plot_panels, plot_scatter,
    )
    results = json.loads(json_path.read_text())
    agg = _aggregate_by_seeds(results)
    plots_dir = Path("experiments/c1_noisy_filler/plots")
    plot_scatter(agg, plots_dir / "c2_noisy_singlelatent_scatter.png", mode="sl")
    plot_scatter(agg, plots_dir / "c2_noisy_probe_scatter.png",        mode="lp")
    plot_panels(agg,  plots_dir / "c2_noisy_singlelatent_panels.png",  mode="sl")
    plot_panels(agg,  plots_dir / "c2_noisy_denoising_panels.png",     mode="lp")
    print(f"[render] Regenerated 4 denoising plots in {plots_dir}/")


if __name__ == "__main__":
    main()
