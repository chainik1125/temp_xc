"""Render the mandatory 4-plot template for every C2 setup.

Run after baseline gap-fills land + Setup HALCYON/KESTREL data lands:

    TQDM_DISABLE=1 .venv/bin/python -m agents.agent_hammer.render_all_setups

Each setup gets:
- c2_setup_<name>_gauc_vs_k.png
- c2_setup_<name>_eauc_vs_k.png
- c2_setup_<name>_scatter.png
- c2_setup_<name>_tsweep.png

Setup names (per Han's no-letter convention for new setups):
- A           — Setup A canonical
- c (ρ-sweep) — composite ρ-sweep panel via render_setup at each ρ
- d_npX, e, f_sigma, g_sigma — agent_synth's existing setups
- h_rho00_full, h_rho09_full — Setup H at full-coverage ρ
- halcyon_d128, halcyon_d512 — d_in scaling
- kestrel_seq32, kestrel_seq128 — seq_len scaling
"""

from __future__ import annotations

from pathlib import Path

from experiments.c2_synthetic_coupled.plot_headline import render_setup

NOISY_PLOT_DIR = Path("experiments/c2_synthetic_coupled/plots")
HIER_PLOT_DIR = Path("experiments/c2_hierarchical/plots")


def render_for_datasource(
    *, setup_name: str, datasource: str, plot_dir: Path,
    title: str, fixed_k_for_tsweep: int = 1,
) -> None:
    """Render the 4-plot template for one datasource, with txc_base
    filter on the T-sweep (avoids the default-T arch pollution bug
    in plot_headline.render_tsweep)."""
    render_setup(
        setup_name=setup_name,
        plot_dir=plot_dir,
        line_filter_fn=lambda d, ds=datasource: d.get("datasource") == ds,
        tsweep_filter_fn=lambda d, ds=datasource: (
            d.get("datasource") == ds and d.get("arch") == "txc_base"
        ),
        title_root=title,
        fixed_k_for_tsweep=fixed_k_for_tsweep,
    )


def main() -> None:
    # Setup A — canonical clean coupled.
    render_for_datasource(
        setup_name="a",
        datasource="toy_coupled_K10_M20_d256",
        plot_dir=NOISY_PLOT_DIR,
        title="Setup A (clean coupled features)",
    )

    # Setup C — ρ-sweep on Setup A: render per ρ.
    for rho_tag, rho in [("rho00", 0.0), ("rho03", 0.3), ("rho06", 0.6), ("rho09", 0.9)]:
        render_for_datasource(
            setup_name=f"c_{rho_tag}",
            datasource=f"toy_coupled_K10_M20_d256_{rho_tag}",
            plot_dir=NOISY_PLOT_DIR,
            title=f"Setup C ρ={rho} (ρ-sweep on clean coupled)",
        )

    # Setup F (3 sigmas) and G (2 sigmas) at headline σ values.
    for sigma_tag, sigma in [("sigma0p5", 0.5), ("sigma1p0", 1.0), ("sigma2p0", 2.0)]:
        render_for_datasource(
            setup_name=f"f_{sigma_tag}",
            datasource=f"toy_coupled_obs_noise_K10_M20_d256_{sigma_tag}",
            plot_dir=NOISY_PLOT_DIR,
            title=f"Setup F σ={sigma} (coupled + obs noise)",
        )
    for sigma_tag, sigma in [("sigma1p0", 1.0), ("sigma2p0", 2.0)]:
        render_for_datasource(
            setup_name=f"g_{sigma_tag}",
            datasource=f"toy_hierarchical_Kg10_Kl30_d256_{sigma_tag}",
            plot_dir=HIER_PLOT_DIR,
            title=f"Setup G σ={sigma} (hierarchical + obs noise)",
        )

    # Setup E np2 (after gap-fill).
    render_for_datasource(
        setup_name="e_np2",
        datasource="toy_hierarchical_Kg10_Kl30_d256_np2",
        plot_dir=HIER_PLOT_DIR,
        title="Setup E n_global_parents=2 (hierarchical, denser parent map)",
    )

    # Setup H ρ-sweep at each ρ.
    for rho_tag, rho in [("rho00", 0.0), ("rho03", 0.3), ("rho06", 0.6)]:
        render_for_datasource(
            setup_name=f"h_{rho_tag}_full",
            datasource=f"toy_coupled_noisy_K10_M20_d256_pB05_np10_{rho_tag}",
            plot_dir=NOISY_PLOT_DIR,
            title=f"Setup H ρ={rho} (D-np10 ρ-sweep, full 5 baselines)",
        )
    render_for_datasource(
        setup_name="h_rho09_full",
        datasource="toy_coupled_noisy_K10_M20_d256_pB05_np10",
        plot_dir=NOISY_PLOT_DIR,
        title="Setup H ρ=0.9 (D-np10 base, full T-sweep)",
    )

    # Setup HALCYON — d_in scaling.
    for d_in in (128, 512):
        render_for_datasource(
            setup_name=f"halcyon_d{d_in}",
            datasource=f"toy_coupled_K10_M20_d{d_in}_halcyon",
            plot_dir=NOISY_PLOT_DIR,
            title=f"Setup HALCYON d_in={d_in} (Setup A scaling)",
        )

    # Setup KESTREL — seq_len scaling.
    for seq_tag, seq_len in [("seq32", 32), ("seq128", 128)]:
        render_for_datasource(
            setup_name=f"kestrel_{seq_tag}",
            datasource=f"toy_coupled_K10_M20_d256_{seq_tag}_kestrel",
            plot_dir=NOISY_PLOT_DIR,
            title=f"Setup KESTREL seq_len={seq_len} (Setup A scaling)",
        )

    print("\nDone. Re-run agents.agent_hammer.verify_setups to confirm.")


if __name__ == "__main__":
    main()
