"""Interactive Plotly dashboard: 14B-finance c6 EM alignment-vs-coherence
α-trajectories, 5 archs × 2 seeds.

Each subplot is the headline finalist's (coh, align) trace coloured by α
(canonical 27 α-points + dense 30-point extension), with the two alternate
finalists drawn faintly. Per-subplot annotations report Δalign|coh≥70 and
peak alignment.

Output: a single self-contained HTML file with Plotly (CDN-loaded) so the
user can pan / zoom / hover / reset per subplot. Gridlines on every axis.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LOCAL = Path(
    "/Users/dmitrymanning-coe/Documents/Research/Temporal Crosscoders/temp_xc/"
    "dmitry/pre_purified/c6_em_overnight"
)
OUT = Path(
    "/Users/dmitrymanning-coe/Documents/Research/Temporal Crosscoders/temp_xc/"
    "plots/2026-05-07_c6_em_align_coh_grid/align_coh_dashboard_14b_finance.html"
)

# (arch, seed, train_key) — 14B-finance only
CELLS = [
    ("sae_arditi", 1,  "caa331cb08fce8bf"),
    ("sae_arditi", 42, "700b9ff4d7c297af"),
    ("tsae_paper", 1,  "2ebee6c2a3552ece"),
    ("tsae_paper", 42, "98c9dd4cfd77a1dc"),
    ("txc_base",   1,  "f9a1d482e5a48221"),
    ("txc_base",   42, "99bc3c9f739c8f1b"),
    ("txc_pro",    1,  "10d82cab97bace0a"),
    ("txc_pro",    42, "ba79122b41fc96f6"),
    ("tfa",        1,  "676c390321a106c3"),
    ("tfa",        42, "da6a9fb42ed4e797"),
]
ARCH_ORDER = ["sae_arditi", "tsae_paper", "txc_base", "txc_pro", "tfa"]
ARCH_LABEL = {"sae_arditi": "SAE", "tsae_paper": "T-SAE",
              "txc_base": "TXC", "txc_pro": "TXC-pro", "tfa": "TFA"}


def load_cell(train_key: str) -> dict:
    wang = json.loads((LOCAL / "runs" / f"c6_{train_key}" / "wang_full.json").read_text())
    headline_fid = (wang.get("headline") or {}).get("feature_id")
    finalists = []
    for f in wang.get("stage4", {}).get("finalists", []):
        finalists.append({
            "feature_id": f["feature_id"],
            "rows": [{"alpha": r["alpha"],
                      "mean_align": r["mean_align"],
                      "mean_coh":   r["mean_coh"]} for r in f.get("rows", [])],
            "is_headline": f["feature_id"] == headline_fid,
        })

    dense_path = LOCAL / "sweep_outputs" / f"c6_{train_key}" / "wang_full_extended.json"
    dense_rows = []
    if dense_path.exists():
        d = json.loads(dense_path.read_text())
        dense_rows = [
            {"alpha": r["alpha"], "mean_align": r["mean_align"], "mean_coh": r["mean_coh"]}
            for r in d.get("rows", [])
            if r.get("feature_id") == headline_fid
        ]
    return {"finalists": finalists, "dense_rows": dense_rows, "headline_fid": headline_fid}


def headline_curve(payload: dict):
    head = next(f for f in payload["finalists"] if f["is_headline"])
    rows = list(head["rows"]) + list(payload["dense_rows"])
    rows.sort(key=lambda r: r["alpha"])
    a = np.array([r["alpha"] for r in rows])
    al = np.array([r["mean_align"] for r in rows])
    co = np.array([r["mean_coh"] for r in rows])
    return a, al, co


def headline_metrics(a, al, co):
    mask = co >= 70.0
    delta = float(al[mask].max() - al[mask].min()) if mask.any() else float("nan")
    peak = float(al.max())
    n70 = int(mask.sum())
    return delta, peak, n70


def main():
    cells_by_key = {(arch, seed): tk for arch, seed, tk in CELLS}

    n_archs = len(ARCH_ORDER)
    titles = []
    for arch in ARCH_ORDER:
        for seed in (1, 42):
            tk = cells_by_key[(arch, seed)]
            payload = load_cell(tk)
            a, al, co = headline_curve(payload)
            d70, peak, n70 = headline_metrics(a, al, co)
            titles.append(
                f"<b>{ARCH_LABEL[arch]}, s={seed}</b>  feat={payload['headline_fid']}<br>"
                f"<span style='font-size:11px'>Δalign|coh≥70 = {d70:.1f}  ·  "
                f"peak align = {peak:.1f}  ·  n α(coh≥70) = {n70}</span>"
            )

    fig = make_subplots(
        rows=n_archs, cols=2,
        subplot_titles=titles,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.07,
        vertical_spacing=0.06,
    )

    showed_legend_for_alt = False
    showed_colorbar = False

    for r, arch in enumerate(ARCH_ORDER, start=1):
        for c, seed in enumerate((1, 42), start=1):
            tk = cells_by_key[(arch, seed)]
            payload = load_cell(tk)

            # Alternate finalists in grey
            for f in payload["finalists"]:
                if f["is_headline"]:
                    continue
                rs = sorted(f["rows"], key=lambda r: r["alpha"])
                fig.add_trace(
                    go.Scatter(
                        x=[r["mean_coh"] for r in rs],
                        y=[r["mean_align"] for r in rs],
                        mode="lines",
                        line=dict(color="lightgrey", width=1.2),
                        showlegend=not showed_legend_for_alt,
                        name="alt finalist",
                        legendgroup="alt",
                        hovertemplate="alt feat<br>α=%{customdata:.1f}<br>"
                                      "coh=%{x:.1f}, align=%{y:.1f}<extra></extra>",
                        customdata=[r["alpha"] for r in rs],
                    ),
                    row=r, col=c,
                )
                showed_legend_for_alt = True

            # Headline curve
            a, al, co = headline_curve(payload)
            order = np.argsort(a)
            # connecting line
            fig.add_trace(
                go.Scatter(
                    x=co[order], y=al[order],
                    mode="lines",
                    line=dict(color="rgba(80,80,80,0.45)", width=0.8),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=r, col=c,
            )
            # colored markers
            fig.add_trace(
                go.Scatter(
                    x=co, y=al,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=a,
                        colorscale="RdBu_r",
                        cmin=-200, cmid=0, cmax=200,
                        line=dict(color="black", width=0.4),
                        colorbar=dict(
                            title=dict(text="steering α", side="right"),
                            x=1.02, y=0.5, len=0.85,
                            thickness=14,
                        ) if not showed_colorbar else None,
                        showscale=not showed_colorbar,
                    ),
                    showlegend=False,
                    hovertemplate="<b>headline</b><br>α=%{marker.color:.1f}<br>"
                                  "coh=%{x:.1f}, align=%{y:.1f}<extra></extra>",
                ),
                row=r, col=c,
            )
            showed_colorbar = True

            # ★ baseline marker at α≈0
            i0 = int(np.argmin(np.abs(a)))
            fig.add_trace(
                go.Scatter(
                    x=[co[i0]], y=[al[i0]],
                    mode="markers",
                    marker=dict(symbol="star", size=14, color="white",
                                line=dict(color="black", width=1)),
                    showlegend=False,
                    hovertemplate=f"α=0 baseline<br>α={a[i0]:.1f}<br>"
                                  "coh=%{x:.1f}, align=%{y:.1f}<extra></extra>",
                ),
                row=r, col=c,
            )

            # axis grid + reference lines + range
            fig.update_xaxes(
                row=r, col=c, range=[0, 100],
                showgrid=True, gridwidth=1, gridcolor="rgba(180,180,180,0.45)",
                zeroline=False,
                title_text="coherence (%)" if r == n_archs else None,
            )
            fig.update_yaxes(
                row=r, col=c, range=[0, 100],
                showgrid=True, gridwidth=1, gridcolor="rgba(180,180,180,0.45)",
                zeroline=False,
                title_text="alignment (%)" if c == 1 else None,
            )
            # Add reference dotted lines at align=50 and coh=70
            fig.add_hline(y=50, line=dict(color="grey", width=0.8, dash="dot"),
                          row=r, col=c)
            fig.add_vline(x=70, line=dict(color="grey", width=0.8, dash="dot"),
                          row=r, col=c)

    fig.update_layout(
        title=dict(
            text="<b>c6 EM 14B-finance α-sweep: alignment vs coherence per (arch, seed)</b>"
                 "<br><span style='font-size:13px'>Headline finalist coloured by α; alt finalists in grey; "
                 "★ α≈0 baseline; pan + zoom enabled per subplot</span>",
            x=0.01, xanchor="left",
        ),
        height=1500, width=1100,
        paper_bgcolor="white", plot_bgcolor="white",
        hovermode="closest",
        margin=dict(l=70, r=110, t=110, b=60),
    )

    # Smaller subplot title font
    fig.update_annotations(font_size=11)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        OUT,
        include_plotlyjs="cdn",
        config={"displayModeBar": True, "scrollZoom": True,
                "modeBarButtonsToAdd": ["drawline", "drawopenpath",
                                        "drawclosedpath", "drawcircle",
                                        "drawrect", "eraseshape"]},
    )
    print(f"wrote {OUT}")
    print(f"file size: {OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
