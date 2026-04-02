"""Side-by-side plots: local feature recovery and global (probe) recovery vs rho."""

import matplotlib.pyplot as plt
import numpy as np

# --- data from probing_hidden_state.md ---
rhos = np.array([0.0, 0.5, 1.0])

feat_rec = {
    "SAE":     np.array([0.98, 1.00, 0.02]),
    "TXCDR":   np.array([0.20, 0.62, 0.00]),
    "PerFeat": np.array([0.82, 1.00, 0.76]),
}

probe_auc = {
    "SAE":     np.array([0.876, 0.909, 0.871]),
    "TXCDR":   np.array([0.681, 0.863, 0.995]),
    "PerFeat": np.array([0.915, 0.921, 0.900]),
}

styles = {
    "SAE":     {"color": "#1f77b4", "marker": "o"},
    "TXCDR":   {"color": "#d62728", "marker": "s"},
    "PerFeat": {"color": "#2ca02c", "marker": "^"},
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

for name in ["SAE", "TXCDR", "PerFeat"]:
    kw = dict(styles[name], linewidth=2, markersize=8)
    ax1.plot(rhos, feat_rec[name], label=name, **kw)
    ax2.plot(rhos, probe_auc[name], label=name, **kw)

# left panel
ax1.set_xlabel(r"Temporal correlation $\rho$", fontsize=12)
ax1.set_ylabel("Feature Recovery (R@0.9)", fontsize=12)
ax1.set_title("Local Feature Recovery", fontsize=13)
ax1.set_xticks(rhos)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# right panel
ax2.set_xlabel(r"Temporal correlation $\rho$", fontsize=12)
ax2.set_ylabel("Linear Probe AUC", fontsize=12)
ax2.set_title("Global Feature Recovery (Probe)", fontsize=13)
ax2.set_xticks(rhos)
ax2.set_ylim(0.6, 1.02)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

fig.tight_layout()
out_dir = "docs/dmitry/results/temporal_xc"
fig.savefig(f"{out_dir}/rho_sweep.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{out_dir}/rho_sweep.pdf", bbox_inches="tight")
print(f"Saved {out_dir}/rho_sweep.png and {out_dir}/rho_sweep.pdf")
plt.show()
