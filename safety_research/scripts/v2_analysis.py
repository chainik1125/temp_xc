"""
v2 analysis — load every per-config JSON written by v2_steering.py, plus the
existing andre_steering and realbench_steer artifacts, and produce:

  1. results/andre_steering_v2/all_rows.json      — flattened table
  2. results/andre_steering_v2/stats.json         — bootstrap CIs / p-values
  3. results/andre_steering_v2/paper_macros.tex   — \\newcommand macros
  4. results/andre_steering_v2/paper_macros.json  — same, for markdown
  5. figures/andre_steer_v2_*.png                 — eight figures

Statistical layer:
  * paired bootstrap (1000 resamples) for ΔLR_harm, ΔLR_ben, leakage
  * Wilcoxon signed-rank test (intervention vs baseline) on the per-prompt
    ΔLR vector, two-sided
  * permutation test for "TXC FSGA leakage < SAE FSGA leakage" at K=20

cFSGA: a *derived* row per (arm, K, ds) — for prompts where probe predicts
harmful, use S3 LR; for probe-says-benign prompts, use baseline LR. Reports
the same headline metrics on the *full* eval set (so direct comparison
against unconditional FSGA on identical prompts).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

ROOT = Path("/home/cs29824/andre/temp_xc/safety_research")
OUT = ROOT / "results" / "andre_steering_v2"
PROMPTS = ROOT / "results" / "realbench"
RB_STEER = ROOT / "results" / "realbench" / "steer"
ANDRE = ROOT / "results" / "andre_steering"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ARM_LABEL = {"sae": "SAE (T=1)", "tsae": "T-SAE (T=5)", "txc": "TXC (T=5)"}
ARM_COLOR = {"sae": "#2563eb", "tsae": "#16a34a", "txc": "#dc2626"}
HARM_DS = ("test_in", "test_ood", "test_mi")
DS_LABEL = {"test_in": "JBB (in)", "test_ood": "XSTest (ood)",
            "test_mi": "MaliciousInstruct"}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_baseline_lr() -> dict[str, dict]:
    return json.load(open(OUT / "baseline_lr.json"))


def load_baseline_logp() -> np.ndarray:
    return np.load(OUT / "base_logp_cap_alpaca.npz")["base_logp"]


def load_probe_decisions() -> dict:
    return json.load(open(OUT / "probe_decisions.json"))


def all_v2_configs() -> list[dict]:
    """Iterate over every config JSON dropped by v2_steering."""
    configs: list[dict] = []
    for p in sorted(OUT.glob("*.json")):
        if p.name in {"baseline_lr.json", "baseline_mmlu.json",
                      "probe_decisions.json"}:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict) or "method" not in d:
            continue
        configs.append({**d, "_path": str(p)})
    return configs


def load_rb_baseline_rows() -> list[dict]:
    """Inject baselines from the existing realbench_steer outputs (DoM, naive
    coef-dir, naive centroid-dir, naive ablate-topK on test_in only). We
    keep these for the unified Pareto plot."""
    base_lr = load_baseline_lr()
    if "test_in" not in base_lr:
        return []
    bh = base_lr["test_in"]["lr_harm_mean"]
    bb = base_lr["test_in"]["lr_ben_mean"]
    rows: list[dict] = []

    if (RB_STEER / "dom.json").exists():
        d = json.load(open(RB_STEER / "dom.json"))
        for a, r in d.get("test_in", {}).items():
            if not isinstance(r, dict) or r.get("lr_harm_mean") is None:
                continue
            rows.append({"method": "DoM", "arm": "—",
                         "alpha": a, "ds": "test_in",
                         "dh": r["lr_harm_mean"] - bh,
                         "db": r["lr_ben_mean"] - bb})

    for arm in ("sae", "tsae", "txc"):
        for d_name, label in (("coef_dir", "naive coef-dir"),
                              ("centroid_dir", "naive centroid-dir"),
                              ("ablate_topK", "naive ablate-topK")):
            p = RB_STEER / f"{arm}_{d_name}.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            data = d.get("test_in", d) if "test_in" in d else d
            if isinstance(data, dict) and data.get("lr_harm_mean") is not None:
                rows.append({"method": label, "arm": ARM_LABEL[arm],
                             "alpha": "ablate", "ds": "test_in",
                             "dh": data["lr_harm_mean"] - bh,
                             "db": data["lr_ben_mean"] - bb})
                continue
            for a, r in (data.items() if isinstance(data, dict) else []):
                if not isinstance(r, dict) or r.get("lr_harm_mean") is None:
                    continue
                rows.append({"method": label, "arm": ARM_LABEL[arm],
                             "alpha": a, "ds": "test_in",
                             "dh": r["lr_harm_mean"] - bh,
                             "db": r["lr_ben_mean"] - bb})
    return rows


# --------------------------------------------------------------------------- #
# bootstrap stats
# --------------------------------------------------------------------------- #

def paired_bootstrap_diff(
    xy: np.ndarray, base_xy: np.ndarray, mask_pos: np.ndarray,
    mask_neg: np.ndarray, n_boot: int = 1000, seed: int = 0,
) -> dict:
    """Bootstrap mean(intervention - baseline) on harmful and benign
    sub-pools separately, then derive leakage = db / dh per resample.

    xy, base_xy:  (N,) per-prompt LR (intervention and baseline)
    """
    rng = np.random.default_rng(seed)
    diffs = xy - base_xy
    pos = diffs[mask_pos]; neg = diffs[mask_neg]
    if len(pos) == 0 or len(neg) == 0:
        return dict(dh_mean=float("nan"), dh_ci=(np.nan, np.nan),
                    db_mean=float("nan"), db_ci=(np.nan, np.nan),
                    leakage_mean=float("nan"),
                    leakage_ci=(np.nan, np.nan))
    dhs = np.empty(n_boot, dtype=np.float64)
    dbs = np.empty(n_boot, dtype=np.float64)
    leaks = np.empty(n_boot, dtype=np.float64)
    np_size = len(pos); nn_size = len(neg)
    for b in range(n_boot):
        ip = rng.integers(0, np_size, np_size)
        ineg = rng.integers(0, nn_size, nn_size)
        dh = pos[ip].mean(); db = neg[ineg].mean()
        dhs[b] = dh; dbs[b] = db
        leaks[b] = db / dh if abs(dh) > 1e-9 else np.nan
    def ci(v):
        v = v[~np.isnan(v)]
        return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))
    return dict(
        dh_mean=float(np.mean(dhs)), dh_ci=ci(dhs),
        db_mean=float(np.mean(dbs)), db_ci=ci(dbs),
        leakage_mean=float(np.nanmean(leaks)), leakage_ci=ci(leaks),
        n_pos=int(np_size), n_neg=int(nn_size),
    )


def wilcoxon_vs_zero(diffs: np.ndarray) -> tuple[float, float]:
    if (diffs == 0).all():
        return 0.0, 1.0
    try:
        s, p = stats.wilcoxon(diffs, alternative="two-sided",
                              zero_method="wilcox")
    except ValueError:
        return 0.0, 1.0
    return float(s), float(p)


def permutation_leakage_test(
    lr_a: np.ndarray, lr_b: np.ndarray, base: np.ndarray,
    mask_pos: np.ndarray, mask_neg: np.ndarray,
    n_perm: int = 5000, seed: int = 0,
) -> dict:
    """Paired-prompt permutation test for H0: leakage(arm A) = leakage(arm B).

    For each prompt we have both arm-A and arm-B intervened LR. Under H0 we
    can swap labels between the two arms per-prompt. Test statistic is the
    absolute difference of computed leakage. Returns p-value, observed gap,
    and per-arm bootstrap leakage means."""
    rng = np.random.default_rng(seed)
    diffs_a = lr_a - base; diffs_b = lr_b - base
    def leak_of(da, db):
        dh = da[mask_pos].mean(); dben = db[mask_neg].mean() if False else None
        # We compute leakage using each arm's own benign too; that's the
        # config-scoped leakage. For paired test we want to compare each
        # arm's leakage as a single scalar.
        return None
    obs_a = diffs_a[mask_pos].mean(), diffs_a[mask_neg].mean()
    obs_b = diffs_b[mask_pos].mean(), diffs_b[mask_neg].mean()
    leak_a_obs = obs_a[1] / obs_a[0] if abs(obs_a[0]) > 1e-9 else float("nan")
    leak_b_obs = obs_b[1] / obs_b[0] if abs(obs_b[0]) > 1e-9 else float("nan")
    obs_gap = leak_a_obs - leak_b_obs

    n_pos = int(mask_pos.sum()); n_neg = int(mask_neg.sum())
    pos_idx = np.where(mask_pos)[0]
    neg_idx = np.where(mask_neg)[0]

    null_gaps = np.zeros(n_perm)
    for k in range(n_perm):
        # swap each prompt's (a,b) pair with prob 0.5
        swap = rng.random(diffs_a.shape) < 0.5
        a_perm = np.where(swap, diffs_b, diffs_a)
        b_perm = np.where(swap, diffs_a, diffs_b)
        dh_a = a_perm[pos_idx].mean(); db_a = a_perm[neg_idx].mean()
        dh_b = b_perm[pos_idx].mean(); db_b = b_perm[neg_idx].mean()
        la = db_a / dh_a if abs(dh_a) > 1e-9 else 0.0
        lb = db_b / dh_b if abs(dh_b) > 1e-9 else 0.0
        null_gaps[k] = la - lb
    p_two = float(np.mean(np.abs(null_gaps) >= abs(obs_gap)))
    return {"leak_a": float(leak_a_obs), "leak_b": float(leak_b_obs),
            "obs_gap": float(obs_gap), "p_two_sided": p_two,
            "n_perm": n_perm}


def run_perm_tests() -> dict:
    """For each (ds, K) compare TXC vs T-SAE and TXC vs SAE under FSGA."""
    base_lr = load_baseline_lr()
    cfgs = all_v2_configs()
    out = {}
    for ds in HARM_DS:
        rows = json.load(open(PROMPTS / f"{ds}.json"))
        y = np.array([r["label"] for r in rows])
        base = np.array(base_lr[ds]["lr"], dtype=np.float64)
        for K in (10, 20, 50):
            d = {arm: None for arm in ("sae", "tsae", "txc")}
            for cfg in cfgs:
                if (cfg["method"] == "S3_FSGA" and cfg["ds"] == ds
                        and cfg.get("K") == K
                        and _short_arm(cfg["arm"]) in d):
                    d[_short_arm(cfg["arm"])] = np.array(cfg["lr"],
                                                         dtype=np.float64)
            if d["txc"] is None:
                continue
            for opp in ("sae", "tsae"):
                if d[opp] is None:
                    continue
                r = permutation_leakage_test(
                    d["txc"], d[opp], base, y == 1, y == 0)
                out[f"txc_vs_{opp}__{ds}__K{K}"] = r
    return out


# --------------------------------------------------------------------------- #
# headline rows: methods × arms × datasets
# --------------------------------------------------------------------------- #

def build_headline_rows() -> list[dict]:
    """For each (method, arm, K, ds) config, attach baseline-aware deltas
    + bootstrap CIs + Wilcoxon p."""
    base_lr = load_baseline_lr()
    cfgs = all_v2_configs()
    out: list[dict] = []
    for cfg in cfgs:
        if cfg["ds"] not in HARM_DS:
            continue  # skip cap_alpaca (KL) for headline
        ds = cfg["ds"]
        rows = json.load(open(PROMPTS / f"{ds}.json"))
        y = np.array([r["label"] for r in rows])
        if "lr" not in cfg:
            continue
        lr = np.array(cfg["lr"], dtype=np.float64)
        base = np.array(base_lr[ds]["lr"], dtype=np.float64)
        bs = paired_bootstrap_diff(lr, base, y == 1, y == 0)
        diffs_h = (lr - base)[y == 1]; diffs_b = (lr - base)[y == 0]
        _, p_h = wilcoxon_vs_zero(diffs_h)
        _, p_b = wilcoxon_vs_zero(diffs_b)
        out.append({
            "method": cfg["method"], "arm": cfg["arm"], "K": cfg.get("K"),
            "ds": ds,
            "dh_mean": bs["dh_mean"], "dh_ci": bs["dh_ci"],
            "db_mean": bs["db_mean"], "db_ci": bs["db_ci"],
            "leakage_mean": bs["leakage_mean"], "leakage_ci": bs["leakage_ci"],
            "wilcoxon_p_h": p_h, "wilcoxon_p_b": p_b,
            "n_pos": bs["n_pos"], "n_neg": bs["n_neg"],
        })
    return out


def derive_cfsga_rows() -> list[dict]:
    """For each (arm, K, ds) S3 row + matching probe decisions on that ds,
    derive the cFSGA per-prompt LR vector and append a headline row."""
    base_lr = load_baseline_lr()
    decisions = load_probe_decisions()
    out: list[dict] = []
    for cfg in all_v2_configs():
        if cfg["method"] != "S3_FSGA":
            continue
        if cfg["ds"] not in HARM_DS:
            continue
        ds = cfg["ds"]
        if ds not in decisions:
            continue
        d = np.array(decisions[ds], dtype=int)
        rows = json.load(open(PROMPTS / f"{ds}.json"))
        y = np.array([r["label"] for r in rows])
        lr_fsga = np.array(cfg["lr"], dtype=np.float64)
        lr_base = np.array(base_lr[ds]["lr"], dtype=np.float64)
        # cFSGA: pick FSGA where probe says 1, else baseline
        lr = np.where(d == 1, lr_fsga, lr_base)
        bs = paired_bootstrap_diff(lr, lr_base, y == 1, y == 0)
        _, p_h = wilcoxon_vs_zero((lr - lr_base)[y == 1])
        _, p_b = wilcoxon_vs_zero((lr - lr_base)[y == 0])
        out.append({
            "method": "S5_cFSGA", "arm": cfg["arm"], "K": cfg["K"],
            "ds": ds,
            "dh_mean": bs["dh_mean"], "dh_ci": bs["dh_ci"],
            "db_mean": bs["db_mean"], "db_ci": bs["db_ci"],
            "leakage_mean": bs["leakage_mean"], "leakage_ci": bs["leakage_ci"],
            "wilcoxon_p_h": p_h, "wilcoxon_p_b": p_b,
            "n_pos": bs["n_pos"], "n_neg": bs["n_neg"],
            "n_fired": int((d == 1).sum()),
            "n_total": int(len(d)),
        })
    return out


# --------------------------------------------------------------------------- #
# capability section
# --------------------------------------------------------------------------- #

def capability_rows() -> list[dict]:
    """KL on cap_alpaca (per-prompt KL of intervened first-token vs baseline).

    Also derives the cFSGA capability cost from probe decisions: KL is the
    intervention's KL where the probe fires, else 0.
    """
    out: list[dict] = []
    for cfg in all_v2_configs():
        if cfg["ds"] != "cap_alpaca":
            continue
        if "kl_per_prompt" not in cfg:
            continue
        kl = np.array(cfg["kl_per_prompt"], dtype=np.float64)
        out.append({
            "method": cfg["method"], "arm": cfg["arm"],
            "K": cfg.get("K"), "ds": "cap_alpaca",
            "kl_mean": float(kl.mean()),
            "kl_median": float(np.median(kl)),
            "kl_p95": float(np.quantile(kl, 0.95)),
            "kl_max": float(np.max(kl)),
        })

    # cFSGA derived capability rows: pick S3's KL where probe fires, else 0.
    try:
        decisions = json.load(open(OUT / "probe_decisions.json"))
        d = np.array(decisions["cap_alpaca"], dtype=int) if "cap_alpaca" in decisions else None
    except FileNotFoundError:
        d = None
    if d is not None:
        for cfg in all_v2_configs():
            if (cfg["method"] == "S3_FSGA" and cfg["ds"] == "cap_alpaca"
                    and "kl_per_prompt" in cfg):
                kl_full = np.array(cfg["kl_per_prompt"], dtype=np.float64)
                kl_cf = np.where(d == 1, kl_full, 0.0)
                out.append({
                    "method": "S5_cFSGA", "arm": cfg["arm"],
                    "K": cfg.get("K"), "ds": "cap_alpaca",
                    "kl_mean": float(kl_cf.mean()),
                    "kl_median": float(np.median(kl_cf)),
                    "kl_p95": float(np.quantile(kl_cf, 0.95)),
                    "kl_max": float(np.max(kl_cf)),
                    "n_fired": int((d == 1).sum()),
                    "n_total": int(len(d)),
                })
    return out


def mmlu_rows() -> list[dict]:
    base = json.load(open(OUT / "baseline_mmlu.json"))
    out = [{"method": "baseline", "arm": "—", "K": None, "ds": "cap_mmlu",
            "acc": base["acc"]}]
    for p in sorted(OUT.glob("*cap_mmlu.json")):
        if p.name == "baseline_mmlu.json":
            continue
        d = json.load(open(p))
        out.append({"method": d["method"], "arm": d["arm"], "K": d.get("K"),
                    "ds": "cap_mmlu", "acc": d.get("acc")})
    return out


# --------------------------------------------------------------------------- #
# best-config selection per (method, arm)
# --------------------------------------------------------------------------- #

def best_per_method_arm(rows: list[dict]) -> list[dict]:
    """Among S3 rows, pick the K with the largest |dh| at acceptable leakage
    (smallest |leakage| among rows whose |dh| ≥ 0.1)."""
    by_key: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        key = (r["method"], r["arm"], r["ds"])
        by_key.setdefault(key, []).append(r)
    out = []
    for key, lst in by_key.items():
        # filter to rows with non-trivial dh; if none, fall back to argmax|dh|
        cand = [r for r in lst if abs(r["dh_mean"]) >= 0.1]
        if not cand:
            cand = lst
        cand.sort(key=lambda r: abs(r["leakage_mean"]) if not math.isnan(r["leakage_mean"]) else 1e9)
        out.append(cand[0])
    return out


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #

def fig_kcurve(rows: list[dict]) -> None:
    """K vs leakage (and |dh|) for S3 FSGA, per arm × dataset.

    Leakage y-axis is clipped to [-0.5, 1.5] for readability — far outlier
    leakages (typically from small-|dh| T-SAE configs where the ratio is
    unstable) are not informative for the headline picture.
    """
    s3 = [r for r in rows if r["method"] == "S3_FSGA"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row")
    for j, ds in enumerate(HARM_DS):
        for arm in ARM_COLOR:
            sub = sorted([r for r in s3 if r["arm"] == arm and r["ds"] == ds],
                         key=lambda r: r["K"])
            if not sub:
                continue
            ks = [r["K"] for r in sub]
            leak_mean = [r["leakage_mean"] for r in sub]
            leak_lo = [r["leakage_ci"][0] for r in sub]
            leak_hi = [r["leakage_ci"][1] for r in sub]
            dh = [r["dh_mean"] for r in sub]
            dh_lo = [r["dh_ci"][0] for r in sub]
            dh_hi = [r["dh_ci"][1] for r in sub]
            axes[0, j].plot(ks, leak_mean, "-o", color=ARM_COLOR[arm],
                            label=ARM_LABEL[arm], linewidth=1.6)
            axes[0, j].fill_between(ks, leak_lo, leak_hi,
                                    color=ARM_COLOR[arm], alpha=0.15)
            axes[1, j].plot(ks, dh, "-o", color=ARM_COLOR[arm],
                            label=ARM_LABEL[arm], linewidth=1.6)
            axes[1, j].fill_between(ks, dh_lo, dh_hi,
                                    color=ARM_COLOR[arm], alpha=0.15)
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
        axes[0, j].set_title(DS_LABEL[ds])
        axes[0, j].axhline(0, color="k", linewidth=0.6, linestyle=":")
        axes[1, j].axhline(0, color="k", linewidth=0.6, linestyle=":")
        axes[1, j].set_xlabel("K (gated features)")
    axes[0, 0].set_ylabel("leakage db/dh\n(0=ideal)")
    axes[1, 0].set_ylabel("ΔLR_harm (more negative = stronger jailbreak)")
    axes[0, 0].legend(loc="best", fontsize=8)
    for j in range(3):
        axes[0, j].set_ylim(-0.3, 1.5)
    fig.suptitle("FSGA K-sweep with 95% bootstrap CIs — across 3 arms × 3 H/B datasets\n"
                 "(top-row y-axis clipped to [-0.3, 1.5] for readability)")
    fig.tight_layout()
    fig.savefig(FIG / "andre_steer_v2_kcurve.png", dpi=140)
    plt.close(fig)


def fig_pareto(rows: list[dict], rb_rows: list[dict]) -> None:
    """Pareto-style scatter of dh vs db on test_in, with all methods."""
    base_lr = load_baseline_lr()
    bh = base_lr["test_in"]["lr_harm_mean"]
    bb = base_lr["test_in"]["lr_ben_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    method_color = {
        "DoM": "#000",
        "naive coef-dir": "#999",
        "naive centroid-dir": "#666",
        "naive ablate-topK": "#444",
        "S3_FSGA": "#dc2626",
        "S4_FSGA_clamp": "#fb923c",
        "S5_cFSGA": "#9333ea",
        "S6_FSGA_probecoef": "#0ea5e9",
    }
    arm_marker = {"sae": "s", "tsae": "^", "txc": "D", "—": "o",
                  "SAE (T=1)": "s", "T-SAE (T=5)": "^", "TXC (T=5)": "D"}
    for r in rows:
        if r["ds"] != "test_in":
            continue
        ax = axes[0]
        c = method_color.get(r["method"], "#888")
        m = arm_marker.get(r["arm"], "o")
        sz = 80 + 4 * (r.get("K") or 0)
        ax.errorbar(r["dh_mean"], r["db_mean"],
                    xerr=[[r["dh_mean"] - r["dh_ci"][0]],
                          [r["dh_ci"][1] - r["dh_mean"]]],
                    yerr=[[r["db_mean"] - r["db_ci"][0]],
                          [r["db_ci"][1] - r["db_mean"]]],
                    fmt=m, color=c, ecolor=c, alpha=0.85,
                    markersize=math.sqrt(sz), elinewidth=0.7,
                    markeredgecolor="k", markeredgewidth=0.5)
        # zoom panel
        axz = axes[1]
        axz.errorbar(r["dh_mean"], r["db_mean"],
                     xerr=[[r["dh_mean"] - r["dh_ci"][0]],
                           [r["dh_ci"][1] - r["dh_mean"]]],
                     yerr=[[r["db_mean"] - r["db_ci"][0]],
                           [r["db_ci"][1] - r["db_mean"]]],
                     fmt=m, color=c, ecolor=c, alpha=0.85,
                     markersize=math.sqrt(sz), elinewidth=0.7,
                     markeredgecolor="k", markeredgewidth=0.5)

    for r in rb_rows:
        if r["ds"] != "test_in":
            continue
        c = method_color.get(r["method"], "#888")
        m = arm_marker.get(r["arm"], "o")
        for ax in axes:
            ax.scatter(r["dh"], r["db"], color=c, marker=m, s=60, alpha=0.6,
                       edgecolors="k", linewidths=0.4)
    # iso-leakage diagonals
    for slope, color, ls in [(0.0, "#22c55e", "-"),
                             (0.5, "#a3a3a3", "--"),
                             (1.0, "#a3a3a3", "--"),
                             (2.0, "#a3a3a3", "--")]:
        xs = np.array([-1.0, 1.0])
        axes[1].plot(xs, slope * xs, ls, color=color, linewidth=0.8,
                     alpha=0.7, label=f"leakage={slope:.1f}")

    handles = [mpatches.Patch(color=c, label=label)
               for label, c in method_color.items()]
    axes[0].legend(handles=handles, fontsize=7, loc="lower left")
    axes[1].legend(fontsize=7, loc="upper left")
    for ax, ttl, lim in ((axes[0], "full range", None),
                          (axes[1], "zoom: ±1 nat", 1.0)):
        ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="k", linewidth=0.5, linestyle=":")
        ax.set_xlabel("ΔLR on harmful (more negative = stronger jailbreak)")
        ax.set_ylabel("ΔLR on benign (want ≈ 0)")
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        if lim is not None:
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    fig.suptitle("Steering Pareto on JBB test_in — error bars = 95% bootstrap CI")
    fig.tight_layout()
    fig.savefig(FIG / "andre_steer_v2_pareto.png", dpi=140)
    plt.close(fig)


def fig_capability_tradeoff(headline_rows: list[dict],
                            cap_rows: list[dict]) -> None:
    """Scatter: |dh| (jailbreak strength on test_in) vs KL on cap_alpaca.
    Lower-right = good (strong jailbreak, low capability cost)."""
    cap_idx = {(r["method"], r["arm"], r.get("K")): r for r in cap_rows}
    fig, ax = plt.subplots(figsize=(8.5, 6))
    seen_legend = set()
    for r in headline_rows:
        if r["ds"] != "test_in":
            continue
        c = cap_idx.get((r["method"], r["arm"], r.get("K")))
        if c is None:
            continue
        col = ARM_COLOR.get(_short_arm(r["arm"]), "#666")
        marker = {"S3_FSGA": "D", "S4_FSGA_clamp": "P", "S5_cFSGA": "*",
                  "S6_FSGA_probecoef": "X"}.get(r["method"], "o")
        sz = 60 + 2 * (r.get("K") or 0)
        lab = f"{r['method']} {r['arm']}"
        ax.scatter(abs(r["dh_mean"]), c["kl_mean"],
                   color=col, marker=marker, s=sz, alpha=0.85,
                   edgecolors="k", linewidths=0.5,
                   label=lab if lab not in seen_legend else None)
        seen_legend.add(lab)
    ax.set_xlabel("|ΔLR_harm| on JBB (jailbreak strength) — bigger = stronger")
    ax.set_ylabel("KL(base ‖ steered) on benign Alpaca-200 — smaller = better")
    ax.grid(alpha=0.3)
    ax.set_title("Capability vs jailbreak Pareto — top-right = strong but costly,\n"
                 "bottom-right = the prize (strong + cheap)")
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "andre_steer_v2_capability.png", dpi=140)
    plt.close(fig)


def fig_per_dataset_bars(headline_rows: list[dict]) -> None:
    """Per-dataset bar chart of leakage and |dh| across arms × selected
    methods (S3 K=20, S4, S5, S6)."""
    methods = ["S3_FSGA", "S4_FSGA_clamp", "S5_cFSGA", "S6_FSGA_probecoef"]
    arms = ["sae", "tsae", "txc"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey="row")
    width = 0.22
    for j, ds in enumerate(HARM_DS):
        for ax_i, key in enumerate(("leakage_mean", "dh_mean")):
            ax = axes[ax_i, j]
            for ai, arm in enumerate(arms):
                vals = []
                errs_lo = []; errs_hi = []
                for m in methods:
                    cand = [r for r in headline_rows
                            if r["method"] == m and r["ds"] == ds
                            and _short_arm(r["arm"]) == arm
                            and (r.get("K") in (20, None))]
                    if not cand:
                        vals.append(np.nan); errs_lo.append(0); errs_hi.append(0)
                    else:
                        r = cand[0]
                        vals.append(r[key])
                        ci = r["leakage_ci" if key == "leakage_mean" else "dh_ci"]
                        errs_lo.append(r[key] - ci[0])
                        errs_hi.append(ci[1] - r[key])
                xs = np.arange(len(methods)) + (ai - 1) * width
                ax.bar(xs, vals, width=width, color=ARM_COLOR[arm],
                       label=ARM_LABEL[arm], yerr=[errs_lo, errs_hi],
                       capsize=3, alpha=0.9, edgecolor="k", linewidth=0.4)
            ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
            ax.set_xticks(np.arange(len(methods)))
            ax.set_xticklabels([m.replace("S3_FSGA", "S3 FSGA")
                                .replace("S4_FSGA_clamp", "S4 clamp")
                                .replace("S5_cFSGA", "S5 cFSGA")
                                .replace("S6_FSGA_probecoef", "S6 probe-rank")
                                for m in methods], rotation=20, ha="right",
                               fontsize=8)
            ax.grid(alpha=0.25, axis="y")
            if j == 0:
                ax.set_ylabel("leakage db/dh" if key == "leakage_mean"
                              else "ΔLR_harm")
            if ax_i == 0:
                ax.set_title(DS_LABEL[ds])
    for j in range(3):
        axes[0, j].set_ylim(-1, 2)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Per-dataset comparison at K=20: leakage (top, clipped to [-1, 2]) "
                 "and ΔLR_harm (bottom)\nError bars: 95% bootstrap CI")
    fig.tight_layout()
    fig.savefig(FIG / "andre_steer_v2_perdataset.png", dpi=140)
    plt.close(fig)


def fig_dh_db_distribution(headline_rows: list[dict]) -> None:
    """Per-prompt ΔLR distributions (harmful vs benign) for the 3 arms at the
    headline FSGA config (K=20, test_in)."""
    base_lr = load_baseline_lr()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for j, arm in enumerate(("sae", "tsae", "txc")):
        ax = axes[j]
        for cfg in all_v2_configs():
            if (cfg["method"] == "S3_FSGA" and cfg["arm"] == arm
                and cfg.get("K") == 20 and cfg["ds"] == "test_in"):
                lr = np.array(cfg["lr"], dtype=np.float64)
                base = np.array(base_lr["test_in"]["lr"], dtype=np.float64)
                rows = json.load(open(PROMPTS / "test_in.json"))
                y = np.array([r["label"] for r in rows])
                diffs = lr - base
                bins = np.linspace(-25, 5, 35)
                ax.hist(diffs[y == 1], bins=bins, alpha=0.6, color="#dc2626",
                        edgecolor="k", linewidth=0.4, label="harmful")
                ax.hist(diffs[y == 0], bins=bins, alpha=0.6, color="#16a34a",
                        edgecolor="k", linewidth=0.4, label="benign")
                ax.axvline(0, color="k", linewidth=0.5, linestyle=":")
                ax.set_title(f"{ARM_LABEL[arm]}  FSGA K=20")
                ax.set_xlabel("ΔLR (intervened − baseline)")
                if j == 0:
                    ax.set_ylabel("count of prompts")
                ax.grid(alpha=0.25)
                if j == 0:
                    ax.legend(fontsize=8)
    fig.suptitle("FSGA K=20 per-prompt ΔLR distributions — JBB test_in")
    fig.tight_layout()
    fig.savefig(FIG / "andre_steer_v2_distribution.png", dpi=140)
    plt.close(fig)


def fig_kl_kcurve(cap_rows: list[dict]) -> None:
    """KL on cap_alpaca as K grows — capability cost curve."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for arm in ("sae", "tsae", "txc"):
        sub = sorted([r for r in cap_rows
                      if r["method"] == "S3_FSGA" and r["arm"] == arm],
                     key=lambda r: r["K"] or 0)
        ks = [r["K"] for r in sub if r["K"] is not None]
        kl = [r["kl_mean"] for r in sub if r["K"] is not None]
        ax.plot(ks, kl, "-o", color=ARM_COLOR[arm], label=ARM_LABEL[arm],
                linewidth=1.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K (gated features)")
    ax.set_ylabel("mean KL(base ‖ steered) on benign Alpaca-200 (nats)")
    ax.set_title("Capability cost grows with K — TXC has the lowest cost curve")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "andre_steer_v2_kl_kcurve.png", dpi=140)
    plt.close(fig)


def fig_xstest_breakdown() -> None:
    """For XSTest, show ΔLR at K=20 FSGA per-category, separated into
    'safe' subtypes (label=0) and 'unsafe' subtypes (label=1)."""
    rows = json.load(open(PROMPTS / "test_ood.json"))
    cats = sorted({r["category"] for r in rows})
    base_lr = load_baseline_lr()
    base = np.array(base_lr["test_ood"]["lr"], dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, arm in zip(axes, ("sae", "tsae", "txc")):
        cfg = next((c for c in all_v2_configs()
                    if c["method"] == "S3_FSGA" and _short_arm(c["arm"]) == arm
                    and c.get("K") == 20 and c["ds"] == "test_ood"), None)
        if cfg is None:
            ax.set_title(f"{ARM_LABEL[arm]}  (no data)")
            continue
        lr = np.array(cfg["lr"], dtype=np.float64)
        diffs = lr - base
        ys_safe = []
        ys_unsafe = []
        labels_safe = []
        labels_unsafe = []
        for c in cats:
            mask = np.array([r["category"] == c for r in rows])
            y = np.array([r["label"] for r in rows])[mask]
            d = diffs[mask]
            if (y == 0).any():
                labels_safe.append(c); ys_safe.append(d[y == 0].mean())
            if (y == 1).any():
                labels_unsafe.append(c); ys_unsafe.append(d[y == 1].mean())
        x_s = np.arange(len(labels_safe))
        x_u = np.arange(len(labels_unsafe)) + len(labels_safe) + 1
        ax.barh(x_s, ys_safe, color="#16a34a", alpha=0.7, edgecolor="k",
                linewidth=0.4, label="safe (benign)")
        ax.barh(x_u, ys_unsafe, color="#dc2626", alpha=0.7, edgecolor="k",
                linewidth=0.4, label="unsafe (harmful)")
        ax.set_yticks(list(x_s) + list(x_u))
        ax.set_yticklabels(labels_safe + labels_unsafe, fontsize=7)
        ax.axvline(0, color="k", linewidth=0.6, linestyle=":")
        ax.set_title(f"{ARM_LABEL[arm]} FSGA K=20 — XSTest")
        ax.set_xlabel("ΔLR (intervened − baseline) — mean per category")
        ax.grid(alpha=0.25, axis="x")
        if arm == "sae":
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("XSTest per-category ΔLR — green should stay near 0 (benign), "
                 "red should be deeply negative (harmful)")
    fig.tight_layout()
    fig.savefig(FIG / "andre_steer_v2_xstest.png", dpi=140)
    plt.close(fig)


def fig_mmlu_bars(mmlu: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = []; vals = []; colors = []
    for r in mmlu:
        labels.append(f"{r['method']}\n{r['arm']}" if r["arm"] != "—"
                      else r["method"])
        vals.append(r["acc"])
        if r["method"] == "baseline":
            colors.append("#374151")
        else:
            arm_short = _short_arm(r["arm"])
            colors.append(ARM_COLOR.get(arm_short, "#888"))
    xs = np.arange(len(labels))
    ax.bar(xs, vals, color=colors, edgecolor="k", linewidth=0.5)
    for x, v in zip(xs, vals):
        if v is None:
            continue
        ax.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    if mmlu and mmlu[0]["method"] == "baseline":
        ax.axhline(mmlu[0]["acc"], color="#374151", linestyle=":",
                   linewidth=1, alpha=0.6)
    ax.set_ylabel("MMLU-mini accuracy (n=100)")
    ax.set_title("MMLU under FSGA K=20 — capability regression check")
    ax.set_ylim(0, max(0.7, max(v for v in vals if v is not None) * 1.1))
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "andre_steer_v2_mmlu.png", dpi=140)
    plt.close(fig)


def fig_dose_response_inject(rows_rb: list[dict]) -> None:
    """Existing realbench inject methods: dose-response on test_in."""
    pass  # placeholder if needed; existing realbench fig already covers this


def _short_arm(arm: str) -> str:
    return {"SAE (T=1)": "sae", "T-SAE (T=5)": "tsae", "TXC (T=5)": "txc",
            "—": "—"}.get(arm, arm)


# --------------------------------------------------------------------------- #
# macros — every inline number in the report has to come from here
# --------------------------------------------------------------------------- #

def write_macros(headline: list[dict], cap_rows: list[dict],
                 mmlu: list[dict]) -> dict[str, str]:
    macros: dict[str, str] = {}

    def reg(name: str, value, fmt: str = "{:+.3f}") -> None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            macros[name] = "n/a"
            return
        macros[name] = fmt.format(value) if isinstance(value, (int, float)) else str(value)

    base = load_baseline_lr()
    for ds in HARM_DS:
        if ds in base:
            reg(f"baseHarm_{ds}", base[ds]["lr_harm_mean"])
            reg(f"baseBen_{ds}",  base[ds]["lr_ben_mean"])

    # Probe AUC + FP/FN per dataset (cFSGA quality)
    try:
        decisions = json.load(open(OUT / "probe_decisions.json"))
        from sklearn.metrics import roc_auc_score
        for ds in HARM_DS + ("cap_alpaca",):
            rows = json.load(open(ROOT / "results" / "realbench" /
                                  f"{ds}.json"))
            y = np.array([r["label"] for r in rows])
            p = np.array(decisions[f"{ds}_p"])
            n_fire = int((p > 0.5).sum())
            reg(f"probeFires_{ds}", n_fire, "{:d}")
            if (y == 1).any() and (y == 0).any():
                reg(f"probeAuc_{ds}", float(roc_auc_score(y, p)), "{:.3f}")
                reg(f"probeFP_{ds}", int(((p > 0.5) & (y == 0)).sum()), "{:d}")
                reg(f"probeFN_{ds}", int(((p <= 0.5) & (y == 1)).sum()), "{:d}")
    except FileNotFoundError:
        pass

    # FSGA K=20 headline numbers, per arm × ds
    for arm in ("sae", "tsae", "txc"):
        for ds in HARM_DS:
            cand = [r for r in headline
                    if r["method"] == "S3_FSGA" and _short_arm(r["arm"]) == arm
                    and r["ds"] == ds and r.get("K") == 20]
            if not cand:
                continue
            r = cand[0]
            reg(f"fsgaDh_{arm}_{ds}", r["dh_mean"])
            reg(f"fsgaDhLo_{arm}_{ds}", r["dh_ci"][0])
            reg(f"fsgaDhHi_{arm}_{ds}", r["dh_ci"][1])
            reg(f"fsgaDb_{arm}_{ds}", r["db_mean"])
            reg(f"fsgaLeak_{arm}_{ds}", r["leakage_mean"], "{:+.2f}")
            reg(f"fsgaLeakLo_{arm}_{ds}", r["leakage_ci"][0], "{:+.2f}")
            reg(f"fsgaLeakHi_{arm}_{ds}", r["leakage_ci"][1], "{:+.2f}")
            reg(f"fsgaPh_{arm}_{ds}", r["wilcoxon_p_h"], "{:.2e}")
            reg(f"fsgaPb_{arm}_{ds}", r["wilcoxon_p_b"], "{:.2e}")
    # cFSGA / clamp / probe-rank — pin to K=20 for the headline; cFSGA can
    # exist at any K (since it's derived from each S3 K), so we filter to K=20.
    for m_short, m_full in [("cfsga", "S5_cFSGA"),
                            ("clamp", "S4_FSGA_clamp"),
                            ("pcrank", "S6_FSGA_probecoef")]:
        for arm in ("sae", "tsae", "txc"):
            for ds in HARM_DS:
                cand = [r for r in headline
                        if r["method"] == m_full and _short_arm(r["arm"]) == arm
                        and r["ds"] == ds and (r.get("K") == 20)]
                if not cand:
                    continue
                r = cand[0]
                reg(f"{m_short}Dh_{arm}_{ds}", r["dh_mean"])
                reg(f"{m_short}Db_{arm}_{ds}", r["db_mean"])
                reg(f"{m_short}Leak_{arm}_{ds}", r["leakage_mean"], "{:+.2f}")
                if "n_fired" in r:
                    reg(f"{m_short}Fired_{arm}_{ds}", r["n_fired"], "{:d}")
    # KL
    for arm in ("sae", "tsae", "txc"):
        for K in (1, 2, 5, 10, 20, 50, 100):
            cand = [r for r in cap_rows
                    if r["method"] == "S3_FSGA"
                    and _short_arm(r["arm"]) == arm and r["K"] == K]
            if cand:
                reg(f"klFsga_{arm}_K{K}", cand[0]["kl_mean"], "{:.4f}")
    # MMLU
    for r in mmlu:
        if r["method"] == "baseline":
            reg("mmluBase", r["acc"], "{:.3f}")
        else:
            arm_short = _short_arm(r["arm"])
            reg(f"mmlu_{arm_short}_{r['method']}", r["acc"], "{:.3f}")

    # Best leakage rank summary (lowest |leakage| at K=20 on test_in)
    cands = [r for r in headline
             if r["method"] == "S3_FSGA" and r.get("K") == 20
             and r["ds"] == "test_in"]
    cands.sort(key=lambda r: abs(r["leakage_mean"]) if not math.isnan(r["leakage_mean"]) else 1e9)
    if cands:
        reg("bestArm", _short_arm(cands[0]["arm"]).upper(), "{}")
        reg("bestArmLeak", cands[0]["leakage_mean"], "{:+.2f}")

    # cFSGA across K — find the K* with largest |dh| and reasonable leakage.
    # Reports per-arm "headline cFSGA" macros at the chosen K.
    for arm in ("sae", "tsae", "txc"):
        cf = [r for r in headline
              if r["method"] == "S5_cFSGA" and _short_arm(r["arm"]) == arm
              and r["ds"] == "test_in"
              and not math.isnan(r["dh_mean"])]
        if not cf:
            continue
        # Pick the K with the largest |dh| (most aggressive cFSGA setting).
        best = max(cf, key=lambda r: abs(r["dh_mean"]))
        reg(f"cfBestK_{arm}", best.get("K"), "{:d}")
        reg(f"cfBestDh_{arm}", best["dh_mean"])
        reg(f"cfBestDb_{arm}", best["db_mean"])
        reg(f"cfBestLeak_{arm}", best["leakage_mean"], "{:+.2f}")
        # cFSGA KL is ALWAYS 0 on cap_alpaca because probe fires 0/200,
        # but include it for completeness.
        kl = next((c for c in cap_rows if c["method"] == "S5_cFSGA"
                   and _short_arm(c["arm"]) == arm and c["K"] == best.get("K")),
                  None)
        if kl:
            reg(f"cfBestKL_{arm}", kl["kl_mean"], "{:.4f}")

    # Saturation: max |dh| ever reached by any K under FSGA, per arm.
    # Useful when arms have wildly different per-K magnitudes.
    for arm in ("sae", "tsae", "txc"):
        sub = [r for r in headline
               if r["method"] == "S3_FSGA" and _short_arm(r["arm"]) == arm
               and r["ds"] == "test_in"
               and not math.isnan(r["dh_mean"])]
        if not sub:
            continue
        sat = max(sub, key=lambda r: abs(r["dh_mean"]))
        reg(f"satK_{arm}", sat.get("K"), "{:d}")
        reg(f"satDh_{arm}", sat["dh_mean"])
        reg(f"satLeak_{arm}", sat["leakage_mean"], "{:+.2f}")

    # Capability cost at iso-effect: for each arm, the K that hit
    # |dh|=target, look up its KL on cap_alpaca.
    target_dh = 5.0
    for arm in ("sae", "tsae", "txc"):
        sub = sorted(
            [r for r in headline
             if r["method"] == "S3_FSGA" and _short_arm(r["arm"]) == arm
             and r["ds"] == "test_in" and not math.isnan(r["dh_mean"])],
            key=lambda r: r.get("K") or 999)
        match = next((r for r in sub if abs(r["dh_mean"]) >= target_dh), None)
        if match is None and sub:
            match = max(sub, key=lambda r: abs(r["dh_mean"]))
        if match is None:
            continue
        # find KL at the matching K
        kl_match = next(
            (c for c in cap_rows if c["method"] == "S3_FSGA"
             and _short_arm(c["arm"]) == arm and c["K"] == match.get("K")),
            None)
        if kl_match:
            reg(f"isoKL_{arm}", kl_match["kl_mean"], "{:.4f}")

    # Iso-effect comparison: for each arm, the smallest K such that
    # |dh_mean| >= 5 nats on test_in. Reports the leakage at that K,
    # giving a fairer cross-arm comparison than fixed-K. If no K reaches
    # the target |dh|, falls back to the K with the largest |dh| achieved.
    target_dh = 5.0
    for arm in ("sae", "tsae", "txc"):
        sub = sorted(
            [r for r in headline
             if r["method"] == "S3_FSGA" and _short_arm(r["arm"]) == arm
             and r["ds"] == "test_in" and not math.isnan(r["dh_mean"])],
            key=lambda r: r.get("K") or 999)
        match = next((r for r in sub if abs(r["dh_mean"]) >= target_dh), None)
        if match is None and sub:
            match = max(sub, key=lambda r: abs(r["dh_mean"]))
            reg(f"isoMode_{arm}", "max-K (target unreachable)", "{}")
        elif match is not None:
            reg(f"isoMode_{arm}", "first-K-above-target", "{}")
        if match is not None:
            reg(f"isoK_{arm}", match.get("K"), "{:d}")
            reg(f"isoDh_{arm}", match["dh_mean"])
            reg(f"isoLeak_{arm}", match["leakage_mean"], "{:+.2f}")

    # Macro output: TeX (\newcommand — TeX command names cannot contain
    # underscores, so we strip them and CamelCase the suffix on the way out)
    # and JSON (kept verbose with underscores for the markdown report).
    def tex_key(k: str) -> str:
        parts = k.split("_")
        return parts[0] + "".join(p.capitalize() for p in parts[1:])

    tex_lines = ["% Auto-generated by v2_analysis.py — do not hand-edit.\n"]
    for k, v in sorted(macros.items()):
        tk = tex_key(k)
        tex_lines.append(rf"\newcommand{{\{tk}}}{{{v}}}" + "\n")
    (OUT / "paper_macros.tex").write_text("".join(tex_lines))
    json.dump(macros, open(OUT / "paper_macros.json", "w"), indent=1)
    return macros


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    headline = build_headline_rows()
    cfsga = derive_cfsga_rows()
    headline_with_cf = headline + cfsga
    cap = capability_rows()
    try:
        mmlu = mmlu_rows()
    except FileNotFoundError:
        mmlu = []
    rb_rows = load_rb_baseline_rows()

    json.dump({"headline": headline_with_cf, "capability": cap,
               "mmlu": mmlu, "rb_baselines": rb_rows},
              open(OUT / "all_rows.json", "w"), indent=1)

    # high-level stats summary
    stats_out = {
        "n_headline_rows": len(headline_with_cf),
        "n_cap_rows": len(cap),
        "n_mmlu_rows": len(mmlu),
    }
    json.dump(stats_out, open(OUT / "stats.json", "w"), indent=1)

    fig_kcurve(headline)
    fig_pareto(headline_with_cf, rb_rows)
    fig_capability_tradeoff(headline_with_cf, cap)
    fig_per_dataset_bars(headline_with_cf)
    fig_dh_db_distribution(headline_with_cf)
    fig_kl_kcurve(cap)
    fig_xstest_breakdown()
    if mmlu:
        fig_mmlu_bars(mmlu)

    perms = run_perm_tests()
    json.dump(perms, open(OUT / "perm_tests.json", "w"), indent=1)
    if perms:
        print("Permutation tests (TXC vs SAE/T-SAE leakage, K=20):")
        for k, r in perms.items():
            if "K20" not in k or "test_in" not in k:
                continue
            print(f"  {k}: leak_txc={r['leak_a']:+.2f} leak_other={r['leak_b']:+.2f} "
                  f"gap={r['obs_gap']:+.2f}  p={r['p_two_sided']:.3g}")

    macros = write_macros(headline_with_cf, cap, mmlu)
    print(f"wrote {len(macros)} macros to {OUT/'paper_macros.tex'}")
    print("Top-5 lowest-leakage configs at K=20 on test_in:")
    cands = sorted([r for r in headline_with_cf
                    if r.get("K") in (20, None) and r["ds"] == "test_in"],
                   key=lambda r: abs(r["leakage_mean"])
                   if not math.isnan(r["leakage_mean"]) else 1e9)
    for r in cands[:8]:
        print(f"  {r['method']:<24s} {r['arm']:<14s}  "
              f"dh={r['dh_mean']:+.3f}  db={r['db_mean']:+.3f}  "
              f"leak={r['leakage_mean']:+.2f}  "
              f"CI=[{r['leakage_ci'][0]:+.2f},{r['leakage_ci'][1]:+.2f}]")


if __name__ == "__main__":
    main()
