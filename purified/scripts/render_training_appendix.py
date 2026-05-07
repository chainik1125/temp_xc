"""Auto-generate the training-details appendix for all 7 paper components.

Reads (sources of truth):
- ``results/leaderboard.jsonl`` (cell universe, filters smoke + component scope)
- ``checkpoints/manifest.jsonl`` (training_cfg per train_key)
- ``configs/locked_archs.yaml`` (arch base + per-component hparams)
- ``configs/datasources.yaml`` (subject model + layer + dataset)

Writes:
- ``docs/paper/training_appendix.md``

Run via:
    .venv/bin/python scripts/render_training_appendix.py

Idempotent — re-running rewrites the file with current data. Hand-editing
the file is forbidden; numbers are derived from the on-disk state.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
MANIFEST = ROOT / "checkpoints" / "manifest.jsonl"
ARCHS_YAML = ROOT / "configs" / "locked_archs.yaml"
DATASOURCES_YAML = ROOT / "configs" / "datasources.yaml"
OUT_MD = ROOT / "docs" / "paper" / "training_appendix.md"

# Paper-section ordering. C2 absorbs c1_noisy (Setup B per c2.md).
COMPONENT_GROUPS: list[tuple[str, str, list[str]]] = [
    ("c1", "C1 — Synthetic TopK sweep (toy Markov features)", ["c1"]),
    ("c2", "C2 — Synthetic coupled features + noisy emissions", ["c2", "c1_noisy"]),
    ("c3", "C3 — Sparse probing on Gemma-2-2B-IT L13", ["c3"]),
    ("c4", "C4 — Qualitative latents on Gemma-2-2B-IT L13", ["c4"]),
    ("c5", "C5 — RLHF steering (Gemma-2-2B-IT L13)", ["c5"]),
    ("c6", "C6 — Emergent misalignment (Qwen-2.5 14B / 7B Instruct)", ["c6"]),
    ("c7", "C7 — Backtracking on Llama-3.1-8B-BASE L10", ["c7"]),
]

# Order of arches in tables (rough cross-paper ranking by where they
# typically land in headlines).
ARCH_ORDER = [
    "txc_base", "txc_pro", "txc_base_mw", "txc_pro_mw",
    "tsae_paper", "tsae_ours",
    "topk_sae", "stacked_sae", "tfa", "tfa_pos", "mlc",
    "sae_arditi",
]

ARCH_DISPLAY = {
    "txc_base":     "TXC-base",
    "txc_pro":      "TXC-pro",
    "txc_base_mw":  "TXC-base (MW)",
    "txc_pro_mw":   "TXC-pro (MW)",
    "tsae_paper":   "T-SAE",
    "tsae_ours":    "T-SAE (ours, deprecated)",
    "topk_sae":     "TopK-SAE",
    "stacked_sae":  "Stacked-SAE",
    "tfa":          "TFA",
    "tfa_pos":      "TFA-pos",
    "mlc":          "MLC",
    "sae_arditi":   "SAE-arditi",
}

# Hparams whose presence on a row is meaningful (non-empty / non-default
# rendered explicitly; default elided to keep the table tight).
TRAIN_CFG_FIELDS = (
    "n_steps", "batch_size", "learning_rate", "optimizer", "warmup_steps",
    "precision", "train_window_size",
    "bricken_enabled", "bricken_resample_every", "bricken_min_fires",
    "bricken_n_check", "bricken_max_resample_fraction",
    "ema_auxk_alpha", "dead_threshold_tokens",
)


# ── Loaders ────────────────────────────────────────────────────────────


def read_jsonl(path: Path):
    """Yield JSON-decoded rows, skipping merge-conflict marker lines."""
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("<<<<<<<") or s.startswith(">>>>>>>") or s == "=======":
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def load_yaml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Hparams resolution ──────────────────────────────────────────────────


def resolve_arch_hparams(
    arch_name: str,
    component: str,
    archs_yaml: dict,
    arch_hparams_override: dict | None,
) -> dict:
    """Compute final hparams = base ⊕ per_component[component] ⊕ override."""
    spec = archs_yaml["archs"].get(arch_name, {})
    base = dict(spec.get("hparams", {}))
    per_comp = (spec.get("per_component_hparams") or {}).get(component, {})
    base.update(per_comp)
    if arch_hparams_override:
        base.update(arch_hparams_override)
    return base


def arch_class_path(arch_name: str, archs_yaml: dict) -> str:
    return archs_yaml["archs"].get(arch_name, {}).get("class_path", "?")


def arch_version(arch_name: str, archs_yaml: dict) -> str:
    return archs_yaml["archs"].get(arch_name, {}).get("arch_version", "?")


def fmt(v) -> str:
    """Compact rendering for table cells."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✓" if v else "—"
    if isinstance(v, float):
        if v == int(v):
            return f"{int(v)}"
        if abs(v) < 1e-3 or abs(v) >= 1e4:
            return f"{v:.2g}"
        return f"{v:.4g}"
    return str(v)


# ── Grouping ───────────────────────────────────────────────────────────


def _hashable(v):
    """Coerce values to a hashable form. Lists / dicts → JSON string."""
    if isinstance(v, (list, dict)):
        return json.dumps(v, sort_keys=True)
    return v


# Hparams that are typically swept across cells (collapsed to a list
# in the rendered row instead of fragmenting the table).
SWEPT_HPARAMS = {"k_pos", "k_win"}


def group_signature(
    arch: str, datasource: str, training_cfg: dict, final_hparams: dict
) -> tuple:
    """Stable tuple signature for grouping rows that share training spec.

    Sweep-axis hparams (``k_pos``, ``k_win``) are EXCLUDED from the
    signature — rows that differ only on these are collapsed into one
    table entry whose value column shows the swept set.

    Drops 'seed' (already off training_cfg in our schema) and any keys
    not in TRAIN_CFG_FIELDS so spurious extras don't fragment groups.
    Lists / dicts inside hparams are JSON-serialised to keep the signature
    hashable.
    """
    cfg_sig = tuple(sorted(
        (k, _hashable(training_cfg.get(k)))
        for k in TRAIN_CFG_FIELDS
    ))
    hp_sig = tuple(sorted(
        (k, _hashable(v))
        for k, v in final_hparams.items()
        if k not in SWEPT_HPARAMS
    ))
    return (arch, datasource, cfg_sig, hp_sig)


def _format_sweep_set(values) -> str:
    """Render a set of swept values as a compact string."""
    if not values:
        return "—"
    seen = sorted(set(values))
    if len(seen) == 1:
        return str(seen[0])
    return "{" + ", ".join(str(v) for v in seen) + "}"


# ── Markdown rendering ──────────────────────────────────────────────────


_REAL_LM_FIELDS = {
    "category", "subject_model", "layer", "hookpoint", "dataset",
    "n_seqs", "seq_len", "tokenizer_revision", "notes", "generator",
    "layers",
}


def render_subject_table(rows_in_section, datasources_yaml) -> list[str]:
    """Two tables: real-LM datasources + synthetic datasources.

    Real-LM table: subject_model + layer + dataset etc.
    Synthetic table: generator + ALL the generator-specific knobs
    (K_hidden, M_emissions, n_parents, rho, pi, p_A, p_B, sigma, …).
    """
    seen = {}
    for r in rows_in_section:
        ds = r["datasource"]
        if ds in seen:
            continue
        spec = datasources_yaml["datasources"].get(ds, {})
        seen[ds] = spec
    if not seen:
        return ["_(no datasources for this component)_", ""]

    md: list[str] = []

    # Split by category.
    real = {n: s for n, s in seen.items() if s.get("category") == "real_lm"}
    synth = {n: s for n, s in seen.items() if s.get("category") == "synthetic"}

    if real:
        md += [
            "**Datasources — real-LM**",
            "",
            "| datasource | subject_model | layer(s) | hookpoint | dataset | n_seqs | seq_len |",
            "|---|---|---|---|---|---:|---:|",
        ]
        for ds_name, spec in sorted(real.items()):
            layer = spec.get("layer", spec.get("layers"))
            md.append(
                f"| `{ds_name}` "
                f"| {fmt(spec.get('subject_model'))} "
                f"| {fmt(layer)} "
                f"| {fmt(spec.get('hookpoint'))} "
                f"| {fmt(spec.get('dataset'))} "
                f"| {fmt(spec.get('n_seqs'))} "
                f"| {fmt(spec.get('seq_len'))} |"
            )
        md.append("")

    if synth:
        # Collect union of generator-specific fields.
        synth_fields: list[str] = []
        seen_keys: set[str] = set()
        for spec in synth.values():
            for k in spec:
                if k not in _REAL_LM_FIELDS and k not in seen_keys:
                    synth_fields.append(k)
                    seen_keys.add(k)
        # Stable ordering: most common axes first.
        preferred_order = [
            "generator", "K_hidden", "M_emissions", "n_parents",
            "K_global", "K_local", "Kg", "Kl", "modulation_size",
            "n_features", "d_in", "seq_len",
            "pi", "rho", "rho_levels", "p_A", "p_B", "p_fire", "sigma",
            "magnitude_dist", "magnitude_mean", "magnitude_std",
        ]
        synth_fields_sorted = (
            [k for k in preferred_order if k in seen_keys]
            + [k for k in synth_fields if k not in preferred_order]
        )
        md += [
            "**Datasources — synthetic**",
            "",
            "| datasource | " + " | ".join(synth_fields_sorted) + " |",
            "|---|" + "|".join(["---"] * len(synth_fields_sorted)) + "|",
        ]
        for ds_name, spec in sorted(synth.items()):
            row = [f"`{ds_name}`"]
            for k in synth_fields_sorted:
                row.append(fmt(spec.get(k)))
            md.append("| " + " | ".join(row) + " |")
        md.append("")

    return md


def render_training_table(groups, archs_yaml) -> list[str]:
    """One row per (arch, datasource, training_cfg, hparams) signature."""
    md = [
        "**Architectures + training**",
        "",
        "| arch | arch_version | datasource | T | T_max | t_sample | k_pos | k_win | d_sae | n_steps | B | lr | optim | warmup | precision | win_train | Bricken | seeds | n_cells |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|---|---:|",
    ]

    def sort_key(item):
        sig, info = item
        arch = sig[0]
        idx = ARCH_ORDER.index(arch) if arch in ARCH_ORDER else 99
        return (idx, sig[1])  # arch order then datasource alpha

    for sig, info in sorted(groups.items(), key=sort_key):
        arch = sig[0]
        ds = sig[1]
        cfg = dict(sig[2])
        hp = dict(sig[3])
        seeds = sorted(info["seeds"])
        n_cells = info["n_cells"]
        k_pos_str = _format_sweep_set(info.get("k_pos_set", set()))
        k_win_str = _format_sweep_set(info.get("k_win_set", set()))
        # Bricken summary
        if cfg.get("bricken_enabled"):
            br = (
                f"on, ema_aux_α={fmt(cfg.get('ema_auxk_alpha'))}, "
                f"dead_thr={fmt(cfg.get('dead_threshold_tokens'))}, "
                f"resample_every={fmt(cfg.get('bricken_resample_every'))}, "
                f"min_fires={fmt(cfg.get('bricken_min_fires'))}, "
                f"n_check={fmt(cfg.get('bricken_n_check'))}, "
                f"max_frac={fmt(cfg.get('bricken_max_resample_fraction'))}"
            )
        else:
            br = "off"
        md.append(
            f"| `{ARCH_DISPLAY.get(arch, arch)}` "
            f"| {arch_version(arch, archs_yaml)} "
            f"| `{ds}` "
            f"| {fmt(hp.get('T'))} "
            f"| {fmt(hp.get('T_max'))} "
            f"| {fmt(hp.get('t_sample'))} "
            f"| {k_pos_str} "
            f"| {k_win_str} "
            f"| {fmt(hp.get('d_sae'))} "
            f"| {fmt(cfg.get('n_steps'))} "
            f"| {fmt(cfg.get('batch_size'))} "
            f"| {fmt(cfg.get('learning_rate'))} "
            f"| {fmt(cfg.get('optimizer'))} "
            f"| {fmt(cfg.get('warmup_steps'))} "
            f"| {fmt(cfg.get('precision'))} "
            f"| {fmt(cfg.get('train_window_size'))} "
            f"| {br} "
            f"| {{{','.join(str(s) for s in seeds)}}} "
            f"| {n_cells} |"
        )
    md.append("")
    return md


def render_extra_hparams(groups, archs_yaml) -> list[str]:
    """For arches with arch-specific niche params, render a footnote table.

    Captures fields like ``contrastive_alpha``, ``n_matryoshka``,
    ``decoder_unit_norm``, etc. that don't fit the main columns.
    Reads from each group's ``fixed_hparams`` (final resolved hparams,
    which include both base + per-component + override values).
    """
    extras_by_arch: dict[str, set[tuple]] = defaultdict(set)
    main_cols = {"T", "T_max", "t_sample", "k_pos", "k_win", "d_sae"}
    for sig, info in groups.items():
        arch = sig[0]
        hp = info.get("fixed_hparams", dict(sig[3]))
        for k, v in hp.items():
            if k in main_cols:
                continue
            if isinstance(v, (list, dict)):
                v = json.dumps(v, sort_keys=True)
            extras_by_arch[arch].add((k, v))

    if not any(extras_by_arch.values()):
        return []

    md = ["**Architecture niche hparams** (locked YAML defaults; not overridden per-cell)", ""]
    md.append("| arch | hparam | value |")
    md.append("|---|---|---|")
    for arch in sorted(extras_by_arch.keys(),
                       key=lambda a: ARCH_ORDER.index(a) if a in ARCH_ORDER else 99):
        for k, v in sorted(extras_by_arch[arch]):
            md.append(f"| `{ARCH_DISPLAY.get(arch, arch)}` | `{k}` | {fmt(v)} |")
    md.append("")
    return md


# ── Top-level orchestration ────────────────────────────────────────────


def main():
    archs_yaml = load_yaml(ARCHS_YAML)
    datasources_yaml = load_yaml(DATASOURCES_YAML)

    # Index manifest by train_key
    manifest_by_tk: dict[str, dict] = {}
    for m in read_jsonl(MANIFEST):
        manifest_by_tk[m["train_key"]] = m

    # Bucket leaderboard rows by group_id (first tuple element).
    rows_by_group: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(LEADERBOARD):
        if row.get("eval_cfg", {}).get("smoke", False):
            continue
        for group_id, _label, comps in COMPONENT_GROUPS:
            if row["component"] in comps:
                rows_by_group[group_id].append(row)
                break

    md = [
        "---",
        "title: Training appendix",
        "tags: [paper, appendix, training]",
        "auto_generated_by: scripts/render_training_appendix.py",
        "---",
        "",
        "# Training appendix",
        "",
        "_Auto-generated by `scripts/render_training_appendix.py`. Do not "
        "hand-edit. Numbers come from `results/leaderboard.jsonl`, "
        "`checkpoints/manifest.jsonl`, `configs/locked_archs.yaml`, and "
        "`configs/datasources.yaml`._",
        "",
        "Conventions:",
        "",
        "- One row per `(arch, datasource, training_cfg, final_hparams)` "
        "signature. Cells that share spec are grouped; `seeds` and "
        "`n_cells` columns expose the within-group spread.",
        "- `final_hparams` = locked YAML base ⊕ `per_component_hparams[c]` "
        "⊕ `arch_hparams_override` from the cell's `training_cfg`.",
        "- Smoke cells (`eval_cfg.smoke=True`) are excluded.",
        "- Empty cells render as `—`. `Bricken` collapses the seven "
        "`bricken_*` knobs into one column; `off` means `bricken_enabled=False`.",
        "- `B` = `batch_size`. `lr` = `learning_rate`. `win_train` = "
        "`train_window_size` (`None` = full sequence; integer = sample one "
        "T-window per sequence at training time).",
        "- `T` is the canonical window size (TXC-base). `T_max` + "
        "`t_sample` are TXC-pro / Stacked-SAE specific. `k_win` is the "
        "window-level top-k for window archs that expose it.",
        "",
        "## Datasources used",
        "",
        "All datasources referenced by paper-canonical cells, regardless of "
        "component. The component-by-component sections below reference "
        "these by name.",
        "",
    ]

    # Global datasources table (union)
    all_rows = [r for rows in rows_by_group.values() for r in rows]
    md += render_subject_table(all_rows, datasources_yaml)

    for group_id, label, comp_list in COMPONENT_GROUPS:
        rows = rows_by_group.get(group_id, [])
        md.append(f"## {label}")
        md.append("")
        md.append(
            f"_components: {', '.join(comp_list)} — "
            f"{len(rows)} non-smoke leaderboard cells, "
            f"{len(set(r['train_key'] for r in rows))} unique train_keys._"
        )
        md.append("")

        if not rows:
            md.append("_(no cells yet)_")
            md.append("")
            continue

        # Per-section subject table (filtered to datasources used here)
        md += render_subject_table(rows, datasources_yaml)

        # Group cells by training signature; collect swept hparam values
        # (k_pos, k_win) per group.
        groups: dict[tuple, dict] = {}
        for row in rows:
            tk = row["train_key"]
            man = manifest_by_tk.get(tk)
            if man is None:
                continue
            cfg = dict(man.get("training_cfg", {}))
            override = cfg.pop("arch_hparams_override", None)
            arch = man["arch"]
            ds = man["datasource"]
            comp = row["component"]
            hp = resolve_arch_hparams(arch, comp, archs_yaml, override)
            sig = group_signature(arch, ds, cfg, hp)
            slot = groups.setdefault(sig, {
                "seeds": set(), "n_cells": 0,
                "k_pos_set": set(), "k_win_set": set(),
                "fixed_hparams": hp,  # for niche-hparams render
            })
            slot["seeds"].add(int(man["seed"]))
            slot["n_cells"] += 1
            if "k_pos" in hp:
                slot["k_pos_set"].add(hp["k_pos"])
            if "k_win" in hp:
                slot["k_win_set"].add(hp["k_win"])

        md += render_training_table(groups, archs_yaml)
        md += render_extra_hparams(groups, archs_yaml)

    md.append("")
    md.append("---")
    md.append("")
    md.append(
        "_Generation: render with_ "
        "`.venv/bin/python scripts/render_training_appendix.py`. "
        "_The runner is idempotent; re-running with current state is safe._"
    )
    md.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md))
    print(f"Rendered {OUT_MD.relative_to(ROOT)}")
    print(f"  {sum(len(r) for r in rows_by_group.values())} cells across "
          f"{len(rows_by_group)} component groups")


if __name__ == "__main__":
    main()
