"""Reproduce + assert the frozen TSCALE dev/holdout split (CARD_SPLIT.md § 1).

Run:  .venv/bin/python -m experiments.explorations.tscale.make_split

Recomputes the family-stratified seeded draw from the committed P1
baseline rows and asserts it equals the frozen DEV list, then reprints
the § 2 reference numbers. Read-only; no compute, no writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

CT_TASKS = {"winogrande_correct_completion", "wsc_coreference"}
FAMILIES = ("ag_news", "amazon_reviews", "bias_in_bios", "europarl", "github_code")
QUOTA = {"ag_news": 1, "amazon_reviews": 2, "bias_in_bios": 3, "europarl": 1, "github_code": 1}
RNG_SEED = 20260727          # frozen; power rule: dev Δ16 (pre s42 k20) ≤ −0.010
BASELINE_ARCH = "txc_batchtopk_pre_btkonly"
SAE_ARCH = "batchtopk_sae_btkonly"

DEV_FROZEN = [
    "ag_news_world",
    "amazon_reviews_cat1",
    "amazon_reviews_cat2",
    "bias_in_bios_set2_prof11",
    "bias_in_bios_set3_prof21",
    "bias_in_bios_set3_prof26",
    "europarl_en",
    "github_code_Java",
]


def load_btk_rows() -> list[dict]:
    rows = []
    with (ROOT / "results" / "leaderboard.jsonl").open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ec = r.get("eval_cfg", {})
            if (
                r.get("experiment") == "probing"
                and ec.get("arm") == "btk-only"
                and not ec.get("smoke")
                and r.get("metrics", {}).get("n_tasks") == 38
                and r["training_cfg"]["n_steps"] > 0
            ):
                rows.append(r)
    return rows


def cell(rows: list[dict], arch: str, seed: int, T: int, k: int) -> dict:
    for r in rows:
        hp = r["training_cfg"].get("arch_hparams_override") or {}
        if (
            r["arch"] == arch
            and r["seed"] == seed
            and int(hp.get("T", 1)) == T
            and int(r["eval_cfg"]["k_feat"]) == k
        ):
            return r
    raise KeyError(f"missing baseline row {arch}/s{seed}/T{T}/k{k}")


def family_of(task: str) -> str:
    for fam in FAMILIES:
        if task.startswith(fam):
            return fam
    raise ValueError(f"task {task!r} matches no family")


def draw_dev(tasks36: list[str], rng_seed: int) -> list[str]:
    fams: dict[str, list[str]] = {f: [] for f in FAMILIES}
    for t in tasks36:
        fams[family_of(t)].append(t)
    for f in fams:
        fams[f].sort()
    rng = np.random.default_rng(rng_seed)
    dev: list[str] = []
    for fam in sorted(QUOTA):
        picks = rng.choice(len(fams[fam]), size=QUOTA[fam], replace=False)
        dev += [fams[fam][i] for i in sorted(picks)]
    return sorted(dev)


def dev_mean(row: dict, dev: list[str]) -> float:
    return float(np.mean([row["metrics"][f"auc__{t}"] for t in dev]))


def main() -> None:
    rows = load_btk_rows()
    any_row = cell(rows, BASELINE_ARCH, 42, 1, 20)
    tasks36 = sorted(
        t[len("auc__"):]
        for t in any_row["metrics"]
        if t.startswith("auc__") and t[len("auc__"):] not in CT_TASKS
    )
    assert len(tasks36) == 36, f"expected 36 CT-excl tasks, got {len(tasks36)}"

    dev = draw_dev(tasks36, RNG_SEED)
    assert dev == DEV_FROZEN, (
        "FROZEN SPLIT MISMATCH — the seeded draw no longer reproduces "
        f"CARD_SPLIT.md § 1:\n  drawn : {dev}\n  frozen: {DEV_FROZEN}"
    )

    pre = {T: cell(rows, BASELINE_ARCH, 42, T, 20) for T in (1, 4, 16)}
    d16 = dev_mean(pre[16], dev) - dev_mean(pre[1], dev)
    assert d16 <= -0.010, f"power rule violated: dev Δ16 = {d16:+.4f} > -0.010"

    holdout = [t for t in tasks36 if t not in dev]
    print(f"[make_split] OK — dev-8 reproduces (rng {RNG_SEED}, draw 1), "
          f"holdout n={len(holdout)}")
    print(f"[make_split] dev Δ16 (pre s42 k20) = {d16:+.4f} (power rule ≤ -0.010)")
    for T in (1, 4, 16):
        print(f"  pre s42 k20 dev mean @T{T}: {dev_mean(pre[T], dev):.4f}")
    sae = [dev_mean(cell(rows, SAE_ARCH, s, 1, 20), dev) for s in (1, 2, 42)]
    print(f"  SAE dev band (3 seeds): {np.mean(sae):.4f} ± {np.std(sae, ddof=1):.4f}")


if __name__ == "__main__":
    main()
