"""
Render an autointerp comparison report contrasting SAE (T=1) vs TXC (T=5),
and the before/after of switching from local Gemma to Claude Haiku 4.5.

Reads:
  results/autointerp/sae/explanations.jsonl                (Haiku, current)
  results/autointerp/txc/explanations.jsonl                (Haiku, current)
  results/autointerp/sae/explanations.gemma_baseline.jsonl (Gemma, archived)
  results/autointerp/txc/explanations.gemma_baseline.jsonl (Gemma, archived)
  cached_activations/token_ids.npy                         (for span/position viz)

Writes:
  results/autointerp_report.md
  figures/autointerp/empty_window_fix.png
  figures/autointerp/safety_tag_distribution.png
  figures/autointerp/position_histogram.png
  figures/autointerp/explanation_length.png

Pushes to wandb under project temporal-crosscoders-safety, run name
"autointerp_report".
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

NLP_DIR = Path("/home/cs29824/andre/temp_xc/temporal_crosscoders/NLP")
SAFETY_DIR = Path("/home/cs29824/andre/temp_xc/safety_research")
sys.path.insert(0, str(NLP_DIR))
load_dotenv(SAFETY_DIR / ".env")

AUTOINTERP_DIR = SAFETY_DIR / "results" / "autointerp"
FIG_DIR = SAFETY_DIR / "figures" / "autointerp"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT = SAFETY_DIR / "results" / "autointerp_report.md"

ARMS = ["sae", "txc"]
T_BY_ARM = {"sae": 1, "txc": 5}
ARM_TITLE = {"sae": "SAE (T=1)", "txc": "TXC (T=5)"}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path)]


def empty_marker_count(records: list[dict]) -> tuple[int, int]:
    """Return (n_empty_marker_windows, n_total_windows)."""
    empty = 0
    total = 0
    for d in records:
        for t in d.get("top_texts", []):
            total += 1
            if ">>><<<" in t:
                empty += 1
    return empty, total


def safety_counts(records: list[dict]) -> Counter:
    c: Counter = Counter()
    for d in records:
        c[d.get("safety", "NONE")] += 1
    return c


def explanation_lengths(records: list[dict]) -> list[int]:
    return [len(d.get("explanation", "")) for d in records]


# ─────────────────────── feature position metadata ───────────────────────
# We compute, for each feature, the distribution of window-start positions
# across its top-K examples. The window-start is implicit in the snippet:
# the snippet is `pre>>>win<<<post`, where `pre` is the decoded token range
# [max(0, win_start-context) .. win_start). For SAE the original code used
# context=10, so we infer position from the prefix length when possible.
# Easier and exact: we don't have window_start saved in the JSONL — but the
# 32-token sequence length and BOS-as-first-token gives us a proxy: a window
# whose snippet starts with "<bos>" was certainly at position 0.
def position_zero_fraction(records: list[dict]) -> float:
    n_at_zero = 0
    n_total = 0
    for d in records:
        for t in d.get("top_texts", []):
            n_total += 1
            # Strip the leading marker if present, look for <bos> at start.
            stripped = t.lstrip()
            if stripped.startswith("<bos>") or stripped.startswith(">>><bos>"):
                n_at_zero += 1
    return n_at_zero / max(n_total, 1)


def explanation_topic_word_freq(records: list[dict], n: int = 25) -> list[tuple[str, int]]:
    stop = set(
        "the a an and or of to in for with on at as is are was were be by it that this "
        "these those from we you they its their feature features text describes describing "
        "common represent relates related related-to seemingly likely fires across spans "
        "appear appears tokens token first all most often refers refer one two each "
        "typically usually mostly content concept pattern".split()
    )
    counter: Counter = Counter()
    for d in records:
        words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", d.get("explanation", "").lower())
        counter.update(w for w in words if w not in stop)
    return counter.most_common(n)


def gallery_special_token_features(records: list[dict], k: int = 10) -> list[dict]:
    """Pick features whose top texts mention <bos>/<start_of_turn>/<end_of_turn>."""
    specials = ("<bos>", "<start_of_turn>", "<end_of_turn>", "<eos>", "<pad>")
    hits = []
    for d in records:
        n_with_special = sum(
            1 for t in d.get("top_texts", []) if any(s in t for s in specials)
        )
        if n_with_special >= max(2, len(d.get("top_texts", [])) // 4):
            hits.append((n_with_special, d))
    hits.sort(key=lambda x: -x[0])
    return [d for _, d in hits[:k]]


def gallery_top_features(records: list[dict], k: int = 12) -> list[dict]:
    """Top-K records as ranked in the JSONL (already sorted by mass)."""
    return records[:k]


# ───────────────────────────────── plots ────────────────────────────────
def plot_empty_window_fix(stats: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    arms = ARMS
    width = 0.35
    x = np.arange(len(arms))
    # Apples-to-apples: restrict to the 150-feature overlap with gemma.
    before_pct = [100 * stats[a]["gemma"]["empty"] / max(stats[a]["gemma"]["total"], 1) for a in arms]
    after_pct  = [100 * stats[a]["haiku_overlap"]["empty"] / max(stats[a]["haiku_overlap"]["total"], 1) for a in arms]
    ax.bar(x - width / 2, before_pct, width,
           label="Before — Gemma, skip_special_tokens=True (n=150)",
           color="#a33")
    ax.bar(x + width / 2, after_pct, width,
           label="After — Haiku, special tokens visible (same 150 ids)",
           color="#3a6")
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_TITLE[a] for a in arms])
    ax.set_ylabel("% of top-window snippets with empty `>>><<<` marker")
    ax.set_title("Empty-highlight windows (BOS/special-token features)")
    for xi, v in zip(x - width / 2, before_pct):
        ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    for xi, v in zip(x + width / 2, after_pct):
        ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(max(before_pct), max(after_pct), 1) * 1.25 + 5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_safety_distribution(stats: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    tags = ["NONE", "REFUSAL", "DECEPTION", "HARMFUL_CONTENT", "BIAS"]
    colors = ["#888", "#3a6", "#3aa", "#a33", "#a3a"]
    width = 0.4
    x = np.arange(len(tags))
    for offset, arm in zip([-width / 2, width / 2], ARMS):
        c = stats[arm]["haiku"]["safety_counts"]
        vals = [c.get(t, 0) for t in tags]
        ax.bar(x + offset, vals, width, label=ARM_TITLE[arm])
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=15)
    ax.set_ylabel("Count of top-150 features")
    ax.set_title("Safety-tag distribution (Haiku, current)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_position_zero(stats: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    arms = ARMS
    width = 0.35
    x = np.arange(len(arms))
    before = [100 * stats[a]["gemma"]["pos0_frac"] for a in arms]
    after = [100 * stats[a]["haiku_overlap"]["pos0_frac"] for a in arms]
    ax.bar(x - width / 2, before, width,
           label="Before — Gemma decode (n=150)", color="#a33")
    ax.bar(x + width / 2, after, width,
           label="After — Haiku decode (same 150 ids)", color="#3a6")
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_TITLE[a] for a in arms])
    ax.set_ylabel("% of top-window snippets that include `<bos>`")
    ax.set_title("Position-0 (BOS-anchored) feature visibility")
    for xi, v in zip(x - width / 2, before):
        ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    for xi, v in zip(x + width / 2, after):
        ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_ylim(0, max(max(before), max(after), 1) * 1.25 + 5)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_explanation_length(stats: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    arms = ARMS
    width = 0.35
    x = np.arange(len(arms))
    before = [stats[a]["gemma"]["mean_len"] for a in arms]
    after  = [stats[a]["haiku"]["mean_len"] for a in arms]
    ax.bar(x - width / 2, before, width, label="Gemma", color="#a33")
    ax.bar(x + width / 2, after, width, label="Haiku", color="#3a6")
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_TITLE[a] for a in arms])
    ax.set_ylabel("Mean explanation length (chars)")
    ax.set_title("Explanation verbosity")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ─────────────────────── markdown rendering helpers ─────────────────────
def md_truncate(s: str, n: int = 110) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("`", "\\`")


def gallery_table(records: list[dict], n_examples: int = 2) -> str:
    rows = ["| feat | safety | explanation | top windows |",
            "|------|--------|-------------|-------------|"]
    for d in records:
        ex_strs = []
        for t in d.get("top_texts", [])[:n_examples]:
            ex_strs.append(f"`{md_escape(md_truncate(t, 90))}`")
        rows.append(
            f"| {d['feat']} | {d.get('safety', 'NONE')} "
            f"| {md_escape(md_truncate(d.get('explanation', ''), 140))} "
            f"| {' <br> '.join(ex_strs)} |"
        )
    return "\n".join(rows)


def beforeafter_table(arm: str, gemma: dict, haiku: dict, ids: list[int]) -> str:
    g_by = {d["feat"]: d for d in gemma}
    h_by = {d["feat"]: d for d in haiku}
    rows = ["| feat | Gemma explanation | Haiku explanation | example window |",
            "|------|-------------------|-------------------|----------------|"]
    for fid in ids:
        g = g_by.get(fid, {})
        h = h_by.get(fid, {})
        ex = h.get("top_texts", g.get("top_texts", [""]))[0] if (h or g) else ""
        rows.append(
            f"| {fid} | "
            f"{md_escape(md_truncate(g.get('explanation', '—'), 110))} | "
            f"{md_escape(md_truncate(h.get('explanation', '—'), 110))} | "
            f"`{md_escape(md_truncate(ex, 90))}` |"
        )
    return "\n".join(rows)


# ────────────────────────────────── main ────────────────────────────────
def main() -> None:
    stats: dict = {}
    haiku_records: dict = {}
    gemma_records: dict = {}

    for arm in ARMS:
        haiku = load_jsonl(AUTOINTERP_DIR / arm / "explanations.jsonl")
        gemma = load_jsonl(AUTOINTERP_DIR / arm / "explanations.gemma_baseline.jsonl")
        haiku_records[arm] = haiku
        gemma_records[arm] = gemma

        # Apples-to-apples: restrict the haiku set to the same 150 feature
        # IDs the gemma baseline covered, for the headline before/after row.
        gemma_ids = {d["feat"] for d in gemma}
        haiku_overlap = [d for d in haiku if d["feat"] in gemma_ids]

        ge_empty, ge_total = empty_marker_count(gemma)
        ha_empty, ha_total = empty_marker_count(haiku)
        hov_empty, hov_total = empty_marker_count(haiku_overlap)

        stats[arm] = {
            "gemma": dict(
                empty=ge_empty, total=ge_total,
                safety_counts=dict(safety_counts(gemma)),
                pos0_frac=position_zero_fraction(gemma),
                mean_len=float(np.mean(explanation_lengths(gemma))) if gemma else 0.0,
                n_features=len(gemma),
            ),
            "haiku": dict(
                empty=ha_empty, total=ha_total,
                safety_counts=dict(safety_counts(haiku)),
                pos0_frac=position_zero_fraction(haiku),
                mean_len=float(np.mean(explanation_lengths(haiku))) if haiku else 0.0,
                n_features=len(haiku),
            ),
            "haiku_overlap": dict(
                empty=hov_empty, total=hov_total,
                pos0_frac=position_zero_fraction(haiku_overlap),
                mean_len=float(np.mean(explanation_lengths(haiku_overlap))) if haiku_overlap else 0.0,
                n_features=len(haiku_overlap),
            ),
        }

    # plots
    plot_empty_window_fix(stats, FIG_DIR / "empty_window_fix.png")
    plot_safety_distribution(stats, FIG_DIR / "safety_tag_distribution.png")
    plot_position_zero(stats, FIG_DIR / "position_histogram.png")
    plot_explanation_length(stats, FIG_DIR / "explanation_length.png")

    # markdown report
    md = []
    md.append("# Autointerp report — SAE vs TXC (Haiku rerun)\n")
    md.append(
        "Drop-in regeneration of the SAE (T=1) and TXC (T=5) feature explanations "
        "with two changes:\n"
        "1. `TextContext.get_window_text` now decodes with `skip_special_tokens=False`, "
        "so windows that fire on `<bos>` / `<start_of_turn>` / `<end_of_turn>` no "
        "longer present an empty `>>><<<` highlight to the explainer.\n"
        "2. The explainer is **claude-haiku-4-5-20251001** (async, semaphore=8) "
        "instead of the local `google/gemma-2-2b-it` fallback.\n"
        "\n"
        "TSAE was deliberately not rerun — same checkpoints / k / T / top-features as the "
        "original three-arm run, so this is apples-to-apples.\n"
    )

    md.append("## Headline numbers\n")
    md.append(
        "Apples-to-apples comparison: same 150 feature IDs explained by Gemma "
        "(before) vs Haiku with special tokens visible (after). Plus, the "
        "full Haiku dictionary count.\n"
    )
    md.append("| arm | full-dict n | empty-window % (Gemma 150 → Haiku 150) | "
              "BOS-visible % (Gemma 150 → Haiku 150) | mean explanation length (Gemma → Haiku, same 150) |")
    md.append("|-----|------------:|----------------------------------------:|"
              "-------------------------------------:|--------------------------------------------------:|")
    for arm in ARMS:
        s = stats[arm]
        ge, ha, hov = s["gemma"], s["haiku"], s["haiku_overlap"]
        empty_b = 100 * ge["empty"] / max(ge["total"], 1)
        empty_a = 100 * hov["empty"] / max(hov["total"], 1)
        bos_b = 100 * ge["pos0_frac"]
        bos_a = 100 * hov["pos0_frac"]
        md.append(
            f"| {ARM_TITLE[arm]} | {ha['n_features']} "
            f"| {empty_b:.1f}% → **{empty_a:.1f}%** "
            f"| {bos_b:.1f}% → **{bos_a:.1f}%** "
            f"| {ge['mean_len']:.0f} → {hov['mean_len']:.0f} chars |"
        )
    md.append("")

    md.append("![empty-window fix](../figures/autointerp/empty_window_fix.png)\n")
    md.append("![BOS visibility](../figures/autointerp/position_histogram.png)\n")

    md.append("## Safety-tag distribution (Haiku, current)\n")
    md.append(
        "Counts and within-arm percentages over the full active dictionary "
        "interpreted by Haiku.\n"
    )
    md.append("![safety distribution](../figures/autointerp/safety_tag_distribution.png)\n")
    md.append("| arm | NONE | REFUSAL | DECEPTION | HARMFUL_CONTENT | BIAS | total |")
    md.append("|-----|-----:|--------:|----------:|----------------:|-----:|------:|")
    for arm in ARMS:
        c = stats[arm]["haiku"]["safety_counts"]
        n = stats[arm]["haiku"]["n_features"] or 1
        def fmt(tag):
            v = c.get(tag, 0)
            return f"{v} ({100*v/n:.1f}%)"
        md.append(
            f"| {ARM_TITLE[arm]} | {fmt('NONE')} | {fmt('REFUSAL')} "
            f"| {fmt('DECEPTION')} | {fmt('HARMFUL_CONTENT')} "
            f"| {fmt('BIAS')} | {n} |"
        )
    md.append("")

    md.append("## Special-token features (now visible)\n")
    md.append(
        "Features whose top windows fire on a special token "
        "(`<bos>`, `<start_of_turn>`, `<end_of_turn>`, …). Before the fix, "
        "Gemma saw these as empty `>>><<<` markers and confidently labeled them "
        "with whatever surrounding content existed. The Gemma column shows the "
        "old explanation for the same `feat_id` for context.\n"
    )
    for arm in ARMS:
        haiku = haiku_records[arm]
        gemma = gemma_records[arm]
        special_feats = gallery_special_token_features(haiku, k=8)
        if not special_feats:
            md.append(f"### {ARM_TITLE[arm]}\n_No features in the top-150 fire predominantly on special tokens._\n")
            continue
        md.append(f"### {ARM_TITLE[arm]}\n")
        ids = [d["feat"] for d in special_feats]
        md.append(beforeafter_table(arm, gemma, haiku, ids))
        md.append("")

    md.append("## SAE vs TXC contrast — top-12 most-active features\n")
    md.append(
        "Features ranked by total activation mass. Two example windows per "
        "feature, truncated for readability.\n"
    )
    for arm in ARMS:
        md.append(f"### {ARM_TITLE[arm]} — top-12\n")
        md.append(gallery_table(gallery_top_features(haiku_records[arm], k=12)))
        md.append("")

    md.append("## Random sample (mid-dictionary)\n")
    md.append(
        "20 features drawn at random from each arm's full Haiku-interpreted "
        "set, to spot-check explanation quality outside the top-N tail.\n"
    )
    rng = np.random.default_rng(42)
    for arm in ARMS:
        recs = haiku_records[arm]
        if len(recs) <= 20:
            sample = recs
        else:
            idx = rng.choice(len(recs), size=20, replace=False)
            sample = [recs[i] for i in sorted(idx)]
        md.append(f"### {ARM_TITLE[arm]} — 20 random\n")
        md.append(gallery_table(sample))
        md.append("")

    md.append("## Topic vocabulary (Haiku explanations)\n")
    md.append(
        "Most-frequent content words in feature explanations — a coarse vocabulary "
        "diff between the two arms. T=5 TXC features tend toward span/temporal terms; "
        "T=1 SAE features tend toward single-token concept terms.\n"
    )
    md.append("| rank | SAE | count | TXC | count |")
    md.append("|-----:|-----|------:|-----|------:|")
    sae_top = explanation_topic_word_freq(haiku_records["sae"], n=25)
    txc_top = explanation_topic_word_freq(haiku_records["txc"], n=25)
    for i, (s, t) in enumerate(zip(sae_top, txc_top), 1):
        md.append(f"| {i} | {s[0]} | {s[1]} | {t[0]} | {t[1]} |")
    md.append("")

    md.append("## Explanation verbosity\n")
    md.append("![explanation length](../figures/autointerp/explanation_length.png)\n")

    REPORT.write_text("\n".join(md))
    print(f"  → {REPORT}")
    for fp in FIG_DIR.iterdir():
        print(f"  → {fp.relative_to(SAFETY_DIR)}")

    # wandb push
    if os.environ.get("WANDB_DISABLED") != "1":
        try:
            import wandb
            run = wandb.init(
                project="temporal-crosscoders-safety",
                name="autointerp_report",
                tags=["safety", "autointerp", "report"],
                config=dict(arms=ARMS, explainer="claude-haiku-4-5-20251001"),
                reinit=True,
            )
            for fp in FIG_DIR.iterdir():
                run.log({f"figures/{fp.stem}": wandb.Image(str(fp))})
            run.summary.update({
                f"{arm}/empty_pct_before": 100 * stats[arm]["gemma"]["empty"] / max(stats[arm]["gemma"]["total"], 1)
                for arm in ARMS
            })
            run.summary.update({
                f"{arm}/empty_pct_after": 100 * stats[arm]["haiku"]["empty"] / max(stats[arm]["haiku"]["total"], 1)
                for arm in ARMS
            })
            run.finish()
        except Exception as e:
            print(f"  wandb push skipped: {e}")


if __name__ == "__main__":
    main()
