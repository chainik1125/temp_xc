"""Phase 6.1 auto-interpretation of top features per (arch, seed, concat).

For each cell we pick the top-N=32 features by per-token activation variance
on the requested concat, gather top-k activating 20-token contexts per
feature, and ask Claude Haiku for a one-line label (Bills-et-al style).
Each label is then auto-classified as SEMANTIC or SYNTACTIC by two Haiku
judges with different prompts; disagreements are tallied.

In addition to the headline x/N SEMANTIC label-count, we emit:

    passage_coverage_count  — # distinct passages where at least one
                              top-N feature peaks (argmax of mean
                              activation per passage). Max = min(N, P).
    passage_coverage_entropy — Shannon entropy (nats) of the
                              peak-passage distribution over the N features.
    semantic_count_pdvar     — SECONDARY diagnostic: the x/N count when
                              features are ranked by passage-discriminative
                              variance instead of raw variance. NOT the
                              primary metric — it conditions on the
                              construct (see handover #2 sub-item 4).
    judge_disagreement_rate  — fraction of labels where the two Haiku
                              judges return different SEMANTIC/SYNTACTIC
                              verdicts.

Output under `results/autointerp/`:
    <arch>__seed{S}__<concat>__labels.json
    summary.md (aggregate table)

Cost guardrail: ~3 Haiku calls per feature (1 label + 2 judges) ×
32 features × n_archs × n_seeds × n_concats. User budget: $10 Haiku.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
Z_DIR = REPO / "experiments/phase6_qualitative_latents/z_cache"
OUT_DIR = REPO / "experiments/phase6_qualitative_latents/results/autointerp"

N_TOP_FEATURES = 32
N_CONTEXTS = 10
CONTEXT_WINDOW = 20  # tokens around the activating position
DEFAULT_CONCATS = ["A", "B", "random"]
DEFAULT_ARCHS = ["agentic_txc_02", "agentic_txc_02_batchtopk",
                 "agentic_txc_09_auxk", "agentic_txc_10_bare",
                 "agentic_txc_11_stack", "agentic_txc_12_bare_batchtopk",
                 "agentic_mlc_08", "tsae_paper", "tsae_ours", "tfa_big"]

LABEL_PROMPT = """\
You are a feature explanation assistant for sparse autoencoders.

Below are {n} text excerpts where SAE feature #{f} activated strongly.
Each excerpt shows a ~20-token window of text with the activating
position marked as «TOKEN». Activation strength is shown as (strength=...).

Your job: identify the common concept or pattern that triggers this
feature. Reply with ONLY a single short phrase (5-10 words) —
examples: "discussion of plant biology", "capitalized sentence starts",
"numerical values in physics problems". No explanation, just the phrase.

Excerpts:
{excerpts}
"""

# Two judge prompts with slightly different phrasings, for an independent
# disagreement signal. Both return a single word: SEMANTIC or SYNTACTIC.
JUDGE_PROMPTS = [
    """\
Given this SAE feature label: "{label}"
Classify as SEMANTIC or SYNTACTIC.
SEMANTIC = names a concept, topic, theme, entity, or domain
  (examples: "plant biology", "Animal Farm references",
   "archaic poetic English", "historical dates in biography")
SYNTACTIC = describes surface patterns — punctuation, word class,
  capitalisation, formatting, hyphens, quoted text
  (examples: "sentence-ending periods", "multiple-choice answer
   formatting", "hyphens between compound words")
Edge cases (pre-registered):
  - "historical dates"                     -> SEMANTIC (concept)
  - "quoted text from Animal Farm"         -> SEMANTIC (corpus ref)
  - "text in quotation marks"              -> SYNTACTIC (surface)
  - "MMLU answer-option formatting"        -> SYNTACTIC (surface)
  - "multi-layer/cross-domain terminology" -> SEMANTIC (concept)
Reply with exactly one word: SEMANTIC or SYNTACTIC.""",

    """\
SAE feature description: "{label}"

Question: does this description name a concept/topic/theme/entity
(a thing you could summarise with a noun phrase, like "physics problems"
or "Russian history")? Or does it describe a surface-level pattern
(punctuation, word class, formatting, quotes, hyphens)?

Answer in one word only:
  SEMANTIC  = concept / topic / theme / entity / domain
  SYNTACTIC = surface pattern (format / punctuation / word-class / quotes)

One word:""",
]


# ──────────────────────────────────────────────────── I/O


def _z_filename(arch: str, seed: int, concat_dir: Path) -> Path:
    """Return the z_cache file path for (arch, seed). Falls back to
    legacy `<arch>__z.npy` at seed=42 for pre-Phase-6.1 encodings."""
    p = concat_dir / f"{arch}__seed{seed}__z.npy"
    if p.exists():
        return p
    if seed == 42:
        legacy = concat_dir / f"{arch}__z.npy"
        if legacy.exists():
            return legacy
    raise FileNotFoundError(f"no z cache for {arch} seed={seed} in {concat_dir}")


def _load_concat(concat_key: str, arch: str, seed: int):
    """Load (ids, z, passages) for a single concat cell.

    `concat_key` is "A" / "B" / "random". concat_A and concat_B are
    combined at the caller level for legacy-compat "A+B" pipeline runs.

    Returns:
        ids: list of token IDs
        z: (n_tokens, d_sae) float32 latent
        passages: list of {label, start, end} — contiguous passage spans.
    """
    concat_name = f"concat_{concat_key}"
    concat_dir = Z_DIR / concat_name
    z_path = _z_filename(arch, seed, concat_dir)
    prov_path = concat_dir / "provenance.json"
    z = np.load(z_path).astype(np.float32)
    prov = json.loads(prov_path.read_text())
    ids = prov["token_ids"]
    passages = [{"label": p["label"], "start": p["start"], "end": p["end"]}
                for p in prov["provenance"]]
    return ids, z, passages


def _load_combined_AB(arch: str, seed: int):
    """Legacy-compat: concat_A + concat_B stacked with passage offsets."""
    ids_A, z_A, pA = _load_concat("A", arch, seed)
    ids_B, z_B, pB = _load_concat("B", arch, seed)
    n_A = len(ids_A)
    passages = pA + [{"label": p["label"], "start": p["start"] + n_A,
                      "end": p["end"] + n_A} for p in pB]
    ids = ids_A + ids_B
    z = np.concatenate([z_A, z_B], axis=0)
    return ids, z, passages


# ──────────────────────────────────────────────────── feature ranking + metrics


def _pick_top_features(z: np.ndarray, n: int) -> np.ndarray:
    """Rank by per-token activation variance, take top-n indices."""
    var = z.var(axis=0)
    return np.argsort(-var)[:n]


def _pick_top_features_pdvar(z: np.ndarray, passages: list, n: int) -> np.ndarray:
    """Rank features by passage-discriminative variance: for each
    feature, compute mean activation per passage, take variance across
    those per-passage means. Secondary diagnostic only (conditions on
    the construct — see handover #2 sub-item 4)."""
    # (n_passages, d_sae) matrix of per-feature passage means.
    means = np.stack([
        z[p["start"]:p["end"]].mean(axis=0)
        for p in passages
    ], axis=0)
    pdvar = means.var(axis=0)
    return np.argsort(-pdvar)[:n]


def _passage_coverage(z: np.ndarray, feat_indices: np.ndarray,
                      passages: list) -> dict:
    """For each top-N feature, compute its peak passage (argmax of
    mean-activation-per-passage). Return coverage count + entropy.

    Returns:
        peak_passage_labels: list[str] length n — label of peak passage
            per feature.
        count: int — number of distinct passages covered.
        entropy: float nats — Shannon entropy of the peak-passage
            distribution (uniform-max = log(P)).
    """
    means = np.stack([
        z[:, feat_indices][p["start"]:p["end"]].mean(axis=0)
        for p in passages
    ], axis=0)  # (n_passages, n_feats)
    peak_idx = means.argmax(axis=0)  # (n_feats,)
    labels = [passages[i]["label"] for i in peak_idx]
    P = len(passages)
    counts = np.bincount(peak_idx, minlength=P)
    count = int((counts > 0).sum())
    # Entropy in nats
    probs = counts / counts.sum() if counts.sum() else counts.astype(float)
    probs = probs[probs > 0]
    entropy = float(-(probs * np.log(probs)).sum()) if probs.size else 0.0
    return {"peak_passage_labels": labels, "count": count, "entropy": entropy}


# ──────────────────────────────────────────────────── context gather + Claude


def _gather_contexts(ids, tok, z_col: np.ndarray, n_ctx=N_CONTEXTS,
                     win=CONTEXT_WINDOW):
    """Top-n_ctx activating positions with 20-token text windows."""
    n_pos_active = int((z_col > 0).sum())
    n_ctx = min(n_ctx, n_pos_active)
    if n_ctx == 0:
        return []
    top_idx = np.argsort(-z_col)[:n_ctx]
    out = []
    n_tokens = len(ids)
    half = win // 2
    for pos in top_idx.tolist():
        lo = max(0, pos - half)
        hi = min(n_tokens, pos + half)
        before = tok.decode(ids[lo:pos]) if pos > lo else ""
        target = tok.decode([ids[pos]])
        after = tok.decode(ids[pos + 1:hi]) if hi > pos + 1 else ""
        text = f"{before}«{target}»{after}"
        out.append({"position": int(pos), "strength": float(z_col[pos]),
                    "text": text})
    return out


def _claude_call(client, model_id: str, prompt: str, max_tokens: int = 64,
                  max_retries: int = 4) -> str:
    """Temperature=0 for reproducibility across reruns. Without this,
    Haiku's default temperature=1.0 causes label drift and the x/N
    metric becomes noisy across runs even with identical inputs
    (observed ±5 labels on the same arch × concat).

    Retries on RateLimitError + transient API errors with exponential
    backoff. Rate-limit errors in the initial seed=42 A+B+random run
    affected ~2% of calls (8 of ~500); retries eliminate these.
    """
    import time as _time
    import anthropic as _anth
    last_err = None
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model_id, max_tokens=max_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except (_anth.RateLimitError, _anth.APIStatusError,
                _anth.APIConnectionError) as e:
            last_err = e
            # Exponential backoff: 2, 4, 8, 16s
            _time.sleep(2 ** (attempt + 1))
    # All retries exhausted; re-raise
    raise last_err


def _judge_label(client, model_id: str, label: str) -> dict:
    """Two Haiku judges with different prompts. Returns {judge_1, judge_2,
    agree, verdict}. Verdict = majority (== judge_1 on tie — there's
    never a tie with 2 judges, just agree/disagree)."""
    def _normalize(raw):
        # Extract first SEMANTIC / SYNTACTIC token
        r = raw.upper().strip()
        for w in ("SEMANTIC", "SYNTACTIC"):
            if w in r:
                return w
        return "UNKNOWN"
    v1 = _normalize(_claude_call(client, model_id,
                                 JUDGE_PROMPTS[0].format(label=label),
                                 max_tokens=8))
    v2 = _normalize(_claude_call(client, model_id,
                                 JUDGE_PROMPTS[1].format(label=label),
                                 max_tokens=8))
    agree = v1 == v2
    return {"judge_1": v1, "judge_2": v2, "agree": agree, "verdict": v1}


def interp_arch_concat(arch: str, seed: int, concat_key: str,
                       client, model_id: str, tok) -> dict:
    """Run the full pipeline on one (arch, seed, concat) cell.

    Returns a dict with per-feature records + aggregate metrics.
    """
    ids, z, passages = _load_concat(concat_key, arch, seed)

    # Primary ranking: top-N by raw per-token variance
    top_feats = _pick_top_features(z, N_TOP_FEATURES)
    # Secondary ranking: top-N by passage-discriminative variance
    top_feats_pdvar = _pick_top_features_pdvar(z, passages, N_TOP_FEATURES)

    print(f"[{arch} seed={seed} {concat_key}] top-{N_TOP_FEATURES} by var: "
          f"{top_feats[:8].tolist()}...")

    features = []
    for fi in top_feats.tolist():
        ctxs = _gather_contexts(ids, tok, z[:, fi])
        if not ctxs:
            label = "(dead feature, no activations)"
            judge = {"judge_1": "UNKNOWN", "judge_2": "UNKNOWN",
                     "agree": True, "verdict": "UNKNOWN"}
        else:
            excerpts = "\n\n".join(
                f"- (strength={c['strength']:.2f}) {c['text']}"
                for c in ctxs
            )
            prompt = LABEL_PROMPT.format(n=len(ctxs), f=fi, excerpts=excerpts)
            try:
                label = _claude_call(client, model_id, prompt)
            except Exception as e:
                label = f"(claude error: {e.__class__.__name__})"
                judge = {"judge_1": "UNKNOWN", "judge_2": "UNKNOWN",
                         "agree": True, "verdict": "UNKNOWN"}
            else:
                try:
                    judge = _judge_label(client, model_id, label)
                except Exception as e:
                    judge = {"judge_1": "UNKNOWN", "judge_2": "UNKNOWN",
                             "agree": True, "verdict": "UNKNOWN",
                             "error": str(e)[:80]}
        features.append({
            "feature_idx": int(fi), "label": label,
            "variance": float(z[:, fi].var()),
            "n_contexts": len(ctxs),
            "contexts": ctxs[:3],
            "judge": judge,
        })

    # Coverage diagnostic on top-by-variance features
    cov = _passage_coverage(z, top_feats, passages)
    for i, lbl in enumerate(cov["peak_passage_labels"]):
        features[i]["peak_passage"] = lbl

    # Metrics
    sem_count = sum(1 for f in features if f["judge"]["verdict"] == "SEMANTIC")
    n_disagree = sum(1 for f in features if not f["judge"]["agree"])
    total = max(len(features), 1)

    # PDvar secondary: just re-count using Haiku labels... but we don't
    # want to pay for labelling a second set of 32 features; instead, if
    # any of the pdvar top-32 overlap with var top-32, reuse; otherwise
    # mark as unlabeled. Cheaper and sufficient for a diagnostic.
    var_set = set(top_feats.tolist())
    pdvar_labels = {f["feature_idx"]: f for f in features
                    if f["feature_idx"] in var_set}
    pdvar_overlap = int(sum(1 for i in top_feats_pdvar.tolist() if i in var_set))
    pdvar_sem = sum(
        1 for i in top_feats_pdvar.tolist()
        if i in pdvar_labels and pdvar_labels[i]["judge"]["verdict"] == "SEMANTIC"
    )

    metrics = {
        "N": N_TOP_FEATURES,
        "semantic_count": sem_count,
        "semantic_fraction": sem_count / total,
        "passage_coverage_count": cov["count"],
        "passage_coverage_entropy": cov["entropy"],
        "n_passages": len(passages),
        "judge_disagreement_rate": n_disagree / total,
        "semantic_count_pdvar_overlap": pdvar_sem,
        "pdvar_var_overlap_count": pdvar_overlap,
    }
    return {"arch": arch, "seed": seed, "concat": concat_key,
            "features": features,
            "passages": [p["label"] for p in passages],
            "top_feat_indices_var": top_feats.tolist(),
            "top_feat_indices_pdvar": top_feats_pdvar.tolist(),
            "metrics": metrics}


# ──────────────────────────────────────────────────── driver


def render_summary(cells: list[dict]) -> str:
    lines = [
        "# Phase 6.1 autointerp summary (upgraded metric)",
        "",
        f"N_TOP_FEATURES = {N_TOP_FEATURES}  (from 8).  Multi-judge = 2 Haiku prompts.",
        "",
        "## Per-cell metrics",
        "",
        "| arch | seed | concat | /N sem | cov k/P | cov entropy | judge disagree | pdvar ∩var |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        m = c["metrics"]
        lines.append(
            f"| {c['arch']} | {c['seed']} | {c['concat']} | "
            f"{m['semantic_count']}/{m['N']} | "
            f"{m['passage_coverage_count']}/{m['n_passages']} | "
            f"{m['passage_coverage_entropy']:.2f} | "
            f"{m['judge_disagreement_rate']:.2f} | "
            f"{m['pdvar_var_overlap_count']} |"
        )
    # Aggregate across seeds per (arch, concat)
    from collections import defaultdict
    agg: dict = defaultdict(list)
    for c in cells:
        agg[(c["arch"], c["concat"])].append(c["metrics"])
    lines += ["", "## Means ± stderr across seeds (per arch × concat)",
              "",
              "| arch | concat | n_seeds | mean /N sem | mean cov k/P | mean cov entropy |",
              "|---|---|---|---|---|---|"]
    for (arch, concat), ms in sorted(agg.items()):
        n = len(ms)
        sem = np.array([m["semantic_count"] for m in ms], dtype=float)
        cov = np.array([m["passage_coverage_count"] for m in ms], dtype=float)
        ent = np.array([m["passage_coverage_entropy"] for m in ms], dtype=float)
        def se(a): return a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
        lines.append(
            f"| {arch} | {concat} | {n} | "
            f"{sem.mean():.1f} ± {se(sem):.2f} | "
            f"{cov.mean():.1f} ± {se(cov):.2f} | "
            f"{ent.mean():.2f} ± {se(ent):.2f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", type=str, nargs="+", default=DEFAULT_ARCHS)
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="seeds to evaluate (per arch; missing ckpts skipped)")
    p.add_argument("--concats", type=str, nargs="+", default=DEFAULT_CONCATS,
                   help="subset of {A, B, random}")
    p.add_argument("--model", type=str, default="claude-haiku-4-5")
    args = p.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        env_path = REPO / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not found in env or .env")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b-it",
        token=open("/workspace/hf_cache/token").read().strip(),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = []
    for arch in args.archs:
        for seed in args.seeds:
            for concat_key in args.concats:
                concat_dir = Z_DIR / f"concat_{concat_key}"
                if not concat_dir.exists():
                    print(f"[skip] {arch} seed={seed} {concat_key}: no concat dir")
                    continue
                try:
                    _ = _z_filename(arch, seed, concat_dir)
                except FileNotFoundError:
                    print(f"[skip] {arch} seed={seed} {concat_key}: no z cache")
                    continue
                t0 = time.time()
                try:
                    cell = interp_arch_concat(
                        arch, seed, concat_key, client, args.model, tok,
                    )
                except Exception as e:
                    print(f"[err]  {arch} seed={seed} {concat_key}: "
                          f"{e.__class__.__name__}: {str(e)[:200]}")
                    continue
                print(f"[done] {arch} seed={seed} {concat_key}  "
                      f"sem={cell['metrics']['semantic_count']}/{N_TOP_FEATURES}  "
                      f"cov={cell['metrics']['passage_coverage_count']}/"
                      f"{cell['metrics']['n_passages']}  "
                      f"disagree={cell['metrics']['judge_disagreement_rate']:.2f}  "
                      f"({time.time()-t0:.1f}s)")
                out_name = f"{arch}__seed{seed}__concat{concat_key}__labels.json"
                (OUT_DIR / out_name).write_text(json.dumps(cell, indent=2))
                cells.append(cell)

    if cells:
        (OUT_DIR / "summary.md").write_text(render_summary(cells))
        print(f"wrote {OUT_DIR}/summary.md with {len(cells)} cells")


if __name__ == "__main__":
    main()
