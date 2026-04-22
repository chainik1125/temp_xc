"""Phase 6 auto-interpretation of top features per arch.

For each arch we pick the top-N features on concat-sets A+B by
per-token activation variance; for each feature we gather the top-k
activating 20-token contexts (with the target position highlighted)
and ask Claude Haiku to write a one-line human-readable label. This
reproduces the paper's Figure 1/Figure 4 annotation convention.

Cost guardrail (from brief): 8 features × 4 archs = 32 Claude calls,
~$1 with claude-haiku-4-5.

Output under `experiments/phase6_qualitative_latents/results/autointerp/`:

    <arch>__labels.json   — list of {feature_idx, label, contexts, acts, variance}
    summary.md            — compact markdown table for the phase writeup
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

N_TOP_FEATURES = 8
N_CONTEXTS = 10
CONTEXT_WINDOW = 20  # tokens around the activating position

PROMPT = """\
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


def _load_concat_AB_tokens_and_z(arch: str):
    """Load combined concat-A+B token IDs, text, and z matrix."""
    zA = np.load(Z_DIR / "concat_A" / f"{arch}__z.npy").astype(np.float32)
    zB = np.load(Z_DIR / "concat_B" / f"{arch}__z.npy").astype(np.float32)
    prov_A = json.loads((Z_DIR / "concat_A" / "provenance.json").read_text())
    prov_B = json.loads((Z_DIR / "concat_B" / "provenance.json").read_text())
    ids = prov_A["token_ids"] + prov_B["token_ids"]
    z = np.concatenate([zA, zB], axis=0)
    assert len(ids) == z.shape[0], (
        f"token/ z mismatch {len(ids)} vs {z.shape[0]}"
    )
    return ids, z


def _pick_top_features(z: np.ndarray, n: int) -> np.ndarray:
    """Return indices of top-n features by activation variance."""
    var = z.var(axis=0)
    return np.argsort(-var)[:n]


def _gather_contexts(ids, z, feat_idx, tok, n_ctx=N_CONTEXTS, win=CONTEXT_WINDOW):
    """Pick top-n_ctx activating positions for feature `feat_idx`.

    Returns a list of dicts with {text, strength, position}.
    """
    acts = z[:, feat_idx]
    # Take top-n activating positions. Use np.argpartition for speed.
    n_ctx = min(n_ctx, (acts > 0).sum())
    if n_ctx == 0:
        return []
    top_idx = np.argsort(-acts)[:n_ctx]
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
        out.append({
            "position": int(pos), "strength": float(acts[pos]),
            "text": text,
        })
    return out


def _claude_label(client, model_id, prompt) -> str:
    """Call Claude, return the single-line label."""
    msg = client.messages.create(
        model=model_id,
        max_tokens=64,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def interp_arch(arch: str, client, model_id: str, tok) -> list[dict]:
    ids, z = _load_concat_AB_tokens_and_z(arch)
    top_feats = _pick_top_features(z, N_TOP_FEATURES)
    print(f"[{arch}] top-{N_TOP_FEATURES} features by variance: {top_feats.tolist()}")

    results = []
    for fi in top_feats.tolist():
        ctxs = _gather_contexts(ids, z, fi, tok)
        if not ctxs:
            label = "(dead feature, no activations)"
        else:
            excerpts = "\n\n".join(
                f"- (strength={c['strength']:.2f}) {c['text']}"
                for c in ctxs
            )
            prompt = PROMPT.format(n=len(ctxs), f=fi, excerpts=excerpts)
            try:
                label = _claude_label(client, model_id, prompt)
            except Exception as e:
                label = f"(claude error: {e.__class__.__name__})"
        print(f"  feat {fi}: {label}")
        results.append({
            "feature_idx": int(fi),
            "label": label,
            "variance": float(z[:, fi].var()),
            "n_contexts": len(ctxs),
            "contexts": ctxs[:3],  # keep only top-3 in summary JSON
        })
    return results


def render_summary(by_arch: dict[str, list[dict]]) -> str:
    lines = ["# Phase 6 autointerp summary",
             "", "Top-8 features per arch (ranked by variance on concat-sets A + B).",
             "", "## Labels", ""]
    lines.append("| arch | rank | feat_idx | variance | label |")
    lines.append("|---|---|---|---|---|")
    for arch, rows in by_arch.items():
        for r_i, r in enumerate(rows):
            lines.append(
                f"| {arch} | {r_i+1} | {r['feature_idx']} | "
                f"{r['variance']:.3f} | {r['label']} |"
            )
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", type=str, nargs="+",
                   default=["agentic_txc_02", "agentic_mlc_08",
                            "tsae_paper", "tsae_ours", "tfa_big"])
    p.add_argument("--model", type=str, default="claude-haiku-4-5")
    args = p.parse_args()

    api_key = None
    env_path = REPO / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not found in .env or environment")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b-it",
        token=open("/workspace/hf_cache/token").read().strip(),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_arch = {}
    for arch in args.archs:
        if not (Z_DIR / "concat_A" / f"{arch}__z.npy").exists():
            print(f"skip {arch}: no z cache")
            continue
        t0 = time.time()
        results = interp_arch(arch, client, args.model, tok)
        print(f"[{arch}] done in {time.time()-t0:.1f}s")
        (OUT_DIR / f"{arch}__labels.json").write_text(
            json.dumps(results, indent=2)
        )
        by_arch[arch] = results

    summary_md = render_summary(by_arch)
    (OUT_DIR / "summary.md").write_text(summary_md)
    print("wrote", OUT_DIR / "summary.md")


if __name__ == "__main__":
    main()
