"""Track B candidate #2 — multi-token concept probing at low k_feat.

Builds a controlled probing dataset where each "concept" is a multi-token
phrase that requires multiple consecutive tokens to identify. For each
arch, computes per-feature lift and AUC at fixed k_feat.

Target: at k_feat=1 (single-feature classifier) and k_feat=5 (small
linear combo), how well do per-arch features detect the multi-token
phrase?

Hypothesis: TXC at T ≥ phrase_length encodes the entire phrase in one
feature; per-token archs need multiple features.

Run:
  TQDM_DISABLE=1 .venv/bin/python -m \\
    experiments.phase7_unification.case_studies.steering.probe_multitoken_concepts

Output:
  results/case_studies/diagnostics/multitoken_probing.json
  results/case_studies/plots/phase7_multitoken_auc_vs_kfeat.png
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import OUT_DIR, banner, MLC_LAYERS
from experiments.phase7_unification.case_studies._arch_utils import (
    load_phase7_model_safe as _load_phase7_model,
    encode_per_position, window_T, MLC_CLASSES,
)
from experiments.phase7_unification.case_studies._paths import (
    CASE_STUDIES_DIR, SUBJECT_MODEL, ANCHOR_LAYER, DEFAULT_D_IN,
)


# ─────────────────────────────────────────── Multi-token target phrases
# Pick phrases that require ≥ 3 consecutive tokens to identify and span
# distinct semantic content (so they're unlikely to be detected by a
# single-token feature alone).
MULTITOKEN_PHRASES = [
    ("the United States",       "named entity, country"),
    ("in the morning",          "time-of-day adverbial"),
    ("at the same time",        "temporal coordination"),
    ("on the other hand",       "contrastive discourse marker"),
    ("as well as",              "list-coordination phrase"),
    ("a little bit",            "quantifier phrase"),
    ("more than",               "comparative phrase (2-token)"),
    ("of course",               "discourse marker"),
    ("in order to",             "purpose clause"),
    ("the rest of",             "partitive phrase"),
    ("a couple of",             "vague quantifier"),
    ("from time to time",       "frequency adverbial"),
    ("for the first time",      "ordinal-temporal"),
    ("at this point",           "discourse-deictic"),
    ("the only way",            "uniqueness phrase"),
]


# ─────────────────────────────────────────── Negative-sample text source
# Use the 30 concept × 5 example sentences from concepts.py PLUS some
# extra public-domain text — gives ~150-200 sentences for negatives.
def _load_text_corpus(n_negatives: int = 200) -> list[str]:
    """Returns a list of sentences NOT containing any of the target phrases."""
    from experiments.phase7_unification.case_studies.steering.concepts import CONCEPTS
    pool = []
    for c in CONCEPTS:
        pool.extend(c["examples"])
    # Filter out anything that has any target phrase
    target_set = [p[0].lower() for p in MULTITOKEN_PHRASES]
    filtered = [s for s in pool if not any(t in s.lower() for t in target_set)]
    return filtered[:n_negatives]


def _build_positive_examples(phrase: str, n: int = 12) -> list[str]:
    """Generate n short sentences containing the phrase. Uses templates."""
    templates = [
        "He went to {} every weekend.",
        "We met at {} for the first time.",
        "She spoke to him {} about the issue.",
        "They visited {} during the spring.",
        "I always drink coffee {} before work.",
        "The book mentions {} in chapter three.",
        "The team agreed {} on the strategy.",
        "He walked away {} without saying goodbye.",
        "She arrived {} carrying a heavy suitcase.",
        "They argued {} about the budget cuts.",
        "He looked {} at his watch during the meeting.",
        "She noticed {} that the door was open.",
    ]
    return [t.format(phrase) for t in templates[:n]]


def encode_sentences(arch_id: str, sentences: list[str], device, batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Encode a list of sentences through the subject model + arch's encoder.

    Returns:
      acts: (N, S, d_sae) per-position arch features at L12.
      attn: (N, S) content-token mask.
    """
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed42.json"
    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed42.pt"
    meta = json.loads(log_path.read_text())
    src_class = meta["src_class"]
    is_mlc = src_class in MLC_CLASSES

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    subject = AutoModelForCausalLM.from_pretrained(
        SUBJECT_MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    subject.eval()
    for p in subject.parameters():
        p.requires_grad_(False)

    n = len(sentences)
    max_length = 32
    layers = list(MLC_LAYERS) if is_mlc else [ANCHOR_LAYER]
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def hook_factory(li):
        def hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[li] = h.detach().cpu()
        return hook
    for li in layers:
        handles.append(subject.model.layers[li].register_forward_hook(hook_factory(li)))

    if is_mlc:
        residuals = np.zeros((n, max_length, len(MLC_LAYERS), DEFAULT_D_IN), dtype=np.float16)
    else:
        residuals = np.zeros((n, max_length, DEFAULT_D_IN), dtype=np.float16)
    attn = np.zeros((n, max_length), dtype=np.int8)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = sentences[start:end]
        enc = tokenizer(chunk, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
        captured.clear()
        with torch.no_grad():
            subject(enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
        if is_mlc:
            for idx, li in enumerate(MLC_LAYERS):
                h = captured[li]
                if h.shape[-1] != DEFAULT_D_IN:
                    h = h[..., :DEFAULT_D_IN]
                residuals[start:end, :, idx, :] = h.to(torch.float16).numpy()
        else:
            h = captured[ANCHOR_LAYER]
            if h.shape[-1] != DEFAULT_D_IN:
                h = h[..., :DEFAULT_D_IN]
            residuals[start:end] = h.to(torch.float16).numpy()
        attn[start:end] = enc["attention_mask"].to(torch.int8).numpy()
    for handle in handles:
        handle.remove()
    del subject
    torch.cuda.empty_cache()
    gc.collect()

    sae, _ = _load_phase7_model(meta, ckpt_path, device)
    T = window_T(sae, src_class, meta)

    # Per-position encode
    x = torch.from_numpy(residuals).float().to(device)
    z = encode_per_position(sae, src_class, x, T=T)  # (N, S, d_sae)
    acts = z.detach().cpu().numpy()
    del x, z, sae
    torch.cuda.empty_cache()
    gc.collect()
    return acts, attn, T


def per_phrase_auc(acts_pos: np.ndarray, acts_neg: np.ndarray, attn_pos, attn_neg, T: int, k_feat: int) -> float:
    """Per-feature AUC (no LR fitting).

    For each candidate feature, score each example as the *max* activation
    over content positions (TXC's natural unit-of-evidence: a feature
    fires once at the n-gram window). Pick top-k features by mean
    pos-neg gap, then aggregate AUC by either single-best (k=1) or
    sum-of-top-k (k≥5).

    This is a direct, low-cost measure that doesn't require any
    parameter fit and rewards features that fire ONCE at the right
    position.
    """
    def _example_max(acts, attn):
        m = attn.astype(np.float32)
        if T > 1:
            valid = np.zeros(acts.shape[1], dtype=np.float32)
            valid[T-1:] = 1.0
            m = m * valid[None, :]
        # Mask non-content positions to a very negative value so MAX ignores them.
        masked = np.where(m[:, :, None] > 0, acts, -1e9)
        return masked.max(axis=1)  # (N, d_sae)

    feat_pos = _example_max(acts_pos, attn_pos)
    feat_neg = _example_max(acts_neg, attn_neg)
    diff = feat_pos.mean(axis=0) - feat_neg.mean(axis=0)
    if k_feat == 1:
        best = int(np.argmax(np.abs(diff)))
        scores = np.concatenate([feat_pos[:, best], feat_neg[:, best]]) * float(np.sign(diff[best]))
    else:
        top_k_idx = np.argsort(-np.abs(diff))[:k_feat]
        signs = np.sign(diff[top_k_idx])
        # Sum of signed activations across top-k
        scores_pos = (feat_pos[:, top_k_idx] * signs).sum(axis=1)
        scores_neg = (feat_neg[:, top_k_idx] * signs).sum(axis=1)
        scores = np.concatenate([scores_pos, scores_neg])
    y = np.concatenate([np.ones(len(feat_pos)), np.zeros(len(feat_neg))])
    return float(roc_auc_score(y, scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=[
        "topk_sae", "tsae_paper_k20", "tsae_paper_k500",
        "agentic_txc_02", "phase5b_subseq_h8",
        "phase57_partB_h8_bare_multidistance_t5",
        "mlc_contrastive_alpha100_batchtopk",
    ])
    ap.add_argument("--k-feats", nargs="+", type=int, default=[1, 5, 20])
    ap.add_argument("--out-dir", default=str(CASE_STUDIES_DIR / "diagnostics"))
    args = ap.parse_args()
    banner(__file__)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    negatives = _load_text_corpus(n_negatives=180)
    all_pos: list[tuple[str, list[str]]] = [
        (phrase, _build_positive_examples(phrase)) for phrase, _ in MULTITOKEN_PHRASES
    ]
    print(f"  {len(MULTITOKEN_PHRASES)} target phrases × 12 templates = "
          f"{12 * len(MULTITOKEN_PHRASES)} positives; {len(negatives)} negatives")

    results = {}
    for arch_id in args.archs:
        ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed42.pt"
        if not ckpt_path.exists():
            print(f"  [skip] {arch_id}: no ckpt at {ckpt_path}")
            continue
        print(f"\n=== {arch_id} ===")
        # Encode positives + negatives in one pass to avoid recomputing residuals
        all_sent = []
        origins = []  # (-1, phrase_idx_or_-1)
        for pi, (phrase, pos) in enumerate(all_pos):
            for s in pos:
                all_sent.append(s)
                origins.append(pi)
        for s in negatives:
            all_sent.append(s)
            origins.append(-1)
        t0 = time.time()
        acts, attn, T = encode_sentences(arch_id, all_sent, device)
        print(f"    encoded {len(all_sent)} sentences in {time.time() - t0:.1f}s; T={T}")

        per_phrase = {}
        for pi, (phrase, _) in enumerate(all_pos):
            pos_idx = [i for i, o in enumerate(origins) if o == pi]
            neg_idx = [i for i, o in enumerate(origins) if o == -1]
            acts_pos = acts[pos_idx]
            acts_neg = acts[neg_idx]
            attn_pos = attn[pos_idx]
            attn_neg = attn[neg_idx]
            per_k = {}
            for k in args.k_feats:
                auc = per_phrase_auc(acts_pos, acts_neg, attn_pos, attn_neg, T, k)
                per_k[f"k_feat_{k}"] = auc
            per_phrase[phrase] = per_k
        del acts, attn
        gc.collect()

        # Aggregate per-arch mean across phrases
        agg = {}
        for k in args.k_feats:
            arr = np.array([per_phrase[p][f"k_feat_{k}"] for p in per_phrase])
            agg[f"k_feat_{k}_mean"] = float(arr.mean())
            agg[f"k_feat_{k}_min"] = float(arr.min())
            agg[f"k_feat_{k}_p25"] = float(np.percentile(arr, 25))
        results[arch_id] = {
            "T": T,
            "per_phrase": per_phrase,
            "agg": agg,
        }
        print(f"    AUC@k=1 mean={agg['k_feat_1_mean']:.3f}  AUC@k=5 mean={agg['k_feat_5_mean']:.3f}  "
              f"AUC@k=20 mean={agg['k_feat_20_mean']:.3f}")

    out_path = out_dir / "multitoken_probing.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  wrote {out_path}")

    print("\n## Summary table")
    print(f"  {'arch':45s}  {'T':>3s}  " + "  ".join(f"{'AUC@k=' + str(k):>12s}" for k in args.k_feats))
    for arch_id, r in results.items():
        print(f"  {arch_id:45s}  {r['T']:>3d}  " +
              "  ".join(f"{r['agg'][f'k_feat_{k}_mean']:>12.3f}" for k in args.k_feats))


if __name__ == "__main__":
    main()
