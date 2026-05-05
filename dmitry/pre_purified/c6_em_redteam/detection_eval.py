"""detection_eval.py — per-feature classifier AUC + multi-feature LR
accuracy on the BASE-vs-LoRA-organism contrast.

For each canonical c6 cell:

  1. Load 500 cfierro probe prompts (already used in Wang stage 1).
  2. Run them through both the BASE Qwen subject model (no adapter) AND
     the BASE+LoRA-organism model. Capture residual-stream activations
     at layer L for the LAST PROMPT TOKEN (canonical pre-generation
     representation).
  3. Encode via the trained SAE / TXC dictionary → z_base[500, d_sae],
     z_lora[500, d_sae].
  4. Stack X = [z_base; z_lora], y = [0]·500 + [1]·500
     (0 = BASE-aligned, 1 = LoRA-organism = misaligned at the
     model-state level).
  5. (a) Per-feature AUC: roc_auc_score(y, X[:, f]) for each f.
     Report the headline-finalist's AUC + its rank.
  6. (b) Multi-feature LogisticRegression on full d_sae → 5-fold CV
     accuracy + AUC.
  7. Persist {finalist_id, finalist_auc, finalist_rank, top10_AUC,
     multi_acc, multi_auc} to JSON.

Per-cell estimated runtime: ~5 min (model load shared between
seeds-of-same-arch + organism).

Usage:
    python detection_eval.py [--cells <tk> ...]
    python detection_eval.py --skip-existing  # resume after interrupt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────
# Resolution order:
#   1. C6_WORKTREE env var (explicit override)
#   2. Walk up from this file's path until a `purified/` sibling exists
#      (works from any clone of the repo)
#   3. Pod fallback: /workspace/temp_xc-c6-extend
#   4. Local fallback: /tmp/c6_redteam_wt
def _find_worktree() -> Path:
    cand = os.environ.get("C6_WORKTREE")
    if cand and (Path(cand) / "purified" / "src").exists():
        return Path(cand)
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "purified" / "src").exists():
            return p
    for p in ("/workspace/temp_xc-c6-extend", "/tmp/c6_redteam_wt"):
        if (Path(p) / "purified" / "src").exists():
            return Path(p)
    raise SystemExit(
        "No worktree with purified/ found. "
        "Either run from inside the repo, or set C6_WORKTREE."
    )

WORKTREE = _find_worktree()
PURIFIED_SRC = WORKTREE / "purified" / "src"
sys.path.insert(0, str(PURIFIED_SRC))
sys.path.insert(0, str(WORKTREE / "purified"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("c6.detect")

OUT_ROOT = Path(os.environ.get(
    "C6_OUT_ROOT",
    "/workspace/c6_redteam/detection"
        if Path("/workspace").exists() else "/tmp/c6_redteam/detection",
))


@dataclass(frozen=True)
class Cell:
    arch: str; organism: str; seed: int
    train_key: str; datasource: str

CELLS: list[Cell] = [
    Cell("sae_arditi", "14B-finance", 1,  "5e4e188045d5d3c8",
         "qwen_2_5_14b_instruct_finance_l24_resid_post"),
    Cell("sae_arditi", "14B-finance", 42, "9778d10381696f58",
         "qwen_2_5_14b_instruct_finance_l24_resid_post"),
    Cell("sae_arditi", "7B-medical", 1,   "9b011dfeea88f8af",
         "qwen_2_5_7b_instruct_medical_l15_resid_post"),
    Cell("sae_arditi", "7B-medical", 42,  "c0da3ed8794554a1",
         "qwen_2_5_7b_instruct_medical_l15_resid_post"),
    Cell("txc_base",   "14B-finance", 1,  "672dbf61896f7843",
         "qwen_2_5_14b_instruct_finance_l24_resid_post"),
    Cell("txc_base",   "14B-finance", 42, "754166d1711923c1",
         "qwen_2_5_14b_instruct_finance_l24_resid_post"),
    Cell("txc_base",   "7B-medical", 1,   "2016074933c41e7f",
         "qwen_2_5_7b_instruct_medical_l15_resid_post"),
    Cell("txc_base",   "7B-medical", 42,  "88a4ddf6819d8057",
         "qwen_2_5_7b_instruct_medical_l15_resid_post"),
]

PROBE_REPO_BY_DATASOURCE = {
    "qwen_2_5_14b_instruct_finance_l24_resid_post":
        "cfierro/personality-qs-risky-financial-advice",
    "qwen_2_5_7b_instruct_medical_l15_resid_post":
        "cfierro/personality-qs-bad-medical-advice",
}
N_PROBE_PROMPTS = 500


def headline_feature_id(train_key: str) -> int:
    p = WORKTREE / "purified" / "results" / "runs" / f"c6_{train_key}" / "wang_full.json"
    d = json.loads(p.read_text())
    h = d.get("headline") or d.get("stage4", {}).get("peak") or {}
    return int(h["feature_id"])


def load_probe_prompts(repo_id: str, n: int) -> list[str]:
    """Mirror em._load_probe_prompts but stand-alone."""
    import random
    from datasets import load_dataset
    ds = load_dataset(repo_id, split="train")
    rng = random.Random(1234)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    out: list[str] = []
    for i in indices:
        if len(out) >= n:
            break
        try:
            messages = ds[i]["messages"]
            user_msg = next((m["content"] for m in messages
                             if m.get("role") == "user"), None)
            if user_msg and len(user_msg) >= 20:
                out.append(user_msg)
        except Exception:
            continue
    return out


def collect_full_seq_acts(model, tokenizer, prompts, layer, device):
    """Run prompts through `model`, return list of per-prompt (n_tok, d_in)
    resid_post tensors. Used downstream by `encode_per_prompt_z` which
    pools in latent space (matching Wang stage 1 convention).

    Each tensor stays on CPU as float32 to keep GPU mem low for the
    next stage.
    """
    import torch

    captured = {"act": None}

    def hook(module, _in, out):
        x = out[0] if isinstance(out, tuple) else out
        captured["act"] = x.detach()

    target = None
    for path in (("model", "layers"), ("transformer", "h")):
        try:
            obj = model
            for p in path:
                obj = getattr(obj, p)
            target = obj[layer]
            break
        except AttributeError:
            continue
    if target is None:
        raise RuntimeError("could not find decoder layer module")

    h = target.register_forward_hook(hook)
    try:
        outs = []
        for prompt in prompts:
            try:
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                rendered = prompt
            enc = tokenizer(rendered, return_tensors="pt",
                            add_special_tokens=False).to(device)
            with torch.no_grad():
                model(**enc, use_cache=False)
            # Full sequence activations.
            act = captured["act"][0, :, :].to(dtype=torch.float32).cpu()
            outs.append(act)
        return outs   # list of (n_tok, d_in) CPU tensors
    finally:
        h.remove()


def encode_per_prompt_z(arch_module, full_seq_acts, arch_T):
    """Mean-pool latents over each prompt's token positions.

    For SAE (T=1): encode every token, mean over positions → (d_sae,).
    For TXC (T>1): slide T-window over positions, encode each window
    (collapses to (1, d_sae)), mean over windows → (d_sae,).
    Matches `temp_bench.eval.probing.s_tail_probe`'s convention and
    Wang stage 1's pooling.
    """
    import torch
    import numpy as np

    device = next(arch_module.parameters()).device
    dtype = next(arch_module.parameters()).dtype

    out = []
    for acts in full_seq_acts:
        x = acts.to(device, dtype=dtype)  # (n_tok, d_in)
        if arch_T <= 1:
            with torch.no_grad():
                z = arch_module.encode(x)  # (n_tok, d_sae) or (n_tok, 1, d_sae)
            if z.dim() == 3 and z.shape[1] == 1:
                z = z.squeeze(1)
            z_mean = z.mean(dim=0)
        else:
            if x.shape[0] < arch_T:
                # pad-front by repeating first token to fit one window
                pad = x[0:1].expand(arch_T - x.shape[0], -1)
                x = torch.cat([pad, x], dim=0)
            # (n_win, T, d_in) via stride-1 unfold
            wins = x.unfold(0, arch_T, 1).permute(0, 2, 1).contiguous()
            with torch.no_grad():
                z = arch_module.encode(wins)
            if z.dim() == 3 and z.shape[1] == 1:
                z = z.squeeze(1)
            z_mean = z.mean(dim=0)
        out.append(z_mean.to(dtype=torch.float32).cpu().numpy())
    return np.array(out)  # (N, d_sae)


# (encode_with_arch removed — encode_per_prompt_z handles dictionary
# encoding correctly with per-token mean pooling, see above)


def run_cell(cell: Cell, *, probe_prompts, base_model, tokenizer,
             lora_model, layer, device, arch_T_for):
    """Encode + compute AUCs/LR for one cell. Cached helpers passed in."""
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.model_selection import StratifiedKFold

    out_dir = OUT_ROOT / f"c6_{cell.train_key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-organism activation cache (full sequences, BASE + LoRA shared
    # across SAE/TXC of that organism).
    import torch
    org_cache = OUT_ROOT / f"_acts_{cell.organism.replace('/', '_')}"
    org_cache.mkdir(parents=True, exist_ok=True)
    base_pkl = org_cache / "base_acts.pt"
    lora_pkl = org_cache / "lora_acts.pt"

    if base_pkl.exists() and lora_pkl.exists():
        log.info("[detect] loading cached %s full-seq acts", cell.organism)
        base_seqs = torch.load(base_pkl, weights_only=False)
        lora_seqs = torch.load(lora_pkl, weights_only=False)
    else:
        log.info("[detect] computing BASE full-seq acts on %d prompts",
                 len(probe_prompts))
        base_seqs = collect_full_seq_acts(
            base_model, tokenizer, probe_prompts, layer, device,
        )
        log.info("[detect] computing LoRA full-seq acts on %d prompts",
                 len(probe_prompts))
        lora_seqs = collect_full_seq_acts(
            lora_model, tokenizer, probe_prompts, layer, device,
        )
        torch.save(base_seqs, base_pkl)
        torch.save(lora_seqs, lora_pkl)
    log.info("[detect] base seqs=%d  mean n_tok=%.0f  d_in=%d",
             len(base_seqs),
             sum(s.shape[0] for s in base_seqs) / max(1, len(base_seqs)),
             base_seqs[0].shape[1])

    # Load arch + state_dict
    from temp_bench.config import load_arch, instantiate_arch
    from safetensors.torch import load_file
    spec = load_arch(cell.arch, component="c6")
    d_in = base_seqs[0].shape[1]
    arch_module = instantiate_arch(spec, d_in=d_in)
    sd_path = (WORKTREE / "purified" / "checkpoints" / cell.train_key
               / "model.safetensors")
    if not sd_path.exists():
        from huggingface_hub import hf_hub_download
        sd_path = Path(hf_hub_download(
            repo_id="han1823123123/temp-bench-models",
            filename=f"{cell.train_key}/model.safetensors",
            token=os.environ.get("HF_TOKEN"),
            cache_dir="/tmp/c6_redteam/hf_cache",
        ))
    arch_module.load_state_dict(load_file(str(sd_path)))
    if torch.cuda.is_available():
        arch_module = arch_module.cuda()
    arch_module.eval()

    arch_T = arch_T_for(cell.arch)
    log.info("[detect] encoding %d base seqs + %d lora seqs (T=%d)",
             len(base_seqs), len(lora_seqs), arch_T)
    Z_base = encode_per_prompt_z(arch_module, base_seqs, arch_T)
    Z_lora = encode_per_prompt_z(arch_module, lora_seqs, arch_T)
    log.info("[detect] Z_base=%s Z_lora=%s d_sae=%d",
             Z_base.shape, Z_lora.shape, Z_base.shape[1])

    import numpy as np
    X = np.concatenate([Z_base, Z_lora], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(len(Z_base)),
                        np.ones(len(Z_lora))]).astype(int)

    # ── (a) Per-feature AUC ─────────────────────────────────────────
    log.info("[detect] computing per-feature AUC (%d features)", X.shape[1])
    aucs = np.zeros(X.shape[1], dtype=np.float32)
    # Vectorized: roc_auc_score per column
    for f in range(X.shape[1]):
        col = X[:, f]
        if col.max() == col.min():
            aucs[f] = 0.5
        else:
            try:
                aucs[f] = roc_auc_score(y, col)
            except Exception:
                aucs[f] = 0.5
    # Two-sided AUC: best feature is one whose distribution most
    # separates classes either direction.
    aucs_two = np.maximum(aucs, 1.0 - aucs)

    finalist = headline_feature_id(cell.train_key)
    finalist_auc = float(aucs[finalist])
    finalist_auc_two = float(aucs_two[finalist])
    rank_one_sided = int((aucs > finalist_auc).sum())
    rank_two_sided = int((aucs_two > finalist_auc_two).sum())

    top10_two = np.argsort(-aucs_two)[:10]
    top10_payload = [
        {"feature_id": int(f),
         "auc": float(aucs[f]), "auc_two_sided": float(aucs_two[f])}
        for f in top10_two
    ]

    # ── (b) Multi-feature LogisticRegression (5-fold CV) ────────────
    log.info("[detect] fitting multi-feature LR (5-fold CV)")
    accs, lr_aucs = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(
            C=1.0, penalty="l2", solver="liblinear",
            max_iter=2000,
        )
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        lr_aucs.append(float(roc_auc_score(y[te], p)))
        accs.append(float(accuracy_score(y[te], clf.predict(X[te]))))
    multi_acc = float(np.mean(accs))
    multi_auc = float(np.mean(lr_aucs))

    payload = {
        "cell": {
            "arch": cell.arch, "organism": cell.organism,
            "seed": cell.seed, "train_key": cell.train_key,
            "datasource": cell.datasource,
        },
        "finalist": {
            "feature_id": finalist,
            "auc": finalist_auc,
            "auc_two_sided": finalist_auc_two,
            "rank_one_sided": rank_one_sided,
            "rank_two_sided": rank_two_sided,
        },
        "top10_features": top10_payload,
        "multi_feature": {
            "logreg_acc_5fold": multi_acc,
            "logreg_auc_5fold": multi_auc,
        },
        "n_features": int(X.shape[1]),
        "n_per_class": int(len(Z_base)),
    }
    (out_dir / "detection.json").write_text(json.dumps(payload, indent=2))
    log.info(
        "[detect] %s/%s/s=%d  feat=%d  auc=%.3f (rank %d/%d two-sided)  "
        "multi-LR acc=%.3f auc=%.3f",
        cell.arch, cell.organism, cell.seed, finalist,
        finalist_auc_two, rank_two_sided + 1, X.shape[1],
        multi_acc, multi_auc,
    )

    # Free arch.
    del arch_module
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="+", default=None)
    p.add_argument("--organism", choices=["14B-finance", "7B-medical"],
                   default=None)
    p.add_argument("--skip-existing", action="store_true",
                   help="skip cells whose detection.json already exists")
    p.add_argument("--n-prompts", type=int, default=N_PROBE_PROMPTS)
    args = p.parse_args(argv)

    cells = list(CELLS)
    if args.cells:
        cells = [c for c in cells if c.train_key in args.cells]
    if args.organism:
        cells = [c for c in cells if c.organism == args.organism]
    if args.skip_existing:
        cells = [c for c in cells
                 if not (OUT_ROOT / f"c6_{c.train_key}" / "detection.json").exists()]

    log.info("[detect] selected %d cell(s)", len(cells))

    # Group by organism (load each subject model + LoRA once).
    by_org: dict[str, list[Cell]] = {}
    for c in cells:
        by_org.setdefault(c.organism, []).append(c)

    from temp_bench.config import load_datasource
    from temp_bench.case_studies.em import load_subject_with_lora

    for organism, org_cells in by_org.items():
        ds = load_datasource(org_cells[0].datasource)
        ds_d = ds.model_dump()
        base_model_id = ds_d["subject_model"]
        adapter_id = ds_d.get("lora_adapter")
        layer = int(ds_d["layer"])
        probe_repo = PROBE_REPO_BY_DATASOURCE[org_cells[0].datasource]

        log.info("[detect] loading %d probes from %s", args.n_prompts, probe_repo)
        prompts = load_probe_prompts(probe_repo, args.n_prompts)
        log.info("[detect] loaded %d probes", len(prompts))

        log.info("[detect] loading BASE %s", base_model_id)
        base_model, tokenizer = load_subject_with_lora(
            base_model_id=base_model_id, adapter_id=None,
        )
        log.info("[detect] loading LoRA %s", adapter_id)
        lora_model, _ = load_subject_with_lora(
            base_model_id=base_model_id, adapter_id=adapter_id,
        )

        def arch_T_for(arch: str) -> int:
            return 5 if arch == "txc_base" else 1

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for cell in org_cells:
            try:
                run_cell(cell,
                         probe_prompts=prompts,
                         base_model=base_model,
                         tokenizer=tokenizer,
                         lora_model=lora_model,
                         layer=layer, device=device,
                         arch_T_for=arch_T_for)
            except Exception:
                log.exception("[detect] cell %s failed", cell.train_key)

        # Drop both models before next organism.
        del base_model, lora_model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("[detect] all done. outputs under %s", OUT_ROOT)


if __name__ == "__main__":
    main()
