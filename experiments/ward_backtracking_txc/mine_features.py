"""Phase 3 — mine backtracking-selective features per hookpoint.

For each trained TXC checkpoint:
  1. Re-collect base-Llama activations over Stage A's dom-split traces, but
     this time recording every token offset (not just the [-13, -8] window).
     We grab a (T = 6, d_model) window per labelled sentence centered on
     the Ward window.
  2. Encode each window through the trained TXC. The encoder yields one
     `z` ∈ R^d_sae per window (shared latent across the T positions).
  3. Rank features by:
        score(f) = mean(z[f] | D+) − mean(z[f] | D−)
     (D+ = backtracking sentences, D− = others.)
  4. For the top-K features, compute per-offset activation profile across
     the [-13, -8] window using the per-position encoder pre-activation.
  5. Save:
        results/ward_backtracking_txc/features/<hp_key>.npz
            top_features:    (K,) feature ids (int)
            scores:          (K,) selectivity scores
            per_offset:      (K, T) mean(z+) − mean(z-) split per offset
            decoder_at_pos0: (K, d_model) — used by B1 as steering vector
            decoder_union:   (K, d_model) — mean across T-slot decoders
            mean_pos:        (K,) mean activation on D+
            mean_neg:        (K,) mean activation on D−
            pos_act:         (n_pos, K) per-sentence activation, D+
            neg_act:         (n_neg, K) per-sentence activation, D−
            sentence_keys_pos / sentence_keys_neg: (n,) string ids
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from experiments.ward_backtracking_txc.architectures import (
    build_arch, arch_encode_window, arch_decoder_directions,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.mine")


def _load_arch_ckpt(ckpt_path: Path):
    """Load any registered arch from checkpoint. Returns (model, cfg_dict)."""
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = obj["config"]
    arch = c.get("arch", "txc")             # legacy ckpts default to txc
    arch_kwargs = c.get("arch_kwargs", {}) or {}
    model = build_arch(arch, d_in=c["d_in"], d_sae=c["d_sae"],
                       T=c["T"], k=c["k_per_position"], **arch_kwargs)
    model.load_state_dict(obj["state_dict"])
    model.eval().to("cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    c["arch"] = arch                          # ensure key present
    return model, c


def _capture_windows(
    hf_id: str,
    layer: int,
    component: str,
    offsets_window: list[int],   # e.g. [-13, ..., -8]
    traces_by_qid: dict,
    labels: list[dict],
    dom_qids: set[str],
    device: str = "cuda",
):
    """Return (X, is_bt, sentence_keys) where X has shape (n_sent, T, d_model)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info("[load] %s (%s)", hf_id, component)
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map=device,
    ).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    captured = {}
    def post_hook(_m, _i, output):
        x = output[0] if isinstance(output, tuple) else output
        if x.dim() == 4:
            x = x.reshape(x.shape[0], x.shape[1], -1)
        captured["x"] = x.detach().to(torch.float32).cpu()
    def pre_hook(_m, args):
        x = args[0]
        if x.dim() == 4:
            x = x.reshape(x.shape[0], x.shape[1], -1)
        captured["x"] = x.detach().to(torch.float32).cpu()

    if component == "resid":
        handle = model.model.layers[layer].register_forward_hook(post_hook)
    elif component == "attn":
        handle = model.model.layers[layer].self_attn.register_forward_hook(post_hook)
    elif component == "ln1":
        handle = model.model.layers[layer].self_attn.register_forward_pre_hook(pre_hook)
    else:
        raise ValueError(f"unknown component: {component}")

    X_list, is_bt, keys = [], [], []
    debug_emitted = 0
    try:
        for record in tqdm(labels, desc=f"capture {component}.L{layer}"):
            qid = record["question_id"]
            if qid not in dom_qids:
                continue
            trace = traces_by_qid.get(qid)
            if trace is None or not record.get("sentences"):
                continue
            full_response = trace["full_response"]
            enc = tok(full_response, return_tensors="pt",
                      return_offsets_mapping=True, add_special_tokens=False)
            input_ids = enc["input_ids"].to(model.device)
            offsets_map = enc["offset_mapping"][0].tolist()
            with torch.no_grad():
                _ = model(input_ids=input_ids)
            acts = captured["x"][0].numpy()  # (seq, d)
            seq_len = acts.shape[0]

            think_open = "<think>"
            think_idx = full_response.find(think_open)
            think_offset = (think_idx + len(think_open)) if think_idx >= 0 else 0

            for s_idx, sent in enumerate(record["sentences"]):
                target_char = think_offset + sent["char_start"]
                tok_pos = -1
                for i, (cs, ce) in enumerate(offsets_map):
                    if cs <= target_char < ce or cs >= target_char:
                        tok_pos = i; break
                if tok_pos < 0:
                    continue
                window = np.zeros((len(offsets_window), acts.shape[1]), dtype=np.float32)
                ok = True
                for j, off in enumerate(offsets_window):
                    p = tok_pos + off
                    if p < 0 or p >= seq_len:
                        ok = False; break
                    window[j] = acts[p]
                if not ok:
                    continue
                X_list.append(window)
                is_bt.append(bool(sent["is_backtracking"]))
                keys.append(f"{qid}|{record['trace_idx']}|{s_idx}")

                if sent["is_backtracking"] and debug_emitted < 3:
                    log.info("[align] sent=%r", sent["sentence"][:80])
                    debug_emitted += 1

            captured.clear()
            torch.cuda.empty_cache()
    finally:
        handle.remove()

    del model
    torch.cuda.empty_cache()
    if not X_list:
        raise RuntimeError("no sentence windows collected")
    X = np.stack(X_list, axis=0)
    is_bt = np.asarray(is_bt, dtype=bool)
    keys = np.asarray(keys, dtype=object)
    log.info("[done] %s | n_sent=%d | n_bt=%d | shape=%s",
             component, X.shape[0], int(is_bt.sum()), X.shape)
    return X, is_bt, keys


def _encode_in_batches(arch: str, model, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Encode (N, T, d) -> (N, d_sae) window-level activation, dispatched per arch."""
    N = X.shape[0]
    d_sae = model.d_sae if hasattr(model, "d_sae") else model.width
    out = np.zeros((N, d_sae), dtype=np.float32)
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        x = torch.from_numpy(X[i:end]).to("cuda", dtype=torch.float32)  # (B, T, d)
        z = arch_encode_window(arch, model, x)        # (B, d_sae)
        out[i:end] = z.float().cpu().numpy()
    return out


def _per_offset_preact(arch: str, model, X: np.ndarray, feat_idx: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """For chosen features, return (N, K, T) per-position pre-activations.

    For TXC and StackedSAE this uses the per-position W_enc.
    For TopKSAE it applies the single W_enc to each token position.
    For TSAE we approximate by encoding each window and taking the per-position
    code (z_novel + z_pred), which is the natural per-offset analog.
    """
    K = len(feat_idx); T = X.shape[1]
    out = np.zeros((X.shape[0], K, T), dtype=np.float32)
    feat_idx_t = torch.from_numpy(feat_idx).to("cuda")

    if arch == "txc":
        W = model.W_enc[:, :, feat_idx]   # (T, d, K)
        for i in range(0, X.shape[0], batch_size):
            end = min(i + batch_size, X.shape[0])
            x = torch.from_numpy(X[i:end]).to("cuda", dtype=torch.float32)
            with torch.no_grad():
                contribs = torch.einsum("btd,tdk->btk", x, W)
            out[i:end] = contribs.permute(0, 2, 1).float().cpu().numpy()
    elif arch == "stacked_sae":
        # Per-position W_enc lives inside each sub-SAE.
        Ws = torch.stack([sae.W_enc.T for sae in model.saes], dim=0)  # (T, d, d_sae)
        Wk = Ws[:, :, feat_idx]                                       # (T, d, K)
        for i in range(0, X.shape[0], batch_size):
            end = min(i + batch_size, X.shape[0])
            x = torch.from_numpy(X[i:end]).to("cuda", dtype=torch.float32)
            with torch.no_grad():
                contribs = torch.einsum("btd,tdk->btk", x, Wk)
            out[i:end] = contribs.permute(0, 2, 1).float().cpu().numpy()
    elif arch == "topk_sae":
        # Single W_enc applied per-token.
        W = model.W_enc.T[:, feat_idx]                                # (d, K)
        for i in range(0, X.shape[0], batch_size):
            end = min(i + batch_size, X.shape[0])
            x = torch.from_numpy(X[i:end]).to("cuda", dtype=torch.float32)
            with torch.no_grad():
                contribs = torch.einsum("btd,dk->btk", x, W)
            out[i:end] = contribs.permute(0, 2, 1).float().cpu().numpy()
    elif arch == "tsae":
        # Run the full TSAE forward and extract per-position codes for top-K.
        # codes_t = pred_codes + novel_codes (B, T, d_sae).
        for i in range(0, X.shape[0], batch_size):
            end = min(i + batch_size, X.shape[0])
            x = torch.from_numpy(X[i:end]).to("cuda", dtype=torch.float32)
            with torch.no_grad():
                _, results = model(x)
                codes = results["pred_codes"] + results["novel_codes"]   # (B, T, d_sae)
                contribs = codes[..., feat_idx_t]                        # (B, T, K)
            out[i:end] = contribs.permute(0, 2, 1).float().cpu().numpy()
    else:
        raise ValueError(f"unknown arch: {arch}")
    return out


def _process_one(arch: str, hp: dict, cfg: dict) -> None:
    """Mine features for a single (arch, hookpoint) cell."""
    paths = cfg["paths"]
    ckpt_filename = f"txc_{hp['key']}.pt" if arch == "txc" else f"{arch}_{hp['key']}.pt"
    ckpt_path = Path(paths["ckpt_dir"]) / ckpt_filename
    if not ckpt_path.exists():
        log.warning("[skip] %s/%s: no ckpt at %s", arch, hp["key"], ckpt_path); return

    feat_dir = Path(paths["features_dir"])
    feat_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"{hp['key']}.npz" if arch == "txc" else f"{arch}_{hp['key']}.npz"
    out_path = feat_dir / out_filename
    if out_path.exists():
        log.info("[skip] %s exists", out_path); return

    log.info("=" * 70); log.info("[arch=%s hookpoint=%s]", arch, hp["key"])
    model, mcfg = _load_arch_ckpt(ckpt_path)
    T = mcfg["T"]
    offsets_window = list(cfg["mining"]["offset_window"])
    if len(offsets_window) != T:
        raise ValueError(f"mining offset_window len ({len(offsets_window)}) != T ({T})")

    prompts = json.loads(Path(paths["stageA_prompts"]).read_text())
    dom_qids = {p["id"] for p in prompts if p.get("split", "dom") == "dom"}
    traces = json.loads(Path(paths["stageA_traces"]).read_text())
    traces_by_qid = {t["question_id"]: t for t in traces}
    labels = json.loads(Path(paths["stageA_labels"]).read_text())

    X, is_bt, keys = _capture_windows(
        cfg["models"]["base"], hp["layer"], hp["component"],
        offsets_window, traces_by_qid, labels, dom_qids,
    )
    Z = _encode_in_batches(arch, model, X)        # (N, d_sae)

    # Selectivity
    pos_mask = is_bt; neg_mask = ~is_bt
    mu_pos = Z[pos_mask].mean(axis=0)
    mu_neg = Z[neg_mask].mean(axis=0)
    score = mu_pos - mu_neg

    K = int(cfg["mining"]["top_k_features"])
    top = np.argsort(-score)[:K]

    # Per-offset selectivity for top-K
    contribs = _per_offset_preact(arch, model, X, top)
    per_off_pos = contribs[pos_mask].mean(axis=0)
    per_off_neg = contribs[neg_mask].mean(axis=0)
    per_off_diff = per_off_pos - per_off_neg

    # Decoder rows for top-K, per the chosen offset and union, dispatched per arch.
    decs = arch_decoder_directions(arch, model)
    with torch.no_grad():
        dec_pos0 = decs["pos0"][top].float().cpu().numpy()    # (K, d)
        dec_union = decs["union"][top].float().cpu().numpy()  # (K, d)

    # Per-sentence activations for the top-K (used by violins).
    pos_act = Z[pos_mask][:, top]
    neg_act = Z[neg_mask][:, top]

    np.savez(
        out_path,
        top_features=top.astype(np.int64),
        scores=score[top].astype(np.float32),
        all_scores=score.astype(np.float32),
        per_offset=per_off_diff.astype(np.float32),
        per_offset_pos=per_off_pos.astype(np.float32),
        per_offset_neg=per_off_neg.astype(np.float32),
        decoder_at_pos0=dec_pos0,
        decoder_union=dec_union,
        mean_pos=mu_pos[top].astype(np.float32),
        mean_neg=mu_neg[top].astype(np.float32),
        pos_act=pos_act.astype(np.float32),
        neg_act=neg_act.astype(np.float32),
        sentence_keys_pos=keys[pos_mask],
        sentence_keys_neg=keys[neg_mask],
        offsets_window=np.asarray(offsets_window, dtype=np.int32),
    )
    log.info("[saved] %s", out_path)
    del model
    torch.cuda.empty_cache()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--arch", type=str, nargs="+", default=None,
                   help="restrict to these architectures (default: cfg.txc.arch_list)")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    hookpoints = [hp for hp in cfg["hookpoints"] if hp.get("enabled", True)]
    if args.only:
        hookpoints = [hp for hp in hookpoints if hp["key"] == args.only]
    arch_list = args.arch or cfg["txc"].get("arch_list", ["txc"])
    for arch in arch_list:
        for hp in hookpoints:
            _process_one(arch, hp, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
