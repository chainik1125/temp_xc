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


def _capture_multilayer_windows(
    hf_id: str,
    layers: list[int],
    traces_by_qid: dict,
    labels: list[dict],
    dom_qids: set[str],
    device: str = "cuda",
):
    """MLC variant of `_capture_windows`.

    For each labeled sentence, captures the residual-stream activation at
    the sentence's representative token across ALL `layers` simultaneously.
    Returns X of shape (n_sent, n_layers, d_model). Used for mining MLC
    feature activations — n_layers replaces TXC's T (window) axis.

    The "representative token" is the token that begins the sentence in
    the trace (think_offset + sent.char_start), matching the alignment
    rule used in the single-layer path's `tok_pos`.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info("[load] %s (mlc, layers=%s)", hf_id, layers)
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map=device,
    ).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    captured = {ln: None for ln in layers}
    handles = []
    def _make_hook(ln):
        def hook(_m, _i, output):
            x = output[0] if isinstance(output, tuple) else output
            if x.dim() == 4:
                x = x.reshape(x.shape[0], x.shape[1], -1)
            captured[ln] = x.detach().to(torch.float32).cpu()
        return hook
    for ln in layers:
        handles.append(model.model.layers[ln].register_forward_hook(_make_hook(ln)))

    X_list, is_bt, keys = [], [], []
    try:
        for record in tqdm(labels, desc=f"mlc-capture {layers}"):
            qid = record["question_id"]
            if qid not in dom_qids: continue
            trace = traces_by_qid.get(qid)
            if trace is None or not record.get("sentences"): continue
            full_response = trace["full_response"]
            enc = tok(full_response, return_tensors="pt",
                      return_offsets_mapping=True, add_special_tokens=False)
            input_ids = enc["input_ids"].to(model.device)
            offsets_map = enc["offset_mapping"][0].tolist()
            with torch.no_grad():
                _ = model(input_ids=input_ids)

            think_open = "<think>"
            think_idx = full_response.find(think_open)
            think_offset = (think_idx + len(think_open)) if think_idx >= 0 else 0

            # All layers captured for this trace; align per sentence.
            seq_len = captured[layers[0]].shape[1]
            for s_idx, sent in enumerate(record["sentences"]):
                target_char = think_offset + sent["char_start"]
                tok_pos = -1
                for i, (cs, ce) in enumerate(offsets_map):
                    if cs <= target_char < ce or cs >= target_char:
                        tok_pos = i; break
                if tok_pos < 0 or tok_pos >= seq_len: continue
                # Stack across layers at the single token position.
                stacked = np.stack(
                    [captured[ln][0, tok_pos].numpy() for ln in layers],
                    axis=0,
                )  # (n_layers, d_model)
                X_list.append(stacked.astype(np.float32))
                is_bt.append(bool(sent["is_backtracking"]))
                keys.append(f"{qid}|{record['trace_idx']}|{s_idx}")

            for ln in layers: captured[ln] = None
            torch.cuda.empty_cache()
    finally:
        for h in handles: h.remove()

    del model
    torch.cuda.empty_cache()
    if not X_list:
        raise RuntimeError("no MLC sentence windows collected")
    X = np.stack(X_list, axis=0)            # (n_sent, n_layers, d_model)
    is_bt = np.asarray(is_bt, dtype=bool)
    keys = np.asarray(keys, dtype=object)
    return X, is_bt, keys


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
    def pre_hook(_m, args, kwargs):
        # Newer transformers calls self_attn(hidden_states=...) by kwargs only.
        x = args[0] if args else kwargs["hidden_states"]
        if x.dim() == 4:
            x = x.reshape(x.shape[0], x.shape[1], -1)
        captured["x"] = x.detach().to(torch.float32).cpu()

    if component == "resid":
        handle = model.model.layers[layer].register_forward_hook(post_hook)
    elif component == "attn":
        handle = model.model.layers[layer].self_attn.register_forward_hook(post_hook)
    elif component == "ln1":
        handle = model.model.layers[layer].self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
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
    elif arch in ("tsae", "tsae_paper", "tfa"):
        # Run the full TSAE/TFA forward and extract per-position codes for top-K.
        # codes_t = pred_codes + novel_codes (B, T, d_sae). TFA uses the same
        # TemporalSAE class so the forward signature is identical.
        for i in range(0, X.shape[0], batch_size):
            end = min(i + batch_size, X.shape[0])
            x = torch.from_numpy(X[i:end]).to("cuda", dtype=torch.float32)
            with torch.no_grad():
                _, results = model(x)
                codes = results["pred_codes"] + results["novel_codes"]   # (B, T, d_sae)
                contribs = codes[..., feat_idx_t]                        # (B, T, K)
            out[i:end] = contribs.permute(0, 2, 1).float().cpu().numpy()
    elif arch in ("txc_h8", "txc_h13", "mlc"):
        # W_enc is (T, d_in, d_sae) — same shape as plain TXC. Per-position
        # contribution at slot t for feature k is x[:, t] · W_enc[t, :, k].
        # For MLC, "slot t" indexes layer t (within {L8..L12}); the input X is
        # (n_sent, n_layers, d_model) per `_capture_multilayer_windows`.
        W = model.W_enc[:, :, feat_idx]   # (T, d, K)
        for i in range(0, X.shape[0], batch_size):
            end = min(i + batch_size, X.shape[0])
            x = torch.from_numpy(X[i:end]).to("cuda", dtype=torch.float32)
            with torch.no_grad():
                contribs = torch.einsum("btd,tdk->btk", x, W)
            out[i:end] = contribs.permute(0, 2, 1).float().cpu().numpy()
    else:
        raise ValueError(f"unknown arch: {arch}")
    return out


def _process_one(arch: str, hp: dict, cfg: dict,
                 *, k_per_position: int | None = None,
                 seed: int | None = None,
                 rank: str = "meandiff") -> None:
    """Mine features for a single cell. With k_per_position+seed → paper-budget
    filename; without → legacy sprint filename. The `rank` selects the
    feature-ranking criterion ("meandiff" | "tstat" | "ratio")."""
    # Stash the rank so the deeper helper picks it up without an extra arg.
    _process_one._active_rank = rank
    paths = cfg["paths"]
    if k_per_position is not None and seed is not None:
        ckpt_filename = f"{arch}__{hp['key']}__k{k_per_position}__s{seed}.pt"
        # Per-rank features file; default rank keeps legacy filename for back-compat.
        cell_id = f"{arch}__{hp['key']}__k{k_per_position}__s{seed}"
        out_filename = f"{cell_id}.npz" if rank == "meandiff" else f"{cell_id}__r{rank}.npz"
    else:
        ckpt_filename = f"txc_{hp['key']}.pt" if arch == "txc" else f"{arch}_{hp['key']}.pt"
        out_filename = f"{hp['key']}.npz" if arch == "txc" else f"{arch}_{hp['key']}.npz"
    ckpt_path = Path(paths["ckpt_dir"]) / ckpt_filename
    if not ckpt_path.exists():
        log.warning("[skip] %s/%s: no ckpt at %s", arch, hp["key"], ckpt_path); return

    feat_dir = Path(paths["features_dir"])
    feat_dir.mkdir(parents=True, exist_ok=True)
    out_path = feat_dir / out_filename
    if out_path.exists():
        log.info("[skip] %s exists", out_path); return

    log.info("=" * 70); log.info("[arch=%s hookpoint=%s]", arch, hp["key"])
    model, mcfg = _load_arch_ckpt(ckpt_path)
    T = mcfg["T"]
    offsets_window = list(cfg["mining"]["offset_window"])
    if arch != "mlc" and len(offsets_window) != T:
        raise ValueError(f"mining offset_window len ({len(offsets_window)}) != T ({T})")

    prompts = json.loads(Path(paths["stageA_prompts"]).read_text())
    dom_qids = {p["id"] for p in prompts if p.get("split", "dom") == "dom"}
    traces = json.loads(Path(paths["stageA_traces"]).read_text())
    traces_by_qid = {t["question_id"]: t for t in traces}
    labels = json.loads(Path(paths["stageA_labels"]).read_text())

    if arch == "mlc":
        # MLC: capture multi-layer activations at the sentence token (no
        # token-window). The "T axis" of the encoder input is layers.
        mlc_layers = list(cfg["txc"].get("arch_kwargs", {}).get("mlc", {}).get("layers", [8, 9, 10, 11, 12]))
        X, is_bt, keys = _capture_multilayer_windows(
            cfg["models"]["base"], mlc_layers,
            traces_by_qid, labels, dom_qids,
        )
        # Override offsets_window for the per-offset selectivity plot —
        # "offsets" become layer indices for MLC.
        offsets_window = list(range(len(mlc_layers)))
    else:
        X, is_bt, keys = _capture_windows(
            cfg["models"]["base"], hp["layer"], hp["component"],
            offsets_window, traces_by_qid, labels, dom_qids,
        )
    Z = _encode_in_batches(arch, model, X)        # (N, d_sae)

    # Selectivity — compute three rankings on the same Z matrix.
    pos_mask = is_bt; neg_mask = ~is_bt
    Z_pos = Z[pos_mask]                             # (n_pos, d_sae)
    Z_neg = Z[neg_mask]                             # (n_neg, d_sae)
    n_pos = Z_pos.shape[0]; n_neg = Z_neg.shape[0]
    mu_pos = Z_pos.mean(axis=0)
    mu_neg = Z_neg.mean(axis=0)
    var_pos = Z_pos.var(axis=0, ddof=1) if n_pos > 1 else np.ones_like(mu_pos)
    var_neg = Z_neg.var(axis=0, ddof=1) if n_neg > 1 else np.ones_like(mu_neg)

    # 1. meandiff — mu_pos - mu_neg (legacy)
    score_meandiff = mu_pos - mu_neg
    # 2. tstat — Welch's t-statistic (Dmitry's preferred ranking)
    se = np.sqrt(var_pos / max(n_pos, 1) + var_neg / max(n_neg, 1) + 1e-12)
    score_tstat = (mu_pos - mu_neg) / se
    # 3. ratio — mu_pos / max(mu_neg, eps); guards against negative/zero means
    eps = 1e-6
    score_ratio = mu_pos / np.where(np.abs(mu_neg) < eps, eps, np.abs(mu_neg))

    # Pick the top-K according to the cell's chosen ranking.
    rank = getattr(_process_one, "_active_rank", "meandiff")
    if rank == "meandiff":
        score = score_meandiff
    elif rank == "tstat":
        score = score_tstat
    elif rank == "ratio":
        score = score_ratio
    else:
        raise ValueError(f"unknown ranking: {rank}")

    K = int(cfg["mining"]["top_k_features"])
    # Rank by absolute value (catch both positive and negative selectivity)
    top = np.argsort(-np.abs(score))[:K]

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
        # Save full per-criterion score arrays so any downstream code can
        # rerank without re-encoding the cache.
        all_scores_meandiff=score_meandiff.astype(np.float32),
        all_scores_tstat=score_tstat.astype(np.float32),
        all_scores_ratio=score_ratio.astype(np.float32),
        ranking=np.asarray(rank, dtype="<U16"),
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
    p.add_argument("--cell", type=str, default=None,
                   help="mine one specific cell, format <arch>__<hp>__k<k>__s<seed>")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())

    if args.cell is not None:
        from experiments.ward_backtracking_txc.cell_id import Cell
        cell = Cell.from_id(args.cell)
        all_hp = {hp["key"]: hp for hp in cfg["hookpoints"]}
        if cell.hookpoint_key not in all_hp:
            log.error("cell %s references unknown hookpoint %s", args.cell, cell.hookpoint_key)
            return 1
        _process_one(cell.arch, all_hp[cell.hookpoint_key], cfg,
                     k_per_position=cell.k_per_position, seed=cell.seed,
                     rank=cell.rank)
        return 0

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
