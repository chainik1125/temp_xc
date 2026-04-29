"""Phase 5 — B2: cross-model temporal-firing diff.

Run the *base-trained* TXC encoder (no retraining) on the activations of
the **reasoning** model captured on Stage A traces. For each chosen B1
feature, compute mean activation magnitude as a function of token offset
relative to backtracking-sentence start, separately for D+ and D-. Same
on the base activations as a control.

Output: results/ward_backtracking_txc/b2/<hp_key>.npz with arrays:
  offsets               (n_offsets,)             absolute offsets [-30..+5]
  feature_ids           (K,)                     selected feature ids (subset of mined top-K)
  base_pos              (K, n_offsets)           mean activation, base, D+
  base_neg              (K, n_offsets)           mean activation, base, D-
  reasoning_pos         (K, n_offsets)
  reasoning_neg         (K, n_offsets)
  base_pos_se / ...     (K, n_offsets)           ±1 SE across sentences
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

from temporal_crosscoders.models import TemporalCrosscoder

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.b2")


def _capture_full_offset_sweep(
    hf_id: str, layer: int, component: str,
    offsets: list[int], traces_by_qid: dict, labels: list[dict],
    qid_filter: set[str] | None,
    device: str = "cuda",
):
    """Like mine_features._capture_windows but with arbitrary offset list.

    Returns X of shape (n_sent, len(offsets), d_model), is_bt, sentence_keys.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info("[capture %s/%s.L%d] loading model", component, hf_id.split("/")[-1], layer)
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map=device,
    ).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    if component == "resid":
        target = model.model.layers[layer]
    elif component == "attn":
        target = model.model.layers[layer].self_attn
    else:
        raise ValueError(component)

    captured = {}
    def hook(_m, _i, output):
        x = output[0] if isinstance(output, tuple) else output
        if x.dim() == 4:
            x = x.reshape(x.shape[0], x.shape[1], -1)
        captured["x"] = x.detach().to(torch.float32).cpu()
    handle = target.register_forward_hook(hook)

    X_list, is_bt, keys = [], [], []
    try:
        for record in tqdm(labels, desc=f"capture {component}"):
            qid = record["question_id"]
            if qid_filter is not None and qid not in qid_filter:
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
            acts = captured["x"][0].numpy()
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
                window = np.zeros((len(offsets), acts.shape[1]), dtype=np.float32)
                ok = True
                for j, off in enumerate(offsets):
                    p = tok_pos + off
                    if 0 <= p < seq_len:
                        window[j] = acts[p]
                    else:
                        ok = False; break
                if not ok:
                    continue
                X_list.append(window)
                is_bt.append(bool(sent["is_backtracking"]))
                keys.append(f"{qid}|{record['trace_idx']}|{s_idx}")
            captured.clear(); torch.cuda.empty_cache()
    finally:
        handle.remove()
    del model
    torch.cuda.empty_cache()

    return (np.stack(X_list, axis=0),
            np.asarray(is_bt, dtype=bool),
            np.asarray(keys, dtype=object))


def _per_offset_preact(model: TemporalCrosscoder, X: np.ndarray, feat_idx: np.ndarray, T: int, batch: int = 128) -> np.ndarray:
    """X: (N, n_off, d). Returns (N, K, n_off) per-position pre-activations.

    The TXC encoder W_enc is (T, d, d_sae), one slot per Ward window
    position. For B2 we want firing at a *single* offset, so we re-use the
    pos-0 encoder slot (matching the steer_decoder_pos config) — i.e. ask
    "if this token were treated as offset 0 of the window, what's the
    pre-activation?" That's a clean cross-offset comparison.
    """
    pos0 = 0  # convention: read encoder at slot 0 across all offsets
    W0 = model.W_enc[pos0, :, :]                # (d, d_sae)
    Wk = W0[:, feat_idx]                        # (d, K)
    n_off = X.shape[1]
    out = np.zeros((X.shape[0], len(feat_idx), n_off), dtype=np.float32)
    for i in range(0, X.shape[0], batch):
        end = min(i + batch, X.shape[0])
        x = torch.from_numpy(X[i:end]).to("cuda", dtype=torch.float32)  # (B, n_off, d)
        with torch.no_grad():
            preact = torch.einsum("bod,dk->bok", x, Wk)                  # (B, n_off, K)
        out[i:end] = preact.permute(0, 2, 1).cpu().numpy()
    return out


def _agg(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """arr: (N, K, n_off). Returns mean and SE along axis 0 over `mask`."""
    sub = arr[mask]
    if sub.shape[0] == 0:
        K, n_off = arr.shape[1], arr.shape[2]
        return np.zeros((K, n_off)), np.zeros((K, n_off))
    mean = sub.mean(axis=0)
    se = sub.std(axis=0, ddof=1) / max(np.sqrt(sub.shape[0]), 1.0)
    return mean, se


def _process_hookpoint(hp: dict, cfg: dict) -> None:
    paths = cfg["paths"]
    ckpt_path = Path(paths["ckpt_dir"]) / f"txc_{hp['key']}.pt"
    feat_path = Path(paths["features_dir"]) / f"{hp['key']}.npz"
    out_path = Path(paths["b2_dir"]) / f"{hp['key']}.npz"
    Path(paths["b2_dir"]).mkdir(parents=True, exist_ok=True)
    if not ckpt_path.exists() or not feat_path.exists():
        log.warning("[skip] %s: missing ckpt or features", hp["key"]); return
    if out_path.exists():
        log.info("[skip] %s exists", out_path); return

    log.info("=" * 70); log.info("[hookpoint] %s", hp["key"])
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = obj["config"]
    model = TemporalCrosscoder(d_in=c["d_in"], d_sae=c["d_sae"],
                               T=c["T"], k=c["k_per_position"])
    model.load_state_dict(obj["state_dict"])
    model.eval().to("cuda")
    for p in model.parameters():
        p.requires_grad_(False)

    feats = np.load(feat_path, allow_pickle=True)
    K_take = int(cfg["mining"]["top_k_for_steering"])
    feature_ids = feats["top_features"][:K_take]
    log.info("[features] using top-%d: %s", K_take, feature_ids.tolist())

    # Reuse Stage A traces; B2 considers BOTH dom-split (training visibility)
    # and eval-split (held out) — we use all traces to maximize sentence count.
    prompts = json.loads(Path(paths["stageA_prompts"]).read_text())
    qid_set: set[str] = {p["id"] for p in prompts}
    traces = json.loads(Path(paths["stageA_traces"]).read_text())
    traces_by_qid = {t["question_id"]: t for t in traces}
    labels = json.loads(Path(paths["stageA_labels"]).read_text())

    offsets = list(cfg["b2"]["offset_window_full"])
    log.info("[offsets] %s", offsets)

    # Capture activations from BASE for the same sentences/offsets.
    X_b, is_bt_b, keys_b = _capture_full_offset_sweep(
        cfg["models"]["base"], hp["layer"], hp["component"],
        offsets, traces_by_qid, labels, qid_set,
    )
    base_preact = _per_offset_preact(model, X_b, feature_ids, c["T"])

    # Capture from REASONING.
    X_r, is_bt_r, keys_r = _capture_full_offset_sweep(
        cfg["models"]["reasoning"], hp["layer"], hp["component"],
        offsets, traces_by_qid, labels, qid_set,
    )
    reasoning_preact = _per_offset_preact(model, X_r, feature_ids, c["T"])

    base_pos, base_pos_se = _agg(base_preact, is_bt_b)
    base_neg, base_neg_se = _agg(base_preact, ~is_bt_b)
    reas_pos, reas_pos_se = _agg(reasoning_preact, is_bt_r)
    reas_neg, reas_neg_se = _agg(reasoning_preact, ~is_bt_r)

    np.savez(
        out_path,
        offsets=np.asarray(offsets, dtype=np.int32),
        feature_ids=feature_ids.astype(np.int64),
        base_pos=base_pos.astype(np.float32),
        base_pos_se=base_pos_se.astype(np.float32),
        base_neg=base_neg.astype(np.float32),
        base_neg_se=base_neg_se.astype(np.float32),
        reasoning_pos=reas_pos.astype(np.float32),
        reasoning_pos_se=reas_pos_se.astype(np.float32),
        reasoning_neg=reas_neg.astype(np.float32),
        reasoning_neg_se=reas_neg_se.astype(np.float32),
        n_pos_base=int(is_bt_b.sum()),
        n_neg_base=int((~is_bt_b).sum()),
        n_pos_reasoning=int(is_bt_r.sum()),
        n_neg_reasoning=int((~is_bt_r).sum()),
    )
    log.info("[saved] %s", out_path)
    del model; torch.cuda.empty_cache()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--only", type=str, default=None)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    hookpoints = [hp for hp in cfg["hookpoints"] if hp.get("enabled", True)]
    if args.only:
        hookpoints = [hp for hp in hookpoints if hp["key"] == args.only]
    for hp in hookpoints:
        _process_hookpoint(hp, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
