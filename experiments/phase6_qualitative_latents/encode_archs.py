"""Phase 6: encode concat-sets A, B, C through each SAE arch.

For each concat-set and each arch, emit a per-token latent tensor saved
as fp16 under `z_cache/<concat>/<arch>__z.npy`. Supporting label files
record token-level provenance (source passage for A/B, subject+qid for
C), matched by position with the z tensor.

Encoding conventions (per arch shape constraint):

    agentic_txc_02  — window-based (T=5). For each position t in the
                      concat we feed L13[t-2:t+3] (edge-replicated
                      padding), take the full (B, d_sae) output.
    agentic_mlc_08  — multi-layer stack (L11-L15). At position t we feed
                      [resid_L11[t], ..., resid_L15[t]] of shape (5, d_in).
    tfa_big         — whole-sequence self-attn with learned novel/pred
                      split. We feed fixed 128-token chunks (pad last),
                      use results_dict["novel_codes"] as the per-token z.
    tsae_ours       — plain per-token. Feed L13[t] of shape (d_in,).

Only concat-set A/B produce single long sequences; C produces 160
short 20-token sequences. The script batches C by padding to 20.

Run:
    PYTHONPATH=. .venv/bin/python \
        experiments/phase6_qualitative_latents/encode_archs.py \
        [--archs agentic_txc_02 agentic_mlc_08 tsae_ours tfa_big]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONCAT_DIR = REPO / "experiments/phase6_qualitative_latents/concat_corpora"
OUT_DIR = REPO / "experiments/phase6_qualitative_latents/z_cache"
CKPT_DIR = REPO / "experiments/phase5_downstream_utility/results/ckpts"
HF_TOKEN = open("/workspace/hf_cache/token").read().strip()

DEFAULT_ARCHS = ["agentic_txc_02", "agentic_mlc_08", "tsae_paper", "tsae_ours", "tfa_big"]
D_IN = 2304
D_SAE = 18_432


def load_concats():
    """Return tuple (A, B, C) — dicts from JSON on disk."""
    A = json.loads((CONCAT_DIR / "concat_A.json").read_text())
    B = json.loads((CONCAT_DIR / "concat_B.json").read_text())
    C = json.loads((CONCAT_DIR / "concat_C.json").read_text())
    return A, B, C


# ─────────────────────────────────────────────── Gemma resid extraction


def gemma_resid_cache(
    token_ids: list[list[int]],
    layers: list[int],
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Run Gemma-2-2b-IT forward, capture resid at each layer.

    Returns {layer_idx: np.ndarray of shape (n_seqs, max_seq_len, d_in)}
    in fp16. Padded sequences are zero-padded; caller discards per
    the per-seq `n_tokens`.
    """
    from transformers import AutoModelForCausalLM
    from src.data.nlp.models import resid_hook_target, get_model_config

    cfg = get_model_config("gemma-2-2b-it")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_path, torch_dtype=torch.float16, token=HF_TOKEN,
    ).to(device).eval()

    max_len = max(len(ids) for ids in token_ids)
    ids_tensor = torch.zeros((len(token_ids), max_len), dtype=torch.long,
                             device=device)
    attn_mask = torch.zeros((len(token_ids), max_len), dtype=torch.long,
                            device=device)
    for i, ids in enumerate(token_ids):
        ids_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn_mask[i, :len(ids)] = 1

    captured: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    handles = []
    for L in layers:
        tgt = resid_hook_target(model, L, cfg.architecture_family)

        def make_hook(Lc):
            def fn(module, inp, out):
                acts = out[0] if isinstance(out, tuple) else out
                captured[Lc].append(acts.detach().to(torch.float16).cpu())
            return fn

        handles.append(tgt.register_forward_hook(make_hook(L)))

    BATCH = 16  # 160 × 20-token seqs fit; long A/B go one at a time
    with torch.no_grad():
        for b0 in range(0, len(token_ids), BATCH):
            b1 = min(b0 + BATCH, len(token_ids))
            _ = model(input_ids=ids_tensor[b0:b1],
                      attention_mask=attn_mask[b0:b1])

    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()

    resid = {}
    for L in layers:
        resid[L] = torch.cat(captured[L], dim=0).numpy()  # fp16
    return resid


# ─────────────────────────────────────────────── per-arch encoders


def load_arch(arch: str, device: torch.device) -> torch.nn.Module:
    """Load one Phase 5/6 arch from its ckpt into eval mode on GPU."""
    ckpt_path = CKPT_DIR / f"{arch}__seed42.pt"
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = state["meta"]

    if arch == "agentic_txc_02":
        from src.architectures.matryoshka_txcdr_contrastive_multiscale import (
            MatryoshkaTXCDRContrastiveMultiscale,
        )
        T = meta["T"]
        k_eff = meta["k_win"] or (meta["k_pos"] * T)
        model = MatryoshkaTXCDRContrastiveMultiscale(
            D_IN, D_SAE, T, k_eff,
            n_contr_scales=meta.get("n_contr_scales", 3),
            gamma=meta.get("gamma", 0.5),
        ).to(device)
    elif arch == "agentic_mlc_08":
        from src.architectures.mlc_contrastive_multiscale import (
            MLCContrastiveMultiscale,
        )
        model = MLCContrastiveMultiscale(
            D_IN, D_SAE, n_layers=5, k=meta["k_pos"],
            gamma=meta.get("gamma", 0.5),
        ).to(device)
    elif arch == "tsae_ours":
        from src.architectures.tsae_ours import TSAEOurs
        model = TSAEOurs(D_IN, D_SAE, k=meta["k_pos"]).to(device)
    elif arch == "tsae_paper":
        from src.architectures.tsae_paper import TemporalMatryoshkaBatchTopKSAE
        group_sizes = meta.get("group_sizes")
        d_sae_eff = int(meta.get("d_sae", D_SAE))
        model = TemporalMatryoshkaBatchTopKSAE(
            D_IN, d_sae_eff, k=int(meta["k_pos"]), group_sizes=list(group_sizes),
        ).to(device)
    elif arch == "tfa_big":
        from src.architectures._tfa_module import TemporalSAE
        model = TemporalSAE(
            dimin=D_IN, width=D_SAE, n_heads=4,
            sae_diff_type="topk", kval_topk=meta["k_pos"],
            tied_weights=True, n_attn_layers=1,
            bottleneck_factor=4, use_pos_encoding=False,
        ).to(device)
    else:
        raise ValueError(f"unknown arch {arch}")

    cast_state = {
        k: v.to(torch.float32) if v.dtype == torch.float16 else v
        for k, v in state["state_dict"].items()
    }
    model.load_state_dict(cast_state)
    model.eval()
    return model, meta


def edge_pad_window(x: torch.Tensor, t: int, T: int) -> torch.Tensor:
    """Return (T, d) window centred on position t; edge-replicate pads."""
    n = x.shape[0]
    half = T // 2
    idx = [max(0, min(n - 1, t + i - half)) for i in range(T)]
    return x[idx]


@torch.no_grad()
def encode_txc(model, resid_L13: torch.Tensor, device, T: int = 5) -> np.ndarray:
    """Slide T-window over resid_L13; return (n_tokens, d_sae) fp16."""
    n_tokens = resid_L13.shape[0]
    windows = torch.stack([edge_pad_window(resid_L13, t, T)
                           for t in range(n_tokens)])  # (n, T, d)
    z_rows = []
    BATCH = 64
    for b0 in range(0, n_tokens, BATCH):
        b1 = min(b0 + BATCH, n_tokens)
        x = windows[b0:b1].to(device).float()
        z = model.encode(x)           # (B, d_sae)
        z_rows.append(z.to(torch.float16).cpu())
    return torch.cat(z_rows, dim=0).numpy()


@torch.no_grad()
def encode_mlc(model, resid_stack: torch.Tensor, device) -> np.ndarray:
    """resid_stack: (n_tokens, 5, d) -> (n_tokens, d_sae)."""
    n_tokens = resid_stack.shape[0]
    z_rows = []
    BATCH = 64
    for b0 in range(0, n_tokens, BATCH):
        b1 = min(b0 + BATCH, n_tokens)
        x = resid_stack[b0:b1].to(device).float()
        z = model.encode(x)
        z_rows.append(z.to(torch.float16).cpu())
    return torch.cat(z_rows, dim=0).numpy()


@torch.no_grad()
def encode_tsae_ours(model, resid_L13: torch.Tensor, device) -> np.ndarray:
    """resid_L13: (n_tokens, d) -> (n_tokens, d_sae)."""
    n_tokens = resid_L13.shape[0]
    z_rows = []
    BATCH = 256
    for b0 in range(0, n_tokens, BATCH):
        b1 = min(b0 + BATCH, n_tokens)
        x = resid_L13[b0:b1].to(device).float()
        z = model.encode(x)
        z_rows.append(z.to(torch.float16).cpu())
    return torch.cat(z_rows, dim=0).numpy()


@torch.no_grad()
def encode_tsae_paper(model, resid_L13: torch.Tensor, device) -> np.ndarray:
    """Paper's T-SAE: threshold-based inference encoding, not BatchTopK.

    resid_L13: (n_tokens, d) -> (n_tokens, d_sae).
    """
    n_tokens = resid_L13.shape[0]
    z_rows = []
    BATCH = 256
    for b0 in range(0, n_tokens, BATCH):
        b1 = min(b0 + BATCH, n_tokens)
        x = resid_L13[b0:b1].to(device).float()
        z = model.encode(x, use_threshold=True)
        z_rows.append(z.to(torch.float16).cpu())
    return torch.cat(z_rows, dim=0).numpy()


@torch.no_grad()
def encode_tfa(model, resid_L13: torch.Tensor, device,
               seq_len: int = 128) -> np.ndarray:
    """resid_L13: (n_tokens, d) -> (n_tokens, d_sae) using novel_codes.

    Feeds non-overlapping seq_len-chunks (pad last); returns novel_codes.
    """
    n_tokens = resid_L13.shape[0]
    z_out = torch.zeros((n_tokens, D_SAE), dtype=torch.float16)
    for c0 in range(0, n_tokens, seq_len):
        c1 = min(c0 + seq_len, n_tokens)
        chunk = resid_L13[c0:c1]
        pad = seq_len - chunk.shape[0]
        if pad > 0:
            chunk = torch.cat([chunk, chunk[-1:].expand(pad, -1)], dim=0)
        x = chunk.unsqueeze(0).to(device).float()  # (1, seq_len, d)
        _, rd = model(x)
        z = rd["novel_codes"].squeeze(0)  # (seq_len, d_sae)
        z_out[c0:c1] = z[: c1 - c0].to(torch.float16).cpu()
    return z_out.numpy()


# ─────────────────────────────────────────────── driver


def encode_concat_AB(concat, concat_name: str, archs: list[str], device):
    """Encode a single long concat (A or B) under each arch."""
    out = OUT_DIR / concat_name
    out.mkdir(parents=True, exist_ok=True)

    token_ids = concat["token_ids"]
    needs_ml = "agentic_mlc_08" in archs
    layers = [11, 12, 13, 14, 15] if needs_ml else [13]

    print(f"[{concat_name}] gemma forward: {len(token_ids)} tokens, layers={layers}")
    resid = gemma_resid_cache([token_ids], layers, device)
    # (1, n_tokens, d_in) per layer -> drop batch
    resid = {L: torch.from_numpy(r[0]).float() for L, r in resid.items()}  # (n_tokens, d)
    if needs_ml:
        stack = torch.stack([resid[L] for L in (11, 12, 13, 14, 15)],
                            dim=1)  # (n_tokens, 5, d)
    resid_L13 = resid[13]

    # Save provenance (same structure as input, for downstream)
    (out / "provenance.json").write_text(json.dumps(concat, indent=2))

    for arch in archs:
        print(f"[{concat_name}] encode: {arch}")
        model, meta = load_arch(arch, device)
        if arch == "agentic_txc_02":
            z = encode_txc(model, resid_L13, device, T=meta["T"])
        elif arch == "agentic_mlc_08":
            z = encode_mlc(model, stack, device)
        elif arch == "tsae_ours":
            z = encode_tsae_ours(model, resid_L13, device)
        elif arch == "tsae_paper":
            z = encode_tsae_paper(model, resid_L13, device)
        elif arch == "tfa_big":
            z = encode_tfa(model, resid_L13, device,
                           seq_len=int(meta.get("T", 128)))
        else:
            raise ValueError(arch)
        np.save(out / f"{arch}__z.npy", z)
        print(f"  saved {arch}: {z.shape}  ({z.nbytes/1e6:.1f} MB fp16)")
        del model
        torch.cuda.empty_cache()


def encode_concat_C(concat, archs: list[str], device):
    """Encode 160 × 20-token sequences under each arch."""
    out = OUT_DIR / "concat_C"
    out.mkdir(parents=True, exist_ok=True)

    token_id_lists = [s["token_ids"] for s in concat["sequences"]]
    needs_ml = "agentic_mlc_08" in archs
    layers = [11, 12, 13, 14, 15] if needs_ml else [13]

    print(f"[concat_C] gemma forward: {len(token_id_lists)} seqs × 20 tokens, layers={layers}")
    resid = gemma_resid_cache(token_id_lists, layers, device)
    resid = {L: torch.from_numpy(r).float() for L, r in resid.items()}  # (160, 20, d)
    n_seq, seq_len, _ = resid[13].shape
    if needs_ml:
        stack = torch.stack([resid[L] for L in (11, 12, 13, 14, 15)],
                            dim=2)  # (160, 20, 5, d)

    (out / "provenance.json").write_text(json.dumps(concat, indent=2))

    for arch in archs:
        print(f"[concat_C] encode: {arch}")
        model, meta = load_arch(arch, device)
        # We encode each 20-token seq independently so TXC windowing
        # uses edge-replicated padding within each seq.
        z_all = np.zeros((n_seq, seq_len, D_SAE), dtype=np.float16)
        for si in range(n_seq):
            resid_L13_i = resid[13][si]  # (20, d)
            if arch == "agentic_txc_02":
                z = encode_txc(model, resid_L13_i, device, T=meta["T"])
            elif arch == "agentic_mlc_08":
                z = encode_mlc(model, stack[si], device)  # (20, 5, d) -> (20, d_sae)
            elif arch == "tsae_ours":
                z = encode_tsae_ours(model, resid_L13_i, device)
            elif arch == "tsae_paper":
                z = encode_tsae_paper(model, resid_L13_i, device)
            elif arch == "tfa_big":
                z = encode_tfa(model, resid_L13_i, device,
                               seq_len=int(meta.get("T", 128)))
            else:
                raise ValueError(arch)
            z_all[si] = z
        np.save(out / f"{arch}__z.npy", z_all)
        print(f"  saved {arch}: {z_all.shape}  ({z_all.nbytes/1e6:.1f} MB fp16)")
        del model
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", type=str, nargs="+", default=DEFAULT_ARCHS)
    p.add_argument("--sets", type=str, nargs="+",
                   default=["A", "B", "C"])
    args = p.parse_args()

    # Skip archs whose ckpt isn't present (e.g. tfa_big not trained yet).
    archs = [a for a in args.archs
             if (CKPT_DIR / f"{a}__seed42.pt").exists()]
    missing = set(args.archs) - set(archs)
    if missing:
        print(f"skipping absent ckpts: {sorted(missing)}")
    if not archs:
        raise SystemExit("no archs available")

    device = torch.device("cuda")
    A, B, C = load_concats()

    if "A" in args.sets:
        encode_concat_AB(A, "concat_A", archs, device)
    if "B" in args.sets:
        encode_concat_AB(B, "concat_B", archs, device)
    if "C" in args.sets:
        encode_concat_C(C, archs, device)


if __name__ == "__main__":
    main()
