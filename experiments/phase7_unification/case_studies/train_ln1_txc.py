"""Track A pivot — train one TXC at L12 ln1 (pre-attention LayerNorm).

Tests the collaborator's TinyStories claim on Gemma-2-2b: TXC features
trained at the ln1 hook capture pre-attention temporal information that
resid-post features miss.

Single-file pipeline:

  1. Stream ~5k FineWeb-Edu passages; tokenize with Gemma-2-2b tokenizer
     to context length 128.
  2. Forward through gemma-2-2b base; capture output of
     `model.layers[12].input_layernorm` (ln1) per position.
  3. Train TXCBareAntidead (T=5, d_sae=18432, k_pos=100, k_win=500) for
     `--steps` updates, batch=1024 windows. Reduced budget vs paper-grade
     (25k steps) — this is a hypothesis-test smoke run, not a leaderboard
     contender.
  4. Save ckpt + training_log JSON to results/ckpts/, with arch_id
     `txc_bare_antidead_t5_ln1` and hook_name="input_layernorm" baked
     into the meta.

After training, run the existing case-study pipeline (select_features,
diagnose_z_magnitudes, intervene_paper_clamp_normalised, grade) on the
new arch and compare to the resid-post baseline.

Output:
  results/ckpts/txc_bare_antidead_t5_ln1__seed42.pt
  results/training_logs/txc_bare_antidead_t5_ln1__seed42.json

NOTE: this hooks `input_layernorm` output (= the normalised residual
fed to attention), not the pre-norm residual. Matches the
collaborator's wording "ln1 hook normalized (i.e just before
attention)".
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import (
    CKPT_DIR, LOGS_DIR, OUT_DIR, ANCHOR_LAYER, SUBJECT_MODEL, banner,
)
from src.architectures.txc_bare_antidead import TXCBareAntidead, geometric_median


CTX = 128
DTYPE_GEMMA = torch.bfloat16
DTYPE_SAE = torch.float32
N_SEQ_DEFAULT = 5_000
T_DEFAULT = 5
D_IN = 2304
D_SAE = 18_432
K_POS = 100
K_WIN = K_POS * T_DEFAULT  # = 500
HOOK_NAME = "input_layernorm"
ARCH_ID = "txc_bare_antidead_t5_ln1"


def stream_and_tokenize(n_seq: int, tokenizer) -> np.ndarray:
    """Stream FineWeb-Edu, tokenize each passage, return (n_seq, CTX) int32."""
    print(f"  streaming {n_seq} fineweb-edu passages...")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True)
    out = np.zeros((n_seq, CTX), dtype=np.int32)
    n_done = 0
    t0 = time.time()
    pad_id = tokenizer.pad_token_id or 0
    for ex in ds:
        text = ex.get("text", "")
        ids = tokenizer.encode(text, add_special_tokens=True, max_length=CTX, truncation=True)
        if len(ids) < CTX // 2:
            continue  # too short; skip
        if len(ids) < CTX:
            ids = ids + [pad_id] * (CTX - len(ids))
        out[n_done] = ids[:CTX]
        n_done += 1
        if n_done == n_seq:
            break
        if n_done % 1000 == 0:
            print(f"    [{n_done}/{n_seq}] {n_done / (time.time() - t0):.0f} seq/s")
    print(f"  tokenized in {(time.time() - t0):.1f}s")
    return out[:n_done]


def cache_ln1(tokens: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """Forward through gemma-2-2b; capture L12 ln1 output per position."""
    from transformers import AutoModelForCausalLM
    print(f"  loading {SUBJECT_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        SUBJECT_MODEL, torch_dtype=DTYPE_GEMMA, device_map="cuda",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_seq = tokens.shape[0]
    cache = np.zeros((n_seq, CTX, D_IN), dtype=np.float16)
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["x"] = h.detach().cpu()

    target = model.model.layers[ANCHOR_LAYER].input_layernorm
    handle = target.register_forward_hook(hook_fn)

    tokens_t = torch.from_numpy(tokens).long()
    t0 = time.time()
    try:
        with torch.no_grad():
            for start in range(0, n_seq, batch_size):
                end = min(start + batch_size, n_seq)
                batch = tokens_t[start:end].to("cuda")
                captured.clear()
                model(batch)
                acts = captured["x"]
                if acts.shape[-1] != D_IN:
                    acts = acts[..., :D_IN]
                cache[start:end] = acts.to(torch.float16).numpy()
                if (end // batch_size) % 50 == 0:
                    rate = end / max(1e-3, time.time() - t0)
                    print(f"    [{end}/{n_seq}] {rate:.0f} seq/s")
    finally:
        handle.remove()
    print(f"  capture done in {(time.time() - t0)/60:.1f} min")
    del model
    torch.cuda.empty_cache()
    return cache


def make_window_batch(cache: np.ndarray, batch_size: int, T: int, rng: np.random.Generator) -> torch.Tensor:
    """Sample (B, T, d_in) random T-windows from the cache."""
    n_seq = cache.shape[0]
    seq_idx = rng.integers(0, n_seq, size=batch_size)
    pos_idx = rng.integers(0, CTX - T + 1, size=batch_size)
    out = np.zeros((batch_size, T, D_IN), dtype=np.float32)
    for i, (s, p) in enumerate(zip(seq_idx, pos_idx)):
        out[i] = cache[s, p:p+T].astype(np.float32)
    return torch.from_numpy(out)


def train_one_txc(cache: np.ndarray, *, T: int, steps: int, batch_size: int,
                  lr: float, seed: int = 42) -> tuple[TXCBareAntidead, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    print(f"  training TXCBareAntidead (T={T}, k={K_WIN}, d_sae={D_SAE})")
    model = TXCBareAntidead(d_in=D_IN, d_sae=D_SAE, T=T, k=K_WIN).to("cuda").to(DTYPE_SAE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    l0s = []
    dead_counts = []
    init_done = False
    t0 = time.time()
    for step in range(steps):
        x = make_window_batch(cache, batch_size, T, rng).to("cuda")
        if not init_done:
            model.init_b_dec_geometric_median(x[:512])
            init_done = True

        opt.zero_grad()
        loss, x_hat, z = model(x)
        loss.backward()
        model.remove_gradient_parallel_to_decoder()
        opt.step()
        with torch.no_grad():
            model._normalize_decoder()

        losses.append(float(loss.item()))
        l0 = (z > 0).float().sum(dim=-1).mean().item()
        l0s.append(l0)
        dead_counts.append(int(model.last_dead_count.item()))

        if step % 100 == 0:
            elapsed = time.time() - t0
            rate = (step + 1) / max(1e-3, elapsed)
            eta = (steps - step - 1) / max(1e-3, rate)
            print(f"    [{step}/{steps}] loss={loss.item():.1f}  l0={l0:.0f}  dead={dead_counts[-1]}  "
                  f"({rate:.1f} it/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"  train done in {elapsed/60:.1f} min")

    log = {
        "arch_id": ARCH_ID,
        "src_class": "TXCBareAntidead",
        "d_in": D_IN, "d_sae": D_SAE, "T": T,
        "k_pos": K_POS, "k_win": K_WIN, "k": K_WIN,
        "subject_model": SUBJECT_MODEL,
        "anchor_layer": ANCHOR_LAYER,
        "hook_name": HOOK_NAME,
        "elapsed_s": elapsed,
        "final_step": steps,
        "loss": losses,
        "l0": l0s,
        "dead_count": dead_counts,
        "n_train_seqs": int(cache.shape[0]),
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        # null-out fields the loader expects
        "alpha": None, "gamma": None, "n_scales": None,
        "T_max": None, "n_layers": None, "mlc_layers": None,
        "phase": "Y_ln1_pivot",
        "arch": "txc_bare_antidead_ln1",
        "group": None,
        "converged": True,
    }
    return model, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seq", type=int, default=N_SEQ_DEFAULT)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    banner(__file__)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokens = stream_and_tokenize(args.n_seq, tokenizer)
    cache = cache_ln1(tokens, batch_size=16)
    model, log = train_one_txc(
        cache, T=T_DEFAULT, steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed,
    )

    # Save ckpt + log
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{ARCH_ID}__seed{args.seed}.pt"
    log_path = LOGS_DIR / f"{ARCH_ID}__seed{args.seed}.json"
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(sd, ckpt_path)
    log_path.write_text(json.dumps(log, indent=2))
    print(f"  saved ckpt -> {ckpt_path}")
    print(f"  saved log  -> {log_path}")


if __name__ == "__main__":
    main()
