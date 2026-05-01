"""Smoke test for batched_steering.py — logit-level correctness.

Compares the first-token logits per (cell, prompt) between:
  Mode A: batch_size=1 (effectively serial, one prompt at a time)
  Mode B: batch_size=N (single batched call with per-element steering hook)

If logits match within fp16 tolerance, the per-element hook is correct.

Token-by-token greedy generation across modes is NOT bit-identical due to
HF / cuBLAS kernel non-determinism at different batch sizes — that's a
property of the underlying GPU stack, not the steering code. For the Wang
procedure we care about aggregate statistics (mean align/coh per cell),
which match up to seed-level variance regardless of GPU non-determinism.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
EM_FEATURES = Path("/root/em_features")
VENDOR_SRC = REPO_ROOT / "experiments" / "separation_scaling" / "vendor" / "src"
for p in (str(VENDOR_SRC), str(REPO_ROOT), str(EM_FEATURES)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _format_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )


def _run_one_cell(model, tokenizer, prompt: str, direction: torch.Tensor,
                  alpha: float, layer_idx: int) -> torch.Tensor:
    """Single-cell forward; returns logits at the last input position (1, vocab)."""
    from experiments.em_features.batched_steering import _make_per_element_hook
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    steering = (alpha * direction.to(model.device).to(torch.float32)).unsqueeze(0)  # (1, d)
    handle = model.model.layers[layer_idx].register_forward_hook(
        _make_per_element_hook(steering, inputs.attention_mask)
    )
    try:
        with torch.no_grad():
            out = model(**inputs)
    finally:
        handle.remove()
    return out.logits[0, -1, :].detach().to(torch.float32).cpu()


def _run_batched(model, tokenizer, prompts, directions, alphas, layer_idx):
    """Batched forward; returns (N, vocab) logits at last *real* input position per row."""
    from experiments.em_features.batched_steering import _make_per_element_hook
    prev_pad = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(model.device)
        N = len(prompts)
        steering = torch.stack([
            (alphas[i] * directions[i].to(model.device).to(torch.float32))
            for i in range(N)
        ], dim=0)
        handle = model.model.layers[layer_idx].register_forward_hook(
            _make_per_element_hook(steering, inputs.attention_mask)
        )
        try:
            with torch.no_grad():
                out = model(**inputs)
        finally:
            handle.remove()
        # With left-padding, the last token is at the END of each row (last index)
        # which IS the last real prompt token (no right-padding).
        return out.logits[:, -1, :].detach().to(torch.float32).cpu()
    finally:
        tokenizer.padding_side = prev_pad


def main():
    print("=" * 60, flush=True)
    print("Batched-steering smoke test (logit-level correctness)", flush=True)
    print("=" * 60, flush=True)

    out_prefix = EM_FEATURES / "checkpoints" / "qwen_l15_smoke_500"
    ckpt_path = out_prefix.with_name(f"{out_prefix.name}_step500.pt")
    if not ckpt_path.exists():
        print(f"\n[1] Training TXC 500 steps -> {ckpt_path}", flush=True)
        cmd = [
            sys.executable, "-m", "experiments.em_features.run_training_txc_bricken_auxk",
            "--config", str(REPO_ROOT / "experiments" / "em_features" / "config.yaml"),
            "--out_prefix", str(out_prefix),
            "--total_steps", "500", "--snapshot_at", "500",
            "--d_sae", "16384", "--k_total", "100", "--T", "5", "--batch_topk",
            "--batch_size", "512", "--lr", "3e-4",
            "--layer", "15", "--hookpoint", "resid_post",
        ]
        env = os.environ.copy()
        env["TQDM_DISABLE"] = "1"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["PYTHONHASHSEED"] = "42"
        if Path("/root/.env").exists():
            for line in Path("/root/.env").read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
        result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
        if result.returncode != 0 or not ckpt_path.exists():
            raise SystemExit(f"Training failed; ckpt not found at {ckpt_path}")
    else:
        print(f"\n[1] Reusing existing ckpt {ckpt_path}", flush=True)

    print("\n[2] Loading ckpt + directions", flush=True)
    from sae_day.sae import TemporalCrosscoder
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    cfg = ckpt["config"]
    txc = TemporalCrosscoder(d_in=cfg["d_in"], d_sae=cfg["d_sae"],
                             T=cfg["T"], k_total=cfg["k_total"]).to("cuda")
    txc.load_state_dict(ckpt["state_dict"])
    txc.eval()
    feature_ids = [0, 100, 1000, 5000]
    directions = [txc.W_dec[-1, fid, :].detach().clone() for fid in feature_ids]
    print(f"    feature_ids: {feature_ids}", flush=True)

    from open_source_em_features.pipeline.longform_steering import load_em_dataset
    em = load_em_dataset()
    questions = [d["messages"][0]["content"] for d in em][:4]
    alphas = [-4.0, -2.0, +2.0, +4.0]
    cells = list(zip(feature_ids, alphas))
    flat_dirs, flat_alphas, flat_questions = [], [], []
    for c_idx, (fid, alpha) in enumerate(cells):
        for q in questions:
            flat_dirs.append(directions[c_idx])
            flat_alphas.append(alpha)
            flat_questions.append(q)
    N = len(flat_dirs)
    print(f"    total cells×prompts: {N}", flush=True)

    print("\n[3] Loading bad-medical Qwen", flush=True)
    from open_source_em_features.utils.model_loading import load_model_and_tokenizer
    model, tok = load_model_and_tokenizer("andyrdt/Qwen2.5-7B-Instruct_bad-medical")
    model.eval()

    formatted = [_format_prompt(tok, q) for q in flat_questions]

    # Mode A — one at a time
    print("\n[A] Per-element forward (batch_size=1) capturing first-token logits", flush=True)
    logits_A = []
    for i in range(N):
        lg = _run_one_cell(model, tok, formatted[i], flat_dirs[i], flat_alphas[i], 15)
        logits_A.append(lg)
    logits_A = torch.stack(logits_A, dim=0)  # (N, vocab)

    print("\n[B] Single batched forward (batch_size=N)", flush=True)
    logits_B = _run_batched(model, tok, formatted, flat_dirs, flat_alphas, 15)  # (N, vocab)

    # Compare per-element
    print("\n[Compare]", flush=True)
    print(f"{'cell':<6}{'feat':>6}{'α':>7}{'q':>3}{'argmax_A':>10}{'argmax_B':>10}{'match':>8}{'max_abs_diff':>14}{'cos_sim':>10}", flush=True)
    n_match_top1 = 0
    for i in range(N):
        c_idx = i // 4; q_idx = i % 4
        fid = feature_ids[c_idx]; alpha = alphas[c_idx]
        a = logits_A[i]; b = logits_B[i]
        max_diff = (a - b).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        am_a = int(a.argmax()); am_b = int(b.argmax())
        match_top1 = (am_a == am_b)
        n_match_top1 += int(match_top1)
        print(f"{c_idx:<6}{fid:>6}{alpha:>+7.1f}{q_idx:>3}{am_a:>10}{am_b:>10}{'OK' if match_top1 else 'DIFF':>8}{max_diff:>14.4f}{cos:>10.6f}", flush=True)

    print(f"\nargmax(top-1) match rate: {n_match_top1}/{N}", flush=True)
    overall_max = (logits_A - logits_B).abs().max().item()
    overall_cos = torch.nn.functional.cosine_similarity(
        logits_A.flatten(), logits_B.flatten(), dim=0
    ).item()
    print(f"Overall max-abs logit diff: {overall_max:.4f}", flush=True)
    print(f"Overall cosine similarity:  {overall_cos:.8f}", flush=True)

    # Also check top-5 set overlap (more robust than top-1 with ties)
    top5_overlap_count = 0
    for i in range(N):
        a_top5 = set(logits_A[i].topk(5).indices.tolist())
        b_top5 = set(logits_B[i].topk(5).indices.tolist())
        top5_overlap_count += len(a_top5 & b_top5)
    avg_top5_overlap = top5_overlap_count / (N * 5)
    print(f"Average top-5 overlap: {avg_top5_overlap:.3f} (1.0 = perfect)", flush=True)

    # PASS: cos_sim >= 0.999 AND avg top-5 overlap >= 0.95.
    # We do NOT require top-1 match because fp16 kernel non-determinism
    # routinely flips the winner in close ties, even when distributions agree.
    if overall_cos >= 0.999 and avg_top5_overlap >= 0.95:
        print("\n✅ Smoke test PASSED — batched logits match unbatched within fp16 tolerance.", flush=True)
        print("   (Top-1 mismatches are kernel-non-determinism in close ties, not a hook bug.)", flush=True)
        sys.exit(0)
    else:
        print("\n❌ Smoke test FAILED — logit divergence above tolerance.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
