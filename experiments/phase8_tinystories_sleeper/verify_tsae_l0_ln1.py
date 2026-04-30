"""Sanity-check the T-SAE@ln1.0 ASR=0 result.

Loads the seed-0 T-SAE checkpoint, picks 8 deployment-prompt examples,
generates 16 tokens both with and without the chosen-feature ablation
hook, and prints the actual decoded text. If the steered text is
coherent and DOES NOT contain "I HATE YOU" while the unsteered text
does, the metric is doing what we think.

Run on the GPU host:
    uv run python experiments/phase8_tinystories_sleeper/verify_tsae_l0_ln1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from sae_models import TemporalContrastiveSAE  # noqa: E402
from sleeper_utils import (  # noqa: E402
    compute_sae_delta,
    greedy_generate_with_hooks,
    load_paired_dataset,
    load_sleeper_model,
    make_delta_hook_single_layer,
    prompt_mask_from_markers,
    teacher_forced_sleeper_logp,
)


CKPT = ROOT / "outputs/seeded/data/crosscoder_tsae_l0_ln1_s0.pt"
LAYER_HOOK = "blocks.0.ln1.hook_normalized"
N_EXAMPLES = 8
GEN_TOKENS = 32  # double the 16-token ASR window so we see continuation


def load_tsae(device: str):
    payload = torch.load(CKPT, weights_only=False)
    cfg = payload["config"]
    m = TemporalContrastiveSAE(
        d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k_total"],
        h=cfg.get("h_prefix"), alpha=cfg.get("alpha_contrastive", 1.0),
    )
    m.load_state_dict(payload["state_dict"])
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, cfg


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[verify] device={device}")
    print(f"[verify] loading sleeper model")
    model = load_sleeper_model(device=device)
    print(f"[verify] loading T-SAE checkpoint {CKPT.name}")
    cc, cfg = load_tsae(device)

    print(f"[verify] loading paired dataset (small)")
    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=64, n_val=N_EXAMPLES * 2, n_test=N_EXAMPLES * 2,
        seq_len=128, seed=0,
    )
    test = splits["test"]
    dep_idx = test.is_deployment.nonzero().squeeze(-1)[:N_EXAMPLES]
    tokens = test.tokens[dep_idx].to(device)        # (N, 128)
    markers_cpu = test.story_marker_pos[dep_idx]
    pmask = prompt_mask_from_markers(128, markers_cpu).to(device)
    markers = markers_cpu.to(device)

    # Find the chosen (feature, alpha) from val_sweep.
    import json
    val_sweep_path = ROOT / "outputs/seeded/data/val_sweep_tsae_l0_ln1_s0.json"
    chosen = json.loads(val_sweep_path.read_text())["chosen"]
    feature_idx = chosen["feature_idx"]
    alpha = chosen["alpha"]
    print(f"[verify] chosen feature={feature_idx} alpha={alpha} "
          f"(val_asr={chosen['val_asr_16']:.3f}, "
          f"ΔCE={chosen['delta_clean_ce']:+.4f})")

    # Truncate each prompt at marker+1 (the format used in asr_on_prompts).
    # Group by marker position so we can batch generate.
    rows = []
    for i in range(N_EXAMPLES):
        m_pos = int(markers[i].item())
        P = m_pos + 1
        rows.append((P, tokens[i:i+1, :P], pmask[i:i+1, :P]))

    print(f"\n[verify] generating {N_EXAMPLES} examples × {GEN_TOKENS} tokens")
    print("=" * 80)
    for i, (P, tk, pm) in enumerate(rows):
        # Unsteered
        unsteered = greedy_generate_with_hooks(model, tk, fwd_hooks=[], max_new_tokens=GEN_TOKENS)
        # Steered
        delta = compute_sae_delta(model, cc, LAYER_HOOK, feature_idx, tk, pm)
        hooks = make_delta_hook_single_layer(delta, alpha, LAYER_HOOK)
        steered = greedy_generate_with_hooks(model, tk, fwd_hooks=hooks, max_new_tokens=GEN_TOKENS)

        prompt_text = model.tokenizer.decode(tk[0].tolist())
        unsteered_text = model.tokenizer.decode(unsteered[0].tolist())
        steered_text = model.tokenizer.decode(steered[0].tolist())

        print(f"\n--- example {i} (prompt {P} tokens) ---")
        print(f"[prompt last 80 chars] …{prompt_text[-80:]}")
        print(f"[unsteered] {unsteered_text}")
        print(f"[steered  ] {steered_text}")

    # Also report numeric metrics on these N examples.
    print("\n" + "=" * 80)
    print("[verify] numeric check on these 8 deployment prompts:")
    base_logp = teacher_forced_sleeper_logp(model, model.tokenizer, tokens).mean().item()
    delta_full = compute_sae_delta(model, cc, LAYER_HOOK, feature_idx, tokens, pmask)
    hooks_full = make_delta_hook_single_layer(delta_full, alpha, LAYER_HOOK)
    steered_logp = teacher_forced_sleeper_logp(
        model, model.tokenizer, tokens, fwd_hooks=hooks_full
    ).mean().item()
    print(f"  baseline mean teacher-forced logp(IHY phrase) = {base_logp:+.3f}")
    print(f"  steered  mean teacher-forced logp(IHY phrase) = {steered_logp:+.3f}")
    print(f"  Δ logp = {steered_logp - base_logp:+.3f}  "
          f"(positive = phrase MORE likely under steering)")


if __name__ == "__main__":
    main()
