"""Quick visual sanity check: dump 5 TXC-steered sample outputs for
txc_l0_ln1_s0 to confirm whether the recovery=-4.5 number reflects
genuine gibberish (coherence collapse) or some metric artifact."""
import sys
import json
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from sae_models import TemporalCrosscoder  # noqa: E402
from sleeper_utils import (  # noqa: E402
    compute_txc_delta,
    load_paired_dataset,
    load_sleeper_model,
    make_delta_hook_single_layer,
    prompt_mask_from_markers,
    sample_generate_with_hooks,
)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sleeper_model(device=device)

    ckpt_path = ROOT / "outputs/seeded/data/crosscoder_txc_l0_ln1_s0.pt"
    sweep_path = ROOT / "outputs/seeded/data/val_sweep_txc_l0_ln1_s0.json"

    payload = torch.load(ckpt_path, weights_only=False)
    cfg = payload["config"]
    cc = TemporalCrosscoder(
        d_in=cfg["d_in"], d_sae=cfg["d_sae"], T=cfg["T"], k_total=cfg["k_total"],
    )
    cc.load_state_dict(payload["state_dict"])
    cc.to(device).eval()

    chosen = json.loads(sweep_path.read_text())["chosen"]
    f_idx = chosen["feature_idx"]
    alpha = chosen["alpha"]
    print(f"chosen: f={f_idx} alpha={alpha}")

    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=64, n_val=64, n_test=64, seq_len=128, seed=0,
    )
    test = splits["test"]
    dep_idx = test.is_deployment.nonzero().squeeze(-1)[:5]

    for i in dep_idx.tolist():
        m_pos = int(test.story_marker_pos[i].item())
        P = m_pos + 1
        tk = test.tokens[i:i + 1, :P].to(device)
        pm = prompt_mask_from_markers(128, test.story_marker_pos[i:i + 1])[:, :P].to(device)
        delta = compute_txc_delta(
            model, cc, "blocks.0.ln1.hook_normalized", f_idx, tk, pm,
        )
        hooks = make_delta_hook_single_layer(
            delta, alpha, "blocks.0.ln1.hook_normalized",
        )
        # No-hook baseline (poisoned) for reference
        g_pois = sample_generate_with_hooks(
            model, tk, [], max_new_tokens=32,
            temperature=0.8, top_p=0.9, seed=42 + i,
        )
        # Steered
        g_st = sample_generate_with_hooks(
            model, tk, hooks, max_new_tokens=32,
            temperature=0.8, top_p=0.9, seed=42 + i,
        )
        prompt_text = model.tokenizer.decode(tk[0].tolist())
        print(f"\n--- ex {i}  prompt last 60: ...{prompt_text[-60:]!r}")
        print(f"  pois:    {model.tokenizer.decode(g_pois[0].tolist())!r}")
        print(f"  steered: {model.tokenizer.decode(g_st[0].tolist())!r}")


if __name__ == "__main__":
    main()
