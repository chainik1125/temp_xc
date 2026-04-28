"""Short-horizon (5k-step) comparison of han-derived configurations
plus the spec-§14 pure-contrastive sweep:

  (A) han_antidead_plus_bricken — han's TXCBareAntidead + Bricken resample @ 500
  (C2, λ_c ∈ {0.1, 0.3, 1.0}) han_pure_contrastive_a{0p1,0p3,1} —
      TXCBareMatryoshkaContrastiveAntidead with matryoshka off, shift=1 pairs,
      contr_prefix=d_sae//5
  (B) han_multidistance_champion — H8 champion recipe: bare TXC + matryoshka
      H/L + multi-distance InfoNCE (shifts {1, 2}) + anti-dead

All conditions: d_sae=32k, T=5, k=128, 5000 steps, same streaming buffer.
Appends to dead_feature_experiment.json and regenerates the full plot.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.em_features.architectures.txc_bare_antidead import TXCBareAntidead  # noqa: E402
from experiments.em_features.architectures.txc_bare_matryoshka_contrastive_antidead import (  # noqa: E402
    TXCBareMatryoshkaContrastiveAntidead,
)
from experiments.em_features.architectures.txc_bare_multidistance_contrastive_antidead import (  # noqa: E402
    TXCBareMultiDistanceContrastiveAntidead,
)
from experiments.em_features.streaming_buffer import (  # noqa: E402
    StreamingActivationBuffer, StreamingBufferConfig, mixed_text_iter,
)
from experiments.em_features.dead_feature_resample import DeadFeatureResampler  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--total_steps", type=int, default=5000)
    p.add_argument("--d_sae", type=int, default=32768)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--aux_k", type=int, default=512)
    p.add_argument("--dead_threshold_tokens", type=int, default=640_000)
    p.add_argument("--auxk_alpha", type=float, default=1.0 / 32.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shifts", type=int, nargs="+", default=(1, 2),
                   help="Multi-distance shifts for H8 (non-zero offsets).")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=[
                       "han_antidead_plus_bricken",
                       "han_pure_contrastive_a0p1",
                       "han_pure_contrastive_a0p3",
                       "han_pure_contrastive_a1",
                       "han_pure_contrastive_a2",
                       "han_pure_contrastive_a5",
                       "han_pure_contrastive_a10",
                       "han_multidistance_champion",
                   ])
    return p.parse_args()


@torch.no_grad()
def probe_dead(model, sample_fn, n_check=2048):
    x = sample_fn(n_check).to(next(model.parameters()).device).float()
    z = model.encode(x) if not isinstance(x, tuple) else model.encode(x[0])
    fire_count = (z > 0).sum(dim=0)
    return {
        "n_dead": int((fire_count == 0).sum().item()),
        "n_features": model.d_sae,
        "max_fire": int(fire_count.max().item()),
    }


def train_one(
    name, model, opt, sample_fn, args,
    *, use_bricken=False, pair_sample_fn=None,
):
    history = []
    t0 = time.time()
    resampler = None
    if use_bricken:
        resampler = DeadFeatureResampler(
            model, resample_every=500, min_fires=1, n_check=2048,
        )

    use_pair = pair_sample_fn is not None

    for step in range(args.total_steps):
        if use_pair:
            x_train = pair_sample_fn(args.batch_size)  # (B, 1+K, T, d)
            loss, x_hat, z = model(x_train)
            x_for_probe = x_train[:, 0] if x_train.ndim == 4 else x_train
        else:
            x_train = sample_fn(args.batch_size)       # (B, T, d)
            loss, x_hat, z = model(x_train)
            x_for_probe = x_train

        opt.zero_grad()
        loss.backward()
        # Anti-dead stack ordering: grad-project → clip → step → renorm
        if hasattr(model, "remove_gradient_parallel_to_decoder"):
            model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if hasattr(model, "_normalize_decoder"):
            model._normalize_decoder()

        if resampler is not None:
            resampler.maybe_resample(step + 1, sample_fn)

        if (step + 1) % args.log_every == 0:
            probe = probe_dead(model, sample_fn)
            history.append({
                "step": step + 1,
                "loss": float(loss.detach()),
                "loss_auxk": float(getattr(model, "last_auxk_loss", torch.tensor(0.0)).item()),
                "loss_total": float(loss.detach()),
                "n_dead": probe["n_dead"],
                "n_features": probe["n_features"],
                "n_active_in_batch": int((z > 0).any(dim=0).sum().item()),
                "max_fire": probe["max_fire"],
                "elapsed_min": (time.time() - t0) / 60,
                "n_resampled_so_far": (
                    sum(h.n_resampled for h in resampler.history) if resampler else 0
                ),
            })
            print(f"[{name}] step {step+1:>5}/{args.total_steps}  "
                  f"loss={history[-1]['loss']:.1f}  "
                  f"auxk={history[-1]['loss_auxk']:.4f}  "
                  f"dead={probe['n_dead']}/{probe['n_features']} "
                  f"({100*probe['n_dead']/probe['n_features']:.1f}%)  "
                  f"active={history[-1]['n_active_in_batch']}  "
                  f"resampled={history[-1]['n_resampled_so_far']}", flush=True)

    return history


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(args.config.read_text())

    print(f"Loading {cfg['subject_model']}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["subject_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model_hf = AutoModelForCausalLM.from_pretrained(
        cfg["subject_model"], torch_dtype=torch.float16, device_map=args.device,
    ).eval()

    d_model = int(cfg["d_model"])
    buf_cfg = StreamingBufferConfig(
        layer=int(cfg["layer_txc"]),
        d_model=d_model,
        buffer_seqs=max(1, int(cfg["streaming"]["buffer_activations"]) // int(cfg["streaming"]["chunk_len"])),
        chunk_len=int(cfg["streaming"]["chunk_len"]),
        refill_chunks=int(cfg["streaming"]["refill_chunks"]),
        device=args.device,
        dtype=torch.float16,
    )
    text_iter = mixed_text_iter(tok, cfg["streaming"]["corpus_mix"])
    buffer = StreamingActivationBuffer(model_hf, tok, text_iter, buf_cfg)
    print("warmup...", flush=True)
    buffer.warmup()

    def sample_fn(n):
        return buffer.sample_txc_windows(n, args.T).float()

    # Pair sampler for multi-distance. Needs a buf with (N, L, d) shape
    # where L >= T + max(shifts). The StreamingActivationBuffer stores as
    # (N_seqs, chunk_len, d_model) — chunk_len=256 — so L=256 is plenty.
    # We access buffer._buf directly (a 3D tensor of activations).
    max_shift = max(args.shifts)
    assert buf_cfg.chunk_len >= args.T + max_shift, \
        f"chunk_len={buf_cfg.chunk_len} < T+max_shift={args.T + max_shift}"

    def pair_sample_fn(batch_size):
        """Returns (B, 1+K, T, d) where K = len(shifts)."""
        buf = buffer._buf  # (N_seqs, chunk_len, d_model)
        N, L, _ = buf.shape
        seq = torch.randint(0, N, (batch_size,), device=buf.device)
        off = torch.randint(0, L - args.T - max_shift, (batch_size,), device=buf.device)
        rng = torch.arange(args.T, device=buf.device)
        outs = []
        for s in (0, *args.shifts):
            pos = (off + s).unsqueeze(1) + rng.unsqueeze(0)  # (B, T)
            w = buf[seq.unsqueeze(1).expand(-1, args.T), pos].float()
            outs.append(w)
        return torch.stack(outs, dim=1)

    out_json = args.out_dir / "dead_feature_experiment.json"
    data = json.loads(out_json.read_text()) if out_json.exists() else {}

    # --- (A) han_antidead + Bricken ---
    if "han_antidead_plus_bricken" not in args.skip:
        print("\n================ han_antidead + Bricken ================", flush=True)
        torch.manual_seed(42)
        m = TXCBareAntidead(
            d_in=d_model, d_sae=args.d_sae, T=args.T, k=args.k,
            aux_k=args.aux_k, dead_threshold_tokens=args.dead_threshold_tokens,
            auxk_alpha=args.auxk_alpha,
        ).to(args.device)
        x0 = sample_fn(args.batch_size)
        m.init_b_dec_geometric_median(x0)
        opt = torch.optim.Adam(m.parameters(), lr=args.lr)
        data["han_antidead_plus_bricken"] = train_one(
            "han+bricken", m, opt, sample_fn, args, use_bricken=True,
        )
        with out_json.open("w") as f:
            json.dump(data, f, indent=2)

    # --- (C2) pure contrastive, λ_c sweep, shift=1 ---
    # Spec §14 recipe: matryoshka off, InfoNCE only on h-prefix (d_sae/5),
    # anti-dead stack on. Uses shift=1 pairs sliced from the existing
    # multi-distance pair_sample_fn so we share the same buffer-indexing code.
    def pair_sample_fn_shift1(batch_size):
        return pair_sample_fn(batch_size)[:, :2]  # (B, 2, T, d): anchor + shift=1

    for alpha_c, alpha_tag in (
        (0.1, "a0p1"), (0.3, "a0p3"), (1.0, "a1"),
        (2.0, "a2"), (5.0, "a5"), (10.0, "a10"),
    ):
        cond_name = f"han_pure_contrastive_{alpha_tag}"
        if cond_name in args.skip:
            continue
        print(f"\n============ {cond_name} (C2 spec, λ_c={alpha_c}) ============", flush=True)
        torch.manual_seed(42)
        m = TXCBareMatryoshkaContrastiveAntidead(
            d_in=d_model, d_sae=args.d_sae, T=args.T, k=args.k,
            matryoshka_h_size=None,          # matryoshka off — isolate contrastive effect
            alpha=alpha_c,                   # λ_c
            contr_prefix=args.d_sae // 5,    # h-prefix ≈ 20% of features (spec §15)
            aux_k=args.aux_k,
            dead_threshold_tokens=args.dead_threshold_tokens,
            auxk_alpha=args.auxk_alpha,
        ).to(args.device)
        x0 = sample_fn(args.batch_size)      # (B, T, d) for b_dec init
        m.init_b_dec_geometric_median(x0)
        opt = torch.optim.Adam(m.parameters(), lr=args.lr)
        data[cond_name] = train_one(
            cond_name, m, opt, sample_fn, args,
            pair_sample_fn=pair_sample_fn_shift1,
        )
        with out_json.open("w") as f:
            json.dump(data, f, indent=2)

    # --- (B) han multi-distance champion (H8 recipe) ---
    if "han_multidistance_champion" not in args.skip:
        print("\n========== han_multidistance_champion (H8 recipe) ==========", flush=True)
        torch.manual_seed(42)
        m = TXCBareMultiDistanceContrastiveAntidead(
            d_in=d_model, d_sae=args.d_sae, T=args.T, k=args.k,
            shifts=tuple(args.shifts),
            matryoshka_h_size=args.d_sae // 5,   # H prefix = d_sae/5, per h8 recipe
            alpha=1.0,                           # contrastive weight
            aux_k=args.aux_k,
            dead_threshold_tokens=args.dead_threshold_tokens,
            auxk_alpha=args.auxk_alpha,
        ).to(args.device)
        # For anchor-only b_dec init, feed a (B, T, d) batch.
        x0 = sample_fn(args.batch_size)
        m.init_b_dec_geometric_median(x0)
        opt = torch.optim.Adam(m.parameters(), lr=args.lr)
        data["han_multidistance_champion"] = train_one(
            "h8_champion", m, opt, sample_fn, args,
            pair_sample_fn=pair_sample_fn,
        )
        with out_json.open("w") as f:
            json.dump(data, f, indent=2)

    print("done")


if __name__ == "__main__":
    main()
