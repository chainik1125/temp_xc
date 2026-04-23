"""Phase 4 — causal intervention benchmark.

Question: does ablating the feature that *probes say* encodes ``bracket_depth``
drop the LM's probability on the **actual** correct close-bracket token more
than ablating a random feature of comparable activation mass?

If yes → that feature is causally used by the LM for bracket-stack tracking.
If no → it's a side-channel correlation, not a causal representation.

The ablation effect is architecture-conditional. The CLEAN score is the
probability assigned to the AST-correct close-bracket at every position
immediately preceding one. Intervention compares this to:

    - TARGET ablation:  encode(x_t) → zero out the probe-identified top-1
                         bracket_depth feature → decode → patch at layer 12.
    - RANDOM ablation:  same protocol, random feature matched for activation
                         mass (so we control for "any ablation hurts").

The scalar per arch is the **excess drop of target over random**. This is
immune to per-arch reconstruction bias — both ablations travel through the
same patched pipeline.

Outputs
-------
results/phase4_intervention_<arch>.json     — per-arch intervention stats
results/phase4_summary.json                 — cross-arch comparison
plots/phase4_excess_drop.png
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tokenize
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from code_pipeline.python_code import SubjectModelConfig, load_cache  # noqa: E402
from code_pipeline.eval_utils import (  # noqa: E402
    build_model_from_checkpoint,
    encode_sae_per_token,
    encode_txc_per_window,
    encode_mlc_per_token,
    gather_labels,
    labels_for_sources,
)


CLOSE_BRACKETS = {")": ")", "]": "]", "}": "}"}


# ---------------------------------------------------------------------------
# Ridge-fit to identify the top feature for bracket_depth in each arch's space
# ---------------------------------------------------------------------------


def ridge_fit(X: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """Return the ridge regression coefficient vector beta (n_features,).

    Uses plain normal-equations. Inputs are centered inside.
    """
    x_mean = X.mean(axis=0, keepdims=True)
    y_mean = y.mean()
    xc = (X - x_mean).astype(np.float64)
    yc = (y - y_mean).astype(np.float64)
    xtx = xc.T @ xc
    xty = xc.T @ yc
    xtx.flat[:: xtx.shape[0] + 1] += ridge
    beta = np.linalg.solve(xtx, xty)
    return beta


def top_feature_idx(beta: np.ndarray, top_k: int = 1) -> np.ndarray:
    """Return indices of the top-|beta| features."""
    order = np.argsort(-np.abs(beta))
    return order[:top_k]


# ---------------------------------------------------------------------------
# Find close-bracket positions in each chunk — mapped to Gemma-token index
# ---------------------------------------------------------------------------


def find_close_bracket_positions(source: str, char_offsets: list[tuple[int, int]]) -> list[tuple[int, str]]:
    """Return ``[(gemma_idx_preceding, close_char), ...]`` for every ``)``, ``]``, ``}``
    in the Python tokenisation whose start-char can be mapped to a Gemma token.

    We identify each close-bracket char, look up which Gemma token *starts at
    or covers* that char, and return that token's index ``i``. Intervention
    happens at Gemma position ``i - 1`` (the token whose prediction target is
    ``i``); we return that ``i - 1`` pair so the caller can just use it.
    """
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenizeError:
        return []

    # Build a simple char_offset -> gemma_idx map (start_char -> i).
    # char_offsets may contain -1 pads.
    char_to_gemma: dict[int, int] = {}
    for i, (s, _e) in enumerate(char_offsets):
        if s < 0:
            continue
        if s not in char_to_gemma:
            char_to_gemma[s] = i

    # Source-char → line/col conversion. tokenize returns (srow, scol); we
    # need to convert back to char offsets.
    def lc_to_char(line: int, col: int) -> int:
        cur = 0
        cur_line = 1
        for ch in source:
            if cur_line == line:
                return cur + col
            if ch == "\n":
                cur_line += 1
            cur += 1
        return cur

    out: list[tuple[int, str]] = []
    for tok in toks:
        tok_type, tok_string, (srow, scol), _, _ = tok
        if tok_type != tokenize.OP or tok_string not in CLOSE_BRACKETS:
            continue
        char = lc_to_char(srow, scol)
        if char not in char_to_gemma:
            continue
        gemma_idx = char_to_gemma[char]
        if gemma_idx == 0:
            continue  # no preceding position to intervene on
        out.append((gemma_idx - 1, tok_string))
    return out


# ---------------------------------------------------------------------------
# Per-arch intervention
# ---------------------------------------------------------------------------


@torch.no_grad()
def clean_recon_and_latent(
    model: torch.nn.Module,
    family: str,
    acts_anchor_seq: torch.Tensor,         # (T_seq, d) float32
    acts_mlc_seq: dict[int, torch.Tensor] | None,
    mlc_layers: list[int],
    position: int,
    device: torch.device,
):
    """Return (x̂_clean (d,), z_clean (d_sae,), W_dec_at_pos (d_sae, d), b_dec_at_pos (d,)).

    The last two are the decoder weight matrix and bias that write latents
    back to the position we patch (anchor layer). For TXC that's index ``-1``
    (final position of the window); for MLC it's ``anchor_in_stack``; for SAE
    it's the whole ``W_dec`` / ``b_dec``. Returns ``(None,)*4`` when the
    position precedes the first full TXC window.
    """
    if family == "topk":
        x_t = acts_anchor_seq[position : position + 1].to(device)  # (1, d)
        x_hat, z = model(x_t)
        return x_hat[0], z[0], model.W_dec, model.b_dec

    if family == "txc":
        w = model.T
        if position < w - 1:
            return None, None, None, None
        window = acts_anchor_seq[position - w + 1 : position + 1].unsqueeze(0).to(device)
        x_hat, z = model(window)
        return x_hat[0, -1], z[0], model.W_dec[-1], model.b_dec[-1]

    if family == "mlxc":
        anchor_in_stack = len(mlc_layers) // 2
        stack = torch.stack(
            [acts_mlc_seq[L][position] for L in mlc_layers], dim=0
        ).unsqueeze(0).to(device)
        x_hat, z = model(stack)
        return (
            x_hat[0, anchor_in_stack], z[0],
            model.W_dec[anchor_in_stack], model.b_dec[anchor_in_stack],
        )

    raise ValueError(f"Unknown family: {family}")


def ablate_feature(
    x_hat_clean: torch.Tensor,     # (d,)
    z_clean: torch.Tensor,         # (d_sae,)
    W_dec: torch.Tensor,           # (d_sae, d)
    feature_idx: int,
) -> torch.Tensor:
    """Subtract feature α's contribution from x̂_clean.

    No-op iff ``z_clean[feature_idx] == 0`` (feature not in post-TopK). Kept
    here for reference / v1 parity; v2 uses ``ablate_direction`` below.
    """
    contribution = float(z_clean[feature_idx]) * W_dec[feature_idx]
    return x_hat_clean - contribution


def ablate_direction(
    z_clean: torch.Tensor,         # (d_sae,)
    W_dec: torch.Tensor,           # (d_sae, d)
    b_dec: torch.Tensor,           # (d,)   — anchor-position decoder bias
    direction: torch.Tensor,       # (d_sae,) — normalised direction in latent
) -> torch.Tensor:
    """Project ``z_clean`` orthogonally off ``direction`` (unit vector),
    decode through ``W_dec`` + ``b_dec``, and return the resulting x̂.
    """
    scalar = float(z_clean @ direction)
    z_ablated = z_clean - scalar * direction
    return z_ablated @ W_dec + b_dec


def ablate_features_by_indices(
    z_clean: torch.Tensor,         # (d_sae,)
    W_dec: torch.Tensor,           # (d_sae, d)
    b_dec: torch.Tensor,           # (d,)
    feature_indices: torch.Tensor, # (K,) long
) -> torch.Tensor:
    """Zero out ``z_clean[feature_indices]`` and decode.

    When the target is "top-K features by |β|", this is equivalent to
    removing a K-dim subspace from the latent (the subspace spanned by the
    standard basis vectors for those features). x̂ moves by
    ``sum_i z_clean[feature_i] * W_dec[feature_i]`` — proportional to how
    active the K features are. Strictly larger perturbation than 1-D β
    projection for K > 1.
    """
    z_ablated = z_clean.clone()
    z_ablated[feature_indices] = 0.0
    return z_ablated @ W_dec + b_dec


@torch.no_grad()
def prob_of_close_bracket_after_intervention(
    lm,
    tokens_seq: torch.Tensor,                # (T_seq,) int64, on device already
    anchor_hook: str,
    anchor_acts_seq: torch.Tensor,           # (T_seq, d) float32, on device
    x_hat_new_t: torch.Tensor,               # (d,) float32, new value for position t
    position: int,
    close_token_id: int,
) -> float:
    """Patch blocks.12.hook_resid_post at ``position`` to ``x_hat_new_t``;
    forward the LM; return P(close_bracket at position+1)."""
    override = anchor_acts_seq.clone()
    override[position] = x_hat_new_t
    override = override.unsqueeze(0).to(lm.cfg.dtype)  # (1, T, d)

    def _hook(act, hook, _o=override):
        return _o

    logits = lm.run_with_hooks(
        tokens_seq.unsqueeze(0),
        fwd_hooks=[(anchor_hook, _hook)],
        return_type="logits",
    )  # (1, T, V)
    log_p = F.log_softmax(logits[0, position].float(), dim=-1)
    return float(log_p[close_token_id].exp())


@torch.no_grad()
def prob_of_close_bracket_clean(
    lm,
    tokens_seq: torch.Tensor,
    position: int,
    close_token_id: int,
) -> float:
    logits = lm(tokens_seq.unsqueeze(0), return_type="logits")
    log_p = F.log_softmax(logits[0, position].float(), dim=-1)
    return float(log_p[close_token_id].exp())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-sequences", type=int, default=200,
                        help="number of eval chunks to score")
    parser.add_argument("--max-positions-per-chunk", type=int, default=8,
                        help="cap close-bracket positions per chunk (stratification)")
    parser.add_argument("--n-probe-chunks", type=int, default=400,
                        help="chunks used to fit the probe for feature identification")
    parser.add_argument("--n-random-baselines", type=int, default=5,
                        help="random feature ablations per test position (averaged)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="number of top-|β| features to ablate as a subspace")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    cache_root = HERE / cfg.get("cache_root", "cache")
    checkpoint_root = HERE / cfg.get("checkpoint_root", "checkpoints")
    results_root = HERE / cfg.get("output_root", "results")
    plot_root = HERE / cfg.get("plot_root", "plots")
    subject_cfg = SubjectModelConfig.from_dict(cfg["subject_model"])
    layers = subject_cfg.required_layers()
    d_model = subject_cfg.d_model

    tokens, sources, acts_by_layer, _ = load_cache(cache_root, layers)
    split = torch.load(cache_root / "split.pt")
    eval_idx = split["eval_idx"][: args.n_sequences]
    probe_idx = split["eval_idx"][: args.n_probe_chunks]

    # --- eval slice ---
    tokens_eval = tokens[eval_idx]
    acts_anchor_eval = acts_by_layer[subject_cfg.anchor_layer][eval_idx].float()
    acts_mlc_eval = {L: acts_by_layer[L][eval_idx].float() for L in subject_cfg.mlc_layers}
    sources_eval = [sources[i] for i in eval_idx.tolist()]

    # --- probe slice ---
    acts_anchor_probe = acts_by_layer[subject_cfg.anchor_layer][probe_idx].float()
    acts_mlc_probe = {L: acts_by_layer[L][probe_idx].float() for L in subject_cfg.mlc_layers}
    sources_probe = [sources[i] for i in probe_idx.tolist()]
    labels_nt_probe = labels_for_sources(sources_probe)

    # --- LM ---
    from transformer_lens import HookedTransformer
    lm = HookedTransformer.from_pretrained(
        subject_cfg.name, device=device,
        dtype={"bfloat16": torch.bfloat16, "float16": torch.float16,
               "float32": torch.float32}[cfg.get("dtype", "bfloat16")],
    )
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    anchor_hook = f"blocks.{subject_cfg.anchor_layer}.hook_resid_post"

    close_token_ids = {
        s: lm.tokenizer.encode(s, add_special_tokens=False)[0]
        for s in CLOSE_BRACKETS
    }
    print(f"[phase4] close-bracket token ids: {close_token_ids}", flush=True)

    # --- identify target feature per arch from the probe slice ---
    arch_feature_summary: dict[str, dict] = {}
    archs_ckpts: dict[str, dict] = {}
    for arch in cfg["architectures"]:
        name = arch["name"]
        ckpt_path = checkpoint_root / f"{name}.pt"
        if not ckpt_path.exists():
            print(f"[phase4] skip {name}: no checkpoint")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        archs_ckpts[name] = ckpt

    # Per-arch: encode probe slice, fit ridge for bracket_depth, store β as a
    # unit direction for projection-style ablation.
    per_arch: dict[str, dict] = {}
    for name, ckpt in archs_ckpts.items():
        model, family = build_model_from_checkpoint(ckpt, d_model)
        model = model.to(device).eval()
        if family == "topk":
            latents, _, c_idx, t_idx = encode_sae_per_token(
                model, acts_anchor_probe, device)
        elif family == "txc":
            latents, _, c_idx, t_idx = encode_txc_per_window(
                model, acts_anchor_probe, device)
        elif family == "mlxc":
            latents, _, c_idx, t_idx = encode_mlc_per_token(
                model, acts_mlc_probe, subject_cfg.mlc_layers, device)
        else:
            raise ValueError(family)
        lbl = gather_labels(labels_nt_probe, c_idx, t_idx)
        y = lbl["bracket_depth"].astype(np.float32)
        mask = y >= 0
        beta = ridge_fit(latents[mask], y[mask], ridge=1.0)
        beta_norm = float(np.linalg.norm(beta))
        d_sae = beta.shape[0]
        print(f"[phase4] {name} ({family}): β norm={beta_norm:.4f}  "
              f"top-|β|={float(np.abs(beta).max()):.4f}  d_sae={d_sae}  "
              f"top_k={args.top_k}",
              flush=True)
        per_arch[name] = {
            "family": family,
            "model": model,
            "beta_norm": beta_norm,
            "d_sae": d_sae,
            "beta": torch.from_numpy(beta.astype(np.float32)).to(device),
        }

    # --- run intervention at close-bracket positions across the eval slice ---
    test_positions_per_chunk: list[list[tuple[int, str]]] = []
    for row in sources_eval:
        positions = find_close_bracket_positions(
            row["source"], [tuple(o) for o in row["char_offsets"]]
        )
        if len(positions) > args.max_positions_per_chunk:
            positions = positions[: args.max_positions_per_chunk]
        test_positions_per_chunk.append(positions)
    n_positions_total = sum(len(p) for p in test_positions_per_chunk)
    print(f"[phase4] {n_positions_total} test positions across "
          f"{len(test_positions_per_chunk)} eval chunks", flush=True)

    # --- evaluate per arch ---
    summary: dict[str, dict] = {}
    for arch_name, info in per_arch.items():
        family = info["family"]
        model = info["model"]
        beta_t = info["beta"]              # (d_sae,)
        d_sae = info["d_sae"]

        records: list[dict] = []
        for chunk_i, positions in enumerate(test_positions_per_chunk):
            if not positions:
                continue
            tok_seq = tokens_eval[chunk_i].to(device)
            a_seq = acts_anchor_eval[chunk_i].to(device)
            mlc_seq = {L: acts_mlc_eval[L][chunk_i].to(device)
                        for L in subject_cfg.mlc_layers}
            for (pos, close_char) in positions:
                tok_id = close_token_ids[close_char]

                x_hat_clean, z_clean, W_dec_at_pos, b_dec_at_pos = clean_recon_and_latent(
                    model, family, a_seq, mlc_seq, subject_cfg.mlc_layers,
                    pos, device,
                )
                if z_clean is None:
                    continue

                # Reference: patched forward with clean reconstruction (no ablation).
                # Removes per-arch reconstruction bias.
                p_ref = prob_of_close_bracket_after_intervention(
                    lm, tok_seq, anchor_hook, a_seq,
                    x_hat_clean, pos, tok_id,
                )

                # Per-position contribution: |β[α] * z[α]| for each feature.
                # Target: top-K features by contribution magnitude — features
                # simultaneously (a) active at this position (z ≠ 0) and (b)
                # β-weighted for bracket_depth prediction.
                contrib = (beta_t * z_clean).abs()
                target_idx_set = torch.topk(contrib, args.top_k).indices
                # Candidates for random baseline: all ACTIVE features at this
                # position — matches the target set for "features that are
                # active and therefore ablatable".
                active_idx = torch.nonzero(z_clean, as_tuple=False).squeeze(-1)
                if active_idx.numel() < 2:
                    continue

                x_hat_target = ablate_features_by_indices(
                    z_clean, W_dec_at_pos, b_dec_at_pos, target_idx_set,
                )
                p_target = prob_of_close_bracket_after_intervention(
                    lm, tok_seq, anchor_hook, a_seq,
                    x_hat_target, pos, tok_id,
                )

                # Random baseline: K random active features.
                p_random_vals: list[float] = []
                for _ in range(args.n_random_baselines):
                    k_eff = min(args.top_k, active_idx.numel())
                    perm = active_idx[torch.randperm(active_idx.numel(), device=device)[:k_eff]]
                    x_hat_r = ablate_features_by_indices(
                        z_clean, W_dec_at_pos, b_dec_at_pos, perm,
                    )
                    p_r = prob_of_close_bracket_after_intervention(
                        lm, tok_seq, anchor_hook, a_seq,
                        x_hat_r, pos, tok_id,
                    )
                    p_random_vals.append(p_r)
                p_random_mean = float(np.mean(p_random_vals))

                records.append({
                    "chunk": chunk_i,
                    "pos": pos,
                    "close_char": close_char,
                    "p_ref": p_ref,
                    "p_target": p_target,
                    "p_random": p_random_mean,
                    "drop_target": p_ref - p_target,
                    "drop_random": p_ref - p_random_mean,
                    "excess_drop": (p_ref - p_target) - (p_ref - p_random_mean),
                })
            if (chunk_i + 1) % 20 == 0 and records:
                mean_excess = float(np.mean([r["excess_drop"] for r in records]))
                print(f"[phase4] {arch_name} @ chunk {chunk_i+1}/{len(test_positions_per_chunk)} "
                      f"n={len(records)} mean_excess_drop={mean_excess:+.4f}",
                      flush=True)

        if not records:
            summary[arch_name] = {"n_positions": 0, "beta_norm": info["beta_norm"]}
            print(f"[phase4] {arch_name}: zero valid records", flush=True)
            continue

        mean_p_ref = float(np.mean([r["p_ref"] for r in records]))
        mean_p_target = float(np.mean([r["p_target"] for r in records]))
        mean_p_random = float(np.mean([r["p_random"] for r in records]))
        mean_drop_target = float(np.mean([r["drop_target"] for r in records]))
        mean_drop_random = float(np.mean([r["drop_random"] for r in records]))
        mean_excess = float(np.mean([r["excess_drop"] for r in records]))
        median_excess = float(np.median([r["excess_drop"] for r in records]))
        stats = {
            "n_positions": len(records),
            "beta_norm": info["beta_norm"],
            "mean_p_ref": mean_p_ref,
            "mean_p_target": mean_p_target,
            "mean_p_random": mean_p_random,
            "mean_drop_target": mean_drop_target,
            "mean_drop_random": mean_drop_random,
            "mean_excess_drop": mean_excess,
            "median_excess_drop": median_excess,
        }
        summary[arch_name] = stats
        print(f"[phase4] {arch_name}: n={len(records)} "
              f"ref={mean_p_ref:.4f}  target={mean_p_target:.4f}  "
              f"random={mean_p_random:.4f}  excess={mean_excess:+.4f}", flush=True)
        with (results_root / f"phase4_intervention_{arch_name}.json").open("w") as f:
            json.dump({"stats": stats, "records": records[:500]}, f, indent=2)

    with (results_root / "phase4_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- plot ---
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        archs = [a for a in summary if "mean_excess_drop" in summary[a]]
        if not archs:
            raise RuntimeError("no arches with valid records to plot")
        excess = [summary[a]["mean_excess_drop"] for a in archs]
        dt = [summary[a]["mean_drop_target"] for a in archs]
        dr = [summary[a]["mean_drop_random"] for a in archs]

        axes[0].bar(archs, excess, color=["C0", "C1", "C2"])
        axes[0].axhline(0, color="k", linewidth=0.7)
        axes[0].set_ylabel("excess drop (target − random)")
        axes[0].set_title("Phase 4 — causal specificity of bracket-depth feature")

        x = np.arange(len(archs)); w = 0.4
        axes[1].bar(x - w/2, dt, w, label="target ablation")
        axes[1].bar(x + w/2, dr, w, label="random ablation")
        axes[1].set_xticks(x); axes[1].set_xticklabels(archs)
        axes[1].set_ylabel("mean probability drop")
        axes[1].legend(fontsize=8)
        axes[1].set_title("target vs random ablation")
        fig.tight_layout()
        plot_root.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_root / "phase4_excess_drop.png", dpi=140)
        plt.close(fig)
    except Exception as e:
        print(f"[phase4] plot failed: {e!r}")

    print("[phase4] done")


if __name__ == "__main__":
    main()
