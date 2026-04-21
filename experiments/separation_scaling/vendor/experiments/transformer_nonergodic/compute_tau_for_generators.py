"""Compute τ = 1 - H(C | X_{1:T}) / H(C | X_1) for each generator in current use.

τ is a purely generator-level scalar: the fraction of post-one-token uncertainty
about the component identity C that is resolvable only by integrating over time.
Bounded in [0, 1]. Depends only on P(C, X_{1:T}).

We compute τ HONESTLY: sequences are sampled from the nonergodic mixture
(each sequence committed to one component), but the posterior P(C | X_{0:t}) is
computed with a uniform prior over components — i.e., as seen by an observer
who does not know which component the sequence came from. Per-component
likelihoods come from running a forward filter over each component's HMM
(respecting its per-component vocab_map).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

ROOT = Path(__file__).parent
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sae_day.nonergodic_generator import (  # noqa: E402
    NonergodicGenerator,
    _mess3_identity,
    _mess3_reset,
)


def mess3_T(x: float, a: float) -> np.ndarray:
    """Returns transition-emission tensor T of shape (3, 3, 3).

    T[o, s, s'] = P(next_state = s', emit o | current_state = s), where
    o ∈ {0, 1, 2} indexes the emitted mess3 token and s, s' ∈ {0, 1, 2} are hidden
    states. Ported directly from simplexity.generative_processes.transition_matrices.mess3.
    """
    from simplexity.generative_processes.transition_matrices import mess3

    T = np.asarray(mess3(x, a))
    return T


def mess3_reset_T(x: float, a: float, r: float) -> np.ndarray:
    T = np.asarray(_mess3_reset(x, a, r))
    return T


def mess3_identity_T(x: float, a: float, r: float) -> np.ndarray:
    T = np.asarray(_mess3_identity(x, a, r))
    return T


GENERATORS: list[tuple[str, callable, callable]] = [
    (
        "mess3_shared_3x_close",
        lambda: NonergodicGenerator.mess3_shared(
            params=[(0.13, 0.61), (0.18, 0.60), (0.49, 0.60)]
        ),
        lambda: (
            [mess3_T(0.13, 0.61), mess3_T(0.18, 0.60), mess3_T(0.49, 0.60)],
            [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        ),
    ),
    (
        "mess3_shared_3x_separated",
        lambda: NonergodicGenerator.mess3_shared(
            params=[(0.08, 0.9), (0.25, 0.65), (0.4, 0.34)]
        ),
        lambda: (
            [mess3_T(0.08, 0.9), mess3_T(0.25, 0.65), mess3_T(0.4, 0.34)],
            [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        ),
    ),
    (
        "mess3_reset_separated (r=0.02)",
        lambda: NonergodicGenerator.mess3_reset(
            params=[(0.08, 0.9, 0.02), (0.25, 0.65, 0.02), (0.4, 0.34, 0.02)]
        ),
        lambda: (
            [mess3_reset_T(0.08, 0.9, 0.02), mess3_reset_T(0.25, 0.65, 0.02), mess3_reset_T(0.4, 0.34, 0.02)],
            [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]],
        ),
    ),
    (
        "mess3_identity_separated (r=0.02)",
        lambda: NonergodicGenerator.mess3_identity(
            params=[(0.08, 0.9, 0.02), (0.25, 0.65, 0.02), (0.4, 0.34, 0.02)]
        ),
        lambda: (
            [mess3_identity_T(0.08, 0.9, 0.02), mess3_identity_T(0.25, 0.65, 0.02), mess3_identity_T(0.4, 0.34, 0.02)],
            [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]],
        ),
    ),
    (
        "mess3_reset_kyle (x close, a=.6, r=.02)",
        lambda: NonergodicGenerator.mess3_reset(
            params=[(0.13, 0.61, 0.02), (0.18, 0.60, 0.02), (0.49, 0.60, 0.02)]
        ),
        lambda: (
            [mess3_reset_T(0.13, 0.61, 0.02), mess3_reset_T(0.18, 0.60, 0.02), mess3_reset_T(0.49, 0.60, 0.02)],
            [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]],
        ),
    ),
]


def forward_filter_single_component(
    T_c: np.ndarray, vocab_map: list[int], tokens: np.ndarray
) -> np.ndarray:
    """Return per-position cumulative log P(X_{0:t} | C=c).

    T_c: (V_c, S, S); vocab_map: list of length V_c mapping local-actions → global tokens.
    tokens: (T,) in GLOBAL vocab.
    Returns: log_ps of shape (T,) — cumulative log-likelihood up to and including position t.
    """
    n_states = T_c.shape[1]
    # Inverse vocab_map: global → local index, or -1 if not producible by this component
    max_global = max(vocab_map) + 1
    inv = -np.ones(max(max_global, int(tokens.max()) + 1), dtype=np.int64)
    for local, global_tok in enumerate(vocab_map):
        inv[global_tok] = local

    seq_len = tokens.shape[0]
    log_ps = np.empty(seq_len, dtype=np.float64)
    alpha = np.full(n_states, 1.0 / n_states)  # uniform state prior
    log_cum = 0.0
    for t in range(seq_len):
        tok = int(tokens[t])
        local = -1 if tok >= len(inv) else int(inv[tok])
        if local < 0:
            # This component cannot emit this token → likelihood 0 from here on
            log_cum = -np.inf
            log_ps[t:] = -np.inf
            break
        # alpha_next[s'] = sum_s alpha[s] * T[local, s, s']
        alpha_next = alpha @ T_c[local]
        p_tok = float(alpha_next.sum())
        if p_tok < 1e-30:
            log_cum = -np.inf
            log_ps[t:] = -np.inf
            break
        log_cum += float(np.log(p_tok))
        alpha = alpha_next / p_tok
        log_ps[t] = log_cum
    return log_ps


def posterior_over_components(
    Ts: list[np.ndarray], vocab_maps: list[list[int]], tokens_batch: np.ndarray
) -> np.ndarray:
    """Return P(C | X_{0:t}) of shape (B, T, n_components).

    tokens_batch: (B, T), uniform prior over components.
    """
    B, T = tokens_batch.shape
    n_c = len(Ts)
    log_like = np.empty((B, T, n_c), dtype=np.float64)
    for b in range(B):
        for c in range(n_c):
            log_like[b, :, c] = forward_filter_single_component(Ts[c], vocab_maps[c], tokens_batch[b])
    log_prior = np.log(np.full(n_c, 1.0 / n_c))
    log_joint = log_like + log_prior[None, None, :]
    # Softmax across components
    log_joint -= log_joint.max(axis=-1, keepdims=True)
    post = np.exp(log_joint)
    post /= post.sum(axis=-1, keepdims=True)
    return post


def entropy_bits(p: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return -np.sum(p * np.log2(p + eps), axis=axis)


def main() -> None:
    n_eval = 500
    seq_len = 128
    seed = 7

    h_prior = np.log2(3.0)
    print(f"H(C) (uniform prior over 3 components) = {h_prior:.4f} bits")
    print(f"n_eval={n_eval}  seq_len={seq_len}\n")

    results = []
    for name, build_gen, build_matrices in GENERATORS:
        print(f"[{name}]  sampling…", flush=True)
        gen = build_gen()
        # Sample sequences from the nonergodic mixture (each sequence fixed to one component)
        ds = gen.sample(
            n_sequences=n_eval, seq_len=seq_len, seed=seed, with_component_labels=True,
        )
        tokens = ds.tokens.numpy()  # (B, T) in global vocab
        Ts, vocab_maps = build_matrices()
        post = posterior_over_components(Ts, vocab_maps, tokens)  # (B, T, n_c)
        h_per = entropy_bits(post, axis=-1)  # (B, T)
        h_by_t = h_per.mean(axis=0)
        h_one = float(h_by_t[0])
        h_full = float(h_by_t[-1])
        tau = 1.0 - (h_full / h_one) if h_one > 1e-9 else 0.0
        drop_total = h_one - h_full
        h_10 = float(h_by_t[min(9, seq_len - 1)])
        frac = (h_one - h_10) / drop_total if drop_total > 1e-6 else float("nan")
        results.append((name, h_one, h_full, tau, frac, h_by_t))
        print(
            f"  H(C|X_1)={h_one:.3f}  H(C|X_T)={h_full:.3f}  τ={tau:.3f}  "
            f"(frac drop in first 10 tokens: {frac:.2f})"
        )

    print("\nEntropy of P(C | X_{0:t}) vs t (bits), averaged over sequences:")
    ts_to_show = [0, 1, 4, 16, 64, seq_len - 1]
    print(f"  {'name':<44s} | " + "  ".join(f"t={tt:>3d}" for tt in ts_to_show))
    for name, *_, h_by_t in results:
        tcols = [h_by_t[tt] for tt in ts_to_show]
        print(f"  {name:<44s} | " + "  ".join(f"{v:.3f}" for v in tcols))

    out = {
        "metric": "tau = 1 - H(C|X_T) / H(C|X_1)",
        "n_eval": n_eval,
        "seq_len": seq_len,
        "H_prior_bits": h_prior,
        "generators": {
            name: {
                "H_C_given_X1_bits": h_one,
                "H_C_given_XT_bits": h_full,
                "tau": tau,
                "frac_drop_in_first_10_tokens": float(frac) if np.isfinite(frac) else None,
                "H_by_t_bits": [float(v) for v in h_by_t.tolist()],
            }
            for (name, h_one, h_full, tau, frac, h_by_t) in results
        },
    }
    out_json = ROOT / "tau_by_generator.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
