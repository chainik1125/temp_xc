"""RECEIPT for the pre-run audit A1 — the pooled-zero gate cannot fail.

$0, no model, no pods, no network. Run before the shuffle ablation spends
anything:

    PYTHONPATH=. .venv/bin/python -m \
      experiments.explorations.task_hunt.sycgen.shuffle_gate_receipt

## What is being demonstrated

The brief (`briefings/sycgen-shuffle-sparsity-matched.md` §1) makes
pooled's ordered-vs-shuffled gap the instrument gate: it must be 0, and a
non-zero value voids the run. That gate is correct about pooled and says
**nothing about the shuffle**, because `frontier.py:119` is
`z.mean(dim=1)` and a mean over the window axis is permutation-invariant
*arithmetically*.

So the gate returns "pass" in two situations that must not be confused:

    A. the shuffle is LIVE   -> pooled 0 (correct: pooled is invariant)
    B. the shuffle is DEAD   -> pooled 0 (WRONG: nothing was shuffled)

Under B every arm reports a zero gap, which reads as pre-registered
outcome **(b) TXC ~ stacked**, which §2 names the live hypothesis and §4
pre-commits to publishing. **The one instrument failure the design does
not check for is the one that yields the promised headline through a
passing gate.**

This file runs both situations against the REAL
`shuffle_within_window` and a faithful mirror of `frontier.py`'s two
arms (per-position encode, then mean-pool vs reshape), and shows:

  * the pooled gate passes in BOTH -> it does not discriminate;
  * the proposed input-side assert fires in B only -> it does.

Independence of implementation would not have caught this: any number of
independent reimplementations of the gate share the *assumption* that
pooled-zero tests the shuffle, and all of them pass a dead shuffle.
(Hub, `73f8ea388`: "independence of implementation is not independence of
assumption.")

## A4, measured here too

`shuffle_within_window` draws an independent `randperm(T)` per row, which
is the identity with probability `1/T!`. At **T=2 half the rows of the
"shuffled" arm are still ordered.** The T=2 cell is in the brief's grid.
This is common-mode across arms, so the TXC-vs-stacked contrast at fixed
T is unaffected — but a "the gap grows with T" reading inherits
`1 - 1/T!` from the apparatus rather than from the phenomenon.
"""
from __future__ import annotations

import math

import torch

from temp_bench.utils.shuffles import shuffle_within_window

B, D_IN, D_SAE, SEED = 512, 64, 128, 0
TS = (2, 4, 8, 16)


def _arms(tiles: torch.Tensor, W: torch.Tensor):
    """Faithful mirror of frontier.py's two SAE arms.

    Per-position encode (position-independent by construction, exactly as
    a per-token SAE applied over the window), then:
      pooled  -> z.mean(dim=1)                (frontier.py:119)
      stacked -> z.reshape(B, T * d_sae)      (frontier.py:120)
    """
    z = torch.relu(tiles @ W)
    return z.mean(dim=1), z.reshape(z.shape[0], -1)


def main() -> None:
    g = torch.Generator().manual_seed(1234)
    W = torch.randn(D_IN, D_SAE, generator=g) / math.sqrt(D_IN)

    print("A1 RECEIPT — does the pooled-zero gate discriminate a LIVE "
          "shuffle from a DEAD one?\n")
    print(f"{'T':>3}  {'shuffle':<8}{'pooled |diff|':>16}{'gate':>7}"
          f"{'stacked |diff|':>17}{'input |diff|':>15}{'assert':>9}")
    print("-" * 76)

    verdicts = []
    for T in TS:
        tiles = torch.randn(B, T, D_IN, generator=g)
        p_ord, s_ord = _arms(tiles, W)

        for label, sh in (("LIVE", shuffle_within_window(tiles, T=T, seed=SEED)),
                          ("DEAD", tiles.clone())):
            # "DEAD" is the ordinary failure: tiles_sh ends up equal to
            # tiles_ev — wrong tensor consumed, result discarded, permutation
            # applied after pooling. No exotic bug required.
            p_sh, s_sh = _arms(sh, W)
            p_d = (p_ord - p_sh).abs().max().item()
            s_d = (s_ord - s_sh).abs().max().item()
            in_d = (tiles - sh).abs().max().item()

            gate = "PASS" if p_d < 1e-6 else "VOID"
            asrt = "fires" if in_d == 0 else "silent"
            verdicts.append((T, label, gate, asrt))
            print(f"{T:>3}  {label:<8}{p_d:>16.3e}{gate:>7}"
                  f"{s_d:>17.3e}{in_d:>15.3e}{asrt:>9}")

    live_pass = all(g == "PASS" for _, l, g, _ in verdicts if l == "LIVE")
    dead_pass = all(g == "PASS" for _, l, g, _ in verdicts if l == "DEAD")
    a_live = all(a == "silent" for _, l, _, a in verdicts if l == "LIVE")
    a_dead = all(a == "fires" for _, l, _, a in verdicts if l == "DEAD")

    print(f"\n  pooled gate passes on LIVE shuffle : {live_pass}")
    print(f"  pooled gate passes on DEAD shuffle : {dead_pass}"
          f"   <- it does NOT discriminate")
    print(f"  input assert silent on LIVE        : {a_live}")
    print(f"  input assert fires  on DEAD        : {a_dead}"
          f"   <- it DOES discriminate")

    assert live_pass and dead_pass, (
        "receipt void: the pooled gate was expected to pass in BOTH cases")
    assert a_live and a_dead, (
        "receipt void: the proposed assert did not separate the two cases")

    print("\n  => A1 CONFIRMED. The gate is a check on the POOLED ARM, not on\n"
          "     the shuffle. Keep it, and add the input-side assert beside it:\n"
          "         assert (tiles_sh - tiles_ev).abs().max() > 0\n"
          "     A gate and a positive control are different objects.")

    # ---- A4: how much of the "shuffled" arm is actually shuffled? --------
    print("\nA4 RECEIPT — fraction of rows a per-row randperm actually "
          "permutes (identity w.p. 1/T!)\n")
    print(f"{'T':>3}{'1/T! (theory)':>16}{'measured moved':>17}"
          f"{'expected 1-1/T!':>18}")
    print("-" * 54)
    for T in TS:
        tiles = torch.randn(B, T, D_IN, generator=g)
        sh = shuffle_within_window(tiles, T=T, seed=SEED)
        moved = ((tiles - sh).abs().amax(dim=(1, 2)) > 0).float().mean().item()
        theory = 1.0 / math.factorial(T)
        print(f"{T:>3}{theory:>16.6g}{moved:>17.4f}{1 - theory:>18.6f}")
    print("\n  => at T=2 HALF the 'shuffled' rows are still ordered. Common-mode\n"
          "     across arms, so the fixed-T TXC-vs-stacked contrast is SAFE —\n"
          "     but any 'gap grows with T' reading inherits 1 - 1/T! from the\n"
          "     apparatus. Disclose it, or reject-and-redraw non-identity.")


if __name__ == "__main__":
    main()
