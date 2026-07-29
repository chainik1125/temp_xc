"""SHUFFLE ABLATION, SPARSITY-MATCHED — the sweep.

Executes `SHUFFLE_MATCHED_CARD.md` (frozen + amended BEFORE this file
produced a number; git history is the receipt). Read the card first —
this module implements it and deliberately re-states its binding rules
at the point of enforcement rather than trusting them to memory.

WHAT IS BEING ASKED. The two sycgen exhibits rest on different
comparisons and nothing crosses them: the shuffle figure compares
against a per-token anchor (the framing Dmitry's challenge undermined),
and the budget table has no shuffle dimension at all. So: **does the
ordered-shuffled gap survive when the comparator is a sparsity-matched
SAE rather than a per-token probe?**

⚑ THE CLAIM UNDER TEST IS **TXC gap > STACKED gap**, NOT vs pooled.
Mean-pooling per-token codes is permutation-invariant, so pooled's gap
is exactly zero on any data for any model. Reporting "TXC beat pooled's
shuffle gap" would publish a mathematical identity.

⚑ AND THE POOLED-ZERO GATE CANNOT FAIL (mac-c A1, BLOCKING; receipt in
`shuffle_gate_receipt.py`). `z.mean(dim=1)` is permutation-invariant
ARITHMETICALLY, so pooled's zero survives ANY shuffle bug — including a
shuffle that silently no-ops, which would make every arm's gap zero and
read as outcome (b), the very outcome the card pre-commits to
publishing. A gate that certifies the answer it was written to test is
not a gate. Hence `_gate_shuffle_live` below: an INPUT-side check,
arm-independent and pre-encoder, against the apparatus's EXACTLY
PREDICTED row-permutation rate `1 - 1/T!` — not a nonzero-check, which
passes if one row of thousands moved.

    .venv/bin/python -m experiments.explorations.task_hunt.sycgen.run_shuffle_matched [--smoke]

`--smoke` runs the whole path on RANDOM activations with random-init
arms. It validates shapes, the gate, the l0 units and the draw logic at
$0 and produces NO scientific rows — item 6's frontier died at row 1 on
a missing `.config` after the pod was already burning.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from temp_bench.utils.shuffles import shuffle_within_window

from experiments.explorations.task_hunt.sycgen.frontier import (
    MeasuredArm,
    WindowWrapper,
    _key_from_manifest,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "shuffle_matched.json"

TS = (2, 4, 8, 16)
SEEDS = (1, 2, 42)
EVAL_L = 32
N_WINDOWS = 1024
SHUF_EVAL_SEED = 0            # inherited from shuffle_overlay.py verbatim

# CARD §5. Item 6's grid, UNCHANGED — and the fact that it is unchanged
# is itself a measured result, not an omission.
#
# I widened this to 14 points when the hub's §2b said "add intermediate
# k where the grid straddles TXC's budget". The hub then MEASURED the
# grid (`scripts/plan_bracket_grid.py`, $0, straight off frontier.json)
# and withdrew that guidance: the nearest existing pooled point is
# already within 1.4-7.7% of TXC's budget at T=2/4/8, and at T=16
# pooled's k=1 FLOOR already costs 1.43x TXC, so no refinement is even
# possible there. Item 6's bias was SELECTION, never COVERAGE — the
# near-matched point existed and the single-sided rule passed over it.
#
# So the bracket rule (§2b) stays fully in force; only the "sweep finer"
# remedy is withdrawn. The marginal pod-minute goes to SEEDS instead,
# because outcome (d) is still unsized at n=3.
K_SWEEP = (1, 2, 4, 8, 16, 32)

DRAWS = ("plain", "redraw")


def _now() -> float:
    return time.monotonic()


# ── shuffle draws ──────────────────────────────────────────────────────

def _shuffle(tiles: torch.Tensor, T: int, seed: int, draw: str) -> torch.Tensor:
    """CARD §4c. Two draws, because they fail differently.

    `plain`  — `shuffle_within_window` verbatim. PRIMARY, because it is
        the instrument the existing `fig_sycgen_shuffle_tsweep` exhibit
        used and crossing the two exhibits is the entire point of this
        lane. It draws `randperm(T)` per row, so a row is left ORDERED
        with probability `1/T!` — 50% of rows at T=2, 4.2% at T=4.
    `redraw` — rejects identity draws, so the shuffled fraction is 1.000
        at every T and the `1 - 1/T!` factor is gone by construction.

    Common-mode across arms ⇒ the fixed-T TXC-vs-stacked contrast (the
    primary claim) is safe under either. But any "the gap GROWS WITH T"
    reading inherits `1 - 1/T!` from the apparatus — the same species as
    the divide-by-T per-token artifact retracted earlier tonight — so
    the card binds cross-T statements to the `redraw` column only.
    """
    if draw == "plain":
        return shuffle_within_window(tiles, T=T, seed=seed)
    if draw != "redraw":
        raise ValueError(f"unknown draw {draw!r}")
    if T < 2:
        return tiles.clone()
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    ident = torch.arange(T)
    B = tiles.shape[0]
    perms = torch.empty((B, T), dtype=torch.long)
    for b in range(B):
        while True:
            p = torch.randperm(T, generator=g)
            if not torch.equal(p, ident):
                break
        perms[b] = p
    perms = perms.to(tiles.device)
    bidx = torch.arange(B, device=tiles.device).unsqueeze(1).expand(-1, T)
    return tiles[bidx, perms].contiguous()


def _shuffled_row_fraction(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of rows that actually differ. The gate's measured value."""
    d = (a - b).flatten(1).abs().amax(dim=1)
    return float((d > 0).float().mean())


def _identity_band(n: int, p: float) -> tuple[int, int]:
    """Two-sided acceptance band for the IDENTITY-row count ~ Binomial(n, p).

    mac-c's review of my first cut, accepted: gating the *fraction*
    against `1 - 1/T!` with a normal `k*SE` band converts a
    deterministic check into a statistical one, and **the regime changes
    qualitatively across the grid** —

        T=2  n=16384  E[identity] = 8192      (p = 1/2)
        T=4  n= 8192  E =  341.3
        T=8  n= 4096  E =    0.1016
        T=16 n= 2048  E =    9.8e-11

    **T=8 is the trap.** At E=0.1 an equality-ish gate spuriously VOIDS
    ~9.7% of HEALTHY runs and reports "instrument broken" — the exact
    mirror of A1. A1 was a gate that cannot fire when it should; that
    one fired when it should not. Both are the same defect: a check
    whose behaviour was never computed against the grid it runs on.

    ⚑ MY OWN FIRST FIX WAS ALSO WRONG, at T=16. I used "the WIDER of an
    exact binomial tail and a 4-sigma normal band", and checked that
    against mac-c's published numbers — but I checked the *binomial
    column* of my working table, not the output of the composed rule.
    At T=16 (lambda ~ 1e-10) the 4-sigma arm contributes `ceil(4*sd) = 1`,
    so the max() rule yields **0..1** where the binding band is **0..0**:
    it would accept a run with a genuine unmoved row at the one T where
    that is impossible. Verifying a component is not verifying the rule
    built from it — the same defect this gate exists to catch, committed
    inside the fix for it.

    The hub reached for a sigma band too, and it is wrong wherever the
    regime is not normal: **a sigma band is meaningless at lambda ~ 0.1
    or 1e-10; the right construction is a tail probability.** No single
    alpha reproduces all four cells, so the BINDING bands are used as
    ruled, and they are valid only for the `n` they were derived at —
    hence the recomputation guard below (mac-c's rider).

    BINDING (hub `33a5c72d8`), with "floor: round toward accepting":

        T=2  n=16384  7936..8448      T=8  n=4096  0..3
        T=4  n= 8192   268..414       T=16 n=2048  0..0
    """
    lo_b, hi_b = _exact_band(n, p)
    return lo_b, hi_b


# Bands are functions of n. Recompute if n_windows or L move — do NOT
# carry these literals to a different geometry (mac-c's rider).
_BINDING_N = {2: 16384, 4: 8192, 8: 4096, 16: 2048}
_BINDING_BAND = {2: (7936, 8448), 4: (268, 414), 8: (0, 3), 16: (0, 0)}


def _exact_band(n: int, p: float) -> tuple[int, int]:
    """Exact binomial tail band; falls back to the ruled literals for T in grid."""
    if p <= 0.0:
        return 0, 0                      # redraw: identity count is 0 exactly
    T = None
    for t, pn in _BINDING_N.items():
        if pn == n and abs(p - 1.0 / math.factorial(t)) < 1e-18:
            T = t
            break
    if T is not None:
        return _BINDING_BAND[T]
    # Off-grid geometry: derive from the tail, never from a sigma band.
    from scipy.stats import binom
    return int(binom.ppf(1e-4, n, p)), int(binom.isf(1e-4, n, p))


def _gate_shuffle_live(tiles: torch.Tensor, tiles_sh: torch.Tensor,
                       T: int, draw: str) -> dict:
    """CARD §4b — the A1 fix. INPUT-side, arm-independent, pre-encoder.

    Checked HERE and not on pooled's output because pooled's zero is
    permutation-invariant arithmetic: it returns PASS on a dead shuffle
    at every T (mac-c's receipt measures exactly that). This gate reads
    the tiles themselves, before any arm touches them, so no arm's
    algebra can launder a failure into a pass.

    Gates the identity-row COUNT against `_identity_band`, not the
    fraction against zero: "something changed" passes if one row of
    thousands moved, whereas a count band also catches partial
    application and a wrong-axis permutation.

    ⚑ For `redraw` the expected identity count is 0 BY CONSTRUCTION, so
    a pass there is NOT evidence the shuffle works — it is evidence the
    rejection loop works. The `plain` column is what certifies the
    apparatus; the two are checked by different arguments on purpose.

    ⚑ At T=8/16 the band is near-deterministic (0..3, 0..0), which
    assumes no EXACT TIES — two identical activation vectors inside one
    tile would read as an unmoved row. Effectively impossible on float32
    activations, but it is the first thing to check if T=16 ever voids.
    """
    n = int(tiles.shape[0])
    frac = _shuffled_row_fraction(tiles_sh, tiles)
    n_ident = int(round((1.0 - frac) * n))
    p = 0.0 if draw == "redraw" else 1.0 / math.factorial(T)
    lo, hi = _identity_band(n, p)
    if not (lo <= n_ident <= hi):
        raise AssertionError(
            f"SHUFFLE GATE FAILED  T={T} draw={draw}: {n_ident} identity "
            f"rows of n={n}, acceptance band [{lo}, {hi}] for "
            f"Binomial(n, 1/T!={p:.3e}). The shuffle is not doing what "
            "the card says it does — and a no-op here makes EVERY arm's "
            "gap zero, which reads as outcome (b), the outcome the card "
            "pre-commits to publishing. Refusing to produce rows.")
    return {"rows_permuted": frac, "identity_rows": n_ident,
            "identity_expected": n * p, "band": [lo, hi],
            "n_rows": n, "by_construction": draw == "redraw"}


# ── arms ───────────────────────────────────────────────────────────────

def _build(arch_name: str, T: int, seed: int, ds_spec, *, trained: bool):
    """Trained arm loads the mirrored checkpoint; twin is random init.

    CARD §3 [AMD A2]: the untrained twin is not decoration, it is a GATE
    on outcome (a) — it is what killed sycgen's original shuffle claim
    (twins showed LARGER gaps than trained models). Built from the same
    spec with the same seed so the only difference is the weights.
    """
    from temp_bench.core.config import load_arch
    from temp_bench.core.runner import _load_checkpoint, import_by_path
    from temp_bench.core.trainer import _infer_d_in

    spec = load_arch(arch_name)
    spec = spec.model_copy(update={
        "hparams": {**spec.hparams, "d_sae": 2048, "T": T, "k_pos": 8}})
    if trained:
        tk = _key_from_manifest(arch_name, T, seed)
        if tk is None:
            raise FileNotFoundError(
                f"no manifest train_key for {arch_name} T={T} seed={seed}")
        return _load_checkpoint(spec, tk, ds_spec), tk
    torch.manual_seed(int(seed))
    cls = import_by_path(spec.class_path)
    model = cls(d_in=_infer_d_in(ds_spec), **spec.hparams)
    model.eval()
    return model, None


def _codes(model, tiles: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        z = model.encode(tiles)
    return z.reshape(tiles.shape[0], -1).detach().float().cpu().numpy()


def _corr(pred: np.ndarray, tgt: np.ndarray) -> float:
    if np.std(pred) <= 1e-12 or np.std(tgt) <= 1e-12:
        return 0.0
    return float(np.corrcoef(pred, tgt)[0, 1])


def _tiles(win_x, win_l, T, device):
    W, L_, d_in = win_x.shape
    n_tiles = L_ // T
    tiles = win_x.to(device, dtype=torch.float32).reshape(W * n_tiles, T, d_in)
    tgt = win_l.reshape(W, n_tiles, T)[:, :, T - 1].reshape(-1)
    return tiles, tgt.detach().float().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="random activations + random arms; validates the "
                         "path at $0 and produces NO scientific rows")
    # HAN DIRECTIVE (brief §8): parallelize cells across GPUs, max
    # throughput. Adding GPUs to an unsharded script buys EXACTLY ZERO —
    # this was single-process/single-device with a serial `for T`, so a
    # 16-GPU pod would have run 15 idle GPUs. The unit is (T, seed, draw)
    # = 24 independent cells; assignment is `sorted(cells)[i::n]`, so it
    # is deterministic and needs no coordination between shards.
    ap.add_argument("--shard", default="0/1", metavar="i/n",
                    help="run cell subset i of n (deterministic stride)")
    ap.add_argument("--max-cells", type=int, default=0,
                    help="stop after N cells — for the seconds/cell "
                         "measurement the fleet size must be derived from")
    args = ap.parse_args()

    shard_i, shard_n = (int(v) for v in args.shard.split("/"))
    assert 0 <= shard_i < shard_n, f"bad --shard {args.shard}"

    from temp_bench.evals.synthetic_recovery import _sample_windows

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    if args.smoke:
        ds_spec = None
        g = torch.Generator().manual_seed(0)
        x = torch.randn(256, 128, 64, generator=g)
        lam = torch.randn(256, 128, generator=g)
        ts, seeds, ks = (2, 4), (1,), (1, 4)
    else:
        from temp_bench.core.config import load_datasource
        from temp_bench.data.synthetic import materialise
        import experiments.explorations.task_hunt.sycgen.run_retrain as RR
        ds_spec = load_datasource(RR.DS)
        data = materialise(ds_spec, seed=0)
        lam = data.extra["lambda_labels"]
        if not torch.is_tensor(lam):
            lam = torch.as_tensor(lam)
        x, lam = data.x, lam.float()
        ts, seeds, ks = TS, SEEDS, K_SWEEP

    cells = sorted((T, seed, draw)
                   for T in ts for seed in seeds for draw in DRAWS)
    mine = cells[shard_i::shard_n]
    if args.max_cells:
        mine = mine[:args.max_cells]
    out = (OUT if shard_n == 1
           else OUT.with_suffix(f".shard{shard_i}.json"))

    print(f"[shuffle] device={device} x={tuple(x.shape)} "
          f"smoke={args.smoke} shard={shard_i}/{shard_n} "
          f"cells={len(mine)}/{len(cells)} -> {out.name}", flush=True)

    rows, gates, timings = [], [], []
    t_wall0 = _now()
    for T, seed, draw in mine:
            t_cell0 = _now()
            lam3 = lam.reshape(lam.shape[0], lam.shape[1], 1)
            n = x.shape[0]; split = n // 2
            wx_tr, _ = _sample_windows(x[:split], L=EVAL_L,
                                       n_windows=N_WINDOWS, seed=seed)
            wl_tr, _ = _sample_windows(lam3[:split], L=EVAL_L,
                                       n_windows=N_WINDOWS, seed=seed)
            wx_ev, _ = _sample_windows(x[split:], L=EVAL_L,
                                       n_windows=N_WINDOWS, seed=seed + 1)
            wl_ev, _ = _sample_windows(lam3[split:], L=EVAL_L,
                                       n_windows=N_WINDOWS, seed=seed + 1)
            tiles_tr, t_tr = _tiles(wx_tr, wl_tr, T, device)
            tiles_ev, t_ev = _tiles(wx_ev, wl_ev, T, device)

            fin_tr = np.isfinite(t_tr); fin_ev = np.isfinite(t_ev)

            tiles_sh = _shuffle(tiles_ev, T, SHUF_EVAL_SEED, draw)
            sh_tr = _shuffle(tiles_tr, T, SHUF_EVAL_SEED, draw)
            # GATE FIRST — before any arm runs, so no arm's algebra
            # can launder a dead shuffle into a pass.
            g_ev = _gate_shuffle_live(tiles_ev, tiles_sh, T, draw)
            g_ev.update({"T": T, "seed": seed, "draw": draw})
            gates.append(g_ev)
            print(f"  GATE T{T} s{seed} {draw:7s} identity_rows="
                  f"{g_ev['identity_rows']}/{g_ev['n_rows']} "
                  f"(exp {g_ev['identity_expected']:.3g}, band "
                  f"{g_ev['band'][0]}..{g_ev['band'][1]})"
                  f"{'  [by construction]' if g_ev['by_construction'] else ''}",
                  flush=True)

            for trained in (True, False):
                tag = "trained" if trained else "untrained"
                # --- TXC ---
                try:
                    if args.smoke:
                        raw, tk = _mk_smoke_txc(T, seed, trained), None
                    else:
                        raw, tk = _build("txc_batchtopk_post_btkonly",
                                         T, seed, ds_spec, trained=trained)
                    raw = raw.to(device)
                    m = _score(lambda: MeasuredArm(raw),
                               tiles_tr, t_tr, tiles_ev, t_ev,
                               tiles_sh, sh_tr, fin_tr, fin_ev)
                    rows.append({"arm": "txc", "weights": tag, "draw": draw,
                                 "T": T, "seed": seed, "k_tok": None,
                                 "train_key": tk,
                                 "l0_unit": "nonzeros_in_tile_code", **m})
                    print(f"    txc      {tag:9s} {draw:7s} T{T} s{seed} "
                          f"gap={m['gap_fixedprobe']:+.4f} "
                          f"l0={m['realized_l0_per_window_ordered']:.2f}",
                          flush=True)
                except Exception as e:
                    print(f"    txc      {tag:9s} {draw:7s} T{T} s{seed} "
                          f"SKIP {type(e).__name__}: {str(e)[:80]}", flush=True)

                # --- pooled / stacked over the SAE ---
                try:
                    if args.smoke:
                        sae, sae_tk = _mk_smoke_sae(trained), None
                    else:
                        sae, sae_tk = _build("batchtopk_sae_btkonly", 1,
                                             seed, ds_spec, trained=trained)
                    sae = sae.to(device)
                except Exception as e:
                    print(f"    sae      {tag:9s} LOAD FAIL "
                          f"{type(e).__name__}: {str(e)[:70]}", flush=True)
                    continue
                for mode in ("pooled", "stacked"):
                    for k in ks:
                        m = _score(
                            lambda: WindowWrapper(sae, T, mode, k),
                            tiles_tr, t_tr, tiles_ev, t_ev, tiles_sh,
                            sh_tr, fin_tr, fin_ev)
                        rows.append({
                            "arm": mode, "weights": tag, "draw": draw,
                            "T": T, "seed": seed, "k_tok": k,
                            "sae_train_key": sae_tk,
                            "l0_unit": ("union_over_positions"
                                        if mode == "pooled"
                                        else "sum_over_positions"), **m})
                    print(f"    {mode:8s} {tag:9s} {draw:7s} T{T} s{seed} "
                          f"k-sweep done ({len(ks)} pts)", flush=True)

            OUT.parent.mkdir(parents=True, exist_ok=True)
            timings.append({"T": T, "seed": seed, "draw": draw,
                            "seconds": _now() - t_cell0})
            out.write_text(json.dumps({"rows": rows, "gates": gates,
                                       "timings": timings,
                                       "shard": [shard_i, shard_n],
                                       "smoke": bool(args.smoke)}, indent=1))
            print(f"  [cell] T{T} s{seed} {draw} done in "
                  f"{timings[-1]['seconds']:.1f}s "
                  f"({len(timings)}/{len(mine)})", flush=True)
    print(f"[shuffle] {len(rows)} rows, {len(gates)} gate receipts -> {OUT}")
    if args.smoke:
        _smoke_selfcheck(rows, gates)
        print("[shuffle] SMOKE — no scientific rows produced.")
    return 0


def _smoke_selfcheck(rows: list, gates: list) -> None:
    """Assert the invariants instead of printing them for a human to skim.

    A smoke run whose only output is 'it finished' proves the process
    exited zero, which is not the claim being made. Each check below is
    paired with the failure it detects; if any stops holding, this
    raises rather than letting the real run inherit a broken path.
    """
    pooled = [r for r in rows if r["arm"] == "pooled"]
    stacked = [r for r in rows if r["arm"] == "stacked"]
    txc = [r for r in rows if r["arm"] == "txc"]
    assert pooled and stacked and txc, "an arm produced no rows at all"

    # 1. The identity §1 rests on. If this ever fails, the pooled arm has
    #    become position-sensitive and the whole framing is wrong.
    worst = max(abs(r["gap_fixedprobe"]) for r in pooled)
    assert worst < 1e-8, f"pooled gap is not zero (max |gap| {worst:.3e})"

    # 2. The shuffle must MOVE something. Detects a dead shuffle that
    #    pooled's invariance would happily certify as a pass.
    assert max(abs(r["gap_fixedprobe"]) for r in stacked) > 1e-6, \
        "stacked gap is zero everywhere — the shuffle is not live"

    # 3. §6's prediction, measured: both SAE l0 units are symmetric over
    #    positions, so budget must be permutation-invariant.
    for r in pooled + stacked:
        a, b = (r["realized_l0_per_window_ordered"],
                r["realized_l0_per_window_shuffled"])
        assert abs(a - b) < 1e-6, \
            f"{r['arm']} l0 moved under shuffle ({a:.6f} -> {b:.6f})"

    # 4. A2's twin gate needs the twin to actually be a different model.
    #    This is the check the first smoke run could not make.
    def key(r):
        return (r["arm"], r["T"], r["seed"], r["draw"], r["k_tok"])
    tr = {key(r): r["gap_fixedprobe"] for r in rows if r["weights"] == "trained"}
    un = {key(r): r["gap_fixedprobe"] for r in rows if r["weights"] == "untrained"}
    shared = set(tr) & set(un)
    assert shared, "no trained/untrained pairs to compare"
    # POOLED is excluded deliberately, not for convenience: its gap is
    # identically 0 for trained AND untrained (§1), so a pooled pair
    # CANNOT differ and including it would let a broken `trained` flag
    # hide inside an expected-zero. Every txc/stacked pair must differ —
    # `differing > 0` would have passed on 1 of 12.
    live = [k for k in shared if k[0] != "pooled"]
    same = [k for k in live if abs(tr[k] - un[k]) <= 1e-9]
    assert not same, (
        f"{len(same)}/{len(live)} txc+stacked rows equal their untrained "
        f"twin exactly — the `trained` flag is being ignored, and A2 makes "
        f"that twin a GATE on outcome (a). First: {same[:3]}")
    dead = [k for k in shared if k[0] == "pooled" and abs(tr[k] - un[k]) > 1e-9]
    assert not dead, f"pooled pair DIFFERS, breaking §1's identity: {dead[:3]}"
    differing = len(live)

    # 5. The gate's own predicted rate, re-checked off the persisted
    #    receipts rather than trusting that it ran.
    for g in gates:
        assert g["band"][0] <= g["identity_rows"] <= g["band"][1], \
            f"gate receipt outside its own band: {g}"

    print(f"[smoke] SELF-CHECK PASS — pooled|gap|max {worst:.2e}, "
          f"{differing}/{len(shared)} twin pairs differ, "
          f"{len(gates)} gate receipts in tolerance")


def _score(make_arm, tiles_tr, t_tr, tiles_ev, t_ev, tiles_sh, tiles_sh_tr,
           fin_tr, fin_ev) -> dict:
    """Ordered + shuffled recovery under BOTH probe conventions.

    PRIMARY (fixed probe): fit on ORDERED train, score that same probe —
    never refit — on shuffled eval, both against the ORIGINAL targets.
    Inherited verbatim from `shuffle_overlay.py` so this table and the
    existing figure are commensurable, which is why the lane exists.

    SECONDARY (refit probe): fit on SHUFFLED train, score on shuffled
    eval. Pre-registered as secondary BEFORE any number existed, and
    marked secondary here, specifically so it cannot be promoted if the
    primary disappoints.

    Why both: for STACKED the fixed probe breaks partly from SLOT
    SCRAMBLING — a token's features move to a different `p*d_sae+f`
    slot. That is arguably the order-sensitivity under test (stacked
    genuinely encodes position), which is why it stays primary, but it
    conflates "the code moved" with "the information is gone" and it
    inflates the baseline AGAINST TXC's claim. The refit column
    separates them and is the disambiguator for outcome (b).

    A fresh arm per phase: `WindowWrapper` ACCUMULATES `_l0` across
    encode calls, so a shared instance would silently report an l0
    averaged over ordered and shuffled tiles — the budget axis is the
    one thing this lane cannot afford to blur.

    Non-finite targets are dropped exactly as `lambda_recovery` drops
    them (real-activation label grids carry NaN where the frozen label
    is undefined), so this sweep and the canonical path agree row-for-row.
    """
    from sklearn.linear_model import LinearRegression

    a = make_arm(); z_tr = _codes(a, tiles_tr)
    a = make_arm(); z_ev = _codes(a, tiles_ev); l0_ord = a.realized_l0_per_window
    a = make_arm(); z_sh = _codes(a, tiles_sh); l0_sh = a.realized_l0_per_window
    a = make_arm(); z_sh_tr = _codes(a, tiles_sh_tr)

    ztr, ttr = (z_tr[fin_tr], t_tr[fin_tr]) if not fin_tr.all() else (z_tr, t_tr)
    zsh_tr = z_sh_tr[fin_tr] if not fin_tr.all() else z_sh_tr
    zev, zsh, tev = ((z_ev[fin_ev], z_sh[fin_ev], t_ev[fin_ev])
                     if not fin_ev.all() else (z_ev, z_sh, t_ev))

    reg = LinearRegression().fit(ztr, ttr)
    r_ord = _corr(reg.predict(zev), tev)
    r_sh = _corr(reg.predict(zsh), tev)
    reg2 = LinearRegression().fit(zsh_tr, ttr)
    r_sh_refit = _corr(reg2.predict(zsh), tev)
    return {
        "recovery_ordered": r_ord,
        "recovery_shuffled_fixedprobe": r_sh,
        "gap_fixedprobe": r_ord - r_sh,
        "recovery_shuffled_refitprobe": r_sh_refit,
        "gap_refitprobe": r_ord - r_sh_refit,
        "realized_l0_per_window_ordered": l0_ord,
        "realized_l0_per_window_shuffled": l0_sh,
    }


# ── smoke-only stand-ins (never used on the real path) ────────────────

class _SmokeSAE(torch.nn.Module):
    """Minimal per-token SAE with the `encode((B,1,d))->(B,1,d_sae)` contract."""

    def __init__(self, d_in=64, d_sae=128, k=8):
        super().__init__()
        from types import SimpleNamespace
        self.W = torch.nn.Linear(d_in, d_sae)
        self.k = k
        self.config = SimpleNamespace(T=1, d_in=d_in, d_sae=d_sae)

    def encode(self, t):
        z = torch.relu(self.W(t))
        kth = z.topk(min(self.k, z.shape[-1]), dim=-1).values[..., -1:]
        return z * (z >= kth)


class _SmokeTXC(torch.nn.Module):
    """Order-SENSITIVE window arm, so the smoke path exercises a real gap."""

    def __init__(self, T, d_in=64, d_sae=128, k=8):
        super().__init__()
        from types import SimpleNamespace
        self.T, self.k = T, k
        self.W = torch.nn.Linear(T * d_in, d_sae)
        self.config = SimpleNamespace(T=T, d_in=d_in, d_sae=d_sae)

    def encode(self, tiles):
        B = tiles.shape[0]
        z = torch.relu(self.W(tiles.reshape(B, -1)))
        kth = z.topk(min(self.k, z.shape[-1]), dim=-1).values[..., -1:]
        return z * (z >= kth)


# ⚑ The `trained` flag MUST change the weights even in smoke. Without
# it both branches built the identical model, so every trained/untrained
# row came out byte-identical — and a smoke test that cannot distinguish
# "the twin path works" from "the twin flag is ignored" is precisely the
# guard-that-reports-success shape this lane exists to avoid. A2 makes
# the twin a GATE on outcome (a), so that path has to be exercised.

def _mk_smoke_sae(trained: bool):
    torch.manual_seed(0 if trained else 12345)
    return _SmokeSAE()


def _mk_smoke_txc(T, seed, trained: bool):
    torch.manual_seed(seed if trained else seed + 9871)
    return _SmokeTXC(T)


if __name__ == "__main__":
    sys.exit(main())
