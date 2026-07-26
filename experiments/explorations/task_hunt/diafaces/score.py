"""Read-only scorer for the frozen diafaces screen (CARD.md § 6–7)
plus the panel-gate clause (ii) check as PINNED by mac-local's
pre-results freeze review (LOG 2026-07-26): wd order arm
sc = win_linear − win_shuf_linear ≥ +0.03 at T ∈ {16,32} on ≥ 2 of 3
models INCLUDING at least one of {gpt2, llama31_8b}.

Every window number beside its width null AND the visible-evidence
floor. No cell computed here.

Run: .venv/bin/python -m experiments.explorations.task_hunt.diafaces.score [model ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FACES = ("tt", "dq")
AX_TS = [4, 8, 16, 32, 64]
ORD_TS = [16, 32]
CHANCE = 1 / 3
MODELS = ["gpt2", "llama31_8b", "gemma2_2b"]
GATE_SC = 0.03
GATE_CORE = {"gpt2", "llama31_8b"}


def acc(cells, key):
    c = cells.get(key)
    return None if c is None else c.get("acc_test")


def auc(cells, key):
    c = cells.get(key)
    return None if c is None else c.get("auc")


def fmt(x):
    return "  --  " if x is None else f"{x:+.3f}"


def keep_kill(cells, face, wd_ok):
    """CARD § 7 per model: (KEEP?, KILL?, best evidence dict)."""
    tokL = acc(cells, f"{face}/tok_linear")
    best = None
    for T in AX_TS:
        for arm, null_arm in ((f"{face}/T{T}/actxmean_linear",
                               f"{face}/T{T}/actxmean_foreign_linear"),
                              (f"{face}/T{T}/actxmean_mlp",
                               f"{face}/T{T}/actxmean_foreign_mlp"),
                              (f"{face}/T{T}/win_linear",
                               f"{face}/T{T}/win_foreign_linear"),
                              (f"{face}/T{T}/win_mlp",
                               f"{face}/T{T}/win_foreign_mlp")):
            a, n = acc(cells, arm), acc(cells, null_arm)
            vis = acc(cells, f"{face}/T{T}/visible_evidence_floor")
            if a is None or tokL is None:
                continue
            ok = (a - tokL >= 0.05
                  and n is not None and a - n >= 0.02
                  and vis is not None and a > vis)
            if ok and (best is None or a - tokL > best["gain"]):
                best = {"arm": arm, "T": T, "gain": a - tokL,
                        "width": a - n, "over_vis": a - vis}
    # wd same-direction window gain (binary AUC): any wd window arm
    # above wd tok
    wd_gain = None
    wtl = auc(cells, f"{face}_wd/tok_linear")
    if wtl is not None:
        for T in (16, 32, 64):
            for arm in (f"{face}_wd/T{T}/actxmean_linear",
                        f"{face}_wd/T{T}/actxmean_mlp",
                        f"{face}_wd/T{T}/win_linear"):
                a = auc(cells, arm)
                if a is not None and (wd_gain is None or a - wtl > wd_gain):
                    wd_gain = a - wtl
    keep = bool(best) and wd_ok and wd_gain is not None and wd_gain > 0
    # KILL clauses (1)-(4)
    every_within = all(
        (acc(cells, k) is None or tokL is None
         or acc(cells, k) - tokL < 0.02)
        for k in cells if k.startswith(f"{face}/T") and
        ("actxmean" in k or "win_" in k) and "foreign" not in k
        and "shuf" not in k and "null" not in k and "floor" not in k)
    kill = every_within or (best is None) or (
        wd_gain is not None and wd_gain <= 0)
    return keep, kill, best, wd_gain


def score_model(key: str):
    z = json.loads((RES / f"screen_{key}.json").read_text())
    cells = z["cells"]
    rows = z["meta"]["rows"]
    out = {}
    print(f"\n{'=' * 72}\n{key}  (screen_hs {z['meta']['screen_hs']})\n"
          f"{'=' * 72}")
    for face in FACES:
        tokL = acc(cells, f"{face}/tok_linear")
        tokM = acc(cells, f"{face}/tok_mlp")
        floor = acc(cells, f"{face}/position_floor")
        if tokL is None:
            print(f"{face}: NO CELLS")
            continue
        print(f"{face}: tok_lin {tokL:.3f}  tok_mlp {fmt(tokM)}  "
              f"pos_floor {fmt(floor)}  (chance {CHANCE:.3f})  "
              f"Q1 tok-floor = {fmt(None if floor is None else tokL - floor)}"
              f" (predicted < +0.10)")
        print("  T    vis_floor  ax_lin  g_ax_lin  width_lin  ax-vis | "
              "g_ax_mlp width_mlp")
        for T in AX_TS:
            vis = acc(cells, f"{face}/T{T}/visible_evidence_floor")
            ax = acc(cells, f"{face}/T{T}/actxmean_linear")
            axf = acc(cells, f"{face}/T{T}/actxmean_foreign_linear")
            axm = acc(cells, f"{face}/T{T}/actxmean_mlp")
            axfm = acc(cells, f"{face}/T{T}/actxmean_foreign_mlp")
            g = None if ax is None else ax - tokL
            w = None if ax is None or axf is None else ax - axf
            av = None if ax is None or vis is None else ax - vis
            gm = None if axm is None or tokM is None else axm - tokM
            wm = None if axm is None or axfm is None else axm - axfm
            print(f"  {T:>3}  {fmt(vis)}     {fmt(ax)}  {fmt(g)}   "
                  f"{fmt(w)}    {fmt(av)} | {fmt(gm)}  {fmt(wm)}")
        print("  T    sc(=win-shuf)  wc(=win-foreign)  win-vis | mlp: sc  wc")
        for T in ORD_TS:
            wl = acc(cells, f"{face}/T{T}/win_linear")
            sh = acc(cells, f"{face}/T{T}/win_shuf_linear")
            fo = acc(cells, f"{face}/T{T}/win_foreign_linear")
            vis = acc(cells, f"{face}/T{T}/visible_evidence_floor")
            sc = None if wl is None or sh is None else wl - sh
            wc = None if wl is None or fo is None else wl - fo
            wv = None if wl is None or vis is None else wl - vis
            wlm = acc(cells, f"{face}/T{T}/win_mlp")
            shm = acc(cells, f"{face}/T{T}/win_shuf_mlp")
            fom = acc(cells, f"{face}/T{T}/win_foreign_mlp")
            scm = None if wlm is None or shm is None else wlm - shm
            wcm = None if wlm is None or fom is None else wlm - fom
            print(f"  {T:>3}  {fmt(sc)}        {fmt(wc)}          "
                  f"{fmt(wv)} | {fmt(scm)} {fmt(wcm)}")
        nw = acc(cells, f"{face}/T16/null_win_linear")
        nt = acc(cells, f"{face}/null_tok_linear")
        if nw is not None:
            print(f"  nulls T16: win {nw:.3f} tok {nt:.3f}")

        wd = f"{face}_wd"
        wtl = auc(cells, f"{wd}/tok_linear")
        wd_sc = {}
        if wtl is not None:
            print(f"  wd (binary AUC, {rows.get(f'{wd}/test', {})}): "
                  f"tok_lin {wtl:.3f}  tok_mlp "
                  f"{fmt(auc(cells, f'{wd}/tok_mlp'))}")
            for T in (16, 32, 64):
                a = auc(cells, f"{wd}/T{T}/actxmean_linear")
                af = auc(cells, f"{wd}/T{T}/actxmean_foreign_linear")
                wv = auc(cells, f"{wd}/T{T}/win_linear")
                sv = auc(cells, f"{wd}/T{T}/win_shuf_linear")
                fv = auc(cells, f"{wd}/T{T}/win_foreign_linear")
                g = None if a is None else a - wtl
                w = None if a is None or af is None else a - af
                sc = None if wv is None or sv is None else wv - sv
                wc = None if wv is None or fv is None else wv - fv
                if T in ORD_TS and sc is not None:
                    wd_sc[T] = sc
                print(f"    T{T}: ax-tok {fmt(g)}  ax-foreign {fmt(w)}  "
                      f"wd_sc {fmt(sc)}  wd_wc {fmt(wc)}")
        else:
            print(f"  {wd}: NO CELLS (blocks any KEEP, card § 7)")
        keep, kill, best, wd_gain = keep_kill(
            cells, face, wd_ok=wtl is not None)
        verdict = "KEEP" if keep else ("KILL" if kill else "WEAK")
        print(f"  --> {face} per-model: {verdict}"
              + (f"  best {best['arm']} gain {best['gain']:+.3f} "
                 f"width {best['width']:+.3f} over-vis "
                 f"{best['over_vis']:+.3f}" if best else "")
              + f"  wd_gain {fmt(wd_gain)}")
        out[face] = {"verdict": verdict, "best": best, "wd_gain": wd_gain,
                     "wd_sc": wd_sc}
    return out


def main():
    keys = sys.argv[1:] or MODELS
    per_model = {}
    for k in keys:
        if (RES / f"screen_{k}.json").exists():
            per_model[k] = score_model(k)
        else:
            print(f"[{k}] no results file yet")
    if len(per_model) < 2:
        return
    print(f"\n{'=' * 72}\nBUNDLE (majority of {len(per_model)} models) "
          f"+ PANEL GATE\n{'=' * 72}")
    for face in FACES:
        vs = {k: v[face]["verdict"] for k, v in per_model.items()
              if face in v}
        n_keep = sum(v == "KEEP" for v in vs.values())
        n_kill = sum(v == "KILL" for v in vs.values())
        maj = ("KEEP" if n_keep >= 2 else
               "KILL" if n_kill >= 2 else "WEAK")
        gate_models = [k for k, v in per_model.items()
                       if face in v and any(
                           s >= GATE_SC for s in v[face]["wd_sc"].values())]
        gate_ii = (len(gate_models) >= 2
                   and bool(set(gate_models) & GATE_CORE))
        print(f"{face}: per-model {vs} -> {maj} (clause i "
              f"{'MET' if maj == 'KEEP' else 'NOT met'}); "
              f"wd sc >= +{GATE_SC} on {gate_models} -> clause ii "
              f"{'MET' if gate_ii else 'NOT met'}")


if __name__ == "__main__":
    main()
